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

"""The numerics declaration tool (docs/NUMERICS.md, DECISIONS.md D67).

    numerics.py capture --url URL --out numerics/references/NAME.json [--suite FILE]
                        [--forced-from REFERENCE] [--max-tokens N] [--top-logprobs K]
                        [--evidence logprobs|text] [--binary PATH] [--repo PATH]
    numerics.py check --url URL [--reference FILE]
    numerics.py compare --anchor REFERENCE --candidate CAPTURE [--floor COMPARE.json]
                        [--json OUT] [--markdown OUT]
    numerics.py coverage --dump FILE [--layout FILE] [--min-inputs N]
    numerics.py coverage-gaps --dump FILE [--layout FILE] [--json OUT]
    numerics.py coverage-sweep --url URL --dump FILE --out SUITE.json [--corpus FILE]
                        [--input-suite FILE] [--seen FILE] [--generate N] [--seed N]
                        [--order shuffled|corpus] [--batch N] [--max-candidates N]
                        [--max-tokens N] [--timeout S] [--json OUT]
    numerics.py monitors --reference REF.json --candidate CAND.json
                        --layer-bar X --expert-bar Y [--json OUT] [--markdown OUT]
    numerics.py fingerprint FILE...
    numerics.py ci [--base REV]
    numerics.py self-test

capture  runs a prompt suite against a running server and writes a reference: every prompt's ids,
         and every generated token's id, logprob and top-k alternatives (evidence "logprobs"), or
         the greedy text where the path serves no logprobs (evidence "text"). --forced-from feeds
         another reference's tokens through the server instead (teacher forcing), which is what
         compare reads. Its server card names the build it came from: --binary (or SPARKFLASH_BIN)
         is hashed into binary_sha256, and the served card's build.id supplies the commit; --repo
         (default: this checkout) is the fallback, and capture warns when neither names one.
         Either may be unknown; the card then records null and a one-line reason, never a guess.
check    recaptures the server's registered reference and fails on any bit that differs. It also
         fails when the server does not report the registry entry this checkout has.
compare  scores a teacher-forced capture against the anchor it was forced from: per-token NLL
         delta, top-1 agreement and a KL lower bound. --floor puts another comparison (the TP=1 to
         TP=4 divergence) beside it.
coverage judges a coverage dump against the model's expert layout and the registry's declared
         bar: every decoder layer present, one hit count per routed expert, at least the declared
         number of inputs, and at least prerequisites.coverage_slots of the routed slots reached.
         It prints the achieved fraction and the per-layer missing experts and exits non-zero
         below the bar. coverage-gaps reports what a partial dump has not reached yet.
coverage-sweep sends a deterministic corpus to a server that is writing the dump, reads the dump
         before and after each prompt, keeps a prompt only when its forward reached a slot no earlier
         kept prompt reached, and writes the kept prompts as the next suite. The dump is the
         evidence; the selection is measured, never the generator's opinion.
monitors compares a candidate monitor dump with the reference forward, per layer and per shared
         expert, reports the experts only one dump routed to, and fails every relative-L2 bar it
         exceeds.
ci       the declaration check: a changed reference without a version bump fails, a bump without a
         change fails, an unapproved bump whose own model has no [[coverage]] row, no [[monitors]]
         row, or a row that is malformed or misses its bar fails with the model named (an approved
         bump needs neither), a true coverage or monitors aggregate flag that claims more than the
         rows hold fails, an anchor never changes, every fingerprint must match its file, an arm
         profile's arithmetic setting changes only with every reference of the model it names
         recaptured, and a field joins the neutral list only with a report.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
NUMERICS = "numerics"
REGISTRY = f"{NUMERICS}/registry.toml"
REFERENCES = f"{NUMERICS}/references/"
CAPTURES = f"{NUMERICS}/captures/"
DEFAULT_SUITE = REPO / NUMERICS / "prompts" / "reference.json"
DEFAULT_COVERAGE_CORPUS = REPO / NUMERICS / "coverage" / "corpus.json"

REFERENCE_SCHEMA = "sparkflash-numerics-reference/1"
# A capture taken after the build record landed writes `server.build_source`: the key's presence is
# how the declaration check tells a new reference from one that predates the field, so a reference
# committed before it is grandfathered by the absent key (docs/NUMERICS.md §5).
BUILD_SOURCE = "build_source"
BUILD_SOURCES = ("card", "repo")
UNKNOWN_BUILD = "unknown"
# The correctness floor (docs/NUMERICS.md section 5). A first reference or a re-declaration must
# show that the arithmetic it pins is right, not merely repeatable: two identical captures of
# token soup are still soup. The floor records evidence taken outside the tool - a forced
# comparison against the pre-change build or the arm's host reference, a known-answer decode, and
# the reference's own NLL on natural text - in the card, and the declaration check re-checks it
# for every reference or [[anchor]] this change adds. A card without the key predates the rule and
# is grandfathered, exactly like the build record; the capture command writes it when asked to
# write a reference or an anchor.
FLOOR_FIELDS = ("against", "forced", "known_answers", "nll_per_token", "explanation")
FORCED_FIELDS = ("positions", "mean_nll_delta", "mean_abs_delta", "max_abs_delta", "top1_agreement")
FLOOR_BAR = {
    "top1_agreement": 0.90,
    "mean_abs_delta": 1.0,
    "max_abs_delta": 20.0,
    "nll_per_token": 2.0,
}
SUITE_SCHEMA = "sparkflash-numerics-prompts/1"
COMPARE_SCHEMA = "sparkflash-numerics-compare/1"
COVERAGE_SCHEMA = "sparkflash-numerics-coverage/1"
COVERAGE_GAPS_SCHEMA = "sparkflash-numerics-coverage-gaps/1"
COVERAGE_SWEEP_SCHEMA = "sparkflash-numerics-coverage-sweep/1"
COVERAGE_CORPUS_SCHEMA = "sparkflash-numerics-coverage-corpus/1"
COVERAGE_SEEN_SCHEMA = "sparkflash-numerics-coverage-seen/1"
MONITORS_SCHEMA = "sparkflash-numerics-monitors/1"
MONITOR_RESULT_SCHEMA = "sparkflash-numerics-monitor-result/1"
LAYOUT_SCHEMA = "sparkflash-numerics-expert-layout/1"
LAYOUTS = f"{NUMERICS}/layouts/"
MONITORS = f"{NUMERICS}/monitors/"
# The per-model evidence tables (docs/NUMERICS.md section 5). A `[[coverage]]` row names one
# model's committed dump and the suite's input floor; a `[[monitors]]` row names one model's pair
# and the bars it is judged at. The shapes are exact so a typo cannot silently leave a bar unset.
COVERAGE_ROW_FIELDS = ("model", "dump", "min_inputs")
MONITOR_ROW_FIELDS = ("model", "reference", "candidate", "layer_bar", "expert_bar")
PREREQUISITES = ("anchor", "declaration_check", "coverage", "monitors")
# The prerequisites that gate every version bump globally. 'anchor', 'coverage' and 'monitors' are
# not here: they are per-model requirements (docs/NUMERICS.md section 5), looked up for the bumping
# configuration's own model. The aggregate flags report those lookups; they do not gate them.
GATING_PREREQUISITES = ("declaration_check",)

KEY_FIELDS = ("model", "world", "profile", "plan", "arch")
# An owner's recorded approval of one declared change (docs/NUMERICS.md §5): a one-line decision,
# the day it was taken, and the report that justifies the change. It sits beside the version it
# lets land and is part of the record, like a version.
APPROVAL_FIELDS = ("decision", "date", "report")
EVIDENCE = ("logprobs", "text")
# A text reference holds no token ids and no logprobs, so it cannot be teacher-forced. A
# configuration with text evidence is exempt from the per-model anchor requirement; it also
# cannot bump, because there is no comparison that would justify the change.
ANCHOR_EXEMPT_EVIDENCE = ("text",)
TOKEN_ID_PREFIX = "token_id:"
# The arm profiles (docs/arm-reference.md §3): the defaults each arm serves. Each names the
# registry model it configures (`model`), which a sparkflash-model test holds to the descriptor its
# weights select; a profile that names none fails the declaration check.
PROFILE_GLOB = "crates/sparkflash-model-*/config/*.toml"
# Where the serving side lists the profile settings an invariance gate holds to the same bits. The
# check trusts the list, so the list is in the diff it judges.
CLASSIFICATION = "crates/sparkflash-model/src/numerics.rs"
ZERO_SHA = "0" * 40
# The smallest tail mass the KL bound divides by. A candidate that puts less than this outside the
# shared top-k while the anchor puts more is far past any bar, and the bound stays finite.
TAIL_FLOOR = 1e-12


class NumericsFailure(Exception):
    """A refusal or a failed check, written for an operator to read."""


# --- references and fingerprints -------------------------------------------------------------


def f32_bits(value: float | None) -> str:
    """The IEEE-754 binary32 bits of a logprob as eight hex digits.

    The server computes logprobs in f32 and serde writes the shortest decimal that round-trips the
    f32, so the nearest f32 of the parsed value is the value the server computed. A non-finite
    logprob reaches JSON as null and is kept as such.
    """
    if value is None:
        return "null"
    return struct.pack("<f", value).hex()


def prompt_bits(evidence: str, prompt: dict) -> dict:
    """The part of one prompt's capture that the fingerprint covers."""
    if evidence == "text":
        return {
            "prompt_ids": prompt["prompt_ids"],
            "text": prompt["text"],
            "finish_reason": prompt["finish_reason"],
        }
    return {
        "prompt_ids": prompt["prompt_ids"],
        "tokens": prompt["tokens"],
        "logprobs": [f32_bits(value) for value in prompt["logprobs"]],
        "top": [
            [[token, f32_bits(value)] for token, value in sorted(position)]
            for position in prompt["top"]
        ],
        "finish_reason": prompt["finish_reason"],
    }


def reference_bits(reference: dict) -> dict:
    """The fingerprinted body: the bits, never the dates, names or server card."""
    evidence = reference["evidence"]
    forced = reference.get("forced_from")
    return {
        "schema": REFERENCE_SCHEMA,
        "evidence": evidence,
        "top_logprobs": reference["settings"].get("top_logprobs"),
        "forced_from": forced["fingerprint"] if forced else None,
        "prompts": [prompt_bits(evidence, prompt) for prompt in reference["prompts"]],
    }


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def fingerprint(reference: dict) -> str:
    """The sha256 of a reference's canonical bits."""
    return hashlib.sha256(canonical(reference_bits(reference))).hexdigest()


def check_reference_shape(reference: dict, origin: str) -> None:
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise NumericsFailure(f"{origin}: schema {reference.get('schema')!r} is not {REFERENCE_SCHEMA}")
    if reference.get("evidence") not in EVIDENCE:
        raise NumericsFailure(f"{origin}: evidence {reference.get('evidence')!r} is not one of {EVIDENCE}")
    if not reference.get("prompts"):
        raise NumericsFailure(f"{origin}: the reference holds no prompts")
    # The server card is outside the fingerprint and grew a build record after the first
    # references were captured: an absent key is valid, a present one must be a string or null.
    server = reference.get("server")
    if server is not None:
        if not isinstance(server, dict):
            raise NumericsFailure(f"{origin}: the server card is not an object")
        for field in ("binary_sha256", "commit", "build_note"):
            value = server.get(field)
            if value is not None and not isinstance(value, str):
                raise NumericsFailure(f"{origin}: server.{field} is neither a string nor null")
        digest = server.get("binary_sha256")
        if isinstance(digest, str) and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise NumericsFailure(f"{origin}: server.binary_sha256 is not 64 lowercase hex digits")
        # The build record's version note: a new capture says where its commit came from, and a
        # reference without this key predates the record and is grandfathered.
        source = server.get(BUILD_SOURCE)
        if source is not None and source not in BUILD_SOURCES:
            raise NumericsFailure(
                f"{origin}: server.{BUILD_SOURCE} {source!r} is not one of {list(BUILD_SOURCES)}"
            )
    # The correctness floor is outside the fingerprint, like the server card: a reference that
    # predates the rule has no key, and one that carries it must be well formed.
    floor = reference.get("floor")
    if floor is not None:
        check_floor_shape(floor, origin)


def build_record_commit(reference: dict) -> str | None:
    """The commit a reference's server card records, or None when it names no build."""
    server = reference.get("server")
    if not isinstance(server, dict):
        return None
    commit = server.get("commit")
    if not isinstance(commit, str):
        return None
    value = commit.strip()
    if not value or value.lower().startswith(UNKNOWN_BUILD):
        return None
    return value


def build_record_problem(reference: dict) -> str | None:
    """Why a capture written after the build record landed cannot name its build.

    A reference that predates the record has no `server.build_source` key and is grandfathered:
    the capture command writes that key unconditionally, so a reference added from now on is
    judged. A dirty build passes, because its `-dirty` marker still names the commit, and the
    marker is what tells a reader the tree was not exact.
    """
    server = reference.get("server")
    if not isinstance(server, dict) or BUILD_SOURCE not in server:
        return None
    if build_record_commit(reference) is None:
        return (
            "server.commit names no build the capture came from; the served card reported no "
            "commit (or 'unknown'), so this reference cannot say which binary produced its bits "
            "(docs/NUMERICS.md §5)"
        )
    return None


def floor_bar(registry: dict, model: str) -> dict:
    """The correctness floor's thresholds for one model: the defaults or its [[precision]] row."""
    bar = dict(FLOOR_BAR)
    for row in registry.get("precision") or []:
        if row.get("model") != model:
            continue
        for field in bar:
            if field in row:
                bar[field] = float(row[field])
    return bar


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_floor_shape(floor: object, origin: str) -> None:
    """The floor block's shape, exact enough that a typo cannot silently drop evidence."""
    if not isinstance(floor, dict):
        raise NumericsFailure(f"{origin}: floor is not an object")
    unknown = sorted(set(floor) - set(FLOOR_FIELDS))
    if unknown:
        raise NumericsFailure(f"{origin}: floor has unknown key(s) {unknown}")
    against = floor.get("against")
    if not isinstance(against, str) or not against.strip():
        raise NumericsFailure(
            f"{origin}: floor.against names no pre-change build or host reference; the floor must "
            "say what the capture was measured against"
        )
    forced = floor.get("forced")
    if not isinstance(forced, dict):
        raise NumericsFailure(f"{origin}: floor.forced is not an object")
    missing = [field for field in FORCED_FIELDS if field not in forced]
    if missing:
        raise NumericsFailure(f"{origin}: floor.forced is missing {missing}")
    for field in FORCED_FIELDS:
        if not is_number(forced[field]):
            raise NumericsFailure(f"{origin}: floor.forced.{field} is not a number")
    if forced["positions"] <= 0:
        raise NumericsFailure(f"{origin}: floor.forced.positions is not positive")
    known = floor.get("known_answers")
    if not isinstance(known, list) or not known:
        raise NumericsFailure(
            f"{origin}: floor.known_answers is empty; the greedily decoded known answers are part "
            "of the floor"
        )
    for index, answer in enumerate(known):
        if not isinstance(answer, dict):
            raise NumericsFailure(f"{origin}: floor.known_answers[{index}] is not an object")
        unknown = sorted(set(answer) - {"prompt", "expected", "got", "pass"})
        if unknown:
            raise NumericsFailure(f"{origin}: floor.known_answers[{index}] has unknown key(s) {unknown}")
        for field in ("prompt", "expected", "got"):
            if not isinstance(answer.get(field), str):
                raise NumericsFailure(f"{origin}: floor.known_answers[{index}].{field} is not a string")
        if not answer["prompt"].strip():
            raise NumericsFailure(f"{origin}: floor.known_answers[{index}].prompt is empty")
        if not isinstance(answer.get("pass"), bool):
            raise NumericsFailure(f"{origin}: floor.known_answers[{index}].pass is not a boolean")
    if not any(answer["expected"].strip() for answer in known):
        raise NumericsFailure(
            f"{origin}: floor.known_answers names no expected answer; one known answer must say "
            "what the model should decode"
        )
    nll = floor.get("nll_per_token")
    if not is_number(nll) or nll <= 0:
        raise NumericsFailure(f"{origin}: floor.nll_per_token is not a positive number")
    explanation = floor.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise NumericsFailure(f"{origin}: floor.explanation is neither a string nor null")


def validate_floor(floor: object, origin: str, bar: dict) -> dict:
    """The floor's shape, then its bars; a threshold miss needs an explanation to pass.

    The bars are in docs/NUMERICS.md section 5 and default to FLOOR_BAR; a model may tighten or
    loosen its own through a [[precision]] row. A distance far past the arm's precision tail
    (Inkling's 47-nat swings and 29% top-1) is a defect until explained, so the explanation is
    recorded beside the numbers rather than the numbers being dropped.
    """
    check_floor_shape(floor, origin)
    forced = floor["forced"]
    problems = []
    if forced["top1_agreement"] < bar["top1_agreement"]:
        problems.append(
            f"top-1 agreement {forced['top1_agreement']:.4%} is below {bar['top1_agreement']:.2%}"
        )
    if abs(forced["mean_abs_delta"]) > bar["mean_abs_delta"]:
        problems.append(
            f"mean absolute NLL delta {forced['mean_abs_delta']:.3e} is above {bar['mean_abs_delta']:.3e}"
        )
    if abs(forced["max_abs_delta"]) > bar["max_abs_delta"]:
        problems.append(
            f"max absolute NLL delta {forced['max_abs_delta']:.3e} is above {bar['max_abs_delta']:.3e}"
        )
    if floor["nll_per_token"] > bar["nll_per_token"]:
        problems.append(
            f"NLL per token {floor['nll_per_token']:.4f} is above {bar['nll_per_token']:.4f}"
        )
    failed = [answer for answer in floor["known_answers"] if not answer["pass"]]
    for answer in failed:
        problems.append(
            f"known answer {answer['prompt']!r} expected {answer['expected']!r} but decoded "
            f"{answer['got'][:60]!r}"
        )
    if problems and not (floor.get("explanation") or "").strip():
        raise NumericsFailure(
            f"{origin}: the correctness floor fails: " + "; ".join(problems)
            + " (docs/NUMERICS.md §5; an explained distance may be pinned, an unexplained one may not)"
        )
    return floor


def floor_problem(reference: dict, model: str, registry: dict) -> str | None:
    """Why a reference or anchor this change adds lacks a usable correctness floor, or None."""
    floor = reference.get("floor")
    if floor is None:
        return (
            "no correctness floor: a first reference or a re-declaration must record a forced "
            "comparison, its known-answer decode and its own NLL per token (docs/NUMERICS.md §5)"
        )
    try:
        validate_floor(floor, "floor", floor_bar(registry, model))
    except NumericsFailure as error:
        return str(error)
    return None


def load_reference(path: Path) -> dict:
    try:
        reference = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise NumericsFailure(f"{path}: {error}") from error
    check_reference_shape(reference, str(path))
    return reference


def render_reference(reference: dict) -> str:
    """The file form: the header indented, one prompt per line, so a diff names the prompt."""
    header = {key: value for key, value in reference.items() if key != "prompts"}
    lines = json.dumps(header, indent=2, sort_keys=True).rstrip("}").rstrip().rstrip(",")
    prompts = ",\n".join(
        "    " + json.dumps(prompt, sort_keys=True, separators=(",", ":"))
        for prompt in reference["prompts"]
    )
    return f'{lines},\n  "prompts": [\n{prompts}\n  ]\n}}\n'


def extends(old: dict, new: dict) -> bool:
    """Whether `new` keeps every bit `old` had and only adds to it.

    A longer generation or another prompt is more evidence, not a changed arithmetic: every old
    prompt must reappear with the same ids, and its generation must reappear whole, or as the
    prefix of a longer one when the old one ran out of budget.
    """
    old_bits, new_bits = reference_bits(old), reference_bits(new)
    for field in ("evidence", "top_logprobs", "forced_from"):
        if old_bits[field] != new_bits[field]:
            return False
    candidates = {}
    for prompt in new_bits["prompts"]:
        candidates.setdefault(canonical(prompt["prompt_ids"]), []).append(prompt)
    for prompt in old_bits["prompts"]:
        matches = candidates.get(canonical(prompt["prompt_ids"]), [])
        if not any(prefix_of(old_bits["evidence"], prompt, match) for match in matches):
            return False
    return True


def prefix_of(evidence: str, old: dict, new: dict) -> bool:
    """Whether `new` is `old`, or `old` (stopped by its budget) continued: a generation of the same
    length keeps every bit, its finish reason included."""
    if old == new:
        return True
    if old["finish_reason"] != "length":
        return False
    if evidence == "text":
        return len(new["text"]) > len(old["text"]) and new["text"].startswith(old["text"])
    count = len(old["tokens"])
    return len(new["tokens"]) > count and all(
        new[field][:count] == old[field] for field in ("tokens", "logprobs", "top")
    )


# --- the server -------------------------------------------------------------------------------


def http_json(url: str, body: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"} if data is not None else {},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise NumericsFailure(f"{url}: HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise NumericsFailure(f"{url}: {error}") from error


def fetch_card(url: str, timeout: float) -> dict:
    models = http_json(f"{url}/v1/models", timeout=timeout).get("data") or []
    if len(models) != 1:
        raise NumericsFailure(f"{url}/v1/models lists {len(models)} models; capture needs exactly one")
    card = models[0]
    if not isinstance(card.get("numerics"), dict):
        raise NumericsFailure(
            f"{url}: the model card has no numerics identity; the server predates D67 or was "
            "started without one, so nothing says which configuration a capture would describe"
        )
    return card


def card_key(card: dict) -> dict:
    numerics = card["numerics"]
    return {
        "model": card["id"],
        "world": numerics["world"],
        "profile": numerics["profile"],
        "plan": numerics["plan"],
        "arch": numerics["arch"],
    }


def checkpoint_identity(path: str | None) -> str | None:
    """The checkpoint a card names, without the directory it sits in on the serving host.

    A Hugging Face cache path keeps its repository and snapshot revision, which is what identifies
    the weights; anything else keeps its last component. A committed reference never records a
    host's home directory.
    """
    if not path:
        return path
    parts = Path(path).parts
    repo = next((index for index, part in enumerate(parts) if part.startswith("models--")), None)
    return "/".join(parts[repo:]) if repo is not None else parts[-1]


def sha256_file(path: Path) -> str:
    """The sha256 of a file, read in blocks so a large binary is not held whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_at(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git in a tree other than this checkout's."""
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=False, check=False)


def card_build_commit(card: dict | None) -> tuple[str | None, str | None]:
    """The commit a served card names, and the one-line reason it names no commit.

    `/v1/models` reports the build it was compiled from as `build.id` (the commit, with
    `-dirty` when the tree was dirty at build time). A card that reports `unknown` names no
    commit, so the caller falls back to `--repo` exactly as it would for a card with no id.
    """
    if not isinstance(card, dict):
        return None, "the capture has no served card to read a build from"
    build = card.get("build")
    if not isinstance(build, dict):
        return None, "the served card has no build record"
    raw = build.get("id")
    if not isinstance(raw, str) or not raw.strip():
        return None, "the served card reports no build id"
    value = raw.strip()
    if value.lower().startswith(UNKNOWN_BUILD):
        return None, f"the served card reports build id {value!r}, which does not name a commit"
    if value.endswith("-dirty"):
        return value, (
            "the served card reports a dirty build; server.commit keeps its -dirty marker, so the "
            "build it names is not an exact tree"
        )
    return value, None


def build_identity(binary: Path | None, repo: Path | None, card: dict | None = None) -> dict:
    """The build a capture came from: its binary, its commit and where the commit was read.

    The commit is the served card's `build.id` first, because that is the only record the running
    binary made of its own build; `--repo` is a fallback for a build that reports nothing. A dirty
    build keeps its `-dirty` marker in `server.commit`, so the reader sees it is not an exact
    tree. When a field cannot be determined the card records null and `build_note` gives the
    one-line reason; a value is never guessed. The reason names no host path, because the reference
    is committed. `build_source` is the version note a new capture writes: "card", "repo", or null.
    """
    reasons = []
    digest = None
    if binary is None:
        reasons.append("no --binary or SPARKFLASH_BIN named the binary")
    else:
        try:
            digest = sha256_file(binary)
        except OSError as error:
            reasons.append(f"the binary is not readable: {error.strerror or error}")
    commit = None
    source = None
    card_commit, card_reason = card_build_commit(card)
    if card_commit is not None:
        commit, source = card_commit, "card"
        if card_reason is not None:
            reasons.append(card_reason)
    else:
        if card is not None:
            reasons.append(card_reason)
        if repo is None:
            reasons.append("no --repo named the tree the server was built from")
        else:
            shown = git_at(repo, "rev-parse", "HEAD")
            if shown.returncode != 0:
                reasons.append("the tree named by --repo is not a git work tree with a HEAD")
            else:
                commit = shown.stdout.decode().strip()
                source = "repo"
                if git_at(repo, "status", "--porcelain").stdout.strip():
                    reasons.append(
                        "the tree named by --repo has uncommitted changes, so its commit does not "
                        "name the exact build"
                    )
    return {
        "binary_sha256": digest,
        "commit": commit,
        "build_note": "; ".join(reasons) or None,
        "build_source": source,
    }


def key_text(key: dict) -> str:
    return " ".join([str(key["model"])] + [f"{field}={key[field]}" for field in KEY_FIELDS[1:]])


def parse_token_id(text: str, origin: str) -> int:
    if not text.startswith(TOKEN_ID_PREFIX):
        raise NumericsFailure(
            f"{origin}: token {text!r} is not rendered as {TOKEN_ID_PREFIX}N; the server ignored "
            "return_tokens_as_token_ids"
        )
    return int(text[len(TOKEN_ID_PREFIX):])


def suite_prompts(suite_path: Path, url: str, model: str, timeout: float) -> list[dict]:
    """The suite's prompts as ids, tokenized by the server's own codec once, at capture."""
    try:
        suite = json.loads(suite_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise NumericsFailure(f"{suite_path}: {error}") from error
    if suite.get("schema") != SUITE_SCHEMA:
        raise NumericsFailure(f"{suite_path}: schema {suite.get('schema')!r} is not {SUITE_SCHEMA}")
    prompts = []
    for entry in suite.get("prompts") or []:
        if "prompt_ids" in entry:
            ids = entry["prompt_ids"]
        elif "messages" in entry:
            ids = http_json(
                f"{url}/tokenize", {"model": model, "messages": entry["messages"]}, timeout
            )["tokens"]
        elif "prompt" in entry:
            ids = http_json(
                f"{url}/tokenize",
                {"model": model, "prompt": entry["prompt"], "add_special_tokens": True},
                timeout,
            )["tokens"]
        else:
            raise NumericsFailure(f"{suite_path}: prompt {entry.get('name')!r} has no messages, prompt or prompt_ids")
        prompts.append({"name": entry["name"], "prompt_ids": ids})
    if not prompts:
        raise NumericsFailure(f"{suite_path}: the suite holds no prompts")
    return prompts


def run_prompt(
    url: str,
    model: str,
    prompt: dict,
    evidence: str,
    max_tokens: int,
    top_logprobs: int,
    forced: list[int] | None,
    timeout: float,
) -> dict:
    origin = f"prompt {prompt['name']!r}"
    # Greedy for both kinds: a request without a temperature samples at the API's default of 1.0,
    # and a sampled capture never reproduces.
    body: dict = {"model": model, "prompt": prompt["prompt_ids"], "temperature": 0}
    if evidence == "text":
        # The DeepSeek device path serves neither logprobs nor token ids, so ask for neither.
        body["max_tokens"] = max_tokens
    else:
        body.update(
            {
                "logprobs": top_logprobs,
                "return_tokens_as_token_ids": True,
            }
        )
        if forced is None:
            body["max_tokens"] = max_tokens
        else:
            body["forced_tokens"] = forced
    choice = http_json(f"{url}/v1/completions", body, timeout)["choices"][0]
    record = {
        "name": prompt["name"],
        "prompt_ids": prompt["prompt_ids"],
        "finish_reason": choice.get("finish_reason"),
        "text": choice.get("text", ""),
    }
    if evidence == "text":
        return record
    logprobs = choice.get("logprobs") or {}
    tokens = [parse_token_id(token, origin) for token in logprobs.get("tokens") or []]
    top = []
    for position in logprobs.get("top_logprobs") or []:
        top.append([[parse_token_id(token, origin), value] for token, value in position.items()])
    if len(top) != len(tokens) or len(logprobs.get("token_logprobs") or []) != len(tokens):
        raise NumericsFailure(f"{origin}: the logprobs block is ragged ({len(tokens)} tokens, {len(top)} top lists)")
    if forced is not None and tokens != forced:
        raise NumericsFailure(
            f"{origin}: the server answered tokens other than the forced ones; first difference at "
            f"position {next((i for i, (a, b) in enumerate(zip(tokens, forced)) if a != b), min(len(tokens), len(forced)))}"
        )
    record.update({"tokens": tokens, "logprobs": logprobs["token_logprobs"], "top": top})
    return record


def capture(
    url: str,
    prompts: list[dict],
    evidence: str,
    max_tokens: int,
    top_logprobs: int,
    forced_from: dict | None,
    timeout: float,
    on_prompt: Callable[[dict], None] = lambda _record: None,
    *,
    binary: Path | None = None,
    repo: Path | None = None,
) -> dict:
    """Capture `prompts` from the server at `url` into a reference document."""
    card = fetch_card(url, timeout)
    if forced_from is not None:
        if evidence != "logprobs":
            raise NumericsFailure("a forced capture scores tokens, so its evidence is logprobs")
        if card.get("capabilities", {}).get("forced_decoding") is not True:
            raise NumericsFailure(
                f"{url}: the server does not advertise forced_decoding, so it cannot score another "
                "reference's tokens; teacher forcing needs a driver that declares it"
            )
    records = []
    for prompt in prompts:
        forced = prompt["tokens"] if forced_from is not None else None
        record = run_prompt(url, card["id"], prompt, evidence, max_tokens, top_logprobs, forced, timeout)
        on_prompt(record)
        records.append(record)
    settings = {"max_tokens": max_tokens}
    if evidence == "logprobs":
        settings["top_logprobs"] = top_logprobs
    server_build = build_identity(binary, repo, card)
    if server_build["build_source"] != "card":
        print(
            f"warning: {url}: the served card names no build commit; server.commit is "
            f"{server_build['commit'] or 'null'} (docs/NUMERICS.md §5)",
            file=sys.stderr,
        )
    reference = {
        "schema": REFERENCE_SCHEMA,
        "evidence": evidence,
        "key": card_key(card),
        "server": {
            "numerics": card["numerics"],
            "build": card.get("build"),
            "checkpoint": checkpoint_identity(card.get("checkpoint")),
            **server_build,
        },
        "settings": settings,
        "forced_from": forced_from,
        "captured": datetime.date.today().isoformat(),
        "prompts": records,
    }
    reference["fingerprint"] = fingerprint(reference)
    return reference


def first_difference(old: dict, new: dict) -> str:
    """Where two captures of the same prompts first disagree, for the failure message."""
    old_bits, new_bits = reference_bits(old), reference_bits(new)
    for field in ("evidence", "top_logprobs", "forced_from"):
        if old_bits[field] != new_bits[field]:
            return f"{field}: {old_bits[field]!r} != {new_bits[field]!r}"
    for index, (a, b) in enumerate(zip(old_bits["prompts"], new_bits["prompts"])):
        if a == b:
            continue
        name = old["prompts"][index]["name"]
        if a["prompt_ids"] != b["prompt_ids"]:
            return f"prompt {name!r}: the prompt ids differ"
        if old_bits["evidence"] == "text":
            common = next((i for i, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y), None)
            return f"prompt {name!r}: text differs at character {common if common is not None else min(len(a['text']), len(b['text']))}"
        for position in range(max(len(a["tokens"]), len(b["tokens"]))):
            for field in ("tokens", "logprobs", "top"):
                left = a[field][position] if position < len(a[field]) else None
                right = b[field][position] if position < len(b[field]) else None
                if left != right:
                    return f"prompt {name!r} position {position}: {field} {left!r} != {right!r}"
        return f"prompt {name!r}: finish_reason {a['finish_reason']!r} != {b['finish_reason']!r}"
    return f"prompt count {len(old_bits['prompts'])} != {len(new_bits['prompts'])}"


# --- the registry -----------------------------------------------------------------------------


def coverage_bar(value: object, origin: str) -> float:
    """The declared coverage bar: the fraction of routed (layer, expert) slots a dump must reach.

    It is a fraction of the committed layout's routed slots, not a count and not 1.0 by default.
    Raising a gate to every slot is a choice the registry states, not an unstated implication of
    "coverage" (docs/NUMERICS.md §5).
    """
    if value is None:
        raise NumericsFailure(
            f"{origin}: prerequisites.coverage_slots is missing; state the fraction of routed "
            "(layer, expert) slots a committed coverage dump must reach (for example 0.93)"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericsFailure(f"{origin}: prerequisites.coverage_slots {value!r} is not a number")
    bar = float(value)
    if not math.isfinite(bar) or not 0.0 < bar <= 1.0:
        raise NumericsFailure(
            f"{origin}: prerequisites.coverage_slots {value!r} is outside (0, 1]; it is the "
            "fraction of routed (layer, expert) slots a coverage dump must reach"
        )
    return bar


def monitor_bar(value: object, field: str, origin: str) -> float:
    """One declared monitor bar: a finite, positive relative error. Zero would be a bit gate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericsFailure(f"{origin}: {field} {value!r} is not a number")
    bar = float(value)
    if not math.isfinite(bar) or bar <= 0.0:
        raise NumericsFailure(
            f"{origin}: {field} {value!r} is not finite and positive; it is the relative-error "
            "bar a committed monitor row must meet"
        )
    return bar


def evidence_path(value: object, directory: str, field: str, origin: str) -> str:
    """A JSON dump under one of the numerics evidence directories, nothing else."""
    if not isinstance(value, str) or not value:
        raise NumericsFailure(f"{origin}: {field} is not a non-empty string")
    parts = PurePosixPath(value).parts
    if (
        not value.startswith(directory)
        or not value.endswith(".json")
        or "." in parts
        or ".." in parts
    ):
        raise NumericsFailure(
            f"{origin}: {field} {value!r} is not a JSON dump under {directory}"
        )
    return value


def row_label(table: str, value: object, origin: str) -> str:
    """The error prefix for one evidence row, naming its model whenever the row says one."""
    model = value.get("model") if isinstance(value, dict) else None
    if isinstance(model, str) and model:
        return f"{origin}: [[{table}]] {model!r}"
    return f"{origin}: [[{table}]]"


def row_fields(
    table: str, value: object, fields: tuple[str, ...], origin: str
) -> dict:
    """One evidence table's exact shape, refusing a missing key, an unknown one or no model."""
    if not isinstance(value, dict):
        raise NumericsFailure(f"{origin}: [[{table}]] is not a table")
    label = row_label(table, value, origin)
    missing = [field for field in fields if field not in value]
    unknown = sorted(set(value) - set(fields))
    if missing:
        raise NumericsFailure(
            f"{label} is missing {missing}; it needs {list(fields)}"
        )
    if unknown:
        raise NumericsFailure(
            f"{label} has unknown key(s) {unknown}; it needs {list(fields)}"
        )
    model = value["model"]
    if not isinstance(model, str) or not model:
        raise NumericsFailure(f"{label}: model is not a non-empty string")
    return value


def coverage_row(value: object, origin: str) -> dict:
    """One model's committed coverage dump and its suite input floor (docs/NUMERICS.md §5)."""
    row = row_fields("coverage", value, COVERAGE_ROW_FIELDS, origin)
    label = row_label("coverage", row, origin)
    evidence_path(row["dump"], f"{NUMERICS}/coverage/", "dump", label)
    inputs = row["min_inputs"]
    if isinstance(inputs, bool) or not isinstance(inputs, int) or inputs <= 0:
        raise NumericsFailure(
            f"{label}: min_inputs {inputs!r} is not a positive integer; it is the declared "
            "coverage suite's input floor"
        )
    return row


def monitors_row(value: object, origin: str) -> dict:
    """One model's committed monitor pair and the bars it is judged at (docs/NUMERICS.md §5)."""
    row = row_fields("monitors", value, MONITOR_ROW_FIELDS, origin)
    label = row_label("monitors", row, origin)
    for field in ("reference", "candidate"):
        evidence_path(row[field], MONITORS, field, label)
    monitor_bar(row["layer_bar"], "layer_bar", label)
    monitor_bar(row["expert_bar"], "expert_bar", label)
    return row


def one_row_a_model(rows: list[dict], table: str, origin: str) -> None:
    """Refuse a second row for a model: one row a model, so the lookup is unambiguous."""
    seen = set()
    for row in rows:
        if row["model"] in seen:
            raise NumericsFailure(
                f"{origin}: [[{table}]] has more than one row for model {row['model']!r}; "
                "one row a model"
            )
        seen.add(row["model"])


def parse_registry(text: str, origin: str, require_bar: bool = False) -> dict:
    try:
        registry = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise NumericsFailure(f"{origin}: {error}") from error
    prerequisites = registry.get("prerequisites") or {}
    missing = [name for name in PREREQUISITES if not isinstance(prerequisites.get(name), bool)]
    if missing:
        raise NumericsFailure(f"{origin}: prerequisites {missing} are not set to true or false")
    # A registry written before the bar existed is readable as a base (the declaration check reads
    # history); the head always states it. A bar that is present is valid everywhere.
    if require_bar or "coverage_slots" in prerequisites:
        coverage_bar(prerequisites.get("coverage_slots"), origin)
    # The evidence is per model (docs/NUMERICS.md §5): every row is validated here, so a malformed
    # row cannot silently leave a bar unset, and two rows for one model cannot disagree.
    for table, check in (("coverage", coverage_row), ("monitors", monitors_row)):
        rows = registry.get(table, [])
        if not isinstance(rows, list):
            raise NumericsFailure(f"{origin}: [[{table}]] is not an array of tables")
        for row in rows:
            check(row, origin)
        one_row_a_model(rows, table, origin)
    # A [[precision]] row tightens or loosens one model's correctness-floor bars. Its fields are
    # the floor's numeric bars; the shape is exact so a typo cannot silently leave a bar unset.
    rows = registry.get("precision", [])
    if not isinstance(rows, list):
        raise NumericsFailure(f"{origin}: [[precision]] is not an array of tables")
    for row in rows:
        if not isinstance(row.get("model"), str) or not row["model"].strip():
            raise NumericsFailure(f"{origin}: a [[precision]] row names no model")
        unknown = sorted(set(row) - {"model", *FLOOR_BAR})
        if unknown:
            raise NumericsFailure(f"{origin}: precision {row['model']} has unknown key(s) {unknown}")
        for field in FLOOR_BAR:
            if field in row and (not is_number(row[field]) or row[field] <= 0):
                raise NumericsFailure(f"{origin}: precision {row['model']}.{field} is not positive")
    one_row_a_model(rows, "precision", origin)
    for table in ("config", "anchor", "retired", "coverage", "monitors", "precision"):
        registry.setdefault(table, [])
    return registry


def approval_problem(entry: dict) -> str | None:
    """Why an entry's `approved` block is not a usable record, or None when it is well formed.

    The block names the owner's decision that lets one bump land past its model's coverage and
    monitors prerequisites (docs/NUMERICS.md §5). Its report's existence and freshness are judged
    at the bump, because a carried record's report belongs to an earlier change.
    """
    approved = entry.get("approved")
    if approved is None:
        return None
    if not isinstance(approved, dict):
        return "approved is not a table"
    unknown = sorted(set(approved) - set(APPROVAL_FIELDS))
    if unknown:
        return f"approved has unknown key(s) {unknown}"
    missing = [field for field in APPROVAL_FIELDS if not isinstance(approved.get(field), str)]
    if missing:
        return f"approved {missing} are not all strings"
    if not approved["decision"].strip():
        return "approved.decision is empty"
    if "\n" in approved["decision"].strip():
        return "approved.decision is not one line"
    if not approved["date"].strip():
        return "approved.date is empty"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", approved["date"]):
        return f"approved.date {approved['date']!r} is not a YYYY-MM-DD date"
    try:
        datetime.date.fromisoformat(approved["date"])
    except ValueError:
        return f"approved.date {approved['date']!r} is not a calendar date"
    if not approved["report"].strip():
        return "approved.report is empty"
    if not approved["report"].startswith("docs/numerics/"):
        return f"approved.report {approved['report']!r} is not under docs/numerics/"
    return None


def entry_key(entry: dict) -> tuple:
    return tuple(entry[field] for field in KEY_FIELDS)


def lookup(registry: dict, key: dict) -> dict | None:
    wanted = tuple(key[field] for field in KEY_FIELDS)
    return next((entry for entry in registry["config"] if entry_key(entry) == wanted), None)


def missing_prerequisites(registry: dict) -> list[str]:
    return [name for name in PREREQUISITES if not registry["prerequisites"][name]]


def anchor_for(registry: dict, model: str) -> dict | None:
    """The frozen anchor for one model, if the registry holds one.

    An anchor is per model (docs/NUMERICS.md §4): the declaration check looks up the bumping
    configuration's own model, not a single global answer that every arm has to satisfy.
    """
    return next((anchor for anchor in registry["anchor"] if anchor["model"] == model), None)


def coverage_for(registry: dict, model: str) -> dict | None:
    """The coverage row for one model, if the registry holds one (docs/NUMERICS.md §5)."""
    return next((row for row in registry.get("coverage", []) if row["model"] == model), None)


def monitors_for(registry: dict, model: str) -> dict | None:
    """The monitors row for one model, if the registry holds one (docs/NUMERICS.md §5)."""
    return next((row for row in registry.get("monitors", []) if row["model"] == model), None)


def logprobs_models(registry: dict) -> list[str]:
    """The models whose captured evidence is logprobs: what the aggregate flags report over.

    A model with no captured reference is not yet part of the aggregate: it has nothing a
    prerequisite could qualify. Text evidence is exempt from these comparisons, as it is from the
    anchor, because a text reference has no token ids and no logprobs to compare.
    """
    return sorted(
        {
            entry["model"]
            for entry in registry.get("config", [])
            if "reference" in entry and entry.get("evidence") not in ANCHOR_EXEMPT_EVIDENCE
        }
    )


def forced_captures(paths: list[str], read: Callable[[str], bytes | None]) -> dict[str, set[str]]:
    """The anchor fingerprints each model has a committed capture teacher-forced from.

    A capture names the reference whose tokens it was fed via `forced_from.fingerprint`; that is
    the input side of the comparison `scripts/numerics.py compare` scores. The map is by model,
    and a malformed capture simply contributes nothing (the comparison is then missing).
    """
    forced: dict[str, set[str]] = {}
    for path in paths:
        data = read(path)
        if data is None:
            continue
        try:
            capture = json.loads(data)
        except json.JSONDecodeError:
            continue
        if capture.get("schema") != REFERENCE_SCHEMA:
            continue
        source = capture.get("forced_from") or {}
        fingerprint, key = source.get("fingerprint"), capture.get("key") or {}
        model = key.get("model")
        if isinstance(fingerprint, str) and isinstance(model, str):
            forced.setdefault(model, set()).add(fingerprint)
    return forced


def anchor_coverage(registry: dict, forced: dict[str, set[str]]) -> dict:
    """Which models the per-model anchor prerequisite covers, and which it does not.

    A captured configuration with `logprobs` evidence can be teacher-forced, so its model needs
    a frozen anchor and at least one committed capture forced from that anchor. A model whose
    every captured configuration has `text` evidence is exempt: text holds no ids to feed, so
    there is no comparison to make. The exemption does not make such a model bumpable; a text
    evidence change is refused separately.
    """
    bumpable: set[str] = set()
    text: set[str] = set()
    for entry in registry["config"]:
        if "reference" not in entry:
            continue
        if entry["evidence"] in ANCHOR_EXEMPT_EVIDENCE:
            text.add(entry["model"])
        else:
            bumpable.add(entry["model"])
    anchors = {anchor["model"]: anchor for anchor in registry["anchor"]}
    covered: dict[str, dict] = {}
    uncovered: dict[str, str] = {}
    for model in sorted(bumpable):
        anchor = anchors.get(model)
        if anchor is None:
            uncovered[model] = "the registry holds no [[anchor]] for it"
        elif anchor["evidence"] in ANCHOR_EXEMPT_EVIDENCE:
            uncovered[model] = (
                f"its anchor's evidence is {anchor['evidence']!r}, which has no token ids to teacher-force"
            )
        elif anchor["fingerprint"] not in forced.get(model, set()):
            uncovered[model] = (
                f"no committed capture is teacher-forced from its anchor {anchor['fingerprint'][:16]}"
            )
        else:
            covered[model] = anchor
    return {
        "exempt": sorted(text - bumpable),
        "bumpable": sorted(bumpable),
        "covered": covered,
        "uncovered": uncovered,
    }


def anchor_findings(
    registry: dict,
    read_head: Callable[[str], bytes | None],
    capture_paths: list[str],
) -> tuple[list[str], dict]:
    """The [[anchor]]-scoped check: every anchor is usable, and every model that needs one has one.

    The aggregate `prerequisites.anchor` flag is a report of this coverage, not a gate on a model
    that has its own anchor. A false flag with an uncovered model is consistent; a true flag with
    one is not, and is refused.
    """
    findings: list[str] = []
    forced = forced_captures(capture_paths, read_head)
    for anchor in registry["anchor"]:
        model = anchor["model"]
        if anchor.get("evidence") in ANCHOR_EXEMPT_EVIDENCE:
            findings.append(
                f"anchor {model}: evidence {anchor['evidence']!r} has no token ids, so no comparison "
                "can be teacher-forced from it; an anchor must hold logprobs"
            )
        if anchor["fingerprint"] not in forced.get(model, set()):
            findings.append(
                f"anchor {model}: no committed capture under {CAPTURES} is teacher-forced from "
                f"{anchor['fingerprint'][:16]}; the anchor has no comparison"
            )
    coverage = anchor_coverage(registry, forced)
    # The aggregate flag reports coverage; it does not gate a model that has its own anchor. A
    # false flag with an uncovered model is therefore consistent, and each bump is refused by the
    # per-model lookup in declaration_findings. A true flag that claims more than the registry
    # holds is not.
    if registry["prerequisites"]["anchor"] and coverage["uncovered"]:
        findings.append(
            "prerequisites.anchor is true, but these models have no anchor coverage: "
            f"{sorted(coverage['uncovered'])}; the flag reports the aggregate, so it cannot claim "
            "more than the registry holds"
        )
    return findings, coverage


def entry_snippet(reference: dict, reference_path: str, registered: dict | None = None) -> str:
    """The registry entry for a capture: version 1 of a new configuration, or the `registered`
    entry it recaptures with its version, report and note kept, since only a declaration moves
    them."""
    key = reference["key"]
    lines = ["[[config]]"]
    for field in KEY_FIELDS:
        value = key[field]
        lines.append(f"{field} = {json.dumps(value)}")
    lines += [
        f"version = {registered['version'] if registered else 1}",
        f"evidence = {json.dumps(reference['evidence'])}",
        f"reference = {json.dumps(reference_path)}",
        f"fingerprint = {json.dumps(reference['fingerprint'])}",
    ]
    if registered and registered.get("report"):
        lines.append(f"report = {json.dumps(registered['report'])}")
    approved = (registered or {}).get("approved")
    if isinstance(approved, dict) and all(isinstance(approved.get(field), str) for field in APPROVAL_FIELDS):
        inline = ", ".join(f"{field} = {json.dumps(approved[field])}" for field in APPROVAL_FIELDS)
        lines.append(f"approved = {{ {inline} }}")
    lines.append(f"captured = {json.dumps(reference['captured'])}")
    if registered and registered.get("note"):
        lines.append(f"note = {json.dumps(registered['note'])}")
    return "\n".join(lines)


# --- the declaration check --------------------------------------------------------------------


def is_profile(path: str) -> bool:
    """Whether a repository path is an arm profile: [PROFILE_GLOB], matched whole."""
    pure = PurePosixPath(path)
    return len(pure.parts) == len(PurePosixPath(PROFILE_GLOB).parts) and pure.match(PROFILE_GLOB)


def profile_settings(data: bytes | None, origin: str) -> dict:
    """A profile's settings by dotted name (`decode.tensor_cores`); none for an absent file."""
    if data is None:
        return {}
    try:
        table = tomllib.loads(data.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise NumericsFailure(f"{origin}: {error}") from error
    settings = {}

    def walk(table: dict, prefix: str) -> None:
        for name, value in table.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{name}.")
            else:
                settings[f"{prefix}{name}"] = value

    walk(table, "")
    return settings


def neutral_fields(source: bytes | None) -> set[str] | None:
    """The serving side's `NEUTRAL_FIELDS`, the profile settings an invariance gate holds to the
    same bits, or None when the source holds no such list."""
    if source is None:
        return None
    listed = re.search(rb"pub const NEUTRAL_FIELDS: &\[&str\] = &\[(.*?)\];", source, re.S)
    if listed is None:
        return None
    return {name.decode() for name in re.findall(rb'"([^"]*)"', listed.group(1))}


def declaration_findings(
    base: dict | None,
    head: dict,
    read_base: Callable[[str], bytes | None],
    read_head: Callable[[str], bytes | None],
    changed: dict[str, str],
    forced: dict[str, set[str]] | None = None,
) -> list[str]:
    """Every way the head registry breaks the declaration rules against the base.

    `changed` maps each repository path the diff touches under numerics/, docs/numerics/, the arm
    profiles and [CLASSIFICATION] to its git status letter (A, M or D); `read_base` and
    `read_head` return a path's bytes at either end, or None. `forced` maps a model to the anchor
    fingerprints a committed capture was teacher-forced from; None means the caller has no
    captures to judge (the synthetic self-test), so the comparison half is not checked.
    """
    findings: list[str] = []

    def loaded(read: Callable[[str], bytes | None], path: str) -> dict | None:
        data = read(path)
        if data is None:
            return None
        try:
            reference = json.loads(data)
            check_reference_shape(reference, path)
            return reference
        except (json.JSONDecodeError, NumericsFailure) as error:
            findings.append(f"{path}: not a reference: {error}")
            return None

    # Every fingerprint at the head matches the file it names, the file was captured from the
    # configuration that names it, every reference file is named, and every named report exists.
    named = set()
    for table in ("config", "anchor"):
        for entry in head[table]:
            key = dict(zip(KEY_FIELDS, entry_key(entry)))
            report = entry.get("report")
            if report and read_head(report) is None:
                findings.append(f"{table} {key_text(key)}: report {report} does not exist")
            if "reference" not in entry:
                if table == "config":
                    # Registered, not yet captured: no reference to match, and the entry is
                    # reported as uncaptured by run_ci.
                    continue
                findings.append(f"{table} {key_text(key)}: an anchor must name a captured reference")
                continue
            path = f"{NUMERICS}/{entry['reference']}"
            named.add(path)
            reference = loaded(read_head, path)
            if reference is None:
                findings.append(f"{table} {key_text(key)}: {path} does not exist")
                continue
            if reference.get("key") != key:
                captured_from = key_text(reference["key"]) if reference.get("key") else "no recorded configuration"
                findings.append(f"{path}: captured from {captured_from}, but the {table} entry naming it is {key_text(key)}")
            actual = fingerprint(reference)
            if actual != entry["fingerprint"]:
                findings.append(
                    f"{path}: its bits fingerprint {actual}, but the registry says {entry['fingerprint']}"
                )
            if reference.get("fingerprint") != actual:
                findings.append(f"{path}: the file's own fingerprint field is stale ({reference.get('fingerprint')})")
            if reference.get("evidence") != entry["evidence"]:
                findings.append(f"{path}: evidence {reference.get('evidence')!r}, but the registry says {entry['evidence']!r}")
    # A retired configuration keeps its last reference in history, and a replaced anchor names the
    # card it supersedes: both keep an older capture named so it is not reported as orphaned.
    for entry in head.get("retired", []):
        reference = entry.get("reference")
        if isinstance(reference, str):
            named.add(f"{NUMERICS}/{reference}")
            retired_card = loaded(read_head, f"{NUMERICS}/{reference}")
            if retired_card is None:
                findings.append(f"retired {entry['model']} version {entry['version']}: {reference} does not exist")
            elif entry.get("fingerprint") and fingerprint(retired_card) != entry["fingerprint"]:
                findings.append(
                    f"retired {entry['model']} version {entry['version']}: {reference} fingerprint "
                    f"{fingerprint(retired_card)}, but the registry says {entry['fingerprint']}"
                )
    for anchor in head["anchor"]:
        replaces = anchor.get("replaces")
        if isinstance(replaces, dict) and isinstance(replaces.get("reference"), str):
            named.add(f"{NUMERICS}/{replaces['reference']}")
    for path, status in sorted(changed.items()):
        if path.startswith(REFERENCES) and status != "D" and path not in named:
            findings.append(f"{path}: no registry entry names this reference; evidence for a report goes under {CAPTURES}")
        if path.startswith(CAPTURES) and status != "A":
            findings.append(f"{path}: a committed capture is evidence for a report and is never edited or deleted")
        # A reference this change adds must name the build its capture came from. The key a new
        # capture writes is the version note: an older reference has no build_source and is
        # grandfathered (docs/NUMERICS.md §5).
        if path.startswith(REFERENCES) and status == "A":
            data = read_head(path)
            if data is not None:
                try:
                    added = json.loads(data)
                except json.JSONDecodeError:
                    added = None
                problem = build_record_problem(added) if isinstance(added, dict) else None
                if problem:
                    findings.append(f"{path}: {problem}")
                # The floor is judged the same way: a card with the build record's version note
                # is new and must carry it; an older card, including one only moved, is
                # grandfathered by the absent key.
                server = (added or {}).get("server") if isinstance(added, dict) else None
                if isinstance(added, dict) and isinstance(server, dict) and BUILD_SOURCE in server:
                    model = (added.get("key") or {}).get("model", "")
                    issue = floor_problem(added, model, head)
                    if issue:
                        findings.append(f"{path}: {issue}")

    base_registry = base or {"config": [], "anchor": [], "retired": [], "prerequisites": {}}
    base_missing = missing_prerequisites(base) if base else list(PREREQUISITES)
    head_missing = missing_prerequisites(head)

    # A prerequisite turns on only with what it names in place.
    if head["prerequisites"]["anchor"] and not head["anchor"]:
        findings.append("prerequisites.anchor is true, but the registry holds no [[anchor]]")

    # Anchors are frozen: never edited, never removed, never their file rewritten.
    head_anchors = {anchor["model"]: anchor for anchor in head["anchor"]}
    base_anchor_models = {anchor["model"] for anchor in base_registry["anchor"]}
    for anchor in head["anchor"]:
        if anchor["model"] in base_anchor_models:
            continue
        # A new anchor's card must name its build too; the same version note grandfathers an anchor
        # whose reference predates the build record.
        path = f"{NUMERICS}/{anchor['reference']}"
        data = read_head(path)
        if data is None:
            continue
        try:
            anchored = json.loads(data)
        except json.JSONDecodeError:
            continue
        problem = build_record_problem(anchored) if isinstance(anchored, dict) else None
        if problem:
            findings.append(f"anchor {anchor['model']}: {problem}")
        server = (anchored or {}).get("server") if isinstance(anchored, dict) else None
        if isinstance(anchored, dict) and isinstance(server, dict) and BUILD_SOURCE in server:
            issue = floor_problem(anchored, anchor["model"], head)
            if issue:
                findings.append(f"anchor {anchor['model']}: {issue}")
    for anchor in base_registry["anchor"]:
        mine = head_anchors.get(anchor["model"])
        if mine is None:
            findings.append(f"anchor {anchor['model']}: removed; an anchor is frozen")
        elif mine != anchor:
            # An anchor is frozen, with one recorded exception: a replacement that names the card
            # it replaces and why, for an anchor that pinned a defect rather than arithmetic. The
            # old card stays in history, unchanged; the new card carries its own build and floor.
            replaces = mine.get("replaces")
            replaced = (
                isinstance(replaces, dict)
                and set(replaces) == {"reference", "fingerprint", "reason"}
                and replaces.get("reference") == anchor.get("reference")
                and replaces.get("fingerprint") == anchor.get("fingerprint")
                and isinstance(replaces.get("reason"), str)
                and bool(replaces["reason"].strip())
            )
            if not replaced:
                findings.append(f"anchor {anchor['model']}: edited; an anchor is frozen")
            else:
                new_path = f"{NUMERICS}/{mine['reference']}"
                data = read_head(new_path)
                if data is None:
                    findings.append(f"anchor {anchor['model']}: {new_path} does not exist")
                else:
                    try:
                        card = json.loads(data)
                    except json.JSONDecodeError as error:
                        card = None
                        findings.append(f"{new_path}: not a reference: {error}")
                    if isinstance(card, dict):
                        problem = build_record_problem(card)
                        if problem:
                            findings.append(f"anchor {anchor['model']}: {problem}")
                        server = (card.get("server") or {}) if isinstance(card.get("server"), dict) else {}
                        if BUILD_SOURCE in server:
                            issue = floor_problem(card, anchor["model"], head)
                            if issue:
                                findings.append(f"anchor {anchor['model']}: {issue}")
        path = f"{NUMERICS}/{anchor['reference']}"
        if path in changed:
            findings.append(f"{path}: the anchor's reference changed; an anchor is frozen")

    base_configs = {entry_key(entry): entry for entry in base_registry["config"]}
    head_configs = {entry_key(entry): entry for entry in head["config"]}
    retired = {entry_key(entry): entry for entry in head["retired"]}
    for key, entry in head_configs.items():
        name = key_text(dict(zip(KEY_FIELDS, key)))
        old = base_configs.get(key)
        if old is None:
            continue
        bumped = entry["version"] > old["version"]
        if entry["version"] < old["version"]:
            findings.append(f"{name}: version {old['version']} -> {entry['version']}; versions never go down")
            continue
        # A revert is a change: the head names the base's own arithmetic again, and the version
        # between them is retired with the arithmetic it had. Comparing against a base that
        # predates the undone version would otherwise read the revert as "bumped but unchanged".
        undone = retired.get(key)
        revert = (
            isinstance(undone, dict)
            and undone.get("version") == entry["version"] - 1
            and old["version"] < undone["version"]
            and entry.get("reference") == old.get("reference")
            and entry["fingerprint"] == old["fingerprint"]
            and undone.get("fingerprint") != entry["fingerprint"]
        )
        if bumped and entry["version"] != old["version"] + 1 and not revert:
            findings.append(f"{name}: version {old['version']} -> {entry['version']}; one declared change is one version")
        if entry["evidence"] != old["evidence"] and not bumped:
            findings.append(f"{name}: evidence {old['evidence']} -> {entry['evidence']} without a version bump")
        # An approval is part of the record, like a version: it is well formed, its report exists,
        # and it is neither removed nor added without the bump it records.
        approval = entry.get("approved")
        old_approval = old.get("approved")
        problem = approval_problem(entry)
        if problem:
            findings.append(f"{name}: {problem}")
        if old_approval is not None and approval is None:
            findings.append(
                f"{name}: removed the approval recorded at version {entry['version']}; an approval "
                "is part of the record, like a version"
            )
        if approval is not None and old_approval is None and not bumped:
            findings.append(
                f"{name}: approved at version {entry['version']} without a version bump; an "
                "approval records the bump it lets land"
            )
        if approval is not None and problem is None and read_head(approval["report"]) is None:
            findings.append(f"{name}: approved.report {approval['report']} does not exist")
        # The fingerprint is the bits: a reference moved to another path with the same bits is a
        # rename, which needs no bump and is no declared change. A first capture fills an
        # uncaptured entry and is not an arithmetic change; removing one is refused.
        old_reference, new_reference = old.get("reference"), entry.get("reference")
        if old_reference is None and new_reference is None:
            continue
        if old_reference is None:
            continue
        if new_reference is None:
            findings.append(f"{name}: removed its captured reference; a capture is not withdrawn")
            continue
        old_path, new_path = f"{NUMERICS}/{old_reference}", f"{NUMERICS}/{new_reference}"
        bits_changed = old["fingerprint"] != entry["fingerprint"]
        file_changed = new_path in changed or old_path in changed
        if not bumped:
            if bits_changed or file_changed:
                old_reference = loaded(read_base, old_path)
                new_reference = loaded(read_head, new_path)
                if old_reference is None or new_reference is None or not extends(old_reference, new_reference):
                    findings.append(
                        f"{name}: undeclared change: the reference's bits changed at version "
                        f"{entry['version']}; bump the version and follow docs/NUMERICS.md §2"
                    )
            continue
        if not bits_changed and not revert:
            findings.append(
                f"{name}: declared non-change: version {entry['version']} names the same bits as "
                f"version {old['version']}; a version is bumped only by a change"
            )
        gating_missing = [
            prerequisite
            for prerequisite in GATING_PREREQUISITES
            if prerequisite in base_missing or prerequisite in head_missing
        ]
        # Coverage and monitors are per model, like the anchor (docs/NUMERICS.md §5): this bump is
        # judged against its own model's rows, not against the aggregate flags. A model with no row
        # has that prerequisite missing, and a row that does not pass is not usable. The approval
        # path replaces these per-model problems (and the global declaration check) for one bump;
        # the declared change, the entry's own report and the model's anchor below still apply.
        per_model = model_evidence_problems(head, read_head, entry["model"])
        approval_ok = (
            approval is not None
            and problem is None
            and read_head(approval["report"]) is not None
        )
        if gating_missing and not approval_ok:
            findings.append(
                f"{name}: version {entry['version']} declares an arithmetic change, but "
                f"prerequisites {gating_missing} are not in place "
                "(docs/NUMERICS.md §5); until they are, the gate is bit-exact, or record the "
                "owner's approval as decision, date and report beside the entry"
            )
        if per_model and not approval_ok:
            findings.append(
                f"{name}: version {entry['version']} declares an arithmetic change, but "
                f"{entry['model']!r} has no passing per-model evidence: "
                + "; ".join(per_model)
                + " (docs/NUMERICS.md §5); add the model's [[coverage]] and [[monitors]] rows "
                "with their committed evidence, or record the owner's approval as decision, "
                "date and report beside the entry"
            )
        if (gating_missing or per_model) and approval_ok and changed.get(approval["report"]) not in ("A", "M"):
            findings.append(
                f"{name}: version {entry['version']} would land under the approval recorded at "
                f"version {old['version']}, but its report {approval['report']} is not new or "
                "changed in this change; an approval names the report that justifies its bump"
            )
        # The anchor is per model (docs/NUMERICS.md §4): this entry is judged against its own
        # model's frozen anchor and the teacher-forced capture made from it. A text reference has
        # no token ids to force, so it can neither be anchored nor bump; that is DeepSeek's
        # exemption, and it is an exemption from participating, not a waiver that lets it change.
        anchor = anchor_for(head, entry["model"])
        if entry["evidence"] in ANCHOR_EXEMPT_EVIDENCE:
            findings.append(
                f"{name}: version {entry['version']} declares an arithmetic change on "
                f"{entry['evidence']!r} evidence, but a text reference holds no token ids to "
                "teacher-force, so no quality comparison can justify it (docs/NUMERICS.md §2, §5)"
            )
        elif anchor is None:
            findings.append(
                f"{name}: version {entry['version']} declares an arithmetic change, but the model "
                f"{entry['model']!r} has no [[anchor]] in the registry; add the model's frozen "
                "anchor and a teacher-forced comparison first (docs/NUMERICS.md §4, §5)"
            )
        elif anchor.get("evidence") in ANCHOR_EXEMPT_EVIDENCE:
            findings.append(
                f"{name}: version {entry['version']} declares an arithmetic change, but the anchor "
                f"for {entry['model']!r} holds {anchor['evidence']!r} evidence, which cannot be "
                "teacher-forced (docs/NUMERICS.md §4)"
            )
        elif forced is not None and anchor["fingerprint"] not in forced.get(entry["model"], set()):
            findings.append(
                f"{name}: version {entry['version']} declares an arithmetic change, but no committed "
                f"capture under {CAPTURES} is teacher-forced from {entry['model']!r}'s anchor "
                f"{anchor['fingerprint'][:16]}; run scripts/numerics.py compare and commit its "
                "evidence first (docs/NUMERICS.md §4)"
            )
        report = entry.get("report")
        if not report:
            findings.append(f"{name}: version {entry['version']} names no report")
        elif read_head(report) is not None and changed.get(report) != "A":
            findings.append(f"{name}: report {report} is not new in this change; each version writes its own")
    for key, old in base_configs.items():
        if key in head_configs:
            continue
        name = key_text(dict(zip(KEY_FIELDS, key)))
        gone = retired.get(key)
        if gone is None:
            findings.append(f"{name}: removed without a [[retired]] entry")
        elif gone["version"] != old["version"]:
            findings.append(f"{name}: retired at version {gone['version']}, but it served {old['version']}")

    # An arm profile's defaults select the arithmetic its model serves, and the profile names that
    # model. A setting the serving side does not list as neutral (at both ends) changes only with
    # every reference of that model recaptured in the same change; the rules above then judge the
    # recaptured bits. The neutral list is an input this check trusts, so it is judged too: a field
    # that becomes neutral needs a new report that names it.
    profile_paths = sorted(path for path in changed if is_profile(path))
    neutral: set[str] = set()
    if profile_paths or CLASSIFICATION in changed:
        head_neutral = neutral_fields(read_head(CLASSIFICATION))
        base_neutral = neutral_fields(read_base(CLASSIFICATION))
        if head_neutral is None:
            findings.append(f"{CLASSIFICATION}: no NEUTRAL_FIELDS list, so no profile change can be classified")
            profile_paths = []
        else:
            neutral = head_neutral if base_neutral is None else head_neutral & base_neutral
            # The change that introduces the list is reviewed as the policy; after it, a field joins
            # only with its evidence.
            joined = set() if base_neutral is None else head_neutral - base_neutral
            reports = [path for path, status in changed.items() if status == "A" and path.startswith("docs/numerics/")]
            for field in sorted(joined):
                if not any(f"`{field}`".encode() in (read_head(path) or b"") for path in reports):
                    findings.append(
                        f"{CLASSIFICATION}: {field} becomes neutral, which lets a profile change it with no "
                        "recapture; add a report under docs/numerics/ that names it and the invariance gate "
                        "that holds it (docs/NUMERICS.md §5)"
                    )
    settings = {}
    for path in profile_paths:
        try:
            settings[path] = (profile_settings(read_base(path), path), profile_settings(read_head(path), path))
        except NumericsFailure as error:
            findings.append(str(error))
    # A profile moved to another path with every setting kept is a rename, which changes nothing.
    for gone in [path for path in settings if changed[path] == "D"]:
        twin = next((path for path in settings if changed[path] == "A" and settings[path][1] == settings[gone][0]), None)
        if twin is not None:
            del settings[gone], settings[twin]
    for path, (before, after) in settings.items():
        if after and not isinstance(after.get("model"), str):
            findings.append(
                f"{path}: names no model, so no registered reference can be held to it; add "
                '`model = "<registry id>"` beside its family and weights (docs/arm-reference.md §3)'
            )
            continue
        models = sorted({side["model"] for side in (before, after) if isinstance(side.get("model"), str)})
        if not models:
            findings.append(f"{path}: removed, and it named no model at the base to hold to its references")
            continue
        moved = sorted(
            setting
            for setting in before.keys() | after.keys()
            if setting not in neutral and before.get(setting) != after.get(setting)
        )
        if not moved:
            continue
        stale = [
            f"{NUMERICS}/{entry['reference']}"
            for entry in head["config"]
            if entry["model"] in models
            and "reference" in entry
            and changed.get(f"{NUMERICS}/{entry['reference']}") not in ("A", "M")
        ]
        if stale:
            what = ", ".join(moved) if before and after else "adding the profile" if after else "removing the profile"
            findings.append(
                f"{path}: {what} can change {' and '.join(models)}'s arithmetic, but "
                f"{', '.join(stale)} {'was' if len(stale) == 1 else 'were'} not recaptured; recapture every "
                "configuration of the model on the changed profile (docs/NUMERICS.md §5)"
            )
    return findings


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=False, check=check)


def has_commit(revision: str) -> bool:
    return git("cat-file", "-e", f"{revision}^{{commit}}", check=False).returncode == 0


def ensure_commit(revision: str) -> None:
    if has_commit(revision):
        return
    for fetch in (
        ["fetch", "--no-tags", "--depth=200", "origin", revision],
        ["fetch", "--no-tags", "--unshallow", "origin"],
    ):
        git(*fetch, check=False)
        if has_commit(revision):
            return
    raise NumericsFailure(
        f"the base {revision} is not in this clone and could not be fetched; without it the "
        "declaration check cannot see what changed (pass --base REV)"
    )


def resolve_base(argument: str | None) -> str:
    """The commit this change is measured against: the MR's diff base, the push's previous tip, or
    the merge base with the default branch."""
    if argument:
        ensure_commit(argument)
        return argument
    for variable in ("CI_MERGE_REQUEST_DIFF_BASE_SHA", "CI_COMMIT_BEFORE_SHA"):
        value = os.environ.get(variable, "")
        if value and value != ZERO_SHA:
            ensure_commit(value)
            return value
    branch = os.environ.get("CI_DEFAULT_BRANCH", "main")
    remote = f"origin/{branch}"
    if not has_commit(remote):
        git("fetch", "--no-tags", "--depth=200", "origin", f"{branch}:refs/remotes/{remote}", check=False)
    merge_base = git("merge-base", "HEAD", remote, check=False)
    if merge_base.returncode != 0:
        raise NumericsFailure(f"no merge base between HEAD and {remote}; pass --base REV")
    return merge_base.stdout.decode().strip()


def read_at(revision: str | None) -> Callable[[str], bytes | None]:
    def read(path: str) -> bytes | None:
        if revision is None:
            file = REPO / path
            return file.read_bytes() if file.is_file() else None
        shown = git("show", f"{revision}:{path}", check=False)
        return shown.stdout if shown.returncode == 0 else None

    return read


def coverage_row_result(registry: dict, row: dict, read: Callable[[str], bytes | None]) -> dict:
    """Judge one model's [[coverage]] row: its dump against its layout and the shared bar."""
    path = row["dump"]
    data = read(path)
    if data is None:
        raise NumericsFailure(
            f"its [[coverage]] dump {path} does not exist; a row is committed with its dump "
            "(docs/NUMERICS.md §5)"
        )
    try:
        dump = json.loads(data)
    except json.JSONDecodeError as error:
        raise NumericsFailure(f"its [[coverage]] dump {path} is not JSON ({error})") from error
    if not isinstance(dump, dict):
        raise NumericsFailure(f"its [[coverage]] dump {path} is not a JSON object")
    model = dump.get("model")
    if model != row["model"]:
        raise NumericsFailure(
            f"its [[coverage]] dump {path} names {model!r}, but the row is for {row['model']!r}"
        )
    layout, layout_origin = find_layout(row["model"], None)
    return judge_coverage(
        dump,
        layout,
        layout_origin,
        row["min_inputs"],
        registry["prerequisites"].get("coverage_slots"),
    )


def coverage_row_findings(registry: dict, read: Callable[[str], bytes | None]) -> tuple[dict[str, dict], list[str]]:
    """Judge every committed [[coverage]] row: the passing ones by model, and the findings."""
    results, findings = {}, []
    for row in registry.get("coverage", []):
        try:
            result = coverage_row_result(registry, row, read)
        except NumericsFailure as error:
            findings.append(f"[[coverage]] {row['model']!r}: {error}")
            continue
        if result["passes"]:
            results[row["model"]] = result
        else:
            findings.append(
                f"[[coverage]] {row['model']!r}: achieved {result['coverage']:.2%} of "
                f"{result['experts']} routed (layer, expert) slots, below the declared "
                f"{result['bar']:.2%} bar"
            )
    return results, findings


def coverage_aggregate_findings(registry: dict, results: dict[str, dict]) -> list[str]:
    """The aggregate `coverage` flag's own claim: true only when every captured model passes.

    A false flag with a model lacking a row is consistent - the per-model bump check is what
    refuses that model - but a true flag that claims more than the rows hold is not
    (docs/NUMERICS.md §5).
    """
    if not registry["prerequisites"]["coverage"]:
        return []
    uncovered = [model for model in logprobs_models(registry) if model not in results]
    if not uncovered:
        return []
    return [
        "prerequisites.coverage is true, but these captured logprobs models have no passing "
        f"[[coverage]] row: {uncovered}; the flag reports the aggregate, so it cannot claim more "
        "than the registry holds"
    ]


def coverage_findings(registry: dict, read: Callable[[str], bytes | None]) -> list[str]:
    """The [[coverage]]-scoped check: every row is judged, and the aggregate flag reports them.

    A row is evidence, so it is judged whether or not the aggregate flag is true.
    """
    results, findings = coverage_row_findings(registry, read)
    return findings + coverage_aggregate_findings(registry, results)


def monitor_row_result(row: dict, read: Callable[[str], bytes | None]) -> tuple[dict | None, list[str]]:
    """Judge one model's [[monitors]] row: two committed dumps at the row's bars."""
    label = f"[[monitors]] {row['model']!r}"
    missing = [path for path in (row["reference"], row["candidate"]) if read(path) is None]
    if missing:
        return None, [f"{label}: its dump does not exist: {', '.join(missing)}"]
    dumps = {}
    for path in (row["reference"], row["candidate"]):
        data = read(path)
        assert data is not None
        try:
            document = json.loads(data)
        except json.JSONDecodeError as error:
            return None, [f"{label}: {path} is not JSON ({error})"]
        try:
            dumps[path] = check_monitor_dump(document, path)
        except NumericsFailure as error:
            return None, [f"{label}: {error}"]
        if dumps[path]["model"] != row["model"]:
            return None, [
                f"{label}: {path} names {dumps[path]['model']!r}, but the row is for "
                f"{row['model']!r}"
            ]
    try:
        result = judge_monitors(
            dumps[row["reference"]],
            dumps[row["candidate"]],
            float(row["layer_bar"]),
            float(row["expert_bar"]),
        )
    except NumericsFailure as error:
        return None, [f"{label}: {error}"]
    return result, [f"{label}: {failure}" for failure in result["failures"]]


def monitor_row_results(
    registry: dict, read: Callable[[str], bytes | None]
) -> tuple[dict[str, dict], list[str]]:
    """Judge every committed [[monitors]] row: the passing ones by model, and the findings."""
    results, findings = {}, []
    for row in registry.get("monitors", []):
        result, problems = monitor_row_result(row, read)
        findings += problems
        if result is not None and result["pass"]:
            results[row["model"]] = result
    return results, findings


def monitors_aggregate_findings(registry: dict, results: dict[str, dict]) -> list[str]:
    """The aggregate `monitors` flag's own claim: true only when every captured model passes."""
    if not registry["prerequisites"]["monitors"]:
        return []
    uncovered = [model for model in logprobs_models(registry) if model not in results]
    if not uncovered:
        return []
    return [
        "prerequisites.monitors is true, but these captured logprobs models have no passing "
        f"[[monitors]] row: {uncovered}; the flag reports the aggregate, so it cannot claim more "
        "than the registry holds"
    ]


def monitors_findings(registry: dict, read: Callable[[str], bytes | None]) -> list[str]:
    """The [[monitors]]-scoped check: every row is judged, and the aggregate flag reports them."""
    results, findings = monitor_row_results(registry, read)
    return findings + monitors_aggregate_findings(registry, results)


def model_evidence_problems(registry: dict, read: Callable[[str], bytes | None], model: str) -> list[str]:
    """Why one model's own coverage and monitors prerequisites are not in place.

    A bump is judged against the bumping entry's model (docs/NUMERICS.md §5), not against the
    aggregate flags: a model with no row has that prerequisite missing, and a row whose dump is
    absent, names another model, or misses its bar is not usable. An `approved` record replaces
    these problems for one bump; nothing else does.
    """
    problems = []
    coverage = coverage_for(registry, model)
    if coverage is None:
        problems.append(f"the model {model!r} has no [[coverage]] row")
    else:
        try:
            result = coverage_row_result(registry, coverage, read)
        except NumericsFailure as error:
            problems.append(f"the model {model!r}'s [[coverage]] row is unusable: {error}")
        else:
            if not result["passes"]:
                problems.append(
                    f"the model {model!r}'s [[coverage]] dump achieves {result['coverage']:.2%}, "
                    f"below the declared {result['bar']:.2%} bar"
                )
    monitors = monitors_for(registry, model)
    if monitors is None:
        problems.append(f"the model {model!r} has no [[monitors]] row")
    else:
        _, row_findings = monitor_row_result(monitors, read)
        problems += row_findings
    return problems


def rust_registry_check(path: Path) -> tuple[bool, str]:
    """Parse `path` with the Rust registry parser a serving binary uses.

    The Python gate validated a registry the Rust parser refused (version 2's `[[retired]]`
    reference), so the gate runs the real parser. Returns (ran, problem); a deliberate
    SPARKFLASH_SKIP_RUST_REGISTRY=1 reports as ran with a note.
    """
    if os.environ.get("SPARKFLASH_SKIP_RUST_REGISTRY") == "1":
        return True, "skipped by SPARKFLASH_SKIP_RUST_REGISTRY=1"
    if not (REPO / "Cargo.toml").is_file():
        # The declaration tests build throwaway git trees that carry only numerics files.
        return True, "skipped: the tree has no Cargo.toml"
    cargo = shutil.which("cargo")
    if cargo is None:
        return False, "cargo is not on PATH, so the Rust registry parse did not run"
    command = [
        cargo,
        "run",
        "-q",
        "-p",
        "sparkflash-model",
        "--example",
        "validate_registry",
        "--",
        str(path),
    ]
    try:
        run = subprocess.run(
            command, cwd=REPO, capture_output=True, text=True, timeout=900, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"the Rust registry parse did not finish: {error}"
    if run.returncode != 0:
        lines = (run.stderr or run.stdout).strip().splitlines()[-3:]
        return True, "the Rust registry parse refused the registry: " + " / ".join(lines)
    return True, "ok"


def run_ci(base_argument: str | None) -> int:
    base = resolve_base(base_argument)
    read_base, read_head = read_at(base), read_at(None)
    base_text = read_base(REGISTRY)
    base_registry = parse_registry(base_text.decode(), f"{base}:{REGISTRY}") if base_text else None
    head_text = read_head(REGISTRY)
    if head_text is None:
        raise NumericsFailure(f"{REGISTRY} does not exist")
    head_registry = parse_registry(head_text.decode(), REGISTRY, require_bar=True)
    # The binary parses the same text with the Rust registry schema; a gate that only reads TOML
    # would accept a registry the next release aborts on.
    parsed, rust_state = rust_registry_check(REPO / REGISTRY)
    # The working tree, not HEAD: a local run checks what is about to be committed too.
    paths = (NUMERICS, "docs/numerics", f":(glob){PROFILE_GLOB}", CLASSIFICATION)
    diff = git("diff", "--name-status", "--no-renames", base, "--", *paths).stdout.decode()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *paths).stdout.decode()
    changed = {}
    for line in diff.splitlines():
        status, path = line.split("\t", 1)
        changed[path] = status[0]
    for path in untracked.splitlines():
        changed[path] = "A"
    capture_root = REPO / NUMERICS / "captures"
    capture_paths = (
        sorted(str(path.relative_to(REPO)) for path in capture_root.rglob("*.json"))
        if capture_root.is_dir()
        else []
    )
    forced = forced_captures(capture_paths, read_head)
    findings = declaration_findings(
        base_registry, head_registry, read_base, read_head, changed, forced
    )
    if not parsed or (rust_state != "ok" and "skipped" not in rust_state):
        findings.append(f"the Rust registry parse: {rust_state}")
    anchor_checks, anchor_coverage = anchor_findings(head_registry, read_head, capture_paths)
    findings += anchor_checks
    coverage_results, coverage_checks = coverage_row_findings(head_registry, read_head)
    findings += coverage_checks
    findings += coverage_aggregate_findings(head_registry, coverage_results)
    monitor_results, monitor_checks = monitor_row_results(head_registry, read_head)
    findings += monitor_checks
    findings += monitors_aggregate_findings(head_registry, monitor_results)
    prerequisites = head_registry["prerequisites"]
    configs, anchors = len(head_registry["config"]), len(head_registry["anchor"])
    missing = missing_prerequisites(head_registry)
    print(f"numerics ci: Rust registry parse: {rust_state}")
    print(
        f"numerics ci: base {base[:12]}, {configs} configuration(s), {anchors} anchor(s), "
        f"{len(head_registry['coverage'])} coverage row(s), "
        f"{len(head_registry['monitors'])} monitors row(s), "
        f"{len(changed)} changed path(s); aggregate prerequisites missing: {missing or 'none'}"
    )
    bar = coverage_bar(prerequisites.get("coverage_slots"), REGISTRY)
    for row in head_registry["coverage"]:
        result = coverage_results.get(row["model"])
        state = (
            f"{result['covered_experts']} of {result['experts']} routed (layer, expert) slots "
            f"({result['coverage']:.2%}), bar {bar:.2%}, {result['inputs']} input(s) >= "
            f"{row['min_inputs']}"
            if result is not None
            else f"FAILED, bar {bar:.2%}, min_inputs {row['min_inputs']}"
        )
        print(f"numerics ci: coverage row {row['model']}: {row['dump']} {state}")
    for row in head_registry["monitors"]:
        result = monitor_results.get(row["model"])
        print(
            f"numerics ci: monitors row {row['model']}: {row['reference']} vs "
            f"{row['candidate']} at layer bar {float(row['layer_bar']):.3e} and expert bar "
            f"{float(row['expert_bar']):.3e}"
        )
        if result is not None:
            routing = result["routing"]
            maxima = result["maxima"]
            expert_maximum = (
                "no shared expert"
                if maxima["expert"] is None
                else f"{maxima['expert']:.3e}"
            )
            print(
                f"numerics ci: monitors routing {row['model']}: {routing['layers_differing']} of "
                f"{len(result['layers'])} layer(s) differ, {routing['experts_unshared']} expert(s) "
                f"unshared; achieved maxima layer {maxima['layer']:.3e}, expert {expert_maximum}"
            )
    for name, results in (("coverage", coverage_results), ("monitors", monitor_results)):
        uncovered = [model for model in logprobs_models(head_registry) if model not in results]
        state = (
            "every captured logprobs model has a passing row"
            if not uncovered
            else "no passing row for " + ", ".join(uncovered)
        )
        print(
            f"numerics ci: aggregate {name} prerequisite: "
            f"{'true' if prerequisites[name] else 'false'}; {state}"
        )
    print(
        "numerics ci: anchor coverage: "
        + (
            "covered " + ", ".join(sorted(anchor_coverage["covered"]))
            if anchor_coverage["covered"]
            else "no logprobs model is anchored"
        )
        + (
            "; exempt (text evidence) " + ", ".join(anchor_coverage["exempt"])
            if anchor_coverage["exempt"]
            else ""
        )
        + (
            "; not covered " + ", ".join(sorted(anchor_coverage["uncovered"]))
            if anchor_coverage["uncovered"]
            else ""
        )
    )
    uncaptured = [entry for entry in head_registry["config"] if "reference" not in entry]
    for entry in uncaptured:
        print(f"numerics ci: uncaptured {key_text(dict(zip(KEY_FIELDS, entry_key(entry))))}: registered, no reference yet")
    for finding in findings:
        print(f"FAIL {finding}")
    print("numerics ci: " + ("FAIL" if findings else "PASS"))
    return 1 if findings else 0


# --- the teacher-forced comparison ------------------------------------------------------------


def position_scores(anchor_lp: float, anchor_top: list, candidate_lp: float, candidate_top: list) -> dict:
    a_top, c_top = dict(map(tuple, anchor_top)), dict(map(tuple, candidate_top))
    a_first = min(a_top, key=lambda token: (-a_top[token], token))
    c_first = min(c_top, key=lambda token: (-c_top[token], token))
    shared = set(a_top) & set(c_top)
    # KL over the partition {each shared top-k token, everything else}: coarser than the
    # vocabulary, so by the data-processing inequality it never exceeds the true KL.
    kl = 0.0
    p_tail, q_tail = 1.0, 1.0
    for token in shared:
        p, q = math.exp(a_top[token]), math.exp(c_top[token])
        p_tail -= p
        q_tail -= q
        if p > 0.0:
            kl += p * (a_top[token] - c_top[token])
    p_tail, q_tail = max(p_tail, 0.0), max(q_tail, TAIL_FLOOR)
    if p_tail > TAIL_FLOOR:
        kl += p_tail * math.log(p_tail / q_tail)
    return {
        "delta_nll": anchor_lp - candidate_lp,
        "agree": a_first == c_first,
        "kl_lower_bound": max(kl, 0.0),
    }


def summarize(scores: list[dict], anchor_nll: list[float]) -> dict:
    count = len(scores)
    deltas = sorted(abs(score["delta_nll"]) for score in scores)
    return {
        "tokens": count,
        "anchor_nll_mean": sum(anchor_nll) / count,
        "delta_nll_mean": sum(score["delta_nll"] for score in scores) / count,
        "abs_delta_nll_mean": sum(deltas) / count,
        "abs_delta_nll_p99": deltas[min(count - 1, math.ceil(0.99 * count) - 1)],
        "abs_delta_nll_max": deltas[-1],
        "top1_agreement": sum(score["agree"] for score in scores) / count,
        "kl_lower_bound_mean": sum(score["kl_lower_bound"] for score in scores) / count,
        "kl_lower_bound_max": max(score["kl_lower_bound"] for score in scores),
    }


def null_logprob(prompt: dict) -> int | None:
    """The first position whose logprob or top-k alternative is null (non-finite, see f32_bits)."""
    for position, (value, top) in enumerate(zip(prompt["logprobs"], prompt["top"])):
        if value is None or any(alternative is None for _, alternative in top):
            return position
    return None


def compare(anchor: dict, candidate: dict) -> dict:
    """Score a teacher-forced capture against the anchor whose tokens it was fed."""
    if anchor["evidence"] != "logprobs" or candidate["evidence"] != "logprobs":
        raise NumericsFailure("the comparison scores logprobs; a text reference has none")
    forced = candidate.get("forced_from") or {}
    if forced.get("fingerprint") != anchor["fingerprint"]:
        raise NumericsFailure(
            "the candidate was not forced from this anchor (its forced_from fingerprint is "
            f"{forced.get('fingerprint')}, the anchor's is {anchor['fingerprint']}); a free-running "
            "candidate diverges at its first near-tie, so capture it with --forced-from"
        )
    if len(anchor["prompts"]) != len(candidate["prompts"]):
        raise NumericsFailure(
            f"the anchor has {len(anchor['prompts'])} prompts, the candidate {len(candidate['prompts'])}"
        )
    scores, nll, prompts = [], [], []
    for a, c in zip(anchor["prompts"], candidate["prompts"]):
        if a["prompt_ids"] != c["prompt_ids"] or a["tokens"] != c["tokens"]:
            raise NumericsFailure(f"prompt {a['name']!r}: the candidate scored different tokens")
        for side, prompt in (("anchor", a), ("candidate", c)):
            position = null_logprob(prompt)
            if position is not None:
                raise NumericsFailure(
                    f"prompt {a['name']!r} position {position}: the {side} holds a null (non-finite) "
                    "logprob, which has no probability to score"
                )
        mine = [
            position_scores(a_lp, a_top, c_lp, c_top)
            for a_lp, a_top, c_lp, c_top in zip(a["logprobs"], a["top"], c["logprobs"], c["top"])
        ]
        if not mine:
            continue
        scores += mine
        nll += [-value for value in a["logprobs"]]
        prompts.append({"name": a["name"], **summarize(mine, [-value for value in a["logprobs"]])})
    if not scores:
        raise NumericsFailure("the anchor generated no tokens to score")
    return {
        "schema": COMPARE_SCHEMA,
        "anchor": {"key": anchor.get("key"), "fingerprint": anchor["fingerprint"]},
        "candidate": {
            "key": candidate.get("key"),
            "fingerprint": candidate["fingerprint"],
            "numerics": candidate.get("server", {}).get("numerics"),
        },
        "top_logprobs": candidate["settings"].get("top_logprobs"),
        "summary": summarize(scores, nll),
        "prompts": prompts,
    }


METRICS = (
    ("tokens", "tokens scored", "{:d}"),
    ("anchor_nll_mean", "anchor NLL / token", "{:.4f}"),
    ("delta_nll_mean", "mean NLL delta", "{:+.3e}"),
    ("abs_delta_nll_mean", "mean absolute NLL delta", "{:.3e}"),
    ("abs_delta_nll_p99", "p99 absolute NLL delta", "{:.3e}"),
    ("abs_delta_nll_max", "max absolute NLL delta", "{:.3e}"),
    ("top1_agreement", "top-1 agreement", "{:.4%}"),
    ("kl_lower_bound_mean", "mean KL lower bound", "{:.3e}"),
    ("kl_lower_bound_max", "max KL lower bound", "{:.3e}"),
)
# The metrics a floor bounds, and which way is better.
BARRED = {"abs_delta_nll_mean": -1, "abs_delta_nll_p99": -1, "kl_lower_bound_mean": -1, "top1_agreement": 1}


def render_comparison(result: dict, floor: dict | None) -> str:
    def describe(side: dict) -> str:
        key = side.get("key") or {}
        return key_text(key) if key else "?"

    lines = [
        f"Anchor: `{describe(result['anchor'])}` (`{result['anchor']['fingerprint'][:16]}`)",
        f"Candidate: `{describe(result['candidate'])}` (`{result['candidate']['fingerprint'][:16]}`),"
        f" teacher-forced on the anchor's tokens, top-{result['top_logprobs']} logprobs",
        "",
    ]
    summary = result["summary"]
    if floor is None:
        lines += ["| metric | candidate |", "| :-- | --: |"]
        lines += [f"| {label} | {fmt.format(summary[field])} |" for field, label, fmt in METRICS]
    else:
        lines += [
            f"Floor: `{describe(floor['candidate'])}` against the same anchor",
            "",
            "| metric | candidate | floor | within the floor |",
            "| :-- | --: | --: | :-- |",
        ]
        for field, label, fmt in METRICS:
            mine, theirs = summary[field], floor["summary"][field]
            verdict = ""
            if field in BARRED:
                verdict = "yes" if (mine - theirs) * BARRED[field] >= 0 else "**no**"
            lines.append(f"| {label} | {fmt.format(mine)} | {fmt.format(theirs)} | {verdict} |")
    lines += [
        "",
        "The KL column is a lower bound: it partitions the vocabulary into the tokens both arms list"
        " in their top-k and one bucket for the rest.",
    ]
    return "\n".join(lines) + "\n"


# --- coverage and monitors --------------------------------------------------------------------


def load_json(path: Path, origin: str | None = None) -> dict:
    where = origin or str(path)
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise NumericsFailure(f"{where}: {error}") from error
    if not isinstance(document, dict):
        raise NumericsFailure(f"{where}: the document is not a JSON object")
    return document


def is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def layout_experts(layout: dict, origin: str) -> list[int]:
    """The routed-expert count of every decoder layer, from a versioned expert layout.

    The spans are contiguous and start at zero, so the layout cannot skip a layer or overlap one.
    A dense layer is a span with zero experts; it is still a layer the dump must account for.
    """
    if layout.get("schema") != LAYOUT_SCHEMA:
        raise NumericsFailure(
            f"{origin}: schema {layout.get('schema')!r} is not {LAYOUT_SCHEMA}"
        )
    model = layout.get("model")
    if not isinstance(model, str) or not model:
        raise NumericsFailure(f"{origin}: the layout names no model")
    spans = layout.get("layers")
    if not isinstance(spans, list) or not spans:
        raise NumericsFailure(f"{origin}: the layout holds no layer spans")
    counts: list[int] = []
    for span in spans:
        if not isinstance(span, dict):
            raise NumericsFailure(f"{origin}: a layer span is not an object")
        first, count, experts = span.get("first"), span.get("count"), span.get("experts")
        if not all(is_integer(value) for value in (first, count, experts)):
            raise NumericsFailure(f"{origin}: a layer span needs integer first, count and experts")
        if first != len(counts):
            raise NumericsFailure(
                f"{origin}: the layer spans must start at 0 and be contiguous; a span starts at "
                f"{first}, but {len(counts)} layers are already described"
            )
        if count <= 0:
            raise NumericsFailure(f"{origin}: a layer span has count {count}; it must be positive")
        if experts < 0:
            raise NumericsFailure(f"{origin}: a layer span has {experts} experts; it cannot be negative")
        counts.extend([experts] * count)
    return counts


def find_layout(model: str, override: Path | None) -> tuple[dict, str]:
    if override is not None:
        layout = load_json(override)
        if layout.get("model") != model:
            raise NumericsFailure(
                f"{override}: the layout names {layout.get('model')!r}, but the dump names {model!r}"
            )
        return layout, str(override)
    root = REPO / LAYOUTS
    if not root.is_dir():
        raise NumericsFailure(f"{root}: no expert layouts are committed")
    for path in sorted(root.glob("*.json")):
        layout = load_json(path)
        if layout.get("model") == model:
            return layout, str(path.relative_to(REPO))
    raise NumericsFailure(
        f"no expert layout under {LAYOUTS} names model {model!r}; add one before judging coverage "
        "(docs/NUMERICS.md §5)"
    )


def coverage_inspect(dump: dict, layout: dict, layout_origin: str, min_inputs: int = 0) -> dict:
    """Inspect one coverage dump's shape and report what it has not reached.

    This is the judge without the gate: every structural refusal is the same, but a zero hit count
    becomes a gap in the report instead of a failure. `judge_coverage` adds the declared bar's
    verdict; `coverage-gaps` and `coverage-sweep` use the report to plan the next inputs.
    """
    if dump.get("schema") != COVERAGE_SCHEMA:
        raise NumericsFailure(f"coverage dump: schema {dump.get('schema')!r} is not {COVERAGE_SCHEMA}")
    if not isinstance(dump.get("model"), str) or not dump["model"]:
        raise NumericsFailure("coverage dump: the dump names no model")
    if min_inputs < 0:
        raise NumericsFailure(f"--min-inputs {min_inputs}: it cannot be negative")
    expected = layout_experts(layout, layout_origin)
    if dump["model"] != layout.get("model"):
        raise NumericsFailure(
            f"coverage dump: the dump names {dump['model']!r}, but {layout_origin} names "
            f"{layout.get('model')!r}"
        )
    inputs = dump.get("inputs")
    if not isinstance(inputs, list):
        raise NumericsFailure("coverage dump: inputs is not a list")
    if not inputs and min_inputs > 0:
        raise NumericsFailure("coverage dump: it holds no inputs")
    names = set()
    for entry in inputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"]:
            raise NumericsFailure("coverage dump: every input needs a non-empty name")
        if entry["name"] in names:
            raise NumericsFailure(f"coverage dump: input {entry['name']!r} appears more than once")
        names.add(entry["name"])
        if "tokens" in entry and (not is_integer(entry["tokens"]) or entry["tokens"] <= 0):
            raise NumericsFailure(
                f"coverage dump: input {entry['name']!r} has tokens {entry['tokens']!r}; it must be positive"
            )
    if len(inputs) < min_inputs:
        raise NumericsFailure(
            f"coverage dump: it names {len(inputs)} input(s), but the declared coverage suite has "
            f"{min_inputs}; a prerequisite run does not get to shorten the suite"
        )
    layers = dump.get("layers")
    if not isinstance(layers, list):
        raise NumericsFailure("coverage dump: layers is not a list")
    by_layer: dict[int, list] = {}
    for entry in layers:
        if not isinstance(entry, dict) or not is_integer(entry.get("layer")):
            raise NumericsFailure("coverage dump: every layer entry needs an integer layer")
        number = entry["layer"]
        if number < 0 or number >= len(expected):
            raise NumericsFailure(
                f"coverage dump: layer {number} is outside the layout's {len(expected)} layers"
            )
        if number in by_layer:
            raise NumericsFailure(f"coverage dump: layer {number} appears more than once")
        hits = entry.get("hits")
        if not isinstance(hits, list) or not all(is_integer(hit) and hit >= 0 for hit in hits):
            raise NumericsFailure(f"coverage dump: layer {number} needs a list of non-negative hit counts")
        by_layer[number] = hits
    covered = []
    total_experts = 0
    covered_experts = 0
    missing_slots: list[list[int]] = []
    missing_by_layer: dict[int, list[int]] = {}
    for number, experts in enumerate(expected):
        if number not in by_layer:
            raise NumericsFailure(
                f"coverage dump: layer {number} is missing; every layer must be accounted for, "
                "including a dense layer with an empty hit list"
            )
        hits = by_layer[number]
        if experts == 0:
            if hits:
                raise NumericsFailure(
                    f"coverage dump: layer {number} is dense (no routed experts), but it holds "
                    f"{len(hits)} hit count(s)"
                )
            covered.append(
                {
                    "layer": number,
                    "experts": 0,
                    "hits": 0,
                    "min": None,
                    "max": None,
                    "missing": [],
                    "counts": [],
                }
            )
            continue
        if len(hits) != experts:
            raise NumericsFailure(
                f"coverage dump: layer {number} has {len(hits)} hit count(s) for {experts} routed "
                "experts; the count must match the layout exactly"
            )
        missing = [expert for expert, hit in enumerate(hits) if hit == 0]
        missing_slots.extend([number, expert] for expert in missing)
        if missing:
            missing_by_layer[number] = missing
        total_experts += experts
        covered_experts += experts - len(missing)
        covered.append(
            {
                "layer": number,
                "experts": experts,
                "hits": sum(hits),
                "min": min(hits),
                "max": max(hits),
                "missing": missing,
                "counts": hits,
            }
        )
    return {
        "schema": COVERAGE_SCHEMA,
        "model": dump["model"],
        "inputs": len(inputs),
        "input_names": [entry["name"] for entry in inputs],
        "layers": covered,
        "experts": total_experts,
        "covered_experts": covered_experts,
        "missing": missing_slots,
        "missing_by_layer": missing_by_layer,
        "layout": layout_origin,
    }


def judge_coverage(
    dump: dict, layout: dict, layout_origin: str, min_inputs: int, bar: object
) -> dict:
    """Judge one coverage dump against a model's expert layout and the declared bar.

    Every decoder layer must appear once, with one hit count per routed expert; a dense layer has
    an empty hit list, and the dump must name at least the declared number of inputs. The verdict
    is the achieved fraction of routed (layer, expert) slots against the registry's bar, not an
    unstated demand for every slot: a bar of 1.0 still means every slot, and a lower bar leaves
    the gaps named beside the count.
    """
    if min_inputs < 1:
        raise NumericsFailure(f"--min-inputs {min_inputs}: a coverage run needs at least one input")
    fraction = coverage_bar(bar, "coverage bar")
    result = coverage_inspect(dump, layout, layout_origin, min_inputs)
    total = result["experts"]
    achieved = (result["covered_experts"] / total) if total else 1.0
    result["bar"] = fraction
    result["coverage"] = achieved
    result["passes"] = achieved >= fraction
    return result


def missing_experts_text(missing: list[int], limit: int = 16) -> str:
    """A layer's missing expert ids, shortened the same way everywhere they are printed."""
    if not missing:
        return "none"
    shown = ", ".join(str(expert) for expert in missing[:limit])
    more = "" if len(missing) <= limit else f" and {len(missing) - limit} more"
    return f"{shown}{more}"


def render_coverage(result: dict) -> str:
    achieved = result["coverage"]
    bar = result["bar"]
    lines = [
        f"coverage: {result['model']}, {len(result['layers'])} layer(s), "
        f"{result['experts']} routed expert(s), {result['inputs']} input(s); layout {result['layout']}",
        f"achieved: {result['covered_experts']} of {result['experts']} routed (layer, expert) "
        f"slots = {achieved:.2%}; declared bar {bar:.2%}",
        "",
        "| layer | routed experts | hits | min | max | missing experts |",
        "| --: | --: | --: | --: | --: | :-- |",
    ]
    for layer in result["layers"]:
        if layer["experts"] == 0:
            lines.append(f"| {layer['layer']} | 0 (dense) | - | - | - | - |")
        else:
            lines.append(
                f"| {layer['layer']} | {layer['experts']} | {layer['hits']} | {layer['min']} | "
                f"{layer['max']} | {missing_experts_text(layer['missing'])} |"
            )
    unreached = len(result["missing"])
    if result["passes"]:
        verdict = (
            f"PASS coverage: achieved {achieved:.2%} of routed (layer, expert) slots over "
            f"{result['inputs']} input(s), at or above the declared {bar:.2%} bar"
        )
    else:
        verdict = (
            f"FAIL coverage: achieved {achieved:.2%} of routed (layer, expert) slots over "
            f"{result['inputs']} input(s), below the declared {bar:.2%} bar"
        )
    if unreached:
        verdict += f"; {unreached} of {result['experts']} slot(s) unreached"
    lines += ["", verdict + "."]
    return "\n".join(lines) + "\n"


# --- the growing coverage suite ---------------------------------------------------------------


def coverage_gaps_document(result: dict, dump_origin: str) -> dict:
    """The judge's missing-expert report, as a machine-readable document."""
    total = result["experts"]
    covered = result["covered_experts"]
    return {
        "schema": COVERAGE_GAPS_SCHEMA,
        "model": result["model"],
        "layout": result["layout"],
        "dump": dump_origin,
        "slots": total,
        "covered": covered,
        "missing": len(result["missing"]),
        "coverage": (covered / total) if total else 1.0,
        "complete": not result["missing"],
        "layers": [
            {"layer": layer["layer"], "experts": layer["experts"], "missing": layer["missing"]}
            for layer in result["layers"]
            if layer["experts"] and layer["missing"]
        ],
    }


def render_coverage_gaps(result: dict) -> str:
    total = result["experts"]
    covered = result["covered_experts"]
    missing = len(result["missing"])
    ratio = (covered / total) if total else 1.0
    lines = [
        f"coverage gaps: {result['model']}, {len(result['layers'])} layer(s), {total} routed "
        f"slot(s): {missing} missing ({ratio:.2%} covered); layout {result['layout']}",
        "",
    ]
    if not missing:
        lines.append("PASS coverage: every routed expert of every layer has a positive hit count.")
        return "\n".join(lines) + "\n"
    lines += [
        "| layer | routed experts | missing | expert ids (first 16) |",
        "| --: | --: | --: | :-- |",
    ]
    for layer in result["layers"]:
        if not layer["experts"] or not layer["missing"]:
            continue
        shown = ", ".join(str(expert) for expert in layer["missing"][:16])
        more = "" if len(layer["missing"]) <= 16 else f" and {len(layer['missing']) - 16} more"
        lines.append(
            f"| {layer['layer']} | {layer['experts']} | {len(layer['missing'])} | {shown}{more} |"
        )
    lines += ["", "The full list is in --json; coverage-sweep uses the same inspection."]
    return "\n".join(lines) + "\n"


def _coverage_entry(raw: object, origin: str) -> dict:
    """One corpus or suite entry: a name and exactly one of prompt or messages."""
    if not isinstance(raw, dict):
        raise NumericsFailure(f"{origin}: a prompt is not an object")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise NumericsFailure(f"{origin}: every prompt needs a non-empty name")
    has_prompt, has_messages = "prompt" in raw, "messages" in raw
    if has_prompt == has_messages:
        raise NumericsFailure(f"{origin}: prompt {name!r} needs exactly one of prompt or messages")
    entry: dict = {"name": name}
    if has_prompt:
        if not isinstance(raw["prompt"], str) or not raw["prompt"].strip():
            raise NumericsFailure(f"{origin}: prompt {name!r} has no text")
        entry["prompt"] = raw["prompt"]
    else:
        messages = raw["messages"]
        if (
            not isinstance(messages, list)
            or not messages
            or not all(isinstance(message, dict) and isinstance(message.get("content"), str) for message in messages)
        ):
            raise NumericsFailure(
                f"{origin}: prompt {name!r} needs a non-empty messages list of role/content objects"
            )
        entry["messages"] = messages
    family = raw.get("family")
    if isinstance(family, str) and family:
        entry["family"] = family
    return entry


def coverage_prompt_digest(entry: dict) -> str:
    """The content identity of one corpus prompt, independent of its name."""
    if "prompt" in entry:
        text = entry["prompt"]
    else:
        text = json.dumps(entry["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_coverage_corpus(path: Path) -> tuple[list[dict], list[str]]:
    """A coverage corpus, with duplicate content dropped (first name wins)."""
    document = load_json(path)
    if document.get("schema") != COVERAGE_CORPUS_SCHEMA:
        raise NumericsFailure(
            f"{path}: schema {document.get('schema')!r} is not {COVERAGE_CORPUS_SCHEMA}"
        )
    raw = document.get("prompts")
    if not isinstance(raw, list) or not raw:
        raise NumericsFailure(f"{path}: the corpus holds no prompts")
    entries: list[dict] = []
    names: set[str] = set()
    digests: set[str] = set()
    duplicates: list[str] = []
    for item in raw:
        entry = _coverage_entry(item, str(path))
        if entry["name"] in names:
            raise NumericsFailure(f"{path}: prompt name {entry['name']!r} appears more than once")
        names.add(entry["name"])
        digest = coverage_prompt_digest(entry)
        if digest in digests:
            duplicates.append(entry["name"])
            continue
        digests.add(digest)
        entries.append(entry)
    return entries, duplicates


def load_coverage_input_suite(path: Path) -> tuple[set[str], set[str]]:
    """The content of a suite that produced the dump's existing inputs.

    Returns the text digests and the content-addressed input names it declares, so a sweep can
    skip every prompt that suite already ran. This is a declaration by the operator: the dump's
    position-addressed inputs (input-0, ...) carry no content of their own.
    """
    document = load_json(path)
    if document.get("schema") not in (SUITE_SCHEMA, COVERAGE_CORPUS_SCHEMA):
        raise NumericsFailure(
            f"{path}: schema {document.get('schema')!r} is not {SUITE_SCHEMA} or {COVERAGE_CORPUS_SCHEMA}"
        )
    raw = document.get("prompts")
    if not isinstance(raw, list) or not raw:
        raise NumericsFailure(f"{path}: the suite holds no prompts")
    digests: set[str] = set()
    input_names: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise NumericsFailure(f"{path}: a prompt is not an object")
        if "prompt_ids" in item:
            ids = item["prompt_ids"]
            if not isinstance(ids, list) or not ids or not all(is_integer(token) and 0 <= token <= 0xFFFFFFFF for token in ids):
                raise NumericsFailure(f"{path}: a prompt_ids entry needs a non-empty list of u32 ids")
            input_names.add(token_input_name(ids))
            digests.add(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            continue
        entry = _coverage_entry(item, str(path))
        digests.add(coverage_prompt_digest(entry))
    return digests, input_names


def token_input_name(tokens: list[int]) -> str:
    """The Rust recorder's content-addressed input name: FNV-1a 64 over little-endian u32s."""
    digest = 0xCBF29CE484222325
    for token in tokens:
        for byte in struct.pack("<I", token):
            digest ^= byte
            digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"input-{digest:016x}"


def load_coverage_seen(path: Path, model: str) -> dict:
    """The prompts a sweep already sent for one dump, by content digest."""
    if not path.exists():
        return {"digests": set(), "inputs": set(), "entries": {}}
    document = load_json(path)
    if document.get("schema") != COVERAGE_SEEN_SCHEMA:
        raise NumericsFailure(f"{path}: schema {document.get('schema')!r} is not {COVERAGE_SEEN_SCHEMA}")
    if document.get("model") != model:
        raise NumericsFailure(f"{path}: it names {document.get('model')!r}, but the dump names {model!r}")
    raw = document.get("inputs")
    if not isinstance(raw, list):
        raise NumericsFailure(f"{path}: inputs is not a list")
    digests: set[str] = set()
    inputs: set[str] = set()
    entries: dict[str, dict] = {}
    for item in raw:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or not all(character in "0123456789abcdef" for character in item["sha256"])
            or not isinstance(item.get("name"), str)
            or not item["name"]
        ):
            raise NumericsFailure(f"{path}: every seen input needs a name and a sha256")
        digest = item["sha256"]
        if digest in digests:
            continue
        digests.add(digest)
        input_name = item.get("input")
        if isinstance(input_name, str) and input_name:
            inputs.add(input_name)
        entries[digest] = {
            "name": item["name"],
            "sha256": digest,
            "input": input_name if isinstance(input_name, str) and input_name else None,
        }
    return {"digests": digests, "inputs": inputs, "entries": entries}


def write_coverage_seen(path: Path, model: str, entries: dict[str, dict]) -> None:
    """Write the seen manifest atomically, sorted by digest so a rerun is byte-identical."""
    document = {
        "schema": COVERAGE_SEEN_SCHEMA,
        "model": model,
        "inputs": [
            {"name": entries[key]["name"], "sha256": key, "input": entries[key]["input"]}
            for key in sorted(entries)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def coverage_covered(result: dict) -> set[tuple[int, int]]:
    return {
        (layer["layer"], expert)
        for layer in result["layers"]
        if layer["experts"]
        for expert, hit in enumerate(layer["counts"])
        if hit > 0
    }


def coverage_counts(result: dict) -> dict[int, list[int]]:
    return {layer["layer"]: layer["counts"] for layer in result["layers"] if layer["experts"]}


# The generator's character classes. Each is a distinct part of a tokenizer's vocabulary; the
# selection is measured, so this list only proposes inputs, it does not choose them.
_COVERAGE_ALPHABETS = {
    "ascii": "".join(chr(code) for code in range(32, 127)),
    "latin-extended": "".join(chr(code) for code in range(0xA0, 0x100)),
    "greek": "".join(chr(code) for code in range(0x391, 0x3CA)),
    "cyrillic": "".join(chr(code) for code in range(0x410, 0x450)),
    "hebrew": "".join(chr(code) for code in range(0x5D0, 0x5EB)),
    "arabic": "".join(chr(code) for code in range(0x621, 0x64B)),
    "devanagari": "".join(chr(code) for code in range(0x900, 0x951)),
    "thai": "".join(chr(code) for code in range(0xE01, 0xE3B)),
    "hangul": "".join(chr(code) for code in range(0xAC00, 0xAC00 + 200)),
    "hiragana": "".join(chr(code) for code in range(0x3041, 0x3097)),
    "han": "".join(chr(code) for code in range(0x4E00, 0x4E00 + 300)),
    "math": "".join(chr(code) for code in range(0x2200, 0x22FF)),
    "box": "".join(chr(code) for code in range(0x2500, 0x2580)),
    "braille": "".join(chr(code) for code in range(0x2800, 0x2900)),
    "combining": "".join(chr(code) for code in range(0x300, 0x370)),
    "emoji": list("😀🚀🧪🔥🌍🎛️🧬🛰️📈🪐🫧🦾🧠🕹️🪄🎼"),
}

_COVERAGE_CODE = (
    "fn @ID@(values: &[u32]) -> u64 {\n    values.iter().map(|value| u64::from(*value)).sum()\n}\n\n// @WORD@ @HEX@\n",
    "def @ID@(values):\n    return sum(value * @N@ for value in values)\n\n# @WORD@ @HEX@\n",
    "static uint64_t @ID@(const uint32_t *values, size_t count) {\n    uint64_t total = 0;\n    for (size_t i = 0; i < count; ++i) total += values[i];\n    return total;\n}\n/* @WORD@ @HEX@ */\n",
    "const @ID@ = (values) => values.reduce((total, value) => total + value * @N@, 0);\n// @WORD@ @HEX@\n",
    "SELECT @ID@, COUNT(*) FROM events\nWHERE stamp >= '2026-@N@-01' AND kind = '@WORD@'\nGROUP BY @ID@ ORDER BY 2 DESC;\n-- @HEX@\n",
    "for file in \"$@\"; do\n  sha256sum \"$file\" | cut -c1-16\ndone\n# @WORD@ @HEX@\n",
    "mov rax, @N@\nxor rcx, rcx\nloop_@N@:\n  add rcx, [rdi + rax * 8]\n  dec rax\n  jnz loop_@N@\nret\n; @WORD@ @HEX@\n",
    "{\"name\": \"@WORD@\", \"id\": @N@, \"digest\": \"@HEX@\", \"values\": [@N@, @N@, @N@]}\n",
    "@WORD@:\n  id: @N@\n  digest: @HEX@\n  paths:\n    - /var/lib/@WORD@/@N@\n    - /srv/@WORD@/cache\n",
    "^(?P<@ID@>[a-z0-9_-]{2,32})(?:\\s+)(?P<@WORD@>\\d{1,9})(?:\\s+)(?P<@HEX@>[0-9a-f]{16})$\n",
    "<section id=\"@ID@\" data-id=\"@N@\">\n  <p>@WORD@ @HEX@</p>\n  <span class=\"@WORD@\">@N@</span>\n</section>\n",
    "\\begin{equation}\n  @ID@(x) = \\sum_{i=1}^{@N@} \\frac{x_i^{2}}{i + @N@}\n\\end{equation}\n% @WORD@ @HEX@\n",
)

_COVERAGE_WORDS = (
    "alpha", "bridge", "cipher", "delta", "ember", "fathom", "glacier", "harbor", "isotope",
    "juniper", "kestrel", "lattice", "magnet", "nebula", "obsidian", "prairie", "quartz",
    "ripple", "saffron", "tundra", "umbra", "vector", "willow", "xenon", "yarrow", "zenith",
)


def _generation_bytes(seed: int, index: int, purpose: str) -> bytes:
    return hashlib.sha256(f"coverage-generation:{seed}:{index}:{purpose}".encode()).digest()


def _generation_string(alphabet: str | list[str], length: int, digest: bytes) -> str:
    out: list[str] = []
    block = digest
    while len(out) < length:
        for byte in block:
            out.append(alphabet[byte % len(alphabet)])
            if len(out) >= length:
                break
        block = hashlib.sha256(block).digest()
    return "".join(out)


def _generation_pick(values: tuple | list, digest: bytes, offset: int = 0) -> object:
    return values[digest[offset % len(digest)] % len(values)]


def _generate_charset(seed: int, index: int, digest: bytes) -> str:
    names = sorted(_COVERAGE_ALPHABETS)
    name = names[digest[0] % len(names)]
    length = 64 + digest[1] * 8
    body = _generation_string(_COVERAGE_ALPHABETS[name], length, _generation_bytes(seed, index, "charset"))
    return f"[{name} {index}]\n{body}"


def _generate_numerals(seed: int, index: int, digest: bytes) -> str:
    mode = digest[0] % 4
    body = _generation_bytes(seed, index, "numerals")
    if mode == 0:
        text = _generation_string("0123456789", 512, body)
    elif mode == 1:
        text = " ".join(f"{byte:02x}" for byte in body * 8)
    elif mode == 2:
        text = "\n".join(
            f"{int.from_bytes(body[start : start + 8], 'big') / (10 ** (start % 12 + 1)):.9e}"
            for start in range(0, 64, 8)
        )
    else:
        text = "\n".join(
            f"2026-{byte % 12 + 1:02d}-{byte % 28 + 1:02d}T{byte % 24:02d}:{byte % 60:02d}:{byte % 60:02d}Z "
            f"{byte}.{byte % 256}.{byte % 256}.{byte % 256}/{byte % 32 + 1}"
            for byte in body
        )
    return f"# numerals {index} mode {mode}\n{text}"


def _generate_code(seed: int, index: int, digest: bytes) -> str:
    template = _generation_pick(_COVERAGE_CODE, digest)
    word = _generation_pick(_COVERAGE_WORDS, digest, 1)
    identifier = f"{word}_{index:05d}"
    text = (
        template.replace("@ID@", identifier)
        .replace("@WORD@", word)
        .replace("@HEX@", digest.hex()[:32])
        .replace("@N@", str(int.from_bytes(digest[:4], "big") % 100000 + 1))
    )
    return f"# code {index}\n{text}"


def _generate_scripts(seed: int, index: int, digest: bytes) -> str:
    names = sorted(name for name in _COVERAGE_ALPHABETS if name not in ("ascii", "latin-extended", "math", "box", "braille", "combining"))
    name = names[digest[0] % len(names)]
    alphabet = _COVERAGE_ALPHABETS[name]
    words = 12 + digest[1] % 24
    pieces = []
    for word_index in range(words):
        length = 3 + (digest[(word_index + 2) % len(digest)] % 7)
        pieces.append(_generation_string(alphabet, length, _generation_bytes(seed, index, f"script-{word_index}")))
    return f"[{name} {index}]\n" + " ".join(pieces)


def _generate_long(seed: int, index: int, digest: bytes) -> str:
    paragraphs = 6 + digest[0] % 20
    lines = [
        f"# long document {index}",
        "",
        "The following record is a fixed test document. It repeats a small vocabulary with changing "
        "identifiers so the model sees a long, self-similar context rather than one short sentence.",
    ]
    for paragraph in range(paragraphs):
        key = _generation_bytes(seed, index, f"paragraph-{paragraph}")
        word = _generation_pick(_COVERAGE_WORDS, key)
        lines += [
            "",
            f"Section {paragraph + 1}.{index}. The {word} record carries identifier "
            f"{key.hex()[:12]} at offset {int.from_bytes(key[:4], 'big') % 100000}. "
            f"Its measured value is {int.from_bytes(key[4:8], 'big') % 10000} units, and the "
            f"corresponding control value is {int.from_bytes(key[8:12], 'big') % 10000} units. "
            f"The same row is repeated in the appendix with the tag {key.hex()[12:20]}.",
        ]
    lines += ["", "Appendix table", "", "| key | value | tag |", "| :-- | --: | :-- |"]
    for row in range(6):
        key = _generation_bytes(seed, index, f"table-{row}")
        lines.append(f"| {key.hex()[:8]} | {int.from_bytes(key[8:12], 'big') % 10000} | {key.hex()[12:20]} |")
    return "\n".join(lines)


def _generate_repetition(seed: int, index: int, digest: bytes) -> str:
    unit = _generation_pick(("the ", "0", "ab", "αβ", "TODO: ", "1 2 3 ", "\n", "∑ "), digest)
    count = 128 + digest[1] * 16
    body = unit * count
    return f"# repetition {index}\n{body}"


def _generate_mixed(seed: int, index: int, digest: bytes) -> str:
    charset = _generate_charset(seed, index + 100000, digest)
    code = _generate_code(seed, index + 100000, _generation_bytes(seed, index, "mixed-code"))
    numerals = _generate_numerals(seed, index + 100000, _generation_bytes(seed, index, "mixed-numerals"))
    scripts = _generate_scripts(seed, index + 100000, _generation_bytes(seed, index, "mixed-scripts"))
    return f"{charset}\n\n{code}\n\n{numerals}\n\n{scripts}"


def generate_coverage_prompts(seed: int, count: int) -> list[dict]:
    """A deterministic corpus of generated inputs; the sweep measures which ones help."""
    if count < 0:
        raise NumericsFailure(f"--generate {count}: it cannot be negative")
    families = ("charset", "numerals", "code", "scripts", "long", "repetition", "mixed")
    generators = {
        "charset": _generate_charset,
        "numerals": _generate_numerals,
        "code": _generate_code,
        "scripts": _generate_scripts,
        "long": _generate_long,
        "repetition": _generate_repetition,
        "mixed": _generate_mixed,
    }
    prompts = []
    for index in range(count):
        family = families[index % len(families)]
        digest = _generation_bytes(seed, index, family)
        prompts.append(
            {
                "name": f"gen/{seed}/{index:05d}",
                "family": family,
                "prompt": generators[family](seed, index, digest),
            }
        )
    return prompts


def tokenize_coverage_prompt(url: str, model: str, entry: dict, timeout: float) -> list[int]:
    if "prompt_ids" in entry:
        ids = entry["prompt_ids"]
    elif "messages" in entry:
        try:
            ids = http_json(f"{url}/tokenize", {"model": model, "messages": entry["messages"]}, timeout)["tokens"]
        except KeyError as error:
            raise NumericsFailure(f"prompt {entry['name']!r}: /tokenize returned no tokens") from error
    else:
        try:
            ids = http_json(
                f"{url}/tokenize",
                {"model": model, "prompt": entry["prompt"], "add_special_tokens": True},
                timeout,
            )["tokens"]
        except KeyError as error:
            raise NumericsFailure(f"prompt {entry['name']!r}: /tokenize returned no tokens") from error
    if not isinstance(ids, list) or not ids:
        raise NumericsFailure(f"prompt {entry['name']!r}: tokenization returned no ids")
    if not all(is_integer(token) and 0 <= token <= 0xFFFFFFFF for token in ids):
        raise NumericsFailure(f"prompt {entry['name']!r}: tokenization returned a non-u32 id")
    return ids


def run_coverage_prompt(url: str, model: str, ids: list[int], max_tokens: int, timeout: float) -> None:
    body = {"model": model, "prompt": ids, "temperature": 0, "max_tokens": max_tokens}
    response = http_json(f"{url}/v1/completions", body, timeout)
    if not isinstance(response.get("choices"), list) or not response["choices"]:
        raise NumericsFailure(f"prompt {ids[0]}: the server returned no choice")


def wait_for_coverage(
    path: Path, before_inputs: int, layout: dict, layout_origin: str, timeout: float
) -> tuple[dict, dict]:
    """Wait for the server to write a dump with one more input, then inspect it."""
    deadline = time.monotonic() + timeout
    last = "the dump has not been read"
    while True:
        try:
            document = load_json(path)
            result = coverage_inspect(document, layout, layout_origin, 0)
            if result["inputs"] > before_inputs:
                return document, result
            last = f"it still names {result['inputs']} input(s)"
        except NumericsFailure as error:
            last = str(error)
        if time.monotonic() >= deadline:
            raise NumericsFailure(
                f"{path}: no new input appeared within {timeout:g}s ({last}); launch the server with "
                "SPARKFLASH_NUMERICS_COVERAGE naming this path and leave the sweep on an idle server"
            )
        time.sleep(0.05)


def render_coverage_sweep(report: dict) -> str:
    lines = [
        f"coverage sweep: {report['model']}, {report['slots']} routed slot(s); layout {report['layout']}",
        f"seed {report['seed']}, order {report['order']}, batch {report['batch_size']}, "
        f"dump {report['dump']}",
        f"missing before: {report['missing_before']} ({report['coverage_before']:.2%} covered)",
        "",
        "| batch | sent | kept | discarded | empty | new slots | missing after | coverage after |",
        "| --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for batch in report["batches"]:
        lines.append(
            f"| {batch['index']} | {batch['sent']} | {batch['kept']} | {batch['discarded']} | "
            f"{batch['empty']} | {batch['new_slots']} | {batch['missing_after']} | "
            f"{batch['coverage_after']:.2%} |"
        )
    skipped = ", ".join(f"{name} {count}" for name, count in report["skipped"].items() if count) or "none"
    lines += [
        "",
        f"selected {report['selected']} prompt(s) from {report['sent']} sent; skipped: {skipped}",
        f"missing after: {report['missing_after']} ({report['coverage_after']:.2%} covered)",
    ]
    if report["complete"]:
        lines.append("PASS coverage: the dump now reaches every routed expert of every layer.")
    else:
        lines.append(
            "not complete: run another window with the same dump and seen manifest, or a larger "
            "--generate corpus; the judge is still the gate."
        )
    return "\n".join(lines) + "\n"


def sweep_coverage(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Greedily select measured prompts from a corpus against a dump's missing experts.

    Every candidate is sent to the live server and the dump is read before and after its forward,
    so the slots a prompt actually reached are what decides whether it is kept. The dump itself is
    the evidence; the returned selection is the suite that reproduces the measured additions.
    """
    dump = load_json(args.dump)
    model = dump.get("model")
    if not isinstance(model, str) or not model:
        raise NumericsFailure(f"{args.dump}: the dump names no model")
    layout, origin = find_layout(model, args.layout)
    result = coverage_inspect(dump, layout, origin, 0)
    total = result["experts"]
    covered = coverage_covered(result)
    missing_before = total - len(covered)
    if args.batch < 1:
        raise NumericsFailure(f"--batch {args.batch}: it must be at least one")
    if args.max_candidates < 1:
        raise NumericsFailure(f"--max-candidates {args.max_candidates}: it must be at least one")
    if args.max_tokens < 1:
        raise NumericsFailure(f"--max-tokens {args.max_tokens}: it must be at least one")
    if missing_before == 0:
        report = {
            "schema": COVERAGE_SWEEP_SCHEMA,
            "model": model,
            "layout": origin,
            "dump": str(args.dump),
            "seed": args.seed,
            "order": args.order,
            "batch_size": args.batch,
            "max_tokens": args.max_tokens,
            "slots": total,
            "missing_before": 0,
            "coverage_before": 1.0,
            "missing_after": 0,
            "coverage_after": 1.0,
            "complete": True,
            "sent": 0,
            "selected": 0,
            "selected_names": [],
            "skipped": {},
            "duplicates": [],
            "batches": [],
            "seen": str(args.seen or Path(f"{args.dump}.seen.json")),
        }
        return report, []

    card = fetch_card(args.url, args.timeout)
    if card.get("id") != model:
        raise NumericsFailure(
            f"{args.url}: the server names {card.get('id')!r}, but {args.dump} names {model!r}; "
            "the sweep would record another model's routing"
        )

    corpus, duplicates = load_coverage_corpus(args.corpus)
    generated = generate_coverage_prompts(args.seed, args.generate) if args.generate else []
    candidates: list[dict] = []
    names: set[str] = set()
    digests: set[str] = set()
    for entry in [*corpus, *generated]:
        if entry["name"] in names:
            raise NumericsFailure(
                f"the corpus and the generator both name {entry['name']!r}; rename one or change --seed"
            )
        names.add(entry["name"])
        digest = coverage_prompt_digest(entry)
        if digest in digests:
            duplicates.append(entry["name"])
            continue
        digests.add(digest)
        candidates.append(entry)
    if not candidates:
        raise NumericsFailure(
            f"{args.corpus}: every prompt is already a duplicate; the sweep has nothing to measure"
        )
    if args.order == "shuffled":
        candidates.sort(
            key=lambda entry: hashlib.sha256(f"{args.seed}:{entry['name']}".encode()).hexdigest()
        )

    seen_path = args.seen or Path(f"{args.dump}.seen.json")
    seen = load_coverage_seen(seen_path, model)
    if args.input_suite:
        suite_digests, suite_inputs = load_coverage_input_suite(args.input_suite)
        seen["digests"] |= suite_digests
        seen["inputs"] |= suite_inputs
    seen_entries = dict(seen["entries"])
    # The old recorder's position-addressed names are input-0, input-1, ...; a content-addressed
    # name is 16 hex digits, so it cannot be mistaken for a legacy one.
    legacy = {name for name in result["input_names"] if re.fullmatch(r"input-\d{1,15}", name)}
    mapped = {entry["input"] for entry in seen_entries.values() if entry.get("input")}
    if legacy and not args.input_suite and not legacy <= mapped:
        raise NumericsFailure(
            f"{args.dump}: it holds {len(legacy)} position-addressed input(s) (input-0, ...) with no "
            "content identity, so the sweep cannot prove it would not repeat one; pass --input-suite "
            "FILE (the suite that produced them) or a --seen manifest, or start a fresh dump"
        )

    def remember(entry: dict, input_name: str | None) -> None:
        digest = coverage_prompt_digest(entry)
        seen_entries[digest] = {
            "name": entry["name"],
            "sha256": digest,
            "input": input_name,
        }
        seen["digests"].add(digest)
        if input_name:
            seen["inputs"].add(input_name)

    dump_names = set(result["input_names"])
    current = result
    selected: list[dict] = []
    batches: list[dict] = []
    skipped = {"already-seen": 0, "already-in-dump": 0, "name-in-dump": 0}
    sent = 0
    batch = {
        "sent": 0,
        "kept": 0,
        "discarded": 0,
        "empty": 0,
        "new_slots": 0,
        "missing_before": missing_before,
    }

    def close_batch() -> None:
        if not batch["sent"]:
            return
        missing_after = total - len(covered)
        batches.append(
            {
                "index": len(batches) + 1,
                "sent": batch["sent"],
                "kept": batch["kept"],
                "discarded": batch["discarded"],
                "empty": batch["empty"],
                "new_slots": batch["new_slots"],
                "missing_before": batch["missing_before"],
                "missing_after": missing_after,
                "coverage_before": (total - batch["missing_before"]) / total,
                "coverage_after": (total - missing_after) / total,
            }
        )
        batch.update(
            {
                "sent": 0,
                "kept": 0,
                "discarded": 0,
                "empty": 0,
                "new_slots": 0,
                "missing_before": missing_after,
            }
        )
        # A window can be interrupted between batches; keep the manifest current so a rerun does
        # not repeat what this one already measured.
        write_coverage_seen(seen_path, model, seen_entries)

    try:
        for entry in candidates:
            if sent >= args.max_candidates or len(covered) == total:
                break
            digest = coverage_prompt_digest(entry)
            if digest in seen["digests"]:
                skipped["already-seen"] += 1
                continue
            if entry["name"] in dump_names:
                skipped["name-in-dump"] += 1
                continue
            ids = tokenize_coverage_prompt(args.url, model, entry, args.timeout)
            input_name = token_input_name(ids)
            if input_name in dump_names or input_name in seen["inputs"]:
                skipped["already-in-dump"] += 1
                remember(entry, input_name)
                continue
            before_inputs = current["inputs"]
            before_counts = coverage_counts(current)
            run_coverage_prompt(args.url, model, ids, args.max_tokens, args.timeout)
            _, current = wait_for_coverage(args.dump, before_inputs, layout, origin, args.timeout)
            after_counts = coverage_counts(current)
            reached: set[tuple[int, int]] = set()
            for layer, counts in after_counts.items():
                previous = before_counts.get(layer)
                if previous is None or len(previous) != len(counts):
                    raise NumericsFailure(
                        f"{args.dump}: layer {layer} changed shape during the sweep; the server is "
                        "writing a different layout"
                    )
                for expert, count in enumerate(counts):
                    if count > previous[expert]:
                        reached.add((layer, expert))
            new = reached - covered
            if new:
                covered |= reached
                selected.append(entry)
                batch["kept"] += 1
                batch["new_slots"] += len(new)
            else:
                batch["discarded"] += 1
                if not reached:
                    batch["empty"] += 1
            sent += 1
            batch["sent"] += 1
            remember(entry, input_name)
            dump_names.add(input_name)
            if batch["sent"] >= args.batch:
                close_batch()
        close_batch()
    finally:
        write_coverage_seen(seen_path, model, seen_entries)

    missing_after = total - len(covered)
    report = {
        "schema": COVERAGE_SWEEP_SCHEMA,
        "model": model,
        "layout": origin,
        "dump": str(args.dump),
        "seed": args.seed,
        "order": args.order,
        "batch_size": args.batch,
        "max_tokens": args.max_tokens,
        "slots": total,
        "missing_before": missing_before,
        "coverage_before": (total - missing_before) / total,
        "missing_after": missing_after,
        "coverage_after": (total - missing_after) / total,
        "complete": missing_after == 0,
        "sent": sent,
        "selected": len(selected),
        "selected_names": [entry["name"] for entry in selected],
        "skipped": skipped,
        "duplicates": duplicates,
        "batches": batches,
        "seen": str(seen_path),
    }
    return report, selected


def check_monitor_dump(dump: dict, origin: str) -> dict:
    """The shape of one monitor dump: a probe, a fixed sample, and per-layer/per-expert values."""
    if dump.get("schema") != MONITORS_SCHEMA:
        raise NumericsFailure(f"{origin}: schema {dump.get('schema')!r} is not {MONITORS_SCHEMA}")
    if not isinstance(dump.get("model"), str) or not dump["model"]:
        raise NumericsFailure(f"{origin}: the dump names no model")
    probe = dump.get("probe")
    if (
        not isinstance(probe, dict)
        or not isinstance(probe.get("name"), str)
        or not isinstance(probe.get("prompt_ids"), list)
        or not all(is_integer(token) for token in probe["prompt_ids"])
        or not is_integer(probe.get("position"))
        or probe["position"] < 0
    ):
        raise NumericsFailure(
            f"{origin}: the probe needs a name, integer prompt_ids and a non-negative position"
        )
    sample = dump.get("sample")
    if (
        not isinstance(sample, dict)
        or not is_integer(sample.get("width"))
        or sample["width"] <= 0
        or not isinstance(sample.get("indices"), list)
        or not sample["indices"]
        or not all(is_integer(index) for index in sample["indices"])
    ):
        raise NumericsFailure(
            f"{origin}: the sample needs a positive width and a non-empty list of integer indices"
        )
    indices = sample["indices"]
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise NumericsFailure(f"{origin}: the sampled indices must be strictly increasing")
    if indices[0] < 0 or indices[-1] >= sample["width"]:
        raise NumericsFailure(
            f"{origin}: a sampled index is outside the declared width {sample['width']}"
        )
    layers = dump.get("layers")
    if not isinstance(layers, list) or not layers:
        raise NumericsFailure(f"{origin}: the dump holds no layers")
    seen_layers = set()
    normalized = []
    for entry in layers:
        if not isinstance(entry, dict) or not is_integer(entry.get("layer")):
            raise NumericsFailure(f"{origin}: every layer entry needs an integer layer")
        number = entry["layer"]
        if number < 0 or number in seen_layers:
            raise NumericsFailure(f"{origin}: layer {number} is negative or appears more than once")
        seen_layers.add(number)
        values = entry.get("values")
        if not isinstance(values, list) or len(values) != len(indices):
            raise NumericsFailure(
                f"{origin}: layer {number} needs one value per sampled index ({len(indices)})"
            )
        if not all(is_number(value) and math.isfinite(value) for value in values):
            raise NumericsFailure(f"{origin}: layer {number} holds a non-finite or non-numeric value")
        experts = entry.get("experts")
        if not isinstance(experts, list) or not experts:
            raise NumericsFailure(f"{origin}: layer {number} holds no expert values")
        seen_experts = set()
        normalized_experts = []
        for expert in experts:
            if not isinstance(expert, dict) or not is_integer(expert.get("expert")):
                raise NumericsFailure(f"{origin}: layer {number} has an expert without an integer id")
            identifier = expert["expert"]
            if identifier < 0 or identifier in seen_experts:
                raise NumericsFailure(
                    f"{origin}: layer {number} expert {identifier} is negative or appears more than once"
                )
            seen_experts.add(identifier)
            expert_values = expert.get("values")
            if not isinstance(expert_values, list) or len(expert_values) != len(indices):
                raise NumericsFailure(
                    f"{origin}: layer {number} expert {identifier} needs one value per sampled index"
                )
            if not all(
                is_number(value) and math.isfinite(value) for value in expert_values
            ):
                raise NumericsFailure(
                    f"{origin}: layer {number} expert {identifier} holds a non-finite or non-numeric value"
                )
            normalized_experts.append({"expert": identifier, "values": expert_values})
        normalized.append({"layer": number, "values": values, "experts": normalized_experts})
    return {
        "model": dump["model"],
        "probe": probe,
        "sample": sample,
        "layers": sorted(normalized, key=lambda layer: layer["layer"]),
    }


def relative_error(reference: list[float], candidate: list[float], origin: str) -> float:
    """The relative L2 error of the candidate sample against the reference sample."""
    if len(reference) != len(candidate):
        raise NumericsFailure(f"{origin}: {len(reference)} reference value(s) against {len(candidate)}")
    denominator = math.sqrt(math.fsum(value * value for value in reference))
    numerator = math.sqrt(math.fsum((left - right) ** 2 for left, right in zip(reference, candidate)))
    if denominator == 0.0:
        if numerator == 0.0:
            return 0.0
        raise NumericsFailure(
            f"{origin}: the reference sample is all zeros, so the relative error is undefined"
        )
    return numerator / denominator


def judge_monitors(reference: dict, candidate: dict, layer_bar: float, expert_bar: float) -> dict:
    """Compare a candidate forward's sample with the reference forward's, per layer and per shared
    expert.

    A router is not arithmetic: a slightly different hidden state can change which experts a row
    reaches, and that is a measured fact about the candidate, not a reason to refuse the pair.
    Each layer's experts are intersected; the experts only one dump holds are counted and named,
    and the expert bar is judged over the ones both hold. The layer bar is always judged, and the
    exit code follows the bars alone.
    """
    if not math.isfinite(layer_bar) or layer_bar <= 0.0:
        raise NumericsFailure(f"--layer-bar {layer_bar}: a bar must be finite and positive")
    if not math.isfinite(expert_bar) or expert_bar <= 0.0:
        raise NumericsFailure(f"--expert-bar {expert_bar}: a bar must be finite and positive")
    if reference["model"] != candidate["model"]:
        raise NumericsFailure(
            f"the reference names {reference['model']!r}, the candidate {candidate['model']!r}"
        )
    if reference["probe"] != candidate["probe"]:
        raise NumericsFailure(
            "the two dumps ran different probes; a monitor compares the same forward input"
        )
    if reference["sample"] != candidate["sample"]:
        raise NumericsFailure(
            "the two dumps sampled different indices or widths; a monitor compares like with like"
        )
    reference_layers = {layer["layer"]: layer for layer in reference["layers"]}
    candidate_layers = {layer["layer"]: layer for layer in candidate["layers"]}
    if reference_layers.keys() != candidate_layers.keys():
        raise NumericsFailure(
            "the two dumps do not hold the same layers: "
            f"reference {sorted(reference_layers)}, candidate {sorted(candidate_layers)}"
        )
    failures, results, routing_per_layer = [], [], []
    layers_differing = 0
    experts_unshared = 0
    reference_only_total = 0
    candidate_only_total = 0
    experts_shared = 0
    max_layer = 0.0
    max_expert = None
    for number in sorted(reference_layers):
        left, right = reference_layers[number], candidate_layers[number]
        layer_error = relative_error(left["values"], right["values"], f"layer {number}")
        max_layer = max(max_layer, layer_error)
        layer_ok = layer_error <= layer_bar
        if not layer_ok:
            failures.append(f"layer {number}: {layer_error:.6e} > layer bar {layer_bar:.6e}")
        reference_experts = {expert["expert"]: expert for expert in left["experts"]}
        candidate_experts = {expert["expert"]: expert for expert in right["experts"]}
        shared = sorted(reference_experts.keys() & candidate_experts.keys())
        reference_only = sorted(reference_experts.keys() - candidate_experts.keys())
        candidate_only = sorted(candidate_experts.keys() - reference_experts.keys())
        unshared = len(reference_only) + len(candidate_only)
        if unshared:
            layers_differing += 1
            experts_unshared += unshared
            reference_only_total += len(reference_only)
            candidate_only_total += len(candidate_only)
        experts_shared += len(shared)
        routing_per_layer.append(
            {
                "layer": number,
                "shared": len(shared),
                "reference_only": reference_only,
                "candidate_only": candidate_only,
            }
        )
        experts, worst = [], None
        for expert in shared:
            error = relative_error(
                reference_experts[expert]["values"],
                candidate_experts[expert]["values"],
                f"layer {number} expert {expert}",
            )
            max_expert = error if max_expert is None else max(max_expert, error)
            ok = error <= expert_bar
            if not ok:
                failures.append(
                    f"layer {number} expert {expert}: {error:.6e} > expert bar {expert_bar:.6e}"
                )
            experts.append({"expert": expert, "rel_error": error, "ok": ok})
            if worst is None or error > worst["rel_error"]:
                worst = {"expert": expert, "rel_error": error}
        results.append(
            {
                "layer": number,
                "rel_error": layer_error,
                "ok": layer_ok,
                "shared_experts": len(shared),
                "reference_only": reference_only,
                "candidate_only": candidate_only,
                "worst_expert": worst,
                "experts": experts,
            }
        )
    return {
        "schema": MONITOR_RESULT_SCHEMA,
        "model": reference["model"],
        "probe": reference["probe"],
        "sample": reference["sample"],
        "bars": {"layer": layer_bar, "expert": expert_bar},
        "routing": {
            "layers_differing": layers_differing,
            "experts_unshared": experts_unshared,
            "reference_only": reference_only_total,
            "candidate_only": candidate_only_total,
            "experts_shared": experts_shared,
            "per_layer": routing_per_layer,
        },
        "maxima": {"layer": max_layer, "expert": max_expert},
        "layers": results,
        "failures": failures,
        "pass": not failures,
    }


def render_monitors(result: dict) -> str:
    routing = result["routing"]
    maxima = result["maxima"]
    lines = [
        f"monitors: {result['model']}, probe {result['probe']['name']!r}, "
        f"{len(result['layers'])} layer(s), sample {len(result['sample']['indices'])} of "
        f"{result['sample']['width']} values",
        f"routing: {routing['layers_differing']} of {len(result['layers'])} layer(s) routed a row "
        f"to a different expert; {routing['experts_unshared']} expert(s) unshared "
        f"({routing['reference_only']} reference-only, {routing['candidate_only']} "
        f"candidate-only), {routing['experts_shared']} shared expert(s) judged",
        "",
        "| layer | relative error | layer bar | shared experts | unshared | worst expert | "
        "worst expert error | expert bar |",
        "| --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for layer in result["layers"]:
        worst = layer["worst_expert"]
        worst_text = "-" if worst is None else f"{worst['expert']}"
        worst_error = "-" if worst is None else f"{worst['rel_error']:.3e}"
        unshared = len(layer["reference_only"]) + len(layer["candidate_only"])
        lines.append(
            f"| {layer['layer']} | {layer['rel_error']:.3e} | {result['bars']['layer']:.3e} | "
            f"{layer['shared_experts']} | {unshared} | {worst_text} | {worst_error} | "
            f"{result['bars']['expert']:.3e} |"
        )
    differing = [
        layer
        for layer in result["layers"]
        if layer["reference_only"] or layer["candidate_only"]
    ]
    if differing:
        lines.append("")
        lines.append("routing differences (the shared experts are what the expert bar judges):")
        for layer in differing:
            reference_only = layer["reference_only"] or "-"
            candidate_only = layer["candidate_only"] or "-"
            lines.append(
                f"  layer {layer['layer']}: reference only {reference_only}; "
                f"candidate only {candidate_only}"
            )
    lines.append("")
    if result["pass"]:
        lines.append(
            f"PASS monitors: every layer is within {result['bars']['layer']:.3e} and every shared "
            f"routed expert within {result['bars']['expert']:.3e} of the reference forward."
        )
    else:
        lines.append(f"FAIL monitors: {len(result['failures'])} bar(s) exceeded")
        lines += [f"  {failure}" for failure in result["failures"]]
    return "\n".join(lines) + "\n"


# --- self-test --------------------------------------------------------------------------------


def synthetic_reference(tokens: list[int], logprob: float, finish: str = "length") -> dict:
    reference = {
        "schema": REFERENCE_SCHEMA,
        "evidence": "logprobs",
        "key": {"model": "m", "world": 4, "profile": "reference", "plan": "tp", "arch": "sm_121a"},
        "settings": {"max_tokens": len(tokens), "top_logprobs": 2},
        "forced_from": None,
        "prompts": [
            {
                "name": "p",
                "prompt_ids": [1, 2],
                "tokens": tokens,
                "logprobs": [logprob] * len(tokens),
                "top": [[[token, logprob], [token + 1, logprob - 1.0]] for token in tokens],
                "finish_reason": finish,
                "text": "",
            }
        ],
    }
    reference["fingerprint"] = fingerprint(reference)
    return reference


ANCHOR_PATH = f"{NUMERICS}/references/anchor.json"


def synthetic_registry(entries: list[dict], prerequisites: bool) -> dict:
    """A registry; with its prerequisites on, it holds the anchor they need."""
    anchors = []
    if prerequisites:
        anchor = synthetic_reference([9], -0.5)
        anchors.append(
            {
                key: value
                for key, value in synthetic_entry(1, anchor, "references/anchor.json").items()
                if key != "version"
            }
        )
    return {
        "prerequisites": {
            **{name: prerequisites for name in PREREQUISITES},
            "coverage_slots": 0.93,
        },
        "config": entries,
        "anchor": anchors,
        "retired": [],
        "coverage": [],
        "monitors": [],
    }


def synthetic_entry(
    version: int,
    reference: dict,
    path: str = "references/q.json",
    report: str | None = None,
    approved: dict | None = None,
) -> dict:
    entry = {
        "model": "m",
        "world": 4,
        "profile": "reference",
        "plan": "tp",
        "arch": "sm_121a",
        "version": version,
        "evidence": "logprobs",
        "reference": path,
        "fingerprint": reference["fingerprint"],
        "captured": "2026-11-03",
    }
    if report:
        entry["report"] = report
    if approved:
        entry["approved"] = approved
    return entry


def self_test() -> int:
    """The declaration check refuses each mistake it exists for and passes a correct declaration."""
    old = synthetic_reference([5, 6], -0.25)
    # One ulp of one logprob: the smallest change there is.
    nudged = struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", -0.25))[0] + 1))[0]
    flipped = synthetic_reference([5, 6], nudged)
    longer = synthetic_reference([5, 6, 7], -0.25)
    stopped = synthetic_reference([5, 6], -0.25, finish="stop")
    report = "docs/numerics/m/2.md"
    path = f"{NUMERICS}/references/q.json"
    moved = f"{NUMERICS}/references/moved.json"
    anchor = render_reference(synthetic_reference([9], -0.5)).encode()

    def files(reference: dict, extra: dict | None = None) -> Callable[[str], bytes | None]:
        store = {path: render_reference(reference).encode(), ANCHOR_PATH: anchor, **(extra or {})}
        return store.get

    # (name, prerequisites on, head entry, head reference, changed paths, extra files, refused)
    cases = [
        ("unchanged", False, synthetic_entry(1, old), old, {}, None, False),
        ("an undeclared one-ulp change", False, synthetic_entry(1, flipped), flipped, {path: "M"}, None, True),
        (
            "a declared non-change",
            True,
            synthetic_entry(2, old, report=report),
            old,
            {report: "A"},
            {report: b"x"},
            True,
        ),
        (
            "a declaration before the prerequisites",
            False,
            synthetic_entry(2, flipped, report=report),
            flipped,
            {path: "M", report: "A"},
            {report: b"x"},
            True,
        ),
        (
            "a declaration without its report",
            True,
            synthetic_entry(2, flipped),
            flipped,
            {path: "M"},
            None,
            True,
        ),
        (
            "a stale fingerprint",
            False,
            dict(synthetic_entry(1, old), fingerprint="0" * 64),
            old,
            {},
            None,
            True,
        ),
        (
            "a reference captured from another configuration",
            False,
            synthetic_entry(1, dict(old, key=dict(old["key"], world=2))),
            dict(old, key=dict(old["key"], world=2)),
            {path: "M"},
            None,
            True,
        ),
        ("a report that does not exist", False, synthetic_entry(1, old, report=report), old, {}, None, True),
        ("a longer capture extends, it does not change", False, synthetic_entry(1, longer), longer, {path: "M"}, None, False),
        # Coverage and monitors are per model (docs/NUMERICS.md section 5), and model "m" has
        # neither row, so even a correctly declared change is refused outside the approval path.
        (
            "a declared change with its report but no per-model evidence",
            True,
            synthetic_entry(2, flipped, report=report),
            flipped,
            {path: "M", report: "A"},
            {report: b"x"},
            True,
        ),
        (
            "a stopped capture that only changes its finish reason",
            False,
            synthetic_entry(1, stopped),
            stopped,
            {path: "M"},
            None,
            True,
        ),
        (
            "a bump that only moves the reference",
            True,
            synthetic_entry(2, old, path="references/moved.json", report=report),
            old,
            {path: "D", moved: "A", report: "A"},
            {moved: render_reference(old).encode(), report: b"x"},
            True,
        ),
        (
            "a reference moved without a bump",
            False,
            synthetic_entry(1, old, path="references/moved.json"),
            old,
            {path: "D", moved: "A"},
            {moved: render_reference(old).encode()},
            False,
        ),
    ]
    failed = 0
    for name, prerequisites, entry, reference, changed, extra, refused in cases:
        base = synthetic_registry([synthetic_entry(1, old)], prerequisites)
        head = synthetic_registry([entry], prerequisites)
        findings = declaration_findings(base, head, files(old), files(reference, extra), changed)
        ok = bool(findings) == refused
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {'refused' if findings else 'passed'}")
        for finding in findings:
            print(f"       {finding}")

    # The correctness floor (docs/NUMERICS.md section 5): a reference this change adds must carry
    # the evidence that its arithmetic is right. Inkling's first reference passed on determinism
    # alone and pinned token soup, so a pin now needs a forced comparison, a known-answer decode
    # and the reference's own NLL.
    def floor_card(floor: dict | None) -> dict:
        card = synthetic_reference([5, 6], -0.25)
        card["server"] = {
            "binary_sha256": "0" * 63 + "1",
            "commit": "1" * 40,
            "build_note": None,
            "build_source": "card",
        }
        if floor is not None:
            card["floor"] = floor
        return card

    good_floor = {
        "against": "the pre-change build",
        "forced": {
            "positions": 970,
            "mean_nll_delta": -5.0023e-05,
            "mean_abs_delta": 1.6914e-04,
            "max_abs_delta": 7.3075e-02,
            "top1_agreement": 1.0,
        },
        "known_answers": [
            {"prompt": "The capital of France is", "expected": " Paris", "got": " Paris.", "pass": True},
        ],
        "nll_per_token": 0.180639,
        "explanation": None,
    }

    def floor_case(name: str, floor: dict | None, refused: bool) -> None:
        nonlocal failed
        new_path = "references/q2.json"
        card = floor_card(floor)
        base = synthetic_registry([], True)
        head = synthetic_registry([synthetic_entry(1, card, path=new_path)], True)
        extra = {f"{NUMERICS}/{new_path}": render_reference(card).encode()}
        findings = declaration_findings(
            base, head, files(old, extra), files(card, extra), {f"{NUMERICS}/{new_path}": "A"}
        )
        ok = bool(findings) == refused
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {'refused' if findings else 'passed'}")
        for finding in findings:
            print(f"       {finding}")

    floor_case("a new reference with a correctness floor", good_floor, False)
    floor_case("a new reference with no correctness floor", None, True)
    floor_case(
        "a new reference whose forced distance is a defect",
        {**good_floor, "forced": {**good_floor["forced"], "top1_agreement": 0.2949}},
        True,
    )
    floor_case(
        "the same defect distance, explained and pinned",
        {
            **good_floor,
            "forced": {**good_floor["forced"], "top1_agreement": 0.2949},
            "explanation": "the head layout defect explains the distance; the card pins the fix",
        },
        False,
    )

    # The approval path (docs/NUMERICS.md section 5): the same bump a missing global prerequisite
    # refuses passes when the head entry records the owner's approval of this change, and stays
    # refused when the approval is not a usable record. The per-model anchor stays present.
    approval_report = "docs/numerics/m/approval.md"
    approval = {
        "decision": "The owner's recorded decision for this change",
        "date": "2026-11-18",
        "report": approval_report,
    }
    approved = dict(synthetic_entry(2, flipped, report=report), approved=approval)

    def approval_case(
        name: str,
        base_entry: dict,
        head_entry: dict,
        head_reference: dict,
        changed: dict,
        extra: dict,
        refused: bool,
        anchored: bool = True,
        base_reference: dict | None = None,
    ) -> None:
        nonlocal failed
        base = synthetic_registry([base_entry], anchored)
        head = synthetic_registry([head_entry], anchored)
        for prerequisite in GATING_PREREQUISITES:
            base["prerequisites"][prerequisite] = False
            head["prerequisites"][prerequisite] = False
        findings = declaration_findings(
            base,
            head,
            files(base_reference or old, {report: b"old", approval_report: b"old"}),
            files(head_reference, extra),
            changed,
        )
        ok = bool(findings) == refused
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {'refused' if findings else 'passed'}")
        for finding in findings:
            print(f"       {finding}")


    # A bump that a missing global prerequisite refuses, carried by the approval instead.
    approval_case(
        "a declared change under the owner's approval, prerequisites missing",
        synthetic_entry(1, old),
        approved,
        flipped,
        {path: "M", report: "A", approval_report: "A"},
        {report: b"x", approval_report: b"approved"},
        False,
    )
    approval_case(
        "the same change with no approval is refused",
        synthetic_entry(1, old),
        synthetic_entry(2, flipped, report=report),
        flipped,
        {path: "M", report: "A"},
        {report: b"x"},
        True,
    )
    approval_case(
        "an approval with an empty decision",
        synthetic_entry(1, old),
        dict(approved, approved=dict(approval, decision="   ")),
        flipped,
        {path: "M", report: "A", approval_report: "A"},
        {report: b"x", approval_report: b"approved"},
        True,
    )
    approval_case(
        "an approval with an empty date",
        synthetic_entry(1, old),
        dict(approved, approved=dict(approval, date="")),
        flipped,
        {path: "M", report: "A", approval_report: "A"},
        {report: b"x", approval_report: b"approved"},
        True,
    )
    approval_case(
        "an approval whose report does not exist",
        synthetic_entry(1, old),
        approved,
        flipped,
        {path: "M", report: "A", approval_report: "A"},
        {report: b"x"},
        True,
    )
    approval_case(
        "an approval whose report is not new in this change",
        synthetic_entry(1, old),
        approved,
        flipped,
        {path: "M", report: "A"},
        {report: b"x", approval_report: b"old"},
        True,
    )
    approval_case(
        "an approved bump that changes nothing",
        synthetic_entry(1, old),
        dict(synthetic_entry(2, old, report=report), approved=approval),
        old,
        {path: "M", report: "A", approval_report: "A"},
        {report: b"x", approval_report: b"approved"},
        True,
    )
    approval_case(
        "an approved bump without its version report",
        synthetic_entry(1, old),
        dict(synthetic_entry(2, flipped), approved=approval),
        flipped,
        {path: "M", approval_report: "A"},
        {approval_report: b"approved"},
        True,
    )
    approval_case(
        "an approval on a model with no anchor",
        synthetic_entry(1, old),
        approved,
        flipped,
        {path: "M", report: "A", approval_report: "A"},
        {report: b"x", approval_report: b"approved"},
        True,
        anchored=False,
    )
    approval_case(
        "a removed approval",
        dict(synthetic_entry(2, flipped, report=report), approved=approval),
        synthetic_entry(2, flipped, report=report),
        flipped,
        {},
        {report: b"x", approval_report: b"approved"},
        True,
        base_reference=flipped,
    )
    approval_case(
        "an approval added without a version bump",
        synthetic_entry(1, old),
        dict(synthetic_entry(1, old), approved=approval),
        old,
        {approval_report: "A"},
        {approval_report: b"approved"},
        True,
    )

    # Model "m"'s arm profile and the serving side's neutral list, against its one registered
    # configuration.
    profile = "crates/sparkflash-model-m/config/m.toml"
    renamed = "crates/sparkflash-model-m/config/m2.toml"
    evidence = "docs/numerics/m/neutral.md"

    def listed(*fields: str) -> bytes:
        return ("pub const NEUTRAL_FIELDS: &[&str] = &[\n" + "".join(f'    "{field}",\n' for field in fields) + "];\n").encode()

    shipped = b'model = "m"\n[runtime]\nconcurrency = 1\n[decode]\ntensor_cores = true\n'
    neutral = shipped.replace(b"concurrency = 1", b"concurrency = 4")
    flip = shipped.replace(b"tensor_cores = true", b"tensor_cores = false")
    nameless = shipped.replace(b'model = "m"\n', b"")
    widened = listed("runtime.concurrency", "decode.tensor_cores")
    base_extra = {profile: shipped, CLASSIFICATION: listed("runtime.concurrency")}
    # (name, head files over the base's, head reference, changed paths, refused)
    profile_cases = [
        ("a neutral profile change", {profile: neutral}, old, {profile: "M"}, False),
        ("an arithmetic profile change without a recapture", {profile: flip}, old, {profile: "M"}, True),
        ("an arithmetic profile change, recaptured to the same bits", {profile: flip}, old, {profile: "M", path: "M"}, False),
        ("an arithmetic profile change, recaptured to new bits", {profile: flip}, flipped, {profile: "M", path: "M"}, True),
        ("a profile that names no model", {profile: nameless}, old, {profile: "M"}, True),
        ("a profile renamed with every setting kept", {profile: None, renamed: shipped}, old, {profile: "D", renamed: "A"}, False),
        ("a profile renamed with an arithmetic change", {profile: None, renamed: flip}, old, {profile: "D", renamed: "A"}, True),
        ("a field made neutral without a report", {CLASSIFICATION: widened}, old, {CLASSIFICATION: "M"}, True),
        (
            "a field made neutral with a report that names it",
            {CLASSIFICATION: widened, evidence: b"`decode.tensor_cores` holds the bits: ..."},
            old,
            {CLASSIFICATION: "M", evidence: "A"},
            False,
        ),
    ]
    for name, head_extra, reference, changed, refused in profile_cases:
        registry = synthetic_registry([synthetic_entry(1, old)], False)
        head_registry = synthetic_registry([synthetic_entry(1, reference)], False)
        base_files = files(old, base_extra)
        head_files = files(reference, {**base_extra, **head_extra})
        findings = declaration_findings(registry, head_registry, base_files, head_files, changed)
        ok = bool(findings) == refused
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {'refused' if findings else 'passed'}")
        for finding in findings:
            print(f"       {finding}")
    # A revert is a change: the head names the base's arithmetic again and the version between
    # them is retired with its own bits. Without that record the bump reads as a non-change (and
    # as a jump of two versions), which is why GLM's version 3 could not pass the gate.
    middle_path = "references/q-v2.json"
    middle = synthetic_reference([5, 6], -0.20)
    approval = {
        "decision": "the body is slower in both pairs; back to the row GEMV",
        "date": "2026-11-19",
        "report": report,
    }

    def revert_case(name: str, with_retired: bool, refused: bool) -> None:
        nonlocal failed
        head_entry = dict(synthetic_entry(3, old, report=report), approved=approval)
        base = synthetic_registry([synthetic_entry(1, old)], True)
        head = synthetic_registry([head_entry], True)
        if with_retired:
            head["retired"].append(
                {
                    "model": "m",
                    "world": 4,
                    "profile": "reference",
                    "plan": "tp",
                    "arch": "sm_121a",
                    "version": 2,
                    "reason": "slower",
                    "reference": middle_path,
                    "fingerprint": middle["fingerprint"],
                }
            )
        extra = {report: b"x", f"{NUMERICS}/{middle_path}": render_reference(middle).encode()}
        findings = declaration_findings(base, head, files(old), files(old, extra), {report: "A"})
        ok = bool(findings) == refused
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {'refused' if findings else 'passed'}")
        for finding in findings:
            print(f"       {finding}")

    revert_case("a version 3 that returns to version 1's bits, with version 2 retired", True, False)
    revert_case("the same revert without the retired version", False, True)

    print("numerics self-test: " + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


# --- the command line -------------------------------------------------------------------------


def relative_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO / NUMERICS))
    except ValueError:
        return str(path)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser("capture", help="capture a reference from a running server")
    capture_parser.add_argument("--url", required=True)
    capture_parser.add_argument("--out", required=True, type=Path)
    capture_parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    capture_parser.add_argument("--forced-from", type=Path)
    capture_parser.add_argument("--max-tokens", type=int, default=128)
    capture_parser.add_argument("--top-logprobs", type=int, default=8)
    capture_parser.add_argument("--evidence", choices=EVIDENCE, default="logprobs")
    capture_parser.add_argument("--timeout", type=float, default=1800.0)
    capture_parser.add_argument(
        "--binary",
        type=Path,
        help="the server binary; its sha256 goes in the reference's server card (or set SPARKFLASH_BIN)",
    )
    capture_parser.add_argument(
        "--repo",
        type=Path,
        default=REPO,
        help="fallback tree when the served card names no commit; its HEAD commit goes in the server card",
    )
    capture_parser.add_argument(
        "--reference",
        action="store_true",
        help="this capture is a reference: it needs --floor and names its build",
    )
    capture_parser.add_argument(
        "--anchor",
        action="store_true",
        help="this capture is an anchor: it needs --floor and names its build",
    )
    capture_parser.add_argument(
        "--floor",
        type=Path,
        help="the correctness floor (forced comparison, known answers, own NLL) a reference or anchor needs",
    )

    check_parser = commands.add_parser("check", help="recapture the registered reference, bit for bit")
    check_parser.add_argument("--url", required=True)
    check_parser.add_argument("--reference", type=Path, help="check this file instead of the registry's entry")
    check_parser.add_argument("--timeout", type=float, default=1800.0)

    compare_parser = commands.add_parser("compare", help="score a forced capture against its anchor")
    compare_parser.add_argument("--anchor", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--floor", type=Path, help="a compare --json result to set beside it")
    compare_parser.add_argument("--json", type=Path)
    compare_parser.add_argument("--markdown", type=Path)

    coverage_parser = commands.add_parser("coverage", help="judge a coverage dump against a model's expert layout")
    coverage_parser.add_argument("--dump", required=True, type=Path)
    coverage_parser.add_argument("--layout", type=Path, help="the model's expert layout; looked up by model by default")
    coverage_parser.add_argument(
        "--min-inputs",
        required=True,
        type=int,
        help="the coverage suite's declared input count; a dump must name at least this many",
    )

    gaps_parser = commands.add_parser(
        "coverage-gaps", help="report the routed experts a coverage dump has not reached"
    )
    gaps_parser.add_argument("--dump", required=True, type=Path)
    gaps_parser.add_argument("--layout", type=Path, help="the model's expert layout; looked up by model by default")
    gaps_parser.add_argument("--json", type=Path, help="write the full missing-expert report here")

    sweep_parser = commands.add_parser(
        "coverage-sweep", help="measure a corpus and greedily select the next coverage inputs"
    )
    sweep_parser.add_argument("--url", required=True)
    sweep_parser.add_argument("--dump", required=True, type=Path)
    sweep_parser.add_argument("--layout", type=Path, help="the model's expert layout; looked up by model by default")
    sweep_parser.add_argument("--corpus", type=Path, default=DEFAULT_COVERAGE_CORPUS)
    sweep_parser.add_argument(
        "--input-suite",
        type=Path,
        help="the suite that produced the dump's existing inputs; declares their content to the sweep",
    )
    sweep_parser.add_argument(
        "--seen",
        type=Path,
        help="the prompts already sent for this dump (default: DUMP.seen.json)",
    )
    sweep_parser.add_argument(
        "--generate", type=int, default=0, help="append N generated prompts, deterministic from --seed"
    )
    sweep_parser.add_argument("--seed", type=int, default=0)
    sweep_parser.add_argument("--order", choices=("shuffled", "corpus"), default="shuffled")
    sweep_parser.add_argument(
        "--batch", type=int, default=32, help="report marginal coverage every N sent prompts"
    )
    sweep_parser.add_argument("--max-candidates", type=int, default=2048)
    sweep_parser.add_argument("--max-tokens", type=int, default=1)
    sweep_parser.add_argument("--timeout", type=float, default=1800.0)
    sweep_parser.add_argument("--out", required=True, type=Path, help="the selected next-batch suite")
    sweep_parser.add_argument("--json", type=Path, help="write the sweep report here")

    monitors_parser = commands.add_parser("monitors", help="compare a candidate monitor dump with the reference forward")
    monitors_parser.add_argument("--reference", required=True, type=Path)
    monitors_parser.add_argument("--candidate", required=True, type=Path)
    monitors_parser.add_argument("--layer-bar", required=True, type=float)
    monitors_parser.add_argument("--expert-bar", required=True, type=float)
    monitors_parser.add_argument("--json", type=Path)
    monitors_parser.add_argument("--markdown", type=Path)

    fingerprint_parser = commands.add_parser("fingerprint", help="print the fingerprint of reference files")
    fingerprint_parser.add_argument("files", nargs="+", type=Path)

    ci_parser = commands.add_parser("ci", help="the declaration check")
    ci_parser.add_argument("--base")

    commands.add_parser("self-test", help="the declaration check refuses what it exists for")

    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            return command_capture(args)
        if args.command == "check":
            return command_check(args)
        if args.command == "compare":
            return command_compare(args)
        if args.command == "coverage":
            return command_coverage(args)
        if args.command == "coverage-gaps":
            return command_coverage_gaps(args)
        if args.command == "coverage-sweep":
            return command_coverage_sweep(args)
        if args.command == "monitors":
            return command_monitors(args)
        if args.command == "fingerprint":
            for path in args.files:
                print(f"{fingerprint(load_reference(path))}  {path}")
            return 0
        if args.command == "ci":
            return run_ci(args.base)
        return self_test()
    except NumericsFailure as error:
        print(f"numerics {args.command}: {error}", file=sys.stderr)
        return 1


def progress(record: dict) -> None:
    count = len(record.get("tokens") or []) or len(record.get("text") or "")
    print(f"  {record['name']}: {count} {'tokens' if 'tokens' in record else 'characters'}, {record['finish_reason']}", file=sys.stderr)


def command_capture(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/")
    writing = args.reference or args.anchor
    if writing and args.floor is None:
        what = "an anchor" if args.anchor else "a reference"
        raise NumericsFailure(
            f"{what} needs its correctness floor: pass --floor FILE with the forced comparison, "
            "the known-answer decode and the reference's own NLL per token (docs/NUMERICS.md §5)"
        )
    if writing and args.forced_from is not None:
        raise NumericsFailure(
            "a reference or anchor is a free-running capture; --forced-from makes a comparison, "
            "not a reference"
        )
    binary = args.binary or (Path(os.environ["SPARKFLASH_BIN"]) if os.environ.get("SPARKFLASH_BIN") else None)
    forced_from = None
    if args.forced_from:
        anchor = load_reference(args.forced_from)
        if anchor["evidence"] != "logprobs":
            raise NumericsFailure(f"{args.forced_from}: a text reference has no token ids to force")
        prompts = [
            {"name": prompt["name"], "prompt_ids": prompt["prompt_ids"], "tokens": prompt["tokens"]}
            for prompt in anchor["prompts"]
        ]
        forced_from = {"reference": relative_reference(args.forced_from), "fingerprint": anchor["fingerprint"]}
        max_tokens = max(len(prompt["tokens"]) for prompt in prompts)
    else:
        card = fetch_card(url, args.timeout)
        prompts = suite_prompts(args.suite, url, card["id"], args.timeout)
        max_tokens = args.max_tokens
    reference = capture(
        url,
        prompts,
        args.evidence,
        max_tokens,
        args.top_logprobs,
        forced_from,
        args.timeout,
        progress,
        binary=binary,
        repo=args.repo,
    )
    if writing:
        try:
            floor = json.loads(args.floor.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise NumericsFailure(f"{args.floor}: {error}") from error
        registry = parse_registry((REPO / REGISTRY).read_text(), REGISTRY)
        reference["floor"] = validate_floor(
            floor, str(args.floor), floor_bar(registry, reference["key"]["model"])
        )
        problem = build_record_problem(reference)
        if problem:
            raise NumericsFailure(f"a reference or anchor must name its build: {problem}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_reference(reference))
    numerics = reference["server"]["numerics"]
    build = reference["server"]
    print(f"wrote {args.out}: {key_text(reference['key'])}, status {numerics['status']}")
    print(f"fingerprint {reference['fingerprint']}")
    if build["binary_sha256"] and build["commit"]:
        print(f"build: binary sha256 {build['binary_sha256']}, commit {build['commit']}")
        if build["build_note"]:
            print(f"warning: {build['build_note']}")
    else:
        print(f"build: not recorded ({build['build_note']})")
    if numerics.get("overrides"):
        print(f"warning: the server ran with {numerics['overrides']} set; this capture does not describe the shipped default")
    if forced_from is None:
        registered = lookup(parse_registry((REPO / REGISTRY).read_text(), REGISTRY), reference["key"])
        print("\nregistry entry (numerics/registry.toml):\n")
        print(entry_snippet(reference, relative_reference(args.out), registered))
        if registered and registered.get("fingerprint") not in (None, reference["fingerprint"]):
            print(
                f"\nthe bits differ from version {registered['version']}'s: unless this capture only "
                f"extends that one, it is version {registered['version'] + 1}, with its own report "
                "(docs/NUMERICS.md §2)"
            )
    return 0


def command_check(args: argparse.Namespace) -> int:
    url = args.url.rstrip("/")
    card = fetch_card(url, args.timeout)
    numerics = card["numerics"]
    key = card_key(card)
    if args.reference:
        reference = load_reference(args.reference)
        print(f"checking {key_text(key)} (status {numerics['status']}) against {args.reference}")
    else:
        registry = parse_registry((REPO / REGISTRY).read_text(), REGISTRY)
        entry = lookup(registry, key)
        if entry is None:
            raise NumericsFailure(f"{key_text(key)} has no registry entry; this configuration is unregistered")
        if "reference" not in entry:
            raise NumericsFailure(
                f"{key_text(key)} has no reference captured yet, nothing vouches for its bits"
            )
        if numerics["status"] != "registered":
            raise NumericsFailure(
                f"the server reports {key_text(key)} as {numerics['status']} "
                f"(overrides {numerics.get('overrides') or 'none'}); a check against the registry "
                "needs the shipped default (pass --reference to measure an override anyway)"
            )
        if numerics.get("fingerprint") != entry["fingerprint"] or numerics.get("registered_version") != entry["version"]:
            raise NumericsFailure(
                f"the server's binary embeds version {numerics.get('registered_version')} "
                f"({numerics.get('fingerprint')}), this checkout registers version {entry['version']} "
                f"({entry['fingerprint']}); rebuild from this checkout"
            )
        reference = load_reference(REPO / NUMERICS / entry["reference"])
        print(f"checking {key_text(key)} version {entry['version']} against {entry['reference']}")
    forced_from = reference.get("forced_from")
    prompts = [
        {"name": prompt["name"], "prompt_ids": prompt["prompt_ids"], "tokens": prompt.get("tokens")}
        for prompt in reference["prompts"]
    ]
    fresh = capture(
        url,
        prompts,
        reference["evidence"],
        reference["settings"]["max_tokens"],
        reference["settings"].get("top_logprobs", 0),
        forced_from,
        args.timeout,
        progress,
    )
    if fresh["fingerprint"] == reference["fingerprint"]:
        print(f"PASS {fresh['fingerprint']}")
        return 0
    print(f"FAIL {fresh['fingerprint']} != {reference['fingerprint']}")
    print(f"first difference: {first_difference(reference, fresh)}")
    return 1


def command_coverage(args: argparse.Namespace) -> int:
    dump = load_json(args.dump)
    model = dump.get("model")
    if not isinstance(model, str) or not model:
        raise NumericsFailure(f"{args.dump}: the dump names no model")
    registry_path = REPO / REGISTRY
    if not registry_path.is_file():
        raise NumericsFailure(f"{REGISTRY} does not exist; the coverage bar is stated there")
    registry = parse_registry(registry_path.read_text(), REGISTRY, require_bar=True)
    bar = coverage_bar(registry["prerequisites"].get("coverage_slots"), REGISTRY)
    layout, origin = find_layout(model, args.layout)
    result = judge_coverage(dump, layout, origin, args.min_inputs, bar)
    print(render_coverage(result), end="")
    return 0 if result["passes"] else 1


def command_coverage_gaps(args: argparse.Namespace) -> int:
    dump = load_json(args.dump)
    model = dump.get("model")
    if not isinstance(model, str) or not model:
        raise NumericsFailure(f"{args.dump}: the dump names no model")
    layout, origin = find_layout(model, args.layout)
    result = coverage_inspect(dump, layout, origin, 0)
    document = coverage_gaps_document(result, str(args.dump))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(render_coverage_gaps(result), end="")
    return 0


def command_coverage_sweep(args: argparse.Namespace) -> int:
    report, selected = sweep_coverage(args)
    suite = {
        "schema": SUITE_SCHEMA,
        "note": (
            "The prompts a coverage sweep measured and kept: each reached at least one routed expert "
            "slot that no earlier selected prompt reached. Run it with capture or as a coverage suite; "
            "the dump is the evidence, this suite is what reproduces its additions."
        ),
        "coverage": {
            "model": report["model"],
            "layout": report["layout"],
            "seed": report["seed"],
            "order": report["order"],
            "max_tokens": report["max_tokens"],
            "selected": report["selected"],
            "missing_after": report["missing_after"],
            "complete": report["complete"],
        },
        "prompts": [
            {key: entry[key] for key in ("name", "family", "prompt", "messages") if key in entry}
            for entry in selected
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(suite, indent=2, ensure_ascii=False) + "\n")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(render_coverage_sweep(report), end="")
    print(f"wrote {args.out}: {len(selected)} selected prompt(s)")
    return 0


def command_monitors(args: argparse.Namespace) -> int:
    reference = check_monitor_dump(load_json(args.reference), str(args.reference))
    candidate = check_monitor_dump(load_json(args.candidate), str(args.candidate))
    result = judge_monitors(reference, candidate, args.layer_bar, args.expert_bar)
    markdown = render_monitors(result)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown)
    print(markdown, end="")
    return 0 if result["pass"] else 1


def command_compare(args: argparse.Namespace) -> int:
    result = compare(load_reference(args.anchor), load_reference(args.candidate))
    floor = None
    if args.floor:
        floor = json.loads(args.floor.read_text())
        if floor.get("schema") != COMPARE_SCHEMA or floor["anchor"]["fingerprint"] != result["anchor"]["fingerprint"]:
            raise NumericsFailure(f"{args.floor}: not a comparison against the same anchor")
    markdown = render_comparison(result, floor)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown)
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
