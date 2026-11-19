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
                        [--evidence logprobs|text]
    numerics.py check --url URL [--reference FILE]
    numerics.py compare --anchor REFERENCE --candidate CAPTURE [--floor COMPARE.json]
                        [--json OUT] [--markdown OUT]
    numerics.py coverage --dump FILE [--layout FILE] [--min-inputs N]
    numerics.py monitors --reference REF.json --candidate CAND.json
                        --layer-bar X --expert-bar Y [--json OUT] [--markdown OUT]
    numerics.py fingerprint FILE...
    numerics.py ci [--base REV]
    numerics.py self-test

capture  runs a prompt suite against a running server and writes a reference: every prompt's ids,
         and every generated token's id, logprob and top-k alternatives (evidence "logprobs"), or
         the greedy text where the path serves no logprobs (evidence "text"). --forced-from feeds
         another reference's tokens through the server instead (teacher forcing), which is what
         compare reads.
check    recaptures the server's registered reference and fails on any bit that differs. It also
         fails when the server does not report the registry entry this checkout has.
compare  scores a teacher-forced capture against the anchor it was forced from: per-token NLL
         delta, top-1 agreement and a KL lower bound. --floor puts another comparison (the TP=1 to
         TP=4 divergence) beside it.
coverage judges a coverage dump against the model's expert layout: every decoder layer present,
         one hit count per routed expert, every routed expert positive, and at least the declared
         number of inputs.
monitors compares a candidate monitor dump with the reference forward, per layer and per expert,
         and fails every relative-L2 bar it exceeds.
ci       the declaration check: a changed reference without a version bump fails, a bump without a
         change fails, a bump while a prerequisite is false fails, an anchor never changes, every
         fingerprint must match its file, an arm profile's arithmetic setting changes only with
         every reference of the model it names recaptured, and a field joins the neutral list only
         with a report.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
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

REFERENCE_SCHEMA = "sparkflash-numerics-reference/1"
SUITE_SCHEMA = "sparkflash-numerics-prompts/1"
COMPARE_SCHEMA = "sparkflash-numerics-compare/1"
COVERAGE_SCHEMA = "sparkflash-numerics-coverage/1"
MONITORS_SCHEMA = "sparkflash-numerics-monitors/1"
MONITOR_RESULT_SCHEMA = "sparkflash-numerics-monitor-result/1"
LAYOUT_SCHEMA = "sparkflash-numerics-expert-layout/1"
LAYOUTS = f"{NUMERICS}/layouts/"
PREREQUISITES = ("anchor", "declaration_check", "coverage", "monitors")
# The prerequisites that gate every version bump. 'anchor' is not here: it is a per-model
# requirement (docs/NUMERICS.md section 5), looked up for the bumping configuration's own model.
GATING_PREREQUISITES = ("declaration_check", "coverage", "monitors")
KEY_FIELDS = ("model", "world", "profile", "plan", "arch")
EVIDENCE = ("logprobs", "text")
# A text reference holds no token ids and no logprobs, so it cannot be teacher-forced. A
# configuration with text evidence is exempt from the per-model anchor requirement; it also
# cannot bump, because there is no comparison that would justify the change.
ANCHOR_EXEMPT_EVIDENCE = ("text",)
TOKEN_ID_PREFIX = "token_id:"
# The arm profiles (docs/adding-a-model.md §3): the defaults each arm serves. Each names the
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
    reference = {
        "schema": REFERENCE_SCHEMA,
        "evidence": evidence,
        "key": card_key(card),
        "server": {
            "numerics": card["numerics"],
            "build": card.get("build"),
            "checkpoint": checkpoint_identity(card.get("checkpoint")),
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


def parse_registry(text: str, origin: str) -> dict:
    try:
        registry = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise NumericsFailure(f"{origin}: {error}") from error
    prerequisites = registry.get("prerequisites") or {}
    missing = [name for name in PREREQUISITES if not isinstance(prerequisites.get(name), bool)]
    if missing:
        raise NumericsFailure(f"{origin}: prerequisites {missing} are not set to true or false")
    for table in ("config", "anchor", "retired"):
        registry.setdefault(table, [])
    return registry


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
    for path, status in sorted(changed.items()):
        if path.startswith(REFERENCES) and status != "D" and path not in named:
            findings.append(f"{path}: no registry entry names this reference; evidence for a report goes under {CAPTURES}")
        if path.startswith(CAPTURES) and status != "A":
            findings.append(f"{path}: a committed capture is evidence for a report and is never edited or deleted")

    base_registry = base or {"config": [], "anchor": [], "retired": [], "prerequisites": {}}
    base_missing = missing_prerequisites(base) if base else list(PREREQUISITES)
    head_missing = missing_prerequisites(head)

    # A prerequisite turns on only with what it names in place.
    if head["prerequisites"]["anchor"] and not head["anchor"]:
        findings.append("prerequisites.anchor is true, but the registry holds no [[anchor]]")

    # Anchors are frozen: never edited, never removed, never their file rewritten.
    head_anchors = {anchor["model"]: anchor for anchor in head["anchor"]}
    for anchor in base_registry["anchor"]:
        mine = head_anchors.get(anchor["model"])
        if mine is None:
            findings.append(f"anchor {anchor['model']}: removed; an anchor is frozen")
        elif mine != anchor:
            findings.append(f"anchor {anchor['model']}: edited; an anchor is frozen")
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
        if bumped and entry["version"] != old["version"] + 1:
            findings.append(f"{name}: version {old['version']} -> {entry['version']}; one declared change is one version")
        if entry["evidence"] != old["evidence"] and not bumped:
            findings.append(f"{name}: evidence {old['evidence']} -> {entry['evidence']} without a version bump")
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
        if not bits_changed:
            findings.append(
                f"{name}: declared non-change: version {entry['version']} names the same bits as "
                f"version {old['version']}; a version is bumped only by a change"
            )
        gating_missing = [
            prerequisite
            for prerequisite in GATING_PREREQUISITES
            if prerequisite in base_missing or prerequisite in head_missing
        ]
        if gating_missing:
            findings.append(
                f"{name}: version {entry['version']} declares an arithmetic change, but "
                f"prerequisites {gating_missing} are not in place "
                "(docs/NUMERICS.md §5); until they are, the gate is bit-exact"
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
                '`model = "<registry id>"` beside its family and weights (docs/adding-a-model.md §3)'
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


def run_ci(base_argument: str | None) -> int:
    base = resolve_base(base_argument)
    read_base, read_head = read_at(base), read_at(None)
    base_text = read_base(REGISTRY)
    base_registry = parse_registry(base_text.decode(), f"{base}:{REGISTRY}") if base_text else None
    head_text = read_head(REGISTRY)
    if head_text is None:
        raise NumericsFailure(f"{REGISTRY} does not exist")
    head_registry = parse_registry(head_text.decode(), REGISTRY)
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
    anchor_checks, coverage = anchor_findings(head_registry, read_head, capture_paths)
    findings += anchor_checks
    configs, anchors = len(head_registry["config"]), len(head_registry["anchor"])
    missing = missing_prerequisites(head_registry)
    print(
        f"numerics ci: base {base[:12]}, {configs} configuration(s), {anchors} anchor(s), "
        f"{len(changed)} changed path(s); prerequisites missing: {missing or 'none'}"
    )
    print(
        "numerics ci: anchor coverage: "
        + (
            "covered " + ", ".join(sorted(coverage["covered"]))
            if coverage["covered"]
            else "no logprobs model is anchored"
        )
        + (
            "; exempt (text evidence) " + ", ".join(coverage["exempt"])
            if coverage["exempt"]
            else ""
        )
        + (
            "; not covered " + ", ".join(sorted(coverage["uncovered"]))
            if coverage["uncovered"]
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


def judge_coverage(dump: dict, layout: dict, layout_origin: str, min_inputs: int) -> dict:
    """Judge one coverage dump against a model's expert layout.

    Every decoder layer must appear once, with one hit count per routed expert; a dense layer has
    an empty hit list. Every routed expert of every layer must have a positive count, and the dump
    must name at least the declared number of inputs.
    """
    if dump.get("schema") != COVERAGE_SCHEMA:
        raise NumericsFailure(f"coverage dump: schema {dump.get('schema')!r} is not {COVERAGE_SCHEMA}")
    if not isinstance(dump.get("model"), str) or not dump["model"]:
        raise NumericsFailure("coverage dump: the dump names no model")
    if min_inputs < 1:
        raise NumericsFailure(f"--min-inputs {min_inputs}: a coverage run needs at least one input")
    expected = layout_experts(layout, layout_origin)
    if dump["model"] != layout.get("model"):
        raise NumericsFailure(
            f"coverage dump: the dump names {dump['model']!r}, but {layout_origin} names "
            f"{layout.get('model')!r}"
        )
    inputs = dump.get("inputs")
    if not isinstance(inputs, list) or not inputs:
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
            covered.append({"layer": number, "experts": 0, "hits": 0, "min": None, "max": None})
            continue
        if len(hits) != experts:
            raise NumericsFailure(
                f"coverage dump: layer {number} has {len(hits)} hit count(s) for {experts} routed "
                "experts; the count must match the layout exactly"
            )
        missing = [expert for expert, hit in enumerate(hits) if hit == 0]
        if missing:
            shown = ", ".join(str(expert) for expert in missing[:8])
            more = "" if len(missing) <= 8 else f" and {len(missing) - 8} more"
            raise NumericsFailure(
                f"coverage dump: layer {number} never reaches expert(s) {shown}{more}; the coverage "
                "inputs must reach every routed expert of every layer"
            )
        total_experts += experts
        covered.append(
            {"layer": number, "experts": experts, "hits": sum(hits), "min": min(hits), "max": max(hits)}
        )
    return {
        "schema": COVERAGE_SCHEMA,
        "model": dump["model"],
        "inputs": len(inputs),
        "layers": covered,
        "experts": total_experts,
        "layout": layout_origin,
    }


def render_coverage(result: dict) -> str:
    lines = [
        f"coverage: {result['model']}, {len(result['layers'])} layer(s), "
        f"{result['experts']} routed expert(s), {result['inputs']} input(s); layout {result['layout']}",
        "",
        "| layer | routed experts | hits | min | max |",
        "| --: | --: | --: | --: | --: |",
    ]
    for layer in result["layers"]:
        if layer["experts"] == 0:
            lines.append(f"| {layer['layer']} | 0 (dense) | - | - | - |")
        else:
            lines.append(
                f"| {layer['layer']} | {layer['experts']} | {layer['hits']} | {layer['min']} | {layer['max']} |"
            )
    lines += [
        "",
        f"PASS coverage: every routed expert of every layer has a positive hit count over "
        f"{result['inputs']} input(s).",
    ]
    return "\n".join(lines) + "\n"


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
    """Compare a candidate forward's sample with the reference forward's, per layer and expert."""
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
    failures, results = [], []
    for number in sorted(reference_layers):
        left, right = reference_layers[number], candidate_layers[number]
        layer_error = relative_error(left["values"], right["values"], f"layer {number}")
        layer_ok = layer_error <= layer_bar
        if not layer_ok:
            failures.append(f"layer {number}: {layer_error:.6e} > layer bar {layer_bar:.6e}")
        reference_experts = {expert["expert"]: expert for expert in left["experts"]}
        candidate_experts = {expert["expert"]: expert for expert in right["experts"]}
        if reference_experts.keys() != candidate_experts.keys():
            raise NumericsFailure(
                f"layer {number}: the two dumps do not hold the same experts: "
                f"reference {sorted(reference_experts)}, candidate {sorted(candidate_experts)}"
            )
        experts, worst = [], None
        for expert in sorted(reference_experts):
            error = relative_error(
                reference_experts[expert]["values"],
                candidate_experts[expert]["values"],
                f"layer {number} expert {expert}",
            )
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
        "layers": results,
        "failures": failures,
        "pass": not failures,
    }


def render_monitors(result: dict) -> str:
    lines = [
        f"monitors: {result['model']}, probe {result['probe']['name']!r}, "
        f"{len(result['layers'])} layer(s), sample {len(result['sample']['indices'])} of "
        f"{result['sample']['width']} values",
        "",
        "| layer | relative error | layer bar | worst expert | worst expert error | expert bar |",
        "| --: | --: | --: | --: | --: | --: |",
    ]
    for layer in result["layers"]:
        worst = layer["worst_expert"]
        worst_text = "-" if worst is None else f"{worst['expert']}"
        worst_error = "-" if worst is None else f"{worst['rel_error']:.3e}"
        lines.append(
            f"| {layer['layer']} | {layer['rel_error']:.3e} | {result['bars']['layer']:.3e} | "
            f"{worst_text} | {worst_error} | {result['bars']['expert']:.3e} |"
        )
    lines.append("")
    if result["pass"]:
        lines.append(
            f"PASS monitors: every layer is within {result['bars']['layer']:.3e} and every routed "
            f"expert within {result['bars']['expert']:.3e} of the reference forward."
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
        "prerequisites": {name: prerequisites for name in PREREQUISITES},
        "config": entries,
        "anchor": anchors,
        "retired": [],
    }


def synthetic_entry(version: int, reference: dict, path: str = "references/q.json", report: str | None = None) -> dict:
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
        (
            "a declared change with its report",
            True,
            synthetic_entry(2, flipped, report=report),
            flipped,
            {path: "M", report: "A"},
            {report: b"x"},
            False,
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
    coverage_parser.add_argument("--min-inputs", type=int, default=1, help="the coverage suite's input count")

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
    reference = capture(url, prompts, args.evidence, max_tokens, args.top_logprobs, forced_from, args.timeout, progress)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_reference(reference))
    numerics = reference["server"]["numerics"]
    print(f"wrote {args.out}: {key_text(reference['key'])}, status {numerics['status']}")
    print(f"fingerprint {reference['fingerprint']}")
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
    layout, origin = find_layout(model, args.layout)
    result = judge_coverage(dump, layout, origin, args.min_inputs)
    print(render_coverage(result), end="")
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
