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

"""Tests for the numerics declaration tool."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import random
import struct
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from unittest import mock


SCRIPT = Path(__file__).resolve().parent.parent / "numerics.py"
SPEC = importlib.util.spec_from_file_location("numerics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
NUMERICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NUMERICS)

MODEL = "toy/numerics"
VOCAB = 50


def next_ulp(value: float) -> float:
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", value))[0] + 1))[0]


class FakeServer:
    """A server whose logprobs are a pure function of the fed prefix, like a deterministic model.

    `shift` moves every logit of one token, `ignore_forced` answers its own choice instead of
    the forced tokens, and `forced_decoding` is what the card advertises. A request without a
    zero temperature samples, as the API does at its default of 1.0.
    """

    def __init__(self, shift: float = 0.0, ignore_forced: bool = False, forced_decoding: bool = True):
        self.shift = shift
        self.ignore_forced = ignore_forced
        self.forced_decoding = forced_decoding
        self.requests: list[dict] = []

    def row(self, prefix: list[int]) -> list[float]:
        seed = sum((index + 1) * token for index, token in enumerate(prefix))
        logits = [math.sin(seed * 0.37 + token * 1.3) * 3.0 for token in range(VOCAB)]
        logits[7] += self.shift
        top = max(logits)
        total = math.log(sum(math.exp(value - top) for value in logits)) + top
        # The server's f32 arithmetic: every logprob it reports is an f32.
        return [struct.unpack("<f", struct.pack("<f", value - total))[0] for value in logits]

    def complete(self, body: dict) -> dict:
        self.requests.append(body)
        prefix = list(body["prompt"])
        forced = body.get("forced_tokens")
        count = len(forced) if forced is not None else body["max_tokens"]
        tokens, logprobs, top = [], [], []
        for position in range(count):
            row = self.row(prefix)
            if body.get("temperature", 1.0) > 0:
                choice = random.randrange(VOCAB)
            else:
                choice = max(range(VOCAB), key=lambda token: (row[token], -token))
            token = forced[position] if forced is not None and not self.ignore_forced else choice
            tokens.append(token)
            logprobs.append(row[token])
            ranked = sorted(range(VOCAB), key=lambda candidate: (-row[candidate], candidate))
            alternatives = {token: row[token]}
            for candidate in ranked[: body.get("logprobs", 0)]:
                alternatives.setdefault(candidate, row[candidate])
            top.append({f"token_id:{candidate}": value for candidate, value in alternatives.items()})
            prefix.append(token)
        text = "".join(chr(97 + token % 26) for token in tokens)
        choice = {"index": 0, "text": text, "finish_reason": "length"}
        if "logprobs" in body:
            choice["logprobs"] = {
                "tokens": [f"token_id:{token}" for token in tokens],
                "token_logprobs": logprobs,
                "top_logprobs": top,
            }
        return {"choices": [choice]}

    def card(self) -> dict:
        return {
            "id": MODEL,
            "capabilities": {"forced_decoding": True} if self.forced_decoding else {},
            "build": {"version": "0", "id": "test"},
            "checkpoint": "/home/someone/.cache/huggingface/hub/models--org--name/snapshots/abc123",
            "numerics": {
                "status": "unregistered",
                "profile": "reference",
                "version": None,
                "registered_version": None,
                "fingerprint": None,
                "evidence": None,
                "world": 4,
                "plan": "tp",
                "arch": "sm_121a",
                "overrides": [],
                "policy": "docs/NUMERICS.md",
            },
        }


@contextlib.contextmanager
def serving(fake: FakeServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def answer(self, value: dict) -> None:
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.answer({"object": "list", "data": [fake.card()]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if self.path == "/tokenize":
                text = body.get("prompt") or json.dumps(body.get("messages"))
                self.answer({"tokens": [1] + [ord(char) % VOCAB for char in text][:12]})
            else:
                self.answer(fake.complete(body))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def capture(url: str, forced_from: dict | None = None, prompts: list | None = None, max_tokens: int = 6) -> dict:
    prompts = prompts or [{"name": "a", "prompt_ids": [1, 2, 3]}, {"name": "b", "prompt_ids": [4, 5]}]
    return NUMERICS.capture(url, prompts, "logprobs", max_tokens, 4, forced_from, 30.0)


class FingerprintTests(unittest.TestCase):
    def test_one_ulp_of_one_logprob_changes_the_fingerprint(self):
        reference = NUMERICS.synthetic_reference([5, 6], -0.25)
        nudged = json.loads(json.dumps(reference))
        nudged["prompts"][0]["logprobs"][1] = next_ulp(-0.25)
        self.assertNotEqual(NUMERICS.fingerprint(reference), NUMERICS.fingerprint(nudged))

    def test_names_dates_text_and_the_server_card_are_not_bits(self):
        reference = NUMERICS.synthetic_reference([5, 6], -0.25)
        renamed = json.loads(json.dumps(reference))
        renamed["prompts"][0]["name"] = "renamed"
        renamed["prompts"][0]["text"] = "other"
        renamed["captured"] = "2030-01-01"
        renamed["server"] = {"build": {"id": "elsewhere"}}
        renamed["settings"]["max_tokens"] = 99
        self.assertEqual(NUMERICS.fingerprint(reference), NUMERICS.fingerprint(renamed))

    def test_the_shortest_f32_decimal_round_trips_to_the_same_bits(self):
        value = struct.unpack("<f", struct.pack("<f", -1.2345678))[0]
        decimal = float(f"{value:.9g}")
        self.assertEqual(NUMERICS.f32_bits(value), NUMERICS.f32_bits(decimal))

    def test_the_file_form_parses_back_to_the_same_document(self):
        reference = NUMERICS.synthetic_reference([5, 6, 7], -0.5)
        rendered = NUMERICS.render_reference(reference)
        self.assertEqual(json.loads(rendered), reference)
        self.assertEqual(rendered.count("\n    {"), len(reference["prompts"]), "one prompt per line")


class ExtensionTests(unittest.TestCase):
    def test_a_longer_generation_with_the_same_prefix_extends(self):
        old = NUMERICS.synthetic_reference([5, 6], -0.25)
        new = NUMERICS.synthetic_reference([5, 6, 7], -0.25)
        self.assertTrue(NUMERICS.extends(old, new))
        self.assertFalse(NUMERICS.extends(new, old), "a shorter capture loses bits")

    def test_a_changed_prefix_does_not_extend(self):
        old = NUMERICS.synthetic_reference([5, 6], -0.25)
        new = NUMERICS.synthetic_reference([5, 6, 7], next_ulp(-0.25))
        self.assertFalse(NUMERICS.extends(old, new))

    def test_a_generation_that_stopped_must_stay_identical(self):
        old = NUMERICS.synthetic_reference([5, 6], -0.25, finish="stop")
        new = NUMERICS.synthetic_reference([5, 6, 7], -0.25)
        self.assertFalse(NUMERICS.extends(old, new))

    def test_a_changed_finish_reason_at_the_same_length_does_not_extend(self):
        old = NUMERICS.synthetic_reference([5, 6], -0.25)
        stopped = NUMERICS.synthetic_reference([5, 6], -0.25, finish="stop")
        self.assertNotEqual(old["fingerprint"], stopped["fingerprint"])
        self.assertFalse(NUMERICS.extends(old, stopped))
        self.assertTrue(NUMERICS.extends(old, NUMERICS.synthetic_reference([5, 6, 7], -0.25, finish="stop")))

    def test_an_added_prompt_extends(self):
        old = NUMERICS.synthetic_reference([5, 6], -0.25)
        new = json.loads(json.dumps(old))
        new["prompts"].append(dict(new["prompts"][0], name="q", prompt_ids=[3]))
        self.assertTrue(NUMERICS.extends(old, new))


class CaptureTests(unittest.TestCase):
    def test_a_capture_records_ids_logprobs_and_the_server_key(self):
        fake = FakeServer()
        with serving(fake) as url:
            reference = capture(url)
        self.assertEqual(reference["key"], {"model": MODEL, "world": 4, "profile": "reference", "plan": "tp", "arch": "sm_121a"})
        self.assertEqual(reference["fingerprint"], NUMERICS.fingerprint(reference))
        first = reference["prompts"][0]
        self.assertEqual(len(first["tokens"]), 6)
        self.assertEqual(first["top"][0][0][0], first["tokens"][0], "the own token leads its top list")
        body = fake.requests[0]
        self.assertEqual(body["temperature"], 0)
        self.assertIs(body["return_tokens_as_token_ids"], True)
        self.assertNotIn("forced_tokens", body)
        # The weights' identity, not the serving host's home directory.
        self.assertEqual(reference["server"]["checkpoint"], "models--org--name/snapshots/abc123")
        self.assertEqual(NUMERICS.checkpoint_identity("/data/checkpoints/glm"), "glm")

    def test_a_text_capture_is_greedy_and_reproduces(self):
        fake = FakeServer()
        prompts = [{"name": "a", "prompt_ids": [1, 2]}, {"name": "b", "prompt_ids": [3]}]
        with serving(fake) as url:
            first, second = (NUMERICS.capture(url, prompts, "text", 40, 0, None, 30.0) for _ in range(2))
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        # Greedy, and nothing the DeepSeek device path does not serve.
        self.assertEqual(fake.requests[0], {"model": MODEL, "prompt": [1, 2], "temperature": 0, "max_tokens": 40})
        self.assertEqual(len(first["prompts"][0]["text"]), 40)

    def test_the_same_server_reproduces_its_fingerprint_and_a_shifted_one_does_not(self):
        with serving(FakeServer()) as url:
            first, second = capture(url), capture(url)
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        with serving(FakeServer(shift=1e-3)) as url:
            shifted = capture(url)
        self.assertNotEqual(first["fingerprint"], shifted["fingerprint"])
        self.assertIn("prompt 'a' position 0", NUMERICS.first_difference(first, shifted))

    def test_a_forced_capture_feeds_the_anchors_tokens(self):
        with serving(FakeServer()) as url:
            anchor = capture(url)
        forced = {"reference": "references/anchor.json", "fingerprint": anchor["fingerprint"]}
        prompts = [dict(prompt) for prompt in anchor["prompts"]]
        fake = FakeServer(shift=0.5)
        with serving(fake) as url:
            candidate = capture(url, forced, prompts)
        self.assertEqual([p["tokens"] for p in candidate["prompts"]], [p["tokens"] for p in anchor["prompts"]])
        self.assertEqual(fake.requests[0]["forced_tokens"], anchor["prompts"][0]["tokens"])
        self.assertNotIn("max_tokens", fake.requests[0], "the forced length is the budget")

    def test_a_server_that_ignores_the_forced_tokens_is_refused(self):
        anchor_prompts = [{"name": "a", "prompt_ids": [1, 2, 3], "tokens": [9, 9, 9]}]
        with serving(FakeServer(ignore_forced=True)) as url:
            with self.assertRaisesRegex(NUMERICS.NumericsFailure, "other than the forced ones"):
                capture(url, {"reference": "x", "fingerprint": "f"}, anchor_prompts)

    def test_a_server_without_forced_decoding_is_refused(self):
        anchor_prompts = [{"name": "a", "prompt_ids": [1], "tokens": [9]}]
        with serving(FakeServer(forced_decoding=False)) as url:
            with self.assertRaisesRegex(NUMERICS.NumericsFailure, "forced_decoding"):
                capture(url, {"reference": "x", "fingerprint": "f"}, anchor_prompts)

    def test_a_card_without_a_numerics_identity_is_refused(self):
        fake = FakeServer()
        fake.card = lambda: {"id": MODEL, "capabilities": {}}
        with serving(fake) as url:
            with self.assertRaisesRegex(NUMERICS.NumericsFailure, "no numerics identity"):
                capture(url)

    def test_the_suite_is_tokenized_by_the_server(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite.json"
            suite.write_text(json.dumps({
                "schema": NUMERICS.SUITE_SCHEMA,
                "prompts": [
                    {"name": "chat", "messages": [{"role": "user", "content": "hi"}]},
                    {"name": "raw", "prompt": "hello"},
                    {"name": "ids", "prompt_ids": [3, 4]},
                ],
            }))
            with serving(FakeServer()) as url:
                prompts = NUMERICS.suite_prompts(suite, url, MODEL, 30.0)
        self.assertEqual([prompt["name"] for prompt in prompts], ["chat", "raw", "ids"])
        self.assertEqual(prompts[2]["prompt_ids"], [3, 4])
        self.assertEqual(prompts[1]["prompt_ids"][0], 1)


class CompareTests(unittest.TestCase):
    def setUp(self):
        with serving(FakeServer()) as url:
            self.anchor = capture(url, max_tokens=12)
        self.forced = {"reference": "references/anchor.json", "fingerprint": self.anchor["fingerprint"]}
        self.prompts = [dict(prompt) for prompt in self.anchor["prompts"]]

    def forced_capture(self, shift: float) -> dict:
        with serving(FakeServer(shift=shift)) as url:
            return capture(url, self.forced, self.prompts)

    def test_the_anchor_against_itself_scores_zero(self):
        summary = NUMERICS.compare(self.anchor, self.forced_capture(0.0))["summary"]
        self.assertEqual(summary["tokens"], 24)
        self.assertEqual(summary["abs_delta_nll_max"], 0.0)
        self.assertEqual(summary["top1_agreement"], 1.0)
        self.assertEqual(summary["kl_lower_bound_max"], 0.0)

    def test_a_larger_perturbation_scores_further_from_the_anchor(self):
        small = NUMERICS.compare(self.anchor, self.forced_capture(0.05))["summary"]
        large = NUMERICS.compare(self.anchor, self.forced_capture(2.0))["summary"]
        self.assertGreater(small["abs_delta_nll_mean"], 0.0)
        self.assertGreater(large["abs_delta_nll_mean"], small["abs_delta_nll_mean"])
        self.assertGreater(large["kl_lower_bound_mean"], small["kl_lower_bound_mean"])
        self.assertLessEqual(large["top1_agreement"], small["top1_agreement"])

    def test_the_kl_bound_is_the_kl_of_the_coarse_partition(self):
        anchor_top = [[1, math.log(0.5)], [2, math.log(0.3)]]
        candidate_top = [[1, math.log(0.4)], [3, math.log(0.4)]]
        score = NUMERICS.position_scores(math.log(0.5), anchor_top, math.log(0.4), candidate_top)
        # Partition {1, everything else}: p = (0.5, 0.5), q = (0.4, 0.6).
        expected = 0.5 * math.log(0.5 / 0.4) + 0.5 * math.log(0.5 / 0.6)
        self.assertAlmostEqual(score["kl_lower_bound"], expected, places=12)
        self.assertAlmostEqual(score["delta_nll"], math.log(0.5) - math.log(0.4), places=12)
        self.assertTrue(score["agree"], "token 1 leads the anchor, and the candidate's tie breaks to 1")
        swapped = [[1, math.log(0.3)], [3, math.log(0.5)]]
        self.assertFalse(NUMERICS.position_scores(math.log(0.5), anchor_top, math.log(0.3), swapped)["agree"])

    def test_a_null_logprob_is_refused_not_scored(self):
        candidate = self.forced_capture(0.0)
        candidate["prompts"][1]["top"][3][0][1] = None
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "position 3: the candidate holds a null"):
            NUMERICS.compare(self.anchor, candidate)
        anchor = json.loads(json.dumps(self.anchor))
        anchor["prompts"][0]["logprobs"][0] = None
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "position 0: the anchor holds a null"):
            NUMERICS.compare(anchor, self.forced_capture(0.0))

    def test_a_free_running_candidate_is_refused(self):
        with serving(FakeServer(shift=0.5)) as url:
            free = capture(url, max_tokens=12)
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "--forced-from"):
            NUMERICS.compare(self.anchor, free)

    def test_the_floor_column_judges_each_barred_metric(self):
        result = NUMERICS.compare(self.anchor, self.forced_capture(0.05))
        floor = NUMERICS.compare(self.anchor, self.forced_capture(2.0))
        markdown = NUMERICS.render_comparison(result, floor)
        self.assertIn("| mean absolute NLL delta |", markdown)
        self.assertNotIn("**no**", markdown.split("top-1 agreement")[0])
        self.assertIn("**no**", NUMERICS.render_comparison(floor, result))


TOY_REGISTRY = """schema = 1
[prerequisites]
anchor = false
declaration_check = true
coverage = false
monitors = false
"""


class DeclarationTests(unittest.TestCase):
    def setUp(self):
        self.old = NUMERICS.synthetic_reference([5, 6], -0.25)
        self.path = "numerics/references/q.json"

    def store(self, reference: dict, **extra: bytes):
        files = {self.path: NUMERICS.render_reference(reference).encode(), **extra}
        return files.get

    def test_the_self_test_passes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(NUMERICS.self_test(), 0)

    def test_a_removed_configuration_needs_a_retired_entry(self):
        base = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        head = NUMERICS.synthetic_registry([], False)
        findings = NUMERICS.declaration_findings(base, head, self.store(self.old), self.store(self.old), {})
        self.assertTrue(any("without a [[retired]]" in finding for finding in findings), findings)
        retired = dict(NUMERICS.synthetic_entry(1, self.old))
        head["retired"] = [{**{key: retired[key] for key in NUMERICS.KEY_FIELDS}, "version": 1, "reason": "gone"}]
        findings = NUMERICS.declaration_findings(base, head, self.store(self.old), self.store(self.old), {})
        self.assertEqual(findings, [])

    def test_an_anchor_is_frozen(self):
        base = NUMERICS.synthetic_registry([], True)
        anchor_path = NUMERICS.ANCHOR_PATH
        anchor_file = NUMERICS.render_reference(NUMERICS.synthetic_reference([9], -0.5)).encode()
        read = {anchor_path: anchor_file}.get
        head = json.loads(json.dumps(base))
        head["anchor"][0]["note"] = "edited"
        findings = NUMERICS.declaration_findings(base, head, read, read, {})
        self.assertTrue(any("anchor m: edited" in finding for finding in findings), findings)
        findings = NUMERICS.declaration_findings(base, base, read, read, {anchor_path: "M"})
        self.assertTrue(any("anchor's reference changed" in finding for finding in findings), findings)
        head["anchor"] = []
        head["prerequisites"]["anchor"] = False
        findings = NUMERICS.declaration_findings(base, head, read, read, {})
        self.assertTrue(any("removed" in finding for finding in findings), findings)

    def test_a_reference_no_entry_names_is_refused_and_captures_are_append_only(self):
        base = head = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        read = self.store(self.old)
        changed = {"numerics/references/stray.json": "A", "numerics/captures/x.json": "M"}
        findings = NUMERICS.declaration_findings(base, head, read, read, changed)
        self.assertTrue(any("stray.json: no registry entry" in finding for finding in findings), findings)
        self.assertTrue(any("captures/x.json: a committed capture" in finding for finding in findings), findings)
        findings = NUMERICS.declaration_findings(base, head, read, read, {"numerics/captures/y.json": "A"})
        self.assertEqual(findings, [])

    def test_a_version_moves_by_one(self):
        flipped = NUMERICS.synthetic_reference([5, 6], next_ulp(-0.25))
        report = "docs/numerics/m/3.md"
        base = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], True)
        head = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(3, flipped, report=report)], True)
        anchor = NUMERICS.render_reference(NUMERICS.synthetic_reference([9], -0.5)).encode()
        read = self.store(flipped, **{report: b"x", NUMERICS.ANCHOR_PATH: anchor})
        findings = NUMERICS.declaration_findings(base, head, self.store(self.old), read, {self.path: "M", report: "A"})
        self.assertTrue(any("one declared change is one version" in finding for finding in findings), findings)

    def test_a_recapture_snippet_keeps_the_entry_it_recaptures(self):
        flipped = NUMERICS.synthetic_reference([5, 6], next_ulp(-0.25))
        flipped["captured"] = "2026-11-04"
        fresh = NUMERICS.entry_snippet(flipped, "references/q.json")
        self.assertIn("version = 1", fresh)
        self.assertNotIn("report", fresh)
        registered = NUMERICS.synthetic_entry(3, self.old, report="docs/numerics/m/3.md")
        registered["note"] = "kept"
        snippet = NUMERICS.entry_snippet(flipped, "references/q.json", registered)
        self.assertIn("version = 3", snippet)
        self.assertIn('report = "docs/numerics/m/3.md"', snippet)
        self.assertIn('note = "kept"', snippet)
        self.assertIn(f'fingerprint = "{flipped["fingerprint"]}"', snippet)

    def test_every_arm_profile_names_a_registered_model(self):
        # Which model a profile names is held to its weights' descriptor by sparkflash-model's
        # `every_arm_profile_names_the_model_its_weights_select`.
        repo = NUMERICS.REPO
        on_disk = sorted(repo.glob(NUMERICS.PROFILE_GLOB))
        self.assertGreaterEqual(len(on_disk), 3)
        registry = NUMERICS.parse_registry((repo / NUMERICS.REGISTRY).read_text(), NUMERICS.REGISTRY)
        models = {entry["model"] for entry in registry["config"]}
        for path in on_disk:
            relative = str(path.relative_to(repo))
            self.assertTrue(NUMERICS.is_profile(relative), relative)
            self.assertIn(NUMERICS.profile_settings(path.read_bytes(), relative).get("model"), models, relative)
        neutral = NUMERICS.neutral_fields((repo / NUMERICS.CLASSIFICATION).read_bytes())
        self.assertIn("runtime.concurrency", neutral)
        self.assertNotIn("decode.tensor_cores", neutral)

    def test_ci_against_a_real_git_history(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args: str) -> str:
                return subprocess.run(
                    ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()

            git("init", "-q")
            (repo / "numerics" / "references").mkdir(parents=True)
            reference = repo / "numerics" / "references" / "q.json"
            reference.write_text(NUMERICS.render_reference(self.old))
            entry = NUMERICS.synthetic_entry(1, self.old)
            registry = TOY_REGISTRY + "\n[[config]]\n" + "\n".join(
                f"{key} = {json.dumps(value)}" for key, value in entry.items()
            ) + "\n"
            (repo / "numerics" / "registry.toml").write_text(registry)
            git("add", "-A")
            git("commit", "-q", "-m", "base")
            base = git("rev-parse", "HEAD")
            with mock.patch.object(NUMERICS, "REPO", repo), contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(NUMERICS.run_ci(base), 0, out.getvalue())
                # The bits change, and the fingerprint follows, but the version does not.
                flipped = NUMERICS.synthetic_reference([5, 6], next_ulp(-0.25))
                reference.write_text(NUMERICS.render_reference(flipped))
                (repo / "numerics" / "registry.toml").write_text(
                    registry.replace(self.old["fingerprint"], flipped["fingerprint"])
                )
                self.assertEqual(NUMERICS.run_ci(base), 1)
            self.assertIn("undeclared change", out.getvalue())

    def profile_repo(self, repo: Path) -> tuple[Callable[..., str], str]:
        """A git repository holding model "m"'s registered reference, its arm profile and the
        neutral list, committed; returns its git runner and the base commit."""

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                check=True, capture_output=True, text=True,
            ).stdout.strip()

        git("init", "-q")
        (repo / "numerics" / "references").mkdir(parents=True)
        (repo / "numerics" / "references" / "q.json").write_text(NUMERICS.render_reference(self.old))
        entry = NUMERICS.synthetic_entry(1, self.old)
        (repo / "numerics" / "registry.toml").write_text(
            TOY_REGISTRY + "\n[[config]]\n" + "\n".join(f"{key} = {json.dumps(value)}" for key, value in entry.items()) + "\n"
        )
        classification = repo / NUMERICS.CLASSIFICATION
        classification.parent.mkdir(parents=True)
        classification.write_text('pub const NEUTRAL_FIELDS: &[&str] = &[\n    "runtime.concurrency",\n];\n')
        profile = repo / "crates" / "sparkflash-model-m" / "config" / "m.toml"
        profile.parent.mkdir(parents=True)
        profile.write_text('model = "m"\n[runtime]\nconcurrency = 1\n[decode]\ntensor_cores = true\n')
        git("add", "-A")
        git("commit", "-q", "-m", "base")
        return git, git("rev-parse", "HEAD")

    def test_ci_sees_an_arm_profile_change(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            git, base = self.profile_repo(repo)
            profile = repo / "crates" / "sparkflash-model-m" / "config" / "m.toml"
            with mock.patch.object(NUMERICS, "REPO", repo), contextlib.redirect_stdout(io.StringIO()) as out:
                profile.write_text('model = "m"\n[runtime]\nconcurrency = 4\n[decode]\ntensor_cores = true\n')
                self.assertEqual(NUMERICS.run_ci(base), 0, out.getvalue())
                # Committed, so the change reaches ci through the diff and not as an untracked file.
                profile.write_text('model = "m"\n[runtime]\nconcurrency = 1\n[decode]\ntensor_cores = false\n')
                git("commit", "-q", "-am", "flip")
                self.assertEqual(NUMERICS.run_ci(base), 1)
            self.assertIn("decode.tensor_cores can change m's arithmetic", out.getvalue())

    def test_ci_judges_the_neutral_list_it_trusts(self):
        # The review's bypass: one change makes a field neutral, so a later one can flip it in a
        # profile with no recapture. The first change is where it stops.
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            git, base = self.profile_repo(repo)
            classification = repo / NUMERICS.CLASSIFICATION
            with mock.patch.object(NUMERICS, "REPO", repo), contextlib.redirect_stdout(io.StringIO()) as out:
                classification.write_text(
                    'pub const NEUTRAL_FIELDS: &[&str] = &[\n    "runtime.concurrency",\n    "decode.tensor_cores",\n];\n'
                )
                git("commit", "-q", "-am", "reclassify")
                self.assertEqual(NUMERICS.run_ci(base), 1)
                self.assertIn("decode.tensor_cores becomes neutral", out.getvalue())
                report = repo / "docs" / "numerics" / "m" / "neutral.md"
                report.parent.mkdir(parents=True)
                report.write_text("`decode.tensor_cores` gives the same bits at both settings: ...\n")
                self.assertEqual(NUMERICS.run_ci(base), 0, out.getvalue())


if __name__ == "__main__":
    unittest.main()
