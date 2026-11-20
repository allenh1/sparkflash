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
import hashlib
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
# The build id a served card reports in these tests: a plausible short commit, so a capture's
# server.commit is visibly the card's value and not the test checkout's HEAD.
CARD_COMMIT = "a6ef32e4"


def with_server(reference: dict, **server) -> dict:
    """A copy of a reference whose server card is `server`, leaving the fingerprinted bits alone."""
    named = json.loads(json.dumps(reference))
    named["server"] = server
    return named


def with_floor(reference: dict) -> dict:
    """A copy of a reference carrying a correctness floor that passes every bar."""
    floored = json.loads(json.dumps(reference))
    floored["floor"] = {
        "against": "the pre-change build",
        "forced": {
            "positions": 2,
            "mean_nll_delta": 1.0e-04,
            "mean_abs_delta": 1.0e-04,
            "max_abs_delta": 1.0e-02,
            "top1_agreement": 1.0,
        },
        "known_answers": [{"prompt": "p", "expected": "x", "got": "x", "pass": True}],
        "nll_per_token": 0.2,
        "explanation": None,
    }
    return floored


def next_ulp(value: float) -> float:
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", value))[0] + 1))[0]


class FakeServer:
    """A server whose logprobs are a pure function of the fed prefix, like a deterministic model.

    `shift` moves every logit of one token, `ignore_forced` answers its own choice instead of
    the forced tokens, and `forced_decoding` is what the card advertises. A request without a
    zero temperature samples, as the API does at its default of 1.0.
    """

    def __init__(
        self,
        shift: float = 0.0,
        ignore_forced: bool = False,
        forced_decoding: bool = True,
        build_id: str | None = CARD_COMMIT,
    ):
        self.shift = shift
        self.ignore_forced = ignore_forced
        self.forced_decoding = forced_decoding
        self.build_id = build_id
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
            "build": {"version": "0", "id": self.build_id},
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

    def git_repo(self, directory: str) -> tuple[Callable[..., str], str]:
        """A clean git tree holding `binary`; returns its git runner and its HEAD commit."""

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", "-C", directory, "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                check=True, capture_output=True, text=True,
            ).stdout.strip()

        git("init", "-q")
        git("add", "-A")
        git("commit", "-q", "-m", "server")
        return git, git("rev-parse", "HEAD")

    def test_a_capture_records_the_commit_the_served_card_names(self):
        prompts = [{"name": "a", "prompt_ids": [1]}]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "sparkflash"
            binary.write_bytes(b"the build")
            git, checkout = self.git_repo(directory)
            with serving(FakeServer()) as url:
                reference = NUMERICS.capture(
                    url, prompts, "logprobs", 4, 2, None, 30.0, binary=binary, repo=Path(directory)
                )
        server = reference["server"]
        self.assertEqual(server["binary_sha256"], hashlib.sha256(b"the build").hexdigest())
        # The served card's build id wins: --repo here names a different, real checkout.
        self.assertNotEqual(checkout, CARD_COMMIT)
        self.assertEqual(server["commit"], CARD_COMMIT)
        self.assertEqual(server["build_source"], "card")
        self.assertIsNone(server["build_note"])

    def test_the_repo_is_a_fallback_when_the_card_names_no_commit(self):
        prompts = [{"name": "a", "prompt_ids": [1]}]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "sparkflash"
            binary.write_bytes(b"the build")
            git, commit = self.git_repo(directory)
            with serving(FakeServer(build_id=None)) as url:
                reference = NUMERICS.capture(
                    url, prompts, "logprobs", 4, 2, None, 30.0, binary=binary, repo=Path(directory)
                )
        server = reference["server"]
        self.assertEqual(server["commit"], commit)
        self.assertEqual(server["build_source"], "repo")
        self.assertIn("no build id", server["build_note"])

    def test_a_dirty_served_build_is_recorded_dirty(self):
        with serving(FakeServer(build_id=f"{CARD_COMMIT}-dirty")) as url:
            reference = capture(url)
        server = reference["server"]
        self.assertEqual(server["commit"], f"{CARD_COMMIT}-dirty")
        self.assertEqual(server["build_source"], "card")
        self.assertIn("dirty", server["build_note"])

    def test_a_capture_warns_when_the_served_card_names_no_commit(self):
        with serving(FakeServer(build_id=None)) as url, contextlib.redirect_stderr(io.StringIO()) as warning:
            reference = capture(url)
        server = reference["server"]
        self.assertIsNone(server["binary_sha256"])
        self.assertIsNone(server["commit"])
        self.assertIsNone(server["build_source"])
        self.assertIn("no --binary", server["build_note"])
        self.assertIn("no build id", server["build_note"])
        self.assertIn("no --repo", server["build_note"])
        self.assertIn("names no build commit", warning.getvalue())

    def test_a_build_that_cannot_be_determined_names_no_host_path(self):
        missing = Path(tempfile.gettempdir()) / "not-a-sparkflash-binary"
        repo = Path(tempfile.gettempdir()) / "not-a-sparkflash-checkout"
        identity = NUMERICS.build_identity(missing, repo)
        self.assertIsNone(identity["binary_sha256"])
        self.assertIsNone(identity["commit"])
        self.assertIn("not readable", identity["build_note"])
        self.assertIn("not a git work tree", identity["build_note"])
        self.assertNotIn(str(missing), identity["build_note"])
        self.assertNotIn(str(repo), identity["build_note"])

    def test_the_build_card_does_not_change_the_fingerprint(self):
        prompts = [{"name": "a", "prompt_ids": [1, 2]}]
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "sparkflash"
            binary.write_bytes(b"the build")
            with serving(FakeServer()) as url:
                plain = NUMERICS.capture(url, prompts, "logprobs", 6, 4, None, 30.0)
                named = NUMERICS.capture(
                    url, prompts, "logprobs", 6, 4, None, 30.0, binary=binary, repo=None
                )
        self.assertEqual(plain["server"]["binary_sha256"], None)
        self.assertNotEqual(plain["server"], named["server"])
        self.assertEqual(plain["fingerprint"], named["fingerprint"])
        self.assertNotIn("binary_sha256", json.dumps(NUMERICS.reference_bits(named)))

    def test_a_reference_without_the_build_keys_stays_valid_and_a_malformed_one_is_refused(self):
        reference = NUMERICS.synthetic_reference([5], -0.25)
        NUMERICS.check_reference_shape(reference, "old")  # no server card at all
        reference["server"] = {"build": None, "checkpoint": "ckpt"}
        NUMERICS.check_reference_shape(reference, "old")
        reference["server"] = {"binary_sha256": 7}
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "binary_sha256"):
            NUMERICS.check_reference_shape(reference, "bad")
        reference["server"] = {"binary_sha256": "A" * 64}
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "64 lowercase hex"):
            NUMERICS.check_reference_shape(reference, "bad")
        reference["server"] = {"build_source": "guessed"}
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "build_source"):
            NUMERICS.check_reference_shape(reference, "bad")

    def test_the_capture_command_hashes_the_binary_from_the_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "sparkflash"
            binary.write_bytes(b"the build")
            out = Path(directory) / "reference.json"
            with serving(FakeServer()) as url, mock.patch.dict(
                "os.environ", {"SPARKFLASH_BIN": str(binary)}
            ), contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(
                    NUMERICS.main(["capture", "--url", url, "--out", str(out), "--max-tokens", "2"]), 0
                )
            reference = json.loads(out.read_text())
        self.assertEqual(reference["server"]["binary_sha256"], hashlib.sha256(b"the build").hexdigest())
        checkout = subprocess.run(
            ["git", "-C", str(NUMERICS.REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        # --repo defaults to this checkout, but the served card's build id is what the dump records.
        self.assertNotEqual(reference["server"]["commit"], checkout)
        self.assertEqual(reference["server"]["commit"], CARD_COMMIT)
        self.assertEqual(reference["server"]["build_source"], "card")
        self.assertIn("binary sha256", printed.getvalue())

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
coverage_slots = 0.93
coverage = false
monitors = false
"""


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return repr(value)


def rows_toml(table: str, rows: list[dict]) -> str:
    """One per-model evidence table as TOML, in the registry's [[table]] shape."""
    parts = []
    for row in rows:
        parts.append(f"[[{table}]]")
        parts.extend(f"{key} = {_toml_value(value)}" for key, value in row.items())
        parts.append("")
    return "\n".join(parts) + ("\n" if parts else "")


def registry_text(
    coverage: list[dict] | None = None,
    monitors: list[dict] | None = None,
    coverage_flag: bool = False,
    monitors_flag: bool = False,
) -> str:
    """The toy registry with the per-model evidence rows a test needs."""
    text = TOY_REGISTRY.replace(
        "coverage = false", f"coverage = {'true' if coverage_flag else 'false'}"
    ).replace("monitors = false", f"monitors = {'true' if monitors_flag else 'false'}")
    return text + rows_toml("coverage", coverage or []) + rows_toml("monitors", monitors or [])


def coverage_row(model: str = MODEL, dump: str = "numerics/coverage/toy.json", inputs: int = 2) -> dict:
    return {"model": model, "dump": dump, "min_inputs": inputs}


def monitors_row(model: str = MODEL, bars: float = 0.5) -> dict:
    return {
        "model": model,
        "reference": "numerics/monitors/ref.json",
        "candidate": "numerics/monitors/cand.json",
        "layer_bar": bars,
        "expert_bar": bars,
    }


def captured(model: str = MODEL, evidence: str = "logprobs") -> dict:
    """A captured configuration, enough for the aggregate flags' model set."""
    return {"model": model, "evidence": evidence, "reference": "references/q.json"}


def evidence_files(model: str = MODEL) -> dict[str, bytes]:
    """Passing synthetic coverage and monitor dumps for the model's rows."""
    return {
        "numerics/coverage/toy.json": json.dumps(
            coverage_dump(
                [
                    {"layer": 0, "hits": []},
                    {"layer": 1, "hits": [1, 1, 1]},
                    {"layer": 2, "hits": [1, 1, 1]},
                ],
                model=model,
            )
        ).encode(),
        "numerics/monitors/ref.json": json.dumps(
            monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (1, [0.0, 2.0])], model=model)
        ).encode(),
        "numerics/monitors/cand.json": json.dumps(
            monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (2, [0.0, 2.0])], model=model)
        ).encode(),
    }


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

    def test_a_new_reference_without_a_build_is_refused(self):
        # A capture taken after the record landed writes build_source; with no commit it cannot
        # say which binary produced its bits, so the reference is refused.
        base = NUMERICS.synthetic_registry([], False)
        head = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        missing = with_server(self.old, build_source="card", commit=None)
        read = {self.path: NUMERICS.render_reference(missing).encode()}.get
        findings = NUMERICS.declaration_findings(base, head, read, read, {self.path: "A"})
        self.assertTrue(any("server.commit names no build" in finding for finding in findings), findings)

    def test_a_new_reference_that_says_unknown_is_refused(self):
        base = NUMERICS.synthetic_registry([], False)
        head = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        unknown = with_server(self.old, build_source="card", commit="unknown")
        read = {self.path: NUMERICS.render_reference(unknown).encode()}.get
        findings = NUMERICS.declaration_findings(base, head, read, read, {self.path: "A"})
        self.assertTrue(any("server.commit names no build" in finding for finding in findings), findings)

    def test_a_new_dirty_build_is_accepted_and_named_dirty(self):
        base = NUMERICS.synthetic_registry([], False)
        head = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        dirty = with_floor(with_server(self.old, build_source="card", commit=f"{CARD_COMMIT}-dirty"))
        read = {self.path: NUMERICS.render_reference(dirty).encode()}.get
        findings = NUMERICS.declaration_findings(base, head, read, read, {self.path: "A"})
        self.assertEqual(findings, [])

    def test_a_new_reference_without_a_floor_is_refused(self):
        # Determinism is not correctness: a build-stamped capture must also carry the evidence
        # that its arithmetic is right (docs/NUMERICS.md §5).
        base = NUMERICS.synthetic_registry([], False)
        head = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        stamped = with_server(self.old, build_source="card", commit=CARD_COMMIT)
        read = {self.path: NUMERICS.render_reference(stamped).encode()}.get
        findings = NUMERICS.declaration_findings(base, head, read, read, {self.path: "A"})
        self.assertTrue(any("no correctness floor" in finding for finding in findings), findings)

    def test_a_new_reference_whose_floor_misses_a_bar_is_refused_unless_explained(self):
        base = NUMERICS.synthetic_registry([], False)
        head = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        floored = with_floor(with_server(self.old, build_source="card", commit=CARD_COMMIT))
        floored["floor"]["forced"]["top1_agreement"] = 0.2949
        read = {self.path: NUMERICS.render_reference(floored).encode()}.get
        findings = NUMERICS.declaration_findings(base, head, read, read, {self.path: "A"})
        self.assertTrue(any("top-1 agreement" in finding for finding in findings), findings)
        floored["floor"]["explanation"] = "the layout defect explains the distance"
        read = {self.path: NUMERICS.render_reference(floored).encode()}.get
        findings = NUMERICS.declaration_findings(base, head, read, read, {self.path: "A"})
        self.assertEqual(findings, [])

    def test_a_grandfathered_reference_without_the_build_key_passes(self):
        # A reference committed before the record existed has no build_source; the absent key is
        # the grandfather note, so it is accepted without a commit.
        base = NUMERICS.synthetic_registry([], False)
        head = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        findings = NUMERICS.declaration_findings(base, head, self.store(self.old), self.store(self.old), {self.path: "A"})
        self.assertEqual(findings, [])

    def test_a_new_anchor_without_a_build_is_refused(self):
        anchored = with_server(self.old, build_source="card", commit=None)
        entry = NUMERICS.synthetic_entry(1, self.old)
        base = NUMERICS.synthetic_registry([entry], False)
        head = NUMERICS.synthetic_registry([entry], True)
        head["anchor"][0]["reference"] = "references/q.json"
        head["anchor"][0]["fingerprint"] = entry["fingerprint"]
        read = {self.path: NUMERICS.render_reference(anchored).encode()}.get
        findings = NUMERICS.declaration_findings(base, head, read, read, {})
        self.assertTrue(
            any("anchor m:" in finding and "server.commit names no build" in finding for finding in findings),
            findings,
        )

    def test_a_grandfathered_anchor_without_the_build_key_passes(self):
        entry = NUMERICS.synthetic_entry(1, self.old)
        base = NUMERICS.synthetic_registry([entry], False)
        head = NUMERICS.synthetic_registry([entry], True)
        head["anchor"][0]["reference"] = "references/q.json"
        head["anchor"][0]["fingerprint"] = entry["fingerprint"]
        findings = NUMERICS.declaration_findings(base, head, self.store(self.old), self.store(self.old), {})
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

    def test_every_arm_profile_names_a_model(self):
        # Which model a profile names is held to its weights' descriptor by sparkflash-model's
        # `every_arm_profile_names_the_model_its_weights_select`. The registry names the models
        # with a captured reference; a profile may serve a model that has none yet (Qwen3.6 is
        # unregistered), so membership is not required - only a name, and a registered model
        # must have a profile.
        repo = NUMERICS.REPO
        on_disk = sorted(repo.glob(NUMERICS.PROFILE_GLOB))
        self.assertGreaterEqual(len(on_disk), 3)
        registry = NUMERICS.parse_registry((repo / NUMERICS.REGISTRY).read_text(), NUMERICS.REGISTRY)
        models = {entry["model"] for entry in registry["config"]}
        profiled = set()
        for path in on_disk:
            relative = str(path.relative_to(repo))
            self.assertTrue(NUMERICS.is_profile(relative), relative)
            model = NUMERICS.profile_settings(path.read_bytes(), relative).get("model")
            self.assertIsInstance(model, str, relative)
            self.assertTrue(model, relative)
            profiled.add(model)
        self.assertLessEqual(models, profiled, "a registered model has no arm profile")
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


class ApprovalTests(unittest.TestCase):
    """The approved path: an owner's recorded decision carries one bump past the global gates."""

    def setUp(self):
        self.old = NUMERICS.synthetic_reference([5, 6], -0.25)
        self.flipped = NUMERICS.synthetic_reference([5, 6], next_ulp(-0.25))
        self.path = "numerics/references/q.json"
        self.report = "docs/numerics/m/2.md"
        self.approval_report = "docs/numerics/m/approval.md"
        self.approval = {
            "decision": "The owner's recorded decision for this change",
            "date": "2026-11-18",
            "report": self.approval_report,
        }
        self.changed = {self.path: "M", self.report: "A", self.approval_report: "A"}
        self.extra = {self.report: b"x", self.approval_report: b"approved\n"}

    def approved_entry(self, **overrides) -> dict:
        version = overrides.pop("version", 2)
        reference = overrides.pop("reference", self.flipped)
        report = overrides.pop("report", self.report)
        approval = overrides.pop("approval", self.approval)
        assert not overrides, overrides
        entry = NUMERICS.synthetic_entry(version, reference, report=report)
        entry["approved"] = dict(approval)
        return entry

    def judge(
        self,
        head_entry: dict,
        changed: dict,
        extra: dict | None = None,
        base_entry: dict | None = None,
        anchored: bool = True,
        head_reference: dict | None = None,
        base_reference: dict | None = None,
    ) -> list:
        base = NUMERICS.synthetic_registry(
            [base_entry or NUMERICS.synthetic_entry(1, self.old)], anchored
        )
        head = NUMERICS.synthetic_registry([head_entry], anchored)
        for prerequisite in NUMERICS.GATING_PREREQUISITES:
            base["prerequisites"][prerequisite] = False
            head["prerequisites"][prerequisite] = False
        anchor = NUMERICS.render_reference(NUMERICS.synthetic_reference([9], -0.5)).encode()
        base_files = {
            self.path: NUMERICS.render_reference(base_reference or self.old).encode(),
            NUMERICS.ANCHOR_PATH: anchor,
            self.report: b"old",
            self.approval_report: b"old",
        }
        head_files = {
            self.path: NUMERICS.render_reference(head_reference or self.flipped).encode(),
            NUMERICS.ANCHOR_PATH: anchor,
            self.report: b"x",
            **(extra or {}),
        }
        return NUMERICS.declaration_findings(base, head, base_files.get, head_files.get, changed)

    def test_an_approval_carries_a_bump_past_the_global_prerequisites(self):
        self.assertEqual(self.judge(self.approved_entry(), self.changed, self.extra), [])

    def test_without_the_approval_the_same_bump_is_refused_by_the_gate(self):
        entry = NUMERICS.synthetic_entry(2, self.flipped, report=self.report)
        findings = self.judge(entry, {self.path: "M", self.report: "A"})
        self.assertTrue(any("prerequisites" in finding for finding in findings), findings)

    def test_an_approval_with_an_empty_decision_or_date_is_refused(self):
        for field in ("decision", "date"):
            approval = dict(self.approval, **{field: "  "})
            findings = self.judge(self.approved_entry(approval=approval), self.changed, self.extra)
            self.assertTrue(
                any(f"approved.{field} is empty" in finding for finding in findings), findings
            )

    def test_an_approval_with_a_date_that_is_not_yyyy_mm_dd_is_refused(self):
        approval = dict(self.approval, date="18-11-2026")
        findings = self.judge(self.approved_entry(approval=approval), self.changed, self.extra)
        self.assertTrue(any("YYYY-MM-DD" in finding for finding in findings), findings)

    def test_an_approval_that_is_not_a_table_is_refused(self):
        entry = dict(self.approved_entry(), approved="yes")
        findings = self.judge(entry, self.changed, self.extra)
        self.assertTrue(any("approved is not a table" in finding for finding in findings), findings)

    def test_an_approval_outside_the_documented_shape_is_refused(self):
        approval = dict(self.approval, report="numerics/approval.md")
        findings = self.judge(self.approved_entry(approval=approval), self.changed, self.extra)
        self.assertTrue(any("under docs/numerics/" in finding for finding in findings), findings)
        unknown = dict(self.approval, note="not a field of the record")
        findings = self.judge(self.approved_entry(approval=unknown), self.changed, self.extra)
        self.assertTrue(any("unknown key" in finding for finding in findings), findings)

    def test_an_approval_whose_report_does_not_exist_is_refused(self):
        findings = self.judge(self.approved_entry(), self.changed, {self.report: b"x"})
        self.assertTrue(
            any("approved.report" in finding and "does not exist" in finding for finding in findings),
            findings,
        )

    def test_an_approval_whose_report_is_not_new_or_changed_is_refused(self):
        findings = self.judge(
            self.approved_entry(),
            {self.path: "M", self.report: "A"},
            {self.report: b"x", self.approval_report: b"old"},
        )
        self.assertTrue(any("not new or changed" in finding for finding in findings), findings)

    def test_an_approval_on_a_bump_that_changes_nothing_is_refused(self):
        entry = self.approved_entry(reference=self.old)
        findings = self.judge(entry, self.changed, self.extra, head_reference=self.old)
        self.assertTrue(any("declared non-change" in finding for finding in findings), findings)

    def test_an_approved_bump_still_names_its_version_report(self):
        entry = self.approved_entry(report=None)
        findings = self.judge(entry, {self.path: "M", self.approval_report: "A"}, {self.approval_report: b"x"})
        self.assertTrue(any("names no report" in finding for finding in findings), findings)

    def test_an_approval_on_a_model_with_no_anchor_is_refused(self):
        findings = self.judge(self.approved_entry(), self.changed, self.extra, anchored=False)
        self.assertTrue(any("has no [[anchor]]" in finding for finding in findings), findings)

    def test_removing_an_approval_is_refused(self):
        base_entry = self.approved_entry()
        head_entry = NUMERICS.synthetic_entry(2, self.flipped, report=self.report)
        findings = self.judge(
            head_entry,
            {},
            {self.approval_report: b"approved\n"},
            base_entry=base_entry,
            base_reference=self.flipped,
        )
        self.assertTrue(any("removed the approval" in finding for finding in findings), findings)

    def test_an_approval_added_without_a_version_bump_is_refused(self):
        entry = dict(NUMERICS.synthetic_entry(1, self.old), approved=dict(self.approval))
        findings = self.judge(
            entry,
            {self.approval_report: "A"},
            {self.approval_report: b"approved\n"},
            head_reference=self.old,
        )
        self.assertTrue(any("without a version bump" in finding for finding in findings), findings)

    def test_a_recapture_snippet_keeps_the_approval(self):
        reference = dict(self.flipped, captured="2026-11-19")
        snippet = NUMERICS.entry_snippet(reference, "references/q.json", self.approved_entry())
        self.assertIn('approved = { decision = "', snippet)
        self.assertIn(self.approval_report, snippet)

    def test_the_registry_parses_a_nested_approval_table(self):
        text = TOY_REGISTRY + """
[[config]]
model = "m"
world = 4
profile = "reference"
plan = "tp"
arch = "sm_121a"
version = 2
evidence = "logprobs"
reference = "references/q.json"
fingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
report = "docs/numerics/m/2.md"
captured = "2026-11-03"
approved = { decision = "The owner's recorded decision", date = "2026-11-18", report = "docs/numerics/m/approval.md" }
"""
        registry = NUMERICS.parse_registry(text, "toy")
        self.assertEqual(registry["config"][0]["approved"]["date"], "2026-11-18")
        self.assertIsNone(NUMERICS.approval_problem(registry["config"][0]))


class PerModelBumpTests(unittest.TestCase):
    """A bump is judged against the bumping entry's own model, not the aggregate flags."""

    def setUp(self):
        self.old = NUMERICS.synthetic_reference([5, 6], -0.25)
        self.flipped = NUMERICS.synthetic_reference([5, 6], next_ulp(-0.25))
        self.anchor = NUMERICS.synthetic_reference([9], -0.5)
        self.path = "numerics/references/q.json"
        self.report = "docs/numerics/m/2.md"
        self.approval_report = "docs/numerics/m/approval.md"
        self.approval = {
            "decision": "The owner's recorded decision for this change",
            "date": "2026-11-18",
            "report": self.approval_report,
        }
        layout = mock.patch.object(
            NUMERICS, "find_layout", return_value=(synthetic_layout(model="m"), "layout.json")
        )
        layout.start()
        self.addCleanup(layout.stop)

    def judge(
        self,
        head_entry: dict,
        coverage_rows: list | None = None,
        monitors_rows: list | None = None,
        changed: dict | None = None,
        extra: dict | None = None,
    ) -> list:
        base = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], True)
        head = NUMERICS.synthetic_registry([head_entry], True)
        for registry in (base, head):
            registry["coverage"] = (
                [coverage_row("m")] if coverage_rows is None else coverage_rows
            )
            registry["monitors"] = (
                [monitors_row("m")] if monitors_rows is None else monitors_rows
            )
        changed = changed if changed is not None else {self.path: "M", self.report: "A"}
        files = {
            self.path: NUMERICS.render_reference(self.flipped).encode(),
            NUMERICS.ANCHOR_PATH: NUMERICS.render_reference(self.anchor).encode(),
            self.report: b"x",
            self.approval_report: b"approved",
            **evidence_files("m"),
            **(extra or {}),
        }
        base_files = dict(files, **{self.path: NUMERICS.render_reference(self.old).encode()})
        return NUMERICS.declaration_findings(
            base,
            head,
            base_files.get,
            files.get,
            changed,
            {"m": {self.anchor["fingerprint"]}},
        )

    def bumped(self, **overrides) -> dict:
        entry = NUMERICS.synthetic_entry(2, self.flipped, report=self.report)
        entry.update(overrides)
        return entry

    def test_an_unapproved_bump_with_no_row_names_the_model(self):
        entry = self.bumped()
        findings = self.judge(entry, coverage_rows=[], monitors_rows=[])
        self.assertTrue(
            any("no passing per-model evidence" in finding and "'m'" in finding for finding in findings),
            findings,
        )
        self.assertTrue(any("no [[coverage]] row" in finding for finding in findings), findings)
        self.assertTrue(any("no [[monitors]] row" in finding for finding in findings), findings)

    def test_an_unapproved_bump_with_a_failing_row_names_the_model(self):
        entry = self.bumped()
        findings = self.judge(
            entry,
            coverage_rows=[coverage_row("m", dump="numerics/coverage/missing.json")],
        )
        self.assertTrue(
            any("'m'" in finding and "unusable" in finding for finding in findings), findings
        )

    def test_an_unapproved_bump_with_its_models_passing_rows_passes(self):
        findings = self.judge(self.bumped())
        self.assertEqual(findings, [], findings)

    def test_an_approved_bump_needs_no_row(self):
        entry = self.bumped(approved=dict(self.approval))
        findings = self.judge(
            entry,
            coverage_rows=[],
            monitors_rows=[],
            changed={self.path: "M", self.report: "A", self.approval_report: "A"},
        )
        self.assertEqual(findings, [], findings)


def synthetic_layout(model: str = MODEL, spans: list | None = None) -> dict:
    return {
        "schema": NUMERICS.LAYOUT_SCHEMA,
        "model": model,
        "layers": spans
        or [
            {"first": 0, "count": 1, "experts": 0},
            {"first": 1, "count": 2, "experts": 3},
        ],
    }


def coverage_dump(layers: list, inputs: list | None = None, model: str = MODEL) -> dict:
    return {
        "schema": NUMERICS.COVERAGE_SCHEMA,
        "model": model,
        "inputs": [{"name": name, "tokens": 4} for name in (inputs or ["a", "b"])],
        "layers": layers,
        "captured": "2026-11-19",
    }


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.layout = synthetic_layout()

    def judge(self, dump: dict, min_inputs: int = 2, bar: float = 0.93):
        return NUMERICS.judge_coverage(dump, self.layout, "layout.json", min_inputs, bar)

    def test_every_layer_and_expert_is_judged(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 2, 3]},
                {"layer": 2, "hits": [4, 5, 6]},
            ]
        )
        result = self.judge(dump)
        self.assertEqual(result["experts"], 6)
        self.assertEqual([layer["hits"] for layer in result["layers"]], [0, 6, 15])
        markdown = NUMERICS.render_coverage(result)
        self.assertIn("PASS coverage", markdown)
        self.assertIn("0 (dense)", markdown)

    def test_a_missing_layer_is_refused(self):
        dump = coverage_dump([{"layer": 0, "hits": []}, {"layer": 1, "hits": [1, 2, 3]}])
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "layer 2 is missing"):
            self.judge(dump)

    def test_an_unreached_expert_is_reported_and_fails_below_the_bar(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 0, 3]},
                {"layer": 2, "hits": [4, 5, 6]},
            ]
        )
        result = self.judge(dump)
        self.assertFalse(result["passes"])
        self.assertEqual(result["missing"], [[1, 1]], "the layer and expert are still named")
        markdown = NUMERICS.render_coverage(result)
        self.assertIn("FAIL coverage", markdown)
        self.assertIn("83.33%", markdown)
        self.assertIn("93.00%", markdown)
        self.assertIn("| 1 | 3 | 4 | 0 | 3 | 1 |", markdown)

    def test_a_dump_at_the_declared_bar_passes(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 1, 0]},
                {"layer": 2, "hits": [1, 0, 0]},
            ]
        )
        result = self.judge(dump, bar=0.5)
        self.assertTrue(result["passes"])
        markdown = NUMERICS.render_coverage(result)
        self.assertIn("PASS coverage", markdown)
        self.assertIn("achieved: 3 of 6", markdown)
        self.assertIn("50.00%; declared bar 50.00%", markdown)

    def test_a_dump_just_below_the_bar_fails_and_names_both_fractions(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 1, 0]},
                {"layer": 2, "hits": [1, 0, 0]},
            ]
        )
        result = self.judge(dump, bar=0.5001)
        self.assertFalse(result["passes"])
        markdown = NUMERICS.render_coverage(result)
        self.assertIn("FAIL coverage", markdown)
        self.assertIn("50.00%", markdown)
        self.assertIn("50.01%", markdown)
        self.assertIn("3 of 6 slot(s) unreached", markdown)

    def test_a_bar_outside_the_unit_interval_or_not_a_number_is_refused(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 1, 1]},
                {"layer": 2, "hits": [1, 1, 1]},
            ]
        )
        for bar in (0, -0.1, 1.0001, float("nan"), float("inf"), True, "0.93", None):
            with self.assertRaisesRegex(NUMERICS.NumericsFailure, "coverage_slots"):
                self.judge(dump, bar=bar)

    def test_the_head_registry_must_state_the_bar(self):
        without = TOY_REGISTRY.replace("coverage_slots = 0.93\n", "")
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "coverage_slots is missing"):
            NUMERICS.parse_registry(without, "toy", require_bar=True)
        # A base written before the bar existed stays readable; only its own bar is judged.
        self.assertNotIn("coverage_slots", NUMERICS.parse_registry(without, "toy")["prerequisites"])

    def test_a_malformed_coverage_row_is_refused_at_parse(self):
        good = coverage_row()
        NUMERICS.parse_registry(registry_text(coverage=[good]), "toy")
        cases = [
            dict(good, dump="docs/numerics/q.json"),
            dict(good, dump="numerics/coverage/../q.json"),
            dict(good, dump="numerics/coverage/q.txt"),
            dict(good, min_inputs=0),
            dict(good, min_inputs=-1),
            dict(good, min_inputs=True),
            {key: value for key, value in good.items() if key != "dump"},
            {key: value for key, value in good.items() if key != "model"},
            dict(good, surprise=1),
        ]
        for row in cases:
            with self.subTest(row=row):
                with self.assertRaises(NUMERICS.NumericsFailure, msg=str(row)) as refused:
                    NUMERICS.parse_registry(registry_text(coverage=[row]), "toy")
                self.assertIn("coverage", str(refused.exception))
                if "model" in row:
                    self.assertIn(MODEL, str(refused.exception), "a malformed row names its model")
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "not a table"):
            NUMERICS.coverage_row("nope", "toy")

    def test_one_coverage_row_a_model_is_refused(self):
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "one row a model"):
            NUMERICS.parse_registry(registry_text(coverage=[coverage_row(), coverage_row()]), "toy")

    def test_a_true_coverage_flag_is_reported_not_enforced_at_parse(self):
        # The aggregate is ci's claim about the rows; parsing does not invent rows for it.
        registry = NUMERICS.parse_registry(registry_text(coverage_flag=True), "toy")
        self.assertTrue(registry["prerequisites"]["coverage"])
        self.assertEqual(registry["coverage"], [])

    def test_a_wrong_expert_count_is_refused(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 2]},
                {"layer": 2, "hits": [4, 5, 6]},
            ]
        )
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "hit count"):
            self.judge(dump)

    def test_a_dense_layer_with_hits_is_refused(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": [1]},
                {"layer": 1, "hits": [1, 2, 3]},
                {"layer": 2, "hits": [4, 5, 6]},
            ]
        )
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "dense"):
            self.judge(dump)

    def test_a_duplicate_or_short_input_list_is_refused(self):
        layers = [
            {"layer": 0, "hits": []},
            {"layer": 1, "hits": [1, 2, 3]},
            {"layer": 2, "hits": [4, 5, 6]},
        ]
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "more than once"):
            self.judge(coverage_dump(layers, inputs=["a", "a"]))
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "declared coverage suite"):
            self.judge(coverage_dump(layers, inputs=["a"]), min_inputs=2)

    def test_an_unknown_layer_is_refused(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 2, 3]},
                {"layer": 2, "hits": [4, 5, 6]},
                {"layer": 3, "hits": [1]},
            ]
        )
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "outside the layout"):
            self.judge(dump)

    def test_a_span_must_be_contiguous(self):
        layout = synthetic_layout(
            spans=[{"first": 0, "count": 1, "experts": 0}, {"first": 2, "count": 1, "experts": 3}]
        )
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "contiguous"):
            NUMERICS.layout_experts(layout, "layout.json")

    def test_the_dump_and_layout_must_name_the_same_model(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 2, 3]},
                {"layer": 2, "hits": [4, 5, 6]},
            ],
            model="toy/other",
        )
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "names"):
            self.judge(dump)

    def test_the_command_writes_the_judgement(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 2, 3]},
                {"layer": 2, "hits": [4, 5, 6]},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            dump_path = Path(directory) / "coverage.json"
            layout_path = Path(directory) / "layout.json"
            dump_path.write_text(json.dumps(dump))
            layout_path.write_text(json.dumps(self.layout))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(
                    NUMERICS.main(
                        [
                            "coverage",
                            "--dump",
                            str(dump_path),
                            "--layout",
                            str(layout_path),
                            "--min-inputs",
                            "2",
                        ]
                    ),
                    0,
                )
        self.assertIn("PASS coverage", out.getvalue())

    def test_the_command_fails_below_the_registry_bar(self):
        dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 1, 0]},
                {"layer": 2, "hits": [1, 0, 0]},
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            dump_path = Path(directory) / "coverage.json"
            layout_path = Path(directory) / "layout.json"
            dump_path.write_text(json.dumps(dump))
            layout_path.write_text(json.dumps(self.layout))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = NUMERICS.main(
                    [
                        "coverage",
                        "--dump",
                        str(dump_path),
                        "--layout",
                        str(layout_path),
                        "--min-inputs",
                        "2",
                    ]
                )
        self.assertEqual(code, 1)
        output = out.getvalue()
        self.assertIn("FAIL coverage", output)
        self.assertIn("50.00%", output)
        self.assertIn("93.00%", output)


class CoveragePrerequisiteTests(unittest.TestCase):
    """What ci does with the per-model [[coverage]] rows and the aggregate flag."""

    def setUp(self):
        self.dump = coverage_dump(
            [
                {"layer": 0, "hits": []},
                {"layer": 1, "hits": [1, 1, 0]},
                {"layer": 2, "hits": [1, 0, 0]},
            ]
        )
        layout = mock.patch.object(
            NUMERICS, "find_layout", return_value=(synthetic_layout(), "layout.json")
        )
        layout.start()
        self.addCleanup(layout.stop)

    def registry(self, coverage_flag: bool = True, bar: float = 0.5, rows: list | None = None, models: list | None = None) -> dict:
        return {
            "prerequisites": {
                "anchor": False,
                "declaration_check": True,
                "coverage": coverage_flag,
                "coverage_slots": bar,
                "monitors": False,
            },
            "coverage": [coverage_row()] if rows is None else rows,
            "monitors": [],
            "config": [captured()] if models is None else models,
        }

    def reader(self, dump: dict | None = None):
        return lambda path: json.dumps(self.dump if dump is None else dump).encode()

    def test_no_rows_and_a_false_flag_reads_nothing(self):
        def read(path: str) -> bytes | None:
            self.fail(f"a registry with no coverage row read {path}")

        self.assertEqual(NUMERICS.coverage_findings(self.registry(False, rows=[]), read), [])

    def test_a_committed_row_is_judged_even_when_the_flag_is_false(self):
        findings = NUMERICS.coverage_findings(self.registry(False), lambda path: None)
        self.assertTrue(any("does not exist" in finding for finding in findings), findings)
        self.assertTrue(any("numerics/coverage/toy.json" in finding for finding in findings), findings)

    def test_a_missing_dump_is_named_with_its_model(self):
        findings = NUMERICS.coverage_findings(self.registry(), lambda path: None)
        self.assertTrue(any("does not exist" in finding for finding in findings), findings)
        self.assertTrue(any("numerics/coverage/toy.json" in finding for finding in findings), findings)
        self.assertTrue(any(f"[[coverage]] {MODEL!r}" in finding for finding in findings), findings)

    def test_a_dump_below_the_bar_names_both_fractions(self):
        findings = NUMERICS.coverage_findings(self.registry(bar=0.93), self.reader())
        self.assertTrue(any("50.00%" in finding and "93.00%" in finding for finding in findings), findings)

    def test_a_dump_at_the_bar_passes(self):
        self.assertEqual(NUMERICS.coverage_findings(self.registry(bar=0.5), self.reader()), [])

    def test_a_dump_for_another_model_is_refused(self):
        findings = NUMERICS.coverage_findings(
            self.registry(), self.reader(dict(self.dump, model="toy/other"))
        )
        self.assertTrue(any("names 'toy/other'" in finding for finding in findings), findings)

    def test_the_true_flag_names_every_model_without_a_passing_row(self):
        # The aggregate is a claim about every captured logprobs model: a true flag that claims
        # more than the rows hold is a finding, and it names the model.
        models = [captured(MODEL), captured("toy/other")]
        findings = NUMERICS.coverage_findings(self.registry(models=models), self.reader())
        self.assertTrue(
            any("prerequisites.coverage is true" in finding and "toy/other" in finding for finding in findings),
            findings,
        )

    def test_a_true_flag_with_no_rows_names_every_captured_model(self):
        models = [captured(MODEL), captured("toy/other")]
        findings = NUMERICS.coverage_findings(self.registry(rows=[], models=models), self.reader())
        self.assertTrue(any("toy/other" in finding for finding in findings), findings)

    def test_a_false_flag_with_an_uncovered_model_is_consistent(self):
        models = [captured(MODEL), captured("toy/other")]
        self.assertEqual(
            NUMERICS.coverage_findings(self.registry(False, rows=[], models=models), self.reader()),
            [],
        )


def monitor_dump(layer_values, experts, model: str = MODEL, probe: dict | None = None) -> dict:
    return {
        "schema": NUMERICS.MONITORS_SCHEMA,
        "model": model,
        "probe": probe or {"name": "p", "prompt_ids": [1, 2], "position": 0},
        "sample": {"width": 4, "indices": [0, 1]},
        "layers": [
            {
                "layer": 0,
                "values": layer_values,
                "experts": [{"expert": expert, "values": values} for expert, values in experts],
            }
        ],
    }


def monitor_layers_dump(layers: list, model: str = MODEL, probe: dict | None = None) -> dict:
    return {
        "schema": NUMERICS.MONITORS_SCHEMA,
        "model": model,
        "probe": probe or {"name": "p", "prompt_ids": [1, 2], "position": 0},
        "sample": {"width": 4, "indices": [0, 1]},
        "layers": layers,
    }


def monitor_layer(number: int, values: list, experts: list) -> dict:
    return {
        "layer": number,
        "values": values,
        "experts": [
            {"expert": expert, "values": expert_values}
            for expert, expert_values in experts
        ],
    }


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.reference = monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (1, [0.0, 2.0])])

    def test_the_reference_against_itself_is_zero(self):
        result = NUMERICS.judge_monitors(self.reference, self.reference, 1e-3, 1e-3)
        self.assertTrue(result["pass"])
        self.assertEqual(result["layers"][0]["rel_error"], 0.0)
        self.assertTrue(all(expert["rel_error"] == 0.0 for expert in result["layers"][0]["experts"]))
        self.assertIn("PASS monitors", NUMERICS.render_monitors(result))

    def test_an_expert_bar_can_fail_while_the_layer_bar_passes(self):
        candidate = monitor_dump([1.0, 2.0], [(0, [1.1, 0.0]), (1, [0.0, 2.0])])
        result = NUMERICS.judge_monitors(self.reference, candidate, 1.0, 0.05)
        self.assertFalse(result["pass"])
        self.assertTrue(result["layers"][0]["ok"])
        self.assertFalse(next(expert for expert in result["layers"][0]["experts"] if expert["expert"] == 0)["ok"])
        self.assertIn("expert 0", result["failures"][0])

    def test_a_layer_bar_can_fail_on_its_own(self):
        candidate = monitor_dump([1.2, 2.0], [(0, [1.0, 0.0]), (1, [0.0, 2.0])])
        result = NUMERICS.judge_monitors(self.reference, candidate, 0.05, 1.0)
        self.assertFalse(result["pass"])
        self.assertFalse(result["layers"][0]["ok"])
        self.assertIn("layer 0", result["failures"][0])

    def test_a_zero_reference_sample_is_refused_not_divided_by(self):
        reference = monitor_dump([0.0, 0.0], [(0, [0.0, 0.0]), (1, [0.0, 0.0])])
        candidate = monitor_dump([1.0, 0.0], [(0, [0.0, 0.0]), (1, [0.0, 0.0])])
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "all zeros"):
            NUMERICS.judge_monitors(reference, candidate, 1.0, 1.0)

    def test_a_different_probe_sample_or_model_is_refused(self):
        other_probe = monitor_dump(
            [1.0, 2.0],
            [(0, [1.0, 0.0]), (1, [0.0, 2.0])],
            probe={"name": "q", "prompt_ids": [1, 2], "position": 0},
        )
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "different probes"):
            NUMERICS.judge_monitors(self.reference, other_probe, 1.0, 1.0)
        other_model = monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (1, [0.0, 2.0])], model="toy/other")
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "reference names"):
            NUMERICS.judge_monitors(self.reference, other_model, 1.0, 1.0)
        other_sample = json.loads(json.dumps(self.reference))
        other_sample["sample"]["indices"] = [0, 2]
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "sampled different"):
            NUMERICS.judge_monitors(self.reference, other_sample, 1.0, 1.0)

    def test_a_missing_layer_is_refused_and_a_missing_expert_is_reported(self):
        missing_expert = json.loads(json.dumps(self.reference))
        missing_expert["layers"][0]["experts"].pop()
        result = NUMERICS.judge_monitors(self.reference, missing_expert, 0.05, 0.05)
        self.assertTrue(result["pass"])
        self.assertEqual(result["routing"]["layers_differing"], 1)
        self.assertEqual(result["routing"]["experts_unshared"], 1)
        self.assertEqual(result["routing"]["reference_only"], 1)
        self.assertEqual(result["routing"]["candidate_only"], 0)
        self.assertEqual(result["routing"]["per_layer"][0]["reference_only"], [1])
        markdown = NUMERICS.render_monitors(result)
        self.assertIn("routing: 1 of 1 layer(s)", markdown)
        self.assertIn("layer 0: reference only [1]", markdown)
        missing_layer = json.loads(json.dumps(self.reference))
        missing_layer["layers"].append(dict(missing_layer["layers"][0], layer=1))
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "same layers"):
            NUMERICS.judge_monitors(self.reference, missing_layer, 1.0, 1.0)

    def test_an_unshared_expert_on_every_layer_passes_and_is_reported(self):
        # The router may send a row to a different expert on a slightly different hidden state.
        # That is a measured fact about the candidate, not a refusal: the shared expert still
        # matches and the pair is judged on the bars.
        reference = monitor_layers_dump(
            [
                monitor_layer(0, [1.0, 2.0], [(0, [1.0, 0.0]), (1, [0.0, 2.0])]),
                monitor_layer(1, [2.0, 1.0], [(2, [2.0, 0.0]), (3, [0.0, 1.0])]),
            ]
        )
        candidate = monitor_layers_dump(
            [
                monitor_layer(0, [1.0, 2.0], [(0, [1.0, 0.0]), (4, [0.0, 2.0])]),
                monitor_layer(1, [2.0, 1.0], [(2, [2.0, 0.0]), (5, [0.0, 1.0])]),
            ]
        )
        result = NUMERICS.judge_monitors(reference, candidate, 1e-3, 1e-3)
        self.assertTrue(result["pass"], result["failures"])
        routing = result["routing"]
        self.assertEqual(routing["layers_differing"], 2)
        self.assertEqual(routing["experts_unshared"], 4)
        self.assertEqual(routing["reference_only"], 2)
        self.assertEqual(routing["candidate_only"], 2)
        self.assertEqual(routing["experts_shared"], 2)
        self.assertEqual(
            [(layer["layer"], layer["shared"], layer["reference_only"], layer["candidate_only"])
             for layer in routing["per_layer"]],
            [(0, 1, [1], [4]), (1, 1, [3], [5])],
        )
        self.assertEqual(result["maxima"]["layer"], 0.0)
        self.assertEqual(result["maxima"]["expert"], 0.0)
        markdown = NUMERICS.render_monitors(result)
        self.assertIn("PASS monitors", markdown)
        self.assertIn("2 of 2 layer(s) routed a row to a different expert", markdown)
        self.assertIn("4 expert(s) unshared", markdown)
        self.assertIn("layer 0: reference only [1]; candidate only [4]", markdown)
        self.assertIn("layer 1: reference only [3]; candidate only [5]", markdown)
        self.assertEqual(json.loads(json.dumps(result))["routing"]["layers_differing"], 2)

    def test_the_bars_still_fail_over_the_shared_experts(self):
        candidate = monitor_dump([1.0, 2.0], [(0, [1.2, 0.0]), (2, [0.0, 2.0])])
        result = NUMERICS.judge_monitors(self.reference, candidate, 1.0, 0.05)
        self.assertFalse(result["pass"])
        self.assertTrue(result["layers"][0]["ok"])
        self.assertEqual(len(result["failures"]), 1)
        self.assertIn("layer 0 expert 0", result["failures"][0])
        self.assertNotIn("expert 1", result["failures"][0])
        self.assertNotIn("expert 2", result["failures"][0])
        self.assertEqual(result["routing"]["experts_unshared"], 2)
        self.assertEqual(
            [expert["expert"] for expert in result["layers"][0]["experts"]], [0]
        )

    def test_a_layer_bar_still_fails_beside_a_routing_divergence(self):
        candidate = monitor_dump([1.2, 2.0], [(0, [1.0, 0.0]), (2, [0.0, 2.0])])
        result = NUMERICS.judge_monitors(self.reference, candidate, 0.05, 1.0)
        self.assertFalse(result["pass"])
        self.assertFalse(result["layers"][0]["ok"])
        self.assertTrue(any("layer 0:" in failure for failure in result["failures"]))
        self.assertEqual(result["routing"]["layers_differing"], 1)

    def test_an_undefined_relative_error_is_still_refused(self):
        reference = monitor_dump([1.0, 2.0], [(0, [0.0, 0.0]), (1, [0.0, 2.0])])
        candidate = monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (1, [0.0, 2.0])])
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "all zeros"):
            NUMERICS.judge_monitors(reference, candidate, 1.0, 1.0)

    def test_a_malformed_dump_is_refused(self):
        for mutate, needle in [
            (lambda dump: dump.update(schema="other"), "schema"),
            (lambda dump: dump["sample"].update(indices=[1, 1]), "strictly increasing"),
            (lambda dump: dump["layers"][0]["values"].__setitem__(0, math.inf), "non-finite"),
            (lambda dump: dump["layers"][0]["experts"].append({"expert": 0, "values": [1.0, 2.0]}), "more than once"),
        ]:
            dump = json.loads(json.dumps(self.reference))
            mutate(dump)
            with self.assertRaisesRegex(NUMERICS.NumericsFailure, needle):
                NUMERICS.check_monitor_dump(dump, "dump.json")

    def test_bars_must_be_finite_and_positive(self):
        for layer_bar, expert_bar in [(0.0, 1.0), (float("nan"), 1.0), (1.0, -1.0)]:
            with self.assertRaisesRegex(NUMERICS.NumericsFailure, "finite and positive"):
                NUMERICS.judge_monitors(self.reference, self.reference, layer_bar, expert_bar)

    def test_the_command_fails_a_bar_and_writes_its_result(self):
        candidate = monitor_dump([1.2, 2.0], [(0, [1.1, 0.0]), (1, [0.0, 2.0])])
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.json"
            candidate_path = Path(directory) / "candidate.json"
            result_path = Path(directory) / "result.json"
            reference_path.write_text(json.dumps(self.reference))
            candidate_path.write_text(json.dumps(candidate))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(
                    NUMERICS.main(
                        [
                            "monitors",
                            "--reference",
                            str(reference_path),
                            "--candidate",
                            str(candidate_path),
                            "--layer-bar",
                            "0.05",
                            "--expert-bar",
                            "0.05",
                            "--json",
                            str(result_path),
                        ]
                    ),
                    1,
                )
            result = json.loads(result_path.read_text())
        self.assertFalse(result["pass"])
        self.assertEqual(result["schema"], NUMERICS.MONITOR_RESULT_SCHEMA)
        self.assertEqual(result["routing"]["layers_differing"], 0)
        self.assertEqual(result["routing"]["experts_unshared"], 0)
        self.assertAlmostEqual(result["maxima"]["layer"], 0.2 / math.sqrt(5.0), places=15)
        self.assertIn("FAIL monitors", out.getvalue())
        self.assertIn("routing:", out.getvalue())


class MonitorPrerequisiteTests(unittest.TestCase):
    """What ci does with the per-model [[monitors]] rows and the aggregate flag."""

    def setUp(self):
        self.reference = monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (1, [0.0, 2.0])])
        self.candidate = monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (2, [0.0, 2.0])])
        self.row = monitors_row()

    def registry(self, monitors_flag: bool = True, rows: list | None = None, models: list | None = None, bars: float = 0.5) -> dict:
        return {
            "prerequisites": {
                "anchor": False,
                "declaration_check": True,
                "coverage": False,
                "coverage_slots": 0.93,
                "monitors": monitors_flag,
            },
            "coverage": [],
            "monitors": [monitors_row(bars=bars)] if rows is None else rows,
            "config": [captured()] if models is None else models,
        }

    def reader(self, documents: dict[str, dict] | None = None):
        docs = documents or {
            self.row["reference"]: self.reference,
            self.row["candidate"]: self.candidate,
        }
        return lambda path: json.dumps(docs[path]).encode() if path in docs else None

    def test_a_row_judges_the_pair_it_names_and_reports_routing(self):
        result, findings = NUMERICS.monitor_row_result(self.row, self.reader())
        self.assertEqual(findings, [])
        self.assertTrue(result["pass"])
        self.assertEqual(result["routing"]["layers_differing"], 1)
        self.assertEqual(result["routing"]["experts_unshared"], 2)
        self.assertEqual(result["maxima"]["layer"], 0.0)

    def test_no_rows_and_a_false_flag_reads_nothing(self):
        def read(path: str) -> bytes | None:
            self.fail(f"a registry with no monitors row read {path}")

        self.assertEqual(NUMERICS.monitors_findings(self.registry(False, rows=[]), read), [])

    def test_a_committed_row_is_judged_even_when_the_flag_is_false(self):
        findings = NUMERICS.monitors_findings(self.registry(False), lambda path: None)
        self.assertTrue(any("does not exist" in finding for finding in findings), findings)

    def test_a_missing_row_dump_names_its_model(self):
        findings = NUMERICS.monitors_findings(self.registry(), lambda path: None)
        self.assertTrue(any("does not exist" in finding for finding in findings), findings)
        self.assertTrue(any(self.row["reference"] in finding for finding in findings), findings)
        self.assertTrue(any(self.row["candidate"] in finding for finding in findings), findings)
        self.assertTrue(any(f"[[monitors]] {MODEL!r}" in finding for finding in findings), findings)

    def test_a_mismatched_pair_is_refused(self):
        other = monitor_dump([1.0, 2.0], [(0, [1.0, 0.0])], model="toy/other")
        findings = NUMERICS.monitors_findings(
            self.registry(),
            self.reader({self.row["reference"]: self.reference, self.row["candidate"]: other}),
        )
        self.assertTrue(any("names 'toy/other'" in finding for finding in findings), findings)

    def test_an_over_bar_row_fails(self):
        candidate = monitor_dump([1.2, 2.0], [(0, [1.1, 0.0]), (2, [0.0, 2.0])])
        findings = NUMERICS.monitors_findings(
            self.registry(bars=0.05),
            self.reader({self.row["reference"]: self.reference, self.row["candidate"]: candidate}),
        )
        self.assertTrue(any("layer bar" in finding for finding in findings), findings)
        self.assertTrue(any("expert bar" in finding for finding in findings), findings)

    def test_the_true_flag_names_every_model_without_a_passing_row(self):
        models = [captured(MODEL), captured("toy/other")]
        findings = NUMERICS.monitors_findings(self.registry(models=models), self.reader())
        self.assertTrue(
            any("prerequisites.monitors is true" in finding and "toy/other" in finding for finding in findings),
            findings,
        )

    def test_a_false_flag_with_an_uncovered_model_is_consistent(self):
        models = [captured(MODEL), captured("toy/other")]
        self.assertEqual(
            NUMERICS.monitors_findings(self.registry(False, rows=[], models=models), self.reader()),
            [],
        )

    def test_a_malformed_row_is_refused_at_parse(self):
        cases = [
            dict(self.row, reference="docs/numerics/ref.json"),
            dict(self.row, reference="numerics/monitors/../ref.json"),
            dict(self.row, candidate="numerics/monitors/cand.txt"),
            dict(self.row, layer_bar=0.0),
            dict(self.row, expert_bar=-1.0),
            dict(self.row, layer_bar=float("nan")),
            dict(self.row, layer_bar=True),
            {key: value for key, value in self.row.items() if key != "candidate"},
            {key: value for key, value in self.row.items() if key != "model"},
            dict(self.row, surprise=1),
        ]
        for row in cases:
            with self.subTest(row=row):
                with self.assertRaises(NUMERICS.NumericsFailure, msg=str(row)) as refused:
                    NUMERICS.parse_registry(registry_text(monitors=[row]), "toy")
                self.assertIn("monitors", str(refused.exception))
                if "model" in row:
                    self.assertIn(MODEL, str(refused.exception), "a malformed row names its model")
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "not a table"):
            NUMERICS.monitors_row("nope", "toy")

    def test_one_row_a_model_is_refused(self):
        with self.assertRaisesRegex(NUMERICS.NumericsFailure, "one row a model"):
            NUMERICS.parse_registry(
                registry_text(monitors=[monitors_row(), monitors_row()]), "toy"
            )


class MonitorCiTests(unittest.TestCase):
    """ci with the aggregate flag false, true, a missing row and an over-bar row."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        (self.repo / "numerics" / "monitors").mkdir(parents=True)
        (self.repo / "numerics" / "references").mkdir(parents=True)
        self.old = NUMERICS.synthetic_reference([5, 6], -0.25)
        (self.repo / "numerics" / "references" / "q.json").write_text(
            NUMERICS.render_reference(self.old)
        )
        self.reference = monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (1, [0.0, 2.0])], model="m")
        self.candidate = monitor_dump([1.0, 2.0], [(0, [1.0, 0.0]), (2, [0.0, 2.0])], model="m")
        self.row = monitors_row(model="m")
        self.write_registry(False, [])
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "base")
        self.base = self.git("rev-parse", "HEAD")

    def git(self, *args: str) -> str:
        return subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.invalid",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def config_toml(self) -> str:
        entry = NUMERICS.synthetic_entry(1, self.old)
        return "\n[[config]]\n" + "\n".join(
            f"{key} = {json.dumps(value)}" for key, value in entry.items()
        ) + "\n"

    def write_registry(self, monitors_flag: bool, rows: list[dict]) -> None:
        (self.repo / "numerics" / "registry.toml").write_text(
            registry_text(monitors=rows, monitors_flag=monitors_flag) + self.config_toml()
        )

    def write_dump(self, name: str, dump: dict) -> None:
        (self.repo / "numerics" / "monitors" / name).write_text(json.dumps(dump))

    def ci(self) -> tuple[int, str]:
        with (
            mock.patch.object(NUMERICS, "REPO", self.repo),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            code = NUMERICS.run_ci(self.base)
        return code, out.getvalue()

    def test_no_rows_and_a_false_flag_passes(self):
        code, out = self.ci()
        self.assertEqual(code, 0, out)

    def test_a_true_flag_without_a_row_is_refused_with_the_model_named(self):
        self.write_registry(True, [])
        code, out = self.ci()
        self.assertEqual(code, 1)
        self.assertIn("prerequisites.monitors is true", out)
        self.assertIn("'m'", out)

    def test_a_true_flag_passes_a_row_inside_the_bars(self):
        self.write_registry(True, [self.row])
        self.write_dump("ref.json", self.reference)
        self.write_dump("cand.json", self.candidate)
        code, out = self.ci()
        self.assertEqual(code, 0, out)
        self.assertIn("monitors row m:", out)
        self.assertIn("monitors routing m: 1 of 1 layer(s) differ", out)
        self.assertIn("achieved maxima", out)

    def test_a_false_flag_still_judges_a_committed_row(self):
        # A committed row is evidence: it is judged whether or not the aggregate flag is on.
        self.write_registry(False, [self.row])
        code, out = self.ci()
        self.assertEqual(code, 1)
        self.assertIn("does not exist", out)

    def test_a_true_flag_fails_a_missing_row_dump(self):
        self.write_registry(True, [self.row])
        self.write_dump("ref.json", self.reference)
        code, out = self.ci()
        self.assertEqual(code, 1)
        self.assertIn("does not exist", out)
        self.assertIn(self.row["candidate"], out)

    def test_a_true_flag_fails_a_mismatched_row(self):
        self.write_registry(True, [self.row])
        self.write_dump("ref.json", self.reference)
        self.write_dump("cand.json", monitor_dump([1.0, 2.0], [(0, [1.0, 0.0])], model="toy/other"))
        code, out = self.ci()
        self.assertEqual(code, 1)
        self.assertIn("names 'toy/other'", out)

    def test_a_true_flag_fails_an_over_bar_row(self):
        self.write_registry(True, [monitors_row(model="m", bars=0.05)])
        self.write_dump("ref.json", self.reference)
        self.write_dump("cand.json", monitor_dump([1.2, 2.0], [(0, [1.1, 0.0]), (2, [0.0, 2.0])], model="m"))
        code, out = self.ci()
        self.assertEqual(code, 1)
        self.assertIn("layer bar", out)
        self.assertIn("expert bar", out)

    def test_one_row_a_model_is_refused(self):
        self.write_registry(True, [self.row, self.row])
        with (
            mock.patch.object(NUMERICS, "REPO", self.repo),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            code = NUMERICS.main(["ci", "--base", self.base])
        self.assertEqual(code, 1)
        self.assertIn("one row a model", error.getvalue())


def synthetic_text_reference(text: str = "abc") -> dict:
    reference = {
        "schema": NUMERICS.REFERENCE_SCHEMA,
        "evidence": "text",
        "key": {"model": "m", "world": 4, "profile": "reference", "plan": "tp", "arch": "sm_121a"},
        "settings": {"max_tokens": len(text)},
        "forced_from": None,
        "prompts": [
            {"name": "p", "prompt_ids": [1, 2], "text": text, "finish_reason": "length"}
        ],
    }
    reference["fingerprint"] = NUMERICS.fingerprint(reference)
    return reference


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.old = NUMERICS.synthetic_reference([5, 6], -0.25)
        self.flipped = NUMERICS.synthetic_reference([5, 6], next_ulp(-0.25))
        self.anchor = NUMERICS.synthetic_reference([9], -0.5)
        self.path = "numerics/references/q.json"
        self.anchor_path = NUMERICS.ANCHOR_PATH
        self.report = "docs/numerics/m/2.md"
        layout = mock.patch.object(
            NUMERICS, "find_layout", return_value=(synthetic_layout(model="m"), "layout.json")
        )
        layout.start()
        self.addCleanup(layout.stop)

    def files(self, reference: dict, report: bytes = b"report") -> Callable[[str], bytes | None]:
        store = {
            self.path: NUMERICS.render_reference(reference).encode(),
            self.anchor_path: NUMERICS.render_reference(self.anchor).encode(),
            self.report: report,
            **evidence_files("m"),
        }
        return store.get

    def registries(self, head_entry: dict, anchor: bool = True, flag: bool = False):
        base = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], anchor)
        head = NUMERICS.synthetic_registry([head_entry], anchor)
        for registry in (base, head):
            registry["prerequisites"] = {
                "anchor": flag,
                "declaration_check": True,
                "coverage": True,
                "coverage_slots": 0.93,
                "monitors": True,
            }
            registry["coverage"] = [coverage_row("m")]
            registry["monitors"] = [monitors_row("m")]
        return base, head

    def test_a_bump_uses_its_own_models_anchor_not_a_global_flag(self):
        entry = NUMERICS.synthetic_entry(2, self.flipped, report=self.report)
        base, head = self.registries(entry)
        forced = {"m": {self.anchor["fingerprint"]}}
        findings = NUMERICS.declaration_findings(
            base, head, self.files(self.old), self.files(self.flipped), {self.path: "M", self.report: "A"}, forced
        )
        self.assertEqual(findings, [], findings)
        self.assertFalse(head["prerequisites"]["anchor"], "the global flag stays false")

    def test_a_bump_without_its_models_anchor_names_the_model(self):
        entry = NUMERICS.synthetic_entry(2, self.flipped, report=self.report)
        base, head = self.registries(entry)
        head["anchor"] = []
        findings = NUMERICS.declaration_findings(
            base,
            head,
            self.files(self.old),
            self.files(self.flipped),
            {self.path: "M", self.report: "A"},
            {"m": {self.anchor["fingerprint"]}},
        )
        self.assertTrue(any("'m' has no [[anchor]]" in finding for finding in findings), findings)

    def test_a_bump_without_a_teacher_forced_capture_is_refused(self):
        entry = NUMERICS.synthetic_entry(2, self.flipped, report=self.report)
        base, head = self.registries(entry)
        findings = NUMERICS.declaration_findings(
            base, head, self.files(self.old), self.files(self.flipped), {self.path: "M", self.report: "A"}, {}
        )
        self.assertTrue(any("teacher-forced" in finding for finding in findings), findings)

    def test_a_text_evidence_bump_is_refused_even_with_an_anchor(self):
        text = synthetic_text_reference()
        entry = dict(NUMERICS.synthetic_entry(2, text, report=self.report), evidence="text")
        base, head = self.registries(entry)
        findings = NUMERICS.declaration_findings(
            base,
            head,
            self.files(self.old),
            self.files(text),
            {self.path: "M", self.report: "A"},
            {"m": {self.anchor["fingerprint"]}},
        )
        self.assertTrue(any("text reference holds no token ids" in finding for finding in findings), findings)

    def test_anchor_coverage_is_per_model_and_excuses_text_evidence(self):
        registry = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], False)
        coverage = NUMERICS.anchor_coverage(registry, {})
        self.assertEqual(coverage["uncovered"], {"m": "the registry holds no [[anchor]] for it"})
        self.assertEqual(coverage["exempt"], [])

        text = synthetic_text_reference()
        text_entry = dict(NUMERICS.synthetic_entry(1, text), evidence="text")
        registry = NUMERICS.synthetic_registry([text_entry], False)
        coverage = NUMERICS.anchor_coverage(registry, {})
        self.assertEqual(coverage["bumpable"], [])
        self.assertEqual(coverage["exempt"], ["m"])

    def test_anchor_findings_require_a_committed_forced_capture(self):
        registry = NUMERICS.synthetic_registry([NUMERICS.synthetic_entry(1, self.old)], True)
        capture = {
            "schema": NUMERICS.REFERENCE_SCHEMA,
            "evidence": "logprobs",
            "key": {"model": "m", "world": 4, "profile": "reference", "plan": "tp", "arch": "sm_121a"},
            "forced_from": {"fingerprint": self.anchor["fingerprint"], "reference": "references/anchor.json"},
        }
        read = {"numerics/captures/m/forced.json": json.dumps(capture).encode()}.get
        findings, coverage = NUMERICS.anchor_findings(
            registry, read, ["numerics/captures/m/forced.json"]
        )
        self.assertEqual(findings, [], findings)
        self.assertEqual(sorted(coverage["covered"]), ["m"])

        findings, _ = NUMERICS.anchor_findings(registry, read, [])
        self.assertTrue(any("no committed capture" in finding for finding in findings), findings)
        registry["prerequisites"]["anchor"] = True
        findings, _ = NUMERICS.anchor_findings(registry, read, [])
        self.assertTrue(any("prerequisites.anchor is true" in finding for finding in findings), findings)


if __name__ == "__main__":
    unittest.main()
