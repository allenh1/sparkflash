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

"""Tests for scripts/serve_window_probe.py (host-only; no device, no sockets)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import serve_window_probe as probe

PROMPT_LEN = 130
FIRST_STEP_POSITION = 129
REFERENCE_IDS = [491, 367, 201, 80, 3302, 201, 54, 201, 4341]
TOKEN_IDS = [111, 222, 333, 444, 555, 666, 777, 888, 999]
DUMP_SHA = "a" * 64
BIN_SHA = "b" * 64


def prompt_ids():
    return list(range(1000, 1000 + PROMPT_LEN))


def forward_ids():
    return prompt_ids()[:-1] + TOKEN_IDS[:8] + [REFERENCE_IDS[-1]]


def embed_value(rank: int, claimed: bool):
    return {
        "schema": "sparkflash-tp4-forward-embed/1",
        "stage": "embedding-gathered",
        "rank": rank,
        "world": 4,
        "prompt_token": TOKEN_IDS[0],
        "prompt_len": PROMPT_LEN,
        "first_step_position": FIRST_STEP_POSITION,
        "decode_positions": 9,
        "reference_ids": REFERENCE_IDS,
        "prompt_ids": prompt_ids(),
        "forward_ids": forward_ids(),
        "prefill_limit": None,
        "evidence_written": "loop-complete, position-complete, head-complete, token-consensus-complete",
        "token_claimed": claimed,
        "tokens": {
            "ids": TOKEN_IDS,
            "positions": [FIRST_STEP_POSITION + index for index in range(9)],
            "consensus": True,
            "claimable": [True] * 9,
            "matches_reference": [True] * 9,
            "argmax_logits": [10.0 + index for index in range(9)],
            "top1_margins": [1.0] * 9,
        },
        "head": {"argmax": {"token": TOKEN_IDS[-1], "value": 18.0, "claimable": True}},
    }


def token_value(rank: int, claimed: bool):
    positions = [FIRST_STEP_POSITION + index for index in range(9)]
    return {
        "schema": "sparkflash-tp4-token/1",
        "rank": rank,
        "world": 4,
        "forward": "run",
        "error": None,
        "tokens": {
            "ids": TOKEN_IDS,
            "positions": positions,
            "argmax_logits": [10.0 + index for index in range(9)],
            "top1_margins": [1.0] * 9,
        },
        "positions_run": 9,
        "wall_ms": [1.0] * 9,
        "step_ms": {"samples": [1.0] * 9},
        "reduce_calls_per_step": 80,
        "launch_census": {"routed_experts_down": True},
        "logits_dump": {
            "path": f"/node/tp4-logits-r{rank}.jsonl",
            "bin": f"/node/tp4-logits-r{rank}.jsonl.bin",
            "sha256": DUMP_SHA,
            "bin_sha256": BIN_SHA,
            "dtype": "f32-le",
            "vocab": 129280,
            "positions": positions,
            "rows": 9,
        },
    }


def write_window(root: str, claimed: bool, prefill_limit=None, forward_ok=True):
    for rank in range(4):
        rank_dir = os.path.join(root, f"rank{rank}")
        os.makedirs(rank_dir)
        embed = embed_value(rank, claimed)
        embed["prefill_limit"] = prefill_limit
        if not forward_ok:
            embed["forward_ids"] = embed["forward_ids"][:-1]
        with open(os.path.join(rank_dir, f"tp4-forward-embed-r{rank}.json"), "w") as handle:
            json.dump(embed, handle)
        with open(os.path.join(rank_dir, f"tp4-token-r{rank}.json"), "w") as handle:
            json.dump(token_value(rank, claimed), handle)


def load(root: str) -> dict:
    window = probe.load_window(root)
    window["window"] = root
    return window


def sse_text(text: str) -> bytes:
    frames = []
    size, remainder = divmod(len(text), 9)
    pieces = []
    offset = 0
    for index in range(9):
        width = size + (1 if index < remainder else 0)
        pieces.append(text[offset:offset + width])
        offset += width
    for piece in pieces:
        frames.append(
            {
                "object": "text_completion",
                "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                "usage": None,
            }
        )
    frames.append(
        {
            "object": "text_completion",
            "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
            "usage": None,
        }
    )
    frames.append(
        {
            "object": "text_completion",
            "choices": [],
            "usage": {
                "prompt_tokens": PROMPT_LEN,
                "completion_tokens": 9,
                "total_tokens": PROMPT_LEN + 9,
            },
        }
    )
    raw = "".join(f"data: {json.dumps(frame, separators=(',', ':'))}\n\n" for frame in frames)
    return (raw + "data: [DONE]\n\n").encode()


def fake_responses(text="Hello device", mode="asserted"):
    buffered_body = {
        "id": "cmpl-dsv41-00000000",
        "object": "text_completion",
        "model": probe.MODEL,
        "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": PROMPT_LEN,
            "completion_tokens": 9,
            "total_tokens": PROMPT_LEN + 9,
        },
    }
    headers = {
        probe.DEVICE_SOURCE_HEADER: probe.DEVICE_SOURCE_VALUE,
        probe.DEVICE_CLAIM_HEADER: probe.claim_label(mode),
    }
    buffered = (200, dict(headers), json.dumps(buffered_body).encode())
    stream = (200, dict(headers), sse_text(text))
    return buffered, stream


class WindowFixtureTest(unittest.TestCase):
    def test_asserted_claim_holds_only_when_every_rank_asserts(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=True)
            claim = probe.claim_window(load(root), "asserted")
            self.assertTrue(claim["claimable"], claim["reasons"])
            self.assertEqual(claim["summary"]["token_claimed"], [True] * 4)
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False)
            claim = probe.claim_window(load(root), "asserted")
            self.assertFalse(claim["claimable"])
            self.assertTrue(any("does not assert token_claimed" in reason for reason in claim["reasons"]))

    def test_derived_claim_holds_on_the_recorded_loop_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False)
            claim = probe.claim_window(load(root), "derived")
            self.assertTrue(claim["claimable"], claim["reasons"])
            self.assertEqual(claim["summary"]["prefill_limit"], None)
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=True)
            self.assertTrue(probe.claim_window(load(root), "derived")["claimable"])

    def test_derived_claim_refuses_a_prefill_limited_replay(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False, prefill_limit=1)
            claim = probe.claim_window(load(root), "derived")
            self.assertFalse(claim["claimable"])
            reasons = "\n".join(claim["reasons"])
            self.assertIn("prefill_limit", reasons)
            self.assertIn("not the full nine-position acceptance window", reasons)

    def test_derived_claim_refuses_a_forward_id_row_shortfall(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False, forward_ok=False)
            claim = probe.claim_window(load(root), "derived")
            self.assertFalse(claim["claimable"])
            self.assertTrue(any("forward_ids" in reason for reason in claim["reasons"]))

    def test_inspect_writes_the_claim_record_and_returns_the_refusal_code(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False)
            arc = os.path.join(root, "arc")
            code = probe.main(
                ["inspect", "--window", root, "--arc-dir", arc, "--mode", "derived"]
            )
            self.assertEqual(code, 0)
            with open(os.path.join(arc, "window-claim.json"), "r") as handle:
                recorded = json.load(handle)
            self.assertEqual(recorded["claim_mode"], "derived")
            self.assertTrue(recorded["claimable"])
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False, prefill_limit=1)
            arc = os.path.join(root, "arc")
            code = probe.main(
                ["inspect", "--window", root, "--arc-dir", arc, "--mode", "derived"]
            )
            self.assertEqual(code, 2)


class ProbeCheckTest(unittest.TestCase):
    def test_every_check_passes_on_a_matching_pair(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=True)
            buffered, stream = fake_responses()
            result = probe.check_probe(load(root), "asserted", buffered, stream)
            self.assertTrue(result["passed"], result["failures"])
            self.assertEqual(result["buffered_usage"]["total_tokens"], PROMPT_LEN + 9)
            self.assertEqual(len(result["stream_deltas"]), 9)

    def test_a_wrong_claim_header_fails_the_probe(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=True)
            buffered, stream = fake_responses()
            status, headers, raw = buffered
            headers[probe.DEVICE_CLAIM_HEADER] = "derived"
            result = probe.check_probe(load(root), "asserted", (status, headers, raw), stream)
            self.assertFalse(result["passed"])
            self.assertTrue(any("claim header" in failure for failure in result["failures"]))

    def test_a_missing_done_frame_fails_the_probe(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=True)
            buffered, stream = fake_responses()
            status, headers, raw = stream
            result = probe.check_probe(
                load(root), "asserted", buffered, (status, headers, raw.replace(b"data: [DONE]\n\n", b""))
            )
            self.assertFalse(result["passed"])
            self.assertTrue(any("[DONE]" in failure for failure in result["failures"]))

    def test_buffered_and_stream_disagreement_fails_the_probe(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=True)
            buffered, _ = fake_responses("first text")
            _, stream = fake_responses("second text")
            result = probe.check_probe(load(root), "asserted", buffered, stream)
            self.assertFalse(result["passed"])
            self.assertTrue(any("concatenate" in failure for failure in result["failures"]))

    def test_probe_command_refuses_an_unclaimable_window_before_touching_http(self):
        class ExplodingClient:
            def post_json(self, path, body):
                raise AssertionError("the probe must refuse before any HTTP request")

        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False)
            args = probe.build_parser().parse_args(
                [
                    "probe",
                    "--window",
                    root,
                    "--base-url",
                    "http://127.0.0.1:1",
                    "--arc-dir",
                    os.path.join(root, "arc"),
                    "--mode",
                    "asserted",
                ]
            )
            self.assertEqual(probe.probe_command(args, client=ExplodingClient()), 2)


def host_responses(text="host twin answer"):
    """The host path's answer when the device claim refused: no device headers, host stats body."""
    buffered_body = {
        "id": "cmpl-host-00000000",
        "object": "text_completion",
        "model": probe.MODEL,
        "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        "x_deepseek_spark_stats": {"steps": 1, "decode_steps": 4},
    }
    buffered = (200, {}, json.dumps(buffered_body).encode())
    stream = (200, {}, sse_text(text))
    return buffered, stream


class ProbeFallbackTest(unittest.TestCase):
    def test_fallback_checks_pass_on_a_host_twin_answer(self):
        with tempfile.TemporaryDirectory() as root:
            # Refused in both claim forms (limited replay): exactly the fallback input.
            write_window(root, claimed=False, prefill_limit=1)
            buffered, stream = host_responses()
            result = probe.check_fallback(load(root), buffered, stream)
            self.assertTrue(result["passed"], result["failures"])
            self.assertFalse(result["claimable"])

    def test_fallback_fails_if_a_device_header_appears(self):
        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False, prefill_limit=1)
            buffered, stream = host_responses()
            status, headers, raw = buffered
            headers[probe.DEVICE_SOURCE_HEADER] = probe.DEVICE_SOURCE_VALUE
            result = probe.check_fallback(load(root), (status, headers, raw), stream)
            self.assertFalse(result["passed"])
            self.assertTrue(
                any("no device source header" in failure for failure in result["failures"])
            )

    def test_fallback_command_refuses_a_claimable_window_before_http(self):
        class ExplodingClient:
            def post_json(self, path, body):
                raise AssertionError("a claimable window must not be probed as a fallback")

        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=True)
            args = probe.build_parser().parse_args(
                [
                    "probe",
                    "--window",
                    root,
                    "--base-url",
                    "http://127.0.0.1:1",
                    "--arc-dir",
                    os.path.join(root, "arc"),
                    "--mode",
                    "asserted",
                    "--expect",
                    "fallback",
                ]
            )
            self.assertEqual(probe.probe_command(args, client=ExplodingClient()), 2)

    def test_fallback_command_writes_the_record_for_a_refused_window(self):
        class HostClient:
            def post_json(self, path, body):
                self.requests.append((path, body))
                buffered, stream = host_responses()
                return stream if body.get("stream") else buffered

        with tempfile.TemporaryDirectory() as root:
            write_window(root, claimed=False, prefill_limit=1)
            arc = os.path.join(root, "arc")
            args = probe.build_parser().parse_args(
                [
                    "probe",
                    "--window",
                    root,
                    "--base-url",
                    "http://127.0.0.1:1",
                    "--arc-dir",
                    arc,
                    "--mode",
                    "asserted",
                    "--expect",
                    "fallback",
                ]
            )
            client = HostClient()
            client.requests = []
            self.assertEqual(probe.probe_command(args, client=client), 0)
            self.assertTrue(os.path.exists(os.path.join(arc, "FALLBACK.md")))
            with open(os.path.join(arc, "probe-summary.json"), "r") as handle:
                recorded = json.load(handle)
            self.assertEqual(recorded["mode"], "fallback")
            self.assertTrue(recorded["claim_refused"]["asserted"])
            self.assertEqual(recorded["schema"], probe.PROBE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
