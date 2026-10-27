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

"""Tests for the OpenAI surface-gap wire census script (pure helpers and validation)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "openai-surface-census.py"
SPEC = importlib.util.spec_from_file_location("openai_surface_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CENSUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CENSUS)


def valid_census() -> dict:
    """One minimal census document that proves exactly the contracts validate_census checks."""
    run = {
        "run": 0,
        "checks": {"completions_serial.choice_count": True, "auth.models_missing_status": True},
        "probe_wall_ms": {"completions_serial": 10},
        "probes": {
            "completions_serial": {
                "label": "serial",
                "buffered": {"choice_indexes": [0, 1], "usage": {"prompt_tokens": 8, "completion_tokens": 6}},
                "payload_hash": "aaaa",
            },
            "completions_batched": {
                "label": "batched",
                "buffered": {"choice_indexes": [0, 1]},
                "payload_hash": "aaaa",
            },
            "chat_choices": {
                "buffered": {"choice_indexes": [0, 1]},
                "streamed": {"finish_indexes": [0, 1]},
            },
            "response_format_refusals": {
                "json_object": {"status": 400, "param": "response_format"},
                "json_schema": {"status": 400, "param": "response_format"},
            },
            "bearer_auth": {"models_missing": {"status": 401}},
        },
    }
    return {
        "schema": CENSUS.SCHEMA,
        "generated_at": "2026-10-21T00:00:00Z",
        "scope": CENSUS.SCOPE,
        "label": "UNBARED host-path diagnostic",
        "git": {"commit": "deadbeef", "branch": "task/serve-openai-parity"},
        "serve_binary": {"path": "/tmp/sparkflash", "sha256": "bbbb", "bytes": 1},
        "tokenizer": {"path": "/tmp/tokenizer.json", "sha256": "cccc"},
        "toolchain": {"rustc": "rustc test", "python": "3.12"},
        "host": {"platform": "test", "machine": "x86_64", "cpus": 8, "loadavg": [0.0, 0.0, 0.0]},
        "fixture": {
            "completion_prompt_token_ids": [1, 2, 3],
            "completion_max_tokens": 3,
            "chat_script_pieces": ["Hello"],
        },
        "accounting": {
            "runs": 1,
            "checks": {"a": True},
            "all_checks_passed": True,
            "check_count": 1,
            "serial_batched_payload_equal": True,
            "probe_wall_ms": [{"completions_serial": 10}],
        },
        "source_manifests": [{"path": "Cargo.lock", "sha256": "dddd"}],
        "raw_runs": [run],
        "non_claims": ["host path only"],
    }


def sse_text(frames: list[dict], done: bool = True) -> str:
    """An SSE body from the given JSON frames."""
    body = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
    if done:
        body += "data: [DONE]\n\n"
    return body


class SseTests(unittest.TestCase):
    def test_data_frames_are_parsed_and_done_is_not_a_frame(self) -> None:
        frames = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}},
        ]
        parsed = CENSUS.sse_frames(sse_text(frames))
        self.assertEqual(parsed, frames)

    def test_comment_frames_are_ignored(self) -> None:
        text = ": keep-alive\n\ndata: [DONE]\n\n"
        self.assertEqual(CENSUS.sse_frames(text), [])

    def test_usage_chunk_is_the_empty_choices_frame(self) -> None:
        frames = [
            {"choices": [{"index": 0}]},
            {"choices": [], "usage": {"prompt_tokens": 1}},
        ]
        self.assertEqual(CENSUS.usage_chunk(frames)["usage"]["prompt_tokens"], 1)

    def test_usage_chunk_requires_one(self) -> None:
        with self.assertRaises(SystemExit):
            CENSUS.usage_chunk([{"choices": [{"index": 0}]}])

    def test_finish_and_choice_indexes_keep_order(self) -> None:
        frames = [
            {"choices": [{"index": 0}]},
            {"choices": [{"index": 0, "finish_reason": "stop"}]},
            {"choices": [{"index": 1}]},
            {"choices": [{"index": 1, "finish_reason": "length"}]},
        ]
        self.assertEqual(CENSUS.choice_indexes(frames), [0, 1])
        self.assertEqual(CENSUS.finish_indexes(frames), [0, 1])


class PayloadTests(unittest.TestCase):
    def test_strip_volatile_drops_ids_times_and_nested_timing(self) -> None:
        payload = {
            "id": "chatcmpl-1",
            "created": 1,
            "choices": [{"index": 0, "message": {"content": "hi"}}],
            "x_deepseek_spark_stats": {"timing": {"ttft_ms": 1.5}, "kv_cache": {"total_prompt_tokens": 3}},
        }
        stripped = CENSUS.strip_volatile(payload)
        self.assertNotIn("id", stripped)
        self.assertNotIn("created", stripped)
        self.assertEqual(
            stripped,
            {
                "choices": [{"index": 0, "message": {"content": "hi"}}],
                "x_deepseek_spark_stats": {"kv_cache": {"total_prompt_tokens": 3}},
            },
        )

    def test_canonical_hash_is_key_order_independent(self) -> None:
        left = {"a": 1, "b": [1, 2]}
        right = {"b": [1, 2], "a": 1}
        self.assertEqual(CENSUS.canonical_hash(left), CENSUS.canonical_hash(right))


class SummarizeTests(unittest.TestCase):
    def test_checks_are_all_across_runs(self) -> None:
        runs = [
            {"checks": {"a": True, "b": False}, "probes": {"completions_serial": {"payload_hash": "h"}, "completions_batched": {"payload_hash": "h"}}, "probe_wall_ms": {}},
            {"checks": {"a": True}, "probes": {"completions_serial": {"payload_hash": "h"}, "completions_batched": {"payload_hash": "h"}}, "probe_wall_ms": {}},
        ]
        accounting = CENSUS.summarize(runs)
        self.assertFalse(accounting["checks"]["b"])
        self.assertTrue(accounting["checks"]["a"])
        self.assertFalse(accounting["all_checks_passed"])
        self.assertTrue(accounting["serial_batched_payload_equal"])

    def test_payload_drift_is_reported(self) -> None:
        runs = [
            {"checks": {}, "probes": {"completions_serial": {"payload_hash": "h1"}, "completions_batched": {"payload_hash": "h2"}}, "probe_wall_ms": {}},
        ]
        self.assertFalse(CENSUS.summarize(runs)["serial_batched_payload_equal"])


class ValidateCensusTests(unittest.TestCase):
    def test_a_valid_census_passes(self) -> None:
        CENSUS.validate_census(valid_census())

    def test_a_failed_check_is_refused(self) -> None:
        census = valid_census()
        census["accounting"]["all_checks_passed"] = False
        with self.assertRaises(SystemExit):
            CENSUS.validate_census(census)

    def test_a_wrong_choice_order_is_refused(self) -> None:
        census = valid_census()
        census["raw_runs"][0]["probes"]["completions_serial"]["buffered"]["choice_indexes"] = [1, 0]
        with self.assertRaises(SystemExit):
            CENSUS.validate_census(census)

    def test_a_missing_response_format_param_is_refused(self) -> None:
        census = valid_census()
        census["raw_runs"][0]["probes"]["response_format_refusals"]["json_object"]["param"] = None
        with self.assertRaises(SystemExit):
            CENSUS.validate_census(census)

    def test_a_missing_401_is_refused(self) -> None:
        census = valid_census()
        census["raw_runs"][0]["probes"]["bearer_auth"]["models_missing"]["status"] = 200
        with self.assertRaises(SystemExit):
            CENSUS.validate_census(census)

    def test_a_wrong_schema_is_refused(self) -> None:
        census = valid_census()
        census["schema"] = "something-else"
        with self.assertRaises(SystemExit):
            CENSUS.validate_census(census)


class MarkdownTests(unittest.TestCase):
    def test_markdown_names_the_scope_checks_and_non_claims(self) -> None:
        text = CENSUS.render_markdown(valid_census())
        self.assertIn("# OpenAI surface-gap wire census", text)
        self.assertIn("## Accounting", text)
        self.assertIn("## Checks", text)
        self.assertIn("## Non-claims", text)
        self.assertIn("host path only", text)
        self.assertIn("| 0 | [0, 1] |", text)


class RenderJsonTests(unittest.TestCase):
    def test_the_artefact_json_is_serialisable(self) -> None:
        census = valid_census()
        rendered = json.dumps(census, sort_keys=True)
        self.assertEqual(json.loads(rendered)["schema"], CENSUS.SCHEMA)
