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

"""Tests for the streaming tool-call (V4.1 DSML) wire census script."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "tool-call-streaming-census.py"
SPEC = importlib.util.spec_from_file_location("tool_call_streaming_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CENSUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CENSUS)

ARGUMENTS = '{"city": "Beijing", "days": 3}'


def probe_document(wall_ms: int = 1000, comment_positions: list[int] | None = None) -> dict:
    """One valid wire probe report with a single streamed tool call."""
    positions = [1, 2, 4] if comment_positions is None else comment_positions
    call = {
        "index": 0,
        "id": "chatcmpl-tool-00000001",
        "type": "function",
        "function": {"name": "get_weather", "arguments": ARGUMENTS},
    }
    return {
        "schema": CENSUS.PROBE_SCHEMA,
        "generated_at_unix": 1792553670,
        "server_binary": "/tmp/sparkflash",
        "script": {"pieces": 50, "delay_ms": 20, "keep_alive_ms": 5},
        "buffered": {
            "content": "I will check the weather.",
            "finish_reason": "tool_calls",
            "tool_calls": [call],
        },
        "streamed": {
            "content": "I will check the weather.",
            "finish_reason": "tool_calls",
            "tool_calls": [call],
        },
        "equality": {"content": True, "tool_calls_byte_for_byte": True},
        "delta_counts": {
            "content_deltas": 5,
            "call_start_deltas": 1,
            "argument_fragment_deltas": 3,
            "sse_data_frames": 11,
        },
        "keep_alive": {
            "raw_frames": 182,
            "comment_frames": 170,
            "comment_positions": positions,
            "first_frame_is_data": True,
            "last_frame_is_done": True,
            "comments_are_whole_frames": True,
        },
        "sse": {
            "role_frame_first": True,
            "ends_with_done": True,
            "dsml_leaked_into_content": False,
        },
        "wall": {"stream_wall_ms": wall_ms},
        "non_claims": ["host path only"],
    }


class ValidateTests(unittest.TestCase):
    def test_a_good_probe_document_validates(self) -> None:
        CENSUS.validate_probe(probe_document())

    def test_a_mismatched_payload_is_refused(self) -> None:
        document = probe_document()
        document["equality"]["tool_calls_byte_for_byte"] = False
        with self.assertRaises(SystemExit):
            CENSUS.validate_probe(document)

    def test_a_streamed_content_mismatch_is_refused(self) -> None:
        document = probe_document()
        document["equality"]["content"] = False
        with self.assertRaises(SystemExit):
            CENSUS.validate_probe(document)

    def test_a_missing_finish_reason_override_is_refused(self) -> None:
        document = probe_document()
        document["streamed"]["finish_reason"] = "stop"
        with self.assertRaises(SystemExit):
            CENSUS.validate_probe(document)

    def test_leaked_dsml_into_content_is_refused(self) -> None:
        document = probe_document()
        document["sse"]["dsml_leaked_into_content"] = True
        with self.assertRaises(SystemExit):
            CENSUS.validate_probe(document)

    def test_a_comment_after_the_terminator_is_refused(self) -> None:
        document = probe_document()
        document["keep_alive"]["last_frame_is_done"] = False
        with self.assertRaises(SystemExit):
            CENSUS.validate_probe(document)

    def test_a_comment_before_the_first_data_frame_is_refused(self) -> None:
        document = probe_document()
        document["keep_alive"]["first_frame_is_data"] = False
        with self.assertRaises(SystemExit):
            CENSUS.validate_probe(document)

    def test_no_argument_fragments_is_refused(self) -> None:
        document = probe_document()
        document["delta_counts"]["argument_fragment_deltas"] = 0
        with self.assertRaises(SystemExit):
            CENSUS.validate_probe(document)

    def test_a_missing_non_claims_list_is_refused(self) -> None:
        document = probe_document()
        document["non_claims"] = []
        with self.assertRaises(SystemExit):
            CENSUS.validate_probe(document)


class SummarizeTests(unittest.TestCase):
    def test_summarize_folds_the_probe_runs(self) -> None:
        runs = [
            probe_document(1000),
            probe_document(1100, [2, 4, 8]),
            probe_document(900),
        ]
        summary = CENSUS.summarize(runs, [4000, 4100, 3900], ["a" * 64] * 3)
        self.assertEqual(summary["runs"], 3)
        self.assertTrue(summary["payload_equal"])
        self.assertTrue(summary["content_equal"])
        self.assertEqual(summary["script_pieces"], [50])
        self.assertEqual(summary["tool_calls"], 1)
        self.assertEqual(summary["arguments"], [ARGUMENTS] * 3)
        self.assertEqual(summary["finish_reasons"], ["tool_calls"])
        self.assertEqual(summary["content_deltas"], [5, 5, 5])
        self.assertEqual(summary["call_start_deltas"], [1, 1, 1])
        self.assertEqual(summary["argument_fragment_deltas"], [3, 3, 3])
        self.assertEqual(summary["sse_data_frames"], [11, 11, 11])
        self.assertEqual(summary["comment_frames"], [170, 170, 170])
        self.assertTrue(summary["comment_positions_monotonic"])
        self.assertEqual(summary["stream_wall_ms"], [1000, 1100, 900])
        self.assertEqual(summary["stream_wall_ms_median"], 1000.0)
        self.assertEqual(summary["stream_wall_ms_min"], 900)
        self.assertEqual(summary["stream_wall_ms_max"], 1100)
        self.assertEqual(summary["probe_process_ms"], [4000, 4100, 3900])
        self.assertEqual(len(summary["fixture_hash"]), 64)
        self.assertEqual(summary["server_binary_sha256"], "a" * 64)

    def test_summarize_refuses_a_fixture_mix(self) -> None:
        other = probe_document()
        other["buffered"]["content"] = "a different fixture"
        with self.assertRaises(SystemExit):
            CENSUS.summarize(
                [probe_document(), other], [4000, 4000], ["a" * 64, "a" * 64]
            )

    def test_summarize_refuses_a_serve_binary_mix(self) -> None:
        with self.assertRaises(SystemExit):
            CENSUS.summarize(
                [probe_document(), probe_document()],
                [4000, 4000],
                ["a" * 64, "b" * 64],
            )

    def test_summarize_marks_unsorted_comment_positions(self) -> None:
        summary = CENSUS.summarize(
            [probe_document(comment_positions=[4, 1])], [4000], ["a" * 64]
        )
        self.assertFalse(summary["comment_positions_monotonic"])


class RenderTests(unittest.TestCase):
    def test_markdown_names_scope_accounting_and_non_claims(self) -> None:
        summary = CENSUS.summarize([probe_document()], [4000], ["a" * 64])
        census = {
            "schema": CENSUS.SCHEMA,
            "generated_at": "2026-10-21T00:00:00Z",
            "scope": CENSUS.SCOPE,
            "git": {"commit": "deadbeef", "branch": "task/serve-openai-parity"},
            "test_binary": {"sha256": "0" * 64},
            "serve_binary": {"sha256": "a" * 64},
            "accounting": summary,
        }
        markdown = CENSUS.render_markdown(census)
        self.assertIn(CENSUS.SCHEMA, markdown)
        self.assertIn(CENSUS.SCOPE, markdown)
        self.assertIn("deadbeef", markdown)
        self.assertIn("tool_calls byte-for-byte equal: True", markdown)
        self.assertIn("argument-fragment deltas", markdown)
        self.assertIn("Non-claims", markdown)
        self.assertIn("no vLLM comparison", markdown)
        self.assertIn("residual gap", markdown)


class HashTests(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"sparkflash\n")
            self.assertEqual(
                CENSUS.sha256_file(path), hashlib.sha256(b"sparkflash\n").hexdigest()
            )

    def test_canonical_hash_is_stable_across_key_order(self) -> None:
        first = CENSUS.canonical_hash({"b": 2, "a": 1})
        second = CENSUS.canonical_hash({"a": 1, "b": 2})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
