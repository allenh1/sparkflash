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

"""Tests for the TP=4 prompt-logprobs rank contract census script."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "tp4-prompt-logprobs-census.py"
SPEC = importlib.util.spec_from_file_location("tp4_prompt_logprobs_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CENSUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CENSUS)


def parity_document(fixture_hash: int, wall_ms: int = 700) -> dict:
    """One parity probe report with a single scored round."""
    return {
        "schema": "sparkflash-tp4-prompt-logprobs-probe/1",
        "fixture_tensors": 2193,
        "fixture_bytes": 21878732,
        "fixture_hash": fixture_hash,
        "world": 4,
        "rounds": [
            {
                "round": 0,
                "wall_ms": wall_ms,
                "scored_positions": 5,
                "alternatives_per_position": 3,
                "ranks_agree_bitwise": True,
                "serial_order_equal": True,
                "max_abs_logprob_delta": 0.0,
            }
        ],
    }


def refusal_document(fixture_hash: int, refusal: str = "TP=4 prompt logprobs refused: budget") -> dict:
    """One refusal probe report."""
    return {
        "fixture_tensors": 2193,
        "fixture_bytes": 21878732,
        "fixture_hash": fixture_hash,
        "world": 4,
        "refusal": refusal,
        "ranks_agree_on_message": True,
        "recovered_after_refusal": True,
    }


class SummarizeTests(unittest.TestCase):
    def test_summarize_folds_rounds_and_refusals(self) -> None:
        parity = [parity_document(7, 700), parity_document(7, 720)]
        refusal = [refusal_document(7), refusal_document(7)]
        summary = CENSUS.summarize(parity, refusal, [4000, 3900], [2000, 2100])
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["rounds"], 2)
        self.assertEqual(summary["scored_positions_total"], 10)
        self.assertEqual(summary["alternatives_per_position"], [3])
        self.assertTrue(summary["ranks_agree_bitwise"])
        self.assertTrue(summary["serial_order_equal"])
        self.assertEqual(summary["max_abs_logprob_delta"], 0.0)
        self.assertEqual(summary["round_wall_ms"], [700, 720])
        self.assertEqual(summary["round_wall_ms_median"], 710.0)
        self.assertEqual(summary["round_wall_ms_min"], 700)
        self.assertEqual(summary["round_wall_ms_max"], 720)
        self.assertEqual(summary["refusal_requests"], 2)
        self.assertEqual(summary["refusal_rank_count"], 4)
        self.assertTrue(summary["refusal_agreed"])
        self.assertTrue(summary["recovered_after_refusal"])
        self.assertEqual(summary["fixture_tensors"], 2193)
        self.assertEqual(summary["fixture_hash"], 7)

    def test_summarize_refuses_a_fixture_mix(self) -> None:
        with self.assertRaises(SystemExit):
            CENSUS.summarize(
                [parity_document(7)], [refusal_document(8)], [4000], [2000]
            )

    def test_summarize_marks_a_parity_miss(self) -> None:
        document = parity_document(7)
        document["rounds"][0]["serial_order_equal"] = False
        document["rounds"][0]["ranks_agree_bitwise"] = False
        document["rounds"][0]["max_abs_logprob_delta"] = 1.0e-6
        summary = CENSUS.summarize([document], [refusal_document(7)], [4000], [2000])
        self.assertFalse(summary["serial_order_equal"])
        self.assertFalse(summary["ranks_agree_bitwise"])
        self.assertEqual(summary["max_abs_logprob_delta"], 1.0e-6)


class RenderTests(unittest.TestCase):
    def test_markdown_names_scope_accounting_and_non_claims(self) -> None:
        summary = CENSUS.summarize(
            [parity_document(7)], [refusal_document(7)], [4000], [2000]
        )
        census = {
            "schema": CENSUS.SCHEMA,
            "generated_at": "2026-10-21T00:00:00Z",
            "scope": CENSUS.SCOPE,
            "git": {"commit": "deadbeef", "branch": "task/serve-openai-parity"},
            "probe_binary": {"sha256": "0" * 64},
            "accounting": summary,
        }
        markdown = CENSUS.render_markdown(census)
        self.assertIn(CENSUS.SCHEMA, markdown)
        self.assertIn(CENSUS.SCOPE, markdown)
        self.assertIn("deadbeef", markdown)
        self.assertIn("scored positions: 5", markdown)
        self.assertIn("Non-claims", markdown)
        self.assertIn("no vLLM comparison run", markdown)


class HashTests(unittest.TestCase):
    def test_sha256_file_matches_hashlib(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"sparkflash\n")
            self.assertEqual(
                CENSUS.sha256_file(path), hashlib.sha256(b"sparkflash\n").hexdigest()
            )

    def test_sha256_file_is_hex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"sparkflash")
            digest = CENSUS.sha256_file(path)
            self.assertEqual(len(digest), 64)
            int(digest, 16)


if __name__ == "__main__":
    unittest.main()
