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

"""Unit tests for scripts/twin_evidence.py (canonical --twin evidence from the host reference fixture)."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import twin_evidence


def write_json(path, doc):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)


class TwinEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.fixture = os.path.join(self.root, "fixture.json")
        write_json(
            self.fixture,
            {
                "prompt_tokens": [11, 22, 33],
                "steps": [
                    {"kind": "prefill", "position": 2, "next_token": 44},
                    {"kind": "decode", "position": 3, "next_token": 55},
                ],
            },
        )

    def run_tool(self, *argv):
        return twin_evidence.main(list(argv))

    def test_writes_canonical_twin_document(self):
        out = os.path.join(self.root, "twin")
        self.assertEqual(self.run_tool("--fixture", self.fixture, "--out-dir", out, "--world", "4"), 0)
        with open(os.path.join(out, "twin-r0.json"), "r", encoding="utf-8") as handle:
            doc = json.load(handle)
        self.assertEqual(doc["schema"], "sparkflash-device-token-evidence/1")
        self.assertEqual(doc["kind"], "host-twin")
        self.assertEqual(doc["prompt_tokens"], [11, 22, 33])
        self.assertEqual(doc["positions"], [2, 3])
        self.assertEqual(doc["token_ids"], [44, 55])
        self.assertEqual(doc["start_position"], 2)
        self.assertTrue(doc["token_emitted"])
        self.assertEqual(doc["notes"][0][:20], "host reference-engine"[:20])
        self.assertIn("NOT a second independent four-rank run", doc["notes"][1])
        self.assertTrue(os.path.getsize(os.path.join(out, "twin-r0.json")) > 0)

    def test_evidence_is_consumable_by_the_acceptance_extractor(self):
        import accept_device_token as accept

        out = os.path.join(self.root, "twin")
        self.assertEqual(self.run_tool("--fixture", self.fixture, "--out-dir", out), 0)
        evidence = accept.extract_evidence([os.path.join(out, "twin-r0.json")], expect_vocab=None, side="twin")
        stream = accept.token_stream(evidence)
        self.assertEqual(stream["tokens"], [44, 55])
        self.assertEqual(stream["positions"], [2, 3])
        self.assertTrue(stream["consistent"])

    def test_refuses_overwrite_without_force(self):
        out = os.path.join(self.root, "twin")
        self.assertEqual(self.run_tool("--fixture", self.fixture, "--out-dir", out), 0)
        self.assertEqual(self.run_tool("--fixture", self.fixture, "--out-dir", out), 2)
        self.assertEqual(self.run_tool("--fixture", self.fixture, "--out-dir", out, "--force"), 0)

    def test_rejects_fixture_without_next_token(self):
        bad = os.path.join(self.root, "bad.json")
        write_json(bad, {"prompt_tokens": [1, 2], "steps": [{"position": 1}]})
        out = os.path.join(self.root, "bad-out")
        self.assertEqual(self.run_tool("--fixture", bad, "--out-dir", out), 2)
        self.assertFalse(os.path.exists(os.path.join(out, "twin-r0.json")))


if __name__ == "__main__":
    unittest.main()
