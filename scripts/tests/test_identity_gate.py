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

"""Tests for scripts/identity-gate.sh: the one-command byte-identity check.

CI cannot run a four-rank device window, so these tests pin the checker itself: --anchor validates
the frozen constants, --self-test proves a good synthetic row passes and corrupted tokens or a
chain-free row that records digests are rejected, and a real run directory (when present) passes.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "identity-gate.sh"


def run(*args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True)


class IdentityGateTest(unittest.TestCase):
    def test_anchor(self):
        result = run("--anchor")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("identity-gate anchor OK", result.stdout)

    def test_self_test(self):
        result = run("--self-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("self-test PASS", result.stdout)

    def test_usage_error_without_arguments(self):
        result = run()
        self.assertNotEqual(result.returncode, 0)

    def test_run_directory_passes_and_corruption_fails(self):
        root = tempfile.mkdtemp(prefix="identity-gate-test-")
        try:
            current = Path(root) / "current"
            for rank in range(4):
                base = current / f"rank{rank}"
                base.mkdir(parents=True)
                (base / f"chain-r{rank}.rc").write_text("0\n")
                (base / f"tp4-token-r{rank}.json").write_text(json.dumps({
                    "tokens": {"ids": [491, 92928, 201, 15, 4893, 16646, 15, 14265, 86]},
                    "logits_dump": {"bin_sha256": "aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2"},
                }))
                (base / f"tp4-forward-embed-r{rank}.json").write_text(json.dumps({
                    "evidence_mode": "off", "evidence_digest_calls": 0, "layer_chain": [],
                }))
            passed = run("--run", str(current))
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertIn("identity-gate PASS", passed.stdout)
            # a rank that exited non-zero must fail the whole run
            (current / "rank2" / "chain-r2.rc").write_text("1\n")
            failed = run("--run", str(current))
            self.assertEqual(failed.returncode, 1)
            self.assertIn("exit code", failed.stderr)
        finally:
            shutil.rmtree(root)

    def test_old_artifact_without_mode_fields_is_rejected(self):
        root = tempfile.mkdtemp(prefix="identity-gate-old-")
        try:
            run_dir = Path(root) / "old"
            for rank in range(4):
                base = run_dir / f"rank{rank}"
                base.mkdir(parents=True)
                (base / f"chain-r{rank}.rc").write_text("0\n")
                (base / f"tp4-token-r{rank}.json").write_text(json.dumps({
                    "tokens": {"ids": [491, 92928, 201, 15, 4893, 16646, 15, 14265, 86]},
                    "logits_dump": {"bin_sha256": "aed6054d77d6704c29812463106039fcdd8e39f34920e2c67aa00e9b52e037c2"},
                }))
                (base / f"tp4-forward-embed-r{rank}.json").write_text(json.dumps({"layer_chain": []}))
            failed = run("--run", str(run_dir))
            self.assertEqual(failed.returncode, 1)
            self.assertIn("lacks", failed.stderr)
        finally:
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
