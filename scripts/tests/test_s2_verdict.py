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

"""Unit tests for scripts/s2-verdict.py: the S2-d replicate and S2-e soak verdicts.

No network, no GPU, no row: the script's own synthetic rows (write_row) are the fixtures, so the
tests cannot drift from the shapes the verdict reads.

Run with:

    python3 -m unittest discover -s scripts/tests -v
"""
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "s2-verdict.py")

SPEC = importlib.util.spec_from_file_location("s2_verdict", SCRIPT)
verdict = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verdict)


class S2VerdictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="s2-verdict-")

    def quiet(self, argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return verdict.main(argv)

    def test_the_self_test_passes(self):
        self.assertEqual(self.quiet(["--self-test"]), 0)

    def test_agreeing_replicates_exit_zero_and_a_drifted_pair_exits_one(self):
        a = verdict.write_row(self.tmp, "a", [57.0] * 9)
        b = verdict.write_row(self.tmp, "b", [58.5] * 9)
        c = verdict.write_row(self.tmp, "c", [60.0] * 9)
        self.assertEqual(self.quiet(["replicates", "--run", a, "--run", b]), 0)
        self.assertEqual(self.quiet(["replicates", "--run", a, "--run", c]), 1)

    def test_a_missing_row_is_bad_input_never_a_pass(self):
        missing = os.path.join(self.tmp, "absent")
        self.assertEqual(self.quiet(["soak", "--run", missing]), 2)

    def test_a_row_without_the_s2_header_fails(self):
        row = verdict.write_row(self.tmp, "host", [57.0] * 1000, arm="host")
        result = verdict.soak(row, 1000, 1.10)
        self.assertFalse(result["pass"])
        self.assertTrue(any("device/nccl" in problem for problem in result["problems"]), result["problems"])

    def test_the_soak_reports_each_rank_and_the_ratio(self):
        row = verdict.write_row(self.tmp, "soak", [50.0] * 995 + [54.0] * 5)
        result = verdict.soak(row, 1000, 1.10)
        self.assertTrue(result["pass"], result["problems"])
        self.assertEqual([entry["rank"] for entry in result["ranks"]], [0, 1, 2, 3])
        self.assertEqual(result["ranks"][0]["p99_over_median"], 1.0)


if __name__ == "__main__":
    unittest.main()
