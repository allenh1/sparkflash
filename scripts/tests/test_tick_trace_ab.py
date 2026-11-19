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

"""Unit tests for scripts/tick-trace-ab.py: the trace parser and the first-difference report.

No GPU, no server, no checkpoint. Run with:
    python3 -m unittest discover -s scripts/tests -v
"""

import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("tick_trace_ab", os.path.join(SCRIPTS, "tick-trace-ab.py"))
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class TickTraceAbTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_identical_decisions_pass_even_with_a_different_clock(self):
        old = write(self.path("old.trace"), "t=1 clock=450 hold 0\nt=1 clock=450 prefill seq=3 from=0 rows=64 to=64 target=80 path=exact\n")
        new = write(self.path("new.trace"), "t=1 clock=999 hold 0\nt=1 clock=999 prefill seq=3 from=0 rows=64 to=64 target=80 path=exact\n")
        self.assertIsNone(ab.first_difference(old, new))

    def test_a_changed_hold_reports_its_tick(self):
        old = write(self.path("old.trace"), "t=1 clock=1 hold 0\n")
        new = write(self.path("new.trace"), "t=1 clock=1 hold 1\n")
        tick, old_line, new_line, why = ab.first_difference(old, new)
        self.assertEqual(tick, 1)
        self.assertIn("hold 0", old_line)
        self.assertIn("hold 1", new_line)
        self.assertIn("differs", why)

    def test_the_same_decision_at_a_different_tick_is_a_difference(self):
        old = write(self.path("old.trace"), "t=1 clock=1 hold 0\nt=2 clock=2 hold 1\n")
        new = write(self.path("new.trace"), "t=1 clock=1 hold 0\nt=3 clock=2 hold 1\n")
        tick, _, _, why = ab.first_difference(old, new)
        self.assertEqual(tick, "2/3")
        self.assertIn("different tick", why)

    def test_a_truncated_trace_reports_the_first_extra_decision(self):
        old = write(self.path("old.trace"), "t=1 clock=1 hold 0\n")
        new = write(self.path("new.trace"), "t=1 clock=1 hold 0\nt=1 clock=1 hold 1\n")
        tick, old_line, new_line, why = ab.first_difference(old, new)
        self.assertEqual(tick, 1)
        self.assertEqual(old_line, "-(ends)")
        self.assertIn("old trace ends", why)
        self.assertIn("t=1 clock=1 hold 1", new_line)

    def test_a_malformed_line_is_refused(self):
        bad = write(self.path("bad.trace"), "hold 0\n")
        good = write(self.path("good.trace"), "t=1 clock=1 hold 0\n")
        with self.assertRaises(SystemExit):
            ab.parse_trace(bad)
        self.assertIsNone(ab.first_difference(good, good))


if __name__ == "__main__":
    unittest.main()
