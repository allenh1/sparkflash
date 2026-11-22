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

"""Unit tests for scripts/open-manifest-ab.py. No GPU, server or checkpoint."""

import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "open_manifest_ab", os.path.join(SCRIPTS, "open-manifest-ab.py")
)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class OpenManifestAbTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_manifest_lines_compare_as_multisets(self):
        old = write(self.path("old.jsonl"), '{"kind":"model","name":"dim","value":5120}\n')
        new = write(self.path("new.jsonl"), '{"kind":"model","name":"dim","value":5120}\n')
        self.assertIsNone(ab.line_difference(old, new))

    def test_a_changed_manifest_value_is_a_difference(self):
        old = write(self.path("old.jsonl"), '{"kind":"model","name":"dim","value":5120}\n')
        new = write(self.path("new.jsonl"), '{"kind":"model","name":"dim","value":4096}\n')
        self.assertIn("differs", ab.line_difference(old, new))

    def test_launch_trace_order_matters(self):
        old = write(self.path("old.trace"), "a\nb\n")
        new = write(self.path("new.trace"), "b\na\n")
        self.assertIsNotNone(ab.ordered_difference(old, new))

    def test_tick_trace_ignores_only_the_clock(self):
        old = write(self.path("old.tick"), "t=1 clock=1 hold 0\n")
        new = write(self.path("new.tick"), "t=1 clock=999 hold 0\n")
        self.assertIsNone(ab.tick_difference(old, new))

    def test_tick_trace_reports_a_changed_decision(self):
        old = write(self.path("old.tick"), "t=1 clock=1 hold 0\n")
        new = write(self.path("new.tick"), "t=1 clock=1 hold 1\n")
        self.assertIn("differs", ab.tick_difference(old, new))

    def test_missing_rank_files_are_a_failure(self):
        old = self.path("old")
        new = self.path("new")
        os.makedirs(old)
        os.makedirs(new)
        self.assertFalse(ab.compare(old, new, 0.05))


if __name__ == "__main__":
    unittest.main()
