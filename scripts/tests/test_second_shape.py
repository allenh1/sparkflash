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

"""Unit tests for scripts/second-shape.py: the plan sidecars, reuse decisions and summary.

The file name has a hyphen, so it is imported by path. No GPU, no server, no checkpoint.
Run with:

    python3 -m unittest discover -s scripts/tests -p 'test_second_shape.py'
"""

import contextlib
import importlib.util
import io
import os
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
SCRIPT = os.path.join(SCRIPTS, "second-shape.py")

SPEC = importlib.util.spec_from_file_location("second_shape", SCRIPT)
second_shape = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(second_shape)

CONFIG = b'{"quantization_config": {"quant_method": "bf16"}}\n'
PROMPTS = [[1, 2, 3], [4, 5]]


class PlanHashTest(unittest.TestCase):
    """The reference plan covers the checkpoint, world, prompt list and token budget."""

    def test_world_changes_the_reference_plan(self):
        one = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        two = second_shape.reference_plan(CONFIG, 2, PROMPTS, 8)
        self.assertNotEqual(one, two)

    def test_prompt_list_changes_the_reference_plan(self):
        narrow = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        wide = second_shape.reference_plan(CONFIG, 1, [[1, 2, 3], [4, 6]], 8)
        self.assertNotEqual(narrow, wide)

    def test_max_tokens_changes_the_reference_plan(self):
        short = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        long = second_shape.reference_plan(CONFIG, 1, PROMPTS, 9)
        self.assertNotEqual(short, long)

    def test_config_bytes_change_the_reference_plan(self):
        left = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        right = second_shape.reference_plan(CONFIG + b" ", 1, PROMPTS, 8)
        self.assertNotEqual(left, right)

    def test_the_same_inputs_give_the_same_reference_plan(self):
        self.assertEqual(
            second_shape.reference_plan(CONFIG, 1, PROMPTS, 8),
            second_shape.reference_plan(CONFIG, 1, PROMPTS, 8),
        )

    def test_checkpoint_plan_changes_with_the_config(self):
        self.assertNotEqual(
            second_shape.checkpoint_plan(CONFIG), second_shape.checkpoint_plan(CONFIG + b" ")
        )

class ReuseDecisionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def reference_files(self):
        reference = self.root / "host-reference-w1.json"
        sidecar = self.root / "host-reference-w1.json.plan"
        reference.write_text("[]\n")
        return reference, sidecar

    def test_reference_is_reused_when_the_sidecar_matches(self):
        reference, sidecar = self.reference_files()
        plan = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        second_shape.write_plan(sidecar, plan)
        self.assertTrue(second_shape.reference_reusable(reference, sidecar, plan))

    def test_reference_is_not_reused_without_the_sidecar(self):
        reference, sidecar = self.reference_files()
        plan = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        self.assertFalse(second_shape.reference_reusable(reference, sidecar, plan))

    def test_reference_is_not_reused_when_the_file_is_missing(self):
        _reference, sidecar = self.reference_files()
        plan = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        second_shape.write_plan(sidecar, plan)
        missing = self.root / "absent.json"
        self.assertFalse(second_shape.reference_reusable(missing, sidecar, plan))

    def test_reference_is_not_reused_for_a_different_plan(self):
        reference, sidecar = self.reference_files()
        written = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        asked = second_shape.reference_plan(CONFIG, 2, PROMPTS, 8)
        second_shape.write_plan(sidecar, written)
        self.assertFalse(second_shape.reference_reusable(reference, sidecar, asked))

    def test_reference_is_not_reused_for_an_unreadable_sidecar(self):
        reference, sidecar = self.reference_files()
        sidecar.write_text("not json\n")
        plan = second_shape.reference_plan(CONFIG, 1, PROMPTS, 8)
        self.assertFalse(second_shape.reference_reusable(reference, sidecar, plan))

    def checkpoint_files(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_bytes(CONFIG)
        stamp = self.root / "checkpoint.plan"
        return checkpoint, stamp

    def test_checkpoint_is_reused_when_the_stamp_and_config_match(self):
        checkpoint, stamp = self.checkpoint_files()
        second_shape.write_plan(stamp, second_shape.checkpoint_plan(CONFIG))
        self.assertTrue(second_shape.checkpoint_reusable(checkpoint, stamp, CONFIG))

    def test_checkpoint_is_not_reused_without_the_stamp(self):
        checkpoint, stamp = self.checkpoint_files()
        self.assertFalse(second_shape.checkpoint_reusable(checkpoint, stamp, CONFIG))

    def test_checkpoint_is_not_reused_when_the_config_bytes_differ(self):
        checkpoint, stamp = self.checkpoint_files()
        second_shape.write_plan(stamp, second_shape.checkpoint_plan(CONFIG))
        self.assertFalse(
            second_shape.checkpoint_reusable(checkpoint, stamp, CONFIG + b" ")
        )

    def test_checkpoint_is_not_reused_when_the_directory_is_missing(self):
        _checkpoint, stamp = self.checkpoint_files()
        second_shape.write_plan(stamp, second_shape.checkpoint_plan(CONFIG))
        missing = self.root / "absent-checkpoint"
        self.assertFalse(second_shape.checkpoint_reusable(missing, stamp, CONFIG))

    def test_plan_sidecar_round_trips(self):
        sidecar = self.root / "x.plan"
        second_shape.write_plan(sidecar, "abc")
        self.assertEqual(second_shape.read_plan(sidecar), "abc")


class SummaryTest(unittest.TestCase):
    def summarised(self, outcomes):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = second_shape.summarise(outcomes)
        return code, stream.getvalue()

    def test_every_requested_world_is_reported(self):
        outcomes = [
            (1, True, "the host group agreed with host-reference-w1.json"),
            (2, True, "the host group agreed with host-reference-w2.json"),
            (4, True, "the host group agreed with host-reference-w4.json"),
        ]
        lines = second_shape.world_summary(outcomes)
        self.assertEqual(len(lines), 3)
        self.assertIn("world 1 PASS", lines[0])
        self.assertIn("world 2 PASS", lines[1])
        self.assertIn("world 4 PASS", lines[2])
        self.assertTrue(all(line.startswith("second-shape:") for line in lines))

    def test_all_passing_worlds_exit_zero(self):
        outcomes = [(1, True, "agreed"), (2, True, "agreed")]
        code, output = self.summarised(outcomes)
        self.assertEqual(code, 0)
        self.assertIn("second-shape: summary", output)
        self.assertIn("world 1 PASS: agreed", output)
        self.assertIn("world 2 PASS: agreed", output)

    def test_a_failed_world_still_reports_the_next_world_and_exits_one(self):
        # A failure is recorded, not raised, so the next requested world is still reported.
        outcomes = [
            (1, False, "the one-rank host arm does not equal the global reference"),
            (2, True, "the host group agreed with host-reference-w2.json"),
            (4, True, "the host group agreed with host-reference-w4.json"),
        ]
        code, output = self.summarised(outcomes)
        self.assertEqual(code, 1)
        self.assertIn("world 1 FAIL: the one-rank host arm", output)
        self.assertIn("world 2 PASS", output)
        self.assertIn("world 4 PASS", output)
        self.assertEqual(len(output.strip().splitlines()), 4)

    def test_a_wrapped_reason_stays_on_one_line(self):
        line = second_shape.world_summary([(2, False, "first part\nsecond part")])[0]
        self.assertEqual(line, "second-shape: world 2 FAIL: first part second part")


class DeviceEnvTest(unittest.TestCase):
    """The device group's environment carries the second shape's own declarations."""

    def test_a_device_world_declares_the_head_screen_and_unpinned_cubins(self):
        args = types.SimpleNamespace(binary="sparkflash", port=8126)
        env = second_shape.env_for_world(
            args,
            Path("/out"),
            Path("/out/checkpoint"),
            "/out/tokenizer.json",
            Path("/out/cubins-w2"),
            2,
            ["worker-0", "worker-1"],
            ["peer-0", "peer-1"],
        )
        # The certified screen covers the released width, not this shape's.
        self.assertEqual(env["SPARKFLASH_HEAD_SCREEN"], "0")
        self.assertEqual(env["SPARKFLASH_UNPINNED_SECOND_SHAPE"], "1")
        self.assertEqual(env["SPARKFLASH_CUBINS"], "/out/cubins-w2")
        self.assertEqual(env["SPARK_NODES"], "worker-0 worker-1")
        self.assertEqual(env["SPARK_WORLD"], "2")

    def test_a_device_run_refuses_without_an_nccl_library(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPARKFLASH_NCCL_LIB", None)
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    second_shape.require_nccl_library()
        self.assertIn("SPARKFLASH_NCCL_LIB", stderr.getvalue())

    def test_a_device_run_takes_the_nccl_library_from_the_operator(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"SPARKFLASH_NCCL_LIB": directory}):
                self.assertEqual(second_shape.require_nccl_library(), Path(directory))
                env = second_shape.env_for_world(
                    types.SimpleNamespace(binary="sparkflash", port=8126),
                    Path("/out"),
                    Path("/out/checkpoint"),
                    "/out/tokenizer.json",
                    Path("/out/cubins-w1"),
                    1,
                    ["worker-0"],
                    ["peer-0"],
                )
                self.assertEqual(env["SPARKFLASH_NCCL_LIB"], directory)

    def test_a_device_run_refuses_a_path_that_is_not_a_directory(self):
        with mock.patch.dict(os.environ, {"SPARKFLASH_NCCL_LIB": "/nonexistent/nccl"}):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    second_shape.require_nccl_library()
        self.assertIn("not a directory", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
