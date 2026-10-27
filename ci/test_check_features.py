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

"""Tests for ci/check_features.py, run by the test-features job before the real sweep.

The script's first version had a bug that mattered: it skipped a feature whose value list was
exactly ["dep:<name>"], which is the shape cargo generates for an optional dependency but also
the shape a manifest uses when it declares that feature by hand. sparkflash-engram declares
io-uring = ["dep:io-uring"], so the ring the script exists to cover was the one thing it skipped.
These tests pin the distinction, and the dead-feature rule that catches a feature gating nothing.
"""

import inspect
import sys
import types
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parent / "check_features.py"
sys.path.insert(0, str(MODULE.parent))

# Round-20 quality, made robust in round 21: a stale __pycache__/check_features.*.pyc whose recorded
# mtime and size still match the source is read INSTEAD of the source, so a mutation battery that edits
# the module, imports it and restores the bytes leaves a cached copy of the MUTATED code behind, and the
# tests then check code that is not in the file. That happened on this host (the command-list test failed
# against the restored source because the cache held the mutation). Two things close it, and the second
# one is why this does not merely unlink: the cache is deleted when it CAN be deleted, and the module
# under test is then executed from its source TEXT, so a cache that cannot be removed - a read-only
# directory, a permission the test run does not have - cannot make the tests read stale code either.
for stale in (MODULE.parent / "__pycache__").glob("check_features.*.pyc"):
    try:
        stale.unlink()
    except OSError:
        pass

cf = types.ModuleType("check_features")
cf.__file__ = str(MODULE)
exec(compile(MODULE.read_text(), str(MODULE), "exec"), cf.__dict__)
sys.modules["check_features"] = cf

MANIFEST = """
[package]
name = "sparkflash-engram"

[features]
io-uring = ["dep:io-uring"]
trace-points = []

[dependencies]
io-uring = { version = "0.7", optional = true }
"""

MULTILINE = """
[features]
io-uring = [
  "dep:io-uring",
]
"""


class DeclaredFeatures(unittest.TestCase):
    def test_reads_keys_from_the_features_table(self):
        self.assertEqual(cf.parse_declared_features(MANIFEST), ["io-uring", "trace-points"])

    def test_reads_a_key_whose_array_spans_lines(self):
        self.assertEqual(cf.parse_declared_features(MULTILINE), ["io-uring"])

    def test_stops_at_the_next_table(self):
        text = "[features]\na = []\n\n[dependencies]\nb = []\n"
        self.assertEqual(cf.parse_declared_features(text), ["a"])

    def test_a_manifest_without_a_features_table_declares_nothing(self):
        self.assertEqual(cf.parse_declared_features("[package]\nname = \"x\"\n"), [])


class DependencyToggle(unittest.TestCase):
    def test_implicit_toggle_is_skipped(self):
        self.assertTrue(cf.is_implicit_dependency_toggle("cudarc", ["dep:cudarc"], []))

    def test_declared_feature_with_the_same_shape_is_not_skipped(self):
        # The regression: this is io-uring in the real manifest.
        self.assertFalse(cf.is_implicit_dependency_toggle("io-uring", ["dep:io-uring"], ["io-uring"]))

    def test_a_feature_with_code_of_its_own_is_never_an_implicit_toggle(self):
        self.assertFalse(cf.is_implicit_dependency_toggle("x", ["other"], []))


class DeadFeatures(unittest.TestCase):
    def test_a_feature_no_cfg_names_gates_nothing(self):
        self.assertTrue(cf.gates_nothing("trace-nvtx", [], ["fn f() {}"]))

    def test_a_feature_a_cfg_names_gates_something(self):
        self.assertFalse(cf.gates_nothing("trace-nvtx", [], ['#[cfg(feature = "trace-nvtx")]']))

    def test_a_feature_forwarding_to_another_feature_is_not_dead(self):
        self.assertFalse(cf.gates_nothing("full", ["a", "b"], ["fn f() {}"]))

    def test_a_feature_that_enables_an_optional_dependency_is_not_dead(self):
        # sparkflash-kernels/cuda is this shape: no cfg names it, but it selects cudarc.
        self.assertFalse(cf.gates_nothing("cuda", ["dep:cudarc"], ["fn f() {}"]))

    def test_an_empty_feature_another_feature_names_is_not_dead(self):
        self.assertFalse(cf.gates_nothing("marker", [], ["fn f() {}"], {"marker"}))

    def test_spacing_and_quote_style_do_not_hide_a_cfg(self):
        for text in ["feature=\"io-uring\"", "feature = 'io-uring'", "#[cfg_attr(feature = \"io-uring\", x)]"]:
            self.assertFalse(cf.gates_nothing("io-uring", ["dep:io-uring"], [text]), text)


class Classify(unittest.TestCase):
    def setUp(self):
        self.packages = [
            {
                "name": "sparkflash-engram",
                "features": {"default": [], "io-uring": ["dep:io-uring"]},
            },
            {
                "name": "sparkflash-trace",
                "features": {"default": ["trace-points"], "trace-points": [], "trace-nvtx": []},
            },
            {
                "name": "sparkflash-kernels",
                "features": {"default": [], "cudarc": ["dep:cudarc"], "cuda": ["dep:cudarc"]},
            },
        ]
        self.manifests = {
            "sparkflash-engram": MANIFEST,
            "sparkflash-trace": "[features]\ntrace-points = []\ntrace-nvtx = []\n",
            "sparkflash-kernels": "[features]\ncuda = [\"dep:cudarc\"]\n",
        }
        self.sources = {
            "sparkflash-engram": ['if cfg!(feature = "io-uring") {}'],
            "sparkflash-trace": ["fn f() {}"],
            "sparkflash-kernels": ['#[cfg(feature = "cuda")]'],
        }

    def test_the_declared_io_uring_feature_is_tested(self):
        found, dead = cf.classify(self.packages, self.manifests, self.sources)
        self.assertIn(("sparkflash-engram", "io-uring"), found)

    def test_an_implicit_dependency_toggle_is_not_tested(self):
        found, _ = cf.classify(self.packages, self.manifests, self.sources)
        self.assertNotIn(("sparkflash-kernels", "cudarc"), found)

    def test_a_declared_feature_that_gates_nothing_is_reported(self):
        _, dead = cf.classify(self.packages, self.manifests, self.sources)
        # classify reports the value list too, so the message can quote it.
        self.assertIn(("sparkflash-trace", "trace-nvtx"), [(p, f) for p, f, _ in dead])

    def test_a_default_feature_is_left_to_the_workspace_run(self):
        found, dead = cf.classify(self.packages, self.manifests, self.sources)
        names = [(p, f) for p, f, *_ in dead]
        self.assertNotIn(("sparkflash-trace", "trace-points"), found)
        self.assertNotIn(("sparkflash-trace", "trace-points"), names)


class ClippyGate(unittest.TestCase):
    """The sweep's clippy step has to be CI's own workspace invocation (round-18 A7).

    The failure this pins: the sweep ran `cargo clippy -p <package> --all-targets`, which does not
    raise a lint from another crate's documentation, while CI ran the workspace command. A
    clippy::doc_lazy_continuation in one crate's test code therefore passed this sweep and failed CI.
    """

    def test_the_workspace_gate_is_the_ci_command(self):
        self.assertEqual(
            cf.CLIPPY_GATE,
            ["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
        )

    def test_the_feature_scoped_gate_names_the_package_and_the_feature(self):
        self.assertEqual(
            cf.feature_clippy("sparkflash-engram", "io-uring"),
            [
                "cargo",
                "clippy",
                "-p",
                "sparkflash-engram",
                "--all-targets",
                "--features",
                "io-uring",
                "--",
                "-D",
                "warnings",
            ],
        )
        for package, feature in [
            ("sparkflash-comm", "rdma"),
            ("sparkflash-kernels", "cuda"),
        ]:
            command = cf.feature_clippy(package, feature)
            self.assertIn(package, command)
            self.assertIn(feature, command)
            self.assertIn("--features", command)

    def test_main_runs_the_command_list(self):
        source = inspect.getsource(cf.main)
        self.assertIn("commands_for(found, toolkit_present)", source)
        # Every command is built by the helpers, not spelled out inside main: the workspace gate is
        # appended by commands_for itself, which is where the pin below reads it.
        self.assertIn("CLIPPY_GATE", inspect.getsource(cf.commands_for))
        self.assertNotIn('"clippy", "-p"', source)
        self.assertNotIn('"cargo", "test"', source)

    def test_the_clippy_call_stands_before_the_hardware_skip(self):
        # Round-19 A4: the skip is for the TESTS, and the lint is not on the far side of it. In the
        # old loop the `continue` came first, so cuda's lint never ran - and the workspace gate
        # resolves default features only, so no gate named that feature at all.
        source = inspect.getsource(cf.commands_for)
        self.assertLess(
            source.index("feature_clippy(package, feature)"),
            source.index("continue"),
            "the feature-scoped lint must stand before the hardware-only skip",
        )

    def test_the_command_list_lints_every_feature_and_skips_only_hardware_tests(self):
        """Round-19 A4/A9: this is the test that catches a gate that does not run.

        It runs the sweep's own command list for a set of features that includes a hardware-only one
        and asserts the exact set, so a command that stops being produced fails here rather than in the
        pipeline. sparkflash-kernels/cuda is the case that was missing: its CODE is lintable in this
        container, only its tests are not.
        """
        found = [
            ("sparkflash-comm", "rdma"),
            ("sparkflash-engram", "io-uring"),
            ("sparkflash-kernels", "cuda"),
        ]
        commands = cf.commands_for(found, toolkit_present=True)
        expected = [cf.feature_clippy(package, feature) for package, feature in found] + [
            ["cargo", "test", "-p", "sparkflash-comm", "--features", "rdma"],
            ["cargo", "test", "-p", "sparkflash-engram", "--features", "io-uring"],
            cf.CLIPPY_GATE,
        ]
        self.assertEqual(sorted(map(tuple, commands)), sorted(map(tuple, expected)))
        # A hardware-only feature is linted and its tests are skipped...
        self.assertIn(cf.feature_clippy("sparkflash-kernels", "cuda"), commands)
        self.assertNotIn(
            ["cargo", "test", "-p", "sparkflash-kernels", "--features", "cuda"], commands
        )
        # ...and for every feature that does run tests, the lint runs first.
        for package, feature in found:
            lint = cf.feature_clippy(package, feature)
            tests = ["cargo", "test", "-p", package, "--features", feature]
            if tests in commands:
                self.assertLess(commands.index(lint), commands.index(tests))
        self.assertEqual(commands[-1], cf.CLIPPY_GATE)
        # Without a toolkit the cuda feature cannot be built at all, so its lint cannot run: the sweep
        # does not pretend, it drops it and says so, and the tests of the other features are untouched.
        without = cf.commands_for(found, toolkit_present=False)
        self.assertNotIn(cf.feature_clippy("sparkflash-kernels", "cuda"), without)
        self.assertIn(cf.feature_clippy("sparkflash-engram", "io-uring"), without)
        self.assertIn(["cargo", "test", "-p", "sparkflash-engram", "--features", "io-uring"], without)


if __name__ == "__main__":
    unittest.main(verbosity=2)
