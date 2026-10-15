#!/usr/bin/env python3
"""Tests for ci/check_features.py, run by the test-features job before the real sweep.

The script's first version had a bug that mattered: it skipped a feature whose value list was
exactly ["dep:<name>"], which is the shape cargo generates for an optional dependency but also
the shape a manifest uses when it declares that feature by hand. sparkflash-engram declares
io-uring = ["dep:io-uring"], so the ring the script exists to cover was the one thing it skipped.
These tests pin the distinction, and the dead-feature rule that catches a feature gating nothing.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_features as cf  # noqa: E402

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
