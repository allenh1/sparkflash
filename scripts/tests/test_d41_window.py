#!/usr/bin/env python3
"""Unit tests for the D41 v2 window runner and the close-margin flip investigation.

Offline only: synthetic captures built with the same fixtures as the floor tests. No server, no GPU.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)

import d41_flip_investigation as flip_tool
import d41_floors as floors
import d41_window as window_tool
import test_d41_floors as fx


def write_json(path, doc):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)
        handle.write(chr(10))


def flip_record(record_id, prompt_len=9):
    record = fx.make_record(record_id, "base", prompt_len=prompt_len)
    for position in record["positions"]:
        top = position["top"]
        top[0][1] = -1.0
        top[1][1] = 0.0
    return record


def flipped_block(block_id, block_class, pair_count=3):
    records = []
    pairs = []
    for index in range(pair_count):
        left = fx.make_record(block_id + "-" + str(index) + "-a", "base", prompt_len=8)
        right = flip_record(block_id + "-" + str(index) + "-b")
        records.extend([left, right])
        pairs.append([left["id"], right["id"]])
    return fx.make_block(block_id, block_class, records, pairs=pairs, prompt_len=None)


def flipped_capture():
    return fx.make_capture([fx.repeat_block(), flipped_block("idle-flip", "idle_distractor"),
                            fx.distractor_block("busy", "busy_distractor", mode="busy")])


class D41FlipInvestigationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def path(self, name, doc):
        target = os.path.join(self.root, name)
        write_json(target, doc)
        return target

    def test_clean_capture_has_no_flips(self):
        capture = self.path("clean.json", fx.standard_capture())
        out = os.path.join(self.root, "flips.json")
        code = flip_tool.main(["--capture", capture, "--out-json", out])
        self.assertEqual(code, 0)
        with open(out, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["schema"], flip_tool.SCHEMA)
        self.assertEqual(report["status"], "NO_FLIPS_ABOVE_FLAT")
        self.assertEqual(report["summary"]["flips_above_flat"], 0)
        self.assertTrue(all(row["agrees_with_investigation"] for row in report["verified_against_floor_tool"].values() if row.get("present")))

    def test_flipped_capture_counts_and_agrees_with_the_floor_tool(self):
        capture = self.path("flipped.json", flipped_capture())
        out = os.path.join(self.root, "flips.json")
        code = flip_tool.main(["--capture", capture, "--out-json", out])
        self.assertEqual(code, 3)
        with open(out, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["status"], "REQUIRES_REVIEW")
        self.assertEqual(report["summary"]["flips_above_flat"], 15)
        self.assertEqual(report["summary"]["flips_above_flat_by_class"]["idle_distractor"], 15)
        self.assertTrue(report["verified_against_floor_tool"]["idle_distractor"]["agrees_with_investigation"])
        first = report["flips"][0]
        self.assertNotEqual(first["left"]["top1_token"], first["right"]["top1_token"])
        self.assertGreaterEqual(first["left"]["top1_prob"], report["parameters"]["flat_prob"])

    def test_invalid_capture_exits_2(self):
        capture = self.path("bad.json", {"schema": "not-a-capture"})
        out = os.path.join(self.root, "flips.json")
        code = flip_tool.main(["--capture", capture, "--out-json", out])
        self.assertEqual(code, 2)


class D41WindowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.capture = os.path.join(self.root, "capture.json")
        write_json(self.capture, fx.standard_capture())
        self.out = os.path.join(self.root, "window")

    def test_analyse_clean_capture_writes_bars_and_manifest(self):
        code = window_tool.main(["analyse", "--capture", self.capture, "--out-dir", self.out])
        self.assertEqual(code, 0)
        with open(os.path.join(self.out, "d41-window.json"), "r", encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["schema"], window_tool.SCHEMA)
        self.assertEqual(result["status"], "OK-BARS-WRITTEN")
        self.assertEqual(result["rows"]["floors"]["verdict"], "OK")
        self.assertEqual(result["rows"]["flip_investigation"]["verdict"], "NO-FLIPS")
        bars = result["rows"]["floors"]["bars"]
        self.assertTrue(bars and os.path.exists(bars))
        self.assertTrue(os.path.exists(os.path.join(self.out, "d41-window-sha256.txt")))

    def test_analyse_flipped_capture_requires_review_and_writes_no_bars(self):
        write_json(self.capture, flipped_capture())
        code = window_tool.main(["analyse", "--capture", self.capture, "--out-dir", self.out])
        self.assertEqual(code, 3)
        with open(os.path.join(self.out, "d41-window.json"), "r", encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["status"], "REQUIRES_REVIEW")
        self.assertEqual(result["rows"]["floors"]["verdict"], "REFUSED")
        self.assertEqual(result["rows"]["flip_investigation"]["verdict"], "FLIPS-FOUND")
        self.assertEqual(result["rows"]["flip_investigation"]["flips_above_flat"], 15)
        self.assertFalse(os.path.exists(os.path.join(self.out, "d41-bars.json")))

    def test_analyse_idle_only_capture_names_the_missing_class(self):
        write_json(self.capture, fx.make_capture([fx.repeat_block(), fx.distractor_block("idle", "idle_distractor")]))
        code = window_tool.main(["analyse", "--capture", self.capture, "--out-dir", self.out])
        self.assertEqual(code, 3)
        with open(os.path.join(self.out, "d41-window.json"), "r", encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["rows"]["floors"]["verdict"], "REFUSED")
        self.assertIn("busy_distractor", result["rows"]["floors"]["refusal"])
        self.assertFalse(os.path.exists(os.path.join(self.out, "d41-bars.json")))

    def test_analyse_invalid_capture_exits_2(self):
        write_json(self.capture, {"schema": "not-a-capture"})
        code = window_tool.main(["analyse", "--capture", self.capture, "--out-dir", self.out])
        self.assertEqual(code, 2)

    def test_capture_dry_run_checks_the_frozen_plan_before_network(self):
        plan = os.path.join(SCRIPTS, "d41_capture_plan.json")
        code = window_tool.main(["capture", "--dry-run", "--out-dir", self.out, "--plan", plan])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(self.out, "d41-plan-dry-run.log")))
        with open(os.path.join(self.out, "d41-plan-dry-run.log"), "r", encoding="utf-8") as handle:
            log = handle.read()
        self.assertIn("plan=", log)
        self.assertFalse(any(name.startswith("d41-capture-") for name in os.listdir(self.out)))

    def test_capture_refuses_changed_plan_sha(self):
        plan = os.path.join(SCRIPTS, "d41_capture_plan_idle.json")
        code = window_tool.main(["capture", "--dry-run", "--out-dir", self.out, "--plan", plan])
        self.assertEqual(code, 2)

    def test_capture_refuses_plan_without_busy_class(self):
        plan = os.path.join(SCRIPTS, "d41_capture_plan_idle.json")
        code = window_tool.main(["capture", "--dry-run", "--out-dir", self.out, "--plan", plan, "--plan-sha256", ""])
        self.assertEqual(code, 2)

    def test_capture_refuses_non_empty_out_dir(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "keep.txt"), "w", encoding="utf-8") as handle:
            handle.write("x")
        code = window_tool.main(["capture", "--dry-run", "--out-dir", self.out])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()