#!/usr/bin/env python3
"""Cross-seam test: the capture envelope is one contract with the floor script.

scripts/d41_recapture.py writes sparkflash-d41-capture/1 and scripts/d41_floors.py (serve-dev,
task-41) consumes it. The seam is where a silent class-mixing bug would hide, so it is tested
here as a checked-in test rather than as a one-off review script:

  * a capture in the producer's exact layout, built from the real plan files
    (scripts/d41_capture_plan.json, scripts/d41_capture_plan_idle.json), validates and
    computes floors with all three classes;
  * a mis-tagged block (idle class with a busy verdict, or the reverse) is refused;
  * an idle-verdict block with co-tenant traffic in metrics_during is refused;
  * D38 root attestation, served.id and metrics_source are cross-checked;
  * the idle-only fallback capture is accepted with --require-classes repeat,idle_distractor and
    its absent busy class is recorded as present: false, never silently aggregated.

Host-only: no server, no GPU, no network. When scripts/d41_floors.py is not on the branch (for
example the capture branch before the floor branch merges) the whole class is skipped, and it
starts running unchanged once both scripts share a tree.
"""

import hashlib
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

try:  # the floor script is serve-dev's task-41 deliverable; not on every branch
    import d41_floors as floors
except ImportError:  # pragma: no cover - exercised on the capture-only branch
    floors = None

EXPECT_ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SERVED_ROOT = "/root/.cache/huggingface/hub/models--deepseek-v4.1-flash/snapshots/" + EXPECT_ROOT
MODEL = "deepseek-v4.1-flash"
TIMESTAMP = "2026-01-01T00:00:00Z"
CAPTURE_SCHEMA = "sparkflash-d41-capture/1"
PLAN_SCHEMA = "sparkflash-d41-capture-plan/1"
PRODUCER_SHA = "0" * 64
SUITE_SHA = "2" * 64
SEAM_POSITIONS = 5


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def metrics_snapshot(running=0, waiting=0):
    return {
        "num_requests_running": running,
        "num_requests_waiting": waiting,
        "raw_sha256": "1" * 64,
        "at_utc": TIMESTAMP,
    }


def record(request_id, prompt_len, variant=0.0, identical=None):
    """One producer record with a full 1000-entry top list per position (the frozen recipe)."""
    positions = []
    for position in range(SEAM_POSITIONS):
        top = []
        for token in range(1000):
            logprob = -0.01 * (token + 1) + variant * math.sin((token + 1) * (position + 3))
            top.append([token, logprob])
        top.sort(key=lambda item: (-item[1], item[0]))
        positions.append({"position": position, "logprob_count": len(top), "top": top})
    prompt_key = identical or request_id
    return {
        "id": request_id,
        "prompt_len": prompt_len,
        "batch_size": 1,
        "prompt_sha256": sha256("prompt:" + prompt_key),
        "request_sha256": sha256("request:" + prompt_key),
        "response_sha256": sha256("response:" + request_id),
        "wall_s": 0.01,
        "repeat_of": None,
        "positions": positions,
    }


def idleness(verdict, during=None):
    if verdict == "idle":
        return {
            "metrics_source": "/metrics",
            "metrics_before": metrics_snapshot(0, 0),
            "metrics_after": metrics_snapshot(0, 0),
            "metrics_during": [metrics_snapshot(0, 0)] if during is None else during,
            "observed_other_traffic": False,
            "external_peers": [],
            "idle_verdict": "idle",
        }
    return {
        "metrics_source": "/metrics",
        "metrics_before": metrics_snapshot(0, 0),
        "metrics_after": metrics_snapshot(0, 0),
        "metrics_during": (
            [metrics_snapshot(1, 0), metrics_snapshot(2, 0)] if during is None else during
        ),
        "observed_other_traffic": True,
        "external_peers": [],
        "idle_verdict": "busy",
    }


def make_block(block_id, block_class, records, pairs, verdict, during=None):
    lengths = [record["prompt_len"] for record in records]
    return {
        "block_id": block_id,
        "class": block_class,
        "started_utc": TIMESTAMP,
        "finished_utc": TIMESTAMP,
        "prompt_len": lengths[0] if len(set(lengths)) == 1 else None,
        "batch_size": 1,
        "idleness": idleness(verdict, during),
        "pairs": pairs,
        "requests": records,
    }


def make_capture(blocks, suite_path, suite_sha=SUITE_SHA, suite_count=6):
    return {
        "schema": CAPTURE_SCHEMA,
        "capture_id": "seam-test-capture",
        "producer": {"script": "scripts/d41_recapture.py", "script_sha256": PRODUCER_SHA},
        "started_utc": TIMESTAMP,
        "finished_utc": TIMESTAMP,
        "status": "ok",
        "warnings": [],
        "plan": {"path": suite_path, "sha256": suite_sha, "schema": PLAN_SCHEMA},
        "served": {"id": MODEL, "root": SERVED_ROOT},
        "expect_root": EXPECT_ROOT,
        "root_verified": True,
        "recipe": {
            "endpoint": "/v1/completions",
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "prompt_logprobs": 1000,
            "ignore_eos": True,
            "model": MODEL,
        },
        "suite": {"path": suite_path, "sha256": suite_sha, "prompt_count": suite_count, "note": "seam test"},
        "blocks": blocks,
    }


def plan_block(plan, block_class):
    for block in plan["blocks"]:
        if block["class"] == block_class:
            return block
    raise AssertionError(f"plan has no {block_class!r} block")


def plan_lengths(block):
    return {request["id"]: len(request["prompt"]) for request in block["requests"]}


def repeat_block_from_plan(plan, block_id="seam-repeat"):
    source = plan_block(plan, "repeat")
    lengths = plan_lengths(source)
    records = [record(request_id, lengths[request_id], identical=source["id"])
               for request_id in lengths]
    pairs = [[source["requests"][index]["id"], source["requests"][index + 1]["id"]]
             for index in range(len(source["requests"]) - 1)]
    return make_block(block_id, "repeat", records, pairs, "idle")


def distractor_blocks_from_plan(plan):
    lengths = plan_lengths(plan_block(plan, "idle_distractor"))
    busy_lengths = plan_lengths(plan_block(plan, "busy_distractor"))
    lefts = {pair[0] for pair in plan_block(plan, "idle_distractor")["pairs"]}
    busy_lefts = {pair[0] for pair in plan_block(plan, "busy_distractor")["pairs"]}
    idle_records = [
        record(request_id, length, variant=0.0 if request_id in lefts else 0.0005)
        for request_id, length in lengths.items()
    ]
    idle_block = make_block(
        "seam-idle-distractor",
        "idle_distractor",
        idle_records,
        [list(pair) for pair in plan_block(plan, "idle_distractor")["pairs"]],
        "idle",
    )
    busy_records = [
        record(request_id, length, variant=0.0 if request_id in busy_lefts else 0.0005)
        for request_id, length in busy_lengths.items()
    ]
    busy_block = make_block(
        "seam-busy-distractor",
        "busy_distractor",
        busy_records,
        [list(pair) for pair in plan_block(plan, "busy_distractor")["pairs"]],
        "busy",
    )
    return idle_block, busy_block


def load_plan(name):
    path = os.path.join(SCRIPTS, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle), path


def standard_capture():
    primary, primary_path = load_plan("d41_capture_plan.json")
    idle_block, busy_block = distractor_blocks_from_plan(primary)
    blocks = [repeat_block_from_plan(primary), idle_block, busy_block]
    distinct = 0
    seen = set()
    for block in primary["blocks"]:
        for request in block["requests"]:
            key = tuple(request["prompt"])
            if key not in seen:
                seen.add(key)
                distinct += 1
    return make_capture(blocks, primary_path, suite_count=distinct)


def validate(capture, tmpdir, name="capture.json"):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(capture, handle)
    return floors.validate_capture(capture, path)


@unittest.skipUnless(floors is not None, "scripts/d41_floors.py is not on this branch")
class CaptureSchemaSeamTests(unittest.TestCase):
    def mutated(self, mutate):
        capture = standard_capture()
        mutate(capture)
        return capture

    def assert_refused(self, capture, fragment):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "capture.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(capture, handle)
            with self.assertRaises(floors.D41Error) as raised:
                floors.validate_capture(capture, path)
        self.assertIn(fragment, str(raised.exception))

    def test_all_three_classes_validate_and_compute_floors(self):
        capture = standard_capture()
        with tempfile.TemporaryDirectory() as tmpdir:
            normalised = validate(capture, tmpdir)
            report = floors.compute([normalised])
            self.assertEqual(
                [block["class"] for block in normalised["blocks"]],
                ["repeat", "idle_distractor", "busy_distractor"],
            )
            for cls in ("repeat", "idle_distractor", "busy_distractor"):
                self.assertTrue(report["classes"][cls]["present"], cls)
            self.assertEqual(report["classes"]["repeat"]["pairs"], 3)
            self.assertEqual(report["classes"]["idle_distractor"]["pairs"], 3)
            self.assertEqual(
                report["classes"]["idle_distractor"]["position_comparisons"], 3 * SEAM_POSITIONS
            )
            self.assertIn("gate_bars", report["bars"])
            self.assertIn("js_mean_max", report["bars"]["gate_bars"])
            self.assertEqual(report["bars"]["bar_class"], "idle_distractor")
            rendered = floors.render_markdown(report)
            self.assertIn("| idle_distractor | 1 | 3 |", rendered)

    def test_idle_class_with_a_busy_verdict_is_refused(self):
        def mutate(capture):
            for block in capture["blocks"]:
                if block["class"] == "idle_distractor":
                    block["idleness"] = idleness("busy")

        self.assert_refused(self.mutated(mutate), "idle_distractor block must be captured idle")

    def test_busy_class_with_an_idle_verdict_is_refused(self):
        def mutate(capture):
            for block in capture["blocks"]:
                if block["class"] == "busy_distractor":
                    block["idleness"] = idleness("idle")

        self.assert_refused(self.mutated(mutate), "busy_distractor block must be captured busy")

    def test_idle_block_with_co_tenant_traffic_in_during_is_refused(self):
        def mutate(capture):
            for block in capture["blocks"]:
                if block["class"] == "idle_distractor":
                    block["idleness"]["metrics_during"] = [metrics_snapshot(2, 0)]

        self.assert_refused(self.mutated(mutate), "co-tenant traffic while the block was tagged idle")

    def test_own_in_flight_request_in_during_is_still_idle(self):
        capture = standard_capture()
        for block in capture["blocks"]:
            if block["class"] == "idle_distractor":
                block["idleness"]["metrics_during"] = [metrics_snapshot(1, 0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            normalised = validate(capture, tmpdir)
        classes = {block["class"]: block for block in normalised["blocks"]}
        self.assertEqual(classes["idle_distractor"]["idleness"]["idle_verdict"], "idle")

    def test_served_root_must_match_expect_root(self):
        def mutate(capture):
            capture["served"]["root"] = "/root/.cache/huggingface/hub/models--x/snapshots/deadbeef"

        self.assert_refused(self.mutated(mutate), "does not match expect_root")

    def test_served_id_is_required(self):
        def mutate(capture):
            del capture["served"]["id"]

        self.assert_refused(self.mutated(mutate), "served.id")

    def test_metrics_source_must_be_the_metrics_endpoint(self):
        def mutate(capture):
            for block in capture["blocks"]:
                block["idleness"]["metrics_source"] = "/stats"

        self.assert_refused(self.mutated(mutate), "expected '/metrics'")

    def test_idle_only_fallback_capture_is_accepted_with_required_classes(self):
        primary, _ = load_plan("d41_capture_plan.json")
        idle_plan, idle_plan_path = load_plan("d41_capture_plan_idle.json")
        # The fallback plan is the primary plan minus the busy block (blocks 0 and 1 verbatim).
        self.assertEqual(idle_plan["blocks"], primary["blocks"][:2])
        idle_block, _ = distractor_blocks_from_plan(primary)
        blocks = [repeat_block_from_plan(primary), idle_block]
        capture = make_capture(blocks, idle_plan_path, suite_count=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fallback.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(capture, handle)
            normalised = floors.validate_capture(capture, path)
            with self.assertRaises(floors.D41Error) as raised:
                floors.compute([normalised])
            self.assertIn("missing required class", str(raised.exception))
            self.assertIn("busy_distractor", str(raised.exception))
            report = floors.compute(
                [normalised], required_classes=("repeat", "idle_distractor")
            )
        self.assertEqual(report["parameters"]["required_classes"], ["repeat", "idle_distractor"])
        self.assertEqual(report["classes"]["busy_distractor"], {"present": False})
        rendered = floors.render_markdown(report)
        self.assertIn("| busy_distractor | 0 | 0 | 0 |", rendered)


if __name__ == "__main__":
    unittest.main()
