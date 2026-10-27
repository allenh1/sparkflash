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

"""Unit tests for scripts/d41_floors.py on synthetic offline captures.

No server, no GPU, no network. Run with:

    python3 -m unittest discover -s scripts/tests -v
"""

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import d41_floors as floors

ROOT = "/root/.cache/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
EXPECT_ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SUITE_SHA = "b" * 64
START = "2026-10-19T17:30:00Z"
FINISH = "2026-10-19T17:31:00Z"
PROMPT_LOGPROBS = 1000
POSITIONS = 5


def metrics(running=0, waiting=0):
    return {
        "num_requests_running": running,
        "num_requests_waiting": waiting,
        "raw_sha256": "c" * 64,
    }


def idleness(mode="idle", during=None):
    if mode == "idle":
        return {
            "metrics_source": "/metrics",
            "metrics_before": metrics(0, 0),
            "metrics_after": metrics(0, 0),
            "metrics_during": list(during or []),
            "observed_other_traffic": False,
            "external_peers": [],
            "idle_verdict": "idle",
        }
    effective = [metrics(2, 0)] if during is None else during
    return {
        "metrics_source": "/metrics",
        "metrics_before": metrics(0, 0),
        "metrics_after": metrics(0, 0),
        "metrics_during": list(effective),
        "observed_other_traffic": True,
        "external_peers": [],
        "idle_verdict": "busy",
    }


def distribution(variant, position):
    entries = []
    for token in range(PROMPT_LOGPROBS):
        if variant == "base":
            logprob = -0.001 * token
        elif variant == "perturbed":
            logprob = -0.001 * token + 0.0003 * math.sin(0.37 * token + 0.11 * position)
        elif variant == "tiny":
            logprob = -0.001 * token + 1e-9
        elif variant == "outlier":
            logprob = -0.05 * token
        else:
            raise AssertionError(variant)
        entries.append([token, logprob])
    return entries


def make_record(record_id, variant, prompt_len=8, request_sha=None, prompt_sha=None, response_sha=None):
    request_sha = request_sha or hashlib.sha256(record_id.encode()).hexdigest()
    prompt_sha = prompt_sha or "d" * 64
    response_sha = response_sha or hashlib.sha256(("response/" + record_id).encode()).hexdigest()
    return {
        "id": record_id,
        "prompt_len": prompt_len,
        "batch_size": 1,
        "prompt_sha256": prompt_sha,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "positions": [
            {"position": position, "logprob_count": PROMPT_LOGPROBS, "top": distribution(variant, position)}
            for position in range(POSITIONS)
        ],
    }


def make_block(block_id, block_class, records, pairs=None, prompt_len=8, mode="idle", during=None):
    block = {
        "block_id": block_id,
        "class": block_class,
        "started_utc": START,
        "finished_utc": FINISH,
        "prompt_len": prompt_len,
        "batch_size": 1,
        "idleness": idleness(mode, during),
        "requests": records,
    }
    if pairs is not None:
        block["pairs"] = pairs
    return block


def make_capture(blocks, capture_id="test-capture"):
    return {
        "schema": floors.CAPTURE_SCHEMA,
        "capture_id": capture_id,
        "producer": {"script": "scripts/d41_recapture.py", "script_sha256": "9" * 64},
        "started_utc": START,
        "finished_utc": FINISH,
        "served": {"id": "deepseek-v4", "root": ROOT},
        "expect_root": EXPECT_ROOT,
        "root_verified": True,
        "recipe": {
            "endpoint": "/v1/completions",
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "prompt_logprobs": PROMPT_LOGPROBS,
            "ignore_eos": True,
        },
        "suite": {"path": "plan.json", "sha256": SUITE_SHA, "prompt_count": 1},
        "plan": {"path": "plan.json", "sha256": "7" * 64, "schema": "test-plan/1"},
        "warnings": [],
        "status": "ok",
        "blocks": blocks,
    }


def repeat_block(block_id="repeat", variants=None, records=4):
    variants = variants or ["base"] * records
    rows = []
    for index in range(records):
        rows.append(
            make_record(
                f"{block_id}-{index}",
                variants[index],
                request_sha="e" * 64,
                prompt_sha="d" * 64,
            )
        )
    return make_block(block_id, "repeat", rows, prompt_len=8)


def distractor_block(block_id, block_class, pair_count=3, mode="idle", during=None):
    records = []
    pairs = []
    for index in range(pair_count):
        left = make_record(f"{block_id}-{index}-a", "base", prompt_len=8)
        right = make_record(f"{block_id}-{index}-b", "perturbed", prompt_len=9)
        records.extend([left, right])
        pairs.append([left["id"], right["id"]])
    return make_block(
        block_id, block_class, records, pairs=pairs, prompt_len=None, mode=mode, during=during
    )


def standard_capture():
    return make_capture(
        [
            repeat_block(),
            distractor_block("idle", "idle_distractor"),
            distractor_block("busy", "busy_distractor", mode="busy"),
        ]
    )


class FloorTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def path(self, name):
        return os.path.join(self.directory.name, name)

    def write_capture(self, document, name="capture.json"):
        path = self.path(name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def run_main(self, capture, **overrides):
        out_json = self.path("floors.json")
        out_md = self.path("floors.md")
        out_bars = self.path("bars.json")
        arguments = [
            "--capture",
            capture,
            "--out-json",
            out_json,
            "--out-md",
            out_md,
            "--out-bars",
            out_bars,
        ]
        for key, value in overrides.items():
            arguments.extend(["--" + key.replace("_", "-"), str(value)])
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = floors.main(arguments)
        return code, stderr.getvalue(), out_json, out_md, out_bars


class RepeatClassTest(FloorTestCase):
    def test_identical_repeat_is_zero_and_accepted(self):
        capture = self.write_capture(standard_capture())
        code, stderr, out_json, out_md, out_bars = self.run_main(capture)
        self.assertEqual(code, 0, stderr)
        with open(out_json, encoding="utf-8") as handle:
            report = json.load(handle)
        repeat = report["classes"]["repeat"]
        self.assertEqual(repeat["floor"]["kl_mean_max"], 0.0)
        self.assertEqual(repeat["floor"]["max_abs_delta_max"], 0.0)
        self.assertEqual(repeat["floor"]["top20_overlap_mean_min"], 1.0)
        self.assertEqual(repeat["rank1"]["flips_above_flat"], 0)
        self.assertEqual(report["status"], "ok")

    def test_repeat_response_sha256_may_differ(self):
        block = repeat_block()
        # Same request bodies, different response bodies (vLLM echoes a new id/created).
        for record in block["requests"]:
            record["response_sha256"] = hashlib.sha256(record["id"].encode()).hexdigest()
        capture = self.write_capture(make_capture([block, distractor_block("idle", "idle_distractor"), distractor_block("busy", "busy_distractor", mode="busy")]))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 0, stderr)

    def test_nonidentical_repeat_is_refused(self):
        block = repeat_block(variants=["base", "base", "tiny", "tiny"])
        capture = self.write_capture(
            make_capture(
                [
                    block,
                    distractor_block("idle", "idle_distractor"),
                    distractor_block("busy", "busy_distractor", mode="busy"),
                ]
            )
        )
        code, stderr, out_json, out_md, out_bars = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("not bit-identical", stderr)
        for path in (out_json, out_md, out_bars):
            self.assertFalse(os.path.exists(path))

    def test_repeat_zero_is_checked_per_shared_position(self):
        block = repeat_block()
        block["requests"][1]["positions"][2]["top"] = distribution("tiny", 2)
        capture = self.write_capture(
            make_capture(
                [
                    block,
                    distractor_block("idle", "idle_distractor"),
                    distractor_block("busy", "busy_distractor", mode="busy"),
                ]
            )
        )
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("same-shape repeat is not bit-identical", stderr)


class MissingClassTest(FloorTestCase):
    def test_missing_busy_class_is_named(self):
        capture = self.write_capture(
            make_capture([repeat_block(), distractor_block("idle", "idle_distractor")])
        )
        code, stderr, out_json, out_md, out_bars = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("busy_distractor", stderr)
        self.assertIn("missing required class", stderr)
        for path in (out_json, out_md, out_bars):
            self.assertFalse(os.path.exists(path))

    def test_missing_idle_class_is_named(self):
        capture = self.write_capture(
            make_capture([repeat_block(), distractor_block("busy", "busy_distractor", mode="busy")])
        )
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("idle_distractor", stderr)

    def test_explicit_require_classes_can_narrow(self):
        capture = self.write_capture(
            make_capture([repeat_block(), distractor_block("idle", "idle_distractor")])
        )
        code, stderr, out_json, _, _ = self.run_main(capture, require_classes="repeat,idle_distractor")
        self.assertEqual(code, 0, stderr)
        with open(out_json, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["parameters"]["required_classes"], ["repeat", "idle_distractor"])
        self.assertEqual(report["classes"]["busy_distractor"], {"present": False})

    def test_unknown_required_class_is_refused(self):
        capture = self.write_capture(standard_capture())
        code, stderr, _, _, _ = self.run_main(capture, require_classes="repeat,nosuch")
        self.assertEqual(code, 2)
        self.assertIn("unknown class", stderr)


class SampleSizeTest(FloorTestCase):
    def test_too_few_pairs_is_refused(self):
        small = distractor_block("idle", "idle_distractor", pair_count=2)
        capture = self.write_capture(
            make_capture(
                [repeat_block(), small, distractor_block("busy", "busy_distractor", mode="busy")]
            )
        )
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("below min-pairs", stderr)
        self.assertIn("idle_distractor", stderr)

    def test_too_few_position_comparisons_is_refused(self):
        # Three pairs x two shared positions = 6 position comparisons, below 12.
        block = distractor_block("idle", "idle_distractor")
        for record in block["requests"]:
            record["positions"] = record["positions"][:2]
        capture = self.write_capture(
            make_capture(
                [repeat_block(), block, distractor_block("busy", "busy_distractor", mode="busy")]
            )
        )
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("below min-position-comparisons", stderr)

    def test_min_pairs_can_be_raised(self):
        capture = self.write_capture(standard_capture())
        code, stderr, _, _, _ = self.run_main(capture, min_pairs=4)
        self.assertEqual(code, 2)
        self.assertIn("below min-pairs 4", stderr)


class OutlierTest(FloorTestCase):
    def _outlier_capture(self, outlier=True):
        records = []
        pairs = []
        for index in range(8):
            left = make_record(f"n{index}a", "base", prompt_len=8)
            variant = "outlier" if (outlier and index == 0) else "perturbed"
            right = make_record(f"n{index}b", variant, prompt_len=9)
            records.extend([left, right])
            pairs.append([left["id"], right["id"]])
        idle = make_block("idle", "idle_distractor", records, pairs=pairs, prompt_len=None)
        return make_capture(
            [repeat_block(), idle, distractor_block("busy", "busy_distractor", mode="busy")]
        )

    def test_one_outlier_does_not_widen_the_median_bar(self):
        capture = self.write_capture(self._outlier_capture(outlier=True), "outlier.json")
        code, stderr, out_json, _, _ = self.run_main(capture)
        self.assertEqual(code, 0, stderr)
        with open(out_json, encoding="utf-8") as handle:
            report = json.load(handle)
        idle = report["classes"]["idle_distractor"]
        pair_kls = [row["kl_mean"] for row in idle["pair_metrics"]]
        normal = sorted(value for value in pair_kls if value < 1.0)
        outlier_value = max(pair_kls)
        self.assertGreater(outlier_value, 10.0 * normal[-1])
        self.assertAlmostEqual(idle["floor"]["kl_mean_max"], floors._percentile(pair_kls, 0.5), places=12)
        self.assertLess(idle["floor"]["kl_mean_max"], outlier_value / 3.0)
        flagged = [
            outlier for outlier in idle["outliers"] if outlier["metric"] == "kl_mean" and outlier["pair"] == ["n0a", "n0b"]
        ]
        self.assertTrue(flagged, idle["outliers"])
        bar = report["bars"]["bars"]["kl_mean_max"]
        self.assertEqual(bar["value"], max(bar["provisional"], 3.0 * bar["floor"]))
        self.assertLess(bar["value"], outlier_value)

    def test_removing_the_outlier_leaves_the_floor_unchanged(self):
        capture_with = self.write_capture(self._outlier_capture(outlier=True), "with.json")
        capture_without = self.write_capture(self._outlier_capture(outlier=False), "without.json")
        with contextlib.redirect_stderr(io.StringIO()):
            report_with = floors.compute(
                [floors.validate_capture(floors.load_json(capture_with), capture_with)]
            )
            report_without = floors.compute(
                [floors.validate_capture(floors.load_json(capture_without), capture_without)]
            )
        self.assertEqual(
            report_with["classes"]["idle_distractor"]["floor"]["kl_mean_max"],
            report_without["classes"]["idle_distractor"]["floor"]["kl_mean_max"],
        )
        self.assertEqual(
            report_with["bars"]["bars"]["kl_mean_max"]["value"],
            report_without["bars"]["bars"]["kl_mean_max"]["value"],
        )


class IdlenessTest(FloorTestCase):
    def test_idle_block_with_nonzero_after_baseline_is_refused(self):
        blocks = [
            repeat_block(),
            distractor_block("idle", "idle_distractor"),
            distractor_block("busy", "busy_distractor", mode="busy"),
        ]
        blocks[1]["idleness"]["metrics_after"] = metrics(1, 0)
        capture = self.write_capture(make_capture(blocks))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("non-zero metrics_after baseline", stderr)

    def test_unproven_idle_verdict_is_refused(self):
        blocks = [
            repeat_block(),
            distractor_block("idle", "idle_distractor"),
            distractor_block("busy", "busy_distractor", mode="busy"),
        ]
        blocks[0]["idleness"]["idle_verdict"] = "unproven"
        capture = self.write_capture(make_capture(blocks))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("unproven", stderr)

    def test_busy_without_co_tenancy_evidence_is_refused(self):
        blocks = [
            repeat_block(),
            distractor_block("idle", "idle_distractor"),
            distractor_block("busy", "busy_distractor", mode="busy", during=[]),
        ]
        capture = self.write_capture(make_capture(blocks))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("no co-tenancy evidence", stderr)

    def test_idle_block_with_observed_traffic_is_refused(self):
        blocks = [
            repeat_block(),
            distractor_block("idle", "idle_distractor"),
            distractor_block("busy", "busy_distractor", mode="busy"),
        ]
        blocks[1]["idleness"]["observed_other_traffic"] = True
        capture = self.write_capture(make_capture(blocks))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("observed_other_traffic", stderr)


class TopKTest(FloorTestCase):
    def test_k_below_20_is_refused(self):
        capture = self.write_capture(standard_capture())
        code, stderr, _, _, _ = self.run_main(capture, k="5,20,100")
        self.assertEqual(code, 2)
        self.assertIn("k >= 20", stderr)

    def test_k_set_must_be_the_d41_set(self):
        capture = self.write_capture(standard_capture())
        code, stderr, _, _, _ = self.run_main(capture, k="20,50,100")
        self.assertEqual(code, 2)
        self.assertIn("k set must be", stderr)

    def test_capture_needs_top_1000_entries(self):
        capture_doc = standard_capture()
        for block in capture_doc["blocks"]:
            if block["class"] == "idle_distractor":
                block["requests"][0]["positions"][0]["top"] = block["requests"][0]["positions"][0]["top"][:100]
                block["requests"][0]["positions"][0]["logprob_count"] = 100
        capture = self.write_capture(capture_doc)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("for top-1000", stderr)


class SchemaValidationTest(FloorTestCase):
    def test_unknown_schema_is_refused(self):
        document = standard_capture()
        document["schema"] = "sparkflash-d41-capture/0"
        capture = self.write_capture(document)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("schema is", stderr)

    def test_root_mismatch_is_refused(self):
        document = standard_capture()
        document["root_verified"] = False
        capture = self.write_capture(document)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("root_verified", stderr)

    def test_missing_pair_record_is_refused(self):
        document = standard_capture()
        for block in document["blocks"]:
            if block["class"] == "idle_distractor":
                block["pairs"][0][0] = "does-not-exist"
        capture = self.write_capture(document)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("unknown record id", stderr)

    def test_null_block_prompt_len_is_accepted_for_distractors(self):
        capture = self.write_capture(standard_capture())
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 0, stderr)

    def test_extra_envelope_fields_are_tolerated(self):
        document = standard_capture()
        document["extra"] = {"ignored": True}
        document["blocks"][2]["load_generator"] = {"request_sha256": "a" * 64, "max_tokens": 512}
        capture = self.write_capture(document)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 0, stderr)

    def test_captures_must_agree_on_suite(self):
        document_a = standard_capture()
        capture_a = self.write_capture(document_a, "a.json")
        document_b = standard_capture()
        document_b["suite"]["sha256"] = "a" * 64
        capture_b = self.write_capture(document_b, "b.json")
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            code = floors.main(
                [
                    "--capture",
                    capture_a,
                    "--capture",
                    capture_b,
                    "--out-json",
                    self.path("o.json"),
                    "--out-md",
                    self.path("o.md"),
                ]
            )
        self.assertEqual(code, 2)
        self.assertIn("prompt-suite sha256", stderr.getvalue())


class OutputTest(FloorTestCase):
    def test_json_markdown_and_bar_file_have_provenance(self):
        capture = self.write_capture(standard_capture())
        code, stderr, out_json, out_md, out_bars = self.run_main(capture, source_revision="deadbeef")
        self.assertEqual(code, 0, stderr)
        with open(out_json, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["schema"], floors.FLOOR_SCHEMA)
        self.assertEqual(report["provenance"]["captures"][0]["sha256"], floors.sha256_file(capture))
        self.assertEqual(report["provenance"]["floor_script"]["sha256"], floors.sha256_file(floors.__file__))
        self.assertEqual(report["provenance"]["prompt_suite"]["sha256"], SUITE_SHA)
        self.assertEqual(report["provenance"]["served_root"], ROOT)
        self.assertEqual(report["provenance"]["source_revision"], "deadbeef")
        self.assertEqual(report["parameters"]["k_values"], list(floors.TOP_K))
        self.assertIn("top5_overlap_mean_min", report["bars"]["dropped_bars"])
        self.assertNotIn("top5_overlap_mean_min", report["bars"]["bars"])
        with open(out_md, encoding="utf-8") as handle:
            markdown = handle.read()
        self.assertIn("# D41 floor computation", markdown)
        self.assertIn(report["provenance"]["captures"][0]["sha256"], markdown)
        self.assertIn("| bar | provisional | measured floor | rule | D41 value | review |", markdown)
        self.assertIn("top20_overlap_mean_min", markdown)
        self.assertIn("## Dropped bars", markdown)
        self.assertNotIn("| top5_overlap_mean_min |", markdown)
        with open(out_bars, encoding="utf-8") as handle:
            bars = json.load(handle)
        self.assertEqual(bars["schema"], floors.BAR_SCHEMA)
        self.assertIn("top20_overlap_mean_min", bars["bars"])
        self.assertNotIn("top5_overlap_mean_min", bars["bars"])
        self.assertEqual(bars["provenance"]["floor_script_sha256"], floors.sha256_file(floors.__file__))

    def test_review_status_exits_three_and_writes_outputs(self):
        # A deliberately noisy idle class: top-k overlap ~1 but one weak metric
        # forces a looser-than-provisional KL bar.
        blocks = [
            repeat_block(),
            distractor_block("idle", "idle_distractor"),
            distractor_block("busy", "busy_distractor", mode="busy"),
        ]
        for record in blocks[1]["requests"]:
            if record["id"].endswith("-b"):
                for position in record["positions"]:
                    position["top"] = distribution("outlier", position["position"])
        capture = self.write_capture(make_capture(blocks))
        code, stderr, out_json, _, _ = self.run_main(capture)
        self.assertEqual(code, 3, stderr)
        self.assertIn("requires_review", stderr)
        self.assertTrue(os.path.exists(out_json))
        with open(out_json, encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["status"], "requires_review")
        self.assertTrue(report["bars"]["review_required"])


class ClassModeAndProvenanceTest(FloorTestCase):
    def _standard_blocks(self):
        return [
            repeat_block(),
            distractor_block("idle", "idle_distractor"),
            distractor_block("busy", "busy_distractor", mode="busy"),
        ]

    def test_idle_distractor_must_be_captured_idle(self):
        blocks = self._standard_blocks()
        blocks[1]["idleness"] = idleness("busy")
        capture = self.write_capture(make_capture(blocks))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("idle_distractor block must be captured idle", stderr)

    def test_busy_distractor_must_be_captured_busy(self):
        blocks = self._standard_blocks()
        blocks[2]["idleness"] = idleness("idle")
        capture = self.write_capture(make_capture(blocks))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("busy_distractor block must be captured busy", stderr)

    def test_co_tenant_traffic_in_idle_during_snapshot_is_refused(self):
        blocks = self._standard_blocks()
        blocks[1]["idleness"] = idleness("idle", during=[metrics(5, 0)])
        capture = self.write_capture(make_capture(blocks))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("co-tenant traffic", stderr)

    def test_own_request_in_idle_during_snapshot_is_allowed(self):
        blocks = self._standard_blocks()
        blocks[1]["idleness"] = idleness("idle", during=[metrics(1, 0)])
        capture = self.write_capture(make_capture(blocks))
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 0, stderr)

    def test_served_root_must_match_expect_root(self):
        document = standard_capture()
        document["served"]["root"] = (
            "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/"
            "snapshots/deadbeef"
        )
        capture = self.write_capture(document)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("does not match expect_root", stderr)

    def test_served_id_is_required(self):
        document = standard_capture()
        del document["served"]["id"]
        capture = self.write_capture(document)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("served.id", stderr)

    def test_metrics_source_must_be_metrics_endpoint(self):
        document = standard_capture()
        document["blocks"][0]["idleness"]["metrics_source"] = "/stats"
        capture = self.write_capture(document)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 2)
        self.assertIn("metrics_source", stderr)

    def test_block_batch_size_is_metadata_not_forced_to_one(self):
        document = standard_capture()
        document["blocks"][1]["batch_size"] = 2
        capture = self.write_capture(document)
        code, stderr, _, _, _ = self.run_main(capture)
        self.assertEqual(code, 0, stderr)


if __name__ == "__main__":
    unittest.main()
