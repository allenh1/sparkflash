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

"""Unit tests for scripts/compare_native_vs_vllm.py on synthetic host data.

No server, no GPU, no network. Run with:

    python3 -m unittest discover -s scripts/tests -v
"""

import json
import math
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import compare_native_vs_vllm as compare

RELEASED_ROOT = "/cache/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/" + compare.DEFAULT_EXPECT_ROOT


class LogSoftmaxTest(unittest.TestCase):
    def test_uniform(self):
        values = compare.log_softmax([0.0, 0.0, 0.0, 0.0])
        for value in values:
            self.assertAlmostEqual(value, -math.log(4.0), places=12)
        self.assertAlmostEqual(sum(math.exp(v) for v in values), 1.0, places=12)

    def test_known_pair(self):
        values = compare.log_softmax([math.log(1.0), math.log(3.0)])
        self.assertAlmostEqual(values[0], math.log(0.25), places=12)
        self.assertAlmostEqual(values[1], math.log(0.75), places=12)

    def test_stability_with_large_values(self):
        values = compare.log_softmax([1000.0, 999.0])
        self.assertTrue(all(math.isfinite(v) for v in values))
        self.assertAlmostEqual(math.exp(values[0]), 0.7310585786300049, places=9)
        self.assertAlmostEqual(math.exp(values[1]), 0.2689414213699951, places=9)

    def test_empty_is_an_error(self):
        with self.assertRaises(compare.CompareError):
            compare.log_softmax([])

    def test_nan_is_an_error(self):
        with self.assertRaises(compare.CompareError):
            compare.log_softmax([0.0, float("nan")])


class RankingTest(unittest.TestCase):
    def test_ranked_ties_break_by_id(self):
        self.assertEqual(compare._ranked({5: 1.0, 3: 1.0, 4: 2.0}), [4, 3, 5])

    def test_ranked_limit(self):
        self.assertEqual(compare._ranked({1: 1.0, 2: 2.0, 3: 3.0}, limit=2), [3, 2])

    def test_overlap(self):
        self.assertEqual(compare.topk_overlap([1, 2, 3], [3, 2, 1], 3), 1.0)
        self.assertEqual(compare.topk_overlap([1, 2, 3], [4, 5, 6], 3), 0.0)
        self.assertAlmostEqual(compare.topk_overlap([1, 2, 3], [2, 3, 4], 3), 2.0 / 3.0)
        self.assertEqual(compare.topk_overlap([], [1], 5), 0.0)
        self.assertEqual(compare.topk_overlap([1], [1], 10), 1.0)


class DivergenceTest(unittest.TestCase):
    def test_kl_identical_is_zero(self):
        p = [0.2, 0.3, 0.5]
        self.assertAlmostEqual(compare.kl_divergence(p, p), 0.0, places=12)

    def test_kl_known_two_point(self):
        self.assertAlmostEqual(compare.kl_divergence([0.5, 0.5], [0.25, 0.75]), 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0), places=12)

    def test_kl_is_asymmetric(self):
        p = [0.9, 0.1]
        q = [0.5, 0.5]
        self.assertNotAlmostEqual(compare.kl_divergence(p, q), compare.kl_divergence(q, p))

    def test_kl_zero_q_is_infinite(self):
        self.assertEqual(compare.kl_divergence([1.0, 0.0], [0.0, 1.0]), float("inf"))

    def test_js_identical_is_zero(self):
        p = [0.2, 0.3, 0.5]
        self.assertAlmostEqual(compare.js_divergence(p, p), 0.0, places=12)

    def test_js_symmetric_and_bounded(self):
        p = [0.9, 0.1]
        q = [0.1, 0.9]
        self.assertAlmostEqual(compare.js_divergence(p, q), compare.js_divergence(q, p), places=12)
        self.assertLessEqual(compare.js_divergence(p, q), math.log(2.0) + 1e-12)

    def test_divergence_length_mismatch(self):
        with self.assertRaises(compare.CompareError):
            compare.kl_divergence([1.0], [1.0, 1.0])


class DumpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dump = os.path.join(self.tmp.name, "dump.jsonl")
        self.bin = self.dump + ".bin"
        rows = [
            {"position": 0, "vocab": 3, "logits_offset": 0},
            {"position": 1, "vocab": 3, "logits_offset": 12},
        ]
        with open(self.dump, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        with open(self.bin, "wb") as handle:
            handle.write(struct.pack("<3f", 1.0, 2.0, 3.0))
            handle.write(struct.pack("<3f", -1.0, 0.0, 1.0))

    def test_read_dump_and_logits(self):
        rows = compare.read_dump(self.dump)
        self.assertEqual([row["position"] for row in rows], [0, 1])
        self.assertEqual(compare.read_logits(self.bin, rows[0]["logits_offset"], 3), [1.0, 2.0, 3.0])
        self.assertEqual(compare.read_logits(self.bin, rows[1]["logits_offset"], 3), [-1.0, 0.0, 1.0])

    def test_short_bin_is_an_error(self):
        with self.assertRaises(compare.CompareError):
            compare.read_logits(self.bin, 12, 4)

    def test_duplicate_positions_are_an_error(self):
        with open(self.dump, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"position": 0, "vocab": 3, "logits_offset": 0}) + "\n")
        with self.assertRaises(compare.CompareError):
            compare.read_dump(self.dump)

    def test_missing_field_is_an_error(self):
        with open(self.dump, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"position": 0, "logits_offset": 0}) + "\n")
        with self.assertRaises(compare.CompareError):
            compare.read_dump(self.dump)


    def test_rows_naming_another_sidecar_are_refused(self):
        rows = compare.read_dump(self.dump)
        rows[0]["logits_bin"] = os.path.join(self.tmp.name, "other.jsonl.bin")
        with self.assertRaises(compare.CompareError):
            compare.check_dump_sidecar(rows, self.dump, self.bin)

    def test_a_sidecar_with_another_dumps_name_is_refused(self):
        rows = compare.read_dump(self.dump)
        with self.assertRaises(compare.CompareError):
            compare.check_dump_sidecar(rows, self.dump, os.path.join(self.tmp.name, "other.jsonl.bin"))

    def test_a_wrong_length_sidecar_is_refused(self):
        rows = compare.read_dump(self.dump)
        with open(self.bin, "ab") as handle:
            handle.write(b"\0\0\0\0")
        with self.assertRaises(compare.CompareError):
            compare.check_dump_sidecar(rows, self.dump, self.bin)

    def test_non_tiling_offsets_are_refused(self):
        rows = compare.read_dump(self.dump)
        rows[1]["logits_offset"] = 0
        with self.assertRaises(compare.CompareError):
            compare.check_dump_sidecar(rows, self.dump, self.bin)

    def test_writer_shaped_dump_and_sidecar_are_accepted(self):
        rows = compare.read_dump(self.dump)
        for row in rows:
            row["logits_bin"] = self.bin
        compare.check_dump_sidecar(rows, self.dump, self.bin)

    def test_the_existing_d40_demo_dump_is_readable(self):
        """The compatibility fixture is a dump this test did not write."""
        path = os.path.normpath(os.path.join(SCRIPTS, "..", "crates", "sparkflash-cli", "tests",
                                             "fixtures", "logits-dump-demo-scale.jsonl"))
        rows = compare.read_dump(path)
        self.assertEqual([row["position"] for row in rows], list(range(129, 138)))
        self.assertEqual([row["vocab"] for row in rows], [129280] * 9)
        self.assertEqual([row["logits_offset"] for row in rows],
                         [index * 129280 * 4 for index in range(9)])


class PositionMetricsTest(unittest.TestCase):
    def _oracle(self, logprobs):
        ranked = sorted(logprobs.items(), key=lambda item: (-item[1], item[0]))
        return {
            "position": 0,
            "target_token": min(logprobs),
            "logprob_count": len(logprobs),
            "logprobs": {str(token): value for token, value in logprobs.items()},
            "top": [[token, value] for token, value in ranked],
        }

    def test_perfect_match(self):
        logits = [1.0, 0.0, -1.0]
        oracle_lp = {token: value for token, value in enumerate(compare.log_softmax(logits))}
        row = {"position": 0, "vocab": 3, "reference_id": 0}
        metrics = compare.position_metrics(row, logits, self._oracle(oracle_lp))
        self.assertEqual(metrics["oracle_argmax"], metrics["device_argmax"])
        self.assertEqual(metrics["top1_agreement"], True)
        self.assertEqual(metrics["top1000_overlap"], 1.0)
        self.assertEqual(metrics["top5_overlap"], 1.0)
        self.assertAlmostEqual(metrics["max_abs_delta"], 0.0, places=12)
        self.assertAlmostEqual(metrics["mean_abs_delta"], 0.0, places=12)
        self.assertAlmostEqual(metrics["kl_oracle_to_device"], 0.0, places=12)
        self.assertAlmostEqual(metrics["js_divergence"], 0.0, places=12)
        self.assertAlmostEqual(metrics["oracle_topk_mass"], 1.0, places=12)
        self.assertTrue(metrics["prefix_match"])

    def test_truncated_oracle_mass_still_matches_on_shared_ids(self):
        full_logits = [1.0, 0.0, -1.0, -5.0]
        full_lp = compare.log_softmax(full_logits)
        oracle_lp = {token: full_lp[token] for token in (0, 1, 2)}
        row = {"position": 0, "vocab": 4, "reference_id": 0}
        metrics = compare.position_metrics(row, full_logits, self._oracle(oracle_lp))
        self.assertAlmostEqual(metrics["max_abs_delta"], 0.0, places=12)
        self.assertAlmostEqual(metrics["kl_oracle_to_device"], 0.0, places=12)
        self.assertAlmostEqual(metrics["oracle_topk_mass"], 1.0 - math.exp(full_lp[3]), places=12)

    def test_argmax_flip_and_positive_kl(self):
        oracle_lp = {token: value for token, value in enumerate(compare.log_softmax([1.0, 0.0, -1.0]))}
        device_logits = [0.5, 1.5, -1.0]
        row = {"position": 0, "vocab": 3, "reference_id": 1}
        metrics = compare.position_metrics(row, device_logits, self._oracle(oracle_lp))
        self.assertEqual(metrics["oracle_argmax"], 0)
        self.assertEqual(metrics["device_argmax"], 1)
        self.assertFalse(metrics["top1_agreement"])
        self.assertGreater(metrics["kl_oracle_to_device"], 0.0)
        self.assertGreater(metrics["max_abs_delta"], 0.0)
        self.assertGreaterEqual(metrics["js_divergence"], 0.0)

    def test_prefix_mismatch_is_flagged(self):
        logits = [1.0, 0.0]
        oracle_lp = {token: value for token, value in enumerate(compare.log_softmax(logits))}
        row = {"position": 0, "vocab": 2, "reference_id": 1}
        oracle_position = self._oracle(oracle_lp)
        oracle_position["target_token"] = 0
        metrics = compare.position_metrics(row, logits, oracle_position)
        self.assertFalse(metrics["prefix_match"])

    def test_absent_reference_id_is_unknown_not_a_match(self):
        logits = [1.0, 0.0]
        oracle_lp = {token: value for token, value in enumerate(compare.log_softmax(logits))}
        metrics = compare.position_metrics({"position": 0, "vocab": 2}, logits, self._oracle(oracle_lp))
        self.assertIsNone(metrics["prefix_match"])

    def test_unreadable_reference_id_is_unknown_not_a_match(self):
        logits = [1.0, 0.0]
        oracle_lp = {token: value for token, value in enumerate(compare.log_softmax(logits))}
        row = {"position": 0, "vocab": 2, "reference_id": "0"}
        metrics = compare.position_metrics(row, logits, self._oracle(oracle_lp))
        self.assertIsNone(metrics["prefix_match"])


class AggregateAndBarsTest(unittest.TestCase):
    def _position(self, top1, kl, delta):
        return {"position": 0, "top1_agreement": top1, "prefix_match": True,
                "top5_overlap": 0.8, "top20_overlap": 0.7, "top100_overlap": 0.6, "top1000_overlap": 0.5,
                "max_abs_delta": delta, "mean_abs_delta": delta / 2.0, "kl_oracle_to_device": kl,
                "js_divergence": kl / 2.0, "oracle_topk_mass": 0.6}

    def test_aggregate_means(self):
        aggregate = compare.aggregate_metrics([self._position(True, 0.1, 2.0), self._position(False, 0.3, 4.0)])
        self.assertEqual(aggregate["positions"], 2)
        self.assertAlmostEqual(aggregate["top1_agreement_rate"], 0.5)
        self.assertAlmostEqual(aggregate["top5_overlap_mean"], 0.8)
        self.assertAlmostEqual(aggregate["max_abs_delta_max"], 4.0)
        self.assertAlmostEqual(aggregate["max_abs_delta_mean"], 3.0)
        self.assertAlmostEqual(aggregate["kl_mean"], 0.2)

    def test_bars_pass_and_fail(self):
        aggregate = compare.aggregate_metrics([self._position(True, 0.1, 2.0)])
        passing = compare.evaluate_bars({"bars": {"top1_agreement_min": 1.0, "max_abs_delta_max": 3.0}}, aggregate)
        self.assertTrue(passing["pass"])
        failing = compare.evaluate_bars({"bars": {"max_abs_delta_max": 1.0}}, aggregate)
        self.assertFalse(failing["pass"])
        self.assertEqual(len(failing["bars"]), 1)

    def test_unknown_prefix_counts_as_a_mismatch(self):
        first = self._position(True, 0.1, 1.0)
        second = self._position(True, 0.1, 1.0)
        second["prefix_match"] = None
        aggregate = compare.aggregate_metrics([first, second])
        self.assertEqual(aggregate["prefix_mismatches"], 1)
        self.assertEqual(aggregate["prefix_unknown"], 1)

    def test_unknown_bar_is_an_error(self):
        aggregate = compare.aggregate_metrics([self._position(True, 0.1, 2.0)])
        with self.assertRaises(compare.CompareError):
            compare.evaluate_bars({"bars": {"token_equality": 1}}, aggregate)

    def test_gate_bars_shape(self):
        compare.check_gate_bars(
            {"schema": compare.GATE_BARS_SCHEMA, "status": "ok", "bar_class": "idle_distractor",
             "bars": {"positions_min": 1}}, "bars.json")
        bad_documents = [
            {"bars": {"positions_min": 1}},
            {"schema": compare.GATE_BARS_SCHEMA, "status": "requires_review",
             "bar_class": "idle_distractor", "bars": {"positions_min": 1}},
            {"schema": compare.GATE_BARS_SCHEMA, "status": "ok",
             "bar_class": "busy_distractor", "bars": {"positions_min": 1}},
            {"schema": compare.GATE_BARS_SCHEMA, "status": "ok",
             "bar_class": "idle_distractor", "bars": {}},
        ]
        for document in bad_documents:
            with self.assertRaises(compare.CompareError):
                compare.check_gate_bars(document, "bars.json")


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dump = os.path.join(self.tmp.name, "dump.jsonl")
        self.bin = self.dump + ".bin"
        self.oracle = os.path.join(self.tmp.name, "oracle.json")
        self.bars = os.path.join(self.tmp.name, "bars.json")
        logits = [1.0, 0.0, -1.0]
        oracle_lp = {token: value for token, value in enumerate(compare.log_softmax(logits))}
        with open(self.dump, "w", encoding="utf-8") as handle:
            for position in (0, 1):
                handle.write(json.dumps({"position": position, "vocab": 3, "logits_offset": position * 12,
                                         "reference_id": 0}) + "\n")
        with open(self.bin, "wb") as handle:
            for _ in range(2):
                handle.write(struct.pack("<3f", *logits))
        positions = []
        for position in (0, 1):
            positions.append({"position": position, "target_token": 0, "logprob_count": 3,
                              "logprobs": {str(token): value for token, value in oracle_lp.items()}})
        oracle = {
            "schema": "sparkflash-native-oracle/1",
            "source": "vllm-released",
            "served": {"root": RELEASED_ROOT, "id": "m"},
            "suite": {"path": "suite.json", "sha256": "0" * 64},
            "prompts": [{"id": "p", "status": "ok", "response_sha256": "1" * 64, "positions": positions}],
        }
        with open(self.oracle, "w", encoding="utf-8") as handle:
            json.dump(oracle, handle)
        self._write_bars()

    def _write_bars(self, **overrides):
        document = {
            "schema": compare.GATE_BARS_SCHEMA,
            "status": "ok",
            "bar_class": "idle_distractor",
            "bars": {"top1_agreement_min": 1.0, "max_abs_delta_max": 0.1, "positions_min": 2},
        }
        document.update(overrides)
        with open(self.bars, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return self.bars

    def test_unbared_run_writes_evidence(self):
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json])
        self.assertEqual(code, 0)
        with open(out_json, encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["verdict"], "UNBARED")
        self.assertEqual(result["aggregate"]["positions"], 2)
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "out.md")))

    def test_bars_pass_run(self):
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json, "--bars", self.bars])
        self.assertEqual(code, 0)
        with open(out_json, encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["verdict"], "PASS")

    def test_plain_bars_file_is_refused(self):
        with open(self.bars, "w", encoding="utf-8") as handle:
            json.dump({"bars": {"top1_agreement_min": 1.0}}, handle)
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--bars", self.bars])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_requires_review_bars_file_is_refused(self):
        self._write_bars(status="requires_review")
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--bars", self.bars])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_busy_bar_class_is_refused(self):
        self._write_bars(bar_class="busy_distractor")
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--bars", self.bars])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_unknown_bar_name_with_gate_shape_is_refused(self):
        self._write_bars(bars={"token_equality": 1.0})
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--bars", self.bars])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_expect_vocab_mismatch_is_refused(self):
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--expect-vocab", "129280"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_positions_missing_from_dump_is_refused(self):
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--positions", "0,1,2"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_positions_missing_from_oracle_is_refused(self):
        self._rewrite_oracle(lambda prompt: prompt.__setitem__("positions", prompt["positions"][:1]))
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--positions", "1"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_duplicate_positions_are_refused(self):
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--positions", "0,0"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_positions_pin_the_compared_set(self):
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--positions", "0"])
        self.assertEqual(code, 0)
        with open(out_json, encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["aggregate"]["positions"], 1)
        self.assertEqual(result["verdict"], "UNBARED")

    def test_not_like_for_like_overrides_passing_bars(self):
        self._rewrite_dump(lambda row: row.pop("reference_id", None))
        self._write_bars()
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--bars", self.bars])
        self.assertEqual(code, 1)
        with open(out_json, encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["verdict"], "NOT_LIKE_FOR_LIKE")
        self.assertTrue(result["bars"]["pass"])
        self.assertEqual(result["aggregate"]["prefix_mismatches"], 2)
        self.assertEqual(result["aggregate"]["prefix_unknown"], 2)

    def test_wrong_root_is_rejected(self):
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json,
                             "--expect-root", "deadbeef"])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_no_overlap_is_rejected(self):
        with open(self.oracle, encoding="utf-8") as handle:
            oracle = json.load(handle)
        oracle["prompts"][0]["positions"][0]["position"] = 99
        oracle["prompts"][0]["positions"][1]["position"] = 98
        with open(self.oracle, "w", encoding="utf-8") as handle:
            json.dump(oracle, handle)
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json])
        self.assertEqual(code, 2)

    def _rewrite_dump(self, mutate):
        with open(self.dump, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row in rows:
            mutate(row)
        with open(self.dump, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _rewrite_oracle(self, mutate):
        with open(self.oracle, encoding="utf-8") as handle:
            oracle = json.load(handle)
        mutate(oracle["prompts"][0])
        with open(self.oracle, "w", encoding="utf-8") as handle:
            json.dump(oracle, handle)

    def test_dump_without_reference_id_is_flagged_not_matched(self):
        self._rewrite_dump(lambda row: row.pop("reference_id", None))
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json])
        self.assertEqual(code, 1)
        with open(out_json, encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["verdict"], "NOT_LIKE_FOR_LIKE")
        self.assertEqual(result["aggregate"]["prefix_mismatches"], 2)
        self.assertEqual(result["aggregate"]["prefix_unknown"], 2)
        with open(os.path.join(self.tmp.name, "out.md"), encoding="utf-8") as handle:
            self.assertIn("UNKNOWN", handle.read())

    def test_dump_naming_another_sidecar_is_rejected(self):
        self._rewrite_dump(lambda row: row.update(logits_bin=os.path.join(self.tmp.name, "other.jsonl.bin")))
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))

    def test_wrong_length_sidecar_is_rejected(self):
        with open(self.bin, "ab") as handle:
            handle.write(b"\0\0\0\0")
        out_json = os.path.join(self.tmp.name, "out.json")
        code = compare.main(["--dump", self.dump, "--oracle", self.oracle, "--out-json", out_json])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(out_json))


if __name__ == "__main__":
    unittest.main()
