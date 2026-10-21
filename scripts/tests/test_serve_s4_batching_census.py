#!/usr/bin/env python3
"""Unit tests for scripts/serve-s4-batching-census.py.

No network and no server process: the pure helpers are imported by path (the file name has
hyphens) and exercised on fabricated measurements, so the aggregation and comparison logic the
census artefact depends on is pinned without needing the host or the binary.

Run with:

    python3 -m unittest discover -s scripts/tests -v
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
MODULE_PATH = os.path.join(SCRIPTS, "serve-s4-batching-census.py")

spec = importlib.util.spec_from_file_location("serve_s4_batching_census", MODULE_PATH)
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


class SummarizeTests(unittest.TestCase):
    def test_empty_sample_is_not_a_measurement(self):
        summary = census.summarize([])
        self.assertEqual(summary["n"], 0)
        for key in ("min", "p50", "p95", "max", "mean"):
            self.assertIsNone(summary[key])

    def test_nearest_rank_percentiles(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(census.percentile(samples, 50), 3.0)
        self.assertEqual(census.percentile(samples, 95), 5.0)
        self.assertEqual(census.percentile(samples, 0), 1.0)
        self.assertEqual(census.percentile(samples, 100), 5.0)
        self.assertIsNone(census.percentile([], 50))

    def test_summary_reports_every_field(self):
        summary = census.summarize([0.5, 1.5])
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["min"], 0.5)
        self.assertEqual(summary["max"], 1.5)
        self.assertAlmostEqual(summary["mean"], 1.0)


class RoundingTests(unittest.TestCase):
    def test_floats_are_six_significant_digits(self):
        self.assertEqual(census.round_floats(0.123456789), 0.123457)
        self.assertEqual(census.round_floats(1234567.0), 1234570.0)
        self.assertEqual(census.round_floats(0.0), 0.0)

    def test_containers_are_rounded_recursively(self):
        value = {"a": [1.23456789, {"b": 2.3456789}], "c": 3}
        self.assertEqual(
            census.round_floats(value),
            {"a": [1.23457, {"b": 2.34568}], "c": 3},
        )


def prompt_logprobs_payload(prompt, scored=True):
    """A fabricated buffered prompt_logprobs=1 payload with the expected id-keyed entries."""
    entries = [None]
    for token in prompt[1:]:
        logprob = -0.5 if scored else None
        entries.append(
            {str(token): {"logprob": logprob, "rank": 1, "decoded_token": f"t{token}"}}
            if scored
            else {}
        )
    return {"choices": [{"text": "abc", "prompt_logprobs": entries}], "usage": {"completion_tokens": 1}}


class PromptLogprobsPayloadChecksTests(unittest.TestCase):
    def test_a_scored_payload_passes_every_check(self):
        prompt = [1, 2, 3]
        checks = census.prompt_logprobs_payload_checks(prompt_logprobs_payload(prompt), prompt)
        self.assertEqual(checks["positions"], 3)
        self.assertEqual(checks["scored_positions"], 2)
        self.assertTrue(checks["position_zero_null"])
        self.assertTrue(checks["every_position_scores_its_prompt_token"])

    def test_a_missing_own_token_breaks_correctness(self):
        prompt = [1, 2, 3]
        payload = prompt_logprobs_payload(prompt)
        payload["choices"][0]["prompt_logprobs"][1] = {"99": payload["choices"][0]["prompt_logprobs"][1]["2"]}
        checks = census.prompt_logprobs_payload_checks(payload, prompt)
        self.assertFalse(checks["every_position_scores_its_prompt_token"])
        self.assertTrue(checks["position_zero_null"])

    def test_a_missing_entry_breaks_correctness(self):
        prompt = [1, 2, 3]
        payload = prompt_logprobs_payload(prompt)
        payload["choices"][0]["prompt_logprobs"] = payload["choices"][0]["prompt_logprobs"][:2]
        checks = census.prompt_logprobs_payload_checks(payload, prompt)
        self.assertFalse(checks["every_position_scores_its_prompt_token"])

    def test_a_missing_prompt_logprobs_array_is_not_correct(self):
        checks = census.prompt_logprobs_payload_checks({"choices": [{"text": "abc"}]}, [1, 2, 3])
        self.assertFalse(checks["position_zero_null"])
        self.assertFalse(checks["every_position_scores_its_prompt_token"])


def rung(concurrency, ttft_p50, rate, requests=4):
    return {
        "concurrency": concurrency,
        "requests": requests,
        "ttft_s": {"p50": ttft_p50, "p95": ttft_p50 * 1.1, "max": ttft_p50 * 1.2},
        "end_to_end_s": {"p50": ttft_p50 * 2, "p95": ttft_p50 * 2.2, "max": ttft_p50 * 2.5},
        "aggregate_client_tokens_per_s": rate,
        "accepted": requests,
        "refused_503": 0,
        "failed": 0,
        "wall_s": 5.0,
        "server_reported_completion_tokens": requests * 8,
        "distinct_texts": 1,
        "all_streams_terminated": True,
    }


def probe(text, finish_reason="length", tokens=8):
    return {
        "status": 200,
        "text": text,
        "finish_reason": finish_reason,
        "completion_tokens": tokens,
    }


class CompareTests(unittest.TestCase):
    def base_mode(self, rungs, probes):
        return {
            "ladder": rungs,
            "fixed_probes": probes,
            "counters_before": {"sparkflash_requests_total": 0.0, "other_total": 1.0},
            "counters_after": {"sparkflash_requests_total": 3.0, "other_total": 9.0},
        }

    def test_ratios_compare_batched_against_serial(self):
        serial = self.base_mode(
            [rung(1, 0.032, 63.0), rung(8, 6.0, 74.0)],
            [probe("abc"), probe("def")],
        )
        batched = self.base_mode(
            [rung(1, 0.048, 52.0), rung(8, 0.36, 63.0)],
            [probe("abc"), probe("def")],
        )
        comparison = census.compare(serial, batched)
        self.assertEqual(len(comparison["rungs"]), 2)
        first, second = comparison["rungs"]
        self.assertAlmostEqual(first["ttft_p50_ratio_batched_over_serial"], 1.5)
        self.assertAlmostEqual(second["ttft_p50_ratio_batched_over_serial"], 0.06)
        self.assertAlmostEqual(
            second["aggregate_tokens_per_s_ratio_batched_over_serial"], 63.0 / 74.0
        )
        self.assertTrue(comparison["fixed_probe_texts_identical"])
        self.assertEqual(comparison["serial_requests_total_delta"], 3.0)
        self.assertEqual(comparison["batched_requests_total_delta"], 3.0)

    def test_different_text_breaks_identity(self):
        serial = self.base_mode([rung(1, 0.032, 63.0)], [probe("abc")])
        batched = self.base_mode([rung(1, 0.032, 63.0)], [probe("abd")])
        comparison = census.compare(serial, batched)
        self.assertFalse(comparison["fixed_probe_texts_identical"])
        self.assertFalse(comparison["fixed_probes"][0]["identical"])

    def test_a_different_finish_reason_is_not_identical(self):
        serial = self.base_mode([rung(1, 0.032, 63.0)], [probe("abc", "length")])
        batched = self.base_mode([rung(1, 0.032, 63.0)], [probe("abc", "stop")])
        self.assertFalse(census.compare(serial, batched)["fixed_probe_texts_identical"])


class MarkdownTests(unittest.TestCase):
    def report(self):
        serial = {
            "server_args": ["/bin/sparkflash", "serve"],
            "loadavg_start": {"loadavg_1_5_15": [1.0, 1.0, 1.0]},
            "loadavg_end": {"loadavg_1_5_15": [1.0, 1.0, 1.0]},
            "dspark_evidence_lines": 5,
            "panic_lines": [],
            "ladder": [rung(1, 0.032, 63.0)],
            "fixed_probes": [probe("abc")],
            "counters_before": {"sparkflash_requests_total": 0.0},
            "counters_after": {"sparkflash_requests_total": 5.0},
            "wire_probes": [
                wire_probe("buffered_prompt_logprobs_1", {"choices": [{"text": "abc"}]})
            ],
            "prompt_logprobs": {
                "name": "prompt_logprobs_one_pass",
                "label": census.LABEL,
                "prompt_tokens": 3,
                "runs": 2,
                "with_prompt_logprobs": {"n": 2, "p50": 0.02},
                "baseline": {"n": 2, "p50": 0.01},
                "overhead_p50_ms": 10.0,
                "statuses": [200],
                "positions": 3,
                "scored_positions": 2,
                "position_zero_null": True,
                "every_position_scores_its_prompt_token": True,
                "stable_across_runs": True,
                "correct": True,
                "token_requests": 2,
                "completion_tokens": 2,
                "payload": prompt_logprobs_payload([1, 2, 3]),
                "shape": census.payload_shape(prompt_logprobs_payload([1, 2, 3])),
                "sample_scored_position": prompt_logprobs_payload([1, 2, 3])["choices"][0][
                    "prompt_logprobs"
                ][1],
            },
            "ops_accounting": {
                "expected_generation_tokens": 8,
                "observed_generation_tokens_delta": 8.0,
                "expected_ttft_samples": 1,
                "observed_ttft_count_delta": 1.0,
                "idle_running": 0.0,
                "idle_waiting": 0.0,
                "ok": True,
            },
            "ops_families_present": list(census.OPS_FAMILIES),
            "ops_metrics_after": {
                "ops_family_help": {family: "help" for family in census.OPS_FAMILIES},
                "ops_family_type": {family: "gauge" for family in census.OPS_FAMILIES},
                "ttft_buckets": [
                    {"le": "0.001", "count": 0.0},
                    {"le": "+Inf", "count": 1.0},
                ],
            },
        }
        batched = json.loads(json.dumps(serial))
        batched["server_args"] = ["/bin/sparkflash", "serve", "--continuous-batching"]
        return {
            "label": census.LABEL,
            "generated_at": "2026-10-21T00:00:00Z",
            "repo": {"commit": "abc", "branch": "task/x"},
            "binary": {"sha256": "0" * 64},
            "method": {
                "modes": {
                    "serial": [],
                    "batched": ["--continuous-batching", "--max-batch", "32"],
                }
            },
            "serial": serial,
            "batched": batched,
            "comparison": census.compare(serial, batched),
            "limitations": ["host path"],
            "artefacts": {
                "json": "/tmp/a.json",
                "markdown": "/tmp/a.md",
                "sha256": "/tmp/a.sha256",
                "serial_log": "/tmp/s.log",
                "batched_log": "/tmp/b.log",
            },
        }

    def test_markdown_contains_both_modes_and_the_label(self):
        report = self.report()
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "census.md"
            census.write_markdown(report, path)
            text = path.read_text()
        self.assertIn(census.LABEL, text)
        self.assertIn("## Prompt logprobs one-pass probe", text)
        self.assertIn("One-pass prompt logprobs correct in both modes", text)
        self.assertIn("serial mode", text)
        self.assertIn("batched mode", text)
        self.assertIn("Serial vs batched, per concurrency", text)
        self.assertIn("All fixed probes identical", text)
        self.assertIn("## Ops metrics", text)
        self.assertIn("All four ops families present in both modes", text)
        self.assertIn("## Wire probes across modes", text)
        self.assertIn("All wire probes identical", text)
        self.assertIn("/tmp/a.json", text)



METRICS_FIXTURE = """# HELP sparkflash_generation_tokens_total Total generated token ids (vLLM vllm:generation_tokens_total semantics)
# TYPE sparkflash_generation_tokens_total counter
sparkflash_generation_tokens_total 7
# HELP sparkflash_num_requests_running Number of requests currently executing
# TYPE sparkflash_num_requests_running gauge
sparkflash_num_requests_running 0
# HELP sparkflash_num_requests_waiting Number of requests accepted by the driver and queued
# TYPE sparkflash_num_requests_waiting gauge
sparkflash_num_requests_waiting 0
# HELP sparkflash_time_to_first_token_seconds Seconds from engine-queue admission
# TYPE sparkflash_time_to_first_token_seconds histogram
sparkflash_time_to_first_token_seconds_bucket{le="0.001"} 0
sparkflash_time_to_first_token_seconds_bucket{le="0.02"} 1
sparkflash_time_to_first_token_seconds_bucket{le="+Inf"} 2
sparkflash_time_to_first_token_seconds_sum 0.10701966
sparkflash_time_to_first_token_seconds_count 2
# HELP sparkflash_requests_total Total inference requests served
# TYPE sparkflash_requests_total counter
sparkflash_requests_total 2
# HELP sparkflash_uma_free_gib Free coherent UMA memory per rank in GiB
# TYPE sparkflash_uma_free_gib gauge
sparkflash_uma_free_gib{rank="0"} 0
"""


class OpsMetricsParsingTests(unittest.TestCase):
    def test_ops_families_help_and_samples_are_parsed(self):
        parsed = census.parse_ops_metrics(METRICS_FIXTURE)
        for family in census.OPS_FAMILIES:
            self.assertIn(family, parsed["present_families"])
            self.assertTrue(parsed["ops_family_help"][family])
        self.assertEqual(
            parsed["ops_family_type"]["sparkflash_time_to_first_token_seconds"], "histogram"
        )
        self.assertEqual(parsed["sample_values"]["sparkflash_generation_tokens_total"], 7.0)
        self.assertEqual(parsed["sample_values"]["sparkflash_requests_total"], 2.0)
        self.assertEqual(
            parsed["sample_values"]["sparkflash_time_to_first_token_seconds_count"], 2.0
        )

    def test_ttft_buckets_keep_their_labels_and_numeric_order(self):
        parsed = census.parse_ops_metrics(METRICS_FIXTURE)
        buckets = parsed["ttft_buckets"]
        self.assertEqual([bucket["le"] for bucket in buckets], ["0.001", "0.02", "+Inf"])
        self.assertEqual([bucket["count"] for bucket in buckets], [0.0, 1.0, 2.0])

    def test_missing_family_is_visible(self):
        parsed = census.parse_ops_metrics("sparkflash_requests_total 1\n")
        self.assertNotIn("sparkflash_num_requests_running", parsed["present_families"])
        self.assertIsNone(parsed["ops_family_help"]["sparkflash_num_requests_running"])


class PayloadHelpersTests(unittest.TestCase):
    def test_volatile_fields_are_dropped_deep(self):
        payload = {
            "id": "cmpl-0",
            "created": 123,
            "choices": [
                {
                    "text": "abc",
                    "logprobs": {"tokens": ["abc"]},
                }
            ],
            "x_deepseek_spark_stats": {"timing": {"ttft_ms": 1.0}, "kv_cache": {"hits": 0}},
        }
        stripped = census.strip_volatile(payload)
        self.assertNotIn("id", stripped)
        self.assertNotIn("created", stripped)
        self.assertNotIn("timing", stripped["x_deepseek_spark_stats"])
        self.assertEqual(stripped["choices"][0]["text"], "abc")

    def test_shape_is_key_and_type_skeleton(self):
        left = {"a": 1.0, "b": [None, {"c": "x"}], "d": True}
        right = {"a": 2.0, "b": [None, {"c": "y"}], "d": False}
        self.assertEqual(census.payload_shape(left), census.payload_shape(right))
        different = {"a": 1.0, "b": [None], "d": True}
        self.assertNotEqual(census.payload_shape(left), census.payload_shape(different))


def wire_probe(name, payload, status=200, kind="buffered", tokens=3, done=None):
    entry = {
        "name": name,
        "kind": kind,
        "status": status,
        "content_type": "application/json",
        "payload": payload,
        "shape": census.payload_shape(payload),
        "done": done,
        "completion_tokens": tokens,
        "serial_status": status,
        "batched_status": status,
    }
    return entry


class WireProbeCompareTests(unittest.TestCase):
    def prompt_logprobs(self, payload=None):
        prompt = [1, 2, 3]
        payload = payload if payload is not None else prompt_logprobs_payload(prompt)
        checks = census.prompt_logprobs_payload_checks(payload, prompt)
        return {
            "correct": checks["position_zero_null"]
            and checks["every_position_scores_its_prompt_token"],
            **checks,
            "stable_across_runs": True,
            "payload": payload,
            "shape": census.payload_shape(payload),
            "baseline": {"p50": 0.01},
            "with_prompt_logprobs": {"p50": 0.02},
            "overhead_p50_ms": 10.0,
        }

    def mode_with(self, probes, accounting_ok=True, prompt_logprobs=None):
        return {
            "ladder": [rung(1, 0.032, 63.0)],
            "fixed_probes": [probe("abc")],
            "counters_before": {"sparkflash_requests_total": 0.0},
            "counters_after": {"sparkflash_requests_total": 1.0},
            "wire_probes": probes,
            "prompt_logprobs": prompt_logprobs
            if prompt_logprobs is not None
            else self.prompt_logprobs(),
            "ops_accounting": {"ok": accounting_ok},
            "ops_families_present": list(census.OPS_FAMILIES),
            "ops_metrics_after": {
                "ops_family_help": {family: "help" for family in census.OPS_FAMILIES},
                "ttft_buckets": [{"le": "0.001", "count": 0.0}],
            },
        }

    def test_identical_wire_payloads_compare_equal(self):
        payload = {"choices": [{"text": "abc", "prompt_logprobs": [None]}]}
        serial = self.mode_with([wire_probe("probe", payload)])
        batched = self.mode_with([wire_probe("probe", json.loads(json.dumps(payload)))])
        comparison = census.compare(serial, batched)
        self.assertTrue(comparison["wire_probes_identical"])
        self.assertTrue(comparison["wire_probes"][0]["payload_identical"])
        self.assertTrue(comparison["ops_metrics"]["accounting_ok"])

    def test_a_different_wire_value_breaks_identity(self):
        serial = self.mode_with([wire_probe("probe", {"text": "abc"})])
        batched = self.mode_with([wire_probe("probe", {"text": "abd"})])
        comparison = census.compare(serial, batched)
        self.assertFalse(comparison["wire_probes_identical"])
        self.assertFalse(comparison["wire_probes"][0]["identical"])
        self.assertTrue(comparison["wire_probes"][0]["shape_identical"])
        self.assertFalse(comparison["wire_probes"][0]["payload_identical"])

    def test_identical_prompt_logprobs_compare_ok(self):
        comparison = census.compare(self.mode_with([]), self.mode_with([]))
        self.assertTrue(comparison["prompt_logprobs"]["ok"])
        self.assertTrue(comparison["prompt_logprobs"]["payload_identical"])
        self.assertTrue(comparison["prompt_logprobs"]["shape_identical"])
        self.assertIn("timing", comparison["prompt_logprobs"])

    def test_a_different_prompt_logprobs_value_breaks_ok(self):
        serial = self.mode_with([])
        payload = prompt_logprobs_payload([1, 2, 3])
        payload["choices"][0]["prompt_logprobs"][1]["2"]["logprob"] = -1.25
        batched = self.mode_with([], prompt_logprobs=self.prompt_logprobs(payload))
        comparison = census.compare(serial, batched)
        self.assertFalse(comparison["prompt_logprobs"]["payload_identical"])
        self.assertFalse(comparison["prompt_logprobs"]["ok"])
        self.assertTrue(comparison["prompt_logprobs"]["shape_identical"])

    def test_an_incorrect_prompt_logprobs_payload_breaks_ok(self):
        payload = prompt_logprobs_payload([1, 2, 3], scored=False)
        comparison = census.compare(
            self.mode_with([], prompt_logprobs=self.prompt_logprobs(payload)),
            self.mode_with([]),
        )
        self.assertFalse(comparison["prompt_logprobs"]["serial_correct"])
        self.assertFalse(comparison["prompt_logprobs"]["ok"])

    def test_failed_ops_accounting_is_visible(self):
        serial = self.mode_with([], accounting_ok=False)
        batched = self.mode_with([], accounting_ok=True)
        comparison = census.compare(serial, batched)
        self.assertFalse(comparison["ops_metrics"]["accounting_ok"])

if __name__ == "__main__":
    unittest.main()
