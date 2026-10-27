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

"""Unit tests for scripts/d41_recapture.py.

Host-only: no server, no GPU, no network. Every test drives the capture through a synthetic HTTP
transport (FakeTransport), so idleness classification, refusals and the busy tag are exercised
without touching a vLLM instance. Run with:

    python3 -m unittest discover -s scripts/tests -v
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import d41_recapture as capture

EXPECT = capture.DEFAULT_EXPECT_ROOT
SERVED_ROOT = "/root/.cache/huggingface/hub/models--deepseek-v4.1-flash/snapshots/" + EXPECT
MODEL = "deepseek-v4.1-flash"


def models_body(root=SERVED_ROOT, model=MODEL):
    return {"data": [{"id": model, "root": root}]}


def metrics_text(running=0, waiting=0):
    return (
        "# HELP vllm:num_requests_running Number of requests running.\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{model_name="%s"} %d\n'
        "# HELP vllm:num_requests_waiting Number of requests waiting.\n"
        'vllm:num_requests_waiting{model_name="%s"} %d\n'
        "# HELP vllm:request_success_total Count of successfully processed requests.\n"
        'vllm:request_success_total{model_name="%s"} 17.0\n'
    ) % (MODEL, running, MODEL, waiting, MODEL)


def completion_body(prompt, scale=1.0):
    prompt_logprobs = [None]
    for index in range(1, len(prompt)):
        token = 1000 + index
        prompt_logprobs.append({str(token): {"logprob": -0.01 * index * scale, "rank": 1}})
    return {"choices": [{"prompt_logprobs": prompt_logprobs, "text": ""}], "id": "cmpl-x"}


class FakeTransport:
    """Synthetic server. Records every call so tests can assert what was (not) sent."""

    def __init__(self, served_root=SERVED_ROOT, metrics_script=None, completions_script=None,
                 metrics_raises=None, background_sets_load=True, background_sleep=0.05,
                 probe_sleep=0.0, realistic_gauge=False, model=MODEL):
        self.served_root = served_root
        self.model = model
        self.metrics_script = list(metrics_script or [])
        self.completions_script = list(completions_script or [])
        self.metrics_raises = metrics_raises
        self.background_sets_load = background_sets_load
        self.background_sleep = background_sleep
        self.probe_sleep = probe_sleep
        #: lag-faithful gauge: running = background request (1) + probe request (1 while in flight);
        #: the probe's contribution is decremented BEFORE the response is returned, so a snapshot
        #: taken after the response can only see the background generation (the C1 race).
        self.realistic_gauge = realistic_gauge
        self.load_active = threading.Event()
        self.background_active = threading.Event()
        self.probe_active = threading.Event()
        self.calls = []
        self.lock = threading.Lock()

    def get_text(self, path):
        with self.lock:
            self.calls.append(("GET", path))
        if path == "/v1/models":
            body = json.dumps(models_body(self.served_root, self.model)).encode()
            return body, body.decode()
        if path == "/metrics":
            if self.metrics_raises:
                raise capture.RecaptureError(self.metrics_raises)
            with self.lock:
                if self.metrics_script:
                    text = self.metrics_script.pop(0) if len(self.metrics_script) > 1 else self.metrics_script[0]
                elif self.realistic_gauge:
                    running = (1 if self.background_active.is_set() else 0) + (
                        1 if self.probe_active.is_set() else 0
                    )
                    text = metrics_text(running, 0)
                else:
                    text = metrics_text(2 if self.load_active.is_set() else 0)
            return text.encode(), text
        raise AssertionError("unexpected GET " + path)

    def post_json(self, path, payload):
        with self.lock:
            self.calls.append(("POST", path, payload))
        if payload.get("max_tokens", 1) > 1:
            if self.background_sets_load:
                self.load_active.set()
                self.background_active.set()
            try:
                time.sleep(self.background_sleep)
            finally:
                self.background_active.clear()
                self.load_active.clear()
            body = {"choices": [{"text": "background", "finish_reason": "length"}]}
            return json.dumps(body).encode(), body
        with self.lock:
            item = self.completions_script.pop(0) if self.completions_script else None
        if isinstance(item, Exception):
            raise item
        scale = 1.0
        if isinstance(item, dict) and "scale" in item:
            scale = float(item["scale"])
        if self.probe_sleep > 0:
            self.probe_active.set()
            try:
                time.sleep(self.probe_sleep)
            finally:
                self.probe_active.clear()
        body = completion_body(payload["prompt"], scale)
        return json.dumps(body).encode(), body

    def posts(self):
        with self.lock:
            return [call for call in self.calls if call[0] == "POST"]


def plan_path(tmpdir, blocks, recipe=None):
    plan = {
        "schema": capture.PLAN_SCHEMA,
        "description": "unit-test plan",
        "recipe": recipe or {
            "max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
            "ignore_eos": True, "prompt_logprobs": 1000,
        },
        "provenance": {"fixture": {"path": "fixture.json", "sha256": "0" * 64}},
        "blocks": blocks,
    }
    path = os.path.join(tmpdir, "plan.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle)
    return path


def run(fake, plan, out, extra=()):
    argv = ["--plan", plan, "--out", out, "--settle-seconds", "0", *extra]
    return capture.run_cli(argv, lambda base_url, timeout: fake)


def read_capture(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class MetricsParseTests(unittest.TestCase):
    def test_parses_labeled_gauges(self):
        metrics = capture.parse_metrics(metrics_text(3, 7))
        gauges = capture.request_gauges(metrics, capture.DEFAULT_RUNNING_METRIC, capture.DEFAULT_WAITING_METRIC)
        self.assertEqual(gauges, {"num_requests_running": 3, "num_requests_waiting": 7})
        self.assertEqual(len(metrics["sha256"]), 64)

    def test_missing_running_metric_refuses(self):
        metrics = capture.parse_metrics("vllm:num_requests_waiting 0\n")
        with self.assertRaises(capture.RecaptureError):
            capture.request_gauges(metrics, capture.DEFAULT_RUNNING_METRIC, capture.DEFAULT_WAITING_METRIC)

    def test_idle_rule(self):
        self.assertTrue(capture.is_idle({"num_requests_running": 0, "num_requests_waiting": 0}))
        self.assertFalse(capture.is_idle({"num_requests_running": 1, "num_requests_waiting": 0}))
        self.assertFalse(capture.is_idle({"num_requests_running": 0, "num_requests_waiting": 1}))


class ServedRootTests(unittest.TestCase):
    def test_root_mismatch_refuses_before_any_completion(self):
        fake = FakeTransport(served_root="/root/.cache/snapshots/deadbeef")
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "repeat", "class": "repeat",
                                       "requests": [{"id": "a", "prompt": list(range(8))},
                                                    {"id": "b", "prompt": list(range(8))}]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out), 2)
            self.assertEqual(fake.posts(), [])
            self.assertFalse(os.path.exists(out))

    def test_root_matches_suffix(self):
        self.assertTrue(capture.root_matches(SERVED_ROOT, EXPECT))
        self.assertFalse(capture.root_matches("/x/deadbeef", EXPECT))
        self.assertTrue(capture.root_matches(None, None))


class MetricsRefusalTests(unittest.TestCase):
    def test_missing_metrics_endpoint_refuses_before_completion(self):
        fake = FakeTransport(metrics_raises="HTTP 404 from /metrics: not found")
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "repeat", "class": "repeat",
                                       "requests": [{"id": "a", "prompt": list(range(8))},
                                                    {"id": "b", "prompt": list(range(8))}]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out), 2)
            self.assertEqual(fake.posts(), [])
            self.assertFalse(os.path.exists(out))

    def test_busy_baseline_refuses_before_completion(self):
        fake = FakeTransport(metrics_script=[metrics_text(1, 0)])
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "repeat", "class": "repeat",
                                       "requests": [{"id": "a", "prompt": list(range(8))},
                                                    {"id": "b", "prompt": list(range(8))}]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out), 2)
            self.assertEqual(fake.posts(), [])
            self.assertFalse(os.path.exists(out))

    def test_snapshot_mismatch_before_first_completion_refuses(self):
        fake = FakeTransport(metrics_script=[metrics_text(0, 0), metrics_text(0, 0), metrics_text(1, 0)])
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "repeat", "class": "repeat",
                                       "requests": [{"id": "a", "prompt": list(range(8))},
                                                    {"id": "b", "prompt": list(range(8))}]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out), 2)
            self.assertEqual(fake.posts(), [])
            self.assertFalse(os.path.exists(out))


class HappyPathTests(unittest.TestCase):
    def make_plan(self, tmpdir):
        return plan_path(tmpdir, [
            {"id": "idle-A-repeat", "class": "repeat",
             "requests": [{"id": "rep-0", "prompt": list(range(10))},
                          {"id": "rep-1", "prompt": list(range(10))},
                          {"id": "rep-2", "prompt": list(range(10))}],
             "pairs": [["rep-0", "rep-1"], ["rep-1", "rep-2"]]},
            {"id": "idle-B-distractor", "class": "idle_distractor",
             "requests": [{"id": "d-130", "prompt": list(range(8))},
                          {"id": "d-131", "prompt": list(range(9))}],
             "pairs": [["d-130", "d-131"]]},
        ])

    def test_idle_capture_writes_valid_evidence(self):
        fake = FakeTransport()
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = self.make_plan(tmpdir)
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out, ["--tag", "unit"]), 0)
            self.assertTrue(os.path.exists(out))
            data = read_capture(out)
            self.assertEqual(data["schema"], capture.CAPTURE_SCHEMA)
            self.assertEqual(data["capture_id"], "unit")
            self.assertEqual(data["producer"]["script_sha256"], capture.sha256_file(capture.__file__))
            self.assertEqual(data["served"]["root"], SERVED_ROOT)
            self.assertTrue(data["root_verified"])
            self.assertEqual(data["expect_root"], EXPECT)
            self.assertEqual(data["recipe"]["prompt_logprobs"], 1000)
            self.assertEqual(data["suite"]["sha256"], capture.sha256_file(plan))
            self.assertEqual(data["status"], "ok")
            self.assertEqual(len(data["blocks"]), 2)
            repeat, distractor = data["blocks"]
            self.assertEqual(repeat["class"], "repeat")
            self.assertEqual(repeat["idleness"]["idle_verdict"], "idle")
            self.assertEqual(repeat["prompt_len"], 10)
            self.assertEqual(repeat["idleness"]["metrics_before"]["num_requests_running"], 0)
            self.assertEqual(repeat["idleness"]["metrics_after"]["num_requests_waiting"], 0)
            self.assertFalse(repeat["idleness"]["observed_other_traffic"])
            self.assertEqual(repeat["idleness"]["external_peers"], [])
            self.assertEqual(len(repeat["requests"]), 3)
            self.assertEqual(repeat["requests"][1]["repeat_of"], "rep-0")
            record = repeat["requests"][0]
            for field in ("prompt_len", "batch_size", "prompt_sha256", "request_sha256",
                          "response_sha256", "wall_s", "positions"):
                self.assertIn(field, record)
            self.assertIsInstance(record["wall_s"], float)
            self.assertNotIn("logprobs", record["positions"][0])
            top = record["positions"][0]["top"]
            self.assertEqual(len(top), 1)
            self.assertEqual(len(top[0]), 2)
            self.assertGreaterEqual(top[0][1], top[-1][1])
            self.assertEqual(distractor["class"], "idle_distractor")
            self.assertIsNone(distractor["prompt_len"])
            self.assertEqual(distractor["batch_size"], 1)
            self.assertEqual(distractor["pairs"], [["d-130", "d-131"]])
            for snapshot in repeat["idleness"]["metrics_during"]:
                self.assertIn("at_utc", snapshot)
                self.assertEqual(snapshot["num_requests_running"], 0)
            self.assertEqual(len(fake.posts()), 5)
            self.assertEqual(data["warnings"], [])

    def test_compact_output_is_valid_and_smaller(self):
        fake = FakeTransport()
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = self.make_plan(tmpdir)
            pretty = os.path.join(tmpdir, "pretty.json")
            compact = os.path.join(tmpdir, "compact.json")
            self.assertEqual(run(fake, plan, pretty), 0)
            self.assertEqual(run(FakeTransport(), plan, compact, ["--compact"]), 0)
            with open(pretty, "rb") as handle:
                pretty_bytes = handle.read()
            with open(compact, "rb") as handle:
                compact_bytes = handle.read()
            self.assertLess(len(compact_bytes), len(pretty_bytes))
            self.assertEqual(read_capture(compact)["schema"], capture.CAPTURE_SCHEMA)

    def test_dry_run_makes_no_requests(self):
        def factory(base_url, timeout):
            raise AssertionError("dry-run must not construct a transport")
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = self.make_plan(tmpdir)
            self.assertEqual(
                capture.run_cli(["--plan", plan, "--dry-run"], factory), 0
            )


class RepeatDeterminismTests(unittest.TestCase):
    def test_repeat_mismatch_is_recorded_not_hidden(self):
        fake = FakeTransport(completions_script=[{"scale": 1.0}, {"scale": 2.0}, {"scale": 1.0}])
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "repeat", "class": "repeat",
                                       "requests": [{"id": "a", "prompt": list(range(8))},
                                                    {"id": "b", "prompt": list(range(8))},
                                                    {"id": "c", "prompt": list(range(8))}]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out), 4)
            data = read_capture(out)
            self.assertEqual(data["status"], "unstable")
            self.assertTrue(any("bit-identical" in warning for warning in data["warnings"]))


class BusyBlockTests(unittest.TestCase):
    def busy_plan(self, tmpdir):
        return plan_path(tmpdir, [
            {"id": "busy", "class": "busy_distractor",
             "requests": [{"id": "p8", "prompt": list(range(8))},
                          {"id": "p9", "prompt": list(range(9))}],
             "pairs": [["p8", "p9"]],
             "background": {"prompt": list(range(8)), "max_tokens": 8}},
        ])

    def test_busy_block_records_co_tenancy_and_is_tagged(self):
        fake = FakeTransport()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, self.busy_plan(tmpdir), out), 0)
            data = read_capture(out)
            self.assertEqual(data["status"], "mixed")
            block = data["blocks"][0]
            self.assertEqual(block["class"], "busy_distractor")
            self.assertEqual(block["idleness"]["idle_verdict"], "busy")
            self.assertTrue(block["idleness"]["observed_other_traffic"])
            self.assertTrue(any(snapshot["num_requests_running"] >= 2
                                for snapshot in block["idleness"]["metrics_during"]))
            self.assertIn("load_generator", block)
            self.assertIsNotNone(block["load_generator"]["response_sha256"])
            self.assertIsNone(block["prompt_len"])
            self.assertEqual(len(fake.posts()), 3)  # background + two probes

    def test_busy_block_without_evidence_refuses(self):
        fake = FakeTransport(background_sets_load=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, self.busy_plan(tmpdir), out, ["--background-start-wait", "0.2"]), 2)
            self.assertFalse(os.path.exists(out))

    def test_busy_evidence_is_sampled_while_a_probe_is_in_flight(self):
        # Lag-faithful gauge: the probe contribution disappears BEFORE its response is returned, so
        # a snapshot taken after the response can only see the background generation (the C1 race).
        # Only the concurrent sampler can catch running == 2.
        fake = FakeTransport(realistic_gauge=True, background_sleep=0.5, probe_sleep=0.05)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "capture.json")
            extra = ["--sample-interval", "0.002", "--background-start-wait", "2"]
            self.assertEqual(run(fake, self.busy_plan(tmpdir), out, extra), 0)
            data = read_capture(out)
            block = data["blocks"][0]
            self.assertEqual(block["class"], "busy_distractor")
            running = [snapshot["num_requests_running"] for snapshot in block["idleness"]["metrics_during"]]
            self.assertIn(2, running)  # probe in flight next to the background generation
            self.assertIn(1, running)  # and the after-response snapshots already show only the background
            self.assertTrue(block["idleness"]["observed_other_traffic"])
            self.assertEqual(data["status"], "mixed")

    def test_busy_block_refuses_when_the_sampler_never_sees_the_probe_window(self):
        # A sampler interval longer than the whole probe sequence misses the in-flight window; the
        # capture must refuse (exit 2, nothing written) instead of writing a busy class the floor
        # script would reject.
        fake = FakeTransport(realistic_gauge=True, background_sleep=0.4, probe_sleep=0.02)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "capture.json")
            extra = ["--sample-interval", "10", "--background-start-wait", "2"]
            self.assertEqual(run(fake, self.busy_plan(tmpdir), out, extra), 2)
            self.assertFalse(os.path.exists(out))


class ReclassificationTests(unittest.TestCase):
    def test_idle_block_that_sees_traffic_is_tagged_busy(self):
        fake = FakeTransport(metrics_script=[
            metrics_text(0, 0), metrics_text(0, 0), metrics_text(0, 0),
            metrics_text(2, 0), metrics_text(2, 0), metrics_text(2, 0),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "idle-B", "class": "idle_distractor",
                                       "requests": [{"id": "a", "prompt": list(range(8))},
                                                    {"id": "b", "prompt": list(range(9))}],
                                       "pairs": [["a", "b"]]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out), 0)
            data = read_capture(out)
            block = data["blocks"][0]
            self.assertEqual(block["class"], "busy_distractor")
            self.assertEqual(block["idleness"]["idle_verdict"], "busy")
            self.assertTrue(block["idleness"]["observed_other_traffic"])
            self.assertEqual(data["status"], "mixed")

    def test_idle_block_that_sees_only_one_running_request_is_refused(self):
        # running == 1 may be the capture's own in-flight request, so it is not co-tenancy
        # evidence. The floor script would reject a busy class without running>1 or waiting>0;
        # the capture refuses at capture time (C2) instead of writing a block that cannot be used.
        fake = FakeTransport(metrics_script=[
            metrics_text(0, 0), metrics_text(0, 0), metrics_text(0, 0),
            metrics_text(1, 0), metrics_text(1, 0), metrics_text(1, 0),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "idle-B", "class": "idle_distractor",
                                       "requests": [{"id": "a", "prompt": list(range(8))},
                                                    {"id": "b", "prompt": list(range(9))}],
                                       "pairs": [["a", "b"]]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out), 2)
            self.assertFalse(os.path.exists(out))
            with open(out + ".log", "r", encoding="utf-8") as handle:
                log = handle.read()
            self.assertIn("no co-tenancy evidence", log)


class FailureTests(unittest.TestCase):
    def test_request_failure_aborts_and_writes_nothing(self):
        fake = FakeTransport(completions_script=[capture.RecaptureError("HTTP 500 from /v1/completions")])
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "repeat", "class": "repeat",
                                       "requests": [{"id": "a", "prompt": list(range(8))},
                                                    {"id": "b", "prompt": list(range(8))}]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out), 2)
            self.assertFalse(os.path.exists(out))
            self.assertTrue(os.path.exists(out + ".log"))
            with open(out + ".log", "r", encoding="utf-8") as handle:
                self.assertIn("REFUSED", handle.read())

    def test_sampler_thread_stops_when_a_busy_request_fails(self):
        fake = FakeTransport(completions_script=[capture.RecaptureError("HTTP 500 from /v1/completions")])
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "busy", "class": "busy_distractor",
                                       "requests": [{"id": "p8", "prompt": list(range(8))},
                                                    {"id": "p9", "prompt": list(range(9))}],
                                       "pairs": [["p8", "p9"]],
                                       "background": {"prompt": list(range(8)), "max_tokens": 8}}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(fake, plan, out, ["--background-start-wait", "2"]), 2)
            self.assertFalse(os.path.exists(out))
            calls_after_failure = len(fake.calls)
            time.sleep(0.05)
            self.assertEqual(len(fake.calls), calls_after_failure)


class IdleFallbackPlanTests(unittest.TestCase):
    """The idle-only fallback plan must be the primary plan minus the busy block, verbatim."""

    def plans(self):
        primary_path = os.path.join(SCRIPTS, "d41_capture_plan.json")
        idle_path = os.path.join(SCRIPTS, "d41_capture_plan_idle.json")
        with open(primary_path, "r", encoding="utf-8") as handle:
            primary = json.load(handle)
        with open(idle_path, "r", encoding="utf-8") as handle:
            idle = json.load(handle)
        return primary, idle, idle_path

    def test_idle_plan_is_the_primary_plan_minus_the_busy_block(self):
        primary, idle, _ = self.plans()
        self.assertEqual([block["class"] for block in primary["blocks"]],
                         ["repeat", "idle_distractor", "busy_distractor"])
        # Verbatim copy: ids, classes, requests, prompt arrays, order and pairs are identical.
        self.assertEqual(idle["blocks"], primary["blocks"][:2])
        self.assertEqual(idle["schema"], primary["schema"])
        self.assertEqual(idle["recipe"], primary["recipe"])
        self.assertEqual(idle["provenance"], primary["provenance"])
        self.assertEqual(sum(len(block["requests"]) for block in idle["blocks"]), 9)
        self.assertEqual([len(block["requests"]) for block in idle["blocks"]], [4, 5])
        self.assertNotIn("busy_distractor", {block["class"] for block in idle["blocks"]})

    def test_idle_plan_dry_runs_without_a_transport(self):
        _, _, idle_path = self.plans()

        def factory(base_url, timeout):
            raise AssertionError("--dry-run must not construct a transport")

        self.assertEqual(capture.run_cli(["--plan", idle_path, "--dry-run"], factory), 0)


class PlanValidationTests(unittest.TestCase):
    def assert_refuses(self, plan, out):
        fake = FakeTransport()
        self.assertEqual(run(fake, plan_path_for(plan), out), 2)
        self.assertFalse(os.path.exists(out))
        self.assertEqual(fake.posts(), [])

    def test_unknown_schema_refuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            blocks = [{"id": "r", "class": "repeat",
                       "requests": [{"id": "a", "prompt": [1, 2]}, {"id": "b", "prompt": [1, 2]}]}]
            plan = plan_path(tmpdir, blocks)
            with open(plan, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            data["schema"] = "something-else/1"
            with open(plan, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(FakeTransport(), plan, out), 2)
            self.assertFalse(os.path.exists(out))

    def test_repeat_block_with_different_bodies_refuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "r", "class": "repeat",
                                       "requests": [{"id": "a", "prompt": [1, 2]},
                                                    {"id": "b", "prompt": [1, 3]}]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(FakeTransport(), plan, out), 2)
            self.assertFalse(os.path.exists(out))

    def test_pair_with_unknown_member_refuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "d", "class": "idle_distractor",
                                       "requests": [{"id": "a", "prompt": [1, 2]},
                                                    {"id": "b", "prompt": [1, 2, 3]}],
                                       "pairs": [["a", "nope"]]}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(FakeTransport(), plan, out), 2)
            self.assertFalse(os.path.exists(out))

    def test_busy_block_without_background_refuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "b", "class": "busy_distractor",
                                       "requests": [{"id": "a", "prompt": [1, 2]}],
                                       "pairs": []}])
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(FakeTransport(), plan, out), 2)
            self.assertFalse(os.path.exists(out))

    def test_prompt_logprobs_below_cap_refuses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = plan_path(tmpdir, [{"id": "r", "class": "repeat",
                                       "requests": [{"id": "a", "prompt": [1, 2]},
                                                    {"id": "b", "prompt": [1, 2]}]}],
                             recipe={"max_tokens": 1, "temperature": 0.0, "top_p": 1.0,
                                     "ignore_eos": True, "prompt_logprobs": 100})
            out = os.path.join(tmpdir, "capture.json")
            self.assertEqual(run(FakeTransport(), plan, out), 2)
            self.assertFalse(os.path.exists(out))


class ShapeAndDistributionTests(unittest.TestCase):
    def test_block_prompt_len_int_or_null(self):
        self.assertEqual(capture.block_prompt_len([{"prompt_len": 139}, {"prompt_len": 139}]), 139)
        self.assertIsNone(capture.block_prompt_len([{"prompt_len": 130}, {"prompt_len": 131}]))

    def test_same_distribution_exact(self):
        left = {"positions": [{"position": 0, "top": [[1, -0.5]]}]}
        right = {"positions": [{"position": 0, "top": [[1, -0.5]]}]}
        other = {"positions": [{"position": 0, "top": [[1, -0.5000001]]}]}
        self.assertTrue(capture.same_distribution(left, right))
        self.assertFalse(capture.same_distribution(left, other))


def plan_path_for(plan_dict):
    """Helper for tests that build a plan dict inline: write it next to a temp dir."""
    directory = tempfile.mkdtemp(prefix="d41-plan-")
    path = os.path.join(directory, "plan.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan_dict, handle)
    return path


if __name__ == "__main__":
    unittest.main()
