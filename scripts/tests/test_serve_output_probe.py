#!/usr/bin/env python3
"""Unit tests for scripts/serve_output_probe.py.

No external network: a local fake HTTP server answers /v1/models and
/v1/completions (buffered, SSE and error cases). Run with:

    python3 -m unittest discover -s scripts/tests -v
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import serve_output_probe as probe

MODELS = {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _route(self, body):
        route = self.server.routes.get(self.path)
        if route is None:
            self._respond(404, {"Content-Type": "application/json"}, b'{"error":"no route"}')
            return
        status, headers, payload = route(body, self)
        self._respond(status, headers, payload)

    def _respond(self, status, headers, payload):
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        if isinstance(payload, list):
            self.send_header("Connection", "close")
            self.end_headers()
            for delay, chunk in payload:
                if delay:
                    time.sleep(delay)
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            return
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._route(None)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self._route(self.rfile.read(length))


class FakeServer:
    def __init__(self, routes):
        self.routes = routes
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.routes = routes
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def json_route(document, status=200):
    payload = json.dumps(document).encode("utf-8")
    return lambda body, handler: (status, {"Content-Type": "application/json"}, payload)


class ServeOutputProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.requests = []

    def start(self, routes):
        server = FakeServer(routes)
        self.addCleanup(server.close)
        return server

    def run_probe(self, base_url, *extra):
        out_json = os.path.join(self.tmp, "out.json")
        out_md = os.path.join(self.tmp, "out.md")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = probe.main(
                ["--base-url", base_url, "--out-json", out_json, "--out-md", out_md, *extra]
            )
        return code, out_json, out_md, stdout.getvalue(), stderr.getvalue()

    def buffered_routes(self, text="<|end_of_sentence|>.\n", prompt_tokens=7, completion_tokens=4, cache=None):
        def completions(body, handler):
            payload = json.loads(body)
            self.requests.append(payload)
            document = {
                "id": "cmpl-1",
                "object": "text_completion",
                "model": "fake-model",
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "x_deepseek_spark_stats": {
                    "kv_cache_hit_rate": 0.0 if cache is None else cache,
                    "engram_layers": [
                        {"layer_id": 1, "hit_rate": 0.5, "mean_prefetch_stall_us": 0.25, "tier1_drm_hits": 1,
                         "tier2_uma_hits": 2, "tier3_nvme_misses": 3}
                    ],
                    "timing": {"ttft_ms": 90000.0, "prefill_tok_per_s": 0.1, "decode_tok_per_s": 0.02},
                },
            }
            return 200, {"Content-Type": "application/json"}, json.dumps(document).encode()

        return {"/v1/models": json_route(MODELS), "/v1/completions": completions}

    def test_buffered_records_text_usage_rates_and_cache(self):
        server = self.start(self.buffered_routes())
        code, out_json, out_md, stdout, stderr = self.run_probe(server.base_url)
        self.assertEqual(code, 0, stderr)
        with open(out_json, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertEqual(document["schema"], probe.SCHEMA)
        self.assertEqual(document["model"], "fake-model")
        self.assertFalse(document["stream"])
        self.assertEqual(len(document["results"]), 2)
        first = document["results"][0]
        self.assertTrue(first["ok"])
        self.assertEqual(first["text"], "<|end_of_sentence|>.\n")
        self.assertEqual(first["prompt_tokens"], 7)
        self.assertEqual(first["completion_tokens"], 4)
        self.assertEqual(first["completion_tokens_source"], "usage")
        self.assertEqual(first["http_status"], 200)
        self.assertEqual(first["server_timing"]["ttft_ms"], 90000.0)
        self.assertEqual(first["server_stats"]["kv_cache_hit_rate"], 0.0)
        self.assertAlmostEqual(first["tokens_per_s"], 4 / first["wall_s"], places=5)
        self.assertGreater(first["decode_tokens_per_s"], 0.0)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0]["model"], "fake-model")
        self.assertEqual(self.requests[0]["prompt"], "The capital of France is")
        self.assertEqual(self.requests[0]["max_tokens"], 4)
        self.assertEqual(self.requests[0]["temperature"], 0.0)
        self.assertNotIn("stream", self.requests[0])
        markdown = open(out_md, encoding="utf-8").read()
        self.assertIn("| # | id | prompt | completion text", markdown)
        self.assertIn("KV hit", markdown)
        self.assertIn("\\n", markdown)
        self.assertIn("<|end_of_sentence|>", markdown)
        self.assertIn('"id": "capital-of-france"', stdout)

    def test_stream_records_ttft_chunks_and_usage(self):
        def completions(body, handler):
            self.requests.append(json.loads(body))
            chunks = [
                (0.05, b'data: {"choices":[{"index":0,"text":"Par","finish_reason":null}]}\n\n'),
                (
                    0.0,
                    b'data: {"choices":[{"index":0,"text":"is","finish_reason":"stop"}],'
                    b'"x_deepseek_spark_stats":{"kv_cache_hit_rate":0.25}}\n\n',
                ),
                (0.0, b'data: {"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}\n\n'),
                (0.0, b"data: [DONE]\n\n"),
            ]
            return 200, {"Content-Type": "text/event-stream"}, chunks

        server = self.start({"/v1/models": json_route(MODELS), "/v1/completions": completions})
        code, out_json, _, _, stderr = self.run_probe(server.base_url, "--stream")
        self.assertEqual(code, 0, stderr)
        document = json.load(open(out_json, encoding="utf-8"))
        first = document["results"][0]
        self.assertTrue(first["ok"])
        self.assertEqual(first["text"], "Paris")
        self.assertEqual(first["completion_tokens"], 2)
        self.assertEqual(first["completion_tokens_source"], "usage")
        self.assertIsNotNone(first["ttft_s"])
        self.assertGreaterEqual(first["ttft_s"], 0.05)
        self.assertEqual(first["chunks"], 3)
        self.assertTrue(first["sse_done"])
        self.assertEqual(first["server_stats"]["kv_cache_hit_rate"], 0.25)
        self.assertTrue(self.requests[0]["stream"])
        self.assertTrue(self.requests[0]["stream_options"]["include_usage"])

    def test_stream_without_usage_counts_chunks(self):
        def completions(body, handler):
            self.requests.append(json.loads(body))
            return 200, {"Content-Type": "text/event-stream"}, [
                (0.0, b'data: {"choices":[{"index":0,"text":"a"}]}\n\n'),
                (0.0, b'data: {"choices":[{"index":0,"text":"b","finish_reason":"stop"}]}\n\n'),
                (0.0, b"data: [DONE]\n\n"),
            ]

        server = self.start({"/v1/models": json_route(MODELS), "/v1/completions": completions})
        code, out_json, _, _, stderr = self.run_probe(server.base_url, "--stream")
        self.assertEqual(code, 0, stderr)
        first = json.load(open(out_json, encoding="utf-8"))["results"][0]
        self.assertEqual(first["text"], "ab")
        self.assertEqual(first["completion_tokens"], 2)
        self.assertEqual(first["completion_tokens_source"], "chunks")
        self.assertIsNone(first["prompt_tokens"])

    def test_non_200_is_recorded_and_exit_one(self):
        def completions(body, handler):
            return 500, {"Content-Type": "application/json"}, b'{"error":"boom"}'

        server = self.start({"/v1/models": json_route(MODELS), "/v1/completions": completions})
        code, out_json, _, _, _ = self.run_probe(server.base_url)
        self.assertEqual(code, 1)
        document = json.load(open(out_json, encoding="utf-8"))
        self.assertEqual(document["summary"]["failed"], 2)
        first = document["results"][0]
        self.assertFalse(first["ok"])
        self.assertEqual(first["http_status"], 500)
        self.assertIn("HTTP 500", first["error"])
        self.assertIn("boom", first["error"])

    def test_models_failure_exits_two_without_outputs(self):
        server = self.start({"/v1/models": json_route({"error": "down"}, status=503)})
        code, out_json, out_md, _, stderr = self.run_probe(server.base_url)
        self.assertEqual(code, 2)
        self.assertIn("/v1/models failed", stderr)
        self.assertFalse(os.path.exists(out_json))
        self.assertFalse(os.path.exists(out_md))

    def test_timeout_is_recorded_and_bounded(self):
        def completions(body, handler):
            time.sleep(0.6)
            return 200, {"Content-Type": "application/json"}, b"{}"

        server = self.start({"/v1/models": json_route(MODELS), "/v1/completions": completions})
        started = time.monotonic()
        code, out_json, _, _, _ = self.run_probe(server.base_url, "--timeout", "0.15")
        elapsed = time.monotonic() - started
        self.assertEqual(code, 1)
        self.assertLess(elapsed, 5.0)
        first = json.load(open(out_json, encoding="utf-8"))["results"][0]
        self.assertFalse(first["ok"])
        self.assertIsNotNone(first["error"])

    def test_custom_prompts_file(self):
        prompts = [{"id": "one", "prompt": "first"}, {"id": "two", "prompt": "second"}]
        path = os.path.join(self.tmp, "prompts.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(prompts, handle)
        server = self.start(self.buffered_routes())
        code, out_json, _, _, stderr = self.run_probe(server.base_url, "--prompts", path)
        self.assertEqual(code, 0, stderr)
        document = json.load(open(out_json, encoding="utf-8"))
        self.assertEqual([result["id"] for result in document["results"]], ["one", "two"])
        self.assertEqual(len(self.requests), 2)

    def test_dry_run_plans_without_network(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = probe.main(["--dry-run", "--base-url", "http://127.0.0.1:1"])
        self.assertEqual(code, 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(len(plan["prompts"]), 2)
        self.assertEqual(plan["prompts"][0]["id"], "capital-of-france")
        self.assertEqual(plan["max_tokens"], 4)

    def test_bad_prompts_file_exits_two(self):
        path = os.path.join(self.tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([{"prompt": ""}], handle)
        code = probe.main(["--prompts", path, "--dry-run"])
        self.assertEqual(code, 2)

    def test_rate_math(self):
        self.assertEqual(probe.compute_rates(2.0, 4, None), {"tokens_per_s": 2.0, "decode_tokens_per_s": 2.0})
        self.assertEqual(probe.compute_rates(2.0, 4, 1.0), {"tokens_per_s": 2.0, "decode_tokens_per_s": 4.0})
        self.assertEqual(probe.compute_rates(0.0, 4, None), {"tokens_per_s": 0.0, "decode_tokens_per_s": 0.0})
        self.assertEqual(probe.compute_rates(2.0, 0, None), {"tokens_per_s": 0.0, "decode_tokens_per_s": 0.0})

    def test_proc_io_parsing_and_sampling(self):
        self.assertEqual(
            probe.parse_proc_io("rchar: 123\nwchar: 4\nread_bytes: 7\n"),
            {"rchar": 123, "wchar": 4, "read_bytes": 7},
        )
        self.assertIsNone(probe.read_proc_io(None))
        server = self.start(self.buffered_routes())
        code, out_json, _, _, stderr = self.run_probe(server.base_url, "--io-pid", str(os.getpid()))
        self.assertEqual(code, 0, stderr)
        first = json.load(open(out_json, encoding="utf-8"))["results"][0]
        self.assertIsInstance(first["io"], dict)
        self.assertGreaterEqual(first["io"]["rchar"], 0)
        self.assertGreaterEqual(first["io"]["read_bytes"], 0)
        self.assertIsNotNone(first["io"]["rchar_per_completion_token"])

    def test_markdown_cell_escapes(self):
        self.assertEqual(probe._cell("a|b\nc"), "a\\|b\\nc")
        self.assertEqual(probe._cell(None), "")


if __name__ == "__main__":
    unittest.main()
