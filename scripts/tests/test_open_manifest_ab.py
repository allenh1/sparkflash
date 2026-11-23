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

"""Unit tests for scripts/open-manifest-ab.py. No GPU, server or checkpoint."""

import base64
import importlib.util
import os
import struct
import tempfile
import types
import unittest
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location(
    "open_manifest_ab", os.path.join(SCRIPTS, "open-manifest-ab.py")
)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_chunks(data):
    """(type, payload) for every chunk, checking each stored CRC."""
    assert data[:8] == PNG_SIGNATURE, "not a PNG"
    at = 8
    chunks = []
    while at < len(data):
        length = struct.unpack(">I", data[at : at + 4])[0]
        kind = data[at + 4 : at + 8]
        payload = data[at + 8 : at + 8 + length]
        stored = struct.unpack(">I", data[at + 8 + length : at + 12 + length])[0]
        computed = zlib.crc32(kind + payload) & 0xFFFFFFFF
        assert stored == computed, f"{kind!r} CRC {stored:08x} != {computed:08x}"
        chunks.append((kind, payload))
        at += 12 + length
    return chunks


class OpenManifestAbTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def test_manifest_lines_compare_as_multisets(self):
        old = write(self.path("old.jsonl"), '{"kind":"model","name":"dim","value":5120}\n')
        new = write(self.path("new.jsonl"), '{"kind":"model","name":"dim","value":5120}\n')
        self.assertIsNone(ab.line_difference(old, new))

    def test_a_changed_manifest_value_is_a_difference(self):
        old = write(self.path("old.jsonl"), '{"kind":"model","name":"dim","value":5120}\n')
        new = write(self.path("new.jsonl"), '{"kind":"model","name":"dim","value":4096}\n')
        self.assertIn("differs", ab.line_difference(old, new))

    def test_launch_trace_order_matters(self):
        old = write(self.path("old.trace"), "a\nb\n")
        new = write(self.path("new.trace"), "b\na\n")
        self.assertIsNotNone(ab.ordered_difference(old, new))

    def test_tick_trace_ignores_only_the_clock(self):
        old = write(self.path("old.tick"), "t=1 clock=1 hold 0\n")
        new = write(self.path("new.tick"), "t=1 clock=999 hold 0\n")
        self.assertIsNone(ab.tick_difference(old, new))

    def test_tick_trace_reports_a_changed_decision(self):
        old = write(self.path("old.tick"), "t=1 clock=1 hold 0\n")
        new = write(self.path("new.tick"), "t=1 clock=1 hold 1\n")
        self.assertIn("differs", ab.tick_difference(old, new))

    def test_missing_rank_files_are_a_failure(self):
        old = self.path("old")
        new = self.path("new")
        os.makedirs(old)
        os.makedirs(new)
        self.assertFalse(ab.compare(old, new, 0.05))

    def test_the_launcher_environment_comes_from_arguments(self):
        args = types.SimpleNamespace(
            nodes=["a", "b", "c", "d"],
            checkpoint="/checkpoint",
            tokenizer="/tokenizer",
            converted_dir="/converted",
            cubins=None,
            build="/build/sparkflash",
        )
        out = self.path("record")
        env = ab.launch_env(args, out)
        self.assertEqual(env["SPARK_NODES"], "a b c d")
        self.assertEqual(env["SPARKFLASH_CHECKPOINT"], "/checkpoint")
        self.assertEqual(env["SPARKFLASH_TOKENIZER"], "/tokenizer")
        self.assertEqual(env["SPARKFLASH_CONVERTED_DIR"], "/converted")
        self.assertNotIn("SPARKFLASH_CUBINS", env)
        self.assertEqual(env["SPARKFLASH_BIN"], "/build/sparkflash")
        self.assertEqual(ab.launcher_env(args)["SPARK_NODES"], "a b c d")

    def test_the_png_fixture_has_valid_chunk_crcs_and_an_iend(self):
        data = base64.b64decode(ab.DOT_PNG, validate=True)
        chunks = png_chunks(data)
        kinds = [kind for kind, _ in chunks]
        self.assertEqual(kinds[0], b"IHDR")
        self.assertIn(b"IDAT", kinds)
        self.assertEqual(kinds[-1], b"IEND")
        width, height = struct.unpack(">II", chunks[0][1][:8])
        self.assertEqual((width, height), (1, 1))
        self.assertEqual(ab.IMAGE_URL, "data:image/png;base64," + ab.DOT_PNG)

    def test_the_replay_workload_parses_the_served_stats(self):
        import http.server
        import json
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": "mock-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                self.assert_request(request)
                body = json.dumps(
                    {
                        "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                        "x_deepseek_spark_stats": {
                            "timing": {
                                "prefill_tok_per_s": 100.0,
                                "decode_tok_per_s": 10.0,
                                "ttft_ms": 5.0,
                            },
                            "dspark_speculative": {
                                "draft_tokens_accepted": 2,
                                "draft_tokens_proposed": 4,
                            },
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def assert_request(self, request):
                if request.get("temperature") != 0.0:
                    raise AssertionError("the replay workload must be greedy")
                if request.get("model") != "mock-model":
                    raise AssertionError("the workload must use the served model id")

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        ab.served_model.cache_clear()
        args = types.SimpleNamespace(
            url="http://127.0.0.1:%d" % server.server_port,
            max_tokens=4,
            bulk_tokens=8,
            draft_tokens=8,
            request_timeout=5,
        )
        speeds = ab.run_workload(args)
        self.assertGreater(speeds["decode_1"], 0)
        self.assertGreater(speeds["decode_8"], 0)
        self.assertGreater(speeds["prefill"], 0)
        self.assertAlmostEqual(speeds["draft_accept"], 0.5)


if __name__ == "__main__":
    unittest.main()
