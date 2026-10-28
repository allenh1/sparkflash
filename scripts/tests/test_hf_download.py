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

"""Tests for the resumable Hugging Face downloader the web control plane runs.

A local HTTP server plays both the Hugging Face API (127.0.0.1) and the storage host file downloads
redirect to (localhost, a different host name). Nothing leaves the machine. The contract guarded
here is the hub-cache layout huggingface_hub reads, the hash checks, resuming a partial file, the
progress lines, and that the token never reaches the storage host.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "hf-download.py"
REPO = "org/tiny-model"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def git_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324


FILES = {
    "config.json": (b'{"model_type": "deepseek_v4"}', False),
    "tokenizer.json": (b'{"version": "1.0"}' * 50, False),
    "model-00001-of-00002.safetensors": (os.urandom(3 << 20), True),
    "nested/model-00002-of-00002.safetensors": (os.urandom(1 << 20), True),
}


def etag(path: str) -> str:
    data, lfs = FILES[path]
    return hashlib.sha256(data).hexdigest() if lfs else git_oid(data)


class FakeHub(http.server.BaseHTTPRequestHandler):
    """The Hugging Face API on 127.0.0.1, its storage host on localhost."""

    port = 0
    requests: list[tuple[str, str, dict]] = []
    cut_once: set[str] = set()
    corrupt: set[str] = set()

    def log_message(self, *args):  # silence the default stderr log
        pass

    def send_json(self, value, link: str | None = None):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if link:
            self.send_header("Link", f'<{link}>; rel="next"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - the http.server hook name
        host = self.headers.get("Host", "").split(":")[0]
        FakeHub.requests.append((host, self.path, dict(self.headers)))
        api = f"http://127.0.0.1:{self.port}"
        if self.path == f"/api/models/{REPO}/revision/main":
            return self.send_json({"sha": COMMIT})
        if self.path == f"/api/models/{REPO}/tree/{COMMIT}?recursive=true":
            # Page one: a directory and two files; page two the rest.
            entries = [{"type": "directory", "path": "nested", "oid": "x", "size": 0}]
            entries += [self.entry(path) for path in list(FILES)[:2]]
            return self.send_json(entries, f"{api}/api/models/{REPO}/tree/{COMMIT}?cursor=2")
        if self.path == f"/api/models/{REPO}/tree/{COMMIT}?cursor=2":
            return self.send_json([self.entry(path) for path in list(FILES)[2:]])
        prefix = f"/{REPO}/resolve/{COMMIT}/"
        if self.path.startswith(prefix):
            path = self.path[len(prefix) :].replace("%2F", "/")
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.port}/cdn/{path}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        if self.path.startswith("/cdn/"):
            return self.serve_blob(self.path[len("/cdn/") :])
        self.send_error(404)
        return None

    def entry(self, path: str) -> dict:
        data, lfs = FILES[path]
        entry = {"type": "file", "path": path, "size": len(data), "oid": git_oid(data)}
        if lfs:
            entry["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
        return entry

    def serve_blob(self, path: str):
        data = FILES[path][0]
        if path in FakeHub.corrupt:
            data = bytes([data[0] ^ 1]) + data[1:]
        start = 0
        ranged = self.headers.get("Range", "")
        if ranged.startswith("bytes="):
            start = int(ranged[len("bytes=") :].split("-")[0])
        body = data[start:]
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if path in FakeHub.cut_once:
            # Send half and drop the connection, once.
            FakeHub.cut_once.discard(path)
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.close_connection = True
            self.connection.shutdown(2)
            return
        self.wfile.write(body)


class HfDownloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHub)
        FakeHub.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeHub.requests = []
        FakeHub.cut_once = set()
        FakeHub.corrupt = set()
        self.cache = Path(tempfile.mkdtemp())
        self.repo = self.cache / "models--org--tiny-model"

    def run_script(self, *extra: str) -> subprocess.CompletedProcess:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.cache),
            "HF_TOKEN": "hf_secret_token",
        }
        return subprocess.run(
            [sys.executable, "-I", str(SCRIPT), "--repo", REPO, "--cache", str(self.cache),
             "--endpoint", f"http://127.0.0.1:{FakeHub.port}", "--jobs", "3",
             "--retry-wait", "0", *extra],
            capture_output=True, text=True, env=env, timeout=120, check=False,
        )

    def assert_complete(self):
        self.assertEqual((self.repo / "refs" / "main").read_text(), COMMIT)
        snapshot = self.repo / "snapshots" / COMMIT
        for path, (data, _) in FILES.items():
            link = snapshot / path
            self.assertTrue(link.is_symlink(), path)
            self.assertFalse(os.path.isabs(os.readlink(link)), "links are relative")
            self.assertEqual(link.read_bytes(), data, path)
            self.assertTrue((self.repo / "blobs" / etag(path)).is_file())
        self.assertEqual(list((self.repo / "blobs").glob("*.incomplete")), [])

    def test_a_download_writes_the_hub_cache_layout_and_reports_progress(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assert_complete()
        progress = [
            json.loads(line[len("@progress ") :])
            for line in result.stdout.splitlines()
            if line.startswith("@progress ")
        ]
        self.assertTrue(progress)
        total = sum(len(data) for data, _ in FILES.values())
        self.assertEqual(progress[-1]["done"], total)
        self.assertEqual(progress[-1]["total"], total)
        self.assertIn("4 of 4 files", progress[-1]["detail"])

        api_auth = [h.get("Authorization") for host, _, h in FakeHub.requests if host == "127.0.0.1"]
        cdn_auth = [h.get("Authorization") for host, _, h in FakeHub.requests if host == "localhost"]
        self.assertTrue(cdn_auth, "the files came from the storage host")
        self.assertTrue(all(auth == "Bearer hf_secret_token" for auth in api_auth))
        self.assertEqual(set(cdn_auth), {None}, "the token never reaches the storage host")

        # A second run finds everything and fetches nothing.
        FakeHub.requests = []
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse([path for host, path, _ in FakeHub.requests if host == "localhost"])
        self.assert_complete()

    def test_a_partial_file_and_a_dropped_connection_are_resumed(self):
        big = "model-00001-of-00002.safetensors"
        blobs = self.repo / "blobs"
        blobs.mkdir(parents=True)
        (blobs / f"{etag(big)}.incomplete").write_bytes(FILES[big][0][:1000])
        FakeHub.cut_once = {"nested/model-00002-of-00002.safetensors"}
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assert_complete()
        ranges = [h.get("Range") for host, path, h in FakeHub.requests if path == f"/cdn/{big}"]
        self.assertEqual(ranges, ["bytes=1000-"])
        self.assertIn("retrying", result.stdout)

    def test_a_file_that_never_matches_its_hash_fails_and_stays_unfinished(self):
        FakeHub.corrupt = {"tokenizer.json"}
        result = self.run_script("--attempts", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tokenizer.json: giving up after 2 tries", result.stderr)
        # The cache scan must see the download as unfinished.
        self.assertTrue(list((self.repo / "blobs").glob("*.incomplete")))

    def test_too_little_disk_space_is_refused_before_downloading(self):
        result = subprocess.run(
            [sys.executable, "-I", "-c",
             "import runpy, shutil, sys, collections;"
             "U = collections.namedtuple('U', 'total used free');"
             "shutil.disk_usage = lambda path: U(10, 10, 0);"
             f"sys.argv = ['hf-download', '--repo', '{REPO}', '--cache', r'{self.cache}',"
             f" '--endpoint', 'http://127.0.0.1:{FakeHub.port}'];"
             f"runpy.run_path(r'{SCRIPT}', run_name='__main__')"],
            capture_output=True, text=True, timeout=60, check=False,
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(self.cache)},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Not enough disk space", result.stderr)
        self.assertFalse([path for host, path, _ in FakeHub.requests if host == "localhost"])


if __name__ == "__main__":
    unittest.main()
