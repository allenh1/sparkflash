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

"""Tests for the lead-to-node copy the web control plane runs.

A stand-in `ssh` runs the remote command locally, with each host's files landing in its own
directory, so the real `rsync` does the copy without a network. The contract guarded here is that
the files arrive at the same absolute path, symlinks stay symlinks, excluded names stay behind, the
launcher's directories are created, host keys are checked under the node's SSH name, and progress
is reported. The test needs rsync 3.1 or newer (a DGX Spark has it; macOS's built-in one does not).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "sync-nodes.py"


def rsync_is_new_enough() -> bool:
    try:
        out = subprocess.run(["rsync", "--version"], capture_output=True, text=True).stdout
    except OSError:
        return False
    match = re.search(r"version\s+(\d+)\.(\d+)", out)
    return bool(match) and (int(match.group(1)), int(match.group(2))) >= (3, 1)


FAKE_SSH = textwrap.dedent(
    """\
    import os, sys
    args = sys.argv[1:]
    with open(os.environ["FAKE_SSH_LOG"], "a") as log:
        log.write(" ".join(args) + "\\n")
    i = 0
    while args[i].startswith("-"):
        i += 2 if args[i] in ("-o", "-l", "-p", "-i", "-c") else 1
    host = args[i].rpartition("@")[2]
    if host == "unreachable":
        sys.stderr.write("ssh: connect to host unreachable port 22: No route to host\\n")
        sys.exit(255)
    command = " ".join(args[i + 1:]).replace("@HOST@", host)
    os.execvp("sh", ["sh", "-c", command])
    """
)


@unittest.skipUnless(rsync_is_new_enough(), "needs rsync 3.1 or newer")
class SyncNodesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fake = self.tmp / "fake_ssh.py"
        self.fake.write_text(FAKE_SSH)
        self.log = self.tmp / "ssh.log"
        self.dest = self.tmp / "dest"
        model = self.tmp / "src" / "hub" / "models--org--tiny"
        (model / "blobs").mkdir(parents=True)
        (model / "snapshots" / "abc" / "nested").mkdir(parents=True)
        (model / "refs").mkdir()
        (model / "blobs" / "aaaa").write_bytes(os.urandom(300_000))
        (model / "blobs" / "bbbb").write_bytes(b"{}")
        (model / "blobs" / "cccc.incomplete").write_bytes(b"partial")
        (model / "refs" / "main").write_text("abc")
        os.symlink("../../../blobs/aaaa", model / "snapshots" / "abc" / "nested" / "weights.bin")
        os.symlink("../../blobs/bbbb", model / "snapshots" / "abc" / "config.json")
        self.model = model
        self.repo = self.tmp / "src" / "checkout"

    def run_script(self, *hosts: str) -> subprocess.CompletedProcess:
        argv = [sys.executable, "-I", str(SCRIPT), "--path", str(self.model),
                "--exclude", "*.incomplete", "--ensure-dir", str(self.repo), "--streams", "2",
                "--ssh-arg=-o", "--ssh-arg=BatchMode=yes",
                "--ssh", shlex.join([sys.executable, str(self.fake)]),
                "--dest-root", f"{self.dest}/@HOST@"]
        for host in hosts:
            argv += ["--host", host]
        env = dict(os.environ, FAKE_SSH_LOG=str(self.log))
        return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120,
                              check=False)

    def test_files_arrive_at_the_same_absolute_paths_on_every_host(self):
        result = self.run_script("spark@198.51.100.12,spark-1", "spark-2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for host in ("198.51.100.12", "spark-2"):
            root = Path(f"{self.dest}/{host}")
            copy = Path(str(root) + str(self.model))
            self.assertEqual((copy / "blobs" / "aaaa").read_bytes(),
                             (self.model / "blobs" / "aaaa").read_bytes())
            self.assertFalse((copy / "blobs" / "cccc.incomplete").exists())
            link = copy / "snapshots" / "abc" / "nested" / "weights.bin"
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), "../../../blobs/aaaa")
            self.assertEqual((copy / "refs" / "main").read_text(), "abc")
            self.assertTrue(Path(str(root) + str(self.repo)).is_dir(), "the checkout dir exists")

        log = self.log.read_text()
        self.assertIn("HostKeyAlias=spark-1", log)
        self.assertIn("-o BatchMode=yes", log)
        progress = [json.loads(line[len("@progress "):]) for line in result.stdout.splitlines()
                    if line.startswith("@progress ")]
        self.assertTrue(progress)
        self.assertEqual(progress[-1]["done"], progress[-1]["total"])
        self.assertEqual(progress[-1]["total"], 2 * (300_000 + 2 + 3), "blobs and refs/main")

        # A rerun has nothing left to send and still succeeds.
        again = self.run_script("spark@198.51.100.12,spark-1", "spark-2")
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)

    def test_an_unreachable_host_is_named_before_anything_is_copied(self):
        result = self.run_script("spark-1", "unreachable")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Can't prepare unreachable", result.stderr)
        self.assertIn("No route to host", result.stderr)
        self.assertFalse(Path(f"{self.dest}/spark-1{self.model}").exists())


class SplitTest(unittest.TestCase):
    def test_streams_get_about_equal_bytes_and_every_file_once(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("sync_nodes", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        entries = [(f"/m/f{i}", size) for i, size in enumerate([50, 40, 30, 20, 10, 10, 0])]
        buckets = module.split(entries, 3)
        self.assertEqual(sorted(e for b in buckets for e in b), sorted(entries))
        totals = sorted(sum(size for _, size in bucket) for bucket in buckets)
        self.assertEqual(totals, [50, 50, 60])


if __name__ == "__main__":
    unittest.main()
