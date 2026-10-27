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

"""Tests for the S2-a gate verdict embedded in ``scripts/gpu-reduce-bench.sh``.

The verdict is the only thing that turns four rank JSONs into PASS/FAIL, and it runs inside a bash
heredoc, so it had no test and -- as a review found -- three ways to PASS a run that should fail:

* it trusted the bench binary's own ``gate.pass`` boolean instead of re-deriving the latency bar;
* it never read the reduce correctness check (``check.differing`` / ``max_ulp`` / ``finite``);
* it never read ``stats.collectives``, so a bench that timed 101 of 10,000 collectives passed.

This test extracts the heredoc from the script verbatim (so it cannot drift from the shipped code),
runs it against synthetic rank directories, and asserts the verdict plus the reason it gives.

Run with ``python3 scripts/tests/test_reduce_gate_verdict.py``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "gpu-reduce-bench.sh")

P50_MAX = 75.0
P90_MAX = 104.0
P99_MAX = 251.0
SAMPLES = 10000


def extract_verdict() -> str:
    """The python program between the verdict's ``<<'PY'`` marker and its closing ``PY``.

    The script carries more than one python heredoc (the capability probe is one, and its body is
    nested inside a quoted ssh command), so this anchors on the verdict's own invocation line rather
    than taking the first heredoc in the file.
    """
    with open(SCRIPT) as handle:
        body = handle.read()
    anchor = body.index('python3 - "$LOCAL_DIR"')
    opening = body.index("<<'PY'\n", anchor) + len("<<'PY'\n")
    closing = body.index("\nPY\n", opening)
    return body[opening:closing]


def writable(path: str, document) -> None:
    with open(path, "w") as handle:
        if isinstance(document, str):
            handle.write(document)
        else:
            json.dump(document, handle)


def rank_document(rank: int, p50: float = 48.4, p90: float = 59.5, p99: float = 83.0, **overrides):
    """One bench-rank JSON in the shape ``gpu-reduce-bench`` writes for the gate."""
    check = {"differing": 0, "max_ulp": 0, "finite": True}
    stats = {
        "collectives": SAMPLES + 1,
        "registered_buffers": 2,
        "registered_bytes": 20480,
    }
    check.update(overrides.pop("check", {}))
    stats.update(overrides.pop("stats", {}))
    gate = {"p50_us": p50, "p90_us": p90, "p99_us": p99, "p50_max_us": P50_MAX,
            "p90_max_us": P90_MAX, "p99_max_us": P99_MAX,
            "pass": p50 <= P50_MAX and p90 <= P90_MAX and p99 <= P99_MAX}
    gate.update(overrides.pop("gate", {}))
    document = {
        "host": "test", "rank": rank, "world": 4, "device_only": True, "connect_ms": 0.0,
        "nccl": {"library": "libnccl.so.2", "version": "2.31.2",
                 "env": {"NCCL_IB_HCA": "rocep1s0f1", "NCCL_IB_GID_INDEX": "3"}},
        "arms": [{
            "arm": "nccl", "bytes": 10240, "elements": 5120, "mode": "sync", "batch": 40,
            "p50_us": p50, "p90_us": p90, "p99_us": p99, "mean_us": 51.0, "max_us": 120.0,
            "enqueue": {"p50_us": 5.8, "p90_us": 6.0, "p99_us": 6.6},
            "check": check, "stats": stats, "gate": gate,
        }],
    }
    document.update(overrides)
    return document


def clock_csv(path: str, values, power: float = 12.0) -> None:
    """Clocks-only CSV, the pre-timestamp format (the verdict must still read those rows)."""
    with open(path, "w") as handle:
        for value in values:
            handle.write("%s, %s\n" % (value, power))


def gate_dir(tmp: str, ranks) -> str:
    """A local result directory: 4 bench JSONs, 4 NCCL logs, 4 clock CSVs, the max-clock file."""
    for rank, document in enumerate(ranks):
        writable(os.path.join(tmp, "bench-rank%d.json" % rank), document)
        with open(os.path.join(tmp, "bench-rank%d.status" % rank), "w") as handle:
            handle.write("0\n")
        with open(os.path.join(tmp, "histogram-rank%d.json" % rank), "w") as handle:
            json.dump({"p50_us": 48.4, "p90_us": 59.5, "p99_us": 83.0}, handle)
        with open(os.path.join(tmp, "nccl-rank%d.log" % rank), "w") as handle:
            # The shape NCCL writes for a RoCE channel on this rig (no /GDRDMA on GB10).
            handle.write("node:1:1 [0] NCCL INFO Channel 00/0 : 3[0] -> 0[0] [receive] via NET/IB/0\n")
            handle.write("node:1:1 [0] NCCL INFO Channel 01/0 : 3[0] -> 0[0] [receive] via NET/IB/1\n")
        clock_csv(os.path.join(tmp, "clocks-rank%d.csv" % rank), [2405, 2411, 2411, 2411, 2405])
        with open(os.path.join(tmp, "platform-rank%d.json" % rank), "w") as handle:
            json.dump({"probe": "ok", "gdr_supported": 0, "dmabuf_supported": 0,
                       "sm_count": 48}, handle)
    with open(os.path.join(tmp, "clocks-max.txt"), "w") as handle:
        for rank in range(4):
            handle.write("%d,3003\n" % rank)
    return tmp


def run_verdict(tmp: str, dirty: str = "false") -> subprocess.CompletedProcess:
    program = extract_verdict()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(program)
        path = handle.name
    try:
        return subprocess.run(
            [sys.executable, path, tmp, "10240", str(SAMPLES), "200", str(P50_MAX), str(P90_MAX),
             str(P99_MAX), "90", "2411", "1500", "20260101T000000Z", "testhead", dirty,
             "testscriptsha"],
            capture_output=True, text=True)
    finally:
        os.unlink(path)


class ReduceGateVerdictTest(unittest.TestCase):
    def assertPass(self, result: subprocess.CompletedProcess) -> None:
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("gate: PASS", result.stdout)

    def assertFailWith(self, result: subprocess.CompletedProcess, needle: str) -> None:
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("gate: FAIL", result.stdout)
        self.assertIn(needle, result.stdout)

    def test_a_clean_run_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertPass(run_verdict(gate_dir(tmp, [rank_document(r) for r in range(4)])))

    def test_the_verdict_rederives_the_latency_bar(self):
        """A bench that mislabels its own pass must not carry the verdict."""
        ranks = [rank_document(r) for r in range(4)]
        ranks[0] = rank_document(0, p50=200.0, p99=500.0,
                                 gate={"pass": True, "p50_us": 200.0, "p99_us": 500.0})
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, ranks))
            self.assertFailWith(result, "latency gate failed")

    def test_the_verdict_rederives_the_p90_bar(self):
        """p90 is a bar in its own right (D57): a tail regression must fail even with p50 and p99 fine."""
        ranks = [rank_document(r) for r in range(4)]
        ranks[1] = rank_document(1, p90=200.0)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFailWith(run_verdict(gate_dir(tmp, ranks)), "p90 200.00 us > 104.0")

    def test_a_binary_that_predates_the_p90_bar_fails(self):
        ranks = [rank_document(r) for r in range(4)]
        gate = dict(ranks[2]["arms"][0]["gate"])
        del gate["p90_max_us"]
        ranks[2]["arms"][0]["gate"] = gate
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFailWith(run_verdict(gate_dir(tmp, ranks)), "no p90 ceiling")

    def test_a_bench_that_reports_pass_on_a_failing_number_is_flagged(self):
        ranks = [rank_document(r) for r in range(4)]
        gate = dict(ranks[0]["arms"][0]["gate"])
        gate["p50_us"] = 200.0
        gate["pass"] = True          # the lie
        ranks[0]["arms"][0]["gate"] = gate
        ranks[0]["arms"][0]["p50_us"] = 200.0
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, ranks))
            self.assertFailWith(result, "the binary and the verdict disagree")

    def test_a_wrong_reduce_result_fails(self):
        ranks = [rank_document(r) for r in range(4)]
        ranks[2] = rank_document(2, check={"differing": 512, "max_ulp": 9, "finite": False})
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, ranks))
            self.assertFailWith(result, "WRONG")

    def test_a_truncated_sample_fails(self):
        ranks = [rank_document(r) for r in range(4)]
        ranks[1] = rank_document(1, stats={"collectives": 101})
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, ranks))
            self.assertFailWith(result, "truncated sample")

    def test_an_unregistered_buffer_fails(self):
        ranks = [rank_document(r) for r in range(4)]
        ranks[3] = rank_document(3, stats={"registered_buffers": 0, "registered_bytes": 0})
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, ranks))
            self.assertFailWith(result, "did not register")

    def test_gdr_is_required_on_a_platform_that_supports_it(self):
        ranks = [rank_document(r) for r in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            gate_dir(tmp, ranks)
            for rank in range(4):
                writable(os.path.join(tmp, "platform-rank%d.json" % rank),
                         {"probe": "ok", "gdr_supported": 1, "dmabuf_supported": 0, "sm_count": 48})
            result = run_verdict(tmp)
            self.assertFailWith(result, "reports support")

    def test_gdr_is_recorded_not_waived_on_a_platform_without_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, [rank_document(r) for r in range(4)]))
            self.assertPass(result)
            self.assertIn("not drivable on this platform, recorded not waived", result.stdout)

    def test_a_missing_capability_probe_keeps_gdr_required(self):
        ranks = [rank_document(r) for r in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            gate_dir(tmp, ranks)
            os.unlink(os.path.join(tmp, "platform-rank2.json"))
            result = run_verdict(tmp)
            self.assertFailWith(result, "GDR stays required")

    def test_an_incomplete_probe_keeps_gdr_required(self):
        ranks = [rank_document(r) for r in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            gate_dir(tmp, ranks)
            writable(os.path.join(tmp, "platform-rank1.json"),
                     {"probe": "ok", "gdr_supported": 0, "sm_count": 48})
            result = run_verdict(tmp)
            self.assertFailWith(result, "GDR stays required")

    def test_a_log_with_no_channel_line_fails(self):
        ranks = [rank_document(r) for r in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            gate_dir(tmp, ranks)
            open(os.path.join(tmp, "nccl-rank0.log"), "w").close()
            result = run_verdict(tmp)
            self.assertFailWith(result, "no NET channel line")

    def test_an_idle_part_is_invalid_not_passed(self):
        ranks = [rank_document(r) for r in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            gate_dir(tmp, ranks)
            clock_csv(os.path.join(tmp, "clocks-rank0.csv"), [208, 300, 405])
            result = run_verdict(tmp)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("gate: INVALID", result.stdout)
            self.assertIn("never left its idle clock", result.stdout)

    def test_a_throttled_part_is_invalid_not_passed(self):
        ranks = [rank_document(r) for r in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            gate_dir(tmp, ranks)
            # A working clock, but well below the boost target: 1800 MHz is not "idle", it is slow.
            clock_csv(os.path.join(tmp, "clocks-rank1.csv"), [1800] * 8)
            result = run_verdict(tmp)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("active median clock", result.stdout)

    def test_a_dirty_tree_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, [rank_document(r) for r in range(4)]), dirty="true")
            self.assertPass(result)
            summary = json.load(open(os.path.join(tmp, "gate-summary.json")))
            self.assertFalse(summary["provenance"]["admissible_as_floor_row"])
            self.assertEqual(summary["provenance"]["git_dirty"], "true")

    def test_a_clean_tree_is_an_admissible_floor_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, [rank_document(r) for r in range(4)]))
            self.assertPass(result)
            summary = json.load(open(os.path.join(tmp, "gate-summary.json")))
            self.assertTrue(summary["provenance"]["admissible_as_floor_row"])
            self.assertEqual(summary["provenance"]["git_head"], "testhead")
            self.assertEqual(summary["clock"]["protocol"], "native-boost")

    def test_the_clock_scope_is_recorded_as_window_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_verdict(gate_dir(tmp, [rank_document(r) for r in range(4)]))
            self.assertPass(result)
            summary = json.load(open(os.path.join(tmp, "gate-summary.json")))
            self.assertIn("window-level", summary["clock"]["scope"])
            self.assertIn("NOT a measurement", summary["clock"]["scope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
