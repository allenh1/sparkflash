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

"""End-to-end gate-correctness rehearsal for the acceptance chain (task-48).

Host-only: no device, no vLLM instance, no network. The window is expensive and
unrepeatable, so every failure mode this file can produce offline is one that must
never cost a window: it drives the real scripts as subprocesses, exactly as the
runbook does, and asserts the machinery refuses (or flags) instead of emitting
misleading evidence.

The chain under rehearsal:

    synthetic capture --scripts/d41_floors.py--> D41 gate bars (sparkflash-d41-bars/1)
    synthetic dump + sidecar + oracle --scripts/compare_native_vs_vllm.py--> verdict

Failure modes covered (each asserts a non-zero exit and no evidence file, or for a
flagged run an evidence file whose verdict is not PASS/UNBARED):

  * wrong vocabulary (--expect-vocab)
  * non-tiling logits offsets / sidecar basename mismatch
  * a missing reference_id (NOT_LIKE_FOR_LIKE, counted)
  * a --positions list the dump or oracle cannot satisfy (and duplicates)
  * a bars file with an unknown bar name
  * the demo-scale synthetic bars file (not a D41 gate file)
  * a capture with a missing required class (d41_floors exits 2, writes nothing)
  * a floors run that emits requires_review (d41_floors exits 3; the consumer refuses its bars)

What cannot be rehearsed offline is listed in docs/archive/build/acceptance-gate-rehearsal.md.

Run with:

    python3 -m unittest discover -s scripts/tests -v
    python3 scripts/tests/test_gate_rehearsal.py -v
"""

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)
# The consumer and the D41 capture builders live next to their unit tests; this
# rehearsal deliberately drives the real scripts, so it imports only constants and
# fixture builders, never the script internals under test.
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)

import compare_native_vs_vllm as compare  # noqa: E402
import test_d41_floors as d41_fixtures  # noqa: E402

COMPARE = os.path.join(SCRIPTS, "compare_native_vs_vllm.py")
FLOORS = os.path.join(SCRIPTS, "d41_floors.py")

#: Synthetic vocabulary and positions. The released checkpoint is 129,280 tokens and
#: the gate compares 129..137; the rehearsal declares --expect-vocab 1000 for speed,
#: and the one wrong-vocabulary case declares the released 129,280 instead.
VOCAB = 1000
RELEASED_VOCAB = 129_280
POSITIONS = list(range(129, 138))
PINNED_POSITIONS = ",".join(str(position) for position in POSITIONS)
RELEASED_ROOT = (
    "/cache/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/" + compare.DEFAULT_EXPECT_ROOT
)

#: Verbatim content of /home/user/phase2-evidence/demo-scale/bars.json (2026-10-19),
#: the protocol demo's bar file. It is evidence on the evidence host, not in this repo;
#: it is kept here as the "demo-scale synthetic bars" case: a plain {"bars": ...}
#: object with no schema, status, bar_class or provenance. The runbook says it is not
#: a gate, so the consumer must refuse it.
DEMO_SCALE_BARS = {
    "bars": {
        "positions_min": 9,
        "top1_agreement_min": 1.0,
        "top1000_overlap_mean_min": 1.0,
        "max_abs_delta_max": 0.0001,
        "kl_mean_max": 1e-06,
        "js_mean_max": 1e-06,
    }
}

#: A descending, deterministic full-vocabulary logit vector: argmax 0, no ties.
LOGITS = [-0.001 * token for token in range(VOCAB)]


def log_softmax(values):
    maximum = max(values)
    shifted = [value - maximum for value in values]
    total = sum(math.exp(value) for value in shifted)
    log_total = math.log(total)
    return [value - log_total for value in shifted]


LOGPROBS = log_softmax(LOGITS)


def oracle_document(positions=None, vocab=VOCAB, target_token=0):
    """An oracle JSON shaped like scripts/oracle_native.py's output."""
    positions = POSITIONS if positions is None else positions
    rows = [
        {
            "position": position,
            "target_token": target_token,
            "logprob_count": VOCAB,
            "logprobs": {str(token): LOGPROBS[token] for token in range(VOCAB)},
        }
        for position in positions
    ]
    return {
        "schema": compare.ORACLE_SCHEMA,
        "source": "vllm-released",
        "served": {"root": RELEASED_ROOT, "id": "synthetic"},
        "suite": {"path": "suite.json", "sha256": "0" * 64},
        "prompts": [
            {
                "id": "b69-fixture-130",
                "status": "ok",
                "response_sha256": "1" * 64,
                "positions": rows,
            }
        ],
    }


def run_script(script, arguments):
    return subprocess.run(
        [sys.executable, script, *arguments],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


class GateRehearsalTest(unittest.TestCase):
    """Drive the acceptance chain as an operator would, on synthetic inputs."""

    @classmethod
    def setUpClass(cls):
        # One real floors run produces the gate bars every compare case consumes: the
        # rehearsal's "valid bars" are the script's own output, not a hand-written stub.
        cls._floors_dir = tempfile.TemporaryDirectory()
        capture = os.path.join(cls._floors_dir.name, "capture.json")
        with open(capture, "w", encoding="utf-8") as handle:
            json.dump(d41_fixtures.standard_capture(), handle)
        cls.gate_bars = os.path.join(cls._floors_dir.name, "bars.json")
        result = run_script(
            FLOORS,
            [
                "--capture",
                capture,
                "--out-json",
                os.path.join(cls._floors_dir.name, "floors.json"),
                "--out-md",
                os.path.join(cls._floors_dir.name, "floors.md"),
                "--out-bars",
                cls.gate_bars,
                "--source-revision",
                "gate-rehearsal",
            ],
        )
        if result.returncode != 0:
            raise AssertionError(f"fixture floors run failed: {result.returncode}: {result.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls._floors_dir.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    # -- fixtures ---------------------------------------------------------

    def path(self, name):
        return os.path.join(self.directory.name, name)

    def write_oracle(self, document, name="oracle.json"):
        path = self.path(name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def write_dump(self, name="logits.jsonl", positions=None, vocab=VOCAB, reference_id=0, mutate=None):
        positions = POSITIONS if positions is None else positions
        dump = self.path(name)
        sidecar = dump + ".bin"
        rows = []
        with open(sidecar, "wb") as handle:
            for position in positions:
                offset = handle.tell()
                handle.write(struct.pack("<%df" % vocab, *LOGITS[:vocab]))
                row = {
                    "position": position,
                    "vocab": vocab,
                    "logits_offset": offset,
                    "logits_bin": sidecar,
                }
                if reference_id is not None:
                    row["reference_id"] = reference_id
                rows.append(row)
            if mutate is not None:
                mutate(rows)
        with open(dump, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return dump

    def compare_arguments(self, dump, oracle, positions=PINNED_POSITIONS, expect_vocab=VOCAB, bars=None):
        out_json = self.path("out.json")
        arguments = [
            "--dump",
            dump,
            "--oracle",
            oracle,
            "--out-json",
            out_json,
            "--bars",
            self.gate_bars if bars is None else bars,
        ]
        if positions is not None:
            arguments.extend(["--positions", positions])
        if expect_vocab is not None:
            arguments.extend(["--expect-vocab", str(expect_vocab)])
        return arguments, out_json

    def assert_no_evidence(self, out_json):
        self.assertFalse(os.path.exists(out_json), f"refused run wrote {out_json}")
        self.assertFalse(os.path.exists(os.path.splitext(out_json)[0] + ".md"), "refused run wrote markdown")

    # -- the positive control ---------------------------------------------

    def test_a_correct_synthetic_run_passes(self):
        dump = self.write_dump()
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(dump, oracle)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(out_json, encoding="utf-8") as handle:
            evidence = json.load(handle)
        self.assertEqual(evidence["verdict"], "PASS")
        self.assertEqual(evidence["aggregate"]["positions"], len(POSITIONS))
        self.assertEqual(evidence["aggregate"]["prefix_mismatches"], 0)
        self.assertTrue(os.path.exists(os.path.splitext(out_json)[0] + ".md"))

    # -- input-shape failures: exit 2, nothing written --------------------

    def test_a_wrong_vocabulary_dump_is_refused(self):
        dump = self.write_dump()
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(dump, oracle, expect_vocab=RELEASED_VOCAB)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("vocab", result.stderr)
        self.assert_no_evidence(out_json)

    def test_non_tiling_offsets_are_refused(self):
        def shift_one_row(rows):
            rows[4]["logits_offset"] += 4

        dump = self.write_dump(mutate=shift_one_row)
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(dump, oracle)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("tile", result.stderr)
        self.assert_no_evidence(out_json)

    def test_a_row_naming_another_dumps_sidecar_is_refused(self):
        def name_another_sidecar(rows):
            rows[2]["logits_bin"] = self.path("another.jsonl.bin")

        dump = self.write_dump(mutate=name_another_sidecar)
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(dump, oracle)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("logits_bin", result.stderr)
        self.assert_no_evidence(out_json)

    def test_a_sidecar_with_another_dumps_name_is_refused(self):
        dump = self.write_dump()
        other = self.path("other.jsonl.bin")
        os.replace(dump + ".bin", other)
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(dump, oracle)
        arguments.extend(["--bin", other])
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("sidecar", result.stderr)
        self.assert_no_evidence(out_json)

    # -- positions: the pinned list is a requirement, not a filter --------

    def test_a_positions_list_the_dump_cannot_satisfy_is_refused(self):
        # The dump is 130..138: the pinned 129..137 list names a position the dump lacks.
        dump = self.write_dump(positions=list(range(130, 139)))
        oracle = self.write_oracle(oracle_document(positions=POSITIONS))
        arguments, out_json = self.compare_arguments(dump, oracle)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("129", result.stderr)
        self.assert_no_evidence(out_json)

    def test_a_positions_list_the_oracle_cannot_satisfy_is_refused(self):
        dump = self.write_dump()
        oracle = self.write_oracle(oracle_document(positions=list(range(129, 137))))
        arguments, out_json = self.compare_arguments(dump, oracle)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("137", result.stderr)
        self.assert_no_evidence(out_json)

    def test_a_duplicate_position_list_is_refused(self):
        dump = self.write_dump()
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(
            dump, oracle, positions="129,130,131,132,133,134,135,136,137,137"
        )
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("duplicate", result.stderr)
        self.assert_no_evidence(out_json)

    # -- a flagged, honestly-labelled run ---------------------------------

    def test_a_missing_reference_id_is_not_like_for_like(self):
        dump = self.write_dump(reference_id=None)
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(dump, oracle)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 1, result.stderr)
        with open(out_json, encoding="utf-8") as handle:
            evidence = json.load(handle)
        self.assertEqual(evidence["verdict"], "NOT_LIKE_FOR_LIKE")
        self.assertEqual(evidence["aggregate"]["prefix_mismatches"], len(POSITIONS))
        self.assertEqual(evidence["aggregate"]["prefix_unknown"], len(POSITIONS))
        with open(os.path.splitext(out_json)[0] + ".md", encoding="utf-8") as handle:
            markdown = handle.read()
        self.assertIn("NOT_LIKE_FOR_LIKE", markdown)
        self.assertIn("UNKNOWN", markdown)

    # -- bars: only a D41 gate file may gate ------------------------------

    def test_an_unknown_bar_name_is_refused(self):
        with open(self.gate_bars, encoding="utf-8") as handle:
            bars = json.load(handle)
        bars["bars"]["token_equality"] = 1.0
        path = self.path("bars.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(bars, handle)
        dump = self.write_dump()
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(dump, oracle, bars=path)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown bar", result.stderr)
        self.assert_no_evidence(out_json)

    def test_the_demo_scale_bars_file_is_refused(self):
        path = self.path("demo-bars.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(DEMO_SCALE_BARS, handle)
        dump = self.write_dump()
        oracle = self.write_oracle(oracle_document())
        arguments, out_json = self.compare_arguments(dump, oracle, bars=path)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not a D41 gate bars file", result.stderr)
        self.assert_no_evidence(out_json)

    # -- the D41 side: capture validation and requires_review -------------

    def write_capture(self, document, name="capture.json"):
        path = self.path(name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def run_floors(self, capture, tag):
        out_json = self.path(f"{tag}-floors.json")
        out_md = self.path(f"{tag}-floors.md")
        out_bars = self.path(f"{tag}-bars.json")
        result = run_script(
            FLOORS,
            [
                "--capture",
                capture,
                "--out-json",
                out_json,
                "--out-md",
                out_md,
                "--out-bars",
                out_bars,
                "--require-classes",
                "repeat,idle_distractor,busy_distractor",
            ],
        )
        return result, out_json, out_md, out_bars

    def test_a_capture_missing_a_required_class_is_refused(self):
        blocks = [
            d41_fixtures.repeat_block(),
            d41_fixtures.distractor_block("busy", "busy_distractor", mode="busy"),
        ]
        capture = self.write_capture(d41_fixtures.make_capture(blocks))
        result, out_json, out_md, out_bars = self.run_floors(capture, "missing-class")
        self.assertEqual(result.returncode, 2)
        self.assertIn("idle_distractor", result.stderr)
        for path in (out_json, out_md, out_bars):
            self.assertFalse(os.path.exists(path), f"refused floors run wrote {path}")

    def test_a_requires_review_floors_run_cannot_gate(self):
        blocks = [
            d41_fixtures.repeat_block(),
            d41_fixtures.distractor_block("idle", "idle_distractor"),
            d41_fixtures.distractor_block("busy", "busy_distractor", mode="busy"),
        ]
        for record in blocks[1]["requests"]:
            if record["id"].endswith("-b"):
                for position in record["positions"]:
                    position["top"] = d41_fixtures.distribution("outlier", position["position"])
        capture = self.write_capture(d41_fixtures.make_capture(blocks))
        result, out_json, _, out_bars = self.run_floors(capture, "review")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertTrue(os.path.exists(out_bars))
        with open(out_bars, encoding="utf-8") as handle:
            bars = json.load(handle)
        self.assertEqual(bars["status"], "requires_review")
        # The chain's second half must refuse the reviewed-later bar file.
        dump = self.write_dump()
        oracle = self.write_oracle(oracle_document())
        arguments, evidence = self.compare_arguments(dump, oracle, bars=out_bars)
        result = run_script(COMPARE, arguments)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires_review", result.stderr)
        self.assert_no_evidence(evidence)


if __name__ == "__main__":
    unittest.main()
