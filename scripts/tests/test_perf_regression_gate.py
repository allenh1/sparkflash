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

"""Unit tests for scripts/perf_regression_gate.py.

No network and no GPU: the tests build tiny evidence JSON files in a temporary directory and assert
the declared-band verdicts plus the exit codes: PASS (0), REGRESSION (1), INVALID (2), SKIP (3).
The SKIP tests assert that the exact evidence path and the rejection reason are printed, because a
gate that silently passes when it has no evidence would be worse than no gate.

Run with:

    python3 -m unittest discover -s scripts/tests -v
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import perf_regression_gate as gate

BASELINE_GENERATED = "2026-10-21T16:40:47Z"
CURRENT_GENERATED = "2026-10-22T09:00:00Z"
NOW = "2026-10-23T00:00:00Z"
ROOT = "12d58ad0878b5008e80d720251adb969a0069412"


def make_document(
    generated,
    *,
    root=ROOT,
    verdict="QUIET",
    decode_p50=8.0,
    ttft_p90=650.0,
    samples_ok=3,
    prompt="b69-fixture-130",
    max_tokens=9,
    gate_eligible=False,
    schema=gate.SCHEMA,
):
    """One evidence document with the single prompt row the gate reads."""
    return {
        "schema": schema,
        "generated_utc": generated,
        "gate_eligible": gate_eligible,
        "d38_root_precondition": {"served_root_id": root, "met": False},
        "load_conditions": {"verdict": verdict},
        "recipe": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20261021,
            "stream": True,
            "ignore_eos": True,
        },
        "prompts": [
            {
                "id": prompt,
                "max_tokens": max_tokens,
                "aggregate": {
                    "samples_ok": samples_ok,
                    "decode_tok_s": {
                        "p50": decode_p50,
                        "p90": decode_p50 * 1.1,
                        "label": "MEASURED-CLIENT",
                    },
                    "ttft_ms": {"p50": ttft_p90 * 0.9, "p90": ttft_p90, "label": "MEASURED-CLIENT"},
                },
            }
        ],
    }


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="perf-gate-")
        self.baseline_path = os.path.join(self.tmp, "d38-vllm-baseline-20261021.json")
        self.write(self.baseline_path, make_document(BASELINE_GENERATED))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, path, document):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def current_path(self, name="d38-vllm-baseline-20261022.json"):
        return os.path.join(self.tmp, name)

    def run_gate(self, *extra):
        argv = ["--baseline", self.baseline_path, "--evidence-dir", self.tmp, "--now", NOW]
        argv.extend(extra)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = gate.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    # ---- comparison math -------------------------------------------------

    def test_compare_within_bands_passes(self):
        comparison = gate.compare_rows(
            {"decode_tok_s_p50": 8.0, "ttft_ms_p90": 650.0},
            {"decode_tok_s_p50": 8.4, "ttft_ms_p90": 600.0},
            0.10,
            0.10,
        )
        self.assertEqual(comparison["verdict"], "PASS")
        self.assertEqual(comparison["decode_tok_s"]["verdict"], "PASS")
        self.assertEqual(comparison["ttft_ms"]["verdict"], "PASS")

    def test_compare_decode_regression(self):
        comparison = gate.compare_rows(
            {"decode_tok_s_p50": 8.0, "ttft_ms_p90": 650.0},
            {"decode_tok_s_p50": 7.19, "ttft_ms_p90": 650.0},
            0.10,
            0.10,
        )
        self.assertEqual(comparison["decode_tok_s"]["verdict"], "REGRESSION")
        self.assertEqual(comparison["verdict"], "REGRESSION")

    def test_compare_ttft_regression(self):
        comparison = gate.compare_rows(
            {"decode_tok_s_p50": 8.0, "ttft_ms_p90": 650.0},
            {"decode_tok_s_p50": 8.0, "ttft_ms_p90": 715.1},
            0.10,
            0.10,
        )
        self.assertEqual(comparison["ttft_ms"]["verdict"], "REGRESSION")
        self.assertEqual(comparison["verdict"], "REGRESSION")

    # ---- end-to-end exit codes -------------------------------------------

    def test_pass_prints_both_paths_and_returns_zero(self):
        path = self.write(self.current_path(), make_document(CURRENT_GENERATED))
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_PASS, out)
        self.assertIn(path, out)
        self.assertIn(os.path.abspath(self.baseline_path), out)
        self.assertIn("PASS (exit 0)", out)

    def test_decode_regression_returns_one(self):
        self.write(
            self.current_path(),
            make_document(CURRENT_GENERATED, decode_p50=6.0),
        )
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_REGRESSION, out)
        self.assertIn("FAIL (exit 1)", out)

    def test_ttft_regression_returns_one(self):
        self.write(
            self.current_path(),
            make_document(CURRENT_GENERATED, ttft_p90=900.0),
        )
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_REGRESSION, out)
        self.assertIn("REGRESSION", out)

    def test_wider_band_can_absolve_a_number(self):
        self.write(self.current_path(), make_document(CURRENT_GENERATED, decode_p50=6.0))
        code, _out, _err = self.run_gate("--decode-band", "0.30", "--ttft-band", "0.50")
        self.assertEqual(code, gate.EXIT_PASS)

    # ---- SKIP paths ------------------------------------------------------

    def test_skip_when_no_current_evidence_prints_expected_path_and_reason(self):
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn("SKIP (exit 3)", out)
        self.assertIn("no fresh, comparable, quiet evidence", out)
        self.assertIn(os.path.join(os.path.abspath(self.tmp), gate.DEFAULT_EVIDENCE_GLOB), out)
        self.assertIn("expected path", out)
        self.assertIn("nothing was compared and no number was fabricated", out)

    def test_skip_when_only_the_baseline_is_present(self):
        code, out, _err = self.run_gate("--evidence-glob", "d38-vllm-baseline-20261021.json")
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn("this is the baseline file itself", out)

    def test_skip_when_current_is_under_load(self):
        path = self.write(
            self.current_path(),
            make_document(CURRENT_GENERATED, verdict="UNDER-LOAD"),
        )
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn(path, out)
        self.assertIn("not QUIET", out)

    def test_skip_when_current_is_from_another_root(self):
        self.write(
            self.current_path(),
            make_document(CURRENT_GENERATED, root="1bae0e1088993b9b82de08f59de5be5a34e42170"),
        )
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn("served root differs", out)

    def test_skip_when_current_is_stale(self):
        # Baseline in the deep past so the current file is newer, but older than --max-age-days.
        self.write(self.baseline_path, make_document("2026-08-01T00:00:00Z"))
        self.write(self.current_path(), make_document("2026-08-02T00:00:00Z"))
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn("stale", out)

    def test_skip_when_current_is_not_newer_than_baseline(self):
        self.write(self.current_path(), make_document(BASELINE_GENERATED))
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn("not newer than the baseline", out)

    def test_skip_when_too_few_samples(self):
        self.write(self.current_path(), make_document(CURRENT_GENERATED, samples_ok=2))
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn("--min-samples", out)

    def test_skip_when_recipe_differs(self):
        doc = make_document(CURRENT_GENERATED)
        doc["recipe"]["seed"] = 1
        self.write(self.current_path(), doc)
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn("recipe mismatch", out)

    def test_explicit_current_skip_is_honest(self):
        path = self.write(
            self.current_path(),
            make_document(CURRENT_GENERATED, verdict="MIXED"),
        )
        code, out, _err = self.run_gate("--current", path)
        self.assertEqual(code, gate.EXIT_SKIP, out)
        self.assertIn(path, out)
        self.assertIn("no number was compared and none was fabricated", out)

    # ---- INVALID paths ---------------------------------------------------

    def test_missing_baseline_is_invalid_not_skip(self):
        argv = ["--baseline", os.path.join(self.tmp, "does-not-exist.json"), "--now", NOW]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = gate.main(argv)
        self.assertEqual(code, gate.EXIT_INVALID)
        self.assertIn("FAIL (exit 2)", stdout.getvalue())
        self.assertIn("does not exist", stderr.getvalue())

    def test_baseline_without_the_prompt_row_is_invalid(self):
        self.write(
            self.baseline_path,
            make_document(BASELINE_GENERATED, prompt="some-other-prompt"),
        )
        code, out, err = self.run_gate()
        self.assertEqual(code, gate.EXIT_INVALID)
        self.assertIn("FAIL (exit 2)", out)
        self.assertIn("no prompt row", err)

    # ---- discovery -------------------------------------------------------

    def test_newest_accepted_candidate_is_used(self):
        older = self.write(
            os.path.join(self.tmp, "d38-vllm-baseline-20261022.json"),
            make_document("2026-10-22T08:00:00Z", decode_p50=8.1),
        )
        newer = self.write(
            os.path.join(self.tmp, "d38-vllm-baseline-20261023.json"),
            make_document("2026-10-22T20:00:00Z", decode_p50=8.2),
        )
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_PASS, out)
        self.assertIn(newer, out)
        self.assertNotIn(older + "\n", out)

    def test_json_out_writes_the_verdict(self):
        self.write(self.current_path(), make_document(CURRENT_GENERATED))
        out_path = os.path.join(self.tmp, "verdict.json")
        code, _out, _err = self.run_gate("--json-out", out_path)
        self.assertEqual(code, gate.EXIT_PASS)
        with open(out_path, "r", encoding="utf-8") as handle:
            verdict = json.load(handle)
        self.assertEqual(verdict["verdict"], "PASS")
        self.assertEqual(verdict["gate"], "perf-regression")

    def test_skip_prints_how_to_produce_the_evidence(self):
        code, out, _err = self.run_gate()
        self.assertEqual(code, gate.EXIT_SKIP)
        self.assertIn("scripts/d38_vllm_recapture.py", out)


if __name__ == "__main__":
    unittest.main()
