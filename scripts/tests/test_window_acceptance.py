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

"""Unit tests for scripts/window_acceptance.py (one-command window acceptance).

No GPU, no network, no node: synthetic device trees built from the same fixtures as the component
tests. The component tools run as subprocesses exactly as they will in the window."""

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)

import accept_device_token as accept
import compare_native_vs_vllm as compare
import window_acceptance as window
import test_d48_step_gate as d48fx

RELEASED_ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
VOCAB = 129280


def write_json(path, doc):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)
        handle.write(chr(10))


def oracle_doc(position, token, logprob=-0.1):
    return {
        "schema": "sparkflash-native-oracle/1",
        "served": {"id": "deepseek-v4.1-flash", "root": RELEASED_ROOT},
        "prompts": [{"id": "p", "status": "ok", "prompt_len": 130,
                     "positions": [{"position": position, "logprobs": {str(token): logprob}, "target_token": token}]}],
    }


def token_dump(rank_dir, rank, position, token, vocab=VOCAB):
    import struct
    logits = [0.0] * vocab
    logits[token] = 40.0
    dump = os.path.join(rank_dir, "tp4-logits-r%d.jsonl" % rank)
    binary = dump + ".bin"
    row = {"position": position, "device_argmax": token, "logits_argmax": token,
           "reference_id": token, "hybrid_id": token, "fixture_id": None, "exact_token_equal": True,
           "vocab": vocab, "logits_offset": 0, "logits_bin": os.path.basename(binary),
           "top20": [[token, logits[token]]]}
    with open(dump, "w", encoding="utf-8") as handle:
        json.dump(row, handle)
        handle.write(chr(10))
    with open(binary, "wb") as handle:
        handle.write(struct.pack("<%df" % vocab, *logits))
    import hashlib
    def sha(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read())
        return digest.hexdigest()
    return dump, binary, sha(dump), sha(binary)


def token_envelope(rank_dir, rank, position, token, dump=None, forward="run", error=None):
    reference = None
    if dump is not None:
        path, binary, dump_sha, bin_sha = dump
        reference = {"path": os.path.basename(path), "bin": os.path.basename(binary),
                     "sha256": dump_sha, "bin_sha256": bin_sha, "dtype": "f32-le",
                     "vocab": VOCAB, "positions": [position], "rows": 1}
    if forward == "run":
        tokens = {"ids": [token], "positions": [position], "start_position": position, "prompt_len": 130,
                  "argmax_logits": [40.0], "top1_margins": [0.5], "logits_sha256": dump[2] if dump else None}
        positions_run = 1
    else:
        tokens = {"ids": [], "positions": [], "start_position": None, "prompt_len": 130}
        positions_run = 0
    document = {
        "schema": "sparkflash-tp4-token/1", "stage": "tp4-chain", "loop_step": "C4-token",
        "rank": rank, "world": 4, "forward": forward,
        "tokens": tokens, "positions_run": positions_run,
        "step_ms": {"samples": [31.5], "mean_ms": 31.5, "worst_ms": 31.5} if forward == "run" else None,
        "wall_ms": [47.2] if forward == "run" else [],
        "reduce_calls_per_step": 80, "logits_dump": reference,
        "head": {"vocab": VOCAB, "rule": "rank-sharded head all-gather", "tie_break": "lowest token id"},
        "error": error,
    }
    write_json(os.path.join(rank_dir, "tp4-token-r%d.json" % rank), document)


class WindowAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.device = os.path.join(self.root, "device")
        self.out = os.path.join(self.root, "qa")

    def make_tree(self, *, bind=True, token=False, position=129, token_id=12345):
        for rank in range(4):
            rank_dir = os.path.join(self.device, "rank%d" % rank)
            os.makedirs(rank_dir, exist_ok=True)
            write_json(os.path.join(rank_dir, "tp4-loop-r%d.json" % rank), d48fx.loop_doc(rank))
            if bind:
                write_json(os.path.join(rank_dir, "tp4-bind-r%d.json" % rank), d48fx.bind_doc(rank))
        if token:
            for rank in range(4):
                rank_dir = os.path.join(self.device, "rank%d" % rank)
                dump = token_dump(rank_dir, rank, position, token_id) if rank == 0 else None
                token_envelope(rank_dir, rank, position, token_id, dump=dump)
        return self.device

    def run_cli(self, *argv):
        out_json = os.path.join(self.out, "window-acceptance.json")
        code = window.main([*argv, "--out-dir", self.out])
        doc = None
        if os.path.exists(out_json):
            with open(out_json, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
        return code, doc

    def test_pre_token_tree_is_pre_token_pass(self):
        device = self.make_tree()
        code, doc = self.run_cli("--device-dir", device)
        self.assertEqual(code, 3)
        self.assertEqual(doc["schema"], window.SCHEMA)
        self.assertEqual(doc["rows"]["bind"]["verdict"], "PASS")
        self.assertEqual(doc["rows"]["token"]["verdict"], "NOT-ATTEMPTED")
        self.assertEqual(doc["rows"]["step_time"]["verdict"], "UNMEASURED")
        self.assertIn("PRE-TOKEN-PASS", doc["overall"]["label"])
        self.assertEqual(doc["overall"]["verdict"], "NOT-ATTEMPTED")
        self.assertTrue(doc["evidence_manifest"])
        self.assertTrue(os.path.exists(os.path.join(self.out, "evidence-sha256.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.out, "window-acceptance.md")))

    def test_token_tree_measures_token_and_step_time(self):
        device = self.make_tree(token=True)
        code, doc = self.run_cli("--device-dir", device, "--positions", "129")
        self.assertEqual(code, 3)
        self.assertEqual(doc["rows"]["token"]["verdict"], "MEASURED")
        self.assertEqual(doc["rows"]["token"]["positions_run"], 1)
        self.assertEqual(doc["rows"]["step_time"]["verdict"], "MEASURED")
        self.assertAlmostEqual(doc["rows"]["step_time"]["step_ms_mean"], 31.5)
        self.assertEqual(doc["rows"]["oracle_comparison"]["verdict"], "UNMEASURED")
        self.assertIn("TOKEN-MEASURED-UNBARED", doc["overall"]["label"])
        self.assertEqual(doc["rows"]["step_time"]["accepted_tokens_per_step"].split(" ")[0], "NOT-ATTEMPTED")

    def test_token_with_oracle_is_unbared_not_pass(self):
        device = self.make_tree(token=True)
        oracle = os.path.join(self.root, "oracle.json")
        write_json(oracle, oracle_doc(129, 12345))
        code, doc = self.run_cli("--device-dir", device, "--positions", "129", "--oracle", oracle, "--oracle-prompt", "p")
        self.assertEqual(code, 3)
        self.assertEqual(doc["rows"]["oracle_comparison"]["verdict"], "UNBARED")
        self.assertIn("TOKEN-MEASURED-UNBARED", doc["overall"]["label"])

    def test_missing_primary_bind_is_a_failure(self):
        device = self.make_tree(bind=False)
        code, doc = self.run_cli("--device-dir", device)
        self.assertEqual(code, 1)
        self.assertEqual(doc["overall"]["verdict"], "FAIL")
        self.assertEqual(doc["rows"]["bind"]["verdict"], "FAIL")

    def test_forward_refused_is_not_attempted_with_named_blocker(self):
        device = self.make_tree()
        for rank in range(4):
            rank_dir = os.path.join(device, "rank%d" % rank)
            token_envelope(rank_dir, rank, 129, 0, forward="refused",
                           error="stored-layout combine kernel unavailable: b12x_mxfp4_moe_sm121")
        code, doc = self.run_cli("--device-dir", device)
        self.assertEqual(code, 3)
        self.assertEqual(doc["rows"]["token"]["verdict"], "NOT-ATTEMPTED")
        text = " ".join(doc["rows"]["token"]["reasons"])
        self.assertIn("stored-layout combine kernel unavailable", text)
        self.assertIn("PRE-TOKEN-PASS", doc["overall"]["label"])

    def test_twin_pass_with_unbared_oracle_is_pass(self):
        device = self.make_tree(token=True)
        twin_dir = os.path.join(self.root, "twin")
        os.makedirs(twin_dir, exist_ok=True)
        twin = {
            "schema": accept.CANONICAL_SCHEMA, "kind": "device", "rank": 0, "world_size": 4,
            "prompt_tokens": [1, 2, 3], "token_ids": [12345], "token_emitted": True,
            "positions": [129],
        }
        write_json(os.path.join(twin_dir, "twin-r0.json"), twin)
        oracle = os.path.join(self.root, "oracle.json")
        write_json(oracle, oracle_doc(129, 12345))
        code, doc = self.run_cli("--device-dir", device, "--positions", "129", "--twin", twin_dir,
                                 "--oracle", oracle, "--oracle-prompt", "p")
        self.assertEqual(code, 0)
        self.assertEqual(doc["rows"]["twin_match"]["verdict"], "PASS")
        self.assertEqual(doc["rows"]["oracle_comparison"]["verdict"], "UNBARED")
        self.assertEqual(doc["overall"]["verdict"], "PASS")
        self.assertNotIn("D41-gated", doc["overall"]["label"])

    def test_explicit_dump_and_bars_are_passed_through(self):
        device = self.make_tree(token=True)
        twin_dir = os.path.join(self.root, "twin")
        os.makedirs(twin_dir, exist_ok=True)
        twin = {
            "schema": accept.CANONICAL_SCHEMA, "kind": "device", "rank": 0, "world_size": 4,
            "prompt_tokens": [1, 2, 3], "token_ids": [12345], "token_emitted": True,
            "positions": [129],
        }
        write_json(os.path.join(twin_dir, "twin-r0.json"), twin)
        oracle = os.path.join(self.root, "oracle.json")
        write_json(oracle, oracle_doc(129, 12345))
        bars = os.path.join(self.root, "bars.json")
        write_json(bars, {
            "schema": compare.GATE_BARS_SCHEMA, "status": "ok", "bar_class": "idle_distractor",
            "bars": {"positions_min": 1, "top1_agreement_min": 1.0},
        })
        explicit_dir = os.path.join(self.root, "explicit")
        os.makedirs(explicit_dir, exist_ok=True)
        dump, binary, _dump_sha, _bin_sha = token_dump(explicit_dir, 9, 129, 12345)
        code, doc = self.run_cli(
            "--device-dir", device, "--positions", "129", "--twin", twin_dir,
            "--oracle", oracle, "--oracle-prompt", "p", "--oracle-bars", bars,
            "--device-dump", dump, "--device-bin", binary, "--dump-source", "explicit-dry-run",
        )
        self.assertEqual(code, 0)
        self.assertEqual(doc["rows"]["oracle_comparison"]["verdict"], "PASS")
        self.assertEqual(doc["overall"]["verdict"], "PASS")
        self.assertEqual(doc["inputs"]["device_dump"], os.path.abspath(dump))
        self.assertEqual(doc["inputs"]["device_bin"], os.path.abspath(binary))
        self.assertEqual(doc["inputs"]["dump_source"], "explicit-dry-run")
        self.assertEqual(doc["inputs"]["oracle_bars"], os.path.abspath(bars))
        self.assertEqual(doc["inputs"]["oracle_bars_sha256"], window.sha256_file(bars))

    def test_require_token_makes_absence_a_failure(self):
        device = self.make_tree()
        code, doc = self.run_cli("--device-dir", device, "--require-token")
        self.assertEqual(code, 1)
        self.assertEqual(doc["overall"]["verdict"], "FAIL")

    def test_out_dir_equal_to_evidence_dir_is_refused(self):
        device = self.make_tree()
        code = window.main(["--device-dir", device, "--out-dir", device])
        self.assertEqual(code, 2)

    def test_broken_dump_sha_fails_shape(self):
        device = self.make_tree(token=True)
        envelope = os.path.join(device, "rank0", "tp4-token-r0.json")
        with open(envelope, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        document["logits_dump"]["sha256"] = "0" * 64
        write_json(envelope, document)
        code, doc = self.run_cli("--device-dir", device, "--positions", "129")
        self.assertEqual(code, 1)
        self.assertEqual(doc["rows"]["token"]["shape_verdict"], "FAIL")


    def test_twin_evidence_from_host_reference_fixture_passes(self):
        import twin_evidence
        device = self.make_tree(token=True)
        fixture = os.path.join(self.root, "fixture.json")
        write_json(fixture, {"prompt_tokens": [1, 2, 3],
                             "steps": [{"kind": "prefill", "position": 129, "next_token": 12345}]})
        twin_dir = os.path.join(self.root, "twin")
        self.assertEqual(twin_evidence.main(["--fixture", fixture, "--out-dir", twin_dir]), 0)
        code, doc = self.run_cli("--device-dir", device, "--positions", "129", "--twin", twin_dir)
        self.assertEqual(code, 0)
        self.assertEqual(doc["rows"]["twin_match"]["verdict"], "PASS")
        self.assertEqual(doc["rows"]["oracle_comparison"]["verdict"], "UNMEASURED")
        self.assertIn("M1 token gate met", doc["overall"]["label"])

    def test_forward_refused_with_oracle_supplied_does_not_join_oracle(self):
        device = self.make_tree()
        for rank in range(4):
            rank_dir = os.path.join(device, "rank%d" % rank)
            token_envelope(rank_dir, rank, 129, 0, forward="refused",
                           error="stored-layout combine kernel unavailable: b12x_mxfp4_moe_sm121")
        oracle = os.path.join(self.root, "oracle.json")
        write_json(oracle, oracle_doc(129, 491))
        code, doc = self.run_cli("--device-dir", device, "--positions", "129",
                                 "--oracle", oracle, "--oracle-prompt", "p")
        self.assertEqual(code, 3)
        self.assertEqual(doc["rows"]["token"]["verdict"], "NOT-ATTEMPTED")
        self.assertEqual(doc["rows"]["oracle_comparison"]["verdict"], "UNMEASURED")
        reasons = " ".join(doc["rows"]["oracle_comparison"]["reasons"])
        self.assertIn("no explicit positions", reasons)
        self.assertIn("PRE-TOKEN-PASS", doc["overall"]["label"])
        self.assertEqual(doc["overall"]["verdict"], "NOT-ATTEMPTED")

if __name__ == "__main__":
    unittest.main()