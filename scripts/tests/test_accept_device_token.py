#!/usr/bin/env python3
"""Unit tests for scripts/accept_device_token.py on synthetic host data.

No server, no GPU, no network, no real checkpoint. Run with:

    python3 -m unittest discover -s scripts/tests -v
"""

import json
import math
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import accept_device_token as accept
import compare_native_vs_vllm as compare

RELEASED_ROOT = "/cache/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/" + compare.DEFAULT_EXPECT_ROOT
VOCAB = 4


def write_json(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)
    return path


def write_dump(path, rows, bin_path=None):
    """rows: list of (position, logits list, reference_id or None)."""
    bin_path = bin_path or (path + ".bin")
    with open(path, "w", encoding="utf-8") as handle:
        for position, logits, reference_id in rows:
            row = {"position": position, "vocab": len(logits), "logits_bin": bin_path}
            if reference_id is not None:
                row["reference_id"] = reference_id
                row["hybrid_id"] = reference_id
            offset = sum(len(previous[1]) * 4 for previous in rows[: len(handle_get_rows(handle))])
            row["logits_offset"] = offset
            handle.write(json.dumps(row) + "\n")
    # offsets are recomputed below because the loop above cannot see its own rows
    rows_with_offset = []
    offset = 0
    for position, logits, reference_id in rows:
        rows_with_offset.append((position, logits, reference_id, offset))
        offset += len(logits) * 4
    with open(path, "w", encoding="utf-8") as handle:
        for position, logits, reference_id, offset in rows_with_offset:
            row = {"position": position, "vocab": len(logits), "logits_bin": bin_path, "logits_offset": offset}
            if reference_id is not None:
                row["reference_id"] = reference_id
                row["hybrid_id"] = reference_id
            handle.write(json.dumps(row) + "\n")
    with open(bin_path, "wb") as handle:
        for _, logits, _ in rows:
            handle.write(struct.pack("<%df" % len(logits), *logits))
    return path


def handle_get_rows(handle):
    return []


def write_text(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def write_bind(path, rank, *, attention_bound=208, attention_expected=208, moe_complete=True, tail_complete=True, census_unclassified=None, error=None, summary_complete=None):
    """A sparkflash-tp4-bind/1 document with the window's confirmed class counts."""
    attention_complete = attention_bound == attention_expected and attention_expected > 0
    moe_classes = [
        {"class": "routed_experts_gate_up", "expected": 61440, "bound": 61440 if moe_complete else 61439, "complete": moe_complete, "missing": [] if moe_complete else ["x"], "unexpected": []},
        {"class": "routed_experts_down", "expected": 30720, "bound": 30720, "complete": moe_complete, "missing": [] if moe_complete else ["x"], "unexpected": []},
        {"class": "shared_experts", "expected": 240, "bound": 240, "complete": moe_complete, "missing": [] if moe_complete else ["x"], "unexpected": []},
        {"class": "router", "expected": 120, "bound": 120, "complete": moe_complete, "missing": [] if moe_complete else ["x"], "unexpected": []},
    ]
    complete = error is None and attention_complete and moe_complete and tail_complete and not (census_unclassified or [])
    doc = {
        "stage": "tp4-chain",
        "schema": "sparkflash-tp4-bind/1",
        "loop_step": "C1-bind",
        "setup_only": True,
        "rank": rank,
        "world": 4,
        "forward": "not-run",
        "tokens": 0,
        "attention": {
            "tensors_expected": attention_expected,
            "tensors_bound": attention_bound,
            "tensors_table_entries": 215,
            "sharded": True,
            "coverage": {
                "class": "attention",
                "expected": attention_expected,
                "bound": attention_bound,
                "host_seams_excluded": 7,
                "missing": [] if attention_complete else ["layers.0.attn.wkv.weight"],
                "unexpected": [],
                "complete": attention_complete,
            },
        },
        "moe": {
            "classes": moe_classes,
            "launch_compatibility": [
                {"class": "routed_experts_gate_up", "launch_compatible": True},
                {"class": "routed_experts_down", "launch_compatible": False, "reason": "A8 pre-padded 640-K row layout"},
                {"class": "shared_experts", "launch_compatible": True},
                {"class": "router", "launch_compatible": True},
            ],
            "weight_h2d_bytes_per_token": 0,
        },
        "tail": {"tensors_bound": 3, "pointers_nonzero": True, "coverage": {"class": "tail", "expected": 3, "bound": 3, "complete": tail_complete, "missing": [] if tail_complete else ["head"], "unexpected": []}},
        "census": {"resident_tensors": 95818, "by_class": {"attention_projection": 208, "routed_expert": 92160}, "unclassified": census_unclassified or []},
        "error": error,
        "summary": {
            "device_bound_names": 92729 if moe_complete and attention_complete else 90000,
            "per_token_weight_h2d_bytes": 0,
            "complete": complete if summary_complete is None else summary_complete,
            "check": "set equality per class, both directions named",
        },
        "names_manifest": "tp4-bind-names-r" + str(rank) + ".txt",
        "note": "progressive bind evidence, written before the walk",
    }
    return write_json(path, doc)


def oracle_doc(positions, *, root=RELEASED_ROOT, prompt_id="p", target_token=None):
    rows = []
    for position, logprobs, target in positions:
        row = {"position": position, "logprobs": {str(key): value for key, value in logprobs.items()}}
        if target is not None:
            row["target_token"] = target
        rows.append(row)
    return {
        "schema": compare.ORACLE_SCHEMA,
        "served": {"id": "deepseek-v4.1-flash", "root": root},
        "prompts": [{"id": prompt_id, "status": "ok", "prompt_len": 100, "positions": rows}],
    }


def logprobs_from_logits(logits):
    logs = compare.log_softmax(logits)
    return {index: value for index, value in enumerate(logs)}


def token_evidence(path, tokens, *, positions=None, rank=0, emitted=True):
    doc = {
        "schema": accept.CANONICAL_SCHEMA,
        "kind": "device",
        "rank": rank,
        "world_size": 4,
        "prompt_tokens": [1, 2, 3],
        "token_ids": tokens,
        "token_emitted": emitted,
    }
    if positions is not None:
        doc["positions"] = positions
    return write_json(path, doc)


class AcceptanceTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.out = os.path.join(self.root, "out.json")

    def run_cli(self, *argv, out=None):
        out = out or self.out
        code = accept.main([*argv, "--out-json", out])
        doc = None
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
        return code, doc


    def make_loop_evidence(self, *, bound=215, expected=215, calls_per_step=80, steps=1, layers=40, rule=None, plan_only=False):
        walk = {
            "performed": True,
            "phase": "collective-skeleton",
            "layers": layers,
            "calls_per_step_expected": 2 * layers,
            "warmup_steps": 0,
            "steady_steps": steps,
            "steps": [{"step": step, "warm": False, "calls": calls_per_step, "calls_per_step_ok": True} for step in range(steps)],
            "canonical_rule": "bf16 payload, ascending-rank f32 sum, one bf16 RNE round",
            "phase_order": ["attn", "moe"],
        }
        if rule is not None:
            walk["canonical_bf16_check"] = rule
        doc = {
            "stage": "tp4-chain",
            "schema": "sparkflash-tp4-loop/1",
            "rank": 0,
            "world": 4,
            "tokens": 0,
            "plan_only": plan_only,
            "attention": {"tensors_bound": bound, "tensors_expected": expected},
            "collectives": walk,
            "step": {},
        }
        loop_path = write_json(os.path.join(self.root, "ev", "tp4-loop-r0.json"), doc)
        if not plan_only:
            write_bind(os.path.join(self.root, "ev", "tp4-bind-r0.json"), 0, attention_bound=bound, attention_expected=expected)
        return loop_path



class ExtractEvidenceTest(AcceptanceTestBase):
    def test_canonical_token_evidence_parses(self):
        path = token_evidence(os.path.join(self.root, "ev.json"), [3, 2], positions=[1, 2])
        evidence = accept.extract_evidence([path], expect_vocab=VOCAB, side="device")
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["tokens"], [3, 2])
        self.assertEqual(evidence[0]["positions"], [1, 2])
        self.assertTrue(evidence[0]["token_emitted"])

    def test_token_outside_vocab_is_refused(self):
        path = token_evidence(os.path.join(self.root, "ev.json"), [9])
        with self.assertRaises(accept.AcceptError):
            accept.extract_evidence([path], expect_vocab=VOCAB, side="device")

    def test_tp4_host_uses_generated_tokens_not_step_tokens(self):
        doc = {
            "stage": "tp4-host",
            "schema": 1,
            "rank": 0,
            "world": 4,
            "prompt_tokens": [1, 2, 3],
            "generated_tokens": [3, 2, 1],
            "step_tokens": [9, 9, 9, 9, 9],
            "timing": {"step_ms": [10.0, 10.0], "mean_step_ms": 10.0},
        }
        path = write_json(os.path.join(self.root, "twin.json"), doc)
        evidence = accept.extract_evidence([path], expect_vocab=VOCAB, side="twin")
        self.assertEqual(evidence[0]["tokens"], [3, 2, 1])
        self.assertEqual(evidence[0]["kind"], "tp4-host")
        self.assertEqual(evidence[0]["step_ms"], [10.0, 10.0])

    def test_tp4_chain_step_schema_tokens_generated(self):
        doc = {
            "stage": "tp4-chain",
            "schema": "sparkflash-tp4-chain/2",
            "rank": 1,
            "world": 4,
            "tokens": {"start_position": 7, "prompt_tokens": 8, "generated": [3, 2]},
            "step": {"step_ms_samples": [5.0, 6.0], "first_token_ms": 4.0, "layers_run": 40},
        }
        path = write_json(os.path.join(self.root, "r1.json"), doc)
        evidence = accept.extract_evidence([path], expect_vocab=VOCAB, side="device")
        self.assertEqual(evidence[0]["tokens"], [3, 2])
        self.assertEqual(evidence[0]["positions"], [7, 8])
        self.assertEqual(evidence[0]["step_ms"], [5.0, 6.0])
        self.assertEqual(evidence[0]["first_token_ms"], 4.0)
        self.assertEqual(evidence[0]["rank"], 1)


class TwinMatchTest(AcceptanceTestBase):
    def test_identical_token_ids_pass(self):
        device = token_evidence(os.path.join(self.root, "dev.json"), [3, 2])
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3, 2])
        code, doc = self.run_cli("--device", device, "--twin", twin)
        self.assertEqual(code, 0)
        self.assertEqual(doc["twin_match"]["verdict"], "PASS")
        self.assertEqual(doc["twin_match"]["mismatches"], 0)
        self.assertEqual(doc["overall"]["verdict"], "PASS")

    def test_mismatch_fails_the_run(self):
        device = token_evidence(os.path.join(self.root, "dev.json"), [3, 2])
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3, 1])
        code, doc = self.run_cli("--device", device, "--twin", twin)
        self.assertEqual(code, 1)
        self.assertEqual(doc["twin_match"]["verdict"], "FAIL")
        self.assertEqual(doc["twin_match"]["mismatches"], 1)

    def test_token_only_evidence_is_aligned_by_index(self):
        device = token_evidence(os.path.join(self.root, "dev.json"), [3, 2])
        twin_dir = os.path.join(self.root, "twin")
        write_json(os.path.join(twin_dir, "tp4-host-r0.json"), {
            "stage": "tp4-host", "rank": 0, "world": 4,
            "prompt_tokens": [1, 2, 3], "generated_tokens": [3, 2],
        })
        code, doc = self.run_cli("--device", device, "--twin", twin_dir)
        self.assertEqual(code, 0)
        self.assertEqual(doc["twin_match"]["mode"], "index")
        self.assertEqual(doc["twin_match"]["verdict"], "PASS")

    def test_positions_mode_when_both_sides_name_positions(self):
        device = token_evidence(os.path.join(self.root, "dev.json"), [3], positions=[7])
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3], positions=[7])
        code, doc = self.run_cli("--device", device, "--twin", twin)
        self.assertEqual(code, 0)
        self.assertEqual(doc["twin_match"]["mode"], "positions")

    def test_no_twin_is_unmeasured_not_fail(self):
        device = token_evidence(os.path.join(self.root, "dev.json"), [3])
        code, doc = self.run_cli("--device", device)
        self.assertEqual(code, 3)
        self.assertEqual(doc["twin_match"]["verdict"], "UNMEASURED")
        self.assertEqual(doc["overall"]["verdict"], "UNMEASURED")

    def test_rank_disagreement_fails(self):
        one = token_evidence(os.path.join(self.root, "dev-r0.json"), [3], rank=0)
        two = token_evidence(os.path.join(self.root, "dev-r1.json"), [2], rank=1)
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3])
        code, doc = self.run_cli("--device", one, "--device", two, "--twin", twin)
        self.assertEqual(code, 1)
        self.assertEqual(doc["twin_match"]["verdict"], "FAIL")


class PreTokenTest(AcceptanceTestBase):
    def test_pre_token_checks_pass_and_overall_is_not_attempted(self):
        path = self.make_loop_evidence()
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(code, 3)
        self.assertEqual(doc["pre_token"]["verdict"], "PASS")
        self.assertEqual(doc["twin_match"]["verdict"], "NOT_ATTEMPTED")
        self.assertEqual(doc["overall"]["verdict"], "NOT_ATTEMPTED")

    def test_bind_coverage_mismatch_fails_the_run(self):
        path = self.make_loop_evidence(bound=208)
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(doc["pre_token"]["verdict"], "FAIL")
        self.assertEqual(doc["shape"]["verdict"], "PASS")  # shape is fine; the pre-token check fails
        self.assertEqual(doc["overall"]["verdict"], "FAIL")
        self.assertEqual(code, 1)

    def test_missing_primary_bind_evidence_fails_loudly(self):
        directory = os.path.join(self.root, "ev")
        write_json(os.path.join(directory, "tp4-loop-r0.json"), {
            "stage": "tp4-chain", "schema": "sparkflash-tp4-loop/1", "rank": 0, "world": 4,
            "plan_only": False, "tokens": 0,
            "attention": {"tensors_bound": 208, "tensors_expected": 208},
            "collectives": {"performed": True, "layers": 40, "calls_per_step_expected": 80, "steps": [{"step": 0, "calls": 80, "calls_per_step_ok": True}]},
        })
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 1)
        self.assertEqual(doc["shape"]["verdict"], "FAIL")
        self.assertEqual(doc["overall"]["verdict"], "FAIL")
        self.assertTrue(any("primary bind evidence" in check["name"] and check["status"] == "fail" for check in doc["shape"]["checks"]))
        self.assertEqual(doc["pre_token"]["verdict"], "FAIL")
        self.assertTrue(any("primary bind file present" in check["name"] and check["status"] == "fail" for check in doc["pre_token"]["checks"]))

    def test_incomplete_bind_file_fails_pre_token(self):
        path = self.make_loop_evidence(bound=215)
        write_bind(os.path.join(os.path.dirname(path), "tp4-bind-r0.json"), 0, attention_bound=214, attention_expected=215)
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(code, 1)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        self.assertEqual(doc["pre_token"]["verdict"], "FAIL")
        self.assertEqual(doc["overall"]["verdict"], "FAIL")

    def test_bind_error_is_reported(self):
        path = self.make_loop_evidence(bound=215)
        write_bind(os.path.join(os.path.dirname(path), "tp4-bind-r0.json"), 0, attention_bound=215, attention_expected=215, error="bind stage failed: out of memory")
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(code, 1)
        self.assertEqual(doc["pre_token"]["verdict"], "FAIL")
        self.assertTrue(any("bind error" in check["name"] for check in doc["pre_token"]["checks"]))

    def test_plan_only_run_is_not_attempted_not_a_bind_failure(self):
        path = self.make_loop_evidence(bound=0, plan_only=True)
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(doc["pre_token"]["verdict"], "NOT_ATTEMPTED")
        self.assertEqual(doc["overall"]["verdict"], "NOT_ATTEMPTED")
        self.assertEqual(code, 3)

    def test_collective_count_mismatch_fails_pre_token(self):
        path = self.make_loop_evidence(calls_per_step=79)
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(doc["pre_token"]["verdict"], "FAIL")

    def test_canonical_bf16_mismatch_fails_pre_token(self):
        path = self.make_loop_evidence(rule={"mismatches": 3})
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(doc["pre_token"]["verdict"], "FAIL")

    def test_hybrid_dump_is_not_promoted_by_a_directory_scan(self):
        path = self.make_loop_evidence()
        dump = os.path.join(os.path.dirname(path), "hybrid-dump-logits.jsonl")
        write_dump(dump, [(0, [0.0, 1.0, 2.0, 3.0], None)])
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertIsNone(doc["inputs"]["device_dump"])
        ignored = [entry["path"] for entry in doc["inputs"]["ignored_files"]]
        self.assertIn(os.path.abspath(dump), ignored)


class OracleTest(AcceptanceTestBase):
    def test_token_only_oracle_leg(self):
        logprobs = logprobs_from_logits([0.0, 1.0, 2.0, 3.0])
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([
            (7, logprobs, 3),
            (8, logprobs, 3),
        ]))
        device = token_evidence(os.path.join(self.root, "dev.json"), list(logprobs.keys())[:0] + [3, 3], positions=[7, 8])
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3, 3])
        code, doc = self.run_cli("--device", device, "--twin", twin, "--oracle", oracle)
        self.assertEqual(code, 0)
        self.assertEqual(doc["oracle_comparison"]["verdict"], "UNBARED")
        self.assertEqual(doc["oracle_comparison"]["top1_hit_rate"], 1.0)
        self.assertEqual(doc["oracle_comparison"]["label"], "ORACLE-TOKEN-ONLY (no device logits dump; distributional bars cannot be evaluated)")

    def test_token_only_oracle_without_positions_is_unmeasured(self):
        logprobs = logprobs_from_logits([0.0, 1.0, 2.0, 0.5])
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([(7, logprobs, 3)]))
        device = token_evidence(os.path.join(self.root, "dev.json"), [3])
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3])
        code, doc = self.run_cli("--device", device, "--twin", twin, "--oracle", oracle)
        self.assertEqual(doc["oracle_comparison"]["verdict"], "UNMEASURED")
        self.assertIn("no explicit positions", doc["oracle_comparison"]["reasons"][0])

    def test_oracle_wrong_root_is_refused(self):
        logprobs = logprobs_from_logits([0.0, 1.0, 2.0, 0.5])
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([(7, logprobs, 3)], root="/cache/other/snapshots/deadbeef"))
        device = token_evidence(os.path.join(self.root, "dev.json"), [3], positions=[7])
        code, doc = self.run_cli("--device", device, "--oracle", oracle)
        self.assertEqual(code, 2)
        self.assertIsNone(doc)

    def test_dump_leg_is_unbared_without_gate_file(self):
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([
            (0, logprobs_from_logits([0.0, 1.0, 2.0, 3.0]), 3),
            (1, logprobs_from_logits([3.0, 2.0, 1.0, 0.0]), 0),
        ]))
        dump = write_dump(os.path.join(self.root, "logits.jsonl"), [
            (0, [0.0, 1.0, 2.0, 3.0], 3),
            (1, [3.0, 2.0, 1.0, 0.0], 0),
        ])
        device = token_evidence(os.path.join(self.root, "dev.json"), [3, 0])
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3, 0])
        code, doc = self.run_cli(
            "--device", device, "--twin", twin, "--device-dump", dump,
            "--oracle", oracle, "--expect-vocab", str(VOCAB), "--positions", "0,1",
        )
        self.assertEqual(code, 0)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        self.assertEqual(doc["oracle_comparison"]["verdict"], "UNBARED")
        self.assertEqual(doc["oracle_comparison"]["aggregate"]["positions"], 2)
        self.assertEqual(doc["oracle_comparison"]["aggregate"]["top1_agreement_rate"], 1.0)
        self.assertEqual(doc["oracle_comparison"]["aggregate"]["prefix_mismatches"], 0)

    def test_dump_prefix_mismatch_is_not_like_for_like(self):
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([
            (0, logprobs_from_logits([0.0, 1.0, 2.0, 3.0]), 3),
        ]))
        dump = write_dump(os.path.join(self.root, "logits.jsonl"), [(0, [0.0, 1.0, 2.0, 3.0], 1)])
        device = token_evidence(os.path.join(self.root, "dev.json"), [3])
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3])
        code, doc = self.run_cli(
            "--device", device, "--twin", twin, "--device-dump", dump,
            "--oracle", oracle, "--expect-vocab", str(VOCAB),
        )
        self.assertEqual(code, 1)
        self.assertEqual(doc["oracle_comparison"]["verdict"], "NOT_LIKE_FOR_LIKE")
        self.assertEqual(doc["overall"]["verdict"], "FAIL")

    def test_provisional_d40_is_labelled_not_a_gate(self):
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([
            (0, logprobs_from_logits([0.0, 1.0, 2.0, 3.0]), 3),
        ]))
        dump = write_dump(os.path.join(self.root, "logits.jsonl"), [(0, [0.0, 1.0, 2.0, 3.0], 3)])
        device = token_evidence(os.path.join(self.root, "dev.json"), [3])
        code, doc = self.run_cli(
            "--device", device, "--device-dump", dump, "--oracle", oracle,
            "--expect-vocab", str(VOCAB), "--provisional-d40",
        )
        self.assertEqual(doc["oracle_comparison"]["provisional_d40"]["label"], "D40-PROVISIONAL-NOT-A-GATE")

    def test_requires_review_bars_file_is_refused(self):
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([
            (0, logprobs_from_logits([0.0, 1.0, 2.0, 3.0]), 3),
        ]))
        dump = write_dump(os.path.join(self.root, "logits.jsonl"), [(0, [0.0, 1.0, 2.0, 3.0], 3)])
        bars = write_json(os.path.join(self.root, "bars.json"), {
            "schema": compare.GATE_BARS_SCHEMA, "status": "requires_review",
            "bar_class": "idle_distractor", "bars": {"top1_agreement_min": 0.9},
        })
        device = token_evidence(os.path.join(self.root, "dev.json"), [3])
        code, doc = self.run_cli(
            "--device", device, "--device-dump", dump, "--oracle", oracle,
            "--expect-vocab", str(VOCAB), "--oracle-bars", bars,
        )
        self.assertEqual(code, 2)
        self.assertIsNone(doc)


class DumpShapeTest(AcceptanceTestBase):
    def test_bad_sidecar_is_refused(self):
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([
            (0, logprobs_from_logits([0.0, 1.0, 2.0, 3.0]), 3),
        ]))
        dump = write_dump(os.path.join(self.root, "logits.jsonl"), [(0, [0.0, 1.0, 2.0, 3.0], 3)])
        os.remove(dump + ".bin")
        device = token_evidence(os.path.join(self.root, "dev.json"), [3])
        code, doc = self.run_cli(
            "--device", device, "--device-dump", dump, "--oracle", oracle, "--expect-vocab", str(VOCAB),
        )
        self.assertEqual(code, 2)
        self.assertIsNone(doc)

    def test_pinned_positions_missing_fails_shape(self):
        oracle = write_json(os.path.join(self.root, "oracle.json"), oracle_doc([
            (0, logprobs_from_logits([0.0, 1.0, 2.0, 3.0]), 3),
        ]))
        dump = write_dump(os.path.join(self.root, "logits.jsonl"), [(0, [0.0, 1.0, 2.0, 3.0], 3)])
        device = token_evidence(os.path.join(self.root, "dev.json"), [3])
        code, doc = self.run_cli(
            "--device", device, "--device-dump", dump, "--oracle", oracle,
            "--expect-vocab", str(VOCAB), "--positions", "0,1",
        )
        self.assertEqual(code, 2)  # the oracle leg refuses a position the dump does not have


class WindowContractTest(AcceptanceTestBase):
    def test_expected_attention_bound_and_moe_counts(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        code, doc = self.run_cli(
            "--device", os.path.dirname(path),
            "--expect-attention-bound", "208",
            "--expect-moe-bounds", "routed_experts_gate_up=61440,routed_experts_down=30720,shared_experts=240,router=120",
        )
        self.assertEqual(code, 3)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        self.assertEqual(doc["pre_token"]["verdict"], "PASS")
        self.assertEqual(doc["pre_token"]["bind_ranks"], [0])

    def test_wrong_attention_bound_fails_shape(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        code, doc = self.run_cli("--device", os.path.dirname(path), "--expect-attention-bound", "999")
        self.assertEqual(code, 1)
        self.assertEqual(doc["shape"]["verdict"], "FAIL")
        self.assertTrue(any("attention bound == 999" in check["name"] and check["status"] == "fail" for check in doc["shape"]["checks"]))

    def test_wrong_moe_bound_fails_shape(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        code, doc = self.run_cli("--device", os.path.dirname(path), "--expect-moe-bounds", "routed_experts_gate_up=1")
        self.assertEqual(code, 1)
        self.assertTrue(any("MoE class bound counts" in check["name"] and check["status"] == "fail" for check in doc["shape"]["checks"]))

    def test_manifest_line_count(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        manifest = os.path.join(directory, "tp4-bind-names-r0.txt")
        write_text(manifest, ("a" + chr(10)) * 5)
        code, doc = self.run_cli("--device", directory, "--expect-manifest-lines", "5")
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        code, doc = self.run_cli("--device", directory, "--expect-manifest-lines", "6")
        self.assertEqual(code, 1)
        self.assertTrue(any("manifest lines" in check["name"] and check["status"] == "fail" for check in doc["shape"]["checks"]))

    def test_run_log_defect_fails_shape(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        log = os.path.join(directory, "run.log")
        write_text(log, "tp4-chain r0 loop: bind 208/208" + chr(10) + "collective walk ok" + chr(10))
        write_text(os.path.join(directory, "run.exit"), "0" + chr(10))
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        with open(log, "a", encoding="utf-8") as handle:
            handle.write("tp4-chain r0 canonical rule mismatch on layer 12" + chr(10))
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 1)
        self.assertTrue(any("run log clean" in check["name"] and check["status"] == "fail" for check in doc["shape"]["checks"]))

    def test_nonzero_run_exit_fails_shape(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        write_text(os.path.join(directory, "run.log"), "ok" + chr(10))
        write_text(os.path.join(directory, "run.exit"), "2" + chr(10))
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 1)
        self.assertTrue(any("rank exit code" in check["name"] and check["status"] == "fail" for check in doc["shape"]["checks"]))

    def test_wrapper_metadata_json_is_auxiliary_not_a_shape_failure(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        with open(os.path.join(directory, "wrapper-timeline.json"), "w", encoding="utf-8") as handle:
            json.dump({"stage": "wrapper", "note": "rank timeline metadata from the run wrapper"}, handle)
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 3)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        self.assertTrue(any("auxiliary evidence ignored" in check["name"] and check["status"] == "info" for check in doc["shape"]["checks"]))

    def test_adjudication_json_is_auxiliary_not_a_shape_failure(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        with open(os.path.join(directory, "bind-window-adjudication.json"), "w", encoding="utf-8") as handle:
            json.dump({"schema": "sparkflash-bind-window-adjudication/2", "status": "PASS"}, handle)
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 3)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        self.assertTrue(any("auxiliary evidence ignored" in check["name"] and check["status"] == "info" for check in doc["shape"]["checks"]))

    def test_chain_named_log_and_rc_are_accepted(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        write_text(os.path.join(directory, "chain-r0.log"), "tp4-chain r0 bind: progressive evidence written" + chr(10))
        write_text(os.path.join(directory, "chain-r0.rc"), "0" + chr(10))
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 3)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        names = [check["name"] for check in doc["shape"]["checks"]]
        self.assertTrue(any("run log clean" in name for name in names))
        self.assertTrue(any("rank exit code" in name for name in names))

    def test_benign_log_wording_is_reported_not_failed(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        write_text(os.path.join(directory, "run.log"), "tp4-chain r0 bind done, no error observed" + chr(10))
        write_text(os.path.join(directory, "run.exit"), "0" + chr(10))
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 3)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        self.assertTrue(any("suspicious log wording" in check["name"] and check["status"] == "info" for check in doc["shape"]["checks"]))

    def test_resident_stage_evidence_is_auxiliary_not_a_shape_failure(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        with open(os.path.join(directory, "tp4-resident-r0.json"), "w", encoding="utf-8") as handle:
            json.dump({"schema": 1, "stage": "tp4-resident", "rank": 0, "world": 4, "plan": {"hash": "0x9a59035fc40776c8"}}, handle)
        with open(os.path.join(directory, "tp4-chain-summary.json"), "w", encoding="utf-8") as handle:
            json.dump({"schema": "sparkflash-tp4-chain/1", "world": 4, "note": "rank 0 summary"}, handle)
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 3)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        self.assertEqual(doc["pre_token"]["verdict"], "PASS")
        self.assertTrue(any("auxiliary evidence ignored" in check["name"] and check["status"] == "info" for check in doc["shape"]["checks"]))

    def test_expected_unclassified_census_names_are_not_a_failure(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        write_bind(os.path.join(os.path.dirname(path), "tp4-bind-r0.json"), 0, attention_bound=208, attention_expected=208,
                   census_unclassified=["image_end", "image_newline", "image_start"])
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(code, 3)
        self.assertEqual(doc["pre_token"]["verdict"], "PASS")
        checks = doc["pre_token"]["checks"]
        census = [check for check in checks if check["name"].endswith("census")]
        self.assertEqual(census[0]["status"], "ok")
        self.assertIn("expected multi-modal", census[0]["detail"])
        caveat = [check for check in checks if check["name"].endswith("census caveat")]
        self.assertEqual(caveat[0]["status"], "info")
        self.assertIn("image_end", caveat[0]["detail"])
        complete = [check for check in checks if check["name"].endswith("r0: bind complete")]
        self.assertEqual(complete[0]["status"], "info")
        self.assertIn("by construction", complete[0]["detail"])

    def test_clean_bind_shape_with_summary_complete_true_passes(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        write_bind(os.path.join(os.path.dirname(path), "tp4-bind-r0.json"), 0, attention_bound=208, attention_expected=208)
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(code, 3)
        self.assertEqual(doc["pre_token"]["verdict"], "PASS")
        checks = doc["pre_token"]["checks"]
        complete = [check for check in checks if check["name"].endswith("r0: bind complete")]
        self.assertEqual(complete[0]["status"], "ok")
        self.assertFalse([check for check in checks if check["name"].endswith("census caveat")])

    def test_unexpected_unclassified_census_name_fails(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        write_bind(os.path.join(os.path.dirname(path), "tp4-bind-r0.json"), 0, attention_bound=208, attention_expected=208,
                   census_unclassified=["image_end", "mystery_weight"])
        code, doc = self.run_cli("--device", os.path.dirname(path))
        self.assertEqual(code, 1)
        self.assertEqual(doc["pre_token"]["verdict"], "FAIL")
        census = [check for check in doc["pre_token"]["checks"] if check["name"].endswith("census")]
        self.assertEqual(census[0]["status"], "fail")
        self.assertIn("mystery_weight", census[0]["detail"])
        complete = [check for check in doc["pre_token"]["checks"] if check["name"].endswith("r0: bind complete")]
        self.assertEqual(complete[0]["status"], "fail")

    def test_malformed_moe_bounds_is_an_input_error(self):
        path = self.make_loop_evidence()
        code = accept.main(["--device", os.path.dirname(path), "--expect-moe-bounds", "oops", "--out-json", self.out])
        self.assertEqual(code, 2)


class StepTimeTest(AcceptanceTestBase):
    def test_step_time_is_measured_and_target_is_explicitly_not_the_gate(self):
        doc = {
            "stage": "tp4-chain",
            "schema": "sparkflash-tp4-chain/2",
            "rank": 0,
            "world": 4,
            "tokens": {"start_position": 7, "prompt_tokens": 8, "generated": [3]},
            "step": {"step_ms_samples": [100.0, 120.0], "first_token_ms": 90.0, "layers_run": 40},
        }
        path = write_json(os.path.join(self.root, "ev", "tp4-loop-r0.json"), doc)
        twin = token_evidence(os.path.join(self.root, "twin.json"), [3])
        code, doc = self.run_cli("--device", os.path.dirname(path), "--twin", twin)
        self.assertEqual(code, 0)
        self.assertEqual(doc["step_time"]["verdict"], "MEASURED")
        self.assertAlmostEqual(doc["step_time"]["step_ms_mean"], 110.0)
        self.assertEqual(doc["step_time"]["target_only"]["step_budget_ms"], 42.0)
        self.assertFalse(doc["step_time"]["target_only"]["step_budget_pass_as_measured"])
        self.assertIn("TARGET-ONLY", doc["step_time"]["label"])


class InputTest(AcceptanceTestBase):
    def test_no_device_is_an_input_error(self):
        code = accept.main(["--out-json", self.out])
        self.assertEqual(code, 2)

    def test_unrecognised_evidence_layout_fails_shape(self):
        path = write_json(os.path.join(self.root, "ev.json"), {"not": "evidence"})
        code, doc = self.run_cli("--device", path)
        self.assertEqual(doc["shape"]["verdict"], "FAIL")
        self.assertEqual(code, 1)

    def test_duplicate_positions_are_refused(self):
        with self.assertRaises(compare.CompareError):
            compare.parse_positions("1,1")



class RefusedForwardEvidenceTest(AcceptanceTestBase):
    """The m1w labelled negative: forward=refused with rc=2, plus device-lead's forward-embed files.

    Only that exact envelope shape is accepted: forward=="refused", positions_run==0, named error.
    The forward-embed layout is reported as pre-token evidence, and a plan-only world-1 resident
    document is auxiliary and must not enter the run's world identity.
    """

    def write_refused_rank(self, *, rank=0, rc_text="2", forward="refused", include_embed=True, include_aux=True, error="four-rank forward loop: the combine preflight refused before any routed expert launched"):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        envelope = {
            "schema": "sparkflash-tp4-token/1",
            "stage": "tp4-chain",
            "loop_step": "C4-token",
            "rank": rank,
            "world": 4,
            "forward": forward,
            "tokens": {"ids": [], "positions": [], "start_position": None, "prompt_len": None},
            "positions_run": 0,
            "step_ms": {"samples": [], "mean_ms": 0.0, "worst_ms": 0.0},
            "wall_ms": [],
            "reduce_calls_per_step": 0,
            "logits_dump": None,
            "error": error,
            "hybrid_sequence_executed": False,
        }
        write_json(os.path.join(directory, "tp4-token-r%d.json" % rank), envelope)
        if include_embed:
            write_json(
                os.path.join(directory, "tp4-forward-embed-r%d.json" % rank),
                {
                    "stage": "embedding-gathered",
                    "rank": rank,
                    "world": 4,
                    "prompt_token": 20462,
                    "prompt_len": 130,
                    "first_step_position": 129,
                    "decode_positions": 9,
                    "reference_ids": [491, 367, 201, 80, 3302, 201, 54, 201, 4341],
                    "prefill_tokens": 130,
                    "prefill_exchange_calls": 130,
                    "token_claimed": False,
                    "attention_launchable": 5,
                    "attention_blocked": [],
                    "note": "refused before a token; this is pre-token evidence",
                },
            )
        if include_aux:
            write_json(
                os.path.join(directory, "plan-world1", "tp4-resident-r0.json"),
                {"schema": 1, "stage": "tp4-resident", "rank": 0, "world": 1, "plan": {"hash": "0x9a59035fc40776c8"}},
            )
        write_text(os.path.join(directory, "chain-r%d.rc" % rank), rc_text + "\n")
        return directory

    def test_refused_forward_with_rc2_passes_shape_and_stays_pre_token(self):
        directory = self.write_refused_rank()
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 3)
        self.assertEqual(doc["shape"]["verdict"], "PASS")
        self.assertEqual(doc["pre_token"]["verdict"], "PASS")
        names = {check["name"]: check for check in doc["shape"]["checks"]}
        self.assertEqual(names["world size"]["status"], "ok")
        self.assertEqual(names["world size"]["detail"], "world [4]")
        self.assertEqual(names["forward-embed evidence"]["status"], "ok")
        exit_checks = [item for item in doc["shape"]["checks"] if item["name"].endswith(": rank exit code")]
        self.assertEqual(len(exit_checks), 1)
        self.assertEqual(exit_checks[0]["status"], "info")
        self.assertIn("labelled forward refusal", exit_checks[0]["detail"])
        self.assertEqual(names["r0: token forward refused"]["status"], "info")
        self.assertIn("auxiliary evidence ignored", names)

    def test_rc2_without_refusal_envelope_still_fails(self):
        path = self.make_loop_evidence(bound=208, expected=208)
        directory = os.path.dirname(path)
        write_text(os.path.join(directory, "chain-r0.rc"), "2\n")
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 1)
        self.assertEqual(doc["shape"]["verdict"], "FAIL")
        check = [item for item in doc["shape"]["checks"] if item["name"].endswith(": rank exit code")][0]
        self.assertEqual(check["status"], "fail")

    def test_rc2_with_non_refused_envelope_still_fails(self):
        directory = self.write_refused_rank(forward="run")
        code, doc = self.run_cli("--device", directory)
        self.assertEqual(code, 1)
        self.assertEqual(doc["shape"]["verdict"], "FAIL")
        check = [item for item in doc["shape"]["checks"] if item["name"].endswith(": rank exit code")][0]
        self.assertEqual(check["status"], "fail")


if __name__ == "__main__":
    unittest.main()
