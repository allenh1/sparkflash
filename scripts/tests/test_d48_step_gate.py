#!/usr/bin/env python3
"""Unit tests for scripts/d48_step_gate.py on synthetic per-rank evidence.

No GPU, no network, no real checkpoint. Run with:
    python3 -m unittest discover -s scripts/tests -v
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

import d48_step_gate as gate

VOCAB = 129280


def write_json(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)
    return path


def loop_doc(rank, *, bound=208, expected=208, walk_steps=3, per_step_calls=80, plan_only=False):
    return {
        "stage": "tp4-chain",
        "schema": "sparkflash-tp4-loop/1",
        "rank": rank,
        "world": 4,
        "tokens": 0,
        "forward": "not-run",
        "plan_only": plan_only,
        "attention": {"tensors_bound": bound, "tensors_expected": expected},
        "collectives": {
            "performed": True,
            "layers": 40,
            "calls_per_step_expected": 80,
            "steps": [
                {"step": step, "calls": per_step_calls, "calls_per_step_ok": True}
                for step in range(walk_steps)
            ],
        },
        "hybrid_sequence": {"executed": False, "note": "no --cubin-dir: the hybrid-sequence runner was not invoked"},
        "plan": {"shards": 196, "resident_tensors": 95818, "disk_tensors": 0, "resident_bytes": 75480924160, "ranges": 1, "plan_hash": "0x9a59035fc40776c8"},
    }


def bind_doc(rank, *, bound=208, expected=208, complete=None, error=None, unclassified=None, moe_complete=True, summary_complete=None):
    """A sparkflash-tp4-bind/1 document with the window's confirmed class counts."""
    classes = [
        {"class": "routed_experts_gate_up", "expected": 61440, "bound": 61440 if moe_complete else 61439, "complete": moe_complete, "missing": [] if moe_complete else ["x"], "unexpected": []},
        {"class": "routed_experts_down", "expected": 30720, "bound": 30720, "complete": moe_complete, "missing": [] if moe_complete else ["x"], "unexpected": []},
        {"class": "shared_experts", "expected": 240, "bound": 240, "complete": moe_complete, "missing": [] if moe_complete else ["x"], "unexpected": []},
        {"class": "router", "expected": 120, "bound": 120, "complete": moe_complete, "missing": [] if moe_complete else ["x"], "unexpected": []},
    ]
    attention_complete = bound == expected and expected > 0
    all_complete = error is None and attention_complete and moe_complete and not (unclassified or [])
    return {
        "stage": "tp4-chain",
        "schema": "sparkflash-tp4-bind/1",
        "loop_step": "C1-bind",
        "setup_only": True,
        "rank": rank,
        "world": 4,
        "forward": "not-run",
        "tokens": 0,
        "attention": {
            "tensors_expected": expected,
            "tensors_bound": bound,
            "tensors_table_entries": 215,
            "sharded": True,
            "coverage": {
                "class": "attention",
                "expected": expected,
                "bound": bound,
                "host_seams_excluded": 7,
                "missing": [] if attention_complete else ["layers.0.attn.wkv.weight"],
                "unexpected": [],
                "complete": attention_complete,
                "expected_names": [],
            },
        },
        "moe": {
            "classes": classes,
            "launch_compatibility": [
                {"class": "routed_experts_gate_up", "launch_compatible": True},
                {"class": "routed_experts_down", "launch_compatible": False, "reason": "A8 pre-padded 640-K row layout"},
                {"class": "shared_experts", "launch_compatible": True},
                {"class": "router", "launch_compatible": True},
            ],
            "weight_h2d_bytes_per_token": 0,
        },
        "tail": {"tensors_bound": 3, "coverage": {"class": "tail", "expected": 3, "bound": 3, "complete": True, "missing": [], "unexpected": []}, "pointers_nonzero": True},
        "census": {"resident_tensors": 95818, "by_class": {"attention_projection": 208, "routed_expert": 92160}, "unclassified": unclassified or []},
        "error": error,
        "summary": {
            "device_bound_names": 92729 if moe_complete and attention_complete else 90000,
            "per_token_weight_h2d_bytes": 0,
            "complete": all_complete if summary_complete is None else summary_complete,
            "check": "set equality per class",
        },
        "names_manifest": f"tp4-bind-names-r{rank}.txt",
    }


def step_doc(
    rank,
    samples,
    *,
    layers=4,
    tokens=None,
    reduces=None,
    weight_bytes_per_token=0,
    per_layer=True,
    attention_bound=215,
    attention_expected=215,
):
    mean_ms = sum(samples) / len(samples)
    layer_total = round(mean_ms * 1000.0 / layers, 6)
    doc = {
        "stage": "tp4-chain",
        "schema": "sparkflash-tp4-chain/2",
        "rank": rank,
        "world": 4,
        "tokens": {
            "start_position": 8,
            "prompt_tokens": 8,
            "generated": tokens if tokens is not None else [3, 2, 1][: len(samples)],
        },
        "step": {
            "step_ms_samples": list(samples),
            "first_token_ms": 1234.5,
            "layers_run": layers,
            "reduces_posted": reduces if reduces is not None else 2 * layers * len(samples),
            "reduces_expected": 2 * layers * len(samples),
            "weight_bytes_per_token": weight_bytes_per_token,
        },
        "attention": {"tensors_bound": attention_bound, "tensors_expected": attention_expected},
    }
    if per_layer:
        doc["per_layer"] = [
            {
                "layer": layer,
                "attention": {"device_us": 100.0, "reduce_us": 10.0},
                "moe": {"device_us": 200.0, "reduce_us": 10.0},
                "total_us": layer_total,
            }
            for layer in range(layers)
        ]
    return doc


class StepGateTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.out = os.path.join(self.root, "out.json")

    def make_run(self, docs, *, subdirs=True, bind=True):
        run = os.path.join(self.root, "run")
        for rank, doc in docs.items():
            name = "tp4-loop-r%d.json" % rank
            directory = os.path.join(run, "rank%d" % rank) if subdirs else run
            write_json(os.path.join(directory, name), doc)
            if bind and doc.get("stage") == "tp4-chain" and doc.get("plan_only") is not True:
                write_json(os.path.join(directory, "tp4-bind-r%d.json" % rank), bind_doc(rank))
        return run

    def run_cli(self, *argv, out=None):
        out = out or self.out
        code = gate.main([*argv, "--out-json", out])
        doc = None
        if os.path.exists(out):
            with open(out, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
        return code, doc


class LoopOnlyTest(StepGateTestBase):
    def test_loop_only_run_is_not_attempted(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)})
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 3)
        self.assertEqual(doc["status"], "NOT-ATTEMPTED")
        self.assertEqual(doc["verdict"]["step_time_budget"], "UNMEASURED")
        self.assertEqual(doc["verdict"]["label"], "STEP-TIME-ONLY")
        self.assertEqual(len(doc["per_rank"]), 4)
        row = doc["per_rank"][0]
        self.assertEqual(row["attention_bound"], 208)
        self.assertEqual(row["attention_expected"], 208)
        self.assertTrue(row["walk_performed"])
        self.assertEqual(row["reduces_posted"], 240)
        self.assertEqual(row["reduces_expected"], 240)
        self.assertEqual(row["steps_run"], None)
        self.assertIn("no --cubin-dir", row["hybrid_sequence"]["note"])
        self.assertIsNotNone(row["bind"])
        self.assertTrue(row["bind"]["summary"]["complete"])
        self.assertEqual(row["plan"]["plan_hash"], "0x9a59035fc40776c8")
        self.assertEqual(row["plan"]["resident_tensors"], 95818)
        self.assertFalse(row["step_ms"]["samples"])

    def test_loop_only_has_no_failure_checks(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)})
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(doc["failures"], [])
        self.assertEqual(doc["verdict"]["accepted_tokens_per_step"], "NOT-WIRED (DSpark/MTP verify step not on the device path)")

    def test_auxiliary_evidence_does_not_crash_or_fail(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)})
        for rank in range(4):
            write_json(os.path.join(run, "rank%d" % rank, "tp4-resident-r%d.json" % rank),
                       {"schema": 1, "stage": "tp4-resident", "rank": rank, "world": 4, "plan": {"hash": "0x9a59035fc40776c8"}})
        write_json(os.path.join(run, "rank0", "tp4-chain-summary.json"),
                   {"schema": "sparkflash-tp4-chain/1", "world": 4, "note": "rank 0 summary"})
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 3)
        self.assertEqual(doc["status"], "NOT-ATTEMPTED")
        self.assertEqual(doc["failures"], [])
        self.assertTrue(any(check["name"] == "auxiliary evidence ignored" and check["status"] == "info" for check in doc["checks"]))

    def test_wrapper_metadata_json_is_info_only(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)})
        for rank in range(4):
            write_json(os.path.join(run, "rank%d" % rank, "timeline-r%d.json" % rank), {"stage": "wrapper", "rank": rank, "note": "wrapper timeline"})
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 3)
        self.assertEqual(doc["failures"], [])
        self.assertTrue(any(check["name"] == "auxiliary evidence ignored" and check["status"] == "info" for check in doc["checks"]))

    def test_chain_rc_is_read_as_rank_exit_code(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)})
        for rank in range(4):
            with open(os.path.join(run, "rank%d" % rank, "chain-r%d.rc" % rank), "w", encoding="utf-8") as handle:
                handle.write("0" + chr(10))
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 3)
        row = doc["per_rank"][0]
        self.assertEqual(row["rc"], 0)
        self.assertEqual(row["rc_source"], "chain-r0.rc")

    def test_missing_primary_bind_is_a_loud_failure(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)}, bind=False)
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertEqual(doc["status"], "BIND-FAIL")
        self.assertIn("primary bind evidence on every rank", doc["failures"])

    def test_incomplete_bind_is_a_loud_failure(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)}, bind=False)
        for rank in range(4):
            write_json(os.path.join(run, "rank%d" % rank, "tp4-bind-r%d.json" % rank), bind_doc(rank, bound=207))
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertEqual(doc["status"], "BIND-FAIL")
        self.assertIn("bind attention coverage complete", doc["failures"])

    def test_expected_unclassified_census_names_do_not_fail(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)}, bind=False)
        for rank in range(4):
            write_json(os.path.join(run, "rank%d" % rank, "tp4-bind-r%d.json" % rank),
                       bind_doc(rank, unclassified=["image_end", "image_newline", "image_start"]))
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 3)
        self.assertEqual(doc["failures"], [])
        self.assertEqual(doc["status"], "NOT-ATTEMPTED")
        names = {check["name"] for check in doc["checks"]}
        self.assertIn("bind census expected unclassified names", names)
        self.assertIn("bind census caveat", names)
        caveat = [check for check in doc["checks"] if check["name"] == "bind census caveat"][0]
        self.assertIn("image_end", caveat["detail"])
        summary = [check for check in doc["checks"] if check["name"] == "bind summary complete (census caveat allowed)"][0]
        self.assertIn(summary["status"], ("pass", "info", True, None))

    def test_unexpected_unclassified_census_name_fails(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)}, bind=False)
        for rank in range(4):
            write_json(os.path.join(run, "rank%d" % rank, "tp4-bind-r%d.json" % rank),
                       bind_doc(rank, unclassified=["image_end", "mystery_weight"]))
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertEqual(doc["status"], "BIND-FAIL")
        self.assertIn("bind summary complete (census caveat allowed)", doc["failures"])
        self.assertIn("bind census classified (expected unclassified names only)", doc["failures"])

    def test_bind_launch_incompatibility_is_reported_not_failed(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)})
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 3)
        self.assertEqual(doc["failures"], [])
        compat = [check for check in doc["checks"] if check["name"] == "bind launch compatibility"]
        self.assertEqual(len(compat), 1)
        self.assertEqual(compat[0]["status"], "info")
        self.assertIn("routed_experts_down", compat[0]["detail"])

    def test_plan_only_loop_run_reports_no_bind_failure(self):
        run = self.make_run({rank: loop_doc(rank, bound=0, plan_only=True) for rank in range(4)})
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 3)
        self.assertEqual(doc["failures"], [])
        self.assertIn({"name": "bind coverage bound==expected", "status": "info", "detail": "plan-only run: binding was not attempted"}, doc["checks"])

    def test_walk_mismatch_shows_up_in_reduce_check(self):
        docs = {rank: loop_doc(rank) for rank in range(4)}
        docs[2]["collectives"]["steps"][0]["calls"] = 79
        run = self.make_run(docs)
        code, doc = self.run_cli("--device-dir", run)
        self.assertIn("reduces posted == 2 per layer per step", doc["failures"])


class StepRowTest(StepGateTestBase):
    def test_step_row_within_budget_passes(self):
        run = self.make_run(
            {
                rank: step_doc(rank, [40.0, 41.0], tokens=[3, 2])
                for rank in range(4)
            }
        )
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 0)
        self.assertEqual(doc["status"], "STEP-TIME-PASS")
        self.assertEqual(doc["verdict"]["step_time_budget"], "PASS")
        self.assertEqual(doc["verdict"]["label"], "STEP-TIME-ONLY")
        self.assertAlmostEqual(doc["aggregate"]["step_ms_worst_rank"], 40.5)
        self.assertAlmostEqual(doc["budget"]["derived"]["single_token_tokens_per_second"], round(1000.0 / 40.5, 6))
        self.assertEqual(doc["kind"], "device")
        self.assertTrue(doc["aggregate"]["tokens_identical_all_ranks"])

    def test_step_row_over_budget_fails(self):
        docs = {rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)}
        docs[2] = step_doc(2, [60.0], tokens=[3])
        run = self.make_run(docs)
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertEqual(doc["status"], "STEP-TIME-FAIL")
        self.assertEqual(doc["verdict"]["label"], "STEP-TIME-FAIL")
        self.assertAlmostEqual(doc["aggregate"]["step_ms_worst_rank"], 60.0)

    def test_every_step_within_budget_check(self):
        run = self.make_run({rank: step_doc(rank, [30.0, 55.0], tokens=[3, 2]) for rank in range(4)})
        code, doc = self.run_cli("--device-dir", run)
        self.assertIn("every counted step within budget", doc["failures"])

    def test_reduce_mismatch_refuses_the_row(self):
        docs = {rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)}
        docs[1] = step_doc(1, [40.0], tokens=[3], reduces=79)
        run = self.make_run(docs)
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertEqual(doc["status"], "STEP-TIME-FAIL")
        self.assertIn("reduces posted == 2 per layer per step", doc["failures"])
        self.assertEqual(doc["verdict"]["label"], "STEP-TIME-REFUSED")

    def test_weight_traffic_refuses_the_row(self):
        docs = {rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)}
        docs[3] = step_doc(3, [40.0], tokens=[3], weight_bytes_per_token=1_000_000)
        run = self.make_run(docs)
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertIn("weight_bytes_per_token == 0 (resident weights)", doc["failures"])

    def test_missing_per_layer_refuses_by_default(self):
        docs = {rank: step_doc(rank, [40.0], tokens=[3], per_layer=(rank != 2)) for rank in range(4)}
        run = self.make_run(docs)
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertIn("per-layer breakdown present", doc["failures"])

    def test_missing_per_layer_allowed_when_explicitly_disabled(self):
        docs = {rank: step_doc(rank, [40.0], tokens=[3], per_layer=False) for rank in range(4)}
        run = self.make_run(docs)
        code, doc = self.run_cli("--device-dir", run, "--no-require-per-layer")
        self.assertEqual(code, 0)
        self.assertEqual(doc["status"], "STEP-TIME-PASS")

    def test_per_layer_reconcile_residual_is_reported(self):
        run = self.make_run({rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)})
        code, doc = self.run_cli("--device-dir", run)
        row = doc["per_rank"][0]
        self.assertEqual(row["per_layer_reconcile"]["layer_total_us_sum"], 40000.0)
        self.assertAlmostEqual(row["per_layer_reconcile"]["residual_us"], 0.0)

    def test_one_rank_without_timed_step_refuses(self):
        docs = {rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)}
        docs[2] = loop_doc(2)
        run = self.make_run(docs)
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertIn("timed steps measured", doc["failures"])

    def test_partial_row_is_labelled_not_refused(self):
        run = self.make_run({rank: step_doc(rank, [40.0, 41.0], tokens=[3, 2]) for rank in range(4)})
        code, doc = self.run_cli("--device-dir", run, "--expected-steps", "10")
        self.assertEqual(code, 0)
        self.assertTrue(any("partial row" in label for label in doc["labels"]))

    def test_host_twin_kind_is_recorded_and_separate(self):
        docs = {}
        for rank in range(4):
            docs[rank] = {
                "stage": "tp4-host",
                "schema": 1,
                "rank": rank,
                "world": 4,
                "prompt_tokens": [1, 2, 3],
                "generated_tokens": [3, 2],
                "timing": {"step_ms": [767.7, 800.0], "mean_step_ms": 783.85},
            }
        run = self.make_run(docs)
        code, doc = self.run_cli("--device-dir", run, "--kind", "host-twin")
        self.assertEqual(code, 0)
        self.assertEqual(doc["kind"], "host-twin")
        self.assertEqual(doc["status"], "HOST-TWIN-ROW")
        self.assertIn("HOST-REFERENCE", doc["verdict"]["label"])
        self.assertEqual(doc["verdict"]["step_time_budget"], "NOT-APPLICABLE (host reference row: the 42 ms budget is a device target)")
        self.assertAlmostEqual(doc["aggregate"]["step_ms_worst_rank"], 783.85)


class AcceptedTokensTest(StepGateTestBase):
    def test_accepted_tokens_without_source_is_an_input_error(self):
        run = self.make_run({rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)})
        code = gate.main(["--device-dir", run, "--accepted-tokens-per-step", "3.9", "--out-json", self.out])
        self.assertEqual(code, 2)
        self.assertFalse(os.path.exists(self.out))

    def test_accepted_tokens_with_source_is_measured(self):
        run = self.make_run({rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)})
        code, doc = self.run_cli(
            "--device-dir",
            run,
            "--accepted-tokens-per-step",
            "3.9",
            "--accepted-tokens-source",
            "arm field step.accepted_tokens_per_step",
        )
        self.assertEqual(code, 0)
        self.assertEqual(doc["budget"]["accepted_tokens_per_step"], 3.9)
        self.assertAlmostEqual(doc["budget"]["derived"]["accepted_tokens_per_second_if_measured"], 97.5)
        self.assertEqual(doc["verdict"]["tok_s_gate"], "MET")
        self.assertEqual(doc["verdict"]["label"], "GATE-GRADE")

    def test_accepted_tokens_below_target_is_unmet(self):
        run = self.make_run({rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)})
        code, doc = self.run_cli(
            "--device-dir",
            run,
            "--accepted-tokens-per-step",
            "2.0",
            "--accepted-tokens-source",
            "synthetic test value",
        )
        self.assertEqual(code, 0)
        self.assertEqual(doc["verdict"]["tok_s_gate"], "UNMET")
        self.assertAlmostEqual(doc["budget"]["derived"]["accepted_tokens_per_second_if_measured"], 50.0)


class InputTest(StepGateTestBase):
    def test_no_device_dir_is_an_input_error(self):
        self.assertEqual(gate.main(["--out-json", self.out]), 2)

    def test_duplicate_rank_evidence_is_refused(self):
        run = self.make_run({0: step_doc(0, [40.0], tokens=[3])})
        write_json(os.path.join(run, "rank0", "duplicate.json"), step_doc(0, [40.0], tokens=[3]))
        code = gate.main(["--device-dir", run, "--out-json", self.out])
        self.assertEqual(code, 2)

    def test_token_outside_vocab_is_refused(self):
        docs = {rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)}
        docs[0]["tokens"]["generated"] = [999999]
        run = self.make_run(docs)
        code = gate.main(["--device-dir", run, "--expect-vocab", str(VOCAB), "--out-json", self.out])
        self.assertEqual(code, 2)

    def test_wrong_world_size_is_flagged(self):
        run = self.make_run({rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(3)})
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertIn("four ranks present", doc["failures"])

    def test_dump_sidecars_are_ignored_not_read(self):
        run = self.make_run({rank: step_doc(rank, [40.0], tokens=[3]) for rank in range(4)})
        dump = os.path.join(run, "rank0", "hybrid-dump-logits.jsonl")
        with open(dump, "w", encoding="utf-8") as handle:
            handle.write("{}\n")
        with open(dump + ".bin", "wb") as handle:
            handle.write(b"\x00" * 16)
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 0)
        ignored = [entry["path"] for entry in doc["provenance"]["ignored_files"]]
        self.assertIn(os.path.abspath(dump), ignored)



class ForwardRefusalTest(StepGateTestBase):
    def write_refusal(self, run, rank, *, rc_text="2", forward="refused"):
        directory = os.path.join(run, "rank%d" % rank)
        write_json(
            os.path.join(directory, "tp4-token-r%d.json" % rank),
            {
                "schema": "sparkflash-tp4-token/1",
                "stage": "tp4-chain",
                "rank": rank,
                "world": 4,
                "forward": forward,
                "tokens": {"ids": [], "positions": []},
                "positions_run": 0,
                "logits_dump": None,
                "error": "four-rank forward loop: the combine preflight refused before any routed expert launched",
            },
        )
        write_text = os.path.join(directory, "chain-r%d.rc" % rank)
        with open(write_text, "w", encoding="utf-8") as handle:
            handle.write(rc_text + chr(10))

    def test_labelled_forward_refusal_rc2_is_accepted(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)})
        for rank in range(4):
            self.write_refusal(run, rank)
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 3)
        self.assertEqual(doc["status"], "NOT-ATTEMPTED")
        self.assertEqual(doc["failures"], [])
        self.assertEqual(doc["per_rank"][0]["rc"], 2)
        checks = {check["name"]: check for check in doc["checks"]}
        self.assertEqual(checks["all ranks rc==0"]["status"], "pass")
        self.assertIn("labelled forward refusal", checks["all ranks rc==0"]["detail"])
        self.assertEqual(checks["forward refusal (rc=2)"]["status"], "info")

    def test_rc2_without_refusal_envelope_is_still_bind_fail(self):
        run = self.make_run({rank: loop_doc(rank) for rank in range(4)})
        for rank in range(4):
            with open(os.path.join(run, "rank%d" % rank, "chain-r%d.rc" % rank), "w", encoding="utf-8") as handle:
                handle.write("2" + chr(10))
        code, doc = self.run_cli("--device-dir", run)
        self.assertEqual(code, 1)
        self.assertEqual(doc["status"], "BIND-FAIL")
        self.assertIn("all ranks rc==0", doc["failures"])


if __name__ == "__main__":
    unittest.main()
