#!/usr/bin/env python3
"""Unit tests for the task-12 D41 bars proposal and its dry-run helpers.

No GPU, no network, no device: the committed proposal fixture is checked against the D41 derivation
rule, the gate loader must refuse status proposal, and the dry-run helpers are tested on tiny inputs.
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

import compare_native_vs_vllm as compare
import d41_bars_proposal as proposal
import d41_bars_dryrun as dryrun

FIXTURES = os.path.join(SCRIPTS, "fixtures")
PROPOSAL_PATH = os.path.join(FIXTURES, "d41-bars-proposal-20261020.json")
OK_PATH = os.path.join(FIXTURES, "d41-bars-dryrun-ok-20261020.json")
IMPOSSIBLE_PATH = os.path.join(FIXTURES, "d41-bars-dryrun-impossible-20261020.json")


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, doc):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle)
        handle.write(chr(10))
    return path


def perfect_aggregate(positions=9):
    return {
        "positions": positions,
        "position_first": 129,
        "position_last": 137,
        "top1_agreement_rate": 1.0,
        "top20_overlap_mean": 1.0,
        "top100_overlap_mean": 1.0,
        "top1000_overlap_mean": 1.0,
        "max_abs_delta_max": 0.0,
        "mean_abs_delta_mean": 0.0,
        "kl_mean": 0.0,
        "kl_p95": 0.0,
        "js_mean": 0.0,
    }


class ProposalFixtureTest(unittest.TestCase):
    def test_proposal_is_not_a_gate(self):
        doc = load(PROPOSAL_PATH)
        self.assertEqual(doc["schema"], compare.GATE_BARS_SCHEMA)
        self.assertEqual(doc["status"], "proposal")
        self.assertTrue(doc["requires_lead_declaration"])
        with self.assertRaises(compare.CompareError) as raised:
            compare.check_gate_bars(doc, PROPOSAL_PATH)
        self.assertIn("status 'proposal'", str(raised.exception))

    def test_dry_run_ok_fixture_loads(self):
        doc = load(OK_PATH)
        self.assertEqual(doc["status"], "ok")
        self.assertTrue(doc.get("dry_run_fixture"))
        compare.check_gate_bars(doc, OK_PATH)
        outcome = compare.evaluate_bars({"bars": doc["bars"]}, perfect_aggregate())
        self.assertTrue(outcome["pass"], outcome)

    def test_impossible_fixture_fails_a_perfect_device(self):
        doc = load(IMPOSSIBLE_PATH)
        compare.check_gate_bars(doc, IMPOSSIBLE_PATH)
        outcome = compare.evaluate_bars({"bars": doc["bars"]}, perfect_aggregate())
        self.assertFalse(outcome["pass"])

    def test_committed_bars_are_supported_names(self):
        bars = load(PROPOSAL_PATH)["bars"]
        self.assertEqual(set(bars), set(compare.SUPPORTED_BARS) & set(bars))
        self.assertTrue(bars)

    def test_fixture_follows_the_d41_derivation_rule(self):
        doc = load(PROPOSAL_PATH)
        bars = doc["bars"]
        derivation = doc["derivation"]["bars"]
        for name, field in proposal.CEIL.items():
            item = derivation[name]
            floor = max(item["reference_repeat_floor"], item["capture_crosscheck_floor"])
            expected = max(proposal.PROVISIONAL[name], 3.0 * floor)
            self.assertAlmostEqual(bars[name], expected, places=12)
        for name, field in proposal.FLOOR.items():
            item = derivation[name]
            floor = min(item["reference_repeat_floor"], item["capture_crosscheck_floor"])
            expected = min(proposal.PROVISIONAL[name], 1.0 - 3.0 * (1.0 - floor))
            self.assertAlmostEqual(bars[name], expected, places=12)
        top1 = derivation["top1_agreement_min"]
        expected = min(proposal.PROVISIONAL["top1_agreement_min"], 1.0 - 3.0 * top1["suite_wide_flip_above_flat_rate"])
        self.assertAlmostEqual(bars["top1_agreement_min"], expected, places=12)
        self.assertEqual(bars["positions_min"], 9)

    def test_fixture_flags_every_looser_bar(self):
        doc = load(PROPOSAL_PATH)
        flags = doc["derivation"]["looser_than_d40_provisional"]
        self.assertEqual(set(flags), set(doc["bars"]) - {"positions_min"})
        for name, flag in flags.items():
            self.assertIn(name, proposal.PROVISIONAL)
            if name.endswith("_min"):
                self.assertLess(flag["proposed"], flag["d40_provisional"])
            else:
                self.assertGreater(flag["proposed"], flag["d40_provisional"])

    def test_reference_repeat_passes_and_pre_fix_dump_fails(self):
        doc = load(PROPOSAL_PATH)
        evaluation = doc["evaluation"]
        self.assertTrue(evaluation["proposed_on_reference_repeat"]["pass"])
        self.assertTrue(evaluation["proposed_on_capture_crosscheck"]["pass"])
        self.assertFalse(evaluation["d40_provisional_on_reference_repeat"]["pass"])
        self.assertFalse(evaluation["proposed_on_pre_fix_b70_dump"]["pass"])
        failed = {bar["bar"] for bar in evaluation["proposed_on_pre_fix_b70_dump"]["bars"] if not bar["pass"]}
        self.assertEqual(failed, set(doc["bars"]) - {"positions_min"})


class DerivationTest(unittest.TestCase):
    def test_derive_uses_worst_floor_and_provisional_caps(self):
        metrics = {field: 0.5 for field in set(proposal.CEIL.values()) | set(proposal.FLOOR.values())}
        metrics["top1_agreement"] = 1.0
        crosscheck = dict(metrics)
        crosscheck["top20_mean"] = 0.25  # worse than the repeat pair: floors must use it
        crosscheck["kl_mean"] = 0.6
        suite = {"strict_top1_rate": 0.94, "flip_above_flat_rate": 0.02}
        bars, evidence = proposal.derive(metrics, suite, crosscheck)
        self.assertAlmostEqual(bars["top20_overlap_mean_min"], 1.0 - 3.0 * 0.75, places=12)
        self.assertAlmostEqual(bars["kl_mean_max"], 3.0 * 0.6, places=12)
        self.assertAlmostEqual(bars["positions_min"], 9)
        self.assertAlmostEqual(bars["top1_agreement_min"], min(proposal.PROVISIONAL["top1_agreement_min"], 1.0 - 3.0 * 0.02), places=12)
        self.assertEqual(evidence["top20_overlap_mean_min"]["measured_reference_floor"], 0.25)

    def test_looser_flags(self):
        bars = dict(proposal.PROVISIONAL)
        bars["kl_mean_max"] = proposal.PROVISIONAL["kl_mean_max"] + 1.0
        bars["top20_overlap_mean_min"] = proposal.PROVISIONAL["top20_overlap_mean_min"] - 0.1
        bars["positions_min"] = 9
        flags = proposal.looser(bars)
        self.assertEqual(set(flags), {"kl_mean_max", "top20_overlap_mean_min"})

    def test_oracle_prompt_refuses_bad_schema(self):
        with self.assertRaises(proposal.ProposalError):
            proposal.oracle_prompt({"schema": "other", "status": "ok", "prompts": []}, "x")

    def test_oracle_prompt_refuses_missing_prompt(self):
        with self.assertRaises(proposal.ProposalError):
            proposal.oracle_prompt({"schema": "sparkflash-native-oracle/1", "status": "ok", "prompts": []}, "x")


class DryRunHelperTest(unittest.TestCase):
    def test_full_logits_reproduces_given_logprobs(self):
        old = dryrun.VOCAB
        try:
            dryrun.VOCAB = 8
            given = {0: -0.5, 1: -1.5, 2: -2.5}
            logits = dryrun.full_logits(given)
            self.assertEqual(len(logits), 8)
            logs = compare.log_softmax(logits)
            for token, value in given.items():
                self.assertAlmostEqual(logs[token], value, places=5)
            self.assertEqual(max(range(8), key=lambda token: (logs[token], -token)), 0)
        finally:
            dryrun.VOCAB = old

    def test_write_dump_tiles_and_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dump")
            rows = [(0, [0.0, 1.0, 2.0], 2), (1, [3.0, 2.0, 1.0], 0)]
            dump_path, bin_path = dryrun.write_dump(path, rows)
            self.assertEqual(dump_path, path + ".jsonl")
            self.assertEqual(os.path.getsize(bin_path), (3 + 3) * 4)
            parsed = compare.read_dump(dump_path)
            self.assertEqual([row["position"] for row in parsed], [0, 1])
            self.assertEqual([row["logits_offset"] for row in parsed], [0, 12])
            compare.check_dump_sidecar(parsed, dump_path, bin_path)
            with open(bin_path, "rb") as handle:
                values = struct.unpack("<6f", handle.read())
            self.assertEqual(list(values), [0.0, 1.0, 2.0, 3.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()

