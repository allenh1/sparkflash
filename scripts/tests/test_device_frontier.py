#!/usr/bin/env python3
"""Unit tests for scripts/device-frontier.py: the four-rank device-probe verdict tool.

No network, no GPU, no row: every test builds synthetic documents in the writer's exact shape
(crates/sparkflash-cli/src/tp4_forward.rs write_layer_device_frontier, win/layer-device @ 5d5a5ffd)
and asserts the three exit codes in both directions:

    0 coherent probe / completed layer
    1 finding (rank frontier disagreement, the known lost-chain defect, host arm, bad geometry,
      incomplete completed frontier, unusable gate input, ...)
    2 cannot testify (missing/unreadable/empty rank document, missing numerical-gate input) - a
      stop, never a pass

Run with:

    python3 -m unittest discover -s scripts/tests -v
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
SCRIPT = os.path.join(SCRIPTS, "device-frontier.py")
FIXTURES = os.path.join(HERE, "fixtures", "device-frontier")
sys.path.insert(0, SCRIPTS)

SPEC = importlib.util.spec_from_file_location("device_frontier", SCRIPT)
frontier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontier)


def rank_document(rank, **overrides):
    """The module's synthetic document builder, so the tests cannot drift from the contract."""
    return frontier.rank_document(rank, **overrides)


def token_document(rank, bin_path, **overrides):
    """A synthetic numerical-gate input (small rows/vocab keeps the bins tiny)."""
    return frontier.token_document(rank, bin_path, **overrides)


class FrontierTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="device-frontier-test-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- helpers ---------------------------------------------------------

    def write_row(self, documents, token_bodies=None, *, layout="rank-dirs", name="row"):
        root = os.path.join(self.tmp, name)
        token_documents = None
        if token_bodies is not None:
            token_documents = {}
            for rank, body in token_bodies.items():
                patched = json.loads(json.dumps(body))
                patched["logits_dump"]["bin"] = os.path.join(
                    root,
                    "rank{}".format(rank),
                    "tp4-logits-r{}.jsonl.bin".format(rank),
                )
                token_documents[rank] = patched
        frontier.write_row(root, documents, token_documents, layout=layout)
        return root

    def write_open_refusal_row(self, name="open-refusal"):
        """A refused-at-open row: bind/loop/token JSON, rc and chain log, no layer-device document."""
        root = self.write_row({}, name=name)
        for relative, content in frontier.open_refusal_files().items():
            path = os.path.join(root, relative)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if isinstance(content, dict):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(content, handle)
            else:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content)
        return root

    def complete_documents(self, **overrides):
        return {rank: rank_document(rank, status=frontier.STATUS_COMPLETED) for rank in (0, 1, 2, 3)}

    def complete_token_bodies(self):
        return {
            rank: token_document(rank, "PLACEHOLDER", rows=2, vocab=16)
            for rank in (0, 1, 2, 3)
        }

    def evaluate(self, root, **kwargs):
        return frontier.evaluate(root, **kwargs)

    def code(self, root, **kwargs):
        return self.evaluate(root, **kwargs)["exit_code"]

    def run_cli(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = frontier.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def finding_ids(self, result):
        return [check["id"] for check in result["checks"] if check["status"] == frontier.STATUS_FINDING]

    def reference_root_for(self, root):
        """Write the frozen reference for the row's fixture key, so readiness sees it present."""
        with open(os.path.join(root, "rank0", "tp4-token-r0.json"), "r", encoding="utf-8") as handle:
            token = json.load(handle)
        key_basis = {
            "positions": token["tokens"]["positions"],
            "prompt_len": token["tokens"]["prompt_len"],
            "ids": token["tokens"]["ids"],
        }
        fixture = frontier.hashlib.sha256(
            json.dumps(key_basis, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        reference = os.path.join(self.tmp, "reference")
        directory = os.path.join(reference, fixture)
        os.makedirs(directory)
        for rank in (0, 1, 2, 3):
            with open(os.path.join(directory, "rank{}.json".format(rank)), "w", encoding="utf-8") as handle:
                json.dump({"synthetic": True, "rank": rank}, handle)
        return reference, fixture

    # ---- direction 1: a coherent probe is coherent -----------------------

    def test_coherent_probe_exits_zero(self):
        result = self.evaluate(self.write_row({rank: rank_document(rank) for rank in (0, 1, 2, 3)}))
        self.assertEqual(result["exit_code"], 0, result["findings"])
        self.assertEqual(result["verdict"], "COHERENT-PROBE")
        self.assertTrue(result["not_a_measurement"])

    def test_coherent_probe_accepts_flat_collection(self):
        root = self.write_row(
            {rank: rank_document(rank) for rank in (0, 1, 2, 3)}, layout="flat"
        )
        self.assertEqual(self.code(root), 0)

    def test_flat_and_rank_subdir_ambiguity_is_a_stop(self):
        root = self.write_row({rank: rank_document(rank) for rank in (0, 1, 2, 3)})
        flat = os.path.join(root, "tp4-layer-device-r0.json")
        shutil.copyfile(os.path.join(root, "rank0", "tp4-layer-device-r0.json"), flat)
        result = self.evaluate(root)
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("two possible documents", result["stop_reason"])

    def test_render_says_a_probe_is_not_a_measurement(self):
        result = self.evaluate(self.write_row({rank: rank_document(rank) for rank in (0, 1, 2, 3)}))
        text = frontier.render(result)
        self.assertIn("COHERENT PROBE", text)
        self.assertIn("NOT a measurement", text)
        self.assertIn("no D54 arm", text)

    def test_an_18_stage_chain_stopping_at_reference_seam_is_a_finding(self):
        # reference_seam is stage index 6; an 18-stage chain would have to stop at moe_shared.
        # This is the literal-reading counterexample, kept so the tool answers it with exit 1.
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:18]), stopped="reference_seam") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-stopped", self.finding_ids(result))

    def test_refusal_at_reference_seam_is_coherent_at_index_6(self):
        # reference_seam is the compressor/indexer/compressed-cache write stage; six stages ran, so
        # it stops at index 6. ref_select (index 7) is the next boundary after it.
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:6]), stopped="reference_seam") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 0, result["findings"])
        self.assertEqual(frontier.STAGE_ORDER[6], "reference_seam")

    def test_refusal_at_ref_select_is_coherent_with_a_7_stage_chain(self):
        # The ref_select split: the stages through reference_seam ran, ref_select refuses. Seven
        # stages ran, so it stops at index 7. This is the boundary whose failure is not a rounding:
        # a different top-k SET changes the attended context.
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:7]), stopped="ref_select") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 0, result["findings"])
        self.assertEqual(frontier.STAGE_ORDER[7], "ref_select")
        self.assertEqual(len(frontier.STAGE_ORDER), 20)

    def test_an_8_stage_chain_stopping_at_ref_select_is_a_finding(self):
        # Chain length 8 would put the next stage at stage_attention, not ref_select.
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:8]), stopped="ref_select") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-stopped", self.finding_ids(result))

    # ---- the known defect is a finding, not a tolerated shape ------------

    def test_degenerate_lost_chain_is_a_finding(self):
        docs = {rank: rank_document(rank, chain=[], stopped="kv_path") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-stopped", self.finding_ids(result))
        detail = " ".join(result["findings"])
        self.assertIn("lost-chain defect", detail)
        self.assertIn("StageRuns::from_refusal", detail)

    def test_rank_frontier_disagreement_is_a_finding(self):
        docs = {rank: rank_document(rank) for rank in (0, 1, 2, 3)}
        docs[2] = rank_document(2, chain=list(frontier.STAGE_ORDER[:3]), stopped="query_path")
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-rank-agreement", self.finding_ids(result))

    def test_host_arm_is_a_finding(self):
        docs = {
            rank: rank_document(
                rank,
                arm="host arm (selected by SPARKFLASH_LAYER_DEVICE=0: the bit-exact host sequence; the device arm is the default)",
            )
            for rank in (0, 1, 2, 3)
        }
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("arm-device", self.finding_ids(result))

    def test_the_s1_device_arm_spelling_still_passes(self):
        # Rows filed before S2 flipped the default carry the opt-in spelling of the same arm.
        docs = {rank: rank_document(rank, arm=frontier.DEVICE_ARM_S1) for rank in (0, 1, 2, 3)}
        self.assertEqual(self.code(self.write_row(docs)), 0)

    def test_an_explicitly_next_stage_stop_passes(self):
        # Five stages ran, so the refusal at the sixth (kv_path) is exactly right.
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:5]), stopped="kv_path") for rank in (0, 1, 2, 3)}
        self.assertEqual(self.code(self.write_row(docs)), 0)

    def test_a_stop_that_skips_a_stage_is_a_finding(self):
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:3]), stopped="kv_path") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-stopped", self.finding_ids(result))

    def test_a_chain_that_is_not_a_prefix_is_a_finding(self):
        docs = {
            rank: rank_document(
                rank,
                chain=["mhc_entry", "engram_write"],
                stopped="quantise_prologue",
            )
            for rank in (0, 1, 2, 3)
        }
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-prefix", self.finding_ids(result))

    def test_unknown_stage_name_is_a_finding(self):
        docs = {rank: rank_document(rank, chain=["engram_write", "not_a_stage"], stopped="mhc_entry") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-prefix", self.finding_ids(result))

    def test_refusal_at_moe_shared_is_coherent_with_an_18_stage_chain(self):
        # moe_shared is the shared-expert stage at index 18: the 18 stages before it ran (through
        # all_reduce_rdma_arm), the arm refused there. Chain length must equal the index, so 18.
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:18]), stopped="moe_shared") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 0, result["findings"])
        self.assertEqual(frontier.STAGE_ORDER.index("moe_shared"), 18)
        self.assertEqual(len(frontier.STAGE_ORDER), 20)

    def test_refusal_at_moe_shared_with_a_19_stage_chain_is_a_finding(self):
        # The boundary off-by-one: 19 stages "ran" and the arm still stopped at moe_shared. Under
        # the unchanged index rule the next stage after 19 is ffn_epilogue, so this is incoherent.
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:19]), stopped="moe_shared") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-stopped", self.finding_ids(result))

    def test_probe_row_with_all_20_stages_and_no_stop_is_a_finding(self):
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER), stopped=None) for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-refused", self.finding_ids(result))

    def test_unparseable_frontier_is_a_finding(self):
        docs = {rank: rank_document(rank, chain=list(frontier.STAGE_ORDER[:5]), stopped="kv_path") for rank in (0, 1, 2, 3)}
        for document in docs.values():
            document["frontier"] = "the arm did something"
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-grammar", self.finding_ids(result))

    def test_frontier_stage_count_mismatch_is_a_finding(self):
        docs = {rank: rank_document(rank) for rank in (0, 1, 2, 3)}
        for document in docs.values():
            document["frontier"] = document["frontier"].replace("(5 stage(s))", "(6 stage(s))")
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-grammar", self.finding_ids(result))

    # ---- exit 2: cannot testify, never a pass ----------------------------

    def test_missing_rank_document_is_a_stop(self):
        docs = {rank: rank_document(rank) for rank in (0, 1, 2)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("rank document(s) missing", result["stop_reason"])

    def test_empty_document_is_a_stop(self):
        root = self.write_row({rank: rank_document(rank) for rank in (0, 1, 2, 3)})
        path = os.path.join(root, "rank1", "tp4-layer-device-r1.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        result = self.evaluate(root)
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("empty (0 bytes)", result["stop_reason"])

    def test_invalid_json_is_a_stop(self):
        root = self.write_row({rank: rank_document(rank) for rank in (0, 1, 2, 3)})
        path = os.path.join(root, "rank2", "tp4-layer-device-r2.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        result = self.evaluate(root)
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("not JSON", result["stop_reason"])

    def test_non_object_json_is_a_stop(self):
        root = self.write_row({rank: rank_document(rank) for rank in (0, 1, 2, 3)})
        path = os.path.join(root, "rank3", "tp4-layer-device-r3.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([1, 2, 3], handle)
        result = self.evaluate(root)
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("not a JSON object", result["stop_reason"])

    def test_missing_row_directory_is_a_stop(self):
        result = self.evaluate(os.path.join(self.tmp, "does-not-exist"))
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("row directory does not exist", result["stop_reason"])

    # ---- open-time refusal: exit 3, not a frontier and not missing evidence ----

    def test_open_time_refusal_is_its_own_verdict(self):
        result = self.evaluate(self.write_open_refusal_row())
        self.assertEqual(result["exit_code"], 3, result["findings"])
        self.assertNotEqual(result["exit_code"], 2)
        self.assertEqual(result["verdict"], "OPEN-REFUSAL")
        self.assertTrue(result["not_a_frontier_row"])
        info = result["open_refusal"]
        self.assertEqual(info["stage"], "mhc_entry")
        self.assertEqual(info["layer"], 0)
        self.assertIn(
            "layer device seams: layer 0: layer device weight: no hc_attn_fn for layer 0 in the resolved set",
            info["line"],
        )
        self.assertTrue(info["uniform"])
        self.assertEqual(len(info["refusals"]), 4)
        self.assertEqual(len(info["evidence"]["bind"]), 4)
        self.assertEqual(len(info["evidence"]["loop"]), 4)
        self.assertEqual(len(info["evidence"]["token"]), 4)

    def test_open_time_refusal_renders_the_blocker_and_the_rule(self):
        text = frontier.render(self.evaluate(self.write_open_refusal_row(name="open-refusal-render")))
        self.assertIn("OPEN REFUSAL (exit 3)", text)
        self.assertIn("no hc_attn_fn for layer 0 in the resolved set", text)
        self.assertIn("mhc_entry (layer 0)", text)
        self.assertIn("NOT 'missing evidence'", text)
        self.assertIn("NOT a frontier stage, NOT a ms/step", text)

    def test_open_time_refusal_under_require_complete_stays_exit_3(self):
        result = self.evaluate(self.write_open_refusal_row(name="open-refusal-complete"), require_complete=True)
        self.assertEqual(result["exit_code"], 3)
        self.assertTrue(any("not applicable" in warning for warning in result["warnings"]))

    def test_open_time_refusal_needs_no_layer_device_document(self):
        root = self.write_open_refusal_row(name="open-refusal-absent")
        self.assertEqual(
            [name for name in os.listdir(root) if name.startswith("tp4-layer-device")], []
        )
        self.assertEqual(self.code(root), 3)

    def test_documents_missing_without_a_named_refusal_stay_exit_2(self):
        # A bare absence with no open-time refusal line is still "cannot testify", not a refusal.
        result = self.evaluate(self.write_row({}))
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("rank document(s) missing", result["stop_reason"])

    def test_open_time_refusal_cli_exit_code_and_verdict_line(self):
        code, out, _err = self.run_cli(
            ["--row-dir", self.write_open_refusal_row(name="open-refusal-cli")]
        )
        self.assertEqual(code, 3, out)
        self.assertIn("OPEN REFUSAL (exit 3)", out)
        self.assertIn("no hc_attn_fn for layer 0 in the resolved set", out)

    # ---- geometry, bookkeeping, note, schema ----------------------------

    def test_rope_geometry_formula_is_checked(self):
        docs = {rank: rank_document(rank, extra={"rope_table_bytes": 12345}) for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("geometry", self.finding_ids(result))

    def test_geometry_must_agree_across_ranks(self):
        docs = {rank: rank_document(rank) for rank in (0, 1, 2, 3)}
        docs[1] = rank_document(1, max_positions=8192)
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("geometry-rank-agreement", self.finding_ids(result))

    def test_geometry_requires_positive_integers(self):
        docs = {rank: rank_document(rank, extra={"slab_bytes": 0}) for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("geometry", self.finding_ids(result))

    def test_optional_allocation_pair_must_fit_slab(self):
        docs = {
            rank: rank_document(rank, extra={"kv_cache_bytes": 60 << 20, "window_ring_bytes": 8 << 20})
            for rank in (0, 1, 2, 3)
        }
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("geometry", self.finding_ids(result))

    def test_optional_allocation_pair_must_be_a_pair(self):
        docs = {rank: rank_document(rank, extra={"kv_cache_bytes": 1 << 20}) for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("geometry", self.finding_ids(result))

    def test_schema_mismatch_is_a_finding(self):
        docs = {rank: rank_document(rank, schema="something-else/1") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("schema", self.finding_ids(result))

    def test_rank_field_mismatch_is_a_finding(self):
        docs = {rank: rank_document(rank) for rank in (0, 1, 2, 3)}
        docs[2]["rank"] = 7
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("rank-field", self.finding_ids(result))

    def test_world_mismatch_is_a_finding(self):
        docs = {rank: rank_document(rank, world=2) for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("world", self.finding_ids(result))

    def test_layer_or_position_disagreement_is_a_finding(self):
        docs = {rank: rank_document(rank) for rank in (0, 1, 2, 3)}
        docs[1] = rank_document(1, layer=1)
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("bookkeeping", self.finding_ids(result))

    def test_refused_note_needs_the_not_host_validated_wording(self):
        docs = {rank: rank_document(rank, note="values fine") for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("note-not-host-validated", self.finding_ids(result))

    def test_refused_note_accepts_a_custom_needle(self):
        docs = {rank: rank_document(rank, note="a probe; not compared with the twin") for rank in (0, 1, 2, 3)}
        root = self.write_row(docs)
        self.assertEqual(self.code(root), 1)
        self.assertEqual(self.code(root, note_needles=("not compared with the twin",)), 0)

    def test_refused_document_needs_a_message(self):
        docs = {rank: rank_document(rank, drop_keys=("message",)) for rank in (0, 1, 2, 3)}
        result = self.evaluate(self.write_row(docs))
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("message-present", self.finding_ids(result))

    # ---- --require-complete ---------------------------------------------

    def test_complete_layer_is_eligible_for_the_numerical_gate(self):
        root = self.write_row(self.complete_documents(), self.complete_token_bodies())
        reference, fixture = self.reference_root_for(root)
        result = self.evaluate(root, require_complete=True, reference_root=reference)
        self.assertEqual(result["exit_code"], 0, result["findings"])
        self.assertEqual(result["verdict"], "COHERENT-COMPLETE-LAYER")
        self.assertTrue(result["numerical_gate"]["available"])
        self.assertIn("numerical-gate.sh", result["next_step"])
        self.assertEqual(result["fixture"], fixture)
        self.assertTrue(result["reference"]["present"])

    def test_complete_layer_without_the_frozen_reference_warns_but_is_coherent(self):
        root = self.write_row(self.complete_documents(), self.complete_token_bodies())
        result = self.evaluate(root, require_complete=True, reference_root=os.path.join(self.tmp, "empty-ref"))
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(any("frozen reference NOT present" in warning for warning in result["warnings"]))

    def test_require_complete_on_a_refused_row_is_a_finding(self):
        root = self.write_row({rank: rank_document(rank) for rank in (0, 1, 2, 3)})
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("status", self.finding_ids(result))

    def test_require_complete_needs_the_full_20_stage_chain(self):
        docs = {
            rank: rank_document(rank, status=frontier.STATUS_COMPLETED, chain=list(frontier.STAGE_ORDER[:19]), stopped=None)
            for rank in (0, 1, 2, 3)
        }
        root = self.write_row(docs, self.complete_token_bodies())
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-complete", self.finding_ids(result))

    def test_completed_document_with_a_stop_is_a_finding(self):
        docs = {rank: rank_document(rank, status=frontier.STATUS_COMPLETED) for rank in (0, 1, 2, 3)}
        for document in docs.values():
            document["frontier"] = frontier.render_frontier(
                list(frontier.STAGE_ORDER[:5]), "kv_path", "boom"
            )
        root = self.write_row(docs, self.complete_token_bodies())
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("frontier-complete", self.finding_ids(result))

    def test_completed_document_with_a_message_key_is_a_finding(self):
        docs = self.complete_documents()
        for document in docs.values():
            document["message"] = "stale refusal text"
        root = self.write_row(docs, self.complete_token_bodies())
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("message-absent", self.finding_ids(result))

    def test_complete_without_token_documents_is_a_stop(self):
        root = self.write_row(self.complete_documents())
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("token documents", result["stop_reason"])

    def test_complete_without_logits_bin_is_a_stop(self):
        root = self.write_row(self.complete_documents(), self.complete_token_bodies())
        os.unlink(os.path.join(root, "rank2", "tp4-logits-r2.jsonl.bin"))
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("logits bin missing", result["stop_reason"])

    def test_complete_with_a_short_logits_bin_is_a_finding(self):
        root = self.write_row(self.complete_documents(), self.complete_token_bodies())
        with open(os.path.join(root, "rank1", "tp4-logits-r1.jsonl.bin"), "wb") as handle:
            handle.write(b"\x00" * 4)
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("numerical-gate-inputs", self.finding_ids(result))

    def test_complete_with_rank_disagreeing_on_the_fixture_key_is_a_finding(self):
        bodies = self.complete_token_bodies()
        bodies[3]["tokens"]["ids"] = [1, 2, 3]
        root = self.write_row(self.complete_documents(), bodies)
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("fixture-key-agreement", self.finding_ids(result))

    def test_tolerance_arm_absent_is_a_warning_not_a_finding(self):
        root = self.write_row(self.complete_documents(), self.complete_token_bodies())
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(any("arm declaration: not machine-checkable" in warning for warning in result["warnings"]))

    def test_tolerance_arm_declared_before_the_row_passes(self):
        docs = self.complete_documents()
        for document in docs.values():
            document["tolerance_arm"] = {"name": "speed-row-v1"}
            document["declared_utc"] = "2026-10-22T10:00:00Z"
            document["row_started_utc"] = "2026-10-22T11:00:00Z"
        root = self.write_row(docs, self.complete_token_bodies())
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 0, result["findings"])
        self.assertNotIn("tolerance-arm-declared-before-row", self.finding_ids(result))

    def test_tolerance_arm_declared_after_the_row_is_a_finding(self):
        docs = self.complete_documents()
        for document in docs.values():
            document["tolerance_arm"] = "speed-row-v1"
            document["declared_utc"] = "2026-10-22T12:00:00Z"
            document["row_started_utc"] = "2026-10-22T11:00:00Z"
        root = self.write_row(docs, self.complete_token_bodies())
        result = self.evaluate(root, require_complete=True)
        self.assertEqual(result["exit_code"], 1)
        self.assertIn("tolerance-arm-declared-before-row", self.finding_ids(result))

    # ---- CLI and the built-in self-test ---------------------------------

    def test_cli_exit_code_and_json_out(self):
        root = self.write_row({rank: rank_document(rank) for rank in (0, 1, 2, 3)})
        out_path = os.path.join(self.tmp, "verdict.json")
        code, out, _err = self.run_cli(["--row-dir", root, "--json-out", out_path])
        self.assertEqual(code, 0, out)
        with open(out_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["schema"], "sparkflash-device-frontier-check/1")
        self.assertEqual(payload["verdict"], "COHERENT-PROBE")

    def test_cli_requires_a_row_dir(self):
        code, _out, err = self.run_cli([])
        self.assertEqual(code, 2)
        self.assertIn("--row-dir is required", err)

    def test_cli_ranks_override(self):
        docs = {rank: rank_document(rank, world=2) for rank in (0, 1)}
        root = self.write_row(docs)
        code, out, _err = self.run_cli(["--row-dir", root, "--ranks", "0,1"])
        self.assertEqual(code, 0, out)

    def test_built_in_selftest_covers_both_directions(self):
        code, out, _err = self.run_cli(["--selftest"])
        self.assertEqual(code, 0, out)
        self.assertIn("19 of 19 case(s)", out)
        self.assertIn("expected=0 got=0", out)
        self.assertIn("expected=1 got=1", out)
        self.assertIn("expected=2 got=2", out)
        self.assertIn("expected=3 got=3", out)

    def test_write_fixtures_writes_a_synthetic_manifest(self):
        target = os.path.join(self.tmp, "fixtures")
        code, out, _err = self.run_cli(["--write-fixtures", target])
        self.assertEqual(code, 0, out)
        with open(os.path.join(target, "manifest.json"), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(len(manifest["cases"]), 19)

    # ---- the committed fixtures are the CI proof of the exit codes ------

    def committed_manifest(self):
        with open(os.path.join(FIXTURES, "manifest.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)

    def test_committed_fixtures_produce_the_documented_exit_codes(self):
        manifest = self.committed_manifest()
        self.assertTrue(manifest["synthetic"])
        expected = {case["case"]: case["expected_exit"] for case in manifest["cases"]}
        # The four directions the probe verdict depends on, named so a regression is obvious.
        self.assertEqual(expected["coherent-6-stage-kv_path"], 0)
        self.assertEqual(expected["coherent-refusal-at-reference_seam"], 0)
        self.assertEqual(expected["coherent-refusal-at-ref_select"], 0)
        self.assertEqual(expected["off-by-one-8-stage-chain-at-ref_select"], 1)
        self.assertEqual(expected["incoherent-18-stage-chain-at-reference_seam"], 1)
        self.assertEqual(expected["off-by-one-19-stage-chain-at-moe_shared"], 1)
        self.assertEqual(expected["coherent-refusal-at-moe_shared"], 0)
        self.assertEqual(expected["degenerate-no-stage-ran-kv_path"], 1)
        self.assertEqual(expected["rank-different-chain"], 1)
        self.assertEqual(expected["missing-rank-document"], 2)
        self.assertEqual(expected["open-refusal-no-frontier"], 3)
        for case in manifest["cases"]:
            row = os.path.join(FIXTURES, case["case"])
            result = frontier.evaluate(row, require_complete=case["mode"] == "complete")
            self.assertEqual(
                result["exit_code"],
                case["expected_exit"],
                "{}: findings={} stop={}".format(
                    case["case"], json.dumps(result["findings"]), result.get("stop_reason")
                ),
            )

    def test_committed_fixtures_match_the_generator(self):
        generated_root = os.path.join(self.tmp, "generated")
        generated = frontier.write_fixture_set(generated_root)

        def collect(root):
            contents = {}
            for dirpath, _dirnames, filenames in os.walk(root):
                for filename in filenames:
                    path = os.path.join(dirpath, filename)
                    with open(path, "rb") as handle:
                        contents[os.path.relpath(path, root)] = handle.read()
            return contents

        generated_files = collect(generated_root)
        committed_files = collect(FIXTURES)
        for relative, data in generated_files.items():
            if relative == "manifest.json":
                continue  # row_dir paths differ between a temp tree and the checkout; contents do not
            self.assertIn(relative, committed_files, "committed fixture is missing " + relative)
            self.assertEqual(data, committed_files[relative], "committed fixture drifted: " + relative)
        extra = {
            relative
            for relative in committed_files
            if relative != "manifest.json"
            and relative not in generated_files
            and relative.endswith((".json", ".jsonl.bin"))
        }
        self.assertEqual(extra, set(), "committed fixtures not produced by the generator: " + repr(extra))
        self.assertEqual(
            [case["case"] for case in generated],
            [case["case"] for case in self.committed_manifest()["cases"]],
        )

    # ---- grammar helpers -------------------------------------------------

    def test_parse_frontier_roundtrip(self):
        for chain, stopped in (
            ([], "kv_path"),
            (list(frontier.STAGE_ORDER[:5]), "kv_path"),
            (list(frontier.STAGE_ORDER), None),
        ):
            text = frontier.render_frontier(chain, stopped, "boom" if stopped else None)
            parsed = frontier.parse_frontier(text)
            self.assertEqual(parsed["chain"], chain)
            self.assertEqual(parsed["stopped"], stopped)

    def test_parse_frontier_rejects_a_bare_stage_name(self):
        with self.assertRaises(frontier.FrontierParseError):
            frontier.parse_frontier("kv_path")


if __name__ == "__main__":
    unittest.main()
