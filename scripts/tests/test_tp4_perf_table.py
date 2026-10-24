"""Tests for scripts/tp4-perf-table.py: the per-stage aggregation the re-tile reads.

The script renders the table in `docs/build/s1-result.md` section 8.2. Three properties matter and
are pinned here, because each silently changes the re-tile's priorities when it is wrong:

  * only `head`-flagged positions are decode steps; a `head=false` prefill position must not enter
    the mean, and `--steady N` must take the LAST N of them (that is how 8.2's eight steps are picked);
  * a block row is trimmed at its last non-zero field (so it can be any length up to the legend's,
    with the missing trailing fields zero) -- a short row is valid, not malformed;
  * the step total is the 40-layer subtotal plus the head's stages, and the head is recorded outside
    `layers`, in `positions[].head_blocks`.

The synthetic fixture has the real document's shape. The numbers themselves are pinned to the
published table by `--expect` on the workstation, where the 40 MB document exists.
"""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tp4-perf-table.py"
FIELDS = [
    "wall_ms", "kernel_ns", "overhead_ms", "launches", "syncs", "sync_ms", "h2d_count", "h2d_bytes",
    "h2d_ms", "d2h_count", "d2h_bytes", "d2h_ms", "dtod_count", "dtod_bytes", "dtod_ms",
    "alloc_count", "free_count", "module_load_count", "wait_ms", "host_ms",
]


def load_script():
    spec = importlib.util.spec_from_file_location("tp4_perf_table", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block(wall_ms=0.0, kernel_ns=0, launches=0, trim=0):
    """A block row in legend order, trimmed at `trim` values (0 = full length)."""
    row = [0] * len(FIELDS)
    row[0] = wall_ms
    row[1] = kernel_ns
    row[3] = launches
    return row[:trim] if trim else row


def position(number, head, layers, head_blocks=None):
    return {
        "position": number,
        "head": head,
        "layers": [{"layer": index, "total_ms": 0.0, "blocks": blocks} for index, blocks in layers],
        "head_blocks": head_blocks or {},
    }


def document(positions, rank=0, world=4, wall_clock="host-only", schema="sparkflash-tp4-forward-perf/2"):
    return {
        "schema": schema,
        "rank": rank,
        "world": world,
        "mode": {"lean": False, "probe_rank": 0, "synchronize_per_block": False, "wall_clock": wall_clock},
        "legend": {"block_fields": FIELDS, "mode": {}},
        "positions": positions,
    }


def layer_stages():
    return [
        (0, {"moe_routed": block(wall_ms=1.0, kernel_ns=2_000_000, launches=8),
             "attention_output": block(wall_ms=0.5, kernel_ns=1_000_000, launches=3)}),
        (1, {"moe_routed": block(wall_ms=0.5, kernel_ns=1_000_000, launches=4),
             "attention_output": [0.25]}),  # trimmed to one value: wall only, everything else zero
    ]


def head_blocks():
    return {
        "hc_pre": [0.007, 5119, 0.002, 1, 0, 0.0, 0, 0, 0.0, 0, 0, 0.0, 0, 0, 0.0, 0, 0, 0, 0.0, 0.007],
        "screen": [1.0, 4_000_000, 0.5, 2, 0, 0.0, 0, 0, 0.0, 1, 1024, 1.0, 0, 0, 0.0, 0, 0, 0, 1.0, 0.044],
        "gather_logits": [0.5],
        "head_total": [1.507, 4_005_119, 0.502, 3],
    }


def standard_row():
    """Three decode steps and one prefill position."""
    return [
        position(0, False, layer_stages(), head_blocks()),
        position(1, True, layer_stages(), head_blocks()),
        position(2, True, layer_stages(), head_blocks()),
        position(3, True, layer_stages(), head_blocks()),
    ]


class TrimmedRowTests(unittest.TestCase):
    def test_short_row_is_padded_with_zeros(self):
        script = load_script()
        record = script.as_record([0.202], FIELDS)
        self.assertEqual(record["wall_ms"], 0.202)
        self.assertEqual(record["kernel_ns"], 0)
        self.assertEqual(record["launches"], 0)

    def test_a_nine_value_row_is_accepted(self):
        script = load_script()
        record = script.as_record(block(wall_ms=1.0, kernel_ns=2_000_000, launches=8, trim=9), FIELDS)
        self.assertEqual(record["launches"], 8)
        self.assertEqual(record["d2h_count"], 0)

    def test_a_nineteen_value_row_is_accepted(self):
        script = load_script()
        record = script.as_record(block(wall_ms=1.0, trim=19), FIELDS)
        self.assertEqual(record["wall_ms"], 1.0)
        self.assertEqual(record["host_ms"], 0)

    def test_empty_and_over_long_rows_are_refused(self):
        script = load_script()
        with self.assertRaises(SystemExit):
            script.as_record([], FIELDS)
        with self.assertRaises(SystemExit):
            script.as_record([0.0] * (len(FIELDS) + 1), FIELDS)


class DecodePositionSelectionTests(unittest.TestCase):
    def test_only_head_positions_are_decode_steps(self):
        script = load_script()
        positions = script.decode_positions(document(standard_row()))
        self.assertEqual([item["position"] for item in positions], [1, 2, 3])

    def test_a_row_with_no_decode_position_is_an_error(self):
        script = load_script()
        doc = document([position(0, False, [(0, {"x": block()})])])
        with self.assertRaises(SystemExit):
            script.rows(doc, script.decode_positions(doc))


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.script = load_script()
        self.doc = document(standard_row())
        self.out, self.totals = self.script.rows(self.doc, self.script.decode_positions(self.doc))
        self.by_key = {row["key"]: row for row in self.out}

    def test_kernel_time_is_summed_over_layers_and_averaged_over_steps(self):
        routed = self.by_key["moe_routed"]
        self.assertAlmostEqual(routed["kernel_ms_per_step"], 3.0, places=6)
        self.assertAlmostEqual(routed["launches_per_step"], 12.0, places=6)
        self.assertAlmostEqual(routed["kernel_us_per_layer"], 3000.0 / 40.0, places=6)

    def test_trimmed_zero_row_counts_as_zero_over_that_layer(self):
        attention = self.by_key["attention_output"]
        self.assertAlmostEqual(attention["kernel_ms_per_step"], 1.0, places=6)
        self.assertAlmostEqual(attention["launches_per_step"], 3.0, places=6)
        self.assertEqual(attention["layers_with_kernel_time"], 1)

    def test_wall_column_is_the_host_wall(self):
        self.assertAlmostEqual(self.by_key["moe_routed"]["wall_ms_per_step"], 1.5, places=6)
        self.assertAlmostEqual(self.by_key["attention_output"]["wall_ms_per_step"], 0.75, places=6)

    def test_share_is_over_the_40_layer_subtotal(self):
        self.assertAlmostEqual(self.by_key["moe_routed"]["share"], 75.0, places=6)
        self.assertAlmostEqual(self.by_key["attention_output"]["share"], 25.0, places=6)

    def test_subtotal_is_the_layer_stages_only(self):
        subtotal = self.by_key["subtotal_40l"]
        self.assertAlmostEqual(subtotal["kernel_ms_per_step"], 4.0, places=6)
        self.assertAlmostEqual(subtotal["wall_ms_per_step"], 2.25, places=6)

    def test_head_stages_come_from_head_blocks(self):
        screen = self.by_key["head_screen"]
        self.assertAlmostEqual(screen["kernel_ms_per_step"], 4.0, places=6)
        self.assertAlmostEqual(screen["wall_ms_per_step"], 1.0, places=6)
        self.assertEqual(screen["launches_per_step"], 2.0)
        self.assertIsNone(screen["share"])

    def test_total_step_is_subtotal_plus_head(self):
        total = self.by_key["total_step"]
        head_kernel = sum(
            self.by_key[key]["kernel_ms_per_step"]
            for key in ("head_hc_pre", "head_screen", "head_gather_logits")
        )
        head_wall = sum(
            self.by_key[key]["wall_ms_per_step"]
            for key in ("head_hc_pre", "head_screen", "head_gather_logits")
        )
        self.assertAlmostEqual(total["kernel_ms_per_step"], 4.0 + head_kernel, places=6)
        self.assertAlmostEqual(total["wall_ms_per_step"], 2.25 + head_wall, places=6)

    def test_head_total_block_is_not_rendered_as_its_own_row(self):
        self.assertNotIn("head_total", self.by_key)
        self.assertEqual(
            [row["key"] for row in self.out[-5:]],
            ["subtotal_40l", "head_hc_pre", "head_screen", "head_gather_logits", "total_step"],
        )

    def test_step_totals_are_reported(self):
        self.assertEqual(self.totals["positions"], 3)
        self.assertEqual((self.totals["first_position"], self.totals["last_position"]), (1, 3))
        self.assertAlmostEqual(self.totals["kernel_ms_per_step"], 4.0, places=6)
        self.assertGreater(self.totals["total_kernel_ms_per_step"], self.totals["kernel_ms_per_step"])


class SchemaAndModeTests(unittest.TestCase):
    def test_unexpected_schema_is_refused(self):
        script = load_script()
        with self.assertRaises(SystemExit):
            script.check_schema(document(standard_row(), schema="some-other-probe/2"))

    def test_device_inclusive_row_is_refused(self):
        script = load_script()
        with self.assertRaises(SystemExit):
            script.check_mode(document(standard_row(), wall_clock="device-inclusive"))

    def test_host_only_row_is_accepted(self):
        script = load_script()
        script.check_mode(document(standard_row()))


class ClockWindowTests(unittest.TestCase):
    def test_says_so_when_the_row_has_no_gpu_csv(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIn("not recorded", script.clock_window(Path(tmp), 0))

    def test_reads_the_requested_ranks_own_csv(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            for rank, clock in ((0, 1469), (2, 1488)):
                rank_dir = Path(tmp) / f"rank{rank}"
                rank_dir.mkdir()
                (rank_dir / "gpu.csv").write_text(f"1.0 {clock},6.09,0\n2.0 {clock},6.09,0\n")
            self.assertIn("1469", script.clock_window(Path(tmp), 0))
            self.assertIn("1488", script.clock_window(Path(tmp), 2))

    def test_says_so_when_the_csv_has_no_rows(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            rank_dir = Path(tmp) / "rank0"
            rank_dir.mkdir()
            (rank_dir / "gpu.csv").write_text("# clock,value,other\n")
            self.assertIn("not recorded", script.clock_window(Path(tmp), 0))


class ProvenanceTests(unittest.TestCase):
    def test_provenance_states_mode_arm_and_clocks(self):
        script = load_script()
        doc = document(standard_row())
        lines = script.provenance(doc, Path("."), 0)
        joined = "\n".join(lines)
        self.assertIn("wall_clock=host-only", joined)
        self.assertIn("arm/commit: not recorded", joined)
        self.assertIn("clocks:", joined)
        self.assertIn("rank 0 of 4", joined)

    def test_fallback_clock_window_says_it_is_another_ranks_file(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            rank_dir = Path(tmp) / "rank0"
            rank_dir.mkdir()
            (rank_dir / "gpu.csv").write_text("1.0 1469,6.09,0\n2.0 1469,6.09,0\n")
            line = script.clock_window(Path(tmp), 2)
        self.assertIn("1469-1469 MHz", line)
        self.assertIn("requested rank's gpu.csv absent", line)


class CliRefusalTests(unittest.TestCase):
    def write(self, root: Path, doc: dict) -> Path:
        rank_dir = root / "rank0"
        rank_dir.mkdir(parents=True)
        (rank_dir / "tp4-forward-perf-r0.json").write_text(json.dumps(doc))
        return root

    def run_main(self, argv):
        script = load_script()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = script.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_a_device_inclusive_row_exits_two_through_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write(Path(tmp), document(standard_row(), wall_clock="device-inclusive"))
            code, _, err = self.run_main([str(root)])
        self.assertEqual(code, 2)
        self.assertIn("host-only", err)

    def test_an_unexpected_schema_exits_two_through_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write(Path(tmp), document(standard_row(), schema="other/1"))
            code, _, err = self.run_main([str(root)])
        self.assertEqual(code, 2)
        self.assertIn("schema", err)

    def test_malformed_expect_is_a_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write(Path(tmp), document(standard_row()))
            code, _, err = self.run_main([str(root), "--expect", "moe_routed"])
        self.assertEqual(code, 2)
        self.assertIn("STAGE=MS", err)


class CliTests(unittest.TestCase):
    def write_row(self, root: Path, layout: str = "rank") -> Path:
        doc = document(standard_row())
        if layout == "rank":
            rank_dir = root / "rank0"
            rank_dir.mkdir(parents=True)
            (rank_dir / "tp4-forward-perf-r0.json").write_text(json.dumps(doc))
        else:
            (root / "tp4-forward-perf-r0.json").write_text(json.dumps(doc))
        return root

    def run_main(self, argv):
        script = load_script()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = script.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_table_prints_stage_subtotal_head_and_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            code, out, _ = self.run_main([str(root)])
        self.assertEqual(code, 0)
        self.assertIn("moe_routed", out)
        self.assertIn("40-layer subtotal", out)
        self.assertIn("head.screen", out)
        self.assertIn("Total step (40 layers + head)", out)
        self.assertIn("40-layer subtotal", out)

    def test_json_report_has_no_position_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            code, out, _ = self.run_main([str(root), "--json"])
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertNotIn("positions", report)
        self.assertEqual(report["totals"]["positions"], 3)

    def test_steady_keeps_the_last_n_decode_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            code, out, _ = self.run_main([str(root), "--steady", "2", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["totals"]["positions"], 2)
        self.assertEqual(json.loads(out)["totals"]["last_position"], 3)
        self.assertEqual(json.loads(out)["totals"]["first_position"], 2)

    def test_negative_steady_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            code, _, err = self.run_main([str(root), "--steady", "-2"])
        self.assertEqual(code, 2)
        self.assertIn("negative", err)

    def test_steady_larger_than_the_row_is_noted_and_clamped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            code, out, err = self.run_main([str(root), "--steady", "20", "--json"])
        self.assertEqual(code, 0)
        self.assertIn("exceeds", err)
        self.assertEqual(json.loads(out)["totals"]["positions"], 3)

    def test_flat_layout_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp), layout="flat")
            code, _, _ = self.run_main([str(root)])
        self.assertEqual(code, 0)

    def test_missing_row_directory_exits_two(self):
        code, _, err = self.run_main(["/nonexistent/row"])
        self.assertEqual(code, 2)
        self.assertIn("not a directory", err)

    def test_matching_expectation_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            code, _, err = self.run_main([str(root), "--expect", "moe_routed=3.0"])
        self.assertEqual(code, 0, err)

    def test_mismatched_expectation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            code, _, err = self.run_main([str(root), "--expect", "moe_routed=69.201"])
        self.assertEqual(code, 1)
        self.assertIn("69.201", err)

    def test_absent_stage_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            code, _, err = self.run_main([str(root), "--expect", "nope=1.0"])
        self.assertEqual(code, 1)
        self.assertIn("absent", err)

    def test_tolerance_is_half_a_thousandth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_row(Path(tmp))
            strict, _, _ = self.run_main([str(root), "--expect", "moe_routed=3.0004"])
            loose, _, _ = self.run_main([str(root), "--expect", "moe_routed=3.0006"])
        self.assertEqual(strict, 0)
        self.assertEqual(loose, 1)


if __name__ == "__main__":
    unittest.main()
