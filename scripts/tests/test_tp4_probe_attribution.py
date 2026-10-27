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

"""Tests for scripts/tp4-probe-attribution.py: the three layer totals and the counter mapping.

The script is the reproducibility anchor for the status doc's attribution table, so the aggregation
rules themselves must be pinned: seam-layers excludes the enigma layers 1 and 14, all-layers does
not, probe-scope divides the probe's own layer_total wall by its count, and the run counters come
from run.loop_total in legend.block_fields order (they are not in run.counters).
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "tp4-probe-attribution.py"


def load_script():
    spec = importlib.util.spec_from_file_location("tp4_probe_attribution", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def perf(layer_blocks):
    """Build a minimal probe perf row: one ordinary position with the given (layer, blocks) list."""
    layers = []
    for layer, blocks in layer_blocks:
        total = round(sum(values[0] for values in blocks.values()), 6)
        layers.append({"layer": layer, "total_ms": total, "blocks": blocks})
    fields = ["wall_ms", "kernel_ns", "overhead_ms", "launches", "syncs", "sync_ms", "h2d_count",
              "h2d_bytes", "h2d_ms", "d2h_count", "d2h_bytes", "d2h_ms", "dtod_count", "dtod_bytes",
              "dtod_ms", "alloc_count", "free_count", "module_load_count"]
    values = [0.0, 0, 0.0, 10, 10, 0.0, 7, 70, 0.0, 12, 120, 0.0, 0, 0, 0.0, 3, 0, 2]
    return {
        "positions": [{"position": 0, "layers": layers}],
        "run": {
            "loop_total": values,
            "counters": {"arena_reads": {"count": 0}, "sha256": {"count": 0}},
            "block_summary": [{"scope": "layer", "name": "layer_total", "count": 3,
                               "wall_ms": 300.0}],
        },
        "legend": {"block_fields": fields},
    }


class AggregateTest(unittest.TestCase):
    def test_seam_layers_excludes_layers_1_and_14_and_all_layers_does_not(self):
        module = load_script()
        data = perf([
            (0, {"a": [10.0]}),
            (1, {"a": [1.0]}),
            (2, {"a": [20.0]}),
            (14, {"a": [2.0]}),
        ])
        stats = module.aggregate(data)
        # seam-layers keeps layers 0 and 2: (10 + 20) / 2
        self.assertAlmostEqual(stats["seam_layers_total"], 15.0)
        # all-layers keeps all four: (10 + 1 + 20 + 2) / 4
        self.assertAlmostEqual(stats["all_layers_total"], 8.25)
        self.assertAlmostEqual(stats["seam_blocks"]["a"][0], 15.0)
        self.assertAlmostEqual(stats["all_blocks"]["a"][0], 8.25)
        self.assertEqual(stats["layer_positions"], 4)

    def test_probe_scope_is_the_probe_layer_timer(self):
        module = load_script()
        stats = module.aggregate(perf([(0, {"a": [10.0]})]))
        self.assertEqual(stats["probe_scope"]["wall_ms"], 300.0)
        self.assertEqual(stats["probe_scope"]["count"], 3)
        self.assertAlmostEqual(stats["probe_scope"]["ms_per_layer"], 100.0)

    def test_counters_come_from_loop_total_in_legend_order(self):
        module = load_script()
        stats = module.aggregate(perf([(0, {"a": [10.0]})]))
        self.assertEqual(stats["counters"]["launches"], 10)
        self.assertEqual(stats["counters"]["alloc_count"], 3)
        self.assertEqual(stats["counters"]["module_load_count"], 2)
        self.assertNotIn("launches", stats["counters"].get("arena_reads", {}))

    def test_load_accepts_a_run_directory_with_rank0_or_flat_layout(self):
        module = load_script()
        import tempfile
        row = perf([(0, {"a": [10.0]})])
        with tempfile.TemporaryDirectory() as root:
            for layout in ("flat", "ranked"):
                path = Path(root) / layout
                if layout == "flat":
                    path.mkdir()
                    target = path / "tp4-forward-perf-r0.json"
                else:
                    (path / "rank0").mkdir(parents=True)
                    target = path / "rank0" / "tp4-forward-perf-r0.json"
                target.write_text(json.dumps(row))
                self.assertEqual(module.load(str(path))["positions"][0]["position"], 0)
                # an explicit file path also works
                self.assertEqual(module.load(str(target))["positions"][0]["position"], 0)


if __name__ == "__main__":
    unittest.main()
