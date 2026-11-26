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

"""seam-diff.py on a written pair of dumps: the layer table, a BF16 pair and a router flip.

Host-only: no server, no GPU, no network.
"""

import array
import contextlib
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "seam_diff", os.path.join(os.path.dirname(HERE), "seam-diff.py")
)
seam_diff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seam_diff)

ROWS, WIDTH = 5, 4


def write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(array.array("f", values).tobytes())


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], list[float]]]) -> None:
    header, blob = {}, b""
    for name, (shape, values) in tensors.items():
        raw = array.array("f", values).tobytes()
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = seam_diff.main([str(item) for item in argv])
    return code, out.getvalue()


class SeamDiff(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        self.host, self.device, self.checkpoint = root / "host", root / "dev", root / "ckpt"
        for path in (self.host, self.device, self.checkpoint):
            path.mkdir()
        # Host rows: row r is [1, r/10, 0, 0]. The device holds four of the five rows, each a
        # hair off, and row 2 is far off.
        host = [value for row in range(ROWS) for value in (1.0, row / 10.0, 0.0, 0.0)]
        device = [value * 1.001 for value in host[: 4 * WIDTH]]
        device[2 * WIDTH + 1] = 0.9
        write_f32(self.host / "L00-residual_out.f32", host)
        write_f32(self.host / "L00-pre_ffn_hidden.f32", host)
        (self.host / "shapes.json").write_text(
            json.dumps(
                {"L00-residual_out.f32": [ROWS, 2, 2], "L00-pre_ffn_hidden.f32": [ROWS, WIDTH]}
            )
        )
        write_f32(self.device / "L00-stream.bin", device)
        write_f32(self.device / "L00-attn_normed.bin", device)
        # Two experts: expert 0 reads column 0 less 0.5, expert 1 reads column 1. The host's row 2
        # scores 0.5 against 0.2 and the device's 0.5005 against 0.9.
        write_safetensors(
            self.checkpoint / "model.safetensors",
            {
                "layers.0.gate.weight": ([2, WIDTH], [0.5, 0, 0, 0, 0, 1.0, 0, 0]),
                "layers.0.gate.bias": ([2], [0.0, 0.0]),
            },
        )

    def tearDown(self) -> None:
        self.dir.cleanup()

    def test_layers_names_the_row_that_left(self) -> None:
        code, text = run("layers", self.host, self.device)
        self.assertEqual(code, 0, text)
        self.assertIn("over 4 rows; 1 row(s) past 10x the median: [2]", text)
        self.assertIn("median 1.000e-03", text)

    def test_rows_prints_each_row(self) -> None:
        code, text = run("rows", self.host, self.device, "--layers", "0")
        self.assertEqual(code, 0, text)
        self.assertEqual(len(text.splitlines()[1].split()), 1 + 4)

    def test_a_bf16_dump_is_widened_exactly(self) -> None:
        values = array.array("f", [1.0, 0.5, -2.0, 0.0] * 4)
        halves = array.array("H", (word >> 16 for word in struct.unpack("<16I", values.tobytes())))
        (self.device / "L00-out.bin").write_bytes(halves.tobytes())
        write_f32(self.host / "L00-ffn_out.f32", list(values) + [0.0] * WIDTH)
        shapes = json.loads((self.host / "shapes.json").read_text())
        shapes["L00-ffn_out.f32"] = [ROWS, WIDTH]
        (self.host / "shapes.json").write_text(json.dumps(shapes))
        code, text = run("layers", self.host, self.device, "--pair", "out:bf16=ffn_out")
        self.assertEqual(code, 0, text)
        self.assertIn("median 0.000e+00 max 0.000e+00 over 4 rows", text)

    def test_router_reports_the_flip_and_the_host_margin(self) -> None:
        code, text = run(
            "router", self.host, self.device, "--checkpoint", self.checkpoint,
            "--weight", "layers.{layer}.gate.weight", "--bias", "layers.{layer}.gate.bias",
            "--score", "none", "--pair", "attn_normed=pre_ffn_hidden",
        )
        self.assertEqual(code, 0, text)
        self.assertIn("1 of 4 row(s) choose a different top-1", text)
        self.assertIn("row 2: host [0] device [1]; host margin to the next expert 3.00e-01", text)

    def test_a_missing_dump_and_a_wrong_width_are_refused_by_name(self) -> None:
        code, text = run("layers", self.host, self.device, "--pair", "absent=residual_out")
        self.assertEqual(code, 1)
        self.assertIn("holds no L00-absent.bin", text)
        write_f32(self.device / "L00-stream.bin", [0.0] * 7)
        code, text = run("layers", self.host, self.device)
        self.assertEqual(code, 1)
        self.assertIn("not rows of 4", text)


if __name__ == "__main__":
    unittest.main()
