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

"""seam-diff.py: where a device forward leaves its host reference, layer by layer and row by row.

When a device group's token ids differ from the host reference's, the ids do not say why. This
script reads the two boundary dumps and shows the shape of the difference, which does:

  * every row drifts, from an early layer on: the wiring is wrong;
  * a few rows jump at a router or a selection layer and the rest hold: a near-tie flipped
    under rounding, and `router` confirms it from the margins.

The inputs are two directories of one prompt's first prefill pass:

  HOST    the host reference's boundary table (`host-reference CKPT PROMPTS N DUMP_DIR`):
          `L<nn>-<boundary>.f32` files of f32 and `shapes.json` giving each file's
          [rows, ...] shape.
  DEVICE  the serving rank's dump (`SPARKFLASH_ROWS_DUMP=DIR`, with `SPARKFLASH_WARMUP=0` so
          the first pass is the prompt's): `L<nn>-<boundary>.bin` files of f32.

A device pass may hold fewer rows than the host table (the last prompt row is the first token
step); the rows the device holds are compared.

    scripts/seam-diff.py layers HOST DEVICE [--pair DEV=HOST ...] [--flip-factor 10]
    scripts/seam-diff.py rows HOST DEVICE [--pair DEV=HOST] [--layers 3,4]
    scripts/seam-diff.py router HOST DEVICE --checkpoint DIR --weight NAME [--bias NAME]
        [--score sigmoid|softmax|none] [--top-k K] [--pair DEV=HOST] [--layers 3,4]

A pair names a device boundary and the host boundary it is compared with; the default is
`stream=residual_out`, each layer's stream out. `DEV` may end in `:bf16` for a dump stored as
BF16. `layers` prints each layer's median and largest row distance (relative L2) and the rows
past `--flip-factor` times the median. `rows` prints every row's distance. `router` recomputes
a router's choice from both sides' inputs to it (`--pair`, for example
`attn_normed=pre_ffn_hidden`) with the checkpoint's gate weight, and prints each row whose
top-k set differs with the host's margin between its last chosen expert and the next. `NAME` is
a tensor name with `{layer}` in it, read from the checkpoint's safetensors files. A flip at a
margin near the rounding is a tie; a flip at a wide margin is a fault upstream.

It needs only the standard library. A dump of a few dozen rows at a few thousand columns takes
seconds.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import statistics
import struct
import sys
from pathlib import Path

DEFAULT_PAIR = "stream=residual_out"


class Refusal(Exception):
    """An input this script cannot compare, said by name."""


def read_f32(path: Path) -> array.array:
    values = array.array("f")
    values.frombytes(path.read_bytes())
    return values


def widen_bf16(raw: bytes) -> array.array:
    """BF16 is the top half of an f32: widen each value exactly."""
    halves = array.array("H")
    halves.frombytes(raw)
    count = len(halves)
    return array.array(
        "f", struct.unpack(f"<{count}f", struct.pack(f"<{count}I", *(h << 16 for h in halves)))
    )


def parse_pair(text: str) -> tuple[str, bool, str]:
    """`DEV[:bf16]=HOST` as (device boundary, stored as BF16, host boundary)."""
    device, sep, host = text.partition("=")
    if not sep or not device or not host:
        raise Refusal(f"--pair {text!r} is not DEV=HOST")
    bf16 = device.endswith(":bf16")
    if bf16:
        device = device[: -len(":bf16")]
    return device, bf16, host


def host_layers(shapes: dict, host_boundary: str) -> list[str]:
    suffix = f"-{host_boundary}.f32"
    layers = sorted(name[: -len(suffix)] for name in shapes if name.endswith(suffix))
    if not layers:
        raise Refusal(f"the host table holds no layer's {host_boundary}")
    return layers


def pick_layers(layers: list[str], wanted: str | None) -> list[str]:
    if wanted is None:
        return layers
    names = {f"L{int(item):02d}" for item in wanted.split(",")}
    missing = sorted(names - set(layers))
    if missing:
        raise Refusal(f"--layers names {', '.join(missing)}, which the host table does not hold")
    return [layer for layer in layers if layer in names]


def load_pair(
    host: Path, device: Path, shapes: dict, layer: str, pair: tuple[str, bool, str]
) -> tuple[array.array, array.array, int, int]:
    """One layer's (device values, host values, row width, rows the device holds)."""
    device_boundary, bf16, host_boundary = pair
    host_name = f"{layer}-{host_boundary}.f32"
    shape = shapes.get(host_name)
    if shape is None:
        raise Refusal(f"shapes.json does not name {host_name}")
    width = math.prod(shape[1:])
    device_file = device / f"{layer}-{device_boundary}.bin"
    if not device_file.is_file():
        raise Refusal(f"the device dump holds no {device_file.name} in {device}")
    got = widen_bf16(device_file.read_bytes()) if bf16 else read_f32(device_file)
    want = read_f32(host / host_name)
    if len(want) != shape[0] * width:
        raise Refusal(f"{host_name} holds {len(want)} values, not the {shape} shapes.json states")
    if len(got) % width or len(got) // width > shape[0]:
        raise Refusal(
            f"{device_file.name} holds {len(got)} values, not rows of {width} within the host's "
            f"{shape[0]}"
        )
    return got, want, width, len(got) // width


def row_distances(got: array.array, want: array.array, width: int, rows: int) -> list[float]:
    """Each row's relative L2 distance of `got` from `want`."""
    distances = []
    for row in range(rows):
        a = got[row * width : (row + 1) * width]
        b = want[row * width : (row + 1) * width]
        difference = math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))
        norm = math.sqrt(sum(y * y for y in b))
        distances.append(difference / norm if norm else difference)
    return distances


def layer_table(
    host: Path, device: Path, pair: tuple[str, bool, str], wanted: str | None
) -> list[tuple[str, list[float]]]:
    shapes = json.loads((host / "shapes.json").read_text())
    table = []
    for layer in pick_layers(host_layers(shapes, pair[2]), wanted):
        got, want, width, rows = load_pair(host, device, shapes, layer, pair)
        table.append((layer, row_distances(got, want, width, rows)))
    return table


def cmd_layers(args: argparse.Namespace) -> int:
    for text in args.pair or [DEFAULT_PAIR]:
        pair = parse_pair(text)
        print(f"{pair[0]} against {pair[2]}")
        for layer, distances in layer_table(args.host, args.device, pair, args.layers):
            median = statistics.median(distances)
            past = [row for row, value in enumerate(distances) if value > args.flip_factor * median]
            line = (
                f"  {layer}: median {median:.3e} max {max(distances):.3e} over "
                f"{len(distances)} rows; {len(past)} row(s) past {args.flip_factor:g}x the median"
            )
            if past:
                line += f": {past}"
            print(line)
    return 0


def cmd_rows(args: argparse.Namespace) -> int:
    for text in args.pair or [DEFAULT_PAIR]:
        pair = parse_pair(text)
        print(f"{pair[0]} against {pair[2]}")
        for layer, distances in layer_table(args.host, args.device, pair, args.layers):
            print(f"  {layer}: " + " ".join(f"{value:.0e}" for value in distances))
    return 0


def tensor_files(checkpoint: Path) -> dict[str, Path] | None:
    """The index's tensor-to-file map, or None for a checkpoint without one."""
    index = checkpoint / "model.safetensors.index.json"
    if not index.is_file():
        return None
    return {
        name: checkpoint / file
        for name, file in json.loads(index.read_text())["weight_map"].items()
    }


def read_tensor(checkpoint: Path, name: str) -> tuple[array.array, list[int]]:
    """One F32 or BF16 tensor from the checkpoint's safetensors files, as f32."""
    files = tensor_files(checkpoint)
    if files is None:
        candidates = sorted(checkpoint.glob("*.safetensors"))
    elif name in files:
        candidates = [files[name]]
    else:
        raise Refusal(f"the checkpoint's index does not name {name}")
    for path in candidates:
        with path.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(length))
            entry = header.get(name)
            if entry is None:
                continue
            begin, end = entry["data_offsets"]
            handle.seek(8 + length + begin)
            raw = handle.read(end - begin)
        if entry["dtype"] == "F32":
            values = array.array("f")
            values.frombytes(raw)
        elif entry["dtype"] == "BF16":
            values = widen_bf16(raw)
        else:
            raise Refusal(f"{name} is stored as {entry['dtype']}; this reads F32 and BF16")
        return values, entry["shape"]
    raise Refusal(f"no safetensors file in {checkpoint} holds {name}")


def route(
    row: array.array, weight: array.array, bias: array.array | None, experts: int, width: int,
    score: str,
) -> tuple[list[int], list[float]]:
    """A row's experts in the router's order, with the value each was ranked by."""
    logits = [
        sum(w * x for w, x in zip(weight[expert * width : (expert + 1) * width], row))
        for expert in range(experts)
    ]
    if score == "sigmoid":
        scores = [1.0 / (1.0 + math.exp(-value)) for value in logits]
    elif score == "softmax":
        top = max(logits)
        exps = [math.exp(value - top) for value in logits]
        total = sum(exps)
        scores = [value / total for value in exps]
    else:
        scores = logits
    ranked = [value + (bias[expert] if bias is not None else 0.0) for expert, value in enumerate(scores)]
    order = sorted(range(experts), key=lambda expert: -ranked[expert])
    return order, [ranked[expert] for expert in order]


def cmd_router(args: argparse.Namespace) -> int:
    if "{layer}" not in args.weight:
        raise Refusal("--weight needs {layer} in the tensor name")
    pair = parse_pair(args.pair[0] if args.pair else DEFAULT_PAIR)
    shapes = json.loads((args.host / "shapes.json").read_text())
    total = 0
    for layer in pick_layers(host_layers(shapes, pair[2]), args.layers):
        index = int(layer[1:])
        try:
            weight, shape = read_tensor(args.checkpoint, args.weight.format(layer=index))
        except Refusal as refusal:
            print(f"  {layer}: no router ({refusal})")
            continue
        if len(shape) != 2:
            raise Refusal(f"{args.weight.format(layer=index)} has shape {shape}, not [experts, width]")
        experts, width = shape
        bias = None
        if args.bias:
            bias, _ = read_tensor(args.checkpoint, args.bias.format(layer=index))
        if args.top_k >= experts:
            raise Refusal(f"--top-k {args.top_k} leaves no next expert among {experts}")
        got, want, row_width, rows = load_pair(args.host, args.device, shapes, layer, pair)
        if row_width != width:
            raise Refusal(
                f"{layer}: the {pair[2]} rows are {row_width} wide and the router reads {width}"
            )
        flips = []
        for row in range(rows):
            span = slice(row * width, (row + 1) * width)
            host_order, host_value = route(want[span], weight, bias, experts, width, args.score)
            device_order, _ = route(got[span], weight, bias, experts, width, args.score)
            if set(host_order[: args.top_k]) != set(device_order[: args.top_k]):
                margin = host_value[args.top_k - 1] - host_value[args.top_k]
                flips.append((row, host_order[: args.top_k], device_order[: args.top_k], margin))
        total += len(flips)
        print(f"  {layer}: {len(flips)} of {rows} row(s) choose a different top-{args.top_k}")
        for row, host_choice, device_choice, margin in flips:
            print(
                f"    row {row}: host {host_choice} device {device_choice}; "
                f"host margin to the next expert {margin:.2e}"
            )
    print(f"{total} flipped row(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("host", type=Path, help="the host reference's boundary table")
        command.add_argument("device", type=Path, help="the serving rank's rows dump")
        command.add_argument("--pair", action="append", metavar="DEV[:bf16]=HOST")
        command.add_argument("--layers", metavar="N,N", help="only these layer indices")

    layers = commands.add_parser("layers", help="each layer's median and largest row distance")
    common(layers)
    layers.add_argument("--flip-factor", type=float, default=10.0)
    layers.set_defaults(run=cmd_layers)

    rows = commands.add_parser("rows", help="every row's distance, a layer a line")
    common(rows)
    rows.set_defaults(run=cmd_rows)

    router = commands.add_parser("router", help="rows whose router choice differs, with margins")
    common(router)
    router.add_argument("--checkpoint", type=Path, required=True)
    router.add_argument("--weight", required=True, metavar="NAME", help="gate weight, with {layer}")
    router.add_argument("--bias", metavar="NAME", help="a bias added to the score, with {layer}")
    router.add_argument("--score", choices=["sigmoid", "softmax", "none"], default="sigmoid")
    router.add_argument("--top-k", type=int, default=1)
    router.set_defaults(run=cmd_router)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except Refusal as refusal:
        print(f"seam-diff: {refusal}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
