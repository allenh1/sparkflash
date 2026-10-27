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

"""Extracts the eight Engram head rows from the released checkpoint, read-only.

This is how golden_rows.txt was produced. It is here so the bit-for-bit check can be repeated
against RELEASED BYTES rather than only against the embedded copy, and so nobody has to trust
a transcript for where those bytes came from.

What it reads, and nothing else:
    <snapshot>/model.safetensors.index.json, to find the two shards
    the two shard headers, to find the data offset of the two tensors
    layers.{1,14}.engram.embed.weight at data0 + id*256   (256 bytes per row)
    layers.{1,14}.engram.embed.scale  at data0 + id*8     (8 bytes per row)

Eight rows, 2,112 bytes of row data and 2,672 bytes of headers in total. It opens the shards
read-only and never writes; it does not use mmap and does not touch any other byte.

    python3 probe_rows.py --snapshot <snapshot directory> > golden_rows.txt

The default below is the standard Hugging Face cache layout for the released repo, on the
host that holds it. It is a default, not a promise: the checkpoint is half a terabyte and lives
on the inference host, so on any other machine pass --snapshot pointing at the snapshot
directory, or anywhere that holds the two shards plus model.safetensors.index.json. The
bit-for-bit check does NOT need the checkpoint at all - golden_rows.txt already holds the eight
rows - so only re-deriving the bytes from released weights needs it.

The ids are fixed and deliberate: 0 and 1 (the head of each table), 4095 and 12345678 (mid),
and 384006167 on layer 1 plus 384016681 on layer 14, which are the LAST rows of their tables.
The last rows are the ones that make the "the tail is data, not padding" claim checkable, so
do not drop them when re-running.
"""

import argparse
import json
import os
import struct
import sys

DEFAULT_SNAPSHOT = ("/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash"
                    "/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d")
PLAN = [
    ("l1", [0, 1, 4095, 12345678, 384006167], "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"),
    ("l14", [0, 7, 384016681], "layers.14.engram.embed.weight", "layers.14.engram.embed.scale"),
]


def tensor_info(base, weight_map, name):
    path = os.path.join(base, weight_map[name])
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    entry = header[name]
    return {
        "path": path,
        "data0": 8 + header_len + entry["data_offsets"][0],
        "shape": entry["shape"],
        "dtype": entry["dtype"],
        "header_bytes": 8 + header_len,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    base = args.snapshot
    index_path = os.path.join(base, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print("cannot find " + index_path, file=sys.stderr)
        print("", file=sys.stderr)
        print("The checkpoint is not at the default cache path on this machine. Pass", file=sys.stderr)
        print("--snapshot <directory> pointing at the released snapshot (or any directory with", file=sys.stderr)
        print("the shards and model.safetensors.index.json).", file=sys.stderr)
        print("", file=sys.stderr)
        print("The bit-for-bit check does not need the checkpoint: golden_rows.txt already", file=sys.stderr)
        print("holds the eight rows, so run golden_decode.py --check-against <rust file>.", file=sys.stderr)
        return 2
    index = json.load(open(index_path))
    weight_map = index["weight_map"]

    info = {}
    header_bytes = 0
    for _, _, weight_name, scale_name in PLAN:
        for name in (weight_name, scale_name):
            if name not in info:
                info[name] = tensor_info(base, weight_map, name)
                header_bytes += info[name]["header_bytes"]

    print("# Extracted by probe_rows.py from %s" % base)
    print("# tensor shapes and dtypes, for the record:")
    for name in sorted(info):
        print("#   %s %s %s" % (name, info[name]["dtype"], info[name]["shape"]))
    print("HEADER_BYTES %d" % header_bytes)

    data_bytes = 0
    for layer, ids, weight_name, scale_name in PLAN:
        for row_id in ids:
            with open(info[weight_name]["path"], "rb") as handle:
                handle.seek(info[weight_name]["data0"] + row_id * 256)
                weight = handle.read(256)
            with open(info[scale_name]["path"], "rb") as handle:
                handle.seek(info[scale_name]["data0"] + row_id * 8)
                scale = handle.read(8)
            if len(weight) != 256 or len(scale) != 8:
                raise SystemExit("short read for %s row %d" % (layer, row_id))
            data_bytes += 264
            print("ROW %s %d %s %s" % (layer, row_id, weight.hex(), scale.hex()))
    print("DATA_BYTES_READ %d" % data_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
