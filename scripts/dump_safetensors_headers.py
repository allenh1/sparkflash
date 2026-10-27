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

"""Dump a safetensors checkpoint's headers as name -> {"dtype", "shape"} JSON.

This is the header source for the host attention-operand gate
(crates/sparkflash-cli/src/tp4_operands.rs, --tensor-index): it reads each shard's header only, so
the full 96k-tensor dump of the released fifty-shard checkpoint is a few megabytes and no tensor
data is read. The checkpoint directory must hold model.safetensors.index.json.

    ./scripts/dump_safetensors_headers.py /path/to/checkpoint > released-headers.json
"""
import json
import struct
import sys


def headers_for(path, cache):
    if path not in cache:
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            cache[path] = json.loads(handle.read(length))
    return cache[path]


def main(argv):
    if len(argv) != 2:
        print("usage: dump_safetensors_headers.py CHECKPOINT_DIR > headers.json", file=sys.stderr)
        return 2
    root = argv[1].rstrip("/")
    with open(root + "/model.safetensors.index.json") as handle:
        index = json.load(handle)
    cache = {}
    out = {}
    for name, shard in index["weight_map"].items():
        meta = headers_for(root + "/" + shard, cache)[name]
        out[name] = {"dtype": meta["dtype"], "shape": list(meta["shape"])}
    json.dump(out, sys.stdout, sort_keys=True)
    print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
