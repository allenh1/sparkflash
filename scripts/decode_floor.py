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

"""Tally what a decode step reads, stage by stage, against the DRAM floor.

The strategy's first rule is to start from the hardware floor: tally the bytes a
token must read, divide by the bandwidth the memory system delivers, and compare
the result with the measured stage. This does that for the Qwen3.8 arm straight
from the checkpoint's safetensors headers, so the tally cannot drift from what is
served, and it names each group after the profiler mark that times it.

    python3 scripts/decode_floor.py --checkpoint ~/.cache/huggingface/hub/.../snapshot
    python3 scripts/decode_floor.py --checkpoint DIR --profile serve.log --rows 1

A profiled log (SPARKFLASH_QWEN4_PROF=1) is read with --profile, which reports the
measured mean of each stage at the requested row count beside its floor. A mark
costs about 4.5 us and serialises the stream, so a profile total is an upper bound
and wall clock is what compares builds.

Only the routed experts scale with rows: a verify's row adds the experts its own
ten slots name. The dense stack is read once a step whatever the row count, which
is the whole argument for speculating.
"""

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter

DTYPES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1,
}
GIB = 1 << 30
# The rate the routed experts and the head are measured at on a GB10 node.
PEAK_GB_S = 270.0

# Tensor-name group -> the profiler mark that times it. The first match wins.
GROUPS = [
    ("experts", re.compile(r"\.layers\.\d+\.mlp\.experts\.")),
    ("shared", re.compile(r"\.mlp\.shared_expert")),
    ("router", re.compile(r"\.mlp\.gate\.weight$")),
    ("hyper", re.compile(r"_hyper_connection\.|hyper_connection_mixer\.")),
    ("gdn_branch", re.compile(r"\.linear_attn\.out_proj\.weight$")),
    ("gdn_proj", re.compile(r"\.linear_attn\.in_proj_")),
    ("gdn_conv", re.compile(r"\.linear_attn\.(conv1d|A_log|dt_bias|norm)\.")),
    ("attn_inject", re.compile(r"\.self_attn\.o_proj\.weight$")),
    ("sparse_proj", re.compile(r"\.self_attn\.")),
    ("ple_proj", re.compile(r"\.ple\.(key_proj|value_proj|conv1d|norm)")),
    ("lm_head", re.compile(r"^lm_head\.weight$")),
    ("embed", re.compile(r"embed_tokens\.weight$")),
    ("drafter", re.compile(r"^(mtp\.|fc\.|hidden_norm\.|norm\.weight$|candidate_selector\.)")),
    ("ple_table", re.compile(r"ple_embedding\.ngram_embedding")),
    ("vision", re.compile(r"^model\.visual\.")),
]


def headers(checkpoint):
    """Yield (name, dtype, shape) for every tensor in the checkpoint's shards."""
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as handle:
            weight_map = json.load(handle)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = sorted(f for f in os.listdir(checkpoint) if f.endswith(".safetensors"))
    for shard in shards:
        with open(os.path.join(checkpoint, shard), "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            yield name, entry["dtype"], entry["shape"]


# A group's bytes are not always one mark: the decode path marks the two hyper passes
# hyper_down and hyper_up back to back, and the PLE projections are part of the ple mark.
MARKS = {"hyper": ("hyper_down", "hyper_up"), "ple_proj": ("ple",)}


def measured_ms(measured, group):
    if measured is None:
        return 0.0
    names = MARKS.get(group, (group,))
    return sum(measured.get(name, 0.0) for name in names)


def group_of(name):
    for group, pattern in GROUPS:
        if pattern.search(name):
            return group
    return "other"


def tally(checkpoint):
    """Bytes each group occupies on disk, and how many experts one layer holds."""
    bytes_of = Counter()
    experts = set()
    for name, dtype, shape in headers(checkpoint):
        count = 1
        for dim in shape:
            count *= dim
        size = count * DTYPES[dtype]
        group = group_of(name)
        if group == "vision":
            continue
        bytes_of[group] += size
        if group == "experts" and name.endswith(".down_proj.weight"):
            # model.language_model.layers.<layer>.mlp.experts.<index>.down_proj.weight
            parts = name.split(".")
            experts.add((int(parts[parts.index("layers") + 1]),
                         int(parts[parts.index("experts") + 1])))
    per_layer = len({index for _, index in experts})
    return bytes_of, per_layer


def profile(path, rows):
    """The mean milliseconds each mark reports in the forwards of that many rows."""
    head = re.compile(r"qwen4 prof p(\d+) rows (\d+): ([\d.]+) ms total")
    line = re.compile(r"qwen4 prof   (\S+)\s+([\d.]+) ms")
    stages = Counter()
    totals = 0.0
    forwards = 0
    inside = False
    with open(path, errors="replace") as handle:
        for text in handle:
            match = head.search(text)
            if match:
                inside = int(match.group(2)) == rows
                if inside:
                    forwards += 1
                    totals += float(match.group(3))
                continue
            match = line.search(text)
            if match and inside:
                stages[match.group(1)] += float(match.group(2))
    if not forwards:
        return None, None
    return {name: ms / forwards for name, ms in stages.items()}, totals / forwards


def main():
    parser = argparse.ArgumentParser(description="Tally a decode step against the DRAM floor.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--profile", help="a SPARKFLASH_QWEN4_PROF=1 log to compare with")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--rate", type=float, default=PEAK_GB_S)
    options = parser.parse_args()

    bytes_of, total_experts = tally(options.checkpoint)
    if not total_experts:
        sys.exit("no routed-expert tensors in that checkpoint")
    # Ten of a layer's experts a token, in every layer; a verify's rows are separate slots
    # into the same table, so a row past the first adds the experts it alone names.
    topk = 10
    row_bytes = bytes_of["experts"] * topk / total_experts
    step_bytes = sum(
        size
        for name, size in bytes_of.items()
        # The embedding is a row lookup (KiB), the PLE table stays on the host and the
        # drafter drafts beside the target, so none of the three is a step's weight pass.
        if name not in ("experts", "ple_table", "drafter", "embed")
    )
    if options.rows > 1:
        # A run of rows draws rows * topk slots into total_experts bins in expectation.
        touched = total_experts * (1.0 - (1.0 - topk / total_experts) ** options.rows)
    else:
        touched = float(topk)
    experts_bytes = row_bytes / topk * touched
    total = step_bytes + experts_bytes

    measured, total_ms = profile(options.profile, options.rows) if options.profile else (None, None)
    print(f"{'stage':<14}{'GiB':>9}{'floor ms':>10}", end="")
    if measured:
        print(f"{'measured':>10}{'GB/s':>8}{'of floor':>9}")
    else:
        print()
    for name, size in sorted(bytes_of.items(), key=lambda item: -item[1]):
        if name in ("experts", "ple_table", "drafter", "embed"):
            continue
        floor = size / (options.rate * 1e9) * 1e3
        print(f"{name:<14}{size / GIB:9.3f}{floor:10.2f}", end="")
        if measured:
            ms = measured_ms(measured, name)
            gbs = size / ms / 1e6 if ms else 0.0
            ratio = floor / ms if ms else 0.0
            print(f"{ms:10.2f}{gbs:8.0f}{ratio:9.0%}")
        else:
            print()
    floor = experts_bytes / (options.rate * 1e9) * 1e3
    print(f"{'experts':<14}{experts_bytes / GIB:9.3f}{floor:10.2f}", end="")
    if measured:
        ms = measured_ms(measured, "experts")
        gbs = experts_bytes / ms / 1e6 if ms else 0.0
        print(f"{ms:10.2f}{gbs:8.0f}{(floor / ms if ms else 0.0):9.0%}")
    else:
        print()
    floor = total / (options.rate * 1e9) * 1e3
    print(f"{'total':<14}{total / GIB:9.3f}{floor:10.2f}", end="")
    if measured:
        print(f"{total_ms:10.2f}{total / total_ms / 1e6:8.0f}{floor / total_ms:9.0%}")
    else:
        print()
    print()
    print(f"rows {options.rows}: the dense stack is {step_bytes / GIB:.3f} GiB once, the experts "
          f"add {experts_bytes / GIB:.3f} GiB ({touched:.0f} distinct), at {options.rate:.0f} GB/s")
    print(f"one token a weight pass floors at {1e3 / (total / (options.rate * 1e9) * 1e3):.1f} tok/s")
    if options.profile:
        print("A profiled mark serialises the stream and costs about 4.5 us, so compare builds by "
              "wall clock and read this only for the shape within one run.")


if __name__ == "__main__":
    main()
