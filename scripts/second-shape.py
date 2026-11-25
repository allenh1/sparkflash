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

"""second-shape.py: the small second-shape checkpoint through the chain at its world.

The model is the fixture beside this test (`tests/fixtures/second-shape/config.json`): a
DeepSeek-shaped configuration whose width, heads, experts, top-k, layer count, compression
ratios, Engram layers and drafter taps all differ from the released model. The run has three
parts, all on one host:

  1. write the checkpoint (`write-synthetic-checkpoint`, the crate's own writer) and, for the
     device arm, stage the second shape's cubins under the released file names so the chain's
     handles keep their names;
  2. run the host reference forward (the crate's f32 `InferenceEngine`) on a few token-id
     prompts at temperature 0;
  3. start `scripts/one-host-group.sh deepseek WORLD ... --check ...` at each requested world
     and compare its greedy token ids with the reference's.

    scripts/second-shape.py run --released DIR --out DIR --binary BIN --writer BIN \
        --reference BIN [--tokenizer FILE] [--built DIR] [--build-cubins] [--world 1,2,4]
        [--port 8126]
    scripts/second-shape.py run ... --device --released-cubins DIR
    scripts/second-shape.py compare --url URL --prompts FILE --expected FILE [--model ID]
    scripts/second-shape.py compare --url URL --prompts FILE --dump FILE

The reference for a world is that world's own split. A format whose accumulation is not
regrouping-invariant (row-scaled MXFP8 / E4M3) has a served answer that depends on the split:
the per-rank partial sums differ from one global sum in the low f32 bits. The run therefore
writes the global host reference once, then starts the host arm at each world and captures that
world's ids as `host-reference-w{W}.json`. World 1 is compared with the global
`host-reference.json` and must equal it; the device group at world W is compared with that
world's file, exactly. Each world's file is also reported against the global one, with the
prompts that differ and the top-2 logprob gap at the first divergence, but that drift never
fails the run. Token ids are compared with a tolerance only under `--tie-band`, and only for a
variant in TIE_BAND_VARIANTS (each with its reason) at a band measured from the reference's own
drift between worlds.

Every requested world runs in one invocation. A world that fails is recorded and the run goes on
to the next one; the closing summary gives each world's verdict, and the exit code is 1 when any
world failed. The checkpoint and each world's reference carry a `.plan` sidecar holding a hash of
the inputs that produced them, so an unchanged variant, world, prompt list and token budget reuse
the file instead of rewriting it.

The host arm runs on a CPU host; `--device` needs a GPU host. `compare` is what `--check`
calls while the group is up; it is also usable alone against a group someone else started. With
`--dump` it writes the served ids as the same JSON list of lists the host reference writes and
exits 0; a sidecar beside the dump records the top alternatives each position saw.

For a four-node device run (SPARK_NODES set), `--out` must be an absolute directory that exists
and is writable on every node at the same path: the checkpoint, the world's staged cubins and,
when it is not under the checkpoint, the tokenizer are copied there with
`scripts/sync-nodes.py` when `SPARK_SYNC=1` (`SPARK_SSH_NAMES` optionally names the trusted SSH
name for each node). After the group, `scripts/serve-four-node.sh collect` gathers every node's
rank logs under `<out>/logs-{world}/` before the ranks stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "crates/sparkflash-deepseek-forward/tests/fixtures/second-shape/config.json"
# GLM-5.3-Flash's architecture at small widths (the base host's described-GLM fixture): KDA
# blocks split by heads, absorbed NoPE MLA with the pooled indexer, a dense prefix, the
# noaux_tc router, FP8 128 x 128 tiles with f32 scales on dynamic activations, and the MTP layer.
GLM_FIXTURE = ROOT / "crates/sparkflash-model-deepseek-v41/tests/fixtures/glm-tiny/config.json"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
# The end-of-sequence id the configuration names; the host reference stops after writing it.
# The alternatives a reference capture records at each position (the served token included).
TOPS = 8
EOS = 1
PROMPTS = [[100, 101, 102, 103], [200, 201, 202], [300, 301, 302, 303, 304, 305]]
# A recurrent variant's prompts: the ones above, then one that extends the last of them, so the
# group serves it from the kept state of the request before (prefix reuse) and the reference runs
# it fresh. The decoder variant's prompts are longer than the decoder's warm-up rows (255), so
# the recurrent block above the encoder layers sees rows the skip would have dropped.
_LONG_A = [1000 + (index * 7) % 5000 for index in range(300)]
_LONG_B = [2000 + index for index in range(270)]
VARIANT_PROMPTS = {
    "gdn-dense": PROMPTS + [PROMPTS[2] + [306, 307, 308]],
    "gdn-decoder": [_LONG_A, _LONG_B, _LONG_A + [6000 + index for index in range(20)]],
    "kda-dense": PROMPTS + [PROMPTS[2] + [306, 307, 308]],
    # Past the indexer's budget (top 8 of pools of 4) so the selection drops pools, and one that
    # extends the last so the group serves it from the kept KDA state.
    "glm": PROMPTS
    + [[400 + (index * 5) % 100 for index in range(40)]]
    + [[400 + (index * 5) % 100 for index in range(40)] + [306, 307, 308]],
}
VARIANT_PROMPTS["glm-wiring"] = VARIANT_PROMPTS["glm"]


def variant_prompts(name: str) -> list[list[int]]:
    """The prompts a variant runs: its own where it names some, PROMPTS otherwise."""
    return VARIANT_PROMPTS.get(name, PROMPTS)

# A sidecar beside a reference or the checkpoint records the hash of every input that produced the
# file. A matching hash lets the next invocation reuse the file instead of rebuilding it.
PLAN_SCHEMA = "second-shape-plan/1"
# The stored formats the checkpoint can be written in: the fixture's released 32 x 32 power-of-two
# grid, FP8 on 128 x 128 tiles with the inverse scale stored f32 or bf16, or an unquantized
# BF16 dense class in its row-major or packed arrangement. The variant is the checkpoint's own
# declaration - the writer, the reference and the group all read this one file.
GDN_GEOMETRY = {
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
STORED_FORMATS = {
    "released": None,
    # The two NVFP4 expert conventions: the dense class stays the fixture's released 32 x 32
    # power-of-two grid, and only the routed experts change format.
    "nvfp4-qwen": {"expert_dtype": "nvfp4-qwen"},
    "nvfp4-inkling": {"expert_dtype": "nvfp4-inkling"},
    # The dense class unquantized: BF16 rows as stored, and the same rows in the packed word
    # form (the escape list travels beside them). The routed experts keep their released fp4
    # arithmetic, which is the format the expert bodies in the tree read.
    "bf16": {"quant_method": "bf16", "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
    "packed-bf16": {
        "quant_method": "packed-bf16",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    },
    "fp8-f32": {"quant_method": "fp8", "scale_fmt": "f32", "expert_dtype": "fp8", "block": 128},
    "fp8-bf16": {"quant_method": "fp8", "scale_fmt": "bf16", "expert_dtype": "fp8", "block": 128},
    # The routed experts one power-of-two scale a row, every 32 values along the contraction:
    # Inkling's last eight layers, and the Engram table's and drafter head's own arithmetic.
    "mx-fp8-row": {"quant_method": "fp8", "scale_fmt": "ue8m0", "expert_dtype": "mx-fp8-row", "block": 32},
    # One stack whose expert format changes part-way up: NVFP4's Qwen convention on the first
    # four backbone layers and row-scaled MXFP8 on the last four. The dense class stays the
    # fixture's released grid; only the routed experts move.
    "joint-nvfp4-mxfp8-row": {
        "expert_dtype": "nvfp4-qwen@0-3,mx-fp8-row@4-7",
    },
    # A block with no experts: the first layer a dense-FFN block (one gated MLP 1024 wide in
    # the MoE's place, no router), the stack after it the fixture's. The dense class is BF16, so
    # the MLP runs on the dense format route; the blocks after it keep the released experts.
    "dense-mlp": {
        "quant_method": "bf16",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "text_config": {"first_k_dense_replace": 1, "intermediate_size": 1024},
    },
    # A foreign token mixer: the first block's attention grouped-query (32 query heads over 8
    # key/value heads, 128 wide, the stack's unscaled rope_theta over every float) and its FFN
    # the dense-mlp variant's gated MLP, on the BF16 dense route. The stack after it is the
    # fixture's MLA.
    "gqa-dense": {
        "quant_method": "bf16",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "text_config": {
            "first_k_dense_replace": 1,
            "intermediate_size": 1024,
            "layer_types": ["gqa_attention"] + ["compressed_attention"] * 7,
            "gqa_attn_config": {
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
            },
        },
    },
    # A recurrent token mixer: the first block a Gated DeltaNet (16 key heads, 32 value heads,
    # 128 wide each, a 4-tap conv; the Qwen3-Next spelling) with the dense-mlp variant's gated MLP
    # on the BF16 dense route. The stack after it is the fixture's MLA.
    "gdn-dense": {
        "quant_method": "bf16",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "text_config": {
            "first_k_dense_replace": 1,
            "intermediate_size": 1024,
            "layer_types": ["linear_attention"] + ["compressed_attention"] * 7,
            **GDN_GEOMETRY,
        },
    },
    # Kimi Delta Attention as the first block, in the GLM-5.3-Flash spelling (linear_attn_config:
    # 16 heads of 128, a 4-tap conv, the forget gate bounded at -5), split by heads at worlds 2
    # and 4, with the dense-mlp variant's gated MLP on the BF16 dense route. The stack after it is
    # the fixture's MLA.
    "kda-dense": {
        "quant_method": "bf16",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "text_config": {
            "first_k_dense_replace": 1,
            "intermediate_size": 1024,
            "layer_types": ["linear_attention"] + ["compressed_attention"] * 7,
            "linear_attn_config": {
                "num_heads": 16,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "kda_layers": [0],
                "full_attn_layers": [1, 2, 3, 4, 5, 6, 7],
            },
        },
    },
    # The same Gated DeltaNet as the last block, above the encoder layers, with its compress
    # ratio 0 (a recurrent block compresses nothing) and the fixture's MoE; the stack below it is
    # the fixture's.
    # GLM-5.3-Flash's own fixture, as its release stores it (GLM_FIXTURE).
    "glm": {"fixture": "glm", "host": "whole"},
    # The same stack with its FP8 weights' inputs unquantised on both arms (`activation_scheme:
    # none`): nothing between the device and the host is discontinuous, so it holds the wiring to
    # the reference where the release's dynamic scheme can only report its drift (REPORTED).
    "glm-wiring": {"fixture": "glm", "host": "whole", "activation_scheme": "none"},
    "gdn-decoder": {
        "quant_method": "bf16",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "text_config": {
            "layer_types": ["compressed_attention"] * 7 + ["linear_attention"],
            "compress_ratios": [0, 0, 2, 2, 2, 2, 1, 0, 1, 1],
            **GDN_GEOMETRY,
        },
    },
}


def fail(message: str) -> "NoReturn":
    print(f"second-shape: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("second-shape:", " ".join(command), flush=True)
    return subprocess.run(command, check=False, **kwargs)


class WorldFailure(Exception):
    """A failure confined to one world; the run records it and continues to the next world."""


def wait_port_free(port: int, seconds: int = 90) -> None:
    """Wait until nothing listens on the port. A stopped group can hold its socket for a few
    seconds after `stop` returns, and the next world's group would then fail to bind it."""
    deadline = time.monotonic() + seconds
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(1)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return
        if time.monotonic() > deadline:
            raise WorldFailure(f"port {port} is still held {seconds} s after the last group stopped")
        time.sleep(1)


def config_digest(config: bytes) -> str:
    """The hash of one variant configuration as it is written to disk."""
    return hashlib.sha256(config).hexdigest()


def checkpoint_plan(config: bytes) -> str:
    """The hash of everything that determines the checkpoint: the variant config bytes."""
    return hashlib.sha256(
        json.dumps(
            {"schema": PLAN_SCHEMA, "kind": "checkpoint", "config": config_digest(config)},
            sort_keys=True,
        ).encode()
    ).hexdigest()


def reference_plan(
    config: bytes, world: int, prompts: list[list[int]], max_tokens: int
) -> str:
    """The hash of everything that determines one world's host reference file."""
    return hashlib.sha256(
        json.dumps(
            {
                "schema": PLAN_SCHEMA,
                "kind": "reference",
                "config": config_digest(config),
                "world": world,
                "prompts": prompts,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def write_plan(path: Path, plan: str) -> None:
    """Write the sidecar that records one plan hash beside the file it describes."""
    path.write_text(json.dumps({"schema": PLAN_SCHEMA, "plan": plan}) + "\n")


def read_plan(path: Path) -> str | None:
    """The plan hash a sidecar records, or None when it is missing or unreadable."""
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != PLAN_SCHEMA:
        return None
    plan = document.get("plan")
    return plan if isinstance(plan, str) else None


def checkpoint_reusable(checkpoint: Path, sidecar: Path, config: bytes) -> bool:
    """True when the stamp matches and the kept checkpoint carries the config bytes it covers."""
    if not checkpoint.is_dir() or read_plan(sidecar) != checkpoint_plan(config):
        return False
    try:
        return (checkpoint / "config.json").read_bytes() == config
    except OSError:
        return False


def reference_reusable(reference: Path, sidecar: Path, plan: str) -> bool:
    """True only when the reference and its sidecar exist and the sidecar records this plan."""
    return reference.is_file() and read_plan(sidecar) == plan


def world_summary(outcomes: list[tuple[int, bool, str]]) -> list[str]:
    """One line per requested world, in order, with PASS or FAIL and the one-line reason."""
    lines = []
    for world, passed, reason in outcomes:
        verdict = "PASS" if passed else "FAIL"
        # A failure reason may wrap over several lines; the summary keeps it on one.
        lines.append(f"second-shape: world {world} {verdict}: {' '.join(reason.split())}")
    return lines


def summarise(outcomes: list[tuple[int, bool, str]]) -> int:
    """Print the summary block and return 1 when any world failed, 0 when all passed."""
    print("second-shape: summary")
    for line in world_summary(outcomes):
        print(line)
    return 0 if all(passed for _world, passed, _reason in outcomes) else 1


def variant_config(name: str, out: Path) -> Path:
    """Write the fixture's configuration with the variant's stored format, and return its path."""
    declared = STORED_FORMATS[name]
    if declared is not None and declared.get("fixture") == "glm":
        # The base's own kernels a stated stack still runs (the rows embedding, the head, the
        # sampler) are compiled at the second shape's width and vocabulary, so the GLM stack is
        # written at those two; every other width stays the fixture's.
        document = json.loads(GLM_FIXTURE.read_text())
        shape = json.loads(FIXTURE.read_text())["text_config"]
        for key in ("hidden_size", "vocab_size"):
            document["text_config"][key] = shape[key]
        # Four ranks split each attention's output projection and each FFN's intermediate on
        # 128-wide FP8 scale tiles, so every split width must hold four tiles: eight heads of
        # 64 on both attentions, and the second shape's expert intermediate. The fixture's
        # narrower widths stay its own world-1 tests'.
        document["text_config"]["num_attention_heads"] = 8
        document["text_config"]["num_key_value_heads"] = 8
        document["text_config"]["linear_attn_config"]["num_heads"] = 8
        document["text_config"]["moe_intermediate_size"] = shape["moe_intermediate_size"]
        if "activation_scheme" in declared:
            document["quantization_config"]["activation_scheme"] = declared["activation_scheme"]
        path = out / f"config-{name}.json"
        path.write_text(json.dumps(document, indent=2) + "\n")
        return path
    document = json.loads(FIXTURE.read_text())
    if declared is not None:
        declared = dict(declared)
        block = declared.pop("block", None)
        # A variant may also change the stack's shape (which blocks carry experts).
        document["text_config"].update(declared.pop("text_config", {}))
        quantization = document["quantization_config"]
        quantization.update(declared)
        if "quant_method" in declared:
            # A tile release states its 128 x 128 block and the row-scaled variant its 32; an
            # unquantized BF16 class has no scale plane at all, so it keeps the fixture's
            # 32 x 32 declaration.
            if block is None:
                block = 32 if declared["quant_method"] in ("bf16", "packed-bf16") else TILE_BLOCK
            quantization["weight_block_size"] = [block, block]
    path = out / f"config-{name}.json"
    path.write_text(json.dumps(document, indent=2))
    return path


def stage_cubins(released: Path, built: Path, out: Path, world: int) -> None:
    """Every released cubin, then a build of the second shape over each name it covers.

    `scripts/instances.py --world-builds` states what the world loads under each released name:
    the four-rank build `<stem>_s2`, or `<stem>_s2w<world>` for a cubin whose rows carry a field
    that changes with the world (the expert intermediate, the vocabulary shard). Every name the
    shape covers is replaced, and a build that is listed but not on disk refuses the world by
    name: a released cubin left under a second shape is compiled for another width, and its first
    launch reads past a row instead of refusing.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import instances  # noqa: PLC0415 - the reader lives beside this script

    try:
        plan = instances.world_builds(instances.load(), world)
    except SystemExit as error:
        raise WorldFailure(str(error)) from None
    out.mkdir(parents=True, exist_ok=True)
    for cubin in released.glob("*.cubin"):
        shutil.copy2(cubin, out / cubin.name)
    missing = [cubin for cubin in plan.values() if not (built / f"{cubin}.cubin").is_file()]
    if missing:
        raise WorldFailure(
            f"{len(missing)} second-shape builds for {world} ranks are not under {built} "
            f"({', '.join(sorted(missing)[:6])}{', ...' if len(missing) > 6 else ''}); "
            "run scripts/build-cubins.sh first"
        )
    for stem, cubin in plan.items():
        shutil.copy2(built / f"{cubin}.cubin", out / f"{stem}.cubin")
    staged = len(plan)
    # The NVFP4 wrappers are a new body, not a second shape of a released cubin: their instance
    # names are already the names the launch asks for, so they are staged as they are built.
    for cubin in built.glob("nvfp4_prefill_sm121_*.cubin"):
        shutil.copy2(cubin, out / cubin.name)
        staged += 1
    print(f"second-shape: staged {staged} second-shape cubins for {world} ranks under {out}")


def rank_failure(logs: Path | None) -> str | None:
    """The first line of a rank log under `logs` that says the rank died, if any: a rank that
    failed at load never becomes ready, and waiting out the timeout hides nothing."""
    if logs is None:
        return None
    for log in sorted(logs.glob("**/rank*.log")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if any(
                mark in line
                for mark in (
                    "device_failure",
                    "panicked at",
                    "forward refused on rank",
                    "the server cannot answer",
                )
            ):
                return f"{log.name}: {line.strip()}"
    return None


def wait_ready(url: str, seconds: int = 600, logs: Path | None = None) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        failure = rank_failure(logs)
        if failure is not None:
            raise WorldFailure(f"a rank of the group at {url} failed: {failure}")
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=3) as answer:
                if answer.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    raise WorldFailure(f"the group at {url} never became ready")


def parse_token_id(token: object) -> int:
    if isinstance(token, str) and token.startswith("token_id:"):
        try:
            return int(token[len("token_id:") :])
        except ValueError:
            pass
    fail(f"the server did not return token ids (got {token!r})")


def request_completion(
    args: argparse.Namespace, prompt: list[int], max_tokens: int | None = None
) -> tuple[list[int], list[list[list[float]]]]:
    """One greedy completion, its ids, and the top alternatives each generated position saw.

    Under `--ids token-ids` (the device arm, which serves no logprobs) the ids are the choice's
    `token_ids` and no alternatives are known."""
    fields: dict[str, object] = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens if max_tokens is None else max_tokens,
        "temperature": 0.0,
    }
    if args.ids == "token-ids":
        fields["return_token_ids"] = True
    else:
        # The engine's Top(n) selection includes the served token: the served token and its
        # nearest rivals, so a divergence can be placed among the reference's own choices.
        fields["logprobs"] = TOPS
        fields["return_tokens_as_token_ids"] = True
    body = json.dumps(fields).encode()
    request = urllib.request.Request(
        args.url + "/v1/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as answer:
            reply = json.loads(answer.read())
    except urllib.error.HTTPError as error:
        fail(f"the server refused the completion ({error.code}): {error.read().decode(errors='replace')}")
    choice = reply["choices"][0]
    if args.ids == "token-ids":
        ids = choice.get("token_ids")
        if not isinstance(ids, list) or not all(isinstance(token, int) for token in ids):
            fail(f"the server did not return token_ids (got {ids!r})")
        if EOS in ids:
            ids = ids[: ids.index(EOS) + 1]
        return ids, []
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    positions = logprobs.get("top_logprobs") or []
    ids: list[int] = []
    tops: list[list[list[float]]] = []
    for index, token in enumerate(tokens):
        ids.append(parse_token_id(token))
        alternatives: list[list[float]] = []
        if index < len(positions) and isinstance(positions[index], dict):
            for alternative, value in positions[index].items():
                alternatives.append([float(parse_token_id(alternative)), float(value)])
        # Descending logprob; the first entry is the served token, the second its nearest rival.
        alternatives.sort(key=lambda entry: entry[1], reverse=True)
        tops.append(alternatives)
    # The server may generate past the end-of-sequence marker; the reference stops there.
    if EOS in ids:
        cut = ids.index(EOS) + 1
        ids = ids[:cut]
        tops = tops[:cut]
    return ids, tops


def first_divergence(got: list[int], want: list[int]) -> int:
    for index, (left, right) in enumerate(zip(got, want)):
        if left != right:
            return index
    return min(len(got), len(want))


def report_divergence(
    got: list[int],
    want: list[int],
    tops: list[list[list[float]]],
    *,
    want_label: str = "reference",
    got_label: str = "device",
) -> None:
    position = first_divergence(got, want)
    print(f"  first divergence at position {position}")
    if position < len(want):
        print(f"    {want_label} token {want[position]}")
    if position < len(got):
        print(f"    {got_label} token {got[position]}")
    alternatives = tops[position] if position < len(tops) else []
    if len(alternatives) >= 2:
        (top_token, top_logprob), (second_token, second_logprob) = alternatives[:2]
        print(
            f"    top-2 at position {position}: token {int(top_token)} "
            f"{top_logprob:.9g} over token {int(second_token)} {second_logprob:.9g}; "
            f"gap {top_logprob - second_logprob:.9g}"
        )
    else:
        print(f"    top-2 at position {position}: the server returned fewer than two alternatives")


def compare(args: argparse.Namespace) -> int:
    prompts = json.loads(Path(args.prompts).read_text())
    try:
        wait_ready(args.url, args.ready_timeout)
    except WorldFailure as error:
        fail(str(error))
    if args.dump:
        captured: list[list[int]] = []
        gaps: list[list[list[list[float]]]] = []
        for index, prompt in enumerate(prompts):
            ids, tops = request_completion(args, prompt)
            captured.append(ids)
            gaps.append(tops)
            print(f"second-shape: prompt {index} captured ({len(ids)} tokens)")
        Path(args.dump).write_text(json.dumps(captured, indent=2) + "\n")
        Path(str(args.dump) + ".tops.json").write_text(json.dumps(gaps) + "\n")
        print(f"second-shape: wrote {len(captured)} prompt(s) to {args.dump}")
        return 0
    if not args.expected:
        fail("--expected is required unless --dump is set")
    expected = json.loads(Path(args.expected).read_text())
    if len(expected) != len(prompts):
        fail(f"{args.expected} holds {len(expected)} prompt(s), {args.prompts} holds {len(prompts)}")
    expected_tops = None
    if args.tie_band is not None:
        tops_file = Path(str(args.expected) + ".tops.json")
        if not tops_file.is_file():
            fail(f"--tie-band needs the reference's alternatives sidecar {tops_file}")
        expected_tops = json.loads(tops_file.read_text())
    passed = True
    for index, prompt in enumerate(prompts):
        got, tops = request_completion(args, prompt)
        want = expected[index]
        status = "ok" if got == want else "MISMATCH"
        if got != want and expected_tops is not None:
            ties = forced_ties(args, prompt, got, want, expected_tops[index], args.tie_band)
            status = "MISMATCH" if ties is None else f"ok at {len(ties)} tie(s) {ties}"
        if status == "MISMATCH":
            passed = False
        print(f"second-shape: prompt {index} {status} ({len(got)} tokens)")
        if got != want:
            print(f"  reference {want}")
            print(f"  device    {got}")
            report_divergence(got, want, tops)
    print(f"second-shape: VERDICT {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def forced_ties(
    args: argparse.Namespace,
    prompt: list[int],
    got: list[int],
    want: list[int],
    want_tops: list[list[list[float]]],
    band: float,
) -> list[int] | None:
    """The positions where the device took another of the reference's near-tied choices, or None.

    For a variant the device cannot match bit for bit (`run --tie-band`), any two accumulation
    orders part at the reference's near-ties, so a free-running comparison stops meaning anything
    after the first one. Every position is still compared under the reference's own context: at
    a divergence the device's id must be among the reference's recorded alternatives there, at
    most `band` nats under the reference's choice; the device then continues from the
    reference's prefix through that position, and the next stretch is held to the reference
    again, to the last token. Any other divergence is a mismatch."""
    ties: list[int] = []
    base = 0
    while got != want[base:]:
        at = base + first_divergence(got, want[base:])
        if at >= len(want) or at - base >= len(got):
            print(f"  prompt length differs from the reference at position {at}")
            return None
        alternatives = want_tops[at] if at < len(want_tops) else []
        device = got[at - base]
        if not alternatives or int(alternatives[0][0]) != want[at]:
            print(f"  position {at}: the reference recorded no alternatives for its choice")
            return None
        top_lp = alternatives[0][1]
        device_lp = next(
            (value for token, value in alternatives if int(token) == device), None
        )
        if device_lp is None or top_lp - device_lp > band:
            where = (
                f"not among the reference's {len(alternatives)} recorded choices"
                if device_lp is None
                else f"{top_lp - device_lp:.6g} nats under it"
            )
            print(
                f"  position {at}: device {device} against reference {want[at]} is {where} "
                f"(band {band:.6g}): not a tie"
            )
            return None
        gap = top_lp - device_lp
        print(f"  position {at}: a tie (gap {gap:.6g} <= {band:.6g}); forcing the reference prefix")
        ties.append(at)
        base = at + 1
        if base >= len(want):
            return ties
        got, _ = request_completion(args, prompt + want[:base], len(want) - base)
    return ties


def differing_prompts(expected: list[list[int]], global_ids: list[list[int]]) -> list[int]:
    """Indexes of prompts whose ids differ, including a prompt only one side wrote."""
    count = max(len(expected), len(global_ids))
    differing = []
    for index in range(count):
        left = expected[index] if index < len(expected) else None
        right = global_ids[index] if index < len(global_ids) else None
        if left != right:
            differing.append(index)
    return differing


def report_world(
    world: int,
    expected_file: Path,
    expected: list[list[int]],
    global_ids: list[list[int]],
    tops: list[list[list[list[float]]]],
) -> list[int]:
    """Print the world's drift from the global reference and return the differing prompts."""
    differing = differing_prompts(expected, global_ids)
    equals = "yes" if not differing else "no"
    print(
        f"second-shape: world {world}: expected {expected_file.name}; "
        f"equals global reference: {equals}"
    )
    for index in differing:
        print(f"  prompt {index} differs")
        # The expected ids are the host arm's output at this world; the global ids are the
        # single-process sum. The top-2 gap below is the host arm's own at the divergence.
        got = expected[index] if index < len(expected) else []
        want = global_ids[index] if index < len(global_ids) else []
        world_tops = tops[index] if index < len(tops) else []
        report_divergence(got, want, world_tops, want_label="global", got_label="host")
    return differing


def tree_bytes(paths: list[str]) -> int:
    """Bytes the lead would send: regular files under every path, symlinks as links."""
    total = 0
    for root in paths:
        if os.path.islink(root) or os.path.isfile(root):
            total += os.lstat(root).st_size
            continue
        for directory, _dirs, files in os.walk(root):
            for name in files:
                candidate = os.path.join(directory, name)
                total += 0 if os.path.islink(candidate) else os.lstat(candidate).st_size
    return total


def human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000:
            return f"{value:.0f} {unit}"
        value /= 1000
    return f"{value:.1f} TB"


def sync_to_nodes(nodes: list[str], paths: list[str]) -> None:
    """Copy the paths to every node at the same absolute path, then print each node and size."""
    names = os.environ.get("SPARK_SSH_NAMES", "").split()
    if names and len(names) < len(nodes):
        raise WorldFailure(
            f"SPARK_SSH_NAMES names {len(names)} node(s); the group needs {len(nodes)}"
        )
    # A rank that runs on this host already has the paths: the copy would go over ssh to itself,
    # which the lead Spark does not trust, and it would be the same bytes the harness just wrote.
    local = {int(rank) for rank in os.environ.get("SPARK_LOCAL_RANKS", "").split() if rank}
    command = [sys.executable, "-I", str(ROOT / "scripts/sync-nodes.py")]
    remote = 0
    for index, node in enumerate(nodes):
        if index in local:
            continue
        # The destination may be a fast-network address; HostKeyAlias checks the key already
        # trusted for the node's SSH name, so no new host key is written.
        alias = names[index] if names else node.rpartition("@")[2]
        command += ["--host", f"{node},{alias}"]
        remote += 1
    if remote == 0:
        print("second-shape: every rank runs on this host; nothing to copy")
        return
    for path in paths:
        command += ["--path", path]
    command += ["--ssh-arg=-o", "--ssh-arg=BatchMode=yes"]
    if run(command).returncode != 0:
        raise WorldFailure("the checkpoint, cubins or tokenizer did not reach every node")
    size = tree_bytes(paths)
    for node in nodes:
        print(f"second-shape: synced {node}: {size} bytes ({human_bytes(size)})")


def compare_command(
    url: str,
    prompts: Path,
    *,
    expected: Path | None = None,
    dump: Path | None = None,
    max_tokens: int,
) -> str:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "compare",
        "--url",
        url,
        "--prompts",
        str(prompts),
        "--max-tokens",
        str(max_tokens),
        "--ready-timeout",
        "60",
    ]
    if expected is not None:
        command += ["--expected", str(expected)]
    if dump is not None:
        command += ["--dump", str(dump)]
    return shlex.join(command)


def host_group_env(logs: Path, binary: str) -> dict[str, str]:
    return {
        **os.environ,
        "SPARKFLASH_BIN": binary,
        "ONE_HOST_LOGS": str(logs),
        # The fixture streams its 2.8 GB checkpoint; the default 6 GiB a rank reserve
        # would refuse a host whose page cache is holding the released model.
        "ONE_HOST_SKIP_MEMORY": "1",
        # The second shape's cubins are built from the instance list, not pinned in the
        # released baseline; the open manifest records their hashes.
        "SPARKFLASH_UNPINNED_SECOND_SHAPE": "1",
    }


def require_nccl_library() -> Path:
    """The NCCL library a device group loads, checked before the run starts.

    The launcher's rank bootstrap loads NCCL before it opens any communicator and refuses on
    every rank when it cannot. A window that reaches that point has spent its downtime on a
    missing environment variable, so the path is the operator's and is named, never guessed.
    """
    value = os.environ.get("SPARKFLASH_NCCL_LIB", "")
    if not value:
        fail(
            "a device run needs SPARKFLASH_NCCL_LIB pointing at the directory that holds "
            "libnccl.so; the launcher refuses at the NCCL bootstrap before any layer without "
            "it (for example the production install's target/deps/nccl)"
        )
    path = Path(value)
    if not path.is_dir():
        fail(f"SPARKFLASH_NCCL_LIB={value} is not a directory")
    return path


def env_for_world(
    args: argparse.Namespace,
    out: Path,
    checkpoint: Path,
    tokenizer: str,
    staged: Path,
    world: int,
    node_list: list[str],
    peer_list: list[str],
) -> dict[str, str]:
    """The environment for one world's device group, declared the same way at every world."""
    return {
        **os.environ,
        "SPARKFLASH_BIN": args.binary,
        "SPARKFLASH_CHECKPOINT": str(checkpoint),
        "SPARKFLASH_TOKENIZER": tokenizer,
        "SPARKFLASH_CUBINS": str(staged),
        "SPARKFLASH_LOG_ROOT": str(out / f"logs-serve-{world}"),
        # Collect needs the launch's run directory; a generated stamp would differ.
        "SPARKFLASH_RUN_ID": f"w{world}",
        "ONE_HOST_LOGS": str(out / f"logs-{world}"),
        "SPARK_NODES": " ".join(node_list),
        "SPARK_PEERS": " ".join(peer_list),
        "SPARK_WORLD": str(world),
        "SPARK_LOCAL_RANKS": "0",
        "SPARKFLASH_HTTP_BIND": f"0.0.0.0:{args.port}",
        # The launcher loads NCCL before any layer; the path is the operator's, carried here.
        "SPARKFLASH_NCCL_LIB": os.environ.get("SPARKFLASH_NCCL_LIB", ""),
        # The certified head screen is built for the released 5,120-value rows (160 blocks of
        # 32); this shape serves its full 4,096-value rows (128 blocks) through that head.
        "SPARKFLASH_HEAD_SCREEN": "0",
        # Every variant is the base's: a family that still has its own typed pack (GLM's) is
        # served by the base from its description, which is what the variant proves.
        "SPARKFLASH_ON_BASE": "1",
        # The second shape's cubins are built from the instance list, not pinned in the
        # released baseline; the open manifest records their hashes.
        "SPARKFLASH_UNPINNED_SECOND_SHAPE": "1",
    }


def run_world(
    args: argparse.Namespace,
    out: Path,
    checkpoint: Path,
    config: bytes,
    built: Path,
    prompts_file: Path,
    global_file: Path,
    global_ids: list[list[int]],
    tokenizer: str,
    world: int,
) -> str:
    """Run one world's reference, arm and comparison, and return the reason it passed.

    A failure that belongs to this world raises WorldFailure so the caller can record it and go
    on to the next requested world.
    """
    url = f"http://127.0.0.1:{args.port}"
    # The expected file for this world is the host arm's own output at this split. It is captured
    # from a live group before any device arm is launched. The sidecar records the inputs that
    # produced it, so an unchanged checkpoint, world, prompt list and token budget reuse it.
    expected_file = out / f"host-reference-w{world}.json"
    tops_file = Path(str(expected_file) + ".tops.json")
    plan_file = Path(str(expected_file) + ".plan")
    plan = reference_plan(config, world, variant_prompts(args.stored_format), args.max_tokens)
    # A stack that states its own wiring runs on the host's whole-model path only: the served
    # host arm's sharded seams are the release's arithmetic and refuse it. Its every world is
    # held to the whole-model reference, the unsplit sum the producer computes.
    whole_only = (STORED_FORMATS.get(args.stored_format) or {}).get("host") == "whole"
    if whole_only:
        if not args.device:
            raise WorldFailure(
                f"{args.stored_format} states its own wiring, which the host runs whole only; "
                "its arms are the device's (--device)"
            )
        print(
            f"second-shape: {args.stored_format} is held to the whole-model reference at "
            f"world {world}; the host has no sharded arm for a stated stack"
        )
        expected_file = global_file
    elif reference_reusable(expected_file, plan_file, plan) and tops_file.is_file():
        print(
            f"second-shape: reusing {expected_file.name}; "
            "the plan covers the same checkpoint, world, prompts and max-tokens"
        )
    else:
        capture = compare_command(
            url, prompts_file, dump=expected_file, max_tokens=args.max_tokens
        )
        wait_port_free(args.port)
        result = run(
            [
                str(ROOT / "scripts/one-host-group.sh"),
                "deepseek",
                str(world),
                "--model",
                MODEL,
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                tokenizer,
                "--port",
                str(args.port),
                "--check",
                capture,
                "--apply",
            ],
            env=host_group_env(out / f"logs-host-ref-{world}", args.binary),
        )
        if result.returncode != 0:
            raise WorldFailure(f"the {world}-rank host arm could not write its reference")
        write_plan(plan_file, plan)
    differing = []
    if not whole_only:
        expected = json.loads(expected_file.read_text())
        tops = json.loads(tops_file.read_text())
        differing = report_world(world, expected_file, expected, global_ids, tops)
    if world == 1 and differing:
        prompts = ", ".join(str(index) for index in differing)
        raise WorldFailure(
            "the one-rank host arm does not equal the global reference at prompt(s) "
            f"{prompts}"
        )

    # A shape that divides by the world has a cubin a world; stage this world's own directory
    # so the loader's released file names resolve to the build for this rank count.
    staged = out / "cubins"
    if args.device:
        staged = out / f"cubins-w{world}"
        stage_cubins(Path(args.released_cubins), built, staged, world)

    # The device arm is held to the split's own file at every world; world 1's file is
    # checked against the global reference above, so the anchor is not lost.
    if args.device:
        expected_for_check = expected_file
    else:
        expected_for_check = global_file if world == 1 else expected_file
    # The device arm takes NCCL collectives, so its group is the one the deployed launcher
    # starts: one process a node over ssh. SPARK_NODES names the nodes in rank order; the first
    # world of them are the group. Without it the device group runs on one host, which is the
    # host arm's exchange and is refused by the device run header.
    all_nodes = os.environ.get("SPARK_NODES", "").split()
    if args.device and all_nodes:
        node_list = all_nodes[:world]
        peers = os.environ.get("SPARK_PEERS") or " ".join(all_nodes)
        peer_list = peers.replace(",", " ").split()[:world]
        if len(node_list) != world or len(peer_list) != world:
            raise WorldFailure(
                f"a {world}-rank device group needs {world} SPARK_NODES and SPARK_PEERS "
                "addresses"
            )
        if os.environ.get("SPARK_SYNC") == "1":
            tokenizer_file = str(Path(tokenizer).resolve())
            sync_paths = [str(checkpoint), str(staged)]
            checkpoint_prefix = str(checkpoint) + os.sep
            if tokenizer_file != str(checkpoint) and not tokenizer_file.startswith(
                checkpoint_prefix
            ):
                sync_paths.append(tokenizer_file)
            print(
                f"second-shape: syncing {len(sync_paths)} path(s) to the first {world} node(s)"
            )
            sync_to_nodes(node_list, sync_paths)
        else:
            print("second-shape: node sync skipped (set SPARK_SYNC=1 to distribute the run)")
        group_env = env_for_world(
            args, out, checkpoint, tokenizer, staged, world, node_list, peer_list
        )
        launcher = str(ROOT / "scripts/serve-four-node.sh")
        started = False
        agreed = 1
        try:
            wait_port_free(args.port)
            result = run([launcher, "launch"], env=group_env)
            started = True
            if result.returncode != 0:
                raise WorldFailure(f"the {world}-rank device group did not start")
            wait_ready(url, logs=out / f"logs-serve-{world}")
            agreed = run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "compare",
                    "--url",
                    url,
                    "--prompts",
                    str(prompts_file),
                    "--expected",
                    str(expected_for_check),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--ready-timeout",
                    "60",
                    "--ids",
                    "token-ids",
                    *(("--tie-band", str(args.tie_band)) if args.tie_band is not None else ()),
                ]
            ).returncode
        finally:
            if started:
                # Pull every worker's own log before the ranks stop; a head-only read keeps
                # rank 0's file and has already hidden a worker failure.
                run([launcher, "collect"], env=group_env)
                run([launcher, "stop"], env=group_env)
        if agreed != 0:
            raise WorldFailure(
                f"the {world}-rank device group did not agree with "
                f"{expected_for_check.name}"
            )
        return f"the device group agreed with {expected_for_check.name}"

    # One host: the host arm itself is the check when no device arm is asked for. World 1 is
    # held to the global reference; every other world to its own split's file.
    check = compare_command(
        url,
        prompts_file,
        expected=expected_for_check,
        max_tokens=args.max_tokens,
    )
    wait_port_free(args.port)
    result = run(
        [
            str(ROOT / "scripts/one-host-group.sh"),
            "deepseek",
            str(world),
            "--model",
            MODEL,
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            tokenizer,
            *(("--cubins", str(staged)) if args.device else ()),
            "--port",
            str(args.port),
            "--check",
            check,
            "--apply",
        ],
        env=host_group_env(out / f"logs-{world}", args.binary),
    )
    if result.returncode != 0:
        raise WorldFailure(
            f"the {world}-rank group did not agree with {expected_for_check.name}"
        )
    return f"the host group agreed with {expected_for_check.name}"


# The variants the device arm cannot match the host reference on bit for bit, each with its
# reason. Only these may run with --tie-band, and each one's band is measured, never chosen: twice
# the largest |logprob| difference of shared alternatives between the host reference at one rank
# and at two and four, rounded up (the drift the reference's own regrouping shows). Every other
# variant is held to exact ids.
_MIXER = (
    "the device mixer's CUDA expf and tiled projection order, which the host reference cannot "
    "reproduce, move the mix coefficients in their last bits"
)
_FAST_MATH = (
    "the foreign part's device kernels use fast-math __expf and split partials the host "
    "reference cannot reproduce"
)
TIE_BAND_VARIANTS = {
    # Row-scaled E4M3 codes change their f32 bits when the routed-expert sum is regrouped (the
    # released E2M1 arm does not), measured by
    # `the_row_scaled_expert_split_rounds_where_the_released_arm_regroups_exactly`.
    "mx-fp8-row": "the regrouped E4M3 routed-expert sum is not regrouping-invariant",
    "joint-nvfp4-mxfp8-row": "the regrouped E4M3 routed-expert sum is not regrouping-invariant",
    "gqa-dense": _FAST_MATH,
    "gdn-dense": _FAST_MATH,
    "gdn-decoder": _FAST_MATH,
    # The device splits KDA's heads and sums o_proj's BF16 partials; the host runs every head and
    # rounds the whole sum once.
    "kda-dense": _FAST_MATH + "; the device sums the split heads' BF16 partials",
    # The stated GLM stack: the device mixer's CUDA expf and tiled projection order move layer 0's
    # mix coefficients by ~1e-7, and the BF16 stream turns that into flipped ulps. With the inputs
    # unquantised the flips stay small (glm-wiring); with the release's dynamic scheme a flipped
    # ulp can move an E4M3 code or a tile's scale, about 12x a dense FFN, to a 6% stream difference
    # by layer 7 on the near-uniform fixture, so glm is reported, not gated (REPORTED).
    "glm-wiring": _MIXER,
    "glm": _MIXER + "; the release's dynamic FP8 activations amplify each flipped BF16 ulp",
}

# The variants whose verdict is printed but does not fail the run, each with its reason. Their
# wiring is held by a gating twin; what they add is the drift of an arithmetic the device cannot
# match bit for bit on this fixture.
REPORTED = {
    "glm": "the release's dynamic FP8 activations amplify the mixer's rounding past any tie on the "
    "near-uniform fixture; glm-wiring gates the same stack, the producer capture its numerics",
}


def run_test(args: argparse.Namespace) -> int:
    if args.tie_band is not None and args.stored_format not in TIE_BAND_VARIANTS:
        fail(
            f"--tie-band is for a variant the device cannot match bit for bit "
            f"({', '.join(sorted(TIE_BAND_VARIANTS))}); {args.stored_format} is held to exact ids"
        )
    if args.tie_band is not None:
        print(f"tie band {args.tie_band} for {args.stored_format}: "
              f"{TIE_BAND_VARIANTS[args.stored_format]}")
    if args.device:
        require_nccl_library()
    all_nodes = os.environ.get("SPARK_NODES", "").split()
    four_node = args.device and bool(all_nodes)
    if four_node and not os.path.isabs(args.out):
        fail(
            "a four-node device run needs an absolute --out that every node shares at the same "
            f"path; got {args.out}"
        )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint"
    # The stamp covers the variant config bytes the writer is given. A matching stamp and a
    # checkpoint whose own config.json is those bytes keep it; anything else rewrites it.
    config = variant_config(args.stored_format, out)
    config_bytes = config.read_bytes()
    checkpoint_stamp = out / "checkpoint.plan"
    if checkpoint_reusable(checkpoint, checkpoint_stamp, config_bytes):
        print(
            f"second-shape: reusing {checkpoint.name}; "
            "the plan covers the same variant config"
        )
    else:
        if checkpoint.exists():
            shutil.rmtree(checkpoint)
        result = run(
            [args.writer, str(config), str(checkpoint), "7", "4"],
        )
        if result.returncode != 0:
            fail("the synthetic checkpoint did not write")
        written = checkpoint / "config.json"
        if not written.is_file() or written.read_bytes() != config_bytes:
            fail(f"{written} is not the variant config the plan covers")
        write_plan(checkpoint_stamp, checkpoint_plan(config_bytes))
    if args.build_cubins:
        result = run(
            [str(ROOT / "scripts/build-cubins.sh")],
            env={**os.environ, "SPARKFLASH_CUBINS": str(out / "built")},
        )
        if result.returncode != 0:
            fail("the cubins did not build")
    built = Path(args.built) if args.built else out / "built"
    if args.device and not args.released_cubins:
        fail("--device needs --released-cubins: the staged directory keeps the released names")

    prompts_file = out / "prompts.json"
    global_file = out / "host-reference.json"
    prompts_file.write_text(json.dumps(variant_prompts(args.stored_format)))
    # The reference hashes Engram n-grams over the tokenizer's compressed map, as the device does;
    # a stack with no Engram layer reads no map, so the reference is given none.
    tokenizer = args.tokenizer or str(Path(args.released) / "tokenizer.json")
    engram = bool(json.loads(config_bytes).get("engram_layer_ids"))
    result = run(
        [
            args.reference,
            str(checkpoint),
            str(prompts_file),
            str(args.max_tokens),
            *(("--tokenizer", tokenizer) if engram else ()),
            # The reference's own alternatives, which a stated stack's tie band reads (its every
            # world is held to this whole-model file).
            "--tops",
            str(global_file) + ".tops.json",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        fail("the host reference failed")
    global_file.write_text(result.stdout)
    global_ids = json.loads(global_file.read_text())
    print(f"second-shape: host reference {result.stdout.strip()}")

    worlds = [int(world) for world in args.world.split(",") if world.strip()]
    if not worlds:
        fail("--world names no rank count")
    outcomes: list[tuple[int, bool, str]] = []
    for world in worlds:
        try:
            reason = run_world(
                args,
                out,
                checkpoint,
                config_bytes,
                built,
                prompts_file,
                global_file,
                global_ids,
                tokenizer,
                world,
            )
        except WorldFailure as error:
            outcomes.append((world, False, str(error)))
        else:
            outcomes.append((world, True, reason))
    verdict = summarise(outcomes)
    if verdict != 0 and args.stored_format in REPORTED:
        print(
            f"second-shape: {args.stored_format} is reported, not gated: "
            f"{REPORTED[args.stored_format]}"
        )
        return 0
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run_parser = subs.add_parser("run")
    run_parser.add_argument("--released", required=True, help="the released checkpoint (tokenizer + cubins)")
    run_parser.add_argument(
        "--released-cubins", help="the released cubin directory (--device only)"
    )
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--binary", required=True, help="the release sparkflash binary")
    run_parser.add_argument("--writer", required=True, help="the write-synthetic-checkpoint example")
    run_parser.add_argument("--reference", required=True, help="the host-reference example")
    run_parser.add_argument("--built", help="the second-shape cubin directory (default OUT/built)")
    run_parser.add_argument("--tokenizer", help="tokenizer.json (default RELEASED/tokenizer.json)")
    run_parser.add_argument("--build-cubins", action="store_true")
    run_parser.add_argument(
        "--device",
        action="store_true",
        help="run the device arm (needs four GPUs); default is the four-rank host arm",
    )
    run_parser.add_argument(
        "--world",
        default="4",
        help="comma-separated rank counts to run (default 4; the proof runs 1,2,4)",
    )
    run_parser.add_argument(
        "--stored-format",
        default="released",
        choices=sorted(STORED_FORMATS),
        help="how the checkpoint stores its weights (default released: the fixture's own declaration)",
    )
    run_parser.add_argument("--port", type=int, default=8126)
    run_parser.add_argument("--max-tokens", type=int, default=8)
    run_parser.add_argument(
        "--tie-band",
        type=float,
        help="hold the device arm to the reference at every position but its near-ties (see "
        "compare --tie-band); only for a variant in TIE_BAND_VARIANTS, at its measured band",
    )
    run_parser.set_defaults(func=run_test)

    compare_parser = subs.add_parser("compare")
    compare_parser.add_argument("--url", required=True)
    compare_parser.add_argument("--prompts", required=True)
    compare_parser.add_argument("--expected", help="the ids to compare with (required without --dump)")
    compare_parser.add_argument(
        "--dump",
        help="write the served ids as a JSON list of lists, plus FILE.tops.json, and exit 0",
    )
    compare_parser.add_argument("--model", default=MODEL)
    compare_parser.add_argument("--max-tokens", type=int, default=8)
    compare_parser.add_argument("--ready-timeout", type=int, default=600)
    compare_parser.add_argument(
        "--ids",
        default="logprobs",
        choices=("logprobs", "token-ids"),
        help="where the ids come from: logprobs (with the top-2 gaps) or the choice's token_ids "
        "(the device arm, which serves no logprobs)",
    )
    compare_parser.add_argument(
        "--tie-band",
        type=float,
        help="accept a divergence only where the device's id is at most this many nats under the "
        "reference's choice among the reference's recorded alternatives, then continue from the "
        "reference's prefix (a variant in TIE_BAND_VARIANTS; default: exact ids)",
    )
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())