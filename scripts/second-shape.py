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

The host arm runs on a CPU host; `--device` needs a GPU host. `compare` is what `--check`
calls while the group is up; it is also usable alone against a group someone else started.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "crates/sparkflash-deepseek-forward/tests/fixtures/second-shape/config.json"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
# The end-of-sequence id the configuration names; the host reference stops after writing it.
EOS = 1
PROMPTS = [[100, 101, 102, 103], [200, 201, 202], [300, 301, 302, 303, 304, 305]]
# The stored formats the checkpoint can be written in: the fixture's released 32 x 32 power-of-two
# grid, or FP8 on 128 x 128 tiles with the inverse scale stored f32 or bf16. The variant is the
# checkpoint's own declaration - the writer, the reference and the group all read this one file.
STORED_FORMATS = {
    "released": None,
    "fp8-f32": {"quant_method": "fp8", "scale_fmt": "f32", "expert_dtype": "fp8"},
    "fp8-bf16": {"quant_method": "fp8", "scale_fmt": "bf16", "expert_dtype": "fp8"},
    # The two NVFP4 expert conventions: the dense class stays the fixture's released 32 x 32
    # power-of-two grid, and only the routed experts change format.
    "nvfp4-qwen": {"expert_dtype": "nvfp4-qwen"},
    "nvfp4-inkling": {"expert_dtype": "nvfp4-inkling"},
}
# The block an FP8 tile release stores: one inverse scale a 128 x 128 tile.
TILE_BLOCK = 128


def fail(message: str) -> "NoReturn":
    print(f"second-shape: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("second-shape:", " ".join(command), flush=True)
    return subprocess.run(command, check=False, **kwargs)


def variant_config(name: str, out: Path) -> Path:
    """Write the fixture's configuration with the variant's stored format, and return its path."""
    document = json.loads(FIXTURE.read_text())
    declared = STORED_FORMATS[name]
    if declared is not None:
        quantization = document["quantization_config"]
        quantization.update(declared)
        if "quant_method" in declared:
            quantization["weight_block_size"] = [TILE_BLOCK, TILE_BLOCK]
    path = out / f"config-{name}.json"
    path.write_text(json.dumps(document, indent=2))
    return path


def stage_cubins(released: Path, built: Path, out: Path) -> None:
    """Every released cubin, then the second shape's over its released name."""
    out.mkdir(parents=True, exist_ok=True)
    for cubin in released.glob("*.cubin"):
        shutil.copy2(cubin, out / cubin.name)
    staged = 0
    for cubin in built.glob("*_s2.cubin"):
        target = out / (cubin.name[: -len("_s2.cubin")] + ".cubin")
        shutil.copy2(cubin, target)
        staged += 1
    if staged == 0:
        fail(f"no second-shape cubins under {built}; run scripts/build-cubins.sh first")
    print(f"second-shape: staged {staged} second-shape cubins under {out}")


def wait_ready(url: str, seconds: int = 600) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=3) as answer:
                if answer.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    fail(f"the group at {url} never became ready")


def generated_ids(choice: dict) -> list[int]:
    logprobs = choice.get("logprobs") or {}
    tokens = logprobs.get("tokens") or []
    ids: list[int] = []
    for token in tokens:
        if isinstance(token, str) and token.startswith("token_id:"):
            ids.append(int(token[len("token_id:") :]))
        else:
            fail(f"the server did not return token ids (got {token!r})")
    return ids


def compare(args: argparse.Namespace) -> int:
    prompts = json.loads(Path(args.prompts).read_text())
    expected = json.loads(Path(args.expected).read_text())
    wait_ready(args.url, args.ready_timeout)
    passed = True
    for index, prompt in enumerate(prompts):
        body = json.dumps(
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            }
        ).encode()
        request = urllib.request.Request(
            args.url + "/v1/completions",
            data=body,
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as answer:
            reply = json.loads(answer.read())
        got = generated_ids(reply["choices"][0])
        # The server may generate past the end-of-sequence marker; the reference stops there.
        if EOS in got:
            got = got[: got.index(EOS) + 1]
        want = expected[index]
        status = "ok" if got == want else "MISMATCH"
        if got != want:
            passed = False
        print(f"second-shape: prompt {index} {status} ({len(got)} tokens)")
        if got != want:
            print(f"  reference {want}")
            print(f"  device    {got}")
    print(f"second-shape: VERDICT {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def run_test(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint"
    cubins = out / "cubins"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    config = variant_config(args.stored_format, out)
    result = run(
        [args.writer, str(config), str(checkpoint), "7", "4"],
    )
    if result.returncode != 0:
        fail("the synthetic checkpoint did not write")
    if args.build_cubins:
        result = run(
            [str(ROOT / "scripts/build-cubins.sh")],
            env={**os.environ, "SPARKFLASH_CUBINS": str(out / "built")},
        )
        if result.returncode != 0:
            fail("the cubins did not build")
    if args.device:
        if not args.released_cubins:
            fail("--device needs --released-cubins: the staged directory keeps the released names")
        built = Path(args.built) if args.built else out / "built"
        stage_cubins(Path(args.released_cubins), built, cubins)

    prompts_file = out / "prompts.json"
    expected_file = out / "host-reference.json"
    prompts_file.write_text(json.dumps(PROMPTS))
    result = run(
        [args.reference, str(checkpoint), str(prompts_file), str(args.max_tokens)],
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        fail("the host reference failed")
    expected_file.write_text(result.stdout)
    print(f"second-shape: host reference {result.stdout.strip()}")

    tokenizer = args.tokenizer or str(Path(args.released) / "tokenizer.json")
    worlds = [int(world) for world in args.world.split(",") if world.strip()]
    if not worlds:
        fail("--world names no rank count")
    for world in worlds:
        url = f"http://127.0.0.1:{args.port}"
        compare_cmd = (
            f"{sys.executable} {Path(__file__).resolve()} compare --url {url} "
            f"--prompts {prompts_file} --expected {expected_file} --max-tokens {args.max_tokens} "
            f"--ready-timeout 60"
        )
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
                *(("--cubins", str(cubins)) if args.device else ()),
                "--port",
                str(args.port),
                "--check",
                compare_cmd,
                "--apply",
            ],
            env={
                **os.environ,
                "SPARKFLASH_BIN": args.binary,
                "ONE_HOST_LOGS": str(out / f"logs-{world}"),
                # The fixture streams its 2.8 GB checkpoint; the default 6 GiB a rank reserve
                # would refuse a host whose page cache is holding the released model.
                "ONE_HOST_SKIP_MEMORY": "1",
                # The second shape's cubins are built from the instance list, not pinned in the
                # released baseline; the open manifest records their hashes.
                "SPARKFLASH_UNPINNED_SECOND_SHAPE": "1",
            },
        )
        if result.returncode != 0:
            fail(f"the {world}-rank group did not agree with the host reference")
    return 0


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
    run_parser.set_defaults(func=run_test)

    compare_parser = subs.add_parser("compare")
    compare_parser.add_argument("--url", required=True)
    compare_parser.add_argument("--prompts", required=True)
    compare_parser.add_argument("--expected", required=True)
    compare_parser.add_argument("--model", default=MODEL)
    compare_parser.add_argument("--max-tokens", type=int, default=8)
    compare_parser.add_argument("--ready-timeout", type=int, default=600)
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
