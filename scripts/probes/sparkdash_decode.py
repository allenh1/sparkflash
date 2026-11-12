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

"""Run sparkDash's DecodeBench against a served model, request for request.

Why this exists: docs/performance.md compares this tree against the MiaAI-Lab SGLang build using
sparkDash 1.8.8's DecodeBench prompts and rate definition, and it could not say where the client
was. This is the client. It takes the prompt set from a sparkDash checkout's own
src/shared/llmPrompts.js, so the prompts are sparkDash's rather than a retyping of them:

    node --input-type=module -e '
      import { pickDecodeBenchPrompts, DECODE_BENCH_TYPES } from "/path/to/sparkdash/src/shared/llmPrompts.js";
      import fs from "fs";
      const out = {};
      for (const type of DECODE_BENCH_TYPES) {
        out[type] = {};
        for (const n of [1, 2, 4, 8, 16]) out[type][n] = pickDecodeBenchPrompts(n, type);
      }
      fs.writeFileSync("/tmp/sparkdash_prompts.json", JSON.stringify(out));'

Protocol: one wave per concurrency level, each stream its own prompt ([pickDecodeBenchPrompts]
mints distinct code tasks and a request-id prefix otherwise), 256 tokens, greedy and thinking off
(this tree implements neither ignore_eos nor min_tokens, which sparkDash sends, so a stream stops
at its natural end). Per-stream rate is the engine's own post-first-token decode rate; aggregate is
completion tokens over the wave's wall span, which includes every stream's cold prefill.

Measure on a fresh epoch. A two-hour-old instance read 1.7x to 3.2x slower on prefill than a fresh
boot of the same binary (docs/serving.md, D107).
"""

import argparse
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


def one(base, prompt, tokens):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            answer = json.loads(response.read())
    except Exception as error:  # noqa: BLE001 - a refused stream is a datum
        return {"error": f"{type(error).__name__}: {error}", "tokens": 0, "wall": time.time() - started}
    stats = answer.get("x_deepseek_spark_stats") or {}
    timing = stats.get("timing") or {}
    spec = stats.get("dspark_speculative") or {}
    usage = answer.get("usage") or {}
    return {
        "tokens": usage.get("completion_tokens", 0),
        "prompt": usage.get("prompt_tokens", 0),
        "decode": timing.get("decode_tok_per_s") or 0.0,
        "ttft": timing.get("ttft_ms") or 0.0,
        "per_step": spec.get("mean_tokens_per_verify_step") or 0.0,
        "wall": time.time() - started,
    }


def idle(base, timeout=300):
    """Waits until the server reports no request in flight, so a wave is never measured over one."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10) as response:
                for line in response.read().decode().splitlines():
                    if line.startswith("sparkflash_num_requests_running "):
                        if float(line.split()[1]) == 0:
                            return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SPARKFLASH_BASE_URL", "http://127.0.0.1:8000"),
        help="the server to drive (default $SPARKFLASH_BASE_URL, else http://127.0.0.1:8000)",
    )
    parser.add_argument("--prompts", default="/tmp/sparkdash_prompts.json")
    parser.add_argument("--types", default="prose,code,structured,json")
    parser.add_argument("--levels", default="1,2,4,8,16")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--out", default="sparkdash-decode.json")
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    prompts = json.load(open(args.prompts))
    warm = one(args.base_url, prompts["prose"]["1"][0], 32)
    print("warm-up", {k: warm.get(k) for k in ("tokens", "wall")}, flush=True)

    rows = []
    for type_name in args.types.split(","):
        for n in [int(x) for x in args.levels.split(",") if x]:
            idle(args.base_url)
            wave = prompts[type_name][str(n)][:n]
            started = time.time()
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(lambda p: one(args.base_url, p, args.tokens), wave))
            span = time.time() - started
            ok = [r for r in results if "error" not in r]
            total = sum(r["tokens"] for r in ok)
            row = {
                "type": type_name,
                "streams": n,
                "per_stream": statistics.median(r["decode"] for r in ok) if ok else 0.0,
                "aggregate": total / span if span else 0.0,
                "tokens": total,
                "span": span,
                "errors": len(results) - len(ok),
                "per_step": statistics.median(r["per_step"] for r in ok) if ok else 0.0,
                "first_error": next((r["error"] for r in results if "error" in r), None),
            }
            rows.append(row)
            print(
                "%-10s c%-2d per-stream %6.1f  aggregate %6.1f  tokens %4d  span %5.1fs  errors %d"
                % (type_name, n, row["per_stream"], row["aggregate"], total, span, row["errors"]),
                flush=True,
            )
            json.dump(
                {"label": args.label, "base_url": args.base_url, "rows": rows},
                open(args.out, "w"),
                indent=1,
            )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
