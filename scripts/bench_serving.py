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

"""Measure a served model's decode rate, interleaving the configurations.

Why interleaved: this arm's numbers move by 30-40% with how much of the host
PLE table has paged in, so a measurement taken before another is not comparable
to it. Every round here measures *every* level in turn, and the report is the
median of the rounds, which is the only shape that survived the noise.

Usage:

    python3 scripts/bench_serving.py --levels 1,2,4 --tokens 256 --rounds 3
    python3 scripts/bench_serving.py --url http://192.0.2.100:8010 \
        --model qwen3.8-flash-next-nvfp4 --levels 1,2 --rounds 5

Nothing about the model is assumed beyond the OpenAI completion shape and the
optional x_deepseek_spark_stats block; a server that does not send the stats
still reports wall-clock rates.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

DEFAULT_PROMPT = (
    "Summarize the design of a GPU prefill kernel in detail, including how its "
    "blocks map onto the SMs, why the weights are read once, and what limits it."
)


def one(url, model, prompt, tokens, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0.0,
        }
    ).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        answer = json.loads(response.read())
    wall = time.time() - started
    stats = answer.get("x_deepseek_spark_stats") or {}
    timing = stats.get("timing") or {}
    spec = stats.get("dspark_speculative") or {}
    return {
        "wall": wall,
        "tokens": answer["usage"]["completion_tokens"],
        "prompt": answer["usage"]["prompt_tokens"],
        "decode": timing.get("decode_tok_per_s") or 0.0,
        "accept": spec.get("acceptance_rate") or 0.0,
        "per_step": spec.get("mean_tokens_per_verify_step") or 0.0,
    }


def level(url, model, prompt, tokens, count, timeout):
    runs = [None] * count
    errors = []

    def work(index):
        try:
            runs[index] = one(url, model, prompt, tokens, timeout)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            errors.append(f"{type(error).__name__}: {error}")

    started = time.time()
    threads = [threading.Thread(target=work, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.time() - started
    if errors:
        return {"count": count, "error": errors[0]}
    generated = sum(run["tokens"] for run in runs)
    return {
        "count": count,
        "aggregate": generated / wall,
        "per_stream": [run["decode"] for run in runs],
        "accept": statistics.mean(run["accept"] for run in runs),
        "per_step": statistics.mean(run["per_step"] for run in runs),
        "prompt": runs[0]["prompt"],
        "tokens": runs[0]["tokens"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010/v1/chat/completions")
    parser.add_argument("--model", default="qwen3.8-flash-next-nvfp4")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--levels", default="1,2,4")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    options = parser.parse_args()
    levels = [int(value) for value in options.levels.split(",")]

    measured = {count: [] for count in levels}
    for round_index in range(options.rounds):
        for count in levels:
            result = level(
                options.url,
                options.model,
                options.prompt,
                options.tokens,
                count,
                options.timeout,
            )
            if "error" in result:
                print(f"round {round_index} c={count}: {result['error']}", flush=True)
                return 1
            measured[count].append(result)
            streams = " ".join(f"{rate:.1f}" for rate in result["per_stream"])
            print(
                f"round {round_index} c={count}: aggregate {result['aggregate']:6.2f} tok/s"
                f" per_stream [{streams}] accept {result['accept']:.2f}"
                f" tokens/step {result['per_step']:.2f}",
                flush=True,
            )

    print("\nmedian of the rounds, every level measured in every round:")
    for count in levels:
        rows = measured[count]
        aggregate = statistics.median(row["aggregate"] for row in rows)
        per_stream = statistics.median(
            statistics.mean(row["per_stream"]) for row in rows
        )
        accept = statistics.median(row["accept"] for row in rows)
        print(
            f"  c={count}: aggregate {aggregate:6.2f} tok/s"
            f" per_stream {per_stream:6.2f} accept {accept:.2f}"
        )
    first = measured[levels[0]][0]
    print(
        f"\n{first['prompt']} prompt tokens, {first['tokens']} generated,"
        f" {options.rounds} rounds of {options.levels}"
    )
    print(
        "Restart the server between configurations: the host PLE table pages in as requests touch\n"
        "n-grams, and a server that has been serving reads 30-40% low against a fresh one."
    )
    if all(row["accept"] == 0.0 for rows in measured.values() for row in rows):
        print(
            "accept 0.00 throughout: the server is not verifying drafts. A fused batch verifies\n"
            "only with SPARKFLASH_QWEN4_FUSE_SPEC=1, and a session verifies only when a drafter\n"
            "loaded (the startup line names it)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
