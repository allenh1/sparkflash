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

"""The shared step's shape: cold N-stream decode, so a cost-model change can be A/B'd.

Each sample sends N prompts at once, each a random slice of /tmp/book_8k.txt behind a nonce
(so no prefix cache answers it), greedy, thinking off, and reports the aggregate and per-stream
rates the engine itself measured. The arms differ only in SPARKFLASH_SPEC_BATCH_COST.

  gate_bench.py URL LABEL STREAMS [SAMPLES] [TOKENS]
"""
import hashlib, json, random, string, sys, threading, time, urllib.request

base = sys.argv[1].rstrip("/")
label = sys.argv[2]
streams = int(sys.argv[3])
samples = int(sys.argv[4]) if len(sys.argv) > 4 else 2
tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 192
# The corpus is not committed (it is public-domain prose, fetched once). The path may be named
# with GATE_CORPUS; the historical /tmp path stays the default so an old invocation still runs.
import os
corpus_path = os.environ.get("GATE_CORPUS") or "/tmp/book_8k.txt"
corpus = open(corpus_path, encoding="utf-8", errors="ignore").read()

def post(prompt, out, errors):
    body = json.dumps({"model": "deepseek-ai/DeepSeek-V4.1-Flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": tokens, "temperature": 0.0,
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            answer = json.loads(response.read())
    except Exception as error:
        errors.append(str(error))
        return
    done = time.time()
    stats = (answer.get("x_deepseek_spark_stats") or {}).get("timing") or {}
    usage = answer.get("usage") or {}
    text = (answer.get("choices") or [{}])[0].get("message", {}).get("content", "")
    out.append({"out": usage.get("completion_tokens", 0), "start": started, "done": done,
                "decode_tok_s": stats.get("decode_tok_per_s", 0.0),
                "sha": hashlib.sha256(text.encode()).hexdigest()[:12]})

# The prompt a sample gets is its seed, and the seed used to be 1000 + sample with no way to move
# it: every invocation with the default sample count re-sent prompt 1000, so three separate runs
# looked like three samples and were one prompt measured three times. Prompts differ by 4-5 % at c1,
# which is the size of the effects this bench exists to resolve, so a repeated seed reads as a result.
# GATE_SEED moves the base; a paired A/B uses the same base on both arms and compares prompt by
# prompt.
seed = int(os.environ.get("GATE_SEED") or 1000)

for sample in range(samples):
    rng = random.Random(seed + sample)
    prompts = []
    for _ in range(streams):
        nonce = "[" + "".join(rng.choices(string.ascii_lowercase, k=20)) + "] "
        at = rng.randrange(0, max(1, len(corpus) - 2400))
        prompts.append(nonce + corpus[at:at + 2000])
    out, errors = [], []
    threads = [threading.Thread(target=post, args=(prompt, out, errors)) for prompt in prompts]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started
    total = sum(row["out"] for row in out)
    first = min((row["start"] for row in out), default=started)
    last = max((row["done"] for row in out), default=started)
    span = max(last - first, 1e-9)
    print(json.dumps({"label": label, "streams": streams, "sample": sample + 1,
                      "errors": errors, "completion_tokens": total,
                      "aggregate_tok_s": round(total / span, 1),
                      "per_stream_tok_s": round(sum(r["decode_tok_s"] for r in out) / max(len(out), 1), 1),
                      "wall_s": round(wall, 2)}))
