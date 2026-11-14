#!/usr/bin/env python3
"""Concurrency probe: the same greedy requests one at a time, then N at once.

    python3 scripts/inkling-lanes-probe.py [--base http://127.0.0.1:8010] [--max-tokens N]
                                           [--widths 1,2,4] [--requests N]

Prints, for each width, the requests' time to first token, each request's decode rate and the
aggregate rate, and checks that every request's text at width N is its text at width 1 (greedy,
and a lane steps as it would alone).
"""
import argparse
import json
import threading
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Write a short poem about the sea:",
    "Explain in two sentences why the sky is blue.",
    "List three prime numbers greater than one hundred:",
    "The quick brown fox",
    "In computer science, a hash table is",
    "Translate to French: The cat sleeps on the warm windowsill.",
    "Once upon a time, in a village by the mountains,",
]


def stream(base, model, prompt, max_tokens):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.time()
    first = None
    text = []
    tokens = 0
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if event.get("usage"):
                tokens = event["usage"].get("completion_tokens", tokens)
            for choice in event.get("choices", []):
                piece = choice.get("text") or ""
                if piece and first is None:
                    first = time.time()
                text.append(piece)
    ended = time.time()
    return {
        "ttft": (first or ended) - started,
        "total": ended - started,
        "tokens": tokens,
        "text": "".join(text),
        "start": started,
        "end": ended,
    }


def run(base, model, prompts, max_tokens, width):
    results = [None] * len(prompts)
    gate = threading.Barrier(width) if width > 1 else None

    def worker(indices):
        for index in indices:
            if gate is not None and index < width:
                gate.wait()
            results[index] = stream(base, model, prompts[index], max_tokens)

    lanes = [list(range(lane, len(prompts), width)) for lane in range(width)]
    threads = [threading.Thread(target=worker, args=(indices,)) for indices in lanes]
    began = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.time() - began
    return results, wall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--widths", default="1,2,4")
    parser.add_argument("--requests", type=int, default=0)
    args = parser.parse_args()
    with urllib.request.urlopen(args.base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]
    widths = [int(width) for width in args.widths.split(",")]
    count = args.requests or max(widths)
    prompts = [PROMPTS[index % len(PROMPTS)] for index in range(count)]
    reference = None
    for width in widths:
        results, wall = run(args.base, model, prompts, args.max_tokens, width)
        generated = sum(result["tokens"] for result in results)
        rates = [
            (result["tokens"] - 1) / max(result["total"] - result["ttft"], 1e-9)
            for result in results
        ]
        print(
            "width {} requests {} wall {:.2f}s aggregate {:.1f} tok/s | ttft {} | per-request "
            "decode {} tok/s".format(
                width,
                len(results),
                wall,
                generated / wall,
                [round(result["ttft"], 2) for result in results],
                [round(rate, 1) for rate in rates],
            ),
            flush=True,
        )
        texts = [result["text"] for result in results]
        if reference is None:
            reference = texts
            for index, text in enumerate(texts):
                print("  [{}] {!r}".format(index, text[:120]), flush=True)
        else:
            same = [left == right for left, right in zip(reference, texts)]
            print("  same text as width {}: {}".format(widths[0], same), flush=True)
            for index, ok in enumerate(same):
                if not ok:
                    print("  [{}] was {!r}".format(index, reference[index][:120]))
                    print("  [{}] now {!r}".format(index, texts[index][:120]))


if __name__ == "__main__":
    main()
