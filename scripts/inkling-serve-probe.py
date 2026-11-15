#!/usr/bin/env python3
"""Measure one served request: the card, the prefill, and the tokens after it.

    python3 scripts/inkling-serve-probe.py [--base http://127.0.0.1:8000] [--prompt TEXT]
                                          [--max-tokens N] [--chat]

The first streamed chunk is the time to first token - the prompt's prefill plus the control
round trip - and the rest of the stream is decode. The card is printed first so the context the
server admits is on the record beside the numbers.
"""
import argparse
import json
import time
import urllib.request


def request(base, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path, data=data, headers={"content-type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=9)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument(
        "--long",
        type=int,
        default=0,
        help="prepend this many words of filler, to make the prompt long",
    )
    args = parser.parse_args()
    if args.long:
        filler = (
            "The harbour lay under a low grey sky and the water carried the smell of tar and rope "
            "while the gulls worked the length of the quay. "
        )
        words = filler.split()
        pieces = [words[index % len(words)] for index in range(args.long)]
        args.prompt = " ".join(pieces) + " " + args.prompt

    with request(args.base, "/v1/models") as response:
        card = json.load(response)
    model = card["data"][0]
    print(
        "card id={} context_length={} max_output_tokens={}".format(
            model.get("id"), model.get("context_length"), model.get("max_output_tokens")
        )
    )
    print("card numerics={}".format(model.get("numerics")))
    print("card transport={}".format(model.get("transport")))

    if args.chat:
        path = "/v1/chat/completions"
        body = {
            "model": model["id"],
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        path = "/v1/completions"
        body = {
            "model": model["id"],
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    started = time.time()
    first = None
    pieces = []
    usage = None
    with request(args.base, path, body) as response:
        for line in response:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("choices"):
                choice = chunk["choices"][0]
                text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if text:
                    if first is None:
                        first = time.time() - started
                    pieces.append(text)
            if chunk.get("usage"):
                usage = chunk["usage"]
    total = time.time() - started
    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    print("prompt_tokens={}".format(prompt_tokens))
    print("first_chunk_s={}".format(None if first is None else round(first, 3)))
    if first and prompt_tokens:
        print("prefill_tok_s={}".format(round(prompt_tokens / first, 2)))
    print("total_s={}".format(round(total, 3)))
    if first is not None and usage:
        tokens = usage.get("completion_tokens") or 0
        if total > first and tokens > 1:
            print("decode_tok_s={}".format(round((tokens - 1) / (total - first), 2)))
    print("usage={}".format(usage))
    print("text={!r}".format("".join(pieces)))


if __name__ == "__main__":
    main()
