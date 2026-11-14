#!/usr/bin/env python3
"""Ask the served model about images it can be checked against.

    python3 scripts/inkling-image-probe.py [--base http://127.0.0.1:8010] [--widths 1,2,4]
                                           [--max-tokens N] [--effort LEVEL] [--geometry]

Draws synthetic PNGs in pure Python (no image library) and asks short greedy questions whose
answers are known: colours, a shape, stripes, two images in one message. It then sends a second
turn over the first image, sends the first request again, and at each width above 1 runs that
many requests together and checks each returns the text it returned alone. `--geometry` asks
only the layout questions (quadrants, a bottom strip, a right strip, four bands), which show
whether the last cell row and column reach the model. Each answer is printed with the opening of
its reasoning.
"""
import argparse
import base64
import json
import struct
import threading
import time
import urllib.request
import zlib


def png(width, height, pixel):
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(pixel(x, y))
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))


def data_url(image):
    return "data:image/png;base64," + base64.b64encode(image).decode()


RED_BLUE = png(400, 300, lambda x, y: (220, 20, 20) if x < 200 else (20, 40, 220))
GREEN = png(256, 256, lambda x, y: (30, 180, 40))
CIRCLE = png(320, 320, lambda x, y: (0, 0, 0) if (x - 160) ** 2 + (y - 160) ** 2 < 90 ** 2 else (255, 255, 255))
STRIPES = png(360, 240, lambda x, y: (255, 210, 0) if (y // 40) % 2 == 0 else (0, 0, 0))


def part(image):
    return {"type": "image_url", "image_url": {"url": data_url(image)}}


def user(text, *images):
    return {"role": "user", "content": [*map(part, images), {"type": "text", "text": text}]}


def quadrant(x, y):
    top, left = y < 160, x < 160
    return {(True, True): (220, 20, 20), (True, False): (30, 180, 40),
            (False, True): (20, 40, 220), (False, False): (255, 210, 0)}[(top, left)]


QUADRANTS = png(320, 320, quadrant)
BOTTOM = png(320, 240, lambda x, y: (220, 20, 20) if y >= 200 else (255, 255, 255))
RIGHT = png(240, 320, lambda x, y: (20, 40, 220) if x >= 200 else (255, 255, 255))
FOUR = png(360, 240, lambda x, y: (255, 210, 0) if (y // 60) % 2 == 0 else (0, 0, 0))

GEOMETRY = [
    ("quadrants", [user("This image is split into four colored quadrants. Name the color of the top-left, top-right, bottom-left and bottom-right quadrants.", QUADRANTS)]),
    ("bottom strip", [user("Where in this image is the red, and how much of the image does it cover? One short sentence.", BOTTOM)]),
    ("right strip", [user("Where in this image is the blue, and how much of the image does it cover? One short sentence.", RIGHT)]),
    ("four bands", [user("How many horizontal bands does this image have, and what color is the top band and the bottom band?", FOUR)]),
]

CASES = [
    ("halves", [user("What color is the left half of this image, and what color is the right half? Answer in one short sentence.", RED_BLUE)]),
    ("green", [user("What single color fills this image? One word.", GREEN)]),
    ("circle", [user("Describe the shape in this image in one short sentence.", CIRCLE)]),
    ("stripes", [user("Describe this image's pattern and colors in one short sentence.", STRIPES)]),
    ("two images", [user("The first image and the second image: name the main color or colors of each, briefly.", GREEN, RED_BLUE)]),
]


def chat(base, model, messages, max_tokens, effort=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    if effort is not None:
        body["reasoning_effort"] = effort
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    started = time.time()
    first = None
    text = []
    thought = []
    usage = {}
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    reason = delta.get("reasoning_content") or ""
                    if (piece or reason) and first is None:
                        first = time.time()
                    text.append(piece)
                    thought.append(reason)
    except urllib.error.HTTPError as error:
        return {"error": f"{error.code} {error.read().decode()[:400]}"}
    done = time.time()
    return {"text": "".join(text), "thought": "".join(thought),
            "ttft": (first or done) - started, "total": done - started,
            "prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens")}


def show(name, result):
    if "error" in result:
        print(f"{name}: ERROR {result['error']}")
        return
    rate = (result["completion"] or 0) / max(result["total"] - result["ttft"], 1e-9)
    print(f"{name}: prompt {result['prompt']} tok, ttft {result['ttft']:.2f} s, "
          f"{result['completion']} tok at {rate:.1f} tok/s: {result['text'].strip()!r}")
    if result.get("thought"):
        thought = " ".join(result["thought"].split())
        print(f"    (reasoning {len(thought)} chars: {thought[:220]!r})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--effort", default=None)
    parser.add_argument("--geometry", action="store_true", help="only the layout cases")
    parser.add_argument("--widths", default="1")
    args = parser.parse_args()
    with urllib.request.urlopen(args.base + "/v1/models", timeout=30) as response:
        model = json.load(response)["data"][0]["id"]
    if args.geometry:
        for name, messages in GEOMETRY:
            show(name, chat(args.base, model, messages, args.max_tokens, args.effort))
        return
    alone = {}
    for name, messages in CASES:
        alone[name] = chat(args.base, model, messages, args.max_tokens, args.effort)
        show(name, alone[name])
    # A second turn over the same image: the held prefix (image included) is reused.
    first = CASES[0][1]
    answer = alone["halves"].get("text", "")
    follow = first + [{"role": "assistant", "content": answer},
                      {"role": "user", "content": "Which of those two colors is warmer? One word."}]
    show("follow-up", chat(args.base, model, follow, args.max_tokens, args.effort))
    show("halves again", chat(args.base, model, first, args.max_tokens, args.effort))
    for width in [int(w) for w in args.widths.split(",") if int(w) > 1]:
        picked = CASES[:width]
        results = [None] * len(picked)
        def run(index, messages):
            results[index] = chat(args.base, model, messages, args.max_tokens, args.effort)
        threads = [threading.Thread(target=run, args=(i, m)) for i, (_, m) in enumerate(picked)]
        started = time.time()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.time() - started
        same = all(r.get("text") == alone[n].get("text") for r, (n, _) in zip(results, picked))
        tokens = sum(r.get("completion") or 0 for r in results)
        print(f"width {width}: {wall:.2f} s wall, {tokens / wall:.1f} tok/s aggregate, "
              f"texts match alone: {same}")
        for (name, _), result in zip(picked, results):
            if result.get("text") != alone[name].get("text"):
                show(f"  {name} at width {width}", result)


if __name__ == "__main__":
    main()
