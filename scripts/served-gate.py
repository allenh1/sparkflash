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

"""served-gate.py: the runtime's served checks, for any model behind an OpenAI-style server.

Every arm on the shared runtime (docs/engine/03-runtime.md) owes the same served behaviour, so a
new model is gated by these checks as they stand, not by a copy written for it:

  lanes PORT [--sampled]       four prompts alone, then all four at once twice: each together
                               answer equals its alone answer (fused lanes compute each row as it
                               would alone; a seeded sample is a function of the row too).
  interleave PORT --long FILE  three decoders stream while a long prompt arrives: the decoders'
                               longest pause while it fills (prompt parts), and each decoder's
                               answer equals its answer alone.
  turns PORT [--strict]        conversations alone and four at once, thinking off and on: each
                               continued turn is served from its kept lane (cached past its first
                               prompt) and a rerun of it runs cold (cached 0). Whether the
                               continued answer equals the cold one is reported; --strict fails
                               on a difference (a continued turn forwards its suffix in other
                               chunks than a cold prompt, which may move greedy bits).
  all PORT --long FILE         lanes, lanes --sampled, interleave and turns.

--out FILE writes a check's digests as JSON; --baseline FILE fails the check unless they equal a
file an earlier run wrote (the bit-neutral comparison against the last deployed build). Prints
RESULT and CHECK lines and one VERDICT line a check; exits non-zero when any check fails.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

TOPICS = [
    "Explain how a hash table handles collisions, with a small Python example.",
    "Summarize the causes of the French Revolution in two paragraphs.",
    "List the prime numbers below 100, one per line.",
    "Describe how photosynthesis works to a ten year old.",
]
DECODERS = [
    "Write a long, detailed essay on the history of the printing press.",
    "Explain, step by step and at length, how a compiler turns source code into machine code.",
    "Describe in detail the water cycle and every stage of it, with examples.",
]
TURN_TOPICS = [
    "Describe a sunset over the sea in two sentences.",
    "Name three uses of a hash table, one line each.",
    "Explain in two sentences why the sky is blue.",
    "Give two tips for writing clear commit messages.",
]
FOLLOW = "Now say the same thing in one short sentence."
SAMPLED = {"temperature": 1.0, "top_p": 0.95, "seed": 12345}


class Server:
    def __init__(self, port):
        self.base = "http://127.0.0.1:%d" % port
        models = json.load(urllib.request.urlopen(self.base + "/v1/models", timeout=60))
        self.model = models["data"][0]["id"]

    def post(self, body, timeout=1800):
        body = dict(body, model=self.model)
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=timeout)

    def ask(self, messages, tokens, extra=None):
        """One answer, not streamed: its digest, usage and timing, or its error."""
        body = {"messages": messages, "max_tokens": tokens, "temperature": 0}
        body.update(extra or {})
        started = time.time()
        try:
            with self.post(body) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as error:
            return {"error": "%s %s" % (error.code, error.read()[:300]),
                    "wall": time.time() - started}
        except Exception as error:  # a dropped connection fails the check too
            return {"error": str(error), "wall": time.time() - started}
        message = answer["choices"][0]["message"]
        usage = answer.get("usage", {})
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        content = message.get("content") or ""
        return {
            "digest": digest(reasoning, content),
            "said": (reasoning, content),
            "tokens": usage.get("completion_tokens", 0),
            "prompt": usage.get("prompt_tokens", -1),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", -1),
            "cut": answer["choices"][0].get("finish_reason") == "length",
            "wall": time.time() - started,
            "head": (content or reasoning)[:60].replace("\n", " "),
        }

    def stream(self, content, tokens, record):
        """One streamed greedy answer into `record`, with each delta's arrival time."""
        record.update({"sent": time.time(), "times": [], "reasoning": "", "content": ""})
        body = {"messages": [{"role": "user", "content": content}], "max_tokens": tokens,
                "temperature": 0, "stream": True}
        try:
            with self.post(body, timeout=3600) as response:
                for line in response:
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if "error" in event:
                        record["error"] = str(event["error"])[:300]
                        break
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        if (delta.get("reasoning_content") or "") + (delta.get("content") or ""):
                            record["times"].append(time.time())
                        record["reasoning"] += delta.get("reasoning_content") or ""
                        record["content"] += delta.get("content") or ""
        except Exception as error:
            record["error"] = str(error)
        record["done"] = time.time()
        record["digest"] = digest(record["reasoning"], record["content"])


def digest(reasoning, content):
    return hashlib.sha256((reasoning + "\0" + content).encode()).hexdigest()[:12]


def show(label, answer):
    if "error" in answer:
        print("RESULT %s: ERROR %s" % (label, answer["error"]), flush=True)
        return
    print("RESULT %s: %d tokens %.1f s (%.1f tok/s), prompt %d, cached %d%s %r %s" % (
        label, answer["tokens"], answer["wall"], answer["tokens"] / max(answer["wall"], 1e-9),
        answer["prompt"], answer["cached"], " (cut)" if answer["cut"] else "", answer["head"],
        answer["digest"]), flush=True)


def against(name, digests, args):
    """Writes `digests` to --out and compares them with --baseline; true when they agree."""
    if args.out:
        with open(args.out, "w") as out:
            json.dump(digests, out)
    if not args.baseline:
        return True
    with open(args.baseline) as baseline:
        old = json.load(baseline)
    same = old == digests
    print("CHECK %s equals the baseline %s: %s%s" % (
        name, args.baseline, same, "" if same else " (%s vs %s)" % (old, digests)), flush=True)
    return same


def lanes(server, args, sampled):
    name = "lanes sampled" if sampled else "lanes"
    extra = SAMPLED if sampled else None
    ask = lambda i: server.ask([{"role": "user", "content": TOPICS[i]}], 192, extra)
    ok = True
    ask(0)  # warm up
    alone = []
    for i in range(len(TOPICS)):
        answer = ask(i)
        show("%s alone c%d" % (name, i), answer)
        alone.append(answer)
        ok &= "error" not in answer
    rate = sum(a.get("tokens", 0) for a in alone) / max(sum(a["wall"] for a in alone), 1e-9)
    print("RESULT %s alone: %.1f tok/s" % (name, rate), flush=True)
    for run in (1, 2):
        started = time.time()
        with cf.ThreadPoolExecutor(len(TOPICS)) as pool:
            together = list(pool.map(ask, range(len(TOPICS))))
        wall = time.time() - started
        for i, answer in enumerate(together):
            show("%s together%d c%d" % (name, run, i), answer)
            same = answer.get("digest") == alone[i].get("digest")
            print("CHECK %s together%d c%d equals alone: %s" % (name, run, i, same), flush=True)
            ok &= same
        total = sum(a.get("tokens", 0) for a in together)
        print("RESULT %s four at once run %d: %d tokens in %.1f s = %.1f tok/s" % (
            name, run, total, wall, total / wall), flush=True)
    ok &= against(name, [a.get("digest") for a in alone], args)
    return ok


def pause(times, lo, hi):
    """The longest gap between deltas over [lo, hi], the gaps spanning its ends included."""
    points = [t for t in times if t < lo][-1:] + [t for t in times if lo <= t <= hi]
    points += [t for t in times if t > hi][:1]
    return max((b - a for a, b in zip(points, points[1:])), default=0.0)


def interleave(server, args):
    with open(args.long, errors="replace") as text:
        prompt = text.read()[: args.chars] + "\n\nIn two sentences, what is the text above about?"
    ok = True
    server.stream("Say hello.", 4, {})  # warm up
    records = [{} for _ in DECODERS]
    threads = [threading.Thread(target=server.stream, args=(p, 320, r))
               for p, r in zip(DECODERS, records)]
    for thread in threads:
        thread.start()
    started = time.time()
    while time.time() - started < 600 and not all(
            len(r.get("times", [])) >= 8 or "error" in r for r in records):
        time.sleep(0.05)
    long = {}
    server.stream(prompt, 48, long)
    for thread in threads:
        thread.join()
    first = long["times"][0] if long.get("times") else long["done"]
    ttft = first - long["sent"]
    print("RESULT interleave long: %d chars, first token %.2f s, %r %s %s" % (
        len(prompt), ttft, long.get("content", "")[:60].replace("\n", " "), long.get("digest"),
        long.get("error", "")), flush=True)
    ok &= "error" not in long
    worst = 0.0
    for i, record in enumerate(records):
        gap = pause(record.get("times", []), long["sent"], first)
        worst = max(worst, gap)
        print("RESULT interleave d%d: %d deltas, longest pause while the long prompt filled "
              "%.2f s %s %s" % (i, len(record.get("times", [])), gap, record.get("digest"),
                                record.get("error", "")), flush=True)
        ok &= "error" not in record
    print("RESULT interleave pause: %.2f s (the long prompt's first token %.2f s)" % (worst, ttft),
          flush=True)
    alone = []
    for i, content in enumerate(DECODERS):
        record = {}
        server.stream(content, 320, record)
        same = record.get("digest") == records[i].get("digest")
        print("CHECK interleave d%d together equals alone: %s" % (i, same), flush=True)
        ok &= same
        alone.append(record.get("digest"))
    ok &= against("interleave", {"long": long.get("digest"),
                                 "together": [r.get("digest") for r in records],
                                 "alone": alone}, args)
    return ok


def turns(server, args):
    ok = True
    for thinking, mode in ((False, "plain"), (True, "think")):
        extra = {"chat_template_kwargs": {"thinking": thinking, "enable_thinking": thinking}}
        for shape, count in (("alone", 1), ("four", len(TURN_TOPICS))):
            label = "turns %s %s" % (mode, shape)

            def first(i):
                opening = {"role": "user",
                           "content": "Run %s. %s" % (uuid.uuid4().hex[:12], TURN_TOPICS[i])}
                return opening, server.ask([opening], 1024, extra)

            def second(pair):
                (opening, answer), suffix = pair
                reply = {"role": "assistant", "content": answer["said"][1]}
                if answer["said"][0]:
                    reply["reasoning_content"] = answer["said"][0]
                return server.ask([opening, reply, {"role": "user", "content": FOLLOW}], 1024,
                                  extra)

            with cf.ThreadPoolExecutor(count) as pool:
                firsts = list(pool.map(first, range(count)))
            if any("error" in answer for _, answer in firsts):
                print("CHECK %s: a first turn failed => FAIL" % label, flush=True)
                ok = False
                continue
            with cf.ThreadPoolExecutor(count) as pool:
                continued = list(pool.map(second, [(f, None) for f in firsts]))
            with cf.ThreadPoolExecutor(count) as pool:
                cold = list(pool.map(second, [(f, None) for f in firsts]))
            notes, passed = [], True
            for i, ((_, a1), a2, c2) in enumerate(zip(firsts, continued, cold)):
                for turn, answer in (("t1", a1), ("t2", a2), ("t2 cold", c2)):
                    show("%s c%d %s" % (label, i, turn), answer)
                if "error" in a2 or "error" in c2:
                    passed = False
                    notes.append("c%d ERROR" % i)
                    continue
                hit = a2["cached"] > a1["prompt"]
                equal = a2["said"] == c2["said"]
                passed &= hit and c2["cached"] == 0 and (equal or not args.strict)
                notes.append("c%d %s cached %d of %d (t1 %d%s), cold cached %d, %s" % (
                    i, "HIT" if hit else "MISS", a2["cached"], a2["prompt"], a1["prompt"],
                    ", cut" if a1["cut"] else "", c2["cached"],
                    "equal to cold" if equal else "differs from cold"))
                if not hit:
                    # The first reply's edges, where a re-rendering most often parts from the
                    # ids the server holds (stripped whitespace, an unfinished character).
                    reasoning, content = a1["said"]
                    print("MISS %s c%d: reasoning tail %r, content head %r, tail %r" % (
                        label, i, (reasoning or "")[-40:], content[:40], content[-40:]),
                        flush=True)
            print("CHECK %s: %s => %s" % (label, "; ".join(notes), "PASS" if passed else "FAIL"),
                  flush=True)
            ok &= passed
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("check", choices=["lanes", "interleave", "turns", "all"])
    parser.add_argument("port", type=int)
    parser.add_argument("--sampled", action="store_true", help="lanes: seeded sampling")
    parser.add_argument("--long", help="interleave: the long prompt's text file")
    parser.add_argument("--chars", type=int, default=60000, help="interleave: its length")
    parser.add_argument("--strict", action="store_true",
                        help="turns: a continued answer must equal the cold one")
    parser.add_argument("--out", help="write the check's digests here (one check only)")
    parser.add_argument("--baseline", help="the digests an earlier run wrote (one check only)")
    args = parser.parse_args()
    if args.check in ("interleave", "all") and not args.long:
        parser.error("interleave needs --long")
    if args.check == "all" and (args.out or args.baseline):
        parser.error("--out and --baseline name one check's digests; run the checks one by one")
    server = Server(args.port)
    print("RESULT model %s on port %d" % (server.model, args.port), flush=True)
    runs = {
        "lanes": [("lanes", lambda: lanes(server, args, args.sampled))],
        "interleave": [("interleave", lambda: interleave(server, args))],
        "turns": [("turns", lambda: turns(server, args))],
    }
    runs["all"] = [
        ("lanes", lambda: lanes(server, args, False)),
        ("lanes sampled", lambda: lanes(server, args, True)),
        ("interleave", lambda: interleave(server, args)),
        ("turns", lambda: turns(server, args)),
    ]
    every = True
    for name, run in runs[args.check]:
        ok = run()
        print("VERDICT %s %s" % (name, "PASS" if ok else "FAIL"), flush=True)
        every &= ok
    sys.exit(0 if every else 1)


if __name__ == "__main__":
    main()
