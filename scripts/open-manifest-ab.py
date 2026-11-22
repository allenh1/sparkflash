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

"""Record and compare the milestone 2 open manifest, traces and served speed.

The proof has four parts, and this is the one script that records and compares all four:
the open manifest, the launch trace, the tick trace and the served speed. It launches the
four-rank group through the site launcher, so the recorded paths are the ones the ranks
actually open.

Usage:
  scripts/open-manifest-ab.py record --build target/release/sparkflash --out target/m2/a-baseline
  scripts/open-manifest-ab.py compare --baseline target/m2/a-baseline --candidate target/m2/a-new
  scripts/open-manifest-ab.py ab --old target/m2/old --new target/m2/new --out target/m2/ab --reps 2

The launcher's own environment (SPARK_NODES, SPARKFLASH_CHECKPOINT, SPARKFLASH_TOKENIZER,
SPARKFLASH_CUBINS, ...) is inherited. Each rank writes its own files because the paths spell
{rank}; the chain expands it before opening. The tick trace is rank 0's file, as before.

The replay workload is fixed and greedy: one image request, one and eight text lanes, one bulk
prefill, then eight drafting lanes. Every request sets temperature 0. The workload is repeated
for each build; `ab` alternates the order so a slow drift cannot favour one build.
"""

import argparse
import base64
import concurrent.futures as cf
import functools
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

TICK_LINE = re.compile(r"^t=(\d+) clock=(\d+) (.*)$")
RANKS = 4
# A 1x1 transparent PNG: enough to exercise the image path without a second fixture file.
DOT_PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c6360000000020001e221bc330000000049454e44ae426082"
    )
).decode()
IMAGE_URL = "data:image/png;base64," + DOT_PNG


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tick_decisions(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            match = TICK_LINE.match(line)
            if not match:
                raise SystemExit(f"{path}:{number}: not a tick trace line: {line!r}")
            rows.append((int(match.group(1)), match.group(3)))
    return rows


def tick_difference(old_path, new_path):
    old = tick_decisions(old_path)
    new = tick_decisions(new_path)
    for index in range(max(len(old), len(new))):
        if index >= len(old):
            return f"the baseline tick trace ends first at decision {index}"
        if index >= len(new):
            return f"the candidate tick trace ends first at decision {index}"
        if old[index] != new[index]:
            return (
                f"decision {index} differs: baseline tick {old[index][0]} "
                f"{old[index][1]!r}, candidate tick {new[index][0]} {new[index][1]!r}"
            )
    return None


def sorted_lines(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def line_difference(old_path, new_path):
    old = sorted_lines(old_path)
    new = sorted_lines(new_path)
    for index in range(max(len(old), len(new))):
        if index >= len(old):
            return f"the baseline ends first at line {index + 1}: {new[index]!r}"
        if index >= len(new):
            return f"the candidate ends first at line {index + 1}: {old[index]!r}"
        if old[index] != new[index]:
            return (
                f"line {index + 1} differs:\n  baseline {old[index]!r}\n"
                f"  candidate {new[index]!r}"
            )
    return None


def ordered_difference(old_path, new_path):
    with open(old_path, "r", encoding="utf-8") as handle:
        old = [line.rstrip("\n") for line in handle]
    with open(new_path, "r", encoding="utf-8") as handle:
        new = [line.rstrip("\n") for line in handle]
    for index in range(max(len(old), len(new))):
        if index >= len(old):
            return f"the baseline ends first at launch {index + 1}: {new[index]!r}"
        if index >= len(new):
            return f"the candidate ends first at launch {index + 1}: {old[index]!r}"
        if old[index] != new[index]:
            return (
                f"launch {index + 1} differs:\n  baseline {old[index]!r}\n"
                f"  candidate {new[index]!r}"
            )
    return None


@functools.lru_cache(maxsize=4)
def served_model(url):
    """The id /v1/models reports; a packed arm may not accept a placeholder."""
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=30) as response:
        body = json.loads(response.read())
    return body["data"][0]["id"]


def request(url, messages, max_tokens, timeout):
    body = json.dumps(
        {
            "model": served_model(url),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    call = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(call, timeout=timeout) as response:
        answer = json.loads(response.read())
    wall = time.time() - started
    stats = answer.get("x_deepseek_spark_stats") or {}
    timing = stats.get("timing") or {}
    speculative = stats.get("dspark_speculative") or {}
    return {
        "wall_s": wall,
        "prompt_tokens": (answer.get("usage") or {}).get("prompt_tokens", 0),
        "completion_tokens": (answer.get("usage") or {}).get("completion_tokens", 0),
        "prefill_tok_per_s": float(timing.get("prefill_tok_per_s") or 0.0),
        "decode_tok_per_s": float(timing.get("decode_tok_per_s") or 0.0),
        "ttft_ms": float(timing.get("ttft_ms") or 0.0),
        "accepted": int(speculative.get("draft_tokens_accepted") or 0),
        "proposed": int(speculative.get("draft_tokens_proposed") or 0),
    }


def concurrent(url, count, messages, max_tokens, timeout):
    with cf.ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(lambda _: request(url, messages, max_tokens, timeout), range(count)))


def wait_ready(url, timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError, ValueError) as error:
            last = error
        time.sleep(2)
    raise SystemExit(f"open-manifest-ab: server not ready in {timeout}s: {last}")


def clean(out):
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(out):
        path = os.path.join(out, name)
        if os.path.isfile(path):
            os.unlink(path)


def launch(args, out):
    served_model.cache_clear()
    env = dict(os.environ)
    env["SPARKFLASH_BIN"] = args.build
    env["SPARKFLASH_RUN_ID"] = "open-manifest-" + str(os.getpid())
    env["SPARKFLASH_TICK_TRACE"] = os.path.join(out, "tick.trace")
    env["SPARKFLASH_LAUNCH_TRACE"] = os.path.join(out, "launch-r{rank}.trace")
    env["SPARKFLASH_OPEN_MANIFEST"] = os.path.join(out, "manifest-r{rank}.jsonl")
    env["SPARKFLASH_LOG_ROOT"] = os.path.join(out, "logs")
    command = ["bash", args.site_script, "launch"]
    print("LAUNCH", " ".join(command))
    print("  SPARKFLASH_BIN=" + args.build)
    print("  SPARKFLASH_TICK_TRACE=" + env["SPARKFLASH_TICK_TRACE"])
    print("  SPARKFLASH_LAUNCH_TRACE=" + env["SPARKFLASH_LAUNCH_TRACE"])
    print("  SPARKFLASH_OPEN_MANIFEST=" + env["SPARKFLASH_OPEN_MANIFEST"])
    log_path = os.path.join(out, "launch.log")
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.launch_timeout
        )
    if result.returncode != 0:
        raise SystemExit(f"open-manifest-ab: launcher failed; see {log_path}")
    wait_ready(args.url, args.timeout)


def stop(args):
    subprocess.run(
        ["bash", args.site_script, "stop"],
        env=dict(os.environ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=args.stop_timeout,
        check=False,
    )


def run_workload(args):
    url = args.url
    speeds = {"decode_1": [], "decode_8": [], "prefill": [], "draft_accept": []}
    image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        }
    ]
    print("WORKLOAD image one request")
    image_result = request(url, image, args.max_tokens, args.request_timeout)
    print("  image prompt_tokens=%s" % image_result["prompt_tokens"])

    text = [{"role": "user", "content": "Explain a GPU prefill kernel in three sentences."}]
    print("WORKLOAD lane 1")
    one = concurrent(url, 1, text, args.max_tokens, args.request_timeout)
    speeds["decode_1"].append(one[0]["decode_tok_per_s"])

    print("WORKLOAD lanes 8")
    eight = concurrent(url, 8, text, args.max_tokens, args.request_timeout)
    speeds["decode_8"].append(statistics.mean(row["decode_tok_per_s"] for row in eight))

    print("WORKLOAD bulk prefill %d tokens" % args.bulk_tokens)
    bulk = [{"role": "user", "content": "alpha " * args.bulk_tokens}]
    prefill = request(url, bulk, 8, args.request_timeout)
    speeds["prefill"].append(prefill["prefill_tok_per_s"])
    print("  prompt_tokens=%s prefill_tok_per_s=%.1f" % (prefill["prompt_tokens"], prefill["prefill_tok_per_s"]))

    print("WORKLOAD drafting lanes 8")
    draft = concurrent(
        url,
        8,
        [{"role": "user", "content": "Write a precise paragraph about speculative decoding."}],
        args.draft_tokens,
        args.request_timeout,
    )
    accepted = sum(row["accepted"] for row in draft)
    proposed = sum(row["proposed"] for row in draft)
    speeds["draft_accept"].append((accepted / proposed) if proposed else 0.0)
    print("  accepted=%d proposed=%d" % (accepted, proposed))

    for key in ("decode_1", "decode_8", "prefill"):
        if not speeds[key] or speeds[key][0] <= 0:
            raise SystemExit(f"open-manifest-ab: {key} had no positive served rate")
    return {key: statistics.median(value) for key, value in speeds.items()}


def record_build(args, out):
    out = os.path.abspath(out)
    args.build = os.path.abspath(args.build)
    clean(out)
    try:
        launch(args, out)
        speeds = run_workload(args)
    finally:
        stop(args)
    with open(os.path.join(out, "speeds.json"), "w", encoding="utf-8") as handle:
        json.dump(speeds, handle, indent=2, sort_keys=True)
        handle.write("\n")
    hashes = {}
    for name in sorted(os.listdir(out)):
        if name.startswith(("manifest-r", "launch-r", "tick.trace")):
            hashes[name] = sha256(os.path.join(out, name))
    with open(os.path.join(out, "hashes.json"), "w", encoding="utf-8") as handle:
        json.dump(hashes, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("RECORDED", out)
    for name, digest in hashes.items():
        print("  %s  %s" % (digest, name))
    return hashes


def compare(baseline, candidate, tolerance):
    ok = True
    for rank in range(RANKS):
        name = f"manifest-r{rank}.jsonl"
        old = os.path.join(baseline, name)
        new = os.path.join(candidate, name)
        if not (os.path.isfile(old) and os.path.isfile(new)):
            print(f"RESULT manifest r{rank}: MISSING")
            ok = False
            continue
        difference = line_difference(old, new)
        if difference:
            print(f"RESULT manifest r{rank}: DIFFERENT: {difference}")
            ok = False
        else:
            print(f"RESULT manifest r{rank}: identical ({len(sorted_lines(new))} records)")

        name = f"launch-r{rank}.trace"
        old = os.path.join(baseline, name)
        new = os.path.join(candidate, name)
        if not (os.path.isfile(old) and os.path.isfile(new)):
            print(f"RESULT launch r{rank}: MISSING")
            ok = False
            continue
        difference = ordered_difference(old, new)
        if difference:
            print(f"RESULT launch r{rank}: DIFFERENT: {difference}")
            ok = False
        else:
            print(f"RESULT launch r{rank}: identical ({len(sorted_lines(new))} launches)")

    old_tick = os.path.join(baseline, "tick.trace")
    new_tick = os.path.join(candidate, "tick.trace")
    if os.path.isfile(old_tick) and os.path.isfile(new_tick):
        difference = tick_difference(old_tick, new_tick)
        if difference:
            print("RESULT tick: DIFFERENT: " + difference)
            ok = False
        else:
            print(f"RESULT tick: identical ({len(tick_decisions(new_tick))} decisions)")
    else:
        print("RESULT tick: MISSING")
        ok = False

    try:
        with open(os.path.join(baseline, "speeds.json"), encoding="utf-8") as handle:
            old_speed = json.load(handle)
        with open(os.path.join(candidate, "speeds.json"), encoding="utf-8") as handle:
            new_speed = json.load(handle)
    except (OSError, ValueError) as error:
        print("RESULT speed: MISSING: %s" % error)
        return False
    for key in ("decode_1", "decode_8", "prefill"):
        floor = old_speed[key] * (1.0 - tolerance)
        verdict = "ok" if new_speed[key] >= floor else "SLOWER"
        print(
            "RESULT speed %s: baseline %.2f candidate %.2f floor %.2f (%s)"
            % (key, old_speed[key], new_speed[key], floor, verdict)
        )
        if new_speed[key] < floor:
            ok = False
    print("RESULT draft acceptance: baseline %.3f candidate %.3f" % (
        old_speed["draft_accept"], new_speed["draft_accept"]
    ))
    return ok


def compare_dirs(baseline, candidate, tolerance):
    print("COMPARE", baseline, "->", candidate)
    return compare(baseline, candidate, tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--site-script", default="scripts/serve-four-node.sh")
        p.add_argument("--url", default="http://127.0.0.1:8000")
        p.add_argument("--max-tokens", type=int, default=32)
        p.add_argument("--draft-tokens", type=int, default=128)
        p.add_argument("--bulk-tokens", type=int, default=16000)
        p.add_argument("--timeout", type=int, default=3600)
        p.add_argument("--request-timeout", type=int, default=3600)
        p.add_argument("--launch-timeout", type=int, default=120)
        p.add_argument("--stop-timeout", type=int, default=180)

    record = sub.add_parser("record", help="launch one build and record its files")
    record.add_argument("--build", required=True)
    record.add_argument("--out", required=True)
    common(record)

    compare_parser = sub.add_parser("compare", help="compare two recorded directories")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--speed-tolerance", type=float, default=0.05)

    ab = sub.add_parser("ab", help="record old and new alternately, then compare")
    ab.add_argument("--old", required=True)
    ab.add_argument("--new", required=True)
    ab.add_argument("--out", required=True)
    ab.add_argument("--reps", type=int, default=2)
    ab.add_argument("--speed-tolerance", type=float, default=0.05)
    common(ab)

    args = parser.parse_args()
    if args.command == "record":
        record_build(args, args.out)
        return 0
    if args.command == "compare":
        return 0 if compare_dirs(args.baseline, args.candidate, args.speed_tolerance) else 1

    os.makedirs(args.out, exist_ok=True)
    results = []
    for rep in range(args.reps):
        order = [("old", args.old), ("new", args.new)]
        if rep % 2:
            order.reverse()
        for name, build in order:
            args.build = build
            out = os.path.join(args.out, f"{name}-rep{rep}")
            record_build(args, out)
            results.append((rep, name, out))
    ok = True
    for rep in range(args.reps):
        old = os.path.join(args.out, f"old-rep{rep}")
        new = os.path.join(args.out, f"new-rep{rep}")
        ok = compare_dirs(old, new, args.speed_tolerance) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
