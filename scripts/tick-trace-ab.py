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

"""tick-trace-ab.py: the phase-6 step-4 A/B, on the tick trace the server writes.

The gate is two claims. First, the decisions the new engine makes on a fixed workload are the
ones the old build made: same lanes, same prompt order, same holds, same chunk boundaries, same
bulk/exact choice, same spec/shared/plain decode plan. Second, alternated served launches at 1, 8
and 16 streams and 2K and 16K prompts keep that true several times each.

The server writes the trace when `SPARKFLASH_TICK_TRACE` names a file (rank 0; the group makes
the same decisions). Each launch here gets its own file, so the comparison is between whole
served runs, not between two processes sharing one.

Usage:
  scripts/tick-trace-ab.py --old PATH --new PATH --checkpoint DIR [--model ID]
  scripts/tick-trace-ab.py --old PATH --new PATH --checkpoint DIR --dry-run
  scripts/tick-trace-ab.py --diff OLD.trace NEW.trace

  --old / --new       the two `sparkflash` binaries. Both must be built from a tree that emits
                      the tick trace: the new engine, and the frozen pre-move build with this
                      emission commit cherry-picked onto it (trace only; no decision changes).
  --checkpoint DIR    the DeepSeek-V4.1 checkpoint the server serves (`sparkflash serve
                      --checkpoint`). One is needed; no checkpoint, no served path.
  --model ID          passed through to `serve --model` for a packed checkpoint.
  --streams           comma list, default 1,8,16.
  --prompt-tokens     comma list, default 2048,16384.
  --reps              launches of each build for each pair, default 3.
  --max-tokens        completion length a request asks for, default 16.
  --stagger SECONDS   start request N N*stagger after the first. 0 fires them together, which
                      makes a tick's admission set a race between launches; spacing them is how
                      the comparison sees the scheduler rather than arrival timing.
  --out DIR           the traces and server logs, default target/tick-trace-ab.
  --site-script FILE  launch the four-rank group through the site's own launcher
                      (scripts/serve-four-node.sh) instead of one serve process. The binary is
                      named by SPARKFLASH_BIN and the trace by SPARKFLASH_TICK_TRACE, which the
                      launcher forwards; this is the served topology the trace measures, because
                      a single process takes the reference path and never constructs Batch.
  --bind HOST:PORT    default 127.0.0.1:8000; one launch at a time.
  --timeout SECONDS   readiness/request timeout, default 3600.
  --dry-run           print the exact commands and workload, run nothing.
  --diff A B          compare two trace files and print the first differing decision.

A difference is reported with the tick index and both lines; the script exits non-zero. The
injected clock is deliberately left out of the comparison: two live builds measure different wall
times, and the decision sequence is what must match. The replay gate
(sparkflash-runtime/tests/tick_trace.rs, plus the forward crate's CPU mirror) is the
host-independent check of the same decisions.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

LINE = re.compile(r"^t=(\d+) clock=(\d+) (.*)$")


def parse_trace(path):
    """[(tick, decision, line, line_number)] with the injected clock dropped."""
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            match = LINE.match(line)
            if not match:
                raise SystemExit("%s:%d: not a tick trace line: %r" % (path, number, line))
            rows.append((int(match.group(1)), match.group(3), line, number))
    return rows


def first_difference(old_path, new_path, ignore_tick=False):
    """(tick, old line, new line, why) or None when the decision sequences agree.

    With `ignore_tick`, the same decisions at different tick numbers compare equal: that is
    the verdict "the decisions agree, the tick boundaries do not", which is what a real-time
    scheduler's admission jitter looks like and what a strict comparison cannot separate from a
    changed decision.
    """
    old = parse_trace(old_path)
    new = parse_trace(new_path)
    if not old and not new:
        return ("-", "-", "-", "both traces are empty")
    for index in range(max(len(old), len(new))):
        if index >= len(old):
            tick, decision, line, number = new[index]
            return (tick, "-(ends)", "%s:%d" % (line, number), "the old trace ends first")
        if index >= len(new):
            tick, decision, line, number = old[index]
            return (tick, "%s:%d" % (line, number), "-(ends)", "the new trace ends first")
        old_tick, old_decision, old_line, old_number = old[index]
        new_tick, new_decision, new_line, new_number = new[index]
        if old_decision != new_decision:
            return (
                old_tick if old_tick == new_tick else "%d/%d" % (old_tick, new_tick),
                "%s:%d" % (old_line, old_number),
                "%s:%d" % (new_line, new_number),
                "decision %d differs" % index,
            )
        if old_tick != new_tick and not ignore_tick:
            # The same decisions at different tick numbers are a different schedule.
            return (
                "%d/%d" % (old_tick, new_tick),
                "%s:%d" % (old_line, old_number),
                "%s:%d" % (new_line, new_number),
                "decision %d ran at a different tick" % index,
            )
    return None


def report_diff(label, old_path, new_path, ignore_tick=False):
    difference = first_difference(old_path, new_path, ignore_tick)
    if difference is None:
        suffix = ", tick boundaries ignored" if ignore_tick else ""
        print(
            "RESULT %s: identical (%d decisions%s)"
            % (label, len(parse_trace(old_path)), suffix),
        )
        return True
    tick, old_line, new_line, why = difference
    print("RESULT %s: DIFFERENT at tick %s (%s)" % (label, tick, why))
    print("  old %s" % old_line)
    print("  new %s" % new_line)
    return False


def report_unordered(label, old_path, new_path):
    """Compares the two traces as multisets of decisions.

    A real-time scheduler can place an admission one tick later and still run every other
    decision unchanged; the two traces then hold the same decisions in a different order. This
    verdict says exactly that, and it is weaker than the ordered one by design.
    """
    old = parse_trace(old_path)
    new = parse_trace(new_path)
    old_decisions = [decision for _, decision, _, _ in old]
    new_decisions = [decision for _, decision, _, _ in new]
    if sorted(old_decisions) == sorted(new_decisions):
        print(
            "RESULT %s: identical as a multiset (%d decisions)"
            % (label, len(old_decisions)),
        )
        return True
    missing = sorted(set(old_decisions) - set(new_decisions))
    extra = sorted(set(new_decisions) - set(old_decisions))
    print("RESULT %s: DIFFERENT decisions" % label)
    print("  only in old: %s" % (missing[:4] or "-"))
    print("  only in new: %s" % (extra[:4] or "-"))
    return False


def prompt_text(tokens):
    """A deterministic prompt of roughly `tokens` ids; the server reports the real count."""
    return " ".join(["alpha"] * tokens)


def ask(url, text, max_tokens, timeout):
    body = {
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        answer = json.load(response)
    return answer.get("usage", {}).get("prompt_tokens", -1)


def model_id(url, timeout):
    with urllib.request.urlopen(url + "/v1/models", timeout=timeout) as response:
        return json.load(response)["data"][0]["id"]


def wait_ready(url, process, timeout):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if process.poll() is not None:
            return False
        try:
            model_id(url, 5)
            return True
        except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
            last = error
            time.sleep(0.5)
    print("tick-trace-ab: not ready in %ss: %s" % (timeout, last), file=sys.stderr)
    return False


def workload(url, streams, text, max_tokens, timeout, stagger=0.0):
    """All streams start together unless staggered; prompt_tokens checks the size.

    With `stagger` seconds, request N starts N*stagger after request 0. A tick's admission set
    depends on when the requests arrive, so a barrier makes the admission lines a race; spacing
    them is what lets the A/B compare the scheduler rather than the network.
    """
    barrier = threading.Barrier(streams) if not stagger else None

    def one(index):
        if barrier is not None:
            barrier.wait()
        elif stagger:
            time.sleep(index * stagger)
        try:
            return ask(url, text, max_tokens, timeout)
        except Exception as error:  # a failed stream is reported, not silently compared
            return "error: %s" % error

    with cf.ThreadPoolExecutor(max_workers=streams) as pool:
        return list(pool.map(one, range(streams)))


def launch(binary, args, trace_path, log_path, timeout):
    environment = dict(os.environ)
    environment["SPARKFLASH_TICK_TRACE"] = trace_path
    command = [binary, "serve"] + args
    print("LAUNCH %s" % " ".join(command))
    print("  SPARKFLASH_TICK_TRACE=%s" % trace_path)
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=environment
    )
    try:
        ready = wait_ready("http://" + args[args.index("--bind") + 1], process, timeout)
        if not ready:
            tail(log_path)
            return None
        return process, log
    except Exception:
        process.kill()
        log.close()
        raise


def stop(process, log, grace=120):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    log.close()


def tail(path, lines=40):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            print("".join(handle.readlines()[-lines:]), file=sys.stderr)
    except OSError:
        pass


def wait_ready_url(url, timeout):
    """Waits for /v1/models to answer, without owning the server's process."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            model_id(url, 5)
            return True
        except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
            last = error
            time.sleep(1)
    print("tick-trace-ab: not ready in %ss: %s" % (timeout, last), file=sys.stderr)
    return False


def group_launch(script, binary, trace_path, log_path, url, timeout):
    """Launches the site's four-rank group with `binary`, its trace on rank 0.

    The site launcher starts the workers detached, checks that rank 0 survived two seconds and
    exits; the ranks outlive it. So the launcher's exit is not the server's, and readiness is
    read from the endpoint rather than from the launcher's process.
    """
    environment = dict(os.environ)
    environment["SPARKFLASH_BIN"] = binary
    environment["SPARKFLASH_TICK_TRACE"] = trace_path
    command = ["bash", script, "launch"]
    print("LAUNCH %s (SPARKFLASH_BIN=%s)" % (" ".join(command), binary))
    print("  SPARKFLASH_TICK_TRACE=%s" % trace_path)
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
    )
    try:
        code = process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        code = None
    if code not in (0, None):
        tail(log_path)
        group_stop(script, log, url)
        return None
    if not wait_ready_url(url, timeout):
        tail(log_path)
        group_stop(script, log, url)
        return None
    return process, log


def group_stop(script, log, url, grace=120):
    """Stops the site's group through its own stop action, then waits for :8000 to close."""
    subprocess.run(
        ["bash", script, "stop"],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(os.environ),
        timeout=grace,
        check=False,
    )
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            model_id(url, 2)
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            break
        time.sleep(1)
    log.close()


def stop_server(args, process, log, url):
    if args.site_script:
        group_stop(args.site_script, log, url)
    else:
        stop(process, log)

def server_args(args):
    out = ["--bind", args.bind]
    if args.checkpoint:
        out += ["--checkpoint", args.checkpoint]
    if args.model:
        out += ["--model", args.model]
    if args.synthetic:
        out.append("--synthetic")
    return out


def run(args):
    os.makedirs(args.out, exist_ok=True)
    streams = [int(value) for value in args.streams.split(",")]
    prompts = [int(value) for value in args.prompt_tokens.split(",")]
    bind_host, bind_port = args.bind.rsplit(":", 1)
    url = "http://%s:%s" % (bind_host, bind_port)
    ok = True
    pairs = 0
    for tokens in prompts:
        text = prompt_text(tokens)
        for count in streams:
            for rep in range(args.reps):
                order = [("old", args.old), ("new", args.new)]
                if rep % 2:
                    order.reverse()
                label = "s%d-p%d-r%d" % (count, tokens, rep)
                traces = {}
                for name, binary in order:
                    trace_path = os.path.join(args.out, "%s-%s.trace" % (label, name))
                    log_path = os.path.join(args.out, "%s-%s.log" % (label, name))
                    if args.site_script:
                        started = group_launch(
                            args.site_script, binary, trace_path, log_path, url, args.timeout
                        )
                    else:
                        started = launch(
                            binary, server_args(args), trace_path, log_path, args.timeout
                        )
                    if started is None:
                        ok = False
                        traces[name] = None
                        continue
                    process, log = started
                    try:
                        seen = workload(
                            url, count, text, args.max_tokens, args.timeout, args.stagger
                        )
                        print(
                            "WORKLOAD %s %s prompt_tokens=%s"
                            % (label, name, sorted({value for value in seen if isinstance(value, int)}))
                        )
                        for value in seen:
                            if not isinstance(value, int):
                                print("  %s" % value, file=sys.stderr)
                                ok = False
                    finally:
                        stop_server(args, process, log, url)
                    traces[name] = trace_path
                if traces.get("old") and traces.get("new"):
                    pairs += 1
                    if not report_diff(label, traces["old"], traces["new"]):
                        ok = False
                else:
                    print("RESULT %s: a trace is missing" % label)
                    ok = False
    print("tick-trace-ab: %d pair(s), %s" % (pairs, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


def dry_run(args):
    args_served = server_args(args)
    streams = [int(value) for value in args.streams.split(",")]
    prompts = [int(value) for value in args.prompt_tokens.split(",")]
    print("# one release binary per build; both need the tick-trace emission")
    if args.site_script:
        for binary in (args.old, args.new):
            print(
                "#   SPARKFLASH_BIN=%s SPARKFLASH_TICK_TRACE=<trace> bash %s launch"
                % (binary, args.site_script)
            )
    else:
        for binary in (args.old, args.new):
            print("#   %s serve %s" % (binary, " ".join(args_served)))
    for tokens in prompts:
        for count in streams:
            for rep in range(args.reps):
                label = "s%d-p%d-r%d" % (count, tokens, rep)
                order = ["old", "new"] if rep % 2 == 0 else ["new", "old"]
                for name in order:
                    trace = os.path.join(args.out, "%s-%s.trace" % (label, name))
                    binary = args.old if name == "old" else args.new
                    if args.site_script:
                        print(
                            "SPARKFLASH_BIN=%s SPARKFLASH_TICK_TRACE=%s bash %s launch"
                            % (binary, trace, args.site_script)
                        )
                        continue
                    print(
                        "SPARKFLASH_TICK_TRACE=%s %s serve %s"
                        % (
                            trace,
                            args.old if name == "old" else args.new,
                            " ".join(args_served),
                        )
                    )
                print(
                    "#   then %d concurrent greedy chat requests of %d prompt tokens, "
                    "%d completion tokens" % (count, tokens, args.max_tokens)
                )
    print("# diff each pair with: scripts/tick-trace-ab.py --diff OLD.trace NEW.trace")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", help="the frozen pre-move sparkflash binary")
    parser.add_argument("--new", help="the new-engine sparkflash binary")
    parser.add_argument("--checkpoint", help="the DeepSeek-V4.1 checkpoint directory")
    parser.add_argument(
        "--site-script",
        help="launch the four-rank group with this site script (e.g. scripts/serve-four-node.sh) "
        "instead of one serve process; SPARKFLASH_BIN names the build and the trace is forwarded "
        "to rank 0",
    )
    parser.add_argument("--model", help="pass through to serve --model")
    parser.add_argument("--synthetic", action="store_true", help="serve --synthetic (no checkpoint)")
    parser.add_argument("--streams", default="1,8,16")
    parser.add_argument("--prompt-tokens", default="2048,16384")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--stagger",
        type=float,
        default=0.0,
        help="seconds between the starts of consecutive requests; 0 fires them together, which "
        "makes a tick's admission set a race between launches",
    )
    parser.add_argument("--out", default="target/tick-trace-ab")
    parser.add_argument("--bind", default="127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument(
        "--unordered",
        action="store_true",
        help="with --diff: compare the traces as multisets of decisions, so an admission "
        "that landed one tick later still matches",
    )
    parser.add_argument(
        "--ignore-tick",
        action="store_true",
        help="with --diff: compare the decisions only, not the tick each ran at (a schedule "
        "whose admissions land a tick apart is then reported as identical decisions)",
    )
    args = parser.parse_args()

    if args.diff:
        if args.unordered:
            return (
                0
                if report_unordered("diff", args.diff[0], args.diff[1])
                else 1
            )
        return (
            0
            if report_diff("diff", args.diff[0], args.diff[1], args.ignore_tick)
            else 1
        )
    if args.dry_run:
        return dry_run(args)
    if not args.old or not args.new:
        parser.error("--old and --new are required (or --dry-run / --diff)")
    if not args.site_script and not args.checkpoint and not args.synthetic:
        parser.error("--checkpoint is required (or --synthetic, or --site-script)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
