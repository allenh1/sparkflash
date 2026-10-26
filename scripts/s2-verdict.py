#!/usr/bin/env python3
"""S2-d / S2-e verdicts for four-rank window rows (docs/build/s2-result.md).

    python3 scripts/s2-verdict.py replicates --run <row A> --run <row B> [--tolerance-ms 2]
    python3 scripts/s2-verdict.py soak --run <soak row> [--min-steps 1000] [--p99-ratio 1.10]
    python3 scripts/s2-verdict.py --self-test

A row directory is the window layout speed-log.sh reads: rank0..rank3/ each holding
tp4-token-rN.json (the envelope), tp4-forward-embed-rN.json (the forward record) and the rank's logs.

Every row, in both modes, must show on all four ranks:
  * forward == "run", no error, one token list and one logits sha across the ranks;
  * the S2 run header: arm "device", reduce_path "nccl", an NCCL build recorded;
  * the forward record's head positions with d2h_copies == 1, allocations == 0, sync_waits == 0,
    and the four collective spans registered (layer_device_memory.reduce_registered_buffers == 4);
  * no NCCL error or timeout in any log under the rank directory.

replicates (S2-d): the two rows' published numbers (the four-rank mean of step_ms.mean_ms, the
speed-log value) agree within --tolerance-ms, and their logits shas are byte-identical. The medians
are reported, not gated: the step samples are integer milliseconds, so a 9-sample median moves in
0.25-0.5 ms steps that say nothing about drift. A row directory does not record its clocks, so the
verdict notes that the clock protocol has to come from the window record; per DECISIONS.md D57 that
protocol is the nodes' native boost clocks (2405-2411 MHz), not the retired 1500 MHz lock, and both
S2-d rows are boost rows. A PASS here is not an S2-d PASS without the window record stating the
clocks.

soak (S2-e): at least --min-steps decode steps on every rank, and on every rank p99 <= --p99-ratio x
median of its step samples (the four ranks are rank-synchronised by the collective, though their step
counters can be offset by one at a stall, so a rank's own samples stay a valid measure of the step).
A rank whose mean lies above its p99 is noted (not failed): the steps beyond the p99 then hold a
stall the bar does not see, and the note states how much.

The numerical gate is separate and must also PASS: scripts/numerical-gate.sh --run <row>
--host-baseline <host row>. Exit 0 PASS, 1 FAIL, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import tempfile

SCHEMA = "sparkflash-s2-verdict/1"
CLOCK_NOTE = ("clock protocol not verified: a row directory does not record its clocks; S2-d's rows are "
              "native-boost rows (DECISIONS.md D57, which retired the 1500 MHz lock) and the window record must "
              "state the clocks they ran at")
RANKS = (0, 1, 2, 3)
LOG_SUFFIXES = (".log", ".err", ".out", ".txt")
# What a healthy NCCL run never prints. "NCCL WARN" is NCCL's own warning prefix; the error codes are
# ncclResult_t names as ncclGetErrorString and the binding's messages spell them.
NCCL_FAULT = re.compile(
    r"NCCL WARN|ncclSystemError|ncclInternalError|ncclRemoteError|ncclUnhandledCudaError"
    r"|ncclInvalidUsage|ncclInProgress|NCCL[^\n]*(?:timeout|timed out)|asynchronous failure"
    r"|AsyncError",
    re.IGNORECASE,
)


class InputError(RuntimeError):
    """A row that cannot be read honestly."""


def load(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise InputError("{}: {}".format(path, error)) from error


def percentile(ordered, fraction):
    """Nearest-rank percentile of an already sorted list."""
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(round(fraction * len(ordered), 9)) - 1))
    return ordered[index]


def nccl_faults(rank_dir):
    """Every log line under the rank directory that names an NCCL fault, as (file, line)."""
    faults = []
    for dirpath, _dirnames, filenames in os.walk(rank_dir):
        for filename in sorted(filenames):
            if not filename.endswith(LOG_SUFFIXES):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, errors="replace") as handle:
                for line in handle:
                    if NCCL_FAULT.search(line):
                        faults.append((os.path.relpath(path, rank_dir), line.strip()[:240]))
    return faults


def read_row(run):
    """One row's per-rank facts and every problem found in them."""
    if not os.path.isdir(run):
        raise InputError("{} is not a row directory".format(run))
    problems = []
    ranks = []
    for rank in RANKS:
        rank_dir = os.path.join(run, "rank{}".format(rank))
        envelope = load(os.path.join(rank_dir, "tp4-token-r{}.json".format(rank)))
        record_path = os.path.join(rank_dir, "tp4-forward-embed-r{}.json".format(rank))
        record = load(record_path) if os.path.exists(record_path) else None
        samples = [float(value) for value in envelope.get("step_ms", {}).get("samples", [])]
        entry = {
            "rank": rank,
            "forward": envelope.get("forward"),
            "error": envelope.get("error"),
            "mean_ms": float(envelope.get("step_ms", {}).get("mean_ms", 0.0)),
            "median_ms": statistics.median(samples) if samples else None,
            "steps": len(samples),
            "samples": samples,
            "tokens": (envelope.get("tokens") or {}).get("ids"),
            "logits_sha256": (envelope.get("logits_dump") or {}).get("bin_sha256"),
            "header": envelope.get("run_header"),
            "nccl_faults": nccl_faults(rank_dir),
        }
        ranks.append(entry)
        where = "rank {}".format(rank)
        if entry["forward"] != "run" or entry["error"]:
            problems.append("{}: forward {!r}, error {!r}".format(where, entry["forward"], entry["error"]))
        header = entry["header"] or {}
        if header.get("arm") != "device" or header.get("reduce_path") != "nccl":
            problems.append("{}: run header arm {!r} reduce_path {!r}, expected device/nccl (an S2 binary records both)".format(
                where, header.get("arm"), header.get("reduce_path")))
        if not (header.get("nccl_build") or {}).get("version"):
            problems.append("{}: run header records no NCCL build".format(where))
        if record is None:
            problems.append("{}: no forward record ({}): the step invariants are unproven".format(where, record_path))
        else:
            heads = [item.get("device_step_invariants") for item in record.get("position_records", [])
                     if item.get("head_position")]
            if not heads or any(item is None for item in heads):
                problems.append("{}: the forward record has no device step invariants on its head positions".format(where))
            else:
                broken = [item for item in heads
                          if (item.get("d2h_copies"), item.get("allocations"), item.get("sync_waits")) != (1, 0, 0)]
                if broken:
                    problems.append("{}: {} head step(s) break d2h=1/alloc=0/sync=0, first {}".format(
                        where, len(broken), broken[0].get("summary")))
            registered = (record.get("layer_device_memory") or {}).get("reduce_registered_buffers")
            if registered != 4:
                problems.append("{}: {!r} collective spans registered, expected 4".format(where, registered))
        for path, line in entry["nccl_faults"][:3]:
            problems.append("{}: NCCL fault in {}: {}".format(where, path, line))
    if len({json.dumps(entry["tokens"]) for entry in ranks}) != 1:
        problems.append("the four ranks emitted different token lists")
    shas = {entry["logits_sha256"] for entry in ranks}
    if len(shas) != 1 or None in shas:
        problems.append("the four ranks' logits shas differ or are missing: {}".format(sorted(map(str, shas))))
    return {
        "run": run,
        "step_mean_ms": statistics.fmean(entry["mean_ms"] for entry in ranks),
        "median_ms": statistics.fmean(entry["median_ms"] for entry in ranks if entry["median_ms"] is not None)
        if all(entry["median_ms"] is not None for entry in ranks) else None,
        "logits_sha256": next(iter(shas)) if len(shas) == 1 else None,
        "ranks": ranks,
        "problems": problems,
    }


def replicates(runs, tolerance_ms):
    rows = [read_row(run) for run in runs]
    problems = []
    for row in rows:
        problems.extend("{}: {}".format(os.path.basename(row["run"].rstrip("/")), problem) for problem in row["problems"])
    delta = abs(rows[0]["step_mean_ms"] - rows[1]["step_mean_ms"])
    if delta > tolerance_ms:
        problems.append("the replicates' published numbers differ by {:.2f} ms (> {} ms): {:.2f} vs {:.2f}".format(
            delta, tolerance_ms, rows[0]["step_mean_ms"], rows[1]["step_mean_ms"]))
    if rows[0]["logits_sha256"] != rows[1]["logits_sha256"]:
        problems.append("the replicates' logits differ: {} vs {}".format(rows[0]["logits_sha256"], rows[1]["logits_sha256"]))
    return {
        "schema": SCHEMA,
        "mode": "replicates",
        "tolerance_ms": tolerance_ms,
        "rows": [{key: row[key] for key in ("run", "step_mean_ms", "median_ms", "logits_sha256")} for row in rows],
        "delta_ms": delta,
        "clock_protocol": "not recorded in the row; native boost per DECISIONS.md D57",
        "problems": problems,
        "notes": [CLOCK_NOTE],
        "pass": not problems,
    }


def soak(run, min_steps, p99_ratio):
    row = read_row(run)
    problems = list(row["problems"])
    notes = []
    per_rank = []
    for entry in row["ranks"]:
        ordered = sorted(entry["samples"])
        median = statistics.median(ordered) if ordered else None
        p99 = percentile(ordered, 0.99)
        ratio = (p99 / median) if median else None
        per_rank.append({"rank": entry["rank"], "steps": entry["steps"], "median_ms": median, "p99_ms": p99,
                         "max_ms": ordered[-1] if ordered else None, "p99_over_median": ratio})
        if entry["steps"] < min_steps:
            problems.append("rank {}: {} decode steps, the soak needs {}".format(entry["rank"], entry["steps"], min_steps))
        if ratio is None or ratio > p99_ratio:
            problems.append("rank {}: p99 {} ms is {} x the {} ms median (limit {})".format(
                entry["rank"], p99, None if ratio is None else round(ratio, 4), median, p99_ratio))
        if p99 is not None and entry["mean_ms"] > p99:
            notes.append("rank {}: the {:.2f} ms mean is above the {} ms p99, so the steps beyond the p99 hold "
                         "stalls: {:.0f} ms over the median in all, worst step {} ms".format(
                             entry["rank"], entry["mean_ms"], p99, (entry["mean_ms"] - median) * entry["steps"],
                             ordered[-1]))
    return {
        "schema": SCHEMA,
        "mode": "soak",
        "run": run,
        "min_steps": min_steps,
        "p99_ratio": p99_ratio,
        "step_mean_ms": row["step_mean_ms"],
        "ranks": per_rank,
        "nccl_faults": sum(len(entry["nccl_faults"]) for entry in row["ranks"]),
        "problems": problems,
        "notes": notes,
        "pass": not problems,
    }


def report(result):
    print(json.dumps(result, indent=2, sort_keys=True))
    print("s2-verdict {}: {}".format(result["mode"], "PASS" if result["pass"] else "FAIL"))
    for problem in result["problems"]:
        print("problem: {}".format(problem))
    for note in result.get("notes", []):
        print("note: {}".format(note))
    return 0 if result["pass"] else 1


# ---- self-test: synthetic rows, no hardware ----

def write_row(root, name, samples, *, sha="ab" * 32, arm="device", registered=4, head_d2h=1, fault=None):
    run = os.path.join(root, name)
    for rank in RANKS:
        rank_dir = os.path.join(run, "rank{}".format(rank))
        os.makedirs(rank_dir)
        envelope = {
            "forward": "run", "error": None,
            "step_ms": {"samples": samples, "mean_ms": statistics.fmean(samples), "worst_ms": max(samples)},
            "tokens": {"ids": [7, 8, 9]},
            "logits_dump": {"bin_sha256": sha},
            "run_header": {"arm": arm, "reduce_path": "nccl" if arm == "device" else "tcp",
                           "nccl_build": {"version": "2.31.2", "library": "libnccl.so.2"}},
        }
        with open(os.path.join(rank_dir, "tp4-token-r{}.json".format(rank)), "w") as handle:
            json.dump(envelope, handle)
        heads = [{"head_position": True, "device_step_invariants": {
            "d2h_copies": head_d2h, "allocations": 0, "sync_waits": 0, "summary": "d2h={}".format(head_d2h)}}
            for _ in samples]
        record = {"position_records": [{"head_position": False}] + heads,
                  "layer_device_memory": {"reduce_registered_buffers": registered}}
        with open(os.path.join(rank_dir, "tp4-forward-embed-r{}.json".format(rank)), "w") as handle:
            json.dump(record, handle)
        with open(os.path.join(rank_dir, "run.log"), "w") as handle:
            handle.write("forward reduce: requested=\"nccl\" path=nccl\n")
            if fault and rank == 2:
                handle.write(fault + "\n")
    return run


def self_test():
    checks = []

    def check(name, condition):
        checks.append((name, bool(condition)))

    with tempfile.TemporaryDirectory() as root:
        steady = [57.0] * 9
        check("agreeing replicates pass",
              replicates([write_row(root, "a", steady), write_row(root, "b", [58.0] * 9)], 2.0)["pass"])
        check("a 3 ms gap fails",
              not replicates([write_row(root, "c", steady), write_row(root, "d", [60.0] * 9)], 2.0)["pass"])
        check("different logits fail",
              not replicates([write_row(root, "e", steady), write_row(root, "f", steady, sha="cd" * 32)], 2.0)["pass"])
        check("a host-arm row fails",
              not replicates([write_row(root, "g", steady, arm="host"), write_row(root, "h", steady)], 2.0)["pass"])
        check("an unregistered row fails",
              not replicates([write_row(root, "i", steady, registered=0), write_row(root, "j", steady)], 2.0)["pass"])
        check("a broken invariant fails",
              not replicates([write_row(root, "k", steady, head_d2h=2), write_row(root, "l", steady)], 2.0)["pass"])
        flat = [57.0] * 990 + [62.0] * 10
        check("a flat soak passes", soak(write_row(root, "m", flat), 1000, 1.10)["pass"])
        tail = [57.0] * 980 + [70.0] * 20
        check("a heavy tail fails", not soak(write_row(root, "n", tail), 1000, 1.10)["pass"])
        check("a short soak fails", not soak(write_row(root, "o", [57.0] * 999), 1000, 1.10)["pass"])
        stalled = soak(write_row(root, "q", [57.0] * 995 + [2000.0] * 5), 1000, 1.10)
        check("stalls beyond the p99 pass the bar but are noted",
              stalled["pass"] and len(stalled["notes"]) == 4 and "worst step 2000.0 ms" in stalled["notes"][0])
        check("a replicate verdict says the clock protocol is not in the row",
              replicates([write_row(root, "r", steady), write_row(root, "s", steady)], 2.0)["notes"] == [CLOCK_NOTE])
        timeout = "spark-2:123:456 [0] NCCL WARN NET/IB : Got completion with error 12, opcode 0"
        result = soak(write_row(root, "p", flat, fault=timeout), 1000, 1.10)
        check("an NCCL warning fails the soak", not result["pass"] and result["nccl_faults"] == 1)
        check("nearest-rank p99 of 1..100 is 99", percentile(list(range(1, 101)), 0.99) == 99)
    for name, ok in checks:
        print("{} {}".format("ok  " if ok else "FAIL", name))
    failed = [name for name, ok in checks if not ok]
    print("self-test: {}/{} passed".format(len(checks) - len(failed), len(checks)))
    return 1 if failed else 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("replicates", "soak"))
    parser.add_argument("--run", action="append", default=[], help="a row directory (twice for replicates)")
    parser.add_argument("--tolerance-ms", type=float, default=2.0)
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--p99-ratio", type=float, default=1.10)
    parser.add_argument("--out", help="also write the verdict JSON here")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        if args.mode == "replicates":
            if len(args.run) != 2:
                parser.error("replicates needs exactly two --run rows")
            result = replicates(args.run, args.tolerance_ms)
        elif args.mode == "soak":
            if len(args.run) != 1:
                parser.error("soak needs exactly one --run row")
            result = soak(args.run[0], args.min_steps, args.p99_ratio)
        else:
            parser.error("name a mode (replicates|soak) or --self-test")
    except InputError as error:
        print("s2-verdict: {}".format(error), file=sys.stderr)
        return 2
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return report(result)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
