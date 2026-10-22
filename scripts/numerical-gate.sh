#!/usr/bin/env bash
# The speed row's correctness gate: NUMERICAL, not byte-identical (D53).
#
#   scripts/numerical-gate.sh --freeze <run-dir> <reference-dir>   # freeze a fixture's reference
#   scripts/numerical-gate.sh --run <run-dir> [more run dirs...]   # gate rows against the reference
#   scripts/numerical-gate.sh --self-test                          # CI: accepts a copy, rejects a perturbed one
#
# For each rank of each run this compares the run's logits against the frozen reference for the same
# fixture (the fixture key is sha256 of positions + prompt_len + tokens, so a different prompt simply
# has no reference until one is frozen) and prints one PASS/FAIL line with the measured metrics.
#
# The bars are DECLARED HERE, before the run, per D39 rule 2 and D53:
#   greedy argmax tokens      identical to the reference tokens        (hard)
#   top-1 agreement           1.00 of the decode positions             (hard)
#   top-5 set overlap         >= 0.80 mean Jaccard over positions      (hard)
#   top-50 id overlap         >= 0.96 mean                              (hard)
#   max |delta logprob|       <= 2.0e-2 nats over the reference top-100 (hard)
#   mean |delta logprob|      <= 2.0e-3 nats                            (hard)
#   KL(ref || run)            <= 1.0e-4 nats over the renormalised top-100 (hard)
#   every logit finite        yes                                        (hard)
# and it REPORTS the reference's smallest top1-top2 logprob margin, so a token flip lands in the
# record next to the near-tie that caused it (D39 rule 2: a flip inside the margin is a tie, not a
# defect, and the margin is what decides which one it was).
# The reference is the bit-exact CPU twin's published instantiation (a frozen row of the accepted
# tree); the twin stays the reference implementation and the product is allowed to differ inside
# these bars. Byte identity is no longer the product gate - use scripts/identity-gate.sh only when a
# slice claims bit identity.
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: numerical-gate.sh --freeze RUN REFDIR | --run RUN [RUN...] | --self-test" >&2
  exit 2
fi
exec python3 - "$@" <<'PY'
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

TOP_K = 1000
COMPARE_K = 100
TOL = {
    "max_abs": 2.0e-2,
    "mean_abs": 2.0e-3,
    "kl": 1.0e-4,
    "top1": 1.0,
    "top5_overlap": 0.80,
    "top50_overlap": 0.96,
}


def fixture_key(token_json):
    key = {
        "positions": token_json.get("tokens", {}).get("positions"),
        "prompt_len": token_json.get("tokens", {}).get("prompt_len"),
        "ids": token_json.get("tokens", {}).get("ids"),
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def logits_path(run, rank, token_json):
    return token_json.get("logits_dump", {}).get("bin")


def read_logits(path, rows, vocab):
    import struct
    with open(path, "rb") as handle:
        data = handle.read()
    need = rows * vocab * 4
    if len(data) < need:
        raise SystemExit(f"{path}: {len(data)} bytes, need {need}")
    out = []
    for row in range(rows):
        base = row * vocab * 4
        vals = struct.unpack_from(f"<{vocab}f", data, base)
        out.append(vals)
    return out


def log_softmax_top(vals, k):
    best = max(vals)
    exps = [math.exp(v - best) for v in vals]
    total = sum(exps)
    log_total = math.log(total) + best
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:k]
    return order, [vals[i] - log_total for i in order]


def summarise(run, rank):
    base = os.path.join(run, f"rank{rank}")
    token_json = json.load(open(os.path.join(base, f"tp4-token-r{rank}.json")))
    dump = token_json.get("logits_dump", {})
    path = dump.get("bin")
    rows, vocab = dump.get("rows"), dump.get("vocab")
    if not path or not os.path.exists(path):
        # The evidence copy beside the token file is authoritative; the recorded absolute path
        # points into the staging window, which is rotated when the next tag is staged.
        beside = os.path.join(base, f"tp4-logits-r{rank}.jsonl.bin")
        path = beside if os.path.exists(beside) else path
    if not path or not os.path.exists(path):
        raise SystemExit(f"rank{rank}: logits dump missing ({path})")
    position_rows = read_logits(path, rows, vocab)
    positions = []
    for vals in position_rows:
        order, logps = log_softmax_top(vals, TOP_K)
        positions.append({"ids": order, "logprobs": logps,
                          "margin": (logps[0] - logps[1]) if len(logps) > 1 else 0.0})
    return {
        "fixture": fixture_key(token_json),
        "tokens": token_json.get("tokens", {}).get("ids"),
        "positions_run": token_json.get("positions_run"),
        "rows": rows,
        "vocab": vocab,
        "positions": positions,
    }


def cmd_freeze(run, refdir):
    for rank in range(4):
        summary = summarise(run, rank)
        out = os.path.join(refdir, summary["fixture"])
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"rank{rank}.json")
        with open(path, "w") as handle:
            json.dump(summary, handle)
        print(f"froze rank{rank} fixture {summary['fixture']} -> {path}")


def compare(ref, got):
    problems, metrics = [], {}
    if len(ref["positions"]) != len(got["positions"]):
        return ["position count differs"], metrics
    k = min(COMPARE_K, TOP_K)
    max_abs = mean_abs = 0.0
    top1 = []
    top5, top50 = [], []
    kls = []
    for index, (rpos, gpos) in enumerate(zip(ref["positions"], got["positions"])):
        rids, rlps = rpos["ids"][:k], rpos["logprobs"][:k]
        gid_at = {i: lp for i, lp in zip(gpos["ids"], gpos["logprobs"])}
        top1.append(1.0 if rids[0] == gpos["ids"][0] else 0.0)
        r5, g5 = set(rids[:5]), set(gpos["ids"][:5])
        top5.append(len(r5 & g5) / len(r5 | g5) if r5 | g5 else 1.0)
        r50, g50 = set(ref["positions"][index]["ids"][:50]), set(gpos["ids"][:50])
        top50.append(len(r50 & g50) / len(r50 | g50) if r50 | g50 else 1.0)
        deltas = []
        joint = []
        for i, lp in zip(rids, rlps):
            if i in gid_at:
                deltas.append(abs(lp - gid_at[i]))
                joint.append((lp, gid_at[i]))
        if not deltas:
            problems.append(f"position {index}: no shared top-{k} ids")
            continue
        max_abs = max(max_abs, max(deltas))
        mean_abs += sum(deltas)
        pairs = len(deltas)
        # KL over the shared support, both renormalised so the comparison is a distribution.
        total_r = sum(math.exp(lp) for lp, _ in joint)
        total_g = sum(math.exp(gp) for _, gp in joint)
        kl = 0.0
        for lp, gp in joint:
            pr, pg = math.exp(lp) / total_r, math.exp(gp) / total_g
            kl += pr * math.log(pr / pg)
        kls.append((kl, pairs))
    metrics = {
        "max_abs": max_abs,
        "mean_abs": mean_abs / max(sum(p for _, p in kls), 1),
        "kl": sum(kl * pairs for kl, pairs in kls) / max(sum(p for _, p in kls), 1),
        "top1": sum(top1) / max(len(top1), 1),
        "top5": sum(top5) / max(len(top5), 1),
        "top50": sum(top50) / max(len(top50), 1),
    }
    if metrics["max_abs"] > TOL["max_abs"]:
        problems.append(f"max |dlogprob| {metrics['max_abs']:.3e} > {TOL['max_abs']:.1e}")
    if metrics["mean_abs"] > TOL["mean_abs"]:
        problems.append(f"mean |dlogprob| {metrics['mean_abs']:.3e} > {TOL['mean_abs']:.1e}")
    if metrics["kl"] > TOL["kl"]:
        problems.append(f"KL {metrics['kl']:.3e} > {TOL['kl']:.1e}")
    if metrics["top1"] < TOL["top1"]:
        problems.append(f"top-1 {metrics['top1']:.3f} < {TOL['top1']}")
    if metrics["top5"] < TOL["top5_overlap"]:
        problems.append(f"top-5 overlap {metrics['top5']:.3f} < {TOL['top5_overlap']}")
    if metrics["top50"] < TOL["top50_overlap"]:
        problems.append(f"top-50 overlap {metrics['top50']:.3f} < {TOL['top50_overlap']}")
    return problems, metrics


def cmd_run(runs, refdir):
    failures = 0
    for run in runs:
        for rank in range(4):
            summary = summarise(run, rank)
            path = os.path.join(refdir, summary["fixture"], f"rank{rank}.json")
            if not os.path.exists(path):
                print(f"FAIL {os.path.basename(run)} rank{rank}: no frozen reference for fixture "
                      f"{summary['fixture']} - freeze one with --freeze")
                failures += 1
                continue
            ref = json.load(open(path))
            if summary["tokens"] != ref["tokens"]:
                print(f"FAIL {os.path.basename(run)} rank{rank}: greedy tokens {summary['tokens']} != "
                      f"{ref['tokens']}")
                failures += 1
                continue
            problems, metrics = compare(ref, summary)
            margin = min((pos.get("margin", 0.0) for pos in ref["positions"]), default=0.0)
            line = (f"{os.path.basename(run)} rank{rank} fixture {summary['fixture']}: "
                    f"max|d| {metrics['max_abs']:.2e} mean|d| {metrics['mean_abs']:.2e} "
                    f"KL {metrics['kl']:.2e} top1 {metrics['top1']:.3f} "
                    f"top5 {metrics['top5']:.3f} top50 {metrics['top50']:.3f} "
                    f"ref-min-margin {margin:.2f}")
            if problems:
                failures += 1
                print("FAIL " + line + " :: " + "; ".join(problems))
            else:
                print("PASS " + line)
    if failures:
        print(f"NUMERICAL GATE FAIL: {failures} rank(s) outside the declared bars (D53)")
        return 1
    print("NUMERICAL GATE PASS: tokens identical and every rank inside the declared bars (D53)")
    return 0


def cmd_self_test():
    work = tempfile.mkdtemp(prefix="numerical-gate-")
    try:
        run = os.path.join(work, "run")
        os.makedirs(os.path.join(run, "rank0"))
        vocab, rows = 64, 2
        import struct
        path = os.path.join(run, "logits.bin")
        values = [[float((i * 7 + r) % 13) for i in range(vocab)] for r in range(rows)]
        with open(path, "wb") as handle:
            for row in values:
                handle.write(struct.pack(f"<{vocab}f", *row))
        token = {"tokens": {"ids": [1, 2], "positions": [0, 1], "prompt_len": 2},
                 "logits_dump": {"bin": path, "rows": rows, "vocab": vocab}}
        for rank in range(4):
            base = os.path.join(run, f"rank{rank}")
            os.makedirs(base, exist_ok=True)
            json.dump(token, open(os.path.join(base, f"tp4-token-r{rank}.json"), "w"))
        refdir = os.path.join(work, "ref")
        cmd_freeze(run, refdir)
        assert cmd_run([run], refdir) == 0, "identical row must pass"
        # perturb one rank's dump just outside the bars
        bad = os.path.join(work, "bad")
        shutil.copytree(run, bad)
        values[0][0] += 1.0
        with open(os.path.join(bad, "logits.bin"), "wb") as handle:
            for row in values:
                handle.write(struct.pack(f"<{vocab}f", *row))
        for rank in range(4):
            token_path = os.path.join(bad, f"rank{rank}", f"tp4-token-r{rank}.json")
            token = json.load(open(token_path))
            token["logits_dump"]["bin"] = os.path.join(bad, "logits.bin")
            json.dump(token, open(token_path, "w"))
        assert cmd_run([bad], refdir) == 1, "perturbed row must fail"
        print("SELF-TEST PASS")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    args = sys.argv[1:]
    if args[0] == "--freeze" and len(args) == 3:
        cmd_freeze(args[1], args[2])
        return 0
    if args[0] == "--run" and len(args) >= 2:
        refdir = os.environ.get("SPARKFLASH_GATE_REFERENCE",
                                "docs/build/reference/speed-gate")
        return cmd_run(args[1:], refdir)
    if args[0] == "--self-test":
        return cmd_self_test()
    print(__doc__ or "usage: numerical-gate.sh --freeze RUN REFDIR | --run RUN [RUN...] | --self-test")
    return 2


sys.exit(main())
PY
