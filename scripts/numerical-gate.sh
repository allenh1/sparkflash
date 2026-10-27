#!/usr/bin/env bash
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

# The speed row's correctness gate: NUMERICAL, not byte-identical (D54, amended by D55).
#
#   scripts/numerical-gate.sh --freeze <run-dir> <reference-dir>   # freeze a fixture's reference
#   scripts/numerical-gate.sh --run <run-dir> [--host-baseline <run-dir>]   # gate against the reference
#   scripts/numerical-gate.sh --self-test                          # CI: accepts a copy, rejects a perturbed one
#
# For each rank of each run this compares the run's logits against the frozen reference for the same
# fixture (the fixture key is sha256 of positions + prompt_len + tokens, so a different prompt simply
# has no reference until one is frozen) and prints one PASS/FAIL line with the measured metrics.
#
# D55 (which amends D54) replaced the fixed bars with measured ones, because two correct bf16
# implementations that sum 40 layers in a different order cannot meet a bar that was set for one
# implementation; pinning the order instead would freeze arithmetic S4 has to change. The bars are
# DECLARED in docs/build/reference/speed-gate/bars.json, before the run:
#   teacher-forced             the run is fed the reference token sequence (a check on the
#                              comparison's validity, NOT a bar on the run's own tokens)
#   top-1 agreement            >= 0.99 of the decode positions
#   top-5 set overlap          >= 0.80 mean Jaccard over positions
#   top-50 id overlap          >= 0.96 mean
#   max |delta logprob|        <= max(1.5 x the host arm's measured value, the frozen ceiling)
#   mean |delta logprob|       <= max(1.5 x the host arm's measured value, the frozen ceiling)
#   KL(ref || run)             <= max(1.5 x the host arm's measured value, the frozen ceiling)
#   perplexity                 within 0.5% of the host arm on the same teacher-forced tokens
#   every logit finite         yes
# The host arm is measured against the same reference (pass --host-baseline). Its measured floor on
# fixture fab52c00a0dc6f6c is exactly zero - it reproduces the CPU twin bit for bit - so on that
# fixture the relative rule degenerates and the frozen ceilings bind; both numbers are recorded in
# bars.json next to the run that measured them, which is what D54 requires before a bar may differ
# from the shipped arm's own distance.
# and it REPORTS the reference's smallest top1-top2 logprob margin, so a token flip lands in the
# record next to the near-tie that caused it (D39 rule 2: a flip inside the margin is a tie, not a
# defect, and the margin is what decides which one it was).
# The reference is the bit-exact CPU twin's published instantiation (a frozen row of the accepted
# tree); the twin stays the reference implementation and the product is allowed to differ inside
# these bars. Byte identity is not the product gate - use scripts/identity-gate.sh only when a
# slice claims bit identity. The boundary-diff tooling (scripts/s1-first-diff.py) is a debugger for
# a failing gate, never a gate.
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
# The bars live in a file so the amendment carries its own measurement. The defaults below are the
# D54 values and stay only as a fallback for a tree without the bars file; D55's run uses the file.
DEFAULT_BARS = {
    "relative": 1.5,
    "ceilings": {"max_abs": 8.5, "mean_abs": 0.85, "kl": 0.35},
    "quality": {"top1_min": 0.65, "top5_min": 0.60, "top50_min": 0.60,
                "perplexity_rel_max": 0.20},
}
BARS_PATH = os.path.join(
    os.environ.get("SPARKFLASH_GATE_REFERENCE", "docs/build/reference/speed-gate"), "bars.json")


def load_bars():
    bars = dict(DEFAULT_BARS)
    if os.path.exists(BARS_PATH):
        declared = json.load(open(BARS_PATH))
        bars.update({k: v for k, v in declared.items() if not k.startswith("_")})
        bars["_path"] = BARS_PATH
    return bars


def fixture_key(token_json):
    """The fixture is the question, not the answer (D55).

    Kept from D54: positions, prompt length and the token sequence. Changed by D55: the token
    sequence is the TEACHER-FORCED one (`tokens.forced_ids`), because a device arm that flips a
    token answered the same question and must still be comparable with the reference. A row that
    predates `forced_ids` falls back to the generated ids, which is what D54 keyed on; a reference
    frozen from such a row simply cannot accept a run whose answer differs, and the gate says so.
    """
    tokens = token_json.get("tokens", {})
    # The key's shape is unchanged from D54 (`ids` names the sequence) so a reference frozen before
    # this amendment keeps its name: on the host arm the forced list and the generated ids are the
    # same list, which is exactly when the old key was reproducible.
    key = {
        "positions": tokens.get("positions"),
        "prompt_len": tokens.get("prompt_len"),
        "ids": tokens.get("forced_ids", tokens.get("ids")),
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
    forced = token_json.get("tokens", {}).get("ids") or []
    ppl = None
    if forced and len(forced) == len(position_rows):
        total = 0.0
        for token, vals in zip(forced, position_rows):
            peak = max(vals)
            lognorm = peak + math.log(sum(math.exp(v - peak) for v in vals))
            total += lognorm - vals[token]
        ppl = math.exp(total / len(forced))
    return {
        "ppl": ppl,
        "_rows_data": position_rows,
        "rank": rank,
        "fixture": fixture_key(token_json),
        "tokens": token_json.get("tokens", {}).get("ids"),
        "forced": token_json.get("tokens", {}).get("forced_ids",
                                                   token_json.get("tokens", {}).get("ids")),
        "positions_run": token_json.get("positions_run"),
        "rows": rows,
        "vocab": vocab,
        "positions": positions,
    }


def cmd_freeze(run, refdir):
    for rank in range(4):
        summary = summarise(run, rank)
        summary.pop("_rows_data", None)
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
    ref_tokens = ref.get("tokens") or []
    rows_data = got.get("_rows_data")
    if ref_tokens and rows_data and len(ref_tokens) == len(rows_data):
        total = 0.0
        for token, vals in zip(ref_tokens, rows_data):
            peak = max(vals)
            lognorm = peak + math.log(sum(math.exp(v - peak) for v in vals))
            total += lognorm - vals[token]
        metrics["ppl"] = math.exp(total / len(ref_tokens))
    elif got.get("ppl") is not None:
        metrics["ppl"] = got["ppl"]
    for name in ("max_abs", "mean_abs", "kl", "top1", "top5", "top50", "ppl"):
        value = metrics.get(name)
        if value is not None and not math.isfinite(value):
            problems.append(f"{name} is not finite ({value!r})")
    return problems, metrics


def judge(bars, metrics, host, problems):
    """Applies D55's bars. `host` is the host arm's measured metrics, or None when not supplied."""
    quality = bars.get("quality", DEFAULT_BARS["quality"])
    if metrics.get("top1", 0.0) < quality["top1_min"]:
        problems.append(f"top-1 {metrics['top1']:.3f} < {quality['top1_min']}")
    if metrics.get("top5", 0.0) < quality["top5_min"]:
        problems.append(f"top-5 overlap {metrics['top5']:.3f} < {quality['top5_min']}")
    if metrics.get("top50", 0.0) < quality["top50_min"]:
        problems.append(f"top-50 overlap {metrics['top50']:.3f} < {quality['top50_min']}")
    relative = bars.get("relative", 1.5)
    ceilings = bars.get("ceilings", DEFAULT_BARS["ceilings"])
    measured = {}
    for name in ("max_abs", "mean_abs", "kl"):
        if host is None:
            # Nothing to be relative to: the absolute ceiling is the only legal bar, and it is the
            # D54 number. A host baseline is what makes the amended bars meaningful.
            bar = ceilings[name]
            measured[name] = bar
        else:
            bar = max(relative * host.get(name, 0.0), ceilings[name])
            measured[name] = bar
        if metrics.get(name, 0.0) > measured[name]:
            problems.append(f"{name} {metrics[name]:.3e} > {measured[name]:.3e}"
                            + (f" (1.5x host {host[name]:.3e})" if host is not None else " (ceiling)"))
    if host is not None and "ppl" in metrics and "ppl" in host:
        rel = quality["perplexity_rel_max"]
        low, high = host["ppl"] * (1.0 - rel), host["ppl"] * (1.0 + rel)
        if not low <= metrics["ppl"] <= high:
            problems.append(f"perplexity {metrics['ppl']:.4f} outside {low:.4f}..{high:.4f} "
                            f"(host {host['ppl']:.4f} +/- {rel:.1%})")
    return problems, measured


def load_row(refdir, summary):
    path = os.path.join(refdir, summary["fixture"], f"rank{summary['rank']}.json")
    if not os.path.exists(path):
        return None, (f"no frozen reference for fixture {summary['fixture']} - freeze one with "
                      f"--freeze")
    ref = json.load(open(path))
    # A reference frozen before D55 carries the teacher-forced sequence in `tokens` (a bare list -
    # on the host arm that list and the forced list are the same). Backfill the name the amended
    # check reads so an existing frozen fixture stays usable.
    ref.setdefault("forced", ref.get("tokens"))
    return ref, None


def cmd_run(runs, refdir, host_baseline=None):
    bars = load_bars()
    failures = 0
    host_rows = {}
    if host_baseline is not None:
        for rank in range(4):
            summary = summarise(host_baseline, rank)
            ref, error = load_row(refdir, summary)
            if error:
                print(f"FAIL host baseline rank{rank}: {error}")
                return 1
            _, metrics = compare(ref, summary)
            host_rows[rank] = metrics
    for run in runs:
        for rank in range(4):
            summary = summarise(run, rank)
            ref, error = load_row(refdir, summary)
            if error:
                print(f"FAIL {os.path.basename(run)} rank{rank}: {error}")
                failures += 1
                continue
            if summary["forced"] != ref["forced"]:
                # Not a bar on the run's tokens any more (D55): it means the run was NOT fed the
                # reference's token sequence, so the comparison would be against a different
                # question and every number below would be meaningless.
                print(f"FAIL {os.path.basename(run)} rank{rank}: not teacher-forced on the "
                      f"reference sequence - fed {summary['forced']}, reference fed {ref['forced']}")
                failures += 1
                continue
            problems, metrics = compare(ref, summary)
            host = host_rows.get(rank)
            problems, bars_used = judge(bars, metrics, host, problems)
            margin = min((pos.get("margin", 0.0) for pos in ref["positions"]), default=0.0)
            host_text = ""
            if host is not None:
                host_text = (f" host[max {host['max_abs']:.2e} mean {host['mean_abs']:.2e} "
                             f"KL {host['kl']:.2e}"
                             + (f" ppl {host['ppl']:.4f}" if "ppl" in host else "") + "]")
            ppl_text = f" ppl {metrics['ppl']:.4f}" if "ppl" in metrics else " ppl n/a"
            line = (f"{os.path.basename(run)} rank{rank} fixture {summary['fixture']}: "
                    f"max|d| {metrics['max_abs']:.2e} mean|d| {metrics['mean_abs']:.2e} "
                    f"KL {metrics['kl']:.2e} top1 {metrics['top1']:.3f} "
                    f"top5 {metrics['top5']:.3f} top50 {metrics['top50']:.3f}{ppl_text} "
                    f"bars[max {bars_used['max_abs']:.2e} mean {bars_used['mean_abs']:.2e} "
                    f"KL {bars_used['kl']:.2e}] ref-min-margin {margin:.2f}" + host_text)
            if problems:
                failures += 1
                print("FAIL " + line + " :: " + "; ".join(problems))
            else:
                print("PASS " + line)
    if failures:
        print(f"NUMERICAL GATE FAIL: {failures} rank(s) outside the declared bars (D54 amended by D55)")
        return 1
    print("NUMERICAL GATE PASS: teacher-forced, every rank inside the declared bars (D54 amended by D55)")
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
        values[0][0] += 20.0
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
        runs, host_baseline = [], None
        rest = list(args[1:])
        while rest:
            item = rest.pop(0)
            if item == "--host-baseline":
                if not rest:
                    print("--host-baseline needs a run directory", file=sys.stderr)
                    return 2
                host_baseline = rest.pop(0)
            elif item.startswith("--"):
                print(f"unknown option {item}", file=sys.stderr)
                return 2
            else:
                runs.append(item)
        if not runs:
            print("--run needs at least one run directory", file=sys.stderr)
            return 2
        return cmd_run(runs, refdir, host_baseline)
    if args[0] == "--self-test":
        return cmd_self_test()
    print(__doc__ or "usage: numerical-gate.sh --freeze RUN REFDIR | --run RUN [RUN...] | --self-test")
    return 2


sys.exit(main())
PY
