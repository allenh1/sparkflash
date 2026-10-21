#!/usr/bin/env python3
"""D41 bars dry-run proof for the task-12 proposal.

Proves, against the canonical oracle and the existing m1w refused tree, that a bars file only
changes the oracle leg's BARED/UNBARED/FAIL state and can never turn a refusal (no token) into a
pass. Also proves the proposal file itself (status proposal) is refused by the gate loader.

All synthetic artefacts are labelled DRY-RUN; they are not device evidence and are written under
--out-dir only. Exit 0 when every expectation holds, 1 when a check fails, 2 on refused input.
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ACCEPT = os.path.join(HERE, "accept_device_token.py")
WINDOW = os.path.join(HERE, "window_acceptance.py")
ORACLE = "/home/user/ev-session-20261020/qa/artifacts/native-oracle-suite.json"
M1W_DEVICE = "/home/user/ev-session-20261020/window/m1w"
PROMPT = "b69-fixture-130"
POSITIONS = list(range(129, 138))
POSITIONS_STR = ",".join(str(position) for position in POSITIONS)
VOCAB = 129280
ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
MOE = "routed_experts_gate_up=61440,routed_experts_down=30720,shared_experts=240,router=120"


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=1)
        handle.write("\n")
    return path


def oracle_rows():
    doc = load(ORACLE)
    prompt = [item for item in doc["prompts"] if item["id"] == PROMPT][0]
    rows = {}
    for item in prompt["positions"]:
        position = int(item["position"])
        logprobs = {int(token): float(value) for token, value in item["logprobs"].items()}
        rows[position] = {
            "logprobs": logprobs,
            "target": item.get("target_token"),
            "argmax": max(logprobs, key=lambda token: (logprobs[token], -token)),
        }
    return rows


def full_logits(logprobs):
    """A full-vocab distribution whose oracle-token logprobs equal the oracle's, with the
    unobserved tail spread uniformly (the same convention the oracle support implies)."""
    ids = list(logprobs)
    ids.sort()
    total = sum(math.exp(logprobs[token]) for token in ids)
    tail = (1.0 - total) / float(VOCAB - len(ids))
    tail_logit = math.log(tail) if tail > 0 else -200.0
    logits = [tail_logit] * VOCAB
    for token in ids:
        logits[token] = logprobs[token]
    return logits


def write_dump(path, rows):
    dump_path = path + ".jsonl"
    bin_path = dump_path + ".bin"
    offset = 0
    lines = []
    with open(bin_path, "wb") as handle:
        for position, logits, reference_id in rows:
            handle.write(struct.pack("<%df" % len(logits), *logits))
            lines.append({"position": position, "vocab": len(logits), "logits_bin": os.path.basename(bin_path), "logits_offset": offset, "reference_id": reference_id})
            offset += len(logits) * 4
    with open(dump_path, "w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
    return dump_path, bin_path


def build_synthetic(oracle, outdir):
    root = os.path.join(outdir, "synthetic")
    device = os.path.join(root, "device", "rank0")
    twin = os.path.join(root, "twin", "rank0")
    os.makedirs(device, exist_ok=True)
    os.makedirs(twin, exist_ok=True)
    tokens = [oracle[position]["argmax"] for position in POSITIONS]
    canonical = {
        "schema": "sparkflash-device-token-evidence/1",
        "kind": "device",
        "rank": 0,
        "world_size": 4,
        "prompt_tokens": [201] * 130,
        "token_ids": tokens,
        "positions": POSITIONS,
        "token_emitted": True,
        "note": "DRY-RUN synthetic token row (not device evidence)",
    }
    write_json(os.path.join(device, "tp4-token-r0.json"), canonical)
    write_json(os.path.join(twin, "tp4-token-r0.json"), dict(canonical, kind="host-twin", note="DRY-RUN synthetic twin (not host evidence)"))
    ideal_rows = [(position, full_logits(oracle[position]["logprobs"]), oracle[position]["target"]) for position in POSITIONS]
    wrong_rows = []
    for position in POSITIONS:
        source = position + 1 if position + 1 in oracle else position - 1
        wrong_rows.append((position, full_logits(oracle[source]["logprobs"]), oracle[position]["target"]))
    ideal = write_dump(os.path.join(root, "dump-ideal"), ideal_rows)
    wrong = write_dump(os.path.join(root, "dump-wrong"), wrong_rows)
    return {"device": os.path.join(root, "device"), "twin": os.path.join(root, "twin"), "ideal": ideal, "wrong": wrong, "tokens": tokens}


def run(command, log_path):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    return completed.returncode



class DryRunError(RuntimeError):
    pass


def accept_scenario(name, work, device, twin=None, dump=None, bars=None):
    out_json = os.path.join(work, "accept", name + ".json")
    out_md = os.path.join(work, "accept", name + ".md")
    log = os.path.join(work, "accept", name + ".log")
    argv = [sys.executable, ACCEPT, "--device", device, "--out-json", out_json, "--out-md", out_md,
            "--oracle", ORACLE, "--oracle-prompt", PROMPT, "--positions", POSITIONS_STR,
            "--expect-vocab", str(VOCAB)]
    if twin:
        argv += ["--twin", twin]
    if dump:
        argv += ["--device-dump", dump[0], "--device-bin", dump[1], "--dump-source", "DRY-RUN synthetic (not device)"]
    if bars:
        argv += ["--oracle-bars", bars]
    code = run(argv, log)
    doc = load(out_json) if os.path.exists(out_json) else None
    return {"name": name, "exit": code, "doc": doc, "out_json": out_json, "log": log}


def window_scenario(name, work, device, bars=None):
    out_dir = os.path.join(work, "window", name)
    log = os.path.join(work, "window", name + ".log")
    argv = [sys.executable, WINDOW, "--device-dir", device, "--oracle", ORACLE,
            "--oracle-prompt", PROMPT, "--positions", POSITIONS_STR, "--expected-steps", "3",
            "--expected-root", ROOT, "--expect-vocab", str(VOCAB), "--expect-attention-bound", "208",
            "--expect-moe-bounds", MOE, "--expect-manifest-lines", "185481", "--out-dir", out_dir]
    if bars:
        argv += ["--oracle-bars", bars]
    code = run(argv, log)
    doc_path = os.path.join(out_dir, "window-acceptance.json")
    doc = load(doc_path) if os.path.exists(doc_path) else None
    return {"name": name, "exit": code, "doc": doc, "log": log, "out_dir": out_dir}


def oracle_verdict(doc):
    return ((doc or {}).get("oracle_comparison") or {}).get("verdict")


def overall_verdict(doc):
    return ((doc or {}).get("overall") or {}).get("verdict")


def overall_label(doc):
    return ((doc or {}).get("overall") or {}).get("label")


def token_verdict(doc):
    doc = doc or {}
    if isinstance(doc.get("twin_match"), dict):
        return doc["twin_match"].get("verdict")
    component = (doc.get("components") or {}).get("accept_device_token") or {}
    return component.get("twin_match")


def log_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="D41 bars dry-run proof (proposal status; never a gate)")
    parser.add_argument("--proposal-dir", required=True, help="directory written by scripts/d41_bars_proposal.py")
    parser.add_argument("--out-dir", required=True, help="dry-run output directory (synthetic tree + logs)")
    parser.add_argument("--m1w-device", default=M1W_DEVICE, help="existing m1w refused device tree")
    args = parser.parse_args(argv)
    work = os.path.abspath(args.out_dir)
    os.makedirs(work, exist_ok=True)
    proposal_file = os.path.join(args.proposal_dir, "d41-bars-proposal.json")
    bars_ok = os.path.join(args.proposal_dir, "d41-bars-dryrun-ok.json")
    bars_impossible = os.path.join(args.proposal_dir, "d41-bars-dryrun-impossible.json")
    for path in (proposal_file, bars_ok, bars_impossible):
        if not os.path.exists(path):
            print("REFUSED: missing " + path, file=sys.stderr)
            return 2
    if load(proposal_file).get("status") != "proposal":
        print("REFUSED: " + proposal_file + " is not a status-proposal file", file=sys.stderr)
        return 2
    m1w_device = args.m1w_device
    try:
        oracle = oracle_rows()
        synthetic = build_synthetic(oracle, work)
    except (OSError, KeyError, ValueError) as error:
        print("REFUSED: " + str(error), file=sys.stderr)
        return 2

    scenarios = {}
    scenarios["refusal_no_bars"] = accept_scenario("refusal_no_bars", work, m1w_device)
    scenarios["refusal_with_bars"] = accept_scenario("refusal_with_bars", work, m1w_device, bars=bars_ok)
    scenarios["proposal_refused"] = accept_scenario("proposal_refused", work, synthetic["device"], twin=synthetic["twin"], dump=synthetic["ideal"], bars=proposal_file)
    scenarios["ideal_no_bars"] = accept_scenario("ideal_no_bars", work, synthetic["device"], twin=synthetic["twin"], dump=synthetic["ideal"])
    scenarios["ideal_ok_bars"] = accept_scenario("ideal_ok_bars", work, synthetic["device"], twin=synthetic["twin"], dump=synthetic["ideal"], bars=bars_ok)
    scenarios["ideal_impossible_bars"] = accept_scenario("ideal_impossible_bars", work, synthetic["device"], twin=synthetic["twin"], dump=synthetic["ideal"], bars=bars_impossible)
    scenarios["wrong_no_bars"] = accept_scenario("wrong_no_bars", work, synthetic["device"], twin=synthetic["twin"], dump=synthetic["wrong"])
    scenarios["wrong_ok_bars"] = accept_scenario("wrong_ok_bars", work, synthetic["device"], twin=synthetic["twin"], dump=synthetic["wrong"], bars=bars_ok)
    scenarios["window_refusal_no_bars"] = window_scenario("window_refusal_no_bars", work, m1w_device)
    scenarios["window_refusal_with_bars"] = window_scenario("window_refusal_with_bars", work, m1w_device, bars=bars_ok)

    checks = []

    def check(name, condition, detail):
        checks.append({"check": name, "pass": bool(condition), "detail": detail})
        return condition

    refusal = scenarios["refusal_no_bars"]
    refusal_bars = scenarios["refusal_with_bars"]
    check("m1w refusal stays NOT-ATTEMPTED", refusal["exit"] == 3 and overall_verdict(refusal["doc"]) in ("NOT_ATTEMPTED", "NOT-ATTEMPTED"), "exit %s overall %s" % (refusal["exit"], overall_verdict(refusal["doc"])))
    check("m1w refusal keeps token NOT-ATTEMPTED", token_verdict(refusal["doc"]) == "NOT_ATTEMPTED", "twin/token %s" % token_verdict(refusal["doc"]))
    check("bars cannot turn the refusal into a pass", refusal_bars["exit"] == 3 and overall_verdict(refusal_bars["doc"]) in ("NOT_ATTEMPTED", "NOT-ATTEMPTED") and token_verdict(refusal_bars["doc"]) == "NOT_ATTEMPTED", "exit %s overall %s token %s" % (refusal_bars["exit"], overall_verdict(refusal_bars["doc"]), token_verdict(refusal_bars["doc"])))
    check("bars leaves the refusal oracle leg UNMEASURED", oracle_verdict(refusal_bars["doc"]) == "UNMEASURED", "oracle %s" % oracle_verdict(refusal_bars["doc"]))
    check("refusal overall reasons unchanged by bars", ((refusal["doc"] or {}).get("overall") or {}).get("reasons") == ((refusal_bars["doc"] or {}).get("overall") or {}).get("reasons"), "reasons equal")
    proposal = scenarios["proposal_refused"]
    check("proposal status is refused by check_gate_bars", proposal["exit"] == 2 and proposal["doc"] is None and "status 'proposal'" in log_text(proposal["log"]), "exit %s log_has_status %s" % (proposal["exit"], "status 'proposal'" in log_text(proposal["log"])))

    ideal_no = scenarios["ideal_no_bars"]
    ideal_ok = scenarios["ideal_ok_bars"]
    ideal_impossible = scenarios["ideal_impossible_bars"]
    check("ideal synthetic device, no bars: oracle UNBARED, overall PASS", ideal_no["exit"] == 0 and overall_verdict(ideal_no["doc"]) == "PASS" and oracle_verdict(ideal_no["doc"]) == "UNBARED", "exit %s overall %s oracle %s" % (ideal_no["exit"], overall_verdict(ideal_no["doc"]), oracle_verdict(ideal_no["doc"])))
    check("ideal synthetic device, dry-run bars: oracle BARED PASS, overall PASS", ideal_ok["exit"] == 0 and overall_verdict(ideal_ok["doc"]) == "PASS" and oracle_verdict(ideal_ok["doc"]) == "PASS", "exit %s overall %s oracle %s" % (ideal_ok["exit"], overall_verdict(ideal_ok["doc"]), oracle_verdict(ideal_ok["doc"])))
    check("impossible bars fail a perfect synthetic device", ideal_impossible["exit"] == 1 and overall_verdict(ideal_impossible["doc"]) == "FAIL" and oracle_verdict(ideal_impossible["doc"]) == "FAIL", "exit %s overall %s oracle %s" % (ideal_impossible["exit"], overall_verdict(ideal_impossible["doc"]), oracle_verdict(ideal_impossible["doc"])))

    wrong_no = scenarios["wrong_no_bars"]
    wrong_ok = scenarios["wrong_ok_bars"]
    check("wrong dump without bars still passes (oracle UNBARED)", wrong_no["exit"] == 0 and overall_verdict(wrong_no["doc"]) == "PASS" and oracle_verdict(wrong_no["doc"]) == "UNBARED", "exit %s overall %s oracle %s" % (wrong_no["exit"], overall_verdict(wrong_no["doc"]), oracle_verdict(wrong_no["doc"])))
    check("wrong dump with bars turns only the oracle leg into FAIL", wrong_ok["exit"] == 1 and overall_verdict(wrong_ok["doc"]) == "FAIL" and oracle_verdict(wrong_ok["doc"]) == "FAIL", "exit %s overall %s oracle %s" % (wrong_ok["exit"], overall_verdict(wrong_ok["doc"]), oracle_verdict(wrong_ok["doc"])))
    same_shape = ((wrong_no["doc"] or {}).get("shape") or {}).get("verdict") == ((wrong_ok["doc"] or {}).get("shape") or {}).get("verdict")
    same_token = token_verdict(wrong_no["doc"]) == token_verdict(wrong_ok["doc"]) == "PASS"
    same_stream = ((wrong_no["doc"] or {}).get("device_stream") or {}).get("tokens") == ((wrong_ok["doc"] or {}).get("device_stream") or {}).get("tokens")
    check("bars changes neither shape nor token verdict", same_shape and same_token and same_stream, "shape_equal %s token %s stream_equal %s" % (same_shape, same_token, same_stream))

    window_no = scenarios["window_refusal_no_bars"]
    window_bars = scenarios["window_refusal_with_bars"]
    check("window m1w refusal unchanged by bars", window_no["exit"] == 3 and window_bars["exit"] == 3 and overall_verdict(window_no["doc"]) == overall_verdict(window_bars["doc"]) in ("NOT_ATTEMPTED", "NOT-ATTEMPTED"), "exit %s/%s overall %s/%s" % (window_no["exit"], window_bars["exit"], overall_verdict(window_no["doc"]), overall_verdict(window_bars["doc"])))
    check("window m1w refusal token NOT-ATTEMPTED with bars", token_verdict(window_bars["doc"]) in ("NOT_ATTEMPTED", "NOT-ATTEMPTED"), "token %s" % token_verdict(window_bars["doc"]))

    summary = {
        "generated_by": "scripts/d41_bars_dryrun.py",
        "proposal_dir": os.path.abspath(args.proposal_dir),
        "proposal_status": load(proposal_file).get("status"),
        "synthetic_tokens": synthetic["tokens"],
        "scenarios": {name: {"exit": item["exit"], "overall": overall_verdict(item["doc"]), "token": token_verdict(item["doc"]), "oracle": oracle_verdict(item["doc"])} for name, item in scenarios.items()},
        "checks": checks,
        "all_pass": all(check["pass"] for check in checks),
    }
    write_json(os.path.join(work, "dryrun-summary.json"), summary)
    lines = ["# D41 bars dry-run proof", "", "- proposal dir: " + os.path.abspath(args.proposal_dir), "- proposal status: " + str(summary["proposal_status"]), "- synthetic device token ids: " + str(synthetic["tokens"]), "", "## Scenarios", "", "| scenario | exit | overall | token | oracle |", "| :-- | ---: | :-- | :-- | :-- |"]
    for name, item in scenarios.items():
        lines.append("| %s | %s | %s | %s | %s |" % (name, item["exit"], overall_verdict(item["doc"]), token_verdict(item["doc"]), oracle_verdict(item["doc"])))
    lines += ["", "## Checks", ""]
    for check in checks:
        lines.append("- " + ("PASS" if check["pass"] else "FAIL") + " - " + check["check"] + ": " + check["detail"])
    lines += ["", "The synthetic tree and dump are DRY-RUN labels, not device evidence; the proposal file is status 'proposal' and was refused (exit 2) as designed.", ""]
    with open(os.path.join(work, "DRYRUN.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("dry-run summary: " + os.path.join(work, "dryrun-summary.json"))
    for check in checks:
        print(("PASS" if check["pass"] else "FAIL") + " - " + check["check"] + ": " + check["detail"])
    if not summary["all_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

