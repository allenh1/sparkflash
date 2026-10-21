#!/usr/bin/env python3
"""D41 bars proposal (task-12): evidence-derived floors, written BEFORE any device token exists.

The output is status "proposal": compare.check_gate_bars refuses status != "ok", so nothing here
can be installed as a gate. A same-values status-ok dry-run fixture and an impossible fixture are
also written for the mechanics tests. Exit 0 = proposal written; 3 = at least one proposed bar is
looser than the D40 provisional (Lead review required); 2 = refused input.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import compare_native_vs_vllm as compare
import d41_floors as floors

POSITIONS = list(range(129, 138))
TOP_K = (20, 100, 1000)
FLAT = 0.05
PROMPT_ID = "b69-fixture-130"
DEFAULTS = {
    "oracle": "/home/user/ev-session-20261020/qa/artifacts/native-oracle-suite.json",
    "repeat": "/home/user/phase2-evidence/window-2026-10-18/native-oracle-130.json",
    "capture": "/home/user/ev-session-20261020/qa/artifacts/d41-capture-20261020T012706Z.json",
    "o4": "/home/user/ev-session-20261020/qa/o4/compare-native-vs-vllm.json",
}
PROVISIONAL = {
    "positions_min": 9,
    "top1_agreement_min": 0.95,
    "top20_overlap_mean_min": 0.90,
    "top100_overlap_mean_min": 0.80,
    "top1000_overlap_mean_min": 0.60,
    "max_abs_delta_max": 1.0,
    "mean_abs_delta_max": 0.10,
    "kl_mean_max": 0.02,
    "kl_p95_max": 0.10,
    "js_mean_max": 0.05,
}
CEIL = {"max_abs_delta_max": "abs_max", "mean_abs_delta_max": "abs_mean", "kl_mean_max": "kl_mean", "kl_p95_max": "kl_p95", "js_mean_max": "js_mean"}
FLOOR = {"top20_overlap_mean_min": "top20_mean", "top100_overlap_mean_min": "top100_mean", "top1000_overlap_mean_min": "top1000_mean"}


class ProposalError(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    if not os.path.exists(path):
        raise ProposalError("missing input: " + path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def oracle_prompt(doc, label):
    if doc.get("schema") != "sparkflash-native-oracle/1" or doc.get("status") != "ok":
        raise ProposalError(label + ": not an ok sparkflash-native-oracle/1 document")
    found = [item for item in doc.get("prompts") or [] if item.get("id") == PROMPT_ID]
    if len(found) != 1:
        raise ProposalError(label + ": expected exactly one prompt " + PROMPT_ID)
    return found[0]


def rows_of(prompt):
    rows = {}
    for item in prompt.get("positions") or []:
        position = int(item["position"])
        logprobs = {int(token): float(value) for token, value in item["logprobs"].items()}
        rows[position] = {"logprobs": logprobs, "ranked": sorted(logprobs.items(), key=lambda kv: (-kv[1], kv[0]))}
    return rows


def capture_rows(capture, request_id):
    for block in capture.get("blocks") or []:
        for request in block.get("requests") or []:
            if request.get("id") == request_id:
                rows = {}
                for item in request["positions"]:
                    position = int(item["position"])
                    logprobs = {int(token): float(value) for token, value in item["top"]}
                    rows[position] = {"logprobs": logprobs, "ranked": sorted(logprobs.items(), key=lambda kv: (-kv[1], kv[0]))}
                return rows, int(request["prompt_len"])
    raise ProposalError("capture has no request " + request_id)


def record(identifier, prompt_len, rows):
    return {"id": identifier, "prompt_len": prompt_len, "positions": rows}


def pair(left, right, where):
    return floors._pair_metrics(left, right, where, FLAT)


def aggregate(metrics):
    result = {
        "positions": metrics["positions"],
        "position_first": metrics["position_first"],
        "position_last": metrics["position_last"],
        "top1_agreement_rate": metrics["top1_agreement"],
        "max_abs_delta_max": metrics["abs_max"],
        "mean_abs_delta_mean": metrics["abs_mean"],
        "kl_mean": metrics["kl_mean"],
        "kl_p95": metrics["kl_p95"],
        "js_mean": metrics["js_mean"],
    }
    for k in TOP_K:
        result["top%d_overlap_mean" % k] = metrics["top%d_mean" % k]
    return result


def suite_repeat(a, b):
    if [p["id"] for p in a["prompts"]] != [p["id"] for p in b["prompts"]]:
        raise ProposalError("the two oracle runs do not contain the same prompt suite")
    prompts, positions, stable, flips = [], 0, 0, 0
    for pa in a["prompts"]:
        pb = [p for p in b["prompts"] if p["id"] == pa["id"]][0]
        if int(pa["prompt_len"]) != int(pb["prompt_len"]):
            raise ProposalError(pa["id"] + ": prompt_len differs between oracle runs")
        if [r["position"] for r in pa["positions"]] != [r["position"] for r in pb["positions"]]:
            raise ProposalError(pa["id"] + ": position tables differ between oracle runs")
        if [r.get("target_token") for r in pa["positions"]] != [r.get("target_token") for r in pb["positions"]]:
            raise ProposalError(pa["id"] + ": target tokens differ between oracle runs")
        metrics = pair(record(pa["id"] + "/a", pa["prompt_len"], rows_of(pa)), record(pa["id"] + "/b", pb["prompt_len"], rows_of(pb)), "suite:" + pa["id"])
        prompts.append({"prompt_id": pa["id"], "metrics": metrics})
        positions += metrics["positions"]
        stable += metrics["rank1_stable"]
        flips += metrics["rank1_flips_above_flat"]
    return {
        "positions": positions,
        "rank1_stable": stable,
        "rank1_flips_above_flat": flips,
        "strict_top1_rate": stable / positions,
        "flip_above_flat_rate": flips / positions,
        "prompts": prompts,
    }


def derive(metrics, suite, crosscheck):
    proposal, evidence = {}, {}
    for name, field in CEIL.items():
        floor = max(metrics[field], crosscheck[field])
        proposal[name] = max(PROVISIONAL[name], 3.0 * floor)
        evidence[name] = {"measured_reference_floor": floor, "reference_repeat_floor": metrics[field], "capture_crosscheck_floor": crosscheck[field], "rule": "max(D40 provisional, 3 x worst measured reference floor)", "proposed": proposal[name]}
    for name, field in FLOOR.items():
        floor = min(metrics[field], crosscheck[field])
        proposal[name] = min(PROVISIONAL[name], 1.0 - 3.0 * (1.0 - floor))
        evidence[name] = {"measured_reference_floor": floor, "reference_repeat_floor": metrics[field], "capture_crosscheck_floor": crosscheck[field], "rule": "min(D40 provisional, 1 - 3 x (1 - worst measured reference floor))", "proposed": proposal[name]}
    floor = min(PROVISIONAL["top1_agreement_min"], 1.0 - 3.0 * suite["flip_above_flat_rate"])
    proposal["top1_agreement_min"] = floor
    evidence["top1_agreement_min"] = {
        "measured_reference_floor_at_positions": metrics["top1_agreement"],
        "suite_wide_strict_top1_rate": suite["strict_top1_rate"],
        "suite_wide_flip_above_flat_rate": suite["flip_above_flat_rate"],
        "rule": "min(D40 provisional, 1 - 3 x suite-wide rank-1 flip rate)",
        "proposed": floor,
    }
    proposal["positions_min"] = PROVISIONAL["positions_min"]
    evidence["positions_min"] = {"rule": "pinned comparison set", "proposed": 9}
    return proposal, evidence


def looser(proposal):
    flags = {}
    for name, prop in proposal.items():
        base = PROVISIONAL.get(name)
        if base is None:
            continue
        looser_now = prop < base if name.endswith("_min") else prop > base
        if looser_now:
            flags[name] = {"proposed": prop, "d40_provisional": base}
    return flags


def evaluate(bars, agg):
    return compare.evaluate_bars({"bars": bars}, agg)


def build(oracle, oracle_repeat, capture, o4, paths):
    pa = oracle_prompt(oracle, "canonical oracle")
    pb = oracle_prompt(oracle_repeat, "oracle repeat")
    if int(pa["prompt_len"]) != int(pb["prompt_len"]):
        raise ProposalError("the two b69 prompts have different prompt_len; not a like-for-like repeat")
    suite = suite_repeat(oracle, oracle_repeat)
    repeat = pair(record("run-a", pa["prompt_len"], {p: v for p, v in rows_of(pa).items() if p in POSITIONS}),
                  record("run-b", pb["prompt_len"], {p: v for p, v in rows_of(pb).items() if p in POSITIONS}),
                  "oracle run A vs run B, positions 129..137")
    cap_rows, cap_len = capture_rows(capture, "len-139")
    cross = pair(record("capture/len-139", cap_len, {p: v for p, v in cap_rows.items() if p in POSITIONS}),
                 record("oracle/suite", pa["prompt_len"], {p: v for p, v in rows_of(pa).items() if p in POSITIONS}),
                 "capture len-139 vs canonical oracle suite, positions 129..137")
    repeats = []
    block = [b for b in capture.get("blocks") or [] if b.get("class") == "repeat"]
    if not block:
        raise ProposalError("capture has no repeat block")
    requests = block[0].get("requests") or []
    if len(requests) < 2:
        raise ProposalError("capture repeat block needs at least two requests")
    for first, second in zip(requests, requests[1:]):
        left = record(first["id"], int(first["prompt_len"]), capture_rows(capture, first["id"])[0])
        right = record(second["id"], int(second["prompt_len"]), capture_rows(capture, second["id"])[0])
        metrics = pair(left, right, "same-request repeat " + first["id"] + " vs " + second["id"])
        repeats.append({"pair": [first["id"], second["id"]], "metrics": metrics})
        if metrics["kl_max"] != 0.0 or metrics["abs_max"] != 0.0 or metrics["js_mean"] != 0.0:
            raise ProposalError("same-request repeat is not bit-identical at " + first["id"])
        for k in TOP_K:
            if metrics["top%d_min" % k] != 1.0:
                raise ProposalError("same-request top-%d overlap is not 1.0" % k)

    all_masses, at_masses, margins, flat = [], [], [], []
    for row in pa["positions"]:
        probs = [math.exp(float(v)) for v in row["logprobs"].values()]
        all_masses.append(sum(probs))
        if int(row["position"]) in POSITIONS:
            at_masses.append(sum(probs))
            ranked = sorted(((int(t), float(v)) for t, v in row["logprobs"].items()), key=lambda kv: (-kv[1], kv[0]))
            top1 = math.exp(ranked[0][1])
            margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
            margins.append({"position": int(row["position"]), "top1_prob": top1, "top1_margin_nats": margin, "top1000_mass": sum(probs)})
            if top1 < FLAT:
                flat.append(int(row["position"]))

    proposal, evidence = derive(repeat, suite, cross)
    flags = looser(proposal)
    agg_repeat, agg_cross = aggregate(repeat), aggregate(cross)
    pre_fix = o4.get("aggregate") if isinstance(o4, dict) else None
    evaluations = {
        "proposed_on_reference_repeat": evaluate(proposal, agg_repeat),
        "proposed_on_capture_crosscheck": evaluate(proposal, agg_cross),
        "d40_provisional_on_reference_repeat": evaluate(PROVISIONAL, agg_repeat),
        "d40_provisional_on_capture_crosscheck": evaluate(PROVISIONAL, agg_cross),
    }
    if pre_fix:
        evaluations["proposed_on_pre_fix_b70_dump"] = evaluate(proposal, pre_fix)
        evaluations["d40_provisional_on_pre_fix_b70_dump"] = evaluate(PROVISIONAL, pre_fix)

    provenance = {
        "oracle_suite": {"path": os.path.abspath(paths["oracle"]), "sha256": sha(paths["oracle"]), "prompt_id": PROMPT_ID, "served_root": oracle.get("served", {}).get("root")},
        "oracle_repeat_run": {"path": os.path.abspath(paths["repeat"]), "sha256": sha(paths["repeat"]), "generated_at": oracle_repeat.get("generated_at")},
        "capture": {"path": os.path.abspath(paths["capture"]), "sha256": sha(paths["capture"]), "capture_id": capture.get("capture_id")},
        "pre_fix_b70_comparison": {"path": os.path.abspath(paths["o4"]), "sha256": sha(paths["o4"])} if pre_fix else None,
        "proposal_script_sha256": sha(os.path.abspath(__file__)),
        "provisional_source": "docs/DECISIONS.md D40 (quoted, never changed here)",
        "flat_prob": FLAT,
    }
    proposal_doc = {
        "schema": "sparkflash-d41-bars/1",
        "status": "proposal",
        "bar_class": "idle_distractor",
        "label": "D41-BARS-PROPOSAL (evidence-derived; NOT a gate; Lead declaration required)",
        "generated_utc": utc(),
        "provenance": provenance,
        "comparison_set": {
            "positions": POSITIONS,
            "prompt_id": PROMPT_ID,
            "top_k": list(TOP_K),
            "support": "oracle top-1000; KL/JS renormalised over the compared support; abs deltas are raw oracle-vs-device logprobs",
            "flat_prob": FLAT,
            "metrics": ["top1 agreement", "top-k overlap", "max/mean abs logprob delta", "KL(oracle||device)", "JS symmetric"],
        },
        "bars": proposal,
        "derivation": {
            "rule": "ceilings max(provisional, 3x worst measured reference floor); floors min(provisional, 1-3(1-measured)); top1 from suite-wide flip rate",
            "bars": evidence,
            "measured_reference_repeat": repeat,
            "measured_capture_crosscheck": cross,
            "measured_same_request_repeat": repeats,
            "looser_than_d40_provisional": flags,
        },
        "evaluation": evaluations,
        "requires_lead_declaration": True,
        "not_a_gate": "status is 'proposal'; compare.check_gate_bars refuses any status != 'ok'",
        "non_claims": [
            "not a D41 capture floor: no capture has all three classes; this derivation class is reference-repeat (vLLM-vs-vLLM)",
            "not installed: a Lead-declared status-ok file is still required",
            "no device token measured: no numeric device-vs-oracle claim",
            "comparison is on the oracle top-1000 support only; retained mass at these positions is recorded",
        ],
    }
    return {"proposal": proposal_doc, "bars": proposal, "evidence": evidence, "flags": flags, "suite": suite, "repeat": repeat, "cross": cross, "repeats": repeats, "aggregate_repeat": agg_repeat, "aggregate_cross": agg_cross, "evaluations": evaluations, "support": {"at_positions_top1000_mass": at_masses, "suite_top1000_mass_mean": sum(all_masses) / len(all_masses), "suite_top1000_mass_min": min(all_masses), "flat_positions": flat, "position_margins": margins}}


def write(result, outdir):
    os.makedirs(outdir, exist_ok=True)
    proposal_path = os.path.join(outdir, "d41-bars-proposal.json")
    with open(proposal_path, "w", encoding="utf-8") as handle:
        json.dump(result["proposal"], handle, indent=1)
        handle.write("\n")
    fixture = json.loads(json.dumps(result["proposal"]))
    fixture["status"] = "ok"
    fixture["label"] = "D41-BARS-DRYRUN-FIXTURE (mechanics only; same values as the proposal; NOT declared by the Lead)"
    fixture["dry_run_fixture"] = True
    fixture_path = os.path.join(outdir, "d41-bars-dryrun-ok.json")
    with open(fixture_path, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=1)
        handle.write("\n")
    impossible = json.loads(json.dumps(fixture))
    impossible["label"] = "D41-BARS-DRYRUN-IMPOSSIBLE (mechanics: must fail a perfect synthetic device)"
    impossible["bars"] = {"positions_min": 9, "top1_agreement_min": 1.5, "kl_mean_max": 0.0}
    impossible_path = os.path.join(outdir, "d41-bars-dryrun-impossible.json")
    with open(impossible_path, "w", encoding="utf-8") as handle:
        json.dump(impossible, handle, indent=1)
        handle.write("\n")
    evidence = {
        "generated_utc": utc(),
        "reference_repeat_at_positions": result["repeat"],
        "capture_crosscheck_at_positions": result["cross"],
        "same_request_repeat": result["repeats"],
        "oracle_suite_repeatability": result["suite"],
        "support": result["support"],
        "bars_derivation": result["evidence"],
        "looser_than_d40_provisional": result["flags"],
        "evaluation": result["evaluations"],
        "proposal_path": os.path.abspath(proposal_path),
        "dryrun_fixture_path": os.path.abspath(fixture_path),
        "impossible_fixture_path": os.path.abspath(impossible_path),
    }
    with open(os.path.join(outdir, "reference-repeat-evidence.json"), "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=1)
        handle.write("\n")
    lines = ["# D41 bars proposal - evidence summary", "", "generated: " + evidence["generated_utc"], "", "## Measured vLLM-vs-vLLM repeatability at positions 129..137", "", "| metric | two oracle runs | third run (capture) |", "| :-- | ---: | ---: |"]
    for label, key in (("positions", "positions"), ("top1 agreement", "top1_agreement"), ("top20 overlap", "top20_mean"), ("top100 overlap", "top100_mean"), ("top1000 overlap", "top1000_mean"), ("max abs delta", "abs_max"), ("mean abs delta", "abs_mean"), ("KL mean", "kl_mean"), ("KL p95", "kl_p95"), ("JS mean", "js_mean")):
        lines.append("| %s | %.9g | %.9g |" % (label, result["repeat"][key], result["cross"][key]))
    lines += ["", "## Proposed bars", "", "| bar | proposed | D40 provisional | measured floor |", "| :-- | ---: | ---: | ---: |"]
    for name in sorted(result["bars"]):
        item = result["evidence"].get(name, {})
        lines.append("| %s | %.9g | %s | %.9g |" % (name, result["bars"][name], PROVISIONAL.get(name, "-"), item.get("measured_reference_floor", 0)))
    lines += ["", "## Evaluation", ""]
    for name, outcome in result["evaluations"].items():
        lines.append("- " + name + ": " + ("PASS" if outcome["pass"] else "FAIL"))
        for bar in outcome["bars"]:
            if not bar["pass"]:
                lines.append("  - FAIL " + bar["bar"] + ": value %.9g vs threshold %.9g" % (bar["value"], bar["threshold"]))
    if result["flags"]:
        lines += ["", "## Requires Lead review (looser than the D40 provisional)", ""]
        for name, flag in sorted(result["flags"].items()):
            lines.append("- %s: %.9g vs provisional %.9g" % (name, flag["proposed"], flag["d40_provisional"]))
    lines += ["", "No device token was measured. Nothing here is a device-vs-oracle numeric claim."]
    with open(os.path.join(outdir, "REFERENCE-REPEAT-EVIDENCE.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return proposal_path, fixture_path, impossible_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="D41 bars proposal (status proposal; never a gate)")
    parser.add_argument("--oracle", default=DEFAULTS["oracle"])
    parser.add_argument("--oracle-repeat", default=DEFAULTS["repeat"])
    parser.add_argument("--capture", default=DEFAULTS["capture"])
    parser.add_argument("--o4", default=DEFAULTS["o4"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    paths = {"oracle": args.oracle, "repeat": args.oracle_repeat, "capture": args.capture, "o4": args.o4}
    try:
        oracle = load(args.oracle)
        oracle_repeat = load(args.oracle_repeat)
        capture = load(args.capture)
        if capture.get("schema") != "sparkflash-d41-capture/1":
            raise ProposalError("capture: not a sparkflash-d41-capture/1 document")
        o4 = load(args.o4) if os.path.exists(args.o4) else {}
        result = build(oracle, oracle_repeat, capture, o4, paths)
    except ProposalError as error:
        print("REFUSED: " + str(error), file=sys.stderr)
        return 2
    proposal_path, fixture_path, impossible_path = write(result, args.out_dir)
    print("proposal: " + proposal_path)
    print("dry-run fixture: " + fixture_path)
    print("impossible fixture: " + impossible_path)
    print("measured reference repeat: top1=%.4f kl_mean=%.6f top20=%.4f" % (result["repeat"]["top1_agreement"], result["repeat"]["kl_mean"], result["repeat"]["top20_mean"]))
    print("proposed bars: " + json.dumps(result["bars"], sort_keys=True))
    if result["flags"]:
        print("REQUIRES LEAD REVIEW: %d proposed bar(s) are looser than the D40 provisional" % len(result["flags"]))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

