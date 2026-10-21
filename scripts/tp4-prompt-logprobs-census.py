#!/usr/bin/env python3
"""TP=4 all-position prompt-logprobs rank contract census (host path, UNBARED).

Runs the two rank-contract probes in the `sparkflash-cli` unit-test harness -- the four-rank
merge-exactness comparison against the serial host path and the pre-collective bounded-budget
refusal with recovery -- and writes one JSON and one markdown artefact.

The probe binary is the crate's lib test executable: the twin fixture and the four loopback ranks
need no GPU, checkpoint or server socket, and the whole measurement is host-side. Nothing here is
a throughput, device or vLLM claim; the wall numbers are repeatability evidence for the same
synthetic plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sparkflash-tp4-prompt-logprobs-census/1"
PARITY_ENV = "SPARKFLASH_TP4_PROMPT_LOGPROBS_PROBE"
REFUSAL_ENV = "SPARKFLASH_TP4_PROMPT_LOGPROBS_REFUSAL_PROBE"
PARITY_TEST = (
    "tp4_host::tests::the_four_rank_prompt_logprob_contract_matches_the_serial_host_path"
)
REFUSAL_TEST = (
    "tp4_host::tests::the_four_rank_contract_refuses_an_over_wide_request_before_any_collective"
)
FEATURES = ["tp4-serve"]
SCOPE = "host path, synthetic tiny plan (40-layer twin fixture, vocab 128), UNBARED"
SOURCE_MANIFESTS = [
    "Cargo.lock",
    "crates/sparkflash-api/src/engine.rs",
    "crates/sparkflash-cli/src/serve.rs",
    "crates/sparkflash-cli/src/tp4_host.rs",
    "crates/sparkflash-engine/src/forward/mod.rs",
    "crates/sparkflash-engine/src/prompt_ranks.rs",
]


def sha256_file(path: Path) -> str:
    """The sha256 of one file, read in chunks so a probe binary does not sit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_test_binary(repo: Path, cargo: str = "cargo") -> tuple[Path, float]:
    """Builds the lib test binary and returns its path and the build's wall milliseconds."""
    started = time.monotonic()
    command = [
        cargo,
        "test",
        "-p",
        "sparkflash-cli",
        "--features",
        FEATURES[0],
        "--lib",
        "--no-run",
        "--message-format=json",
    ]
    completed = subprocess.run(
        command, cwd=repo, capture_output=True, text=True, timeout=1800
    )
    build_ms = round((time.monotonic() - started) * 1000.0)
    if completed.returncode != 0:
        raise SystemExit(
            "the probe binary did not build:\n"
            f"command: {' '.join(command)}\nstdout tail:\n{completed.stdout[-4000:]}\n"
            f"stderr tail:\n{completed.stderr[-4000:]}"
        )
    executables: list[str] = []
    for line in completed.stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        executable = message.get("executable")
        if executable and message.get("target", {}).get("name") == "sparkflash_cli":
            executables.append(executable)
    if not executables:
        raise SystemExit("cargo reported no sparkflash_cli test executable")
    return Path(executables[-1]), build_ms


def run_probe(
    binary: Path,
    test: str,
    env_name: str,
    report_path: Path,
    repo: Path,
) -> tuple[dict, int, str]:
    """Runs one probe test and returns its JSON report, wall milliseconds and output tail."""
    environment = os.environ.copy()
    environment[env_name] = str(report_path)
    started = time.monotonic()
    completed = subprocess.run(
        [str(binary), "--exact", test, "--nocapture", "--test-threads=1"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed_ms = round((time.monotonic() - started) * 1000.0)
    output = (completed.stdout + completed.stderr).strip().splitlines()
    tail = "\n".join(output[-20:])
    if completed.returncode != 0 or not report_path.is_file():
        raise SystemExit(
            f"probe {test} failed (exit {completed.returncode}):\n{tail}"
        )
    return json.loads(report_path.read_text(encoding="utf-8")), elapsed_ms, tail


def summarize(
    parity_runs: list[dict],
    refusal_runs: list[dict],
    parity_ms: list[int],
    refusal_ms: list[int],
) -> dict:
    """Folds the per-run probes into the census accounting and repeatability blocks."""
    rounds = [
        round_entry
        for document in parity_runs
        for round_entry in document.get("rounds", [])
    ]
    wall_ms = [int(entry.get("wall_ms", 0)) for entry in rounds]
    fixtures = {
        (document.get("fixture_tensors"), document.get("fixture_bytes"), document.get("fixture_hash"))
        for document in parity_runs + refusal_runs
    }
    if len(fixtures) != 1:
        raise SystemExit(f"the probe runs used different fixtures: {sorted(fixtures)}")
    fixture_tensors, fixture_bytes, fixture_hash = fixtures.pop()
    refusal_texts = {document.get("refusal") for document in refusal_runs}
    return {
        "runs": len(parity_runs),
        "rounds": len(rounds),
        "scored_positions_total": sum(
            int(entry.get("scored_positions", 0)) for entry in rounds
        ),
        "alternatives_per_position": sorted(
            {int(entry.get("alternatives_per_position", 0)) for entry in rounds}
        ),
        "ranks_agree_bitwise": all(
            bool(entry.get("ranks_agree_bitwise")) for entry in rounds
        ),
        "serial_order_equal": all(
            bool(entry.get("serial_order_equal")) for entry in rounds
        ),
        "max_abs_logprob_delta": max(
            (float(entry.get("max_abs_logprob_delta", 0.0)) for entry in rounds),
            default=0.0,
        ),
        "round_wall_ms": wall_ms,
        "round_wall_ms_median": statistics.median(wall_ms) if wall_ms else 0.0,
        "round_wall_ms_min": min(wall_ms) if wall_ms else 0,
        "round_wall_ms_max": max(wall_ms) if wall_ms else 0,
        "probe_process_ms": parity_ms,
        "refusal_probe_process_ms": refusal_ms,
        "refusal_requests": len(refusal_runs),
        "refusal_rank_count": max(
            (int(document.get("world", 0)) for document in refusal_runs), default=0
        ),
        "refusal_agreed": all(
            bool(document.get("ranks_agree_on_message")) for document in refusal_runs
        ),
        "recovered_after_refusal": all(
            bool(document.get("recovered_after_refusal")) for document in refusal_runs
        ),
        "refusal_texts": sorted(text for text in refusal_texts if text),
        "fixture_tensors": int(fixture_tensors),
        "fixture_bytes": int(fixture_bytes),
        "fixture_hash": int(fixture_hash),
    }


def render_markdown(census: dict) -> str:
    """The human-readable census: scope, criterion, accounting, repeatability, non-claims."""
    account = census["accounting"]
    lines = [
        "# TP=4 prompt-logprobs rank contract census",
        "",
        f"- schema: `{census['schema']}`",
        f"- generated: {census['generated_at']}",
        f"- git: {census['git']['commit']} on {census['git']['branch']}",
        f"- scope: {census['scope']}",
        f"- probe binary sha256: `{census['probe_binary']['sha256']}`",
        f"- fixture writer hash: {account['fixture_hash']} "
        f"({account['fixture_tensors']} tensors, {account['fixture_bytes']} bytes)",
        "",
        "## Criterion",
        "",
        "Each rank screens its local LM-head shard and exchanges bounded candidate blocks over the",
        "same TP=4 collectives the forward uses; the deterministic host merge reports a position",
        "only when the f32 logprob boundary proves the requested order (boundary strictly above the",
        "omitted mass and separated from the next union value by more than the representability",
        "margin). A rank entry holding a certified upper bound, an unprovable order, a malformed",
        "shard layout, a missing owning token or no distribution is a typed refusal, never an",
        "approximation.",
        "",
        "## Accounting",
        "",
        f"- runs: {account['runs']} ({account['rounds']} scored rounds)",
        f"- scored positions: {account['scored_positions_total']} "
        f"({account['alternatives_per_position']} alternatives per position)",
        f"- every rank's merged payload bit-identical: {account['ranks_agree_bitwise']}",
        f"- serial host-path top order equal: {account['serial_order_equal']}",
        f"- max |logprob delta| against the serial host path: {account['max_abs_logprob_delta']:.3e}",
        f"- refusal requests: {account['refusal_requests']} x {account['refusal_rank_count']} ranks; "
        f"same typed message on every rank: {account['refusal_agreed']}; group recovered: "
        f"{account['recovered_after_refusal']}",
        "",
        "## Repeatability (same synthetic plan, UNBARED)",
        "",
        f"- one round wall ms: {account['round_wall_ms']} "
        f"(median {account['round_wall_ms_median']}, min {account['round_wall_ms_min']}, "
        f"max {account['round_wall_ms_max']})",
        f"- probe process ms per run: {account['probe_process_ms']}",
        f"- refusal probe process ms per run: {account['refusal_probe_process_ms']}",
        "",
        "Wall numbers are repeatability of the same host run, not a throughput or speedup claim.",
        "",
        "## Non-claims",
        "",
        "- Host path only: no CUDA, no device kernel, no GPU, no vLLM comparison run.",
        "- The model is the synthetic 40-layer tiny twin fixture (vocab 128), not the released",
        "  checkpoint; the released vocabulary path is exercised by the same code but not measured",
        "  here.",
        "- Prompt logprobs over HTTP on the TP=4 path are exercised at the rank/control seam with a",
        "  stub rank, not against a real four-process serve deployment (the serve CLI opens the",
        "  released checkpoint; the twin fixture is the CLI probe's source mode).",
        "- No claim that the bounded top-k contract covers vLLM's whole-vocabulary -1 selection; that",
        "  selection is refused on this path and stays refused.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/user/ev-session-20261020/serve"),
        help="where the JSON and markdown artefacts are written",
    )
    parser.add_argument("--runs", type=int, default=3, help="probe repetitions (default 3)")
    parser.add_argument(
        "--cargo", default="cargo", help="cargo executable (default: cargo)"
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    repo = Path(__file__).resolve().parent.parent
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"tp4-prompt-logprobs-census-{stamp}"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    binary, build_ms = resolve_test_binary(repo, args.cargo)
    parity_runs: list[dict] = []
    refusal_runs: list[dict] = []
    parity_ms: list[int] = []
    refusal_ms: list[int] = []
    with tempfile.TemporaryDirectory(prefix="tp4-prompt-logprobs-probe-") as scratch:
        scratch_path = Path(scratch)
        for run in range(args.runs):
            report, elapsed, _ = run_probe(
                binary,
                PARITY_TEST,
                PARITY_ENV,
                scratch_path / f"parity-{run}.json",
                repo,
            )
            parity_runs.append(report)
            parity_ms.append(elapsed)
            report, elapsed, _ = run_probe(
                binary,
                REFUSAL_TEST,
                REFUSAL_ENV,
                scratch_path / f"refusal-{run}.json",
                repo,
            )
            refusal_runs.append(report)
            refusal_ms.append(elapsed)
    accounting = summarize(parity_runs, refusal_runs, parity_ms, refusal_ms)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    git_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cargo_version = subprocess.run(
        [args.cargo, "--version"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    rustc_version = subprocess.run(
        ["rustc", "--version"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    census = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": SCOPE,
        "label": "UNBARED host-path diagnostic",
        "git": {"commit": git_commit, "branch": git_branch},
        "probe_binary": {
            "path": str(binary),
            "sha256": sha256_file(binary),
            "bytes": binary.stat().st_size,
            "build_ms": build_ms,
        },
        "toolchain": {"cargo": cargo_version, "rustc": rustc_version},
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "loadavg": list(load),
        },
        "fixture": {
            "tensors": accounting["fixture_tensors"],
            "bytes": accounting["fixture_bytes"],
            "writer_hash": accounting["fixture_hash"],
            "description": "deterministic 40-layer tiny twin safetensors fixture, vocab 128",
        },
        "accounting": accounting,
        "source_manifests": [
            {"path": relative, "sha256": sha256_file(repo / relative)}
            for relative in SOURCE_MANIFESTS
            if (repo / relative).is_file()
        ],
        "raw_runs": {"parity": parity_runs, "refusal": refusal_runs},
        "non_claims": [
            "host path only: no CUDA, no device kernel, no GPU, no vLLM run",
            "synthetic 40-layer tiny twin fixture (vocab 128), not the released checkpoint",
            "rank/control seam with a stub rank, not a four-process serve deployment",
            "wall numbers are repeatability, not a throughput or speedup claim",
            "the bounded top-k contract does not cover whole-vocabulary -1; that stays refused",
        ],
    }
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"
    json_path.write_text(
        json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(census), encoding="utf-8")
    manifest_path = out_dir / f"{base}.sha256"
    manifest = {
        "json": sha256_file(json_path),
        "md": sha256_file(md_path),
        "probe_binary": census["probe_binary"]["sha256"],
        "source_manifests": {
            entry["path"]: entry["sha256"] for entry in census["source_manifests"]
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"json: {json_path} sha256={manifest['json']}")
    print(f"md: {md_path} sha256={manifest['md']}")
    print(f"probe binary: {binary} sha256={manifest['probe_binary']}")
    print(
        "accounting: "
        f"{accounting['rounds']} rounds, {accounting['scored_positions_total']} scored positions, "
        f"order_equal={accounting['serial_order_equal']}, "
        f"ranks_agree={accounting['ranks_agree_bitwise']}, "
        f"max_delta={accounting['max_abs_logprob_delta']:.3e}, "
        f"refused={accounting['refusal_requests']}x{accounting['refusal_rank_count']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
