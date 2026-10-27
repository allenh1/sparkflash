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

"""Streaming tool-call (V4.1 DSML) wire census (host path, UNBARED).

Runs the scripted-engine wire-parity probe in the sparkflash-cli serve_http integration tests:
the real serve binary and router, driven over a loopback socket by a debug-build mock engine, so
the streaming DSML parser is observed as an SSE client sees it. Each probe compares the streamed
OpenAI tool_calls array with the buffered one, counts the content/call-start/argument deltas, and
records the keep-alive comment interleave. One JSON and one markdown artefact are written.

Nothing here is a throughput, device, model-quality or vLLM claim. The wall numbers are
repeatability evidence for the same scripted stub on the host path.
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

SCHEMA = "sparkflash-tool-call-streaming-census/1"
PROBE_SCHEMA = "sparkflash-tool-stream-wire-probe/1"
PROBE_ENV = "SPARKFLASH_TOOL_STREAM_PROBE"
PROBE_TEST = "scripted_wire_streaming_dsml_tool_calls_match_the_buffered_body"
TEST_TARGET = "serve_http"
PACKAGE = "sparkflash-cli"
FEATURES = []
SCOPE = (
    "host path, real serve binary + axum router over loopback, debug-build scripted stub engine "
    "(V4.1 DSML answer split at arbitrary character boundaries), UNBARED"
)
SOURCE_MANIFESTS = [
    "Cargo.lock",
    "crates/sparkflash-api/src/tool_parser.rs",
    "crates/sparkflash-api/src/server.rs",
    "crates/sparkflash-api/src/state.rs",
    "crates/sparkflash-api/tests/router.rs",
    "crates/sparkflash-cli/src/serve.rs",
    "crates/sparkflash-cli/tests/serve_http.rs",
    "scripts/tool-call-streaming-census.py",
]


def sha256_file(path: Path) -> str:
    """The sha256 of one file, read in chunks so a probe binary does not sit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    """The sha256 of a canonical JSON rendering of a value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_test_binary(repo: Path, cargo: str = "cargo") -> tuple[Path, int]:
    """Builds the serve_http test binary and returns its path and the build wall milliseconds."""
    started = time.monotonic()
    command = [
        cargo,
        "test",
        "-p",
        PACKAGE,
        "--test",
        TEST_TARGET,
        "--no-run",
        "--message-format=json",
    ]
    completed = subprocess.run(
        command, cwd=repo, capture_output=True, text=True, timeout=1800
    )
    build_ms = round((time.monotonic() - started) * 1000.0)
    if completed.returncode != 0:
        raise SystemExit(
            "the wire probe binary did not build:\n"
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
        if executable and message.get("target", {}).get("name") == TEST_TARGET:
            executables.append(executable)
    if not executables:
        raise SystemExit("cargo reported no serve_http test executable")
    return Path(executables[-1]), build_ms


def validate_probe(document: dict) -> None:
    """Raises SystemExit unless the probe report proves the contract the census claims."""
    if document.get("schema") != PROBE_SCHEMA:
        raise SystemExit(f"unexpected probe schema: {document.get('schema')!r}")
    equality = document.get("equality", {})
    if equality.get("content") is not True:
        raise SystemExit("the streamed content does not equal the buffered content")
    if equality.get("tool_calls_byte_for_byte") is not True:
        raise SystemExit("the streamed tool_calls array does not equal the buffered array")
    if document.get("streamed", {}).get("finish_reason") != "tool_calls":
        raise SystemExit("the streamed final chunk is not finish_reason tool_calls")
    if document.get("buffered", {}).get("finish_reason") != "tool_calls":
        raise SystemExit("the buffered choice is not finish_reason tool_calls")
    sse = document.get("sse", {})
    if sse.get("role_frame_first") is not True or sse.get("ends_with_done") is not True:
        raise SystemExit(f"SSE ordering is broken: {sse}")
    if sse.get("dsml_leaked_into_content") is not False:
        raise SystemExit("the DSML block leaked into streamed content")
    counts = document.get("delta_counts", {})
    if int(counts.get("call_start_deltas", 0)) < 1:
        raise SystemExit("the stream carried no call-start delta")
    if int(counts.get("argument_fragment_deltas", 0)) < 1:
        raise SystemExit("the stream carried no argument fragments")
    keep_alive = document.get("keep_alive", {})
    if keep_alive.get("last_frame_is_done") is not True:
        raise SystemExit("a keep-alive comment followed the SSE terminator")
    if keep_alive.get("first_frame_is_data") is not True:
        raise SystemExit("a keep-alive comment preceded the first data frame")
    if keep_alive.get("comments_are_whole_frames") is not True:
        raise SystemExit("a keep-alive comment was not a whole frame")
    if not document.get("non_claims"):
        raise SystemExit("the probe report carries no non-claims list")


def summarize(runs: list[dict], probe_ms: list[int], server_hashes: list[str]) -> dict:
    """Folds the per-run probe reports into the census accounting and repeatability blocks."""
    if not runs:
        raise SystemExit("no probe runs to summarize")
    for document in runs:
        validate_probe(document)
    wall_ms = [int(document.get("wall", {}).get("stream_wall_ms", 0)) for document in runs]
    fixtures = {
        canonical_hash(
            {
                "buffered": document.get("buffered"),
                "pieces": document.get("script", {}).get("pieces"),
            }
        )
        for document in runs
    }
    if len(fixtures) != 1:
        raise SystemExit(f"the probe runs used different fixtures: {sorted(fixtures)}")
    if len(set(server_hashes)) != 1:
        raise SystemExit(f"the probe runs used different serve binaries: {server_hashes}")
    counts = [document.get("delta_counts", {}) for document in runs]
    keep_alive = [document.get("keep_alive", {}) for document in runs]
    buffered_calls = [document.get("buffered", {}).get("tool_calls", []) for document in runs]
    streamed_calls = [document.get("streamed", {}).get("tool_calls", []) for document in runs]
    return {
        "runs": len(runs),
        "fixture_hash": fixtures.pop(),
        "server_binary_sha256": server_hashes[0],
        "payload_equal": all(
            document.get("equality", {}).get("tool_calls_byte_for_byte") is True
            for document in runs
        ),
        "content_equal": all(
            document.get("equality", {}).get("content") is True for document in runs
        ),
        "script_pieces": sorted(
            {int(document.get("script", {}).get("pieces", 0)) for document in runs}
        ),
        "tool_calls": len(buffered_calls[0]) if buffered_calls else 0,
        "arguments": [
            call.get("function", {}).get("arguments") if isinstance(call, dict) else None
            for document in runs
            for call in document.get("buffered", {}).get("tool_calls", [])
        ],
        "finish_reasons": sorted(
            {
                str(document.get("streamed", {}).get("finish_reason"))
                for document in runs
            }
        ),
        "content_deltas": [int(entry.get("content_deltas", 0)) for entry in counts],
        "call_start_deltas": [int(entry.get("call_start_deltas", 0)) for entry in counts],
        "argument_fragment_deltas": [
            int(entry.get("argument_fragment_deltas", 0)) for entry in counts
        ],
        "sse_data_frames": [int(entry.get("sse_data_frames", 0)) for entry in counts],
        "raw_frames": [int(entry.get("raw_frames", 0)) for entry in keep_alive],
        "comment_frames": [
            int(entry.get("comment_frames", 0)) for entry in keep_alive
        ],
        "comment_positions_monotonic": all(
            positions == sorted(positions)
            for entry in keep_alive
            for positions in [entry.get("comment_positions", [])]
        ),
        "stream_wall_ms": wall_ms,
        "stream_wall_ms_median": statistics.median(wall_ms) if wall_ms else 0.0,
        "stream_wall_ms_min": min(wall_ms) if wall_ms else 0,
        "stream_wall_ms_max": max(wall_ms) if wall_ms else 0,
        "probe_process_ms": probe_ms,
        "streamed_calls": streamed_calls[0] if streamed_calls else [],
    }


def render_markdown(census: dict) -> str:
    """The human-readable census: scope, criterion, accounting, repeatability, non-claims."""
    account = census["accounting"]
    lines = [
        "# Streaming tool-call (V4.1 DSML) wire census",
        "",
        f"- schema: {census['schema']}",
        f"- generated: {census['generated_at']}",
        f"- git: {census['git']['commit']} on {census['git']['branch']}",
        f"- scope: {census['scope']}",
        f"- serve binary sha256: {census['serve_binary']['sha256']}",
        f"- test binary sha256: {census['test_binary']['sha256']}",
        f"- fixture hash: {account['fixture_hash']} "
        f"({', '.join(str(pieces) for pieces in account['script_pieces'])} scripted content "
        f"pieces, {account['tool_calls']} tool call)",
        "",
        "## Criterion",
        "",
        "The model answer is the V4.1 DSML block the buffered parser lifts, scripted as content",
        "deltas split at arbitrary character boundaries (DSML delimiters and JSON included). The",
        "streamed SSE deltas are reassembled the way an OpenAI client would: call start carries the",
        "id and empty arguments string, each later delta appends arguments, and the final choice",
        "chunk carries finish_reason tool_calls. The reassembled array must equal the buffered",
        "tool_calls array byte for byte, the streamed content must equal the buffered content, and",
        "the DSML text must never appear in content. The same run measures the keep-alive comment",
        "interleave: comments are whole SSE frames, none precedes the first data frame, none follows",
        "the terminator, and the deltas are unchanged by the interleave.",
        "",
        "## Accounting",
        "",
        f"- runs: {account['runs']} (same scripted fixture, loopback socket)",
        f"- tool_calls byte-for-byte equal: {account['payload_equal']}",
        f"- content equal: {account['content_equal']}",
        f"- finish reason: {', '.join(account['finish_reasons'])}",
        f"- assembled arguments: {account['arguments']}",
        f"- content deltas per run: {account['content_deltas']}",
        f"- call-start deltas per run: {account['call_start_deltas']}",
        f"- argument-fragment deltas per run: {account['argument_fragment_deltas']}",
        f"- SSE data frames per run: {account['sse_data_frames']} (terminator excluded)",
        f"- keep-alive comment frames per run: {account['comment_frames']} "
        f"(raw frames {account['raw_frames']}; positions monotonic: "
        f"{account['comment_positions_monotonic']})",
        "",
        "## Repeatability (same scripted stub, UNBARED)",
        "",
        f"- streamed request wall ms: {account['stream_wall_ms']} "
        f"(median {account['stream_wall_ms_median']}, min {account['stream_wall_ms_min']}, "
        f"max {account['stream_wall_ms_max']})",
        f"- probe process ms per run: {account['probe_process_ms']}",
        "",
        "Wall numbers are repeatability of the same host run, not a throughput or speedup claim.",
        "",
        "## Non-claims",
        "",
        "- Host path only: no CUDA, no device kernel, no GPU, no cluster run, no vLLM comparison.",
        "- The engine is a debug-build scripted stub, not the released checkpoint: this measures",
        "  the parser and the wire framing, not model quality or reference-decoder parity.",
        "- The fixture is a fixed synthetic DSML answer, not a captured model output; the real",
        "  model's DSML emission is not measured here.",
        "- Streaming logprobs combined with offered tools is a documented residual gap (the parser",
        "  is disabled in that mode); this census does not exercise that combination.",
        "- Wall numbers are repeatability, not a throughput or speedup claim.",
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
    base = f"tool-call-streaming-census-{stamp}"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    test_binary, build_ms = resolve_test_binary(repo, args.cargo)
    runs: list[dict] = []
    probe_ms: list[int] = []
    server_hashes: list[str] = []
    server_binary: Path | None = None
    with tempfile.TemporaryDirectory(prefix="tool-call-streaming-probe-") as scratch:
        scratch_path = Path(scratch)
        for run in range(args.runs):
            report_path = scratch_path / f"probe-{run}.json"
            environment = os.environ.copy()
            environment[PROBE_ENV] = str(report_path)
            started = time.monotonic()
            completed = subprocess.run(
                [str(test_binary), "--exact", PROBE_TEST, "--nocapture"],
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
                raise SystemExit(f"probe run {run} failed (exit {completed.returncode}):\n{tail}")
            document = json.loads(report_path.read_text(encoding="utf-8"))
            validate_probe(document)
            binary = Path(str(document.get("server_binary", "")))
            if not binary.is_file():
                raise SystemExit(f"the probe names a missing serve binary: {binary}")
            if server_binary is None:
                server_binary = binary
            elif server_binary != binary:
                raise SystemExit(
                    f"probe runs used different serve binaries: {server_binary} vs {binary}"
                )
            server_hashes.append(sha256_file(binary))
            runs.append(document)
            probe_ms.append(elapsed_ms)
    assert server_binary is not None
    accounting = summarize(runs, probe_ms, server_hashes)
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
        "test_binary": {
            "path": str(test_binary),
            "sha256": sha256_file(test_binary),
            "bytes": test_binary.stat().st_size,
            "build_ms": build_ms,
        },
        "serve_binary": {
            "path": str(server_binary),
            "sha256": accounting["server_binary_sha256"],
            "bytes": server_binary.stat().st_size,
        },
        "toolchain": {"cargo": cargo_version, "rustc": rustc_version},
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "loadavg": list(load),
        },
        "fixture": {
            "hash": accounting["fixture_hash"],
            "description": (
                "fixed V4.1 DSML answer (two parameters, one call) scripted as content pieces "
                "split every 5 characters; buffered and streamed payloads are the fixture identity"
            ),
            "script_pieces": accounting["script_pieces"],
        },
        "accounting": accounting,
        "source_manifests": [
            {"path": relative, "sha256": sha256_file(repo / relative)}
            for relative in SOURCE_MANIFESTS
            if (repo / relative).is_file()
        ],
        "raw_runs": runs,
        "non_claims": [
            "host path only: no CUDA, no device kernel, no GPU, no cluster run, no vLLM run",
            "debug-build scripted stub engine, not the released checkpoint",
            "fixed synthetic DSML answer, not a captured model output",
            "streaming logprobs combined with offered tools is a documented residual gap, not measured",
            "wall numbers are repeatability, not a throughput or speedup claim",
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
        "test_binary": census["test_binary"]["sha256"],
        "serve_binary": census["serve_binary"]["sha256"],
        "source_manifests": {
            entry["path"]: entry["sha256"] for entry in census["source_manifests"]
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"json: {json_path} sha256={manifest['json']}")
    print(f"md: {md_path} sha256={manifest['md']}")
    print(f"test binary: {test_binary} sha256={manifest['test_binary']}")
    print(f"serve binary: {server_binary} sha256={manifest['serve_binary']}")
    print(
        "accounting: "
        f"{accounting['runs']} runs, payload_equal={accounting['payload_equal']}, "
        f"content_equal={accounting['content_equal']}, "
        f"calls={accounting['tool_calls']}, "
        f"content_deltas={accounting['content_deltas']}, "
        f"argument_fragments={accounting['argument_fragment_deltas']}, "
        f"comment_frames={accounting['comment_frames']}, "
        f"wall_ms={accounting['stream_wall_ms']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
