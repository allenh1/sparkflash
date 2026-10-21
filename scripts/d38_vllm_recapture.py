#!/usr/bin/env python3
"""D38 vLLM recapture: a measured, reproducible baseline from a live OpenAI-compatible endpoint.

WHY THIS EXISTS
    D52 (2026-10-21) parked the decode-mode vLLM recapture, but replacing vLLM needs a real
    comparison, so a live four-node TP4 deployment serving deepseek-v4.1-flash is the recapture
    opportunity. This script captures that deployment as evidence instead of prose: per-request
    client-measured TTFT and decode tok/s, generated tokens and their logprobs, prompt logprobs
    (target-token rank/logprob), and the provenance needed to reproduce the row (served model
    root, server argv, node GPU occupancy, timestamps, source revision).

HONESTY RULES (enforced by the document this script writes)
    * Every number is labelled MEASURED-CLIENT (this client's monotonic clock) or
      SERVER-REPORTED (the server's own response and /metrics counters). No number is inferred.
    * A vLLM baseline is only comparable when it was captured on the checkpoint the repo's
      parity oracle uses. D38's released root is
      .../snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170; the live deployment has
      historically served the ...-heretic snapshot 12d58ad0.... A different root is recorded as
      d38_root_precondition.met = false and the document is marked gate_eligible = false; it is
      then a DEPLOYMENT baseline, never a parity gate.
    * Co-tenant load is recorded, not hidden. The server-reported num_requests_running is sampled
      before and after every request; a capture taken while other streams are running is
      latency-under-load and is labelled as such. This script never touches the server state (no
      stop/restart/config change) and sends only a small, fixed number of short requests.
    * No token-equality claim (D39) and no tok/s claim about SparkFlash: the only numbers here
      are vLLM's.

WHAT IS MEASURED
    canonical prompt  : b69-fixture-130 - the frozen 130-token D41/D48 prompt, max_tokens 9 (the
                        D38 nine-position token row / D48 nine-token row), 3 samples.
    d41 control prompt: the D41 plan's idle-A-repeat-139 request rep-139-0, same recipe with
                        max_tokens 9 (D41's own recipe uses max_tokens 1 for prompt_logprobs;
                        generating tokens is required for a decode row, so the deviation is
                        recorded in the row), 3 samples.
    steady-state      : b69-fixture-130 again with max_tokens 64, 3 samples, labelled
                        diagnostic-steady-state (a 9-token window is noisy, and speculative
                        decoding emits several tokens per stream chunk).
    prompt logprobs   : prompt_logprobs rows for the b69-fixture-130 random-token prompt and for
                        three natural-language prompts. The fixture prompt is 130 RANDOM token ids
                        (seed 7; task32-root-cause.md), so a low target logprob and a huge rank
                        there are EXPECTED of a correct model and are labelled that way. Whether
                        the served checkpoint can predict its own natural-text continuation is a
                        separate measured row, and that row is what an "is this a usable oracle?"
                        question must read.

Usage (one command; no third party packages, stdlib only):

    python3 scripts/d38_vllm_recapture.py \
        --base-url http://192.0.2.80:8000 --model deepseek-v4.1-flash \
        --out-json docs/build/d38-vllm-baseline-20261021.json \
        --out-md   docs/build/d38-vllm-recapture-20261021.md

Exit codes: 0 capture written (all samples ok), 1 capture written with failed samples, 2 bad
input / missing fixture (no capture), 3 dry-run completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA = "sparkflash-d38-vllm-baseline/1"
DEFAULT_BASE_URL = "http://192.0.2.80:8000"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_SEED = 20261021
DEFAULT_LOGPROBS = 5
DEFAULT_SSH_TIMEOUT_S = 15.0

#: D38 root: the released snapshot the fixture, the device engine and the frozen D41 oracle use.
D38_RELEASED_ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
#: The deployment observed on 2026-10-21 (serving-readiness.md 5.1: "not a parity target").
KNOWN_HERETIC_ROOT = "12d58ad0878b5008e80d720251adb969a0069412"

FIXTURE_PATH = "/home/user/phase2-evidence/sequence-130-released.json"
D41_PLAN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d41_capture_plan.json")

#: Metric families read from /metrics. Deliberately a named subset: the evidence records exactly
#: the counters it interprets, not a snapshot of the whole exposition.
METRIC_GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)
METRIC_COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:request_prefill_time_seconds_count",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_count",
    "vllm:request_decode_time_seconds_sum",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)

#: Prompts are token-id lists on purpose: the fixture and the D41 plan pin integers, so the
#: capture does not depend on a tokenizer version (and cannot silently re-tokenise the prompt).
PROMPT_SPECS = (
    {
        "id": "b69-fixture-130",
        "role": "canonical-d38-d48",
        "source": {"kind": "fixture", "path": FIXTURE_PATH, "prompt_key": "prompt_tokens"},
        "max_tokens": 9,
        "samples": 3,
        "label": "MEASURED-CLIENT canonical prompt (D41/D48 130-token prompt, D38 nine-position row)",
    },
    {
        "id": "d41-repeat-139",
        "role": "d41-control",
        "source": {
            "kind": "d41-plan",
            "path": D41_PLAN_PATH,
            "block_id": "idle-A-repeat-139",
            "request_id": "rep-139-0",
        },
        "max_tokens": 9,
        "samples": 3,
        "label": "MEASURED-CLIENT D41 repeat-block prompt (max_tokens 9 vs D41 recipe's 1: decode row)",
    },
    {
        "id": "b69-fixture-130-steady",
        "role": "diagnostic-steady-state",
        "source": {"kind": "fixture", "path": FIXTURE_PATH, "prompt_key": "prompt_tokens"},
        "max_tokens": 64,
        "samples": 3,
        "label": "MEASURED-CLIENT diagnostic steady-state row (longer window; not a D41/D48 row)",
    },
)

#: Natural-language prompt_logprobs probes. These are the degeneracy / usable-oracle test: the
#: D41/D48 fixture prompt is random token ids, so it can never be that test.
NATURAL_PROMPTS = (
    {"id": "nl-capital", "text": "The capital of France is"},
    {"id": "nl-rust", "text": "In Rust, the borrow checker rejects this code because:"},
    {"id": "nl-chat", "text": "User: What is the capital of France?\nAssistant:"},
)


def utc_now() -> str:
    """The evidence timestamp format used by the other scripts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc() -> str:
    """Compact UTC stamp for run ids."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(payload: bytes) -> str:
    """Hex sha256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str) -> str:
    """Hex sha256 of a file, streamed so a large fixture does not have to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_nearest_rank(values: list[float], quantile: float) -> float | None:
    """Nearest-rank percentile (no interpolation), documented so p50/p90 are reproducible.

    For n samples the rank is ceil(q*n), 1-based; q=0.5 with n=3 reports the median sample, and
    q=0.9 with n=3 reports the maximum. Interpolation is deliberately not used: the evidence is a
    handful of real samples, and a linearly interpolated p90 would be a fabricated value between
    samples.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(math.ceil(quantile * len(ordered)))))
    return ordered[rank - 1]


def describe_error(exc: BaseException) -> str:
    """One-line error description, including the HTTP body when there is one."""
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        return f"HTTP {exc.code}: {body}".strip()
    return f"{type(exc).__name__}: {exc}"


def http_json(url: str, timeout: float) -> dict:
    """GET a JSON document; raise ValueError when the body is not a JSON object."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", "replace")
    document = json.loads(body)
    if not isinstance(document, dict):
        raise ValueError(f"{url} returned a non-object body")
    return document


def parse_metrics(text: str) -> dict[str, float]:
    """Parse the named metric families out of a Prometheus text exposition.

    The exposition may carry several label sets per family (per engine, per finish reason, per
    speculative position). Gauges keep the maximum over label sets; counters are summed, which is
    what the aggregate rate arithmetic needs. A family absent from the exposition is absent from
    the result rather than zero, so "no metric" can never be read as "measured zero".
    """
    wanted = set(METRIC_GAUGES) | set(METRIC_COUNTERS)
    gauges: dict[str, float] = {}
    counters: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, remainder = line.partition("{")
        if name not in wanted:
            continue
        if "}" in remainder:
            _, _, value_text = remainder.rpartition("}")
        else:  # bare "name value" line
            _, _, value_text = line.partition(" ")
        try:
            value = float(value_text.strip())
        except ValueError:
            continue
        if name in METRIC_GAUGES:
            gauges[name] = max(gauges.get(name, value), value)
        else:
            counters[name] = counters.get(name, 0.0) + value
    result = dict(gauges)
    result.update(counters)
    return result


def snapshot_load(base_url: str, timeout: float) -> dict:
    """One server-reported load snapshot, used before/after the run and around every request."""
    snapshot = {"captured_utc": utc_now(), "source": base_url.rstrip("/") + "/metrics"}
    try:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/metrics", headers={"Accept": "text/plain"}
        )
        with urllib.request.urlopen(request, timeout=min(timeout, 30.0)) as response:
            text = response.read().decode("utf-8", "replace")
        metrics = parse_metrics(text)
        snapshot["ok"] = True
        snapshot["metrics"] = metrics
        snapshot["exposition_sha256"] = sha256_bytes(text.encode("utf-8"))
        running = metrics.get("vllm:num_requests_running")
        snapshot["num_requests_running"] = running
        snapshot["co_tenant_note"] = (
            "server-reported gauge; a positive value means latency measured next is under load"
        )
    except Exception as error:  # noqa: BLE001 - a missing /metrics is evidence, not a crash
        snapshot["ok"] = False
        snapshot["error"] = describe_error(error)
    return snapshot


def local_endpoint_clients(base_url: str, timeout: float) -> dict:
    """Identify local sockets holding connections to the measured endpoint (read-only ss).

    This answers "who is the co-tenant?" with local evidence instead of a guess: it lists the
    local process name/pid that owns each connection to host:port. It does not stop anything.
    """
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    needle = f"{host}:{port}"
    result = {"endpoint": needle, "command": ["ss", "-tnp"], "ok": False, "clients": []}
    try:
        completed = subprocess.run(["ss", "-tnp"], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        result["error"] = describe_error(error)
        return result
    result["ok"] = completed.returncode == 0
    seen = {}
    for line in completed.stdout.splitlines():
        if needle not in line:
            continue
        parts = line.split()
        local = parts[3] if len(parts) > 3 else None
        peer = parts[4] if len(parts) > 4 else None
        process = None
        pid = None
        if "users:((" in line:
            inside = line.split("users:((", 1)[1].split(")", 1)[0]
            bits = inside.split(",")
            process = bits[0].strip().strip('"')
            for bit in bits[1:]:
                if bit.startswith("pid="):
                    pid = int(bit[4:])
        key = (local, peer, process, pid)
        seen[key] = seen.get(key, 0) + 1
    result["clients"] = [
        {"local": local, "peer": peer, "process": process, "pid": pid, "connections": count}
        for (local, peer, process, pid), count in seen.items()
    ]
    result["connection_count"] = sum(result["clients"][i]["connections"] for i in range(len(result["clients"])))
    result["label"] = "MEASURED-LOCAL (ss socket owners; read-only, nothing stopped)"
    return result


def wait_for_idle(base_url: str, args: argparse.Namespace, idle_state: dict) -> dict:
    """Poll until the server reports num_requests_running == 0, bounded by the idle budget.

    A bounded wait is how a quiet sample set is attempted without stopping anyone: if the engine
    never goes idle, the sample is taken anyway and the wait result is recorded (never hidden).
    """
    info = {
        "requested": bool(getattr(args, "wait_idle_s", 0) or 0) > 0,
        "waited_s": 0.0,
        "idle_at_start": None,
        "budget_remaining_s_before": idle_state.get("remaining"),
        "source": "server-reported /metrics vllm:num_requests_running polled every 0.5 s",
    }
    if not info["requested"]:
        info["skipped"] = "not requested (--wait-idle-s 0)"
        return info
    if idle_state.get("remaining", 0.0) <= 0:
        info["skipped"] = "idle budget exhausted"
        return info
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        budget_left = idle_state["remaining"] - elapsed
        if elapsed >= args.wait_idle_s or budget_left <= 0:
            break
        snapshot = snapshot_load(base_url, args.timeout)
        if snapshot.get("num_requests_running") == 0:
            info["idle_at_start"] = True
            info["waited_s"] = round(elapsed, 3)
            idle_state["remaining"] = max(0.0, idle_state["remaining"] - elapsed)
            info["budget_remaining_s_after"] = idle_state["remaining"]
            return info
        time.sleep(0.5)
    info["idle_at_start"] = False
    info["waited_s"] = round(time.monotonic() - started, 3)
    idle_state["remaining"] = max(0.0, idle_state["remaining"] - info["waited_s"])
    info["budget_remaining_s_after"] = idle_state["remaining"]
    return info


def run_readonly(command: list[str], timeout: float) -> dict:
    """Run one short read-only command (ssh or local ps/git) and record its output or error."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"argv": command, "ok": False, "error": describe_error(error)}
    return {
        "argv": command,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip()[:400],
    }


def collect_node_provenance(ssh_hosts: list[str], timeout: float) -> list[dict]:
    """One read-only ssh per node: identity, vLLM worker argv, GPU occupancy.

    This never starts, stops or reconfigures anything. The GPU memory row is also the evidence
    for why a SparkFlash device measurement cannot run while this deployment holds the cards.
    """
    remote = (
        "echo HOST=$(hostname); "
        "ps -eo pid,args | grep -m1 'vllm serve' | grep -v grep | sed 's/^/SERVE=/'; "
        "ps -eo pid,args | grep -m1 'VLLM::Worker' | grep -v grep | sed 's/^/WORKER=/'; "
        "ps -eo pid,args | grep -m1 'launch-cluster.sh' | grep -v grep | sed 's/^/LAUNCHER=/'; "
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader 2>&1 | sed 's/^/GPU=/' | head -2"
    )
    nodes = []
    for host in ssh_hosts:
        record = {"host": host, "captured_utc": utc_now()}
        command = [
            "ssh",
            "-n",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=6",
            "-o",
            "StrictHostKeyChecking=no",
            host,
            remote,
        ]
        result = run_readonly(command, timeout)
        record.update(result)
        if result.get("ok"):
            record["hostname"] = None
            record["server_argv"] = None
            record["worker_argv"] = None
            record["launcher_argv"] = None
            record["gpu"] = []
            for line in result["stdout"].splitlines():
                line = line.strip()
                if line.startswith("HOST="):
                    record["hostname"] = line[len("HOST="):].strip() or None
                elif line.startswith("SERVE="):
                    record["server_argv"] = line[len("SERVE="):].strip()
                elif line.startswith("WORKER="):
                    record["worker_argv"] = line[len("WORKER="):].strip()
                elif line.startswith("LAUNCHER="):
                    record["launcher_argv"] = line[len("LAUNCHER="):].strip()
                elif line.startswith("GPU="):
                    record["gpu"].append(line[len("GPU="):].strip())
        nodes.append(record)
    return nodes


def git_provenance() -> dict:
    """Source revision of the capture client; the tree is shared, so dirty is expected."""
    head = run_readonly(["git", "rev-parse", "HEAD"], 10.0)
    status = run_readonly(["git", "status", "--porcelain"], 15.0)
    porcelain = status.get("stdout", "") if status.get("ok") else ""
    dirty_paths = [line for line in porcelain.splitlines() if line.strip()]
    return {
        "head": head.get("stdout") if head.get("ok") else None,
        "dirty": bool(dirty_paths),
        "dirty_path_count": len(dirty_paths),
        "dirty_paths_sha256": sha256_bytes(porcelain.encode("utf-8")) if porcelain else None,
        "note": "the working tree is shared with parallel workstreams; dirty != this script's diff",
    }


def load_fixture_prompt(path: str) -> tuple[list[int], dict]:
    """Load and validate the frozen 130-token prompt from the D38/D41 fixture."""
    sha = sha256_file(path)
    with open(path, "r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    tokens = fixture.get("prompt_tokens")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, int) for t in tokens):
        raise ValueError(f"{path}: prompt_tokens is not a non-empty list of ints")
    steps = fixture.get("steps") or []
    reference = [step.get("next_token") for step in steps[:9] if isinstance(step, dict)]
    provenance = {
        "path": path,
        "sha256": sha,
        "prompt_len": len(tokens),
        "prompt_sha256": sha256_bytes(",".join(str(t) for t in tokens).encode("utf-8")),
        "schema_version": fixture.get("schema_version"),
        "config_label": fixture.get("config_label"),
        "nine_step_reference_tokens": reference,
        "prompt_nature": (
            "RANDOM token ids: the 130-token fixture prompt is 130 random ids (seed 7; see "
            "task32-root-cause.md 'The fixture prompt ids (130 random ids, seed 7)'). A correct "
            "model assigns low logprob to a random-token continuation, so prompt-logprob rows on "
            "this prompt are NOT a natural-text degeneracy test and are labelled accordingly."
        ),
    }
    return tokens, provenance


def load_d41_prompt(path: str, block_id: str, request_id: str) -> tuple[list[int], dict]:
    """Load one pinned token-id prompt from the frozen D41 capture plan."""
    with open(path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    for block in plan.get("blocks", []):
        if block.get("id") != block_id:
            continue
        for request in block.get("requests", []):
            if request.get("id") != request_id:
                continue
            tokens = request.get("prompt")
            if not isinstance(tokens, list) or not tokens or not all(
                isinstance(t, int) for t in tokens
            ):
                raise ValueError(f"{path}: {block_id}/{request_id} prompt is not a token-id list")
            return tokens, {
                "path": path,
                "sha256": sha256_file(path),
                "plan_schema": plan.get("schema"),
                "block_id": block_id,
                "request_id": request_id,
                "prompt_len": len(tokens),
                "prompt_sha256": sha256_bytes(",".join(str(t) for t in tokens).encode("utf-8")),
                "d41_recipe": plan.get("recipe"),
            }
    raise ValueError(f"{path}: block {block_id!r} request {request_id!r} not found")


def build_payload(model: str, prompt: list[int], max_tokens: int, seed: int, logprobs: int) -> dict:
    """The request recipe. Every knob that affects the row is explicit and recorded."""
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "ignore_eos": True,
        "seed": int(seed),
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": int(logprobs),
        "return_token_ids": True,
    }


@dataclass
class StreamResult:
    """One streamed completion request: timings, tokens, logprobs and the raw final usage."""

    ok: bool
    error: str | None
    http_status: int | None
    t0_ns: int
    headers_ns: int | None
    first_token_ns: int | None
    last_token_ns: int | None
    done_ns: int | None
    chunks: list[dict]
    token_ids: list[int]
    token_logprobs: list[float | None]
    top_logprobs: list[dict]
    text: str
    finish_reason: str | None
    usage: dict | None
    system_fingerprint: str | None
    response_id: str | None


def empty_stream_result(t0_ns: int) -> StreamResult:
    """A StreamResult for a request that failed before any chunk was parsed."""
    return StreamResult(
        ok=False,
        error=None,
        http_status=None,
        t0_ns=t0_ns,
        headers_ns=None,
        first_token_ns=None,
        last_token_ns=None,
        done_ns=None,
        chunks=[],
        token_ids=[],
        token_logprobs=[],
        top_logprobs=[],
        text="",
        finish_reason=None,
        usage=None,
        system_fingerprint=None,
        response_id=None,
    )


def parse_chunk(payload: dict, arrival_ns: int, state: StreamResult) -> None:
    """Fold one SSE data payload into the running StreamResult."""
    if state.response_id is None and isinstance(payload.get("id"), str):
        state.response_id = payload["id"]
    if state.system_fingerprint is None and isinstance(payload.get("system_fingerprint"), str):
        state.system_fingerprint = payload["system_fingerprint"]
    usage = payload.get("usage")
    if isinstance(usage, dict):
        state.usage = usage
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        return
    if state.finish_reason is None and choice.get("finish_reason") is not None:
        state.finish_reason = str(choice["finish_reason"])
    text = choice.get("text") or ""
    prefix = len(state.text)
    state.text += text
    logprobs = choice.get("logprobs")
    token_ids = choice.get("token_ids")
    chunk_ids: list[int] = []
    if isinstance(token_ids, list) and token_ids:
        chunk_ids = [int(token) for token in token_ids if isinstance(token, int)]
    if isinstance(logprobs, dict):
        token_logprobs = logprobs.get("token_logprobs")
        tops = logprobs.get("top_logprobs")
        if isinstance(token_logprobs, list):
            state.token_logprobs.extend(
                float(value) if isinstance(value, (int, float)) else None
                for value in token_logprobs
            )
        if isinstance(tops, list):
            for entry in tops:
                state.top_logprobs.append(entry if isinstance(entry, dict) else {})
    state.token_ids.extend(chunk_ids)
    token_count = len(chunk_ids) if chunk_ids else (1 if text else 0)
    state.chunks.append(
        {
            "arrival_ms": round((arrival_ns - state.t0_ns) / 1e6, 3),
            "text_offset": prefix,
            "text": text,
            "token_count": token_count,
            "token_ids": chunk_ids,
        }
    )
    if token_count:
        if state.first_token_ns is None:
            state.first_token_ns = arrival_ns
        state.last_token_ns = arrival_ns


def stream_completion(url: str, payload: dict, timeout: float) -> StreamResult:
    """POST a streaming completion and fold every SSE chunk into a StreamResult.

    A failed request is a labelled sample error, not a crash: the capture continues so the
    evidence records which samples failed and why.
    """
    state = empty_stream_result(time.monotonic_ns())
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            state.http_status = int(response.status)
            state.headers_ns = time.monotonic_ns()
            for raw_line in response:
                arrival_ns = time.monotonic_ns()
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    state.done_ns = arrival_ns
                    break
                try:
                    event = json.loads(body)
                except json.JSONDecodeError as error:
                    state.error = f"bad SSE JSON: {error}"
                    continue
                if isinstance(event, dict):
                    parse_chunk(event, arrival_ns, state)
        if state.done_ns is None:
            state.done_ns = time.monotonic_ns()
        if state.error is None:
            state.ok = True
    except Exception as error:  # noqa: BLE001 - recorded as a failed sample
        state.error = describe_error(error)
        state.ok = False
        if state.done_ns is None:
            state.done_ns = time.monotonic_ns()
    return state


def sample_metrics(state: StreamResult) -> dict:
    """Client-measured timings and rates for one sample, with null where nothing was measured."""
    completion_tokens = None
    prompt_tokens = None
    if isinstance(state.usage, dict):
        if isinstance(state.usage.get("completion_tokens"), int):
            completion_tokens = state.usage["completion_tokens"]
        if isinstance(state.usage.get("prompt_tokens"), int):
            prompt_tokens = state.usage["prompt_tokens"]
    stream_tokens = len(state.token_ids)
    if completion_tokens is None and stream_tokens:
        completion_tokens = stream_tokens
    ttft_ms = None
    if state.first_token_ns is not None:
        ttft_ms = round((state.first_token_ns - state.t0_ns) / 1e6, 3)
    wall_ms = round((state.done_ns - state.t0_ns) / 1e6, 3) if state.done_ns else None
    decode_window_ms = None
    decode_tok_s = None
    if state.first_token_ns is not None and state.last_token_ns is not None and completion_tokens:
        decode_window_ms = round((state.last_token_ns - state.first_token_ns) / 1e6, 3)
        if decode_window_ms > 0 and completion_tokens > 1:
            decode_tok_s = round((completion_tokens - 1) / (decode_window_ms / 1000.0), 3)
    tokens_per_s = None
    if wall_ms and wall_ms > 0 and completion_tokens:
        tokens_per_s = round(completion_tokens / (wall_ms / 1000.0), 3)
    chunk_token_counts = [chunk["token_count"] for chunk in state.chunks if chunk["token_count"]]
    return {
        "ttft_ms": ttft_ms,
        "wall_ms": wall_ms,
        "decode_window_ms": decode_window_ms,
        "decode_tok_s": decode_tok_s,
        "tokens_per_s": tokens_per_s,
        "completion_tokens_server_reported": (
            state.usage.get("completion_tokens") if isinstance(state.usage, dict) else None
        ),
        "completion_tokens_streamed": stream_tokens,
        "prompt_tokens_server_reported": prompt_tokens,
        "stream_chunks": len(state.chunks),
        "stream_chunks_with_tokens": len(chunk_token_counts),
        "max_tokens_per_chunk": max(chunk_token_counts) if chunk_token_counts else None,
        "finish_reason": state.finish_reason,
        "usage": state.usage,
        "system_fingerprint": state.system_fingerprint,
        "response_id": state.response_id,
        "ttft_source": (
            "MEASURED-CLIENT (monotonic clock, first chunk carrying text/tokens; includes HTTP "
            "connect, queueing and prefill)"
        ),
        "decode_tok_s_source": (
            "MEASURED-CLIENT ((usage.completion_tokens - 1) / (last token chunk arrival - first "
            "token chunk arrival)); chunk granularity with speculative decoding is recorded"
        ),
    }


def _rounded(value: float | None) -> float | None:
    """Round an aggregate to millisecond/tok-s precision, preserving None."""
    return None if value is None else round(value, 3)


def capture_prompt(
    base_url: str,
    model: str,
    spec: dict,
    prompt: list[int],
    prompt_provenance: dict,
    args: argparse.Namespace,
    idle_state: dict,
) -> dict:
    """Run every sample for one prompt and aggregate with nearest-rank p50/p90."""
    payload = build_payload(model, prompt, spec["max_tokens"], args.seed, args.logprobs)
    samples = []
    for index in range(spec["samples"]):
        idle_wait = wait_for_idle(base_url, args, idle_state)
        load_before = snapshot_load(base_url, args.timeout)
        state = stream_completion(base_url.rstrip("/") + "/v1/completions", payload, args.timeout)
        load_after = snapshot_load(base_url, args.timeout)
        metrics = sample_metrics(state)
        sample = {
            "sample_index": index,
            "label": "MEASURED-CLIENT" if state.ok else "MEASURED-CLIENT-FAILED",
            "ok": state.ok,
            "error": state.error,
            "http_status": state.http_status,
            "request_utc": utc_now(),
            "request": {
                "endpoint": "/v1/completions",
                "prompt_len": len(prompt),
                "max_tokens": spec["max_tokens"],
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "seed": args.seed,
                "logprobs": args.logprobs,
                "stream": True,
                "return_token_ids": True,
            },
            "timings": metrics,
            "token_ids": state.token_ids,
            "text": state.text,
            "token_logprobs": state.token_logprobs,
            "top_logprobs": state.top_logprobs,
            "chunks": state.chunks,
            "idle_wait": idle_wait,
            "load_before": load_before,
            "load_after": load_after,
        }
        samples.append(sample)
        print(
            "  {} sample {}/{}: ttft={} ms decode={} tok/s tokens={} running(before/after)={}/{} ok={}".format(
                spec["id"],
                index + 1,
                spec["samples"],
                metrics["ttft_ms"],
                metrics["decode_tok_s"],
                metrics["completion_tokens_server_reported"],
                load_before.get("num_requests_running"),
                load_after.get("num_requests_running"),
                state.ok,
            ),
            flush=True,
        )

    ok_samples = [sample for sample in samples if sample["ok"]]
    ttfts = [s["timings"]["ttft_ms"] for s in ok_samples if s["timings"]["ttft_ms"] is not None]
    decodes = [s["timings"]["decode_tok_s"] for s in ok_samples if s["timings"]["decode_tok_s"] is not None]
    walls = [s["timings"]["wall_ms"] for s in ok_samples if s["timings"]["wall_ms"] is not None]
    rates = [s["timings"]["tokens_per_s"] for s in ok_samples if s["timings"]["tokens_per_s"] is not None]
    token_counts = [
        s["timings"]["completion_tokens_server_reported"]
        for s in ok_samples
        if s["timings"]["completion_tokens_server_reported"] is not None
    ]
    token_sequences = [s["token_ids"] for s in ok_samples]
    deterministic = bool(token_sequences) and all(
        sequence == token_sequences[0] for sequence in token_sequences
    )
    running_values = [s["load_before"].get("num_requests_running") for s in samples] + [
        s["load_after"].get("num_requests_running") for s in samples
    ]
    max_running = max(
        [value for value in running_values if isinstance(value, (int, float))], default=None
    )
    aggregate = {
        "samples_planned": spec["samples"],
        "samples_ok": len(ok_samples),
        "samples_failed": len(samples) - len(ok_samples),
        "ttft_ms": {
            "label": "MEASURED-CLIENT",
            "samples": [round(value, 3) for value in ttfts],
            "p50": _rounded(percentile_nearest_rank(ttfts, 0.5)),
            "p90": _rounded(percentile_nearest_rank(ttfts, 0.9)),
            "min": _rounded(min(ttfts)) if ttfts else None,
            "max": _rounded(max(ttfts)) if ttfts else None,
            "method": "nearest-rank percentile over the ok samples; no interpolation",
        },
        "decode_tok_s": {
            "label": "MEASURED-CLIENT",
            "samples": [round(value, 3) for value in decodes],
            "p50": _rounded(percentile_nearest_rank(decodes, 0.5)),
            "p90": _rounded(percentile_nearest_rank(decodes, 0.9)),
            "min": _rounded(min(decodes)) if decodes else None,
            "max": _rounded(max(decodes)) if decodes else None,
            "method": "nearest-rank percentile over the ok samples; no interpolation",
        },
        "wall_ms": {
            "label": "MEASURED-CLIENT",
            "p50": _rounded(percentile_nearest_rank(walls, 0.5)),
            "p90": _rounded(percentile_nearest_rank(walls, 0.9)),
            "min": _rounded(min(walls)) if walls else None,
            "max": _rounded(max(walls)) if walls else None,
        },
        "tokens_per_s_wall": {
            "label": "MEASURED-CLIENT",
            "p50": _rounded(percentile_nearest_rank(rates, 0.5)),
            "note": "completion tokens / full request wall time; includes prefill and TTFT",
        },
        "completion_tokens": {
            "label": "SERVER-REPORTED (usage.completion_tokens) cross-checked against streamed ids",
            "values": token_counts,
            "all_equal": len(set(token_counts)) == 1 if token_counts else None,
        },
        "co_tenant": {
            "max_num_requests_running_seen": max_running,
            "per_sample_num_requests_running": [
                {
                    "before": s["load_before"].get("num_requests_running"),
                    "after": s["load_after"].get("num_requests_running"),
                }
                for s in samples
            ],
            "label": "SERVER-REPORTED /metrics gauge sampled by this client around every request",
        },
        "greedy_deterministic_across_samples": deterministic,
        "greedy_token_ids_first_ok_sample": token_sequences[0] if token_sequences else [],
        "greedy_text_first_ok_sample": ok_samples[0]["text"] if ok_samples else None,
    }
    if prompt_provenance.get("nine_step_reference_tokens"):
        reference = prompt_provenance["nine_step_reference_tokens"]
        first = aggregate["greedy_token_ids_first_ok_sample"]
        compared = min(len(reference), len(first))
        matches = sum(1 for i in range(compared) if reference[i] == first[i]) if compared else 0
        aggregate["vs_b69_nine_step_reference"] = {
            "reference_root": D38_RELEASED_ROOT,
            "reference_tokens": reference,
            "generated_tokens": first,
            "positions_compared": compared,
            "positions_matching": matches,
            "label": (
                "DIAGNOSTIC (not a gate): the reference is the released-root fixture; a different "
                "served root is expected to disagree and D39 forbids a token-equality gate"
            ),
        }
    return {
        "id": spec["id"],
        "role": spec["role"],
        "label": spec["label"],
        "prompt_source": prompt_provenance,
        "max_tokens": spec["max_tokens"],
        "recipe": payload,
        "samples": samples,
        "aggregate": aggregate,
    }


def capture_prompt_logprobs(
    base_url: str,
    model: str,
    prompt: list[int] | None,
    args: argparse.Namespace,
    prompt_text: str | None = None,
    prompt_id: str | None = None,
    prompt_nature: str | None = None,
) -> dict:
    """One request with prompt_logprobs: target-token logprob and rank at every position.

    prompt_logprobs[i] is the distribution that predicts prompt token i; index 0 is null. The
    target ids are the ids the server itself tokenised (a token-id prompt sends the pinned ids; a
    text prompt returns prompt_token_ids because return_token_ids is set). target_rank_eq_1 is the
    measured top-1 self-prediction rate. On the random-token fixture a large rank is expected of a
    correct model; on natural-language prompts it is not. The aggregate and per-position rows are
    both kept so a reviewer can re-derive either.
    """
    if prompt is None and prompt_text is None:
        raise ValueError("capture_prompt_logprobs needs a token-id prompt or prompt_text")
    request_prompt: list[int] | str = prompt if prompt is not None else prompt_text
    payload = {
        "model": model,
        "prompt": request_prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "seed": args.seed,
        "prompt_logprobs": args.prompt_logprobs,
        "return_token_ids": True,
    }
    result = {
        "prompt_id": prompt_id,
        "prompt_nature": prompt_nature,
        "prompt_text": prompt_text,
        "label": "MEASURED-CLIENT request / SERVER-REPORTED logprobs",
        "request": {key: value for key, value in payload.items() if key != "prompt"},
        "ok": False,
    }
    result["load_before"] = snapshot_load(base_url, args.timeout)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result["http_status"] = int(response.status)
            body = response.read().decode("utf-8", "replace")
        document = json.loads(body)
        choice = (document.get("choices") or [{}])[0]
        positions = choice.get("prompt_logprobs")
        if not isinstance(positions, list):
            result["error"] = "response has no prompt_logprobs list (is --max-logprobs set?)"
            return result
        if prompt is not None:
            target_ids = list(prompt)
            target_source = "pinned token-id prompt"
        else:
            target_ids = choice.get("prompt_token_ids")
            target_source = "server-tokenised text prompt (response prompt_token_ids)"
        if not isinstance(target_ids, list) or len(target_ids) != len(positions):
            result["error"] = (
                "cannot resolve the prompt token ids for the prompt_logprobs positions "
                f"(ids={type(target_ids).__name__}, positions={len(positions)})"
            )
            return result
        result["prompt_token_ids"] = target_ids
        result["prompt_token_ids_source"] = target_source
        rows = []
        target_logprobs = []
        target_ranks = []
        for index, entry in enumerate(positions):
            if index == 0 or not isinstance(entry, dict):
                rows.append(
                    {"position": index, "target_token": None, "logprob": None, "rank": None}
                )
                continue
            target_token = target_ids[index]
            target = entry.get(str(target_token))
            if not isinstance(target, dict):
                rows.append(
                    {"position": index, "target_token": target_token, "logprob": None, "rank": None}
                )
                continue
            logprob = target.get("logprob")
            rank = target.get("rank")
            top = sorted(
                [
                    {"token_id": int(key), "logprob": value.get("logprob")}
                    for key, value in entry.items()
                    if isinstance(value, dict) and isinstance(value.get("logprob"), (int, float))
                ],
                key=lambda item: item["logprob"],
                reverse=True,
            )[: args.prompt_logprobs + 1]
            rows.append(
                {
                    "position": index,
                    "target_token": target_token,
                    "logprob": logprob,
                    "rank": rank,
                    "top": top,
                }
            )
            if isinstance(logprob, (int, float)):
                target_logprobs.append(float(logprob))
            if isinstance(rank, int):
                target_ranks.append(rank)
        top1_rate = (
            round(sum(1 for rank in target_ranks if rank == 1) / len(target_ranks), 4)
            if target_ranks
            else None
        )
        result["ok"] = True
        result["positions"] = rows
        result["aggregate"] = {
            "positions_measured": len(target_logprobs),
            "prompt_token_count": len(target_ids),
            "target_logprob_mean": round(statistics.fmean(target_logprobs), 4)
            if target_logprobs
            else None,
            "target_logprob_median": round(statistics.median(target_logprobs), 4)
            if target_logprobs
            else None,
            "target_logprob_min": round(min(target_logprobs), 4) if target_logprobs else None,
            "target_logprob_max": round(max(target_logprobs), 4) if target_logprobs else None,
            "target_rank_median": round(statistics.median(target_ranks), 1)
            if target_ranks
            else None,
            "target_rank_p90": percentile_nearest_rank([float(r) for r in target_ranks], 0.9),
            "target_rank_eq_1_fraction": top1_rate,
            "top1_rate": top1_rate,
            "label": (
                "SERVER-REPORTED logprobs (vLLM prompt_logprobs); ranks are over the full "
                "vocabulary for the returned top-k plus the target"
            ),
        }
        result["usage"] = document.get("usage")
        result["system_fingerprint"] = document.get("system_fingerprint")
    except Exception as error:  # noqa: BLE001 - recorded as a failed row
        result["ok"] = False
        result["error"] = describe_error(error)
    finally:
        result["load_after"] = snapshot_load(base_url, args.timeout)
    return result


def root_check(models_document: dict, expect_root: str) -> dict:
    """Compare the served root with the D38 released root; never raise on a mismatch."""
    entries = models_document.get("data") if isinstance(models_document.get("data"), list) else []
    entry = entries[0] if entries else {}
    served_root = entry.get("root") if isinstance(entry, dict) else None
    root_id = None
    if isinstance(served_root, str):
        root_id = served_root.rstrip("/").rsplit("/", 1)[-1]
    met = bool(root_id and expect_root and root_id == expect_root)
    return {
        "expected_d38_released_root": expect_root,
        "served_root": served_root,
        "served_root_id": root_id,
        "served_model": entry.get("id") if isinstance(entry, dict) else None,
        "met": met,
        "known_heretic_root": KNOWN_HERETIC_ROOT,
        "served_known_heretic_snapshot": root_id == KNOWN_HERETIC_ROOT,
        "consequence": (
            "served root matches the released snapshot: rows are parity-join candidates"
            if met
            else (
                "served root is NOT the released snapshot: this is a DEPLOYMENT baseline. It "
                "cannot be joined to the frozen D41/D38 released-root oracle; distributional "
                "parity needs a capture on the released snapshot, which needs a GPU window "
                "(vLLM restart under Lead authorization)."
            )
        ),
    }


def derive_load(load_before: dict, load_after: dict) -> dict:
    """Derived server-reported rates between two /metrics snapshots (window aggregates)."""
    derived = {}
    before = load_before.get("metrics", {})
    after = load_after.get("metrics", {})

    def delta(name: str) -> float | None:
        if name in before and name in after:
            return after[name] - before[name]
        return None

    def keep(name: str) -> None:
        derived[name.replace("vllm:", "")] = {
            "before": before.get(name),
            "after": after.get(name),
            "method": "server-reported /metrics snapshot value",
        }

    for name in (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
    ):
        keep(name)
    for name in (
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
        "vllm:time_to_first_token_seconds_count",
        "vllm:request_prefill_time_seconds_count",
        "vllm:request_decode_time_seconds_count",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    ):
        derived[name.replace("vllm:", "") + "_delta_over_window"] = {
            "value": delta(name),
            "method": "after - before over the capture window; not attributable to this client",
        }
    ttft_count = delta("vllm:time_to_first_token_seconds_count")
    ttft_sum = delta("vllm:time_to_first_token_seconds_sum")
    if ttft_count and ttft_sum is not None and ttft_count > 0:
        derived["mean_server_ttft_s_over_window"] = {
            "value": round(ttft_sum / ttft_count, 6),
            "method": "sum/count delta; aggregate over every request the engine finished",
        }
    prefill_count = delta("vllm:request_prefill_time_seconds_count")
    prefill_sum = delta("vllm:request_prefill_time_seconds_sum")
    if prefill_count and prefill_sum is not None and prefill_count > 0:
        derived["mean_server_prefill_s_over_window"] = {
            "value": round(prefill_sum / prefill_count, 6),
            "method": "sum/count delta; aggregate",
        }
    decode_count = delta("vllm:request_decode_time_seconds_count")
    decode_sum = delta("vllm:request_decode_time_seconds_sum")
    if decode_count and decode_sum is not None and decode_count > 0:
        derived["mean_server_decode_s_over_window"] = {
            "value": round(decode_sum / decode_count, 6),
            "method": "sum/count delta; aggregate",
        }
    drafts = delta("vllm:spec_decode_num_drafts_total")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
    if drafts and accepted is not None and drafts > 0:
        derived["server_reported_acceptance_length_over_window"] = {
            "value": round(accepted / drafts, 4),
            "method": "accepted_tokens / drafts delta; server-reported speculative decoding",
        }
    gen = delta("vllm:generation_tokens_total")
    wall = None
    try:
        start = datetime.strptime(load_before["captured_utc"], "%Y-%m-%dT%H:%M:%SZ")
        end = datetime.strptime(load_after["captured_utc"], "%Y-%m-%dT%H:%M:%SZ")
        wall = (end - start).total_seconds()
    except (KeyError, ValueError):
        wall = None
    if gen is not None and wall and wall > 0:
        derived["aggregate_generation_tokens_per_s_over_window"] = {
            "value": round(gen / wall, 3),
            "method": "generation token counter delta / wall seconds of the capture window",
        }
    return derived


def write_markdown(document: dict, out_md: str) -> None:
    """Write the readable table beside the JSON. The JSON is the evidence; this is the summary."""
    lines = []
    lines.append("# D38 vLLM live recapture - measured deployment baseline (2026-10-21)")
    lines.append("")
    lines.append(
        "Generated by scripts/d38_vllm_recapture.py; the JSON beside this file is the evidence of "
        "record. Every number is labelled MEASURED-CLIENT (this client's monotonic clock) or "
        "SERVER-REPORTED (the server's own response and /metrics)."
    )
    lines.append("")
    lines.append(
        "D52 is in force: this is a PERF row, not a gate row. No token-accuracy, tok/s, TTFT or "
        "gate-validity claim is made from it (docs/DECISIONS.md D52 item 3). The released-root "
        "parity programme is documented in docs/build/ws-g-vllm-parity-plan-20261021.md and stays "
        "parked until the Lead authorises a GPU window; this capture is the deployment reference "
        "that work will replace, not a substitute for it."
    )
    lines.append("")
    root = document["d38_root_precondition"]
    lines.append("## Root verdict (D38 precondition)")
    lines.append("")
    lines.append("| item | value |")
    lines.append("| :-- | :-- |")
    lines.append("| expected D38 released root | " + str(root["expected_d38_released_root"]) + " |")
    lines.append("| served root | " + str(root["served_root"]) + " |")
    lines.append("| root precondition met | **" + str(root["met"]) + "** |")
    lines.append("| gate eligible | **" + str(document["gate_eligible"]) + "** |")
    lines.append("")
    lines.append(root["consequence"])
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("| item | value |")
    lines.append("| :-- | :-- |")
    endpoint = document["endpoint"]
    lines.append("| endpoint | " + str(endpoint["base_url"]) + " |")
    lines.append("| model id | " + str(endpoint["model_id"]) + " |")
    lines.append("| served root | " + str(root["served_root"]) + " |")
    lines.append("| server version | " + str(endpoint.get("version")) + " |")
    lines.append("| system fingerprint | " + str(document.get("system_fingerprint")) + " |")
    lines.append("| capture started | " + document["capture"]["started_utc"] + " |")
    lines.append("| capture finished | " + document["capture"]["finished_utc"] + " |")
    lines.append("| source head | " + str(document["source"]["head"]) + " |")
    lines.append("| capture script sha256 | " + str(document["source"]["script_sha256"]) + " |")
    lines.append("| python | " + str(document["source"]["python"]) + " |")
    lines.append(
        "| co-tenant max running seen | "
        + str(document["load"]["max_num_requests_running_seen"])
        + " |"
    )
    lines.append("")
    argv_rows = [node for node in document["nodes"] if node.get("server_argv")]
    if argv_rows:
        lines.append("Server argv read read-only from the head node (ps, no restart):")
        lines.append("")
        lines.append("~~~text")
        lines.append(argv_rows[0]["server_argv"])
        lines.append("~~~")
        lines.append("")
    lines.append("## Co-tenant identification (who else holds the engine)")
    lines.append("")
    clients = document.get("co_tenant_identification", {})
    local = clients.get("local_clients", {})
    lines.append(
        "Read-only local socket evidence (ss -tnp), so the load row is attributed rather than "
        "guessed; nothing was stopped."
    )
    lines.append("")
    lines.append("| item | value |")
    lines.append("| :-- | :-- |")
    lines.append("| endpoint | " + str(local.get("endpoint")) + " |")
    lines.append("| local connection count | " + str(local.get("connection_count")) + " |")
    for client in local.get("clients", []):
        lines.append(
            "| local socket {} | process {} pid {}, {} connection(s) |".format(
                client.get("local"), client.get("process"), client.get("pid"), client.get("connections")
            )
        )
    note = clients.get("operator_note")
    if note:
        lines.append("| operator note | " + str(note) + " |")
    lines.append("")
    conditions = document.get("load_conditions", {})
    lines.append("## Load conditions")
    lines.append("")
    lines.append("| item | value |")
    lines.append("| :-- | :-- |")
    lines.append("| verdict | **" + str(conditions.get("verdict")) + "** |")
    lines.append("| quiet samples (idle at start) | " + str(conditions.get("quiet_sample_count")) + " |")
    lines.append(
        "| under-load samples (running > 0 at start) | "
        + str(conditions.get("under_load_sample_count"))
        + " |"
    )
    lines.append(
        "| max num_requests_running seen | "
        + str(conditions.get("max_num_requests_running_seen"))
        + " |"
    )
    idle = conditions.get("idle_wait", {})
    lines.append(
        "| idle wait | wait_idle_s {}, budget {} s, remaining {} s |".format(
            idle.get("wait_idle_s"), idle.get("budget_s"), idle.get("remaining_s")
        )
    )
    lines.append("")
    lines.append(str(conditions.get("note")))
    lines.append("")
    lines.append("## Measured rows (client-measured latency/rate)")
    lines.append("")
    lines.append(
        "| prompt | role | samples ok | TTFT p50 ms | TTFT p90 ms | decode tok/s p50 | decode "
        "tok/s p90 | wall p50 ms | completion tokens | greedy deterministic | max running |"
    )
    lines.append("| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- | --: |")
    for prompt in document["prompts"]:
        aggregate = prompt["aggregate"]
        ttft = aggregate["ttft_ms"]
        decode = aggregate["decode_tok_s"]
        wall = aggregate["wall_ms"]
        tokens = aggregate["completion_tokens"]["values"]
        lines.append(
            "| {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                prompt["id"],
                prompt["role"],
                aggregate["samples_ok"],
                aggregate["samples_planned"],
                ttft["p50"],
                ttft["p90"],
                decode["p50"],
                decode["p90"],
                wall["p50"],
                ",".join(str(value) for value in tokens) if tokens else "-",
                aggregate["greedy_deterministic_across_samples"],
                aggregate["co_tenant"]["max_num_requests_running_seen"],
            )
        )
    lines.append("")
    lines.append(
        "percentile method: nearest-rank over the ok samples (no interpolation). Recipe: "
        + json.dumps(document["recipe"], sort_keys=True)
    )
    lines.append("")
    lines.append("How to read these rows honestly:")
    lines.append("")
    lines.append(
        "- TTFT is MEASURED-CLIENT and includes HTTP connect, queueing and prefill, so a load "
        "spike (see a max of ~2 s in one sample) is contention, not a fixed cost."
    )
    lines.append(
        "- decode tok/s on the 9-token rows is measured over 8 token intervals, and speculative "
        "decoding (dspark, 5 draft tokens) delivers 1-3 tokens per stream chunk: the per-sample "
        "'max tok/chunk' column is the stream granularity. The 64-token steady row is the "
        "lower-variance decode estimate; neither row is a single-stream baseline while "
        "load_conditions.verdict is UNDER-LOAD."
    )
    lines.append(
        "- greedy_deterministic_across_samples can be false on the random-token fixture even at "
        "temperature 0: the fixture has flat near-tie ranks (task32-root-cause.md), so batch "
        "composition changes under co-tenant load flip near-ties. That is exactly why D39 forbids "
        "a token-equality gate; every sample's token ids are recorded."
    )
    lines.append("")
    lines.append("Per-sample rows:")
    lines.append("")
    lines.append(
        "| prompt | sample | ok | TTFT ms | decode tok/s | tokens | chunks | max tok/chunk | "
        "running before/after |"
    )
    lines.append("| :-- | --: | :-- | --: | --: | --: | --: | --: | :-- |")
    for prompt in document["prompts"]:
        for sample in prompt["samples"]:
            timings = sample["timings"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {}/{} |".format(
                    prompt["id"],
                    sample["sample_index"],
                    sample["ok"],
                    timings["ttft_ms"],
                    timings["decode_tok_s"],
                    timings["completion_tokens_server_reported"],
                    timings["stream_chunks_with_tokens"],
                    timings["max_tokens_per_chunk"],
                    sample["load_before"].get("num_requests_running"),
                    sample["load_after"].get("num_requests_running"),
                )
            )
    lines.append("")
    lines.append("## Greedy outputs (diagnostic, never a token-equality gate - D39)")
    lines.append("")
    lines.append("| prompt | first-ok greedy text (repr) | token ids | vs b69 released reference |")
    lines.append("| :-- | :-- | :-- | :-- |")
    for prompt in document["prompts"]:
        aggregate = prompt["aggregate"]
        reference = aggregate.get("vs_b69_nine_step_reference")
        detail = (
            "{} / {} positions match".format(
                reference["positions_matching"], reference["positions_compared"]
            )
            if reference
            else "no reference for this prompt (diagnostic only)"
        )
        text = (aggregate["greedy_text_first_ok_sample"] or "").replace("|", "\\|")
        lines.append(
            "| {} | {} | {} | {} |".format(
                prompt["id"],
                repr(text[:80]),
                aggregate["greedy_token_ids_first_ok_sample"],
                detail,
            )
        )
    lines.append("")
    lines.append("## Prompt self-consistency: random-token fixture vs natural language")
    lines.append("")
    lines.append(
        "The D41/D48 130-token fixture prompt is 130 RANDOM token ids (seed 7; task32-root-cause.md "
        "says so explicitly). On a random-token continuation a correct model assigns the actual "
        "next token a low logprob and a large rank, so the fixture row below is EXPECTED behaviour "
        "and is NOT evidence of degeneracy or of an unusable oracle. The natural-language rows are "
        "the measured test of whether the served checkpoint predicts its own text."
    )
    lines.append("")
    lines.append(
        "| row | prompt nature | positions | target logprob mean | median | target rank median | "
        "target rank p90 | target rank == 1 (top-1 rate) |"
    )
    lines.append("| :-- | :-- | --: | --: | --: | --: | --: | --: |")
    prompt_rows = [("fixture", document.get("prompt_logprobs"))] + [
        ("natural", row) for row in document.get("prompt_logprobs_natural") or []
    ]
    for kind, row in prompt_rows:
        name = (row or {}).get("prompt_id") or kind
        nature = (
            "random-token fixture (EXPECTED low)"
            if kind == "fixture"
            else "natural language"
        )
        if not row or not row.get("ok"):
            lines.append(
                "| {} | {} | - | - | - | - | - | unavailable: {} |".format(
                    name, nature, str((row or {}).get("error", "not attempted"))
                )
            )
            continue
        aggregate = row["aggregate"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                nature,
                aggregate["positions_measured"],
                aggregate["target_logprob_mean"],
                aggregate["target_logprob_median"],
                aggregate["target_rank_median"],
                aggregate["target_rank_p90"],
                aggregate["target_rank_eq_1_fraction"],
            )
        )
    lines.append("")
    natural = document.get("natural_language_self_consistency", {})
    lines.append(
        "Natural-language self-consistency verdict: **{}**".format(natural.get("verdict"))
    )
    lines.append("")
    lines.append(str(natural.get("criterion")))
    lines.append("")
    lines.append(
        "The fixture row is kept to show the difference; do not read it as a degeneracy claim. A "
        "'UNUSABLE-AS-DISTRIBUTIONAL-ORACLE' verdict would need every natural-language row to show "
        "median target rank >= 10000 and top-1 rate < 0.10."
    )
    lines.append("")
    lines.append("## Server-reported aggregates (/metrics, aggregate lifetime - not per request)")
    lines.append("")
    before = document["load"]["before_run"].get("metrics", {})
    after = document["load"]["after_run"].get("metrics", {})
    lines.append("| metric | before | after |")
    lines.append("| :-- | --: | --: |")
    for key in sorted(set(before) | set(after)):
        lines.append("| {} | {} | {} |".format(key, before.get(key), after.get(key)))
    lines.append("")
    lines.append("Derived server-reported values (arithmetic on the counters above):")
    lines.append("")
    lines.append("| value | value measured | method |")
    lines.append("| :-- | --: | :-- |")
    for label, entry in sorted(document["load"]["derived"].items()):
        if isinstance(entry, dict) and "value" in entry:
            lines.append("| {} | {} | {} |".format(label, entry.get("value"), entry.get("method")))
        elif isinstance(entry, dict):
            lines.append(
                "| {} | before {} -> after {} | {} |".format(
                    label, entry.get("before"), entry.get("after"), entry.get("method")
                )
            )
        else:
            lines.append("| {} | {} | - |".format(label, entry))
    lines.append("")
    lines.append(
        "These are server lifetime/window aggregates. This capture shares the engine with "
        "co-tenant streams (see num_requests_running), so they are NOT attributed to these "
        "samples; the per-request rows above are the client-measured evidence."
    )
    lines.append("")
    lines.append("## What this capture does NOT claim")
    lines.append("")
    for item in document["non_claims"]:
        lines.append("- " + item)
    lines.append("")
    lines.append("Evidence of record: " + str(document["evidence"]["json_path"]))
    lines.append("")
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI: every knob that affects the evidence is explicit."""
    parser = argparse.ArgumentParser(description="D38 vLLM recapture")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served model id")
    parser.add_argument(
        "--expect-root",
        default=D38_RELEASED_ROOT,
        help="D38 released snapshot id the capture should be comparable to",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sampling seed")
    parser.add_argument(
        "--logprobs", type=int, default=DEFAULT_LOGPROBS, help="top logprobs per generated token"
    )
    parser.add_argument(
        "--prompt-logprobs",
        type=int,
        default=5,
        help="top logprobs per prompt position for the self-consistency row (0 disables)",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="request timeout s")
    parser.add_argument(
        "--ssh-hosts",
        default="192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68",
        help="comma-separated nodes for read-only provenance (argv/GPU); empty disables",
    )
    parser.add_argument("--out-json", required=True, help="evidence JSON to write")
    parser.add_argument("--out-md", required=True, help="markdown summary to write")
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="override samples per prompt (0 = protocol default; >=3 for a claim)",
    )
    parser.add_argument("--only", default="", help="capture only this protocol prompt id")
    parser.add_argument(
        "--wait-idle-s",
        type=float,
        default=0.0,
        help="before each sample, poll /metrics up to this many seconds for num_requests_running==0",
    )
    parser.add_argument(
        "--idle-budget-s",
        type=float,
        default=0.0,
        help="total idle-wait budget across the run (0 = wait_idle_s * planned samples)",
    )
    parser.add_argument(
        "--natural-prompts",
        type=int,
        default=3,
        help="natural-language prompt_logprobs rows (0 disables); the random fixture is not a test",
    )
    parser.add_argument(
        "--co-tenant-note",
        default="",
        help="operator note recorded with the local socket-owner evidence",
    )
    parser.add_argument("--dry-run", action="store_true", help="build and print the plan, no HTTP")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started = utc_now()

    prompts = []
    for spec in PROMPT_SPECS:
        try:
            if spec["source"]["kind"] == "fixture":
                tokens, provenance = load_fixture_prompt(spec["source"]["path"])
            else:
                tokens, provenance = load_d41_prompt(
                    spec["source"]["path"],
                    spec["source"]["block_id"],
                    spec["source"]["request_id"],
                )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"ERROR: cannot load prompt {spec['id']}: {error}", file=sys.stderr)
            return 2
        prompts.append((spec, tokens, provenance))

    if args.only:
        prompts = [entry for entry in prompts if entry[0]["id"] == args.only]
        if not prompts:
            print(f"ERROR: --only {args.only!r} does not name a protocol prompt", file=sys.stderr)
            return 2
    planned_samples = sum((args.samples or spec["samples"]) for spec, _t, _p in prompts)
    idle_budget = (
        args.idle_budget_s if args.idle_budget_s > 0 else args.wait_idle_s * max(1, planned_samples)
    )
    idle_state = {
        "remaining": idle_budget,
        "initial": idle_budget,
        "planned_samples": planned_samples,
    }

    if args.dry_run:
        plan = {
            "base_url": args.base_url,
            "model": args.model,
            "prompts": [
                {
                    "id": spec["id"],
                    "role": spec["role"],
                    "prompt_len": len(tokens),
                    "prompt_sha256": provenance["prompt_sha256"],
                    "max_tokens": spec["max_tokens"],
                    "samples": args.samples or spec["samples"],
                    "payload": build_payload(
                        args.model, tokens, spec["max_tokens"], args.seed, args.logprobs
                    ),
                }
                for spec, tokens, provenance in prompts
            ],
            "prompt_logprobs": args.prompt_logprobs,
            "natural_prompts": args.natural_prompts,
            "wait_idle_s": args.wait_idle_s,
            "idle_budget_s": idle_budget,
            "requests": planned_samples
            + (1 if args.prompt_logprobs else 0)
            + (min(args.natural_prompts, len(NATURAL_PROMPTS)) if args.prompt_logprobs else 0),
        }
        print(json.dumps(plan, indent=2)[:8000])
        return 3

    run_id = compact_utc()
    print(f"D38 vLLM recapture run {run_id} against {args.base_url}", flush=True)
    try:
        models_document = http_json(args.base_url.rstrip("/") + "/v1/models", 30.0)
    except Exception as error:  # noqa: BLE001 - no models endpoint is fatal for provenance
        print(f"FATAL: cannot read /v1/models: {describe_error(error)}", file=sys.stderr)
        return 2
    version = None
    version_error = None
    for version_path in ("/version", "/v1/version"):
        try:
            version_document = http_json(args.base_url.rstrip("/") + version_path, 15.0)
            version = version_document.get("version")
            version_error = None
            break
        except Exception as error:  # noqa: BLE001 - version is optional provenance
            version_error = describe_error(error)

    script_path = os.path.abspath(__file__)
    source = {
        "script": script_path,
        "script_sha256": sha256_file(script_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": git_provenance(),
    }
    source["head"] = source["git"]["head"]
    nodes = collect_node_provenance(
        [host.strip() for host in args.ssh_hosts.split(",") if host.strip()],
        DEFAULT_SSH_TIMEOUT_S,
    )
    load_before = snapshot_load(args.base_url, args.timeout)
    print(
        "load before: num_requests_running={} waiting={}".format(
            load_before.get("num_requests_running"),
            load_before.get("metrics", {}).get("vllm:num_requests_waiting"),
        ),
        flush=True,
    )

    captured_prompts = []
    for spec, tokens, provenance in prompts:
        effective = dict(spec)
        if args.samples:
            effective["samples"] = args.samples
        print(
            "capturing {} ({} prompt tokens, max_tokens {}, {} samples)".format(
                spec["id"], len(tokens), spec["max_tokens"], effective["samples"]
            ),
            flush=True,
        )
        try:
            captured_prompts.append(
                capture_prompt(
                    args.base_url, args.model, effective, tokens, provenance, args, idle_state
                )
            )
        except Exception as error:  # noqa: BLE001 - a prompt-level failure is evidence
            print(f"ERROR capturing {spec['id']}: {describe_error(error)}", file=sys.stderr)
            captured_prompts.append(
                {
                    "id": spec["id"],
                    "role": spec["role"],
                    "label": spec["label"],
                    "prompt_source": provenance,
                    "max_tokens": spec["max_tokens"],
                    "error": describe_error(error),
                    "samples": [],
                    "aggregate": {"samples_planned": effective["samples"], "samples_ok": 0},
                }
            )

    prompt_logprobs = None
    natural_prompt_logprobs = []
    if args.prompt_logprobs:
        canonical, canonical_tokens, canonical_provenance = prompts[0]
        print(f"capturing prompt_logprobs row for {canonical['id']}", flush=True)
        prompt_logprobs = capture_prompt_logprobs(
            args.base_url,
            args.model,
            canonical_tokens,
            args,
            prompt_id=canonical["id"],
            prompt_nature=canonical_provenance.get("prompt_nature"),
        )
        for natural in NATURAL_PROMPTS[: args.natural_prompts]:
            print(f"capturing natural-language prompt_logprobs row {natural['id']}", flush=True)
            natural_prompt_logprobs.append(
                capture_prompt_logprobs(
                    args.base_url,
                    args.model,
                    None,
                    args,
                    prompt_text=natural["text"],
                    prompt_id=natural["id"],
                    prompt_nature=(
                        "natural-language text prompt (the test the random-token fixture cannot "
                        "provide)"
                    ),
                )
            )
    co_tenant = {
        "local_clients": local_endpoint_clients(args.base_url, args.timeout),
        "operator_note": args.co_tenant_note,
        "label": "MEASURED-LOCAL (ss socket owners) + operator note: who holds the engine",
    }

    natural_rows = [row for row in natural_prompt_logprobs if row.get("ok")]
    natural_verdict = {
        "rows_ok": len(natural_rows),
        "criterion": (
            "A natural-language row with median target rank >= 10000 and top-1 rate < 0.10 means "
            "the served checkpoint does not reproduce its own text continuation. The random-token "
            "fixture row is excluded from this test by construction."
        ),
        "median_target_rank_values": [
            row["aggregate"]["target_rank_median"] for row in natural_rows
        ],
        "top1_rate_values": [row["aggregate"]["target_rank_eq_1_fraction"] for row in natural_rows],
    }
    if natural_rows:
        degraded = all(
            (row["aggregate"]["target_rank_median"] or 0) >= 10000
            and (row["aggregate"]["target_rank_eq_1_fraction"] or 0) < 0.10
            for row in natural_rows
        )
        normal = all(
            (row["aggregate"]["target_rank_median"] or 10**9) < 1000 for row in natural_rows
        )
        natural_verdict["verdict"] = (
            "UNUSABLE-AS-DISTRIBUTIONAL-ORACLE"
            if degraded
            else ("NATURAL-LANGUAGE-SELF-PREDICTION-NORMAL" if normal else "INCONCLUSIVE")
        )
    else:
        natural_verdict["verdict"] = "NOT-MEASURED"

    load_after = snapshot_load(args.base_url, args.timeout)
    finished = utc_now()
    root = root_check(models_document, args.expect_root)
    running_values = [
        sample["load_before"].get("num_requests_running")
        for prompt in captured_prompts
        for sample in prompt.get("samples", [])
    ] + [
        sample["load_after"].get("num_requests_running")
        for prompt in captured_prompts
        for sample in prompt.get("samples", [])
    ]
    running_values = [value for value in running_values if isinstance(value, (int, float))]
    max_running = max(running_values) if running_values else None

    samples_total = sum(len(prompt.get("samples", [])) for prompt in captured_prompts)
    samples_ok = sum(
        1
        for prompt in captured_prompts
        for sample in prompt.get("samples", [])
        if sample.get("ok")
    )
    failed_samples = samples_total - samples_ok
    gate_eligible = bool(root["met"]) and failed_samples == 0 and samples_ok >= 3 and not max_running
    reasons = []
    if not root["met"]:
        reasons.append(
            "served root is not the D38 released root, so rows cannot be joined to the frozen "
            "released-root oracle (D38 precondition NOT MET)"
        )
    if failed_samples:
        reasons.append(f"{failed_samples} of {samples_total} samples failed")
    if samples_ok < 3:
        reasons.append("fewer than three ok samples: no p50/p90 claim")
    if max_running:
        reasons.append(
            "co-tenant load observed (num_requests_running up to {}); latency is under-load, not "
            "a single-stream baseline".format(max_running)
        )
    quiet_samples = sum(
        1
        for prompt in captured_prompts
        for sample in prompt.get("samples", [])
        if isinstance(sample.get("idle_wait"), dict)
        and sample["idle_wait"].get("idle_at_start") is True
    )
    under_load_samples = sum(
        1
        for prompt in captured_prompts
        for sample in prompt.get("samples", [])
        if isinstance(sample.get("load_before", {}).get("num_requests_running"), (int, float))
        and sample["load_before"]["num_requests_running"] > 0
    )
    if max_running is None:
        load_verdict = "UNKNOWN"
    elif max_running == 0:
        load_verdict = "QUIET"
    elif quiet_samples:
        load_verdict = "MIXED"
    else:
        load_verdict = "UNDER-LOAD"
    load_conditions = {
        "verdict": load_verdict,
        "quiet_sample_count": quiet_samples,
        "under_load_sample_count": under_load_samples,
        "max_num_requests_running_seen": max_running,
        "idle_wait": {
            "wait_idle_s": args.wait_idle_s,
            "budget_s": idle_state["initial"],
            "remaining_s": idle_state["remaining"],
        },
        "note": "UNDER-LOAD rows are latency with contention, NOT a single-stream baseline",
    }
    document = {
        "schema": SCHEMA,
        "generated_utc": finished,
        "run_id": run_id,
        "title": "D38 vLLM live recapture - measured deployment baseline",
        "endpoint": {
            "base_url": args.base_url,
            "completions_url": args.base_url.rstrip("/") + "/v1/completions",
            "model_id": root.get("served_model"),
            "version": version,
            "version_error": version_error,
            "models_document": models_document,
        },
        "d38_root_precondition": root,
        "gate_eligible": gate_eligible,
        "gate_eligible_reasons": reasons,
        "source": source,
        "nodes": nodes,
        "capture": {"started_utc": started, "finished_utc": finished, "dry_run": False},
        "recipe": {
            "endpoint": "/v1/completions",
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "ignore_eos": True,
            "seed": args.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            "logprobs": args.logprobs,
            "return_token_ids": True,
            "percentile_method": "nearest-rank, no interpolation",
        },
        "prompts": captured_prompts,
        "prompt_logprobs": prompt_logprobs,
        "prompt_logprobs_natural": natural_prompt_logprobs,
        "natural_language_self_consistency": natural_verdict,
        "co_tenant_identification": co_tenant,
        "load_conditions": load_conditions,
        "load": {
            "before_run": load_before,
            "after_run": load_after,
            "max_num_requests_running_seen": max_running,
            "derived": derive_load(load_before, load_after),
            "note": (
                "server-reported /metrics gauges and counters; lifetime aggregates, not attributed "
                "to this client's requests"
            ),
        },
        "system_fingerprint": next(
            (
                sample["timings"].get("system_fingerprint")
                for prompt in captured_prompts
                for sample in prompt.get("samples", [])
                if sample.get("timings", {}).get("system_fingerprint")
            ),
            None,
        ),
        "sample_counts": {
            "requests_total": samples_total,
            "requests_ok": samples_ok,
            "requests_failed": failed_samples,
            "prompt_logprobs_requests": (1 if args.prompt_logprobs else 0)
            + len(natural_prompt_logprobs),
        },
        "evidence": {"json_path": os.path.abspath(args.out_json)},
        "non_claims": [
            "no SparkFlash number: this host has no GPU and the deployment holds the cards (see "
            "node GPU rows), so every SparkFlash comparison row is NOT-MEASURED here",
            "no token-equality gate (D39): reference-token agreement is a diagnostic only",
            "no clean single-stream baseline when load_conditions.verdict is UNDER-LOAD: "
            "num_requests_running is recorded around every request",
            "no D41/D48 gate: gate_eligible is false when the served root is not the released "
            "snapshot, and no D41 bars file exists",
            "the 130-token fixture prompt is 130 RANDOM token ids (seed 7): a low target logprob "
            "or a huge target rank on it is EXPECTED of a correct model and is NOT evidence of "
            "degeneracy; only the natural-language prompt_logprobs rows test that",
            "server-reported /metrics aggregates are lifetime/window values, not per-request "
            "measurements of this client's traffic",
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=False)
        handle.write("\n")
    write_markdown(document, args.out_md)

    print("")
    print(f"wrote {os.path.abspath(args.out_json)}")
    print(f"wrote {os.path.abspath(args.out_md)}")
    print(f"root precondition met: {root['met']} (served {root['served_root_id']})")
    print(f"gate_eligible: {gate_eligible}")
    for reason in reasons:
        print(f"  not gate-eligible: {reason}")
    if not root["met"]:
        print(
            "LABEL: LIVE-DEPLOYMENT-BASELINE (not a D38/D41 parity row; released-root capture "
            "needs a GPU window)"
        )
    return 1 if failed_samples else 0


if __name__ == "__main__":
    sys.exit(main())
