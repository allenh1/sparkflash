#!/usr/bin/env python3
"""Measured S4 host continuous-batching A/B census for SparkFlash serve.

This is a HOST PATH census of two serving modes of the same binary over the
synthetic tiny plan (128-row vocabulary, no checkpoint, no device, no CUDA):

  serial   --synthetic                 (the default: one request at a time)
  batched  --synthetic --continuous-batching --max-batch N

It measures the same streaming concurrency ladder (1/8/32 concurrent clients)
against each mode and records client-measured TTFT, end-to-end latency, aggregate
completion tokens/s, accepted/refused/failed counts, and the server counters.
It also sends a fixed set of non-streaming requests through each mode and records
their exact generated text, so the two modes can be compared for output identity
on the same token-id prompts.

Every number is a host-path number on the reference CPU engine and is explicitly
UNBARED / diagnostic (no frozen floor; D41): it supports regression tracking and
S4 parity triage, and it is not a device result and never a vLLM comparison. The
device arm's batched attention, paged KV, TTFT and throughput are window
measurements and nothing here speaks to them.

The script:
  1. starts the production serve binary on 127.0.0.1:0 (loopback only; the live
     vLLM oracle is never contacted and no code path can address it),
  2. runs the streaming ladder in each mode,
  3. runs the fixed non-streaming probes in each mode and compares generated text
     byte for byte across modes,
  4. scrapes /metrics before and after the workload and checks the ops families
     against the summed response usage: the generation-token counter delta must
     equal the completion tokens every successful request reported, the TTFT
     histogram count delta must equal the number of token-producing requests,
     and the scheduler gauges must be zero at idle,
  5. runs wire probes (buffered prompt_logprobs, buffered and SSE stop, SSE sample
     logprobs, and the streaming prompt_logprobs refusal) and compares their
     payload shape and stable values across modes,
  6. writes serve-s4-batching-census-<utc>.json and .md, a per-mode server log,
     and a sha256 file pinning the binary and every artefact, under the evidence
     directory (default /home/user/ev-session-20261020/serve/).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "sparkflash-serve-s4-batching-census/3"
LABEL = "host path, synthetic tiny plan (vocab 128), UNBARED / diagnostic"
MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_EVIDENCE_DIR = Path("/home/user/ev-session-20261020/serve")
PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]
# The tiny plan's only generated token is the BOS special token, so text is visible only with the
# explicit false passthrough. TTFT is measured on the first text delta, which needs that text.
SKIP_SPECIAL_FALSE = False


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def percentile(samples: list[float], p: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "n": len(samples),
        "min": min(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "max": max(samples),
        "mean": sum(samples) / len(samples),
    }


def round_floats(value: Any) -> Any:
    if isinstance(value, float):
        if value == 0.0 or value != value:
            return value
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


def host_load() -> dict[str, Any]:
    try:
        load = list(os.getloadavg())
    except (OSError, AttributeError):
        load = []
    return {
        "cpu_count": os.cpu_count(),
        "loadavg_1_5_15": load,
        "load_1m_per_cpu": (load[0] / os.cpu_count()) if load and os.cpu_count() else None,
    }


class Server:
    """One serve process on a kernel-picked loopback port, in the requested mode."""

    def __init__(self, binary: Path, extra_args: list[str], log_path: Path) -> None:
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        self._log_handle = log_path.open("a")
        self.log_lines: list[str] = []
        self._lock = threading.Lock()
        command = [
            str(binary),
            "serve",
            "--synthetic",
            "--bind",
            "127.0.0.1:0",
            *extra_args,
        ]
        self.command = command
        self._proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.base: str | None = None
        for pipe in (self._proc.stdout, self._proc.stderr):
            threading.Thread(target=self._drain, args=(pipe,), daemon=True).start()
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            with self._lock:
                for line in self.log_lines:
                    marker = "listening on "
                    if marker in line:
                        self.base = line.split(marker, 1)[1].strip()
            if self.base:
                break
            if self._proc.poll() is not None:
                break
            time.sleep(0.05)
        if not self.base:
            self.stop()
            raise SystemExit("the server did not report its address; log:\n" + self.log())

    def _drain(self, pipe: Any) -> None:
        for line in pipe:
            line = line.rstrip("\n")
            with self._lock:
                self.log_lines.append(line)
            self._log_handle.write(line + "\n")
            self._log_handle.flush()

    def log(self) -> str:
        with self._lock:
            return "\n".join(self.log_lines)

    def dspark_evidence_lines(self) -> int:
        return sum(1 for line in self.log().splitlines() if "accept_status=" in line)

    def host_port(self) -> tuple[str, int]:
        assert self.base is not None
        stripped = self.base.removeprefix("http://")
        host, _, port = stripped.partition(":")
        return host, int(port)

    def stop(self) -> None:
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)
        self._log_handle.close()


def get_json(server: Server, path: str, timeout: float = 30.0) -> tuple[int, Any]:
    host, port = server.host_port()
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        raw = response.read()
        try:
            body: Any = json.loads(raw)
        except json.JSONDecodeError:
            body = raw.decode(errors="replace")
        return response.status, body
    finally:
        connection.close()


def metric_values(server: Server) -> dict[str, float]:
    host, port = server.host_port()
    values: dict[str, float] = {}
    connection = http.client.HTTPConnection(host, port, timeout=30.0)
    try:
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        text = response.read().decode(errors="replace")
    finally:
        connection.close()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, raw = line.partition(" ")
        name = name.split("{")[0]
        try:
            values[name] = float(raw)
        except ValueError:
            continue
    return values


OPS_FAMILIES = [
    "sparkflash_num_requests_running",
    "sparkflash_num_requests_waiting",
    "sparkflash_generation_tokens_total",
    "sparkflash_time_to_first_token_seconds",
]
OPS_SAMPLES = [
    "sparkflash_num_requests_running",
    "sparkflash_num_requests_waiting",
    "sparkflash_generation_tokens_total",
    "sparkflash_requests_total",
    "sparkflash_time_to_first_token_seconds_sum",
    "sparkflash_time_to_first_token_seconds_count",
]
VOLATILE_KEYS = frozenset({"id", "created", "timing"})


def get_text(server: Server, path: str, timeout: float = 30.0) -> tuple[int, str]:
    host, port = server.host_port()
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.read().decode(errors="replace")
    finally:
        connection.close()


def parse_ops_metrics(text: str) -> dict[str, Any]:
    """Parses the Prometheus text exposition for the ops families and the TTFT histogram.

    Values are kept exactly as exported, without rounding: this is the measured side of
    the accounting check, and a rounded counter could hide a one-token mismatch.
    """
    help_by_family: dict[str, str] = {}
    type_by_family: dict[str, str] = {}
    samples: dict[str, float] = {}
    buckets: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("# HELP "):
            rest = line.removeprefix("# HELP ")
            family, _, help_text = rest.partition(" ")
            help_by_family[family] = help_text
            continue
        if line.startswith("# TYPE "):
            rest = line.removeprefix("# TYPE ")
            family, _, type_name = rest.partition(" ")
            type_by_family[family] = type_name
            continue
        if line.startswith("#"):
            continue
        sample, _, raw = line.partition(" ")
        try:
            value = float(raw)
        except ValueError:
            continue
        if sample.startswith("sparkflash_time_to_first_token_seconds_bucket{"):
            label = sample.split('le="', 1)[1].split('"', 1)[0]
            buckets.append({"le": label, "count": value})
        elif sample.split("{", 1)[0] in OPS_SAMPLES:
            samples[sample] = value

    def bucket_order(bucket: dict[str, Any]) -> float:
        label = bucket["le"]
        return float("inf") if label == "+Inf" else float(label)

    return {
        "present_families": sorted(set(help_by_family) | set(type_by_family)),
        "ops_family_help": {name: help_by_family.get(name) for name in OPS_FAMILIES},
        "ops_family_type": {name: type_by_family.get(name) for name in OPS_FAMILIES},
        "sample_values": samples,
        "ttft_buckets": sorted(buckets, key=bucket_order),
    }


def ops_metrics(server: Server) -> dict[str, Any]:
    status, text = get_text(server, "/metrics")
    parsed = parse_ops_metrics(text)
    parsed["status"] = status
    return parsed


def strip_volatile(value: Any) -> Any:
    """Drops fields two separate server processes legitimately differ in.

    Request ids and wall-clock creation times are per-process; the timing block is a
    measurement, not payload semantics. Everything else must be identical across modes.
    """
    if isinstance(value, dict):
        return {
            key: strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


def payload_shape(value: Any) -> Any:
    """The key-and-type skeleton of a payload, as a JSON-comparable value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return {
            "array_len": len(value),
            "item": payload_shape(value[0]) if value else "none",
        }
    return {key: payload_shape(item) for key, item in sorted(value.items())}


def stream_completion(server: Server, body: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    """One streaming completion, with client-measured TTFT and end-to-end latency."""
    host, port = server.host_port()
    payload = json.dumps(body).encode()
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    ttft: float | None = None
    frames = 0
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    done = False
    status = None
    error = None
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        response = connection.getresponse()
        status = response.status
        if status != 200:
            raw = response.read()
            try:
                error = json.loads(raw).get("error", {}).get("message")
            except json.JSONDecodeError:
                error = raw.decode(errors="replace")[:200]
            return {
                "status": status,
                "error": error,
                "ttft_s": None,
                "elapsed_s": time.perf_counter() - started,
                "frames": 0,
                "text": "",
                "usage": None,
                "done": False,
            }
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                done = True
                break
            try:
                frame = json.loads(data)
            except json.JSONDecodeError:
                continue
            frames += 1
            if frame.get("usage"):
                usage = frame["usage"]
            choices = frame.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                text_parts.append(text)
                if ttft is None:
                    ttft = time.perf_counter() - started
    finally:
        connection.close()
    return {
        "status": status,
        "error": error,
        "ttft_s": ttft,
        "elapsed_s": time.perf_counter() - started,
        "frames": frames,
        "text": "".join(text_parts),
        "usage": usage,
        "done": done,
    }


def fixed_probe(server: Server, prompt: list[int], max_tokens: int, timeout: float = 120.0) -> dict[str, Any]:
    """One buffered completion, whose exact text two modes must reproduce."""
    host, port = server.host_port()
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "skip_special_tokens": SKIP_SPECIAL_FALSE,
    }
    payload = json.dumps(body).encode()
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        response = connection.getresponse()
        raw = response.read()
        status = response.status
        elapsed = time.perf_counter() - started
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        choice = (parsed.get("choices") or [{}])[0]
        return {
            "status": status,
            "elapsed_s": elapsed,
            "text": choice.get("text"),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (parsed.get("usage") or {}).get("completion_tokens"),
        }
    finally:
        connection.close()



def post_json(
    server: Server, body: dict[str, Any], timeout: float = 120.0
) -> tuple[int, str, Any, float]:
    """One buffered POST to /v1/completions.

    Returns (status, content-type, parsed JSON, client-measured elapsed seconds). The elapsed
    number is the client's wall clock around the whole HTTP exchange, so it is a diagnostic host
    path timing (loopback included), not a server-internal measure.
    """
    host, port = server.host_port()
    payload = json.dumps(body).encode()
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    started = time.perf_counter()
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        response = connection.getresponse()
        content_type = response.getheader("Content-Type") or ""
        raw = response.read()
        elapsed = time.perf_counter() - started
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw.decode(errors="replace")[:400]}
        return response.status, content_type, parsed, elapsed
    finally:
        connection.close()


def buffered_probe(server: Server, body: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    """One buffered wire probe, recorded as the stable payload both modes must agree on."""
    status, content_type, parsed, _elapsed = post_json(server, body, timeout=timeout)
    payload = strip_volatile(parsed)
    choice = {}
    usage: dict[str, Any] = {}
    if isinstance(parsed, dict):
        choices = parsed.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
        usage = parsed.get("usage") or {}
    return {
        "kind": "buffered",
        "status": status,
        "content_type": content_type,
        "payload": payload,
        "shape": payload_shape(payload),
        "text": choice.get("text"),
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def sse_probe(server: Server, body: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    """One SSE wire probe with include_usage, recorded frame by frame."""
    host, port = server.host_port()
    request_body = dict(body)
    request_body.setdefault("stream_options", {"include_usage": True})
    payload = json.dumps(request_body).encode()
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    frames: list[Any] = []
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    done = False
    status = None
    content_type = ""
    try:
        connection.request(
            "POST",
            "/v1/completions",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        response = connection.getresponse()
        status = response.status
        content_type = response.getheader("Content-Type") or ""
        if status != 200:
            raw = response.read()
            try:
                parsed: Any = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"raw": raw.decode(errors="replace")[:400]}
            return {
                "kind": "sse",
                "status": status,
                "content_type": content_type,
                "frames": [],
                "frame_shapes": [],
                "done": False,
                "text": "",
                "usage": None,
                "completion_tokens": 0,
                "prompt_logprobs_in_frames": False,
                "payload": strip_volatile(parsed),
                "shape": payload_shape(strip_volatile(parsed)),
            }
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                done = True
                break
            try:
                frame = json.loads(data)
            except json.JSONDecodeError:
                continue
            frames.append(frame)
            if frame.get("usage"):
                usage = frame["usage"]
            choices = frame.get("choices") or []
            if choices and choices[0].get("text"):
                text_parts.append(choices[0]["text"])
    finally:
        connection.close()
    stable_frames = [strip_volatile(frame) for frame in frames]
    stable_usage = strip_volatile(usage) if isinstance(usage, dict) else None
    return {
        "kind": "sse",
        "status": status,
        "content_type": content_type,
        "frames": stable_frames,
        "frame_shapes": [payload_shape(frame) for frame in stable_frames],
        "done": done,
        "text": "".join(text_parts),
        "usage": stable_usage,
        "completion_tokens": int((stable_usage or {}).get("completion_tokens") or 0),
        "prompt_logprobs_in_frames": any(
            "prompt_logprobs" in json.dumps(frame) for frame in stable_frames
        ),
    }


def wire_probes(
    server: Server, prompt: list[int], stop: str, timeout: float = 120.0
) -> list[dict[str, Any]]:
    """The prompt_logprobs and stop probes whose payload shape and values must not change.

    Each probe is a real client request; the buffered ones and the SSE ones with usage
    contribute their completion tokens to the ops-metrics accounting below.
    """
    probes: list[dict[str, Any]] = []

    def record(name: str, body: dict[str, Any], streaming: bool) -> None:
        outcome = (
            sse_probe(server, body, timeout=timeout)
            if streaming
            else buffered_probe(server, body, timeout=timeout)
        )
        outcome["name"] = name
        probes.append(outcome)

    base = {
        "prompt": prompt,
        "max_tokens": 3,
        "temperature": 0.0,
        "skip_special_tokens": SKIP_SPECIAL_FALSE,
    }
    record("buffered_prompt_logprobs_0", {**base, "prompt_logprobs": 0}, streaming=False)
    record("buffered_prompt_logprobs_1", {**base, "prompt_logprobs": 1}, streaming=False)
    record(
        "buffered_stop",
        {
            "prompt": prompt,
            "max_tokens": 12,
            "temperature": 0.0,
            "skip_special_tokens": SKIP_SPECIAL_FALSE,
            "stop": stop,
        },
        streaming=False,
    )
    record(
        "sse_stop",
        {
            "prompt": prompt,
            "max_tokens": 12,
            "temperature": 0.0,
            "skip_special_tokens": SKIP_SPECIAL_FALSE,
            "stop": stop,
            "stream": True,
        },
        streaming=True,
    )
    record(
        "sse_logprobs",
        {
            "prompt": prompt,
            "max_tokens": 3,
            "temperature": 0.0,
            "skip_special_tokens": SKIP_SPECIAL_FALSE,
            "logprobs": 2,
            "stream": True,
        },
        streaming=True,
    )
    record(
        "sse_prompt_logprobs_refused",
        {
            "prompt": prompt,
            "max_tokens": 3,
            "temperature": 0.0,
            "prompt_logprobs": 1,
            "stream": True,
        },
        streaming=True,
    )
    record(
        "sse_prompt_logprobs_zero",
        {
            "prompt": prompt,
            "max_tokens": 3,
            "temperature": 0.0,
            "prompt_logprobs": 0,
            "stream": True,
        },
        streaming=True,
    )
    return probes


def prompt_logprobs_payload_checks(payload: Any, prompt: list[int]) -> dict[str, Any]:
    """Checks a buffered prompt_logprobs=1 payload position by position.

    The contract: one entry per prompt token, position 0 null (no conditioning token), and every
    later entry an id-keyed dict that carries the prompt token itself with a numeric logprob, a
    rank >= 1 and a decoded string. The checks are what a one-pass regression would trip.
    """
    choices = payload.get("choices") or [{}] if isinstance(payload, dict) else [{}]
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    positions = choice.get("prompt_logprobs")
    position_list = positions if isinstance(positions, list) else []
    position_zero_null = bool(position_list) and position_list[0] is None
    every_position_scores_its_prompt_token = len(position_list) == len(prompt)
    for index in range(1, len(prompt)):
        entry = position_list[index] if index < len(position_list) else None
        if not isinstance(entry, dict):
            every_position_scores_its_prompt_token = False
            break
        own = entry.get(str(prompt[index]))
        if not isinstance(own, dict):
            every_position_scores_its_prompt_token = False
            break
        logprob = own.get("logprob")
        rank = own.get("rank")
        decoded = own.get("decoded_token")
        if (
            not isinstance(logprob, (int, float))
            or isinstance(logprob, bool)
            or not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
            or not isinstance(decoded, str)
        ):
            every_position_scores_its_prompt_token = False
            break
    return {
        "positions": len(position_list),
        "scored_positions": max(0, len(position_list) - 1),
        "position_zero_null": position_zero_null,
        "every_position_scores_its_prompt_token": every_position_scores_its_prompt_token,
        "sample_scored_position": position_list[1] if len(position_list) > 1 else None,
    }


def prompt_logprobs_probe(
    server: Server,
    prompt: list[int],
    runs: int = 6,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Timing and payload correctness of the one-pass prompt-logprobs path.

    The same buffered prompt_logprobs=1 request and the same request without prompt logprobs are
    sent `runs` times each, interleaved so a load drift lands on both series. The correctness
    flags are the part a regression would trip: one entry per prompt token, position 0 null, and
    every later entry an id-keyed dict that carries the prompt token itself with a numeric
    logprob, a rank >= 1 and a decoded string. The elapsed difference is a diagnostic host-path
    number (loopback included, tiny plan, UNBARED); it is recorded, not turned into a claim.
    """
    base = {
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "seed": 0,
        "skip_special_tokens": SKIP_SPECIAL_FALSE,
    }
    with_logprobs = {**base, "prompt_logprobs": 1}
    without_logprobs = dict(base)
    elapsed: dict[str, list[float]] = {"with_prompt_logprobs": [], "baseline": []}
    payloads: list[Any] = []
    statuses: list[int] = []
    for _ in range(runs):
        for name, body in (
            ("with_prompt_logprobs", with_logprobs),
            ("baseline", without_logprobs),
        ):
            status, _content_type, parsed, seconds = post_json(server, body, timeout=timeout)
            statuses.append(status)
            elapsed[name].append(seconds)
            if name == "with_prompt_logprobs":
                payloads.append(strip_volatile(parsed))

    payload = payloads[0] if payloads else {}
    checks = prompt_logprobs_payload_checks(payload, prompt)
    stable_across_runs = len(payloads) >= 2 and all(
        payload == payloads[0] for payload in payloads[1:]
    )
    all_statuses_ok = bool(statuses) and all(status == 200 for status in statuses)
    with_summary = summarize(elapsed["with_prompt_logprobs"])
    baseline_summary = summarize(elapsed["baseline"])
    overhead_p50_ms = None
    if with_summary["p50"] is not None and baseline_summary["p50"] is not None:
        overhead_p50_ms = 1000.0 * (with_summary["p50"] - baseline_summary["p50"])
    token_requests = sum(1 for status in statuses if status == 200)
    correct = bool(
        all_statuses_ok
        and checks["position_zero_null"]
        and checks["every_position_scores_its_prompt_token"]
        and stable_across_runs
    )
    return {
        "name": "prompt_logprobs_one_pass",
        "label": LABEL,
        "prompt_tokens": len(prompt),
        "runs": runs,
        "with_prompt_logprobs": with_summary,
        "baseline": baseline_summary,
        "overhead_p50_ms": overhead_p50_ms,
        "statuses": sorted(set(statuses)),
        "positions": checks["positions"],
        "scored_positions": checks["scored_positions"],
        "position_zero_null": checks["position_zero_null"],
        "every_position_scores_its_prompt_token": checks["every_position_scores_its_prompt_token"],
        "stable_across_runs": stable_across_runs,
        "correct": correct,
        "token_requests": token_requests,
        "completion_tokens": token_requests,
        "payload": payload,
        "shape": payload_shape(payload),
        "sample_scored_position": checks["sample_scored_position"],
    }


def ladder(
    server: Server,
    levels: list[int],
    requests_per_client: int,
    max_tokens: int,
    timeout: float,
) -> list[dict[str, Any]]:
    """One streaming rung per level; every number in it is client-measured."""
    body = {
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 0,
        "skip_special_tokens": SKIP_SPECIAL_FALSE,
    }
    rungs: list[dict[str, Any]] = []
    for level in levels:
        results: list[dict[str, Any]] = []
        lock = threading.Lock()

        def one() -> None:
            outcome = stream_completion(server, body, timeout=timeout)
            with lock:
                results.append(outcome)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [pool.submit(one) for _ in range(level * requests_per_client)]
            for future in futures:
                future.result()
        wall = time.perf_counter() - started

        ok = [result for result in results if result["status"] == 200]
        refused = [result for result in results if result["status"] == 503]
        failed = [result for result in results if result["status"] not in (200, 503)]
        ttfts = [result["ttft_s"] for result in ok if result["ttft_s"] is not None]
        latencies = [result["elapsed_s"] for result in ok]
        tokens = sum(
            int(((result.get("usage") or {}).get("completion_tokens")) or 0) for result in ok
        )
        client_rates = [
            int(((result.get("usage") or {}).get("completion_tokens")) or 0) / result["elapsed_s"]
            for result in ok
            if result["elapsed_s"] > 0
        ]
        texts = {result["text"] for result in ok}
        rungs.append(
            {
                "concurrency": level,
                "requests": len(results),
                "requests_per_client": requests_per_client,
                "accepted": len(ok),
                "refused_503": len(refused),
                "failed": len(failed),
                "failure_details": [
                    {"status": result["status"], "error": result.get("error")}
                    for result in failed[:8]
                ],
                "wall_s": wall,
                "aggregate_client_tokens_per_s": tokens / wall if wall > 0 else None,
                "server_reported_completion_tokens": tokens,
                "per_request_tokens_per_s": summarize(client_rates),
                "ttft_s": summarize(ttfts),
                "end_to_end_s": summarize(latencies),
                "frames": summarize([float(result["frames"]) for result in ok]),
                "distinct_texts": len(texts),
                "all_streams_terminated": all(result["done"] for result in ok),
                "max_tokens_requested": max_tokens,
                "label": LABEL,
            }
        )
        print(
            f"census: concurrency {level}: {len(ok)} ok, {len(refused)} refused, "
            f"{tokens} tokens, {tokens / wall if wall else 0:.2f} tokens/s",
            flush=True,
        )
    return rungs


def run_mode(
    binary: Path,
    mode: str,
    extra_args: list[str],
    args: argparse.Namespace,
    log_path: Path,
) -> dict[str, Any]:
    """One whole mode: start, fixed probes, ladder, counters, stop."""
    serialized_args = [str(binary), "serve", "--synthetic", "--bind", "127.0.0.1:0", *extra_args]
    load_start = host_load()
    server = Server(binary, extra_args, log_path)
    load_after_start = host_load()
    try:
        health_status, _ = get_json(server, "/health")
        models_status, models = get_json(server, "/v1/models")
        counters_before = metric_values(server)
        ops_before = ops_metrics(server)
        fixed = [
            fixed_probe(server, [1, 2, 3, 4], args.fixed_max_tokens),
            fixed_probe(server, [5, 6, 7], args.fixed_max_tokens),
            fixed_probe(server, [2], args.fixed_max_tokens),
        ]
        # The stop string is the first three characters of the text the tiny plan generates, so
        # the same client request goes to both modes and the probe is like-for-like.
        first_text = fixed[0].get("text") if fixed else None
        stop = first_text[:3] if isinstance(first_text, str) and len(first_text) >= 3 else "<"
        wire = wire_probes(server, PROMPT, stop, timeout=args.timeout)
        prompt_logprobs = prompt_logprobs_probe(
            server, PROMPT, runs=args.prompt_logprobs_runs, timeout=args.timeout
        )
        rungs = ladder(
            server,
            args.levels,
            args.requests_per_client,
            args.max_tokens,
            args.timeout,
        )
        counters_after = metric_values(server)
        ops_after = ops_metrics(server)
        load_end = host_load()

        # Accounting proof for the ops families: the generation counter covers exactly the
        # completion tokens the responses reported, and the TTFT histogram has exactly one
        # sample per token-producing request. Idle gauges must be back at zero.
        fixed_tokens = sum(
            int(probe.get("completion_tokens") or 0)
            for probe in fixed
            if probe.get("status") == 200
        )
        wire_tokens = sum(
            int(probe.get("completion_tokens") or 0)
            for probe in wire
            if probe.get("status") == 200
        )
        ladder_tokens = sum(
            int(rung.get("server_reported_completion_tokens") or 0) for rung in rungs
        )
        prompt_logprobs_tokens = int(prompt_logprobs.get("completion_tokens") or 0)
        prompt_logprobs_requests = int(prompt_logprobs.get("token_requests") or 0)
        expected_tokens = fixed_tokens + wire_tokens + prompt_logprobs_tokens + ladder_tokens
        expected_ttft = (
            sum(
                1
                for probe in fixed
                if probe.get("status") == 200 and int(probe.get("completion_tokens") or 0) > 0
            )
            + sum(
                1
                for probe in wire
                if probe.get("status") == 200 and int(probe.get("completion_tokens") or 0) > 0
            )
            + prompt_logprobs_requests
            + sum(int(rung.get("accepted") or 0) for rung in rungs)
        )
        before_values = ops_before.get("sample_values") or {}
        after_values = ops_after.get("sample_values") or {}
        ops_deltas = {
            name: after_values.get(name, 0.0) - before_values.get(name, 0.0)
            for name in OPS_SAMPLES
        }
        observed_generation = ops_deltas.get("sparkflash_generation_tokens_total", 0.0)
        observed_ttft = ops_deltas.get("sparkflash_time_to_first_token_seconds_count", 0.0)
        idle_running = after_values.get("sparkflash_num_requests_running")
        idle_waiting = after_values.get("sparkflash_num_requests_waiting")
        ops_accounting = {
            "expected_generation_tokens": expected_tokens,
            "observed_generation_tokens_delta": observed_generation,
            "generation_tokens_match": observed_generation == float(expected_tokens),
            "expected_ttft_samples": expected_ttft,
            "observed_ttft_count_delta": observed_ttft,
            "ttft_samples_match": observed_ttft == float(expected_ttft),
            "idle_running": idle_running,
            "idle_waiting": idle_waiting,
            "idle_gauges_zero": idle_running == 0.0 and idle_waiting == 0.0,
        }
        ops_accounting["ok"] = bool(
            ops_accounting["generation_tokens_match"]
            and ops_accounting["ttft_samples_match"]
            and ops_accounting["idle_gauges_zero"]
        )
        ops_families_present = [
            name for name in OPS_FAMILIES if name in (ops_after.get("present_families") or [])
        ]
        evidence_lines = server.dspark_evidence_lines()
        log_text = server.log()
        panics = [line for line in log_text.splitlines() if "panicked" in line]
        failure_lines = [
            line
            for line in log_text.splitlines()
            if "could not" in line or "failed" in line or "ERROR" in line
        ]
        model_id = None
        if models_status == 200 and isinstance(models, dict):
            data = models.get("data") or [{}]
            model_id = data[0].get("id") if data else None
        return {
            "mode": mode,
            "server_args": serialized_args,
            "label": LABEL,
            "health_status": health_status,
            "models_status": models_status,
            "model_id": model_id,
            "expected_model_id": MODEL_ID,
            "loadavg_start": load_start,
            "loadavg_after_start": load_after_start,
            "loadavg_end": load_end,
            "fixed_probes": fixed,
            "wire_stop": stop,
            "wire_probes": wire,
            "prompt_logprobs": prompt_logprobs,
            "ladder": rungs,
            "counters_before": counters_before,
            "counters_after": counters_after,
            "ops_metrics_before": ops_before,
            "ops_metrics_after": ops_after,
            "ops_families_present": ops_families_present,
            "ops_deltas": ops_deltas,
            "ops_accounting": ops_accounting,
            "dspark_evidence_lines": evidence_lines,
            "server_log_lines": len(log_text.splitlines()),
            "panic_lines": panics[:8],
            "failure_lines": failure_lines[:8],
            "server_log": str(log_path),
        }
    finally:
        server.stop()


def counter_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name in sorted(set(before) | set(after)):
        if not name.startswith("sparkflash_"):
            continue
        deltas[name] = after.get(name, 0.0) - before.get(name, 0.0)
    return deltas


def compare(serial: dict[str, Any], batched: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side deltas, computed only from the two modes' measured values."""
    rungs: list[dict[str, Any]] = []
    by_level = {rung["concurrency"]: rung for rung in serial["ladder"]}
    for batched_rung in batched["ladder"]:
        level = batched_rung["concurrency"]
        serial_rung = by_level.get(level)
        if serial_rung is None:
            continue
        entry: dict[str, Any] = {
            "concurrency": level,
            "requests": batched_rung["requests"],
            "serial": {
                "ttft_p50_s": serial_rung["ttft_s"]["p50"],
                "ttft_p95_s": serial_rung["ttft_s"]["p95"],
                "end_to_end_p50_s": serial_rung["end_to_end_s"]["p50"],
                "aggregate_client_tokens_per_s": serial_rung["aggregate_client_tokens_per_s"],
            },
            "batched": {
                "ttft_p50_s": batched_rung["ttft_s"]["p50"],
                "ttft_p95_s": batched_rung["ttft_s"]["p95"],
                "end_to_end_p50_s": batched_rung["end_to_end_s"]["p50"],
                "aggregate_client_tokens_per_s": batched_rung["aggregate_client_tokens_per_s"],
            },
        }
        serial_ttft = serial_rung["ttft_s"]["p50"]
        batched_ttft = batched_rung["ttft_s"]["p50"]
        entry["ttft_p50_ratio_batched_over_serial"] = (
            batched_ttft / serial_ttft if serial_ttft and batched_ttft is not None else None
        )
        serial_rate = serial_rung["aggregate_client_tokens_per_s"]
        batched_rate = batched_rung["aggregate_client_tokens_per_s"]
        entry["aggregate_tokens_per_s_ratio_batched_over_serial"] = (
            batched_rate / serial_rate if serial_rate and batched_rate is not None else None
        )
        rungs.append(entry)

    fixed_texts: list[dict[str, Any]] = []
    identical = True
    for index, (serial_probe, batched_probe) in enumerate(
        zip(serial["fixed_probes"], batched["fixed_probes"])
    ):
        same = (
            serial_probe["status"] == batched_probe["status"]
            and serial_probe["text"] == batched_probe["text"]
            and serial_probe["completion_tokens"] == batched_probe["completion_tokens"]
            and serial_probe["finish_reason"] == batched_probe["finish_reason"]
        )
        identical = identical and same
        fixed_texts.append(
            {
                "probe": index,
                "identical": same,
                "serial": serial_probe,
                "batched": batched_probe,
            }
        )
    wire_entries: list[dict[str, Any]] = []
    wire_identical = True
    batched_wire = {probe["name"]: probe for probe in batched.get("wire_probes") or []}
    for serial_probe in serial.get("wire_probes") or []:
        name = serial_probe["name"]
        batched_probe = batched_wire.get(name)
        if batched_probe is None:
            wire_identical = False
            wire_entries.append({"name": name, "identical": False, "missing_in_batched": True})
            continue

        def probe_payload(probe: dict[str, Any]) -> Any:
            return probe.get("frames") if probe.get("kind") == "sse" else probe.get("payload")

        serial_payload = probe_payload(serial_probe)
        batched_payload = probe_payload(batched_probe)
        if serial_probe.get("kind") == "sse":
            shape_identical = serial_probe.get("frame_shapes") == batched_probe.get("frame_shapes")
        else:
            shape_identical = serial_probe.get("shape") == batched_probe.get("shape")
        payload_identical = serial_payload == batched_payload
        same = (
            serial_probe.get("status") == batched_probe.get("status")
            and serial_probe.get("content_type") == batched_probe.get("content_type")
            and shape_identical
            and payload_identical
            and serial_probe.get("done") == batched_probe.get("done")
            and serial_probe.get("completion_tokens") == batched_probe.get("completion_tokens")
        )
        wire_identical = wire_identical and same
        wire_entries.append(
            {
                "name": name,
                "kind": serial_probe.get("kind"),
                "identical": same,
                "serial_status": serial_probe.get("status"),
                "batched_status": batched_probe.get("status"),
                "shape_identical": shape_identical,
                "payload_identical": payload_identical,
                "serial_completion_tokens": serial_probe.get("completion_tokens"),
                "batched_completion_tokens": batched_probe.get("completion_tokens"),
            }
        )

    serial_pl = serial.get("prompt_logprobs") or {}
    batched_pl = batched.get("prompt_logprobs") or {}
    serial_pl_timing = {
        "baseline_p50_s": (serial_pl.get("baseline") or {}).get("p50"),
        "with_prompt_logprobs_p50_s": (serial_pl.get("with_prompt_logprobs") or {}).get("p50"),
        "overhead_p50_ms": serial_pl.get("overhead_p50_ms"),
    }
    batched_pl_timing = {
        "baseline_p50_s": (batched_pl.get("baseline") or {}).get("p50"),
        "with_prompt_logprobs_p50_s": (batched_pl.get("with_prompt_logprobs") or {}).get("p50"),
        "overhead_p50_ms": batched_pl.get("overhead_p50_ms"),
    }
    prompt_logprobs = {
        "serial_correct": bool(serial_pl.get("correct")),
        "batched_correct": bool(batched_pl.get("correct")),
        "serial_payload_checks": {
            "positions": serial_pl.get("positions"),
            "scored_positions": serial_pl.get("scored_positions"),
            "position_zero_null": serial_pl.get("position_zero_null"),
            "every_position_scores_its_prompt_token": serial_pl.get(
                "every_position_scores_its_prompt_token"
            ),
            "stable_across_runs": serial_pl.get("stable_across_runs"),
        },
        "batched_payload_checks": {
            "positions": batched_pl.get("positions"),
            "scored_positions": batched_pl.get("scored_positions"),
            "position_zero_null": batched_pl.get("position_zero_null"),
            "every_position_scores_its_prompt_token": batched_pl.get(
                "every_position_scores_its_prompt_token"
            ),
            "stable_across_runs": batched_pl.get("stable_across_runs"),
        },
        "shape_identical": serial_pl.get("shape") == batched_pl.get("shape"),
        "payload_identical": serial_pl.get("payload") == batched_pl.get("payload"),
        "timing": {"serial": serial_pl_timing, "batched": batched_pl_timing},
        "label": LABEL,
    }
    prompt_logprobs["ok"] = bool(
        prompt_logprobs["serial_correct"]
        and prompt_logprobs["batched_correct"]
        and prompt_logprobs["shape_identical"]
        and prompt_logprobs["payload_identical"]
    )

    serial_ops = serial.get("ops_accounting") or {}
    batched_ops = batched.get("ops_accounting") or {}
    serial_present = serial.get("ops_families_present") or []
    batched_present = batched.get("ops_families_present") or []
    serial_after = serial.get("ops_metrics_after") or {}
    batched_after = batched.get("ops_metrics_after") or {}
    serial_bounds = [bucket["le"] for bucket in serial_after.get("ttft_buckets") or []]
    batched_bounds = [bucket["le"] for bucket in batched_after.get("ttft_buckets") or []]
    ops = {
        "families_present": {
            "serial": serial_present,
            "batched": batched_present,
            "all_four_in_both": all(
                family in serial_present and family in batched_present
                for family in OPS_FAMILIES
            ),
        },
        "family_help_identical": (serial_after.get("ops_family_help") or {})
        == (batched_after.get("ops_family_help") or {}),
        "ttft_bucket_bounds": serial_bounds,
        "ttft_bucket_layout_identical": serial_bounds == batched_bounds,
        "accounting": {"serial": serial_ops, "batched": batched_ops},
        "accounting_ok": bool(serial_ops.get("ok")) and bool(batched_ops.get("ok")),
    }

    serial_deltas = counter_deltas(serial["counters_before"], serial["counters_after"])
    batched_deltas = counter_deltas(batched["counters_before"], batched["counters_after"])
    return {
        "rungs": rungs,
        "fixed_probe_texts_identical": identical,
        "fixed_probes": fixed_texts,
        "wire_probes_identical": wire_identical,
        "wire_probes": wire_entries,
        "prompt_logprobs": prompt_logprobs,
        "ops_metrics": ops,
        "serial_requests_total_delta": serial_deltas.get("sparkflash_requests_total"),
        "batched_requests_total_delta": batched_deltas.get("sparkflash_requests_total"),
    }


def build_report(args: argparse.Namespace, repo: Path, binary: Path) -> dict[str, Any]:
    serial_args: list[str] = []
    batched_args = ["--continuous-batching", "--max-batch", str(args.max_batch)]
    stamp = now_utc()
    stamp_id = stamp.replace("-", "").replace(":", "")
    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    name = f"serve-s4-batching-census-{stamp_id}"
    serial_log = evidence / f"{name}-server-serial.log"
    batched_log = evidence / f"{name}-server-batched.log"

    print("census: serial mode", flush=True)
    serial = run_mode(binary, "serial", serial_args, args, serial_log)
    print("census: batched mode", flush=True)
    batched = run_mode(binary, "batched", batched_args, args, batched_log)

    report = {
        "schema": SCHEMA,
        "label": LABEL,
        "generated_at": stamp,
        "host": {
            "hostname": os.uname().nodename,
            "cpu_count": os.cpu_count(),
            "loadavg_whole_run": host_load(),
        },
        "repo": {
            "path": str(repo),
            "commit": git(repo, "rev-parse", "HEAD"),
            "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_entries": git(repo, "status", "--porcelain").splitlines(),
        },
        "binary": {
            "path": str(binary),
            "sha256": sha256_file(binary),
        },
        "method": {
            "engine": "reference CPU engine",
            "modes": {
                "serial": serial_args,
                "batched": batched_args,
            },
            "levels": args.levels,
            "requests_per_client": args.requests_per_client,
            "max_tokens": args.max_tokens,
            "fixed_max_tokens": args.fixed_max_tokens,
            "timeout_s": args.timeout,
            "prompt_token_ids": PROMPT,
            "streaming": True,
            "client_measurement": "TTFT is the first non-empty text delta; end-to-end is request close",
            "ops_metrics": (
                "GET /metrics before and after the workload: the generation-token counter delta "
                "must equal the summed response usage.completion_tokens, the TTFT histogram count "
                "delta must equal the number of token-producing requests, and the scheduler gauges "
                "must be zero at idle"
            ),
            "wire_probes": (
                "buffered prompt_logprobs (0 and 1), buffered stop, SSE stop, SSE sample logprobs, "
                "streaming prompt_logprobs refusal and streaming prompt_logprobs=0; payload shape "
                "and stable values are compared across modes"
            ),
            "prompt_logprobs_one_pass": (
                "a repeated buffered prompt_logprobs=1 request against the same request without "
                "prompt logprobs: every prompt position is checked (one entry per token, position 0 "
                "null, own token with numeric logprob, rank >= 1 and decoded string, stable across "
                "runs) and the client-measured elapsed difference is recorded as a diagnostic"
            ),
            "note": "no request leaves loopback; no vLLM endpoint is contacted by this script",
        },
        "mode_order": ["serial", "batched"],
        "serial": serial,
        "batched": batched,
        "comparison": compare(serial, batched),
        "limitations": [
            "host path, synthetic tiny plan (vocab 128): a reference-arithmetic exercise, not model quality",
            "UNBARED / diagnostic (D41): the numbers support regression tracking, not a frozen floor",
            "reference CPU engine only: this says nothing about device batch attention, paged KV, TTFT or throughput",
            "the batch path skips the serial loop's final unused decode after the token budget is spent",
            "one run per mode, not a repeated measurement study; host loadavg is recorded beside it",
            "the ops-metrics accounting covers exactly the requests this census sends (fixed, wire and ladder); it is not a scrape-sampling test and does not exercise live gauges under load",
            "wire probes are API-shape probes, not a load test; they say nothing about device batch behaviour",
            "the prompt-logprobs timing probe is a client-side loopback wall clock on the tiny plan, UNBARED: it is repeatability evidence for regression tracking, not a latency or speedup claim",
        ],
    }
    json_path = evidence / f"{name}.json"
    md_path = evidence / f"{name}.md"
    json_path.write_text(json.dumps(round_floats(report), indent=2) + "\n")
    sha_path = evidence / f"{name}.sha256"
    report["artefacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "sha256": str(sha_path),
        "serial_log": str(serial_log),
        "batched_log": str(batched_log),
    }
    write_markdown(report, md_path)
    lines = []
    for path in [binary, json_path, md_path, serial_log, batched_log]:
        lines.append(f"{sha256_file(path)}  {path}")
    sha_path.write_text("\n".join(lines) + "\n")
    return report


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# S4 host continuous-batching A/B census")
    lines.append("")
    lines.append(f"- label: **{report['label']}**")
    lines.append(f"- generated: {report['generated_at']}")
    lines.append(f"- commit: {report['repo']['commit']} ({report['repo']['branch']})")
    lines.append(f"- binary sha256: {report['binary']['sha256']}")
    lines.append(
        "- modes: serial = 'default (no extra flags)', batched = '"
        + " ".join(report["method"]["modes"]["batched"])
        + "'"
    )
    lines.append("")
    lines.append("## What this measures")
    lines.append("")
    lines.append(
        "The same streaming concurrency ladder runs against the same binary in each mode: "
        "1/8/32 concurrent clients, each client sending a fixed number of streaming completions "
        "against token-id prompts. TTFT is the client's first non-empty text delta; end-to-end is "
        "the request close. Everything is host-path on the reference CPU engine, UNBARED / diagnostic. "
        "Both modes are also scraped at /metrics before and after the workload: the generation-token "
        "counter delta is checked against the summed completion tokens, the TTFT histogram count "
        "delta against the number of token-producing requests, and the scheduler gauges must be "
        "zero at idle. The same fixed wire probes (prompt_logprobs and stop, buffered and SSE) run "
        "against both modes and their payload shape and stable values must be identical."
    )
    lines.append("")
    for mode in ("serial", "batched"):
        data = report[mode]
        lines.append(f"## {mode} mode")
        lines.append("")
        lines.append("- server: " + " ".join(data["server_args"]))
        lines.append(f"- loadavg at start: {data['loadavg_start']}")
        lines.append(f"- loadavg at end: {data['loadavg_end']}")
        lines.append(f"- DSpark evidence lines: {data['dspark_evidence_lines']}")
        lines.append(f"- panic lines: {len(data['panic_lines'])}")
        lines.append("")
        lines.append(
            "| concurrency | requests | accepted | refused | failed | wall s | tokens | tokens/s | TTFT p50 ms | TTFT p95 ms | TTFT max ms | e2e p50 ms | e2e max ms |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rung in data["ladder"]:
            ttft = rung["ttft_s"]
            e2e = rung["end_to_end_s"]
            ttft_p50 = 1000 * ttft["p50"] if ttft["p50"] is not None else None
            ttft_p95 = 1000 * ttft["p95"] if ttft["p95"] is not None else None
            ttft_max = 1000 * ttft["max"] if ttft["max"] is not None else None
            e2e_p50 = 1000 * e2e["p50"] if e2e["p50"] is not None else None
            e2e_max = 1000 * e2e["max"] if e2e["max"] is not None else None
            lines.append(
                f"| {rung['concurrency']} | {rung['requests']} | {rung['accepted']} |"
                f" {rung['refused_503']} | {rung['failed']} | {fmt(rung['wall_s'])} |"
                f" {rung['server_reported_completion_tokens']} |"
                f" {fmt(rung['aggregate_client_tokens_per_s'], 2)} |"
                f" {fmt(ttft_p50, 1)} | {fmt(ttft_p95, 1)} | {fmt(ttft_max, 1)} |"
                f" {fmt(e2e_p50, 1)} | {fmt(e2e_max, 1)} |"
            )
        lines.append("")
    lines.append("## Serial vs batched, per concurrency")
    lines.append("")
    lines.append(
        "| concurrency | serial TTFT p50 ms | batched TTFT p50 ms | TTFT p50 ratio | serial tokens/s | batched tokens/s | tokens/s ratio |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for entry in report["comparison"]["rungs"]:
        serial = entry["serial"]
        batched = entry["batched"]
        serial_ttft = 1000 * serial["ttft_p50_s"] if serial["ttft_p50_s"] is not None else None
        batched_ttft = 1000 * batched["ttft_p50_s"] if batched["ttft_p50_s"] is not None else None
        lines.append(
            f"| {entry['concurrency']} | {fmt(serial_ttft, 1)} | {fmt(batched_ttft, 1)} |"
            f" {fmt(entry['ttft_p50_ratio_batched_over_serial'], 3)} |"
            f" {fmt(serial['aggregate_client_tokens_per_s'], 2)} |"
            f" {fmt(batched['aggregate_client_tokens_per_s'], 2)} |"
            f" {fmt(entry['aggregate_tokens_per_s_ratio_batched_over_serial'], 3)} |"
        )
    lines.append("")
    lines.append("## Output identity across modes")
    lines.append("")
    lines.append(
        "Three fixed buffered completions (token-id prompts, temperature 0, explicit "
        "skip_special_tokens false) are sent to each mode; their text, finish reason and "
        "completion count must be identical, because batching may interleave work but must not "
        "change it."
    )
    lines.append("")
    lines.append(
        "All fixed probes identical: **"
        + str(report["comparison"]["fixed_probe_texts_identical"])
        + "**"
    )
    for probe in report["comparison"]["fixed_probes"]:
        lines.append("")
        lines.append(f"- probe {probe['probe']}: identical = {probe['identical']}")
        lines.append(
            f"  - serial: {probe['serial']['text']!r} ({probe['serial']['finish_reason']})"
        )
        lines.append(
            f"  - batched: {probe['batched']['text']!r} ({probe['batched']['finish_reason']})"
        )

    ops = report["comparison"]["ops_metrics"]
    lines.append("")
    lines.append("## Ops metrics")
    lines.append("")
    lines.append(
        "Each mode is scraped at /metrics before and after the workload. The accounting proof "
        "compares the counter deltas with the responses: the generation-token delta must equal the "
        "summed completion tokens, the TTFT histogram count delta must equal the number of "
        "token-producing requests, and the scheduler gauges must be zero at idle."
    )
    lines.append("")
    lines.append(
        "| mode | expected generation tokens | observed delta | expected TTFT samples | observed delta | idle running | idle waiting | ok |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for mode in ("serial", "batched"):
        accounting = ops["accounting"][mode]
        lines.append(
            f"| {mode} | {accounting.get('expected_generation_tokens')} |"
            f" {accounting.get('observed_generation_tokens_delta')} |"
            f" {accounting.get('expected_ttft_samples')} |"
            f" {accounting.get('observed_ttft_count_delta')} |"
            f" {accounting.get('idle_running')} | {accounting.get('idle_waiting')} |"
            f" {accounting.get('ok')} |"
        )
    lines.append("")
    lines.append(
        "All four ops families present in both modes: **"
        + str(ops["families_present"]["all_four_in_both"])
        + "**; help text identical: **"
        + str(ops["family_help_identical"])
        + "**; TTFT bucket layout identical: **"
        + str(ops["ttft_bucket_layout_identical"])
        + "**."
    )
    lines.append("")
    lines.append("Metric families exported (help text from the serial scrape):")
    lines.append("")
    after = report["serial"].get("ops_metrics_after") or {}
    for family in OPS_FAMILIES:
        help_text = (after.get("ops_family_help") or {}).get(family)
        type_name = (after.get("ops_family_type") or {}).get(family)
        lines.append(f"- {family} ({type_name}): {help_text}")
    lines.append("")
    bounds = ops["ttft_bucket_bounds"]
    lines.append("TTFT histogram bounds: " + ", ".join(bounds))
    lines.append("")
    lines.append("## Wire probes across modes")
    lines.append("")
    lines.append(
        "The same client requests run against both modes: buffered prompt_logprobs (0 and 1), a "
        "buffered stop string, an SSE stop string, SSE sample logprobs, the streaming "
        "prompt_logprobs refusal, and streaming prompt_logprobs=0 (accepted and dropped, vLLM's "
        "rule). Payload shape and stable values (everything except request id, wall-clock created "
        "and the timing block) must be identical."
    )
    lines.append("")
    lines.append(
        "All wire probes identical: **"
        + str(report["comparison"]["wire_probes_identical"])
        + "**"
    )
    lines.append("")
    lines.append(
        "| probe | kind | serial status | batched status | shape identical | payload identical | serial tokens | batched tokens |"
    )
    lines.append("|---|---|---:|---:|---|---|---:|---:|")
    for probe in report["comparison"]["wire_probes"]:
        lines.append(
            f"| {probe['name']} | {probe.get('kind')} | {probe.get('serial_status')} |"
            f" {probe.get('batched_status')} | {probe.get('shape_identical')} |"
            f" {probe.get('payload_identical')} | {probe.get('serial_completion_tokens')} |"
            f" {probe.get('batched_completion_tokens')} |"
        )
    lines.append("")
    pl = report["comparison"]["prompt_logprobs"]
    lines.append("")
    lines.append("## Prompt logprobs one-pass probe")
    lines.append("")
    lines.append(
        "The buffered prompt_logprobs=1 path is repeated against each mode and checked position by "
        "position: one entry per prompt token, position 0 null, and every later entry an id-keyed "
        "dict that carries the prompt token itself with a numeric logprob, a rank >= 1 and a "
        "decoded string. The same request without prompt logprobs is repeated beside it; the "
        "elapsed difference is a diagnostic client-measured loopback number on the tiny plan "
        "(UNBARED), not a latency claim."
    )
    lines.append("")
    lines.append(
        "| mode | positions | scored | position 0 null | every position scores its prompt token | stable across runs | correct |"
    )
    lines.append("|---|---:|---:|---|---|---|---|")
    for mode in ("serial", "batched"):
        checks = pl[f"{mode}_payload_checks"]
        correct = pl[f"{mode}_correct"]
        lines.append(
            f"| {mode} | {checks.get('positions')} | {checks.get('scored_positions')} |"
            f" {checks.get('position_zero_null')} |"
            f" {checks.get('every_position_scores_its_prompt_token')} |"
            f" {checks.get('stable_across_runs')} | {correct} |"
        )
    lines.append("")
    lines.append(
        "| mode | baseline p50 ms | prompt_logprobs p50 ms | diagnostic overhead p50 ms |"
    )
    lines.append("|---|---:|---:|---:|")
    for mode in ("serial", "batched"):
        timing = pl["timing"][mode]
        baseline_ms = 1000 * timing["baseline_p50_s"] if timing["baseline_p50_s"] is not None else None
        with_ms = (
            1000 * timing["with_prompt_logprobs_p50_s"]
            if timing["with_prompt_logprobs_p50_s"] is not None
            else None
        )
        lines.append(
            f"| {mode} | {fmt(baseline_ms, 1)} | {fmt(with_ms, 1)} |"
            f" {fmt(timing['overhead_p50_ms'], 1)} |"
        )
    lines.append("")
    lines.append(
        "One-pass prompt logprobs correct in both modes and payload-identical across modes: **"
        + str(pl["ok"])
        + "**; shape identical: **"
        + str(pl["shape_identical"])
        + "**."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append(f"- JSON: {report['artefacts']['json']}")
    lines.append(f"- Markdown: {report['artefacts']['markdown']}")
    lines.append(f"- sha256: {report['artefacts']['sha256']}")
    lines.append(f"- serial server log: {report['artefacts']['serial_log']}")
    lines.append(f"- batched server log: {report['artefacts']['batched_log']}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="serve binary to measure (default: <repo>/target/debug/sparkflash)",
    )
    parser.add_argument(
        "--evidence-dir",
        default=str(DEFAULT_EVIDENCE_DIR),
        help="where the JSON, Markdown, sha256 and server logs are written",
    )
    parser.add_argument("--levels", default="1,8,32", help="concurrency ladder, comma separated")
    parser.add_argument("--requests-per-client", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--fixed-max-tokens", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument(
        "--prompt-logprobs-runs",
        type=int,
        default=6,
        help="repetitions of each prompt-logprobs timing series (with and without prompt logprobs)",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args(argv)
    args.levels = [int(part) for part in args.levels.split(",") if part.strip()]
    if not args.levels or any(level < 1 for level in args.levels):
        parser.error("--levels must be a comma-separated list of positive integers")

    repo = Path(__file__).resolve().parent.parent
    binary = Path(args.binary) if args.binary else repo / "target" / "debug" / "sparkflash"
    if not binary.is_file():
        print(
            f"census: no serve binary at {binary}; build it first "
            "(cargo build -p sparkflash-cli)",
            file=sys.stderr,
        )
        return 1
    report = build_report(args, repo, binary)
    artefacts = report["artefacts"]
    print("census: wrote", flush=True)
    for name in ("json", "markdown", "sha256", "serial_log", "batched_log"):
        print(f"  {name}: {artefacts[name]}", flush=True)
    comparison = report["comparison"]
    for entry in comparison["rungs"]:
        print(
            f"census: concurrency {entry['concurrency']}: TTFT p50 serial "
            f"{fmt(entry['serial']['ttft_p50_s'])} s -> batched "
            f"{fmt(entry['batched']['ttft_p50_s'])} s (ratio "
            f"{fmt(entry['ttft_p50_ratio_batched_over_serial'])}); tokens/s serial "
            f"{fmt(entry['serial']['aggregate_client_tokens_per_s'], 2)} -> batched "
            f"{fmt(entry['batched']['aggregate_client_tokens_per_s'], 2)}",
            flush=True,
        )
    print(
        "census: fixed probe text identical across modes:",
        comparison["fixed_probe_texts_identical"],
        flush=True,
    )
    ops = comparison["ops_metrics"]
    for mode in ("serial", "batched"):
        accounting = ops["accounting"][mode]
        print(
            f"census: {mode} ops accounting: generation tokens "
            f"{accounting.get('observed_generation_tokens_delta')}/"
            f"{accounting.get('expected_generation_tokens')}, TTFT samples "
            f"{accounting.get('observed_ttft_count_delta')}/"
            f"{accounting.get('expected_ttft_samples')}, idle gauges "
            f"running={accounting.get('idle_running')} waiting={accounting.get('idle_waiting')}, "
            f"ok={accounting.get('ok')}",
            flush=True,
        )
    print(
        "census: wire probes identical across modes:",
        comparison["wire_probes_identical"],
        flush=True,
    )
    pl = comparison["prompt_logprobs"]
    print(
        "census: prompt logprobs one-pass:",
        f"correct serial={pl['serial_correct']} batched={pl['batched_correct']},",
        f"payload identical={pl['payload_identical']},",
        f"timing p50 serial baseline="
        f"{pl['timing']['serial']['baseline_p50_s']} -> with="
        f"{pl['timing']['serial']['with_prompt_logprobs_p50_s']} s,",
        f"batched baseline="
        f"{pl['timing']['batched']['baseline_p50_s']} -> with="
        f"{pl['timing']['batched']['with_prompt_logprobs_p50_s']} s",
        flush=True,
    )
    if (
        not comparison["wire_probes_identical"]
        or not ops["accounting_ok"]
        or not pl["ok"]
    ):
        print(
            "census: PARITY CHECK FAILED; see the JSON artefact for the mismatching probe, "
            "prompt-logprobs check or accounting entry",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
