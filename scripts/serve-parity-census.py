#!/usr/bin/env python3
"""Measured host-path census of SparkFlash's OpenAI-compatible serving surface.

Every number in the JSON and Markdown this script writes is a HOST PATH number,
measured on the development host against the synthetic tiny plan: a reference
arithmetic exercise with a 128-row vocabulary, no checkpoint, no device, no
CUDA. It is explicitly UNBARED / diagnostic (D41 has no frozen floor yet), so
the numbers support regression tracking and parity triage; they are not a
device result, not a model-quality claim, and must never be compared with a GPU
number or a vLLM number.

The script:
  1. starts 'sparkflash serve --synthetic --bind 127.0.0.1:0' (loopback only),
  2. probes the vLLM-mirroring HTTP surface (health, models, metrics, stop
     strings, logprob cap, empty-stop refusal, seed determinism,
     skip_special_tokens default/passthrough, sampler-option acceptance and
     range refusals, streaming usage/[DONE], and the known host limitations),
  3. runs a streaming concurrency ladder (1 / 8 / 32 clients) and records
     client-measured TTFT, end-to-end latency and tokens/s, plus the
     server-reported usage tokens,
  4. writes 'serve-parity-census-<utc>.json' and '.md' under the evidence
     directory (default /home/user/ev-session-20261020/serve/).

No request ever leaves the loopback interface. The live vLLM oracle on
192.0.2.80:8000 is never contacted: this script has no code path that can
address it.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "sparkflash-serve-parity-census/1"
LABEL = "host path, synthetic tiny plan (vocab 128), UNBARED / diagnostic"
MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_EVIDENCE_DIR = Path("/home/user/ev-session-20261020/serve")
PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]


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
    """Nearest-rank percentile (no interpolation); None for an empty sample."""
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
    """Rounds every float to 6 significant digits for a readable, stable file."""
    if isinstance(value, float):
        if value == 0.0 or value != value:
            return value
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {key: round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item) for item in value]
    return value


class Server:
    """The production serve binary on a kernel-picked loopback port."""

    def __init__(self, binary: Path, log_dir: Path, max_logprobs: int) -> None:
        self.binary = binary
        self.log_lines: list[str] = []
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [
                str(binary),
                "serve",
                "--synthetic",
                "--bind",
                "127.0.0.1:0",
                "--max-logprobs",
                str(max_logprobs),
            ],
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
            raise SystemExit(
                "the synthetic server did not report its address; log:\n" + self.log()
            )
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "serve-parity-census-server.log"
        self.log_path.write_text("")
        self._log_handle = self.log_path.open("a")
        self._forward = True

    def _drain(self, pipe: Any) -> None:
        if pipe is None:
            return
        for line in pipe:
            with self._lock:
                self.log_lines.append(line.rstrip("\n"))
                handle = getattr(self, "_log_handle", None)
                if handle is not None and getattr(self, "_forward", False):
                    handle.write(line)
                    handle.flush()

    def log(self) -> str:
        with self._lock:
            return "\n".join(self.log_lines[-80:])

    def stop(self) -> None:
        self._forward = False
        if getattr(self, "_log_handle", None) is not None:
            self._log_handle.close()
        if self._proc.poll() is None:
            self._proc.kill()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def host_port(self) -> tuple[str, int]:
        assert self.base is not None
        host, port = self.base.removeprefix("http://").split(":")
        return host, int(port)


def request_json(
    server: Server,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    host, port = server.host_port()
    payload = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    if payload is not None:
        headers["Content-Length"] = str(len(payload))
    started = time.perf_counter()
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        status = response.status
        response_headers = {name.lower(): value for name, value in response.getheaders()}
    finally:
        connection.close()
    elapsed = time.perf_counter() - started
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    return {
        "status": status,
        "headers": response_headers,
        "json": parsed,
        "bytes": len(raw),
        "elapsed_s": elapsed,
    }


def _raw_get(server: Server, path: str) -> str:
    host, port = server.host_port()
    connection = http.client.HTTPConnection(host, port, timeout=60)
    try:
        connection.request("GET", path)
        return connection.getresponse().read().decode(errors="replace")
    finally:
        connection.close()


def stream_completion(
    server: Server, body: dict[str, Any], timeout: float = 120.0
) -> dict[str, Any]:
    """Sends one streaming completion and records TTFT, frames and usage."""
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



def capability_rows(server: Server, max_logprobs: int) -> list[dict[str, Any]]:
    """One row per vLLM-mirroring behaviour worth a wire-level check."""
    rows: list[dict[str, Any]] = []

    def row(
        surface: str,
        expected: str,
        result: dict[str, Any],
        observed: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "surface": surface,
            "expected": expected,
            "observed": observed,
            "status": result.get("status"),
            "seconds": result.get("elapsed_s"),
        }
        if extra:
            for key, value in extra.items():
                if key not in entry:
                    entry[key] = value
        rows.append(entry)

    health = request_json(server, "GET", "/health")
    row(
        "GET /health",
        "200 with an empty body (vLLM health.py)",
        health,
        f"{health['status']} / {health['bytes']} body bytes",
        {"empty_body": health["bytes"] == 0},
    )

    models = request_json(server, "GET", "/v1/models")
    card = ((models.get("json") or {}).get("data") or [{}])[0]
    row(
        "GET /v1/models",
        "ModelCard members root/parent/max_model_len/permission (vLLM protocol.py)",
        models,
        f"{models['status']} owned_by={card.get('owned_by')}",
        {
            "members": {
                "root": card.get("root"),
                "parent": card.get("parent"),
                "max_model_len": card.get("max_model_len"),
                "permission_is_list": isinstance(card.get("permission"), list),
            }
        },
    )

    metrics = request_json(server, "GET", "/metrics")
    metric_text = _raw_get(server, "/metrics") if metrics.get("status") == 200 else ""
    families = sorted(
        {
            line.split()[0].split("{")[0]
            for line in metric_text.splitlines()
            if line and not line.startswith("#") and len(line.split()) >= 2
        }
    )
    row(
        "GET /metrics",
        "Prometheus text with vllm:* families and http_* instrumentator metrics",
        metrics,
        f"{metrics['status']} sparkflash_* families: {', '.join(families) or 'none'}",
        {
            "families": families,
            "detail": "names differ from vLLM (sparkflash_* vs vllm:*); no http_* request "
            "metrics and no generation-token counter",
        },
    )

    # vLLM's OpenAI default (skip_special_tokens: true) hides the generated BOS special token,
    # so the text probes ask for the explicit false passthrough; the default has its own probe.
    completion = request_json(
        server,
        "POST",
        "/v1/completions",
        {"prompt": PROMPT, "max_tokens": 16, "temperature": 0.0, "skip_special_tokens": False},
    )
    parsed = completion.get("json") or {}
    choice = (parsed.get("choices") or [{}])[0]
    full_text = choice.get("text") or ""
    row(
        "POST /v1/completions (token ids)",
        "200 text_completion with usage",
        completion,
        f"{completion['status']} finish={choice.get('finish_reason')} "
        f"ctoks={(parsed.get('usage') or {}).get('completion_tokens')}",
        {"generated_special_tokens": "▁of▁sentence" in full_text},
    )

    stop_value = full_text[:3]
    stopped = request_json(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 16,
            "temperature": 0.0,
            "stop": stop_value,
            "skip_special_tokens": False,
        },
    )
    stopped_choice = ((stopped.get("json") or {}).get("choices") or [{}])[0]
    stopped_text = stopped_choice.get("text") or ""
    row(
        "completion stop string",
        "matched stop truncated, absent from text, finish_reason stop",
        stopped,
        f"{stopped['status']} text={stopped_text!r} "
        f"finish={stopped_choice.get('finish_reason')} "
        f"ctoks={((stopped.get('json') or {}).get('usage') or {}).get('completion_tokens')}",
        {
            "stop_string": stop_value,
            "matches": stop_value not in stopped_text
            and stopped_choice.get("finish_reason") == "stop"
            and full_text.startswith(stopped_text),
        },
    )

    empty_stop = request_json(
        server,
        "POST",
        "/v1/completions",
        {"prompt": PROMPT, "max_tokens": 4, "stop": [""]},
    )
    message = (((empty_stop.get("json") or {}).get("error")) or {}).get("message")
    row(
        "empty stop string",
        '400 "stop cannot contain an empty string." (sampling_params.py)',
        empty_stop,
        f"{empty_stop['status']} {message!r}",
        {"matches": message == "stop cannot contain an empty string."},
    )

    full_logprobs = request_json(
        server,
        "POST",
        "/v1/completions",
        {"prompt": PROMPT, "max_tokens": 4, "temperature": 0.0, "logprobs": -1},
    )
    scored = (
        (((full_logprobs.get("json") or {}).get("choices") or [{}])[0].get("logprobs") or {}).get(
            "tokens"
        )
        or []
    )
    row(
        "completion logprobs=-1 (whole vocabulary)",
        f"200, allowed because --max-logprobs {max_logprobs} >= vocab 128",
        full_logprobs,
        f"{full_logprobs['status']} scored tokens={len(scored)}",
        {"scored": len(scored)},
    )

    capped = request_json(
        server,
        "POST",
        "/v1/completions",
        {"prompt": PROMPT, "max_tokens": 4, "logprobs": max_logprobs + 1},
    )
    capped_message = (((capped.get("json") or {}).get("error")) or {}).get("message") or ""
    row(
        "completion logprobs over the cap",
        f"400 with vLLM text 'greater than max allowed: {max_logprobs}'",
        capped,
        f"{capped['status']} {capped_message!r}",
        {"matches": f"greater than max allowed: {max_logprobs}" in capped_message},
    )

    def text_of(result: dict[str, Any]) -> str:
        return ((((result.get("json") or {}).get("choices") or [{}])[0].get("text")) or "")

    # vLLM's OpenAI default hides special tokens; an explicit false is the only way to see them.
    hidden = request_json(
        server,
        "POST",
        "/v1/completions",
        {"prompt": PROMPT, "max_tokens": 4, "temperature": 0.0},
    )
    shown = request_json(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 4,
            "temperature": 0.0,
            "skip_special_tokens": False,
        },
    )
    hidden_text = text_of(hidden)
    shown_text = text_of(shown)
    row(
        "skip_special_tokens default vs explicit false",
        "default true hides a generated special token; explicit false passes it through "
        "(completion/protocol.py:82, chat/protocol.py:274)",
        hidden,
        f"default_text={hidden_text!r} explicit_false_text={shown_text!r}",
        {
            "default_hides": hidden_text == "",
            "explicit_false_shows": "\u2581of\u2581sentence" in shown_text,
        },
    )

    repeat = request_json(
        server,
        "POST",
        "/v1/completions",
        {"prompt": PROMPT, "max_tokens": 16, "temperature": 0.0, "skip_special_tokens": False},
    )
    row(
        "determinism (same request twice)",
        "byte-identical text at temperature 0",
        repeat,
        f"identical={text_of(repeat) == full_text}",
        {"identical": text_of(repeat) == full_text},
    )

    seeded_b = request_json(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 8,
            "temperature": 1.0,
            "seed": 7,
            "skip_special_tokens": False,
        },
    )
    seeded_a = request_json(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 8,
            "temperature": 1.0,
            "seed": 7,
            "skip_special_tokens": False,
        },
    )
    row(
        "seed reproducibility at temperature 1",
        "same seed gives the same sampled text",
        seeded_a,
        f"same_seed_identical={text_of(seeded_a) == text_of(seeded_b)}",
        {"same_seed_identical": text_of(seeded_a) == text_of(seeded_b)},
    )

    top_p_a = request_json(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 8,
            "temperature": 1.0,
            "seed": 7,
            "top_p": 1.0,
            "skip_special_tokens": False,
        },
    )
    top_p_b = request_json(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 8,
            "temperature": 1.0,
            "seed": 7,
            "top_p": 0.0001,
            "skip_special_tokens": False,
        },
    )
    top_p_identical = text_of(top_p_a) == text_of(top_p_b)
    row(
        "top_p is applied",
        "top_p 1.0 and 0.0001 with the same seed must sample differently (vLLM); the reference "
        "sampler now honours top_p (unit-tested), but this probe cannot demonstrate it under "
        "the synthetic plan",
        top_p_a,
        f"identical_text={top_p_identical} (indistinguishable on non-finite synthetic logits)",
        {
            "identical": top_p_identical,
            "detail": "when a sampler option is named the engine runs its candidate filter "
            "instead of the default fast path (unit-tested). The synthetic plan's logits are "
            "non-finite, so every step pins token id 0 and the wire probe cannot show the "
            "difference: identical text here is NOT evidence that top_p is dropped",
        },
    )

    # Explicit false: this row checks SSE framing/usage, and the new default would hide the
    # generated BOS special token and leave no visible text delta. The default has its own probe.
    streamed = stream_completion(
        server,
        {
            "prompt": PROMPT,
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "skip_special_tokens": False,
        },
    )
    usage = streamed.get("usage") or {}
    row(
        "streaming completion (include_usage)",
        "SSE frames, final usage chunk with choices=[], then [DONE]",
        {"status": streamed["status"]},
        f"{streamed['status']} frames={streamed['frames']} done={streamed['done']} "
        f"usage_ctoks={usage.get('completion_tokens')}",
        {"frames": streamed["frames"], "done": streamed["done"], "usage": usage},
    )

    unknown_model = request_json(
        server,
        "POST",
        "/v1/completions",
        {"model": "not-a-model", "prompt": PROMPT, "max_tokens": 2},
    )
    error = ((unknown_model.get("json") or {}).get("error")) or {}
    row(
        "unknown model",
        "404 with error.code as the integer HTTP status (vLLM ErrorInfo.code is an int)",
        unknown_model,
        f"{unknown_model['status']} code={error.get('code')!r} type={error.get('type')!r}",
        {
            "matches": unknown_model["status"] == 404
            and error.get("code") == unknown_model["status"]
        },
    )

    n_two = request_json(
        server,
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "n": 2},
    )
    n_message = (((n_two.get("json") or {}).get("error")) or {}).get("message") or ""
    row(
        "chat n=2",
        "vLLM serves n best_of completions; our server refuses n != 1",
        n_two,
        f"{n_two['status']} {n_message!r}",
        {"matches": n_two["status"] == 400 and 'only "n": 1 is supported' in n_message},
    )

    sampler_options = request_json(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 4,
            "temperature": 0.0,
            "top_k": 5,
            "repetition_penalty": 1.2,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "skip_special_tokens": False,
        },
    )
    row(
        "sampler options accepted (top_k / penalties)",
        "vLLM accepts top_k and the three penalties; the reference sampler now honours them "
        "when a request names one (engine unit tests)",
        sampler_options,
        f"{sampler_options['status']} accepted={sampler_options['status'] == 200}",
        {
            "accepted": sampler_options["status"] == 200,
            "detail": "absent fields keep the default fast path that calls sample_gumbel "
            "verbatim; a named knob switches to the engine's candidate filter. Non-finite "
            "synthetic logits pin token id 0, so this probe shows acceptance only",
        },
    )

    rejected = request_json(
        server,
        "POST",
        "/v1/completions",
        {"prompt": PROMPT, "max_tokens": 4, "presence_penalty": 2.5},
    )
    rejected_json = (rejected.get("json") or {}).get("error") or {}
    rejected_message = rejected_json.get("message") or ""
    rejected_code = rejected_json.get("code")
    row(
        "sampler option out of range",
        "400 with vLLM's range-check text (sampling_params.py: presence_penalty in [-2, 2]) "
        "and an integer error.code",
        rejected,
        f"{rejected['status']} {rejected_message!r} code={rejected_code!r}",
        {
            "matches": rejected["status"] == 400
            and "must be in [-2, 2], got 2.5." in rejected_message
            and rejected_code == rejected["status"],
        },
    )

    ignored = request_json(
        server,
        "POST",
        "/v1/completions",
        {
            "prompt": PROMPT,
            "max_tokens": 4,
            "response_format": {"type": "json_object"},
            "min_p": 0.05,
        },
    )
    row(
        "response_format / min_p",
        "vLLM applies guided decoding and min_p; here both fields are absent from the schema "
        "and serde ignores the unknown JSON",
        ignored,
        f"{ignored['status']} accepted_ignored={ignored['status'] == 200}",
        {
            "accepted_ignored": ignored["status"] == 200,
            "detail": "no schema field: no constrained decoding and no min_p filter",
        },
    )

    chat_generation = request_json(
        server,
        "POST",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2},
    )
    chat_message = (((chat_generation.get("json") or {}).get("error")) or {}).get("message") or ""
    row(
        "chat generation on this host",
        "host limitation, not a parity result: the tokenizer seam maps real ids outside "
        "the synthetic 128-row vocabulary",
        chat_generation,
        f"{chat_generation['status']} {chat_message[:100]!r}",
        {
            "host_limitation": True,
            "detail": "text prompts need the released tokenizer's ids; the synthetic tiny "
            "plan has 128 vocabulary rows, so prefill refuses the embedding lookup",
        },
    )

    with_tools = request_json(
        server,
        "POST",
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )
    tools_message = (((with_tools.get("json") or {}).get("error")) or {}).get("message") or ""
    row(
        "tools / tool_choice",
        "vLLM (--enable-auto-tool-choice --tool-call-parser deepseek_v41) parses tool calls",
        with_tools,
        f"{with_tools['status']} {tools_message[:100]!r}",
        {
            "host_limitation": True,
            "detail": "the buffered deepseek_v41 parser exists (sparkflash-api/src/"
            "tool_parser.rs) but this host refuses chat generation before an answer exists, so "
            "the parser cannot be exercised here; streaming tool-call deltas are not parsed "
            "(documented gap), and the reference engine reports capabilities.tools=false",
        },
    )

    return rows



def ladder(
    server: Server,
    levels: list[int],
    requests_per_client: int,
    max_tokens: int,
    timeout: float,
) -> list[dict[str, Any]]:
    """One streaming concurrency rung per level; all numbers are client-measured."""
    rungs: list[dict[str, Any]] = []
    body = {
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 0,
        # Explicit false: TTFT is measured on the first text delta, and the new OpenAI default
        # hides the synthetic plan's only generated (special) token, which would yield no delta.
        "skip_special_tokens": False,
    }
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


def _counter_values(server: Server, path: str) -> dict[str, float]:
    text = _raw_get(server, path)
    values: dict[str, float] = {}
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


def build_json(
    server: Server,
    args: argparse.Namespace,
    repo: Path,
    capability: list[dict[str, Any]],
    rungs: list[dict[str, Any]],
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    single: dict[str, Any] | None,
) -> dict[str, Any]:
    binary = Path(args.binary).resolve()
    tokenizer = (repo / "reference" / "tokenizer.json").resolve()
    return {
        "schema": SCHEMA,
        "label": LABEL,
        "evidence_class": "host_path",
        "device": None,
        "generated_at": now_utc(),
        "git": {
            "repo": str(repo),
            "commit": git(repo, "rev-parse", "HEAD"),
            "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_entries": len(
                [line for line in git(repo, "status", "--porcelain").splitlines() if line]
            ),
        },
        "binary": {
            "path": str(binary),
            "sha256": sha256_file(binary) if binary.exists() else None,
            "bytes": binary.stat().st_size if binary.exists() else None,
        },
        "tokenizer": {
            "path": str(tokenizer),
            "sha256": sha256_file(tokenizer) if tokenizer.exists() else None,
        },
        "serve_command": {
            "args": [
                str(binary),
                "serve",
                "--synthetic",
                "--bind",
                "127.0.0.1:0",
                "--max-logprobs",
                str(args.max_logprobs),
            ],
            "bind": server.base,
            "max_logprobs": args.max_logprobs,
            "synthetic_vocab": 128,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "loadavg": list(os.getloadavg()),
        },
        "workload": {
            "prompt_token_ids": PROMPT,
            "max_tokens_per_request": args.max_tokens,
            "levels": args.levels,
            "requests_per_client": args.requests_per_client,
            "single_request": single,
        },
        "capability": capability,
        "ladder": rungs,
        "server_metrics": {
            "before": before_metrics,
            "after": after_metrics,
            "delta": {
                name: after_metrics.get(name, 0.0) - before_metrics.get(name, 0.0)
                for name in set(before_metrics) | set(after_metrics)
            },
            "note": "sparkflash_requests_total and sparkflash_prompt_tokens_total are the "
            "only request counters; vLLM also exposes generation tokens, running/waiting "
            "requests and http_* request metrics",
        },
        "limitations": [
            "host path only: reference CPU arithmetic, no device, no CUDA, no checkpoint",
            "synthetic tiny plan: 128 vocabulary rows; chat and any real-tokenizer text "
            "prompt is refused at prefill, so the parity probes use token-id prompts",
            "UNBARED / diagnostic: D41 has no frozen floor, so these numbers are for "
            "regression tracking and parity triage, not a qualification result",
            "TTFT and tokens/s are client-measured over loopback and include queue wait; a "
            "concurrency rung can therefore be slower than its predecessor by design",
            "no vLLM comparison is made and no vLLM number is reproduced here; the vLLM "
            "oracle on 192.0.2.80:8000 is never contacted",
            "text and latency probes pass skip_special_tokens: false because the synthetic "
            "plan's only generated token is the BOS special token, which vLLM's OpenAI default "
            "now hides; the default hiding has its own capability probe",
            "the measured server is the default reference path; the TP=4 feature path keeps a "
            "temperature+seed-only sampler and does not parse tool-call text, neither of which "
            "this census can reach",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    add = lines.append
    add(f"# SparkFlash serve parity census - {report['generated_at']}")
    add("")
    add(
        f"**Label: {report['label']}.** Host path only: reference CPU arithmetic, no device, "
        "no CUDA, no checkpoint. These are not device numbers, not a model-quality result and "
        "not a qualification claim (the D41 floor is not frozen)."
    )
    add("")
    add("## Identity")
    add("")
    git_info = report["git"]
    add(
        f"- commit {git_info['commit'] or 'unknown'} on {git_info['branch'] or 'unknown'} "
        f"({git_info['dirty_entries']} dirty entries)"
    )
    add(f"- binary {report['binary']['path']} sha256 {report['binary']['sha256']}")
    add(
        f"- bind {report['serve_command']['bind']} (loopback), "
        f"--max-logprobs {report['serve_command']['max_logprobs']}, "
        f"synthetic vocab {report['serve_command']['synthetic_vocab']}"
    )
    add(
        f"- host: {report['host']['cpu_count']} CPUs, load {report['host']['loadavg']}, "
        f"{report['host']['platform']}"
    )
    add("")
    add("## Capability probes (wire level)")
    add("")
    add("| surface | expected | observed | status |")
    add("| :--- | :--- | :--- | ---: |")
    for row in report["capability"]:
        observed = str(row.get("observed", "")).replace("|", "\\|")
        expected = str(row.get("expected", "")).replace("|", "\\|")
        add(f"| {row['surface']} | {expected} | {observed} | {row.get('status')} |")
    add("")
    limitations = [row for row in report["capability"] if row.get("host_limitation")]
    if limitations:
        add("Host limitations (not parity results):")
        add("")
        for row in limitations:
            add(f"- **{row['surface']}**: {row.get('detail')}")
        add("")
    add("## Streaming concurrency ladder")
    add("")
    add(
        f"Workload: prompt token ids {report['workload']['prompt_token_ids']}, "
        f"max_tokens={report['workload']['max_tokens_per_request']}, temperature 0, streaming "
        f"with include_usage, {report['workload']['requests_per_client']} requests per client."
    )
    add("")
    add(
        "| clients | requests | accepted | refused 503 | TTFT p50 (ms) | TTFT p95 (ms) | "
        "end-to-end p50 (ms) | tokens (server) | client tokens/s (aggregate) |"
    )
    add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    def ms(value: float | None) -> str:
        return "-" if value is None else f"{value * 1000.0:.1f}"

    for rung in report["ladder"]:
        add(
            f"| {rung['concurrency']} | {rung['requests']} | {rung['accepted']} | "
            f"{rung['refused_503']} | {ms(rung['ttft_s']['p50'])} | {ms(rung['ttft_s']['p95'])} | "
            f"{ms(rung['end_to_end_s']['p50'])} | {rung['server_reported_completion_tokens']} | "
            f"{rung['aggregate_client_tokens_per_s']:.3f} |"
        )
    add("")
    single = report["workload"]["single_request"]
    if single:
        add("Single streaming request (headline, no concurrency):")
        add("")
        add(
            f"- TTFT {ms(single['ttft_s'])} ms, end-to-end {ms(single['elapsed_s'])} ms, "
            f"{single['usage_tokens']} completion tokens, {single['tokens_per_s']:.3f} client "
            f"tokens/s, {single['frames']} SSE frames"
        )
        add("")
    add("## Server metrics delta")
    add("")
    add("| metric | before | after | delta |")
    add("| :--- | ---: | ---: | ---: |")
    for name, delta in sorted(report["server_metrics"]["delta"].items()):
        add(
            f"| {name} | {report['server_metrics']['before'].get(name)} | "
            f"{report['server_metrics']['after'].get(name)} | {delta} |"
        )
    add("")
    add(report["server_metrics"]["note"])
    add("")
    add("## Limitations")
    add("")
    for limitation in report["limitations"]:
        add(f"- {limitation}")
    add("")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--binary",
        default=str(Path(__file__).resolve().parent.parent / "target" / "debug" / "sparkflash"),
        help="serve binary to measure (default: <repo>/target/debug/sparkflash)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_EVIDENCE_DIR),
        help=f"evidence directory (default: {DEFAULT_EVIDENCE_DIR})",
    )
    parser.add_argument("--levels", default="1,8,32", help="concurrency ladder, comma separated")
    parser.add_argument("--requests-per-client", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-logprobs", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    args.levels = [int(level) for level in str(args.levels).split(",") if level.strip()]
    repo = Path(__file__).resolve().parent.parent
    binary = Path(args.binary).resolve()
    if not binary.exists():
        print(f"error: serve binary not found: {binary}", file=sys.stderr)
        return 2
    if not all(level >= 1 for level in args.levels):
        print("error: every concurrency level must be >= 1", file=sys.stderr)
        return 2
    evidence_dir = Path(args.out_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"census: {binary} -> {evidence_dir} ({LABEL})", flush=True)
    with tempfile.TemporaryDirectory(prefix="serve-parity-census-") as scratch:
        with Server(binary, Path(scratch), args.max_logprobs) as server:
            print(f"census: server at {server.base}", flush=True)
            before = _counter_values(server, "/metrics")
            capability = capability_rows(server, args.max_logprobs)
            print(f"census: {len(capability)} capability probes", flush=True)
            single = stream_completion(
                server,
                {
                    "prompt": PROMPT,
                    "max_tokens": args.max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    # See the ladder body: the explicit false keeps a visible first text delta.
                    "skip_special_tokens": False,
                },
                timeout=args.timeout,
            )
            single_tokens = int(((single.get("usage") or {}).get("completion_tokens")) or 0)
            single_row = {
                "ttft_s": single["ttft_s"],
                "elapsed_s": single["elapsed_s"],
                "frames": single["frames"],
                "usage_tokens": single_tokens,
                "tokens_per_s": single_tokens / single["elapsed_s"]
                if single["elapsed_s"] > 0
                else None,
            }
            print(
                f"census: single request ttft={single_row['ttft_s']} s, "
                f"elapsed={single_row['elapsed_s']:.3f} s, tokens={single_tokens}",
                flush=True,
            )
            rungs = ladder(
                server, args.levels, args.requests_per_client, args.max_tokens, args.timeout
            )
            after = _counter_values(server, "/metrics")
            report = build_json(server, args, repo, capability, rungs, before, after, single_row)
            shutil.copyfile(
                server.log_path, evidence_dir / f"serve-parity-census-{stamp}.server.log"
            )
    report = round_floats(report)
    json_path = evidence_dir / f"serve-parity-census-{stamp}.json"
    md_path = evidence_dir / f"serve-parity-census-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    write_markdown(report, md_path)
    print(f"census: wrote {json_path}")
    print(f"census: wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

