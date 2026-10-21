#!/usr/bin/env python3
"""OpenAI surface-gap wire census: n > 1 choices, response_format refusals, bearer auth.

Starts the real serve binary on loopback -- the synthetic tiny plan for the token-id completion
probes (serial and --continuous-batching), and the debug-build scripted stub for the chat probes
that need a tokenizer-compatible vocabulary -- then speaks OpenAI HTTP to it as a client would.
Every probe asserts its contract before the artefact is written: choices and their indexes, the
prompt-counted-once/completion-summed usage relation, the SSE ordering, the typed 400 envelopes
for n = 0 and for the unenforceable response_format modes, and the 401 envelope plus /health and
/metrics exemptions in bearer-key mode.

Nothing here is a throughput, device, model-quality or vLLM claim. This is a host-path,
UNBARED diagnostic of the local server's wire surface; the wall numbers are repeatability only.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import platform
import queue
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sparkflash-openai-surface-census/1"
SCOPE = (
    "host path, real serve binary + axum router over loopback: synthetic tiny plan for token-id "
    "completion probes (serial and continuous batching), debug-build scripted stub for chat "
    "probes, UNBARED"
)
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_OUT_DIR = Path("/home/user/ev-session-20261020/serve")
STARTUP_TIMEOUT_SECONDS = 120.0
REQUEST_TIMEOUT_SECONDS = 120.0
SOURCE_MANIFESTS = [
    "Cargo.lock",
    "crates/sparkflash-api/src/engine.rs",
    "crates/sparkflash-api/src/error.rs",
    "crates/sparkflash-api/src/schema.rs",
    "crates/sparkflash-api/src/server.rs",
    "crates/sparkflash-api/src/state.rs",
    "crates/sparkflash-api/tests/router.rs",
    "crates/sparkflash-cli/src/serve.rs",
    "crates/sparkflash-cli/tests/serve_http.rs",
    "docs/build/serving-openai-parity-audit.md",
    "scripts/openai-surface-census.py",
]
NON_CLAIMS = [
    "host path only: no CUDA, no device kernel, no GPU, no cluster run, no vLLM run",
    "synthetic tiny plan (vocab 128) for completion probes: reference arithmetic diagnostics, not the released checkpoint",
    "debug-build scripted stub engine for chat probes: fixed pieces, not a captured model output",
    "constrained decoding is not implemented: response_format json_object/json_schema are refused typed and recorded as an audit gap",
    "the TP=4 n > 1 refusal is router-level unit coverage; no TP=4 group exists on this host, so it is not wire-probed here",
    "bearer auth is a local static key (constant-time compare), not OIDC or multi-tenant authorization",
    "wall numbers are repeatability evidence only, not a throughput or speedup claim",
]


def sha256_file(path: Path) -> str:
    """The sha256 of one file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    """The sha256 of a canonical JSON rendering of a value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sse_frames(text: str) -> list[dict]:
    """The JSON data frames of an SSE body, in order; [DONE] is checked separately."""
    frames = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        frames.append(json.loads(payload))
    return frames


def strip_volatile(value: object) -> object:
    """Drops the fields two separate servers legitimately differ in."""
    if isinstance(value, dict):
        return {
            key: strip_volatile(item)
            for key, item in value.items()
            if key not in ("id", "created", "timing")
        }
    if isinstance(value, list):
        return [strip_volatile(item) for item in value]
    return value


def check(checks: dict, name: str, condition: bool, detail: str = "") -> None:
    """Records one named check and raises with the detail when it is false."""
    checks[name] = bool(condition)
    if not condition:
        raise SystemExit(f"probe check failed: {name}{(': ' + detail) if detail else ''}")


def request(
    base: str,
    method: str,
    path: str,
    body: dict | None = None,
    headers: dict | None = None,
) -> tuple[int, dict[str, str], str]:
    """One HTTP exchange against the loopback test server."""
    parsed = base.removeprefix("http://")
    host, _, port_text = parsed.partition(":")
    connection = http.client.HTTPConnection(host, int(port_text), timeout=REQUEST_TIMEOUT_SECONDS)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    try:
        connection.request(method, path, body=payload, headers=request_headers)
        response = connection.getresponse()
        text = response.read().decode("utf-8", "replace")
        return response.status, {key.lower(): value for key, value in response.getheaders()}, text
    finally:
        connection.close()


def json_body(text: str) -> dict:
    """Parses a JSON response body, naming the body when it is not JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise SystemExit(f"response body is not JSON ({error}): {text[:400]!r}") from error


def usage_of(completion_body: dict) -> dict:
    """The usage block of a buffered completion body."""
    return completion_body["usage"]


def chat_body(extra: dict | None = None) -> dict:
    """A deterministic chat request body."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    if extra:
        body.update(extra)
    return body


def completion_body(prompt: list[int], n: int | None, stream: bool = False) -> dict:
    """A deterministic token-id completion request body."""
    body = {"prompt": prompt, "max_tokens": 3, "temperature": 0.0}
    if n is not None:
        body["n"] = n
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def usage_chunk(frames: list[dict]) -> dict:
    """The include_usage frame of a stream: the one with an empty choices array."""
    for frame in frames:
        if frame.get("choices") == [] and "usage" in frame:
            return frame
    raise SystemExit(f"no include_usage chunk in frames: {frames}")


def choice_indexes(frames: list[dict]) -> list[int]:
    """The distinct choice indexes a stream carried, in first-seen order."""
    indexes: list[int] = []
    for frame in frames:
        for choice in frame.get("choices", []):
            index = choice.get("index")
            if index is not None and index not in indexes:
                indexes.append(index)
    return indexes


def finish_indexes(frames: list[dict]) -> list[int]:
    """The choice indexes whose finish chunk was streamed, in order."""
    indexes = []
    for frame in frames:
        for choice in frame.get("choices", []):
            if choice.get("finish_reason") is not None:
                indexes.append(choice["index"])
    return indexes


class ServeProcess:
    """One sparkflash serve child bound to a kernel-picked loopback port."""

    def __init__(
        self,
        binary: Path,
        tokenizer: Path,
        extra_args: list[str],
        script: dict | None = None,
    ) -> None:
        self.binary = binary
        self.tokenizer = tokenizer
        self.extra_args = extra_args
        self.script = script
        self.child: subprocess.Popen | None = None
        self.base = ""
        self.lines: queue.Queue = queue.Queue()
        self.log: list[str] = []

    def __enter__(self) -> "ServeProcess":
        raw_script = None
        if self.script is not None:
            self.script_dir = tempfile.TemporaryDirectory(prefix="sparkflash-openai-surface-")
            raw_script = Path(self.script_dir.name) / "script.json"
            raw_script.write_text(json.dumps(self.script), encoding="utf-8")
        command = [
            str(self.binary),
            "serve",
            "--synthetic",
            "--bind",
            "127.0.0.1:0",
            *self.extra_args,
            "--tokenizer",
            str(self.tokenizer),
        ]
        environment = os.environ.copy()
        if raw_script is not None:
            environment["SPARKFLASH_SERVE_SCRIPT"] = str(raw_script)
        self.child = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for pipe in (self.child.stdout, self.child.stderr):
            threading.Thread(target=self._drain, args=(pipe,), daemon=True).start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=0.2)
            except queue.Empty:
                if self.child.poll() is not None:
                    break
                continue
            if "listening on " in line:
                self.base = line.split("listening on ", 1)[1].strip()
                return self
        self._stop()
        raise SystemExit(f"the serve binary never reported its address; log:\n{self.log_text()}")

    def _drain(self, pipe) -> None:
        if pipe is None:
            return
        for line in pipe:
            line = line.rstrip("\n")
            self.log.append(line)
            self.lines.put(line)

    def log_text(self) -> str:
        return "\n".join(self.log[-40:])

    def _stop(self) -> None:
        if self.child is None:
            return
        self.child.terminate()
        try:
            self.child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait(timeout=30)

    def __exit__(self, *_exc) -> None:
        self._stop()
        if self.script is not None:
            self.script_dir.cleanup()


def probe_completion_choices(
    binary: Path,
    tokenizer: Path,
    extra_args: list[str],
    label: str,
    checks: dict,
) -> dict:
    """Buffered and streamed token-id completions with n = 2 on the tiny plan."""
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    record: dict = {"label": label, "extra_args": list(extra_args)}
    with ServeProcess(binary, tokenizer, extra_args) as server:
        status, _, text = request(server.base, "POST", "/v1/completions", completion_body(prompt, None))
        check(checks, f"{label}.n1_status", status == 200, text[:400])
        baseline = json_body(text)
        status, _, text = request(server.base, "POST", "/v1/completions", completion_body(prompt, 2))
        check(checks, f"{label}.n2_status", status == 200, text[:400])
        body = json_body(text)
        choices = body["choices"]
        indexes = [choice["index"] for choice in choices]
        check(checks, f"{label}.choice_count", len(choices) == 2, f"choices: {choices}")
        check(checks, f"{label}.choice_indexes", indexes == [0, 1], f"indexes: {indexes}")
        usage = usage_of(body)
        baseline_usage = usage_of(baseline)
        check(
            checks,
            f"{label}.prompt_counted_once",
            usage["prompt_tokens"] == baseline_usage["prompt_tokens"],
            f"{usage} vs {baseline_usage}",
        )
        check(
            checks,
            f"{label}.completion_summed",
            usage["completion_tokens"] == 2 * baseline_usage["completion_tokens"],
            f"{usage} vs {baseline_usage}",
        )
        check(
            checks,
            f"{label}.total_consistent",
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            f"{usage}",
        )
        record["buffered"] = {
            "choice_indexes": indexes,
            "usage": usage,
            "baseline_usage": baseline_usage,
            "choices": [
                {"index": choice["index"], "text": choice["text"], "finish_reason": choice["finish_reason"]}
                for choice in choices
            ],
        }
        status, _, text = request(server.base, "POST", "/v1/completions", completion_body(prompt, None, stream=True))
        check(checks, f"{label}.stream_n1_status", status == 200, text[:400])
        baseline_frames = sse_frames(text)
        status, _, text = request(server.base, "POST", "/v1/completions", completion_body(prompt, 2, stream=True))
        check(checks, f"{label}.stream_n2_status", status == 200, text[:400])
        frames = sse_frames(text)
        check(checks, f"{label}.stream_ends_with_done", text.rstrip().endswith("data: [DONE]"), text[-120:])
        check(
            checks,
            f"{label}.stream_finish_indexes",
            finish_indexes(frames) == [0, 1],
            f"{finish_indexes(frames)}",
        )
        stream_usage = usage_chunk(frames)["usage"]
        baseline_stream_usage = usage_chunk(baseline_frames)["usage"]
        check(
            checks,
            f"{label}.stream_prompt_counted_once",
            stream_usage["prompt_tokens"] == baseline_stream_usage["prompt_tokens"],
            f"{stream_usage} vs {baseline_stream_usage}",
        )
        check(
            checks,
            f"{label}.stream_completion_summed",
            stream_usage["completion_tokens"] == 2 * baseline_stream_usage["completion_tokens"],
            f"{stream_usage} vs {baseline_stream_usage}",
        )
        record["streamed"] = {
            "finish_indexes": finish_indexes(frames),
            "usage": stream_usage,
            "baseline_usage": baseline_stream_usage,
            "frames": len(frames),
        }
        payload = strip_volatile(body)
        record["payload_hash"] = canonical_hash(payload)
        record["payload"] = payload
    return record


def probe_chat_choices(binary: Path, tokenizer: Path, checks: dict) -> dict:
    """Buffered and streamed chat choices with n = 2 over the scripted stub wire."""
    record: dict = {}
    script = {"pieces": ["Hello", " ", "world"], "finish_reason": "stop", "delay_ms": 0}
    with ServeProcess(binary, tokenizer, [], script=script) as server:
        status, _, text = request(server.base, "POST", "/v1/chat/completions", chat_body())
        check(checks, "chat.n1_status", status == 200, text[:400])
        baseline = json_body(text)
        status, _, text = request(server.base, "POST", "/v1/chat/completions", chat_body({"n": 2}))
        check(checks, "chat.n2_status", status == 200, text[:400])
        body = json_body(text)
        choices = body["choices"]
        indexes = [choice["index"] for choice in choices]
        check(checks, "chat.choice_count", len(choices) == 2, f"choices: {choices}")
        check(checks, "chat.choice_indexes", indexes == [0, 1], f"indexes: {indexes}")
        check(
            checks,
            "chat.content_matches_baseline",
            all(
                choice["message"]["content"] == baseline["choices"][0]["message"]["content"]
                for choice in choices
            ),
            f"choices: {choices}",
        )
        usage = usage_of(body)
        baseline_usage = usage_of(baseline)
        check(
            checks,
            "chat.prompt_counted_once",
            usage["prompt_tokens"] == baseline_usage["prompt_tokens"],
            f"{usage} vs {baseline_usage}",
        )
        check(
            checks,
            "chat.completion_summed",
            usage["completion_tokens"] == 2 * baseline_usage["completion_tokens"],
            f"{usage} vs {baseline_usage}",
        )
        check(
            checks,
            "chat.stats_aggregated",
            isinstance(body.get("x_deepseek_spark_stats"), dict),
            f"stats: {body.get('x_deepseek_spark_stats')}",
        )
        record["buffered"] = {
            "choice_indexes": indexes,
            "usage": usage,
            "baseline_usage": baseline_usage,
            "finish_reasons": [choice["finish_reason"] for choice in choices],
        }
        stream_request = chat_body({"n": 2, "stream": True, "stream_options": {"include_usage": True}})
        status, _, text = request(server.base, "POST", "/v1/chat/completions", stream_request)
        check(checks, "chat.stream_status", status == 200, text[:400])
        check(checks, "chat.stream_ends_with_done", text.rstrip().endswith("data: [DONE]"), text[-120:])
        frames = sse_frames(text)
        role_frames = [
            frame
            for frame in frames
            if frame.get("choices") and frame["choices"][0].get("delta", {}).get("role") == "assistant"
        ]
        check(checks, "chat.stream_one_role_chunk", len(role_frames) == 1, f"{role_frames}")
        check(checks, "chat.stream_role_index_zero", role_frames[0]["choices"][0]["index"] == 0, f"{role_frames}")
        check(
            checks,
            "chat.stream_finish_indexes",
            finish_indexes(frames) == [0, 1],
            f"{finish_indexes(frames)}",
        )
        stream_usage = usage_chunk(frames)["usage"]
        check(
            checks,
            "chat.stream_prompt_counted_once",
            stream_usage["prompt_tokens"] == baseline_usage["prompt_tokens"],
            f"{stream_usage} vs {baseline_usage}",
        )
        check(
            checks,
            "chat.stream_completion_summed",
            stream_usage["completion_tokens"] == 2 * baseline_usage["completion_tokens"],
            f"{stream_usage} vs {baseline_usage}",
        )
        last_finish = [
            frame for frame in frames if frame.get("choices") and frame["choices"][0].get("finish_reason")
        ][-1]
        check(
            checks,
            "chat.stream_stats_on_last_finish",
            isinstance(last_finish.get("x_deepseek_spark_stats"), dict),
            f"{last_finish}",
        )
        record["streamed"] = {
            "finish_indexes": finish_indexes(frames),
            "usage": stream_usage,
            "baseline_usage": baseline_usage,
            "frames": len(frames),
        }
    return record


def probe_response_format_refusals(binary: Path, tokenizer: Path, checks: dict) -> dict:
    """The typed 400s for n = 0 and for response_format modes the decoder cannot enforce."""
    record: dict = {}
    script = {"pieces": ["Hello"], "finish_reason": "stop", "delay_ms": 0}
    with ServeProcess(binary, tokenizer, [], script=script) as server:
        record["json_object"] = {}
        for mode, response_format in [
            ("json_object", {"type": "json_object"}),
            (
                "json_schema",
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
                    },
                },
            ),
        ]:
            status, _, text = request(
                server.base,
                "POST",
                "/v1/chat/completions",
                chat_body({"response_format": response_format}),
            )
            body = json_body(text)
            error = body.get("error", {})
            check(checks, f"response_format.{mode}.status", status == 400, text[:400])
            check(checks, f"response_format.{mode}.param", error.get("param") == "response_format", f"{body}")
            check(checks, f"response_format.{mode}.type", error.get("type") == "invalid_request_error", f"{body}")
            check(checks, f"response_format.{mode}.code", error.get("code") == 400, f"{body}")
            check(
                checks,
                f"response_format.{mode}.message_names_constrained_decoding",
                "constrained decoding" in str(error.get("message", "")),
                f"{body}",
            )
            record[mode] = {
                "status": status,
                "param": error.get("param"),
                "type": error.get("type"),
                "code": error.get("code"),
                "message": error.get("message"),
            }
        status, _, text = request(
            server.base,
            "POST",
            "/v1/chat/completions",
            chat_body({"response_format": {"type": "text"}}),
        )
        check(checks, "response_format.text_status", status == 200, text[:400])
        record["text"] = {"status": status}
        status, _, text = request(server.base, "POST", "/v1/chat/completions", chat_body({"n": 0}))
        body = json_body(text)
        check(checks, "n_zero.chat_status", status == 400, text[:400])
        check(checks, "n_zero.chat_param", body.get("error", {}).get("param") == "n", f"{body}")
        record["n_zero_chat"] = {"status": status, "error": body.get("error")}
        status, _, text = request(server.base, "POST", "/v1/completions", completion_body([1, 2, 3], 0))
        body = json_body(text)
        check(checks, "n_zero.completion_status", status == 400, text[:400])
        check(checks, "n_zero.completion_param", body.get("error", {}).get("param") == "n", f"{body}")
        record["n_zero_completion"] = {"status": status, "error": body.get("error")}
    return record


def probe_bearer_auth(binary: Path, tokenizer: Path, checks: dict) -> dict:
    """The 401 envelope, the health/metrics exemptions and the with-key positive control."""
    record: dict = {}
    key = "s3cret"
    with ServeProcess(binary, tokenizer, ["--api-key", key]) as server:
        for path in ("/health", "/metrics"):
            status, _, text = request(server.base, "GET", path)
            check(checks, f"auth.{path}.open", status == 200, f"{status}: {text[:200]}")
            record[path] = {"open_status": status}
        status, _, text = request(server.base, "GET", "/v1/models")
        body = json_body(text)
        check(checks, "auth.models_missing_status", status == 401, text[:400])
        check(
            checks,
            "auth.models_envelope",
            body
            == {
                "error": {
                    "message": "Unauthorized",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": 401,
                }
            },
            f"{body}",
        )
        record["models_missing"] = {"status": status, "body": body}
        for header_value in ("Bearer wrong", "Basic s3cret", "s3cret"):
            status, _, text = request(
                server.base, "GET", "/v1/models", headers={"Authorization": header_value}
            )
            check(checks, f"auth.models_wrong.{header_value}", status == 401, f"{status}: {text[:200]}")
        record["wrong_headers"] = ["Bearer wrong", "Basic s3cret", "s3cret"]
        for header_value in ("Bearer s3cret", "bearer s3cret"):
            status, _, text = request(
                server.base, "GET", "/v1/models", headers={"Authorization": header_value}
            )
            check(checks, f"auth.models_ok.{header_value}", status == 200, f"{status}: {text[:200]}")
            check(checks, f"auth.models_ok.{header_value}.data", isinstance(json_body(text).get("data"), list))
        record["models_ok"] = {"status": 200, "headers": ["Bearer s3cret", "bearer s3cret"]}
        generation = completion_body([1, 2, 3, 4], None)
        status, _, text = request(server.base, "POST", "/v1/completions", generation)
        check(checks, "auth.completion_missing_status", status == 401, text[:400])
        status, _, text = request(
            server.base,
            "POST",
            "/v1/completions",
            generation,
            headers={"Authorization": "Bearer wrong"},
        )
        check(checks, "auth.completion_wrong_status", status == 401, text[:400])
        status, _, text = request(
            server.base,
            "POST",
            "/v1/completions",
            generation,
            headers={"Authorization": "Bearer s3cret"},
        )
        check(checks, "auth.completion_ok_status", status == 200, text[:400])
        check(checks, "auth.completion_ok_choices", isinstance(json_body(text).get("choices"), list))
        record["completion"] = {
            "missing": 401,
            "wrong": 401,
            "with_key": 200,
        }
    return record


def probe_open_mode(binary: Path, tokenizer: Path, checks: dict) -> dict:
    """No --api-key keeps the pre-auth behaviour: every route is open."""
    record: dict = {}
    with ServeProcess(binary, tokenizer, []) as server:
        status, _, text = request(server.base, "GET", "/v1/models")
        check(checks, "open.models_status", status == 200, text[:300])
        status, _, text = request(server.base, "POST", "/v1/completions", completion_body([1, 2, 3], None))
        check(checks, "open.completion_status", status == 200, text[:300])
        record = {"models": 200, "completion": 200}
    return record


def run_probe(name: str, function, *args) -> tuple[dict, int]:
    """Runs one probe, timing it, and returns its record and wall milliseconds."""
    started = time.monotonic()
    record = function(*args)
    return record, round((time.monotonic() - started) * 1000.0)


def summarize(runs: list[dict]) -> dict:
    """Flattens the named checks across runs into the census accounting block."""
    names = sorted({name for run in runs for name in run["checks"]})
    checks = {name: all(run["checks"].get(name, False) for run in runs) for name in names}
    return {
        "runs": len(runs),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "check_count": len(checks),
        "serial_batched_payload_equal": all(
            run["probes"]["completions_serial"]["payload_hash"]
            == run["probes"]["completions_batched"]["payload_hash"]
            for run in runs
        ),
        "probe_wall_ms": [run["probe_wall_ms"] for run in runs],
    }


def validate_census(census: dict) -> None:
    """Raises SystemExit unless the census proves every contract it claims."""
    if census.get("schema") != SCHEMA:
        raise SystemExit(f"unexpected census schema: {census.get('schema')!r}")
    accounting = census.get("accounting", {})
    if accounting.get("runs", 0) < 1:
        raise SystemExit("the census has no runs")
    if accounting.get("all_checks_passed") is not True:
        failed = [name for name, ok in accounting.get("checks", {}).items() if not ok]
        raise SystemExit(f"census checks failed: {failed}")
    if accounting.get("serial_batched_payload_equal") is not True:
        raise SystemExit("the serial and continuous-batching completion payloads differ")
    for run in census.get("raw_runs", []):
        probes = run.get("probes", {})
        if probes.get("completions_serial", {}).get("buffered", {}).get("choice_indexes") != [0, 1]:
            raise SystemExit("the serial completion probe did not return two ordered choices")
        if not probes.get("response_format_refusals", {}).get("json_object", {}).get("param") == "response_format":
            raise SystemExit("the json_object refusal does not name response_format")
        if probes.get("bearer_auth", {}).get("models_missing", {}).get("status") != 401:
            raise SystemExit("the keyed server served /v1/models without a key")


def render_markdown(census: dict) -> str:
    """One readable summary of the census."""
    accounting = census["accounting"]
    checks = accounting["checks"]
    lines = [
        "# OpenAI surface-gap wire census (n > 1, response_format, bearer auth)",
        "",
        f"- schema: {census['schema']}",
        f"- generated: {census['generated_at']}",
        f"- git: {census['git']['branch']} @ {census['git']['commit']}",
        f"- serve binary: {census['serve_binary']['path']} sha256={census['serve_binary']['sha256']}",
        f"- host: {census['host']['platform']} ({census['host']['machine']}, {census['host']['cpus']} cpus)",
        f"- label: {census['label']}",
        "",
        "## Scope",
        "",
        census["scope"],
        "",
        "## Accounting",
        "",
        f"- runs: {accounting['runs']}",
        f"- named checks: {accounting['check_count']} (all passed: {accounting['all_checks_passed']})",
        f"- serial and continuous-batching completion payloads equal: {accounting['serial_batched_payload_equal']}",
        f"- probe wall ms: {accounting['probe_wall_ms']}",
        "",
        "## Checks",
        "",
    ]
    for name in sorted(checks):
        lines.append(f"- {name}: {'PASS' if checks[name] else 'FAIL'}")
    lines += [
        "",
        "## Wire facts per run",
        "",
        "| run | completion choices (serial) | n=2 usage (serial) | chat choices | chat stream finish indexes | refusal status | auth models missing |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, run in enumerate(census["raw_runs"]):
        probes = run["probes"]
        serial_usage = probes["completions_serial"]["buffered"]["usage"]
        chat = probes["chat_choices"]["buffered"]
        lines.append(
            "| {} | {} | prompt={} completion={} | {} | {} | {} | {} |".format(
                index,
                probes["completions_serial"]["buffered"]["choice_indexes"],
                serial_usage["prompt_tokens"],
                serial_usage["completion_tokens"],
                chat["choice_indexes"],
                probes["chat_choices"]["streamed"]["finish_indexes"],
                probes["response_format_refusals"]["json_object"]["status"],
                probes["bearer_auth"]["models_missing"]["status"],
            )
        )
    lines += [
        "",
        "## Non-claims",
        "",
    ]
    for claim in census["non_claims"]:
        lines.append(f"- {claim}")
    lines += [
        "",
        "## Source manifests",
        "",
    ]
    for entry in census["source_manifests"]:
        lines.append(f"- {entry['path']} sha256={entry['sha256']}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="the sparkflash serve binary (default: target/debug/sparkflash)",
    )
    parser.add_argument("--runs", type=int, default=3, help="probe repetitions (default 3)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="where the JSON and markdown artefacts are written",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    repo = Path(__file__).resolve().parent.parent
    binary = args.binary or (repo / "target" / "debug" / "sparkflash")
    if not binary.is_file():
        raise SystemExit(
            f"the serve binary is missing: {binary}; build it with: cargo build -p sparkflash-cli"
        )
    tokenizer = repo / "reference" / "tokenizer.json"
    if not tokenizer.is_file():
        raise SystemExit(f"the reference tokenizer is missing: {tokenizer}")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"openai-surface-census-{stamp}"
    runs: list[dict] = []
    for run_index in range(args.runs):
        checks: dict = {}
        probes: dict = {}
        probe_wall: dict = {}
        probes["completions_serial"], probe_wall["completions_serial"] = run_probe(
            "completions_serial",
            probe_completion_choices,
            binary,
            tokenizer,
            [],
            "serial",
            checks,
        )
        probes["completions_batched"], probe_wall["completions_batched"] = run_probe(
            "completions_batched",
            probe_completion_choices,
            binary,
            tokenizer,
            ["--continuous-batching", "--max-batch", "2"],
            "batched",
            checks,
        )
        probes["chat_choices"], probe_wall["chat_choices"] = run_probe(
            "chat_choices", probe_chat_choices, binary, tokenizer, checks
        )
        probes["response_format_refusals"], probe_wall["response_format_refusals"] = run_probe(
            "response_format_refusals", probe_response_format_refusals, binary, tokenizer, checks
        )
        probes["bearer_auth"], probe_wall["bearer_auth"] = run_probe(
            "bearer_auth", probe_bearer_auth, binary, tokenizer, checks
        )
        probes["open_mode"], probe_wall["open_mode"] = run_probe(
            "open_mode", probe_open_mode, binary, tokenizer, checks
        )
        runs.append(
            {
                "run": run_index,
                "checks": checks,
                "probes": probes,
                "probe_wall_ms": probe_wall,
            }
        )
    accounting = summarize(runs)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    git_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    rustc_version = subprocess.run(
        ["rustc", "--version"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    census = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": SCOPE,
        "label": "UNBARED host-path diagnostic",
        "git": {"commit": git_commit, "branch": git_branch},
        "serve_binary": {
            "path": str(binary),
            "sha256": sha256_file(binary),
            "bytes": binary.stat().st_size,
        },
        "tokenizer": {"path": str(tokenizer), "sha256": sha256_file(tokenizer)},
        "toolchain": {
            "rustc": rustc_version,
            "python": sys.version.split()[0],
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else [0.0, 0.0, 0.0],
        },
        "fixture": {
            "completion_prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            "completion_max_tokens": 3,
            "chat_script_pieces": ["Hello", " ", "world"],
            "note": "fixed deterministic requests; n=1 baselines and n=2 responses come from the same server process",
        },
        "accounting": accounting,
        "source_manifests": [
            {"path": relative, "sha256": sha256_file(repo / relative)}
            for relative in SOURCE_MANIFESTS
            if (repo / relative).is_file()
        ],
        "raw_runs": runs,
        "non_claims": NON_CLAIMS,
    }
    validate_census(census)
    json_path = out_dir / f"{base_name}.json"
    md_path = out_dir / f"{base_name}.md"
    json_path.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(census), encoding="utf-8")
    manifest_path = out_dir / f"{base_name}.sha256"
    manifest = {
        "json": sha256_file(json_path),
        "md": sha256_file(md_path),
        "serve_binary": census["serve_binary"]["sha256"],
        "source_manifests": {
            entry["path"]: entry["sha256"] for entry in census["source_manifests"]
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"json: {json_path} sha256={manifest['json']}")
    print(f"md: {md_path} sha256={manifest['md']}")
    print(f"serve binary: {binary} sha256={manifest['serve_binary']}")
    print(
        "accounting: "
        f"{accounting['runs']} runs, {accounting['check_count']} checks, "
        f"all_passed={accounting['all_checks_passed']}, "
        f"serial_batched_equal={accounting['serial_batched_payload_equal']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
