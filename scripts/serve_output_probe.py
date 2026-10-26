#!/usr/bin/env python3
"""Measure OUTPUT (text, latency, tokens/s) of an OpenAI-compatible serve endpoint.

Task-56/58: this is the baseline probe for the reference serve on spark-0
(127.0.0.1:18000) and the same script is used for the TP=4 endpoint (task-55/58).
Requests are sent strictly one at a time: the reference serve is a single-process,
disk-bound engine, so parallel requests would measure queueing rather than the engine.

It speaks both OpenAI surfaces: /v1/completions (choices[].text) and
/v1/chat/completions (choices[].message.content buffered, choices[].delta.content in SSE).
The tp4-chat preset carries the three task-58 chat prompts with their per-prompt
max_tokens; --base-url/--model point it at any OpenAI-compatible rank-0 front without
code changes. With --stream it records client time-to-first-token (first non-empty text /
delta content), SSE chunk count and chunk-inferred token counts when usage is absent.
--io-pid samples /proc/<pid>/io around each request, locally or over --io-ssh HOST, so a
server on another node yields rchar/read_bytes per completion token. One JSON document and
one markdown table are written per run.

A slow response is not an error: the per-request timeout is configurable and defaults to
30 minutes. A timeout, a non-200 or a broken stream is recorded in the JSON and the probe
continues with the next prompt. Text is recorded verbatim; coherence is judged by reading
it, not by comparing token ids.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

SCHEMA = "sparkflash-serve-output/1"
DEFAULT_BASE_URL = "http://127.0.0.1:18000"
DEFAULT_MODELS_TIMEOUT_S = 30.0
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_MAX_TOKENS = 4
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SSH_TIMEOUT_S = 15.0
DEFAULT_ENDPOINT = "completion"
ENDPOINTS = ("completion", "chat")

# The approved single-node baseline budget is two short prompts; the same two are the
# prompts for the TP=4 comparison. Each got max_tokens 2 (see the task-56 baseline doc).
BASELINE_PROMPTS = [
    {"id": "capital-of-france", "prompt": "The capital of France is"},
    {"id": "rust-borrow", "prompt": "In Rust, the borrow checker rejects this code because:"},
]

# task-58: the three chat prompts (and the per-prompt max_tokens from the task card).
TP4_CHAT_PROMPTS = [
    {"id": "capital", "prompt": "What is the capital of France?", "max_tokens": 32},
    {"id": "rust-borrower", "prompt": "Explain what a Rust borrow checker does in two sentences.", "max_tokens": 64},
    {"id": "haiku", "prompt": "Write a haiku about four GPUs on one rack.", "max_tokens": 48},
]

PRESETS = {
    "baseline": {"endpoint": "completion", "prompts": BASELINE_PROMPTS},
    "tp4-chat": {"endpoint": "chat", "prompts": TP4_CHAT_PROMPTS},
}
DEFAULT_PRESET = "baseline"
DEFAULT_PROMPTS = PRESETS[DEFAULT_PRESET]["prompts"]

SERVER_TIMING_FIELDS = ("ttft_ms", "prefill_tok_per_s", "decode_tok_per_s", "min_rank_uma_free_gib")


def utc_now() -> str:
    """Current UTC time as the evidence timestamp format used by the other scripts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compact_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_proc_io(text: str) -> dict:
    """Parse the integer fields of /proc/<pid>/io (rchar, read_bytes, ...)."""
    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if value.isdigit():
            fields[key.strip()] = int(value)
    return fields


def _read_proc_io_over_ssh(pid: int, ssh_host: str) -> dict | None:
    """Read /proc/<pid>/io on another node with one non-interactive ssh call."""
    command = ["ssh", "-n", "-o", "BatchMode=yes", ssh_host, "cat", f"/proc/{int(pid)}/io"]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=DEFAULT_SSH_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return parse_proc_io(completed.stdout)


def read_proc_io(pid: int | None, ssh_host: str | None = None) -> dict | None:
    """Read /proc/<pid>/io locally or on ssh_host; None when unavailable/unreadable."""
    if not pid:
        return None
    if ssh_host:
        return _read_proc_io_over_ssh(pid, ssh_host)
    try:
        with open(f"/proc/{int(pid)}/io", encoding="utf-8") as handle:
            return parse_proc_io(handle.read())
    except (OSError, ValueError):
        return None


def _io_transport(ssh_host: str | None) -> str:
    return f"ssh:{ssh_host}" if ssh_host else "local"


def _attach_io(result: dict, before: dict | None, pid: int | None, ssh_host: str | None = None) -> dict:
    """Attach the request's /proc/<pid>/io deltas and their per-completion-token rates."""
    if not pid:
        return result
    transport = _io_transport(ssh_host)
    if not before:
        result["io"] = {"pid": pid, "transport": transport, "error": "first /proc/<pid>/io sample unavailable"}
        return result
    after = read_proc_io(pid, ssh_host)
    if not after:
        result["io"] = {"pid": pid, "transport": transport, "error": "could not re-read /proc/<pid>/io"}
        return result
    rchar = max(0, after.get("rchar", 0) - before.get("rchar", 0))
    read_bytes = max(0, after.get("read_bytes", 0) - before.get("read_bytes", 0))
    tokens = int(result.get("completion_tokens") or 0)
    wall = float(result.get("wall_s") or 0.0)
    result["io"] = {
        "pid": pid,
        "transport": transport,
        "rchar": rchar,
        "read_bytes": read_bytes,
        "rchar_per_completion_token": round(rchar / tokens, 1) if tokens else None,
        "read_bytes_per_completion_token": round(read_bytes / tokens, 1) if tokens else None,
        "rchar_per_s": round(rchar / wall, 1) if wall > 0 else None,
        "read_bytes_per_s": round(read_bytes / wall, 1) if wall > 0 else None,
    }
    return result


def compute_rates(wall_s: float, completion_tokens: int | None, ttft_s: float | None = None) -> dict:
    """Return completion-token rates per wall second and per post-TTFT second."""
    rates = {"tokens_per_s": 0.0, "decode_tokens_per_s": 0.0}
    if completion_tokens is None or completion_tokens <= 0:
        return rates
    if wall_s > 0:
        rates["tokens_per_s"] = round(completion_tokens / wall_s, 6)
    window = wall_s if ttft_s is None else wall_s - ttft_s
    if window > 0:
        rates["decode_tokens_per_s"] = round(completion_tokens / window, 6)
    return rates


def describe_error(exc: BaseException) -> str:
    """One-line description of a failed request, with the HTTP body when there is one."""
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        return f"HTTP {exc.code}: {body}".strip()
    return f"{type(exc).__name__}: {exc}"


def _json_request(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status), response.read().decode("utf-8", "replace")


def fetch_models(base_url: str, timeout: float) -> dict:
    """GET /v1/models; raises on failure so main() can exit 2 before any request."""
    url = base_url.rstrip("/") + "/v1/models"
    limit = min(float(timeout), DEFAULT_MODELS_TIMEOUT_S)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=limit) as response:
        body = response.read().decode("utf-8", "replace")
    document = json.loads(body)
    if not isinstance(document, dict):
        raise ValueError("/v1/models returned a non-object body")
    return document


def resolve_model(models_document: dict, requested: str | None) -> str:
    if requested:
        return requested
    data = models_document.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("/v1/models returned no data entry")
    model = data[0].get("id") if isinstance(data[0], dict) else None
    if not isinstance(model, str) or not model:
        raise ValueError("/v1/models returned no model id")
    return model


def load_prompts(path: str | None) -> list[dict]:
    """Load prompts from a JSON list of strings or {id, prompt, max_tokens} objects."""
    raw = DEFAULT_PROMPTS
    if path:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("prompts must be a non-empty JSON list")
    prompts: list[dict] = []
    for index, item in enumerate(raw):
        max_tokens = None
        if isinstance(item, str):
            prompt = item
            prompt_id = f"prompt-{index}"
        elif isinstance(item, dict):
            prompt = item.get("prompt")
            prompt_id = item.get("id") or f"prompt-{index}"
            max_tokens = item.get("max_tokens")
        else:
            raise ValueError(f"prompt {index}: expected a string or an object")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"prompt {index}: missing non-empty 'prompt' string")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"prompt {index}: 'id' must be a non-empty string")
        if max_tokens is not None and (
            not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
        ):
            raise ValueError(f"prompt {index}: 'max_tokens' must be a positive integer")
        prompts.append({"id": prompt_id, "prompt": prompt, "max_tokens": max_tokens})
    return prompts


def select_prompts(args: argparse.Namespace) -> list[dict]:
    """--prompts overrides the preset; otherwise the preset's prompts are copied."""
    if args.prompts:
        return load_prompts(args.prompts)
    return [dict(item) for item in PRESETS[args.preset]["prompts"]]


def _server_stats(document: dict) -> dict | None:
    """The serve's x_deepseek_spark_stats block (cache/timing evidence), if present."""
    stats = document.get("x_deepseek_spark_stats")
    return stats if isinstance(stats, dict) else None


def _server_timing(document: dict) -> dict | None:
    stats = document.get("x_deepseek_spark_stats")
    if not isinstance(stats, dict):
        return None
    timing = stats.get("timing")
    if not isinstance(timing, dict):
        return None
    picked = {field: timing.get(field) for field in SERVER_TIMING_FIELDS if field in timing}
    return picked or None


def _fill_usage(result: dict, usage) -> None:
    if not isinstance(usage, dict):
        return
    result["prompt_tokens"] = usage.get("prompt_tokens")
    result["completion_tokens"] = usage.get("completion_tokens")
    result["total_tokens"] = usage.get("total_tokens")
    result["completion_tokens_source"] = "usage"


def _max_tokens(item: dict, args: argparse.Namespace) -> int:
    """Per-prompt max_tokens from the preset/file wins over the CLI default."""
    return int(item.get("max_tokens") or args.max_tokens)


def _completions_url(base_url: str, endpoint: str) -> str:
    suffix = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
    return base_url.rstrip("/") + suffix


def _request_payload(endpoint: str, model: str, item: dict, args: argparse.Namespace, stream: bool = False) -> dict:
    payload = {
        "model": model,
        "temperature": args.temperature,
        "max_tokens": _max_tokens(item, args),
    }
    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": item["prompt"]}]
    else:
        payload["prompt"] = item["prompt"]
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def _chat_content(message) -> str:
    """Text from a chat message/delta; handles string content and multipart content lists."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


def _base_result(index: int, item: dict, args: argparse.Namespace) -> dict:
    return {
        "index": index,
        "id": item["id"],
        "prompt": item["prompt"],
        "endpoint": getattr(args, "endpoint", DEFAULT_ENDPOINT),
        "max_tokens": _max_tokens(item, args),
        "temperature": args.temperature,
        "ok": False,
        "http_status": None,
        "text": None,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "completion_tokens_source": None,
        "wall_s": 0.0,
        "ttft_s": None,
        "tokens_per_s": 0.0,
        "decode_tokens_per_s": 0.0,
        "server_timing": None,
        "server_stats": None,
        "io": None,
        "error": None,
    }


def _finish_result(result: dict, started: float) -> dict:
    result["wall_s"] = round(time.monotonic() - started, 6)
    result.update(compute_rates(result["wall_s"], result["completion_tokens"], result["ttft_s"]))
    if not result["ok"] and result["error"] is None:
        result["error"] = "request did not complete"
    return result


def probe_buffered(base_url: str, model: str, item: dict, args: argparse.Namespace) -> dict:
    result = _base_result(item["index"], item, args)
    endpoint = getattr(args, "endpoint", DEFAULT_ENDPOINT)
    payload = _request_payload(endpoint, model, item, args)
    started = time.monotonic()
    io_ssh = getattr(args, "io_ssh", None)
    io_before = read_proc_io(getattr(args, "io_pid", None), io_ssh)
    try:
        status, body = _json_request(
            _completions_url(base_url, endpoint), payload, args.timeout
        )
        result["http_status"] = status
        document = json.loads(body)
        choice = (document.get("choices") or [{}])[0]
        if endpoint == "chat":
            result["text"] = _chat_content(choice.get("message"))
        else:
            result["text"] = choice.get("text") or ""
        result["finish_reason"] = choice.get("finish_reason")
        _fill_usage(result, document.get("usage"))
        result["server_stats"] = _server_stats(document)
        result["server_timing"] = _server_timing(document)
        result["ok"] = True
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error"] = describe_error(exc)
    except Exception as exc:  # URLError, timeout, JSON decode, malformed body
        result["error"] = describe_error(exc)
    result = _finish_result(result, started)
    _attach_io(result, io_before, getattr(args, "io_pid", None), io_ssh)
    return result


def probe_stream(base_url: str, model: str, item: dict, args: argparse.Namespace) -> dict:
    result = _base_result(item["index"], item, args)
    result["chunks"] = 0
    result["bad_chunks"] = 0
    result["sse_done"] = False
    endpoint = getattr(args, "endpoint", DEFAULT_ENDPOINT)
    payload = _request_payload(endpoint, model, item, args, stream=True)
    started = time.monotonic()
    parts: list[str] = []
    io_ssh = getattr(args, "io_ssh", None)
    io_before = read_proc_io(getattr(args, "io_pid", None), io_ssh)
    usage = None
    finish_reason = None
    ttft = None
    try:
        request = urllib.request.Request(
            _completions_url(base_url, endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result["http_status"] = int(response.status)
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    result["sse_done"] = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    result["bad_chunks"] += 1
                    continue
                result["chunks"] += 1
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                if _server_stats(chunk) is not None:
                    result["server_stats"] = _server_stats(chunk)
                if _server_timing(chunk) is not None:
                    result["server_timing"] = _server_timing(chunk)
                for choice in chunk.get("choices") or []:
                    if endpoint == "chat":
                        piece = _chat_content(choice.get("delta"))
                    else:
                        piece = choice.get("text") or ""
                    if piece:
                        if ttft is None:
                            ttft = time.monotonic() - started
                        parts.append(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        result["text"] = "".join(parts)
        result["finish_reason"] = finish_reason
        result["ttft_s"] = round(ttft, 6) if ttft is not None else None
        if usage is not None:
            _fill_usage(result, usage)
        else:
            result["completion_tokens"] = len(parts) if parts else None
            result["completion_tokens_source"] = "chunks"
        result["ok"] = result["text"] != "" or finish_reason is not None
        if not result["ok"]:
            result["error"] = "stream produced no text and no finish_reason"
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["text"] = "".join(parts) or None
        result["error"] = describe_error(exc)
    except Exception as exc:  # URLError, read timeout, broken stream
        result["text"] = "".join(parts) or None
        result["error"] = describe_error(exc)
    result = _finish_result(result, started)
    _attach_io(result, io_before, getattr(args, "io_pid", None), io_ssh)
    return result


def _cell(value) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("|", "\\|")


def _raw_text(result: dict) -> str:
    if result.get("text"):
        return result["text"]
    if result.get("error"):
        return f"ERROR: {result['error']}"
    return "(no text)"


def _kv_hit_cell(result: dict) -> str:
    """Server-reported KV-cache hit rate, when the endpoint reports one (cache state evidence)."""
    stats = result.get("server_stats") or {}
    value = stats.get("kv_cache_hit_rate") if isinstance(stats, dict) else None
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return _cell(value)


def _rchar_cell(result: dict) -> str:
    """Logical bytes read per completion token from /proc/<pid>/io, when sampled."""
    io = result.get("io")
    value = io.get("rchar_per_completion_token") if isinstance(io, dict) else None
    return "" if value is None else str(value)


def _io_note(result: dict) -> str | None:
    io = result.get("io")
    if not isinstance(io, dict):
        return None
    if io.get("error"):
        return str(io["error"])
    return (
        f"rchar={io.get('rchar')} B ({io.get('rchar_per_s')} B/s, "
        f"{io.get('rchar_per_completion_token')} B/completion token); "
        f"read_bytes={io.get('read_bytes')} B ({io.get('read_bytes_per_s')} B/s, "
        f"{io.get('read_bytes_per_completion_token')} B/completion token)"
    )


def _cache_note(result: dict) -> str | None:
    """One-line cache/timing summary from x_deepseek_spark_stats, or None."""
    stats = result.get("server_stats")
    if not isinstance(stats, dict) or not stats:
        return None
    bits: list[str] = []
    if "kv_cache_hit_rate" in stats:
        bits.append(f"kv_cache_hit_rate={stats['kv_cache_hit_rate']}")
    layers = stats.get("engram_layers")
    if isinstance(layers, list):
        picked = [
            {
                key: layer.get(key)
                for key in (
                    "layer_id",
                    "hit_rate",
                    "mean_prefetch_stall_us",
                    "tier1_drm_hits",
                    "tier2_uma_hits",
                    "tier3_nvme_misses",
                )
            }
            for layer in layers
            if isinstance(layer, dict)
        ]
        bits.append("engram=" + json.dumps(picked, sort_keys=True))
    timing = stats.get("timing")
    if isinstance(timing, dict):
        bits.append(
            "timing="
            + json.dumps({field: timing.get(field) for field in SERVER_TIMING_FIELDS if field in timing}, sort_keys=True)
        )
    return "; ".join(bits) if bits else None


def render_markdown(document: dict) -> str:
    lines = [
        f"# Serve output probe - {document.get('tag') or document['base_url']}",
        "",
        f"Generated {document['generated_utc']}.",
        "",
        f"- base URL: {document['base_url']}",
        f"- model: {document['model']}",
        f"- endpoint: {document.get('endpoint', DEFAULT_ENDPOINT)}",
        f"- preset: {document.get('preset', '')}",
        f"- stream: {document['stream']}",
        f"- max_tokens: {document['max_tokens']} (per-prompt overrides apply)",
        f"- temperature: {document['temperature']}",
        f"- per-request timeout: {document['request_timeout_s']} s",
        f"- io sampling: pid={document.get('io_pid')} transport={document.get('io_ssh') or 'local'}",
        "",
        "| # | id | prompt | completion text | prompt tok | completion tok | max tok | wall s | tok/s | TTFT s | finish | KV hit | rchar/tok |",
        "| --: | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | :-- | --: | --: |",
    ]
    for result in document["results"]:
        lines.append(
            "| {index} | {id} | {prompt} | {text} | {prompt_tokens} | {completion_tokens} | "
            "{max_tokens} | {wall_s} | {tokens_per_s} | {ttft_s} | {finish} | {kv_hit} | {rchar_tok} |".format(
                index=result["index"],
                id=_cell(result["id"]),
                prompt=_cell(result["prompt"]),
                text=_cell(_raw_text(result)) if result.get("text") is not None else _cell(result.get("error")),
                prompt_tokens=result.get("prompt_tokens"),
                completion_tokens=result.get("completion_tokens"),
                max_tokens=result.get("max_tokens"),
                wall_s=result["wall_s"],
                tokens_per_s=result["tokens_per_s"],
                ttft_s=result.get("ttft_s"),
                finish=_cell(result.get("finish_reason")),
                kv_hit=_kv_hit_cell(result),
                rchar_tok=_rchar_cell(result),
            )
        )
    summary = document["summary"]
    lines.extend(
        [
            "",
            f"Summary: {summary['ok']}/{summary['requests']} requests ok; "
            f"prompt tokens {summary['prompt_tokens']}, completion tokens {summary['completion_tokens']}, "
            f"wall {summary['wall_s']} s, {summary['tokens_per_s']} completion tokens/s.",
            "",
            "## Raw completions",
            "",
        ]
    )
    for result in document["results"]:
        lines.append(f"### {result['id']}")
        lines.append("")
        lines.append(
            f"- finish_reason={result.get('finish_reason')!r} "
            f"prompt_tokens={result.get('prompt_tokens')} "
            f"completion_tokens={result.get('completion_tokens')} "
            f"wall_s={result['wall_s']} tokens_per_s={result['tokens_per_s']} "
            f"ttft_s={result.get('ttft_s')} ok={result['ok']}"
        )
        if result.get("server_timing"):
            lines.append(f"- server timing: {json.dumps(result['server_timing'], sort_keys=True)}")
        cache_note = _cache_note(result)
        if cache_note:
            lines.append(f"- cache: {cache_note}")
        io_note = _io_note(result)
        if io_note:
            lines.append(f"- io: {io_note}")
        lines.append("")
        text = _raw_text(result)
        lines.extend("    " + line for line in text.splitlines())
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def build_document(
    *,
    args: argparse.Namespace,
    models_document: dict,
    model: str,
    results: list[dict],
    started_utc: str,
    finished_utc: str,
    models_fetch_s: float,
) -> dict:
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in results)
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in results)
    wall_s = round(sum(float(r["wall_s"]) for r in results), 6)
    summary = {
        "requests": len(results),
        "ok": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_s": wall_s,
        "tokens_per_s": round(completion_tokens / wall_s, 6) if wall_s > 0 else 0.0,
        "models_fetch_s": round(models_fetch_s, 6),
        "stream": bool(args.stream),
    }
    return {
        "schema": SCHEMA,
        "generated_utc": finished_utc,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "base_url": args.base_url,
        "model": model,
        "endpoint": getattr(args, "endpoint", DEFAULT_ENDPOINT),
        "preset": getattr(args, "preset", DEFAULT_PRESET),
        "stream": bool(args.stream),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "request_timeout_s": args.timeout,
        "io_pid": getattr(args, "io_pid", None),
        "io_ssh": getattr(args, "io_ssh", None),
        "tag": args.tag,
        "models": models_document,
        "results": results,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure text, latency and tokens/s on an OpenAI-compatible completion or chat endpoint.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="serve base URL (default: %(default)s)")
    parser.add_argument("--model", default=None, help="model id; default: first id from /v1/models")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=DEFAULT_PRESET,
        help="named prompt preset (default: %(default)s); tp4-chat selects the three chat prompts",
    )
    parser.add_argument(
        "--endpoint",
        choices=ENDPOINTS,
        default=None,
        help="completion (/v1/completions) or chat (/v1/chat/completions); default: from the preset",
    )
    parser.add_argument("--prompts", default=None, help="JSON list of strings or {id, prompt, max_tokens} objects")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="completion cap (default: %(default)s)")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="sampling temperature (default: %(default)s)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="per-request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument("--stream", action="store_true", help="request SSE streaming and record TTFT/chunks")
    parser.add_argument(
        "--io-pid",
        type=int,
        default=None,
        help="serve pid; sample /proc/<pid>/io around each request (Linux node only)",
    )
    parser.add_argument(
        "--io-ssh",
        default=None,
        help="ssh target for the io pid (e.g. user@198.51.100.80); samples are read over ssh",
    )
    parser.add_argument("--tag", default="", help="label written into the JSON and markdown")
    parser.add_argument("--out-json", default=None, help="JSON output path (default: serve-output-<utc>.json)")
    parser.add_argument("--out-md", default=None, help="markdown output path (default: serve-output-<utc>.md)")
    parser.add_argument("--dry-run", action="store_true", help="print the planned requests and exit; no network")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.endpoint is None:
        args.endpoint = PRESETS[args.preset]["endpoint"]
    try:
        prompts = select_prompts(args)
    except Exception as exc:
        print(f"serve_output_probe: bad prompts: {describe_error(exc)}", file=sys.stderr)
        return 2
    if args.dry_run:
        plan = {
            "base_url": args.base_url,
            "model": args.model or "(first id from /v1/models)",
            "preset": args.preset,
            "endpoint": args.endpoint,
            "stream": bool(args.stream),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "timeout_s": args.timeout,
            "io_pid": args.io_pid,
            "io_ssh": args.io_ssh,
            "prompts": prompts,
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    models_started = time.monotonic()
    try:
        models_document = fetch_models(args.base_url, args.timeout)
        model = resolve_model(models_document, args.model)
    except Exception as exc:
        print(f"serve_output_probe: /v1/models failed: {describe_error(exc)}", file=sys.stderr)
        return 2
    models_fetch_s = time.monotonic() - models_started

    started_utc = utc_now()
    results: list[dict] = []
    for index, item in enumerate(prompts):
        item = dict(item, index=index)
        probe = probe_stream if args.stream else probe_buffered
        result = probe(args.base_url, model, item, args)
        results.append(result)
        print(
            json.dumps(
                {
                    "id": result["id"],
                    "ok": result["ok"],
                    "wall_s": result["wall_s"],
                    "prompt_tokens": result.get("prompt_tokens"),
                    "completion_tokens": result.get("completion_tokens"),
                    "tokens_per_s": result["tokens_per_s"],
                    "text": result.get("text"),
                    "error": result.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finished_utc = utc_now()

    document = build_document(
        args=args,
        models_document=models_document,
        model=model,
        results=results,
        started_utc=started_utc,
        finished_utc=finished_utc,
        models_fetch_s=models_fetch_s,
    )
    stem = f"serve-output-{_compact_utc()}"
    out_json = args.out_json or f"{stem}.json"
    out_md = args.out_md or f"{stem}.md"
    try:
        _write_text(out_json, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        _write_text(out_md, render_markdown(document))
    except Exception as exc:
        print(f"serve_output_probe: cannot write output: {describe_error(exc)}", file=sys.stderr)
        return 2
    print(f"wrote {out_json} and {out_md}", file=sys.stderr)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
