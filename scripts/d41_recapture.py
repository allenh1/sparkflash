#!/usr/bin/env python3
"""D41 idle re-capture: capture the D41 floor classes from a RUNNING vLLM, with an idleness proof.

Why this exists
---------------
D41 freezes its floor from a measured floor at >=3x noise. The cheapest legitimate floor is the
class that can be captured WITHOUT stopping the instance: repeated identical requests (the
zero-noise control) and close-margin distractor pairs captured while the server is idle. An
untrustworthy capture is worse than no capture, so this script proves idleness from the server's own
/metrics instead of asserting it:

* /v1/models is read first and the served root must be the released snapshot (D38). A mismatch is a
  refusal before any completion request is sent.
* /metrics must expose num_requests_running and num_requests_waiting. A missing metrics endpoint,
  or a missing metric, is a refusal before any completion request is sent.
* Immediately before a block the metrics snapshot must be idle (running == 0, waiting == 0). If it
  is not, the run refuses BEFORE the block's first completion; it never mixes classes silently.
* The pre-block snapshot is taken twice (settle poll, then a confirmation snapshot right before the
  first request). If the two disagree the run refuses before any completion.
* A metrics snapshot is taken after every completion. Requests other than ours tag the block
  "busy": it is recorded as class "busy_distractor", never silently mixed with the idle classes.
* A request failure aborts the run: no partial block is written as complete, and --out is written
  only after the whole capture succeeds (written to a temporary file, then renamed).

What it captures
----------------
The plan file (schema "sparkflash-d41-capture-plan/1") lists blocks and their request bodies:

* "repeat":>=2 identical bodies sent consecutively; the pair is the zero-noise control.
* "idle_distractor": pairs of bodies that differ by prompt_len at a close-margin position; the pair
  envelope is the floor.
* "busy_distractor": an optional "background" request (one long generation in a thread) makes the
  probe requests run under a controlled co-tenant load; metrics_during must show running >= 2.

The output is schema "sparkflash-d41-capture/1", exactly the input of scripts/d41_floors.py
(task-41, serve-dev). Token ids are never compared as evidence (D39) and determinism across runs is
never assumed: the repeat class is compared only inside one capture, and the floor script decides
whether it is bit-identical.

Host-only: no GPU, no node access, HTTP to --base-url only. Run with --dry-run to validate a plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_EXPECT_ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_PROMPT_LOGPROBS = 1000
DEFAULT_RUNNING_METRIC = "vllm:num_requests_running"
DEFAULT_WAITING_METRIC = "vllm:num_requests_waiting"
DEFAULT_REQUEST_TIMEOUT = 900.0
DEFAULT_SETTLE_SECONDS = 8.0
DEFAULT_SETTLE_POLL = 0.25
BACKGROUND_START_WAIT = 30.0

CAPTURE_SCHEMA = "sparkflash-d41-capture/1"
PLAN_SCHEMA = "sparkflash-d41-capture-plan/1"

KNOWN_CLASSES = ("repeat", "idle_distractor", "busy_distractor")

#: Block-level prompt_len contract (serve-dev, frozen): int when every record in the block has the
#: same prompt_len, null when they differ. Distractor blocks always differ, repeat blocks never do.


class RecaptureError(RuntimeError):
    """A capture that cannot produce trustworthy evidence."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise RecaptureError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RecaptureError(f"{path} is not JSON: {error}") from error


def write_atomic(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".d41-capture-", suffix=".json", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=1)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


# --------------------------------------------------------------------------------------------
# /metrics
# --------------------------------------------------------------------------------------------


def parse_metrics(text: str) -> dict:
    """Parse Prometheus text into {"samples": {name: [{"labels", "value"}]}, "sha256"}."""
    samples: dict[str, list[dict]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        if not rest:
            continue
        parts = rest.split()
        try:
            value = float(parts[0])
        except (IndexError, ValueError):
            continue
        name, _, labels = head.partition("{")
        samples.setdefault(name, []).append({"labels": labels, "value": value})
    return {"samples": samples, "sha256": sha256_bytes(text.encode("utf-8"))}


def metric_sum(metrics: dict, name: str) -> float | None:
    entries = metrics["samples"].get(name)
    if not entries:
        return None
    return float(sum(entry["value"] for entry in entries))


def request_gauges(metrics: dict, running_name: str, waiting_name: str) -> dict:
    running = metric_sum(metrics, running_name)
    waiting = metric_sum(metrics, waiting_name)
    missing = []
    if running is None:
        missing.append(running_name)
    if waiting is None:
        missing.append(waiting_name)
    if missing:
        raise RecaptureError(
            f"/metrics does not expose {missing}; cannot prove idleness, and the floor script "
            "refuses a capture without an idleness proof"
        )
    return {"num_requests_running": int(running), "num_requests_waiting": int(waiting)}


def is_idle(gauges: dict) -> bool:
    return gauges["num_requests_running"] == 0 and gauges["num_requests_waiting"] == 0


class HttpTransport:
    """The only object the capture uses to talk to the server (injectable for unit tests)."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_REQUEST_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_text(self, path: str) -> tuple[bytes, str]:
        url = self.base_url + path
        try:
            with urllib.request.urlopen(url, timeout=min(self.timeout, 60.0)) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read()[:400].decode("utf-8", "replace")
            raise RecaptureError(f"HTTP {error.code} from {url}: {detail}") from error
        except urllib.error.URLError as error:
            raise RecaptureError(f"cannot reach {url}: {error.reason}") from error
        return body, body.decode("utf-8", "replace")

    def post_json(self, path: str, payload: dict) -> tuple[bytes, dict]:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read()[:400].decode("utf-8", "replace")
            raise RecaptureError(f"HTTP {error.code} from {url}: {detail}") from error
        except urllib.error.URLError as error:
            raise RecaptureError(f"cannot reach {url}: {error.reason}") from error
        try:
            return body, json.loads(body)
        except json.JSONDecodeError as error:
            raise RecaptureError(f"response from {url} is not JSON: {error}") from error


# --------------------------------------------------------------------------------------------
# Serving contract (D38)
# --------------------------------------------------------------------------------------------


def parse_served(models: dict) -> tuple[str | None, str | None]:
    data = models.get("data")
    if not isinstance(data, list) or not data:
        raise RecaptureError("/v1/models returned no served model")
    entry = data[0]
    return entry.get("id"), entry.get("root")


def root_matches(root: str | None, expect: str | None) -> bool:
    if not expect:
        return True
    if not root:
        return False
    normalised = root.rstrip("/")
    return normalised == expect or normalised.endswith("/" + expect)


def check_served(transport, expect_root: str) -> dict:
    raw, text = transport.get_text("/v1/models")
    try:
        models = json.loads(text)
    except json.JSONDecodeError as error:
        raise RecaptureError(f"/v1/models is not JSON: {error}") from error
    served_id, served_root = parse_served(models)
    if not root_matches(served_root, expect_root):
        raise RecaptureError(
            f"served root {served_root!r} is not the released snapshot {expect_root!r}; "
            "refusing to capture D41 evidence against another checkpoint (D38)"
        )
    return {"id": served_id, "root": served_root, "models_response_sha256": sha256_bytes(raw)}


# --------------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------------


def build_request(recipe: dict, prompt: list[int], model: str) -> dict:
    return {
        "model": model,
        "prompt": [int(token) for token in prompt],
        "max_tokens": int(recipe["max_tokens"]),
        "temperature": float(recipe["temperature"]),
        "top_p": float(recipe["top_p"]),
        "ignore_eos": bool(recipe["ignore_eos"]),
        "prompt_logprobs": int(recipe["prompt_logprobs"]),
    }


def validate_plan(plan: dict, path: str) -> dict:
    if plan.get("schema") != PLAN_SCHEMA:
        raise RecaptureError(f"{path}: unknown plan schema {plan.get('schema')!r} (want {PLAN_SCHEMA})")
    raw_recipe = plan.get("recipe") or {}
    recipe = {
        "max_tokens": int(raw_recipe.get("max_tokens", 1)),
        "temperature": float(raw_recipe.get("temperature", 0.0)),
        "top_p": float(raw_recipe.get("top_p", 1.0)),
        "ignore_eos": bool(raw_recipe.get("ignore_eos", True)),
        "prompt_logprobs": int(raw_recipe.get("prompt_logprobs", DEFAULT_PROMPT_LOGPROBS)),
    }
    if recipe["max_tokens"] != 1:
        raise RecaptureError(f"{path}: recipe.max_tokens must be 1 for a prompt_logprobs capture")
    if recipe["prompt_logprobs"] < 1000:
        raise RecaptureError(
            f"{path}: recipe.prompt_logprobs {recipe['prompt_logprobs']} is below the recipe cap 1000"
        )
    blocks = plan.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise RecaptureError(f"{path}: plan has no blocks")
    seen_ids: set[str] = set()
    for block in blocks:
        block_id = block.get("id")
        if not isinstance(block_id, str) or not block_id:
            raise RecaptureError(f"{path}: a block has no id")
        if block_id in seen_ids:
            raise RecaptureError(f"{path}: duplicate block id {block_id!r}")
        seen_ids.add(block_id)
        block_class = block.get("class")
        if block_class not in KNOWN_CLASSES:
            raise RecaptureError(f"{path}: block {block_id!r} has unknown class {block_class!r}")
        requests = block.get("requests")
        if not isinstance(requests, list) or not requests:
            raise RecaptureError(f"{path}: block {block_id!r} has no requests")
        request_ids: list[str] = []
        prompts: dict[str, list[int]] = {}
        for request in requests:
            request_id = request.get("id")
            if not isinstance(request_id, str) or not request_id:
                raise RecaptureError(f"{path}: block {block_id!r} has a request without an id")
            if request_id in prompts:
                raise RecaptureError(f"{path}: block {block_id!r} has duplicate request id {request_id!r}")
            prompt = request.get("prompt")
            if not isinstance(prompt, list) or not prompt:
                raise RecaptureError(f"{path}: block {block_id!r} request {request_id!r} has no prompt")
            if any(not isinstance(token, int) or token < 0 for token in prompt):
                raise RecaptureError(f"{path}: block {block_id!r} request {request_id!r} has a non-token id")
            request_ids.append(request_id)
            prompts[request_id] = list(prompt)
        pairs = block.get("pairs", [])
        if not isinstance(pairs, list):
            raise RecaptureError(f"{path}: block {block_id!r} pairs is not a list")
        normalised_pairs: list[list[str]] = []
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise RecaptureError(f"{path}: block {block_id!r} has a pair that is not two ids")
            for member in pair:
                if member not in prompts:
                    raise RecaptureError(f"{path}: block {block_id!r} pair member {member!r} is not a request id")
            normalised_pairs.append([pair[0], pair[1]])
        if block_class == "repeat":
            if len(requests) < 2:
                raise RecaptureError(f"{path}: repeat block {block_id!r} needs at least two requests")
            if len({tuple(prompts[request_id]) for request_id in request_ids}) != 1:
                raise RecaptureError(f"{path}: repeat block {block_id!r} must send the same body every time")
        if block_class in ("idle_distractor", "busy_distractor") and not normalised_pairs:
            raise RecaptureError(f"{path}: {block_class} block {block_id!r} needs pairs")
        background = block.get("background")
        if background is not None:
            if not isinstance(background, dict) or not isinstance(background.get("prompt"), list):
                raise RecaptureError(f"{path}: block {block_id!r} background has no prompt")
            if any(not isinstance(t, int) or t < 0 for t in background["prompt"]):
                raise RecaptureError(f"{path}: block {block_id!r} background prompt has a non-token id")
            if int(background.get("max_tokens", 0)) < 2:
                raise RecaptureError(f"{path}: block {block_id!r} background needs max_tokens >= 2")
        if block_class == "busy_distractor" and background is None:
            raise RecaptureError(f"{path}: busy_distractor block {block_id!r} needs a background load")
        block["_ids"] = request_ids
        block["_prompts"] = prompts
        block["_pairs"] = normalised_pairs
    return {"recipe": recipe, "blocks": blocks}


# --------------------------------------------------------------------------------------------
# Capture primitives
# --------------------------------------------------------------------------------------------


def parse_positions(prompt_logprobs, prompt: list[int]) -> list[dict]:
    """prompt_logprobs[i] -> device position i-1; top is [[token, logprob], ...]."""
    if not isinstance(prompt_logprobs, list):
        raise RecaptureError(f"prompt_logprobs is not a list but {type(prompt_logprobs).__name__}")
    if len(prompt_logprobs) != len(prompt):
        raise RecaptureError(
            f"prompt_logprobs has {len(prompt_logprobs)} entries for {len(prompt)} prompt tokens"
        )
    if prompt_logprobs[0] is not None:
        raise RecaptureError("prompt_logprobs[0] is not null")
    positions = []
    for index in range(1, len(prompt_logprobs)):
        entry = prompt_logprobs[index]
        if not isinstance(entry, dict) or not entry:
            raise RecaptureError(f"prompt_logprobs[{index}] is {entry!r}, not an id-keyed dict")
        distribution: dict[int, float] = {}
        for key, value in entry.items():
            try:
                token_id = int(key)
            except (TypeError, ValueError) as error:
                raise RecaptureError(f"prompt_logprobs[{index}] key {key!r} is not a token id") from error
            try:
                logprob = float(value["logprob"] if isinstance(value, dict) else value)
            except (KeyError, TypeError, ValueError) as error:
                raise RecaptureError(f"prompt_logprobs[{index}][{key!r}] has no usable logprob") from error
            if logprob != logprob or logprob in (float("inf"), float("-inf")):
                raise RecaptureError(f"prompt_logprobs[{index}][{token_id}] is not finite: {logprob!r}")
            distribution[token_id] = logprob
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        position = index - 1
        if position + 2 > len(prompt):
            raise RecaptureError(f"position {position} needs prompt_len >= {position + 2}, got {len(prompt)}")
        positions.append(
            {
                "position": position,
                "logprob_count": len(distribution),
                "top": [[token, logprob] for token, logprob in ranked],
            }
        )
    return positions


class Snapshotter:
    def __init__(self, transport, running_name: str, waiting_name: str):
        self.transport = transport
        self.running_name = running_name
        self.waiting_name = waiting_name

    def snapshot(self) -> dict:
        raw, text = self.transport.get_text("/metrics")
        metrics = parse_metrics(text)
        gauges = request_gauges(metrics, self.running_name, self.waiting_name)
        return {
            "num_requests_running": gauges["num_requests_running"],
            "num_requests_waiting": gauges["num_requests_waiting"],
            "raw_sha256": metrics["sha256"],
            "at_utc": utc_now(),
        }

    def poll_idle(self, settle_seconds: float, poll: float, clock, pause) -> tuple[dict, bool]:
        deadline = clock() + settle_seconds
        while True:
            snapshot = self.snapshot()
            if is_idle(snapshot):
                return snapshot, True
            if clock() >= deadline:
                return snapshot, False
            pause(poll)

    def require_idle(self, settle_seconds: float, poll: float, clock, pause, context: str) -> dict:
        snapshot, idle = self.poll_idle(settle_seconds, poll, clock, pause)
        if not idle:
            raise RecaptureError(
                f"{context}: /metrics shows running={snapshot['num_requests_running']} "
                f"waiting={snapshot['num_requests_waiting']}; refusing before any completion "
                "(a non-idle baseline cannot be attributed to this capture)"
            )
        confirmation = self.snapshot()
        if not is_idle(confirmation):
            raise RecaptureError(
                f"{context}: metrics snapshot mismatch before the first completion "
                f"(running={confirmation['num_requests_running']} "
                f"waiting={confirmation['num_requests_waiting']}); refusing"
            )
        return confirmation


def block_prompt_len(records: list[dict]) -> int | None:
    """int when every record shares one prompt_len, null when they differ (frozen schema)."""
    lengths = [record["prompt_len"] for record in records]
    if len(set(lengths)) == 1:
        return lengths[0]
    return None


def same_distribution(left: dict, right: dict) -> bool:
    """Exact equality of the returned distributions (floats only; no token-id equality evidence)."""
    pl = {entry["position"]: entry["top"] for entry in left["positions"]}
    pr = {entry["position"]: entry["top"] for entry in right["positions"]}
    return pl == pr


def start_background(transport, snapshotter, block, recipe, model, opts, log) -> dict:
    background = block["background"]
    payload = {
        "model": model,
        "prompt": [int(token) for token in background["prompt"]],
        "max_tokens": int(background["max_tokens"]),
        "temperature": 0.0,
        "ignore_eos": True,
    }
    state = {
        "payload_sha256": sha256_bytes(json.dumps(payload).encode("utf-8")),
        "max_tokens": int(background["max_tokens"]),
        "result": None,
        "error": None,
        "wall_s": None,
        "metrics_after_start": None,
    }

    def run() -> None:
        started = opts.clock()
        try:
            raw, response = transport.post_json("/v1/completions", payload)
            state["result"] = {"response_sha256": sha256_bytes(raw)}
        except RecaptureError as error:
            state["error"] = str(error)
        finally:
            state["wall_s"] = round(opts.clock() - started, 3)

    started = opts.clock()
    thread = threading.Thread(target=run, name="d41-background", daemon=True)
    state["thread"] = thread
    thread.start()
    start_wait = float(getattr(opts, "background_start_wait", BACKGROUND_START_WAIT))
    deadline = opts.clock() + start_wait
    while opts.clock() < deadline:
        during = snapshotter.snapshot()
        if during["num_requests_running"] >= 1 or during["num_requests_waiting"] >= 1:
            state["metrics_after_start"] = during
            log(f"block {block['id']}: background generation running (running={during['num_requests_running']})")
            return state
        opts.pause(0.1)
    thread.join(timeout=1.0)
    raise RecaptureError(
        f"block {block['id']}: background generation did not start within {start_wait}s; "
        "refusing to capture a busy block without co-tenancy evidence"
    )


def finish_background(state: dict, opts, log) -> dict:
    thread = state["thread"]
    thread.join(timeout=opts.request_timeout)
    info = {
        "request_sha256": state["payload_sha256"],
        "max_tokens": state["max_tokens"],
        "wall_s": state["wall_s"],
        "response_sha256": None,
        "error": state["error"],
        "thread_alive_at_join": bool(thread.is_alive()),
        "metrics_after_start": state["metrics_after_start"],
    }
    if state["result"]:
        info["response_sha256"] = state["result"]["response_sha256"]
    if thread.is_alive():
        log("background generation still running at join timeout; the server finishes it on its own")
    return info


def capture_block(transport, snapshotter, block, recipe, model, opts, log) -> dict:
    block_id = block["id"]
    block_class = block["class"]
    started = utc_now()
    log(f"block {block_id}: class={block_class} requests={len(block['_ids'])}")
    metrics_before = snapshotter.require_idle(
        opts.settle_seconds, opts.settle_poll, opts.clock, opts.pause, f"block {block_id} pre"
    )
    background_state = None
    if block.get("background"):
        background_state = start_background(transport, snapshotter, block, recipe, model, opts, log)
    records: list[dict] = []
    metrics_during: list[dict] = []
    busy_reason: str | None = None
    for request_id in block["_ids"]:
        prompt = block["_prompts"][request_id]
        payload = build_request(recipe, prompt, model)
        raw_request = json.dumps(payload).encode("utf-8")
        started_request = opts.clock()
        raw, response = transport.post_json("/v1/completions", payload)
        wall = round(opts.clock() - started_request, 3)
        choices = response.get("choices") if isinstance(response, dict) else None
        if not choices:
            raise RecaptureError(f"block {block_id} request {request_id}: response has no choices")
        choice = choices[0]
        if "prompt_logprobs" not in choice:
            raise RecaptureError(
                f"block {block_id} request {request_id}: response has no prompt_logprobs "
                "(is --max-logprobs 1000 set on the server?)"
            )
        record = {
            "id": request_id,
            "prompt_len": len(prompt),
            "batch_size": 1,
            "prompt_sha256": sha256_bytes(json.dumps(prompt).encode("utf-8")),
            "request_sha256": sha256_bytes(raw_request),
            "response_sha256": sha256_bytes(raw),
            "wall_s": wall,
            "repeat_of": None,
            "positions": parse_positions(choice["prompt_logprobs"], prompt),
        }
        records.append(record)
        if block_class == "busy_distractor":
            during = snapshotter.snapshot()
            metrics_during.append(during)
        else:
            during, idle = snapshotter.poll_idle(
                opts.settle_seconds, opts.settle_poll, opts.clock, opts.pause
            )
            metrics_during.append(during)
            if not idle:
                busy_reason = busy_reason or (
                    f"traffic during block: running={during['num_requests_running']} "
                    f"waiting={during['num_requests_waiting']}"
                )
    load_generator = None
    if background_state is not None:
        load_generator = finish_background(background_state, opts, log)
    if block_class == "busy_distractor":
        metrics_after, _ = snapshotter.poll_idle(
            opts.settle_seconds, opts.settle_poll, opts.clock, opts.pause
        )
        if not any(
            snapshot["num_requests_running"] > 1 or snapshot["num_requests_waiting"] > 0
            for snapshot in metrics_during
        ):
            raise RecaptureError(
                f"block {block_id}: no co-tenancy evidence (no metrics_during snapshot with "
                "running>1 or waiting>0); refusing to write a busy class without load evidence"
            )
        verdict = "busy"
        observed_other_traffic = True
        class_effective = "busy_distractor"
    else:
        metrics_after, idle_after = snapshotter.poll_idle(
            opts.settle_seconds, opts.settle_poll, opts.clock, opts.pause
        )
        if not idle_after:
            busy_reason = busy_reason or (
                f"traffic after block: running={metrics_after['num_requests_running']} "
                f"waiting={metrics_after['num_requests_waiting']}"
            )
        verdict = "busy" if busy_reason else "idle"
        observed_other_traffic = bool(busy_reason)
        class_effective = block_class if verdict == "idle" else "busy_distractor"
    idleness = {
        "metrics_source": "/metrics",
        "metrics_before": {
            "num_requests_running": metrics_before["num_requests_running"],
            "num_requests_waiting": metrics_before["num_requests_waiting"],
            "raw_sha256": metrics_before["raw_sha256"],
        },
        "metrics_after": {
            "num_requests_running": metrics_after["num_requests_running"],
            "num_requests_waiting": metrics_after["num_requests_waiting"],
            "raw_sha256": metrics_after["raw_sha256"],
        },
        "metrics_during": metrics_during,
        "observed_other_traffic": observed_other_traffic,
        "external_peers": [],
        "idle_verdict": verdict,
    }
    if block_class == "repeat":
        first = records[0]
        for record in records[1:]:
            record["repeat_of"] = first["id"]
        stable = all(same_distribution(first, record) for record in records[1:])
    else:
        stable = True
    block_out = {
        "block_id": block_id,
        "class": class_effective,
        "started_utc": started,
        "finished_utc": utc_now(),
        "prompt_len": block_prompt_len(records),
        "batch_size": 1,
        "idleness": idleness,
        "pairs": [list(pair) for pair in block["_pairs"]],
        "requests": records,
    }
    if load_generator is not None:
        block_out["load_generator"] = load_generator
    if busy_reason:
        block_out["busy_reason"] = busy_reason
    block_out["_stable"] = stable
    return block_out


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def capture_block_ids(plan: dict) -> list[str]:
    return [block["id"] for block in plan["blocks"]]


def plan_summary(plan: dict) -> list[dict]:
    table = []
    for block in plan["blocks"]:
        table.append(
            {
                "block_id": block["id"],
                "class": block["class"],
                "requests": [
                    {"id": request_id, "prompt_len": len(block["_prompts"][request_id])}
                    for request_id in block["_ids"]
                ],
                "pairs": block["_pairs"],
                "background": None if not block.get("background") else {
                    "prompt_len": len(block["background"]["prompt"]),
                    "max_tokens": block["background"]["max_tokens"],
                },
            }
        )
    return table


def orchestrate(args, transport, clock, pause, log) -> dict:
    plan = load_json(args.plan)
    plan_sha = sha256_file(args.plan)
    validated = validate_plan(plan, args.plan)
    recipe = validated["recipe"]
    if args.dry_run:
        return {
            "status": "dry-run",
            "plan": {"path": args.plan, "sha256": plan_sha, "schema": plan["schema"]},
            "recipe": recipe,
            "blocks": plan_summary(plan),
        }
    args.clock = clock
    args.pause = pause
    served = check_served(transport, args.expect_root)
    model = served["id"] or args.model
    log(f"served id={served['id']!r} root={served['root']!r} (root verified)")
    snapshotter = Snapshotter(transport, args.running_metric, args.waiting_metric)
    # Prove the metrics endpoint before the first completion.
    probe = snapshotter.snapshot()
    log(f"metrics endpoint present: running={probe['num_requests_running']} waiting={probe['num_requests_waiting']}")
    started = utc_now()
    blocks_out: list[dict] = []
    unstable: list[str] = []
    mixed = False
    for block in plan["blocks"]:
        block_out = capture_block(transport, snapshotter, block, recipe, model, args, log)
        stable = block_out.pop("_stable")
        blocks_out.append(block_out)
        if block_out["class"] == "busy_distractor":
            mixed = True
        if not stable:
            unstable.append(block_out["block_id"])
        log(
            f"block {block_out['block_id']}: class={block_out['class']} "
            f"verdict={block_out['idleness']['idle_verdict']} requests={len(block_out['requests'])}"
        )
    finished = utc_now()
    distinct_prompts = {
        tuple(block["_prompts"][request_id]) for block in plan["blocks"] for request_id in block["_ids"]
    }
    if unstable:
        status = "unstable"
    elif mixed:
        status = "mixed"
    else:
        status = "ok"
    warnings = []
    if unstable:
        warnings.append(
            "repeat class is NOT bit-identical within this capture: " + ", ".join(unstable)
        )
    if mixed:
        warnings.append("at least one block was tagged busy_distractor; idle classes may be absent")
    return {
        "schema": CAPTURE_SCHEMA,
        "capture_id": args.capture_id or f"d41-recapture-{started}",
        "producer": {"script": "scripts/d41_recapture.py", "script_sha256": sha256_file(__file__)},
        "started_utc": started,
        "finished_utc": finished,
        "status": status,
        "warnings": warnings,
        "plan": {"path": args.plan, "sha256": plan_sha, "schema": plan["schema"]},
        "served": served,
        "expect_root": args.expect_root,
        "root_verified": root_matches(served["root"], args.expect_root),
        "recipe": {
            "endpoint": "/v1/completions",
            "max_tokens": recipe["max_tokens"],
            "temperature": recipe["temperature"],
            "top_p": recipe["top_p"],
            "prompt_logprobs": recipe["prompt_logprobs"],
            "ignore_eos": recipe["ignore_eos"],
            "model": model,
        },
        "suite": {
            "path": args.plan,
            "sha256": plan_sha,
            "prompt_count": len(distinct_prompts),
            "note": "fixture-prompt plan serves as the prompt suite; its sha256 is the provenance",
        },
        "blocks": blocks_out,
    }


def run_cli(argv: list[str] | None, transport_factory) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--plan", required=True, help="capture plan JSON (sparkflash-d41-capture-plan/1)")
    parser.add_argument("--out", help="capture JSON to write (nothing is written on refusal)")
    parser.add_argument("--log", help="log path (default: --out with .log)")
    parser.add_argument("--expect-root", default=DEFAULT_EXPECT_ROOT,
                        help=f"released snapshot root (default {DEFAULT_EXPECT_ROOT}); empty disables the check")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS,
                        help="how long to wait for /metrics to show the server idle")
    parser.add_argument("--settle-poll", type=float, default=DEFAULT_SETTLE_POLL)
    parser.add_argument("--background-start-wait", type=float, default=BACKGROUND_START_WAIT,
                        help="how long a busy block waits for its background generation to show in /metrics")
    parser.add_argument("--running-metric", default=DEFAULT_RUNNING_METRIC)
    parser.add_argument("--waiting-metric", default=DEFAULT_WAITING_METRIC)
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry-run", action="store_true", help="validate the plan; no network")
    args = parser.parse_args(argv)
    if args.expect_root == "":
        args.expect_root = None
    if args.tag:
        args.capture_id = args.capture_id or args.tag
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run")
    log_path = args.log or (args.out + ".log" if args.out else os.path.join(tempfile.gettempdir(), "d41-recapture.log"))
    lines: list[str] = []

    def log(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        lines.append(line)

    transport = None if args.dry_run else transport_factory(args.base_url, args.request_timeout)
    log(f"d41 re-capture start plan={args.plan} out={args.out} dry_run={args.dry_run}")
    try:
        result = orchestrate(args, transport, time.monotonic, time.sleep, log)
    except RecaptureError as error:
        log(f"REFUSED: {error}")
        with open(log_path, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        print(f"d41_recapture: {error}", file=sys.stderr)
        return 2
    if result["status"] == "dry-run":
        with open(log_path, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        print(json.dumps(result["blocks"], indent=1))
        return 0
    write_atomic(args.out, result)
    with open(log_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} (status={result['status']}, blocks={len(result['blocks'])})")
    return 0 if result["status"] != "unstable" else 4


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv, lambda base_url, timeout: HttpTransport(base_url, timeout))


if __name__ == "__main__":
    sys.exit(main())
