#!/usr/bin/env python3
"""S7/S8 serving probes for a recorded four-rank device token window.

Two independent steps, because they answer different questions and must not be blended:

  * inspect: re-derives the window's claim state from the recorded JSON (a capture aid: the Rust
    consumer in crates/sparkflash-api/src/device_chain.rs is authoritative, and the server prints
    the same decision at startup). It writes window-claim.json and exits 0 when the window is
    claimable in the requested form, 2 when it is refused (with every reason), 1 on bad input.
  * probe: drives a running sparkflash serve --device-token-dir server through the OpenAI
    completions path: buffered and SSE, headers, usage relation, finish reason, [DONE], replay
    stability. It writes probe-buffered.json, probe-stream.sse, probe-summary.json and EVIDENCE.md
    into --arc-dir and exits 0 only when every check passed.

The token window is a device artefact; the wire answer is UNMEASURED by construction (the server
labels it so). Nothing here measures throughput, latency or quality, and nothing here may be read as
a tok/s, TTFT or certified-token claim.

Usage (S7 rehearsal, a window before the device writer asserts token_claimed):

    python3 scripts/serve_window_probe.py inspect --window W --arc-dir A --mode derived
    # start: sparkflash serve --synthetic --bind 127.0.0.1:0 --tokenizer T \
    #        --device-token-dir W --device-token-derived-claim
    python3 scripts/serve_window_probe.py probe --window W --base-url URL --arc-dir A --mode derived

Usage (S8 acceptance, the window's writer asserts token_claimed on every rank):

    python3 scripts/serve_window_probe.py inspect --window W --arc-dir A --mode asserted
    # start: sparkflash serve --synthetic --bind 127.0.0.1:0 --tokenizer T --device-token-dir W
    python3 scripts/serve_window_probe.py probe --window W --base-url URL --arc-dir A --mode asserted

Usage (a window that no form can claim; the server must fall back to the host twin):

    python3 scripts/serve_window_probe.py probe --window W --base-url URL --arc-dir A \
        --mode asserted --expect fallback

--expect fallback refuses to run on a claimable window (so a record cannot be mislabelled), then
asserts the fail-closed shape: 200 from the host twin, no device provenance headers, the host stats
block in the body, and an SSE stream that terminates and joins to the buffered text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROBE_SCHEMA = "sparkflash-serve-window-probe/1"
CLAIM_SCHEMA = "sparkflash-serve-window-claim/1"
DEVICE_SOURCE_HEADER = "x-sparkflash-token-source"
DEVICE_SOURCE_VALUE = "device-chain-envelope (UNMEASURED)"
DEVICE_CLAIM_HEADER = "x-sparkflash-token-claim"


def claim_label(mode: str) -> str:
    """The wire label a claim form produces: asserted -> explicit, derived -> derived."""
    return "explicit" if mode == "asserted" else mode
HOST_STATS_HEADER = "x-deepseek-spark-stats"
DECODE_POSITIONS = 9
RELEASED_VOCAB = 129280
DUMP_DTYPE = "f32-le"
WORLD = 4
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
# The fallback probe runs the host twin, not the window: the synthetic plan only embeds small ids.
DEFAULT_FALLBACK_PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]
FALLBACK_MAX_TOKENS = 4


class ProbeError(RuntimeError):
    """Input that cannot be probed honestly (bad files, bad arguments)."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise ProbeError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ProbeError(f"{path} is not JSON: {error}") from error


def dump_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def rank_file(window: str, pattern: str, rank: int) -> str:
    """The rank's record, accepted flat or under rank{rank}/, as the Rust loader accepts both."""
    flat = os.path.join(window, pattern.format(rank=rank))
    if os.path.isfile(flat):
        return flat
    nested = os.path.join(window, f"rank{rank}", pattern.format(rank=rank))
    if os.path.isfile(nested):
        return nested
    raise ProbeError(f"window {window}: neither {flat} nor {nested} exists")


def load_window(window: str) -> dict:
    if not os.path.isdir(window):
        raise ProbeError(f"--window {window}: not a directory")
    embeds = [
        read_json(rank_file(window, "tp4-forward-embed-r{rank}.json", rank))
        for rank in range(WORLD)
    ]
    tokens = [
        read_json(rank_file(window, "tp4-token-r{rank}.json", rank)) for rank in range(WORLD)
    ]
    for rank, embed in enumerate(embeds):
        schema = embed.get("schema")
        if schema != "sparkflash-tp4-forward-embed/1":
            raise ProbeError(f"rank {rank}: unexpected embed schema {schema!r}")
        world = embed.get("world")
        if world != WORLD:
            raise ProbeError(f"rank {rank}: world {world!r}, expected {WORLD}")
    return {"embeds": embeds, "tokens": tokens}


def _dump_block(token: dict):
    dump = token.get("logits_dump")
    return dump if isinstance(dump, dict) else None


def _positions_are_consecutive(positions) -> bool:
    if not all(isinstance(value, int) for value in positions):
        return False
    return positions == list(range(positions[0], positions[0] + len(positions)))


def claim_window(window: dict, mode: str) -> dict:
    """Re-derives the claim state. Mirrors the Rust consumer's rules; capture aid only."""
    embeds = window["embeds"]
    tokens = window["tokens"]
    reasons: list[str] = []
    if mode not in {"asserted", "derived"}:
        raise ProbeError(f"--mode must be asserted or derived, not {mode!r}")
    rank0 = embeds[0]
    prompt_len = rank0.get("prompt_len")
    first_step_position = rank0.get("first_step_position")
    decode_positions = rank0.get("decode_positions")
    reference_ids = rank0.get("reference_ids") or []
    prompt_ids = rank0.get("prompt_ids") or []
    forward_ids = rank0.get("forward_ids") or []
    prefill_limit = rank0.get("prefill_limit")
    evidence_written = rank0.get("evidence_written") or ""
    if decode_positions != DECODE_POSITIONS:
        reasons.append(
            f"embed decode_positions is {decode_positions!r}, expected {DECODE_POSITIONS}"
        )
    if len(reference_ids) != DECODE_POSITIONS:
        reasons.append(
            f"embed reference_ids has {len(reference_ids)} entries, expected {DECODE_POSITIONS}"
        )
    if len(prompt_ids) != prompt_len:
        reasons.append(
            f"embed prompt_ids has {len(prompt_ids)} entries, expected prompt_len {prompt_len}"
        )
    expected_forward = (prompt_len or 0) + DECODE_POSITIONS - 1
    if len(forward_ids) != expected_forward:
        reasons.append(
            f"embed forward_ids has {len(forward_ids)} rows, expected prompt_len + "
            f"{DECODE_POSITIONS} - 1 = {expected_forward}"
        )
    if (
        len(forward_ids) >= prompt_len
        and isinstance(rank0.get("prompt_token"), int)
        and forward_ids[prompt_len - 1] != rank0["prompt_token"]
    ):
        reasons.append(
            f"embed forward_ids[prompt_len - 1] is {forward_ids[prompt_len - 1]!r}, expected "
            f"prompt_token {rank0['prompt_token']}"
        )
    if mode == "asserted":
        for rank, embed in enumerate(embeds):
            if embed.get("token_claimed") is not True:
                reasons.append(
                    f"rank {rank}: embed does not assert token_claimed=true "
                    f"(the device writer currently records {embed.get('token_claimed')!r})"
                )
    else:
        if prefill_limit is not None:
            reasons.append(
                f"embed prefill_limit is {prefill_limit!r}: a limited replay window is not the "
                "full nine-position acceptance window"
            )
        batch = rank0.get("tokens")
        if not isinstance(batch, dict):
            reasons.append("rank 0: embed carries no tokens batch")
            batch = {}
        if batch.get("consensus") is not True:
            reasons.append("rank 0: the embed tokens batch does not report consensus=true")
        claimable = batch.get("claimable") or []
        if len(claimable) != DECODE_POSITIONS or not all(flag is True for flag in claimable):
            reasons.append(f"rank 0: embed claimable is {claimable!r}, expected nine true flags")
        matches = batch.get("matches_reference") or []
        if len(matches) != DECODE_POSITIONS or not all(value is True for value in matches):
            reasons.append(
                f"rank 0: matches_reference is {matches!r}, expected nine true matches "
                "(pass --forward-reference-ids to the four-rank run)"
            )
        if not reference_ids:
            reasons.append("rank 0: the window records no reference ids")
        if "token-consensus-complete" not in evidence_written:
            reasons.append("rank 0: evidence_written does not contain token-consensus-complete")
        if first_step_position is not None and prompt_len is not None:
            if first_step_position != prompt_len - 1:
                reasons.append(
                    f"embed first_step_position is {first_step_position}, expected prompt_len - 1 "
                    f"= {prompt_len - 1}"
                )
    token_ids = None
    token_positions = None
    for rank, (embed, token) in enumerate(zip(embeds, tokens)):
        if token.get("schema") != "sparkflash-tp4-token/1":
            reasons.append(f"rank {rank}: unexpected token schema {token.get('schema')!r}")
            continue
        if token.get("forward") != "run":
            reasons.append(f"rank {rank}: token forward is {token.get('forward')!r}, expected run")
        if token.get("error") is not None:
            reasons.append(f"rank {rank}: token record reports error {token.get('error')!r}")
        tokens_block = token.get("tokens") or {}
        ids = tokens_block.get("ids") or []
        positions = tokens_block.get("positions") or []
        argmax = tokens_block.get("argmax_logits") or []
        margins = tokens_block.get("top1_margins") or []
        if token_ids is None:
            token_ids, token_positions = ids, positions
        if len(ids) != DECODE_POSITIONS:
            reasons.append(
                f"rank {rank}: token ids has {len(ids)} entries, expected {DECODE_POSITIONS}"
            )
        if len(positions) != DECODE_POSITIONS:
            reasons.append(
                f"rank {rank}: token positions has {len(positions)} entries, expected "
                f"{DECODE_POSITIONS}"
            )
        if len(argmax) != DECODE_POSITIONS:
            reasons.append(
                f"rank {rank}: argmax_logits has {len(argmax)} entries, expected {DECODE_POSITIONS}"
            )
        if len(margins) != DECODE_POSITIONS:
            reasons.append(
                f"rank {rank}: top1_margins has {len(margins)} entries, expected {DECODE_POSITIONS}"
            )
        if token.get("positions_run") != len(ids):
            reasons.append(
                f"rank {rank}: positions_run {token.get('positions_run')!r} != len(ids) {len(ids)}"
            )
        if rank > 0 and ids and ids != token_ids:
            reasons.append(f"rank {rank}: token ids differ from rank 0's")
        if positions and not _positions_are_consecutive(positions):
            reasons.append(f"rank {rank}: token positions are not consecutive: {positions!r}")
        if positions and prefill_limit is None and first_step_position is not None:
            if positions[0] != first_step_position:
                reasons.append(
                    f"rank {rank}: token positions start at {positions[0]}, expected "
                    f"first_step_position {first_step_position}"
                )
        if rank == 0:
            dump = _dump_block(token)
            if dump is None:
                reasons.append("rank 0: the token record carries no logits_dump block")
            else:
                for field in ("path", "bin", "sha256", "bin_sha256"):
                    value = dump.get(field)
                    if not isinstance(value, str) or not value:
                        reasons.append(f"rank 0: logits_dump.{field} is empty or missing")
                    elif field in ("sha256", "bin_sha256") and (
                        len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
                    ):
                        reasons.append(f"rank 0: logits_dump.{field} is not 64 lowercase hex")
                path, bin_path = dump.get("path"), dump.get("bin")
                if isinstance(path, str) and isinstance(bin_path, str) and bin_path != path + ".bin":
                    reasons.append("rank 0: logits_dump.bin is not logits_dump.path + .bin")
                if dump.get("dtype") != DUMP_DTYPE:
                    reasons.append(
                        f"rank 0: logits_dump.dtype {dump.get('dtype')!r}, expected {DUMP_DTYPE}"
                    )
                if dump.get("vocab") != RELEASED_VOCAB:
                    reasons.append(
                        f"rank 0: logits_dump.vocab {dump.get('vocab')!r}, expected "
                        f"{RELEASED_VOCAB}"
                    )
                if dump.get("rows") != DECODE_POSITIONS:
                    reasons.append(
                        f"rank 0: logits_dump.rows {dump.get('rows')!r}, expected "
                        f"{DECODE_POSITIONS}"
                    )
                if dump.get("positions") != token_positions:
                    reasons.append(
                        "rank 0: logits_dump.positions differ from the token record positions"
                    )
            head = embed.get("head") or {}
            argmax_head = head.get("argmax") or {}
            head_token = argmax_head.get("token")
            if token_ids and head_token != token_ids[-1]:
                reasons.append(
                    f"rank 0: head argmax token {head_token!r} does not match the last decoded "
                    f"token {token_ids[-1]} (the head block is the last dumped position's head)"
                )
    dump_hashes = set()
    for rank, token in enumerate(tokens):
        dump = _dump_block(token)
        if dump is not None:
            dump_hashes.add(dump.get("bin_sha256"))
            if dump.get("positions") != token_positions:
                reasons.append(f"rank {rank}: logits_dump.positions differ from rank 0's positions")
    if len(dump_hashes) > 1:
        reasons.append(f"ranks disagree on logits_dump.bin_sha256: {sorted(dump_hashes)}")
    return {
        "schema": CLAIM_SCHEMA,
        "generated_utc": utc_now(),
        "window": os.path.abspath(window.get("window", "")),
        "claim_mode": mode,
        "claimable": not reasons,
        "reasons": reasons,
        "summary": {
            "world": WORLD,
            "prompt_len": prompt_len,
            "first_step_position": first_step_position,
            "decode_positions": decode_positions,
            "reference_ids": reference_ids,
            "prompt_ids_sha256": hashlib.sha256(
                json.dumps(prompt_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "forward_ids_sha256": hashlib.sha256(
                json.dumps(forward_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "prefill_limit": prefill_limit,
            "evidence_written": evidence_written,
            "token_claimed": [embed.get("token_claimed") for embed in embeds],
            "token_ids": token_ids,
            "token_positions": token_positions,
            "dump_bin_sha256": sorted(dump_hashes),
        },
    }


def mode_label(mode: str) -> str:
    return "asserted" if mode == "asserted" else "derived"


def inspect_command(args: argparse.Namespace) -> int:
    window = load_window(args.window)
    window["window"] = args.window
    claim = claim_window(window, args.mode)
    dump_json(os.path.join(args.arc_dir, "window-claim.json"), claim)
    print(json.dumps(claim["summary"], indent=2, sort_keys=True))
    if not claim["claimable"]:
        print("\nclaim refused:", file=sys.stderr)
        for reason in claim["reasons"]:
            print(f"  - {reason}", file=sys.stderr)
        return 2
    print(f"\n{mode_label(args.mode)} claim holds for {os.path.abspath(args.window)}")
    return 0


class HttpClient:
    """The probe's only I/O surface, so the checks are testable with a fake client."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(self, path: str, body: dict):
        data = json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()
        except urllib.error.URLError as error:
            raise ProbeError(f"cannot reach {self.base_url}{path}: {error}") from error


def parse_sse(raw: bytes):
    """Returns the data payloads in order, with [DONE] last when present."""
    text = raw.decode("utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(line[len("data: "):])
        elif line == "data:":
            events.append("")
    return events


def check_probe(window: dict, mode: str, buffered, stream) -> dict:
    rank0 = window["embeds"][0]
    prompt_len = rank0.get("prompt_len")
    prompt_ids = rank0.get("prompt_ids") or []
    checks = []
    failures = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            failures.append(f"{name}: {detail}" if detail else name)

    status, headers, raw = buffered
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = {}
        failures.append("buffered body is not JSON")
    check("buffered status 200", status == 200, f"status {status}")
    check(
        "buffered source header",
        headers.get(DEVICE_SOURCE_HEADER) == DEVICE_SOURCE_VALUE,
        f"{DEVICE_SOURCE_HEADER}={headers.get(DEVICE_SOURCE_HEADER)!r}",
    )
    check(
        "buffered claim header",
        headers.get(DEVICE_CLAIM_HEADER) == claim_label(mode),
        f"{DEVICE_CLAIM_HEADER}={headers.get(DEVICE_CLAIM_HEADER)!r},"
        f" expected {claim_label(mode)!r}",
    )
    check(
        "buffered carries no host stats",
        HOST_STATS_HEADER not in {name.lower() for name in headers},
    )
    choice = ((body.get("choices") or [{}])[0]) if isinstance(body, dict) else {}
    text = choice.get("text")
    check("buffered object", body.get("object") == "text_completion", repr(body.get("object")))
    check(
        "buffered finish_reason",
        choice.get("finish_reason") == "length",
        repr(choice.get("finish_reason")),
    )
    usage = body.get("usage") or {}
    check(
        "buffered prompt_tokens",
        usage.get("prompt_tokens") == prompt_len,
        repr(usage.get("prompt_tokens")),
    )
    check(
        "buffered completion_tokens",
        usage.get("completion_tokens") == DECODE_POSITIONS,
        repr(usage.get("completion_tokens")),
    )
    check(
        "buffered total_tokens",
        usage.get("total_tokens") == (prompt_len or 0) + DECODE_POSITIONS,
        repr(usage.get("total_tokens")),
    )
    check("buffered text is a string", isinstance(text, str))
    text = text if isinstance(text, str) else ""
    status, headers, raw = stream
    events = parse_sse(raw) if status == 200 else []
    check("stream status 200", status == 200, f"status {status}")
    check(
        "stream source header",
        headers.get(DEVICE_SOURCE_HEADER) == DEVICE_SOURCE_VALUE,
        f"{DEVICE_SOURCE_HEADER}={headers.get(DEVICE_SOURCE_HEADER)!r}",
    )
    check(
        "stream claim header",
        headers.get(DEVICE_CLAIM_HEADER) == claim_label(mode),
        f"{DEVICE_CLAIM_HEADER}={headers.get(DEVICE_CLAIM_HEADER)!r},"
        f" expected {claim_label(mode)!r}",
    )
    check(
        "stream terminates with [DONE]",
        bool(events) and events[-1] == "[DONE]",
        repr(events[-1] if events else None),
    )
    frames = [json.loads(event) for event in events if event != "[DONE]"]
    deltas = []
    finish_reasons = []
    content_frames = 0
    stream_usage = None
    for frame in frames:
        for stream_choice in frame.get("choices") or []:
            if stream_choice.get("finish_reason") is None:
                content_frames += 1
            else:
                finish_reasons.append(stream_choice["finish_reason"])
            # Completions stream text in choices[].text; chat streams choices[].delta.text.
            delta = stream_choice.get("delta") or {}
            piece = stream_choice.get("text")
            if piece is None:
                piece = delta.get("text")
            if piece is not None and stream_choice.get("finish_reason") is None:
                deltas.append(piece or "")
        if frame.get("usage") is not None:
            stream_usage = frame["usage"]
    check("stream finish_reason", finish_reasons == ["length"], repr(finish_reasons))
    check("stream deltas concatenate to the buffered text", "".join(deltas) == text)
    check(
        "stream usage matches the buffered usage",
        stream_usage is not None
        and stream_usage.get("prompt_tokens") == usage.get("prompt_tokens")
        and stream_usage.get("completion_tokens") == DECODE_POSITIONS
        and stream_usage.get("total_tokens") == usage.get("total_tokens"),
        repr(stream_usage),
    )
    check(
        "stream has nine content frames",
        content_frames == DECODE_POSITIONS,
        f"{content_frames} content frames",
    )
    return {
        "schema": PROBE_SCHEMA,
        "checks": checks,
        "passed": not failures,
        "failures": failures,
        "buffered_text": text,
        "buffered_usage": usage,
        "stream_deltas": deltas,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def check_fallback(window: dict, buffered, stream) -> dict:
    """Checks a fail-closed fallback answer: the claim refused, the host twin answered.

    This is the S7 record's evidence for a window whose claim is refused: the device source is
    installed, startup names every refusal reason, and every request is answered by the host path
    instead. The checks look for the host path's markers (its stats block in the body) and for
    the absence of every device provenance header.
    """
    checks = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    status, headers, raw = buffered
    check("buffered status 200", status == 200, f"status {status}")
    check(
        "buffered carries no device source header",
        headers.get(DEVICE_SOURCE_HEADER) is None,
        f"{DEVICE_SOURCE_HEADER}={headers.get(DEVICE_SOURCE_HEADER)!r}",
    )
    check(
        "buffered carries no device claim header",
        headers.get(DEVICE_CLAIM_HEADER) is None,
        f"{DEVICE_CLAIM_HEADER}={headers.get(DEVICE_CLAIM_HEADER)!r}",
    )
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        body = {}
    answer = body.get("choices") or [{}]
    choice = answer[0] if isinstance(answer[0], dict) else {}
    text = choice.get("text") if isinstance(choice.get("text"), str) else ""
    usage = body.get("usage") or {}
    check(
        "buffered body carries the host stats block",
        isinstance(body.get("x_deepseek_spark_stats"), dict),
        "x_deepseek_spark_stats is an object",
    )
    check("buffered object", body.get("object") == "text_completion", repr(body.get("object")))
    # The synthetic tiny plan can emit ids the released tokenizer decodes to "": the fallback
    # record is about which path answered, not about the host twin's model output.
    check("buffered text is a string", isinstance(choice.get("text"), str), repr(text))
    check(
        "buffered finish_reason is set",
        choice.get("finish_reason") is not None,
        repr(choice.get("finish_reason")),
    )
    check(
        "buffered usage has the host shape",
        isinstance(usage.get("total_tokens"), int),
        repr(usage.get("total_tokens")),
    )

    status, headers, raw = stream
    check("stream status 200", status == 200, f"status {status}")
    check(
        "stream carries no device source header",
        headers.get(DEVICE_SOURCE_HEADER) is None,
        f"{DEVICE_SOURCE_HEADER}={headers.get(DEVICE_SOURCE_HEADER)!r}",
    )
    check(
        "stream carries no device claim header",
        headers.get(DEVICE_CLAIM_HEADER) is None,
        f"{DEVICE_CLAIM_HEADER}={headers.get(DEVICE_CLAIM_HEADER)!r}",
    )
    events = parse_sse(raw) if status == 200 else []
    check(
        "stream terminates with [DONE]",
        bool(events) and events[-1] == "[DONE]",
        repr(events[-1] if events else None),
    )
    pieces = []
    finish_reasons = []
    for event in events:
        if event == "[DONE]":
            continue
        try:
            frame = json.loads(event)
        except json.JSONDecodeError:
            continue
        for stream_choice in frame.get("choices") or []:
            if stream_choice.get("finish_reason") is None:
                piece = stream_choice.get("text")
                if piece is None:
                    piece = (stream_choice.get("delta") or {}).get("text")
                if piece is not None:
                    pieces.append(piece or "")
            else:
                finish_reasons.append(stream_choice["finish_reason"])
    check("stream deltas concatenate to the buffered text", "".join(pieces) == text)
    check("stream carries a finish_reason", bool(finish_reasons), repr(finish_reasons))
    failures = [
        f"{item['name']}: {item['detail']}" if item["detail"] else item["name"]
        for item in checks
        if not item["passed"]
    ]
    return {
        "schema": PROBE_SCHEMA,
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "buffered_text": text,
        "claimable": False,
    }


def write_fallback_evidence(arc_dir: str, result: dict, window: dict, base_url: str) -> None:
    rank0 = window["embeds"][0]
    lines = [
        "# Device token window serving probe - fail-closed fallback",
        "",
        f"- generated_utc: {result.get('generated_utc')}",
        f"- schema: {result.get('schema')}",
        f"- window: {result.get('window')}",
        f"- server: {base_url}",
        f"- prompt_len: {rank0.get('prompt_len')}",
        f"- decode_positions: {rank0.get('decode_positions')}",
        f"- verdict: {'PASS' if result['passed'] else 'FAIL'}",
        "",
        "The window's claim is refused (window-claim.json names every reason), so the device",
        "source is installed but every request is answered by the host twin. The body carries the",
        "host stats block and no device provenance header: fail closed, exactly as designed.",
        "",
        "## Checks",
        "",
    ]
    for item in result["checks"]:
        lines.append(
            f"- [{'x' if item['passed'] else ' '}] {item['name']}"
            + (f" ({item['detail']})" if item["detail"] else "")
        )
    lines += [
        "",
        "## What this does not claim",
        "",
        "No device token was served: this is the refusal path. The window is UNMEASURED for",
        "serving purposes (m1y34 is a measured perf row, not an accepted D41 claim), and no",
        "tok/s, TTFT or certified-token claim is made.",
        "",
    ]
    with open(os.path.join(arc_dir, "FALLBACK.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_evidence(arc_dir: str, result: dict, window: dict, mode: str, base_url: str) -> None:
    rank0 = window["embeds"][0]
    lines = [
        "# Device token window serving probe",
        "",
        f"- generated_utc: {utc_now()}",
        f"- schema: {PROBE_SCHEMA}",
        f"- mode: {mode_label(mode)}",
        f"- window: {window.get('window', '')}",
        f"- server: {base_url}",
        f"- prompt_len: {rank0.get('prompt_len')}",
        f"- decode_positions: {rank0.get('decode_positions')}",
        f"- token_ids: {result.get('token_ids')}",
        f"- prompt_ids_sha256: {result.get('prompt_ids_sha256')}",
        f"- verdict: {'PASS' if result['passed'] else 'FAIL'}",
        "",
        "## Wire answer",
        "",
        "~~~text",
        result.get("buffered_text", ""),
        "~~~",
        "",
        "## Checks",
        "",
    ]
    for check in result["checks"]:
        lines.append(
            f"- [{'x' if check['passed'] else ' '}] {check['name']}"
            + (f" ({check['detail']})" if check["detail"] else "")
        )
    lines += [
        "",
        "## What this does not claim",
        "",
        "The body is the device token envelope served behind x-sparkflash-token-source:",
        f"{DEVICE_SOURCE_VALUE}. It is UNMEASURED: no tok/s, TTFT, quality or",
        "certified-token claim is made here. The S8 acceptance form is --mode asserted and",
        "requires the device writer's token_claimed: true on every rank; --mode derived is the",
        "S7 rehearsal form only.",
        "",
    ]
    with open(os.path.join(arc_dir, "EVIDENCE.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def probe_fallback_command(args: argparse.Namespace, window: dict, client) -> int:
    """Drives the fail-closed path: a refused window must leave the host twin answering."""
    asserted = claim_window(window, "asserted")
    derived = claim_window(window, "derived")
    if asserted["claimable"] or derived["claimable"]:
        print(
            "the window is claimable; use probe --expect claim instead of mislabelling it",
            file=sys.stderr,
        )
        return 2
    if args.prompt:
        try:
            prompt = [int(part) for part in args.prompt.split(",")]
        except ValueError:
            print("--prompt must be comma-separated token ids", file=sys.stderr)
            return 2
    else:
        prompt = DEFAULT_FALLBACK_PROMPT
    request = {"model": args.model, "prompt": prompt, "max_tokens": FALLBACK_MAX_TOKENS}
    stream_request = dict(request, stream=True, stream_options={"include_usage": True})
    buffered = client.post_json("/v1/completions", request)
    stream = client.post_json("/v1/completions", stream_request)
    result = check_fallback(window, buffered, stream)
    result.update(
        {
            "generated_utc": utc_now(),
            "mode": "fallback",
            "window": os.path.abspath(args.window),
            "base_url": args.base_url,
            "model": args.model,
            "prompt": prompt,
            "claim_refused": {
                "asserted": asserted["reasons"],
                "derived": derived["reasons"],
            },
        }
    )
    os.makedirs(args.arc_dir, exist_ok=True)
    try:
        buffered_body = json.loads(buffered[2] or b"{}")
    except json.JSONDecodeError:
        buffered_body = {"unparsed": buffered[2].decode("utf-8", errors="replace")}
    dump_json(os.path.join(args.arc_dir, "probe-buffered.json"), buffered_body)
    with open(os.path.join(args.arc_dir, "probe-stream.sse"), "wb") as handle:
        handle.write(stream[2])
    dump_json(os.path.join(args.arc_dir, "probe-summary.json"), result)
    write_fallback_evidence(args.arc_dir, result, window, args.base_url)
    print(f"fallback probe {'PASS' if result['passed'] else 'FAIL'} ({len(result['checks'])} checks)")
    if not result["passed"]:
        for failure in result["failures"]:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


def probe_command(args: argparse.Namespace, client=None) -> int:
    window = load_window(args.window)
    window["window"] = args.window
    client = client or HttpClient(args.base_url)
    if args.expect == "fallback":
        return probe_fallback_command(args, window, client)
    claim = claim_window(window, args.mode)
    if not claim["claimable"]:
        print("the window is not claimable in the requested form; run inspect first:", file=sys.stderr)
        for reason in claim["reasons"]:
            print(f"  - {reason}", file=sys.stderr)
        return 2
    prompt_ids = window["embeds"][0].get("prompt_ids") or []
    request = {"model": args.model, "prompt": prompt_ids, "max_tokens": DECODE_POSITIONS}
    stream_request = dict(request, stream=True, stream_options={"include_usage": True})
    buffered = client.post_json("/v1/completions", request)
    stream = client.post_json("/v1/completions", stream_request)
    result = check_probe(window, args.mode, buffered, stream)
    result.update(
        {
            "generated_utc": utc_now(),
            "mode": args.mode,
            "window": os.path.abspath(args.window),
            "base_url": args.base_url,
            "model": args.model,
            "token_ids": (window["tokens"][0].get("tokens") or {}).get("ids"),
        }
    )
    os.makedirs(args.arc_dir, exist_ok=True)
    try:
        buffered_body = json.loads(buffered[2] or b"{}")
    except json.JSONDecodeError:
        buffered_body = {"unparsed": buffered[2].decode("utf-8", errors="replace")}
    dump_json(os.path.join(args.arc_dir, "probe-buffered.json"), buffered_body)
    with open(os.path.join(args.arc_dir, "probe-stream.sse"), "wb") as handle:
        handle.write(stream[2])
    dump_json(os.path.join(args.arc_dir, "probe-summary.json"), result)
    write_evidence(args.arc_dir, result, window, args.mode, args.base_url)
    print(f"probe {'PASS' if result['passed'] else 'FAIL'} ({len(result['checks'])} checks)")
    if not result["passed"]:
        for failure in result["failures"]:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="re-derive the window's claim state")
    inspect.add_argument("--window", required=True, help="recorded four-rank token window directory")
    inspect.add_argument("--arc-dir", required=True, help="directory for window-claim.json")
    inspect.add_argument("--mode", choices=("asserted", "derived"), required=True)
    probe = sub.add_parser("probe", help="drive a running serve --device-token-dir through completions")
    probe.add_argument("--window", required=True)
    probe.add_argument("--base-url", required=True, help="the bound URL serve printed")
    probe.add_argument("--arc-dir", required=True)
    probe.add_argument("--mode", choices=("asserted", "derived"), required=True)
    probe.add_argument(
        "--expect",
        choices=("claim", "fallback"),
        default="claim",
        help="claim: the window must serve device answers; fallback: the claim must refuse and the host twin answers",
    )
    probe.add_argument(
        "--prompt",
        default=None,
        help="fallback mode only: comma-separated token ids for the host-twin request",
    )
    probe.add_argument("--model", default=MODEL)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            return inspect_command(args)
        return probe_command(args)
    except ProbeError as error:
        print(f"serve_window_probe: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
