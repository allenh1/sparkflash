#!/usr/bin/env python3
"""O2: fetch id-keyed next-token distributions from the vLLM-released server.

DECISIONS D39 makes vLLM the behavioural oracle and retires token equality:
acceptance is numeric/distributional over the logprobs each side assigns to the
same prefix. This script is the oracle half of that comparison. For every prompt
in the pinned suite it asks the server for prompt_logprobs over the prompt plus
a teacher-forced continuation, and records, for every compared position, the
full id-keyed distribution the server produced (up to the recipe's
--max-logprobs cap), the served model root from /v1/models, and the sha256 of
the response body.

Position convention
-------------------
prompt_logprobs[i] is the distribution for prompt token i, conditioned on
tokens 0..i-1. That is the same object a device logits row at position p=i-1
produces: the distribution over the token that follows position p. The script
therefore stores positions keyed by *device position* p (0-based), so the
comparison harness can join the two files on one integer.

What to send
------------
A request is  prompt = suite ids + continuation + sentinel  (sentinel defaults
to token 0, ONE token, because a distribution at position p needs prompt entry
p+1; the sentinel makes the final prompt-position prediction available).
The continuation is the fixture's reference next tokens (one per decode step),
so the oracle is teacher-forced on the same prefix the device compared; with it
the server returns the distributions for the decode positions too. Without a
continuation only positions 0..n-2 are covered, which is enough for a prefill
comparison.

Evidence contract (serving-parity-plan.md section 4)
----------------------------------------------------
* served root from /v1/models, checked against the released snapshot root
  (D38: 1bae0e1088993b9b82de08f59de5be5a34e42170). A mismatch is an error, not
  a warning: an oracle run against another snapshot proves nothing about this
  checkpoint.
* every request's response body sha256, prompt sha256 and wall time recorded;
* the suite file sha256 and the tokenizer sha256 recorded with the run;
* the prompt_logprobs integer recorded, and the recipe cap read from the
  server's --max-logprobs (the canonical released recipe sets 1000).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_EXPECT_ROOT = "1bae0e1088993b9b82de08f59de5be5a34e42170"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_PROMPT_LOGPROBS = 1000
DEFAULT_SENTINEL = 0
DEFAULT_TIMEOUT = 900

#: The server clamps logprobs it cannot represent to this floor; keep the
#: sentinel recognisable in evidence but do not treat it as a real probability.
LOGPROB_FLOOR = -9999.0


class OracleError(RuntimeError):
    """An oracle run that cannot produce trustworthy evidence."""


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
        raise OracleError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise OracleError(f"{path} is not JSON: {error}") from error


def load_suite(path: str) -> dict:
    suite = load_json(path)
    if suite.get("schema") != "sparkflash-oracle-suite/1":
        raise OracleError(f"{path}: unknown suite schema {suite.get('schema')!r}")
    prompts = suite.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise OracleError(f"{path}: suite has no prompts")
    seen: set[str] = set()
    for prompt in prompts:
        prompt_id = prompt.get("id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise OracleError(f"{path}: a prompt has no id")
        if prompt_id in seen:
            raise OracleError(f"{path}: duplicate prompt id {prompt_id!r}")
        seen.add(prompt_id)
        tokens = prompt.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            raise OracleError(f"{path}: prompt {prompt_id!r} has no tokens")
        if any(not isinstance(token, int) or token < 0 for token in tokens):
            raise OracleError(f"{path}: prompt {prompt_id!r} has a non-integer token id")
        if prompt.get("n_tokens") not in (None, len(tokens)):
            raise OracleError(f"{path}: prompt {prompt_id!r} n_tokens disagrees with tokens")
    return suite


def load_continuation(path: str) -> dict[str, list[int]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise OracleError(f"{path}: continuation must map prompt id to token ids")
    result: dict[str, list[int]] = {}
    for prompt_id, tokens in data.items():
        if not isinstance(tokens, list) or any(not isinstance(t, int) or t < 0 for t in tokens):
            raise OracleError(f"{path}: continuation for {prompt_id!r} is not a token list")
        result[prompt_id] = list(tokens)
    return result


def continuation_from_fixture(fixture_path: str, prompt_id: str) -> dict[str, list[int]]:
    fixture = load_json(fixture_path)
    steps = fixture.get("steps")
    if not isinstance(steps, list) or not steps:
        raise OracleError(f"{fixture_path}: fixture has no steps")
    tokens = []
    for step in steps:
        token = step.get("next_token")
        if not isinstance(token, int) or token < 0:
            raise OracleError(f"{fixture_path}: step {step.get('position')!r} has no next_token")
        tokens.append(token)
    return {prompt_id: tokens}


def build_prompt(tokens: list[int], continuation: list[int] | None = None, sentinel: int = DEFAULT_SENTINEL) -> list[int]:
    """Suite ids + teacher-forced continuation + one sentinel token."""
    prompt = [int(token) for token in tokens]
    if continuation:
        prompt.extend(int(token) for token in continuation)
    prompt.append(int(sentinel))
    return prompt


def build_request_body(model: str, prompt: list[int], prompt_logprobs: int) -> dict:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "prompt_logprobs": int(prompt_logprobs),
    }


def parse_served(models: dict) -> tuple[str | None, str | None]:
    data = models.get("data")
    if not isinstance(data, list) or not data:
        raise OracleError("/v1/models returned no served model")
    entry = data[0]
    return entry.get("id"), entry.get("root")


def root_matches(root: str | None, expect: str | None) -> bool:
    if not expect:
        return True
    if not root:
        return False
    normalised = root.rstrip("/")
    return normalised == expect or normalised.endswith("/" + expect) or normalised.endswith(expect)


def parse_prompt_logprobs(plp, prompt_tokens: list[int]) -> tuple[list[dict], bool]:
    """Turn the vLLM prompt_logprobs list into one record per device position.

    Returns (positions, first_entry_is_null). Position p = index - 1, because
    entry i scores prompt token i (conditioned on 0..i-1), which is exactly the
    distribution a logits row at position i-1 produces.
    """
    if not isinstance(plp, list):
        raise OracleError(f"prompt_logprobs is not a list but {type(plp).__name__}")
    if len(plp) != len(prompt_tokens):
        raise OracleError(f"prompt_logprobs has {len(plp)} entries for {len(prompt_tokens)} prompt tokens")
    first_is_null = plp[0] is None
    positions: list[dict] = []
    for index in range(1, len(plp)):
        entry = plp[index]
        if not isinstance(entry, dict) or not entry:
            raise OracleError(f"prompt_logprobs[{index}] is {entry!r}, not an id-keyed dict")
        distribution: dict[int, float] = {}
        for key, value in entry.items():
            try:
                token_id = int(key)
            except (TypeError, ValueError) as error:
                raise OracleError(f"prompt_logprobs[{index}] key {key!r} is not a token id") from error
            try:
                if isinstance(value, dict):
                    logprob = float(value["logprob"])
                else:
                    logprob = float(value)
            except (KeyError, TypeError, ValueError) as error:
                raise OracleError(
                    f"prompt_logprobs[{index}][{key!r}] has no usable logprob: {error}"
                ) from error
            if logprob != logprob or logprob in (float("inf"), float("-inf")):
                raise OracleError(f"prompt_logprobs[{index}][{token_id}] is not finite: {logprob!r}")
            distribution[token_id] = logprob
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        positions.append(
            {
                "position": index - 1,
                "target_token": int(prompt_tokens[index]),
                "logprob_count": len(distribution),
                "target_in_distribution": int(prompt_tokens[index]) in distribution,
                "top": [[token, logprob] for token, logprob in ranked],
                "logprobs": {str(token): logprob for token, logprob in sorted(distribution.items())},
            }
        )
    return positions, first_is_null


def parse_response(body: bytes) -> dict:
    try:
        response = json.loads(body)
    except json.JSONDecodeError as error:
        raise OracleError(f"response is not JSON: {error}") from error
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OracleError("response has no choices")
    choice = choices[0]
    if "prompt_logprobs" not in choice:
        raise OracleError("response has no prompt_logprobs; is --max-logprobs set on the server?")
    return response


def request_json(url: str, payload: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> tuple[bytes, dict]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read()[:400].decode("utf-8", "replace")
        raise OracleError(f"HTTP {error.code} from {url}: {detail}") from error
    except urllib.error.URLError as error:
        raise OracleError(f"cannot reach {url}: {error.reason}") from error
    return body, parse_response(body) if payload is not None else json.loads(body)


def wait_for_server(base_url: str, expect_root: str | None, deadline: float, log) -> dict:
    """Wait until /v1/models answers, then verify the served root (D38)."""
    last_error: str | None = None
    while time.time() < deadline:
        try:
            body, models = request_json(base_url.rstrip("/") + "/v1/models", timeout=20)
        except OracleError as error:
            last_error = str(error)
            time.sleep(15)
            continue
        served_id, served_root = parse_served(models)
        if not root_matches(served_root, expect_root):
            raise OracleError(
                f"served root {served_root!r} is not the released snapshot {expect_root!r}; "
                "refusing to record oracle evidence against another checkpoint (D38)"
            )
        log(f"server up: id={served_id!r} root={served_root!r} models_sha256={sha256_bytes(body)}")
        return {"id": served_id, "root": served_root, "models_response_sha256": sha256_bytes(body)}
    raise OracleError(f"server did not answer before the deadline; last error: {last_error}")


def run(args: argparse.Namespace, log) -> dict:
    suite = load_suite(args.suite)
    suite_sha = sha256_file(args.suite)
    prompts = {prompt["id"]: prompt for prompt in suite["prompts"]}
    selected = list(prompts)
    if args.only:
        selected = [prompt_id.strip() for prompt_id in args.only.split(",") if prompt_id.strip()]
    missing = [prompt_id for prompt_id in selected if prompt_id not in prompts]
    if missing:
        raise OracleError(f"--only names unknown prompt ids: {missing}")

    continuation: dict[str, list[int]] = {}
    continuation_source = "none (sentinel only)"
    if args.continuation:
        continuation = load_continuation(args.continuation)
        continuation_source = args.continuation
    elif args.fixture:
        prompt_id = args.fixture_prompt_id or selected[0]
        continuation = continuation_from_fixture(args.fixture, prompt_id)
        continuation_source = f"{args.fixture} (steps[].next_token for {prompt_id})"

    plans = []
    for prompt_id in selected:
        tokens = list(prompts[prompt_id]["tokens"])
        extra = continuation.get(prompt_id, [])
        prompt = build_prompt(tokens, extra, args.sentinel_token)
        plans.append(
            {
                "id": prompt_id,
                "domain": prompts[prompt_id]["domain"],
                "suite_tokens": len(tokens),
                "continuation_tokens": len(extra),
                "prompt": prompt,
                "request": build_request_body(args.model, prompt, args.prompt_logprobs),
            }
        )

    plan_table = [
        {"id": plan["id"], "suite_tokens": plan["suite_tokens"],
         "continuation_tokens": plan["continuation_tokens"], "prompt_len": len(plan["prompt"]),
         "device_positions": [0, len(plan["prompt"]) - 2]}
        for plan in plans
    ]
    if args.dry_run:
        return {
            "status": "dry-run",
            "suite": {"path": args.suite, "sha256": suite_sha},
            "continuation_source": continuation_source,
            "sentinel_token": args.sentinel_token,
            "prompt_logprobs": args.prompt_logprobs,
            "plans": plan_table,
        }

    deadline = time.time() + args.wait_seconds
    base = args.base_url.rstrip("/")
    served = wait_for_server(base, args.expect_root, deadline, log)
    model = served["id"] or args.model

    records = []
    failures = 0
    for plan in plans:
        payload = dict(plan["request"])
        payload["model"] = model
        started = time.time()
        try:
            body, response = request_json(base + "/v1/completions", payload, timeout=args.request_timeout)
        except OracleError as error:
            failures += 1
            log(f"prompt {plan['id']}: FAILED: {error}")
            records.append({"id": plan["id"], "domain": plan["domain"], "status": "failed", "error": str(error)})
            continue
        wall = round(time.time() - started, 3)
        choice = response["choices"][0]
        positions, first_is_null = parse_prompt_logprobs(choice["prompt_logprobs"], plan["prompt"])
        response_sha = sha256_bytes(body)
        record = {
            "id": plan["id"],
            "domain": plan["domain"],
            "status": "ok",
            "suite_tokens": plan["suite_tokens"],
            "continuation_tokens": plan["continuation_tokens"],
            "prompt_len": len(plan["prompt"]),
            "prompt_sha256": sha256_bytes(json.dumps(plan["prompt"]).encode()),
            "request_sha256": sha256_bytes(json.dumps(payload).encode()),
            "response_sha256": response_sha,
            "wall_s": wall,
            "first_entry_is_null": first_is_null,
            "positions": positions,
        }
        records.append(record)
        last = positions[-1]
        log(
            f"prompt {plan['id']}: {len(positions)} positions p={positions[0]['position']}..{positions[-1]['position']} "
            f"last_top1={last['top'][0][0]} last_lp={last['top'][0][1]:.6f} wall={wall}s sha256={response_sha[:16]}"
        )

    out = {
        "schema": "sparkflash-native-oracle/1",
        "source": "vllm-released",
        "status": "ok" if failures == 0 else "partial",
        "generated_at": utc_now(),
        "tag": args.tag,
        "base_url": base,
        "served": served,
        "expect_root": args.expect_root,
        "root_verified": root_matches(served["root"], args.expect_root),
        "recipe": {
            "prompt_logprobs": args.prompt_logprobs,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "sentinel_token": args.sentinel_token,
            "max_logprobs_cap_note": "the released recipe passes --max-logprobs 1000; a request above the cap is rejected by the server",
        },
        "suite": {"path": args.suite, "sha256": suite_sha, "prompt_count": len(suite["prompts"])},
        "continuation": {"source": continuation_source},
        "prompts": records,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--suite", required=True, help="pinned suite JSON (scripts/oracle_prompt_suite.json)")
    parser.add_argument("--out", required=True, help="evidence JSON to write (with a .log next to it)")
    parser.add_argument("--log", help="log path (default: --out with .log)")
    parser.add_argument("--expect-root", default=DEFAULT_EXPECT_ROOT,
                        help=f"released snapshot root (default {DEFAULT_EXPECT_ROOT}); empty disables the check")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-logprobs", type=int, default=DEFAULT_PROMPT_LOGPROBS)
    parser.add_argument("--sentinel-token", type=int, default=DEFAULT_SENTINEL)
    parser.add_argument("--continuation", help="JSON mapping prompt id -> continuation token ids")
    parser.add_argument("--fixture", help="sequence fixture JSON; continuation is its steps[].next_token")
    parser.add_argument("--fixture-prompt-id", help="suite prompt id the fixture belongs to (default: first)")
    parser.add_argument("--only", help="comma-separated prompt ids to query")
    parser.add_argument("--wait-seconds", type=int, default=2700)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--tag", default="")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the request plan; no network")
    args = parser.parse_args(argv)
    if args.expect_root == "":
        args.expect_root = None
    log_path = args.log or (args.out + ".log")
    lines: list[str] = []

    def log(message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        lines.append(line)

    log(f"oracle start suite={args.suite} out={args.out} expect_root={args.expect_root}")
    try:
        result = run(args, log)
    except OracleError as error:
        log(f"FAILED: {error}")
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"oracle_native: {error}", file=sys.stderr)
        return 2
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
        handle.write("\n")
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    if result["status"] == "dry-run":
        print(json.dumps(result["plans"], indent=1))
        return 0
    print(f"wrote {args.out} ({len(result['prompts'])} prompts, status={result['status']})")
    return 0 if result["status"] == "ok" else 4


if __name__ == "__main__":
    sys.exit(main())
