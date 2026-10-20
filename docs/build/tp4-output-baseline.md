# TP=4 output baseline: single-node reference serve, text and latency

Status: measured on the reference serve (no vLLM, no GPU window, no checkpoint changes).
This is an output/latency baseline for the endpoint, not an exactness comparison. The texts
below are wrong or truncated for both prompts and are recorded verbatim; the TP=4 number
(task-55) is the one that matters for TTFT and tokens/s.

## What was measured

- Node: node0, reference serve listening on 127.0.0.1:18000 (single process, CPU/f32,
  disk-bound; one sequence at a time).
- Model: deepseek-ai/DeepSeek-V4.1-Flash, reference engine. `/v1/models` reports
  `fabric_mode: "none (reference serve: one process, no RoCE)"`.
- Date: 2026-10-20T02:43-02:47Z (first request started ~02:43:21Z, second ~02:45:04Z).
- Request budget for this run: exactly two `/v1/completions` requests, sent strictly one at
  a time, `max_tokens=2`, `temperature=0`. The only other traffic was one `GET /v1/models`
  before the run. serve-dev's single request (which preceded this run) had completed and its
  slot was clear before the first request here; nothing overlapped.
- Instrument: `scripts/serve_output_probe.py` at commit 7f021d1 (sha256
  a2e958a38fed8a31c64309d81c591eafbb44c8c8e4e5ffaf2b4bb9db09dfcc54), copied to the node
  evidence directory and run there. `--io-pid 440530` sampled `/proc/440530/io` around each
  request.
- Prompts (the same two are the TP=4 comparison prompts):
  1. `capital-of-france`: "The capital of France is"
  2. `rust-borrow`: "In Rust, the borrow checker rejects this code because:"

Command (run on node0 in `/home/user/phase2-evidence/window-2026-10-18`):

~~~sh
python3 serve_output_probe.py \
  --base-url http://127.0.0.1:18000 \
  --max-tokens 2 --temperature 0 --timeout 1800 \
  --io-pid 440530 --tag b75-single-node \
  --out-json serve-output-b75-single-node.json \
  --out-md serve-output-b75-single-node.md
~~~

## Results

Both requests returned HTTP 200, `finish_reason: "length"`, and usage
`prompt_tokens/completion_tokens` 5/2 and 11/2. Client `tokens/s` is completion tokens over
end-to-end wall time (it includes prefill); server TTFT and decode rate are from
`x_deepseek_spark_stats.timing`. rchar/read_bytes are deltas of `/proc/<pid>/io` around the
request.

| # | prompt | completion text (verbatim) | prompt tok | completion tok | wall s | server TTFT s | client tok/s | server decode tok/s | rchar GB | read_bytes GB | rchar/token GB | KV-cache hit | finish |
| --: | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- |
| 1 | capital-of-france | `" a country"` | 5 | 2 | 102.44 | 42.70 | 0.0195 | 0.01674 | 62.90 | 0.648 | 31.45 | 0.0 | length |
| 2 | rust-borrow | `" It is"` | 11 | 2 | 129.55 | 68.41 | 0.0154 | 0.01636 | 74.78 | 14.857 | 37.39 | 0.0 | length |

Totals: 16 prompt tokens, 4 completion tokens, 231.99 s of wall time, 0.0172 completion
tokens/s end to end. The server's own timing says prefill ran at 0.117 and 0.161 prompt
tokens/s (about 8.5 and 6.2 s per prompt token), which explains the 42.7 s and 68.4 s TTFT;
decode was about 1 token per 60 s server-measured (0.0167 and 0.0164 tok/s).

### Texts, verbatim

- capital-of-france: `" a country"` (with a leading space; not "Paris").
- rust-borrow: `" It is"` (with a leading space; not an explanation of the borrow checker).

An earlier bare-question request against the same instance this night produced
`"<|end_of_sentence|>.\n"` for "What is the capital of France?". All three answers are
consistent with a reference path that is not producing coherent general text, and that is
itself a valid baseline datapoint. Do not read these strings as a token-level result; they
are what the endpoint returned, recorded without correction.

### Cache state: cold first request, only partly warm later

The cache that matters here is the OS page cache, not the serve's KV cache. The serve
reported `kv_cache_hit_rate: 0.0` for both requests (total_prompt_tokens 5 and 11, cached
tokens 0: two distinct prompts, no prefix reuse), and both Engram layers reported
`lookups_total: 0` (the reference loader refuses/pages the Engram tables; they are never
materialised). So the KV/Engram numbers show no reuse; the page-cache state is visible in
the io deltas.

- The earlier 2m54s datapoint (7 prompt + 4 completion tokens, ~43.5 s per completion token)
  was the first completion request after the server started at 02:31:11Z (the only
  completion before serve-dev's request). It is the cold-page-cache number: no 475 GiB of
  weights are resident, so the first request pays mostly real NVMe reads.
- Our request 1 came after serve-dev's completed request and was mostly warm: 62.90 GB of
  logical reads with only 0.648 GB from disk (~99% page-cache hits).
- Our request 2 was much less warm: 74.78 GB logical with 14.86 GB from disk (~80% hits).
  A 475 GiB checkpoint does not fit in ~117 GiB of node RAM, and 63-75 GB of logical reads
  per request evicts the working set, so the hit share varies request to request.
- No cold-start run was made tonight (it would cost another multi-minute request); the
  2m54s figure is therefore the cold datapoint and the two table rows are warm/variable.

Do not compare the 2m54s figure (or the two table rows) against TP=4 TTFT or tokens/s as if
they were the same measurement. They measure a different path and a different cache state.

### Why this endpoint is not the TP=4 comparison point

The reference engine has no resident copy of the weights: for every step it decodes each
tensor through `sparkflash_engine::linalg::decode_by_name` from the checkpoint on disk
(`CHECKPOINT_TENSOR_LIMIT = 1 << 30` elements), one sequence at a time, in one process. The
checkpoint is 48 shards / 510,311,070,460 bytes (about 475.24 GiB of tensors) and only about
117 GiB of RAM is available; the 98 GB Engram tables are paged out/never materialised. Every
token therefore re-reads weights through the page cache and, on a miss, the NVMe device.

The io sampling shows this directly: `rchar` advanced 62.90 GB and 74.78 GB during the two
requests (614 MB/s and 577 MB/s of logical reads while decoding), i.e. ~31.45 GB and
~37.39 GB of logical reads per completion token, with 0.65 GB and 14.86 GB actually coming
back from disk. An independent `/proc/<pid>/io` sample by the Lead during an in-flight
request saw ~540 MB/s of logical reads with mostly page-cache hits, consistent with these
deltas. That is a page-cache/NVMe-bandwidth measurement, not a compute measurement.
On TP=4 the weights are resident across the four ranks and the serving path is the one under
test (task-55); only its TTFT and tokens/s belong in the TP=4 comparison.

## Reproducing on the TP=4 endpoint

The same script and the same two default prompts are used, with the TP=4 base URL and
streaming enabled so TTFT is measured client-side as well:

~~~sh
python3 scripts/serve_output_probe.py \
  --base-url http://<tp4-host>:<port> \
  --max-tokens 2 --temperature 0 --timeout 600 --stream \
  --tag tp4-single-request \
  --out-json serve-output-tp4.json --out-md serve-output-tp4.md
~~~

The table above is the single-node baseline to compare against. On TP=4, expect TTFT well
below 42.7/68.4 s and tokens/s well above 0.017; record the server's
`x_deepseek_spark_stats.timing` fields if the endpoint emits them.

## Evidence

Node and host copy (identical bytes):

- `/home/user/phase2-evidence/window-2026-10-18/serve-output-b75-single-node.json`
  (sha256 69a9af289b43db096b6147f42d63160c3e4182e0167ccad4fb3a8a2c3ead508f)
- `/home/user/phase2-evidence/window-2026-10-18/serve-output-b75-single-node.md`
  (sha256 906902baee798fa62fc26f06ef6f7270f785afc28dadcd5cbdc9486c9f3fafd6)
- probe script on the node: `.../window-2026-10-18/serve_output_probe.py`, sha256
  a2e958a38fed8a31c64309d81c591eafbb44c8c8e4e5ffaf2b4bb9db09dfcc54

Caveats: the io deltas include prefill and decode for the whole request; the completion
sample is only 2 tokens per request, so client tokens/s is an end-to-end rate dominated by
prefill. The server's decode rate and the client wall/token differ slightly (server 0.0164-
0.0167 tok/s vs client 0.0154-0.0195 tok/s); both are reported as measured. No retries and
no extra completion requests were made.
