# TP=4 output baseline: four-rank serve and the single-node reference

Status: two measured output/latency baselines on the DeepSeek-V4.1-Flash Spark endpoint.
The **four-rank TP=4 run (task-55)** is the comparison point for TTFT and tokens/s; the
**single-node reference serve (task-56)** is the baseline that explains why the reference
path is not a TP=4 comparison. The texts are recorded verbatim; they are not an exactness
comparison.

## Four-rank TP=4 chat completion (task-55)

Status: task-55 Phase-1 output run. This line records the first real four-rank
resident `sparkflash serve` endpoint and the text it produced, including the final
capital answer that names Paris. It is an output/correctness baseline, not a
throughput claim.

### 1. What ran

| item | value |
| --- | --- |
| branch / commit | `task/task55-serve` / `fd09d5c` (adapter on tp-dev `47f8ccb`) |
| aarch64 binary | `sparkflash_tp4serve_fd09d5c_aarch64`, sha256 `03494b7a15f664b4b35a81053f44c71b3d94f3d0832469141062797c61ab47d7` |
| nodes | `node0` 192.0.2.80, `node1` 192.0.2.88, `node2` 192.0.2.68, `node3` 192.0.2.59 |
| rank 0 | HTTP `0.0.0.0:8000`; tokenizer + chat template + sampling + streaming |
| ranks 1-3 | same binary, resident forward only; never detokenize |
| engine collective | `--tp4-base-port 29601`; rank R binds 29601+R |
| serve control | separate TCP channel on rank 0, port 29701 (`base_port + 100`) |
| checkpoint | released snapshot `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| fabric mode | `tcp tp4 host` (host TCP control + canonical host collective; no RoCE/NCCL claim) |
| Phase-1 shape | one sequence, one request in flight, one request may queue; no batching |

Command shape (rank 0; ranks 1-3 omit `--tokenizer` and `--bind`):

```sh
sparkflash serve --checkpoint $SNAP \
  --tokenizer $SNAP/tokenizer.json --bind 0.0.0.0:8000 \
  --tp4-rank 0 --tp4-world-size 4 \
  --tp4-peers 192.0.2.80,192.0.2.88,192.0.2.68,192.0.2.59 \
  --tp4-base-port 29601 --max-tokens 64
```

### 2. Load

All four ranks opened the resident checkpoint before the control handshake. Open took
229-261 s per rank; no HTTP request was accepted before all four sent Hello and rank 0
sent Ready.

`sparkflash serve: tp4_metrics rank=0 phase=open load_ms=229274 last_step_ms=0 steps=0 reduce_calls=0 reduce_elements=0 resident_cache_bytes=90944617800 resident_cache_tensors=96081`

Rank 1 load_ms=261113, rank 2 load_ms=252423, rank 3 load_ms=245477. Every rank held
90,944,617,800 resident cache bytes across 96,081 tensors.

### 3. First request: capital, max_tokens 8, temperature 0

Request started 2026-10-20T03:42:27Z and returned 2026-10-20T03:52:46Z.

Verbatim assistant text:

> We need answer. Need be concise.

| field | value |
| --- | --- |
| prompt tokens | 37 (canonical chat template, thinking on) |
| completion tokens | 8 |
| finish reason | `length` |
| wall (local curl) | 618.6378757953644 s |
| TTFT | 519.910889539 s |
| prefill tok/s (reported) | 0.07116604165683216 |
| decode tok/s (reported) | 0.07092241303511708 |
| rank steps | 44 = 37 prefill + 7 decode (max_tokens 8) |
| all-reduce calls per rank | 1760 = 44 steps x 40 layers |
| all-reduce elements per rank | 9,011,200 |
| last step ms per rank | 14065 / 14044 / 14037 / 14030 (r0..r3) |

The 37 prompt tokens are the thinking-mode chat template. Eight completion tokens
end during the thinking prefix, so this is a real token output but not yet the city
the user asked for; section 8 records the 24-token answer.

### 4. Per-rank metrics after request 1

| rank | node | load_ms | last_step_ms | steps | reduce_calls | reduce_elements |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | .80 | 229274 | 14065 | 44 | 1760 | 9011200 |
| 1 | .88 | 261113 | 14044 | 44 | 1760 | 9011200 |
| 2 | .68 | 252423 | 14037 | 44 | 1760 | 9011200 |
| 3 | .59 | 245477 | 14030 | 44 | 1760 | 9011200 |

### 5. Repeat: same request, temperature 0

Lead submitted the same 37/8 capital request at 2026-10-20T03:46:57Z while request 1
was in flight. It queued in the one-deep job channel and ran immediately after request 1.

- text: identical `We need answer. Need be concise.`
- usage: 37 prompt + 8 completion, finish `length`
- server TTFT: 518.090461903 s; decode tok/s: 0.07105943719225268
- HTTP wall: 964.876325 s, including the queue wait behind request 1
- counters after both: `requests_total=2`, `prompt_tokens_total=74`, per-rank reduce_calls=3520

This is a temperature-0 determinism datapoint across two runs on the four-rank endpoint.

#### Audit probes (task-58, two sequential requests, one in flight)

Audit-dev ran the two task-58 prompts against the same endpoint after the repeat:

| prompt | prompt/completion | text | wall | TTFT |
| --- | --- | --- | ---: | ---: |
| capital | 37 / 8 | `We need answer. Need be concise.` | 618.063734 s | 519.476231 s |
| rust-borrower | 41 / 8 | `We need answer. Need explain borrow checker` | 674.988837 s | 576.196739 s |

The io sampling succeeded over ssh: capital rchar 15,970,630 B / read_bytes 0; rust rchar 17,451,668 B / read_bytes 6,197,248 B. Those rows are audit-dev's task-58 evidence, not throughput claims.

### 6. Single-node baseline for contrast

The earlier single-process reference chat on the same prompt family produced 37 prompt + 16
completion tokens in 653.781803 s at 0.0309066813 tok/s (TTFT 168451.205457 ms;
`finish_reason=length`). That run is the baseline this endpoint is compared against, not a
throughput target: TP=4 at 37+8 took 618.64 s and reported 0.0709 decode tok/s on four ranks.

### 7. Limitations this run keeps explicit

- One sequence, one request in flight, one queued; no batching. A second concurrent request
  waits in the one-deep queue, and a third is refused rather than desynchronising the ranks.
- Attention is replicated: each rank owns its own `SequenceState` and per-layer KV copy, so
  four ranks hold four copies of the sequence KV (4x KV memory) while computing identical
  arithmetic. No shared KV pool or paged-serving claim is made here.
- Only greedy (`temperature=0`) is accepted by the extracted `Tp4Rank::set_sampling`; a
  non-greedy request is a typed refusal, not silently greedy output.
- Prompt logprobs are refused on this path; generated-token logprobs are not supported in
  this first output run either.
- Load is reported separately from request wall time (`load_ms` above).

### 8. Final capital answer: max_tokens 24, temperature 0

The final request started 2026-10-20T04:22:47Z and returned 2026-10-20T04:38:58Z.
It was queued behind audit-dev's rust-borrower probe for part of that interval.

Verbatim assistant text:

> We need answer. Need be concise. User asks capital of France. Answer Paris. Maybe mention. Need no extra.

The answer names Paris. It is still thinking-prefix prose rather than a one-word answer,
but the city asked for is present at 24 completion tokens.

| field | value |
| --- | --- |
| prompt tokens | 37 |
| completion tokens | 24 (total 61) |
| finish reason | `length` |
| wall (local curl) | 970.3598704338074 s, including queue wait |
| TTFT | 519.296727981 s |
| prefill tok/s (reported) | 0.07125020825067434 |
| decode tok/s (reported) | 0.07036087477104187 |
| rank steps | 60 = 37 prefill + 23 decode |
| rank-0 last step | 14300 ms |
| rank-0 reduce_calls | 9600 cumulative; +2400 for this request (60 x 40 layers) |
| rank-0 reduce_elements | 49,152,000 cumulative |

Per-rank final request metrics (steps are reset per request; reduce counters are cumulative):

| rank | node | steps | last_step_ms | reduce_calls | reduce_elements |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | .80 | 60 | 14300 | 9600 | 49152000 |
| 1 | .88 | 60 | 14113 | 9600 | 49152000 |
| 2 | .68 | 60 | 14121 | 9600 | 49152000 |
| 3 | .59 | 60 | 14124 | 9600 | 49152000 |

### 9. Teardown

All four ranks were stopped with SIGTERM after the final request. Ports 29601-29604
(engine collective), 29701 (serve control on rank 0), and 8000 (HTTP on rank 0) were
confirmed free on all four nodes before the server was handed back for the kernel-dev
profiled run.

### 10. Raw evidence paths

- `/tmp/tp4-chat.json` - first request curl output (JSON + HTTP trailer); sha256 `a897188dba48fc426e337b5f8d918801fea4de7f28a5db87cbb2ca939284f7ce`
- `/home/user/phase2-evidence/window-2026-10-18/tp4-serve-first-20261020T035246Z/chat-response.json` - cleaned first response
- `.../repeat-raw.json` - Lead's queued repeat; sha256 `033d5467be0ac87dd23df20c17b86d969dcb0c1f7b3680a39993b22f3ee59a0b`
- `.../repeat-response.json` - cleaned repeat; sha256 `e95b09daf14a3bb1a71daa8c0c922a8d9a9203f892827c6241cde78dc4084aab`
- `/tmp/tp4-chat-paris.json` - final raw response (JSON + HTTP trailer); sha256 `94974e30918244b12bdf668876cf73725b93463631a9cb9833e579de89612a3f`
- `.../paris-response.json` - cleaned final response; sha256 `2a4425f39a4a015c1fa01bdfc469565150df38c02c0ca709f76797be14922bf5`
- `.../r0.log` ... `.../r3.log` - the four rank logs, including `tp4_metrics` lines
- `.../SHA256SUMS` - hashes of the captured files

---

## Appendix: detailed single-node reference serve baseline (task-56)

## TP=4 output baseline: single-node reference serve, text and latency

Status: measured on the reference serve (no vLLM, no GPU window, no checkpoint changes).
This is an output/latency baseline for the endpoint, not an exactness comparison. The texts
below are wrong or truncated for both prompts and are recorded verbatim; the TP=4 number
(task-55) is the one that matters for TTFT and tokens/s.

### What was measured

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

### Results

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

#### Texts, verbatim

- capital-of-france: `" a country"` (with a leading space; not "Paris").
- rust-borrow: `" It is"` (with a leading space; not an explanation of the borrow checker).

An earlier bare-question request against the same instance this night produced
`"<|end_of_sentence|>.\n"` for "What is the capital of France?". All three answers are
consistent with a reference path that is not producing coherent general text, and that is
itself a valid baseline datapoint. Do not read these strings as a token-level result; they
are what the endpoint returned, recorded without correction.

#### Cache state: cold first request, only partly warm later

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

#### Why this endpoint is not the TP=4 comparison point

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

### Reproducing on the TP=4 endpoint

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

### Evidence

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
