# TP=4 online baseline: four-rank host serve, correct but compute-bound

Status: measured on the live TP=4 host endpoint (node0 rank 0, 0.0.0.0:8000) with
four ranks and the resident non-Engram weight set (task-55/task-57). No vLLM, no GPU
window, no checkpoint changes. This is the first online chat measurement of the four-rank
path; it is a correctness/characterisation baseline, not a performance win. The endpoint
was torn down after the final request: :8000 now refuses connections, nothing is listening on
that port, and no sparkflash_tp4serve process remains, so the port is free. These are
recorded numbers, not a live service.

## What ran

- Endpoint: `http://192.0.2.80:8000`, model `deepseek-ai/DeepSeek-V4.1-Flash`;
  `/v1/models` reports `fabric_mode: "tcp tp4 host"` and chat+completion endpoints.
- Rank 0 pid 2399562 (`sparkflash_tp4serve ... --tp4-rank 0 --tp4-world-size 4
  --tp4-peers 192.0.2.80,192.0.2.88,192.0.2.68,192.0.2.59`).
- Startup evidence (`tp4-run-20261020T033645Z/r0.log`):
  `tp4_metrics rank=0 phase=open load_ms=229274 last_step_ms=0 steps=0 reduce_calls=0
  reduce_elements=0 resident_cache_bytes=90944617800 resident_cache_tensors=96081`,
  then `TP=4 rank 0/4 on 192.0.2.80; control 29701; workers never detokenize` and
  `listening on http://0.0.0.0:8000`. That is 90.94 GB resident per rank (one rank per
  node, four nodes), i.e. the 84.70 GiB shard-resident set plus the ~9.8 GiB replicated
  set; rank load measured 229-261 s across the four ranks (r0-r3 logs: 229274, 261113,
  252423, 245477 ms), all with resident_cache_bytes=90944617800.
- Requests, in order, one in flight at a time (queue capacity 1):
  1. serve-dev's first capital chat: max_tokens 8, temperature 0.
  2. Lead's identical capital repeat: max_tokens 8, temperature 0; identical bytes
     (ids 00000000 and 00000001). Its 964.90 s wall includes ~5.5 min of queue wait, so
     it is a determinism point, not a clean latency point.
  3. this probe: max_tokens 8 chat, capital then rust-borrower, `--stream`,
     `--io-pid 2399562 --io-ssh allenh1@192.0.2.80`.
  4. serve-dev's final capital max_tokens 24: text names Paris; its wall includes queue
     wait behind the rust probe, so only its server-side TTFT/decode are clean. Endpoint
     was torn down after it returned; no further requests.
- Probe command (host 192.0.2.71, run in `/home/user/wt/tp4-online`):

~~~sh
python3 scripts/serve_output_probe.py \
  --base-url http://192.0.2.80:8000 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --preset tp4-chat --prompts /home/user/phase2-evidence/window-2026-10-18/task58-tp4-prompts.json \
  --stream --timeout 1800 \
  --io-pid 2399562 --io-ssh allenh1@192.0.2.80 \
  --tag tp4-online \
  --out-json serve-output-tp4-online.json \
  --out-md serve-output-tp4-online.md
~~~

  (The prompts file held capital and rust-borrower, each `max_tokens: 8`; the full
  `tp4-chat` preset keeps 32/64/48 for a longer run.)

## Four-rank results

All five rows are chat completions, temperature 0, finish_reason `length`. "completion
tok/s" is completion tokens / end-to-end wall; "total tok/s" is (prompt+completion) tokens
/ wall; server decode/prefill are from `x_deepseek_spark_stats.timing`. rchar/read_bytes
are `/proc/2399562/io` deltas over ssh around the request.

| source | prompt | completion text (verbatim) | ptok | ctok | wall s | client TTFT s | completion tok/s | total tok/s | server decode tok/s | server prefill tok/s | rchar B | read_bytes B | KV hit |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| serve-dev first | capital | `"We need answer. Need be concise."` | 37 | 8 | 618.64 | not captured | 0.0129 | 0.0727 | 0.0710 | 0.0713 | not sampled | not sampled | n/a |
| Lead repeat | capital | identical bytes | 37 | 8 | 964.90 (incl ~330 s queue) | not captured | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| probe | capital | `"We need answer. Need be concise."` | 37 | 8 | 618.063734 | 519.476231 | 0.012944 | 0.0727 | 0.071003 | 0.071268 | 15,970,630 | 0 | 0.0 |
| probe | rust-borrower | `"We need answer. Need explain borrow checker"` | 41 | 8 | 674.988837 | 576.196739 | 0.011852 | 0.0726 | 0.070856 | 0.071192 | 17,451,668 | 6,197,248 | 0.0 |
| serve-dev final | capital | `"We need answer. Need be concise. User asks capital of France. Answer Paris. Maybe mention. Need no extra."` | 37 | 24 | 970.36 (incl queue) | 519.296728 (server) | 0.0247 (incl queue) | 0.0629 (incl queue) | 0.070361 | 0.071250 | not sampled | not sampled | 0.0 |

The two probe requests: 78 prompt + 16 completion tokens in 1293.05 s of wall time =
0.01237 completion tokens/s end to end. The client TTFTs (519.48 s and 576.20 s) match
the server's `ttft_ms` (519168.08 ms, 575905.23 ms); prefill at ~0.0712 prompt tokens/s
is the dominant cost, and the server decode rate is ~0.0709-0.0710 tokens/s (~14.1 s per
step).

The final 24-token capital request returned 37/24 at finish length, with text naming Paris;
its 970.36 s wall includes queue wait behind the rust probe, so it is not a clean
end-to-end latency point. Its clean server numbers are TTFT 519.296728 s, prefill
0.071250 prompt tokens/s and decode 0.070361 tokens/s; at 24 tokens it finally names the
city ("... Answer Paris. ..."), still truncated at the cap. Its rank-0 trace was
`last_step_ms=14300 steps=60 reduce_calls=9600 reduce_elements=49152000`, the same
~14 s/step profile.

### Headline number, with its definitions

The Lead's headline is: 37+8 = 45 tokens in 618.64 s = **0.072 tok/s** at four ranks, versus
the single-node 37+16 = 53 tokens in 653.78 s = **0.031 tok/s**. The two numbers use
different denominators and must not be turned into a factor:

- 0.072 tok/s = 45 total tokens / 618.64 s. On completion tokens alone the same request is
  8 / 618.64 = 0.0129 tok/s.
- 0.031 tok/s is the single-node **server decode rate** (`decode_tok_per_s =
  0.0309066813`), not 53/653.78. On the same definitions as the four-rank row: total tokens
  53/653.78 = 0.0811 tok/s; completion tokens 16/653.78 = 0.0245 tok/s; server decode
  0.0309 tok/s; server prefill 0.2196 tok/s.
- Matched at completion tokens/wall, the four-rank short-output requests (0.0129, 0.0119)
  are **slower** than the single-node chat (0.0245). Matched at server decode rate, the
  four-rank engine is about 2.3x faster (0.0710 vs 0.0309). Matched at TTFT it is about
  3.1x slower (519.48 s vs 168.45 s for the same capital prompt). No tokens/s win is
  claimed here; the end-to-end rate for short outputs loses because prefill/TTFT dominates.
- Final 24-token capital request: 61 total tokens in 970.36 s including queue wait =
  0.0629 total tok/s; 24/970.36 = 0.0247 completion tok/s with queue included; its clean
  server decode is 0.0704 tok/s. Because the wall includes queue wait, do not use it as an
  end-to-end rate; the clean comparison remains server decode 0.0704 vs single-node 0.0309
  (~2.3x) and TTFT 519.30 vs 168.45 (~3.1x slower).

## Single-node comparison

| run | endpoint | prompt | text (verbatim) | ptok | ctok | wall s | client TTFT s | completion tok/s | total tok/s | server decode tok/s | rchar/token | read_bytes/token | cache state |
| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- |
| task-56 | /v1/completions | capital | `" a country"` | 5 | 2 | 102.44 | n/a | 0.0195 | 0.0683 | 0.01674 | 31.45 GB | 0.324 GB | ~99% page-cache hits |
| task-56 | /v1/completions | rust-borrow | `" It is"` | 11 | 2 | 129.55 | n/a | 0.0154 | 0.1004 | 0.01636 | 37.39 GB | 7.43 GB | ~80% page-cache hits |
| serve-dev | /v1/chat/completions | capital | `"We need answer simple. Need answer. "Paris" is capital of France."` | 37 | 16 | 653.781803 | 168.451205 | 0.02446 | 0.0811 | 0.030907 | not sampled | not sampled | not instrumented (warm-ish, followed earlier requests) |
| task-56 cold note | /v1/completions | 7+4 earlier | n/a | 7 | 4 | 174 | n/a | 0.0230 | 0.0632 | n/a | not sampled | not sampled | cold page cache, ~43.5 s/token |

The task-56 rows were the raw-completion baseline (two prompts, max_tokens 2); the
serve-dev row is the 37/16 single-node chat request. The single-node path re-reads weights
per step: 31.45-37.39 GB of logical reads per completion token, mostly page-cache hits but
0.65 GB / 14.86 GB from disk. Its text is also incoherent at short budgets (" a country",
" It is"), unlike the four-rank chat rows; that is a prompt/token-distribution effect
recorded elsewhere, not a serve arithmetic bug.

## Texts, verbatim

- Four-rank capital (serve-dev first, Lead repeat, our probe): `"We need answer. Need be concise."`
- Four-rank rust-borrower (our probe): `"We need answer. Need explain borrow checker"`
- Four-rank capital (serve-dev final, max_tokens 24): `"We need answer. Need be concise. User asks capital of France. Answer Paris. Maybe mention. Need no extra."`
- Single-node chat capital (serve-dev, max_tokens 16): `"We need answer simple. Need answer. "Paris" is capital of France."`
- Single-node raw-completion capital (task-56): `" a country"`; rust-borrow: `" It is"`.

The 8-token texts are grammatical English but generic and truncated by the cap; they do not
answer the questions. The 24-token final capital request does name the city ("Answer
Paris") while still finishing at length, and the two repeated 8-token capital requests
returned identical bytes, which is the determinism evidence.

## Resident-weight proof (the io deltas)

The probe sampled `/proc/2399562/io` over ssh before and after each request
(`transport: ssh:allenh1@192.0.2.80`; sampling succeeded, not reconstructed):

| request | rchar B | read_bytes B | rchar / completion token | read_bytes / completion token | rchar / s |
| :-- | --: | --: | --: | --: | --: |
| capital 37/8 | 15,970,630 | 0 | 1,996,329 (1.996 MB) | 0 | 25,839.8 |
| rust-borrower 41/8 | 17,451,668 | 6,197,248 | 2,181,459 (2.181 MB) | 774,656 (0.775 MB) | 25,854.8 |

That is ~2.0-2.2 MB of logical reads per completion token, with 0 B / 6.2 MB from disk,
versus 31.45-37.39 GB rchar and 0.65-14.86 GB read_bytes per completion token on the
single-node reference. The ~15,000x drop in logical reads and the essentially zero disk
reads are the proof that the 90.94 GB/rank resident set is being used instead of decoding
weights from the checkpoint every step. (The measured 2.0-2.2 MB/token is above the
expected ~0.3 MB/token quoted before the run; the observed value is recorded here, not the
expectation. It still includes prefill and non-weight reads, and it is small regardless.)

## Why it is correct but slow

Correct: four ranks produced identical coherent bytes for the same capital prompt (first
request and Lead repeat), and the resident set is demonstrably used.

Slow: the host reference engine is compute-bound. Run 2 measured **13.9 s/step** on the
four-rank host forward, with the profile split **attention 64%, decode 19%, collective
1.6%** (Lead-reported run-2 evidence; not independently re-profiled in this window). Our
numbers agree: server decode ~0.0704-0.0710 tok/s = ~14.0-14.2 s/step, prefill ~0.0712
prompt tokens/s = ~14.0 s per prompt token, so a 37-token prompt costs ~519 s of TTFT
before the first token.

The rank logs for the final request make the same point without inference: rank 0/1/2/3
load_ms 229274 / 261113 / 252423 / 245477, resident_cache_bytes 90944617800 on every rank,
last_step_ms 14300 / 14113 / 14121 / 14124, steps=60, reduce_calls=9600 (160 per step),
reduce_elements=49152000. That is the ~14 s/step and the 229-261 s load range, measured in
this window rather than quoted. The 9.8 GiB replicated set is resident now, so this is not
the old disk-read bottleneck; it is the reference attention/decode arithmetic on the host,
paid four times per step across four nodes.

The next lever is the device path (M4/M5): the host reference engine is the correctness
baseline, not the performance target. Avoid reading these numbers as a reason to tune the
host loop first, and avoid quoting them as a TP=4 speed win over the single node.

## Evidence

- Probe JSON: `/home/user/phase2-evidence/window-2026-10-18/serve-output-tp4-online.json`
  sha256 `d34a22944923980b598c93c00737577529e0582960dc2f55c36098d485e04f65`
- Probe markdown: `/home/user/phase2-evidence/window-2026-10-18/serve-output-tp4-online.md`
  sha256 `8652ebfe46bdd0d806a2f4fd8e2f086404abd2f9d14fa6ef2591f805f51afcf8`
- Rank-0 startup log: `/home/user/phase2-evidence/window-2026-10-18/tp4-run-20261020T033645Z/r0.log`
  (host copy with the same bytes: `.../tp4-run-20261020T033645Z-r0.log`, sha256
  `8672cb3dc60b09b85e9ea11ebcae5bbf206ef6acdf1d46c3ad80026f3bbb96ef`)
- Single-node chat datapoint (serve-dev): `chat-response-b75-20261020T030832Z.json`
  sha256 `5a29a4a2db8cb029c4e628481da4286178bbd0f21b1bbfac8ecdb89ce8f10551`;
  `serve-chat-b75-20261020T030832Z.out` sha256
  `8fb8e4d2f17398f445a26df94ffa1fcda9ef23909c13b507b37e53e7f6fb893a`
- Task-56 single-node raw baseline: `serve-output-b75-single-node.json`
  sha256 `69a9af289b43db096b6147f42d63160c3e4182e0167ccad4fb3a8a2c3ead508f`.
- Final capital max_tokens-24 evidence:
  `/home/user/phase2-evidence/window-2026-10-18/tp4-serve-first-20261020T035246Z/paris-response.json`
  sha256 `2a4425f39a4a015c1fa01bdfc469565150df38c02c0ca709f76797be14922bf5`;
  `paris-raw.json` sha256 `94974e30918244b12bdf668876cf73725b93463631a9cb9833e579de89612a3f`;
  `paris.meta` sha256 `bcfb6144e13c3502c42f50113bb66ade24c1d5b025f61c54b03ebff58986d73d`.
  Its rank logs are in the same directory: `r0.log` sha256
  `edca130bf46b377a925f3556611e2f6235f335cdd4ac49abc204c9df9954deae`, `r1.log`
  `27b0d6b9ae73a1b3e5e050ecf04a99e8147ddfe2ff0e1e24390051a27c1a09d8`, `r2.log`
  `777fb5047dd3c4060aacb7f8ad70a0190eb363b3bfcd49f58a71ae4fc1fab050`, `r3.log`
  `3d3eebce979a6f6771fbad67858c017f1a17db7085912104b4fbc6989d8170c8`.
- Requests: 5 total in the rank-0 log (serve-dev first, Lead repeat, the two probe calls,
  and serve-dev's final 24); exactly two requests were issued by this probe, sequential,
  one in flight.

## Final capital max_tokens 24 (serve-dev)

Returned after the two probe calls; the endpoint was torn down afterwards and no further
requests were sent. Text: `"We need answer. Need be concise. User asks capital of France.
Answer Paris. Maybe mention. Need no extra."`; usage 37 prompt + 24 completion = 61, finish
length. Server TTFT 519.296728 s, prefill 0.071250 prompt tokens/s, decode 0.070361
tokens/s. The 970.36 s local wall started at 04:22:47Z while the rust probe was still in
flight, so it includes queue wait and is not a clean end-to-end point. This is the request
that names the city; it is still cut off by the 24-token cap.
