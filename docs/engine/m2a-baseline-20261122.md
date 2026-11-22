# Milestone 2 part A baseline — 2026-11-22

Status: **recorded and passing**. This is the "recorded once on the starting tip" baseline
from [06-base-reads-description.md](06-base-reads-description.md) §3. Later milestone-2 steps
compare their open manifest, launch trace, tick trace and speed against these files.

## Provenance

- Run directory on the head node (not in the repo):
  `/home/allenh1/cluster-tests/20261122T202505Z`
- Test binary: `sparkflash-m2a-prod6`,
  sha256 `90a21bcf6c5476fd6d88c5ef90c0d239e4f2d04bf602c67bf650e1d5bbb49a0b`
- Source: the production line, commit `10a245df` (the commit the serving binary reports),
  plus the launch-tracing commit `9ed22a5`, plus the harness commits on this branch and
  `harness: record bound tensors where the bind happens` (test branch `harness/m2a-prod2`
  at `e08ba14`). The binary was built on the head node with
  `cargo build --release -p sparkflash-cli --features cuda,tp4-serve` and
  `SPARKFLASH_NVCC=/usr/local/cuda/bin/nvcc`, so its embedded model images are present.
- The integration tip `4b2c510` (`scan/gate`) refuses this checkpoint at open with 16
  unresolved vision/hyper-connection rows. That is a description gap, not a baseline failure:
  the baseline is the starting tip's own behaviour, which is the production line.

## Command

```
cd /home/allenh1/m2a-harness-prod
timeout -k 60 2400 python3 scripts/open-manifest-ab.py record \
  --build /home/allenh1/sparkflash/target/release/sparkflash-m2a-prod6 \
  --out /home/allenh1/cluster-tests/20261122T202505Z/baseline
```

Environment: the four-node launcher env (nodes `spark0 sg1-carter.local sg1-teilc.local
sg1-jackson.local`, peers `192.168.1.85 192.168.1.63 192.168.1.88 192.168.1.68`,
`SPARK_LOCAL_RANKS=0`, `SPARKFLASH_CHECKPOINT` the released snapshot,
`SPARKFLASH_CONVERTED_DIR`, `SPARKFLASH_CUBINS`, `SPARKFLASH_NCCL_LIB`,
`SPARKFLASH_TOKENIZER`, `SPARKFLASH_TP4_BASE_PORT=29601`, `SPARK_MAX_SEQ_LEN=1048576`,
`SPARKFLASH_LANES=8`, `SPARKFLASH_DRAFT_DRAFTS=5`, `SPARKFLASH_HTTP_BIND=0.0.0.0:8000`,
and the production NCCL vars). No `SPARK_COMM_CONTROL_PORT`: the exchange takes the
fabric default 18515.

## Counts

| rank | manifest lines | model | tensor | kernel | launch trace lines |
|---|---:|---:|---:|---:|---:|
| 0 | 93386 | 66 | 93131 | 189 | 254743 |
| 1 | 93386 | 66 | 93131 | 189 | 254743 |
| 2 | 93386 | 66 | 93131 | 189 | 254743 |
| 3 | 93394 | 66 | 93131 | 197 | 255332 |

`tick.trace`: 274 non-empty lines. The bind reported
`shards=95822 bound=93131 matched=93131` on every rank.

## Speed

```json
{"decode_1": 100.9777, "decode_8": 29.1516, "prefill": 5072.7985, "draft_accept": 0.7856}
```

The replay workload covered the image request, one lane, eight lanes, a 16k-token bulk
prefill, and eight drafting lanes (524 of 667 drafts accepted), all greedy.

## Hashes

```
a1066f8584a0a8e49c8678fcfaabe019e4370a281ab161dc745be7ddbe6c404d  launch-r0.trace
6d463afb864d83f129bad2a3e4471dc882a325a0cd815b1ad268a2d98f97bbaa  launch-r1.trace
6d96ee7abd6b38b8a01aa4e944831b0cb1679072251d5af10582df78365ec19d  launch-r2.trace
6a4528a541efe2317d047ed44e508cecb3f36714efe3c6105e0acb979811abce  launch-r3.trace
0f10a6cb3744302ca5edb71c5c56431d0e751e1c10f1ac4f3283b4748c93c1fd  manifest-r0.jsonl
f51ca18c2af7e7ba4268c84281b7908becc23227f48a32213acc3009427eec39  manifest-r1.jsonl
79a47322e07b247ba278607ebfd9a95680e61d7b39e0154e6d6c8cb9bdba74cc  manifest-r2.jsonl
df4e581c3f25cedd01c5bf56081514a10a278a2a638bd46c481d516ad8f806be  manifest-r3.jsonl
530d3b9368cbfe035b7a8880673dd20b72bf88d09f62dc88a45872a190e1e3c6  tick.trace
```

## Fabric and restore

Every rank opened the direct-RDMA exchange on control port 18515
(`4 x 655360-byte rows`), rank 0 warmed up in 978 ms, and the server listened on
`0.0.0.0:8000` before the workload. The production service was stopped for the window and
restored from `target/release/sparkflash` (sha256 `5a299b15...b132`, unchanged); /readyz
returned 200, the model card was identical before and after, and all four ranks reported
running. Exclusive window: 2 m 44 s.

## How later steps use it

```
python3 scripts/open-manifest-ab.py compare --baseline <this baseline> --candidate <new>
python3 scripts/open-manifest-ab.py ab --old <this baseline> --new <new> --out <dir> --reps 2
```

Manifest and launch trace comparisons are exact; the tick trace ignores only the clock; the
speed check is a tolerance comparison and is only required for a step that runs every token.
