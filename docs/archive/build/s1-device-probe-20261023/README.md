# S1-0 device-arm probe, 2026-10-23 — evidence

The four DGX Spark nodes, one rank each (`node0` 192.0.2.80, `node1` .88, `node3`
.59, `node2` .68), ran the device arm once. This directory is the evidence; `probe-summary.json`
is the distilled record (command, binary/cubin/NCCL shas, per-rank bind and frontier results).

**Verdict: S1-0 PASS.** The arm opened on 4/4 ranks, the mHC-seam bind was `400 expected = 400 bound`
with no unexpected MTP names, the arena bind was complete, the 40-layer **TCP reference** collective
walk (transport `tcp`, phase `collective-skeleton`; not the device arm's NCCL enqueue, which the
forward never reaches) executed
80 reduces/step, and the arm ran its nine wired stage bodies on layer 0 position 0 before refusing at
`attention_body` (stage 10). All four ranks exited rc=2, the documented stage refusal.

**This run produced no step time, no logits, and no D54 row.** `speed-log.csv` is untouched and the
published number stays 179.81 ms/step.

## Files

| file | what |
| --- | --- |
| `probe-summary.json` | distilled record: argv, binary sha256, 30/30 cubin pins, checkpoint, nodes, per-rank bind + collective + frontier fields, verdict |
| `rank{0..3}-run.log` | rank stdout/stderr, including the frontier refusal line |
| `rank{0..3}-run.exit` | process exit code (`2` on every rank: documented stage refusal) |
| `rank{0..3}-bind.json` | arena bind record (`sparkflash-tp4-bind/1`): mHC seam 400/400, attention 208/208, MoE classes, census, per-token weight H2D 0 by construction (no per-token weight-copy path exists; not a probe counter) |
| `tp4-layer-device-r{0..3}.json` | device-arm frontier record (`sparkflash-tp4-layer-device-frontier/1`): status, layer, position, nine stages ran, frontier message, arm geometry (max_positions 138, rope_table_bytes 70,656, slab_bytes 996,864) |
| `collective-diagnostic-*.json` | **INVALID** collectives diagnostic (§2.2 of `docs/build/s1-result.md`): `scripts/gpu-reduce-bench.sh gate` with `SPARK_REDUCE_ALLOW_UNLOCKED=1`, because `nvidia-smi -lgc` needs privileges this account does not have. `valid: false`, `clock.locked: false`; do not quote its p50/p99 against the 30/60 us ceilings. |

Not committed (on the nodes under `~/ev-session-window/s1w01/`): the 11.9 MB `tp4-bind-names-r*.txt`
name manifests and the `tp4-loop-r*.json` walk records; `probe-summary.json` carries the fields read
from them.

## How it ran

Binary: `scripts/build-aarch64.sh --features=cuda` →
`target/aarch64-unknown-linux-gnu/release/sparkflash`, sha256
`c5a657e084c3261f3b6a9fad3208ec20b86c89329fedd2c9cd52d36b756ff91e`.

Staged per node as `~/ev-session-window/s1w01/stage/` with `sparkflash`, `cubins/` (the 30 cubins, each
sha256 equal to its `crates/sparkflash-kernels/baseline/kernel_baseline.txt` pin), `lib/libnccl.so.2`
(sha256 `aeda7a87ab4c3707216d428c6f4dee0f178ed87a6ed76f5bd6be9639e737a51a`, version 2.31.2) and the
138-token id list.

Per rank (R = 0..3):

```
cd ~/ev-session-window/s1w01/ev/rank$R
env SPARKFLASH_LAYER_DEVICE=1 LD_LIBRARY_PATH=~/ev-session-window/s1w01/stage/lib \
  ~/ev-session-window/s1w01/stage/sparkflash device-chain --run-stage tp4-chain \
  --checkpoint <DeepSeek-V4.1-Flash-1017 snapshot 1bae0e10...> \
  --tp-rank $R --tp-world-size 4 --evidence-dir $PWD --upload --loop \
  --comm-peers 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 --comm-base-port 29611 \
  --comm-timeout 120 --comm-connect-timeout 120 --warmup-steps 2 --decode-steps 9 \
  --forward --forward-preflight --kernel-cubin-dir ~/ev-session-window/s1w01/stage/cubins \
  --token-ids <138 ids> --forward-reference-ids 491,367,201,80,3302,201,54,201,4341 --forward-lean
```

The 138 ids are the 130-token b69 fixture prompt plus the first 8 reference ids (`reference_ids[0..8]`)
and are committed as `token-ids-138.txt` (sha256 `af1d3635ca32a9aa1f43bf137a2776a55a240ca7f13651d0b07d8cf6f4faf950`);
the prompt was read from
a host-arm evidence window on the nodes (`.../ev-m1w37-probe-lean/tp4-forward-embed-r0.json`
`prompt_ids`), not retyped. Four ranks exited rc=2 in 303 s wall (arena boot + bind + walk + the
forward attempt).

The collective diagnostic staged `target/aarch64-unknown-linux-gnu/release/gpu-reduce-bench`
(sha256 `f2b8b6790f31b6a889903358395ca5dcb7384d631c6cb0253fa72da6c91238cd`) with the same NCCL library
into `/home/user/gpu-reduce/s1diag/` on the nodes and ran:

```
SPARK_REDUCE_DIR=/home/user/gpu-reduce/s1diag SPARK_REDUCE_ALLOW_UNLOCKED=1 \
  scripts/gpu-reduce-bench.sh gate
```

It reported `gate: INVALID`: clocks stayed at 208 MHz (idle; 7 % of the 3003 MHz maximum) because no
rank could lock them, so the `p50 <= 30 us` / `p99 <= 60 us` latency ceilings cannot be evaluated from
this run at all: the verdict is INVALID, not a near miss. The
reduction itself was bit-exact (`max_ulp 0`, 0 differing elements).
