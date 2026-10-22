# RDMA hardware bring-up: first four-node run (task-35, 2026-10-20)

Status banner: the fabric came up, the counters gate is green, and the 20 us latency bar and the
10,000-replay determinism gate are not green. This is the first time the sparkflash-comm RDMA data
plane ran on the live GB10 cluster. The numbers below are measurements, not the pre-run hypotheses
in [rdma-backend.md](rdma-backend.md) section 11.4. The one-shot path is usable as a correctness
carrier (one full 10,000-replay run passed bit-for-bit) but it is not a qualifying transport yet:
median 10 KiB All-Reduce is 36.0-37.1 us against the 20 us bar on two rails, and only 1 of 5
dual-rail 10k runs completed without a transport error. No code path silently switched to the
host-staging transport; the failing runs aborted with the libibverbs completion error named below.

## 1. What ran

| | |
| --- | --- |
| date | 2026-10-20 UTC (2026-10-19 evening local) |
| base | main `2de9c07` (!131 batch 10) |
| branch | `task/task35-hardware` |
| gate binary | `hardware_gates-aa6cbdd9d8ce7373`, sha256 `5154d929973bce3eec1a82335c77bbcb03f585cd67791804ff140bb2c0c7db32` |
| probe binary | `rocenante-probe`, sha256 `0f8f0866ae03162c61d776b1890070f386c1a64d646cce7de8774f741fff5e43` (adds `p99_us` to the JSON; see section 3) |
| build | `cargo +1.98.1 test -p sparkflash-comm --features rdma --release --target aarch64-unknown-linux-gnu --test hardware_gates --no-run`, same for `--bin rocenante-probe` |
| cross toolchain | zig 0.17.0 wrappers at `~/opt/bin/sparkflash-aarch64-*`, `SPARKFLASH_RDMA_INCLUDE=/home/user/opt/aarch64-rdma/include` |
| shim | system (real `<infiniband/verbs.h>` inline wrappers); the build did not take the ENOSYS stub path |
| evidence on the nodes | `/home/user/phase2-evidence/window-2026-10-18/rdma-t35-2026-10-19/` |
| runner | `scripts/rdma-bringup.sh` in this branch. The runner rebuilds from this branch, so its printed hashes differ from the measured pair above (the probe `p99_us` edit changes the crate artifacts); the gate's behavior is unchanged. |

Every rank ran as user `allenh1`; `/dev/infiniband/uverbs0..3` are world-readable and
world-writable on all four nodes, so no root or sudo was needed. No GPU, no vLLM, no module reload
and no node reboot were used.

## 2. Fabric facts and the table that worked

The design doc's rail-1 default table is `.80/.84/.68/.59` (node0, node1, node2, node3). The Lead's
rank table for this run was rank 0 node0 .80, rank 1 node1 .88, rank 2 node3 .59, rank 3 node2 .68,
and that is the table that worked: all four control-plane connections established on the first
attempt, with no address fallback. It is a mixed rail-1/rail-2 table because `.88` is node1's
`roceP2p1s0f1` address and `.68` is node2's `roceP2p1s0f1` address; the peer table is
control-plane only (the data plane uses RoCE GIDs), so a mixed table is legal as long as every
address is local to its rank and rank 0's address is reachable from all four nodes. Task-53/55
should use this exact table for this cluster and record it in their evidence.

| rank | node | table address | `rocep1s0f1` | `roceP2p1s0f1` | f0 twins |
| --- | --- | --- | --- | --- | --- |
| 0 | node0 | 192.0.2.80 | ACTIVE, 200 Gb/s (4X HDR), MTU 1024 | ACTIVE, 200 Gb/s, MTU 1024 | `rocep1s0f0` DOWN, 40 Gb/s |
| 1 | node1 | 192.0.2.88 | ACTIVE, 200 Gb/s, MTU 1024 | ACTIVE, 200 Gb/s, MTU 1024 | `rocep1s0f0` DOWN, 40 Gb/s |
| 2 | node3 | 192.0.2.59 | ACTIVE, 200 Gb/s, MTU 1024 | ACTIVE, 200 Gb/s, MTU 1024 | `rocep1s0f0` DOWN, 40 Gb/s |
| 3 | node2 | 192.0.2.68 | ACTIVE, 200 Gb/s, MTU 1024 | ACTIVE, 200 Gb/s, MTU 1024 | `rocep1s0f0` DOWN, 40 Gb/s |

Raw per-node capture is in `linkstate-rank0..3.txt` in the evidence directory. The netdevs are
`enp1s0f1np1` (rail 1, `.80/.84/.59/.64`) and `enP2p1s0f1np1` (rail 2,
`.85/.88/.63/.68`), both at kernel speed 200000 Mb/s; `ibv_devinfo` reports
`active_mtu 1024` and `link_layer Ethernet` on both active devices. This is better than the
design's 400 Gb/s dual-rail assumption only in that both 200 Gb/s twins are actually up; the
measured one-shot latency below is nowhere near a wire limit.

## 3. Exact commands

The runner wraps all of this; the commands are repeated here so the run can be reproduced without
reading the script.

Build (on the x86 dev box, cross to aarch64):

~~~text
export CARGO_TARGET_DIR=/home/user/deepseek-v4.1-flash-spark/target
export SPARKFLASH_RDMA_INCLUDE=/home/user/opt/aarch64-rdma/include
export CC_aarch64_unknown_linux_gnu=/home/user/opt/bin/sparkflash-aarch64-cc
export CXX_aarch64_unknown_linux_gnu=/home/user/opt/bin/sparkflash-aarch64-cxx
export AR_aarch64_unknown_linux_gnu=/home/user/opt/bin/sparkflash-aarch64-ar
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/home/user/opt/bin/sparkflash-aarch64-cc
cargo +1.98.1 test -p sparkflash-comm --features rdma --release --target aarch64-unknown-linux-gnu --test hardware_gates --no-run
cargo +1.98.1 build -p sparkflash-comm --features rdma --release --target aarch64-unknown-linux-gnu --bin rocenante-probe
~~~

Stage the two binaries under `/home/user/phase2-evidence/window-2026-10-18/rdma-t35-2026-10-19/bin/`
on every node and verify the sha256 on both sides.

Gate rendezvous, one line per rank, launched together (the four addresses are the table that
worked):

~~~text
cd /home/user/phase2-evidence/window-2026-10-18/rdma-t35-2026-10-19
SPARK_COMM_HW_PERMISSION=1 SPARK_COMM_RANK=<0..3> SPARK_COMM_WORLD=4 SPARK_COMM_PEERS=192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 SPARK_COMM_CONTROL_PORT=18515 SPARK_COMM_TIMEOUT_MS=60000 nohup ./bin/hardware_gates-aa6cbdd9d8ce7373 --ignored --test-threads=1 --nocapture > rank<rank>.log 2>&1 &
~~~

Single-test form: append `--exact <test name>`. The gate names are:

- `dual_rail_counters_show_both_interfaces_carrying_half_the_bytes`
- `roce_bits_match_the_cpu_reference_over_ten_thousand_replays`
- `roce_oneshot_two_mib_all_reduce_meets_the_latency_budget`

Probe rendezvous after the gates (all four at once, full peer table because the transport opens
with one control-plane entry per rank):

~~~text
nohup env SPARK_COMM_HW_PERMISSION=1 SPARK_COMM_TIMEOUT_MS=60000 ./bin/rocenante-probe --rank <0..3> --world 4 --peer 192.0.2.80,192.0.2.88,192.0.2.59,192.0.2.68 --bytes 10240 --samples 200 --warmup 20 --timeout-ms 60000 --json > probe-rank<rank>.json 2> probe-rank<rank>.err &
~~~

Single-rail mode is `SPARK_COMM_RAIL_COUNT=1` on the gate and probe processes. Cleanup is
`scripts/rdma-bringup.sh cleanup`, which kills the recorded rank pids and checks that
port 18515 is free on all four nodes.

The probe JSON now carries `p99_us` as well as `min_us`, `p90_us` and `max_us`;
this branch adds it because the 20 us bar is a tail question as much as a median question and the
20-sample gate p99 was too coarse to record the distribution the Lead asked for. The doctor reads
only `oneshot_allreduce_us`, so the added field does not change its contract.

## 4. Gate results

### 4.1 Dual-rail counter gate: PASS

Every rank printed the same exact accounting for the 1 MiB one-shot:

~~~text
dual-rail run: rail0 1572936 bytes (50.00%), rail1 1572864 bytes (50.00%),
total 3145800 bytes vs 3145800 expected;
devices ["rocep1s0f1 (enp1s0f1np1)", "roceP2p1s0f1 (enP2p1s0f1np1)"]
~~~

This is the one gate that met its acceptance condition: both rail twins opened and carried the
whole-element split with the expected 3,145,800 B total (3 x (1,048,576 + 24) B posted per rank).
It proves the dual-rail data plane moves bytes on real hardware; it does not prove latency or
replay determinism.

### 4.2 Ten-thousand-replay determinism gate: not green

Each process opens one transport and runs 10,000 one-shot bf16 All-Reduce exchanges at 10,240 B,
comparing the output with the canonical CPU twin on every replay. Dual-rail attempts:

| attempt | result | where it stopped / what it reported |
| --- | --- | --- |
| 1 (three gates in one process) | FAILED | replay 1297: rank 0 and rank 3 `ibv_poll_cq failed: completion failed with IBV_WC_RETRY_EXC_ERR (opcode 0, qp 685, wr_id 0x100000000000000)`; rank 1 protocol violation, frame seq 1298 against schedule seq 1297 |
| 2 (exact test only) | PASSED | 10,000/10,000 replays bit-identical, 11.7 s |
| 3 (exact test only) | FAILED | replay 19/20: rank 1 `IBV_WC_RETRY_EXC_ERR`; rank 0 protocol violation, frame seq 21 against schedule seq 20; rank 2 and rank 3 retry-exceeded |
| 4 (exact test only) | FAILED | replay 5781: `IBV_WC_RETRY_EXC_ERR` (qp 719) |
| 5 (exact test only) | FAILED | replay 4411: `IBV_WC_RETRY_EXC_ERR` (qp 722) |

Single-rail attempts (`SPARK_COMM_RAIL_COUNT=1`): the exact 10,000-replay test passed once
(10.0 s, all four ranks bit-identical); the latency gate in the same mode failed during its
untimed warmup at replay 19 with `IBV_WC_RETRY_EXC_ERR`.

So the gate is demonstrated but not repeatable: 1 of 5 dual-rail runs and 1 of 1 single-rail
runs completed all 10,000 replays. A completed run produced bit-identical output on every replay
(the gate asserts this before the run can pass), so the failures are transport completions, not
silent numerical corruption. The failure shape is a completion queue retry-exceeded
(`IBV_WC_RETRY_EXC_ERR`) after a few hundred to a few thousand exchanges, with the peer that is
still waiting reporting a schedule mismatch because the failed rank has already advanced. That is
the bring-up risk the design predicted (section 10.1 of [tp4-design.md](tp4-design.md)), now with a
measured failure rate instead of a warning.

### 4.3 One-shot latency gate: FAILED

One complete 10 KiB measurement was obtained before the gate asserted:

~~~text
[rank 0] 10 KiB one-shot all-reduce over 4 ranks: median 39.616us, p99 3.994835ms;
specification bar 20us
~~~

The 2 MiB leg never ran in that attempt because the 10 KiB assertion stops the test first. The
probe measured the distribution directly; it is the better record of where the latency sits.

Dual rail, 10,240 B, 200 timed samples after 20 warmup, microseconds:

| rank | min | median | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| 0 node0 | 29.184 | 36.008 | 2065.225 | 2103.433 | 2178.809 |
| 1 node1 | 29.600 | 36.432 | 2102.601 | 3967.968 | 4042.080 |
| 2 node3 | 29.472 | 36.568 | 1924.237 | 4144.793 | 4218.393 |
| 3 node2 | 29.312 | 37.136 | 2056.946 | 2094.674 | 2170.849 |

Single rail, same payload and sample count:

| rank | min | median | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| 0 node0 | 25.936 | 29.584 | 31.104 | 2055.562 | 2405.369 |
| 1 node1 | 25.568 | 29.552 | 31.360 | 2098.073 | 2411.722 |
| 2 node3 | 26.720 | 29.536 | 31.055 | 2087.485 | 2102.109 |
| 3 node2 | 26.320 | 29.488 | 31.072 | 2106.114 | 4414.436 |

Reading the table: the best one-shot time is about 25.6 us (single rail), so even the minimum is
above the 20 us bar; the median is 36-37 us on two rails and 29.5 us on one. The p90 is normally
31 us on one rail, but about 1% of the samples sit at 2.0-4.2 ms in both modes, i.e. a retry-timer
sized stall, and that tail is what turns the gate's 20-sample p99 into milliseconds. The dual-rail
split costs about 7 us of median against single rail, consistent with twice the posted frames and
the second completion path; it does not pay for itself at 10 KiB.

2 MiB payload, dual rail, 30 timed samples after 5 warmup, microseconds:

| rank | min | median | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| 0 | 7926.183 | 8779.676 | 8922.132 | 8974.803 | 8974.803 |
| 1 | 7841.664 | 8767.707 | 8905.524 | 8977.412 | 8977.412 |
| 2 | 7899.923 | 8761.041 | 9026.960 | 9373.248 | 9373.248 |
| 3 | 7907.543 | 8761.503 | 9047.512 | 9187.911 | 9187.911 |

The gate's 2 MiB acceptance bar is 1.5 x the wire floor: 6,291,528 B posted per rank at 400 Gb/s
gives 125.8 us, and the bar is 188.7 us. The measured median is about 8.76 ms, roughly 70x the
wire floor and about 46x the bar. This is the host-staging cost the design predicted plus the
completion stalls; it is not a wire-rate result and must not be quoted as one. An earlier 2 MiB
probe with the same 30-sample shape had max outliers of 67.9-74.3 ms, so the tail is worse than the
table above suggests.

### 4.4 What is green and what is not

| item | result |
| --- | --- |
| two live rails per node, 200 Gb/s each, MTU 1024 | measured, all four nodes |
| dual-rail byte accounting, 1 MiB | PASS, exact 50/50 split and 3,145,800 B total |
| 10 KiB All-Reduce median <= 20 us | FAIL, 36.0-37.1 us dual, 29.5 us single |
| 10 KiB All-Reduce p99 over 200 samples | 2.1-4.1 ms, dominated by ~2 ms stalls |
| 10,000-replay determinism | 1/5 dual-rail runs, 1/1 single-rail run completed; completed run bit-identical |
| 2 MiB All-Reduce | 8.76 ms median against a 188.7 us acceptance bar |
| zero DtoH/HtoD, device seam | not run: no GPU window tonight; host seam tests still green (section 6) |

## 5. Measured gap and the decision point

The data plane is real: it opens both rails, registers memory, posts and completes, and its byte
accounting is exact. It is not fast enough (20 us bar missed by 1.8x on the median, and even the
minimum is above the bar) and it is not reliable enough (retry-exceeded at random replay counts,
1/5 dual-rail 10k runs). The honest options, in the order this document recommends them:

1. Keep the host-staging path as the correctness carrier for anything that needs a result
   tonight; it is the path the existing tests exercise, and this run does not qualify the RDMA
   path for serving.
2. Use single rail for the first device-seam attempt if a GPU window opens before the transport
   is fixed: it is 29.5 us median and it passed one full 10k run, but it has the same ~2 ms tail and
   the same retry failure mode, so it is a debugging configuration, not a qualified gate.
3. Do not redefine the 20 us gate. The measured gap is to be quoted as 36-37 us dual /
   29.5 us single median (min 25.6 us) with the p99 tail, and the NCCL-via-cudarc fallback described
   in [tp4-design.md](tp4-design.md) section 11 stays the fallback if the seam cannot be brought up
   on the RDMA path. There was no silent transport switch in this run: every failing rank returned
   the libibverbs error and the gate refused to report a number.

The next transport investigation (not done tonight) is the completion retry/stall: the
`IBV_WC_RETRY_EXC_ERR` completions and the ~2 ms p90/p99 plateau point at QP retry/RNR and
receive-posting behavior under a tight 10k loop, not at the wire. That is a data-plane fix in
`crates/sparkflash-comm/src/rdma/`, followed by a re-run of this exact runner.

For task-53/55: the numbers to use are the 10 KiB dual-rail row (median 36-37 us, p99 2.1-4.1 ms,
min 29.2-29.6 us) and, if they test the deployed single-rail configuration, the single-rail row
(median 29.5 us, p99 2.1 ms, min 25.6 us). Do not quote the 20 us target as achieved.

## 6. Repo side: runner, host tests, batch gate

`scripts/rdma-bringup.sh` is the repeatable form of this run: `build` cross-builds the
aarch64 gate and probe with the libibverbs include path, `linkstate` records the per-node rail
facts, `stage` copies and sha-verifies the binaries, `gates` and `probe` launch all
four ranks with `ssh -f` so the rendezvous is not serialized, `collect` copies logs and JSON
back, and `cleanup` kills the recorded pids and checks port 18515. It was exercised against
the live cluster after the measurements above: the counter gate ran green through the script and
the probe path, collect and cleanup all completed, and `cleanup` printed `port free` on
all four nodes. The `--samples 20 --warmup 2` probe smoke run through the script is not used for
any number above (its samples caught a stall-heavy window); the table rows come from the 200-sample
runs launched directly.

Host-side tests for the changed probe: `cargo test -p sparkflash-comm --release --lib probe`
passed 13/13; `cargo clippy -p sparkflash-comm --all-targets --features rdma -- -D warnings`
was clean. The branch also passes `scripts/batch-gate.sh --fast` (result recorded in the commit
message and handoff). The hardware-sensitive tests remain `#[ignore]`d and refuse without
`SPARK_COMM_HW_PERMISSION=1`.

## 7. Evidence, cleanup, and limits

Evidence directory (all four nodes): `/home/user/phase2-evidence/window-2026-10-18/rdma-t35-2026-10-19/`

- `rank0.log` ... `rank3.log`: first three-gate run, including the counter pass, the
  replay 1297 failure, and the 10 KiB median/p99 line.
- `gate2-rerun-rank*.log`: the passing 10,000-replay run.
- `gate2-third-rank*.log`, `gate2-dual-2-rank*.log`, `gate2-dual-3-rank*.log`:
  the failing repeat runs (replay 19/20, 5781, 4411).
- `gate2-single-rail-rank*.log`: the passing single-rail 10,000-replay run.
- `gate1-single-rail-rank*.log`, `gate1-dual-2-rank*.log`: latency-gate warmup failures.
- `probe-p99-rank*.json`: dual-rail 10 KiB, 200 samples.
- `probe-1rail-rank*.json`: single-rail 10 KiB, 200 samples.
- `probe-2mib-p99-rank*.json`: dual-rail 2 MiB, 30 samples.
- `linkstate-rank*.txt`: per-node devices, rates, states, addresses, uverbs permissions.

Cleanup was verified after the last rendezvous: no rank process remained on any node and
port 18515 had no listener on any node. No GPU, vLLM, module, network-config or reboot action was
taken.

Honest limits:

1. These are one evening's runs on a shared cluster, not a qualified benchmark. The probe's 200
   samples per rank are enough to expose the ~2 ms tail and not enough to characterize it; the
   10k gate result is a flake rate of 1/5 dual rail, not a confidence interval.
2. The dual-rail 1 MiB counter gate uses the backend's own posted-byte ledger. It verifies the
   split and total exactly, but a passing ledger is not independent proof that every byte crossed
   the wire; the 10k bit-identity run is the correctness evidence and it is flaky.
3. The 2 MiB probe number is host-staging latency, not a device-buffer number. The device seam
   (section 15 of [rdma-backend.md](rdma-backend.md)) remains a host-tested design until a GPU
   window opens; nothing here claims zero DtoH/HtoD.
4. The table that worked is the mixed table in section 2. The design doc's rail-1 table remains the
   documented default for a window where the rail-1 segment is routable; the mixed table is the
   measured one for this cluster's management path.
