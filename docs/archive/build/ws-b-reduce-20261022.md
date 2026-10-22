# WS-B host collective reduce: where the 0.974 ms goes, the bit-identical fast form, and the overlap seam

Owner: comm-lead. Branch `task/comm-reduce`, based on `origin/main` 268f2ab1 (the 249.25 ms/step
batch). No node was touched and no kernel or pin was changed; the four-rank rows are staged and run
by the Lead.

The measured target from the Lead's probe: one `all_reduce_bf16` of the served shape (5120 bf16
elements = 10 KiB per rank, `CollectiveLayout` one-shot all-to-all) shows **0.974 ms/layer** in the
`--forward-perf-probe-nosync` mode and **0.295 ms/layer** in the drained (device-inclusive) mode.
The truth is between them; this work explains both and cuts the call.

## 1. The canonical order (hard constraint, stated from the code)

One call reduces one element like this, and only like this:

- every rank contributes a bf16 payload; rank `r` sends its whole payload to each of the three
  peers (one-shot all-to-all, `FabricSchedule::OneShotAllToAll`);
- each rank receives the three peer payloads. Arrival order is irrelevant: the frames are placed
  in a pane table by **sender rank**, not by arrival;
- per element the accumulator is a f32 seeded with `0.0f` and the panes are added in ascending
  rank order: `0.0 + rank0 + rank1 + rank2 + rank3`;
- the f32 sum is rounded to bf16 exactly once, with round-to-nearest-even, in
  `f32_to_bf16_rn` (`crates/sparkflash-comm/src/reduce.rs`);
- negative-zero seed behaviour is preserved because the seed is `+0.0`, and the all-negative-zero
  payload reduces to `+0.0` as on the device.

Every transport (loopback, TCP, RoCEnante RDMA) executes the same `exchange::all_reduce` driver and
the same `reduce::canonical_all_reduce_into`. The asynchronous API applies the identical rule in
`wait_reduce`; only *when* it is applied moves, never the add sequence. The new four-rank unroll in
`reduce.rs` writes the same four additions in the same order with the pane index hoisted out of
the element loop; floating-point reassociation cannot happen because the crate does not enable fast
math.

Evidence: `cargo test -p sparkflash-comm` (bit-identity suites `reduction_bits`,
`loopback_transport`, `tcp_transport` and the canonical unit tests) passes, and
`comm-reduce-bench` recomputes the canonical reference locally on every rank and exits non-zero on
a mismatch; every run in the evidence directory printed `check=ok`.

## 2. Where the time goes (instrumented, not reasoned)

### 2.1 Instrumentation

`crates/sparkflash-comm/src/profile.rs` is compiled in and off unless `SPARK_COMM_PROFILE` is set.
One all-reduce records:

| field | what it measures |
| --- | --- |
| `prepare_us` | reader-thread handoff / socket `try_clone` (the spawn cost used to be here) |
| `send_us` | the three per-peer writes (sum), `send_max_us` is the slowest single send |
| `collect_us` | waiting for peer payloads (join or sequential read) |
| `reduce_us` | the local canonical reduction |
| `total_us` | the whole call |
| `peer_wait_us` | start-of-reader to complete-frame, summed over peers then averaged per call; this is the peer-wait evidence |
| `peer_wait_max_us` | slowest single peer wait in the window |

A window line is written to stderr every `SPARK_COMM_PROFILE_EVERY` calls (default 40): `0` turns
the periodic line off. Overhead when disabled is one relaxed atomic load per call.

### 2.2 One call in isolation, local four-process run

`comm-reduce-bench` is a new binary in the crate: it connects a real four-rank `TcpTransport`
cluster, performs the served shape repeatedly, measures per-call times, and checks the canonical
reference. It touches no GPU and needs no node configuration. The numbers below are a release
build, four processes pinned to cores 40-43 on this host, 200 samples after 20 warmup calls
(localhost TCP, so they price the protocol and the host path, not the switch RTT):

| variant | per-call p50 | phases (prepare / send / collect / reduce) |
| --- | --- | --- |
| origin/main sync, three reader threads per call | 381-566 us (two runs) | spawn+join dominate; no phase lines on base |
| new sync, sequential small-frame read | 153 us | 0.1 / 85 / 32 / 36 us |
| + four-rank unrolled reduce | 142 us | 0.1 / 85 / 31 / 26 us |
| + one write per small frame | **113 us** profiled, 120-126 us unprofiled | 0.1 / 64 / 25 / 26 us |

The asynchronous path (post + wait, persistent demultiplexers) is **130 us** per call in the same
run: post 56-78 us, wait 59-73 us. It is not faster per call; its value is the overlap seam (section
4).

The three-reader-thread path was the served path before this branch. It measured 0.381 ms and
0.566 ms p50 in two local runs (the spread is scheduler/load dependent, and both runs spawn and join
three threads per call); the final path measures 0.113-0.126 ms, so the handoff alone was **3-5x**
the call. The time was thread creation and join, not the wire. This directly predicts a large part
of the row's 0.974 ms.

### 2.3 The wait is rank skew, measured with a controlled stagger

The same bench takes `--stagger-us N`: rank `r` sleeps `r * N` microseconds before each call, so
the ranks arrive at the reduce at different times. With `N = 300` and 60 samples:

| rank | total p50 | collect p50 | peer_wait p50 | what it means |
| --- | --- | --- | --- | --- |
| 0 | 1039 us | 921 us | 917 us | arrived first, waited for rank 3 |
| 1 | 741 us | 626 us | 622 us | waited for rank 3 |
| 2 | 442 us | 329 us | 325 us | waited for rank 3 |
| 3 | 146 us | 25 us | 22 us | arrived last, executed the call |

The differences are exactly the stagger (300 us per rank step), while the last rank's 146 us is the
real work. This is the mechanism behind 0.974 vs 0.295: the no-sync probe removes the per-layer
device sync, ranks drift, and every rank except the last pays the drift inside `collect`. The
drained probe's sync boundaries align the ranks, so it sees ~0.295 ms of actual work. Neither
number is wrong; they measure different things. The instrumented forward row (`SPARK_COMM_PROFILE=1`)
will show the same split on the real four nodes, and a *large* `peer_wait_us` there is the drift
signature, not transport slowness.

There is no sleep or poll interval in the synchronous path: reads block in the kernel, and the
async path waits on a condition variable. The only millisecond-scale timeouts are frame deadlines
and the async readers' 50 ms shutdown-observation timeout, neither of which is on a completed call.

## 3. The fastest bit-identical form

All rows below preserve the canonical sequence in section 1; the choice is transport and handoff,
not arithmetic.

| form | per-call (10 KiB) | setup | status |
| --- | --- | --- | --- |
| TCP sync, three threads/call (main) | 566 us local; row 0.974 ms nosync / 0.295 ms drained | TCP connect ~5 ms once | replaced on this branch |
| TCP sync, sequential small frames + single write + unrolled reduce | **113 us local** (measured); row run pending | same TCP connect, no API change | **new default** for frames up to 256 KiB |
| TCP async post/wait | 130 us local (post 56-78 + wait 59-73) | persistent reader threads on first post | existing API; for overlap |
| RoCEnante RDMA one-shot (host-staged, dual rail) | **35.70 us p50** on the fabric (accepted 2026-10-21 WS-B run, 10 KiB, 2 x 200G) | QP/rail exchange once per process (ms scale); needs `rdma` feature and `SPARK_COMM_PEERS` | exists and passes the same canonical driver; **not wired into the forward row** (forward only builds `TcpTransport`) |
| NCCL device reduce | not measured | NCCL in the image | not a candidate yet: no bit-identical proof for a bf16 all-reduce in NCCL order, and the device seam is `Unsupported` |

Bytes on the wire are unchanged: each rank writes 3 x 10 KiB and reads 3 x 10 KiB per call (the
one-shot all-to-all), 60 KiB in and 60 KiB out cluster-wide for the 40-layer decode row at one call
per layer. The RDMA price is the same byte count on two rails.

Recommendation: land the TCP improvements now (no call-site change, ~5x per call locally, bit
identical). The next step after that is wiring the existing RDMA canonical path for this reduce,
which is measured at 35.70 us p50 and is the same `exchange::all_reduce` driver; that is a
forward-owner change (the row currently builds one `TcpTransport`), not a comm-crate change.

## 4. Overlap: what is possible without touching the sums

The synchronous call blocks exactly once: after the three writes it joins the readers (or reads
sequentially) and applies the canonical reduction. There is no second blocking point.

An issue/poll split already exists in the comm crate: `TcpTransport::post_reduce_bf16` writes the
frames and returns a handle that owns the caller's buffer; `TcpTransport::wait_reduce` waits and
then applies the *same* canonical reduction. The handle is exactly-once by construction, so the
contribution cannot be overwritten between post and wait, and the bits are identical to the
synchronous call (bench `check=ok`, tests).

Measured on localhost: post 56-78 us, wait 59-73 us, and under the stagger the wait is the skew
(rank 0: 906 us in the overlap run). The split is only worth something if there is independent
work to do between issue and poll.

That is the catch at the current call site: in `tp4_forward.rs` the reduce result is consumed
immediately after the call (`delta` is built from the reduced words right there), so within that
block there is no independent host work to hide the wait behind. The MoE reduce cannot start
before the partial exists, and the next layer cannot start before the combined delta exists.
Making overlap free would require the forward loop to issue the reduce at the end of one seam and
wait at a later consumer that is genuinely independent (the candidate is the device copy/combine
that consumes `delta`, if it can be moved after other host work); that is a forward-owner
restructure with its own measurement, not something this crate can decide.

So: the API is ready and bit-identical, and the mechanical "issue then poll later" is available; the
forward call site as written has nothing independent between the two points. Do not count on
overlap until a call-site A/B shows a real gap.

## 5. The three-line number

Per call, measured:

1. now: **0.974 ms** (no-sync probe) / **0.295 ms** (drained probe), Lead's row.
2. after this branch, isolated local call: **0.113-0.126 ms** sync (0.381-0.566 ms before,
   3-5x), async split 0.130 ms (post 0.06-0.08 + wait 0.06-0.07); the four-rank row number is the
   Lead's run.
3. at 40 layers x 2 calls/layer (arithmetic from the per-call number, not a new measurement):
   now 77.9 ms/step (no-sync) or 23.6 ms/step (drained); after 9.0 ms/step at 0.113 ms/call;
   with the existing RDMA path 2.9 ms/step at 0.0357 ms/call (fabric-measured per call).

## 6. What the Lead should run (no node config, no kernel, no vLLM)

1. Re-run the four-rank `--forward-perf-probe-nosync` row with

   ```text
   SPARK_COMM_PROFILE=1 SPARK_COMM_PROFILE_EVERY=40
   ```

   The stderr lines start with `comm-profile window_calls=...` and give the row's real
   prepare/send/collect/reduce and peer wait. Before the change the thread spawn was in
   `prepare_us`; on this branch it should be near zero and `collect_us` carries the drift.
2. Four-rank isolated bench in the same window (no GPU needed):

   ```text
   SPARK_COMM_PROFILE=1 ./target/release/comm-reduce-bench --rank R \
     --addrs 192.0.2.80:19701,192.0.2.88:19701,192.0.2.59:19701,192.0.2.68:19701 \
     --samples 200 --warmup 20 --mode sync
   ```

   Expected: p50 a little above the local 113 us (switch RTT plus the nodes' CPUs), and
   `check=ok` on all four ranks. If it is much above 113 us, the difference is in the fabric path,
   not in the handoff.

## 7. Changes on the branch

- `crates/sparkflash-comm/src/profile.rs` (new, private): opt-in phase and peer-wait recorder.
- `crates/sparkflash-comm/src/exchange.rs`: one-shot reduce instrumented only; no change to the
  transfer list or reduction.
- `crates/sparkflash-comm/src/tcp.rs`: small-frame exchanges read sequentially after the sends
  instead of spawning one reader thread per peer (deadlock-safe below 256 KiB because all writes
  complete before the reads and fit the socket buffers); small frames are written as one buffer in
  both the synchronous and posted path; reader-thread peer waits are recorded.
- `crates/sparkflash-comm/src/reduce.rs`: four-rank inner-loop unroll; same add order, same single
  RNE round.
- `crates/sparkflash-comm/src/bin/comm-reduce-bench.rs` and `Cargo.toml`: isolated per-call bench.
- `crates/sparkflash-comm/src/lib.rs`: registers the profile module.

The TCP path remains the fallback and the reference implementation; nothing was deleted. All
previous gates still pass: `cargo test -p sparkflash-comm`, `cargo clippy -p sparkflash-comm
--all-targets -- -D warnings`, `cargo fmt -p sparkflash-comm -- --check`.

Evidence: `/home/user/phase2-evidence/ws-b-20261022/comm-reduce/` (base, lean, unroll,
single-write, async, stagger and 2 MiB local runs; each log carries the bench line and the profile
windows).
