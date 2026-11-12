# The direct-RDMA exchange: what we tried

This records one attempt to remove a collective library from the decode step's critical path, the
measurements that decided each move, and the result. The exchange is built, wired, measured, and
**on by default**; `SPARKFLASH_EXCHANGE=nccl` puts the collective group back.

The short version: the fabric can do this, and over both rails the exchange makes the decode step
**2% shorter than the collective group does, at c1, c8 and c16**, with the group's bits. The premise
promised more: 32 us off each of a step's 86 collectives would be 2.8 ms of 38.6, or 7%. A quarter
of that arrived. A c1 collective carries 60 KiB and not the 10 the premise measured, so the wire is
bound by throughput and not by latency, and around the wire sit a fence, the ranks' own skew and a
memory read that no transport removes. The table under "Where it stands" prices each of them.

It took three moves, and one of them did not pay. The first version, with the host waiting on the
device once per collective, was 2% faster than the group at c1 and neutral at c16. Taking the host
off the path, which this document used to name as the next move, bought nothing at c1 on its own.
Splitting every row over both rails is where the rest came from: against the first version the
final build's step is 2.5% shorter at c16, 1.3% at c8 and about 0.3% at c1 (the 0.6% this
line first carried was the size of the *slower* direction; see the correction under "Against the
first version").

## The premise

The layer's all-reduce was one NCCL group of point-to-point sends and receives, measured at **43 us**
per collective with 84-95 collectives in a c1 step. The proposal was to replace it with what the
fabric actually needs for an all-to-all: one RDMA write per peer into a pinned slot, an arrival flag
per rank, and a spin. On the vendor's numbers that projected ~11 us per exchange.

Everything here is on four DGX Spark GB10 nodes, TP=4, one rack.

## What the hardware allowed

The plan depends on a NIC writing into memory the GPU reads, without `nvidia-peermem`. That is the
gate, and it is a hard yes or no:

| question | measurement |
| :-- | :-- |
| does `ibv_reg_mr` take GPU-visible pinned memory? | **yes**: `cudaHostAlloc(MAPPED, PORTABLE)`, registered on both nodes |
| does it take `cudaMalloc` device memory? | **no**: `EBADF`, so the peermem limitation is real and unchanged |
| does the GPU read exactly what the NIC wrote? | **yes**: 2000 round trips of 10 KB, payload verified word for word, 0 bad words |
| is the mapped alias the host pointer? | **yes** (`host_alias_equal=1`), so nothing is copied in either direction |

Two costs decided the design and are worth stating because they are not in the vendor note:

- **The doorbell is cheap until it is fenced.** A resident kernel publishing a word to the host costs
  1.07 us (plain store), 1.39 us (`st.global.wt`), and **7.54 us with `__threadfence_system()`**,
  with a 3.26 ms outlier. The store is not the cost; the fence is.
- **A kernel launched at an idle GPU starts 2.2 ms late** (2.9 us to enqueue, 2.2 ms to completion,
  repeatably), while a resident spin kernel sees an arriving flag in 1.2 us. So a kernel that exists
  only to wait for a peer is 2.2 ms worse than a resident one, and any per-collective kernel has to
  be launched into a busy stream.

Probes: `scripts/probes/gb10-rdma-gate.cu` (registration, coherence, the round-trip verify) and
`scripts/probes/gb10-doorbell.cu` (the three doorbell variants).

## Measuring the exchange itself

`scripts/probes/gb10-exchange-4rank.cu` runs the real thing on all four nodes: a registered slot
per peer, one write per peer, an arrival flag per rank, and a wait until every peer's flag is at
least this rank's sequence. `scripts/probes/run-exchange-4rank.sh` drives it. 200 rounds, the first
16 discarded:

| row | first arrival | all three peers | NCCL group |
| --: | --: | --: | --: |
| 10 KiB (c1) | 5.5-6.0 us | **7.3 us** p50 (p99 9-13) | 43 us |
| 160 KiB (c16) | 42.5 us | **46.5 us** p50 (p99 50) | 43 us |

Re-running on an idle box gave 7.30 and 46.5: the same within noise, so at these sizes the exchange is
latency-bound and does not care about the collective traffic sharing the NICs. The 160 KiB row is a
wash against 43 us, and that is the same "the all-to-all is at the transport's floor" finding the
survey measured from the other direction.

**Two bugs cost most of a session here, and both look like a lost packet from the outside.**

- The arrival wait compared each peer's flag with `==` and counted *transitions*. A peer already
  ahead was skipped by the loop's early `continue` and never counted, so the tally could never reach
  the world. It deadlocked three ranks out of four on every run. The fix is to treat arrival as a
  predicate over every peer, re-evaluated every poll, compared `>=` and never `==`: a rank that gets
  lapped must not wait for a round that has already been overwritten, and it cannot be more than one
  round ahead because every rank waits for the whole world before advancing.
- The probe trusted GID index 3. A wrong index builds a QP that connects, accepts posts, and never
  delivers, which reads exactly like a lost packet. Selecting the GID by the rank's own rail address
  is the fix.

## The module, and three things only hardware said

`crates/sparkflash-comm/src/rdma/write_exchange.rs` carries `WriteExchange` and `SlabLayout`: the
QP mesh over the endpoint-exchange control plane the RDMA backend already runs (so nothing new
listens), the payload-plus-flag post, the `>=` arrival wait, and the completion reap that has to
happen before the send row is rewritten.

`write-exchange-smoke` drives it with the row staged on the host, checking every peer's row and flag
word for word every round: **p50 9.0 us at 10 KiB and 46.7 us at 160 KiB**, the probe's numbers plus
about 1.7 us for the reap - which is the 10 KiB reading; at 160 KiB the reap's 1.7 us is inside the
rounding of a 46.7 us round and does not move it. Three findings:

- **The flag must be published from inside the registered region.** A first version posted an 8-byte
  SGE pointing at a stack buffer; every payload write completed SUCCESS and the QP then died with
  `LOC_PROT_ERR`, which downstream is indistinguishable from peers that never publish. The timeout
  now carries the completion statuses, because that is what named it.
- **One row set is unsafe.** A peer a round ahead overwrites rows a slower rank is still reading,
  which the smoke test caught as "peer 3's row reads ...2, not ...1". Two sets selected by `seq & 1`
  are sufficient, and `SlabLayout` carries the parity in its row offset so the sum kernel keeps its
  `gather + peer * count * 2` addressing inside one set.
- **The first collective of a run is not the one to measure.** Round 0 costs 10-22 ms on every rank;
  everything after it is the 9 us above. This was put down to neighbour resolution and QP ramp. Most
  of it was ranks writing to peers that had not connected yet, which "The first round after the
  open" below measures and removes.

## The first A/B, and why it was 2% and not 20%

`SPARKFLASH_EXCHANGE` selects the arm: unset means the exchange, `nccl` means the group, `rdma`
means the exchange and refuses to start if it cannot be had, and anything else is refused at bring-up
by name. That distinction matters: an explicit `rdma` **never** falls back, while the default does
stand down to the group with one printed line when the run has no peer table to configure the
exchange with, because a default must not refuse to start a run that was never told where its peers
are. It opens **once per process** on the first collective
(`crates/sparkflash-deepseek-forward/src/rdma_exchange.rs`, a `OnceLock` and a mutex): `DeviceLayer`
is rebuilt every step, so a pinned registration and a QP mesh per step would cost more than the
collective they replace, and the rank is only known from the run's reduce handle.

The exchange replaces **only the per-layer one-shot all-reduce** - the decode and verify rows. Bulk
prefill chunks, the overlapped bulk collective and any run without a device reduce handle keep the
NCCL group, and the open prints one line saying so, so a log reader does not conclude that every
collective moved.

Same binary, the switch the only difference, three samples each:

| shape | NCCL group | direct-RDMA exchange | change |
| :-- | --: | --: | --: |
| c1 decode | 64.5 / 64.6 / 64.8 tok/s | 65.9 / 66.0 / 66.2 tok/s | **+2.2%** |
| c16 batch | 247.9 / 247.9 / 248.9 tok/s | 249.4 / 250.3 tok/s | inside the noise |

The greedy 48-token generation came back byte for byte identical on both arms, so the swap is
numerically free.

Both A/B tables and the smoke test's round timings were taken before the round's row length went on
the wire (`d0889175`); that check is arithmetic over words the round already carries and does not
change the timing, but the measurements are not from this commit and have not been re-run on it.

What it replaces is the per-layer one-shot all-reduce *and* the step's one-shot slot exchanges (the
argmax, embedding, Engram and draft gathers); bulk prefill, the ring gathers and the overlapped
collectives keep the group. So a decode step's exchange rounds are about 86 per-layer all-reduces
plus about nine slot exchanges - the pump's `rounds` counter counts both, and it is the counter to
compare against a step, not the 86.

The exchange is genuinely on the path, not merely configured: all four ranks announce the open with
their own pinned slab and the control plane on a separate port, and the switch is visible in the
live process's environment. (The rail counter this paragraph first quoted - 119 MiB on the
exchange's rail and zero on the other - came from the one-rail build and the 42-collective
arithmetic that the next section retracts. The reproducible figure is per step: 86 collectives at
a 56-60 KiB mean row sent to three peers is about 14 MB from each rank, half of it on each rail.
Multiply by the steps of a run - a c1 run at ~5 tokens a step is roughly 300 steps - to get a
run's total; the paragraph's 119 MiB is not that number and this one earlier tried to be.) A later one-shot sample straddled the baseline's (64.3-65.2 vs 64.5-64.8), so the honest
claim is "slightly faster", and the paired run above is the one that resolves it: restart-to-restart
scatter on c1 is about +-1 tok/s, which is the same size as the effect.

**Why the arithmetic did not hold.** This section first divided the rail's traffic by a 10 KiB row
and concluded there were 42 collectives a step. The pump now counts them, and both terms were wrong.
A decode step makes **86** exchange collectives, and a c1 row is rarely one hidden vector: the step
sends the token's row and its draft rows through the same collective, so the usual c1 row is
**60 KiB** (the token and five drafts) and the mean over a run is 56 KiB. At that size the wire is
not the probe's 7.3 us either; it is about 17 us, measured below.

What that section got right is the cause. The launch thread spun on the doorbell, posted the writes
and spun on the arrival flags before it launched the fold. So once per collective the device drained
to idle while the host waited for it, and nothing could be queued behind the collective. The
handshake around the wire cost more than the wire, and a faster transport could not help while it
did.

## Taking the host off the path

The rework keeps the wire and removes the wait. It has three parts.

- **The launch thread only submits.** `WriteExchange::submit` takes the next sequence number,
  leaves the row size where the pump will find it, and returns. The caller launches two kernels back
  to back and waits for neither, which is exactly what it does with the collective library.
- **The device does the waiting.** `opt_exchange_stage_bf16` is one block: it copies the row into
  the slab, fences, rings the doorbell with the sequence number, and then polls the peers' arrival
  flags, one warp with a lane to a flag, until every one has reached that number. The fold,
  `opt_sum_ranks_bf16`, is unchanged and queued behind it on the same stream. It reads the same rows
  in the same rank order as the group arm's fold, so the bits are the group's.
- **A pump thread per rank owns the verbs objects.** It watches the doorbell word, posts the round's
  writes when it rings, reaps completions between rounds, and keeps an eye on the last round's
  arrival. If a peer never publishes, the pump sets an abort word that the kernel also reads, so a
  dead peer ends the run with a reason instead of holding the stream forever.

Rail opening is now the RDMA backend's own (`open_rail`, `registered_at`), so the exchange and the
transport refuse the same things: a port that is not ACTIVE, a link layer that is not Ethernet, a
GID that is not RoCE v2 over IPv4, and a slab that was not registered at the address the peers were
told. Those were the review's open MR-base, GID-kind and port-state items.

## Four things the rework measured

- **A waiter in every block starves the copy it is waiting behind.** The first non-blocking version
  put the wait in the fold, so every block of the fold polled the flags in host-mapped memory. The
  c1 step went from 38.6 ms to **52.4 ms** and the c16 step from 138 ms to 376-380 ms. One block
  ahead of the fold, with one warp of it waiting, is the design now.
- **The system fence is not optional.** It costs about 6 us a round, which made it the obvious thing
  to remove. With `__threadfence()` in its place, and with no fence at all, six c1 runs of 1536
  greedy tokens came back with six different hashes; with `__threadfence_system()` every run has
  the NCCL arm's hash. Without it the NIC sends a row that the device's stores have not reached yet.
  The bit-exact gate caught this, and the kernel's comment now records it.
- **Rows in the slab are slower to fold, and the slab's kind of memory is not why.** The fold
  averages 14.7 us when its three peers' rows are in the host-mapped slab and 9.7 us when they are
  in device memory, where the group arm leaves them. `scripts/probes/gb10-slab-read.cu` asked
  whether other pinned memory would do better. Reading 240 KiB costs the GPU 4.6 us from device
  memory and 10.6 us from `cudaHostAlloc` memory even when nothing has touched it; ordinary memory
  registered afterwards reads at device speed while it is left alone, but once the rows have been
  rewritten from outside and are in nobody's cache, both kinds cost 9.8-9.9 us. A row that arrives
  from the NIC costs a memory read wherever it lands. The group arm pays the same read inside the
  library's own copy, so this is not a cost the exchange added.
- **A row set's base must not move with the row size.** The first layout put a row at
  `(parity * world + rank) * row_bytes`. Consecutive rounds do change size (a step boundary changes
  the number of draft rows), and then a small round's second set lands inside the large round's
  first, where a slower rank may still be folding. Each set now starts at a fixed offset. The smoke
  test alternates the row size every round and checks every word of every row, so this case is
  covered where it was not before.

## Where a collective's time goes

With the handshake gone, a trace in the pump (the time of each doorbell, each post, and each peer's
flag first seen, on all four ranks) says what a c1 collective still costs. A step is 38.4 ms over 86
collectives, so one collective and the compute up to the next take about 450 us.

| what | measured |
| :-- | --: |
| spread between the first and the last rank's doorbell | 10.7 us p50, 13.6 mean |
| share of rounds in which each rank rang last | 20-31%: no rank is persistently late |
| the last rank's wait after its own doorbell, 60 KiB row | **17.4 us** p50 |
| the same at 10 / 20 / 30 / 40 / 50 KiB | 4.1 / 6.4 / 8.3 / 11.9 / 15.6 us |
| mean wait of any rank, 60 KiB row | 26.5-28 us |

The last rank to ring waits only for the wire, and its wait grows in a straight line with the row:
about 0.27 us per KiB of row, which is three copies of it leaving through one device at roughly
100 Gb/s. So at the sizes a step really sends, the wire is bound by throughput and not by latency.
The other ranks wait longer only because they rang earlier.

Three costs are left that this design cannot remove: the fence (6 us), the spread between the ranks
(about 10 us, which is compute finishing at slightly different times), and the slower fold (5 us).

## Two rails

Every node has two RoCE devices, and the NCCL arm is already launched over both. `ib_write_lat`
between two nodes says they are independent: a 60 KiB write costs 9.8 us on either rail and the same
on both at once (7.0 us at 30 KiB, 12.0 at 90, 19.4 at 180, 33.8 at 360, so about 5 us plus 0.078 us
per KB).

So the exchange now splits every row in two on a 64-byte boundary and gives each rail one part. A
part travels with its own arrival flag chained behind it on its own QP, the flags are per rail and
per rank, and the kernel and the pump wait for all of them: a peer's row has arrived when both of
its flags have. `SPARK_COMM_RAIL_COUNT` (the RDMA backend's existing setting, 2 unless told
otherwise) selects one rail or two, and the endpoint exchange already refuses ranks that disagree
about it.

`write-exchange-smoke --rails 1|2`, staging on the host, 2000 rounds beside a live server, every
word verified:

| rows alternate between | one rail, p50 | two rails, p50 |
| :-- | --: | --: |
| 60 and 30 KiB | 18.2-21.3 us | **14.6-17.1 us** |
| 160 and 80 KiB | 37.1-42.8 us | **25.7-28.7 us** |
| 10 and 5 KiB | - | 6.7-6.9 us |

## The A/B now

One binary, the switches the only difference, every arm relaunched. A c1 sample is 1536 greedy
tokens. The figure is the mean decode step, so lower is better.

| shape | NCCL group | exchange, one rail | exchange, two rails | two rails against the group |
| :-- | --: | --: | --: | --: |
| c1 step, ms | 38.61 / 38.65 / 38.67 | 38.38 / 38.43 / 38.44 | **37.84 / 37.92 / 37.92** | **-1.9%** |
| c8 step, ms | 93.57 | 92.78 | **91.80** | -1.9% |
| c16 step, ms | 139.25 / 139.90 | 139.55 / 137.96 | **137.59 / 136.44** | -1.8% |
| c16 throughput, tok/s | 195.3 / 195.3 | 197.5 / 199.8 | **200.2 / 201.0** | +2.7% |

The served tokens hash the same on all three arms at every shape, so the swap is still numerically
free.

The lead's transmit counters around each arm say which rail carried it. The group sends 91% of its
bytes on one rail although it is launched over both; the exchange on one rail sends 98% there; the
exchange on two sends 50.5% and 49.5%.

The pump's trace again, on two rails beside the one-rail figures above (the last rank's wait after
its own doorbell, p50):

| row | 10 KiB | 20 | 30 | 40 | 50 | 60 |
| :-- | --: | --: | --: | --: | --: | --: |
| one rail, us | 4.1 | 6.4 | 8.3 | 11.9 | 15.6 | 17.4 |
| two rails, us | 3.4 | 4.9 | 5.9 | 8.7 | 10.9 | 11.3 |

Six microseconds off each of 86 collectives is the half millisecond the step lost. It is also all
this fabric has: `ib_write_lat` moves 90 KiB on one rail in 12.0 us, and each rail now carries three
30 KiB parts of a 60 KiB row.

## Against the first version

What the rework was worth is a comparison with the branch's first, blocking exchange, not with the
group. The first table is c1 only, with the rework on one rail and on two, paired and alternated
twice. The second is the final build on two rails against the blocking build at every shape, again
alternated twice (a comma separates the two launches).

| c1 step, ms | first pass | second pass |
| :-- | --: | --: |
| blocking exchange, one rail | 38.13 / 38.17 / 38.18 | 38.15 / 38.31 / 38.31 |
| non-blocking, one rail | 38.37 / 38.42 / 38.42 | 38.48 / 38.48 / 38.43 |
| non-blocking, two rails | 38.03 / 38.01 / 38.06 | 38.03 / 38.08 / 38.07 |

| shape | blocking, one rail | final build, two rails |
| :-- | --: | --: |
| c1 step, ms | 38.02 / 38.04 / 38.23, 38.27 / 38.24 / 38.46 | 38.59 / 38.56 / 38.53, 37.98 / 37.96 / 37.99 |
| c8 step, ms | 93.19, 92.79 | 91.72, 91.78 |
| c16 step, ms | 140.69 / 138.29, 143.10 / 140.20 | 137.65 / 136.56, 137.83 / 136.34 |
| c16 throughput, tok/s | 195.3 / 197.4, 193.4 / 196.5 | 197.1 / 198.5, 196.9 / 197.6 |

- **Taking the host off the path did not pay at c1 on its own.** On one rail the rework is 0.2 ms a
  step (0.6%) slower than the blocking version it replaced, in both passes. That is not explained.
- **Two rails put it ahead, and by more at c16 than at c1.** The final build's step is 2.5% shorter
  than the blocking build's at c16 and 1.3% at c8. At c1 the three pairings where neither launch was
  a slow one read 38.02 / 38.04 / 38.23 ms against 37.98 / 37.96 / 37.99, which is 0.12 ms and 0.3%
  and not the 0.2 ms / 0.6% this bullet first carried: 0.6% is the size of the *slower* direction
  quoted above it, and the two signs were given the same number.
- **The fourth pairing had a slow launch on the final build's side**: 38.5-38.6 at c1, with its c8
  and c16 steps ordinary. The note on scatter below says what is known about those.

## One warp waits

The kernel first had one thread read the flags in turn. The idea behind giving the wait to a warp, a
lane to a flag, is that a look at every flag becomes one load where the single thread made six, one
after another, out of host-mapped memory. Same binary, the kernel's cubin the only difference, two
rails, launches alternated, c1 step in ms with two samples a launch:

| wait | launches |
| :-- | :-- |
| one thread | 37.97 / 38.05, 38.08 / 38.04, 38.04 / 38.02, 37.98 / 38.04, 38.05 / 38.04 |
| one warp | 37.81 / 37.85, 37.87 / 37.95, 37.90 / 37.93, 37.93 / 38.04, 37.88 / 37.97, 37.97 / 37.99 |

That is 0.1 ms a step, 0.3%. It is at the edge of what relaunching resolves (see the note on scatter
below), and the warp is kept because in both runs of launches every warp launch beat every
one-thread launch. How often the kernel looks at the abort word, every look or every 256th, made no
difference that could be seen. The warp bounds the flags the kernel can watch at 32, and the open
refuses a world and rail count beyond that.

## The host does not wait better

The rework on one rail came out 0.2 ms behind the blocking version (above), and the obvious suspect
was the device's wait: perhaps the host notices an arrival sooner than a kernel polling host-mapped
memory does. An experiment build let the launch thread wait again, on two rails and with the pump
still posting: the kernel rings the doorbell and ends, the launch thread spins on the flags and
launches the fold when they are all there. One binary, an environment switch the only difference,
launches alternated:

| wait | c1 step, ms, two samples a launch | c16 step, ms |
| :-- | :-- | :-- |
| on the device | 37.97 / 38.03, 37.95 / 37.97, 38.01 / 38.00 | 137.80, 134.90, 137.68 |
| on the host | 38.31 / 38.42, 37.84 / 37.95, 37.73 / 38.00 | 139.56, 136.28, 138.50 |

Nothing at c1 and about 1% lost at c16, so the wait stays on the device and the switch was not kept.
It also says where the 0.2 ms is not.

**A note on scatter.** Launches of one build differ. Most two-rail launches sit within 0.1 ms of
each other at c1, but three of the 27 that were benchmarked came up 0.4-0.7 ms slower at c1 for as
long as they lived: 38.50 / 38.68 / 38.58, the 38.31 / 38.42 above, and the 38.59 / 38.56 / 38.53 in
the comparison with the first version. In those launches rank 0's pump sees its peers' flags later
(28.0 us from post to last flag over a c1 run, where an ordinary launch has 21.3), so some rank is
ringing late or the wire is slow for that launch. Which rank, and which of the two, is answered
below under "Which rank a slow launch waits for"; why is not. No launch of
another arm did this, but there were only nine of them. A difference under 0.2 ms therefore needs
alternated launches and more than one of each, which is how the tables in this half of the document
were taken.

## The first round after the open

The exchange opens on the first collective of the first request. The endpoint table tells every rank
where its peers' queue pairs are, each rank then connects its own, and until this was looked at a
rank posted its first round the moment its own were connected. Nothing made it wait for the others.

Two launches lost their first request to that round. Both were on two rails, and both times rank 1's
first write on its second rail completed with a remote invalid request (`IBV_WC_REM_INV_REQ_ERR`;
the device's `req_remote_invalid_request` counter agrees) while the other three ranks left only
because the control channel closed. That is two of the thirteen two-rail launches made before
anything was changed; the same binary served the next time it was launched, and so did every other
build, so it is a race and not a build. It never reproduced on demand: thirty opens of the smoke
test, which posts its first round just as early, all passed. So what the peer refused is not
established, and the link to a peer that was still connecting is the likely explanation, not a
proven one.

Two changes came out of it.

- **A failed write says which write it was.** A write's `wr_id` now carries its round, its peer and
  whether it is a row part or a flag, and the pump's report reads "round N's row part to rank P on
  rail R failed T us after it was posted" with the status by name, where it used to give a bare
  status number.
- **No rank posts before every rank is connected.** The star the endpoint exchange already runs is
  kept open (`open_star`). When a rank's queue pairs are connected it tells rank 0 and waits; rank 0
  answers everyone once it has heard from everyone (`Star::ready`). The exchange opens once per
  process, so this costs one more round trip on a connection that already existed.

What the meeting measurably does is make the first round an ordinary one. Thirty opens of the smoke
test on two rails with 60 KiB rows, each rank's slowest round, which is its first:

| | a rank's slowest round |
| :-- | --: |
| without the meeting | 0.13-17 ms, mostly 2-10 ms |
| with it | 0.05-1.9 ms |

Whether it also ends the failures cannot be shown directly, because the failure never came on
demand. What can be counted is launches. Thirty two-rail launches of the model have been made with
the meeting in place: fourteen in the comparisons above, and sixteen made only to open the exchange,
serve one request and stop, every one of which returned the same tokens. All thirty served. At the
earlier rate of two in thirteen, thirty clean launches in a row would happen by chance less than one
time in a hundred. That is good evidence that the meeting removed the failure and no evidence of
what the failure was.

## Which rank a slow launch waits for

The note on scatter above could not say which rank a slow launch waits for, because only rank 0
reported. Every rank now prints its pump's figures as it leaves, and among them what each peer's flag
took to arrive after this rank posted, averaged over the run:

```
rdma_exchange rank=2 rounds=128758 ... post_us=0.65 arrival_us=22.10 peer_arrival_us=[14.32, 14.44, 0.00, 16.91]
```

A worker prints it when rank 0 tells it the run has ended, before it releases its weights: that
takes seconds, and whoever stopped rank 0 is usually about to stop the workers too.

Ten launches of one build, a c1 run on each. Nine stepped in 37.84-38.05 ms. One stepped in
38.31 / 38.38 / 38.44, and its four lines are the first picture of a slow launch from every side:

| | time to post, us | what it waited for ranks 0 / 1 / 2 / 3, us |
| :-- | --: | :-- |
| rank 0 | 0.64 | - / 14.1 / 14.0 / **22.2** |
| rank 1 | 0.66 | 13.1 / - / 14.6 / **22.8** |
| rank 2 | 0.67 | 14.5 / 15.1 / - / **23.8** |
| rank 3 | **1.92** | 13.7 / 14.3 / 13.9 / - |

In the other nine launches every rank posts in 0.60-0.68 us, and rank 3's flag is seen 15-17 us
after the post. So in the slow launch one rank's rows reached every peer 7 us late, for the whole
launch, while that rank saw its own peers at the usual time. It was not ringing late: a rank that
rings late sees its peers' flags sooner after its own post, and rank 3 did not. Its way out was
slow, and so was its pump, which took three times as long to hand a round to the device. Seven
microseconds over a step's rounds is the 0.5 ms the step lost.

What it was not:

- **The slower cores.** A Spark has ten cores of capacity 997-1024 and ten of 718-731, and nothing
  pins the pump. A sampler found it on a 718 core for the whole of the benchmark in two of the
  ordinary launches, posting in 0.60 and 0.62 us. Keeping the pump to the fast cores was built
  (`sched_setaffinity` from the capacities the kernel publishes) and changed nothing in three
  alternated launches, so it was not kept.
- **Reclaim or compaction** on that node: its `kswapd` and `kcompactd` clocks moved as on the other
  nodes.
- **The link.** Every node's two devices are up at the same rate, on the same firmware.

What it was is not found. The slow launches so far are one in ten here and three in 27 before, and
the per-rank line is what will name the rank the next time.

## The step's other exchanges

A decode step makes about nine one-shot exchanges besides its per-layer all-reduces: the draft's
seed and each draft token's argmax candidates, the verify rows' embedding and Engram slices, and the
final argmax (the top-p candidates as well, under a sampler). Each rank's part is small, from twelve
bytes to a few KiB, every rank's part goes to every peer, and nothing is summed. They had stayed on
the collective group.

A round's row set already holds every rank's row at `rank * row_bytes`, which is how these
exchanges lay their slots out. So a slot exchange is a round like any other with a copy where the
all-reduce has its fold: the exchange kernel stages this rank's slot and waits for the peers', and
`opt_exchange_collect` copies their slots out of the row set into the caller's buffer. The staging
copy learned to move a row that is not a multiple of sixteen bytes, or not aligned to it. A slot
wider than a round carries (640 KiB a rank) falls through to the group, on every rank alike, and
the bulk chunks' ring gathers were never one-shot exchanges and stay where they were.

One binary serves both arms only by build, so this is the stats build against the same build with
the slot exchanges moved, launches alternated, three c1 samples a launch:

| c1 step, ms | first round | second | third |
| :-- | --: | --: | --: |
| slot exchanges on the group | 37.80 / 37.91 / 37.91 | 37.86 / 37.89 / 37.96 | 37.91 / 37.91 / 37.95 |
| through the exchange | 37.64 / 37.69 / 37.67 | 37.69 / 37.73 / 37.74 | 37.64 / 37.68 / 37.73 |

The same requests made 128,758 rounds before and 142,284 after, which is 9.2 more rounds a step,
and every sample returned the same tokens (`4a3f8049dc2d`). The step is 0.2 ms shorter, 0.5%:
about 20 us for each exchange that moved.

The lanes share these exchanges (a step makes about as many whatever it carries), so the longer
steps gain no more. Two more alternated launches a build, with one c8 and two c16 samples
each:

| step, ms (tokens a second) | c8 | c16 |
| :-- | --: | --: |
| slot exchanges on the group | 92.02, 92.56 (157.5, 156.0) | 137.66, 136.32, 138.58, 138.94 (198.0, 199.4, 195.8, 196.9) |
| through the exchange | 91.30, 91.47 (158.2, 158.2) | 139.28, 137.91, 136.92, 136.98 (197.5, 200.1, 197.9, 203.1) |

That is 1% at c8 and nothing a launch's spread (2% at c16) lets through, with the same tokens from
both arms (`67f0d3df9a51`, `ee2d06705060`). Those launches' c1 samples repeat the table above:
37.82 to 38.09 before, 37.73 to 37.77 after.

## Where it stands, and what would change the answer

The exchange is implemented, unit-tested, exercised on hardware, wired behind a typed switch, and on
by default. Over both rails it makes the decode step 2% shorter than the collective group does at
every concurrency measured, with the group's bits. Against the branch's first, blocking version the
step is 2.5% shorter at c16, 1.3% at c8 and about 0.3% at c1.

What a c1 collective of 60 KiB still costs, and why none of it is the transport's to give back:

| what | a collective | a step of 86 |
| :-- | --: | --: |
| the wire: three peers over two rails | 11.3 us | 1.0 ms |
| the spread between the first and the last rank's doorbell | 10.7 us | 0.9 ms |
| the system fence | 6 us | 0.5 ms |
| the fold's first read of rows the NIC wrote | 5 us | 0.4 ms |

That is 2.8 ms of a 37.9 ms step.

The group's 43 us a collective comes to 3.7 ms, and it is tempting to read the 0.9 ms between the
two sums as the gain. It is not: the group arm pays its own fence, its own skew and its own
non-cached read inside that 43 us, so subtracting the exchange's residual costs from the group's
total does not estimate anything. The table is a cost breakdown of the exchange arm, not a
decomposition of the measured difference - which is 38.6 against 37.9 ms, or 0.7 ms, in the A/B
tables above.

- **The wire** is at the fabric's own figure for these bytes, with both rails busy for the whole of
  it. Fewer bytes or a third path would shorten it; a better handshake would not.
- **The spread** is four ranks finishing the same compute at slightly different times. Whichever is
  last, the others wait for it under any transport.
- **The fence** is what lets the NIC see the device's stores, and the bit-exact gate says it cannot
  be weakened.
- **The read** is paid wherever a row lands that the device did not write, and the group pays it
  too.

One idea is left that was not built. A large row could leave in two shots, so that the wire of one
half overlaps the production or the fold of the other. It needs the producer or the fold split in
two, and it pays only where the wire is bound by throughput.

**Both halves of that precondition were then checked, and the second one fails.** The wire is
throughput-bound: the p50 of a round against the bytes a rank writes is 6.7-6.9 us at 10 and 5 KiB,
14.6-17.1 at 60 and 30, 25.7-28.7 at 160 and 80 (§"The flags"), which is a marginal 0.19 us a KiB
over a fixed ~5 us. But the wire is not *waiting*: taking the wait off the host entirely - the
change that made the exchange non-blocking - is worth 1.9% of a c16 step and 1% at c1, and putting
the wait back loses about 1% rather than the ms a serialised round would cost. So the exchange's
share of a step that is *exposed* is already ~1%, and a two-shot split can only recover a fraction
of that, because it does not remove any bytes: it moves where they overlap. Its ceiling is a
fraction of 1% of a decode step, against a personalised all-to-all with its own ordering proof.

That is the answer for #20's WS3(d), and it retires the idea rather than filing it: the primitive
is the wrong shape for two-shot's scatter, the group's collective is not, and nothing that
reshapes the wire can pay while the wire is already hidden.

**Issue #20's WS3 asks for four things, and three of them are this branch's history.** Its evidence
is from before `e209b675` ("the exchange stops blocking the host and uses both rails"): a host spin
that blocked the launch stream, and the rail counters showing one rail carrying 119 MiB and the
other zero. (a) and (c) are that commit; (a) was then re-measured against putting the wait back on
the host, which loses about 1% at c16 (§"The host does not wait better"). (b) — a write-through
doorbell in place of the fence — is the one thing that *cannot* be done: §"Where it stands" has the
fence at 6 µs a round and the gate says weakening it changes every c1 run's tokens.

(d) is not the reorder the issue describes. It reads the dispatch as calling the exchange before the
two-shot branch and proposes swapping them, but the two-shot sits *after* the exchange's early
return inside the same match arm: it is the fallback for a step the exchange declines, so it is
unreachable whenever the exchange takes the step, and that is by construction rather than by
mistake. It also cannot ride the exchange: the primitive is "write my row to every peer" — one
address per round on every rank — while two-shot's scatter is a personalised all-to-all (rank k's
chunk j goes to rank j alone), and a broadcast cannot express that. Making it work means either a
new all-to-all with its own ordering proof or paying two collective-group collectives (2 x 43 µs)
for one exchange round (~35 µs at c1), which is why the idea above is filed as "needs the producer
or the fold split in two" rather than as a dispatch change.

One thing was tried after that and changed nothing: the order a rank posts a round's sends in.
Posting them peer by peer, with each rank starting from a different peer so that no rank is every
sender's last, gave 37.58 / 37.63 / 37.64 and 37.57 / 37.62 / 37.66 ms a c1 step against the
committed order's 37.55 / 37.59 / 37.65 and 37.58 / 37.64 / 37.68, launches alternated and the same
tokens from both. The spread between the ranks' doorbells is when each finishes its compute, not the
order it posts in.

Three things are not explained, and are listed so that they are not taken for answers:

- why the non-blocking exchange on one rail is 0.2 ms slower at c1 than the blocking one it
  replaced. Where the wait happens is not it;
- what the peer refused in the two first-round failures;
- why one launch in ten has a rank whose rows leave 7 us late for as long as it lives. Which rank
  it is can now be read off any launch.

## Running it

```sh
# The four-rank fabric measurement (addresses are deployment facts; pass them in):
export SPARK_NODES="<rail addr of rank 0..3>"
export SPARK_HOSTS="<ssh target of rank 0..3>"
export SSH_USER=<user>
scripts/probes/run-exchange-4rank.sh 10240 200

# The module without the model, one process per rank: rows staged on the host, the row size
# alternating every round, every word of every row checked.
write-exchange-smoke --rank <r> --world 4 --peers <rail addr of rank 0..3, comma separated> \
  --port 29711 --bytes 61440 --iters 2000 --rails 2

# Serving: the exchange is the default, so a launch needs only its configuration. They must reach
# every rank in the launcher's env list, not just the shell rank 0 runs in: scripts/serve-four-node.sh
# starts the other ranks over ssh, which forwards nothing, and its rank_env() whitelist is what
# carries them. A table that reaches rank 0 alone is the one configuration that can split the group.
SPARK_COMM_PEERS=<addr of rank 0..3, comma separated> \
SPARK_COMM_CONTROL_PORT=29702 \
SPARK_COMM_TIMEOUT_MS=120000 \
  bash ~/sparkflash/launch_best.sh

# One rail instead of both (add to the line above):
SPARK_COMM_RAIL_COUNT=1

# The doorbell deadline, for a device whose queue is deeper than the default allows (default: four
# times SPARK_COMM_TIMEOUT_MS, floored at 60 s and capped at 300 s):
SPARK_COMM_DOORBELL_TIMEOUT_MS=120000

# Back to the collective group:
SPARKFLASH_EXCHANGE=nccl bash ~/sparkflash/launch_best.sh
```

The control port must not collide with the NCCL bootstrap's, and the deadline is generous because the
open happens on the first collective of the first request: a peer that is a few seconds late must not
fail the run. The open costs 5 MB of pinned memory per node.

Every rank prints one `rdma_exchange` line to its log as it leaves: the rounds it made and their
mean row, what its pump took to post a round, and what each peer's flag took to arrive after it. A
launch that steps slowly names its rank there.

## What keeps a wrong run from being a quiet one

The wire carries four things that make a mistake a refusal or a named error rather than a hang or a
wrong row. They exist because each failure mode below was reachable, not because the protocol needed
them:

- **The arm is on the wire.** Each rank's endpoint record publishes which collective it is in
  (`ARM_EXCHANGE` or `ARM_TRANSPORT`), every record in a table must agree on it, and the exchange
  refuses a table where any rank published the other one. Without it, a rank that stands down to the
  NCCL group after the rendezvous is invisible: its peers spin on an arrival flag nobody will write
  and the only symptom is the timeout. The rendezvous itself is the *last* fallible step of the open -
  the slab, every queue pair and the pump thread are live before it, and the caller loads the
  exchange's kernels before it - so a rank that gets past it has every resource it needs. The
  barrier is not atomic and cannot be: rank 0 broadcasts the go message to each peer in turn, so a
  failure part way leaves the earlier peers past it. What is guaranteed is that the failing rank does
  not *stand down*: it reports the error, its streams close so peers still waiting see the end of the
  stream, and `shared` treats an open failure as fatal once the peer configuration exists, so the
  split is a loud error on every rank that reaches it rather than one rank quietly joining the group.

  Its scope is worth stating plainly, because it is narrower than "one rank cannot differ": the check
  only runs on ranks that *join the control plane*. A rank started with `SPARKFLASH_EXCHANGE=nccl`,
  or with no peer table at all, never opens one and publishes nothing, so its peers block in the
  rendezvous for the full control timeout while it blocks in the first NCCL collective - a
  two-sided hang, not a named error. Nothing short of making the NCCL arm announce itself in the
  same control plane can change that. The launchers therefore set the exchange configuration on all
  four ranks or on none: that is a deployment rule, and it is why `scripts/serve-four-node.sh`
  derives `SPARK_COMM_PEERS` itself instead of forwarding whatever the operator's shell exported.
- **The round's row length is on the wire.** An arrival flag says a row arrived; it does not say how
  long it was. Both the write offset and the fold's addressing are derived from the receiver's own row
  size, so two ranks that disagree about a round's row length write and read different bytes of the
  same set - in bounds, with no fault, and the fold silently sums the wrong rows. Each sender
  publishes the row length into a ring keyed by `(peer, seq)` ahead of the flag on the same queue
  pair, and both the waiting kernel and the host's `wait_arrival` compare it against their own before
  the fold reads a byte; a mismatch sets the abort word with a tag that names the peer, and the pump
  turns that into an error that says the ranks disagree about the round rather than about the fabric.
  Keyed by round, not by peer, because a peer is allowed to be a round ahead: one word per peer would
  have been overwritten by the round it is checking.
- **The doorbell has its own deadline.** The interval between a submit and the device ringing the
  doorbell contains every kernel the launch thread queued ahead of the exchange kernel, so it measures
  the device's queue depth and not the fabric's latency. It was first held to the control-plane
  timeout - 5 s by default - and an expiry was fatal for the rest of the run, which turned a deep queue
  on a busy box into a dead server. It now has its own deadline (`SPARK_COMM_DOORBELL_TIMEOUT_MS`),
  a single expiry is retried, and three in a row are reported as a stalled launch with the queue named
  as the cause.
- **Teardown is explicit.** A peer can be midway through a write into this slab when this rank stops,
  so deregistration alone does not make the memory safe to free. Two things have to hold, and they
  are not the same thing: no peer may still be writing *into* this slab, and none of this rank's
  sends may still be in flight. The smoke test ends with one more round both ranks submit and wait
  for, which settles the first; the pump, on stop, drains every rail's outstanding completions to
  zero before it returns, and the join is what makes the registration go after them - that settles
  the second. The serving process still holds the slab for its lifetime and the kernel reclaims it,
  because there is no control channel left to say goodbye on once `Star::ready` has consumed its
  streams. In *serving* the static exchange is never dropped - the process leaves through
  `std::process::exit`, which runs no destructors - so the drain and the join described above are
  the smoke test's path, not the server's. A serving rank that exits while its peers keep serving
  leaves those peers' NIC writes addressed to pages the OS is about to reclaim: the peers see a
  remote access error, not a clean teardown. That is a stated limitation, not a property the code
  has. A failed open **leaks** its slab on purpose: the endpoint table carrying this rank's MR
  address goes out inside the open, before the barrier, so a rank that fails at the barrier can
  have peers past it with a write already addressed to that memory. Freeing there would be a
  use-after-free across the fabric; one leaked 5 MiB pinned allocation in a process that is about
  to serve on the group is the cheaper mistake, and exit reclaims it. A rank that is several rounds
  ahead of the pump makes the pump slow to notice a device-set abort, so `check` reads the abort
  word itself rather than waiting for the pump's next look.
- **The fold and the collect read peer rows with `__ldcg`.** On the exchange path the gather is
  memory this rank's NIC wrote. A plain load may be served from L1, which the NIC does not
  invalidate, and the same row set is re-read every second round. The arithmetic is unchanged -
  same bits, same ascending-rank order, one rounding - only the cache operator is.
  `__ldcg` rules out L1 and no more: it is sufficient only if the GPU's L2 is the coherence point
  for this host-mapped allocation, which is plausible on GB10's NVLink-C2C and is what the
  bit-identical serving runs are consistent with, but the operator PTX documents for memory another
  agent wrote is `.cv` ("consider cached system lines stale"). A probe that rewrites the same slab
  addresses from the NIC every round with a counter, and reads them back with `__ldcg` and with
  `__ldcv` over thousands of rounds, would settle it in an hour; if `__ldcg` holds, that result
  belongs here.
