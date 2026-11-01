# Model optimization strategy

How SparkFlash makes a model fast on GB10, distilled from two arms: DeepSeek-V4.1-Flash at
tensor parallelism 4 over four DGX Sparks (1.06 tok/s at the first device run, 125.5 tok/s on
code today) and Qwen3.8-Flash-Next (NVFP4) on one GB10 (0.20 tok/s decode at the start, about
45 tok/s with DFlash 2, and prefill from under 100 tok/s to about 1,000 tok/s flat from 2k to 36k
tokens). It covers the probes and how to read them, how to write kernels for this part that the
next model can reuse, the hardware's measured limits, the strategy that worked, and the record of
what was tried, what it bought and which probe pointed at it.

Related: [design.md](design.md) (the engine), [performance.md](performance.md) (current numbers)
and [models/qwen38-flash-next-nvfp4.md](models/qwen38-flash-next-nvfp4.md) (the Qwen arm).

## 0. The rules everything else follows

1. **The model's output does not change.** Greedy output is bit-identical before and after every
   optimisation, and identical at every batch size: a row computes exactly as it would alone.
   That is what makes speculative decoding exact (a verify row stands in for a one-token step)
   and what makes every change checkable by a hash diff instead of a tolerance. Speedups that
   change arithmetic (FP8 attention, native FP4 activation MMA, a chunked Gated DeltaNet, moving
   an f32 CUDA-core GEMM to the tensor cores) are out, however small the change. The one licence:
   a **drafter's** arithmetic may be lossy, because the target verifies every draft; quantising
   the drafter changes which tokens are proposed, never which are emitted.
2. **Start from the hardware floor.** Before touching a kernel, tally the bytes a token (or a
   chunk) must read and divide by the bandwidth the memory system actually delivers. The gap
   between that floor and the measured time is the budget, and a stage's GB/s against the best
   observed rate says whether it is waiting on memory, on arithmetic, or on its own shape.
3. **State the expected effect, then measure wall clock.** Write down what a change should save
   before building it, measure end-to-end without the profiler, and compare it with the estimate.
   An estimate that misses by several times means the model of the stage is wrong; replace it
   with a hardware-floor estimate before the next change.
4. **One idea per change, A/B interleaved, texts compared.** Two builds alternate on one idle
   node with the same prompts; the generated texts must match exactly, and the row hashes must
   match for anything touching arithmetic.
5. **Keep neutral results out of the tree, and write down why they were neutral.** The reason a
   plausible idea did nothing is often the most useful fact about the stage.

## 1. Profiling with the injected probes

Every probe is an environment variable, off by default and free when off (a `OnceLock` check),
so the serving binary is the profiling binary.

### 1.1 Stage timing: `SPARKFLASH_QWEN4_PROF`

`DeviceForward::prof_mark(name)` synchronises the stream and charges the host time since the
previous mark to `name`. At the end of each forward the accumulator prints, sorted by cost:

```text
qwen4 prof p4095 rows 4096: 3810.12 ms total
qwen4 prof   experts          837.92 ms     48 x   22.0%
qwen4 prof   gdn_proj         583.35 ms     36 x   15.3%
qwen4 prof   sparse_attn      524.20 ms    384 x   13.8%
qwen4 prof   inject_mlp       175.73 ms     48 x    4.6%
...
```

To profile a new stage, put a mark after its last launch:

```rust
self.gemm(&block.down, self.normed, self.hyper_low, batch, 1, hc)?;
self.prof_mark("hyper_low")?;
```

Reading it:

* **It serialises.** Every mark drains the stream, so overlap disappears: a stage that normally
  hides behind device work (the host PLE gather is meant to overlap the layers before it) shows
  its full cost. Treat a stage's number as an upper bound on what removing it saves.
* **A mark costs about 4.5 us** (one synchronise round trip, see section 3), about 6 ms over the
  ~1,400 marks of a decode token. Benchmark without it.
* **Totals do not compare across runs.** The same binary produced 64.61 and 82.99 ms of profile
  total on two runs, every stage 15-45% apart, while the wall-clock benchmark read 16.1 and
  16.2 tok/s on both. Use the profile for the *shape within one run*; use wall clock across runs.
* **Time is charged to the next mark.** Work between forwards (the drafter's layers and head)
  lands in the next forward's report, which is why `d_layers` and `d_lm_head` appear in target
  forwards. Group reports by row count: a 1-row step, an 8-row verify and a 4,096-row prefill
  chunk are different programs.

A short summariser that averages the reports by row count:

```python
import re, sys
from collections import defaultdict
head = re.compile(r"qwen4 prof p(\d+) rows (\d+): ([\d.]+) ms total")
line = re.compile(r"qwen4 prof   (\S+)\s+([\d.]+) ms")
groups, current = defaultdict(lambda: [0, defaultdict(float), 0.0]), None
for text in open(sys.argv[1], errors="replace"):
    if m := head.search(text):
        current = groups[int(m[2])]; current[0] += 1; current[2] += float(m[3])
    elif (m := line.search(text)) and current is not None:
        current[1][m[1]] += float(m[2])
for rows, (n, stages, total) in sorted(groups.items()):
    print(f"rows {rows}: {n} forwards, {total / n:.2f} ms mean")
    for name, ms in sorted(stages.items(), key=lambda kv: -kv[1]):
        print(f"   {name:<14} {ms / n:7.2f}")
```

The marks are deliberately fine-grained where the cost hides. Splitting the indexer's score,
pick and attention into separate marks (`sparse_score`, `sparse_pick`, `sparse_attn`) is what
showed, in a 6.9 s forward of 4,096 rows, 1.0 s in the score and pick and 2.6 s in the attention,
where one undivided mark had only said the sparse attention layers were large.

### 1.2 Exactness: `SPARKFLASH_QWEN4_PROBE`

`probe_finite(layer, label, buffer, elements)` sits after each interesting buffer of each layer.

| value | what it does |
| :-- | :-- |
| any | checks every probed buffer for non-finite values and names the first 64 |
| `hash` | prints a 64-bit FNV-1a hash of each probed buffer, **one per row, keyed by position** |
| `dump` | min, max, mean and the first values of `after_mlp` and `logits` |

The hash mode is the exactness gate. Run the baseline build and the candidate on the same prompts
(a 1,000- and a 2,300-token prompt give about 409,000 and 957,000 row hashes), then compare the
keys `(position, layer, label)`. Because a verify row's hash is keyed by its position, a
speculative step's rows compare against the one-token steps they stand in for. The first key that
differs, in layer order and then probe order, names the first kernel whose output changed:

```python
import re, sys
pat = re.compile(r"qwen4 hash p(\d+) layer(\d+) (\S+?): ([0-9a-f]{16})")
def load(path):
    order, last = {}, {}
    for line in open(path, errors="replace"):
        if m := pat.search(line):
            key = (int(m[1]), int(m[2]), m[3]); order.setdefault(key, len(order)); last[key] = m[4]
    return order, last
order, a = load(sys.argv[1]); _, b = load(sys.argv[2])
diff = [k for k in order if k in b and a[k] != b[k]]
print("compared", sum(k in b for k in order), "keys,", len(diff), "differ")
for k in sorted(diff, key=lambda k: (k[1], order[k]))[:15]:
    print("differs", k)
```

Hashing copies every probed buffer to the host, so prefill runs at 30-80 tok/s under it; it is a
correctness tool, never a timing one.

### 1.3 The other Qwen switches

| variable | use |
| :-- | :-- |
| `SPARKFLASH_QWEN4_TRACE=1` | synchronise after every launch and print it, so a fault the driver reports late is pinned to one kernel |
| `SPARKFLASH_QWEN4_SPEC_TRACE=1` | each draft's confidence, its rank in the target's row, and whether it was kept: the input for tuning the drafter and the depth policy |
| `SPARKFLASH_QWEN4_PREFILL_PARTS=dense,experts,gdn` | which tensor-core prefill paths a chunk may take: bisects a difference against the token kernels |
| `SPARKFLASH_QWEN4_PREFILL_MMA=0`, `SPARKFLASH_QWEN4_PREFILL_CHUNK=N` | the token-kernel prefill, and the chunk width |

### 1.4 The DeepSeek probes

| variable | what it records |
| :-- | :-- |
| `SPARKFLASH_STAGE_TIMING=1` | a per-kernel ledger: CUDA events around every launch, attributed to a (phase, kernel) scope with calls, total, min and max, plus per-step rows with warm-up steps marked so cold and steady state are not averaged together. Byte-identical run when unset |
| `SPARKFLASH_DEVICE_TRACE=1` (`_EVENTS=N`, a power of two) | the in-kernel trace ring: block 0 thread 0 of each seam kernel writes a 32-byte `%globaltimer` record into host-mapped memory, with a stage code in its payload. No synchronise and no copy, so it times the real, overlapped pipeline |
| `SPARKFLASH_SPEC_TRACE=1`, `SPARKFLASH_BATCH_TRACE=1` | rank 0's per-step speculation and batching decisions |
| `device-chain --profile` | the host timers of the TP=4 host harness |

Two lessons from building them. A switch that changes behaviour on one rank but not the others
must refuse to start: the bootstrap exchanges a word of per-process toggles and rejects a
mismatch, because a rank that silently differs produces wrong answers rather than errors. And
the in-kernel ring is the right tool once kernels overlap (programmatic dependent launch, side
streams): event or synchronise timing measures a pipeline that no longer exists.

### 1.5 Kernel benches next to parity tests

Each rewritten kernel gets a parity test that holds it to the kernel it replaces, bit for bit, at
the released shapes and at edge counts, and an `#[ignore]`d bench in the same file:

```bash
cargo test --release -p sparkflash-model-qwen4-exp --test attn_kernel -- --nocapture
cargo test --release -p sparkflash-model-qwen4-exp --test attn_kernel -- --ignored --nocapture bench
```

The bench gives the per-launch number at each row count (1, 4, 8, 16, 128) the serving paths
use, so a change can be judged before a four-minute server start.

### 1.6 End-to-end protocol

* **Prefill:** no profiler, prompts of 1k, 2k, 4k, 8k, 16k and 32k words, time to first token,
  texts compared per length.
* **Decode:** a fixed set of eight prompts (six chats, two raw completions, 256 tokens each),
  best of two or more runs a prompt, aggregate tok/s, texts compared with the reference run.
* **An idle, dedicated node.** On a laptop with a desktop session the same build measured 40-46
  tok/s run to run (about ±7%); on a dedicated GB10 node two builds whose decode paths barely
  differ measured within 0.5% of each other on every one of the eight prompts.
* **Discard the first request after a start.** It pays first launches and first touches of
  scratch: a 1,073-token prompt measured 503 tok/s as the first request and 1,032 tok/s at
  2,261 tokens right after it.
* **Unified memory needs a guard.** A server holds most of the node's memory. Before starting
  one, wait until `MemAvailable` is back above what it needs (poll, and abort on a timeout
  rather than start), and never build or benchmark while one is up: back-to-back restarts in an
  A/B loop took a 118 GiB machine out of memory.

## 2. Writing fast, reusable kernels for GB10

### 2.1 Shape the grid to the machine

The machine is 48 SMs, each holding up to 1,536 threads. Streaming reads need enough bytes in
flight: a 1 GiB read with 12 blocks of 256 threads measured 104 GB/s, with 24 blocks 185 GB/s,
48 blocks 233 GB/s and 96 blocks 241 GB/s, after which it is flat. Most of the early 78x on the
Qwen arm came from kernels shaped so that almost none of the machine could work on them:

* **One warp per row, 16-byte loads, a shuffle reduction.** A GEMV with a 256-thread block per
  row and a block-wide tree streams at 286 GB/s over 248,320 vocabulary rows, but over a 512-row
  projection it is five loads per thread between eight `__syncthreads` and measured 2.6 GB/s.
* **No serial loops on one thread.** A router top-k scanned 512 experts on thread 0 for each of
  ten slots; ten block-wide arg-max rounds that keep the reference's tie order took it from
  48.09 to 1.41 ms a token. A one-thread heap pick became a 256-bin radix select.
* **All lanes busy.** The hyper injection looped over its four streams by `threadIdx.x`, so 4
  threads of 256 each walked 10,240 columns serially; a warp per stream with 16-byte chunks took
  it from 44 to 2.55 ms a token.
* **Coalesced activations.** Experts read their activation from global memory with adjacent
  lanes 64 columns apart, a 32-way gather per load; staging it in shared memory took the stage
  from 16.13 to 9.91 ms.

### 2.2 Read each byte once per launch, for every consumer

Most later wins share one shape: several consumers of the same bytes, made to share one read.

* Verify rows share each weight read (a tiled GEMV over up to eight rows).
* A prefill chunk shares each expert panel across all the chunk's rows routed to that expert.
* Four query heads of one KV head share each selected key and value (`q4e_attn_group`).
* The indexer pools each compressed block once a forward instead of once a row.
* An expert routed by several rows of a step is read once (DeepSeek's routed MoE deduplicates).

### 2.3 Exactness-preserving techniques

* **Fix every reduction order independently of the batch and the tiling.** A row's
  multiply-adds run in the same order whatever rows share its tile; grouping rows or heads may
  only change which loads are shared.
* **Choose splits per fixed piece, not per launch.** The dense prefill GEMM splits K when a
  matrix has too few tiles for 48 SMs; the split is chosen per 512-row piece, so a 4,096-token
  chunk computes every row exactly as a 512-token chunk did.
* **Forbid contraction where a tree must match.** `__fmul_rn` and `__fadd_rn` keep a
  rewritten reduction from fusing into FMAs the old one did not have.
* **Integer keys that keep the order of floats.** A radix select over the float bits mapped to
  order-preserving unsigned keys (with -0 mapped to +0) keeps exactly the blocks a heap kept,
  ties broken by index.
* **Lookup tables in shared memory, not shuffles.** A shuffle-based E2M1 table faulted on the
  released 640-wide rows: lanes leave a strided loop at different counts, and a full-mask
  `__shfl_sync` over a partial warp is undefined. Every reduced-geometry test passed; test the
  released shapes.
* **Template on the format, keep the old instantiation's code.** The drafter's E4M3 GEMV is the
  BF16 GEMV body templated on the weight format; the BF16 instantiation compiles to what it was.

### 2.4 Resources

* **Registers:** 65,536 per SM. At 1,536 threads that is 42 a thread; a kernel that grows from
  54 to 98 registers halves its occupancy (doubling a hyper kernel's batch did exactly that,
  for no gain). Six attention heads a CTA lost to four on register pressure.
* **Shared memory:** 48 KiB per block without opting in, 99 KiB per block with
  `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` (`Cuda::load_named_shared`), 100 KiB per
  SM. A kernel that needs the opt-in must have a fallback when the reservation is refused.
* **Static shared arrays** above 48 KiB do not compile; prefer two pipeline stages that fit over
  three that need dynamic memory, unless the third stage is measured to pay.

### 2.5 Reuse

* **Shared kernel headers.** The tensor-core tile body (`kernels/opt_mma_common_sm121.cuh`,
  `kernels/opt_mma_bf16_sm121.cuh`), programmatic dependent launch (`kernels/opt_pdl_sm121.cuh`)
  and the trace ring (`kernels/trace_globaltimer_sm121.cuh`) are included by both model packs.
* **Pinned cubins.** `crates/sparkflash-kernels/baseline/kernel_baseline.txt` pins the shared
  kernels' cubins by hash. A change that would alter an existing instantiation's code goes into
  a new helper, so the other model's kernels stay what they were.
* **Shared policy code.** The speculative depth policy (`sparkflash_model::speculation`), the
  finite arg-max and the GEMV bodies are written once and called by both arms.
* **Attribution.** Code or patterns taken from outside projects carry a header attribution and a
  NOTICE entry with a verified licence, from the first commit.

## 3. GB10: measured limits and quirks

Measured on a DGX Spark with a microbenchmark (device attributes, empty launches, synchronise
round trips, and a float4 grid-stride read at working sets from 1 MiB to 4 GiB):

| | |
| :-- | :-- |
| GPU | `sm_121a` (compute 12.1), 48 SMs at up to 2.42 GHz |
| L2 | 24 MiB (18 MiB can be set aside as persisting) |
| Memory | 128 GB LPDDR5X-8533, 256-bit: 273 GB/s on paper, shared with the CPU |
| Shared memory | 48 KiB a block by default, 99 KiB opted in, 100 KiB an SM |
| Registers | 65,536 an SM and a block |
| Threads, blocks | 1,536 threads and 24 blocks an SM |
| Tensor cores | warp-level `mma.sync`: BF16 m16n8k16, block-scaled `kind::mxf4nvf4` and `kind::mxf8f6f4`. The Hopper warpgroup and datacenter-Blackwell `tcgen05` paths do not exist on this part |

**Bandwidth.**

| working set | read rate |
| ---: | ---: |
| 4-16 MiB | 1.0-1.5 TB/s (L2) |
| 20 MiB | 666 GB/s |
| 24 MiB | 577 GB/s |
| 32 MiB and up | 234-241 GB/s (DRAM) |

* **Plan on about 16 MiB of reusable L2, not 24.** Reads stay at L2 speed to 16 MiB and fall
  apart well before the nominal 24.
* **DRAM delivers 235-241 GB/s to a plain streaming kernel and up to about 286 GB/s to a
  well-shaped one** (the `lm_head` GEMV reads 1.27 GB in 4.44 ms). A read-plus-write copy
  reaches 217 GB/s. Judge a decode stage against 267-286 GB/s: DeepSeek's working rule was 60%
  of the 273 GB/s peak as the minimum for a single-row kernel.
* **DRAM is shared.** Prefetching weights during collective waits made the next kernels faster
  and the overlapped send/receive slower (3.87 to 5.00 ms a step) from contention.

**Launches and round trips.**

| | per call |
| :-- | ---: |
| enqueue an empty kernel (host) | 1.9 us |
| empty kernels back to back (GPU) | 2.05 us |
| launch, then synchronise | 4.6 us |
| a 4-byte device-to-host copy and synchronise | 4.5 us pinned, 5.3 us pageable |
| an event record and synchronise | 4.6 us |

* A kernel that moves less than about 1 MiB is launch-bound: a 1 MiB read takes 4.1 us, the
  same as a 4 MiB one.
* **A host round trip inside a step costs the drain of everything queued**, not 4.5 us: the GPU
  idles until the host has decided and launched again. Two scalar reads per layer (a gate and a
  count) were 60 drains a token; moving those decisions onto the device took `shared` from 4.62
  to 3.39 ms and `attn_inject` from 32.65 to 30.29.
* **Dependent launches on an idle GPU are serial.** A GDN replay that ran 36 small dependent
  launches after every verify lost to the snapshots it replaced; replaying only when a step keeps
  fewer rows than it verified won.
* **Programmatic dependent launch pays only when the GPU waits on launches.** On DeepSeek's
  four-rank decode, where a step is short and collectives interleave, the PDL-entry kernels were
  part of taking a step from about 60 to 31 ms. On the Qwen arm the host runs ~1,400 launches
  ahead of the GPU, there is no launch latency to hide, and PDL measured slightly worse
  (`attn_inject` 29.21 to 29.90 ms). Graph capture was neutral on DeepSeek (596.00 against
  595.94 ms a step) for the same reason.

**Unified memory.**

* Host and device memory are the same DRAM: offloading to the CPU saves nothing, and the
  weights, KV cache, scratch, the OS and anything else on the machine share one pool.
* Host-mapped memory is a zero-copy channel in both directions: the trace ring and the PLE
  staging slab use it. Fence reuse of such a slab with an event, not a full synchronise.
* Large host-side tables (the PLE n-gram table, Engram) are read through the page cache, and
  their cost moves with its state: the same PLE gather measured 0.95 and 7.05 ms on two runs.

**Network (TP=4).** A ConnectX-7 reaches the GB10 over two PCIe Gen5 x4 links of about 100 Gb/s
each, so one cable needs both RoCE devices in `NCCL_IB_HCA` to reach 200 Gb/s. A per-layer
exchange is ~30 us, during which DRAM idles unless something is scheduled into the wait.

## 4. Strategy

1. **Tally the bytes first.** From the safetensors headers, by class and dtype: Qwen3.8 reads
   10.9 GB a token, 63% of it the BF16 attention stack, so one token per weight pass floors at
   26 tok/s and the 80-100 tok/s target needs several tokens a pass. That single table decided
   that speculative decoding, not kernel tuning, was the next order of magnitude.
   [`scripts/decode_floor.py`](../scripts/decode_floor.py) does the tally per stage from a
   checkpoint directory and prints each stage's floor beside the profiled mark that times it, so
   the table cannot drift from what is served and the next change starts from the largest gap.
   Run it whenever a step's shape changes: on the Qwen arm it is what closed the question of
   where the remaining decode time is (see the model card's "Where the decode step's floor is"),
   and it is what said three of the largest stages were already at the floor.
2. **Fix the shapes before the arithmetic.** Of the Qwen arm's 78x from 0.20 to 15.9 tok/s, five
   defects did most of it and none was arithmetic: three kernels shaped so the machine could not
   work on them, and two host round trips per layer.
3. **Compare every stage with the best rate the same kind of read achieves.** The experts ran at
   172 GB/s while `lm_head` streamed the same DRAM at 267: the experts were waiting on their own
   instructions (thirteen issue slots to decode one E2M1 weight), not on memory. A table in
   shared memory fixed it.
4. **Make extra rows cheap, then buy rows.** Speculative decoding multiplies whatever a verify
   row costs. Make rows share every weight read, keep each row's arithmetic identical to the
   one-row step, then choose a drafter for acceptance (DFlash 2's block drafts cover 3.9 tokens a
   verify step against the MTP head's 1.7) and verify only as many drafts as pay for themselves,
   from calibrated confidence against a modelled step cost.
5. **Make the drafter cheap with lossy arithmetic.** It is the one place precision is free: E4M3
   drafter projections and head cut the drafter's head from 4.65 to 2.56 ms.
6. **For prefill, widen the chunk until the weights amortise, and keep each row's arithmetic
   pinned.** In a 128-token chunk most of the 512 experts see a row or two, so each read of an
   expert's panel serves almost nothing; in a 4,096-token chunk an expert averages 80 rows and
   is read about three times for them. Splits chosen per fixed piece keep the bits.
7. **Make per-chunk cost flat in context length.** Anything a row recomputes over the whole
   context (pooling every compressed block, a heap over every block score) grows with the prompt;
   do it once a forward, or in parallel, and prefill stops degrading with length.
8. **Fill the waits.** Collective waits leave DRAM idle; prefetch the next weights into L2 on a
   side stream there. Host-side table reads should overlap the device's work on the layers
   before them.
9. **Correctness bugs show up as speed.** Hashing DeepSeek's Engram n-grams over the wrong token
   ids made every lookup fetch unrelated rows; fixing it raised code acceptance from 5.0 to 5.3
   tokens a step. Low acceptance is worth debugging as a correctness question first.
10. **Keep a ranked plan.** A local list of candidates with expected savings, the probe numbers
    behind each and the results of what was tried keeps the work on the largest remaining stage.

## 5. The record: what was tried, what it bought, and the probe that found it

Results are end-to-end wall clock unless the row says otherwise; every kept change was verified
bit-identical (row hashes, parity tests and generated texts).

### Qwen3.8-Flash-Next, decode

| change | probe that pointed at it | result |
| :-- | :-- | :-- |
| GEMV as one warp per row, 16-byte loads, shuffle reduction | `PROF`: 512-row projections at 2.6 GB/s against `lm_head`'s 286 | decode 6.4 to 9.9 tok/s, together with the next two rows |
| router top-k: ten block-wide arg-max rounds instead of a serial scan | `PROF` `router`: 48.09 ms a token, 32% of it | `router` 48.09 to 1.41 ms |
| experts read a staged activation from shared memory | `PROF` `experts`; a 32-way gather in the access pattern | `experts` 16.13 to 9.91 ms |
| hyper injection: a warp per stream instead of 4 threads | `PROF` hyper stages: ~400 us a call for 80 KiB of loads | 44 to 2.55 ms; decode 9.9 to 15.5 tok/s |
| no host readback per layer: kernels read the device scalars | `PROF` `shared`, `attn_inject`; 60 drains a token | `shared` 4.62 to 3.39, `attn_inject` 32.65 to 30.29 ms |
| E2M1 decode through a shared-memory table | `PROF` `experts` at 172 GB/s against `lm_head`'s 267 | decode 17.9 to 19.2 tok/s |
| chunk-wide sparse attention (a warp per slot, no per-key barrier) | prefill falling 22% from a 35- to a 679-token prompt: a cost per selected key, nine barriers each | decode 19.2 to 21.0 tok/s; prefill flat in length |
| wide GEMVs and hyper passes tiled so a verify's rows share each activation read | `PROF` grouped by verify width: those stages grew with every row | DFlash 2 decode 41.73 to 44.71 tok/s |
| GDN: replay the kept rows instead of snapshotting every row | the snapshot cost, 12.3 us a row a layer (3 MiB each) | 42.4 to 44.8 tok/s |
| drafter projections and head in E4M3 (drafts only) | `PROF` `d_lm_head` 4.65 ms, `d_layers` ~0.70 ms a layer | head 2.56 ms, layers ~0.45; 44.8 to 46.5 tok/s |
| attention: four query heads of a KV head per CTA | the attention bench by row count | one row 322 to 148 us, a 16-row verify 1,188 to 279 us |
| one forward over the rows of every concurrent session: row-independent stages shared, caches and recurrence per row | the concurrency A/B, which was flat (1.10x at two sessions) because two streams read the same 10.9 GB twice | aggregate 20.4 to 35.4 to 42.5 tok/s at 1, 2 and 4 sessions (against 29.9/32.8/32.4 for a session a worker) |

### Qwen3.8-Flash-Next, prefill (one GB10, no profiler)

| change | probe that pointed at it | result |
| :-- | :-- | :-- |
| a tensor-core image for chunks of 64 tokens or more | prefill at 96-174 tok/s on the token kernels | 446-527 tok/s |
| 4,096-token chunks, the dense split chosen per 512-row piece | the weight-traffic tally: at 512 tokens the experts alone cap prefill near 2,300 tok/s | 2.2k tokens 636 to 810 tok/s; 36k 399 to 472 |
| new `PROF` marks splitting the indexer | the 4,096-row forward at 6.9 s, with `sparse_*` large but undivided | score + pick 1.0 s, attention 2.6 s |
| the indexer pools each block once a forward; a warp a block for the scores | `sparse_score` | score and pool 37 ms |
| radix-select pick instead of a one-thread heap | `sparse_pick` | the pick 9.5 ms |
| grouped attention (above) | `sparse_attn`; the 128-row bench at 6.67 ms a launch | 2.01 ms a launch; `sparse_attn` 622 ms |
| the three together | | the forward 6.9 to 3.9 s; prefill ~1,000 tok/s from 2k to 36k (36k: 472 to 990) |

### DeepSeek-V4.1-Flash (four DGX Sparks, TP=4)

| change | probe that pointed at it | result |
| :-- | :-- | :-- |
| asynchronous device work on the host path | the per-step timer, byte-identical gate | 1,201.81 to 599.94 ms a step |
| single-row dense re-tile, then an ordered-FMA decode GEMV (no split-K) | per-step timer; GEMV kernel GB/s against peak | 599.94 to 469.81 to 356.39 ms a step |
| the shared expert's f32 path on the device | probe block counters: 0.2068 ms a layer for 35.4 MB, 171 GB/s, 62.7% of peak | 350.39 to 256.17 ms a step |
| opt kernels with programmatic dependent launch, NCCL send/receive reduce | per-step timer | ~60 to ~31 ms a step |
| indexer: one warp per block, stage two once | the measured launch | 36.2x on that launch |
| L2 prefetch of the next weights during each all-reduce | per-kernel times per step with PDL off; DRAM idle for ~30 us a wait | router 0.99 to 0.49 ms, shared gate/up 1.66 to 0.99, `wq_a` 1.43 to 0.63; chat 57.3 to 58.5 tok/s |
| FP4 index scorer, paired chunks, sequence-parallel seams, merged expert pass | cold time to first token on a 200K prompt | 119 to 100 to 87 to 70 to 63.9 s |
| bulk attention at two blocks an SM | cold 32K prompts | 9.2 to 8.2 s |
| only the encoder layers on a prompt's early rows | the layer structure: decoder layers reach back only 2,541 rows | cold 32K 8.2 to 5.1 s; the 209K needle 57.3 to 44.4 s |
| Engram n-grams hashed over compressed ids | acceptance by prompt | code acceptance 5.0 to 5.3 tokens a step |

### Measured and not kept

| idea | result | why |
| :-- | :-- | :-- |
| programmatic dependent launch on the Qwen arm | `attn_inject` 29.21 to 29.90 ms | the host is ~1,400 launches ahead; no latency to hide |
| graph capture on DeepSeek | 596.00 against 595.94 ms a step | launches were not the bottleneck |
| `q4e_hyper_down` split across eight warps a row | 16.23 against 16.36 tok/s | more warps was not the limit |
| a 16-token token-kernel chunk instead of 8 | 65.5 to 65.7 tok/s | 54 to 98 registers, half the occupancy |
| a shuffle-based E2M1 table | faulted on the released 640-wide rows | `__shfl_sync` over a partial warp is undefined |
| GDN replay after every verify | slower on shallow prompts | 36 serial dependent launches on an idle GPU |
| six attention heads a CTA | slower than four | register pressure |
| a fused member verifying its drafts inside the batch | the batch gate at one, two and four members; `PROF` on the steps the pair actually takes | exact, and free for a lone member once the engine drains the answers a verify queues: 14.6 to 38.1 tok/s at one session against 39.3 unbatched. Two members do not measure better (43-63 against 65, one pair at parity and two below): their profiled steps are two and three rows, not sixteen, because queued answers keep them out of phase |
| the speculative depth policy exploring before it can calibrate | `target_verify_steps` 0 with the drafter loaded at 63% acceptance | decode 20.4 to 39.3 tok/s at one session: a policy that verifies nothing never learns that drafts are worth verifying |
| `--concurrency 8` with `--fuse-batch` at a 16k context | the process was killed during start-up with the checkpoint resident | memory, not the batch: the dense half is read once whatever the row count, so a bigger node or a smaller context is what moves the width |
| verifying the whole draft block every step (`DFLASH2_DEPTH=fixed`) | the adaptive policy's own choice, three runs each on one server | 32.0 against 39.4 tok/s: the seventh row is worth less than its 5.7 ms, and the policy's four is the better bet |
| prefills on a thread of their own, with the sequence moving to it | a 2,333-token prompt in front of a decoding stream | the decode continues through the prompt: 8.08 s against 9.27 s if the engine had waited, and 5.99 s alone. It loses the rest to bandwidth contention |
| a prefill worker a member, so prompts do not queue | a 60-token prompt submitted behind a 2,333-token one | 2.06 s against 3.8 s queued behind a single worker and 0.57 s alone; two equal 1,193-token prompts take 2.78 s together against 1.45 each |
| a wider down-pass warp (8 rows) and a narrower one (2), on the routed experts' 640-column down matrix | the expert down pass at 8 rows a warp read 128 GB/s (71.8 us at one token, 347 at eight) and at 2 rows 201 (45.8 / 339.7), against 196 and 179.5 at the released 4 (47.0 / 287.5) | 640 columns are 20 of a warp's 32 chunks, so the pass is short by construction; four rows a warp is the shape that keeps the most of its weight bytes in flight without spilling, and it was already the measured optimum. What is left there is the fixed per-block work (the E2M1 tables, the member staging), not a row count |
| recalibrating the speculative step cost to the measured step | `DFLASH2_STEP_COST` said 59 ms a step and 8.4 ms a verified row; the wall clock says 44.0 and 6.5. The depth the policy then chose on the benchmark prompts moved by at most one draft | the policy compares a draft's chained probability against `rate * row_ms`, and both parts of the model were overpriced by nearly the same factor (1.34 and 1.29), so the threshold was already within 3% of the measured one. A cost model only has to be right in that ratio; the constants stay as they were measured at block depth |
| merging two prompts' chunks into one prefill pass | the prefill profile (a 4,096-row forward: experts 1,048 ms, sparse attention 844, gdn_proj 622) and the weight read already amortized over those rows | not built: it shares bytes a compute-bound pass has already amortized, at the cost of a second per-session tensor-core image and a combined-row layout for it |
| 48-row expert items and two pipeline stages, instead of 32 and three | `PROF` on a 4,225-token prompt, two forwards a build: `experts` 963 and 974 ms against 933 and 934, the 4,096-row forward 4,054 and 4,071 against 4,048 and 4,045; 8,632-token prefill 936.8 against 942.0 tok/s; three prompts' 256 greedy tokens identical, and identical at `PREFILL_CHUNK` 512 and 4096 | exact, but a 35 ms stage (3.6%) and a 16 ms forward (0.4%), inside the drift. An 80-row expert is five m16 tiles at either width, and the stage's ~0.93 s is that MMA/instruction stream; the third item's panel reads overlapped with it rather than adding to it. The tally counted 204 GB, but the kernel was not waiting on the 68 GB between three items and two |
| folding the hyper norm into its readers: a scale kernel writes one f32 a group, and the dense MMA's A load, the inject GEMM and the mix apply `x * scale * gain` where they read, so the normed rows are never written | `PROF` on a 4,225-token prompt, two forwards a build: `hyper_norm`'s 136 ms went away, but `hyper_inject` went 102 to 803 ms and `hyper_low` 137 to 208, so the 4,096-row forward went 4,364 to 4,792 ms; an 8,632-token prefill 855 to 648 tok/s | the readers are GEMVs that re-read their A operand once per output row, so applying the norm on the fly buys one 168 MB write and pays a reconstruction per (row, column, panel), 96 passes a forward. The token GEMM on the inject rows is the worst case: 4 rows against a 10,240-wide A, so almost nothing amortises and it went up eightfold. Exactness was not established either - the comparison harness returned empty `content` on both builds (this model answers in `reasoning_content`), so it compared two empty strings and would have passed anything |
