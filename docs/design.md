# Engine design

How SparkFlash serves DeepSeek-V4.1-Flash on four DGX Sparks, and why it is built this way. The
model itself (attention, mHC, MoE, Engram, DSpark, vision, encoding) is described in
[architecture/](architecture/); this page is about the engine around it. Measured numbers are in
[performance.md](performance.md).

## The hardware it is built for

A DGX Spark is one GB10: a 20-core Grace CPU and a Blackwell GPU (`sm_121a`, 48 SMs, 24 MB L2)
sharing 128 GB of LPDDR5X at roughly 250 GB/s. Four of them hold the model at tensor parallelism
4, linked by ConnectX-7 (RoCE, NCCL).

Two facts shape everything:

1. **Memory bandwidth is the budget.** A decode step reads every active weight once; at ~250
   GB/s per node the cost of a step is bytes read, not FLOPs. The engine is organised around
   reading each weight as few times as possible per step and keeping the reads that remain
   streaming.
2. **Memory is unified.** "Host" and "device" memory are the same DRAM. Offloading to the CPU
   saves nothing, and the weights, KV cache and Engram tables all compete for one ~120 GB pool
   per node.

## Process and data layout

- One `sparkflash serve` process per node. Rank 0 owns HTTP, the tokenizer, the scheduler and the
  control channel; ranks 1-3 run the same device loop and receive each tick's plan from rank 0.
- Weights are converted once (`sparkflash convert`) into one contiguous per-rank file already in
  device layout: dense FP8 matrices pre-tiled for the GEMV kernels, each rank's FP4 expert slices
  stored next to their scales. Boot is a parallel `O_DIRECT` read of that file into one
  allocation, then the kernels load from pre-compiled cubins. No Python and no JIT compilation at
  boot.
- Routed experts are split by intermediate width (576 of 2304 per rank), not by expert, so every
  rank does a quarter of every expert's work and no token waits on a single rank's experts. The
  shared expert is split the same way. Attention heads are split by rank.
- The embedding table is replicated on every rank, so token embedding needs no exchange.
- Engram's n-gram tables live on each node's NVMe and are read with `io_uring` while the layers
  ahead of the Engram layers run; a keep-alive reader holds the drive out of its idle state while
  sequences are running.

## Decode: the rows step

Every running sequence's next tokens are verified together in one **rows step** of up to 16 rows
(the anchor token plus the drafts of each sequence).

- **One weight read per step.** The dense GEMVs read each weight tile once for all rows. The
  routed MoE kernels deduplicate across rows: an expert routed by several rows is read once, so
  extra rows mostly cost only their new experts.
- **Bit-exact batching.** Per-row arithmetic is the same at any batch size: decode GEMVs do not
  use tensor-core MMA and keep a fixed reduction order. Greedy output is identical whether a
  request runs alone or with seven others, and identical to the one-row reference path.
- **Collectives sized for the fabric.** Each layer's all-reduces run on one NCCL communicator over
  RoCE; small exchanges (argmax, embeddings, draft tokens) and the `RoCEnante` one-shot RDMA
  transport (`crates/sparkflash-comm`, `kernels/rocenante_oneshot_sm121.cu`) follow the
  `RoCEnante` one-shot dual-rail RoCE collective architecture originated by
  [`b12x`](https://github.com/local-inference-lab/b12x) (Luke Alonso `@lukealonso`, `@original-el8`,
  and contributors), [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference), and
  [`dgx-spark-networking`](https://github.com/christopherowen/dgx-spark-networking) (Christopher Owen).
  Multi-microsecond launch gaps are removed with programmatic dependent launch and an L2 prefetch
  of the next weights on a side stream.
- **Tick-ahead.** Rank 0 plans tick N+1 while tick N's device work is still running and sends the
  plan to the workers, which find it waiting when they finish.

## Speculative decoding: DSpark

V4.1-Flash ships a DSpark multi-token-prediction head. SparkFlash runs it on the device:

- The drafter proposes `k = 5` tokens per sequence from taps of the target layers, all sequences'
  drafts in one batched pass (its kernels may use tensor cores; drafts need not be bit-exact,
  only the verification is).
- **Adaptive depth.** Each step verifies only as many drafts as pay for themselves: the expected
  accepted tokens (draft confidence, calibrated online against observed acceptances) against a
  cost model of what each extra verified row costs. Every rank computes the same depth from the
  same exchanged inputs, so no extra synchronisation is needed.
- **Sampling that speculation cannot change.** A sampled token is the argmax of logits / T plus
  Gumbel noise drawn from a hash of (seed, position, token): an exact sample of the softmax,
  with no state. The drafter and the verifier use the same noise, so a draft is accepted exactly
  when it equals the target's sample, and speculative and plain decoding emit the same tokens for
  the same seed. Greedy, temperature-only and top-k / top-p requests all decode speculatively.

## Prefill

Short and cached prompts take the exact rows path. Long uncached prompts take the **bulk prefill**:

- **4096-row chunks on the tensor cores.** Dense projections run as block-scaled FP8 x FP4 /
  FP8 MMA GEMMs; the sparse-attention indexer scores keys with FP4 MMA; the routed experts group
  the chunk's rows by expert so each expert's weights are read once per chunk.
- **Three chunks in flight.** Consecutive chunks of a prompt are interleaved layer by layer, so
  one chunk's collectives run on a separate stream underneath the other chunks' compute.
- **Sequence-parallel seams.** The mHC mixing seams of a chunk are split by rows across the
  ranks (reduce-scatter, then all-gather) instead of every rank computing every row after an
  all-reduce.
- **One expert pass for several chunks.** The in-flight chunks share their routed-expert inputs,
  so the routed experts run as one grouped pass over all of them: each expert's FP4 weights are
  read once per three chunks instead of once per chunk. Per-row results are unchanged.
- **The decoder layers only on a prompt's tail.** Layers 21-39 keep nothing of their own but a
  128-row window; they read the compressed and index caches layer 20 owns. A prompt's rows before
  its last 2,541 (the decoder's 19 windows of reach, plus one window) therefore run only layers
  0-20, on both paths. The decoder layers start cold on the tail, and by its last window their
  outputs, the drafter's seeds and the windows decode reads are the ones running every layer
  gives, so the exact path stays bit-identical. A later prompt that forks inside the skipped rows
  re-runs up to 2,541 rows.
- The bulk path's results differ from the one-row path in the last bits (different accumulation
  order); `SPARKFLASH_PREFILL_EXACT=1` forces the exact path for every prompt.

## Serving

- Up to 32 sequences run at once and more wait in a bounded queue. Up to eight decoding
  sequences share one speculative step; past eight, each gets one plain token per step.
- **Prefix reuse.** A request whose prompt extends a previous turn reuses that turn's KV, and
  concurrent requests sharing a prefix prefill it once. Cached turns are scheduled first.
- **Decoders keep running through long prefills.** A long prompt is prefilled in chunks between
  decode ticks, so running streams slow down but never stall behind a 200K-token prompt.
- Context up to 1,048,576 tokens per sequence.
- The HTTP surface (OpenAI chat/completions/responses, Anthropic messages, tokenize/detokenize,
  metrics) is described in [serving.md](serving.md).

## Correctness discipline

- Greedy decode on the exact path is checked bit-for-bit against a stored reference at
  concurrency 1 and 8 before a build is deployed.
- Kernel optimisations keep each row's arithmetic order; where a change cannot (bulk prefill, the
  drafter), it is opt-out and the reference path stays available.
- Collective reductions use a canonical rank order (`0 + 1 + 2 + 3`), so all four ranks hold
  bit-identical results (see [DECISIONS.md](DECISIONS.md) D21 and `b12x`'s `RoCEnante` design).

## Prior art and attributions

SparkFlash's `RoCEnante` one-shot RDMA collective transport, `b12x_*` block-scaled `sm_121a`
kernel naming and micro-tile conventions, bounded `O_DIRECT` / `io_uring` Engram table reader,
pre-converted weight boot path, stateless seeded Gumbel-max sampling, and online cost-model-driven
adaptive speculation draw on prior art from [`b12x`](https://github.com/local-inference-lab/b12x)
(Local Inference Lab / [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x)),
[NVIDIA Flare (`flare-local-inference`)](https://github.com/NVIDIA/flare-local-inference),
[`dgx-spark-networking`](https://github.com/christopherowen/dgx-spark-networking),
[`MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks),
[`knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4),
[`blaesus/dsgx10`](https://github.com/blaesus/dsgx10), and
[`deepseek-ai/deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe). See the root
[NOTICE](../NOTICE) file for complete attribution details.
