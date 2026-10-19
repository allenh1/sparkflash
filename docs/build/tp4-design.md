# TP=4 across the four GB10 nodes: design

**Status: design for review, tp-dev, 2026-10-19. Task-23 (S5). No implementation code and no
GPU work happens until the Lead approves this document.** Decisions requested in section 11.

Scope: one DeepSeek-V4.1-Flash forward pass on four GB10 nodes with tensor parallelism TP=4, one
rank per node, matching the released vLLM reference's TP=4 numerics within the D40 bars. EP is
**not** designed here: it is task-31 and comes after TP=4 works, per the user directive ("You
cannot run this model on a single node. Go for TP=4 first, then experiment with EP on top once
TP=4 works").

## 0. What this document asks for

1. **Approve this design** (board task-23, branch `task/tp4-design`) and merge its MR.
2. **Confirm the memory correction** in section 6.3: the per-rank resident weight set is
   **72.31 GiB**, not ~113 GiB. The ~113 GiB figure is the checkpoint total divided by four
   *with the 188.83 GiB of Engram tables resident*; the reference image's disk patch removes them.
   72.31 GiB is the number every later budget statement must use.
3. **Pick the transport path** (section 4.3): sparkflash-comm (recommended, D29-consistent) or
   NCCL-via-cudarc (matches the reference, but a new external dependency).
4. **Grant the stage-1 window** (section 5.5): one 15-minute Lead-owned window, two nodes
   (.80 and .88), exact commands below.

## 1. Survey: what exists today

The workspace has the arithmetic pieces of TP=4 but none of the plumbing: one device chain, one
process, one GPU, and the four TP ranks emulated by four serial launches on that one GPU.

| crate | what exists today | what TP=4 needs |
|---|---|---|
| `sparkflash-comm` (12,950 lines + 8 integration tests) | `Transport` trait (`all_reduce_bf16`, `all_gather_bf16`), schedule driver `exchange.rs`, canonical deterministic reduce `reduce.rs` (f32 accumulate ascending rank from a `0.0f` seed, one bf16 RNE round, NaN payloads excluded), `LoopbackTransport` + real-socket `TcpTransport` (framing, partial reads, peer death, deadlines), dual-rail `fabric.rs`, RDMA backend `rdma/{verbs,dataplane,abi,qpinfo}.rs` behind feature `rdma`, `rocenante-probe` | cross-node addressing (`TcpTransport` binds `127.0.0.1` today); a GPU-buffer seam; the RDMA dataplane has **never run on cluster hardware** (`docs/build/rdma-backend.md`: real backend, no hardware run, GPU-kernel to dataplane seam is a proposal, section 15) |
| `sparkflash-kv` (12,493 lines) | three-tier KV (hot / memory / O_DIRECT NVMe), paged radix, 4 global-KV source layers (2, 8, 14, 20), 128-slot SWA in all 40 layers, FP4 codec, 890 B/token logical budget, `DeviceKvStore` | topology: the MLA latent has `num_kv_heads=1`, so KV is **replicated** per rank like the reference. Paged device KV is task-21 (S3) |
| `sparkflash-scheduler` (4,915 lines) | collective-backend planning (`CollectiveBackend::{RoCEnante, Nccl}`, the 2 MiB routing rule), prefill/decode scheduling, DSPark draft, gates | no execution; it already has the API shape |
| `sparkflash-device` | `staging.rs` host per-layer weight staging; `chain_device.rs` `CudaLauncher` with 16-buffer mHC sets, `stage_layer`/`restage_layer`; `budget.rs` admission from `MemAvailable` | a sharded **resident** loader and per-rank admission (sections 6.2-6.4) |
| `sparkflash-cli` | `device_chain.rs` single-process driver (one position at a time, per-layer staging, host seams); `attention_device.rs` five B12x MXFP8 GEMMs + format-2 MLA body with **one reused weight buffer** and `upload_weight` per call; `moe_device.rs` router readback, selected-expert staging, **four serial rank** gate/up + weighted SwiGLU + combine launches, shared expert, four serial LM-head rank launches | four processes with one rank each (section 5); resident weights (section 6); one all-reduce per row-parallel seam (section 4); sharded head logits assembly |
| `sparkflash-engine` | f32 host reference path and the arithmetic twins every device bar compares against | unchanged: it remains the source of truth for the single-rank oracle |

Two consequences worth stating up front:

* **The device chain already contains the TP=4 arithmetic layout.** `moe_device.rs` runs the MoE
  as four rank slices sequentially (`rank_slice`, `rank_slice_padded`, `local_table`, four w1/w3
  launches, four weighted-SwiGLU launches, four combine launches per layer), and
  `cert_lm_head_sm121` runs four rank shards of the LM head. Today that is a single-device
  emulation of four ranks; TP=4 gives each rank its own process and node.
* **The single-node chain is therefore the natural numerics oracle**: the same binary keeps a
  `--tp-world-size 1` path that runs the old serial emulation unchanged. TP=4 evidence is
  compared against it, not against a second implementation.

## 2. How the released reference runs TP=4 here (measured)

### 2.1 Launch command (from the reference container's process table)

```sh
vllm serve /root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --served-model-name deepseek-v4.1-flash --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --disable-custom-all-reduce \
  --kv-cache-dtype fp8 --block-size 128 --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
  --max-logprobs 1000 --max-num-seqs 6 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,6,10,12,15,18,20,24,25,30,35,36,40,42,48]}' \
  --nnodes 4 --node-rank 0 --master-addr 192.0.2.80 --master-port 29501
```

One rank per node, one GB10 GPU per node. Relevant negative facts: `--disable-custom-all-reduce`
means the collectives are plain NCCL; sequence parallel is **off** (no `--enable-sequence-parallel`,
so `wo_b.reduce_results` stays `True`, `deepseek_v4_1/nvidia/model.py:207`); no expert parallelism.

### 2.2 Collective environment (measured in the container)

`NCCL_IB_HCA=rocep1s0f1`, `NCCL_SOCKET_IFNAME=enp1s0f1np1`, `NCCL_IB_GID_INDEX=3`,
`NCCL_MAX_NCHANNELS=1`, `NCCL_MIN_NCHANNELS=1`, `NCCL_IGNORE_CPU_AFFINITY=1`,
`GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME=enp1s0f1np1`, `VLLM_HOST_IP=192.0.2.80`,
`VLLM_KV_CACHE_LAYOUT=BLNHC`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the UCX/OMPI pins, and the Engram-disk
variables in 2.4. The transport is **RoCE over the active rail `rocep1s0f1`** (rail 0
`rocep1s0f0` is DOWN at 40 Gb/s) -- the same fabric sparkflash-comm targets.

### 2.3 Measured memory, KV and timings (TP=4 reference)

| quantity | value |
|---|---|
| node memory at engine start | free 114.02 GiB / 121.69 GiB |
| `--gpu-memory-utilization 0.85` | 103.44 GiB allowed |
| consumed by weights + non-torch | **87.07 GiB** |
| peak activation | 1.37 GiB |
| CUDA-graph pool | 0.55 GiB |
| KV cache | **14.99 GiB = 3,682,262 tokens** |
| init engine | 90.02 s |
| weight load (reference image doc) | 37 s with the Engram disk patch, 34+ min wedged without |
| checkpoint | 48 safetensors shards, 510,311,070,460 B on disk, 475.24 GiB of tensors |

### 2.4 The Engram disk rule (binding)

The two Engram tables (`layers.{1,14}.engram.embed.weight`, F8_E4M3 `[384006168,256]` = 91.56 GiB
each, plus `.scale` F8_E8M0 `[384006168,8]` = 2.86 GiB each) are **188.83 GiB** of the checkpoint
and must stay on local NVMe, read per hash row. The reference patch (`dsv41_engram_disk`) replaces
the parameter with a one-row stub the checkpoint loader drops; the runtime reads 6 of the 24 hash
columns per rank. Its findings record 23.60 GiB/rank/layer if resident (47.21 GiB/rank for two
layers) and the 37 s load. We keep the same rule; section 6.3 shows why it is what makes TP=4 fit.

### 2.5 Sharding rules taken from the reference code (verified symbol by symbol)

| reference tensor | module:line | rule |
|---|---|---|
| `attn.fused_wqa_wkv` (wq_a, wkv) | `deepseek_v4_1/attention.py:305-313` | `MergedColumnParallelLinear(..., disable_tp=True)` -> **replicated** |
| `attn.q_norm`, `attn.kv_norm` | `attention.py:313,322` | replicated |
| `attn.wq_b` | `attention.py:314-321` | `ColumnParallelLinear`: `n_heads*head_dim = 32768` -> 8192 rows/rank |
| `attn.wo_a` | `attention.py:324-336` | `ColumnParallelLinear` over `n_groups=8`: 8192 out -> 2048 rows/rank (2 local groups, grouped BMM; input per group 8 heads x 512 = 4096) |
| `attn.wo_b` | `attention.py:337-344` | `RowParallelLinear`: 8192 in -> 2048/rank, **all-reduce** |
| `attn.attn_sink` | `attention.py:297-303` | per-head, 16 padded heads/rank |
| indexer (`wq_b`, `weights_proj`, `wk`, `k_norm`) | `attention.py:1064,1071,1097` | all `ReplicatedLinear` -> **replicated** (8 layers) |
| compressor (`wkv`, `wgate`) | `compressor.py:219-226` | `disable_tp=True` -> **replicated** (4 layers) |
| routed experts (384) | `DeepseekV4MoE`, `deepseek_v4/nvidia/model.py:773+` | TP over the intermediate axis: all 384 experts on every rank, 2304/4 = **576** wide/rank, final all-reduce. `n_local_physical_experts=96` is EPLB bookkeeping, not EP (`nvidia/model.py:971-986`) |
| shared expert | `DeepseekV4MLP`, `deepseek_v4/nvidia/model.py:100-160` | column/row parallel 2304 -> 576/rank, **but block-FP8 padding**: padded to 2560 (5 x 512), 640/rank, trailing rank zero-filled (`_pad_shared_expert_weight`, `nvidia/model.py:824-850`) |
| router `ffn.gate` (+ bias, bias_vl) | `deepseek_v4/nvidia/model.py:879-912` | replicated (ids computed identically on every rank) |
| `embed_tokens`, `lm_head` | `nvidia/model.py:446,1028` | `VocabParallelEmbedding` / `ParallelLMHead`: 129,280 -> **32,320 rows/rank** |
| Engram `embed.weight` | `common/engram.py:703-724,821-836` | hash-head shard: 24 hash cols / TP4 = **6/rank**, row lookup returns `[T,6,256]` zero-padded, then `tensor_model_parallel_all_gather(out, dim=1)` and truncate to 24 |
| Engram `wkv`, `k_weight`, `q_weight` | `common/engram.py:1036` | `ReplicatedLinear` / replicated |
| MTP / DSPark (layers 40, 41, 42) | `nvidia/dspark.py:89-95,250,316-323` | same attention rules; 128 experts TP-sharded; `main_proj` and the markov/dspark heads replicated; `embed_tokens` vocab-parallel |
| vision, `aligner` | `nvidia/model.py:731-733` | **not loaded** by the LLM class |
| KV cache | `num_kv_heads=1` (MLA latent) | **replicated** per rank, no collective |

## 3. Per-tensor sharding map for our TP=4

Sizes are per rank, computed from the released checkpoint headers (`/tmp/tp4ref/tensors.json`,
96,085 tensors) with the rules above. "resident" = persistent device buffer; "disk" = NVMe.

| group | checkpoint tensors | dtype / global shape | per-rank rule | per-rank shape | per-rank GiB | resident? |
|---|---|---|---|---|---|---|
| attention, replicated | `attn.wq_a.weight` | F8_E4M3 [1280,5120] + E8M0 [40,160] | replicated | full | 0.0066 x 40 | device |
| attention, replicated | `attn.wkv.weight` | F8 [512,5120] + [16,160] | replicated | full | 0.0026 x 40 | device |
| attention, column | `attn.wq_b.weight` | F8 [32768,1280] + [1024,40] | output rows /4 | [8192,1280] + [256,40] | 0.0098 x 40 | device |
| attention, column (grouped) | `attn.wo_a.weight` | F8 [8192,4096] + [256,128] | output rows /4 (2 groups) | [2048,4096] + [64,128] | 0.0078 x 40 | device |
| attention, row | `attn.wo_b.weight` | F8 [5120,8192] + [160,256] | input cols /4 | [5120,2048] + [160,64] | 0.0098 x 40, **all-reduce** | device |
| attention sink | `attn.attn_sink` | F32 [64] | per head | [16] | negligible | device |
| attention norms | `q_norm`, `kv_norm` | BF16 | replicated | full | negligible | device |
| indexer (8 layers) | `indexer.{wq_b,wk,k_norm,weights_proj}` | F8/BF16, small | replicated | full | 0.042 total | device |
| compressor (4 layers) | `compressor.{wkv,wgate,norm}` | BF16 | replicated | full | 0.034 total | device |
| routed experts (40 layers x 384) | `ffn.experts.N.{w1,w3}` | I8 (FP4 packed) [2304,2560] + E8M0 [2304,160] | intermediate /4 | [576,2560] + [576,160] | 0.0132 x 384 x 40 | device |
| routed experts | `ffn.experts.N.w2` | I8 [5120,1152] + E8M0 [5120,72] | input /4 | [5120,288] + [5120,18] | 0.0132 x 384 x 40 | device |
| shared expert | `ffn.shared_experts.{w1,w3,w2}` | F8 [2304,.] + grids | padded to 2560, /4 = 640 | [640,5120] etc. | 0.0089 x 40 | device |
| router | `ffn.gate.{weight,bias,bias_vl}` | BF16 [384,5120], F32 | replicated | full | 0.0037 x 40 | device |
| embed / head | `embed.weight`, `head.weight` | BF16 [129280,5120] | vocab rows /4 | [32320,5120] | 0.308 each | device |
| Engram tables (2 layers) | `engram.embed.weight/.scale` | F8 91.56 GiB + E8M0 2.86 GiB | hash heads /4 = 6 | rows of 6 heads | **0 resident**; 23.60 GiB/layer on NVMe | **disk** |
| Engram other | `engram.{wkv,q_weight,k_weight}` | F8 [25600,6144], BF16 | replicated | full | 0.147 total | device |
| MTP / DSPark (3 layers) | attention + 128 experts + `main_proj` + markov/confidence heads | as above | attention column/row; experts /4; `main_proj` + heads replicated | -- | 0.215 total | device |
| mHC / hyper-connections | `hc_{attn,ffn}_{fn,base,scale}` | BF16/F32 | replicated | full | 0.157 total | device |
| final norm | `norm.weight` | BF16 [5120] | replicated | negligible | -- | device |
| vision / aligner | `vision.*`, `aligner.*` | ~1.5 GiB | **not loaded** | -- | 0 | -- |

Total: **72.31 GiB/rank = 77.64 GB/rank**, matching the parity plan's ~79 GB figure and the spec's
76.4 GiB per-node weight budget (§2.1 of `deepseek_v41_flash_4x_dgx_spark_rust_spec.md`).
Categories: routed experts 68.917, attention 1.945, embed+head 0.616, replicated (norms, router,
mHC, MTP) 0.651, shared expert 0.355, Engram non-table 0.293, indexer 0.042, compressor 0.034.

Divergences from the spec, resolved in the reference's favour (the spec table predates the
released image):

* **indexer**: the spec says 8 heads/rank; the released code replicates all indexer projections
  because indexer heads are not attention heads. We replicate.
* **compressor**: the spec implies a shard; the reference disables TP, so every rank holds the
  full 512-wide state. We replicate.
* **KV**: the spec says FP4 global KV; the reference runs `--kv-cache-dtype fp8` with the
  `fp8_ds_mla` packed layout (584 B/token/latent). The dtype is task-21's call; the topology
  (replicated) is fixed here.

## 4. Collectives

### 4.1 Seams (every row-parallel reduction in the forward pass)

| seam | payload (bf16) | T=1 | T=128 | T=4096 | collective |
|---|---|---|---|---|---|
| after `attn.wo_b` | hidden [T,5120] | 10 KiB | 1.25 MiB | 40 MiB | all-reduce |
| after routed-expert combine | hidden [T,5120] | 10 KiB | 1.25 MiB | 40 MiB | all-reduce |
| after shared-expert `w2` | hidden [T,5120] | 10 KiB | 1.25 MiB | 40 MiB | all-reduce |
| embedding (vocab-parallel rows) | hidden [T,5120] | 10 KiB | 1.25 MiB | 40 MiB | all-reduce |
| Engram head assembly | [T,24,256] | 12 KiB | 1.50 MiB | 48 MiB | all-gather (dim 1) |
| LM-head logits assembly | [T,32320] f32/rank | 129 KiB | 15.8 MiB | 505 MiB | all-gather (rank 0 for the dump; argmax needs only a max/argmax exchange) |

At the one-position-at-a-time granularity of the current chain every seam is inside the 2 MiB
RoCEnante one-shot domain; only prefill batches (T > ~200) cross it. Hence the staging: S5.1/S5.2
use one-shot at position granularity, S5.4 owns the >2 MiB ring/chunked path.

Count per position: 40 layers x (1 wo_b + 1 routed + 1 shared) + 1 embedding + 2 Engram
(layers 1, 14) = **123 all-reduces + 2 all-gathers** (plus the head gather at dump positions).

There is **no collective in the attention body**: the MLA latent KV is replicated (section 7),
each rank runs its own 16 heads and 2 output groups. The mHC residual path is replicated
bit-for-bit on every rank (same inputs, same cubins), so no collective there either.

### 4.2 Arithmetic rule (this is what D39/D40 compare)

The reference reduces per-rank bf16 partials with NCCL AllReduce(Sum) on bf16 buffers, which
accumulates in f32 and rounds once to bf16. Our canonical reduce (`crates/sparkflash-comm/src/reduce.rs`)
is exactly that: f32 accumulation in ascending rank order from a `0.0f` seed, one bf16 RNE
rounding, NaN payloads excluded, bit-identical on every rank by construction (proved by
`tests/reduction_bits.rs` and the loopback/TCP transport tests).

The single-device emulation's serial rank chain is **not** native: `moe_combine` decodes the
previous bf16 accumulator, adds its own f32 partial and rounds again -- four roundings instead of
one. That difference (a few bf16 ulps per element) is the declared deviation class of section 8.
Task-4's N1 (native MoE combine) will bring the single-rank path onto the same one-rounding rule;
until then the two paths differ only at this seam, and S5.2 records the measured delta.

### 4.3 Transport: two paths, and the recommendation

**Path T -- sparkflash-comm (recommended).** Reuse the crate's protocol and canonical arithmetic:
S5.1/S5.2 over `TcpTransport` (adding a cluster address table; wire protocol, framing, timeouts
and reduce arithmetic are already tested, including peer death and partial reads), S5.3 replaces
it with the RoCEnante one-shot RDMA dataplane plus the missing GPU-buffer seam. Rationale: D29
already chose RoCEnante, the <=2 MiB one-shot domain covers every per-position seam, the
deterministic reduce is a property no other transport gives us, and the crate is 12,950 lines of
reviewed code. Cost and risk: the RDMA dataplane has never run on real hardware and the GPU-buffer
seam is unwritten -- S5.3 is the schedule risk, gated by a measured latency bar (20 us at 10 KiB
per the spec).

**Path N -- NCCL via cudarc FFI (alternative, not recommended as primary).** Bind `ncclAllReduce`
and `ncclAllGather` and use device buffers directly: exactly what the reference does, no host
staging, no new algorithmic risk, known-good performance. Costs: a new external dependency
(`libnccl` on every node, an FFI surface the workspace does not have -- there is no NCCL binding
anywhere today), collectives become implementation- and topology-dependent (deterministic in
practice on a fixed topology, but not a property we can prove and test like the canonical
reduce), it bypasses the crate D29 chose, and our evidence arithmetic would depend on a
third-party library version. Path N is the right **fallback** if S5.3's first hardware bring-up
fails or misses the bar, and it must then be recorded as a measured-gap decision, never a silent
switch.

Common to both: same seams, same payloads; the numerics class (f32 accumulation, one bf16 round)
does not change as long as NCCL's bf16 sum is f32-accumulated, which S5.3 verifies by pushing the
same payloads through both paths.

## 5. Process and rank orchestration

### 5.1 Topology

```
192.0.2.80 (node0)  rank 0  coordinator + evidence owner
192.0.2.88               rank 1
192.0.2.59               rank 2
192.0.2.68               rank 3
```

Four processes, one per node, one CUDA context each, launched from the Lead-owned window script
over ssh. Rank r binds `comm_addr[r]:base_port`; lower ranks connect upward, the hello frame
carries the rank (the existing `TcpTransport` handshake); the first collective after connection is
the implicit barrier. No MPI, no torchrun, no Ray.

### 5.2 CLI surface (added to `sparkflash device-chain --run-stage`)

```sh
# tp4-resident: the S5.2 loader plan and in-place read (host-only, no driver)
--tensor-index FILE      # header dump {name: {dtype, shape}}; plans without the checkpoint data
--checkpoint DIR         # authoritative safetensors headers; required with --read
--tp-rank 0..3 --tp-world-size 4
--evidence-dir DIR       # tp4-resident-rN.json per rank plus tp4-resident-summary.json
--read                   # read this rank's slices in place (needs --checkpoint)

# tp4-smoke: the S5.1 host stage (no driver opened, no device memory, no token)
--checkpoint DIR          # snapshot dir holding model.safetensors.index.json; --synthetic instead
--tp-rank 0..3            # this process's TP rank
--tp-world-size 4         # 1 keeps the single-rank path (the oracle's configuration)
--comm-backend tcp|lo     # tcp: one process per rank; lo: all ranks in this process; rdma: S5.3
--comm-peers IP,IP,...    # address table, index = rank; rank i listens on (peer i, comm-port + i)
--comm-port 29601         # base port
--layer 0 --position 0    # the layer the attention projections are staged from
--repeats 100             # canonical-reduce repeat count
--comm-timeout SECS       # frame deadline; global default 5 s, a cross-node run raises it
--comm-connect-timeout SECS  # connection-setup deadline; same default and reason
--evidence-dir DIR        # one tp4-smoke-rN.json per rank; lo also writes the summary
--resident                # recorded by the host stage; the device resident loader is S5.2
--cubin-dir DIR           # recorded provenance only; the host stage opens no cubin
--oneshot-preflight       # validate + record the S5.3 device-seam contract (host side only)
```

The tp4-smoke arm is reachable from a build without the `cuda` feature, which is deliberate, and
S5.1 is the proof: it was gated on CPU-only nodes. The arm is a host numerics/transport run, so the
same code must stay testable on a host with no driver; the `cuda` feature adds device arms, it does
not gate the host ones. Keep that property. S5.2 adds the resident device loader flags, the logits
assembly and the chain arms.

Rank 0 additionally owns the prompt fixture, the token-id/position broadcast, the logits assembly
(`[T,129280]` f32 from four `[T,32320]` slices via all-gather at dump positions) and writing the
existing dump format, so the D40 harness (`scripts/oracle_native.py`, task-14/15) reads TP=4 and
single-rank artifacts identically.

### 5.3 Failure handling

* Every collective frame has a deadline (`TcpConfig::timeout`, 5 s; the RDMA path has its own
  completion timeout). A silent peer is a typed `CommError::Timeout`, a dead peer
  `CommError::PeerClosed`; both abort the run. No retry inside a run.
* All four ranks write their own evidence file before exiting; rank 0 writes the combined one. A
  failed run leaves per-rank files naming the failing seam -- the artifact the next window reads.
* Startup: any rank that cannot load its shard or admit its memory exits non-zero, and rank 0
  tears the others down by closing its listener (peers see `PeerClosed` on the barrier).
* No silent single-rank fallback: a `--tp-world-size 4` run either completes as TP=4 or fails.
* The chain is deterministic per position, so a failed window is repeated, never resumed.

### 5.4 Launch, staging and window protocol

Build and staging happen **outside** the window (`scripts/build-aarch64.sh`, then one scp per node
into the evidence bin directory); the window does not pay for a cross-build. The window is
Lead-owned per `docs/build/window-runbook.md`: the Lead stops the vLLM server, we run, the Lead
restarts it and runs the `/v1/models` health check. I never stop or start the server.

### 5.5 The stage-1 window request (S5.1, task-33)

Two nodes, 15 minutes asked (expected run ~3 minutes including load), any quiet period:

```sh
# staged before the window (no GPU):
scripts/build-aarch64.sh
scp target/aarch64-unknown-linux-gnu/release/sparkflash \
    allenh1@192.0.2.80:/home/user/phase2-evidence/window-<W>/bin/sparkflash_tp4s1_<sha>_aarch64
scp target/aarch64-unknown-linux-gnu/release/sparkflash \
    allenh1@192.0.2.88:/home/user/phase2-evidence/window-<W>/bin/sparkflash_tp4s1_<sha>_aarch64
# in the window (rank 0 on .80 and rank 1 on .88, background, then wait):
SNAP=/home/user/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
ssh allenh1@192.0.2.80 '/home/user/phase2-evidence/window-<W>/bin/sparkflash_tp4s1_<sha>_aarch64 \
  device-chain --run-stage tp4-smoke \
  --checkpoint '"$SNAP"' \
  --tp-rank 0 --tp-world-size 2 --comm-backend tcp \
  --comm-peers 192.0.2.80,192.0.2.88 --comm-port 29601 \
  --layer 0 --position 0 --repeats 100 \
  --comm-timeout 120 --comm-connect-timeout 120 \
  --evidence-dir /home/user/phase2-evidence/window-<W>/tp4-s1/r0'
ssh allenh1@192.0.2.88 '...same binary and $SNAP... --tp-rank 1 --comm-peers ... --evidence-dir .../r1'
```

The layer's five attention projections plus their scale grids are ~130 MB, so the load is seconds
and the whole run fits comfortably in the 15 minutes. The two processes are in the same collective
from their first frame: rank 1 dials rank 0 (lower rank connects upward), the hello frame carries
the rank, and the first all-reduce is the implicit barrier - a rank that cannot load or connect
exits non-zero instead of waiting.

`--comm-backend lo` on one host runs the same arm with all ranks in one process
and is the pre-window rehearsal (it writes the per-rank evidence files plus the summary); it needs
no window and no GPU, so it is part of the host gate too.

Why two ranks and one layer: it exercises every new piece (cross-node transport, bootstrap,
sharded load of real tensors, the column-parallel byte-identity check, one row-parallel
all-reduce) without paying for the 68.9 GiB expert load. Pass criteria are in section 9 (S5.1).

## 6. Resident weight loading per rank

### 6.1 What "resident" means and how kernels get the pointers

The kernels are already pointer-driven: `attention_device.rs` allocates one `Buffers` set (`w`,
`w_scales`, activation, output; `MAX_WEIGHT_BYTES` = wo_b's 41.9 MB) and every `dense()` call
re-reads the checkpoint tensor, re-tiles it into the B12x layout (`read_packed` /
`b12x_weight_index`) and `upload_weight` copies it into the same two buffers. The mHC launcher
restages its 16 small buffers per layer (`copy_layer_tensors` / `restage_layer`), and
`moe_device.rs` stages only the distinct experts the router selected per layer per position.

Resident mode changes only *where the pointers point*:

1. At rank bootstrap, allocate persistent device arenas (one per category, 512-byte aligned
   suballocations; no per-layer `cudaMalloc` churn): attention 1.54 GiB, routed experts
   68.92 GiB, shared expert 0.36 GiB, embed+head 0.62 GiB, replicated 0.87 GiB.
2. Load each tensor **once** from its safetensors shard, transform it into the exact bytes the
   kernel expects (B12x tile order for the MXFP8 attention projections; checkpoint order is
   already correct for the FP4 MoE tables; BF16/F32 as-is), and copy it into its slot.
3. Launch sites pass the resident `Deviceptr` (per layer, per projection, per expert, per head
   shard) instead of the reused staging buffer. **No kernel changes are required**: kernels take
   pointers, and the tiled layout is a load-time transform.

The per-position H2D then drops from 5.066 GB (the cumulative attention weight set) to the
activation operands only (a few hundred KiB per layer), and the b68-measured 89.45 ms/position of
attention staging leaves the hot loop. `AttentionDeviceStats.staged_bytes` and
`MoeDeviceStats.staged_bytes` become load-time counters reported once in the evidence file.

### 6.2 Loader mechanics

* Parse `model.safetensors.index.json` (7.4 MB) plus each shard header; build the
  (tensor -> file, dtype, shape, byte offset) table and a static **ShardingPlan** that maps each
  tensor to (TP axis, rule, layout transform) as in section 3. Startup verifies the plan against
  the headers (shape and scale-grid divisibility) before touching the GPU, so a wrong plan is a
  load-time error, not a corrupt forward. The check runs on the dev host with no GPU and is part
  of the host tests; `--evidence-dir` records the verified plan hash.
* Slice reads use `pread` at computed byte offsets, never a whole-file mmap (the checkpoint is
  475 GiB; only this rank's 72.31 GiB may be read). Column-parallel weights are row-contiguous
  slices (one range). Row-parallel weights (`wo_b`, `w2`) need a strided slice whose per-row bytes
  are contiguous but whose rows are 4x apart. `shard.rs` reads exactly the per-rank segments (one
  range per row), so a rank reads only its 72.31 GiB and not the ~134 GiB a whole-row read would
  cost; the price is one seek per row, which the S5.1 window measures on `wo_b` (5,120 ranges) and
  S5.2 measures on the expert tables (1.9M ranges per layer over 384 experts) where it may become
  the load-time problem. Decision (Lead, 2026-10-19): read in place first and measure; a one-time
  `pack-tp4` repack onto each node's NVMe is built only if the measurement shows the seek path
  dominating load time -- open decision 11.3. It is not built now.
* Page cache: reads go through the page cache today and the checkpoint does not fit, so the
  loader issues `posix_fadvise(DONTNEED)` per finished shard region (or O_DIRECT) so host page
  cache cannot compete with the 121.69 GiB UMA the device allocations share.
* Order: small replicated tensors and embed/head first, then attention layer by layer, then the
  expert tables (largest last), freeing each host buffer after upload; peak host staging buffer
  capped at 1 GiB. Target <=60 s/rank (reference 37 s); measured in S5.2.

### 6.3 Memory budget per rank (measured, not estimated)

| item | GiB/rank | source |
|---|---|---|
| resident weights | **72.31** | computed from the released checkpoint headers + the reference rules (section 3) |
| KV cache (replicated latent) | 14.99 at 3.68M tokens | reference's measured KV line; our budget is sized by max context (section 7) |
| activation workspace | ~1.4 | reference peak activation 1.37 |
| CUDA graphs / cubins / mHC buffers | ~0.7 | reference CUDAGraph 0.55; our 16 mHC buffers x 40 layers |
| Rust context, driver, collectives | ~1.0 | leaner than torch + NCCL (reference non-torch ~14.8) |
| **total** | **~90.4** | **against 121.69 GiB node, 114.02 GiB free at reference startup** |

Reconciliation of the ~113 GiB number (this decides whether TP=4 fits at all):

* 475.24 GiB / 4 = 118.81 GiB per rank is only reachable **with the 188.83 GiB of Engram tables
  resident** (47.21 GiB/rank), which is exactly the configuration that wedged the reference for
  34+ minutes and forced the disk patch. 72.31 + 47.21 = 119.52 GiB/rank against a 121.69 GiB
  node leaves no room for KV or activations.
* With the disk rule the resident set is 72.31 GiB/rank: 34.5 GiB of headroom. The reference
  independently confirms the scale: it consumes 87.07 GiB of weights + non-torch (72.31 weights +
  ~14.8 torch/NCCL context), then 14.99 GiB of KV, inside its 103.44 GiB budget.
* Our binary has no torch/NCCL context to pay for, so admission (`budget.rs`, now given the
  resident plan instead of a per-position estimate) should pass with ~20 GiB to spare. It runs
  before the first allocation and refuses rather than OOM-killing mid-load.

### 6.4 What changes in the code (stage-2 inventory, not written yet)

| file | change |
|---|---|
| `crates/sparkflash-device/src/shard.rs` (new) | `ShardingPlan`, header/index parsing, `pread` slice reader, plan verification; pure host logic, unit-tested without GPU |
| `crates/sparkflash-device/src/chain_device.rs` | `CudaLauncher` gains resident-buffer allocation/reporting; the per-layer mHC restage stays (small) |
| `crates/sparkflash-device/src/budget.rs` | resident-plan admission (projected bytes + KV + workspace vs `MemAvailable` and the 121.69 GiB node budget); refuses before loading |
| `crates/sparkflash-device/src/staging.rs` | **unchanged**: the per-layer staging path stays for the single-rank oracle (`--tp-world-size 1`) |
| `crates/sparkflash-cli/src/device_chain.rs` | the two `LayerStaging::from_source` / `LayerPostStaging::from_source` calls at lines 721/727 leave the position loop; the provider reads the resident bundle; new `--tp-*` args, evidence file, cross-rank logits dump |
| `crates/sparkflash-cli/src/attention_device.rs` | `Buffers` (one reused slot) becomes `ResidentAttention` (per-layer pointers per projection); `upload_weight` leaves the hot path; `read_packed`/`b12x_weight_index` move to load time; `open_resident` replaces the staging-buffer path; `n_local_heads=16`, `n_local_groups=2`, wo_b inner dim 2048; `staging_us` per position becomes activation-only |
| `crates/sparkflash-cli/src/moe_device.rs` | selected-expert staging becomes `MoeResidentExperts` (per layer: w1/w3 `[384,576,2560]` I8, w2 `[384,5120,288]` I8 + E8M0 grids); `rank_slice`/`rank_slice_padded` leave the position path; `local_table` maps a global id to the same resident row; four serial combine launches become one launch + one all-reduce whose twin is the canonical reduce; shared expert runs 640-wide with the trailing-rank zero pad; `cert_lm_head_sm121` becomes one launch per rank + gather |
| `crates/sparkflash-comm/src/tcp.rs` | `TcpConfig` gains an address table (bind + peers); `loopback4` stays for tests; new tests bind a non-loopback address |

The single-rank oracle keeps `--tp-world-size 1` and the old code path, so S5.2's comparison is
same-binary, same-cubins, same stage bars.

## 7. KV cache per rank

The MLA latent has `num_kv_heads=1`: one 512-value latent (+64 RoPE dims) per token per source
layer, shared by all 64 heads. TP therefore **replicates** the cache (the reference does; there is
no head axis to shard). Consequences:

* Every rank stores its own copy: no KV collective, no cross-rank KV traffic, no KV sharding bugs.
* Budget: 14.99 GiB/rank at 3.68M tokens in the reference (`fp8_ds_mla`, 584 B/token/latent, block
  128); our crate's logical 890 B/token plan must be reconciled with the `--kv-cache-dtype fp8`
  decision inside task-21, but the topology is fixed here: each rank sizes its own store from the
  section 6.3 budget and the admitted max context.
* SWA: the 128-slot ring in all 40 layers stays per rank, same replication rule.
* S5.1/S5.2 use the existing host-side KV building blocks per rank; the paged format-2 device
  store arrives with task-21 (S3) and is wired in S5.4.
* D40 artifacts do not compare KV bytes (a cache, not an output), but the attention outputs at
  every layer are stage bars, which covers KV misuse.

## 8. Numerical acceptance: the single-node chain remains the oracle

Requirement: **TP=4 output must match the single-node device chain within the D40 metrics**, with
the single-node stage bars still enforced.

**The oracle's validity is conditional on the reduce rule, and that condition is part of the
acceptance, not a footnote.** The single-rank chain is the numerics oracle *only while the TP
seam's arithmetic is the canonical one-rounding rule*: f32 accumulation in ascending rank order
from `0.0`, exactly one bf16 round-to-nearest-even, then the routed epilogue. If a TP seam applies
a different rule (for example the serial emulation's repeated bf16 decode/add/round), the two paths
stop being the same function and a match is no longer evidence of anything. Task-4's f32
`slot_acc` accumulation plus routed epilogue is the single-rank form of the rule and must stay
consistent with the TP=4 collective; a conflict is a design defect to raise with the Lead, not a
bar to loosen. Consequence for **task-34's acceptance**: it must include the L1 and L2 twins
(collective vs `canonical_all_reduce_into`, and the TP stage outputs vs the single-rank twin from
the **same binary**) run as part of the TP=4 evidence, not a comparison of TP=4 output tokens
against a released-vLLM artifact - D39 forbids token equality as an acceptance criterion, and a
vLLM-token comparison would not test the reduce rule at all.

Three levels:

**L1 -- collective twin (bit-exact, 0 mismatches).** At every seam, the transport's all-reduce
over the four bf16 rank payloads must equal `canonical_all_reduce_into` on the same bytes, bit for
bit, on every rank, in every repetition (`tests/reduction_bits.rs` and `loopback_transport.rs`
already prove the arithmetic; S5.1 adds the device-path instance for the wo_b seam and S5.2 for all
123 seams). Repeating the same payload 100x must be bit-identical; the recorded contract is
"same bytes in, same bytes out, every time".

**L2 -- per-tensor stage bars.** The existing device-chain bars compare each layer's projections,
attention body, MoE, shared expert and head against their twins with zero-mismatch expectations.
Under TP=4:

* *column-parallel* outputs (`wq_b`, `wo_a`, expert `w1/w3`, shared `w1/w3`) are per-row dot
  products over an unchanged input axis: rank r's rows must be **bit-identical** to the same rows
  of the single-rank twin. Bar: 0 mismatches.
* *replicated* tensors (`wq_a`, `wkv`, norms, indexer, compressor, router, mHC, Engram `wkv`) must
  be bit-identical across ranks and to the twin. Bar: 0 mismatches.
* *row-parallel* seams (`wo_b`, `w2` of routed + shared expert, embedding, LM head) sum across
  ranks in a different association order than the single-rank 8192-wide dot: not bit-identical by
  construction. The bar is declared, not "small": max abs delta <= 2 bf16 ulp of the
  max-magnitude element at that seam, median <= 1 ulp, measured over 40 layers x >=8 positions.
  The serial emulation's 4-rounding chain is the upper bound of the deviation, and the TP=4
  one-rounding path must be *closer* to the host f32 twin than the emulation is. If it is not,
  that is a wiring defect, not FP spread.
  The evidence also carries a **diagnostic, never a gate**: the same deltas normalised by each
  element's *own* bf16 ulp (max, median and the fraction of elements within 1 ulp of their own
  value). A cancellation-heavy seam shows a low fraction there while every absolute error is a
  fraction of the seam's step; it makes the seam visible without gating on its own rounding noise
  (Lead decision, 2026-10-19).

**L3 -- end-to-end, D40 metrics.** Same prompt suite and positions: top-1 agreement, median
top-20 recall, per-position KL, median/p95 KL, max abs delta logp. Provisional bars: top1 >= 95 %,
top20 recall >= 90 %, median KL <= 0.02 nat, p95 KL <= 0.10 nat, max abs delta logp <= 1.0 nat and
none > 2.0 (D40, on branch `docs/d40-bars`, to be frozen as D41 at >=3x the measured oracle
self-consistency floor before the first TP=4 evidence run is judged). Artifact: the same
JSONL/`.bin` logits dump the harness already reads, rank 0 assembling four 32,320-wide slices into
the 129,280-wide f32 row; token equality is never compared (D39).

We must not loosen a bar because TP=4 is new. A bar change is a new decision record.

## 9. Staged plan (board tasks created with this design)

**S5.1 -- task-33: cluster transport + bootstrap + sharded resident loader + 2-rank smoke.**
Transport address table, `--tp-*` CLI, `shard.rs` plan/loader, one layer's five attention
projections resident on two ranks, one all-reduce. Gate: host tests (plan verified against the
real headers without GPU; two-process TCP test; twin bit-exactness), then the section-5.5
15-minute window: (a) 0 mismatches on column-parallel/replicated outputs, (b) 0 mismatches
all-reduce vs canonical twin, repeated 100x bit-identical, (c) wo_b delta within the L2 bar, (d)
evidence JSON from both ranks. Expected load: one layer ~130 MB, seconds.

*Status (host implementation landed with this design, MR !120).* `sparkflash-comm` grew the
cluster address table (`TcpConfig::with_addresses`; a partial table is refused before any socket
opens, a wildcard bind is allowed, and the hello handshake is unchanged). `sparkflash-device`
grew `shard.rs`: the plan (`Replicated` / `Column` / `Row`) is verified against the shard
headers before any tensor bytes are read, and the slice reader issues one contiguous range for a
row block and one range per row for a column block through `SafeTensorsSource::read_ranges` - so
a rank reads exactly its own bytes, never the 4x whole-row amplification (that remains the
reference's behaviour, and the measurement that would justify `pack-tp4` is now a bytes-read
counter rather than a 4x floor). `sparkflash-cli` grew the `--run-stage tp4-smoke` arm (host only,
reachable without the `cuda` feature): it reads the layer's five attention projections plus scale
grids, byte-checks every slice against the same rows/segments of the full tensor, runs the wo_b
row-parallel seam through the transport, compares the result to `canonical_all_reduce_into` for
100 repeats, reports the seam's ulp deltas against the single-rank f32 twin, and writes one
evidence JSON per rank. The in-process `lo` backend runs all ranks in one process (useful on a
host); `tcp` runs one process per rank; `rdma` is refused until S5.3.

*Status: the S5.1 gate is met.* The gate run was made on the released snapshot's layer 0 with rank
0 on node0 (.80) and rank 1 on node1 (.88), binary from commit c3d9e4e, evidence in
`/home/user/phase2-evidence/window-2026-10-18/tp4-s1/{r0,r1}` plus a summary compiled from the
two files (tcp mode has no shared filesystem, so the binary cannot write a summary itself). Both
ranks, one process per node, `--comm-timeout 120 --comm-connect-timeout 120`:

| check | result |
|---|---|
| plan verified against the real headers, same plan hash on both ranks | 10 tensors, fnv `0xdf31e75f2d12fa1b` |
| byte-identity of every slice against the full tensor | 0 mismatched segments on all 10 tensors, both ranks |
| bytes a rank read for its slices | 67,961,600 B (64.8 MiB) = exactly its own shard, no amplification, in 5,288 `pread` ranges (wo_b: 5,120 ranges x 4 KiB; column blocks: 1 range) |
| slice read time / full-tensor twin read time | 37.9-105.1 ms / 236-297 ms (the 4x read is the check, run once) |
| L1: all-reduce vs `canonical_all_reduce_into` | 0 mismatches; 100/100 repeats bit-identical; payload 5,120 bf16 = 10,240 B |
| collective latency (median / max, 100 repeats) | 126-129 us / 1.24-165 ms (max is first-exchange + ssh/OS jitter on an otherwise idle host pair) |
| connection setup | 3.6 ms (rank 0) / 37.0 ms (rank 1) |
| L2: wo_b seam vs the single-rank f32 twin | max 1 ulp, median 0 ulp (bar 2 / 1); max abs delta 0.015625 = 1 ulp of the seam's 2.0 peak |
| L2 diagnostic (per element's own ulp) | max 1,889 ulp, median 0, 91.72 % of elements within 1 own-ulp - a cancellation-heavy seam made visible, not gated |
| whole run after both processes were up | 1 s |

What the two failed attempts before it taught (kept because it is a transport-level trap): launching
a rank with `ssh host 'nohup ... &'` without `-n` leaves the session open until the *child* exits
(the child inherits the ssh channel), so the second rank started after the first one's connect
deadline and the run failed with `Connection refused` even though the transport was correct. The
working pattern is either `ssh -n` with a detached child or, simpler, both ranks as local ssh
children in the same shell (`ssh host "binary ..." & ssh host2 "binary ..." & wait`). The 5 s
default deadlines are too tight for a cross-node pair whose ranks also load ~130 MB before the first
collective, so the arm now takes `--comm-timeout` / `--comm-connect-timeout` (defaults unchanged;
the window command in section 5.5 sets 120 s).

**S5.2 -- task-34: full resident chain on 4 ranks, D40 comparison.** Depends on S5.1 and task-6
(N3 native `wo_a`; the shard layout depends on its final form). All 40 layers + 3 MTP layers,
experts resident, embed/head sharded, Engram seam (disk + all-gather), logits assembly. Gate:
L1/L2/L3 in one window (90 minutes asked; load is the long pole), memory admission, and per
position H2D measured down from 5.066 GB to activation-only. **The L1 and L2 twins are part of the
acceptance in this gate, not a side artifact**: they must be produced from the same TP=4 binary in
the same run (collective vs `canonical_all_reduce_into`, and the TP stage outputs vs the
single-rank twin), because the single-rank chain is the oracle only while the canonical
one-rounding reduce rule matches (section 8). No comparison of TP=4 output tokens against
released-vLLM output is part of this gate (D39). This is the "TP=4 works" gate the user asked
for; EP (task-31) unblocks after it.

*Status (host side landed 2026-10-19, task-39; the window is not booked).* The resident loader's
**plan** now exists: `sparkflash_device::resident` classifies every checkpoint name the LLM loads
(attention, routed/shared experts, router, norms, indexer/compressor, Engram, mHC, MTP/DSPark heads,
embedding/head) into the design groups with the section-3 rule, and `sparkflash device-chain
--run-stage tp4-resident` plans a rank from headers alone (a tensor-index dump, or the
checkpoint's own safetensors headers) and, with `--read`, reads the rank's slices in place through
the S5.1 `ShardPlan` range reader. Classification is exhaustive: an unknown name is a refusal, not
a silent skip; only `vision.*`/`aligner.*` are deliberately not loaded.

Planned from the released index (95,822 tensors), rank 0: **72.308 GiB resident** (design estimate
72.31 GiB, delta -0.002 GiB) plus the Engram tables as **188.833 GiB disk** (exactly the design's
figure), 95,818 resident + 4 disk tensors, plan hash `0x9a59035fc40776c8`. So the section-6.3
budget is reproduced from the headers rather than taken on trust, and a 4-rank run needs no
repacked copy: each rank reads its own quarter where the checkpoint lies.

**The plan also measures the one risk that decides read-in-place vs `pack-tp4`**: the row-rule
slices issue **161,736,798 ranges at a mean of 480 B** (the routed-expert `w2` rows are 288 B and
their scale rows 18 B per range), while the column/replicated groups are few large ranges. S5.1
measured ~20 us per range on the same reader, so the syscall count alone would be ~54 minutes per
rank before any device work - a host-computable prediction, not a measurement, but a strong one.
The window measures the alternatives before S5.2's four-rank gate: (i) batched/vectored pread, (ii)
read each row-parallel tensor whole (4x bytes, one range per tensor) and slice on the host, or (iii)
`pack-tp4` (one repack, four contiguous per-rank copies). None is chosen here and none is claimed
measured; the decision and its numbers belong in the S5.2 window evidence.

The stage writes `tp4-resident-r<rank>.json` per rank and a `tp4-resident-summary.json` with
`device_execution: false`, `token_produced: false` and `comm_rdma_shim` in every schema, and names
the measurements it still needs: device allocation of the planned bytes, admission against
`MemAvailable` before the load, in-place read throughput/page-cache pressure on a node, and the
four-rank load in the S5.2 window.

**S5.3 -- task-35: device-buffer collectives and the RDMA first hardware run.** Depends on S5.1;
transport decision from 11.1. The three existing hardware gates, `rocenante-probe` on the active
rail, the 20 us / 10 KiB one-shot bar, repeat determinism, then the GPU-buffer seam (no
per-collective DtoH/HtoD). If it fails, the measured gap and the NCCL fallback plan are recorded;
no silent transport switch.

*Status (host side landed 2026-10-19; the window is not booked).* The seam's **contract** now
exists and is host-tested in `crates/sparkflash-comm/src/oneshot.rs`: the pinned slot geometry
(32 B arrival words + four 2 MiB rows = 8,388,640 B, 8-byte aligned), the
`Exchange`/`DeviceBuffer`/`SlotRegistration`/`Dispatch` descriptors (a payload over a row, a null
or misaligned pointer, a slot of the wrong size and a sequence that does not strictly increase are
all refused before any device work), the receive-side proxy state machine of `rdma-backend.md`
15.2(b) (`on_doorbell`, then `on_frame` publishes a peer's arrival exactly once and only after
every rail; a frame before the doorbell or outside the dispatch is a protocol error), and
`Transport::all_reduce_bf16_device`, whose default refuses - a host backend answers
`CommError::Unsupported` rather than quietly running the host-slice collective.
`crates/sparkflash-comm/tests/device_seam.rs` runs the contract with four rank threads over shared
memory for 100 sequences (arrival order shuffled, rails reversed per peer) and requires the result
to equal `canonical_all_reduce_into` bit for bit. The tp4-smoke arm's `--oneshot-preflight` validates
the same descriptors plus the cross-crate ABI pin (comm's constants against
`sparkflash-kernels::roce_abi`) and records an `oneshot` evidence section stating what the host
proved and what it did not.

The host side **cannot** prove: that the CUDA kernel launches or that the slot is reachable from
it, that the pinned-host alias or the device-memory MR registers (`ibv_reg_mr`,
`ibv_reg_dmabuf_mr`), that the QP posting and the proxy's completions agree with the NIC, or any
latency. Those are exactly `rdma-backend.md` 15.3 items 1, 3 (threaded to real QPs), 4 and 5: the
launcher, the proxy thread, the MR wiring and the corrected header text.

The window measurement that closes S5.3 (the line b71's gate needs) is: on the four nodes, one
`--comm-backend rdma` dispatch of the **device-buffer** form at decode size - hidden 5,120 bf16
(10,240 B) per seam - recorded per rank and per seam (wo_b, routed MoE, shared expert, embedding),
with (a) one-shot all-reduce p50 <= 20 us (QUALIFICATION 3.1), measured before any TP step; (b) the
output bit-identical to the canonical CPU twin on **every** repeat and
`repeats_bit_identical == repeats`; (c) two runs per configuration byte-identical (the existing
10,000-replay determinism check); (d) zero DtoH/HtoD between dispatches, asserted by the launcher's
dispatch record (device pointers only), not inferred; (e) the one-shot call count and the per-seam
byte counts recorded per position (123 all-reduces + 2 all-gathers), so a path that quietly fell
back to host staging fails the gate. Prerequisites in order: the three existing hardware gates
green, `rocenante-probe` on the active rail (`rocep1s0f1`) printing its 20 us / 10 KiB result and
repeat determinism, then the device seam. If (a) misses, the measured gap and the NCCL fallback
(section 11) are recorded; the gate is not redefined.

Item (d) has a host-side carrier: `DispatchRecord`/`DispatchLog` in `oneshot.rs` are the record
type the launcher fills for **every** collective call (unconditionally, no verbosity flag, no
log-only path), carrying the slot and row device addresses and a `host_copy_bytes` that must be 0;
`device_path_ok` refuses a record whose path was host staging and names the first offending
sequence. `--oneshot-preflight` exercises the record shape and that refusal on the host, and the
evidence section states `emitted: false` until the window's launcher fills it.

*Loopback host twin (2026-10-19, task-35 host portion; the node run stays frozen).* The
device-buffer path now runs end to end on a host with no GPU and no NIC:
`sparkflash_comm::oneshot::loopback::LoopbackDeviceTransport` implements the device form behind the
same seam, with a two-deep slot ring per rank, the doorbell/arrival handshake, a CPU twin of the
kernel's one-rounding reduction (`kernel_reduce_into`), a dispatch record with zero host-staging
bytes on every call, and a chunked pipeline whose ring reuse waits for every rank's read-done epoch.
`crates/sparkflash-comm/tests/oneshot_host.rs` proves bit-identity against the canonical twin,
arrival-order independence, the one-rounding-versus-four-rounding distinction, the ring shape, and
the refusal of the host-slice forms; `--oneshot-preflight` records a `host_twin` evidence section
whose unclaimed list names latency, bandwidth and four-node mesh correctness. None of it is a node
measurement, and 11.1's transport decision is untouched.

*Window-prep note (2026-10-19): the aarch64 release binary for the S5.3 window must be built with
`--features=rdma,cuda` **and** `SPARKFLASH_RDMA_INCLUDE=<staged aarch64 rdma-core include root>`,
because the zig cross sysroot has no `infiniband/verbs.h`; without it the build silently takes the
ENOSYS stub path (`SPARKFLASH_IBVERBS_SHIM=stub`) and the node binary then refuses at
`Verbs::load` instead of posting. The aarch64 headers are staged on this dev host under
`/home/user/opt/aarch64-rdma/include` (copied read-only from node0). The build log must not
contain the "rdma feature built without <infiniband/verbs.h>" warning.*

**S5.4 -- task-36: prefill-sized collectives + KV.** Depends on S5.2 (coordinates with task-21):
>2 MiB ring/chunked path, embedding all-reduce and Engram all-gather at 4096 tokens, KV per rank.

*Status (host side landed 2026-10-19, task-39; the window is not booked).* The seam now has the
chunked form prefill needs: `sparkflash_comm::oneshot::{Chunk, ChunkPlan, ChunkedProxy}` split a
payload larger than one row into ordered chunks of at most 2 MiB, one `Exchange` per chunk with its
own sequence number, dispatch and doorbell, reusing the same slot rows and the same release/
peer-epoch words (`ChunkPlan::dispatches` is the per-chunk descriptor list a launcher consumes).
Order is the contract that makes the reuse safe: chunk i+1 may be posted only after every peer's
arrival word for chunk i is published, and the receive state machine refuses a frame whose chunk is
not the one in flight. `sparkflash device-chain --run-stage tp4-smoke --oneshot-preflight
--prefill-plan-elements 20971520` records the 4096-token plan in the evidence: 41,943,040 B =
exactly 20 rows, 1,048,576 elements per chunk, 3 peers x 2 MiB posted per chunk and 125,829,120 B
per rank over the series, under `chunked`, with its `does_not_prove` list.

The host test `four_ranks_chain_three_chunks_through_one_slot_and_match_the_canonical_twin`
(`crates/sparkflash-comm/tests/device_seam.rs`) runs four rank threads over a three-row payload
through one slot and checks every chunk bit for bit against `canonical_all_reduce_into`; the unit
tests refuse a future chunk's frame and refuse sizes that are not whole bf16 elements or row splits.
The loopback host twin (15.6 of rdma-backend.md) also runs this chunked plan through the two-slot ring and matches the canonical twin chunk for chunk with zero recorded host-staging bytes, so the reuse order is exercised on the host, not only designed. **What is still a window measurement and is not claimed here**: per-chunk and total chunked latency
against the single-seam path, bit-identity to the canonical twin for every chunk and every repeat
twice per configuration, zero DtoH/HtoD between chunks, and the kernel trace showing no lost or
stale doorbell when the arrival words are reused across chunks. It needs S5.3's device-path gate
green first, then four GPU nodes opened by the Lead and a cuda-feature build with the launcher.

**S5.5 -- task-37: 4-process serving orchestration.** Depends on S5.2/S5.4, coordinates with
task-20/21/22. Feeds S7 (task-25) and S8 (task-26).

## 10. What could go wrong

1. **RDMA dataplane first bring-up.** It has never run on real hardware; libibverbs dlopen, memory
   registration, GID/MTU and QP handshake can each fail on a first run. Mitigation: the TCP path
   is the correctness carrier (S5.1/S5.2 do not depend on RDMA), S5.3 is a separate window with a
   fallback decision, and the hardware gates separate environment from code.
2. **The GPU-kernel to dataplane seam is half written.** The contract (slot geometry, descriptors,
   proxy state machine, device-buffer trait method, cross-crate ABI pin) is host-tested; the kernel
   launcher, the alias/MR registration and the threaded proxy over real QPs are not, and are the
   window's first job (`rdma-backend.md` 15.3). The measured gate is section 9/S5.3.
3. **Memory.** 72.31 GiB device weights + KV + workspace against 121.69 GiB UMA shared with the
   host. A loader that reads through page cache can evict device memory under pressure; admission
   is checked before load, page cache is dropped per shard, and the load is streamed. A failure
   here is a load-time refusal, not a mid-run OOM.
4. **Load time.** ~134 GiB read/rank with 4x amplification on w2, plus host-side B12x tiling of
   5 GB of attention weights. Target <=60 s; if worse, the optional `pack-tp4` repack is the fix
   (decision 11.3).
5. **Numeric divergence at row-parallel seams.** Association order differs from the single-rank
   dot. Mitigated by the declared L2 bar and by comparing the TP=4 path against the host f32 twin
   (it must be closer than the serial emulation). If the emulation ever matches *better* than the
   one-rounding native rule at a seam, that seam's rule is wrong.
6. **The oracle changes under us.** task-4/task-5/task-6 are rewriting the native arithmetic (MoE
   combine, SwiGLU round, `wo_a`). S5.2 depends on task-6, and task-4's N1 changes exactly the
   seam where the TP reduction lives. Coordinate the seam rule with kernel-dev before S5.2 lands
   and compare against the oracle version S5.2 is pinned to.
7. **Ownership and merge conflicts.** `attention_device.rs` is engine-dev's task-6 scope,
   `serve.rs` is serve-dev's task-20/21/22 scope, `device_chain.rs` is shared with merge-dev's
   task-17. S5.2/S5.5 must be sequenced against those, not run beside them; write scopes are
   advisory, so the Lead should order the claims.
8. **Collective latency.** Host-staged TCP all-reduce at 10 KiB is milliseconds, not microseconds:
   a correctness carrier, not a perf answer. Perf parity (task-25) needs S5.3's device path; if
   RDMA misses, NCCL is the only likely way to hit the bar.
9. **Deep collectives per token.** 123 all-reduces per position adds up (~37 ms/token at 0.3 ms
   host-staged: unusable for serving, fine for evidence runs). If the count hurts serving, the
   native rule allows fusing the routed and shared reduction into one payload (the reference fuses
   the shared expert into the mega-MoE kernel), and/or batching layers' reductions. Recorded as a
   design question, not silently optimized.
10. **Engram all-gather at prefill size** (48 MiB at 4096 tokens) crosses the one-shot domain and
    carries disk-read latency; cold rows (~2 ms/row measured on the reference) can dominate a
    prefill. S5.4 measures warm/cold and the cache policy (the reference keeps a 2 GB cache).
11. **Rank failure mid-window.** Four ssh processes; a node reboot or container OOM kills one
    rank. Timeouts turn it into a clean abort, but the window is spent. Keep windows short and
    stage everything beforehand.
12. **Cross-build drift.** Nodes are aarch64; `scripts/build-aarch64.sh` pins Rust 1.98.1 and Zig
    0.17.0. The `rdma` feature adds dlopen-only code (no link dependency), but its C shim must
    compile for aarch64. Build and smoke-test the binary on the dev host before every window.
13. **Fake equivalence.** A TP=4 run that falls back to the single-rank path, or a collective whose
    result a host seam overwrites, could "pass" the stage bars while doing nothing. Evidence must
    include per-rank peer ids, per-seam byte counts and the collective call count (123/position)
    so a run cannot silently degenerate.
14. **D40 floor rule.** Bars are provisional until D41 is frozen at >=3x the oracle
    self-consistency floor (task-13/task-14). A TP=4 evidence run must not be judged against bars
    the Lead has not frozen.

## 11. Open decisions for the Lead

1. **Transport (before S5.3).** Path T (sparkflash-comm: TCP for correctness, then the RoCEnante
   dataplane) -- recommended, D29-consistent, deterministic reduce, 12,950 reviewed lines; risk is
   the first hardware bring-up. Path N (NCCL via cudarc FFI) -- matches the reference exactly and
   is the perf safe bet, but adds an external dependency and gives up the bit-canonical
   reduction. Recommendation: T as primary, N as the documented measured-gap fallback only.
2. **Memory correction (for the plan).** Confirm 72.31 GiB/rank resident (not ~113 GiB); ~90.4
   GiB per rank all-in against 121.69 GiB, with the Engram tables on NVMe per the reference patch.
   Every later budget or concurrency statement (max context, batch size, TP rows) must use this.
3. **Read amplification vs a repack (before S5.2's window).** Read in place with ~4x amplification
   on w2 (~134 GiB/rank, reference-comparable) or add a one-time `pack-tp4` repack to each node's
   NVMe (~72 GiB/rank). Recommendation: measure S5.2's load time first; repack only if it exceeds
   the window's tolerance.
4. **The row-parallel seam rule (before S5.2).** Adopt the canonical one-rounding reduction as the
   native rule at the MoE/wo_b/shared/embedding seams (matches NCCL f32 accumulation) and have
   task-4's N1 bring the single-rank emulation onto the same rule, so the oracle and TP=4 agree by
   construction. Alternative: keep the 4-rounding serial chain and accept the L2 bar as a
   permanent deviation -- not recommended; it freezes an artifact as the oracle.
5. **Window sequence.** After approval: S5.1 15-minute window (2 nodes), then S5.2 90-minute
   window (4 nodes) once task-6 lands. Both are requested through the Lead per the runbook; I
   never stop or start the server.
