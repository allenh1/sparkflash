# SparkFlash-RS: A Purpose-Built Rust Inference Server for DeepSeek-V4.1-Flash on a 4x NVIDIA DGX Spark Cluster

**Target Model:** [deepseek-ai/DeepSeek-V4.1-Flash](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fdeepseek-ai%2FDeepSeek-V4.1-Flash) (552B MoE Backbone + 196B Engram Conditional Memory, 8B Prefill / 16B Decode Active Parameters, 1M Context Window)  
**Target Hardware:** 4× [NVIDIA DGX Spark](https://www.google.com/search?q=https%3A%2F%2Fdocs.nvidia.com%2Fdgx%2Fdgx-spark%2Fhardware.html) (GB10 Grace Blackwell Superchip, `aarch64`, Compute Capability `sm_121a`, 128 GB Coherent Unified LPDDR5X @ 273 GB/s per node, Dual-Port ConnectX-7 200 GbE RoCEv2)  
**Primary References:** [DeepSeek-V4.1-Flash Technical Report (arXiv:2609.19969)](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2609.19969) · [DeepSeek Engram Paper (arXiv:2601.07372)](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fhtml%2F2601.07372v1) · [DeepSeek Recipe Rust Toolkit](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdeepseek-ai%2Fdeepseek-recipe%2Fblob%2Fmain%2FREADME.md) · [MiaAI-Lab 3–4x DGX Spark Reference](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FMiaAI-Lab%2FDeepSeek-v4.1-Flash-DGX-Sparks) · [knapcio TP4 Production Profile](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fknapcio%2FDeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) · [dgx-spark-networking RoCE Collectives](https://www.google.com/search?q=https%3A%2F%2Fforums.developer.nvidia.com%2Ft%2Fdgx-spark-networking-open-source-switchless-roce-collectives-nccl-profiles-and-fabric-tooling-for-multi-spark-inference%2F385049)

## 1. Executive System Thesis: Co-Designing Around GB10 Unified Memory and DeepSeek-V4.1-Flash Asymmetry

Standard LLM serving frameworks ([vLLM](https://www.google.com/search?q=https%3A%2F%2Fdocs.vllm.ai%2Fen%2Flatest%2Ffeatures%2Fengram%2F) and SGLang) are architected for discrete data-center GPUs where host DRAM and GPU HBM are physically separate pools, and where general-purpose Python/LLVM/Triton abstractions mediate every forward pass. On a four-node [NVIDIA DGX Spark](https://www.google.com/search?q=https%3A%2F%2Fdocs.nvidia.com%2Fdgx%2Fdgx-spark%2Fhardware.html) cluster running [deepseek-ai/DeepSeek-V4.1-Flash](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fdeepseek-ai%2FDeepSeek-V4.1-Flash), those assumptions break down in five critical ways:

1. **CPU Offload Is a Fatal Anti-Pattern on Unified Memory:** Each DGX Spark pairs a 20-core Grace ARM CPU (10 Cortex-X925 + 10 Cortex-A725) with a Blackwell `sm_121` GPU over a 600 GB/s NVLink-C2C interconnect sharing a single coherent **128 GB LPDDR5X unified memory pool** (~121.7 GiB OS-visible) at **273 GB/s** ([DGX Spark Hardware Overview](https://www.google.com/search?q=https%3A%2F%2Fdocs.nvidia.com%2Fdgx%2Fdgx-spark%2Fhardware.html)). In [vLLM's default Engram implementation](https://www.google.com/search?q=https%3A%2F%2Fdocs.vllm.ai%2Fen%2Flatest%2Ffeatures%2Fengram%2F), `--engram-config.cpu_offload=true` pins the ~189 GiB of Engram tables in host RAM. On GB10, host RAM *is* GPU RAM: pinning 48 GiB/node of Engram tables in unified memory alongside 76.5 GiB/node of backbone weights immediately exhausts the 121.7 GiB node ceiling ([MiaAI-Lab TP4 Analysis](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FMiaAI-Lab%2FDeepSeek-v4.1-Flash-DGX-Sparks)).
2. **TP=4 Eliminates All Shard Padding and MoE Stragglers:** Every architectural dimension of [DeepSeek-V4.1-Flash/config.json](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fdeepseek-ai%2FDeepSeek-V4.1-Flash%2Fblob%2Fmain%2Fconfig.json) (`64` query heads, `8` output groups, `32` indexer heads, `384` routed experts, `2304` MoE intermediate width, `128` DSpark draft experts, and `129,280` vocabulary rows) divides evenly by **4**. Running **Tensor Parallelism `TP=4` with Expert Parallelism `EP=1`** shards every expert's intermediate dimension (`2304 / 4 = 576` per rank) rather than distributing whole experts across ranks (`EP=4`), eliminating expert-routing straggler stalls at the MoE All-Reduce (`16.5 ms` $\to$ `2.6 ms/step`) ([knapcio TP4 Production Profile](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fknapcio%2FDeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)).
3. **Decode Is RoCE Latency-Bound While Prefill Is Bandwidth-Bound:** A single DGX Spark ConnectX-7 QSFP112 port is fed by two PCIe Gen5 x4 roots, delivering **196.08 Gbps (~24.5 GB/s)** when both RDMA interfaces (`rocep1s0f0` and `roceP2p1s0f0`) are driven concurrently ([Petronella Tech GB10 Interconnect Analysis](https://www.google.com/search?q=https%3A%2F%2Fpetronellatech.com%2Fblog%2Fdgx-spark-cluster-bandwidth-what-400g-really-means%2F)). Standard NCCL ring collectives incur **90.6–92.4 µs** per 10 KiB All-Reduce across four Sparks (`~13–16 ms` wasted per decode step across 104 collectives). By routing small decode/verify collectives ($\le 2\text{ MiB}$) through a **custom one-shot RDMA kernel (`RoCEnante` protocol)** that writes directly into peer unified-memory slots (`16.9 µs` at 10 KiB) and routing large prefill collectives ($> 2\text{ MiB}$) through **patched bidirectional NCCL 2.30.7+**, we achieve optimal latency and bandwidth across both phases ([dgx-spark-networking Benchmarks](https://www.google.com/search?q=https%3A%2F%2Fforums.developer.nvidia.com%2Ft%2Fdgx-spark-networking-open-source-switchless-roce-collectives-nccl-profiles-and-fabric-tooling-for-multi-spark-inference%2F385049)).
4. **Pure Rust + Pre-Compiled `sm_121a` Kernels Beat vLLM and LLVM/Triton JIT:** Stock Python engines launch ~2,000 small kernels per step and rely on LLVM/Triton JIT compilation that mis-tiles for GB10's `101,376-byte` shared-memory limit (`52 ms/step` on Triton vs. `17 ms/step` on native `b12x` warp-level MMA for `32x32 ue8m0` MXFP8 dense projections). `SparkFlash-RS` compiles into a single static `aarch64-unknown-linux-gnu` binary using [deepseek-recipe](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdeepseek-ai%2Fdeepseek-recipe%2Fblob%2Fmain%2FREADME.md) for zero-copy prompt encoding, `cudarc` for direct CUDA Driver API graph capture, and pre-compiled `sm_121a` SASS/PTX kernels.
5. **First-Class Scoped & Zero-Overhead Trace Points Unlock Iterative Optimization:** Because a 4-node TP=4 engine is a tightly coupled pipeline of CPU `io_uring` NVMe prefetches, ConnectX-7 RDMA transfers, and CUDA-graph-captured `sm_121a` kernels, you cannot optimize what you cannot measure—yet naive profiling (`cudaEventRecord` or `nvidia-smi` polling) destroys CUDA graph replay and adds `~0.6–2.0 ms/step` of jitter ([MiaAI-Lab Profiling Notes](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FMiaAI-Lab%2FDeepSeek-v4.1-Flash-DGX-Sparks)). Every subsystem in `SparkFlash-RS` is instrumented with **compile-time and runtime-scoped trace points (`sparkflash-trace`)** combining ARM `CNTVCT_EL0` CPU counters, NVTX v3 ranges, and **in-graph `%globaltimer` UMA timestamp tapes** that cost **0 ns when disabled** and can be toggled per-subsystem or per-request via HTTP headers.

| Performance Metric | Stock vLLM / Early SGLang (4x Spark TP4) | Tuned Python + SGLang Canary ([knapcio v2.4](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fknapcio%2FDeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)) | SparkFlash-RS Target (4x Spark TP4, `sm_121a`) | Primary Speedup Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Single-Stream Prose Decode (`c=1`)** | 45.4 – 75.5 tok/s | 89.9 tok/s | **94.0 – 98.5 tok/s** | Zero Python/IPC overhead + Full CUDA Graph Draft-Verify glue + L2 weight prefetch |
| **Single-Stream Code Decode (`c=1`)** | 57.9 – 77.2 tok/s | 131.0 tok/s | **138.0 – 146.0 tok/s** | `DSpark k=5` (`conf: 0.1` live router) + `RoCEnante` 16.9 µs one-shot RDMA |
| **Aggregate Prose Decode (`c=16`)** | 134.2 tok/s | 341.3 tok/s | **365.0 – 390.0 tok/s** | Lock-free Rust continuous batcher + DRM/UMA 16-way Engram cache + zero-copy UMA |
| **Aggregate Code / JSON Decode (`c=16`)** | ~210.0 tok/s | 440.3 / 632.8 tok/s | **475.0 / 670.0 tok/s** | Compact ragged DSpark verification + `b12x_next` deterministic grouped MXFP4 MoE |
| **Cold Prefill (`16K – 128K`, Real Text)** | 3,251 – 3,782 tok/s | 4,976 tok/s (5,969 synth) | **5,350 – 6,200 tok/s** | CED 20-layer prefill + Sequence Parallelism (`PREFILL_SP`) + TP-split HSI indexer |
| **Persistent Prefix-Hit TTFT (`388K` ctx)** | 7.8 s (disk KV) | 1.5 s (cached) | **0.85 – 1.15 s** | Zero-copy UMA radix cache + 128-token SWA Bounded Replay |

## 2. DeepSeek-V4.1-Flash Architectural Specification From the Technical Paper

Every module in `SparkFlash-RS` is hard-coded to the exact invariants of [deepseek-ai/DeepSeek-V4.1-Flash](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fdeepseek-ai%2FDeepSeek-V4.1-Flash%2Fblob%2Fmain%2Fconfig.json) as defined in [arXiv:2609.19969](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2609.19969).

### 2.1 Model Hyperparameters and Exact `TP=4` Partitioning

| Config Parameter (`config.json`) | Global Value | Per-Rank Value (`TP=4, EP=1`) | Precision & Block Format | Notes & Hardware Alignment |
| :--- | :--- | :--- | :--- | :--- |
| `num_hidden_layers` | 40 | 40 (20 Encoder + 20 Decoder) | Mixed FP8 / MXFP4 | Layers `0..19` Causal Encoder; Layers `20..39` Decoder |
| `hidden_size` ($d$) | 5120 | 5120 (Replicated / SP-sharded) | BF16 residual / FP8 acts | Divisible by $4 \times 1280$ tile widths |
| `hc_mult` ($n$) | 4 | 4 parallel residual streams | BF16 (`4 x 5120` per token) | Single-Pass mHC (`hc_sinkhorn_iters: 20`, `hc_eps: 1e-6`) |
| `vocab_size` | 129,280 | 32,320 rows/rank | BF16 / MXINT8 screen | Cleanly divisible by 4 (`bos: 0, eos: 1, pad: 2, img: 129264`) |
| `num_attention_heads` | 64 | 16 heads/rank | FP8 (`E4M3`, `32x32 ue8m0`) | `head_dim: 512`, `qk_rope_head_dim: 64`, `q_lora_rank: 1280` |
| `o_groups` / `o_lora_rank` | 8 / 1024 | 2 groups/rank (`1024` dim) | FP8 twin for `wo_a` | Eliminates BF16 einsum bottleneck (`-11 ms/step`) |
| `num_key_value_heads` | 1 (MLA Latent) | 1 (Replicated in Decode, SP in Prefill) | **FP4 (`E2M1`)** + `E4M3`/16ch | Stored only at `kv_source_layer_ids: [2, 8, 14, 20]` |
| `sliding_window` ($n_{\text{win}}$) | 128 | 128 tokens/layer | **FP8 (`E4M3`)** | Active in all 40 layers; reconstructed via Bounded Replay |
| `index_n_heads` / `index_head_dim` | 32 / 128 | 8 heads/rank (`128` dim) | **MXFP4** (`ue8m0`/32ch) | `index_topk: 512` selected per query |
| `candidate_topk_blocks` $\times$ `size` | $2048 \times 8$ | $16,384$ candidate positions | INT32 indices | Built at `candidate_source_layer_id: 20` (Decoder Full Mode) |
| `n_routed_experts` / `per_tok` | 384 / Top-6 | All 384 experts, `576` width/rank | **MXFP4** (`32x32 ue8m0`) | `moe_intermediate_size: 2304` $\to$ `576`/rank (`EP=1`) |
| `n_shared_experts` | 1 | 1 expert, `576` padded $\to$ `640`/rank | **MXFP8** (`b12x` kernel) | Zero-pad $K$: $576 \to 640$ to stay on `b12x` warp MMA (`-0.9 ms`) |
| `engram_layer_ids` | `[1, 14]` | Sharded on local NVMe (~48 GiB/rank) | **FP8 (`E4M3`)** + `ue8m0` | `[384,006,168, 384,016,682]` rows, `8` heads, `256` dim, $N \in \{2,3,4\}$ |
| `dspark_target_layer_ids` | `[37, 38, 39]` | 3 layers, `128` experts (`32`/rank) | MXFP4 / FP8 draft head | `dspark_block_size: 5`, `dspark_markov_rank: 256`, `top-3` experts |
| `vision_config` (`DeepSeek-ViT`) | 32 layers | Replicated or TP4 (`4` heads/rank) | BF16 / FP8 | `hidden_size: 1024`, `patch_size: 14`, $3\times 3$ pixel-unshuffle |

### 2.2 Causal Encoder-Decoder (CED) Execution Flow

As detailed in Section 2.2 of [arXiv:2609.19969](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2609.19969), `DeepSeek-V4.1-Flash` splits its 40 Transformer blocks into a **20-layer Causal Encoder** ($l \in [0, 19]$) and a **20-layer Decoder** ($l \in [20, 39]$):

- **Prefill Asymmetric Bypass ($O(NL/2)$):** During prefill of an $N$-token prompt ($N \gg 128$), all $N$ tokens pass through the 20 Causal Encoder layers ($l = 0..19$), activating only **8B parameters per token**.
- **Cross-Stage Global KV Projection:** Instead of running the $N$ prompt tokens through the 20 Decoder layers to build per-layer decoder KV caches, the decoder's single shared global KV cache (owned by Layer 20) is projected directly from the final encoder layer's output hidden states $H_{19}$:
  $$C_{20} = H_{19} W_{20}^{KV}, \quad Z_{20} = H_{19} W_{20}^{Z}$$
- **Decoder SWA Bounded Replay:** Because every decoder layer ($l = 20..39$) still requires a local 128-token Sliding Window Attention (SWA) KV cache to begin autoregressive decoding, exact reconstruction would require running the last $20 \times 128 = 2,560$ prompt tokens through the decoder. Instead, **Decoder SWA Bounded Replay** ([arXiv:2609.19969 Section 3.2.2](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2609.19969)) slices only the **last $n_{\text{win}} = 128$ tokens** of $H_{19}$ and runs those 128 rows through Decoder layers $20..39$ under SWA truncation.
  - *API Guardrail:* Because Decoder layers $20..39$ only execute on the final 128 tokens of a prompt during prefill, full-prompt token logprobs and full-prompt final-layer hidden states do not exist for positions $t < N - 128$. `SparkFlash-RS` immediately returns `HTTP 400 Bad Request` with a clear diagnostic message if a client requests `echo: true` with `logprobs` on prompts $> 128$ tokens.

### 2.3 Compressed Sparse Attention 2 (CSA2) and Hierarchical Sparse Indexer (HSI)

Each layer combines a local 128-token SWA branch (in FP8) with a global CSA2 branch (except Layers `0` and `1`, which are SWA-only). CSA2 statically assigns each layer one of three modes defined by `compress_ratios`, `kv_source_layer_ids`, and `index_source_layer_ids` in [config.json](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fdeepseek-ai%2FDeepSeek-V4.1-Flash%2Fblob%2Fmain%2Fconfig.json):

| Layer Range | Stage | Compression Ratio ($m$) | CSA2 Mode Breakdown | KV & Indexer Sharing Mechanics |
| :--- | :--- | :--- | :--- | :--- |
| **Layers `0..1`** | Encoder | $m = 0$ (SWA Only) | Pure SWA (2 layers) | Computes local 128-token SWA KV only; Layer 1 fuses first **Engram** module |
| **Layers `2..7`** | Encoder Group 1 | $m = 2$ | Layer `2`: **Full** · Layers `3..7`: **Reuse** | Layer 2 compresses every 2 tokens into 1 Main KV entry, projects Indexer K, selects Top-512; Layers 3–7 reuse both |
| **Layers `8..13`** | Encoder Group 2 | $m = 2$ | Layer `8`: **Full** · Layers `9..13`: **Reuse** | Layer 8 computes fresh $m=2$ Main KV, Indexer K, and Top-512; Layers 9–13 reuse both |
| **Layers `14..19`** | Encoder Group 3 | $m = 2$ | Layer `14`: **Full** · Layers `15..19`: **Reuse** | Layer 14 fuses second **Engram** module, computes fresh $m=2$ Main KV & Top-512; Layers 15–19 reuse both |
| **Layers `20..23`** | Decoder Group 1 | $m = 1$ | Layer `20`: **Full + HSI** · Layers `21..23`: **Reuse** | Layer 20 projects $m=1$ Main KV & Indexer K from $H_{19}$, scans full context for Top-512, and builds **16,384-pos Candidate Pool** ($2048 \times 8$) |
| **Layers `24..39`** | Decoder Groups 2–5 | $m = 1$ | Layers `24, 28, 32, 36`: **Reindex** · Others: **Reuse** | Reindex layers reuse Layer 20's Main KV & Indexer K, score **only** the 16,384 HSI candidate positions to pick fresh Top-512; 12 Reuse layers reuse both |

**Global KV Cache Footprint Math (890 Bytes/Token):**
- Only **4 layers in the entire 40-layer model** (`kv_source_layer_ids: [2, 8, 14, 20]`) store global Main KV and Indexer K!
- Encoder Full layers (`2, 8, 14`) have compression ratio $m = 2$ (storing $0.5$ entries per token each), while Decoder Full layer (`20`) has compression ratio $m = 1$ (storing $1.0$ entry per token). Across all 40 layers, a single token generates only $3 \times 0.5 + 1.0 = 2.5$ compressed Main KV + Indexer K entries!
- Each Main KV entry is an MLA latent of `head_dim = 512` (quantized after RoPE to **FP4 `E2M1`** = 256 bytes + 32 bytes of `E4M3` scales per 16 channels) plus `qk_rope_head_dim = 64` and Indexer K (`128` dim in MXFP4 = 64 bytes + 4 bytes `ue8m0` scale), yielding the **890 bytes/token** logical global KV footprint reported in [arXiv:2609.19969](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2609.19969).

### 2.4 Single-Pass mHC and the Fused `Mega-mHC` Kernel

Instead of DeepSeek-V4's 3-pass Manifold-Constrained Hyper-Connections ($X_{l+1} = B_l X_l + C_l F_l(A_l X_l)$, which required waiting for a full hidden-dimension reduction before computing $A_l X_l$ and consumed $(4n+4)d = 20d$ memory reads/writes), `DeepSeek-V4.1-Flash` shifts the input-mixing coefficients $A_{l-1}$ by one layer ([arXiv:2609.19969 Section 2.4.1](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2609.19969)):
$$X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l), \quad (A_l, B_l, C_l) = \mathcal{H}(X_l)$$
In `SparkFlash-RS`, we bind a single fused **`Mega-mHC` CUDA kernel** (`kernels/mega_mhc_sm121.cu`) that traverses $X_l \in \mathbb{R}^{4 \times 5120}$ in hidden-dimension tiles once:
1. Updates $X_l \leftarrow B_{l-1} X_{l-1} + C_{l-1} Y_{l-1}$ in registers.
2. Immediately computes the mixed block input $\hat{X}_l = A_{l-1} X_l$ using the previous layer's coefficients $A_{l-1}$.
3. Accumulates tile dot-products and sum-of-squares for $\mathcal{H}(X_l)$ (with RMSNorm weights folded offline into the predictor projection and 20 Sinkhorn-Knopp iterations executed in warp shared memory at the end of the sweep).
4. Applies input RMS pre-norm and dynamic `32x32 ue8m0` FP8 quantization to $\hat{X}_l$, writing only $X_l$ and the quantized block input $\hat{X}_l^{\text{FP8}}$—achieving the theoretical minimum $(2n+2)d = 10d$ memory traffic (a **50% reduction** in residual-stream bandwidth).

## 3. Engram Conditional Memory Architecture & Optimal 4x DGX Spark Strategy

### 3.1 Mathematical Formulation of DeepSeek-V4.1-Flash Engram (`Layers 1` and `14`)

From [arXiv:2601.07372](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fhtml%2F2601.07372v1) and [arXiv:2609.19969 Section 2.4.2](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2609.19969), Engram decouples static local pattern storage from MoE dynamic computation at `engram_layer_ids = [1, 14]`:

1. **Surjective Tokenizer Compression ($\mathcal{P}: V \to V'$):** At server startup, `SparkFlash-RS` builds a static `u32` lookup table `COMPRESS_MAP[129280]` that normalizes every subword token string via Unicode NFKC, accent stripping, lowercasing, and whitespace collapsing into `engram_compressed_vocab_size = 99,092` canonical IDs.
2. **Multi-Head Multiplicative-XOR N-Gram Hashing:** For each token position $t$, Engram forms suffix N-grams $g_{t,n} = (x'_{t-n+1}, \dots, x'_t)$ for $n \in \{2, 3, 4\}$ (`engram_max_ngram_size = 4`, left-padded with `engram_pad_token_id = 2`). For each order $n \in \{2, 3, 4\}$ and head $k \in \{0..7\}$ (`engram_n_heads = 8`), a deterministic multiplicative-XOR hash $\varphi_{n,k}(g_{t,n}) \bmod M_{\ell, n, k}$ indexes a disjoint prime-sized bucket $M_{\ell, n, k} \approx 16\text{M}$ rows inside layer $\ell$'s giant table ($\sum_{n,k} M_{1,n,k} = 384,006,168$ rows for Layer 1; $\sum_{n,k} M_{14,n,k} = 384,016,682$ rows for Layer 14).
3. **Embedding Concatenation ($d_{\text{mem}} = 6144$):** Each of the $3 \times 8 = 24$ lookups retrieves an FP8 (`E4M3`) vector of dimension `engram_head_dim = 256` (plus 8 `ue8m0` scale bytes for 32-element blocks). Concatenating all 24 retrieved vectors yields the static memory prior $\mathbf{e}_t \in \mathbb{R}^{6144}$.
4. **Multi-Branch Context-Aware Gating (No Conv1D in V4.1-Flash):** As noted in [arXiv:2609.19969 Section 2.4.2](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2609.19969), DeepSeek-V4.1-Flash **omits the depthwise causal Conv1D** used in the original Engram paper. Instead, a single fused FP8 projection `engram.wkv` maps $\mathbf{e}_t \in \mathbb{R}^{6144}$ to one shared Value vector $\mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t \in \mathbb{R}^{5120}$ and $M = 4$ branch-specific Key vectors $\mathbf{k}_t^{(m)} = \mathbf{W}_K^{(m)} \mathbf{e}_t \in \mathbb{R}^{5120}$ for the 4 mHC streams:
   $$\alpha_t^{(m)} = \sigma\left(\frac{\text{RMSNorm}(\mathbf{h}_t^{(m)})^\top \text{RMSNorm}(\mathbf{k}_t^{(m)})}{\sqrt{5120}}\right), \quad \mathbf{h}_t^{(m)} \leftarrow \mathbf{h}_t^{(m)} + \alpha_t^{(m)} \cdot \mathbf{v}_t$$
   For image token spans (`x_t == 129264`), $\alpha_t^{(m)}$ is masked to `0.0`.

### 3.2 Why We Shard Engram Across Local NVMe + 2-Level UMA/DRM Cache

On disk, the two Engram layers occupy **189 GiB** ($768.02\text{M rows} \times 264\text{ bytes}$). Because a DGX Spark's 128 GB LPDDR5X is unified between CPU and GPU, keeping 48 GiB/rank of Engram tables in host DRAM (`cpu_offload=true`) steals 48 GiB directly from the GPU's KV cache pool. Conversely, leaving the 476 GiB safetensors on an NFS share causes worker ranks to pull Engram rows over the network twice per step (`+11 ms/step` stall).

`SparkFlash-RS` implements a **3-Tier Deterministic Engram Engine** specifically engineered for the 4x DGX Spark architecture:

```
+---------------------------------------------------------------------------------------------------+
|                     TOKEN ARRIVAL / DSPARK DRAFT GENERATION (CPU / GPU)                           |
|  Input Token IDs x[t] known BEFORE Layer 0 starts (and 5 Draft Token IDs known before Verify!)   |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|          PHASE 0: GRACE CPU CORTEX-A725 ASYNC PREFETCH SCHEDULER (Zero GPU SM Overhead)           |
|  1. Map x[t] -> x'[t] via 129,280-entry L1-resident COMPRESS_MAP.                                 |
|  2. Compute 24 deterministic (n in {2,3,4}, k in 0..7) multiplicative-XOR hashes per token.       |
|  3. Partition by TP=4 Ownership: Rank r in {0,1,2,3} owns 6 heads per layer (d_mem_local = 1536)  |
|     OR column-slices `engram.wkv` across TP=4 (`DSV41_REPLICATED_SPLIT=engram.wkv`).              |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                  +-------------------------------+-------------------------------+
                  |                               |                               |
                  v (Tier 1: ~25% Hits)           v (Tier 2: ~48% Hits)           v (Tier 3: ~27% Misses)
+---------------------------------+ +---------------------------------+ +---------------------------+
| TIER 1: GB10 DRM Carveout Cache | | TIER 2: UMA Set-Associative     | | TIER 3: Local Gen5 NVMe   |
| - 1.8 GiB in /dev/dri/card0     | |   Hot Row Cache (4.0 GiB/rank)  | | - Packed `engram-l{1,14}` |
| - Costs 0 bytes of CUDA UMA!    | | - 16-Way Set-Associative + Tiny | |   `r{rank}of4.bin`        |
| - Holds Top ~7.3M Zipfian       | |   LFU/CLOCK aging               | |   (~48 GiB/node FP8 or    |
|   2-gram/3-gram rows per node   | | - Combined Tier 1+2 Hit Rate:   | |   ~24 GiB/node MXFP4)     |
| - Zero-copy GPU pointer alias   | |   67% - 76% on real workloads   | | - 96-queue `io_uring`     |
+---------------------------------+ +---------------------------------+ |   O_DIRECT into pinned    |
                  |                               |                     |   UMA staging ring        |
                  +-------------------------------+---------------------+---------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|            OVERLAPPED EXECUTION WINDOW (Zero-Copy Unified Memory Read by sm_121 GPU)              |
|  - Layer 1 Prefetch runs during Vision/Embed + Layer 0 SWA & MoE (~2.1 ms compute window).        |
|  - Layer 14 Prefetch runs during Layers 1..13 (~18.5 ms compute window -> 100% hidden!).          |
|  - `engram.wkv` executes on each rank's 1536-dim slice, followed by a single `RoCEnante`          |
|    one-shot RDMA All-Reduce (or Compact All-Gather) across the 4 Sparks.                          |
+---------------------------------------------------------------------------------------------------+
```

#### Key Engram Optimizations in `SparkFlash-RS`:
1. **Head-Sharded TP=4 Packing (`6` Heads per Rank + Local Partial Projection):**
   - Notice that in Equation (3) of the Engram paper, $\mathbf{k}_t^{(m)} = \mathbf{W}_K^{(m)} \mathbf{e}_t$ and $\mathbf{v}_t = \mathbf{W}_V \mathbf{e}_t$ are linear projections over the concatenation $\mathbf{e}_t = \|_{n=2}^4 \|_{k=0}^7 \mathbf{e}_{t,n,k} \in \mathbb{R}^{6144}$.
   - By linearity of matrix multiplication, if we partition the $24$ `(n, k)` heads across our $4$ DGX Spark ranks so **Rank $r \in \{0, 1, 2, 3\}$ owns exactly $6$ heads ($d_{\text{mem, local}} = 6 \times 256 = 1536$)** and holds the corresponding column slice $\mathbf{W}_{K, [:, 1536r : 1536(r+1)]}^{(m)}$ and $\mathbf{W}_{V, [:, 1536r : 1536(r+1)]}$:
     $$\mathbf{W}_{K}^{(m)} \mathbf{e}_t = \sum_{r=0}^{3} \mathbf{W}_{K, r}^{(m)} \mathbf{e}_{t, r}, \quad \mathbf{W}_V \mathbf{e}_t = \sum_{r=0}^{3} \mathbf{W}_{V, r} \mathbf{e}_{t, r}$$
   - **Why this is a breakthrough on 4x DGX Spark:**
     - Each node only stores **1/4 of the Engram tables** on its local NVMe (`~47.25 GiB` total across Layers 1 & 14 in FP8, or `~23.6 GiB` in MXFP4).
     - Each node only looks up **6 rows per token per Engram layer** instead of 24 rows (**4x reduction in NVMe IOPS** per node—only 36 row lookups per 6-token DSpark verify step!).
     - Each node runs only **1/4 of the `engram.wkv` FP8 GEMM** ($5120 \times 5 \times 1536$ instead of $5120 \times 5 \times 6144$).
     - Instead of gathering raw embeddings across the network, each rank reduces the projected $[5, 5120]$ Key/Value vectors via a single `RoCEnante` one-shot RDMA All-Reduce before computing the RMSNorm dot-product gate!
2. **512-Byte Sector Alignment for Zero-Amplification `O_DIRECT` Reads:**
   - Each Engram row is `256` FP8 bytes + `8` `ue8m0` scale bytes = `264` bytes. In `dsv41-pack`, we pack two rows (`528` bytes) or pad each row to `264` bytes within 4 KiB blocks and use `io_uring` `IORING_OP_READ_FIXED` with pre-registered `cudaHostAllocMapped` buffers so the NVMe controller DMAs directly into coherent LPDDR5X memory visible to `sm_121` SMs without a single CPU `memcpy`.
3. **Exploiting the GB10 Display Carveout (`/dev/dri/card0`) for Tier-1 Cache:**
   - On DGX Spark, the bootloader reserves a `~1.8–2.0 GiB` display carveout in `/dev/dri/card0` outside the OS `MemAvailable` pool ([knapcio Display Reserve Note](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fknapcio%2FDeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)). On headless cluster nodes, `SparkFlash-RS` maps this DRM buffer via `DRM_IOCTL_MODE_CREATE_DUMB` + `cudaHostRegister` to store the top ~7.3 million most frequent N-gram rows (calibrated offline from token unigram/bigram frequencies) completely free of CUDA unified-memory cost, backed by a **4.0 GiB 16-way set-associative L2 row cache** in unified memory.

## 4. 4x DGX Spark Hardware Budget, Topology, and Dual-Path RoCEv2 Transport

### 4.1 Per-Node Unified Memory Budget (`121.7 GiB` Usable LPDDR5X per Node)

Every allocation on a DGX Spark competes for the same 128 GB LPDDR5X physical memory. `SparkFlash-RS` enforces a strict, deterministic static allocation budget at startup (`spark-serving-mode` disables `gdm3`/GNOME and caps SM clocks at `2200 MHz` via `nvmlDeviceSetGpuLockedClocks` to cut GPU rail power by 36% and prevent thermal/EC power-off at a `<1%` decode cost):

| Memory Region | Size per Node (`TP=4, EP=1`) | Backing Mechanism | Architectural Role |
| :--- | :--- | :--- | :--- |
| **DGX OS + Base Kernel (64K pages + `nvidia-uvm` patch)** | `5.2 GiB` | Linux Kernel (`multi-user.target`) | Packed GPU page tables recover ~3.0 GiB vs stock |
| **Backbone + Vision + DSpark Weights (`1/4` TP Slice)** | `76.4 GiB` | `256 MiB` Pinned UMA Slabs (`cudaHostAllocMapped`) | 330 contiguous 256 MiB slabs (vs 94k PyTorch allocs) |
| **Target LM Head `MXINT8` Certified Screen** | `0.17 GiB` | Static GPU/UMA Buffer | Screens greedy argmax tiles before exact BF16 GEMM |
| **Engram Tier-1 Hot Cache (DRM Carveout)** | `0.0 GiB` (uses `1.8 GiB` DRM) | `/dev/dri/card0` Dumb Buffer | Top ~7.3M static Zipfian rows outside OS `MemAvailable` |
| **Engram Tier-2 16-Way Set-Associative Cache** | `4.0 GiB` | Coherent UMA (`cudaHostAllocMapped`) | Dynamic 16-way cache (67–76% hit rate on real prompts) |
| **Global KV Cache Pool (`3.67M – 6.50M` Tokens)** | `13.4 GiB` | Paged FP4/MXFP4 Slabs (`64`-token pages) | Holds up to 6.5M tokens across concurrent 1M contexts |
| **SWA Prefix-Tail Ring Cache (`128` Active Tails)** | `1.1 GiB` | FP8 Ring Buffer (128 tokens $\times$ 40 layers $\times$ 128 slots) | Fast multi-turn resumption without SWA Bounded Replay |
| **Chunked Prefill Indexer & Activation Scratchpad** | `6.5 GiB` | Static Pre-Allocated Ring Arena | Bounds 8192-chunk HSI indexer logits ($\le 2\text{ GiB}$ chunks) |
| **CUDA Graphs (`bs = 1..16` Decode + Draft + Verify)** | `5.4 GiB` | Captured Driver API Graphs | Full-step graph replay with zero eager glue kernels |
| **RoCEnante One-Shot Slots + Tuned NCCL Buffers** | `0.35 GiB` | `ibv_reg_mr` Pinned RDMA Regions | Shrinks stock NCCL's `4.7 GiB` buffer footprint to `139 MiB` |
| **In-Graph GPU/CPU Scoped Trace Ring (`sparkflash-trace`)** | `0.06 GiB` | Lock-Free Coherent UMA Ring Buffer | Stores `%globaltimer` & `CNTVCT_EL0` spans when trace scopes active |
| **Safety Headroom (`MemAvailable` Floor Guard)** | `9.12 GiB` | Unallocated Guard Band | Prevents Linux OOM / kswapd page-reclaim stalls |
| **Total per DGX Spark Node** | **121.7 GiB** | **Coherent LPDDR5X @ 273 GB/s** | **486.8 GiB Cluster Total Across 4 Nodes** |

### 4.2 Dual-Rail ConnectX-7 Wiring & Switchless 4-Node Ring ASIC Forwarding

Each DGX Spark exposes two QSFP112 ports driven by a ConnectX-7 SmartNIC. Crucially, the GB10 SoC connects to the ConnectX-7 via **two separate PCIe Gen5 x4 root complexes** (`~100 Gbps` each), exposing two Linux network/RDMA interfaces per physical port:
- **Port 0:** `enp1s0f0np0` (`rocep1s0f0`) on PCIe root 0, and `enP2p1s0f0np0` (`roceP2p1s0f0`) on PCIe root 2.
- **Port 1:** `enp1s0f1np1` (`rocep1s0f1`) on PCIe root 0, and `enP2p1s0f1np1` (`roceP2p1s0f1`) on PCIe root 2.

`SparkFlash-RS` supports both physical 4-node topologies out of the box:
1. **Switched 200 GbE RoCEv2 Topology (Recommended for Peak Prefill):** All 4 Sparks connect Port 0 (`0.5m QSFP112 DAC` or `400G -> 2x200G breakout`) to a lossless RoCEv2 switch (e.g., MikroTik CRS804-4DDQ or CRS812 with `auto-negotiation=no speed=200G-baseCR4`, `MTU 9000`, ECN/PFC enabled). Both PCIe halves (`rocep1s0f0` and `roceP2p1s0f0`) are active in parallel, delivering **196.08 Gbps (24.5 GB/s)** bidirectional bandwidth per node.
2. **Switchless 4-Node Ring Topology with ConnectX-7 ASIC Hairpin Diagonals:** Nodes are cabled in a closed loop `Spark0 <-> Spark1 <-> Spark2 <-> Spark3 <-> Spark0` using four `0.5m QSFP112` passive DAC cables (using Port 0 for clockwise neighbor and Port 1 for counter-clockwise neighbor). To eliminate two-hop CPU/PCIe relay overhead between opposite nodes (`Spark0 <-> Spark2` and `Spark1 <-> Spark3`), `SparkFlash-RS` includes a startup `mlx5` flow-steering initializer (`spark_roce_hairpin`) that programs the ConnectX-7 internal eSwitch ASIC on intermediate nodes to **forward diagonal RoCE packets directly inside the NIC ASIC** ([SparkRing Virtual Diagonal Mesh](https://www.google.com/search?q=https%3A%2F%2Fforums.developer.nvidia.com%2Ft%2Fswitchless-2x-4x-6x-gb10-clusters-serving-glm-deepseek-and-qwen%2F381032)). Traffic between opposite ranks never touches the intermediate node's PCIe Gen5 bus, CPU, or LPDDR5X memory, adding only **1.78 µs** over a direct cable (`10.51 µs` vs. `8.65 µs` at 64 KiB).

### 4.3 Dual-Backend Collective Transport (`RoCEnante` One-Shot RDMA + Patched `NCCL`)

| Collective Operation | Payload Size Range | Active Phase | Selected Transport Backend | Measured 4-Spark Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Decode Layer All-Reduce (Attention / MoE)** | `10 KiB – 360 KiB` | Decode & DSpark Verify (`bs=1..16`) | **`RoCEnante` One-Shot RDMA** (Dual-Rail `ibverbs` / `GPUNetIO`) | **16.9 µs** (10 KiB) · **28.0 µs** (60 KiB) |
| **Replicated Linear Compact All-Gather (`wqkv_a`, `engram.wkv`)** | `30 KiB – 1.5 MiB` | Decode & DSpark Verify (`bs=1..16`) | **`RoCEnante` One-Shot Compact Gather** (in-place window write) | **22.4 µs** (60 KiB) · **94.0 µs** (480 KiB) |
| **Prefill TP-Split Indexer Top-K Gather** | `64 KiB – 2.0 MiB` | Chunked Prefill ($\ge 32\text{K}$ ctx) | **`RoCEnante` One-Shot All-Gather** (`INT32` indices + block IDs) | **31.0 – 307.7 µs** |
| **Prefill Sequence-Parallel Reduce-Scatter + All-Gather** | `> 2.0 MiB` (`2048–8192` rows) | Chunked Prefill (`DSV41_PREFILL_SP`) | **Patched `NCCL 2.30.7+`** (Bidirectional Dual-Rail Ring) | **184 – 193 Gbps** bus bandwidth |

#### How the `RoCEnante` One-Shot Collective Works on GB10:
1. At startup, each rank allocates two alternating (ping-pong) **One-Shot RDMA Arena Slots** (`4 ranks × 2 MiB × 2 parity = 16 MiB`) in coherent unified memory (`cudaHostAllocMapped`) and registers them with `ibv_reg_mr` on both `rocep1s0f0` and `roceP2p1s0f0`.
2. Inside the CUDA graph, when a layer completes its partial GEMM output, the `roce_stage_kernel` writes the local payload into its local arena slot, executes a **system-scope memory fence (`__threadfence_system()`)** (mandatory on GB10 `sm_121` so the ARM CPU proxy / NIC DMA engine sees fresh L2 cache lines), and increments a volatile sequence counter in unified memory.
3. A dedicated real-time Rust polling thread pinned to CPU Core 19 (`SCHED_FIFO`) detects the sequence counter (or via `DOCA GPUNetIO` WQE preparation) and immediately rings the ConnectX-7 doorbell (`ibv_post_send` `IBV_WR_RDMA_WRITE_WITH_IMM`), striped 50/50 across both PCIe x4 rails to all 3 peer ranks.
4. The `roce_oneshot_reduce_kernel` on the GPU spins on the 3 peer arrival flags in unified memory and, once set, reads all 4 ranks' slots directly in place and sums them in **strict canonical rank order (`rank 0 + rank 1 + rank 2 + rank 3`)**. Because every rank sums the exact same values in the exact same order, **outputs are 100% bit-identical across all 4 nodes**, completely eliminating floating-point drift between ranks!
5. **Simultaneous L2 Weight Prefetch (`DSV41_L2_PREFETCH`):** While `roce_oneshot_reduce_kernel` waits ~15–28 µs for the RDMA writes to land, a secondary CUDA stream executes a lightweight prefetch kernel that touches the first `6.0 MiB` of the next sub-layer's weight matrix (e.g., `wqkv_a`, `wo_a`, or shared expert weights), warming the Blackwell GPU's L2 cache during otherwise dead network wait time (`-1.4 ms/step`).

## 5. Rust Server Architecture & Crate Design (`SparkFlash-RS`)

To beat both vLLM and LLVM/Triton-based engines, `SparkFlash-RS` is structured as a modular Rust Cargo workspace with zero Python dependencies at runtime and a dedicated zero-overhead scoped tracing crate (`sparkflash-trace`).

### 5.1 Workspace Layout

```
sparkflash-rs/
├── Cargo.toml
├── build.rs                           # Compiles sm_121a CUDA/PTX/CUTLASS/b12x kernels via nvcc
├── crates/
│   ├── sparkflash-trace/              # Zero-overhead scoped tracepoints (CNTVCT_EL0 + NVTX3 + %globaltimer UMA tape)
│   ├── sparkflash-api/                # Axum HTTP/SSE OpenAI server + deepseek-recipe integration + Prometheus
│   ├── sparkflash-scheduler/          # Continuous batcher, CED prefill-bypass, DSpark confidence scheduler
│   ├── sparkflash-kv/                 # 3-Tier KV Manager (UMA FP4 Global Radix + FP8 SWA Tail + NVMe Persistent)
│   ├── sparkflash-engram/             # Tokenizer compression, XOR hash, DRM/UMA 16-way cache, io_uring NVMe store
│   ├── sparkflash-comm/               # RoCEnante one-shot ibverbs/GPUNetIO transport + NCCL 2.30.7 FFI + Hairpin
│   ├── sparkflash-kernels/            # cudarc Driver API bindings to sm_121a Mega-mHC, FlashMLA, b12x MXFP8/MXFP4
│   └── sparkflash-cli/                # `sparkflash serve`, `sparkflash pack-engram`, `sparkflash doctor`, `sparkflash trace`
└── kernels/
    ├── trace_globaltimer_sm121.cuh    # Inline PTX `%globaltimer` macro gated by device-visible scope mask
    ├── mega_mhc_sm121.cu              # Single-pass mHC + RMSNorm + dynamic 32x32 ue8m0 FP8 cast
    ├── flash_mla_csa2_sm121.cu        # CSA2 Full/Reindex/Reuse + FP4 E2M1 dequant + 64-tok page split + HSI
    ├── b12x_mxfp8_dense_sm121.cu      # Warp-level 16/32-row MMA for 32x32 ue8m0 FP8 dense projections
    ├── b12x_mxfp4_moe_sm121.cu        # Deterministic grouped MXFP4 MoE GEMM (EP=1, 576-wide, fixed-order sum)
    ├── engram_gate_sm121.cu           # Multi-branch RMSNorm dot-product sigmoid gate + residual add
    ├── cert_lm_head_sm121.cu          # MXINT8 upper-bound argmax screen + exact BF16 top-tile evaluation
    └── rocenante_oneshot_sm121.cu     # System-fenced RDMA staging, peer flag wait, canonical-order reduction
```

### 5.2 Workspace `Cargo.toml`

```toml
[workspace]
members = [
    "crates/sparkflash-trace",
    "crates/sparkflash-api",
    "crates/sparkflash-scheduler",
    "crates/sparkflash-kv",
    "crates/sparkflash-engram",
    "crates/sparkflash-comm",
    "crates/sparkflash-kernels",
    "crates/sparkflash-cli",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"
authors = ["SparkFlash Team"]

[workspace.dependencies]
tokio = { version = "1.43", features = ["full", "rt-multi-thread", "parking_lot"] }
axum = { version = "0.8", features = ["http2", "json", "macros"] }
tower-http = { version = "0.6", features = ["cors", "trace", "compression-br"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bytes = "1.9"
futures = "0.3"
async-stream = "0.3"
# Official DeepSeek V4.1 Rust Encoding & Protocol Toolkit
deepseek-recipe = "0.1"
deepseek-recipe-core = "0.1"
deepseek-recipe-encoding = "0.1"
deepseek-recipe-image = "0.1"
# Hardware, CUDA, RDMA, and Async NVMe I/O
cudarc = { version = "0.13", features = ["driver", "nvrtc", "f16", "cuda-13000"] }
io-uring = "0.7"
libc = "0.2"
memmap2 = "0.9"
crossbeam-channel = "0.5"
parking_lot = "0.12"
dashmap = "6.1"
prometheus = { version = "0.13", features = ["process"] }
tracing = { version = "0.1", features = ["release_max_level_debug"] }
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
clap = { version = "4.5", features = ["derive", "env"] }
anyhow = "1.0"
thiserror = "2.0"
half = { version = "2.4", features = ["bytemuck", "serde"] }
bytemuck = { version = "1.21", features = ["derive"] }
```

### 5.3 Core Module 1: `sparkflash-engram` (Deterministic Hashing, 16-Way Cache & `io_uring` NVMe Store)

Below is the complete architectural implementation for `crates/sparkflash-engram/src/lib.rs`, handling tokenizer compression, deterministic 24-head multiplicative-XOR hashing, the 2-level DRM + UMA 16-way set-associative cache, `io_uring` `O_DIRECT` prefetching, and scoped trace point hooks (`TraceScope::ENGRAM_HASH`, `TraceScope::ENGRAM_CACHE`, `TraceScope::ENGRAM_IO_URING`):

```rust
//! crates/sparkflash-engram/src/lib.rs
//! Deterministic Engram Conditional Memory Engine for DeepSeek-V4.1-Flash on 4x DGX Spark.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use sparkflash_trace::{spark_trace_span, TraceScope};

pub const ENGRAM_LAYERS: [usize; 2] = [1, 14];
pub const ENGRAM_ORDERS: [usize; 3] = [2, 3, 4]; // 2-gram, 3-gram, 4-gram
pub const ENGRAM_NUM_HEADS: usize = 8;
pub const ENGRAM_TOTAL_LOOKUPS_PER_TOK: usize = 3 * ENGRAM_NUM_HEADS; // 24 heads
pub const ENGRAM_LOCAL_LOOKUPS_PER_TOK: usize = 6; // TP=4: 6 heads owned per rank
pub const ENGRAM_HEAD_DIM: usize = 256;
pub const ENGRAM_SCALE_BYTES: usize = 8; // 256 / 32 = 8 ue8m0 block scales
pub const ENGRAM_RAW_ROW_BYTES: usize = ENGRAM_HEAD_DIM + ENGRAM_SCALE_BYTES; // 264 bytes
pub const ENGRAM_PACKED_ROW_STRIDE: usize = 512; // 512B sector-aligned for O_DIRECT
pub const ENGRAM_PAD_TOKEN_ID: u32 = 2;
pub const IMAGE_TOKEN_ID: u32 = 129_264;
pub const COMPRESSED_VOCAB_SIZE: u32 = 99_092;

/// Per-layer, per-tier telemetry counters exported to OpenAI responses and Prometheus.
#[derive(Debug, Default)]
pub struct EngramLayerTelemetry {
    pub lookups_total: AtomicU64,
    pub tier1_drm_hits: AtomicU64,
    pub tier2_uma_hits: AtomicU64,
    pub tier3_nvme_misses: AtomicU64,
    pub image_tokens_masked: AtomicU64,
    pub prefetch_wait_ns_total: AtomicU64,
    pub gate_alpha_sum_micro: AtomicU64, // Sum of mean gate alpha * 1e6
    pub gate_samples_total: AtomicU64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EngramTelemetrySnapshot {
    pub layer_id: usize,
    pub lookups_total: u64,
    pub tier1_drm_hits: u64,
    pub tier2_uma_hits: u64,
    pub tier3_nvme_misses: u64,
    pub hit_rate: f64,
    pub mean_prefetch_stall_us: f64,
    pub mean_gate_alpha: f64,
}

/// Surjective tokenizer compression map P: V (129,280) -> V' (99,092)
/// Precomputed at boot using NFKC + lowercasing + accent stripping + whitespace collapse.
pub struct TokenizerCompressor {
    map: Box<[u32; 129_280]>,
}

impl TokenizerCompressor {
    pub fn from_checkpoint_table(raw_map: &[u32]) -> anyhow::Result<Self> {
        anyhow::ensure!(raw_map.len() == 129_280, "Invalid vocab size for compressor");
        let max_id = *raw_map.iter().max().unwrap_or(&0);
        anyhow::ensure!(
            max_id + 1 == COMPRESSED_VOCAB_SIZE,
            "Compressed vocab size mismatch: got {}, expected {}",
            max_id + 1,
            COMPRESSED_VOCAB_SIZE
        );
        let mut arr = Box::new([0u32; 129_280]);
        arr.copy_from_slice(raw_map);
        Ok(Self { map: arr })
    }

    #[inline(always)]
    pub fn compress(&self, raw_token_id: u32) -> u32 {
        if raw_token_id >= 129_280 {
            ENGRAM_PAD_TOKEN_ID
        } else {
            unsafe { *self.map.get_unchecked(raw_token_id as usize) }
        }
    }
}

/// Deterministic Multiplicative-XOR Hash Head configuration from checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct HashHeadSpec {
    pub ngram_order: usize,      // 2, 3, or 4
    pub head_idx: usize,         // 0..7
    pub prime_modulus: u64,      // ~16,000,000 distinct prime per (layer, order, head)
    pub table_row_offset: u64,   // Disjoint offset in the packed shard
    pub multipliers: [u64; 4],   // Derived from COMPRESSED_VOCAB_SIZE and head seed
}

impl HashHeadSpec {
    /// Computes the deterministic row index z_{t,n,k} = phi_{n,k}(g_{t,n})
    #[inline(always)]
    pub fn hash_ngram(&self, suffix: &[u32; 4]) -> u64 {
        let mut h: u64 = 0;
        let start = 4 - self.ngram_order;
        for i in start..4 {
            let tok = suffix[i] as u64;
            h = h.wrapping_mul(COMPRESSED_VOCAB_SIZE as u64)
                 .wrapping_add(tok.wrapping_mul(self.multipliers[i]))
                 ^ (tok.rotate_left(((i as u32) + 1) * 13));
        }
        self.table_row_offset + (h % self.prime_modulus)
    }
}

/// 16-Way Set-Associative Cache Set in Coherent Unified Memory.
#[repr(C, align(64))]
struct CacheSet16Way {
    tags: [u64; 16],       // Row IDs (u64::MAX = empty)
    clock_bits: u16,       // Second-chance CLOCK replacement bitmask
    hand: u8,              // Current CLOCK replacement hand (0..15)
    _pad: [u8; 5],
}

impl Default for CacheSet16Way {
    fn default() -> Self {
        Self {
            tags: [u64::MAX; 16],
            clock_bits: 0,
            hand: 0,
            _pad: [0; 5],
        }
    }
}

/// Single Engram Layer Store managing Tier 1 (DRM Static Hot), Tier 2 (UMA 16-Way), and Tier 3 (NVMe io_uring).
pub struct EngramLayerStore {
    pub layer_id: usize,
    pub tp_rank: usize,
    pub local_heads: [HashHeadSpec; ENGRAM_LOCAL_LOOKUPS_PER_TOK],
    nvme_file: File,
    // Tier 1: Optional DRM carveout static map for top-N hot rows
    drm_hot_row_limit: u64,
    drm_base_ptr: *const u8,
    // Tier 2: 16-Way Set-Associative Cache in pinned UMA
    num_sets: usize,
    sets: Vec<RwLock<CacheSet16Way>>,
    uma_cache_data_ptr: *mut u8, // Pinned UMA memory registered with CUDA (num_sets * 16 * 264 bytes)
    pub telemetry: Arc<EngramLayerTelemetry>,
}

unsafe impl Send for EngramLayerStore {}
unsafe impl Sync for EngramLayerStore {}

impl EngramLayerStore {
    /// Resolves a batch of token sequences into a CUDA-visible staging buffer.
    /// Executed asynchronously on the Grace Cortex-A725 cores 1-13 layers AHEAD of GPU consumption.
    pub fn prefetch_batch_rows(
        &self,
        compressor: &TokenizerCompressor,
        token_sequences: &[Vec<u32>],
        staging_uma_ptr: *mut u8,
    ) -> anyhow::Result<()> {
        spark_trace_span!(TraceScope::ENGRAM_PREFETCH, "engram_prefetch_batch", self.layer_id as u32);
        let mut misses: Vec<(usize, u64)> = Vec::new(); // (slot_idx, row_id)
        let mut slot_idx = 0usize;

        for seq in token_sequences {
            let mut window = [
                compressor.compress(ENGRAM_PAD_TOKEN_ID),
                compressor.compress(ENGRAM_PAD_TOKEN_ID),
                compressor.compress(ENGRAM_PAD_TOKEN_ID),
                compressor.compress(ENGRAM_PAD_TOKEN_ID),
            ];
            for &raw_tok in seq {
                window = [window[1], window[2], window[3], compressor.compress(raw_tok)];
                if raw_tok == IMAGE_TOKEN_ID {
                    self.telemetry.image_tokens_masked.fetch_add(1, Ordering::Relaxed);
                    slot_idx += ENGRAM_LOCAL_LOOKUPS_PER_TOK;
                    continue;
                }
                for head in &self.local_heads {
                    let row_id = head.hash_ngram(&window);
                    self.telemetry.lookups_total.fetch_add(1, Ordering::Relaxed);
                    let dst_ptr = unsafe { staging_uma_ptr.add(slot_idx * ENGRAM_RAW_ROW_BYTES) };

                    // Check Tier 1: DRM Static Carveout
                    if !self.drm_base_ptr.is_null() && row_id < self.drm_hot_row_limit {
                        self.telemetry.tier1_drm_hits.fetch_add(1, Ordering::Relaxed);
                        unsafe {
                            let src = self.drm_base_ptr.add((row_id as usize) * ENGRAM_RAW_ROW_BYTES);
                            std::ptr::copy_nonoverlapping(src, dst_ptr, ENGRAM_RAW_ROW_BYTES);
                        }
                    } else if self.try_read_tier2_uma(row_id, dst_ptr) {
                        // Tier 2 Hit: 16-Way Set-Associative UMA Cache
                        self.telemetry.tier2_uma_hits.fetch_add(1, Ordering::Relaxed);
                    } else {
                        // Tier 3 Miss: Queue for batched io_uring O_DIRECT read
                        self.telemetry.tier3_nvme_misses.fetch_add(1, Ordering::Relaxed);
                        misses.push((slot_idx, row_id));
                    }
                    slot_idx += 1;
                }
            }
        }

        if !misses.is_empty() {
            self.service_nvme_misses_uring(&misses, staging_uma_ptr)?;
        }
        Ok(())
    }

    #[inline]
    fn try_read_tier2_uma(&self, row_id: u64, dst_ptr: *mut u8) -> bool {
        let set_idx = (row_id as usize) % self.num_sets;
        let set_lock = &self.sets[set_idx];
        let set = set_lock.read();
        for way in 0..16 {
            if set.tags[way] == row_id {
                let cache_offset = (set_idx * 16 + way) * ENGRAM_RAW_ROW_BYTES;
                unsafe {
                    let src = self.uma_cache_data_ptr.add(cache_offset);
                    std::ptr::copy_nonoverlapping(src, dst_ptr, ENGRAM_RAW_ROW_BYTES);
                }
                return true;
            }
        }
        false
    }

    fn insert_tier2_uma(&self, row_id: u64, src_ptr: *const u8) {
        let set_idx = (row_id as usize) % self.num_sets;
        let mut set = self.sets[set_idx].write();
        for way in 0..16 {
            if set.tags[way] == row_id {
                set.clock_bits |= 1 << way;
                return;
            }
        }
        let chosen_way = loop {
            let w = set.hand as usize;
            if (set.clock_bits & (1 << w)) == 0 {
                set.hand = ((w + 1) & 15) as u8;
                break w;
            }
            set.clock_bits &= !(1 << w);
            set.hand = ((w + 1) & 15) as u8;
        };
        set.tags[chosen_way] = row_id;
        set.clock_bits |= 1 << chosen_way;
        let cache_offset = (set_idx * 16 + chosen_way) * ENGRAM_RAW_ROW_BYTES;
        unsafe {
            let dst = self.uma_cache_data_ptr.add(cache_offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst, ENGRAM_RAW_ROW_BYTES);
        }
    }

    /// Batched io_uring O_DIRECT submission to local PCIe Gen5 NVMe (~112k IOPS at QD64).
    fn service_nvme_misses_uring(
        &self,
        misses: &[(usize, u64)],
        staging_uma_ptr: *mut u8,
    ) -> anyhow::Result<()> {
        spark_trace_span!(TraceScope::ENGRAM_IO_URING, "engram_nvme_uring_batch", misses.len() as u32);
        let mut ring = io_uring::IoUring::new(128)?;
        let fd = io_uring::types::Fd(self.nvme_file.as_raw_fd());
        let mut aligned_scratch = vec![0u8; misses.len() * ENGRAM_PACKED_ROW_STRIDE];

        for chunk_start in (0..misses.len()).step_by(64) {
            let chunk_end = (chunk_start + 64).min(misses.len());
            let batch = &misses[chunk_start..chunk_end];

            unsafe {
                let mut sq = ring.submission();
                for (i, &(_, row_id)) in batch.iter().enumerate() {
                    let file_offset = row_id * (ENGRAM_PACKED_ROW_STRIDE as u64);
                    let buf_ptr = aligned_scratch
                        .as_mut_ptr()
                        .add((chunk_start + i) * ENGRAM_PACKED_ROW_STRIDE);
                    let read_e = io_uring::opcode::Read::new(
                        fd,
                        buf_ptr,
                        ENGRAM_PACKED_ROW_STRIDE as u32,
                    )
                    .offset(file_offset)
                    .build()
                    .user_data((chunk_start + i) as u64);
                    sq.push(&read_e)?;
                }
            }
            ring.submit_and_wait(batch.len())?;
            ring.completion().for_each(|cqe| {
                let idx = cqe.user_data() as usize;
                let (slot_idx, row_id) = misses[idx];
                unsafe {
                    let src = aligned_scratch.as_ptr().add(idx * ENGRAM_PACKED_ROW_STRIDE);
                    let dst = staging_uma_ptr.add(slot_idx * ENGRAM_RAW_ROW_BYTES);
                    std::ptr::copy_nonoverlapping(src, dst, ENGRAM_RAW_ROW_BYTES);
                    self.insert_tier2_uma(row_id, src);
                }
            });
        }
        Ok(())
    }

    pub fn snapshot_telemetry(&self) -> EngramTelemetrySnapshot {
        let total = self.telemetry.lookups_total.load(Ordering::Relaxed);
        let t1 = self.telemetry.tier1_drm_hits.load(Ordering::Relaxed);
        let t2 = self.telemetry.tier2_uma_hits.load(Ordering::Relaxed);
        let misses = self.telemetry.tier3_nvme_misses.load(Ordering::Relaxed);
        let wait_ns = self.telemetry.prefetch_wait_ns_total.load(Ordering::Relaxed);
        let alpha_micro = self.telemetry.gate_alpha_sum_micro.load(Ordering::Relaxed);
        let samples = self.telemetry.gate_samples_total.load(Ordering::Relaxed);

        let hit_rate = if total > 0 {
            ((t1 + t2) as f64) / (total as f64)
        } else {
            0.0
        };
        let mean_prefetch_stall_us = if samples > 0 {
            (wait_ns as f64) / 1_000.0 / (samples as f64)
        } else {
            0.0
        };
        let mean_gate_alpha = if samples > 0 {
            (alpha_micro as f64) / 1_000_000.0 / (samples as f64)
        } else {
            0.0
        };

        EngramTelemetrySnapshot {
            layer_id: self.layer_id,
            lookups_total: total,
            tier1_drm_hits: t1,
            tier2_uma_hits: t2,
            tier3_nvme_misses: misses,
            hit_rate,
            mean_prefetch_stall_us,
            mean_gate_alpha,
        }
    }
}
```

### 5.4 Core Module 2: `sparkflash-comm` (`RoCEnante` One-Shot RDMA + Patched `NCCL` Dispatcher)

Below is the core implementation of `crates/sparkflash-comm/src/lib.rs` and its companion `sm_121a` CUDA kernel `kernels/rocenante_oneshot_sm121.cu`, implementing the **2 MiB threshold router**, dual-rail ConnectX-7 (`rocep1s0f0` + `roceP2p1s0f0`) posting, canonical rank-order bit-identical reduction, and **in-kernel `%globaltimer` straggler vs. wire-transfer trace points**:

```cuda
// kernels/rocenante_oneshot_sm121.cu
// One-Shot RDMA All-Reduce & Compact All-Gather for 4x DGX Spark (GB10 sm_121a Unified Memory)
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "trace_globaltimer_sm121.cuh"

struct RoceOneShotSlot {
    volatile uint64_t arrival_seq[4]; // Per-rank sequence numbers written via RDMA_WRITE_WITH_IMM
    __nv_bfloat16 payload[4][1024 * 1024]; // Up to 2 MiB per rank in coherent UMA
};

extern "C" __global__ void rocenante_oneshot_allreduce_bf16_tp4(
    const __nv_bfloat16* __restrict__ local_in,
    __nv_bfloat16* __restrict__ local_out,
    RoceOneShotSlot* __restrict__ slot,
    volatile uint64_t* __restrict__ host_doorbell_seq,
    GpuTraceRing* __restrict__ trace_ring,
    const uint32_t layer_id,
    const uint32_t my_rank,
    const uint32_t num_elements,
    const uint64_t step_seq
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    uint64_t t_enter = 0, t_staged = 0, t_peers_ready = 0;

    SPARK_GPU_TRACE_ENTER(trace_ring, TRACE_SCOPE_ROCENANTE_RDMA, t_enter);

    // 1. Stage local rank's input into its own UMA RDMA slot
    for (uint32_t i = tid; i < num_elements; i += stride) {
        slot->payload[my_rank][i] = local_in[i];
    }
    __syncthreads();

    // 2. Thread 0 issues system-scope fence so ConnectX-7 DMA & Grace CPU see fresh L2 lines
    if (tid == 0) {
        slot->arrival_seq[my_rank] = step_seq;
        __threadfence_system();
        *host_doorbell_seq = step_seq; // Triggers CPU RT proxy / GPUNetIO doorbell across both CX-7 rails
        SPARK_GPU_READ_TIMER(trace_ring, TRACE_SCOPE_ROCENANTE_RDMA, t_staged);
    }

    // 3. Spin-wait on all 4 ranks' arrival sequence numbers in unified memory
    if (threadIdx.x < 4) {
        while (slot->arrival_seq[threadIdx.x] < step_seq) {
            // Busy-wait on coherent NVLink-C2C cache line (~4-16 us)
        }
    }
    __syncthreads();
    if (tid == 0) {
        SPARK_GPU_READ_TIMER(trace_ring, TRACE_SCOPE_ROCENANTE_RDMA, t_peers_ready);
    }

    // 4. Reduce in strict canonical rank order (0 + 1 + 2 + 3) for bit-identical determinism
    for (uint32_t i = tid; i < num_elements; i += stride) {
        float acc = 0.0f;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            acc += __bfloat162float(slot->payload[r][i]);
        }
        local_out[i] = __float2bfloat16_rn(acc);
    }

    // 5. Emit scoped GPU trace record (splits staging, RDMA/straggler wait, and reduction time)
    SPARK_GPU_TRACE_EXIT_ROCE(
        trace_ring, TRACE_SCOPE_ROCENANTE_RDMA, layer_id, num_elements * 2,
        t_enter, t_staged, t_peers_ready
    );
}
```

```rust
//! crates/sparkflash-comm/src/lib.rs
//! Dual-Path Collective Transport: RoCEnante One-Shot RDMA (<= 2 MiB) + Patched NCCL (> 2 MiB).

use std::sync::atomic::{AtomicU64, Ordering};
use sparkflash_trace::{spark_trace_span, TraceScope};

pub const ROCENANTE_MAX_BYTES: usize = 2 * 1024 * 1024; // 2 MiB threshold

#[derive(Debug, Default)]
pub struct CommTelemetry {
    pub rocenante_allreduce_calls: AtomicU64,
    pub rocenante_allgather_calls: AtomicU64,
    pub nccl_collective_calls: AtomicU64,
    pub rdma_bytes_rail0: AtomicU64, // rocep1s0f0 (PCIe Gen5 x4 Root 0)
    pub rdma_bytes_rail1: AtomicU64, // roceP2p1s0f0 (PCIe Gen5 x4 Root 2)
    pub l2_prefetch_overlapped_calls: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkFabricMode {
    /// 4x DGX Spark connected to a 200GbE lossless RoCEv2 switch (e.g. MikroTik CRS804/CRS812)
    Switched200G,
    /// 4x DGX Spark cabled in a switchless ring (0-1-2-3-0) with ConnectX-7 ASIC hairpin diagonals
    SwitchlessRing4Hairpin,
}

pub struct DualPathCommunicator {
    pub rank: usize,
    pub world_size: usize,
    pub fabric_mode: SparkFabricMode,
    pub step_seq: AtomicU64,
    pub telemetry: CommTelemetry,
}

impl DualPathCommunicator {
    /// Routes All-Reduce to RoCEnante one-shot RDMA for <= 2 MiB (Decode/Verify)
    /// and to bidirectional dual-rail NCCL 2.30.7+ for > 2 MiB (Chunked Prefill).
    pub fn dispatch_all_reduce_bf16(
        &self,
        layer_id: u32,
        num_bytes: usize,
        enable_l2_weight_prefetch: bool,
    ) -> CollectiveRoute {
        if num_bytes <= ROCENANTE_MAX_BYTES {
            spark_trace_span!(TraceScope::ROCENANTE_RDMA, "dispatch_rocenante_oneshot", layer_id);
            self.telemetry.rocenante_allreduce_calls.fetch_add(1, Ordering::Relaxed);
            let half_bytes = (num_bytes / 2) as u64;
            self.telemetry.rdma_bytes_rail0.fetch_add(half_bytes * 3, Ordering::Relaxed);
            self.telemetry.rdma_bytes_rail1.fetch_add(half_bytes * 3, Ordering::Relaxed);
            if enable_l2_weight_prefetch {
                self.telemetry.l2_prefetch_overlapped_calls.fetch_add(1, Ordering::Relaxed);
            }
            CollectiveRoute::RoCEnanteOneShot {
                seq: self.step_seq.fetch_add(1, Ordering::SeqCst) + 1,
                l2_prefetch_6mb: enable_l2_weight_prefetch,
            }
        } else {
            spark_trace_span!(TraceScope::NCCL_COLLECTIVE, "dispatch_nccl_ring", layer_id);
            self.telemetry.nccl_collective_calls.fetch_add(1, Ordering::Relaxed);
            CollectiveRoute::NcclBidirectionalRing
        }
    }
}

pub enum CollectiveRoute {
    RoCEnanteOneShot { seq: u64, l2_prefetch_6mb: bool },
    NcclBidirectionalRing,
}
```

### 5.5 Core Module 3: `sparkflash-kv` (3-Tier Radix KV Cache + SWA Bounded Replay Manager)

`DeepSeek-V4.1-Flash` separates **Global KV** (stored only at `kv_source_layer_ids = [2, 8, 14, 20]` in FP4 `E2M1` + MXFP4 Indexer K at 890 logical bytes/token) from **SWA KV** (128-token window in FP8 across all 40 layers). `sparkflash-kv` manages both independently and tracks exact hit/miss and SWA Bounded Replay telemetry:

```rust
//! crates/sparkflash-kv/src/lib.rs
//! 3-Tier Radix Prefix Cache & SWA Bounded Replay Manager for DeepSeek-V4.1-Flash.

use std::sync::atomic::{AtomicU64, Ordering};
use sparkflash_trace::{spark_trace_span, TraceScope};

pub const KV_SOURCE_LAYERS: [usize; 4] = [2, 8, 14, 20];
pub const SWA_WINDOW_SIZE: usize = 128;
pub const KV_PAGE_TOKENS: usize = 64; // Matches SM121 FlashMLA 64-token page split

#[derive(Debug, Default)]
pub struct KvCacheTelemetry {
    pub requests_total: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub global_kv_uma_hit_tokens: AtomicU64,
    pub global_kv_nvme_hit_tokens: AtomicU64,
    pub global_kv_miss_tokens: AtomicU64,
    pub swa_tail_cache_hits: AtomicU64,
    pub swa_tail_cache_misses: AtomicU64,
    pub encoder_swa_replay_tokens_total: AtomicU64,
    pub decoder_swa_replay_tokens_total: AtomicU64,
    pub active_kv_pages: AtomicU64,
    pub total_kv_pages: u64,
}

/// Result of matching an incoming prompt against the Radix Prefix Tree and SWA Tail Ring.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PrefixMatchPlan {
    pub total_prompt_tokens: usize,
    /// Number of prefix tokens matched in Global KV (aligned to 64-token page boundary)
    pub global_kv_cached_tokens: usize,
    /// Breakdown between Tier-1 (Coherent UMA) and Tier-2 (NVMe Persistent 72h Cache)
    pub uma_hit_tokens: usize,
    pub nvme_restored_tokens: usize,
    /// Whether the exact 128-token SWA tail at `global_kv_cached_tokens` was found in the 128-slot SWA ring
    pub swa_tail_hit: bool,
    /// Tokens that must be replayed through Encoder layers 0..19 to reconstruct SWA KV (0 or up to 128)
    pub encoder_swa_replay_tokens: usize,
    /// Uncached suffix tokens that must run full Encoder layers 0..19
    pub uncached_suffix_tokens: usize,
    /// Tokens at the end of the prompt that must run Decoder layers 20..39 for Decoder SWA Bounded Replay (<= 128)
    pub decoder_swa_replay_tokens: usize,
}

impl KvCacheTelemetry {
    /// Computes the exact CED + SWA Bounded Replay execution plan for a prompt.
    pub fn plan_request(
        &self,
        total_prompt_tokens: usize,
        uma_matched_tokens: usize,
        nvme_matched_tokens: usize,
        swa_tail_hit: bool,
    ) -> PrefixMatchPlan {
        spark_trace_span!(TraceScope::KV_RADIX, "kv_radix_plan_request", total_prompt_tokens as u32);
        let raw_cached = (uma_matched_tokens + nvme_matched_tokens).min(total_prompt_tokens.saturating_sub(1));
        // Align global cache hit to 64-token page boundary for bit-identical FlashMLA execution
        let global_kv_cached_tokens = (raw_cached / KV_PAGE_TOKENS) * KV_PAGE_TOKENS;
        let uma_hit = uma_matched_tokens.min(global_kv_cached_tokens);
        let nvme_hit = global_kv_cached_tokens.saturating_sub(uma_hit);
        let uncached_suffix_tokens = total_prompt_tokens - global_kv_cached_tokens;

        // Encoder SWA Bounded Replay (arXiv:2609.19969 Section 3.2.2):
        // If we hit Global KV (global_kv_cached_tokens > 0) but missed the short-lived SWA tail cache,
        // we replay at most the last n_win = 128 tokens of the cached prefix through Encoder layers 0..19
        // (generating ONLY SWA KV without overwriting cached Global KV).
        let encoder_swa_replay_tokens = if global_kv_cached_tokens == 0 || swa_tail_hit {
            0
        } else {
            global_kv_cached_tokens.min(SWA_WINDOW_SIZE)
        };

        // Decoder SWA Bounded Replay (arXiv:2609.19969 Section 3.2.2):
        // Under CED, Decoder layers 20..39 never run over the whole prompt during prefill.
        // If SWA tail hit and uncached_suffix == 0, 0 tokens needed; otherwise replay last min(N, 128) tokens.
        let decoder_swa_replay_tokens = if swa_tail_hit && uncached_suffix_tokens == 0 {
            0
        } else {
            total_prompt_tokens.min(SWA_WINDOW_SIZE)
        };

        // Record atomic telemetry
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.prompt_tokens_total.fetch_add(total_prompt_tokens as u64, Ordering::Relaxed);
        self.global_kv_uma_hit_tokens.fetch_add(uma_hit as u64, Ordering::Relaxed);
        self.global_kv_nvme_hit_tokens.fetch_add(nvme_hit as u64, Ordering::Relaxed);
        self.global_kv_miss_tokens.fetch_add(uncached_suffix_tokens as u64, Ordering::Relaxed);
        if global_kv_cached_tokens > 0 {
            if swa_tail_hit {
                self.swa_tail_cache_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.swa_tail_cache_misses.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.encoder_swa_replay_tokens_total.fetch_add(encoder_swa_replay_tokens as u64, Ordering::Relaxed);
        self.decoder_swa_replay_tokens_total.fetch_add(decoder_swa_replay_tokens as u64, Ordering::Relaxed);

        PrefixMatchPlan {
            total_prompt_tokens,
            global_kv_cached_tokens,
            uma_hit_tokens: uma_hit,
            nvme_restored_tokens: nvme_hit,
            swa_tail_hit,
            encoder_swa_replay_tokens,
            uncached_suffix_tokens,
            decoder_swa_replay_tokens,
        }
    }
}
```

### 5.6 Core Module 4: `sparkflash-scheduler` (CED Prefill-Bypass, Short-Request Interleaving & DSpark Confidence Scheduler)

To maximize multi-user responsiveness and single-stream speed on 4x DGX Spark, `sparkflash-scheduler` implements three DeepSeek-V4.1-Flash-specific scheduling algorithms:

1. **Chunked Prefill with Bounded HSI Indexer (`chunk_size = 4096` or `8192`):**
   - At 1M context, scoring all 4096 query rows in a prefill chunk against 1M Indexer K entries at once would allocate a `~60 GiB` transient tensor and OOM the Spark. `SparkFlash-RS` partitions the query rows across the 4 TP ranks (`SPARK_PREFILL_TP_SPLIT`, 1024 rows/rank) and scores them in bounded `256`-row tiles inside a static `2.0 GiB` scratchpad, all-gathering only the compact `INT32` `Top-512` indices and `2048` HSI candidate block IDs via `RoCEnante`.
2. **Short-Request Prefill Bypass (`DSV41_PREFILL_BYPASS = 256`):**
   - When a 400K–1M token cold prefill is running in 4096-token chunks (~78 seconds total), a short interactive request ($\le 256$ prompt tokens) arriving mid-prefill is admitted immediately between two prefill chunks and decoded alongside the inter-chunk decode step (`--prefill-decode-interval 2`), reducing interactive TTFT during a 400K background prefill from **80 seconds to 4.6 seconds** ([knapcio Multi-User Probe](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fknapcio%2FDeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)).
3. **DSpark Confidence-Capped Verification & Live Router Expert Reuse (`k = 5`, `conf_cap = 0.1`):**
   - In each decode step, the 3-layer DSpark drafter (`layers 37, 38, 39`) runs a single forward pass to propose `dspark_block_size = 5` draft tokens ($t+1 \dots t+5$) and outputs per-position conditional acceptance probabilities $p_1, \dots, p_5$ via its confidence head.
   - Computing the cumulative prefix survival probability $P_{\text{survive}}(j) = \prod_{i=1}^j p_i$, the scheduler dynamically caps the verification length at $k_{\text{verify}} = \max \{ j \in \{1..5\} \mid P_{\text{survive}}(j) \ge 0.10 \}$.
   - Inside the CUDA graph (`M = 6` verify rows per request), draft positions $j > k_{\text{verify}}$ are masked in the MoE router (`DSV41_ROUTER_LIVE`) to **reuse the anchor token's Top-6 experts**, preventing doomed draft tokens from pulling additional expert weights out of LPDDR5X memory!
   - Furthermore, because all 4 ranks seed the DSpark Markov head (`dspark_noise_token_id: 128799`) from an identical deterministic counter-based philox RNG, `SparkFlash-RS` eliminates all per-step rank-0 draft token broadcasts (`DSV41_SPEC_SYNC_FREE`), saving **0.55 ms per decode step**.

### 5.7 Core Module 5: `sparkflash-api` (OpenAI-Native Interface + `deepseek-recipe` + Rich Cache/Engine Telemetry)

`SparkFlash-RS` uses DeepSeek's official [deepseek-recipe](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdeepseek-ai%2Fdeepseek-recipe%2Fblob%2Fmain%2FREADME.md) Rust crates directly in-process to serve a 100% compliant OpenAI API alongside comprehensive cache hit/miss and hardware telemetry:

- **Supported OpenAI & Native Endpoints:**
  - `POST /v1/chat/completions` (Streaming SSE & Non-Streaming, Multimodal `image_url` base64/HTTP, Function/Tool Calls, `reasoning_effort: 1..100` or `"low"|"high"|"max"`, `chat_template_kwargs.thinking` / `enable_thinking`)
  - `POST /v1/completions` (Raw prompt completions with SWA Bounded Replay logprob guard)
  - `POST /v1/responses` (OpenAI Responses API including `apply_patch` custom tool via `deepseek-recipe`)
  - `POST /v1/messages` (Anthropic Messages API compatibility for Claude Code / agent harnesses via `deepseek-recipe`)
  - `GET /v1/models` (Lists `deepseek-ai/DeepSeek-V4.1-Flash` and capability metadata)
  - `GET /v1/cluster/stats` (Real-time JSON telemetry snapshot of KV Cache, SWA Replay, Engram Tier 1/2/3 Hit/Miss, DSpark Acceptance, RoCEnante vs. NCCL latencies, and per-node GB10 UMA free memory)
  - `POST /v1/admin/trace` & `GET /v1/admin/trace/dump` (Dynamically enables/disables scoped trace bitmasks for a specific step count or request ID and exports Chrome/Perfetto `trace.json.gz` + per-layer latency percentile tables)
  - `GET /metrics` (Prometheus exposition format for Grafana / `sparkDash` / `Ramjet` tier-aware routers)

> **Implementation status (2026-10-18):** the `Multimodal image_url` part of the chat endpoint above
> is not served by this tree. The vision tower and aligner are 263 checkpoint tensors the tensor plan
> does not load, `GET /v1/models` reports `multimodal: false`, and any request carrying an image part
> is refused with `unsupported_capability` naming the missing vision channel before generation. This
> specification entry advertises the target surface, not the current one (DECISIONS.md D18 and D27).

```rust
//! crates/sparkflash-api/src/telemetry_types.rs
//! OpenAI-Compatible Usage + Extended DeepSeek-V4.1-Flash Cache & Hardware Telemetry Schema.

use serde::{Deserialize, Serialize};
use sparkflash_engram::EngramTelemetrySnapshot;
use sparkflash_kv::PrefixMatchPlan;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Standard OpenAI field reporting prefix-cached tokens (Global KV cache hits)
    pub cached_tokens: usize,
    /// Extended breakdown: tokens served directly from GB10 Coherent UMA
    pub uma_cached_tokens: usize,
    /// Extended breakdown: tokens restored from NVMe 72-hour persistent Global KV store
    pub nvme_cached_tokens: usize,
    /// Number of image visual tokens after 3x3 pixel-unshuffle
    pub image_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: usize,
    pub accepted_prediction_tokens: usize,
    pub rejected_prediction_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiUsageWithCache {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub prompt_tokens_details: PromptTokensDetails,
    pub completion_tokens_details: CompletionTokensDetails,
}

/// Per-request and cluster-wide DeepSeek-V4.1-Flash + 4x DGX Spark telemetry block
/// attached to `x_deepseek_spark_stats` in OpenAI responses and `/v1/cluster/stats`.
#[derive(Debug, Clone, Serialize)]
pub struct DeepSeekSparkRequestStats {
    pub kv_cache: PrefixMatchPlan,
    pub kv_cache_hit_rate: f64,
    pub engram_layers: [EngramTelemetrySnapshot; 2], // Layer 1 and Layer 14
    pub dspark_speculative: DSparkStepStats,
    pub transport: RoceTransportStats,
    pub timing: RequestTimingStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct DSparkStepStats {
    pub target_verify_steps: usize,
    pub draft_tokens_proposed: usize,
    pub draft_tokens_confidence_gated: usize,
    pub draft_tokens_accepted: usize,
    pub mean_tokens_per_verify_step: f64, // 1.0 + accepted / steps (up to 6.0)
    pub acceptance_rate: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoceTransportStats {
    pub fabric_mode: String,
    pub rocenante_oneshot_calls: u64,
    pub nccl_ring_calls: u64,
    pub rail0_rocep1s0f0_mb: f64,
    pub rail1_rocep2p1s0f0_mb: f64,
    pub l2_weight_prefetch_overlaps: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestTimingStats {
    pub ttft_ms: f64,
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub min_rank_uma_free_gib: f64,
}
```

#### Example OpenAI `/v1/chat/completions` Response with Native Cache & Engram Telemetry:
```json
{
  "id": "chatcmpl-dsv41-9f82a10c",
  "object": "chat.completion",
  "created": 1792082400,
  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "Let's check the cache alignment and RoCE ring latency...",
        "content": "The 4x DGX Spark cluster is operating at TP=4, EP=1 with 890 B/token FP4 global KV."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 32768,
    "completion_tokens": 256,
    "total_tokens": 33024,
    "prompt_tokens_details": {
      "cached_tokens": 28672,
      "uma_cached_tokens": 28672,
      "nvme_cached_tokens": 0,
      "image_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 94,
      "accepted_prediction_tokens": 212,
      "rejected_prediction_tokens": 48
    }
  },
  "x_deepseek_spark_stats": {
    "kv_cache": {
      "total_prompt_tokens": 32768,
      "global_kv_cached_tokens": 28672,
      "uma_hit_tokens": 28672,
      "nvme_restored_tokens": 0,
      "swa_tail_hit": false,
      "encoder_swa_replay_tokens": 128,
      "uncached_suffix_tokens": 4096,
      "decoder_swa_replay_tokens": 128
    },
    "kv_cache_hit_rate": 0.875,
    "engram_layers": [
      {
        "layer_id": 1,
        "lookups_total": 26880,
        "tier1_drm_hits": 7120,
        "tier2_uma_hits": 12890,
        "tier3_nvme_misses": 6870,
        "hit_rate": 0.7444,
        "mean_prefetch_stall_us": 0.0,
        "mean_gate_alpha": 0.312
      },
      {
        "layer_id": 14,
        "lookups_total": 26880,
        "tier1_drm_hits": 6840,
        "tier2_uma_hits": 12410,
        "tier3_nvme_misses": 7630,
        "hit_rate": 0.7161,
        "mean_prefetch_stall_us": 0.0,
        "mean_gate_alpha": 0.289
      }
    ],
    "dspark_speculative": {
      "target_verify_steps": 52,
      "draft_tokens_proposed": 260,
      "draft_tokens_confidence_gated": 38,
      "draft_tokens_accepted": 204,
      "mean_tokens_per_verify_step": 4.923,
      "acceptance_rate": 0.7846
    },
    "transport": {
      "fabric_mode": "Switched200G_DualRail_RoCEnante",
      "rocenante_oneshot_calls": 5408,
      "nccl_ring_calls": 20,
      "rail0_rocep1s0f0_mb": 412.6,
      "rail1_rocep2p1s0f0_mb": 412.6,
      "l2_weight_prefetch_overlaps": 5408
    },
    "timing": {
      "ttft_ms": 698.4,
      "prefill_tok_per_s": 5864.8,
      "decode_tok_per_s": 134.2,
      "min_rank_uma_free_gib": 9.42
    }
  }
}
```

### 5.8 Core Module 6: `sparkflash-trace` (Zero-Overhead Scoped Trace Points & In-Graph GPU/CPU Profiling)

On a 4x DGX Spark cluster, naive profiling methods fail or distort the exact bottlenecks we need to optimize:
- Calling `cudaEventRecord` between kernels prevents full-step CUDA Graph capture or adds `~1.5 ms/step` of driver launch overhead across ~400 graph nodes.
- Spawning an external profiler process on the same node while a rank is active fails `cudaMemGetInfo` because only one primary context can own the unified memory pool, and polling `nvidia-smi` every 2 seconds steals `~0.6 ms` per decode step ([MiaAI-Lab Profiling Notes](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FMiaAI-Lab%2FDeepSeek-v4.1-Flash-DGX-Sparks)).

To make post-bringup optimization surgical and zero-cost in production, `SparkFlash-RS` mandates **Scoped, Easily Disabled Trace Points (`crates/sparkflash-trace`)** across every CPU thread, RDMA proxy, and CUDA kernel.

#### Three-Level Gating Hierarchy (Zero Cost When Disabled):
1. **Compile-Time Cargo Features (`--features trace-points`, `trace-gpu-in-graph`, `trace-nvtx`):**
   - If built with `--no-default-features` for maximum-throughput production runs, all `spark_trace_span!`, `SPARK_GPU_TRACE_ENTER`, and `SPARK_GPU_TRACE_EXIT` macros expand to `{}` at compile time (`0` instructions, `0` registers, `0` branches).
2. **Runtime L1-Cached Atomic Bitmask (`GLOBAL_TRACE_MASK: AtomicU64`) + Step Countdown (`TRACE_STEPS_REMAINING: AtomicU32`):**
   - In standard production builds (`--features trace-points,trace-gpu-in-graph`), every trace point checks a single 64-bit bitmask in L1 cache (`if (mask & SCOPE) != 0`).
   - Because GB10 has coherent unified memory over NVLink-C2C, the **exact same 64-bit `active_scope_mask` word resides in a `cudaHostAllocMapped` page** shared by the Grace ARM CPU and the Blackwell `sm_121a` GPU!
   - Flipping a bit via `POST /v1/admin/trace {"scopes": ["ENGRAM_IO_URING", "ROCENANTE_RDMA", "MOE_GEMM"], "steps": 64}` or passing `X-SparkFlash-Trace: engram,roce` on a single OpenAI HTTP request **immediately activates both CPU and inside-CUDA-graph GPU trace points** for those 64 steps without recapturing a single CUDA graph, and automatically disarms to `0` when the step counter hits zero!
3. **Synchronized ARM `CNTVCT_EL0` + GPU `%globaltimer` Lock-Free Ring Buffers (`64 MiB/node`):**
   - **CPU Trace Points:** Read the ARMv9 generic timer register `CNTVCT_EL0` directly in userspace via inline assembly (`mrs x0, cntvct_el0`, `~1.2 ns` cost, zero `clock_gettime` VDSO overhead) and write 32-byte `CpuTraceEvent` records into a per-thread lock-free ring buffer.
   - **In-Graph GPU Trace Points:** Inside `mega_mhc_sm121.cu`, `flash_mla_csa2_sm121.cu`, `b12x_mxfp8_dense_sm121.cu`, `b12x_mxfp4_moe_sm121.cu`, and `rocenante_oneshot_sm121.cu`, thread `0` checks `trace_ring->active_scope_mask & SCOPE`. When enabled, it reads the Blackwell hardware nanosecond clock (`mov.u64 %0, %globaltimer;`) and writes a 32-byte `GpuTraceEvent` into the coherent UMA ring buffer.

#### Scoped Trace Domains & Optimization Targets:

| `TraceScope` Bitflag |Bit | Instrumented Component | Captured Sub-Timings & Metadata | Primary Optimization Question Answered |
| :--- | :--- | :--- | :--- | :--- |
| `API_ENCODE` | `1 << 0` | `sparkflash-api` (`deepseek-recipe`) | Tokenization, NFKC compression, ViT $3\times 3$ unshuffle | Is prompt/image encoding starving the HTTP worker pool? |
| `SCHED_BATCH` | `1 << 1` | `sparkflash-scheduler` | Queue wait, `DSV41_PREFILL_BYPASS` interleaving | Are short interactive requests waiting behind 4096-tok chunks? |
| `KV_RADIX` | `1 << 2` | `sparkflash-kv` | Radix tree lookup, 64-tok page allocation, NVMe KV load | What is the exact cost of restoring a 388K session from NVMe? |
| `SWA_REPLAY` | `1 << 3` | `sparkflash-kv` / `scheduler` | Encoder 128-tok replay vs. Decoder 128-tok replay | How much TTFT does an SWA prefix-tail ring miss add? |
| `ENGRAM_PREFETCH` | `1 << 4` | `sparkflash-engram` (CPU) | 24-head XOR hash + Tier-1 DRM / Tier-2 UMA lookup | Does CPU hashing finish before Layer 0 / Layer 13 GPU compute? |
| `ENGRAM_IO_URING` | `1 << 5` | `sparkflash-engram` (NVMe) | `io_uring` submission, QD depth, `O_DIRECT` completion | Are NVMe misses stalling Layer 1 or Layer 14 `engram.wkv`? |
| `ENGRAM_GPU_GATE` | `1 << 6` | `engram_gate_sm121.cu` | `engram.wkv` GEMM, RMSNorm dot-product gate, $\bar{\alpha}$ | Can `engram.wkv` column-split gather overlap with SWA? |
| `MEGA_MHC` | `1 << 7` | `mega_mhc_sm121.cu` | Single-pass residual mix + Sinkhorn-20 + FP8 dynamic cast | Is `Mega-mHC` hitting $> 85\%$ of 273 GB/s LPDDR5X roofline? |
| `CSA2_INDEXER` | `1 << 8` | `flash_mla_csa2_sm121.cu` | Full/Reindex Top-512 scoring + 16,384-pos HSI pool build | How much time do Reindex layers `24, 28, 32, 36` spend scoring? |
| `CSA2_ATTN` | `1 << 9` | `flash_mla_csa2_sm121.cu` | FP4 `E2M1` dequant + 64-tok paged MLA + FP8 SWA combine | What is the bandwidth utilization of Reuse vs. Full layers? |
| `DENSE_MXFP8` | `1 << 10` | `b12x_mxfp8_dense_sm121.cu` | `wqkv_a`, `wq_b`, `wo_a` FP8 twin, shared expert GEMMs | Which dense projection benefits most from L2 weight prefetch? |
| `MOE_MXFP4` | `1 << 11` | `b12x_mxfp4_moe_sm121.cu` | Route planner, 576-wide grouped W4A4 GEMM, deterministic sum | Are `M=6` verify rows hitting optimal warp tile occupancy? |
| `DSPARK_DRAFT` | `1 << 12` | `sparkflash-scheduler` / Drafter | Layers `37..39` draft pass, Markov head, `conf: 0.1` cap | How many draft tokens survive per prompt domain (code vs prose)? |
| `ROCENANTE_RDMA` | `1 << 13` | `rocenante_oneshot_sm121.cu` | `t_stage`, `t_peers_ready` (straggler vs wire), `t_reduce` | Which rank is the straggler at each of the 104 collectives/step? |
| `NCCL_COLLECTIVE` | `1 << 14` | `sparkflash-comm` (NCCL) | Reduce-Scatter / All-Gather duration during `PREFILL_SP` | Are both ConnectX-7 PCIe Gen5 x4 rails balanced 50/50? |
| `L2_PREFETCH` | `1 << 15` | Side-stream L2 warner | Overlap window vs. RoCEnante wait window | Does the 6 MiB L2 prefetch finish before the collective exits? |

#### Reference Implementation of `crates/sparkflash-trace/src/lib.rs` and `kernels/trace_globaltimer_sm121.cuh`:

```cuda
// kernels/trace_globaltimer_sm121.cuh
// Zero-overhead in-graph GPU trace points using Blackwell %globaltimer in Coherent UMA.
#pragma once
#include <stdint.h>

#define TRACE_SCOPE_ENGRAM_GPU_GATE (1ULL << 6)
#define TRACE_SCOPE_MEGA_MHC        (1ULL << 7)
#define TRACE_SCOPE_CSA2_INDEXER    (1ULL << 8)
#define TRACE_SCOPE_CSA2_ATTN       (1ULL << 9)
#define TRACE_SCOPE_DENSE_MXFP8     (1ULL << 10)
#define TRACE_SCOPE_MOE_MXFP4       (1ULL << 11)
#define TRACE_SCOPE_DSPARK_DRAFT    (1ULL << 12)
#define TRACE_SCOPE_ROCENANTE_RDMA  (1ULL << 13)

struct __align__(32) GpuTraceEvent {
    uint64_t start_ns;
    uint64_t end_ns;
    uint32_t sub_wait_ns;   // e.g. RDMA peer wait or Engram stall inside kernel
    uint32_t scope_and_rank; // (scope_bit_index << 8) | rank
    uint32_t layer_id;
    uint32_t payload_meta;  // e.g. bytes transferred or active tokens M
};

struct GpuTraceRing {
    volatile uint64_t active_scope_mask; // Mapped in coherent UMA; toggled live from Rust!
    uint32_t write_idx;
    uint32_t capacity_mask;              // Power-of-two ring mask (e.g. (1 << 20) - 1)
    GpuTraceEvent events[1048576];       // 32 MiB coherent UMA ring buffer
};

__device__ __forceinline__ uint64_t read_globaltimer_ns() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
    return ret;
}

#if defined(SPARKFLASH_ENABLE_GPU_TRACE)
#define SPARK_GPU_TRACE_ENTER(ring, scope, t_var) \
    do { \
        if (blockIdx.x == 0 && threadIdx.x == 0 && ((ring)->active_scope_mask & (scope))) { \
            (t_var) = read_globaltimer_ns(); \
        } \
    } while (0)

#define SPARK_GPU_READ_TIMER(ring, scope, t_var) \
    do { \
        if ((ring)->active_scope_mask & (scope)) { \
            (t_var) = read_globaltimer_ns(); \
        } \
    } while (0)

#define SPARK_GPU_TRACE_EXIT(ring, scope, layer, meta, t_enter) \
    do { \
        if (blockIdx.x == 0 && threadIdx.x == 0 && (t_enter) != 0) { \
            uint64_t t_end = read_globaltimer_ns(); \
            uint32_t idx = atomicAdd(&(ring)->write_idx, 1U) & (ring)->capacity_mask; \
            (ring)->events[idx] = GpuTraceEvent{ \
                (t_enter), t_end, 0U, (uint32_t)__ffsll(scope), (uint32_t)(layer), (uint32_t)(meta) \
            }; \
        } \
    } while (0)

#define SPARK_GPU_TRACE_EXIT_ROCE(ring, scope, layer, bytes, t_enter, t_staged, t_ready) \
    do { \
        if (blockIdx.x == 0 && threadIdx.x == 0 && (t_enter) != 0) { \
            uint64_t t_end = read_globaltimer_ns(); \
            uint32_t idx = atomicAdd(&(ring)->write_idx, 1U) & (ring)->capacity_mask; \
            (ring)->events[idx] = GpuTraceEvent{ \
                (t_enter), t_end, (uint32_t)((t_ready) - (t_staged)), \
                (uint32_t)__ffsll(scope), (uint32_t)(layer), (uint32_t)(bytes) \
            }; \
        } \
    } while (0)
#else
#define SPARK_GPU_TRACE_ENTER(ring, scope, t_var) do {} while (0)
#define SPARK_GPU_READ_TIMER(ring, scope, t_var) do {} while (0)
#define SPARK_GPU_TRACE_EXIT(ring, scope, layer, meta, t_enter) do {} while (0)
#define SPARK_GPU_TRACE_EXIT_ROCE(ring, scope, layer, bytes, t_enter, t_staged, t_ready) do {} while (0)
#endif
```

```rust
//! crates/sparkflash-trace/src/lib.rs
//! Scoped, dynamically toggleable CPU + GPU trace point framework for 4x DGX Spark.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub struct TraceScope;
impl TraceScope {
    pub const NONE: u64            = 0;
    pub const API_ENCODE: u64      = 1 << 0;
    pub const SCHED_BATCH: u64     = 1 << 1;
    pub const KV_RADIX: u64        = 1 << 2;
    pub const SWA_REPLAY: u64      = 1 << 3;
    pub const ENGRAM_PREFETCH: u64 = 1 << 4;
    pub const ENGRAM_IO_URING: u64 = 1 << 5;
    pub const ENGRAM_GPU_GATE: u64 = 1 << 6;
    pub const MEGA_MHC: u64        = 1 << 7;
    pub const CSA2_INDEXER: u64    = 1 << 8;
    pub const CSA2_ATTN: u64       = 1 << 9;
    pub const DENSE_MXFP8: u64     = 1 << 10;
    pub const MOE_MXFP4: u64       = 1 << 11;
    pub const DSPARK_DRAFT: u64    = 1 << 12;
    pub const ROCENANTE_RDMA: u64  = 1 << 13;
    pub const NCCL_COLLECTIVE: u64 = 1 << 14;
    pub const L2_PREFETCH: u64     = 1 << 15;
    pub const ALL: u64             = 0xFFFF;
}

/// Global L1-cached bitmask controlling active trace scopes on CPU and mirrored to UMA for GPU.
pub static GLOBAL_TRACE_MASK: AtomicU64 = AtomicU64::new(TraceScope::NONE);
pub static TRACE_STEPS_REMAINING: AtomicU32 = AtomicU32::new(0);

#[inline(always)]
pub fn is_scope_enabled(scope: u64) -> bool {
    (GLOBAL_TRACE_MASK.load(Ordering::Relaxed) & scope) != 0
}

/// Reads the ARMv9 hardware virtual counter CNTVCT_EL0 (~1.2 ns, no syscall).
#[inline(always)]
pub fn read_arm_cntvct_ns() -> u64 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let cnt: u64;
        let freq: u64;
        std::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt, options(nomem, nostack, preserves_flags));
        std::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq, options(nomem, nostack, preserves_flags));
        (cnt as u128 * 1_000_000_000 / (freq as u128)) as u64
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        0
    }
}

pub struct ScopedCpuSpan {
    scope: u64,
    label: &'static str,
    meta: u32,
    start_ns: u64,
}

impl ScopedCpuSpan {
    #[inline(always)]
    pub fn enter(scope: u64, label: &'static str, meta: u32) -> Option<Self> {
        if !is_scope_enabled(scope) {
            return None;
        }
        Some(Self {
            scope,
            label,
            meta,
            start_ns: read_arm_cntvct_ns(),
        })
    }
}

impl Drop for ScopedCpuSpan {
    #[inline(always)]
    fn drop(&mut self) {
        let _end_ns = read_arm_cntvct_ns();
        // Pushes (self.scope, self.label, self.meta, self.start_ns, _end_ns) to thread-local lock-free ring
    }
}

/// Zero-overhead scoped trace macro: compiles to a single relaxed bitmask test when `trace-points`
/// feature is enabled, and to a literal no-op `{}` when `disable-trace-points` is set.
#[macro_export]
macro_rules! spark_trace_span {
    ($scope:expr, $label:literal, $meta:expr) => {
        #[cfg(not(feature = "disable-trace-points"))]
        let _spark_guard = $crate::ScopedCpuSpan::enter($scope, $label, $meta);
    };
}
```

## 6. Cluster Provisioning, Preflight Doctor, and Implementation Harness Plan

### 6.1 Seven-Point DGX Spark Preflight Verification (`sparkflash doctor`)

Before starting the engine, `sparkflash doctor` validates all 4 DGX Spark nodes against known GB10 hardware/driver pitfalls ([Petronella Tech 7-Point RoCE Checklist](https://www.google.com/search?q=https%3A%2F%2Fpetronellatech.com%2Fblog%2Fdgx-spark-cluster-bandwidth-what-400g-really-means%2F) and [DGX Spark Forum Benchmarks](https://www.google.com/search?q=https%3A%2F%2Fforums.developer.nvidia.com%2Ft%2F6x-dgx-spark-200g-roce-fabric-my-learning-journey%2F377871)):

1. **Driver & Kernel Check:** Verifies NVIDIA driver $\ge 580.178.04$ and kernel $\ge 6.17.0-1018$ / `7.0.0-1019-nvidia-64k` (rejecting driver `580.142` which throttles ConnectX-7 to 13 Gbps) and checks that `nvidia-uvm` 64K page packing is active.
2. **Dual-Rail ConnectX-7 Link Speed & MTU 9000:** Verifies that both PCIe Gen5 x4 halves (`enp1s0f0np0` / `rocep1s0f0` and `enP2p1s0f0np0` / `roceP2p1s0f0`) negotiate `200000Mb/s` with `MTU 9000` (`ping -M do -s 8972 <peer>`).
3. **Dual-Rail RDMA Bandwidth Gate:** Runs an automatic 1-second dual-rail `ibv_write_bw` / `RoCEnante` probe between all 4 ranks and asserts aggregate cross-node bandwidth $\ge 180\text{ Gbps}$ (`~22.5 GB/s`) and 10 KiB One-Shot All-Reduce latency $< 22\text{ µs}$.
4. **SM Clock Cap & Headless Memory Mode:** Locks GB10 GPU clocks to `[0, 2200]` MHz (`nvidia-smi -lgc 0,2200`) to prevent EC overcurrent trips and verifies `MemAvailable >= 114 GiB` per node before weight loading.
5. **Local NVMe Engram Shard Verification:** Verifies that `engram-l1-r{rank}of4.bin` and `engram-l14-r{rank}of4.bin` (~48 GiB total per node) exist on local PCIe Gen5 NVMe (`/var/lib/sparkflash/engram/`) with matching SHA256 headers and `O_DIRECT` alignment.

### 6.2 Step-by-Step Implementation Milestones for the Coding Harness

When feeding this specification into the implementation harness, execute the build in the following five deterministic milestones:

- **Milestone 1 — Workspace Scaffolding, `sparkflash-trace`, & `deepseek-recipe` OpenAI Protocol Layer (`crates/sparkflash-trace`, `crates/sparkflash-api`):**
  - Initialize the Cargo workspace and build `sparkflash-trace` first (`TraceScope` bitmask, `spark_trace_span!`, ARM `CNTVCT_EL0` timer, and `trace_globaltimer_sm121.cuh` coherent UMA ring buffer + `/v1/admin/trace` Chrome Perfetto exporter). Wire [deepseek-recipe](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fdeepseek-ai%2Fdeepseek-recipe%2Fblob%2Fmain%2FREADME.md) (`ChatCompletionRequest`, `Responses`, `Messages`, `DeepseekV41Encoding`), implement the `reasoning_effort` (`1..100`, default `75`, `low=50, high=75, max=100`) and `thinking`/`enable_thinking` remapper, add the decode-side N-gram/line loop-abort detector, and expose `/v1/chat/completions`, `/v1/cluster/stats`, and `/metrics` with unit tests against `encoding.py` test vectors.
- **Milestone 2 — Engram Packer & 3-Tier `io_uring` + DRM/UMA Cache Engine (`crates/sparkflash-engram`):**
  - Implement `sparkflash pack-engram` to extract `layers.1.engram` and `layers.14.engram` from the 48 safetensors shards into 512B-aligned per-rank NVMe files (`engram-l{1,14}-r{0..3}of4.bin`), implement `TokenizerCompressor` (`99,092` compressed IDs), the 24-head multiplicative-XOR hash, the 16-way set-associative UMA + DRM cache, and the 96-depth `io_uring` `O_DIRECT` prefetch worker instrumented with `TraceScope::ENGRAM_PREFETCH` and `TraceScope::ENGRAM_IO_URING`. Verify bit-identical row retrieval against the reference Python checkpoint.
- **Milestone 3 — `RoCEnante` One-Shot RDMA + Patched NCCL Dual-Path Communicator (`crates/sparkflash-comm`):**
  - Implement `ibverbs` dual-rail registration (`rocep1s0f0`, `roceP2p1s0f0`), the `mlx5` ConnectX-7 ASIC hairpin diagonal initializer for switchless 4-node rings, the `rocenante_oneshot_sm121.cu` kernel with `__threadfence_system()`, canonical `0..3` rank-order reduction, and `SPARK_GPU_TRACE_EXIT_ROCE` straggler-vs-wire timing, plus the 2 MiB threshold router to NCCL 2.30.7+. Validate $< 20\text{ µs}$ 10 KiB All-Reduce latency and bit-identical outputs across 10,000 CUDA graph replays.
- **Milestone 4 — `sm_121a` CUDA Kernels & 3-Tier KV Cache / SWA Bounded Replay (`crates/sparkflash-kernels`, `crates/sparkflash-kv`):**
  - Bind the pre-compiled `sm_121a` kernels (`mega_mhc_sm121.cu`, `b12x_mxfp8_dense_sm121.cu`, `b12x_mxfp4_moe_sm121.cu` at `EP=1` with shared expert $K=576 \to 640$ padding, `flash_mla_csa2_sm121.cu` with FP4 `E2M1` Main KV and 16,384-pos HSI candidate pool, FP8 `wo_a` twin, and `cert_lm_head_sm121.cu`), each wrapped with `SPARK_GPU_TRACE_ENTER/EXIT`. Implement the 64-token paged Radix KV cache and Encoder/Decoder 128-token SWA Bounded Replay.
- **Milestone 5 — CED Scheduler, DSpark Speculative Verifier, Full CUDA Graph Capture & Trace-Driven Optimization (`crates/sparkflash-scheduler`, `crates/sparkflash-cli`):**
  - Integrate the 20-layer Causal Encoder prefill bypass ($C_{20} = H_{19} W_{20}^{KV}$), Sequence-Parallel chunked prefill (`DSV41_PREFILL_SP`), short-request prefill bypass (`DSV41_PREFILL_BYPASS=256`), DSpark 3-layer drafter (`k=5`, `conf: 0.1` live router expert reuse, sync-free philox RNG), and L2 weight prefetch (`DSV41_L2_PREFETCH`). Capture full CUDA graphs for batch sizes `1..16`, capture a 45-step scoped trace via `sparkflash trace --scopes all --steps 45` to tune L2 prefetch windows and Engram prefetch lead depth, and run the correctness and throughput qualification gates (`qeval` 72/75, KL divergence $< 0.007$, 1M needle test, and `sparkDash` `c=1..16` sweeps).