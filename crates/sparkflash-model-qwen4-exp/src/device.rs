// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The TP=1 device arm.
//!
//! One rank, one CUDA context and the shared kernel images. There is no host forward and no
//! silent fallback: a build without the cubin, a TP world other than one, or a checkpoint that
//! disagrees with the plan is a typed refusal.
//!
//! # Attribution
//!
//! * Model architecture (Qwen4ExpForConditionalGeneration: 3:1 Gated DeltaNet + Qwen Sparse
//!   Attention, 4-stream hyper-connections, 512-expert top-10 MoE + shared expert, and Layer-2
//!   16-head bigram/trigram PLE) originates from
//!   [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Alibaba Cloud Qwen
//!   team) and the vLLM reference implementation
//!   ([vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0).
//! * NVFP4 W4A4 block-scaled quantization and 128-shard FP8 E4M3 PLE layout originate from
//!   [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
//!   (NVIDIA TensorRT ModelOpt 0.46).
//! * SM120/SM121 (DGX Spark GB10) kernel and local-inference execution patterns draw on prior art
//!   from [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0; Luke Alonso
//!   @lukealonso, @original-el8, and contributors) and
//!   [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (Apache-2.0). See the root
//!   NOTICE file.
//!
//! # What lives where
//!
//! * Device: the text language model and its NVFP4 experts (DEVICE_RESIDENT_BYTES, 77.8 GiB).
//! * Host, paged: the 51.2 GiB PLE n-gram table, through [crate::ple::PleTable] and a row cache.
//!   A token reads 16 rows of 160 bytes, so the host side of the PLE is 2.5 KiB per token.
//! * Host, transient: one token's PLE embedding (2560 f32) is copied to the device per step.
//!
//! # Arithmetic
//!
//! Every kernel in kernels/qwen4_exp_sm121.cu and kernels/qwen4_attn_sm121.cu mirrors
//! [crate::forward], which is the oracle:
//! 256-thread blocks, f32 accumulation and exact BF16 widening, no tensor cores. The NVFP4 expert
//! kernels dequantise on the fly (E2M1 nibble x E4M3 block scale x per-tensor scale) and use f32
//! activations: the checkpoint's static input_scale is carried but not applied, because applying
//! it would require the matching activation quantiser; this is the documented numeric difference
//! from vLLM's W4A4 path, and the reason the parity gate is a tolerance, not bit identity.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use sparkflash_kernels::cuda_driver::{
    Cuda, Deviceptr, HostMapped, Kernel, RunStream, StreamEvent,
};

/// A CUDA context handle that the model contract can carry.
///
/// The driver context is a process-wide primary context. The raw pointer inside Cuda is what makes
/// the compiler refuse Send/Sync. Each session owns a [RunStream] and puts all of its copies and
/// launches there; the weights and kernel handles shared by those sessions are read-only. Buffer
/// allocation and free go through the context's own pool mutex.
struct CudaHandle(Cuda);

// SAFETY: the context is the retained primary context for this process. Sessions use separate
// streams and separate buffers; the only shared state is immutable weights, immutable kernel
// handles, and buffers whose lifetime is ordered by each session's stream. The driver's context
// is current on each worker thread (every entry point calls `make_current`).
unsafe impl Send for CudaHandle {}
// SAFETY: as above; the handle is shared, but no mutable context state is reached without the
// driver's own synchronization or a session-local stream.
unsafe impl Sync for CudaHandle {}

impl std::ops::Deref for CudaHandle {
    type Target = Cuda;
    fn deref(&self) -> &Cuda {
        &self.0
    }
}
use sparkflash_model::{
    BackendError, Capabilities, DepthPolicy, ModelBackend, ModelDescriptor, ModelSession,
    PackIdentity, Part as SharedPart, Sampling, Speculation, StepOutput,
};
use sparkflash_weights::linalg::ScaleGrid;
use sparkflash_weights::safetensors::{INDEX_NAME, SafeTensorsSource};
use sparkflash_weights::tp::{TpLayout, TpRuntime};
use sparkflash_weights::{WeightSource, check_length};

use crate::config::{ExpertQuant, LayerKind, Qwen4ExpConfig};
use crate::ple::PleTable;
use crate::tensor::{DEVICE_RESIDENT_BYTES, Role};

mod batch;
mod blocks;
mod dflash;
mod fusion;
mod prefill;
pub(crate) mod tp;

use dflash::DFLASH2_STEP_COST;
pub use dflash::embedded_dflash2_cubin;
pub use prefill::{
    ATTENTION_CHUNK, CONV_HISTORY, DEFAULT_PREFILL_CHUNK, DENSE_ROWS, GDN_KEY, GDN_QK_ROW,
    GDN_VALUE, MAX_PREFILL_CHUNK, MIN_PREFILL_CHUNK, PrefillPlan, TOKEN_PREFILL_CHUNK,
    embedded_glm_moe_cubin, embedded_glm_prefill_cubin, embedded_prefill_cubin,
    embedded_scan_cubin, gdn_scan_shape,
};
use prefill::{PrefillKernels, PrefillScratch, Projection};
pub use sparkflash_ops::dflash::{DraftDevice, DraftScratch, GpuDflash2};
pub use sparkflash_ops::gemv::gemv_e4m3;
pub use tp::{reduce_two_shot, splits_two_shot};

/// Whether `SPARKFLASH_QWEN4_PROBE=hash` asks every probe for a hash of its buffer's bytes
/// instead of the finite check: two runs of one prompt then diff line by line, and the first line
/// that differs names the first buffer whose contents changed between them.
fn probe_hashing() -> bool {
    sparkflash_model::active_profile().diagnostics.probe
        == Some(sparkflash_model::profile::ProbeMode::Hash)
}

/// 64-bit FNV-1a over a probed buffer's bytes.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Per-stage device timing, enabled with `SPARKFLASH_QWEN4_PROF`.
///
/// A deployment diagnostic, not a serving feature: every `mark` drains the stream so a stage's
/// cost is attributable, which serialises the layer and is far too expensive to leave on. It is
/// what turns "decode is slow" into "the router is slow" without a profiler on the node.
mod prof {
    use std::cell::RefCell;
    use std::time::Instant;

    struct State {
        last: Instant,
        totals: Vec<(&'static str, f64, usize)>,
    }

    thread_local! {
        static STATE: RefCell<State> = RefCell::new(State {
            last: Instant::now(),
            totals: Vec::new(),
        });
    }

    /// Whether `SPARKFLASH_QWEN4_PROF` is set.
    pub fn enabled() -> bool {
        sparkflash_model::active_profile()
            .diagnostics
            .profile
            .unwrap_or(false)
    }

    /// Start a token's accounting.
    pub fn begin() {
        STATE.with(|state| state.borrow_mut().last = Instant::now());
    }

    /// Charge the time since the previous mark to `name`.
    pub fn mark(name: &'static str) {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            let elapsed = state.last.elapsed().as_secs_f64() * 1e3;
            state.last = Instant::now();
            match state
                .totals
                .iter_mut()
                .find(|(existing, _, _)| *existing == name)
            {
                Some((_, total, count)) => {
                    *total += elapsed;
                    *count += 1;
                }
                None => state.totals.push((name, elapsed, 1)),
            }
        });
    }

    /// Print and reset the accumulator, naming the forward's last position and its rows.
    pub fn report(position: usize, rows: usize) {
        STATE.with(|state| {
            let mut state = state.borrow_mut();
            if state.totals.is_empty() {
                return;
            }
            let total: f64 = state.totals.iter().map(|(_, time, _)| *time).sum();
            eprintln!("qwen4 prof p{position} rows {rows}: {total:.2} ms total");
            state
                .totals
                .sort_by(|left, right| right.1.total_cmp(&left.1));
            for (name, time, count) in &state.totals {
                eprintln!(
                    "qwen4 prof   {name:<14} {time:8.2} ms  {count:5} x  {:5.1}%",
                    time / total * 100.0
                );
            }
            state.totals.clear();
        });
    }
}

/// The QSA selection buffer the attention kernel reserves, in tokens.
///
/// Mirrors Q4E_MAX_SELECTED in kernels/qwen4_exp_sm121.cu. The released configuration selects at
/// most 2,051 (512 complete blocks of four plus the incomplete tail), so the margin is small on
/// purpose: a configuration that outgrows it is refused here rather than writing past the buffer.
pub const MAX_SELECTED_TOKENS: usize = 2304;

/// Per-row activation buffers a session exposes to a batch, in one canonical order.
const SCRATCH_SLOTS: usize = 48;

/// The slot of [DeviceSequence::scratch_pointers] that is `gemm_partials`.
const PARTIALS_SLOT: usize = 42;

/// The query heads q4e_attn_group computes together (Q4E_ATTN_GROUP).
pub const ATTN_GROUP_HEADS: usize = 4;

/// The selected tokens q4e_attn_group stages in shared memory (Q4E_ATTN_GROUP_SELECTED); a
/// configuration that selects more attends through q4e_attn.
pub const ATTN_GROUP_SELECTED: usize = 2052;

/// The grouped attention kernels by the query heads each computes together, in the order a
/// configuration takes them: the first whose count divides its query groups. Six is a four-rank
/// group's share of the released twelve.
pub const ATTN_GROUP_KERNELS: [(usize, &str); 2] =
    [(ATTN_GROUP_HEADS, "q4e_attn_group"), (6, "q4e_attn_group6")];

/// A grouped attention kernel's dynamic shared memory over `heads` heads: a row of logits and a
/// block of reduction floats for each head, then the token's selection.
#[must_use]
pub const fn attn_group_shared_bytes(heads: usize) -> u32 {
    ((heads * (ATTN_GROUP_SELECTED + 256) + ATTN_GROUP_SELECTED) * 4) as u32
}

/// q4e_attn_group's dynamic shared memory (45,136 bytes).
pub const ATTN_GROUP_SHARED_BYTES: u32 = attn_group_shared_bytes(ATTN_GROUP_HEADS);

/// The rows an MTP verify forward takes at most: the token being decoded and two chained drafts.
const MTP_VERIFY_ROWS: usize = 3;

/// The hyper-connection bottleneck the mix kernel's shared buffer holds (Q4E_MAX_LOWRANK).
pub const MAX_LOWRANK: usize = 1024;

/// The expert intermediate width the expert kernel's shared buffer holds
/// (Q4E_MAX_INTERMEDIATE).
pub const MAX_INTERMEDIATE: usize = 1024;

/// The number of indexer blocks the selection kernel folds into (Q4E_MAX_BUDGET).
pub const MAX_BUDGET_BLOCKS: usize = 1024;
/// The indexer heads (Q4E_QSA_MAX_HEADS) and channels a pooled key scores against.
pub const MAX_INDEXER_HEADS: usize = 8;
/// The widest indexer head the kernel holds (Q4E_QSA_HEAD_DIM).
pub const MAX_INDEXER_HEAD_DIM: usize = 128;

/// Tokens the token GEMM's tile holds (QG_TOKENS); a wider chunk goes through it in slices.
const GEMM_TILE_TOKENS: usize = 128;

use sparkflash_ops::engine_plan::EnginePlan;
use sparkflash_ops::gemv::tiled_operands;
pub use sparkflash_ops::gemv::{GEMV_MAX_BATCH, TILED_GEMV_BLOCKS, TILED_GEMV_PARTIALS};

/// Below this many tokens the GEMM's fixed 128-token tile wastes more work than the extra weight
/// passes it saves: the GEMV is then run once per eight tokens instead. At the tile size the
/// crossover measured at around sixty-four tokens.
const GEMM_MIN_BATCH: usize = 64;

/// Floats reserved for a split GEMM's segment sums: sixteen MiB covers every narrow matrix the
/// model has at the default chunk width, and the launchers clamp their segment counts to fit.
const GEMM_PARTIAL_FLOATS: usize = 4 << 20;

/// Arrival counters after the partials of every `gemm_partials` region, so they go wherever the
/// region is swapped or carved: one for each tile of a split few-row launch, which counts its
/// slices in and whose last slice resets it (q4p_last_slice). Zero between launches.
const GEMM_SPLIT_TICKETS: usize = GEMM_PARTIAL_FLOATS / 256;

/// The words a `gemm_partials` region holds: its partials, then its tickets.
const GEMM_PARTIAL_REGION: usize = GEMM_PARTIAL_FLOATS + GEMM_SPLIT_TICKETS;

/// A `gemm_partials` region's tickets.
fn split_tickets(partials: Deviceptr) -> Deviceptr {
    partials.offset(GEMM_PARTIAL_FLOATS * 4)
}

/// Zero the tickets of the `gemm_partials` regions `count` regions apart from `first` on
/// `stream`, and wait: a fresh allocation's are whatever it held.
fn zero_split_tickets(
    cuda: &Cuda,
    stream: &RunStream,
    zero: Kernel,
    first: Deviceptr,
    count: usize,
) -> Result<(), String> {
    for region in 0..count {
        let tickets = split_tickets(first.offset(region * GEMM_PARTIAL_REGION * 4));
        cuda.launch_on(
            stream,
            zero,
            (GEMM_SPLIT_TICKETS.div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            &[tickets.0, GEMM_SPLIT_TICKETS as u64],
        )?;
    }
    cuda.synchronize_stream(stream)
}

pub use sparkflash_ops::gemv::{
    TILED_GEMV_E4M3_ROUND, TILED_GEMV_MIN_COLUMNS, TILED_GEMV_ROUND, tiled_gemv_e4m3_splits,
    tiled_gemv_splits,
};
const _: () = assert!(TILED_GEMV_PARTIALS <= GEMM_PARTIAL_FLOATS);

/// Whether this build embeds a Qwen4-Exp cubin.
///
/// A build without nvcc compiles the pack as a host-only library; a test that needs a device run
/// checks this first and skips rather than failing the whole suite on a workstation.
#[must_use]
pub fn cubin_available() -> bool {
    !embedded_cubin().is_empty()
}

/// The cubin this build embeds.
///
/// Exposed so a kernel-level test can launch one kernel with chosen arguments instead of paying
/// for a whole checkpoint upload to reach it: the router is a hundred lines of selection logic and
/// a wrong index from it is otherwise only visible as a panic two hundred lines away.
#[must_use]
pub fn embedded_cubin() -> &'static [u8] {
    sparkflash_kernels::images::qwen4_exp_cubin()
}

/// The attention, sparse-indexer and KV-cache cubin this build embeds.
///
/// The grouped-query attention, the indexer's pool/score/pick kernels and the BF16 cache append
/// used to live in the token image; they now have their own image.
#[must_use]
pub fn embedded_attn_cubin() -> &'static [u8] {
    sparkflash_kernels::images::qwen4_attn_cubin()
}

/// The decode/token routed-expert (MoE) cubin this build embeds.
///
/// The router, the slot permutation and the expert passes used to live at the end of the token
/// image; they now have their own image so the token image carries no routed-expert kernel.
#[must_use]
pub fn embedded_moe_cubin() -> &'static [u8] {
    sparkflash_kernels::images::qwen4_moe_cubin()
}

pub use sparkflash_ops::gemv::{GpuMat, GpuMatE4m3};

/// One uploaded NVFP4 expert matrix and its two per-tensor scales.
#[derive(Debug, Clone, Copy)]
struct GpuNvfp4 {
    weight: Deviceptr,
    scales: Deviceptr,
    /// The per-tensor weight scale the checkpoint applies on top of the E4M3 block scales.
    scale_2: f32,
    /// The checkpoint's static activation scale.
    ///
    /// Carried so the uploaded description is complete, and deliberately not applied: this arm
    /// multiplies the dequantised weight by an f32 activation (W4A16), which is more accurate than
    /// the W4A4 arithmetic the scale was calibrated for. A parity run against vLLM therefore has to
    /// be a tolerance, not a bit comparison.
    #[allow(dead_code)]
    input_scale: f32,
}

#[derive(Debug, Clone, Copy)]
struct GpuExpert {
    gate: GpuNvfp4,
    up: GpuNvfp4,
    down: GpuNvfp4,
}

#[derive(Debug)]
struct GpuHyper {
    norm: Deviceptr,
    down: GpuMat,
    up: GpuMat,
    inject: Option<GpuMat>,
}

#[derive(Debug)]
struct GpuMoe {
    router: GpuMat,
    shared: GpuMat,
    shared_up: GpuMat,
    shared_down: GpuMat,
    shared_gate: GpuMat,
    experts: GpuExperts,
}

/// A layer's routed experts, as device tables indexed by the router's output.
///
/// The MoE used to copy the router's indices to the host, loop ten experts on the host and
/// launch one kernel each; the tables are what let two kernels cover all ten slots, with no
/// device-to-host copy and no host-side dispatch.
#[derive(Debug, Clone, Copy)]
enum GpuExperts {
    /// The model's layers: six NVFP4 pointers an expert (gate, gate scales, up, up scales, down,
    /// down scales) and a second table of the three per-tensor scales.
    Nvfp4 { table: Deviceptr, scales: Deviceptr },
    /// The MTP layer, the checkpoint's only FP8 experts: the same six pointers, the scale rows
    /// BF16 [rows/128, columns/128] blocks rather than per-tensor scalars.
    Fp8 { table: Deviceptr },
}

#[derive(Debug)]
enum GpuBranch {
    Linear {
        in_qkv: GpuMat,
        in_z: GpuMat,
        in_b: GpuMat,
        in_a: GpuMat,
        out_proj: GpuMat,
        conv: Deviceptr,
        a_log: Deviceptr,
        dt_bias: Deviceptr,
        norm: Deviceptr,
    },
    Sparse {
        q_proj: GpuMat,
        k_proj: GpuMat,
        v_proj: GpuMat,
        o_proj: GpuMat,
        q_norm: Deviceptr,
        k_norm: Deviceptr,
        index_qk: GpuMat,
        index_q_norm: Deviceptr,
        index_k_norm: Deviceptr,
    },
}

#[derive(Debug)]
struct GpuPle {
    key_proj: GpuMat,
    value_proj: GpuMat,
    norm_key: Deviceptr,
    norm_query: Deviceptr,
    norm_conv: Deviceptr,
    conv: Deviceptr,
    table: std::sync::Mutex<PleTable<crate::checkpoint::CheckpointRows>>,
}

#[derive(Debug)]
struct GpuLayer {
    branch: GpuBranch,
    attn_hyper: GpuHyper,
    mlp_hyper: GpuHyper,
    moe: GpuMoe,
    ple: Option<GpuPle>,
}

/// The MTP module: the two fusion projections that fold the model's widened stream and the next
/// token's embedding to one row, the mixer that folds the drafted layer's streams back, and the
/// one sparse-attention + MoE layer between them.
///
/// Its routed experts are the checkpoint's only FP8 weights with 128x128 block scales, so the
/// layer's expert table is the FP8 kind ([GpuExperts::Fp8]).
#[derive(Debug)]
struct GpuMtp {
    fc_embedding: GpuMat,
    fc_hidden: GpuMat,
    pre_norm_embedding: Deviceptr,
    pre_norm_hidden: Deviceptr,
    mixer: GpuHyper,
    attn_hyper: GpuHyper,
    mlp_hyper: GpuHyper,
    attn: GpuBranch,
    moe: GpuMoe,
}

#[derive(Debug)]
pub(crate) struct GpuWeights {
    embed: GpuMat,
    lm_head: GpuMat,
    mixer: GpuHyper,
    layers: Vec<GpuLayer>,
    mtp: Option<GpuMtp>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct Kernels {
    embed: Kernel,
    gemv_bf16: Kernel,
    gemv_tiled_bf16: Kernel,
    gemv4_bf16: Kernel,
    gemm_bf16: Kernel,
    gemm_reduce: Kernel,
    shared_gate_up: Kernel,
    rmsnorm: Kernel,
    zero: Kernel,
    add: Kernel,
    hyper_down: Kernel,
    hyper_up: Kernel,
    hyper_down_tiled: Kernel,
    hyper_up_tiled: Kernel,
    hyper_mix: Kernel,
    hyper_inject: Kernel,
    inject: Kernel,
    /// The two-shot all-reduce's in-place sum of this rank's chunk ([tp::reduce_two_shot]).
    sum_slots: Kernel,
    swiglu: Kernel,
    router: Kernel,
    /// The shared routed-expert plan (kernels/moe_plan_sm121.cu): the run order its expert
    /// kernels read and the run bounds that tell each block where its run starts.
    moe_plan: Kernel,
    expert_gate_up: Kernel,
    expert_down: Kernel,
    expert_gate_up_fp8: Kernel,
    expert_down_fp8: Kernel,
    fold_streams: Kernel,
    widen_streams: Kernel,
    moe_combine: Kernel,
    gdn_conv: Kernel,
    gdn_recur: Kernel,
    rope: Kernel,
    qsa_pool: Kernel,
    qsa_scores: Kernel,
    qsa_pick: Kernel,
    attn: Kernel,
    /// q4e_attn for several query heads of a KV head at once, bit-identical per head: the
    /// [ATTN_GROUP_KERNELS] the device grants their shared memory, with their head counts.
    attn_group: [Option<(usize, Kernel)>; ATTN_GROUP_KERNELS.len()],
    ple_gate: Kernel,
    ple_conv: Kernel,
    deinterleave: Kernel,
    rmsnorm_rows: Kernel,
    copy_flat: Kernel,
    /// Narrow a f32 projection row into the BF16 KV cache, flat or one row at a time.
    cache_bf16: Kernel,
    cache_rows_bf16: Kernel,
    /// The tensor-core forms a prefill chunk takes (kernels/qwen4_prefill_sm121.cu).
    prefill: PrefillKernels,
}

impl Kernels {
    fn load(cuda: &Cuda, gdn_stage_bytes: u32) -> Result<Self, String> {
        if embedded_cubin().is_empty() {
            return Err(
                "this build has no Qwen4-Exp cubin: rebuild on a host with nvcc (or set \
                 SPARKFLASH_NVCC) so build.rs compiles \
                 kernels/qwen4_exp_sm121.cu"
                    .to_owned(),
            );
        }
        if embedded_attn_cubin().is_empty() {
            return Err(
                "this build has no Qwen4-Exp attention cubin: rebuild on a host with nvcc (or set \
                 SPARKFLASH_NVCC) so build.rs compiles \
                 kernels/qwen4_attn_sm121.cu"
                    .to_owned(),
            );
        }
        if embedded_moe_cubin().is_empty() {
            return Err(
                "this build has no Qwen4-Exp MoE cubin: rebuild on a host with nvcc (or set \
                 SPARKFLASH_NVCC) so build.rs compiles \
                 kernels/qwen4_moe_sm121.cu"
                    .to_owned(),
            );
        }
        fn named(cuda: &Cuda, name: &'static str) -> Result<Kernel, String> {
            let (_, kernel) = cuda.load_named(embedded_cubin(), name)?;
            Ok(kernel)
        }
        // The attention, indexer and KV-cache family lives in its own image
        // (kernels/qwen4_attn_sm121.cu).
        fn attn(cuda: &Cuda, name: &'static str) -> Result<Kernel, String> {
            let (_, kernel) = cuda.load_named(embedded_attn_cubin(), name)?;
            Ok(kernel)
        }
        // The routed-expert family lives in its own image (kernels/qwen4_moe_sm121.cu).
        fn moe(cuda: &Cuda, name: &'static str) -> Result<Kernel, String> {
            let (_, kernel) = cuda.load_named(embedded_moe_cubin(), name)?;
            Ok(kernel)
        }
        /// A handle that reserves `shared` bytes of dynamic shared memory per launch, falling back
        /// to the plain one (and a warning) when the device refuses the reservation.
        fn staged(cuda: &Cuda, name: &'static str, shared: u32) -> Result<Kernel, String> {
            if shared == 0 {
                return named(cuda, name);
            }
            match cuda.load_named_shared(embedded_cubin(), name, shared) {
                Ok((_, kernel)) => Ok(kernel),
                Err(error) => {
                    eprintln!("{name}: {shared} bytes of shared memory refused ({error}); global");
                    named(cuda, name)
                }
            }
        }
        Ok(Self {
            embed: named(cuda, "q4e_embed")?,
            gemv_bf16: named(cuda, "q4e_gemv_bf16")?,
            gemv_tiled_bf16: named(cuda, "q4e_gemv_tiled_bf16")?,
            gemv4_bf16: named(cuda, "q4e_gemv4_bf16")?,
            gemm_bf16: named(cuda, "q4e_gemm_bf16")?,
            gemm_reduce: named(cuda, "q4e_gemm_reduce")?,
            shared_gate_up: named(cuda, "q4e_shared_gate_up")?,
            rmsnorm: named(cuda, "q4e_rmsnorm")?,
            zero: named(cuda, "q4e_zero")?,
            add: named(cuda, "q4e_add")?,
            hyper_down: named(cuda, "q4e_hyper_down")?,
            hyper_up: named(cuda, "q4e_hyper_up")?,
            hyper_down_tiled: named(cuda, "q4e_hyper_down_tiled")?,
            hyper_up_tiled: named(cuda, "q4e_hyper_up_tiled")?,
            hyper_mix: named(cuda, "q4e_hyper_mix")?,
            hyper_inject: named(cuda, "q4e_hyper_inject")?,
            inject: named(cuda, "q4e_inject")?,
            sum_slots: named(cuda, "q4e_sum_slots")?,
            swiglu: named(cuda, "q4e_swiglu")?,
            router: moe(cuda, "q4e_router")?,
            moe_plan: {
                if sparkflash_kernels::images::moe_plan_cubin().is_empty() {
                    return Err(
                        "this build has no routed-expert plan cubin: rebuild on a host with nvcc \
                         (or set SPARKFLASH_NVCC) so build.rs compiles kernels/moe_plan_sm121.cu"
                            .to_owned(),
                    );
                }
                let (_, kernel) =
                    cuda.load_named(sparkflash_kernels::images::moe_plan_cubin(), "moe_plan")?;
                kernel
            },
            expert_gate_up: moe(cuda, "q4e_expert_gate_up")?,
            expert_down: moe(cuda, "q4e_expert_down")?,
            expert_gate_up_fp8: moe(cuda, "q4e_expert_gate_up_fp8")?,
            expert_down_fp8: moe(cuda, "q4e_expert_down_fp8")?,
            fold_streams: named(cuda, "q4e_fold_streams")?,
            widen_streams: named(cuda, "q4e_widen_streams")?,
            moe_combine: moe(cuda, "q4e_moe_combine")?,
            gdn_conv: named(cuda, "q4e_gdn_conv")?,
            gdn_recur: staged(cuda, "q4e_gdn_recur", gdn_stage_bytes)?,
            rope: named(cuda, "q4e_rope")?,
            qsa_pool: attn(cuda, "q4e_qsa_pool")?,
            qsa_scores: attn(cuda, "q4e_qsa_scores")?,
            qsa_pick: attn(cuda, "q4e_qsa_pick")?,
            attn: attn(cuda, "q4e_attn")?,
            // Their buffers live only in the dynamic reservation, so a refusal means q4e_attn.
            attn_group: ATTN_GROUP_KERNELS.map(|(heads, name)| {
                let bytes = attn_group_shared_bytes(heads);
                cuda.load_named_shared(embedded_attn_cubin(), name, bytes)
                    .map_err(|error| {
                        eprintln!(
                            "{name}: {bytes} bytes of shared memory refused ({error}); q4e_attn"
                        );
                    })
                    .ok()
                    .map(|(_, kernel)| (heads, kernel))
            }),
            ple_gate: named(cuda, "q4e_ple_gate")?,
            ple_conv: named(cuda, "q4e_ple_conv")?,
            deinterleave: named(cuda, "q4e_deinterleave")?,
            rmsnorm_rows: named(cuda, "q4e_rmsnorm_rows")?,
            copy_flat: named(cuda, "q4e_copy_flat")?,
            cache_bf16: attn(cuda, "q4e_cache_bf16")?,
            cache_rows_bf16: attn(cuda, "q4e_cache_rows_bf16")?,
            prefill: PrefillKernels::load(cuda)?,
        })
    }
}

fn bytes_f32(bytes: &[u8], name: &str) -> Result<f32, String> {
    let array = <[u8; 4]>::try_from(bytes)
        .map_err(|_| format!("{name} is not one F32 ({})", bytes.len()))?;
    Ok(f32::from_le_bytes(array))
}

fn bytes_bf16_scalar(bytes: &[u8], name: &str) -> Result<f32, String> {
    let array = <[u8; 2]>::try_from(bytes)
        .map_err(|_| format!("{name} is not one BF16 ({})", bytes.len()))?;
    Ok(half::bf16::from_le_bytes(array).to_f32())
}

/// Where the weight walk puts each upload: a device allocation per weight ([Direct], the
/// checkpoint path), or a converted rank file's slots ([crate::convert]).
pub(crate) trait Sink {
    /// Whether [Sink::weight] needs the bytes; a sink that only plans where they go, or already
    /// holds them in a slab, takes the shapes alone and is handed no bytes.
    fn wants_bytes(&self) -> bool;

    /// Whether the walk should read the two per-tensor NVFP4 scales; a slab sink gets them from
    /// the rank file's index instead.
    fn wants_scalars(&self) -> bool {
        self.wants_bytes()
    }

    /// The device address of the next weight, `key`, `len` bytes, which are `bytes` when the
    /// sink wants them and empty when it does not.
    fn weight(&mut self, key: &str, len: usize, bytes: Vec<u8>) -> Result<Deviceptr, String>;

    /// Bytes built from addresses (the routed experts' pointer tables), which no rank file holds.
    fn derived(&mut self, key: &str, len: usize, bytes: &[u8]) -> Result<Deviceptr, String>;

    /// The two per-tensor NVFP4 scales; a sink that records them returns what the walk read, and
    /// a slab sink returns the pair the rank file's index recorded.
    fn nvfp4_scalars(&mut self, scale_2: f32, input_scale: f32) -> Result<(f32, f32), String>;
}

/// Each weight in its own device allocation, uploaded as it is read.
pub(crate) struct Direct<'a> {
    pub(crate) cuda: &'a Cuda,
}

impl Sink for Direct<'_> {
    fn wants_bytes(&self) -> bool {
        true
    }

    fn weight(&mut self, key: &str, len: usize, bytes: Vec<u8>) -> Result<Deviceptr, String> {
        check_length(key, len, &bytes)?;
        self.derived(key, len, &bytes)
    }

    fn derived(&mut self, _: &str, _: usize, bytes: &[u8]) -> Result<Deviceptr, String> {
        let mut total = 0usize;
        let pointer = self.cuda.alloc(bytes.len().max(4), &mut total)?;
        self.cuda.copy_htod(&pointer, bytes)?;
        Ok(pointer)
    }

    fn nvfp4_scalars(&mut self, scale_2: f32, input_scale: f32) -> Result<(f32, f32), String> {
        Ok((scale_2, input_scale))
    }
}

/// The checkpoint walk: each upload goes to the sink in the order the model is built.
#[allow(dead_code)]
pub(crate) struct Uploader<'a> {
    /// The context [Uploader::mat_e4m3] quantizes on; unset for the host-only convert walks.
    cuda: Option<&'a Cuda>,
    sink: RefCell<&'a mut dyn Sink>,
}

/// The rows or the columns `first..first + count` of a matrix: what one rank of a
/// tensor-parallel world holds of it. `Rows(0, rows)` is the whole matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Part {
    Rows(usize, usize),
    Columns(usize, usize),
}

impl Part {
    /// The part's shape in a `[rows, columns]` matrix, or why it does not fit.
    fn shape(self, (rows, columns): (usize, usize)) -> Result<(usize, usize), String> {
        let (first, count, within) = match self {
            Self::Rows(first, count) => (first, count, rows),
            Self::Columns(first, count) => (first, count, columns),
        };
        if count == 0 || first + count > within {
            return Err(format!(
                "{self:?} is not a part of a [{rows}, {columns}] matrix"
            ));
        }
        Ok(match self {
            Self::Rows(..) => (count, columns),
            Self::Columns(..) => (rows, count),
        })
    }

    /// The part's bytes of the row-major `bytes` of a `[rows, columns]` matrix, after
    /// [Self::shape] has accepted it.
    fn of(self, bytes: &[u8], (rows, columns): (usize, usize)) -> std::borrow::Cow<'_, [u8]> {
        let row = bytes.len() / rows;
        let element = row / columns;
        match self {
            Self::Rows(first, count) => bytes[first * row..(first + count) * row].into(),
            Self::Columns(first, count) => bytes
                .chunks_exact(row)
                .flat_map(|values| &values[first * element..(first + count) * element])
                .copied()
                .collect::<Vec<u8>>()
                .into(),
        }
    }
}

#[allow(dead_code)]
impl<'a> Uploader<'a> {
    pub(crate) fn new(cuda: &'a Cuda, sink: &'a mut dyn Sink) -> Self {
        Self {
            cuda: Some(cuda),
            sink: RefCell::new(sink),
        }
    }

    /// A walk that places every weight but reads none, for the host-only convert passes.
    pub(crate) fn planning(sink: &'a mut dyn Sink) -> Self {
        Self {
            cuda: None,
            sink: RefCell::new(sink),
        }
    }

    fn wants_bytes(&self) -> bool {
        self.sink.borrow().wants_bytes()
    }

    fn mat(
        &self,
        source: &dyn WeightSource,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<GpuMat, String> {
        self.mat_part(source, name, (rows, columns), Part::Rows(0, rows))
    }

    /// `part` of the BF16 checkpoint matrix `name`, `[rows, columns]`.
    fn mat_part(
        &self,
        source: &dyn WeightSource,
        name: &str,
        (rows, columns): (usize, usize),
        part: Part,
    ) -> Result<GpuMat, String> {
        let (part_rows, part_columns) = part.shape((rows, columns))?;
        let len = part_rows * part_columns * 2;
        let bytes = if self.wants_bytes() {
            let bytes = source.read(name).map_err(|error| error.to_string())?;
            let expected = rows * columns * 2;
            if bytes.len() != expected {
                return Err(format!(
                    "{name} is {} bytes, expected {expected} for a BF16 [{rows}, {columns}]",
                    bytes.len()
                ));
            }
            part.of(&bytes, (rows, columns)).into_owned()
        } else {
            Vec::new()
        };
        Ok(GpuMat {
            pointer: self.sink.borrow_mut().weight(name, len, bytes)?,
            rows: part_rows,
            columns: part_columns,
        })
    }

    /// The row runs `ranges` (first, count) of the BF16 checkpoint matrix `name`, `[rows,
    /// columns]`, one after another: a rank's heads of a projection that stacks several kinds of
    /// head, such as the GDN's Q, K and V.
    fn mat_rows(
        &self,
        source: &dyn WeightSource,
        name: &str,
        (rows, columns): (usize, usize),
        ranges: &[(usize, usize)],
    ) -> Result<GpuMat, String> {
        if let [(first, count)] = ranges {
            return self.mat_part(source, name, (rows, columns), Part::Rows(*first, *count));
        }
        let bytes = self.rows_of(source, name, (rows, columns * 2), ranges)?;
        let part_rows = ranges.iter().map(|(_, count)| count).sum();
        Ok(GpuMat {
            pointer: self
                .sink
                .borrow_mut()
                .weight(name, part_rows * columns * 2, bytes)?,
            rows: part_rows,
            columns,
        })
    }

    /// The row runs `ranges` of the tensor `name`, `rows` rows of `row_bytes` each, one after
    /// another; nothing when the sink wants no bytes.
    fn rows_of(
        &self,
        source: &dyn WeightSource,
        name: &str,
        (rows, row_bytes): (usize, usize),
        ranges: &[(usize, usize)],
    ) -> Result<Vec<u8>, String> {
        for &(first, count) in ranges {
            Part::Rows(first, count).shape((rows, row_bytes))?;
        }
        if !self.wants_bytes() {
            return Ok(Vec::new());
        }
        let bytes = source.read(name).map_err(|error| error.to_string())?;
        if bytes.len() != rows * row_bytes {
            return Err(format!(
                "{name} is {} bytes, expected {} for {rows} rows of {row_bytes}",
                bytes.len(),
                rows * row_bytes
            ));
        }
        Ok(ranges
            .iter()
            .flat_map(|&(first, count)| &bytes[first * row_bytes..(first + count) * row_bytes])
            .copied()
            .collect())
    }

    /// A BF16 checkpoint matrix held as E4M3 ([GpuMatE4m3::quantize]); the BF16 upload is freed
    /// once quantized.
    fn mat_e4m3(
        &self,
        quantize: Kernel,
        source: &dyn WeightSource,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<GpuMatE4m3, String> {
        let cuda = self
            .cuda
            .ok_or_else(|| "the E4M3 quantization needs a device".to_owned())?;
        let matrix = self.mat(source, name, rows, columns)?;
        let quantized = GpuMatE4m3::quantize(cuda, quantize, &matrix, &mut 0);
        cuda.free_buffers(&[&matrix.pointer]);
        quantized
    }

    /// `part` of [Self::mat_e4m3]'s matrix: the whole matrix's codes and row scales, so a part
    /// holds the rounding the whole one does.
    fn mat_e4m3_part(
        &self,
        quantize: Kernel,
        source: &dyn WeightSource,
        name: &str,
        (rows, columns): (usize, usize),
        part: Part,
    ) -> Result<GpuMatE4m3, String> {
        let whole = self.mat_e4m3(quantize, source, name, rows, columns)?;
        if part == Part::Rows(0, rows) {
            return Ok(whole);
        }
        let cuda = self
            .cuda
            .ok_or_else(|| "the E4M3 quantization needs a device".to_owned())?;
        let (mut codes, mut scales) = (vec![0u8; rows * columns], vec![0u8; rows * 4]);
        let copied = cuda
            .copy_dtoh(&mut codes, &whole.weight)
            .and_then(|()| cuda.copy_dtoh(&mut scales, &whole.scales));
        cuda.free_buffers(&[&whole.weight, &whole.scales]);
        copied?;
        let (part_rows, part_columns) = part.shape((rows, columns))?;
        if !part_columns.is_multiple_of(16) {
            return Err(format!("{part:?} of {name} has no whole 16-byte E4M3 rows"));
        }
        // A row keeps its scale whichever of its columns a part holds.
        let scales = match part {
            Part::Rows(..) => part.of(&scales, (rows, 1)),
            Part::Columns(..) => scales.as_slice().into(),
        };
        let codes = part.of(&codes, (rows, columns));
        Ok(GpuMatE4m3 {
            weight: self.bytes(codes.as_ref())?,
            scales: self.bytes(scales.as_ref())?,
            rows: part_rows,
            columns: part_columns,
        })
    }

    fn vector(
        &self,
        source: &dyn WeightSource,
        name: &str,
        elements: usize,
    ) -> Result<Deviceptr, String> {
        let len = elements * 2;
        let bytes = if self.wants_bytes() {
            let bytes = source.read(name).map_err(|error| error.to_string())?;
            if bytes.len() != len {
                return Err(format!(
                    "{name} is {} bytes, expected {len} for {elements} BF16 values",
                    bytes.len()
                ));
            }
            bytes
        } else {
            Vec::new()
        };
        self.sink.borrow_mut().weight(name, len, bytes)
    }

    /// The row runs `ranges` of the BF16 tensor `name`, `rows` rows of `row_len` values, one
    /// after another: a rank's channels of a per-channel vector or filter.
    fn vector_rows(
        &self,
        source: &dyn WeightSource,
        name: &str,
        (rows, row_len): (usize, usize),
        ranges: &[(usize, usize)],
    ) -> Result<Deviceptr, String> {
        if let [(0, count)] = ranges
            && *count == rows
        {
            return self.vector(source, name, rows * row_len);
        }
        let bytes = self.rows_of(source, name, (rows, row_len * 2), ranges)?;
        let len = ranges.iter().map(|(_, count)| count).sum::<usize>() * row_len * 2;
        self.sink.borrow_mut().weight(name, len, bytes)
    }

    /// `part` of one block-scaled FP8 expert matrix: the E4M3 codes and the BF16 inverse-scale
    /// grid, both taken in whole 128-wide tiles.
    ///
    /// The tile is the sharing unit: its one BF16 scale covers all 128 of its columns, so a rank
    /// holds a run of whole tiles and its scale grid is the same run of the global grid. No zeros
    /// are appended here - [tp::Shard::expert_tiles] hands out whole tiles and nothing narrower -
    /// so the codes a rank holds are exactly `rows x columns` of the released weight.
    fn fp8_part(
        &self,
        source: &dyn WeightSource,
        prefix: &str,
        (rows, columns): (usize, usize),
        part: Part,
    ) -> Result<(Deviceptr, Deviceptr), String> {
        const TILE: usize = 128;
        let (part_rows, part_columns) = part.shape((rows, columns))?;
        let (first, count) = match part {
            Part::Rows(first, count) | Part::Columns(first, count) => (first, count),
        };
        if !first.is_multiple_of(TILE) || !count.is_multiple_of(TILE) {
            return Err(format!(
                "{part:?} of the FP8 [{rows}, {columns}] at {prefix} splits a 128x128 scale tile"
            ));
        }
        let weight_len = part_rows * part_columns;
        let scale_len = (part_rows / TILE) * (part_columns / TILE) * 2;
        let (weight, scales) = if self.wants_bytes() {
            let weight = source
                .read(&format!("{prefix}.weight"))
                .map_err(|error| error.to_string())?;
            let scales = source
                .read(&format!("{prefix}.weight_scale_inv"))
                .map_err(|error| error.to_string())?;
            if weight.len() != rows * columns {
                return Err(format!(
                    "{prefix}.weight is {} bytes, expected {} for an FP8 [{rows}, {columns}]",
                    weight.len(),
                    rows * columns
                ));
            }
            let grid = (rows / TILE, columns / TILE);
            if scales.len() != grid.0 * grid.1 * 2 {
                return Err(format!(
                    "{prefix}.weight_scale_inv is {} bytes, expected {} for a {:?} tile grid",
                    scales.len(),
                    grid.0 * grid.1 * 2,
                    grid
                ));
            }
            // One code a weight, so the part of the weight is the part itself; one BF16 a tile,
            // so the part of the grid is the part in tile units.
            let tile_part = match part {
                Part::Rows(first, count) => Part::Rows(first / TILE, count / TILE),
                Part::Columns(first, count) => Part::Columns(first / TILE, count / TILE),
            };
            (
                part.of(&weight, (rows, columns)).into_owned(),
                tile_part.of(&scales, grid).into_owned(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let weight_pointer =
            self.sink
                .borrow_mut()
                .weight(&format!("{prefix}.weight"), weight_len, weight)?;
        let scale_pointer = self.sink.borrow_mut().weight(
            &format!("{prefix}.weight_scale_inv"),
            scale_len,
            scales,
        )?;
        Ok((weight_pointer, scale_pointer))
    }

    /// One block-scaled FP8 expert matrix: the codes and the BF16 inverse-scale grid.
    fn fp8(
        &self,
        source: &dyn WeightSource,
        prefix: &str,
        rows: usize,
        columns: usize,
    ) -> Result<(Deviceptr, Deviceptr), String> {
        let weight_len = rows * columns;
        let scale_len = ScaleGrid::Tile(128).len(rows, columns) * 2;
        let (weight, scales) = if self.wants_bytes() {
            let weight = source
                .read(&format!("{prefix}.weight"))
                .map_err(|error| error.to_string())?;
            let scales = source
                .read(&format!("{prefix}.weight_scale_inv"))
                .map_err(|error| error.to_string())?;
            if weight.len() != weight_len {
                return Err(format!(
                    "{prefix}.weight is {} bytes, expected {weight_len} for an FP8 matrix",
                    weight.len()
                ));
            }
            if scales.len() != scale_len {
                return Err(format!(
                    "{prefix}.weight_scale_inv is {} bytes, expected {scale_len}",
                    scales.len()
                ));
            }
            (weight, scales)
        } else {
            (Vec::new(), Vec::new())
        };
        let weight_pointer =
            self.sink
                .borrow_mut()
                .weight(&format!("{prefix}.weight"), weight_len, weight)?;
        let scale_pointer = self.sink.borrow_mut().weight(
            &format!("{prefix}.weight_scale_inv"),
            scale_len,
            scales,
        )?;
        Ok((weight_pointer, scale_pointer))
    }

    /// `part` of the NVFP4 checkpoint matrix at `prefix`, `[rows, columns]`, its rows or columns
    /// zero-padded to `padded`: a rank's share of an expert, widened to the kernels' tile. Zero
    /// codes under a zero block scale add nothing to any dot product, so the padding changes no
    /// sum. The whole matrix unpadded (one rank's share) is the checkpoint's bytes as they are,
    /// a last partial scale group included.
    fn nvfp4_part(
        &self,
        source: &dyn WeightSource,
        prefix: &str,
        (rows, columns): (usize, usize),
        part: Part,
        padded: usize,
    ) -> Result<GpuNvfp4, String> {
        let (part_rows, part_columns) = part.shape((rows, columns))?;
        let (out_rows, out_columns, kept) = match part {
            Part::Rows(..) => (padded, part_columns, part_rows),
            Part::Columns(..) => (part_rows, padded, part_columns),
        };
        let whole = (out_rows, out_columns) == (rows, columns) && kept == padded;
        if !whole
            && (padded < kept || !columns.is_multiple_of(16) || !out_columns.is_multiple_of(16))
        {
            return Err(format!(
                "{part:?} of the NVFP4 [{rows}, {columns}] at {prefix} does not pad to {padded} \
                 in whole 16-column scale groups"
            ));
        }
        if let Part::Columns(first, count) = part
            && !whole
            && (!first.is_multiple_of(16) || !count.is_multiple_of(16))
        {
            return Err(format!(
                "{part:?} of {prefix} splits a 16-column NVFP4 scale group"
            ));
        }
        let weight_len = out_rows * out_columns.div_ceil(2);
        let scale_len = out_rows * out_columns.div_ceil(16);
        let (weight, scales) = if self.wants_bytes() {
            let weight = source
                .read(&format!("{prefix}.weight"))
                .map_err(|error| error.to_string())?;
            let scales = source
                .read(&format!("{prefix}.weight_scale"))
                .map_err(|error| error.to_string())?;
            let (whole_weight, whole_scales) =
                (rows * columns.div_ceil(2), rows * columns.div_ceil(16));
            if weight.len() != whole_weight {
                return Err(format!(
                    "{prefix}.weight is {} bytes, expected {whole_weight} for an NVFP4 [{rows}, {columns}]",
                    weight.len()
                ));
            }
            if scales.len() != whole_scales {
                return Err(format!(
                    "{prefix}.weight_scale is {} bytes, expected {whole_scales}",
                    scales.len()
                ));
            }
            // The codes are two a byte and the scales one a 16-column group, so the part of each
            // is the same part of a byte matrix that many times narrower.
            let narrow = |bytes: Vec<u8>, per: usize| -> Vec<u8> {
                if whole {
                    return bytes;
                }
                let shape = (rows, columns / per);
                let piece = match part {
                    Part::Rows(first, count) => Part::Rows(first, count),
                    Part::Columns(first, count) => Part::Columns(first / per, count / per),
                };
                let kept = piece.of(&bytes, shape);
                match part {
                    // Whole zero rows after the part's.
                    Part::Rows(..) => {
                        let mut out = kept.into_owned();
                        out.resize(out_rows * out_columns / per, 0);
                        out
                    }
                    // Each row's zero tail after its columns.
                    Part::Columns(..) => kept
                        .chunks_exact(part_columns / per)
                        .flat_map(|row| {
                            row.iter()
                                .copied()
                                .chain(std::iter::repeat_n(0, (out_columns - part_columns) / per))
                        })
                        .collect(),
                }
            };
            (narrow(weight, 2), narrow(scales, 16))
        } else {
            (Vec::new(), Vec::new())
        };
        let weight_pointer =
            self.sink
                .borrow_mut()
                .weight(&format!("{prefix}.weight"), weight_len, weight)?;
        let scale_pointer =
            self.sink
                .borrow_mut()
                .weight(&format!("{prefix}.weight_scale"), scale_len, scales)?;
        let (scale_2, input_scale) = if self.sink.borrow().wants_scalars() {
            (
                bytes_f32(
                    &source
                        .read(&format!("{prefix}.weight_scale_2"))
                        .map_err(|error| error.to_string())?,
                    prefix,
                )?,
                bytes_f32(
                    &source
                        .read(&format!("{prefix}.input_scale"))
                        .map_err(|error| error.to_string())?,
                    prefix,
                )?,
            )
        } else {
            (0.0, 0.0)
        };
        let (scale_2, input_scale) = self.sink.borrow_mut().nvfp4_scalars(scale_2, input_scale)?;
        Ok(GpuNvfp4 {
            weight: weight_pointer,
            scales: scale_pointer,
            scale_2,
            input_scale,
        })
    }

    fn derived(&self, key: &str, bytes: &[u8]) -> Result<Deviceptr, String> {
        self.sink.borrow_mut().derived(key, bytes.len(), bytes)
    }

    /// `bytes` of host memory as its own device allocation.
    fn bytes(&self, bytes: &[u8]) -> Result<Deviceptr, String> {
        let cuda = self
            .cuda
            .ok_or_else(|| "a device upload needs a device".to_owned())?;
        let mut total = 0usize;
        let pointer = cuda.alloc(bytes.len().max(4), &mut total)?;
        cuda.copy_htod(&pointer, bytes)?;
        Ok(pointer)
    }
}

impl GpuWeights {
    /// Upload every resident weight from the checkpoint into its own allocation.
    pub(crate) fn upload(
        cuda: &Cuda,
        directory: &Path,
        source: &dyn WeightSource,
        config: &Qwen4ExpConfig,
        shard: tp::Shard,
    ) -> Result<Self, String> {
        let mut sink = Direct { cuda };
        build_shard(
            &Uploader::new(cuda, &mut sink),
            directory,
            source,
            config,
            shard,
        )
    }
}

/// The weight walk: each upload goes to the uploader's sink in the model's order, for one rank's
/// `shard` of the model - its share of every attention block's heads, every GDN block's key and
/// value heads and every expert's width, and the rest whole ([tp::rank_config] describes the
/// shares). `config` is the whole model's; [tp::Shard::WHOLE] is the whole model.
pub(crate) fn build_shard(
    uploader: &Uploader<'_>,
    directory: &Path,
    source: &dyn WeightSource,
    config: &Qwen4ExpConfig,
    shard: tp::Shard,
) -> Result<GpuWeights, String> {
    let hidden = config.hidden_size;
    // Each share as (first, count) in heads or columns, and the parts of a matrix they name.
    let heads = shard.even(config.num_attention_heads);
    let kv_heads = shard.kv_heads(config.num_key_value_heads);
    let key_heads = shard.even(config.linear_num_key_heads);
    let value_heads = shard.even(config.linear_num_value_heads);
    let expert_share = shard.even(config.moe_intermediate_size);
    let expert_width = shard.expert_width(config.moe_intermediate_size);

    let shared = shard.even(config.shared_expert_intermediate_size);
    let rows =
        |(first, count): (usize, usize), width: usize| Part::Rows(first * width, count * width);
    let columns =
        |(first, count): (usize, usize), width: usize| Part::Columns(first * width, count * width);
    let (key_dim, value_dim) = (config.linear_key_head_dim, config.linear_value_head_dim);
    // The GDN's mixed projection stacks the Q, K and V heads: a rank's rows are a run of each.
    let qkv_rows = [
        (key_heads.0 * key_dim, key_heads.1 * key_dim),
        (
            config.linear_key_width() + key_heads.0 * key_dim,
            key_heads.1 * key_dim,
        ),
        (
            2 * config.linear_key_width() + value_heads.0 * value_dim,
            value_heads.1 * value_dim,
        ),
    ];
    let width = config.hyper_width();
    let hyper = |prefix: String, inject: bool| -> Result<GpuHyper, String> {
        Ok(GpuHyper {
            norm: uploader.vector(source, &format!("{prefix}.hc_norm.weight"), width)?,
            down: uploader.mat(
                source,
                &format!("{prefix}.input_mix_weight_down.weight"),
                config.hc_lowrank,
                width,
            )?,
            up: uploader.mat(
                source,
                &format!("{prefix}.input_mix_weight_up.weight"),
                width,
                config.hc_lowrank,
            )?,
            inject: if inject {
                Some(uploader.mat(
                    source,
                    &format!("{prefix}.block_inject_weight.weight"),
                    config.hc_count,
                    width,
                )?)
            } else {
                None
            },
        })
    };
    let embed = uploader.mat(
        source,
        "model.language_model.embed_tokens.weight",
        config.vocab_size,
        hidden,
    )?;
    let lm_head = uploader.mat(source, "lm_head.weight", config.vocab_size, hidden)?;
    let mixer = hyper(
        "model.language_model.hyper_connection_mixer".to_owned(),
        false,
    )?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    // The shared walk is the one selection: the attention and FFN a layer runs come from the
    // typed architecture, not from a second reading of `layer_types`.
    let steps = config.steps();
    if steps.len() != config.num_hidden_layers {
        return Err(format!(
            "the shared walk names {} layers, and the configuration serves {}",
            steps.len(),
            config.num_hidden_layers
        ));
    }
    for (index, step) in steps.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        let branch = match step.attention {
            SharedPart::GdnScan => {
                let prefix = format!("{prefix}.linear_attn");
                GpuBranch::Linear {
                    in_qkv: uploader.mat_rows(
                        source,
                        &format!("{prefix}.in_proj_qkv.weight"),
                        (config.linear_conv_width(), hidden),
                        &qkv_rows,
                    )?,
                    in_z: uploader.mat_part(
                        source,
                        &format!("{prefix}.in_proj_z.weight"),
                        (config.linear_value_width(), hidden),
                        rows(value_heads, value_dim),
                    )?,
                    in_b: uploader.mat_part(
                        source,
                        &format!("{prefix}.in_proj_b.weight"),
                        (config.linear_num_value_heads, hidden),
                        rows(value_heads, 1),
                    )?,
                    in_a: uploader.mat_part(
                        source,
                        &format!("{prefix}.in_proj_a.weight"),
                        (config.linear_num_value_heads, hidden),
                        rows(value_heads, 1),
                    )?,
                    out_proj: uploader.mat_part(
                        source,
                        &format!("{prefix}.out_proj.weight"),
                        (hidden, config.linear_value_width()),
                        columns(value_heads, value_dim),
                    )?,
                    // A channel's taps are a row of the depthwise filter.
                    conv: uploader.vector_rows(
                        source,
                        &format!("{prefix}.conv1d.weight"),
                        (config.linear_conv_width(), config.linear_conv_kernel_dim),
                        &qkv_rows,
                    )?,
                    a_log: uploader.vector_rows(
                        source,
                        &format!("{prefix}.A_log"),
                        (config.linear_num_value_heads, 1),
                        &[value_heads],
                    )?,
                    dt_bias: uploader.vector_rows(
                        source,
                        &format!("{prefix}.dt_bias"),
                        (config.linear_num_value_heads, 1),
                        &[value_heads],
                    )?,
                    norm: uploader.vector(
                        source,
                        &format!("{prefix}.norm.weight"),
                        config.linear_value_head_dim,
                    )?,
                }
            }
            SharedPart::GqaAttention => {
                let prefix = format!("{prefix}.self_attn");
                // A query head's rows are its query and its gate, interleaved.
                GpuBranch::Sparse {
                    q_proj: uploader.mat_part(
                        source,
                        &format!("{prefix}.q_proj.weight"),
                        (config.q_proj_width(), hidden),
                        rows(heads, 2 * config.head_dim),
                    )?,
                    k_proj: uploader.mat_part(
                        source,
                        &format!("{prefix}.k_proj.weight"),
                        (config.kv_proj_width(), hidden),
                        rows(kv_heads, config.head_dim),
                    )?,
                    v_proj: uploader.mat_part(
                        source,
                        &format!("{prefix}.v_proj.weight"),
                        (config.kv_proj_width(), hidden),
                        rows(kv_heads, config.head_dim),
                    )?,
                    o_proj: uploader.mat_part(
                        source,
                        &format!("{prefix}.o_proj.weight"),
                        (hidden, config.attn_out_width()),
                        columns(heads, config.head_dim),
                    )?,
                    q_norm: uploader.vector(
                        source,
                        &format!("{prefix}.q_norm.weight"),
                        config.head_dim,
                    )?,
                    k_norm: uploader.vector(
                        source,
                        &format!("{prefix}.k_norm.weight"),
                        config.head_dim,
                    )?,
                    index_qk: uploader.mat(
                        source,
                        &format!("{prefix}.indexer.index_qk_proj.weight"),
                        (config.indexer_n_heads + config.indexer_kv_heads)
                            * config.indexer_head_dim,
                        hidden,
                    )?,
                    index_q_norm: uploader.vector(
                        source,
                        &format!("{prefix}.indexer.q_layernorm.weight"),
                        config.indexer_head_dim,
                    )?,
                    index_k_norm: uploader.vector(
                        source,
                        &format!("{prefix}.indexer.k_layernorm.weight"),
                        config.indexer_head_dim,
                    )?,
                }
            }
            other => {
                return Err(format!(
                    "layer {index} runs the {} attention, which this pack does not build",
                    other.name()
                ));
            }
        };
        if step.ffn != SharedPart::Moe {
            return Err(format!(
                "layer {index} runs the {} feed-forward, which this pack does not build",
                step.ffn.name()
            ));
        }
        let mlp = format!("{prefix}.mlp");
        // The two released checkpoints differ only in how the routed experts are stored: the same
        // [experts][2] pointer table either way, which is what lets one expert kernel serve both.
        let (expert_table, expert_scales) = match config.expert_quant {
            ExpertQuant::Nvfp4 => {
                let experts = (0..config.num_experts)
                    .map(|expert| {
                        let prefix = format!("{mlp}.experts.{expert}");
                        Ok(GpuExpert {
                            gate: uploader.nvfp4_part(
                                source,
                                &format!("{prefix}.gate_proj"),
                                (config.moe_intermediate_size, hidden),
                                rows(expert_share, 1),
                                expert_width,
                            )?,
                            up: uploader.nvfp4_part(
                                source,
                                &format!("{prefix}.up_proj"),
                                (config.moe_intermediate_size, hidden),
                                rows(expert_share, 1),
                                expert_width,
                            )?,
                            down: uploader.nvfp4_part(
                                source,
                                &format!("{prefix}.down_proj"),
                                (hidden, config.moe_intermediate_size),
                                columns(expert_share, 1),
                                expert_width,
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let mut pointers = Vec::with_capacity(experts.len() * 6 * 8);
                for expert in &experts {
                    for pointer in [
                        expert.gate.weight.0,
                        expert.gate.scales.0,
                        expert.up.weight.0,
                        expert.up.scales.0,
                        expert.down.weight.0,
                        expert.down.scales.0,
                    ] {
                        pointers.extend_from_slice(&pointer.to_le_bytes());
                    }
                }
                let mut scalars = Vec::with_capacity(experts.len() * 3 * 4);
                for expert in &experts {
                    for scale in [expert.gate.scale_2, expert.up.scale_2, expert.down.scale_2] {
                        scalars.extend_from_slice(&scale.to_le_bytes());
                    }
                }
                (pointers, Some(scalars))
            }
            ExpertQuant::Fp8Block => {
                // One E4M3 code a weight and one BF16 inverse scale a 128x128 tile: the MTP
                // experts' format, which the FP8 release uses for every layer. A rank holds a run
                // of whole tiles of every expert ([tp::Shard::expert_tiles]), not a fraction of
                // the width, so its weight and its scale grid are the same run of the global ones.
                // An FP8 expert's share is a run of whole 128-wide scale tiles, which is uneven
                // across ranks (640 columns are 2/1/1/1 tiles at TP4) where the NVFP4 share is an
                // even split padded up to the next 64.
                let (tile_first, tile_count) = shard.expert_tiles(config.moe_intermediate_size);
                let fp8_share = (tile_first * tp::FP8_TILE, tile_count * tp::FP8_TILE);
                let mut pointers = Vec::with_capacity(config.num_experts * 6 * 8);
                for expert in 0..config.num_experts {
                    let prefix = format!("{mlp}.experts.{expert}");
                    for (name, expert_rows, expert_columns, part) in [
                        (
                            "gate_proj",
                            config.moe_intermediate_size,
                            hidden,
                            rows(fp8_share, 1),
                        ),
                        (
                            "up_proj",
                            config.moe_intermediate_size,
                            hidden,
                            rows(fp8_share, 1),
                        ),
                        (
                            "down_proj",
                            hidden,
                            config.moe_intermediate_size,
                            columns(fp8_share, 1),
                        ),
                    ] {
                        let (weight, scales) = uploader.fp8_part(
                            source,
                            &format!("{prefix}.{name}"),
                            (expert_rows, expert_columns),
                            part,
                        )?;
                        pointers.extend_from_slice(&weight.0.to_le_bytes());
                        pointers.extend_from_slice(&scales.0.to_le_bytes());
                    }
                }
                (pointers, None)
            }
        };
        let moe = GpuMoe {
            router: uploader.mat(
                source,
                &format!("{mlp}.gate.weight"),
                config.num_experts,
                hidden,
            )?,
            shared: uploader.mat_part(
                source,
                &format!("{mlp}.shared_expert.gate_proj.weight"),
                (config.shared_expert_intermediate_size, hidden),
                rows(shared, 1),
            )?,
            shared_up: uploader.mat_part(
                source,
                &format!("{mlp}.shared_expert.up_proj.weight"),
                (config.shared_expert_intermediate_size, hidden),
                rows(shared, 1),
            )?,
            shared_down: uploader.mat_part(
                source,
                &format!("{mlp}.shared_expert.down_proj.weight"),
                (hidden, config.shared_expert_intermediate_size),
                columns(shared, 1),
            )?,
            shared_gate: uploader.mat(
                source,
                &format!("{mlp}.shared_expert_gate.weight"),
                1,
                hidden,
            )?,
            experts: match expert_scales {
                Some(scales) => GpuExperts::Nvfp4 {
                    table: uploader.derived(&format!("{mlp}.experts.table"), &expert_table)?,
                    scales: uploader.derived(&format!("{mlp}.experts.scales"), &scales)?,
                },
                None => GpuExperts::Fp8 {
                    table: uploader.derived(&format!("{mlp}.experts.table"), &expert_table)?,
                },
            },
        };
        let ple = if config.ple_layer_ids.contains(&(index + 1)) {
            let prefix = format!("{prefix}.ple");
            let scale = bytes_bf16_scalar(
                &source
                    .read(&format!(
                        "{prefix}.ple_embedding.ngram_embedding.weight_scale"
                    ))
                    .map_err(|error| error.to_string())?,
                "the PLE scale",
            )?;
            let reader = crate::checkpoint::CheckpointRows::open(directory, config, index)?;
            Some(GpuPle {
                key_proj: uploader.mat(
                    source,
                    &format!("{prefix}.key_proj.weight"),
                    width,
                    config.ple_embed_dim,
                )?,
                value_proj: uploader.mat(
                    source,
                    &format!("{prefix}.value_proj.weight"),
                    hidden,
                    config.ple_embed_dim,
                )?,
                norm_key: uploader.vector(source, &format!("{prefix}.norm_key.weight"), width)?,
                norm_query: uploader.vector(
                    source,
                    &format!("{prefix}.norm_query.weight"),
                    width,
                )?,
                norm_conv: uploader.vector(source, &format!("{prefix}.norm_conv.weight"), width)?,
                conv: uploader.vector(
                    source,
                    &format!("{prefix}.conv1d.weight"),
                    width * config.ple_conv_kernel_size,
                )?,
                table: std::sync::Mutex::new(PleTable::new(config, reader, scale)?),
            })
        } else {
            None
        };
        layers.push(GpuLayer {
            branch,
            attn_hyper: hyper(format!("{prefix}.attn_hyper_connection"), true)?,
            mlp_hyper: hyper(format!("{prefix}.mlp_hyper_connection"), true)?,
            moe,
            ple,
        });
    }
    // No rank holds a share of the MTP layer ([tp::rank_config]); a sharded rank drafts with the
    // DFlash 2 head or not at all.
    let mtp = if config.mtp_num_hidden_layers > 0 && shard.world == 1 {
        let prefix = "mtp";
        let layer = format!("{prefix}.layers.0");
        let attn_prefix = format!("{layer}.self_attn");
        let attn = GpuBranch::Sparse {
            q_proj: uploader.mat(
                source,
                &format!("{attn_prefix}.q_proj.weight"),
                config.q_proj_width(),
                hidden,
            )?,
            k_proj: uploader.mat(
                source,
                &format!("{attn_prefix}.k_proj.weight"),
                config.kv_proj_width(),
                hidden,
            )?,
            v_proj: uploader.mat(
                source,
                &format!("{attn_prefix}.v_proj.weight"),
                config.kv_proj_width(),
                hidden,
            )?,
            o_proj: uploader.mat(
                source,
                &format!("{attn_prefix}.o_proj.weight"),
                hidden,
                config.attn_out_width(),
            )?,
            q_norm: uploader.vector(
                source,
                &format!("{attn_prefix}.q_norm.weight"),
                config.head_dim,
            )?,
            k_norm: uploader.vector(
                source,
                &format!("{attn_prefix}.k_norm.weight"),
                config.head_dim,
            )?,
            index_qk: uploader.mat(
                source,
                &format!("{attn_prefix}.indexer.index_qk_proj.weight"),
                (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim,
                hidden,
            )?,
            index_q_norm: uploader.vector(
                source,
                &format!("{attn_prefix}.indexer.q_layernorm.weight"),
                config.indexer_head_dim,
            )?,
            index_k_norm: uploader.vector(
                source,
                &format!("{attn_prefix}.indexer.k_layernorm.weight"),
                config.indexer_head_dim,
            )?,
        };
        let mlp = format!("{layer}.mlp");
        let mut pointers = Vec::with_capacity(config.num_experts * 6 * 8);
        for expert in 0..config.num_experts {
            let expert_prefix = format!("{mlp}.experts.{expert}");
            let gate = uploader.fp8(
                source,
                &format!("{expert_prefix}.gate_proj"),
                config.moe_intermediate_size,
                hidden,
            )?;
            let up = uploader.fp8(
                source,
                &format!("{expert_prefix}.up_proj"),
                config.moe_intermediate_size,
                hidden,
            )?;
            let down = uploader.fp8(
                source,
                &format!("{expert_prefix}.down_proj"),
                hidden,
                config.moe_intermediate_size,
            )?;
            for pointer in [gate.0, gate.1, up.0, up.1, down.0, down.1] {
                pointers.extend_from_slice(&pointer.0.to_le_bytes());
            }
        }
        Some(GpuMtp {
            fc_embedding: uploader.mat(
                source,
                &format!("{prefix}.fc_embedding.weight"),
                hidden,
                hidden,
            )?,
            fc_hidden: uploader.mat(
                source,
                &format!("{prefix}.fc_hidden.weight"),
                hidden,
                hidden,
            )?,
            pre_norm_embedding: uploader.vector(
                source,
                &format!("{prefix}.pre_fc_norm_embedding.weight"),
                hidden,
            )?,
            pre_norm_hidden: uploader.vector(
                source,
                &format!("{prefix}.pre_fc_norm_hidden.weight"),
                width,
            )?,
            mixer: hyper(format!("{prefix}.hyper_connection_mixer"), false)?,
            attn_hyper: hyper(format!("{layer}.attn_hyper_connection"), true)?,
            mlp_hyper: hyper(format!("{layer}.mlp_hyper_connection"), true)?,
            attn,
            moe: GpuMoe {
                router: uploader.mat(
                    source,
                    &format!("{mlp}.gate.weight"),
                    config.num_experts,
                    hidden,
                )?,
                shared: uploader.mat(
                    source,
                    &format!("{mlp}.shared_expert.gate_proj.weight"),
                    config.shared_expert_intermediate_size,
                    hidden,
                )?,
                shared_up: uploader.mat(
                    source,
                    &format!("{mlp}.shared_expert.up_proj.weight"),
                    config.shared_expert_intermediate_size,
                    hidden,
                )?,
                shared_down: uploader.mat(
                    source,
                    &format!("{mlp}.shared_expert.down_proj.weight"),
                    hidden,
                    config.shared_expert_intermediate_size,
                )?,
                shared_gate: uploader.mat(
                    source,
                    &format!("{mlp}.shared_expert_gate.weight"),
                    1,
                    hidden,
                )?,
                experts: GpuExperts::Fp8 {
                    table: uploader.derived(&format!("{mlp}.experts.table"), &pointers)?,
                },
            },
        })
    } else {
        None
    };
    Ok(GpuWeights {
        embed,
        lm_head,
        mixer,
        layers,
        mtp,
    })
}

/// The TP=1 device backend.
pub struct Qwen4ExpDeviceBackend {
    descriptor: ModelDescriptor,
    shared: Arc<SequenceShared>,
    context_limit: usize,
    /// The sessions-stepped-together engine, when the operator asked for one. Unset means every
    /// session runs its own forward on its own stream, the one-row-a-session path.
    fused: Option<Arc<sparkflash_model::FusedEngine<fusion::QwenFusion>>>,
}

impl std::fmt::Debug for Qwen4ExpDeviceBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen4ExpDeviceBackend")
            .field("descriptor", &self.descriptor.id)
            .field("context_limit", &self.context_limit)
            .field("layers", &self.shared.weights.layers.len())
            .finish()
    }
}

/// The merged configuration every open path serves from.
///
/// [Qwen4ExpConfig::from_checkpoint_dir] reads `config.json` through the shared architecture,
/// merges `generation_config.json`, and appends the pack's `<|im_end|>`; parsing the config
/// document alone here is what made a generation-only stop id invisible to the session.
fn open_config(directory: &Path) -> Result<Qwen4ExpConfig, BackendError> {
    Qwen4ExpConfig::from_checkpoint_dir(directory)
        .map_err(|error| BackendError::new("qwen4_config", error.to_string()))
}

impl Qwen4ExpDeviceBackend {
    /// Open the checkpoint for one device rank and upload every resident weight.
    ///
    /// # Errors
    /// A missing cubin, a context limit outside the model's window, a missing or malformed
    /// checkpoint, or a device allocation/load failure. There is no host fallback.
    pub fn open(checkpoint: &Path, context_limit: usize) -> Result<Self, BackendError> {
        Self::open_for_world(checkpoint, context_limit, 1)
    }

    /// [Self::open] with an explicit prefill plan instead of the environment's, so a test can
    /// pin the chunk width and choose between the token kernels and the tensor-core image.
    ///
    /// # Errors
    /// Everything [Self::open] refuses.
    pub fn open_with_prefill(
        checkpoint: &Path,
        context_limit: usize,
        prefill: PrefillPlan,
    ) -> Result<Self, BackendError> {
        Self::open_inner(checkpoint, context_limit, prefill, None, None, None)
    }

    /// [Self::open] from an already-built execution plan.
    ///
    /// The plan's architecture replaces the typed configuration's layer schedule, so the weight
    /// walk reads the same layer selection the boot resolver checked. The typed config still owns
    /// the tensor shapes and weight names.
    ///
    /// # Errors
    /// Everything [Self::open] refuses, and a plan of another family or layer count.
    pub fn open_with_plan(
        checkpoint: &Path,
        context_limit: usize,
        plan: &EnginePlan,
    ) -> Result<Self, BackendError> {
        Self::open_inner(
            checkpoint,
            context_limit,
            PrefillPlan::from_env()?,
            None,
            None,
            Some(plan),
        )
    }

    /// [Self::open_converted] from an already-built execution plan.
    ///
    /// # Errors
    /// Everything [Self::open_converted] refuses, and a plan of another family or layer count.
    pub fn open_converted_with_plan(
        checkpoint: &Path,
        context_limit: usize,
        rank_file: &Path,
        plan: &EnginePlan,
    ) -> Result<Self, BackendError> {
        Self::open_inner(
            checkpoint,
            context_limit,
            PrefillPlan::from_env()?,
            Some(rank_file),
            None,
            Some(plan),
        )
    }

    /// [Self::open] over a converted rank file instead of the checkpoint's tensors, so the
    /// weights are one parallel read into one allocation. The checkpoint directory is still the
    /// configuration, tokenizer and host-paged PLE source.
    ///
    /// # Errors
    /// Everything [Self::open] refuses, and a rank file this configuration did not write.
    pub fn open_converted(
        checkpoint: &Path,
        context_limit: usize,
        rank_file: &Path,
    ) -> Result<Self, BackendError> {
        Self::open_inner(
            checkpoint,
            context_limit,
            PrefillPlan::from_env()?,
            Some(rank_file),
            None,
            None,
        )
    }

    /// [Self::open_converted] with an explicit prefill plan.
    ///
    /// # Errors
    /// Everything [Self::open_converted] refuses.
    pub fn open_with_prefill_converted(
        checkpoint: &Path,
        context_limit: usize,
        prefill: PrefillPlan,
        rank_file: &Path,
    ) -> Result<Self, BackendError> {
        Self::open_inner(
            checkpoint,
            context_limit,
            prefill,
            Some(rank_file),
            None,
            None,
        )
    }

    /// Open the checkpoint for a world of this size with no collective to sum its partials over.
    ///
    /// Only a world of one has nothing to sum: a larger one is one rank of a tensor-parallel
    /// group, which [Self::open_rank] opens with the group's collective. A world larger than one
    /// is a typed refusal here rather than a rank that would quietly compute a fraction of the
    /// model and answer as if it had not.
    ///
    /// # Errors
    /// The refusal, then everything [Self::open] refuses.
    pub fn open_for_world(
        checkpoint: &Path,
        context_limit: usize,
        world_size: usize,
    ) -> Result<Self, BackendError> {
        if world_size != 1 {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "a {world_size}-rank group sums every branch over its collective; open each \
                     rank with Qwen4ExpDeviceBackend::open_rank"
                ),
            ));
        }
        Self::open_inner(
            checkpoint,
            context_limit,
            PrefillPlan::from_env()?,
            None,
            None,
            None,
        )
    }

    /// Open rank `tp.rank()` of a tensor-parallel group of `tp.world()` ranks: this rank's share
    /// of every sharded weight ([tp::rank_config] says which), with each branch's partials summed
    /// over `tp` before they are injected, so every rank holds the same residual streams. The
    /// drafter runs whole on every rank; the MTP layer is not loaded.
    ///
    /// The rank serves one sequence at a time - every forward is a collective the whole group
    /// steps through together - so it refuses [Self::enable_fusion].
    ///
    /// # Errors
    /// A world the model's heads or widths do not split over, then everything [Self::open]
    /// refuses.
    pub fn open_rank(
        checkpoint: &Path,
        context_limit: usize,
        tp: TpRuntime,
    ) -> Result<Self, BackendError> {
        Self::open_inner(
            checkpoint,
            context_limit,
            PrefillPlan::from_env()?,
            None,
            Some(tp),
            None,
        )
    }

    /// [Self::open_rank] over rank `tp.rank()`'s converted file instead of the checkpoint's
    /// tensors: one parallel read into one allocation. The file holds the rank's share, so it
    /// must have been written for the same `tp.world()` (see
    /// [crate::convert::convert_rank]). The checkpoint directory is still the configuration,
    /// tokenizer and host-paged PLE source.
    ///
    /// # Errors
    /// Everything [Self::open_rank] refuses, and a rank file of another world or rank.
    pub fn open_rank_converted(
        checkpoint: &Path,
        context_limit: usize,
        tp: TpRuntime,
        rank_file: &Path,
    ) -> Result<Self, BackendError> {
        Self::open_inner(
            checkpoint,
            context_limit,
            PrefillPlan::from_env()?,
            Some(rank_file),
            Some(tp),
            None,
        )
    }

    /// [Self::open_rank_converted] with an explicit prefill plan, so a test can pin the chunk
    /// width.
    ///
    /// # Errors
    /// Everything [Self::open_rank_converted] refuses.
    pub fn open_with_prefill_rank_converted(
        checkpoint: &Path,
        context_limit: usize,
        prefill: PrefillPlan,
        tp: TpRuntime,
        rank_file: &Path,
    ) -> Result<Self, BackendError> {
        Self::open_inner(
            checkpoint,
            context_limit,
            prefill,
            Some(rank_file),
            Some(tp),
            None,
        )
    }

    /// A sequence of an [Self::open_rank] group that drafts only when `speculative` says so.
    ///
    /// Every rank of the group must take the same steps: a speculative step verifies a draft
    /// chain, a plain one a token, and each is a collective. The ranks that do not sample run
    /// greedy, so only a request whose tokens are each its row's argmax may speculate; rank 0
    /// decides and every rank opens the request's sequence with that decision.
    ///
    /// # Errors
    /// A window that does not fit beside the weights.
    pub fn open_rank_session(
        &self,
        speculative: bool,
    ) -> Result<Box<dyn ModelSession>, BackendError> {
        let mut session = DeviceSequence::allocate(Arc::clone(&self.shared), self.context_limit)?;
        session.spec_enabled &= speculative;
        Ok(Box::new(session))
    }

    fn open_inner(
        checkpoint: &Path,
        context_limit: usize,
        prefill: PrefillPlan,
        converted: Option<&Path>,
        tp: Option<TpRuntime>,
        plan: Option<&EnginePlan>,
    ) -> Result<Self, BackendError> {
        let directory = if checkpoint.is_dir() {
            checkpoint.to_path_buf()
        } else {
            checkpoint.parent().unwrap_or(Path::new(".")).to_path_buf()
        };
        let mut config = open_config(&directory)?;
        if let Some(plan) = plan {
            if plan.family() != "qwen4_exp" {
                return Err(BackendError::new(
                    "qwen4_plan",
                    format!(
                        "the execution plan is {}, the Qwen arm is qwen4_exp",
                        plan.family()
                    ),
                ));
            }
            if plan.layers().len() != config.num_hidden_layers {
                return Err(BackendError::new(
                    "qwen4_plan",
                    format!(
                        "the execution plan has {} layers, the Qwen configuration has {}",
                        plan.layers().len(),
                        config.num_hidden_layers
                    ),
                ));
            }
            // The boot plan is the layer selection the resolver checked; the typed config keeps
            // the shapes and weight names. The plan parsed `config.json` on its own, so put the
            // merged generation stop set back: the backend and the codec must read one list.
            config.architecture = plan.architecture().clone();
            config.architecture.tokens.eos_token_ids = config.eos_token_ids.clone();
        }
        let config = Arc::new(config);
        if context_limit == 0 || context_limit > config.max_position_embeddings {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "context limit {context_limit} is outside 1..={}",
                    config.max_position_embeddings
                ),
            ));
        }
        // The kernels hold their per-block scratch in fixed shared arrays, and each guards itself by
        // returning early. A configuration past one of those bounds would therefore produce a
        // silently wrong answer rather than a crash, so it is refused here instead.
        if config.hc_lowrank > MAX_LOWRANK {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "hc_lowrank {} exceeds the kernel's {MAX_LOWRANK}-row shared buffer",
                    config.hc_lowrank
                ),
            ));
        }
        if config.moe_intermediate_size > MAX_INTERMEDIATE {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "moe_intermediate_size {} exceeds the expert kernel's {MAX_INTERMEDIATE}-wide shared buffer",
                    config.moe_intermediate_size
                ),
            ));
        }
        if config.indexer_block_budget() > MAX_BUDGET_BLOCKS {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "the indexer budget selects {} blocks, over the selection kernel's {MAX_BUDGET_BLOCKS}",
                    config.indexer_block_budget()
                ),
            ));
        }
        if config.max_selected_tokens() > MAX_SELECTED_TOKENS {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "the indexer selects up to {} tokens, over the attention kernel's \
                     {MAX_SELECTED_TOKENS}",
                    config.max_selected_tokens()
                ),
            ));
        }
        if config.indexer_head_dim > MAX_INDEXER_HEAD_DIM
            || config.indexer_n_heads > MAX_INDEXER_HEADS
        {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "the indexer's {} heads of {} channels exceed the score kernel's \
                     {MAX_INDEXER_HEADS} of {MAX_INDEXER_HEAD_DIM}",
                    config.indexer_n_heads, config.indexer_head_dim
                ),
            ));
        }
        // The checkpoint is checked before the build is: a wrong context limit or a malformed
        // configuration is a fact about the request, and reporting it should not depend on whether
        // this binary happens to carry a cubin.
        if embedded_cubin().is_empty() {
            return Err(BackendError::new(
                "qwen4_device",
                "this build has no Qwen4-Exp sm_121 cubin. Rebuild with nvcc available (or \
                 SPARKFLASH_NVCC pointing at it) so build.rs compiles \
                 kernels/qwen4_exp_sm121.cu.",
            ));
        }
        let cuda = Cuda::load()
            .map_err(|error| BackendError::new("qwen4_device", format!("CUDA: {error}")))?;
        sparkflash_kernels::cuda_driver::enable_forward_async_launches();
        let source = SafeTensorsSource::open(&directory, &directory.join(INDEX_NAME))
            .map_err(|error| BackendError::new("qwen4_weights", error.to_string()))?;
        // A conformance check before a 77.8 GiB upload. Every served tensor must be present, and
        // any other name must be one the whole-checkpoint plan knows: the released checkpoint
        // carries an MTP stage (3,101 tensors) and a vision tower (333), both planned and
        // deliberately not served by this text-only arm. Their presence is not a mismatch; a name
        // neither plan knows still is. Compared as sets, because the plan is in plan order and the
        // index is in whatever order the weight map sorts to.
        let served = crate::tensor::Qwen4ExpTensorPlan::served(&config);
        let full = crate::tensor::Qwen4ExpTensorPlan::checkpoint(&config);
        served
            .conformance_with_unserved(source.names(), &full)
            .map_err(|detail| {
                BackendError::new(
                    "qwen4_weights",
                    format!("the checkpoint does not carry the served tensor set: {detail}"),
                )
            })?;
        // The uploaded set is the plan minus the PLE table role: the 128 shards and the 282 bytes of
        // hash metadata around them stay host-side and are read through the pager. Deriving this
        // from the roles is the arithmetic the resident constant was computed with; the substring
        // form counted those 282 bytes as uploaded and refused the released checkpoint by exactly
        // that much. Checked before the upload, because a plan that overshoots should cost nothing.
        let resident = served.storage_bytes() - served.role_bytes(Role::PleTable);
        // The resident total is a property of the stored expert format: the same text model with
        // FP8 experts is 121.7 GiB where the NVFP4 one is 72.5 GiB.
        let total = match config.expert_quant {
            ExpertQuant::Nvfp4 => DEVICE_RESIDENT_BYTES,
            ExpertQuant::Fp8Block => crate::tensor::FP8_DEVICE_RESIDENT_BYTES,
        };
        if resident > total {
            return Err(BackendError::new(
                "qwen4_device",
                format!("the plan uploads {resident} bytes over the resident total {total}"),
            ));
        }
        let shard = tp.as_ref().map_or(tp::Shard::WHOLE, |tp| {
            tp::Shard::of(TpLayout {
                rank: tp.rank(),
                world_size: tp.world(),
            })
        });
        // The rank's kernels run on its share; the walk reads the whole tensors and keeps it.
        let local = tp::rank_config(&config, shard)
            .map_err(|error| BackendError::new("qwen4_tp", error))?;
        let weights =
            match converted {
                Some(path) => crate::convert::load_converted(
                    &cuda,
                    &config,
                    &directory,
                    path,
                    &source,
                    TpLayout {
                        rank: shard.rank,
                        world_size: shard.world,
                    },
                )
                .map_err(|error| {
                    BackendError::new(
                        "qwen4_weights",
                        format!("converted rank {}: {error}", path.display()),
                    )
                })?,
                None => GpuWeights::upload(&cuda, &directory, &source, &config, shard).map_err(
                    |error| BackendError::new("qwen4_device", format!("weight upload: {error}")),
                )?,
            };
        // The GDN recurrence's state is one head's `key_dim x value_dim` floats; staging it in
        // shared memory turns three global trips per token into three per chunk.
        let stage =
            config.linear_key_head_dim * config.linear_value_head_dim * std::mem::size_of::<f32>();
        let kernels = Kernels::load(&cuda, u32::try_from(stage).unwrap_or(0))
            .map_err(|error| BackendError::new("qwen4_device", format!("kernel load: {error}")))?;
        // A configured drafter that cannot be loaded is refused rather than served without: the
        // caller asked for it by name.
        let dflash = dflash::load_dflash_from_env(&cuda, &config, &weights.lm_head)?;
        // One line, because a run that silently drafts nothing is indistinguishable from one whose
        // drafter failed to resolve: the flag and the cache are the two knobs that decide it.
        match (&dflash, crate::dflash::drafter_directory()) {
            (Some(draft), directory) => eprintln!(
                "qwen4: draft head {} (block {}) from {}",
                crate::dflash::DFLASH2_MODEL_ID,
                draft.config.block_size,
                directory.map_or_else(|| "?".to_owned(), |path| path.display().to_string())
            ),
            (None, directory) => eprintln!(
                "qwen4: no draft head (SPARKFLASH_QWEN4_DFLASH2 is unset and the cache has no {} snapshot); directory {}",
                crate::dflash::DFLASH2_MODEL_ID,
                directory.map_or_else(
                    || "unresolved".to_owned(),
                    |path| path.display().to_string()
                )
            ),
        }
        let verify_rows = dflash.as_ref().map_or(MTP_VERIFY_ROWS, |draft| {
            draft.config.block_size.max(MTP_VERIFY_ROWS)
        });
        let tp = tp
            .map(|runtime| {
                tp::TensorParallel::new(&cuda, runtime, prefill.chunk, config.hidden_size)
            })
            .transpose()?;
        if let Some(tp) = &tp {
            eprintln!(
                "qwen4: rank {} of {}: {} query heads, {} GDN value heads, experts {} wide; {} collective",
                tp.runtime.rank(),
                tp.runtime.world(),
                local.num_attention_heads,
                local.linear_num_value_heads,
                local.moe_intermediate_size,
                tp.runtime.label()
            );
        }
        let shared = Arc::new(SequenceShared {
            cuda: CudaHandle(cuda),
            kernels: KernelSet(kernels),
            config: Arc::new(local),
            weights,
            dflash,
            prefill,
            pooled: std::sync::Mutex::new(None),
            verify_rows,
            tp,
        });
        let _ = DeviceSequence::allocate(Arc::clone(&shared), context_limit)?;
        Ok(Self {
            descriptor: descriptor_for(&config, context_limit),
            shared,
            context_limit,
            fused: None,
        })
    }

    /// The context window this backend allocated.
    #[must_use]
    pub fn context_limit(&self) -> usize {
        self.context_limit
    }

    /// Serve this backend's sessions as one batch: `capacity` sequences whose decode steps share a
    /// pass over the weights.
    ///
    /// Call it before the backend is shared with the serving driver, with the driver's concurrency
    /// as the capacity: each request takes one of the pooled sequences for its lifetime, so a
    /// smaller capacity than the driver's concurrency would leave requests waiting for a slot. A
    /// fused session does not speculate, and a prefill runs alone.
    ///
    /// # Errors
    /// A pooled session the device refuses, or a thread that cannot be started.
    pub fn enable_fusion(&mut self, capacity: usize) -> Result<(), BackendError> {
        if let Some(tp) = &self.shared.tp {
            return Err(BackendError::new(
                "qwen4_tp",
                format!(
                    "rank {} of {} serves one sequence at a time: every forward is a collective \
                     its group steps through together",
                    tp.runtime.rank(),
                    tp.runtime.world()
                ),
            ));
        }
        let arm = Arc::new(fusion::QwenFusion::new(
            Arc::clone(&self.shared),
            self.context_limit,
            capacity,
        ));
        self.fused = Some(sparkflash_model::FusedEngine::start(arm, capacity)?);
        Ok(())
    }

    /// Whether this backend steps its sessions together.
    #[must_use]
    pub fn fused(&self) -> bool {
        self.fused.is_some()
    }

    /// Run a small arithmetic self-test on the device: one RMSNorm and one BF16 GEMV.
    ///
    /// This proves the cubin loads and executes; it is not a model run and claims no parity.
    ///
    /// # Errors
    /// When a kernel fails or the read-back disagrees with the host computation.
    pub fn self_test(&self) -> Result<(), BackendError> {
        self.shared
            .cuda
            .make_current()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let mut total = 0usize;
        let width = 8usize;
        let input: Vec<f32> = (0..width)
            .map(|index| (index as f32) * 0.25 + 0.5)
            .collect();
        let weight = vec![half::bf16::from_f32(0.0); width];
        let mean_square: f32 = input.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let scale = 1.0 / (mean_square + 1e-6).sqrt();
        let expected: Vec<f32> = input.iter().map(|value| value * scale).collect();
        let mut weight_bytes = Vec::with_capacity(width * 2);
        for value in &weight {
            weight_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut input_bytes = Vec::with_capacity(width * 4);
        for value in &input {
            input_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let device_input = self
            .shared
            .cuda
            .alloc(input_bytes.len(), &mut total)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let device_weight = self
            .shared
            .cuda
            .alloc(weight_bytes.len(), &mut total)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let device_output = self
            .shared
            .cuda
            .alloc(input_bytes.len(), &mut total)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        self.shared
            .cuda
            .copy_htod(&device_input, &input_bytes)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        self.shared
            .cuda
            .copy_htod(&device_weight, &weight_bytes)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        self.shared
            .cuda
            .launch(
                self.shared.kernels.rmsnorm,
                (1, 1, 1),
                (256, 1, 1),
                &[
                    device_input.0,
                    device_weight.0,
                    device_output.0,
                    width as u64,
                    width as u64,
                    (1e-6f32).to_bits() as u64,
                    1,
                    1,
                ],
            )
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let mut bytes = vec![0u8; width * 4];
        self.shared
            .cuda
            .copy_dtoh(&mut bytes, &device_output)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if (value - expected[index]).abs() > 1e-5 {
                return Err(BackendError::new(
                    "qwen4_device",
                    format!(
                        "rmsnorm mismatch at {index}: {value} vs {}",
                        expected[index]
                    ),
                ));
            }
        }
        self.shared
            .cuda
            .free_buffers(&[&device_input, &device_weight, &device_output]);
        Ok(())
    }
}

/// This pack's identity in the registry.
pub const IDENTITY: PackIdentity = PackIdentity {
    model_id: crate::MODEL_ID,
    family: crate::FAMILY,
    model_type: "qwen4_exp",
    label: "Qwen",
    weights_id: crate::WEIGHTS_ID,
    max_output_tokens: crate::MAX_OUTPUT_TOKENS,
};

/// The official FP8 checkpoint's identity. The geometry is [IDENTITY]'s; only the served id and
/// the weight selector name the other quantisation.
pub const FP8_IDENTITY: PackIdentity = PackIdentity {
    weights_id: sparkflash_model::descriptor::QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID,
    model_id: sparkflash_model::descriptor::QWEN38_FLASH_NEXT_FP8_ID,
    ..IDENTITY
};

/// What this arm can serve of `config` once loaded.
///
/// The text channels are the codec's. The vision tower is planned but not loaded, so the channel
/// is not claimed; a build that loads it returns [Capabilities::ALL_WITH_VISION].
#[must_use]
pub fn capabilities(_config: &Qwen4ExpConfig) -> Capabilities {
    Capabilities::ALL
}

/// What the device arm serves over a window of `context` tokens.
#[must_use]
pub fn descriptor_for(config: &Qwen4ExpConfig, context: usize) -> ModelDescriptor {
    let identity = match config.expert_quant {
        ExpertQuant::Nvfp4 => IDENTITY,
        ExpertQuant::Fp8Block => FP8_IDENTITY,
    };
    identity.descriptor(&config.architecture, context, capabilities(config))
}

/// A per-session device workspace and cache set.
struct DeviceSequence {
    backend: Arc<SequenceShared>,
    /// This session's stream: every copy and launch of its forward is ordered here.
    stream: RunStream,
    /// Where the launches go while the session is a member of a [batch::DeviceBatch]: the batch's
    /// stream, so a step's row-independent stages read the weights once for every member.
    batch_stream: Option<RunStream>,
    /// Recorded after the PLE staging copy; the host waits on it before rewriting the slab.
    ple_staged: StagingFence,
    /// Recorded after the embed that reads the token slab, the same way.
    tokens_staged: StagingFence,
    position: usize,
    /// Every token this sequence has forwarded, oldest first: every prefill's fresh tokens and
    /// every decode's. The serving adapter may keep the sequence between requests and hand it a
    /// prompt that starts with these ([ModelSession::reusable_prefix]).
    consumed: Vec<u32>,
    /// How many of the last prefill's tokens its caches already held, for the request's usage.
    prefill_cached: usize,
    /// Floats per token in the indexer's block-score row; the chunk's score grid strides by it.
    score_stride: usize,
    /// Each sparse layer's BF16 key cache, `context_limit * kv_proj_width` elements; the MTP
    /// layer, when present, is one extra slot after the stack.
    keys: Vec<Option<Deviceptr>>,
    /// Each sparse layer's BF16 value cache, the same shape as [Self::keys].
    values: Vec<Option<Deviceptr>>,
    /// Each sparse layer's BF16 indexer-key cache, `context_limit * indexer_head_dim` elements;
    /// the MTP layer, when present, is one extra slot after the stack.
    index_keys: Vec<Option<Deviceptr>>,
    gdn_conv: Vec<Option<Deviceptr>>,
    gdn_recurrent: Vec<Option<Deviceptr>>,
    /// The other buffer of each GDN layer's recurrent pair: a verify leaves the live state as it
    /// found it and its last row's state here, which a step that keeps fewer rows overwrites by
    /// replaying them, before the two swap. Until the next verify it holds the state from before
    /// the last one, which a late rollback replays.
    gdn_spare: Vec<Option<Deviceptr>>,
    /// Each GDN layer's verify inputs - the convolved rows, then `b` and `a` - which the replay
    /// reads: `verify_rows` rows of the convolution width and two of the value heads.
    gdn_replay: Vec<Option<Deviceptr>>,
    ple_conv: Vec<Option<Deviceptr>>,
    hyper: Deviceptr,
    normed: Deviceptr,
    mixed: Deviceptr,
    injection: Deviceptr,
    attention_out: Deviceptr,
    branch: Deviceptr,
    qg: Deviceptr,
    query: Deviceptr,
    gate: Deviceptr,
    kv_row: Deviceptr,
    value_row: Deviceptr,
    index_row: Deviceptr,
    index_query: Deviceptr,
    selected: Deviceptr,
    selected_count: Deviceptr,
    /// One score per compressed indexer block, written by `q4e_qsa_scores` and folded by
    /// `q4e_qsa_pick`.
    qsa_block_scores: Deviceptr,
    /// Every compressed indexer block's pooled key, written by `q4e_qsa_pool` once a layer for
    /// all the rows of a forward.
    qsa_pooled_keys: Deviceptr,
    linear_qkv: Deviceptr,
    linear_conv: Deviceptr,
    linear_z: Deviceptr,
    linear_b: Deviceptr,
    linear_a: Deviceptr,
    linear_out: Deviceptr,
    logits: Deviceptr,
    probabilities: Deviceptr,
    expert_indices: Deviceptr,
    /// The routed block order the shared plan writes, so the expert kernels need no inline sort.
    expert_perm: Deviceptr,
    /// [slots][2] First and last rank of each rank's expert run, from the shared plan.
    expert_run: Deviceptr,
    expert_accumulator: Deviceptr,
    /// [slots][intermediate] SwiGLU rows the two expert passes share.
    expert_activated: Deviceptr,
    /// The hyper-connection's lowrank reduction, between its two passes.
    hyper_low: Deviceptr,
    /// [tokens][hc * hidden] Raw dot products of the hyper read pass, before the stream mixing.
    hyper_mix_raw: Deviceptr,
    /// Split-GEMM segment sums, [segments][tokens][rows], when the rows are too few to fill the GPU.
    gemm_partials: Deviceptr,
    expert_gate: Deviceptr,
    shared_hidden: Deviceptr,
    shared_up: Deviceptr,
    shared_out: Deviceptr,
    ple_embedding: Deviceptr,
    ple_key: Deviceptr,
    ple_query: Deviceptr,
    ple_value: Deviceptr,
    ple_gated: Deviceptr,
    ple_normed: Deviceptr,
    ple_conv_out: Deviceptr,
    /// Rewritten only once this session's staging fence says the last copy out of it has run.
    ple_host_staging: HostMapped,
    /// A chunk's token ids, which the embed reads in place: one launch embeds the whole chunk.
    token_staging: HostMapped,
    ple_hashers: Vec<Option<crate::ngram::PleHasher>>,
    /// The widened stream the MTP layer reads: one row per position, assembled shifted by one so
    /// that entry `q` is (hidden_{q-1}, embedding(t_q)).
    mtp_rows: Deviceptr,
    /// The MTP module's fold of `mtp_rows` and the next tokens' normed embeddings.
    mtp_folded: Deviceptr,
    mtp_embed: Deviceptr,
    /// The drafted layer's widened residual, folded again by its own mixer.
    mtp_hyper: Deviceptr,
    /// The last widened row of the previous forward, which is the MTP's row `q - 1`.
    mtp_stash: Deviceptr,
    /// The convolution windows after each of a speculative verify's rows but its last, one
    /// region of `snapshot_stride` floats per row, restored when the draft past that row was
    /// rejected. The GDN recurrent state is not here: it is replayed (`gdn_spare`).
    snapshot: Deviceptr,
    snapshot_stride: usize,
    /// One arrival counter per tile of a split tiled GEMV, zero between launches (the tile's last
    /// block resets it). The split sums go through `gemm_partials`, which a decode step never
    /// shares with a prefill GEMM.
    tiled_counters: Deviceptr,
    /// The allocation `gdn_replay` slices.
    replay_inputs: Deviceptr,
    /// The tensor-core prefill path's own buffers.
    prefill: PrefillScratch,
    /// The DFlash 2 drafter's feature rows, context rings and block buffers, when it is loaded.
    draft: Option<DraftScratch>,
    /// The positions the caches hold; a verify that would run past it is never drafted.
    context_limit: usize,
    /// Verified answers whose tokens the engine has not consumed yet, oldest first.
    spec_queue: Vec<SpecPending>,
    /// The MTP module's proposals for the positions after the last step, oldest first.
    spec_drafts: Vec<u32>,
    /// The DFlash 2 drafter's confidence in each of `spec_drafts`.
    spec_confidence: Vec<f32>,
    /// How deep to draft and what the drafts were worth: the shared rule's depth and counters.
    spec: sparkflash_model::Speculator,
    /// What `mtp_folded` is scaled by: a mean over the streams, unless the checkpoint wants the
    /// sum, which `SPARKFLASH_QWEN4_MTP_FOLD=sum` selects.
    mtp_fold_scale: f32,
    spec_enabled: bool,
    /// This rank's `(first, count)` rows while a layer of a forward runs sequence-parallel
    /// ([Self::sequence_parallel]); None outside one.
    own_rows: Option<(usize, usize)>,
    /// The first row of the rows a sequence-parallel collective spans: 0, or the second half's
    /// first while a chunk runs as two halves ([Self::overlapped_layers]).
    window: usize,
    /// The second stream of a chunk that runs as two halves, made with the first such chunk.
    overlap: Option<tp::Overlap>,
    spec_trace: bool,
    /// The rows of the forward in flight, so a hash probe can name each row's position.
    probe_rows: usize,
    sampling: Sampling,
    sampler: crate::sampling::Sampler,
    eos: Vec<u32>,
    trace: bool,
}

/// The answer a speculative step already computed for the engine's next `decode` call.
///
/// The pair's second position is only real if the caller feeds the token the model chose; anything
/// here is what it takes to undo it when the caller feeds a different one.
struct SpecPending {
    /// The token the engine must consume for this answer to be the right one.
    token: u32,
    /// The logits at the position after it.
    logits: Vec<f32>,
    /// The token those logits chose, sampled once so the MTP builds on the token the engine emits.
    next_token: Option<u32>,
    /// The position this token occupies, which a mismatch rolls the model back to.
    position: usize,
    /// Which snapshot region restores the state to just before that position: region `r` holds
    /// the state after the chain's row `r`.
    region: usize,
    /// The n-gram state from before the forward, replayed past the rows that were kept.
    hashers: std::sync::Arc<Vec<Option<crate::ngram::PleHasher>>>,
    /// The rows of that forward the model agreed with, which the windows must be pushed past.
    replay: Vec<u32>,
}

struct ReusableBuffers {
    keys: Vec<Option<Deviceptr>>,
    values: Vec<Option<Deviceptr>>,
    index_keys: Vec<Option<Deviceptr>>,
    gdn_conv: Vec<Option<Deviceptr>>,
    gdn_recurrent: Vec<Option<Deviceptr>>,
    gdn_spare: Vec<Option<Deviceptr>>,
    ple_conv: Vec<Option<Deviceptr>>,
    scratch: [Deviceptr; 52],
    prefill: PrefillScratch,
    draft: Option<DraftScratch>,
    ple_host_staging: HostMapped,
    token_staging: HostMapped,
}

impl ReusableBuffers {
    fn free(&self, cuda: &Cuda) {
        let _ = cuda.make_current();
        let mut ptrs: Vec<&Deviceptr> = self.scratch.iter().collect();
        ptrs.extend(self.prefill.pointers());
        if let Some(draft) = &self.draft {
            ptrs.extend(draft.pointers());
        }
        ptrs.extend(
            self.keys
                .iter()
                .chain(&self.values)
                .chain(&self.index_keys)
                .chain(&self.gdn_conv)
                .chain(&self.gdn_recurrent)
                .chain(&self.gdn_spare)
                .chain(&self.ple_conv)
                .flatten(),
        );
        cuda.free_buffers(&ptrs);
        cuda.free_host_mapped(&self.ple_host_staging);
        cuda.free_host_mapped(&self.token_staging);
    }
}

impl Drop for DeviceSequence {
    fn drop(&mut self) {
        // Drain this session's stream before returning its buffers to the pool. No other session
        // uses this stream, so this waits for this session's work only.
        let _ = self.backend.cuda.make_current();
        let _ = self.backend.cuda.synchronize_stream(&self.stream);
        if let Some(overlap) = self.overlap.take() {
            overlap.release(&self.backend.cuda);
        }
        let reusable = ReusableBuffers {
            keys: std::mem::take(&mut self.keys),
            values: std::mem::take(&mut self.values),
            index_keys: std::mem::take(&mut self.index_keys),
            gdn_conv: std::mem::take(&mut self.gdn_conv),
            gdn_recurrent: std::mem::take(&mut self.gdn_recurrent),
            gdn_spare: std::mem::take(&mut self.gdn_spare),
            ple_conv: std::mem::take(&mut self.ple_conv),
            scratch: [
                self.hyper,
                self.normed,
                self.mixed,
                self.injection,
                self.attention_out,
                self.branch,
                self.qg,
                self.query,
                self.gate,
                self.kv_row,
                self.value_row,
                self.index_row,
                self.index_query,
                self.selected,
                self.selected_count,
                self.qsa_block_scores,
                self.linear_qkv,
                self.linear_conv,
                self.linear_z,
                self.linear_b,
                self.linear_a,
                self.linear_out,
                self.logits,
                self.probabilities,
                self.expert_indices,
                self.expert_accumulator,
                self.expert_activated,
                self.hyper_low,
                self.expert_gate,
                self.shared_hidden,
                self.shared_up,
                self.shared_out,
                self.ple_embedding,
                self.ple_key,
                self.ple_query,
                self.ple_value,
                self.ple_gated,
                self.ple_normed,
                self.ple_conv_out,
                self.expert_perm,
                self.expert_run,
                self.hyper_mix_raw,
                self.gemm_partials,
                self.mtp_rows,
                self.mtp_folded,
                self.mtp_embed,
                self.mtp_hyper,
                self.mtp_stash,
                self.snapshot,
                self.tiled_counters,
                self.replay_inputs,
                self.qsa_pooled_keys,
            ],
            prefill: self.prefill,
            draft: self.draft,
            ple_host_staging: self.ple_host_staging,
            token_staging: self.token_staging,
        };
        if let Ok(mut guard) = self.backend.pooled.lock()
            && guard.is_none()
        {
            *guard = Some(reusable);
            self.backend.cuda.destroy_event(&self.ple_staged.0);
            self.backend.cuda.destroy_event(&self.tokens_staged.0);
            self.backend.cuda.destroy_run_stream(&self.stream);
            return;
        }
        reusable.free(&self.backend.cuda);
        self.backend.cuda.destroy_event(&self.ple_staged.0);
        self.backend.cuda.destroy_event(&self.tokens_staged.0);
        self.backend.cuda.destroy_run_stream(&self.stream);
    }
}

struct KernelSet(Kernels);

/// The event a session records after each copy out of its PLE staging slab.
///
/// Launches are queued, so a forward reaches its PLE layer on the host while the device is still
/// on earlier work, and a prefill's next chunk can reach it while the last chunk's copy is still
/// queued; the slab is rewritten only once the copy that read it has run. The event belongs to one
/// session's stream, so it never observes another session's copies.
struct StagingFence(StreamEvent);

// SAFETY: an event handle of the backend's context, owned by one session and recorded/waited on
// that session's stream by one thread at a time.
unsafe impl Send for StagingFence {}
// SAFETY: as above.
unsafe impl Sync for StagingFence {}

// SAFETY: a Kernel is a function handle loaded from the cubin; the handle is immutable and each
// session launches it on its own stream.
unsafe impl Send for KernelSet {}
// SAFETY: as above.
unsafe impl Sync for KernelSet {}

impl std::ops::Deref for KernelSet {
    type Target = Kernels;
    fn deref(&self) -> &Kernels {
        &self.0
    }
}

struct SequenceShared {
    cuda: CudaHandle,
    kernels: KernelSet,
    config: Arc<Qwen4ExpConfig>,
    weights: GpuWeights,
    /// The DFlash 2 drafter `SPARKFLASH_QWEN4_DFLASH2` names, which replaces the MTP head as the
    /// speculative drafter when it is loaded.
    dflash: Option<GpuDflash2>,
    prefill: PrefillPlan,
    /// The freed-session buffer pool. The mutex covers only allocation bookkeeping; it is not a
    /// run lock, and a session's forward never holds it.
    pooled: std::sync::Mutex<Option<ReusableBuffers>>,
    /// The most rows a speculative verify forward takes: the token being decoded and the drafts
    /// after it. Every row but the last has a snapshot region to roll back to.
    verify_rows: usize,
    /// The world this rank's partials are summed over, when the arm is one rank of several; its
    /// `config` is then the rank's share ([tp::rank_config]).
    tp: Option<tp::TensorParallel>,
}

impl Drop for SequenceShared {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.pooled.lock()
            && let Some(reusable) = guard.take()
        {
            reusable.free(&self.cuda);
        }
        if let Some(tp) = &self.tp {
            self.cuda.free_buffers(&[&tp.slots]);
        }
    }
}

impl DeviceSequence {
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), BackendError> {
        self.backend
            .cuda
            .launch_on(self.clock(), kernel, grid, block, args)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        // An opt-in trace: synchronize after every launch and name it, so a fault that the driver
        // only reports at the next boundary can still be attributed to one kernel.
        if self.trace {
            self.backend
                .cuda
                .synchronize_stream(self.clock())
                .map_err(|error| BackendError::new("qwen4_device", error))?;
            eprintln!("qwen4: {kernel:?} grid {grid:?} block {block:?}");
        }
        Ok(())
    }

    /// The stream this session's work goes on: its own, or the batch's while it is a member.
    fn clock(&self) -> &RunStream {
        self.batch_stream.as_ref().unwrap_or(&self.stream)
    }

    /// Report a device buffer that holds a non-finite value, under `SPARKFLASH_QWEN4_PROBE`.
    ///
    /// `SPARKFLASH_QWEN4_PROBE=dump` widens it: every `after_mlp` row and the final logits row are
    /// printed with their min, max, mean |x| and first four elements even when finite. That is what
    /// localises a `silu`-for-`sigmoid` gate or a transposed tensor, which produces plausible-looking
    /// magnitudes everywhere and only reveals itself as a flat logit row many layers later.
    ///
    /// A deployment diagnostic, not a serving feature. A wrong tensor or an unapplied scale shows
    /// up as a NaN, and the first thing a NaN reaches is the router, which turns it into an
    /// all-`-1` selection rather than an obvious error - so this walks the layer stack and names
    /// the first buffer that goes bad, which the 77.8 GiB upload makes far cheaper than bisecting
    /// by guess. Silent unless the variable is set, and capped so a broken run cannot flood the log.
    fn probe_finite(
        &self,
        layer: usize,
        label: &str,
        pointer: Deviceptr,
        elements: usize,
    ) -> Result<(), BackendError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static REPORTED: AtomicUsize = AtomicUsize::new(0);
        if sparkflash_model::active_profile()
            .diagnostics
            .probe
            .is_none()
        {
            return Ok(());
        }
        let mut bytes = vec![0u8; elements * 4];
        // Ordered behind this session's stream: the probe reads buffers its own launches filled,
        // and the default-stream copy would race them now that sessions own their streams.
        self.backend
            .cuda
            .copy_dtoh_on(self.clock(), &mut bytes, &pointer)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        if probe_hashing() {
            // One hash per row, at that row's position, so a speculative verify's rows can be
            // matched against the one-token steps they stand in for. A buffer that is not one
            // equal slice per row (a prefill's last-row logits) is hashed whole.
            let rows = self.probe_rows.max(1);
            let rows = if bytes.len().is_multiple_of(rows) {
                rows
            } else {
                1
            };
            for (row, slice) in bytes.chunks(bytes.len().div_ceil(rows).max(1)).enumerate() {
                eprintln!(
                    "qwen4 hash p{} layer{layer} {label}: {:016x}",
                    self.position + row,
                    fnv1a(slice)
                );
            }
            return Ok(());
        }
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let bad = values.iter().filter(|value| !value.is_finite()).count();
        if sparkflash_model::active_profile().diagnostics.probe
            == Some(sparkflash_model::profile::ProbeMode::Dump)
            && (label == "after_mlp" || label == "logits")
        {
            let head = &values[..values.len().min(4)];
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean_abs = values.iter().map(|v| v.abs()).sum::<f32>() / (values.len() as f32);
            eprintln!(
                "qwen4 dump p{} layer{layer} {label}: n={} min={min:.6} max={max:.6} mean_abs={mean_abs:.6} head={head:?}",
                self.position,
                values.len()
            );
        }
        if bad == 0 {
            return Ok(());
        }
        if REPORTED.fetch_add(1, Ordering::Relaxed) < 64 {
            let first = values
                .iter()
                .position(|value| !value.is_finite())
                .unwrap_or(0);
            let head = &values[..values.len().min(6)];
            eprintln!(
                "qwen4 probe p{} layer{layer} {label}: {bad} of {} not finite, first at {first}, \
                 head={head:?}",
                self.position,
                values.len()
            );
        }
        Ok(())
    }

    /// Under `SPARKFLASH_QWEN4_PROBE=hash`, the hash of each row's selected token positions (only
    /// the first `count` of each row are defined), for rows `base..base + rows` of the chunk.
    pub(super) fn probe_selected(
        &self,
        layer: usize,
        base: usize,
        rows: usize,
    ) -> Result<(), BackendError> {
        if !probe_hashing() {
            return Ok(());
        }
        let cuda = &self.backend.cuda;
        let mut counts = vec![0u8; rows * 4];
        cuda.copy_dtoh_on(self.clock(), &mut counts, &self.selected_count)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let mut selected = vec![0u8; rows * MAX_SELECTED_TOKENS * 4];
        cuda.copy_dtoh_on(self.clock(), &mut selected, &self.selected)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let mut defined = Vec::new();
        for row in 0..rows {
            let count =
                i32::from_le_bytes(counts[row * 4..row * 4 + 4].try_into().unwrap_or([0; 4]));
            let start = row * MAX_SELECTED_TOKENS * 4;
            let count = (count.max(0) as usize).min(MAX_SELECTED_TOKENS);
            defined.extend_from_slice(&counts[row * 4..row * 4 + 4]);
            defined.extend_from_slice(&selected[start..start + count * 4]);
        }
        eprintln!(
            "qwen4 hash p{} layer{layer} selected[{base}..{}]: {:016x}",
            self.position,
            base + rows,
            fnv1a(&defined)
        );
        Ok(())
    }

    /// Drain the stream and charge the elapsed time to `name`, under `SPARKFLASH_QWEN4_PROF`.
    fn prof_mark(&self, name: &'static str) -> Result<(), BackendError> {
        if prof::enabled() {
            self.backend
                .cuda
                .synchronize_stream(self.clock())
                .map_err(|error| BackendError::new("qwen4_device", error))?;
            prof::mark(name);
        }
        Ok(())
    }

    /// A projection over a chunk of GEMM_MIN_BATCH tokens or more.
    ///
    /// The tensor-core MMA takes every matrix whose shape fits its tile; the rest (a handful of
    /// rows wide) run the token GEMM in tiles of GEMM_TILE_TOKENS tokens, the whole tiles in one
    /// launch and the remainder in its own.
    fn gemm(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        batch: usize,
        epilogue: u32,
        scale: f32,
    ) -> Result<(), BackendError> {
        if self.dense_mma(matrix, input, output, batch, epilogue, scale)? {
            return Ok(());
        }
        let tiles = batch / GEMM_TILE_TOKENS;
        if tiles > 0 {
            self.gemm_tokens(
                matrix,
                input,
                output,
                GEMM_TILE_TOKENS,
                tiles,
                epilogue,
                scale,
            )?;
        }
        let base = tiles * GEMM_TILE_TOKENS;
        if base < batch {
            self.gemm_tokens(
                matrix,
                input.offset(base * matrix.columns * 4),
                output.offset(base * matrix.rows * 4),
                batch - base,
                1,
                epilogue,
                scale,
            )?;
        }
        Ok(())
    }

    /// The K split [Self::gemm_tokens] gives `matrix` over a run of `batch` tokens.
    pub(super) fn token_split(&self, matrix: &GpuMat, batch: usize) -> usize {
        // A block computes an 8 x 8 tile of (row, token) per thread, so a matrix with few rows
        // leaves the GPU almost entirely idle: the hyper-connection's 320-row down projection is
        // 640 threads, five blocks on forty-eight SMs, and measured 0.67 ms for a 6.5 MiB read
        // that should take a fortieth of that. Splitting the inner dimension over blockIdx.y and
        // reducing the slices costs one extra read of a few hundred kilobytes and fills the GPU.
        let split_for = if self.backend.prefill.decode {
            GEMM_TILE_TOKENS
        } else {
            batch
        };
        let price = (matrix.rows.div_ceil(64) * split_for.div_ceil(64)).max(1);
        let budget = (GEMM_PARTIAL_FLOATS / (split_for * matrix.rows).max(1)).max(1);
        (96 / price)
            .clamp(1, (matrix.columns / 32).max(1))
            .min(budget)
    }

    /// The shared-memory token GEMM over `runs` runs of `batch` tokens (at most
    /// GEMM_TILE_TOKENS) laid end to end, as few launches as the partials allow. A token's sums
    /// depend on the K split and nothing else in the launch. With the decode steps on the chunk
    /// path ([prefill::PrefillPlan::decode]) the split is a whole tile's whatever the run's width,
    /// so a token gets the bits it gets in any run; otherwise it is chosen from the run's own
    /// `batch`, the bits a launch of its run alone would give it.
    #[allow(clippy::too_many_arguments)]
    fn gemm_tokens(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        batch: usize,
        runs: usize,
        epilogue: u32,
        scale: f32,
    ) -> Result<(), BackendError> {
        let segments = self.token_split(matrix, batch);
        let partials = if segments > 1 {
            self.gemm_partials.0
        } else {
            0_u64
        };
        let count = batch * matrix.rows;
        // A few rows take q4p_gemm_rows, the same sums a thread an output, whose split tiles add
        // their slices themselves: one ticket a 128-output tile of a run.
        let rows_kernel = batch < GEMM_MIN_BATCH;
        // The runs whose slices fit the partials, and their tiles the tickets, at once. A run
        // alone always fits both: its split slices take `segments * count` partials, so at most
        // GEMM_PARTIAL_FLOATS / 256 128-output tiles.
        let per_launch = if segments > 1 {
            let fit = (GEMM_PARTIAL_FLOATS / (segments * count)).max(1);
            if rows_kernel {
                fit.min((GEMM_SPLIT_TICKETS / count.div_ceil(128)).max(1))
            } else {
                fit
            }
        } else {
            runs
        };
        let tickets = split_tickets(self.gemm_partials).0;
        let mut done = 0usize;
        while done < runs {
            let launched = per_launch.min(runs - done);
            let output = output.offset(done * count * 4);
            let (kernel, blocks) = if rows_kernel {
                (self.backend.kernels.prefill.gemm_rows, count.div_ceil(128))
            } else {
                (self.backend.kernels.gemm_bf16, matrix.rows.div_ceil(64))
            };
            let args = [
                matrix.pointer.0,
                input.offset(done * batch * matrix.columns * 4).0,
                output.0,
                matrix.rows as u64,
                matrix.columns as u64,
                batch as u64,
                epilogue as u64,
                u64::from(scale.to_bits()),
                segments as u64,
                partials,
                tickets,
            ];
            // q4e_gemm_bf16 takes no tickets.
            let arity = if rows_kernel { 11 } else { 10 };
            self.launch(
                kernel,
                (blocks as u32, segments as u32, launched as u32),
                (128, 1, 1),
                &args[..arity],
            )?;
            if segments > 1 && !rows_kernel {
                self.launch(
                    self.backend.kernels.gemm_reduce,
                    (count.div_ceil(256).max(1) as u32, launched as u32, 1),
                    (256, 1, 1),
                    &[
                        partials,
                        output.0,
                        count as u64,
                        segments as u64,
                        epilogue as u64,
                        u64::from(scale.to_bits()),
                    ],
                )?;
            }
            done += launched;
        }
        Ok(())
    }

    /// The register-tiled GEMV, once per group of eight tokens.
    ///
    /// A chunk below GEMM_MIN_BATCH would waste most of the GEMM's fixed 128-token tile; this
    /// walks the chunk in the GEMV's own tile instead, trading extra weight passes for a tile that
    /// matches the work.
    fn gemv_grouped(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        batch: usize,
    ) -> Result<(), BackendError> {
        const WARPS: usize = 8;
        let mut base = 0usize;
        while base < batch {
            let count = GEMV_MAX_BATCH.min(batch - base);
            self.launch(
                self.backend.kernels.gemv_bf16,
                (matrix.rows.div_ceil(WARPS) as u32, 1, 1),
                ((WARPS * 32) as u32, 1, 1),
                &[
                    matrix.pointer.0,
                    input.offset(base * matrix.columns * 4).0,
                    output.offset(base * matrix.rows * 4).0,
                    matrix.rows as u64,
                    matrix.columns as u64,
                    count as u64,
                ],
            )?;
            base += count;
        }
        Ok(())
    }

    fn gemv(
        &self,
        matrix: &GpuMat,
        input: Deviceptr,
        output: Deviceptr,
        batch: usize,
    ) -> Result<(), BackendError> {
        // A forward on the chunk path takes the chunk's GEMM at every row count, so a decode
        // step's row gets the bits the same row gets in a prompt chunk.
        if batch >= GEMM_MIN_BATCH || self.chunk_path(batch) {
            return self.gemm(matrix, input, output, batch, 0, 1.0);
        }
        if batch > GEMV_MAX_BATCH {
            return self.gemv_grouped(matrix, input, output, batch);
        }
        if matrix.columns >= TILED_GEMV_MIN_COLUMNS
            && tiled_operands(&[matrix.pointer, input], matrix.columns)
        {
            let tiles = matrix.rows.div_ceil(32);
            return self.launch(
                self.backend.kernels.gemv_tiled_bf16,
                (
                    tiles as u32,
                    tiled_gemv_splits(tiles, matrix.columns) as u32,
                    1,
                ),
                (256, 1, 1),
                &[
                    matrix.pointer.0,
                    input.0,
                    output.0,
                    matrix.rows as u64,
                    matrix.columns as u64,
                    batch as u64,
                    self.gemm_partials.0,
                    self.tiled_counters.0,
                ],
            );
        }
        // One warp per output row, so the grid is the row count over the warps in a block.
        const WARPS: usize = 8;
        self.launch(
            self.backend.kernels.gemv_bf16,
            (matrix.rows.div_ceil(WARPS) as u32, 1, 1),
            ((WARPS * 32) as u32, 1, 1),
            &[
                matrix.pointer.0,
                input.0,
                output.0,
                matrix.rows as u64,
                matrix.columns as u64,
                batch as u64,
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv4(
        &self,
        m0: &GpuMat,
        m1: &GpuMat,
        m2: &GpuMat,
        m3: &GpuMat,
        input: Deviceptr,
        y0: Deviceptr,
        y1: Deviceptr,
        y2: Deviceptr,
        y3: Deviceptr,
        batch: usize,
    ) -> Result<(), BackendError> {
        if batch > GEMV_MAX_BATCH || self.chunk_path(batch) {
            if batch >= GEMM_MIN_BATCH || self.chunk_path(batch) {
                self.gemm_group(
                    &[
                        Projection::plain(m0, y0),
                        Projection::plain(m1, y1),
                        Projection::plain(m2, y2),
                        Projection::plain(m3, y3),
                    ],
                    input,
                    batch,
                )?;
            } else {
                self.gemv_grouped(m0, input, y0, batch)?;
                self.gemv_grouped(m1, input, y1, batch)?;
                self.gemv_grouped(m2, input, y2, batch)?;
                self.gemv_grouped(m3, input, y3, batch)?;
            }
            return Ok(());
        }
        const WARPS: usize = 8;
        let total_rows = m0.rows + m1.rows + m2.rows + m3.rows;
        self.launch(
            self.backend.kernels.gemv4_bf16,
            (total_rows.div_ceil(WARPS) as u32, 1, 1),
            ((WARPS * 32) as u32, 1, 1),
            &[
                m0.pointer.0,
                m1.pointer.0,
                m2.pointer.0,
                m3.pointer.0,
                input.0,
                y0.0,
                y1.0,
                y2.0,
                y3.0,
                m0.rows as u64,
                m1.rows as u64,
                m2.rows as u64,
                m3.rows as u64,
                m0.columns as u64,
                batch as u64,
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        input: Deviceptr,
        weight: Deviceptr,
        output: Deviceptr,
        width: usize,
        group: usize,
        zero_centred: bool,
        weight_per_group: bool,
        batch: usize,
    ) -> Result<(), BackendError> {
        self.launch(
            self.backend.kernels.rmsnorm,
            (((batch * width) / group) as u32, 1, 1),
            (256, 1, 1),
            &[
                input.0,
                weight.0,
                output.0,
                width as u64,
                group as u64,
                (self.backend.config.rms_norm_eps as f32).to_bits() as u64,
                u64::from(zero_centred),
                u64::from(weight_per_group),
            ],
        )
    }

    fn hyper(
        &self,
        block: &GpuHyper,
        input: Deviceptr,
        inject: bool,
        batch: usize,
    ) -> Result<(Deviceptr, Option<Deviceptr>), BackendError> {
        self.hyper_rows(block, input, inject, 0, batch)
    }

    /// [Self::hyper] over rows `first..first + batch` of a chunk whose widened rows start at
    /// `input`: each row is read and its outputs written at its own row of every buffer, so a
    /// rank of a sequence-parallel forward leaves its share where the gather puts its peers'.
    fn hyper_rows(
        &self,
        block: &GpuHyper,
        input: Deviceptr,
        inject: bool,
        first: usize,
        batch: usize,
    ) -> Result<(Deviceptr, Option<Deviceptr>), BackendError> {
        let config = &self.backend.config;
        let width = config.hyper_width();
        const WARPS: usize = 8;
        let (inject_ptr, down_rows) = if inject {
            let weights = block
                .inject
                .as_ref()
                .ok_or_else(|| BackendError::new("qwen4_device", "missing block_inject weight"))?;
            (weights.pointer.0, config.hc_lowrank + config.hc_count)
        } else {
            (0u64, config.hc_lowrank)
        };
        // A prefill chunk is wide enough for the shared-memory GEMM, which reads the two 6.5 MiB
        // hyper matrices once per chunk. The grouped GEMV below re-reads them once per eight
        // tokens, sixteen times over, and measured 145 ms of a 128-token chunk against the few
        // milliseconds one pass costs - a quarter of the stage was the hyper streams' weight
        // traffic alone. The nonlinearities ride along as the GEMM tile is written.
        let normed = self.normed.offset(first * width * 4);
        let mixed = self.mixed.offset(first * config.hidden_size * 4);
        let injection = self.injection.offset(first * config.hc_count * 4);
        if batch >= GEMM_MIN_BATCH || self.chunk_path(batch) {
            let low = self.hyper_low.offset(first * config.hc_lowrank * 4);
            let raw = self.hyper_mix_raw.offset(first * width * 4);
            self.norm(
                input.offset(first * width * 4),
                block.norm,
                normed,
                width,
                config.hidden_size,
                true,
                false,
                batch,
            )?;
            let hc = config.hc_count as f32;
            self.prof_mark("hyper_norm")?;
            // The injection reads the same normed rows as the down projection: a few rows take it
            // beside that launch ([Self::gemm_group]).
            let mut projections = vec![Projection {
                matrix: &block.down,
                output: low,
                epilogue: 1,
                scale: hc,
            }];
            if inject {
                let weights = block.inject.as_ref().ok_or_else(|| {
                    BackendError::new("qwen4_device", "missing block_inject weight")
                })?;
                projections.push(Projection {
                    matrix: weights,
                    output: injection,
                    epilogue: 2,
                    scale: hc,
                });
            }
            self.gemm_group(&projections, normed, batch)?;
            self.prof_mark("hyper_low")?;
            self.gemm(&block.up, low, raw, batch, 0, 1.0)?;
            self.prof_mark("hyper_raw")?;
            self.launch(
                self.backend.kernels.hyper_mix,
                // The kernel walks the chunk as one flat range, so the token count belongs in the
                // block count, not in a second grid dimension: as `gridDim.y` every block ran the
                // whole tensor and the launch cost a hundred and twenty-eight times its work.
                (
                    (batch * config.hidden_size).div_ceil(256).max(1) as u32,
                    1,
                    1,
                ),
                (256, 1, 1),
                &[
                    raw.0,
                    normed.0,
                    mixed.0,
                    config.hidden_size as u64,
                    config.hc_count as u64,
                    batch as u64,
                ],
            )?;
            self.prof_mark("hyper_down")?;
            self.prof_mark("hyper_up")?;
            return Ok((mixed, inject.then_some(injection)));
        }
        // Below the GEMM's chunk width these passes keep one register accumulator per token, so
        // they are launched in groups of GEMV_MAX_BATCH tokens and each group is offset into the
        // chunk's buffers. The weights are small next to the dense stack, so the extra passes
        // cost little at this size - which is what made them invisible until a chunk went wide.
        //
        // The tiled passes take four streams, a warp's four rows; the shape alone decides, so a
        // verify's rows and the one-token step always go the same way.
        let tiled = config.hc_count == 4
            && tiled_operands(&[block.down.pointer, block.up.pointer], width)
            && tiled_operands(&[block.up.pointer], config.hc_lowrank)
            && block
                .inject
                .as_ref()
                .is_none_or(|weights| tiled_operands(&[weights.pointer], width));
        let mut base = first;
        while base < first + batch {
            let count = GEMV_MAX_BATCH.min(first + batch - base);
            let normed = self.normed.offset(base * width * 4);
            let low = self.hyper_low.offset(base * config.hc_lowrank * 4);
            let injection = self.injection.offset(base * config.hc_count * 4);
            let mixed = self.mixed.offset(base * config.hidden_size * 4);
            self.norm(
                input.offset(base * width * 4),
                block.norm,
                normed,
                width,
                config.hidden_size,
                true,
                false,
                count,
            )?;
            if tiled {
                let tiles = down_rows.div_ceil(32);
                self.launch(
                    self.backend.kernels.hyper_down_tiled,
                    (tiles as u32, tiled_gemv_splits(tiles, width) as u32, 1),
                    (256, 1, 1),
                    &[
                        normed.0,
                        block.down.pointer.0,
                        inject_ptr,
                        low.0,
                        injection.0,
                        config.hc_count as u64,
                        width as u64,
                        config.hc_lowrank as u64,
                        count as u64,
                        self.gemm_partials.0,
                        self.tiled_counters.0,
                    ],
                )?;
                self.launch(
                    self.backend.kernels.hyper_up_tiled,
                    (config.hidden_size.div_ceil(8) as u32, 1, 1),
                    (256, 1, 1),
                    &[
                        normed.0,
                        block.up.pointer.0,
                        low.0,
                        mixed.0,
                        config.hc_count as u64,
                        config.hidden_size as u64,
                        config.hc_lowrank as u64,
                        count as u64,
                    ],
                )?;
                base += count;
                continue;
            }
            self.launch(
                self.backend.kernels.hyper_down,
                (down_rows.div_ceil(WARPS) as u32, 1, 1),
                ((WARPS * 32) as u32, 1, 1),
                &[
                    normed.0,
                    block.down.pointer.0,
                    inject_ptr,
                    low.0,
                    injection.0,
                    config.hc_count as u64,
                    width as u64,
                    config.hc_lowrank as u64,
                    count as u64,
                ],
            )?;
            self.launch(
                self.backend.kernels.hyper_up,
                (config.hidden_size.div_ceil(WARPS) as u32, 1, 1),
                ((WARPS * 32) as u32, 1, 1),
                &[
                    normed.0,
                    block.up.pointer.0,
                    low.0,
                    mixed.0,
                    config.hc_count as u64,
                    config.hidden_size as u64,
                    config.hc_lowrank as u64,
                    count as u64,
                ],
            )?;
            base += count;
        }
        self.prof_mark("hyper_down")?;
        self.prof_mark("hyper_up")?;
        Ok((mixed, inject.then_some(injection)))
    }

    /// Assemble the MTP's widened rows: the previous forward's last row, then this forward's
    /// own stream. The module's entry `q` pairs `hidden_{q-1}` with `embedding(t_q)`, so every
    /// batch is one row out of step with the model's own stream.
    fn assemble_mtp_rows(&self, count: usize) -> Result<(), BackendError> {
        let width = self.backend.config.hyper_width();
        for row in 0..count {
            let source = if row == 0 {
                self.mtp_stash
            } else {
                self.hyper.offset((row - 1) * width * 4)
            };
            self.launch(
                self.backend.kernels.copy_flat,
                (width.div_ceil(256).max(1) as u32, 1, 1),
                (256, 1, 1),
                &[
                    self.mtp_rows.offset(row * width * 4).0,
                    source.0,
                    width as u64,
                ],
            )?;
        }
        Ok(())
    }

    /// Carry the drafted layer's own widened row into the MTP's stash, for the chained proposal.
    fn stash_mtp_hyper(&self) -> Result<(), BackendError> {
        let width = self.backend.config.hyper_width();
        self.launch(
            self.backend.kernels.copy_flat,
            (width.div_ceil(256).max(1) as u32, 1, 1),
            (256, 1, 1),
            &[self.mtp_stash.0, self.mtp_hyper.0, width as u64],
        )
    }

    /// Carry this forward's last widened row into the MTP's stash for the next batch.
    fn stash_last_hyper(&self, count: usize) -> Result<(), BackendError> {
        let width = self.backend.config.hyper_width();
        self.launch(
            self.backend.kernels.copy_flat,
            (width.div_ceil(256).max(1) as u32, 1, 1),
            (256, 1, 1),
            &[
                self.mtp_stash.0,
                self.hyper.offset((count - 1) * width * 4).0,
                width as u64,
            ],
        )
    }

    /// One pass of the MTP module over `tokens`, whose positions start at `position`.
    ///
    /// The module fuses the model's widened stream and the next token's embedding to one row
    /// (`fc_hidden` of the folded, normed stream plus `fc_embedding` of the normed embedding),
    /// widens it again, and runs the checkpoint's one sparse-attention + MoE layer on it. Its own
    /// mixer then folds the result back to one row, which is what the language head multiplies.
    ///
    /// Writes the module's key and value caches for those positions, and when `logits` is set
    /// leaves the last position's logits in `self.logits`: the module's proposal for the token
    /// after `position + tokens.len()`.
    fn mtp_pass(
        &mut self,
        tokens: &[u32],
        position: usize,
        logits: bool,
    ) -> Result<Option<Vec<f32>>, BackendError> {
        let config = Arc::clone(&self.backend.config);
        let kernels = &*self.backend.kernels;
        let weights = &self.backend.weights;
        let Some(mtp) = weights.mtp.as_ref() else {
            return Ok(None);
        };
        let batch = tokens.len();
        let width = config.hyper_width();
        let hidden = config.hidden_size;
        let flat = |count: usize| count.div_ceil(256).max(1) as u32;
        self.norm(
            self.mtp_rows,
            mtp.pre_norm_hidden,
            self.normed,
            width,
            width,
            true,
            false,
            batch,
        )?;
        self.launch(
            kernels.fold_streams,
            (flat(batch * hidden), 1, 1),
            (256, 1, 1),
            &[
                self.normed.0,
                self.mtp_folded.0,
                hidden as u64,
                config.hc_count as u64,
                batch as u64,
                self.mtp_fold_scale.to_bits() as u64,
            ],
        )?;
        for (t, &token) in tokens.iter().enumerate() {
            self.launch(
                kernels.embed,
                (flat(hidden), 1, 1),
                (256, 1, 1),
                &[
                    weights.embed.pointer.0,
                    self.mtp_embed.offset(t * hidden * 4).0,
                    u64::from(token),
                    hidden as u64,
                    hidden as u64,
                    0,
                ],
            )?;
        }
        self.norm(
            self.mtp_embed,
            mtp.pre_norm_embedding,
            self.mtp_embed,
            hidden,
            hidden,
            true,
            false,
            batch,
        )?;
        self.gemv(&mtp.fc_embedding, self.mtp_embed, self.mixed, batch)?;
        self.gemv(&mtp.fc_hidden, self.mtp_folded, self.mtp_embed, batch)?;
        self.launch(
            kernels.add,
            (flat(batch * hidden), 1, 1),
            (256, 1, 1),
            &[self.mixed.0, self.mtp_embed.0, (batch * hidden) as u64],
        )?;
        self.launch(
            kernels.widen_streams,
            (flat(width), batch as u32, 1),
            (256, 1, 1),
            &[
                self.mixed.0,
                self.mtp_hyper.0,
                hidden as u64,
                config.hc_count as u64,
            ],
        )?;
        self.prof_mark("mtp_fuse")?;
        let slot = config.num_hidden_layers;
        let (mixed, injection) = self.hyper(&mtp.attn_hyper, self.mtp_hyper, true, batch)?;
        let injection = injection.expect("the MTP attention hyper-connection injects");
        self.sparse_attention(&mtp.attn, slot, mixed, position, batch)?;
        self.inject_branch(self.mtp_hyper, injection, batch)?;
        self.prof_mark("mtp_attn")?;
        let (mixed, injection) = self.hyper(&mtp.mlp_hyper, self.mtp_hyper, true, batch)?;
        let injection = injection.expect("the MTP MLP hyper-connection injects");
        let chunk = self.chunk_path(batch);
        self.mlp(
            &mtp.moe,
            mixed,
            self.mtp_hyper,
            injection,
            slot,
            chunk,
            batch,
        )?;
        self.prof_mark("mtp_mlp")?;
        let last = self.mtp_hyper.offset((batch - 1) * width * 4);
        let (folded, _) = self.hyper(&mtp.mixer, last, false, 1)?;
        if !logits {
            return Ok(None);
        }
        self.gemv(&weights.lm_head, folded, self.logits, 1)?;
        self.prof_mark("mtp_head")?;
        self.read_logits(1).map(Some)
    }

    /// Run the whole model over `tokens`, starting at `self.position`.
    ///
    /// `logit_rows` is how many of the batch's trailing rows come back as logits (up to the
    /// backend's `verify_rows`), oldest first. A speculative step needs one per row of the chain
    /// it verifies: the first is the token the engine asked for, and each later one is the model's
    /// continuation of a draft.
    ///
    /// `snapshot_rows` is how many leading rows leave their convolution windows in a snapshot
    /// region (row `r` in region `r`), for a rejected chain to roll back to; 0 writes none. A
    /// forward with snapshots is a speculative verify: it leaves the GDN recurrent state as it
    /// found it, puts its last row's state in the spare, and keeps its rows' inputs for
    /// `commit_recurrent`, which the caller runs once it knows how many rows to keep.
    /// One layer's PLE block over `batch` rows of `tokens`.
    ///
    /// The n-gram table's rows for this layer, the key and query mixers, the gate and the causal
    /// convolution, added back into the widened residual `self.hyper`. Each row's lookup is that
    /// sequence's own n-gram history, so a batched step runs this once per row with `batch` 1.
    #[allow(clippy::too_many_arguments)]
    fn ple_step(
        &mut self,
        index: usize,
        layer: &GpuLayer,
        tokens: &[u32],
        batch: usize,
        share: Option<(usize, usize)>,
        ple_snapshot: u64,
        snapshot_rows: usize,
        region_stride: u64,
    ) -> Result<(), BackendError> {
        let Some(ple) = &layer.ple else {
            return Ok(());
        };
        // The layer's weights and kernels out of a handle of their own: the table lock and the
        // n-gram windows need the session mutably, and the launches below are on the session too.
        let backend = Arc::clone(&self.backend);
        let config = &backend.config;
        let kernels = &*backend.kernels;
        let cuda = &backend.cuda;
        let width = config.hyper_width();
        let hasher = self.ple_hashers[index]
            .as_mut()
            .ok_or_else(|| BackendError::new("qwen4_device", "the PLE layer has no hasher"))?;
        let rows = hasher
            .push(tokens, None)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let embed_dim = config.ple_embed_dim;
        // A sequence-parallel chunk reads only this rank's share of the table rows and gathers
        // its peers' embeddings: the same bytes decoded the same way, a quarter of the reads at
        // TP=4.
        let (first, count) = share.unwrap_or((0, batch));
        let row_bytes = embed_dim * 4;
        let mine = self.ple_embedding.offset(first * row_bytes);
        // The fence waits for the last copy out of the slab, not for the layers queued
        // since, so the table reads below overlap the device's work on the layers before.
        cuda.host_wait(&self.ple_staged.0)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        {
            let staging = &self.ple_host_staging;
            ple.table
                .lock()
                .map_err(|_| BackendError::new("qwen4_device", "the PLE table lock is poisoned"))?
                .embeddings(&rows[first..first + count], |t, embedding| {
                    staging.write_at(t * row_bytes, bytemuck::cast_slice::<f32, u8>(embedding))
                })
                .map_err(|error| BackendError::new("qwen4_device", error))?;
        }
        self.prof_mark("ple_host")?;
        cuda.copy_htod_async_on(
            self.clock(),
            &mine,
            &self.ple_host_staging,
            0,
            count * row_bytes,
        )
        .map_err(|error| BackendError::new("qwen4_device", error))?;
        cuda.record_event_on(self.clock(), &self.ple_staged.0)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        if share.is_some() {
            self.gather_rows(self.ple_embedding, mine, count * row_bytes)?;
            self.prof_mark("ple_gather")?;
        }
        self.probe_finite(
            index,
            "ple_embedding",
            self.ple_embedding,
            batch * embed_dim,
        )?;
        self.gemv(&ple.key_proj, self.ple_embedding, self.ple_key, batch)?;
        self.gemv(&ple.value_proj, self.ple_embedding, self.ple_value, batch)?;
        self.probe_finite(
            index,
            "ple_value",
            self.ple_value,
            batch * config.hidden_size,
        )?;
        self.norm(
            self.ple_key,
            ple.norm_key,
            self.ple_key,
            width,
            config.hidden_size,
            true,
            false,
            batch,
        )?;
        self.norm(
            self.hyper,
            ple.norm_query,
            self.ple_query,
            width,
            config.hidden_size,
            true,
            false,
            batch,
        )?;
        self.launch(
            kernels.ple_gate,
            (config.hc_count as u32, batch as u32, 1),
            (256, 1, 1),
            &[
                self.ple_key.0,
                self.ple_value.0,
                self.ple_query.0,
                ple.norm_conv.0,
                self.ple_gated.0,
                self.ple_normed.0,
                config.hc_count as u64,
                config.hidden_size as u64,
                (config.rms_norm_eps as f32).to_bits() as u64,
            ],
        )?;
        self.probe_finite(index, "ple_gated", self.ple_gated, batch * width)?;
        self.probe_finite(index, "ple_normed", self.ple_normed, batch * width)?;
        let state = self.ple_conv[index]
            .ok_or_else(|| BackendError::new("qwen4_device", "the PLE layer has no conv state"))?;
        self.launch(
            kernels.ple_conv,
            (width.div_ceil(256).max(1) as u32, 1, 1),
            (256, 1, 1),
            &[
                self.ple_normed.0,
                state.0,
                ple.conv.0,
                self.ple_conv_out.0,
                width as u64,
                config.ple_conv_kernel_size as u64,
                config.ngram_size as u64,
                batch as u64,
                ple_snapshot,
                snapshot_rows as u64,
                region_stride,
            ],
        )?;
        self.probe_finite(index, "ple_conv_out", self.ple_conv_out, batch * width)?;
        let add_blocks = (batch * width).div_ceil(256).max(1) as u32;
        self.launch(
            kernels.add,
            (add_blocks, 1, 1),
            (256, 1, 1),
            &[
                self.ple_gated.0,
                self.ple_conv_out.0,
                (batch * width) as u64,
            ],
        )?;
        self.launch(
            kernels.add,
            (add_blocks, 1, 1),
            (256, 1, 1),
            &[self.hyper.0, self.ple_gated.0, (batch * width) as u64],
        )?;
        self.probe_finite(index, "after_ple", self.hyper, batch * width)?;
        self.prof_mark("ple")?;
        Ok(())
    }

    /// A layer's attention block over `batch` rows of `mixed` from `position` on, its output in
    /// `self.branch`: a linear (Gated DeltaNet) block's projections, convolution, recurrence and
    /// output projection, or a sparse-attention block.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        index: usize,
        layer: &GpuLayer,
        mixed: Deviceptr,
        position: usize,
        batch: usize,
        chunk: bool,
        snapshot_rows: usize,
        conv_snapshot: u64,
        region_stride: u64,
    ) -> Result<(), BackendError> {
        match &layer.branch {
            GpuBranch::Linear { .. } => {
                // A verify's convolved rows, `b` and `a` go where the replay that commits them
                // reads them; any other forward uses the shared scratch.
                let (conv_out, b_out, a_out) = match (snapshot_rows > 0, self.gdn_replay[index]) {
                    (true, Some(inputs)) => self.replay_slices(inputs),
                    _ => (self.linear_conv, self.linear_b, self.linear_a),
                };
                self.linear_project(&layer.branch, mixed, b_out, a_out, batch)?;
                self.linear_state(
                    index,
                    &layer.branch,
                    conv_out,
                    b_out,
                    a_out,
                    batch,
                    chunk,
                    snapshot_rows,
                    conv_snapshot,
                    region_stride,
                )?;
                self.linear_output(index, &layer.branch, batch)
            }
            GpuBranch::Sparse { .. } => {
                self.sparse_attention(&layer.branch, index, mixed, position, batch)
            }
        }
    }

    fn forward_batch(
        &mut self,
        tokens: &[u32],
        logit_rows: usize,
        snapshot_rows: usize,
    ) -> Result<Option<Vec<f32>>, BackendError> {
        let batch = tokens.len();
        self.probe_rows = batch;
        debug_assert!((1..=self.backend.prefill.chunk).contains(&batch));
        debug_assert!(logit_rows <= self.backend.verify_rows && logit_rows <= batch);
        debug_assert!(snapshot_rows < self.backend.verify_rows && snapshot_rows < batch.max(1));
        // A wide chunk takes the tensor cores, and with the decode steps there every forward
        // does. A forward that snapshots a row is a speculative verify, which computes as the
        // decode steps it stands in for: on the chunk path exactly when they are.
        let chunk = self.chunk_path(batch);
        // The model's weights and kernels out of a handle of their own, so the layer loop can take
        // the session mutably for a layer's PLE block without giving up the weights it is reading.
        let backend = Arc::clone(&self.backend);
        let config = Arc::clone(&backend.config);
        let kernels = &*backend.kernels;
        let weights = &backend.weights;
        let width = config.hyper_width();
        // A sharded chunk splits its hyper-connections over the ranks from the layer after the
        // last PLE block on: a PLE block convolves each row with the rows before it, which a rank
        // holding only its own share would not have.
        self.own_rows = None;
        self.window = 0;
        let shares = self.sequence_parallel(batch, chunk);
        let parallel_from = weights
            .layers
            .iter()
            .rposition(|layer| layer.ple.is_some())
            .map_or(0, |last| last + 1);
        // From there on, a chunk whose halves split into whole pieces runs them overlapped.
        let overlap_from = if shares.is_some() && self.overlaps(batch) {
            parallel_from
        } else {
            weights.layers.len()
        };
        if prof::enabled() {
            prof::begin();
        }
        // The whole chunk in one launch, grid row t widening tokens[t], on the session's stream
        // like every other launch of the forward: the default stream the loader uses is not
        // ordered against it, so an embed left there raced the layers that read its rows. The
        // ids are read in place from the slab, which is rewritten only once the last embed ran.
        let cuda = &backend.cuda;
        cuda.host_wait(&self.tokens_staged.0)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        self.token_staging
            .write_at(0, bytemuck::cast_slice::<u32, u8>(tokens))
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        self.launch(
            kernels.embed,
            (width.div_ceil(256).max(1) as u32, batch as u32, 1),
            (256, 1, 1),
            &[
                weights.embed.pointer.0,
                self.hyper.0,
                0,
                config.hidden_size as u64,
                width as u64,
                self.token_staging.device().0,
            ],
        )?;
        cuda.record_event_on(self.clock(), &self.tokens_staged.0)
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        self.probe_finite(0, "embed", self.hyper, batch * width)?;
        self.prof_mark("embed")?;
        let position = self.position;
        for (index, layer) in weights.layers.iter().enumerate().take(overlap_from) {
            // A speculative verify saves the convolution windows after each of its rows but the
            // last, so the rows past the ones the model agreed with can be given back; a plain
            // forward passes no snapshot at all. The kernels write row r's windows `r` regions
            // further on.
            let (conv_slot, ple_slot) = self.snapshot_slots(index);
            let snapshot_ptr = |slot: Option<usize>| -> u64 {
                match (snapshot_rows > 0, slot) {
                    (true, Some(offset)) => self.snapshot.offset(offset * 4).0,
                    _ => 0,
                }
            };
            let conv_snapshot = snapshot_ptr(conv_slot);
            let ple_snapshot = snapshot_ptr(ple_slot);
            let region_stride = self.snapshot_stride as u64;
            self.own_rows = shares.filter(|_| index >= parallel_from);
            self.ple_step(
                index,
                layer,
                tokens,
                batch,
                shares,
                ple_snapshot,
                snapshot_rows,
                region_stride,
            )?;
            let (mixed, _) = self.branch_input(&layer.attn_hyper, batch)?;
            self.prof_mark("hyper_attn")?;
            self.attend(
                index,
                layer,
                mixed,
                position,
                batch,
                chunk,
                snapshot_rows,
                conv_snapshot,
                region_stride,
            )?;
            self.inject_branch(self.hyper, self.injection, batch)?;
            self.probe_finite(index, "after_attn", self.hyper, batch * width)?;
            self.prof_mark("attn_inject")?;
            let (mixed, injection) = self.branch_input(&layer.mlp_hyper, batch)?;
            self.probe_finite(index, "mixed", mixed, batch * config.hidden_size)?;
            self.prof_mark("hyper_mlp")?;
            self.mlp(
                &layer.moe, mixed, self.hyper, injection, index, chunk, batch,
            )?;
            self.probe_finite(index, "after_mlp", self.hyper, batch * width)?;
            self.capture_features(index, batch)?;
            self.prof_mark("inject_mlp")?;
        }
        if overlap_from < weights.layers.len() {
            self.overlapped_layers(&weights.layers[overlap_from..], overlap_from, chunk, batch)?;
        } else if self.own_rows.is_some() {
            self.gather_features()?;
        }
        let last_hyper = match logit_rows {
            0 => None,
            rows => Some(self.head_rows(batch, rows)?),
        };
        self.own_rows = None;
        self.window = 0;
        let Some(last_hyper) = last_hyper else {
            prof::report(self.position + batch - 1, batch);
            self.position += batch;
            return Ok(None);
        };
        // The head sees only the logit rows; in a decode or a verify they are every row.
        self.probe_rows = logit_rows;
        let (mixed, _) = self.hyper(&weights.mixer, last_hyper, false, logit_rows)?;
        self.gemv(&weights.lm_head, mixed, self.logits, logit_rows)?;
        self.probe_finite(48, "logits", self.logits, config.vocab_size * logit_rows)?;
        self.prof_mark("lm_head")?;
        let logits = self.read_logits(logit_rows)?;
        self.prof_mark("logits_dtoh")?;
        prof::report(self.position + batch - 1, batch);
        self.position += batch;
        Ok(Some(logits))
    }

    /// The first `rows` rows of the logits buffer, copied straight into the vector they are
    /// returned in.
    fn read_logits(&self, rows: usize) -> Result<Vec<f32>, BackendError> {
        let mut logits = vec![0.0f32; self.backend.config.vocab_size * rows];
        self.backend
            .cuda
            .copy_dtoh_on(
                self.clock(),
                bytemuck::cast_slice_mut(&mut logits),
                &self.logits,
            )
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        Ok(logits)
    }

    /// Where layer `index`'s convolution windows sit in the step snapshot, in elements: the GDN
    /// history, then the PLE frames.
    ///
    /// The snapshot is written by the state kernels themselves, which are the only ones that see
    /// the state between a pair's two rows; this lays the same regions out for the roll-back.
    fn snapshot_slots(&self, index: usize) -> (Option<usize>, Option<usize>) {
        let config = &self.backend.config;
        let history = config.linear_conv_kernel_dim.saturating_sub(1).max(1);
        let conv_elems = history * config.linear_conv_width();
        let mut at = 0usize;
        let mut conv = None;
        for layer in 0..config.layer_types.len() {
            if self.gdn_conv[layer].is_none() {
                continue;
            }
            if layer == index {
                conv = Some(at);
            }
            at += conv_elems;
        }
        let ple_elems = config.ple_conv_kernel_size.saturating_sub(1).max(1)
            * config.ngram_size
            * config.hyper_width();
        let mut ple = None;
        for layer in 0..config.layer_types.len() {
            if self.ple_conv[layer].is_none() {
                continue;
            }
            if layer == index {
                ple = Some(at);
            }
            at += ple_elems;
        }
        (conv, ple)
    }

    /// Roll the recurrent state back to the snapshot the last pair's first row left behind.
    /// Advances every n-gram window past one token, exactly as a forward pass does.
    ///
    /// A pair's first row pushes its token before the row runs, so rolling a rejected pair back
    /// and replaying that token has to leave the windows where the row found them - otherwise the
    /// PLE rows the replay reads are one token stale.
    fn push_hashers(&mut self, token: u32) -> Result<(), BackendError> {
        for hasher in self.ple_hashers.iter_mut().flatten() {
            hasher
                .push(&[token], None)
                .map_err(|error| BackendError::new("qwen4_device", error))?;
        }
        Ok(())
    }

    /// A GDN layer's verify inputs, as `gdn_replay` lays them out: the convolved rows, `b`, `a`.
    fn replay_slices(&self, inputs: Deviceptr) -> (Deviceptr, Deviceptr, Deviceptr) {
        let config = &self.backend.config;
        let rows = self.backend.verify_rows;
        let b = inputs.offset(rows * config.linear_conv_width() * 4);
        let a = b.offset(rows * config.linear_num_value_heads * 4);
        (inputs, b, a)
    }

    /// Keep the first `rows` of a verify's `verified` rows in every GDN layer's recurrent state.
    ///
    /// The verify left the live state as it found it and its last row's state in the spare; a
    /// step that keeps fewer rows replays them from the live state into the spare. The spare then
    /// becomes the live state, and the state from before the verify stays in the spare for a late
    /// rollback (`rollback_recurrent`).
    fn commit_recurrent(&mut self, rows: usize, verified: usize) -> Result<(), BackendError> {
        self.prof_mark("host_answer")?;
        for index in 0..self.gdn_recurrent.len() {
            let (Some(live), Some(spare)) = (self.gdn_recurrent[index], self.gdn_spare[index])
            else {
                continue;
            };
            if rows < verified {
                self.replay_recurrent(index, rows, live, spare)?;
            }
            self.gdn_recurrent[index] = Some(spare);
            self.gdn_spare[index] = Some(live);
        }
        self.prof_mark("gdn_commit")
    }

    /// Rebuild the live recurrent state as the last verify's first `rows` rows left it, from the
    /// state before that verify, which the commit kept in the spare.
    fn rollback_recurrent(&self, rows: usize) -> Result<(), BackendError> {
        for index in 0..self.gdn_recurrent.len() {
            let (Some(live), Some(spare)) = (self.gdn_recurrent[index], self.gdn_spare[index])
            else {
                continue;
            };
            self.replay_recurrent(index, rows, spare, live)?;
        }
        Ok(())
    }

    /// Run layer `index`'s recurrence over the first `rows` rows of its last verify, from the
    /// state in `from`, leaving the state in `to`. It is the verify's own kernels on the verify's
    /// own inputs - the chunk path's scan when the verify took it - so the state is bit for bit
    /// the one those rows left behind.
    fn replay_recurrent(
        &self,
        index: usize,
        rows: usize,
        from: Deviceptr,
        to: Deviceptr,
    ) -> Result<(), BackendError> {
        let config = &self.backend.config;
        let kernels = &*self.backend.kernels;
        let GpuBranch::Linear { a_log, dt_bias, .. } = &self.backend.weights.layers[index].branch
        else {
            return Err(BackendError::new(
                "qwen4_device",
                format!("layer {index} has a recurrent state but no linear branch"),
            ));
        };
        let inputs = self.gdn_replay[index]
            .ok_or_else(|| BackendError::new("qwen4_device", "a GDN layer has no verify inputs"))?;
        let (conv, b, a) = self.replay_slices(inputs);
        if self.chunk_path(rows)
            && self.gdn_recurrence(*a_log, *dt_bias, [conv, b, a], (from, to), rows)?
        {
            return Ok(());
        }
        self.launch(
            kernels.gdn_recur,
            (config.linear_num_value_heads as u32, 1, 1),
            (128, 1, 1),
            &[
                conv.0,
                0,
                b.0,
                a.0,
                a_log.0,
                dt_bias.0,
                0,
                from.0,
                to.0,
                0,
                config.linear_key_width() as u64,
                config.linear_num_value_heads as u64,
                config.linear_key_head_dim as u64,
                config.linear_value_head_dim as u64,
                config.value_groups() as u64,
                (config.rms_norm_eps as f32).to_bits() as u64,
                u64::from(config.output_gate_is_sigmoid()),
                rows as u64,
                u64::from(kernels.gdn_recur.shared_bytes()),
                // No channel decay: Gated DeltaNet decays a head by one factor.
                0,
            ],
        )
    }

    /// Copy one region of the convolution windows back over the live buffers.
    ///
    /// Region `r` is the windows after a chain's row `r`; a rejected draft restores whichever
    /// prefix the model actually agreed with. The recurrent state is replayed instead
    /// (`replay_recurrent`).
    fn restore_states(&self, which: usize) -> Result<(), BackendError> {
        if !self.spec_enabled {
            return Ok(());
        }
        debug_assert!(which + 1 < self.backend.verify_rows);
        let region = self.snapshot.offset(which * self.snapshot_stride * 4);
        let config = &self.backend.config;
        let kernels = &*self.backend.kernels;
        let history = config.linear_conv_kernel_dim.saturating_sub(1).max(1);
        let conv_elems = history * config.linear_conv_width();
        let mut at = 0usize;
        for conv in self.gdn_conv.iter().flatten() {
            let snapshot = region.offset(at * 4);
            self.launch(
                kernels.copy_flat,
                (conv_elems.div_ceil(256).max(1) as u32, 1, 1),
                (256, 1, 1),
                &[conv.0, snapshot.0, conv_elems as u64],
            )?;
            at += conv_elems;
        }
        // The PLE convolution reads a window of recent frames, stored per layer the same way.
        let ple_elems = config.ple_conv_kernel_size.saturating_sub(1).max(1)
            * config.ngram_size
            * config.hyper_width();
        for index in 0..self.ple_conv.len() {
            let Some(state) = self.ple_conv[index] else {
                continue;
            };
            let snapshot = region.offset(at * 4);
            self.launch(
                kernels.copy_flat,
                (ple_elems.div_ceil(256).max(1) as u32, 1, 1),
                (256, 1, 1),
                &[state.0, snapshot.0, ple_elems as u64],
            )?;
            at += ple_elems;
        }
        Ok(())
    }

    /// Run the MTP for `token` at `position` and keep its proposals for the positions after it.
    ///
    /// The first proposal verifies the token the model will sample next. The second is the
    /// chained one: it reads the drafted layer's own widened row at `position`, which is the
    /// same trick the reference applies when one MTP layer is stepped twice, so a pair's second
    /// row costs nothing extra to verify.
    fn draft_chain(&mut self, token: u32, position: usize) -> Result<(), BackendError> {
        self.spec_drafts.clear();
        if !self.spec_enabled || self.backend.weights.mtp.is_none() {
            return Ok(());
        }
        self.assemble_mtp_rows(1)?;
        let Some(logits) = self.mtp_pass(&[token], position, true)? else {
            return Ok(());
        };
        let Some(draft) = argmax(&logits) else {
            return Ok(());
        };
        self.spec_drafts.push(draft);
        if self.spec.max_depth() < 2 {
            return Ok(());
        }
        self.stash_mtp_hyper()?;
        self.assemble_mtp_rows(1)?;
        if let Some(logits) = self.mtp_pass(&[draft], position + 1, true)?
            && let Some(second) = argmax(&logits)
        {
            self.spec_drafts.push(second);
        }
        Ok(())
    }
}
/// The token a logits row picks, ignoring ties the way the sampler's greedy path does.
fn argmax(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(token, _)| token as u32)
}

impl DeviceSequence {
    fn allocate(shared: Arc<SequenceShared>, context_limit: usize) -> Result<Self, BackendError> {
        let config = Arc::clone(&shared.config);
        let cuda = &shared.cuda;
        cuda.make_current()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let stream = cuda
            .run_stream()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        let ple_staged = StagingFence(
            cuda.stream_event()
                .map_err(|error| BackendError::new("qwen4_device", error))?,
        );
        let tokens_staged = StagingFence(
            cuda.stream_event()
                .map_err(|error| BackendError::new("qwen4_device", error))?,
        );
        let hidden = config.hidden_size;
        let width = config.hyper_width();
        // Every per-row buffer holds one prefill chunk. The indexer's score grid is the exception:
        // it is a row per token over the whole context, so the sparse block walks the chunk in
        // ATTENTION_CHUNK-row pieces and only that many rows of it, and of the selection it
        // feeds, are kept.
        let rows = shared.prefill.chunk;
        let attention_rows = rows.min(ATTENTION_CHUNK);
        let zero_gpu = |pointer: Deviceptr, elements: usize| -> Result<(), BackendError> {
            let blocks = elements.div_ceil(256).max(1) as u32;
            cuda.launch_on(
                &stream,
                shared.kernels.zero,
                (blocks, 1, 1),
                (256, 1, 1),
                &[pointer.0, elements as u64],
            )
            .map_err(|error| BackendError::new("qwen4_device", error))
        };
        let mut ple_hashers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            if let Some(ple_layer_index) =
                config.ple_layer_ids.iter().position(|id| *id == index + 1)
            {
                ple_hashers.push(Some(
                    crate::ngram::PleHasher::new(&config, ple_layer_index)
                        .map_err(|error| BackendError::new("qwen4_device", error))?,
                ));
            } else {
                ple_hashers.push(None);
            }
        }
        // A region holds every GDN layer's conv history, then every PLE layer's frames, in the
        // order snapshot_slots lays them out.
        let linear_layers = config
            .layer_types
            .iter()
            .filter(|kind| matches!(kind, LayerKind::Linear))
            .count();
        let ple_layers = (0..config.layer_types.len())
            .filter(|index| config.ple_layer_ids.contains(&(index + 1)))
            .count();
        let snapshot_stride = linear_layers
            * config.linear_conv_kernel_dim.saturating_sub(1).max(1)
            * config.linear_conv_width()
            + ple_layers
                * config.ple_conv_kernel_size.saturating_sub(1).max(1)
                * config.ngram_size
                * width;
        // A GDN layer's verify rows, laid out as gdn_replay names them.
        let replay_stride =
            shared.verify_rows * (config.linear_conv_width() + 2 * config.linear_num_value_heads);
        let reused = shared.pooled.lock().ok().and_then(|mut guard| guard.take());
        let buffers = if let Some(buf) = reused {
            buf
        } else {
            let mut total = 0usize;
            let f32_bytes = |elements: usize| elements * 4;
            macro_rules! alloc {
                ($elements:expr) => {
                    cuda.alloc(f32_bytes($elements), &mut total)
                        .map_err(|error| BackendError::new("qwen4_device", error))
                };
            }
            macro_rules! alloc_bf16 {
                ($elements:expr) => {
                    cuda.alloc(($elements) * 2, &mut total)
                        .map_err(|error| BackendError::new("qwen4_device", error))
                };
            }
            let mut keys = Vec::with_capacity(config.num_hidden_layers);
            let mut values = Vec::with_capacity(config.num_hidden_layers);
            let mut index_keys = Vec::with_capacity(config.num_hidden_layers);
            let mut gdn_conv = Vec::with_capacity(config.num_hidden_layers);
            let mut gdn_recurrent = Vec::with_capacity(config.num_hidden_layers);
            let mut gdn_spare = Vec::with_capacity(config.num_hidden_layers);
            let mut ple_conv = Vec::with_capacity(config.num_hidden_layers);
            for (index, kind) in config.layer_types.iter().enumerate() {
                match kind {
                    LayerKind::Sparse => {
                        keys.push(Some(alloc_bf16!(context_limit * config.kv_proj_width())?));
                        values.push(Some(alloc_bf16!(context_limit * config.kv_proj_width())?));
                        index_keys
                            .push(Some(alloc_bf16!(context_limit * config.indexer_head_dim)?));
                        gdn_conv.push(None);
                        gdn_recurrent.push(None);
                        gdn_spare.push(None);
                    }
                    LayerKind::Linear => {
                        let history = config.linear_conv_kernel_dim.saturating_sub(1);
                        let conv = alloc!(history.max(1) * config.linear_conv_width())?;
                        let rec_elems = config.linear_num_value_heads
                            * config.linear_key_head_dim
                            * config.linear_value_head_dim;
                        let recurrent = alloc!(rec_elems)?;
                        keys.push(None);
                        values.push(None);
                        index_keys.push(None);
                        gdn_conv.push(Some(conv));
                        gdn_recurrent.push(Some(recurrent));
                        gdn_spare.push(if shared.verify_rows > 1 {
                            Some(alloc!(rec_elems)?)
                        } else {
                            None
                        });
                    }
                }
                if config.ple_layer_ids.contains(&(index + 1)) {
                    let frames =
                        (config.ple_conv_kernel_size.saturating_sub(1)) * config.ngram_size;
                    ple_conv.push(Some(alloc!(frames.max(1) * width)?));
                } else {
                    ple_conv.push(None);
                }
            }
            if config.mtp_num_hidden_layers > 0 {
                // The drafted layer has its own attention, so its keys, values and indexer keys
                // are one more entry in the same lists the per-layer caches live in.
                keys.push(Some(alloc_bf16!(context_limit * config.kv_proj_width())?));
                values.push(Some(alloc_bf16!(context_limit * config.kv_proj_width())?));
                index_keys.push(Some(alloc_bf16!(context_limit * config.indexer_head_dim)?));
                gdn_conv.push(None);
                gdn_recurrent.push(None);
                gdn_spare.push(None);
                ple_conv.push(None);
            }
            let snapshot_regions = shared.verify_rows.saturating_sub(1).max(1);
            let scratch = [
                alloc!(rows * width)?,
                alloc!(rows * width)?,
                alloc!(rows * hidden)?,
                alloc!(rows * config.hc_count)?,
                alloc!(rows * config.attn_out_width())?,
                alloc!(rows * hidden)?,
                alloc!(rows * config.q_proj_width())?,
                alloc!(rows * config.attn_out_width())?,
                alloc!(rows * config.attn_out_width())?,
                alloc!(rows * config.kv_proj_width())?,
                alloc!(rows * config.kv_proj_width())?,
                alloc!(
                    rows * (config.indexer_n_heads + config.indexer_kv_heads)
                        * config.indexer_head_dim
                )?,
                alloc!(rows * config.indexer_n_heads * config.indexer_head_dim)?,
                alloc!(attention_rows * MAX_SELECTED_TOKENS)?,
                alloc!(attention_rows)?,
                alloc!(
                    attention_rows
                        * (context_limit.div_ceil(config.indexer_compress_ratio.max(1)) + 1)
                )?,
                alloc!(rows * config.linear_conv_width())?,
                alloc!(rows * config.linear_conv_width())?,
                alloc!(rows * config.linear_value_width())?,
                alloc!(rows * config.linear_num_value_heads)?,
                alloc!(rows * config.linear_num_value_heads)?,
                alloc!(rows * config.linear_value_width())?,
                alloc!(shared.verify_rows * config.vocab_size)?,
                alloc!(rows * config.num_experts)?,
                alloc!(rows * config.num_experts_per_tok)?,
                alloc!(rows * config.num_experts_per_tok * hidden)?,
                alloc!(rows * config.num_experts_per_tok * config.moe_intermediate_size)?,
                alloc!(rows * config.hc_lowrank)?,
                alloc!(rows)?,
                alloc!(rows * config.shared_expert_intermediate_size)?,
                alloc!(rows * config.shared_expert_intermediate_size)?,
                alloc!(rows * hidden)?,
                alloc!(rows * config.ple_embed_dim)?,
                alloc!(rows * width)?,
                alloc!(rows * width)?,
                alloc!(rows * hidden)?,
                alloc!(rows * width)?,
                alloc!(rows * width)?,
                alloc!(rows * width)?,
                alloc!(rows * config.num_experts_per_tok)?,
                alloc!(2 * rows * config.num_experts_per_tok)?,
                alloc!(rows * config.hc_count * hidden)?,
                alloc!(GEMM_PARTIAL_REGION)?,
                alloc!(rows * width)?,
                alloc!(rows * hidden)?,
                alloc!(rows * hidden)?,
                alloc!(rows * width)?,
                alloc!(width)?,
                alloc!(snapshot_regions * snapshot_stride)?,
                alloc!(TILED_GEMV_BLOCKS)?,
                alloc!(linear_layers.max(1) * replay_stride)?,
                alloc!(
                    (context_limit.div_ceil(config.indexer_compress_ratio.max(1)) + 1)
                        * config.indexer_head_dim
                )?,
            ];
            let ple_host_staging = cuda
                .alloc_host_mapped((rows * config.ple_embed_dim * 4).max(4), &mut total)
                .map_err(|error| BackendError::new("qwen4_device", error))?;
            let token_staging = cuda
                .alloc_host_mapped(rows * 4, &mut total)
                .map_err(|error| BackendError::new("qwen4_device", error))?;
            let prefill = PrefillScratch::allocate(cuda, &config, rows, &mut total)
                .map_err(|error| BackendError::new("qwen4_device", error))?;
            let draft = shared
                .dflash
                .as_ref()
                .map(|draft| {
                    DraftScratch::allocate(draft, rows, |floats| cuda.alloc(floats * 4, &mut total))
                })
                .transpose()
                .map_err(|error| BackendError::new("qwen4_dflash2", error))?;
            ReusableBuffers {
                keys,
                values,
                index_keys,
                gdn_conv,
                gdn_recurrent,
                gdn_spare,
                ple_conv,
                scratch,
                prefill,
                draft,
                ple_host_staging,
                token_staging,
            }
        };
        let history = config.linear_conv_kernel_dim.saturating_sub(1).max(1);
        let conv_elems = history * config.linear_conv_width();
        let rec_elems = config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
        let ple_frames = (config.ple_conv_kernel_size.saturating_sub(1)) * config.ngram_size;
        let ple_elems = ple_frames.max(1) * width;
        for &ptr in buffers.gdn_conv.iter().flatten() {
            zero_gpu(ptr, conv_elems)?;
        }
        for &ptr in buffers.gdn_recurrent.iter().flatten() {
            zero_gpu(ptr, rec_elems)?;
        }
        for &ptr in buffers.ple_conv.iter().flatten() {
            zero_gpu(ptr, ple_elems)?;
        }
        let s = buffers.scratch;
        zero_gpu(s[49], TILED_GEMV_BLOCKS)?;
        zero_gpu(split_tickets(s[42]), GEMM_SPLIT_TICKETS)?;
        let mut linear_at = 0usize;
        let gdn_replay = buffers
            .gdn_conv
            .iter()
            .map(|conv| {
                conv.map(|_| {
                    linear_at += 1;
                    s[50].offset((linear_at - 1) * replay_stride * 4)
                })
            })
            .collect();
        let score_stride = context_limit.div_ceil(config.indexer_compress_ratio.max(1)) + 1;
        Ok(Self {
            backend: Arc::clone(&shared),
            stream,
            batch_stream: None,
            ple_staged,
            tokens_staged,
            position: 0,
            consumed: Vec::new(),
            prefill_cached: 0,
            score_stride,
            keys: buffers.keys,
            values: buffers.values,
            index_keys: buffers.index_keys,
            gdn_conv: buffers.gdn_conv,
            gdn_recurrent: buffers.gdn_recurrent,
            gdn_spare: buffers.gdn_spare,
            gdn_replay,
            ple_conv: buffers.ple_conv,
            hyper: s[0],
            normed: s[1],
            mixed: s[2],
            injection: s[3],
            attention_out: s[4],
            branch: s[5],
            qg: s[6],
            query: s[7],
            gate: s[8],
            kv_row: s[9],
            value_row: s[10],
            index_row: s[11],
            index_query: s[12],
            selected: s[13],
            selected_count: s[14],
            qsa_block_scores: s[15],
            qsa_pooled_keys: s[51],
            linear_qkv: s[16],
            linear_conv: s[17],
            linear_z: s[18],
            linear_b: s[19],
            linear_a: s[20],
            linear_out: s[21],
            logits: s[22],
            probabilities: s[23],
            expert_indices: s[24],
            expert_perm: s[39],
            expert_run: s[40],
            gemm_partials: s[42],
            expert_accumulator: s[25],
            expert_activated: s[26],
            hyper_low: s[27],
            hyper_mix_raw: s[41],
            expert_gate: s[28],
            shared_hidden: s[29],
            shared_up: s[30],
            shared_out: s[31],
            ple_embedding: s[32],
            ple_key: s[33],
            ple_query: s[34],
            ple_value: s[35],
            ple_gated: s[36],
            ple_normed: s[37],
            ple_conv_out: s[38],
            mtp_rows: s[43],
            mtp_folded: s[44],
            mtp_embed: s[45],
            mtp_hyper: s[46],
            mtp_stash: s[47],
            snapshot: s[48],
            snapshot_stride,
            tiled_counters: s[49],
            replay_inputs: s[50],
            prefill: buffers.prefill,
            draft: buffers.draft,
            context_limit,
            spec_queue: Vec::new(),
            spec_drafts: Vec::new(),
            spec_confidence: Vec::new(),
            // The chain is as wide as the drafter that fills it: the MTP pair is two rows by
            // construction (`draft_chain` stops there), and a DFlash 2 block proposes
            // `block_size - 1` drafts, which is one verify of the block's own width. The
            // profile's depth is the ceiling on top of that; the earlier `clamp(1, 2)` was a
            // TP=1 measurement of the MTP pair, applied to a drafter that offers seven.
            spec: sparkflash_model::Speculator::new(
                {
                    let chain = shared
                        .dflash
                        .as_ref()
                        .map_or(2, |draft| draft.config.block_size.saturating_sub(1));
                    sparkflash_model::active_profile()
                        .speculation
                        .depth
                        .unwrap_or(chain)
                        .clamp(1, chain)
                },
                shared.dflash.as_ref().map(|draft| {
                    DepthPolicy::from_profile(draft.config.block_size - 1, DFLASH2_STEP_COST)
                }),
            ),
            spec_enabled: (config.mtp_num_hidden_layers > 0 || shared.dflash.is_some())
                && sparkflash_model::active_profile()
                    .speculation
                    .enabled
                    .unwrap_or(true),
            // A mean over the hyper-connection streams unless the profile names a scale; `sum`
            // is the spelling `SPARKFLASH_QWEN4_MTP_FOLD` has always documented for one, and it
            // arrives here as the number one (crates/sparkflash-model/src/profile.rs).
            mtp_fold_scale: sparkflash_model::active_profile()
                .speculation
                .mtp_fold_scale
                .unwrap_or(1.0 / config.hc_count as f32),
            own_rows: None,
            window: 0,
            overlap: None,
            spec_trace: sparkflash_model::active_profile()
                .diagnostics
                .spec_trace
                .unwrap_or(false),
            probe_rows: 1,
            ple_host_staging: buffers.ple_host_staging,
            token_staging: buffers.token_staging,
            ple_hashers,
            sampling: Sampling::default(),
            sampler: crate::sampling::Sampler::new(),
            eos: config.eos_token_ids.clone(),
            trace: sparkflash_model::active_profile()
                .diagnostics
                .trace
                .unwrap_or(false),
        })
    }
    /// The per-row activation scratch, in a canonical order.
    ///
    /// A batch of sequences points each member's scratch at that member's row of the batch's
    /// combined buffers for the duration of a step, then restores what it saved here.
    pub(super) fn scratch_pointers(&self) -> [Deviceptr; SCRATCH_SLOTS] {
        [
            self.hyper,
            self.normed,
            self.mixed,
            self.injection,
            self.attention_out,
            self.branch,
            self.qg,
            self.query,
            self.gate,
            self.kv_row,
            self.value_row,
            self.index_row,
            self.index_query,
            self.selected,
            self.selected_count,
            self.qsa_block_scores,
            self.linear_qkv,
            self.linear_conv,
            self.linear_z,
            self.linear_b,
            self.linear_a,
            self.linear_out,
            self.logits,
            self.probabilities,
            self.expert_indices,
            self.expert_accumulator,
            self.expert_activated,
            self.hyper_low,
            self.expert_gate,
            self.shared_hidden,
            self.shared_up,
            self.shared_out,
            self.ple_embedding,
            self.ple_key,
            self.ple_query,
            self.ple_value,
            self.ple_gated,
            self.ple_normed,
            self.ple_conv_out,
            self.expert_perm,
            self.expert_run,
            self.hyper_mix_raw,
            self.gemm_partials,
            self.mtp_rows,
            self.mtp_folded,
            self.mtp_embed,
            self.mtp_hyper,
            self.mtp_stash,
        ]
    }

    /// Point every per-row scratch buffer at the pointers given, in [Self::scratch_pointers] order.
    pub(super) fn set_scratch_pointers(&mut self, p: [Deviceptr; SCRATCH_SLOTS]) {
        self.hyper = p[0];
        self.normed = p[1];
        self.mixed = p[2];
        self.injection = p[3];
        self.attention_out = p[4];
        self.branch = p[5];
        self.qg = p[6];
        self.query = p[7];
        self.gate = p[8];
        self.kv_row = p[9];
        self.value_row = p[10];
        self.index_row = p[11];
        self.index_query = p[12];
        self.selected = p[13];
        self.selected_count = p[14];
        self.qsa_block_scores = p[15];
        self.linear_qkv = p[16];
        self.linear_conv = p[17];
        self.linear_z = p[18];
        self.linear_b = p[19];
        self.linear_a = p[20];
        self.linear_out = p[21];
        self.logits = p[22];
        self.probabilities = p[23];
        self.expert_indices = p[24];
        self.expert_accumulator = p[25];
        self.expert_activated = p[26];
        self.hyper_low = p[27];
        self.expert_gate = p[28];
        self.shared_hidden = p[29];
        self.shared_up = p[30];
        self.shared_out = p[31];
        self.ple_embedding = p[32];
        self.ple_key = p[33];
        self.ple_query = p[34];
        self.ple_value = p[35];
        self.ple_gated = p[36];
        self.ple_normed = p[37];
        self.ple_conv_out = p[38];
        self.expert_perm = p[39];
        self.expert_run = p[40];
        self.hyper_mix_raw = p[41];
        self.gemm_partials = p[PARTIALS_SLOT];
        self.mtp_rows = p[43];
        self.mtp_folded = p[44];
        self.mtp_embed = p[45];
        self.mtp_hyper = p[46];
        self.mtp_stash = p[47];
    }

    /// Floats one row of each buffer in [Self::scratch_pointers] needs.
    ///
    /// The caller multiplies by four for bytes. The strides are the ones the kernels are handed:
    /// a row of `hyper` is one hyper-connection width, a row of `mixed` one hidden width, and so on.
    pub(super) fn scratch_strides(
        config: &Qwen4ExpConfig,
        score_stride: usize,
    ) -> [usize; SCRATCH_SLOTS] {
        let hidden = config.hidden_size;
        let width = config.hyper_width();
        let attn = config.attn_out_width();
        let kv = config.kv_proj_width();
        let idx_row = (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        let idx_q = config.indexer_n_heads * config.indexer_head_dim;
        let topk = config.num_experts_per_tok;
        let heads = config.linear_num_value_heads;
        let value = config.linear_value_width();
        [
            width,
            width,
            hidden,
            config.hc_count,
            attn,
            hidden,
            config.q_proj_width(),
            attn,
            attn,
            kv,
            kv,
            idx_row,
            idx_q,
            MAX_SELECTED_TOKENS,
            1,
            score_stride,
            config.linear_conv_width(),
            config.linear_conv_width(),
            value,
            heads,
            heads,
            value,
            config.vocab_size,
            config.num_experts,
            topk,
            topk * hidden,
            topk * config.moe_intermediate_size,
            config.hc_lowrank,
            1,
            config.shared_expert_intermediate_size,
            config.shared_expert_intermediate_size,
            hidden,
            config.ple_embed_dim,
            width,
            width,
            hidden,
            width,
            width,
            width,
            topk,
            2 * topk,
            config.hc_count * hidden,
            GEMM_PARTIAL_REGION,
            width,
            hidden,
            hidden,
            width,
            width,
        ]
    }
    /// Choose the next token, ending the sequence at an end-of-generation id.
    ///
    /// The released configuration names two stop ids (end-of-text and <|im_end|>), and the serving
    /// loop treats `None` as "the model finished". Returning a stop id as a token instead would
    /// append it to the completion and then keep generating past the end of the answer.
    fn next(&mut self, logits: &[f32]) -> Option<u32> {
        let token = self.sampler.choose(logits, &self.sampling);
        if self.eos.contains(&token) {
            None
        } else {
            Some(token)
        }
    }
    /// Fold a forward's logits rows back into one step: the token the model chose at the first row,
    /// the drafts it agreed with, and the state those rows left behind.
    ///
    /// `rows` are the tokens the forward consumed, this step's own token first, and `position` is
    /// where the first of them sits: the forward has already advanced the session past all of them.
    /// A one-row step is the same call with an empty `drafts` - nothing is agreed with, nothing is
    /// given back - which is what lets a batched step run every member through one path.
    ///
    /// # Errors
    /// A driver failure while the rejected rows' state is given back.
    fn resolve_verify(
        &mut self,
        rows: &[u32],
        logits: &[f32],
        position: usize,
        drafts: &[u32],
        confidence: &[f32],
        hashers: &std::sync::Arc<Vec<Option<crate::ngram::PleHasher>>>,
    ) -> Result<StepOutput, BackendError> {
        let vocab = self.backend.config.vocab_size;
        // The model's choice at each row, in order, up to the first that disagrees with the draft
        // after it: every later row's prefix holds a token the model did not choose. An agreed
        // draft is in the sampler's history before the row after it is chosen, and no row past
        // the disagreement draws, so a penalty or a seeded draw sees exactly what the one-token
        // steps this verify stands in for would have seen.
        let history = self.sampler.history().len();
        let mut answers: Vec<Option<u32>> = Vec::with_capacity(rows.len());
        let mut agreed = 0;
        loop {
            let answer = self.next(&logits[agreed * vocab..(agreed + 1) * vocab]);
            answers.push(answer);
            if agreed == drafts.len() || answer != Some(drafts[agreed]) {
                break;
            }
            self.sampler.record(drafts[agreed]);
            agreed += 1;
        }
        self.sampler.truncate(history);
        self.spec.observe(confidence, drafts.len(), agreed + 1);
        {
            if self.spec_trace && !drafts.is_empty() {
                // A draft's rank in the model's own row is the quality signal: a proposal that
                // is always tens of places away points at the MTP's arithmetic, not at its luck.
                for (index, draft) in drafts.iter().enumerate() {
                    let row = &logits[index * vocab..(index + 1) * vocab];
                    let rank = row
                        .iter()
                        .filter(|value| **value > row[*draft as usize])
                        .count();
                    eprintln!(
                        "qwen4: draft {draft} p {:.3} answer {:?} rank {rank} at {} {}",
                        confidence.get(index).copied().unwrap_or(f32::NAN),
                        answers.get(index).copied().flatten(),
                        position + index,
                        if index < agreed {
                            "accepted"
                        } else {
                            "rejected"
                        }
                    );
                }
            }
        }
        // Everything the model agreed with stays; the rest of the chain gives its state back so
        // the positions it consumed can be written again by the steps that follow. The verify
        // left the recurrent state untouched, so the kept rows are committed to it whatever the
        // count. The count is the shared rule's ([sparkflash_model::verdict]) and the loop above
        // must read the same chain: it stops at the first row that disagreed for the same reason.
        let picks: Vec<u32> = answers
            .iter()
            .map(|answer| answer.unwrap_or(u32::MAX))
            .collect();
        let kept = sparkflash_model::verdict(rows, &picks);
        debug_assert_eq!(kept, agreed + 1, "the verify and the shared rule disagree");
        // A verify leaves the live recurrent state as it found it and writes its last row's into
        // the spare, so the kept rows are committed by swapping the two. A plain one-row step
        // wrote the live state itself and must not swap: there is nothing to keep and nothing to
        // give back.
        if rows.len() > 1 {
            self.commit_recurrent(kept, rows.len())?;
        }
        if kept < rows.len() {
            self.restore_states(kept - 1)?;
            // The forward pushed every row's token into the n-gram windows; the ones the model
            // did not agree with have to come back off, or the PLE rows read one token ahead.
            self.ple_hashers = (**hashers).clone();
            for row in &rows[..kept] {
                self.push_hashers(*row)?;
            }
        }
        self.position = position + kept;
        for index in 1..=agreed {
            self.spec_queue.push(SpecPending {
                token: answers[index - 1].expect("an agreed draft has an answer"),
                logits: logits[index * vocab..(index + 1) * vocab].to_vec(),
                next_token: answers[index],
                position: position + index,
                region: index - 1,
                hashers: std::sync::Arc::clone(hashers),
                replay: rows[..index].to_vec(),
            });
        }
        if self.drafting() {
            // The drafter's rings take every position the model kept, and the next block is
            // anchored on the answer after the last of them.
            self.dflash_commit(0, kept, position)?;
            if let Some(token) = answers[kept - 1] {
                let base = self.position;
                self.dflash_propose(token, base)?;
            }
        } else if self.spec_enabled
            && let Some(token) = answers[kept - 1]
        {
            // One row of the MTP feeding the next chain: the last row the model agreed with,
            // which is the position the next step starts from.
            self.stash_last_hyper(kept)?;
            self.draft_chain(token, self.position)?;
        }
        let answer = answers[0];
        Ok(StepOutput {
            next_token: answer,
            logits: logits[..vocab].to_vec(),
            draft: Vec::new(),
        })
    }
}

impl ModelBackend for Qwen4ExpDeviceBackend {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> Capabilities {
        // The text arm renders the released tool catalogue, the `<think>` continuation and the
        // reasoning-effort instruction. The vision tower and MTP are planned but not loaded, so
        // only those text channels are claimed.
        Capabilities::ALL
    }

    fn open_session(&self) -> Result<Box<dyn ModelSession>, BackendError> {
        if let Some(fused) = &self.fused {
            return fused
                .open()
                .map(|session| Box::new(session) as Box<dyn ModelSession>);
        }
        DeviceSequence::allocate(Arc::clone(&self.shared), self.context_limit)
            .map(|session| Box::new(session) as Box<dyn ModelSession>)
    }
}

impl ModelSession for DeviceSequence {
    fn speculation(&self) -> Speculation {
        self.spec.stats()
    }

    /// The tokens this sequence has forwarded, while its caches end exactly where they end.
    ///
    /// A verified chain the adapter has not handed back is already in the caches, so the state is
    /// ahead of the tokens the caller knows: that sequence reports nothing and the next request
    /// opens a fresh one. The same holds for a position the tokens do not explain.
    fn reusable_prefix(&self) -> Option<&[u32]> {
        (self.spec_queue.is_empty() && self.position == self.consumed.len())
            .then_some(self.consumed.as_slice())
    }

    fn cached_tokens(&self) -> usize {
        self.prefill_cached
    }

    fn set_sampling(&mut self, sampling: Sampling) {
        // The penalties, temperature, top-k and top-p run on the host over the logits row this
        // step already copied back, so non-greedy requests are served rather than approximated.
        self.sampling = sampling;
        self.sampler.reset();
        // The adapter calls this once per request, and a kept sequence carries the last one's
        // counts: a request reports its own speculation, not the session's lifetime.
        self.spec.reset();
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<StepOutput, BackendError> {
        if tokens.is_empty() {
            return Err(BackendError::new(
                "qwen4_device",
                "an empty prompt has no logits",
            ));
        }
        self.backend
            .cuda
            .make_current()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        // A sequence the serving adapter kept may already hold a prefix of this prompt, in which
        // case only the tokens past it are forwarded. Its sampler history is the whole prompt
        // either way: the penalties count every token the request reads.
        let reused = self
            .consumed
            .iter()
            .zip(tokens)
            .take_while(|(held, given)| held == given)
            .count();
        if reused != self.consumed.len() || reused >= tokens.len() {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "a prefill must extend the {} token(s) this sequence holds by at least one; \
                     this one does not",
                    self.consumed.len()
                ),
            ));
        }
        self.prefill_cached = reused;
        self.sampler.record_prompt(tokens);
        self.consumed.extend_from_slice(&tokens[reused..]);
        // A proposal drafted past the previous end anchors on a token the new suffix replaces; it
        // is re-drafted from the new end below, like a fresh prompt's.
        self.spec_drafts.clear();
        self.spec_confidence.clear();
        let tokens = &tokens[reused..];
        // Each chunk reads its own PLE rows in one pass (ple_step). A whole-prompt pass here would
        // read them all twice: past 1,024 tokens the prompt names more rows than the table's cache
        // keeps, so the chunks would find most of them evicted again.
        let forwards = self.backend.prefill.forwards(tokens.len());
        let last_idx = forwards.len() - 1;
        // The first draft block sits just past the prompt and its anchor row reaches
        // `window_left` positions back; nothing before that is ever read from the rings.
        let visible = self.backend.dflash.as_ref().map_or(0, |draft| {
            (self.position + tokens.len()).saturating_sub(draft.config.window_left)
        });
        let mut logits = Vec::new();
        for (idx, rows) in forwards.into_iter().enumerate() {
            let chunk = &tokens[rows];
            let is_last = idx == last_idx;
            if let Some(last_logits) = self.forward_batch(chunk, usize::from(is_last), 0)? {
                logits = last_logits;
            }
            let start = self.position - chunk.len();
            if self.drafting() {
                // The drafter's context path takes the chunk DENSE_ROWS rows at a time, as its
                // projections split when that was the chunk.
                for at in (0..chunk.len()).step_by(DENSE_ROWS) {
                    let rows = DENSE_ROWS.min(chunk.len() - at);
                    let first = visible.saturating_sub(start + at).min(rows);
                    self.dflash_commit(at + first, rows - first, start + at + first)?;
                }
                continue;
            }
            // The MTP reads the prompt one row behind the model, so each chunk feeds it the rows
            // assembled from the previous chunk's last state and this one's. Its own key and value
            // caches have to cover the prompt too, or its first draft would attend to nothing.
            if self.spec_enabled {
                self.assemble_mtp_rows(chunk.len())?;
                self.mtp_pass(chunk, start, false)?;
                self.stash_last_hyper(chunk.len())?;
            }
        }
        let next_token = self.next(&logits);
        if let Some(token) = next_token {
            let position = self.position;
            if self.drafting() {
                self.dflash_propose(token, position)?;
            } else if self.spec_enabled {
                // The first generated token's own embedding is the last row the MTP has not seen.
                self.draft_chain(token, position)?;
            }
        }
        Ok(StepOutput {
            next_token,
            logits,
            draft: Vec::new(),
        })
    }
    fn decode(&mut self, token: u32) -> Result<StepOutput, BackendError> {
        self.backend
            .cuda
            .make_current()
            .map_err(|error| BackendError::new("qwen4_device", error))?;
        self.sampler.record(token);
        self.consumed.push(token);
        // A verified chain already fed the model this token, and its logits here are the ones the
        // last step read back; only the choice of the following token is left to make.
        if let Some(front) = self.spec_queue.first() {
            if front.token == token {
                let pending = self.spec_queue.remove(0);
                return Ok(StepOutput {
                    next_token: pending.next_token,
                    logits: pending.logits,
                    draft: Vec::new(),
                });
            }
            // The caller fed something else, so every position past this one is a token the model
            // never chose: hand them all back and resume from where this one belongs.
            let pending = self
                .spec_queue
                .drain(..)
                .next()
                .expect("the queue was not empty");
            self.restore_states(pending.region)?;
            self.rollback_recurrent(pending.region + 1)?;
            self.ple_hashers = (*pending.hashers).clone();
            for row in &pending.replay {
                self.push_hashers(*row)?;
            }
            self.position = pending.position;
            self.spec_drafts.clear();
            self.spec_confidence.clear();
        }
        // Offer the drafts as the chain's later tokens. A step the policy gave no drafts is the
        // same call with one row - the verify path agrees with nothing and gives nothing back -
        // so a batch can step every one of its members through this one path.
        let position = self.position;
        let drafts = std::mem::take(&mut self.spec_drafts);
        let confidence = std::mem::take(&mut self.spec_confidence);
        let mut rows = Vec::with_capacity(drafts.len() + 1);
        rows.push(token);
        rows.extend_from_slice(&drafts);
        let hashers = std::sync::Arc::new(self.ple_hashers.clone());
        let logits = self
            .forward_batch(&rows, rows.len(), rows.len() - 1)?
            .ok_or_else(|| BackendError::new("qwen4_device", "a step has no logits"))?;
        self.resolve_verify(&rows, &logits, position, &drafts, &confidence, &hashers)
    }
}

/// Load the cubin and exercise two kernels on synthetic buffers.
///
/// This is the first device gate: it proves the cubin loads, the context opens and a launch's
/// result reads back. It is not a model run and claims no parity.
///
/// # Errors
/// A build without the cubin, a driver failure, or a read-back that disagrees with the host.
pub fn device_self_test() -> Result<(), BackendError> {
    if embedded_cubin().is_empty() {
        return Err(BackendError::new(
            "qwen4_device",
            "this build has no Qwen4-Exp cubin; rebuild with nvcc to run the device self-test",
        ));
    }
    let cuda = Cuda::load()
        .map_err(|error| BackendError::new("qwen4_device", format!("CUDA: {error}")))?;
    // The self-test has no config; the recurrence streams its state from global memory.
    let kernels = Kernels::load(&cuda, 0)
        .map_err(|error| BackendError::new("qwen4_device", format!("kernel load: {error}")))?;
    let mut total = 0usize;
    let width = 8usize;
    let input: Vec<f32> = (0..width).map(|index| index as f32 * 0.25 + 0.5).collect();
    let mean_square: f32 = input.iter().map(|value| value * value).sum::<f32>() / width as f32;
    let scale = 1.0 / (mean_square + 1e-6).sqrt();
    let expected: Vec<f32> = input.iter().map(|value| value * scale).collect();
    let input_bytes: Vec<u8> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
    let weight_bytes = vec![0u8; width * 2];
    let device_input = cuda
        .alloc(input_bytes.len(), &mut total)
        .map_err(|error| BackendError::new("qwen4_device", error))?;
    let device_weight = cuda
        .alloc(weight_bytes.len(), &mut total)
        .map_err(|error| BackendError::new("qwen4_device", error))?;
    let device_output = cuda
        .alloc(input_bytes.len(), &mut total)
        .map_err(|error| BackendError::new("qwen4_device", error))?;
    cuda.copy_htod(&device_input, &input_bytes)
        .map_err(|error| BackendError::new("qwen4_device", error))?;
    cuda.copy_htod(&device_weight, &weight_bytes)
        .map_err(|error| BackendError::new("qwen4_device", error))?;
    cuda.launch(
        kernels.rmsnorm,
        (1, 1, 1),
        (256, 1, 1),
        &[
            device_input.0,
            device_weight.0,
            device_output.0,
            width as u64,
            width as u64,
            (1e-6f32).to_bits() as u64,
            1,
            1,
        ],
    )
    .map_err(|error| BackendError::new("qwen4_device", error))?;
    let mut bytes = vec![0u8; width * 4];
    cuda.copy_dtoh(&mut bytes, &device_output)
        .map_err(|error| BackendError::new("qwen4_device", error))?;
    for (index, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if (value - expected[index]).abs() > 1e-5 {
            return Err(BackendError::new(
                "qwen4_device",
                format!(
                    "rmsnorm mismatch at {index}: {value} vs {}",
                    expected[index]
                ),
            ));
        }
    }
    let _ = (device_input, device_weight, device_output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_open_config_reads_the_merged_generation_stop_set() {
        let directory = std::env::temp_dir().join(format!(
            "sparkflash-qwen-device-stop-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&directory).expect("a temp directory");
        std::fs::write(
            directory.join("config.json"),
            include_str!("../config/qwen38_flash_next_nvfp4.hf.json"),
        )
        .expect("config.json");
        std::fs::write(
            directory.join("generation_config.json"),
            r#"{"eos_token_id": [248099]}"#,
        )
        .expect("generation_config.json");
        let config = open_config(&directory).expect("the checkpoint configuration");
        assert_eq!(
            config.eos_token_ids.first(),
            Some(&248_044),
            "the config's <|endoftext|> still comes first"
        );
        assert!(
            config.eos_token_ids.contains(&248_099),
            "the generation-only stop id reaches the open path: {:?}",
            config.eos_token_ids
        );
        assert_eq!(
            config.architecture.tokens.eos_token_ids, config.eos_token_ids,
            "the device backend and the codec serve one stop set"
        );
        let _ = std::fs::remove_dir_all(&directory);
    }
}
