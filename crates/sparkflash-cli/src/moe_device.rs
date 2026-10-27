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

//! The c4b device routed-MoE backend.
//!
//! The router, the four rank gate/up projections, the weighted SwiGLU epilogue and the four
//! serial combine launches run on the device; the host stages only the distinct experts the router
//! selected (the m17a-proven path), and the engine's host shared-expert seam adds the shared expert
//! afterwards. This module is only compiled with `--features cuda`.

#![allow(clippy::items_after_test_module)]

use std::path::Path;
use std::sync::Arc;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::dtype::DType;
use sparkflash_engine::forward::moe::{
    MoeDeviceBackend, MoeScratch, add_shared_expert, routed_forward,
};
use sparkflash_engine::linalg::EngineError;
use sparkflash_engine::quant::quantize_activation_fp8;
use sparkflash_engine::weights::WeightSource;
use sparkflash_kernels::cert_lm_head_launch::{self, CertLmHeadParams};
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use sparkflash_kernels::lm_head_reduce_launch::{
    self as head_reduce, LmHeadReduceParams, STATUS_WINNER_CERTIFIED,
};
use sparkflash_kernels::lm_head_ref;
use sparkflash_kernels::m4_quant::bf16_bits;
use sparkflash_kernels::mxfp4_moe_launch::{self, Mxfp4MoeParams};
use sparkflash_kernels::mxfp4_ref;
use sparkflash_kernels::mxfp8_dense_launch::{self, Mxfp8DenseParams};
use sparkflash_kernels::mxfp8_ref;
use sparkflash_kernels::rmsnorm_launch::{self, RmsNormLaunchParams};
use sparkflash_kernels::rmsnorm_ref;
use sparkflash_kernels::routed_gate_up_launch::{self, GateUpLaunchParams};
use sparkflash_kernels::routed_gate_up_ref as gate_up_ref;
use sparkflash_kernels::router_launch::{self, RouterLaunchParams};
use sparkflash_kernels::{router_ref, swiglu_ref};

const ROUTER: &str = "moe_router_sm121";
const GATE_UP: &str = "routed_gate_up_sm121";
const WEIGHTED: &str = "swiglu_mxfp8_quant_weighted_sm121";
const COMBINE: &str = "b12x_mxfp4_moe_sm121";
const DENSE: &str = "b12x_mxfp8_dense_sm121";
const SWIGLU: &str = "swiglu_mxfp8_quant_sm121";
const HEAD: &str = "cert_lm_head_sm121";
const HEAD_REDUCE: &str = "lm_head_reduce_sm121";
const RMSNORM: &str = "rmsnorm_sm121";
const RANKS: usize = 4;

/// The routed kernels a layer launches, by kind. The plan is one device router, then for each of
/// the four serial ranks a gate GEMM, an up GEMM, the weighted SwiGLU quantiser and the combine
/// add. The plan is a constant here so the counter in the routed path can be checked against what
/// actually launched: a launch left out of the plan, or a readback turned into a launch, mismatches.
const ROUTED_ROUTER_LAUNCHES: usize = 1;
const ROUTED_GATE_UP_LAUNCHES: usize = RANKS * 2;
const ROUTED_WEIGHTED_LAUNCHES: usize = RANKS;
const ROUTED_COMBINE_LAUNCHES: usize = RANKS;
/// Routed launches one layer performs: the router plus four rank kernels per rank.
const ROUTED_LAUNCHES_PER_LAYER: usize = ROUTED_ROUTER_LAUNCHES
    + ROUTED_GATE_UP_LAUNCHES
    + ROUTED_WEIGHTED_LAUNCHES
    + ROUTED_COMBINE_LAUNCHES;
const HEAD_VOCAB_PER_RANK: usize = 32_320;
const SWIGLU_LIMIT_BITS: u64 = 10.0f32.to_bits() as u64;
const PADDED: usize = 640;
const QUANT_BLOCK: usize = 32;

/// The shared expert's two E8M0 scale grids, as (w1/w3, w2): one block scale per 32x32 tile.
fn shared_scale_shapes(inter: usize, dim: usize) -> ([usize; 2], [usize; 2]) {
    (
        [inter / QUANT_BLOCK, dim / QUANT_BLOCK],
        [dim / QUANT_BLOCK, inter / QUANT_BLOCK],
    )
}

type SharedScaleGrids = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Reads the three shared-expert E8M0 scale grids and pins the two literal 32x32 shapes: the
/// w1/w3 gate/up grid is [inter/32, dim/32], and the w2 down grid is [dim/32, inter/32]. The
/// production shapes for the released model are therefore [72, 160] and [160, 72]. A free function
/// (rather than three inline reads) lets a unit test swap one grid and require an error.
pub(crate) fn read_shared_scale_grids(
    source: &dyn WeightSource,
    base: &str,
    inter: usize,
    dim: usize,
) -> Result<SharedScaleGrids, EngineError> {
    let (up_scale_shape, down_scale_shape) = shared_scale_shapes(inter, dim);
    let s1 = read_spec(
        source,
        &format!("{base}.w1.scale"),
        DType::F8E8M0,
        &up_scale_shape,
    )?;
    let s3 = read_spec(
        source,
        &format!("{base}.w3.scale"),
        DType::F8E8M0,
        &up_scale_shape,
    )?;
    let s2 = read_spec(
        source,
        &format!("{base}.w2.scale"),
        DType::F8E8M0,
        &down_scale_shape,
    )?;
    Ok((s1, s3, s2))
}

/// Counters since construction.
#[derive(Debug, Default, Clone, Copy)]
pub struct MoeDeviceStats {
    /// Total routed launches, 17 per layer.
    pub launches: usize,
    /// Host microseconds reading, slicing and uploading.
    pub staging_us: u64,
    /// Host microseconds inside launches and readbacks.
    pub device_us: u64,
    /// Cumulative uploaded routed-expert bytes.
    pub staged_bytes: u64,
    /// Cumulative activation/readback bytes.
    pub host_bytes: u64,
    /// Layers that ran the device block.
    pub layers: usize,
    /// Layer-0 max abs difference between the device routed contribution and the host routed
    /// reference computed on the same input; NaN when layer 0 has not run.
    pub layer0_host_max_abs: f32,
    /// Layer-0 relative difference against the host routed reference (max abs / max |host|).
    pub layer0_host_rel: f32,
    /// Layer-0 max |host routed contribution|, the denominator of `layer0_host_rel`.
    pub layer0_host_magnitude: f32,
    /// Layer-0 mismatch count between the device bf16 combine chain and the host bf16 twin on the
    /// SAME device codes/scales; `usize::MAX` until layer 0 has run.
    pub layer0_bf16_twin_mismatches: usize,
    /// Layer-0 max abs difference between the device output and that host bf16 twin.
    pub layer0_bf16_twin_max_abs: f32,
    /// Shared-expert launches (four per layer in c4c).
    pub shared_launches: usize,
    /// Host microseconds staging, launching and reading the shared expert.
    pub shared_us: u64,
    /// Host microseconds staging the shared expert operands.
    pub shared_staging_us: u64,
    /// Host microseconds inside the shared-expert launches and readback.
    pub shared_device_us: u64,
    /// Cumulative uploaded shared-expert bytes.
    pub shared_bytes: u64,
    /// Layer-0 shared-expert bf16 mismatches against the same-device-byte host twin.
    pub layer0_shared_bf16_mismatches: usize,
    /// Layer-0 shared-expert f32 max abs difference from the host reference (calibration).
    pub layer0_shared_host_max_abs: f32,
    /// Layer-0 shared-expert relative difference against the host reference.
    pub layer0_shared_host_rel: f32,
    /// Layer-0 max |host shared-expert contribution|.
    pub layer0_shared_host_magnitude: f32,
    /// Routed-only launches (17 per layer), excluding the shared expert.
    pub routed_launches: usize,
    /// Routed-only host staging microseconds, excluding the shared expert.
    pub routed_staging_us: u64,
    /// Routed-only device microseconds, excluding the shared expert.
    pub routed_device_us: u64,
    /// Routed-only host bytes, excluding the shared expert.
    pub routed_host_bytes: u64,
    /// LM-head launches (four rank shards in c4c).
    pub head_launches: usize,
    /// Shared-expert-only host bytes, excluding the routed experts.
    pub shared_host_bytes: u64,
    /// Total certified (pruned) head entries over the four rank shards.
    pub head_certified: usize,
    /// Certified count per rank shard, in rank order.
    pub head_certified_by_rank: [usize; RANKS],
    /// Host microseconds staging the head weight and converting the hidden row.
    pub head_staging_us: u64,
    /// Host microseconds in the four head launches and their readbacks.
    pub head_device_us: u64,
    /// Uploaded head-weight bytes.
    pub head_staged_bytes: u64,
    /// True when the winning entry is certified, so the token is not claimable.
    pub head_winner_certified: bool,
    /// Whether the device reduce produced the claimed token instead of the host assembly fallback.
    pub head_reduce_used: bool,
    /// The reduce's token readback (diagnostic on the fallback path).
    pub head_reduce_token: u32,
    /// The reduce's argmax value readback.
    pub head_reduce_value: f32,
    /// The reduce's status word.
    pub head_reduce_status: u32,
    /// The reduce's total certified count readback.
    pub head_reduce_certified_count: u32,
    /// Head calls seen so far, used as the position label for the head A/B diagnostic.
    pub head_calls: usize,
    /// Final-RMSNorm launches (one in the tail).
    pub final_norm_launches: usize,
    /// Host microseconds in the final-RMSNorm launch and readback.
    pub final_norm_us: u64,
    /// Cumulative norm weight/input/output bytes.
    pub final_norm_bytes: u64,
    /// BF16 mismatches between the device norm and the host norm twin; `usize::MAX` until it runs.
    pub final_norm_bf16_mismatches: usize,
}

/// The layer-0 shared-expert stage comparison inputs, grouped so the diagnostic reads as one call.
struct SharedDiagnose<'a> {
    config: &'a ModelConfig,
    source: &'a dyn WeightSource,
    limit: usize,
    layer_prefix: &'a str,
    x: &'a [f32],
    tokens: usize,
    dim: usize,
    inter: usize,
    act_codes: &'a [u8],
    act_scales: &'a [u8],
    w1_packed: &'a [u8],
    s1: &'a [u8],
    w3_packed: &'a [u8],
    s3: &'a [u8],
    w2_packed: &'a [u8],
    s2: &'a [u8],
    gate_d: &'a Deviceptr,
    up_d: &'a Deviceptr,
    codes_d: &'a Deviceptr,
    scales_d: &'a Deviceptr,
    shared_bytes: &'a [u8],
}

/// One device router decision kept for the stage-aligned router twin: layer, selected ids, and the
/// unbiased scores the kernel published.
#[derive(Debug, Clone)]
pub struct RouterTwinRecord {
    /// Backbone layer.
    pub layer: usize,
    /// The selected expert ids read back after the device router launch.
    pub ids: Vec<u32>,
    /// The device router scores for the token.
    pub scores: Vec<f32>,
}

struct Kernels {
    router: Kernel,
    router_topk: Kernel,
    gate_up: Kernel,
    weighted: Kernel,
    combine: Kernel,
    dense: Kernel,
    swiglu: Kernel,
    head: Kernel,
    head_reduce: Kernel,
    rmsnorm: Kernel,
    _router_module: Module,
    _router_topk_module: Module,
    _gate_up_module: Module,
    _weighted_module: Module,
    _combine_module: Module,
    _dense_module: Module,
    _swiglu_module: Module,
    _head_module: Module,
    _head_reduce_module: Module,
    _rmsnorm_module: Module,
}

/// The c4b routed-MoE device backend.
pub struct MoeDeviceBackendImpl {
    cuda: Cuda,
    kernels: Kernels,
    stats: MoeDeviceStats,
    launches_by_layer: Vec<usize>,
    diagnose_layer0: bool,
    router_twin_enabled: bool,
    router_records: Vec<RouterTwinRecord>,
    head_certified: Vec<u8>,
    // The 1.32 GB host head weight, read once and reused across sequence steps.
    head_weight: Option<Arc<Vec<u8>>>,
    // 2.58 MB of exact BF16 block row maxima, one [vocab][40] image per rank, precomputed once.
    head_block_max: Option<Arc<Vec<u8>>>,
}

impl MoeDeviceBackendImpl {
    /// Opens the driver and loads the four routed-MoE cubins.
    ///
    /// # Errors
    /// A missing/mis-pinned cubin or a driver failure.
    pub fn open(cubin_dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = cubin_dir.as_ref();
        let router_bytes = verified(dir, ROUTER)?;
        let gate_up_bytes = verified(dir, GATE_UP)?;
        let weighted_bytes = verified(dir, WEIGHTED)?;
        let combine_bytes = verified(dir, COMBINE)?;
        let dense_bytes = verified(dir, DENSE)?;
        let swiglu_bytes = verified(dir, SWIGLU)?;
        let head_bytes = verified(dir, HEAD)?;
        let head_reduce_bytes = verified(dir, HEAD_REDUCE)?;
        let rmsnorm_bytes = verified(dir, RMSNORM)?;
        let cuda = Cuda::load()?;
        let (router_module, router, router_topk_module, router_topk) =
            router_launch::load(&cuda, &router_bytes)?;
        let (gate_up_module, gate_up) = routed_gate_up_launch::load(&cuda, &gate_up_bytes)?;
        let (weighted_module, weighted) =
            cuda.load_kernel(&weighted_bytes, KernelId::SwigluMxfp8QuantWeighted)?;
        let (combine_module, combine) = mxfp4_moe_launch::load(&cuda, &combine_bytes)?;
        let (dense_module, dense) = mxfp8_dense_launch::load(&cuda, &dense_bytes)?;
        let (swiglu_module, swiglu) =
            cuda.load_kernel(&swiglu_bytes, KernelId::SwigluMxfp8Quant)?;
        let (head_module, head) = cert_lm_head_launch::load(&cuda, &head_bytes)?;
        let (head_reduce_module, head_reduce) = head_reduce::load(&cuda, &head_reduce_bytes)?;
        let (rmsnorm_module, rmsnorm) = rmsnorm_launch::load(&cuda, &rmsnorm_bytes)?;
        Ok(Self {
            cuda,
            kernels: Kernels {
                router,
                router_topk,
                gate_up,
                weighted,
                combine,
                dense,
                swiglu,
                head,
                head_reduce,
                rmsnorm,
                _router_module: router_module,
                _router_topk_module: router_topk_module,
                _gate_up_module: gate_up_module,
                _weighted_module: weighted_module,
                _combine_module: combine_module,
                _dense_module: dense_module,
                _swiglu_module: swiglu_module,
                _head_module: head_module,
                _head_reduce_module: head_reduce_module,
                _rmsnorm_module: rmsnorm_module,
            },
            stats: MoeDeviceStats {
                layer0_host_max_abs: f32::NAN,
                layer0_host_rel: f32::NAN,
                layer0_host_magnitude: f32::NAN,
                layer0_bf16_twin_mismatches: usize::MAX,
                layer0_bf16_twin_max_abs: f32::NAN,
                layer0_shared_bf16_mismatches: usize::MAX,
                layer0_shared_host_max_abs: f32::NAN,
                layer0_shared_host_rel: f32::NAN,
                layer0_shared_host_magnitude: f32::NAN,
                final_norm_bf16_mismatches: usize::MAX,
                ..MoeDeviceStats::default()
            },
            diagnose_layer0: false,
            router_twin_enabled: std::env::var("SPARKFLASH_ROUTER_TWIN").is_ok(),
            router_records: Vec::new(),
            launches_by_layer: vec![0; 64],
            head_certified: Vec::new(),
            head_weight: None,
            head_block_max: None,
        })
    }

    /// Prints the per-stage layer-0 diagnostics (router, gate/up, weighted operand, combine) in
    /// addition to the always-on layer-0 bar.
    #[must_use]
    pub fn with_layer0_diagnose(mut self) -> Self {
        self.diagnose_layer0 = true;
        self
    }

    /// The layer 0/1/14 router decisions captured for the stage-aligned router twin.
    #[must_use]
    pub fn router_records(&self) -> &[RouterTwinRecord] {
        &self.router_records
    }

    /// The counters since construction.
    #[must_use]
    pub fn stats(&self) -> MoeDeviceStats {
        self.stats
    }

    /// Routed launches on one layer.
    #[must_use]
    pub fn launches_for(&self, layer: usize) -> usize {
        self.launches_by_layer.get(layer).copied().unwrap_or(0)
    }

    fn diagnose_shared(&mut self, diag: SharedDiagnose<'_>) -> Result<(), EngineError> {
        let SharedDiagnose {
            config,
            source,
            limit,
            layer_prefix,
            x,
            tokens,
            dim,
            inter,
            act_codes,
            act_scales,
            w1_packed,
            s1,
            w3_packed,
            s3,
            w2_packed,
            s2,
            gate_d,
            up_d,
            codes_d,
            scales_d,
            shared_bytes,
        } = diag;
        let mut gate_bytes = vec![0u8; tokens * inter * 2];
        let mut up_bytes = vec![0u8; tokens * inter * 2];
        let mut codes = vec![0u8; tokens * inter];
        let mut scales = vec![0u8; tokens * (inter / QUANT_BLOCK)];
        self.cuda.copy_dtoh(&mut gate_bytes, gate_d).map_err(drv)?;
        self.cuda.copy_dtoh(&mut up_bytes, up_d).map_err(drv)?;
        self.cuda.copy_dtoh(&mut codes, codes_d).map_err(drv)?;
        self.cuda.copy_dtoh(&mut scales, scales_d).map_err(drv)?;
        let expected_gate =
            mxfp8_ref::dense_gemm(act_codes, act_scales, w1_packed, s1, tokens, inter, dim);
        let expected_up =
            mxfp8_ref::dense_gemm(act_codes, act_scales, w3_packed, s3, tokens, inter, dim);
        let gate_mismatches = expected_gate
            .iter()
            .zip(gate_bytes.as_chunks::<2>().0.iter())
            .filter(|(expected, device)| **expected != u16::from_le_bytes(**device))
            .count();
        let up_mismatches = expected_up
            .iter()
            .zip(up_bytes.as_chunks::<2>().0.iter())
            .filter(|(expected, device)| **expected != u16::from_le_bytes(**device))
            .count();
        let gate_f32 = decode_bf16(&gate_bytes);
        let up_f32 = decode_bf16(&up_bytes);
        let mut expected_codes = Vec::with_capacity(tokens * inter);
        let mut expected_scales = Vec::with_capacity(tokens * (inter / QUANT_BLOCK));
        for row in 0..tokens {
            let (row_codes, row_scales) = swiglu_ref::quantise_row(
                &gate_f32[row * inter..(row + 1) * inter],
                &up_f32[row * inter..(row + 1) * inter],
                swiglu_ref::SWIGLU_LIMIT,
            );
            expected_codes.extend_from_slice(&row_codes);
            expected_scales.extend_from_slice(&row_scales);
        }
        let code_mismatches = expected_codes
            .iter()
            .zip(&codes)
            .filter(|(a, b)| a != b)
            .count();
        let scale_mismatches = expected_scales
            .iter()
            .zip(&scales)
            .filter(|(a, b)| a != b)
            .count();
        let expected_shared =
            mxfp8_ref::dense_gemm(&codes, &scales, w2_packed, s2, tokens, dim, inter);
        let shared_mismatches = expected_shared
            .iter()
            .zip(shared_bytes.as_chunks::<2>().0.iter())
            .filter(|(expected, device)| **expected != u16::from_le_bytes(**device))
            .count();
        let mut host_shared = vec![0.0f32; x.len()];
        let mut host_scratch = MoeScratch::for_config(config);
        add_shared_expert(
            config,
            layer_prefix,
            x,
            source,
            limit,
            &mut host_scratch,
            &mut host_shared,
        )?;
        let device_shared = decode_bf16(shared_bytes);
        let mut max_abs = 0.0f32;
        let mut host_max = 0.0f32;
        for (device, host) in device_shared.iter().zip(&host_shared) {
            max_abs = (device - host).abs().max(max_abs);
            host_max = host.abs().max(host_max);
        }
        println!(
            "moe layer0 diag: shared gate/up bf16 mismatched={gate_mismatches}/{up_mismatches} codes={code_mismatches} scales={scale_mismatches} shared_bf16_mismatches={shared_mismatches}"
        );
        println!(
            "moe layer0 diag: shared host_f32_max_abs={max_abs:e} host_max={host_max:e} relative={:e}",
            if host_max > 0.0 {
                max_abs / host_max
            } else {
                0.0
            }
        );
        self.stats.layer0_shared_bf16_mismatches = shared_mismatches;
        self.stats.layer0_shared_host_max_abs = max_abs;
        self.stats.layer0_shared_host_magnitude = host_max;
        self.stats.layer0_shared_host_rel = if host_max > 0.0 {
            max_abs / host_max
        } else {
            0.0
        };
        if gate_mismatches + up_mismatches + code_mismatches + scale_mismatches + shared_mismatches
            != 0
        {
            return Err(err(
                "device shared expert layer-0 diagnose",
                format!(
                    "stage mismatch: gate={gate_mismatches} up={up_mismatches} codes={code_mismatches} scales={scale_mismatches} shared={shared_mismatches}"
                ),
            ));
        }
        Ok(())
    }
}

impl MoeDeviceBackend for MoeDeviceBackendImpl {
    fn ffn(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        sparkflash_kernels::cuda_driver::stage_timing_set_layer(layer);
        let dim = config.dim;
        let inter = config.moe_inter_dim;
        let n_routed = config.n_routed_experts;
        let top_k = config.n_activated_experts;
        if dim == 0 || inter == 0 || !x.len().is_multiple_of(dim) || out.len() != x.len() {
            return Err(err(
                "device routed MoE",
                format!(
                    "dim {dim}, inter {inter}, {} input values and {} output values do not describe whole tokens",
                    x.len(),
                    out.len()
                ),
            ));
        }
        if top_k == 0 || n_routed == 0 || top_k > 64 {
            return Err(err(
                "device routed MoE",
                format!(
                    "routing {n_routed} experts with top-k {top_k} is outside the kernel contract"
                ),
            ));
        }
        let tokens = x.len() / dim;
        if !inter.is_multiple_of(RANKS) || (inter / RANKS) > PADDED {
            return Err(err(
                "device routed MoE",
                format!("intermediate width {inter} is not four rank slices of at most {PADDED}"),
            ));
        }
        let width = inter / RANKS;
        let rows = tokens * top_k;
        let staging = std::time::Instant::now();
        let mut ptrs: Vec<Deviceptr> = Vec::new();
        let mut staged_bytes = 0u64;
        let mut host_bytes = 0u64;
        let mut allocated = 0usize;

        let gate_name = format!("{prefix}.ffn.gate.weight");
        let bias_name = format!("{prefix}.ffn.gate.bias");
        let gate = source
            .read(&gate_name)
            .map_err(|error| read_err(&gate_name, error))?;
        let bias = source
            .read(&bias_name)
            .map_err(|error| read_err(&bias_name, error))?;
        if gate.len() != n_routed * dim * 2 {
            return Err(err(
                &gate_name,
                format!(
                    "expected {} BF16 bytes, got {}",
                    n_routed * dim * 2,
                    gate.len()
                ),
            ));
        }
        if bias.len() != n_routed * 4 {
            return Err(err(
                &bias_name,
                format!("expected {} F32 bytes, got {}", n_routed * 4, bias.len()),
            ));
        }
        let x_bf16: Vec<u8> = x
            .iter()
            .flat_map(|value| bf16_bits(*value).to_le_bytes())
            .collect();
        let (act_codes, act_scales) = quantise_activation(x, "device routed activation")?;

        let x_d = self.cuda.alloc(x_bf16.len(), &mut allocated).map_err(drv)?;
        let gate_d = self.cuda.alloc(gate.len(), &mut allocated).map_err(drv)?;
        let bias_d = self.cuda.alloc(bias.len(), &mut allocated).map_err(drv)?;
        let ids_d = self.cuda.alloc(rows * 4, &mut allocated).map_err(drv)?;
        let weights_d = self.cuda.alloc(rows * 4, &mut allocated).map_err(drv)?;
        let scores_d = self
            .cuda
            .alloc(tokens * n_routed * 4, &mut allocated)
            .map_err(drv)?;
        ptrs.extend([x_d, gate_d, bias_d, ids_d, weights_d, scores_d]);
        self.cuda.copy_htod(&x_d, &x_bf16).map_err(drv)?;
        self.cuda.copy_htod(&gate_d, &gate).map_err(drv)?;
        self.cuda.copy_htod(&bias_d, &bias).map_err(drv)?;
        staged_bytes += (x_bf16.len() + gate.len() + bias.len()) as u64;

        let started = std::time::Instant::now();
        // Count what this layer actually launches, so the accounting is measured and not
        // asserted out of thin air; the plan constant is checked against it before it is stored.
        let mut launched = 0usize;
        router_launch::launch(
            &self.cuda,
            self.kernels.router,
            self.kernels.router_topk,
            &RouterLaunchParams {
                x: x_d,
                gate: gate_d,
                bias: bias_d,
                ids: ids_d,
                weights: weights_d,
                scores: scores_d,
                tokens: tokens as u32,
                trace_ring: 0,
                layer_id: layer as u32,
                step_seq: 0,
            },
        )
        .map_err(drv)?;
        launched += ROUTED_ROUTER_LAUNCHES;
        let mut id_bytes = vec![0u8; rows * 4];
        self.cuda.copy_dtoh(&mut id_bytes, &ids_d).map_err(drv)?;
        host_bytes += id_bytes.len() as u64;
        let ids: Vec<u32> = id_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        let (table, local_ids) = local_table(&ids, n_routed)?;
        let local_count = table.len();
        let router_twin =
            (self.diagnose_layer0 || self.router_twin_enabled) && layer < config.n_layers;
        let device_weights: Vec<f32> = if router_twin {
            let mut weight_bytes = vec![0u8; rows * 4];
            self.cuda
                .copy_dtoh(&mut weight_bytes, &weights_d)
                .map_err(drv)?;
            host_bytes += weight_bytes.len() as u64;
            weight_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_bits(u32::from_le_bytes(*chunk)))
                .collect()
        } else {
            Vec::new()
        };
        let device_scores: Vec<f32> = if router_twin {
            let mut score_bytes = vec![0u8; tokens * n_routed * 4];
            self.cuda
                .copy_dtoh(&mut score_bytes, &scores_d)
                .map_err(drv)?;
            host_bytes += score_bytes.len() as u64;
            score_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_bits(u32::from_le_bytes(*chunk)))
                .collect()
        } else {
            Vec::new()
        };

        let experts = read_experts(source, prefix, &table, dim, inter)?;
        let expert_bytes: usize = experts.iter().map(Expert::bytes).sum();
        staged_bytes += expert_bytes as u64;
        host_bytes += expert_bytes as u64;
        if router_twin
            && !self
                .router_records
                .iter()
                .any(|record| record.layer == layer)
        {
            self.router_records.push(RouterTwinRecord {
                layer,
                ids: ids.clone(),
                scores: device_scores.clone(),
            });
        }
        if router_twin {
            let x_bits: Vec<u16> = x_bf16
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| u16::from_le_bytes(*chunk))
                .collect();
            let gate_bits: Vec<u16> = gate
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| u16::from_le_bytes(*chunk))
                .collect();
            let bias_f32: Vec<f32> = bias
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_bits(u32::from_le_bytes(*chunk)))
                .collect();
            let params = router_ref::RouterParams {
                hidden: dim,
                n_routed,
                top_k,
                gate_temp: config.gate_temp as f32,
                norm_eps: config.norm_eps as f32,
                scale: config.route_scale as f32,
                norm_topk_prob: config.norm_topk_prob,
            };
            match router_ref::route_bf16(&x_bits, &gate_bits, &bias_f32, params) {
                Ok(host) => {
                    let ids_equal = host.ids == ids;
                    let weight_delta = host
                        .weights
                        .iter()
                        .zip(device_weights.iter())
                        .map(|(host, device)| (host - device).abs())
                        .fold(0.0f32, f32::max);
                    let score_delta = host
                        .scores
                        .iter()
                        .zip(device_scores.iter())
                        .map(|(host, device)| (host - device).abs())
                        .fold(0.0f32, f32::max);
                    println!(
                        "moe router twin: layer={layer} ids device={ids:?} host={:?} equal={ids_equal} scores_max_abs={score_delta:e} weights_max_abs={weight_delta:e}",
                        host.ids
                    );
                    if !ids_equal {
                        println!(
                            "moe router twin: layer={layer} DISCRETE SELECTION FLIP - device ids {ids:?} vs host ids {:?}",
                            host.ids
                        );
                    }
                }
                Err(message) => {
                    return Err(err("device routed MoE layer-0 diagnose", message));
                }
            }
            println!(
                "moe layer0 diag: staged {local_count} experts local->global {table:?}, expert_bytes={expert_bytes}, rows={rows}"
            );
        }
        let mut ids_local_bytes = Vec::with_capacity(local_ids.len() * 4);
        for id in &local_ids {
            ids_local_bytes.extend_from_slice(&id.to_le_bytes());
        }

        let act_d = self
            .cuda
            .alloc(act_codes.len(), &mut allocated)
            .map_err(drv)?;
        let act_s_d = self
            .cuda
            .alloc(act_scales.len(), &mut allocated)
            .map_err(drv)?;
        let ids_local_d = self
            .cuda
            .alloc(ids_local_bytes.len(), &mut allocated)
            .map_err(drv)?;
        ptrs.extend([act_d, act_s_d, ids_local_d]);
        self.cuda.copy_htod(&act_d, &act_codes).map_err(drv)?;
        self.cuda.copy_htod(&act_s_d, &act_scales).map_err(drv)?;
        self.cuda
            .copy_htod(&ids_local_d, &ids_local_bytes)
            .map_err(drv)?;
        staged_bytes += (act_codes.len() + act_scales.len() + ids_local_bytes.len()) as u64;

        let out_values = rows * PADDED;
        let mut gate_outs = Vec::with_capacity(RANKS);
        let mut up_outs = Vec::with_capacity(RANKS);
        let mut codes = Vec::with_capacity(RANKS);
        let mut scales = Vec::with_capacity(RANKS);
        let mut rank_gate = Vec::with_capacity(RANKS);
        for rank in 0..RANKS {
            let w_gate = rank_slice(&experts, |expert| &expert.w1, rank, width * (dim / 2));
            let s_gate = rank_slice(
                &experts,
                |expert| &expert.s1,
                rank,
                width * (dim / QUANT_BLOCK),
            );
            let w_up = rank_slice(&experts, |expert| &expert.w3, rank, width * (dim / 2));
            let s_up = rank_slice(
                &experts,
                |expert| &expert.s3,
                rank,
                width * (dim / QUANT_BLOCK),
            );
            let w_gate_d = self.cuda.alloc(w_gate.len(), &mut allocated).map_err(drv)?;
            let s_gate_d = self.cuda.alloc(s_gate.len(), &mut allocated).map_err(drv)?;
            let w_up_d = self.cuda.alloc(w_up.len(), &mut allocated).map_err(drv)?;
            let s_up_d = self.cuda.alloc(s_up.len(), &mut allocated).map_err(drv)?;
            let gate_out_d = self
                .cuda
                .alloc(out_values * 2, &mut allocated)
                .map_err(drv)?;
            let up_out_d = self
                .cuda
                .alloc(out_values * 2, &mut allocated)
                .map_err(drv)?;
            ptrs.extend([w_gate_d, s_gate_d, w_up_d, s_up_d, gate_out_d, up_out_d]);
            self.cuda.copy_htod(&w_gate_d, &w_gate).map_err(drv)?;
            self.cuda.copy_htod(&s_gate_d, &s_gate).map_err(drv)?;
            self.cuda.copy_htod(&w_up_d, &w_up).map_err(drv)?;
            self.cuda.copy_htod(&s_up_d, &s_up).map_err(drv)?;
            staged_bytes += (w_gate.len() + s_gate.len() + w_up.len() + s_up.len()) as u64;
            let base = GateUpLaunchParams {
                act_fp8: act_d,
                act_scales: act_s_d,
                w_fp4: w_gate_d,
                w_scales: s_gate_d,
                expert_ids: ids_local_d,
                out: gate_out_d,
                tokens: tokens as u32,
                trace_ring: 0,
                layer_id: layer as u32,
                step_seq: rank as u64,
            };
            routed_gate_up_launch::launch(&self.cuda, self.kernels.gate_up, &base).map_err(drv)?;
            launched += 1;
            routed_gate_up_launch::launch(
                &self.cuda,
                self.kernels.gate_up,
                &GateUpLaunchParams {
                    w_fp4: w_up_d,
                    w_scales: s_up_d,
                    out: up_out_d,
                    ..base
                },
            )
            .map_err(drv)?;
            launched += 1;

            let codes_d = self
                .cuda
                .alloc(rows * PADDED, &mut allocated)
                .map_err(drv)?;
            let scales_d = self
                .cuda
                .alloc(rows * (PADDED / QUANT_BLOCK), &mut allocated)
                .map_err(drv)?;
            ptrs.extend([codes_d, scales_d]);
            let params = weighted_words(
                gate_out_d, up_out_d, weights_d, codes_d, scales_d, rows, layer, rank,
            );
            self.cuda
                .launch(
                    self.kernels.weighted,
                    (((rows * (PADDED / QUANT_BLOCK)) as u32).div_ceil(8), 1, 1),
                    (256, 1, 1),
                    &params,
                )
                .map_err(drv)?;
            launched += ROUTED_WEIGHTED_LAUNCHES / RANKS;
            if layer == 0 {
                rank_gate.push((w_gate, s_gate, w_up, s_up));
            }
            gate_outs.push(gate_out_d);
            up_outs.push(up_out_d);
            codes.push(codes_d);
            scales.push(scales_d);
        }
        let _ = (&gate_outs, &up_outs);
        let mut device_operand = Vec::with_capacity(RANKS);
        if layer == 0 {
            for rank in 0..RANKS {
                let mut rank_codes = vec![0u8; rows * PADDED];
                let mut rank_scales = vec![0u8; rows * (PADDED / QUANT_BLOCK)];
                self.cuda
                    .copy_dtoh(&mut rank_codes, &codes[rank])
                    .map_err(drv)?;
                self.cuda
                    .copy_dtoh(&mut rank_scales, &scales[rank])
                    .map_err(drv)?;
                host_bytes += (rank_codes.len() + rank_scales.len()) as u64;
                device_operand.push((rank_codes, rank_scales));
            }
        }
        let diag_gate_up: Option<(Vec<u16>, Vec<u16>)> = if self.diagnose_layer0 && layer == 0 {
            let mut gate_bits = vec![0u16; out_values];
            let mut up_bits = vec![0u16; out_values];
            let mut gate_bytes = vec![0u8; out_values * 2];
            let mut up_bytes = vec![0u8; out_values * 2];
            self.cuda
                .copy_dtoh(&mut gate_bytes, &gate_outs[0])
                .map_err(drv)?;
            self.cuda
                .copy_dtoh(&mut up_bytes, &up_outs[0])
                .map_err(drv)?;
            for (slot, chunk) in gate_bytes.as_chunks::<2>().0.iter().enumerate() {
                gate_bits[slot] = u16::from_le_bytes(*chunk);
            }
            for (slot, chunk) in up_bytes.as_chunks::<2>().0.iter().enumerate() {
                up_bits[slot] = u16::from_le_bytes(*chunk);
            }
            Some((gate_bits, up_bits))
        } else {
            None
        };
        self.stats.device_us += started.elapsed().as_micros() as u64;

        let combine_started = std::time::Instant::now();
        // The four serial rank launches ADD their f32 partials into ONE slot accumulator, the
        // [tokens][top_k][dim] sum over the tensor-parallel ranks, and nothing rounds between
        // ranks (audit M2/F7: native sums the rank partials in f32 and rounds each expert's w2
        // output once, in the epilogue, model.py:899-903). The old chain seeded a bf16 hidden row
        // and read a bf16 output back after every rank, rounding each partial.
        let slot_acc_d = self
            .cuda
            .alloc(tokens * top_k * dim * 4, &mut allocated)
            .map_err(drv)?;
        ptrs.push(slot_acc_d);
        // Every rank ADDS into the accumulator, so it starts zeroed.
        self.cuda
            .copy_htod(&slot_acc_d, &vec![0u8; tokens * top_k * dim * 4])
            .map_err(drv)?;
        let mut rank_w2 = Vec::with_capacity(RANKS);
        let mut rank_s2 = Vec::with_capacity(RANKS);
        for rank in 0..RANKS {
            let w2 = rank_slice_padded(
                &experts,
                |expert| &expert.w2,
                dim,
                inter / 2,
                rank * (width / 2),
                width / 2,
                PADDED / 2,
            );
            let s2 = rank_slice_padded(
                &experts,
                |expert| &expert.s2,
                dim,
                inter / QUANT_BLOCK,
                rank * (width / QUANT_BLOCK),
                width / QUANT_BLOCK,
                PADDED / QUANT_BLOCK,
            );
            let w2_d = self.cuda.alloc(w2.len(), &mut allocated).map_err(drv)?;
            let s2_d = self.cuda.alloc(s2.len(), &mut allocated).map_err(drv)?;
            ptrs.extend([w2_d, s2_d]);
            self.cuda.copy_htod(&w2_d, &w2).map_err(drv)?;
            self.cuda.copy_htod(&s2_d, &s2).map_err(drv)?;
            staged_bytes += (w2.len() + s2.len()) as u64;
            mxfp4_moe_launch::launch(
                &self.cuda,
                self.kernels.combine,
                &Mxfp4MoeParams::new(
                    codes[rank],
                    scales[rank],
                    w2_d,
                    s2_d,
                    ids_local_d,
                    slot_acc_d,
                    tokens as u32,
                    0,
                    layer as u32,
                    rank as u64,
                ),
            )
            .map_err(drv)?;
            launched += ROUTED_COMBINE_LAUNCHES / RANKS;
            if layer == 0 {
                rank_w2.push(w2);
                rank_s2.push(s2);
            }
        }
        // ONE readback after the four ranks, of the f32 accumulator.
        let mut slot_acc_bytes = vec![0u8; tokens * top_k * dim * 4];
        self.cuda
            .copy_dtoh(&mut slot_acc_bytes, &slot_acc_d)
            .map_err(drv)?;
        host_bytes += slot_acc_bytes.len() as u64;
        let slot_acc: Vec<f32> = slot_acc_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        // The native routed epilogue up to the task-10 boundary: one bf16 round per expert, the
        // expert outputs added in ascending expert-id order in f32. The shared-expert bf16 add and
        // the single MoE bf16 round are task-10's host seams, so the chain keeps handing the
        // engine an f32 routed tensor exactly as before task-4 (no half-implemented boundary).
        let routed = mxfp4_ref::moe_combine_routed_f32(&slot_acc, &local_ids, tokens, dim);
        out.copy_from_slice(&routed);
        let routed_device_us = combine_started.elapsed().as_micros() as u64;
        let routed_staging_us = staging.elapsed().as_micros() as u64;
        self.stats.device_us += routed_device_us;
        self.stats.routed_device_us += routed_device_us;
        self.stats.staging_us += routed_staging_us;
        self.stats.routed_staging_us += routed_staging_us;
        // The plan is a constant but the count is measured: a launch that moved, appeared or did
        // not run fails here rather than being recorded as if the plan had run.
        if launched != ROUTED_LAUNCHES_PER_LAYER {
            return Err(err(
                "device routed MoE launches",
                format!(
                    "the layer launched {launched} kernels but the plan is {ROUTED_LAUNCHES_PER_LAYER}: one router, then per rank a gate GEMM, an up GEMM, the weighted SwiGLU quantiser and the combine add"
                ),
            ));
        }
        self.stats.routed_launches += launched;
        self.stats.launches += launched;
        self.stats.staged_bytes += staged_bytes;
        self.stats.host_bytes += host_bytes;
        self.stats.routed_host_bytes = self.stats.routed_host_bytes.saturating_add(host_bytes);
        self.stats.layers += 1;
        if self.launches_by_layer.len() <= layer {
            self.launches_by_layer.resize(layer + 1, 0);
        }
        self.launches_by_layer[layer] = launched;
        let _ = (limit, local_count);
        self.cuda.free_buffers(&ptrs.iter().collect::<Vec<_>>());
        if layer == 0 {
            // The bar the chain gate alone could not provide. The device combine is the native
            // structure now: one f32 slot accumulator over the four ranks, no per-rank bf16
            // rounding. The twin is mxfp4_ref::moe_combine_rank over the SAME device-built operand
            // bytes and the SAME rank weight slices, then the same routed epilogue over that twin
            // accumulator; any mismatch is a wiring defect, not FP4 spread.
            let mut twin_acc = vec![0.0f32; tokens * top_k * dim];
            for rank in 0..RANKS {
                let _cost = mxfp4_ref::moe_combine_rank(
                    &mut twin_acc,
                    &device_operand[rank].0,
                    &device_operand[rank].1,
                    &rank_w2[rank],
                    &rank_s2[rank],
                    &local_ids,
                    tokens,
                    dim,
                    local_count,
                );
            }
            let acc_mismatches = slot_acc
                .iter()
                .zip(twin_acc.iter())
                .filter(|(device, twin)| device.to_bits() != twin.to_bits())
                .count();
            let mut twin_max_abs = 0.0f32;
            for (device, twin) in slot_acc.iter().zip(twin_acc.iter()) {
                twin_max_abs = twin_max_abs.max((device - twin).abs());
            }
            let twin_routed = mxfp4_ref::moe_combine_routed_f32(&twin_acc, &local_ids, tokens, dim);
            let routed_mismatches = routed
                .iter()
                .zip(twin_routed.iter())
                .filter(|(device, twin)| device.to_bits() != twin.to_bits())
                .count();
            let twin_mismatches = acc_mismatches + routed_mismatches;
            self.stats.layer0_bf16_twin_mismatches = twin_mismatches;
            self.stats.layer0_bf16_twin_max_abs = twin_max_abs;
            if twin_mismatches != 0 {
                return Err(err(
                    "device routed MoE layer-0 bar",
                    format!(
                        "the device f32 combine accumulator differs from the host moe_combine_rank twin on the same device operand: {twin_mismatches} f32 bits mismatch ({acc_mismatches} accumulator, {routed_mismatches} routed epilogue), max abs {twin_max_abs:e}"
                    ),
                ));
            }

            let mut host_routed = vec![0.0f32; out.len()];
            let mut scratch = MoeScratch::for_config(config);
            routed_forward(
                config,
                prefix,
                x,
                source,
                limit,
                &mut scratch,
                &mut host_routed,
            )?;
            let delta = out
                .iter()
                .zip(host_routed.iter())
                .map(|(device, host)| (device - host).abs())
                .fold(0.0f32, f32::max);
            let host_magnitude = host_routed
                .iter()
                .map(|host| host.abs())
                .fold(0.0f32, f32::max);
            let relative = delta / host_magnitude.max(f32::MIN_POSITIVE);
            self.stats.layer0_host_max_abs = delta;
            self.stats.layer0_host_magnitude = host_magnitude;
            self.stats.layer0_host_rel = relative;
            if !delta.is_finite() || !host_magnitude.is_finite() {
                return Err(err(
                    "device routed MoE layer-0 bar",
                    "the layer-0 routed comparison is not finite".to_string(),
                ));
            }

            if self.diagnose_layer0 {
                let width = inter / RANKS;
                let (device_gate, device_up) = diag_gate_up
                    .as_ref()
                    .expect("diagnose mode reads the rank-0 gate/up outputs");
                let (host_gate, _) = gate_up_ref::gate_up(
                    &act_codes,
                    &act_scales,
                    &rank_gate[0].0,
                    &rank_gate[0].1,
                    &local_ids,
                    tokens,
                    local_count,
                );
                let (host_up, _) = gate_up_ref::gate_up(
                    &act_codes,
                    &act_scales,
                    &rank_gate[0].2,
                    &rank_gate[0].3,
                    &local_ids,
                    tokens,
                    local_count,
                );
                let gate_mismatches = host_gate
                    .iter()
                    .zip(device_gate.iter())
                    .filter(|(host, device)| host != device)
                    .count();
                let up_mismatches = host_up
                    .iter()
                    .zip(device_up.iter())
                    .filter(|(host, device)| host != device)
                    .count();
                println!(
                    "moe layer0 diag: gate_up rank0 device-vs-host bits mismatched={gate_mismatches} up={up_mismatches} of {} padded values",
                    tokens * top_k * PADDED
                );
                let mut weighted_code_mismatches = 0usize;
                let mut weighted_scale_mismatches = 0usize;
                for row in 0..rows {
                    let gate_row: Vec<f32> = (0..width)
                        .map(|column| router_ref::bf16_to_f32(device_gate[row * PADDED + column]))
                        .collect();
                    let up_row: Vec<f32> = (0..width)
                        .map(|column| router_ref::bf16_to_f32(device_up[row * PADDED + column]))
                        .collect();
                    let (host_codes, host_scales) = swiglu_ref::quantise_row_weighted(
                        &gate_row,
                        &up_row,
                        swiglu_ref::SWIGLU_LIMIT,
                        device_weights[row],
                    );
                    let device_codes = &device_operand[0].0[row * PADDED..row * PADDED + width];
                    let device_scales = &device_operand[0].1[row * (PADDED / QUANT_BLOCK)
                        ..row * (PADDED / QUANT_BLOCK) + width / QUANT_BLOCK];
                    weighted_code_mismatches += host_codes
                        .iter()
                        .zip(device_codes.iter())
                        .filter(|(host, device)| host != device)
                        .count();
                    weighted_scale_mismatches += host_scales
                        .iter()
                        .zip(device_scales.iter())
                        .filter(|(host, device)| host != device)
                        .count();
                }
                println!(
                    "moe layer0 diag: weighted rank0 codes_mismatched={weighted_code_mismatches} scales_mismatched={weighted_scale_mismatches} (width={width}, rows={rows})"
                );
                let stage_error = gate_mismatches != 0
                    || up_mismatches != 0
                    || weighted_code_mismatches != 0
                    || weighted_scale_mismatches != 0;
                println!(
                    "moe layer0 diag: native combine twin exact={} mismatches={twin_mismatches} acc={acc_mismatches} routed_epilogue={routed_mismatches} host_f32_max_abs={delta:e} host_max={host_magnitude:e} relative={relative:e}",
                    twin_mismatches == 0
                );
                if stage_error {
                    return Err(err(
                        "device routed MoE layer-0 diagnose",
                        format!(
                            "stage mismatch: gate={gate_mismatches} up={up_mismatches} weighted_codes={weighted_code_mismatches} weighted_scales={weighted_scale_mismatches}"
                        ),
                    ));
                }
            }
        }
        Ok(true)
    }

    fn shared_expert(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        sparkflash_kernels::cuda_driver::stage_timing_set_layer(layer);
        let dim = config.dim;
        let inter = config.moe_inter_dim;
        if config.n_shared_experts != 1 {
            return Ok(false);
        }
        if dim == 0 || inter == 0 || !x.len().is_multiple_of(dim) || out.len() != x.len() {
            return Err(err(
                "device shared expert",
                format!(
                    "dim {dim}, inter {inter}, {} input values and {} output values do not describe whole tokens",
                    x.len(),
                    out.len()
                ),
            ));
        }
        let tokens = x.len() / dim;
        let staging_started = std::time::Instant::now();
        let base = format!("{prefix}.ffn.shared_experts");
        let w1 = read_spec(
            source,
            &format!("{base}.w1.weight"),
            DType::F8E4M3,
            &[inter, dim],
        )?;
        let (s1, s3, s2) = read_shared_scale_grids(source, &base, inter, dim)?;
        let w3 = read_spec(
            source,
            &format!("{base}.w3.weight"),
            DType::F8E4M3,
            &[inter, dim],
        )?;
        let w2 = read_spec(
            source,
            &format!("{base}.w2.weight"),
            DType::F8E4M3,
            &[dim, inter],
        )?;
        let w1_packed = pack_b12x(&w1, inter, dim);
        let w3_packed = pack_b12x(&w3, inter, dim);
        let w2_packed = pack_b12x(&w2, dim, inter);
        if w1_packed.len() != mxfp8_ref::b12x_buffer_bytes(inter, dim)
            || w3_packed.len() != mxfp8_ref::b12x_buffer_bytes(inter, dim)
            || w2_packed.len() != mxfp8_ref::b12x_buffer_bytes(dim, inter)
        {
            return Err(err(
                "device shared expert",
                "the B12x repack does not match the kernel's buffer sizes".to_owned(),
            ));
        }
        let (act_codes, act_scales) = quantise_activation(x, "device shared activation")?;
        let mut allocated = 0usize;
        let mut ptrs = Vec::new();
        let mut staged_bytes = 0u64;
        let a_d = self
            .cuda
            .alloc(act_codes.len(), &mut allocated)
            .map_err(drv)?;
        let as_d = self
            .cuda
            .alloc(act_scales.len(), &mut allocated)
            .map_err(drv)?;
        let w1_d = self
            .cuda
            .alloc(w1_packed.len(), &mut allocated)
            .map_err(drv)?;
        let s1_d = self.cuda.alloc(s1.len(), &mut allocated).map_err(drv)?;
        let w3_d = self
            .cuda
            .alloc(w3_packed.len(), &mut allocated)
            .map_err(drv)?;
        let s3_d = self.cuda.alloc(s3.len(), &mut allocated).map_err(drv)?;
        let gate_d = self
            .cuda
            .alloc(tokens * inter * 2, &mut allocated)
            .map_err(drv)?;
        let up_d = self
            .cuda
            .alloc(tokens * inter * 2, &mut allocated)
            .map_err(drv)?;
        let codes_d = self
            .cuda
            .alloc(tokens * inter, &mut allocated)
            .map_err(drv)?;
        let scales_d = self
            .cuda
            .alloc(tokens * (inter / QUANT_BLOCK), &mut allocated)
            .map_err(drv)?;
        let w2_d = self
            .cuda
            .alloc(w2_packed.len(), &mut allocated)
            .map_err(drv)?;
        let s2_d = self.cuda.alloc(s2.len(), &mut allocated).map_err(drv)?;
        let shared_d = self
            .cuda
            .alloc(tokens * dim * 2, &mut allocated)
            .map_err(drv)?;
        ptrs.extend([
            a_d, as_d, w1_d, s1_d, w3_d, s3_d, gate_d, up_d, codes_d, scales_d, w2_d, s2_d,
            shared_d,
        ]);
        self.cuda.copy_htod(&a_d, &act_codes).map_err(drv)?;
        self.cuda.copy_htod(&as_d, &act_scales).map_err(drv)?;
        self.cuda.copy_htod(&w1_d, &w1_packed).map_err(drv)?;
        self.cuda.copy_htod(&s1_d, &s1).map_err(drv)?;
        self.cuda.copy_htod(&w3_d, &w3_packed).map_err(drv)?;
        self.cuda.copy_htod(&s3_d, &s3).map_err(drv)?;
        self.cuda.copy_htod(&w2_d, &w2_packed).map_err(drv)?;
        self.cuda.copy_htod(&s2_d, &s2).map_err(drv)?;
        staged_bytes += (act_codes.len()
            + act_scales.len()
            + w1_packed.len()
            + s1.len()
            + w3_packed.len()
            + s3.len()
            + w2_packed.len()
            + s2.len()) as u64;
        let staging_us = staging_started.elapsed().as_micros() as u64;
        let device_started = std::time::Instant::now();
        for (w, s, out) in [(&w1_d, &s1_d, gate_d), (&w3_d, &s3_d, up_d)] {
            let params = Mxfp8DenseParams::new(
                a_d,
                as_d,
                *w,
                *s,
                out,
                tokens as u32,
                inter as u32,
                dim as u32,
                0,
                layer as u32,
                0,
            );
            mxfp8_dense_launch::launch(&self.cuda, self.kernels.dense, &params).map_err(drv)?;
        }
        let total_blocks = (tokens * inter / QUANT_BLOCK) as u32;
        let grid = (total_blocks.div_ceil(8), 1, 1);
        let swiglu_words = [
            gate_d.0,
            up_d.0,
            codes_d.0,
            scales_d.0,
            tokens as u64,
            inter as u64,
            SWIGLU_LIMIT_BITS,
            0,
            layer as u64,
            0,
        ];
        self.cuda
            .launch(self.kernels.swiglu, grid, (256, 1, 1), &swiglu_words)
            .map_err(drv)?;
        let params = Mxfp8DenseParams::new(
            codes_d,
            scales_d,
            w2_d,
            s2_d,
            shared_d,
            tokens as u32,
            dim as u32,
            inter as u32,
            0,
            layer as u32,
            0,
        );
        mxfp8_dense_launch::launch(&self.cuda, self.kernels.dense, &params).map_err(drv)?;
        let mut shared_bytes = vec![0u8; tokens * dim * 2];
        self.cuda
            .copy_dtoh(&mut shared_bytes, &shared_d)
            .map_err(drv)?;
        let device_us = device_started.elapsed().as_micros() as u64;
        let shared_f32 = decode_bf16(&shared_bytes);
        for (destination, value) in out.iter_mut().zip(&shared_f32) {
            *destination += *value;
        }
        if self.diagnose_layer0 && layer == 0 {
            self.diagnose_shared(SharedDiagnose {
                config,
                source,
                limit,
                layer_prefix: prefix,
                x,
                tokens,
                dim,
                inter,
                act_codes: &act_codes,
                act_scales: &act_scales,
                w1_packed: &w1_packed,
                s1: &s1,
                w3_packed: &w3_packed,
                s3: &s3,
                w2_packed: &w2_packed,
                s2: &s2,
                gate_d: &gate_d,
                up_d: &up_d,
                codes_d: &codes_d,
                scales_d: &scales_d,
                shared_bytes: &shared_bytes,
            })?;
        }
        let host_bytes = (shared_bytes.len() + act_codes.len() + act_scales.len()) as u64;
        self.stats.shared_launches += 4;
        self.stats.launches += 4;
        if self.launches_by_layer.len() <= layer {
            self.launches_by_layer.resize(layer + 1, 0);
        }
        self.launches_by_layer[layer] += 4;
        self.stats.shared_us += staging_us + device_us;
        self.stats.shared_staging_us += staging_us;
        self.stats.shared_device_us += device_us;
        self.stats.staging_us += staging_us;
        self.stats.device_us += device_us;
        self.stats.shared_bytes += staged_bytes;
        self.stats.host_bytes += host_bytes;
        self.stats.shared_host_bytes = self.stats.shared_host_bytes.saturating_add(host_bytes);
        self.cuda.free_buffers(&ptrs.iter().collect::<Vec<_>>());
        Ok(true)
    }

    fn final_norm(
        &mut self,
        config: &ModelConfig,
        x: &[f32],
        source: &dyn WeightSource,
        _limit: usize,
        out: &mut Vec<u8>,
    ) -> Result<bool, EngineError> {
        let dim = config.dim;
        if dim == 0 || x.is_empty() || !x.len().is_multiple_of(dim) {
            return Ok(false);
        }
        let rows = x.len() / dim;
        let eps = config.norm_eps as f32;
        let started = std::time::Instant::now();
        let weight = read_spec(source, "norm.weight", DType::BF16, &[dim])?;
        let x_bf16: Vec<u8> = x
            .iter()
            .flat_map(|value| bf16_bits(*value).to_le_bytes())
            .collect();
        let mut allocated = 0usize;
        let mut ptrs = Vec::new();
        let x_d = self.cuda.alloc(x_bf16.len(), &mut allocated).map_err(drv)?;
        let weight_d = self.cuda.alloc(weight.len(), &mut allocated).map_err(drv)?;
        let out_d = self.cuda.alloc(x_bf16.len(), &mut allocated).map_err(drv)?;
        ptrs.extend([x_d, weight_d, out_d]);
        self.cuda.copy_htod(&x_d, &x_bf16).map_err(drv)?;
        self.cuda.copy_htod(&weight_d, &weight).map_err(drv)?;
        let params = RmsNormLaunchParams {
            x: x_d,
            weight: weight_d,
            out: out_d,
            rows: rows as u32,
            cols: dim as u32,
            eps,
            trace_ring: 0,
            layer_id: 0,
            step_seq: 0,
        };
        rmsnorm_launch::launch(&self.cuda, self.kernels.rmsnorm, &params).map_err(drv)?;
        let mut bytes = vec![0u8; x_bf16.len()];
        self.cuda.copy_dtoh(&mut bytes, &out_d).map_err(drv)?;
        let expected = rmsnorm_ref::final_norm_bf16(&x_bf16, &weight, rows, dim, eps)
            .map_err(|error| err("device final norm", error))?;
        let mismatches = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .zip(expected.as_chunks::<2>().0)
            .filter(|(device, twin)| device != twin)
            .count();
        println!(
            "final norm diag: rows={rows} cols={dim} bf16_mismatches={mismatches} bytes={}",
            bytes.len()
        );
        if mismatches != 0 {
            return Err(err(
                "device final norm diagnose",
                format!(
                    "the device BF16 norm differs from the host bf16 twin in {mismatches} of {} values",
                    bytes.len() / 2
                ),
            ));
        }
        self.cuda.free_buffers(&ptrs.iter().collect::<Vec<_>>());
        let elapsed_us = started.elapsed().as_micros() as u64;
        self.stats.final_norm_launches += 1;
        self.stats.final_norm_us += elapsed_us;
        self.stats.device_us += elapsed_us;
        self.stats.final_norm_bytes += (weight.len() + x_bf16.len() + bytes.len()) as u64;
        self.stats.final_norm_bf16_mismatches = mismatches;
        *out = bytes;
        Ok(true)
    }

    fn lm_head(
        &mut self,
        config: &ModelConfig,
        hidden: &[f32],
        source: &dyn WeightSource,
        _limit: usize,
        logits: &mut [f32],
    ) -> Result<Option<u32>, EngineError> {
        let dim = config.dim;
        let vocab = config.vocab_size;
        if dim == 0
            || vocab == 0
            || !vocab.is_multiple_of(RANKS)
            || hidden.len() != dim
            || logits.len() != vocab
        {
            return Ok(None);
        }
        let per_rank = vocab / RANKS;
        if per_rank != HEAD_VOCAB_PER_RANK {
            return Ok(None);
        }
        let started = std::time::Instant::now();
        let staging_started = std::time::Instant::now();
        let head = if let Some(head) = &self.head_weight {
            Arc::clone(head)
        } else {
            let head = Arc::new(read_spec(
                source,
                "head.weight",
                DType::BF16,
                &[vocab, dim],
            )?);
            self.head_weight = Some(Arc::clone(&head));
            head
        };
        if head.len() != vocab * dim * 2 {
            return Err(err(
                "device head",
                format!(
                    "head.weight is {} bytes, expected {} BF16 bytes",
                    head.len(),
                    vocab * dim * 2
                ),
            ));
        }
        // The screen's per-block row maxima are a load-time precompute: one exact BF16 maximum per
        // [entry][40] block, 2.58 MB per rank. They are cached with the head weight because they are
        // a function of it, and the kernel's bound is then the same arithmetic the scan produced.
        let head_block_max = if let Some(maxima) = &self.head_block_max {
            Arc::clone(maxima)
        } else {
            let maxima = Arc::new(lm_head_ref::block_maxima_bf16(&head));
            self.head_block_max = Some(Arc::clone(&maxima));
            maxima
        };
        let hidden_bf16: Vec<u8> = hidden
            .iter()
            .flat_map(|value| bf16_bits(*value).to_le_bytes())
            .collect();
        let mut allocated = 0usize;
        let mut ptrs = Vec::new();
        let hidden_d = self
            .cuda
            .alloc(hidden_bf16.len(), &mut allocated)
            .map_err(drv)?;
        let weight_d = self.cuda.alloc(head.len(), &mut allocated).map_err(drv)?;
        let w_block_max_d = self
            .cuda
            .alloc(head_block_max.len(), &mut allocated)
            .map_err(drv)?;
        self.cuda.copy_htod(&hidden_d, &hidden_bf16).map_err(drv)?;
        self.cuda.copy_htod(&weight_d, &head).map_err(drv)?;
        self.cuda
            .copy_htod(&w_block_max_d, &head_block_max)
            .map_err(drv)?;
        let per_rank_out = per_rank * 4;
        let block_count = per_rank.div_ceil(cert_lm_head_launch::CHUNK as usize);
        let mut logits_d = Vec::with_capacity(RANKS);
        let mut certified_d = Vec::with_capacity(RANKS);
        for _ in 0..RANKS {
            logits_d.push(self.cuda.alloc(per_rank_out, &mut allocated).map_err(drv)?);
            certified_d.push(self.cuda.alloc(per_rank, &mut allocated).map_err(drv)?);
        }
        let count_d = self.cuda.alloc(4, &mut allocated).map_err(drv)?;
        let block_max_d = self
            .cuda
            .alloc(block_count * 4, &mut allocated)
            .map_err(drv)?;
        let block_argmax_d = self
            .cuda
            .alloc(block_count * 4, &mut allocated)
            .map_err(drv)?;
        let reduce_token_d = self.cuda.alloc(4, &mut allocated).map_err(drv)?;
        let reduce_value_d = self.cuda.alloc(4, &mut allocated).map_err(drv)?;
        let reduce_block_max_d = self
            .cuda
            .alloc(head_reduce::BLOCKS * 4, &mut allocated)
            .map_err(drv)?;
        let reduce_block_argmax_d = self
            .cuda
            .alloc(head_reduce::BLOCKS * 4, &mut allocated)
            .map_err(drv)?;
        let reduce_top5_value_d = self
            .cuda
            .alloc(head_reduce::TOP * 4, &mut allocated)
            .map_err(drv)?;
        let reduce_top5_index_d = self
            .cuda
            .alloc(head_reduce::TOP * 4, &mut allocated)
            .map_err(drv)?;
        let reduce_status_d = self.cuda.alloc(4, &mut allocated).map_err(drv)?;
        let reduce_count_d = self.cuda.alloc(4, &mut allocated).map_err(drv)?;
        ptrs.extend([
            hidden_d,
            weight_d,
            w_block_max_d,
            count_d,
            block_max_d,
            block_argmax_d,
        ]);
        ptrs.extend(logits_d.iter().copied());
        ptrs.extend(certified_d.iter().copied());
        ptrs.extend([
            reduce_token_d,
            reduce_value_d,
            reduce_block_max_d,
            reduce_block_argmax_d,
            reduce_top5_value_d,
            reduce_top5_index_d,
            reduce_status_d,
            reduce_count_d,
        ]);
        let rank_stride = per_rank * dim * 2;
        let rank_max_stride = per_rank * lm_head_ref::LM_BLOCKS * 2;
        let mut best_value = f32::NEG_INFINITY;
        let mut best_index = 0u32;
        let mut certified_total = 0usize;
        let mut certified_by_rank = [0usize; RANKS];
        let mut certified_flags = vec![0u8; vocab];
        let mut host_top: Vec<(f32, u32)> = Vec::new();
        let head_position = self.stats.head_calls;
        self.stats.head_calls = self.stats.head_calls.saturating_add(1);
        let head_staging_us = staging_started.elapsed().as_micros() as u64;
        self.stats.head_staging_us += head_staging_us;
        self.stats.head_staged_bytes += head.len() as u64;
        for rank in 0..RANKS {
            self.cuda.copy_htod(&count_d, &[0u8; 4]).map_err(drv)?;
            let params = CertLmHeadParams::new(
                hidden_d,
                Deviceptr(weight_d.0 + (rank * rank_stride) as u64),
                Deviceptr(w_block_max_d.0 + (rank * rank_max_stride) as u64),
                logits_d[rank],
                certified_d[rank],
                count_d,
                block_max_d,
                block_argmax_d,
                per_rank as u32,
                0,
                0,
                0,
            );
            cert_lm_head_launch::launch(&self.cuda, self.kernels.head, &params).map_err(drv)?;
            let mut logits_bytes = vec![0u8; per_rank_out];
            let mut certified = vec![0u8; per_rank];
            let mut count_bytes = [0u8; 4];
            let mut block_max = vec![0u8; block_count * 4];
            let mut block_argmax = vec![0u8; block_count * 4];
            self.cuda
                .copy_dtoh(&mut logits_bytes, &logits_d[rank])
                .map_err(drv)?;
            self.cuda
                .copy_dtoh(&mut certified, &certified_d[rank])
                .map_err(drv)?;
            self.cuda
                .copy_dtoh(&mut count_bytes, &count_d)
                .map_err(drv)?;
            self.cuda
                .copy_dtoh(&mut block_max, &block_max_d)
                .map_err(drv)?;
            self.cuda
                .copy_dtoh(&mut block_argmax, &block_argmax_d)
                .map_err(drv)?;
            let rank_certified = u32::from_le_bytes(count_bytes) as usize;
            certified_total += rank_certified;
            certified_by_rank[rank] = rank_certified;
            for entry in 0..per_rank {
                certified_flags[rank * per_rank + entry] = certified[entry];
                let offset = entry * 4;
                logits[rank * per_rank + entry] = f32::from_bits(u32::from_le_bytes([
                    logits_bytes[offset],
                    logits_bytes[offset + 1],
                    logits_bytes[offset + 2],
                    logits_bytes[offset + 3],
                ]));
            }
            for block in 0..block_count {
                let value = f32::from_bits(u32::from_le_bytes([
                    block_max[block * 4],
                    block_max[block * 4 + 1],
                    block_max[block * 4 + 2],
                    block_max[block * 4 + 3],
                ]));
                let relative = u32::from_le_bytes([
                    block_argmax[block * 4],
                    block_argmax[block * 4 + 1],
                    block_argmax[block * 4 + 2],
                    block_argmax[block * 4 + 3],
                ]) as usize;
                let global = head_global_index(rank, relative, per_rank);
                consider_head_block(&mut best_value, &mut best_index, value, global);
                consider_head_top(&mut host_top, value, global as u32);
            }
        }
        let reduce_params = LmHeadReduceParams::new(
            [logits_d[0], logits_d[1], logits_d[2], logits_d[3]],
            [
                certified_d[0],
                certified_d[1],
                certified_d[2],
                certified_d[3],
            ],
            reduce_token_d,
            reduce_value_d,
            reduce_block_max_d,
            reduce_block_argmax_d,
            reduce_top5_value_d,
            reduce_top5_index_d,
            reduce_status_d,
            reduce_count_d,
            0,
            0,
            0,
        );
        head_reduce::launch(&self.cuda, self.kernels.head_reduce, &reduce_params).map_err(drv)?;
        let mut reduce_token_bytes = [0u8; 4];
        let mut reduce_status_bytes = [0u8; 4];
        let mut reduce_value_bytes = [0u8; 4];
        let mut reduce_count_bytes = [0u8; 4];
        self.cuda
            .copy_dtoh(&mut reduce_token_bytes, &reduce_token_d)
            .map_err(drv)?;
        self.cuda
            .copy_dtoh(&mut reduce_status_bytes, &reduce_status_d)
            .map_err(drv)?;
        self.cuda
            .copy_dtoh(&mut reduce_value_bytes, &reduce_value_d)
            .map_err(drv)?;
        self.cuda
            .copy_dtoh(&mut reduce_count_bytes, &reduce_count_d)
            .map_err(drv)?;
        let reduce_token = u32::from_le_bytes(reduce_token_bytes);
        let reduce_status = u32::from_le_bytes(reduce_status_bytes);
        let reduce_value = f32::from_bits(u32::from_le_bytes(reduce_value_bytes));
        let reduce_certified_count = u32::from_le_bytes(reduce_count_bytes);
        let mut reduce_top5_values = [0f32; head_reduce::TOP];
        let mut reduce_top5_indices = [0u32; head_reduce::TOP];
        let mut reduce_top5_value_bytes = vec![0u8; head_reduce::TOP * 4];
        let mut reduce_top5_index_bytes = vec![0u8; head_reduce::TOP * 4];
        self.cuda
            .copy_dtoh(&mut reduce_top5_value_bytes, &reduce_top5_value_d)
            .map_err(drv)?;
        self.cuda
            .copy_dtoh(&mut reduce_top5_index_bytes, &reduce_top5_index_d)
            .map_err(drv)?;
        for slot in 0..head_reduce::TOP {
            reduce_top5_values[slot] = f32::from_bits(u32::from_le_bytes([
                reduce_top5_value_bytes[slot * 4],
                reduce_top5_value_bytes[slot * 4 + 1],
                reduce_top5_value_bytes[slot * 4 + 2],
                reduce_top5_value_bytes[slot * 4 + 3],
            ]));
            reduce_top5_indices[slot] = u32::from_le_bytes([
                reduce_top5_index_bytes[slot * 4],
                reduce_top5_index_bytes[slot * 4 + 1],
                reduce_top5_index_bytes[slot * 4 + 2],
                reduce_top5_index_bytes[slot * 4 + 3],
            ]);
        }
        let host_text: Vec<String> = host_top
            .iter()
            .map(|(value, index)| format!("{index}:{value:.9}"))
            .collect();
        let device_text: Vec<String> = reduce_top5_indices
            .iter()
            .zip(reduce_top5_values.iter())
            .map(|(index, value)| format!("{index}:{value:.9}"))
            .collect();
        println!(
            "head A/B: position={head_position} host_top1={best_index} host_top5=[{}] device_top1={reduce_token} device_top5=[{}] block_max_equal={} host_value={best_value:.9} device_value={reduce_value:.9}",
            host_text.join(", "),
            device_text.join(", "),
            best_value.to_bits() == reduce_value.to_bits()
        );
        let force_host = std::env::var("SPARKFLASH_HEAD_HOST_ASSEMBLY")
            .is_ok_and(|value| value == "1" || value == "true");
        if reduce_status & !0x7 != 0 {
            return Err(err(
                "device head reduce",
                format!("unknown status bits in {reduce_status:#x}"),
            ));
        }
        let reduce_winner_certified = reduce_winner_certified(reduce_status);
        if !reduce_winner_certified && reduce_token != best_index {
            return Err(err(
                "device head reduce",
                format!(
                    "the reduce token {reduce_token} differs from the host block assembly token {best_index}, status={reduce_status:#x}"
                ),
            ));
        }
        if force_host && !reduce_winner_certified {
            let head_device_us = started.elapsed().as_micros() as u64 - head_staging_us;
            self.stats.head_launches += RANKS + 1;
            self.stats.head_certified += certified_total;
            self.stats.head_certified_by_rank = certified_by_rank;
            self.head_certified = certified_flags;
            self.stats.head_winner_certified = false;
            self.stats.head_reduce_used = false;
            self.stats.head_reduce_token = reduce_token;
            self.stats.head_reduce_value = reduce_value;
            self.stats.head_device_us += head_device_us;
            self.stats.device_us += started.elapsed().as_micros() as u64;
            self.cuda.free_buffers(&ptrs.iter().collect::<Vec<_>>());
            println!(
                "head A/B: position={head_position} returned host assembly token={best_index} (SPARKFLASH_HEAD_HOST_ASSEMBLY)"
            );
            return Ok(Some(best_index));
        }
        self.stats.head_reduce_token = reduce_token;
        self.stats.head_reduce_value = reduce_value;
        self.stats.head_reduce_status = reduce_status;
        self.stats.head_reduce_certified_count = reduce_certified_count;
        let head_device_end_us = started.elapsed().as_micros() as u64;
        if head_action(reduce_winner_certified, self.diagnose_layer0) == HeadAction::Refuse {
            let head_device_us = head_device_end_us - head_staging_us;
            self.stats.head_launches += RANKS + 1;
            self.stats.head_certified += certified_total;
            self.stats.head_certified_by_rank = certified_by_rank;
            self.head_certified = certified_flags;
            self.stats.head_winner_certified = true;
            self.stats.head_reduce_used = false;
            self.stats.head_device_us += head_device_us;
            self.stats.device_us += started.elapsed().as_micros() as u64;
            self.cuda.free_buffers(&ptrs.iter().collect::<Vec<_>>());
            println!(
                "lm head refusal: reduce status={reduce_status:#x} token={reduce_token} is certified; falling back to host block assembly token={best_index} certified_by_rank={certified_by_rank:?} certified_total={certified_total} head_staging_us={head_staging_us} head_device_us={head_device_us}"
            );
            return Ok(None);
        }
        if head_action(reduce_winner_certified, self.diagnose_layer0) == HeadAction::Diagnose {
            let hidden_values = decode_bf16(&hidden_bf16);
            let mut twin_logits = vec![0.0f32; vocab];
            let mut twin_best = f32::NEG_INFINITY;
            let mut twin_token = 0usize;
            for entry in 0..vocab {
                let row = &head[entry * dim * 2..(entry + 1) * dim * 2];
                let mut acc = 0.0f32;
                for (index, hidden) in hidden_values.iter().enumerate() {
                    let weight = router_ref::bf16_to_f32(u16::from_le_bytes([
                        row[index * 2],
                        row[index * 2 + 1],
                    ]));
                    acc += hidden * weight;
                }
                twin_logits[entry] = acc;
                if acc > twin_best {
                    twin_best = acc;
                    twin_token = entry;
                }
            }
            let mut logit_mismatches = 0usize;
            let mut max_abs = 0.0f32;
            if certified_total == 0 {
                for (device, twin) in logits.iter().zip(&twin_logits) {
                    if device.to_bits() != twin.to_bits() {
                        logit_mismatches += 1;
                    }
                    max_abs = (device - twin).abs().max(max_abs);
                }
            }
            println!(
                "lm head diag: certified={certified_total} bf16_twin_logit_mismatches={logit_mismatches} bf16_twin_max_abs={max_abs:e} device_token={best_index} twin_token={twin_token} token_match={}",
                best_index as usize == twin_token
            );
            if certified_total == 0 && logit_mismatches != 0 {
                return Err(err(
                    "device head diagnose",
                    format!(
                        "the device logits differ from the same-bf16-byte host twin in {logit_mismatches} entries"
                    ),
                ));
            }
            if best_index as usize != twin_token {
                return Err(err(
                    "device head diagnose",
                    format!(
                        "the device token {best_index} differs from the same-bf16-byte host twin token {twin_token}"
                    ),
                ));
            }
        }
        self.cuda.free_buffers(&ptrs.iter().collect::<Vec<_>>());
        self.stats.head_launches += RANKS + 1;
        self.stats.head_certified += certified_total;
        self.stats.head_certified_by_rank = certified_by_rank;
        self.head_certified = certified_flags;
        self.stats.head_winner_certified = false;
        self.stats.head_reduce_used = true;
        let head_device_us = head_device_end_us - head_staging_us;
        self.stats.head_device_us += head_device_us;
        self.stats.device_us += started.elapsed().as_micros() as u64;
        Ok(Some(reduce_token))
    }

    fn head_certified_mask(&self) -> Option<&[u8]> {
        Some(&self.head_certified)
    }
}

struct Expert {
    w1: Vec<u8>,
    s1: Vec<u8>,
    w3: Vec<u8>,
    s3: Vec<u8>,
    w2: Vec<u8>,
    s2: Vec<u8>,
}

impl Expert {
    fn bytes(&self) -> usize {
        self.w1.len()
            + self.s1.len()
            + self.w3.len()
            + self.s3.len()
            + self.w2.len()
            + self.s2.len()
    }
}

fn read_experts(
    source: &dyn WeightSource,
    prefix: &str,
    table: &[(usize, usize)],
    dim: usize,
    inter: usize,
) -> Result<Vec<Expert>, EngineError> {
    let mut experts = Vec::with_capacity(table.len());
    for (_, expert) in table {
        let base = format!("{prefix}.ffn.experts.{expert}");
        let w1 = read_len(source, &format!("{base}.w1.weight"), inter * (dim / 2))?;
        let s1 = read_len(
            source,
            &format!("{base}.w1.scale"),
            inter * (dim / QUANT_BLOCK),
        )?;
        let w3 = read_len(source, &format!("{base}.w3.weight"), inter * (dim / 2))?;
        let s3 = read_len(
            source,
            &format!("{base}.w3.scale"),
            inter * (dim / QUANT_BLOCK),
        )?;
        let w2 = read_len(source, &format!("{base}.w2.weight"), dim * (inter / 2))?;
        let s2 = read_len(
            source,
            &format!("{base}.w2.scale"),
            dim * (inter / QUANT_BLOCK),
        )?;
        experts.push(Expert {
            w1,
            s1,
            w3,
            s3,
            w2,
            s2,
        });
    }
    Ok(experts)
}

/// Read a tensor after checking the checkpoint's declared dtype and logical shape.
///
/// The byte length alone cannot tell an FP8 weight from a BF16 twin of half the elements; the
/// m17b harness asserts the declared shape and format before it trusts a staged operand, and this
/// is the same check on the chain path.
fn read_spec(
    source: &dyn WeightSource,
    name: &str,
    dtype: DType,
    shape: &[usize],
) -> Result<Vec<u8>, EngineError> {
    let spec = source.spec(name).map_err(|error| read_err(name, error))?;
    if spec.dtype != dtype || spec.shape != shape {
        return Err(err(
            name,
            format!(
                "declared {:?} {:?}, expected {:?} {:?}",
                spec.dtype, spec.shape, dtype, shape
            ),
        ));
    }
    let elements = shape.iter().product::<usize>();
    read_len(source, name, dtype.storage_bytes(elements))
}

fn read_len(
    source: &dyn WeightSource,
    name: &str,
    expected: usize,
) -> Result<Vec<u8>, EngineError> {
    let bytes = source.read(name).map_err(|error| read_err(name, error))?;
    if bytes.len() != expected {
        return Err(err(
            name,
            format!("expected {expected} bytes, got {}", bytes.len()),
        ));
    }
    Ok(bytes)
}

#[allow(clippy::type_complexity)]
fn local_table(
    ids: &[u32],
    n_routed: usize,
) -> Result<(Vec<(usize, usize)>, Vec<u32>), EngineError> {
    let mut map = vec![usize::MAX; n_routed];
    let mut next = 0usize;
    for id in ids {
        let id = *id as usize;
        if id >= n_routed {
            return Err(err(
                "device router ids",
                format!("id {id} is outside {n_routed} experts"),
            ));
        }
        if map[id] == usize::MAX {
            map[id] = next;
            next += 1;
        }
    }
    let mut table: Vec<(usize, usize)> = map
        .iter()
        .enumerate()
        .filter(|(_, local)| **local != usize::MAX)
        .map(|(expert, local)| (*local, expert))
        .collect();
    table.sort_unstable();
    let local_ids: Vec<u32> = ids.iter().map(|id| map[*id as usize] as u32).collect();
    Ok((table, local_ids))
}

fn rank_slice(
    experts: &[Expert],
    pick: impl Fn(&Expert) -> &[u8],
    rank: usize,
    take: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(experts.len() * take);
    for expert in experts {
        let tensor = pick(expert);
        let start = rank * take;
        out.extend_from_slice(&tensor[start..start + take]);
    }
    out
}

/// The weighted-SwiGLU kernel's raw 11-word argument list, in the kernel's declaration order:
/// gate, up, router weights, codes, scales, rows, cols (always the padded 640), limit bits,
/// trace ring, layer, step. A unit test pins this order; a swap is a window-discovery otherwise.
#[allow(clippy::too_many_arguments)]
fn weighted_words(
    gate: Deviceptr,
    up: Deviceptr,
    weights: Deviceptr,
    codes: Deviceptr,
    scales: Deviceptr,
    rows: usize,
    layer: usize,
    step: usize,
) -> [u64; 11] {
    [
        gate.0,
        up.0,
        weights.0,
        codes.0,
        scales.0,
        rows as u64,
        PADDED as u64,
        SWIGLU_LIMIT_BITS,
        0,
        layer as u64,
        step as u64,
    ]
}

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn rank_slice_padded(
    experts: &[Expert],
    pick: impl Fn(&Expert) -> &[u8],
    rows: usize,
    row_bytes: usize,
    start: usize,
    take: usize,
    padded: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; experts.len() * rows * padded];
    for (expert_index, expert) in experts.iter().enumerate() {
        let tensor = pick(expert);
        let base = expert_index * rows * padded;
        for row in 0..rows {
            let source = row * row_bytes + start;
            let target = base + row * padded;
            out[target..target + take].copy_from_slice(&tensor[source..source + take]);
        }
    }
    out
}

fn quantise_activation(x: &[f32], what: &str) -> Result<(Vec<u8>, Vec<u8>), EngineError> {
    let q = quantize_activation_fp8(x).map_err(|error| err(what, error.to_string()))?;
    Ok((q.codes().to_vec(), q.scales().to_vec()))
}

/// Fold one per-rank block maximum into the assembled head argmax.
///
/// Ties break to the lower GLOBAL vocabulary index, the kernel's declared order (not the order the
/// four rank shards happen to be visited in).
/// Map a rank-local head entry to the global vocabulary index. Rank shards are contiguous, so
/// rank r owns `[r * per_rank, (r + 1) * per_rank)`.
fn head_global_index(rank: usize, relative: usize, per_rank: usize) -> usize {
    rank * per_rank + relative
}

fn consider_head_top(top: &mut Vec<(f32, u32)>, value: f32, index: u32) {
    if top.len() < 5 || value > top.last().copied().unwrap_or((f32::NEG_INFINITY, 0)).0 {
        top.push((value, index));
        top.sort_by(|left, right| right.0.total_cmp(&left.0));
        top.truncate(5);
    }
}

fn consider_head_block(best_value: &mut f32, best_index: &mut u32, value: f32, global: usize) {
    if value > *best_value || (value == *best_value && global < *best_index as usize) {
        *best_value = value;
        *best_index = global as u32;
    }
}

/// Whether the device reduce's winning token is certified by the tournament, in which case the
/// host block assembly is not trusted and the device answer must be refused (host fallback).
fn reduce_winner_certified(status: u32) -> bool {
    status & STATUS_WINNER_CERTIFIED != 0
}

/// What the head does after the reduce readback. The certified refusal is unconditional and comes
/// before the layer-0 diagnostic, so a certified winner can never be served (or diagnosed) instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadAction {
    Refuse,
    Diagnose,
    Serve,
}

fn head_action(winner_certified: bool, diagnose_layer0: bool) -> HeadAction {
    if winner_certified {
        HeadAction::Refuse
    } else if diagnose_layer0 {
        HeadAction::Diagnose
    } else {
        HeadAction::Serve
    }
}

/// Repack one checkpoint FP8 weight into the B12x micro-tile order the dense kernel walks.
///
/// A value-preserving index permutation, no arithmetic (the m12/M7-stage-7 declaration).
fn pack_b12x(weights: &[u8], cols: usize, inner: usize) -> Vec<u8> {
    assert_eq!(
        weights.len(),
        cols * inner,
        "the checkpoint stores one E4M3 byte per FP8 value"
    );
    let mut tiled = vec![0u8; cols * inner];
    for col in 0..cols {
        for k in 0..inner {
            tiled[mxfp8_ref::b12x_weight_index(col, k, inner)] = weights[col * inner + k];
        }
    }
    tiled
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
        .collect()
}

/// The routed-MoE totals line. It is a free function so a unit test can pin that the routed-only
/// counters are printed rather than the cumulative ones.
pub fn routed_totals_line(stats: &MoeDeviceStats) -> String {
    format!(
        "routed MoE totals: layers={} combine_arith=native_f32_slot_acc router_launches={} gate_up_launches={} weighted_launches={} combine_launches={} moe_launches={} staging_ms={} device_us={} staged_expert_bytes={} host_bytes={} layer0_host_max_abs={:.3e} layer0_host_rel={:.3e} layer0_host_max={:.3e} layer0_bf16_twin_mismatches={} layer0_bf16_twin_max_abs={:.3e}",
        stats.layers,
        stats.layers * ROUTED_ROUTER_LAUNCHES,
        stats.layers * ROUTED_GATE_UP_LAUNCHES,
        stats.layers * ROUTED_WEIGHTED_LAUNCHES,
        stats.layers * ROUTED_COMBINE_LAUNCHES,
        stats.routed_launches,
        stats.routed_staging_us / 1000,
        stats.routed_device_us,
        stats.staged_bytes,
        stats.routed_host_bytes,
        stats.layer0_host_max_abs,
        stats.layer0_host_rel,
        stats.layer0_host_magnitude,
        stats.layer0_bf16_twin_mismatches,
        stats.layer0_bf16_twin_max_abs
    )
}

/// The shared-expert totals line, with the shared-only counters and bytes.
pub fn shared_totals_line(stats: &MoeDeviceStats) -> String {
    format!(
        "shared expert totals: layers={} shared_launches={} staging_ms={} device_us={} total_us={} staged_shared_bytes={} host_bytes={} layer0_shared_bf16_mismatches={} layer0_shared_host_max_abs={:.3e} layer0_shared_host_rel={:.3e} layer0_shared_host_max={:.3e}",
        stats.layers,
        stats.shared_launches,
        stats.shared_staging_us / 1000,
        stats.shared_device_us,
        stats.shared_us,
        stats.shared_bytes,
        stats.shared_host_bytes,
        stats.layer0_shared_bf16_mismatches,
        stats.layer0_shared_host_max_abs,
        stats.layer0_shared_host_rel,
        stats.layer0_shared_host_magnitude
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weighted_launch_argument_order_is_pinned() {
        let words = weighted_words(
            Deviceptr(1),
            Deviceptr(2),
            Deviceptr(3),
            Deviceptr(4),
            Deviceptr(5),
            6,
            7,
            8,
        );
        assert_eq!(words[0], 1, "gate is the first kernel argument");
        assert_eq!(words[1], 2, "up is the second kernel argument");
        assert_eq!(words[2], 3, "the router weights are the third argument");
        assert_eq!(words[3], 4, "the quantised codes follow the weights");
        assert_eq!(words[4], 5, "the scales follow the codes");
        assert_eq!(words[5], 6, "rows is the sixth argument");
        assert_eq!(words[6], 640, "cols is always the padded 640, never 576");
        assert_eq!(
            words[7],
            u64::from(10.0f32.to_bits()),
            "limit bits are f32 10.0"
        );
        assert_eq!(words[8], 0, "trace ring is zero when not tracing");
        assert_eq!(words[9], 7, "layer is the tenth argument");
        assert_eq!(words[10], 8, "step is the last argument");
    }

    #[derive(Default)]
    struct GridSource {
        specs: std::collections::BTreeMap<String, sparkflash_engine::tensor::TensorSpec>,
        bytes: std::collections::BTreeMap<String, Vec<u8>>,
    }

    impl WeightSource for GridSource {
        fn names(&self) -> Vec<String> {
            self.specs.keys().cloned().collect()
        }

        fn spec(
            &self,
            name: &str,
        ) -> Result<sparkflash_engine::tensor::TensorSpec, sparkflash_engine::weights::WeightError>
        {
            self.specs.get(name).cloned().ok_or_else(|| {
                sparkflash_engine::weights::WeightError::Missing {
                    name: name.to_string(),
                }
            })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, sparkflash_engine::weights::WeightError> {
            self.bytes.get(name).cloned().ok_or_else(|| {
                sparkflash_engine::weights::WeightError::Missing {
                    name: name.to_string(),
                }
            })
        }
    }

    #[test]
    fn the_shared_scale_reads_pin_the_32x32_grid_shapes() {
        let (inter, dim) = (2304usize, 5120usize);
        assert_eq!(shared_scale_shapes(inter, dim), ([72, 160], [160, 72]));
        let mut source = GridSource::default();
        for (name, shape) in [
            ("layers.0.ffn.shared_experts.w1.scale", [72usize, 160]),
            ("layers.0.ffn.shared_experts.w3.scale", [72usize, 160]),
            ("layers.0.ffn.shared_experts.w2.scale", [160usize, 72]),
        ] {
            source.specs.insert(
                name.to_string(),
                sparkflash_engine::tensor::TensorSpec {
                    name: name.to_string(),
                    dtype: DType::F8E8M0,
                    shape: shape.to_vec(),
                },
            );
            source
                .bytes
                .insert(name.to_string(), vec![0u8; shape[0] * shape[1]]);
        }
        let (s1, s3, s2) =
            read_shared_scale_grids(&source, "layers.0.ffn.shared_experts", inter, dim)
                .expect("literal grids");
        assert_eq!(
            (s1.len(), s3.len(), s2.len()),
            (72 * 160, 72 * 160, 160 * 72)
        );
        // Swapping the w1 gate/up grid for the w2 down grid must fail, not pass because the test
        // derived its expectation from the shape helper.
        source.specs.insert(
            "layers.0.ffn.shared_experts.w1.scale".to_string(),
            sparkflash_engine::tensor::TensorSpec {
                name: "layers.0.ffn.shared_experts.w1.scale".to_string(),
                dtype: DType::F8E8M0,
                shape: vec![160, 72],
            },
        );
        let error = read_shared_scale_grids(&source, "layers.0.ffn.shared_experts", inter, dim)
            .expect_err("a swapped w1/w2 grid must be refused");
        assert!(error.to_string().contains("declared"), "{error}");
    }

    #[test]
    fn routed_and_shared_totals_print_their_own_counters() {
        let mut stats = MoeDeviceStats {
            layers: 2,
            ..Default::default()
        };
        stats.routed_launches = 34;
        stats.routed_staging_us = 111_000;
        stats.routed_device_us = 222_000;
        stats.routed_host_bytes = 333_000;
        stats.shared_launches = 8;
        stats.shared_staging_us = 444_000;
        stats.shared_device_us = 555_000;
        stats.shared_host_bytes = 666_000;
        stats.staging_us = 999_000;
        stats.device_us = 888_000;
        stats.host_bytes = 777_000;
        let routed = routed_totals_line(&stats);
        assert!(routed.contains("staging_ms=111"), "{routed}");
        assert!(routed.contains("device_us=222000"), "{routed}");
        assert!(routed.contains("host_bytes=333000"), "{routed}");
        assert!(!routed.contains("999"), "{routed}");
        let shared = shared_totals_line(&stats);
        assert!(shared.contains("staging_ms=444"), "{shared}");
        assert!(shared.contains("device_us=555000"), "{shared}");
        assert!(shared.contains("host_bytes=666000"), "{shared}");
        assert!(!shared.contains("999"), "{shared}");
    }

    /// The routed path measures its launches and checks the measurement against this plan before it
    /// records the accounting, so the plan has to be pinned: one router, and per rank a gate GEMM, an
    /// up GEMM, the weighted SwiGLU quantiser and the combine add. 17 is the count the accounting and
    /// the log carry; the native f32 slot accumulator adds no launch, and a per-rank readback turned
    /// into a launch would be 21 and fail the check inside the layer.
    #[test]
    fn the_routed_launch_plan_counts_seventeen_kernels_per_layer() {
        assert_eq!(
            RANKS, 4,
            "the c4c routed path is four tensor-parallel ranks"
        );
        assert_eq!(ROUTED_ROUTER_LAUNCHES, 1);
        assert_eq!(
            ROUTED_GATE_UP_LAUNCHES,
            RANKS * 2,
            "a gate and an up GEMM per rank"
        );
        assert_eq!(ROUTED_WEIGHTED_LAUNCHES, RANKS);
        assert_eq!(ROUTED_COMBINE_LAUNCHES, RANKS);
        assert_eq!(ROUTED_LAUNCHES_PER_LAYER, 17);
        assert_eq!(RANKS * 4 + 1, ROUTED_LAUNCHES_PER_LAYER);
        let stats = MoeDeviceStats {
            layers: 1,
            routed_launches: ROUTED_LAUNCHES_PER_LAYER,
            ..Default::default()
        };
        let line = routed_totals_line(&stats);
        // The arithmetic is named in the log, so a window log says which combine rule ran rather
        // than leaving it implicit. The chain has no legacy arm: native is the only and default path.
        assert!(line.contains("combine_arith=native_f32_slot_acc"), "{line}");
        assert!(
            line.contains(&format!("moe_launches={ROUTED_LAUNCHES_PER_LAYER}")),
            "{line}"
        );
    }

    #[test]
    fn a_certified_winner_is_refused_before_the_diagnose_branch() {
        assert_eq!(head_action(true, true), HeadAction::Refuse);
        assert_eq!(head_action(true, false), HeadAction::Refuse);
        assert_eq!(head_action(false, true), HeadAction::Diagnose);
        assert_eq!(head_action(false, false), HeadAction::Serve);
    }

    #[test]
    fn the_shared_expert_operand_sizes_are_the_fp8_and_32x32_grid() {
        let (inter, dim) = (2304usize, 5120usize);
        assert_eq!(inter * dim, 11_796_480, "w1/w3 are one E4M3 byte per value");
        assert_eq!(dim * inter, 11_796_480, "w2 is one E4M3 byte per value");
        assert_eq!(
            mxfp8_ref::b12x_scale_bytes(inter, dim),
            72 * 160,
            "w1/w3 scales are the [2304/32][5120/32] grid"
        );
        assert_eq!(
            mxfp8_ref::b12x_scale_bytes(dim, inter),
            160 * 72,
            "w2 scales are the [5120/32][2304/32] grid"
        );
    }

    #[test]
    fn the_certified_winner_is_refused_and_the_twin_check_does_not_run_for_it() {
        assert!(reduce_winner_certified(STATUS_WINNER_CERTIFIED));
        assert!(!reduce_winner_certified(head_reduce::STATUS_TOP5_CLAIMED));
        assert!(!reduce_winner_certified(head_reduce::STATUS_ANY_CERTIFIED));
        assert!(!reduce_winner_certified(0));
    }

    #[test]
    fn the_head_rank_to_global_mapping_is_contiguous_and_pinned() {
        let per_rank = 32_320usize;
        let mut best_value = f32::NEG_INFINITY;
        let mut best_index = 0u32;
        for rank in 0..4 {
            let relative = 7usize;
            let global = head_global_index(rank, relative, per_rank);
            assert_eq!(global, rank * per_rank + relative);
            consider_head_block(&mut best_value, &mut best_index, 10.0 + rank as f32, global);
        }
        assert_eq!(best_index as usize, 3 * per_rank + 7, "rank 3's shard wins");
        assert_eq!(head_global_index(0, 0, per_rank), 0);
        assert_eq!(
            head_global_index(3, per_rank - 1, per_rank),
            4 * per_rank - 1
        );
    }

    #[test]
    fn the_head_argmax_breaks_ties_towards_the_lower_global_index() {
        let mut value = 0.0f32;
        let mut index = 0u32;
        consider_head_block(&mut value, &mut index, 1.0, 7);
        consider_head_block(&mut value, &mut index, 1.0, 3);
        assert_eq!(index, 3, "the later tie must win only if it is lower");
        consider_head_block(&mut value, &mut index, 2.0, 9);
        assert_eq!(index, 9, "a larger value always wins");
    }

    #[test]
    fn the_b12x_repack_is_a_bijection_of_every_weight_byte() {
        let (cols, inner) = (64, 64);
        let weights: Vec<u8> = (0..cols * inner).map(|value| value as u8).collect();
        let tiled = pack_b12x(&weights, cols, inner);
        assert_eq!(tiled.len(), cols * inner);
        for col in 0..cols {
            for k in 0..inner {
                let at = mxfp8_ref::b12x_weight_index(col, k, inner);
                assert_eq!(
                    tiled[at],
                    weights[col * inner + k],
                    "column {col} value {k} must land at its B12x index {at}"
                );
            }
        }
        let mut seen = vec![false; cols * inner];
        for col in 0..cols {
            for k in 0..inner {
                let at = mxfp8_ref::b12x_weight_index(col, k, inner);
                assert!(!seen[at], "index {at} visited twice");
                seen[at] = true;
            }
        }
        assert!(seen.iter().all(|seen| *seen));
    }

    #[test]
    fn the_certified_head_shard_is_the_released_vocabulary_over_four() {
        assert_eq!(HEAD_VOCAB_PER_RANK, 129_280 / 4);
        assert_eq!(cert_lm_head_launch::CHUNK, 1024);
        assert_eq!(
            HEAD_VOCAB_PER_RANK.div_ceil(cert_lm_head_launch::CHUNK as usize),
            32
        );
    }

    #[test]
    fn the_local_expert_table_maps_global_ids_by_first_appearance() {
        let (table, local) = local_table(&[5, 7, 5], 8).expect("table");
        assert_eq!(table, vec![(0, 5), (1, 7)]);
        assert_eq!(local, vec![0, 1, 0]);
    }

    #[test]
    fn a_rank_slice_of_w2_pads_each_row_to_the_640_column_contract() {
        let expert = Expert {
            w1: Vec::new(),
            s1: Vec::new(),
            w3: Vec::new(),
            s3: Vec::new(),
            w2: (0u8..8).collect(),
            s2: Vec::new(),
        };
        let sliced = rank_slice_padded(&[expert], |expert| &expert.w2, 2, 4, 1, 2, 3);
        assert_eq!(
            sliced,
            vec![1, 2, 0, 5, 6, 0],
            "columns 0..2 and a zero pad column"
        );
    }
}

fn verified(dir: &Path, name: &str) -> Result<Vec<u8>, String> {
    let pin = sparkflash_kernels::baseline::pin(name)?
        .ok_or_else(|| format!("kernel {name} is not pinned in the baseline"))?;
    let path = sparkflash_device::cubins::cubin_path(dir, name);
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    sparkflash_device::cubins::verify_bytes(&bytes, &pin)?;
    Ok(bytes)
}

fn err(what: &str, detail: String) -> EngineError {
    EngineError::Shape {
        what: what.to_owned(),
        detail,
    }
}

fn read_err(name: &str, error: sparkflash_engine::weights::WeightError) -> EngineError {
    err(name, error.to_string())
}

fn drv(error: String) -> EngineError {
    err("device routed MoE", error)
}
