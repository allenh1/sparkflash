#![forbid(unsafe_code)]

//! M7: layer 0 of DeepSeek-V4.1-Flash on the GB10, one stage per run.
//!
//! The layer is the unit of proof: a stage that matches at a boundary localises a failure to that
//! boundary, which one final tensor cannot. Every run is ONE stage (--stage 0 through 8). The argv,
//! the device bytes and the comparison class are printed before any numbers, and MemAvailable is
//! printed and re-checked before and after every launch, with the process refusing to continue
//! under the memory floor (32 GiB by default) - and refusing an allocation that would take the
//! node below it.
//!
//! THE RAM RULE. The CPU and the GPU share one pool on the node, and an over-allocation there
//! locks the machine until someone can reach it, so the harness never holds what it does not need:
//!
//!   * expert weights are read as packed bytes plus scale bytes and handed to the reference in
//!     exactly those bytes. No expert is ever decoded to f32 on the host, and the reference
//!     signatures (mxfp8_ref::dense_gemm, mxfp8_ref::grouped_wo_a) are what makes that structural
//!     rather than a rule someone could forget;
//!   * stage 3 runs a bounded expert subset (--experts 8, 32 or 64), never all 384: the full
//!     f32 reference is about 54 GB and is a second-order experiment, not a first move;
//!   * embed.weight is read one row at a time, because the whole tensor is 1.3 GB and this harness
//!     needs one token's row.
//!
//! Stage 0 (CPU only, no CUDA context, runnable while vLLM holds the node): the fold-closure
//! cross-check. It opens the released `hc_attn_fn` / `hc_ffn_fn` [24, 4*d] matrices, permutes
//! their rows into the kernel's B/A/C order with NO averaging, runs the released H() arithmetic
//! (full flattened projection, projection rstd, `mix * scale + base`, sigmoid pre/post, released
//! split-Sinkhorn) through mega_mhc_ref::mega_mhc_layer_with_scale, and diffs the resulting
//! coefficient tensors against the ones the ENGINE captured in the host boundary row 0
//! (`hc_attn_pre/post/comb`, `hc_ffn_pre/post/comb` in the v1 manifest/blob; the attn streams come
//! from the row's `residual_in`, the ffn streams from its `post_attention_hidden`). Its
//! non-vacuity half runs the same reference over the RETIRED fold (predictor rows averaged over the
//! four copies, projection over the stream mean) and prints how far that lands from the same row-0
//! tensors, which is the fold gap measured on the captured arithmetic instead of on a synthetic
//! fixture. Since D8 it also re-derives the row's activation-quantised block inputs from their own
//! captured f32 values with m4_quant::act_quant_fp8_row - the reference's activation ceil rule, the
//! rule the engine wrote pre_attention_xhat/scales and pre_ffn_xhat/scales with - and reports how
//! many E4M3 codes and ue8m0 scales it reproduces exactly; the retired floor-log2 weight-grid rule is
//! printed on the same bytes, unjudged, as the size of the gap. Pass bar: every full-projection
//! coefficient within 1e-5 absolute of the engine's captured value (both sides are f32 sequential
//! reductions, so the expected delta is a few ULP) and every recomputed activation quantiser code
//! and scale identical to the captured one; the folded comparison is printed as evidence and is not
//! judged.
//!
//! Stage 1 (small, and it gates the rest): mega_mhc_fused_bf16_sm121 against
//! mega_mhc_ref::mega_mhc_layer_with_scale. The kernel takes the released `hc_attn_base` [24] and
//! `hc_attn_scale` [3] itself, and both sides run the released arithmetic end to end: the full
//! released [24, 4*d] projection over the flattened stream (B0 removed the old d-wide fold from BOTH
//! sides), the projection rstd, `mix * scale + base`, the sigmoid pre/post and the released
//! split-Sinkhorn. The reference returns the coefficients, the FP8 codes and the ue8m0 scales as well
//! as the f32 streams, so the comparison is two-sided, and the coefficients are compared RAW (the
//! old harness-side softmax around C is gone because the kernel and the reference now agree on C's
//! released activation). The kernel's B0 `xhat_normed` output is compared with
//! mega_mhc_ref::block_input_normed on the same device inputs, so the f32 block input is checked as
//! a captured value rather than reconstructed from codes times scale. Nothing in stage 1 is larger
//! than a megabyte. With the fold closed, stage 1 is a comparison of the released projection
//! arithmetic (modulo the kernel's per-tile reduction order), not of a folded approximation; the
//! pre-B0 stage-1 runs measured folded semantics and those numbers are retired.
//!
//! Stage 2: mla_ref::attention_with_sink against flash_mla_csa2_sm121, the checkpoint's per-head
//! attn_sink consumed on BOTH sides: the kernel takes the [64] f32 sink as an argument and adds
//! it to the softmax denominator after the maximum, exactly as the reference does, so this is
//! internal consistency of the sink arithmetic, not (yet) a device-vs-release claim - the device
//! run of this ABI change is the proof. Its input arrives from stage 1, so it shares stage 1's
//! chain: with B0 the mHC predictor is the released flattened projection on both sides, so no
//! predictor-fold limitation is inherited any more. Stage 2 feeds the KV rows in one of two
//! formats, selected explicitly by --stage2-kv-format (default fp4, the byte-identical historical
//! path):
//!
//!   * fp4: the FP4 main-KV format the kernel and mla_ref::kv_latent both consume
//!     (per-16-channel E4M3 scale of amax/6, E2M1 codes, low nibble first), quantised by
//!     sparkflash-kv's quantize_main_kv rule from kv_norm(wkv(xhat)) with the rope tail rotated.
//!     This is the merged Stage-2 column, and a --tokens 130 run re-asserts its historical totals.
//!   * engine: the engine's released window format (model.py:700-707, kernel.py:98-124) - bf16
//!     rows quantised to E4M3 codes with one E8M0 power-of-two scale per 32 channels, with the
//!     engine's in-place bf16 storage round trip reproduced on both sides. This is a separate
//!     format-fidelity class, not a bit-identity claim against the released kernel, and it does
//!     not reproduce the engine's bf16 arithmetic: sparkflash-engine's reference forward keeps
//!     f32 dequant and that seam is named in the docs.
//!
//! The candidate list is supplied by the harness (the top-512 merge over the HSI pool is a separate
//! pipeline step), and the kernel's indexer outputs are not compared. Run it with --tokens 130: a
//! prompt at or under the 128-token window never produces an out-of-window candidate, and the
//! exclusion branch is part of what the stage must exercise; in engine mode the harness also
//! injects ids at and past the supplied row count and asserts their contribution is exactly zero.
//! The stage-2 output prints every one of these labels before its numbers.
//!
//! Stage 3: the MoE, bounded, with the router on the host and labelled as a host seam. The router
//! semantics reproduced are the engine reference's: scores from gate.weight divided by gate_temp,
//! then sqrtsoftplus; the noaux bias is added only for selection, and the weights come from the
//! unbiased scores. gate.bias_vl is the image-span bias and is not read on the text path.
//!
//! Stage 4: determinism, not accuracy. It launches the Mega-mHC kernel N times on the SAME input
//! bytes and requires every output byte of every run to match run 0: hc_stats (the final 26 and the
//! per-tile partials), hc_coeffs, hc_rms_out, the E4M3 codes, the ue8m0 scales and the residual
//! streams X_l. There is no tolerance and no reference here - a single differing byte fails the
//! stage. This is the regression test for the fp32 atomicAdd grid reduction the kernel used to
//! carry: the four block partials are now written to slots fixed by blockIdx and combined in tile
//! order in binary64, so the arrival order of the blocks cannot change a bit.
//!
//! Stage 5: the certified LM head (cert_lm_head_sm121) against a host reference on one token's
//! hidden state. The input and reference logits are the pair boundary_dump writes with
//! --head-input-out and --logits-out; the harness converts the head input to the kernel's bf16 ABI,
//! launches one rank per 32,320-row shard of head.weight, and recomputes the reference in the
//! kernel's own bf16 order so the comparison is the head's arithmetic rather than the hidden
//! vector's rounding. It checks the argmax against both the f32-hidden reference and the matched
//! reference, the exact entries against the matched reference, and that every certified bound is
//! still an upper bound. The kernel's contract is the argmax, not the top-5 (a certified entry
//! stores a bound), so no top-5 claim is made here.
//!
//! Stage 6: the CPU half of the criterion-3 head comparison, runnable while vLLM holds the node. It
//! never opens a CUDA context: it streams head.weight one 10 KB row at a time and recomputes every
//! logit twice, once from the reference's f32 head input and once from the bf16 codes the
//! kernel ABI is fed, both in the kernel's ascending order, and compares both to the reference logits
//! (argmax and top-5, plus a gated f32-order delta). It prints the measured bf16-conversion delta,
//! which is why the device comparison uses a matched reference; the measured device-vs-matched delta
//! was 0 and the bar remains the provisional 2e-2 ceiling.
//!
//! Stages 1, 2 and 3 take --layer N and read that layer's tensor names; the released run used layer 0
//! for development, and the same stages pass unchanged on other layers: stage 1 at layers 1, 4, 14
//! and 39, stages 2 and 3 at layer 14 (recon logs in docs/build/phase1-boundary-table.md section 11),
//! so the per-layer kernels are not layer-0-specific. Stage 0 is the CPU-only exception: it needs
//! the host boundary row and compares that row's captured layer-0 coefficients, so it is layer 0
//! only. Stage 7 is the chained replay: it runs the layer's kernels in order, captures all eighteen
//! boundary tensors of row 0 and diffs them by class against the host row (B2); it is layer 0 and one
//! token only because its chain input is the embedding row with the identity mix. The chained
//! 40-layer token harness is a separate program.
//!
//! Build the cubins first, one per entry point and named after it, WITHOUT
//! -DSPARKFLASH_ENABLE_GPU_TRACE (the trace macros then compile out and this launcher passes a
//! null ring pointer):
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/mega_mhc_fused_bf16_sm121.cubin kernels/mega_mhc_sm121.cu
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/flash_mla_csa2_sm121.cubin kernels/flash_mla_csa2_sm121.cu
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/b12x_mxfp4_moe_sm121.cubin kernels/b12x_mxfp4_moe_sm121.cu
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/cert_lm_head_sm121.cubin kernels/cert_lm_head_sm121.cu
//!
//! Stage 8 additionally needs the three kernels it probes, each named after its entry point:
//!
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/b12x_mxfp8_dense_sm121.cubin kernels/b12x_mxfp8_dense_sm121.cu
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/b12x_mxfp8_wo_a_sm121.cubin kernels/b12x_mxfp8_wo_a_sm121.cu
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/swiglu_mxfp8_quant_sm121.cubin kernels/swiglu_mxfp8_quant_sm121.cu
//!
//! Usage:
//!   m7_layer0 --stage 0 --checkpoint <snapshot dir> --host-row <manifest prefix>   (CPU only)
//!   m7_layer0 --stage 1 --checkpoint <snapshot dir> [--cubin-dir out] [--tokens 4] [--layer N]
//!   m7_layer0 --stage 2 --checkpoint <snapshot dir> [--cubin-dir out] [--tokens 130] [--layer N]
//!   m7_layer0 --stage 3 --checkpoint <snapshot dir> [--experts 8|32|64] [--tokens 4] [--layer N]
//!   m7_layer0 --stage 4 --checkpoint <snapshot dir> [--tokens 4] [--repeat 10] [--layer N]
//!   m7_layer0 --stage 5 --checkpoint <snapshot dir> --head-input HEAD.f32 --logits LOGITS.f32 [--cubin-dir out]
//!   m7_layer0 --stage 6 --checkpoint <snapshot dir> --head-input HEAD.f32 --logits LOGITS.f32 [--mem-floor-gib 1]   (CPU only)
//!   m7_layer0 --stage 8 [--cubin-dir out] [--layer N]   (probes: dense, wo_a, swiglu vs in-crate refs)
//!   m7_layer0 --stage 7 --checkpoint <snapshot dir> --host-row <manifest prefix> [--cubin-dir out]
//!                                            (B2: the chained layer-0 replay, one token, 18-tensor row diff)
//!   m7_layer0 --stage 9 --checkpoint <snapshot dir> --host-row <manifest prefix> --stage9-out DIR [--cubin-dir out] [--stage9-layers 40] [--stage9-runs 2] [--stage9-kv-format fp4|engine]
//!                                            (Stage C: the 40-row chained device forward, bars from the B2 deltas, v1 pair + determinism;
//!                                             --stage9-kv-format engine selects the engine's released window-KV format for K5, fp4 is the default)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sparkflash_engine::safetensors::SafeTensorsSource;
use sparkflash_engine::{ModelConfig, TensorPlan, WeightSource};
use sparkflash_kernels::cuda_driver::{
    Cuda, Deviceptr, Kernel as CuFunction, KernelId, Module as CuModule, mem_available_kb,
};
use sparkflash_kernels::cuda_harness::{
    EngineWindowStats, FlashMlaKernel, FlashMlaParams, assert_engine_window_finite,
    engine_window_row, flash_mla_params, launch_flash_mla as flash_mla_launch,
    launch_flash_mla_repeated as flash_mla_launch_repeated,
    load_flash_mla as load_flash_mla_kernel,
};
use sparkflash_kernels::m4_quant::{
    E4M3_ELEMENT_EMAX, QUANT_BLOCK, act_quant_fp8_row, fp4_e2m1_encode, fp8_e4m3_decode,
    fp8_e4m3_encode, ue8m0_decode, ue8m0_from_amax,
};
use sparkflash_kernels::mega_mhc_ref::{
    HC_MULT, PREDICTOR_ROWS, SINKHORN_EPS, SINKHORN_ITERS, Streams, TrafficLedger,
    bias_into_kernel_rows, block_input_normed, mega_mhc_layer_with_scale,
    released_predictor_into_kernel_rows, released_split_sinkhorn, sigmoid,
};
use sparkflash_kernels::{mla_ref, mxfp4_ref, mxfp8_ref, swiglu_ref};

#[path = "../../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{
    ForcedOperandGeometry, HostBoundaryTable, RouterParams, Snapshot, bytes_to_f32,
    combine_bf16_routed_with_shared, forced_control_verdict, free_forced_metrics, gate_up_rows,
    route_token_with, slice_and_pad_rows, verify_forced_ffn_out, verify_forced_operand_provenance,
};

/// Maps a cubin kernel name to the shared driver KernelId.
///
/// The mapping lives in the crate (`KernelId::from_cubin_name`) so the loader and the pinned
/// baseline name are one list, and a kernel's documented alias resolves here too. The flash_mla
/// kernel is deliberately absent: it is loaded only through `cuda_harness::load_flash_mla`, which
/// owns the symbol and returns the opaque FlashMlaKernel, so the harness never holds a raw handle
/// for it.
fn kernel_id(name: &str) -> Result<KernelId, String> {
    KernelId::from_cubin_name(name).ok_or_else(|| {
        format!(
            "{name} is not one of the shared driver kernels; the flash_mla kernel is loaded through cuda_harness::load_flash_mla and has no KernelId"
        )
    })
}

/// The built cubin for any kernel this harness launches.
///
/// The names live in the crate (`cuda_driver::cubin_path`): the [`KernelId`]'s symbol and any
/// documented alias such as the pinned baseline manifest's Mega-MHC source stem, plus the flash_mla
/// kernel, which has its own resolver because `load_flash_mla` owns its symbol. Keeping the
/// resolution in the crate is what lets a test cover it - an example's own `#[test]` is not run by
/// `cargo test`.
fn cubin_path(cubin_dir: &Path, name: &str) -> Result<PathBuf, String> {
    sparkflash_kernels::cuda_driver::cubin_path(cubin_dir, name)
}

/// The hidden width the checkpoint's layer 0 is built around.
const HIDDEN: usize = 5120;
/// The stage-1 grid: four 1280-element tiles of the hidden dimension, one block each.
const MHC_TILES: u32 = 4;
/// The stage-1 block: a whole number of warps, at least one thread per statistic.
const MHC_THREADS: u32 = 128;
/// The 26 statistics the grid reduces: 24 predictor dot products plus the two sum-of-squares
/// statistics the released model needs (over X_hat for the block RMSNorm, over the four streams for
/// the coefficient projection's rsqrt).
const MHC_STATS: usize = 26;
/// The kernel's hc_stats buffer: one final vector plus one block-reduced partial per tile.
/// Mirrors SPARK_M4_MHC_GRID_STATS in kernels/mega_mhc_sm121.cu.
const MHC_GRID_STATS: usize = MHC_STATS * (MHC_TILES as usize + 1);
const _: () = assert!(MHC_STATS == 26 && MHC_GRID_STATS == 130);
/// The memory floor in GiB: the harness refuses to continue, and refuses to allocate, below it.
const DEFAULT_FLOOR_GIB: u64 = 32;
/// The working ceiling in GiB, from the owner's condition: keep the node out of the OOM reaper.
const WORKING_CEILING_GIB: u64 = 100;

const _: () = assert!(HIDDEN == 5120 && HC_MULT == 4 && PREDICTOR_ROWS == 24 && QUANT_BLOCK == 32);

/// One run's arguments. A stage is required: the harness never runs all three in one process,
/// which keeps a failure attributable and a run's peak memory small.
struct Args {
    checkpoint: PathBuf,
    stage: u32,
    cubin_dir: PathBuf,
    tokens: usize,
    experts: usize,
    layer: usize,
    repeat: usize,
    floor_gib: u64,
    /// The reference head input (boundary_dump --head-input-out), for --stage 5.
    head_input: Option<PathBuf>,
    /// The reference logits (boundary_dump --logits-out), for --stage 5.
    logits: Option<PathBuf>,
    /// The phase-1 host boundary v1 pair (manifest prefix) for stage 0's row-0 cross-check.
    host_row: Option<PathBuf>,
    /// Stage 2's KV format: fp4 is the historical kernel FP4 latent path (the default, unchanged),
    /// engine is the engine's released FP8 E4M3 + E8M0-per-32 window format with bf16 storage.
    stage2_kv_format: WindowKvFormat,
    /// The kernel's indexer grid.y. Every slice walks a disjoint stripe of the 2,048 HSI blocks,
    /// sixteen blocks per slice; 64 stays the optimum once stage 2 stops running in every slice
    /// (one slice is 0.161 ms at 128 candidates, and 64 is 0.1613 against 0.1813 at 8).
    grid_y: u32,
    /// Stage 9's row count: the criterion-1 chain is all 40 backbone layers; a smoke run can stop
    /// earlier, and the log names how many rows it actually ran.
    stage9_layers: usize,
    /// Stage 9's determinism repeat count: 2 is the evidence run (two chains, byte-compared),
    /// 1 is a smoke run that makes no determinism claim.
    stage9_runs: usize,
    /// Stage 9's v1 pair destination: `<dir>/device.manifest` and `<dir>/device.blob` are written
    /// there for examples/boundary_diff.rs to run unchanged.
    stage9_out: Option<PathBuf>,
    /// Stage 9's K5 window-KV format: fp4 keeps the merged health-check column byte-identical;
    /// engine selects the engine's released window format for each layer's window row (bf16
    /// storage round, E4M3 codes, one E8M0 scale per 32 channels). Default fp4.
    stage9_kv_format: WindowKvFormat,
}

/// The stage-2 KV format, an explicit selector rather than a sentinel over another argument.
///
/// fp4 keeps the historical FP4 E2M1 latent path byte-identical; engine selects the engine's
/// released window format: one E4M3 code per latent channel, one E8M0 power-of-two scale per 32
/// channels, and the engine's in-place quantise/dequantise round trip, whose output dtype is BF16
/// and which therefore stores bf16 values. The launcher validates this value here, before a cubin
/// is loaded or a launch is built; the kernel's own runtime value guard is a contract-violation
/// backstop and is not the validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WindowKvFormat {
    /// The historical kernel format: E2M1 codes with an E4M3 scale per 16 channels.
    Fp4,
    /// The engine's released window format: E4M3 codes with an E8M0 scale per 32 channels.
    Engine,
}

impl WindowKvFormat {
    fn parse(flag: &str, value: &str) -> Result<Self, String> {
        match value {
            "fp4" => Ok(Self::Fp4),
            "engine" => Ok(Self::Engine),
            other => Err(format!(
                "{flag} {other}: unsupported. Only fp4 (the historical kernel format) and engine \
                 (the engine's released FP8 window format) are supported; every other value is \
                 rejected here, before the kernel can see it."
            )),
        }
    }

    fn kernel_value(self) -> u32 {
        match self {
            Self::Fp4 => 0,
            Self::Engine => 1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fp4 => "fp4 (historical kernel format)",
            Self::Engine => "engine (released FP8 window format, bf16 storage)",
        }
    }
}

fn next_value(values: &mut std::vec::IntoIter<String>, flag: &str) -> Result<String, String> {
    values.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        checkpoint: PathBuf::new(),
        stage: 0,
        cubin_dir: PathBuf::from("out"),
        tokens: 4,
        experts: 8,
        layer: 0,
        repeat: 1,
        floor_gib: DEFAULT_FLOOR_GIB,
        grid_y: 64,
        stage9_layers: 40,
        stage9_runs: 2,
        stage9_out: None,
        stage9_kv_format: WindowKvFormat::Fp4,
        head_input: None,
        logits: None,
        host_row: None,
        stage2_kv_format: WindowKvFormat::Fp4,
    };
    let mut values: std::vec::IntoIter<String> =
        std::env::args().skip(1).collect::<Vec<_>>().into_iter();
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--checkpoint" => args.checkpoint = PathBuf::from(next_value(&mut values, &flag)?),
            "--stage" => {
                args.stage = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--stage: {e}"))?;
            }
            "--cubin-dir" => args.cubin_dir = PathBuf::from(next_value(&mut values, &flag)?),
            "--tokens" => {
                args.tokens = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--tokens: {e}"))?;
            }
            "--experts" => {
                args.experts = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--experts: {e}"))?;
            }
            "--layer" => {
                args.layer = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--layer: {e}"))?;
            }
            "--repeat" => {
                args.repeat = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--repeat: {e}"))?;
            }
            "--mem-floor-gib" => {
                args.floor_gib = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--mem-floor-gib: {e}"))?;
            }
            "--grid-y" => {
                args.grid_y = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--grid-y: {e}"))?;
            }
            "--head-input" => {
                args.head_input = Some(PathBuf::from(next_value(&mut values, &flag)?));
            }
            "--logits" => {
                args.logits = Some(PathBuf::from(next_value(&mut values, &flag)?));
            }
            "--host-row" => {
                args.host_row = Some(PathBuf::from(next_value(&mut values, &flag)?));
            }
            "--stage2-kv-format" => {
                args.stage2_kv_format =
                    WindowKvFormat::parse("--stage2-kv-format", &next_value(&mut values, &flag)?)?;
            }
            "--stage9-layers" => {
                args.stage9_layers = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--stage9-layers: {e}"))?;
            }
            "--stage9-runs" => {
                args.stage9_runs = next_value(&mut values, &flag)?
                    .parse()
                    .map_err(|e| format!("--stage9-runs: {e}"))?;
            }
            "--stage9-out" => {
                args.stage9_out = Some(PathBuf::from(next_value(&mut values, &flag)?));
            }
            "--stage9-kv-format" => {
                args.stage9_kv_format =
                    WindowKvFormat::parse("--stage9-kv-format", &next_value(&mut values, &flag)?)?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.stage > 9 {
        return Err("--stage 0 through 9 is required: one stage per run".to_string());
    }
    if args.host_row.is_some() && args.stage != 0 && args.stage != 7 && args.stage != 9 {
        return Err(format!(
            "--host-row is consumed by stage 0, stage 7 and stage 9, not by stage {}",
            args.stage
        ));
    }
    if args.stage == 9 {
        if args.host_row.is_none() {
            return Err(
                "--stage 9 needs --host-row <manifest prefix>: the emitted v1 pair is diffed against \
                 the host pair by examples/boundary_diff.rs"
                    .to_string(),
            );
        }
        if args.stage9_out.is_none() {
            return Err(
                "--stage 9 needs --stage9-out <dir> for its v1 manifest/blob pair".to_string(),
            );
        }
        if args.stage9_layers == 0 || args.stage9_layers > 40 {
            return Err(format!(
                "--stage9-layers must be in 1..=40 (the backbone has 40 rows); {} requested",
                args.stage9_layers
            ));
        }
        if args.stage9_runs == 0 {
            return Err("--stage9-runs must be at least 1".to_string());
        }
    }
    if args.stage == 0 && args.host_row.is_none() {
        return Err(
            "--stage 0 needs --host-row <manifest prefix>: it cross-checks the \
                    captured row-0 coefficients"
                .to_string(),
        );
    }
    if args.checkpoint.as_os_str().is_empty() {
        return Err("--checkpoint is required".to_string());
    }
    if args.tokens == 0 {
        return Err("--tokens must be at least 1".to_string());
    }
    // 2,048 HSI blocks over eight warps and two heads per indexer head leaves at most 128 useful
    // slices with work in each; beyond that the slice loop is empty and the block is pure launch
    // overhead. The lower bound is the other side: one slice serialises the pool.
    if args.grid_y == 0 || args.grid_y > 256 {
        return Err("--grid-y must be in 1..=256 (2,048 HSI blocks over eight warps)".to_string());
    }
    Ok(args)
}

fn mem_available_kb_checked() -> Result<u64, String> {
    mem_available_kb().ok_or_else(|| "cannot read MemAvailable from /proc/meminfo".to_string())
}

/// Print MemAvailable and refuse to continue under the floor. The floor is recomputed from bytes
/// every time rather than trusted from a literal, and the refusal prints the recomputed figure.
fn check_floor(args: &Args, point: &str) -> Result<(), String> {
    let kb = mem_available_kb_checked()?;
    let bytes = kb.saturating_mul(1024);
    let floor_bytes = args.floor_gib.saturating_mul(1024 * 1024 * 1024);
    if bytes < floor_bytes {
        return Err(format!(
            "MemAvailable at {point} is {kb} KiB = {:.2} GiB, below the {}-GiB floor              (recomputed as {floor_bytes} bytes): refusing to continue",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            args.floor_gib,
        ));
    }
    println!(
        "MemAvailable at {point}: {kb} kB ({:.1} GiB), floor {} GiB - ok",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        args.floor_gib,
    );
    Ok(())
}

/// Admission control for an allocation: it must leave the floor free afterward, and it must stay
/// under the working ceiling. This is the check that makes an allocation which only succeeds if
/// nothing else grows impossible to make by accident.
fn admit_memory(args: &Args, planned_bytes: u64, what: &str) -> Result<(), String> {
    let kb = mem_available_kb_checked()?;
    let available = kb.saturating_mul(1024);
    let floor_bytes = args.floor_gib.saturating_mul(1024 * 1024 * 1024);
    let ceiling_bytes = WORKING_CEILING_GIB.saturating_mul(1024 * 1024 * 1024);
    let planned_gib = planned_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let available_gib = available as f64 / (1024.0 * 1024.0 * 1024.0);
    if planned_bytes > ceiling_bytes {
        return Err(format!(
            "{what}: {planned_bytes} bytes ({planned_gib:.1} GiB) is over the              {WORKING_CEILING_GIB}-GiB working ceiling (recomputed as {ceiling_bytes} bytes)"
        ));
    }
    if available < planned_bytes.saturating_add(floor_bytes) {
        return Err(format!(
            "{what}: {planned_bytes} bytes ({planned_gib:.1} GiB) would leave              {} KiB of MemAvailable, below the {}-GiB floor (recomputed as {floor_bytes} bytes);              MemAvailable now {kb} kB ({available_gib:.1} GiB). Refusing to allocate",
            available.saturating_sub(planned_bytes) / 1024,
            args.floor_gib,
        ));
    }
    println!(
        "{what}: {planned_bytes} bytes ({planned_gib:.1} GiB) admitted; MemAvailable {available_gib:.1} GiB          with a {} GiB floor and a {WORKING_CEILING_GIB} GiB working ceiling",
        args.floor_gib,
    );
    Ok(())
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_bits_to_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn bf16_bits_of_f32(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|v| half::bf16::from_f32(*v).to_bits())
        .collect()
}

/// One comparison's numbers, so a verdict can be computed from what was printed rather than
/// asserted around it.
struct Stats {
    len: usize,
    exact: usize,
    max_delta: f32,
    at: usize,
}

impl Stats {
    fn exact_pct(&self) -> f64 {
        100.0 * self.exact as f64 / self.len as f64
    }

    fn all_exact(&self) -> bool {
        self.exact == self.len
    }
}

fn compare_f32(label: &str, device: &[f32], reference: &[f32]) -> Result<Stats, String> {
    if device.len() != reference.len() {
        return Err(format!(
            "{label}: {} device values against {} reference values",
            device.len(),
            reference.len()
        ));
    }
    let mut stats = Stats {
        len: device.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    for (index, (d, r)) in device.iter().zip(reference).enumerate() {
        if d.to_bits() == r.to_bits() {
            stats.exact += 1;
        }
        let delta = (*d - *r).abs();
        if !delta.is_finite() || delta > stats.max_delta {
            stats.max_delta = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
            stats.at = index;
        }
    }
    println!(
        "  {label}: {} values, exact {}/{} ({:.4}%), max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    Ok(stats)
}

fn compare_u8(label: &str, device: &[u8], reference: &[u8]) -> Result<Stats, String> {
    if device.len() != reference.len() {
        return Err(format!(
            "{label}: {} device values against {} reference values",
            device.len(),
            reference.len()
        ));
    }
    let mut stats = Stats {
        len: device.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    for (index, (d, r)) in device.iter().zip(reference).enumerate() {
        if d == r {
            stats.exact += 1;
        }
        let delta = (*d as f32 - *r as f32).abs();
        if delta > stats.max_delta {
            stats.max_delta = delta;
            stats.at = index;
        }
    }
    println!(
        "  {label}: {} bytes, exact {}/{} ({:.4}%), max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    Ok(stats)
}

/// Compare a device BF16 write against the BF16 rounding of the expected f32 value, which is the
/// same rounding point on both sides: the device stores bf16, so the expectation is rounded to
/// bf16 before it is compared rather than compared as f32.
fn compare_bf16_write(label: &str, device_bits: &[u16], expected: &[f32]) -> Result<Stats, String> {
    let expected_bits = bf16_bits_of_f32(expected);
    if device_bits.len() != expected_bits.len() {
        return Err(format!(
            "{label}: {} device values against {} expected values",
            device_bits.len(),
            expected_bits.len()
        ));
    }
    let mut stats = Stats {
        len: device_bits.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    for (index, (d, e)) in device_bits.iter().zip(&expected_bits).enumerate() {
        if d == e {
            stats.exact += 1;
        }
        let delta = (half::bf16::from_bits(*d).to_f32() - half::bf16::from_bits(*e).to_f32()).abs();
        if !delta.is_finite() || delta > stats.max_delta {
            stats.max_delta = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
            stats.at = index;
        }
    }
    println!(
        "  {label}: {} bf16 values, exact {}/{} ({:.4}%), max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    Ok(stats)
}

/// The released per-sublayer mHC tensors in the form the kernel and the reference consume.
///
/// The predictor is the RELEASED [24, n*d] matrix, in the kernel's B/A/C row order with the
/// released pre/post/comb rows mapped and the comb terms transposed - B0 removed the d-wide fold, so
/// the rows are permuted and never averaged, and both the reference and the kernel take this full
/// form; `hc_scale` stays in the release's [pre, post, comb] order because the kernel indexes it by
/// group. Nothing is pre-folded: the scale must NOT multiply the predictor here, or the kernel would
/// apply it a second time.
///
/// The bias is kept twice because the two consumers want different orders. `hc_base` is the
/// checkpoint's 24-row order, which is what the reference takes (`mega_mhc_layer_with_scale` maps
/// it itself); `hc_base_kernel` is the same bias mapped onto the kernel's B/A/C rows and is the
/// only form handed to the device. Passing `hc_base_kernel` to the reference would map it twice,
/// so the field names carry the difference.
struct ReleasedMhcTensors {
    /// The released matrix exactly as the checkpoint stores it: [24][n*d], released row order.
    released: Vec<f32>,
    /// The same matrix permuted into kernel row order, copies kept.
    predictor: Vec<f32>,
    hc_base: [f32; PREDICTOR_ROWS],
    hc_base_kernel: [f32; PREDICTOR_ROWS],
    hc_scale: [f32; 3],
}

fn read_released_mhc_tensors(
    snap: &mut Snapshot,
    prefix: &str,
    sublayer: &str,
) -> Result<ReleasedMhcTensors, String> {
    let fn_name = format!("{prefix}.hc_{sublayer}_fn");
    let fn_shape = [PREDICTOR_ROWS, HC_MULT * HIDDEN];
    let raw_fn = snap.read_f32(&fn_name, &fn_shape)?;
    let base = snap.read_f32(&format!("{prefix}.hc_{sublayer}_base"), &[PREDICTOR_ROWS])?;
    let group_scale = snap.read_f32(&format!("{prefix}.hc_{sublayer}_scale"), &[3])?;
    let hc_base: [f32; PREDICTOR_ROWS] = base
        .as_slice()
        .try_into()
        .map_err(|_| format!("{prefix}.hc_{sublayer}_base must be {PREDICTOR_ROWS} values"))?;
    let hc_scale: [f32; 3] = group_scale
        .as_slice()
        .try_into()
        .map_err(|_| format!("{prefix}.hc_{sublayer}_scale must be three values"))?;
    // B0: permute the released rows into kernel order, keep all n copies. The old
    // fold_predictor_into_kernel_rows no longer exists; nothing here may average.
    let mut predictor = vec![0.0f32; PREDICTOR_ROWS * HC_MULT * HIDDEN];
    released_predictor_into_kernel_rows(&raw_fn, &mut predictor);
    Ok(ReleasedMhcTensors {
        released: raw_fn,
        predictor,
        hc_base,
        hc_base_kernel: bias_into_kernel_rows(&hc_base),
        hc_scale,
    })
}

/// Compare one coefficient group the reference computed against the engine's captured tensor.
///
/// Returns (values compared, max abs delta, index of the max) and prints a line. \`tensor\` is the
/// row-0 tensor name, \`captured\` its f32 values as the engine wrote them, and \`reference\` the
/// reference's values in the same released orientation (the caller transposes B, because the
/// reference's B is [destination][source] while the captured comb is [source][destination]).
fn compare_row0_coefficient(
    tensor: &str,
    reference: &[f32],
    captured: &[f32],
) -> Result<(usize, f32, usize), String> {
    if reference.len() != captured.len() {
        return Err(format!(
            "{tensor}: {} reference values against {} captured values",
            reference.len(),
            captured.len()
        ));
    }
    let mut max_delta = 0.0f32;
    let mut at = 0;
    for (index, (reference, captured)) in reference.iter().zip(captured).enumerate() {
        let delta = (reference - captured).abs();
        if !delta.is_finite() || delta > max_delta {
            max_delta = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
            at = index;
        }
    }
    println!(
        "  {tensor}: {} values, max abs delta {:e} at index {}",
        captured.len(),
        max_delta,
        at
    );
    Ok((captured.len(), max_delta, at))
}

/// The released H(X) coefficient tensors for one predictor, computed through the reference's
/// released arithmetic from raw logits in kernel row order. Used only by stage 0's non-vacuity
/// half, so the retired fold can be compared against the engine's captured row without a second
/// coefficient implementation.
fn released_coefficients_from_raw(
    raw: &[f32; PREDICTOR_ROWS],
    hc_base: &[f32; PREDICTOR_ROWS],
    hc_scale: &[f32; 3],
) -> ([f32; 16], [f32; HC_MULT], [f32; HC_MULT]) {
    let mut mixing = [[0.0f32; HC_MULT]; HC_MULT];
    for (source, row_mixing) in mixing.iter_mut().enumerate() {
        for (destination, slot) in row_mixing.iter_mut().enumerate() {
            let row = destination * HC_MULT + source;
            *slot = raw[row] * hc_scale[2] + hc_base[row];
        }
    }
    released_split_sinkhorn(&mut mixing, SINKHORN_ITERS, SINKHORN_EPS);
    let mut b = [0.0f32; 16];
    for destination in 0..HC_MULT {
        for source in 0..HC_MULT {
            b[destination * HC_MULT + source] = mixing[source][destination];
        }
    }
    let mut a = [0.0f32; HC_MULT];
    let mut c = [0.0f32; HC_MULT];
    for j in 0..HC_MULT {
        a[j] = sigmoid(raw[HC_MULT * HC_MULT + j] * hc_scale[0] + hc_base[HC_MULT * HC_MULT + j])
            + SINKHORN_EPS;
        c[j] = 2.0
            * sigmoid(
                raw[HC_MULT * HC_MULT + HC_MULT + j] * hc_scale[1]
                    + hc_base[HC_MULT * HC_MULT + HC_MULT + j],
            );
    }
    (b, a, c)
}

/// The raw logits of the RETIRED d-wide fold: each released row averaged over the four copies, then
/// projected against the stream mean in the same accumulation order the reference uses. This is the
/// pre-B0 semantics, kept only as stage 0's non-vacuity evidence; nothing in the device path may use
/// it.
fn retired_fold_raw_logits(released: &[f32], streams: &[f32]) -> [f32; PREDICTOR_ROWS] {
    let mut folded = vec![0.0f32; PREDICTOR_ROWS * HIDDEN];
    for kernel_row in 0..PREDICTOR_ROWS {
        let released_row = match kernel_row {
            row if row < HC_MULT * HC_MULT => {
                let destination = row / HC_MULT;
                let source = row % HC_MULT;
                2 * HC_MULT + source * HC_MULT + destination
            }
            row if row < HC_MULT * HC_MULT + HC_MULT => row - HC_MULT * HC_MULT,
            row => HC_MULT + (row - HC_MULT * HC_MULT - HC_MULT),
        };
        for column in 0..HIDDEN {
            let mut sum = 0.0f32;
            for copy in 0..HC_MULT {
                sum += released[released_row * HC_MULT * HIDDEN + copy * HIDDEN + column];
            }
            folded[kernel_row * HIDDEN + column] = sum / HC_MULT as f32;
        }
    }
    let mut mean = vec![0.0f32; HIDDEN];
    for copy in 0..HC_MULT {
        for column in 0..HIDDEN {
            mean[column] += streams[copy * HIDDEN + column];
        }
    }
    for value in mean.iter_mut() {
        *value /= HC_MULT as f32;
    }
    let mut raw = [0.0f32; PREDICTOR_ROWS];
    for row in 0..PREDICTOR_ROWS {
        let mut acc = 0.0f32;
        for column in 0..HIDDEN {
            acc += mean[column] * folded[row * HIDDEN + column];
        }
        raw[row] = acc;
    }
    raw
}

// Layout: this file is the crate root. It keeps the shared argument parsing, memory
// discipline, comparison helpers and main; one module per stage sits beside it
// (stage0.rs ... stage9.rs in this directory), and that is where each stage code and
// doc comment live. The CLI and the stage set are unchanged by the split.

mod stage0;
mod stage1;
mod stage2;
mod stage3;
mod stage4;
mod stage5;
mod stage6;
mod stage7;
mod stage8;
mod stage9;

use crate::stage0::stage0;
use crate::stage1::stage1;
use crate::stage2::stage2;
use crate::stage3::stage3;
use crate::stage4::stage4;
use crate::stage5::stage5;
use crate::stage6::stage6;
use crate::stage7::stage7;
use crate::stage8::stage8;
use crate::stage9::stage9;

fn main() -> Result<(), String> {
    let args = parse_args()?;
    println!("argv: {}", std::env::args().collect::<Vec<_>>().join(" "));
    println!(
        "memory discipline: floor {} GiB, working ceiling {WORKING_CEILING_GIB} GiB, MemAvailable checked before and after every launch",
        args.floor_gib
    );
    if args.stage == 0 {
        println!(
            "stage 0 is CPU-only: no CUDA context is created and the device memory floor is not \
             checked; the only allocation is the row-0 tensors from the manifest/blob"
        );
    } else {
        check_floor(&args, "process start")?;
    }
    match args.stage {
        0 => stage0(&args),
        1 => stage1(&args),
        2 => stage2(&args),
        3 => stage3(&args),
        4 => stage4(&args),
        5 => stage5(&args),
        6 => stage6(&args),
        8 => stage8(&args),
        7 => stage7(&args),
        9 => stage9(&args),
        other => Err(format!(
            "unknown stage {other}: one of 0, 1, 2, 3, 4, 5, 6, 7, 8, 9"
        )),
    }
}
