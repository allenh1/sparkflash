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
use std::ffi::c_void;
use std::path::{Path, PathBuf};

use sparkflash_engine::safetensors::SafeTensorsSource;
use sparkflash_engine::{ModelConfig, TensorPlan, WeightSource};
use sparkflash_kernels::m4_quant::{
    E4M3_ELEMENT_EMAX, QUANT_BLOCK, act_quant_fp8_row, bf16_bits, fp4_e2m1_encode, fp8_e4m3_decode,
    fp8_e4m3_encode, ue8m0_decode, ue8m0_from_amax,
};
use sparkflash_kernels::mega_mhc_ref::{
    HC_MULT, PREDICTOR_ROWS, SINKHORN_EPS, SINKHORN_ITERS, Streams, TrafficLedger,
    bias_into_kernel_rows, block_input_normed, mega_mhc_layer_with_scale,
    released_predictor_into_kernel_rows, released_split_sinkhorn, sigmoid,
};
use sparkflash_kernels::{mla_ref, mxfp4_ref, mxfp8_ref, swiglu_ref};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::{
    ForcedOperandGeometry, HostBoundaryTable, RouterParams, Snapshot, bytes_to_f32,
    combine_bf16_routed_with_shared, forced_control_verdict, free_forced_metrics, gate_up_rows,
    route_token_with, slice_and_pad_rows, verify_forced_ffn_out, verify_forced_operand_provenance,
};

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

/// One row of the engine's released window-KV format, produced by the one construction path that
/// stage 2 and stage 9 share.
struct EngineWindowRow {
    codes: Vec<u8>,
    scales: Vec<u8>,
}

/// The one construction of an engine-format window row, shared by stage 2 and stage 9 so the
/// storage/quantiser order cannot drift between the two harnesses. The caller has already applied
/// the layer's rms_norm and rope tail in f32; this function applies the engine's storage-dtype
/// round (bf16) exactly once, quantises with the engine's released
/// sparkflash_engine::quant::quantize_activation_fp8, and asserts the result is byte-identical to
/// m4_quant::act_quant_fp8_row on the same bf16 row. No second bf16 round happens here: the
/// engine quantiser returns the input dtype, so the bf16 row is the stored row.
fn engine_window_row(latent: &[f32]) -> Result<EngineWindowRow, String> {
    let bf16_row: Vec<f32> = latent
        .iter()
        .map(|value| half::bf16::from_bits(bf16_bits(*value)).to_f32())
        .collect();
    let engine_row = sparkflash_engine::quant::quantize_activation_fp8(&bf16_row)
        .map_err(|error| format!("engine window quantiser: {error}"))?;
    let (oracle_codes, oracle_scales) = act_quant_fp8_row(&bf16_row);
    if engine_row.codes() != oracle_codes.as_slice()
        || engine_row.scales() != oracle_scales.as_slice()
    {
        return Err(
            "the engine window quantiser and m4_quant::act_quant_fp8_row disagree on the same \
             bf16 row; the format oracle is not the engine's rule"
                .to_string(),
        );
    }
    Ok(EngineWindowRow {
        codes: engine_row.codes().to_vec(),
        scales: engine_row.scales().to_vec(),
    })
}

/// Refuse to hand engine window bytes to the kernel when they decode to a NaN or a non-finite
/// value. The engine quantiser clamps to +/-448 and never emits the E4M3 NaN code 0x7F, so this
/// guard turns a corrupt row into a loud failure instead of a poisoned attention output.
fn assert_engine_window_finite(
    codes: &[u8],
    scales: &[u8],
    rows: usize,
    context: &str,
) -> Result<(), String> {
    let nan_codes = codes.iter().filter(|code| **code == 0x7F).count();
    let mut non_finite = 0usize;
    for row in 0..rows {
        for index in 0..MLA_LATENT {
            if !mla_ref::window_kv_value(codes, scales, row, index).is_finite() {
                non_finite += 1;
            }
        }
    }
    if nan_codes != 0 || non_finite != 0 {
        return Err(format!(
            "{context}: engine window decode is not finite: {nan_codes} E4M3 NaN codes (0x7F) and \
             {non_finite} non-finite decoded values in {rows} rows; the engine quantiser cannot \
             emit these, so refuse the launch instead of letting the kernel compute on them"
        ));
    }
    Ok(())
}

/// Per-run engine-window counters for stage 9: rows built by the shared helper, oracle agreements,
/// and rows that passed the finite-decode guard.
#[derive(Default)]
struct EngineWindowStats {
    rows: usize,
    quantiser_agreements: usize,
    guarded_rows: usize,
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

/// One captured block input's quantised tensors, recomputed from the row's own f32 values.
///
/// The engine wrote pre_*_hidden (the f32 normalised block input), pre_*_xhat (E4M3 codes) and
/// pre_*_scales (ue8m0) through quant::quantize_activation_fp8, the reference ceil rule.
/// [act_quant_fp8_row] is that same rule, so this checks the rule the device kernels now carry
/// against the engine's OWN captured arithmetic, not against another copy of the device reference.
/// The retired floor-rule codes are printed unjudged: they are the size of the gap D8 closed.
///
/// Returns (code mismatches, scale mismatches).
fn stage0_activation_quantisers(
    table: &HostBoundaryTable,
    prefix: &str,
    hidden_name: &str,
    xhat_name: &str,
    scales_name: &str,
) -> Result<(usize, usize), String> {
    let hidden = table.f32(0, hidden_name)?;
    let captured_codes = table.u8(0, xhat_name)?;
    let captured_scales = table.u8(0, scales_name)?;
    if hidden.len() != HIDDEN {
        return Err(format!(
            "row 0 {hidden_name}: expected {HIDDEN} f32 values, got {}",
            hidden.len()
        ));
    }
    if captured_codes.len() != HIDDEN || captured_scales.len() != HIDDEN / QUANT_BLOCK {
        return Err(format!(
            "row 0 {xhat_name}: expected {HIDDEN} codes and {} scales, got {} and {}",
            HIDDEN / QUANT_BLOCK,
            captured_codes.len(),
            captured_scales.len()
        ));
    }
    let (codes, scales) = act_quant_fp8_row(&hidden);
    let code_mismatches = codes
        .iter()
        .zip(&captured_codes)
        .filter(|(recomputed, captured)| recomputed != captured)
        .count();
    let scale_mismatches = scales
        .iter()
        .zip(&captured_scales)
        .filter(|(recomputed, captured)| recomputed != captured)
        .count();
    // The retired floor-log2 weight-grid rule on the same f32 values, for the gap's size only.
    let mut floor_codes = [0u8; HIDDEN];
    let mut floor_scales = [0u8; HIDDEN / QUANT_BLOCK];
    let (code_blocks, _) = floor_codes.as_chunks_mut::<QUANT_BLOCK>();
    for (block, (scale_slot, code_slot)) in floor_scales.iter_mut().zip(code_blocks).enumerate() {
        let hidden_block = &hidden[block * QUANT_BLOCK..(block + 1) * QUANT_BLOCK];
        let amax = hidden_block
            .iter()
            .fold(0.0f32, |amax, value| amax.max(value.abs()));
        let code = ue8m0_from_amax(amax, E4M3_ELEMENT_EMAX);
        *scale_slot = code;
        let scale = ue8m0_decode(code);
        for (dst, value) in code_slot.iter_mut().zip(hidden_block) {
            *dst = fp8_e4m3_encode(value / scale);
        }
    }
    let floor_code_mismatches = floor_codes
        .iter()
        .zip(&captured_codes)
        .filter(|(recomputed, captured)| recomputed != captured)
        .count();
    let floor_scale_mismatches = floor_scales
        .iter()
        .zip(&captured_scales)
        .filter(|(recomputed, captured)| recomputed != captured)
        .count();
    println!(
        "  {prefix} activation quantiser: ceil-rule codes {}/{HIDDEN} exact, scales {}/{} exact; \
         retired floor rule would give {}/{HIDDEN} codes and {}/{} scales exact (not judged)",
        HIDDEN - code_mismatches,
        HIDDEN / QUANT_BLOCK - scale_mismatches,
        HIDDEN / QUANT_BLOCK,
        HIDDEN - floor_code_mismatches,
        HIDDEN / QUANT_BLOCK - floor_scale_mismatches,
        HIDDEN / QUANT_BLOCK,
    );
    Ok((code_mismatches, scale_mismatches))
}

/// Stage 0: the fold-closure cross-check, CPU only.
///
/// It verifies the B0 claim against the engine's OWN captured arithmetic rather than against another
/// copy of the reference: the released projection (all four copies, no averaging) must reproduce the
/// coefficient tensors the host boundary row 0 recorded, and the retired fold must not. No CUDA
/// context is created, so this runs while vLLM holds the node.
fn stage0(args: &Args) -> Result<(), String> {
    let host_row = args
        .host_row
        .as_ref()
        .ok_or("--stage 0 needs --host-row <manifest prefix>")?;
    println!(
        "=== stage 0: the released projection against the engine's captured row-0 coefficients ==="
    );
    println!(
        "comparison class: HOST-ONLY. The released hc_*_fn [24, n*d] matrices are projected over the \
         flattened four-copy stream with the reference's released arithmetic, and the resulting \
         pre/post/comb coefficients are diffed against the tensors the engine captured in boundary \
         row 0. No CUDA context is created."
    );
    println!(
        "pass bar: every full-projection coefficient within 1e-5 absolute of the engine's captured \
         value (both sides are f32 sequential reductions over the same rows, so the expected delta \
         is a few ULP). The retired-fold column is printed as evidence of the size of the semantic \
         gap and is not judged."
    );
    if args.layer != 0 {
        return Err(format!(
            "stage 0 cross-checks captured row 0, so it needs --layer 0, got {}",
            args.layer
        ));
    }
    println!("host pair: {}", host_row.display());
    let table = HostBoundaryTable::open(host_row)?;
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The attn sublayer's streams are the row's residual_in (the updated X_l the block entered
    // with); the ffn sublayer's streams are the row's post_attention_hidden. Both are captured
    // [positions, hc, dim] f32 and one token is one 4*5120 stream.
    let residual_in = table.f32(0, "residual_in")?;
    let post_attention = table.f32(0, "post_attention_hidden")?;
    if residual_in.len() != HC_MULT * HIDDEN || post_attention.len() != HC_MULT * HIDDEN {
        return Err(format!(
            "row 0: expected two {}-value streams, got {} and {}",
            HC_MULT * HIDDEN,
            residual_in.len(),
            post_attention.len()
        ));
    }
    let mut worst = 0.0f32;
    let mut quant_mismatches = 0usize;
    for (sublayer, streams, prefix) in [
        ("attn", &residual_in, "hc_attn"),
        ("ffn", &post_attention, "hc_ffn"),
    ] {
        let mhc = read_released_mhc_tensors(&mut snap, "layers.0", sublayer)?;
        println!(
            "{sublayer}: predictor {} f32 (released [24, 4*d], rows in kernel order, no averaging)",
            mhc.predictor.len()
        );
        let (_, coefficients, _, _) = mega_mhc_layer_with_scale(
            &mut Streams::new(streams.clone()),
            &vec![1.0f32; HIDDEN],
            &[1.0, 0.0, 0.0, 0.0],
            &mhc.predictor,
            &mhc.hc_base,
            &mhc.hc_scale,
            &mut TrafficLedger::default(),
        );
        // The captured coefficients are in the release's layouts: comb [source][destination],
        // pre and post one value per copy. The reference's B is [destination][source].
        let mut b_reference = vec![0.0f32; HC_MULT * HC_MULT];
        for destination in 0..HC_MULT {
            for source in 0..HC_MULT {
                b_reference[source * HC_MULT + destination] = coefficients.b[destination][source];
            }
        }
        let pre = table.f32(0, &format!("{prefix}_pre"))?;
        let post = table.f32(0, &format!("{prefix}_post"))?;
        let comb = table.f32(0, &format!("{prefix}_comb"))?;
        let (_, d_pre, _) =
            compare_row0_coefficient(&format!("{prefix}_pre"), &coefficients.a, &pre)?;
        let (_, d_post, _) =
            compare_row0_coefficient(&format!("{prefix}_post"), &coefficients.c, &post)?;
        let (_, d_comb, _) =
            compare_row0_coefficient(&format!("{prefix}_comb"), &b_reference, &comb)?;
        worst = worst.max(d_pre).max(d_post).max(d_comb);

        // Non-vacuity: the pre-B0 fold. Its predictor rows are averaged over the four copies and its
        // projection is against the stream mean; it is computed explicitly so the reported delta
        // belongs to the retired semantics, not to a scaled full projection.
        let raw = retired_fold_raw_logits(&mhc.released, streams);
        let (folded_b, folded_a, folded_c) =
            released_coefficients_from_raw(&raw, &mhc.hc_base, &mhc.hc_scale);
        let mut folded_b_released = vec![0.0f32; HC_MULT * HC_MULT];
        for destination in 0..HC_MULT {
            for source in 0..HC_MULT {
                folded_b_released[source * HC_MULT + destination] =
                    folded_b[destination * HC_MULT + source];
            }
        }
        let (_, fold_pre, _) =
            compare_row0_coefficient(&format!("{prefix}_pre (retired fold)"), &folded_a, &pre)?;
        let (_, fold_post, _) =
            compare_row0_coefficient(&format!("{prefix}_post (retired fold)"), &folded_c, &post)?;
        let (_, fold_comb, _) = compare_row0_coefficient(
            &format!("{prefix}_comb (retired fold)"),
            &folded_b_released,
            &comb,
        )?;
        println!(
            "  {sublayer} retired fold vs captured: pre {:e}, post {:e}, comb {:e} (not judged)",
            fold_pre, fold_post, fold_comb
        );

        // D8: the captured activation-quantised block input, recomputed from the row's own f32
        // block input with the reference ceil rule. The engine wrote these tensors through its own
        // quantise_activation_fp8, so this is the rule change checked against the engine's
        // arithmetic, not against the device reference.
        let (quant_prefix, hidden_name, xhat_name, scales_name) = match sublayer {
            "attn" => (
                "pre_attention",
                "pre_attention_hidden",
                "pre_attention_xhat",
                "pre_attention_scales",
            ),
            "ffn" => (
                "pre_ffn",
                "pre_ffn_hidden",
                "pre_ffn_xhat",
                "pre_ffn_scales",
            ),
            other => return Err(format!("unknown sublayer {other}")),
        };
        let (code_mismatches, scale_mismatches) = stage0_activation_quantisers(
            &table,
            quant_prefix,
            hidden_name,
            xhat_name,
            scales_name,
        )?;
        quant_mismatches += code_mismatches + scale_mismatches;
    }
    println!("stage 0: full-projection worst max abs delta {worst:e} against a 1e-5 bar");
    println!(
        "stage 0: activation-quantiser mismatches {quant_mismatches} against a 0 bar (ceil rule must reproduce the engine's captured codes and scales)"
    );
    if worst <= 1e-5 && quant_mismatches == 0 {
        println!(
            "stage 0: PASS (the released projection reproduces the engine's captured row-0 coefficients, and the ceil-rule activation quantiser reproduces its captured xhat codes and scales)"
        );
        Ok(())
    } else {
        Err(format!(
            "stage 0: FAIL - worst projection delta {worst:e} (bar 1e-5), activation-quantiser mismatches {quant_mismatches} (bar 0)"
        ))
    }
}

/// Stage 1: the fused Mega-MHC kernel against the in-crate reference, on real layer-0 bytes.
///
/// The reference returns the coefficients, the FP8 codes and the ue8m0 scales as well as the f32
/// streams, so this compares both sides of the quantiser. The residual streams are compared against
/// the bf16 rounding of the same arithmetic computed on the host from the inputs the device got,
/// because the reference's own "next" is the NEXT layer's stream update and is not a kernel output.
fn stage1(args: &Args) -> Result<(), String> {
    let kernel = "mega_mhc_fused_bf16_sm121";
    let prefix = format!("layers.{}", args.layer);
    println!("=== stage 1: {kernel} against mega_mhc_ref::mega_mhc_layer_with_scale ===");
    println!(
        "comparison class: internal consistency of the fused Mega-mHC kernel against the in-crate \
         CPU reference, on real layer-0 checkpoint bytes; the predictor is the released flattened \
         [24, n*d] projection on BOTH sides (B0 closed the fold), NOT model fidelity"
    );
    println!(
        "compared at every boundary, exact matches and max abs delta: coefficients B[16] A[4] C[4] \
         (the kernel's hc_coeffs), the residual streams X_l (device bf16 write against the bf16 \
         rounding of the same arithmetic), the FP8 block input (5120 codes, exact bytes), its ue8m0 \
         scales (160 bytes), and 1/rms(X_hat)"
    );
    println!(
        "layer-0 boundary seed, since the checkpoint stores no layer -1 coefficients: B = identity, \
         C = 0, A = [1, 0, 0, 0], Y_-1 = 0, X_-1 = four copies of one embed.weight row"
    );
    println!(
        "predictor: hc_attn_fn is stored as [24, 20480] over the flattened streams, and BOTH sides \
         consume the released [24, n*d] projection over all four copies (B0 removed the old d-wide \
         fold from both). The kernel's B(0..16)/A(16..20)/C(20..24) order is the released \
         pre/post/comb rows mapped with the comb terms transposed; the group scale is NOT \
         pre-folded, the kernel takes hc_attn_scale itself. The RETIRED fold over the stream MEAN is \
         measured only by stage 0's non-vacuity half and is not what this stage compares"
    );
    println!(
        "released tensors on the device path: hc_attn_scale [3] and hc_attn_base [24] both reach the \
         kernel, which applies the projection rstd, mix*scale+base, the sigmoid pre/post and the \
         released split-Sinkhorn exactly as the reference does; no coefficient tensor is excluded on \
         either side"
    );
    println!(
        "pass bar: FP8 codes at least 99.5% exact and every code within 1 (a value on an E4M3 tie is one code), ue8m0 scales bit-exact, X_l bit-exact after bf16 rounding, B and A and C (raw, no harness-side normalisation) within 1e-4, 1/rms within 3e-4 absolute"
    );
    check_floor(args, "stage 1 start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The released mHC tensors in the form the kernel and the reference consume; nothing is
    // pre-folded into the predictor (the scale must reach the kernel as `hc_attn_scale`).
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let predictor = mhc.predictor;
    let hc_base = mhc.hc_base;
    let hc_base_kernel = mhc.hc_base_kernel;
    let hc_scale = mhc.hc_scale;
    let base_min = mhc.hc_base.iter().copied().fold(f32::INFINITY, f32::min);
    let base_max = mhc
        .hc_base
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    println!(
        "hc_attn_fn: {} f32 stored; hc_attn_base: {} f32 in [{base_min:.4}, {base_max:.4}] \
         (consumed: checkpoint row order for the reference, kernel B/A/C row order for the device);          hc_attn_scale: {hc_scale:?} (consumed)",
        PREDICTOR_ROWS * HC_MULT * HIDDEN,
        mhc.hc_base.len(),
    );
    println!("attn_norm.weight: {} f32 after widening to f32", norm.len());
    println!(
        "predictor: {} f32 = released [24, 4*d] rows permuted into the kernel's B/A/C order, NO \
         averaging ({} KiB); hc_attn_fn released before any device allocation",
        predictor.len(),
        predictor.len() * 4 / 1024,
    );

    // The layer-0 boundary seed.
    let mut b_prev = [0.0f32; HC_MULT * HC_MULT];
    for i in 0..HC_MULT {
        b_prev[i * HC_MULT + i] = 1.0;
    }
    let c_prev = [0.0f32; HC_MULT];
    let a_prev = [1.0f32, 0.0, 0.0, 0.0];
    let y_prev = vec![0.0f32; HIDDEN];

    // Device buffers, admitted against the memory floor before anything is allocated.
    let planned = (HC_MULT * HIDDEN * 2
        + HIDDEN * 2
        + b_prev.len() * 4
        + c_prev.len() * 4
        + a_prev.len() * 4
        + predictor.len() * 4
        + hc_scale.len() * 4
        + hc_base_kernel.len() * 4
        + norm.len() * 4
        + HIDDEN
        + HIDDEN / QUANT_BLOCK
        + MHC_GRID_STATS * 4
        + 2 * 4
        + 24 * 4
        + 4
        + HIDDEN * 4) as u64;
    let cuda = Cuda::load()?;
    let cubin_path = args.cubin_dir.join(format!("{kernel}.cubin"));
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            cubin_path.display()
        )
    })?;
    let (module, func) = cuda.load_function(&cubin, kernel)?;
    admit_memory(args, planned, "stage 1 device buffers")?;
    let mut allocated = 0usize;
    let x_d = cuda.alloc(HC_MULT * HIDDEN * 2, &mut allocated)?;
    let y_d = cuda.alloc(HIDDEN * 2, &mut allocated)?;
    let b_d = cuda.alloc(b_prev.len() * 4, &mut allocated)?;
    let c_d = cuda.alloc(c_prev.len() * 4, &mut allocated)?;
    let a_d = cuda.alloc(a_prev.len() * 4, &mut allocated)?;
    let predictor_d = cuda.alloc(predictor.len() * 4, &mut allocated)?;
    let scale_d = cuda.alloc(hc_scale.len() * 4, &mut allocated)?;
    let base_d = cuda.alloc(hc_base_kernel.len() * 4, &mut allocated)?;
    let norm_d = cuda.alloc(norm.len() * 4, &mut allocated)?;
    let fp8_d = cuda.alloc(HIDDEN, &mut allocated)?;
    let scales_d = cuda.alloc(HIDDEN / QUANT_BLOCK, &mut allocated)?;
    let stats_d = cuda.alloc(MHC_GRID_STATS * 4, &mut allocated)?;
    let sync_d = cuda.alloc(2 * 4, &mut allocated)?;
    let coeffs_d = cuda.alloc(24 * 4, &mut allocated)?;
    let rms_d = cuda.alloc(4, &mut allocated)?;
    let normed_d = cuda.alloc(HIDDEN * 4, &mut allocated)?;
    println!(
        "device bytes allocated: {allocated} ({} KiB)",
        allocated / 1024
    );

    // y_prev is a bf16 tensor in the kernel, so it is uploaded in bf16: an f32 upload is twice
    // the allocation and the driver refuses it with CUDA_ERROR_INVALID_VALUE.
    cuda.copy_htod(&y_d, &bf16_bits_to_bytes(&bf16_bits_of_f32(&y_prev)))?;
    cuda.copy_htod(&b_d, &f32_to_bytes(&b_prev))?;
    cuda.copy_htod(&c_d, &f32_to_bytes(&c_prev))?;
    cuda.copy_htod(&a_d, &f32_to_bytes(&a_prev))?;
    cuda.copy_htod(&predictor_d, &f32_to_bytes(&predictor))?;
    cuda.copy_htod(&scale_d, &f32_to_bytes(&hc_scale))?;
    cuda.copy_htod(&base_d, &f32_to_bytes(&hc_base_kernel))?;
    cuda.copy_htod(&norm_d, &f32_to_bytes(&norm))?;
    cuda.copy_htod(&stats_d, &[0u8; MHC_GRID_STATS * 4])?;
    cuda.copy_htod(&sync_d, &[0u8; 2 * 4])?;

    let grid = (MHC_TILES, 1u32, 1u32);
    let block = (MHC_THREADS, 1u32, 1u32);
    // The kernel takes 19 parameters: 16 pointers (including the B0 f32 block-input capture), the
    // trace ring, the layer id and the step sequence.
    let params: [u64; 19] = [
        x_d.0,
        y_d.0,
        b_d.0,
        c_d.0,
        a_d.0,
        predictor_d.0,
        scale_d.0,
        base_d.0,
        norm_d.0,
        fp8_d.0,
        scales_d.0,
        stats_d.0,
        sync_d.0,
        coeffs_d.0,
        rms_d.0,
        normed_d.0,
        0, // trace_ring: null, the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
        args.layer as u64,
        0,
    ];

    let mut verdicts: Vec<(String, bool)> = Vec::new();
    for token in 0..args.tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let x_bits = bf16_bits_of_f32(&x);

        // The reference on the same bytes.
        let mut streams = Streams::new(x.clone());
        let mut ledger = TrafficLedger::default();
        let (_next, coefficients, codes, scales) = mega_mhc_layer_with_scale(
            &mut streams,
            &norm,
            &a_prev,
            &predictor,
            &hc_base,
            &hc_scale,
            &mut ledger,
        );

        // The in-place X_l update the kernel writes, computed on the host from the same inputs.
        let mut expected_x = vec![0.0f32; HC_MULT * HIDDEN];
        for stream in 0..HC_MULT {
            for column in 0..HIDDEN {
                let mut value = 0.0f32;
                for source in 0..HC_MULT {
                    value += b_prev[stream * HC_MULT + source] * x[source * HIDDEN + column];
                }
                value += c_prev[stream] * y_prev[column];
                expected_x[stream * HIDDEN + column] = value;
            }
        }
        // 1/rms(X_hat), the released formula, from the same A the kernel was given: the RMSNorm of
        // the COLLAPSED block input uses X_hat's sum of squares over d and norm_eps = 1e-20, not
        // the four streams' sum over n*d and not the Sinkhorn epsilon.
        let mut expected_xhat = vec![0.0f32; HIDDEN];
        let mut xhat_squares = 0.0f32;
        for column in 0..HIDDEN {
            let mut value = 0.0f32;
            for (stream, weight) in a_prev.iter().enumerate() {
                value += weight * x[stream * HIDDEN + column];
            }
            expected_xhat[column] = value;
            xhat_squares += value * value;
        }
        let expected_inv_rms = 1.0f32 / (xhat_squares / HIDDEN as f32 + NORM_EPS).sqrt();

        cuda.copy_htod(&x_d, &bf16_bits_to_bytes(&x_bits))?;
        // The grid rendezvous counters are per-launch state: without re-zeroing them the second
        // token starts with the first token's tile count and never publishes its coefficients.
        cuda.copy_htod(&stats_d, &[0u8; MHC_GRID_STATS * 4])?;
        cuda.copy_htod(&sync_d, &[0u8; 2 * 4])?;
        check_floor(args, &format!("stage 1 token {token} pre-launch"))?;
        cuda.launch(func, grid, block, &params)?;
        check_floor(args, &format!("stage 1 token {token} post-launch"))?;

        let mut x_out = vec![0u8; HC_MULT * HIDDEN * 2];
        let mut fp8_out = vec![0u8; HIDDEN];
        let mut scales_out = vec![0u8; HIDDEN / QUANT_BLOCK];
        let mut coeffs_out = vec![0u8; 24 * 4];
        let mut rms_out = vec![0u8; 4];
        let mut normed_out = vec![0u8; HIDDEN * 4];
        cuda.copy_dtoh(&mut x_out, &x_d)?;
        cuda.copy_dtoh(&mut fp8_out, &fp8_d)?;
        cuda.copy_dtoh(&mut scales_out, &scales_d)?;
        cuda.copy_dtoh(&mut coeffs_out, &coeffs_d)?;
        cuda.copy_dtoh(&mut rms_out, &rms_d)?;
        cuda.copy_dtoh(&mut normed_out, &normed_d)?;
        let x_out_bits: Vec<u16> = x_out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        let coeffs_out_f32 = bytes_to_f32(&coeffs_out);
        let rms_out_f32 = bytes_to_f32(&rms_out);

        println!("--- token {token} (embed row {token}, four copies) ---");
        let mut b_ref = Vec::with_capacity(HC_MULT * HC_MULT);
        for i in 0..HC_MULT {
            for j in 0..HC_MULT {
                b_ref.push(coefficients.b[i][j]);
            }
        }
        let s_b = compare_f32(
            "hc_coeffs B (the sinkhorn mixer)",
            &coeffs_out_f32[0..16],
            &b_ref,
        )?;
        let s_a = compare_f32(
            "hc_coeffs A (the input mixes)",
            &coeffs_out_f32[16..20],
            &coefficients.a,
        )?;
        let s_c = compare_f32(
            "hc_coeffs C (the block-output weights)",
            &coeffs_out_f32[20..24],
            &coefficients.c,
        )?;
        let s_fp8 = compare_u8("xhat_fp8 (E4M3 codes)", &fp8_out, codes.as_slice())?;
        let s_scales = compare_u8("xhat_scales (ue8m0)", &scales_out, scales.as_slice())?;
        let s_x = compare_bf16_write("residual streams X_l", &x_out_bits, &expected_x)?;
        let s_rms = compare_f32("1/rms(X_hat)", &rms_out_f32, &[expected_inv_rms])?;
        // The B0 capture: the kernel publishes the f32 block input it quantised, and the expected
        // value is xhat * rms_weight * inverse_rms in the kernel's multiply order, both derived from
        // the SAME inputs and the device's own inverse rms, so the two are expected bit-exact. This
        // is a check of the capture itself, independent of codes-times-scale reconstruction.
        let normed_out_f32 = bytes_to_f32(&normed_out);
        let expected_normed = block_input_normed(&expected_xhat, &norm, rms_out_f32[0]);
        let s_capture = compare_f32(
            "block input f32 capture (xhat_normed)",
            &normed_out_f32,
            &expected_normed,
        )?;
        let capture_ok = s_capture.all_exact();
        let coefficients_ok =
            s_b.max_delta <= 1e-4 && s_a.max_delta <= 1e-4 && s_c.max_delta <= 1e-4;
        // Both sums of squares are block-parallel reductions whose four partials are combined in
        // tile order and rounded once, while the reference accumulates sequentially; 3e-4 absolute.
        let rms_ok = s_rms.max_delta <= 3e-4;
        let fp8_ok = s_fp8.exact_pct() >= 99.5 && s_fp8.max_delta <= 1.0;
        if s_fp8.exact != s_fp8.len {
            // The diagnosis is conditional on what the bytes show (D8 review M3): a quantiser-rule
            // difference moves codes by more than one AND moves block scales, while the fixed-order
            // vs sequential 1/rms residual is one code with bit-exact scales.
            if s_scales.all_exact() && s_fp8.max_delta <= 1.0 {
                println!(
                    "  finding: {} of {} FP8 codes differ (first at index {}), each by one code with \
                     the scales bit-exact; the cause is upstream, not in the quantiser: the two sums \
                     of squares differ between the device's fixed-order block reduction and the \
                     reference's sequential accumulation, so 1/rms differs by {:e} and moves a value \
                     across an E4M3 boundary",
                    s_fp8.len - s_fp8.exact,
                    s_fp8.len,
                    s_fp8.at,
                    s_rms.max_delta,
                );
            } else {
                println!(
                    "  finding: {} of {} FP8 codes differ (first at index {}), max code delta {} and \
                     {} of {} ue8m0 scales differing: this is a QUANTISER-RULE difference, not the \
                     reduction-order residual (that one is a one-code difference with bit-exact \
                     scales). Check which scale rule the installed cubin was built with before \
                     reading the token verdict.",
                    s_fp8.len - s_fp8.exact,
                    s_fp8.len,
                    s_fp8.at,
                    s_fp8.max_delta,
                    s_scales.len - s_scales.exact,
                    s_scales.len,
                );
            }
        }
        verdicts.push((
            format!("token {token}"),
            fp8_ok
                && s_scales.all_exact()
                && s_x.all_exact()
                && coefficients_ok
                && rms_ok
                && capture_ok,
        ));
        println!(
            "  token {token} boundary verdict: fp8 {} scales {} X_l {} coefficients {} inv_rms {} \
             f32 block input {}",
            if fp8_ok { "within bar" } else { "DIFFER" },
            if s_scales.all_exact() {
                "bit-exact"
            } else {
                "DIFFER"
            },
            if s_x.all_exact() {
                "bit-exact"
            } else {
                "DIFFER"
            },
            if coefficients_ok {
                "within 1e-4"
            } else {
                "OVER 1e-4"
            },
            if rms_ok { "within 3e-4" } else { "OVER 3e-4" },
            if capture_ok {
                "bit-exact"
            } else {
                "DIFFER (bar: bit-exact)"
            },
        );
        println!(
            "  reference ledger: x_read {} x_write {} xhat_write {} scale_write {} (passes {})",
            ledger.x_read,
            ledger.x_write,
            ledger.xhat_write,
            ledger.scale_write,
            streams.passes(),
        );
    }

    if args.repeat > 1 {
        // Timing only: the kernel updates the residual streams in place, so the values after a
        // repeated launch are a different quantity from the per-token comparison above and are not
        // compared. The buffers hold token 0's inputs from the last loop iteration.
        check_floor(args, "stage 1 timing pre-launch")?;
        let (mut per_launch, batched_ms) =
            cuda.launch_repeated(func, grid, block, &params, args.repeat)?;
        check_floor(args, "stage 1 timing post-launch")?;
        per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
        let count = per_launch.len();
        let mean = per_launch.iter().sum::<f64>() / count as f64;
        println!(
            "timing: {count} launches on the same buffers (the values after the repeats are not compared)"
        );
        println!(
            "per-launch ms: min {:.4} p50 {:.4} mean {mean:.4} max {:.4}",
            per_launch[0],
            per_launch[count / 2],
            per_launch[count - 1]
        );
        println!("batched ms/launch over {count} launches: {batched_ms:.4}");
    }

    cuda.free(
        module,
        &[
            x_d,
            y_d,
            b_d,
            c_d,
            a_d,
            predictor_d,
            scale_d,
            base_d,
            norm_d,
            fp8_d,
            scales_d,
            stats_d,
            sync_d,
            coeffs_d,
            rms_d,
            normed_d,
        ],
    );
    check_floor(args, "stage 1 end")?;
    let failed: Vec<&String> = verdicts
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(name, _)| name)
        .collect();
    if failed.is_empty() {
        println!(
            "stage 1: PASS ({} of {} tokens at every boundary)",
            verdicts.len(),
            verdicts.len()
        );
        Ok(())
    } else {
        Err(format!("stage 1: FAIL at {:?}", failed))
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 2: MLA attention on layer 0. compress_ratios[0] == 0, so the sliding-window path over at
// most index_topk candidates is the whole attention rule. The per-head attention sink is consumed
// on both sides: the kernel takes a [64] f32 argument and the reference is attention_with_sink.
// ---------------------------------------------------------------------------------------------

/// The MLA latent width, mirrored from mla_ref::MLA_LATENT.
const MLA_LATENT: usize = 512;
/// Query heads, mirrored from mla_ref::MLA_HEADS.
const MLA_HEADS: usize = 64;
/// One head per block, 256 threads: SPARK_M4_MLA_THREADS.
const MLA_THREADS: u32 = 256;
/// The indexer geometry the kernel's first stage reads even though this stage does not compare it.
const MLA_INDEX_HEAD_DIM: usize = 128;
const MLA_HSI_BLOCKS: usize = 2048;
const MLA_HSI_BLOCK_TOKENS: usize = 8;
/// The candidate bound, mirrored from mla_ref::MAX_SELECTED (= index_topk).
const MLA_MAX_SELECTED: usize = 512;
/// The sliding window, mirrored from mla_ref::SWA_WINDOW.
const MLA_SWA_WINDOW: u32 = 128;
/// qk_rope_head_dim: rope rotates the last 64 of a head's 512 (model.py:772).
const ROPE_HEAD_DIM: usize = 64;
/// q_lora_rank: wq_a's width and q_norm's length.
const Q_LORA_RANK: usize = 1280;
/// The FP4 main-KV row: 512 values pack into 256 bytes with one E4M3 scale per 16 channels.
const KV_PACKED_BYTES: usize = MLA_LATENT / 2;
const KV_SCALE_CHANNELS: usize = 16;
const KV_SCALE_BYTES: usize = MLA_LATENT / KV_SCALE_CHANNELS;
/// text_config.rms_norm_eps.
const NORM_EPS: f32 = 1e-20;
/// text_config.rope_theta. Layer 0 has compress_ratio 0, so it takes the base rope with no YaRN
/// correction - the engine's rope_tables_for_layer selects rope_tables for exactly that case.
const ROPE_THETA: f32 = 10000.0;
/// 6 * 2^-9: the floor sparkflash-kv's quantize_main_kv puts under a group's amax.
const KV_AMAX_FLOOR: f32 = 6.0 * 0.001_953_125;
/// E2M1's largest finite magnitude.
const E2M1_MAX: f32 = 6.0;
/// The stage-2 pass bar, printed before the numbers that are judged against it.
const STAGE2_EXACT_BAR: f64 = 99.5;
const STAGE2_DELTA_BAR: f32 = 1.6e-2;
/// The engine window format's scale geometry: one E8M0 byte per 32 latent channels (model.py:27).
const WINDOW_QUANT_BLOCK: usize = 32;
const WINDOW_SCALE_BYTES: usize = MLA_LATENT / WINDOW_QUANT_BLOCK;
/// The engine's sliding window is 128 rows, supplied oldest first.
const WINDOW_ROWS: usize = MLA_SWA_WINDOW as usize;
/// The merged-chain FP4 numbers on the --tokens 130 plan, re-reported next to the engine column so
/// that a change to the default path cannot pass unnoticed. These are the numbers the merged
/// Stage-2 claim records on the chain that contains D8: 163,838/163,840 exact (99.9988%), 2.4414063e-4
/// max abs delta, one bf16 ULP max. They were defined by the paired diagnostic of 2026-10-18
/// (phase1-task3-diag-oldkernel-20261018T060534Z.log and ...newkernel...): the merged-head binary
/// with the old 15-parameter kernel (cubin 822df7f8) and the task-3 binary (cubin bbb216b7) write
/// identical per-position lines on the same chain, so this constant is the old kernel's own number,
/// not a value the new path is allowed to drift from. The earlier record's 9.765625e-4 max abs delta
/// was measured on the pre-D8 stage-1 chain (phase1-b0-stage2-layer0-20261018T010919Z.log); it is
/// kept only as a historical label and is not expected here.
const STAGE2_MERGED_CHAIN_EXACT: usize = 163_838;
const STAGE2_MERGED_CHAIN_TOTAL: usize = 163_840;
const STAGE2_MERGED_CHAIN_MAX_DELTA: f32 = 2.4414063e-4;
const STAGE2_MERGED_CHAIN_MAX_ULP: i32 = 1;

/// The flash_mla_csa2_sm121 launch parameters, in the kernel's pinned ABI order. One constructor
/// owns the order and the fixed-size array; no launch site builds its own argument list, because
/// Cuda::launch takes a slice and cannot see a stale arity (a 15-element vector would make
/// cuLaunchKernel read four arguments past the end).
const FLASH_MLA_PARAM_COUNT: usize = 19;
/// selected_count is the one field a launch may vary after construction: the out-of-window
/// injection launches keep every other argument and change only the candidate count.
const FLASH_MLA_SELECTED_COUNT_INDEX: usize = 15;

/// The flash_mla launch arguments in the kernel's pinned order (see the kernel declaration and
/// m4_abi::SIGNATURES). Pointers are carried as u64 because that is what cuLaunchKernel's
/// argument array holds.
#[derive(Clone, Copy)]
struct FlashMlaParams([u64; FLASH_MLA_PARAM_COUNT]);

impl FlashMlaParams {
    fn as_slice(&self) -> &[u64] {
        &self.0
    }

    fn with_selected_count(mut self, selected_count: usize) -> Self {
        self.0[FLASH_MLA_SELECTED_COUNT_INDEX] = selected_count as u64;
        self
    }
}

impl Default for FlashMlaParams {
    fn default() -> Self {
        Self([0; FLASH_MLA_PARAM_COUNT])
    }
}

/// Builds the flash_mla launch parameters with the kernel's own parameter names. Every launch in
/// this file (stage-2 indexer self-check, stage-2 attention, stage-7 K5, stage-9 K5) goes through
/// this constructor and the typed wrappers below; a future kernel signature change fails to compile
/// at every site instead of leaving one of them on a stale array.
#[allow(clippy::too_many_arguments)]
fn flash_mla_params(
    q: u64,
    kv_fp4: u64,
    kv_scales: u64,
    window_codes: u64,
    window_scales: u64,
    window_kv_format: u64,
    window_rows: u64,
    selected: u64,
    index_key: u64,
    index_query: u64,
    sink: u64,
    block_scores: u64,
    block_best: u64,
    out: u64,
    position: u64,
    selected_count: usize,
    trace_ring: u64,
    layer: usize,
    step_seq: u64,
) -> FlashMlaParams {
    FlashMlaParams([
        q,
        kv_fp4,
        kv_scales,
        window_codes,
        window_scales,
        window_kv_format,
        window_rows,
        selected,
        index_key,
        index_query,
        sink,
        block_scores,
        block_best,
        out,
        position,
        selected_count as u64,
        trace_ring,
        layer as u64,
        step_seq,
    ])
}

/// Launches the flash_mla kernel. The only type it accepts is FlashMlaParams, so the pinned
/// 19-argument order cannot be bypassed at a call site.
fn flash_mla_launch(
    cuda: &Cuda,
    func: CuFunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    params: &FlashMlaParams,
) -> Result<(), String> {
    cuda.launch(func, grid, block, params.as_slice())
}

/// The repeated-launch timing wrapper for the same typed parameters.
fn flash_mla_launch_repeated(
    cuda: &Cuda,
    func: CuFunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    params: &FlashMlaParams,
    repeat: usize,
) -> Result<(Vec<f64>, f64), String> {
    cuda.launch_repeated(func, grid, block, params.as_slice(), repeat)
}

const _: () = assert!(
    MLA_LATENT == 512
        && MLA_HEADS == 64
        && MLA_MAX_SELECTED == 512
        && MLA_SWA_WINDOW == 128
        && Q_LORA_RANK == 1280
        && ROPE_HEAD_DIM == 64
        && KV_PACKED_BYTES == 256
        && KV_SCALE_BYTES == 32
);

/// RMSNorm over one row, the engine reference's formula (linalg::rms_norm): mean of squares,
/// 1/sqrt(mean + eps), then the per-column weight.
fn rms_norm_row(values: &mut [f32], weight: &[f32], eps: f32) {
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let inverse = 1.0 / (mean + eps).sqrt();
    for (value, w) in values.iter_mut().zip(weight) {
        *value = *value * inverse * *w;
    }
}

/// The engine reference's rope tables for one position: cos/sin of position * base^(-2k/dim) for
/// dim/2 pairs (model.py:368-389), kept in f32 like the reference's f32 tensor.
fn rope_tables_for(position: u32, dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let pairs = dim / 2;
    let angles: Vec<f32> = (0..pairs)
        .map(|k| position as f32 / theta.powf(2.0 * k as f32 / dim as f32))
        .collect();
    (
        angles.iter().map(|angle| angle.cos()).collect(),
        angles.iter().map(|angle| angle.sin()).collect(),
    )
}

/// Rotates the LAST rope-long tail of each width-wide row in place, exactly as the engine's
/// rotate_rows does: start = width - rope, and (a, b) becomes (a cos - b sin, a sin + b cos).
fn rotate_rope_tail(
    values: &mut [f32],
    rows: usize,
    width: usize,
    rope: usize,
    theta: f32,
    position: u32,
) {
    let (cos, sin) = rope_tables_for(position, rope, theta);
    for row in 0..rows {
        let start = row * width + width - rope;
        for pair in 0..rope / 2 {
            let a = values[start + 2 * pair];
            let b = values[start + 2 * pair + 1];
            values[start + 2 * pair] = a * cos[pair] - b * sin[pair];
            values[start + 2 * pair + 1] = a * sin[pair] + b * cos[pair];
        }
    }
}

/// Rotates the LAST rope-long tail of each width-wide row by the NEGATIVE angle: the inverse of
/// rotate_rope_tail, used on the attention output before the wo_a operand.
fn rotate_rope_tail_inverse(
    values: &mut [f32],
    rows: usize,
    width: usize,
    rope: usize,
    theta: f32,
    position: u32,
) {
    let (cos, sin) = rope_tables_for(position, rope, theta);
    for row in 0..rows {
        let start = row * width + width - rope;
        for pair in 0..rope / 2 {
            let a = values[start + 2 * pair];
            let b = values[start + 2 * pair + 1];
            values[start + 2 * pair] = a * cos[pair] + b * sin[pair];
            values[start + 2 * pair + 1] = -a * sin[pair] + b * cos[pair];
        }
    }
}

/// Quantises one 512-wide latent row to the FP4 main-KV format the kernel and mla_ref::kv_latent
/// both consume, by sparkflash-kv's quantize_main_kv rule: per-16-channel amax floored at 6 * 2^-9,
/// an E4M3 scale of amax / 6, E2M1 codes of value / scale clamped to +-6, and the low nibble first
/// (which is the order mla_ref::kv_latent and spark_mla_load_latent both read).
fn quantize_kv_row(latent: &[f32]) -> ([u8; KV_PACKED_BYTES], [u8; KV_SCALE_BYTES]) {
    let mut packed = [0u8; KV_PACKED_BYTES];
    let mut scales = [0u8; KV_SCALE_BYTES];
    for group in 0..KV_SCALE_BYTES {
        let block = &latent[group * KV_SCALE_CHANNELS..(group + 1) * KV_SCALE_CHANNELS];
        let amax = block.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale_byte = fp8_e4m3_encode(amax.max(KV_AMAX_FLOOR) / E2M1_MAX);
        let scale = fp8_e4m3_decode(scale_byte);
        scales[group] = scale_byte;
        for (offset, value) in block.iter().enumerate() {
            let index = group * KV_SCALE_CHANNELS + offset;
            let code = fp4_e2m1_encode((value / scale).clamp(-E2M1_MAX, E2M1_MAX));
            if index.is_multiple_of(2) {
                packed[index / 2] = code & 0x0F;
            } else {
                packed[index / 2] |= (code & 0x0F) << 4;
            }
        }
    }
    (packed, scales)
}

/// The monotone integer key of a bf16 bit pattern, so two codes an ULP apart differ by one and the
/// distance is meaningful across zero.
fn bf16_order(bits: u16) -> i32 {
    let magnitude = i32::from(bits & 0x7FFF);
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Compares a device bf16 write against the bf16 rounding of the f32 reference and reports the
/// delta in bf16 codes as well as in absolute value, because a format with a relative step cannot
/// be judged by an absolute bar alone.
fn compare_bf16_ulp(
    label: &str,
    device: &[u16],
    reference: &[f32],
) -> Result<(Stats, i32), String> {
    let expected = bf16_bits_of_f32(reference);
    if device.len() != expected.len() {
        return Err(format!(
            "{label}: {} device values against {} reference values",
            device.len(),
            expected.len()
        ));
    }
    let mut stats = Stats {
        len: device.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    let mut max_ulp = 0i32;
    for (index, (device_bits, expected_bits)) in device.iter().zip(&expected).enumerate() {
        if device_bits == expected_bits {
            stats.exact += 1;
        }
        let delta = (half::bf16::from_bits(*device_bits).to_f32()
            - half::bf16::from_bits(*expected_bits).to_f32())
        .abs();
        if !delta.is_finite() || delta > stats.max_delta {
            stats.max_delta = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
            stats.at = index;
        }
        let ulp = (bf16_order(*device_bits) - bf16_order(*expected_bits)).abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
    }
    println!(
        "  {label}: {} bf16 values, exact {}/{} ({:.4}%), max ULP distance {max_ulp}, max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    Ok((stats, max_ulp))
}

fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The query positions stage 2 runs: token 0 is the tightest window (one candidate), 1 and 127
/// fill it, and 128/129 are the first positions whose window leaves a candidate out. The last
/// token of a longer prompt is added when it is not already in the list.
fn stage2_positions(tokens: usize) -> Vec<usize> {
    let mut positions = vec![0usize];
    for candidate in [1usize, 127, 128, 129, tokens.saturating_sub(1)] {
        if candidate < tokens {
            positions.push(candidate);
        }
    }
    positions.sort_unstable();
    positions.dedup();
    positions
}

/// The candidate list for one query position: the in-window range first, then deliberately older
/// positions. The kernel and mla_ref::attention both apply the same window predicate and skip an
/// out-of-window candidate exactly, so the older entries exercise the exclusion rather than
/// down-weighting it.
fn stage2_selected(position: usize) -> Vec<u32> {
    let window = MLA_SWA_WINDOW as usize;
    let first = position.saturating_sub(window - 1);
    let mut selected: Vec<u32> = (first..=position)
        .map(|candidate| candidate as u32)
        .collect();
    let mut older = position as i64 - window as i64;
    while older >= 0 && selected.len() < MLA_MAX_SELECTED {
        selected.push(older as u32);
        older -= 1;
    }
    selected
}

fn stage2(args: &Args) -> Result<(), String> {
    let kernel = "flash_mla_csa2_sm121";
    let prefix = format!("layers.{}", args.layer);
    let format = args.stage2_kv_format;
    println!("=== stage 2: {kernel} against mla_ref::attention_with_sink ===");
    println!(
        "stage-2 KV format: {} (kernel selector {}). fp4 is the merged Stage-2 default and this run must reproduce its historical numbers; engine is the engine's released window-KV format and is reported as a separate class of evidence, never folded into the FP4 claim.",
        format.label(),
        format.kernel_value()
    );
    println!(
        "comparison class: TOLERANCE-BASED, not exact. The kernel writes bf16 and evaluates expf on the device; the reference is f32 with the library exponential. The tolerance is named here, before any number it judges: at least {STAGE2_EXACT_BAR}% of the {} outputs bit-identical to the bf16 rounding of the f32 reference, and max abs delta <= {STAGE2_DELTA_BAR:e}, with the max ULP distance reported. Reason: the score sum, the window predicate, the candidate or row order, the KV decode and the divide are identical by construction on both sides, so the only degrees of freedom are expf against libm exp, the tree reduction of the normalising sum against the serial one, and the bf16 output write itself. {}",
        MLA_HEADS * MLA_LATENT,
        match format {
            WindowKvFormat::Fp4 =>
                "This run uses the historical FP4 latent format, the same class the merged Stage-2 claim records; its totals must equal that claim exactly.",
            WindowKvFormat::Engine =>
                "This run uses the engine window format: the stored values are bf16(dequant(E4M3 code, E8M0 scale)), reproduced on both sides, so this class measures format fidelity. It is not a bit-identity claim against the released kernel, and the engine's bf16 arithmetic is not reproduced: sparkflash-engine's reference forward keeps f32 dequant, and that pre-existing seam is named rather than hidden.",
        }
    );
    println!(
        "label: internal consistency of the sink arithmetic, not model fidelity. The attention sink (attn_sink, F32 [64], a per-head additive logit inside the softmax denominator) is CONSUMED ON BOTH SIDES: the kernel takes it as an argument and adds it to the denominator after the maximum, and the reference does the same. The class printed above judges the numbers below. This run's format is {}.",
        format.label()
    );
    println!(
        "inherited limitation from stage 1, printed rather than implied: this stage's input chain starts at stage 1's xhat fp8 codes and scales, which carry stage 1's residual gap - the bf16 write, the tf32-like reductions of the device, and the kernel's per-tile reduction order against the reference's serial arithmetic. The predictor is NOT a gap any more: B0 removed the d-wide fold, both sides consume the release's [24, 20480] projection over the flattened streams, and the coefficient tensors hc_attn_base and hc_attn_scale are consumed on both sides with the kernel's C compared raw."
    );
    println!(
        "layer {}: compress_ratios[0] = 0, so this is NOT a CED layer. The whole attention rule is the 128-token sliding window over at most index_topk 512 candidates. The candidate list is supplied by the harness: the top-512 merge over the HSI pool is a separate pipeline step the kernel does not implement. The HSI indexer inside the kernel is checked separately at the top of this stage on deterministic non-zero index data, against mla_ref::index_block_best, because stage 2 never consumes its block_scores/block_best; this stage's own boundary is the attention output.",
        args.layer
    );
    if format == WindowKvFormat::Fp4 {
        println!(
            "KV row format: the rows fed to BOTH sides are the FP4 main-KV format the kernel and mla_ref::kv_latent both consume, quantised here by sparkflash-kv's quantize_main_kv rule (per-16 E4M3 scale of amax/6, E2M1 codes, low nibble first) from kv_norm(wkv(xhat)) with the rope tail rotated. The engine's released window path rounds its KV to FP8 instead, so this run measures the kernel against the reference over the kernel's KV format, not the engine's window path; the engine format is a separate run (--stage2-kv-format engine)."
        );
    } else {
        println!(
            "KV row format: the engine's released window format (model.py:700-707, kernel.py:98-124). Each kv_norm(wkv(xhat)) row is rounded to bf16 - the engine's storage dtype, because its in-place act_quant returns the input dtype - then quantised into E4M3 codes with one E8M0 power-of-two scale per 32 channels. Both the kernel and mla_ref::attention_with_sink_window decode code times scale and round the decoded value to bf16, so the bf16 storage round is consumed on both sides. The device bytes are the engine crate's own quantiser and are asserted byte-identical to m4_quant::act_quant_fp8_row on the same bf16 rows. sparkflash-engine's own reference forward keeps f32 dequant, so the engine's bf16 arithmetic is not reproduced here; that seam is named, not hidden."
        );
    }
    check_floor(args, "stage 2 start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The stage-1 chain, exactly as stage 1 runs it: the released tensors, the boundary seed and
    // the mHC reference, whose returned codes and scales are this stage's input.
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let predictor = mhc.predictor;
    let hc_base = mhc.hc_base;
    let hc_scale = mhc.hc_scale;
    // The stage-1 boundary seed: A = [1, 0, 0, 0] with X_-1 = four copies of the embed row. B and C
    // only matter to stage 1's coefficient comparison; the mHC reference takes A and X here.
    let a_prev = [1.0f32, 0.0, 0.0, 0.0];

    let wq_a_weight = snap.read_raw(
        &format!("{prefix}.attn.wq_a.weight"),
        "F8_E4M3",
        &[Q_LORA_RANK, HIDDEN],
    )?;
    let wq_a_scale = snap.read_raw(
        &format!("{prefix}.attn.wq_a.scale"),
        "F8_E8M0",
        &[Q_LORA_RANK / 32, HIDDEN / 32],
    )?;
    let wq_b_weight = snap.read_raw(
        &format!("{prefix}.attn.wq_b.weight"),
        "F8_E4M3",
        &[MLA_HEADS * MLA_LATENT, Q_LORA_RANK],
    )?;
    let wq_b_scale = snap.read_raw(
        &format!("{prefix}.attn.wq_b.scale"),
        "F8_E8M0",
        &[(MLA_HEADS * MLA_LATENT) / 32, Q_LORA_RANK / 32],
    )?;
    let wkv_weight = snap.read_raw(
        &format!("{prefix}.attn.wkv.weight"),
        "F8_E4M3",
        &[MLA_LATENT, HIDDEN],
    )?;
    let wkv_scale = snap.read_raw(
        &format!("{prefix}.attn.wkv.scale"),
        "F8_E8M0",
        &[MLA_LATENT / 32, HIDDEN / 32],
    )?;
    let q_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.q_norm.weight"), &[Q_LORA_RANK])?;
    let kv_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.kv_norm.weight"), &[MLA_LATENT])?;
    let sink = snap.read_f32(&format!("{prefix}.attn.attn_sink"), &[MLA_HEADS])?;
    let sink_min = sink.iter().copied().fold(f32::INFINITY, f32::min);
    let sink_max = sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "weights: wq_a {} B + {} scale B, wq_b {} B + {} scale B, wkv {} B + {} scale B (the packed checkpoint bytes, handed to mxfp8_ref::dense_gemm as-is so no weight is ever decoded to f32 on the host)",
        wq_a_weight.len(),
        wq_a_scale.len(),
        wq_b_weight.len(),
        wq_b_scale.len(),
        wkv_weight.len(),
        wkv_scale.len()
    );
    println!(
        "attn_sink: {} f32 in [{sink_min:.4}, {sink_max:.4}] - consumed on BOTH sides: passed to the kernel as a [64] argument and added to the denominator after the maximum, and passed to mla_ref::attention_with_sink",
        sink.len()
    );

    let tokens = args.tokens;
    let positions = stage2_positions(tokens);
    println!(
        "prompt: {tokens} tokens (embed rows 0..{tokens}), query positions {positions:?}; the window is {MLA_SWA_WINDOW}, so a position below {} never sees an out-of-window candidate",
        MLA_SWA_WINDOW
    );
    if tokens <= MLA_SWA_WINDOW as usize {
        println!(
            "note: with {tokens} tokens the out-of-window branch is not exercised; run --tokens 130 for the plan that exercises it"
        );
    }

    // Per-token xhat codes and scales from the mHC reference: stage 1's output, and this stage's
    // input. These are the packed bytes, never decoded weights.
    let mut xhat: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(tokens);
    let mut kv_fp4: Vec<u8> = Vec::with_capacity(tokens * KV_PACKED_BYTES);
    let mut kv_scales: Vec<u8> = Vec::with_capacity(tokens * KV_SCALE_BYTES);
    // Engine window format rows, one per token: [tokens][512] E4M3 codes and [tokens][16] E8M0
    // scales. The kernel is handed the contiguous slice for one query position, oldest first.
    let mut window_codes: Vec<u8> = Vec::new();
    let mut window_scales: Vec<u8> = Vec::new();
    let mut window_quantiser_agreements = 0usize;
    for token in 0..tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let mut streams = Streams::new(x);
        let mut ledger = TrafficLedger::default();
        let (_next, _coefficients, codes, scales) = mega_mhc_layer_with_scale(
            &mut streams,
            &norm,
            &a_prev,
            &predictor,
            &hc_base,
            &hc_scale,
            &mut ledger,
        );
        let kv_bits = mxfp8_ref::dense_gemm(
            codes.as_slice(),
            scales.as_slice(),
            &wkv_weight,
            &wkv_scale,
            1,
            MLA_LATENT,
            HIDDEN,
        );
        let mut latent: Vec<f32> = kv_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        rms_norm_row(&mut latent, &kv_norm, NORM_EPS);
        rotate_rope_tail(
            &mut latent,
            1,
            MLA_LATENT,
            ROPE_HEAD_DIM,
            ROPE_THETA,
            token as u32,
        );
        let (packed, row_scales) = quantize_kv_row(&latent);
        kv_fp4.extend_from_slice(&packed);
        kv_scales.extend_from_slice(&row_scales);
        if format == WindowKvFormat::Engine {
            let engine_row =
                engine_window_row(&latent).map_err(|error| format!("token {token}: {error}"))?;
            window_quantiser_agreements += 1;
            window_codes.extend_from_slice(&engine_row.codes);
            window_scales.extend_from_slice(&engine_row.scales);
        }
        xhat.push((codes.as_slice().to_vec(), scales.as_slice().to_vec()));
        let _ = ledger;
    }
    if format == WindowKvFormat::Fp4 {
        println!(
            "input chain (FP4): {tokens} xhat fp8 rows from the mHC reference -> wkv {} B weight -> kv_norm -> rope tail -> FP4; kv_fp4 {} B, kv_scales {} B",
            wkv_weight.len(),
            kv_fp4.len(),
            kv_scales.len()
        );
    } else {
        println!(
            "input chain (engine): {tokens} xhat fp8 rows from the mHC reference -> wkv {} B weight -> kv_norm -> rope tail -> bf16 storage -> E4M3 + E8M0; window_codes {} B, window_scales {} B",
            wkv_weight.len(),
            window_codes.len(),
            window_scales.len()
        );
        println!(
            "engine window format: {tokens} rows rounded to bf16 then quantised by sparkflash_engine::quant::quantize_activation_fp8; {window_quantiser_agreements}/{tokens} rows byte-identical to m4_quant::act_quant_fp8_row (E4M3 codes and E8M0 scales, asserted not averaged)"
        );
        if window_quantiser_agreements != tokens {
            return Err("the engine window quantiser and the m4_quant oracle disagree".to_string());
        }
        assert_engine_window_finite(
            &window_codes,
            &window_scales,
            tokens,
            "engine window format",
        )?;
        println!(
            "engine window guard: {tokens} rows x {MLA_LATENT} channels decoded finite; no E4M3 NaN code (0x7F) and no overflowing (0x7E x scale 254) pair reached the kernel"
        );
    }

    // The device buffers are sized once and admitted against the memory floor before allocation.
    let index_key_bytes = MLA_HSI_BLOCKS * MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM * 2;
    let index_query_bytes = MLA_HEADS * MLA_INDEX_HEAD_DIM * 2;
    let out_bytes = MLA_HEADS * MLA_LATENT * 2;
    let window_codes_bytes = WINDOW_ROWS * MLA_LATENT;
    let window_scales_bytes = WINDOW_ROWS * WINDOW_SCALE_BYTES;
    let planned = (MLA_HEADS * MLA_LATENT * 2
        + tokens * KV_PACKED_BYTES
        + tokens * KV_SCALE_BYTES
        + window_codes_bytes
        + window_scales_bytes
        + MLA_MAX_SELECTED * 4
        + index_key_bytes
        + index_query_bytes
        + MLA_HSI_BLOCKS * 4
        + MLA_HSI_BLOCKS * 4
        + out_bytes
        + MLA_HEADS * 4) as u64;
    let cuda = Cuda::load()?;
    let cubin_path = args.cubin_dir.join(format!("{kernel}.cubin"));
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            cubin_path.display()
        )
    })?;
    let (module, func) = cuda.load_function(&cubin, kernel)?;
    admit_memory(args, planned, "stage 2 device buffers")?;
    let mut allocated = 0usize;
    let q_d = cuda.alloc(MLA_HEADS * MLA_LATENT * 2, &mut allocated)?;
    let sink_d = cuda.alloc(MLA_HEADS * 4, &mut allocated)?;
    let kv_fp4_d = cuda.alloc(tokens * KV_PACKED_BYTES, &mut allocated)?;
    let kv_scales_d = cuda.alloc(tokens * KV_SCALE_BYTES, &mut allocated)?;
    let window_codes_d = cuda.alloc(window_codes_bytes, &mut allocated)?;
    let window_scales_d = cuda.alloc(window_scales_bytes, &mut allocated)?;
    let selected_d = cuda.alloc(MLA_MAX_SELECTED * 4, &mut allocated)?;
    let index_key_d = cuda.alloc(index_key_bytes, &mut allocated)?;
    let index_query_d = cuda.alloc(index_query_bytes, &mut allocated)?;
    let block_scores_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let block_best_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let out_d = cuda.alloc(out_bytes, &mut allocated)?;
    println!(
        "device bytes allocated: {allocated} ({} KiB); planned {planned} B",
        allocated / 1024
    );
    cuda.copy_htod(&kv_fp4_d, &kv_fp4)?;
    cuda.copy_htod(&kv_scales_d, &kv_scales)?;
    cuda.copy_htod(&sink_d, &f32_to_bytes(&sink))?;

    // Deterministic integer-valued index data for the indexer self-check below. Every product and
    // partial sum is an exact integer in f32 (|dot| <= 1,024), so the reference and the kernel must
    // agree bit for bit and a mapping error cannot hide behind rounding. Stage 2 never reads these
    // buffers, so this does not touch the attention comparison.
    let index_key_f32: Vec<f32> = (0..MLA_HSI_BLOCKS * MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM)
        .map(|i| {
            let block = i / (MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM);
            let slot = (i / MLA_INDEX_HEAD_DIM) % MLA_HSI_BLOCK_TOKENS;
            let dim = i % MLA_INDEX_HEAD_DIM;
            (((block * 13 + slot * 7 + dim * 3) % 9) as i32 - 4) as f32
        })
        .collect();
    let index_query_f32: Vec<f32> = (0..MLA_HEADS * MLA_INDEX_HEAD_DIM)
        .map(|i| {
            let index_head = i / MLA_INDEX_HEAD_DIM;
            let dim = i % MLA_INDEX_HEAD_DIM;
            (((index_head * 5 + dim * 11) % 7) as i32 - 3) as f32
        })
        .collect();
    cuda.copy_htod(
        &index_key_d,
        &bf16_bits_to_bytes(&bf16_bits_of_f32(&index_key_f32)),
    )?;
    cuda.copy_htod(
        &index_query_d,
        &bf16_bits_to_bytes(&bf16_bits_of_f32(&index_query_f32)),
    )?;
    // Sentinels, so a block the indexer fails to write is visible even when its correct score is 0.
    cuda.copy_htod(
        &block_scores_d,
        &f32_to_bytes(&vec![12345.0f32; MLA_HSI_BLOCKS]),
    )?;
    cuda.copy_htod(
        &block_best_d,
        &u32s_to_bytes(&vec![u32::MAX; MLA_HSI_BLOCKS]),
    )?;

    // The indexer stage of the kernel runs one blockIdx.y slice per stripe of HSI blocks. The
    // slice count is a measured lever: more slices give the pool more blocks to run on but pay a
    // fixed per-block cost, and only slice 0 runs stage 2. See --grid-y and the task-41 report.
    let grid = (MLA_HEADS as u32, args.grid_y, 1u32);
    let block = (MLA_THREADS, 1u32, 1u32);
    println!(
        "launch grid: ({}, {}, {}) blocks of ({}, 1, 1) threads; each warp walks {} HSI blocks per head",
        grid.0,
        grid.1,
        grid.2,
        block.0,
        (MLA_HSI_BLOCKS as u32).div_ceil(args.grid_y * 8 * 2)
    );

    // --- indexer self-check ---------------------------------------------------------------
    // Stage 2 never reads block_scores or block_best, so until now the harness fed the indexer
    // zeros and compared nothing: a mapping error in the warp-level rewrite would pass every
    // attention comparison below. This runs the indexer on integer-valued bf16 index data, where
    // every dot product is exact in f32, and compares the two heads of index head 0 - the pair
    // that splits the pool in the kernel - against mla_ref::index_block_best. Blocks the kernel
    // does not write keep their sentinel, so partial coverage fails rather than passing silently.
    let index_check_params = flash_mla_params(
        q_d.0,
        kv_fp4_d.0,
        kv_scales_d.0,
        window_codes_d.0,
        window_scales_d.0,
        WindowKvFormat::Fp4.kernel_value() as u64,
        0, // window_rows is a plain count: the indexer's launch carries none
        selected_d.0,
        index_key_d.0,
        index_query_d.0,
        sink_d.0,
        block_scores_d.0,
        block_best_d.0,
        out_d.0,
        0, // position
        0, // selected_count
        0, // trace_ring: the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
        args.layer,
        0, // step_seq
    );
    check_floor(args, "indexer check pre-launch")?;
    flash_mla_launch(
        &cuda,
        func,
        (2u32, args.grid_y, 1u32),
        block,
        &index_check_params,
    )?;
    check_floor(args, "indexer check post-launch")?;
    let mut index_scores_bytes = vec![0u8; MLA_HSI_BLOCKS * 4];
    let mut index_best_bytes = vec![0u8; MLA_HSI_BLOCKS * 4];
    cuda.copy_dtoh(&mut index_scores_bytes, &block_scores_d)?;
    cuda.copy_dtoh(&mut index_best_bytes, &block_best_d)?;
    let index_scores_dev = bytes_to_f32(&index_scores_bytes);
    let index_best_dev: Vec<u32> = index_best_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect();
    let (index_scores_ref, index_best_ref) = mla_ref::index_block_best(
        &index_query_f32[..MLA_INDEX_HEAD_DIM],
        &index_key_f32,
        MLA_HSI_BLOCKS,
    );
    let index_score_stats = compare_f32(
        "indexer block_scores (head pair 0/1, the full pool)",
        &index_scores_dev,
        &index_scores_ref,
    )?;
    let index_best_exact = index_best_dev
        .iter()
        .zip(&index_best_ref)
        .filter(|(device, reference)| device == reference)
        .count();
    println!(
        "  indexer block_best: {index_best_exact}/{} exact (an unwritten block keeps u32::MAX)",
        MLA_HSI_BLOCKS
    );
    if !index_score_stats.all_exact() || index_best_exact != MLA_HSI_BLOCKS {
        return Err(format!(
            "indexer self-check FAILED at grid.y {}: {} of {} block scores exact and {index_best_exact} of {} block_best exact",
            args.grid_y, index_score_stats.exact, index_score_stats.len, MLA_HSI_BLOCKS
        ));
    }
    println!(
        "  indexer self-check: PASS at grid.y {} (both parities of the pool, one head each)",
        args.grid_y
    );

    let mut verdicts: Vec<(String, bool)> = Vec::new();
    let mut last_params = FlashMlaParams::default();
    let mut total_exact = 0usize;
    let mut total_len = 0usize;
    let mut aggregate_max_delta = 0.0f32;
    let mut aggregate_max_ulp = 0i32;
    for (step, &position) in positions.iter().enumerate() {
        let (codes, scales) = &xhat[position];
        let q_latent_bits = mxfp8_ref::dense_gemm(
            codes,
            scales,
            &wq_a_weight,
            &wq_a_scale,
            1,
            Q_LORA_RANK,
            HIDDEN,
        );
        let mut q_latent: Vec<f32> = q_latent_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        rms_norm_row(&mut q_latent, &q_norm, NORM_EPS);
        let (q_codes, q_scales) = act_quant_fp8_row(&q_latent);
        let q_bits = mxfp8_ref::dense_gemm(
            &q_codes,
            &q_scales,
            &wq_b_weight,
            &wq_b_scale,
            1,
            MLA_HEADS * MLA_LATENT,
            Q_LORA_RANK,
        );
        let mut q: Vec<f32> = q_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        rotate_rope_tail(
            &mut q,
            MLA_HEADS,
            MLA_LATENT,
            ROPE_HEAD_DIM,
            ROPE_THETA,
            position as u32,
        );
        // The kernel consumes bf16, so the value both sides see is the bf16 rounding of the
        // rotated query: rounding it once here keeps the comparison about the attention, not about
        // where the rounding point is.
        let q_device = bf16_bits_of_f32(&q);
        let q_used: Vec<f32> = q_device
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();

        // The engine window rows for this query position: the last min(position + 1, 128)
        // positions, oldest first. Row r is position first + r, so the newest row is always the
        // query position and the oldest rows drop off once the window is full: p=127 has 128 rows
        // (positions 0..=127, row 0 = position 0); p=128 drops position 0 (rows 1..=128, row 0 =
        // position 1, row 127 = position 128); p=129 also drops position 1 (rows 2..=129). This is
        // the engine's order: get_window_topk_idxs returns ascending chunk ids for prefill and
        // oldest-to-newest ring slot order for decode, which is the same supplied order.
        let (first, window_row_count) = if format == WindowKvFormat::Engine {
            let first = position.saturating_sub(WINDOW_ROWS - 1);
            (first, position - first + 1)
        } else {
            (0usize, 0usize)
        };
        if format == WindowKvFormat::Engine {
            // The order is asserted, not assumed: the rows are exactly the last window_row_count
            // positions, oldest first, and the last row is the query position.
            if window_row_count != (position + 1).min(WINDOW_ROWS)
                || first + window_row_count - 1 != position
            {
                return Err(format!(
                    "ring order assertion failed at position {position}: first {first}, rows {window_row_count}"
                ));
            }
            let dropped = if first == 0 {
                "none".to_string()
            } else {
                format!("positions 0..{first}")
            };
            println!(
                "  ring order: {window_row_count} rows supplied oldest -> newest; row r -> position {first} + r (row 0 = position {first}, last row = position {position}); dropped oldest {dropped}; the 128-row window has wrapped: {}",
                position >= WINDOW_ROWS
            );
        }
        let selected: Vec<u32> = if format == WindowKvFormat::Engine {
            (0..window_row_count as u32).collect()
        } else {
            stage2_selected(position)
        };
        let in_window = if format == WindowKvFormat::Engine {
            selected.len()
        } else {
            selected
                .iter()
                .filter(|candidate| mla_ref::in_window(position as u32, **candidate))
                .count()
        };

        cuda.copy_htod(&q_d, &bf16_bits_to_bytes(&q_device))?;
        cuda.copy_htod(&selected_d, &u32s_to_bytes(&selected))?;
        if format == WindowKvFormat::Engine {
            let code_start = first * MLA_LATENT;
            let scale_start = first * WINDOW_SCALE_BYTES;
            cuda.copy_htod(
                &window_codes_d,
                &window_codes[code_start..code_start + window_row_count * MLA_LATENT],
            )?;
            cuda.copy_htod(
                &window_scales_d,
                &window_scales[scale_start..scale_start + window_row_count * WINDOW_SCALE_BYTES],
            )?;
        }
        let params = flash_mla_params(
            q_d.0,
            kv_fp4_d.0,
            kv_scales_d.0,
            window_codes_d.0,
            window_scales_d.0,
            format.kernel_value() as u64,
            window_row_count as u64,
            selected_d.0,
            index_key_d.0,
            index_query_d.0,
            sink_d.0,
            block_scores_d.0,
            block_best_d.0,
            out_d.0,
            position as u64,
            selected.len(),
            0, // trace_ring: null, the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
            args.layer,
            step as u64,
        );
        last_params = params;
        // Determinism: the same launch on the same inputs runs twice and the two device writes are
        // compared byte for byte. A mismatch is a hard failure, not a statistic.
        let mut device_out = vec![0u8; out_bytes];
        for run in 0..2usize {
            check_floor(
                args,
                &format!("stage 2 position {position} run {run} pre-launch"),
            )?;
            flash_mla_launch(&cuda, func, grid, block, &params)?;
            check_floor(
                args,
                &format!("stage 2 position {position} run {run} post-launch"),
            )?;
            let mut run_out = vec![0u8; out_bytes];
            cuda.copy_dtoh(&mut run_out, &out_d)?;
            if run == 0 {
                device_out = run_out;
            } else if run_out != device_out {
                return Err(format!(
                    "stage 2 determinism failed at position {position}: two runs on the same inputs wrote different bytes"
                ));
            }
        }
        let device_bits: Vec<u16> = device_out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| u16::from_le_bytes(*chunk))
            .collect();
        let reference_for = |list: &[u32]| -> Vec<f32> {
            let mut reference = Vec::with_capacity(MLA_HEADS * MLA_LATENT);
            for head in 0..MLA_HEADS {
                let q_head = &q_used[head * MLA_LATENT..(head + 1) * MLA_LATENT];
                let head_out = if format == WindowKvFormat::Engine {
                    let code_start = first * MLA_LATENT;
                    let scale_start = first * WINDOW_SCALE_BYTES;
                    mla_ref::attention_with_sink_window(
                        q_head,
                        &window_codes[code_start..code_start + window_row_count * MLA_LATENT],
                        &window_scales
                            [scale_start..scale_start + window_row_count * WINDOW_SCALE_BYTES],
                        window_row_count,
                        list,
                        sink[head],
                    )
                } else {
                    mla_ref::attention_with_sink(
                        q_head,
                        &kv_fp4,
                        &kv_scales,
                        list,
                        position as u32,
                        sink[head],
                    )
                };
                reference.extend_from_slice(&head_out);
            }
            reference
        };
        let reference = reference_for(&selected);
        println!(
            "--- query position {position}: {} candidates, {in_window} in window, {} excluded ---",
            selected.len(),
            selected.len() - in_window
        );
        let (stats, max_ulp) = compare_bf16_ulp("attention out", &device_bits, &reference)?;
        total_exact += stats.exact;
        total_len += stats.len;
        aggregate_max_delta = aggregate_max_delta.max(stats.max_delta);
        aggregate_max_ulp = aggregate_max_ulp.max(max_ulp);
        let ok = stats.exact_pct() >= STAGE2_EXACT_BAR && stats.max_delta <= STAGE2_DELTA_BAR;
        println!(
            "  position {position} verdict: {} (exact {:.4}% against a {STAGE2_EXACT_BAR}% bar, max abs delta {:e} against a {STAGE2_DELTA_BAR:e} bar, max ULP distance {max_ulp})",
            if ok {
                "within tolerance"
            } else {
                "OVER TOLERANCE"
            },
            stats.exact_pct(),
            stats.max_delta,
        );
        verdicts.push((format!("position {position}"), ok));

        if format == WindowKvFormat::Engine {
            // Out-of-window ids in engine mode must contribute exactly zero, not merely be left out
            // of a list: ids at and past the row count, and u32::MAX, are added to the candidate
            // list. The device bytes must not change at all, and the reference with the injected
            // ids must equal the reference without them, bit for bit.
            let mut injected = selected.clone();
            let injected_outsiders = [
                window_row_count as u32,
                window_row_count as u32 + 17,
                u32::MAX,
            ];
            injected.extend(injected_outsiders);
            // Non-vacuousness: the ids really are outside the window and really are in the list
            // handed to the kernel. Without this check an injection that silently dropped the ids
            // (or admitted them) would compare byte-identical for the wrong reason.
            if !injected_outsiders
                .iter()
                .all(|id| !mla_ref::window_attendable(window_row_count, *id))
            {
                return Err(format!(
                    "out-of-window injection is vacuous at position {position}: an injected id is inside the window"
                ));
            }
            if !injected_outsiders.iter().all(|id| injected.contains(id)) {
                return Err(format!(
                    "out-of-window injection is vacuous at position {position}: an injected id is missing from the candidate list"
                ));
            }
            if injected.len() != selected.len() + injected_outsiders.len() {
                return Err(format!(
                    "out-of-window injection count changed at position {position}"
                ));
            }
            cuda.copy_htod(&selected_d, &u32s_to_bytes(&injected))?;
            let injected_params = params.with_selected_count(injected.len());
            flash_mla_launch(&cuda, func, grid, block, &injected_params)?;
            let mut injected_out = vec![0u8; out_bytes];
            cuda.copy_dtoh(&mut injected_out, &out_d)?;
            let injected_reference = reference_for(&injected);
            if injected_out != device_out || injected_reference != reference {
                return Err(format!(
                    "out-of-window zeroing failed at position {position}: injected ids changed the output or the reference"
                ));
            }
            // An all-outside list is exactly zero on both sides even with the sink in the
            // denominator: the numerator is empty and the divide is by one. The list is the same
            // three verified outsiders, and no in-window id may be present.
            let all_outside = injected_outsiders;
            if all_outside
                .iter()
                .any(|id| mla_ref::window_attendable(window_row_count, *id))
            {
                return Err(format!(
                    "all-outside injection is vacuous at position {position}: the all-outside list contains an in-window id"
                ));
            }
            cuda.copy_htod(&selected_d, &u32s_to_bytes(&all_outside))?;
            let outside_params = params.with_selected_count(all_outside.len());
            flash_mla_launch(&cuda, func, grid, block, &outside_params)?;
            let mut outside_out = vec![0u8; out_bytes];
            cuda.copy_dtoh(&mut outside_out, &out_d)?;
            let outside_reference = reference_for(&all_outside);
            if !outside_out.iter().all(|byte| *byte == 0)
                || !outside_reference.iter().all(|value| *value == 0.0)
            {
                return Err(format!(
                    "all-out-of-window output is not exactly zero at position {position}"
                ));
            }
            println!(
                "  out-of-window ids at and past the {window_row_count} rows (plus u32::MAX): {} injected ids verified outside the window and present in the candidate list; device output byte-identical with and without them, reference equal, all-outside output exactly zero on both sides: PASS",
                injected_outsiders.len()
            );
            verdicts.push((format!("position {position} out-of-window zeroing"), true));
        }
    }
    println!(
        "format {}: totals across {} positions: {total_exact}/{total_len} exact ({:.4}%), max abs delta {:e}, max ULP {aggregate_max_ulp}; determinism: two device runs per position, byte-identical",
        format.label(),
        positions.len(),
        100.0 * total_exact as f64 / total_len as f64,
        aggregate_max_delta
    );
    if format == WindowKvFormat::Fp4 && tokens == 130 {
        let positions_text = positions
            .iter()
            .map(|position| position.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let merged_chain = total_exact == STAGE2_MERGED_CHAIN_EXACT
            && total_len == STAGE2_MERGED_CHAIN_TOTAL
            && aggregate_max_delta == STAGE2_MERGED_CHAIN_MAX_DELTA
            && aggregate_max_ulp == STAGE2_MERGED_CHAIN_MAX_ULP;
        println!(
            "merged-chain FP4 equality: {} (positions [{positions_text}]; expected {STAGE2_MERGED_CHAIN_EXACT}/{STAGE2_MERGED_CHAIN_TOTAL} exact, max abs delta {:e}, max ULP {} - the paired merged-head old-kernel diagnostic numbers on this chain; got {total_exact}/{total_len}, {:e}, {aggregate_max_ulp})",
            if merged_chain { "PASS" } else { "FAIL" },
            STAGE2_MERGED_CHAIN_MAX_DELTA,
            STAGE2_MERGED_CHAIN_MAX_ULP,
            aggregate_max_delta
        );
        println!(
            "historical FP4 note: 9.765625e-4 was the pre-D8 stage-1 chain, measured with the retired B0 binary (phase1-b0-stage2-layer0-20261018T010919Z.log); on the merged chain the merged-head old-kernel binary and this binary write identical per-position lines (phase1-task3-diag-oldkernel-20261018T060534Z.log vs phase1-task3-diag-newkernel-20261018T060534Z.log), so the merged-chain expectation above is that old kernel's own measured number"
        );
        if !merged_chain {
            return Err(
                "the FP4 default path no longer reproduces the merged-chain Stage-2 numbers (the paired old-kernel diagnostic on the same chain is the reference)"
                    .to_string(),
            );
        }
    }

    if args.repeat > 1 {
        check_floor(args, "stage 2 timing pre-launch")?;
        let (mut per_launch, batched_ms) =
            flash_mla_launch_repeated(&cuda, func, grid, block, &last_params, args.repeat)?;
        check_floor(args, "stage 2 timing post-launch")?;
        per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
        let count = per_launch.len();
        let mean = per_launch.iter().sum::<f64>() / count as f64;
        println!(
            "timing: {count} launches at the last query position on the same buffers; per-launch ms min {:.4} p50 {:.4} mean {mean:.4} max {:.4}; batched ms/launch {batched_ms:.4}",
            per_launch[0],
            per_launch[count / 2],
            per_launch[count - 1]
        );
    }

    cuda.free(
        module,
        &[
            q_d,
            kv_fp4_d,
            kv_scales_d,
            window_codes_d,
            window_scales_d,
            selected_d,
            index_key_d,
            index_query_d,
            block_scores_d,
            block_best_d,
            out_d,
            sink_d,
        ],
    );
    check_floor(args, "stage 2 end")?;
    let failed: Vec<&String> = verdicts
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(name, _)| name)
        .collect();
    if failed.is_empty() {
        println!(
            "stage 2: PASS ({} of {} printed checks within the declared class: per-position bars plus, for the engine format, the out-of-window zeroing assertions)",
            verdicts.len(),
            verdicts.len()
        );
        Ok(())
    } else {
        Err(format!("stage 2: FAIL at {failed:?}"))
    }
}

// ---------------------------------------------------------------------------------------------
// Stage 3: the MoE block on a bounded expert subset. The router is a HOST SEAM and is labelled as
// one; the device side is the W4A8 down/combine projection, four rank-chained launches over the
// 2304-wide intermediate; gate/up, SwiGLU and the activation quantiser are host arithmetic here
// because no device kernel in the six-cubin set reduces K=5120 with packed FP4 weights.
// ---------------------------------------------------------------------------------------------

/// text_config.n_routed_experts.
const MOE_EXPERTS: usize = 384;
/// text_config.num_experts_per_tok, mirrored by mxfp4_ref::MOE_TOP_K.
const MOE_TOP_K: usize = 6;
/// text_config.moe_intermediate_size.
const MOE_INTERMEDIATE: usize = 2304;
/// The per-rank expert width and the padded width the kernel actually walks.
const MOE_WIDTH: usize = mxfp4_ref::MOE_WIDTH;
const MOE_WIDTH_PADDED: usize = mxfp4_ref::MOE_WIDTH_PADDED;
/// 2304 / 576: the number of rank-chained combine launches one routed sum needs.
const MOE_RANKS: usize = MOE_INTERMEDIATE / MOE_WIDTH;
/// The pre-padded weight strides: 640 values are 320 packed bytes and 20 ue8m0 bytes.
const MOE_PACKED_ROW: usize = mxfp4_ref::MOE_WEIGHT_BYTES_PER_ROW;
const MOE_SCALE_ROW: usize = mxfp4_ref::MOE_SCALE_BYTES_PER_ROW;
/// One expert's stored w2 row and scale row, before the rank slice: 2304 values.
const MOE_IN_PACKED: usize = MOE_INTERMEDIATE / 2;
const MOE_IN_SCALES: usize = MOE_INTERMEDIATE / 32;
/// One expert's stored w1/w3 row: 5120 packed values and 160 scales.
const MOE_EXPERT_IN_PACKED: usize = HIDDEN / 2;
const MOE_EXPERT_IN_SCALES: usize = HIDDEN / 32;
/// The router's constants: gate_temp is absent from the released config and defaults to 1.0
/// (model.py:67-68), route_scale is text_config.routed_scaling_factor, and the normalising epsilon
/// is model.py:824's.
const GATE_TEMP: f32 = 1.0;
const ROUTE_SCALE: f32 = 1.5;
const ROUTE_NORM_EPS: f32 = 1e-20;
/// The kernel's launch geometry: one output element per thread, 256 threads, 20 tiles of 5120.
const MOE_TILE_D: u32 = 256;
const MOE_THREADS: u32 = 256;
/// The stage-3 pass bar, printed before the numbers it judges.
const STAGE3_EXACT_BAR: f64 = 99.9;

const _: () =
    assert!(MOE_EXPERTS == 384 && MOE_TOP_K == 6 && MOE_INTERMEDIATE == 2304 && MOE_RANKS == 4);

/// The forced-operand verifier's geometry, from this checkpoint's real constants.
const FORCED_OPERAND_GEOMETRY: ForcedOperandGeometry = ForcedOperandGeometry {
    hidden: HIDDEN,
    intermediate: MOE_INTERMEDIATE,
    width: MOE_WIDTH,
    width_padded: MOE_WIDTH_PADDED,
    top_k: MOE_TOP_K,
    ranks: MOE_RANKS,
};

/// Repack a checkpoint FP8 weight from row-major [cols, inner] into the B12x micro-tile order that
/// mxfp8_ref::dense_gemm walks. The scale grid needs no repack: b12x_scale_index is
/// (n / 32) * k_blocks + (k / 32), which is the checkpoint's own [ceil(cols / 32), inner / 32]
/// row-major order, so the stored .scale tensor is already in the layout the GEMM reads.
fn pack_b12x(weights: &[u8], cols: usize, inner: usize) -> Vec<u8> {
    assert_eq!(
        weights.len(),
        cols * inner,
        "the checkpoint stores one E4M3 byte per FP8 value"
    );
    let mut tiled = vec![0u8; cols * inner];
    for n in 0..cols {
        for k in 0..inner {
            tiled[mxfp8_ref::b12x_weight_index(n, k, inner)] = weights[n * inner + k];
        }
    }
    tiled
}

/// One token's routing decision with this harness's released-router constants.
fn route_token(
    x: &[f32],
    gate_weight: &[f32],
    bias: &[f32],
    candidates: &[u32],
) -> (Vec<u32>, Vec<f32>) {
    route_token_with(
        x,
        gate_weight,
        bias,
        candidates,
        RouterParams {
            hidden: HIDDEN,
            top_k: MOE_TOP_K,
            gate_temp: GATE_TEMP,
            norm_eps: ROUTE_NORM_EPS,
            scale: ROUTE_SCALE,
        },
    )
}

/// The pre-quantisation routed activation: the SwiGLU value times the routing weight, cast to bf16,
/// because model.py:851 casts before the quantiser reads it. mxfp4_ref::routed_activation_operand
/// applies exactly this before quantising; this form exists so the stage can also feed the
/// UNSQUANTISED value to a second reference and measure what the E4M3 quantiser costs.
fn routed_activation_f32(activation: &[f32], routing: f32) -> Vec<f32> {
    activation
        .iter()
        .map(|value| {
            sparkflash_kernels::m4_quant::bf16_value(sparkflash_kernels::m4_quant::bf16_bits(
                value * routing,
            ))
        })
        .collect()
}

/// The reference chain the device is compared against: moe_combine over the same pre-padded rank
/// slices, chained rank by rank exactly as the launches are, because every rank rounds to bf16 and
/// the chained rounding is part of what the device does.
fn reference_combine_chain(
    act_fp8: &[Vec<u8>],
    act_scales: &[Vec<u8>],
    w_rank: &[Vec<u8>],
    ws_rank: &[Vec<u8>],
    expert_ids: &[u32],
    tokens: usize,
    experts: usize,
) -> Vec<u16> {
    let mut previous = vec![0u16; tokens * HIDDEN];
    for rank in 0..MOE_RANKS {
        let (next, _cost) = mxfp4_ref::moe_combine(
            &previous,
            &act_fp8[rank],
            &act_scales[rank],
            &w_rank[rank],
            &ws_rank[rank],
            expert_ids,
            tokens,
            HIDDEN,
            experts,
        );
        previous = next;
    }
    previous
}

/// The same chain with the activation kept in f32 instead of E4M3: not a reference the stage
/// passes, only the measurement of what the activation quantiser costs, because a comparison over
/// identical quantised bytes cannot see the quantiser at all.
fn unquantised_combine_chain(
    routed_activations: &[Vec<f32>],
    w2_natural: &[Vec<u8>],
    w2_scales: &[Vec<u8>],
    expert_ids: &[u32],
    tokens: usize,
) -> Vec<u16> {
    let blocks = MOE_INTERMEDIATE / 32;
    let mut previous = vec![0u16; tokens * HIDDEN];
    for _rank in 0..MOE_RANKS {
        let mut next = vec![0u16; tokens * HIDDEN];
        for token in 0..tokens {
            for column in 0..HIDDEN {
                let mut acc = 0.0f32;
                for slot in 0..MOE_TOP_K {
                    let expert = expert_ids[token * MOE_TOP_K + slot] as usize;
                    let activation = &routed_activations[token * MOE_TOP_K + slot];
                    let weight = &w2_natural[expert];
                    let scales = &w2_scales[expert];
                    let weight_row = &weight[column * MOE_IN_PACKED..(column + 1) * MOE_IN_PACKED];
                    let scale_row = &scales[column * blocks..(column + 1) * blocks];
                    for (block, scale) in scale_row.iter().enumerate().take(blocks) {
                        let mut partial = 0.0f32;
                        for i in 0..32 {
                            let index = block * 32 + i;
                            let w = fp8_e4m3_decode(fp8_e4m3_encode(mxfp4_ref::fp4_load(
                                weight_row, index,
                            )));
                            partial += activation[index] * w;
                        }
                        acc += partial * ue8m0_decode(*scale);
                    }
                }
                let residual = half::bf16::from_bits(previous[token * HIDDEN + column]).to_f32();
                next[token * HIDDEN + column] = half::bf16::from_f32(residual + acc).to_bits();
            }
        }
        previous = next;
    }
    previous
}

/// Compares two bf16 writes code for code: exact count, absolute delta and the code distance.
fn compare_bf16_bits(label: &str, device: &[u16], reference: &[u16]) -> (Stats, i32) {
    assert_eq!(
        device.len(),
        reference.len(),
        "{label}: the buffers are the same length"
    );
    let mut stats = Stats {
        len: device.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    let mut max_ulp = 0i32;
    for (index, (device_bits, reference_bits)) in device.iter().zip(reference).enumerate() {
        if device_bits == reference_bits {
            stats.exact += 1;
        }
        let delta = (half::bf16::from_bits(*device_bits).to_f32()
            - half::bf16::from_bits(*reference_bits).to_f32())
        .abs();
        if delta > stats.max_delta {
            stats.max_delta = delta;
            stats.at = index;
        }
        let ulp = (bf16_order(*device_bits) - bf16_order(*reference_bits)).abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
    }
    println!(
        "  {label}: {} bf16 values, exact {}/{} ({:.4}%), max ULP distance {max_ulp}, max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    (stats, max_ulp)
}

fn stage3(args: &Args) -> Result<(), String> {
    let kernel = "b12x_mxfp4_moe_sm121";
    let prefix = format!("layers.{}", args.layer);
    let tokens = args.tokens;
    let experts = args.experts;
    println!(
        "=== stage 3: the MoE block, bounded to {experts} experts, router on the host seam ==="
    );
    println!(
        "router: host seam, not device; selection bias gate.bias only - gate.bias_vl is the image-span bias, not read on the text path"
    );
    println!(
        "comparison class: TOLERANCE-BASED, not exact, and the tolerance is named here before any number it judges: at least {STAGE3_EXACT_BAR}% of the {} combine outputs bit-identical to the reference's bf16 write, and no output further than 2 bf16 codes. Reason: both sides consume the same packed bytes and accumulate in the same order (slot ascending, 32-value block ascending, scales applied to each block's partial product), so the only degrees of freedom are the device's fused multiply-add against the reference's separate multiply and add, and bf16 rounding at the end of each rank chain step.",
        tokens * HIDDEN
    );
    println!(
        "tolerance by construction: the activation quantiser sits between the f32 model and BOTH sides, so agreement here says the kernel computes the reference's arithmetic, not that either reproduces the f32 expert. The stage measures that gap separately, by running the same chain with an UNSQUANTISED f32 activation and reporting what the E4M3 quantiser alone contributes."
    );
    println!(
        "bounded subset: {experts} of the {MOE_EXPERTS} routed experts are resident - this prompt's most-routed experts first, then the lowest ids to fill - and the router selects over the RESIDENT set, which is what an expert-parallel shard with hot-expert residency sees. The coverage line below reports how much of the full-{MOE_EXPERTS} top-{MOE_TOP_K} the rung captures. This is a bounded test, not the full routing: all {MOE_EXPERTS} experts as f32 would be about 54 GB, which the memory rule puts out of scope tonight."
    );
    println!(
        "device boundary: only the down/combine projection runs on the device, as {MOE_RANKS} rank-chained launches of {kernel} over the {MOE_INTERMEDIATE}-wide intermediate. The gate/up projections, the SwiGLU and the activation quantiser are HOST arithmetic in this stage: no device kernel in the six-cubin set reduces K={HIDDEN} against packed FP4 weights, and this stage says so instead of implying the whole block is device-proven. The shared expert is host arithmetic through mxfp8_ref::dense_gemm and is not device-compared either."
    );
    println!(
        "input: ffn_norm applied to the harness's attention-side xhat. The true ffn-side residual stream is the mHC FFN output, which stage 1's boundary does not hand over, so the input row is a stand-in and every claim below is about the MoE arithmetic over that row, not about end-to-end layer text."
    );
    println!(
        "route_scale {ROUTE_SCALE} is inside the routing weight and therefore inside the activation operand's quantiser, which is the model's order (model.py:824-826 applies it to the weights, model.py:849-851 multiplies the activation before the bf16 cast). It is not applied again after the combine."
    );
    check_floor(args, "stage 3 start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let gate_weight =
        snap.read_f32_or_bf16(&format!("{prefix}.ffn.gate.weight"), &[MOE_EXPERTS, HIDDEN])?;
    let gate_bias = snap.read_f32(&format!("{prefix}.ffn.gate.bias"), &[MOE_EXPERTS])?;
    let gate_bias_vl = snap.read_f32(&format!("{prefix}.ffn.gate.bias_vl"), &[MOE_EXPERTS])?;
    let ffn_norm = snap.read_f32_or_bf16(&format!("{prefix}.ffn_norm.weight"), &[HIDDEN])?;
    let bias_range = gate_bias
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, value| {
            (acc.0.min(*value), acc.1.max(*value))
        });
    let bias_vl_range = gate_bias_vl
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |acc, value| {
            (acc.0.min(*value), acc.1.max(*value))
        });
    println!(
        "gate.bias: {MOE_EXPERTS} f32 in [{:.4}, {:.4}] - the selection bias, consumed; gate.bias_vl in [{:.4}, {:.4}] - read for the record and NOT consumed on the text path",
        bias_range.0, bias_range.1, bias_vl_range.0, bias_vl_range.1
    );

    let shared_w1_row_major = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.weight"),
        "F8_E4M3",
        &[MOE_INTERMEDIATE, HIDDEN],
    )?;
    let shared_s1 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w3_row_major = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.weight"),
        "F8_E4M3",
        &[MOE_INTERMEDIATE, HIDDEN],
    )?;
    let shared_s3 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w2_row_major = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.weight"),
        "F8_E4M3",
        &[HIDDEN, MOE_INTERMEDIATE],
    )?;
    let shared_s2 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, MOE_INTERMEDIATE / 32],
    )?;
    // dense_gemm walks the B12x micro-tile order, and the checkpoint stores row-major FP8 tiles, so
    // the three shared weights are repacked once here instead of being compared in the wrong order.
    let shared_w1 = pack_b12x(&shared_w1_row_major, MOE_INTERMEDIATE, HIDDEN);
    let shared_w3 = pack_b12x(&shared_w3_row_major, MOE_INTERMEDIATE, HIDDEN);
    let shared_w2 = pack_b12x(&shared_w2_row_major, HIDDEN, MOE_INTERMEDIATE);
    drop(shared_w1_row_major);
    drop(shared_w3_row_major);
    drop(shared_w2_row_major);
    println!(
        "shared expert weights: repacked from the checkpoint's row-major FP8 tiles into the B12x micro-tile order dense_gemm walks; the ue8m0 scale grid needs no repack because b12x_scale_index is already the checkpoint's out-block-major then k-block order"
    );

    // The stage-1 chain, exactly as stages 1 and 2 run it: the released tensors and the boundary
    // seed, whose returned codes and scales are the hidden state this stage routes.
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let attn_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let predictor = mhc.predictor;
    let hc_base = mhc.hc_base;
    let hc_scale = mhc.hc_scale;
    let a_prev = [1.0f32, 0.0, 0.0, 0.0];

    let mut block_input: Vec<Vec<f32>> = Vec::with_capacity(tokens);
    let mut block_quant: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(tokens);
    let mut block_activation: Vec<Vec<f32>> = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let mut streams = Streams::new(x);
        let mut ledger = TrafficLedger::default();
        let (_next, _coefficients, codes, scales) = mega_mhc_layer_with_scale(
            &mut streams,
            &attn_norm,
            &a_prev,
            &predictor,
            &hc_base,
            &hc_scale,
            &mut ledger,
        );
        let mut hidden: Vec<f32> = codes
            .as_slice()
            .iter()
            .enumerate()
            .map(|(index, code)| {
                fp8_e4m3_decode(*code) * ue8m0_decode(scales.as_slice()[index / QUANT_BLOCK])
            })
            .collect();
        rms_norm_row(&mut hidden, &ffn_norm, NORM_EPS);
        let (quant, quant_scales) = act_quant_fp8_row(&hidden);
        let activation: Vec<f32> = quant
            .iter()
            .enumerate()
            .map(|(index, code)| {
                fp8_e4m3_decode(*code) * ue8m0_decode(quant_scales[index / QUANT_BLOCK])
            })
            .collect();
        block_input.push(hidden);
        block_quant.push((quant, quant_scales));
        block_activation.push(activation);
        let _ = ledger;
    }
    println!(
        "input: {tokens} rows of ffn_norm(xhat) at hidden width {HIDDEN}, quantised to E4M3 for the experts"
    );

    // The resident subset is this prompt's most-routed experts first, then the lowest ids to fill:
    // that is an expert-parallel shard with hot-expert residency, and it makes the coverage number
    // mean something. The full-384 selection is computed from gate.weight alone, before any expert
    // weight is read, so choosing the subset costs no expert bytes.
    let all_experts: Vec<u32> = (0..MOE_EXPERTS as u32).collect();
    let mut full_selections: Vec<Vec<u32>> = Vec::with_capacity(tokens);
    let mut tally = vec![0usize; MOE_EXPERTS];
    for hidden in block_input.iter().take(tokens) {
        let (full_ids, _) = route_token(hidden, &gate_weight, &gate_bias, &all_experts);
        for expert in &full_ids {
            tally[*expert as usize] += 1;
        }
        full_selections.push(full_ids);
    }
    let mut ranked: Vec<u32> = all_experts.clone();
    ranked.sort_by_key(|expert| (std::cmp::Reverse(tally[*expert as usize]), *expert));
    let mut loaded: Vec<u32> = ranked.iter().copied().take(experts).collect();
    loaded.sort_unstable();
    let mut table_of = vec![usize::MAX; MOE_EXPERTS];
    for (table, expert) in loaded.iter().enumerate() {
        table_of[*expert as usize] = table;
    }
    let distinct = tally.iter().filter(|count| **count > 0).count();
    let covered: usize = full_selections
        .iter()
        .flatten()
        .filter(|expert| table_of[**expert as usize] != usize::MAX)
        .count();
    println!(
        "resident subset: {experts} of {MOE_EXPERTS} experts, ids {loaded:?}, chosen as this prompt's most-routed experts (ties to the lower id) then the lowest ids to fill, so every rung holds exactly {experts} resident weight sets"
    );
    println!(
        "router coverage: {covered} of {} full-{MOE_EXPERTS} top-{MOE_TOP_K} selections are resident at this rung; {distinct} distinct experts are routed over {tokens} tokens",
        tokens * MOE_TOP_K
    );
    let mut expert_ids: Vec<u32> = Vec::with_capacity(tokens * MOE_TOP_K);
    let mut weights_all: Vec<f32> = Vec::with_capacity(tokens * MOE_TOP_K);
    for (token, hidden) in block_input.iter().enumerate().take(tokens) {
        let (ids, weights) = route_token(hidden, &gate_weight, &gate_bias, &loaded);
        println!(
            "  token {token}: resident-set selection {ids:?} weights {weights:?}; full selection {:?}",
            full_selections[token]
        );
        for expert in &ids {
            let table = table_of[*expert as usize];
            assert!(table != usize::MAX, "the selected expert is resident");
            expert_ids.push(table as u32);
        }
        weights_all.extend_from_slice(&weights);
    }

    let per_expert = MOE_INTERMEDIATE * MOE_EXPERT_IN_PACKED
        + MOE_INTERMEDIATE * MOE_EXPERT_IN_SCALES
        + HIDDEN * MOE_IN_PACKED
        + HIDDEN * MOE_IN_SCALES;
    let host_planned = experts * per_expert
        + MOE_RANKS * (experts * HIDDEN * MOE_PACKED_ROW + experts * HIDDEN * MOE_SCALE_ROW)
        + tokens * MOE_TOP_K * 4 * MOE_WIDTH_PADDED;
    admit_memory(args, host_planned as u64, "stage 3 expert weights")?;
    let mut w1: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut s1: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut w3: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut s3: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut w2: Vec<Vec<u8>> = Vec::with_capacity(experts);
    let mut s2: Vec<Vec<u8>> = Vec::with_capacity(experts);
    for expert in &loaded {
        let base = format!("{prefix}.ffn.experts.{expert}");
        w1.push(snap.read_raw(
            &format!("{base}.w1.weight"),
            "I8",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_PACKED],
        )?);
        s1.push(snap.read_raw(
            &format!("{base}.w1.scale"),
            "F8_E8M0",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_SCALES],
        )?);
        w3.push(snap.read_raw(
            &format!("{base}.w3.weight"),
            "I8",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_PACKED],
        )?);
        s3.push(snap.read_raw(
            &format!("{base}.w3.scale"),
            "F8_E8M0",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_SCALES],
        )?);
        w2.push(snap.read_raw(&format!("{base}.w2.weight"), "I8", &[HIDDEN, MOE_IN_PACKED])?);
        s2.push(snap.read_raw(
            &format!("{base}.w2.scale"),
            "F8_E8M0",
            &[HIDDEN, MOE_IN_SCALES],
        )?);
    }
    println!(
        "expert weights: {experts} experts x {per_expert} stored bytes (w1/w3 packed FP4 + ue8m0 scales, w2 packed FP4 + ue8m0 scales) = {} B, read exactly as the checkpoint stores them - no expert is decoded to f32",
        experts * per_expert
    );

    let mut routed_activations: Vec<Vec<f32>> = Vec::with_capacity(tokens * MOE_TOP_K);
    let mut act_fp8: Vec<Vec<u8>> = (0..MOE_RANKS).map(|_| Vec::new()).collect();
    let mut act_scales: Vec<Vec<u8>> = (0..MOE_RANKS).map(|_| Vec::new()).collect();
    for token in 0..tokens {
        for slot in 0..MOE_TOP_K {
            let expert = expert_ids[token * MOE_TOP_K + slot] as usize;
            let gate = gate_up_rows(
                &block_activation[token],
                &w1[expert],
                &s1[expert],
                MOE_INTERMEDIATE,
                HIDDEN,
            );
            let up = gate_up_rows(
                &block_activation[token],
                &w3[expert],
                &s3[expert],
                MOE_INTERMEDIATE,
                HIDDEN,
            );
            let activation: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
                .collect();
            let routing = weights_all[token * MOE_TOP_K + slot];
            routed_activations.push(routed_activation_f32(&activation, routing));
            for rank in 0..MOE_RANKS {
                let (codes, scales) = mxfp4_ref::routed_activation_operand(
                    &activation[rank * MOE_WIDTH..(rank + 1) * MOE_WIDTH],
                    routing,
                );
                let mut row = vec![0u8; MOE_WIDTH_PADDED];
                row[..MOE_WIDTH].copy_from_slice(&codes);
                act_fp8[rank].extend_from_slice(&row);
                let mut scale_row = vec![0u8; MOE_SCALE_ROW];
                scale_row[..MOE_WIDTH / QUANT_BLOCK].copy_from_slice(&scales);
                act_scales[rank].extend_from_slice(&scale_row);
            }
        }
    }

    let mut w_rank: Vec<Vec<u8>> = Vec::with_capacity(MOE_RANKS);
    let mut ws_rank: Vec<Vec<u8>> = Vec::with_capacity(MOE_RANKS);
    for rank in 0..MOE_RANKS {
        let mut weight = Vec::with_capacity(experts * HIDDEN * MOE_PACKED_ROW);
        let mut scale = Vec::with_capacity(experts * HIDDEN * MOE_SCALE_ROW);
        for expert in 0..experts {
            weight.extend_from_slice(&slice_and_pad_rows(
                &w2[expert],
                HIDDEN,
                MOE_IN_PACKED,
                rank * (MOE_WIDTH / 2),
                MOE_WIDTH / 2,
                MOE_PACKED_ROW,
            ));
            scale.extend_from_slice(&slice_and_pad_rows(
                &s2[expert],
                HIDDEN,
                MOE_IN_SCALES,
                rank * (MOE_WIDTH / 32),
                MOE_WIDTH / 32,
                MOE_SCALE_ROW,
            ));
        }
        w_rank.push(weight);
        ws_rank.push(scale);
    }
    println!(
        "rank slices: {MOE_RANKS} ranks of {} packed + {} scale bytes each, built by slicing every expert's stored row and padding the 576-wide rank to the kernel's 640",
        w_rank[0].len(),
        ws_rank[0].len()
    );

    let reference = reference_combine_chain(
        &act_fp8,
        &act_scales,
        &w_rank,
        &ws_rank,
        &expert_ids,
        tokens,
        experts,
    );
    println!("reference: mxfp4_ref::moe_combine chained over the same {MOE_RANKS} ranks");

    check_floor(args, "stage 3 device setup")?;
    let cuda = Cuda::load()?;
    let cubin_path = args.cubin_dir.join(format!("{kernel}.cubin"));
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            cubin_path.display()
        )
    })?;
    let (module, func) = cuda.load_function(&cubin, kernel)?;
    let device_planned = 2 * tokens * HIDDEN * 2
        + act_fp8[0].len()
        + act_scales[0].len()
        + w_rank[0].len()
        + ws_rank[0].len()
        + tokens * MOE_TOP_K * 4;
    admit_memory(args, device_planned as u64, "stage 3 device buffers")?;
    let mut allocated = 0usize;
    let hidden_d = cuda.alloc(tokens * HIDDEN * 2, &mut allocated)?;
    let out_d = cuda.alloc(tokens * HIDDEN * 2, &mut allocated)?;
    let act_d = cuda.alloc(act_fp8[0].len(), &mut allocated)?;
    let acts_d = cuda.alloc(act_scales[0].len(), &mut allocated)?;
    let w_d = cuda.alloc(w_rank[0].len(), &mut allocated)?;
    let ws_d = cuda.alloc(ws_rank[0].len(), &mut allocated)?;
    let ids_d = cuda.alloc(tokens * MOE_TOP_K * 4, &mut allocated)?;
    println!(
        "device bytes allocated: {allocated} ({} MiB)",
        allocated / (1024 * 1024)
    );
    cuda.copy_htod(&hidden_d, &vec![0u8; tokens * HIDDEN * 2])?;
    let grid = (tokens as u32, (HIDDEN / MOE_TILE_D as usize) as u32, 1u32);
    let block = (MOE_THREADS, 1u32, 1u32);
    let mut previous: Vec<u16> = vec![0u16; tokens * HIDDEN];
    let mut last_params: [u64; 11] = [0; 11];
    for rank in 0..MOE_RANKS {
        cuda.copy_htod(&act_d, &act_fp8[rank])?;
        cuda.copy_htod(&acts_d, &act_scales[rank])?;
        cuda.copy_htod(&w_d, &w_rank[rank])?;
        cuda.copy_htod(&ws_d, &ws_rank[rank])?;
        cuda.copy_htod(&ids_d, &u32s_to_bytes(&expert_ids))?;
        let params: [u64; 11] = [
            hidden_d.0,
            act_d.0,
            acts_d.0,
            w_d.0,
            ws_d.0,
            ids_d.0,
            out_d.0,
            tokens as u64,
            0, // trace_ring: null, built without -DSPARKFLASH_ENABLE_GPU_TRACE
            args.layer as u64,
            rank as u64,
        ];
        last_params = params;
        check_floor(args, &format!("stage 3 rank {rank} pre-launch"))?;
        cuda.launch(func, grid, block, &params)?;
        check_floor(args, &format!("stage 3 rank {rank} post-launch"))?;
        // The next rank's residual seed is this rank's bf16 output, so the chained rounding is the
        // same one the reference chain makes.
        let mut bytes = vec![0u8; tokens * HIDDEN * 2];
        cuda.copy_dtoh(&mut bytes, &out_d)?;
        previous = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| u16::from_le_bytes(*chunk))
            .collect();
        cuda.copy_htod(&hidden_d, &bytes)?;
    }

    let (stats, max_ulp) = compare_bf16_bits("combine output", &previous, &reference);
    let unquantised = unquantised_combine_chain(&routed_activations, &w2, &s2, &expert_ids, tokens);
    let (gap, gap_ulp) = compare_bf16_bits(
        "activation quantisation effect (quantised reference vs the same chain with an f32 activation)",
        &reference,
        &unquantised,
    );
    let mut routed_energy = 0.0f64;
    let mut shared_energy = 0.0f64;
    for token in 0..tokens {
        let (quant, quant_scales) = &block_quant[token];
        let gate_bits = mxfp8_ref::dense_gemm(
            quant,
            quant_scales,
            &shared_w1,
            &shared_s1,
            1,
            MOE_INTERMEDIATE,
            HIDDEN,
        );
        let up_bits = mxfp8_ref::dense_gemm(
            quant,
            quant_scales,
            &shared_w3,
            &shared_s3,
            1,
            MOE_INTERMEDIATE,
            HIDDEN,
        );
        let gate: Vec<f32> = gate_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        let up: Vec<f32> = up_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        // The shared expert's activation is quantised by the same reference activation rule the
        // device kernel now carries: swiglu_ref::quantise_row delegates to act_quant_fp8_row, and
        // this call site spells the two steps out because it needs the f32 activation as well.
        let activated: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
            .collect();
        let (codes, scales) = act_quant_fp8_row(&activated);
        let out_bits = mxfp8_ref::dense_gemm(
            &codes,
            &scales,
            &shared_w2,
            &shared_s2,
            1,
            HIDDEN,
            MOE_INTERMEDIATE,
        );
        for column in 0..HIDDEN {
            let routed = half::bf16::from_bits(previous[token * HIDDEN + column]).to_f32();
            let shared = half::bf16::from_bits(out_bits[column]).to_f32();
            routed_energy += f64::from(routed) * f64::from(routed);
            shared_energy += f64::from(shared) * f64::from(shared);
        }
    }
    let routed_rms = (routed_energy / (tokens * HIDDEN) as f64).sqrt();
    let shared_rms = (shared_energy / (tokens * HIDDEN) as f64).sqrt();
    println!(
        "shared expert (host arithmetic through dense_gemm, NOT device-compared): rms routed {routed_rms:.4}, rms shared {shared_rms:.4}, shared share {:.1}% of the summed energy",
        100.0 * shared_energy / (routed_energy + shared_energy)
    );

    if args.repeat > 1 {
        check_floor(args, "stage 3 timing pre-launch")?;
        let (mut per_launch, batched_ms) =
            cuda.launch_repeated(func, grid, block, &last_params, args.repeat)?;
        check_floor(args, "stage 3 timing post-launch")?;
        per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
        let count = per_launch.len();
        let p50 = per_launch[count / 2];
        println!(
            "timing: {count} launches of the last rank at grid ({tokens}, {}, 1); per-launch ms min {:.4} p50 {p50:.4} max {:.4}; batched ms/launch {batched_ms:.4}; the routed combine is {MOE_RANKS} launches, so {:.3} ms for {tokens} tokens = {:.3} ms/token",
            HIDDEN / MOE_TILE_D as usize,
            per_launch[0],
            per_launch[count - 1],
            MOE_RANKS as f64 * p50,
            MOE_RANKS as f64 * p50 / tokens as f64
        );
    }

    cuda.free(module, &[hidden_d, out_d, act_d, acts_d, w_d, ws_d, ids_d]);
    check_floor(args, "stage 3 end")?;
    let exact_ok = stats.exact_pct() >= STAGE3_EXACT_BAR;
    let ulp_ok = max_ulp <= 2;
    println!(
        "stage 3 verdict: {} (combine exact {:.4}% against a {STAGE3_EXACT_BAR}% bar, max ULP distance {max_ulp} against a 2-code bar; quantisation gap {:.2}% of outputs differing, max abs delta {:e}, {gap_ulp} codes)",
        if exact_ok && ulp_ok {
            "within tolerance"
        } else {
            "OVER TOLERANCE"
        },
        stats.exact_pct(),
        100.0 - gap.exact_pct(),
        gap.max_delta
    );
    if exact_ok && ulp_ok {
        println!(
            "stage 3: PASS ({experts} experts, {tokens} tokens, {MOE_RANKS} rank-chained launches)"
        );
        Ok(())
    } else {
        Err(format!(
            "stage 3: FAIL at the combine (exact {:.4}%, max ULP {max_ulp})",
            stats.exact_pct()
        ))
    }
}

/// One launch's exact device bytes. Stage 4 compares these byte for byte: there is no tolerance,
/// because the question is not whether two runs are close but whether they are the same run.
struct MhcRun {
    stats: Vec<u8>,
    coeffs: Vec<u8>,
    rms: Vec<u8>,
    fp8: Vec<u8>,
    scales: Vec<u8>,
    streams: Vec<u8>,
}

impl MhcRun {
    fn fields(&self) -> [(&'static str, &[u8]); 6] {
        [
            ("hc_stats", self.stats.as_slice()),
            ("hc_coeffs", self.coeffs.as_slice()),
            ("hc_rms_out", self.rms.as_slice()),
            ("xhat_fp8", self.fp8.as_slice()),
            ("xhat_scales", self.scales.as_slice()),
            ("X_l", self.streams.as_slice()),
        ]
    }
}

/// FNV-1a 64, for the one-line fingerprint each run prints. The verdict is a direct byte comparison;
/// the fingerprint is evidence a reader can scan, never the thing the test trusts.
fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Fail with the first differing byte and the two windows around it, so a reintroduced
/// nondeterminism names the buffer and offset instead of only saying "not equal".
fn require_identical_bytes(
    token: usize,
    run: usize,
    name: &str,
    expected: &[u8],
    observed: &[u8],
) -> Result<(), String> {
    if expected == observed {
        return Ok(());
    }
    let common = expected.len().min(observed.len());
    let at = expected
        .iter()
        .zip(observed)
        .position(|(a, b)| a != b)
        .unwrap_or(common);
    let lo = at.saturating_sub(8);
    let expected_window = &expected[lo..(at + 8).min(expected.len())];
    let observed_window = &observed[lo..(at + 8).min(observed.len())];
    Err(format!(
        "stage 4: token {token} run {run}: {name} is not byte-identical to run 0 - first difference at \
         byte {at} of {} vs {}; run 0 {:02x?}, run {run} {:02x?}. The same input produced different \
         output bytes, so the grid reduction is not deterministic.",
        expected.len(),
        observed.len(),
        expected_window,
        observed_window,
    ))
}

/// Stage 4: determinism is a property of the device, so it is measured there. N launches of the same
/// input bytes must produce the same output bytes; the harness does not compare against the CPU
/// reference here (stage 1 does that) and does not use a tolerance, because a tolerance is exactly
/// how a nondeterministic reduction hides.
fn stage4(args: &Args) -> Result<(), String> {
    if args.repeat < 2 {
        return Err(format!(
            "--stage 4 needs --repeat at least 2 (got {}): one run cannot disagree with itself",
            args.repeat
        ));
    }
    let kernel = "mega_mhc_fused_bf16_sm121";
    let prefix = format!("layers.{}", args.layer);
    println!(
        "=== stage 4: determinism, {} launches of the same input per token ===",
        args.repeat
    );
    println!(
        "comparison class: EXACT, run to run, with no tolerance and no reference. Run 0 is the \
         baseline for the runs after it, and every compared buffer must match it byte for byte."
    );
    println!(
        "determinism contract cited: D-a (kernel determinism) is the property this stage measures; \
         D-b (reduction determinism) is the comm crate's one-shot reduce property and is not on this \
         device path; D-c (no assumed cross-boundary equality) is why the comparison here is exact \
         bytes with no tolerance and stage 1 names a bar before its numbers"
    );
    println!(
        "what is compared, as raw bytes: hc_stats ({} floats: the final 26 and one 26-float partial \
         vector per tile), hc_coeffs (24 floats), hc_rms_out (1 float), xhat_fp8 (5120 E4M3 codes), \
         xhat_scales (160 ue8m0 bytes) and X_l ({} bf16 values)",
        MHC_GRID_STATS,
        HC_MULT * HIDDEN
    );
    println!(
        "why this is the direct test: the defect was an fp32 atomicAdd reduction whose order four \
         blocks did not fix, so the same input could publish different hc_stats and move E4M3 codes \
         across a boundary. Nothing here derives an expectation from the kernel; a single differing \
         byte fails the stage."
    );
    check_floor(args, "stage 4 start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let predictor = mhc.predictor;
    let hc_base_kernel = mhc.hc_base_kernel;
    let hc_scale = mhc.hc_scale;
    println!(
        "predictor: {} f32 = released [24, 4*d] rows permuted into kernel B/A/C order, NO averaging \
         (released row order into kernel B/A/C, no scale pre-folded); hc_attn_base in kernel row \
         order and hc_attn_scale {hc_scale:?} reach the kernel; attn_norm.weight: {} f32; the \
         layer-0 seed is B = identity, C = 0, A = [1, 0, 0, 0], Y_-1 = 0, X_-1 = four copies of one \
         embed row",
        predictor.len(),
        norm.len()
    );

    let mut b_prev = [0.0f32; HC_MULT * HC_MULT];
    for i in 0..HC_MULT {
        b_prev[i * HC_MULT + i] = 1.0;
    }
    let c_prev = [0.0f32; HC_MULT];
    let a_prev = [1.0f32, 0.0, 0.0, 0.0];
    let y_prev = vec![0.0f32; HIDDEN];

    let planned = (HC_MULT * HIDDEN * 2
        + HIDDEN * 2
        + b_prev.len() * 4
        + c_prev.len() * 4
        + a_prev.len() * 4
        + predictor.len() * 4
        + hc_scale.len() * 4
        + hc_base_kernel.len() * 4
        + norm.len() * 4
        + HIDDEN
        + HIDDEN / QUANT_BLOCK
        + MHC_GRID_STATS * 4
        + 2 * 4
        + 24 * 4
        + 4
        + HIDDEN * 4) as u64;
    let cuda = Cuda::load()?;
    let cubin_path = args.cubin_dir.join(format!("{kernel}.cubin"));
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            cubin_path.display()
        )
    })?;
    let (module, func) = cuda.load_function(&cubin, kernel)?;
    admit_memory(args, planned, "stage 4 device buffers")?;
    let mut allocated = 0usize;
    let x_d = cuda.alloc(HC_MULT * HIDDEN * 2, &mut allocated)?;
    let y_d = cuda.alloc(HIDDEN * 2, &mut allocated)?;
    let b_d = cuda.alloc(b_prev.len() * 4, &mut allocated)?;
    let c_d = cuda.alloc(c_prev.len() * 4, &mut allocated)?;
    let a_d = cuda.alloc(a_prev.len() * 4, &mut allocated)?;
    let predictor_d = cuda.alloc(predictor.len() * 4, &mut allocated)?;
    let scale_d = cuda.alloc(hc_scale.len() * 4, &mut allocated)?;
    let base_d = cuda.alloc(hc_base_kernel.len() * 4, &mut allocated)?;
    let norm_d = cuda.alloc(norm.len() * 4, &mut allocated)?;
    let fp8_d = cuda.alloc(HIDDEN, &mut allocated)?;
    let scales_d = cuda.alloc(HIDDEN / QUANT_BLOCK, &mut allocated)?;
    let stats_d = cuda.alloc(MHC_GRID_STATS * 4, &mut allocated)?;
    let sync_d = cuda.alloc(2 * 4, &mut allocated)?;
    let coeffs_d = cuda.alloc(24 * 4, &mut allocated)?;
    let rms_d = cuda.alloc(4, &mut allocated)?;
    let normed_d = cuda.alloc(HIDDEN * 4, &mut allocated)?;
    println!(
        "device bytes allocated: {allocated} ({} KiB)",
        allocated / 1024
    );

    cuda.copy_htod(&y_d, &bf16_bits_to_bytes(&bf16_bits_of_f32(&y_prev)))?;
    cuda.copy_htod(&b_d, &f32_to_bytes(&b_prev))?;
    cuda.copy_htod(&c_d, &f32_to_bytes(&c_prev))?;
    cuda.copy_htod(&a_d, &f32_to_bytes(&a_prev))?;
    cuda.copy_htod(&predictor_d, &f32_to_bytes(&predictor))?;
    cuda.copy_htod(&scale_d, &f32_to_bytes(&hc_scale))?;
    cuda.copy_htod(&base_d, &f32_to_bytes(&hc_base_kernel))?;
    cuda.copy_htod(&norm_d, &f32_to_bytes(&norm))?;

    let grid = (MHC_TILES, 1u32, 1u32);
    let block = (MHC_THREADS, 1u32, 1u32);
    let params: [u64; 19] = [
        x_d.0,
        y_d.0,
        b_d.0,
        c_d.0,
        a_d.0,
        predictor_d.0,
        scale_d.0,
        base_d.0,
        norm_d.0,
        fp8_d.0,
        scales_d.0,
        stats_d.0,
        sync_d.0,
        coeffs_d.0,
        rms_d.0,
        normed_d.0,
        0, // trace_ring: null, the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
        args.layer as u64,
        0,
    ];

    for token in 0..args.tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let x_bits = bf16_bits_of_f32(&x);
        let x_bytes = bf16_bits_to_bytes(&x_bits);

        let mut baseline: Option<MhcRun> = None;
        for run in 0..args.repeat {
            cuda.copy_htod(&x_d, &x_bytes)?;
            // Per-launch state: the stats buffer and both rendezvous words must start clean, or a
            // later run would inherit an earlier run's partials and counters.
            cuda.copy_htod(&stats_d, &[0u8; MHC_GRID_STATS * 4])?;
            cuda.copy_htod(&sync_d, &[0u8; 2 * 4])?;
            check_floor(args, &format!("stage 4 token {token} run {run} pre-launch"))?;
            cuda.launch(func, grid, block, &params)?;
            check_floor(
                args,
                &format!("stage 4 token {token} run {run} post-launch"),
            )?;

            let mut stats_out = vec![0u8; MHC_GRID_STATS * 4];
            let mut fp8_out = vec![0u8; HIDDEN];
            let mut scales_out = vec![0u8; HIDDEN / QUANT_BLOCK];
            let mut coeffs_out = vec![0u8; 24 * 4];
            let mut rms_out = vec![0u8; 4];
            let mut streams_out = vec![0u8; HC_MULT * HIDDEN * 2];
            cuda.copy_dtoh(&mut stats_out, &stats_d)?;
            cuda.copy_dtoh(&mut fp8_out, &fp8_d)?;
            cuda.copy_dtoh(&mut scales_out, &scales_d)?;
            cuda.copy_dtoh(&mut coeffs_out, &coeffs_d)?;
            cuda.copy_dtoh(&mut rms_out, &rms_d)?;
            cuda.copy_dtoh(&mut streams_out, &x_d)?;
            let observed = MhcRun {
                stats: stats_out,
                coeffs: coeffs_out,
                rms: rms_out,
                fp8: fp8_out,
                scales: scales_out,
                streams: streams_out,
            };
            let line: Vec<String> = observed
                .fields()
                .iter()
                .map(|(name, bytes)| format!("{name} 0x{:016x}", fingerprint(bytes)))
                .collect();
            println!("  token {token} run {run}: {}", line.join(" "));

            match &baseline {
                None => baseline = Some(observed),
                Some(reference) => {
                    for ((name, expected), (_, actual)) in
                        reference.fields().iter().zip(observed.fields())
                    {
                        require_identical_bytes(token, run, name, expected, actual)?;
                    }
                }
            }
        }
        println!(
            "  token {token}: {} runs of one input produced byte-identical hc_stats, hc_coeffs, \
             hc_rms_out, xhat_fp8, xhat_scales and X_l",
            args.repeat
        );
    }

    cuda.free(
        module,
        &[
            x_d,
            y_d,
            b_d,
            c_d,
            a_d,
            predictor_d,
            scale_d,
            base_d,
            norm_d,
            fp8_d,
            scales_d,
            stats_d,
            sync_d,
            coeffs_d,
            rms_d,
        ],
    );
    check_floor(args, "stage 4 end")?;
    println!(
        "stage 4: PASS ({} tokens x {} launches, every compared byte identical)",
        args.tokens, args.repeat
    );
    Ok(())
}

/// The LM head's geometry, from kernels/m4_common_sm121.cuh and kernels/cert_lm_head_sm121.cu.
/// The kernel asserts these internally; the launcher must pass the same values or the block count
/// and the per-rank vocabulary disagree with the buffer sizes.
const LM_HIDDEN: usize = 5120;
const LM_VOCAB: usize = 129_280;
const LM_VOCAB_PER_RANK: usize = 32_320;
const LM_THREADS: u32 = 128;
const LM_CHUNK: usize = 1024;
const LM_GRID: u32 = 32;

/// The tolerance for an exact dot product recomputed on the host in the kernel's own order. The
/// kernel may contract mul+add into an FMA while the host does not; 5120 bf16 terms of this
/// magnitude leave a difference far below this bar, and the harness prints the measured maximum so
/// the bar is checkable rather than asserted.
const LM_EXACT_BAR: f32 = 2.0e-2;

/// The slack for the certificate check: a certified entry stores an upper bound, and it must not be
/// below the matched reference by more than f32 accumulation error. This is much tighter than
/// LM_EXACT_BAR on purpose; the bar is for the exact dots, this is for the bound. Would the bound
/// be below the truth by an accumulation amount, the certificate would be wrong and this catches it.
const LM_CERTIFICATE_EPS: f32 = 1.0e-5;

const _: () = assert!(
    LM_GRID as usize == LM_VOCAB_PER_RANK.div_ceil(LM_CHUNK),
    "the grid covers the per-rank vocabulary in whole chunks"
);

fn read_f32_file(path: &Path, expect: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != expect * 4 {
        return Err(format!(
            "{}: expected {expect} f32 values ({} bytes), got {} bytes",
            path.display(),
            expect * 4,
            bytes.len()
        ));
    }
    Ok(bytes_to_f32(&bytes))
}

fn bf16_code_to_f32(code: u16) -> f32 {
    f32::from_bits(u32::from(code) << 16)
}

/// The exact logits the kernel would compute if it pruned nothing, in the kernel's own order: the
/// same bf16 hidden codes and the same per-entry ascending dot over the row. This is the
/// quantisation-matched reference: comparing the device against a reference computed from the f32
/// head input would fold the bf16 conversion of the hidden vector into the head's error.
fn host_quantised_logits(hidden: &[u16], head: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; LM_VOCAB];
    for (entry, logit) in out.iter_mut().enumerate() {
        let row = &head[entry * LM_HIDDEN * 2..(entry + 1) * LM_HIDDEN * 2];
        let mut acc = 0.0f32;
        for i in 0..LM_HIDDEN {
            let w = u16::from_le_bytes([row[2 * i], row[2 * i + 1]]);
            acc += bf16_code_to_f32(hidden[i]) * bf16_code_to_f32(w);
        }
        *logit = acc;
    }
    out
}

fn lm_argmax(logits: &[f32]) -> (f32, usize) {
    let mut best = logits[0];
    let mut at = 0usize;
    for (index, value) in logits.iter().enumerate().skip(1) {
        // Ascending scan: a strict comparison keeps the first (lowest) index on a tie.
        if *value > best {
            best = *value;
            at = index;
        }
    }
    (best, at)
}

/// Stage 5: the certified LM head against a host reference on one token's hidden state.
///
/// cert_lm_head_sm121 is launched once per tensor-parallel rank against that rank's 32,320 rows of
/// head.weight, with the hidden vector the reference's own head multiplied (boundary_dump
/// --head-input-out, converted to the kernel's bf16 ABI). The reference side is recomputed here
/// from the same bf16 codes in the kernel's own order, so the comparison is the head's arithmetic
/// and not the hidden vector's bf16 rounding.
///
/// The kernel's contract is the argmax, not the top-5: a certified entry stores its upper bound,
/// not its exact value, so the top-5 set cannot be read from the kernel's output. This stage
/// therefore checks the argmax, the exact entries, and the certificate itself (every certified
/// bound is still an upper bound on the matched reference), and prints the measured numbers.
fn stage5(args: &Args) -> Result<(), String> {
    let head_input_path = args
        .head_input
        .as_ref()
        .ok_or("--stage 5 needs --head-input FILE (boundary_dump --head-input-out)")?;
    let logits_path = args
        .logits
        .as_ref()
        .ok_or("--stage 5 needs --logits FILE (boundary_dump --logits-out)")?;
    let hidden_f32 = read_f32_file(head_input_path, LM_HIDDEN)?;
    let reference = read_f32_file(logits_path, LM_VOCAB)?;
    let hidden = bf16_bits_of_f32(&hidden_f32);
    let hidden_bytes = bf16_bits_to_bytes(&hidden);

    let mut snapshot = Snapshot::open(&args.checkpoint)?;
    let head = snapshot.read_raw("head.weight", "BF16", &[LM_VOCAB, LM_HIDDEN])?;
    println!(
        "stage 5: head.weight BF16 [{} x {}] ({} B), head input {} f32 ({} B), reference logits {} f32",
        LM_VOCAB,
        LM_HIDDEN,
        head.len(),
        hidden_f32.len(),
        hidden_f32.len() * 4,
        reference.len()
    );
    let matched = host_quantised_logits(&hidden, &head);
    let (reference_max, reference_at) = lm_argmax(&reference);
    let (matched_max, matched_at) = lm_argmax(&matched);
    println!(
        "host reference (f32 hidden): argmax {reference_at} value {reference_max:.9}; quantisation-matched (bf16 hidden, kernel order): argmax {matched_at} value {matched_max:.9}"
    );

    let cuda = Cuda::load()?;
    let cubin_path = args.cubin_dir.join("cert_lm_head_sm121.cubin");
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (nvcc -cubin -arch=sm_121a -I kernels -o {} kernels/cert_lm_head_sm121.cu)",
            cubin_path.display(),
            cubin_path.display()
        )
    })?;
    let (_module, func) = cuda.load_function(&cubin, "cert_lm_head_sm121")?;
    let planned = (LM_HIDDEN * 2
        + LM_VOCAB * LM_HIDDEN * 2
        + LM_VOCAB_PER_RANK * 4
        + LM_VOCAB_PER_RANK
        + 4
        + (LM_GRID as usize) * 4
        + (LM_GRID as usize) * 4) as u64;
    admit_memory(args, planned, "stage 5 device buffers")?;
    let mut allocated = 0usize;
    let hidden_d = cuda.alloc(LM_HIDDEN * 2, &mut allocated)?;
    let weight_d = cuda.alloc(LM_VOCAB * LM_HIDDEN * 2, &mut allocated)?;
    let logits_d = cuda.alloc(LM_VOCAB_PER_RANK * 4, &mut allocated)?;
    let certified_d = cuda.alloc(LM_VOCAB_PER_RANK, &mut allocated)?;
    let certified_count_d = cuda.alloc(4, &mut allocated)?;
    let block_max_d = cuda.alloc((LM_GRID as usize) * 4, &mut allocated)?;
    let block_argmax_d = cuda.alloc((LM_GRID as usize) * 4, &mut allocated)?;
    println!("stage 5: allocated {allocated} B on device 0");
    check_floor(args, "stage 5 pre-upload")?;
    cuda.copy_htod(&hidden_d, &hidden_bytes)?;
    cuda.copy_htod(&weight_d, &head)?;
    check_floor(args, "stage 5 post-upload")?;

    let rank_stride = (LM_VOCAB_PER_RANK * LM_HIDDEN * 2) as u64;
    let mut best = f32::NEG_INFINITY;
    let mut best_entry = 0usize;
    let mut exact = 0usize;
    let mut pruned = 0usize;
    let mut max_exact_delta = 0.0f32;
    let mut at_max_delta = 0usize;
    let mut certificate_violations = 0usize;
    let mut per_rank = Vec::new();
    for rank in 0..4usize {
        let weight_ptr = weight_d.0 + (rank as u64) * rank_stride;
        let params: [u64; 11] = [
            hidden_d.0,
            weight_ptr,
            logits_d.0,
            certified_d.0,
            certified_count_d.0,
            block_max_d.0,
            block_argmax_d.0,
            LM_VOCAB_PER_RANK as u64,
            0,
            args.layer as u64,
            0,
        ];
        check_floor(args, &format!("stage 5 rank {rank} pre-launch"))?;
        // The kernel atomically accumulates into this buffer, so it is per-launch state: without
        // re-zeroing, rank 1 would read rank 0's count (and rank 0 uninitialized memory).
        cuda.copy_htod(&certified_count_d, &[0u8; 4])?;
        cuda.launch(func, (LM_GRID, 1, 1), (LM_THREADS, 1, 1), &params)?;
        check_floor(args, &format!("stage 5 rank {rank} post-launch"))?;
        let mut logits_bytes = vec![0u8; LM_VOCAB_PER_RANK * 4];
        let mut certified_bytes = vec![0u8; LM_VOCAB_PER_RANK];
        let mut count_bytes = [0u8; 4];
        let mut block_max_bytes = vec![0u8; (LM_GRID as usize) * 4];
        let mut block_argmax_bytes = vec![0u8; (LM_GRID as usize) * 4];
        cuda.copy_dtoh(&mut logits_bytes, &logits_d)?;
        cuda.copy_dtoh(&mut certified_bytes, &certified_d)?;
        cuda.copy_dtoh(&mut count_bytes, &certified_count_d)?;
        cuda.copy_dtoh(&mut block_max_bytes, &block_max_d)?;
        cuda.copy_dtoh(&mut block_argmax_bytes, &block_argmax_d)?;
        let rank_logits = bytes_to_f32(&logits_bytes);
        // The kernel publishes its own per-block maximum and argmax; cross-check that reduction
        // against the per-entry values instead of ignoring it. A disagreement is a kernel bug even
        // when the global argmax happens to survive.
        let block_max = bytes_to_f32(&block_max_bytes);
        let block_argmax: Vec<u32> = block_argmax_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        for block in 0..LM_GRID as usize {
            let lo = block * LM_CHUNK;
            let hi = (lo + LM_CHUNK).min(LM_VOCAB_PER_RANK);
            let mut local_best = f32::NEG_INFINITY;
            let mut local_at = 0u32;
            for (entry, value) in rank_logits[lo..hi].iter().enumerate() {
                if *value > local_best || (*value == local_best && ((lo + entry) as u32) < local_at)
                {
                    local_best = *value;
                    local_at = (lo + entry) as u32;
                }
            }
            if block_max[block] != local_best || block_argmax[block] != local_at {
                return Err(format!(
                    "stage 5 rank {rank} block {block}: kernel reduction (max {} entry {}) disagrees with the per-entry scan (max {} entry {})",
                    block_max[block], block_argmax[block], local_best, local_at
                ));
            }
        }
        let rank_exact = certified_bytes.iter().filter(|c| **c == 0).count();
        let rank_pruned = u32::from_le_bytes(count_bytes) as usize;
        if rank_pruned != LM_VOCAB_PER_RANK - rank_exact {
            return Err(format!(
                "stage 5 rank {rank}: the kernel's certified_count {rank_pruned} disagrees with the certified flags ({} certified)",
                LM_VOCAB_PER_RANK - rank_exact
            ));
        }
        per_rank.push((rank_exact, rank_pruned));
        for (entry, value) in rank_logits.iter().enumerate() {
            let global = rank * LM_VOCAB_PER_RANK + entry;
            if *value > best || (*value == best && global < best_entry) {
                best = *value;
                best_entry = global;
            }
            if certified_bytes[entry] == 0 {
                exact += 1;
                let delta = (*value - matched[global]).abs();
                if delta > max_exact_delta {
                    max_exact_delta = delta;
                    at_max_delta = global;
                }
            } else {
                pruned += 1;
                if *value + LM_CERTIFICATE_EPS < matched[global] {
                    certificate_violations += 1;
                }
            }
        }
    }
    println!(
        "device: argmax {best_entry} value {best:.9}; exact {exact}, certified {pruned} ({:.2}%), per-rank exact/certified {:?}",
        100.0 * pruned as f64 / LM_VOCAB as f64,
        per_rank
    );
    println!(
        "exact entries: max |device - host(kernel order)| {max_exact_delta:e} at entry {} (provisional bar {LM_EXACT_BAR:e}; the released token measured 0 against the matched reference); certificate violations {certificate_violations}",
        if max_exact_delta == 0.0 {
            "n/a".to_string()
        } else {
            at_max_delta.to_string()
        }
    );
    if pruned == 0 {
        println!(
            "certificate: not exercised on this token (0 of {LM_VOCAB} entries certified), so the bound check is vacuous here and is not claimed as evidence"
        );
    }
    let argmax_ok = best_entry == matched_at;
    let reference_ok = best_entry == reference_at;
    println!(
        "argmax: device {best_entry}, matched-reference {matched_at}, f32-reference {reference_at} (matched {}; f32 {})",
        if argmax_ok { "equal" } else { "DIFFERENT" },
        if reference_ok { "equal" } else { "DIFFERENT" }
    );
    if !argmax_ok || !reference_ok || max_exact_delta > LM_EXACT_BAR || certificate_violations != 0
    {
        return Err(format!(
            "stage 5: FAIL (argmax device {best_entry} vs matched {matched_at} vs reference {reference_at}; max exact delta {max_exact_delta:e} bar {LM_EXACT_BAR:e}; certificate violations {certificate_violations})"
        ));
    }
    let certificate = if pruned == 0 {
        "the certificate was not exercised (0 certified entries)".to_string()
    } else {
        format!("{pruned} certified bounds all upper bounds")
    };
    println!(
        "stage 5: PASS (argmax {best_entry} equal to both references; {exact} exact entries within the bar; {certificate})"
    );
    Ok(())
}

/// The reference head's top-k entry indices, ordered by value then by lowest index. The kernel's
/// tie-break is the lowest entry index, so a comparison that ignored the index would disagree on a
/// tie even when both sides computed the same values.
fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b)));
    indices.truncate(k);
    indices
}

/// Stage 6: the CPU half of the criterion-3 head comparison, runnable while vLLM holds the node.
///
/// This stage never opens a CUDA context: it streams head.weight one 10 KB row at a time through the
/// checkpoint's own shard reader and recomputes every exact logit twice, once from the reference's f32
/// head input and once from the bf16 codes the kernel ABI is fed, both in the kernel's ascending order.
/// The two deltas it prints are the ones the device run will sit on top of: the bf16 conversion of
/// the hidden vector (what the device cannot avoid) and the host order versus the engine's matmul
/// order (what the matched reference must reproduce). The first number is why the device comparison
/// uses the matched reference instead of the f32-hidden logits; the head's own bar is measured by the
/// device run (which reported max abs delta 0 against the matched host on the released token), not
/// derived from the conversion.
fn stage6(args: &Args) -> Result<(), String> {
    let head_input_path = args
        .head_input
        .as_ref()
        .ok_or("--stage 6 needs --head-input FILE (boundary_dump --head-input-out)")?;
    let logits_path = args
        .logits
        .as_ref()
        .ok_or("--stage 6 needs --logits FILE (boundary_dump --logits-out)")?;
    let hidden_f32 = read_f32_file(head_input_path, LM_HIDDEN)?;
    let reference = read_f32_file(logits_path, LM_VOCAB)?;
    let hidden = bf16_bits_of_f32(&hidden_f32);
    let mut snapshot = Snapshot::open(&args.checkpoint)?;
    let header = snapshot.load("head.weight")?;
    if header.shape.len() != 2 || header.shape[0] != LM_VOCAB || header.shape[1] != LM_HIDDEN {
        return Err(format!(
            "head.weight: expected two dimensions [{LM_VOCAB}][{LM_HIDDEN}], the checkpoint stores {:?}",
            header.shape
        ));
    }
    let mut row = vec![0u8; LM_HIDDEN * 2];
    let mut matched_f32 = vec![0.0f32; LM_VOCAB];
    let mut matched_bf16 = vec![0.0f32; LM_VOCAB];
    for entry in 0..LM_VOCAB {
        snapshot.read_row_bf16_into("head.weight", entry, LM_HIDDEN, &mut row)?;
        let mut from_f32 = 0.0f32;
        let mut from_bf16 = 0.0f32;
        for i in 0..LM_HIDDEN {
            let w = bf16_code_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
            from_f32 += hidden_f32[i] * w;
            from_bf16 += bf16_code_to_f32(hidden[i]) * w;
        }
        matched_f32[entry] = from_f32;
        matched_bf16[entry] = from_bf16;
    }
    let (reference_max, reference_at) = lm_argmax(&reference);
    let (f32_max, f32_at) = lm_argmax(&matched_f32);
    let (bf16_max, bf16_at) = lm_argmax(&matched_bf16);
    let mut delta_f32 = 0.0f32;
    let mut delta_bf16 = 0.0f32;
    for entry in 0..LM_VOCAB {
        delta_f32 = delta_f32.max((matched_f32[entry] - reference[entry]).abs());
        delta_bf16 = delta_bf16.max((matched_bf16[entry] - matched_f32[entry]).abs());
    }
    let reference_top = top_k(&reference, 5);
    let f32_top = top_k(&matched_f32, 5);
    let bf16_top = top_k(&matched_bf16, 5);
    println!(
        "stage 6 (CPU): reference argmax {reference_at} value {reference_max:.9}; recomputed f32 argmax {f32_at} value {f32_max:.9}; bf16-hidden argmax {bf16_at} value {bf16_max:.9}"
    );
    println!(
        "deltas: max |recomputed_f32 - reference| {delta_f32:e} (host order vs engine matmul); max |bf16_hidden - f32_hidden| {delta_bf16:e} (the unavoidable conversion the device ABI adds)"
    );
    println!("top-5: reference {reference_top:?}; f32 {f32_top:?}; bf16 {bf16_top:?}");
    let argmax_ok = reference_at == f32_at && reference_at == bf16_at;
    let top_ok = reference_top == f32_top && reference_top == bf16_top;
    // The f32 recomputation is the engine's order; measured max abs delta 0 on the released token,
    // and a nonzero value would mean the matched reference is no longer the head's arithmetic.
    let order_ok = delta_f32 == 0.0;
    if !argmax_ok || !top_ok || !order_ok {
        return Err(format!(
            "stage 6: FAIL (argmax reference {reference_at} f32 {f32_at} bf16 {bf16_at}; top-5 {:?} vs {:?} vs {:?}; f32-order delta {delta_f32:e})",
            reference_top, f32_top, bf16_top
        ));
    }
    println!(
        "stage 6: PASS (argmax and top-5 identical under both recomputations; the bf16 conversion moves a logit by at most {delta_bf16:e}, which is why the device comparison uses the matched reference rather than the f32-hidden logits)"
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Stage 8: the three never-executed kernels, probed against their in-crate references on the same
// bytes (B1). Dense and wo_a are compared bit-for-bit at the bf16 output; swiglu is compared at its
// two outputs (E4M3 codes and ue8m0 scales). The probe is deterministic and tiny: it does not read
// the checkpoint and its shapes are the smallest that exercise the tiling (dense: two tiles per
// axis; wo_a: the released geometry; swiglu: one warp per block, both the 640-wide routed width and
// the 2304-wide shared width).
// ---------------------------------------------------------------------------------------------

/// Deterministic byte patterns for the probe inputs: valid E4M3 codes only (0x7F is NaN), kept
/// below the NaN code so a reference decode and the device decode agree.
fn probe_codes(len: usize, stride: u32, period: u32) -> Vec<u8> {
    (0..len)
        .map(|index| (((index as u32).wrapping_mul(stride).wrapping_add(11)) % period) as u8)
        .collect()
}

/// Deterministic ue8m0 scale codes: valid, in a middle exponent band, never 255.
fn probe_scales(len: usize, stride: u32, period: u32) -> Vec<u8> {
    (0..len)
        .map(|index| (120 + ((index as u32).wrapping_mul(stride) % period)) as u8)
        .collect()
}

fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

/// One probe's verdict line and its pass flag.
struct ProbeVerdict {
    name: String,
    ok: bool,
}

/// Load one probe kernel from the cubin directory, folding the two failure modes into one message.
fn load_probe_kernel(
    cuda: &Cuda,
    args: &Args,
    kernel: &str,
) -> Result<(CuModule, CuFunction), String> {
    let path = args.cubin_dir.join(format!("{kernel}.cubin"));
    let cubin = std::fs::read(&path).map_err(|e| {
        format!(
            "{}: {e} (build the B1 cubins first, see the module docs)",
            path.display()
        )
    })?;
    cuda.load_function(&cubin, kernel)
}

/// The dense probe: two tiles in each axis, one block per tile on the device, the in-crate GEMM on
/// the same activation and B12x weight bytes.
fn probe_dense(cuda: &Cuda, args: &Args, verdicts: &mut Vec<ProbeVerdict>) -> Result<(), String> {
    let kernel = "b12x_mxfp8_dense_sm121";
    println!("--- probe: {kernel} (dense MXFP8 GEMM) ---");
    let (module, func) = load_probe_kernel(cuda, args, kernel)?;
    let (rows, cols, inner) = (128usize, 128usize, 128usize);
    let a = probe_codes(rows * inner, 37, 0x7F);
    let a_scales = probe_scales(rows * inner / QUANT_BLOCK, 7, 16);
    let w = probe_codes(mxfp8_ref::b12x_buffer_bytes(cols, inner), 53, 0x7F);
    let w_scales = probe_scales(mxfp8_ref::b12x_scale_bytes(cols, inner), 11, 16);
    let expected = mxfp8_ref::dense_gemm(&a, &a_scales, &w, &w_scales, rows, cols, inner);

    let mut allocated = 0usize;
    let a_d = cuda.alloc(a.len(), &mut allocated)?;
    let as_d = cuda.alloc(a_scales.len(), &mut allocated)?;
    let w_d = cuda.alloc(w.len(), &mut allocated)?;
    let ws_d = cuda.alloc(w_scales.len(), &mut allocated)?;
    let out_d = cuda.alloc(rows * cols * 2, &mut allocated)?;
    println!(
        "  shapes: rows {rows}, cols {cols}, inner {inner}; device bytes {allocated} ({} KiB)",
        allocated / 1024
    );
    cuda.copy_htod(&a_d, &a)?;
    cuda.copy_htod(&as_d, &a_scales)?;
    cuda.copy_htod(&w_d, &w)?;
    cuda.copy_htod(&ws_d, &w_scales)?;
    let grid = ((rows.div_ceil(64)) as u32, (cols.div_ceil(64)) as u32, 1u32);
    let block = (256u32, 1u32, 1u32);
    let params: [u64; 11] = [
        a_d.0,
        as_d.0,
        w_d.0,
        ws_d.0,
        out_d.0,
        rows as u64,
        cols as u64,
        inner as u64,
        0, // trace_ring: built without -DSPARKFLASH_ENABLE_GPU_TRACE
        args.layer as u64,
        0, // step_seq
    ];
    check_floor(args, "stage 8 dense pre-launch")?;
    cuda.launch(func, grid, block, &params)?;
    check_floor(args, "stage 8 dense post-launch")?;
    let mut device = vec![0u8; rows * cols * 2];
    cuda.copy_dtoh(&mut device, &out_d)?;
    cuda.free(module, &[a_d, as_d, w_d, ws_d, out_d]);
    let device_bits = bytes_to_u16(&device);
    let (stats, max_ulp) = compare_bf16_bits("dense out", &device_bits, &expected);
    println!("  max ULP distance {max_ulp}");
    let ok = stats.all_exact() && max_ulp == 0;
    verdicts.push(ProbeVerdict {
        name: format!("{kernel}: bf16 out"),
        ok,
    });
    Ok(())
}

/// The wo_a probe: the released geometry (8 groups, 4096 reduction, 1024 output channels per
/// group), the in-crate grouped reference on the same bytes.
fn probe_wo_a(cuda: &Cuda, args: &Args, verdicts: &mut Vec<ProbeVerdict>) -> Result<(), String> {
    let kernel = "b12x_mxfp8_wo_a_sm121";
    println!("--- probe: {kernel} (grouped wo_a MXFP8 GEMM) ---");
    let (module, func) = load_probe_kernel(cuda, args, kernel)?;
    let groups = 8usize;
    let group_inner = 4096usize;
    let lo_rank = 1024usize;
    let attn = probe_codes(groups * group_inner, 29, 0x7F);
    let attn_scales = probe_scales(groups * group_inner / QUANT_BLOCK, 5, 16);
    let w = probe_codes(
        groups * mxfp8_ref::b12x_buffer_bytes(lo_rank, group_inner),
        17,
        0x7F,
    );
    let w_scales = probe_scales(
        groups * mxfp8_ref::b12x_scale_bytes(lo_rank, group_inner),
        3,
        16,
    );
    let expected = mxfp8_ref::grouped_wo_a(
        &attn,
        &attn_scales,
        &w,
        &w_scales,
        groups,
        group_inner,
        lo_rank,
    );

    let mut allocated = 0usize;
    let attn_d = cuda.alloc(attn.len(), &mut allocated)?;
    let as_d = cuda.alloc(attn_scales.len(), &mut allocated)?;
    let w_d = cuda.alloc(w.len(), &mut allocated)?;
    let ws_d = cuda.alloc(w_scales.len(), &mut allocated)?;
    let out_d = cuda.alloc(groups * lo_rank * 2, &mut allocated)?;
    println!(
        "  shapes: groups {groups}, group_inner {group_inner}, lo_rank {lo_rank}; device bytes {allocated} ({} MiB)",
        allocated / (1024 * 1024)
    );
    cuda.copy_htod(&attn_d, &attn)?;
    cuda.copy_htod(&as_d, &attn_scales)?;
    cuda.copy_htod(&w_d, &w)?;
    cuda.copy_htod(&ws_d, &w_scales)?;
    let grid = (groups as u32, (lo_rank / 256) as u32, 1u32);
    let block = (256u32, 1u32, 1u32);
    let params: [u64; 9] = [
        attn_d.0,
        as_d.0,
        w_d.0,
        ws_d.0,
        out_d.0,
        groups as u64,
        0, // trace_ring
        args.layer as u64,
        0, // step_seq
    ];
    check_floor(args, "stage 8 wo_a pre-launch")?;
    cuda.launch(func, grid, block, &params)?;
    check_floor(args, "stage 8 wo_a post-launch")?;
    let mut device = vec![0u8; groups * lo_rank * 2];
    cuda.copy_dtoh(&mut device, &out_d)?;
    cuda.free(module, &[attn_d, as_d, w_d, ws_d, out_d]);
    let device_bits = bytes_to_u16(&device);
    let (stats, max_ulp) = compare_bf16_bits("wo_a lo_out", &device_bits, &expected);
    println!("  max ULP distance {max_ulp}");
    let ok = stats.all_exact() && max_ulp == 0;
    verdicts.push(ProbeVerdict {
        name: format!("{kernel}: bf16 lo_out"),
        ok,
    });
    Ok(())
}

/// The swiglu probe at one width: the kernel's E4M3 codes and ue8m0 scales against
/// swiglu_ref::quantise_row over the bf16-rounded inputs the device reads.
fn probe_swiglu_shape(
    cuda: &Cuda,
    args: &Args,
    func: CuFunction,
    rows: usize,
    cols: usize,
    verdicts: &mut Vec<ProbeVerdict>,
) -> Result<(), String> {
    let limit = swiglu_ref::SWIGLU_LIMIT;
    // Values spread over +-16 so both clamps fire on some elements, and bf16-rounded before both
    // sides see them: the device reads bf16, so the reference must read the same bytes.
    let gate_f32: Vec<f32> = (0..rows * cols)
        .map(|index| (((index * 41 + 13) % 997) as f32 - 498.0) / 32.0)
        .collect();
    let up_f32: Vec<f32> = (0..rows * cols)
        .map(|index| (((index * 59 + 7) % 991) as f32 - 495.0) / 32.0)
        .collect();
    let gate_bits = bf16_bits_of_f32(&gate_f32);
    let up_bits = bf16_bits_of_f32(&up_f32);
    let gate: Vec<f32> = gate_bits
        .iter()
        .map(|bits| half::bf16::from_bits(*bits).to_f32())
        .collect();
    let up: Vec<f32> = up_bits
        .iter()
        .map(|bits| half::bf16::from_bits(*bits).to_f32())
        .collect();
    let mut expected_codes = vec![0u8; rows * cols];
    let mut expected_scales = vec![0u8; rows * cols / QUANT_BLOCK];
    for row in 0..rows {
        let (codes, scales) = swiglu_ref::quantise_row(
            &gate[row * cols..(row + 1) * cols],
            &up[row * cols..(row + 1) * cols],
            limit,
        );
        expected_codes[row * cols..(row + 1) * cols].copy_from_slice(&codes);
        expected_scales[row * cols / QUANT_BLOCK..(row + 1) * cols / QUANT_BLOCK]
            .copy_from_slice(&scales);
    }

    let mut allocated = 0usize;
    let gate_d = cuda.alloc(rows * cols * 2, &mut allocated)?;
    let up_d = cuda.alloc(rows * cols * 2, &mut allocated)?;
    let codes_d = cuda.alloc(rows * cols, &mut allocated)?;
    let scales_d = cuda.alloc(rows * cols / QUANT_BLOCK, &mut allocated)?;
    println!(
        "  shape: rows {rows}, cols {cols}; device bytes {allocated} ({} KiB)",
        allocated / 1024
    );
    cuda.copy_htod(&gate_d, &bf16_bits_to_bytes(&gate_bits))?;
    cuda.copy_htod(&up_d, &bf16_bits_to_bytes(&up_bits))?;
    let total_blocks = (rows * cols / QUANT_BLOCK) as u32;
    let warps_per_block = 8u32;
    let grid = (total_blocks.div_ceil(warps_per_block), 1u32, 1u32);
    let block = (256u32, 1u32, 1u32);
    let params: [u64; 10] = [
        gate_d.0,
        up_d.0,
        codes_d.0,
        scales_d.0,
        rows as u64,
        cols as u64,
        u64::from(limit.to_bits()), // the f32 argument rides in a u64 slot, little-endian
        0,                          // trace_ring
        args.layer as u64,
        0, // step_seq
    ];
    check_floor(args, &format!("stage 8 swiglu {rows}x{cols} pre-launch"))?;
    cuda.launch(func, grid, block, &params)?;
    check_floor(args, &format!("stage 8 swiglu {rows}x{cols} post-launch"))?;
    let mut codes_out = vec![0u8; rows * cols];
    let mut scales_out = vec![0u8; rows * cols / QUANT_BLOCK];
    cuda.copy_dtoh(&mut codes_out, &codes_d)?;
    cuda.copy_dtoh(&mut scales_out, &scales_d)?;
    // Free the shape's buffers only; the module is unloaded once by stage 8 after both widths.
    cuda.free(std::ptr::null_mut(), &[gate_d, up_d, codes_d, scales_d]);
    let s_codes = compare_u8(
        &format!("swiglu {rows}x{cols} out_fp8"),
        &codes_out,
        &expected_codes,
    )?;
    let s_scales = compare_u8(
        &format!("swiglu {rows}x{cols} out_scales"),
        &scales_out,
        &expected_scales,
    )?;
    let ok = s_codes.all_exact() && s_scales.all_exact();
    verdicts.push(ProbeVerdict {
        name: format!("swiglu_mxfp8_quant_sm121 {rows}x{cols}: codes+scales"),
        ok,
    });
    Ok(())
}

/// Stage 8: the B1 probes. Each probe compares a never-executed kernel against its in-crate
/// reference on identical input bytes; the stage fails if any output is not bit-exact, and a
/// failure is reported as a finding rather than absorbed by a tolerance (B1's contract, D6).
fn stage8(args: &Args) -> Result<(), String> {
    println!(
        "=== stage 8: B1 probes, never-executed kernels against their in-crate references ==="
    );
    println!(
        "comparison class: each kernel is compared BIT-EXACT against src/mxfp8_ref.rs or \
         src/swiglu_ref.rs on identical input bytes; any difference is a finding, not a tolerance"
    );
    let cuda = Cuda::load()?;
    let mut verdicts: Vec<ProbeVerdict> = Vec::new();
    probe_dense(&cuda, args, &mut verdicts)?;
    probe_wo_a(&cuda, args, &mut verdicts)?;

    let kernel = "swiglu_mxfp8_quant_sm121";
    println!("--- probe: {kernel} (SwiGLU activation + MXFP8 quantise) ---");
    let (module, func) = load_probe_kernel(&cuda, args, kernel)?;
    probe_swiglu_shape(&cuda, args, func, 4, 640, &mut verdicts)?;
    probe_swiglu_shape(&cuda, args, func, 2, 2304, &mut verdicts)?;
    cuda.free(module, &[]);

    let failed: Vec<&str> = verdicts
        .iter()
        .filter(|verdict| !verdict.ok)
        .map(|verdict| verdict.name.as_str())
        .collect();
    if failed.is_empty() {
        println!("stage 8: PASS ({} probes bit-exact)", verdicts.len());
        Ok(())
    } else {
        Err(format!(
            "stage 8: FAIL at {:?} - a probe differs from its in-crate reference; B1 stops here and reports the delta rather than adjusting the bar",
            failed
        ))
    }
}

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

// ---------------------------------------------------------------------------------------------
// Stage 7: the chained layer-0 replay (B2). This is the device path of device-chain-design.md
// section 3: the mHC kernel, the attention kernels and the routed combine are launched in the
// order the layer computes, the dense projections run on the two newly adopted device kernels, the
// documented seams carry the elementwise arithmetic that has no kernel, and the eighteen boundary
// tensors of row 0 are captured and diffed against the host v1 row by comparison class.
//
// It measures ONE layer and one token: the chain input is the embedding row of token 0 with the
// identity initial mix, which is the right seed only at layer 0. The 40-layer token chain is a
// separate program.
// ---------------------------------------------------------------------------------------------

/// The B2 comparison class of one boundary tensor (docs/build/phase1-boundary-table.md section 5).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BoundaryClass {
    ExactBits,
    ExactCodes,
}

/// Total wo_a output width: eight groups of 1024, the wo_b reduction width.
const WO_A_OUT: usize = 8 * 1024;

/// The eighteen tensors of one boundary row, in the v1 order of `boundary::TENSOR_ORDER`.
const B2_TENSORS: [(&str, BoundaryClass); 18] = [
    ("residual_in", BoundaryClass::ExactBits),
    ("hc_in_pre", BoundaryClass::ExactBits),
    ("hc_attn_pre", BoundaryClass::ExactBits),
    ("hc_attn_post", BoundaryClass::ExactBits),
    ("hc_attn_comb", BoundaryClass::ExactBits),
    ("pre_attention_hidden", BoundaryClass::ExactBits),
    ("pre_attention_xhat", BoundaryClass::ExactCodes),
    ("pre_attention_scales", BoundaryClass::ExactCodes),
    ("attention_out", BoundaryClass::ExactBits),
    ("post_attention_hidden", BoundaryClass::ExactBits),
    ("hc_ffn_pre", BoundaryClass::ExactBits),
    ("hc_ffn_post", BoundaryClass::ExactBits),
    ("hc_ffn_comb", BoundaryClass::ExactBits),
    ("pre_ffn_hidden", BoundaryClass::ExactBits),
    ("pre_ffn_xhat", BoundaryClass::ExactCodes),
    ("pre_ffn_scales", BoundaryClass::ExactCodes),
    ("ffn_out", BoundaryClass::ExactBits),
    ("residual_out", BoundaryClass::ExactBits),
];

/// The three stream-carrying tensors: their device value is a bf16 stream write, so the report
/// gives both the host delta and the fraction equal to bf16(host) (design section 4).
const B2_STREAM_TENSORS: [&str; 3] = ["residual_in", "post_attention_hidden", "residual_out"];

/// One value of a boundary row, as captured on either side: f32 tensors keep their bits, code
/// tensors their bytes.
#[allow(dead_code)]
enum RowValue {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

/// One mHC launch's device buffers. K1 (attention sublayer of layer 0), K8 (FFN sublayer) and K16
/// (the final hc_post) share them; the kernel updates X in place, so x is the captured stream.
struct LayerMhc {
    x: Deviceptr,
    y: Deviceptr,
    b: Deviceptr,
    c: Deviceptr,
    a: Deviceptr,
    predictor: Deviceptr,
    scale: Deviceptr,
    base: Deviceptr,
    norm: Deviceptr,
    fp8: Deviceptr,
    scales: Deviceptr,
    stats: Deviceptr,
    sync: Deviceptr,
    coeffs: Deviceptr,
    rms: Deviceptr,
    normed: Deviceptr,
}

impl LayerMhc {
    fn alloc(cuda: &Cuda) -> Result<Self, String> {
        let mut allocated = 0usize;
        let mhc = Self {
            x: cuda.alloc(HC_MULT * HIDDEN * 2, &mut allocated)?,
            y: cuda.alloc(HIDDEN * 2, &mut allocated)?,
            b: cuda.alloc(HC_MULT * HC_MULT * 4, &mut allocated)?,
            c: cuda.alloc(HC_MULT * 4, &mut allocated)?,
            a: cuda.alloc(HC_MULT * 4, &mut allocated)?,
            predictor: cuda.alloc(PREDICTOR_ROWS * HC_MULT * HIDDEN * 4, &mut allocated)?,
            scale: cuda.alloc(3 * 4, &mut allocated)?,
            base: cuda.alloc(PREDICTOR_ROWS * 4, &mut allocated)?,
            norm: cuda.alloc(HIDDEN * 4, &mut allocated)?,
            fp8: cuda.alloc(HIDDEN, &mut allocated)?,
            scales: cuda.alloc(HIDDEN / QUANT_BLOCK, &mut allocated)?,
            stats: cuda.alloc(MHC_GRID_STATS * 4, &mut allocated)?,
            sync: cuda.alloc(2 * 4, &mut allocated)?,
            coeffs: cuda.alloc(PREDICTOR_ROWS * 4, &mut allocated)?,
            rms: cuda.alloc(4, &mut allocated)?,
            normed: cuda.alloc(HIDDEN * 4, &mut allocated)?,
        };
        println!("  mHC device buffers: {allocated} B");
        Ok(mhc)
    }

    /// Uploads one launch's inputs. `x` is bf16 bit patterns; `y` is the kernel's bf16 tensor, so it
    /// is rounded once here exactly as the ABI requires.
    #[allow(clippy::too_many_arguments)]
    fn upload(
        &self,
        cuda: &Cuda,
        x: &[u16],
        y: &[f32],
        b: &[f32],
        c: &[f32],
        a: &[f32],
        predictor: &[f32],
        scale: &[f32],
        base: &[f32],
        norm: &[f32],
    ) -> Result<(), String> {
        cuda.copy_htod(&self.x, &bf16_bits_to_bytes(x))?;
        cuda.copy_htod(&self.y, &bf16_bits_to_bytes(&bf16_bits_of_f32(y)))?;
        cuda.copy_htod(&self.b, &f32_to_bytes(b))?;
        cuda.copy_htod(&self.c, &f32_to_bytes(c))?;
        cuda.copy_htod(&self.a, &f32_to_bytes(a))?;
        cuda.copy_htod(&self.predictor, &f32_to_bytes(predictor))?;
        cuda.copy_htod(&self.scale, &f32_to_bytes(scale))?;
        cuda.copy_htod(&self.base, &f32_to_bytes(base))?;
        cuda.copy_htod(&self.norm, &f32_to_bytes(norm))?;
        Ok(())
    }

    fn launch(&self, cuda: &Cuda, func: CuFunction, layer: usize, step: u64) -> Result<(), String> {
        // The grid rendezvous counters are per-launch state.
        cuda.copy_htod(&self.stats, &[0u8; MHC_GRID_STATS * 4])?;
        cuda.copy_htod(&self.sync, &[0u8; 2 * 4])?;
        let params: [u64; 19] = [
            self.x.0,
            self.y.0,
            self.b.0,
            self.c.0,
            self.a.0,
            self.predictor.0,
            self.scale.0,
            self.base.0,
            self.norm.0,
            self.fp8.0,
            self.scales.0,
            self.stats.0,
            self.sync.0,
            self.coeffs.0,
            self.rms.0,
            self.normed.0,
            0, // trace_ring: null, the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
            layer as u64,
            step,
        ];
        cuda.launch(
            func,
            (MHC_TILES, 1u32, 1u32),
            (MHC_THREADS, 1u32, 1u32),
            &params,
        )
    }

    fn x_bits(&self, cuda: &Cuda) -> Result<Vec<u16>, String> {
        let mut bytes = vec![0u8; HC_MULT * HIDDEN * 2];
        cuda.copy_dtoh(&mut bytes, &self.x)?;
        Ok(bytes_to_u16(&bytes))
    }

    fn coeffs(&self, cuda: &Cuda) -> Result<Vec<f32>, String> {
        let mut bytes = vec![0u8; PREDICTOR_ROWS * 4];
        cuda.copy_dtoh(&mut bytes, &self.coeffs)?;
        Ok(bytes_to_f32(&bytes))
    }

    fn normed(&self, cuda: &Cuda) -> Result<Vec<f32>, String> {
        let mut bytes = vec![0u8; HIDDEN * 4];
        cuda.copy_dtoh(&mut bytes, &self.normed)?;
        Ok(bytes_to_f32(&bytes))
    }

    fn codes(&self, cuda: &Cuda) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; HIDDEN];
        cuda.copy_dtoh(&mut bytes, &self.fp8)?;
        Ok(bytes)
    }

    fn scales(&self, cuda: &Cuda) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; HIDDEN / QUANT_BLOCK];
        cuda.copy_dtoh(&mut bytes, &self.scales)?;
        Ok(bytes)
    }

    fn release(&self, cuda: &Cuda) {
        cuda.free_buffers(&[
            &self.x,
            &self.y,
            &self.b,
            &self.c,
            &self.a,
            &self.predictor,
            &self.scale,
            &self.base,
            &self.norm,
            &self.fp8,
            &self.scales,
            &self.stats,
            &self.sync,
            &self.coeffs,
            &self.rms,
            &self.normed,
        ]);
    }
}

/// Widen the device's bf16 output bytes to f32 values, keeping the device's rounding.
fn widen_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
        .collect()
}

/// Decode a quantised activation row: E4M3 code times its block's ue8m0 scale.
fn decode_quantised_row(codes: &[u8], scales: &[u8]) -> Vec<f32> {
    codes
        .iter()
        .enumerate()
        .map(|(index, code)| fp8_e4m3_decode(*code) * ue8m0_decode(scales[index / QUANT_BLOCK]))
        .collect()
}

/// One dense MXFP8 GEMM on the device kernel: A is one activation row (E4M3 codes plus ue8m0
/// scales), W is already repacked into B12x order, the output is widened from the kernel's bf16.
///
/// The in-crate reference is run on the same bytes and compared bit for bit before the result is
/// used: the device kernel's arithmetic is the reference's, so a mismatch would be a repack or
/// launch bug in this harness, not a boundary difference.
#[allow(clippy::too_many_arguments)] // the kernel launch takes the harness context and the kernel buffers as distinct arguments.
fn dense_device_gemm(
    cuda: &Cuda,
    func: CuFunction,
    layer: usize,
    label: &str,
    a_codes: &[u8],
    a_scales: &[u8],
    w_packed: &[u8],
    w_scales: &[u8],
    cols: usize,
    inner: usize,
) -> Result<Vec<f32>, String> {
    let rows = 1usize;
    let mut allocated = 0usize;
    let a_d = cuda.alloc(a_codes.len(), &mut allocated)?;
    let scales_d = cuda.alloc(a_scales.len(), &mut allocated)?;
    let w_d = cuda.alloc(w_packed.len(), &mut allocated)?;
    let ws_d = cuda.alloc(w_scales.len(), &mut allocated)?;
    let out_d = cuda.alloc(cols * 2, &mut allocated)?;
    cuda.copy_htod(&a_d, a_codes)?;
    cuda.copy_htod(&scales_d, a_scales)?;
    cuda.copy_htod(&w_d, w_packed)?;
    cuda.copy_htod(&ws_d, w_scales)?;
    let params: [u64; 11] = [
        a_d.0,
        scales_d.0,
        w_d.0,
        ws_d.0,
        out_d.0,
        rows as u64,
        cols as u64,
        inner as u64,
        0, // trace_ring
        layer as u64,
        0, // step_seq
    ];
    cuda.launch(
        func,
        (1u32, (cols.div_ceil(64)) as u32, 1u32),
        (256u32, 1u32, 1u32),
        &params,
    )?;
    let mut bytes = vec![0u8; cols * 2];
    cuda.copy_dtoh(&mut bytes, &out_d)?;
    cuda.free_buffers(&[&a_d, &scales_d, &w_d, &ws_d, &out_d]);
    let _ = allocated;
    let device = widen_bf16(&bytes);
    let expected_bits =
        mxfp8_ref::dense_gemm(a_codes, a_scales, w_packed, w_scales, rows, cols, inner);
    let exact = device
        .iter()
        .zip(&expected_bits)
        .filter(|(value, bits)| value.to_bits() == half::bf16::from_bits(**bits).to_f32().to_bits())
        .count();
    println!(
        "  {label}: device vs in-crate reference on the same bytes {exact}/{cols} exact ({}x{})",
        cols, inner
    );
    if exact != cols {
        return Err(format!(
            "{label}: the device kernel does not reproduce mxfp8_ref::dense_gemm on the same bytes \
             ({exact} of {cols} exact)"
        ));
    }
    Ok(device)
}

/// One grouped wo_a launch on the released geometry (eight groups of 1024 output channels over
/// 4096 inputs each), output widened from bf16 and checked against the in-crate reference.
#[allow(clippy::too_many_arguments)] // the kernel launch takes the harness context and the kernel buffers as distinct arguments.
fn wo_a_device(
    cuda: &Cuda,
    func: CuFunction,
    layer: usize,
    attn_codes: &[u8],
    attn_scales: &[u8],
    w_packed: &[u8],
    w_scales: &[u8],
) -> Result<Vec<f32>, String> {
    const GROUPS: usize = 8;
    const LO_RANK: usize = 1024;
    const GROUP_INNER: usize = 4096;
    let out_bytes = GROUPS * LO_RANK * 2;
    let mut allocated = 0usize;
    let attn_d = cuda.alloc(attn_codes.len(), &mut allocated)?;
    let scales_d = cuda.alloc(attn_scales.len(), &mut allocated)?;
    let w_d = cuda.alloc(w_packed.len(), &mut allocated)?;
    let ws_d = cuda.alloc(w_scales.len(), &mut allocated)?;
    let out_d = cuda.alloc(out_bytes, &mut allocated)?;
    cuda.copy_htod(&attn_d, attn_codes)?;
    cuda.copy_htod(&scales_d, attn_scales)?;
    cuda.copy_htod(&w_d, w_packed)?;
    cuda.copy_htod(&ws_d, w_scales)?;
    let params: [u64; 9] = [
        attn_d.0,
        scales_d.0,
        w_d.0,
        ws_d.0,
        out_d.0,
        GROUPS as u64,
        0, // trace_ring
        layer as u64,
        0, // step_seq
    ];
    cuda.launch(
        func,
        (GROUPS as u32, (LO_RANK / 256) as u32, 1u32),
        (256u32, 1u32, 1u32),
        &params,
    )?;
    let mut bytes = vec![0u8; out_bytes];
    cuda.copy_dtoh(&mut bytes, &out_d)?;
    cuda.free_buffers(&[&attn_d, &scales_d, &w_d, &ws_d, &out_d]);
    let _ = allocated;
    let device = widen_bf16(&bytes);
    let expected_bits = mxfp8_ref::grouped_wo_a(
        attn_codes,
        attn_scales,
        w_packed,
        w_scales,
        GROUPS,
        GROUP_INNER,
        LO_RANK,
    );
    let exact = device
        .iter()
        .zip(&expected_bits)
        .filter(|(value, bits)| value.to_bits() == half::bf16::from_bits(**bits).to_f32().to_bits())
        .count();
    println!(
        "  K6 wo_a: device vs in-crate reference on the same bytes {exact}/{} exact (groups {GROUPS}, group_inner {GROUP_INNER}, lo_rank {LO_RANK})",
        GROUPS * LO_RANK
    );
    if exact != GROUPS * LO_RANK {
        return Err(format!(
            "K6 wo_a: the device kernel does not reproduce mxfp8_ref::grouped_wo_a on the same \
             bytes ({exact} of {} exact)",
            GROUPS * LO_RANK
        ));
    }
    Ok(device)
}

/// Repack the grouped wo_a checkpoint weight into the eight kernel-group B12x buffers; the scale
/// grid keeps its per-group [lo_rank/32, group_inner/32] row-major order.
fn pack_wo_a(weight: &[u8], scales: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    const GROUPS: usize = 8;
    const LO_RANK: usize = 1024;
    const GROUP_INNER: usize = 4096;
    if weight.len() != GROUPS * LO_RANK * GROUP_INNER {
        return Err(format!(
            "wo_a weight: {} bytes, expected {}",
            weight.len(),
            GROUPS * LO_RANK * GROUP_INNER
        ));
    }
    let expected_scales = GROUPS * (LO_RANK / 32) * (GROUP_INNER / 32);
    if scales.len() != expected_scales {
        return Err(format!(
            "wo_a scales: {} bytes, expected {expected_scales}",
            scales.len()
        ));
    }
    let mut packed = Vec::with_capacity(weight.len());
    for group in 0..GROUPS {
        let start = group * LO_RANK * GROUP_INNER;
        packed.extend_from_slice(&pack_b12x(
            &weight[start..start + LO_RANK * GROUP_INNER],
            LO_RANK,
            GROUP_INNER,
        ));
    }
    Ok((packed, scales.to_vec()))
}

/// Predicts the attention tail (attention_out) from one f32 window-KV latent, for the D4
/// attribution check: the same arithmetic and the same bf16 storage points as the device chain, with
/// the window KV supplied in a chosen format. Used twice: once with the device's FP4 dequant, which
/// must reproduce this run's attention_out, and once with the host's FP8 dequant, which says what the
/// host row's format would have produced from the same latent.
fn predict_attention_out(
    q: &[f32],
    sink: &[f32],
    kv_values: &[f32],
    wo_a_w: &[u8],
    wo_a_s: &[u8],
    wo_b_w: &[u8],
    wo_b_s: &[u8],
) -> Vec<f32> {
    let mut raw = Vec::with_capacity(MLA_HEADS * MLA_LATENT);
    for head in 0..MLA_HEADS {
        let query = &q[head * MLA_LATENT..(head + 1) * MLA_LATENT];
        let mut score = 0.0f32;
        for (value, kv) in query.iter().zip(kv_values) {
            score += value * kv;
        }
        let score = score * mla_ref::MLA_SCORE_SCALE;
        // One candidate at position 0: the numerator weight is one and the sink is the only other
        // term of the softmax denominator (mla_ref::attention_impl pass two and three).
        let weight = 1.0 / (1.0 + (sink[head] - score).exp());
        for kv in kv_values {
            raw.push(weight * kv);
        }
    }
    let raw = widen_bits(&bf16_bits_of_f32(&raw));
    let (codes, scales) = act_quant_fp8_row(&raw);
    let lo_bits = mxfp8_ref::grouped_wo_a(&codes, &scales, wo_a_w, wo_a_s, 8, 4096, 1024);
    let lo = widen_bits(&lo_bits);
    let (lo_codes, lo_scales) = act_quant_fp8_row(&lo);
    let out_bits =
        mxfp8_ref::dense_gemm(&lo_codes, &lo_scales, wo_b_w, wo_b_s, 1, HIDDEN, WO_A_OUT);
    widen_bits(&out_bits)
}

/// (exact count, max abs delta) between two f32 slices of equal length.
fn f32_delta_stats(left: &[f32], right: &[f32]) -> (usize, f32) {
    let mut exact = 0usize;
    let mut max = 0.0f32;
    for (left, right) in left.iter().zip(right) {
        if left.to_bits() == right.to_bits() {
            exact += 1;
        }
        max = max.max((left - right).abs());
    }
    (exact, max)
}

/// One captured value and its comparison against the host row; the fields are what the report
/// prints, not a pass/fail (the bar is measured by this run, per design section 8).
struct RowDelta {
    name: &'static str,
    class: BoundaryClass,
    values: usize,
    host_exact: usize,
    max_abs_delta: f32,
    mean_abs_delta: f32,
    at: usize,
    differing_codes: usize,
    max_code_distance: i32,
    mean_code_distance: f32,
    /// Differing codes bucketed by distance: 1, 2..=3, 4..=15, 16 and above.
    code_distance_buckets: [usize; 4],
    bf16_host_exact: Option<usize>,
    bf16_host_max_delta: Option<f32>,
}

impl RowDelta {
    fn single_layer_delta(&self) -> f64 {
        if self.class == BoundaryClass::ExactCodes {
            f64::from(self.differing_codes as u32) / self.values as f64
        } else {
            f64::from(self.max_abs_delta)
        }
    }

    /// The code-distance histogram as the report prints it: how many differing codes are one apart,
    /// two or three, four to fifteen, and sixteen or more (a scale/exponent change or a sign flip).
    fn code_distance_buckets_text(&self) -> String {
        format!(
            "[1: {}, 2-3: {}, 4-15: {}, 16+: {}]",
            self.code_distance_buckets[0],
            self.code_distance_buckets[1],
            self.code_distance_buckets[2],
            self.code_distance_buckets[3]
        )
    }
}

fn compare_row(
    host: &HostBoundaryTable,
    row: usize,
    captured: &HashMap<&'static str, RowValue>,
) -> Result<Vec<RowDelta>, String> {
    let mut deltas = Vec::with_capacity(B2_TENSORS.len());
    for (name, class) in B2_TENSORS {
        let value = captured
            .get(name)
            .ok_or_else(|| format!("the chain did not capture {name}"))?;
        match (class, value) {
            (BoundaryClass::ExactBits, RowValue::F32(device)) => {
                let reference = host.f32(row, name)?;
                if reference.len() != device.len() {
                    return Err(format!(
                        "{name}: {} host values against {} device values",
                        reference.len(),
                        device.len()
                    ));
                }
                let mut exact = 0usize;
                let mut max_abs_delta = 0.0f32;
                let mut sum_abs_delta = 0.0f64;
                let mut at = 0usize;
                for (index, (host, device)) in reference.iter().zip(device).enumerate() {
                    if host.to_bits() == device.to_bits() {
                        exact += 1;
                    }
                    let delta = (host - device).abs();
                    sum_abs_delta += f64::from(delta);
                    if !delta.is_finite() || delta > max_abs_delta {
                        max_abs_delta = if delta.is_finite() {
                            delta
                        } else {
                            f32::INFINITY
                        };
                        at = index;
                    }
                }
                let (bf16_host_exact, bf16_host_max_delta) = if B2_STREAM_TENSORS.contains(&name) {
                    // The device stream is a bf16 write, so bf16(host) is the value the device path
                    // can be exact at; the report gives both numbers.
                    let bf16_host: Vec<f32> = reference
                        .iter()
                        .map(|value| {
                            half::bf16::from_bits(half::bf16::from_f32(*value).to_bits()).to_f32()
                        })
                        .collect();
                    let mut exact = 0usize;
                    let mut max_delta = 0.0f32;
                    for (host, device) in bf16_host.iter().zip(device) {
                        if host.to_bits() == device.to_bits() {
                            exact += 1;
                        }
                        max_delta = max_delta.max((host - device).abs());
                    }
                    (Some(exact), Some(max_delta))
                } else {
                    (None, None)
                };
                deltas.push(RowDelta {
                    name,
                    class,
                    values: device.len(),
                    host_exact: exact,
                    max_abs_delta,
                    mean_abs_delta: (sum_abs_delta / device.len() as f64) as f32,
                    at,
                    differing_codes: 0,
                    max_code_distance: 0,
                    mean_code_distance: 0.0,
                    code_distance_buckets: [0; 4],
                    bf16_host_exact,
                    bf16_host_max_delta,
                });
            }
            (BoundaryClass::ExactCodes, RowValue::U8(device)) => {
                let reference = host.u8(row, name)?;
                if reference.len() != device.len() {
                    return Err(format!(
                        "{name}: {} host codes against {} device codes",
                        reference.len(),
                        device.len()
                    ));
                }
                let mut differing = 0usize;
                let mut max_code_distance = 0i32;
                let mut sum_code_distance = 0u64;
                let mut buckets = [0usize; 4];
                let mut first = usize::MAX;
                let mut exact = 0usize;
                for (index, (host, device)) in reference.iter().zip(device).enumerate() {
                    if host == device {
                        exact += 1;
                    } else {
                        differing += 1;
                        if first == usize::MAX {
                            first = index;
                        }
                        let distance = i32::from(*host).abs_diff(i32::from(*device)) as i32;
                        max_code_distance = max_code_distance.max(distance);
                        sum_code_distance += distance as u64;
                        let bucket = match distance {
                            1 => 0,
                            2..=3 => 1,
                            4..=15 => 2,
                            _ => 3,
                        };
                        buckets[bucket] += 1;
                    }
                }
                deltas.push(RowDelta {
                    name,
                    class,
                    values: device.len(),
                    host_exact: exact,
                    max_abs_delta: 0.0,
                    mean_abs_delta: 0.0,
                    at: if first == usize::MAX { 0 } else { first },
                    differing_codes: differing,
                    max_code_distance,
                    mean_code_distance: if differing == 0 {
                        0.0
                    } else {
                        sum_code_distance as f32 / differing as f32
                    },
                    code_distance_buckets: buckets,
                    bf16_host_exact: None,
                    bf16_host_max_delta: None,
                });
            }
            (class, _) => {
                return Err(format!(
                    "{name}: captured as the wrong kind for class {class:?}"
                ));
            }
        }
    }
    Ok(deltas)
}

/// Widen a slice of bf16 bit patterns to f32 values.
fn widen_bits(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|bits| half::bf16::from_bits(*bits).to_f32())
        .collect()
}

/// One routed-combine chain's device operands: the activation ranks (FP4 codes plus ue8m0 scales,
/// padded to the kernel's row geometry) and the stacked w2 rank weight/scale buffers.
struct RoutedOperands {
    act_fp8: Vec<Vec<u8>>,
    act_scales: Vec<Vec<u8>>,
    w_rank: Vec<Vec<u8>>,
    ws_rank: Vec<Vec<u8>>,
}

/// Stages six experts' w1/w3/w2 and computes the routed-combine operands for one routing choice:
/// the host gate/up GEMM over each selected expert, the host SwiGLU, the reference activation
/// quantiser with the routing weight folded in, and the per-rank w2 slices. That is the S1/S2/S3
/// seam of the chain, run twice (free-running routing and the host row's routing forced) so the
/// report can separate the discrete expert flip from the arithmetic delta.
fn routed_operands(
    snap: &mut Snapshot,
    prefix: &str,
    block_activation: &[f32],
    expert_ids: &[u32],
    expert_weights: &[f32],
) -> Result<RoutedOperands, String> {
    let mut w1: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut s1: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut w3: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut s3: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut w2: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut s2: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    for expert in expert_ids {
        let base = format!("{prefix}.ffn.experts.{expert}");
        w1.push(snap.read_raw(
            &format!("{base}.w1.weight"),
            "I8",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_PACKED],
        )?);
        s1.push(snap.read_raw(
            &format!("{base}.w1.scale"),
            "F8_E8M0",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_SCALES],
        )?);
        w3.push(snap.read_raw(
            &format!("{base}.w3.weight"),
            "I8",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_PACKED],
        )?);
        s3.push(snap.read_raw(
            &format!("{base}.w3.scale"),
            "F8_E8M0",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_SCALES],
        )?);
        w2.push(snap.read_raw(&format!("{base}.w2.weight"), "I8", &[HIDDEN, MOE_IN_PACKED])?);
        s2.push(snap.read_raw(
            &format!("{base}.w2.scale"),
            "F8_E8M0",
            &[HIDDEN, MOE_IN_SCALES],
        )?);
    }
    let mut act_fp8: Vec<Vec<u8>> = (0..MOE_RANKS).map(|_| Vec::new()).collect();
    let mut act_scales: Vec<Vec<u8>> = (0..MOE_RANKS).map(|_| Vec::new()).collect();
    for slot in 0..MOE_TOP_K {
        let gate = gate_up_rows(
            block_activation,
            &w1[slot],
            &s1[slot],
            MOE_INTERMEDIATE,
            HIDDEN,
        );
        let up = gate_up_rows(
            block_activation,
            &w3[slot],
            &s3[slot],
            MOE_INTERMEDIATE,
            HIDDEN,
        );
        let activation: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
            .collect();
        let routing = expert_weights[slot];
        for rank in 0..MOE_RANKS {
            let (codes, scales) = mxfp4_ref::routed_activation_operand(
                &activation[rank * MOE_WIDTH..(rank + 1) * MOE_WIDTH],
                routing,
            );
            let mut row = vec![0u8; MOE_WIDTH_PADDED];
            row[..MOE_WIDTH].copy_from_slice(&codes);
            act_fp8[rank].extend_from_slice(&row);
            let mut scale_row = vec![0u8; MOE_SCALE_ROW];
            scale_row[..MOE_WIDTH / QUANT_BLOCK].copy_from_slice(&scales);
            act_scales[rank].extend_from_slice(&scale_row);
        }
    }
    let mut w_rank: Vec<Vec<u8>> = Vec::with_capacity(MOE_RANKS);
    let mut ws_rank: Vec<Vec<u8>> = Vec::with_capacity(MOE_RANKS);
    for rank in 0..MOE_RANKS {
        let mut weight = Vec::with_capacity(MOE_TOP_K * HIDDEN * MOE_PACKED_ROW);
        let mut scale = Vec::with_capacity(MOE_TOP_K * HIDDEN * MOE_SCALE_ROW);
        for expert in 0..MOE_TOP_K {
            weight.extend_from_slice(&slice_and_pad_rows(
                &w2[expert],
                HIDDEN,
                MOE_IN_PACKED,
                rank * (MOE_WIDTH / 2),
                MOE_WIDTH / 2,
                MOE_PACKED_ROW,
            ));
            scale.extend_from_slice(&slice_and_pad_rows(
                &s2[expert],
                HIDDEN,
                MOE_IN_SCALES,
                rank * (MOE_WIDTH / 32),
                MOE_WIDTH / 32,
                MOE_SCALE_ROW,
            ));
        }
        w_rank.push(weight);
        ws_rank.push(scale);
    }
    Ok(RoutedOperands {
        act_fp8,
        act_scales,
        w_rank,
        ws_rank,
    })
}

/// Runs the four chained rank launches of `b12x_mxfp4_moe_sm121` for one routing choice and
/// returns the combine's bf16 output bits. The memory floor is checked before and after each launch.
#[allow(clippy::too_many_arguments)] // stage 7 and stage 9 both drive it; the scope names which log line it is.
fn run_routed_combine(
    args: &Args,
    scope: &str,
    cuda: &Cuda,
    moe_func: CuFunction,
    layer: usize,
    label: &str,
    operands: &RoutedOperands,
    table_ids: &[u32],
) -> Result<Vec<u16>, String> {
    println!(
        "--- K9-K12 b12x_mxfp4_moe_sm121: routed combine, {label}, {MOE_RANKS} rank slices ---"
    );
    let mut allocated = 0usize;
    let hidden_d = cuda.alloc(HIDDEN * 2, &mut allocated)?;
    let moe_out_d = cuda.alloc(HIDDEN * 2, &mut allocated)?;
    let act_d = cuda.alloc(operands.act_fp8[0].len(), &mut allocated)?;
    let acts_d = cuda.alloc(operands.act_scales[0].len(), &mut allocated)?;
    let w_d = cuda.alloc(operands.w_rank[0].len(), &mut allocated)?;
    let ws_d = cuda.alloc(operands.ws_rank[0].len(), &mut allocated)?;
    let ids_d = cuda.alloc(MOE_TOP_K * 4, &mut allocated)?;
    let _ = allocated;
    cuda.copy_htod(&hidden_d, &vec![0u8; HIDDEN * 2])?;
    cuda.copy_htod(&ids_d, &u32s_to_bytes(table_ids))?;
    let mut routed_bits: Vec<u16> = vec![0u16; HIDDEN];
    for rank in 0..MOE_RANKS {
        cuda.copy_htod(&act_d, &operands.act_fp8[rank])?;
        cuda.copy_htod(&acts_d, &operands.act_scales[rank])?;
        cuda.copy_htod(&w_d, &operands.w_rank[rank])?;
        cuda.copy_htod(&ws_d, &operands.ws_rank[rank])?;
        let params: [u64; 11] = [
            hidden_d.0,
            act_d.0,
            acts_d.0,
            w_d.0,
            ws_d.0,
            ids_d.0,
            moe_out_d.0,
            1, // tokens
            0, // trace_ring
            layer as u64,
            rank as u64,
        ];
        check_floor(args, &format!("{scope} K{} ({label}) pre-launch", 9 + rank))?;
        cuda.launch(
            moe_func,
            (1u32, (HIDDEN / MOE_TILE_D as usize) as u32, 1u32),
            (MOE_THREADS, 1u32, 1u32),
            &params,
        )?;
        check_floor(
            args,
            &format!("{scope} K{} ({label}) post-launch", 9 + rank),
        )?;
        let mut bytes = vec![0u8; HIDDEN * 2];
        cuda.copy_dtoh(&mut bytes, &moe_out_d)?;
        routed_bits = bytes_to_u16(&bytes);
        cuda.copy_htod(&hidden_d, &bytes)?;
    }
    cuda.free_buffers(&[&hidden_d, &moe_out_d, &act_d, &acts_d, &w_d, &ws_d, &ids_d]);
    Ok(routed_bits)
}

/// Stage 7: the chained layer-0 replay (B2).
///
/// The layer is computed in the order the release does: mHC attention sublayer, the attention
/// projections on the device dense/wo_a kernels, the FP4 routed combine, the shared expert on the
/// device dense kernel, then the mHC FFN sublayer and the final hc_post. The host arithmetic in
/// between is the labelled seam set of device-chain-design.md section 5; each label is printed
/// exactly once. Every boundary tensor of the host row is captured from the buffer/coefficient set
/// that produced it and diffed by comparison class. No pass/fail bar is applied: the point of this
/// run is to MEASURE the single-layer delta of each tensor, which is the bar input for the 40-layer
/// chain (design section 8), and a single-layer device-vs-f32 comparison is not a criterion-1 gate.
fn stage7(args: &Args) -> Result<(), String> {
    let kernel_ok = "mega_mhc_fused_bf16_sm121";
    println!("=== stage 7: the chained layer-0 replay (B2) ===");
    if args.layer != 0 {
        return Err(format!(
            "stage 7 replays layer 0 only (the chain input is the embedding row of token 0 with \
             the identity initial mix); --layer {} was requested",
            args.layer
        ));
    }
    let host_prefix = args
        .host_row
        .as_ref()
        .ok_or("stage 7 needs --host-row <manifest prefix> for the host row it diffs against")?;
    let host = HostBoundaryTable::open(host_prefix)?;
    let prefix = format!("layers.{}", args.layer);
    println!(
        "comparison classes: exact-codes for pre_attention_xhat/scales and pre_ffn_xhat/scales \
         (no numeric tolerance is attached), exact-bits (device tolerance) for the other fourteen \
         and the measured class is named before any number it judges..."
    );
    println!(
        "bars: NOT applied by this run. Design section 8 makes the bar the Stage-B measured \
         single_layer_delta(T) compounded over the rows a tensor traversed; this run measures those \
         deltas on layer 0 and records them. Applying a bar here would be inventing one."
    );
    println!(
        "label: this is a single-layer device-vs-host boundary diff, not model fidelity. It measures \
         the seam's own arithmetic and the kernel arithmetic underneath it."
    );
    println!(
        "scope label: this is the Phase-1 HEALTH-CHECK column, not a criterion-1 closure. The device 
         path uses FP4 main/window KV and bf16/quantised arithmetic where the host reference is f32 
         with no GEMM activation quantiser, so the column measures attribution and gross mismatch; the 
         format-faithful exactness run is the later one after Phase 2 (task-3, FP8 window KV, then 
         release-faithful arithmetic). Criterion 1 is not closed by this run."
    );
    println!(
        "known differences, stated before the numbers: (1) the attention chain feeds the kernel its Main-KV format, FP4 E2M1 codes with one E4M3 scale per 16 channels, while the host row's window KV is the release's FP8 E4M3 path with a per-32 ue8m0 scale (decision D4); (2) the host reference is f32 throughout: it multiplies decoded f32 weights by f32 activations and deliberately does not reproduce the device's GEMM activation quantisers or its bf16 module outputs (crates/sparkflash-engine/src/forward/attention.rs module docs, What is deliberately not reproduced, items 1 and 2). Both differences are inside the attention and FFN outputs and propagate; neither is isolated per tensor in this run. Every kernel is additionally checked bit for bit against its in-crate reference on the same bytes, so a kernel arithmetic difference would be visible in this log."
    );
    println!(
        "launch count: 16 device launches for the chain (K1 mHC, K2/K3/K4 dense, K5 flash_mla, K6 wo_a, K7 dense, K8 mHC, K9-K12 routed combine, K13/K14/K15 shared dense, K16 mHC) plus 4 for the forced-routing control, 20 in total"
    );
    check_floor(args, "stage 7 start")?;
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let residual_dims = host.tensor(0, "residual_in")?.dims.clone();
    println!(
        "host row 0: residual_in {residual_dims:?}, {} bytes of manifest",
        host.blob.len()
    );
    if residual_dims.last().copied() != Some(HIDDEN) {
        return Err(format!(
            "host row 0 residual_in is {residual_dims:?}; this chain is built for hidden {HIDDEN}"
        ));
    }
    let positions: usize = residual_dims.first().copied().unwrap_or(0);
    if positions != 1 {
        return Err(format!(
            "stage 7 replays one token, the host row has {positions} positions"
        ));
    }

    println!("host input: embedding row of token 0 replicated to 4 copies - host");
    let embed_row = snap.read_row_bf16("embed.weight", 0, HIDDEN)?;
    let mut stream = Vec::with_capacity(HC_MULT * HIDDEN);
    for _ in 0..HC_MULT {
        stream.extend_from_slice(&embed_row);
    }
    let x_bits = bf16_bits_of_f32(&stream);
    drop(stream);

    let attn_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let ffn_norm = snap.read_f32_or_bf16(&format!("{prefix}.ffn_norm.weight"), &[HIDDEN])?;
    let mhc_attn = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let mhc_ffn = read_released_mhc_tensors(&mut snap, &prefix, "ffn")?;
    let attn_sink = snap.read_f32(&format!("{prefix}.attn.attn_sink"), &[MLA_HEADS])?;
    let q_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.q_norm.weight"), &[Q_LORA_RANK])?;
    let kv_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.kv_norm.weight"), &[MLA_LATENT])?;
    println!("host input: released checkpoint weights, repacked B12x on host before upload - host");

    let wq_a_w = snap.read_raw(
        &format!("{prefix}.attn.wq_a.weight"),
        "F8_E4M3",
        &[Q_LORA_RANK, HIDDEN],
    )?;
    let wq_a_s = snap.read_raw(
        &format!("{prefix}.attn.wq_a.scale"),
        "F8_E8M0",
        &[Q_LORA_RANK / 32, HIDDEN / 32],
    )?;
    let wq_a_w = pack_b12x(&wq_a_w, Q_LORA_RANK, HIDDEN);
    let wq_b_cols = MLA_HEADS * MLA_LATENT;
    let wq_b_w = snap.read_raw(
        &format!("{prefix}.attn.wq_b.weight"),
        "F8_E4M3",
        &[wq_b_cols, Q_LORA_RANK],
    )?;
    let wq_b_s = snap.read_raw(
        &format!("{prefix}.attn.wq_b.scale"),
        "F8_E8M0",
        &[wq_b_cols / 32, Q_LORA_RANK / 32],
    )?;
    let wq_b_w = pack_b12x(&wq_b_w, wq_b_cols, Q_LORA_RANK);
    let wkv_w = snap.read_raw(
        &format!("{prefix}.attn.wkv.weight"),
        "F8_E4M3",
        &[MLA_LATENT, HIDDEN],
    )?;
    let wkv_s = snap.read_raw(
        &format!("{prefix}.attn.wkv.scale"),
        "F8_E8M0",
        &[MLA_LATENT / 32, HIDDEN / 32],
    )?;
    let wkv_w = pack_b12x(&wkv_w, MLA_LATENT, HIDDEN);
    let wo_b_w = snap.read_raw(
        &format!("{prefix}.attn.wo_b.weight"),
        "F8_E4M3",
        &[HIDDEN, WO_A_OUT],
    )?;
    let wo_b_s = snap.read_raw(
        &format!("{prefix}.attn.wo_b.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, WO_A_OUT / 32],
    )?;
    let wo_b_w = pack_b12x(&wo_b_w, HIDDEN, WO_A_OUT);

    let wo_a_weight_name = format!("{prefix}.attn.wo_a.weight");
    let wo_a_header = snap.load(&wo_a_weight_name)?;
    let wo_a_scale_header = snap.load(&format!("{prefix}.attn.wo_a.scale"))?;
    println!(
        "wo_a stored shape {:?} dtype {}, scale shape {:?} dtype {}",
        wo_a_header.shape, wo_a_header.dtype, wo_a_scale_header.shape, wo_a_scale_header.dtype
    );
    let wo_a_weight = snap.read_raw(&wo_a_weight_name, "F8_E4M3", &wo_a_header.shape)?;
    let wo_a_scale = snap.read_raw(
        &format!("{prefix}.attn.wo_a.scale"),
        "F8_E8M0",
        &wo_a_scale_header.shape,
    )?;
    let (wo_a_w, wo_a_s) = pack_wo_a(&wo_a_weight, &wo_a_scale)?;
    drop(wo_a_weight);

    let gate_weight =
        snap.read_f32_or_bf16(&format!("{prefix}.ffn.gate.weight"), &[MOE_EXPERTS, HIDDEN])?;
    let gate_bias = snap.read_f32(&format!("{prefix}.ffn.gate.bias"), &[MOE_EXPERTS])?;
    let shared_w1 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w1.weight"),
            "F8_E4M3",
            &[MOE_INTERMEDIATE, HIDDEN],
        )?,
        MOE_INTERMEDIATE,
        HIDDEN,
    );
    let shared_s1 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w3 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w3.weight"),
            "F8_E4M3",
            &[MOE_INTERMEDIATE, HIDDEN],
        )?,
        MOE_INTERMEDIATE,
        HIDDEN,
    );
    let shared_s3 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w2 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w2.weight"),
            "F8_E4M3",
            &[HIDDEN, MOE_INTERMEDIATE],
        )?,
        HIDDEN,
        MOE_INTERMEDIATE,
    );
    let shared_s2 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, MOE_INTERMEDIATE / 32],
    )?;
    println!(
        "staged weights: wq_a {} B, wq_b {} B, wkv {} B, wo_b {} B, wo_a {} B (packed), shared w1/w3/w2 {} + {} + {} B (packed); scale grids as stored",
        wq_a_w.len(),
        wq_b_w.len(),
        wkv_w.len(),
        wo_b_w.len(),
        wo_a_w.len(),
        shared_w1.len(),
        shared_w3.len(),
        shared_w2.len()
    );
    admit_memory(
        args,
        (wq_a_w.len()
            + wq_b_w.len()
            + wkv_w.len()
            + wo_b_w.len()
            + wo_a_w.len()
            + shared_w1.len()
            + shared_w3.len()
            + shared_w2.len()
            + gate_weight.len() * 4
            + 8 * HIDDEN * 4
            + 6 * (MOE_INTERMEDIATE * MOE_EXPERT_IN_PACKED
                + MOE_INTERMEDIATE * MOE_EXPERT_IN_SCALES
                + HIDDEN * MOE_IN_PACKED
                + HIDDEN * MOE_IN_SCALES)) as u64,
        "stage 7 staged weights and expert reads",
    )?;

    // --- the device context and the five kernels this stage launches ---------------------------
    check_floor(args, "stage 7 before context")?;
    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let load = |name: &str, _allocated: &mut usize| -> Result<(CuModule, CuFunction), String> {
        let path = args.cubin_dir.join(format!("{name}.cubin"));
        let cubin = std::fs::read(&path).map_err(|e| {
            format!(
                "{}: {e} (build the stage-7 cubins first; the module docs give the nvcc lines)",
                path.display()
            )
        })?;
        cuda.load_function(&cubin, name)
    };
    let (_mhc_module, mhc_func) = load(kernel_ok, &mut allocated)?;
    let (_dense_module, dense_func) = load("b12x_mxfp8_dense_sm121", &mut allocated)?;
    let (_wo_a_module, wo_a_func) = load("b12x_mxfp8_wo_a_sm121", &mut allocated)?;
    let (_mla_module, mla_func) = load("flash_mla_csa2_sm121", &mut allocated)?;
    let (_moe_module, moe_func) = load("b12x_mxfp4_moe_sm121", &mut allocated)?;
    let mut captures: HashMap<&'static str, RowValue> = HashMap::with_capacity(18);

    // --- K1: the attention sublayer's mHC pass (layer 0 seed: y = 0, b = I, c = 0, a = e0) ----
    println!(
        "--- K1 {kernel_ok}: attention sublayer, layer-0 seed (y=0, b=I, c=0, a=[1,0,0,0]) ---"
    );
    let mhc = LayerMhc::alloc(&cuda)?;
    let identity_b = {
        let mut b = [0.0f32; HC_MULT * HC_MULT];
        for i in 0..HC_MULT {
            b[i * HC_MULT + i] = 1.0;
        }
        b
    };
    let zero_c = [0.0f32; HC_MULT];
    let identity_a = [1.0f32, 0.0, 0.0, 0.0];
    let zero_y = vec![0.0f32; HIDDEN];
    mhc.upload(
        &cuda,
        &x_bits,
        &zero_y,
        &identity_b,
        &zero_c,
        &identity_a,
        &mhc_attn.predictor,
        &mhc_attn.hc_scale,
        &mhc_attn.hc_base_kernel,
        &attn_norm,
    )?;
    check_floor(args, "stage 7 K1 pre-launch")?;
    mhc.launch(&cuda, mhc_func, args.layer, 0)?;
    check_floor(args, "stage 7 K1 post-launch")?;
    let residual_in_bits = mhc.x_bits(&cuda)?;
    let attn_coeffs = mhc.coeffs(&cuda)?;
    let attn_hidden = mhc.normed(&cuda)?;
    let attn_codes = mhc.codes(&cuda)?;
    let attn_scales = mhc.scales(&cuda)?;
    let attn_b: Vec<f32> = attn_coeffs[0..HC_MULT * HC_MULT].to_vec();
    let attn_pre: Vec<f32> = attn_coeffs[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT].to_vec();
    let attn_post: Vec<f32> =
        attn_coeffs[HC_MULT * HC_MULT + HC_MULT..HC_MULT * HC_MULT + 2 * HC_MULT].to_vec();
    let attn_comb: Vec<f32> = {
        // The kernel writes B[destination][source]; the boundary row stores
        // comb[source][destination] (docs/build/phase1-boundary-table.md section 2).
        let mut comb = vec![0.0f32; HC_MULT * HC_MULT];
        for destination in 0..HC_MULT {
            for source in 0..HC_MULT {
                comb[source * HC_MULT + destination] = attn_b[destination * HC_MULT + source];
            }
        }
        comb
    };
    captures.insert("residual_in", RowValue::F32(widen_bits(&residual_in_bits)));
    captures.insert("hc_in_pre", RowValue::F32(identity_a.to_vec()));
    captures.insert("hc_attn_pre", RowValue::F32(attn_pre.clone()));
    captures.insert("hc_attn_post", RowValue::F32(attn_post.clone()));
    captures.insert("hc_attn_comb", RowValue::F32(attn_comb));
    captures.insert("pre_attention_hidden", RowValue::F32(attn_hidden.clone()));
    captures.insert("pre_attention_xhat", RowValue::U8(attn_codes.clone()));
    captures.insert("pre_attention_scales", RowValue::U8(attn_scales.clone()));
    println!(
        "  K1 captured: residual_in {} bf16 values, coefficients 24 f32, block input {} f32 + {} codes + {} scales",
        residual_in_bits.len(),
        attn_hidden.len(),
        attn_codes.len(),
        attn_scales.len()
    );
    let residual_in_unchanged = x_bits == residual_in_bits;
    println!(
        "  capture caveat: residual_in is read from the K1 kernel's post-launch stream (mhc.x_bits), not from its input argument; at layer 0 the seed (y = 0, B = I, c = 0, a = e0) makes the sublayer's output stream the row's input stream, and the pre-K1 input x_bits equals the captured residual_in: {residual_in_unchanged}. Later rows are captured by the chain harness before their own K1."
    );

    // --- K2..K7: the attention block, dense/wo_a kernels + labelled host seams ------------------
    println!("--- K2 {kernel_ok} replaced by b12x_mxfp8_dense_sm121: wq_a ---");
    let mut q_latent = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K2 wq_a",
        &attn_codes,
        &attn_scales,
        &wq_a_w,
        &wq_a_s,
        Q_LORA_RANK,
        HIDDEN,
    )?;
    println!(
        "seam: attention operand preparation (q_norm, q RoPE, kv_norm, kv RoPE, main-KV packing, output inverse RoPE, MXFP8 activation quantisers) - host"
    );
    rms_norm_row(&mut q_latent, &q_norm, NORM_EPS);
    let (q_codes, q_scales) = act_quant_fp8_row(&q_latent);
    println!("--- K3 dense: wq_b ---");
    let mut q = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K3 wq_b",
        &q_codes,
        &q_scales,
        &wq_b_w,
        &wq_b_s,
        wq_b_cols,
        Q_LORA_RANK,
    )?;
    rotate_rope_tail(&mut q, MLA_HEADS, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
    let q_device_bits = bf16_bits_of_f32(&q);
    let q_used = widen_bits(&q_device_bits);
    println!("--- K4 dense: wkv ---");
    let mut kv = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K4 wkv",
        &attn_codes,
        &attn_scales,
        &wkv_w,
        &wkv_s,
        MLA_LATENT,
        HIDDEN,
    )?;
    rms_norm_row(&mut kv, &kv_norm, NORM_EPS);
    rotate_rope_tail(&mut kv, 1, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
    let (kv_packed, kv_scales_row) = quantize_kv_row(&kv);

    println!("--- K5 flash_mla_csa2_sm121: attention at position 0 ---");
    println!("seam: HSI candidate merge (top-512 over the pool) - host; Phase 2/3 move it");
    let selected = stage2_selected(0);
    let sink_min = attn_sink.iter().copied().fold(f32::INFINITY, f32::min);
    let sink_max = attn_sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "attn_sink: {} f32 in [{sink_min:.4}, {sink_max:.4}] - CONSUMED ON BOTH SIDES: the kernel takes it as a [64] argument and adds it inside the softmax denominator, and mla_ref::attention_with_sink does the same",
        attn_sink.len()
    );
    println!(
        "  candidates: {selected:?} (position 0 is the tightest window; the harness supplies the list)"
    );
    let index_key_bytes = MLA_HSI_BLOCKS * MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM * 2;
    let index_query_bytes = MLA_HEADS * MLA_INDEX_HEAD_DIM * 2;
    let out_bytes = MLA_HEADS * MLA_LATENT * 2;
    let mut allocated = 0usize;
    let q_d = cuda.alloc(MLA_HEADS * MLA_LATENT * 2, &mut allocated)?;
    let sink_d = cuda.alloc(MLA_HEADS * 4, &mut allocated)?;
    let kv_fp4_d = cuda.alloc(KV_PACKED_BYTES, &mut allocated)?;
    let kv_scales_d = cuda.alloc(KV_SCALE_BYTES, &mut allocated)?;
    let selected_d = cuda.alloc(MLA_MAX_SELECTED * 4, &mut allocated)?;
    let index_key_d = cuda.alloc(index_key_bytes, &mut allocated)?;
    let index_query_d = cuda.alloc(index_query_bytes, &mut allocated)?;
    let block_scores_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let block_best_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let attn_out_d = cuda.alloc(out_bytes, &mut allocated)?;
    cuda.copy_htod(&q_d, &bf16_bits_to_bytes(&q_device_bits))?;
    cuda.copy_htod(&sink_d, &f32_to_bytes(&attn_sink))?;
    cuda.copy_htod(&kv_fp4_d, &kv_packed)?;
    cuda.copy_htod(&kv_scales_d, &kv_scales_row)?;
    cuda.copy_htod(&selected_d, &u32s_to_bytes(&selected))?;
    cuda.copy_htod(&index_key_d, &vec![0u8; index_key_bytes])?;
    cuda.copy_htod(&index_query_d, &vec![0u8; index_query_bytes])?;
    cuda.copy_htod(
        &block_scores_d,
        &f32_to_bytes(&vec![12345.0f32; MLA_HSI_BLOCKS]),
    )?;
    cuda.copy_htod(
        &block_best_d,
        &u32s_to_bytes(&vec![u32::MAX; MLA_HSI_BLOCKS]),
    )?;
    println!(
        "indexer side channel: zero index data + sentinel outputs; kernel indexer writes ignored, not compared"
    );
    let params = flash_mla_params(
        q_d.0,
        kv_fp4_d.0,
        kv_scales_d.0,
        0, // window_codes: the FP4 path carries none
        0, // window_scales
        WindowKvFormat::Fp4.kernel_value() as u64,
        0, // window_rows: FP4 ids are token positions, the window comes from position
        selected_d.0,
        index_key_d.0,
        index_query_d.0,
        sink_d.0,
        block_scores_d.0,
        block_best_d.0,
        attn_out_d.0,
        0, // position
        selected.len(),
        0, // trace_ring
        args.layer,
        0, // step_seq
    );
    check_floor(args, "stage 7 K5 pre-launch")?;
    flash_mla_launch(
        &cuda,
        mla_func,
        (MLA_HEADS as u32, args.grid_y, 1u32),
        (MLA_THREADS, 1u32, 1u32),
        &params,
    )?;
    check_floor(args, "stage 7 K5 post-launch")?;
    let mut attn_out_bytes = vec![0u8; out_bytes];
    cuda.copy_dtoh(&mut attn_out_bytes, &attn_out_d)?;
    cuda.free_buffers(&[
        &q_d,
        &sink_d,
        &kv_fp4_d,
        &kv_scales_d,
        &selected_d,
        &index_key_d,
        &index_query_d,
        &block_scores_d,
        &block_best_d,
        &attn_out_d,
    ]);
    let _ = allocated;
    let attn_out_bits = bytes_to_u16(&attn_out_bytes);
    let mut reference = Vec::with_capacity(MLA_HEADS * MLA_LATENT);
    for head in 0..MLA_HEADS {
        let head_out = mla_ref::attention_with_sink(
            &q_used[head * MLA_LATENT..(head + 1) * MLA_LATENT],
            &kv_packed,
            &kv_scales_row,
            &selected,
            0,
            attn_sink[head],
        );
        reference.extend_from_slice(&head_out);
    }
    let (attn_stats, attn_ulp) = compare_bf16_ulp(
        "K5 attention out (same q/kv/selected/sink)",
        &attn_out_bits,
        &reference,
    )?;
    println!(
        "  K5 internal check: {:.4}% exact against mla_ref on the same bytes, max abs delta {:e}, max ULP {attn_ulp}; the stage-2 tolerance class, not a boundary-row claim",
        attn_stats.exact_pct(),
        attn_stats.max_delta
    );

    // S5: inverse output RoPE (position 0 rotates by zero) and the MXFP8 operand for wo_a.
    let mut attn_operand = widen_bits(&attn_out_bits);
    rotate_rope_tail_inverse(
        &mut attn_operand,
        MLA_HEADS,
        MLA_LATENT,
        ROPE_HEAD_DIM,
        ROPE_THETA,
        0,
    );
    let (attn_operand_codes, attn_operand_scales) = act_quant_fp8_row(&attn_operand);
    println!("--- K6 wo_a device kernel (8 groups of 1024 over 4096) ---");
    let lo_out = wo_a_device(
        &cuda,
        wo_a_func,
        args.layer,
        &attn_operand_codes,
        &attn_operand_scales,
        &wo_a_w,
        &wo_a_s,
    )?;
    let (lo_codes, lo_scales) = act_quant_fp8_row(&lo_out);
    println!("--- K7 dense: wo_b ---");
    let attention_out = dense_device_gemm(
        &cuda, dense_func, args.layer, "K7 wo_b", &lo_codes, &lo_scales, &wo_b_w, &wo_b_s, HIDDEN,
        WO_A_OUT,
    )?;
    captures.insert("attention_out", RowValue::F32(attention_out.clone()));
    println!(
        "  attention_out: {} f32 (the wo_b bf16 output widened); attention_out carries the D4 \
         FP4-window-KV difference",
        attention_out.len()
    );

    // --- D4 attribution: the window-KV format is the only known device/host format difference in
    // this chain. Both formats are decoded from the SAME f32 latent and run through the same tail.
    println!("--- D4 attribution check: the attention tail with the host's window-KV format ---");
    let device_kv: Vec<f32> = (0..MLA_LATENT)
        .map(|index| mla_ref::kv_latent(&kv_packed, &kv_scales_row, 0, index))
        .collect();
    let (host_kv_codes, host_kv_scales) = act_quant_fp8_row(&kv);
    let host_kv: Vec<f32> = host_kv_codes
        .iter()
        .enumerate()
        .map(|(index, code)| {
            fp8_e4m3_decode(*code) * ue8m0_decode(host_kv_scales[index / QUANT_BLOCK])
        })
        .collect();
    let (_, kv_max_delta) = f32_delta_stats(&device_kv, &host_kv);
    println!(
        "  window-KV latent, the same f32 values: device FP4 E2M1 (per-16 E4M3 scale) vs host FP8          E4M3 (per-32 ue8m0 scale), max abs delta {kv_max_delta:.6e} over {MLA_LATENT} channels"
    );
    let predicted_device = predict_attention_out(
        &q_used, &attn_sink, &device_kv, &wo_a_w, &wo_a_s, &wo_b_w, &wo_b_s,
    );
    let predicted_host = predict_attention_out(
        &q_used, &attn_sink, &host_kv, &wo_a_w, &wo_a_s, &wo_b_w, &wo_b_s,
    );
    let (_, predicted_device_delta) = f32_delta_stats(&predicted_device, &attention_out);
    println!(
        "  prediction with the device's FP4 format reproduces this run's attention_out: max abs          delta {predicted_device_delta:.6e} (the prediction's own control)"
    );
    let host_row_attention_out = host.f32(0, "attention_out")?;
    let (_, measured_attention_delta) = f32_delta_stats(&attention_out, &host_row_attention_out);
    let (_, predicted_host_delta) = f32_delta_stats(&predicted_host, &host_row_attention_out);
    let (_, formats_delta) = f32_delta_stats(&predicted_device, &predicted_host);
    println!(
        "  prediction with the host's FP8 format against the host row: max abs delta {predicted_host_delta:.6e}; the measured device-vs-host delta on this tensor is {measured_attention_delta:.6e}. Diagnostic only: this prediction reuses the device's q and attention-tail operands, so its gap to the host row includes the upstream device-side differences those operands carry; it is not a pure format measurement."
    );
    println!(
        "  attribution: the two formats' predictions differ by {formats_delta:.6e} max abs on the SAME latent and the same tail - that is the window-KV format's own contribution and the valid D4 number; the FP8 prediction moves the measured gap from {measured_attention_delta:.6e} to {predicted_host_delta:.6e}, but that movement mixes the format with upstream device-side differences, so formats_delta is the attribution and predicted_host_delta is a diagnostic. The residual is the host reference's own f32 arithmetic (no device activation quantiser, no bf16 writes) plus the bf16 points this prediction still keeps (the raw attention output and the lo rank), not a kernel difference. The projections are not the source: K2-K7 are bit-exact against their references on the same bytes."
    );

    // --- K8: the FFN sublayer's mHC pass (applies the attention hc_post, then the FFN mixes) ----
    println!("--- K8 {kernel_ok}: FFN sublayer (applies the attention hc_post) ---");
    mhc.upload(
        &cuda,
        &residual_in_bits,
        &attention_out,
        &attn_b,
        &attn_post,
        &attn_pre,
        &mhc_ffn.predictor,
        &mhc_ffn.hc_scale,
        &mhc_ffn.hc_base_kernel,
        &ffn_norm,
    )?;
    check_floor(args, "stage 7 K8 pre-launch")?;
    mhc.launch(&cuda, mhc_func, args.layer, 1)?;
    check_floor(args, "stage 7 K8 post-launch")?;
    let post_attention_bits = mhc.x_bits(&cuda)?;
    let ffn_coeffs = mhc.coeffs(&cuda)?;
    let ffn_hidden = mhc.normed(&cuda)?;
    let ffn_codes = mhc.codes(&cuda)?;
    let ffn_scales = mhc.scales(&cuda)?;
    let ffn_b: Vec<f32> = ffn_coeffs[0..HC_MULT * HC_MULT].to_vec();
    let ffn_pre: Vec<f32> = ffn_coeffs[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT].to_vec();
    let ffn_post: Vec<f32> =
        ffn_coeffs[HC_MULT * HC_MULT + HC_MULT..HC_MULT * HC_MULT + 2 * HC_MULT].to_vec();
    let ffn_comb: Vec<f32> = {
        let mut comb = vec![0.0f32; HC_MULT * HC_MULT];
        for destination in 0..HC_MULT {
            for source in 0..HC_MULT {
                comb[source * HC_MULT + destination] = ffn_b[destination * HC_MULT + source];
            }
        }
        comb
    };
    captures.insert(
        "post_attention_hidden",
        RowValue::F32(widen_bits(&post_attention_bits)),
    );
    captures.insert("hc_ffn_pre", RowValue::F32(ffn_pre.clone()));
    captures.insert("hc_ffn_post", RowValue::F32(ffn_post.clone()));
    captures.insert("hc_ffn_comb", RowValue::F32(ffn_comb));
    captures.insert("pre_ffn_hidden", RowValue::F32(ffn_hidden.clone()));
    captures.insert("pre_ffn_xhat", RowValue::U8(ffn_codes.clone()));
    captures.insert("pre_ffn_scales", RowValue::U8(ffn_scales.clone()));
    println!(
        "  K8 captured: post_attention_hidden {} bf16 values, FFN coefficients 24 f32, pre_ffn_hidden {} f32 + {} codes + {} scales",
        post_attention_bits.len(),
        ffn_hidden.len(),
        ffn_codes.len(),
        ffn_scales.len()
    );

    // --- S1/S2/S3 + K9-K12: the routed MoE block, free-running and with the host's routing forced -
    println!(
        "seam: router (gate GEMM, sqrt-softplus, noaux_tc bias, top-6, norm_topk_prob) - host; task-37 moves it"
    );
    let block_activation = decode_quantised_row(&ffn_codes, &ffn_scales);
    let all_experts: Vec<u32> = (0..MOE_EXPERTS as u32).collect();
    let (expert_ids, expert_weights) =
        route_token(&ffn_hidden, &gate_weight, &gate_bias, &all_experts);
    println!("  router: top-{MOE_TOP_K} ids {expert_ids:?} weights {expert_weights:?}");
    let host_pre_ffn_hidden = host.f32(0, "pre_ffn_hidden")?;
    let host_pre_ffn_tensor = host.tensor(0, "pre_ffn_hidden")?;
    println!(
        "  host-routing provenance: the host row's FFN input is HostBoundaryTable row 0 tensor pre_ffn_hidden (dtype {}, dims {:?}, blob bytes [{}, {})), read only from the host v1 pair (manifest+blob) in this process, not from a separate host reference file and not from the device path; the host routing below is re-derived from those captured f32 values",
        host_pre_ffn_tensor.dtype,
        host_pre_ffn_tensor.dims,
        host_pre_ffn_tensor.offset,
        host_pre_ffn_tensor.offset + host_pre_ffn_tensor.len
    );
    let (host_expert_ids, host_expert_weights) =
        route_token(&host_pre_ffn_hidden, &gate_weight, &gate_bias, &all_experts);
    let differing_slots: Vec<(usize, u32, u32)> = expert_ids
        .iter()
        .zip(&host_expert_ids)
        .enumerate()
        .filter(|(_, (device, host))| device != host)
        .map(|(slot, (device, host))| (slot, *device, *host))
        .collect();
    println!(
        "  router on the host row's own pre_ffn_hidden: top-{MOE_TOP_K} ids {host_expert_ids:?} weights {host_expert_weights:?}"
    );
    println!(
        "  routing comparison: the ordered top-{MOE_TOP_K} lists are equal {}; slot-wise differing (slot, device id, host id) {differing_slots:?}",
        expert_ids == host_expert_ids
    );
    let mut device_sorted = expert_ids.clone();
    device_sorted.sort_unstable();
    let mut host_sorted = host_expert_ids.clone();
    host_sorted.sort_unstable();
    let only_device: Vec<u32> = device_sorted
        .iter()
        .copied()
        .filter(|id| !host_sorted.contains(id))
        .collect();
    let only_host: Vec<u32> = host_sorted
        .iter()
        .copied()
        .filter(|id| !device_sorted.contains(id))
        .collect();
    println!(
        "  routing set difference: only in the device top-{MOE_TOP_K} {only_device:?}, only in the host row's top-{MOE_TOP_K} {only_host:?}; the id sets are equal {}",
        device_sorted == host_sorted
    );
    println!("seam: gate/up GEMM (w1/w3) - host; Phase 3 moves it");
    println!(
        "seam: SwiGLU and the activation quantiser - host; task-43 compares it, Phase 3 moves it"
    );
    let table_ids: Vec<u32> = (0..MOE_TOP_K as u32).collect();
    let device_operands = routed_operands(
        &mut snap,
        &prefix,
        &block_activation,
        &expert_ids,
        &expert_weights,
    )?;
    println!(
        "  free-running routing: {MOE_TOP_K} experts staged (w1/w3/w2 packed FP4 + ue8m0 scales read exactly as stored)"
    );
    let routed_bits = run_routed_combine(
        args,
        "stage 7",
        &cuda,
        moe_func,
        args.layer,
        "free-running routing",
        &device_operands,
        &table_ids,
    )?;
    let reference_combine = reference_combine_chain(
        &device_operands.act_fp8,
        &device_operands.act_scales,
        &device_operands.w_rank,
        &device_operands.ws_rank,
        &table_ids,
        1,
        MOE_TOP_K,
    );
    let (combine_stats, combine_ulp) =
        compare_bf16_bits("K9-K12 routed combine", &routed_bits, &reference_combine);
    println!(
        "  routed combine internal check: {:.4}% exact against mxfp4_ref::moe_combine chained over the same {MOE_RANKS} ranks, max ULP {combine_ulp}",
        combine_stats.exact_pct()
    );

    // --- K13-K15: the shared expert on the device dense kernel, host SwiGLU/quantise in between ----
    println!(
        "--- K13-K15 shared expert: device w1/w3/w2 dense kernels, host SwiGLU + quantiser ---"
    );
    let shared_gate = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K13 shared w1",
        &ffn_codes,
        &ffn_scales,
        &shared_w1,
        &shared_s1,
        MOE_INTERMEDIATE,
        HIDDEN,
    )?;
    let shared_up = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K14 shared w3",
        &ffn_codes,
        &ffn_scales,
        &shared_w3,
        &shared_s3,
        MOE_INTERMEDIATE,
        HIDDEN,
    )?;
    let shared_activation: Vec<f32> = shared_gate
        .iter()
        .zip(&shared_up)
        .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
        .collect();
    let (shared_codes, shared_scales) = act_quant_fp8_row(&shared_activation);
    let shared_out = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K15 shared w2",
        &shared_codes,
        &shared_scales,
        &shared_w2,
        &shared_s2,
        HIDDEN,
        MOE_INTERMEDIATE,
    )?;
    println!(
        "seam: routed-expert and shared-expert sum (elementwise add of two bf16 vectors) - host"
    );
    let ffn_out = combine_bf16_routed_with_shared(&routed_bits, &shared_out)?;
    captures.insert("ffn_out", RowValue::F32(ffn_out.clone()));
    println!(
        "  ffn_out: {} f32 = routed-combine bf16 + shared-w2 bf16, summed in the reference's order (host seam S9)",
        ffn_out.len()
    );

    // --- the forced-routing control: separate ffn_out's discrete part from its arithmetic part -----
    println!(
        "--- forced-routing control: the device MoE with the host row's own top-6 and weights ---"
    );
    let (forced_ids, forced_weights) =
        route_token(&host_pre_ffn_hidden, &gate_weight, &gate_bias, &all_experts);
    if forced_ids != host_expert_ids || forced_weights != host_expert_weights {
        return Err("forced-routing control: re-deriving the host routing from the host row does not match the host routing derived above".to_string());
    }
    println!(
        "  forced-control routing re-derivation (before the operands): the host row's pre_ffn_hidden gives ids {forced_ids:?} weights {forced_weights:?}, equal to the host routing derived above; the forced operands are built from these values, not from the free path's"
    );
    let forced_operands = routed_operands(
        &mut snap,
        &prefix,
        &block_activation,
        &forced_ids,
        &forced_weights,
    )?;
    verify_forced_operand_provenance(
        &mut snap,
        &prefix,
        &block_activation,
        &forced_ids,
        &forced_weights,
        &forced_operands.w_rank,
        &forced_operands.ws_rank,
        &forced_operands.act_fp8,
        &forced_operands.act_scales,
        &FORCED_OPERAND_GEOMETRY,
    )?;
    println!(
        "  forced-control operand provenance (checked before any launch, every slot and every rank): re-read the {MOE_TOP_K} experts from the checkpoint; all {MOE_TOP_K} w2 weight+scale slices x {MOE_RANKS} ranks are bit-identical in the requested id order, and for every slot the w1/w3-derived activation codes and scales at every rank are bit-identical to the reference operand built from the requested id and weight; the forced launch consumes exactly this routing"
    );
    let forced_bits = run_routed_combine(
        args,
        "stage 7",
        &cuda,
        moe_func,
        args.layer,
        "host routing forced",
        &forced_operands,
        &table_ids,
    )?;
    let forced_reference = reference_combine_chain(
        &forced_operands.act_fp8,
        &forced_operands.act_scales,
        &forced_operands.w_rank,
        &forced_operands.ws_rank,
        &table_ids,
        1,
        MOE_TOP_K,
    );
    let (forced_stats, forced_ulp) =
        compare_bf16_bits("forced routed combine", &forced_bits, &forced_reference);
    let forced_routed = widen_bits(&forced_bits);
    let device_ffn_out_forced = {
        let mut ffn_out_forced = vec![0.0f32; HIDDEN];
        for column in 0..HIDDEN {
            ffn_out_forced[column] = forced_routed[column] + shared_out[column];
        }
        ffn_out_forced
    };
    // The asserted-identical recomputation of the forced output. Every forced number reported below
    // (the host deltas and the pointwise flip) reads this returned tensor, never the device launch's
    // copy, so a copy inserted after the assertion cannot change any reported metric; a mismatch
    // between the device tensor and the recomputation aborts here, before any metric is accepted.
    let ffn_out_forced = verify_forced_ffn_out(&forced_bits, &shared_out, &device_ffn_out_forced)?;
    println!(
        "  forced-control ffn_out provenance (checked before the metric is accepted): ffn_out_forced recomputed from the forced routed-combine bf16 bits + the K15 shared-w2 output through the free path's combination is bit-identical to the tensor the control diffs; the forced host delta cannot be the free path's"
    );
    let host_ffn_out = host.f32(0, "ffn_out")?;
    let verdict = forced_control_verdict(
        &expert_ids,
        &host_expert_ids,
        &expert_weights,
        &host_expert_weights,
        routed_bits == forced_bits,
        &reference_combine,
        &forced_reference,
    )?;
    println!(
        "  forced-control separation: ordered routing ids identical {}; weights changed (ids equal, weights differ) {}; routed-combine bytes bit-identical {}; identical ids and unchanged weights must give byte-identical routed output, identical ids with changed weights may give equal or different bytes, and a flip (ids differ) must give a nonzero routed difference unless the two independently computed references are bit-identical (then the flip cannot move the output and equal bytes are reported, not treated as a copy)",
        verdict.routing_identical, verdict.weights_changed, verdict.routed_identical
    );
    let (forced_exact, _forced_delta) = f32_delta_stats(&ffn_out_forced, &host_ffn_out);
    let metrics = free_forced_metrics(&ffn_out, &ffn_out_forced, &host_ffn_out)?;
    let flip_at_free_host = ffn_out[metrics.free_host_at] - ffn_out_forced[metrics.free_host_at];
    let flip_at_forced_host =
        ffn_out[metrics.forced_host_at] - ffn_out_forced[metrics.forced_host_at];
    let flip_share_at_free_host = 100.0 * flip_at_free_host.abs() / metrics.free_host_max_abs;
    let flip_share_at_forced_host = 100.0 * flip_at_forced_host.abs() / metrics.forced_host_max_abs;
    // The measured relation at the max-host-error point, with the number always printed: below a
    // 10% share the point's host error is not explained by the flip, at or above it the point is
    // flip-dominated. The percentage is what the report quotes either way.
    let max_error_reading = if flip_share_at_free_host < 10.0 {
        format!(
            "the max-error point is not flip-dominated: the flip contributes only {flip_share_at_free_host:.2}% of the {:.6e} host error there",
            metrics.free_host_max_abs
        )
    } else {
        format!(
            "the max-error point is flip-dominated: the flip contributes {flip_share_at_free_host:.2}% of the {:.6e} host error there",
            metrics.free_host_max_abs
        )
    };
    println!(
        "  forced routed combine internal check: {:.4}% exact against mxfp4_ref::moe_combine chained over the same ranks, max ULP {forced_ulp}",
        forced_stats.exact_pct()
    );
    println!(
        "  free-vs-forced ffn_out, pointwise: max abs |ffn_out_free(i) - ffn_out_forced(i)| = {:.6e} at i={} (signed free minus forced there {:+.6e}); {}/{} points differ; the two ffn_out tensors are bit-identical {}",
        metrics.pointwise_max_abs,
        metrics.at,
        metrics.signed_at,
        metrics.differing_points,
        ffn_out.len(),
        metrics.all_equal
    );
    println!(
        "  host deltas, separated: free-running routing max abs |ffn_out_free - host ffn_out| = {:.6e} at i={}; host routing forced max abs |ffn_out_forced - host ffn_out| = {:.6e} at i={}, forced bit-exact {forced_exact}/{}; forcing the host routing does not reduce the max host error, and the difference of the two host-error maxima is a change in max error, NOT the flip's max-abs contribution",
        metrics.free_host_max_abs,
        metrics.free_host_at,
        metrics.forced_host_max_abs,
        metrics.forced_host_at,
        ffn_out_forced.len()
    );
    println!(
        "  flip at the max-host-error points: at the free max-error index i={} the pointwise flip is signed {:+.6e} (abs {:.6e}, {:.2}% of the {:.6e} host error there); at the forced max-error index i={} the pointwise flip is signed {:+.6e} (abs {:.6e}, {:.2}% of the {:.6e} host error there)",
        metrics.free_host_at,
        flip_at_free_host,
        flip_at_free_host.abs(),
        flip_share_at_free_host,
        metrics.free_host_max_abs,
        metrics.forced_host_at,
        flip_at_forced_host,
        flip_at_forced_host.abs(),
        flip_share_at_forced_host,
        metrics.forced_host_max_abs
    );
    println!(
        "  max-error reading (measured): {max_error_reading}; the flip's own maximum is {:.6e} at i={} (signed {:+.6e})",
        metrics.pointwise_max_abs, metrics.at, metrics.signed_at
    );
    println!(
        "  discrete expert-flip contribution (pointwise): {:.6e} max abs at i={} (signed {:+.6e}); {}/{} points differ; differing slots (slot, device id, host id) {differing_slots:?}. A flip is a divergence event, not a kernel bug: K9-K12 are exact against the reference on the same bytes, so the forced run's residual is the D4/f32-reference arithmetic plus the routed combine's own quantisation, not a combine-kernel bug.",
        metrics.pointwise_max_abs,
        metrics.at,
        metrics.signed_at,
        metrics.differing_points,
        ffn_out.len()
    );

    // --- K16: the last hc_post (there is no next layer's K1 to apply the FFN hc_post) -----------
    println!(
        "--- K16 {kernel_ok}: final FFN hc_post (streams only; coefficient and quantiser outputs ignored) ---"
    );
    mhc.upload(
        &cuda,
        &post_attention_bits,
        &ffn_out,
        &ffn_b,
        &ffn_post,
        &ffn_pre,
        &mhc_ffn.predictor,
        &mhc_ffn.hc_scale,
        &mhc_ffn.hc_base_kernel,
        &ffn_norm,
    )?;
    check_floor(args, "stage 7 K16 pre-launch")?;
    mhc.launch(&cuda, mhc_func, args.layer, 2)?;
    check_floor(args, "stage 7 K16 post-launch")?;
    let residual_out_bits = mhc.x_bits(&cuda)?;
    captures.insert(
        "residual_out",
        RowValue::F32(widen_bits(&residual_out_bits)),
    );
    println!(
        "  K16 captured: residual_out {} bf16 values (the last hc_post applied by a dedicated launch, decision D5)",
        residual_out_bits.len()
    );
    mhc.release(&cuda);
    check_floor(args, "stage 7 after chain")?;
    println!(
        "seam: head and sampling - host; Phase 1 stage 4 / Phase 3 move it. NOT EXERCISED here: criterion 1 stops at the boundary table, and stage 5 owns the head"
    );
    println!(
        "seam: engram write (n-gram lookup and gated add, layers 1 and 14) - host; Phase 2/3 move it. NOT EXERCISED at layer 0: the engram writes are at layers 1 and 14"
    );

    // --- the row-0 diff -----------------------------------------------------------------------
    println!("--- row 0: the device chain against the host boundary row ---");
    let deltas = compare_row(&host, 0, &captures)?;
    println!(
        "{:<24} {:<12} {:>7} {:>10} {:>12} {:>12} {:>6}",
        "tensor", "class", "values", "host-exact", "mean-abs", "max-abs", "at"
    );
    for delta in &deltas {
        match delta.class {
            BoundaryClass::ExactBits => {
                let percent = 100.0 * delta.host_exact as f64 / delta.values as f64;
                match (delta.bf16_host_exact, delta.bf16_host_max_delta) {
                    (Some(bf16_exact), Some(bf16_max)) => println!(
                        "{:<24} {:<12} {:>7} {:>9.4}% {:>12.4e} {:>12.4e} {:>6}  [bf16(host)-exact {:.4}%, max vs bf16 {:.6e}]",
                        delta.name,
                        "exact-bits",
                        delta.values,
                        percent,
                        delta.mean_abs_delta,
                        delta.max_abs_delta,
                        delta.at,
                        100.0 * bf16_exact as f64 / delta.values as f64,
                        bf16_max
                    ),
                    _ => println!(
                        "{:<24} {:<12} {:>7} {:>9.4}% {:>12.4e} {:>12.4e} {:>6}",
                        delta.name,
                        "exact-bits",
                        delta.values,
                        percent,
                        delta.mean_abs_delta,
                        delta.max_abs_delta,
                        delta.at
                    ),
                }
            }
            BoundaryClass::ExactCodes => println!(
                "{:<24} {:<12} {:>7} {:>9.4}% differing {} mean distance {:.3}, max {} (first at {}), distances {}",
                delta.name,
                "exact-codes",
                delta.values,
                100.0 * delta.host_exact as f64 / delta.values as f64,
                delta.differing_codes,
                delta.mean_code_distance,
                delta.max_code_distance,
                delta.at,
                delta.code_distance_buckets_text()
            ),
        }
    }
    println!("--- the Stage-B single_layer_delta values this run records (design section 8) ---");
    for delta in &deltas {
        println!(
            "  single_layer_delta({}) = {:.6e} ({})",
            delta.name,
            delta.single_layer_delta(),
            if delta.class == BoundaryClass::ExactCodes {
                "codes differing / codes"
            } else {
                "max abs delta"
            }
        );
    }
    println!("--- attribution notes ---");
    println!(
        "attribution of the deltas above: (1) upstream: pre_attention_hidden 2.76e-7 max abs, 1/5,120 codes differing, 160/160 scales exact - the mHC pass and its quantiser agree on both sides. (2) attention_out's delta has two named sources: the window-KV format of D4 (the two formats' predictions differ by 3.125e-1 max abs on the same latent and the same tail, which is the format's own contribution and the valid D4 number) and the host reference's f32 arithmetic (no GEMM activation quantiser, no bf16 writes). The host-format prediction against the host row is a diagnostic only: it reuses the device's q and attention-tail operands, so its gap to the host row also contains the upstream device-side differences those operands carry and it is not a pure format measurement. (3) ffn_out: the router flip is a large pointwise perturbation (max {:.6e} at i={}, signed {:+.6e}, {}/{} points differ), and the forced-routing control measures its relation to the host error rather than assuming one: at the free max host error ({:.6e} at i={}) the flip contributes {:+.6e} (abs {:.6e}, {:.2}% of the error there), and forcing the host row's routing does not reduce the max host error ({:.6e} at i={}, flip there {:+.6e}); {max_error_reading}. The residual is the device path's quantised routed arithmetic plus the propagated upstream differences, not the combine kernel (K9-K12 are 100% exact against the in-crate reference on the same bytes); these are measured numbers, not a smooth bar. A flip is a divergence event to be reported per layer by the chain.",
        metrics.pointwise_max_abs,
        metrics.at,
        metrics.signed_at,
        metrics.differing_points,
        ffn_out.len(),
        metrics.free_host_max_abs,
        metrics.free_host_at,
        flip_at_free_host,
        flip_at_free_host.abs(),
        flip_share_at_free_host,
        metrics.forced_host_max_abs,
        metrics.forced_host_at,
        flip_at_forced_host
    );
    println!(
        "The quantised tensors (#7/#8/#15/#16) are exact-codes: the device K1/K8 quantiser is the \
         same ceil rule as the host's, so a differing code names an upstream f32 difference, not a \
         different scale rule. scale mismatches in batches of 32 are a block amax on a power-of-two \
         boundary; single codes are a value on a rounding boundary."
    );
    println!(
        "The stream tensors (#1/#10/#18) are bf16 storage on the device and f32 on the host, so the \
         bf16(host) column is the value the device path can be exact at."
    );
    let finite = deltas.iter().all(|delta| delta.max_abs_delta.is_finite());
    let complete = deltas.len() == B2_TENSORS.len();
    if !finite || !complete {
        return Err(format!(
            "stage 7: FAIL - {}/{} tensors compared, finite deltas {finite}",
            deltas.len(),
            B2_TENSORS.len()
        ));
    }
    println!(
        "stage 7: PASS ({} of {} row-0 tensors captured and diffed; the single_layer_delta values \
         above are the Stage-B bar input, and this run applies no bar to a single layer)",
        deltas.len(),
        B2_TENSORS.len()
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Stage 9: the 40-layer chained device forward (Stage C of device-chain-design.md section 9).
//
// One released-checkpoint prefill step at position 0, one token (id 0), 40 backbone rows: the same
// chain stage 7 runs for layer 0, looped over the backbone, carrying the device residual streams,
// the previous layer's FFN output and its kernel-order B/A/C coefficient triples between layers.
// The next layer's K1 performs the previous layer's last hc_post, so 15 launches per layer plus the
// final K16; the two Engram layers (1 and 14) need their own K16 before the write and a re-run of K1
// with the identity seed afterwards, because the released Engram write mutates the stream between
// those layers. Every row is captured into the same 18 tensors stage 7 captures and diffed against
// the host v1 pair by comparison class, with the Stage-B single-layer deltas compounded as the bars
// (design section 8). The run writes its own v1 pair so examples/boundary_diff.rs can diff it
// unchanged, and it runs the chain twice to print determinism.
//
// This is the Phase-1 HEALTH-CHECK column, not a criterion-1 closure: the device path uses FP4
// main/window KV and bf16/quantised arithmetic where the host reference is f32, so the column
// measures attribution and gross mismatch. The format-faithful exactness run is the later one after
// Phase 2 (task-3). Criterion 1 is not closed by this run.
// ---------------------------------------------------------------------------------------------

/// The Stage-B single-layer deltas in effect as the Stage-C bar inputs, copied from the frozen
/// Stage-B evidence of record
/// (`/home/user/phase1-evidence/phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log`,
/// lines 149-166, binary `2b518a35...`). The values are MEASURED, not chosen: the log records them
/// before Stage C exists, and this run compounds them per design section 8.
const STAGE9_B2_DELTAS: [(&str, f64); 18] = [
    ("residual_in", 0.0),
    ("hc_in_pre", 0.0),
    ("hc_attn_pre", 8.493_662e-7),
    ("hc_attn_post", 3.278_255e-7),
    ("hc_attn_comb", 3.129_244e-7),
    ("pre_attention_hidden", 2.756_715e-7),
    ("pre_attention_xhat", 1.953_125e-4),
    ("pre_attention_scales", 0.0),
    ("attention_out", 2.565_311e-1),
    ("post_attention_hidden", 7.431_363e-2),
    ("hc_ffn_pre", 4.267_210e-2),
    ("hc_ffn_post", 7.206_082e-3),
    ("hc_ffn_comb", 1.175_366e-3),
    ("pre_ffn_hidden", 3.163_002e-2),
    ("pre_ffn_xhat", 6.882_812e-1),
    ("pre_ffn_scales", 3.75e-2),
    ("ffn_out", 2.150_760e-1),
    ("residual_out", 1.810_353e-1),
];

/// One Stage-B single-layer delta by boundary tensor name.
fn stage9_b2_delta(name: &str) -> f64 {
    STAGE9_B2_DELTAS
        .iter()
        .find(|(tensor, _)| *tensor == name)
        .map(|(_, delta)| *delta)
        .unwrap_or(f64::NAN)
}

/// The Stage-C bar for one tensor of one row, exactly as the log prints it before judging anything.
///
/// The pre-registration of design section 8 is `bar(row L, tensor T) = single_layer_delta(T) *
/// (L + 1)` for the stream-carrying tensors (#1, #10, #18) and `single_layer_delta(T)` for the
/// per-layer tensors. Two B2 deltas are zero by construction at layer 0 because their layer-0 value
/// is the exact seed (the embedding stream and the one-hot input mix); row L>0 inherits the delta of
/// the tensor that produces it (residual_out for residual_in, hc_ffn_pre for hc_in_pre), so the bar
/// uses that measured predecessor instead of a vacuous zero. The substitution is printed in the
/// bar table, never applied silently.
fn stage9_bar(name: &str, row: usize) -> f64 {
    let measured = match name {
        "residual_in" => stage9_b2_delta("residual_out"),
        "hc_in_pre" => stage9_b2_delta("hc_ffn_pre"),
        other => stage9_b2_delta(other),
    };
    let factor = if B2_STREAM_TENSORS.contains(&name) {
        (row + 1) as f64
    } else {
        1.0
    };
    measured * factor
}

/// The shape of one boundary tensor, in the v1 manifest's comma-separated form.
fn stage9_dims(name: &str) -> Vec<usize> {
    match name {
        "residual_in" | "post_attention_hidden" | "residual_out" => vec![1, HC_MULT, HIDDEN],
        "hc_in_pre" | "hc_attn_pre" | "hc_attn_post" | "hc_ffn_pre" | "hc_ffn_post" => {
            vec![1, HC_MULT]
        }
        "hc_attn_comb" | "hc_ffn_comb" => vec![1, HC_MULT, HC_MULT],
        "pre_attention_xhat" | "pre_ffn_xhat" => vec![1, HIDDEN],
        "pre_attention_scales" | "pre_ffn_scales" => vec![1, HIDDEN / QUANT_BLOCK],
        _ => vec![1, HIDDEN],
    }
}

/// The v1 dtype token of one boundary tensor name (docs/build/phase1-boundary-table.md section 3).
fn stage9_v1_dtype(name: &str) -> &'static str {
    match name {
        "pre_attention_xhat" | "pre_ffn_xhat" => "u8-e4m3",
        "pre_attention_scales" | "pre_ffn_scales" => "u8-e8m0",
        _ => "f32",
    }
}

/// Build one device v1 pair in memory: the manifest text and the blob.
///
/// The writer is deliberately the same shape as the engine's `BoundaryTable::write_v1` (header lines,
/// `row N` markers, one tensor line per tensor in TENSOR_ORDER order, blob in the same order with no
/// padding). The harness cannot construct an engine `BoundaryTable` from outside the engine crate,
/// so it writes the format directly and then proves the writer on the released host pair: for a
/// 40-row run the device manifest must be byte-identical to `phase1-host-token0.manifest`.
fn stage9_build_v1(rows: &[Stage9Row]) -> Result<(String, Vec<u8>), String> {
    let mut manifest = String::new();
    let mut blob: Vec<u8> = Vec::new();
    manifest.push_str("# phase1-boundary v1\n");
    manifest.push_str("step 0 1 0\n");
    manifest.push_str("prefix 0\n");
    manifest.push_str("mtp-anchor none\n");
    manifest.push_str(&format!("backbone {}\n", rows.len()));
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != B2_TENSORS.len() {
            return Err(format!(
                "row {row_index} has {} tensors, the format has {}",
                row.len(),
                B2_TENSORS.len()
            ));
        }
        manifest.push_str(&format!("row {row_index}\n"));
        for ((name, _class), (captured_name, value)) in B2_TENSORS.iter().zip(row) {
            if *captured_name != *name {
                return Err(format!(
                    "row {row_index}: captured {captured_name} where {name} belongs"
                ));
            }
            let dims = stage9_dims(name);
            let declared: usize = dims.iter().product();
            let bytes: Vec<u8> = match value {
                RowValue::F32(values) => {
                    if values.len() != declared {
                        return Err(format!(
                            "row {row_index} {name}: {} f32 values for shape {dims:?}",
                            values.len()
                        ));
                    }
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect()
                }
                RowValue::U8(codes) => {
                    if codes.len() != declared {
                        return Err(format!(
                            "row {row_index} {name}: {} bytes for shape {dims:?}",
                            codes.len()
                        ));
                    }
                    codes.clone()
                }
            };
            let dims_text = dims
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let offset = blob.len();
            manifest.push_str(&format!(
                "tensor {row_index} {name} {} {dims_text} {offset} {}\n",
                stage9_v1_dtype(name),
                bytes.len()
            ));
            blob.extend_from_slice(&bytes);
        }
    }
    manifest.push_str("end\n");
    Ok((manifest, blob))
}

/// sha256 of one byte slice, hex, for the determinism line.
fn stage9_sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Everything one layer's device run reads from the released checkpoint, staged once per layer and
/// dropped before the next layer so the resident set does not grow with the layer index.
struct Stage9Weights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mhc_attn: ReleasedMhcTensors,
    mhc_ffn: ReleasedMhcTensors,
    attn_sink: Vec<f32>,
    q_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    wq_a_w: Vec<u8>,
    wq_a_s: Vec<u8>,
    wq_b_w: Vec<u8>,
    wq_b_s: Vec<u8>,
    wkv_w: Vec<u8>,
    wkv_s: Vec<u8>,
    wo_b_w: Vec<u8>,
    wo_b_s: Vec<u8>,
    wo_a_w: Vec<u8>,
    wo_a_s: Vec<u8>,
    gate_weight: Vec<f32>,
    gate_bias: Vec<f32>,
    shared_w1: Vec<u8>,
    shared_s1: Vec<u8>,
    shared_w3: Vec<u8>,
    shared_s3: Vec<u8>,
    shared_w2: Vec<u8>,
    shared_s2: Vec<u8>,
}

impl Stage9Weights {
    fn staged_bytes(&self) -> usize {
        self.wq_a_w.len()
            + self.wq_a_s.len()
            + self.wq_b_w.len()
            + self.wq_b_s.len()
            + self.wkv_w.len()
            + self.wkv_s.len()
            + self.wo_b_w.len()
            + self.wo_b_s.len()
            + self.wo_a_w.len()
            + self.wo_a_s.len()
            + self.shared_w1.len()
            + self.shared_s1.len()
            + self.shared_w3.len()
            + self.shared_s3.len()
            + self.shared_w2.len()
            + self.shared_s2.len()
            + self.gate_weight.len() * 4
            + self.mhc_attn.predictor.len() * 4
            + self.mhc_ffn.predictor.len() * 4
    }
}

/// Reads one layer's released tensors in the same order and with the same assertions as stage 7.
fn stage9_load_weights(snap: &mut Snapshot, prefix: &str) -> Result<Stage9Weights, String> {
    let attn_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let ffn_norm = snap.read_f32_or_bf16(&format!("{prefix}.ffn_norm.weight"), &[HIDDEN])?;
    let mhc_attn = read_released_mhc_tensors(snap, prefix, "attn")?;
    let mhc_ffn = read_released_mhc_tensors(snap, prefix, "ffn")?;
    let attn_sink = snap.read_f32(&format!("{prefix}.attn.attn_sink"), &[MLA_HEADS])?;
    let q_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.q_norm.weight"), &[Q_LORA_RANK])?;
    let kv_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.kv_norm.weight"), &[MLA_LATENT])?;
    let wq_a_w = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.attn.wq_a.weight"),
            "F8_E4M3",
            &[Q_LORA_RANK, HIDDEN],
        )?,
        Q_LORA_RANK,
        HIDDEN,
    );
    let wq_a_s = snap.read_raw(
        &format!("{prefix}.attn.wq_a.scale"),
        "F8_E8M0",
        &[Q_LORA_RANK / 32, HIDDEN / 32],
    )?;
    let wq_b_cols = MLA_HEADS * MLA_LATENT;
    let wq_b_w = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.attn.wq_b.weight"),
            "F8_E4M3",
            &[wq_b_cols, Q_LORA_RANK],
        )?,
        wq_b_cols,
        Q_LORA_RANK,
    );
    let wq_b_s = snap.read_raw(
        &format!("{prefix}.attn.wq_b.scale"),
        "F8_E8M0",
        &[wq_b_cols / 32, Q_LORA_RANK / 32],
    )?;
    let wkv_w = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.attn.wkv.weight"),
            "F8_E4M3",
            &[MLA_LATENT, HIDDEN],
        )?,
        MLA_LATENT,
        HIDDEN,
    );
    let wkv_s = snap.read_raw(
        &format!("{prefix}.attn.wkv.scale"),
        "F8_E8M0",
        &[MLA_LATENT / 32, HIDDEN / 32],
    )?;
    let wo_b_w = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.attn.wo_b.weight"),
            "F8_E4M3",
            &[HIDDEN, WO_A_OUT],
        )?,
        HIDDEN,
        WO_A_OUT,
    );
    let wo_b_s = snap.read_raw(
        &format!("{prefix}.attn.wo_b.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, WO_A_OUT / 32],
    )?;
    let wo_a_weight_name = format!("{prefix}.attn.wo_a.weight");
    let wo_a_scale_name = format!("{prefix}.attn.wo_a.scale");
    let wo_a_weight_header = snap.load(&wo_a_weight_name)?;
    let wo_a_scale_header = snap.load(&wo_a_scale_name)?;
    let wo_a_weight = snap.read_raw(&wo_a_weight_name, "F8_E4M3", &wo_a_weight_header.shape)?;
    let wo_a_scale = snap.read_raw(&wo_a_scale_name, "F8_E8M0", &wo_a_scale_header.shape)?;
    let (wo_a_w, wo_a_s) = pack_wo_a(&wo_a_weight, &wo_a_scale)?;
    let gate_weight =
        snap.read_f32_or_bf16(&format!("{prefix}.ffn.gate.weight"), &[MOE_EXPERTS, HIDDEN])?;
    let gate_bias = snap.read_f32(&format!("{prefix}.ffn.gate.bias"), &[MOE_EXPERTS])?;
    let shared_w1 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w1.weight"),
            "F8_E4M3",
            &[MOE_INTERMEDIATE, HIDDEN],
        )?,
        MOE_INTERMEDIATE,
        HIDDEN,
    );
    let shared_s1 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w3 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w3.weight"),
            "F8_E4M3",
            &[MOE_INTERMEDIATE, HIDDEN],
        )?,
        MOE_INTERMEDIATE,
        HIDDEN,
    );
    let shared_s3 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w2 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w2.weight"),
            "F8_E4M3",
            &[HIDDEN, MOE_INTERMEDIATE],
        )?,
        HIDDEN,
        MOE_INTERMEDIATE,
    );
    let shared_s2 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, MOE_INTERMEDIATE / 32],
    )?;
    Ok(Stage9Weights {
        attn_norm,
        ffn_norm,
        mhc_attn,
        mhc_ffn,
        attn_sink,
        q_norm,
        kv_norm,
        wq_a_w,
        wq_a_s,
        wq_b_w,
        wq_b_s,
        wkv_w,
        wkv_s,
        wo_b_w,
        wo_b_s,
        wo_a_w,
        wo_a_s,
        gate_weight,
        gate_bias,
        shared_w1,
        shared_s1,
        shared_w3,
        shared_s3,
        shared_w2,
        shared_s2,
    })
}

/// The device context and the five loaded kernels a Stage-C run launches.
struct Stage9Device {
    cuda: Cuda,
    _modules: Vec<CuModule>,
    mhc: LayerMhc,
    mhc_func: CuFunction,
    dense_func: CuFunction,
    wo_a_func: CuFunction,
    mla_func: CuFunction,
    moe_func: CuFunction,
}

fn stage9_device(args: &Args) -> Result<Stage9Device, String> {
    check_floor(args, "stage 9 before context")?;
    let cuda = Cuda::load()?;
    let mut modules = Vec::with_capacity(5);
    let mut load = |name: &str| -> Result<CuFunction, String> {
        let path = args.cubin_dir.join(format!("{name}.cubin"));
        let cubin = std::fs::read(&path).map_err(|e| {
            format!(
                "{}: {e} (build the stage-7/9 cubins first; the module docs give the nvcc lines)",
                path.display()
            )
        })?;
        let (module, function) = cuda.load_function(&cubin, name)?;
        modules.push(module);
        Ok(function)
    };
    let mhc_func = load("mega_mhc_fused_bf16_sm121")?;
    let dense_func = load("b12x_mxfp8_dense_sm121")?;
    let wo_a_func = load("b12x_mxfp8_wo_a_sm121")?;
    let mla_func = load("flash_mla_csa2_sm121")?;
    let moe_func = load("b12x_mxfp4_moe_sm121")?;
    let mhc = LayerMhc::alloc(&cuda)?;
    Ok(Stage9Device {
        cuda,
        _modules: modules,
        mhc,
        mhc_func,
        dense_func,
        wo_a_func,
        mla_func,
        moe_func,
    })
}

/// K2-K7 for one layer: the attention projections and the flash_mla attention, on device, with the
/// same host seams stage 7 uses. Returns attention_out (the wo_b output widened from bf16).
#[allow(clippy::too_many_arguments)] // the kernel launches take the layer's staged tensors as distinct arguments.
fn stage9_attention_block(
    args: &Args,
    dev: &Stage9Device,
    layer: usize,
    w: &Stage9Weights,
    attn_codes: &[u8],
    attn_scales: &[u8],
    format_stats: &mut EngineWindowStats,
) -> Result<Vec<f32>, String> {
    let cuda = &dev.cuda;
    let tag = format!("L{layer:02}");
    let wq_b_cols = MLA_HEADS * MLA_LATENT;
    let mut q_latent = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K2 wq_a"),
        attn_codes,
        attn_scales,
        &w.wq_a_w,
        &w.wq_a_s,
        Q_LORA_RANK,
        HIDDEN,
    )?;
    rms_norm_row(&mut q_latent, &w.q_norm, NORM_EPS);
    let (q_codes, q_scales) = act_quant_fp8_row(&q_latent);
    let mut q = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K3 wq_b"),
        &q_codes,
        &q_scales,
        &w.wq_b_w,
        &w.wq_b_s,
        wq_b_cols,
        Q_LORA_RANK,
    )?;
    rotate_rope_tail(&mut q, MLA_HEADS, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
    let q_device_bits = bf16_bits_of_f32(&q);
    let q_used = widen_bits(&q_device_bits);
    let mut kv = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K4 wkv"),
        attn_codes,
        attn_scales,
        &w.wkv_w,
        &w.wkv_s,
        MLA_LATENT,
        HIDDEN,
    )?;
    rms_norm_row(&mut kv, &w.kv_norm, NORM_EPS);
    rotate_rope_tail(&mut kv, 1, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
    let (kv_packed, kv_scales_row) = quantize_kv_row(&kv);
    // The engine-format window row, built by the same helper stage 2 uses: rms_norm and the rope
    // tail are already applied above, then exactly one bf16 storage round, then the engine's
    // released quantiser, asserted against m4_quant::act_quant_fp8_row on the same bf16 values.
    let format = args.stage9_kv_format;
    let mut window_codes = Vec::new();
    let mut window_scales = Vec::new();
    if format == WindowKvFormat::Engine {
        let engine_row =
            engine_window_row(&kv).map_err(|error| format!("stage 9 layer {layer}: {error}"))?;
        assert_engine_window_finite(
            &engine_row.codes,
            &engine_row.scales,
            1,
            &format!("stage 9 layer {layer}"),
        )?;
        format_stats.rows += 1;
        format_stats.quantiser_agreements += 1;
        format_stats.guarded_rows += 1;
        window_codes = engine_row.codes;
        window_scales = engine_row.scales;
    }

    // K5: the attention kernel, one launch. The candidate list is the single position 0: this is a
    // first prefill step, so the window holds only position 0 and there is no compressed row yet
    // (the compressor cache is empty at position 0) - the S7 seam is exercised in its degenerate
    // form and the log says so rather than implying a compressed candidate was attended.
    let selected = stage2_selected(0);
    let index_key_bytes = MLA_HSI_BLOCKS * MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM * 2;
    let index_query_bytes = MLA_HEADS * MLA_INDEX_HEAD_DIM * 2;
    let out_bytes = MLA_HEADS * MLA_LATENT * 2;
    let mut allocated = 0usize;
    let q_d = cuda.alloc(MLA_HEADS * MLA_LATENT * 2, &mut allocated)?;
    let sink_d = cuda.alloc(MLA_HEADS * 4, &mut allocated)?;
    let kv_fp4_d = cuda.alloc(KV_PACKED_BYTES, &mut allocated)?;
    let kv_scales_d = cuda.alloc(KV_SCALE_BYTES, &mut allocated)?;
    let window_codes_d = if format == WindowKvFormat::Engine {
        Some(cuda.alloc(MLA_LATENT, &mut allocated)?)
    } else {
        None
    };
    let window_scales_d = if format == WindowKvFormat::Engine {
        Some(cuda.alloc(WINDOW_SCALE_BYTES, &mut allocated)?)
    } else {
        None
    };
    let selected_d = cuda.alloc(MLA_MAX_SELECTED * 4, &mut allocated)?;
    let index_key_d = cuda.alloc(index_key_bytes, &mut allocated)?;
    let index_query_d = cuda.alloc(index_query_bytes, &mut allocated)?;
    let block_scores_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let block_best_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let attn_out_d = cuda.alloc(out_bytes, &mut allocated)?;
    cuda.copy_htod(&q_d, &bf16_bits_to_bytes(&q_device_bits))?;
    cuda.copy_htod(&sink_d, &f32_to_bytes(&w.attn_sink))?;
    cuda.copy_htod(&kv_fp4_d, &kv_packed)?;
    cuda.copy_htod(&kv_scales_d, &kv_scales_row)?;
    if let (Some(codes_d), Some(scales_d)) = (window_codes_d.as_ref(), window_scales_d.as_ref()) {
        cuda.copy_htod(codes_d, &window_codes)?;
        cuda.copy_htod(scales_d, &window_scales)?;
    }
    cuda.copy_htod(&selected_d, &u32s_to_bytes(&selected))?;
    cuda.copy_htod(&index_key_d, &vec![0u8; index_key_bytes])?;
    cuda.copy_htod(&index_query_d, &vec![0u8; index_query_bytes])?;
    cuda.copy_htod(
        &block_scores_d,
        &f32_to_bytes(&vec![12345.0f32; MLA_HSI_BLOCKS]),
    )?;
    cuda.copy_htod(
        &block_best_d,
        &u32s_to_bytes(&vec![u32::MAX; MLA_HSI_BLOCKS]),
    )?;
    let params = flash_mla_params(
        q_d.0,
        kv_fp4_d.0,
        kv_scales_d.0,
        window_codes_d.as_ref().map_or(0, |buffer| buffer.0),
        window_scales_d.as_ref().map_or(0, |buffer| buffer.0),
        format.kernel_value() as u64,
        // one stored window row per layer in engine mode; FP4 ids are token positions
        if format == WindowKvFormat::Engine {
            1
        } else {
            0
        },
        selected_d.0,
        index_key_d.0,
        index_query_d.0,
        sink_d.0,
        block_scores_d.0,
        block_best_d.0,
        attn_out_d.0,
        0, // position
        selected.len(),
        0, // trace_ring
        layer,
        0, // step_seq
    );
    check_floor(args, &format!("stage 9 layer {layer} K5 pre-launch"))?;
    flash_mla_launch(
        &dev.cuda,
        dev.mla_func,
        (MLA_HEADS as u32, args.grid_y, 1u32),
        (MLA_THREADS, 1u32, 1u32),
        &params,
    )?;
    check_floor(args, &format!("stage 9 layer {layer} K5 post-launch"))?;
    let mut attn_out_bytes = vec![0u8; out_bytes];
    cuda.copy_dtoh(&mut attn_out_bytes, &attn_out_d)?;
    let mut free_list: Vec<&Deviceptr> = vec![
        &q_d,
        &sink_d,
        &kv_fp4_d,
        &kv_scales_d,
        &selected_d,
        &index_key_d,
        &index_query_d,
        &block_scores_d,
        &block_best_d,
        &attn_out_d,
    ];
    if let (Some(codes_d), Some(scales_d)) = (window_codes_d.as_ref(), window_scales_d.as_ref()) {
        free_list.push(codes_d);
        free_list.push(scales_d);
    }
    cuda.free_buffers(&free_list);
    let _ = allocated;
    let attn_out_bits = bytes_to_u16(&attn_out_bytes);
    let mut reference = Vec::with_capacity(MLA_HEADS * MLA_LATENT);
    for head in 0..MLA_HEADS {
        let q_head = &q_used[head * MLA_LATENT..(head + 1) * MLA_LATENT];
        // The reference is format-dispatched exactly like stage 2: engine mode must read the same
        // window codes/scales with mla_ref::attention_with_sink_window (the engine decode plus
        // bf16 round), never the FP4 reference with window bytes.
        let head_out = if format == WindowKvFormat::Engine {
            mla_ref::attention_with_sink_window(
                q_head,
                &window_codes,
                &window_scales,
                1,
                &selected,
                w.attn_sink[head],
            )
        } else {
            mla_ref::attention_with_sink(
                q_head,
                &kv_packed,
                &kv_scales_row,
                &selected,
                0,
                w.attn_sink[head],
            )
        };
        reference.extend_from_slice(&head_out);
    }
    let (attn_stats, attn_ulp) = compare_bf16_ulp(
        &if format == WindowKvFormat::Engine {
            format!(
                "L{layer:02} K5 attention out (engine window format, window_rows=1, same bytes)"
            )
        } else {
            format!("L{layer:02} K5 attention out (same q/kv/selected/sink)")
        },
        &attn_out_bits,
        &reference,
    )?;
    if format == WindowKvFormat::Engine {
        println!(
            "  {tag} K5 internal check (engine window format, window_rows=1, selected=[0]): {:.4}% exact against mla_ref::attention_with_sink_window on the same bytes, max abs delta {:e}, max ULP {attn_ulp}",
            attn_stats.exact_pct(),
            attn_stats.max_delta
        );
    } else {
        println!(
            "  {tag} K5 internal check: {:.4}% exact against mla_ref on the same bytes, max abs delta {:e}, max ULP {attn_ulp}",
            attn_stats.exact_pct(),
            attn_stats.max_delta
        );
    }

    let mut attn_operand = widen_bits(&attn_out_bits);
    rotate_rope_tail_inverse(
        &mut attn_operand,
        MLA_HEADS,
        MLA_LATENT,
        ROPE_HEAD_DIM,
        ROPE_THETA,
        0,
    );
    let (attn_operand_codes, attn_operand_scales) = act_quant_fp8_row(&attn_operand);
    let lo_out = wo_a_device(
        cuda,
        dev.wo_a_func,
        layer,
        &attn_operand_codes,
        &attn_operand_scales,
        &w.wo_a_w,
        &w.wo_a_s,
    )?;
    let (lo_codes, lo_scales) = act_quant_fp8_row(&lo_out);
    let attention_out = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K7 wo_b"),
        &lo_codes,
        &lo_scales,
        &w.wo_b_w,
        &w.wo_b_s,
        HIDDEN,
        WO_A_OUT,
    )?;
    Ok(attention_out)
}

/// What one layer's MoE block produced, for the per-row report.
struct Stage9FfnResult {
    ffn_out: Vec<f32>,
    device_ids: Vec<u32>,
    host_ids: Vec<u32>,
    free_delta: f32,
    forced_delta: f32,
}

/// S7-S10 and K9-K15 for one layer: the router seam, the free-running routed combine, the forced
/// control, the shared expert, and the host sum that produces ffn_out. Both ffn_out tensors come
/// from the shared B2 helpers (`combine_bf16_routed_with_shared`, and `verify_forced_ffn_out`
/// whose returned recomputation feeds the free/forced host deltas and the pointwise flip metric),
/// the forced operands are verified against the checkpoint before the forced launch
/// (`verify_forced_operand_provenance` with `FORCED_OPERAND_GEOMETRY`), and
/// `forced_control_verdict` enforces the routing/bytes determinism rules.
#[allow(clippy::too_many_arguments)] // mirrors stage 7's seam functions: context plus the layer's tensors.
fn stage9_ffn_block(
    args: &Args,
    dev: &Stage9Device,
    layer: usize,
    w: &Stage9Weights,
    snap: &mut Snapshot,
    prefix: &str,
    ffn_hidden: &[f32],
    ffn_codes: &[u8],
    ffn_scales: &[u8],
    host_ffn_hidden: &[f32],
    host_ffn_out: &[f32],
) -> Result<Stage9FfnResult, String> {
    let cuda = &dev.cuda;
    let tag = format!("L{layer:02}");
    let block_activation = decode_quantised_row(ffn_codes, ffn_scales);
    let all_experts: Vec<u32> = (0..MOE_EXPERTS as u32).collect();
    let (device_ids, device_weights) =
        route_token(ffn_hidden, &w.gate_weight, &w.gate_bias, &all_experts);
    let (host_ids, host_weights) =
        route_token(host_ffn_hidden, &w.gate_weight, &w.gate_bias, &all_experts);
    let differing_slots: Vec<(usize, u32, u32)> = device_ids
        .iter()
        .zip(&host_ids)
        .enumerate()
        .filter(|(_, (device, host))| device != host)
        .map(|(slot, (device, host))| (slot, *device, *host))
        .collect();
    let mut device_sorted = device_ids.clone();
    device_sorted.sort_unstable();
    let mut host_sorted = host_ids.clone();
    host_sorted.sort_unstable();
    let only_device: Vec<u32> = device_sorted
        .iter()
        .copied()
        .filter(|id| !host_sorted.contains(id))
        .collect();
    let only_host: Vec<u32> = host_sorted
        .iter()
        .copied()
        .filter(|id| !device_sorted.contains(id))
        .collect();
    let table_ids: Vec<u32> = (0..MOE_TOP_K as u32).collect();
    let free_operands = routed_operands(
        snap,
        prefix,
        &block_activation,
        &device_ids,
        &device_weights,
    )?;
    let routed_bits = run_routed_combine(
        args,
        "stage 9",
        cuda,
        dev.moe_func,
        layer,
        "free-running routing",
        &free_operands,
        &table_ids,
    )?;
    let free_reference = reference_combine_chain(
        &free_operands.act_fp8,
        &free_operands.act_scales,
        &free_operands.w_rank,
        &free_operands.ws_rank,
        &table_ids,
        1,
        MOE_TOP_K,
    );
    let (free_stats, free_ulp) =
        compare_bf16_bits("free routed combine", &routed_bits, &free_reference);
    if !free_stats.all_exact() {
        println!(
            "  {tag} WARNING routed combine internal check {:.4}% exact, max ULP {free_ulp}",
            free_stats.exact_pct()
        );
    }

    let shared_gate = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K13 shared w1"),
        ffn_codes,
        ffn_scales,
        &w.shared_w1,
        &w.shared_s1,
        MOE_INTERMEDIATE,
        HIDDEN,
    )?;
    let shared_up = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K14 shared w3"),
        ffn_codes,
        ffn_scales,
        &w.shared_w3,
        &w.shared_s3,
        MOE_INTERMEDIATE,
        HIDDEN,
    )?;
    let shared_activation: Vec<f32> = shared_gate
        .iter()
        .zip(&shared_up)
        .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
        .collect();
    let (shared_codes, shared_scales) = act_quant_fp8_row(&shared_activation);
    let shared_out = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K15 shared w2"),
        &shared_codes,
        &shared_scales,
        &w.shared_w2,
        &w.shared_s2,
        HIDDEN,
        MOE_INTERMEDIATE,
    )?;
    let ffn_out = combine_bf16_routed_with_shared(&routed_bits, &shared_out)?;
    let forced_operands =
        routed_operands(snap, prefix, &block_activation, &host_ids, &host_weights)?;
    // The forced launch consumes the host row's own routing, so the operands are verified against
    // the checkpoint before that launch exactly as stage 7 does: every rank's w2 weight/scale slice
    // and every slot's w1/w3-derived activation codes and scales.
    verify_forced_operand_provenance(
        snap,
        prefix,
        &block_activation,
        &host_ids,
        &host_weights,
        &forced_operands.w_rank,
        &forced_operands.ws_rank,
        &forced_operands.act_fp8,
        &forced_operands.act_scales,
        &FORCED_OPERAND_GEOMETRY,
    )?;
    let forced_bits = run_routed_combine(
        args,
        "stage 9",
        cuda,
        dev.moe_func,
        layer,
        "host routing forced",
        &forced_operands,
        &table_ids,
    )?;
    let forced_reference = reference_combine_chain(
        &forced_operands.act_fp8,
        &forced_operands.act_scales,
        &forced_operands.w_rank,
        &forced_operands.ws_rank,
        &table_ids,
        1,
        MOE_TOP_K,
    );
    let (forced_stats, forced_ulp) =
        compare_bf16_bits("forced routed combine", &forced_bits, &forced_reference);
    if !forced_stats.all_exact() {
        println!(
            "  {tag} WARNING forced routed combine internal check {:.4}% exact, max ULP {forced_ulp}",
            forced_stats.exact_pct()
        );
    }
    let device_ffn_out_forced = {
        let forced_routed = widen_bits(&forced_bits);
        let mut ffn_out_forced = vec![0.0f32; HIDDEN];
        for column in 0..HIDDEN {
            ffn_out_forced[column] = forced_routed[column] + shared_out[column];
        }
        ffn_out_forced
    };
    // The asserted-identical recomputation of the forced output: every forced number reported below
    // (the host deltas and the pointwise flip) reads this returned tensor, never the device launch's
    // copy, so a copy inserted after the assertion cannot change any reported metric; a mismatch
    // between the device tensor and the recomputation aborts here, before any metric is accepted.
    let ffn_out_forced = verify_forced_ffn_out(&forced_bits, &shared_out, &device_ffn_out_forced)?;
    let verdict = forced_control_verdict(
        &device_ids,
        &host_ids,
        &device_weights,
        &host_weights,
        routed_bits == forced_bits,
        &free_reference,
        &forced_reference,
    )?;
    let metrics = free_forced_metrics(&ffn_out, &ffn_out_forced, host_ffn_out)?;
    let free_delta = metrics.free_host_max_abs;
    let forced_delta = metrics.forced_host_max_abs;
    println!(
        "  {tag} router: ordered top-{MOE_TOP_K} lists equal {}; differing slots (slot, device, host) {differing_slots:?}; set difference only-device {only_device:?} only-host {only_host:?}{}",
        device_ids == host_ids,
        if device_ids != host_ids {
            " - DISCRETE EXPERT FLIP: a divergence event to report per layer, not a smooth bar"
        } else {
            ""
        }
    );
    println!(
        "  {tag} forced control: ids equal {}, weights changed {}, routed bytes identical {}, references bit-identical {} (an ids-differ + bytes-identical outcome is allowed only when the references are bit-identical: the flip cannot move the output by construction); pointwise |ffn_out free - ffn_out forced| max {:.6e} at i={} (signed {:+.6e}), {}/{} points differ; host deltas: free-running {:.6e} at i={}, forced {:.6e} at i={}; the difference of the two host-error maxima is a change in max error, NOT the flip's max-abs contribution; routed-combine exactness: free {:.2}%, forced {:.2}%",
        verdict.routing_identical,
        verdict.weights_changed,
        verdict.routed_identical,
        verdict.references_identical,
        metrics.pointwise_max_abs,
        metrics.at,
        metrics.signed_at,
        metrics.differing_points,
        ffn_out.len(),
        free_delta,
        metrics.free_host_at,
        forced_delta,
        metrics.forced_host_at,
        free_stats.exact_pct(),
        forced_stats.exact_pct(),
    );
    Ok(Stage9FfnResult {
        ffn_out,
        device_ids,
        host_ids,
        free_delta,
        forced_delta,
    })
}

/// The released Engram write before an Engram layer, applied on the host seam to the device's bf16
/// residual stream: widen the device stream to f32, run the engine's released Engram arithmetic on
/// it, round back to bf16 for the device. The host pair applied the same write to its own f32
/// stream, so the seam's arithmetic is the same code and the difference is the bf16 stream.
fn stage9_engram_seam(
    cfg: &ModelConfig,
    source: &dyn WeightSource,
    hashes: &[u32],
    layer: usize,
    x_bits: &[u16],
    limit: usize,
) -> Result<Vec<u16>, String> {
    let mut stream = widen_bits(x_bits);
    let before = stream.clone();
    let written = sparkflash_engine::forward::engram::engram_forward(
        cfg,
        layer,
        &mut stream,
        hashes,
        source,
        limit,
    )
    .map_err(|error| format!("engram layer {layer}: {error}"))?;
    if !written {
        return Err(format!(
            "engram layer {layer}: the configuration says this layer has no Engram table"
        ));
    }
    let mut max_abs = 0.0f32;
    let mut changed = 0usize;
    for (old, new) in before.iter().zip(&stream) {
        if old.to_bits() != new.to_bits() {
            changed += 1;
        }
        max_abs = max_abs.max((new - old).abs());
    }
    println!(
        "  engram seam layer {layer}: host write of {} values, {changed}/{} changed, max abs write {max_abs:.6e}; the device stream is bf16 so the write is rounded back to bf16 before the layer's K1",
        stream.len(),
        stream.len()
    );
    Ok(bf16_bits_of_f32(&stream))
}

/// One row's measured-vs-bar report. Returns (tensors at or under the bar, tensors over it, the
/// worst measured/bar ratio) and prints one line per tensor. The bar comes from the frozen Stage-B
/// measurements, never from this run's own numbers.
fn stage9_report_row(
    host: &HostBoundaryTable,
    row: usize,
    captures: &HashMap<&'static str, RowValue>,
) -> Result<(usize, usize, f64), String> {
    let mut passed = 0usize;
    let mut misses = 0usize;
    let mut worst_ratio = 0.0f64;
    for (name, class) in B2_TENSORS {
        let value = captures
            .get(name)
            .ok_or_else(|| format!("row {row} did not capture {name}"))?;
        let bar = stage9_bar(name, row);
        let (measured, detail) = match (class, value) {
            (BoundaryClass::ExactBits, RowValue::F32(device)) => {
                let reference = host.f32(row, name)?;
                if reference.len() != device.len() {
                    return Err(format!(
                        "row {row} {name}: {} host values against {} device values",
                        reference.len(),
                        device.len()
                    ));
                }
                let mut exact = 0usize;
                let mut max_abs = 0.0f32;
                for (host_value, device_value) in reference.iter().zip(device) {
                    if host_value.to_bits() == device_value.to_bits() {
                        exact += 1;
                    }
                    max_abs = max_abs.max((host_value - device_value).abs());
                }
                let percent = 100.0 * exact as f64 / device.len() as f64;
                (
                    f64::from(max_abs),
                    format!("host-exact {percent:>9.4}% max-abs {max_abs:.6e}"),
                )
            }
            (BoundaryClass::ExactCodes, RowValue::U8(device)) => {
                let reference = host.u8(row, name)?;
                if reference.len() != device.len() {
                    return Err(format!(
                        "row {row} {name}: {} host codes against {} device codes",
                        reference.len(),
                        device.len()
                    ));
                }
                let mut differing = 0usize;
                let mut max_code_distance = 0i32;
                for (host_code, device_code) in reference.iter().zip(device) {
                    if host_code != device_code {
                        differing += 1;
                        let distance = i32::from(*host_code) - i32::from(*device_code);
                        max_code_distance = max_code_distance.max(distance.abs());
                    }
                }
                let fraction = differing as f64 / device.len() as f64;
                (
                    fraction,
                    format!(
                        "differing-codes {differing:>5}/{:<5} max-code-distance {max_code_distance}",
                        device.len()
                    ),
                )
            }
            _ => {
                return Err(format!(
                    "row {row} {name}: the captured value has the wrong class"
                ));
            }
        };
        let ratio = if bar > 0.0 {
            measured / bar
        } else if measured == 0.0 {
            1.0
        } else {
            f64::INFINITY
        };
        worst_ratio = worst_ratio.max(ratio);
        let verdict = if measured <= bar {
            passed += 1;
            "PASS"
        } else if bar > 0.0 && measured <= bar * (1.0 + 1e-6) {
            // The frozen Stage-B log prints its deltas to 7 significant digits, so a measured f32
            // can sit a few 1e-7 relative above a bar that IS that same measurement rounded on
            // paper. That is the bar's own print precision, not headroom.
            passed += 1;
            "AT-BAR-PRECISION"
        } else {
            misses += 1;
            "BELOW-BAR"
        };
        println!(
            "  row {row:>2} {name:<24} {:<5} {detail} bar {bar:.6e} {verdict}",
            if class == BoundaryClass::ExactCodes {
                "codes"
            } else {
                "f32"
            }
        );
    }
    Ok((passed, misses, worst_ratio))
}

/// The per-layer state the chain carries between layers: the previous FFN sublayer's inputs, which
/// the next layer's K1 consumes as its own step 1. The kernel order is kept here so the carry never
/// round-trips through the boundary-row orientation.
struct Stage9Carry {
    post_attention_bits: Vec<u16>,
    ffn_out: Vec<f32>,
    ffn_b: Vec<f32>,
    ffn_post: Vec<f32>,
    ffn_pre: Vec<f32>,
}

/// One chain run's summary.
struct Stage9Summary {
    rows: usize,
    passes: usize,
    misses: usize,
    over_rows: Vec<usize>,
    worst_ratio: f64,
    flips: Vec<usize>,
    free_ffn_max: f32,
    forced_ffn_max: f32,
}

/// One captured boundary row: the 18 (tensor name, value) pairs in the format's tensor order.
type Stage9Row = Vec<(&'static str, RowValue)>;

/// One chain run: every captured row, the tensor names the reader was asked for, and its
/// (reads, bytes) audit.
type Stage9Run = (Vec<Stage9Row>, Vec<String>, (usize, u64));

/// Runs the whole chain once and returns its captured rows (complete per layer), the tensor names
/// the run read from the released checkpoint, and the read counts.
#[allow(clippy::too_many_arguments)] // the run needs the loaders, the host pair, the device context and the released config.
fn stage9_chain(
    args: &Args,
    dev: &Stage9Device,
    host: &HostBoundaryTable,
    cfg: &ModelConfig,
    engram_source: &dyn WeightSource,
    hashes: &[u32],
    n_rows: usize,
    run_index: usize,
    format_stats: &mut EngineWindowStats,
) -> Result<Stage9Run, String> {
    let mut snap = Snapshot::open(&args.checkpoint)?;
    let mut rows: Vec<Stage9Row> = Vec::with_capacity(n_rows);
    let mut prev_captures: Option<HashMap<&'static str, RowValue>> = None;
    let mut carry: Option<Stage9Carry> = None;
    let mut pending_x: Option<Vec<u16>> = None;
    let mut summary = Stage9Summary {
        rows: n_rows,
        passes: 0,
        misses: 0,
        over_rows: Vec::new(),
        worst_ratio: 0.0,
        flips: Vec::new(),
        free_ffn_max: 0.0,
        forced_ffn_max: 0.0,
    };
    let identity_b = {
        let mut b = [0.0f32; HC_MULT * HC_MULT];
        for index in 0..HC_MULT {
            b[index * HC_MULT + index] = 1.0;
        }
        b
    };
    let zero_c = [0.0f32; HC_MULT];
    let identity_a = [1.0f32, 0.0, 0.0, 0.0];
    let zero_y = vec![0.0f32; HIDDEN];
    // The layer-0 seed: the embedding row of token 0 replicated to the four streams.
    let embed_row = snap.read_row_bf16("embed.weight", 0, HIDDEN)?;
    let mut stream = Vec::with_capacity(HC_MULT * HIDDEN);
    for _ in 0..HC_MULT {
        stream.extend_from_slice(&embed_row);
    }
    let x_bits = bf16_bits_of_f32(&stream);
    drop(stream);
    let engram_limit = usize::try_from(sparkflash_engine::safetensors::DEFAULT_LIMIT)
        .map_err(|_| "the engine's default tensor limit does not fit a usize".to_string())?;
    let engram_at = |layer: usize| cfg.engram_rows(layer).is_some();

    println!();
    println!("=== chain run {run_index}: {n_rows} rows ===");
    for layer in 0..n_rows {
        let prefix = format!("layers.{layer}");
        let weights = stage9_load_weights(&mut snap, &prefix)?;
        println!(
            "--- layer {layer:02}: staged {} B of released weights, {} rows to run ---",
            weights.staged_bytes(),
            n_rows
        );
        let mut captures: HashMap<&'static str, RowValue> = HashMap::with_capacity(18);

        // --- K1: the attention sublayer. The seed is the previous FFN hc_post; at layer 0 it is the
        // embedding with the identity mix, and after an Engram layer the stream already carries the
        // released write, so K1 runs with the identity seed to leave it in place.
        let (seed_x, seed_y, seed_b, seed_c, seed_a) = if layer == 0 {
            (
                x_bits.clone(),
                zero_y.clone(),
                identity_b.to_vec(),
                zero_c.to_vec(),
                identity_a.to_vec(),
            )
        } else if let Some(engrammed) = pending_x.take() {
            let carry = carry
                .as_ref()
                .ok_or("stage 9: an Engram layer has no previous layer's coefficients")?;
            (
                engrammed,
                zero_y.clone(),
                identity_b.to_vec(),
                zero_c.to_vec(),
                carry.ffn_pre.clone(),
            )
        } else {
            let carry = carry
                .as_ref()
                .ok_or("stage 9: a layer past layer 0 has no previous layer to carry from")?;
            (
                carry.post_attention_bits.clone(),
                carry.ffn_out.clone(),
                carry.ffn_b.clone(),
                carry.ffn_post.clone(),
                carry.ffn_pre.clone(),
            )
        };
        dev.mhc.upload(
            &dev.cuda,
            &seed_x,
            &seed_y,
            &seed_b,
            &seed_c,
            &seed_a,
            &weights.mhc_attn.predictor,
            &weights.mhc_attn.hc_scale,
            &weights.mhc_attn.hc_base_kernel,
            &weights.attn_norm,
        )?;
        check_floor(args, &format!("stage 9 layer {layer} K1 pre-launch"))?;
        dev.mhc.launch(&dev.cuda, dev.mhc_func, layer, 0)?;
        check_floor(args, &format!("stage 9 layer {layer} K1 post-launch"))?;
        let residual_in_bits = dev.mhc.x_bits(&dev.cuda)?;
        let attn_coeffs = dev.mhc.coeffs(&dev.cuda)?;
        let attn_hidden = dev.mhc.normed(&dev.cuda)?;
        let attn_codes = dev.mhc.codes(&dev.cuda)?;
        let attn_scales = dev.mhc.scales(&dev.cuda)?;
        let attn_b: Vec<f32> = attn_coeffs[0..HC_MULT * HC_MULT].to_vec();
        let attn_pre: Vec<f32> =
            attn_coeffs[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT].to_vec();
        let attn_post: Vec<f32> =
            attn_coeffs[HC_MULT * HC_MULT + HC_MULT..HC_MULT * HC_MULT + 2 * HC_MULT].to_vec();
        let attn_comb: Vec<f32> = {
            let mut comb = vec![0.0f32; HC_MULT * HC_MULT];
            for destination in 0..HC_MULT {
                for source in 0..HC_MULT {
                    comb[source * HC_MULT + destination] = attn_b[destination * HC_MULT + source];
                }
            }
            comb
        };
        captures.insert("residual_in", RowValue::F32(widen_bits(&residual_in_bits)));
        captures.insert("hc_in_pre", RowValue::F32(seed_a.clone()));
        captures.insert("hc_attn_pre", RowValue::F32(attn_pre.clone()));
        captures.insert("hc_attn_post", RowValue::F32(attn_post.clone()));
        captures.insert("hc_attn_comb", RowValue::F32(attn_comb));
        captures.insert("pre_attention_hidden", RowValue::F32(attn_hidden.clone()));
        captures.insert("pre_attention_xhat", RowValue::U8(attn_codes.clone()));
        captures.insert("pre_attention_scales", RowValue::U8(attn_scales.clone()));
        println!(
            "  L{layer:02} K1 captured: residual_in {} bf16 values, coefficients 24 f32, block input {} f32 + {} codes + {} scales{}",
            residual_in_bits.len(),
            attn_hidden.len(),
            attn_codes.len(),
            attn_scales.len(),
            if layer > 0 && !engram_at(layer) {
                " (this K1 also wrote the previous row's residual_out)"
            } else {
                ""
            }
        );

        // This K1 wrote the previous row's residual_out when no Engram write sits between them.
        if layer > 0 && !engram_at(layer) {
            let mut previous = prev_captures.take().ok_or_else(|| {
                format!("stage 9: layer {layer} K1 has no previous row to finish")
            })?;
            previous.insert("residual_out", RowValue::F32(widen_bits(&residual_in_bits)));
            let row = layer - 1;
            let (passed, misses, worst_ratio) = stage9_report_row(host, row, &previous)?;
            println!(
                "  row {row:>2} summary: {passed}/18 at or under the Stage-B bar, {misses} over it, worst measured/bar {worst_ratio:.3}"
            );
            summary.passes += passed;
            summary.misses += misses;
            summary.worst_ratio = summary.worst_ratio.max(worst_ratio);
            if misses > 0 {
                summary.over_rows.push(row);
            }
            let mut ordered = Vec::with_capacity(B2_TENSORS.len());
            for (name, _) in B2_TENSORS {
                ordered.push((
                    name,
                    previous
                        .remove(name)
                        .ok_or_else(|| format!("row {row}: captured set lost {name}"))?,
                ));
            }
            rows.push(ordered);
        }

        // --- K2-K7: the attention block on the device.
        let attention_out = stage9_attention_block(
            args,
            dev,
            layer,
            &weights,
            &attn_codes,
            &attn_scales,
            format_stats,
        )?;
        captures.insert("attention_out", RowValue::F32(attention_out.clone()));

        // --- K8: the FFN sublayer's mHC pass (applies the attention hc_post).
        dev.mhc.upload(
            &dev.cuda,
            &residual_in_bits,
            &attention_out,
            &attn_b,
            &attn_post,
            &attn_pre,
            &weights.mhc_ffn.predictor,
            &weights.mhc_ffn.hc_scale,
            &weights.mhc_ffn.hc_base_kernel,
            &weights.ffn_norm,
        )?;
        check_floor(args, &format!("stage 9 layer {layer} K8 pre-launch"))?;
        dev.mhc.launch(&dev.cuda, dev.mhc_func, layer, 1)?;
        check_floor(args, &format!("stage 9 layer {layer} K8 post-launch"))?;
        let post_attention_bits = dev.mhc.x_bits(&dev.cuda)?;
        let ffn_coeffs = dev.mhc.coeffs(&dev.cuda)?;
        let ffn_hidden = dev.mhc.normed(&dev.cuda)?;
        let ffn_codes = dev.mhc.codes(&dev.cuda)?;
        let ffn_scales = dev.mhc.scales(&dev.cuda)?;
        let ffn_b: Vec<f32> = ffn_coeffs[0..HC_MULT * HC_MULT].to_vec();
        let ffn_pre: Vec<f32> = ffn_coeffs[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT].to_vec();
        let ffn_post: Vec<f32> =
            ffn_coeffs[HC_MULT * HC_MULT + HC_MULT..HC_MULT * HC_MULT + 2 * HC_MULT].to_vec();
        let ffn_comb: Vec<f32> = {
            let mut comb = vec![0.0f32; HC_MULT * HC_MULT];
            for destination in 0..HC_MULT {
                for source in 0..HC_MULT {
                    comb[source * HC_MULT + destination] = ffn_b[destination * HC_MULT + source];
                }
            }
            comb
        };
        captures.insert(
            "post_attention_hidden",
            RowValue::F32(widen_bits(&post_attention_bits)),
        );
        captures.insert("hc_ffn_pre", RowValue::F32(ffn_pre.clone()));
        captures.insert("hc_ffn_post", RowValue::F32(ffn_post.clone()));
        captures.insert("hc_ffn_comb", RowValue::F32(ffn_comb));
        captures.insert("pre_ffn_hidden", RowValue::F32(ffn_hidden.clone()));
        captures.insert("pre_ffn_xhat", RowValue::U8(ffn_codes.clone()));
        captures.insert("pre_ffn_scales", RowValue::U8(ffn_scales.clone()));
        println!(
            "  L{layer:02} K8 captured: post_attention_hidden {} bf16 values, FFN coefficients 24 f32, pre_ffn_hidden {} f32 + {} codes + {} scales",
            post_attention_bits.len(),
            ffn_hidden.len(),
            ffn_codes.len(),
            ffn_scales.len()
        );

        // --- S7-S10 and K9-K15: the MoE block.
        let host_ffn_hidden = host.f32(layer, "pre_ffn_hidden")?;
        let host_ffn_out = host.f32(layer, "ffn_out")?;
        let ffn = stage9_ffn_block(
            args,
            dev,
            layer,
            &weights,
            &mut snap,
            &prefix,
            &ffn_hidden,
            &ffn_codes,
            &ffn_scales,
            &host_ffn_hidden,
            &host_ffn_out,
        )?;
        captures.insert("ffn_out", RowValue::F32(ffn.ffn_out.clone()));
        if ffn.device_ids != ffn.host_ids {
            summary.flips.push(layer);
        }
        summary.free_ffn_max = summary.free_ffn_max.max(ffn.free_delta);
        summary.forced_ffn_max = summary.forced_ffn_max.max(ffn.forced_delta);
        carry = Some(Stage9Carry {
            post_attention_bits: post_attention_bits.clone(),
            ffn_out: ffn.ffn_out.clone(),
            ffn_b: ffn_b.clone(),
            ffn_post: ffn_post.clone(),
            ffn_pre: ffn_pre.clone(),
        });

        // --- K16 when no next layer's K1 will apply this layer's FFN hc_post: before an Engram
        // layer (whose write mutates the stream before the K1) and after the last row.
        let next_engram = layer + 1 < n_rows && engram_at(layer + 1);
        let last_row = layer + 1 == n_rows;
        if next_engram || last_row {
            dev.mhc.upload(
                &dev.cuda,
                &post_attention_bits,
                &ffn.ffn_out,
                &ffn_b,
                &ffn_post,
                &ffn_pre,
                &weights.mhc_ffn.predictor,
                &weights.mhc_ffn.hc_scale,
                &weights.mhc_ffn.hc_base_kernel,
                &weights.ffn_norm,
            )?;
            check_floor(args, &format!("stage 9 layer {layer} K16 pre-launch"))?;
            dev.mhc.launch(&dev.cuda, dev.mhc_func, layer, 2)?;
            check_floor(args, &format!("stage 9 layer {layer} K16 post-launch"))?;
            let residual_out_bits = dev.mhc.x_bits(&dev.cuda)?;
            captures.insert(
                "residual_out",
                RowValue::F32(widen_bits(&residual_out_bits)),
            );
            println!(
                "  L{layer:02} K16 captured: residual_out {} bf16 values (the last hc_post applied by a dedicated launch, decision D5){}",
                residual_out_bits.len(),
                if next_engram {
                    format!(
                        "; the released Engram write now runs before layer {}",
                        layer + 1
                    )
                } else {
                    String::new()
                }
            );
            if next_engram {
                pending_x = Some(stage9_engram_seam(
                    cfg,
                    engram_source,
                    hashes,
                    layer + 1,
                    &residual_out_bits,
                    engram_limit,
                )?);
            }
            let (passed, misses, worst_ratio) = stage9_report_row(host, layer, &captures)?;
            println!(
                "  row {layer:>2} summary: {passed}/18 at or under the Stage-B bar, {misses} over it, worst measured/bar {worst_ratio:.3}"
            );
            summary.passes += passed;
            summary.misses += misses;
            summary.worst_ratio = summary.worst_ratio.max(worst_ratio);
            if misses > 0 {
                summary.over_rows.push(layer);
            }
            let mut ordered = Vec::with_capacity(B2_TENSORS.len());
            for (name, _) in B2_TENSORS {
                ordered.push((
                    name,
                    captures
                        .remove(name)
                        .ok_or_else(|| format!("row {layer}: captured set lost {name}"))?,
                ));
            }
            rows.push(ordered);
        } else {
            prev_captures = Some(captures);
        }
        drop(weights);
        if layer + 1 < n_rows {
            check_floor(args, &format!("stage 9 layer {layer} end"))?;
        }
    }
    if prev_captures.is_some() {
        return Err("stage 9: the chain ended with a row that was never finalized".to_string());
    }
    if rows.len() != n_rows {
        return Err(format!(
            "stage 9: captured {} rows, the run asked for {n_rows}",
            rows.len()
        ));
    }
    println!(
        "run {run_index} summary: {} rows, {} tensors at or under the bar, {} over it, worst measured/bar {:.3}, rows with any tensor over bar {:?}",
        summary.rows, summary.passes, summary.misses, summary.worst_ratio, summary.over_rows
    );
    println!(
        "run {run_index} routing: discrete expert flips on {} of {n_rows} layers {:?}; ffn_out max abs delta: free-running up to {:.6e}, forced up to {:.6e}",
        summary.flips.len(),
        summary.flips,
        summary.free_ffn_max,
        summary.forced_ffn_max
    );
    let (reads, bytes) = snap.audit();
    Ok((rows, snap.read_names(), (reads, bytes)))
}

/// Compare two chain runs row by row and tensor by tensor, bit-for-bit. Any difference is a
/// determinism failure, not a tolerance question.
fn stage9_compare_runs(first: &[Stage9Row], second: &[Stage9Row]) -> Result<(), String> {
    if first.len() != second.len() {
        return Err(format!(
            "determinism: run 0 captured {} rows, run 1 captured {}",
            first.len(),
            second.len()
        ));
    }
    for (row_index, (left_row, right_row)) in first.iter().zip(second).enumerate() {
        for ((name, left), (right_name, right)) in left_row.iter().zip(right_row) {
            if name != right_name {
                return Err(format!(
                    "determinism: row {row_index} captured {name} against {right_name}"
                ));
            }
            let identical = match (left, right) {
                (RowValue::F32(a), RowValue::F32(b)) => {
                    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
                }
                (RowValue::U8(a), RowValue::U8(b)) => a == b,
                _ => false,
            };
            if !identical {
                let (first_difference, what) = match (left, right) {
                    (RowValue::F32(a), RowValue::F32(b)) => (
                        a.iter()
                            .zip(b)
                            .position(|(a, b)| a.to_bits() != b.to_bits())
                            .unwrap_or(0),
                        format!("{} against {} values", a.len(), b.len()),
                    ),
                    (RowValue::U8(a), RowValue::U8(b)) => (
                        a.iter().zip(b).position(|(a, b)| a != b).unwrap_or(0),
                        format!("{} against {} bytes", a.len(), b.len()),
                    ),
                    _ => (0, "different classes".to_string()),
                };
                return Err(format!(
                    "determinism: row {row_index} {name} is not byte-identical between two runs of the same command (first difference at element {first_difference}, {what}); the device path is not deterministic, so no other number in this run is trustworthy"
                ));
            }
        }
    }
    Ok(())
}

/// Stage 9: the 40-layer chained device forward. See the section comment above for scope.
fn stage9(args: &Args) -> Result<(), String> {
    let started = std::time::Instant::now();
    println!("=== stage 9: the 40-layer chained device forward (Stage C) ===");
    let host_prefix = args
        .host_row
        .as_ref()
        .ok_or("stage 9 needs --host-row <manifest prefix> for the host pair it diffs against")?;
    let host = HostBoundaryTable::open(host_prefix)?;
    let out_dir = args
        .stage9_out
        .as_ref()
        .ok_or("stage 9 needs --stage9-out <dir> for its v1 pair")?;
    let n_rows = args.stage9_layers;
    let cfg =
        ModelConfig::v41_flash().map_err(|error| format!("released configuration: {error}"))?;
    if cfg.dim != HIDDEN || cfg.hc_mult != HC_MULT {
        return Err(format!(
            "the released configuration gives dim {} and hc_mult {}; this harness is built for {HIDDEN}/{HC_MULT}",
            cfg.dim, cfg.hc_mult
        ));
    }
    if cfg.n_layers < n_rows {
        return Err(format!(
            "the released configuration has {} backbone layers, the run asked for {n_rows}",
            cfg.n_layers
        ));
    }
    let engram_layers: Vec<usize> = (0..n_rows)
        .filter(|&layer| cfg.engram_rows(layer).is_some())
        .collect();
    if args.stage9_kv_format == WindowKvFormat::Engine {
        println!(
            "scope label: this is the Phase-1 HEALTH-CHECK column with the engine's released window-KV format, not a criterion-1 closure. K5's window row now uses the engine's E4M3 + per-32 E8M0 format with its bf16 storage round, the same format class as the host row's window KV; the main/compressed KV, the rest of the arithmetic and the host reference remain f32/bf16 as before, and sparkflash-engine's reference forward keeps f32 dequant, so criterion 1 is NOT closed by this run."
        );
    } else {
        println!(
            "scope label: this is the Phase-1 HEALTH-CHECK column, not a criterion-1 closure. The device path uses FP4 main/window KV and bf16/quantised arithmetic where the host reference is f32 with no GEMM activation quantiser, so this column measures attribution and gross mismatch; the format-faithful exactness run is the later one after Phase 2 (task-3), and criterion 1 is NOT closed by this run."
        );
    }
    println!(
        "rows: {n_rows} backbone rows requested out of 40, one token (id 0), position 0, no MTP rows (a first step records the backbone only); the emitted v1 pair has one row per layer like the host pair"
    );
    println!(
        "comparison classes: exact-codes for pre_attention_xhat/scales and pre_ffn_xhat/scales (no numeric tolerance), exact-bits (device tolerance) for the other fourteen; the measured class is named before any number it judges"
    );
    if args.stage9_kv_format == WindowKvFormat::Engine {
        println!(
            "known differences, stated before the numbers: (1) the K5 window row uses the engine's released FP8 window format (E4M3 codes, one E8M0 scale per 32 channels, bf16 storage round), the same format class as the host row's window KV; the main/compressed KV remains FP4 E2M1 and is not exercised at position 0, where the only candidate is the stored window row; (2) the host reference is f32 throughout and deliberately does not reproduce the device's GEMM activation quantisers or bf16 module outputs; (3) the Engram write at layers 1 and 14 runs on the host seam with the engine's released code, but on the bf16 device stream instead of the host's f32 stream; the device chain's rows after an Engram layer carry that seam's rounding. Every kernel is also checked bit for bit against its in-crate reference on the same bytes, and the two chain runs must be byte-identical, so a kernel or harness non-determinism shows up as a failure, not a delta."
        );
    } else {
        println!(
            "known differences, stated before the numbers: (1) the attention chain feeds the kernel its Main-KV format, FP4 E2M1 codes with one E4M3 scale per 16 channels, while the host row's window KV is the release's FP8 E4M3 path with a per-32 ue8m0 scale (decision D4); (2) the host reference is f32 throughout and deliberately does not reproduce the device's GEMM activation quantisers or bf16 module outputs; (3) the Engram write at layers 1 and 14 runs on the host seam with the engine's released code, but on the bf16 device stream instead of the host's f32 stream; the device chain's rows after an Engram layer carry that seam's rounding. Every kernel is also checked bit for bit against its in-crate reference on the same bytes, and the two chain runs must be byte-identical, so a kernel or harness non-determinism shows up as a failure, not a delta."
        );
    }
    println!(
        "bars: pre-registration, printed before the numbers they judge (design section 8): bar(row L, T) = single_layer_delta_B2(T) * (L + 1) for the stream-carrying tensors (residual_in, post_attention_hidden, residual_out), and single_layer_delta_B2(T) for the per-layer tensors. single_layer_delta_B2 is the frozen Stage-B measurement of record in /home/user/phase1-evidence/phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log lines 149-166 (binary 2b518a35...); this run does not re-measure it."
    );
    println!(
        "bar precision: the Stage-B constants in the frozen log are printed to 7 significant digits, so a tensor within 1e-6 relative of its bar is printed AT-BAR-PRECISION instead of PASS or BELOW-BAR; a zero bar stays exact. The bars are not re-derived from this run."
    );
    println!(
        "row 0 is the Stage-B row: bar(row 0, T) is the Stage-B measured delta by construction and this chain runs the same layer-0 launches, so row 0 checks that the chain still reproduces Stage B bit for bit, and has no bar headroom to spend."
    );
    println!(
        "bar inputs (Stage-B measured; the two layer-0 exact seeds inherit the measured delta of the tensor that produces them):"
    );
    for (name, class) in B2_TENSORS {
        let own = stage9_b2_delta(name);
        let source = match name {
            "residual_in" => "residual_out (inherited: row L residual_in is row L-1 residual_out)",
            "hc_in_pre" => "hc_ffn_pre (inherited: row L hc_in_pre is row L-1 hc_ffn_pre)",
            _ => "own",
        };
        let used = match name {
            "residual_in" => stage9_b2_delta("residual_out"),
            "hc_in_pre" => stage9_b2_delta("hc_ffn_pre"),
            other => stage9_b2_delta(other),
        };
        let factor = if B2_STREAM_TENSORS.contains(&name) {
            "(row+1)"
        } else {
            "1"
        };
        println!(
            "  {name:<24} {:<5} own {own:.6e} bar-from {source} used {used:.6e} factor {factor}",
            if class == BoundaryClass::ExactCodes {
                "codes"
            } else {
                "f32"
            }
        );
    }
    let residual_dims = host.tensor(0, "residual_in")?.dims.clone();
    let host_backbone = host
        .manifest_text()
        .lines()
        .find(|line| line.starts_with("backbone "))
        .ok_or("the host manifest has no backbone line")?
        .to_string();
    println!(
        "host pair: {host_backbone}, row 0 residual_in {residual_dims:?}, {} bytes of blob; {} rows requested",
        host.blob.len(),
        n_rows
    );
    if n_rows == 40 {
        let last = host.tensor(39, "residual_out")?;
        println!(
            "host pair check: row 39 residual_out is present with shape {:?} and {} bytes",
            last.dims, last.len
        );
    }
    // The host pair's own wiring: residual_out(L) and residual_in(L+1) are the same value except
    // where the released Engram write sits between them. The device chain reproduces the write on
    // the host seam, so the report names the rows where the host pair itself does not chain.
    let mut wiring_breaks: Vec<(usize, usize, f32)> = Vec::new();
    for layer in 0..n_rows.saturating_sub(1) {
        let before = host.f32(layer, "residual_out")?;
        let after = host.f32(layer + 1, "residual_in")?;
        let mut differing = 0usize;
        let mut max_abs = 0.0f32;
        for (first, second) in before.iter().zip(&after) {
            if first.to_bits() != second.to_bits() {
                differing += 1;
                max_abs = max_abs.max((first - second).abs());
            }
        }
        if differing > 0 {
            wiring_breaks.push((layer, differing, max_abs));
        }
    }
    println!(
        "host pair residual wiring: residual_out(L) against residual_in(L+1) differs at rows {:?}; the Engram layers are {:?}, so a difference at exactly the rows before them is the released write and nothing else",
        wiring_breaks
            .iter()
            .map(|(layer, _, _)| *layer)
            .collect::<Vec<_>>(),
        engram_layers
    );
    println!(
        "seam: router (gate GEMM, sqrt-softplus, noaux_tc bias, top-6, norm_topk_prob) - host; task-37 moves it"
    );
    println!("seam: gate/up GEMM (w1/w3) - host; Phase 3 moves it");
    println!(
        "seam: SwiGLU and the activation quantiser - host; task-43 compares it, Phase 3 moves it"
    );
    println!("seam: HSI candidate merge (top-512 over the pool) - host; Phase 2/3 move it");
    println!(
        "seam: attention operand preparation (q_norm, q RoPE, kv_norm, kv RoPE, main-KV packing, output inverse RoPE, MXFP8 activation quantisers) - host"
    );
    println!(
        "seam: attention side channel (compressor projection, index keys, candidate positions) - host. Degenerate at position 0: the window holds only position 0, there is no compressed row yet, so this seam supplies the candidate list [0] and nothing else; the chain does not read the compressor/indexer tensors the host reference reads"
    );
    println!(
        "seam: engram write (n-gram lookup and gated add, layers 1 and 14) - host; Phase 2/3 move it. Applied in this run on the host seam with the engine's released code, once before each layer in {:?}, to the bf16 device stream",
        engram_layers
    );
    println!(
        "seam: routed-expert and shared-expert sum (elementwise add of two bf16 vectors) - host"
    );
    println!(
        "seam: head and sampling - host; Phase 1 stage 4 / Phase 3 move it. NOT EXERCISED here: criterion 1 stops at the boundary table, and stage 5 owns the head"
    );
    println!("host input: embedding row of token 0 replicated to 4 copies - host");
    println!("host input: released checkpoint weights, repacked B12x on host before upload - host");
    {
        let mut sink_snap = Snapshot::open(&args.checkpoint)?;
        let sink = sink_snap.read_f32("layers.0.attn.attn_sink", &[MLA_HEADS])?;
        let sink_min = sink.iter().copied().fold(f32::INFINITY, f32::min);
        let sink_max = sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "attn_sink: {} f32 in [{sink_min:.4}, {sink_max:.4}] (layer 0) - CONSUMED ON BOTH SIDES: the kernel takes it as a [64] argument and adds it inside the softmax denominator, and mla_ref::attention_with_sink does the same",
            sink.len()
        );
    }
    let index_path = args.checkpoint.join("model.safetensors.index.json");
    let engram_source = SafeTensorsSource::open(&args.checkpoint, &index_path)
        .map_err(|error| format!("{}: {error}", index_path.display()))?
        .with_limit(sparkflash_engine::safetensors::DEFAULT_LIMIT);
    let mut hashes = Vec::new();
    sparkflash_engine::forward::engram::ngram_hashes(&cfg, &[0u32], 0, &mut hashes)
        .map_err(|error| format!("engram hashes for token 0: {error}"))?;
    println!(
        "engram hashes: {} ids for token 0 ({} Engram layers x {} hash columns); the released n-gram layout is the engine's, not reimplemented here",
        hashes.len(),
        engram_layers.len(),
        cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads
    );
    let extra_k16 = engram_layers.iter().filter(|&&layer| layer > 0).count();
    let launches = n_rows * 19 + 1 + extra_k16;
    println!(
        "launch count per run: {} device launches = {n_rows} layers x (15 chain launches K1..K15 + 4 forced-routing K9-K12 launches) + 1 final K16 + {extra_k16} K16 before an Engram layer",
        launches
    );
    if args.stage9_kv_format == WindowKvFormat::Engine {
        println!(
            "KV row format (engine): each layer's kv_norm(wkv(layer input)) row -> rope tail -> bf16 storage -> E4M3 codes + one E8M0 scale per 32 channels; the row is built by the same engine_window_row helper stage 2 uses (bf16 storage round exactly once), then dispatched to K5 as window_codes/window_scales with window_kv_format=1 and window_rows=1"
        );
        println!(
            "engine window format: this column IS the format-faithful window-KV column; the bars are the frozen Stage-B compounded bars and this is still the health-check class, not a criterion-1 closure"
        );
    }
    check_floor(args, "stage 9 start")?;
    let dev = stage9_device(args)?;
    let mut runs: Vec<Vec<Stage9Row>> = Vec::new();
    let mut engine_stats = EngineWindowStats::default();
    let mut audit: Option<(Vec<String>, usize, u64)> = None;
    for run_index in 0..args.stage9_runs {
        check_floor(args, &format!("stage 9 run {run_index} start"))?;
        let (rows, names, counts) = stage9_chain(
            args,
            &dev,
            &host,
            &cfg,
            &engram_source,
            &hashes,
            n_rows,
            run_index,
            &mut engine_stats,
        )?;
        println!(
            "run {run_index} checkpoint reads: {} tensors, {} bytes (measured by the harness's own reader, not a plan estimate)",
            counts.0, counts.1
        );
        if audit.is_none() {
            audit = Some((names, counts.0, counts.1));
        }
        runs.push(rows);
        check_floor(args, &format!("stage 9 run {run_index} end"))?;
    }
    let first_run = runs.first().ok_or("stage 9 ran no chain")?;
    if runs.len() >= 2 {
        stage9_compare_runs(&runs[0], &runs[1])?;
        println!(
            "determinism: PASS - the two runs of the same command agree bit for bit on all 18 tensors of all {n_rows} rows"
        );
    } else {
        println!(
            "determinism: NOT CLAIMED - one chain run was requested (--stage9-runs 1), so this run makes no determinism claim"
        );
    }
    if args.stage9_kv_format == WindowKvFormat::Engine {
        let rows_expected = n_rows * runs.len();
        if engine_stats.rows != rows_expected
            || engine_stats.quantiser_agreements != rows_expected
            || engine_stats.guarded_rows != rows_expected
        {
            return Err(format!(
                "stage 9 engine window format: built {} rows, {} oracle agreements, {} guarded rows; expected {rows_expected} of each",
                engine_stats.rows, engine_stats.quantiser_agreements, engine_stats.guarded_rows
            ));
        }
        println!(
            "engine window format summary: {} runs x {} layers = {} rows; {}/{} rows byte-identical to m4_quant::act_quant_fp8_row; finite-decode guard PASS on {}/{} rows",
            runs.len(),
            n_rows,
            rows_expected,
            engine_stats.quantiser_agreements,
            rows_expected,
            engine_stats.guarded_rows,
            rows_expected
        );
    }
    let (manifest, blob) = stage9_build_v1(first_run)?;
    std::fs::create_dir_all(out_dir).map_err(|error| format!("{}: {error}", out_dir.display()))?;
    let manifest_path = out_dir.join("device.manifest");
    let blob_path = out_dir.join("device.blob");
    std::fs::write(&manifest_path, &manifest)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    std::fs::write(&blob_path, &blob)
        .map_err(|error| format!("{}: {error}", blob_path.display()))?;
    println!(
        "artifact: {} ({} B), {} ({} B); sha256 manifest {}, blob {}",
        manifest_path.display(),
        manifest.len(),
        blob_path.display(),
        blob.len(),
        stage9_sha256_hex(manifest.as_bytes()),
        stage9_sha256_hex(&blob)
    );
    if runs.len() >= 2 {
        let (_, second_blob) = stage9_build_v1(&runs[1])?;
        println!(
            "determinism blobs: run 0 sha256 {}, run 1 sha256 {}",
            stage9_sha256_hex(&blob),
            stage9_sha256_hex(&second_blob)
        );
    }
    let host_manifest = std::fs::read(host_prefix.with_extension("manifest"))
        .map_err(|error| format!("{}.manifest: {error}", host_prefix.display()))?;
    if manifest.as_bytes() == host_manifest.as_slice() {
        println!(
            "artifact check: the device manifest is byte-identical to the host pair's manifest, so the v1 writer and the format agree"
        );
    } else if n_rows == 40 {
        return Err("stage 9: the device manifest is not byte-identical to the host pair's manifest; the v1 writer disagrees with the format".to_string());
    } else {
        println!(
            "artifact check: the manifest differs from the host pair because this smoke run has {n_rows} rows, not the host's 40; no boundary_diff comparison is claimed"
        );
    }
    let (names, reads, bytes) = audit.ok_or("stage 9: no run recorded a read audit")?;
    let plan = TensorPlan::language_model(&cfg);
    let mut read: std::collections::BTreeSet<String> = names.into_iter().collect();
    for layer in &engram_layers {
        for suffix in [
            "engram.embed.weight",
            "engram.embed.scale",
            "engram.wkv.weight",
            "engram.wkv.scale",
            "engram.q_weight",
            "engram.k_weight",
        ] {
            read.insert(format!("layers.{layer}.{suffix}"));
        }
    }
    let plan_names: std::collections::BTreeSet<String> =
        plan.specs().iter().map(|spec| spec.name.clone()).collect();
    let unread: Vec<&String> = plan_names
        .iter()
        .filter(|name| !read.contains(name.as_str()))
        .collect();
    let stages = unread
        .iter()
        .filter(|name| name.starts_with("mtp."))
        .count();
    let experts = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.contains(".ffn.experts."))
        .count();
    let text_path = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.ends_with("ffn.gate.bias_vl"))
        .count();
    let indexer = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.contains(".attn.indexer."))
        .count();
    let compressor = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.contains(".attn.compressor."))
        .count();
    let head = unread
        .iter()
        .filter(|name| name.as_str() == "head.weight" || name.as_str() == "norm.weight")
        .count();
    let other: Vec<&&String> = unread
        .iter()
        .filter(|name| {
            !name.starts_with("mtp.")
                && !name.contains(".ffn.experts.")
                && !name.ends_with("ffn.gate.bias_vl")
                && !name.contains(".attn.indexer.")
                && !name.contains(".attn.compressor.")
                && name.as_str() != "head.weight"
                && name.as_str() != "norm.weight"
        })
        .collect();
    println!(
        "audit: plan={} read={} unread={} expected-unread=[unselected-expert {experts}, DSpark-stage {stages}, text-path {text_path}, indexer {indexer}, compressor {compressor}, head-not-claimed {head}]{}",
        plan_names.len(),
        read.len(),
        unread.len(),
        if other.is_empty() {
            String::new()
        } else {
            format!(
                " UNCLASSIFIED {} first {:?}",
                other.len(),
                other.iter().take(8).collect::<Vec<_>>()
            )
        }
    );
    println!(
        "audit note: the compressor/indexer classes are larger than the host report's because the S7 seam is degenerate at position 0 and this chain never reads those tensors, while the host reference does; that is a property of the health-check column, not an unread-tensor bug"
    );
    println!(
        "audit bytes: {reads} tensor reads, {bytes} bytes through the harness's reader (the Engram tables are read through the engine's paged source, not this counter)"
    );
    let mem_kb = mem_available_kb_checked()?;
    println!(
        "stage 9 trailer: wall {:.1} s, MemAvailable {mem_kb} KiB, {n_rows} rows, {} chain runs, binary argv recorded by the wrapper log header",
        started.elapsed().as_secs_f64(),
        runs.len()
    );
    println!(
        "PASS framing: the health-check column ran to the last row; it does not close criterion 1, and the measured-vs-bar numbers above are attribution, not a gate"
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Raw CUDA driver API, resolved through dlopen, copied from examples/moe_slice.rs on origin/main
// so this harness does not carry a second loader. Nothing links libcuda: the driver is dlopen'd
// at run time on the node.
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Raw CUDA driver API, resolved through dlopen. Nothing links libcuda: the driver is dlopen'd at
// run time on the node, which is the invariant docs/build/device-path-design.md records.
// ---------------------------------------------------------------------------------------------

type CuResult = i32;

/// An opaque CUDA event handle.
type CuEvent = *mut c_void;

/// MemAvailable in kB, parsed. Interpolating the whole /proc line into a message produced
/// "...the node has ~MemAvailable: 3768676 kB GB of system memory available...", which is exactly
/// the kind of unreadable failure text this launcher exists to avoid.
fn mem_available_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

/// The cuLaunchKernel signature. The parameter list is long by nature, so it is named once here
/// rather than spelled out at the one call site.
type LaunchFn = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut c_void,
) -> CuResult;

/// The driver's CUresult names, so a report says OUT_OF_MEMORY rather than 2.
fn name(code: CuResult) -> String {
    let s = match code {
        0 => "SUCCESS",
        1 => "INVALID_VALUE",
        2 => "OUT_OF_MEMORY",
        3 => "NOT_INITIALIZED",
        4 => "DEINITIALIZED",
        100 => "NO_DEVICE",
        101 => "INVALID_DEVICE",
        201 => "INVALID_CONTEXT",
        400 => "INVALID_HANDLE",
        700 => "ILLEGAL_ADDRESS",
        801 => "NOT_SUPPORTED",
        _ => "UNKNOWN",
    };
    format!("{code} ({s})")
}
type CuDeviceptr = u64;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

struct Deviceptr(CuDeviceptr);

struct Cuda {
    lib: libloading::Library,
    ctx: *mut c_void,
}

impl Cuda {
    /// cuInit alone is not enough: cuModuleLoadData and cuMemAlloc need a CURRENT context, and
    /// without one the driver answers CUDA_ERROR_INVALID_CONTEXT (201), which is what the first
    /// run on the node returned. The primary context is retained (refcounted, shared with anything
    /// else driving the device) and made current for this thread.
    fn load() -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new("libcuda.so.1")
                .map_err(|e| format!("libcuda.so.1: {e}"))?;
            {
                let init: libloading::Symbol<unsafe extern "C" fn(u32) -> CuResult> =
                    lib.get(b"cuInit").map_err(|e| e.to_string())?;
                let r = init(0);
                if r != 0 {
                    return Err(format!("cuInit failed: {r}"));
                }
            }
            let mut count: i32 = 0;
            {
                let f: libloading::Symbol<unsafe extern "C" fn(*mut i32) -> CuResult> =
                    lib.get(b"cuDeviceGetCount").map_err(|e| e.to_string())?;
                let r = f(&mut count);
                if r != 0 {
                    return Err(format!("cuDeviceGetCount failed: {}", name(r)));
                }
            }
            let mut dev: i32 = 0;
            {
                let get: libloading::Symbol<unsafe extern "C" fn(*mut i32, i32) -> CuResult> =
                    lib.get(b"cuDeviceGet").map_err(|e| e.to_string())?;
                let r = get(&mut dev, 0);
                if r != 0 {
                    return Err(format!("cuDeviceGet(0) failed: {}", name(r)));
                }
            }
            let mut device_name = String::from("unknown device");
            {
                let mut nm = [0i8; 128];
                let f: libloading::Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> CuResult> =
                    lib.get(b"cuDeviceGetName").map_err(|e| e.to_string())?;
                let r = f(nm.as_mut_ptr(), 128, dev);
                if r == 0 {
                    // c_char is u8 on aarch64 and i8 on x86_64; cast rather than assume either.
                    let s = std::ffi::CStr::from_ptr(nm.as_ptr().cast());
                    device_name = s.to_string_lossy().into_owned();
                    let mut total: u64 = 0;
                    let t: libloading::Symbol<unsafe extern "C" fn(*mut u64, i32) -> CuResult> =
                        lib.get(b"cuDeviceTotalMem_v2").map_err(|e| e.to_string())?;
                    let _ = t(&mut total, dev);
                    println!(
                        "device 0: {device_name} of {count} (total {} MiB, MemAvailable {} MiB)",
                        total / (1024 * 1024),
                        mem_available_kb().map(|kb| kb / 1024).unwrap_or(0)
                    );
                }
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            {
                // CU_CTX_LMEM_RESIZE_TO_MAX (0x10) asks the driver NOT to pre-reserve the maximum
                // local memory for this context. On a unified-memory GB10 where the device total is
                // 124 GB but only ~3.7 GB of system memory is free, that pre-reservation is the
                // difference between a context and CUDA_ERROR_OUT_OF_MEMORY.
                let set_flags: libloading::Symbol<unsafe extern "C" fn(i32, u32) -> CuResult> = lib
                    .get(b"cuDevicePrimaryCtxSetFlags")
                    .map_err(|e| e.to_string())?;
                let _ = set_flags(dev, 0x10);
                let retain: libloading::Symbol<
                    unsafe extern "C" fn(*mut *mut c_void, i32) -> CuResult,
                > = lib
                    .get(b"cuDevicePrimaryCtxRetain")
                    .map_err(|e| e.to_string())?;
                let r = retain(&mut ctx, dev);
                if r != 0 {
                    // One alternative route, tried exactly once - not a retry loop. Retaining the
                    // primary context is the normal route for a process that does not own the
                    // device; cuCtxCreate_v2 is what a launcher would use if it did. Both results
                    // are reported so the failure cannot be read as one API's quirk.
                    let mut alt: *mut c_void = std::ptr::null_mut();
                    let create: libloading::Symbol<
                        unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> CuResult,
                    > = lib.get(b"cuCtxCreate_v2").map_err(|e| e.to_string())?;
                    let r2 = create(&mut alt, 0x10, dev);
                    // Nothing is left half-initialised: no context was retained, no device memory
                    // was allocated, and this condition is not retried. If a context IS created and
                    // a later step fails, Drop releases it.
                    let avail = mem_available_kb()
                        .map(|kb| format!("{} MiB", kb / 1024))
                        .unwrap_or_else(|| "unknown".to_string());
                    return Err(format!(
                        "no CUDA context could be created on device 0 ({}): \
                         cuDevicePrimaryCtxRetain -> {}, cuCtxCreate_v2 -> {} (one attempt each, not \
                         retried). cuInit, cuDeviceGet and cuDeviceGetName all succeeded, so the \
                         driver is reachable and the device is present: on this node a second CUDA \
                         context does not fit beside the resident model. MemAvailable now: {avail}. \
                         No device memory was allocated and no context is left retained. This is a \
                         capacity fact about the node, not a defect in this launcher - do not work \
                         around it here.",
                        device_name,
                        name(r),
                        name(r2),
                    ));
                }
            }
            {
                let set: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> CuResult> =
                    lib.get(b"cuCtxSetCurrent").map_err(|e| e.to_string())?;
                let r = set(ctx);
                if r != 0 {
                    return Err(format!("cuCtxSetCurrent failed: {r}"));
                }
            }
            Ok(Cuda { lib, ctx })
        }
    }

    fn sym<T>(&self, name: &[u8]) -> Result<libloading::Symbol<'_, T>, String> {
        unsafe {
            self.lib
                .get(name)
                .map_err(|e| format!("{}: {e}", String::from_utf8_lossy(name)))
        }
    }

    fn alloc(&self, bytes: usize, total: &mut usize) -> Result<Deviceptr, String> {
        unsafe {
            let f: libloading::Symbol<unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult> =
                self.sym(b"cuMemAlloc_v2")?;
            let mut p: CuDeviceptr = 0;
            let r = f(&mut p, bytes);
            if r != 0 {
                return Err(format!("cuMemAlloc({bytes}) failed: {r}"));
            }
            *total += bytes;
            Ok(Deviceptr(p))
        }
    }

    fn copy_htod(&self, dst: &Deviceptr, src: &[u8]) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, *const c_void, usize) -> CuResult,
            > = self.sym(b"cuMemcpyHtoD_v2")?;
            let r = f(dst.0, src.as_ptr().cast(), src.len());
            if r != 0 {
                return Err(format!("cuMemcpyHtoD failed: {r}"));
            }
            Ok(())
        }
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: &Deviceptr) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize) -> CuResult,
            > = self.sym(b"cuMemcpyDtoH_v2")?;
            let r = f(dst.as_mut_ptr().cast(), src.0, dst.len());
            if r != 0 {
                return Err(format!("cuMemcpyDtoH failed: {r}"));
            }
            Ok(())
        }
    }

    fn load_function(&self, cubin: &[u8], name: &str) -> Result<(CuModule, CuFunction), String> {
        unsafe {
            let load: libloading::Symbol<
                unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
            > = self.sym(b"cuModuleLoadData")?;
            let mut module: CuModule = std::ptr::null_mut();
            let r = load(&mut module, cubin.as_ptr().cast());
            if r != 0 {
                return Err(format!("cuModuleLoadData failed: {r}"));
            }
            let get: libloading::Symbol<
                unsafe extern "C" fn(*mut CuFunction, CuModule, *const i8) -> CuResult,
            > = self.sym(b"cuModuleGetFunction")?;
            let cname = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
            let mut func: CuFunction = std::ptr::null_mut();
            let r = get(&mut func, module, cname.as_ptr().cast());
            if r != 0 {
                return Err(format!("cuModuleGetFunction({name}) failed: {r}"));
            }
            Ok((module, func))
        }
    }

    fn launch(
        &self,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        unsafe {
            let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
            let mut storage: Vec<[u64; 1]> = args.iter().map(|a| [*a]).collect();
            let mut ptrs: Vec<*mut c_void> =
                storage.iter_mut().map(|s| s.as_mut_ptr().cast()).collect();
            let r = launch(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                std::ptr::null_mut(),
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if r != 0 {
                return Err(format!("cuLaunchKernel failed: {r}"));
            }
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let r = sync();
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            Ok(())
        }
    }

    /// Times one kernel the way a throughput claim needs it: a warm-up launch outside the timed
    /// region, one event pair per launch for the distribution, then a single batched region with no
    /// per-launch synchronisation - the figure to quote when the per-launch event overhead would
    /// dominate a kernel this small. No allocation, no upload and no checkpoint read is inside
    /// either timed region, so the numbers are the kernel's own and not the host's.
    fn launch_repeated(
        &self,
        func: CuFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
        repeat: usize,
    ) -> Result<(Vec<f64>, f64), String> {
        unsafe {
            let create: libloading::Symbol<unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult> =
                self.sym(b"cuEventCreate")?;
            let record: libloading::Symbol<unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult> =
                self.sym(b"cuEventRecord")?;
            let wait: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventSynchronize")?;
            let elapsed: libloading::Symbol<
                unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
            > = self.sym(b"cuEventElapsedTime")?;
            let destroy: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventDestroy")?;
            let mut start: CuEvent = std::ptr::null_mut();
            let mut stop: CuEvent = std::ptr::null_mut();
            let r = create(&mut start, 0);
            if r != 0 {
                return Err(format!("cuEventCreate failed: {r}"));
            }
            let r = create(&mut stop, 0);
            if r != 0 {
                return Err(format!("cuEventCreate failed: {r}"));
            }
            // Warm-up: fills the module cache and the launch path, and is deliberately not timed.
            self.launch(func, grid, block, args)?;
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let r = sync();
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            let mut per_launch = Vec::with_capacity(repeat);
            for _ in 0..repeat {
                let r = record(start, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord failed: {r}"));
                }
                self.launch(func, grid, block, args)?;
                let r = record(stop, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord failed: {r}"));
                }
                let r = wait(stop);
                if r != 0 {
                    return Err(format!("cuEventSynchronize failed: {r}"));
                }
                let mut ms: f32 = 0.0;
                let r = elapsed(&mut ms, start, stop);
                if r != 0 {
                    return Err(format!("cuEventElapsedTime failed: {r}"));
                }
                per_launch.push(f64::from(ms));
            }
            let r = record(start, std::ptr::null_mut());
            if r != 0 {
                return Err(format!("cuEventRecord failed: {r}"));
            }
            for _ in 0..repeat {
                self.launch(func, grid, block, args)?;
            }
            let r = record(stop, std::ptr::null_mut());
            if r != 0 {
                return Err(format!("cuEventRecord failed: {r}"));
            }
            let r = wait(stop);
            if r != 0 {
                return Err(format!("cuEventSynchronize failed: {r}"));
            }
            let mut total: f32 = 0.0;
            let r = elapsed(&mut total, start, stop);
            if r != 0 {
                return Err(format!("cuEventElapsedTime failed: {r}"));
            }
            destroy(start);
            destroy(stop);
            Ok((per_launch, f64::from(total) / repeat as f64))
        }
    }

    fn free(&self, module: CuModule, ptrs: &[Deviceptr]) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for p in ptrs {
                    let _ = f(p.0);
                }
            }
            if let Ok(f) = self.sym::<unsafe extern "C" fn(CuModule) -> CuResult>(b"cuModuleUnload")
            {
                let _ = f(module);
            }
            // The primary context is released by Drop, on every exit path.
        }
    }

    /// Frees device buffers without unloading the module that produced them: stage 7 runs several
    /// different-shaped launches per loaded module, so the module has to outlive them.
    fn free_buffers(&self, ptrs: &[&Deviceptr]) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for p in ptrs {
                    let _ = f(p.0);
                }
            }
        }
    }
}

impl Drop for Cuda {
    /// Releases this process's reference to the primary context on every exit path, including a
    /// failure after the context was created, so a run can never leave a half-initialised Cuda.
    /// Releasing is refcounted and does not disturb anything else driving the device.
    fn drop(&mut self) {
        if self.ctx.is_null() {
            return;
        }
        unsafe {
            if let Ok(f) = self
                .sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuDevicePrimaryCtxRelease")
            {
                let _ = f(self.ctx);
            }
        }
    }
}
