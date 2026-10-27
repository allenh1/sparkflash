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

//! m21: the D37 BF16-activation projection pair on the GPU, device output against the pure-Rust
//! twin on the SAME packed weight bytes - bit for bit.
//!
//! WHEN RUN IN A LEAD-OWNED WINDOW (b69), THIS EXAMPLE IS THE EVIDENCE THAT THE TWO NEW KERNELS
//! (kernels/b12x_bf16a_mxfp8w_dense_sm121.cu and kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu) ARE THE
//! ARITHMETIC THEIR TWIN DEFINES. At this commit it has NOT been run on a GPU: the device-vs-twin
//! comparison is PENDING window b69, and this file records no device number. It is not the served
//! forward pass, not engine integration (the kernels are deliberately UNWIRED so REQUIRED_KERNELS
//! stays unchanged), not a token/chain claim and not a timing claim.
//!
//! WHY THIS HARNESS EXISTS. D37 measured that MXFP8 ACTIVATION quantisation in the attention
//! projections drifts the router input: the first expert-set flip is layer 3, 36/40 layers differ
//! at position 0, and the criterion run produces 0 of 9 tokens. The precision-first path replaces
//! the activation operand of the five projection GEMMs with BF16 while keeping the checkpoint's
//! FP8 weight codes and ue8m0 scales. The arithmetic contract is
//! `bf16(sum_k bf16_a[k] * dequant(w[c,k]))` with ONE f32 accumulation in the existing kernel's K
//! order and ONE bf16 rounding at the end, and src/bf16a_mxfp8w_ref.rs is its definition. This
//! binary launches the kernels on a GPU and compares every output code with that twin.
//!
//! COMPARISON CLASS AND BAR, stated before this program prints a number of its own:
//!
//! 1. CLASS: exact arithmetic, not a tolerance. The kernels are the scalar reference path whose
//!    accumulation order the twin implements; a difference of ONE BF16 code is a finding, not
//!    noise. BAR: 0 mismatched BF16 codes of rows x cols per dense projection and of
//!    groups x o_lora_rank per wo_a position, maximum |device - twin| in BF16 value units = 0.0.
//! 2. DETERMINISM: a second launch on unchanged operands must return byte-identical output. BAR:
//!    byte equality.
//! 3. PINS: each cubin is refused unless its size AND sha256 equal the pinned compiled form in
//!    baseline/kernel_baseline.txt (read through baseline::pin) before any device allocation.
//! 4. OPERANDS: the weights are the released layer-0 checkpoint bytes, read verbatim through the
//!    measured m7_layer0_shapes table and repacked with pack_b12x (a declared, value-preserving
//!    index permutation - no arithmetic); scales are uploaded as stored in the checkpoint's
//!    [out/32, in/32] grid. The activation operands are the released embed.weight rows (BF16), a
//!    declared stand-in for the attention-boundary hidden exactly as m12/m18 declare theirs:
//!    wq_a/wkv take the whole 5120-wide row, wq_b takes its first 1280 BF16 codes (a declared
//!    stand-in for the q latent, NOT the device K2->K3 seam), and wo_a takes the row's first 512
//!    BF16 codes repeated 8 times to fill each group's 4096-wide reduction (a declared stand-in
//!    for the attention output heads, NOT the K4/K5 seam). The same operand bytes go to the
//!    kernel and the twin, so the seam is declared and not claimed as part of the arithmetic.
//!
//! WHAT IT MEASURES WHEN RUN: device-vs-twin equality on those operands at the released geometry,
//! the determinism bar, and the cubin pins. At this commit the pins are measured (compiled by nvcc
//! V13.0.88 and recorded in baseline/kernel_baseline.txt) and the host tests are green, but no
//! device number exists yet: GPU comparison PENDING window b69. WHAT IT DOES NOT MEASURE: released
//! activation values (the checkpoint stores none for this stage), the router-input drift that D37
//! attributes to the quantiser, any token or criterion run, latency/bandwidth, or the wiring that
//! would replace the MXFP8 calls in the device chain.
//!
//! PINS.
//!   b12x_bf16a_mxfp8w_dense_sm121  11,736 B  sha256
//!     a2e0a63c8b78e3fe9117d7c12b06353bbc46af7c16019cf71e572c93806eff7e  0 PTX membar lines
//!   b12x_bf16a_mxfp8w_wo_a_sm121    8,976 B  sha256
//!     d11a8610ff557617116c437b0bc7a0367f66a8069ccfc97373433ebd99ffd249  0 PTX membar lines
//!
//! BUILD THE CUBINS (nvcc 13.0.88 / CUDA 13.0.3, on the node):
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_bf16a_mxfp8w_dense_sm121.cu \
//!       -o <dir>/b12x_bf16a_mxfp8w_dense_sm121.cubin
//!   nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/b12x_bf16a_mxfp8w_wo_a_sm121.cu \
//!       -o <dir>/b12x_bf16a_mxfp8w_wo_a_sm121.cubin
//!
//! BINARY (cross-compiled on the dev box, copied to the node):
//!   ./scripts/build-aarch64.sh --example=m21_bf16a_projections
//!
//! RUN (Lead-owned window, node):
//!   m21_bf16a_projections --checkpoint <snapshot> --cubin-dir <dir> --tokens 8
//!   m21_bf16a_projections --checkpoint <snapshot> --cubin-dir <dir> --tokens 130

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[path = "../tests/support/b2_harness.rs"]
mod b2_harness;

use self::b2_harness::Snapshot;
use sparkflash_kernels::baseline;
use sparkflash_kernels::bf16a_mxfp8w_launch::{self as dense_launch, Bf16aMxfp8wDenseParams};
use sparkflash_kernels::bf16a_mxfp8w_ref::{dense_gemm_bf16a, grouped_wo_a_bf16a};
use sparkflash_kernels::bf16a_mxfp8w_wo_a_launch::{self as woa_launch, Bf16aMxfp8wWoAParams};
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::m4_quant::bf16_value;
use sparkflash_kernels::m7_layer0_shapes::LAYER0_ATTENTION_WEIGHTS;
use sparkflash_kernels::mxfp8_ref::{b12x_buffer_bytes, b12x_scale_bytes};

/// The dense BF16-activation cubin name and pin (baseline/kernel_baseline.txt).
const DENSE_KERNEL: &str = "b12x_bf16a_mxfp8w_dense_sm121";
const DENSE_PIN_BYTES: usize = 27_824;
const DENSE_PIN_SHA256: &str = "a2e0a63c8b78e3fe9117d7c12b06353bbc46af7c16019cf71e572c93806eff7e";
/// The grouped wo_a BF16-activation cubin name and pin.
const WOA_KERNEL: &str = "b12x_bf16a_mxfp8w_wo_a_sm121";
const WOA_PIN_BYTES: usize = 11_568;
const WOA_PIN_SHA256: &str = "d11a8610ff557617116c437b0bc7a0367f66a8069ccfc97373433ebd99ffd249";
/// Both pinned cubins carry no fence: the scalar projections synchronise only within a block.
const PIN_MEMBAR_LINES: u32 = 0;
/// The released hidden width, the layer-0 attention-boundary row width.
const HIDDEN: usize = 5_120;
/// wo_a's released geometry: 8 groups of 1024 output channels over 4096 reduction values.
const O_GROUPS: usize = 8;
const O_LORA_RANK: usize = 1_024;
const GROUP_INNER: usize = 4_096;
/// The grouped output width: 8 groups x 1024 channels.
const WOA_OUT: usize = O_GROUPS * O_LORA_RANK;
/// The attention latent chunk one group repeats: 512 BF16 codes from the row.
const LATENT: usize = 512;
/// The three dense projection shapes this harness must exercise, as (name, out channels, reduction
/// width). These are the chain's K2 wq_a, K3 wq_b and K4 wkv shapes from the measured layer-0
/// table; wo_a is exercised through the grouped kernel's (groups, o_lora_rank, group_inner) below.
const DENSE_SHAPES: [(&str, usize, usize); 3] = [
    ("wq_a", 1_280, 5_120),
    ("wq_b", 32_768, 1_280),
    ("wkv", 512, 5_120),
];

struct Verdict {
    name: String,
    ok: bool,
    detail: String,
}

impl Verdict {
    fn push(verdicts: &mut Vec<Self>, name: &str, ok: bool, detail: String) {
        println!(
            "  check {name}: {} ({detail})",
            if ok { "ok" } else { "FAIL" }
        );
        verdicts.push(Self {
            name: name.to_owned(),
            ok,
            detail,
        });
    }
}

struct Args {
    checkpoint: PathBuf,
    cubin_dir: PathBuf,
    tokens: usize,
    floor_gib: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        Self::parse_from(std::env::args().skip(1).collect())
    }

    fn parse_from(values: Vec<String>) -> Result<Self, String> {
        let mut args = Self {
            checkpoint: PathBuf::new(),
            cubin_dir: PathBuf::from("out"),
            tokens: 8,
            floor_gib: 32,
        };
        let mut values = values;
        while let Some(flag) = values.first().cloned() {
            values.remove(0);
            let mut next = || {
                values
                    .first()
                    .cloned()
                    .inspect(|_| {
                        values.remove(0);
                    })
                    .ok_or_else(|| format!("{flag} needs a value"))
            };
            match flag.as_str() {
                "--checkpoint" => args.checkpoint = PathBuf::from(next()?),
                "--cubin-dir" => args.cubin_dir = PathBuf::from(next()?),
                "--tokens" => {
                    args.tokens = next()?
                        .parse()
                        .map_err(|error| format!("--tokens: {error}"))?;
                }
                "--floor-gib" => {
                    args.floor_gib = next()?
                        .parse()
                        .map_err(|error| format!("--floor-gib: {error}"))?;
                }
                "--help" | "-h" => {
                    return Err("usage: m21_bf16a_projections --checkpoint <snapshot dir> \
                         [--cubin-dir out] [--tokens 8] [--floor-gib 32]"
                        .to_owned());
                }
                other => return Err(format!("unknown argument {other:?}; try --help")),
            }
        }
        if args.checkpoint.as_os_str().is_empty() {
            return Err("--checkpoint is required (the released snapshot directory)".to_owned());
        }
        if args.tokens == 0 || args.tokens > 1024 {
            return Err("--tokens must be in 1..=1024".to_owned());
        }
        if u32::try_from(args.tokens).is_err() {
            return Err("--tokens must fit the kernel's u32 rows parameter".to_owned());
        }
        Ok(args)
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("m21_bf16a_projections: {error}");
        std::process::exit(1);
    }
}

/// One dense projection's released spec plus its prepared B12x bytes.
struct DenseProjection {
    name: &'static str,
    rows: usize,
    inner: usize,
    packed: Vec<u8>,
    scales: Vec<u8>,
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    println!(
        "=== m21: D37 BF16-activation projection pair (b12x_bf16a_mxfp8w_{{dense,wo_a}}_sm121) ==="
    );
    println!(
        "geometry: --tokens {} output rows per dense projection; wo_a {} groups x {} channels over \
         {} reduction values per position, one launch per position",
        args.tokens, O_GROUPS, O_LORA_RANK, GROUP_INNER
    );
    println!(
        "CLASS: exact arithmetic (0 mismatched BF16 codes), not a tolerance. BAR, declared before any \
         number below: every dense output code and every wo_a output code equals the twin's code on \
         the same packed weight bytes and the same BF16 activation bytes (max_abs reported in BF16 \
         value units = 0.0), and a second launch is byte-identical. A difference of one code is a \
         finding."
    );
    println!(
        "reference: sparkflash_kernels::bf16a_mxfp8w_ref::dense_gemm_bf16a / grouped_wo_a_bf16a - \
         one f32 accumulator over K in ascending 32-value blocks, one bf16 round at the end"
    );
    println!(
        "operands: released layer-0 attn weights verbatim (F8_E4M3, pack_b12x value-preserving) and \
         scales as stored; activations are released embed.weight BF16 rows, a declared stand-in \
         (wq_b: first 1280 codes; wo_a: first 512 codes repeated 8x per group - declared seams, not \
         the K2->K3 / K4->K5 device chain)"
    );
    println!(
        "what this is not: not the served forward pass, not the router/criterion measurement, not \
         engine wiring, not latency. The kernels are UNWIRED so REQUIRED_KERNELS is unchanged."
    );
    check_floor(args.floor_gib, "m21 start")?;

    let mut verdicts: Vec<Verdict> = Vec::new();
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // Pins first: no device allocation happens before both compiled forms are checked.
    let dense_cubin = pinned_cubin(
        &args.cubin_dir,
        DENSE_KERNEL,
        DENSE_PIN_BYTES,
        DENSE_PIN_SHA256,
    )?;
    let woa_cubin = pinned_cubin(&args.cubin_dir, WOA_KERNEL, WOA_PIN_BYTES, WOA_PIN_SHA256)?;
    check_pin_matches_baseline(DENSE_KERNEL, DENSE_PIN_BYTES, DENSE_PIN_SHA256)?;
    check_pin_matches_baseline(WOA_KERNEL, WOA_PIN_BYTES, WOA_PIN_SHA256)?;

    // The released weights and scales: read verbatim through the measured table.
    let mut dense_projections: Vec<DenseProjection> = Vec::new();
    for name in ["wq_a", "wq_b", "wkv"] {
        let spec = *LAYER0_ATTENTION_WEIGHTS
            .iter()
            .find(|spec| spec.name == name)
            .ok_or_else(|| format!("{name} is not in the measured layer-0 table"))?;
        let weight_name = format!("layers.0.attn.{name}.weight");
        let scale_name = format!("layers.0.attn.{name}.scale");
        let raw = snap.read_raw(&weight_name, "F8_E4M3", &[spec.rows, spec.cols])?;
        let scales = snap.read_raw(&scale_name, "F8_E8M0", &[spec.scale_rows, spec.scale_cols])?;
        let packed = pack_b12x(&raw, spec.rows, spec.cols);
        if packed.len() != b12x_buffer_bytes(spec.rows, spec.cols) {
            return Err(format!(
                "{name}: packed {} bytes, the B12x rule needs {}",
                packed.len(),
                b12x_buffer_bytes(spec.rows, spec.cols)
            ));
        }
        if scales.len() != b12x_scale_bytes(spec.rows, spec.cols) {
            return Err(format!(
                "{name}: stored scale grid is {} bytes, the kernel reads {}",
                scales.len(),
                b12x_scale_bytes(spec.rows, spec.cols)
            ));
        }
        println!(
            "released {name}: F8_E4M3 [{} x {}] -> B12x {} B (value-preserving pack_b12x); scale grid \
             [{} x {}] = {} B uploaded as stored",
            spec.rows,
            spec.cols,
            packed.len(),
            spec.scale_rows,
            spec.scale_cols,
            scales.len()
        );
        dense_projections.push(DenseProjection {
            name: spec.name,
            rows: spec.rows,
            inner: spec.cols,
            packed,
            scales,
        });
    }
    // The shape contract, checked against literals rather than only inherited from the table: a
    // table edit must not silently change what this harness claims to exercise.
    let shapes_ok = dense_projections.len() == DENSE_SHAPES.len()
        && dense_projections
            .iter()
            .zip(DENSE_SHAPES)
            .all(|(projection, (name, rows, inner))| {
                projection.name == name && projection.rows == rows && projection.inner == inner
            });
    Verdict::push(
        &mut verdicts,
        "dense shape contract (out x reduction), released widths",
        shapes_ok,
        format!(
            "wq_a 1280x5120, wq_b 32768x1280, wkv 512x5120; got {}",
            dense_projections
                .iter()
                .map(|projection| format!(
                    "{} {}x{}",
                    projection.name, projection.rows, projection.inner
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    );
    if !shapes_ok {
        return Err("the released dense shapes are not the chain's projection shapes".to_owned());
    }
    println!(
        "shape contract: dense wq_a 1280x5120, wq_b 32768x1280, wkv 512x5120 (out x reduction); \
         wo_a grouped {O_GROUPS} groups x ({O_LORA_RANK} out x {GROUP_INNER} reduction) from released \
         [{WOA_OUT} x {GROUP_INNER}]"
    );
    let woa_spec = *LAYER0_ATTENTION_WEIGHTS
        .iter()
        .find(|spec| spec.name == "wo_a")
        .ok_or_else(|| "wo_a is not in the measured layer-0 table".to_owned())?;
    if (woa_spec.rows, woa_spec.cols) != (O_GROUPS * O_LORA_RANK, GROUP_INNER) {
        return Err(format!(
            "wo_a: measured [{} x {}] is not the grouped [{} x {GROUP_INNER}] geometry",
            woa_spec.rows,
            woa_spec.cols,
            O_GROUPS * O_LORA_RANK
        ));
    }
    let woa_weight_name = "layers.0.attn.wo_a.weight".to_owned();
    let woa_scale_name = "layers.0.attn.wo_a.scale".to_owned();
    let woa_raw = snap.read_raw(&woa_weight_name, "F8_E4M3", &[woa_spec.rows, woa_spec.cols])?;
    let woa_scales = snap.read_raw(
        &woa_scale_name,
        "F8_E8M0",
        &[woa_spec.scale_rows, woa_spec.scale_cols],
    )?;
    let (woa_packed, woa_scales) = pack_wo_a(&woa_raw, &woa_scales)?;
    println!(
        "released wo_a: F8_E4M3 [{} x {}] -> per-group B12x {} B; scale grid [{} x {}] = {} B as \
         stored",
        woa_spec.rows,
        woa_spec.cols,
        woa_packed.len(),
        woa_spec.scale_rows,
        woa_spec.scale_cols,
        woa_scales.len()
    );
    Verdict::push(
        &mut verdicts,
        "released weight and scale byte counts are the B12x rules'",
        dense_projections.iter().all(|projection| {
            projection.packed.len() == b12x_buffer_bytes(projection.rows, projection.inner)
                && projection.scales.len() == b12x_scale_bytes(projection.rows, projection.inner)
        }) && woa_packed.len() == O_GROUPS * b12x_buffer_bytes(O_LORA_RANK, GROUP_INNER)
            && woa_scales.len() == O_GROUPS * b12x_scale_bytes(O_LORA_RANK, GROUP_INNER),
        format!(
            "{} dense projections and {} wo_a groups checked against ceil(out/32) * inner/32",
            dense_projections.len(),
            O_GROUPS
        ),
    );

    // The released activation stand-in: embed.weight rows, raw BF16.
    let mut embed_rows: Vec<Vec<u16>> = Vec::with_capacity(args.tokens);
    for token in 0..args.tokens {
        let mut row = vec![0u8; HIDDEN * 2];
        snap.read_row_bf16_into("embed.weight", token, HIDDEN, &mut row)?;
        embed_rows.push(
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_le_bytes(*pair))
                .collect(),
        );
    }
    println!(
        "released activation stand-in: embed.weight BF16 rows 0..{} ({} codes per row)",
        args.tokens, HIDDEN
    );
    println!(
        "expected exact codes at --tokens {}: wq_a {}, wq_b {}, wkv {}, wo_a {} per position x {} \
         positions (dense compared twice for the determinism bar)",
        args.tokens,
        args.tokens * 1_280,
        args.tokens * 32_768,
        args.tokens * 512,
        O_GROUPS * O_LORA_RANK,
        args.tokens
    );

    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let (dense_module, dense_kernel) = dense_launch::load(&cuda, &dense_cubin)?;
    let (woa_module, woa_kernel) = woa_launch::load(&cuda, &woa_cubin)?;
    let mut keep: Vec<Deviceptr> = Vec::new();

    for projection in &dense_projections {
        // The operand is the released row itself for a 5120-wide projection and its first `inner`
        // codes for wq_b; the declared rule is in the header.
        let mut operand: Vec<u16> = Vec::with_capacity(args.tokens * projection.inner);
        for row in &embed_rows {
            operand.extend_from_slice(&row[..projection.inner]);
        }
        let twin = dense_gemm_bf16a(
            &operand,
            &projection.packed,
            &projection.scales,
            args.tokens,
            projection.rows,
            projection.inner,
        );
        let weight_bytes = projection.packed.len();
        let scale_bytes = projection.scales.len();
        let a_d = cuda.alloc(operand.len() * 2, &mut allocated)?;
        let w_d = cuda.alloc(weight_bytes, &mut allocated)?;
        let s_d = cuda.alloc(scale_bytes, &mut allocated)?;
        let out_d = cuda.alloc(args.tokens * projection.rows * 2, &mut allocated)?;
        cuda.copy_htod(&a_d, &bf16_bytes(&operand))?;
        cuda.copy_htod(&w_d, &projection.packed)?;
        cuda.copy_htod(&s_d, &projection.scales)?;
        let params = Bf16aMxfp8wDenseParams::new(
            a_d,
            w_d,
            s_d,
            out_d,
            args.tokens as u32,
            projection.rows as u32,
            projection.inner as u32,
            0,
            0,
            0,
        );
        dense_launch::launch(&cuda, dense_kernel, &params)?;
        let first = read_u16(&cuda, &out_d, args.tokens * projection.rows)?;
        let (exact, total, max_abs) = compare(&first, &twin);
        Verdict::push(
            &mut verdicts,
            &format!("{}: device BF16 is the twin's codes", projection.name),
            exact == total,
            format!(
                "{exact}/{total} exact, max_abs {max_abs:e} ({} rows x {} cols x {} reduction, released weights, declared activation stand-in)",
                args.tokens, projection.rows, projection.inner
            ),
        );
        dense_launch::launch(&cuda, dense_kernel, &params)?;
        let second = read_u16(&cuda, &out_d, args.tokens * projection.rows)?;
        Verdict::push(
            &mut verdicts,
            &format!("{}: two launches byte-identical", projection.name),
            first == second,
            format!("{} BF16 codes compared", first.len()),
        );
        keep.extend([a_d, w_d, s_d, out_d]);
    }

    // wo_a: one position per launch, the kernel's own geometry. The declared operand is the row's
    // first 512 BF16 codes repeated 8 times inside each group (the head-slice stand-in).
    let woa_weight_d = cuda.alloc(woa_packed.len(), &mut allocated)?;
    let woa_scale_d = cuda.alloc(woa_scales.len(), &mut allocated)?;
    cuda.copy_htod(&woa_weight_d, &woa_packed)?;
    cuda.copy_htod(&woa_scale_d, &woa_scales)?;
    keep.extend([woa_weight_d, woa_scale_d]);
    for (position, row) in embed_rows.iter().enumerate() {
        let latent = &row[..LATENT];
        let mut operand: Vec<u16> = Vec::with_capacity(O_GROUPS * GROUP_INNER);
        for _group in 0..O_GROUPS {
            for _head in 0..(GROUP_INNER / LATENT) {
                operand.extend_from_slice(latent);
            }
        }
        let twin = grouped_wo_a_bf16a(
            &operand,
            &woa_packed,
            &woa_scales,
            O_GROUPS,
            GROUP_INNER,
            O_LORA_RANK,
        );
        let a_d = cuda.alloc(operand.len() * 2, &mut allocated)?;
        let out_d = cuda.alloc(O_GROUPS * O_LORA_RANK * 2, &mut allocated)?;
        cuda.copy_htod(&a_d, &bf16_bytes(&operand))?;
        let params = Bf16aMxfp8wWoAParams::new(
            a_d,
            woa_weight_d,
            woa_scale_d,
            out_d,
            O_GROUPS as u32,
            0,
            0,
            0,
        );
        woa_launch::launch(&cuda, woa_kernel, &params)?;
        let first = read_u16(&cuda, &out_d, O_GROUPS * O_LORA_RANK)?;
        let (exact, total, max_abs) = compare(&first, &twin);
        Verdict::push(
            &mut verdicts,
            &format!("wo_a position {position}: device BF16 is the twin's codes"),
            exact == total,
            format!(
                "{exact}/{total} exact, max_abs {max_abs:e} ({} groups x {} channels, released weights, declared activation stand-in)",
                O_GROUPS, O_LORA_RANK
            ),
        );
        if position == 0 {
            woa_launch::launch(&cuda, woa_kernel, &params)?;
            let second = read_u16(&cuda, &out_d, O_GROUPS * O_LORA_RANK)?;
            Verdict::push(
                &mut verdicts,
                "wo_a: two launches byte-identical",
                first == second,
                format!("{} BF16 codes compared", first.len()),
            );
        }
        keep.extend([a_d, out_d]);
    }

    cuda.free(dense_module, &[]);
    cuda.free(woa_module, &keep);
    println!("device bytes admitted: {allocated}");
    check_floor(args.floor_gib, "m21 end")?;

    println!("--- NOT claimed by m21 ---");
    println!(
        "not the served forward pass, not engine wiring (the kernels stay UNWIRED; REQUIRED_KERNELS is \
         unchanged), not the router-input drift or a token/criterion run, not latency or bandwidth. \
         The released weights are real; the activation operands are declared stand-ins at the real \
         widths. The measured class is exact arithmetic: device BF16 codes equal the twin's codes on \
         the same packed bytes."
    );

    let failed: Vec<&str> = verdicts
        .iter()
        .filter(|verdict| !verdict.ok)
        .map(|verdict| verdict.name.as_str())
        .collect();
    if failed.is_empty() {
        println!(
            "m21: PASS ({} checks; tokens {}; two pinned cubins)",
            verdicts.len(),
            args.tokens
        );
        Ok(())
    } else {
        let details: Vec<String> = verdicts
            .iter()
            .filter(|verdict| !verdict.ok)
            .map(|verdict| format!("{}: {}", verdict.name, verdict.detail))
            .collect();
        Err(format!(
            "m21: FAIL at {} of {} checks: {:?}",
            failed.len(),
            verdicts.len(),
            details
        ))
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------------------------

/// Pack a checkpoint row-major [cols, inner] FP8 weight into B12x micro-tiles. This is a declared,
/// value-preserving index permutation: it moves bytes, decodes nothing and computes nothing. The
/// unit test below proves every packed byte is the checkpoint byte at the B12x index.
fn pack_b12x(weights: &[u8], cols: usize, inner: usize) -> Vec<u8> {
    assert_eq!(
        weights.len(),
        cols * inner,
        "the weight buffer is [cols, inner]"
    );
    let mut packed = vec![0u8; b12x_buffer_bytes(cols, inner)];
    for n in 0..cols {
        for k in 0..inner {
            packed[sparkflash_kernels::mxfp8_ref::b12x_weight_index(n, k, inner)] =
                weights[n * inner + k];
        }
    }
    packed
}

/// The per-group repack of wo_a: each group's 1024-row slice is its own B12x buffer, concatenated
/// in group order; the stored [256, 128] scale tensor is already the concatenation of the per-group
/// [32, 128] grids, so it is returned unchanged.
fn pack_wo_a(weight: &[u8], scales: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    if weight.len() != O_GROUPS * O_LORA_RANK * GROUP_INNER {
        return Err(format!(
            "wo_a weight: {} bytes, expected {}",
            weight.len(),
            O_GROUPS * O_LORA_RANK * GROUP_INNER
        ));
    }
    let group_scale_bytes = b12x_scale_bytes(O_LORA_RANK, GROUP_INNER);
    if scales.len() != O_GROUPS * group_scale_bytes {
        return Err(format!(
            "wo_a scales: {} bytes, expected {}",
            scales.len(),
            O_GROUPS * group_scale_bytes
        ));
    }
    let group_weight_bytes = b12x_buffer_bytes(O_LORA_RANK, GROUP_INNER);
    let mut packed = Vec::with_capacity(weight.len());
    for group in 0..O_GROUPS {
        let start = group * O_LORA_RANK * GROUP_INNER;
        packed.extend_from_slice(&pack_b12x(
            &weight[start..start + O_LORA_RANK * GROUP_INNER],
            O_LORA_RANK,
            GROUP_INNER,
        ));
    }
    if packed.len() != O_GROUPS * group_weight_bytes {
        return Err("wo_a: the per-group pack disagrees with the B12x buffer rule".to_owned());
    }
    Ok((packed, scales.to_vec()))
}

/// Compares two BF16 code sequences: exact count, total, and the largest decoded difference.
fn compare(device: &[u16], twin: &[u16]) -> (usize, usize, f32) {
    assert_eq!(
        device.len(),
        twin.len(),
        "the device and twin shapes differ"
    );
    let mut exact = 0usize;
    let mut max_abs = 0.0f32;
    for (a, b) in device.iter().zip(twin) {
        if a == b {
            exact += 1;
        }
        let delta = (bf16_value(*a) - bf16_value(*b)).abs();
        if delta > max_abs {
            max_abs = delta;
        }
    }
    (exact, device.len(), max_abs)
}

fn bf16_bytes(codes: &[u16]) -> Vec<u8> {
    codes.iter().flat_map(|code| code.to_le_bytes()).collect()
}

fn read_u16(cuda: &Cuda, ptr: &Deviceptr, values: usize) -> Result<Vec<u16>, String> {
    let mut bytes = vec![0u8; values * 2];
    cuda.copy_dtoh(&mut bytes, ptr)?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect())
}

fn check_floor(floor_gib: u64, point: &str) -> Result<(), String> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| format!("cannot read /proc/meminfo: {error}"))?;
    let kb = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| "cannot parse MemAvailable".to_owned())?;
    let bytes = kb.saturating_mul(1024);
    let floor = floor_gib.saturating_mul(1024 * 1024 * 1024);
    if bytes < floor {
        return Err(format!(
            "MemAvailable at {point} is {kb} kB ({:.2} GiB), below the {floor_gib} GiB floor: refusing to run",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        ));
    }
    println!(
        "MemAvailable at {point}: {kb} kB ({:.1} GiB), floor {floor_gib} GiB - ok",
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

/// Reads one cubin and refuses it unless its bytes and sha256 are the pinned compiled form.
fn pinned_cubin(
    dir: &Path,
    kernel: &str,
    pinned_bytes: usize,
    pinned_sha: &str,
) -> Result<Vec<u8>, String> {
    let path = dir.join(format!("{kernel}.cubin"));
    let cubin = std::fs::read(&path).map_err(|error| {
        format!(
            "{}: {error} (build the cubin first: nvcc -std=c++17 -arch=sm_121a -cubin -I kernels kernels/{kernel}.cu -o {})",
            path.display(),
            path.display()
        )
    })?;
    let digest = Sha256::digest(&cubin);
    let hex = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!(
        "cubin: {} ({} B, sha256 {hex})",
        path.display(),
        cubin.len()
    );
    if cubin.len() != pinned_bytes || hex != pinned_sha {
        return Err(format!(
            "{kernel}: pinned {pinned_bytes} B / {pinned_sha}, got {} B / {hex}; refusing to measure a \
             different kernel",
            cubin.len()
        ));
    }
    Ok(cubin)
}

/// Cross-checks the harness constants against the crate's embedded baseline manifest, so the two
/// records of the pin cannot drift apart silently.
fn check_pin_matches_baseline(
    kernel: &str,
    pinned_bytes: usize,
    pinned_sha: &str,
) -> Result<(), String> {
    let pin = baseline::pin(kernel)?
        .ok_or_else(|| format!("the baseline manifest does not pin {kernel}"))?;
    if pin.cubin_bytes != pinned_bytes
        || pin.cubin_sha256 != pinned_sha
        || pin.ptx_membar_lines != PIN_MEMBAR_LINES
    {
        return Err(format!(
            "{kernel}: baseline pins {} B / {} / membar {}, the harness declares {pinned_bytes} B / \
             {pinned_sha} / membar {PIN_MEMBAR_LINES}",
            pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
        ));
    }
    println!(
        "baseline manifest agrees: {kernel} {} B, {}, {} PTX membar lines",
        pin.cubin_bytes, pin.cubin_sha256, pin.ptx_membar_lines
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_harness_pins_are_the_baselines_pins() {
        check_pin_matches_baseline(DENSE_KERNEL, DENSE_PIN_BYTES, DENSE_PIN_SHA256)
            .expect("the dense pin agrees with the manifest");
        check_pin_matches_baseline(WOA_KERNEL, WOA_PIN_BYTES, WOA_PIN_SHA256)
            .expect("the wo_a pin agrees with the manifest");
    }

    #[test]
    fn the_harness_named_kernels_are_the_launch_modules_and_the_manifest() {
        assert_eq!(dense_launch::KERNEL, DENSE_KERNEL);
        assert_eq!(woa_launch::KERNEL, WOA_KERNEL);
        let manifest = sparkflash_kernels::baseline::BASELINE_TEXT;
        assert!(
            manifest.contains(&format!(
                "kernel {DENSE_KERNEL} cubin_bytes={DENSE_PIN_BYTES} cubin_sha256={DENSE_PIN_SHA256} \
                 ptx_membar_lines={PIN_MEMBAR_LINES}"
            )),
            "the dense pin must be the manifest line"
        );
        assert!(
            manifest.contains(&format!(
                "kernel {WOA_KERNEL} cubin_bytes={WOA_PIN_BYTES} cubin_sha256={WOA_PIN_SHA256} \
                 ptx_membar_lines={PIN_MEMBAR_LINES}"
            )),
            "the wo_a pin must be the manifest line"
        );
    }

    #[test]
    fn the_released_geometry_is_the_kernels_geometry() {
        let wo_a = LAYER0_ATTENTION_WEIGHTS
            .iter()
            .find(|spec| spec.name == "wo_a")
            .expect("wo_a is measured");
        assert_eq!(wo_a.rows, O_GROUPS * O_LORA_RANK);
        assert_eq!(wo_a.cols, GROUP_INNER);
        assert_eq!(GROUP_INNER / LATENT, 8, "8 heads per group x 512");
        for name in ["wq_a", "wq_b", "wkv"] {
            let spec = LAYER0_ATTENTION_WEIGHTS
                .iter()
                .find(|spec| spec.name == name)
                .expect("the dense projection is measured");
            assert!(
                spec.cols == HIDDEN || spec.cols == 1280,
                "{name} reduction width must be 5120 or the q LoRA rank"
            );
        }
    }

    #[test]
    fn the_harness_dense_shapes_are_the_chain_projection_shapes() {
        // The literal contract the Lead cites must equal the measured table: three dense GEMMs at
        // the released out x reduction widths, and the grouped wo_a at 8192 = 8 x 1024 over 4096.
        assert_eq!(DENSE_SHAPES.len(), 3);
        for (name, rows, inner) in DENSE_SHAPES {
            let spec = LAYER0_ATTENTION_WEIGHTS
                .iter()
                .find(|spec| spec.name == name)
                .unwrap_or_else(|| panic!("{name} is measured"));
            assert_eq!(
                (spec.rows, spec.cols),
                (rows, inner),
                "{name}: the harness literal and the measured table disagree"
            );
        }
        let wo_a = LAYER0_ATTENTION_WEIGHTS
            .iter()
            .find(|spec| spec.name == "wo_a")
            .expect("wo_a is measured");
        assert_eq!((wo_a.rows, wo_a.cols), (WOA_OUT, GROUP_INNER));
        assert_eq!(WOA_OUT, O_GROUPS * O_LORA_RANK);
        assert_eq!((O_GROUPS, O_LORA_RANK, GROUP_INNER), (8, 1_024, 4_096));
        assert_eq!(GROUP_INNER, LATENT * 8, "8 head slices of 512 fill a group");
    }

    #[test]
    fn pack_b12x_is_a_value_preserving_permutation() {
        let (cols, inner) = (33usize, 96usize);
        let raw: Vec<u8> = (0..cols * inner).map(|i| (i * 37 + 11) as u8).collect();
        let packed = pack_b12x(&raw, cols, inner);
        assert_eq!(packed.len(), b12x_buffer_bytes(cols, inner));
        let mut seen = vec![false; packed.len()];
        for n in 0..cols {
            for k in 0..inner {
                let index = sparkflash_kernels::mxfp8_ref::b12x_weight_index(n, k, inner);
                assert!(!seen[index], "two elements share byte {index}");
                seen[index] = true;
                assert_eq!(
                    packed[index],
                    raw[n * inner + k],
                    "({n}, {k}) moved its value"
                );
            }
        }
        assert_eq!(
            seen.iter().filter(|hit| **hit).count(),
            cols * inner,
            "every element lands on its own byte"
        );
    }

    #[test]
    fn the_declared_activation_builders_are_deterministic() {
        // The wq_b rule: the first `inner` codes of the row. The wo_a rule: the first 512 codes
        // repeated 8 times per group. Both are pure slices/repetitions of the released row, so a
        // second build must be byte-identical and the sizes must be the kernels'.
        let row: Vec<u16> = (0..HIDDEN).map(|i| (i * 3 + 1) as u16).collect();
        let wq_b: Vec<u16> = row[..1280].to_vec();
        assert_eq!(wq_b.len(), 1280);
        assert_eq!(wq_b, row[..1280].to_vec());
        let latent = &row[..LATENT];
        let mut woa: Vec<u16> = Vec::with_capacity(O_GROUPS * GROUP_INNER);
        for _group in 0..O_GROUPS {
            for _head in 0..(GROUP_INNER / LATENT) {
                woa.extend_from_slice(latent);
            }
        }
        assert_eq!(woa.len(), O_GROUPS * GROUP_INNER);
        assert_eq!(&woa[..LATENT], latent);
        assert_eq!(&woa[LATENT..2 * LATENT], latent);
    }

    #[test]
    fn the_harness_calls_the_twin_and_the_typed_launchers() {
        // A source-level pin, in the m18 style: the harness must compare against the twin module
        // and launch through the typed param arrays. If a second formula or a raw launch appears
        // here, this test is what says so.
        let source = include_str!("m21_bf16a_projections.rs");
        let (production, _) = source
            .split_once("fn the_harness_calls_the_twin_and_the_typed_launchers")
            .expect("the source contains this test");
        assert!(production.contains("dense_gemm_bf16a"));
        assert!(production.contains("grouped_wo_a_bf16a"));
        assert!(production.contains("Bf16aMxfp8wDenseParams::new"));
        assert!(production.contains("Bf16aMxfp8wWoAParams::new"));
        assert!(production.contains("dense_launch::launch"));
        assert!(production.contains("woa_launch::launch"));
        assert!(production.contains("check_pin_matches_baseline"));
        assert!(
            !production.contains("fn dense_twin("),
            "no second GEMM formula in the harness"
        );
    }

    #[test]
    fn arguments_default_to_tokens_8_and_reject_bad_shapes() {
        let parsed = Args::parse_from(vec!["--checkpoint".to_owned(), "ckpt".to_owned()])
            .expect("checkpoint alone parses");
        assert_eq!(parsed.tokens, 8);
        assert!(Args::parse_from(vec!["--tokens".to_owned(), "8".to_owned()]).is_err());
        assert!(
            Args::parse_from(vec![
                "--checkpoint".to_owned(),
                "ckpt".to_owned(),
                "--tokens".to_owned(),
                "0".to_owned()
            ])
            .is_err()
        );
    }

    #[test]
    fn compare_reports_exact_total_and_max_abs() {
        assert_eq!(compare(&[0x3F80, 0x3F00], &[0x3F80, 0x3F00]), (2, 2, 0.0));
        let (exact, total, max_abs) = compare(&[0x3F80, 0x3F80], &[0x3F80, 0x3F00]);
        assert_eq!((exact, total), (1, 2));
        assert!(
            max_abs > 0.0,
            "a one-code difference must show a nonzero max_abs"
        );
    }
}
