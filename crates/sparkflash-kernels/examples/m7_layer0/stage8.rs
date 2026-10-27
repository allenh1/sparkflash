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

use super::*;
use crate::stage3::*;

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

pub(crate) fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
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
    let path = crate::cubin_path(&args.cubin_dir, kernel)?;
    let cubin = std::fs::read(&path).map_err(|e| {
        format!(
            "{}: {e} (build the B1 cubins first, see the module docs)",
            path.display()
        )
    })?;
    cuda.load_kernel(&cubin, kernel_id(kernel)?)
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
    cuda.free_buffers(&[&gate_d, &up_d, &codes_d, &scales_d]);
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
pub(crate) fn stage8(args: &Args) -> Result<(), String> {
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
