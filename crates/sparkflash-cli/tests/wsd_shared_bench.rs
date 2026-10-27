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

//! WS-D Band C contingency: split the shared-expert block into launches vs host prologue.
//!
//! Ignored by default (it needs a CUDA device and a staged cubin directory). Run:
//!
//!   SPARKFLASH_KERNEL_DIR=/path/to/cubins cargo test -p sparkflash-cli --features cuda \
//!       --test wsd_shared_bench -- --ignored --nocapture
//!
//! Every number it prints is measured on this node with the same cubins the row launched.
//!
//! Two sequences are measured, on the same resident operands and the same driver calls, so one run
//! gives both variants:
//!
//! * the **coded native arm** (`--native-rounding shared`): gate, up, the wide SwiGLU + MX-quantise
//!   and down, four device launches, against the E4M3 activation row and bf16 output of the released
//!   store, plus its host prologue (bf16 round + quantise + H2D + D2H);
//! * the **f32-reference arm** (the default path): the pre-FFN f32 row uploaded unchanged, gate and
//!   up through the f32-activation/f32-store GEMV, D2H of both rows, the host SwiGLU (clamps only),
//!   H2D of the SwiGLU row, down through the same GEMV, and the 20 KB f32 output readback. Three
//!   launches and no quantise.
//!
//! The native prints are unchanged, so the 181.1 us/layer / 195.6 GB/s four-launch baseline stays
//! comparable; the f32 prints sit beside them.

#![cfg(feature = "cuda")]

use std::time::Instant;

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::mxfp8_dense_launch::Mxfp8DenseParams;
use sparkflash_kernels::mxfp8_gemv_launch::{self, Mode};

const DIM: usize = 5120;
const INTER: usize = 2304;
const BLOCK: usize = 32;
const ITERATIONS: usize = 60;

fn fill(
    cuda: &Cuda,
    bytes: usize,
    value: u8,
    total: &mut usize,
) -> sparkflash_kernels::cuda_driver::Deviceptr {
    let pointer = cuda.alloc(bytes, total).expect("alloc");
    cuda.copy_htod(&pointer, &vec![value; bytes]).expect("htod");
    pointer
}

fn f32_le(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_from_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// The engine's `sigmoid`, copied so the host SwiGLU this bench times is the arithmetic the f32
/// reference arm runs (the two-branch form is part of it).
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// The host SwiGLU of the f32 arm: clamps only, no bf16 round, over the two f32 readbacks.
fn host_swiglu(gate_bytes: &[u8], up_bytes: &[u8], out: &mut Vec<u8>) {
    let gate = f32_from_le(gate_bytes);
    let up = f32_from_le(up_bytes);
    let limit = 10.0f32;
    let mut row = Vec::with_capacity(gate.len());
    for (gate, up) in gate.iter().zip(up.iter()) {
        let clamped_gate = gate.min(limit);
        let clamped_up = up.clamp(-limit, limit);
        row.push(silu(clamped_gate) * clamped_up);
    }
    *out = f32_le(&row);
}

#[test]
#[ignore = "needs a CUDA device and SPARKFLASH_KERNEL_DIR"]
fn the_shared_expert_block_split_into_launches_and_host_prologue() {
    let dir = std::env::var("SPARKFLASH_KERNEL_DIR").expect("SPARKFLASH_KERNEL_DIR");
    let dir = std::path::PathBuf::from(dir);
    let cuda = Cuda::load().expect("cuda driver");
    let gemv_bytes = std::fs::read(dir.join("b12x_mxfp8_gemv_m1_sm121.cubin")).expect("gemv cubin");
    let (gemv_module, gemv) =
        mxfp8_gemv_launch::load_mode(&cuda, &gemv_bytes, Mode::Fused).expect("load gemv");
    let (gemv_f32_module, gemv_f32) =
        mxfp8_gemv_launch::load_mode(&cuda, &gemv_bytes, Mode::NonFusedF32ActivationF32Stores)
            .expect("load gemv f32a/f32o");
    let swiglu_bytes =
        std::fs::read(dir.join("swiglu_mxfp8_quant_sm121.cubin")).expect("swiglu cubin");
    let (swiglu_module, swiglu) = cuda.load_swiglu_quant(&swiglu_bytes).expect("load swiglu");
    let mut total = 0usize;
    // The exact resident shapes: two [2304,5120] gate/up operands, one [5120,2304] down operand,
    // their three E8M0 grids, and the per-token workspace.
    let weights = INTER * DIM;
    let up_scales = (INTER / BLOCK) * (DIM / BLOCK);
    let down_scales = (DIM / BLOCK) * (INTER / BLOCK);
    let w1 = fill(&cuda, weights, 0x3c, &mut total);
    let w3 = fill(&cuda, weights, 0x3c, &mut total);
    let w2 = fill(&cuda, weights, 0x3c, &mut total);
    let s1 = fill(&cuda, up_scales, 0x7f, &mut total);
    let s3 = fill(&cuda, up_scales, 0x7f, &mut total);
    let s2 = fill(&cuda, down_scales, 0x7f, &mut total);
    // The coded arm's row and readbacks.
    let a_codes = fill(&cuda, DIM, 0x3c, &mut total);
    let a_scales = fill(&cuda, DIM / BLOCK, 0x7f, &mut total);
    let gate = fill(&cuda, 2 * INTER, 0, &mut total);
    let up = fill(&cuda, 2 * INTER, 0, &mut total);
    let codes = fill(&cuda, INTER, 0x3c, &mut total);
    let scales = fill(&cuda, INTER / BLOCK, 0x7f, &mut total);
    let out = fill(&cuda, 2 * DIM, 0, &mut total);
    let mut readback = vec![0u8; 2 * DIM];
    // The f32 arm's row, rows and readbacks: a_f32 20 KB, gate/up/swiglu 9.2 KB each, out 20 KB.
    let a_f32 = fill(&cuda, 4 * DIM, 0, &mut total);
    let gate_f32 = fill(&cuda, 4 * INTER, 0, &mut total);
    let up_f32 = fill(&cuda, 4 * INTER, 0, &mut total);
    let swiglu_f32 = fill(&cuda, 4 * INTER, 0, &mut total);
    let out_f32 = fill(&cuda, 4 * DIM, 0, &mut total);
    let mut gate_readback = vec![0u8; 4 * INTER];
    let mut up_readback = vec![0u8; 4 * INTER];
    let mut swiglu_row = vec![0u8; 4 * INTER];
    let mut out_readback = vec![0u8; 4 * DIM];
    let hidden: Vec<f32> = (0..DIM).map(|index| (index as f32) * 1.0e-3).collect();
    let hidden_f32 = f32_le(&hidden);
    let mut host_codes = vec![0u8; DIM];
    let mut host_scales = vec![0u8; DIM / BLOCK];
    let gemv_params = |a_codes, a_scales, w, s, out, cols: usize, inner: usize| {
        Mxfp8DenseParams::new(
            a_codes,
            a_scales,
            w,
            s,
            out,
            1,
            cols as u32,
            inner as u32,
            0,
            0,
            0,
        )
    };
    let four_launches = |cuda: &Cuda| {
        mxfp8_gemv_launch::launch(
            cuda,
            gemv,
            &gemv_params(a_codes, a_scales, w1, s1, gate, INTER, DIM),
        )
        .expect("gate");
        mxfp8_gemv_launch::launch(
            cuda,
            gemv,
            &gemv_params(a_codes, a_scales, w3, s3, up, INTER, DIM),
        )
        .expect("up");
        let blocks = (INTER / BLOCK) as u32;
        let words = [
            gate.0,
            up.0,
            codes.0,
            scales.0,
            1,
            INTER as u64,
            u64::from(10.0f32.to_bits()),
            0,
            0,
            0,
        ];
        cuda.launch(swiglu, (blocks.div_ceil(8), 1, 1), (256, 1, 1), &words)
            .expect("swiglu");
        mxfp8_gemv_launch::launch(
            cuda,
            gemv,
            &gemv_params(codes, scales, w2, s2, out, DIM, INTER),
        )
        .expect("down");
    };
    let mut host_prologue = |cuda: &Cuda| {
        let rounded: Vec<f32> = hidden
            .iter()
            .map(|value| sparkflash_engine::quant::bf16_round(*value))
            .collect();
        let quantised =
            sparkflash_engine::quant::quantize_activation_fp8(&rounded).expect("quantise");
        host_codes.copy_from_slice(quantised.codes());
        host_scales.copy_from_slice(quantised.scales());
        cuda.copy_htod(&a_codes, &host_codes).expect("act htod");
        cuda.copy_htod(&a_scales, &host_scales).expect("scale htod");
        cuda.copy_dtoh(&mut readback, &out).expect("readback");
    };
    // The f32 arm: three launches, no SwiGLU launch, no quantise.
    let three_f32_launches = |cuda: &Cuda| {
        mxfp8_gemv_launch::launch_mode(
            cuda,
            gemv_f32,
            Mode::NonFusedF32ActivationF32Stores,
            &gemv_params(a_f32, a_scales, w1, s1, gate_f32, INTER, DIM),
        )
        .expect("gate f32");
        mxfp8_gemv_launch::launch_mode(
            cuda,
            gemv_f32,
            Mode::NonFusedF32ActivationF32Stores,
            &gemv_params(a_f32, a_scales, w3, s3, up_f32, INTER, DIM),
        )
        .expect("up f32");
        mxfp8_gemv_launch::launch_mode(
            cuda,
            gemv_f32,
            Mode::NonFusedF32ActivationF32Stores,
            &gemv_params(swiglu_f32, a_scales, w2, s2, out_f32, DIM, INTER),
        )
        .expect("down f32");
    };
    // The f32 arm's host side: the unchanged f32 row up, both gate/up readbacks, the host SwiGLU,
    // the SwiGLU row up, and the f32 output down. No bf16 round and no quantise anywhere.
    let mut f32_host_prologue = |cuda: &Cuda| {
        cuda.copy_htod(&a_f32, &hidden_f32).expect("row htod");
        cuda.copy_dtoh(&mut gate_readback, &gate_f32)
            .expect("gate readback");
        cuda.copy_dtoh(&mut up_readback, &up_f32)
            .expect("up readback");
        host_swiglu(&gate_readback, &up_readback, &mut swiglu_row);
        cuda.copy_htod(&swiglu_f32, &swiglu_row)
            .expect("swiglu htod");
        cuda.copy_dtoh(&mut out_readback, &out_f32)
            .expect("out readback");
    };
    let _ = (&gemv_module, &gemv_f32_module, &swiglu_module);
    for _ in 0..3 {
        four_launches(&cuda);
        host_prologue(&cuda);
        three_f32_launches(&cuda);
        f32_host_prologue(&cuda);
    }
    cuda.synchronize().expect("sync");
    let time = |label: &str, mut body: Box<dyn FnMut()>| {
        let started = Instant::now();
        for _ in 0..ITERATIONS {
            body();
        }
        cuda.synchronize().expect("sync");
        let micros = started.elapsed().as_micros() as f64 / ITERATIONS as f64;
        println!("  {label:<34} {micros:>9.1} us/layer");
        micros
    };
    println!(
        "WS-D shared-expert split, {} iterations, {} device bytes resident",
        ITERATIONS, total
    );
    let launches = time(
        "four launches (no prologue)",
        Box::new(|| four_launches(&cuda)),
    );
    let prologue = time(
        "host prologue + readback",
        Box::new(|| host_prologue(&cuda)),
    );
    let sync_only = time("synchronize only", Box::new(|| ()));
    let bytes = (2 * weights + weights + 2 * up_scales + down_scales) as f64;
    println!(
        "  launches vs 35.4 MB/layer: {:.1} GB/s",
        bytes / launches / 1000.0
    );
    println!(
        "  launches + prologue:       {:.3} ms/layer (row measured 2.626)",
        (launches + prologue) / 1000.0
    );
    println!("  sync-only baseline:        {sync_only:.1} us");
    // The f32-reference arm, the default path: three GEMV launches and its own host staging.
    let f32_launches = time(
        "f32 three launches (no prologue)",
        Box::new(|| three_f32_launches(&cuda)),
    );
    let f32_prologue = time(
        "f32 row staging + SwiGLU + readbacks",
        Box::new(|| f32_host_prologue(&cuda)),
    );
    println!(
        "  f32 launches vs 35.4 MB/layer: {:.1} GB/s",
        bytes / f32_launches / 1000.0
    );
    println!(
        "  f32 launches + prologue:       {:.3} ms/layer",
        (f32_launches + f32_prologue) / 1000.0
    );
    cuda.free_buffers(&[
        &w1,
        &w3,
        &w2,
        &s1,
        &s3,
        &s2,
        &a_codes,
        &a_scales,
        &gate,
        &up,
        &codes,
        &scales,
        &out,
        &a_f32,
        &gate_f32,
        &up_f32,
        &swiglu_f32,
        &out_f32,
    ]);
    cuda.free(gemv_module, &[]);
    cuda.free(gemv_f32_module, &[]);
    cuda.free(swiglu_module, &[]);
}
