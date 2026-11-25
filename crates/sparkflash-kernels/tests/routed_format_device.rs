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

//! The row-scaled MXFP8 routed-expert entries (`routed_format_gate_up_mxfp8_row_sm121`,
//! `routed_format_down_mxfp8_row_sm121`, kernels/routed_format_sm121.cu) against an f64 sum of
//! the host reference's decode: every E4M3 code times its row's `ue8m0` scale for its 32 columns,
//! times the f32 activation.
//!
//! The tile is not bit-equal to the host's per-element f32 sum and is not meant to be: it splits
//! the f32 activation into BF16 head and remainder and accumulates in tensor-core order. What it
//! must be is that sum to within the split and the f32 accumulation:
//!
//!   |device - exact| <= (2^-16 + K * 2^-23) * sum_k |w_k x_k|
//!
//! for a contraction of K. The weights and scales vary, so a scale read from the wrong row or
//! block, or a code read from the wrong column, is an error of the order of the sum itself and
//! fails by orders of magnitude. The down entry is held to the same bound times the slot's
//! routing weight.
//!
//! Ignored by default (it needs a CUDA device and the staged cubin); run on the CUDA host:
//!
//!   SPARKFLASH_KERNEL_DIR=<staged cubins> \
//!     cargo test -p sparkflash-kernels --test routed_format_device -- --ignored --nocapture

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::m4_quant::{fp8_e4m3_decode, ue8m0_decode};
use sparkflash_kernels::test_io::{download, upload};

/// The token counts the entries are measured at (one item each, top-1, slot = token).
const TOKENS: [usize; 4] = [1, 3, 8, 37];

/// A deterministic byte and value stream.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0
    }

    /// An E4M3 code that is not NaN.
    fn e4m3(&mut self) -> u8 {
        loop {
            let code = (self.next() & 0xff) as u8;
            if code & 0x7f != 0x7f {
                return code;
            }
        }
    }

    /// A `ue8m0` scale between 2^-7 and 2^7.
    fn scale(&mut self) -> u8 {
        120 + (self.next() % 15) as u8
    }
}

/// One `[rows, cols]` row-scaled MXFP8 matrix: its codes, its `[rows, cols / 32]` scales and its
/// decoded f64 values.
struct Matrix {
    codes: Vec<u8>,
    scales: Vec<u8>,
    values: Vec<f64>,
}

fn matrix(rows: usize, cols: usize, salt: u64) -> Matrix {
    let mut stream = Lcg(salt);
    let codes: Vec<u8> = (0..rows * cols).map(|_| stream.e4m3()).collect();
    let scales: Vec<u8> = (0..rows * cols / 32).map(|_| stream.scale()).collect();
    let values = (0..rows * cols)
        .map(|at| {
            let (row, col) = (at / cols, at % cols);
            f64::from(fp8_e4m3_decode(codes[at]))
                * f64::from(ue8m0_decode(scales[row * (cols / 32) + col / 32]))
        })
        .collect();
    Matrix {
        codes,
        scales,
        values,
    }
}

/// Rows of activations whose magnitudes span four binades, so the remainder half matters.
fn activations(rows: usize, cols: usize, salt: u64) -> Vec<f32> {
    let mut stream = Lcg(salt);
    (0..rows * cols)
        .map(|at| stream.unit() * [1.0f32, 0.37, 3.1, 11.5][at % 4])
        .collect()
}

/// Each output's worst error against its exact sum, as a fraction of the bound.
fn worst_fraction(
    device: &[f32],
    weights: &Matrix,
    x: &[f32],
    tokens: usize,
    cols: usize,
    inner: usize,
    route: &[f32],
) -> f64 {
    let unit = 2f64.powi(-16) + inner as f64 * 2f64.powi(-23);
    let mut worst = 0.0f64;
    for token in 0..tokens {
        for col in 0..cols {
            let (mut exact, mut magnitude) = (0.0f64, 0.0f64);
            for k in 0..inner {
                let term = weights.values[col * inner + k] * f64::from(x[token * inner + k]);
                exact += term;
                magnitude += term.abs();
            }
            let weight = f64::from(route[token]);
            let error = (f64::from(device[token * cols + col]) - exact * weight).abs();
            let bound = unit * magnitude * weight.abs() + f64::MIN_POSITIVE;
            worst = worst.max(error / bound);
        }
    }
    worst
}

#[test]
#[ignore = "needs a CUDA device and SPARKFLASH_KERNEL_DIR"]
fn the_row_scaled_mxfp8_entries_are_the_host_sum_within_the_split() {
    let dir = std::path::PathBuf::from(
        std::env::var("SPARKFLASH_KERNEL_DIR").expect("SPARKFLASH_KERNEL_DIR"),
    );
    let cuda = Cuda::load().expect("cuda driver");
    let cubin = std::fs::read(dir.join("routed_format_sm121.cubin")).expect("routed format cubin");
    let (_gate_module, gate_up) = cuda
        .load_named(&cubin, "routed_format_gate_up_mxfp8_row_sm121")
        .expect("gate/up entry");
    let (_down_module, down) = cuda
        .load_named(&cubin, "routed_format_down_mxfp8_row_sm121")
        .expect("down entry");

    // gate/up: [inter, dim] over dim; down: [dim, inter] over inter.
    let (dim, inter) = (512usize, 256usize);
    let gate = matrix(inter, dim, 0x51ed);
    let w2 = matrix(dim, inter, 0x7a11);
    let gate_codes = upload(&cuda, &gate.codes);
    let gate_scales = upload(&cuda, &gate.scales);
    let down_codes = upload(&cuda, &w2.codes);
    let down_scales = upload(&cuda, &w2.scales);
    // One expert's nine table words: (weight, scales, wide) for w1, w3 and w2. The test reads w1
    // as both gate and up; the entry picks the matrix by its launch word.
    let table: Vec<u64> = vec![
        gate_codes.0,
        gate_scales.0,
        0,
        gate_codes.0,
        gate_scales.0,
        0,
        down_codes.0,
        down_scales.0,
        0,
    ];
    let table = upload(&cuda, &table);

    for tokens in TOKENS {
        let x = activations(tokens, dim, 0x900d + tokens as u64);
        let act = activations(tokens, inter, 0xacdc + tokens as u64);
        let route: Vec<f32> = (0..tokens).map(|token| 0.25 + token as f32 * 0.125).collect();
        let sorted: Vec<i32> = (0..tokens as i32).collect();
        let items: Vec<i32> = vec![0, 0, tokens as i32, 0];
        let (x_d, act_d, route_d) = (upload(&cuda, &x), upload(&cuda, &act), upload(&cuda, &route));
        let (sorted_d, items_d) = (upload(&cuda, &sorted), upload(&cuda, &items));
        let gate_out = upload(&cuda, &vec![0f32; tokens * inter]);
        let down_out = upload(&cuda, &vec![0f32; tokens * dim]);

        cuda.launch(
            gate_up,
            (inter.div_ceil(128) as u32, 1, 1),
            (256, 1, 1),
            &[
                table.0,
                items_d.0,
                x_d.0,
                gate_out.0,
                inter as u64,
                dim as u64,
                sorted_d.0,
                1,
                0,
            ],
        )
        .expect("gate/up launch");
        cuda.launch(
            down,
            (dim.div_ceil(128) as u32, 1, 1),
            (256, 1, 1),
            &[
                table.0,
                items_d.0,
                act_d.0,
                route_d.0,
                down_out.0,
                dim as u64,
                inter as u64,
                sorted_d.0,
                1,
            ],
        )
        .expect("down launch");
        let gate_device: Vec<f32> = download(&cuda, &gate_out, tokens * inter);
        let down_device: Vec<f32> = download(&cuda, &down_out, tokens * dim);
        let unit_route = vec![1.0f32; tokens];
        let gate_worst = worst_fraction(&gate_device, &gate, &x, tokens, inter, dim, &unit_route);
        let down_worst = worst_fraction(&down_device, &w2, &act, tokens, dim, inter, &route);
        eprintln!(
            "tokens {tokens}: gate/up worst {gate_worst:.3e} of the bound, down {down_worst:.3e}"
        );
        assert!(
            gate_worst <= 1.0,
            "gate/up at {tokens} tokens is {gate_worst:.3e} times the split bound"
        );
        assert!(
            down_worst <= 1.0,
            "down at {tokens} tokens is {down_worst:.3e} times the split bound"
        );
    }
}
