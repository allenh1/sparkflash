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

//! The tensor-core decode forms (kernels/opt_gemv_sm121.cu's opt_gemv_mma_*), the default decode
//! path of the E4M3- and bf16-row projections, on a device.
//!
//! Ignored by default (it needs a CUDA device and the opt GEMV cubin). Run:
//!
//!   SPARKFLASH_KERNEL_DIR=/path/to/cubins cargo test -p sparkflash-kernels \
//!       --test decode_mma_device -- --ignored --nocapture
//!
//! Two claims, on every decode projection the forms serve (E4M3 rows: wq_a, wkv, wq_b, wo_b; bf16
//! rows: the two-group wo_a), through the same [launch_opt_split] routing the decode step takes:
//!
//! 1. batch invariance: every row of an M-row launch, M = 1..=64, is bit-identical to that row
//!    launched alone, whichever form (one or two m16 tiles a block) and slot it lands in. This is
//!    the property that lets a request decode the same alone as in a batch.
//! 2. accuracy: against an f64 reference, the forms' worst error is the FMA GEMV's (which this test
//!    runs first, before the forms are installed): both are one bf16 rounding of an f32 sum.
//!
//! The forms are installed process-wide and cannot be uninstalled, so both claims live in one test.

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::m4_quant::{bf16_bits, bf16_bits_to_f32, fp8_e4m3_decode, ue8m0_decode};
use sparkflash_kernels::mxfp8_gemv_launch::{
    OptArgs, OptKind, install_decode_mma, install_opt, launch_opt_split,
};
use sparkflash_kernels::mxfp8_ref::{
    b12x_buffer_bytes, b12x_scale_bytes, b12x_scale_index, b12x_weight_index,
};

/// One MX scale per 32 values along K.
const BLOCK: usize = 32;
/// The most rows a decode step puts through one launch (eight streams of an eight-row verify).
const ROWS: usize = 64;
/// The rows checked against the f64 reference: both ends and two inside, across the m16 tiles.
const REFERENCE_ROWS: [usize; 4] = [0, 13, 37, 63];
/// The bf16 output rounding alone allows half an ulp, up to 2^-8 of a value (8 significant bits):
/// the bound on the worst error relative to the largest |reference| either path may reach.
const RELATIVE_BOUND: f64 = 1.0 / 256.0;

/// A decode projection: output columns and reduction length per group.
struct Shape {
    name: &'static str,
    kind: OptKind,
    cols: usize,
    inner: usize,
    groups: usize,
}

const SHAPES: [Shape; 5] = [
    Shape {
        name: "wq_a",
        kind: OptKind::Fp8aBf16o,
        cols: 1280,
        inner: 5120,
        groups: 1,
    },
    Shape {
        name: "wkv",
        kind: OptKind::Fp8aBf16o,
        cols: 512,
        inner: 5120,
        groups: 1,
    },
    Shape {
        name: "wq_b",
        kind: OptKind::Fp8aBf16o,
        cols: 8192,
        inner: 1280,
        groups: 1,
    },
    Shape {
        name: "wo_a",
        kind: OptKind::Bf16aBf16o,
        cols: 1024,
        inner: 4096,
        groups: 2,
    },
    Shape {
        name: "wo_b",
        kind: OptKind::Fp8aBf16o,
        cols: 5120,
        inner: 2048,
        groups: 1,
    },
];

/// xorshift32: deterministic operand bytes without a dependency.
struct Bytes(u32);

impl Bytes {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// An E4M3 code with an even exponent field, so never the NaN code.
    fn e4m3(&mut self) -> u8 {
        (self.next() & 0xf7) as u8
    }
}

/// One shape's host operands, its device buffers, and the FMA GEMV's output for all rows.
struct Case {
    shape: &'static Shape,
    /// Per group: B12x weight tiles, then the 32x32 block scales.
    w: Vec<u8>,
    ws: Vec<u8>,
    /// [ROWS][groups][inner] activations (E4M3 codes or bf16 words) and, for E4M3, their scales.
    a: Vec<u8>,
    a_scales: Vec<u8>,
    device_w: Deviceptr,
    device_ws: Deviceptr,
    device_a: Deviceptr,
    device_a_scales: Deviceptr,
    gemv: Vec<u8>,
}

impl Case {
    fn element_bytes(&self) -> usize {
        match self.shape.kind {
            OptKind::Bf16aBf16o => 2,
            _ => 1,
        }
    }

    fn w_bytes(&self) -> usize {
        b12x_buffer_bytes(self.shape.cols, self.shape.inner)
    }

    fn s_bytes(&self) -> usize {
        b12x_scale_bytes(self.shape.cols, self.shape.inner)
    }

    fn row_out_bytes(&self) -> usize {
        self.shape.cols * self.shape.groups * 2
    }

    /// `tokens` rows from row `first`, into `out`.
    fn args(&self, first: usize, tokens: usize, out: Deviceptr) -> OptArgs {
        let Shape {
            cols,
            inner,
            groups,
            ..
        } = *self.shape;
        let row = inner * groups;
        OptArgs {
            a: self.device_a.offset(first * row * self.element_bytes()).0,
            a_scales: self.device_a_scales.offset(first * row / BLOCK).0,
            w: self.device_w.0,
            ws: self.device_ws.0,
            out: out.0,
            cols: cols as u32,
            inner: inner as u32,
            tokens: tokens as u32,
            a_tok_stride: row as u64,
            out_tok_stride: (cols * groups) as u64,
            groups: groups as u32,
            a_grp_stride: inner as u64,
            w_grp_stride: self.w_bytes() as u64,
            s_grp_stride: self.s_bytes() as u64,
            out_grp_stride: cols as u64,
            k_stride_blocks: 0,
        }
    }

    /// The activation value of (row, group, k), exact in f64.
    fn activation(&self, row: usize, group: usize, k: usize) -> f64 {
        let index = (row * self.shape.groups + group) * self.shape.inner + k;
        match self.shape.kind {
            OptKind::Bf16aBf16o => f64::from(bf16_bits_to_f32(u16::from_le_bytes([
                self.a[2 * index],
                self.a[2 * index + 1],
            ]))),
            _ => {
                f64::from(fp8_e4m3_decode(self.a[index]))
                    * f64::from(ue8m0_decode(self.a_scales[index / BLOCK]))
            }
        }
    }

    /// Row `row` of the exact product, [groups][cols] as the output holds it.
    fn reference_row(&self, row: usize) -> Vec<f64> {
        let Shape {
            cols,
            inner,
            groups,
            ..
        } = *self.shape;
        let mut exact = Vec::with_capacity(groups * cols);
        for group in 0..groups {
            let (w, ws) = (group * self.w_bytes(), group * self.s_bytes());
            for col in 0..cols {
                exact.push(
                    (0..inner)
                        .map(|k| {
                            let weight = f64::from(fp8_e4m3_decode(
                                self.w[w + b12x_weight_index(col, k, inner)],
                            )) * f64::from(ue8m0_decode(
                                self.ws[ws + b12x_scale_index(col, k, inner)],
                            ));
                            self.activation(row, group, k) * weight
                        })
                        .sum(),
                );
            }
        }
        exact
    }
}

fn upload(cuda: &Cuda, bytes: &[u8], total: &mut usize) -> Deviceptr {
    let pointer = cuda.alloc(bytes.len().max(1), total).expect("alloc");
    if !bytes.is_empty() {
        cuda.copy_htod(&pointer, bytes).expect("htod");
    }
    pointer
}

fn bf16_word(bytes: &[u8], index: usize) -> f64 {
    f64::from(bf16_bits_to_f32(u16::from_le_bytes([
        bytes[2 * index],
        bytes[2 * index + 1],
    ])))
}

/// Runs `tokens` rows from `first` into a buffer first filled with NaN words, so a row the launch
/// never writes cannot pass as matching.
fn run(cuda: &Cuda, case: &Case, first: usize, tokens: usize, out: Deviceptr) -> Vec<u8> {
    let bytes = tokens * case.row_out_bytes();
    cuda.copy_htod(&out, &vec![0xff; bytes]).expect("clear");
    launch_opt_split(cuda, case.shape.kind, &case.args(first, tokens, out)).expect("launch");
    cuda.synchronize().expect("sync");
    let mut host = vec![0u8; bytes];
    cuda.copy_dtoh(&mut host, &out).expect("dtoh");
    host
}

/// The worst |output - exact| over the row in `slot` of `out`.
fn worst(out: &[u8], slot: usize, exact: &[f64]) -> f64 {
    let base = slot * exact.len();
    exact
        .iter()
        .enumerate()
        .map(|(index, exact)| (bf16_word(out, base + index) - exact).abs())
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "needs a CUDA device and SPARKFLASH_KERNEL_DIR"]
fn the_tensor_core_decode_is_batch_invariant_and_as_accurate_as_the_gemv() {
    let dir = std::path::PathBuf::from(
        std::env::var("SPARKFLASH_KERNEL_DIR").expect("SPARKFLASH_KERNEL_DIR"),
    );
    let cuda = Cuda::load().expect("cuda driver");
    let cubin = std::fs::read(dir.join("opt_gemv_sm121.cubin")).expect("opt gemv cubin");
    let _gemv_module = install_opt(&cuda, &cubin).expect("install the opt GEMV");
    let mut total = 0usize;
    let out = cuda
        .alloc(ROWS * 8192 * 2, &mut total)
        .expect("output rows");

    // The FMA GEMV first, while it still serves every launch.
    let mut cases = Vec::new();
    for (salt, shape) in SHAPES.iter().enumerate() {
        let mut bytes = Bytes(0x9e37_79b9 ^ salt as u32);
        let (w_bytes, s_bytes) = (
            b12x_buffer_bytes(shape.cols, shape.inner),
            b12x_scale_bytes(shape.cols, shape.inner),
        );
        let w: Vec<u8> = (0..w_bytes * shape.groups).map(|_| bytes.e4m3()).collect();
        let ws: Vec<u8> = (0..s_bytes * shape.groups)
            .map(|_| 118 + (bytes.next() & 15) as u8)
            .collect();
        let elements = ROWS * shape.groups * shape.inner;
        let (a, a_scales): (Vec<u8>, Vec<u8>) = match shape.kind {
            OptKind::Bf16aBf16o => (
                (0..elements)
                    .flat_map(|_| {
                        let value = ((bytes.next() % 20_001) as f32 - 10_000.0) / 4_000.0;
                        bf16_bits(value).to_le_bytes()
                    })
                    .collect(),
                Vec::new(),
            ),
            _ => (
                (0..elements).map(|_| bytes.e4m3()).collect(),
                (0..elements / BLOCK)
                    .map(|_| 116 + (bytes.next() & 15) as u8)
                    .collect(),
            ),
        };
        let mut case = Case {
            shape,
            device_w: upload(&cuda, &w, &mut total),
            device_ws: upload(&cuda, &ws, &mut total),
            device_a: upload(&cuda, &a, &mut total),
            device_a_scales: upload(&cuda, &a_scales, &mut total),
            w,
            ws,
            a,
            a_scales,
            gemv: Vec::new(),
        };
        case.gemv = run(&cuda, &case, 0, ROWS, out);
        cases.push(case);
    }

    let _mma_module = install_decode_mma(&cuda, &cubin).expect("install the tensor-core forms");
    let mut differing = 0usize;
    for case in &cases {
        let name = case.shape.name;
        let row_bytes = case.row_out_bytes();
        let alone: Vec<Vec<u8>> = (0..ROWS).map(|row| run(&cuda, case, row, 1, out)).collect();
        for (row, bytes) in alone.iter().enumerate() {
            assert!(
                (0..bytes.len() / 2).all(|index| bf16_word(bytes, index).is_finite()),
                "{name}: row {row} alone has non-finite outputs (an unwritten column?)"
            );
        }
        for tokens in 1..=ROWS {
            let together = run(&cuda, case, 0, tokens, out);
            for (row, alone) in alone.iter().enumerate().take(tokens) {
                assert!(
                    together[row * row_bytes..(row + 1) * row_bytes] == alone[..],
                    "{name}: row {row} of a {tokens}-row launch differs from the row alone"
                );
            }
        }

        let exact: Vec<Vec<f64>> = REFERENCE_ROWS
            .iter()
            .map(|&row| case.reference_row(row))
            .collect();
        let scale = exact
            .iter()
            .flatten()
            .fold(0.0f64, |max, value| max.max(value.abs()));
        let (mut gemv, mut mma) = (0.0f64, 0.0f64);
        for (&row, exact) in REFERENCE_ROWS.iter().zip(&exact) {
            gemv = gemv.max(worst(&case.gemv, row, exact));
            mma = mma.max(worst(&alone[row], 0, exact));
        }
        // The launches really left the FMA chain: a different chain rounds some sums differently.
        let moved = alone
            .iter()
            .enumerate()
            .map(|(row, bytes)| {
                let gemv = &case.gemv[row * row_bytes..(row + 1) * row_bytes];
                bytes
                    .chunks(2)
                    .zip(gemv.chunks(2))
                    .filter(|(a, b)| a != b)
                    .count()
            })
            .sum::<usize>();
        differing += moved;
        println!(
            "{name}: worst error / max |reference|: tensor cores {:.3e}, FMA GEMV {:.3e}; \
             {moved} of {} outputs differ from the GEMV's",
            mma / scale,
            gemv / scale,
            ROWS * row_bytes / 2
        );
        assert!(
            gemv / scale <= RELATIVE_BOUND && mma / scale <= RELATIVE_BOUND,
            "{name}: error past the bf16 rounding bound (tensor cores {:.3e}, GEMV {:.3e})",
            mma / scale,
            gemv / scale
        );
    }
    assert!(
        differing > 0,
        "no output differs from the FMA GEMV's: the launches never reached the tensor-core forms"
    );
}
