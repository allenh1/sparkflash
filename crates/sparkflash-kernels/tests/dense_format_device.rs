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

//! The dense format route's twelve entries (kernels/dense_format_sm121.cu, launched through
//! [sparkflash_kernels::dense_format::DenseFormat::project]) against an f64 sum of the host
//! reference's decode: BF16 weights as stored, FP8 E4M3 codes times their 128 x 128 tile's f32 or
//! BF16 inverse scale.
//!
//! The tile splits an f32 row into a BF16 head and remainder and accumulates in tensor-core
//! order, so it is the exact sum to within
//!
//!   |device - exact| <= (2^-16 + K * 2^-23) * sum_k |w_k x_k|  (+ 2^-8 |exact| for BF16 out)
//!
//! for a contraction of K (a BF16 row is read exactly and its 2^-16 term is slack). Weights and
//! scales vary across tiles and groups, so a scale read from the wrong tile, a code from the
//! wrong column or a group from the wrong offset is an error of the order of the sum itself.
//!
//! Ignored by default (it needs a CUDA device and the staged cubin); run on the CUDA host:
//!
//!   SPARKFLASH_KERNEL_DIR=<staged cubins> \
//!     cargo test -p sparkflash-kernels --test dense_format_device -- --ignored --nocapture

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_kernels::dense_format::{
    DenseFormat, DenseGroups, DenseIo, DenseMatrix, DenseRows, RowForm,
};
use sparkflash_kernels::m4_quant::fp8_e4m3_decode;
use sparkflash_kernels::test_io::{download, upload};

/// The row counts each entry is measured at: one tile row, part of one, and two tiles.
const TOKENS: [usize; 5] = [1, 3, 8, 37, 130];

/// A deterministic value stream.
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

    fn e4m3(&mut self) -> u8 {
        loop {
            let code = (self.next() & 0xff) as u8;
            if code & 0x7f != 0x7f {
                return code;
            }
        }
    }
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// How the weight is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Bf16,
    Fp8F32,
    Fp8Bf16,
}

impl Class {
    fn suffix(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8F32 => "fp8",
            Self::Fp8Bf16 => "fp8_bf16",
        }
    }
}

/// `groups` stored `[cols, inner]` matrices back to back: their bytes, their scale planes
/// (`[cols / 128][inner / 128]` a group) and their decoded f64 values.
struct Stored {
    bytes: Vec<u8>,
    scales: Vec<u8>,
    scale_elements: usize,
    values: Vec<f64>,
}

fn stored(class: Class, groups: usize, cols: usize, inner: usize, salt: u64) -> Stored {
    let mut stream = Lcg(salt);
    let tiles = cols.div_ceil(128) * (inner / 128);
    let mut bytes = Vec::new();
    let mut scales = Vec::new();
    let mut values = Vec::with_capacity(groups * cols * inner);
    for _ in 0..groups {
        let plane: Vec<f32> = (0..tiles)
            .map(|_| 2f32.powf(stream.unit() * 4.0) * (1.0 + 0.3 * stream.unit()))
            .collect();
        let plane: Vec<f32> = match class {
            Class::Fp8Bf16 => plane.iter().map(|&s| bf16_value(bf16_bits(s))).collect(),
            _ => plane,
        };
        match class {
            Class::Bf16 => {}
            Class::Fp8F32 => plane
                .iter()
                .for_each(|s| scales.extend_from_slice(&s.to_le_bytes())),
            Class::Fp8Bf16 => plane
                .iter()
                .for_each(|&s| scales.extend_from_slice(&bf16_bits(s).to_le_bytes())),
        }
        for col in 0..cols {
            for k in 0..inner {
                let value = match class {
                    Class::Bf16 => {
                        let bits = bf16_bits(stream.unit() * [1.0f32, 0.25, 4.0][col % 3]);
                        bytes.extend_from_slice(&bits.to_le_bytes());
                        f64::from(bf16_value(bits))
                    }
                    Class::Fp8F32 | Class::Fp8Bf16 => {
                        let code = stream.e4m3();
                        bytes.push(code);
                        let scale = plane[(col / 128) * (inner / 128) + k / 128];
                        f64::from(fp8_e4m3_decode(code)) * f64::from(scale)
                    }
                };
                values.push(value);
            }
        }
    }
    Stored {
        bytes,
        scales,
        scale_elements: match class {
            Class::Bf16 => 0,
            _ => tiles,
        },
        values,
    }
}

#[test]
#[ignore = "needs a CUDA device and SPARKFLASH_KERNEL_DIR"]
fn the_dense_format_entries_are_the_host_sum_within_the_split() {
    let dir = std::path::PathBuf::from(
        std::env::var("SPARKFLASH_KERNEL_DIR").expect("SPARKFLASH_KERNEL_DIR"),
    );
    let cuda = Cuda::load().expect("cuda driver");
    let cubin = std::fs::read(dir.join("dense_format_sm121.cubin")).expect("dense format cubin");
    let mut modules = Vec::new();
    // Columns that end inside a tile, an inner of five 128-wide tiles, two groups.
    let (cols, inner, groups) = (288usize, 640usize, 2usize);
    for class in [Class::Bf16, Class::Fp8F32, Class::Fp8Bf16] {
        let entries = DenseIo::ALL.map(|io| {
            // The kernel handle names its symbol for the run, as the catalogue's names do.
            let symbol: &'static str =
                Box::leak(format!("{}{}_sm121", io.prefix(), class.suffix()).into_boxed_str());
            let (module, kernel) = cuda.load_named(&cubin, symbol).expect("entry");
            modules.push(module);
            kernel
        });
        let format = DenseFormat {
            entries,
            scale_block: (class != Class::Bf16).then_some(DenseFormat::SCALE_BLOCK),
        };
        let weights = stored(class, groups, cols, inner, 0xd3a5 + class as u64);
        let w = upload(&cuda, &weights.bytes);
        let ws = if weights.scales.is_empty() {
            None
        } else {
            Some(upload(&cuda, &weights.scales))
        };
        let value_bytes = if class == Class::Bf16 { 2 } else { 1 };
        for io in DenseIo::ALL {
            for tokens in TOKENS {
                // [tokens][groups * inner] rows, a group's slice beside the next (wo_a's form),
                // with spare values at each row's end so the stride is not the width.
                let a_stride = groups * inner + 32;
                let mut stream = Lcg(0x900d + tokens as u64);
                let x: Vec<f32> = (0..tokens * a_stride)
                    .map(|at| {
                        let value = stream.unit() * [1.0f32, 0.37, 3.1, 11.5][at % 4];
                        match io.input {
                            RowForm::F32 => value,
                            RowForm::Bf16 => bf16_value(bf16_bits(value)),
                        }
                    })
                    .collect();
                let a = match io.input {
                    RowForm::F32 => upload(&cuda, &x),
                    RowForm::Bf16 => upload(
                        &cuda,
                        &x.iter().map(|&v| bf16_bits(v)).collect::<Vec<u16>>(),
                    ),
                };
                let out_stride = groups * cols + 64;
                let out_bytes = match io.output {
                    RowForm::F32 => 4,
                    RowForm::Bf16 => 2,
                };
                let out = upload(&cuda, &vec![0u8; tokens * out_stride * out_bytes]);
                format
                    .project(
                        &cuda,
                        io,
                        &DenseMatrix {
                            weight: w.0,
                            scales: ws.map_or(0, |ws| ws.0),
                            cols,
                            inner,
                        },
                        &DenseRows {
                            a: a.0,
                            a_stride,
                            out: out.0,
                            out_stride,
                            tokens,
                        },
                        DenseGroups {
                            count: groups,
                            a: inner as u64,
                            out: cols as u64,
                            weight: (cols * inner * value_bytes) as u64,
                            scales: weights.scale_elements as u64,
                        },
                    )
                    .expect("project");
                let device: Vec<f32> = match io.output {
                    RowForm::F32 => download(&cuda, &out, tokens * out_stride),
                    RowForm::Bf16 => download::<u16>(&cuda, &out, tokens * out_stride)
                        .into_iter()
                        .map(bf16_value)
                        .collect(),
                };
                let unit = 2f64.powi(-16) + inner as f64 * 2f64.powi(-23);
                let mut worst = 0.0f64;
                for token in 0..tokens {
                    for group in 0..groups {
                        for col in 0..cols {
                            let (mut exact, mut magnitude) = (0.0f64, 0.0f64);
                            for k in 0..inner {
                                let term = weights.values[(group * cols + col) * inner + k]
                                    * f64::from(x[token * a_stride + group * inner + k]);
                                exact += term;
                                magnitude += term.abs();
                            }
                            let got = f64::from(device[token * out_stride + group * cols + col]);
                            let rounding = match io.output {
                                RowForm::F32 => 0.0,
                                RowForm::Bf16 => 2f64.powi(-8) * exact.abs(),
                            };
                            let bound = unit * magnitude + rounding + f64::MIN_POSITIVE;
                            worst = worst.max((got - exact).abs() / bound);
                        }
                    }
                }
                eprintln!(
                    "{class:?} {:?}->{:?} tokens {tokens}: worst {worst:.3e} of the bound",
                    io.input, io.output
                );
                assert!(
                    worst <= 1.0,
                    "{class:?} {io:?} at {tokens} tokens is {worst:.3e} times the split bound"
                );
            }
        }
    }
}
