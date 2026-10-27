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

#![forbid(unsafe_code)]

//! The Engram head's device-vs-twin numeric comparison, ready for the window (task-9, audit F4).
//!
//! This runs the three pinned kernels (engram_embed_sm121, engram_wkv_sm121,
//! engram_gate_write_sm121) on staging built here and compares every value against
//! [sparkflash_kernels::engram_native_ref], which the twin tests already pin op for op against the
//! released rule. It is the "numeric device comparison" task-9 leaves open: it needs a device, so it
//! is compiled and host-checked, NOT run by CI. The operator runs it when the GPUs are released.
//!
//! WHAT IS REAL AND WHAT IS SYNTHETIC. The embed rows are the eight released checkpoint rows in
//! [engram_head_ref::REAL_ROWS] (two layers, including the last row of each table). The wkv weights,
//! the stream and the q*k product are synthetic, from a documented LCG, because this binary compares
//! arithmetic, not the checkpoint read path (that comparison is the audit's separate open line). The
//! twin consumes the very same byte slices that are uploaded, so "staging is byte-consistent with
//! the twin" is printed as digests and asserted, not assumed.
//!
//! VERDICT RULE (printed in the last line, and the exit status):
//!   K1 must be bit-exact: the bf16 row, the activation codes and the activation scales.
//!   K2 must be bit-exact: the bf16 W8A8 output.
//!   K3's gates may differ only in the F8 class - f32 expf against the host's exp, bounded by
//!   4 f32 epsilon - and a write may differ only in a copy whose gate differed. Any other
//!   difference, and any mismatch at all in K1/K2, fails the run with the first differing index.
//!
//! Build the cubins first, on a node (they are already pinned by kernel_baseline.txt):
//!   for k in engram_embed_sm121 engram_wkv_sm121 engram_gate_write_sm121; do
//!     nvcc -cubin -arch=sm_121a -I kernels -o out/$k.cubin kernels/$k.cu; done
//!
//! Usage: engram_native_slice [--cubins DIR]   (default out/)
//!
//! Expected output on a correct window run:
//!   engram_native_slice: cubins out, 3 launches, rows 8x256 released, inner 2048, out_width 1280, hc 4, dim 256
//!   staging: row codes fnv1a64 <hex>, row scales <hex>, wkv codes <hex>, wkv grid <hex>, stream <hex>, product <hex>
//!   staging: every uploaded byte slice is the twin's input slice (same buffers)
//!   K1 engram_embed_sm121: bf16 row 2048/2048 bit-exact; act codes 2048/2048; act scales 64/64
//!   K2 engram_wkv_sm121: 1280/1280 bf16 outputs bit-exact
//!   K3 engram_gate_write_sm121: gates 4/4 exact (max |delta| 0e0), writes 1024/1024 bit-exact
//!   VERDICT: PASS - the three device kernels execute the native rule bit for bit; the only permitted class (sigmoid expf) did not even appear

use std::path::PathBuf;

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::engram_embed_launch::{self, EngramEmbedLaunchParams};
use sparkflash_kernels::engram_gate_write_launch::{self, EngramGateWriteLaunchParams};
use sparkflash_kernels::engram_head_ref::{HEAD_DIM, REAL_ROWS};
use sparkflash_kernels::engram_native_ref;
use sparkflash_kernels::engram_wkv_launch::{self, EngramWkvLaunchParams};

const ROWS_COLS: usize = REAL_ROWS.len();
const INNER: usize = ROWS_COLS * HEAD_DIM;
const HC: usize = 4;
const DIM: usize = 256;
const OUT_WIDTH: usize = DIM * (HC + 1);
/// The engine's Enram epsilon is the configured norm epsilon (forward/engram.rs:853).
const EPS: f32 = 1e-20;
/// E4M3 codes used by the synthetic weight are never NaN: 0x7f and 0xff are skipped.
const E4M3_NAN_A: u8 = 0x7f;
const E4M3_NAN_B: u8 = 0xff;

/// A tiny deterministic stream, so the staging is reproducible without a fixture file.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }

    /// A finite E4M3 code (NaN patterns skipped).
    fn e4m3(&mut self) -> u8 {
        loop {
            let code = self.byte();
            if code != E4M3_NAN_A && code != E4M3_NAN_B {
                return code;
            }
        }
    }

    /// An E8M0 code in 120..=134, i.e. a finite power-of-two scale.
    fn e8m0(&mut self) -> u8 {
        120 + self.byte() % 15
    }

    /// A small f32 in [-1, 1).
    fn f32(&mut self) -> f32 {
        (self.byte() as f32 / 255.0) - 0.5
    }
}

/// Decodes a lowercase hex string; the fixture strings are checked by the kernel tests already.
fn hex_bytes(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len() % 2, 0, "hex strings have whole bytes");
    (0..hex.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("the fixture is hex"))
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

fn bf16_bits(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])
}

fn f32_bf16_bits(value: f32) -> u16 {
    engram_native_ref::bf16_bits_of(value)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn parse_args() -> PathBuf {
    let mut cubins = PathBuf::from("out");
    let argv: Vec<String> = std::env::args().collect();
    let mut at = 1;
    while at < argv.len() {
        match argv[at].as_str() {
            "--cubins" => {
                cubins = PathBuf::from(&argv[at + 1]);
                at += 2;
            }
            other => panic!("unknown argument {other}; usage: engram_native_slice [--cubins DIR]"),
        }
    }
    cubins
}

fn cubin(dir: &std::path::Path, name: &str) -> Result<Vec<u8>, String> {
    let path = dir.join(format!("{name}.cubin"));
    std::fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn run() -> Result<(), String> {
    let dir = parse_args();

    // ---- staging, host side (the twin's inputs are these exact vectors) ----
    let mut row_codes = Vec::with_capacity(ROWS_COLS * HEAD_DIM);
    let mut row_scales = Vec::with_capacity(ROWS_COLS * HEAD_DIM / 32);
    for row in &REAL_ROWS {
        row_codes.extend(hex_bytes(row.weight));
        row_scales.extend(hex_bytes(row.scale));
    }
    let mut weights = Lcg::new(0x9e37_79b9_7f4a_7c15);
    let wkv_codes: Vec<u8> = (0..OUT_WIDTH * INNER).map(|_| weights.e4m3()).collect();
    let wkv_grid: Vec<u8> = (0..OUT_WIDTH.div_ceil(32) * (INNER / 32))
        .map(|_| weights.e8m0())
        .collect();
    let mut synthetic = Lcg::new(0x1234_5678_9abc_def0);
    let stream_f32: Vec<f32> = (0..HC * DIM)
        .map(|_| engram_native_ref::bf16_round(synthetic.f32()))
        .collect();
    let stream_bits: Vec<u16> = stream_f32
        .iter()
        .map(|value| f32_bf16_bits(*value))
        .collect();
    let mut stream_bytes = Vec::with_capacity(HC * DIM * 2);
    for bits in &stream_bits {
        stream_bytes.extend_from_slice(&bits.to_le_bytes());
    }
    let product: Vec<f32> = (0..HC * DIM).map(|_| synthetic.f32()).collect();
    let mut product_bytes = Vec::with_capacity(HC * DIM * 4);
    for value in &product {
        product_bytes.extend_from_slice(&value.to_le_bytes());
    }

    assert_eq!(row_codes.len(), ROWS_COLS * HEAD_DIM);
    assert_eq!(row_scales.len(), ROWS_COLS * HEAD_DIM / 32);
    assert_eq!(wkv_codes.len(), OUT_WIDTH * INNER);
    assert_eq!(wkv_grid.len(), OUT_WIDTH.div_ceil(32) * (INNER / 32));
    assert_eq!(stream_f32.len(), HC * DIM);
    println!(
        "engram_native_slice: cubins {}, 3 launches, rows {ROWS_COLS}x{HEAD_DIM} released, inner {INNER}, out_width {OUT_WIDTH}, hc {HC}, dim {DIM}",
        dir.display()
    );
    println!(
        "staging: row codes fnv1a64 {:016x}, row scales {:016x}, wkv codes {:016x}, wkv grid {:016x}, stream {:016x}, product {:016x}",
        fnv1a64(&row_codes),
        fnv1a64(&row_scales),
        fnv1a64(&wkv_codes),
        fnv1a64(&wkv_grid),
        fnv1a64(&stream_bytes),
        fnv1a64(&product_bytes)
    );
    println!("staging: every uploaded byte slice is the twin's input slice (same buffers)");

    // ---- device side ----
    let cuda = Cuda::load()?;
    let mut total = 0usize;
    let alloc =
        |bytes: usize, total: &mut usize| -> Result<Deviceptr, String> { cuda.alloc(bytes, total) };
    let row_codes_d = alloc(row_codes.len(), &mut total)?;
    let row_scales_d = alloc(row_scales.len(), &mut total)?;
    let fetched_d = alloc(ROWS_COLS * HEAD_DIM * 2, &mut total)?;
    let act_codes_d = alloc(ROWS_COLS * HEAD_DIM, &mut total)?;
    let act_scales_d = alloc(ROWS_COLS * HEAD_DIM / 32, &mut total)?;
    let wkv_codes_d = alloc(wkv_codes.len(), &mut total)?;
    let wkv_grid_d = alloc(wkv_grid.len(), &mut total)?;
    let wkv_out_d = alloc(OUT_WIDTH * 2, &mut total)?;
    let stream_d = alloc(stream_bytes.len(), &mut total)?;
    let product_d = alloc(product_bytes.len(), &mut total)?;
    let write_d = alloc(HC * DIM * 2, &mut total)?;
    let gates_d = alloc(HC * 4, &mut total)?;

    let (embed_module, embed_kernel) =
        engram_embed_launch::load(&cuda, &cubin(&dir, "engram_embed_sm121")?)?;
    let (wkv_module, wkv_kernel) =
        engram_wkv_launch::load(&cuda, &cubin(&dir, "engram_wkv_sm121")?)?;
    let (gate_module, gate_kernel) =
        engram_gate_write_launch::load(&cuda, &cubin(&dir, "engram_gate_write_sm121")?)?;

    let mut failures: Vec<String> = Vec::new();

    // ---- K1: row dequant -> bf16, then act_quant ----
    cuda.copy_htod(&row_codes_d, &row_codes)?;
    cuda.copy_htod(&row_scales_d, &row_scales)?;
    let embed_params = EngramEmbedLaunchParams {
        row_codes: row_codes_d,
        row_scales: row_scales_d,
        fetched: fetched_d,
        act_codes: act_codes_d,
        act_scales: act_scales_d,
        rows_cols: ROWS_COLS as u32,
        head_dim: HEAD_DIM as u32,
        trace_ring: 0,
        layer_id: 1,
        step_seq: 0,
    };
    engram_embed_launch::launch(&cuda, embed_kernel, &embed_params)?;
    let mut fetched = vec![0u8; ROWS_COLS * HEAD_DIM * 2];
    cuda.copy_dtoh(&mut fetched, &fetched_d)?;
    let mut act_codes = vec![0u8; ROWS_COLS * HEAD_DIM];
    let mut act_scales = vec![0u8; ROWS_COLS * HEAD_DIM / 32];
    cuda.copy_dtoh(&mut act_codes, &act_codes_d)?;
    cuda.copy_dtoh(&mut act_scales, &act_scales_d)?;

    let expected_fetched: Vec<f32> = (0..ROWS_COLS)
        .flat_map(|row| {
            engram_native_ref::embed_row(
                &row_codes[row * HEAD_DIM..(row + 1) * HEAD_DIM],
                &row_scales[row * HEAD_DIM / 32..(row + 1) * HEAD_DIM / 32],
                HEAD_DIM,
            )
        })
        .collect();
    let (expected_codes, expected_scales) = engram_native_ref::act_quantise(&expected_fetched);
    let mut fetched_exact = 0usize;
    for (index, value) in expected_fetched.iter().enumerate() {
        let want = f32_bf16_bits(*value);
        let got = bf16_bits(&fetched, index);
        if got == want {
            fetched_exact += 1;
        } else if failures.len() < 3 {
            failures.push(format!(
                "K1 bf16 row at {index}: device 0x{got:04x} twin 0x{want:04x}"
            ));
        }
    }
    let mut codes_exact = 0usize;
    for (index, (got, want)) in act_codes.iter().zip(&expected_codes).enumerate() {
        if got == want {
            codes_exact += 1;
        } else if failures.len() < 3 {
            failures.push(format!(
                "K1 act code at {index}: device 0x{got:02x} twin 0x{want:02x}"
            ));
        }
    }
    let mut scales_exact = 0usize;
    for (index, (got, want)) in act_scales.iter().zip(&expected_scales).enumerate() {
        if got == want {
            scales_exact += 1;
        } else if failures.len() < 3 {
            failures.push(format!(
                "K1 act scale at {index}: device 0x{got:02x} twin 0x{want:02x}"
            ));
        }
    }
    println!(
        "K1 engram_embed_sm121: bf16 row {fetched_exact}/{} bit-exact; act codes {codes_exact}/{}; act scales {scales_exact}/{}",
        expected_fetched.len(),
        expected_codes.len(),
        expected_scales.len()
    );

    // ---- K2: fp8 W8A8 over the K1 activation codes ----
    cuda.copy_htod(&wkv_codes_d, &wkv_codes)?;
    cuda.copy_htod(&wkv_grid_d, &wkv_grid)?;
    let wkv_params = EngramWkvLaunchParams {
        act_codes: act_codes_d,
        act_scales: act_scales_d,
        weight: wkv_codes_d,
        weight_scales: wkv_grid_d,
        out: wkv_out_d,
        out_width: OUT_WIDTH as u32,
        inner: INNER as u32,
        trace_ring: 0,
        layer_id: 1,
        step_seq: 0,
    };
    engram_wkv_launch::launch(&cuda, wkv_kernel, &wkv_params)?;
    let mut wkv_out = vec![0u8; OUT_WIDTH * 2];
    cuda.copy_dtoh(&mut wkv_out, &wkv_out_d)?;
    let expected_wkv = engram_native_ref::wkv(
        &expected_codes,
        &expected_scales,
        &wkv_codes,
        &wkv_grid,
        OUT_WIDTH,
        INNER,
    );
    let mut wkv_exact = 0usize;
    for (index, value) in expected_wkv.iter().enumerate() {
        let want = f32_bf16_bits(*value);
        let got = bf16_bits(&wkv_out, index);
        if got == want {
            wkv_exact += 1;
        } else if failures.len() < 3 {
            failures.push(format!(
                "K2 wkv out at {index}: device 0x{got:04x} twin 0x{want:04x}"
            ));
        }
    }
    println!(
        "K2 engram_wkv_sm121: {wkv_exact}/{} bf16 outputs bit-exact",
        expected_wkv.len()
    );

    // ---- K3: gate and write over the K2 bf16 output ----
    cuda.copy_htod(&stream_d, &stream_bytes)?;
    cuda.copy_htod(&product_d, &product_bytes)?;
    let gate_params = EngramGateWriteLaunchParams {
        kv: wkv_out_d,
        stream: stream_d,
        product: product_d,
        out: write_d,
        gates: gates_d,
        dim: DIM as u32,
        hc: HC as u32,
        eps: EPS,
        trace_ring: 0,
        layer_id: 1,
        step_seq: 0,
    };
    engram_gate_write_launch::launch(&cuda, gate_kernel, &gate_params)?;
    let mut write_out = vec![0u8; HC * DIM * 2];
    cuda.copy_dtoh(&mut write_out, &write_d)?;
    let mut gates = vec![0u8; HC * 4];
    cuda.copy_dtoh(&mut gates, &gates_d)?;
    let device_gates: Vec<f32> = (0..HC)
        .map(|copy| {
            f32::from_le_bytes(
                gates[4 * copy..4 * copy + 4]
                    .try_into()
                    .expect("four bytes"),
            )
        })
        .collect();
    let (twin_gates, twin_write) =
        engram_native_ref::gate_and_write(&expected_wkv, &stream_f32, &product, DIM, HC, EPS);
    let mut gates_exact = 0usize;
    let mut max_gate_delta = 0.0f32;
    for copy in 0..HC {
        let delta = (device_gates[copy] - twin_gates[copy]).abs();
        max_gate_delta = max_gate_delta.max(delta);
        if delta == 0.0 {
            gates_exact += 1;
        } else if delta > 4.0 * f32::EPSILON && failures.len() < 3 {
            failures.push(format!(
                "K3 gate at copy {copy}: device {} twin {} delta {delta:e}",
                device_gates[copy], twin_gates[copy]
            ));
        }
    }
    let mut writes_exact = 0usize;
    for copy in 0..HC {
        for column in 0..DIM {
            let index = copy * DIM + column;
            let got = bf16_bits(&write_out, index);
            let want = f32_bf16_bits(twin_write[index]);
            if got == want {
                writes_exact += 1;
            } else if device_gates[copy] == twin_gates[copy] && failures.len() < 3 {
                failures.push(format!(
                    "K3 write at copy {copy} column {column}: device 0x{got:04x} twin 0x{want:04x} with equal gates"
                ));
            }
        }
    }
    println!(
        "K3 engram_gate_write_sm121: gates {gates_exact}/{HC} exact (max |delta| {max_gate_delta:e}), writes {writes_exact}/{} bit-exact",
        HC * DIM
    );

    // ---- release and verdict ----
    cuda.free_buffers(&[
        &row_codes_d,
        &row_scales_d,
        &fetched_d,
        &act_codes_d,
        &act_scales_d,
        &wkv_codes_d,
        &wkv_grid_d,
        &wkv_out_d,
        &stream_d,
        &product_d,
        &write_d,
        &gates_d,
    ]);
    cuda.free(embed_module, &[]);
    cuda.free(wkv_module, &[]);
    cuda.free(gate_module, &[]);

    let bit_exact = fetched_exact == expected_fetched.len()
        && codes_exact == expected_codes.len()
        && scales_exact == expected_scales.len()
        && wkv_exact == expected_wkv.len()
        && writes_exact == HC * DIM
        && max_gate_delta <= 4.0 * f32::EPSILON;
    if !failures.is_empty() || !bit_exact {
        for failure in &failures {
            eprintln!("MISMATCH: {failure}");
        }
        println!(
            "VERDICT: FAIL - the device and the native-rule twin disagree; the first mismatch is printed above"
        );
        return Err(format!(
            "{fetched_exact}/{}, {codes_exact}/{}, {scales_exact}/{}, {wkv_exact}/{}, {writes_exact}/{} exact, max gate delta {max_gate_delta:e}",
            expected_fetched.len(),
            expected_codes.len(),
            expected_scales.len(),
            expected_wkv.len(),
            HC * DIM
        ));
    }
    let class = if gates_exact == HC {
        "the only permitted class (sigmoid expf) did not even appear".to_owned()
    } else {
        format!(
            "the only difference is the sigmoid's f32 expf class on {} of {HC} gates, max |delta| {max_gate_delta:e}",
            HC - gates_exact
        )
    };
    println!(
        "VERDICT: PASS - the three device kernels execute the native rule bit for bit; {class}"
    );
    Ok(())
}
