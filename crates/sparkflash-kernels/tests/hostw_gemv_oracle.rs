//! The host-weight decode GEMV against the engine's own host rules, at the four shapes the ref_seam
//! block serves.
//!
//! `kernels/b12x_hostw_gemv_m1_sm121.cu` replaces exactly one thing in each of these statements: the
//! matrix multiply. The compressor's `wkv`/`wgate` rows are `linalg::matmul_hot_rows`'s output, the
//! indexer's `wk` row is `WeightView::matmul`'s, and `weights_proj`'s row is that matmul rounded to
//! bf16 - all three are `matmul_checked`'s inner loop, `acc += xb[col] * wr[col]`, ascending `col`,
//! one accumulator per output. This test drives the kernel's model
//! ([sparkflash_kernels::hostw_gemv_ref]) on the operands the seam stages - the checkpoint's own
//! f32 or bf16 values in the kernel's B12x byte order - and compares its rows with
//! [sparkflash_engine::linalg::matmul] on the same operands, bit for bit, with no tolerance.
//!
//! It runs on a CPU host: the model is the kernel's arithmetic, not the kernel. The identity claim
//! the seam relies on is the arithmetic (non-fused multiply-then-add, one accumulator, ascending K),
//! and the kernel's PTX carries the same `mul.rn.f32` / `add.rn.f32` pair. What this cannot prove is
//! the device run itself; that is the row's gate.

use sparkflash_engine::linalg::matmul;
use sparkflash_engine::quant::bf16_round;
use sparkflash_kernels::hostw_gemv_ref::{
    bf16_bits_to_f32, bf16_store, bf16_weight_row, f32_weight_row, stage_bf16_weight,
    stage_f32_weight,
};

/// A deterministic value stream: the test needs variety, not randomness, and a rerun must compare
/// the same bits.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 40) as f32) / ((1u32 << 24) as f32);
        unit * 2.0 - 1.0
    }

    fn weights(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next() * 0.125).collect()
    }

    fn activation(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next()).collect()
    }
}

/// The host rule the seam runs before the arm: `linalg::matmul` over the decoded weight.
fn host_row(x: &[f32], weight: &[f32], cols: usize, inner: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    matmul(&mut out, x, weight, cols, inner, 1).expect("one row against a whole weight");
    out
}

fn assert_same_bits(symbol: &str, device: &[f32], host: &[f32]) {
    assert_eq!(device.len(), host.len(), "{symbol}: row width");
    for (col, (device, host)) in device.iter().zip(host).enumerate() {
        assert_eq!(
            device.to_bits(),
            host.to_bits(),
            "{symbol}: column {col}: the device chain {device} and the host chain {host} must agree bit for bit"
        );
    }
}

/// The compressor's `wkv`/`wgate` at ratio > 1: f32 weight [512][5120], f32 accumulator row.
#[test]
fn the_f32_weight_row_is_the_host_matmul() {
    let (cols, inner) = (512usize, 5120usize);
    let mut stream = Stream(0x51ed_2701);
    let x = stream.activation(inner);
    let weight = stream.weights(cols * inner);
    let staged = stage_f32_weight(&weight, cols, inner);
    let device = f32_weight_row(&x, &staged, cols, inner);
    let host = host_row(&x, &weight, cols, inner);
    assert_same_bits("b12x_hostw_gemv_m1_f32w_f32o_sm121", &device, &host);
}

/// The compressor at ratio == 1 and the indexer's `wk`: bf16 weight widened exactly, f32
/// accumulator row.
#[test]
fn the_bf16_weight_f32_row_is_the_host_matmul_over_the_widened_weight() {
    for (cols, inner, seed) in [(512usize, 5120usize, 7u64), (128, 512, 11)] {
        let mut stream = Stream(seed);
        let x = stream.activation(inner);
        let weight: Vec<u16> = stream
            .weights(cols * inner)
            .into_iter()
            .map(|value| (value.to_bits() >> 16) as u16)
            .collect();
        let widened: Vec<f32> = weight.iter().map(|bits| bf16_bits_to_f32(*bits)).collect();
        let staged = stage_bf16_weight(&weight, cols, inner);
        let device = bf16_weight_row(&x, &staged, cols, inner);
        let host = host_row(&x, &widened, cols, inner);
        assert_same_bits("b12x_hostw_gemv_m1_bf16w_f32o_sm121", &device, &host);
    }
}

/// The indexer's `weights_proj`: bf16 weight, bf16-stored row. The host rounds the matmul's row to
/// bf16 before the index factor multiplies it, and this is the same single rounding.
#[test]
fn the_bf16_weight_bf16_row_is_the_host_matmul_and_one_rounding() {
    let (cols, inner) = (32usize, 5120usize);
    let mut stream = Stream(0xfeed);
    let mut x = stream.activation(inner);
    // The host narrows the layer input to bf16 before the projection (a bf16 Linear input).
    for value in x.iter_mut() {
        *value = bf16_round(*value);
    }
    let weight: Vec<u16> = stream
        .weights(cols * inner)
        .into_iter()
        .map(|value| (value.to_bits() >> 16) as u16)
        .collect();
    let widened: Vec<f32> = weight.iter().map(|bits| bf16_bits_to_f32(*bits)).collect();
    let staged = stage_bf16_weight(&weight, cols, inner);
    let rows = bf16_weight_row(&x, &staged, cols, inner);
    let stored = bf16_store(&rows);
    let host: Vec<f32> = host_row(&x, &widened, cols, inner)
        .into_iter()
        .map(bf16_round)
        .collect();
    assert_eq!(stored.len(), host.len());
    for (col, (bits, host)) in stored.iter().zip(&host).enumerate() {
        assert_eq!(
            bf16_bits_to_f32(*bits).to_bits(),
            host.to_bits(),
            "b12x_hostw_gemv_m1_bf16w_sm121: column {col}"
        );
    }
}

/// The kernel family the baseline pins and the launch module loads is the one compiled here: the
/// manifest's byte count and digest are the launch module's constants, and every mode's declaration
/// is in the source the module embeds.
#[test]
fn the_pinned_artifact_is_the_one_the_launch_module_declares() {
    use sparkflash_kernels::hostw_gemv_launch::{
        CUBIN_BYTES, CUBIN_SHA256, KERNEL, Mode, declaration_matches,
    };
    let manifest = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/baseline/kernel_baseline.txt"
    ))
    .expect("the baseline manifest is in the crate");
    let line = manifest
        .lines()
        .find(|line| line.starts_with(&format!("kernel {KERNEL} ")))
        .expect("the manifest pins the host-weight GEMV");
    assert!(
        line.contains(&format!("cubin_bytes={CUBIN_BYTES}")),
        "the launch module's byte count is the manifest's: {line}"
    );
    assert!(
        line.contains(&format!("cubin_sha256={CUBIN_SHA256}")),
        "the launch module's digest is the manifest's: {line}"
    );
    for mode in Mode::all() {
        assert!(
            declaration_matches(mode),
            "{}: the declaration in the pinned source moved",
            mode.symbol()
        );
    }
}
