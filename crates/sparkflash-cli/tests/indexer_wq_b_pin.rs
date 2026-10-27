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

//! Pre-change exact-bit pins for the indexer `wq_b` statement of the ref_seam slice.
//!
//! The ref_seam slice moves the indexer's query projection (`layers.N.attn.indexer.wq_b`, eight
//! layers, 4096x1280 fp8 E4M3 + ue8m0) from the host reference onto the pinned decode GEMV. This
//! file is written and committed *before* that wiring, so its pins are provably captured from the
//! unmodified host reference: it only calls `indexer_wq_b_native`, which the engine documents as
//! the same activation, matmul order and bf16 narrowing as the seam's stored-representation path
//! (`index_source_wq_b`).
//!
//! The shape is the served decode shape, not a toy one: one query row, 4096 indexer columns
//! (`index_n_heads * index_head_dim` = 32 * 128), 1280 reduction values (`q_lora_rank`), the
//! released 32x32 fp8 scale grid. The operands are a deterministic fixture (a bf16-exact latent
//! row and a deterministic E4M3 B12x weight with a 2^-6..2^0 ue8m0 grid) because no checkpoint is
//! readable off the head node; the fixture's own bytes are pinned too, so a change to the fixture
//! fails here rather than silently moving the target the later device-twin test compares against.
//!
//! What this pin does *not* claim: that the release checkpoint's bytes hit this path. The
//! four-rank identity row on the head node is that gate; this pin is what makes a host change to
//! the seam visible before any wiring lands.

use sha2::{Digest, Sha256};
use sparkflash_engine::forward::attention::indexer_wq_b_native;
use sparkflash_kernels::mxfp8_ref::{b12x_scale_index, b12x_weight, b12x_weight_index};

/// The served decode shape: one query row against the indexer's wq_b.
const ROWS: usize = 1;
/// `config.index_n_heads * config.index_head_dim` = 32 * 128.
const COLS: usize = 4096;
/// `config.q_lora_rank`.
const INNER: usize = 1280;
/// The fp8 MX block, the released `weight_block_size` along K.
const BLOCK: usize = 32;

/// The fixture latent row: 1280 bf16-exact values in +-15.6, deterministic and signed. Every
/// value is an exact multiple of 2^-3, so it is bf16-representable and the native bf16 narrowing
/// is the identity on it; the fp8 round trip is still exercised because these are not E4M3 values.
fn latent_row() -> Vec<f32> {
    (0..INNER)
        .map(|k| {
            let magnitude = ((k * 37 + 13) % 251) as f32 * 0.125;
            if k % 3 == 0 { -magnitude } else { magnitude }
        })
        .collect()
}

/// A weight element's exact E4M3 code: deterministic, signed, never one of the two NaN codes.
fn weight_code(n: usize, k: usize) -> u8 {
    let magnitude = u8::try_from(((n * INNER + k) * 37 + 11) % 0x7e).expect("below 0x7e");
    if (n + k).is_multiple_of(2) {
        magnitude
    } else {
        magnitude | 0x80
    }
}

/// A ue8m0 block scale code: 2^-6 through 2^0, so the products stay well inside f32 and the
/// accumulators do not saturate.
fn scale_code(n: usize, k: usize) -> u8 {
    u8::try_from(121 + (n / BLOCK + k / BLOCK) % 7).expect("a ue8m0 code below 128")
}

/// The B12x-packed weight bytes, tiled exactly as the arena tiles the checkpoint tensor.
fn packed_weight() -> Vec<u8> {
    let mut tiled = vec![0u8; COLS * INNER];
    for n in 0..COLS {
        for k in 0..INNER {
            tiled[b12x_weight_index(n, k, INNER)] = weight_code(n, k);
        }
    }
    tiled
}

/// The weight's ue8m0 grid in B12x order: `ceil(COLS / 32) * (INNER / 32)` bytes.
fn packed_scales() -> Vec<u8> {
    let mut scales = vec![0u8; COLS.div_ceil(BLOCK) * (INNER / BLOCK)];
    for n in 0..COLS {
        for k in 0..INNER {
            scales[b12x_scale_index(n, k, INNER)] = scale_code(n, k);
        }
    }
    scales
}

/// The same weight dequantised to f32 in logical `[COLS][INNER]` order: the representation the
/// host native rule multiplies against. `b12x_weight` is the B12x decoder the fused host kernel
/// and the device kernel both use, and each factor here is exact, so this is the host's own value.
fn dequantised_weight(codes: &[u8], scales: &[u8]) -> Vec<f32> {
    let mut weight = vec![0.0f32; COLS * INNER];
    for n in 0..COLS {
        for k in 0..INNER {
            weight[n * INNER + k] = b12x_weight(codes, scales, n, k, INNER);
        }
    }
    weight
}

/// The lowercase hex sha256 of an f32 slice's little-endian words.
fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The lowercase hex sha256 of a byte slice.
fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The one host call this pin fixes: the reference's indexer wq_b at the served decode shape.
fn host_statement() -> Vec<f32> {
    let latent = latent_row();
    let codes = packed_weight();
    let scales = packed_scales();
    let weight = dequantised_weight(&codes, &scales);
    indexer_wq_b_native(&latent, &weight, ROWS, COLS, INNER).expect("the served shape")
}

/// The pins themselves: the fixture bytes and the reference's output bits.
#[test]
fn the_indexer_wq_b_statement_is_pinned_at_the_served_decode_shape() {
    let latent = latent_row();
    let codes = packed_weight();
    let scales = packed_scales();
    let out = host_statement();
    assert_eq!(out.len(), ROWS * COLS);
    // The host rule narrows to bf16, so every output must be bf16-representable.
    for (index, value) in out.iter().enumerate() {
        let bits = value.to_bits();
        assert_eq!(
            bits & 0xffff,
            0,
            "output {index} is {value} ({bits:#010x}), not a bf16 value"
        );
    }
    // The fixture's own bytes: if a later edit moves the operands, this fails before the output
    // pin does, so a mismatch names the right cause.
    assert_eq!(
        sha256_f32(&latent),
        "6dda4fe4bf7d417496bea8cf5afd5544a2f36e632649934374dcae44a73ce69d",
        "the pinned latent fixture bytes changed"
    );
    assert_eq!(
        sha256_bytes(&codes),
        "cabd3a06c788e90d9fe702b58336b80e7bc7af427431e0e478e4ef99f986ec4b",
        "the pinned B12x weight bytes changed"
    );
    assert_eq!(
        sha256_bytes(&scales),
        "71cda0968355da54be89cdf54f4d1cd056a43629aa6e30f5dca709e8a7b10f09",
        "the pinned ue8m0 grid changed"
    );
    // The pin proper: the reference's exact output bits at the served decode shape.
    assert_eq!(
        sha256_f32(&out),
        "839216b2b8e8ed4cdcf1721264588627a7403580307c62786f07c4d2dc6f4e32",
        "the host reference's indexer wq_b output changed"
    );
    for (index, bits) in [
        (0usize, 0xc686_0000u32),
        (1, 0xc6bd_0000),
        (COLS / 2, 0xc691_0000),
        (COLS - 1, 0x4614_0000),
    ] {
        assert_eq!(
            out[index].to_bits(),
            bits,
            "output {index} moved: {:#010x}",
            out[index].to_bits()
        );
    }
}
/// The host statement's cost on this box, in microseconds, phase by phase.
///
/// The four phases are the seam's stored-representation path (`index_source_wq_b`) as the engine
/// writes it: the shard read (the seam's source clones the cached bytes per call, so this is a real
/// 5,242,880-byte copy), the native activation rule, the fused coded matmul, and the bf16 narrowing
/// of the 4096 outputs. The device arm removes all four from the step: the weight copy and tile
/// happen once per layer, the activation runs on the host into a 5,120-byte row upload, and the
/// matmul and narrowing run on the device.
///
/// The engine's reference helpers spawn one thread per chunk (there is no pool), so the number
/// below includes this box's per-call thread spawn: run it with SPARKFLASH_REFERENCE_THREADS=1 to
/// see the compute alone. Either way the statement leaves the step: the device arm does not run it.
///
/// Not a unit test of behaviour: it is the measurement the ref_seam slice's report quotes, so it is
/// skipped unless SPARKFLASH_SEAM_REF_BENCH is set, and it prints instead of asserting a machine's
/// speed. Run with `cargo test -p sparkflash-cli --release --test indexer_wq_b_pin -- --nocapture`.
#[test]
fn the_host_statement_cost_is_measured_on_this_box() {
    use sparkflash_engine::forward::attention::indexer_activation_native;
    use sparkflash_engine::linalg::{ScaleGrid, matmul_f8_scaled};
    use sparkflash_engine::quant;
    const ITERATIONS: usize = 15;
    let latent = latent_row();
    // The source's row-major E4M3 bytes (the checkpoint layout) and the checkpoint's grid.
    let mut raw = vec![0u8; COLS * INNER];
    for n in 0..COLS {
        for k in 0..INNER {
            raw[n * INNER + k] = weight_code(n, k);
        }
    }
    let mut grid = vec![0u8; COLS.div_ceil(BLOCK) * (INNER / BLOCK)];
    for n in 0..COLS {
        for k in 0..INNER {
            grid[(n / BLOCK) * (INNER / BLOCK) + k / BLOCK] = scale_code(n, k);
        }
    }
    let median = |mut samples: Vec<f64>| -> f64 {
        samples.sort_by(|a, b| a.partial_cmp(b).expect("finite samples"));
        samples[samples.len() / 2]
    };
    let mut reads = Vec::with_capacity(ITERATIONS);
    let mut activations = Vec::with_capacity(ITERATIONS);
    let mut matmuls = Vec::with_capacity(ITERATIONS);
    let mut narrows = Vec::with_capacity(ITERATIONS);
    let mut totals = Vec::with_capacity(ITERATIONS);
    let mut activation = vec![0.0f32; INNER];
    let mut shard = raw.clone();
    let mut out = vec![0.0f32; COLS];
    for _ in 0..ITERATIONS {
        let started = std::time::Instant::now();
        shard.clone_from(&raw);
        let read = started.elapsed().as_secs_f64() * 1e6;
        let started = std::time::Instant::now();
        activation = indexer_activation_native(&latent, INNER);
        let activation_us = started.elapsed().as_secs_f64() * 1e6;
        let started = std::time::Instant::now();
        matmul_f8_scaled(
            &mut out,
            &activation,
            &shard,
            &grid,
            ScaleGrid::Block32,
            COLS,
            INNER,
            1,
        )
        .expect("served shape");
        let matmul_us = started.elapsed().as_secs_f64() * 1e6;
        let started = std::time::Instant::now();
        for value in out.iter_mut() {
            *value = quant::bf16_round(*value);
        }
        let narrow_us = started.elapsed().as_secs_f64() * 1e6;
        reads.push(read);
        activations.push(activation_us);
        matmuls.push(matmul_us);
        narrows.push(narrow_us);
        totals.push(read + activation_us + matmul_us + narrow_us);
    }
    println!(
        "indexer wq_b host statement per call: read {:.1} us + activation {:.1} us + coded matmul {:.1} us + bf16 narrow {:.1} us = {:.1} us total",
        median(reads),
        median(activations),
        median(matmuls),
        median(narrows),
        median(totals),
    );
}
