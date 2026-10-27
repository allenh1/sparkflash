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

//! The indexer's query projection: the engine's host native rule and the pinned B12x MXFP8 dense
//! kernel's twin must agree on the same operand bytes, and the row chunking the device arm uses
//! must not change a bit (F3, task-8).
//!
//! This is a no-GPU test. The kernel twin it compares against is `sparkflash_kernels::mxfp8_ref`,
//! the specification of what `b12x_mxfp8_dense_sm121.cu` computes: f32 accumulation, K ascending in
//! 32-value blocks and ascending inside a block, one bf16 round at the end. The engine's host path
//! is `indexer_wq_b_native`: bf16-narrow the latent, quantise it E4M3 with 32-value ue8m0 blocks,
//! multiply the same-quantised weight in f32, narrow the output to bf16. If those two disagree on
//! the same bytes, the device arm would be a different arithmetic from the host arm, so agreement
//! here is the F3 seam's proof rather than a tolerance.

use sparkflash_engine::forward::attention::{
    indexer_chunk_rows, indexer_wq_b_chunked, indexer_wq_b_native,
};
use sparkflash_engine::quant;
use sparkflash_kernels::m4_quant::act_quant_fp8_row;
use sparkflash_kernels::mxfp8_ref::{b12x_scale_index, b12x_weight, b12x_weight_index, dense_gemm};

/// Query rows: one prefill chunk, small enough for a unit test.
const ROWS: usize = 5;
/// Indexer output columns: a whole number of B12x 32-column tiles, so the tiling is a bijection.
const COLS: usize = 64;
/// The activation width: whole 32-value MX blocks.
const INNER: usize = 64;

/// A weight element's exact E4M3 code: deterministic, signed, and never one of the two NaN codes.
fn weight_code(n: usize, k: usize) -> u8 {
    let magnitude = ((n * INNER + k) * 37 + 11) % 0x7e;
    let magnitude = u8::try_from(magnitude).expect("below 0x7e");
    if (n + k).is_multiple_of(2) {
        magnitude
    } else {
        magnitude | 0x80
    }
}

/// A ue8m0 block scale code: 2^-6 through 2^0.
fn scale_code(n: usize, k: usize) -> u8 {
    u8::try_from(121 + (n / 32 + k / 32) % 7).expect("a ue8m0 code below 128")
}

/// The B12x-packed weight bytes, packed exactly as `read_packed` tiles the checkpoint tensor.
fn packed_weight() -> Vec<u8> {
    let mut tiled = vec![0u8; COLS * INNER];
    for n in 0..COLS {
        for k in 0..INNER {
            tiled[b12x_weight_index(n, k, INNER)] = weight_code(n, k);
        }
    }
    tiled
}

/// The weight's ue8m0 grid in checkpoint order, `[COLS / 32][INNER / 32]`.
fn weight_scales() -> Vec<u8> {
    (0..COLS.div_ceil(32) * INNER.div_ceil(32))
        .map(|index| scale_code((index / (INNER / 32)) * 32, (index % (INNER / 32)) * 32))
        .collect()
}

/// The dequantised weight, row-major [COLS][INNER], exactly as the engine's host path reads it.
fn dequantised_weight() -> Vec<f32> {
    let scales = weight_scales();
    (0..COLS * INNER)
        .map(|index| {
            let (n, k) = (index / INNER, index % INNER);
            quant::f8e4m3_to_f32(weight_code(n, k))
                * quant::f8e8m0_to_f32(scales[b12x_scale_index(n, k, INNER)])
        })
        .collect()
}

/// `ROWS x INNER` bf16-representable activations: the q_norm output the indexer's wq_b receives.
fn latent() -> Vec<f32> {
    (0..ROWS * INNER)
        .map(|index| quant::bf16_round(((index % 13) as f32) * 0.031_25 - 0.187_5))
        .collect()
}

/// The activation codes and scales the native rule produces for `values`.
fn activation(values: &[f32]) -> quant::QuantisedFp8 {
    let narrowed: Vec<f32> = values
        .iter()
        .map(|value| quant::bf16_round(*value))
        .collect();
    quant::quantize_activation_fp8(&narrowed).expect("a whole number of 32-value blocks")
}

#[test]
fn the_engine_host_rule_and_the_kernel_twin_agree_bit_for_bit() {
    let latent = latent();
    let tiled = packed_weight();

    // Op for op first: the two arms must agree on the operands, not only on the final bits.
    let narrowed: Vec<f32> = latent
        .iter()
        .map(|value| quant::bf16_round(*value))
        .collect();
    let keys = activation(&latent);
    for (row, engine_codes) in keys.codes().chunks(INNER).enumerate() {
        let (twin_codes, twin_scales) =
            act_quant_fp8_row(&narrowed[row * INNER..(row + 1) * INNER]);
        assert_eq!(
            engine_codes,
            twin_codes.as_slice(),
            "row {row} activation codes"
        );
        assert_eq!(
            &keys.scales()[row * (INNER / 32)..(row + 1) * (INNER / 32)],
            twin_scales.as_slice(),
            "row {row} activation scales"
        );
    }
    let scales = weight_scales();
    let dequantised = dequantised_weight();
    for n in 0..COLS {
        for k in 0..INNER {
            assert_eq!(
                dequantised[n * INNER + k].to_bits(),
                b12x_weight(&tiled, &scales, n, k, INNER).to_bits(),
                "weight ({n}, {k}) dequantises differently"
            );
        }
    }

    let engine = indexer_wq_b_native(&latent, &dequantised_weight(), ROWS, COLS, INNER)
        .expect("the native rule over the whole chunk");

    let activation_codes = activation(&latent);
    let kernel = dense_gemm(
        activation_codes.codes(),
        activation_codes.scales(),
        &tiled,
        &scales,
        ROWS,
        COLS,
        INNER,
    );
    assert_eq!(kernel.len(), ROWS * COLS);
    for (index, (host, twin)) in engine.iter().zip(kernel.iter()).enumerate() {
        assert_eq!(
            u16::try_from(host.to_bits() >> 16).expect("a bf16 bit pattern"),
            *twin,
            "value {index} of {} differs between the host rule and the kernel twin",
            ROWS * COLS
        );
    }
}

#[test]
fn the_device_row_chunking_is_bit_transparent() {
    let latent = latent();
    let tiled = packed_weight();
    let buffer_bytes = 3 * COLS * 2;
    assert_eq!(indexer_chunk_rows(COLS, buffer_bytes), 3);

    let single = indexer_wq_b_native(&latent, &dequantised_weight(), ROWS, COLS, INNER)
        .expect("the native rule over the whole chunk");
    let chunked = indexer_wq_b_chunked(ROWS, COLS, buffer_bytes, |start, count| {
        let chunk = activation(&latent[start * INNER..(start + count) * INNER]);
        let kernel = dense_gemm(
            chunk.codes(),
            chunk.scales(),
            &tiled,
            &weight_scales(),
            count,
            COLS,
            INNER,
        );
        Ok(kernel
            .iter()
            .map(|bits| f32::from_bits(u32::from(*bits) << 16))
            .collect())
    })
    .expect("the same rows through the device's chunking");
    assert_eq!(chunked.len(), single.len());
    for (index, (whole, parts)) in single.iter().zip(chunked.iter()).enumerate() {
        assert_eq!(
            whole.to_bits(),
            parts.to_bits(),
            "value {index} differs between one projection and the chunked one"
        );
    }
}
