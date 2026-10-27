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

//! The compressor's whole host statement, with only the projection replaced by the host-weight
//! GEMV's row.
//!
//! The engine's compressor step (crates/sparkflash-engine/src/forward/attention.rs:1055, ratio == 1
//! branch at 1092-1109) runs two things per position: a projection of the layer input, and after it
//! a narrowing/norm chain that no device arm may move. The chain is, exactly:
//!
//!   latent = matmul(x, wkv)          // f32 accumulators, one rounded mul + rounded add, ascending K
//!   latent = bf16_round(latent)      // narrow_bf16 (attention.rs:764)
//!   rms_norm(latent, compressor.norm.weight, head_dim, norm_eps)  // f32
//!   latent = bf16_round(latent)      // narrow_bf16 again
//!
//! kernels/b12x_hostw_gemv_m1_sm121.cu replaces the first line only, for weights with no MXFP8
//! encoding: compressor.wkv is bf16 at ratio == 1 and f32 above it, and compressor.norm.weight is
//! bf16 (docs/archive/build/native-arithmetic-audit.md:153; model.py:446-449, 461-486). Its
//! b12x_hostw_gemv_m1_bf16w_f32o_sm121 / b12x_hostw_gemv_m1_f32w_f32o_sm121 rows are
//! __fadd_rn(acc, __fmul_rn(a, w)) with one accumulator per output column, ascending K, which is
//! matmul_checked's inner loop exactly.
//!
//! This test proves the substitution at the level the seam actually performs it: the latent after
//! the whole chain is bit-identical whether the projection came from the host matmul or from the
//! kernel's row, and a bf16-stored row (what the MXFP8 f32-activation symbol hands back) changes it,
//! which is why the f32-out symbol is the one the seam must use.
//!
//! The ratio > 1 branch (attention.rs:1112-1204) adds compressor.wgate and pool_group between the
//! projections and this chain; pool_group is host code that neither arm replaces, and the f32 weight
//! row is proved equal to matmul by tests/hostw_gemv_oracle.rs's f32 case, so the chain below is the
//! same statement there.
//!
//! This runs on a CPU host: it is the kernel's arithmetic model, not the kernel. The device run is
//! the row's gate, not this test's claim.

use sparkflash_engine::linalg::{matmul, rms_norm};
use sparkflash_engine::quant::bf16_round;
use sparkflash_kernels::hostw_gemv_ref::{
    bf16_bits_to_f32, bf16_store, bf16_weight_row, f32_weight_row, stage_bf16_weight,
    stage_f32_weight,
};

/// A deterministic value stream: variety without randomness, so a rerun compares the same bits.
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

/// The host rule's projection: linalg::matmul over the decoded weight, one row.
fn host_row(x: &[f32], weight: &[f32], cols: usize, inner: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    matmul(&mut out, x, weight, cols, inner, 1).expect("one row against a whole weight");
    out
}

/// compressor_step's chain after the projection: narrow_bf16, f32 rms_norm, narrow_bf16.
fn compressor_latent(
    mut latent: Vec<f32>,
    norm: &[f32],
    head_dim: usize,
    norm_eps: f32,
) -> Vec<f32> {
    for value in latent.iter_mut() {
        *value = bf16_round(*value);
    }
    rms_norm(&mut latent, norm, head_dim, norm_eps).expect("a whole row of head_dim");
    for value in latent.iter_mut() {
        *value = bf16_round(*value);
    }
    latent
}

fn assert_same_bits(what: &str, device: &[f32], host: &[f32]) {
    assert_eq!(device.len(), host.len(), "{what}: latent width");
    for (index, (device, host)) in device.iter().zip(host).enumerate() {
        assert_eq!(
            device.to_bits(),
            host.to_bits(),
            "{what}: element {index}: kernel row {device} and host row {host} must agree bit for bit"
        );
    }
}

/// A bf16-weight compressor at ratio == 1 (head_dim 512 over dim 5120): the whole latent, i.e. the
/// projection plus narrow_bf16 + rms_norm + narrow_bf16, is identical for both projections.
#[test]
fn the_ratio_one_compressor_latent_is_unchanged_when_the_projection_moves_to_the_kernel_row() {
    let (head_dim, dim) = (512usize, 5120usize);
    let norm_eps = 1e-6f32;
    let mut stream = Stream(0x9c4f_1a27);
    let x = stream.activation(dim);
    // The checkpoint's own values: bf16 bits widened exactly to f32.
    let weight: Vec<u16> = stream
        .weights(head_dim * dim)
        .into_iter()
        .map(|value| (value.to_bits() >> 16) as u16)
        .collect();
    let widened: Vec<f32> = weight.iter().map(|bits| bf16_bits_to_f32(*bits)).collect();
    let norm: Vec<f32> = stream
        .weights(head_dim)
        .into_iter()
        .map(bf16_round)
        .collect();

    let host = compressor_latent(
        host_row(&x, &widened, head_dim, dim),
        &norm,
        head_dim,
        norm_eps,
    );
    let staged = stage_bf16_weight(&weight, head_dim, dim);
    let kernel = compressor_latent(
        bf16_weight_row(&x, &staged, head_dim, dim),
        &norm,
        head_dim,
        norm_eps,
    );
    assert_same_bits(
        "b12x_hostw_gemv_m1_bf16w_f32o_sm121 + narrow/norm chain",
        &kernel,
        &host,
    );
}

/// The same statement at ratio > 1's f32 weight (compressor.wkv/wgate above ratio 1): the latent is
/// also identical, which is the f32 row through the same chain.
#[test]
fn the_f32_weight_compressor_latent_is_unchanged_when_the_projection_moves_to_the_kernel_row() {
    let (head_dim, dim) = (512usize, 5120usize);
    let norm_eps = 1e-6f32;
    let mut stream = Stream(0x2d17_88e1);
    let x = stream.activation(dim);
    let weight = stream.weights(head_dim * dim);
    let norm: Vec<f32> = stream
        .weights(head_dim)
        .into_iter()
        .map(bf16_round)
        .collect();

    let host = compressor_latent(
        host_row(&x, &weight, head_dim, dim),
        &norm,
        head_dim,
        norm_eps,
    );
    let staged = stage_f32_weight(&weight, head_dim, dim);
    let kernel = compressor_latent(
        f32_weight_row(&x, &staged, head_dim, dim),
        &norm,
        head_dim,
        norm_eps,
    );
    assert_same_bits(
        "b12x_hostw_gemv_m1_f32w_f32o_sm121 + narrow/norm chain",
        &kernel,
        &host,
    );
}

/// A measured property of the ratio == 1 branch: the store width of the projection row is NOT
/// observable. compressor_step narrows the row to bf16 immediately (narrow_bf16 at
/// crates/sparkflash-engine/src/forward/attention.rs:1103), so rounding it to bf16 once more before
/// that narrow is the identity. The f32-out and bf16-store symbols therefore produce the same
/// ratio == 1 latent; the seam keeps the f32-out symbol because the ratio > 1 arm below needs raw f32
/// rows, and one symbol per dtype is cheaper than a second launch path.
#[test]
fn the_ratio_one_latent_does_not_depend_on_the_projection_store_width() {
    let (head_dim, dim) = (512usize, 5120usize);
    let norm_eps = 1e-6f32;
    let mut stream = Stream(0x5b3d_44c9);
    let x = stream.activation(dim);
    let weight: Vec<u16> = stream
        .weights(head_dim * dim)
        .into_iter()
        .map(|value| (value.to_bits() >> 16) as u16)
        .collect();
    let widened: Vec<f32> = weight.iter().map(|bits| bf16_bits_to_f32(*bits)).collect();
    let norm: Vec<f32> = stream
        .weights(head_dim)
        .into_iter()
        .map(bf16_round)
        .collect();

    let from_f32_rows = compressor_latent(
        host_row(&x, &widened, head_dim, dim),
        &norm,
        head_dim,
        norm_eps,
    );
    let staged = stage_bf16_weight(&weight, head_dim, dim);
    let narrowed: Vec<f32> = bf16_store(&bf16_weight_row(&x, &staged, head_dim, dim))
        .iter()
        .map(|bits| bf16_bits_to_f32(*bits))
        .collect();
    let from_bf16_store = compressor_latent(narrowed, &norm, head_dim, norm_eps);
    assert_same_bits(
        "ratio == 1: the mandatory narrow_bf16 makes the store width unobservable",
        &from_bf16_store,
        &from_f32_rows,
    );
}

/// pool_group over one whole group, mirrored from crates/sparkflash-engine/src/forward/attention.rs
/// (985-1020): per element, max over the group, exp of (score - max), normalise, weighted f32 sum.
fn pool_group(kv: &[f32], score: &[f32], ratio: usize, head_dim: usize) -> Vec<f32> {
    let mut dest = vec![0.0f32; head_dim];
    for element in 0..head_dim {
        let mut max = f32::NEG_INFINITY;
        for row in 0..ratio {
            max = max.max(score[row * head_dim + element]);
        }
        if !max.is_finite() {
            dest[element] = 0.0;
            continue;
        }
        let mut weights = vec![0.0f32; ratio];
        let mut total = 0.0f32;
        for row in 0..ratio {
            let weight = (score[row * head_dim + element] - max).exp();
            weights[row] = weight;
            total += weight;
        }
        let mut acc = 0.0f32;
        for row in 0..ratio {
            let weight = if total > 0.0 {
                weights[row] / total
            } else {
                0.0
            };
            acc += weight * kv[row * head_dim + element];
        }
        dest[element] = acc;
    }
    dest
}

/// The ratio > 1 branch, where the store width IS load-bearing: compressor_step pools the raw f32
/// wkv and wgate rows with pool_group (attention.rs:1112-1204) BEFORE any bf16 narrowing, so a
/// bf16-stored row changes the pooled latent. Two claims, both on one group of three rows:
///   1. the kernel's f32 rows pool and chain to the host's latent bit for bit;
///   2. the same rows stored as bf16 pool to a different latent (the control), which is why the seam
///      must consume the f32-out symbols b12x_hostw_gemv_m1_f32w_f32o_sm121 and its wgate sibling.
#[test]
fn the_ratio_above_one_latent_requires_the_f32_rows() {
    let (head_dim, dim, ratio) = (512usize, 5120usize, 3usize);
    let norm_eps = 1e-6f32;
    let mut stream = Stream(0x77b1_02dd);
    let x_rows: Vec<Vec<f32>> = (0..ratio).map(|_| stream.activation(dim)).collect();
    let wkv = stream.weights(head_dim * dim);
    let wgate = stream.weights(head_dim * dim);
    let norm: Vec<f32> = stream
        .weights(head_dim)
        .into_iter()
        .map(bf16_round)
        .collect();

    let mut host_kv = Vec::with_capacity(ratio * head_dim);
    let mut host_score = Vec::with_capacity(ratio * head_dim);
    for x in &x_rows {
        host_kv.extend_from_slice(&host_row(x, &wkv, head_dim, dim));
        host_score.extend_from_slice(&host_row(x, &wgate, head_dim, dim));
    }
    let host = compressor_latent(
        pool_group(&host_kv, &host_score, ratio, head_dim),
        &norm,
        head_dim,
        norm_eps,
    );

    let staged_kv = stage_f32_weight(&wkv, head_dim, dim);
    let staged_score = stage_f32_weight(&wgate, head_dim, dim);
    let mut kernel_kv = Vec::with_capacity(ratio * head_dim);
    let mut kernel_score = Vec::with_capacity(ratio * head_dim);
    let mut stored_kv = Vec::with_capacity(ratio * head_dim);
    let mut stored_score = Vec::with_capacity(ratio * head_dim);
    for x in &x_rows {
        kernel_kv.extend_from_slice(&f32_weight_row(x, &staged_kv, head_dim, dim));
        kernel_score.extend_from_slice(&f32_weight_row(x, &staged_score, head_dim, dim));
        stored_kv.extend(
            bf16_store(&f32_weight_row(x, &staged_kv, head_dim, dim))
                .iter()
                .map(|bits| bf16_bits_to_f32(*bits)),
        );
        stored_score.extend(
            bf16_store(&f32_weight_row(x, &staged_score, head_dim, dim))
                .iter()
                .map(|bits| bf16_bits_to_f32(*bits)),
        );
    }
    let kernel = compressor_latent(
        pool_group(&kernel_kv, &kernel_score, ratio, head_dim),
        &norm,
        head_dim,
        norm_eps,
    );
    assert_same_bits(
        "ratio > 1: b12x_hostw_gemv_m1_f32w_f32o_sm121 rows through pool_group",
        &kernel,
        &host,
    );

    let from_stored = compressor_latent(
        pool_group(&stored_kv, &stored_score, ratio, head_dim),
        &norm,
        head_dim,
        norm_eps,
    );
    let differing = host
        .iter()
        .zip(&from_stored)
        .filter(|(host, stored)| host.to_bits() != stored.to_bits())
        .count();
    assert!(
        differing > 0,
        "the ratio > 1 control: bf16-stored rows before pool_group must change at least one latent          element, or the f32-store requirement has no measured basis"
    );
}
