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
use crate::stage2::*;
use crate::stage3::*;
use crate::stage8::*;

// ---------------------------------------------------------------------------------------------
// Stage 7: the chained layer-0 replay (B2). This is the device path of device-chain-design.md
// section 3: the mHC kernel, the attention kernels and the routed combine are launched in the
// order the layer computes, the dense projections run on the two newly adopted device kernels, the
// documented seams carry the elementwise arithmetic that has no kernel, and the eighteen boundary
// tensors of row 0 are captured and diffed against the host v1 row by comparison class.
//
// It measures ONE layer and one token: the chain input is the embedding row of token 0 with the
// identity initial mix, which is the right seed only at layer 0. The 40-layer token chain is a
// separate program.
// ---------------------------------------------------------------------------------------------

/// The B2 comparison class of one boundary tensor (docs/archive/build/phase1-boundary-table.md section 5).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BoundaryClass {
    ExactBits,
    ExactCodes,
}

/// Total wo_a output width: eight groups of 1024, the wo_b reduction width.
pub(crate) const WO_A_OUT: usize = 8 * 1024;

/// The eighteen tensors of one boundary row, in the v1 order of `boundary::TENSOR_ORDER`.
pub(crate) const B2_TENSORS: [(&str, BoundaryClass); 18] = [
    ("residual_in", BoundaryClass::ExactBits),
    ("hc_in_pre", BoundaryClass::ExactBits),
    ("hc_attn_pre", BoundaryClass::ExactBits),
    ("hc_attn_post", BoundaryClass::ExactBits),
    ("hc_attn_comb", BoundaryClass::ExactBits),
    ("pre_attention_hidden", BoundaryClass::ExactBits),
    ("pre_attention_xhat", BoundaryClass::ExactCodes),
    ("pre_attention_scales", BoundaryClass::ExactCodes),
    ("attention_out", BoundaryClass::ExactBits),
    ("post_attention_hidden", BoundaryClass::ExactBits),
    ("hc_ffn_pre", BoundaryClass::ExactBits),
    ("hc_ffn_post", BoundaryClass::ExactBits),
    ("hc_ffn_comb", BoundaryClass::ExactBits),
    ("pre_ffn_hidden", BoundaryClass::ExactBits),
    ("pre_ffn_xhat", BoundaryClass::ExactCodes),
    ("pre_ffn_scales", BoundaryClass::ExactCodes),
    ("ffn_out", BoundaryClass::ExactBits),
    ("residual_out", BoundaryClass::ExactBits),
];

/// The three stream-carrying tensors: their device value is a bf16 stream write, so the report
/// gives both the host delta and the fraction equal to bf16(host) (design section 4).
pub(crate) const B2_STREAM_TENSORS: [&str; 3] =
    ["residual_in", "post_attention_hidden", "residual_out"];

/// One value of a boundary row, as captured on either side: f32 tensors keep their bits, code
/// tensors their bytes.
#[allow(dead_code)]
pub(crate) enum RowValue {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

/// One mHC launch's device buffers. K1 (attention sublayer of layer 0), K8 (FFN sublayer) and K16
/// (the final hc_post) share them; the kernel updates X in place, so x is the captured stream.
pub(crate) struct LayerMhc {
    pub(crate) x: Deviceptr,
    pub(crate) y: Deviceptr,
    pub(crate) b: Deviceptr,
    pub(crate) c: Deviceptr,
    pub(crate) a: Deviceptr,
    pub(crate) predictor: Deviceptr,
    pub(crate) scale: Deviceptr,
    pub(crate) base: Deviceptr,
    pub(crate) norm: Deviceptr,
    pub(crate) fp8: Deviceptr,
    pub(crate) scales: Deviceptr,
    pub(crate) stats: Deviceptr,
    pub(crate) sync: Deviceptr,
    pub(crate) coeffs: Deviceptr,
    pub(crate) rms: Deviceptr,
    pub(crate) normed: Deviceptr,
}

impl LayerMhc {
    pub(crate) fn alloc(cuda: &Cuda) -> Result<Self, String> {
        let mut allocated = 0usize;
        let mhc = Self {
            x: cuda.alloc(HC_MULT * HIDDEN * 2, &mut allocated)?,
            y: cuda.alloc(HIDDEN * 2, &mut allocated)?,
            b: cuda.alloc(HC_MULT * HC_MULT * 4, &mut allocated)?,
            c: cuda.alloc(HC_MULT * 4, &mut allocated)?,
            a: cuda.alloc(HC_MULT * 4, &mut allocated)?,
            predictor: cuda.alloc(PREDICTOR_ROWS * HC_MULT * HIDDEN * 4, &mut allocated)?,
            scale: cuda.alloc(3 * 4, &mut allocated)?,
            base: cuda.alloc(PREDICTOR_ROWS * 4, &mut allocated)?,
            norm: cuda.alloc(HIDDEN * 4, &mut allocated)?,
            fp8: cuda.alloc(HIDDEN, &mut allocated)?,
            scales: cuda.alloc(HIDDEN / QUANT_BLOCK, &mut allocated)?,
            stats: cuda.alloc(MHC_GRID_STATS * 4, &mut allocated)?,
            sync: cuda.alloc(2 * 4, &mut allocated)?,
            coeffs: cuda.alloc(PREDICTOR_ROWS * 4, &mut allocated)?,
            rms: cuda.alloc(4, &mut allocated)?,
            normed: cuda.alloc(HIDDEN * 4, &mut allocated)?,
        };
        println!("  mHC device buffers: {allocated} B");
        Ok(mhc)
    }

    /// Uploads one launch's inputs. `x` is bf16 bit patterns; `y` is the kernel's bf16 tensor, so it
    /// is rounded once here exactly as the ABI requires.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload(
        &self,
        cuda: &Cuda,
        x: &[u16],
        y: &[f32],
        b: &[f32],
        c: &[f32],
        a: &[f32],
        predictor: &[f32],
        scale: &[f32],
        base: &[f32],
        norm: &[f32],
    ) -> Result<(), String> {
        cuda.copy_htod(&self.x, &bf16_bits_to_bytes(x))?;
        cuda.copy_htod(&self.y, &bf16_bits_to_bytes(&bf16_bits_of_f32(y)))?;
        cuda.copy_htod(&self.b, &f32_to_bytes(b))?;
        cuda.copy_htod(&self.c, &f32_to_bytes(c))?;
        cuda.copy_htod(&self.a, &f32_to_bytes(a))?;
        cuda.copy_htod(&self.predictor, &f32_to_bytes(predictor))?;
        cuda.copy_htod(&self.scale, &f32_to_bytes(scale))?;
        cuda.copy_htod(&self.base, &f32_to_bytes(base))?;
        cuda.copy_htod(&self.norm, &f32_to_bytes(norm))?;
        Ok(())
    }

    pub(crate) fn launch(
        &self,
        cuda: &Cuda,
        func: CuFunction,
        layer: usize,
        step: u64,
    ) -> Result<(), String> {
        // The grid rendezvous counters are per-launch state.
        cuda.copy_htod(&self.stats, &[0u8; MHC_GRID_STATS * 4])?;
        cuda.copy_htod(&self.sync, &[0u8; 2 * 4])?;
        let params: [u64; 19] = [
            self.x.0,
            self.y.0,
            self.b.0,
            self.c.0,
            self.a.0,
            self.predictor.0,
            self.scale.0,
            self.base.0,
            self.norm.0,
            self.fp8.0,
            self.scales.0,
            self.stats.0,
            self.sync.0,
            self.coeffs.0,
            self.rms.0,
            self.normed.0,
            0, // trace_ring: null, the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
            layer as u64,
            step,
        ];
        cuda.launch(
            func,
            (MHC_TILES, 1u32, 1u32),
            (MHC_THREADS, 1u32, 1u32),
            &params,
        )
    }

    pub(crate) fn x_bits(&self, cuda: &Cuda) -> Result<Vec<u16>, String> {
        let mut bytes = vec![0u8; HC_MULT * HIDDEN * 2];
        cuda.copy_dtoh(&mut bytes, &self.x)?;
        Ok(bytes_to_u16(&bytes))
    }

    pub(crate) fn coeffs(&self, cuda: &Cuda) -> Result<Vec<f32>, String> {
        let mut bytes = vec![0u8; PREDICTOR_ROWS * 4];
        cuda.copy_dtoh(&mut bytes, &self.coeffs)?;
        Ok(bytes_to_f32(&bytes))
    }

    pub(crate) fn normed(&self, cuda: &Cuda) -> Result<Vec<f32>, String> {
        let mut bytes = vec![0u8; HIDDEN * 4];
        cuda.copy_dtoh(&mut bytes, &self.normed)?;
        Ok(bytes_to_f32(&bytes))
    }

    pub(crate) fn codes(&self, cuda: &Cuda) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; HIDDEN];
        cuda.copy_dtoh(&mut bytes, &self.fp8)?;
        Ok(bytes)
    }

    pub(crate) fn scales(&self, cuda: &Cuda) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; HIDDEN / QUANT_BLOCK];
        cuda.copy_dtoh(&mut bytes, &self.scales)?;
        Ok(bytes)
    }

    pub(crate) fn release(&self, cuda: &Cuda) {
        cuda.free_buffers(&[
            &self.x,
            &self.y,
            &self.b,
            &self.c,
            &self.a,
            &self.predictor,
            &self.scale,
            &self.base,
            &self.norm,
            &self.fp8,
            &self.scales,
            &self.stats,
            &self.sync,
            &self.coeffs,
            &self.rms,
            &self.normed,
        ]);
    }
}

/// Widen the device's bf16 output bytes to f32 values, keeping the device's rounding.
fn widen_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
        .collect()
}

/// Decode a quantised activation row: E4M3 code times its block's ue8m0 scale.
pub(crate) fn decode_quantised_row(codes: &[u8], scales: &[u8]) -> Vec<f32> {
    codes
        .iter()
        .enumerate()
        .map(|(index, code)| fp8_e4m3_decode(*code) * ue8m0_decode(scales[index / QUANT_BLOCK]))
        .collect()
}

/// One dense MXFP8 GEMM on the device kernel: A is one activation row (E4M3 codes plus ue8m0
/// scales), W is already repacked into B12x order, the output is widened from the kernel's bf16.
///
/// The in-crate reference is run on the same bytes and compared bit for bit before the result is
/// used: the device kernel's arithmetic is the reference's, so a mismatch would be a repack or
/// launch bug in this harness, not a boundary difference.
#[allow(clippy::too_many_arguments)] // the kernel launch takes the harness context and the kernel buffers as distinct arguments.
pub(crate) fn dense_device_gemm(
    cuda: &Cuda,
    func: CuFunction,
    layer: usize,
    label: &str,
    a_codes: &[u8],
    a_scales: &[u8],
    w_packed: &[u8],
    w_scales: &[u8],
    cols: usize,
    inner: usize,
) -> Result<Vec<f32>, String> {
    let rows = 1usize;
    let mut allocated = 0usize;
    let a_d = cuda.alloc(a_codes.len(), &mut allocated)?;
    let scales_d = cuda.alloc(a_scales.len(), &mut allocated)?;
    let w_d = cuda.alloc(w_packed.len(), &mut allocated)?;
    let ws_d = cuda.alloc(w_scales.len(), &mut allocated)?;
    let out_d = cuda.alloc(cols * 2, &mut allocated)?;
    cuda.copy_htod(&a_d, a_codes)?;
    cuda.copy_htod(&scales_d, a_scales)?;
    cuda.copy_htod(&w_d, w_packed)?;
    cuda.copy_htod(&ws_d, w_scales)?;
    let params: [u64; 11] = [
        a_d.0,
        scales_d.0,
        w_d.0,
        ws_d.0,
        out_d.0,
        rows as u64,
        cols as u64,
        inner as u64,
        0, // trace_ring
        layer as u64,
        0, // step_seq
    ];
    cuda.launch(
        func,
        (1u32, (cols.div_ceil(64)) as u32, 1u32),
        (256u32, 1u32, 1u32),
        &params,
    )?;
    let mut bytes = vec![0u8; cols * 2];
    cuda.copy_dtoh(&mut bytes, &out_d)?;
    cuda.free_buffers(&[&a_d, &scales_d, &w_d, &ws_d, &out_d]);
    let _ = allocated;
    let device = widen_bf16(&bytes);
    let expected_bits =
        mxfp8_ref::dense_gemm(a_codes, a_scales, w_packed, w_scales, rows, cols, inner);
    let exact = device
        .iter()
        .zip(&expected_bits)
        .filter(|(value, bits)| value.to_bits() == half::bf16::from_bits(**bits).to_f32().to_bits())
        .count();
    println!(
        "  {label}: device vs in-crate reference on the same bytes {exact}/{cols} exact ({}x{})",
        cols, inner
    );
    if exact != cols {
        return Err(format!(
            "{label}: the device kernel does not reproduce mxfp8_ref::dense_gemm on the same bytes \
             ({exact} of {cols} exact)"
        ));
    }
    Ok(device)
}

/// One grouped wo_a launch on the released geometry (eight groups of 1024 output channels over
/// 4096 inputs each), output widened from bf16 and checked against the in-crate reference.
#[allow(clippy::too_many_arguments)] // the kernel launch takes the harness context and the kernel buffers as distinct arguments.
pub(crate) fn wo_a_device(
    cuda: &Cuda,
    func: CuFunction,
    layer: usize,
    attn_codes: &[u8],
    attn_scales: &[u8],
    w_packed: &[u8],
    w_scales: &[u8],
) -> Result<Vec<f32>, String> {
    const GROUPS: usize = 8;
    const LO_RANK: usize = 1024;
    const GROUP_INNER: usize = 4096;
    let out_bytes = GROUPS * LO_RANK * 2;
    let mut allocated = 0usize;
    let attn_d = cuda.alloc(attn_codes.len(), &mut allocated)?;
    let scales_d = cuda.alloc(attn_scales.len(), &mut allocated)?;
    let w_d = cuda.alloc(w_packed.len(), &mut allocated)?;
    let ws_d = cuda.alloc(w_scales.len(), &mut allocated)?;
    let out_d = cuda.alloc(out_bytes, &mut allocated)?;
    cuda.copy_htod(&attn_d, attn_codes)?;
    cuda.copy_htod(&scales_d, attn_scales)?;
    cuda.copy_htod(&w_d, w_packed)?;
    cuda.copy_htod(&ws_d, w_scales)?;
    let params: [u64; 9] = [
        attn_d.0,
        scales_d.0,
        w_d.0,
        ws_d.0,
        out_d.0,
        GROUPS as u64,
        0, // trace_ring
        layer as u64,
        0, // step_seq
    ];
    cuda.launch(
        func,
        (GROUPS as u32, (LO_RANK / 256) as u32, 1u32),
        (256u32, 1u32, 1u32),
        &params,
    )?;
    let mut bytes = vec![0u8; out_bytes];
    cuda.copy_dtoh(&mut bytes, &out_d)?;
    cuda.free_buffers(&[&attn_d, &scales_d, &w_d, &ws_d, &out_d]);
    let _ = allocated;
    let device = widen_bf16(&bytes);
    let expected_bits = mxfp8_ref::grouped_wo_a(
        attn_codes,
        attn_scales,
        w_packed,
        w_scales,
        GROUPS,
        GROUP_INNER,
        LO_RANK,
    );
    let exact = device
        .iter()
        .zip(&expected_bits)
        .filter(|(value, bits)| value.to_bits() == half::bf16::from_bits(**bits).to_f32().to_bits())
        .count();
    println!(
        "  K6 wo_a: device vs in-crate reference on the same bytes {exact}/{} exact (groups {GROUPS}, group_inner {GROUP_INNER}, lo_rank {LO_RANK})",
        GROUPS * LO_RANK
    );
    if exact != GROUPS * LO_RANK {
        return Err(format!(
            "K6 wo_a: the device kernel does not reproduce mxfp8_ref::grouped_wo_a on the same \
             bytes ({exact} of {} exact)",
            GROUPS * LO_RANK
        ));
    }
    Ok(device)
}

/// Repack the grouped wo_a checkpoint weight into the eight kernel-group B12x buffers; the scale
/// grid keeps its per-group [lo_rank/32, group_inner/32] row-major order.
pub(crate) fn pack_wo_a(weight: &[u8], scales: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
    const GROUPS: usize = 8;
    const LO_RANK: usize = 1024;
    const GROUP_INNER: usize = 4096;
    if weight.len() != GROUPS * LO_RANK * GROUP_INNER {
        return Err(format!(
            "wo_a weight: {} bytes, expected {}",
            weight.len(),
            GROUPS * LO_RANK * GROUP_INNER
        ));
    }
    let expected_scales = GROUPS * (LO_RANK / 32) * (GROUP_INNER / 32);
    if scales.len() != expected_scales {
        return Err(format!(
            "wo_a scales: {} bytes, expected {expected_scales}",
            scales.len()
        ));
    }
    let mut packed = Vec::with_capacity(weight.len());
    for group in 0..GROUPS {
        let start = group * LO_RANK * GROUP_INNER;
        packed.extend_from_slice(&pack_b12x(
            &weight[start..start + LO_RANK * GROUP_INNER],
            LO_RANK,
            GROUP_INNER,
        ));
    }
    Ok((packed, scales.to_vec()))
}

/// Predicts the attention tail (attention_out) from one f32 window-KV latent, for the D4
/// attribution check: the same arithmetic and the same bf16 storage points as the device chain, with
/// the window KV supplied in a chosen format. Used twice: once with the device's FP4 dequant, which
/// must reproduce this run's attention_out, and once with the host's FP8 dequant, which says what the
/// host row's format would have produced from the same latent.
fn predict_attention_out(
    q: &[f32],
    sink: &[f32],
    kv_values: &[f32],
    wo_a_w: &[u8],
    wo_a_s: &[u8],
    wo_b_w: &[u8],
    wo_b_s: &[u8],
) -> Vec<f32> {
    let mut raw = Vec::with_capacity(MLA_HEADS * MLA_LATENT);
    for head in 0..MLA_HEADS {
        let query = &q[head * MLA_LATENT..(head + 1) * MLA_LATENT];
        let mut score = 0.0f32;
        for (value, kv) in query.iter().zip(kv_values) {
            score += value * kv;
        }
        let score = score * mla_ref::MLA_SCORE_SCALE;
        // One candidate at position 0: the numerator weight is one and the sink is the only other
        // term of the softmax denominator (mla_ref::attention_impl pass two and three).
        let weight = 1.0 / (1.0 + (sink[head] - score).exp());
        for kv in kv_values {
            raw.push(weight * kv);
        }
    }
    let raw = widen_bits(&bf16_bits_of_f32(&raw));
    let (codes, scales) = act_quant_fp8_row(&raw);
    let lo_bits = mxfp8_ref::grouped_wo_a(&codes, &scales, wo_a_w, wo_a_s, 8, 4096, 1024);
    let lo = widen_bits(&lo_bits);
    let (lo_codes, lo_scales) = act_quant_fp8_row(&lo);
    let out_bits =
        mxfp8_ref::dense_gemm(&lo_codes, &lo_scales, wo_b_w, wo_b_s, 1, HIDDEN, WO_A_OUT);
    widen_bits(&out_bits)
}

/// (exact count, max abs delta) between two f32 slices of equal length.
fn f32_delta_stats(left: &[f32], right: &[f32]) -> (usize, f32) {
    let mut exact = 0usize;
    let mut max = 0.0f32;
    for (left, right) in left.iter().zip(right) {
        if left.to_bits() == right.to_bits() {
            exact += 1;
        }
        max = max.max((left - right).abs());
    }
    (exact, max)
}

/// One captured value and its comparison against the host row; the fields are what the report
/// prints, not a pass/fail (the bar is measured by this run, per design section 8).
struct RowDelta {
    name: &'static str,
    class: BoundaryClass,
    values: usize,
    host_exact: usize,
    max_abs_delta: f32,
    mean_abs_delta: f32,
    at: usize,
    differing_codes: usize,
    max_code_distance: i32,
    mean_code_distance: f32,
    /// Differing codes bucketed by distance: 1, 2..=3, 4..=15, 16 and above.
    code_distance_buckets: [usize; 4],
    bf16_host_exact: Option<usize>,
    bf16_host_max_delta: Option<f32>,
}

impl RowDelta {
    pub(crate) fn single_layer_delta(&self) -> f64 {
        if self.class == BoundaryClass::ExactCodes {
            f64::from(self.differing_codes as u32) / self.values as f64
        } else {
            f64::from(self.max_abs_delta)
        }
    }

    /// The code-distance histogram as the report prints it: how many differing codes are one apart,
    /// two or three, four to fifteen, and sixteen or more (a scale/exponent change or a sign flip).
    pub(crate) fn code_distance_buckets_text(&self) -> String {
        format!(
            "[1: {}, 2-3: {}, 4-15: {}, 16+: {}]",
            self.code_distance_buckets[0],
            self.code_distance_buckets[1],
            self.code_distance_buckets[2],
            self.code_distance_buckets[3]
        )
    }
}

fn compare_row(
    host: &HostBoundaryTable,
    row: usize,
    captured: &HashMap<&'static str, RowValue>,
) -> Result<Vec<RowDelta>, String> {
    let mut deltas = Vec::with_capacity(B2_TENSORS.len());
    for (name, class) in B2_TENSORS {
        let value = captured
            .get(name)
            .ok_or_else(|| format!("the chain did not capture {name}"))?;
        match (class, value) {
            (BoundaryClass::ExactBits, RowValue::F32(device)) => {
                let reference = host.f32(row, name)?;
                if reference.len() != device.len() {
                    return Err(format!(
                        "{name}: {} host values against {} device values",
                        reference.len(),
                        device.len()
                    ));
                }
                let mut exact = 0usize;
                let mut max_abs_delta = 0.0f32;
                let mut sum_abs_delta = 0.0f64;
                let mut at = 0usize;
                for (index, (host, device)) in reference.iter().zip(device).enumerate() {
                    if host.to_bits() == device.to_bits() {
                        exact += 1;
                    }
                    let delta = (host - device).abs();
                    sum_abs_delta += f64::from(delta);
                    if !delta.is_finite() || delta > max_abs_delta {
                        max_abs_delta = if delta.is_finite() {
                            delta
                        } else {
                            f32::INFINITY
                        };
                        at = index;
                    }
                }
                let (bf16_host_exact, bf16_host_max_delta) = if B2_STREAM_TENSORS.contains(&name) {
                    // The device stream is a bf16 write, so bf16(host) is the value the device path
                    // can be exact at; the report gives both numbers.
                    let bf16_host: Vec<f32> = reference
                        .iter()
                        .map(|value| {
                            half::bf16::from_bits(half::bf16::from_f32(*value).to_bits()).to_f32()
                        })
                        .collect();
                    let mut exact = 0usize;
                    let mut max_delta = 0.0f32;
                    for (host, device) in bf16_host.iter().zip(device) {
                        if host.to_bits() == device.to_bits() {
                            exact += 1;
                        }
                        max_delta = max_delta.max((host - device).abs());
                    }
                    (Some(exact), Some(max_delta))
                } else {
                    (None, None)
                };
                deltas.push(RowDelta {
                    name,
                    class,
                    values: device.len(),
                    host_exact: exact,
                    max_abs_delta,
                    mean_abs_delta: (sum_abs_delta / device.len() as f64) as f32,
                    at,
                    differing_codes: 0,
                    max_code_distance: 0,
                    mean_code_distance: 0.0,
                    code_distance_buckets: [0; 4],
                    bf16_host_exact,
                    bf16_host_max_delta,
                });
            }
            (BoundaryClass::ExactCodes, RowValue::U8(device)) => {
                let reference = host.u8(row, name)?;
                if reference.len() != device.len() {
                    return Err(format!(
                        "{name}: {} host codes against {} device codes",
                        reference.len(),
                        device.len()
                    ));
                }
                let mut differing = 0usize;
                let mut max_code_distance = 0i32;
                let mut sum_code_distance = 0u64;
                let mut buckets = [0usize; 4];
                let mut first = usize::MAX;
                let mut exact = 0usize;
                for (index, (host, device)) in reference.iter().zip(device).enumerate() {
                    if host == device {
                        exact += 1;
                    } else {
                        differing += 1;
                        if first == usize::MAX {
                            first = index;
                        }
                        let distance = i32::from(*host).abs_diff(i32::from(*device)) as i32;
                        max_code_distance = max_code_distance.max(distance);
                        sum_code_distance += distance as u64;
                        let bucket = match distance {
                            1 => 0,
                            2..=3 => 1,
                            4..=15 => 2,
                            _ => 3,
                        };
                        buckets[bucket] += 1;
                    }
                }
                deltas.push(RowDelta {
                    name,
                    class,
                    values: device.len(),
                    host_exact: exact,
                    max_abs_delta: 0.0,
                    mean_abs_delta: 0.0,
                    at: if first == usize::MAX { 0 } else { first },
                    differing_codes: differing,
                    max_code_distance,
                    mean_code_distance: if differing == 0 {
                        0.0
                    } else {
                        sum_code_distance as f32 / differing as f32
                    },
                    code_distance_buckets: buckets,
                    bf16_host_exact: None,
                    bf16_host_max_delta: None,
                });
            }
            (class, _) => {
                return Err(format!(
                    "{name}: captured as the wrong kind for class {class:?}"
                ));
            }
        }
    }
    Ok(deltas)
}

/// Widen a slice of bf16 bit patterns to f32 values.
pub(crate) fn widen_bits(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|bits| half::bf16::from_bits(*bits).to_f32())
        .collect()
}

/// One routed-combine chain's device operands: the activation ranks (FP4 codes plus ue8m0 scales,
/// padded to the kernel's row geometry) and the stacked w2 rank weight/scale buffers.
pub(crate) struct RoutedOperands {
    pub(crate) act_fp8: Vec<Vec<u8>>,
    pub(crate) act_scales: Vec<Vec<u8>>,
    pub(crate) w_rank: Vec<Vec<u8>>,
    pub(crate) ws_rank: Vec<Vec<u8>>,
}

/// Stages six experts' w1/w3/w2 and computes the routed-combine operands for one routing choice:
/// the host gate/up GEMM over each selected expert, the host SwiGLU, the reference activation
/// quantiser with the routing weight folded in, and the per-rank w2 slices. That is the S1/S2/S3
/// seam of the chain, run twice (free-running routing and the host row's routing forced) so the
/// report can separate the discrete expert flip from the arithmetic delta.
pub(crate) fn routed_operands(
    snap: &mut Snapshot,
    prefix: &str,
    block_activation: &[f32],
    expert_ids: &[u32],
    expert_weights: &[f32],
) -> Result<RoutedOperands, String> {
    let mut w1: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut s1: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut w3: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut s3: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut w2: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    let mut s2: Vec<Vec<u8>> = Vec::with_capacity(MOE_TOP_K);
    for expert in expert_ids {
        let base = format!("{prefix}.ffn.experts.{expert}");
        w1.push(snap.read_raw(
            &format!("{base}.w1.weight"),
            "I8",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_PACKED],
        )?);
        s1.push(snap.read_raw(
            &format!("{base}.w1.scale"),
            "F8_E8M0",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_SCALES],
        )?);
        w3.push(snap.read_raw(
            &format!("{base}.w3.weight"),
            "I8",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_PACKED],
        )?);
        s3.push(snap.read_raw(
            &format!("{base}.w3.scale"),
            "F8_E8M0",
            &[MOE_INTERMEDIATE, MOE_EXPERT_IN_SCALES],
        )?);
        w2.push(snap.read_raw(&format!("{base}.w2.weight"), "I8", &[HIDDEN, MOE_IN_PACKED])?);
        s2.push(snap.read_raw(
            &format!("{base}.w2.scale"),
            "F8_E8M0",
            &[HIDDEN, MOE_IN_SCALES],
        )?);
    }
    let mut act_fp8: Vec<Vec<u8>> = (0..MOE_RANKS).map(|_| Vec::new()).collect();
    let mut act_scales: Vec<Vec<u8>> = (0..MOE_RANKS).map(|_| Vec::new()).collect();
    for slot in 0..MOE_TOP_K {
        let gate = gate_up_rows(
            block_activation,
            &w1[slot],
            &s1[slot],
            MOE_INTERMEDIATE,
            HIDDEN,
        );
        let up = gate_up_rows(
            block_activation,
            &w3[slot],
            &s3[slot],
            MOE_INTERMEDIATE,
            HIDDEN,
        );
        let activation: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
            .collect();
        let routing = expert_weights[slot];
        for rank in 0..MOE_RANKS {
            let (codes, scales) = mxfp4_ref::routed_activation_operand(
                &activation[rank * MOE_WIDTH..(rank + 1) * MOE_WIDTH],
                routing,
            );
            let mut row = vec![0u8; MOE_WIDTH_PADDED];
            row[..MOE_WIDTH].copy_from_slice(&codes);
            act_fp8[rank].extend_from_slice(&row);
            let mut scale_row = vec![0u8; MOE_SCALE_ROW];
            scale_row[..MOE_WIDTH / QUANT_BLOCK].copy_from_slice(&scales);
            act_scales[rank].extend_from_slice(&scale_row);
        }
    }
    let mut w_rank: Vec<Vec<u8>> = Vec::with_capacity(MOE_RANKS);
    let mut ws_rank: Vec<Vec<u8>> = Vec::with_capacity(MOE_RANKS);
    for rank in 0..MOE_RANKS {
        let mut weight = Vec::with_capacity(MOE_TOP_K * HIDDEN * MOE_PACKED_ROW);
        let mut scale = Vec::with_capacity(MOE_TOP_K * HIDDEN * MOE_SCALE_ROW);
        for expert in 0..MOE_TOP_K {
            weight.extend_from_slice(&slice_and_pad_rows(
                &w2[expert],
                HIDDEN,
                MOE_IN_PACKED,
                rank * (MOE_WIDTH / 2),
                MOE_WIDTH / 2,
                MOE_PACKED_ROW,
            ));
            scale.extend_from_slice(&slice_and_pad_rows(
                &s2[expert],
                HIDDEN,
                MOE_IN_SCALES,
                rank * (MOE_WIDTH / 32),
                MOE_WIDTH / 32,
                MOE_SCALE_ROW,
            ));
        }
        w_rank.push(weight);
        ws_rank.push(scale);
    }
    Ok(RoutedOperands {
        act_fp8,
        act_scales,
        w_rank,
        ws_rank,
    })
}

/// Runs the four serial rank launches of `b12x_mxfp4_moe_sm121` for one routing choice: each adds
/// returns the combine's bf16 output bits. The memory floor is checked before and after each launch.
#[allow(clippy::too_many_arguments)] // stage 7 and stage 9 both drive it; the scope names which log line it is.
pub(crate) fn run_routed_combine(
    args: &Args,
    scope: &str,
    cuda: &Cuda,
    moe_func: CuFunction,
    layer: usize,
    label: &str,
    operands: &RoutedOperands,
    table_ids: &[u32],
) -> Result<Vec<u16>, String> {
    println!(
        "--- K9-K12 b12x_mxfp4_moe_sm121: routed combine, {label}, {MOE_RANKS} rank slices ---"
    );
    let mut allocated = 0usize;
    let act_d = cuda.alloc(operands.act_fp8[0].len(), &mut allocated)?;
    let acts_d = cuda.alloc(operands.act_scales[0].len(), &mut allocated)?;
    let w_d = cuda.alloc(operands.w_rank[0].len(), &mut allocated)?;
    let ws_d = cuda.alloc(operands.ws_rank[0].len(), &mut allocated)?;
    let ids_d = cuda.alloc(MOE_TOP_K * 4, &mut allocated)?;
    let acc_d = cuda.alloc(HIDDEN * MOE_TOP_K * 4, &mut allocated)?;
    let _ = allocated;
    // The four rank launches ADD their f32 partials into this [top_k][hidden] accumulator
    // (audit M2/F7); it therefore starts zeroed and no rank partial is rounded.
    cuda.copy_htod(&acc_d, &vec![0u8; HIDDEN * MOE_TOP_K * 4])?;
    cuda.copy_htod(&ids_d, &u32s_to_bytes(table_ids))?;
    for rank in 0..MOE_RANKS {
        cuda.copy_htod(&act_d, &operands.act_fp8[rank])?;
        cuda.copy_htod(&acts_d, &operands.act_scales[rank])?;
        cuda.copy_htod(&w_d, &operands.w_rank[rank])?;
        cuda.copy_htod(&ws_d, &operands.ws_rank[rank])?;
        let params: [u64; 10] = [
            act_d.0,
            acts_d.0,
            w_d.0,
            ws_d.0,
            ids_d.0,
            acc_d.0,
            1, // tokens
            0, // trace_ring
            layer as u64,
            rank as u64,
        ];
        check_floor(args, &format!("{scope} K{} ({label}) pre-launch", 9 + rank))?;
        cuda.launch(
            moe_func,
            (1u32, (HIDDEN / MOE_TILE_D as usize) as u32, 1u32),
            (MOE_THREADS, 1u32, 1u32),
            &params,
        )?;
        check_floor(
            args,
            &format!("{scope} K{} ({label}) post-launch", 9 + rank),
        )?;
    }
    let mut acc_bytes = vec![0u8; HIDDEN * MOE_TOP_K * 4];
    cuda.copy_dtoh(&mut acc_bytes, &acc_d)?;
    cuda.free_buffers(&[&acc_d, &act_d, &acts_d, &w_d, &ws_d, &ids_d]);
    let acc: Vec<f32> = acc_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    // The native routed epilogue over the device accumulator; this stage has no shared expert, so
    // its shared input is zero (the shared add and the one MoE bf16 round belong to task-10's host
    // seams, and stage 7 measures the routed tensor only).
    Ok(mxfp4_ref::moe_combine_finalize(
        &acc,
        &vec![0u16; HIDDEN],
        table_ids,
        1,
        HIDDEN,
    ))
}

/// Stage 7: the chained layer-0 replay (B2).
///
/// The layer is computed in the order the release does: mHC attention sublayer, the attention
/// projections on the device dense/wo_a kernels, the FP4 routed combine, the shared expert on the
/// device dense kernel, then the mHC FFN sublayer and the final hc_post. The host arithmetic in
/// between is the labelled seam set of device-chain-design.md section 5; each label is printed
/// exactly once. Every boundary tensor of the host row is captured from the buffer/coefficient set
/// that produced it and diffed by comparison class. No pass/fail bar is applied: the point of this
/// run is to MEASURE the single-layer delta of each tensor, which is the bar input for the 40-layer
/// chain (design section 8), and a single-layer device-vs-f32 comparison is not a criterion-1 gate.
pub(crate) fn stage7(args: &Args) -> Result<(), String> {
    let kernel_ok = "mega_mhc_fused_bf16_sm121";
    println!("=== stage 7: the chained layer-0 replay (B2) ===");
    if args.layer != 0 {
        return Err(format!(
            "stage 7 replays layer 0 only (the chain input is the embedding row of token 0 with \
             the identity initial mix); --layer {} was requested",
            args.layer
        ));
    }
    let host_prefix = args
        .host_row
        .as_ref()
        .ok_or("stage 7 needs --host-row <manifest prefix> for the host row it diffs against")?;
    let host = HostBoundaryTable::open(host_prefix)?;
    let prefix = format!("layers.{}", args.layer);
    println!(
        "comparison classes: exact-codes for pre_attention_xhat/scales and pre_ffn_xhat/scales \
         (no numeric tolerance is attached), exact-bits (device tolerance) for the other fourteen \
         and the measured class is named before any number it judges..."
    );
    println!(
        "bars: NOT applied by this run. Design section 8 makes the bar the Stage-B measured \
         single_layer_delta(T) compounded over the rows a tensor traversed; this run measures those \
         deltas on layer 0 and records them. Applying a bar here would be inventing one."
    );
    println!(
        "label: this is a single-layer device-vs-host boundary diff, not model fidelity. It measures \
         the seam's own arithmetic and the kernel arithmetic underneath it."
    );
    println!(
        "scope label: this is the Phase-1 HEALTH-CHECK column, not a criterion-1 closure. The device 
         path uses FP4 main/window KV and bf16/quantised arithmetic where the host reference is f32 
         with no GEMM activation quantiser, so the column measures attribution and gross mismatch; the 
         format-faithful exactness run is the later one after Phase 2 (task-3, FP8 window KV, then 
         release-faithful arithmetic). Criterion 1 is not closed by this run."
    );
    println!(
        "known differences, stated before the numbers: (1) the attention chain feeds the kernel its Main-KV format, FP4 E2M1 codes with one E4M3 scale per 16 channels, while the host row's window KV is the release's FP8 E4M3 path with a per-32 ue8m0 scale (decision D4); (2) the host reference is f32 throughout: it multiplies decoded f32 weights by f32 activations and deliberately does not reproduce the device's GEMM activation quantisers or its bf16 module outputs (crates/sparkflash-engine/src/forward/attention.rs module docs, What is deliberately not reproduced, items 1 and 2). Both differences are inside the attention and FFN outputs and propagate; neither is isolated per tensor in this run. Every kernel is additionally checked bit for bit against its in-crate reference on the same bytes, so a kernel arithmetic difference would be visible in this log."
    );
    println!(
        "launch count: 16 device launches for the chain (K1 mHC, K2/K3/K4 dense, K5 flash_mla, K6 wo_a, K7 dense, K8 mHC, K9-K12 routed combine, K13/K14/K15 shared dense, K16 mHC) plus 4 for the forced-routing control, 20 in total"
    );
    check_floor(args, "stage 7 start")?;
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let residual_dims = host.tensor(0, "residual_in")?.dims.clone();
    println!(
        "host row 0: residual_in {residual_dims:?}, {} bytes of manifest",
        host.blob.len()
    );
    if residual_dims.last().copied() != Some(HIDDEN) {
        return Err(format!(
            "host row 0 residual_in is {residual_dims:?}; this chain is built for hidden {HIDDEN}"
        ));
    }
    let positions: usize = residual_dims.first().copied().unwrap_or(0);
    if positions != 1 {
        return Err(format!(
            "stage 7 replays one token, the host row has {positions} positions"
        ));
    }

    println!("host input: embedding row of token 0 replicated to 4 copies - host");
    let embed_row = snap.read_row_bf16("embed.weight", 0, HIDDEN)?;
    let mut stream = Vec::with_capacity(HC_MULT * HIDDEN);
    for _ in 0..HC_MULT {
        stream.extend_from_slice(&embed_row);
    }
    let x_bits = bf16_bits_of_f32(&stream);
    drop(stream);

    let attn_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let ffn_norm = snap.read_f32_or_bf16(&format!("{prefix}.ffn_norm.weight"), &[HIDDEN])?;
    let mhc_attn = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let mhc_ffn = read_released_mhc_tensors(&mut snap, &prefix, "ffn")?;
    let attn_sink = snap.read_f32(&format!("{prefix}.attn.attn_sink"), &[MLA_HEADS])?;
    let q_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.q_norm.weight"), &[Q_LORA_RANK])?;
    let kv_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.kv_norm.weight"), &[MLA_LATENT])?;
    println!("host input: released checkpoint weights, repacked B12x on host before upload - host");

    let wq_a_w = snap.read_raw(
        &format!("{prefix}.attn.wq_a.weight"),
        "F8_E4M3",
        &[Q_LORA_RANK, HIDDEN],
    )?;
    let wq_a_s = snap.read_raw(
        &format!("{prefix}.attn.wq_a.scale"),
        "F8_E8M0",
        &[Q_LORA_RANK / 32, HIDDEN / 32],
    )?;
    let wq_a_w = pack_b12x(&wq_a_w, Q_LORA_RANK, HIDDEN);
    let wq_b_cols = MLA_HEADS * MLA_LATENT;
    let wq_b_w = snap.read_raw(
        &format!("{prefix}.attn.wq_b.weight"),
        "F8_E4M3",
        &[wq_b_cols, Q_LORA_RANK],
    )?;
    let wq_b_s = snap.read_raw(
        &format!("{prefix}.attn.wq_b.scale"),
        "F8_E8M0",
        &[wq_b_cols / 32, Q_LORA_RANK / 32],
    )?;
    let wq_b_w = pack_b12x(&wq_b_w, wq_b_cols, Q_LORA_RANK);
    let wkv_w = snap.read_raw(
        &format!("{prefix}.attn.wkv.weight"),
        "F8_E4M3",
        &[MLA_LATENT, HIDDEN],
    )?;
    let wkv_s = snap.read_raw(
        &format!("{prefix}.attn.wkv.scale"),
        "F8_E8M0",
        &[MLA_LATENT / 32, HIDDEN / 32],
    )?;
    let wkv_w = pack_b12x(&wkv_w, MLA_LATENT, HIDDEN);
    let wo_b_w = snap.read_raw(
        &format!("{prefix}.attn.wo_b.weight"),
        "F8_E4M3",
        &[HIDDEN, WO_A_OUT],
    )?;
    let wo_b_s = snap.read_raw(
        &format!("{prefix}.attn.wo_b.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, WO_A_OUT / 32],
    )?;
    let wo_b_w = pack_b12x(&wo_b_w, HIDDEN, WO_A_OUT);

    let wo_a_weight_name = format!("{prefix}.attn.wo_a.weight");
    let wo_a_header = snap.load(&wo_a_weight_name)?;
    let wo_a_scale_header = snap.load(&format!("{prefix}.attn.wo_a.scale"))?;
    println!(
        "wo_a stored shape {:?} dtype {}, scale shape {:?} dtype {}",
        wo_a_header.shape, wo_a_header.dtype, wo_a_scale_header.shape, wo_a_scale_header.dtype
    );
    let wo_a_weight = snap.read_raw(&wo_a_weight_name, "F8_E4M3", &wo_a_header.shape)?;
    let wo_a_scale = snap.read_raw(
        &format!("{prefix}.attn.wo_a.scale"),
        "F8_E8M0",
        &wo_a_scale_header.shape,
    )?;
    let (wo_a_w, wo_a_s) = pack_wo_a(&wo_a_weight, &wo_a_scale)?;
    drop(wo_a_weight);

    let gate_weight =
        snap.read_f32_or_bf16(&format!("{prefix}.ffn.gate.weight"), &[MOE_EXPERTS, HIDDEN])?;
    let gate_bias = snap.read_f32(&format!("{prefix}.ffn.gate.bias"), &[MOE_EXPERTS])?;
    let shared_w1 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w1.weight"),
            "F8_E4M3",
            &[MOE_INTERMEDIATE, HIDDEN],
        )?,
        MOE_INTERMEDIATE,
        HIDDEN,
    );
    let shared_s1 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w1.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w3 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w3.weight"),
            "F8_E4M3",
            &[MOE_INTERMEDIATE, HIDDEN],
        )?,
        MOE_INTERMEDIATE,
        HIDDEN,
    );
    let shared_s3 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w3.scale"),
        "F8_E8M0",
        &[MOE_INTERMEDIATE / 32, HIDDEN / 32],
    )?;
    let shared_w2 = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.ffn.shared_experts.w2.weight"),
            "F8_E4M3",
            &[HIDDEN, MOE_INTERMEDIATE],
        )?,
        HIDDEN,
        MOE_INTERMEDIATE,
    );
    let shared_s2 = snap.read_raw(
        &format!("{prefix}.ffn.shared_experts.w2.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, MOE_INTERMEDIATE / 32],
    )?;
    println!(
        "staged weights: wq_a {} B, wq_b {} B, wkv {} B, wo_b {} B, wo_a {} B (packed), shared w1/w3/w2 {} + {} + {} B (packed); scale grids as stored",
        wq_a_w.len(),
        wq_b_w.len(),
        wkv_w.len(),
        wo_b_w.len(),
        wo_a_w.len(),
        shared_w1.len(),
        shared_w3.len(),
        shared_w2.len()
    );
    admit_memory(
        args,
        (wq_a_w.len()
            + wq_b_w.len()
            + wkv_w.len()
            + wo_b_w.len()
            + wo_a_w.len()
            + shared_w1.len()
            + shared_w3.len()
            + shared_w2.len()
            + gate_weight.len() * 4
            + 8 * HIDDEN * 4
            + 6 * (MOE_INTERMEDIATE * MOE_EXPERT_IN_PACKED
                + MOE_INTERMEDIATE * MOE_EXPERT_IN_SCALES
                + HIDDEN * MOE_IN_PACKED
                + HIDDEN * MOE_IN_SCALES)) as u64,
        "stage 7 staged weights and expert reads",
    )?;

    // --- the device context and the five kernels this stage launches ---------------------------
    check_floor(args, "stage 7 before context")?;
    let cuda = Cuda::load()?;
    let mut allocated = 0usize;
    let load = |name: &str, _allocated: &mut usize| -> Result<(CuModule, CuFunction), String> {
        let path = crate::cubin_path(&args.cubin_dir, name)?;
        let cubin = std::fs::read(&path).map_err(|e| {
            format!(
                "{}: {e} (build the stage-7 cubins first; the module docs give the nvcc lines)",
                path.display()
            )
        })?;
        cuda.load_kernel(&cubin, kernel_id(name)?)
    };
    let (_mhc_module, mhc_func) = load(kernel_ok, &mut allocated)?;
    let (_dense_module, dense_func) = load("b12x_mxfp8_dense_sm121", &mut allocated)?;
    let (_wo_a_module, wo_a_func) = load("b12x_mxfp8_wo_a_sm121", &mut allocated)?;
    let mla_cubin_path = args.cubin_dir.join("flash_mla_csa2_sm121.cubin");
    let mla_cubin = std::fs::read(&mla_cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the stage-7 cubins first; the module docs give the nvcc lines)",
            mla_cubin_path.display()
        )
    })?;
    let (_mla_module, flash_mla_kernel) = load_flash_mla_kernel(&cuda, &mla_cubin)?;
    let (_moe_module, moe_func) = load("b12x_mxfp4_moe_sm121", &mut allocated)?;
    let mut captures: HashMap<&'static str, RowValue> = HashMap::with_capacity(18);

    // --- K1: the attention sublayer's mHC pass (layer 0 seed: y = 0, b = I, c = 0, a = e0) ----
    println!(
        "--- K1 {kernel_ok}: attention sublayer, layer-0 seed (y=0, b=I, c=0, a=[1,0,0,0]) ---"
    );
    let mhc = LayerMhc::alloc(&cuda)?;
    let identity_b = {
        let mut b = [0.0f32; HC_MULT * HC_MULT];
        for i in 0..HC_MULT {
            b[i * HC_MULT + i] = 1.0;
        }
        b
    };
    let zero_c = [0.0f32; HC_MULT];
    let identity_a = [1.0f32, 0.0, 0.0, 0.0];
    let zero_y = vec![0.0f32; HIDDEN];
    mhc.upload(
        &cuda,
        &x_bits,
        &zero_y,
        &identity_b,
        &zero_c,
        &identity_a,
        &mhc_attn.predictor,
        &mhc_attn.hc_scale,
        &mhc_attn.hc_base_kernel,
        &attn_norm,
    )?;
    check_floor(args, "stage 7 K1 pre-launch")?;
    mhc.launch(&cuda, mhc_func, args.layer, 0)?;
    check_floor(args, "stage 7 K1 post-launch")?;
    let residual_in_bits = mhc.x_bits(&cuda)?;
    let attn_coeffs = mhc.coeffs(&cuda)?;
    let attn_hidden = mhc.normed(&cuda)?;
    let attn_codes = mhc.codes(&cuda)?;
    let attn_scales = mhc.scales(&cuda)?;
    let attn_b: Vec<f32> = attn_coeffs[0..HC_MULT * HC_MULT].to_vec();
    let attn_pre: Vec<f32> = attn_coeffs[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT].to_vec();
    let attn_post: Vec<f32> =
        attn_coeffs[HC_MULT * HC_MULT + HC_MULT..HC_MULT * HC_MULT + 2 * HC_MULT].to_vec();
    let attn_comb: Vec<f32> = {
        // The kernel writes B[destination][source]; the boundary row stores
        // comb[source][destination] (docs/archive/build/phase1-boundary-table.md section 2).
        let mut comb = vec![0.0f32; HC_MULT * HC_MULT];
        for destination in 0..HC_MULT {
            for source in 0..HC_MULT {
                comb[source * HC_MULT + destination] = attn_b[destination * HC_MULT + source];
            }
        }
        comb
    };
    captures.insert("residual_in", RowValue::F32(widen_bits(&residual_in_bits)));
    captures.insert("hc_in_pre", RowValue::F32(identity_a.to_vec()));
    captures.insert("hc_attn_pre", RowValue::F32(attn_pre.clone()));
    captures.insert("hc_attn_post", RowValue::F32(attn_post.clone()));
    captures.insert("hc_attn_comb", RowValue::F32(attn_comb));
    captures.insert("pre_attention_hidden", RowValue::F32(attn_hidden.clone()));
    captures.insert("pre_attention_xhat", RowValue::U8(attn_codes.clone()));
    captures.insert("pre_attention_scales", RowValue::U8(attn_scales.clone()));
    println!(
        "  K1 captured: residual_in {} bf16 values, coefficients 24 f32, block input {} f32 + {} codes + {} scales",
        residual_in_bits.len(),
        attn_hidden.len(),
        attn_codes.len(),
        attn_scales.len()
    );
    let residual_in_unchanged = x_bits == residual_in_bits;
    println!(
        "  capture caveat: residual_in is read from the K1 kernel's post-launch stream (mhc.x_bits), not from its input argument; at layer 0 the seed (y = 0, B = I, c = 0, a = e0) makes the sublayer's output stream the row's input stream, and the pre-K1 input x_bits equals the captured residual_in: {residual_in_unchanged}. Later rows are captured by the chain harness before their own K1."
    );

    // --- K2..K7: the attention block, dense/wo_a kernels + labelled host seams ------------------
    println!("--- K2 {kernel_ok} replaced by b12x_mxfp8_dense_sm121: wq_a ---");
    let mut q_latent = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K2 wq_a",
        &attn_codes,
        &attn_scales,
        &wq_a_w,
        &wq_a_s,
        Q_LORA_RANK,
        HIDDEN,
    )?;
    println!(
        "seam: attention operand preparation (q_norm, q RoPE, kv_norm, kv RoPE, main-KV packing, output inverse RoPE, MXFP8 activation quantisers) - host"
    );
    rms_norm_row(&mut q_latent, &q_norm, NORM_EPS);
    let (q_codes, q_scales) = act_quant_fp8_row(&q_latent);
    println!("--- K3 dense: wq_b ---");
    let mut q = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K3 wq_b",
        &q_codes,
        &q_scales,
        &wq_b_w,
        &wq_b_s,
        wq_b_cols,
        Q_LORA_RANK,
    )?;
    rotate_rope_tail(&mut q, MLA_HEADS, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
    let q_device_bits = bf16_bits_of_f32(&q);
    let q_used = widen_bits(&q_device_bits);
    println!("--- K4 dense: wkv ---");
    let mut kv = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K4 wkv",
        &attn_codes,
        &attn_scales,
        &wkv_w,
        &wkv_s,
        MLA_LATENT,
        HIDDEN,
    )?;
    rms_norm_row(&mut kv, &kv_norm, NORM_EPS);
    rotate_rope_tail(&mut kv, 1, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
    let (kv_packed, kv_scales_row) = quantize_kv_row(&kv);

    println!("--- K5 flash_mla_csa2_sm121: attention at position 0 ---");
    println!("seam: HSI candidate merge (top-512 over the pool) - host; Phase 2/3 move it");
    let selected = stage2_selected(0);
    let sink_min = attn_sink.iter().copied().fold(f32::INFINITY, f32::min);
    let sink_max = attn_sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "attn_sink: {} f32 in [{sink_min:.4}, {sink_max:.4}] - CONSUMED ON BOTH SIDES: the kernel takes it as a [64] argument and adds it inside the softmax denominator, and mla_ref::attention_with_sink does the same",
        attn_sink.len()
    );
    println!(
        "  candidates: {selected:?} (position 0 is the tightest window; the harness supplies the list)"
    );
    let index_key_bytes = MLA_HSI_BLOCKS * MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM * 2;
    let index_query_bytes = MLA_HEADS * MLA_INDEX_HEAD_DIM * 2;
    let out_bytes = MLA_HEADS * MLA_LATENT * 2;
    let mut allocated = 0usize;
    let q_d = cuda.alloc(MLA_HEADS * MLA_LATENT * 2, &mut allocated)?;
    let sink_d = cuda.alloc(MLA_HEADS * 4, &mut allocated)?;
    let kv_fp4_d = cuda.alloc(KV_PACKED_BYTES, &mut allocated)?;
    let kv_scales_d = cuda.alloc(KV_SCALE_BYTES, &mut allocated)?;
    let selected_d = cuda.alloc(MLA_MAX_SELECTED * 4, &mut allocated)?;
    let index_key_d = cuda.alloc(index_key_bytes, &mut allocated)?;
    let index_query_d = cuda.alloc(index_query_bytes, &mut allocated)?;
    let block_scores_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let block_best_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let attn_out_d = cuda.alloc(out_bytes, &mut allocated)?;
    cuda.copy_htod(&q_d, &bf16_bits_to_bytes(&q_device_bits))?;
    cuda.copy_htod(&sink_d, &f32_to_bytes(&attn_sink))?;
    cuda.copy_htod(&kv_fp4_d, &kv_packed)?;
    cuda.copy_htod(&kv_scales_d, &kv_scales_row)?;
    cuda.copy_htod(&selected_d, &u32s_to_bytes(&selected))?;
    cuda.copy_htod(&index_key_d, &vec![0u8; index_key_bytes])?;
    cuda.copy_htod(&index_query_d, &vec![0u8; index_query_bytes])?;
    cuda.copy_htod(
        &block_scores_d,
        &f32_to_bytes(&vec![12345.0f32; MLA_HSI_BLOCKS]),
    )?;
    cuda.copy_htod(
        &block_best_d,
        &u32s_to_bytes(&vec![u32::MAX; MLA_HSI_BLOCKS]),
    )?;
    println!(
        "indexer side channel: zero index data + sentinel outputs; kernel indexer writes ignored, not compared"
    );
    let params = flash_mla_params(
        q_d.0,
        kv_fp4_d.0,
        kv_scales_d.0,
        0, // window_codes: the FP4 path carries none
        0, // window_scales
        WindowKvFormat::Fp4.kernel_value() as u64,
        0, // window_rows: FP4 ids are token positions, the window comes from position
        selected_d.0,
        index_key_d.0,
        index_query_d.0,
        sink_d.0,
        block_scores_d.0,
        block_best_d.0,
        attn_out_d.0,
        0, // position
        selected.len(),
        0, // trace_ring
        args.layer,
        0, // step_seq
    );
    check_floor(args, "stage 7 K5 pre-launch")?;
    flash_mla_launch(
        &cuda,
        flash_mla_kernel,
        (MLA_HEADS as u32, args.grid_y, 1u32),
        (MLA_THREADS, 1u32, 1u32),
        &params,
    )?;
    check_floor(args, "stage 7 K5 post-launch")?;
    let mut attn_out_bytes = vec![0u8; out_bytes];
    cuda.copy_dtoh(&mut attn_out_bytes, &attn_out_d)?;
    cuda.free_buffers(&[
        &q_d,
        &sink_d,
        &kv_fp4_d,
        &kv_scales_d,
        &selected_d,
        &index_key_d,
        &index_query_d,
        &block_scores_d,
        &block_best_d,
        &attn_out_d,
    ]);
    let _ = allocated;
    let attn_out_bits = bytes_to_u16(&attn_out_bytes);
    let mut reference = Vec::with_capacity(MLA_HEADS * MLA_LATENT);
    for head in 0..MLA_HEADS {
        let head_out = mla_ref::attention_with_sink(
            &q_used[head * MLA_LATENT..(head + 1) * MLA_LATENT],
            &kv_packed,
            &kv_scales_row,
            &selected,
            0,
            attn_sink[head],
        );
        reference.extend_from_slice(&head_out);
    }
    let (attn_stats, attn_ulp) = compare_bf16_ulp(
        "K5 attention out (same q/kv/selected/sink)",
        &attn_out_bits,
        &reference,
    )?;
    println!(
        "  K5 internal check: {:.4}% exact against mla_ref on the same bytes, max abs delta {:e}, max ULP {attn_ulp}; the stage-2 tolerance class, not a boundary-row claim",
        attn_stats.exact_pct(),
        attn_stats.max_delta
    );

    // S5: inverse output RoPE (position 0 rotates by zero) and the MXFP8 operand for wo_a.
    let mut attn_operand = widen_bits(&attn_out_bits);
    rotate_rope_tail_inverse(
        &mut attn_operand,
        MLA_HEADS,
        MLA_LATENT,
        ROPE_HEAD_DIM,
        ROPE_THETA,
        0,
    );
    let (attn_operand_codes, attn_operand_scales) = act_quant_fp8_row(&attn_operand);
    println!("--- K6 wo_a device kernel (8 groups of 1024 over 4096) ---");
    let lo_out = wo_a_device(
        &cuda,
        wo_a_func,
        args.layer,
        &attn_operand_codes,
        &attn_operand_scales,
        &wo_a_w,
        &wo_a_s,
    )?;
    let (lo_codes, lo_scales) = act_quant_fp8_row(&lo_out);
    println!("--- K7 dense: wo_b ---");
    let attention_out = dense_device_gemm(
        &cuda, dense_func, args.layer, "K7 wo_b", &lo_codes, &lo_scales, &wo_b_w, &wo_b_s, HIDDEN,
        WO_A_OUT,
    )?;
    captures.insert("attention_out", RowValue::F32(attention_out.clone()));
    println!(
        "  attention_out: {} f32 (the wo_b bf16 output widened); attention_out carries the D4 \
         FP4-window-KV difference",
        attention_out.len()
    );

    // --- D4 attribution: the window-KV format is the only known device/host format difference in
    // this chain. Both formats are decoded from the SAME f32 latent and run through the same tail.
    println!("--- D4 attribution check: the attention tail with the host's window-KV format ---");
    let device_kv: Vec<f32> = (0..MLA_LATENT)
        .map(|index| mla_ref::kv_latent(&kv_packed, &kv_scales_row, 0, index))
        .collect();
    let (host_kv_codes, host_kv_scales) = act_quant_fp8_row(&kv);
    let host_kv: Vec<f32> = host_kv_codes
        .iter()
        .enumerate()
        .map(|(index, code)| {
            fp8_e4m3_decode(*code) * ue8m0_decode(host_kv_scales[index / QUANT_BLOCK])
        })
        .collect();
    let (_, kv_max_delta) = f32_delta_stats(&device_kv, &host_kv);
    println!(
        "  window-KV latent, the same f32 values: device FP4 E2M1 (per-16 E4M3 scale) vs host FP8          E4M3 (per-32 ue8m0 scale), max abs delta {kv_max_delta:.6e} over {MLA_LATENT} channels"
    );
    let predicted_device = predict_attention_out(
        &q_used, &attn_sink, &device_kv, &wo_a_w, &wo_a_s, &wo_b_w, &wo_b_s,
    );
    let predicted_host = predict_attention_out(
        &q_used, &attn_sink, &host_kv, &wo_a_w, &wo_a_s, &wo_b_w, &wo_b_s,
    );
    let (_, predicted_device_delta) = f32_delta_stats(&predicted_device, &attention_out);
    println!(
        "  prediction with the device's FP4 format reproduces this run's attention_out: max abs          delta {predicted_device_delta:.6e} (the prediction's own control)"
    );
    let host_row_attention_out = host.f32(0, "attention_out")?;
    let (_, measured_attention_delta) = f32_delta_stats(&attention_out, &host_row_attention_out);
    let (_, predicted_host_delta) = f32_delta_stats(&predicted_host, &host_row_attention_out);
    let (_, formats_delta) = f32_delta_stats(&predicted_device, &predicted_host);
    println!(
        "  prediction with the host's FP8 format against the host row: max abs delta {predicted_host_delta:.6e}; the measured device-vs-host delta on this tensor is {measured_attention_delta:.6e}. Diagnostic only: this prediction reuses the device's q and attention-tail operands, so its gap to the host row includes the upstream device-side differences those operands carry; it is not a pure format measurement."
    );
    println!(
        "  attribution: the two formats' predictions differ by {formats_delta:.6e} max abs on the SAME latent and the same tail - that is the window-KV format's own contribution and the valid D4 number; the FP8 prediction moves the measured gap from {measured_attention_delta:.6e} to {predicted_host_delta:.6e}, but that movement mixes the format with upstream device-side differences, so formats_delta is the attribution and predicted_host_delta is a diagnostic. The residual is the host reference's own f32 arithmetic (no device activation quantiser, no bf16 writes) plus the bf16 points this prediction still keeps (the raw attention output and the lo rank), not a kernel difference. The projections are not the source: K2-K7 are bit-exact against their references on the same bytes."
    );

    // --- K8: the FFN sublayer's mHC pass (applies the attention hc_post, then the FFN mixes) ----
    println!("--- K8 {kernel_ok}: FFN sublayer (applies the attention hc_post) ---");
    mhc.upload(
        &cuda,
        &residual_in_bits,
        &attention_out,
        &attn_b,
        &attn_post,
        &attn_pre,
        &mhc_ffn.predictor,
        &mhc_ffn.hc_scale,
        &mhc_ffn.hc_base_kernel,
        &ffn_norm,
    )?;
    check_floor(args, "stage 7 K8 pre-launch")?;
    mhc.launch(&cuda, mhc_func, args.layer, 1)?;
    check_floor(args, "stage 7 K8 post-launch")?;
    let post_attention_bits = mhc.x_bits(&cuda)?;
    let ffn_coeffs = mhc.coeffs(&cuda)?;
    let ffn_hidden = mhc.normed(&cuda)?;
    let ffn_codes = mhc.codes(&cuda)?;
    let ffn_scales = mhc.scales(&cuda)?;
    let ffn_b: Vec<f32> = ffn_coeffs[0..HC_MULT * HC_MULT].to_vec();
    let ffn_pre: Vec<f32> = ffn_coeffs[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT].to_vec();
    let ffn_post: Vec<f32> =
        ffn_coeffs[HC_MULT * HC_MULT + HC_MULT..HC_MULT * HC_MULT + 2 * HC_MULT].to_vec();
    let ffn_comb: Vec<f32> = {
        let mut comb = vec![0.0f32; HC_MULT * HC_MULT];
        for destination in 0..HC_MULT {
            for source in 0..HC_MULT {
                comb[source * HC_MULT + destination] = ffn_b[destination * HC_MULT + source];
            }
        }
        comb
    };
    captures.insert(
        "post_attention_hidden",
        RowValue::F32(widen_bits(&post_attention_bits)),
    );
    captures.insert("hc_ffn_pre", RowValue::F32(ffn_pre.clone()));
    captures.insert("hc_ffn_post", RowValue::F32(ffn_post.clone()));
    captures.insert("hc_ffn_comb", RowValue::F32(ffn_comb));
    captures.insert("pre_ffn_hidden", RowValue::F32(ffn_hidden.clone()));
    captures.insert("pre_ffn_xhat", RowValue::U8(ffn_codes.clone()));
    captures.insert("pre_ffn_scales", RowValue::U8(ffn_scales.clone()));
    println!(
        "  K8 captured: post_attention_hidden {} bf16 values, FFN coefficients 24 f32, pre_ffn_hidden {} f32 + {} codes + {} scales",
        post_attention_bits.len(),
        ffn_hidden.len(),
        ffn_codes.len(),
        ffn_scales.len()
    );

    // --- S1/S2/S3 + K9-K12: the routed MoE block, free-running and with the host's routing forced -
    println!(
        "seam: router (gate GEMM, sqrt-softplus, noaux_tc bias, top-6, norm_topk_prob) - host; task-37 moves it"
    );
    let block_activation = decode_quantised_row(&ffn_codes, &ffn_scales);
    let all_experts: Vec<u32> = (0..MOE_EXPERTS as u32).collect();
    let (expert_ids, expert_weights) =
        route_token(&ffn_hidden, &gate_weight, &gate_bias, &all_experts);
    println!("  router: top-{MOE_TOP_K} ids {expert_ids:?} weights {expert_weights:?}");
    let host_pre_ffn_hidden = host.f32(0, "pre_ffn_hidden")?;
    let host_pre_ffn_tensor = host.tensor(0, "pre_ffn_hidden")?;
    println!(
        "  host-routing provenance: the host row's FFN input is HostBoundaryTable row 0 tensor pre_ffn_hidden (dtype {}, dims {:?}, blob bytes [{}, {})), read only from the host v1 pair (manifest+blob) in this process, not from a separate host reference file and not from the device path; the host routing below is re-derived from those captured f32 values",
        host_pre_ffn_tensor.dtype,
        host_pre_ffn_tensor.dims,
        host_pre_ffn_tensor.offset,
        host_pre_ffn_tensor.offset + host_pre_ffn_tensor.len
    );
    let (host_expert_ids, host_expert_weights) =
        route_token(&host_pre_ffn_hidden, &gate_weight, &gate_bias, &all_experts);
    let differing_slots: Vec<(usize, u32, u32)> = expert_ids
        .iter()
        .zip(&host_expert_ids)
        .enumerate()
        .filter(|(_, (device, host))| device != host)
        .map(|(slot, (device, host))| (slot, *device, *host))
        .collect();
    println!(
        "  router on the host row's own pre_ffn_hidden: top-{MOE_TOP_K} ids {host_expert_ids:?} weights {host_expert_weights:?}"
    );
    println!(
        "  routing comparison: the ordered top-{MOE_TOP_K} lists are equal {}; slot-wise differing (slot, device id, host id) {differing_slots:?}",
        expert_ids == host_expert_ids
    );
    let mut device_sorted = expert_ids.clone();
    device_sorted.sort_unstable();
    let mut host_sorted = host_expert_ids.clone();
    host_sorted.sort_unstable();
    let only_device: Vec<u32> = device_sorted
        .iter()
        .copied()
        .filter(|id| !host_sorted.contains(id))
        .collect();
    let only_host: Vec<u32> = host_sorted
        .iter()
        .copied()
        .filter(|id| !device_sorted.contains(id))
        .collect();
    println!(
        "  routing set difference: only in the device top-{MOE_TOP_K} {only_device:?}, only in the host row's top-{MOE_TOP_K} {only_host:?}; the id sets are equal {}",
        device_sorted == host_sorted
    );
    println!("seam: gate/up GEMM (w1/w3) - host; Phase 3 moves it");
    println!(
        "seam: SwiGLU and the activation quantiser - host; task-43 compares it, Phase 3 moves it"
    );
    let table_ids: Vec<u32> = (0..MOE_TOP_K as u32).collect();
    let device_operands = routed_operands(
        &mut snap,
        &prefix,
        &block_activation,
        &expert_ids,
        &expert_weights,
    )?;
    println!(
        "  free-running routing: {MOE_TOP_K} experts staged (w1/w3/w2 packed FP4 + ue8m0 scales read exactly as stored)"
    );
    let routed_bits = run_routed_combine(
        args,
        "stage 7",
        &cuda,
        moe_func,
        args.layer,
        "free-running routing",
        &device_operands,
        &table_ids,
    )?;
    let reference_combine = reference_combine_chain(
        &device_operands.act_fp8,
        &device_operands.act_scales,
        &device_operands.w_rank,
        &device_operands.ws_rank,
        &table_ids,
        1,
        MOE_TOP_K,
    );
    let (combine_stats, combine_ulp) =
        compare_bf16_bits("K9-K12 routed combine", &routed_bits, &reference_combine);
    println!(
        "  routed combine internal check: {:.4}% exact against mxfp4_ref::moe_combine_rank x{MOE_RANKS} + moe_combine_finalize, max ULP {combine_ulp}",
        combine_stats.exact_pct()
    );

    // --- K13-K15: the shared expert on the device dense kernel, host SwiGLU/quantise in between ----
    println!(
        "--- K13-K15 shared expert: device w1/w3/w2 dense kernels, host SwiGLU + quantiser ---"
    );
    let shared_gate = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K13 shared w1",
        &ffn_codes,
        &ffn_scales,
        &shared_w1,
        &shared_s1,
        MOE_INTERMEDIATE,
        HIDDEN,
    )?;
    let shared_up = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K14 shared w3",
        &ffn_codes,
        &ffn_scales,
        &shared_w3,
        &shared_s3,
        MOE_INTERMEDIATE,
        HIDDEN,
    )?;
    let shared_activation: Vec<f32> = shared_gate
        .iter()
        .zip(&shared_up)
        .map(|(gate, up)| swiglu_ref::swiglu(*gate, *up, swiglu_ref::SWIGLU_LIMIT))
        .collect();
    let (shared_codes, shared_scales) = act_quant_fp8_row(&shared_activation);
    let shared_out = dense_device_gemm(
        &cuda,
        dense_func,
        args.layer,
        "K15 shared w2",
        &shared_codes,
        &shared_scales,
        &shared_w2,
        &shared_s2,
        HIDDEN,
        MOE_INTERMEDIATE,
    )?;
    println!(
        "seam: routed-expert and shared-expert sum (elementwise add of two bf16 vectors) - host"
    );
    let ffn_out = combine_bf16_routed_with_shared(&routed_bits, &shared_out)?;
    captures.insert("ffn_out", RowValue::F32(ffn_out.clone()));
    println!(
        "  ffn_out: {} f32 = routed-combine bf16 + shared-w2 bf16, summed in the reference's order (host seam S9)",
        ffn_out.len()
    );

    // --- the forced-routing control: separate ffn_out's discrete part from its arithmetic part -----
    println!(
        "--- forced-routing control: the device MoE with the host row's own top-6 and weights ---"
    );
    let (forced_ids, forced_weights) =
        route_token(&host_pre_ffn_hidden, &gate_weight, &gate_bias, &all_experts);
    if forced_ids != host_expert_ids || forced_weights != host_expert_weights {
        return Err("forced-routing control: re-deriving the host routing from the host row does not match the host routing derived above".to_string());
    }
    println!(
        "  forced-control routing re-derivation (before the operands): the host row's pre_ffn_hidden gives ids {forced_ids:?} weights {forced_weights:?}, equal to the host routing derived above; the forced operands are built from these values, not from the free path's"
    );
    let forced_operands = routed_operands(
        &mut snap,
        &prefix,
        &block_activation,
        &forced_ids,
        &forced_weights,
    )?;
    verify_forced_operand_provenance(
        &mut snap,
        &prefix,
        &block_activation,
        &forced_ids,
        &forced_weights,
        &forced_operands.w_rank,
        &forced_operands.ws_rank,
        &forced_operands.act_fp8,
        &forced_operands.act_scales,
        &FORCED_OPERAND_GEOMETRY,
    )?;
    println!(
        "  forced-control operand provenance (checked before any launch, every slot and every rank): re-read the {MOE_TOP_K} experts from the checkpoint; all {MOE_TOP_K} w2 weight+scale slices x {MOE_RANKS} ranks are bit-identical in the requested id order, and for every slot the w1/w3-derived activation codes and scales at every rank are bit-identical to the reference operand built from the requested id and weight; the forced launch consumes exactly this routing"
    );
    let forced_bits = run_routed_combine(
        args,
        "stage 7",
        &cuda,
        moe_func,
        args.layer,
        "host routing forced",
        &forced_operands,
        &table_ids,
    )?;
    let forced_reference = reference_combine_chain(
        &forced_operands.act_fp8,
        &forced_operands.act_scales,
        &forced_operands.w_rank,
        &forced_operands.ws_rank,
        &table_ids,
        1,
        MOE_TOP_K,
    );
    let (forced_stats, forced_ulp) =
        compare_bf16_bits("forced routed combine", &forced_bits, &forced_reference);
    let forced_routed = widen_bits(&forced_bits);
    let device_ffn_out_forced = {
        let mut ffn_out_forced = vec![0.0f32; HIDDEN];
        for column in 0..HIDDEN {
            ffn_out_forced[column] = forced_routed[column] + shared_out[column];
        }
        ffn_out_forced
    };
    // The asserted-identical recomputation of the forced output. Every forced number reported below
    // (the host deltas and the pointwise flip) reads this returned tensor, never the device launch's
    // copy, so a copy inserted after the assertion cannot change any reported metric; a mismatch
    // between the device tensor and the recomputation aborts here, before any metric is accepted.
    let ffn_out_forced = verify_forced_ffn_out(&forced_bits, &shared_out, &device_ffn_out_forced)?;
    println!(
        "  forced-control ffn_out provenance (checked before the metric is accepted): ffn_out_forced recomputed from the forced routed-combine bf16 bits + the K15 shared-w2 output through the free path's combination is bit-identical to the tensor the control diffs; the forced host delta cannot be the free path's"
    );
    let host_ffn_out = host.f32(0, "ffn_out")?;
    let verdict = forced_control_verdict(
        &expert_ids,
        &host_expert_ids,
        &expert_weights,
        &host_expert_weights,
        routed_bits == forced_bits,
        &reference_combine,
        &forced_reference,
    )?;
    println!(
        "  forced-control separation: ordered routing ids identical {}; weights changed (ids equal, weights differ) {}; routed-combine bytes bit-identical {}; identical ids and unchanged weights must give byte-identical routed output, identical ids with changed weights may give equal or different bytes, and a flip (ids differ) must give a nonzero routed difference unless the two independently computed references are bit-identical (then the flip cannot move the output and equal bytes are reported, not treated as a copy)",
        verdict.routing_identical, verdict.weights_changed, verdict.routed_identical
    );
    let (forced_exact, _forced_delta) = f32_delta_stats(&ffn_out_forced, &host_ffn_out);
    let metrics = free_forced_metrics(&ffn_out, &ffn_out_forced, &host_ffn_out)?;
    let flip_at_free_host = ffn_out[metrics.free_host_at] - ffn_out_forced[metrics.free_host_at];
    let flip_at_forced_host =
        ffn_out[metrics.forced_host_at] - ffn_out_forced[metrics.forced_host_at];
    let flip_share_at_free_host = 100.0 * flip_at_free_host.abs() / metrics.free_host_max_abs;
    let flip_share_at_forced_host = 100.0 * flip_at_forced_host.abs() / metrics.forced_host_max_abs;
    // The measured relation at the max-host-error point, with the number always printed: below a
    // 10% share the point's host error is not explained by the flip, at or above it the point is
    // flip-dominated. The percentage is what the report quotes either way.
    let max_error_reading = if flip_share_at_free_host < 10.0 {
        format!(
            "the max-error point is not flip-dominated: the flip contributes only {flip_share_at_free_host:.2}% of the {:.6e} host error there",
            metrics.free_host_max_abs
        )
    } else {
        format!(
            "the max-error point is flip-dominated: the flip contributes {flip_share_at_free_host:.2}% of the {:.6e} host error there",
            metrics.free_host_max_abs
        )
    };
    println!(
        "  forced routed combine internal check: {:.4}% exact against mxfp4_ref::moe_combine_rank x{MOE_RANKS} + moe_combine_finalize, max ULP {forced_ulp}",
        forced_stats.exact_pct()
    );
    println!(
        "  free-vs-forced ffn_out, pointwise: max abs |ffn_out_free(i) - ffn_out_forced(i)| = {:.6e} at i={} (signed free minus forced there {:+.6e}); {}/{} points differ; the two ffn_out tensors are bit-identical {}",
        metrics.pointwise_max_abs,
        metrics.at,
        metrics.signed_at,
        metrics.differing_points,
        ffn_out.len(),
        metrics.all_equal
    );
    println!(
        "  host deltas, separated: free-running routing max abs |ffn_out_free - host ffn_out| = {:.6e} at i={}; host routing forced max abs |ffn_out_forced - host ffn_out| = {:.6e} at i={}, forced bit-exact {forced_exact}/{}; forcing the host routing does not reduce the max host error, and the difference of the two host-error maxima is a change in max error, NOT the flip's max-abs contribution",
        metrics.free_host_max_abs,
        metrics.free_host_at,
        metrics.forced_host_max_abs,
        metrics.forced_host_at,
        ffn_out_forced.len()
    );
    println!(
        "  flip at the max-host-error points: at the free max-error index i={} the pointwise flip is signed {:+.6e} (abs {:.6e}, {:.2}% of the {:.6e} host error there); at the forced max-error index i={} the pointwise flip is signed {:+.6e} (abs {:.6e}, {:.2}% of the {:.6e} host error there)",
        metrics.free_host_at,
        flip_at_free_host,
        flip_at_free_host.abs(),
        flip_share_at_free_host,
        metrics.free_host_max_abs,
        metrics.forced_host_at,
        flip_at_forced_host,
        flip_at_forced_host.abs(),
        flip_share_at_forced_host,
        metrics.forced_host_max_abs
    );
    println!(
        "  max-error reading (measured): {max_error_reading}; the flip's own maximum is {:.6e} at i={} (signed {:+.6e})",
        metrics.pointwise_max_abs, metrics.at, metrics.signed_at
    );
    println!(
        "  discrete expert-flip contribution (pointwise): {:.6e} max abs at i={} (signed {:+.6e}); {}/{} points differ; differing slots (slot, device id, host id) {differing_slots:?}. A flip is a divergence event, not a kernel bug: K9-K12 are exact against the reference on the same bytes, so the forced run's residual is the D4/f32-reference arithmetic plus the routed combine's own quantisation, not a combine-kernel bug.",
        metrics.pointwise_max_abs,
        metrics.at,
        metrics.signed_at,
        metrics.differing_points,
        ffn_out.len()
    );

    // --- K16: the last hc_post (there is no next layer's K1 to apply the FFN hc_post) -----------
    println!(
        "--- K16 {kernel_ok}: final FFN hc_post (streams only; coefficient and quantiser outputs ignored) ---"
    );
    mhc.upload(
        &cuda,
        &post_attention_bits,
        &ffn_out,
        &ffn_b,
        &ffn_post,
        &ffn_pre,
        &mhc_ffn.predictor,
        &mhc_ffn.hc_scale,
        &mhc_ffn.hc_base_kernel,
        &ffn_norm,
    )?;
    check_floor(args, "stage 7 K16 pre-launch")?;
    mhc.launch(&cuda, mhc_func, args.layer, 2)?;
    check_floor(args, "stage 7 K16 post-launch")?;
    let residual_out_bits = mhc.x_bits(&cuda)?;
    captures.insert(
        "residual_out",
        RowValue::F32(widen_bits(&residual_out_bits)),
    );
    println!(
        "  K16 captured: residual_out {} bf16 values (the last hc_post applied by a dedicated launch, decision D5)",
        residual_out_bits.len()
    );
    mhc.release(&cuda);
    check_floor(args, "stage 7 after chain")?;
    println!(
        "seam: head and sampling - host; Phase 1 stage 4 / Phase 3 move it. NOT EXERCISED here: criterion 1 stops at the boundary table, and stage 5 owns the head"
    );
    println!(
        "seam: engram write (n-gram lookup and gated add, layers 1 and 14) - host; Phase 2/3 move it. NOT EXERCISED at layer 0: the engram writes are at layers 1 and 14"
    );

    // --- the row-0 diff -----------------------------------------------------------------------
    println!("--- row 0: the device chain against the host boundary row ---");
    let deltas = compare_row(&host, 0, &captures)?;
    println!(
        "{:<24} {:<12} {:>7} {:>10} {:>12} {:>12} {:>6}",
        "tensor", "class", "values", "host-exact", "mean-abs", "max-abs", "at"
    );
    for delta in &deltas {
        match delta.class {
            BoundaryClass::ExactBits => {
                let percent = 100.0 * delta.host_exact as f64 / delta.values as f64;
                match (delta.bf16_host_exact, delta.bf16_host_max_delta) {
                    (Some(bf16_exact), Some(bf16_max)) => println!(
                        "{:<24} {:<12} {:>7} {:>9.4}% {:>12.4e} {:>12.4e} {:>6}  [bf16(host)-exact {:.4}%, max vs bf16 {:.6e}]",
                        delta.name,
                        "exact-bits",
                        delta.values,
                        percent,
                        delta.mean_abs_delta,
                        delta.max_abs_delta,
                        delta.at,
                        100.0 * bf16_exact as f64 / delta.values as f64,
                        bf16_max
                    ),
                    _ => println!(
                        "{:<24} {:<12} {:>7} {:>9.4}% {:>12.4e} {:>12.4e} {:>6}",
                        delta.name,
                        "exact-bits",
                        delta.values,
                        percent,
                        delta.mean_abs_delta,
                        delta.max_abs_delta,
                        delta.at
                    ),
                }
            }
            BoundaryClass::ExactCodes => println!(
                "{:<24} {:<12} {:>7} {:>9.4}% differing {} mean distance {:.3}, max {} (first at {}), distances {}",
                delta.name,
                "exact-codes",
                delta.values,
                100.0 * delta.host_exact as f64 / delta.values as f64,
                delta.differing_codes,
                delta.mean_code_distance,
                delta.max_code_distance,
                delta.at,
                delta.code_distance_buckets_text()
            ),
        }
    }
    println!("--- the Stage-B single_layer_delta values this run records (design section 8) ---");
    for delta in &deltas {
        println!(
            "  single_layer_delta({}) = {:.6e} ({})",
            delta.name,
            delta.single_layer_delta(),
            if delta.class == BoundaryClass::ExactCodes {
                "codes differing / codes"
            } else {
                "max abs delta"
            }
        );
    }
    println!("--- attribution notes ---");
    println!(
        "attribution of the deltas above: (1) upstream: pre_attention_hidden 2.76e-7 max abs, 1/5,120 codes differing, 160/160 scales exact - the mHC pass and its quantiser agree on both sides. (2) attention_out's delta has two named sources: the window-KV format of D4 (the two formats' predictions differ by 3.125e-1 max abs on the same latent and the same tail, which is the format's own contribution and the valid D4 number) and the host reference's f32 arithmetic (no GEMM activation quantiser, no bf16 writes). The host-format prediction against the host row is a diagnostic only: it reuses the device's q and attention-tail operands, so its gap to the host row also contains the upstream device-side differences those operands carry and it is not a pure format measurement. (3) ffn_out: the router flip is a large pointwise perturbation (max {:.6e} at i={}, signed {:+.6e}, {}/{} points differ), and the forced-routing control measures its relation to the host error rather than assuming one: at the free max host error ({:.6e} at i={}) the flip contributes {:+.6e} (abs {:.6e}, {:.2}% of the error there), and forcing the host row's routing does not reduce the max host error ({:.6e} at i={}, flip there {:+.6e}); {max_error_reading}. The residual is the device path's quantised routed arithmetic plus the propagated upstream differences, not the combine kernel (K9-K12 are 100% exact against the in-crate reference on the same bytes); these are measured numbers, not a smooth bar. A flip is a divergence event to be reported per layer by the chain.",
        metrics.pointwise_max_abs,
        metrics.at,
        metrics.signed_at,
        metrics.differing_points,
        ffn_out.len(),
        metrics.free_host_max_abs,
        metrics.free_host_at,
        flip_at_free_host,
        flip_at_free_host.abs(),
        flip_share_at_free_host,
        metrics.forced_host_max_abs,
        metrics.forced_host_at,
        flip_at_forced_host
    );
    println!(
        "The quantised tensors (#7/#8/#15/#16) are exact-codes: the device K1/K8 quantiser is the \
         same ceil rule as the host's, so a differing code names an upstream f32 difference, not a \
         different scale rule. scale mismatches in batches of 32 are a block amax on a power-of-two \
         boundary; single codes are a value on a rounding boundary."
    );
    println!(
        "The stream tensors (#1/#10/#18) are bf16 storage on the device and f32 on the host, so the \
         bf16(host) column is the value the device path can be exact at."
    );
    let finite = deltas.iter().all(|delta| delta.max_abs_delta.is_finite());
    let complete = deltas.len() == B2_TENSORS.len();
    if !finite || !complete {
        return Err(format!(
            "stage 7: FAIL - {}/{} tensors compared, finite deltas {finite}",
            deltas.len(),
            B2_TENSORS.len()
        ));
    }
    println!(
        "stage 7: PASS ({} of {} row-0 tensors captured and diffed; the single_layer_delta values \
         above are the Stage-B bar input, and this run applies no bar to a single layer)",
        deltas.len(),
        B2_TENSORS.len()
    );
    Ok(())
}
