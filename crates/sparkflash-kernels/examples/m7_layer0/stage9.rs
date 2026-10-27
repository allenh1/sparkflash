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
use crate::stage7::*;
use crate::stage8::*;

// ---------------------------------------------------------------------------------------------
// Stage 9: the 40-layer chained device forward (Stage C of device-chain-design.md section 9).
//
// One released-checkpoint prefill step at position 0, one token (id 0), 40 backbone rows: the same
// chain stage 7 runs for layer 0, looped over the backbone, carrying the device residual streams,
// the previous layer's FFN output and its kernel-order B/A/C coefficient triples between layers.
// The next layer's K1 performs the previous layer's last hc_post, so 15 launches per layer plus the
// final K16; the two Engram layers (1 and 14) need their own K16 before the write and a re-run of K1
// with the identity seed afterwards, because the released Engram write mutates the stream between
// those layers. Every row is captured into the same 18 tensors stage 7 captures and diffed against
// the host v1 pair by comparison class, with the Stage-B single-layer deltas compounded as the bars
// (design section 8). The run writes its own v1 pair so examples/boundary_diff.rs can diff it
// unchanged, and it runs the chain twice to print determinism.
//
// This is the Phase-1 HEALTH-CHECK column, not a criterion-1 closure: the device path uses FP4
// main/window KV and bf16/quantised arithmetic where the host reference is f32, so the column
// measures attribution and gross mismatch. The format-faithful exactness run is the later one after
// Phase 2 (task-3). Criterion 1 is not closed by this run.
// ---------------------------------------------------------------------------------------------

/// The Stage-B single-layer deltas in effect as the Stage-C bar inputs, copied from the frozen
/// Stage-B evidence of record
/// (`/home/user/phase1-evidence/phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log`,
/// lines 149-166, binary `2b518a35...`). The values are MEASURED, not chosen: the log records them
/// before Stage C exists, and this run compounds them per design section 8.
const STAGE9_B2_DELTAS: [(&str, f64); 18] = [
    ("residual_in", 0.0),
    ("hc_in_pre", 0.0),
    ("hc_attn_pre", 8.493_662e-7),
    ("hc_attn_post", 3.278_255e-7),
    ("hc_attn_comb", 3.129_244e-7),
    ("pre_attention_hidden", 2.756_715e-7),
    ("pre_attention_xhat", 1.953_125e-4),
    ("pre_attention_scales", 0.0),
    ("attention_out", 2.565_311e-1),
    ("post_attention_hidden", 7.431_363e-2),
    ("hc_ffn_pre", 4.267_210e-2),
    ("hc_ffn_post", 7.206_082e-3),
    ("hc_ffn_comb", 1.175_366e-3),
    ("pre_ffn_hidden", 3.163_002e-2),
    ("pre_ffn_xhat", 6.882_812e-1),
    ("pre_ffn_scales", 3.75e-2),
    ("ffn_out", 2.150_760e-1),
    ("residual_out", 1.810_353e-1),
];

/// One Stage-B single-layer delta by boundary tensor name.
fn stage9_b2_delta(name: &str) -> f64 {
    STAGE9_B2_DELTAS
        .iter()
        .find(|(tensor, _)| *tensor == name)
        .map(|(_, delta)| *delta)
        .unwrap_or(f64::NAN)
}

/// The Stage-C bar for one tensor of one row, exactly as the log prints it before judging anything.
///
/// The pre-registration of design section 8 is `bar(row L, tensor T) = single_layer_delta(T) *
/// (L + 1)` for the stream-carrying tensors (#1, #10, #18) and `single_layer_delta(T)` for the
/// per-layer tensors. Two B2 deltas are zero by construction at layer 0 because their layer-0 value
/// is the exact seed (the embedding stream and the one-hot input mix); row L>0 inherits the delta of
/// the tensor that produces it (residual_out for residual_in, hc_ffn_pre for hc_in_pre), so the bar
/// uses that measured predecessor instead of a vacuous zero. The substitution is printed in the
/// bar table, never applied silently.
fn stage9_bar(name: &str, row: usize) -> f64 {
    let measured = match name {
        "residual_in" => stage9_b2_delta("residual_out"),
        "hc_in_pre" => stage9_b2_delta("hc_ffn_pre"),
        other => stage9_b2_delta(other),
    };
    let factor = if B2_STREAM_TENSORS.contains(&name) {
        (row + 1) as f64
    } else {
        1.0
    };
    measured * factor
}

/// The shape of one boundary tensor, in the v1 manifest's comma-separated form.
fn stage9_dims(name: &str) -> Vec<usize> {
    match name {
        "residual_in" | "post_attention_hidden" | "residual_out" => vec![1, HC_MULT, HIDDEN],
        "hc_in_pre" | "hc_attn_pre" | "hc_attn_post" | "hc_ffn_pre" | "hc_ffn_post" => {
            vec![1, HC_MULT]
        }
        "hc_attn_comb" | "hc_ffn_comb" => vec![1, HC_MULT, HC_MULT],
        "pre_attention_xhat" | "pre_ffn_xhat" => vec![1, HIDDEN],
        "pre_attention_scales" | "pre_ffn_scales" => vec![1, HIDDEN / QUANT_BLOCK],
        _ => vec![1, HIDDEN],
    }
}

/// The v1 dtype token of one boundary tensor name (docs/archive/build/phase1-boundary-table.md section 3).
fn stage9_v1_dtype(name: &str) -> &'static str {
    match name {
        "pre_attention_xhat" | "pre_ffn_xhat" => "u8-e4m3",
        "pre_attention_scales" | "pre_ffn_scales" => "u8-e8m0",
        _ => "f32",
    }
}

/// Build one device v1 pair in memory: the manifest text and the blob.
///
/// The writer is deliberately the same shape as the engine's `BoundaryTable::write_v1` (header lines,
/// `row N` markers, one tensor line per tensor in TENSOR_ORDER order, blob in the same order with no
/// padding). The harness cannot construct an engine `BoundaryTable` from outside the engine crate,
/// so it writes the format directly and then proves the writer on the released host pair: for a
/// 40-row run the device manifest must be byte-identical to `phase1-host-token0.manifest`.
fn stage9_build_v1(rows: &[Stage9Row]) -> Result<(String, Vec<u8>), String> {
    let mut manifest = String::new();
    let mut blob: Vec<u8> = Vec::new();
    manifest.push_str("# phase1-boundary v1\n");
    manifest.push_str("step 0 1 0\n");
    manifest.push_str("prefix 0\n");
    manifest.push_str("mtp-anchor none\n");
    manifest.push_str(&format!("backbone {}\n", rows.len()));
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != B2_TENSORS.len() {
            return Err(format!(
                "row {row_index} has {} tensors, the format has {}",
                row.len(),
                B2_TENSORS.len()
            ));
        }
        manifest.push_str(&format!("row {row_index}\n"));
        for ((name, _class), (captured_name, value)) in B2_TENSORS.iter().zip(row) {
            if *captured_name != *name {
                return Err(format!(
                    "row {row_index}: captured {captured_name} where {name} belongs"
                ));
            }
            let dims = stage9_dims(name);
            let declared: usize = dims.iter().product();
            let bytes: Vec<u8> = match value {
                RowValue::F32(values) => {
                    if values.len() != declared {
                        return Err(format!(
                            "row {row_index} {name}: {} f32 values for shape {dims:?}",
                            values.len()
                        ));
                    }
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect()
                }
                RowValue::U8(codes) => {
                    if codes.len() != declared {
                        return Err(format!(
                            "row {row_index} {name}: {} bytes for shape {dims:?}",
                            codes.len()
                        ));
                    }
                    codes.clone()
                }
            };
            let dims_text = dims
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let offset = blob.len();
            manifest.push_str(&format!(
                "tensor {row_index} {name} {} {dims_text} {offset} {}\n",
                stage9_v1_dtype(name),
                bytes.len()
            ));
            blob.extend_from_slice(&bytes);
        }
    }
    manifest.push_str("end\n");
    Ok((manifest, blob))
}

/// sha256 of one byte slice, hex, for the determinism line.
fn stage9_sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Everything one layer's device run reads from the released checkpoint, staged once per layer and
/// dropped before the next layer so the resident set does not grow with the layer index.
struct Stage9Weights {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mhc_attn: ReleasedMhcTensors,
    mhc_ffn: ReleasedMhcTensors,
    attn_sink: Vec<f32>,
    q_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    wq_a_w: Vec<u8>,
    wq_a_s: Vec<u8>,
    wq_b_w: Vec<u8>,
    wq_b_s: Vec<u8>,
    wkv_w: Vec<u8>,
    wkv_s: Vec<u8>,
    wo_b_w: Vec<u8>,
    wo_b_s: Vec<u8>,
    wo_a_w: Vec<u8>,
    wo_a_s: Vec<u8>,
    gate_weight: Vec<f32>,
    gate_bias: Vec<f32>,
    shared_w1: Vec<u8>,
    shared_s1: Vec<u8>,
    shared_w3: Vec<u8>,
    shared_s3: Vec<u8>,
    shared_w2: Vec<u8>,
    shared_s2: Vec<u8>,
}

impl Stage9Weights {
    fn staged_bytes(&self) -> usize {
        self.wq_a_w.len()
            + self.wq_a_s.len()
            + self.wq_b_w.len()
            + self.wq_b_s.len()
            + self.wkv_w.len()
            + self.wkv_s.len()
            + self.wo_b_w.len()
            + self.wo_b_s.len()
            + self.wo_a_w.len()
            + self.wo_a_s.len()
            + self.shared_w1.len()
            + self.shared_s1.len()
            + self.shared_w3.len()
            + self.shared_s3.len()
            + self.shared_w2.len()
            + self.shared_s2.len()
            + self.gate_weight.len() * 4
            + self.mhc_attn.predictor.len() * 4
            + self.mhc_ffn.predictor.len() * 4
    }
}

/// Reads one layer's released tensors in the same order and with the same assertions as stage 7.
fn stage9_load_weights(snap: &mut Snapshot, prefix: &str) -> Result<Stage9Weights, String> {
    let attn_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let ffn_norm = snap.read_f32_or_bf16(&format!("{prefix}.ffn_norm.weight"), &[HIDDEN])?;
    let mhc_attn = read_released_mhc_tensors(snap, prefix, "attn")?;
    let mhc_ffn = read_released_mhc_tensors(snap, prefix, "ffn")?;
    let attn_sink = snap.read_f32(&format!("{prefix}.attn.attn_sink"), &[MLA_HEADS])?;
    let q_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.q_norm.weight"), &[Q_LORA_RANK])?;
    let kv_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.kv_norm.weight"), &[MLA_LATENT])?;
    let wq_a_w = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.attn.wq_a.weight"),
            "F8_E4M3",
            &[Q_LORA_RANK, HIDDEN],
        )?,
        Q_LORA_RANK,
        HIDDEN,
    );
    let wq_a_s = snap.read_raw(
        &format!("{prefix}.attn.wq_a.scale"),
        "F8_E8M0",
        &[Q_LORA_RANK / 32, HIDDEN / 32],
    )?;
    let wq_b_cols = MLA_HEADS * MLA_LATENT;
    let wq_b_w = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.attn.wq_b.weight"),
            "F8_E4M3",
            &[wq_b_cols, Q_LORA_RANK],
        )?,
        wq_b_cols,
        Q_LORA_RANK,
    );
    let wq_b_s = snap.read_raw(
        &format!("{prefix}.attn.wq_b.scale"),
        "F8_E8M0",
        &[wq_b_cols / 32, Q_LORA_RANK / 32],
    )?;
    let wkv_w = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.attn.wkv.weight"),
            "F8_E4M3",
            &[MLA_LATENT, HIDDEN],
        )?,
        MLA_LATENT,
        HIDDEN,
    );
    let wkv_s = snap.read_raw(
        &format!("{prefix}.attn.wkv.scale"),
        "F8_E8M0",
        &[MLA_LATENT / 32, HIDDEN / 32],
    )?;
    let wo_b_w = pack_b12x(
        &snap.read_raw(
            &format!("{prefix}.attn.wo_b.weight"),
            "F8_E4M3",
            &[HIDDEN, WO_A_OUT],
        )?,
        HIDDEN,
        WO_A_OUT,
    );
    let wo_b_s = snap.read_raw(
        &format!("{prefix}.attn.wo_b.scale"),
        "F8_E8M0",
        &[HIDDEN / 32, WO_A_OUT / 32],
    )?;
    let wo_a_weight_name = format!("{prefix}.attn.wo_a.weight");
    let wo_a_scale_name = format!("{prefix}.attn.wo_a.scale");
    let wo_a_weight_header = snap.load(&wo_a_weight_name)?;
    let wo_a_scale_header = snap.load(&wo_a_scale_name)?;
    let wo_a_weight = snap.read_raw(&wo_a_weight_name, "F8_E4M3", &wo_a_weight_header.shape)?;
    let wo_a_scale = snap.read_raw(&wo_a_scale_name, "F8_E8M0", &wo_a_scale_header.shape)?;
    let (wo_a_w, wo_a_s) = pack_wo_a(&wo_a_weight, &wo_a_scale)?;
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
    Ok(Stage9Weights {
        attn_norm,
        ffn_norm,
        mhc_attn,
        mhc_ffn,
        attn_sink,
        q_norm,
        kv_norm,
        wq_a_w,
        wq_a_s,
        wq_b_w,
        wq_b_s,
        wkv_w,
        wkv_s,
        wo_b_w,
        wo_b_s,
        wo_a_w,
        wo_a_s,
        gate_weight,
        gate_bias,
        shared_w1,
        shared_s1,
        shared_w3,
        shared_s3,
        shared_w2,
        shared_s2,
    })
}

/// The device context and the five loaded kernels a Stage-C run launches.
struct Stage9Device {
    cuda: Cuda,
    _modules: Vec<CuModule>,
    mhc: LayerMhc,
    mhc_func: CuFunction,
    dense_func: CuFunction,
    wo_a_func: CuFunction,
    flash_mla_kernel: FlashMlaKernel,
    moe_func: CuFunction,
}

fn stage9_device(args: &Args) -> Result<Stage9Device, String> {
    check_floor(args, "stage 9 before context")?;
    let cuda = Cuda::load()?;
    let mut modules = Vec::with_capacity(5);
    let mut load = |name: &str| -> Result<CuFunction, String> {
        let path = crate::cubin_path(&args.cubin_dir, name)?;
        let cubin = std::fs::read(&path).map_err(|e| {
            format!(
                "{}: {e} (build the stage-7/9 cubins first; the module docs give the nvcc lines)",
                path.display()
            )
        })?;
        let (module, function) = cuda.load_kernel(&cubin, kernel_id(name)?)?;
        modules.push(module);
        Ok(function)
    };
    let mhc_func = load("mega_mhc_fused_bf16_sm121")?;
    let dense_func = load("b12x_mxfp8_dense_sm121")?;
    let wo_a_func = load("b12x_mxfp8_wo_a_sm121")?;
    let mla_cubin_path = args.cubin_dir.join("flash_mla_csa2_sm121.cubin");
    let mla_cubin = std::fs::read(&mla_cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the stage-7/9 cubins first; the module docs give the nvcc lines)",
            mla_cubin_path.display()
        )
    })?;
    let (_mla_module, flash_mla_kernel) = load_flash_mla_kernel(&cuda, &mla_cubin)?;
    let moe_func = load("b12x_mxfp4_moe_sm121")?;
    let mhc = LayerMhc::alloc(&cuda)?;
    Ok(Stage9Device {
        cuda,
        _modules: modules,
        mhc,
        mhc_func,
        dense_func,
        wo_a_func,
        flash_mla_kernel,
        moe_func,
    })
}

/// K2-K7 for one layer: the attention projections and the flash_mla attention, on device, with the
/// same host seams stage 7 uses. Returns attention_out (the wo_b output widened from bf16).
#[allow(clippy::too_many_arguments)] // the kernel launches take the layer's staged tensors as distinct arguments.
fn stage9_attention_block(
    args: &Args,
    dev: &Stage9Device,
    layer: usize,
    w: &Stage9Weights,
    attn_codes: &[u8],
    attn_scales: &[u8],
    format_stats: &mut EngineWindowStats,
) -> Result<Vec<f32>, String> {
    let cuda = &dev.cuda;
    let tag = format!("L{layer:02}");
    let wq_b_cols = MLA_HEADS * MLA_LATENT;
    let mut q_latent = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K2 wq_a"),
        attn_codes,
        attn_scales,
        &w.wq_a_w,
        &w.wq_a_s,
        Q_LORA_RANK,
        HIDDEN,
    )?;
    rms_norm_row(&mut q_latent, &w.q_norm, NORM_EPS);
    let (q_codes, q_scales) = act_quant_fp8_row(&q_latent);
    let mut q = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K3 wq_b"),
        &q_codes,
        &q_scales,
        &w.wq_b_w,
        &w.wq_b_s,
        wq_b_cols,
        Q_LORA_RANK,
    )?;
    rotate_rope_tail(&mut q, MLA_HEADS, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
    let q_device_bits = bf16_bits_of_f32(&q);
    let q_used = widen_bits(&q_device_bits);
    let mut kv = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K4 wkv"),
        attn_codes,
        attn_scales,
        &w.wkv_w,
        &w.wkv_s,
        MLA_LATENT,
        HIDDEN,
    )?;
    rms_norm_row(&mut kv, &w.kv_norm, NORM_EPS);
    rotate_rope_tail(&mut kv, 1, MLA_LATENT, ROPE_HEAD_DIM, ROPE_THETA, 0);
    let (kv_packed, kv_scales_row) = quantize_kv_row(&kv);
    // The engine-format window row, built by the same helper stage 2 uses: rms_norm and the rope
    // tail are already applied above, then exactly one bf16 storage round, then the engine's
    // released quantiser, asserted against m4_quant::act_quant_fp8_row on the same bf16 values.
    let format = args.stage9_kv_format;
    let mut window_codes = Vec::new();
    let mut window_scales = Vec::new();
    if format == WindowKvFormat::Engine {
        let engine_row =
            engine_window_row(&kv).map_err(|error| format!("stage 9 layer {layer}: {error}"))?;
        assert_engine_window_finite(
            &engine_row.codes,
            &engine_row.scales,
            1,
            &format!("stage 9 layer {layer}"),
        )?;
        format_stats.rows += 1;
        format_stats.quantiser_agreements += 1;
        format_stats.guarded_rows += 1;
        window_codes = engine_row.codes;
        window_scales = engine_row.scales;
    }

    // K5: the attention kernel, one launch. The candidate list is the single position 0: this is a
    // first prefill step, so the window holds only position 0 and there is no compressed row yet
    // (the compressor cache is empty at position 0) - the S7 seam is exercised in its degenerate
    // form and the log says so rather than implying a compressed candidate was attended.
    let selected = stage2_selected(0);
    let index_key_bytes = MLA_HSI_BLOCKS * MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM * 2;
    let index_query_bytes = MLA_HEADS * MLA_INDEX_HEAD_DIM * 2;
    let out_bytes = MLA_HEADS * MLA_LATENT * 2;
    let mut allocated = 0usize;
    let q_d = cuda.alloc(MLA_HEADS * MLA_LATENT * 2, &mut allocated)?;
    let sink_d = cuda.alloc(MLA_HEADS * 4, &mut allocated)?;
    let kv_fp4_d = cuda.alloc(KV_PACKED_BYTES, &mut allocated)?;
    let kv_scales_d = cuda.alloc(KV_SCALE_BYTES, &mut allocated)?;
    let window_codes_d = if format == WindowKvFormat::Engine {
        Some(cuda.alloc(MLA_LATENT, &mut allocated)?)
    } else {
        None
    };
    let window_scales_d = if format == WindowKvFormat::Engine {
        Some(cuda.alloc(WINDOW_SCALE_BYTES, &mut allocated)?)
    } else {
        None
    };
    let selected_d = cuda.alloc(MLA_MAX_SELECTED * 4, &mut allocated)?;
    let index_key_d = cuda.alloc(index_key_bytes, &mut allocated)?;
    let index_query_d = cuda.alloc(index_query_bytes, &mut allocated)?;
    let block_scores_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let block_best_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let attn_out_d = cuda.alloc(out_bytes, &mut allocated)?;
    cuda.copy_htod(&q_d, &bf16_bits_to_bytes(&q_device_bits))?;
    cuda.copy_htod(&sink_d, &f32_to_bytes(&w.attn_sink))?;
    cuda.copy_htod(&kv_fp4_d, &kv_packed)?;
    cuda.copy_htod(&kv_scales_d, &kv_scales_row)?;
    if let (Some(codes_d), Some(scales_d)) = (window_codes_d.as_ref(), window_scales_d.as_ref()) {
        cuda.copy_htod(codes_d, &window_codes)?;
        cuda.copy_htod(scales_d, &window_scales)?;
    }
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
    let params = flash_mla_params(
        q_d.0,
        kv_fp4_d.0,
        kv_scales_d.0,
        window_codes_d.as_ref().map_or(0, |buffer| buffer.0),
        window_scales_d.as_ref().map_or(0, |buffer| buffer.0),
        format.kernel_value() as u64,
        // one stored window row per layer in engine mode; FP4 ids are token positions
        if format == WindowKvFormat::Engine {
            1
        } else {
            0
        },
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
        layer,
        0, // step_seq
    );
    check_floor(args, &format!("stage 9 layer {layer} K5 pre-launch"))?;
    flash_mla_launch(
        &dev.cuda,
        dev.flash_mla_kernel,
        (MLA_HEADS as u32, args.grid_y, 1u32),
        (MLA_THREADS, 1u32, 1u32),
        &params,
    )?;
    check_floor(args, &format!("stage 9 layer {layer} K5 post-launch"))?;
    let mut attn_out_bytes = vec![0u8; out_bytes];
    cuda.copy_dtoh(&mut attn_out_bytes, &attn_out_d)?;
    let mut free_list: Vec<&Deviceptr> = vec![
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
    ];
    if let (Some(codes_d), Some(scales_d)) = (window_codes_d.as_ref(), window_scales_d.as_ref()) {
        free_list.push(codes_d);
        free_list.push(scales_d);
    }
    cuda.free_buffers(&free_list);
    let _ = allocated;
    let attn_out_bits = bytes_to_u16(&attn_out_bytes);
    let mut reference = Vec::with_capacity(MLA_HEADS * MLA_LATENT);
    for head in 0..MLA_HEADS {
        let q_head = &q_used[head * MLA_LATENT..(head + 1) * MLA_LATENT];
        // The reference is format-dispatched exactly like stage 2: engine mode must read the same
        // window codes/scales with mla_ref::attention_with_sink_window (the engine decode plus
        // bf16 round), never the FP4 reference with window bytes.
        let head_out = if format == WindowKvFormat::Engine {
            mla_ref::attention_with_sink_window(
                q_head,
                &window_codes,
                &window_scales,
                1,
                &selected,
                w.attn_sink[head],
            )
        } else {
            mla_ref::attention_with_sink(
                q_head,
                &kv_packed,
                &kv_scales_row,
                &selected,
                0,
                w.attn_sink[head],
            )
        };
        reference.extend_from_slice(&head_out);
    }
    let (attn_stats, attn_ulp) = compare_bf16_ulp(
        &if format == WindowKvFormat::Engine {
            format!(
                "L{layer:02} K5 attention out (engine window format, window_rows=1, same bytes)"
            )
        } else {
            format!("L{layer:02} K5 attention out (same q/kv/selected/sink)")
        },
        &attn_out_bits,
        &reference,
    )?;
    if format == WindowKvFormat::Engine {
        println!(
            "  {tag} K5 internal check (engine window format, window_rows=1, selected=[0]): {:.4}% exact against mla_ref::attention_with_sink_window on the same bytes, max abs delta {:e}, max ULP {attn_ulp}",
            attn_stats.exact_pct(),
            attn_stats.max_delta
        );
    } else {
        println!(
            "  {tag} K5 internal check: {:.4}% exact against mla_ref on the same bytes, max abs delta {:e}, max ULP {attn_ulp}",
            attn_stats.exact_pct(),
            attn_stats.max_delta
        );
    }

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
    let lo_out = wo_a_device(
        cuda,
        dev.wo_a_func,
        layer,
        &attn_operand_codes,
        &attn_operand_scales,
        &w.wo_a_w,
        &w.wo_a_s,
    )?;
    let (lo_codes, lo_scales) = act_quant_fp8_row(&lo_out);
    let attention_out = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K7 wo_b"),
        &lo_codes,
        &lo_scales,
        &w.wo_b_w,
        &w.wo_b_s,
        HIDDEN,
        WO_A_OUT,
    )?;
    Ok(attention_out)
}

/// What one layer's MoE block produced, for the per-row report.
struct Stage9FfnResult {
    ffn_out: Vec<f32>,
    device_ids: Vec<u32>,
    host_ids: Vec<u32>,
    free_delta: f32,
    forced_delta: f32,
}

/// S7-S10 and K9-K15 for one layer: the router seam, the free-running routed combine, the forced
/// control, the shared expert, and the host sum that produces ffn_out. Both ffn_out tensors come
/// from the shared B2 helpers (`combine_bf16_routed_with_shared`, and `verify_forced_ffn_out`
/// whose returned recomputation feeds the free/forced host deltas and the pointwise flip metric),
/// the forced operands are verified against the checkpoint before the forced launch
/// (`verify_forced_operand_provenance` with `FORCED_OPERAND_GEOMETRY`), and
/// `forced_control_verdict` enforces the routing/bytes determinism rules.
#[allow(clippy::too_many_arguments)] // mirrors stage 7's seam functions: context plus the layer's tensors.
fn stage9_ffn_block(
    args: &Args,
    dev: &Stage9Device,
    layer: usize,
    w: &Stage9Weights,
    snap: &mut Snapshot,
    prefix: &str,
    ffn_hidden: &[f32],
    ffn_codes: &[u8],
    ffn_scales: &[u8],
    host_ffn_hidden: &[f32],
    host_ffn_out: &[f32],
) -> Result<Stage9FfnResult, String> {
    let cuda = &dev.cuda;
    let tag = format!("L{layer:02}");
    let block_activation = decode_quantised_row(ffn_codes, ffn_scales);
    let all_experts: Vec<u32> = (0..MOE_EXPERTS as u32).collect();
    let (device_ids, device_weights) =
        route_token(ffn_hidden, &w.gate_weight, &w.gate_bias, &all_experts);
    let (host_ids, host_weights) =
        route_token(host_ffn_hidden, &w.gate_weight, &w.gate_bias, &all_experts);
    let differing_slots: Vec<(usize, u32, u32)> = device_ids
        .iter()
        .zip(&host_ids)
        .enumerate()
        .filter(|(_, (device, host))| device != host)
        .map(|(slot, (device, host))| (slot, *device, *host))
        .collect();
    let mut device_sorted = device_ids.clone();
    device_sorted.sort_unstable();
    let mut host_sorted = host_ids.clone();
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
    let table_ids: Vec<u32> = (0..MOE_TOP_K as u32).collect();
    let free_operands = routed_operands(
        snap,
        prefix,
        &block_activation,
        &device_ids,
        &device_weights,
    )?;
    let routed_bits = run_routed_combine(
        args,
        "stage 9",
        cuda,
        dev.moe_func,
        layer,
        "free-running routing",
        &free_operands,
        &table_ids,
    )?;
    let free_reference = reference_combine_chain(
        &free_operands.act_fp8,
        &free_operands.act_scales,
        &free_operands.w_rank,
        &free_operands.ws_rank,
        &table_ids,
        1,
        MOE_TOP_K,
    );
    let (free_stats, free_ulp) =
        compare_bf16_bits("free routed combine", &routed_bits, &free_reference);
    if !free_stats.all_exact() {
        println!(
            "  {tag} WARNING routed combine internal check {:.4}% exact, max ULP {free_ulp}",
            free_stats.exact_pct()
        );
    }

    let shared_gate = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K13 shared w1"),
        ffn_codes,
        ffn_scales,
        &w.shared_w1,
        &w.shared_s1,
        MOE_INTERMEDIATE,
        HIDDEN,
    )?;
    let shared_up = dense_device_gemm(
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K14 shared w3"),
        ffn_codes,
        ffn_scales,
        &w.shared_w3,
        &w.shared_s3,
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
        cuda,
        dev.dense_func,
        layer,
        &format!("{tag} K15 shared w2"),
        &shared_codes,
        &shared_scales,
        &w.shared_w2,
        &w.shared_s2,
        HIDDEN,
        MOE_INTERMEDIATE,
    )?;
    let ffn_out = combine_bf16_routed_with_shared(&routed_bits, &shared_out)?;
    let forced_operands =
        routed_operands(snap, prefix, &block_activation, &host_ids, &host_weights)?;
    // The forced launch consumes the host row's own routing, so the operands are verified against
    // the checkpoint before that launch exactly as stage 7 does: every rank's w2 weight/scale slice
    // and every slot's w1/w3-derived activation codes and scales.
    verify_forced_operand_provenance(
        snap,
        prefix,
        &block_activation,
        &host_ids,
        &host_weights,
        &forced_operands.w_rank,
        &forced_operands.ws_rank,
        &forced_operands.act_fp8,
        &forced_operands.act_scales,
        &FORCED_OPERAND_GEOMETRY,
    )?;
    let forced_bits = run_routed_combine(
        args,
        "stage 9",
        cuda,
        dev.moe_func,
        layer,
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
    if !forced_stats.all_exact() {
        println!(
            "  {tag} WARNING forced routed combine internal check {:.4}% exact, max ULP {forced_ulp}",
            forced_stats.exact_pct()
        );
    }
    let device_ffn_out_forced = {
        let forced_routed = widen_bits(&forced_bits);
        let mut ffn_out_forced = vec![0.0f32; HIDDEN];
        for column in 0..HIDDEN {
            ffn_out_forced[column] = forced_routed[column] + shared_out[column];
        }
        ffn_out_forced
    };
    // The asserted-identical recomputation of the forced output: every forced number reported below
    // (the host deltas and the pointwise flip) reads this returned tensor, never the device launch's
    // copy, so a copy inserted after the assertion cannot change any reported metric; a mismatch
    // between the device tensor and the recomputation aborts here, before any metric is accepted.
    let ffn_out_forced = verify_forced_ffn_out(&forced_bits, &shared_out, &device_ffn_out_forced)?;
    let verdict = forced_control_verdict(
        &device_ids,
        &host_ids,
        &device_weights,
        &host_weights,
        routed_bits == forced_bits,
        &free_reference,
        &forced_reference,
    )?;
    let metrics = free_forced_metrics(&ffn_out, &ffn_out_forced, host_ffn_out)?;
    let free_delta = metrics.free_host_max_abs;
    let forced_delta = metrics.forced_host_max_abs;
    println!(
        "  {tag} router: ordered top-{MOE_TOP_K} lists equal {}; differing slots (slot, device, host) {differing_slots:?}; set difference only-device {only_device:?} only-host {only_host:?}{}",
        device_ids == host_ids,
        if device_ids != host_ids {
            " - DISCRETE EXPERT FLIP: a divergence event to report per layer, not a smooth bar"
        } else {
            ""
        }
    );
    println!(
        "  {tag} forced control: ids equal {}, weights changed {}, routed bytes identical {}, references bit-identical {} (an ids-differ + bytes-identical outcome is allowed only when the references are bit-identical: the flip cannot move the output by construction); pointwise |ffn_out free - ffn_out forced| max {:.6e} at i={} (signed {:+.6e}), {}/{} points differ; host deltas: free-running {:.6e} at i={}, forced {:.6e} at i={}; the difference of the two host-error maxima is a change in max error, NOT the flip's max-abs contribution; routed-combine exactness: free {:.2}%, forced {:.2}%",
        verdict.routing_identical,
        verdict.weights_changed,
        verdict.routed_identical,
        verdict.references_identical,
        metrics.pointwise_max_abs,
        metrics.at,
        metrics.signed_at,
        metrics.differing_points,
        ffn_out.len(),
        free_delta,
        metrics.free_host_at,
        forced_delta,
        metrics.forced_host_at,
        free_stats.exact_pct(),
        forced_stats.exact_pct(),
    );
    Ok(Stage9FfnResult {
        ffn_out,
        device_ids,
        host_ids,
        free_delta,
        forced_delta,
    })
}

/// The released Engram write before an Engram layer, applied on the host seam to the device's bf16
/// residual stream: widen the device stream to f32, run the engine's released Engram arithmetic on
/// it, round back to bf16 for the device. The host pair applied the same write to its own f32
/// stream, so the seam's arithmetic is the same code and the difference is the bf16 stream.
fn stage9_engram_seam(
    cfg: &ModelConfig,
    source: &dyn WeightSource,
    hashes: &[u32],
    layer: usize,
    x_bits: &[u16],
    limit: usize,
) -> Result<Vec<u16>, String> {
    let mut stream = widen_bits(x_bits);
    let before = stream.clone();
    let written = sparkflash_engine::forward::engram::engram_forward(
        cfg,
        layer,
        &mut stream,
        hashes,
        source,
        limit,
    )
    .map_err(|error| format!("engram layer {layer}: {error}"))?;
    if !written {
        return Err(format!(
            "engram layer {layer}: the configuration says this layer has no Engram table"
        ));
    }
    let mut max_abs = 0.0f32;
    let mut changed = 0usize;
    for (old, new) in before.iter().zip(&stream) {
        if old.to_bits() != new.to_bits() {
            changed += 1;
        }
        max_abs = max_abs.max((new - old).abs());
    }
    println!(
        "  engram seam layer {layer}: host write of {} values, {changed}/{} changed, max abs write {max_abs:.6e}; the device stream is bf16 so the write is rounded back to bf16 before the layer's K1",
        stream.len(),
        stream.len()
    );
    Ok(bf16_bits_of_f32(&stream))
}

/// One row's measured-vs-bar report. Returns (tensors at or under the bar, tensors over it, the
/// worst measured/bar ratio) and prints one line per tensor. The bar comes from the frozen Stage-B
/// measurements, never from this run's own numbers.
fn stage9_report_row(
    host: &HostBoundaryTable,
    row: usize,
    captures: &HashMap<&'static str, RowValue>,
) -> Result<(usize, usize, f64), String> {
    let mut passed = 0usize;
    let mut misses = 0usize;
    let mut worst_ratio = 0.0f64;
    for (name, class) in B2_TENSORS {
        let value = captures
            .get(name)
            .ok_or_else(|| format!("row {row} did not capture {name}"))?;
        let bar = stage9_bar(name, row);
        let (measured, detail) = match (class, value) {
            (BoundaryClass::ExactBits, RowValue::F32(device)) => {
                let reference = host.f32(row, name)?;
                if reference.len() != device.len() {
                    return Err(format!(
                        "row {row} {name}: {} host values against {} device values",
                        reference.len(),
                        device.len()
                    ));
                }
                let mut exact = 0usize;
                let mut max_abs = 0.0f32;
                for (host_value, device_value) in reference.iter().zip(device) {
                    if host_value.to_bits() == device_value.to_bits() {
                        exact += 1;
                    }
                    max_abs = max_abs.max((host_value - device_value).abs());
                }
                let percent = 100.0 * exact as f64 / device.len() as f64;
                (
                    f64::from(max_abs),
                    format!("host-exact {percent:>9.4}% max-abs {max_abs:.6e}"),
                )
            }
            (BoundaryClass::ExactCodes, RowValue::U8(device)) => {
                let reference = host.u8(row, name)?;
                if reference.len() != device.len() {
                    return Err(format!(
                        "row {row} {name}: {} host codes against {} device codes",
                        reference.len(),
                        device.len()
                    ));
                }
                let mut differing = 0usize;
                let mut max_code_distance = 0i32;
                for (host_code, device_code) in reference.iter().zip(device) {
                    if host_code != device_code {
                        differing += 1;
                        let distance = i32::from(*host_code) - i32::from(*device_code);
                        max_code_distance = max_code_distance.max(distance.abs());
                    }
                }
                let fraction = differing as f64 / device.len() as f64;
                (
                    fraction,
                    format!(
                        "differing-codes {differing:>5}/{:<5} max-code-distance {max_code_distance}",
                        device.len()
                    ),
                )
            }
            _ => {
                return Err(format!(
                    "row {row} {name}: the captured value has the wrong class"
                ));
            }
        };
        let ratio = if bar > 0.0 {
            measured / bar
        } else if measured == 0.0 {
            1.0
        } else {
            f64::INFINITY
        };
        worst_ratio = worst_ratio.max(ratio);
        let verdict = if measured <= bar {
            passed += 1;
            "PASS"
        } else if bar > 0.0 && measured <= bar * (1.0 + 1e-6) {
            // The frozen Stage-B log prints its deltas to 7 significant digits, so a measured f32
            // can sit a few 1e-7 relative above a bar that IS that same measurement rounded on
            // paper. That is the bar's own print precision, not headroom.
            passed += 1;
            "AT-BAR-PRECISION"
        } else {
            misses += 1;
            "BELOW-BAR"
        };
        println!(
            "  row {row:>2} {name:<24} {:<5} {detail} bar {bar:.6e} {verdict}",
            if class == BoundaryClass::ExactCodes {
                "codes"
            } else {
                "f32"
            }
        );
    }
    Ok((passed, misses, worst_ratio))
}

/// The per-layer state the chain carries between layers: the previous FFN sublayer's inputs, which
/// the next layer's K1 consumes as its own step 1. The kernel order is kept here so the carry never
/// round-trips through the boundary-row orientation.
struct Stage9Carry {
    post_attention_bits: Vec<u16>,
    ffn_out: Vec<f32>,
    ffn_b: Vec<f32>,
    ffn_post: Vec<f32>,
    ffn_pre: Vec<f32>,
}

/// One chain run's summary.
struct Stage9Summary {
    rows: usize,
    passes: usize,
    misses: usize,
    over_rows: Vec<usize>,
    worst_ratio: f64,
    flips: Vec<usize>,
    free_ffn_max: f32,
    forced_ffn_max: f32,
}

/// One captured boundary row: the 18 (tensor name, value) pairs in the format's tensor order.
type Stage9Row = Vec<(&'static str, RowValue)>;

/// One chain run: every captured row, the tensor names the reader was asked for, and its
/// (reads, bytes) audit.
type Stage9Run = (Vec<Stage9Row>, Vec<String>, (usize, u64));

/// Runs the whole chain once and returns its captured rows (complete per layer), the tensor names
/// the run read from the released checkpoint, and the read counts.
#[allow(clippy::too_many_arguments)] // the run needs the loaders, the host pair, the device context and the released config.
fn stage9_chain(
    args: &Args,
    dev: &Stage9Device,
    host: &HostBoundaryTable,
    cfg: &ModelConfig,
    engram_source: &dyn WeightSource,
    hashes: &[u32],
    n_rows: usize,
    run_index: usize,
    format_stats: &mut EngineWindowStats,
) -> Result<Stage9Run, String> {
    let mut snap = Snapshot::open(&args.checkpoint)?;
    let mut rows: Vec<Stage9Row> = Vec::with_capacity(n_rows);
    let mut prev_captures: Option<HashMap<&'static str, RowValue>> = None;
    let mut carry: Option<Stage9Carry> = None;
    let mut pending_x: Option<Vec<u16>> = None;
    let mut summary = Stage9Summary {
        rows: n_rows,
        passes: 0,
        misses: 0,
        over_rows: Vec::new(),
        worst_ratio: 0.0,
        flips: Vec::new(),
        free_ffn_max: 0.0,
        forced_ffn_max: 0.0,
    };
    let identity_b = {
        let mut b = [0.0f32; HC_MULT * HC_MULT];
        for index in 0..HC_MULT {
            b[index * HC_MULT + index] = 1.0;
        }
        b
    };
    let zero_c = [0.0f32; HC_MULT];
    let identity_a = [1.0f32, 0.0, 0.0, 0.0];
    let zero_y = vec![0.0f32; HIDDEN];
    // The layer-0 seed: the embedding row of token 0 replicated to the four streams.
    let embed_row = snap.read_row_bf16("embed.weight", 0, HIDDEN)?;
    let mut stream = Vec::with_capacity(HC_MULT * HIDDEN);
    for _ in 0..HC_MULT {
        stream.extend_from_slice(&embed_row);
    }
    let x_bits = bf16_bits_of_f32(&stream);
    drop(stream);
    let engram_limit = usize::try_from(sparkflash_engine::safetensors::DEFAULT_LIMIT)
        .map_err(|_| "the engine's default tensor limit does not fit a usize".to_string())?;
    let engram_at = |layer: usize| cfg.engram_rows(layer).is_some();

    println!();
    println!("=== chain run {run_index}: {n_rows} rows ===");
    for layer in 0..n_rows {
        let prefix = format!("layers.{layer}");
        let weights = stage9_load_weights(&mut snap, &prefix)?;
        println!(
            "--- layer {layer:02}: staged {} B of released weights, {} rows to run ---",
            weights.staged_bytes(),
            n_rows
        );
        let mut captures: HashMap<&'static str, RowValue> = HashMap::with_capacity(18);

        // --- K1: the attention sublayer. The seed is the previous FFN hc_post; at layer 0 it is the
        // embedding with the identity mix, and after an Engram layer the stream already carries the
        // released write, so K1 runs with the identity seed to leave it in place.
        let (seed_x, seed_y, seed_b, seed_c, seed_a) = if layer == 0 {
            (
                x_bits.clone(),
                zero_y.clone(),
                identity_b.to_vec(),
                zero_c.to_vec(),
                identity_a.to_vec(),
            )
        } else if let Some(engrammed) = pending_x.take() {
            let carry = carry
                .as_ref()
                .ok_or("stage 9: an Engram layer has no previous layer's coefficients")?;
            (
                engrammed,
                zero_y.clone(),
                identity_b.to_vec(),
                zero_c.to_vec(),
                carry.ffn_pre.clone(),
            )
        } else {
            let carry = carry
                .as_ref()
                .ok_or("stage 9: a layer past layer 0 has no previous layer to carry from")?;
            (
                carry.post_attention_bits.clone(),
                carry.ffn_out.clone(),
                carry.ffn_b.clone(),
                carry.ffn_post.clone(),
                carry.ffn_pre.clone(),
            )
        };
        dev.mhc.upload(
            &dev.cuda,
            &seed_x,
            &seed_y,
            &seed_b,
            &seed_c,
            &seed_a,
            &weights.mhc_attn.predictor,
            &weights.mhc_attn.hc_scale,
            &weights.mhc_attn.hc_base_kernel,
            &weights.attn_norm,
        )?;
        check_floor(args, &format!("stage 9 layer {layer} K1 pre-launch"))?;
        dev.mhc.launch(&dev.cuda, dev.mhc_func, layer, 0)?;
        check_floor(args, &format!("stage 9 layer {layer} K1 post-launch"))?;
        let residual_in_bits = dev.mhc.x_bits(&dev.cuda)?;
        let attn_coeffs = dev.mhc.coeffs(&dev.cuda)?;
        let attn_hidden = dev.mhc.normed(&dev.cuda)?;
        let attn_codes = dev.mhc.codes(&dev.cuda)?;
        let attn_scales = dev.mhc.scales(&dev.cuda)?;
        let attn_b: Vec<f32> = attn_coeffs[0..HC_MULT * HC_MULT].to_vec();
        let attn_pre: Vec<f32> =
            attn_coeffs[HC_MULT * HC_MULT..HC_MULT * HC_MULT + HC_MULT].to_vec();
        let attn_post: Vec<f32> =
            attn_coeffs[HC_MULT * HC_MULT + HC_MULT..HC_MULT * HC_MULT + 2 * HC_MULT].to_vec();
        let attn_comb: Vec<f32> = {
            let mut comb = vec![0.0f32; HC_MULT * HC_MULT];
            for destination in 0..HC_MULT {
                for source in 0..HC_MULT {
                    comb[source * HC_MULT + destination] = attn_b[destination * HC_MULT + source];
                }
            }
            comb
        };
        captures.insert("residual_in", RowValue::F32(widen_bits(&residual_in_bits)));
        captures.insert("hc_in_pre", RowValue::F32(seed_a.clone()));
        captures.insert("hc_attn_pre", RowValue::F32(attn_pre.clone()));
        captures.insert("hc_attn_post", RowValue::F32(attn_post.clone()));
        captures.insert("hc_attn_comb", RowValue::F32(attn_comb));
        captures.insert("pre_attention_hidden", RowValue::F32(attn_hidden.clone()));
        captures.insert("pre_attention_xhat", RowValue::U8(attn_codes.clone()));
        captures.insert("pre_attention_scales", RowValue::U8(attn_scales.clone()));
        println!(
            "  L{layer:02} K1 captured: residual_in {} bf16 values, coefficients 24 f32, block input {} f32 + {} codes + {} scales{}",
            residual_in_bits.len(),
            attn_hidden.len(),
            attn_codes.len(),
            attn_scales.len(),
            if layer > 0 && !engram_at(layer) {
                " (this K1 also wrote the previous row's residual_out)"
            } else {
                ""
            }
        );

        // This K1 wrote the previous row's residual_out when no Engram write sits between them.
        if layer > 0 && !engram_at(layer) {
            let mut previous = prev_captures.take().ok_or_else(|| {
                format!("stage 9: layer {layer} K1 has no previous row to finish")
            })?;
            previous.insert("residual_out", RowValue::F32(widen_bits(&residual_in_bits)));
            let row = layer - 1;
            let (passed, misses, worst_ratio) = stage9_report_row(host, row, &previous)?;
            println!(
                "  row {row:>2} summary: {passed}/18 at or under the Stage-B bar, {misses} over it, worst measured/bar {worst_ratio:.3}"
            );
            summary.passes += passed;
            summary.misses += misses;
            summary.worst_ratio = summary.worst_ratio.max(worst_ratio);
            if misses > 0 {
                summary.over_rows.push(row);
            }
            let mut ordered = Vec::with_capacity(B2_TENSORS.len());
            for (name, _) in B2_TENSORS {
                ordered.push((
                    name,
                    previous
                        .remove(name)
                        .ok_or_else(|| format!("row {row}: captured set lost {name}"))?,
                ));
            }
            rows.push(ordered);
        }

        // --- K2-K7: the attention block on the device.
        let attention_out = stage9_attention_block(
            args,
            dev,
            layer,
            &weights,
            &attn_codes,
            &attn_scales,
            format_stats,
        )?;
        captures.insert("attention_out", RowValue::F32(attention_out.clone()));

        // --- K8: the FFN sublayer's mHC pass (applies the attention hc_post).
        dev.mhc.upload(
            &dev.cuda,
            &residual_in_bits,
            &attention_out,
            &attn_b,
            &attn_post,
            &attn_pre,
            &weights.mhc_ffn.predictor,
            &weights.mhc_ffn.hc_scale,
            &weights.mhc_ffn.hc_base_kernel,
            &weights.ffn_norm,
        )?;
        check_floor(args, &format!("stage 9 layer {layer} K8 pre-launch"))?;
        dev.mhc.launch(&dev.cuda, dev.mhc_func, layer, 1)?;
        check_floor(args, &format!("stage 9 layer {layer} K8 post-launch"))?;
        let post_attention_bits = dev.mhc.x_bits(&dev.cuda)?;
        let ffn_coeffs = dev.mhc.coeffs(&dev.cuda)?;
        let ffn_hidden = dev.mhc.normed(&dev.cuda)?;
        let ffn_codes = dev.mhc.codes(&dev.cuda)?;
        let ffn_scales = dev.mhc.scales(&dev.cuda)?;
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
            "  L{layer:02} K8 captured: post_attention_hidden {} bf16 values, FFN coefficients 24 f32, pre_ffn_hidden {} f32 + {} codes + {} scales",
            post_attention_bits.len(),
            ffn_hidden.len(),
            ffn_codes.len(),
            ffn_scales.len()
        );

        // --- S7-S10 and K9-K15: the MoE block.
        let host_ffn_hidden = host.f32(layer, "pre_ffn_hidden")?;
        let host_ffn_out = host.f32(layer, "ffn_out")?;
        let ffn = stage9_ffn_block(
            args,
            dev,
            layer,
            &weights,
            &mut snap,
            &prefix,
            &ffn_hidden,
            &ffn_codes,
            &ffn_scales,
            &host_ffn_hidden,
            &host_ffn_out,
        )?;
        captures.insert("ffn_out", RowValue::F32(ffn.ffn_out.clone()));
        if ffn.device_ids != ffn.host_ids {
            summary.flips.push(layer);
        }
        summary.free_ffn_max = summary.free_ffn_max.max(ffn.free_delta);
        summary.forced_ffn_max = summary.forced_ffn_max.max(ffn.forced_delta);
        carry = Some(Stage9Carry {
            post_attention_bits: post_attention_bits.clone(),
            ffn_out: ffn.ffn_out.clone(),
            ffn_b: ffn_b.clone(),
            ffn_post: ffn_post.clone(),
            ffn_pre: ffn_pre.clone(),
        });

        // --- K16 when no next layer's K1 will apply this layer's FFN hc_post: before an Engram
        // layer (whose write mutates the stream before the K1) and after the last row.
        let next_engram = layer + 1 < n_rows && engram_at(layer + 1);
        let last_row = layer + 1 == n_rows;
        if next_engram || last_row {
            dev.mhc.upload(
                &dev.cuda,
                &post_attention_bits,
                &ffn.ffn_out,
                &ffn_b,
                &ffn_post,
                &ffn_pre,
                &weights.mhc_ffn.predictor,
                &weights.mhc_ffn.hc_scale,
                &weights.mhc_ffn.hc_base_kernel,
                &weights.ffn_norm,
            )?;
            check_floor(args, &format!("stage 9 layer {layer} K16 pre-launch"))?;
            dev.mhc.launch(&dev.cuda, dev.mhc_func, layer, 2)?;
            check_floor(args, &format!("stage 9 layer {layer} K16 post-launch"))?;
            let residual_out_bits = dev.mhc.x_bits(&dev.cuda)?;
            captures.insert(
                "residual_out",
                RowValue::F32(widen_bits(&residual_out_bits)),
            );
            println!(
                "  L{layer:02} K16 captured: residual_out {} bf16 values (the last hc_post applied by a dedicated launch, decision D5){}",
                residual_out_bits.len(),
                if next_engram {
                    format!(
                        "; the released Engram write now runs before layer {}",
                        layer + 1
                    )
                } else {
                    String::new()
                }
            );
            if next_engram {
                pending_x = Some(stage9_engram_seam(
                    cfg,
                    engram_source,
                    hashes,
                    layer + 1,
                    &residual_out_bits,
                    engram_limit,
                )?);
            }
            let (passed, misses, worst_ratio) = stage9_report_row(host, layer, &captures)?;
            println!(
                "  row {layer:>2} summary: {passed}/18 at or under the Stage-B bar, {misses} over it, worst measured/bar {worst_ratio:.3}"
            );
            summary.passes += passed;
            summary.misses += misses;
            summary.worst_ratio = summary.worst_ratio.max(worst_ratio);
            if misses > 0 {
                summary.over_rows.push(layer);
            }
            let mut ordered = Vec::with_capacity(B2_TENSORS.len());
            for (name, _) in B2_TENSORS {
                ordered.push((
                    name,
                    captures
                        .remove(name)
                        .ok_or_else(|| format!("row {layer}: captured set lost {name}"))?,
                ));
            }
            rows.push(ordered);
        } else {
            prev_captures = Some(captures);
        }
        drop(weights);
        if layer + 1 < n_rows {
            check_floor(args, &format!("stage 9 layer {layer} end"))?;
        }
    }
    if prev_captures.is_some() {
        return Err("stage 9: the chain ended with a row that was never finalized".to_string());
    }
    if rows.len() != n_rows {
        return Err(format!(
            "stage 9: captured {} rows, the run asked for {n_rows}",
            rows.len()
        ));
    }
    println!(
        "run {run_index} summary: {} rows, {} tensors at or under the bar, {} over it, worst measured/bar {:.3}, rows with any tensor over bar {:?}",
        summary.rows, summary.passes, summary.misses, summary.worst_ratio, summary.over_rows
    );
    println!(
        "run {run_index} routing: discrete expert flips on {} of {n_rows} layers {:?}; ffn_out max abs delta: free-running up to {:.6e}, forced up to {:.6e}",
        summary.flips.len(),
        summary.flips,
        summary.free_ffn_max,
        summary.forced_ffn_max
    );
    let (reads, bytes) = snap.audit();
    Ok((rows, snap.read_names(), (reads, bytes)))
}

/// Compare two chain runs row by row and tensor by tensor, bit-for-bit. Any difference is a
/// determinism failure, not a tolerance question.
fn stage9_compare_runs(first: &[Stage9Row], second: &[Stage9Row]) -> Result<(), String> {
    if first.len() != second.len() {
        return Err(format!(
            "determinism: run 0 captured {} rows, run 1 captured {}",
            first.len(),
            second.len()
        ));
    }
    for (row_index, (left_row, right_row)) in first.iter().zip(second).enumerate() {
        for ((name, left), (right_name, right)) in left_row.iter().zip(right_row) {
            if name != right_name {
                return Err(format!(
                    "determinism: row {row_index} captured {name} against {right_name}"
                ));
            }
            let identical = match (left, right) {
                (RowValue::F32(a), RowValue::F32(b)) => {
                    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
                }
                (RowValue::U8(a), RowValue::U8(b)) => a == b,
                _ => false,
            };
            if !identical {
                let (first_difference, what) = match (left, right) {
                    (RowValue::F32(a), RowValue::F32(b)) => (
                        a.iter()
                            .zip(b)
                            .position(|(a, b)| a.to_bits() != b.to_bits())
                            .unwrap_or(0),
                        format!("{} against {} values", a.len(), b.len()),
                    ),
                    (RowValue::U8(a), RowValue::U8(b)) => (
                        a.iter().zip(b).position(|(a, b)| a != b).unwrap_or(0),
                        format!("{} against {} bytes", a.len(), b.len()),
                    ),
                    _ => (0, "different classes".to_string()),
                };
                return Err(format!(
                    "determinism: row {row_index} {name} is not byte-identical between two runs of the same command (first difference at element {first_difference}, {what}); the device path is not deterministic, so no other number in this run is trustworthy"
                ));
            }
        }
    }
    Ok(())
}

/// Stage 9: the 40-layer chained device forward. See the section comment above for scope.
pub(crate) fn stage9(args: &Args) -> Result<(), String> {
    let started = std::time::Instant::now();
    println!("=== stage 9: the 40-layer chained device forward (Stage C) ===");
    let host_prefix = args
        .host_row
        .as_ref()
        .ok_or("stage 9 needs --host-row <manifest prefix> for the host pair it diffs against")?;
    let host = HostBoundaryTable::open(host_prefix)?;
    let out_dir = args
        .stage9_out
        .as_ref()
        .ok_or("stage 9 needs --stage9-out <dir> for its v1 pair")?;
    let n_rows = args.stage9_layers;
    let cfg =
        ModelConfig::v41_flash().map_err(|error| format!("released configuration: {error}"))?;
    if cfg.dim != HIDDEN || cfg.hc_mult != HC_MULT {
        return Err(format!(
            "the released configuration gives dim {} and hc_mult {}; this harness is built for {HIDDEN}/{HC_MULT}",
            cfg.dim, cfg.hc_mult
        ));
    }
    if cfg.n_layers < n_rows {
        return Err(format!(
            "the released configuration has {} backbone layers, the run asked for {n_rows}",
            cfg.n_layers
        ));
    }
    let engram_layers: Vec<usize> = (0..n_rows)
        .filter(|&layer| cfg.engram_rows(layer).is_some())
        .collect();
    if args.stage9_kv_format == WindowKvFormat::Engine {
        println!(
            "scope label: this is the Phase-1 HEALTH-CHECK column with the engine's released window-KV format, not a criterion-1 closure. K5's window row now uses the engine's E4M3 + per-32 E8M0 format with its bf16 storage round, the same format class as the host row's window KV; the main/compressed KV, the rest of the arithmetic and the host reference remain f32/bf16 as before, and sparkflash-engine's reference forward keeps f32 dequant, so criterion 1 is NOT closed by this run."
        );
    } else {
        println!(
            "scope label: this is the Phase-1 HEALTH-CHECK column, not a criterion-1 closure. The device path uses FP4 main/window KV and bf16/quantised arithmetic where the host reference is f32 with no GEMM activation quantiser, so this column measures attribution and gross mismatch; the format-faithful exactness run is the later one after Phase 2 (task-3), and criterion 1 is NOT closed by this run."
        );
    }
    println!(
        "rows: {n_rows} backbone rows requested out of 40, one token (id 0), position 0, no MTP rows (a first step records the backbone only); the emitted v1 pair has one row per layer like the host pair"
    );
    println!(
        "comparison classes: exact-codes for pre_attention_xhat/scales and pre_ffn_xhat/scales (no numeric tolerance), exact-bits (device tolerance) for the other fourteen; the measured class is named before any number it judges"
    );
    if args.stage9_kv_format == WindowKvFormat::Engine {
        println!(
            "known differences, stated before the numbers: (1) the K5 window row uses the engine's released FP8 window format (E4M3 codes, one E8M0 scale per 32 channels, bf16 storage round), the same format class as the host row's window KV; the main/compressed KV remains FP4 E2M1 and is not exercised at position 0, where the only candidate is the stored window row; (2) the host reference is f32 throughout and deliberately does not reproduce the device's GEMM activation quantisers or bf16 module outputs; (3) the Engram write at layers 1 and 14 runs on the host seam with the engine's released code, but on the bf16 device stream instead of the host's f32 stream; the device chain's rows after an Engram layer carry that seam's rounding. Every kernel is also checked bit for bit against its in-crate reference on the same bytes, and the two chain runs must be byte-identical, so a kernel or harness non-determinism shows up as a failure, not a delta."
        );
    } else {
        println!(
            "known differences, stated before the numbers: (1) the attention chain feeds the kernel its Main-KV format, FP4 E2M1 codes with one E4M3 scale per 16 channels, while the host row's window KV is the release's FP8 E4M3 path with a per-32 ue8m0 scale (decision D4); (2) the host reference is f32 throughout and deliberately does not reproduce the device's GEMM activation quantisers or bf16 module outputs; (3) the Engram write at layers 1 and 14 runs on the host seam with the engine's released code, but on the bf16 device stream instead of the host's f32 stream; the device chain's rows after an Engram layer carry that seam's rounding. Every kernel is also checked bit for bit against its in-crate reference on the same bytes, and the two chain runs must be byte-identical, so a kernel or harness non-determinism shows up as a failure, not a delta."
        );
    }
    println!(
        "bars: pre-registration, printed before the numbers they judge (design section 8): bar(row L, T) = single_layer_delta_B2(T) * (L + 1) for the stream-carrying tensors (residual_in, post_attention_hidden, residual_out), and single_layer_delta_B2(T) for the per-layer tensors. single_layer_delta_B2 is the frozen Stage-B measurement of record in /home/user/phase1-evidence/phase1-b2-stage7-layer0-rereview5-20261018T044122Z.log lines 149-166 (binary 2b518a35...); this run does not re-measure it."
    );
    println!(
        "bar precision: the Stage-B constants in the frozen log are printed to 7 significant digits, so a tensor within 1e-6 relative of its bar is printed AT-BAR-PRECISION instead of PASS or BELOW-BAR; a zero bar stays exact. The bars are not re-derived from this run."
    );
    println!(
        "row 0 is the Stage-B row: bar(row 0, T) is the Stage-B measured delta by construction and this chain runs the same layer-0 launches, so row 0 checks that the chain still reproduces Stage B bit for bit, and has no bar headroom to spend."
    );
    println!(
        "bar inputs (Stage-B measured; the two layer-0 exact seeds inherit the measured delta of the tensor that produces them):"
    );
    for (name, class) in B2_TENSORS {
        let own = stage9_b2_delta(name);
        let source = match name {
            "residual_in" => "residual_out (inherited: row L residual_in is row L-1 residual_out)",
            "hc_in_pre" => "hc_ffn_pre (inherited: row L hc_in_pre is row L-1 hc_ffn_pre)",
            _ => "own",
        };
        let used = match name {
            "residual_in" => stage9_b2_delta("residual_out"),
            "hc_in_pre" => stage9_b2_delta("hc_ffn_pre"),
            other => stage9_b2_delta(other),
        };
        let factor = if B2_STREAM_TENSORS.contains(&name) {
            "(row+1)"
        } else {
            "1"
        };
        println!(
            "  {name:<24} {:<5} own {own:.6e} bar-from {source} used {used:.6e} factor {factor}",
            if class == BoundaryClass::ExactCodes {
                "codes"
            } else {
                "f32"
            }
        );
    }
    let residual_dims = host.tensor(0, "residual_in")?.dims.clone();
    let host_backbone = host
        .manifest_text()
        .lines()
        .find(|line| line.starts_with("backbone "))
        .ok_or("the host manifest has no backbone line")?
        .to_string();
    println!(
        "host pair: {host_backbone}, row 0 residual_in {residual_dims:?}, {} bytes of blob; {} rows requested",
        host.blob.len(),
        n_rows
    );
    if n_rows == 40 {
        let last = host.tensor(39, "residual_out")?;
        println!(
            "host pair check: row 39 residual_out is present with shape {:?} and {} bytes",
            last.dims, last.len
        );
    }
    // The host pair's own wiring: residual_out(L) and residual_in(L+1) are the same value except
    // where the released Engram write sits between them. The device chain reproduces the write on
    // the host seam, so the report names the rows where the host pair itself does not chain.
    let mut wiring_breaks: Vec<(usize, usize, f32)> = Vec::new();
    for layer in 0..n_rows.saturating_sub(1) {
        let before = host.f32(layer, "residual_out")?;
        let after = host.f32(layer + 1, "residual_in")?;
        let mut differing = 0usize;
        let mut max_abs = 0.0f32;
        for (first, second) in before.iter().zip(&after) {
            if first.to_bits() != second.to_bits() {
                differing += 1;
                max_abs = max_abs.max((first - second).abs());
            }
        }
        if differing > 0 {
            wiring_breaks.push((layer, differing, max_abs));
        }
    }
    println!(
        "host pair residual wiring: residual_out(L) against residual_in(L+1) differs at rows {:?}; the Engram layers are {:?}, so a difference at exactly the rows before them is the released write and nothing else",
        wiring_breaks
            .iter()
            .map(|(layer, _, _)| *layer)
            .collect::<Vec<_>>(),
        engram_layers
    );
    println!(
        "seam: router (gate GEMM, sqrt-softplus, noaux_tc bias, top-6, norm_topk_prob) - host; task-37 moves it"
    );
    println!("seam: gate/up GEMM (w1/w3) - host; Phase 3 moves it");
    println!(
        "seam: SwiGLU and the activation quantiser - host; task-43 compares it, Phase 3 moves it"
    );
    println!("seam: HSI candidate merge (top-512 over the pool) - host; Phase 2/3 move it");
    println!(
        "seam: attention operand preparation (q_norm, q RoPE, kv_norm, kv RoPE, main-KV packing, output inverse RoPE, MXFP8 activation quantisers) - host"
    );
    println!(
        "seam: attention side channel (compressor projection, index keys, candidate positions) - host. Degenerate at position 0: the window holds only position 0, there is no compressed row yet, so this seam supplies the candidate list [0] and nothing else; the chain does not read the compressor/indexer tensors the host reference reads"
    );
    println!(
        "seam: engram write (n-gram lookup and gated add, layers 1 and 14) - host; Phase 2/3 move it. Applied in this run on the host seam with the engine's released code, once before each layer in {:?}, to the bf16 device stream",
        engram_layers
    );
    println!(
        "seam: routed-expert and shared-expert sum (elementwise add of two bf16 vectors) - host"
    );
    println!(
        "seam: head and sampling - host; Phase 1 stage 4 / Phase 3 move it. NOT EXERCISED here: criterion 1 stops at the boundary table, and stage 5 owns the head"
    );
    println!("host input: embedding row of token 0 replicated to 4 copies - host");
    println!("host input: released checkpoint weights, repacked B12x on host before upload - host");
    {
        let mut sink_snap = Snapshot::open(&args.checkpoint)?;
        let sink = sink_snap.read_f32("layers.0.attn.attn_sink", &[MLA_HEADS])?;
        let sink_min = sink.iter().copied().fold(f32::INFINITY, f32::min);
        let sink_max = sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "attn_sink: {} f32 in [{sink_min:.4}, {sink_max:.4}] (layer 0) - CONSUMED ON BOTH SIDES: the kernel takes it as a [64] argument and adds it inside the softmax denominator, and mla_ref::attention_with_sink does the same",
            sink.len()
        );
    }
    let index_path = args.checkpoint.join("model.safetensors.index.json");
    let engram_source = SafeTensorsSource::open(&args.checkpoint, &index_path)
        .map_err(|error| format!("{}: {error}", index_path.display()))?
        .with_limit(sparkflash_engine::safetensors::DEFAULT_LIMIT);
    let mut hashes = Vec::new();
    sparkflash_engine::forward::engram::ngram_hashes(&cfg, &[0u32], 0, &mut hashes)
        .map_err(|error| format!("engram hashes for token 0: {error}"))?;
    println!(
        "engram hashes: {} ids for token 0 ({} Engram layers x {} hash columns); the released n-gram layout is the engine's, not reimplemented here",
        hashes.len(),
        engram_layers.len(),
        cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads
    );
    let extra_k16 = engram_layers.iter().filter(|&&layer| layer > 0).count();
    let launches = n_rows * 19 + 1 + extra_k16;
    println!(
        "launch count per run: {} device launches = {n_rows} layers x (15 chain launches K1..K15 + 4 forced-routing K9-K12 launches) + 1 final K16 + {extra_k16} K16 before an Engram layer",
        launches
    );
    if args.stage9_kv_format == WindowKvFormat::Engine {
        println!(
            "KV row format (engine): each layer's kv_norm(wkv(layer input)) row -> rope tail -> bf16 storage -> E4M3 codes + one E8M0 scale per 32 channels; the row is built by the same engine_window_row helper stage 2 uses (bf16 storage round exactly once), then dispatched to K5 as window_codes/window_scales with window_kv_format=1 and window_rows=1"
        );
        println!(
            "engine window format: this column IS the format-faithful window-KV column; the bars are the frozen Stage-B compounded bars and this is still the health-check class, not a criterion-1 closure"
        );
    }
    check_floor(args, "stage 9 start")?;
    let dev = stage9_device(args)?;
    let mut runs: Vec<Vec<Stage9Row>> = Vec::new();
    let mut engine_stats = EngineWindowStats::default();
    let mut audit: Option<(Vec<String>, usize, u64)> = None;
    for run_index in 0..args.stage9_runs {
        check_floor(args, &format!("stage 9 run {run_index} start"))?;
        let (rows, names, counts) = stage9_chain(
            args,
            &dev,
            &host,
            &cfg,
            &engram_source,
            &hashes,
            n_rows,
            run_index,
            &mut engine_stats,
        )?;
        println!(
            "run {run_index} checkpoint reads: {} tensors, {} bytes (measured by the harness's own reader, not a plan estimate)",
            counts.0, counts.1
        );
        if audit.is_none() {
            audit = Some((names, counts.0, counts.1));
        }
        runs.push(rows);
        check_floor(args, &format!("stage 9 run {run_index} end"))?;
    }
    let first_run = runs.first().ok_or("stage 9 ran no chain")?;
    if runs.len() >= 2 {
        stage9_compare_runs(&runs[0], &runs[1])?;
        println!(
            "determinism: PASS - the two runs of the same command agree bit for bit on all 18 tensors of all {n_rows} rows"
        );
    } else {
        println!(
            "determinism: NOT CLAIMED - one chain run was requested (--stage9-runs 1), so this run makes no determinism claim"
        );
    }
    if args.stage9_kv_format == WindowKvFormat::Engine {
        let rows_expected = n_rows * runs.len();
        if engine_stats.rows != rows_expected
            || engine_stats.quantiser_agreements != rows_expected
            || engine_stats.guarded_rows != rows_expected
        {
            return Err(format!(
                "stage 9 engine window format: built {} rows, {} oracle agreements, {} guarded rows; expected {rows_expected} of each",
                engine_stats.rows, engine_stats.quantiser_agreements, engine_stats.guarded_rows
            ));
        }
        println!(
            "engine window format summary: {} runs x {} layers = {} rows; {}/{} rows byte-identical to m4_quant::act_quant_fp8_row; finite-decode guard PASS on {}/{} rows",
            runs.len(),
            n_rows,
            rows_expected,
            engine_stats.quantiser_agreements,
            rows_expected,
            engine_stats.guarded_rows,
            rows_expected
        );
    }
    let (manifest, blob) = stage9_build_v1(first_run)?;
    std::fs::create_dir_all(out_dir).map_err(|error| format!("{}: {error}", out_dir.display()))?;
    let manifest_path = out_dir.join("device.manifest");
    let blob_path = out_dir.join("device.blob");
    std::fs::write(&manifest_path, &manifest)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    std::fs::write(&blob_path, &blob)
        .map_err(|error| format!("{}: {error}", blob_path.display()))?;
    println!(
        "artifact: {} ({} B), {} ({} B); sha256 manifest {}, blob {}",
        manifest_path.display(),
        manifest.len(),
        blob_path.display(),
        blob.len(),
        stage9_sha256_hex(manifest.as_bytes()),
        stage9_sha256_hex(&blob)
    );
    if runs.len() >= 2 {
        let (_, second_blob) = stage9_build_v1(&runs[1])?;
        println!(
            "determinism blobs: run 0 sha256 {}, run 1 sha256 {}",
            stage9_sha256_hex(&blob),
            stage9_sha256_hex(&second_blob)
        );
    }
    let host_manifest = std::fs::read(host_prefix.with_extension("manifest"))
        .map_err(|error| format!("{}.manifest: {error}", host_prefix.display()))?;
    if manifest.as_bytes() == host_manifest.as_slice() {
        println!(
            "artifact check: the device manifest is byte-identical to the host pair's manifest, so the v1 writer and the format agree"
        );
    } else if n_rows == 40 {
        return Err("stage 9: the device manifest is not byte-identical to the host pair's manifest; the v1 writer disagrees with the format".to_string());
    } else {
        println!(
            "artifact check: the manifest differs from the host pair because this smoke run has {n_rows} rows, not the host's 40; no boundary_diff comparison is claimed"
        );
    }
    let (names, reads, bytes) = audit.ok_or("stage 9: no run recorded a read audit")?;
    let plan = TensorPlan::language_model(&cfg);
    let mut read: std::collections::BTreeSet<String> = names.into_iter().collect();
    for layer in &engram_layers {
        for suffix in [
            "engram.embed.weight",
            "engram.embed.scale",
            "engram.wkv.weight",
            "engram.wkv.scale",
            "engram.q_weight",
            "engram.k_weight",
        ] {
            read.insert(format!("layers.{layer}.{suffix}"));
        }
    }
    let plan_names: std::collections::BTreeSet<String> =
        plan.specs().iter().map(|spec| spec.name.clone()).collect();
    let unread: Vec<&String> = plan_names
        .iter()
        .filter(|name| !read.contains(name.as_str()))
        .collect();
    let stages = unread
        .iter()
        .filter(|name| name.starts_with("mtp."))
        .count();
    let experts = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.contains(".ffn.experts."))
        .count();
    let text_path = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.ends_with("ffn.gate.bias_vl"))
        .count();
    let indexer = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.contains(".attn.indexer."))
        .count();
    let compressor = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.contains(".attn.compressor."))
        .count();
    let head = unread
        .iter()
        .filter(|name| name.as_str() == "head.weight" || name.as_str() == "norm.weight")
        .count();
    let other: Vec<&&String> = unread
        .iter()
        .filter(|name| {
            !name.starts_with("mtp.")
                && !name.contains(".ffn.experts.")
                && !name.ends_with("ffn.gate.bias_vl")
                && !name.contains(".attn.indexer.")
                && !name.contains(".attn.compressor.")
                && name.as_str() != "head.weight"
                && name.as_str() != "norm.weight"
        })
        .collect();
    println!(
        "audit: plan={} read={} unread={} expected-unread=[unselected-expert {experts}, DSpark-stage {stages}, text-path {text_path}, indexer {indexer}, compressor {compressor}, head-not-claimed {head}]{}",
        plan_names.len(),
        read.len(),
        unread.len(),
        if other.is_empty() {
            String::new()
        } else {
            format!(
                " UNCLASSIFIED {} first {:?}",
                other.len(),
                other.iter().take(8).collect::<Vec<_>>()
            )
        }
    );
    println!(
        "audit note: the compressor/indexer classes are larger than the host report's because the S7 seam is degenerate at position 0 and this chain never reads those tensors, while the host reference does; that is a property of the health-check column, not an unread-tensor bug"
    );
    println!(
        "audit bytes: {reads} tensor reads, {bytes} bytes through the harness's reader (the Engram tables are read through the engine's paged source, not this counter)"
    );
    let mem_kb = mem_available_kb_checked()?;
    println!(
        "stage 9 trailer: wall {:.1} s, MemAvailable {mem_kb} KiB, {n_rows} rows, {} chain runs, binary argv recorded by the wrapper log header",
        started.elapsed().as_secs_f64(),
        runs.len()
    );
    println!(
        "PASS framing: the health-check column ran to the last row; it does not close criterion 1, and the measured-vs-bar numbers above are attribution, not a gate"
    );
    Ok(())
}
