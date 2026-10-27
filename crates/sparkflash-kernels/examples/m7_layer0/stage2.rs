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

// ---------------------------------------------------------------------------------------------
// Stage 2: MLA attention on layer 0. compress_ratios[0] == 0, so the sliding-window path over at
// most index_topk candidates is the whole attention rule. The per-head attention sink is consumed
// on both sides: the kernel takes a [64] f32 argument and the reference is attention_with_sink.
// ---------------------------------------------------------------------------------------------

/// The MLA latent width, mirrored from mla_ref::MLA_LATENT.
pub(crate) const MLA_LATENT: usize = 512;
/// Query heads, mirrored from mla_ref::MLA_HEADS.
pub(crate) const MLA_HEADS: usize = 64;
/// One head per block, 256 threads: SPARK_M4_MLA_THREADS.
pub(crate) const MLA_THREADS: u32 = 256;
/// The indexer geometry the kernel's first stage reads even though this stage does not compare it.
pub(crate) const MLA_INDEX_HEAD_DIM: usize = 128;
pub(crate) const MLA_HSI_BLOCKS: usize = 2048;
pub(crate) const MLA_HSI_BLOCK_TOKENS: usize = 8;
/// The candidate bound, mirrored from mla_ref::MAX_SELECTED (= index_topk).
pub(crate) const MLA_MAX_SELECTED: usize = 512;
/// The sliding window, mirrored from mla_ref::SWA_WINDOW.
const MLA_SWA_WINDOW: u32 = 128;
/// qk_rope_head_dim: rope rotates the last 64 of a head's 512 (model.py:772).
pub(crate) const ROPE_HEAD_DIM: usize = 64;
/// q_lora_rank: wq_a's width and q_norm's length.
pub(crate) const Q_LORA_RANK: usize = 1280;
/// The FP4 main-KV row: 512 values pack into 256 bytes with one E4M3 scale per 16 channels.
pub(crate) const KV_PACKED_BYTES: usize = MLA_LATENT / 2;
const KV_SCALE_CHANNELS: usize = 16;
pub(crate) const KV_SCALE_BYTES: usize = MLA_LATENT / KV_SCALE_CHANNELS;
/// text_config.rms_norm_eps.
pub(crate) const NORM_EPS: f32 = 1e-20;
/// text_config.rope_theta. Layer 0 has compress_ratio 0, so it takes the base rope with no YaRN
/// correction - the engine's rope_tables_for_layer selects rope_tables for exactly that case.
pub(crate) const ROPE_THETA: f32 = 10000.0;
/// 6 * 2^-9: the floor sparkflash-kv's quantize_main_kv puts under a group's amax.
const KV_AMAX_FLOOR: f32 = 6.0 * 0.001_953_125;
/// E2M1's largest finite magnitude.
const E2M1_MAX: f32 = 6.0;
/// The stage-2 pass bar, printed before the numbers that are judged against it.
const STAGE2_EXACT_BAR: f64 = 99.5;
const STAGE2_DELTA_BAR: f32 = 1.6e-2;
/// The engine window format's scale geometry: one E8M0 byte per 32 latent channels (model.py:27).
const WINDOW_QUANT_BLOCK: usize = 32;
pub(crate) const WINDOW_SCALE_BYTES: usize = MLA_LATENT / WINDOW_QUANT_BLOCK;
/// The engine's sliding window is 128 rows, supplied oldest first.
const WINDOW_ROWS: usize = MLA_SWA_WINDOW as usize;
/// The merged-chain FP4 numbers on the --tokens 130 plan, re-reported next to the engine column so
/// that a change to the default path cannot pass unnoticed. These are the numbers the merged
/// Stage-2 claim records on the chain that contains D8: 163,838/163,840 exact (99.9988%), 2.4414063e-4
/// max abs delta, one bf16 ULP max. They were defined by the paired diagnostic of 2026-10-18
/// (phase1-task3-diag-oldkernel-20261018T060534Z.log and ...newkernel...): the merged-head binary
/// with the old 15-parameter kernel (cubin 822df7f8) and the task-3 binary (cubin bbb216b7) write
/// identical per-position lines on the same chain, so this constant is the old kernel's own number,
/// not a value the new path is allowed to drift from. The earlier record's 9.765625e-4 max abs delta
/// was measured on the pre-D8 stage-1 chain (phase1-b0-stage2-layer0-20261018T010919Z.log); it is
/// kept only as a historical label and is not expected here.
const STAGE2_MERGED_CHAIN_EXACT: usize = 163_838;
const STAGE2_MERGED_CHAIN_TOTAL: usize = 163_840;
const STAGE2_MERGED_CHAIN_MAX_DELTA: f32 = 2.4414063e-4;
const STAGE2_MERGED_CHAIN_MAX_ULP: i32 = 1;

const _: () = assert!(
    MLA_LATENT == 512
        && MLA_HEADS == 64
        && MLA_MAX_SELECTED == 512
        && MLA_SWA_WINDOW == 128
        && Q_LORA_RANK == 1280
        && ROPE_HEAD_DIM == 64
        && KV_PACKED_BYTES == 256
        && KV_SCALE_BYTES == 32
);

/// RMSNorm over one row, the engine reference's formula (linalg::rms_norm): mean of squares,
/// 1/sqrt(mean + eps), then the per-column weight.
pub(crate) fn rms_norm_row(values: &mut [f32], weight: &[f32], eps: f32) {
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let inverse = 1.0 / (mean + eps).sqrt();
    for (value, w) in values.iter_mut().zip(weight) {
        *value = *value * inverse * *w;
    }
}

/// The engine reference's rope tables for one position: cos/sin of position * base^(-2k/dim) for
/// dim/2 pairs (model.py:368-389), kept in f32 like the reference's f32 tensor.
fn rope_tables_for(position: u32, dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let pairs = dim / 2;
    let angles: Vec<f32> = (0..pairs)
        .map(|k| position as f32 / theta.powf(2.0 * k as f32 / dim as f32))
        .collect();
    (
        angles.iter().map(|angle| angle.cos()).collect(),
        angles.iter().map(|angle| angle.sin()).collect(),
    )
}

/// Rotates the LAST rope-long tail of each width-wide row in place, exactly as the engine's
/// rotate_rows does: start = width - rope, and (a, b) becomes (a cos - b sin, a sin + b cos).
pub(crate) fn rotate_rope_tail(
    values: &mut [f32],
    rows: usize,
    width: usize,
    rope: usize,
    theta: f32,
    position: u32,
) {
    let (cos, sin) = rope_tables_for(position, rope, theta);
    for row in 0..rows {
        let start = row * width + width - rope;
        for pair in 0..rope / 2 {
            let a = values[start + 2 * pair];
            let b = values[start + 2 * pair + 1];
            values[start + 2 * pair] = a * cos[pair] - b * sin[pair];
            values[start + 2 * pair + 1] = a * sin[pair] + b * cos[pair];
        }
    }
}

/// Rotates the LAST rope-long tail of each width-wide row by the NEGATIVE angle: the inverse of
/// rotate_rope_tail, used on the attention output before the wo_a operand.
pub(crate) fn rotate_rope_tail_inverse(
    values: &mut [f32],
    rows: usize,
    width: usize,
    rope: usize,
    theta: f32,
    position: u32,
) {
    let (cos, sin) = rope_tables_for(position, rope, theta);
    for row in 0..rows {
        let start = row * width + width - rope;
        for pair in 0..rope / 2 {
            let a = values[start + 2 * pair];
            let b = values[start + 2 * pair + 1];
            values[start + 2 * pair] = a * cos[pair] + b * sin[pair];
            values[start + 2 * pair + 1] = -a * sin[pair] + b * cos[pair];
        }
    }
}

/// Quantises one 512-wide latent row to the FP4 main-KV format the kernel and mla_ref::kv_latent
/// both consume, by sparkflash-kv's quantize_main_kv rule: per-16-channel amax floored at 6 * 2^-9,
/// an E4M3 scale of amax / 6, E2M1 codes of value / scale clamped to +-6, and the low nibble first
/// (which is the order mla_ref::kv_latent and spark_mla_load_latent both read).
pub(crate) fn quantize_kv_row(latent: &[f32]) -> ([u8; KV_PACKED_BYTES], [u8; KV_SCALE_BYTES]) {
    let mut packed = [0u8; KV_PACKED_BYTES];
    let mut scales = [0u8; KV_SCALE_BYTES];
    for group in 0..KV_SCALE_BYTES {
        let block = &latent[group * KV_SCALE_CHANNELS..(group + 1) * KV_SCALE_CHANNELS];
        let amax = block.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale_byte = fp8_e4m3_encode(amax.max(KV_AMAX_FLOOR) / E2M1_MAX);
        let scale = fp8_e4m3_decode(scale_byte);
        scales[group] = scale_byte;
        for (offset, value) in block.iter().enumerate() {
            let index = group * KV_SCALE_CHANNELS + offset;
            let code = fp4_e2m1_encode((value / scale).clamp(-E2M1_MAX, E2M1_MAX));
            if index.is_multiple_of(2) {
                packed[index / 2] = code & 0x0F;
            } else {
                packed[index / 2] |= (code & 0x0F) << 4;
            }
        }
    }
    (packed, scales)
}

/// The monotone integer key of a bf16 bit pattern, so two codes an ULP apart differ by one and the
/// distance is meaningful across zero.
pub(crate) fn bf16_order(bits: u16) -> i32 {
    let magnitude = i32::from(bits & 0x7FFF);
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Compares a device bf16 write against the bf16 rounding of the f32 reference and reports the
/// delta in bf16 codes as well as in absolute value, because a format with a relative step cannot
/// be judged by an absolute bar alone.
pub(crate) fn compare_bf16_ulp(
    label: &str,
    device: &[u16],
    reference: &[f32],
) -> Result<(Stats, i32), String> {
    let expected = bf16_bits_of_f32(reference);
    if device.len() != expected.len() {
        return Err(format!(
            "{label}: {} device values against {} reference values",
            device.len(),
            expected.len()
        ));
    }
    let mut stats = Stats {
        len: device.len(),
        exact: 0,
        max_delta: 0.0,
        at: 0,
    };
    let mut max_ulp = 0i32;
    for (index, (device_bits, expected_bits)) in device.iter().zip(&expected).enumerate() {
        if device_bits == expected_bits {
            stats.exact += 1;
        }
        let delta = (half::bf16::from_bits(*device_bits).to_f32()
            - half::bf16::from_bits(*expected_bits).to_f32())
        .abs();
        if !delta.is_finite() || delta > stats.max_delta {
            stats.max_delta = if delta.is_finite() {
                delta
            } else {
                f32::INFINITY
            };
            stats.at = index;
        }
        let ulp = (bf16_order(*device_bits) - bf16_order(*expected_bits)).abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
    }
    println!(
        "  {label}: {} bf16 values, exact {}/{} ({:.4}%), max ULP distance {max_ulp}, max abs delta {:e} at index {}",
        stats.len,
        stats.exact,
        stats.len,
        stats.exact_pct(),
        stats.max_delta,
        stats.at,
    );
    Ok((stats, max_ulp))
}

pub(crate) fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The query positions stage 2 runs: token 0 is the tightest window (one candidate), 1 and 127
/// fill it, and 128/129 are the first positions whose window leaves a candidate out. The last
/// token of a longer prompt is added when it is not already in the list.
fn stage2_positions(tokens: usize) -> Vec<usize> {
    let mut positions = vec![0usize];
    for candidate in [1usize, 127, 128, 129, tokens.saturating_sub(1)] {
        if candidate < tokens {
            positions.push(candidate);
        }
    }
    positions.sort_unstable();
    positions.dedup();
    positions
}

/// The candidate list for one query position: the in-window range first, then deliberately older
/// positions. The kernel and mla_ref::attention both apply the same window predicate and skip an
/// out-of-window candidate exactly, so the older entries exercise the exclusion rather than
/// down-weighting it.
pub(crate) fn stage2_selected(position: usize) -> Vec<u32> {
    let window = MLA_SWA_WINDOW as usize;
    let first = position.saturating_sub(window - 1);
    let mut selected: Vec<u32> = (first..=position)
        .map(|candidate| candidate as u32)
        .collect();
    let mut older = position as i64 - window as i64;
    while older >= 0 && selected.len() < MLA_MAX_SELECTED {
        selected.push(older as u32);
        older -= 1;
    }
    selected
}

pub(crate) fn stage2(args: &Args) -> Result<(), String> {
    let kernel = "flash_mla_csa2_sm121";
    let prefix = format!("layers.{}", args.layer);
    let format = args.stage2_kv_format;
    println!("=== stage 2: {kernel} against mla_ref::attention_with_sink ===");
    println!(
        "stage-2 KV format: {} (kernel selector {}). fp4 is the merged Stage-2 default and this run must reproduce its historical numbers; engine is the engine's released window-KV format and is reported as a separate class of evidence, never folded into the FP4 claim.",
        format.label(),
        format.kernel_value()
    );
    println!(
        "comparison class: TOLERANCE-BASED, not exact. The kernel writes bf16 and evaluates expf on the device; the reference is f32 with the library exponential. The tolerance is named here, before any number it judges: at least {STAGE2_EXACT_BAR}% of the {} outputs bit-identical to the bf16 rounding of the f32 reference, and max abs delta <= {STAGE2_DELTA_BAR:e}, with the max ULP distance reported. Reason: the score sum, the window predicate, the candidate or row order, the KV decode and the divide are identical by construction on both sides, so the only degrees of freedom are expf against libm exp, the tree reduction of the normalising sum against the serial one, and the bf16 output write itself. {}",
        MLA_HEADS * MLA_LATENT,
        match format {
            WindowKvFormat::Fp4 =>
                "This run uses the historical FP4 latent format, the same class the merged Stage-2 claim records; its totals must equal that claim exactly.",
            WindowKvFormat::Engine =>
                "This run uses the engine window format: the stored values are bf16(dequant(E4M3 code, E8M0 scale)), reproduced on both sides, so this class measures format fidelity. It is not a bit-identity claim against the released kernel, and the engine's bf16 arithmetic is not reproduced: sparkflash-engine's reference forward keeps f32 dequant, and that pre-existing seam is named rather than hidden.",
        }
    );
    println!(
        "label: internal consistency of the sink arithmetic, not model fidelity. The attention sink (attn_sink, F32 [64], a per-head additive logit inside the softmax denominator) is CONSUMED ON BOTH SIDES: the kernel takes it as an argument and adds it to the denominator after the maximum, and the reference does the same. The class printed above judges the numbers below. This run's format is {}.",
        format.label()
    );
    println!(
        "inherited limitation from stage 1, printed rather than implied: this stage's input chain starts at stage 1's xhat fp8 codes and scales, which carry stage 1's residual gap - the bf16 write, the tf32-like reductions of the device, and the kernel's per-tile reduction order against the reference's serial arithmetic. The predictor is NOT a gap any more: B0 removed the d-wide fold, both sides consume the release's [24, 20480] projection over the flattened streams, and the coefficient tensors hc_attn_base and hc_attn_scale are consumed on both sides with the kernel's C compared raw."
    );
    println!(
        "layer {}: compress_ratios[0] = 0, so this is NOT a CED layer. The whole attention rule is the 128-token sliding window over at most index_topk 512 candidates. The candidate list is supplied by the harness: the top-512 merge over the HSI pool is a separate pipeline step the kernel does not implement. The HSI indexer inside the kernel is checked separately at the top of this stage on deterministic non-zero index data, against mla_ref::index_block_best, because stage 2 never consumes its block_scores/block_best; this stage's own boundary is the attention output.",
        args.layer
    );
    if format == WindowKvFormat::Fp4 {
        println!(
            "KV row format: the rows fed to BOTH sides are the FP4 main-KV format the kernel and mla_ref::kv_latent both consume, quantised here by sparkflash-kv's quantize_main_kv rule (per-16 E4M3 scale of amax/6, E2M1 codes, low nibble first) from kv_norm(wkv(xhat)) with the rope tail rotated. The engine's released window path rounds its KV to FP8 instead, so this run measures the kernel against the reference over the kernel's KV format, not the engine's window path; the engine format is a separate run (--stage2-kv-format engine)."
        );
    } else {
        println!(
            "KV row format: the engine's released window format (model.py:700-707, kernel.py:98-124). Each kv_norm(wkv(xhat)) row is rounded to bf16 - the engine's storage dtype, because its in-place act_quant returns the input dtype - then quantised into E4M3 codes with one E8M0 power-of-two scale per 32 channels. Both the kernel and mla_ref::attention_with_sink_window decode code times scale and round the decoded value to bf16, so the bf16 storage round is consumed on both sides. The device bytes are the engine crate's own quantiser and are asserted byte-identical to m4_quant::act_quant_fp8_row on the same bf16 rows. sparkflash-engine's own reference forward keeps f32 dequant, so the engine's bf16 arithmetic is not reproduced here; that seam is named, not hidden."
        );
    }
    check_floor(args, "stage 2 start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The stage-1 chain, exactly as stage 1 runs it: the released tensors, the boundary seed and
    // the mHC reference, whose returned codes and scales are this stage's input.
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let predictor = mhc.predictor;
    let hc_base = mhc.hc_base;
    let hc_scale = mhc.hc_scale;
    // The stage-1 boundary seed: A = [1, 0, 0, 0] with X_-1 = four copies of the embed row. B and C
    // only matter to stage 1's coefficient comparison; the mHC reference takes A and X here.
    let a_prev = [1.0f32, 0.0, 0.0, 0.0];

    let wq_a_weight = snap.read_raw(
        &format!("{prefix}.attn.wq_a.weight"),
        "F8_E4M3",
        &[Q_LORA_RANK, HIDDEN],
    )?;
    let wq_a_scale = snap.read_raw(
        &format!("{prefix}.attn.wq_a.scale"),
        "F8_E8M0",
        &[Q_LORA_RANK / 32, HIDDEN / 32],
    )?;
    let wq_b_weight = snap.read_raw(
        &format!("{prefix}.attn.wq_b.weight"),
        "F8_E4M3",
        &[MLA_HEADS * MLA_LATENT, Q_LORA_RANK],
    )?;
    let wq_b_scale = snap.read_raw(
        &format!("{prefix}.attn.wq_b.scale"),
        "F8_E8M0",
        &[(MLA_HEADS * MLA_LATENT) / 32, Q_LORA_RANK / 32],
    )?;
    let wkv_weight = snap.read_raw(
        &format!("{prefix}.attn.wkv.weight"),
        "F8_E4M3",
        &[MLA_LATENT, HIDDEN],
    )?;
    let wkv_scale = snap.read_raw(
        &format!("{prefix}.attn.wkv.scale"),
        "F8_E8M0",
        &[MLA_LATENT / 32, HIDDEN / 32],
    )?;
    let q_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.q_norm.weight"), &[Q_LORA_RANK])?;
    let kv_norm = snap.read_f32_or_bf16(&format!("{prefix}.attn.kv_norm.weight"), &[MLA_LATENT])?;
    let sink = snap.read_f32(&format!("{prefix}.attn.attn_sink"), &[MLA_HEADS])?;
    let sink_min = sink.iter().copied().fold(f32::INFINITY, f32::min);
    let sink_max = sink.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "weights: wq_a {} B + {} scale B, wq_b {} B + {} scale B, wkv {} B + {} scale B (the packed checkpoint bytes, handed to mxfp8_ref::dense_gemm as-is so no weight is ever decoded to f32 on the host)",
        wq_a_weight.len(),
        wq_a_scale.len(),
        wq_b_weight.len(),
        wq_b_scale.len(),
        wkv_weight.len(),
        wkv_scale.len()
    );
    println!(
        "attn_sink: {} f32 in [{sink_min:.4}, {sink_max:.4}] - consumed on BOTH sides: passed to the kernel as a [64] argument and added to the denominator after the maximum, and passed to mla_ref::attention_with_sink",
        sink.len()
    );

    let tokens = args.tokens;
    let positions = stage2_positions(tokens);
    println!(
        "prompt: {tokens} tokens (embed rows 0..{tokens}), query positions {positions:?}; the window is {MLA_SWA_WINDOW}, so a position below {} never sees an out-of-window candidate",
        MLA_SWA_WINDOW
    );
    if tokens <= MLA_SWA_WINDOW as usize {
        println!(
            "note: with {tokens} tokens the out-of-window branch is not exercised; run --tokens 130 for the plan that exercises it"
        );
    }

    // Per-token xhat codes and scales from the mHC reference: stage 1's output, and this stage's
    // input. These are the packed bytes, never decoded weights.
    let mut xhat: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(tokens);
    let mut kv_fp4: Vec<u8> = Vec::with_capacity(tokens * KV_PACKED_BYTES);
    let mut kv_scales: Vec<u8> = Vec::with_capacity(tokens * KV_SCALE_BYTES);
    // Engine window format rows, one per token: [tokens][512] E4M3 codes and [tokens][16] E8M0
    // scales. The kernel is handed the contiguous slice for one query position, oldest first.
    let mut window_codes: Vec<u8> = Vec::new();
    let mut window_scales: Vec<u8> = Vec::new();
    let mut window_quantiser_agreements = 0usize;
    for token in 0..tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let mut streams = Streams::new(x);
        let mut ledger = TrafficLedger::default();
        let (_next, _coefficients, codes, scales) = mega_mhc_layer_with_scale(
            &mut streams,
            &norm,
            &a_prev,
            &predictor,
            &hc_base,
            &hc_scale,
            &mut ledger,
        );
        let kv_bits = mxfp8_ref::dense_gemm(
            codes.as_slice(),
            scales.as_slice(),
            &wkv_weight,
            &wkv_scale,
            1,
            MLA_LATENT,
            HIDDEN,
        );
        let mut latent: Vec<f32> = kv_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        rms_norm_row(&mut latent, &kv_norm, NORM_EPS);
        rotate_rope_tail(
            &mut latent,
            1,
            MLA_LATENT,
            ROPE_HEAD_DIM,
            ROPE_THETA,
            token as u32,
        );
        let (packed, row_scales) = quantize_kv_row(&latent);
        kv_fp4.extend_from_slice(&packed);
        kv_scales.extend_from_slice(&row_scales);
        if format == WindowKvFormat::Engine {
            let engine_row =
                engine_window_row(&latent).map_err(|error| format!("token {token}: {error}"))?;
            window_quantiser_agreements += 1;
            window_codes.extend_from_slice(&engine_row.codes);
            window_scales.extend_from_slice(&engine_row.scales);
        }
        xhat.push((codes.as_slice().to_vec(), scales.as_slice().to_vec()));
        let _ = ledger;
    }
    if format == WindowKvFormat::Fp4 {
        println!(
            "input chain (FP4): {tokens} xhat fp8 rows from the mHC reference -> wkv {} B weight -> kv_norm -> rope tail -> FP4; kv_fp4 {} B, kv_scales {} B",
            wkv_weight.len(),
            kv_fp4.len(),
            kv_scales.len()
        );
    } else {
        println!(
            "input chain (engine): {tokens} xhat fp8 rows from the mHC reference -> wkv {} B weight -> kv_norm -> rope tail -> bf16 storage -> E4M3 + E8M0; window_codes {} B, window_scales {} B",
            wkv_weight.len(),
            window_codes.len(),
            window_scales.len()
        );
        println!(
            "engine window format: {tokens} rows rounded to bf16 then quantised by sparkflash_engine::quant::quantize_activation_fp8; {window_quantiser_agreements}/{tokens} rows byte-identical to m4_quant::act_quant_fp8_row (E4M3 codes and E8M0 scales, asserted not averaged)"
        );
        if window_quantiser_agreements != tokens {
            return Err("the engine window quantiser and the m4_quant oracle disagree".to_string());
        }
        assert_engine_window_finite(
            &window_codes,
            &window_scales,
            tokens,
            "engine window format",
        )?;
        println!(
            "engine window guard: {tokens} rows x {MLA_LATENT} channels decoded finite; no E4M3 NaN code (0x7F) and no overflowing (0x7E x scale 254) pair reached the kernel"
        );
    }

    // The device buffers are sized once and admitted against the memory floor before allocation.
    let index_key_bytes = MLA_HSI_BLOCKS * MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM * 2;
    let index_query_bytes = MLA_HEADS * MLA_INDEX_HEAD_DIM * 2;
    let out_bytes = MLA_HEADS * MLA_LATENT * 2;
    let window_codes_bytes = WINDOW_ROWS * MLA_LATENT;
    let window_scales_bytes = WINDOW_ROWS * WINDOW_SCALE_BYTES;
    let planned = (MLA_HEADS * MLA_LATENT * 2
        + tokens * KV_PACKED_BYTES
        + tokens * KV_SCALE_BYTES
        + window_codes_bytes
        + window_scales_bytes
        + MLA_MAX_SELECTED * 4
        + index_key_bytes
        + index_query_bytes
        + MLA_HSI_BLOCKS * 4
        + MLA_HSI_BLOCKS * 4
        + out_bytes
        + MLA_HEADS * 4) as u64;
    let cuda = Cuda::load()?;
    let cubin_path = crate::cubin_path(&args.cubin_dir, kernel)?;
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            cubin_path.display()
        )
    })?;
    let (module, func) = load_flash_mla_kernel(&cuda, &cubin)?;
    admit_memory(args, planned, "stage 2 device buffers")?;
    let mut allocated = 0usize;
    let q_d = cuda.alloc(MLA_HEADS * MLA_LATENT * 2, &mut allocated)?;
    let sink_d = cuda.alloc(MLA_HEADS * 4, &mut allocated)?;
    let kv_fp4_d = cuda.alloc(tokens * KV_PACKED_BYTES, &mut allocated)?;
    let kv_scales_d = cuda.alloc(tokens * KV_SCALE_BYTES, &mut allocated)?;
    let window_codes_d = cuda.alloc(window_codes_bytes, &mut allocated)?;
    let window_scales_d = cuda.alloc(window_scales_bytes, &mut allocated)?;
    let selected_d = cuda.alloc(MLA_MAX_SELECTED * 4, &mut allocated)?;
    let index_key_d = cuda.alloc(index_key_bytes, &mut allocated)?;
    let index_query_d = cuda.alloc(index_query_bytes, &mut allocated)?;
    let block_scores_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let block_best_d = cuda.alloc(MLA_HSI_BLOCKS * 4, &mut allocated)?;
    let out_d = cuda.alloc(out_bytes, &mut allocated)?;
    println!(
        "device bytes allocated: {allocated} ({} KiB); planned {planned} B",
        allocated / 1024
    );
    cuda.copy_htod(&kv_fp4_d, &kv_fp4)?;
    cuda.copy_htod(&kv_scales_d, &kv_scales)?;
    cuda.copy_htod(&sink_d, &f32_to_bytes(&sink))?;

    // Deterministic integer-valued index data for the indexer self-check below. Every product and
    // partial sum is an exact integer in f32 (|dot| <= 1,024), so the reference and the kernel must
    // agree bit for bit and a mapping error cannot hide behind rounding. Stage 2 never reads these
    // buffers, so this does not touch the attention comparison.
    let index_key_f32: Vec<f32> = (0..MLA_HSI_BLOCKS * MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM)
        .map(|i| {
            let block = i / (MLA_HSI_BLOCK_TOKENS * MLA_INDEX_HEAD_DIM);
            let slot = (i / MLA_INDEX_HEAD_DIM) % MLA_HSI_BLOCK_TOKENS;
            let dim = i % MLA_INDEX_HEAD_DIM;
            (((block * 13 + slot * 7 + dim * 3) % 9) as i32 - 4) as f32
        })
        .collect();
    let index_query_f32: Vec<f32> = (0..MLA_HEADS * MLA_INDEX_HEAD_DIM)
        .map(|i| {
            let index_head = i / MLA_INDEX_HEAD_DIM;
            let dim = i % MLA_INDEX_HEAD_DIM;
            (((index_head * 5 + dim * 11) % 7) as i32 - 3) as f32
        })
        .collect();
    cuda.copy_htod(
        &index_key_d,
        &bf16_bits_to_bytes(&bf16_bits_of_f32(&index_key_f32)),
    )?;
    cuda.copy_htod(
        &index_query_d,
        &bf16_bits_to_bytes(&bf16_bits_of_f32(&index_query_f32)),
    )?;
    // Sentinels, so a block the indexer fails to write is visible even when its correct score is 0.
    cuda.copy_htod(
        &block_scores_d,
        &f32_to_bytes(&vec![12345.0f32; MLA_HSI_BLOCKS]),
    )?;
    cuda.copy_htod(
        &block_best_d,
        &u32s_to_bytes(&vec![u32::MAX; MLA_HSI_BLOCKS]),
    )?;

    // The indexer stage of the kernel runs one blockIdx.y slice per stripe of HSI blocks. The
    // slice count is a measured lever: more slices give the pool more blocks to run on but pay a
    // fixed per-block cost, and only slice 0 runs stage 2. See --grid-y and the task-41 report.
    let grid = (MLA_HEADS as u32, args.grid_y, 1u32);
    let block = (MLA_THREADS, 1u32, 1u32);
    println!(
        "launch grid: ({}, {}, {}) blocks of ({}, 1, 1) threads; each warp walks {} HSI blocks per head",
        grid.0,
        grid.1,
        grid.2,
        block.0,
        (MLA_HSI_BLOCKS as u32).div_ceil(args.grid_y * 8 * 2)
    );

    // --- indexer self-check ---------------------------------------------------------------
    // Stage 2 never reads block_scores or block_best, so until now the harness fed the indexer
    // zeros and compared nothing: a mapping error in the warp-level rewrite would pass every
    // attention comparison below. This runs the indexer on integer-valued bf16 index data, where
    // every dot product is exact in f32, and compares the two heads of index head 0 - the pair
    // that splits the pool in the kernel - against mla_ref::index_block_best. Blocks the kernel
    // does not write keep their sentinel, so partial coverage fails rather than passing silently.
    let index_check_params = flash_mla_params(
        q_d.0,
        kv_fp4_d.0,
        kv_scales_d.0,
        window_codes_d.0,
        window_scales_d.0,
        WindowKvFormat::Fp4.kernel_value() as u64,
        0, // window_rows is a plain count: the indexer's launch carries none
        selected_d.0,
        index_key_d.0,
        index_query_d.0,
        sink_d.0,
        block_scores_d.0,
        block_best_d.0,
        out_d.0,
        0, // position
        0, // selected_count
        0, // trace_ring: the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
        args.layer,
        0, // step_seq
    );
    check_floor(args, "indexer check pre-launch")?;
    flash_mla_launch(
        &cuda,
        func,
        (2u32, args.grid_y, 1u32),
        block,
        &index_check_params,
    )?;
    check_floor(args, "indexer check post-launch")?;
    let mut index_scores_bytes = vec![0u8; MLA_HSI_BLOCKS * 4];
    let mut index_best_bytes = vec![0u8; MLA_HSI_BLOCKS * 4];
    cuda.copy_dtoh(&mut index_scores_bytes, &block_scores_d)?;
    cuda.copy_dtoh(&mut index_best_bytes, &block_best_d)?;
    let index_scores_dev = bytes_to_f32(&index_scores_bytes);
    let index_best_dev: Vec<u32> = index_best_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect();
    let (index_scores_ref, index_best_ref) = mla_ref::index_block_best(
        &index_query_f32[..MLA_INDEX_HEAD_DIM],
        &index_key_f32,
        MLA_HSI_BLOCKS,
    );
    let index_score_stats = compare_f32(
        "indexer block_scores (head pair 0/1, the full pool)",
        &index_scores_dev,
        &index_scores_ref,
    )?;
    let index_best_exact = index_best_dev
        .iter()
        .zip(&index_best_ref)
        .filter(|(device, reference)| device == reference)
        .count();
    println!(
        "  indexer block_best: {index_best_exact}/{} exact (an unwritten block keeps u32::MAX)",
        MLA_HSI_BLOCKS
    );
    if !index_score_stats.all_exact() || index_best_exact != MLA_HSI_BLOCKS {
        return Err(format!(
            "indexer self-check FAILED at grid.y {}: {} of {} block scores exact and {index_best_exact} of {} block_best exact",
            args.grid_y, index_score_stats.exact, index_score_stats.len, MLA_HSI_BLOCKS
        ));
    }
    println!(
        "  indexer self-check: PASS at grid.y {} (both parities of the pool, one head each)",
        args.grid_y
    );

    let mut verdicts: Vec<(String, bool)> = Vec::new();
    let mut last_params = FlashMlaParams::default();
    let mut total_exact = 0usize;
    let mut total_len = 0usize;
    let mut aggregate_max_delta = 0.0f32;
    let mut aggregate_max_ulp = 0i32;
    for (step, &position) in positions.iter().enumerate() {
        let (codes, scales) = &xhat[position];
        let q_latent_bits = mxfp8_ref::dense_gemm(
            codes,
            scales,
            &wq_a_weight,
            &wq_a_scale,
            1,
            Q_LORA_RANK,
            HIDDEN,
        );
        let mut q_latent: Vec<f32> = q_latent_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        rms_norm_row(&mut q_latent, &q_norm, NORM_EPS);
        let (q_codes, q_scales) = act_quant_fp8_row(&q_latent);
        let q_bits = mxfp8_ref::dense_gemm(
            &q_codes,
            &q_scales,
            &wq_b_weight,
            &wq_b_scale,
            1,
            MLA_HEADS * MLA_LATENT,
            Q_LORA_RANK,
        );
        let mut q: Vec<f32> = q_bits
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();
        rotate_rope_tail(
            &mut q,
            MLA_HEADS,
            MLA_LATENT,
            ROPE_HEAD_DIM,
            ROPE_THETA,
            position as u32,
        );
        // The kernel consumes bf16, so the value both sides see is the bf16 rounding of the
        // rotated query: rounding it once here keeps the comparison about the attention, not about
        // where the rounding point is.
        let q_device = bf16_bits_of_f32(&q);
        let q_used: Vec<f32> = q_device
            .iter()
            .map(|bits| half::bf16::from_bits(*bits).to_f32())
            .collect();

        // The engine window rows for this query position: the last min(position + 1, 128)
        // positions, oldest first. Row r is position first + r, so the newest row is always the
        // query position and the oldest rows drop off once the window is full: p=127 has 128 rows
        // (positions 0..=127, row 0 = position 0); p=128 drops position 0 (rows 1..=128, row 0 =
        // position 1, row 127 = position 128); p=129 also drops position 1 (rows 2..=129). This is
        // the engine's order: get_window_topk_idxs returns ascending chunk ids for prefill and
        // oldest-to-newest ring slot order for decode, which is the same supplied order.
        let (first, window_row_count) = if format == WindowKvFormat::Engine {
            let first = position.saturating_sub(WINDOW_ROWS - 1);
            (first, position - first + 1)
        } else {
            (0usize, 0usize)
        };
        if format == WindowKvFormat::Engine {
            // The order is asserted, not assumed: the rows are exactly the last window_row_count
            // positions, oldest first, and the last row is the query position.
            if window_row_count != (position + 1).min(WINDOW_ROWS)
                || first + window_row_count - 1 != position
            {
                return Err(format!(
                    "ring order assertion failed at position {position}: first {first}, rows {window_row_count}"
                ));
            }
            let dropped = if first == 0 {
                "none".to_string()
            } else {
                format!("positions 0..{first}")
            };
            println!(
                "  ring order: {window_row_count} rows supplied oldest -> newest; row r -> position {first} + r (row 0 = position {first}, last row = position {position}); dropped oldest {dropped}; the 128-row window has wrapped: {}",
                position >= WINDOW_ROWS
            );
        }
        let selected: Vec<u32> = if format == WindowKvFormat::Engine {
            (0..window_row_count as u32).collect()
        } else {
            stage2_selected(position)
        };
        let in_window = if format == WindowKvFormat::Engine {
            selected.len()
        } else {
            selected
                .iter()
                .filter(|candidate| mla_ref::in_window(position as u32, **candidate))
                .count()
        };

        cuda.copy_htod(&q_d, &bf16_bits_to_bytes(&q_device))?;
        cuda.copy_htod(&selected_d, &u32s_to_bytes(&selected))?;
        if format == WindowKvFormat::Engine {
            let code_start = first * MLA_LATENT;
            let scale_start = first * WINDOW_SCALE_BYTES;
            cuda.copy_htod(
                &window_codes_d,
                &window_codes[code_start..code_start + window_row_count * MLA_LATENT],
            )?;
            cuda.copy_htod(
                &window_scales_d,
                &window_scales[scale_start..scale_start + window_row_count * WINDOW_SCALE_BYTES],
            )?;
        }
        let params = flash_mla_params(
            q_d.0,
            kv_fp4_d.0,
            kv_scales_d.0,
            window_codes_d.0,
            window_scales_d.0,
            format.kernel_value() as u64,
            window_row_count as u64,
            selected_d.0,
            index_key_d.0,
            index_query_d.0,
            sink_d.0,
            block_scores_d.0,
            block_best_d.0,
            out_d.0,
            position as u64,
            selected.len(),
            0, // trace_ring: null, the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
            args.layer,
            step as u64,
        );
        last_params = params;
        // Determinism: the same launch on the same inputs runs twice and the two device writes are
        // compared byte for byte. A mismatch is a hard failure, not a statistic.
        let mut device_out = vec![0u8; out_bytes];
        for run in 0..2usize {
            check_floor(
                args,
                &format!("stage 2 position {position} run {run} pre-launch"),
            )?;
            flash_mla_launch(&cuda, func, grid, block, &params)?;
            check_floor(
                args,
                &format!("stage 2 position {position} run {run} post-launch"),
            )?;
            let mut run_out = vec![0u8; out_bytes];
            cuda.copy_dtoh(&mut run_out, &out_d)?;
            if run == 0 {
                device_out = run_out;
            } else if run_out != device_out {
                return Err(format!(
                    "stage 2 determinism failed at position {position}: two runs on the same inputs wrote different bytes"
                ));
            }
        }
        let device_bits: Vec<u16> = device_out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| u16::from_le_bytes(*chunk))
            .collect();
        let reference_for = |list: &[u32]| -> Vec<f32> {
            let mut reference = Vec::with_capacity(MLA_HEADS * MLA_LATENT);
            for head in 0..MLA_HEADS {
                let q_head = &q_used[head * MLA_LATENT..(head + 1) * MLA_LATENT];
                let head_out = if format == WindowKvFormat::Engine {
                    let code_start = first * MLA_LATENT;
                    let scale_start = first * WINDOW_SCALE_BYTES;
                    mla_ref::attention_with_sink_window(
                        q_head,
                        &window_codes[code_start..code_start + window_row_count * MLA_LATENT],
                        &window_scales
                            [scale_start..scale_start + window_row_count * WINDOW_SCALE_BYTES],
                        window_row_count,
                        list,
                        sink[head],
                    )
                } else {
                    mla_ref::attention_with_sink(
                        q_head,
                        &kv_fp4,
                        &kv_scales,
                        list,
                        position as u32,
                        sink[head],
                    )
                };
                reference.extend_from_slice(&head_out);
            }
            reference
        };
        let reference = reference_for(&selected);
        println!(
            "--- query position {position}: {} candidates, {in_window} in window, {} excluded ---",
            selected.len(),
            selected.len() - in_window
        );
        let (stats, max_ulp) = compare_bf16_ulp("attention out", &device_bits, &reference)?;
        total_exact += stats.exact;
        total_len += stats.len;
        aggregate_max_delta = aggregate_max_delta.max(stats.max_delta);
        aggregate_max_ulp = aggregate_max_ulp.max(max_ulp);
        let ok = stats.exact_pct() >= STAGE2_EXACT_BAR && stats.max_delta <= STAGE2_DELTA_BAR;
        println!(
            "  position {position} verdict: {} (exact {:.4}% against a {STAGE2_EXACT_BAR}% bar, max abs delta {:e} against a {STAGE2_DELTA_BAR:e} bar, max ULP distance {max_ulp})",
            if ok {
                "within tolerance"
            } else {
                "OVER TOLERANCE"
            },
            stats.exact_pct(),
            stats.max_delta,
        );
        verdicts.push((format!("position {position}"), ok));

        if format == WindowKvFormat::Engine {
            // Out-of-window ids in engine mode must contribute exactly zero, not merely be left out
            // of a list: ids at and past the row count, and u32::MAX, are added to the candidate
            // list. The device bytes must not change at all, and the reference with the injected
            // ids must equal the reference without them, bit for bit.
            let mut injected = selected.clone();
            let injected_outsiders = [
                window_row_count as u32,
                window_row_count as u32 + 17,
                u32::MAX,
            ];
            injected.extend(injected_outsiders);
            // Non-vacuousness: the ids really are outside the window and really are in the list
            // handed to the kernel. Without this check an injection that silently dropped the ids
            // (or admitted them) would compare byte-identical for the wrong reason.
            if !injected_outsiders
                .iter()
                .all(|id| !mla_ref::window_attendable(window_row_count, *id))
            {
                return Err(format!(
                    "out-of-window injection is vacuous at position {position}: an injected id is inside the window"
                ));
            }
            if !injected_outsiders.iter().all(|id| injected.contains(id)) {
                return Err(format!(
                    "out-of-window injection is vacuous at position {position}: an injected id is missing from the candidate list"
                ));
            }
            if injected.len() != selected.len() + injected_outsiders.len() {
                return Err(format!(
                    "out-of-window injection count changed at position {position}"
                ));
            }
            cuda.copy_htod(&selected_d, &u32s_to_bytes(&injected))?;
            let injected_params = params.with_selected_count(injected.len());
            flash_mla_launch(&cuda, func, grid, block, &injected_params)?;
            let mut injected_out = vec![0u8; out_bytes];
            cuda.copy_dtoh(&mut injected_out, &out_d)?;
            let injected_reference = reference_for(&injected);
            if injected_out != device_out || injected_reference != reference {
                return Err(format!(
                    "out-of-window zeroing failed at position {position}: injected ids changed the output or the reference"
                ));
            }
            // An all-outside list is exactly zero on both sides even with the sink in the
            // denominator: the numerator is empty and the divide is by one. The list is the same
            // three verified outsiders, and no in-window id may be present.
            let all_outside = injected_outsiders;
            if all_outside
                .iter()
                .any(|id| mla_ref::window_attendable(window_row_count, *id))
            {
                return Err(format!(
                    "all-outside injection is vacuous at position {position}: the all-outside list contains an in-window id"
                ));
            }
            cuda.copy_htod(&selected_d, &u32s_to_bytes(&all_outside))?;
            let outside_params = params.with_selected_count(all_outside.len());
            flash_mla_launch(&cuda, func, grid, block, &outside_params)?;
            let mut outside_out = vec![0u8; out_bytes];
            cuda.copy_dtoh(&mut outside_out, &out_d)?;
            let outside_reference = reference_for(&all_outside);
            if !outside_out.iter().all(|byte| *byte == 0)
                || !outside_reference.iter().all(|value| *value == 0.0)
            {
                return Err(format!(
                    "all-out-of-window output is not exactly zero at position {position}"
                ));
            }
            println!(
                "  out-of-window ids at and past the {window_row_count} rows (plus u32::MAX): {} injected ids verified outside the window and present in the candidate list; device output byte-identical with and without them, reference equal, all-outside output exactly zero on both sides: PASS",
                injected_outsiders.len()
            );
            verdicts.push((format!("position {position} out-of-window zeroing"), true));
        }
    }
    println!(
        "format {}: totals across {} positions: {total_exact}/{total_len} exact ({:.4}%), max abs delta {:e}, max ULP {aggregate_max_ulp}; determinism: two device runs per position, byte-identical",
        format.label(),
        positions.len(),
        100.0 * total_exact as f64 / total_len as f64,
        aggregate_max_delta
    );
    if format == WindowKvFormat::Fp4 && tokens == 130 {
        let positions_text = positions
            .iter()
            .map(|position| position.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let merged_chain = total_exact == STAGE2_MERGED_CHAIN_EXACT
            && total_len == STAGE2_MERGED_CHAIN_TOTAL
            && aggregate_max_delta == STAGE2_MERGED_CHAIN_MAX_DELTA
            && aggregate_max_ulp == STAGE2_MERGED_CHAIN_MAX_ULP;
        println!(
            "merged-chain FP4 equality: {} (positions [{positions_text}]; expected {STAGE2_MERGED_CHAIN_EXACT}/{STAGE2_MERGED_CHAIN_TOTAL} exact, max abs delta {:e}, max ULP {} - the paired merged-head old-kernel diagnostic numbers on this chain; got {total_exact}/{total_len}, {:e}, {aggregate_max_ulp})",
            if merged_chain { "PASS" } else { "FAIL" },
            STAGE2_MERGED_CHAIN_MAX_DELTA,
            STAGE2_MERGED_CHAIN_MAX_ULP,
            aggregate_max_delta
        );
        println!(
            "historical FP4 note: 9.765625e-4 was the pre-D8 stage-1 chain, measured with the retired B0 binary (phase1-b0-stage2-layer0-20261018T010919Z.log); on the merged chain the merged-head old-kernel binary and this binary write identical per-position lines (phase1-task3-diag-oldkernel-20261018T060534Z.log vs phase1-task3-diag-newkernel-20261018T060534Z.log), so the merged-chain expectation above is that old kernel's own measured number"
        );
        if !merged_chain {
            return Err(
                "the FP4 default path no longer reproduces the merged-chain Stage-2 numbers (the paired old-kernel diagnostic on the same chain is the reference)"
                    .to_string(),
            );
        }
    }

    if args.repeat > 1 {
        check_floor(args, "stage 2 timing pre-launch")?;
        let (mut per_launch, batched_ms) =
            flash_mla_launch_repeated(&cuda, func, grid, block, &last_params, args.repeat)?;
        check_floor(args, "stage 2 timing post-launch")?;
        per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
        let count = per_launch.len();
        let mean = per_launch.iter().sum::<f64>() / count as f64;
        println!(
            "timing: {count} launches at the last query position on the same buffers; per-launch ms min {:.4} p50 {:.4} mean {mean:.4} max {:.4}; batched ms/launch {batched_ms:.4}",
            per_launch[0],
            per_launch[count / 2],
            per_launch[count - 1]
        );
    }

    cuda.free(
        module,
        &[
            q_d,
            kv_fp4_d,
            kv_scales_d,
            window_codes_d,
            window_scales_d,
            selected_d,
            index_key_d,
            index_query_d,
            block_scores_d,
            block_best_d,
            out_d,
            sink_d,
        ],
    );
    check_floor(args, "stage 2 end")?;
    let failed: Vec<&String> = verdicts
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(name, _)| name)
        .collect();
    if failed.is_empty() {
        println!(
            "stage 2: PASS ({} of {} printed checks within the declared class: per-position bars plus, for the engine format, the out-of-window zeroing assertions)",
            verdicts.len(),
            verdicts.len()
        );
        Ok(())
    } else {
        Err(format!("stage 2: FAIL at {failed:?}"))
    }
}
