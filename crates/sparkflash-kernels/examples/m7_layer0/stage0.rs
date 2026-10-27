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

/// One captured block input's quantised tensors, recomputed from the row's own f32 values.
///
/// The engine wrote pre_*_hidden (the f32 normalised block input), pre_*_xhat (E4M3 codes) and
/// pre_*_scales (ue8m0) through quant::quantize_activation_fp8, the reference ceil rule.
/// [act_quant_fp8_row] is that same rule, so this checks the rule the device kernels now carry
/// against the engine's OWN captured arithmetic, not against another copy of the device reference.
/// The retired floor-rule codes are printed unjudged: they are the size of the gap D8 closed.
///
/// Returns (code mismatches, scale mismatches).
fn stage0_activation_quantisers(
    table: &HostBoundaryTable,
    prefix: &str,
    hidden_name: &str,
    xhat_name: &str,
    scales_name: &str,
) -> Result<(usize, usize), String> {
    let hidden = table.f32(0, hidden_name)?;
    let captured_codes = table.u8(0, xhat_name)?;
    let captured_scales = table.u8(0, scales_name)?;
    if hidden.len() != HIDDEN {
        return Err(format!(
            "row 0 {hidden_name}: expected {HIDDEN} f32 values, got {}",
            hidden.len()
        ));
    }
    if captured_codes.len() != HIDDEN || captured_scales.len() != HIDDEN / QUANT_BLOCK {
        return Err(format!(
            "row 0 {xhat_name}: expected {HIDDEN} codes and {} scales, got {} and {}",
            HIDDEN / QUANT_BLOCK,
            captured_codes.len(),
            captured_scales.len()
        ));
    }
    let (codes, scales) = act_quant_fp8_row(&hidden);
    let code_mismatches = codes
        .iter()
        .zip(&captured_codes)
        .filter(|(recomputed, captured)| recomputed != captured)
        .count();
    let scale_mismatches = scales
        .iter()
        .zip(&captured_scales)
        .filter(|(recomputed, captured)| recomputed != captured)
        .count();
    // The retired floor-log2 weight-grid rule on the same f32 values, for the gap's size only.
    let mut floor_codes = [0u8; HIDDEN];
    let mut floor_scales = [0u8; HIDDEN / QUANT_BLOCK];
    let (code_blocks, _) = floor_codes.as_chunks_mut::<QUANT_BLOCK>();
    for (block, (scale_slot, code_slot)) in floor_scales.iter_mut().zip(code_blocks).enumerate() {
        let hidden_block = &hidden[block * QUANT_BLOCK..(block + 1) * QUANT_BLOCK];
        let amax = hidden_block
            .iter()
            .fold(0.0f32, |amax, value| amax.max(value.abs()));
        let code = ue8m0_from_amax(amax, E4M3_ELEMENT_EMAX);
        *scale_slot = code;
        let scale = ue8m0_decode(code);
        for (dst, value) in code_slot.iter_mut().zip(hidden_block) {
            *dst = fp8_e4m3_encode(value / scale);
        }
    }
    let floor_code_mismatches = floor_codes
        .iter()
        .zip(&captured_codes)
        .filter(|(recomputed, captured)| recomputed != captured)
        .count();
    let floor_scale_mismatches = floor_scales
        .iter()
        .zip(&captured_scales)
        .filter(|(recomputed, captured)| recomputed != captured)
        .count();
    println!(
        "  {prefix} activation quantiser: ceil-rule codes {}/{HIDDEN} exact, scales {}/{} exact; \
         retired floor rule would give {}/{HIDDEN} codes and {}/{} scales exact (not judged)",
        HIDDEN - code_mismatches,
        HIDDEN / QUANT_BLOCK - scale_mismatches,
        HIDDEN / QUANT_BLOCK,
        HIDDEN - floor_code_mismatches,
        HIDDEN / QUANT_BLOCK - floor_scale_mismatches,
        HIDDEN / QUANT_BLOCK,
    );
    Ok((code_mismatches, scale_mismatches))
}

/// Stage 0: the fold-closure cross-check, CPU only.
///
/// It verifies the B0 claim against the engine's OWN captured arithmetic rather than against another
/// copy of the reference: the released projection (all four copies, no averaging) must reproduce the
/// coefficient tensors the host boundary row 0 recorded, and the retired fold must not. No CUDA
/// context is created, so this runs while vLLM holds the node.
pub(crate) fn stage0(args: &Args) -> Result<(), String> {
    let host_row = args
        .host_row
        .as_ref()
        .ok_or("--stage 0 needs --host-row <manifest prefix>")?;
    println!(
        "=== stage 0: the released projection against the engine's captured row-0 coefficients ==="
    );
    println!(
        "comparison class: HOST-ONLY. The released hc_*_fn [24, n*d] matrices are projected over the \
         flattened four-copy stream with the reference's released arithmetic, and the resulting \
         pre/post/comb coefficients are diffed against the tensors the engine captured in boundary \
         row 0. No CUDA context is created."
    );
    println!(
        "pass bar: every full-projection coefficient within 1e-5 absolute of the engine's captured \
         value (both sides are f32 sequential reductions over the same rows, so the expected delta \
         is a few ULP). The retired-fold column is printed as evidence of the size of the semantic \
         gap and is not judged."
    );
    if args.layer != 0 {
        return Err(format!(
            "stage 0 cross-checks captured row 0, so it needs --layer 0, got {}",
            args.layer
        ));
    }
    println!("host pair: {}", host_row.display());
    let table = HostBoundaryTable::open(host_row)?;
    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The attn sublayer's streams are the row's residual_in (the updated X_l the block entered
    // with); the ffn sublayer's streams are the row's post_attention_hidden. Both are captured
    // [positions, hc, dim] f32 and one token is one 4*5120 stream.
    let residual_in = table.f32(0, "residual_in")?;
    let post_attention = table.f32(0, "post_attention_hidden")?;
    if residual_in.len() != HC_MULT * HIDDEN || post_attention.len() != HC_MULT * HIDDEN {
        return Err(format!(
            "row 0: expected two {}-value streams, got {} and {}",
            HC_MULT * HIDDEN,
            residual_in.len(),
            post_attention.len()
        ));
    }
    let mut worst = 0.0f32;
    let mut quant_mismatches = 0usize;
    for (sublayer, streams, prefix) in [
        ("attn", &residual_in, "hc_attn"),
        ("ffn", &post_attention, "hc_ffn"),
    ] {
        let mhc = read_released_mhc_tensors(&mut snap, "layers.0", sublayer)?;
        println!(
            "{sublayer}: predictor {} f32 (released [24, 4*d], rows in kernel order, no averaging)",
            mhc.predictor.len()
        );
        let (_, coefficients, _, _) = mega_mhc_layer_with_scale(
            &mut Streams::new(streams.clone()),
            &vec![1.0f32; HIDDEN],
            &[1.0, 0.0, 0.0, 0.0],
            &mhc.predictor,
            &mhc.hc_base,
            &mhc.hc_scale,
            &mut TrafficLedger::default(),
        );
        // The captured coefficients are in the release's layouts: comb [source][destination],
        // pre and post one value per copy. The reference's B is [destination][source].
        let mut b_reference = vec![0.0f32; HC_MULT * HC_MULT];
        for destination in 0..HC_MULT {
            for source in 0..HC_MULT {
                b_reference[source * HC_MULT + destination] = coefficients.b[destination][source];
            }
        }
        let pre = table.f32(0, &format!("{prefix}_pre"))?;
        let post = table.f32(0, &format!("{prefix}_post"))?;
        let comb = table.f32(0, &format!("{prefix}_comb"))?;
        let (_, d_pre, _) =
            compare_row0_coefficient(&format!("{prefix}_pre"), &coefficients.a, &pre)?;
        let (_, d_post, _) =
            compare_row0_coefficient(&format!("{prefix}_post"), &coefficients.c, &post)?;
        let (_, d_comb, _) =
            compare_row0_coefficient(&format!("{prefix}_comb"), &b_reference, &comb)?;
        worst = worst.max(d_pre).max(d_post).max(d_comb);

        // Non-vacuity: the pre-B0 fold. Its predictor rows are averaged over the four copies and its
        // projection is against the stream mean; it is computed explicitly so the reported delta
        // belongs to the retired semantics, not to a scaled full projection.
        let raw = retired_fold_raw_logits(&mhc.released, streams);
        let (folded_b, folded_a, folded_c) =
            released_coefficients_from_raw(&raw, &mhc.hc_base, &mhc.hc_scale);
        let mut folded_b_released = vec![0.0f32; HC_MULT * HC_MULT];
        for destination in 0..HC_MULT {
            for source in 0..HC_MULT {
                folded_b_released[source * HC_MULT + destination] =
                    folded_b[destination * HC_MULT + source];
            }
        }
        let (_, fold_pre, _) =
            compare_row0_coefficient(&format!("{prefix}_pre (retired fold)"), &folded_a, &pre)?;
        let (_, fold_post, _) =
            compare_row0_coefficient(&format!("{prefix}_post (retired fold)"), &folded_c, &post)?;
        let (_, fold_comb, _) = compare_row0_coefficient(
            &format!("{prefix}_comb (retired fold)"),
            &folded_b_released,
            &comb,
        )?;
        println!(
            "  {sublayer} retired fold vs captured: pre {:e}, post {:e}, comb {:e} (not judged)",
            fold_pre, fold_post, fold_comb
        );

        // D8: the captured activation-quantised block input, recomputed from the row's own f32
        // block input with the reference ceil rule. The engine wrote these tensors through its own
        // quantise_activation_fp8, so this is the rule change checked against the engine's
        // arithmetic, not against the device reference.
        let (quant_prefix, hidden_name, xhat_name, scales_name) = match sublayer {
            "attn" => (
                "pre_attention",
                "pre_attention_hidden",
                "pre_attention_xhat",
                "pre_attention_scales",
            ),
            "ffn" => (
                "pre_ffn",
                "pre_ffn_hidden",
                "pre_ffn_xhat",
                "pre_ffn_scales",
            ),
            other => return Err(format!("unknown sublayer {other}")),
        };
        let (code_mismatches, scale_mismatches) = stage0_activation_quantisers(
            &table,
            quant_prefix,
            hidden_name,
            xhat_name,
            scales_name,
        )?;
        quant_mismatches += code_mismatches + scale_mismatches;
    }
    println!("stage 0: full-projection worst max abs delta {worst:e} against a 1e-5 bar");
    println!(
        "stage 0: activation-quantiser mismatches {quant_mismatches} against a 0 bar (ceil rule must reproduce the engine's captured codes and scales)"
    );
    if worst <= 1e-5 && quant_mismatches == 0 {
        println!(
            "stage 0: PASS (the released projection reproduces the engine's captured row-0 coefficients, and the ceil-rule activation quantiser reproduces its captured xhat codes and scales)"
        );
        Ok(())
    } else {
        Err(format!(
            "stage 0: FAIL - worst projection delta {worst:e} (bar 1e-5), activation-quantiser mismatches {quant_mismatches} (bar 0)"
        ))
    }
}
