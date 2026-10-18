use super::*;
use crate::stage5::*;

/// The reference head's top-k entry indices, ordered by value then by lowest index. The kernel's
/// tie-break is the lowest entry index, so a comparison that ignored the index would disagree on a
/// tie even when both sides computed the same values.
pub(crate) fn top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]).then_with(|| a.cmp(b)));
    indices.truncate(k);
    indices
}

/// Stage 6: the CPU half of the criterion-3 head comparison, runnable while vLLM holds the node.
///
/// This stage never opens a CUDA context: it streams head.weight one 10 KB row at a time through the
/// checkpoint's own shard reader and recomputes every exact logit twice, once from the reference's f32
/// head input and once from the bf16 codes the kernel ABI is fed, both in the kernel's ascending order.
/// The two deltas it prints are the ones the device run will sit on top of: the bf16 conversion of
/// the hidden vector (what the device cannot avoid) and the host order versus the engine's matmul
/// order (what the matched reference must reproduce). The first number is why the device comparison
/// uses the matched reference instead of the f32-hidden logits; the head's own bar is measured by the
/// device run (which reported max abs delta 0 against the matched host on the released token), not
/// derived from the conversion.
pub(crate) fn stage6(args: &Args) -> Result<(), String> {
    let head_input_path = args
        .head_input
        .as_ref()
        .ok_or("--stage 6 needs --head-input FILE (boundary_dump --head-input-out)")?;
    let logits_path = args
        .logits
        .as_ref()
        .ok_or("--stage 6 needs --logits FILE (boundary_dump --logits-out)")?;
    let hidden_f32 = read_f32_file(head_input_path, LM_HIDDEN)?;
    let reference = read_f32_file(logits_path, LM_VOCAB)?;
    let hidden = bf16_bits_of_f32(&hidden_f32);
    let mut snapshot = Snapshot::open(&args.checkpoint)?;
    let header = snapshot.load("head.weight")?;
    if header.shape.len() != 2 || header.shape[0] != LM_VOCAB || header.shape[1] != LM_HIDDEN {
        return Err(format!(
            "head.weight: expected two dimensions [{LM_VOCAB}][{LM_HIDDEN}], the checkpoint stores {:?}",
            header.shape
        ));
    }
    let mut row = vec![0u8; LM_HIDDEN * 2];
    let mut matched_f32 = vec![0.0f32; LM_VOCAB];
    let mut matched_bf16 = vec![0.0f32; LM_VOCAB];
    for entry in 0..LM_VOCAB {
        snapshot.read_row_bf16_into("head.weight", entry, LM_HIDDEN, &mut row)?;
        let mut from_f32 = 0.0f32;
        let mut from_bf16 = 0.0f32;
        for i in 0..LM_HIDDEN {
            let w = bf16_code_to_f32(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
            from_f32 += hidden_f32[i] * w;
            from_bf16 += bf16_code_to_f32(hidden[i]) * w;
        }
        matched_f32[entry] = from_f32;
        matched_bf16[entry] = from_bf16;
    }
    let (reference_max, reference_at) = lm_argmax(&reference);
    let (f32_max, f32_at) = lm_argmax(&matched_f32);
    let (bf16_max, bf16_at) = lm_argmax(&matched_bf16);
    let mut delta_f32 = 0.0f32;
    let mut delta_bf16 = 0.0f32;
    for entry in 0..LM_VOCAB {
        delta_f32 = delta_f32.max((matched_f32[entry] - reference[entry]).abs());
        delta_bf16 = delta_bf16.max((matched_bf16[entry] - matched_f32[entry]).abs());
    }
    let reference_top = top_k(&reference, 5);
    let f32_top = top_k(&matched_f32, 5);
    let bf16_top = top_k(&matched_bf16, 5);
    println!(
        "stage 6 (CPU): reference argmax {reference_at} value {reference_max:.9}; recomputed f32 argmax {f32_at} value {f32_max:.9}; bf16-hidden argmax {bf16_at} value {bf16_max:.9}"
    );
    println!(
        "deltas: max |recomputed_f32 - reference| {delta_f32:e} (host order vs engine matmul); max |bf16_hidden - f32_hidden| {delta_bf16:e} (the unavoidable conversion the device ABI adds)"
    );
    println!("top-5: reference {reference_top:?}; f32 {f32_top:?}; bf16 {bf16_top:?}");
    let argmax_ok = reference_at == f32_at && reference_at == bf16_at;
    let top_ok = reference_top == f32_top && reference_top == bf16_top;
    // The f32 recomputation is the engine's order; measured max abs delta 0 on the released token,
    // and a nonzero value would mean the matched reference is no longer the head's arithmetic.
    let order_ok = delta_f32 == 0.0;
    if !argmax_ok || !top_ok || !order_ok {
        return Err(format!(
            "stage 6: FAIL (argmax reference {reference_at} f32 {f32_at} bf16 {bf16_at}; top-5 {:?} vs {:?} vs {:?}; f32-order delta {delta_f32:e})",
            reference_top, f32_top, bf16_top
        ));
    }
    println!(
        "stage 6: PASS (argmax and top-5 identical under both recomputations; the bf16 conversion moves a logit by at most {delta_bf16:e}, which is why the device comparison uses the matched reference rather than the f32-hidden logits)"
    );
    Ok(())
}
