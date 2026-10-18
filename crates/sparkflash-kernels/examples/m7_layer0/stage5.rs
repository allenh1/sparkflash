use super::*;

/// The LM head's geometry, from kernels/m4_common_sm121.cuh and kernels/cert_lm_head_sm121.cu.
/// The kernel asserts these internally; the launcher must pass the same values or the block count
/// and the per-rank vocabulary disagree with the buffer sizes.
pub(crate) const LM_HIDDEN: usize = 5120;
pub(crate) const LM_VOCAB: usize = 129_280;
const LM_VOCAB_PER_RANK: usize = 32_320;
const LM_THREADS: u32 = 128;
const LM_CHUNK: usize = 1024;
const LM_GRID: u32 = 32;

/// The tolerance for an exact dot product recomputed on the host in the kernel's own order. The
/// kernel may contract mul+add into an FMA while the host does not; 5120 bf16 terms of this
/// magnitude leave a difference far below this bar, and the harness prints the measured maximum so
/// the bar is checkable rather than asserted.
const LM_EXACT_BAR: f32 = 2.0e-2;

/// The slack for the certificate check: a certified entry stores an upper bound, and it must not be
/// below the matched reference by more than f32 accumulation error. This is much tighter than
/// LM_EXACT_BAR on purpose; the bar is for the exact dots, this is for the bound. Would the bound
/// be below the truth by an accumulation amount, the certificate would be wrong and this catches it.
const LM_CERTIFICATE_EPS: f32 = 1.0e-5;

const _: () = assert!(
    LM_GRID as usize == LM_VOCAB_PER_RANK.div_ceil(LM_CHUNK),
    "the grid covers the per-rank vocabulary in whole chunks"
);

pub(crate) fn read_f32_file(path: &Path, expect: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != expect * 4 {
        return Err(format!(
            "{}: expected {expect} f32 values ({} bytes), got {} bytes",
            path.display(),
            expect * 4,
            bytes.len()
        ));
    }
    Ok(bytes_to_f32(&bytes))
}

pub(crate) fn bf16_code_to_f32(code: u16) -> f32 {
    f32::from_bits(u32::from(code) << 16)
}

/// The exact logits the kernel would compute if it pruned nothing, in the kernel's own order: the
/// same bf16 hidden codes and the same per-entry ascending dot over the row. This is the
/// quantisation-matched reference: comparing the device against a reference computed from the f32
/// head input would fold the bf16 conversion of the hidden vector into the head's error.
fn host_quantised_logits(hidden: &[u16], head: &[u8]) -> Vec<f32> {
    let mut out = vec![0.0f32; LM_VOCAB];
    for (entry, logit) in out.iter_mut().enumerate() {
        let row = &head[entry * LM_HIDDEN * 2..(entry + 1) * LM_HIDDEN * 2];
        let mut acc = 0.0f32;
        for i in 0..LM_HIDDEN {
            let w = u16::from_le_bytes([row[2 * i], row[2 * i + 1]]);
            acc += bf16_code_to_f32(hidden[i]) * bf16_code_to_f32(w);
        }
        *logit = acc;
    }
    out
}

pub(crate) fn lm_argmax(logits: &[f32]) -> (f32, usize) {
    let mut best = logits[0];
    let mut at = 0usize;
    for (index, value) in logits.iter().enumerate().skip(1) {
        // Ascending scan: a strict comparison keeps the first (lowest) index on a tie.
        if *value > best {
            best = *value;
            at = index;
        }
    }
    (best, at)
}

/// Stage 5: the certified LM head against a host reference on one token's hidden state.
///
/// cert_lm_head_sm121 is launched once per tensor-parallel rank against that rank's 32,320 rows of
/// head.weight, with the hidden vector the reference's own head multiplied (boundary_dump
/// --head-input-out, converted to the kernel's bf16 ABI). The reference side is recomputed here
/// from the same bf16 codes in the kernel's own order, so the comparison is the head's arithmetic
/// and not the hidden vector's bf16 rounding.
///
/// The kernel's contract is the argmax, not the top-5: a certified entry stores its upper bound,
/// not its exact value, so the top-5 set cannot be read from the kernel's output. This stage
/// therefore checks the argmax, the exact entries, and the certificate itself (every certified
/// bound is still an upper bound on the matched reference), and prints the measured numbers.
pub(crate) fn stage5(args: &Args) -> Result<(), String> {
    let head_input_path = args
        .head_input
        .as_ref()
        .ok_or("--stage 5 needs --head-input FILE (boundary_dump --head-input-out)")?;
    let logits_path = args
        .logits
        .as_ref()
        .ok_or("--stage 5 needs --logits FILE (boundary_dump --logits-out)")?;
    let hidden_f32 = read_f32_file(head_input_path, LM_HIDDEN)?;
    let reference = read_f32_file(logits_path, LM_VOCAB)?;
    let hidden = bf16_bits_of_f32(&hidden_f32);
    let hidden_bytes = bf16_bits_to_bytes(&hidden);

    let mut snapshot = Snapshot::open(&args.checkpoint)?;
    let head = snapshot.read_raw("head.weight", "BF16", &[LM_VOCAB, LM_HIDDEN])?;
    println!(
        "stage 5: head.weight BF16 [{} x {}] ({} B), head input {} f32 ({} B), reference logits {} f32",
        LM_VOCAB,
        LM_HIDDEN,
        head.len(),
        hidden_f32.len(),
        hidden_f32.len() * 4,
        reference.len()
    );
    let matched = host_quantised_logits(&hidden, &head);
    let (reference_max, reference_at) = lm_argmax(&reference);
    let (matched_max, matched_at) = lm_argmax(&matched);
    println!(
        "host reference (f32 hidden): argmax {reference_at} value {reference_max:.9}; quantisation-matched (bf16 hidden, kernel order): argmax {matched_at} value {matched_max:.9}"
    );

    let cuda = Cuda::load()?;
    let cubin_path = args.cubin_dir.join("cert_lm_head_sm121.cubin");
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (nvcc -cubin -arch=sm_121a -I kernels -o {} kernels/cert_lm_head_sm121.cu)",
            cubin_path.display(),
            cubin_path.display()
        )
    })?;
    let (_module, func) = cuda.load_kernel(&cubin, KernelId::CertLmHead)?;
    let planned = (LM_HIDDEN * 2
        + LM_VOCAB * LM_HIDDEN * 2
        + LM_VOCAB_PER_RANK * 4
        + LM_VOCAB_PER_RANK
        + 4
        + (LM_GRID as usize) * 4
        + (LM_GRID as usize) * 4) as u64;
    admit_memory(args, planned, "stage 5 device buffers")?;
    let mut allocated = 0usize;
    let hidden_d = cuda.alloc(LM_HIDDEN * 2, &mut allocated)?;
    let weight_d = cuda.alloc(LM_VOCAB * LM_HIDDEN * 2, &mut allocated)?;
    let logits_d = cuda.alloc(LM_VOCAB_PER_RANK * 4, &mut allocated)?;
    let certified_d = cuda.alloc(LM_VOCAB_PER_RANK, &mut allocated)?;
    let certified_count_d = cuda.alloc(4, &mut allocated)?;
    let block_max_d = cuda.alloc((LM_GRID as usize) * 4, &mut allocated)?;
    let block_argmax_d = cuda.alloc((LM_GRID as usize) * 4, &mut allocated)?;
    println!("stage 5: allocated {allocated} B on device 0");
    check_floor(args, "stage 5 pre-upload")?;
    cuda.copy_htod(&hidden_d, &hidden_bytes)?;
    cuda.copy_htod(&weight_d, &head)?;
    check_floor(args, "stage 5 post-upload")?;

    let rank_stride = (LM_VOCAB_PER_RANK * LM_HIDDEN * 2) as u64;
    let mut best = f32::NEG_INFINITY;
    let mut best_entry = 0usize;
    let mut exact = 0usize;
    let mut pruned = 0usize;
    let mut max_exact_delta = 0.0f32;
    let mut at_max_delta = 0usize;
    let mut certificate_violations = 0usize;
    let mut per_rank = Vec::new();
    for rank in 0..4usize {
        let weight_ptr = weight_d.0 + (rank as u64) * rank_stride;
        let params: [u64; 11] = [
            hidden_d.0,
            weight_ptr,
            logits_d.0,
            certified_d.0,
            certified_count_d.0,
            block_max_d.0,
            block_argmax_d.0,
            LM_VOCAB_PER_RANK as u64,
            0,
            args.layer as u64,
            0,
        ];
        check_floor(args, &format!("stage 5 rank {rank} pre-launch"))?;
        // The kernel atomically accumulates into this buffer, so it is per-launch state: without
        // re-zeroing, rank 1 would read rank 0's count (and rank 0 uninitialized memory).
        cuda.copy_htod(&certified_count_d, &[0u8; 4])?;
        cuda.launch(func, (LM_GRID, 1, 1), (LM_THREADS, 1, 1), &params)?;
        check_floor(args, &format!("stage 5 rank {rank} post-launch"))?;
        let mut logits_bytes = vec![0u8; LM_VOCAB_PER_RANK * 4];
        let mut certified_bytes = vec![0u8; LM_VOCAB_PER_RANK];
        let mut count_bytes = [0u8; 4];
        let mut block_max_bytes = vec![0u8; (LM_GRID as usize) * 4];
        let mut block_argmax_bytes = vec![0u8; (LM_GRID as usize) * 4];
        cuda.copy_dtoh(&mut logits_bytes, &logits_d)?;
        cuda.copy_dtoh(&mut certified_bytes, &certified_d)?;
        cuda.copy_dtoh(&mut count_bytes, &certified_count_d)?;
        cuda.copy_dtoh(&mut block_max_bytes, &block_max_d)?;
        cuda.copy_dtoh(&mut block_argmax_bytes, &block_argmax_d)?;
        let rank_logits = bytes_to_f32(&logits_bytes);
        // The kernel publishes its own per-block maximum and argmax; cross-check that reduction
        // against the per-entry values instead of ignoring it. A disagreement is a kernel bug even
        // when the global argmax happens to survive.
        let block_max = bytes_to_f32(&block_max_bytes);
        let block_argmax: Vec<u32> = block_argmax_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        for block in 0..LM_GRID as usize {
            let lo = block * LM_CHUNK;
            let hi = (lo + LM_CHUNK).min(LM_VOCAB_PER_RANK);
            let mut local_best = f32::NEG_INFINITY;
            let mut local_at = 0u32;
            for (entry, value) in rank_logits[lo..hi].iter().enumerate() {
                if *value > local_best || (*value == local_best && ((lo + entry) as u32) < local_at)
                {
                    local_best = *value;
                    local_at = (lo + entry) as u32;
                }
            }
            if block_max[block] != local_best || block_argmax[block] != local_at {
                return Err(format!(
                    "stage 5 rank {rank} block {block}: kernel reduction (max {} entry {}) disagrees with the per-entry scan (max {} entry {})",
                    block_max[block], block_argmax[block], local_best, local_at
                ));
            }
        }
        let rank_exact = certified_bytes.iter().filter(|c| **c == 0).count();
        let rank_pruned = u32::from_le_bytes(count_bytes) as usize;
        if rank_pruned != LM_VOCAB_PER_RANK - rank_exact {
            return Err(format!(
                "stage 5 rank {rank}: the kernel's certified_count {rank_pruned} disagrees with the certified flags ({} certified)",
                LM_VOCAB_PER_RANK - rank_exact
            ));
        }
        per_rank.push((rank_exact, rank_pruned));
        for (entry, value) in rank_logits.iter().enumerate() {
            let global = rank * LM_VOCAB_PER_RANK + entry;
            if *value > best || (*value == best && global < best_entry) {
                best = *value;
                best_entry = global;
            }
            if certified_bytes[entry] == 0 {
                exact += 1;
                let delta = (*value - matched[global]).abs();
                if delta > max_exact_delta {
                    max_exact_delta = delta;
                    at_max_delta = global;
                }
            } else {
                pruned += 1;
                if *value + LM_CERTIFICATE_EPS < matched[global] {
                    certificate_violations += 1;
                }
            }
        }
    }
    println!(
        "device: argmax {best_entry} value {best:.9}; exact {exact}, certified {pruned} ({:.2}%), per-rank exact/certified {:?}",
        100.0 * pruned as f64 / LM_VOCAB as f64,
        per_rank
    );
    println!(
        "exact entries: max |device - host(kernel order)| {max_exact_delta:e} at entry {} (provisional bar {LM_EXACT_BAR:e}; the released token measured 0 against the matched reference); certificate violations {certificate_violations}",
        if max_exact_delta == 0.0 {
            "n/a".to_string()
        } else {
            at_max_delta.to_string()
        }
    );
    if pruned == 0 {
        println!(
            "certificate: not exercised on this token (0 of {LM_VOCAB} entries certified), so the bound check is vacuous here and is not claimed as evidence"
        );
    }
    let argmax_ok = best_entry == matched_at;
    let reference_ok = best_entry == reference_at;
    println!(
        "argmax: device {best_entry}, matched-reference {matched_at}, f32-reference {reference_at} (matched {}; f32 {})",
        if argmax_ok { "equal" } else { "DIFFERENT" },
        if reference_ok { "equal" } else { "DIFFERENT" }
    );
    if !argmax_ok || !reference_ok || max_exact_delta > LM_EXACT_BAR || certificate_violations != 0
    {
        return Err(format!(
            "stage 5: FAIL (argmax device {best_entry} vs matched {matched_at} vs reference {reference_at}; max exact delta {max_exact_delta:e} bar {LM_EXACT_BAR:e}; certificate violations {certificate_violations})"
        ));
    }
    let certificate = if pruned == 0 {
        "the certificate was not exercised (0 certified entries)".to_string()
    } else {
        format!("{pruned} certified bounds all upper bounds")
    };
    println!(
        "stage 5: PASS (argmax {best_entry} equal to both references; {exact} exact entries within the bar; {certificate})"
    );
    Ok(())
}
