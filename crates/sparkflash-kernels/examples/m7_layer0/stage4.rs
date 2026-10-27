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

/// One launch's exact device bytes. Stage 4 compares these byte for byte: there is no tolerance,
/// because the question is not whether two runs are close but whether they are the same run.
struct MhcRun {
    stats: Vec<u8>,
    coeffs: Vec<u8>,
    rms: Vec<u8>,
    fp8: Vec<u8>,
    scales: Vec<u8>,
    streams: Vec<u8>,
}

impl MhcRun {
    fn fields(&self) -> [(&'static str, &[u8]); 6] {
        [
            ("hc_stats", self.stats.as_slice()),
            ("hc_coeffs", self.coeffs.as_slice()),
            ("hc_rms_out", self.rms.as_slice()),
            ("xhat_fp8", self.fp8.as_slice()),
            ("xhat_scales", self.scales.as_slice()),
            ("X_l", self.streams.as_slice()),
        ]
    }
}

/// FNV-1a 64, for the one-line fingerprint each run prints. The verdict is a direct byte comparison;
/// the fingerprint is evidence a reader can scan, never the thing the test trusts.
fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Fail with the first differing byte and the two windows around it, so a reintroduced
/// nondeterminism names the buffer and offset instead of only saying "not equal".
fn require_identical_bytes(
    token: usize,
    run: usize,
    name: &str,
    expected: &[u8],
    observed: &[u8],
) -> Result<(), String> {
    if expected == observed {
        return Ok(());
    }
    let common = expected.len().min(observed.len());
    let at = expected
        .iter()
        .zip(observed)
        .position(|(a, b)| a != b)
        .unwrap_or(common);
    let lo = at.saturating_sub(8);
    let expected_window = &expected[lo..(at + 8).min(expected.len())];
    let observed_window = &observed[lo..(at + 8).min(observed.len())];
    Err(format!(
        "stage 4: token {token} run {run}: {name} is not byte-identical to run 0 - first difference at \
         byte {at} of {} vs {}; run 0 {:02x?}, run {run} {:02x?}. The same input produced different \
         output bytes, so the grid reduction is not deterministic.",
        expected.len(),
        observed.len(),
        expected_window,
        observed_window,
    ))
}

/// Stage 4: determinism is a property of the device, so it is measured there. N launches of the same
/// input bytes must produce the same output bytes; the harness does not compare against the CPU
/// reference here (stage 1 does that) and does not use a tolerance, because a tolerance is exactly
/// how a nondeterministic reduction hides.
pub(crate) fn stage4(args: &Args) -> Result<(), String> {
    if args.repeat < 2 {
        return Err(format!(
            "--stage 4 needs --repeat at least 2 (got {}): one run cannot disagree with itself",
            args.repeat
        ));
    }
    let kernel = "mega_mhc_fused_bf16_sm121";
    let prefix = format!("layers.{}", args.layer);
    println!(
        "=== stage 4: determinism, {} launches of the same input per token ===",
        args.repeat
    );
    println!(
        "comparison class: EXACT, run to run, with no tolerance and no reference. Run 0 is the \
         baseline for the runs after it, and every compared buffer must match it byte for byte."
    );
    println!(
        "determinism contract cited: D-a (kernel determinism) is the property this stage measures; \
         D-b (reduction determinism) is the comm crate's one-shot reduce property and is not on this \
         device path; D-c (no assumed cross-boundary equality) is why the comparison here is exact \
         bytes with no tolerance and stage 1 names a bar before its numbers"
    );
    println!(
        "what is compared, as raw bytes: hc_stats ({} floats: the final 26 and one 26-float partial \
         vector per tile), hc_coeffs (24 floats), hc_rms_out (1 float), xhat_fp8 (5120 E4M3 codes), \
         xhat_scales (160 ue8m0 bytes) and X_l ({} bf16 values)",
        MHC_GRID_STATS,
        HC_MULT * HIDDEN
    );
    println!(
        "why this is the direct test: the defect was an fp32 atomicAdd reduction whose order four \
         blocks did not fix, so the same input could publish different hc_stats and move E4M3 codes \
         across a boundary. Nothing here derives an expectation from the kernel; a single differing \
         byte fails the stage."
    );
    check_floor(args, "stage 4 start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let predictor = mhc.predictor;
    let hc_base_kernel = mhc.hc_base_kernel;
    let hc_scale = mhc.hc_scale;
    println!(
        "predictor: {} f32 = released [24, 4*d] rows permuted into kernel B/A/C order, NO averaging \
         (released row order into kernel B/A/C, no scale pre-folded); hc_attn_base in kernel row \
         order and hc_attn_scale {hc_scale:?} reach the kernel; attn_norm.weight: {} f32; the \
         layer-0 seed is B = identity, C = 0, A = [1, 0, 0, 0], Y_-1 = 0, X_-1 = four copies of one \
         embed row",
        predictor.len(),
        norm.len()
    );

    let mut b_prev = [0.0f32; HC_MULT * HC_MULT];
    for i in 0..HC_MULT {
        b_prev[i * HC_MULT + i] = 1.0;
    }
    let c_prev = [0.0f32; HC_MULT];
    let a_prev = [1.0f32, 0.0, 0.0, 0.0];
    let y_prev = vec![0.0f32; HIDDEN];

    let planned = (HC_MULT * HIDDEN * 2
        + HIDDEN * 2
        + b_prev.len() * 4
        + c_prev.len() * 4
        + a_prev.len() * 4
        + predictor.len() * 4
        + hc_scale.len() * 4
        + hc_base_kernel.len() * 4
        + norm.len() * 4
        + HIDDEN
        + HIDDEN / QUANT_BLOCK
        + MHC_GRID_STATS * 4
        + 2 * 4
        + 24 * 4
        + 4
        + HIDDEN * 4) as u64;
    let cuda = Cuda::load()?;
    let cubin_path = crate::cubin_path(&args.cubin_dir, kernel)?;
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the module docs give the nvcc line)",
            cubin_path.display()
        )
    })?;
    let (module, func) = cuda.load_kernel(&cubin, kernel_id(kernel)?)?;
    admit_memory(args, planned, "stage 4 device buffers")?;
    let mut allocated = 0usize;
    let x_d = cuda.alloc(HC_MULT * HIDDEN * 2, &mut allocated)?;
    let y_d = cuda.alloc(HIDDEN * 2, &mut allocated)?;
    let b_d = cuda.alloc(b_prev.len() * 4, &mut allocated)?;
    let c_d = cuda.alloc(c_prev.len() * 4, &mut allocated)?;
    let a_d = cuda.alloc(a_prev.len() * 4, &mut allocated)?;
    let predictor_d = cuda.alloc(predictor.len() * 4, &mut allocated)?;
    let scale_d = cuda.alloc(hc_scale.len() * 4, &mut allocated)?;
    let base_d = cuda.alloc(hc_base_kernel.len() * 4, &mut allocated)?;
    let norm_d = cuda.alloc(norm.len() * 4, &mut allocated)?;
    let fp8_d = cuda.alloc(HIDDEN, &mut allocated)?;
    let scales_d = cuda.alloc(HIDDEN / QUANT_BLOCK, &mut allocated)?;
    let stats_d = cuda.alloc(MHC_GRID_STATS * 4, &mut allocated)?;
    let sync_d = cuda.alloc(2 * 4, &mut allocated)?;
    let coeffs_d = cuda.alloc(24 * 4, &mut allocated)?;
    let rms_d = cuda.alloc(4, &mut allocated)?;
    let normed_d = cuda.alloc(HIDDEN * 4, &mut allocated)?;
    println!(
        "device bytes allocated: {allocated} ({} KiB)",
        allocated / 1024
    );

    cuda.copy_htod(&y_d, &bf16_bits_to_bytes(&bf16_bits_of_f32(&y_prev)))?;
    cuda.copy_htod(&b_d, &f32_to_bytes(&b_prev))?;
    cuda.copy_htod(&c_d, &f32_to_bytes(&c_prev))?;
    cuda.copy_htod(&a_d, &f32_to_bytes(&a_prev))?;
    cuda.copy_htod(&predictor_d, &f32_to_bytes(&predictor))?;
    cuda.copy_htod(&scale_d, &f32_to_bytes(&hc_scale))?;
    cuda.copy_htod(&base_d, &f32_to_bytes(&hc_base_kernel))?;
    cuda.copy_htod(&norm_d, &f32_to_bytes(&norm))?;

    let grid = (MHC_TILES, 1u32, 1u32);
    let block = (MHC_THREADS, 1u32, 1u32);
    let params: [u64; 19] = [
        x_d.0,
        y_d.0,
        b_d.0,
        c_d.0,
        a_d.0,
        predictor_d.0,
        scale_d.0,
        base_d.0,
        norm_d.0,
        fp8_d.0,
        scales_d.0,
        stats_d.0,
        sync_d.0,
        coeffs_d.0,
        rms_d.0,
        normed_d.0,
        0, // trace_ring: null, the kernel is built without -DSPARKFLASH_ENABLE_GPU_TRACE
        args.layer as u64,
        0,
    ];

    for token in 0..args.tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let x_bits = bf16_bits_of_f32(&x);
        let x_bytes = bf16_bits_to_bytes(&x_bits);

        let mut baseline: Option<MhcRun> = None;
        for run in 0..args.repeat {
            cuda.copy_htod(&x_d, &x_bytes)?;
            // Per-launch state: the stats buffer and both rendezvous words must start clean, or a
            // later run would inherit an earlier run's partials and counters.
            cuda.copy_htod(&stats_d, &[0u8; MHC_GRID_STATS * 4])?;
            cuda.copy_htod(&sync_d, &[0u8; 2 * 4])?;
            check_floor(args, &format!("stage 4 token {token} run {run} pre-launch"))?;
            cuda.launch(func, grid, block, &params)?;
            check_floor(
                args,
                &format!("stage 4 token {token} run {run} post-launch"),
            )?;

            let mut stats_out = vec![0u8; MHC_GRID_STATS * 4];
            let mut fp8_out = vec![0u8; HIDDEN];
            let mut scales_out = vec![0u8; HIDDEN / QUANT_BLOCK];
            let mut coeffs_out = vec![0u8; 24 * 4];
            let mut rms_out = vec![0u8; 4];
            let mut streams_out = vec![0u8; HC_MULT * HIDDEN * 2];
            cuda.copy_dtoh(&mut stats_out, &stats_d)?;
            cuda.copy_dtoh(&mut fp8_out, &fp8_d)?;
            cuda.copy_dtoh(&mut scales_out, &scales_d)?;
            cuda.copy_dtoh(&mut coeffs_out, &coeffs_d)?;
            cuda.copy_dtoh(&mut rms_out, &rms_d)?;
            cuda.copy_dtoh(&mut streams_out, &x_d)?;
            let observed = MhcRun {
                stats: stats_out,
                coeffs: coeffs_out,
                rms: rms_out,
                fp8: fp8_out,
                scales: scales_out,
                streams: streams_out,
            };
            let line: Vec<String> = observed
                .fields()
                .iter()
                .map(|(name, bytes)| format!("{name} 0x{:016x}", fingerprint(bytes)))
                .collect();
            println!("  token {token} run {run}: {}", line.join(" "));

            match &baseline {
                None => baseline = Some(observed),
                Some(reference) => {
                    for ((name, expected), (_, actual)) in
                        reference.fields().iter().zip(observed.fields())
                    {
                        require_identical_bytes(token, run, name, expected, actual)?;
                    }
                }
            }
        }
        println!(
            "  token {token}: {} runs of one input produced byte-identical hc_stats, hc_coeffs, \
             hc_rms_out, xhat_fp8, xhat_scales and X_l",
            args.repeat
        );
    }

    cuda.free(
        module,
        &[
            x_d,
            y_d,
            b_d,
            c_d,
            a_d,
            predictor_d,
            scale_d,
            base_d,
            norm_d,
            fp8_d,
            scales_d,
            stats_d,
            sync_d,
            coeffs_d,
            rms_d,
        ],
    );
    check_floor(args, "stage 4 end")?;
    println!(
        "stage 4: PASS ({} tokens x {} launches, every compared byte identical)",
        args.tokens, args.repeat
    );
    Ok(())
}
