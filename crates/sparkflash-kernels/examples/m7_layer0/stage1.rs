use super::*;
use crate::stage2::*;

/// Stage 1: the fused Mega-MHC kernel against the in-crate reference, on real layer-0 bytes.
///
/// The reference returns the coefficients, the FP8 codes and the ue8m0 scales as well as the f32
/// streams, so this compares both sides of the quantiser. The residual streams are compared against
/// the bf16 rounding of the same arithmetic computed on the host from the inputs the device got,
/// because the reference's own "next" is the NEXT layer's stream update and is not a kernel output.
pub(crate) fn stage1(args: &Args) -> Result<(), String> {
    let kernel = "mega_mhc_fused_bf16_sm121";
    let prefix = format!("layers.{}", args.layer);
    println!("=== stage 1: {kernel} against mega_mhc_ref::mega_mhc_layer_with_scale ===");
    println!(
        "comparison class: internal consistency of the fused Mega-mHC kernel against the in-crate \
         CPU reference, on real layer-0 checkpoint bytes; the predictor is the released flattened \
         [24, n*d] projection on BOTH sides (B0 closed the fold), NOT model fidelity"
    );
    println!(
        "compared at every boundary, exact matches and max abs delta: coefficients B[16] A[4] C[4] \
         (the kernel's hc_coeffs), the residual streams X_l (device bf16 write against the bf16 \
         rounding of the same arithmetic), the FP8 block input (5120 codes, exact bytes), its ue8m0 \
         scales (160 bytes), and 1/rms(X_hat)"
    );
    println!(
        "layer-0 boundary seed, since the checkpoint stores no layer -1 coefficients: B = identity, \
         C = 0, A = [1, 0, 0, 0], Y_-1 = 0, X_-1 = four copies of one embed.weight row"
    );
    println!(
        "predictor: hc_attn_fn is stored as [24, 20480] over the flattened streams, and BOTH sides \
         consume the released [24, n*d] projection over all four copies (B0 removed the old d-wide \
         fold from both). The kernel's B(0..16)/A(16..20)/C(20..24) order is the released \
         pre/post/comb rows mapped with the comb terms transposed; the group scale is NOT \
         pre-folded, the kernel takes hc_attn_scale itself. The RETIRED fold over the stream MEAN is \
         measured only by stage 0's non-vacuity half and is not what this stage compares"
    );
    println!(
        "released tensors on the device path: hc_attn_scale [3] and hc_attn_base [24] both reach the \
         kernel, which applies the projection rstd, mix*scale+base, the sigmoid pre/post and the \
         released split-Sinkhorn exactly as the reference does; no coefficient tensor is excluded on \
         either side"
    );
    println!(
        "pass bar: FP8 codes at least 99.5% exact and every code within 1 (a value on an E4M3 tie is one code), ue8m0 scales bit-exact, X_l bit-exact after bf16 rounding, B and A and C (raw, no harness-side normalisation) within 1e-4, 1/rms within 3e-4 absolute"
    );
    check_floor(args, "stage 1 start")?;

    let mut snap = Snapshot::open(&args.checkpoint)?;
    println!("checkpoint: {}", args.checkpoint.display());

    // The released mHC tensors in the form the kernel and the reference consume; nothing is
    // pre-folded into the predictor (the scale must reach the kernel as `hc_attn_scale`).
    let mhc = read_released_mhc_tensors(&mut snap, &prefix, "attn")?;
    let norm = snap.read_f32_or_bf16(&format!("{prefix}.attn_norm.weight"), &[HIDDEN])?;
    let predictor = mhc.predictor;
    let hc_base = mhc.hc_base;
    let hc_base_kernel = mhc.hc_base_kernel;
    let hc_scale = mhc.hc_scale;
    let base_min = mhc.hc_base.iter().copied().fold(f32::INFINITY, f32::min);
    let base_max = mhc
        .hc_base
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    println!(
        "hc_attn_fn: {} f32 stored; hc_attn_base: {} f32 in [{base_min:.4}, {base_max:.4}] \
         (consumed: checkpoint row order for the reference, kernel B/A/C row order for the device);          hc_attn_scale: {hc_scale:?} (consumed)",
        PREDICTOR_ROWS * HC_MULT * HIDDEN,
        mhc.hc_base.len(),
    );
    println!("attn_norm.weight: {} f32 after widening to f32", norm.len());
    println!(
        "predictor: {} f32 = released [24, 4*d] rows permuted into the kernel's B/A/C order, NO \
         averaging ({} KiB); hc_attn_fn released before any device allocation",
        predictor.len(),
        predictor.len() * 4 / 1024,
    );

    // The layer-0 boundary seed.
    let mut b_prev = [0.0f32; HC_MULT * HC_MULT];
    for i in 0..HC_MULT {
        b_prev[i * HC_MULT + i] = 1.0;
    }
    let c_prev = [0.0f32; HC_MULT];
    let a_prev = [1.0f32, 0.0, 0.0, 0.0];
    let y_prev = vec![0.0f32; HIDDEN];

    // Device buffers, admitted against the memory floor before anything is allocated.
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
    admit_memory(args, planned, "stage 1 device buffers")?;
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

    // y_prev is a bf16 tensor in the kernel, so it is uploaded in bf16: an f32 upload is twice
    // the allocation and the driver refuses it with CUDA_ERROR_INVALID_VALUE.
    cuda.copy_htod(&y_d, &bf16_bits_to_bytes(&bf16_bits_of_f32(&y_prev)))?;
    cuda.copy_htod(&b_d, &f32_to_bytes(&b_prev))?;
    cuda.copy_htod(&c_d, &f32_to_bytes(&c_prev))?;
    cuda.copy_htod(&a_d, &f32_to_bytes(&a_prev))?;
    cuda.copy_htod(&predictor_d, &f32_to_bytes(&predictor))?;
    cuda.copy_htod(&scale_d, &f32_to_bytes(&hc_scale))?;
    cuda.copy_htod(&base_d, &f32_to_bytes(&hc_base_kernel))?;
    cuda.copy_htod(&norm_d, &f32_to_bytes(&norm))?;
    cuda.copy_htod(&stats_d, &[0u8; MHC_GRID_STATS * 4])?;
    cuda.copy_htod(&sync_d, &[0u8; 2 * 4])?;

    let grid = (MHC_TILES, 1u32, 1u32);
    let block = (MHC_THREADS, 1u32, 1u32);
    // The kernel takes 19 parameters: 16 pointers (including the B0 f32 block-input capture), the
    // trace ring, the layer id and the step sequence.
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

    let mut verdicts: Vec<(String, bool)> = Vec::new();
    for token in 0..args.tokens {
        let row = snap.read_row_bf16("embed.weight", token, HIDDEN)?;
        let mut x = Vec::with_capacity(HC_MULT * HIDDEN);
        for _ in 0..HC_MULT {
            x.extend_from_slice(&row);
        }
        let x_bits = bf16_bits_of_f32(&x);

        // The reference on the same bytes.
        let mut streams = Streams::new(x.clone());
        let mut ledger = TrafficLedger::default();
        let (_next, coefficients, codes, scales) = mega_mhc_layer_with_scale(
            &mut streams,
            &norm,
            &a_prev,
            &predictor,
            &hc_base,
            &hc_scale,
            &mut ledger,
        );

        // The in-place X_l update the kernel writes, computed on the host from the same inputs.
        let mut expected_x = vec![0.0f32; HC_MULT * HIDDEN];
        for stream in 0..HC_MULT {
            for column in 0..HIDDEN {
                let mut value = 0.0f32;
                for source in 0..HC_MULT {
                    value += b_prev[stream * HC_MULT + source] * x[source * HIDDEN + column];
                }
                value += c_prev[stream] * y_prev[column];
                expected_x[stream * HIDDEN + column] = value;
            }
        }
        // 1/rms(X_hat), the released formula, from the same A the kernel was given: the RMSNorm of
        // the COLLAPSED block input uses X_hat's sum of squares over d and norm_eps = 1e-20, not
        // the four streams' sum over n*d and not the Sinkhorn epsilon.
        let mut expected_xhat = vec![0.0f32; HIDDEN];
        let mut xhat_squares = 0.0f32;
        for column in 0..HIDDEN {
            let mut value = 0.0f32;
            for (stream, weight) in a_prev.iter().enumerate() {
                value += weight * x[stream * HIDDEN + column];
            }
            expected_xhat[column] = value;
            xhat_squares += value * value;
        }
        let expected_inv_rms = 1.0f32 / (xhat_squares / HIDDEN as f32 + NORM_EPS).sqrt();

        cuda.copy_htod(&x_d, &bf16_bits_to_bytes(&x_bits))?;
        // The grid rendezvous counters are per-launch state: without re-zeroing them the second
        // token starts with the first token's tile count and never publishes its coefficients.
        cuda.copy_htod(&stats_d, &[0u8; MHC_GRID_STATS * 4])?;
        cuda.copy_htod(&sync_d, &[0u8; 2 * 4])?;
        check_floor(args, &format!("stage 1 token {token} pre-launch"))?;
        cuda.launch(func, grid, block, &params)?;
        check_floor(args, &format!("stage 1 token {token} post-launch"))?;

        let mut x_out = vec![0u8; HC_MULT * HIDDEN * 2];
        let mut fp8_out = vec![0u8; HIDDEN];
        let mut scales_out = vec![0u8; HIDDEN / QUANT_BLOCK];
        let mut coeffs_out = vec![0u8; 24 * 4];
        let mut rms_out = vec![0u8; 4];
        let mut normed_out = vec![0u8; HIDDEN * 4];
        cuda.copy_dtoh(&mut x_out, &x_d)?;
        cuda.copy_dtoh(&mut fp8_out, &fp8_d)?;
        cuda.copy_dtoh(&mut scales_out, &scales_d)?;
        cuda.copy_dtoh(&mut coeffs_out, &coeffs_d)?;
        cuda.copy_dtoh(&mut rms_out, &rms_d)?;
        cuda.copy_dtoh(&mut normed_out, &normed_d)?;
        let x_out_bits: Vec<u16> = x_out
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        let coeffs_out_f32 = bytes_to_f32(&coeffs_out);
        let rms_out_f32 = bytes_to_f32(&rms_out);

        println!("--- token {token} (embed row {token}, four copies) ---");
        let mut b_ref = Vec::with_capacity(HC_MULT * HC_MULT);
        for i in 0..HC_MULT {
            for j in 0..HC_MULT {
                b_ref.push(coefficients.b[i][j]);
            }
        }
        let s_b = compare_f32(
            "hc_coeffs B (the sinkhorn mixer)",
            &coeffs_out_f32[0..16],
            &b_ref,
        )?;
        let s_a = compare_f32(
            "hc_coeffs A (the input mixes)",
            &coeffs_out_f32[16..20],
            &coefficients.a,
        )?;
        let s_c = compare_f32(
            "hc_coeffs C (the block-output weights)",
            &coeffs_out_f32[20..24],
            &coefficients.c,
        )?;
        let s_fp8 = compare_u8("xhat_fp8 (E4M3 codes)", &fp8_out, codes.as_slice())?;
        let s_scales = compare_u8("xhat_scales (ue8m0)", &scales_out, scales.as_slice())?;
        let s_x = compare_bf16_write("residual streams X_l", &x_out_bits, &expected_x)?;
        let s_rms = compare_f32("1/rms(X_hat)", &rms_out_f32, &[expected_inv_rms])?;
        // The B0 capture: the kernel publishes the f32 block input it quantised, and the expected
        // value is xhat * rms_weight * inverse_rms in the kernel's multiply order, both derived from
        // the SAME inputs and the device's own inverse rms, so the two are expected bit-exact. This
        // is a check of the capture itself, independent of codes-times-scale reconstruction.
        let normed_out_f32 = bytes_to_f32(&normed_out);
        let expected_normed = block_input_normed(&expected_xhat, &norm, rms_out_f32[0]);
        let s_capture = compare_f32(
            "block input f32 capture (xhat_normed)",
            &normed_out_f32,
            &expected_normed,
        )?;
        let capture_ok = s_capture.all_exact();
        let coefficients_ok =
            s_b.max_delta <= 1e-4 && s_a.max_delta <= 1e-4 && s_c.max_delta <= 1e-4;
        // Both sums of squares are block-parallel reductions whose four partials are combined in
        // tile order and rounded once, while the reference accumulates sequentially; 3e-4 absolute.
        let rms_ok = s_rms.max_delta <= 3e-4;
        let fp8_ok = s_fp8.exact_pct() >= 99.5 && s_fp8.max_delta <= 1.0;
        if s_fp8.exact != s_fp8.len {
            // The diagnosis is conditional on what the bytes show (D8 review M3): a quantiser-rule
            // difference moves codes by more than one AND moves block scales, while the fixed-order
            // vs sequential 1/rms residual is one code with bit-exact scales.
            if s_scales.all_exact() && s_fp8.max_delta <= 1.0 {
                println!(
                    "  finding: {} of {} FP8 codes differ (first at index {}), each by one code with \
                     the scales bit-exact; the cause is upstream, not in the quantiser: the two sums \
                     of squares differ between the device's fixed-order block reduction and the \
                     reference's sequential accumulation, so 1/rms differs by {:e} and moves a value \
                     across an E4M3 boundary",
                    s_fp8.len - s_fp8.exact,
                    s_fp8.len,
                    s_fp8.at,
                    s_rms.max_delta,
                );
            } else {
                println!(
                    "  finding: {} of {} FP8 codes differ (first at index {}), max code delta {} and \
                     {} of {} ue8m0 scales differing: this is a QUANTISER-RULE difference, not the \
                     reduction-order residual (that one is a one-code difference with bit-exact \
                     scales). Check which scale rule the installed cubin was built with before \
                     reading the token verdict.",
                    s_fp8.len - s_fp8.exact,
                    s_fp8.len,
                    s_fp8.at,
                    s_fp8.max_delta,
                    s_scales.len - s_scales.exact,
                    s_scales.len,
                );
            }
        }
        verdicts.push((
            format!("token {token}"),
            fp8_ok
                && s_scales.all_exact()
                && s_x.all_exact()
                && coefficients_ok
                && rms_ok
                && capture_ok,
        ));
        println!(
            "  token {token} boundary verdict: fp8 {} scales {} X_l {} coefficients {} inv_rms {} \
             f32 block input {}",
            if fp8_ok { "within bar" } else { "DIFFER" },
            if s_scales.all_exact() {
                "bit-exact"
            } else {
                "DIFFER"
            },
            if s_x.all_exact() {
                "bit-exact"
            } else {
                "DIFFER"
            },
            if coefficients_ok {
                "within 1e-4"
            } else {
                "OVER 1e-4"
            },
            if rms_ok { "within 3e-4" } else { "OVER 3e-4" },
            if capture_ok {
                "bit-exact"
            } else {
                "DIFFER (bar: bit-exact)"
            },
        );
        println!(
            "  reference ledger: x_read {} x_write {} xhat_write {} scale_write {} (passes {})",
            ledger.x_read,
            ledger.x_write,
            ledger.xhat_write,
            ledger.scale_write,
            streams.passes(),
        );
    }

    if args.repeat > 1 {
        // Timing only: the kernel updates the residual streams in place, so the values after a
        // repeated launch are a different quantity from the per-token comparison above and are not
        // compared. The buffers hold token 0's inputs from the last loop iteration.
        check_floor(args, "stage 1 timing pre-launch")?;
        let (mut per_launch, batched_ms) =
            cuda.launch_repeated(func, grid, block, &params, args.repeat)?;
        check_floor(args, "stage 1 timing post-launch")?;
        per_launch.sort_by(|a, b| a.partial_cmp(b).expect("event times are finite"));
        let count = per_launch.len();
        let mean = per_launch.iter().sum::<f64>() / count as f64;
        println!(
            "timing: {count} launches on the same buffers (the values after the repeats are not compared)"
        );
        println!(
            "per-launch ms: min {:.4} p50 {:.4} mean {mean:.4} max {:.4}",
            per_launch[0],
            per_launch[count / 2],
            per_launch[count - 1]
        );
        println!("batched ms/launch over {count} launches: {batched_ms:.4}");
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
            normed_d,
        ],
    );
    check_floor(args, "stage 1 end")?;
    let failed: Vec<&String> = verdicts
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(name, _)| name)
        .collect();
    if failed.is_empty() {
        println!(
            "stage 1: PASS ({} of {} tokens at every boundary)",
            verdicts.len(),
            verdicts.len()
        );
        Ok(())
    } else {
        Err(format!("stage 1: FAIL at {:?}", failed))
    }
}
