//! The device-chain subcommand: M3-a prints the plan, M3 wiring dry-runs the dispatch, and the
//! cuda-only arms run a staged layer-0 stage or the M3-c layer-0 hybrid pass.
//!
//!   sparkflash device-chain --plan-only
//!   sparkflash device-chain --dry-run
//!   sparkflash device-chain --run-stage mhc_entry --checkpoint DIR --cubin-dir DIR [--positions 0]
//!   sparkflash device-chain --run-stage layer_zero --checkpoint DIR --cubin-dir DIR --token 0 [--element-limit N]
//!
//! --plan-only prints the CED schedule, the window accounting, the stage map, the planned host-seam
//! set and the exact design 4.3 labels. --dry-run additionally runs the full 130 + 8 plan through
//! the real ChainDevice wiring with the fake launcher and the host-seam seat delegate, validates the
//! plan-vs-ledger self-check (108,017 ChainOps calls) and prints the per-stage call counts and
//! host-seam seats. Both exit 2: a plan or a fake-launcher dry-run is not an execution. The
//! --run-stage arm exists only in a cuda build. `mhc_entry` stages the layer-0 mHC tensors and
//! launches the mHC entry through ChainDevice; `layer_zero` additionally runs the host reference
//! continuation and the c2 trace comparison, with `--token 0` as its only accepted token. Both
//! refuse every other stage, claim no token or chain, and are not full chain runs. A Bar comparison
//! is a failure, not a success. The labels are printed verbatim so a plan cannot be read as a device
//! result (docs/build/device-forward-chain.md 4.3 item 3).

use std::process::ExitCode;

use sparkflash_device::chain::{
    ChainLabels, ChainSchedule, ChainSeamPlan, LAYER_STAGE_ORDER, STEP_STAGE_ORDER,
    TAIL_STAGE_ORDER,
};
use sparkflash_device::chain_device::dry_run;
use sparkflash_device::cubins::{baseline_pins, required_pins};
use sparkflash_device::window::WindowAccounting;

/// The usage line for this subcommand.
pub const USAGE: &str = "sparkflash device-chain --plan-only|--dry-run|--run-stage mhc_entry|layer_zero --checkpoint DIR --cubin-dir DIR [--positions 0] [--layer 0] [--token 0] [--element-limit N]";

/// The exit code a plan-only or dry-run invocation returns: 2, refusing to serve.
pub const PLAN_ONLY_EXIT: u8 = 2;

/// Builds the plan-only report lines.
///
/// # Errors
/// A schedule, seam-plan or cubin-catalog error, printed by [run] before it exits nonzero.
pub fn plan_lines() -> Result<Vec<String>, String> {
    let schedule = ChainSchedule::milestone().map_err(|error| error.to_string())?;
    let seam_plan = ChainSeamPlan::for_schedule(&schedule);
    let accounting = WindowAccounting::for_schedule(&schedule);
    let baseline = baseline_pins()?;
    let pins = required_pins()?;
    let mut lines = vec![
        "sparkflash device-chain: PLAN ONLY - no driver opened, no cubin loaded, no kernel launched, no token produced".to_string(),
        "execution flags: chain_execution=false; device_execution=false; token_produced=false"
            .to_string(),
        schedule.describe(),
        format!(
            "derivation: 21 x 130 + 19 x 128 = {} prefill; {} x 40 = {} decode; {} total (plain 130 x 40 + 8 x 40 = 5,520 counts 38 decoder positions that do not run)",
            schedule.prefill_token_layer_executions(),
            schedule.decode_steps,
            schedule.decode_token_layer_executions(),
            schedule.token_layer_executions()
        ),
        format!(
            "stage map: embedding prefix, {} per-layer stages and three tail stages in design 2.2 order",
            LAYER_STAGE_ORDER.len()
        ),
    ];
    for stage in STEP_STAGE_ORDER {
        lines.push(stage.map_line());
    }
    for stage in LAYER_STAGE_ORDER {
        lines.push(stage.map_line());
    }
    for stage in TAIL_STAGE_ORDER {
        lines.push(stage.map_line());
    }
    lines.extend(seam_plan.report_lines());
    lines.extend(accounting.report_lines());
    lines.push(format!(
        "cubin catalog: {} baseline pins parsed from crates/sparkflash-kernels/baseline/kernel_baseline.txt; {} required for the device path (swiglu_mxfp8_quant_sm121 is a host seam)",
        baseline.len(),
        pins.len()
    ));
    for pin in &pins {
        lines.push(format!(
            "cubin pin {}: {} bytes, sha256 {}",
            pin.name, pin.cubin_bytes, pin.cubin_sha256
        ));
    }
    lines.push(
        "labels (design 4.3): the plan label is the state; the rest are future-only".to_string(),
    );
    lines.extend(ChainLabels::plan_lines());
    lines.push(
        "device-chain: exiting 2, refusing to serve: M3-a has no chain execution path".to_string(),
    );
    Ok(lines)
}

/// Builds the dry-run report lines.
///
/// It runs the full plan through the real [sparkflash_device::chain_device::ChainDevice] wiring
/// with the fake launcher. No driver is opened, no cubin is loaded and no kernel is launched; the
/// output says so and still exits 2 refusing to serve.
///
/// # Errors
/// A schedule, wiring or plan-vs-ledger error, printed by [run] before it exits nonzero.
pub fn dry_run_lines() -> Result<Vec<String>, String> {
    let schedule = ChainSchedule::milestone().map_err(|error| error.to_string())?;
    let dry = dry_run(&schedule).map_err(|error| error.to_string())?;
    let report = &dry.report;
    let seats = sparkflash_device::chain_device::host_seat_counts(report);
    // Derive the flags and the headline counts from the report and the launcher, not from
    // constants: a future wiring change cannot drift the printed numbers away from what ran.
    let host_calls = report.ledger.records().len();
    let device_calls = dry.device_calls.len();
    let total_calls = host_calls + device_calls;
    let mut lines = vec![
        "sparkflash device-chain: DRY RUN - no driver opened, no cubin loaded, no kernel launched, no token produced; fake launcher only".to_string(),
        format!(
            "execution flags: chain_execution={}; device_execution={}; token_produced={}",
            report.chain_execution, report.device_execution, report.token_produced
        ),
        schedule.describe(),
        format!(
            "dry-run wiring: ChainDevice<FakeLauncher, SeatHostSeams> over the full 130+8 plan; {total_calls} ChainOps calls ({host_calls} host-seam seats, {device_calls} device-launcher calls)"
        ),
        "per-stage ChainOps calls:".to_string(),
    ];
    for (stage, count) in &dry.stats.per_stage {
        lines.push(format!("  stage call {stage}: {count}"));
    }
    lines.push("host-seam seats (zero host bytes; real arithmetic is M3-c):".to_string());
    for (seam, count) in seats {
        lines.push(format!("  host-seam seat {seam}: {count} records"));
    }
    lines.push("plan-vs-ledger self-check: PASS".to_string());
    lines.push(
        "no comparison was made: M3 wiring is a dispatch/ledger dry-run, not a device run"
            .to_string(),
    );
    lines.extend(ChainLabels::plan_lines());
    lines.push(
        "device-chain: exiting 2, refusing to serve: this dry-run has no device execution path"
            .to_string(),
    );
    Ok(lines)
}

/// Runs the subcommand with the arguments that followed it.
pub fn run(args: impl Iterator<Item = String>) -> ExitCode {
    let args: Vec<String> = args.collect();
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    #[cfg(feature = "cuda")]
    if arg_refs.first().copied() == Some("--run-stage") {
        return run_stage(&arg_refs[1..]);
    }
    let lines = match arg_refs.as_slice() {
        ["--plan-only"] => plan_lines(),
        ["--dry-run"] => dry_run_lines(),
        _ => {
            eprintln!("sparkflash device-chain: usage: {USAGE}");
            eprintln!("sparkflash device-chain: --plan-only or --dry-run is required");
            return ExitCode::from(PLAN_ONLY_EXIT);
        }
    };
    match lines {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
            ExitCode::from(PLAN_ONLY_EXIT)
        }
        Err(error) => {
            eprintln!("sparkflash device-chain: {error}");
            ExitCode::FAILURE
        }
    }
}

/// The parsed --run-stage arguments. This arm is compiled only with the cuda feature.
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct StageArgs {
    checkpoint: std::path::PathBuf,
    cubin_dir: std::path::PathBuf,
    positions: Vec<usize>,
    token: u32,
    element_limit: Option<usize>,
}

#[cfg(feature = "cuda")]
fn parse_positions(raw: &str) -> Result<Vec<usize>, String> {
    let mut positions = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err("--positions has an empty entry".to_string());
        }
        let position = part
            .parse::<usize>()
            .map_err(|_| format!("--positions: {part} is not a position"))?;
        positions.push(position);
    }
    if positions.is_empty() {
        return Err("--positions must name at least one position".to_string());
    }
    Ok(positions)
}

#[cfg(feature = "cuda")]
fn stage_args(args: &[&str]) -> Result<(String, StageArgs), String> {
    let stage = args
        .first()
        .ok_or_else(|| format!("usage: {USAGE}"))?
        .to_string();
    if stage != "mhc_entry" && stage != "layer_zero" {
        let reason = match stage.as_str() {
            "moe_router" | "moe_gate_up" => {
                "NotWired (measured in isolation, not wired into the chain)"
            }
            _ => "NotStaged (M3-b stages only the layer-0 mHC entry and the M3-c layer-0 pass)",
        };
        return Err(format!(
            "stage {stage} is {reason}; --run-stage currently exposes only mhc_entry and layer_zero"
        ));
    }
    let mut checkpoint = None;
    let mut cubin_dir = None;
    let mut positions = vec![0usize];
    let mut layer = 0usize;
    let mut token = 0u32;
    let mut element_limit = None;
    let mut index = 1usize;
    while index < args.len() {
        let flag = args[index];
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--checkpoint" => checkpoint = Some(std::path::PathBuf::from(value)),
            "--cubin-dir" => cubin_dir = Some(std::path::PathBuf::from(value)),
            "--positions" => positions = parse_positions(value)?,
            "--layer" => {
                layer = value
                    .parse::<usize>()
                    .map_err(|_| format!("--layer: {value} is not a layer"))?;
            }
            "--token" => {
                token = value
                    .parse::<u32>()
                    .map_err(|_| format!("--token: {value} is not a token id"))?;
            }
            "--element-limit" => {
                element_limit =
                    Some(value.parse::<usize>().map_err(|_| {
                        format!("--element-limit: {value} is not an element count")
                    })?);
            }
            other => return Err(format!("unknown --run-stage option {other}")),
        }
        index += 2;
    }
    if layer != 0 {
        return Err(format!("--layer {layer}: M3-b/M3-c stages only layer 0"));
    }
    if stage == "layer_zero" && positions != [0] {
        return Err(format!(
            "layer_zero runs position 0 only; --positions was {positions:?}"
        ));
    }
    if stage == "layer_zero" && token != 0 {
        return Err(format!(
            "layer_zero stages embed row 0; --token must be 0, got {token}"
        ));
    }
    Ok((
        stage,
        StageArgs {
            checkpoint: checkpoint
                .ok_or_else(|| "--run-stage needs --checkpoint DIR".to_string())?,
            cubin_dir: cubin_dir.ok_or_else(|| "--run-stage needs --cubin-dir DIR".to_string())?,
            positions,
            token,
            element_limit,
        },
    ))
}

/// The element cap the layer_zero arm uses: an explicit flag, else the checkpoint cap.
///
/// The layer_zero arm always stages a checkpoint, so its no-flag default is
/// `CHECKPOINT_TENSOR_LIMIT`; the reference guard would refuse the released `wq_b` (41,943,040
/// elements) before any device work. The choice is the same function `serve` uses, so the two
/// checkpoint entry points cannot drift.
#[cfg(feature = "cuda")]
fn layer_zero_element_limit(args: &StageArgs) -> usize {
    args.element_limit
        .unwrap_or_else(|| crate::serve::default_element_limit(Some(args.checkpoint.as_path())))
}

/// Host reference `1/rms` for the block input, with the same accumulation order as `rms_norm`.
#[cfg(feature = "cuda")]
fn host_inv_rms(values: &[f32], norm_eps: f64) -> f32 {
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    1.0 / (mean + norm_eps as f32).sqrt()
}

/// The number of non-zero device-readback coefficients; all 24 zero means the launch ran with no
/// staged predictor or with an unwritten output buffer.
#[cfg(feature = "cuda")]
fn nonzero_coefficients(entry: &sparkflash_engine::forward::layer_zero::LayerZeroEntry) -> usize {
    entry
        .pre
        .iter()
        .chain(&entry.post)
        .chain(&entry.comb)
        .filter(|value| **value != 0.0)
        .count()
}

/// Refuses a hybrid entry outside the m11 b7 classes; the caller prints the per-quantity lines.
#[cfg(feature = "cuda")]
fn require_hybrid_entry(
    comparison: &sparkflash_engine::forward::layer_zero::LayerZeroComparison,
) -> Result<(), String> {
    if comparison.within_declared_bars() {
        return Ok(());
    }
    let outside = comparison
        .quantities
        .iter()
        .filter(|quantity| !quantity.within_declared_bar())
        .map(|quantity| quantity.name)
        .collect::<Vec<_>>();
    Err(format!(
        "device MhcEntry readback is outside its m11 b7 class for {outside:?}"
    ))
}

/// Runs the cuda-only single-stage arm and returns the lines to print.
#[cfg(feature = "cuda")]
fn run_stage_lines(args: &[&str]) -> Result<Vec<String>, String> {
    use sparkflash_device::budget::read_mem_available_bytes;
    use sparkflash_device::chain::{ChainOps, ChainStage, StageCall};
    use sparkflash_device::chain_device::{ChainDevice, SeatHostSeams};
    use sparkflash_device::staging::LayerZeroStaging;
    use sparkflash_engine::safetensors::SafeTensorsSource;

    let (stage, args) = stage_args(args)?;
    let source = SafeTensorsSource::open(
        &args.checkpoint,
        &args.checkpoint.join("model.safetensors.index.json"),
    )
    .map_err(|error| format!("checkpoint {}: {error}", args.checkpoint.display()))?;
    let staging = LayerZeroStaging::from_source(&source, &args.positions)
        .map_err(|error| format!("stage {stage}: {error}"))?;
    let mem_available_bytes = read_mem_available_bytes()?;
    let mut launcher =
        sparkflash_device::CudaLauncher::open_for_stage(&args.cubin_dir, "mega_mhc_sm121")?;
    let cubin_bytes = launcher.cubin_bytes();
    let request = sparkflash_device::BudgetRequest {
        mem_available_bytes,
        projected_host_bytes: staging.host_bytes() + cubin_bytes,
        projected_device_bytes: staging.device_buffer_bytes(),
    };
    let mut lines = vec![
        "sparkflash device-chain: M3-b STAGE RUN - one isolated layer-0 mHC entry launch; no chain, no token, no comparison".to_string(),
        "execution flags: stage_execution=true; chain_execution=false; token_produced=false".to_string(),
        format!(
            "stage: {stage} layer 0 positions {:?}; checkpoint {}; cubin dir {}",
            args.positions,
            args.checkpoint.display(),
            args.cubin_dir.display()
        ),
    ];
    lines.push(format!(
        "cubin: mega_mhc_sm121 {cubin_bytes} B, verified against the baseline pin before any driver or allocation"
    ));
    lines.extend(staging.provenance_lines());
    let admission = launcher
        .stage_layer_zero(&staging, request)
        .map_err(|error| format!("{stage}: {error}"))?;
    lines.extend(admission.report_lines());
    if stage == "layer_zero" {
        use sparkflash_device::layer_zero::{LayerZeroRun, run_layer_zero_device_entry};
        use sparkflash_engine::config::ModelConfig;
        use sparkflash_engine::forward::layer_zero::{
            M11_HC_COEFF_BAR, M11_INV_RMS_BAR, compare_entry, compare_traces,
            run_layer_zero_with_embedding,
        };

        let config_text = std::fs::read_to_string(args.checkpoint.join("config.json"))
            .map_err(|error| format!("checkpoint config.json: {error}"))?;
        let config = ModelConfig::from_json(&config_text)
            .map_err(|error| format!("checkpoint config.json: {error}"))?;
        let limit = layer_zero_element_limit(&args);
        let row = staging
            .embed_row(0)
            .ok_or_else(|| "no staged embed row for position 0".to_string())?;
        let embedding = row.values_f32();
        let host_trace =
            run_layer_zero_with_embedding(&config, &source, args.token, &embedding, limit, None)
                .map_err(|error| format!("host layer-0 reference trace: {error}"))?;
        let host_run = LayerZeroRun::from_trace(&config, host_trace)
            .map_err(|error| format!("host layer-0 ledger: {error}"))?;
        let (device_run, entry, evidence) =
            run_layer_zero_device_entry(&mut launcher, row, &config, &source, args.token, limit)?;
        // m11's capture comparison uses the DEVICE's own 1/rms, so the expected f32 capture is
        // built in the kernel's exact FMA order: step 1 X_0 = B X_-1 + C Y_-1 (C*Y is a plain
        // mul, then one mul_add per B term), step 2 X_hat = A X_0 (one mul_add per A term), then
        // xhat * rms_weight * inverse_rms. The scalar's own 3e-4 bar is part of compare_entry.
        let width = config.dim;
        let mut x_prev = vec![0.0f32; width * config.hc_mult];
        for copy in 0..config.hc_mult {
            x_prev[copy * config.dim..(copy + 1) * config.dim].copy_from_slice(&embedding);
        }
        let (b_prev, c_prev, a_prev, y_prev) =
            sparkflash_device::staging::LayerZeroStaging::layer_zero_seed();
        let host_update = sparkflash_engine::forward::layer_zero::kernel_mhc_update(
            &x_prev, &b_prev, &c_prev, &y_prev,
        );
        let host_xhat =
            sparkflash_engine::forward::layer_zero::kernel_mhc_xhat(&host_update, &a_prev);
        let expected_capture = sparkflash_engine::forward::layer_zero::kernel_mhc_capture(
            &host_xhat,
            &staging.attn_norm,
            evidence.inv_rms,
        );
        let expected_units = sparkflash_engine::quant::quantize_activation_fp8(&expected_capture)
            .map_err(|error| format!("host block-input quantisation: {error}"))?;
        let device_codes = launcher.read_layer_zero_codes()?;
        let device_scales = launcher.read_layer_zero_scales()?;
        let codes_exact = device_codes == expected_units.codes();
        let scales_exact = device_scales == expected_units.scales();
        let host_update_matches_trace = host_update == x_prev;
        let host_xhat_matches_trace = host_xhat == host_run.trace.attn_in;
        let host_inv_rms = host_inv_rms(&host_run.trace.attn_in, config.norm_eps);
        let entry_comparison = compare_entry(
            &host_run.trace,
            &entry,
            &expected_capture,
            host_inv_rms,
            evidence.inv_rms,
        )
        .map_err(|error| format!("device MhcEntry entry comparison: {error}"))?;
        // Diagnostic only: the host trace starts from the host-computed entry while the device
        // trace starts from the readback, so every downstream host value inherits the entry
        // deltas. Those eight quantities are not device-kernel outputs and have no m11 class.
        let observed = compare_traces(&host_run.trace, &device_run.trace)
            .map_err(|error| format!("layer-0 diagnostic comparison: {error}"))?;
        let inv_rms_delta = (evidence.inv_rms - host_inv_rms).abs();
        let nonzero = nonzero_coefficients(&entry);
        // Drop the M3-b "isolated stage, no comparison" title and flags; the c1 report below is the
        // title for this arm. The staging/provenance/admission lines stay.
        lines.drain(0..2);
        lines.extend(device_run.report_lines());
        lines.push(format!(
            "device row: X readback before launch {} B, equals the admitted four-copy BF16 row: {}; X_l readback after launch equals the FMA host update: {}; the FMA host update equals the host trace's four-copy X_0: {}; the FMA X_hat equals the host trace's attn_in: {}",
            evidence.x_bytes,
            evidence.x_matches,
            evidence.x_after_launch_matches,
            host_update_matches_trace,
            host_xhat_matches_trace,
        ));
        if let Some((index, actual, expected)) = evidence.x_after_launch_first_difference {
            lines.push(format!(
                "X_l first difference: element={index} actual=0x{actual:04x} expected=0x{expected:04x} (BF16)"
            ));
        }
        lines.push(format!(
            "block input: capture exact={} ({} values); FP8 codes exact={} ({} bytes); E8M0 scales exact={} ({} bytes)",
            expected_capture == entry.pre_attention_hidden,
            expected_capture.len(),
            codes_exact,
            device_codes.len(),
            scales_exact,
            device_scales.len(),
        ));
        if let Some((index, expected, actual)) = device_codes
            .iter()
            .zip(expected_units.codes())
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
            .map(|(index, (actual, expected))| (index, *expected, *actual))
        {
            lines.push(format!(
                "FP8 codes first difference: index={index} expected=0x{expected:02x} actual=0x{actual:02x}"
            ));
        }
        if let Some((index, expected, actual)) = device_scales
            .iter()
            .zip(expected_units.scales())
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
            .map(|(index, (actual, expected))| (index, *expected, *actual))
        {
            lines.push(format!(
                "E8M0 scales first difference: index={index} expected=0x{expected:02x} actual=0x{actual:02x}"
            ));
        }
        lines.push(
            "staged boundary seed: B identity, C zero, A [1, 0, 0, 0], Y_-1 zero; the host trace starts from the same four-copy embedding row the X seed was built from"
                .to_string(),
        );
        lines.push(format!(
            "device 1/rms(X_hat): {:.9e}; host reference 1/rms: {:.9e}; max_abs delta {:.6e} (m11 b7 bar {:.1e})",
            evidence.inv_rms, host_inv_rms, inv_rms_delta, M11_INV_RMS_BAR
        ));
        lines.push(format!(
            "readback coefficients: {nonzero} of 24 non-zero; source={:?}",
            entry.source
        ));
        lines.push(
            "kernel outputs: xhat_normed, hc_coeffs and hc_rms_out are unconditional writes; trace_ring=0 disables the ring, not these buffers (kernels/mega_mhc_sm121.cu:291-294,329)"
                .to_string(),
        );
        lines.push(
            "entry comparison: device readback against the FMA host entry, m11 b7 classes (capture exact, hc_coeffs 1e-4, 1/rms 3e-4):"
                .to_string(),
        );
        lines.extend(entry_comparison.report_lines());
        lines.push(
            "diagnostic: device-entry trace against host-entry trace; every downstream value is host arithmetic from a different entry, so these deltas are not a device-kernel class:"
                .to_string(),
        );
        for quantity in &observed.quantities {
            lines.push(quantity.report_line());
        }
        let failure = if !evidence.x_matches {
            Some(format!(
                "the X readback ({} B) is not the admitted row",
                evidence.x_bytes
            ))
        } else if !evidence.x_after_launch_matches {
            Some(
                "the kernel's X_l write-back is not the BF16 FMA host update; m11 b7 measured X_l bit-identical"
                    .to_string(),
            )
        } else if !host_update_matches_trace {
            Some("the FMA host update is not the host trace's four-copy X_0 row".to_string())
        } else if !host_xhat_matches_trace {
            Some("the FMA X_hat is not the host trace's attn_in".to_string())
        } else if !codes_exact || !scales_exact {
            Some(format!(
                "the block-input quantisation is not bit-identical: codes exact={codes_exact}, scales exact={scales_exact}"
            ))
        } else if nonzero == 0 {
            Some(
                "every readback coefficient is zero; the launch did not produce coefficients"
                    .to_string(),
            )
        } else if inv_rms_delta > M11_INV_RMS_BAR {
            Some(format!(
                "device 1/rms is outside the m11 b7 3e-4 bar (delta {inv_rms_delta:.6e})"
            ))
        } else {
            require_hybrid_entry(&entry_comparison).err()
        };
        if let Some(reason) = failure {
            lines.push(format!("layer-0 hybrid verdict: FAIL: {reason}"));
            return Err(lines.join("\n"));
        }
        lines.push(format!(
            "layer-0 hybrid verdict: PASS - the device MhcEntry readback is within the m11 b7 classes (hc_coeffs {M11_HC_COEFF_BAR:.1e}, 1/rms {M11_INV_RMS_BAR:.1e}); downstream differences are host arithmetic from a different entry, not a device claim"
        ));
        lines.push(
            "the host reference comparison remains the exact c2 host pass; this window run is one layer, one position, no token and no chain"
                .to_string(),
        );
        return Ok(lines);
    }
    let schedule = ChainSchedule::milestone().map_err(|error| error.to_string())?;
    let mut device = ChainDevice::new(&schedule, launcher, SeatHostSeams);
    for position in &args.positions {
        let row = staging
            .embed_row(*position)
            .ok_or_else(|| format!("no staged embed row for position {position}"))?;
        device
            .launcher_mut()
            .write_layer_zero_x(row)
            .map_err(|error| format!("position {position}: {error}"))?;
        device
            .stage(StageCall {
                layer: 0,
                position: *position,
                stage: ChainStage::MhcEntry,
            })
            .map_err(|error| format!("position {position}: {error}"))?;
    }
    lines.push(format!(
        "device chain wiring: ChainDevice<CudaLauncher, SeatHostSeams>; {stage} launched for {} position(s); {} device stage calls",
        args.positions.len(),
        device.stats().device_calls
    ));
    lines.push(
        "no readback, no comparison and no token: this is one isolated stage launch, not a chain run"
            .to_string(),
    );
    lines.push(
        "the mHC stage numbers on main are m11's isolated measurements, not a chain result"
            .to_string(),
    );
    Ok(lines)
}

#[cfg(feature = "cuda")]
fn run_stage(args: &[&str]) -> ExitCode {
    match run_stage_lines(args) {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sparkflash device-chain: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_device::chain::{
        FIRST_CHAIN_LABEL, HEAD_LABEL, HYBRID_TOKEN_LABEL, PLAN_ONLY_LABEL,
    };

    #[test]
    fn a_plan_prints_the_counts_and_labels_and_claims_no_execution() {
        let lines = plan_lines().expect("plan");
        let text = lines.join("\n");
        assert!(
            text.contains("5482 total"),
            "the plan quotes the derived total"
        );
        assert!(
            text.contains("21 x 130"),
            "the prefill derivation is explained"
        );
        assert!(text.contains("5,520"), "the plain product is reconciled");
        assert!(text.contains("moe_gate_up: device kernel routed_gate_up_sm121"));
        assert!(text.contains("measured in isolation (m10)"));
        assert!(text.contains("moe_router: device kernel moe_router_sm121"));
        assert!(text.contains("embedding: host seam embedding_gather"));
        assert!(text.contains("final_norm: host seam final_norm"));
        assert!(text.contains("lm_head_screen: device kernel cert_lm_head_sm121"));
        assert!(text.contains("lm_head_argmax: host seam lm_head_argmax"));
        assert!(text.contains("PLAN ONLY - no driver opened"));
        assert!(
            text.contains("chain_execution=false; device_execution=false; token_produced=false")
        );
        assert!(text.contains("9 baseline pins parsed"));
        assert!(
            text.contains("crates/sparkflash-kernels/baseline/kernel_baseline.txt"),
            "the catalog names the repository path, not a crate-relative one"
        );
        assert!(text.contains("8 required for the device path"));
        // STEP -> LAYER -> TAIL, and the tail's own order, by position in the printed stage map.
        // Presence alone cannot catch a map that prints the groups out of order.
        let map_line = |prefix: &str| {
            lines
                .iter()
                .position(|line| line.starts_with(prefix))
                .unwrap_or_else(|| panic!("missing stage-map line {prefix:?}"))
        };
        assert!(map_line("embedding:") < map_line("engram:"));
        assert!(map_line("engram:") < map_line("ffn_epilogue:"));
        assert!(map_line("ffn_epilogue:") < map_line("final_norm:"));
        assert!(map_line("final_norm:") < map_line("lm_head_screen:"));
        assert!(map_line("lm_head_screen:") < map_line("lm_head_argmax:"));
        assert!(text.contains("exiting 2, refusing to serve"));
        assert!(
            text.lines()
                .any(|line| line == format!("label: {PLAN_ONLY_LABEL}"))
        );
        for label in [HYBRID_TOKEN_LABEL, FIRST_CHAIN_LABEL, HEAD_LABEL] {
            assert!(
                text.lines()
                    .any(|line| line == format!("label reserved for a later stage: {label}")),
                "future label missing: {label}"
            );
        }
        assert!(!text.lines().any(|line| line.starts_with("label: hybrid:")));
        // No line may read as an execution: no launch count, no token, no logits comparison.
        assert!(!text.contains("launches="));
        assert!(!text.contains("tokens="));
        assert!(!text.contains("outputs_sha256"));
        assert!(!text.contains("chain execution: true"));
    }

    #[test]
    fn a_dry_run_drives_the_full_plan_and_claims_no_execution() {
        let lines = dry_run_lines().expect("dry run");
        let text = lines.join("\n");
        assert!(text.contains("DRY RUN - no driver opened"));
        assert!(
            text.contains("chain_execution=false; device_execution=false; token_produced=false")
        );
        assert!(text.contains(
            "dry-run wiring: ChainDevice<FakeLauncher, SeatHostSeams> over the full 130+8 plan; 108017 ChainOps calls (58670 host-seam seats, 49347 device-launcher calls)"
        ));
        // Pin both totals explicitly: 108,017 ChainOps calls, of which the fake launcher records
        // 49,347. A wiring change must update the derived output and this test together.
        assert!(text.contains("108017 ChainOps calls"));
        assert!(text.contains("49347 device-launcher calls"));
        assert!(text.contains("stage call mhc_entry: 5482"));
        assert!(text.contains("stage call embedding: 138"));
        assert!(text.contains("stage call final_norm: 9"));
        assert!(text.contains("host-seam seat dispatch: 5482"));
        assert!(text.contains("plan-vs-ledger self-check: PASS"));
        assert!(text.contains("no comparison was made"));
        assert!(
            text.lines()
                .any(|line| line == format!("label: {PLAN_ONLY_LABEL}"))
        );
        for label in [HYBRID_TOKEN_LABEL, FIRST_CHAIN_LABEL, HEAD_LABEL] {
            assert!(
                text.lines()
                    .any(|line| line == format!("label reserved for a later stage: {label}")),
                "future label missing: {label}"
            );
        }
        assert!(!text.lines().any(|line| line.starts_with("label: hybrid:")));
        assert!(!text.contains("execution flags: chain_execution=true"));
        assert!(!text.contains("device_execution=true"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn run_stage_refuses_every_other_stage_by_name() {
        let error = run_stage_lines(&["moe_gate_up"]).expect_err("not wired");
        assert!(error.contains("NotWired"), "{error}");
        assert!(error.contains("moe_gate_up"), "{error}");
        let error = run_stage_lines(&["attention"]).expect_err("not staged");
        assert!(error.contains("NotStaged"), "{error}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn run_stage_needs_checkpoint_and_cubin_dir_before_touching_a_driver() {
        let error = run_stage_lines(&["mhc_entry"]).expect_err("checkpoint");
        assert!(error.contains("--checkpoint"), "{error}");
        let error = run_stage_lines(&["mhc_entry", "--checkpoint", "/tmp"]).expect_err("cubin");
        assert!(error.contains("--cubin-dir"), "{error}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn run_stage_positions_are_parsed_and_validated() {
        assert_eq!(parse_positions("0,2,129").unwrap(), vec![0, 2, 129]);
        assert!(parse_positions("").is_err());
        assert!(parse_positions("0,,1").is_err());
        assert!(parse_positions("zero").is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_layer_zero_arm_parses_a_token_and_requires_position_zero() {
        let (stage, args) = stage_args(&[
            "layer_zero",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--token",
            "0",
        ])
        .expect("layer_zero arguments");
        assert_eq!(stage, "layer_zero");
        assert_eq!(args.token, 0);
        assert_eq!(args.positions, vec![0]);
        let error = stage_args(&[
            "layer_zero",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--positions",
            "1",
        ])
        .expect_err("layer_zero is position 0 only");
        assert!(error.contains("position 0 only"), "{error}");
        let error = stage_args(&[
            "layer_zero",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--token",
            "not-a-token",
        ])
        .expect_err("token id");
        assert!(error.contains("--token"), "{error}");

        let (_, explicit) = stage_args(&[
            "layer_zero",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--element-limit",
            "1234",
        ])
        .expect("an explicit element limit");
        assert_eq!(explicit.element_limit, Some(1234));

        use sparkflash_engine::linalg::CHECKPOINT_TENSOR_LIMIT;
        let default_args = StageArgs {
            checkpoint: std::path::PathBuf::from("/checkpoint"),
            cubin_dir: std::path::PathBuf::from("/cubins"),
            positions: vec![0],
            token: 0,
            element_limit: None,
        };
        assert_eq!(
            layer_zero_element_limit(&default_args),
            CHECKPOINT_TENSOR_LIMIT
        );
        let explicit_args = StageArgs {
            element_limit: Some(1234),
            ..default_args
        };
        assert_eq!(layer_zero_element_limit(&explicit_args), 1234);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_hybrid_entry_outside_its_bar_is_a_failed_run_not_success() {
        use sparkflash_engine::forward::layer_zero::{
            ComparisonClass, LayerZeroComparison, LayerZeroQuantity, QuantityDifference,
        };
        let quantity = |max_abs, bar| LayerZeroQuantity {
            name: "wq_b",
            max_abs,
            exact: false,
            max_ulp: 3,
            first_difference: Some(QuantityDifference {
                index: 0,
                expected_bits: 0x3f80_0000,
                actual_bits: 0x3f80_0001,
            }),
            bar,
        };
        let within = LayerZeroComparison {
            class: ComparisonClass::Bar,
            quantities: vec![quantity(1e-5, Some(1e-4))],
        };
        assert!(require_hybrid_entry(&within).is_ok());
        let outside = LayerZeroComparison {
            class: ComparisonClass::Bar,
            quantities: vec![quantity(1.0, Some(1e-4))],
        };
        let error = require_hybrid_entry(&outside).expect_err("outside the bar must fail the arm");
        assert!(error.contains("wq_b"), "{error}");
        assert!(error.contains("m11"), "{error}");
        let exact = LayerZeroComparison {
            class: ComparisonClass::Exact,
            quantities: vec![],
        };
        assert!(require_hybrid_entry(&exact).is_ok());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_host_reference_inv_rms_matches_the_rms_norm_order() {
        let values = [3.0f32, 4.0];
        let expected = 1.0 / (12.5f32 + 0.0).sqrt();
        assert_eq!(host_inv_rms(&values, 0.0), expected);
    }
}
