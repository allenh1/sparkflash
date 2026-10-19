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
pub const USAGE: &str = "sparkflash device-chain --plan-only|--dry-run|--run-stage mhc_entry|layer_zero|hybrid_token --checkpoint DIR --cubin-dir DIR [--positions 0] [--layer 0] [--token 0] [--element-limit N]";

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
    if stage != "mhc_entry" && stage != "layer_zero" && stage != "hybrid_token" {
        let reason = match stage.as_str() {
            "moe_router" | "moe_gate_up" => {
                "NotWired (measured in isolation, not wired into the chain)"
            }
            _ => {
                "NotStaged (M3-c stages only the layer-0 mHC entry, the layer-0 hybrid pass and the 40-layer hybrid token)"
            }
        };
        return Err(format!(
            "stage {stage} is {reason}; --run-stage currently exposes only mhc_entry, layer_zero and hybrid_token"
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
    if stage == "hybrid_token" && positions != [0] {
        return Err(format!(
            "hybrid_token runs position 0 only; --positions was {positions:?}"
        ));
    }
    if stage == "hybrid_token" && token != 0 {
        return Err(format!(
            "hybrid_token compares against the reference token 0 pass; --token must be 0, got {token}"
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

/// The c3 hybrid-token arm. It runs the in-process reference prefill first, then the 40-layer
/// chain with a device MhcEntry per layer and host stand-ins everywhere else. The only PASS
/// condition is token-id equality; the top-k is reported, not gated.
#[cfg(feature = "cuda")]
fn run_hybrid_lines(args: &StageArgs) -> Result<Vec<String>, String> {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    use std::time::Instant;

    use sparkflash_device::budget::read_mem_available_bytes;
    use sparkflash_device::chain::ChainStage;
    use sparkflash_device::chain_device::{CudaLauncher, DeviceLauncher, DeviceStageCall};
    use sparkflash_device::staging::{LayerStaging, LayerZeroStaging, StagingError};
    use sparkflash_engine::forward::layer_zero::{
        M11_HC_COEFF_BAR, M11_INV_RMS_BAR, bf16_bytes, kernel_mhc_capture, kernel_mhc_update,
        kernel_mhc_xhat,
    };
    use sparkflash_engine::forward::mhc::{self, Mixes};
    use sparkflash_engine::forward::{HybridEntry, InferenceEngine, SequenceState};
    use sparkflash_engine::linalg;
    use sparkflash_engine::quant::quantize_activation_fp8;
    use sparkflash_engine::safetensors::SafeTensorsSource;

    let config_text = std::fs::read_to_string(args.checkpoint.join("config.json"))
        .map_err(|error| format!("checkpoint config.json: {error}"))?;
    let config = sparkflash_engine::config::ModelConfig::from_json(&config_text)
        .map_err(|error| format!("checkpoint config.json: {error}"))?;
    let limit = layer_zero_element_limit(args);
    let dim = config.dim;
    let started = Instant::now();
    let mut lines = vec![
        "sparkflash device-chain: M3-c HYBRID TOKEN - 40 layers, one prompt token, one position".to_string(),
        "device entry path: step-1 identity (B=I, C=0, Y=0, A=previous FFN pre_mix); the kernel's previous-layer B/C/Y mixer is NOT exercised because mhc_post and ffn_epilogue are host in this stage map".to_string(),
        "residual staging seam: each layer stages the f32 residual to BF16 and the host continues from the device-readback X; the chain is therefore not bit-identical to the reference even with exact host arithmetic".to_string(),
        "claim: CHAIN-WIRING parity, not a device-token result; criterion 1 remains OPEN".to_string(),
        "caching seam: no 40-layer staging metadata cache; LayerStaging is rebuilt and dropped per layer. The one reused device allocation is the 16-buffer mHC set (2,064,352 B), restaged per layer.".to_string(),
        "c3b follow-up: defer each layer's ffn_epilogue into the next device entry so the real previous-layer B/C/Y mixer is exercised and host stand-ins drop 8 -> 7; c3 is not a full device mHC path.".to_string(),
        "per-layer device stages: 1 MhcEntry. Per-layer host stand-ins (device-status but host-executed here): attention_projections, attention, attention_output_projection, mhc_post, moe_router, moe_gate_up, moe_down_combine, shared_expert_gate. Tail stand-in: lm_head_screen. Tail host seams: final_norm, lm_head_argmax.".to_string(),
        format!(
            "checkpoint {}; cubin dir {}; token {}; position 0; element limit {}",
            args.checkpoint.display(),
            args.cubin_dir.display(),
            args.token,
            limit
        ),
    ];

    // The reference half is measured first, in this same process, on the same checkpoint.
    let reference_source = SafeTensorsSource::open(
        &args.checkpoint,
        &args.checkpoint.join("model.safetensors.index.json"),
    )
    .map_err(|error| {
        format!(
            "reference checkpoint {}: {error}",
            args.checkpoint.display()
        )
    })?;
    let reference_engine = InferenceEngine::new(config.clone(), Box::new(reference_source), limit)
        .map_err(|error| format!("reference engine: {error}"))?;
    let mut reference_state = SequenceState::new(&config);
    let observed = reference_engine
        .prefill_observed(&mut reference_state, &[args.token])
        .map_err(|error| format!("reference prefill: {error}"))?;
    let reference_elapsed = started.elapsed();
    let reference_output = observed.output;
    let reference_boundary = observed.boundary;
    let reference_token = reference_output.next_token;
    let reference_top5 = top5(&reference_output.logits);
    lines.push(format!(
        "reference engine: token_id={}; wall_ms={}; top5={}",
        token_text(reference_token),
        reference_elapsed.as_millis(),
        format_top5(&reference_top5)
    ));
    drop(reference_engine);
    drop(reference_state);

    // The embedding row is paged, not the whole 661M-element table.
    let embed_source = SafeTensorsSource::open(
        &args.checkpoint,
        &args.checkpoint.join("model.safetensors.index.json"),
    )
    .map_err(|error| format!("embed checkpoint {}: {error}", args.checkpoint.display()))?;
    let embed_staging = LayerZeroStaging::from_source(&embed_source, &[0])
        .map_err(|error| format!("embed row stage: {error}"))?;
    let embedding = embed_staging
        .embed_row(0)
        .ok_or_else(|| "no staged embed row for position 0".to_string())?
        .values_f32();
    drop(embed_source);

    let mem_available_bytes = read_mem_available_bytes()?;
    let mut launcher = CudaLauncher::open_for_stage(&args.cubin_dir, "mega_mhc_sm121")?;
    let cubin_bytes = launcher.cubin_bytes();
    let provider_source = SafeTensorsSource::open(
        &args.checkpoint,
        &args.checkpoint.join("model.safetensors.index.json"),
    )
    .map_err(|error| format!("staging checkpoint {}: {error}", args.checkpoint.display()))?;

    struct DeviceTiming {
        staging_ms: u128,
        device_ms: u128,
        capture_exact: bool,
        codes_exact: bool,
        scales_exact: bool,
        coeff_max_abs: f32,
        inv_rms_delta: f32,
        x_bytes: usize,
        matrix_bytes: u64,
    }
    let mut device_timings: Vec<DeviceTiming> = Vec::with_capacity(config.n_layers);
    let mut staged_first = false;
    let device_total_ms = Rc::new(Cell::new(0u64));
    let completed = Rc::new(Cell::new(0usize));
    let progress_lines: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let mut provider = |layer: usize,
                        hidden: &[f32],
                        pre_mix: &[f32]|
     -> Result<HybridEntry, sparkflash_engine::linalg::EngineError> {
        let done = completed.get();
        if done > 0 {
            let chain_elapsed = started.elapsed().as_secs_f64() - reference_elapsed.as_secs_f64();
            let projected = reference_elapsed.as_secs_f64()
                + chain_elapsed * (config.n_layers as f64 / done as f64);
            if projected > 600.0 {
                return Err(sparkflash_engine::linalg::EngineError::Shape {
                    what: "c3 runtime guard".to_string(),
                    detail: format!(
                        "projected total {:.1} s exceeds the 600 s self-limit after {} of {} layers; aborting cleanly before layer {}",
                        projected, done, config.n_layers, layer
                    ),
                });
            }
        }
        let staging_started = Instant::now();
        let staging = LayerStaging::from_source(&provider_source, layer, &[]).map_err(
            |error: StagingError| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} staging"),
                detail: error.to_string(),
            },
        )?;
        let matrix_bytes = staging.host_bytes();
        if !staged_first {
            let request = sparkflash_device::BudgetRequest {
                mem_available_bytes,
                projected_host_bytes: staging.host_bytes() + cubin_bytes,
                projected_device_bytes: staging.device_buffer_bytes(),
            };
            launcher.stage_layer(&staging, request).map_err(|error| {
                sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} staging admission"),
                    detail: error.to_string(),
                }
            })?;
            staged_first = true;
        } else {
            launcher.restage_layer(&staging).map_err(|error| {
                sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} restage"),
                    detail: error,
                }
            })?;
        }
        let staging_ms = staging_started.elapsed().as_millis();
        launcher.write_layer_a(pre_mix).map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} A write"),
                detail: error,
            }
        })?;
        let x_bytes = bf16_bytes(hidden);
        let device_started = Instant::now();
        launcher.write_layer_x_bf16(&x_bytes).map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} X write"),
                detail: error,
            }
        })?;
        launcher
            .launch(DeviceStageCall {
                stage: ChainStage::MhcEntry,
                layer,
                position: 0,
                kernel: "mega_mhc_sm121",
            })
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} MhcEntry launch"),
                detail: error.to_string(),
            })?;
        let entry = launcher.read_layer_entry().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} entry readback"),
                detail: error,
            }
        })?;
        let x_after = launcher.read_layer_x().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} X readback"),
                detail: error,
            }
        })?;
        let inv_rms = launcher.read_layer_inv_rms().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} 1/rms readback"),
                detail: error,
            }
        })?;
        let codes = launcher.read_layer_codes().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} FP8 codes readback"),
                detail: error,
            }
        })?;
        let scales = launcher.read_layer_scales().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} FP8 scales readback"),
                detail: error,
            }
        })?;
        let device_ms = device_started.elapsed().as_millis();
        device_total_ms.set(device_total_ms.get() + device_ms as u64);
        if x_after != x_bytes {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} identity X write"),
                detail:
                    "the X_l readback is not the BF16 stream that was written before the launch"
                        .to_string(),
            });
        }
        let x_readback = decode_bf16(&x_after);
        let (b_prev, c_prev, _a_seed, y_prev) = LayerZeroStaging::layer_zero_seed();
        let host_update = kernel_mhc_update(&x_readback, &b_prev, &c_prev, &y_prev);
        let host_xhat = kernel_mhc_xhat(&host_update, pre_mix);
        let expected_capture = kernel_mhc_capture(&host_xhat, &staging.attn_norm, inv_rms);
        let units = quantize_activation_fp8(&expected_capture).map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} host block-input quantisation"),
                detail: error.to_string(),
            }
        })?;
        let capture_exact = entry.pre_attention_hidden == expected_capture;
        let codes_exact = codes == units.codes();
        let scales_exact = scales == units.scales();
        let fn_name = format!("layers.{layer}.hc_attn_fn");
        let fn_w = linalg::decode_by_name(&provider_source, &fn_name, limit).map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} host hc_attn_fn"),
                detail: error.to_string(),
            }
        })?;
        let mut host_mixes = Mixes::for_config(&config);
        mhc::hc_mixes(
            &config,
            &x_readback,
            &fn_w,
            &staging.hc_scale,
            &staging.hc_base,
            &mut host_mixes,
        )
        .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
            what: format!("layer {layer} host hc_mixes"),
            detail: error.to_string(),
        })?;
        let mut host_attn_in = vec![0.0f32; dim];
        mhc::hc_pre(&config, pre_mix, &x_readback, &mut host_attn_in).map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} host hc_pre"),
                detail: error.to_string(),
            }
        })?;
        let host_inv = host_inv_rms(&host_attn_in, config.norm_eps);
        let coeff_max_abs = entry
            .pre
            .iter()
            .zip(&host_mixes.pre)
            .chain(entry.post.iter().zip(&host_mixes.post))
            .chain(entry.comb.iter().zip(&host_mixes.comb))
            .map(|(device, host)| (device - host).abs())
            .fold(0.0f32, f32::max);
        let inv_rms_delta = (inv_rms - host_inv).abs();
        if !capture_exact || !codes_exact || !scales_exact {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} block-input comparison"),
                detail: format!(
                    "capture exact={capture_exact}, codes exact={codes_exact}, scales exact={scales_exact}"
                ),
            });
        }
        if coeff_max_abs > M11_HC_COEFF_BAR {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} hc_coeffs bar"),
                detail: format!("max abs {coeff_max_abs:.6e} exceeds the 1e-4 bar"),
            });
        }
        if inv_rms_delta > M11_INV_RMS_BAR {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} 1/rms bar"),
                detail: format!("max abs delta {inv_rms_delta:.6e} exceeds the 3e-4 bar"),
            });
        }
        device_timings.push(DeviceTiming {
            staging_ms,
            device_ms,
            capture_exact,
            codes_exact,
            scales_exact,
            coeff_max_abs,
            inv_rms_delta,
            x_bytes: x_after.len(),
            matrix_bytes,
        });
        completed.set(done + 1);
        Ok(HybridEntry {
            entry,
            x: x_readback,
        })
    };
    let mut progress = |layer: usize, host_ms: u64| {
        if layer % 8 == 7 {
            let chain_ms =
                started.elapsed().as_millis() as u64 - reference_elapsed.as_millis() as u64;
            let line = format!(
                "progress: layer {}/{} complete; chain_elapsed_ms={}; device_ms={}; host_ms_last={}",
                layer + 1,
                config.n_layers,
                chain_ms,
                device_total_ms.get(),
                host_ms
            );
            println!("{line}");
            progress_lines.borrow_mut().push(line);
        }
    };
    let chain_source = SafeTensorsSource::open(
        &args.checkpoint,
        &args.checkpoint.join("model.safetensors.index.json"),
    )
    .map_err(|error| format!("chain checkpoint {}: {error}", args.checkpoint.display()))?;
    let chain_engine = InferenceEngine::new(config.clone(), Box::new(chain_source), limit)
        .map_err(|error| format!("chain engine: {error}"))?;
    let chain_result =
        chain_engine.run_hybrid_chain(args.token, &embedding, &mut provider, &mut progress);
    let output = match chain_result {
        Ok(output) => output,
        Err(error) => {
            lines.extend(progress_lines.borrow().iter().cloned());
            for (layer, timing) in device_timings.iter().enumerate() {
                lines.push(format!(
                    "partial layer {layer} timing: staging_ms={} device_ms={} capture_exact={} codes_exact={} scales_exact={} coeff_max_abs={:.6e} inv_rms_delta={:.6e}",
                    timing.staging_ms,
                    timing.device_ms,
                    timing.capture_exact,
                    timing.codes_exact,
                    timing.scales_exact,
                    timing.coeff_max_abs,
                    timing.inv_rms_delta
                ));
            }
            lines.push(format!("c3 chain abort: {error}"));
            return Err(lines.join("\n"));
        }
    };
    let total_elapsed = started.elapsed();
    let chain_elapsed = total_elapsed.saturating_sub(reference_elapsed);
    lines.extend(progress_lines.borrow().iter().cloned());
    let total_device_ms: u128 = device_timings.iter().map(|timing| timing.device_ms).sum();
    let total_staging_ms: u128 = device_timings.iter().map(|timing| timing.staging_ms).sum();
    let total_host_ms: u64 = output.layers.iter().map(|trace| trace.host_ms).sum();
    let total_host_bytes: u64 = output
        .layers
        .iter()
        .flat_map(|trace| trace.stages.iter())
        .map(|event| event.host_bytes_in.saturating_add(event.host_bytes_out))
        .sum();
    for (device_index, trace) in output.layers.iter().enumerate() {
        let timing = device_timings
            .get(device_index)
            .ok_or_else(|| format!("layer {} has no device timing record", trace.layer))?;
        let host_bytes: u64 = trace
            .stages
            .iter()
            .map(|event| event.host_bytes_in.saturating_add(event.host_bytes_out))
            .sum();
        let ledger = trace
            .stages
            .iter()
            .map(|event| {
                let seam = match event.stage {
                    "mhc_entry" => "[device identity seam]",
                    "attn_norm" => "[device capture]",
                    _ => "",
                };
                format!(
                    "{}:{}/{} {}",
                    event.stage, event.host_bytes_in, event.host_bytes_out, seam
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        lines.push(format!("ledger layer {:02}: {ledger}", trace.layer));
        lines.push(format!(
            "layer {:02}: staging_ms={} device_ms={} host_ms={} host_bytes={} staged_matrix_bytes={} x_bytes={} capture_exact={} codes_exact={} scales_exact={} coeff_max_abs={:.6e} inv_rms_delta={:.6e}",
            trace.layer,
            timing.staging_ms,
            timing.device_ms,
            trace.host_ms,
            host_bytes,
            timing.matrix_bytes,
            timing.x_bytes,
            timing.capture_exact,
            timing.codes_exact,
            timing.scales_exact,
            timing.coeff_max_abs,
            timing.inv_rms_delta
        ));
    }
    lines.push(format!(
        "chain totals: layers={} device_launches={} device_ms={} staging_ms={} host_ms={} total_chain_ms={} host_bytes={} cubin_bytes={} device_buffer_bytes={}",
        output.layers.len(),
        device_timings.len(),
        total_device_ms,
        total_staging_ms,
        total_host_ms,
        chain_elapsed.as_millis(),
        total_host_bytes,
        cubin_bytes,
        sparkflash_device::MhcBufferPlan::LAYER_ZERO.device_bytes() as u64
    ));
    for stage in &output.tail_stages {
        lines.push(format!(
            "tail stage {}: host_bytes_in={} host_bytes_out={} note={}",
            stage.stage, stage.host_bytes_in, stage.host_bytes_out, stage.note
        ));
    }
    let hybrid_top5 = top5(&output.logits);
    lines.push(format!(
        "hybrid chain: token_id={}; wall_ms={}; top5={}",
        token_text(output.token),
        chain_elapsed.as_millis(),
        format_top5(&hybrid_top5)
    ));
    let token_match = reference_token == output.token;
    lines.push(format!(
        "token comparison: reference_id={} hybrid_id={} exact_equal={} top5_overlap={}",
        token_text(reference_token),
        token_text(output.token),
        token_match,
        top5_overlap(&reference_top5, &hybrid_top5)
    ));
    if !token_match {
        lines.extend(first_layer_difference(&reference_boundary, &output));
        lines.push("CHAIN-WIRING token parity: FAIL - the hybrid token id is not the in-process reference token id".to_string());
        return Err(lines.join("\n"));
    }
    lines.push("CHAIN-WIRING token parity: PASS - the hybrid chain produced the same token id as the in-process reference engine; this is chain-wiring parity, not a device-token claim, and criterion 1 remains OPEN".to_string());
    Ok(lines)
}

/// Decodes little-endian BF16 bytes to f32, the format the kernel's X buffer holds.
#[cfg(feature = "cuda")]
fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
        .collect()
}

/// The five largest logits, descending by value and then ascending by index.
#[cfg(feature = "cuda")]
fn token_text(token: Option<u32>) -> String {
    token.map_or_else(|| "none".to_string(), |id| id.to_string())
}

#[cfg(feature = "cuda")]
fn top5(logits: &[f32]) -> Vec<(u32, f32)> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    indices
        .into_iter()
        .take(5)
        .map(|index| (index as u32, logits[index]))
        .collect()
}

#[cfg(feature = "cuda")]
fn format_top5(top: &[(u32, f32)]) -> String {
    top.iter()
        .map(|(token, value)| format!("{token}: {value:.9}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(feature = "cuda")]
fn top5_overlap(reference: &[(u32, f32)], hybrid: &[(u32, f32)]) -> usize {
    reference
        .iter()
        .filter(|(token, _)| hybrid.iter().any(|(other, _)| other == token))
        .count()
}

#[cfg(feature = "cuda")]
fn first_layer_difference(
    reference: &sparkflash_engine::forward::boundary::BoundaryTable,
    hybrid: &sparkflash_engine::forward::HybridTokenOutput,
) -> Vec<String> {
    for trace in &hybrid.layers {
        let Some(row) = reference.row(trace.layer) else {
            return vec![format!(
                "first differing layer: layer {} has no reference boundary row",
                trace.layer
            )];
        };
        let Some(tensor) = row.tensor(sparkflash_engine::forward::boundary::RESIDUAL_OUT) else {
            return vec![format!(
                "first differing layer: layer {} has no reference residual_out",
                trace.layer
            )];
        };
        let Some(values) = tensor.data().as_f32() else {
            continue;
        };
        if values.len() != trace.residual_out.len() {
            return vec![format!(
                "first differing layer: layer {} residual_out lengths differ ({} vs {})",
                trace.layer,
                values.len(),
                trace.residual_out.len()
            )];
        }
        let max_abs = values
            .iter()
            .zip(&trace.residual_out)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        if max_abs > 0.0 {
            return vec![format!(
                "first differing layer (best effort): layer {} residual_out max_abs_delta={:.6e}; this includes the BF16 staging seam and any device-entry delta",
                trace.layer, max_abs
            )];
        }
    }
    vec!["first differing layer (best effort): no residual_out difference found above zero; the token difference is later in the tail".to_string()]
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
    if stage == "hybrid_token" {
        return run_hybrid_lines(&args);
    }
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
    fn the_hybrid_token_arm_accepts_only_token_zero_and_position_zero() {
        let (stage, args) = stage_args(&[
            "hybrid_token",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--token",
            "0",
        ])
        .expect("hybrid_token arguments");
        assert_eq!(stage, "hybrid_token");
        assert_eq!(args.token, 0);
        assert_eq!(args.positions, vec![0]);
        let error = stage_args(&[
            "hybrid_token",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--token",
            "1",
        ])
        .expect_err("hybrid_token compares against the token-0 pass");
        assert!(error.contains("--token must be 0"), "{error}");
        let error = stage_args(&[
            "hybrid_token",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--positions",
            "1",
        ])
        .expect_err("hybrid_token is position 0 only");
        assert!(error.contains("position 0 only"), "{error}");
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
