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

/// The stream the deferred step-1 entry consumes: the previous layer's readback X, never this
/// layer's own X readback or the FFN output Y. The b28 bug used the wrong stream here.
#[cfg(any(feature = "cuda", test))]
fn deferred_entry_input(deferred: &sparkflash_engine::forward::DeferredHandoff) -> &[f32] {
    deferred.x.as_slice()
}

#[cfg(any(feature = "cuda", test))]
fn over_bar(delta: f32, bar: f32) -> bool {
    // `f32::max` ignores NaN and `NaN > bar` is false, so a NaN delta would otherwise pass every
    // bar in this file. A non-finite delta is always over the bar.
    !delta.is_finite() || delta > bar
}

/// The released-row FFN predictor the host `hc_mixes` twin compares against.
///
/// `LayerPostStaging::predictor` is permuted into the kernel's B/A/C row order for upload;
/// `hc_mixes` takes the released pre/post/comb row order. Passing the kernel-order vector here
/// compares a device coefficient with a twin projected from the wrong row map.
#[cfg(any(feature = "cuda", test))]
fn post_host_predictor(staging: &sparkflash_device::LayerPostStaging) -> &[f32] {
    &staging.released_fn
}

/// The deferred step-1 twin's input stream, with the device readback deliberately ignored.
///
/// `readback` is the device's X_l; passing it back into the update reproduces the b28 bug. The
/// twin must consume `input` (the previous layer's X / Engram mixer input) and only `input`.
#[cfg(any(feature = "cuda", test))]
fn deferred_twin_update(
    input: &[u8],
    readback: &[u8],
    b: &[f32],
    c: &[f32],
    y: &[f32],
) -> Vec<f32> {
    let _ = readback;
    sparkflash_engine::forward::layer_zero::kernel_mhc_update_fused(
        &decode_bf16(input),
        b,
        c,
        &decode_bf16(&sparkflash_engine::forward::layer_zero::bf16_bytes(y)),
    )
}

/// The layers whose attention body did not launch exactly once, in device order.
#[cfg(any(feature = "cuda", test))]
fn missing_body_layers(layers: &[usize], body_launches: &[usize]) -> Vec<usize> {
    layers
        .iter()
        .zip(body_launches)
        .filter_map(|(layer, launches)| (*launches != 1).then_some(*layer))
        .collect()
}

/// The finite maximum of two deltas, with any non-finite delta mapped to `+inf`.
///
/// A plain `f32::max` silently drops a NaN delta, which would let a non-finite coefficient
/// through the m11 bars; routing every fold through this function keeps that rule at the call
/// site instead of relying on a later comparison.
#[cfg(any(feature = "cuda", test))]
fn finite_max(acc: f32, delta: f32) -> f32 {
    if delta.is_finite() {
        acc.max(delta)
    } else {
        f32::INFINITY
    }
}

/// The finite max-abs delta of two aligned rows, with non-finite values mapped to `+inf`.
#[cfg(any(feature = "cuda", test))]
fn max_abs_delta(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, finite_max)
}

/// The c4a-2 seam decision: the format-2 body must launch exactly once on every layer.
///
/// Returns the seam line only when that holds; otherwise it names the layers that did not and
/// withholds the line, so the printer cannot emit a seam line that contradicts the ledger.
#[cfg(any(feature = "cuda", test))]
fn c4a2_seam_line(
    layers: &[usize],
    body_launches: &[usize],
    entry_launches: usize,
) -> Result<String, String> {
    let missing = missing_body_layers(layers, body_launches);
    if !missing.is_empty() {
        return Err(format!(
            "c4a-2 seam line withheld: the format-2 body must launch exactly once on every layer, but layers {missing:?} do not (body_launches != 1); the per-layer ledger and totals would otherwise disagree"
        ));
    }
    let post_launches = layers.len();
    let mhc_launches = entry_launches + post_launches;
    Ok(format!(
        "c4a-2 seam: format-2 body launched exactly once on all {} layers; {} projection + {} body launches plus the {} mHC launches ({} entry + {} post)",
        layers.len(),
        layers.len() * 5,
        layers.len(),
        mhc_launches,
        entry_launches,
        post_launches
    ))
}

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
    use sparkflash_device::staging::{
        LayerPostStaging, LayerStaging, LayerZeroStaging, StagingError,
    };
    use sparkflash_engine::forward::engram;
    use sparkflash_engine::forward::layer_zero::{
        LayerZeroEntrySource, M11_HC_COEFF_BAR, M11_INV_RMS_BAR, bf16_bytes, kernel_mhc_capture,
        kernel_mhc_update_fused, kernel_mhc_xhat,
    };
    use sparkflash_engine::forward::mhc::{self, Mixes};
    use sparkflash_engine::forward::{
        DeferredHandoff, HybridEntry, HybridPost, InferenceEngine, SequenceState,
    };
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
        "sparkflash device-chain: M3-c c4a-2 HYBRID TOKEN - 40 layers, one prompt token, one position".to_string(),
        "device entry path: layer 0 keeps the step-1 identity (B=I, C=0, Y=0, A=initial pre_mix); layers 1..39 consume the previous-layer deferred FFN epilogue: B/C/A read back then re-uploaded (declared coefficient round-trip); Y is the host FFN output, BF16-staged; the add runs in device step 1.".to_string(),
        "device post path: each layer runs a device MhcPost after attention. It writes the post-attention residual X and the 24 FFN coefficients, and captures the normalized FFN input; the host FFN consumes those device values.".to_string(),
        "ffn_epilogue split: layers 0..38 move the FFN epilogue add into the next device MhcEntry step-1; layer 39 keeps the host FFN epilogue add because there is no next entry. Count: 39 deferred, 1 host.".to_string(),
        "Engram seam: the released config puts Engram at layers 1 and 14. Engram adds to H_l after the deferred step-1 and before the attention entry, which one fused entry call cannot host-insert; those two layers run the real B/C/Y mixer device, apply Engram on the host readback, then run a second identity device entry. That is 2 extra MhcEntry launches and a declared host Engram seam, not a device Engram.".to_string(),
        "residual staging seam: each layer stages the f32 residual to BF16 and the host continues from the device-readback X; the chain is therefore not bit-identical to the reference even with exact host arithmetic".to_string(),
        "attention projection seam: wq_a, wq_b, wkv, wo_a and wo_b are device B12x MXFP8 GEMMs replacing the host f32 projections; q_norm, kv_norm, RoPE, the window/compressor/indexer host seams, the inverse RoPE and the D8 quantisers remain declared HOST stand-ins.".to_string(),
        "attention body seam (c4a-2): m14 is the real device format-2 flash_mla_csa2_sm121 launch, 64 head blocks of 256 threads, one launch per layer at position 0. Window codes/scales are built by the engine window row helper from the host kv row; the selected compressed list is empty at position 0, and the kernel's indexer stage-1 buffers are uploaded as zeros plus sentinels and their writes are ignored. The 512-candidate / 640-slot regime is NOT exercised at position 0.".to_string(),
        "attention seam ledger: 3 launches/layer for wq_a+wq_b+wkv (120), 1 format-2 attention launch/layer (40) and 2 launches/layer for wo_a+wo_b (80) = 240 attention launches; 42 MhcEntry + 40 MhcPost + 240 attention = 322 chain launches. The chain token comparison remains the CHAIN-WIRING gate.".to_string(),
        "claim: CHAIN-WIRING parity, not a device-token result; criterion 1 remains OPEN".to_string(),
        "caching seam: no 40-layer staging metadata cache; attention and FFN staging objects are rebuilt and dropped per layer, and the five attention weight sets are restaged from the checkpoint per layer. Reused device allocations: the 16-buffer mHC set (2,064,352 B) plus the attention projection buffers.".to_string(),
        "per-layer host stand-ins after c4a-2 (device-status but host-executed here): moe_router, moe_gate_up, moe_down_combine, shared_expert_gate. The five attention projections and the format-2 attention body are device at position 0. Attention host seams: q_norm, kv_norm, RoPE, window row construction, compressor/cache, selected-list assembly, indexer scoring, inverse RoPE, D8 quantisers. Tail stand-in: lm_head_screen. Tail host seams: final_norm, lm_head_argmax.".to_string(),
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
    let mut engram_hashes = Vec::new();
    engram::ngram_hashes(&config, &[args.token], 0, &mut engram_hashes)
        .map_err(|error| format!("engram token hashes: {error}"))?;

    let mem_available_bytes = read_mem_available_bytes()?;
    let mut attention_backend =
        crate::attention_device::AttentionDeviceBackend::open(&args.cubin_dir)?;
    let launcher = Rc::new(RefCell::new(CudaLauncher::open_for_stage(
        &args.cubin_dir,
        "mega_mhc_sm121",
    )?));
    let cubin_bytes = launcher.borrow().cubin_bytes();
    let provider_source = SafeTensorsSource::open(
        &args.checkpoint,
        &args.checkpoint.join("model.safetensors.index.json"),
    )
    .map_err(|error| format!("staging checkpoint {}: {error}", args.checkpoint.display()))?;

    #[derive(Clone)]
    struct DeviceTiming {
        staging_ms: u128,
        entry_us: u128,
        post_us: u128,
        capture_exact: bool,
        codes_exact: bool,
        scales_exact: bool,
        coeff_max_abs: f32,
        inv_rms_delta: f32,
        post_capture_exact: bool,
        post_codes_exact: bool,
        post_scales_exact: bool,
        post_coeff_max_abs: f32,
        post_inv_rms_delta: f32,
        engram_split: bool,
        x_bytes: usize,
        matrix_bytes: u64,
        body_launches: usize,
    }
    let blank_timing = DeviceTiming {
        staging_ms: 0,
        entry_us: 0,
        post_us: 0,
        capture_exact: false,
        codes_exact: false,
        scales_exact: false,
        coeff_max_abs: 0.0,
        inv_rms_delta: 0.0,
        post_capture_exact: false,
        post_codes_exact: false,
        post_scales_exact: false,
        post_coeff_max_abs: 0.0,
        post_inv_rms_delta: 0.0,
        engram_split: false,
        x_bytes: 0,
        matrix_bytes: 0,
        body_launches: 0,
    };
    let device_timings = Rc::new(RefCell::new(vec![blank_timing; config.n_layers]));
    let post_staging_slot: Rc<RefCell<Option<LayerPostStaging>>> = Rc::new(RefCell::new(None));
    let mut staged_first = false;
    let device_total_us = Rc::new(Cell::new(0u128));
    let completed = Rc::new(Cell::new(0usize));
    let progress_lines: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let mut provider = |layer: usize,
                        hidden: &[f32],
                        pre_mix: &[f32],
                        deferred: Option<&DeferredHandoff>|
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
        let post_staging = LayerPostStaging::from_source(&provider_source, layer).map_err(
            |error: StagingError| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post staging"),
                detail: error.to_string(),
            },
        )?;
        let matrix_bytes = staging.host_bytes() + post_staging.host_bytes();
        if !staged_first {
            let request = sparkflash_device::BudgetRequest {
                mem_available_bytes,
                projected_host_bytes: matrix_bytes + cubin_bytes,
                projected_device_bytes: staging.device_buffer_bytes(),
            };
            launcher
                .borrow_mut()
                .stage_layer(&staging, request)
                .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} staging admission"),
                    detail: error.to_string(),
                })?;
            staged_first = true;
        } else {
            launcher
                .borrow_mut()
                .restage_layer(&staging)
                .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} restage"),
                    detail: error,
                })?;
        }
        *post_staging_slot.borrow_mut() = Some(post_staging);
        let staging_ms = staging_started.elapsed().as_millis();
        let engram_layer = config.engram_rows(layer).is_some();
        let mut device_started: Option<Instant> = None;
        let mut engram_mixer_us: u128 = 0;
        let x_bytes = if layer == 0 {
            launcher
                .borrow_mut()
                .write_layer_a(pre_mix)
                .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} A write"),
                    detail: error,
                })?;
            let mut x = hidden.to_vec();
            if engram_layer {
                engram::engram_forward(
                    &config,
                    layer,
                    &mut x,
                    &engram_hashes,
                    &provider_source,
                    limit,
                )
                .map_err(|error| {
                    sparkflash_engine::linalg::EngineError::Shape {
                        what: format!("layer {layer} Engram"),
                        detail: error.to_string(),
                    }
                })?;
            }
            bf16_bytes(&x)
        } else {
            let deferred =
                deferred.ok_or_else(|| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} deferred handoff"),
                    detail: "layers 1..39 require the previous layer's deferred FFN epilogue"
                        .to_string(),
                })?;
            launcher
                .borrow_mut()
                .write_layer_bc(&deferred.mixes.post, &deferred.mixes.comb)
                .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} B/C write"),
                    detail: error,
                })?;
            launcher
                .borrow_mut()
                .write_layer_a(&deferred.mixes.pre)
                .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} A write"),
                    detail: error,
                })?;
            launcher
                .borrow_mut()
                .write_layer_y_bf16(&bf16_bytes(&deferred.y))
                .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} Y write"),
                    detail: error,
                })?;
            if engram_layer {
                // Engram [1,14] adds to H_l after the deferred FFN epilogue and before the
                // attention entry. One fused kernel call cannot host-insert that addition
                // between step 1 and step 2, so run the real B/C/Y mixer first, apply Engram on
                // the readback, then run an identity step-1 entry for the Engram-adjusted H_l.
                // Two declared host seams, not a device-to-device Engram.
                let first_started = Instant::now();
                let first_x = bf16_bytes(deferred_entry_input(deferred));
                launcher
                    .borrow_mut()
                    .write_layer_x_bf16(&first_x)
                    .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                        what: format!("layer {layer} Engram mixer X write"),
                        detail: error,
                    })?;
                launcher
                    .borrow_mut()
                    .launch(DeviceStageCall {
                        stage: ChainStage::MhcEntry,
                        layer,
                        position: 0,
                        kernel: "mega_mhc_sm121",
                    })
                    .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                        what: format!("layer {layer} Engram mixer launch"),
                        detail: error.to_string(),
                    })?;
                let mixer_x = launcher.borrow_mut().read_layer_x().map_err(|error| {
                    sparkflash_engine::linalg::EngineError::Shape {
                        what: format!("layer {layer} Engram mixer X readback"),
                        detail: error,
                    }
                })?;
                let mut b = vec![0.0f32; config.hc_mult * config.hc_mult];
                for destination in 0..config.hc_mult {
                    for source in 0..config.hc_mult {
                        b[destination * config.hc_mult + source] =
                            deferred.mixes.comb[source * config.hc_mult + destination];
                    }
                }
                let mixer_update =
                    deferred_twin_update(&first_x, &mixer_x, &b, &deferred.mixes.post, &deferred.y);
                if mixer_x != bf16_bytes(&mixer_update) {
                    let difference = bf16_write_difference(&mixer_x, &mixer_update);
                    return Err(sparkflash_engine::linalg::EngineError::Shape {
                        what: format!("layer {layer} Engram mixer write"),
                        detail: format!(
                            "the deferred step-1 X_l is not the BF16 write of the fused host twin; first differing X element: {difference:?}; input X=BF16(deferred A_{{l-1}}), B=kernel-order transpose(deferred FFN comb), C=deferred FFN post, Y=BF16(deferred FFN output)"
                        ),
                    });
                }
                engram_mixer_us = first_started.elapsed().as_micros();
                let mut engrammed = decode_bf16(&mixer_x);
                engram::engram_forward(
                    &config,
                    layer,
                    &mut engrammed,
                    &engram_hashes,
                    &provider_source,
                    limit,
                )
                .map_err(|error| {
                    sparkflash_engine::linalg::EngineError::Shape {
                        what: format!("layer {layer} Engram"),
                        detail: error.to_string(),
                    }
                })?;
                launcher
                    .borrow_mut()
                    .restage_layer(&staging)
                    .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                        what: format!("layer {layer} Engram restage"),
                        detail: error,
                    })?;
                launcher
                    .borrow_mut()
                    .write_layer_a(&deferred.mixes.pre)
                    .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                        what: format!("layer {layer} Engram A write"),
                        detail: error,
                    })?;
                device_started = Some(Instant::now());
                bf16_bytes(&engrammed)
            } else {
                bf16_bytes(deferred_entry_input(deferred))
            }
        };
        let device_started = device_started.unwrap_or_else(Instant::now);
        launcher
            .borrow_mut()
            .write_layer_x_bf16(&x_bytes)
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} X write"),
                detail: error,
            })?;
        launcher
            .borrow_mut()
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
        let entry = launcher.borrow_mut().read_layer_entry().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} entry readback"),
                detail: error,
            }
        })?;
        let x_after = launcher.borrow_mut().read_layer_x().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} X readback"),
                detail: error,
            }
        })?;
        let inv_rms = launcher
            .borrow_mut()
            .read_layer_inv_rms()
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} 1/rms readback"),
                detail: error,
            })?;
        let codes = launcher.borrow_mut().read_layer_codes().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} FP8 codes readback"),
                detail: error,
            }
        })?;
        let scales = launcher.borrow_mut().read_layer_scales().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} FP8 scales readback"),
                detail: error,
            }
        })?;
        let entry_us = device_started.elapsed().as_micros() + engram_mixer_us;
        device_total_us.set(device_total_us.get() + entry_us);
        let x_readback = decode_bf16(&x_after);
        let (b_prev, c_prev, a_prev, y_prev) = match deferred {
            Some(deferred) if engram_layer => {
                // The final launch after Engram is the identity step-1 entry; only A is the
                // previous FFN pre row.
                let (b, c, _, y) = LayerZeroStaging::layer_zero_seed();
                (b.to_vec(), c.to_vec(), deferred.mixes.pre.clone(), y)
            }
            Some(deferred) => {
                let mut b = vec![0.0f32; config.hc_mult * config.hc_mult];
                for destination in 0..config.hc_mult {
                    for source in 0..config.hc_mult {
                        b[destination * config.hc_mult + source] =
                            deferred.mixes.comb[source * config.hc_mult + destination];
                    }
                }
                (
                    b,
                    deferred.mixes.post.clone(),
                    deferred.mixes.pre.clone(),
                    decode_bf16(&bf16_bytes(&deferred.y)),
                )
            }
            None => {
                let (b, c, a, y) = LayerZeroStaging::layer_zero_seed();
                (b.to_vec(), c.to_vec(), a.to_vec(), y)
            }
        };
        let host_update = deferred_twin_update(&x_bytes, &x_after, &b_prev, &c_prev, &y_prev);
        let expected_x = bf16_bytes(&host_update);
        if x_after != expected_x {
            let difference = bf16_write_difference(&x_after, &host_update);
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} deferred step-1 write"),
                detail: format!(
                    "the X_l readback is not the BF16 write of the host twin's B X + C Y update; first differing X element: {difference:?}; input X=BF16({}), B={}, C={}, Y={}",
                    if layer == 0 {
                        "hidden"
                    } else {
                        "deferred A_{l-1}"
                    },
                    if engram_layer && deferred.is_some() {
                        "identity"
                    } else if deferred.is_some() {
                        "transpose(deferred FFN comb)"
                    } else {
                        "identity"
                    },
                    if engram_layer && deferred.is_some() {
                        "identity"
                    } else if deferred.is_some() {
                        "deferred FFN post"
                    } else {
                        "identity"
                    },
                    if engram_layer && deferred.is_some() {
                        "identity"
                    } else if deferred.is_some() {
                        "BF16(deferred FFN output)"
                    } else {
                        "identity"
                    }
                ),
            });
        }
        let host_xhat = kernel_mhc_xhat(&host_update, &a_prev);
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
            &host_update,
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
        mhc::hc_pre(&config, &a_prev, &host_update, &mut host_attn_in).map_err(|error| {
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
            .fold(0.0f32, finite_max);
        let inv_rms_delta = (inv_rms - host_inv).abs();
        if !capture_exact || !codes_exact || !scales_exact {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} block-input comparison"),
                detail: format!(
                    "capture exact={capture_exact}, codes exact={codes_exact}, scales exact={scales_exact}"
                ),
            });
        }
        if over_bar(coeff_max_abs, M11_HC_COEFF_BAR) {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} hc_coeffs bar"),
                detail: format!("max abs {coeff_max_abs:.6e} exceeds the 1e-4 bar"),
            });
        }
        if over_bar(inv_rms_delta, M11_INV_RMS_BAR) {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} 1/rms bar"),
                detail: format!("max abs delta {inv_rms_delta:.6e} exceeds the 3e-4 bar"),
            });
        }
        {
            let mut timings = device_timings.borrow_mut();
            let timing = &mut timings[layer];
            timing.staging_ms = staging_ms;
            timing.entry_us = entry_us;
            timing.capture_exact = capture_exact;
            timing.codes_exact = codes_exact;
            timing.scales_exact = scales_exact;
            timing.coeff_max_abs = coeff_max_abs;
            timing.inv_rms_delta = inv_rms_delta;
            timing.engram_split = engram_layer && deferred.is_some();
            timing.x_bytes = x_after.len();
            timing.matrix_bytes = matrix_bytes;
        }
        completed.set(done + 1);
        Ok(HybridEntry {
            entry,
            x: x_readback,
        })
    };
    let mut post = |layer: usize,
                    residual: &[f32],
                    attention_out: &[f32],
                    attn_mixes: &Mixes|
     -> Result<HybridPost, sparkflash_engine::linalg::EngineError> {
        let post_staging = post_staging_slot.borrow_mut().take().ok_or_else(|| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post staging"),
                detail: "the entry provider did not leave one FFN staging object for the post"
                    .to_string(),
            }
        })?;
        launcher
            .borrow_mut()
            .restage_layer_post(&post_staging)
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post restage"),
                detail: error,
            })?;
        launcher
            .borrow_mut()
            .write_layer_bc(&attn_mixes.post, &attn_mixes.comb)
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post B/C write"),
                detail: error,
            })?;
        launcher
            .borrow_mut()
            .write_layer_a(&attn_mixes.pre)
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post A write"),
                detail: error,
            })?;
        launcher
            .borrow_mut()
            .write_layer_y_bf16(&bf16_bytes(attention_out))
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post Y write"),
                detail: error,
            })?;
        let post_started = Instant::now();
        launcher
            .borrow_mut()
            .launch(DeviceStageCall {
                stage: ChainStage::MhcPost,
                layer,
                position: 0,
                kernel: "mega_mhc_sm121",
            })
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} MhcPost launch"),
                detail: error.to_string(),
            })?;
        let entry = launcher.borrow_mut().read_layer_entry().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post entry readback"),
                detail: error.to_string(),
            }
        })?;
        let a_after = launcher.borrow_mut().read_layer_x().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post X readback"),
                detail: error.to_string(),
            }
        })?;
        let inv_rms = launcher
            .borrow_mut()
            .read_layer_inv_rms()
            .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post 1/rms readback"),
                detail: error.to_string(),
            })?;
        let codes = launcher.borrow_mut().read_layer_codes().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post FP8 codes readback"),
                detail: error.to_string(),
            }
        })?;
        let scales = launcher.borrow_mut().read_layer_scales().map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post FP8 scales readback"),
                detail: error.to_string(),
            }
        })?;
        let post_us = post_started.elapsed().as_micros();
        device_total_us.set(device_total_us.get() + post_us);
        let mut b_prev = vec![0.0f32; config.hc_mult * config.hc_mult];
        for destination in 0..config.hc_mult {
            for source in 0..config.hc_mult {
                b_prev[destination * config.hc_mult + source] =
                    attn_mixes.comb[source * config.hc_mult + destination];
            }
        }
        let y_prev = decode_bf16(&bf16_bytes(attention_out));
        let host_update = kernel_mhc_update_fused(residual, &b_prev, &attn_mixes.post, &y_prev);
        let expected_x = bf16_bytes(&host_update);
        if a_after != expected_x {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post step-2 write"),
                detail: "the post X_l readback is not the BF16 write of the host twin's B X + C Y update".to_string(),
            });
        }
        let host_xhat = kernel_mhc_xhat(&host_update, &attn_mixes.pre);
        let expected_capture = kernel_mhc_capture(&host_xhat, &post_staging.ffn_norm, inv_rms);
        let units = quantize_activation_fp8(&expected_capture).map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post host block-input quantisation"),
                detail: error.to_string(),
            }
        })?;
        let capture_exact = entry.pre_attention_hidden == expected_capture;
        let codes_exact = codes == units.codes();
        let scales_exact = scales == units.scales();
        let x_readback = decode_bf16(&a_after);
        let mut host_mixes = Mixes::for_config(&config);
        mhc::hc_mixes(
            &config,
            &host_update,
            post_host_predictor(&post_staging),
            &post_staging.hc_scale,
            &post_staging.hc_base,
            &mut host_mixes,
        )
        .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
            what: format!("layer {layer} post host hc_mixes"),
            detail: error.to_string(),
        })?;
        let mut host_ffn_in = vec![0.0f32; dim];
        mhc::hc_pre(&config, &attn_mixes.pre, &host_update, &mut host_ffn_in).map_err(|error| {
            sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post host hc_pre"),
                detail: error.to_string(),
            }
        })?;
        let host_inv = host_inv_rms(&host_ffn_in, config.norm_eps);
        let mut coeff_max_abs = 0.0f32;
        let mut first_bad = String::new();
        let mut worst = String::new();
        let hc = config.hc_mult;
        for (set, device, host) in [
            ("pre", entry.pre.as_slice(), host_mixes.pre.as_slice()),
            ("post", entry.post.as_slice(), host_mixes.post.as_slice()),
            ("comb", entry.comb.as_slice(), host_mixes.comb.as_slice()),
        ] {
            for (index, (device, host)) in device.iter().zip(host).enumerate() {
                let delta = (device - host).abs();
                let label = if set == "comb" {
                    format!("comb[row={}][col={}]", index / hc, index % hc)
                } else {
                    format!("{set}[{index}]")
                };
                if !delta.is_finite() {
                    if first_bad.is_empty() {
                        first_bad =
                            format!("{label} device={device:.9e} twin={host:.9e} delta=non-finite");
                    }
                    coeff_max_abs = f32::INFINITY;
                    worst = format!("{label} delta=non-finite");
                    continue;
                }
                if delta > M11_HC_COEFF_BAR && first_bad.is_empty() {
                    first_bad =
                        format!("{label} device={device:.9e} twin={host:.9e} delta={delta:.3e}");
                }
                if delta > coeff_max_abs {
                    coeff_max_abs = delta;
                    worst =
                        format!("{label} device={device:.9e} twin={host:.9e} delta={delta:.3e}");
                }
            }
        }
        let inv_rms_delta = (inv_rms - host_inv).abs();
        if !capture_exact || !codes_exact || !scales_exact {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post block-input comparison"),
                detail: format!(
                    "capture exact={capture_exact}, codes exact={codes_exact}, scales exact={scales_exact}"
                ),
            });
        }
        if over_bar(coeff_max_abs, M11_HC_COEFF_BAR) {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post hc_coeffs bar"),
                detail: format!(
                    "max abs {coeff_max_abs:.6e} exceeds the 1e-4 bar; first over bar: {first_bad}; worst: {worst}"
                ),
            });
        }
        if over_bar(inv_rms_delta, M11_INV_RMS_BAR) {
            return Err(sparkflash_engine::linalg::EngineError::Shape {
                what: format!("layer {layer} post 1/rms bar"),
                detail: format!("max abs delta {inv_rms_delta:.6e} exceeds the 3e-4 bar"),
            });
        }
        {
            let mut timings = device_timings.borrow_mut();
            let timing = &mut timings[layer];
            timing.post_us = post_us;
            timing.post_capture_exact = capture_exact;
            timing.post_codes_exact = codes_exact;
            timing.post_scales_exact = scales_exact;
            timing.post_coeff_max_abs = coeff_max_abs;
            timing.post_inv_rms_delta = inv_rms_delta;
        }
        let mut ffn_mixes = Mixes::for_config(&config);
        ffn_mixes.pre.copy_from_slice(&entry.pre);
        ffn_mixes.post.copy_from_slice(&entry.post);
        ffn_mixes.comb.copy_from_slice(&entry.comb);
        Ok(HybridPost {
            x: x_readback,
            ffn_in: entry.pre_attention_hidden,
            ffn_mixes,
            source: LayerZeroEntrySource::Device,
        })
    };
    let mut progress = |layer: usize, host_ms: u64| {
        if layer % 8 == 7 {
            let chain_ms =
                started.elapsed().as_millis() as u64 - reference_elapsed.as_millis() as u64;
            let line = format!(
                "progress: layer {}/{} complete; chain_elapsed_ms={}; device_us={}; host_ms_last={}",
                layer + 1,
                config.n_layers,
                chain_ms,
                device_total_us.get(),
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
    let chain_result = chain_engine.run_hybrid_chain_deferred(
        args.token,
        &embedding,
        &mut provider,
        &mut post,
        Some(&mut attention_backend),
        &mut progress,
    );
    let output = match chain_result {
        Ok(output) => output,
        Err(error) => {
            lines.extend(progress_lines.borrow().iter().cloned());
            for (layer, timing) in device_timings.borrow().iter().enumerate() {
                lines.push(format!(
                    "partial layer {layer} timing: staging_ms={} entry_us={} post_us={} capture_exact={} codes_exact={} scales_exact={} coeff_max_abs={:.6e} inv_rms_delta={:.6e} post_capture_exact={} post_codes_exact={} post_scales_exact={} post_coeff_max_abs={:.6e} post_inv_rms_delta={:.6e}",
                    timing.staging_ms,
                    timing.entry_us,
                    timing.post_us,
                    timing.capture_exact,
                    timing.codes_exact,
                    timing.scales_exact,
                    timing.coeff_max_abs,
                    timing.inv_rms_delta,
                    timing.post_capture_exact,
                    timing.post_codes_exact,
                    timing.post_scales_exact,
                    timing.post_coeff_max_abs,
                    timing.post_inv_rms_delta
                ));
            }
            let attention_stats = attention_backend.stats();
            lines.push(format!(
                "attention partial totals: projection_launches={} body_launches={} staging_ms={} projection_us={} body_us={} staged_matrix_bytes={} host_bytes={}",
                attention_stats.launches,
                attention_stats.attention_launches,
                attention_stats.staging_us / 1000,
                attention_stats.device_us,
                attention_stats.attention_us,
                attention_stats.staged_bytes,
                attention_stats.host_bytes
            ));
            lines.push(format!("c4a-2 chain abort: {error}"));
            return Err(lines.join("\n"));
        }
    };
    for (layer, timing) in device_timings.borrow_mut().iter_mut().enumerate() {
        timing.body_launches = attention_backend.body_launches(layer);
    }
    let total_elapsed = started.elapsed();
    let chain_elapsed = total_elapsed.saturating_sub(reference_elapsed);
    lines.extend(progress_lines.borrow().iter().cloned());
    let attention_stats = attention_backend.stats();
    let timings = device_timings.borrow();
    let attention_device_us = attention_stats.device_us as u128;
    let attention_body_us = attention_stats.attention_us as u128;
    let attention_staging_ms = (attention_stats.staging_us / 1000) as u128;
    let total_device_us: u128 = timings
        .iter()
        .map(|timing| timing.entry_us + timing.post_us)
        .sum::<u128>()
        + attention_device_us
        + attention_body_us;
    let total_staging_ms: u128 =
        timings.iter().map(|timing| timing.staging_ms).sum::<u128>() + attention_staging_ms;
    let total_host_ms: u64 = output.layers.iter().map(|trace| trace.host_ms).sum();
    let total_host_bytes: u64 = output
        .layers
        .iter()
        .flat_map(|trace| trace.stages.iter())
        .map(|event| event.host_bytes_in.saturating_add(event.host_bytes_out))
        .sum::<u64>()
        + attention_stats.host_bytes;
    let total_matrix_bytes: u64 = timings
        .iter()
        .map(|timing| timing.matrix_bytes)
        .sum::<u64>()
        + attention_stats.staged_bytes;
    let attention_cubin_bytes: u64 = [
        "b12x_mxfp8_dense_sm121",
        "b12x_mxfp8_wo_a_sm121",
        "flash_mla_csa2_sm121",
    ]
    .iter()
    .filter_map(|name| sparkflash_kernels::baseline::pin(name).ok().flatten())
    .map(|pin| pin.cubin_bytes as u64)
    .sum();
    let total_cubin_bytes = cubin_bytes + attention_cubin_bytes;
    let total_device_buffer_bytes = sparkflash_device::MhcBufferPlan::LAYER_ZERO.device_bytes()
        as u64
        + attention_backend.device_bytes();
    let body_layers: Vec<usize> = output.layers.iter().map(|trace| trace.layer).collect();
    let body_counts: Vec<usize> = output
        .layers
        .iter()
        .map(|trace| {
            timings
                .get(trace.layer)
                .map_or(0, |timing| timing.body_launches)
        })
        .collect();
    let engram_entries = timings.iter().filter(|timing| timing.engram_split).count();
    let c4a2_line = c4a2_seam_line(
        &body_layers,
        &body_counts,
        body_layers.len() + engram_entries,
    )?;
    for (device_index, trace) in output.layers.iter().enumerate() {
        let timing = timings
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
                let seam: String = match event.stage {
                    "mhc_entry" if trace.layer == 0 => "[device identity seam]".to_string(),
                    "mhc_entry" if timing.engram_split => {
                        "[device deferred FFN epilogue + Engram mixer + identity re-entry]"
                            .to_string()
                    }
                    "mhc_entry" => "[device deferred FFN epilogue + attention entry]".to_string(),
                    "mhc_post" => "[device post; 24 FFN coeffs + FFN input capture]".to_string(),
                    "ffn_epilogue" if event.host_bytes_in == 0 => {
                        "[deferred into next MhcEntry]".to_string()
                    }
                    "ffn_epilogue" => "[host FFN add; layer 39 only]".to_string(),
                    "attn_norm" => "[device capture]".to_string(),
                    "attention_projections" => {
                        "[device: wq_a+wq_b+wkv, 3 launches; host q_norm + D8 quant seams]"
                            .to_string()
                    }
                    "attention" if timing.body_launches == 1 => {
                        "[device: format-2 flash-MLA, 1 launch; host q/window/selected seams]"
                            .to_string()
                    }
                    "attention" => format!(
                        "[HOST STAND-IN: m14 flash-MLA; body_launches={}]",
                        timing.body_launches
                    ),
                    "attention_output_projection" => {
                        "[device: wo_a+wo_b, 2 launches; host inverse-RoPE + D8 quant seams]"
                            .to_string()
                    }
                    _ => String::new(),
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
            "layer {:02}: staging_ms={} engram_split={} entry_us={} post_us={} device_us={} host_ms={} host_bytes={} staged_matrix_bytes={} x_bytes={} capture_exact={} codes_exact={} scales_exact={} coeff_max_abs={:.6e} inv_rms_delta={:.6e} post_capture_exact={} post_codes_exact={} post_scales_exact={} post_coeff_max_abs={:.6e} post_inv_rms_delta={:.6e}",
            trace.layer,
            timing.staging_ms,
            timing.engram_split,
            timing.entry_us,
            timing.post_us,
            timing.entry_us + timing.post_us,
            trace.host_ms,
            host_bytes,
            timing.matrix_bytes,
            timing.x_bytes,
            timing.capture_exact,
            timing.codes_exact,
            timing.scales_exact,
            timing.coeff_max_abs,
            timing.inv_rms_delta,
            timing.post_capture_exact,
            timing.post_codes_exact,
            timing.post_scales_exact,
            timing.post_coeff_max_abs,
            timing.post_inv_rms_delta
        ));
    }
    lines.push(c4a2_line);
    lines.push(format!(
        "attention device path: attention_projections=wq_a+wq_b+wkv 3 launches/layer; attention=format-2 flash-MLA 1 launch/layer at position 0; attention_output_projection=wo_a+wo_b 2 launches/layer; 200 projection + 40 attention launches plus 82 mHC = {}",
        output.layers.len() * 5 + output.layers.len() + output.layers.len() * 2 + engram_entries
    ));
    lines.push(format!(
        "attention totals: projection_launches={} body_launches={} staging_ms={} projection_us={} body_us={} staged_matrix_bytes={} host_bytes={} cubin_bytes={} device_buffer_bytes={}",
        attention_stats.launches,
        attention_stats.attention_launches,
        attention_staging_ms,
        attention_stats.device_us,
        attention_stats.attention_us,
        attention_stats.staged_bytes,
        attention_stats.host_bytes,
        attention_cubin_bytes,
        attention_backend.device_bytes()
    ));
    lines.push(format!(
        "chain totals: layers={} entry_launches={} post_launches={} attention_projection_launches={} attention_body_launches={} device_launches={} engram_split_layers={} device_us={} device_ms={:.3} staging_ms={} host_ms={} total_chain_ms={} host_bytes={} staged_matrix_bytes={} cubin_bytes={} device_buffer_bytes={}",
        output.layers.len(),
        output.layers.len() + engram_entries,
        output.layers.len(),
        attention_stats.launches,
        attention_stats.attention_launches,
        output.layers.len() * 2
            + engram_entries
            + attention_stats.launches
            + attention_stats.attention_launches,
        engram_entries,
        total_device_us,
        total_device_us as f64 / 1000.0,
        total_staging_ms,
        total_host_ms,
        chain_elapsed.as_millis(),
        total_host_bytes,
        total_matrix_bytes,
        total_cubin_bytes,
        total_device_buffer_bytes
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
#[cfg(any(feature = "cuda", test))]
fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
        .collect()
}

/// The first element where a BF16 readback is not the BF16 write of the twin.
#[cfg(feature = "cuda")]
fn bf16_write_difference(actual: &[u8], twin: &[f32]) -> Option<(usize, f32, f32)> {
    let expected = sparkflash_engine::forward::layer_zero::bf16_bytes(twin);
    for index in 0..actual.len() / 2 {
        let actual_chunk = &actual[index * 2..index * 2 + 2];
        if actual_chunk != &expected[index * 2..index * 2 + 2] {
            let device = f32::from_bits(
                u32::from(u16::from_le_bytes([actual_chunk[0], actual_chunk[1]])) << 16,
            );
            return Some((index, device, twin[index]));
        }
    }
    None
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
        let max_abs = max_abs_delta(values, &trace.residual_out);
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
        } else if over_bar(inv_rms_delta, M11_INV_RMS_BAR) {
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
    fn the_deferred_entry_input_is_the_previous_x_not_the_ffn_output() {
        let config = sparkflash_engine::config::ModelConfig::tiny();
        let handoff = sparkflash_engine::forward::DeferredHandoff {
            x: vec![1.0, 2.0],
            y: vec![9.0, 10.0],
            mixes: sparkflash_engine::forward::mhc::Mixes::for_config(&config),
        };
        assert_eq!(deferred_entry_input(&handoff), [1.0, 2.0]);
    }

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
        assert!(text.contains("10 baseline pins parsed"));
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

    #[cfg(feature = "cuda")]
    #[test]
    fn the_released_post_predictor_reproduces_the_kernel_coefficients_and_the_kernel_rows_do_not() {
        use sparkflash_engine::config::ModelConfig;
        use sparkflash_engine::forward::mhc::{self, Mixes};
        use sparkflash_kernels::mega_mhc_ref::{
            HC_MULT, HIDDEN, Streams, h_of_with_scale, released_predictor_into_kernel_rows,
        };

        let mut config = ModelConfig::tiny();
        config.dim = HIDDEN;
        config.hc_mult = HC_MULT;
        config.hc_sinkhorn_iters = 3;
        config.hc_eps = 1e-6;
        config.norm_eps = 1e-6;
        let width = config.hc_width();
        let released_fn: Vec<f32> = (0..24 * width)
            .map(|i| (((i * 13) % 97) as f32 - 48.0) / 1000.0)
            .collect();
        let mut kernel_fn = vec![0.0f32; released_fn.len()];
        released_predictor_into_kernel_rows(&released_fn, &mut kernel_fn);
        let hc_base: [f32; 24] = core::array::from_fn(|i| (i as f32 - 12.0) / 100.0);
        let hc_scale = [0.5f32, 1.0, 0.25];
        let x: Vec<f32> = (0..width)
            .map(|i| (((i * 7) % 101) as f32 - 50.0) / 20.0)
            .collect();

        let mut host = Mixes::for_config(&config);
        mhc::hc_mixes(&config, &x, &released_fn, &hc_scale, &hc_base, &mut host)
            .expect("released-row host twin");
        let mut streams = Streams::new(x.clone());
        let mut registers = streams.capture();
        let kernel_coefficients = h_of_with_scale(&mut registers, &kernel_fn, &hc_base, &hc_scale);
        let mut released_max = 0.0f32;
        for index in 0..HC_MULT {
            released_max = released_max.max((host.pre[index] - kernel_coefficients.a[index]).abs());
            released_max =
                released_max.max((host.post[index] - kernel_coefficients.c[index]).abs());
            for source in 0..HC_MULT {
                released_max = released_max.max(
                    (host.comb[source * HC_MULT + index] - kernel_coefficients.b[index][source])
                        .abs(),
                );
            }
        }
        assert!(
            released_max < 1e-4,
            "the released-row host twin differs from the kernel reference by {released_max:.3e}"
        );

        let mut wrong = Mixes::for_config(&config);
        mhc::hc_mixes(&config, &x, &kernel_fn, &hc_scale, &hc_base, &mut wrong)
            .expect("kernel-row host twin");
        let mut wrong_max = 0.0f32;
        for index in 0..HC_MULT {
            wrong_max = wrong_max.max((wrong.pre[index] - host.pre[index]).abs());
            wrong_max = wrong_max.max((wrong.post[index] - host.post[index]).abs());
        }
        for (wrong, released) in wrong.comb.iter().zip(host.comb.iter()) {
            wrong_max = wrong_max.max((wrong - released).abs());
        }
        assert!(
            wrong_max > 1e-3,
            "feeding kernel-order rows to hc_mixes must differ, got {wrong_max:.3e}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_deferred_step_one_twin_uses_the_input_stream_not_its_own_readback() {
        // b28 reproduced: the Engram-split check fed the device's X_l readback back into the
        // step-1 twin as if it were X_{l-1}. For a non-identity B/C row the update is not a fixed
        // point, so this test pins the input-stream contract and the first-difference diagnostic.
        use sparkflash_engine::forward::layer_zero::{bf16_bytes, kernel_mhc_update_fused};

        let x = vec![0.5f32, 1.5, 2.5, 3.5];
        let mut b = vec![0.0f32; 16];
        for i in 0..4 {
            b[i * 4 + i] = 1.0 + 0.125 * i as f32;
        }
        let c = vec![0.25f32, -0.5, 0.75, -1.0];
        let y = 2.0f32;
        let updated = kernel_mhc_update_fused(&x, &b, &c, &[y]);
        let updated_bytes = bf16_bytes(&updated);
        assert_eq!(
            updated_bytes,
            bf16_bytes(&kernel_mhc_update_fused(&x, &b, &c, &[y]))
        );
        let wrong = kernel_mhc_update_fused(&updated, &b, &c, &[y]);
        assert_ne!(
            updated_bytes,
            bf16_bytes(&wrong),
            "a non-identity B/C row must not treat its output as a fixed point"
        );
        let difference = bf16_write_difference(&bf16_bytes(&wrong), &updated)
            .expect("the wrong input must be visible");
        assert!(difference.0 < x.len());
    }

    #[test]
    fn the_post_host_twin_projects_the_released_rows_not_the_kernel_rows() {
        // Regression for window b27: LayerPostStaging::predictor is permuted into kernel B/A/C
        // row order for upload, while hc_mixes consumes the released pre/post/comb order. Feeding
        // the kernel-order vector to hc_mixes compared the device FFN coefficients against a twin
        // built from the wrong rows (max abs 0.9965 at layer 0).
        let staging = sparkflash_device::LayerPostStaging {
            layer: 0,
            predictor: vec![9.0; 24],
            released_fn: vec![1.0; 24],
            hc_base: [0.0; 24],
            hc_scale: [0.0; 3],
            ffn_norm: Vec::new(),
            hc_base_kernel: [0.0; 24],
            digests: Vec::new(),
            buffer_digests: Vec::new(),
        };
        assert_eq!(post_host_predictor(&staging)[0], 1.0);
        assert_ne!(post_host_predictor(&staging)[0], staging.predictor[0]);
    }

    #[test]
    fn the_deferred_twin_ignores_its_own_readback_and_returns_the_bf16_write() {
        use sparkflash_engine::forward::layer_zero::{bf16_bytes, kernel_mhc_update_fused};
        let input = bf16_bytes(&[0.5, 1.5, 2.5, 3.5]);
        let readback = bf16_bytes(&[9.0, 9.0, 9.0, 9.0]);
        let other_readback = bf16_bytes(&[-4.0, 7.0, 1.0, 0.0]);
        let mut b = vec![0.0f32; 16];
        for i in 0..4 {
            b[i * 4 + i] = 1.0 + 0.125 * i as f32;
        }
        let c = vec![0.25f32, -0.5, 0.75, -1.0];
        let y = [2.0f32];
        let twin = deferred_twin_update(&input, &readback, &b, &c, &y);
        let twin_other = deferred_twin_update(&input, &other_readback, &b, &c, &y);
        assert_eq!(
            sparkflash_engine::forward::layer_zero::bf16_bytes(&twin),
            sparkflash_engine::forward::layer_zero::bf16_bytes(&twin_other),
            "the readback must not be an input to the twin"
        );
        let expected =
            kernel_mhc_update_fused(&decode_bf16(&input), &b, &c, &decode_bf16(&bf16_bytes(&y)));
        assert_eq!(
            sparkflash_engine::forward::layer_zero::bf16_bytes(&twin),
            sparkflash_engine::forward::layer_zero::bf16_bytes(&expected),
            "the twin is the BF16 write of the input stream"
        );
        let from_readback = kernel_mhc_update_fused(
            &decode_bf16(&readback),
            &b,
            &c,
            &decode_bf16(&bf16_bytes(&y)),
        );
        assert_ne!(
            sparkflash_engine::forward::layer_zero::bf16_bytes(&twin),
            sparkflash_engine::forward::layer_zero::bf16_bytes(&from_readback),
            "a non-identity row must distinguish the input from its own readback"
        );
    }

    #[test]
    fn both_m11_bars_reject_non_finite_deltas_and_over_bar_values() {
        use sparkflash_engine::forward::layer_zero::{M11_HC_COEFF_BAR, M11_INV_RMS_BAR};
        assert!(!over_bar(0.0, M11_HC_COEFF_BAR));
        assert!(!over_bar(M11_HC_COEFF_BAR, M11_HC_COEFF_BAR));
        assert!(over_bar(M11_HC_COEFF_BAR * 1.5, M11_HC_COEFF_BAR));
        assert!(over_bar(f32::NAN, M11_HC_COEFF_BAR));
        assert!(over_bar(f32::INFINITY, M11_HC_COEFF_BAR));
        assert!(!over_bar(0.0, M11_INV_RMS_BAR));
        assert!(over_bar(M11_INV_RMS_BAR * 1.5, M11_INV_RMS_BAR));
        assert!(over_bar(f32::NAN, M11_INV_RMS_BAR));
        assert!(over_bar(f32::NEG_INFINITY, M11_INV_RMS_BAR));
    }

    #[test]
    fn missing_body_layers_reports_every_layer_whose_body_did_not_launch_once() {
        assert_eq!(missing_body_layers(&[0, 1, 2], &[1, 0, 2]), vec![1, 2]);
        assert!(missing_body_layers(&[5, 6], &[1, 1]).is_empty());
    }

    /// The chain's two deferred-twin call sites, pinned as source text. A helper-only test did not
    /// catch the b28-class mutation that swaps the device readback into the twin; this test reads
    /// the chain source and requires the exact argument expressions at both call sites.
    #[test]
    fn the_chain_twin_call_sites_feed_the_deferred_input_not_the_device_readback() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/device_chain.rs");
        let source = std::fs::read_to_string(path).expect("the chain source is readable");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("the test module follows the production source");
        let normalized = production.split_whitespace().collect::<Vec<_>>().join(" ");
        for call in [
            "deferred_twin_update(&first_x, &mixer_x, &b, &deferred.mixes.post, &deferred.y)",
            "deferred_twin_update(&x_bytes, &x_after, &b_prev, &c_prev, &y_prev)",
        ] {
            assert!(
                normalized.contains(call),
                "the chain must call {call}; it is not in the source"
            );
        }
        for mutated in [
            "deferred_twin_update(&mixer_x",
            "deferred_twin_update(&x_after",
            "deferred_twin_update(&first_x, &first_x",
        ] {
            assert!(
                !normalized.contains(mutated),
                "the chain must not feed the readback into the twin: found {mutated}"
            );
        }
        assert_eq!(
            normalized.matches("deferred_twin_update(").count(),
            3,
            "one definition plus exactly two chain call sites"
        );
    }

    /// The coefficient fold must go through the finite helper; restoring a plain `f32::max` fold
    /// would silently drop a NaN coefficient delta and is the mutation this pins.
    #[test]
    fn the_chain_coefficient_fold_uses_the_finite_helper() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/device_chain.rs");
        let source = std::fs::read_to_string(path).expect("the chain source is readable");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("the test module follows the production source");
        let normalized = production.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            normalized.contains(".fold(0.0f32, finite_max);"),
            "the coefficient fold must call finite_max"
        );
        assert!(
            !normalized.contains(".fold(0.0f32, f32::max)"),
            "a plain f32::max fold is NaN-blind"
        );
    }

    /// The c4a-2 seam line is obtained from the checked function and printed; deleting the guard
    /// or bypassing the function shows up here as missing source text.
    ///
    /// The literals below duplicate the production call text on purpose. A mutation applied to BOTH
    /// production and this string would false-green, but that is not the realistic edit; a
    /// production-only mutation is red, and the 40-layer exact-line test is the behavioural pin.
    #[test]
    fn the_chain_prints_the_checked_c4a2_seam_line() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/device_chain.rs");
        let source = std::fs::read_to_string(path).expect("the chain source is readable");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("the test module follows the production source");
        let normalized = production.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            normalized.contains("let c4a2_line = c4a2_seam_line(")
                && normalized
                    .contains("&body_layers, &body_counts, body_layers.len() + engram_entries"),
            "the chain must obtain the c4a-2 seam line from the checked function"
        );
        assert!(
            normalized.contains("lines.push(c4a2_line);"),
            "the checked c4a-2 seam line must be printed"
        );
    }

    #[test]
    fn finite_max_maps_non_finite_deltas_to_infinity() {
        assert_eq!(finite_max(0.0, 0.25), 0.25);
        assert_eq!(finite_max(0.5, 0.25), 0.5);
        assert_eq!(finite_max(0.5, f32::NAN), f32::INFINITY);
        assert_eq!(finite_max(0.5, f32::INFINITY), f32::INFINITY);
        assert_eq!(finite_max(0.5, f32::NEG_INFINITY), f32::INFINITY);
        assert_eq!(max_abs_delta(&[1.0, 2.0], &[1.0, 2.5]), 0.5);
        assert_eq!(max_abs_delta(&[1.0, f32::NAN], &[1.0, 2.0]), f32::INFINITY);
    }

    #[test]
    fn the_c4a2_seam_line_is_emitted_only_when_every_body_launched_once() {
        let line = c4a2_seam_line(&[0, 1, 2], &[1, 1, 1], 3).expect("all layers launch once");
        assert!(
            line.contains("3 layers") && line.contains("6 mHC launches (3 entry + 3 post)"),
            "{line}"
        );
        let error = c4a2_seam_line(&[0, 1, 2], &[1, 0, 1], 3).expect_err("layer 1 did not launch");
        assert!(error.contains("[1]"), "{error}");
        assert!(error.contains("withheld"), "{error}");
    }

    #[test]
    fn the_c4a2_line_counts_the_two_engram_re_entries_in_the_mhc_total() {
        let layers: Vec<usize> = (0..40).collect();
        let launches = vec![1; 40];
        let engram_entries = 2;
        let line = c4a2_seam_line(&layers, &launches, layers.len() + engram_entries)
            .expect("all 40 bodies launch once");
        assert_eq!(
            line,
            "c4a-2 seam: format-2 body launched exactly once on all 40 layers; 200 projection + 40 body launches plus the 82 mHC launches (42 entry + 40 post)",
            "the seam line must agree with the adjacent attention total: 200 + 40 + 82 = 322"
        );
        let total = 200 + 40 + 82;
        assert_eq!(total, 322);
    }
}
