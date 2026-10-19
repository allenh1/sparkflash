//! The device-chain subcommand: M3-a prints the plan, M3 wiring dry-runs the dispatch, and the
//! cuda-only M3-b arm runs the layer-0 mHC entry as a single staged stage.
//!
//!   sparkflash device-chain --plan-only
//!   sparkflash device-chain --dry-run
//!   sparkflash device-chain --run-stage mhc_entry --checkpoint DIR --cubin-dir DIR [--positions 0]
//!
//! --plan-only prints the CED schedule, the window accounting, the stage map, the planned host-seam
//! set and the exact design 4.3 labels. --dry-run additionally runs the full 130 + 8 plan through
//! the real ChainDevice wiring with the fake launcher and the host-seam seat delegate, validates the
//! plan-vs-ledger self-check (108,017 ChainOps calls) and prints the per-stage call counts and
//! host-seam seats. Both exit 2: a plan or a fake-launcher dry-run is not an execution. The
//! --run-stage arm exists only in a cuda build: it stages the layer-0 mHC tensors through
//! WeightSource, admits the memory budget before the first device allocation, stages the sixteen
//! mHC buffers and launches the mHC entry through ChainDevice. It refuses every other stage, makes
//! no chain/token/comparison claim, and is not a full chain run. The labels are printed verbatim so
//! a plan cannot be read as a device result (docs/build/device-forward-chain.md 4.3 item 3).

use std::process::ExitCode;

use sparkflash_device::chain::{
    ChainLabels, ChainSchedule, ChainSeamPlan, LAYER_STAGE_ORDER, STEP_STAGE_ORDER,
    TAIL_STAGE_ORDER,
};
use sparkflash_device::chain_device::dry_run;
use sparkflash_device::cubins::{baseline_pins, required_pins};
use sparkflash_device::window::WindowAccounting;

/// The usage line for this subcommand.
pub const USAGE: &str = "sparkflash device-chain --plan-only|--dry-run|--run-stage mhc_entry --checkpoint DIR --cubin-dir DIR [--positions 0] [--layer 0]";

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
struct StageArgs {
    checkpoint: std::path::PathBuf,
    cubin_dir: std::path::PathBuf,
    positions: Vec<usize>,
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
        .ok_or_else(|| "usage: sparkflash device-chain --run-stage mhc_entry --checkpoint DIR --cubin-dir DIR [--positions 0] [--layer 0]".to_string())?
        .to_string();
    if stage != "mhc_entry" {
        let reason = match stage.as_str() {
            "moe_router" | "moe_gate_up" => {
                "NotWired (measured in isolation, not wired into the chain)"
            }
            _ => "NotStaged (M3-b stages only the layer-0 mHC entry)",
        };
        return Err(format!(
            "stage {stage} is {reason}; --run-stage currently exposes only mhc_entry"
        ));
    }
    let mut checkpoint = None;
    let mut cubin_dir = None;
    let mut positions = vec![0usize];
    let mut layer = 0usize;
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
            other => return Err(format!("unknown --run-stage option {other}")),
        }
        index += 2;
    }
    if layer != 0 {
        return Err(format!("--layer {layer}: M3-b stages only layer 0"));
    }
    Ok((
        stage,
        StageArgs {
            checkpoint: checkpoint
                .ok_or_else(|| "--run-stage needs --checkpoint DIR".to_string())?,
            cubin_dir: cubin_dir.ok_or_else(|| "--run-stage needs --cubin-dir DIR".to_string())?,
            positions,
        },
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
}
