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
//! result (docs/archive/build/device-forward-chain.md 4.3 item 3).

use std::collections::HashMap;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use sparkflash_device::chain::{
    ChainLabels, ChainSchedule, ChainSeamPlan, LAYER_STAGE_ORDER, STEP_STAGE_ORDER,
    TAIL_STAGE_ORDER,
};
use sparkflash_device::chain_device::dry_run;
use sparkflash_device::cubins::{baseline_pins, required_pins};
use sparkflash_device::window::WindowAccounting;
use sparkflash_engine::tensor::TensorSpec;
use sparkflash_engine::weights::{WeightError, WeightSource};

/// A read-through cache over the checkpoint source used by the device path.
///
/// The sequence run touches the same per-layer matrices at every position. The released checkpoint
/// is memory-mapped but each weight read still copies the tensor, and the per-position cost of that
/// copy dominates the chain. The cache keeps the exact bytes once and returns a copy of them, so the
/// run does not page the checkpoint 138 times; it is the host half of the M3-d staging-reuse shape
/// (the device side keeps its own persistent buffers). The key is the tensor name, so a read is
/// either the cached bytes or the source's exact bytes.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct CachedWeightSource {
    inner: sparkflash_engine::safetensors::SafeTensorsSource,
    cache: Mutex<HashMap<String, Arc<Vec<u8>>>>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl CachedWeightSource {
    fn new(inner: sparkflash_engine::safetensors::SafeTensorsSource) -> Self {
        Self {
            inner,
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl WeightSource for CachedWeightSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        // The routed-expert weights are the one tensor family a sequence does not revisit often
        // enough to justify an unbounded cache: every position selects different experts, and
        // caching all of them retains the whole expert set in RSS. Read them through the mmap
        // source (the page cache keeps the file-backed pages reclaimable) and cache only the
        // per-layer matrices the positions reuse.
        if name.contains(".ffn.experts.") {
            return self.inner.read(name);
        }
        if let Some(bytes) = self
            .cache
            .lock()
            .map_err(|_| WeightError::Missing {
                name: name.to_owned(),
            })?
            .get(name)
        {
            return Ok((**bytes).clone());
        }
        let bytes = Arc::new(self.inner.read(name)?);
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(name.to_owned(), Arc::clone(&bytes));
        }
        Ok((*bytes).clone())
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        WeightSource::read_ranges(&self.inner, name, ranges)
    }
}

/// The usage line for this subcommand.
pub const USAGE: &str = "sparkflash device-chain --plan-only|--dry-run|--run-stage mhc_entry|layer_zero|hybrid_token|hybrid_sequence|tp4-smoke|tp4-resident|attn-operands --checkpoint DIR --cubin-dir DIR [--positions 0] [--layer 0] [--token 0] [--element-limit N] [--moe-preflight] [--moe-layer0-diagnose] [--fixture FILE --prompt-tokens N --decode-steps N] [--dump-logits FILE] [--timing-json FILE] [--reference-check] [--attention-twin-positions LIST] [--host-attention-from POSITION] [--operand-audit]; tp4-smoke also takes --tp-rank N --tp-world-size N --comm-backend tcp|lo --comm-peers IP,IP,... --comm-port 29601 --evidence-dir DIR";

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
fn missing_body_layers(
    layers: &[usize],
    body_launches: &[usize],
    body_per_layer: usize,
) -> Vec<usize> {
    layers
        .iter()
        .zip(body_launches)
        .filter_map(|(layer, launches)| (*launches != body_per_layer).then_some(*layer))
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
    body_per_layer: usize,
) -> Result<String, String> {
    let missing = missing_body_layers(layers, body_launches, body_per_layer);
    if !missing.is_empty() {
        return Err(format!(
            "c4a-2 seam line withheld: the format-2 body must launch exactly {body_per_layer} time(s) on every layer, but layers {missing:?} do not (body_launches != {body_per_layer}); the per-layer ledger and totals would otherwise disagree"
        ));
    }
    let post_launches = layers.len();
    let mhc_launches = entry_launches + post_launches;
    let total_body: usize = body_launches.iter().sum();
    let launch_word = if body_per_layer == 1 {
        "once".to_string()
    } else {
        format!("{body_per_layer} times")
    };
    Ok(format!(
        "c4a-2 seam: format-2 body launched exactly {launch_word} on all {} layers; {} projection + {} body launches plus the {} mHC launches ({} entry + {} post)",
        layers.len(),
        layers.len() * 5,
        total_body,
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
        "cubin catalog: {} baseline pins parsed from crates/sparkflash-kernels/baseline/kernel_baseline.txt; {} required for the device path (rmsnorm_sm121 is the c4c tail norm; swiglu_mxfp8_quant_sm121 is a device stage in the shared expert and is verified by the CLI backend)",
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
    // The tp4-smoke stage is host-only: it opens no driver, so it must be reachable from a build
    // without the cuda feature. A rank of the four-node window runs the same code a CPU-only host
    // builds, and the host tests exercise it without a device.
    if arg_refs.first().copied() == Some("--run-stage")
        && arg_refs.get(1).copied() == Some("tp4-smoke")
    {
        return crate::tp4_smoke::run(&arg_refs[2..]);
    }
    // The tp4-resident stage is the S5.2 loader plan and read; like tp4-smoke it opens no driver,
    // so a CPU-only build reaches it and the four-node window runs the same code.
    if arg_refs.first().copied() == Some("--run-stage")
        && arg_refs.get(1).copied() == Some("tp4-resident")
    {
        return crate::tp4_resident::run(&arg_refs[2..]);
    }
    // The attn-operands stage is a host-only gate: it builds the rank plan from the released
    // headers and derives every attention projection's device operand layout, so a CPU-only build
    // reaches it and it opens no driver. A pass means the plan offers B12x operands; a stored
    // projection fails by name instead of becoming a per-layer host repack at run time.
    if arg_refs.first().copied() == Some("--run-stage")
        && arg_refs.get(1).copied() == Some(crate::tp4_operands::STAGE)
    {
        return crate::tp4_operands::run(&arg_refs[2..]);
    }
    // The tp4-chain stage is the four-rank device arm: per-rank arena plus the per-layer collective.
    // It is reachable from a CPU-only build for --plan-only and --comm-smoke.
    if arg_refs.first().copied() == Some("--run-stage")
        && arg_refs.get(1).copied() == Some("tp4-chain")
    {
        return crate::tp4_chain::run(&arg_refs[2..]);
    }
    // The tp4-host stage is the S5.2 host TWIN and the four-node output run: it opens no driver and
    // moves weights only through the checkpoint range reader, so a CPU-only build reaches it.
    if arg_refs.first().copied() == Some("--run-stage")
        && arg_refs.get(1).copied() == Some("tp4-host")
    {
        return crate::tp4_host::run(&arg_refs[2..]);
    }
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
    moe_preflight: bool,
    moe_layer0_diagnose: bool,
    fixture: Option<std::path::PathBuf>,
    prompt_tokens: usize,
    decode_steps: usize,
    stop_after_layer: Option<usize>,
    time_budget_s: u64,
    sequence: bool,
    reference_check: bool,
    dump_logits: Option<std::path::PathBuf>,
    // When set, the run records the D42 per-source staging lines (window rows, absolute main-KV
    // rows, indexer key pool, host staging, H2D copies, launches, sync-gated device time) and the
    // per-step chain time, then writes the JSON document here. None means no counter is touched.
    timing_json: Option<std::path::PathBuf>,
    wo_a_mode: crate::attention_device::WoAMode,
    wo_a_operand_dump: Option<std::path::PathBuf>,
    // When set, replaces the in-chain attention twin's armed positions; None keeps the default
    // (prefill 0..2 plus the ring's first two wrap steps).
    attention_twin_positions: Option<Vec<usize>>,
    // When set, positions at or after it skip the device attention body and run the engine's host
    // attention loop instead. The rest of the device chain (MoE, projections) keeps running, so a
    // clean run past the window separates an attention-path defect from an upstream one.
    host_attention_from: Option<usize>,
    // When set, the reference half of the run captures the reference's own attention inputs at the
    // armed twin positions, and the report compares the device's staged operands against them. This
    // is the audit the in-chain twin cannot do: the twin compares device body against host body on
    // the same staged operands, so a staging defect is invisible to it.
    operand_audit: bool,
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
    if stage != "mhc_entry"
        && stage != "layer_zero"
        && stage != "hybrid_token"
        && stage != "hybrid_sequence"
    {
        let reason = match stage.as_str() {
            "moe_router" | "moe_gate_up" => {
                "NotWired (measured in isolation, not wired into the chain)"
            }
            _ => {
                "NotStaged (M3-c stages only the layer-0 mHC entry, the layer-0 hybrid pass, the 40-layer hybrid token and the hybrid sequence)"
            }
        };
        return Err(format!(
            "stage {stage} is {reason}; --run-stage currently exposes only mhc_entry, layer_zero, hybrid_token and hybrid_sequence"
        ));
    }
    let mut checkpoint = None;
    let mut cubin_dir = None;
    let mut positions = vec![0usize];
    let mut layer = 0usize;
    let mut token = 0u32;
    let mut element_limit = None;
    let mut moe_preflight = false;
    let mut moe_layer0_diagnose = false;
    let mut fixture = None;
    let mut prompt_tokens = 1usize;
    let mut decode_steps = 0usize;
    let mut stop_after_layer: Option<usize> = None;
    let mut time_budget_s = 3600u64;
    let mut reference_check = false;
    let mut dump_logits = None;
    let mut timing_json = None;
    let mut wo_a_mode = crate::attention_device::WoAMode::default();
    let mut wo_a_operand_dump = None;
    let mut operand_audit = false;
    let mut attention_twin_positions = None;
    let mut host_attention_from = None;
    let mut index = 1usize;
    while index < args.len() {
        let flag = args[index];
        if flag == "--moe-preflight" {
            moe_preflight = true;
            index += 1;
            continue;
        }
        if flag == "--moe-layer0-diagnose" {
            moe_layer0_diagnose = true;
            index += 1;
            continue;
        }
        if flag == "--reference-check" {
            reference_check = true;
            index += 1;
            continue;
        }
        if flag == "--operand-audit" {
            operand_audit = true;
            index += 1;
            continue;
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag {
            "--checkpoint" => checkpoint = Some(std::path::PathBuf::from(value)),
            "--cubin-dir" => cubin_dir = Some(std::path::PathBuf::from(value)),
            "--positions" => positions = parse_positions(value)?,
            "--attention-twin-positions" => {
                attention_twin_positions = Some(parse_positions(value)?);
            }
            "--host-attention-from" => {
                let mut parsed = parse_positions(value)?;
                if parsed.len() != 1 {
                    return Err("--host-attention-from takes one position, not a list".to_string());
                }
                host_attention_from = parsed.pop();
            }
            "--fixture" => fixture = Some(std::path::PathBuf::from(value)),
            "--wo-a-mode" => {
                wo_a_mode = crate::attention_device::WoAMode::parse(value)?;
            }
            "--dump-wo-a-operand" => {
                if value.is_empty() {
                    return Err("--dump-wo-a-operand needs a non-empty path".to_string());
                }
                wo_a_operand_dump = Some(std::path::PathBuf::from(value));
            }
            "--dump-logits" => {
                if value.is_empty() {
                    return Err("--dump-logits needs a non-empty path".to_string());
                }
                dump_logits = Some(std::path::PathBuf::from(value));
            }
            "--timing-json" => {
                if value.is_empty() {
                    return Err("--timing-json needs a non-empty path".to_string());
                }
                timing_json = Some(std::path::PathBuf::from(value));
            }
            "--prompt-tokens" => {
                prompt_tokens = value
                    .parse::<usize>()
                    .map_err(|_| format!("--prompt-tokens: {value} is not a count"))?;
                if prompt_tokens == 0 {
                    return Err("--prompt-tokens must be at least 1".to_string());
                }
            }
            "--decode-steps" => {
                decode_steps = value
                    .parse::<usize>()
                    .map_err(|_| format!("--decode-steps: {value} is not a count"))?;
            }
            "--stop-after-layer" => {
                stop_after_layer = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("--stop-after-layer: {value} is not a layer"))?,
                );
            }
            "--time-budget-s" => {
                time_budget_s = value
                    .parse::<u64>()
                    .map_err(|_| format!("--time-budget-s: {value} is not seconds"))?;
                if time_budget_s == 0 {
                    return Err("--time-budget-s must be positive".to_string());
                }
            }
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
    let sequence = stage == "hybrid_sequence";
    if stop_after_layer.is_some() && !sequence {
        return Err(
            "--stop-after-layer is wired into hybrid_sequence: the diagnostic layer window only has meaning on a sequence run"
                .to_string(),
        );
    }
    if dump_logits.is_some() && !sequence {
        return Err(
            "--dump-logits is only wired into hybrid_sequence, whose per-position logits the vLLM numeric comparison consumes"
                .to_string(),
        );
    }
    if timing_json.is_some() && !sequence {
        return Err(
            "--timing-json is only wired into hybrid_sequence, whose attention body records the D42 per-source staging lines"
                .to_string(),
        );
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
            moe_preflight,
            moe_layer0_diagnose,
            fixture,
            prompt_tokens,
            decode_steps,
            stop_after_layer,
            time_budget_s,
            reference_check,
            sequence,
            dump_logits,
            timing_json,
            wo_a_mode,
            wo_a_operand_dump,
            attention_twin_positions,
            host_attention_from,
            operand_audit,
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
/// Runs the hybrid-sequence stage from another stage with exactly the behaviour of
/// `--run-stage hybrid_sequence`. World=1 host-side, no collectives; the caller is responsible for
/// labelling its own evidence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_hybrid_sequence_for(
    checkpoint: &std::path::Path,
    cubin_dir: &std::path::Path,
    prompt_tokens: usize,
    decode_steps: usize,
    dump_logits: Option<&std::path::Path>,
    timing_json: Option<&std::path::Path>,
    stop_after_layer: Option<usize>,
) -> Result<Vec<String>, String> {
    let args = StageArgs {
        checkpoint: checkpoint.to_path_buf(),
        cubin_dir: cubin_dir.to_path_buf(),
        positions: vec![0],
        token: 0,
        element_limit: None,
        moe_preflight: false,
        moe_layer0_diagnose: false,
        fixture: None,
        prompt_tokens,
        decode_steps,
        stop_after_layer,
        time_budget_s: 3600,
        sequence: true,
        reference_check: false,
        dump_logits: dump_logits.map(std::path::Path::to_path_buf),
        timing_json: timing_json.map(std::path::Path::to_path_buf),
        wo_a_mode: crate::attention_device::WoAMode::default(),
        wo_a_operand_dump: None,
        attention_twin_positions: None,
        host_attention_from: None,
        operand_audit: false,
    };
    run_hybrid_lines(&args)
}

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
    use sparkflash_engine::forward::attention_case;
    use sparkflash_engine::forward::boundary::BoundaryTable;
    use sparkflash_engine::forward::engram;
    use sparkflash_engine::forward::layer_zero::{
        LayerZeroEntrySource, M11_HC_COEFF_BAR, M11_INV_RMS_BAR, bf16_bytes, kernel_mhc_capture,
        kernel_mhc_update_fused, kernel_mhc_xhat,
    };
    use sparkflash_engine::forward::mhc::{self, Mixes};
    use sparkflash_engine::forward::moe::MoeDeviceBackend;
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
    let sequence = args.sequence;
    let sequence_fixture = if sequence {
        let path = args
            .fixture
            .as_ref()
            .ok_or_else(|| "hybrid_sequence needs --fixture FILE".to_string())?;
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("sequence fixture {}: {error}", path.display()))?;
        let fixture: sparkflash_engine::SequenceFixture = serde_json::from_str(&text)
            .map_err(|error| format!("sequence fixture {}: {error}", path.display()))?;
        fixture
            .validate()
            .map_err(|error| format!("sequence fixture {}: {error}", path.display()))?;
        if args.prompt_tokens > fixture.prompt_tokens.len() {
            return Err(format!(
                "--prompt-tokens {} exceeds the fixture's {} prompt tokens",
                args.prompt_tokens,
                fixture.prompt_tokens.len()
            ));
        }
        if args.decode_steps > fixture.decode_steps {
            return Err(format!(
                "--decode-steps {} exceeds the fixture's {} decode steps",
                args.decode_steps, fixture.decode_steps
            ));
        }
        Some(fixture)
    } else {
        None
    };
    let prompt_ids: Vec<u32> = match &sequence_fixture {
        Some(fixture) => fixture.prompt_tokens[..args.prompt_tokens].to_vec(),
        None => vec![args.token],
    };
    let started = Instant::now();
    let mut lines = vec![
        "sparkflash device-chain: M3-c c4c HYBRID TOKEN - 40 layers, one prompt token, one position".to_string(),
        "device entry path: layer 0 keeps the step-1 identity (B=I, C=0, Y=0, A=initial pre_mix); layers 1..39 consume the previous-layer deferred FFN epilogue: B/C/A read back then re-uploaded (declared coefficient round-trip); Y is the host FFN output, BF16-staged; the add runs in device step 1.".to_string(),
        "device post path: each layer runs a device MhcPost after attention. It writes the post-attention residual X and the 24 FFN coefficients, and captures the normalized FFN input; the host FFN consumes those device values.".to_string(),
        "ffn_epilogue split: layers 0..38 move the FFN epilogue add into the next device MhcEntry step-1; layer 39 keeps the host FFN epilogue add because there is no next entry. Count: 39 deferred, 1 host.".to_string(),
        "Engram seam: the released config puts Engram at layers 1 and 14. Engram adds to H_l after the deferred step-1 and before the attention entry, which one fused entry call cannot host-insert; those two layers run the real B/C/Y mixer device, apply Engram on the host readback, then run a second identity device entry. That is 2 extra MhcEntry launches and a declared host Engram seam, not a device Engram.".to_string(),
        "residual staging seam: each layer stages the f32 residual to BF16 and the host continues from the device-readback X; the chain is therefore not bit-identical to the reference even with exact host arithmetic".to_string(),
        "attention projection seam: wq_a, wq_b, wkv, wo_a and wo_b are device B12x MXFP8 GEMMs replacing the host f32 projections; q_norm, kv_norm, RoPE, the window/compressor/indexer-scoring host seams, the inverse RoPE and the D8 quantisers remain declared HOST stand-ins.".to_string(),
        "attention body seam (c4a-2): m14 is the real device format-2 flash_mla_csa2_sm121 launch, 64 head blocks of 256 threads, one launch per layer at position 0. Window codes/scales are built by the engine window row helper from the host kv row; the selected compressed list is empty at position 0, and the kernel's indexer stage-1 buffers are uploaded as zeros plus sentinels and their writes are ignored. The 512-candidate / 640-slot regime is NOT exercised at position 0.".to_string(),
        "attention seam ledger: 3 launches/layer for wq_a+wq_b+wkv (120), 1 format-2 attention launch/layer (40) and 2 launches/layer for wo_a+wo_b (80) = 240 attention launches; 42 MhcEntry + 40 MhcPost + 240 attention = 322 chain launches.".to_string(),
        "routed MoE seam (c4c): 17 device launches/layer - 1 moe_router, 8 routed_gate_up (4 ranks x w1/w3), 4 weighted swiglu_mxfp8_quant_weighted, 4 serial b12x_mxfp4_moe combine - 680 launches over 40 layers. The shared expert runs four real device launches/layer (160), and the final RMSNorm is one device launch. The router selected-expert ids are read back once per layer; only the distinct selected experts are staged; the weighted epilogue reads the router's weights buffer in place, so no host weight multiply exists.".to_string(),
        "chain launch ledger: 322 attention/mHC + 680 routed MoE + 160 shared expert + 4 head screen launches + 1 head reduce + 1 final norm = 1168 device launches; the chain token comparison remains the CHAIN-WIRING gate.".to_string(),
        "claim: CHAIN-WIRING parity over c4c device MoE/norm/head, with the token produced by the device lm_head_reduce_sm121 launch; the only declared token-path seam is the 4-byte token plus status readback. Criterion 1 remains OPEN until a full 130 + 8 run meets design 4.3 items 1-6.".to_string(),
        "caching seam: the checkpoint source is a lazy read-through host cache (no 40-layer preload), the five attention weight sets are read and packed once per layer and kept in the attention backend, and the per-layer mHC staging objects are built once and reused across positions. Reused device allocations: the 16-buffer mHC set (2,064,352 B) plus the attention projection buffers.".to_string(),
        "per-layer host stand-ins after c4c: none in the routed or shared expert, and the final RMSNorm is device. The five attention projections and the format-2 attention body are device at position 0. Attention host seams: q_norm, kv_norm, RoPE, window row construction, compressor/cache, selected-list assembly, indexer scoring, inverse RoPE, D8 quantisers. MoE host seams: router selected-id readback/local mapping and selected-expert staging. Tail host seams: the head-weight read/upload and the 4-byte token plus status readback; the four-shard block-max/argmax assembly remains only as the WINNER_CERTIFIED fallback.".to_string(),
        format!(
            "checkpoint {}; cubin dir {}; token {}; position 0; element limit {}",
            args.checkpoint.display(),
            args.cubin_dir.display(),
            args.token,
            limit
        ),
    ];
    if sequence {
        lines[0] = format!(
            "sparkflash device-chain: M3-c c4c HYBRID SEQUENCE - {} prompt positions + {} decode steps through the deferred chain",
            args.prompt_tokens, args.decode_steps
        );
        let fixture_path = args
            .fixture
            .as_ref()
            .map_or_else(|| "none".to_string(), |path| path.display().to_string());
        let fixture_sha = args.fixture.as_ref().map_or_else(
            || "none".to_string(),
            |path| {
                sparkflash_engram::packer::sha256_file(path)
                    .unwrap_or_else(|_| "unreadable".to_string())
            },
        );
        let fixture_bytes = args
            .fixture
            .as_ref()
            .and_then(|path| std::fs::metadata(path).ok())
            .map_or(0, |meta| meta.len());
        lines.push(format!(
            "sequence metadata: binary={} fixture={} fixture_sha256={} fixture_bytes={} checkpoint={} cubin_dir={} prompt_tokens={} decode_steps={} time_budget_s={} argv={:?}",
            std::env::current_exe().map_or_else(|_| "unknown".to_string(), |path| path.display().to_string()),
            fixture_path,
            fixture_sha,
            fixture_bytes,
            args.checkpoint.display(),
            args.cubin_dir.display(),
            args.prompt_tokens,
            args.decode_steps,
            args.time_budget_s,
            std::env::args().collect::<Vec<_>>()
        ));
        let binary_path = std::env::current_exe()
            .map_or_else(|_| "unknown".to_string(), |path| path.display().to_string());
        let binary_sha = std::env::current_exe().ok().map_or_else(
            || "unknown".to_string(),
            |path| {
                sparkflash_engram::packer::sha256_file(&path)
                    .unwrap_or_else(|_| "unreadable".to_string())
            },
        );
        lines.push(format!(
            "sequence metadata binary: path={} sha256={}",
            binary_path, binary_sha
        ));
        if let Ok(entries) = std::fs::read_dir(&args.cubin_dir) {
            let mut cubins: Vec<(String, String)> = entries
                .flatten()
                .filter(|entry| {
                    entry
                        .path()
                        .extension()
                        .is_some_and(|extension| extension == "cubin")
                })
                .map(|entry| {
                    let path = entry.path();
                    let name = path.file_name().map_or_else(
                        || "?".to_string(),
                        |name| name.to_string_lossy().to_string(),
                    );
                    let sha = sparkflash_engram::packer::sha256_file(&path)
                        .unwrap_or_else(|_| "unreadable".to_string());
                    (name, sha)
                })
                .collect();
            cubins.sort();
            for (name, sha) in cubins {
                lines.push(format!("sequence cubin: {name} sha256={sha}"));
            }
        }
        let sequence_utc_start = utc_now();
        let sequence_mem_before = read_mem_available_bytes().unwrap_or(0);
        lines.push(format!(
            "sequence metadata start: utc_start={} mem_available_before={} host={}",
            sequence_utc_start,
            sequence_mem_before,
            host_name()
        ));
    }

    if args.moe_preflight {
        let source = SafeTensorsSource::open(
            &args.checkpoint,
            &args.checkpoint.join("model.safetensors.index.json"),
        )
        .map_err(|error| {
            format!(
                "preflight checkpoint {}: {error}",
                args.checkpoint.display()
            )
        })?;
        let embed_staging = LayerZeroStaging::from_source(&source, &[0])
            .map_err(|error| format!("preflight embed row: {error}"))?;
        let x = embed_staging
            .embed_row(0)
            .ok_or_else(|| "preflight has no embed row 0".to_string())?
            .values_f32();
        let mut backend = crate::moe_device::MoeDeviceBackendImpl::open(&args.cubin_dir)?;
        if args.moe_layer0_diagnose {
            backend = backend.with_layer0_diagnose();
        }
        let mut routed = vec![0.0f32; dim];
        let start = Instant::now();
        let ran = backend
            .ffn(&config, 0, "layers.0", &x, &source, limit, &mut routed)
            .map_err(|error| format!("moe preflight: {error}"))?;
        if !ran {
            return Err("moe preflight: the device backend declined layer 0".to_string());
        }
        let one_layer = start.elapsed();
        let stats = backend.stats();
        let seconds = one_layer.as_secs_f64().max(1e-9);
        let gbs = stats.staged_bytes as f64 / seconds / 1e9;
        let projected_s = one_layer.as_secs_f64() * config.n_layers as f64;
        lines.push(format!(
            "moe preflight: layer=0 tokens=1 launches={} one_layer_ms={:.3} staging_ms={} device_us={} staged_expert_bytes={} host_bytes={} staging_gbs={:.3} projected_full_chain_s={:.1}",
            stats.launches,
            one_layer.as_secs_f64() * 1000.0,
            stats.staging_us / 1000,
            stats.device_us,
            stats.staged_bytes,
            stats.host_bytes,
            gbs,
            projected_s
        ));
        if projected_s > 500.0 {
            lines.push(format!(
                "MOE PREFLIGHT REFUSED: the measured projection {projected_s:.1} s exceeds the 500 s bar; do not run the full window, bring a shard/streaming plan"
            ));
            return Err(lines.join("\n"));
        }
        lines.push("MOE PREFLIGHT PASS: projection under the 500 s bar".to_string());
        return Ok(lines);
    }

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
    let mut reference_tokens = Vec::new();
    let mut reference_logits = Vec::new();
    let mut reference_top5s: Vec<Vec<(u32, f32)>> = Vec::new();
    let mut reference_boundary = None;
    let mut reference_boundaries: Vec<BoundaryTable> = Vec::new();
    let mut reference_attention_records: Vec<Vec<attention_case::AttentionCaseRecord>> = Vec::new();
    let mut reference_head_inputs: Vec<Vec<f32>> = Vec::new();
    // --operand-audit: the reference's own attention inputs at the armed twin positions, keyed by
    // position and captured from the same reference pass that produces the logits comparison, so the
    // audit costs one capture per position and no extra run.
    let mut reference_attention_cases: std::collections::BTreeMap<
        usize,
        Vec<attention_case::AttentionCaseRecord>,
    > = std::collections::BTreeMap::new();
    let reference_elapsed;
    if sequence {
        let full_claim = sequence_fixture.as_ref().is_some_and(|fixture| {
            args.prompt_tokens == fixture.prompt_tokens.len()
                && args.decode_steps == fixture.decode_steps
        });
        if args.reference_check && !full_claim {
            return Err(
                "--reference-check needs the fixture prompt/decode counts so the live reference and the fixture name the same positions"
                    .to_string(),
            );
        }
        if full_claim {
            let audit_positions: Vec<usize> = if args.operand_audit {
                match &args.attention_twin_positions {
                    Some(positions) => positions.clone(),
                    None => crate::attention_device::default_twin_positions(config.window_size),
                }
            } else {
                Vec::new()
            };
            let audit_layers: Vec<usize> = (0..config.n_layers).collect();
            let prefill_position = prompt_ids.len().saturating_sub(1);
            let capture_prefill = audit_positions.contains(&prefill_position);
            if capture_prefill {
                attention_case::begin(&audit_layers, prefill_position).map_err(|error| {
                    format!("reference attention capture {prefill_position}: {error}")
                })?;
            }
            let observed = reference_engine
                .prefill_observed(&mut reference_state, &prompt_ids)
                .map_err(|error| format!("reference prefill: {error}"))?;
            if capture_prefill {
                let captured = attention_case::finish();
                let line = format!(
                    "sequence operand audit: reference capture position={prefill_position} layers={} armed={audit_positions:?}",
                    captured.len()
                );
                println!("{line}");
                lines.push(line);
                reference_attention_cases.insert(prefill_position, captured);
            }
            let mut next = observed.output.next_token;
            reference_tokens.push(next);
            reference_top5s.push(top5(&observed.output.logits));
            reference_logits = observed.output.logits.clone();
            reference_boundaries.push(observed.boundary.clone());
            reference_boundary = Some(observed.boundary);
            for step_index in 0..args.decode_steps {
                let token =
                    next.ok_or_else(|| "reference sequence has no token to decode".to_string())?;
                let position = prompt_ids.len() + step_index;
                let capture = audit_positions.contains(&position);
                if capture {
                    attention_case::begin(&audit_layers, position).map_err(|error| {
                        format!("reference attention capture {position}: {error}")
                    })?;
                }
                let decoded = reference_engine
                    .decode(&mut reference_state, token)
                    .map_err(|error| format!("reference decode: {error}"))?;
                if capture {
                    let captured = attention_case::finish();
                    let line = format!(
                        "sequence operand audit: reference capture position={position} layers={}",
                        captured.len()
                    );
                    println!("{line}");
                    lines.push(line);
                    reference_attention_cases.insert(position, captured);
                }
                next = decoded.next_token;
                reference_tokens.push(next);
                reference_logits = decoded.logits;
            }
            let fixture = sequence_fixture
                .as_ref()
                .ok_or_else(|| "sequence fixture disappeared".to_string())?;
            let expected: Vec<Option<u32>> =
                fixture.steps.iter().map(|step| step.next_token).collect();
            let mut all_equal = reference_tokens.len() == expected.len();
            for (index, (reference, fixture_token)) in
                reference_tokens.iter().zip(expected.iter()).enumerate()
            {
                let position = prompt_ids.len().saturating_sub(1) + index;
                let equal = reference == fixture_token;
                all_equal &= equal;
                let line = format!(
                    "reference vs fixture: position={position} reference_token={} fixture_token={} equal={equal}",
                    token_text(*reference),
                    token_text(*fixture_token)
                );
                println!("{line}");
                lines.push(line);
            }
            let summary = format!(
                "reference vs fixture: positions={} all_equal={all_equal}",
                reference_tokens.len()
            );
            println!("{summary}");
            lines.push(summary);
            if !all_equal {
                return Err(format!(
                    "reference engine does not reproduce the fixture tokens: got {:?}, fixture {:?}; the fixture is stale or from another checkpoint",
                    reference_tokens, expected
                ));
            }
        } else {
            // Sanity prefix: compare every prompt position against a reference run of that prefix.
            let capture_layers: Vec<usize> = (0..config.n_layers).collect();
            for position in 0..prompt_ids.len() {
                let mut prefix_state = SequenceState::new(&config);
                attention_case::begin(&capture_layers, position)
                    .map_err(|error| format!("reference attention capture {position}: {error}"))?;
                let observed = reference_engine
                    .prefill_observed(&mut prefix_state, &prompt_ids[..=position])
                    .map_err(|error| format!("reference sanity prefix {position}: {error}"))?;
                let captured = attention_case::finish();
                reference_attention_records.push(captured);
                reference_tokens.push(observed.output.next_token);
                reference_top5s.push(top5(&observed.output.logits));
                reference_head_inputs.push(observed.output.head_input.clone());
                reference_logits = observed.output.logits;
                reference_boundaries.push(observed.boundary.clone());
                reference_boundary = Some(observed.boundary);
            }
        }
        reference_elapsed = started.elapsed();
        lines.push(format!(
            "reference engine: sequence positions={} decoded={} tokens={:?}; wall_ms={}",
            args.prompt_tokens,
            args.decode_steps,
            reference_tokens,
            reference_elapsed.as_millis()
        ));
    } else {
        let observed = reference_engine
            .prefill_observed(&mut reference_state, &[args.token])
            .map_err(|error| format!("reference prefill: {error}"))?;
        reference_elapsed = started.elapsed();
        reference_tokens.push(observed.output.next_token);
        reference_top5s.push(top5(&observed.output.logits));
        reference_logits = observed.output.logits;
        reference_boundary = Some(observed.boundary);
        if let Some(stop) = args.stop_after_layer {
            lines.push(format!(
                        "DIAGNOSTIC partial model: --stop-after-layer {stop} ran layers 0..={stop} of {}; these logits are NOT the full checkpoint's and must not be compared as parity",
                        config.n_layers
                    ));
        }
        let reference_token = reference_tokens[0];
        let reference_top5 = top5(&reference_logits);
        lines.push(format!(
            "reference engine: token_id={}; wall_ms={}; top5={}",
            token_text(reference_token),
            reference_elapsed.as_millis(),
            format_top5(&reference_top5)
        ));
    }
    drop(reference_engine);
    drop(reference_state);
    if sequence && args.reference_check {
        return Ok(lines);
    }

    // The embedding row is paged, not the whole 661M-element table.
    let embed_source = SafeTensorsSource::open(
        &args.checkpoint,
        &args.checkpoint.join("model.safetensors.index.json"),
    )
    .map_err(|error| format!("embed checkpoint {}: {error}", args.checkpoint.display()))?;
    let mut embed_ids: Vec<usize> = prompt_ids.iter().map(|token| *token as usize).collect();
    if sequence {
        embed_ids.extend(
            reference_tokens
                .iter()
                .flatten()
                .map(|token| *token as usize),
        );
        embed_ids.sort_unstable();
        embed_ids.dedup();
    }
    let embed_staging = LayerZeroStaging::from_source(&embed_source, &embed_ids)
        .map_err(|error| format!("embed row stage: {error}"))?;
    let embedding = if sequence {
        Vec::new()
    } else {
        embed_staging
            .embed_row(args.token as usize)
            .ok_or_else(|| format!("no staged embed row for input token {}", args.token))?
            .values_f32()
    };
    let engram_hashes = Rc::new(RefCell::new(Vec::new()));
    if !sequence {
        engram::ngram_hashes(&config, &[args.token], 0, &mut engram_hashes.borrow_mut())
            .map_err(|error| format!("engram token hashes: {error}"))?;
    }

    let mem_available_bytes = read_mem_available_bytes()?;
    // The recorder exists only when --timing-json was passed. A disabled recorder is never
    // constructed on the default path, so the default run allocates nothing and records nothing.
    let timing = args
        .timing_json
        .as_ref()
        .map(|_| crate::chain_timing::share(crate::chain_timing::Recorder::enabled()));
    let mut attention_backend =
        crate::attention_device::AttentionDeviceBackend::open(&args.cubin_dir)?
            .with_wo_a_mode(args.wo_a_mode);
    if let Some(recorder) = &timing {
        attention_backend = attention_backend.with_timing(std::rc::Rc::clone(recorder));
    }
    if args.moe_layer0_diagnose || args.wo_a_operand_dump.is_some() {
        attention_backend = attention_backend.with_wo_a_native_diagnose();
    }
    if let Some(dir) = args.wo_a_operand_dump.clone() {
        attention_backend = attention_backend.with_wo_a_operand_dump(dir);
    }
    lines.push(format!(
        "wo_a path: mode={} (native = BF16 activation, no activation quantisation; mxfp8 = E4M3 activation with per-32 ue8m0 scales); operand_dump={}",
        args.wo_a_mode.name(),
        args.wo_a_operand_dump
            .as_ref()
            .map_or_else(|| "none".to_string(), |path| path.display().to_string())
    ));
    lines.push(
        "indexer arithmetic: native dtype (bf16 q_norm output quantised to E4M3 over 32-value blocks, fp8 W8A8 wq_b with a bf16 output, bf16 weights and scores, discrete two-level top-k; the wq_b projection is the same device B12x MXFP8 GEMM in row chunks, while indexer scoring, weights and the two-level top-k stay host seams, and at position 0 the compressed set is empty so no projection launch runs)"
            .to_string(),
    );
    crate::attention_device::AttentionDeviceBackend::open(&args.cubin_dir)?;
    // The in-chain attention twin's armed positions. The default keeps the prefill diagnostics at
    // 0..2 and adds the ring's first two wrap steps, so the boundary is compared without an extra
    // flag; --attention-twin-positions replaces the set for a targeted window ladder.
    let twin_positions = match &args.attention_twin_positions {
        Some(positions) => positions.clone(),
        None => crate::attention_device::default_twin_positions(config.window_size),
    };
    attention_backend.set_twin_positions(twin_positions.clone());
    if let Some(from) = args.host_attention_from {
        // An A/B switch, not a production path: every position at or after `from` skips the device
        // attention body and runs the engine's host attention loop on the same q/kv/ids. The rest
        // of the device chain still runs, so a clean result past the window says the defect is in
        // the device attention path (staging or body), and a dirty one says it is upstream.
        attention_backend.set_host_attention_from(from);
    }
    if args.sequence {
        if let Some(from) = args.host_attention_from {
            lines.push(format!(
                "sequence host-attention A/B: positions {from}.. run the host attention loop; the device attention body is skipped from there on (expect the position-support line below to report the shortfall)"
            ));
        }
        lines.push(format!(
            "sequence attention twin armed: positions={:?}; default is prefill 0..2 plus window and window+1 (the ring's first two wrap steps); per-layer comparisons print as 'sequence attention twin: position=...', the eviction contract as 'sequence attention boundary: position=...'",
            twin_positions
        ));
    }
    lines.push(
        "host quantiser seams: bf16 pre-rounds (the reference rounds the module output before every dynamic quantiser: the W8A8 activation staging, q_norm, kv_norm/window rope, the compressor latent, the compressed-cache rope and the indexer k/q rope all narrow to bf16 first; the ffn_norm activation seams stay device-side)"
            .to_string(),
    );
    lines.push(
        "body P*V: the host twin rounds every softmax probability to bf16 before the value product (native kernel.py:339-341); the device body keeps f32 until the F2 variant lands (declared difference)"
            .to_string(),
    );
    let mut moe_backend = crate::moe_device::MoeDeviceBackendImpl::open(&args.cubin_dir)?;
    if args.moe_layer0_diagnose {
        moe_backend = moe_backend.with_layer0_diagnose();
    }
    let launcher = Rc::new(RefCell::new(CudaLauncher::open_for_stage(
        &args.cubin_dir,
        "mega_mhc_sm121",
    )?));
    let cubin_bytes = launcher.borrow().cubin_bytes();
    let provider_source = CachedWeightSource::new(
        SafeTensorsSource::open(
            &args.checkpoint,
            &args.checkpoint.join("model.safetensors.index.json"),
        )
        .map_err(|error| format!("checkpoint source {}: {error}", args.checkpoint.display()))?,
    );
    // Opt-in resident attention projections: the five per-layer projection weights (about 6 GB of
    // E4M3 codes on one node) read, B12x-tiled and uploaded once, then bound by pointer with zero
    // weight H2D per token. Routed experts and Engram stay staged on one node; the four-rank plan is
    // what makes them resident, so the evidence reports the two H2D classes separately.
    let resident_cuda: Option<sparkflash_kernels::cuda_driver::Cuda> =
        if std::env::var("SPARKFLASH_RESIDENT_ATTENTION")
            .ok()
            .as_deref()
            == Some("1")
        {
            Some(
                sparkflash_kernels::cuda_driver::Cuda::load()
                    .map_err(|error| format!("resident attention driver: {error}"))?,
            )
        } else {
            None
        };
    let mut _resident_attention_allocator: Option<
        sparkflash_device::arena::CudaArenaAllocator<'_>,
    > = None;
    if let Some(cuda) = resident_cuda.as_ref() {
        // UMA-coherent managed memory on the single node: the arena is filled once and both sides
        // see it; the projection weight H2D counter stays 0 by construction.
        let mut allocator = sparkflash_device::arena::CudaArenaAllocator::with_mode(
            cuda,
            sparkflash_device::arena::AllocationMode::Unified,
        );
        let resident = crate::resident_weights::ResidentAttention::build(
            &mut allocator,
            &provider_source,
            &config,
        )?;
        lines.push(format!(
            "resident attention: {} tensors, {} bytes ({} GiB) uploaded once; projection weight H2D per token is zero, activation H2D is reported separately",
            resident.tensors(),
            resident.bytes(),
            resident.bytes() as f64 / (1u64 << 30) as f64
        ));
        attention_backend.set_resident_attention(resident);
        _resident_attention_allocator = Some(allocator);
    }

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
        moe_launches: usize,
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
        moe_launches: 0,
    };
    let device_timings = Rc::new(RefCell::new(vec![blank_timing; config.n_layers]));
    let post_staging_slot: Rc<RefCell<Option<LayerPostStaging>>> = Rc::new(RefCell::new(None));
    let mut staged_first = false;
    let device_total_us = Rc::new(Cell::new(0u128));
    // The per-layer staging objects are built once and reused across sequence positions. They are
    // small (the matrices that dominate staging live in the attention/MoE backends), but rebuilding
    // them per token was still a per-position checkpoint read.
    let mut layer_staging_cache: Vec<Option<LayerStaging>> = vec![None; config.n_layers];
    let mut post_staging_cache: Vec<Option<LayerPostStaging>> = vec![None; config.n_layers];
    let progress_lines: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
    let mut provider = |layer: usize,
                        hidden: &[f32],
                        pre_mix: &[f32],
                        deferred: Option<&DeferredHandoff>|
     -> Result<HybridEntry, sparkflash_engine::linalg::EngineError> {
        let staging_started = Instant::now();
        let staging = if let Some(staging) = &layer_staging_cache[layer] {
            staging.clone()
        } else {
            let staging = LayerStaging::from_source(&provider_source, layer, &[]).map_err(
                |error: StagingError| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} staging"),
                    detail: error.to_string(),
                },
            )?;
            layer_staging_cache[layer] = Some(staging.clone());
            staging
        };
        let post_staging = if let Some(staging) = &post_staging_cache[layer] {
            staging.clone()
        } else {
            let staging = LayerPostStaging::from_source(&provider_source, layer).map_err(
                |error: StagingError| sparkflash_engine::linalg::EngineError::Shape {
                    what: format!("layer {layer} post staging"),
                    detail: error.to_string(),
                },
            )?;
            post_staging_cache[layer] = Some(staging.clone());
            staging
        };
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
                    &engram_hashes.borrow(),
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
                    &engram_hashes.borrow(),
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
    let chain_source = CachedWeightSource::new(
        SafeTensorsSource::open(
            &args.checkpoint,
            &args.checkpoint.join("model.safetensors.index.json"),
        )
        .map_err(|error| format!("chain checkpoint {}: {error}", args.checkpoint.display()))?,
    );
    let chain_engine = InferenceEngine::new(config.clone(), Box::new(chain_source), limit)
        .map_err(|error| format!("chain engine: {error}"))?;
    let mut sequence_outputs = Vec::new();
    let chain_result = if sequence {
        let mut state = SequenceState::new(&config);
        let mut chain_tokens: Vec<Option<u32>> = Vec::new();
        let mut sequence_abort: Option<String> = None;
        let mut run_token = |token: u32,
                             position: usize,
                             history: &[u32],
                             state: &mut SequenceState,
                             outputs: &mut Vec<sparkflash_engine::forward::HybridTokenOutput>,
                             tokens: &mut Vec<Option<u32>>,
                             lines: &mut Vec<String>|
         -> Result<(), sparkflash_engine::linalg::EngineError> {
            if started.elapsed().as_secs() > args.time_budget_s {
                let line = format!(
                    "sequence time budget EXCEEDED after {} of {} positions: elapsed_s={} budget_s={}",
                    position,
                    args.prompt_tokens + args.decode_steps,
                    started.elapsed().as_secs(),
                    args.time_budget_s
                );
                println!("{line}");
                lines.push(line);
                return Err(sparkflash_engine::linalg::EngineError::Shape {
                    what: "sequence time budget".to_string(),
                    detail: format!(
                        "elapsed {} s exceeds --time-budget-s {}",
                        started.elapsed().as_secs(),
                        args.time_budget_s
                    ),
                });
            }
            let available = read_mem_available_bytes().map_err(|error| {
                sparkflash_engine::linalg::EngineError::Shape {
                    what: "sequence memory read".to_string(),
                    detail: error,
                }
            })?;
            if available < sparkflash_device::budget::ADMISSION_FLOOR_BYTES + 512 * 1024 * 1024 {
                let line = format!(
                    "sequence memory guard ABORT before position {position}: mem_available={available} B is within 512 MiB of the {} B admission floor",
                    sparkflash_device::budget::ADMISSION_FLOOR_BYTES
                );
                println!("{line}");
                lines.push(line.clone());
                return Err(sparkflash_engine::linalg::EngineError::Shape {
                    what: "sequence memory guard".to_string(),
                    detail: line,
                });
            }
            let mut hashes = Vec::new();
            engram::ngram_hashes(&config, history, position, &mut hashes).map_err(|error| {
                sparkflash_engine::linalg::EngineError::Shape {
                    what: "sequence Engram hashes".to_string(),
                    detail: error.to_string(),
                }
            })?;
            *engram_hashes.borrow_mut() = hashes;
            let embedding = match embed_staging.embed_row(token as usize) {
                Some(row) => row.values_f32(),
                None => {
                    sparkflash_device::staging::embed_row_from_source(&embed_source, token as usize)
                        .map_err(|error| sparkflash_engine::linalg::EngineError::Shape {
                            what: "sequence embedding".to_string(),
                            detail: format!(
                                "no staged or readable embed row for token {token}: {error}"
                            ),
                        })?
                        .values_f32()
                }
            };
            let rss_before_kib = read_rss_kib();
            let position_started = Instant::now();
            let attention_before = attention_backend.stats();
            let moe_before = moe_backend.stats();
            let layer_window = match args.stop_after_layer {
                Some(stop) => {
                    sparkflash_engine::forward::StepLayers::through(stop, config.n_layers)
                }
                None => sparkflash_engine::forward::StepLayers::all(config.n_layers),
            };
            let output = chain_engine.run_hybrid_sequence_step_ranged(
                state,
                token,
                &embedding,
                position,
                &mut provider,
                &mut post,
                Some(&mut attention_backend),
                Some(&mut moe_backend),
                layer_window,
                &mut progress,
            )?;
            if let Some(recorder) = &timing {
                recorder
                    .borrow_mut()
                    .record_step(crate::chain_timing::StepSample {
                        position: position as u64,
                        total_ns: position_started.elapsed().as_nanos() as u64,
                    });
            }
            tokens.push(output.token);
            // Only the diagnostic prefix and the two positions the criterion compares need the
            // per-layer/head buffers. Keeping the other 130+ outputs whole is the runner half of
            // the per-position RSS growth; the logits stay because the token table prints top5.
            let keep_full = position < 3
                || position + 1 == args.prompt_tokens
                || position + 1 == args.prompt_tokens + args.decode_steps;
            let mut output = output;
            if !keep_full {
                output.layers.clear();
                output.tail_stages.clear();
                output.head_input.clear();
                output.certified_mask = None;
            }
            outputs.push(output);
            let attention_after = attention_backend.stats();
            let moe_after = moe_backend.stats();
            let mem_after = read_mem_available_bytes().unwrap_or(0);
            let rss_after_kib = read_rss_kib();
            let mem_line = format!(
                "sequence memory: position={position} mem_available_before={available} mem_available_after={mem_after} mem_delta_bytes={} rss_kib_before={rss_before_kib} rss_kib_after={rss_after_kib} rss_delta_kib={}",
                mem_after as i128 - available as i128,
                rss_after_kib as i128 - rss_before_kib as i128
            );
            println!("{mem_line}");
            lines.push(mem_line);
            let done = position + 1;
            let total_positions = args.prompt_tokens + args.decode_steps;
            let chain_cumulative =
                (started.elapsed().as_secs_f64() - reference_elapsed.as_secs_f64()).max(0.0);
            let full_positions = sequence_fixture
                .as_ref()
                .map(|fixture| fixture.prompt_tokens.len() + fixture.decode_steps)
                .unwrap_or(total_positions);
            let chain_full = chain_cumulative * full_positions as f64 / done as f64;
            let reference_basis = if full_positions == total_positions {
                "measured_full"
            } else {
                "requested_only"
            };
            let projected_total = reference_elapsed.as_secs_f64() + chain_full;
            let timing_line = format!(
                "sequence position: position={} done={}/{} wall_ms={} chain_ms={} projected_total_s={:.1} projected_chain_full_s={:.1} full_positions={} reference_basis={} budget_s={} attention_staging_us={} attention_device_us={} moe_staging_us={} moe_device_us={} routed_staging_us={} routed_device_us={} shared_staging_us={} shared_device_us={} total_staging_us={} total_device_us={} mem_available={}",
                position,
                done,
                total_positions,
                position_started.elapsed().as_millis(),
                started
                    .elapsed()
                    .as_millis()
                    .saturating_sub(reference_elapsed.as_millis()),
                projected_total,
                chain_full,
                full_positions,
                reference_basis,
                args.time_budget_s,
                attention_after
                    .staging_us
                    .saturating_sub(attention_before.staging_us),
                attention_after
                    .device_us
                    .saturating_sub(attention_before.device_us),
                moe_after.staging_us.saturating_sub(moe_before.staging_us),
                moe_after.device_us.saturating_sub(moe_before.device_us),
                moe_after
                    .routed_staging_us
                    .saturating_sub(moe_before.routed_staging_us),
                moe_after
                    .routed_device_us
                    .saturating_sub(moe_before.routed_device_us),
                moe_after
                    .shared_staging_us
                    .saturating_sub(moe_before.shared_staging_us),
                moe_after
                    .shared_device_us
                    .saturating_sub(moe_before.shared_device_us),
                (attention_after.staging_us - attention_before.staging_us)
                    + (moe_after.staging_us - moe_before.staging_us),
                (attention_after.device_us - attention_before.device_us)
                    + (moe_after.device_us - moe_before.device_us),
                read_mem_available_bytes().unwrap_or(0)
            );
            println!("{timing_line}");
            lines.push(timing_line);
            if position == 0
                || (position + 1).is_multiple_of(8)
                || position + 1 == args.prompt_tokens + args.decode_steps
            {
                let line = format!(
                    "sequence progress: position={}/{} token={:?} elapsed_s={} mem_available={} moe_device_us={} routed_device_us={} shared_device_us={} attention_body_launches={} attention_body_us={}",
                    position + 1,
                    args.prompt_tokens + args.decode_steps,
                    tokens.last().copied().flatten(),
                    started.elapsed().as_secs(),
                    read_mem_available_bytes().unwrap_or(0),
                    moe_backend.stats().device_us,
                    moe_backend.stats().routed_device_us,
                    moe_backend.stats().shared_device_us,
                    attention_backend.stats().attention_launches,
                    attention_backend.stats().attention_us
                );
                println!("{line}");
                lines.push(line);
            }
            Ok(())
        };
        // Warmup steps are real steps whose cost is reported separately: the cold first steps
        // (module load, first-touch pages, plan upload) must not be averaged into steady state.
        // SPARKFLASH_WARMUP_STEPS=1 on the device arm separates step 0 from steps 1..N in the ledger.
        let warmup_steps: usize = std::env::var("SPARKFLASH_WARMUP_STEPS")
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(0);
        for (position, token) in prompt_ids.iter().copied().enumerate() {
            sparkflash_kernels::cuda_driver::stage_timing_set_warm(false);
            sparkflash_kernels::cuda_driver::stage_timing_scope("prefill", 0, 0, position);
            let history = &prompt_ids[..=position];
            if let Err(error) = run_token(
                token,
                position,
                history,
                &mut state,
                &mut sequence_outputs,
                &mut chain_tokens,
                &mut lines,
            ) {
                sequence_abort = Some(error.to_string());
                break;
            }
        }
        if sequence_abort.is_none() {
            let mut decode_token = reference_tokens[0].unwrap_or(0);
            for step in 0..args.decode_steps {
                let position = prompt_ids.len() + step;
                sparkflash_kernels::cuda_driver::stage_timing_set_warm(step < warmup_steps);
                sparkflash_kernels::cuda_driver::stage_timing_scope("decode", step, 0, position);
                let mut history = prompt_ids.clone();
                history.extend(reference_tokens[..=step].iter().flatten().copied());
                if let Err(error) = run_token(
                    decode_token,
                    position,
                    &history,
                    &mut state,
                    &mut sequence_outputs,
                    &mut chain_tokens,
                    &mut lines,
                ) {
                    sequence_abort = Some(error.to_string());
                    break;
                }
                decode_token = chain_tokens.last().copied().flatten().unwrap_or(0);
            }
        }
        if let Some(error) = sequence_abort {
            Err(sparkflash_engine::linalg::EngineError::Shape {
                what: "hybrid sequence".to_string(),
                detail: error,
            })
        } else {
            for (layer, timing) in device_timings.borrow_mut().iter_mut().enumerate() {
                timing.body_launches = attention_backend.body_launches(layer);
                timing.moe_launches = moe_backend.launches_for(layer);
            }
            sequence_outputs.last().cloned().ok_or_else(|| {
                sparkflash_engine::linalg::EngineError::Shape {
                    what: "hybrid sequence".to_string(),
                    detail: "no sequence output".to_string(),
                }
            })
        }
    } else {
        chain_engine.run_hybrid_chain_deferred(
            args.token,
            &embedding,
            &mut provider,
            &mut post,
            Some(&mut attention_backend),
            Some(&mut moe_backend),
            &mut progress,
        )
    };
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
        timing.moe_launches = moe_backend.launches_for(layer);
    }
    let total_elapsed = started.elapsed();
    let chain_elapsed = total_elapsed.saturating_sub(reference_elapsed);
    lines.extend(progress_lines.borrow().iter().cloned());
    let attention_stats = attention_backend.stats();
    let moe_stats = moe_backend.stats();
    let timings = device_timings.borrow();
    let attention_device_us = attention_stats.device_us as u128;
    let attention_body_us = attention_stats.attention_us as u128;
    let attention_staging_ms = (attention_stats.staging_us / 1000) as u128;
    let total_device_us: u128 = timings
        .iter()
        .map(|timing| timing.entry_us + timing.post_us)
        .sum::<u128>()
        + attention_device_us
        + attention_body_us
        + moe_stats.device_us as u128;
    let total_staging_ms: u128 = timings.iter().map(|timing| timing.staging_ms).sum::<u128>()
        + attention_staging_ms
        + (moe_stats.staging_us / 1000) as u128;
    let total_host_ms: u64 = output.layers.iter().map(|trace| trace.host_ms).sum();
    let total_host_bytes: u64 = output
        .layers
        .iter()
        .flat_map(|trace| trace.stages.iter())
        .map(|event| event.host_bytes_in.saturating_add(event.host_bytes_out))
        .sum::<u64>()
        + attention_stats.host_bytes
        + moe_stats.host_bytes;
    let total_matrix_bytes: u64 = timings
        .iter()
        .map(|timing| timing.matrix_bytes)
        .sum::<u64>()
        + attention_stats.staged_bytes
        + moe_stats.staged_bytes
        + moe_stats.shared_bytes
        + moe_stats.final_norm_bytes;
    let attention_cubin_bytes: u64 = [
        "b12x_mxfp8_dense_sm121",
        "b12x_mxfp8_wo_a_sm121",
        "flash_mla_csa2_sm121",
    ]
    .iter()
    .filter_map(|name| sparkflash_kernels::baseline::pin(name).ok().flatten())
    .map(|pin| pin.cubin_bytes as u64)
    .sum();
    let moe_cubin_bytes: u64 =
        37_384 + 9_752 + 13_000 + 10_048 + 190_400 + 21_776 + 38_088 + 31_480;
    let total_cubin_bytes = cubin_bytes + attention_cubin_bytes + moe_cubin_bytes;
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
    let sequence_positions = if sequence {
        args.prompt_tokens + args.decode_steps
    } else {
        1
    };
    let c4a2_line = c4a2_seam_line(
        &body_layers,
        &body_counts,
        body_layers.len() + engram_entries,
        sequence_positions,
    )?;
    let moe_missing: Vec<usize> = output
        .layers
        .iter()
        .filter_map(|trace| {
            let count = timings
                .get(trace.layer)
                .map_or(0, |timing| timing.moe_launches);
            (count != 21).then_some(trace.layer)
        })
        .collect();
    if !moe_missing.is_empty() {
        return Err(format!(
            "c4c seam line withheld: the routed MoE plus shared expert must launch exactly 21 times on every layer, but layers {moe_missing:?} do not (moe_launches != 21); the per-layer ledger and totals would otherwise disagree"
        ));
    }
    for (device_index, trace) in output.layers.iter().enumerate() {
        let timing = timings
            .get(device_index)
            .ok_or_else(|| format!("layer {} has no device timing record", trace.layer))?;
        let host_bytes: u64 = trace
            .stages
            .iter()
            .map(|event| event.host_bytes_in.saturating_add(event.host_bytes_out))
            .sum();
        let body_ok = timing.body_launches == sequence_positions;
        let moe_ok = timing.moe_launches == 21;
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
                    "attention" if body_ok => format!(
                        "[device: format-2 flash-MLA, {} launch(es), one per sequence position; host q/window/selected seams]",
                        timing.body_launches
                    ),
                    "attention" => format!(
                        "[HOST STAND-IN: m14 flash-MLA; body_launches={}]",
                        timing.body_launches
                    ),
                    "attention_output_projection" => {
                        "[device: wo_a+wo_b, 2 launches; host inverse-RoPE + D8 quant seams]"
                            .to_string()
                    }
                    "moe_router" if moe_ok => {
                        "[device: moe_router, 1 launch per position; host dispatch/readback seam]"
                            .to_string()
                    },
                    "moe_router" => format!(
                        "[HOST STAND-IN: routed MoE; moe_launches={}]",
                        timing.moe_launches
                    ),
                    "moe_gate_up" if moe_ok => {
                        "[device: routed_gate_up, 8 launches per position (4 ranks x w1/w3); host selected-expert staging]"
                            .to_string()
                    }
                    "moe_swiglu_quant" if timing.moe_launches == 21 => {
                        "[device: weighted swiglu+quant, 4 launches; routing weights stay on device]"
                            .to_string()
                    }
                    "moe_down_combine" if timing.moe_launches == 21 => {
                        "[device: mxfp4_moe combine, 4 serial rank launches]".to_string()
                    }
                    "shared_expert_gate" if timing.moe_launches == 21 => {
                        "[device: b12x_mxfp8_dense, w1+w3, 2 launches; host B12x repack seam]"
                            .to_string()
                    }
                    "shared_expert_swiglu_quant" if timing.moe_launches == 21 => {
                        "[device: swiglu_mxfp8_quant at 2304, 1 launch]".to_string()
                    }
                    "shared_expert_down" if timing.moe_launches == 21 => {
                        "[device: b12x_mxfp8_dense w2, 1 launch; host shared-add seam]"
                            .to_string()
                    }
                    "shared_expert_gate" | "shared_expert_swiglu_quant" | "shared_expert_down" => {
                        "[HOST STAND-IN: shared expert (c4c)]".to_string()
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
            "layer {:02}: staging_ms={} engram_split={} entry_us={} post_us={} device_us={} host_ms={} host_bytes={} staged_matrix_bytes={} x_bytes={} body_launches={} moe_launches={} capture_exact={} codes_exact={} scales_exact={} coeff_max_abs={:.6e} inv_rms_delta={:.6e} post_capture_exact={} post_codes_exact={} post_scales_exact={} post_coeff_max_abs={:.6e} post_inv_rms_delta={:.6e}",
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
            timing.body_launches,
            timing.moe_launches,
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
    lines.push(crate::moe_device::routed_totals_line(&moe_stats));
    lines.push(crate::moe_device::shared_totals_line(&moe_stats));
    lines.push(format!(
        "final norm totals: final_norm_launches={} device_us={} staged_norm_bytes={} bf16_mismatches={} screen=rmsnorm_sm121; host readback into the certified head",
        moe_stats.final_norm_launches,
        moe_stats.final_norm_us,
        moe_stats.final_norm_bytes,
        moe_stats.final_norm_bf16_mismatches
    ));
    lines.push(format!(
        "lm head totals: head_launches={} head_certified={} certified_by_rank={:?} winner_certified={} head_staging_us={} head_device_us={} head_staged_bytes={} reduce_used={} reduce_token={} reduce_status={:#x} reduce_value={:e} reduce_certified_count={} screen=cert_lm_head_sm121 argmax=in-kernel per four rank shards + lm_head_reduce_sm121; host block assembly is the fallback",
        moe_stats.head_launches,
        moe_stats.head_certified,
        moe_stats.head_certified_by_rank,
        moe_stats.head_winner_certified,
        moe_stats.head_staging_us,
        moe_stats.head_device_us,
        moe_stats.head_staged_bytes,
        moe_stats.head_reduce_used,
        moe_stats.head_reduce_token,
        moe_stats.head_reduce_status,
        moe_stats.head_reduce_value,
        moe_stats.head_reduce_certified_count
    ));
    lines.push(format!(
        "moe counts: routed_launches={} shared_launches={} total_launches={} routed_staging_us={} shared_staging_us={} total_staging_us={}",
        moe_stats.routed_launches,
        moe_stats.shared_launches,
        moe_stats.launches,
        moe_stats.routed_staging_us,
        moe_stats.shared_staging_us,
        moe_stats.staging_us
    ));
    let sequence_positions = if sequence {
        prompt_ids.len() + args.decode_steps
    } else {
        1
    };
    let entry_launches = (output.layers.len() + engram_entries) * sequence_positions;
    let post_launches = output.layers.len() * sequence_positions;
    let device_launches = entry_launches
        + post_launches
        + attention_stats.launches
        + attention_stats.attention_launches
        + moe_stats.launches
        + moe_stats.head_launches
        + moe_stats.final_norm_launches;
    lines.push(format!(
        "chain totals: layers={} sequence_positions={} entry_launches={} post_launches={} attention_projection_launches={} attention_body_launches={} routed_moe_launches={} shared_moe_launches={} head_launches={} tail_norm_launches={} moe_launches={} device_launches={} engram_split_layers={} device_us={} device_ms={:.3} staging_ms={} host_ms={} total_chain_ms={} host_bytes={} staged_matrix_bytes={} cubin_bytes={} device_buffer_bytes={} (entry/post/moe/head/norm totals are cumulative over {} positions; host/bytes/timing are the last position only)",
        output.layers.len(),
        sequence_positions,
        entry_launches,
        post_launches,
        attention_stats.launches,
        attention_stats.attention_launches,
        moe_stats.routed_launches,
        moe_stats.shared_launches,
        moe_stats.head_launches,
        moe_stats.final_norm_launches,
        moe_stats.launches,
        device_launches,
        engram_entries,
        total_device_us,
        total_device_us as f64 / 1000.0,
        total_staging_ms,
        total_host_ms,
        chain_elapsed.as_millis(),
        total_host_bytes,
        total_matrix_bytes,
        total_cubin_bytes,
        total_device_buffer_bytes,
        sequence_positions
    ));
    for stage in &output.tail_stages {
        lines.push(format!(
            "tail stage {}: host_bytes_in={} host_bytes_out={} note={}",
            stage.stage, stage.host_bytes_in, stage.host_bytes_out, stage.note
        ));
    }
    if sequence {
        let expected_body = output.layers.len() * sequence_positions;
        let actual_body = attention_stats.attention_launches;
        lines.push(format!(
            "sequence attention position support: body_launches={} expected={} positions={} device_position_0_body={} other_position_bodies={}",
            actual_body,
            expected_body,
            sequence_positions,
            output.layers.len(),
            actual_body.saturating_sub(output.layers.len())
        ));
        if actual_body != expected_body {
            lines.push(
                "sequence attention position support: FAIL - the C4A-2 format-2 body or the csa2 sequence entry did not run on every layer and position; at least one step fell back to the host attention loop"
                    .to_string(),
            );
        }
    }
    if sequence && !reference_attention_cases.is_empty() {
        let audit = crate::attention_device::operand_audit_lines(
            &reference_attention_cases,
            attention_backend.operand_traces(),
            config.window_size,
        );
        for line in &audit {
            println!("{line}");
        }
        lines.extend(audit);
    }
    if sequence && reference_attention_records.len() == prompt_ids.len() {
        lines.extend(sequence_alignment_and_residual_lines(
            &reference_tokens,
            &reference_top5s,
            &reference_boundaries,
            &reference_attention_records,
            &reference_head_inputs,
            moe_backend.router_records(),
            attention_backend.diag_latent_raw(),
            attention_backend.diag_q_raw(),
            attention_backend.diag_wq_b_input(),
            attention_backend.diag_wkv_raw(),
            attention_backend.diag_window_codes(),
            attention_backend.diag_wq_b_kernel(),
            &args.checkpoint,
            limit,
            &prompt_ids,
            &sequence_outputs,
            &config,
            attention_backend.operand_traces(),
        ));
    }
    if !attention_backend.wo_a_comparisons().is_empty() {
        for comparison in attention_backend.wo_a_comparisons() {
            lines.push(format!(
                "wo_a native comparison: mode={} layer={} device_vs_native_mismatches={}/{} device_max_abs={:.9e} mxfp8_codes_equal={}/{} mxfp8_max_abs={:.9e} output_max_abs={:.9e} operand_checksum={:#018x}",
                comparison.mode.name(),
                comparison.layer,
                comparison.device_vs_native.mismatches,
                comparison.device_vs_native.total,
                comparison.device_vs_native.max_abs,
                comparison.mxfp8_vs_native.codes_equal,
                comparison.mxfp8_vs_native.total,
                comparison.mxfp8_vs_native.max_abs,
                comparison.mxfp8_vs_native.output_max_abs,
                comparison.operand_checksum
            ));
        }
    }
    let hybrid_top5 = top5(&output.logits);
    lines.push(format!(
        "hybrid chain: token_id={}; wall_ms={}; top5={}",
        token_text(output.token),
        chain_elapsed.as_millis(),
        format_top5(&hybrid_top5)
    ));
    if sequence {
        let all_tokens: Vec<Option<u32>> = sequence_outputs.iter().map(|out| out.token).collect();
        let full_claim = sequence_fixture.as_ref().is_some_and(|fixture| {
            args.prompt_tokens == fixture.prompt_tokens.len()
                && args.decode_steps == fixture.decode_steps
        });
        let chain_tokens: Vec<Option<u32>> = if full_claim {
            all_tokens[prompt_ids.len() - 1..].to_vec()
        } else {
            all_tokens.clone()
        };
        let rows = sequence_comparison_rows(
            &reference_tokens,
            &chain_tokens,
            prompt_ids.len(),
            full_claim,
        );
        let mut all_match = chain_tokens.len() == reference_tokens.len();
        let first_output = if full_claim { prompt_ids.len() - 1 } else { 0 };
        let fixture_steps: Vec<Option<u32>> = sequence_fixture
            .as_ref()
            .map(|fixture| fixture.steps.iter().map(|step| step.next_token).collect())
            .unwrap_or_default();
        for (row, (position, reference, hybrid, exact)) in rows.iter().enumerate() {
            all_match &= *exact;
            let fixture_index = position.saturating_sub(prompt_ids.len().saturating_sub(1));
            let (fixture_token, fixture_equal) = if full_claim {
                (
                    fixture_steps.get(fixture_index).copied().flatten(),
                    fixture_steps.get(fixture_index).copied().flatten() == *reference,
                )
            } else {
                (None, false)
            };
            lines.push(format!(
                "token comparison: position={} reference_id={} hybrid_id={} fixture_id={} exact_equal={} reference_equals_fixture={}",
                position,
                token_text(*reference),
                token_text(*hybrid),
                token_text(fixture_token),
                exact,
                if full_claim { fixture_equal.to_string() } else { "n/a".to_string() }
            ));
            let output_index = first_output + row;
            if let Some(output) = sequence_outputs.get(output_index) {
                lines.push(format!(
                    "position {} hybrid top5: {}",
                    position,
                    format_top5(&top5(&output.logits))
                ));
            }
            if let Some(reference_top5) = reference_top5s.get(output_index) {
                lines.push(format!(
                    "position {} reference top5: {}",
                    position,
                    format_top5(reference_top5)
                ));
            }
        }
        if let Some(path) = args.dump_logits.as_deref() {
            // The machine-readable companion for the vLLM numeric comparison. It is written as
            // soon as the compared rows exist and before the token-parity gate can return, so a
            // failing parity run still leaves its logits; it changes no arithmetic and prints no
            // line, so the stage's existing output stays byte-identical.
            let _ = crate::logits_dump::write_logits_dump(
                path,
                &rows,
                &sequence_outputs,
                first_output,
                full_claim,
                &fixture_steps,
                prompt_ids.len(),
            )?;
        }
        if chain_tokens.len() != reference_tokens.len() {
            lines.push(format!(
                "sequence comparison count: hybrid={} reference={} (all prefill {} + decode {})",
                chain_tokens.len(),
                reference_tokens.len(),
                args.prompt_tokens,
                args.decode_steps
            ));
        }
        if full_claim {
            let fixture = sequence_fixture
                .as_ref()
                .ok_or_else(|| "sequence fixture disappeared".to_string())?;
            let position129 = sequence_outputs
                .get(prompt_ids.len() - 1)
                .ok_or_else(|| "sequence has no position-129 output".to_string())?;
            let certified = position129.certified_mask.as_ref().ok_or_else(|| {
                "the device head did not provide a certified mask; criterion-1 item 4 cannot be compared"
                    .to_string()
            })?;
            let mut words = vec![0u64; certified.len().div_ceil(64)];
            for (index, bit) in certified.iter().enumerate() {
                if *bit != 0 {
                    words[index / 64] |= 1u64 << (index % 64);
                }
            }
            let mask = sparkflash_engine::sequence_fixture::CertifiedMask::from_words(
                words,
                certified.len(),
            )
            .map_err(|error| format!("certified mask: {error}"))?;
            let screened = sparkflash_engine::ScreenedLogits::new(position129.logits.clone(), mask)
                .map_err(|error| format!("screened logits: {error}"))?;
            let comparison = sparkflash_engine::compare_logits_exact(fixture, 129, &screened, 2e-2)
                .map_err(|error| format!("logits 129: {error}"))?;
            lines.push(format!(
                "logits 129: compared_entries={} mask_exact_entries={} mask_certified_entries={} bit_identical_entries={} mismatches={} max_abs_diff={:.9e} max_abs_index={} argmax_index={} bar=0.02 PASS",
                comparison.compared_entries,
                comparison.mask_exact_entries,
                comparison.mask_certified_entries,
                comparison.bit_identical_entries,
                comparison.mismatches,
                comparison.max_abs_diff,
                comparison.max_abs_index,
                comparison.argmax_index
            ));
            let head_exact = position129.token == reference_tokens[0];
            lines.push(format!(
                "head reduction: reduced_argmax={} expected={} certified={} exact={}",
                token_text(position129.token),
                token_text(reference_tokens[0]),
                certified.iter().filter(|bit| **bit != 0).count(),
                head_exact
            ));
            lines.push(format!(
                "head certified: device_mask_entries={} certified={} token={}",
                certified.len(),
                certified.iter().filter(|bit| **bit != 0).count(),
                token_text(position129.token)
            ));
        }
        if !all_match {
            if let Some(boundary) = reference_boundary.as_ref() {
                lines.extend(first_layer_difference(
                    boundary,
                    &output,
                    prompt_ids.len().saturating_sub(1),
                ));
            }
            lines.push("HYBRID SEQUENCE token parity: FAIL - a position's hybrid token is not the in-process reference token".to_string());
            return Err(lines.join(
                "
",
            ));
        }
        lines.push(format!(
            "HYBRID SEQUENCE token parity: PASS - {} compared positions exact; criterion 1 item 2 {}",
            reference_tokens.len(),
            if full_claim { "satisfied" } else { "sanity prefix, not a claim" }
        ));
        let sequence_tokens = (args.prompt_tokens + args.decode_steps).max(1);
        lines.push(format!(
            "sequence H2D: projection_weight_bytes={} wo_a_weight_bytes={} activation_bytes={} resident_weight_binds={} weight_bytes_per_token={:.1} activation_bytes_per_token={:.1}",
            attention_stats.weight_h2d_bytes,
            attention_stats.wo_a_weight_h2d_bytes,
            attention_stats.activation_h2d_bytes,
            attention_stats.resident_weight_binds,
            attention_stats.weight_h2d_bytes as f64 / sequence_tokens as f64,
            attention_stats.activation_h2d_bytes as f64 / sequence_tokens as f64
        ));
        lines.push(format!(
            "sequence metadata tail: utc_end={} mem_available_after={} total_wall_ms={} exit=0",
            utc_now(),
            read_mem_available_bytes().unwrap_or(0),
            started.elapsed().as_millis()
        ));
        if let (Some(path), Some(recorder)) = (&args.timing_json, &timing) {
            recorder
                .borrow()
                .write(path)
                .map_err(|error| format!("timing file {}: {error}", path.display()))?;
            attach_stage_timing_ledger(path)?;
            if let Some(stop) = args.stop_after_layer {
                lines.push(format!(
                        "DIAGNOSTIC partial model: --stop-after-layer {stop} ran layers 0..={stop} of {}; these logits are NOT the full checkpoint's and must not be compared as parity",
                        config.n_layers
                    ));
            }
            let totals = recorder.borrow().totals();
            lines.push(format!(
                "timing: wrote {} (schema {}; {} layer samples, {} step samples; measured staging bytes={} H2D bytes={} host_staging_ns={} sync_gated_device_ns={} step_total_ns={})",
                path.display(),
                crate::chain_timing::SCHEMA,
                totals.layer_samples,
                totals.step_samples,
                totals.window_bytes + totals.absolute_bytes + totals.indexer_pool_bytes,
                totals.h2d_bytes,
                totals.host_staging_ns,
                totals.sync_gated_device_ns,
                totals.step_total_ns
            ));
        }
        return Ok(lines);
    }
    let reference_token = reference_tokens[0];
    let reference_top5 = top5(&reference_logits);
    let token_match = reference_token == output.token;
    lines.push(format!(
        "token comparison: reference_id={} hybrid_id={} exact_equal={} top5_overlap={}",
        token_text(reference_token),
        token_text(output.token),
        token_match,
        top5_overlap(&reference_top5, &hybrid_top5)
    ));
    if !token_match {
        if let Some(boundary) = reference_boundary.as_ref() {
            lines.extend(first_layer_difference(boundary, &output, 0));
        }
        lines.push("CHAIN-WIRING token parity: FAIL - the hybrid token id is not the in-process reference token id".to_string());
        return Err(lines.join(
            "
",
        ));
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

/// A UTC wall-clock stamp for the sequence metadata, independent of a date library.
#[cfg(feature = "cuda")]
fn utc_now() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map_or_else(
            || {
                format!(
                    "unix:{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |elapsed| elapsed.as_secs())
                )
            },
            |output| String::from_utf8_lossy(&output.stdout).trim().to_string(),
        )
}

/// The node hostname for the sequence metadata.
#[cfg(feature = "cuda")]
fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map_or_else(|_| "unknown".to_string(), |name| name.trim().to_string())
}

/// The positions a sequence comparison reports: the last prefill token plus every decode
/// position for a claim, or every prompt position for a prefix sanity run. A missing hybrid token
/// still gets a row, so a short capture is visible instead of being silently compared.
#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
fn diag_checksum_f32(values: &[f32]) -> u64 {
    values.iter().fold(0u64, |acc, value| {
        acc.rotate_left(5) ^ value.to_bits() as u64
    })
}

#[cfg(feature = "cuda")]
fn diag_checksum_bytes(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0u64, |acc, value| acc.rotate_left(5) ^ *value as u64)
}

#[cfg(feature = "cuda")]
fn diag_ids_text(ids: &[usize], cap: usize) -> String {
    let shown: Vec<String> = ids
        .iter()
        .take(cap)
        .map(|id| {
            if *id == usize::MAX {
                "NO".to_string()
            } else {
                id.to_string()
            }
        })
        .collect();
    if ids.len() > cap {
        format!("[{},...{}]", shown.join(","), ids.len())
    } else {
        format!("[{}]", shown.join(","))
    }
}

#[cfg(feature = "cuda")]
fn diag_stage_compare(device: &[f32], host: &[f32]) -> (usize, f32) {
    let mut mismatches = 0usize;
    let mut max_abs = 0f32;
    for (device, host) in device.iter().zip(host.iter()) {
        let diff = (device - host).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if sparkflash_kernels::m4_quant::bf16_bits(*device)
            != sparkflash_kernels::m4_quant::bf16_bits(*host)
        {
            mismatches += 1;
        }
    }
    (mismatches, max_abs)
}

#[cfg(feature = "cuda")]
fn diag_boundary_row<'a>(
    boundaries: &'a [sparkflash_engine::forward::boundary::BoundaryTable],
    position: usize,
    layer: usize,
    name: &str,
    width: usize,
) -> Option<&'a [f32]> {
    let values = boundaries
        .get(position)?
        .row(layer)?
        .tensor(name)?
        .data()
        .as_f32()?;
    if values.len() == width {
        Some(values)
    } else if width != 0
        && values.len().is_multiple_of(width)
        && (position + 1) * width <= values.len()
    {
        Some(&values[position * width..(position + 1) * width])
    } else {
        None
    }
}

#[cfg(feature = "cuda")]
fn diag_bf16(value: f32) -> f32 {
    f32::from_bits((sparkflash_kernels::m4_quant::bf16_bits(value) as u32) << 16)
}

#[cfg(feature = "cuda")]
fn read_rss_kib() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0)
}

/// The b56 follow-up: the fed-token/prediction alignment table and the per-position, per-layer
/// residual twin. The reference boundary is token-major, so for prefix position + 1 the hybrid
/// position's residual row is position; this slices it before comparing instead of treating the
/// whole multi-position tensor as the hybrid's single row.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn sequence_alignment_and_residual_lines(
    reference_tokens: &[Option<u32>],
    reference_top5s: &[Vec<(u32, f32)>],
    reference_boundaries: &[sparkflash_engine::forward::boundary::BoundaryTable],
    reference_attention_records: &[Vec<
        sparkflash_engine::forward::attention_case::AttentionCaseRecord,
    >],
    reference_head_inputs: &[Vec<f32>],
    router_records: &[crate::moe_device::RouterTwinRecord],
    diag_latent_raw: Option<&[f32]>,
    diag_q_raw: Option<&[f32]>,
    diag_wq_b_input: Option<&[f32]>,
    diag_wkv_raw: Option<&[f32]>,
    diag_window_codes: Option<&[u8]>,
    wq_b_kernel: Option<(usize, usize, f32)>,
    checkpoint: &std::path::Path,
    limit: usize,
    prompt_ids: &[u32],
    sequence_outputs: &[sparkflash_engine::forward::HybridTokenOutput],
    config: &sparkflash_engine::config::ModelConfig,
    operand_traces: &[crate::attention_device::AttentionOperandTrace],
) -> Vec<String> {
    let mut lines = Vec::new();
    let n = prompt_ids.len().min(3);
    for (position, prompt_id) in prompt_ids.iter().take(n).enumerate() {
        if let Some(output) = sequence_outputs.get(position) {
            let top = top5(&output.logits);
            lines.push(format!(
                "alignment: position={} hybrid fed_token={} predicted_top1={} logit_max={:.9} top5=[{}]",
                position,
                prompt_id,
                token_text(output.token),
                top.first().map_or(f32::NAN, |(_, value)| *value),
                format_top5(&top)
            ));
        }
        if let (Some(predicted), Some(top)) = (
            reference_tokens.get(position).copied().flatten(),
            reference_top5s.get(position),
        ) {
            lines.push(format!(
                "alignment: position={} reference fed_token={} predicted_top1={} logit_max={:.9} top5=[{}]",
                position,
                prompt_id,
                predicted,
                top.first().map_or(f32::NAN, |(_, value)| *value),
                format_top5(top)
            ));
        }
    }
    for position in 0..n {
        let Some(output) = sequence_outputs.get(position) else {
            continue;
        };
        let Some(host) = reference_head_inputs.get(position) else {
            lines.push(format!(
                "head-input twin: position={position} no reference head input captured"
            ));
            continue;
        };
        if output.head_input.len() != host.len() {
            lines.push(format!(
                "head-input twin: position={position} device width {} reference width {}",
                output.head_input.len(),
                host.len()
            ));
            continue;
        }
        let (mismatches, max_abs) = diag_stage_compare(&output.head_input, host);
        let blocks = 8usize;
        let mut block_max = vec![0f32; blocks];
        for (index, (device_value, host_value)) in
            output.head_input.iter().zip(host.iter()).enumerate()
        {
            let block = index * blocks / host.len().max(1);
            block_max[block] = block_max[block].max((device_value - host_value).abs());
        }
        let block_text: Vec<String> = block_max
            .iter()
            .map(|value| format!("{value:.6e}"))
            .collect();
        lines.push(format!(
            "head-input twin: position={position} bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e} per_block_max_abs=[{}]",
            host.len(),
            block_text.join(",")
        ));
    }
    let width = config.hc_width();
    if let Ok(source) = sparkflash_engine::safetensors::SafeTensorsSource::open(
        checkpoint,
        &checkpoint.join("model.safetensors.index.json"),
    ) {
        use sparkflash_engine::weights::WeightSource as _;
        let mut first_set_flip: Option<usize> = None;
        for record in router_records {
            let Some(input) = diag_boundary_row(
                reference_boundaries,
                0,
                record.layer,
                sparkflash_engine::forward::boundary::PRE_FFN_HIDDEN,
                config.dim,
            ) else {
                continue;
            };
            let gate_name = format!("layers.{}.ffn.gate.weight", record.layer);
            let bias_name = format!("layers.{}.ffn.gate.bias", record.layer);
            let Ok(gate_raw) = source.read(&gate_name) else {
                continue;
            };
            let Ok(bias_raw) = source.read(&bias_name) else {
                continue;
            };
            let gate = decode_bf16(&gate_raw);
            let bias: Vec<f32> = bias_raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_bits(u32::from_le_bytes(*chunk)))
                .collect();
            let params = sparkflash_kernels::router_ref::RouterParams {
                hidden: config.dim,
                n_routed: config.n_routed_experts,
                top_k: config.n_activated_experts,
                gate_temp: config.gate_temp as f32,
                norm_eps: config.norm_eps as f32,
                scale: config.route_scale as f32,
                norm_topk_prob: config.norm_topk_prob,
            };
            let bf16_input: Vec<f32> = input.iter().map(|value| diag_bf16(*value)).collect();
            let Ok(reference) =
                sparkflash_kernels::router_ref::route_f32(input, &gate, &bias, params)
            else {
                continue;
            };
            let Ok(bf16_control) =
                sparkflash_kernels::router_ref::route_f32(&bf16_input, &gate, &bias, params)
            else {
                continue;
            };
            let mut device_set = record.ids.clone();
            device_set.sort_unstable();
            let mut reference_set = reference.ids.clone();
            reference_set.sort_unstable();
            let mut bf16_set = bf16_control.ids.clone();
            bf16_set.sort_unstable();
            let ordered_equal = record.ids == reference.ids;
            let same_set = device_set == reference_set;
            let bf16_same_set = reference_set == bf16_set;
            let bf16_matches_device_set = device_set == bf16_set;
            let score_delta = record
                .scores
                .iter()
                .zip(reference.scores.iter())
                .map(|(device, host)| (device - host).abs())
                .fold(0.0f32, finite_max);
            if first_set_flip.is_none() && !same_set {
                first_set_flip = Some(record.layer);
            }
            if matches!(record.layer, 0 | 1 | 14) || !same_set {
                lines.push(format!(
                    "router twin: position=0 layer={} device_ids={:?} reference_ids={:?} ordered_equal={} same_set={} scores_max_abs={:.6e} bf16_control_ids={:?} bf16_control_same_set={} bf16_control_matches_device_set={}",
                    record.layer,
                    record.ids,
                    reference.ids,
                    ordered_equal,
                    same_set,
                    score_delta,
                    bf16_control.ids,
                    bf16_same_set,
                    bf16_matches_device_set
                ));
            }
        }
        lines.push(match first_set_flip {
            Some(layer) => format!(
                "router twin: first_set_flip_layer={layer} (ordered-only differences before it are not semantic)"
            ),
            None => "router twin: first_set_flip_layer=none".to_string(),
        });
    }
    // The deferred chain's hybrid layer output is the post-attention X, not the reference's
    // after-FFN residual. Compare the declared stage on each side explicitly.
    for position in 0..n {
        let Some(output) = sequence_outputs.get(position) else {
            continue;
        };
        let Some(first) = output.layers.first() else {
            continue;
        };
        let Some(post) = diag_boundary_row(
            reference_boundaries,
            position,
            0,
            sparkflash_engine::forward::boundary::POST_ATTENTION_HIDDEN,
            width,
        ) else {
            continue;
        };
        if first.post_attention_x.len() == post.len() {
            let (mismatches, max_abs) = diag_stage_compare(&first.post_attention_x, post);
            lines.push(format!(
                "stage twin: position={position} (a) post-attention hidden layer=0 bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                post.len()
            ));
        }
        if let Some(ffn) = diag_boundary_row(
            reference_boundaries,
            position,
            0,
            sparkflash_engine::forward::boundary::FFN_OUT,
            config.dim,
        ) && first.ffn_out.len() == ffn.len()
        {
            let (mismatches, max_abs) = diag_stage_compare(&first.ffn_out, ffn);
            lines.push(format!(
                "stage twin: position={position} (c) host FFN output Y layer=0 bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                ffn.len()
            ));
        }
        if let Some(after_ffn) = diag_boundary_row(
            reference_boundaries,
            position,
            0,
            sparkflash_engine::forward::boundary::RESIDUAL_OUT,
            width,
        ) && let Some(second) = output.layers.get(1)
            && second.entry_input_x.len() == after_ffn.len()
        {
            let (mismatches, max_abs) = diag_stage_compare(&second.entry_input_x, after_ffn);
            lines.push(format!(
                "stage twin: position={position} (d) deferred residual_out(0) vs layer-1 entry bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                after_ffn.len()
            ));
        }
        if output.layers.get(1).is_some()
            && let Some(after_ffn_1) = diag_boundary_row(
                reference_boundaries,
                position,
                1,
                sparkflash_engine::forward::boundary::RESIDUAL_OUT,
                width,
            )
            && let Some(third) = output.layers.get(2)
            && third.entry_input_x.len() == after_ffn_1.len()
        {
            let (mismatches, max_abs) = diag_stage_compare(&third.entry_input_x, after_ffn_1);
            lines.push(format!(
                "stage twin: position={position} (d2) deferred residual_out(1) vs layer-2 entry bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                after_ffn_1.len()
            ));
        }
        if let Some(last) = output.layers.last()
            && let Some(after_ffn) = diag_boundary_row(
                reference_boundaries,
                position,
                last.layer,
                sparkflash_engine::forward::boundary::RESIDUAL_OUT,
                width,
            )
            && last.residual_out.len() == after_ffn.len()
        {
            let (mismatches, max_abs) = diag_stage_compare(&last.residual_out, after_ffn);
            lines.push(format!(
                "stage twin: position={position} (e) layer-39 host epilogue residual bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                after_ffn.len()
            ));
        }
    }
    let mut first: Option<(usize, usize)> = None;
    for position in 0..n {
        let Some(boundary) = reference_boundaries.get(position) else {
            lines.push(format!(
                "residual twin: position={position} no reference boundary captured"
            ));
            continue;
        };
        let Some(output) = sequence_outputs.get(position) else {
            continue;
        };
        for layer in 0..config.n_layers {
            let Some(row) = boundary.row(layer) else {
                continue;
            };
            let reference_name = if layer + 1 < config.n_layers {
                sparkflash_engine::forward::boundary::POST_ATTENTION_HIDDEN
            } else {
                sparkflash_engine::forward::boundary::RESIDUAL_OUT
            };
            let Some(tensor) = row.tensor(reference_name) else {
                continue;
            };
            let Some(values) = tensor.data().as_f32() else {
                continue;
            };
            let expected = if values.len() == width {
                Some(values)
            } else if width != 0
                && values.len().is_multiple_of(width)
                && (position + 1) * width <= values.len()
            {
                Some(&values[position * width..(position + 1) * width])
            } else {
                None
            };
            let Some(expected) = expected else {
                lines.push(format!(
                    "residual twin: position={position} layer={layer} reference tensor has {} values, not rows of {width}",
                    values.len()
                ));
                continue;
            };
            let Some(trace) = output.layers.iter().find(|trace| trace.layer == layer) else {
                continue;
            };
            if trace.residual_out.len() != expected.len() {
                lines.push(format!(
                    "residual twin: position={position} layer={layer} hybrid row has {} values, reference row {}",
                    trace.residual_out.len(),
                    expected.len()
                ));
                continue;
            }
            let mut mismatches = 0usize;
            let mut max_abs = 0f32;
            for (device, host) in trace.residual_out.iter().zip(expected.iter()) {
                let diff = (device - host).abs();
                if diff > max_abs {
                    max_abs = diff;
                }
                if sparkflash_kernels::m4_quant::bf16_bits(*device)
                    != sparkflash_kernels::m4_quant::bf16_bits(*host)
                {
                    mismatches += 1;
                }
            }
            lines.push(format!(
                "residual twin (stage-aligned): position={position} layer={layer} bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                expected.len()
            ));
            if first.is_none() && mismatches > 0 {
                first = Some((position, layer));
            }
        }
    }
    if let Some((position, layer)) = first {
        lines.push(format!(
            "residual first divergence: position={position} layer={layer}; operand comparison follows"
        ));
        let device = operand_traces
            .iter()
            .find(|trace| trace.position == position && trace.layer == layer);
        let host = reference_attention_records
            .get(position)
            .and_then(|records| records.iter().find(|record| record.layer == layer));
        match (device, host) {
            (Some(device), Some(host)) => {
                lines.push(format!(
                    "operand q: device first={:.9} last={:.9} checksum={:016x} | host first={:.9} last={:.9} checksum={:016x}",
                    device.q_first,
                    device.q_last,
                    device.q_checksum,
                    host.q.first().copied().unwrap_or(0.0),
                    host.q.last().copied().unwrap_or(0.0),
                    diag_checksum_f32(&host.q)
                ));
                lines.push(format!(
                    "operand sink: device first={:.9} last={:.9} checksum={:016x} | host first={:.9} last={:.9} checksum={:016x}",
                    device.sink_first,
                    device.sink_last,
                    device.sink_checksum,
                    host.sink.first().copied().unwrap_or(0.0),
                    host.sink.last().copied().unwrap_or(0.0),
                    diag_checksum_f32(&host.sink)
                ));
                lines.push(format!(
                    "operand ids: device window_count={} selected_count={} compressed_count={} ratio={} window={} selected={} | host window_ids={} compressed_ids={} compressed_rows={} ratio={} window_ids={} compressed_ids={}",
                    device.window_ids.len(),
                    device.selected.len(),
                    device.compressed_count,
                    device.ratio,
                    diag_ids_text(&device.window_ids, 6),
                    diag_ids_text(&device.selected, 6),
                    host.window_ids.len(),
                    host.compressed_ids.len(),
                    host.compressed_rows,
                    host.ratio,
                    diag_ids_text(&host.window_ids, 6),
                    diag_ids_text(&host.compressed_ids, 6)
                ));
                let host_first = if host.window_rows == 0 {
                    0
                } else {
                    sparkflash_kernels::cuda_harness::engine_window_row(
                        &host.window_kv[..host.head_dim.min(host.window_kv.len())],
                    )
                    .map_or(0, |window| {
                        diag_checksum_bytes(&window.codes[..window.codes.len().min(512)])
                    })
                };
                let host_last = if host.window_rows == 0 {
                    0
                } else {
                    let row = host.window_rows - 1;
                    let start = row * host.head_dim;
                    let end = (start + host.head_dim).min(host.window_kv.len());
                    sparkflash_kernels::cuda_harness::engine_window_row(&host.window_kv[start..end])
                        .map_or(0, |window| {
                            diag_checksum_bytes(&window.codes[..window.codes.len().min(512)])
                        })
                };
                lines.push(format!(
                    "operand window-codes: device len={} first_checksum={:016x} last_checksum={:016x} compressed_code_bytes={} | host quantised first_checksum={:016x} last_checksum={:016x}",
                    device.window_codes_len,
                    device.window_first_checksum,
                    device.window_last_checksum,
                    device.compressed_codes_len,
                    host_first,
                    host_last
                ));
                // Stage-aligned projection twin: the host pre-attention row is the reference
                // boundary's row for this position, and the probe reproduces the host projection
                // stages on that exact row. Raw wq_a/wq_b/wkv are compared before q_norm/RoPE, the
                // q the body consumed is compared as BF16, and the window-row quantiser is compared
                // on the host row it quantised.
                let host_pre = reference_boundaries
                    .get(position)
                    .and_then(|boundary| boundary.row(layer))
                    .and_then(|row| {
                        row.tensor(
                            sparkflash_engine::forward::boundary::PRE_ATTENTION_HIDDEN,
                        )
                    })
                    .and_then(|tensor| tensor.data().as_f32())
                    .and_then(|values| {
                        let width = config.dim;
                        if width != 0
                            && values.len().is_multiple_of(width)
                            && (position + 1) * width <= values.len()
                        {
                            Some(&values[position * width..(position + 1) * width])
                        } else {
                            None
                        }
                    });
                if let (Some(host_pre), Some(trace)) = (
                    host_pre,
                    sequence_outputs
                        .get(position)
                        .and_then(|output| output.layers.iter().find(|trace| trace.layer == layer)),
                ) {
                    let (mismatches, max_abs) =
                        diag_stage_compare(&trace.pre_attention_hidden, host_pre);
                    lines.push(format!(
                        "operand pre-attention: device vs host bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                        host_pre.len()
                    ));
                    let source = sparkflash_engine::safetensors::SafeTensorsSource::open(
                        checkpoint,
                        &checkpoint.join("model.safetensors.index.json"),
                    );
                    if let Ok(source) = source {
                        let prefix = format!("layers.{layer}");
                        match sparkflash_engine::forward::attention::attention_projection_probe(
                            config,
                            layer,
                            &prefix,
                            host_pre,
                            position,
                            &source,
                            limit,
                        ) {
                            Ok(probe) => {
                                let mut stage = |name: &str, device: Option<&[f32]>, host: &[f32]| {
                                    if let Some(device) = device
                                        && device.len() == host.len()
                                    {
                                        let (mismatches, max_abs) = diag_stage_compare(device, host);
                                        lines.push(format!(
                                            "projection {name}: device vs host bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                                            host.len()
                                        ));
                                    }
                                };
                                stage("raw wq_a latent", diag_latent_raw, &probe.latent_raw);
                                stage("wq_b input latent normed", diag_wq_b_input, &probe.latent_normed);
                                stage("raw wq_b q pre-RoPE", diag_q_raw, &probe.q_raw);
                                stage("raw wkv", diag_wkv_raw, &probe.wkv_raw);
                                if let Some((mismatches, len, max_abs)) = wq_b_kernel {
                                    lines.push(format!(
                                        "projection wq_b kernel vs mxfp8_ref: bf16_mismatches={mismatches}/{len} max_abs_diff={max_abs:.6e}"
                                    ));
                                }
                                if let Some(device_input) = diag_wq_b_input {
                                    use sparkflash_engine::forward::attention::AttentionProjectionBackend as _;
                                    let mut host_backend =
                                        sparkflash_engine::forward::attention::HostAttentionProjectionBackend;
                                    match host_backend.wq_b(
                                        config,
                                        layer,
                                        &format!("layers.{layer}"),
                                        device_input,
                                        &source,
                                        limit,
                                    ) {
                                        Ok(host_on_device) => {
                                            if let Some(device_q) = diag_q_raw
                                                && device_q.len() == host_on_device.len()
                                            {
                                                let (mismatches, max_abs) =
                                                    diag_stage_compare(device_q, &host_on_device);
                                                lines.push(format!(
                                                    "projection wq_b host-on-device-latent: device vs host bf16_mismatches={mismatches}/{} max_abs_diff={max_abs:.6e}",
                                                    host_on_device.len()
                                                ));
                                            }
                                        }
                                        Err(error) => lines.push(format!(
                                            "projection wq_b host-on-device-latent failed: {error}"
                                        )),
                                    }
                                }
                                let (q_mismatches, q_max_abs) =
                                    diag_stage_compare(&device.q_values, &probe.q_rot);
                                let expected_quant = probe
                                    .q_rot
                                    .iter()
                                    .map(|value| (diag_bf16(*value) - *value).abs())
                                    .fold(0f32, f32::max);
                                lines.push(format!(
                                    "projection q staged BF16: device vs host bf16_mismatches={q_mismatches}/{} max_abs_diff={q_max_abs:.6e} expected_bf16_quant_delta={expected_quant:.6e}",
                                    probe.q_rot.len()
                                ));
                                if let Some(codes) = diag_window_codes {
                                    match sparkflash_kernels::cuda_harness::engine_window_row(
                                        &probe.window_kv[..config.head_dim.min(probe.window_kv.len())],
                                    ) {
                                        Ok(window) => {
                                            let equal = codes == window.codes.as_slice();
                                            lines.push(format!(
                                                "projection window-row quantiser: device_codes_len={} host_codes_len={} byte_equal={} device_checksum={:016x} host_checksum={:016x}",
                                                codes.len(),
                                                window.codes.len(),
                                                equal,
                                                diag_checksum_bytes(codes),
                                                diag_checksum_bytes(&window.codes)
                                            ));
                                        }
                                        Err(detail) => lines.push(format!(
                                            "projection window-row quantiser: host engine_window_row failed: {detail}"
                                        )),
                                    }
                                }
                            }
                            Err(error) => lines.push(format!(
                                "projection host probe failed: {error}"
                            )),
                        }
                    } else if let Err(error) = source {
                        lines.push(format!("projection host source failed: {error}"));
                    }
                }
            }
            (Some(_), None) => lines.push(format!(
                "operand comparison position={position} layer={layer}: no host attention_case record"
            )),
            (None, Some(_)) => lines.push(format!(
                "operand comparison position={position} layer={layer}: no device operand trace"
            )),
            (None, None) => lines.push(format!(
                "operand comparison position={position} layer={layer}: neither trace present"
            )),
        }
    }
    lines
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn sequence_comparison_rows(
    reference: &[Option<u32>],
    hybrid: &[Option<u32>],
    prompt_len: usize,
    full_claim: bool,
) -> Vec<(usize, Option<u32>, Option<u32>, bool)> {
    let count = reference.len().max(hybrid.len());
    let mut rows = Vec::with_capacity(count);
    for index in 0..count {
        let position = if full_claim {
            if index == 0 {
                prompt_len.saturating_sub(1)
            } else {
                prompt_len + index - 1
            }
        } else {
            index
        };
        let expected = reference.get(index).copied().flatten();
        let found = hybrid.get(index).copied().flatten();
        rows.push((position, expected, found, expected == found));
    }
    rows
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
/// Adds the per-kernel event ledger to the timing JSON when SPARKFLASH_STAGE_TIMING=1 armed it.
///
/// The ledger is a separate section (schema sparkflash-stage-timing/1) written into the same file so
/// one artifact carries the step wall samples and the per-launch event samples. It is omitted
/// entirely when the recorder was not armed, and it changes no launch: the brackets are the only
/// difference.
fn attach_stage_timing_ledger(path: &std::path::Path) -> Result<(), String> {
    let Some(report) = sparkflash_kernels::cuda_driver::stage_timing_report() else {
        return Ok(());
    };
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("read the timing file {}: {error}", path.display()))?;
    let mut value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("parse the timing file {}: {error}", path.display()))?;
    let kernels: Vec<serde_json::Value> = report
        .kernels
        .iter()
        .map(|row| {
            serde_json::json!({
                "phase": row.phase,
                "kernel": row.kernel,
                "calls": row.calls,
                "total_ns": row.total_ns,
                "mean_ns": row.mean_ns,
                "min_ns": row.min_ns,
                "max_ns": row.max_ns,
                "first_launch_order": row.first_launch_order,
            })
        })
        .collect();
    let steps: Vec<serde_json::Value> = report
        .steps
        .iter()
        .map(|row| {
            let per_kernel: serde_json::Map<String, serde_json::Value> = row
                .kernels
                .iter()
                .map(|entry| {
                    (
                        entry.kernel.to_string(),
                        serde_json::json!({
                            "calls": entry.calls,
                            "total_ns": entry.total_ns,
                        }),
                    )
                })
                .collect();
            serde_json::json!({
                "step": row.step,
                "warm": row.warm,
                "launches": row.launches,
                "kernel_ns": row.kernel_ns,
                "kernels": per_kernel,
            })
        })
        .collect();
    let Some(object) = value.as_object_mut() else {
        return Err(format!(
            "the timing file {} is not a JSON object",
            path.display()
        ));
    };
    object.insert(
        "kernels_schema".to_string(),
        serde_json::json!("sparkflash-stage-timing/1"),
    );
    object.insert("kernels_clock".to_string(), serde_json::json!("cuda_event"));
    let warm_steps: Vec<&serde_json::Value> = steps
        .iter()
        .filter(|row| row.get("warm").and_then(serde_json::Value::as_bool) == Some(true))
        .collect();
    let steady_steps: Vec<&serde_json::Value> = steps
        .iter()
        .filter(|row| row.get("warm").and_then(serde_json::Value::as_bool) != Some(true))
        .collect();
    object.insert(
        "warmup_steps".to_string(),
        serde_json::json!(warm_steps.len()),
    );
    object.insert(
        "steady_steps".to_string(),
        serde_json::json!(steady_steps.len()),
    );
    object.insert("steps_warm".to_string(), serde_json::json!(warm_steps));
    object.insert("steps_steady".to_string(), serde_json::json!(steady_steps));
    object.insert("kernels_enabled".to_string(), serde_json::json!(true));
    object.insert("kernels".to_string(), serde_json::Value::Array(kernels));
    object.insert(
        "per_step_kernels".to_string(),
        serde_json::Value::Array(steps),
    );
    object.insert(
        "kernels_dropped_names".to_string(),
        serde_json::json!(report.dropped_names),
    );
    object.insert(
        "kernels_notes".to_string(),
        serde_json::json!("event brackets add ~1-3 us per launch; kernel_ns sums overlap and can exceed step wall; the bracketed launches are the same launches in the same order, so the arithmetic is unchanged"),
    );
    let text = serde_json::to_string_pretty(&value)
        .map_err(|error| format!("serialise the timing file {}: {error}", path.display()))?;
    std::fs::write(path, text)
        .map_err(|error| format!("write the timing file {}: {error}", path.display()))?;
    Ok(())
}

#[allow(dead_code)] // kept for the hybrid comparison report that is not wired yet
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
    position: usize,
) -> Vec<String> {
    for trace in &hybrid.layers {
        let Some(row) = reference.row(trace.layer) else {
            return vec![format!(
                "first differing layer: layer {} has no reference boundary row",
                trace.layer
            )];
        };
        let reference_name = if trace.layer + 1 < hybrid.layers.len() {
            sparkflash_engine::forward::boundary::POST_ATTENTION_HIDDEN
        } else {
            sparkflash_engine::forward::boundary::RESIDUAL_OUT
        };
        let Some(tensor) = row.tensor(reference_name) else {
            return vec![format!(
                "first differing layer: layer {} has no reference {reference_name}",
                trace.layer
            )];
        };
        let Some(values) = tensor.data().as_f32() else {
            continue;
        };
        let width = trace.residual_out.len();
        let values = if values.len() == width {
            values
        } else if width != 0
            && values.len().is_multiple_of(width)
            && (position + 1) * width <= values.len()
        {
            &values[position * width..(position + 1) * width]
        } else {
            return vec![format!(
                "first differing layer: layer {} reference residual tensor has {} values, not rows of {width}",
                trace.layer,
                values.len()
            )];
        };
        let max_abs = max_abs_delta(values, &trace.residual_out);
        if max_abs > 0.0 {
            return vec![format!(
                "first differing layer (best effort, stage-aligned): layer {} {} max_abs_delta={:.6e}; this includes the BF16 staging seam and any device-entry delta",
                trace.layer, reference_name, max_abs
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
    if stage == "hybrid_token" || stage == "hybrid_sequence" {
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
        // The count is derived, not hardcoded: a kernel added to the catalog must not fail this
        // test for the number alone (the count is the catalog's, and the catalog is asserted above).
        let pins_now = sparkflash_device::cubins::baseline_pins()
            .expect("baseline")
            .len();
        assert!(
            text.contains(&format!("{pins_now} baseline pins parsed")),
            "the plan quotes the catalog's own pin count ({pins_now})"
        );
        assert!(
            text.contains("crates/sparkflash-kernels/baseline/kernel_baseline.txt"),
            "the catalog names the repository path, not a crate-relative one"
        );
        assert!(text.contains("11 required for the device path"));
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

        let (_, diagnosed) = stage_args(&[
            "hybrid_token",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--moe-layer0-diagnose",
        ])
        .expect("the layer-0 diagnose flag");
        assert!(diagnosed.moe_layer0_diagnose);

        use sparkflash_engine::linalg::CHECKPOINT_TENSOR_LIMIT;
        let default_args = StageArgs {
            checkpoint: std::path::PathBuf::from("/checkpoint"),
            cubin_dir: std::path::PathBuf::from("/cubins"),
            positions: vec![0],
            token: 0,
            element_limit: None,
            moe_preflight: false,
            moe_layer0_diagnose: false,
            fixture: None,
            prompt_tokens: 1,
            decode_steps: 0,
            stop_after_layer: None,
            time_budget_s: 3600,
            sequence: false,
            reference_check: false,
            dump_logits: None,
            timing_json: None,
            wo_a_mode: crate::attention_device::WoAMode::default(),
            wo_a_operand_dump: None,
            attention_twin_positions: None,
            host_attention_from: None,
            operand_audit: false,
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
    fn the_sequence_arm_parses_the_wo_a_mode_and_operand_dump() {
        let (_, args) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--wo-a-mode",
            "mxfp8",
            "--dump-wo-a-operand",
            "/tmp/wo-a-dump",
        ])
        .expect("wo_a arguments");
        assert_eq!(args.wo_a_mode, crate::attention_device::WoAMode::Mxfp8);
        assert_eq!(
            args.wo_a_operand_dump,
            Some(std::path::PathBuf::from("/tmp/wo-a-dump"))
        );
        let (_, defaults) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
        ])
        .expect("defaults");
        assert_eq!(defaults.wo_a_mode, crate::attention_device::WoAMode::Native);
        assert_eq!(defaults.wo_a_operand_dump, None);
        let error = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--wo-a-mode",
            "fp8",
        ])
        .expect_err("an unknown mode is refused");
        assert!(error.contains("not a wo_a mode"), "{error}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_hybrid_sequence_arm_parses_its_own_shape() {
        let (stage, args) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--fixture",
            "/fixture.json",
            "--prompt-tokens",
            "3",
            "--decode-steps",
            "0",
            "--time-budget-s",
            "120",
            "--dump-logits",
            "/tmp/hybrid-logits.jsonl",
        ])
        .expect("hybrid_sequence arguments");
        assert_eq!(stage, "hybrid_sequence");
        assert!(args.sequence);
        assert_eq!(args.prompt_tokens, 3);
        assert_eq!(args.decode_steps, 0);
        assert_eq!(args.time_budget_s, 120);
        assert_eq!(
            args.dump_logits,
            Some(std::path::PathBuf::from("/tmp/hybrid-logits.jsonl"))
        );
        let error = stage_args(&[
            "hybrid_token",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--dump-logits",
            "/tmp/hybrid-logits.jsonl",
        ])
        .expect_err("the dump is wired into hybrid_sequence only");
        assert!(error.contains("only wired into hybrid_sequence"), "{error}");
        let error = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--prompt-tokens",
            "0",
        ])
        .expect_err("zero prompt tokens");
        assert!(error.contains("at least 1"), "{error}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_attention_twin_ladder_flag_selects_positions() {
        let (_, args) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--attention-twin-positions",
            "126,127,128,129,130",
        ])
        .expect("an explicit twin ladder");
        assert_eq!(
            args.attention_twin_positions,
            Some(vec![126, 127, 128, 129, 130])
        );
        let (_, defaulted) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
        ])
        .expect("the default twin ladder");
        assert_eq!(defaulted.attention_twin_positions, None);
        let error = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--attention-twin-positions",
            "126,,129",
        ])
        .expect_err("an empty twin-position entry");
        assert!(error.contains("empty entry"), "{error}");
        // The A/B switch takes one position: a list would silently mis-arm the fallback.
        let (_, ab) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--host-attention-from",
            "127",
        ])
        .expect("a host-attention A/B position");
        assert_eq!(ab.host_attention_from, Some(127));
        let error = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--host-attention-from",
            "127,128",
        ])
        .expect_err("a host-attention list");
        assert!(error.contains("not a list"), "{error}");
        // The operand audit is off unless asked for: it captures the reference's attention inputs,
        // which is memory the default run does not spend.
        assert!(!defaulted.operand_audit);
        let (_, audited) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--attention-twin-positions",
            "126,127,128,129,130",
            "--operand-audit",
        ])
        .expect("the operand audit flag");
        assert!(audited.operand_audit);
        assert_eq!(
            audited.attention_twin_positions,
            Some(vec![126, 127, 128, 129, 130]),
            "the audit captures at the twin ladder, it does not replace it"
        );
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
        assert_eq!(missing_body_layers(&[0, 1, 2], &[1, 0, 2], 1), vec![1, 2]);
        assert!(missing_body_layers(&[5, 6], &[1, 1], 1).is_empty());
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
        let line = c4a2_seam_line(&[0, 1, 2], &[1, 1, 1], 3, 1).expect("all layers launch once");
        assert!(
            line.contains("3 layers") && line.contains("6 mHC launches (3 entry + 3 post)"),
            "{line}"
        );
        let error =
            c4a2_seam_line(&[0, 1, 2], &[1, 0, 1], 3, 1).expect_err("layer 1 did not launch");
        assert!(error.contains("[1]"), "{error}");
        assert!(error.contains("withheld"), "{error}");
    }

    #[test]
    fn the_sequence_seam_accepts_one_body_per_position_and_rejects_a_short_layer() {
        let line = c4a2_seam_line(&[0, 1], &[3, 3], 2, 3).expect("three positions on both layers");
        assert!(line.contains("3 times on all 2 layers"), "{line}");
        assert!(line.contains("10 projection + 6 body launches"), "{line}");
        let error =
            c4a2_seam_line(&[0, 1], &[3, 2], 2, 3).expect_err("layer 1 lost a position body");
        assert!(error.contains("[1]"), "{error}");
    }

    #[test]
    fn the_c4a2_line_counts_the_two_engram_re_entries_in_the_mhc_total() {
        let layers: Vec<usize> = (0..40).collect();
        let launches = vec![1; 40];
        let engram_entries = 2;
        let line = c4a2_seam_line(&layers, &launches, layers.len() + engram_entries, 1)
            .expect("all 40 bodies launch once");
        assert_eq!(
            line,
            "c4a-2 seam: format-2 body launched exactly once on all 40 layers; 200 projection + 40 body launches plus the 82 mHC launches (42 entry + 40 post)",
            "the seam line must agree with the adjacent attention total: 200 + 40 + 82 = 322"
        );
        let total = 200 + 40 + 82;
        assert_eq!(total, 322);
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn sequence_comparison_rows_cover_three_prefix_positions() {
        let reference = vec![Some(10), Some(11), Some(12)];
        let hybrid = vec![Some(10), Some(11), Some(12)];
        let rows = sequence_comparison_rows(&reference, &hybrid, 3, false);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].0, 0);
        assert_eq!(rows[2].0, 2);
        assert!(rows.iter().all(|row| row.3), "{rows:?}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sequence_comparison_rows_cover_the_claim_positions_and_a_short_capture() {
        let reference = vec![Some(491), Some(367), Some(201)];
        let hybrid = vec![Some(491), Some(367)];
        let rows = sequence_comparison_rows(&reference, &hybrid, 130, true);
        assert_eq!(
            rows.iter().map(|row| row.0).collect::<Vec<_>>(),
            vec![129, 130, 131]
        );
        assert!(rows[0].3);
        assert!(!rows[2].3);
        assert_eq!(rows[2].2, None);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_timing_flag_is_opt_in_and_only_for_the_sequence_arm() {
        let (_, plain) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--fixture",
            "/fixture.json",
            "--prompt-tokens",
            "2",
        ])
        .expect("sequence args");
        assert!(
            plain.timing_json.is_none(),
            "no flag constructs no recorder"
        );
        let (_, timed) = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--fixture",
            "/fixture.json",
            "--prompt-tokens",
            "2",
            "--timing-json",
            "/tmp/timing.json",
        ])
        .expect("the timing flag");
        assert_eq!(
            timed.timing_json.as_deref(),
            Some(std::path::Path::new("/tmp/timing.json"))
        );
        let error = stage_args(&[
            "hybrid_token",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--timing-json",
            "/tmp/timing.json",
        ])
        .expect_err("only the sequence arm records");
        assert!(error.contains("only wired into hybrid_sequence"), "{error}");
        let error = stage_args(&[
            "hybrid_sequence",
            "--checkpoint",
            "/checkpoint",
            "--cubin-dir",
            "/cubins",
            "--fixture",
            "/fixture.json",
            "--timing-json",
            "",
        ])
        .expect_err("an empty path is refused");
        assert!(
            error.contains("--timing-json needs a non-empty path"),
            "{error}"
        );
    }

    #[test]
    fn the_default_plan_and_dry_run_print_no_timing_line() {
        // Flag off is the default: the host-observable output of the two host-only arms must not
        // gain a timing line, and a disabled recorder writes no document (chain_timing's test).
        let plan = plan_lines().expect("the plan");
        let dry = dry_run_lines().expect("the dry-run wiring");
        for line in plan.iter().chain(dry.iter()) {
            assert!(
                !line.starts_with("timing:"),
                "no default timing line: {line}"
            );
        }
    }
}
