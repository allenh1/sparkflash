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

#![forbid(unsafe_code)]
//! One-stage device verification of CSA-union attention (kernel window_kv_format = 2) on
//! engine-produced inputs (task-14 / 2B-i(a)).
//!
//! The input is the JSON attention-case fixture written by the engine-side
//! attention_case_fixture example: the reference forward's rotated query, engine window rows and
//! their FP8 codes/scales, the compressed rows the query reaches in the device main-KV layout, the
//! selected absolute positions, the per-head sink, and the reference attention output for one
//! prefill position of the released 130-token prompt.
//!
//! This example is a harness, not a claim. It launches the merged flash_mla kernel with format 2
//! over exactly those buffers and compares the bf16 device output against the bf16 rounding of the
//! reference output. It also runs format 1 (engine window format, same window rows, the window ids
//! as its selected list) as the structural-exclusion demonstration: format 1 cannot attend the
//! compressed candidates by construction, so its distance from the reference is the size of the
//! contribution the union is supposed to carry.
//!
//! Honest scope: one attention stage on engine-produced inputs. The host seam is the engine
//! reference forward; the device seam is the quantised window rows, the packed main-KV store rows,
//! and the bf16 query/output round trip. This is not the 2B sequence milestone (no sequence, no
//! KV persistence, no chain), and it is not a serving claim. It says nothing about token or logits
//! agreement; the golden sequence fixture is a provenance check on the same prompt, recorded by the
//! fixture generator and echoed in the report.
//!
//! Build the cubin first, as for the other harnesses (pinned nvcc V13.0.88, sm_121a):
//!
//!   nvcc -cubin -arch=sm_121a -I kernels -o out/flash_mla_csa2_sm121.cubin kernels/flash_mla_csa2_sm121.cu
//!
//! Run:
//!
//!   m8_union_attention --fixture attention-case-released.json --cubin-dir out --out report.json
//!   m8_union_attention --fixture attention-case-released.json --dry-run
//!
//! --dry-run validates the fixture and builds every device buffer on the host (including the
//! engine-window rebuild check and the store-row decode seam) without touching CUDA.

use std::path::PathBuf;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_kernels::cuda_harness::{
    FlashMlaKernel, FlashMlaParams, assert_engine_window_finite, engine_window_row,
    flash_mla_params, launch_flash_mla, load_flash_mla,
};
use sparkflash_kernels::mla_ref;

/// The kernel's window_kv_format value for the CSA union.
const UNION_FORMAT: u64 = 2;
/// The kernel's window_kv_format value for the engine FP8 window (window-only).
const WINDOW_ONLY_FORMAT: u64 = 1;
/// The comparison bar the 2B contract states for the attention output.
const DEFAULT_BAR: f32 = 2e-2;
/// The kernel's thread count, a static_assert in the kernel source.
const MLA_THREADS: u32 = 256;
/// The fixture schema version this example reads.
const FIXTURE_SCHEMA_VERSION: u32 = 1;

fn main() {
    if let Err(error) = run() {
        eprintln!("m8_union_attention: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let fixture_bytes =
        std::fs::read(&args.fixture).map_err(|e| format!("{}: {e}", args.fixture.display()))?;
    let fixture_sha256 = sha256_bytes(&fixture_bytes);
    let fixture: Fixture = serde_json::from_slice(&fixture_bytes)
        .map_err(|e| format!("{}: {e}", args.fixture.display()))?;
    fixture.validate()?;
    let cases: Vec<&LayerCase> = match &args.layers {
        Some(layers) => fixture
            .layer_cases
            .iter()
            .filter(|case| layers.contains(&case.layer))
            .collect(),
        None => fixture.layer_cases.iter().collect(),
    };
    if cases.is_empty() {
        return Err("no layer cases selected".to_string());
    }
    for case in &cases {
        if case.mapping_error.is_some() {
            return Err(format!(
                "layer {}: the fixture recorded a mapping error ({}); the selected ids cannot be placed in the device buffers",
                case.layer,
                case.mapping_error.as_deref().unwrap_or_default()
            ));
        }
    }

    let prepared: Vec<Prepared> = cases
        .iter()
        .map(|case| prepare(&fixture, case))
        .collect::<Result<Vec<_>, String>>()?;

    let argv: Vec<String> = std::env::args().collect();
    let binary_sha256 = current_binary_sha256()?;
    let mut report = serde_json::json!({
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "generator": format!("sparkflash-kernels m8_union_attention v{FIXTURE_SCHEMA_VERSION}"),
        "kind": "one_attention_stage_union_device_verification",
        "argv": argv,
        "host": std::env::var("HOSTNAME").ok(),
        "binary_sha256": binary_sha256,
        "cwd": std::env::current_dir().map(|p| p.display().to_string()).ok(),
        "dry_run": args.dry_run,
        "fixture": {
            "path": args.fixture.display().to_string(),
            "sha256": fixture_sha256,
            "schema_version": fixture.schema_version,
            "generator": fixture.generator,
            "config_label": fixture.config_label,
            "position": fixture.position,
            "prompt_tokens": fixture.prompt_tokens,
            "prompt_sha256": fixture.prompt_sha256,
            "n_heads": fixture.n_heads,
            "head_dim": fixture.head_dim,
            "window": fixture.window,
            "provenance": fixture.provenance,
            "golden_fixture": fixture.golden_fixture,
        },
        "notes": args.notes,
        "honesty": [
            "one attention stage on engine-produced inputs, compared against the engine reference forward output",
            "the window rows and main-KV rows reach the device as the store formats, so the seam includes their quantisation; this is not bit-identity with the released kernel",
            "not the 2B sequence milestone (no sequence, no KV persistence, no chain) and not a serving claim",
            "no token or logits bar is claimed here; the golden fixture echo is provenance for the prompt, not a new comparison",
        ],
    });

    if args.dry_run {
        report["layers"] = serde_json::Value::Array(
            prepared
                .iter()
                .map(|prepared| prepared.report_json())
                .collect(),
        );
        write_report(&args, &report, "dry run")?;
        return Ok(());
    }

    let cubin_path = args.cubin_dir.join("flash_mla_csa2_sm121.cubin");
    let cubin = std::fs::read(&cubin_path).map_err(|e| {
        format!(
            "{}: {e} (build the cubin first; the example module docs give the nvcc line and the pinned toolkit)",
            cubin_path.display()
        )
    })?;
    let cubin_sha256 = sha256_bytes(&cubin);
    let cuda = Cuda::load()?;
    let (module, kernel) = load_flash_mla(&cuda, &cubin)?;
    report["cubin"] = serde_json::json!({
        "path": cubin_path.display().to_string(),
        "bytes": cubin.len(),
        "sha256": cubin_sha256,
    });

    let planned = fixture_bytes.len() + prepared.iter().map(Prepared::host_bytes).sum::<usize>();
    let mut allocated = 0usize;
    let buffers = Buffers::allocate(&cuda, &fixture, &prepared, &mut allocated)?;
    report["device"] = serde_json::json!({
        "allocated_bytes": allocated,
        "planned_host_bytes": planned,
        "mem_available_kb": sparkflash_kernels::cuda_driver::mem_available_kb(),
        "grid": [fixture.n_heads, 1, 1],
        "block": [MLA_THREADS, 1, 1],
    });
    let mut layer_reports = Vec::new();
    let mut all_pass = true;
    for prepared in &prepared {
        let result = run_layer(&cuda, kernel, &buffers, prepared, args.bar)?;
        if !result.format_2.pass {
            all_pass = false;
        }
        layer_reports.push(result.report_json(prepared, args.bar));
        println!(
            "layer {}: format 2 exact {}/{} max abs {:e} max ULP {} pass={}; device-side twin exact {}/{} max abs {:e}; format 1 window-only max abs {:e} vs reference ({:e} vs format 2)",
            prepared.layer,
            result.format_2.exact,
            result.format_2.total,
            result.format_2.max_abs,
            result.format_2.max_ulp,
            result.format_2.pass,
            result.device_side_union_twin.exact,
            result.device_side_union_twin.total,
            result.device_side_union_twin.max_abs,
            result.format_1.max_abs,
            result.format_1_vs_format_2.max_abs,
        );
    }
    cuda.free(module, &buffers.ptrs());
    report["layers"] = serde_json::Value::Array(layer_reports);
    report["verdict"] = serde_json::json!({
        "format_2_pass": all_pass,
        "bar_abs": args.bar,
        "criterion": "every device bf16 output value is within bar_abs of the matching reference value, and no value is non-finite",
    });
    write_report(&args, &report, "device run")?;
    if !all_pass {
        return Err(format!(
            "format-2 comparison exceeded the {:.1e} bar on at least one layer",
            args.bar
        ));
    }
    Ok(())
}

/// Parsed command line.
struct Args {
    fixture: PathBuf,
    cubin_dir: PathBuf,
    out: Option<PathBuf>,
    layers: Option<Vec<usize>>,
    dry_run: bool,
    bar: f32,
    notes: Vec<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut fixture = None;
        let mut cubin_dir = PathBuf::from("out");
        let mut out = None;
        let mut layers = None;
        let mut dry_run = false;
        let mut bar = DEFAULT_BAR;
        let mut notes = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--fixture" => fixture = Some(PathBuf::from(value()?)),
                "--cubin-dir" => cubin_dir = PathBuf::from(value()?),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--layers" => {
                    layers = Some(
                        value()?
                            .split(',')
                            .map(|part| part.trim().parse::<usize>())
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| format!("--layers: {e}"))?,
                    );
                }
                "--bar" => bar = value()?.parse().map_err(|e| format!("--bar: {e}"))?,
                "--note" => notes.push(value()?),
                "--dry-run" => dry_run = true,
                other => return Err(format!("unknown argument {other}")),
            }
        }
        if !(bar.is_finite() && bar > 0.0) {
            return Err("--bar must be positive and finite".to_string());
        }
        Ok(Self {
            fixture: fixture.ok_or("--fixture is required")?,
            cubin_dir,
            out,
            layers,
            dry_run,
            bar,
            notes,
        })
    }
}

/// The fixture root.
#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u32,
    generator: Option<String>,
    config_label: Option<String>,
    prompt_tokens: Vec<u32>,
    prompt_sha256: String,
    position: usize,
    layers: Vec<usize>,
    window: usize,
    head_dim: usize,
    n_heads: usize,
    layer_cases: Vec<LayerCase>,
    provenance: serde_json::Value,
    golden_fixture: serde_json::Value,
}

impl Fixture {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != FIXTURE_SCHEMA_VERSION {
            return Err(format!(
                "fixture schema {} is not the {} this example reads",
                self.schema_version, FIXTURE_SCHEMA_VERSION
            ));
        }
        if self.head_dim != mla_ref::MLA_LATENT {
            return Err(format!(
                "fixture head_dim {} is not the kernel latent {}",
                self.head_dim,
                mla_ref::MLA_LATENT
            ));
        }
        if self.n_heads != mla_ref::MLA_HEADS {
            return Err(format!(
                "fixture n_heads {} is not the kernel heads {}",
                self.n_heads,
                mla_ref::MLA_HEADS
            ));
        }
        if self.window != mla_ref::SWA_WINDOW as usize {
            return Err(format!(
                "fixture window {} is not the kernel window {}",
                self.window,
                mla_ref::SWA_WINDOW
            ));
        }
        if self.layer_cases.is_empty() {
            return Err("the fixture has no layer cases".to_string());
        }
        if self.layers.len() != self.layer_cases.len() {
            return Err("the fixture layer list and layer cases disagree in length".to_string());
        }
        Ok(())
    }
}

/// One layer's case, as the engine fixture writes it.
#[derive(Debug, Deserialize)]
struct LayerCase {
    layer: usize,
    position: usize,
    start_pos: usize,
    seqlen: usize,
    ratio: usize,
    source_layer: Option<usize>,
    source_ratio: Option<usize>,
    window_rows: usize,
    window_width: usize,
    window_buffer_rows: usize,
    window_offset: usize,
    compressed_stride: usize,
    compressed_rows: usize,
    heads: usize,
    head_dim: usize,
    q: Vec<f32>,
    sink: Vec<f32>,
    window_ids: Vec<usize>,
    window_kv_f32: Vec<f32>,
    window_codes: Vec<u8>,
    window_scales: Vec<u8>,
    window_codes_note: String,
    window_kv_sha256: String,
    window_buffer_sha256: String,
    compressed_engine_ids: Vec<usize>,
    compressed_no_index: usize,
    compressed_kv_f32: Vec<f32>,
    compressed_store: Vec<StoreRow>,
    selected: SelectedIds,
    mapping_error: Option<String>,
    reference_out: Vec<f32>,
    reference_out_sha256: String,
}

/// One compressed row in the device main-KV layout.
#[derive(Debug, Deserialize)]
struct StoreRow {
    store_row: usize,
    absolute_position: usize,
    data: Vec<u8>,
    scales: Vec<u8>,
    latent_sha256: String,
}

/// The selection the engine captured, mapped to absolute token positions.
#[derive(Debug, Deserialize)]
struct SelectedIds {
    compressed_positions: Vec<usize>,
    no_index: usize,
    kernel_selected: Vec<u32>,
}

/// The host-side device buffers for one layer, after validation.
struct Prepared {
    layer: usize,
    position: usize,
    heads: usize,
    head_dim: usize,
    window_rows: usize,
    selected_count: usize,
    q_bf16: Vec<u8>,
    q_bf16_f32: Vec<f32>,
    sink_bytes: Vec<u8>,
    sink: Vec<f32>,
    window_codes: Vec<u8>,
    window_scales: Vec<u8>,
    kv_codes: Vec<u8>,
    kv_scales: Vec<u8>,
    selected: Vec<u32>,
    reference: Vec<f32>,
    window_seam_max_abs: f32,
    compressed_seam_max_abs: f32,
    window_rows_checked: usize,
    store_rows_checked: usize,
    compressed_engine_ids: Vec<usize>,
    compressed_no_index: usize,
    source_layer: Option<usize>,
    source_ratio: Option<usize>,
    window_codes_note: String,
    window_kv_sha256: String,
    reference_out_sha256: String,
    latent_sha256s: Vec<String>,
    case_ratio: usize,
    start_pos: usize,
    seqlen: usize,
    window_width: usize,
    window_buffer_rows: usize,
    window_buffer_sha256: String,
    window_offset: usize,
    compressed_stride: usize,
    compressed_rows: usize,
}

impl Prepared {
    fn host_bytes(&self) -> usize {
        self.q_bf16.len()
            + self.sink_bytes.len()
            + self.window_codes.len()
            + self.window_scales.len()
            + self.kv_codes.len()
            + self.kv_scales.len()
            + self.selected.len() * 4
            + self.reference.len() * 4
    }

    fn report_json(&self) -> serde_json::Value {
        serde_json::json!({
            "layer": self.layer,
            "position": self.position,
            "start_pos": self.start_pos,
            "seqlen": self.seqlen,
            "ratio": self.case_ratio,
            "source_layer": self.source_layer,
            "source_ratio": self.source_ratio,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "window_rows": self.window_rows,
            "window_width": self.window_width,
            "window_buffer_rows": self.window_buffer_rows,
            "window_offset": self.window_offset,
            "compressed_stride": self.compressed_stride,
            "compressed_rows": self.compressed_rows,
            "selected_count": self.selected_count,
            "compressed_no_index": self.compressed_no_index,
            "compressed_engine_ids": self.compressed_engine_ids,
            "input_seams": {
                "window_rebuild_rows_checked": self.window_rows_checked,
                "window_decode_vs_stored_f32_max_abs": self.window_seam_max_abs,
                "store_rows_checked": self.store_rows_checked,
                "store_decode_vs_engine_f32_max_abs": self.compressed_seam_max_abs,
                "window_codes_note": self.window_codes_note,
                "window_kv_sha256": self.window_kv_sha256,
                "window_buffer_sha256": self.window_buffer_sha256,
                "reference_out_sha256": self.reference_out_sha256,
                "store_latent_sha256s": self.latent_sha256s,
            },
            "buffers": {
                "q_bf16_bytes": self.q_bf16.len(),
                "sink_bytes": self.sink_bytes.len(),
                "window_codes_bytes": self.window_codes.len(),
                "window_scales_bytes": self.window_scales.len(),
                "kv_packed_bytes": self.kv_codes.len(),
                "kv_scale_bytes": self.kv_scales.len(),
                "selected_bytes": self.selected.len() * 4,
                "out_bytes": self.reference.len() * 2,
            },
        })
    }
}

fn prepare(fixture: &Fixture, case: &LayerCase) -> Result<Prepared, String> {
    if case.start_pos != 0 {
        return Err(format!(
            "layer {}: the fixture captured a non-prefill chunk (start_pos {}); this example documents prefill queries",
            case.layer, case.start_pos
        ));
    }
    if case.position != fixture.position {
        return Err(format!(
            "layer {}: case position {} is not the fixture position {}",
            case.layer, case.position, fixture.position
        ));
    }
    if case.heads != fixture.n_heads || case.head_dim != fixture.head_dim {
        return Err(format!(
            "layer {}: case geometry {}x{} disagrees with the fixture {}x{}",
            case.layer, case.heads, case.head_dim, fixture.n_heads, fixture.head_dim
        ));
    }
    if case.window_ids.len() != case.window_width || case.window_width != case.window_rows {
        return Err(format!(
            "layer {}: window ids {}, width {} and supplied rows {} disagree",
            case.layer,
            case.window_ids.len(),
            case.window_width,
            case.window_rows
        ));
    }
    if case.window_rows == 0 || case.window_rows > mla_ref::SWA_WINDOW as usize {
        return Err(format!(
            "layer {}: window_rows {} is outside 1..={}",
            case.layer,
            case.window_rows,
            mla_ref::SWA_WINDOW
        ));
    }
    let head_dim = case.head_dim;
    let heads = case.heads;
    let width = head_dim * heads;
    if case.q.len() != width || case.reference_out.len() != width {
        return Err(format!(
            "layer {}: q {} / reference out {} are not {} values",
            case.layer,
            case.q.len(),
            case.reference_out.len(),
            width
        ));
    }
    if case.sink.len() != heads {
        return Err(format!(
            "layer {}: sink has {} values, expected {heads}",
            case.layer,
            case.sink.len()
        ));
    }
    if case.window_kv_f32.len() != case.window_rows * head_dim {
        return Err(format!(
            "layer {}: window f32 buffer has {} values, expected {}",
            case.layer,
            case.window_kv_f32.len(),
            case.window_rows * head_dim
        ));
    }
    if case.window_kv_sha256 != sha256_f32(&case.window_kv_f32) {
        return Err(format!(
            "layer {}: window f32 buffer does not match its recorded sha256",
            case.layer
        ));
    }
    if case.reference_out_sha256 != sha256_f32(&case.reference_out) {
        return Err(format!(
            "layer {}: reference out does not match its recorded sha256",
            case.layer
        ));
    }

    // The window side: the fixture carries already-quantised engine-format rows. Rebuild them with
    // the one shared helper and require byte equality, so the fixture's window view cannot drift
    // from what every other harness feeds the kernel.
    let mut rebuilt_codes = Vec::with_capacity(case.window_rows * head_dim);
    let mut rebuilt_scales = Vec::with_capacity(case.window_rows * mla_ref::WINDOW_SCALE_BYTES);
    let mut window_seam_max_abs = 0.0f32;
    for row in 0..case.window_rows {
        let latent = &case.window_kv_f32[row * head_dim..(row + 1) * head_dim];
        let built = engine_window_row(latent)
            .map_err(|e| format!("layer {} row {row}: {e}", case.layer))?;
        rebuilt_codes.extend_from_slice(&built.codes);
        rebuilt_scales.extend_from_slice(&built.scales);
        for (index, expected) in latent.iter().enumerate() {
            let decoded = mla_ref::window_kv_value(&built.codes, &built.scales, 0, index);
            let delta = (decoded - expected).abs();
            if delta > window_seam_max_abs {
                window_seam_max_abs = delta;
            }
        }
    }
    if rebuilt_codes != case.window_codes || rebuilt_scales != case.window_scales {
        let mismatch = rebuilt_codes
            .iter()
            .zip(&case.window_codes)
            .position(|(a, b)| a != b)
            .or_else(|| {
                rebuilt_scales
                    .iter()
                    .zip(&case.window_scales)
                    .position(|(a, b)| a != b)
            });
        return Err(format!(
            "layer {}: the fixture window codes/scales differ from cuda_harness::engine_window_row at {:?}; the fixture builder and the shared helper are not the same rule",
            case.layer, mismatch
        ));
    }
    assert_engine_window_finite(
        &case.window_codes,
        &case.window_scales,
        case.window_rows,
        &format!("layer {} fixture window rows", case.layer),
    )?;

    // The compressed side: place each store row at its absolute token position. The kernel reads
    // kv_fp4[kv_fp4 + candidate * 256] and kv_scales[candidate * 32], so the buffers are indexed by
    // absolute position and the unused (odd, ratio-2) slots stay zero.
    let packed = mla_ref::ROW_PACKED_BYTES;
    let scale_bytes = mla_ref::ROW_SCALE_BYTES;
    let positions = fixture.position + 1;
    let mut kv_codes = vec![0u8; positions * packed];
    let mut kv_scales = vec![0u8; positions * scale_bytes];
    let mut occupied = vec![false; positions];
    let mut selected = Vec::with_capacity(case.compressed_store.len());
    let mut compressed_seam_max_abs = 0.0f32;
    let mut latent_sha256s = Vec::with_capacity(case.compressed_store.len());
    for row in &case.compressed_store {
        if row.data.len() != packed || row.scales.len() != scale_bytes {
            return Err(format!(
                "layer {} store row {}: {} data / {} scale bytes, expected {packed}/{scale_bytes}",
                case.layer,
                row.store_row,
                row.data.len(),
                row.scales.len()
            ));
        }
        if row.absolute_position >= positions {
            return Err(format!(
                "layer {} store row {}: absolute position {} is outside the query position {}",
                case.layer, row.store_row, row.absolute_position, fixture.position
            ));
        }
        if occupied[row.absolute_position] {
            return Err(format!(
                "layer {}: two store rows claim absolute position {}",
                case.layer, row.absolute_position
            ));
        }
        occupied[row.absolute_position] = true;
        kv_codes[row.absolute_position * packed..(row.absolute_position + 1) * packed]
            .copy_from_slice(&row.data);
        kv_scales[row.absolute_position * scale_bytes..(row.absolute_position + 1) * scale_bytes]
            .copy_from_slice(&row.scales);
        let engine_latent =
            &case.compressed_kv_f32[row.store_row * head_dim..(row.store_row + 1) * head_dim];
        for (index, expected) in engine_latent.iter().enumerate() {
            let decoded = mla_ref::kv_latent(&row.data, &row.scales, 0, index);
            let delta = (decoded - expected).abs();
            if delta > compressed_seam_max_abs {
                compressed_seam_max_abs = delta;
            }
        }
        let absolute = u32::try_from(row.absolute_position)
            .map_err(|_| format!("position {} does not fit a u32", row.absolute_position))?;
        selected.push(absolute);
        latent_sha256s.push(row.latent_sha256.clone());
        if row.latent_sha256 != sha256_f32(engine_latent) {
            return Err(format!(
                "layer {} store row {}: recorded latent sha256 does not match the captured f32 row",
                case.layer, row.store_row
            ));
        }
    }
    if selected != case.selected.kernel_selected
        || case.selected.compressed_positions.len() != selected.len()
    {
        return Err(format!(
            "layer {}: store rows and the captured selected id list disagree",
            case.layer
        ));
    }
    for position in &selected {
        if *position > case.position as u32 {
            return Err(format!(
                "layer {}: selected position {position} is past the query position {}",
                case.layer, case.position
            ));
        }
    }
    if selected.len() > mla_ref::MAX_SELECTED {
        return Err(format!(
            "layer {}: {} selected candidates exceed the kernel's {} candidate array",
            case.layer,
            selected.len(),
            mla_ref::MAX_SELECTED
        ));
    }
    if case.window_rows + selected.len() > mla_ref::MAX_ATTENDED {
        return Err(format!(
            "layer {}: {} window rows plus {} candidates exceed the union capacity {}",
            case.layer,
            case.window_rows,
            selected.len(),
            mla_ref::MAX_ATTENDED
        ));
    }
    if case.compressed_engine_ids.len() != case.compressed_stride {
        return Err(format!(
            "layer {}: {} engine ids for a stride of {}",
            case.layer,
            case.compressed_engine_ids.len(),
            case.compressed_stride
        ));
    }
    if case.selected.no_index != case.compressed_no_index {
        return Err(format!(
            "layer {}: NO_INDEX counts disagree between selection and case",
            case.layer
        ));
    }

    let q_bf16 = bf16_bytes(&case.q);
    let sink_bytes = f32_bytes(&case.sink);
    Ok(Prepared {
        layer: case.layer,
        position: case.position,
        heads,
        head_dim,
        window_rows: case.window_rows,
        selected_count: selected.len(),
        q_bf16,
        q_bf16_f32: case
            .q
            .iter()
            .map(|value| bf16_bits_to_f32(bf16_bits(*value)))
            .collect(),
        sink_bytes,
        sink: case.sink.clone(),

        window_codes: case.window_codes.clone(),
        window_scales: case.window_scales.clone(),
        kv_codes,
        kv_scales,
        selected,
        reference: case.reference_out.clone(),
        window_seam_max_abs,
        compressed_seam_max_abs,
        window_rows_checked: case.window_rows,
        store_rows_checked: case.compressed_store.len(),
        compressed_engine_ids: case.compressed_engine_ids.clone(),
        compressed_no_index: case.compressed_no_index,
        source_layer: case.source_layer,
        source_ratio: case.source_ratio,
        window_codes_note: case.window_codes_note.clone(),
        window_kv_sha256: case.window_kv_sha256.clone(),
        reference_out_sha256: case.reference_out_sha256.clone(),
        latent_sha256s,
        case_ratio: case.ratio,
        start_pos: case.start_pos,
        seqlen: case.seqlen,
        window_width: case.window_width,
        window_buffer_rows: case.window_buffer_rows,
        window_buffer_sha256: case.window_buffer_sha256.clone(),
        window_offset: case.window_offset,
        compressed_stride: case.compressed_stride,
        compressed_rows: case.compressed_rows,
    })
}

/// The device buffers, allocated once for the whole run.
struct Buffers {
    q: Deviceptr,
    sink: Deviceptr,
    kv_codes: Deviceptr,
    kv_scales: Deviceptr,
    window_codes: Deviceptr,
    window_scales: Deviceptr,
    selected: Deviceptr,
    index_key: Deviceptr,
    index_query: Deviceptr,
    block_scores: Deviceptr,
    block_best: Deviceptr,
    out: Deviceptr,
    out_bytes: usize,
    q_bytes: usize,
    sink_bytes: usize,
    window_codes_bytes: usize,
    window_scales_bytes: usize,
    kv_codes_bytes: usize,
    kv_scales_bytes: usize,
    selected_bytes: usize,
}

impl Buffers {
    fn allocate(
        cuda: &Cuda,
        fixture: &Fixture,
        prepared: &[Prepared],
        allocated: &mut usize,
    ) -> Result<Self, String> {
        let positions = fixture.position + 1;
        let q_bytes = fixture.n_heads * fixture.head_dim * 2;
        let sink_bytes = fixture.n_heads * 4;
        let window_codes_bytes = mla_ref::SWA_WINDOW as usize * mla_ref::MLA_LATENT;
        let window_scales_bytes = mla_ref::SWA_WINDOW as usize * mla_ref::WINDOW_SCALE_BYTES;
        let kv_codes_bytes = positions * mla_ref::ROW_PACKED_BYTES;
        let kv_scales_bytes = positions * mla_ref::ROW_SCALE_BYTES;
        let selected_bytes = prepared
            .iter()
            .map(|prepared| prepared.window_rows.max(prepared.selected_count))
            .max()
            .unwrap_or(1)
            * 4;
        let index_key_bytes =
            mla_ref::HSI_BLOCKS * mla_ref::HSI_BLOCK_TOKENS * mla_ref::INDEX_HEAD_DIM * 2;
        let index_query_bytes = mla_ref::MLA_HEADS * mla_ref::INDEX_HEAD_DIM * 2;
        let block_bytes = mla_ref::HSI_BLOCKS * 4;
        let out_bytes = q_bytes;
        let planned = q_bytes
            + sink_bytes
            + window_codes_bytes
            + window_scales_bytes
            + kv_codes_bytes
            + kv_scales_bytes
            + selected_bytes
            + index_key_bytes
            + index_query_bytes
            + 2 * block_bytes
            + out_bytes;
        if let Some(available) = sparkflash_kernels::cuda_driver::mem_available_kb()
            && planned as u64 > available.saturating_mul(2)
        {
            return Err(format!(
                "the planned {planned} B of device buffers does not fit the {available} KB the driver reports available"
            ));
        }
        Ok(Self {
            q: cuda.alloc(q_bytes, allocated)?,
            sink: cuda.alloc(sink_bytes, allocated)?,
            kv_codes: cuda.alloc(kv_codes_bytes, allocated)?,
            kv_scales: cuda.alloc(kv_scales_bytes, allocated)?,
            window_codes: cuda.alloc(window_codes_bytes, allocated)?,
            window_scales: cuda.alloc(window_scales_bytes, allocated)?,
            selected: cuda.alloc(selected_bytes, allocated)?,
            index_key: cuda.alloc(index_key_bytes, allocated)?,
            index_query: cuda.alloc(index_query_bytes, allocated)?,
            block_scores: cuda.alloc(block_bytes, allocated)?,
            block_best: cuda.alloc(block_bytes, allocated)?,
            out: cuda.alloc(out_bytes, allocated)?,
            out_bytes,
            q_bytes,
            sink_bytes,
            window_codes_bytes,
            window_scales_bytes,
            kv_codes_bytes,
            kv_scales_bytes,
            selected_bytes,
        })
    }

    fn ptrs(&self) -> Vec<Deviceptr> {
        vec![
            self.q,
            self.sink,
            self.kv_codes,
            self.kv_scales,
            self.window_codes,
            self.window_scales,
            self.selected,
            self.index_key,
            self.index_query,
            self.block_scores,
            self.block_best,
            self.out,
        ]
    }
}

/// One format's comparison result.
struct Stats {
    exact: usize,
    total: usize,
    max_abs: f32,
    max_ulp: u32,
    max_rel: f32,
    non_finite: usize,
    pass: bool,
}

impl Stats {
    fn report_json(&self) -> serde_json::Value {
        serde_json::json!({
            "exact_bf16": self.exact,
            "total": self.total,
            "max_abs_delta": self.max_abs,
            "max_ulp": self.max_ulp,
            "max_rel_delta": self.max_rel,
            "non_finite_values": self.non_finite,
            "pass": self.pass,
        })
    }
}

/// One layer's device results.
struct LayerResult {
    format_2: Stats,
    device_side_union_twin: Stats,
    format_1: Stats,
    format_1_vs_format_2: Stats,
    deterministic: bool,
}

impl LayerResult {
    fn report_json(&self, prepared: &Prepared, bar: f32) -> serde_json::Value {
        let mut report = prepared.report_json();
        report["format_2"] = self.format_2.report_json();
        report["device_side_union_twin"] = self.device_side_union_twin.report_json();
        report["device_side_note"] = serde_json::json!(
            "mla_ref::attention_with_sink_union over the exact device-format inputs (bf16 q, engine window bytes, position-indexed store bytes); this is a device-vs-twin consistency diagnostic, not the engine-reference bar"
        );
        report["format_1_window_only"] = self.format_1.report_json();
        report["format_1_vs_format_2"] = self.format_1_vs_format_2.report_json();
        report["deterministic_format_2"] = serde_json::json!(self.deterministic);
        report["bar_abs"] = serde_json::json!(bar);
        report["structural_exclusion"] = serde_json::json!({
            "statement": "format 1 cannot attend the selected compressed candidates; its distance from the reference is the contribution the union carries",
            "max_abs_contribution": self.format_1.max_abs,
            "union_restores": serde_json::json!(self.format_2.pass && !self.format_1.pass),
        });
        report
    }
}

fn run_layer(
    cuda: &Cuda,
    kernel: FlashMlaKernel,
    buffers: &Buffers,
    prepared: &Prepared,
    bar: f32,
) -> Result<LayerResult, String> {
    // Buffers are fixed-size: pad the host side to the allocation and copy what the layer owns.
    let mut q_host = vec![0u8; buffers.q_bytes];
    q_host[..prepared.q_bf16.len()].copy_from_slice(&prepared.q_bf16);
    cuda.copy_htod(&buffers.q, &q_host)?;
    let mut sink_host = vec![0u8; buffers.sink_bytes];
    sink_host[..prepared.sink_bytes.len()].copy_from_slice(&prepared.sink_bytes);
    cuda.copy_htod(&buffers.sink, &sink_host)?;
    let mut window_codes_host = vec![0u8; buffers.window_codes_bytes];
    window_codes_host[..prepared.window_codes.len()].copy_from_slice(&prepared.window_codes);
    cuda.copy_htod(&buffers.window_codes, &window_codes_host)?;
    let mut window_scales_host = vec![0u8; buffers.window_scales_bytes];
    window_scales_host[..prepared.window_scales.len()].copy_from_slice(&prepared.window_scales);
    cuda.copy_htod(&buffers.window_scales, &window_scales_host)?;
    let mut kv_codes_host = vec![0u8; buffers.kv_codes_bytes];
    kv_codes_host[..prepared.kv_codes.len()].copy_from_slice(&prepared.kv_codes);
    cuda.copy_htod(&buffers.kv_codes, &kv_codes_host)?;
    let mut kv_scales_host = vec![0u8; buffers.kv_scales_bytes];
    kv_scales_host[..prepared.kv_scales.len()].copy_from_slice(&prepared.kv_scales);
    cuda.copy_htod(&buffers.kv_scales, &kv_scales_host)?;
    // The union format takes the supplied candidates; the indexer still executes stage 1 in this
    // kernel, so its buffers must be readable/writable and are zeroed here. Nothing in this harness
    // reads the indexer result.
    cuda.copy_htod(
        &buffers.index_key,
        &vec![0u8; buffers_window_index_key_bytes()],
    )?;
    cuda.copy_htod(
        &buffers.index_query,
        &vec![0u8; mla_ref::MLA_HEADS * mla_ref::INDEX_HEAD_DIM * 2],
    )?;
    cuda.copy_htod(&buffers.block_scores, &vec![0u8; mla_ref::HSI_BLOCKS * 4])?;
    cuda.copy_htod(&buffers.block_best, &vec![0u8; mla_ref::HSI_BLOCKS * 4])?;

    // Format 2: window rows plus the selected absolute positions.
    let mut selected_host = vec![0u8; buffers.selected_bytes];
    let selected_bytes = u32s_to_bytes(&prepared.selected);
    selected_host[..selected_bytes.len()].copy_from_slice(&selected_bytes);
    cuda.copy_htod(&buffers.selected, &selected_host)?;
    let union_params = union_params(buffers, prepared);
    let union_first = launch_once(cuda, kernel, buffers, &union_params)?;
    let union_second = launch_once(cuda, kernel, buffers, &union_params)?;
    let deterministic = union_first == union_second;
    let format_2 = compare("format 2", &union_first, &prepared.reference, bar);
    let device_side_reference = device_side_reference(prepared);
    let device_side_union_twin = compare(
        "device-side union twin",
        &union_first,
        &device_side_reference,
        bar,
    );

    // Format 1: the same window rows, the window row indices as the selected list. The kernel's
    // engine-format branch attends candidate < window_rows and nothing else.
    let window_ids: Vec<u32> = (0..prepared.window_rows as u32).collect();
    let mut window_selected_host = vec![0u8; buffers.selected_bytes];
    let window_selected_bytes = u32s_to_bytes(&window_ids);
    window_selected_host[..window_selected_bytes.len()].copy_from_slice(&window_selected_bytes);
    cuda.copy_htod(&buffers.selected, &window_selected_host)?;
    let window_params = window_only_params(buffers, prepared);
    let window_first = launch_once(cuda, kernel, buffers, &window_params)?;
    let window_second = launch_once(cuda, kernel, buffers, &window_params)?;
    let window_deterministic = window_first == window_second;
    let format_1 = compare(
        "format 1 window-only",
        &window_first,
        &prepared.reference,
        bar,
    );

    // The contribution the union carries: how far window-only attention is from the union output.
    let union_values: Vec<f32> = union_first
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16_bits_to_f32(u16::from_le_bytes(*chunk)))
        .collect();
    let format_1_vs_format_2 = compare("format 1 vs format 2", &window_first, &union_values, bar);

    if !deterministic || !window_deterministic {
        return Err(format!(
            "layer {}: the same launch on the same inputs wrote different bytes (format2 deterministic {deterministic}, format1 deterministic {window_deterministic}); this is a hard failure, not a statistic",
            prepared.layer
        ));
    }
    if format_2.non_finite != 0 {
        return Err(format!(
            "layer {}: format 2 produced {} non-finite values",
            prepared.layer, format_2.non_finite
        ));
    }
    Ok(LayerResult {
        format_2,
        device_side_union_twin,
        format_1,
        format_1_vs_format_2,
        deterministic,
    })
}

fn union_params(buffers: &Buffers, prepared: &Prepared) -> FlashMlaParams {
    flash_mla_params(
        buffers.q.0,
        buffers.kv_codes.0,
        buffers.kv_scales.0,
        buffers.window_codes.0,
        buffers.window_scales.0,
        UNION_FORMAT,
        prepared.window_rows as u64,
        buffers.selected.0,
        buffers.index_key.0,
        buffers.index_query.0,
        buffers.sink.0,
        buffers.block_scores.0,
        buffers.block_best.0,
        buffers.out.0,
        prepared.position as u64,
        prepared.selected_count,
        0, // trace_ring: null; the cubin is built without -DSPARKFLASH_ENABLE_GPU_TRACE
        prepared.layer,
        0, // step_seq: one stage, no chain
    )
}

fn window_only_params(buffers: &Buffers, prepared: &Prepared) -> FlashMlaParams {
    flash_mla_params(
        buffers.q.0,
        buffers.kv_codes.0,
        buffers.kv_scales.0,
        buffers.window_codes.0,
        buffers.window_scales.0,
        WINDOW_ONLY_FORMAT,
        prepared.window_rows as u64,
        buffers.selected.0,
        buffers.index_key.0,
        buffers.index_query.0,
        buffers.sink.0,
        buffers.block_scores.0,
        buffers.block_best.0,
        buffers.out.0,
        prepared.position as u64,
        prepared.window_rows,
        0,
        prepared.layer,
        0,
    )
}

fn launch_once(
    cuda: &Cuda,
    kernel: FlashMlaKernel,
    buffers: &Buffers,
    params: &FlashMlaParams,
) -> Result<Vec<u8>, String> {
    launch_flash_mla(
        cuda,
        kernel,
        (mla_ref::MLA_HEADS as u32, 1u32, 1u32),
        (MLA_THREADS, 1u32, 1u32),
        params,
    )?;
    let mut out = vec![0u8; buffers.out_bytes];
    cuda.copy_dtoh(&mut out, &buffers.out)?;
    Ok(out)
}

/// The CPU union twin over the exact device-format inputs. The kernel is expected to match this
/// closely (it is the same three-pass order); the engine-reference comparison above is the bar.
fn device_side_reference(prepared: &Prepared) -> Vec<f32> {
    let head_dim = mla_ref::MLA_LATENT;
    let mut out = Vec::with_capacity(prepared.heads * head_dim);
    for head in 0..prepared.heads {
        let q = &prepared.q_bf16_f32[head * head_dim..(head + 1) * head_dim];
        let sink = prepared.sink[head];
        let head_out = mla_ref::attention_with_sink_union(
            q,
            &prepared.window_codes,
            &prepared.window_scales,
            prepared.window_rows,
            &prepared.kv_codes,
            &prepared.kv_scales,
            &prepared.selected,
            u32::try_from(prepared.position).unwrap_or(u32::MAX),
            sink,
        );
        out.extend_from_slice(&head_out);
    }
    out
}

fn compare(_context: &str, device: &[u8], reference: &[f32], bar: f32) -> Stats {
    let device_bits: Vec<u16> = device
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| u16::from_le_bytes(*chunk))
        .collect();
    let reference_bits: Vec<u16> = reference.iter().map(|value| bf16_bits(*value)).collect();
    if device_bits.len() != reference_bits.len() {
        return Stats {
            exact: 0,
            total: reference.len(),
            max_abs: f32::INFINITY,
            max_ulp: u32::MAX,
            max_rel: f32::INFINITY,
            non_finite: reference.len(),
            pass: false,
        };
    }
    let mut exact = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0u32;
    let mut max_rel = 0.0f32;
    let mut non_finite = 0usize;
    for (index, (&device_bits, &reference_bits)) in
        device_bits.iter().zip(&reference_bits).enumerate()
    {
        let device_value = bf16_bits_to_f32(device_bits);
        let reference_value = reference[index];
        if !device_value.is_finite() || !reference_value.is_finite() {
            non_finite += 1;
            continue;
        }
        if device_bits == reference_bits {
            exact += 1;
        }
        let delta = (device_value - reference_value).abs();
        if delta > max_abs {
            max_abs = delta;
        }
        let relative = delta / reference_value.abs().max(f32::MIN_POSITIVE);
        if relative > max_rel {
            max_rel = relative;
        }
        let ulp = u32::from(ordered_bf16(device_bits).abs_diff(ordered_bf16(reference_bits)));
        if ulp > max_ulp {
            max_ulp = ulp;
        }
    }
    let total = reference_bits.len();
    Stats {
        exact,
        total,
        max_abs,
        max_ulp,
        max_rel,
        non_finite,
        pass: non_finite == 0 && max_abs <= bar,
    }
}

fn ordered_bf16(bits: u16) -> u16 {
    if bits & 0x8000 != 0 {
        !bits
    } else {
        bits | 0x8000
    }
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&bf16_bits(*value).to_le_bytes());
    }
    bytes
}

fn u32s_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn bf16_bits(value: f32) -> u16 {
    half::bf16::from_f32(value).to_bits()
}

fn bf16_bits_to_f32(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn current_binary_sha256() -> Result<String, String> {
    let path = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn write_report(args: &Args, report: &serde_json::Value, label: &str) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(report).map_err(|e| e.to_string())?;
    if let Some(path) = &args.out {
        std::fs::write(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        println!(
            "{label}: wrote {} ({} B, sha256 {})",
            path.display(),
            bytes.len(),
            sha256_bytes(&bytes)
        );
    }
    Ok(())
}

fn buffers_window_index_key_bytes() -> usize {
    mla_ref::HSI_BLOCKS * mla_ref::HSI_BLOCK_TOKENS * mla_ref::INDEX_HEAD_DIM * 2
}
