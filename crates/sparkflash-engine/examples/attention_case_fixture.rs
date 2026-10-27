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

//! Generate the engine-produced CSA-union attention case fixture (task-14 / 2B-i(a)).
//!
//! Runs the real reference forward for the released 130-token prompt, captures the attention
//! inputs and pre-projection output for the requested decoder layers at the last prefill position
//! with [sparkflash_engine::forward::attention_case], and writes a self-describing JSON fixture for
//! the device-side m8_union_attention example.
//!
//! The fixture is host-produced evidence only: it records the engine reference forward's own
//! values, not a device result. The device comparison is the separate example's job.
//!
//! # What the fixture carries
//!
//! For each requested layer: the rotated query [heads][head_dim] f32, the per-head sink, the window
//! ids the reference attends, the window KV rows the reference stored (f32 after its FP8 round trip)
//! with the engine-format E4M3 codes and E8M0 scales the device window path takes, the compressed
//! ids (engine shared index space, NO_INDEX entries included), the compressed rows the reference
//! built (f32) with their store-row bytes (256 B packed FP4 + 32 B E4M3 scales, the device main-KV
//! layout), the engine selection mapped to absolute token positions, and the reference attention
//! output (pre-inverse-RoPE, [heads][head_dim] f32).
//!
//! The window codes/scales are produced the way the device harness produces them: bf16-round the
//! stored f32 row, then the engine's released FP8 activation quantiser (E4M3 elements, one E8M0
//! scale per 32 channels). The compressed store bytes use sparkflash_kv::fp4::quantize_main_kv, the
//! same rule the device store uses (E2M1 packed two per byte, one E4M3 scale per 16 channels).
//!
//! # Provenance
//!
//! The fixture records the exact argv, host, binary sha256, checkpoint path, the golden sequence
//! fixture it cross-checks against (path, sha256, prompt equality, prefill next-token agreement and
//! the logits delta), and free-form notes. It refuses to mix provenance kinds: --provenance ci
//! requires the tiny finite-weight config, --provenance node requires the released checkpoint.

use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention_case::{self, AttentionCaseRecord, NO_INDEX};
use sparkflash_engine::forward::{InferenceEngine, SequenceState, StepOutput};
use sparkflash_engine::linalg::{CHECKPOINT_TENSOR_LIMIT, REFERENCE_TENSOR_LIMIT};
use sparkflash_engine::loader::ReferenceWeights;
use sparkflash_engine::sequence_fixture::{
    FiniteFp32Weights, SequenceFixture, deterministic_prompt,
};
use sparkflash_engine::tensor::TensorPlan;
use sparkflash_engine::weights::WeightSource;

/// The fixture schema version this generator writes.
const FIXTURE_SCHEMA_VERSION: u32 = 1;
/// The released prompt length the device CED path compares (last prefill position 129).
const DEFAULT_PROMPT_TOKENS: usize = 130;
/// The deterministic seed the golden sequence fixture uses.
const DEFAULT_SEED: u64 = 7;

/// Which configuration to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfigChoice {
    /// The deterministic tiny configuration (weight-free, fast, host-only smoke test).
    Tiny,
    /// The released DeepSeek-V4.1-Flash configuration with a real checkpoint.
    Released,
}

/// Which provenance discipline the fixture records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProvenanceChoice {
    /// Derived from the configuration when not given: tiny -> ci, released -> node.
    Auto,
    /// Weight-free CI provenance: tiny config, no checkpoint.
    Ci,
    /// Real-checkpoint node provenance: host and binary sha256 recorded.
    Node,
}

/// Parsed command line.
struct Args {
    config: ConfigChoice,
    checkpoint: Option<PathBuf>,
    index: Option<PathBuf>,
    prompt_tokens: usize,
    seed: u64,
    layers: Vec<usize>,
    position: Option<usize>,
    golden_fixture: Option<PathBuf>,
    out: PathBuf,
    provenance: ProvenanceChoice,
    notes: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("attention_case_fixture: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse()?;
    let config = match args.config {
        ConfigChoice::Tiny => ModelConfig::tiny(),
        ConfigChoice::Released => ModelConfig::v41_flash().map_err(|e| e.to_string())?,
    };
    let provenance_label = match (args.provenance, args.config) {
        (ProvenanceChoice::Auto, ConfigChoice::Tiny) => "ci",
        (ProvenanceChoice::Auto, ConfigChoice::Released) => "node",
        (ProvenanceChoice::Ci, ConfigChoice::Tiny) => "ci",
        (ProvenanceChoice::Node, ConfigChoice::Released) => "node",
        (ProvenanceChoice::Ci, ConfigChoice::Released) => {
            return Err("--provenance ci cannot describe the released checkpoint".to_string());
        }
        (ProvenanceChoice::Node, ConfigChoice::Tiny) => {
            return Err("--provenance node requires --config released".to_string());
        }
    };
    if args.layers.is_empty() {
        return Err("--layers needs at least one layer".to_string());
    }
    let prompt_tokens = args.prompt_tokens;
    if prompt_tokens == 0 {
        return Err("--prompt-tokens must be positive".to_string());
    }
    let position = args.position.unwrap_or(prompt_tokens - 1);
    if position >= prompt_tokens {
        return Err(format!(
            "capture position {position} is outside the {prompt_tokens}-token prefill; this fixture captures a prefill query",
        ));
    }
    for &layer in &args.layers {
        if layer >= config.n_layers {
            return Err(format!(
                "layer {layer} is outside the backbone ({} layers)",
                config.n_layers
            ));
        }
    }
    let prompt = deterministic_prompt(config.vocab_size, prompt_tokens, args.seed);
    let plan = TensorPlan::language_model(&config);
    let (source, source_kind, source_seed, source_checkpoint): (
        Box<dyn WeightSource>,
        &'static str,
        u64,
        Option<String>,
    ) = match args.config {
        ConfigChoice::Tiny => {
            let source = FiniteFp32Weights::from_plan(&plan, args.seed);
            (Box::new(source), "finite_fp32", args.seed, None)
        }
        ConfigChoice::Released => {
            let dir = args
                .checkpoint
                .as_deref()
                .ok_or("--config released requires --checkpoint")?;
            let index = args
                .index
                .clone()
                .unwrap_or_else(|| dir.join("model.safetensors.index.json"));
            let weights = ReferenceWeights::open(dir, &index, &plan).map_err(|e| e.to_string())?;
            (
                Box::new(weights),
                "reference_checkpoint",
                0,
                Some(dir.display().to_string()),
            )
        }
    };
    let limit = match args.config {
        ConfigChoice::Tiny => REFERENCE_TENSOR_LIMIT,
        ConfigChoice::Released => CHECKPOINT_TENSOR_LIMIT,
    };
    let engine = InferenceEngine::new(config.clone(), source, limit).map_err(|e| e.to_string())?;
    let mut state = SequenceState::new(&config);

    let golden = match args.golden_fixture.as_deref() {
        Some(path) => Some(load_golden(path, &prompt)?),
        None => None,
    };
    if let Some(golden) = &golden {
        if !golden.prompt_equal {
            return Err(format!(
                "{}: golden fixture prompt differs from the generated prompt",
                golden.path
            ));
        }
        if !golden.config_equal {
            return Err(format!(
                "{}: golden fixture configuration differs from this run's",
                golden.path
            ));
        }
    }

    attention_case::begin(&args.layers, position)?;
    let prefill = engine
        .prefill(&mut state, &prompt)
        .map_err(|e| format!("prefill failed: {e}"))?;
    let captured = attention_case::finish();
    if captured.is_empty() {
        return Err(format!(
            "the prefill produced no capture for layers {:?} at position {position}",
            args.layers
        ));
    }
    for &layer in &args.layers {
        if !captured.iter().any(|record| record.layer == layer) {
            return Err(format!(
                "layer {layer} was not captured at position {position}; the reference pass did not reach it"
            ));
        }
    }
    if captured.iter().any(|record| record.start_pos != 0) {
        return Err(
            "a capture landed on a non-prefill chunk; this fixture documents prefill queries only"
                .to_string(),
        );
    }
    let layer_cases = captured
        .iter()
        .map(|record| layer_case(&config, record))
        .collect::<Vec<_>>();
    let golden_json = golden.as_ref().map(|golden| golden_json(golden, &prefill));

    let argv: Vec<String> = std::env::args().collect();
    let binary_sha256 = current_binary_sha256()?;
    let mut fixture = serde_json::json!({
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "generator": format!("sparkflash-engine attention_case_fixture v{FIXTURE_SCHEMA_VERSION}"),
        "source": {
            "kind": source_kind,
            "seed": source_seed,
            "checkpoint": source_checkpoint,
        },
        "provenance_label": provenance_label,
        "config_label": match args.config { ConfigChoice::Tiny => "tiny", ConfigChoice::Released => "released" },
        "config": config,
        "prompt_tokens": prompt,
        "prompt_sha256": sha256_u32(&prompt),
        "position": position,
        "capture_kind": "prefill",
        "layers": args.layers,
        "window": config.window_size,
        "kv_source_layers": config.kv_source_layers,
        "index_source_layers": config.index_source_layers,
        "candidate_source_layer": config.candidate_source_layer,
        "compress_ratios": config.compress_ratios,
        "head_dim": config.head_dim,
        "n_heads": config.n_heads,
        "layer_cases": layer_cases,
        "prefill": {
            "next_token": prefill.next_token,
            "logits_sha256": sha256_f32(&prefill.logits),
            "logits_len": prefill.logits.len(),
            "head_input_sha256": sha256_f32(&prefill.head_input),
        },
        "golden_fixture": golden_json,
        "provenance": {
            "command": format_command(&argv),
            "argv": argv,
            "host": std::env::var("HOSTNAME").ok(),
            "binary_sha256": binary_sha256,
            "cwd": std::env::current_dir().map(|p| p.display().to_string()).ok(),
            "notes": args.notes,
        },
        "honesty": [
            "host-produced attention-case fixture: inputs and reference output come from the engine reference forward, not from a device chain",
            "the device comparison over these inputs is a one-stage attention verification, not the 2B sequence milestone and not a serving claim",
            "no token or logits bar is claimed from this fixture; the golden sequence cross-check is a provenance check on the same prompt",
        ],
    });
    if let Some(map_errors) = fixture
        .get("layer_cases")
        .and_then(|cases| cases.as_array())
        .and_then(|cases| {
            let errors: Vec<serde_json::Value> = cases
                .iter()
                .filter_map(|case| case.get("mapping_error").cloned())
                .filter(|error| !error.is_null())
                .collect();
            (!errors.is_empty()).then_some(errors)
        })
    {
        fixture["mapping_errors"] = serde_json::Value::Array(map_errors);
    }
    let bytes = serde_json::to_vec_pretty(&fixture).map_err(|e| e.to_string())?;
    std::fs::write(&args.out, &bytes).map_err(|e| format!("{}: {e}", args.out.display()))?;
    println!(
        "wrote {} ({} B, sha256 {})",
        args.out.display(),
        bytes.len(),
        sha256_bytes(&bytes)
    );
    println!("captured layers {:?} at position {position}", args.layers);
    if let Some(golden) = &golden_json {
        println!(
            "golden cross-check: prompt_equal={} prefill_next_token_equal={} logits_max_abs_delta={:e} logits_sha256_equal={}",
            golden["prompt_equal"],
            golden["prefill_next_token_equal"],
            golden["prefill_logits_max_abs_delta"]
                .as_f64()
                .unwrap_or(f64::NAN),
            golden["prefill_logits_sha256_equal"]
        );
    }
    Ok(())
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut config = ConfigChoice::Released;
        let mut checkpoint = None;
        let mut index = None;
        let mut prompt_tokens = DEFAULT_PROMPT_TOKENS;
        let mut seed = DEFAULT_SEED;
        let mut layers = vec![21usize, 39];
        let mut position = None;
        let mut golden_fixture = None;
        let mut out = None;
        let mut provenance = ProvenanceChoice::Auto;
        let mut notes = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--config" => {
                    config = match value()?.as_str() {
                        "tiny" => ConfigChoice::Tiny,
                        "released" => ConfigChoice::Released,
                        other => return Err(format!("unknown --config value {other}")),
                    };
                }
                "--checkpoint" => checkpoint = Some(PathBuf::from(value()?)),
                "--index" => index = Some(PathBuf::from(value()?)),
                "--prompt-tokens" => {
                    prompt_tokens = value()?
                        .parse()
                        .map_err(|e| format!("--prompt-tokens: {e}"))?;
                }
                "--seed" => seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?,
                "--layers" => {
                    layers = value()?
                        .split(',')
                        .map(|part| part.trim().parse::<usize>())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| format!("--layers: {e}"))?;
                }
                "--position" => {
                    position = Some(value()?.parse().map_err(|e| format!("--position: {e}"))?);
                }
                "--golden-fixture" => golden_fixture = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--provenance" => {
                    provenance = match value()?.as_str() {
                        "auto" => ProvenanceChoice::Auto,
                        "ci" => ProvenanceChoice::Ci,
                        "node" => ProvenanceChoice::Node,
                        other => return Err(format!("unknown --provenance value {other}")),
                    };
                }
                "--note" => notes.push(value()?),
                other => return Err(format!("unknown argument {other}")),
            }
        }
        Ok(Self {
            config,
            checkpoint,
            index,
            prompt_tokens,
            seed,
            layers,
            position,
            golden_fixture,
            out: out.ok_or("--out is required")?,
            provenance,
            notes,
        })
    }
}

/// The loaded golden sequence fixture, before the comparison runs.
struct Golden {
    path: String,
    sha256: String,
    prompt_equal: bool,
    config_equal: bool,
    expected_next_token: Option<u32>,
    expected_logits: Option<Vec<f32>>,
    expected_logits_sha256: String,
}

fn load_golden(path: &Path, prompt: &[u32]) -> Result<Golden, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let fixture: SequenceFixture =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    fixture
        .validate()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let prefill = fixture
        .steps
        .iter()
        .find(|step| step.kind == sparkflash_engine::sequence_fixture::StepKind::Prefill)
        .ok_or_else(|| format!("{}: no prefill step", path.display()))?;
    Ok(Golden {
        path: path.display().to_string(),
        sha256: sha256_bytes(&bytes),
        prompt_equal: fixture.prompt_tokens == prompt,
        config_equal: fixture.config == ModelConfig::v41_flash().map_err(|e| e.to_string())?,
        expected_next_token: prefill.next_token,
        expected_logits_sha256: prefill.logits_sha256.clone(),
        expected_logits: prefill.logits.clone(),
    })
}

fn golden_json(golden: &Golden, prefill: &StepOutput) -> serde_json::Value {
    let delta = golden.expected_logits.as_ref().map(|expected| {
        expected
            .iter()
            .zip(&prefill.logits)
            .map(|(a, b)| f64::from((a - b).abs()))
            .fold(0.0f64, f64::max)
    });
    serde_json::json!({
        "path": golden.path,
        "sha256": golden.sha256,
        "prompt_equal": golden.prompt_equal,
        "config_equal": golden.config_equal,
        "prefill_next_token_equal": golden.expected_next_token == prefill.next_token,
        "prefill_logits_max_abs_delta": delta,
        "prefill_logits_sha256_equal": golden.expected_logits_sha256 == sha256_f32(&prefill.logits),
    })
}

/// One layer's serialised case.
#[derive(Serialize)]
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
    q_sha256: String,
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
    compressed_store: Vec<CompressedRow>,
    selected: SelectedIds,
    mapping_error: Option<String>,
    reference_out: Vec<f32>,
    reference_out_sha256: String,
}

/// The device-format view of one compressed row.
#[derive(Serialize)]
struct CompressedRow {
    store_row: usize,
    absolute_position: usize,
    data: Vec<u8>,
    scales: Vec<u8>,
    latent_sha256: String,
}

/// The selection mapped to the kernel's absolute-id form.
#[derive(Serialize)]
struct SelectedIds {
    window_ids: Vec<usize>,
    compressed_positions: Vec<usize>,
    compressed_engine_ids: Vec<usize>,
    no_index: usize,
    kernel_selected: Vec<u32>,
}

fn layer_case(config: &ModelConfig, record: &AttentionCaseRecord) -> LayerCase {
    let head_dim = record.head_dim;
    let heads = record.heads;
    let width = head_dim * heads;
    let mut mapping_error = None;
    let mut window_kv_f32 = Vec::new();
    for &id in &record.window_ids {
        if id < record.window_rows && record.window_kv.len() >= (id + 1) * head_dim {
            window_kv_f32.extend_from_slice(&record.window_kv[id * head_dim..(id + 1) * head_dim]);
        } else {
            mapping_error = Some(format!(
                "window id {id} is outside the {}-row window buffer",
                record.window_rows
            ));
            break;
        }
    }
    if record.q.len() != width || record.out.len() != width {
        mapping_error = Some(format!(
            "captured q/out are not {width} values (q {}, out {})",
            record.q.len(),
            record.out.len()
        ));
    }
    let window_kv_sha256 = sha256_f32(&window_kv_f32);
    let window_buffer_sha256 = sha256_f32(&record.window_kv);
    let (window_codes, window_scales, window_codes_note) =
        engine_window_view(head_dim, &window_kv_f32);
    let window_offset = record.window_rows;
    let source_layer = config
        .kv_source_layers
        .iter()
        .copied()
        .filter(|source| *source <= record.layer)
        .max();
    let source_ratio = source_layer.and_then(|source| config.compress_ratio(source));
    let mut rows = Vec::new();
    let mut positions = Vec::new();
    let mut no_index = 0usize;
    if mapping_error.is_none() {
        for &id in &record.compressed_ids {
            if id == NO_INDEX {
                no_index += 1;
                continue;
            }
            if id < window_offset {
                mapping_error = Some(format!(
                    "compressed id {id} is below the window offset {window_offset}"
                ));
                break;
            }
            let store_row = id - window_offset;
            if store_row >= record.compressed_rows {
                mapping_error = Some(format!(
                    "compressed id {id} maps to store row {store_row}, outside the {} captured rows",
                    record.compressed_rows
                ));
                break;
            }
            let Some(ratio) = source_ratio.filter(|ratio| *ratio > 0) else {
                mapping_error = Some(format!(
                    "layer {} has no positive source ratio for its compressed rows",
                    record.layer
                ));
                break;
            };
            let absolute_position = store_row * ratio;
            if absolute_position > record.position {
                mapping_error = Some(format!(
                    "compressed absolute position {absolute_position} is past query position {}",
                    record.position
                ));
                break;
            }
            let latent = &record.compressed_kv[store_row * head_dim..(store_row + 1) * head_dim];
            rows.push(compressed_row(
                store_row,
                absolute_position,
                head_dim,
                latent,
            ));
            positions.push(absolute_position);
        }
    }
    let kernel_selected = positions
        .iter()
        .filter_map(|position| u32::try_from(*position).ok())
        .collect::<Vec<u32>>();
    LayerCase {
        layer: record.layer,
        position: record.position,
        start_pos: record.start_pos,
        seqlen: record.seqlen,
        ratio: record.ratio,
        source_layer,
        source_ratio,
        window_rows: record.window_width,
        window_width: record.window_width,
        window_buffer_rows: record.window_rows,
        window_offset,
        compressed_stride: record.compressed_stride,
        compressed_rows: record.compressed_rows,
        heads,
        head_dim,
        q: record.q.clone(),
        q_sha256: sha256_f32(&record.q),
        sink: record.sink.clone(),
        window_ids: record.window_ids.clone(),
        window_kv_f32,
        window_codes,
        window_scales,
        window_codes_note: window_codes_note.to_string(),
        window_kv_sha256,
        window_buffer_sha256,
        compressed_engine_ids: record.compressed_ids.clone(),
        compressed_no_index: no_index,
        compressed_kv_f32: record.compressed_kv.clone(),
        compressed_store: rows,
        selected: SelectedIds {
            window_ids: record.window_ids.clone(),
            compressed_positions: positions.clone(),
            compressed_engine_ids: record.compressed_ids.clone(),
            no_index,
            kernel_selected,
        },
        mapping_error,
        reference_out: record.out.clone(),
        reference_out_sha256: sha256_f32(&record.out),
    }
}

/// The engine-format window view of the attended rows: bf16-round each stored f32 then use the
/// engine's released FP8 activation quantiser (E4M3 codes, one E8M0 scale per 32 channels).
fn engine_window_view(head_dim: usize, rows: &[f32]) -> (Vec<u8>, Vec<u8>, &'static str) {
    if head_dim == 0 || !head_dim.is_multiple_of(32) {
        return (
            Vec::new(),
            Vec::new(),
            "not emitted: the engine window quantiser needs head_dim to be a multiple of 32",
        );
    }
    let mut codes = Vec::with_capacity(rows.len());
    let mut scales = Vec::with_capacity(rows.len() / 32);
    for row in rows.chunks_exact(head_dim) {
        let rounded: Vec<f32> = row.iter().map(|&value| bf16_round(value)).collect();
        match sparkflash_engine::quant::quantize_activation_fp8(&rounded) {
            Ok(quantised) => {
                codes.extend_from_slice(quantised.codes());
                scales.extend_from_slice(quantised.scales());
            }
            Err(_) => {
                return (
                    Vec::new(),
                    Vec::new(),
                    "not emitted: the engine window quantiser refused this row",
                );
            }
        }
    }
    (
        codes,
        scales,
        "engine FP8 E4M3 codes + E8M0 scales, bf16-round then released activation quantiser",
    )
}

/// One compressed row in the device main-KV layout.
fn compressed_row(
    store_row: usize,
    absolute_position: usize,
    head_dim: usize,
    latent: &[f32],
) -> CompressedRow {
    if head_dim != 512 {
        return CompressedRow {
            store_row,
            absolute_position,
            data: Vec::new(),
            scales: Vec::new(),
            latent_sha256: sha256_f32(latent),
        };
    }
    match sparkflash_kv::fp4::quantize_main_kv(latent) {
        Ok(entry) => CompressedRow {
            store_row,
            absolute_position,
            data: entry.data.to_vec(),
            scales: entry.scales.to_vec(),
            latent_sha256: sha256_f32(latent),
        },
        Err(_) => CompressedRow {
            store_row,
            absolute_position,
            data: Vec::new(),
            scales: Vec::new(),
            latent_sha256: sha256_f32(latent),
        },
    }
}

/// Round-to-nearest-even f32 to bf16, the CUDA __float2bfloat16_rn rule the kernels use.
fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn sha256_u32(values: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex(&hasher.finalize())
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

fn format_command(argv: &[String]) -> String {
    let mut line = String::new();
    for arg in argv {
        if !line.is_empty() {
            line.push(' ');
        }
        line.push('"');
        line.push_str(arg);
        line.push('"');
    }
    line
}

fn current_binary_sha256() -> Result<String, String> {
    let path = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}
