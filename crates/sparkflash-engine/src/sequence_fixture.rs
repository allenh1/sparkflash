//! The host sequence fixture: the reference output the device path is compared against.
//!
//! Phase 2 milestone 2 runs one 130-token prompt (it crosses the 128-token window) and eight
//! greedy decode steps on the device path, and compares the result token-for-token against the
//! host reference. This module is the host side of that contract:
//!
//! * it generates a [SequenceFixture] from a weight source and a prompt: the prompt ids, the
//!   compared positions, per-step next tokens, per-step logits digests, and (optionally) the
//!   full logits of the compared steps;
//! * it fixes the compared set to exactly the last prefill position plus the decode positions,
//!   and refuses any older position by name (D7: the engine reference runs all 40 layers over
//!   the prompt, but the device CED path runs layers 21..=39 over the last 128 tokens only, so
//!   prefill positions older than the window are not comparable at all);
//! * it records provenance, so a fixture cannot be mistaken for a run it did not come from.
//!
//! The fixture is deliberately two-tiered. The CI fixture is generated from the deterministic
//! finite-FP32 weight source, is weight-free and CUDA-free, and proves the schema, the compared
//! set and the harness plumbing on the dev host. The node golden fixture is generated from the
//! real checkpoint with [crate::loader::ReferenceWeights]; only that one is evidence about the
//! reference model's numbers.
//!
//! The comparison unit is the token id first. The logits are the second bar, on the same last
//! positions, and only where the fixture recorded them.

use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::ModelConfig;
use crate::dtype::DType;
use crate::forward::{InferenceEngine, SequenceState, StepOutput};
use crate::linalg::{CHECKPOINT_TENSOR_LIMIT, EngineError};
use crate::loader::ReferenceWeights;
use crate::tensor::{TensorPlan, TensorSpec};
use crate::weights::{WeightError, WeightSource};

/// Version of the fixture schema this module writes and reads.
pub const FIXTURE_SCHEMA_VERSION: u32 = 1;

/// The prompt length Phase 2 milestone 2 runs.
pub const MILESTONE_PROMPT_TOKENS: usize = 130;

/// The decode step count Phase 2 milestone 2 runs.
pub const MILESTONE_DECODE_STEPS: usize = 8;

/// The Phase-1 logits acceptance bar, as an absolute ceiling on one logit.
///
/// This is the provisional 2e-2 ceiling of the Phase-1 reports (roadmap-to-serving.md, "the bar
/// remains the provisional 2e-2 ceiling"): the device path's last-position logits are compared
/// against this fixture's logits and pass only when the largest absolute difference is at most
/// 2e-2. It is quoted here numerically so no caller invents an ad-hoc tolerance, and the tests use
/// this constant instead of literal 0.1/1.0 values. A 2B/2C run may report the measured delta, but
/// it may not loosen this value without a Lead decision.
pub const LOGITS_ABS_BAR: f32 = 2e-2;

/// Which weight source a fixture was generated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureWeightKind {
    /// The deterministic finite-FP32 source: weight-free, CI only, never evidence about the model.
    FiniteFp32,
    /// The real checkpoint through [crate::loader::ReferenceWeights]: the node golden fixture.
    ReferenceCheckpoint,
}

/// Where a fixture's numbers came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureSource {
    /// Source kind.
    pub kind: FixtureWeightKind,
    /// Seed of the finite source; recorded even for a checkpoint so the field cannot be forgotten.
    pub seed: u64,
    /// Checkpoint directory for [FixtureWeightKind::ReferenceCheckpoint].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<String>,
}

/// Who generated a fixture, and from what.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureProvenance {
    /// The command a person can re-run to reproduce the fixture.
    pub command: String,
    /// Host that generated it. The node golden fixture records it; the weight-free CI fixture
    /// deliberately does not, because it must regenerate byte-identically on any host and binds
    /// no evidence to one machine or one binary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    /// SHA-256 of the generator binary that wrote the file. Same rule as [Self::host]: recorded
    /// for the node golden fixture, deliberately absent for the CI fixture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary_sha256: Option<String>,
    /// Free-form notes that must travel with the numbers.
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Refuses notes on a CI fixture, whose recorded command cannot carry them.
///
/// A note is appended to [FixtureProvenance::notes], which is part of the serialised fixture, so a
/// note changes the fixture's bytes even though it cannot change a number. The CI provenance also
/// records the canonical command that regenerates the fixture, and that command has no `--note`
/// flag. A noted CI fixture would therefore not regenerate from its own recorded command, which is
/// the one property a CI fixture exists for; the generator refuses the combination instead.
///
/// # Errors
///
/// [FixtureError::NotesOnCiProvenance] when `notes` is not empty.
pub fn ci_provenance_admits_notes(notes: &[String]) -> Result<(), FixtureError> {
    if notes.is_empty() {
        Ok(())
    } else {
        Err(FixtureError::NotesOnCiProvenance)
    }
}

/// One fixture step: the prefill, or one decode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureStep {
    /// Whether this is the prefill step or a decode step.
    pub kind: StepKind,
    /// The position the step's logits describe (the last position it consumed).
    pub position: usize,
    /// The token the step selected; the comparison's primary unit.
    pub next_token: Option<u32>,
    /// SHA-256 of the logits as little-endian f32, so a digest comparison needs no float parsing.
    pub logits_sha256: String,
    /// The full logits, when the fixture recorded them (CI records all; the node fixture records
    /// the last prefill position only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits: Option<Vec<f32>>,
}

/// The kind of a fixture step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// The one prefill over the prompt.
    Prefill,
    /// One greedy decode step.
    Decode,
}

/// A canonical KV row digest, reserved for the milestone 2C audit comparison.
///
/// Milestone 2A leaves this list empty: the device store does not exist on this branch yet, and
/// the 2A fixture is about tokens and logits. The field is in the schema now so 2C can add rows
/// without a schema break.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvRowDigest {
    /// Backbone layer.
    pub layer: usize,
    /// Which KV array the row lives in.
    pub kind: KvRowKind,
    /// Compressed entry index (main, indexer) or token position (window).
    pub position: usize,
    /// Window slot, for window rows only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot: Option<usize>,
    /// SHA-256 of the raw stored bytes.
    pub stored_sha256: String,
    /// SHA-256 of the dequantised round-trip.
    pub canonical_sha256: String,
}

/// Which KV array a [KvRowDigest] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvRowKind {
    /// The compressed main KV.
    Main,
    /// The indexer key.
    Indexer,
    /// The sliding-window row.
    Window,
}

/// One host reference result: the fixture file's in-memory form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceFixture {
    /// Schema version; [FIXTURE_SCHEMA_VERSION] for this module.
    pub schema_version: u32,
    /// What wrote the file.
    pub generator: String,
    /// Where the weights came from.
    pub source: FixtureSource,
    /// A short label for the configuration ("tiny" for the CI fixture, "released" for the node one).
    pub config_label: String,
    /// The full model configuration, so a reader needs no code constant to interpret the numbers.
    pub config: ModelConfig,
    /// The prompt token ids, in order.
    pub prompt_tokens: Vec<u32>,
    /// Number of greedy decode steps after the prefill.
    pub decode_steps: usize,
    /// The sliding-window size the exclusion boundary uses.
    pub window: usize,
    /// The compared positions: the last prefill position and every decode position.
    pub compared_positions: Vec<usize>,
    /// The prefill step followed by the decode steps.
    pub steps: Vec<FixtureStep>,
    /// KV digests; empty in 2A, populated by 2C.
    #[serde(default)]
    pub kv_rows: Vec<KvRowDigest>,
    /// Provenance that must travel with the numbers.
    pub provenance: FixtureProvenance,
}

impl SequenceFixture {
    /// Checks the invariants the comparison relies on.
    ///
    /// # Errors
    /// [FixtureError::SchemaVersion] for a version this code cannot interpret,
    /// [FixtureError::Inconsistent] for a fixture whose positions, steps or window do not agree.
    pub fn validate(&self) -> Result<(), FixtureError> {
        if self.schema_version != FIXTURE_SCHEMA_VERSION {
            return Err(FixtureError::SchemaVersion {
                found: self.schema_version,
                expected: FIXTURE_SCHEMA_VERSION,
            });
        }
        if self.prompt_tokens.is_empty() {
            return Err(FixtureError::Inconsistent("the prompt is empty"));
        }
        if self.decode_steps == 0 {
            return Err(FixtureError::Inconsistent("there is no decode step"));
        }
        if self.window != self.config.window_size {
            return Err(FixtureError::Inconsistent(
                "the fixture window does not match its configuration",
            ));
        }
        let expected_positions = compared_positions(self.prompt_tokens.len(), self.decode_steps);
        if self.compared_positions != expected_positions {
            return Err(FixtureError::Inconsistent(
                "compared_positions is not the last prefill position plus the decode positions",
            ));
        }
        if self.steps.len() != 1 + self.decode_steps {
            return Err(FixtureError::Inconsistent(
                "there is not exactly one prefill step and one step per decode",
            ));
        }
        for (index, step) in self.steps.iter().enumerate() {
            let expected_kind = if index == 0 {
                StepKind::Prefill
            } else {
                StepKind::Decode
            };
            if step.kind != expected_kind {
                return Err(FixtureError::Inconsistent(
                    "the step kinds are not one prefill followed by decodes",
                ));
            }
            if step.position != expected_positions[index] {
                return Err(FixtureError::Inconsistent(
                    "a step position does not match the compared set",
                ));
            }
        }
        Ok(())
    }
}

/// How much of each step's logits a fixture records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsCapture {
    /// Every step: used by the tiny CI fixture, whose vocabulary is 128 wide.
    All,
    /// Only the last prefill position: the node golden fixture, per the milestone-2 acceptance.
    PrefillOnly,
    /// None: digests only.
    None,
}

impl LogitsCapture {
    fn stores(self, is_prefill: bool) -> bool {
        match self {
            LogitsCapture::All => true,
            LogitsCapture::PrefillOnly => is_prefill,
            LogitsCapture::None => false,
        }
    }
}

/// The weight source a fixture is generated from.
pub enum FixtureWeights<'a> {
    /// The deterministic finite-FP32 source; only for small configurations.
    FiniteFp32 {
        /// Seed of the deterministic pattern.
        seed: u64,
    },
    /// The real checkpoint through [ReferenceWeights].
    Checkpoint {
        /// Checkpoint directory.
        dir: &'a Path,
        /// Path to the checkpoint's safetensors index.
        index: &'a Path,
    },
}

/// Everything that can stop fixture generation or validation.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    /// A schema version this code cannot interpret.
    #[error("fixture schema version {found} is not supported (expected {expected})")]
    SchemaVersion {
        /// Version found in the file.
        found: u32,
        /// Version this code writes and reads.
        expected: u32,
    },
    /// A zero-token prompt.
    #[error("a sequence fixture needs at least one prompt token")]
    EmptyPrompt,
    /// Zero decode steps.
    #[error("a sequence fixture needs at least one decode step")]
    NoDecodeSteps,
    /// A token id outside the configuration's vocabulary.
    #[error("token {token} is outside the {vocab}-token vocabulary")]
    TokenOutOfVocab {
        /// The offending token.
        token: u32,
        /// The vocabulary size.
        vocab: usize,
    },
    /// A finite source on a configuration that is not cheap enough for CI.
    #[error(
        "the finite-FP32 source is only allowed for a small configuration (dim {dim}, {n_layers} layers); use a checkpoint source for the released model"
    )]
    SyntheticConfigTooLarge {
        /// Configuration width.
        dim: usize,
        /// Configuration layer count.
        n_layers: usize,
    },
    /// The fixture breaks an invariant the comparison relies on.
    #[error("the fixture is inconsistent: {0}")]
    Inconsistent(&'static str),
    /// The reference forward failed.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// A weight read failed.
    #[error(transparent)]
    Weights(#[from] WeightError),
    /// A free-form note on the CI provenance.
    #[error(
        "a CI fixture cannot carry --note: its recorded canonical command has no note flag, so a noted fixture would not regenerate from the command it records"
    )]
    NotesOnCiProvenance,
}

/// A deterministic finite-FP32 weight source: the weight-free CI fixture's weights.
///
/// Values are small, strictly positive and vary per element and per tensor name, so a mis-indexed
/// read is visible rather than masked by uniform weights. It is not the model and never claims to
/// be: its only purpose is to make the harness deterministic on a machine with no checkpoint.
pub struct FiniteFp32Weights {
    specs: Vec<TensorSpec>,
    seed: u64,
}

impl FiniteFp32Weights {
    /// The source for `plan` at `seed`.
    #[must_use]
    pub fn from_plan(plan: &TensorPlan, seed: u64) -> Self {
        Self {
            specs: plan.specs().to_vec(),
            seed,
        }
    }
}

impl WeightSource for FiniteFp32Weights {
    fn names(&self) -> Vec<String> {
        self.specs.iter().map(|spec| spec.name.clone()).collect()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        match self.specs.iter().find(|spec| spec.name == name) {
            Some(spec) => Ok(TensorSpec {
                dtype: DType::F32,
                ..spec.clone()
            }),
            None => Err(WeightError::Missing {
                name: name.to_string(),
            }),
        }
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        let elements = usize::try_from(self.spec(name)?.elements()).expect("a test tensor fits");
        let salt = name.bytes().fold(0u64, |acc, byte| {
            acc.wrapping_mul(31).wrapping_add(u64::from(byte))
        });
        let mut bytes = Vec::with_capacity(elements * 4);
        for index in 0..elements {
            let mixed = (index as u64)
                .wrapping_add(salt)
                .wrapping_add(self.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let hashed = (mixed ^ (mixed >> 13)).wrapping_mul(0x2545_f491_4f6c_dd1d);
            let value = 0.01 + 0.000_1 * (hashed >> 56) as f32;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// The positions a sequence fixture compares: the last prefill position, then every decode.
///
/// This is exactly the set D7 allows. The engine's prefill runs all 40 layers over the whole
/// prompt and returns logits for its last position only; the device CED path runs layers 21..=39
/// over the last 128 tokens, so an older position has no comparable decoder output. Widening the
/// set by even one older position fails [compare_steps] by construction.
#[must_use]
pub fn compared_positions(prompt_tokens: usize, decode_steps: usize) -> Vec<usize> {
    let mut positions = Vec::with_capacity(1 + decode_steps);
    if prompt_tokens == 0 {
        return positions;
    }
    positions.push(prompt_tokens - 1);
    positions.extend(prompt_tokens..prompt_tokens.saturating_add(decode_steps));
    positions
}

/// Positions below this are outside the sliding window and are never compared (D7 boundary).
#[must_use]
pub fn exclusion_boundary(prompt_tokens: usize, window: usize) -> usize {
    prompt_tokens.saturating_sub(window)
}

/// A deterministic prompt of `tokens` ids inside a `vocab`-token vocabulary.
///
/// Recorded in the fixture, so the exact ids are part of the reproduction, not a hidden default.
#[must_use]
pub fn deterministic_prompt(vocab: usize, tokens: usize, seed: u64) -> Vec<u32> {
    assert!(vocab > 0, "the vocabulary is not empty");
    let mut state = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(0x2545_f491_4f6c_dd1d);
    let mut prompt = Vec::with_capacity(tokens);
    for _ in 0..tokens {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        prompt.push(((state >> 33) % vocab as u64) as u32);
    }
    prompt
}

/// The per-tensor element limit the generator hands the engine, by weight source.
///
/// The finite source is synthetic and small by construction, so it keeps the tight 4 Mi-element
/// guard. A real checkpoint needs [CHECKPOINT_TENSOR_LIMIT]: the released `embed.weight` and
/// `head.weight` are 661,913,600 elements each and many attention tensors are 41,943,040, all
/// above [crate::linalg::REFERENCE_TENSOR_LIMIT], which is why the first real-checkpoint run was
/// refused before this function existed.
const fn element_limit(kind: FixtureWeightKind) -> usize {
    match kind {
        FixtureWeightKind::FiniteFp32 => 1 << 22,
        FixtureWeightKind::ReferenceCheckpoint => CHECKPOINT_TENSOR_LIMIT,
    }
}

/// Generates a fixture from a weight source and a prompt.
///
/// # Errors
/// [FixtureError] when the prompt or step count is empty, a token is outside the vocabulary, a
/// finite source is asked to drive a large configuration, or the reference forward fails.
pub fn generate_fixture(
    config: &ModelConfig,
    config_label: &str,
    weights: FixtureWeights<'_>,
    prompt_tokens: &[u32],
    decode_steps: usize,
    capture: LogitsCapture,
    provenance: FixtureProvenance,
) -> Result<SequenceFixture, FixtureError> {
    if prompt_tokens.is_empty() {
        return Err(FixtureError::EmptyPrompt);
    }
    if decode_steps == 0 {
        return Err(FixtureError::NoDecodeSteps);
    }
    for &token in prompt_tokens {
        if token as usize >= config.vocab_size {
            return Err(FixtureError::TokenOutOfVocab {
                token,
                vocab: config.vocab_size,
            });
        }
    }
    let plan = TensorPlan::language_model(config);
    let (source, source_info): (Box<dyn WeightSource>, FixtureSource) = match weights {
        FixtureWeights::FiniteFp32 { seed } => {
            if config.n_layers > 8 || config.dim > 256 {
                return Err(FixtureError::SyntheticConfigTooLarge {
                    dim: config.dim,
                    n_layers: config.n_layers,
                });
            }
            (
                Box::new(FiniteFp32Weights::from_plan(&plan, seed)),
                FixtureSource {
                    kind: FixtureWeightKind::FiniteFp32,
                    seed,
                    checkpoint: None,
                },
            )
        }
        FixtureWeights::Checkpoint { dir, index } => (
            Box::new(ReferenceWeights::open(dir, index, &plan)?),
            FixtureSource {
                kind: FixtureWeightKind::ReferenceCheckpoint,
                seed: 0,
                checkpoint: Some(dir.display().to_string()),
            },
        ),
    };
    let engine = InferenceEngine::new(config.clone(), source, element_limit(source_info.kind))?;
    let mut state = SequenceState::new(config);
    let prefill = engine.prefill(&mut state, prompt_tokens)?;
    let mut steps = Vec::with_capacity(1 + decode_steps);
    steps.push(record_step(
        StepKind::Prefill,
        state.len() - 1,
        &prefill,
        capture.stores(true),
    ));
    let mut next = prefill.next_token.ok_or(FixtureError::Inconsistent(
        "the prefill produced no next token",
    ))?;
    for _ in 0..decode_steps {
        let output = engine.decode(&mut state, next)?;
        steps.push(record_step(
            StepKind::Decode,
            state.len() - 1,
            &output,
            capture.stores(false),
        ));
        next = output.next_token.ok_or(FixtureError::Inconsistent(
            "a decode step produced no next token",
        ))?;
    }
    let fixture = SequenceFixture {
        schema_version: FIXTURE_SCHEMA_VERSION,
        generator: format!("sparkflash-engine sequence_fixture v{FIXTURE_SCHEMA_VERSION}"),
        source: source_info,
        config_label: config_label.to_string(),
        config: config.clone(),
        prompt_tokens: prompt_tokens.to_vec(),
        decode_steps,
        window: config.window_size,
        compared_positions: compared_positions(prompt_tokens.len(), decode_steps),
        steps,
        kv_rows: Vec::new(),
        provenance,
    };
    fixture.validate()?;
    Ok(fixture)
}

fn record_step(
    kind: StepKind,
    position: usize,
    output: &StepOutput,
    store_logits: bool,
) -> FixtureStep {
    FixtureStep {
        kind,
        position,
        next_token: output.next_token,
        logits_sha256: logits_sha256(&output.logits),
        logits: if store_logits {
            Some(output.logits.clone())
        } else {
            None
        },
    }
}

/// SHA-256 of a logits vector as little-endian f32, lowercase hex.
#[must_use]
pub fn logits_sha256(logits: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in logits {
        hasher.update(value.to_le_bytes());
    }
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// One step a device run reports for comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceStep {
    /// The position the step's logits describe.
    pub position: usize,
    /// The token the step selected.
    pub next_token: Option<u32>,
    /// The logits, when the device run captured them.
    pub logits: Option<Vec<f32>>,
    /// The logits digest, when the device run captured one.
    pub logits_sha256: Option<String>,
}

impl DeviceStep {
    /// A step that reports only its position and token.
    #[must_use]
    pub fn tokens(position: usize, next_token: Option<u32>) -> Self {
        Self {
            position,
            next_token,
            logits: None,
            logits_sha256: None,
        }
    }
}

/// Everything the comparison can reject.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CompareError {
    /// The device reported a position outside the compared set.
    #[error(
        "position {position} is not in the CED compared set: only the last prefill position and the decode positions are compared (D7: the engine prefill runs all 40 layers over the prompt, but the device CED path runs layers 21..=39 over the last 128 tokens, so older positions are not comparable)"
    )]
    UnexpectedPosition {
        /// The offending position.
        position: usize,
    },
    /// The device reported a position more than once.
    #[error("position {position} is reported more than once by the device run")]
    DuplicatePosition {
        /// The duplicated position.
        position: usize,
    },
    /// A compared position is missing from the device run.
    #[error("the device run is missing compared position {position}")]
    MissingPosition {
        /// The missing position.
        position: usize,
    },
    /// A token disagrees with the fixture.
    #[error("position {position}: fixture token {expected:?}, device token {found:?}")]
    TokenMismatch {
        /// The position.
        position: usize,
        /// The fixture's token.
        expected: Option<u32>,
        /// The device's token.
        found: Option<u32>,
    },
    /// A logits digest disagrees with the fixture.
    #[error("position {position}: logits digest {found} does not match fixture {expected}")]
    DigestMismatch {
        /// The position.
        position: usize,
        /// The fixture digest.
        expected: String,
        /// The device digest.
        found: String,
    },
    /// The device run captured no logits for a position the fixture compares.
    #[error("position {position}: the fixture records logits but the device run has none")]
    LogitsMissing {
        /// The position.
        position: usize,
    },
    /// A logits vector has a different length than the fixture's.
    #[error("position {position}: logits length {found} does not match fixture {expected}")]
    LogitsShape {
        /// The position.
        position: usize,
        /// The fixture length.
        expected: usize,
        /// The device length.
        found: usize,
    },
    /// A logits value is outside the bar.
    #[error("position {position}: max |logit difference| {max_abs_diff} exceeds the bar {bar}")]
    LogitsMismatch {
        /// The position.
        position: usize,
        /// The largest absolute difference found.
        max_abs_diff: f32,
        /// The bar it exceeded.
        bar: f32,
    },
}

/// Compares the token ids (and logits digests, when the device reports them) on the compared set.
///
/// This is the primary comparison. It checks set equality in both directions, so widening the
/// device run with an older position fails here instead of being quietly ignored.
///
/// # Errors
/// [CompareError] naming the first position that disagrees.
pub fn compare_steps(fixture: &SequenceFixture, device: &[DeviceStep]) -> Result<(), CompareError> {
    let mut seen = Vec::with_capacity(device.len());
    for step in device {
        if seen.contains(&step.position) {
            return Err(CompareError::DuplicatePosition {
                position: step.position,
            });
        }
        seen.push(step.position);
        if !fixture.compared_positions.contains(&step.position) {
            return Err(CompareError::UnexpectedPosition {
                position: step.position,
            });
        }
    }
    for &position in &fixture.compared_positions {
        let found = device
            .iter()
            .find(|step| step.position == position)
            .ok_or(CompareError::MissingPosition { position })?;
        let expected = fixture
            .steps
            .iter()
            .find(|step| step.position == position)
            .ok_or(CompareError::MissingPosition { position })?;
        if expected.next_token != found.next_token {
            return Err(CompareError::TokenMismatch {
                position,
                expected: expected.next_token,
                found: found.next_token,
            });
        }
        if let Some(found_digest) = &found.logits_sha256
            && found_digest != &expected.logits_sha256
        {
            return Err(CompareError::DigestMismatch {
                position,
                expected: expected.logits_sha256.clone(),
                found: found_digest.clone(),
            });
        }
    }
    Ok(())
}

/// Compares the full logits where the fixture recorded them, against an absolute bar.
///
/// This is the second bar: tokens decide pass/fail first, and the logits check the values behind
/// them on the same positions. Steps the fixture did not record are skipped. Callers pass
/// [LOGITS_ABS_BAR] (the provisional Phase-1 2e-2 ceiling) for a device comparison; a 0.0 bar is
/// meaningful only for proving a host fixture regenerates exactly.
///
/// # Errors
/// [CompareError] naming the first position whose logits are missing or out of bar.
pub fn compare_logits(
    fixture: &SequenceFixture,
    device: &[DeviceStep],
    bar: f32,
) -> Result<(), CompareError> {
    for expected in &fixture.steps {
        let Some(want) = &expected.logits else {
            continue;
        };
        let found = device
            .iter()
            .find(|step| step.position == expected.position)
            .ok_or(CompareError::MissingPosition {
                position: expected.position,
            })?;
        let got = found.logits.as_ref().ok_or(CompareError::LogitsMissing {
            position: expected.position,
        })?;
        if got.len() != want.len() {
            return Err(CompareError::LogitsShape {
                position: expected.position,
                expected: want.len(),
                found: got.len(),
            });
        }
        let max_abs_diff = want
            .iter()
            .zip(got)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        if max_abs_diff.is_nan() || max_abs_diff > bar {
            return Err(CompareError::LogitsMismatch {
                position: expected.position,
                max_abs_diff,
                bar,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::REFERENCE_TENSOR_LIMIT;

    fn tiny_fixture(decode_steps: usize) -> SequenceFixture {
        let config = ModelConfig::tiny();
        let prompt = deterministic_prompt(config.vocab_size, 130, 7);
        generate_fixture(
            &config,
            "tiny",
            FixtureWeights::FiniteFp32 { seed: 7 },
            &prompt,
            decode_steps,
            LogitsCapture::All,
            ci_provenance(),
        )
        .expect("the tiny fixture generates")
    }

    fn ci_provenance() -> FixtureProvenance {
        FixtureProvenance {
            command: "cargo run -p sparkflash-engine --example sequence_fixture -- --config tiny"
                .to_string(),
            host: None,
            binary_sha256: None,
            notes: vec!["weight-free finite-FP32 CI fixture".to_string()],
        }
    }

    fn device_from(fixture: &SequenceFixture) -> Vec<DeviceStep> {
        fixture
            .steps
            .iter()
            .map(|step| DeviceStep {
                position: step.position,
                next_token: step.next_token,
                logits: step.logits.clone(),
                logits_sha256: Some(step.logits_sha256.clone()),
            })
            .collect()
    }

    #[test]
    fn the_compared_set_is_the_last_prefill_position_and_the_decodes() {
        assert_eq!(
            compared_positions(130, 8),
            vec![129, 130, 131, 132, 133, 134, 135, 136, 137]
        );
        assert_eq!(compared_positions(1, 1), vec![0, 1]);
        assert!(compared_positions(0, 8).is_empty());
    }

    #[test]
    fn the_exclusion_boundary_is_prompt_minus_window() {
        assert_eq!(exclusion_boundary(130, 128), 2);
        assert_eq!(exclusion_boundary(130, 8), 122);
        assert_eq!(exclusion_boundary(64, 128), 0);
    }

    #[test]
    fn a_matching_tiny_fixture_passes_token_and_logits_comparison() {
        let fixture = tiny_fixture(8);
        assert_eq!(fixture.steps.len(), 9);
        assert_eq!(fixture.prompt_tokens.len(), 130);
        let device = device_from(&fixture);
        compare_steps(&fixture, &device).expect("tokens match");
        compare_logits(&fixture, &device, 0.0).expect("identical logits match a zero bar");
    }

    #[test]
    fn widening_the_set_with_an_older_position_fails_and_names_d7() {
        let fixture = tiny_fixture(8);
        let mut device = device_from(&fixture);
        device.push(DeviceStep::tokens(128, None));
        let error = compare_steps(&fixture, &device).expect_err("an older position fails");
        assert!(matches!(
            error,
            CompareError::UnexpectedPosition { position: 128 }
        ));
        let text = error.to_string();
        assert!(text.contains("D7"), "the error names D7: {text}");
        assert!(text.contains("128"), "the error names the position: {text}");

        // Even a position inside the window but before the last prefill is not compared.
        let mut device = device_from(&fixture);
        device.push(DeviceStep::tokens(0, None));
        assert!(matches!(
            compare_steps(&fixture, &device).expect_err("position 0 fails"),
            CompareError::UnexpectedPosition { position: 0 }
        ));
    }

    #[test]
    fn a_missing_compared_position_fails() {
        let fixture = tiny_fixture(8);
        let mut device = device_from(&fixture);
        device.retain(|step| step.position != 133);
        assert!(matches!(
            compare_steps(&fixture, &device).expect_err("a missing decode fails"),
            CompareError::MissingPosition { position: 133 }
        ));
    }

    #[test]
    fn a_wrong_token_fails_the_primary_bar() {
        let fixture = tiny_fixture(8);
        let mut device = device_from(&fixture);
        let step = device
            .iter_mut()
            .find(|step| step.position == 131)
            .expect("position 131");
        step.next_token = step.next_token.map(|token| token ^ 1);
        let error = compare_steps(&fixture, &device).expect_err("a wrong token fails");
        assert!(matches!(
            error,
            CompareError::TokenMismatch { position: 131, .. }
        ));
        // The logits bar alone would not catch it; tokens are the primary bar.
        compare_logits(&fixture, &device, LOGITS_ABS_BAR).expect("logits unchanged");
    }

    #[test]
    fn the_logits_bar_is_absolute_and_second() {
        let fixture = tiny_fixture(8);
        let mut device = device_from(&fixture);
        let step = device
            .iter_mut()
            .find(|step| step.position == 129)
            .expect("the prefill position");
        let logits = step.logits.as_mut().expect("recorded logits");
        logits[0] += LOGITS_ABS_BAR * 1.5;
        assert!(matches!(
            compare_logits(&fixture, &device, LOGITS_ABS_BAR)
                .expect_err("a 1.5x bar perturbation is outside the bar"),
            CompareError::LogitsMismatch { position: 129, .. }
        ));
        compare_logits(&fixture, &device, LOGITS_ABS_BAR * 2.0)
            .expect("a 1.5x bar perturbation is inside twice the bar");
    }

    #[test]
    fn missing_device_logits_fail_the_logits_bar_but_not_the_token_bar() {
        let fixture = tiny_fixture(8);
        let mut device = device_from(&fixture);
        let step = device
            .iter_mut()
            .find(|step| step.position == 130)
            .expect("a decode position");
        step.logits = None;
        // Digest and token are still present, so the primary comparison passes.
        compare_steps(&fixture, &device).expect("tokens and digests match");
        assert!(matches!(
            compare_logits(&fixture, &device, LOGITS_ABS_BAR)
                .expect_err("recorded logits are required"),
            CompareError::LogitsMissing { position: 130 }
        ));
    }

    #[test]
    fn a_fixture_with_a_wrong_schema_version_is_refused() {
        let mut fixture = tiny_fixture(8);
        fixture.schema_version = FIXTURE_SCHEMA_VERSION + 1;
        assert!(matches!(
            fixture.validate().expect_err("the version is wrong"),
            FixtureError::SchemaVersion { .. }
        ));
    }

    #[test]
    fn a_fixture_with_a_widened_compared_set_is_refused() {
        let mut fixture = tiny_fixture(8);
        fixture.compared_positions.push(128);
        assert!(matches!(
            fixture
                .validate()
                .expect_err("a widened set is inconsistent"),
            FixtureError::Inconsistent(_)
        ));
    }

    #[test]
    fn the_prompt_is_deterministic_and_inside_the_vocabulary() {
        let first = deterministic_prompt(128, 130, 7);
        let second = deterministic_prompt(128, 130, 7);
        let other = deterministic_prompt(128, 130, 8);
        assert_eq!(first, second);
        assert_ne!(first, other);
        assert!(first.iter().all(|&token| (token as usize) < 128));
        assert_eq!(first.len(), 130);
    }

    #[test]
    fn the_finite_source_is_refused_for_a_large_configuration() {
        let config = ModelConfig::v41_flash().expect("the released config is valid");
        let prompt = deterministic_prompt(config.vocab_size, 130, 7);
        let error = generate_fixture(
            &config,
            "released",
            FixtureWeights::FiniteFp32 { seed: 7 },
            &prompt,
            8,
            LogitsCapture::None,
            ci_provenance(),
        )
        .expect_err("a released configuration needs a checkpoint");
        assert!(matches!(
            error,
            FixtureError::SyntheticConfigTooLarge { .. }
        ));
    }

    #[test]
    fn the_checkpoint_limit_admits_the_released_embedding_but_the_small_shape_limit_does_not() {
        let config = ModelConfig::v41_flash().expect("the released configuration is consistent");
        let plan = TensorPlan::language_model(&config);
        let spec = plan
            .get("embed.weight")
            .expect("the plan names embed.weight");
        let elements = spec.elements();
        assert_eq!(elements, 661_913_600);
        assert!(
            elements > REFERENCE_TENSOR_LIMIT as u128,
            "the released embedding must be above the small-shape guard"
        );
        assert!(
            elements <= CHECKPOINT_TENSOR_LIMIT as u128,
            "the released embedding must fit the checkpoint limit"
        );
        // head.weight ties the embedding exactly, so it is the second tensor this guard admits.
        let head = plan.get("head.weight").expect("the plan names head.weight");
        assert_eq!(head.elements(), elements);
    }

    #[test]
    fn the_generator_selects_the_checkpoint_limit_only_for_a_real_checkpoint() {
        assert_eq!(element_limit(FixtureWeightKind::FiniteFp32), 1 << 22);
        assert_eq!(
            element_limit(FixtureWeightKind::ReferenceCheckpoint),
            CHECKPOINT_TENSOR_LIMIT
        );
        const {
            assert!(CHECKPOINT_TENSOR_LIMIT > REFERENCE_TENSOR_LIMIT);
        };
    }

    #[test]
    fn empty_prompt_and_zero_decode_steps_are_refused() {
        let config = ModelConfig::tiny();
        assert!(matches!(
            generate_fixture(
                &config,
                "tiny",
                FixtureWeights::FiniteFp32 { seed: 7 },
                &[],
                8,
                LogitsCapture::None,
                ci_provenance(),
            )
            .expect_err("no prompt"),
            FixtureError::EmptyPrompt
        ));
        let prompt = deterministic_prompt(config.vocab_size, 4, 7);
        assert!(matches!(
            generate_fixture(
                &config,
                "tiny",
                FixtureWeights::FiniteFp32 { seed: 7 },
                &prompt,
                0,
                LogitsCapture::None,
                ci_provenance(),
            )
            .expect_err("no decode"),
            FixtureError::NoDecodeSteps
        ));
    }

    #[test]
    fn a_ci_fixture_refuses_notes_its_recorded_command_cannot_carry() {
        ci_provenance_admits_notes(&[]).expect("an un-noted CI fixture is canonical");
        assert!(matches!(
            ci_provenance_admits_notes(&["fix commit abc123".to_string()])
                .expect_err("a note cannot be part of the CI command"),
            FixtureError::NotesOnCiProvenance
        ));
    }
}
