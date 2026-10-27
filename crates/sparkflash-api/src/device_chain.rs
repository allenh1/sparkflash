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

//! Typed consumer scaffold for the four-rank device token envelope.
//!
//! Device-lead's TP=4 token run writes, per rank, a tp4-forward-embed-r{rank}.json record (the
//! gathered prompt row plus the layer-39 head-tail evidence block) and a
//! tp4-token-r{rank}.json record (the forward status, the decoded nine-position token batch and
//! the optional logits dump). This module is the serving side's typed reader for that envelope: it
//! loads the four rank pairs, cross-checks the metadata and the head argmax, and either exposes the
//! claimed token stream or names exactly why no token can be claimed.
//!
//! Everything here is a consumer contract over records that do not exist in a claimed form yet.
//! The schemas are taken from the m1y7 token window
//! (/home/user/ev-session-20261020/window/m1y7/rank{0..3}/), which is a refused run: the loop
//! records the head argmax but claims no token because the reference attention body is not wired.
//! The reader therefore parses the same shapes in both states, and treats a claim as an assertion
//! the device side has to make (token_claimed plus a complete nine-position batch), never as
//! something inferred from partial evidence. No device, GPU or vLLM action is implied or
//! performed here; the tests use synthetic records only and are labelled UNMEASURED.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

/// Schema of the per-rank forward/embedding record written by the TP=4 token run.
pub const FORWARD_EMBED_SCHEMA: &str = "sparkflash-tp4-forward-embed/1";
/// Schema of the per-rank token record written by the TP=4 token run.
pub const TOKEN_SCHEMA: &str = "sparkflash-tp4-token/1";
/// Schema of the head-tail evidence block nested in the forward/embedding record.
pub const HEAD_TAIL_SCHEMA: &str = "sparkflash-head-tail-evidence/1";
/// The TP=4 world size the serving consumer expects for a four-rank token envelope.
pub const FOUR_RANK_WORLD: u32 = 4;
/// The nine decode positions the token window runs (--decode-steps 9 in the m1y7 argv).
pub const DECODE_POSITIONS: u32 = 9;

/// One rank's pair of records: the forward/embedding evidence and the token batch.
#[derive(Debug, Clone)]
pub struct RankEnvelope {
    /// tp4-forward-embed-r{rank}.json.
    pub forward_embed: ForwardEmbedRecord,
    /// tp4-token-r{rank}.json.
    pub token: TokenRecord,
}

/// The metadata every rank of one token envelope has to agree on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvelopeMetadata {
    /// Token id of the first dumped prefill row (prompt_token in the embedding record).
    pub prompt_token: u32,
    /// Prompt length in tokens (prompt_len, 130 in the m1y7 window).
    pub prompt_len: u32,
    /// Position of the first decode step (first_step_position, 129 in the m1y7 window).
    pub first_step_position: u32,
    /// Number of decode positions the device loop produced (decode_positions, 9).
    pub decode_positions: u32,
    /// The reference ids the loop compares its own nine tokens against.
    pub reference_ids: Vec<u32>,
    /// The prompt token ids the window was recorded with, when the device side wrote them.
    ///
    /// Empty for the m1y7 records (the field did not exist yet). The serving consumer refuses to
    /// replay an envelope whose prompt cannot be matched, so the S8 probe needs this field: see
    /// docs/archive/build/serving-s7-s8-plan.md, wiring point W2.
    pub prompt_ids: Vec<u32>,
    /// Whether any rank's forward record asserts a claimed token.
    pub token_claimed: bool,
    /// The full teacher-forced input ids (prompt plus decode rows) the window fed, when written.
    ///
    /// The device writer adds this replay vector next to prompt_ids (commit b6f69251): slice
    /// prompt_len..prompt_len+decode_positions must equal the claimed token ids. Empty on records
    /// that predate the field.
    pub forward_ids: Vec<u32>,
    /// The --prefill-limit the window ran with, when recorded (None is the full prefill run).
    pub prefill_limit: Option<u32>,
    /// The device-side evidence completion label, when recorded.
    pub evidence_written: Option<String>,
}

/// The typed four-rank token envelope: four rank pairs, or a named refusal.
#[derive(Debug, Clone)]
pub struct DeviceChainEnvelope {
    dir: PathBuf,
    world: u32,
    records: Vec<RankEnvelope>,
    metadata: EnvelopeMetadata,
}

/// One claimed decode position.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimedPosition {
    /// Zero-based position inside the nine-token window.
    pub index: usize,
    /// Absolute model position of this token.
    pub position: u32,
    /// The argmax token id.
    pub token_id: u32,
    /// The f32 argmax logit value recorded for the position.
    pub argmax_value: f64,
    /// The top-1 margin recorded for the position, when the record carries one.
    pub top1_margin: Option<f64>,
    /// The per-position logits digest, when the record carries one.
    pub logits_sha256: Option<String>,
    /// Per-position certification, from the logits dump when present.
    pub certified: Option<bool>,
    /// Per-position claimability, from the logits dump when present.
    pub claimable: Option<bool>,
}

/// How the consumer decided the window is claimed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimMode {
    /// Every rank's forward record asserts token_claimed. This is the S8 acceptance form.
    Asserted,
    /// The writer did not assert the flag, but the complete loop evidence is present: the four
    /// ranks agree on the decoded ids, every position is claimable, all nine ids match the
    /// teacher-forced reference, and the fed forward_ids rows equal the decoded ids. This is the
    /// S7 rehearsal form; it is never an S8 acceptance substitute.
    Derived,
}

impl ClaimMode {
    /// The wire label for the claim form.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Asserted => "explicit",
            Self::Derived => "derived",
        }
    }
}

/// A validated claimed token envelope.
#[derive(Debug, Clone)]
pub struct ClaimedEnvelope {
    /// How the claim was established.
    pub claim_mode: ClaimMode,
    /// The cross-rank metadata.
    pub metadata: EnvelopeMetadata,
    /// The nine claimed decode positions, in order.
    pub tokens: Vec<ClaimedPosition>,
    /// The logits dump evidence the claim exposed, when one was published.
    pub logits_dump: Option<LogitsDumpEvidence>,
}

/// Why the device chain cannot claim a token yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimRefusal {
    /// Human-readable reasons; an empty list never occurs in a refusal.
    pub reasons: Vec<String>,
}

impl fmt::Display for ClaimRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.reasons.join("; "))
    }
}

impl std::error::Error for ClaimRefusal {}

/// A typed envelope-read failure.
#[derive(Debug, thiserror::Error)]
pub enum DeviceChainError {
    /// A record could not be read.
    #[error("read {path}: {source}")]
    Io {
        /// The offending path.
        path: PathBuf,
        /// The I/O error.
        source: std::io::Error,
    },
    /// A record is not the JSON this consumer expects.
    #[error("parse {path}: {source}")]
    Json {
        /// The offending path.
        path: PathBuf,
        /// The serde error.
        source: serde_json::Error,
    },
    /// A rank pair is missing.
    #[error("rank {rank} record is missing: {path}")]
    MissingRank {
        /// The missing rank.
        rank: u32,
        /// The path that was expected.
        path: PathBuf,
    },
    /// The record declares the wrong schema.
    #[error("{path}: schema {actual:?}, expected {expected:?}")]
    Schema {
        /// The offending path.
        path: PathBuf,
        /// The expected schema constant.
        expected: &'static str,
        /// What the record carries.
        actual: String,
    },
    /// The record's declared rank is not the rank its file name says.
    #[error("{path}: declares rank {actual}, expected {expected}")]
    RankMismatch {
        /// The offending path.
        path: PathBuf,
        /// The rank the file name encodes.
        expected: u32,
        /// The rank the record declares.
        actual: u32,
    },
    /// The rank records disagree on the world size.
    #[error("rank records disagree on world: rank {rank} says {actual}, rank 0 says {expected}")]
    WorldMismatch {
        /// The rank that disagrees.
        rank: u32,
        /// Its world size.
        actual: u32,
        /// Rank 0's world size.
        expected: u32,
    },
    /// The rank records disagree on metadata the envelope needs to be one request.
    #[error(
        "rank records disagree on {field}: rank {rank} says {actual:?}, rank 0 says {expected:?}"
    )]
    MetadataMismatch {
        /// The metadata field name.
        field: &'static str,
        /// The rank that disagrees.
        rank: u32,
        /// Its value.
        actual: String,
        /// Rank 0's value.
        expected: String,
    },
    /// The nested head block is not the head-tail evidence schema.
    #[error("{path}: head block schema {actual:?}, expected {expected:?}")]
    HeadSchema {
        /// The offending path.
        path: PathBuf,
        /// The expected schema constant.
        expected: &'static str,
        /// What the record carries.
        actual: String,
    },
}

/// One rank's tp4-forward-embed-r{rank}.json.
#[derive(Debug, Clone, Deserialize)]
pub struct ForwardEmbedRecord {
    /// Record schema.
    pub schema: String,
    /// Stage name.
    #[serde(default)]
    pub stage: String,
    /// Rank this record was written on.
    pub rank: u32,
    /// TP world size.
    pub world: u32,
    /// Token id of the first dumped prefill row.
    pub prompt_token: u32,
    /// Prompt length in tokens.
    pub prompt_len: u32,
    /// Position of the first decode step.
    pub first_step_position: u32,
    /// Decode positions the loop planned.
    pub decode_positions: u32,
    /// The prompt token ids the window was recorded with, when written.
    #[serde(default)]
    pub prompt_ids: Vec<u32>,
    /// Reference ids the loop compares its own tokens against.
    #[serde(default)]
    pub reference_ids: Vec<u32>,
    /// Whether the device side asserts a claimed token.
    #[serde(default)]
    pub token_claimed: bool,
    /// The full teacher-forced input ids (prompt plus decode rows), when written.
    #[serde(default)]
    pub forward_ids: Vec<u32>,
    /// The --prefill-limit the window ran with, when written.
    #[serde(default)]
    pub prefill_limit: Option<u32>,
    /// The device-side evidence completion label, when written.
    #[serde(default)]
    pub evidence_written: Option<String>,
    /// The loop's per-position token batch, when written (the four-rank consensus).
    #[serde(default)]
    pub tokens: Option<ForwardTokenBatch>,
    /// The layer-39 head-tail evidence block.
    pub head: HeadTailEvidence,
    /// Final stream digest, when written.
    #[serde(default)]
    pub final_stream_sha256: Option<String>,
    /// Final normed (f32) digest, when written.
    #[serde(default)]
    pub final_normed_sha256: Option<String>,
    /// Final normed BF16 digest, when written.
    #[serde(default)]
    pub final_normed_bf16_sha256: Option<String>,
    /// Operator note from the device side.
    #[serde(default)]
    pub note: Option<String>,
}

/// The loop's per-position token batch carried by a forward/embedding record.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ForwardTokenBatch {
    /// The four-rank consensus token ids.
    #[serde(default)]
    pub ids: Vec<u32>,
    /// The absolute model positions of those ids.
    #[serde(default)]
    pub positions: Vec<u32>,
    /// Whether the four ranks agreed on every id.
    #[serde(default)]
    pub consensus: bool,
    /// Per-position head claimability.
    #[serde(default)]
    pub claimable: Vec<bool>,
    /// Per-position teacher-forced reference comparison; null when no reference was supplied.
    #[serde(default)]
    pub matches_reference: Vec<Option<bool>>,
    /// Per-position argmax logit values.
    #[serde(default)]
    pub argmax_logits: Vec<f64>,
    /// Per-position top-1 margins.
    #[serde(default)]
    pub top1_margins: Vec<f64>,
}

/// The head block of a forward/embedding record.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadTailEvidence {
    /// Block schema (sparkflash-head-tail-evidence/1).
    pub schema: String,
    /// Stage name.
    #[serde(default)]
    pub stage: String,
    /// Rank that wrote the block.
    pub rank: u32,
    /// The per-rank head logits partitions.
    pub head: HeadLogitsBlock,
    /// The exact-f32 gather.
    pub gather: HeadGather,
    /// The reduced argmax.
    pub argmax: HeadArgmax,
    /// Operator note.
    #[serde(default)]
    pub note: Option<String>,
}

/// The head.head partition digest block.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadLogitsBlock {
    /// Kernel name.
    #[serde(default)]
    pub kernel: String,
    /// Launch grid.
    #[serde(default)]
    pub grid: Vec<u32>,
    /// Launch block.
    #[serde(default)]
    pub block: Vec<u32>,
    /// Per-rank partition digests.
    #[serde(default)]
    pub ranks: Vec<RankHeadLogits>,
}

/// One rank's logits partition digest.
#[derive(Debug, Clone, Deserialize)]
pub struct RankHeadLogits {
    /// Rank of the partition.
    pub rank: u32,
    /// Digest of the partition's f32 logits.
    #[serde(default)]
    pub logits_sha256: Option<String>,
    /// Number of f32 values in the partition.
    #[serde(default)]
    pub logits_f32_values: Option<u64>,
    /// Number of values that passed certification.
    #[serde(default)]
    pub certified_count: u64,
    /// Digest of the certified subset.
    #[serde(default)]
    pub certified_sha256: Option<String>,
    /// Digest of the partition's per-block max.
    #[serde(default)]
    pub block_max_sha256: Option<String>,
    /// Digest of the partition's per-block argmax.
    #[serde(default)]
    pub block_argmax_sha256: Option<String>,
}

/// The head.gather block.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadGather {
    /// Gather route (for example exact-f32-word-gather).
    #[serde(default)]
    pub route: Option<String>,
    /// Gathered f32 value count (the vocabulary size).
    #[serde(default)]
    pub gathered_logits_f32_values: u64,
    /// Digest of the gathered f32 logits.
    #[serde(default)]
    pub gathered_logits_sha256: Option<String>,
}

/// The head.argmax block: the one token the device side would claim.
#[derive(Debug, Clone, Deserialize)]
pub struct HeadArgmax {
    /// Argmax token id.
    #[serde(default)]
    pub token: Option<u32>,
    /// Argmax logit value.
    #[serde(default)]
    pub value: Option<f64>,
    /// Whether the value passed certification.
    #[serde(default)]
    pub certified: bool,
    /// Whether the device side considers the token claimable.
    #[serde(default)]
    pub claimable: bool,
    /// Host-twin comparison token, when the window ran one.
    #[serde(default)]
    pub twin_token: Option<u32>,
    /// Host-twin comparison value, when the window ran one.
    #[serde(default)]
    pub twin_value: Option<f64>,
}

/// One rank's tp4-token-r{rank}.json.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenRecord {
    /// Record schema.
    pub schema: String,
    /// Stage name.
    #[serde(default)]
    pub stage: String,
    /// Rank this record was written on.
    pub rank: u32,
    /// TP world size.
    pub world: u32,
    /// Forward status (ran, refused, ...).
    #[serde(default)]
    pub forward: String,
    /// The decoded token batch.
    #[serde(default)]
    pub tokens: TokenPositions,
    /// Decode positions the loop actually ran.
    #[serde(default)]
    pub positions_run: u32,
    /// The refusal/error text, when the forward did not run.
    #[serde(default)]
    pub error: Option<String>,
    /// The optional nine-position logits dump (inline object or a relative JSON file name).
    #[serde(default)]
    pub logits_dump: Option<LogitsDumpRef>,
    /// The head rule block.
    #[serde(default)]
    pub head: Option<TokenHeadRule>,
    /// Whether a hybrid sequence executed.
    #[serde(default)]
    pub hybrid_sequence_executed: bool,
}

/// The head rule block of a token record.
#[derive(Debug, Clone, Deserialize)]
pub struct TokenHeadRule {
    /// Reduction rule.
    #[serde(default)]
    pub rule: Option<String>,
    /// Vocabulary size.
    #[serde(default)]
    pub vocab: Option<u64>,
    /// Tie-break rule.
    #[serde(default)]
    pub tie_break: Option<String>,
}

/// The decoded token batch (tokens in a token record).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TokenPositions {
    /// Decoded token ids, one per decode position.
    #[serde(default)]
    pub ids: Vec<u32>,
    /// Absolute positions, one per decode position.
    #[serde(default)]
    pub positions: Vec<u32>,
    /// First decode position, when written.
    #[serde(default)]
    pub start_position: Option<u32>,
    /// Prompt length, when written.
    #[serde(default)]
    pub prompt_len: Option<u32>,
    /// Argmax logit value per decode position.
    #[serde(default)]
    pub argmax_logits: Vec<f64>,
    /// Top-1 margin per decode position.
    #[serde(default)]
    pub top1_margins: Vec<f64>,
    /// Per-position logits digests, as one digest or one per position.
    #[serde(default)]
    pub logits_sha256: Option<LogitsHashes>,
}

/// tokens.logits_sha256 as observed: absent/null, one digest, or one digest per position.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum LogitsHashes {
    /// One digest for the whole batch.
    One(String),
    /// One digest per decode position.
    Many(Vec<String>),
}

impl LogitsHashes {
    /// The digest for one decode position, when the record names one.
    #[must_use]
    pub fn at(&self, index: usize) -> Option<String> {
        match self {
            Self::One(one) => Some(one.clone()),
            Self::Many(many) => many.get(index).cloned(),
        }
    }
}

/// logits_dump as the device writer emits it, as an inline consumer dump, or as a JSON file name.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum LogitsDumpRef {
    /// The device token writer's dump block: evidence paths, digests and the covered positions.
    Device(DeviceLogitsDump),
    /// An inline consumer-shaped dump object.
    Inline(LogitsDump),
    /// A JSON file name relative to the envelope directory.
    Path(String),
}

/// The device token writer's logits_dump block (tp4_token.rs / logits_dump.rs).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct DeviceLogitsDump {
    /// The JSONL sidecar path on the node that wrote the window.
    #[serde(default)]
    pub path: String,
    /// The raw f32 binary sidecar path on the node that wrote the window.
    #[serde(default)]
    pub bin: Option<String>,
    /// Digest of the JSONL sidecar.
    #[serde(default)]
    pub sha256: Option<String>,
    /// Digest of the f32 binary sidecar.
    #[serde(default)]
    pub bin_sha256: Option<String>,
    /// The binary dtype (f32-le).
    #[serde(default)]
    pub dtype: Option<String>,
    /// The vocabulary size (129,280).
    #[serde(default)]
    pub vocab: Option<u64>,
    /// The absolute positions the dump covers.
    #[serde(default)]
    pub positions: Vec<u32>,
    /// The number of dumped rows.
    #[serde(default)]
    pub rows: Option<u64>,
}

/// The logits dump a claim exposes: either the device writer's block or a consumer dump.
#[derive(Debug, Clone)]
pub enum LogitsDumpEvidence {
    /// The device token writer's block.
    Device(DeviceLogitsDump),
    /// A consumer-shaped inline or file dump.
    Inline(LogitsDump),
}

impl LogitsDumpEvidence {
    /// The number of covered positions, when the evidence declares it.
    #[must_use]
    pub fn positions(&self) -> Option<usize> {
        match self {
            Self::Device(dump) => Some(dump.positions.len()),
            Self::Inline(dump) => Some(dump.positions.len()),
        }
    }

    /// The device writer's block, when this is one.
    #[must_use]
    pub fn device(&self) -> Option<&DeviceLogitsDump> {
        match self {
            Self::Device(dump) => Some(dump),
            Self::Inline(_) => None,
        }
    }
}

/// The nine-position logits dump.
///
/// This is the consumer shape for the dump device-lead is producing. The m1y7 window writes
/// null (the forward refused), so every field beyond the position list is optional: a dump that
/// only carries digests is enough to cross-check the token batch, while a dump that carries full
/// rows names them by logits_f32 (a relative file) plus a value count.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LogitsDump {
    /// Optional dump schema.
    #[serde(default)]
    pub schema: Option<String>,
    /// Decode positions the dump covers.
    #[serde(default)]
    pub decode_positions: Option<u32>,
    /// One entry per decode position.
    #[serde(default)]
    pub positions: Vec<LogitsPosition>,
}

/// One position of the logits dump.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LogitsPosition {
    /// Absolute model position.
    #[serde(default)]
    pub position: Option<u32>,
    /// The argmax token id at this position.
    #[serde(default)]
    pub token: Option<u32>,
    /// The argmax logit value.
    #[serde(default)]
    pub value: Option<f64>,
    /// The top-1 margin.
    #[serde(default)]
    pub top1_margin: Option<f64>,
    /// Whether the position passed certification.
    #[serde(default)]
    pub certified: Option<bool>,
    /// Whether the position is claimable.
    #[serde(default)]
    pub claimable: Option<bool>,
    /// Digest of the position's f32 logits.
    #[serde(default)]
    pub logits_sha256: Option<String>,
    /// Relative file name of the raw f32 logits row, when the dump carries one.
    #[serde(default)]
    pub logits_f32: Option<String>,
    /// Number of f32 values in logits_f32.
    #[serde(default)]
    pub logits_f32_values: Option<u64>,
}

impl DeviceChainEnvelope {
    /// Loads and cross-checks the four-rank envelope from one token-window directory.
    ///
    /// The load succeeds for a refused run too: the m1y7 window's forward: refused record set is
    /// a valid envelope that simply has no claim. Use claim() for the claim decision.
    ///
    /// # Errors
    /// Returns [DeviceChainError] when a rank pair is missing, a record is not the expected
    /// schema, or the ranks disagree on the request metadata or world size.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, DeviceChainError> {
        let dir = dir.as_ref().to_path_buf();
        let rank0_path = embed_path(&dir, 0);
        let rank0_embed: ForwardEmbedRecord = read_json(&rank0_path)?;
        check_schema(&rank0_path, FORWARD_EMBED_SCHEMA, &rank0_embed.schema)?;
        check_head_schema(&rank0_path, &rank0_embed.head.schema)?;
        let world = rank0_embed.world;
        if world != FOUR_RANK_WORLD {
            return Err(DeviceChainError::WorldMismatch {
                rank: 0,
                actual: world,
                expected: FOUR_RANK_WORLD,
            });
        }
        let mut records = Vec::with_capacity(world as usize);
        for rank in 0..world {
            let embed_path = embed_path(&dir, rank);
            let token_path = token_path(&dir, rank);
            if !embed_path.is_file() {
                return Err(DeviceChainError::MissingRank {
                    rank,
                    path: embed_path,
                });
            }
            if !token_path.is_file() {
                return Err(DeviceChainError::MissingRank {
                    rank,
                    path: token_path,
                });
            }
            let forward_embed: ForwardEmbedRecord = read_json(&embed_path)?;
            let token: TokenRecord = read_json(&token_path)?;
            check_schema(&embed_path, FORWARD_EMBED_SCHEMA, &forward_embed.schema)?;
            check_schema(&token_path, TOKEN_SCHEMA, &token.schema)?;
            check_head_schema(&embed_path, &forward_embed.head.schema)?;
            check_rank(&embed_path, rank, forward_embed.rank)?;
            check_rank(&token_path, rank, token.rank)?;
            check_world(rank, world, forward_embed.world)?;
            check_world(rank, world, token.world)?;
            check_metadata(rank, &rank0_embed, &forward_embed)?;
            records.push(RankEnvelope {
                forward_embed,
                token,
            });
        }
        let metadata = EnvelopeMetadata {
            prompt_token: rank0_embed.prompt_token,
            prompt_len: rank0_embed.prompt_len,
            first_step_position: rank0_embed.first_step_position,
            decode_positions: rank0_embed.decode_positions,
            reference_ids: rank0_embed.reference_ids.clone(),
            prompt_ids: rank0_embed.prompt_ids.clone(),
            token_claimed: records
                .iter()
                .any(|record| record.forward_embed.token_claimed),
            forward_ids: rank0_embed.forward_ids.clone(),
            prefill_limit: rank0_embed.prefill_limit,
            evidence_written: rank0_embed.evidence_written.clone(),
        };
        Ok(Self {
            dir,
            world,
            records,
            metadata,
        })
    }

    /// The envelope directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The TP world size the envelope covers.
    #[must_use]
    pub fn world(&self) -> u32 {
        self.world
    }

    /// The cross-rank metadata.
    #[must_use]
    pub fn metadata(&self) -> &EnvelopeMetadata {
        &self.metadata
    }

    /// One rank's record pair.
    #[must_use]
    pub fn rank(&self, rank: u32) -> Option<&RankEnvelope> {
        self.records.get(rank as usize)
    }

    /// The logits dump the envelope exposes, when a rank wrote one.
    ///
    /// # Errors
    /// Returns the load error when a path-shaped consumer dump cannot be read.
    pub fn logits_dump(&self) -> Result<Option<LogitsDumpEvidence>, DeviceChainError> {
        let Some(record) = self.records.first() else {
            return Ok(None);
        };
        match record.token.logits_dump.as_ref() {
            None => Ok(None),
            Some(LogitsDumpRef::Device(dump)) => Ok(Some(LogitsDumpEvidence::Device(dump.clone()))),
            Some(LogitsDumpRef::Inline(dump)) => Ok(Some(LogitsDumpEvidence::Inline(dump.clone()))),
            Some(LogitsDumpRef::Path(name)) => {
                let path = dump_path(&self.dir, name);
                let dump: LogitsDump = read_json(&path)?;
                Ok(Some(LogitsDumpEvidence::Inline(dump)))
            }
        }
    }

    /// Decides whether the device side claims the nine-token stream.
    ///
    /// This is the strict S8 form: every rank's forward record must assert token_claimed.
    ///
    /// # Errors
    /// Returns a [ClaimRefusal] naming every concrete reason the envelope cannot claim.
    pub fn claim(&self) -> Result<ClaimedEnvelope, ClaimRefusal> {
        self.claim_with(ClaimMode::Asserted)
    }

    /// Decides whether the envelope claims the nine-token stream under one claim form.
    ///
    /// Both forms require the complete nine-position batch: forward=run on every rank, no error,
    /// positions_run == decode_positions == 9, consecutive absolute positions starting at the
    /// recorded first step, nine ids/positions/argmax values that every rank agrees on, the head
    /// argmax equal to the first token, the head-tail partitions summing to the gathered logits,
    /// and a nine-position logits dump that agrees with the batch. ClaimMode::Asserted additionally
    /// requires the device writer's token_claimed assertion on every rank; ClaimMode::Derived
    /// requires the loop's consensus/claimable/reference-match evidence and the fed forward_ids.
    ///
    /// # Errors
    /// Returns a [ClaimRefusal] naming every concrete reason the envelope cannot claim.
    pub fn claim_with(&self, mode: ClaimMode) -> Result<ClaimedEnvelope, ClaimRefusal> {
        let mut reasons = self.claim_reasons(mode);
        let dump = match self.logits_dump() {
            Ok(dump) => dump,
            Err(error) => {
                reasons.push(format!("logits dump unavailable: {error}"));
                None
            }
        };
        if let Some(dump) = dump.as_ref() {
            self.check_dump(dump, &mut reasons);
        }
        if !reasons.is_empty() {
            return Err(ClaimRefusal { reasons });
        }
        let Some(first) = self.records.first() else {
            return Err(ClaimRefusal {
                reasons: vec!["envelope has no rank records".to_owned()],
            });
        };
        let batch = first.token.tokens.clone();
        let batch_tokens = first.forward_embed.tokens.as_ref();
        let tokens = batch
            .ids
            .iter()
            .enumerate()
            .map(|(index, token_id)| ClaimedPosition {
                index,
                position: batch.positions[index],
                token_id: *token_id,
                argmax_value: batch.argmax_logits[index],
                top1_margin: batch.top1_margins.get(index).copied(),
                logits_sha256: batch
                    .logits_sha256
                    .as_ref()
                    .and_then(|hashes| hashes.at(index)),
                certified: match dump.as_ref() {
                    Some(LogitsDumpEvidence::Inline(dump)) => dump
                        .positions
                        .get(index)
                        .and_then(|position| position.certified),
                    _ => None,
                },
                claimable: batch_tokens
                    .and_then(|batch| batch.claimable.get(index).copied())
                    .or_else(|| match dump.as_ref() {
                        Some(LogitsDumpEvidence::Inline(dump)) => dump
                            .positions
                            .get(index)
                            .and_then(|position| position.claimable),
                        _ => None,
                    }),
            })
            .collect();
        Ok(ClaimedEnvelope {
            claim_mode: mode,
            metadata: self.metadata.clone(),
            tokens,
            logits_dump: dump,
        })
    }

    fn claim_reasons(&self, mode: ClaimMode) -> Vec<String> {
        let mut reasons = Vec::new();
        let decode_positions = self.metadata.decode_positions;
        if mode == ClaimMode::Derived {
            for (rank, record) in self.records.iter().enumerate() {
                self.check_derived_claim(rank as u32, record, &mut reasons);
            }
        } else {
            if !self.metadata.token_claimed {
                reasons.push(
                    "no rank's forward record asserts token_claimed=true; the window is not explicitly claimed".to_owned(),
                );
            }
            for (rank, record) in self.records.iter().enumerate() {
                if !record.forward_embed.token_claimed {
                    reasons.push(format!(
                        "rank {rank}: forward record does not assert token_claimed"
                    ));
                }
            }
        }
        if decode_positions != DECODE_POSITIONS {
            reasons.push(format!(
                "decode_positions is {decode_positions}, expected {DECODE_POSITIONS}"
            ));
        }
        if self.metadata.reference_ids.len() != decode_positions as usize {
            reasons.push(format!(
                "reference_ids has {} entries, expected {decode_positions}",
                self.metadata.reference_ids.len()
            ));
        }
        for (rank, record) in self.records.iter().enumerate() {
            let rank = rank as u32;
            let token = &record.token;
            if token.forward != "run" {
                reasons.push(format!(
                    "rank {rank}: token record forward = {:?}, expected \"run\"",
                    token.forward
                ));
            }
            if let Some(error) = token.error.as_ref() {
                reasons.push(format!("rank {rank}: token record error = {error}"));
            }
            if token.positions_run != decode_positions {
                reasons.push(format!(
                    "rank {rank}: positions_run is {}, expected {decode_positions}",
                    token.positions_run
                ));
            }
            check_token_arrays(rank, decode_positions, &token.tokens, &mut reasons);
            // A full (unlimited) window's head positions are the absolute positions
            // prompt_len-1..prompt_len+decode_positions-2, so the first one is first_step_position.
            // A prefill-limited replay window runs positions from 0 and is not claimable anyway.
            if self.metadata.prefill_limit.is_none()
                && token.tokens.positions.first().copied()
                    != Some(self.metadata.first_step_position)
            {
                reasons.push(format!(
                    "rank {rank}: token positions start at {:?}, expected first_step_position {} from the forward record",
                    token.tokens.positions.first(),
                    self.metadata.first_step_position
                ));
            }
            if rank == 0 && token.logits_dump.is_none() {
                reasons.push(
                    "rank 0: token record carries no logits_dump; a claimed window must publish its dump".to_owned(),
                );
            }
            self.check_forward_batch(rank, record, &mut reasons);
        }
        if self.records.len() > 1 {
            let first_ids = self.records[0].token.tokens.ids.clone();
            let first_positions = self.records[0].token.tokens.positions.clone();
            let first_values = self.records[0].token.tokens.argmax_logits.clone();
            for (rank, record) in self.records.iter().enumerate().skip(1) {
                if record.token.tokens.ids != first_ids {
                    reasons.push(format!(
                        "rank {rank}: token ids {:?} differ from rank 0's {first_ids:?}",
                        record.token.tokens.ids
                    ));
                }
                if record.token.tokens.positions != first_positions {
                    reasons.push(format!(
                        "rank {rank}: token positions {:?} differ from rank 0's {first_positions:?}",
                        record.token.tokens.positions
                    ));
                }
                if record.token.tokens.argmax_logits != first_values {
                    reasons.push(format!("rank {rank}: argmax values differ from rank 0's"));
                }
            }
            let first_bin = self.records[0]
                .token
                .logits_dump
                .as_ref()
                .and_then(device_dump)
                .and_then(|dump| dump.bin_sha256.clone());
            for (rank, record) in self.records.iter().enumerate().skip(1) {
                let bin = record
                    .token
                    .logits_dump
                    .as_ref()
                    .and_then(device_dump)
                    .and_then(|dump| dump.bin_sha256.clone());
                if let (Some(expected), Some(actual)) = (first_bin.as_ref(), bin.as_ref())
                    && expected != actual
                {
                    reasons.push(format!(
                        "rank {rank}: logits dump bin_sha256 differs from rank 0's"
                    ));
                }
            }
        }
        for (rank, record) in self.records.iter().enumerate() {
            self.check_head_rank(rank as u32, record, &mut reasons);
        }
        reasons
    }

    fn check_head_rank(&self, rank: u32, record: &RankEnvelope, reasons: &mut Vec<String>) {
        let head = &record.forward_embed.head;
        if head.rank != rank {
            reasons.push(format!(
                "rank {rank}: head block declares rank {}",
                head.rank
            ));
        }
        let world = self.world;
        if head.head.ranks.len() != world as usize {
            reasons.push(format!(
                "rank {rank}: head block lists {} partitions, expected {world}",
                head.head.ranks.len()
            ));
        }
        let mut partition_ranks: Vec<u32> = head
            .head
            .ranks
            .iter()
            .map(|partition| partition.rank)
            .collect();
        partition_ranks.sort_unstable();
        let expected_partition_ranks: Vec<u32> = (0..world).collect();
        if partition_ranks != expected_partition_ranks {
            reasons.push(format!(
                "rank {rank}: head partitions cover {partition_ranks:?}, expected {expected_partition_ranks:?}"
            ));
        }
        let partition_values: u64 = head
            .head
            .ranks
            .iter()
            .filter_map(|partition| partition.logits_f32_values)
            .sum();
        if partition_values != head.gather.gathered_logits_f32_values {
            reasons.push(format!(
                "rank {rank}: partition logits values {partition_values} do not sum to the gathered {}",
                head.gather.gathered_logits_f32_values
            ));
        }
        let argmax = &head.argmax;
        if !argmax.claimable {
            reasons.push(format!("rank {rank}: head argmax is not claimable"));
        }
        match argmax.token {
            None => reasons.push(format!("rank {rank}: head argmax carries no token")),
            Some(token) => {
                // The head block is the last dumped position's head, so its token is the last
                // decoded token, not the first.
                let last_token = record.token.tokens.ids.last().copied();
                if last_token != Some(token) {
                    reasons.push(format!(
                        "rank {rank}: head argmax token {token} does not match the last decoded token {last_token:?}"
                    ));
                }
            }
        }
        if let (Some(head_value), Some(last_value)) = (
            argmax.value,
            record.token.tokens.argmax_logits.last().copied(),
        ) && (head_value - last_value).abs() > 1e-6
        {
            reasons.push(format!(
                "rank {rank}: head argmax value {head_value} does not match the last decoded argmax value {last_value}"
            ));
        }
        if rank > 0 {
            let rank0 = &self.records[0].forward_embed.head.argmax;
            if rank0.token != argmax.token {
                reasons.push(format!(
                    "rank {rank}: head argmax token {:?} differs from rank 0's {:?}",
                    argmax.token, rank0.token
                ));
            }
        }
    }

    fn check_derived_claim(&self, rank: u32, record: &RankEnvelope, reasons: &mut Vec<String>) {
        if record.forward_embed.token_claimed {
            return;
        }
        let Some(batch) = record.forward_embed.tokens.as_ref() else {
            reasons.push(format!(
                "rank {rank}: forward record carries no loop token batch, so a derived claim is impossible"
            ));
            return;
        };
        if !batch.consensus {
            reasons.push(format!(
                "rank {rank}: the loop token batch does not assert four-rank consensus"
            ));
        }
        if batch.ids.len() != DECODE_POSITIONS as usize {
            reasons.push(format!(
                "rank {rank}: the loop token batch has {} ids, expected {DECODE_POSITIONS}",
                batch.ids.len()
            ));
        }
        if batch.claimable.len() != batch.ids.len() || !batch.claimable.iter().all(|flag| *flag) {
            reasons.push(format!(
                "rank {rank}: not every loop position is claimable: {:?}",
                batch.claimable
            ));
        }
        if self.metadata.prefill_limit.is_some() {
            reasons.push(format!(
                "rank {rank}: the window ran with --prefill-limit, so it is a replay smoke, not the full nine-position acceptance run"
            ));
        }
        if record.forward_embed.prompt_ids.len() != record.forward_embed.prompt_len as usize {
            reasons.push(format!(
                "rank {rank}: prompt_ids has {} entries, expected prompt_len {}",
                record.forward_embed.prompt_ids.len(),
                record.forward_embed.prompt_len
            ));
        }
        if self.metadata.reference_ids.is_empty() {
            reasons.push(format!(
                "rank {rank}: the window records no reference ids, so no derived claim is possible; run the acceptance window with --forward-reference-ids"
            ));
        } else if batch.matches_reference.len() != self.metadata.reference_ids.len()
            || !batch
                .matches_reference
                .iter()
                .all(|matched| *matched == Some(true))
        {
            reasons.push(format!(
                "rank {rank}: the decoded ids do not all match the reference ids: {:?}",
                batch.matches_reference
            ));
        }
        self.check_forward_ids(rank, record, reasons);
        let evidence = record
            .forward_embed
            .evidence_written
            .as_deref()
            .unwrap_or("");
        if !evidence.contains("token-consensus-complete") {
            reasons.push(format!(
                "rank {rank}: evidence_written {evidence:?} does not record token-consensus-complete"
            ));
        }
    }

    fn check_forward_ids(&self, rank: u32, record: &RankEnvelope, reasons: &mut Vec<String>) {
        let Some(batch) = record.forward_embed.tokens.as_ref() else {
            return;
        };
        if batch.ids.is_empty() {
            return;
        }
        if record.forward_embed.forward_ids.is_empty() {
            reasons.push(format!(
                "rank {rank}: forward record carries no forward_ids, so the decoded ids cannot be tied to the fed rows"
            ));
            return;
        }
        // A full run feeds prompt_len + decode_positions - 1 teacher-forced rows and dumps the last
        // decode_positions of them; the first dumped row's input is forward_ids[prompt_len - 1],
        // which is the record's prompt_token. The decoded ids are outputs, not inputs, so they are
        // checked against reference_ids by the reference-match rule, never against forward_ids.
        let expected = record.forward_embed.prompt_len as usize + batch.ids.len() - 1;
        if record.forward_embed.forward_ids.len() != expected {
            reasons.push(format!(
                "rank {rank}: forward_ids has {} rows, expected prompt_len + decode_positions - 1 = {expected}",
                record.forward_embed.forward_ids.len()
            ));
        }
        let first_input = record
            .forward_embed
            .forward_ids
            .get(record.forward_embed.prompt_len.saturating_sub(1) as usize)
            .copied();
        if first_input != Some(record.forward_embed.prompt_token) {
            reasons.push(format!(
                "rank {rank}: forward_ids[prompt_len - 1] is {first_input:?}, expected prompt_token {:?} (the first dumped row's input)",
                record.forward_embed.prompt_token
            ));
        }
    }

    fn check_forward_batch(&self, rank: u32, record: &RankEnvelope, reasons: &mut Vec<String>) {
        let Some(batch) = record.forward_embed.tokens.as_ref() else {
            return;
        };
        let token = &record.token;
        if batch.ids != token.tokens.ids {
            reasons.push(format!(
                "rank {rank}: forward token batch ids {:?} differ from the token record ids {:?}",
                batch.ids, token.tokens.ids
            ));
        }
        if batch.positions != token.tokens.positions {
            reasons.push(format!(
                "rank {rank}: forward token batch positions {:?} differ from the token record positions {:?}",
                batch.positions, token.tokens.positions
            ));
        }
        if !batch.claimable.is_empty() && !batch.claimable.iter().all(|flag| *flag) {
            reasons.push(format!(
                "rank {rank}: the forward token batch is not claimable at every position: {:?}",
                batch.claimable
            ));
        }
        if !self.metadata.reference_ids.is_empty()
            && !batch.matches_reference.is_empty()
            && !batch
                .matches_reference
                .iter()
                .all(|matched| *matched == Some(true))
        {
            reasons.push(format!(
                "rank {rank}: the forward token batch does not match the reference at every position: {:?}",
                batch.matches_reference
            ));
        }
        self.check_forward_ids(rank, record, reasons);
    }

    fn check_dump(&self, dump: &LogitsDumpEvidence, reasons: &mut Vec<String>) {
        let decode_positions = self.metadata.decode_positions as usize;
        match dump {
            LogitsDumpEvidence::Device(dump) => {
                if dump.positions.len() != decode_positions {
                    reasons.push(format!(
                        "logits dump covers {} positions, expected {decode_positions}",
                        dump.positions.len()
                    ));
                }
                for (index, position) in dump.positions.iter().enumerate() {
                    let expected = self.records[0].token.tokens.positions.get(index).copied();
                    if let Some(expected) = expected
                        && *position != expected
                    {
                        reasons.push(format!(
                            "logits dump position {index} is {position}, expected the batch position {expected}"
                        ));
                    }
                }
                if let Some(rows) = dump.rows
                    && rows as usize != decode_positions
                {
                    reasons.push(format!(
                        "logits dump declares {rows} rows, expected {decode_positions}"
                    ));
                }
                if dump.vocab != Some(129280) {
                    reasons.push(format!(
                        "logits dump vocab {:?}, expected 129280",
                        dump.vocab
                    ));
                }
                if let Some(dtype) = dump.dtype.as_deref()
                    && dtype != "f32-le"
                {
                    reasons.push(format!("logits dump dtype {dtype:?}, expected \"f32-le\""));
                }
                if dump.path.is_empty() {
                    reasons.push("logits dump names no JSONL path".to_owned());
                }
                match dump.bin.as_deref() {
                    Some(bin) => {
                        let expected_bin = format!("{}.bin", dump.path);
                        if bin != expected_bin {
                            reasons.push(format!(
                                "logits dump bin {bin:?} is not path + .bin ({expected_bin:?})"
                            ));
                        }
                    }
                    None => reasons.push("logits dump carries no bin sidecar".to_owned()),
                }
                for (label, digest) in [
                    ("sha256", dump.sha256.as_deref()),
                    ("bin_sha256", dump.bin_sha256.as_deref()),
                ] {
                    match digest {
                        Some(value) if is_sha256_hex(value) => {}
                        Some(value) => reasons.push(format!(
                            "logits dump {label} {value:?} is not 64 hex characters"
                        )),
                        None => reasons.push(format!("logits dump carries no {label}")),
                    }
                }
            }
            LogitsDumpEvidence::Inline(dump) => {
                if let Some(count) = dump.decode_positions
                    && count as usize != decode_positions
                {
                    reasons.push(format!(
                        "logits dump covers {count} positions, expected {decode_positions}"
                    ));
                }
                if dump.positions.len() != decode_positions {
                    reasons.push(format!(
                        "logits dump has {} positions, expected {decode_positions}",
                        dump.positions.len()
                    ));
                }
                for (index, position) in dump.positions.iter().enumerate() {
                    let Some(expected) = self.records[0].token.tokens.ids.get(index).copied()
                    else {
                        break;
                    };
                    if let Some(token) = position.token
                        && token != expected
                    {
                        reasons.push(format!(
                            "logits dump position {index} token {token} differs from the batch token {expected}"
                        ));
                    }
                    let expected_position =
                        self.records[0].token.tokens.positions.get(index).copied();
                    if let (Some(position), Some(expected_position)) =
                        (position.position, expected_position)
                        && position != expected_position
                    {
                        reasons.push(format!(
                            "logits dump position {index} absolute position {position} differs from the batch position {expected_position}"
                        ));
                    }
                }
            }
        }
    }
}

/// The device dump block, when a record's logits_dump is the writer's object form.
fn device_dump(dump_ref: &LogitsDumpRef) -> Option<&DeviceLogitsDump> {
    match dump_ref {
        LogitsDumpRef::Device(dump) => Some(dump),
        LogitsDumpRef::Inline(_) | LogitsDumpRef::Path(_) => None,
    }
}

/// Whether a string is a 64-character lowercase or uppercase hex digest.
fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit())
}
fn check_token_arrays(
    rank: u32,
    decode_positions: u32,
    tokens: &TokenPositions,
    reasons: &mut Vec<String>,
) {
    let expected = decode_positions as usize;
    if tokens.ids.len() != expected {
        reasons.push(format!(
            "rank {rank}: {} token ids, expected {expected}",
            tokens.ids.len()
        ));
    }
    if tokens.positions.len() != expected {
        reasons.push(format!(
            "rank {rank}: {} token positions, expected {expected}",
            tokens.positions.len()
        ));
    }
    if tokens.argmax_logits.len() != expected {
        reasons.push(format!(
            "rank {rank}: {} argmax logit values, expected {expected}",
            tokens.argmax_logits.len()
        ));
    }
    if !tokens.top1_margins.is_empty() && tokens.top1_margins.len() != expected {
        reasons.push(format!(
            "rank {rank}: {} top1 margins, expected {expected}",
            tokens.top1_margins.len()
        ));
    }
    for (index, pair) in tokens.positions.windows(2).enumerate() {
        if pair[1] != pair[0] + 1 {
            reasons.push(format!(
                "rank {rank}: token positions are not consecutive at index {index}: {:?}",
                pair
            ));
            break;
        }
    }
    if let Some(start) = tokens.start_position
        && tokens.positions.first() != Some(&start)
    {
        reasons.push(format!(
            "rank {rank}: start_position {start} does not match the first position {:?}",
            tokens.positions.first()
        ));
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, DeviceChainError> {
    let bytes = std::fs::read(path).map_err(|source| DeviceChainError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| DeviceChainError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn check_schema(path: &Path, expected: &'static str, actual: &str) -> Result<(), DeviceChainError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DeviceChainError::Schema {
            path: path.to_path_buf(),
            expected,
            actual: actual.to_owned(),
        })
    }
}

fn check_head_schema(path: &Path, actual: &str) -> Result<(), DeviceChainError> {
    if actual == HEAD_TAIL_SCHEMA {
        Ok(())
    } else {
        Err(DeviceChainError::HeadSchema {
            path: path.to_path_buf(),
            expected: HEAD_TAIL_SCHEMA,
            actual: actual.to_owned(),
        })
    }
}

fn check_rank(path: &Path, expected: u32, actual: u32) -> Result<(), DeviceChainError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DeviceChainError::RankMismatch {
            path: path.to_path_buf(),
            expected,
            actual,
        })
    }
}

fn check_world(rank: u32, expected: u32, actual: u32) -> Result<(), DeviceChainError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DeviceChainError::WorldMismatch {
            rank,
            actual,
            expected,
        })
    }
}

fn check_metadata(
    rank: u32,
    expected: &ForwardEmbedRecord,
    actual: &ForwardEmbedRecord,
) -> Result<(), DeviceChainError> {
    for (field, same, expected_value, actual_value) in [
        (
            "prompt_token",
            expected.prompt_token == actual.prompt_token,
            expected.prompt_token.to_string(),
            actual.prompt_token.to_string(),
        ),
        (
            "prompt_len",
            expected.prompt_len == actual.prompt_len,
            expected.prompt_len.to_string(),
            actual.prompt_len.to_string(),
        ),
        (
            "first_step_position",
            expected.first_step_position == actual.first_step_position,
            expected.first_step_position.to_string(),
            actual.first_step_position.to_string(),
        ),
        (
            "decode_positions",
            expected.decode_positions == actual.decode_positions,
            expected.decode_positions.to_string(),
            actual.decode_positions.to_string(),
        ),
        (
            "reference_ids",
            expected.reference_ids == actual.reference_ids,
            format!("{:?}", expected.reference_ids),
            format!("{:?}", actual.reference_ids),
        ),
        (
            "prompt_ids",
            expected.prompt_ids == actual.prompt_ids,
            format!("{:?}", expected.prompt_ids),
            format!("{:?}", actual.prompt_ids),
        ),
        (
            "forward_ids",
            expected.forward_ids == actual.forward_ids,
            format!("{:?}", expected.forward_ids.len()),
            format!("{:?}", actual.forward_ids.len()),
        ),
        (
            "prefill_limit",
            expected.prefill_limit == actual.prefill_limit,
            format!("{:?}", expected.prefill_limit),
            format!("{:?}", actual.prefill_limit),
        ),
    ] {
        if !same {
            return Err(DeviceChainError::MetadataMismatch {
                field,
                rank,
                actual: actual_value,
                expected: expected_value,
            });
        }
    }
    Ok(())
}

/// Locates one rank's record inside one layout.
///
/// The device token windows write rank{rank}/tp4-*.json (m1y7 does); a flattened evidence copy puts
/// the four pairs next to each other. Both layouts are accepted, rank directories first because
/// that is what the device loop produces.
fn rank_record_path(dir: &Path, rank: u32, name: String) -> PathBuf {
    let nested = dir.join(format!("rank{rank}")).join(&name);
    if nested.is_file() {
        return nested;
    }
    dir.join(name)
}

fn embed_path(dir: &Path, rank: u32) -> PathBuf {
    rank_record_path(dir, rank, format!("tp4-forward-embed-r{rank}.json"))
}

fn token_path(dir: &Path, rank: u32) -> PathBuf {
    rank_record_path(dir, rank, format!("tp4-token-r{rank}.json"))
}

/// Resolves a path-shaped logits dump: the window root first, then rank 0's directory.
fn dump_path(dir: &Path, name: &str) -> PathBuf {
    let nested = dir.join("rank0").join(name);
    if nested.is_file() {
        return nested;
    }
    dir.join(name)
}

/// Rejects a logits payload path that escapes the envelope directory.
///
/// The scaffold does not read the raw f32 rows yet (the S8 probe can compare digests), but a
/// consumer that does must never follow a name outside the envelope directory. This helper is the
/// one place that rule lives.
///
/// # Errors
/// Returns the offending name when it is absolute or contains a parent-directory component.
pub fn check_relative_payload(name: &str) -> Result<(), String> {
    let path = Path::new(name);
    if path.is_absolute() {
        return Err(format!("payload path {name:?} is absolute"));
    }
    if path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(format!(
            "payload path {name:?} escapes the envelope directory"
        ));
    }
    Ok(())
}

/// Returns the JSON value of a record, for forward-compatible diagnostics.
///
/// The device side will add fields over time; this keeps a consumer able to print the raw record
/// without pinning it to the typed struct.
///
/// # Errors
/// Propagates the read/parse failure.
pub fn read_raw(path: &Path) -> Result<Value, DeviceChainError> {
    read_json(path)
}
#[cfg(any(test, feature = "device-token-fixtures"))]
pub mod fixtures {
    //! Synthetic four-rank token-envelope records for tests only (UNMEASURED).
    //!
    //! The numbers are paper values: nine ids, nine consecutive positions and one argmax value
    //! per position. No device run produced them, and no test in this module reads a device file.

    use super::*;
    use serde_json::{Value, json};

    /// The synthetic nine-token batch.
    pub const DECODE_IDS: [u32; 9] = [11, 22, 33, 44, 55, 66, 77, 88, 99];
    /// The synthetic prompt length.
    pub const PROMPT_LEN: u32 = 130;
    /// The synthetic first decode position.
    pub const FIRST_STEP_POSITION: u32 = 129;
    /// The synthetic dump schema label.
    pub const DUMP_SCHEMA: &str = "sparkflash-tp4-logits-dump/1";

    /// The synthetic prompt token ids the window was recorded with.
    pub fn prompt_ids() -> Vec<u32> {
        (0..PROMPT_LEN).map(|offset| 1000 + offset).collect()
    }

    /// The synthetic device-writer logits_dump block (shape of tp4_token.rs, values synthetic).
    pub fn device_dump_value() -> Value {
        json!({
            "path": "synthetic/tp4-logits-r0.jsonl",
            "bin": "synthetic/tp4-logits-r0.jsonl.bin",
            "sha256": "a".repeat(64),
            "bin_sha256": "b".repeat(64),
            "dtype": "f32-le",
            "vocab": 129280,
            "positions": (FIRST_STEP_POSITION..FIRST_STEP_POSITION + DECODE_POSITIONS).collect::<Vec<_>>(),
            "rows": DECODE_POSITIONS,
        })
    }

    /// Writes a claimed-envelope fixture whose writer did not assert token_claimed (derive mode).
    pub fn write_derived_envelope(dir: &Path) {
        write_envelope(dir, true);
        for rank in 0..FOUR_RANK_WORLD {
            let path = dir.join(format!("tp4-forward-embed-r{rank}.json"));
            rewrite_value(&path, |record| {
                record["token_claimed"] = Value::Bool(false);
            });
        }
    }

    /// Writes a synthetic four-rank envelope into dir.
    pub fn write_envelope(dir: &Path, claimed: bool) {
        std::fs::create_dir_all(dir).expect("test envelope directory");
        let ids: Vec<u32> = if claimed {
            DECODE_IDS.to_vec()
        } else {
            Vec::new()
        };
        let positions: Vec<u32> = if claimed {
            (FIRST_STEP_POSITION..FIRST_STEP_POSITION + DECODE_POSITIONS).collect()
        } else {
            Vec::new()
        };
        let values: Vec<f64> = if claimed {
            (0..DECODE_POSITIONS)
                .map(|index| 10.0 + index as f64)
                .collect()
        } else {
            Vec::new()
        };
        let margins: Vec<f64> = if claimed {
            vec![1.0; DECODE_POSITIONS as usize]
        } else {
            Vec::new()
        };
        let hashes: Value = if claimed {
            json!(
                ids.iter()
                    .map(|id| format!("digest-{id}"))
                    .collect::<Vec<_>>()
            )
        } else {
            Value::Null
        };
        // A full run feeds prompt_len + decode_positions - 1 teacher-forced rows; the first dumped
        // row's input is forward_ids[prompt_len - 1], which the fixture sets to its prompt_token.
        let mut fed_ids: Vec<u32> = prompt_ids()
            .into_iter()
            .take(PROMPT_LEN as usize - 1)
            .collect();
        fed_ids.push(20462);
        fed_ids.extend_from_slice(&DECODE_IDS[..DECODE_POSITIONS as usize - 1]);
        let claimable_flags: Vec<bool> = vec![
            true;
            if claimed {
                DECODE_POSITIONS as usize
            } else {
                0
            }
        ];
        let matches: Vec<Option<bool>> = if claimed {
            ids.iter().map(|_| Some(true)).collect()
        } else {
            Vec::new()
        };
        for rank in 0..FOUR_RANK_WORLD {
            let partitions: Vec<Value> = (0..FOUR_RANK_WORLD)
                .map(|partition| {
                    json!({
                        "rank": partition,
                        "logits_sha256": format!("rank-{partition}-logits"),
                        "logits_f32_values": 32320,
                        "certified_count": 0,
                        "certified_sha256": "certified",
                        "block_max_sha256": "block-max",
                        "block_argmax_sha256": "block-argmax",
                    })
                })
                .collect();
            let embed = json!({
                "schema": FORWARD_EMBED_SCHEMA,
                "stage": "synthetic-test",
                "rank": rank,
                "world": FOUR_RANK_WORLD,
                "prompt_token": 20462,
                "prompt_len": PROMPT_LEN,
                "first_step_position": FIRST_STEP_POSITION,
                "decode_positions": DECODE_POSITIONS,
                "reference_ids": DECODE_IDS,
                "prompt_ids": prompt_ids(),
                "forward_ids": fed_ids,
                "prefill_limit": Value::Null,
                "evidence_written": if claimed {
                    "loop-complete, position-complete, head-complete, token-consensus-complete"
                } else {
                    "refused"
                },
                "prefill_tokens": PROMPT_LEN,
                "token_claimed": claimed,
                "tokens": {
                    "ids": ids,
                    "positions": positions,
                    "consensus": claimed,
                    "claimable": claimable_flags,
                    "matches_reference": matches,
                    "argmax_logits": values,
                    "top1_margins": margins,
                },
                "final_stream_sha256": "synthetic-stream-digest",
                "head": {
                    "schema": HEAD_TAIL_SCHEMA,
                    "stage": "synthetic-test",
                    "rank": rank,
                    "head": {
                        "kernel": "synthetic",
                        "grid": [1, 1, 1],
                        "block": [1, 1, 1],
                        "ranks": partitions,
                    },
                    "gather": {
                        "route": "exact-f32-word-gather",
                        "gathered_logits_f32_values": 129280,
                        "gathered_logits_sha256": "gathered",
                    },
                    "argmax": {
                        "token": DECODE_IDS[DECODE_POSITIONS as usize - 1],
                        "value": 10.0 + f64::from(DECODE_POSITIONS - 1),
                        "certified": claimed,
                        "claimable": true,
                        "twin_token": null,
                        "twin_value": null,
                    },
                    "note": "synthetic test record; UNMEASURED",
                },
                "note": "synthetic test record; UNMEASURED",
            });
            let token = json!({
                "schema": TOKEN_SCHEMA,
                "stage": "tp4-chain",
                "loop_step": "C4-token",
                "rank": rank,
                "world": FOUR_RANK_WORLD,
                "forward": if claimed { "run" } else { "refused" },
                "tokens": {
                    "ids": ids,
                    "positions": positions,
                    "start_position": if claimed { json!(FIRST_STEP_POSITION) } else { Value::Null },
                    "prompt_len": if claimed { json!(PROMPT_LEN) } else { Value::Null },
                    "argmax_logits": values,
                    "top1_margins": margins,
                    "logits_sha256": hashes,
                },
                "positions_run": if claimed { DECODE_POSITIONS } else { 0 },
                "step_ms": {"samples": 0, "mean_ms": 0.0, "worst_ms": 0.0},
                "wall_ms": [],
                "reduce_calls_per_step": 0,
                "head": {"rule": "gathered-f32-argmax", "vocab": 129280, "tie_break": "lowest-id"},
                "logits_dump": if claimed && rank == 0 {
                    device_dump_value()
                } else {
                    Value::Null
                },
                "error": if claimed { Value::Null } else { json!("synthetic refusal: the reference attention body is not wired") },
                "hybrid_sequence_executed": false,
            });
            write_value(&dir.join(format!("tp4-forward-embed-r{rank}.json")), &embed);
            write_value(&dir.join(format!("tp4-token-r{rank}.json")), &token);
        }
    }

    /// Writes one JSON value.
    pub fn write_value(path: &Path, value: &Value) {
        std::fs::write(path, serde_json::to_vec_pretty(value).expect("json"))
            .expect("synthetic record write");
    }

    /// Writes a claimed fixture window recorded for an arbitrary prompt.
    ///
    /// The stock [`write_envelope`] records the synthetic 130-id prompt. A chat request's prompt is
    /// whatever the released template renders and the checkpoint tokenizer produces, so the
    /// exact-prompt rule needs a window whose metadata describes *that* prompt. Every derived field
    /// is recomputed (prompt length, first step position, forwarded rows, batch positions, and the
    /// rank-0 logits dump's positions), so the result passes the same consistency checks a
    /// device-written window passes — it is still a paper record, UNMEASURED.
    pub fn write_envelope_for_prompt(dir: &Path, prompt: &[u32]) {
        assert!(
            !prompt.is_empty(),
            "a window needs at least one prompt token"
        );
        write_envelope(dir, true);
        let prompt_len = u32::try_from(prompt.len()).expect("prompt fits in u32");
        let first_step_position = prompt_len - 1;
        let positions: Vec<u32> =
            (first_step_position..first_step_position + DECODE_POSITIONS).collect();
        // The full run feeds prompt_len + decode_positions - 1 teacher-forced rows: the prompt,
        // then the prompt's last token (the record's prompt_token), then the first eight decoded
        // ids. forward_ids[prompt_len - 1] is therefore the prompt_token, as the claim check
        // requires.
        let mut forward_ids: Vec<u32> = prompt[..prompt.len() - 1].to_vec();
        forward_ids.push(20462);
        forward_ids.extend_from_slice(&DECODE_IDS[..DECODE_POSITIONS as usize - 1]);
        for rank in 0..FOUR_RANK_WORLD {
            let embed = dir.join(format!("tp4-forward-embed-r{rank}.json"));
            rewrite_value(&embed, |record| {
                record["prompt_len"] = json!(prompt_len);
                record["first_step_position"] = json!(first_step_position);
                record["prompt_ids"] = json!(prompt);
                record["prefill_tokens"] = json!(prompt_len);
                record["forward_ids"] = json!(forward_ids);
                record["tokens"]["positions"] = json!(positions);
            });
            let token = dir.join(format!("tp4-token-r{rank}.json"));
            rewrite_value(&token, |record| {
                record["tokens"]["start_position"] = json!(first_step_position);
                record["tokens"]["prompt_len"] = json!(prompt_len);
                record["tokens"]["positions"] = json!(positions);
                if rank == 0 {
                    record["logits_dump"]["positions"] = json!(positions);
                }
            });
        }
    }

    /// Moves a flat fixture envelope into the device window's rank{rank}/ layout.
    pub fn relocate_to_rank_directories(dir: &Path) {
        for rank in 0..FOUR_RANK_WORLD {
            let rank_dir = dir.join(format!("rank{rank}"));
            std::fs::create_dir_all(&rank_dir).expect("rank directory");
            for name in [
                format!("tp4-forward-embed-r{rank}.json"),
                format!("tp4-token-r{rank}.json"),
            ] {
                std::fs::rename(dir.join(&name), rank_dir.join(&name))
                    .expect("move synthetic record into the rank directory");
            }
        }
    }

    /// Reads one JSON value.
    pub fn read_value(path: &Path) -> Value {
        serde_json::from_slice(&std::fs::read(path).expect("synthetic record read"))
            .expect("synthetic record json")
    }

    /// Rewrites one JSON value through a mutation closure.
    pub fn rewrite_value(path: &Path, mutate: impl FnOnce(&mut Value)) {
        let mut value = read_value(path);
        mutate(&mut value);
        write_value(path, &value);
    }

    /// Writes rank 0's logits_dump field.
    pub fn write_rank0_dump(dir: &Path, dump: Value) {
        let path = dir.join("tp4-token-r0.json");
        rewrite_value(&path, |token| token["logits_dump"] = dump);
    }

    /// A synthetic nine-position logits dump; consistent=false drifts every token by 1000.
    pub fn dump_value(consistent: bool) -> Value {
        let positions: Vec<Value> = (0..DECODE_POSITIONS)
            .map(|index| {
                let token = if consistent {
                    DECODE_IDS[index as usize]
                } else {
                    DECODE_IDS[index as usize] + 1000
                };
                json!({
                    "position": FIRST_STEP_POSITION + index,
                    "token": token,
                    "value": 10.0 + f64::from(index),
                    "top1_margin": 1.0,
                    "certified": true,
                    "claimable": true,
                    "logits_sha256": format!("position-{index}"),
                })
            })
            .collect();
        json!({
            "schema": DUMP_SCHEMA,
            "decode_positions": DECODE_POSITIONS,
            "positions": positions,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Synthetic reader tests only: UNMEASURED, no device record is read.

    use super::fixtures::*;
    use super::*;

    fn claimed_envelope() -> (tempfile::TempDir, DeviceChainEnvelope) {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), true);
        let envelope =
            DeviceChainEnvelope::load(dir.path()).expect("synthetic claimed envelope loads");
        (dir, envelope)
    }

    #[test]
    fn the_refused_window_loads_and_refuses_the_claim() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), false);
        let envelope = DeviceChainEnvelope::load(dir.path()).expect("refused envelope loads");
        assert!(!envelope.metadata().token_claimed);
        assert_eq!(envelope.metadata().decode_positions, DECODE_POSITIONS);
        assert_eq!(envelope.metadata().reference_ids, DECODE_IDS);
        assert_eq!(envelope.metadata().prompt_ids, prompt_ids());
        assert_eq!(envelope.world(), FOUR_RANK_WORLD);
        let refusal = envelope.claim().expect_err("a refused window cannot claim");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("token_claimed")),
            "the refusal must name the missing claim assertion: {}",
            refusal
        );
    }

    #[test]
    fn a_complete_claimed_envelope_claims_nine_ordered_tokens() {
        let (_dir, envelope) = claimed_envelope();
        let claimed = envelope
            .claim()
            .expect("the synthetic claimed envelope claims");
        assert_eq!(claimed.tokens.len(), DECODE_POSITIONS as usize);
        assert_eq!(
            claimed
                .tokens
                .iter()
                .map(|token| token.token_id)
                .collect::<Vec<_>>(),
            DECODE_IDS.to_vec()
        );
        assert_eq!(
            claimed
                .tokens
                .iter()
                .map(|token| token.position)
                .collect::<Vec<_>>(),
            (FIRST_STEP_POSITION..FIRST_STEP_POSITION + DECODE_POSITIONS).collect::<Vec<_>>()
        );
        assert_eq!(claimed.metadata.prompt_len, PROMPT_LEN);
        assert_eq!(
            claimed.tokens[2].logits_sha256.as_deref(),
            Some("digest-33")
        );
        assert_eq!(claimed.claim_mode, ClaimMode::Asserted);
        let dump = claimed
            .logits_dump
            .as_ref()
            .expect("the claimed fixture carries a dump");
        assert_eq!(dump.positions(), Some(DECODE_POSITIONS as usize));
    }

    #[test]
    fn a_missing_rank_pair_is_a_typed_error() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), true);
        std::fs::remove_file(dir.path().join("tp4-token-r2.json"))
            .expect("remove rank 2 token record");
        let error = DeviceChainEnvelope::load(dir.path()).expect_err("a missing rank pair fails");
        assert!(
            matches!(error, DeviceChainError::MissingRank { rank: 2, .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_schema_mismatch_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), true);
        rewrite_value(&dir.path().join("tp4-forward-embed-r0.json"), |record| {
            record["schema"] = serde_json::json!("not-the-schema");
        });
        let error = DeviceChainEnvelope::load(dir.path()).expect_err("a schema mismatch fails");
        assert!(
            matches!(error, DeviceChainError::Schema { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_world_mismatch_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), true);
        rewrite_value(&dir.path().join("tp4-forward-embed-r1.json"), |record| {
            record["world"] = serde_json::json!(2);
        });
        let error = DeviceChainEnvelope::load(dir.path()).expect_err("a world mismatch fails");
        assert!(
            matches!(error, DeviceChainError::WorldMismatch { rank: 1, .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_metadata_disagreement_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), true);
        rewrite_value(&dir.path().join("tp4-forward-embed-r3.json"), |record| {
            record["prompt_len"] = serde_json::json!(131);
        });
        let error =
            DeviceChainEnvelope::load(dir.path()).expect_err("a metadata disagreement fails");
        assert!(
            matches!(
                error,
                DeviceChainError::MetadataMismatch {
                    field: "prompt_len",
                    rank: 3,
                    ..
                }
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_rank_without_the_claim_assertion_is_refused() {
        let (dir, envelope) = claimed_envelope();
        rewrite_value(&dir.path().join("tp4-forward-embed-r1.json"), |record| {
            record["token_claimed"] = serde_json::json!(false);
        });
        let refreshed = DeviceChainEnvelope::load(dir.path()).expect("the envelope still loads");
        assert!(envelope.metadata().token_claimed);
        let refusal = refreshed
            .claim()
            .expect_err("one unclaimed rank refuses the claim");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("rank 1") && reason.contains("token_claimed")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn a_head_argmax_disagreement_is_refused() {
        let (dir, envelope) = claimed_envelope();
        rewrite_value(&dir.path().join("tp4-forward-embed-r2.json"), |record| {
            record["head"]["argmax"]["token"] = serde_json::json!(777);
        });
        let refreshed = DeviceChainEnvelope::load(dir.path()).expect("the envelope still loads");
        assert!(envelope.claim().is_ok());
        let refusal = refreshed
            .claim()
            .expect_err("a head disagreement refuses the claim");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("head argmax token 777")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn an_inconsistent_logits_dump_is_refused() {
        let (dir, _envelope) = claimed_envelope();
        write_rank0_dump(dir.path(), dump_value(false));
        let refreshed = DeviceChainEnvelope::load(dir.path()).expect("the envelope still loads");
        let refusal = refreshed
            .claim()
            .expect_err("an inconsistent dump refuses the claim");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("logits dump position 0 token")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn a_digest_logits_dump_is_claimed_and_exposed() {
        let (dir, _envelope) = claimed_envelope();
        write_rank0_dump(dir.path(), dump_value(true));
        let refreshed = DeviceChainEnvelope::load(dir.path()).expect("the envelope still loads");
        let claimed = refreshed
            .claim()
            .expect("a consistent dump does not block the claim");
        let dump = claimed.logits_dump.as_ref().expect("the dump is exposed");
        let LogitsDumpEvidence::Inline(inline) = dump else {
            panic!("the inline dump fixture must expose an inline dump");
        };
        assert_eq!(inline.positions.len(), DECODE_POSITIONS as usize);
        assert_eq!(claimed.tokens[0].certified, Some(true));
        assert_eq!(claimed.tokens[0].claimable, Some(true));
        // The batch's own per-position digest wins over the dump digest: the dump is evidence
        // about the same position, not a replacement for the token record's hash.
        assert_eq!(
            claimed.tokens[3].logits_sha256.as_deref(),
            Some("digest-44")
        );
        assert_eq!(
            inline.positions[3].logits_sha256.as_deref(),
            Some("position-3")
        );
    }

    #[test]
    fn a_path_shaped_logits_dump_is_loaded() {
        let (dir, _envelope) = claimed_envelope();
        write_value(&dir.path().join("logits.json"), &dump_value(true));
        write_rank0_dump(dir.path(), serde_json::json!("logits.json"));
        let refreshed = DeviceChainEnvelope::load(dir.path()).expect("the envelope still loads");
        let claimed = refreshed
            .claim()
            .expect("a readable path dump does not block the claim");
        assert!(claimed.logits_dump.is_some());
    }

    #[test]
    fn an_unreadable_logits_dump_refuses_the_claim() {
        let (dir, _envelope) = claimed_envelope();
        write_rank0_dump(dir.path(), serde_json::json!("missing-dump.json"));
        let refreshed = DeviceChainEnvelope::load(dir.path()).expect("the envelope still loads");
        let refusal = refreshed
            .claim()
            .expect_err("a missing dump refuses the claim");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("logits dump unavailable")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn payload_paths_must_stay_inside_the_envelope_directory() {
        assert!(check_relative_payload("rows/000.bin").is_ok());
        assert!(check_relative_payload("../outside.bin").is_err());
        assert!(check_relative_payload("/tmp/outside.bin").is_err());
    }

    #[test]
    fn the_device_rank_directory_layout_loads_and_claims() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), true);
        relocate_to_rank_directories(dir.path());
        let envelope =
            DeviceChainEnvelope::load(dir.path()).expect("the rank-directory layout loads");
        let claimed = envelope.claim().expect("the rank-directory layout claims");
        assert_eq!(claimed.tokens.len(), DECODE_POSITIONS as usize);
        assert_eq!(claimed.tokens[0].token_id, DECODE_IDS[0]);
    }

    #[test]
    fn a_derived_claim_requires_the_loop_evidence_and_is_not_asserted() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_derived_envelope(dir.path());
        let envelope = DeviceChainEnvelope::load(dir.path()).expect("derived fixture loads");
        assert!(!envelope.metadata().token_claimed);
        let asserted = envelope
            .claim()
            .expect_err("without the writer assertion the strict claim refuses");
        assert!(
            asserted
                .reasons
                .iter()
                .any(|reason| reason.contains("token_claimed")),
            "unexpected refusal: {asserted}"
        );
        let claimed = envelope
            .claim_with(ClaimMode::Derived)
            .expect("the complete loop evidence derives the claim");
        assert_eq!(claimed.claim_mode, ClaimMode::Derived);
        assert_eq!(
            claimed
                .tokens
                .iter()
                .map(|token| token.token_id)
                .collect::<Vec<_>>(),
            DECODE_IDS.to_vec()
        );
        assert!(claimed.logits_dump.is_some());
    }

    #[test]
    fn a_derived_claim_refuses_when_the_first_fed_row_is_not_the_prompt_token() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_derived_envelope(dir.path());
        for rank in 0..FOUR_RANK_WORLD {
            rewrite_value(
                &dir.path().join(format!("tp4-forward-embed-r{rank}.json")),
                |record| {
                    let ids = record["forward_ids"].as_array_mut().expect("forward_ids");
                    let first_dumped_input = ids
                        .get_mut(PROMPT_LEN as usize - 1)
                        .expect("the first dumped row's input");
                    *first_dumped_input = serde_json::json!(7);
                },
            );
        }
        let envelope = DeviceChainEnvelope::load(dir.path()).expect("fixture still loads");
        let refusal = envelope
            .claim_with(ClaimMode::Derived)
            .expect_err("a forward_ids mismatch neither asserts nor derives a claim");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("forward_ids[prompt_len - 1]")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn a_derived_claim_refuses_when_a_reference_match_is_missing() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_derived_envelope(dir.path());
        rewrite_value(&dir.path().join("tp4-forward-embed-r0.json"), |record| {
            record["tokens"]["matches_reference"][0] = Value::Null;
        });
        let envelope = DeviceChainEnvelope::load(dir.path()).expect("fixture still loads");
        let refusal = envelope
            .claim_with(ClaimMode::Derived)
            .expect_err("an unmatched reference id refuses the derived claim");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("do not all match the reference")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn a_device_dump_with_the_wrong_declared_row_count_is_refused() {
        let (dir, _envelope) = claimed_envelope();
        write_rank0_dump(dir.path(), {
            let mut dump = device_dump_value();
            dump["rows"] = serde_json::json!(8);
            dump
        });
        let envelope = DeviceChainEnvelope::load(dir.path()).expect("fixture still loads");
        let refusal = envelope
            .claim()
            .expect_err("the dump row count must match the batch");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("declares 8 rows")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn a_device_dump_with_a_bad_digest_is_refused() {
        let (dir, _envelope) = claimed_envelope();
        write_rank0_dump(dir.path(), {
            let mut dump = device_dump_value();
            dump["bin_sha256"] = serde_json::json!("not-a-digest");
            dump
        });
        let envelope = DeviceChainEnvelope::load(dir.path()).expect("fixture still loads");
        let refusal = envelope
            .claim()
            .expect_err("the dump digest must be a sha256");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("bin_sha256") && reason.contains("64 hex")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    #[ignore = "reads the token window named by SPARKFLASH_TOKEN_ENVELOPE_DIR; opt in with --ignored"]
    fn a_real_token_window_loads_and_reports_its_claim_state() {
        // The S7/S8 probe entry point: load one recorded four-rank window and print the claim
        // decision so a script can capture the evidence. It never asserts a claim, so it is safe
        // on a refused or partial window; it fails only when the records do not load.
        let Ok(dir) = std::env::var("SPARKFLASH_TOKEN_ENVELOPE_DIR") else {
            eprintln!("skip: set SPARKFLASH_TOKEN_ENVELOPE_DIR to a four-rank token window");
            return;
        };
        let envelope = DeviceChainEnvelope::load(&dir).expect("the token window loads");
        println!(
            "window: world={} prompt_len={} first_step_position={} decode_positions={} reference_ids={:?} prompt_ids={} token_claimed={} prefill_limit={:?} evidence_written={:?}",
            envelope.world(),
            envelope.metadata().prompt_len,
            envelope.metadata().first_step_position,
            envelope.metadata().decode_positions,
            envelope.metadata().reference_ids,
            envelope.metadata().prompt_ids.len(),
            envelope.metadata().token_claimed,
            envelope.metadata().prefill_limit,
            envelope.metadata().evidence_written,
        );
        println!(
            "logits_dump_present={}",
            envelope.logits_dump().expect("dump read").is_some()
        );
        match envelope.claim() {
            Ok(claimed) => println!(
                "claim: explicit ok, {} tokens, positions {:?}, ids {:?}",
                claimed.tokens.len(),
                claimed
                    .tokens
                    .iter()
                    .map(|token| token.position)
                    .collect::<Vec<_>>(),
                claimed
                    .tokens
                    .iter()
                    .map(|token| token.token_id)
                    .collect::<Vec<_>>(),
            ),
            Err(refusal) => println!("claim: explicit refused: {refusal}"),
        }
        match envelope.claim_with(ClaimMode::Derived) {
            Ok(claimed) => println!(
                "claim: derived ok, {} tokens, ids {:?}",
                claimed.tokens.len(),
                claimed
                    .tokens
                    .iter()
                    .map(|token| token.token_id)
                    .collect::<Vec<_>>(),
            ),
            Err(refusal) => println!("claim: derived refused: {refusal}"),
        }
    }

    #[test]
    #[ignore = "reads the m1y7 evidence window on this host; opt in with --ignored"]
    fn the_real_m1y7_refused_window_loads_and_refuses_the_claim() {
        // Evidence check, not a CI test: the window is a refused device run, and the consumer must
        // read its real schema without error and refuse the claim for the recorded reasons.
        let dir = std::path::Path::new("/home/user/ev-session-20261020/window/m1y7");
        if !dir.is_dir() {
            eprintln!("skip: the m1y7 evidence window is not present on this host");
            return;
        }
        let envelope = DeviceChainEnvelope::load(dir).expect("the m1y7 envelope loads");
        assert_eq!(envelope.world(), FOUR_RANK_WORLD);
        assert!(!envelope.metadata().token_claimed);
        assert_eq!(envelope.metadata().prompt_len, PROMPT_LEN);
        assert_eq!(envelope.metadata().first_step_position, FIRST_STEP_POSITION);
        assert_eq!(envelope.metadata().decode_positions, DECODE_POSITIONS);
        assert_eq!(
            envelope.metadata().reference_ids,
            vec![491, 367, 201, 80, 3302, 201, 54, 201, 4341]
        );
        assert!(envelope.metadata().prompt_ids.is_empty());
        let refusal = envelope
            .claim()
            .expect_err("the m1y7 window claims no token");
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("token_claimed")),
            "unexpected refusal: {refusal}"
        );
        assert!(
            refusal
                .reasons
                .iter()
                .any(|reason| reason.contains("forward = \"refused\"")),
            "unexpected refusal: {refusal}"
        );
    }

    #[test]
    fn the_m1y34_forward_perf_progress_rows_are_not_part_of_the_envelope() {
        // The m1y34 rows add a forward_perf: {lean} field to every forward-progress entry. The
        // envelope is the pair of embed/token records; a progress row is neither read nor
        // validated, so the new evidence shape cannot change a claim. This pins the tolerance
        // against a window directory that carries the extra row (the identity and lean rows of
        // m1y34 both do).
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), true);
        for rank in 0..FOUR_RANK_WORLD as usize {
            let entries: Vec<Value> = (0..40)
                .map(|layer| {
                    serde_json::json!({
                        "layer": layer,
                        "forward_perf": {"lean": false},
                        "current_layer": layer,
                    })
                })
                .collect();
            let progress = serde_json::json!({
                "schema": "sparkflash-tp4-forward-progress/1",
                "rank": rank,
                "world": FOUR_RANK_WORLD,
                "entries": entries,
            });
            std::fs::write(
                dir.path()
                    .join(format!("tp4-forward-progress-r{rank}.json")),
                serde_json::to_vec(&progress).expect("progress JSON"),
            )
            .expect("progress row writes");
        }
        let envelope =
            DeviceChainEnvelope::load(dir.path()).expect("the m1y34 window shape still loads");
        let claimed = envelope
            .claim()
            .expect("forward-progress evidence does not affect the claim");
        assert_eq!(claimed.tokens.len(), DECODE_POSITIONS as usize);
        assert_eq!(claimed.tokens[0].token_id, DECODE_IDS[0]);
    }
}
