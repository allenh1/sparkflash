//! The four-rank token evidence: `sparkflash-tp4-token/1` and the full-vocabulary dump.
//!
//! This module owns the second half of the M1 token deliverable: the per-rank envelope qa-lead's
//! `scripts/accept_device_token.py` consumes, and the D40 logits dump it points at. The schema is
//! frozen field-for-field by qa-lead's message; [`TokenEnvelope::validate`] encodes their shape
//! rules so a malformed envelope fails on the node that writes it instead of at review time.
//!
//! The dump is written by the frozen D40 writer in [`crate::logits_dump`] - the same code path the
//! single-node chain uses - from the consensus logits the four ranks all-gather, so there is one
//! implementation of the artefact. Rank 0 writes it; ranks 1-3 record only duplicated consensus
//! values, because qa-lead's consumer compares the four rank envelopes for identical token ids and
//! reads one dump.
//!
//! What this module does NOT claim: a token. A `run` envelope exists only when the forward loop
//! executed and produced these values; the refusal constructor exists so the window's preflight
//! can report "not attempted, here is why" without fabricating ids.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::logits_dump;

/// The envelope schema this module writes.
pub const SCHEMA: &str = "sparkflash-tp4-token/1";
/// The stage every envelope names (the chain stage that owns the file).
pub const STAGE: &str = "tp4-chain";
/// The loop step every envelope names (the C4-token step of the campaign).
pub const LOOP_STEP: &str = "C4-token";
/// The released vocabulary the four ranks share.
pub const VOCAB: usize = 129_280;
/// The rank-sharded head width per rank (`VOCAB / 4`), read back for a sanity check.
pub const VOCAB_PER_RANK: usize = VOCAB / 4;
/// The four-rank arm's reduce count per decode step: two per layer over 40 layers.
pub const REDUCE_CALLS_PER_STEP: usize = 80;
/// The frozen stride of one position's logits block in the dump sidecar.
pub const DUMP_BLOCK_BYTES: usize = VOCAB * 4;

/// How the forward/token stage describes itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForwardState {
    /// The loop executed and produced the token ids in this envelope.
    Run,
    /// The stage was requested but a precondition failed; `error` names it.
    Refused,
    /// The stage was not requested at all (an envelope is not written in this state by the chain,
    /// but the enum keeps the accepted wire vocabulary explicit).
    NotRun,
}

impl ForwardState {
    /// The wire spelling qa-lead's consumer accepts.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Refused => "refused",
            Self::NotRun => "not-run",
        }
    }
}

/// The position-level token evidence: one entry per position the loop ran.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tokens {
    /// The consensus token ids, in position order.
    pub ids: Vec<u32>,
    /// The position each id was produced at.
    pub positions: Vec<usize>,
    /// The first position of this run, or `null` when nothing ran.
    pub start_position: Option<usize>,
    /// The prompt length this run used, or `null` when nothing ran.
    pub prompt_len: Option<usize>,
    /// The winning pre-softmax logit of each position's full-vocabulary vector.
    pub argmax_logits: Vec<f64>,
    /// The top-1 minus top-2 logit gap of each position, as a confidence witness.
    pub top1_margins: Vec<f64>,
    /// The SHA-256 of the full-vocabulary vector(s) as f32-le, or `null` when no dump was written.
    pub logits_sha256: Option<String>,
    /// The teacher-forced list the run was given: the whole `--token-ids` argument.
    ///
    /// Recorded separately from [`Self::ids`] (the run's own answers) because D55's fixture key is
    /// built from these: a device arm that flips a token answered the same question and must stay
    /// comparable with the reference, which is the whole point of teacher forcing. The whole list,
    /// not just its tail, because the prompt is part of the question - two prompts of the same
    /// length with the same forced tail are different questions. Empty on a run that predates the
    /// field, and the gate then falls back to `ids` (the D54 key).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forced_ids: Vec<u32>,
}

impl Tokens {
    /// Builds the empty token block a refusal reports.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Whether the block carries at least one position.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Appends one position's consensus values.
    ///
    /// # Errors
    /// A logits vector whose length is not [`VOCAB`], or a non-increasing position.
    pub fn push(&mut self, position: usize, logits: &[f32], token: u32) -> Result<(), String> {
        if logits.len() != VOCAB {
            return Err(format!(
                "token evidence: position {position} has {} logits, expected {VOCAB}",
                logits.len()
            ));
        }
        if let Some(last) = self.positions.last()
            && position <= *last
        {
            return Err(format!(
                "token evidence: position {position} does not follow {last}"
            ));
        }
        let (argmax, second) = top_two(logits);
        self.ids.push(token);
        self.positions.push(position);
        self.argmax_logits.push(f64::from(logits[argmax]));
        self.top1_margins.push(f64::from(logits[argmax] - second));
        Ok(())
    }
}

fn top_two(logits: &[f32]) -> (usize, f32) {
    let mut first = 0usize;
    let mut second = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if value.total_cmp(&logits[first]).is_gt() {
            second = first;
            first = index;
        } else if index != first && value.total_cmp(&logits[second]).is_gt() {
            second = index;
        }
    }
    (first, logits[second])
}

/// Per-step wall times for the token run: every measured sample, its mean and its worst case.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepMs {
    /// One sample per executed step, in order.
    pub samples: Vec<f64>,
    /// The arithmetic mean of [`Self::samples`] (0 when empty).
    pub mean_ms: f64,
    /// The worst sample (0 when empty).
    pub worst_ms: f64,
}

impl StepMs {
    /// Builds the block from measured samples.
    #[must_use]
    pub fn from_samples(samples: Vec<f64>) -> Self {
        let mean_ms = if samples.is_empty() {
            0.0
        } else {
            samples.iter().sum::<f64>() / samples.len() as f64
        };
        let worst_ms = samples.iter().copied().fold(0.0_f64, f64::max);
        Self {
            samples,
            mean_ms,
            worst_ms,
        }
    }
}

/// The head description: the rule, the vocabulary and the tie-break every rank used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Head {
    /// The consensus rule, frozen to the arm's real implementation.
    pub rule: String,
    /// The vocabulary the argmax ran over.
    pub vocab: usize,
    /// The tie-break rule.
    pub tie_break: String,
}

impl Default for Head {
    fn default() -> Self {
        Self {
            rule: "rank-sharded head rows all-gathered over tcp; every rank argmaxes the full vector; rank0 dumps full logits".to_string(),
            vocab: VOCAB,
            tie_break: "lowest token id".to_string(),
        }
    }
}

/// The dump pointer: the JSONL, its f32-le sidecar and both hashes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogitsDump {
    /// The JSONL dump path.
    pub path: String,
    /// The sidecar path (frozen: dump path + `.bin`).
    pub bin: String,
    /// SHA-256 of the JSONL.
    pub sha256: String,
    /// SHA-256 of the sidecar.
    pub bin_sha256: String,
    /// The sidecar encoding; frozen to f32 little-endian.
    pub dtype: String,
    /// The vocabulary of every dumped block.
    pub vocab: usize,
    /// The positions present, in dump order.
    pub positions: Vec<usize>,
    /// The number of JSONL rows.
    pub rows: usize,
}

/// One rank's token evidence envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEnvelope {
    /// [`SCHEMA`].
    pub schema: String,
    /// [`STAGE`].
    pub stage: String,
    /// [`LOOP_STEP`].
    pub loop_step: String,
    /// This rank.
    pub rank: usize,
    /// The world size (4).
    pub world: usize,
    /// Whether the loop ran, was refused, or was not attempted.
    pub forward: ForwardState,
    /// The position-level consensus evidence.
    pub tokens: Tokens,
    /// How many positions the loop advanced.
    pub positions_run: usize,
    /// The measured step times.
    pub step_ms: StepMs,
    /// Per-step wall times in milliseconds, in order.
    pub wall_ms: Vec<f64>,
    /// The reduce calls one decode step performed (80 on this arm).
    pub reduce_calls_per_step: usize,
    /// The head description.
    pub head: Head,
    /// The dump pointer, or `null` when nothing ran.
    pub logits_dump: Option<LogitsDump>,
    /// The refusal reason (or a post-run warning), or `null` on a clean run.
    pub error: Option<String>,
    /// The launch census for the combine path (QA reads `routed_experts_down`).
    pub launch_census: serde_json::Value,
    /// Whether the hybrid (non-TP) sequence ran inside this stage; always `false` on this arm.
    pub hybrid_sequence_executed: bool,
    /// The run header ([`crate::run_header`]): the arm, the collective path, the switches and the
    /// NCCL transport environment the run started from, plus the NCCL build once a communicator
    /// opened. Absent only in envelopes written before S2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_header: Option<serde_json::Value>,
}

impl TokenEnvelope {
    /// The empty launch census a refusal reports: nothing was launched, so nothing is claimed.
    #[must_use]
    pub fn refusal_census(note: &str) -> serde_json::Value {
        serde_json::json!({
            "routed_experts_down": false,
            "note": note,
        })
    }

    /// Builds the envelope for a stage that was requested but could not run.
    #[must_use]
    pub fn refused(rank: usize, world: usize, error: String) -> Self {
        Self {
            schema: SCHEMA.to_string(),
            stage: STAGE.to_string(),
            loop_step: LOOP_STEP.to_string(),
            rank,
            world,
            forward: ForwardState::Refused,
            tokens: Tokens::empty(),
            positions_run: 0,
            step_ms: StepMs::default(),
            wall_ms: Vec::new(),
            reduce_calls_per_step: 0,
            head: Head::default(),
            logits_dump: None,
            error: Some(error),
            launch_census: Self::refusal_census(
                "forward refused before any routed expert launched",
            ),
            hybrid_sequence_executed: false,
            run_header: None,
        }
    }

    /// Builds the envelope for a completed run.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn ran(
        rank: usize,
        world: usize,
        tokens: Tokens,
        step_ms: StepMs,
        wall_ms: Vec<f64>,
        reduce_calls_per_step: usize,
        dump: LogitsDump,
        launch_census: serde_json::Value,
    ) -> Self {
        let positions_run = tokens.ids.len();
        Self {
            schema: SCHEMA.to_string(),
            stage: STAGE.to_string(),
            loop_step: LOOP_STEP.to_string(),
            rank,
            world,
            forward: ForwardState::Run,
            tokens,
            positions_run,
            step_ms,
            wall_ms,
            reduce_calls_per_step,
            head: Head::default(),
            logits_dump: Some(dump),
            error: None,
            launch_census,
            hybrid_sequence_executed: false,
            run_header: None,
        }
    }

    /// The file name this rank's envelope is written under.
    #[must_use]
    pub fn file_name(rank: usize) -> String {
        format!("tp4-token-r{rank}.json")
    }

    /// Writes the envelope into `dir`, returning the path.
    ///
    /// # Errors
    /// A directory that cannot be written or a serialisation failure.
    pub fn write(&self, dir: &Path) -> Result<PathBuf, String> {
        let path = dir.join(Self::file_name(self.rank));
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| format!("token envelope serialise: {error}"))?;
        std::fs::write(&path, bytes)
            .map_err(|error| format!("write {}: {error}", path.display()))?;
        Ok(path)
    }

    /// Checks the envelope against qa-lead's frozen shape rules.
    ///
    /// This is the writer-side half of the consumer's validation: a run must carry ids, positions,
    /// a dump and no error; a refusal must carry none of those ids and must name its reason; the
    /// dump must tile the sidecar the frozen contract expects.
    ///
    /// # Errors
    /// The first violated rule, with the envelope's own values in the message.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!("schema {} is not {SCHEMA}", self.schema));
        }
        if self.stage != STAGE || self.loop_step != LOOP_STEP {
            return Err(format!(
                "stage/loop_step {}/{} are not {STAGE}/{LOOP_STEP}",
                self.stage, self.loop_step
            ));
        }
        if self.world != 4 {
            return Err(format!("world {} is not 4", self.world));
        }
        if self.rank >= self.world {
            return Err(format!(
                "rank {} is outside world {}",
                self.rank, self.world
            ));
        }
        if self.head.vocab != VOCAB {
            return Err(format!("head vocab {} is not {VOCAB}", self.head.vocab));
        }
        match self.forward {
            ForwardState::Run => self.validate_run(),
            ForwardState::Refused => self.validate_refusal(),
            ForwardState::NotRun => Err(
                "a not-run envelope is never written: the chain omits the file instead".to_string(),
            ),
        }
    }

    fn validate_run(&self) -> Result<(), String> {
        if self.tokens.is_empty() {
            return Err("forward=run carries no token ids".to_string());
        }
        if self.positions_run != self.tokens.ids.len() {
            return Err(format!(
                "positions_run {} != {} token ids",
                self.positions_run,
                self.tokens.ids.len()
            ));
        }
        if self.tokens.positions.len() != self.tokens.ids.len() {
            return Err(format!(
                "{} positions for {} token ids",
                self.tokens.positions.len(),
                self.tokens.ids.len()
            ));
        }
        if self.tokens.argmax_logits.len() != self.tokens.ids.len()
            || self.tokens.top1_margins.len() != self.tokens.ids.len()
        {
            return Err("argmax_logits/top1_margins do not cover every token id".to_string());
        }
        let Some(first) = self.tokens.start_position else {
            return Err("forward=run carries no start_position".to_string());
        };
        if self.tokens.positions[0] != first {
            return Err(format!(
                "start_position {first} != first dumped position {}",
                self.tokens.positions[0]
            ));
        }
        for pair in self.tokens.positions.windows(2) {
            if pair[1] <= pair[0] {
                return Err(format!(
                    "positions are not strictly increasing: {} then {}",
                    pair[0], pair[1]
                ));
            }
        }
        if let Some(prompt_len) = self.tokens.prompt_len
            && first >= prompt_len
        {
            return Err(format!(
                "start_position {first} is not before prompt_len {prompt_len}"
            ));
        }
        let Some(dump) = &self.logits_dump else {
            if self.rank == 0 {
                return Err(
                    "forward=run carries no logits_dump pointer on rank 0, the primary dump owner"
                        .to_string(),
                );
            }
            // Ranks 1-3 may publish the same consensus ids without a dump: qa-lead's consumer
            // allows replica dumps to be omitted (their sidecars, when present, must be identical).
            if self.error.is_some()
                || self.reduce_calls_per_step != REDUCE_CALLS_PER_STEP
                || self.launch_census["routed_experts_down"] != serde_json::Value::Bool(true)
            {
                return Err("forward=run replica without a dump still needs reduce_calls_per_step=80, routed_experts_down=true and no error".to_string());
            }
            return Ok(());
        };
        if dump.dtype != "f32-le" {
            return Err(format!("dump dtype {} is not f32-le", dump.dtype));
        }
        if dump.vocab != VOCAB {
            return Err(format!("dump vocab {} is not {VOCAB}", dump.vocab));
        }
        // A lean-head run (SPARKFLASH_LEAN_HEAD=1) dumps only the heads that read their logits
        // back: a subset of the token block's positions, never a position outside it.
        let lean = std::env::var("SPARKFLASH_LEAN_HEAD").as_deref() == Ok("1");
        let paired = if lean {
            dump.rows == dump.positions.len()
                && dump
                    .positions
                    .iter()
                    .all(|position| self.tokens.positions.contains(position))
        } else {
            dump.rows == self.tokens.ids.len() && dump.positions == self.tokens.positions
        };
        if !paired {
            return Err(format!(
                "dump ({} rows, {:?} positions) does not match the token block ({} ids)",
                dump.rows,
                dump.positions,
                self.tokens.ids.len()
            ));
        }
        let expected_bin = format!("{}.bin", dump.path);
        if dump.bin != expected_bin {
            return Err(format!("dump sidecar {} is not {}", dump.bin, expected_bin));
        }
        if dump.sha256.len() != 64 || dump.bin_sha256.len() != 64 {
            return Err("dump hashes are not 64-hex strings".to_string());
        }
        if self.reduce_calls_per_step != REDUCE_CALLS_PER_STEP {
            return Err(format!(
                "reduce_calls_per_step {} is not {REDUCE_CALLS_PER_STEP}",
                self.reduce_calls_per_step
            ));
        }
        if self.launch_census["routed_experts_down"] != serde_json::Value::Bool(true) {
            return Err("forward=run must report routed_experts_down=true".to_string());
        }
        if self.error.is_some() {
            return Err(format!(
                "forward=run carries an error: {}",
                self.error.as_deref().unwrap_or_default()
            ));
        }
        if self.wall_ms.len() != self.step_ms.samples.len() {
            return Err(format!(
                "wall_ms {} != step_ms samples {}",
                self.wall_ms.len(),
                self.step_ms.samples.len()
            ));
        }
        Ok(())
    }

    fn validate_refusal(&self) -> Result<(), String> {
        if !self.tokens.is_empty() || !self.tokens.positions.is_empty() || self.positions_run != 0 {
            return Err("forward=refused must not carry token ids or positions".to_string());
        }
        if self.tokens.start_position.is_some() || self.tokens.prompt_len.is_some() {
            return Err(
                "forward=refused must leave start_position and prompt_len null".to_string(),
            );
        }
        if self.logits_dump.is_some() {
            return Err("forward=refused must not point at a logits dump".to_string());
        }
        match &self.error {
            None => Err("forward=refused carries no error".to_string()),
            Some(error) if error.trim().is_empty() => {
                Err("forward=refused carries an empty error".to_string())
            }
            Some(_) => {
                if self.launch_census["routed_experts_down"] != serde_json::Value::Bool(false) {
                    return Err("forward=refused must report routed_experts_down=false".to_string());
                }
                Ok(())
            }
        }
    }
}

/// Writes the D40 dump for one rank from the consensus full-vocabulary logits.
///
/// Reuses crate::logits_dump::write_logits_dump - the frozen writer - so the artefact has one
/// implementation. The rows carry the consensus token as both device_argmax and hybrid_id, the
/// teacher-forced reference token as reference_id when the caller supplies the vector (QA's
/// like-for-like rule: a null reference_id is counted as a prefix mismatch), and
/// exact_token_equal = whether the reference id and the hybrid id are equal - the field's schema
/// meaning; the sidecar is the full f32-le vocabulary per position.
///
/// # Errors
/// Empty input, a logits vector whose length is not [`VOCAB`], or a write failure.
pub fn write_dump(
    dir: &Path,
    rank: usize,
    positions: &[usize],
    tokens: &[u32],
    logits: &[Vec<f32>],
    references: Option<&[u32]>,
) -> Result<LogitsDump, String> {
    if positions.is_empty() {
        return Err("token dump: nothing to dump".to_string());
    }
    if positions.len() != tokens.len() || tokens.len() != logits.len() {
        return Err(format!(
            "token dump: {} positions, {} tokens, {} logits do not pair",
            positions.len(),
            tokens.len(),
            logits.len()
        ));
    }
    if let Some(references) = references
        && references.len() != positions.len()
    {
        return Err(format!(
            "token dump: {} reference ids for {} positions: a teacher-forced run needs one reference token per dumped row",
            references.len(),
            positions.len()
        ));
    }
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("token dump: create {}: {error}", dir.display()))?;
    let path = dir.join(format!("tp4-logits-r{rank}.jsonl"));
    let mut rows: Vec<logits_dump::ComparedRow> = Vec::with_capacity(positions.len());
    let mut outputs = Vec::with_capacity(positions.len());
    for (index, logits) in logits.iter().enumerate() {
        if logits.len() != VOCAB {
            return Err(format!(
                "token dump: position {} has {} logits, expected {VOCAB}",
                positions[index],
                logits.len()
            ));
        }
        let token = tokens[index];
        let reference = references.map(|references| references[index]);
        // exact_token_equal is the schema's row verdict: reference id == hybrid id. The old
        // expression compared the consensus token with its own logits argmax, which is true by
        // construction, so the field could never report a mismatch (m1y14 read 9/9 rows true).
        rows.push((
            positions[index],
            reference,
            Some(token),
            reference == Some(token),
        ));
        outputs.push(sparkflash_engine::forward::HybridTokenOutput {
            layers: Vec::new(),
            tail_stages: Vec::new(),
            head_input: Vec::new(),
            logits: logits.clone(),
            token: Some(token),
            certified_mask: None,
        });
    }
    let written = logits_dump::write_logits_dump(&path, &rows, &outputs, 0, false, &[], 0)?;
    if written != positions.len() {
        return Err(format!(
            "token dump: wrote {written} rows for {} positions",
            positions.len()
        ));
    }
    let bin = logits_dump::sidecar_path(&path);
    let sha256 = sparkflash_engram::packer::sha256_file(&path)
        .map_err(|error| format!("token dump {}: {error}", path.display()))?;
    let bin_sha256 = sparkflash_engram::packer::sha256_file(&bin)
        .map_err(|error| format!("token dump {}: {error}", bin.display()))?;
    Ok(LogitsDump {
        path: path.display().to_string(),
        bin: bin.display().to_string(),
        sha256,
        bin_sha256,
        dtype: "f32-le".to_string(),
        vocab: VOCAB,
        positions: positions.to_vec(),
        rows: written,
    })
}

/// The first argmax under the frozen tie-break (lowest token id).
#[must_use]
pub fn argmax_of(logits: &[f32]) -> usize {
    let mut argmax = 0usize;
    for (index, value) in logits.iter().enumerate() {
        if value.total_cmp(&logits[argmax]).is_gt() {
            argmax = index;
        }
    }
    argmax
}

/// The SHA-256 of a full-vocabulary f32-le vector, matching the dump sidecar's encoding.
///
/// # Errors
/// A vector whose length is not [`VOCAB`].
pub fn logits_sha256(logits: &[f32]) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    if logits.len() != VOCAB {
        return Err(format!(
            "logits sha256: {} values, expected {VOCAB}",
            logits.len()
        ));
    }
    let mut hasher = Sha256::new();
    let mut block = Vec::with_capacity(DUMP_BLOCK_BYTES);
    for value in logits {
        block.extend_from_slice(&value.to_le_bytes());
    }
    hasher.update(&block);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_logits(winner: usize, second: usize) -> Vec<f32> {
        let mut logits = vec![-1.0f32; VOCAB];
        logits[winner] = 7.5;
        logits[second] = 6.0;
        logits
    }

    fn ran_envelope(dir: &Path) -> TokenEnvelope {
        let positions = [129usize, 130];
        let tokens = [5u32, 1];
        let logits = vec![synthetic_logits(5, 3), synthetic_logits(1, 9)];
        let mut tokens_block = Tokens::empty();
        tokens_block
            .push(positions[0], &logits[0], tokens[0])
            .expect("push");
        tokens_block
            .push(positions[1], &logits[1], tokens[1])
            .expect("push");
        tokens_block.start_position = Some(positions[0]);
        tokens_block.prompt_len = Some(130);
        tokens_block.logits_sha256 = Some(logits_sha256(&logits[1]).expect("sha"));
        let dump = write_dump(dir, 0, &positions, &tokens, &logits, None).expect("dump");
        let step_ms = StepMs::from_samples(vec![31.5, 18.25]);
        TokenEnvelope::ran(
            0,
            4,
            tokens_block,
            step_ms,
            vec![41.0, 29.0],
            REDUCE_CALLS_PER_STEP,
            dump,
            serde_json::json!({
                "routed_experts_down": true,
                "layout": "stored_576k",
                "kernel": "b12x_mxfp4_moe_stored_sm121",
                "note": "stored-layout combine launched on arena rows",
            }),
        )
    }

    #[test]
    fn a_refusal_is_a_valid_envelope_with_no_token_claim() {
        let envelope = TokenEnvelope::refused(
            3,
            4,
            "stored-layout combine ABI not wired: b12x_mxfp4_moe_stored_sm121 has no loader"
                .to_string(),
        );
        envelope.validate().expect("refusal validates");
        assert_eq!(envelope.forward.as_str(), "refused");
        assert_eq!(envelope.tokens.ids, Vec::<u32>::new());
        assert_eq!(envelope.positions_run, 0);
        assert!(envelope.logits_dump.is_none());
        assert!(
            envelope
                .error
                .as_deref()
                .expect("error")
                .contains("stored-layout")
        );
        assert_eq!(envelope.launch_census["routed_experts_down"], false);
        let value = serde_json::to_value(&envelope).expect("serde");
        for key in [
            "schema",
            "rank",
            "world",
            "forward",
            "tokens",
            "positions_run",
            "error",
            "logits_dump",
            "head",
            "launch_census",
        ] {
            assert!(value.get(key).is_some(), "envelope lacks {key}");
        }
        assert!(value["tokens"].get("ids").is_some());
        assert!(value["tokens"]["start_position"].is_null());
    }

    #[test]
    fn a_refusal_with_no_reason_is_rejected() {
        let mut envelope = TokenEnvelope::refused(0, 4, "x".to_string());
        envelope.error = None;
        assert!(envelope.validate().unwrap_err().contains("no error"));
    }

    #[test]
    fn a_run_envelope_validates_and_carries_one_dump_block_per_position() {
        let dir = tempfile::tempdir().expect("tempdir");
        let envelope = ran_envelope(dir.path());
        envelope.validate().expect("run validates");
        assert_eq!(envelope.positions_run, 2);
        assert_eq!(envelope.tokens.positions, vec![129, 130]);
        assert_eq!(envelope.tokens.ids, vec![5, 1]);
        assert_eq!(envelope.tokens.argmax_logits, vec![7.5, 7.5]);
        assert!((envelope.tokens.top1_margins[0] - 1.5).abs() < 1e-6);
        let dump = envelope.logits_dump.as_ref().expect("dump");
        assert_eq!(dump.dtype, "f32-le");
        assert_eq!(dump.vocab, VOCAB);
        assert_eq!(dump.rows, 2);
        assert_eq!(dump.positions, vec![129, 130]);
        assert!(dump.bin.ends_with(".jsonl.bin"));
        assert_eq!(
            std::fs::read(&dump.bin).expect("bin").len(),
            2 * DUMP_BLOCK_BYTES
        );
        // The frozen consumer's checks, run here so the writer-side validation cannot drift.
        let text = std::fs::read_to_string(&dump.path).expect("jsonl");
        let rows: Vec<serde_json::Value> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("row"))
            .collect();
        assert_eq!(rows.len(), 2);
        let mut offset = 0usize;
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row["vocab"], VOCAB);
            assert_eq!(row["logits_offset"], offset);
            assert_eq!(row["position"], serde_json::json!(129 + index));
            assert_eq!(
                Path::new(row["logits_bin"].as_str().expect("bin string"))
                    .file_name()
                    .expect("basename"),
                Path::new(&dump.bin).file_name().expect("basename")
            );
            offset += DUMP_BLOCK_BYTES;
        }
    }

    #[test]
    fn a_run_without_a_dump_or_with_a_short_dump_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut envelope = ran_envelope(dir.path());
        envelope.logits_dump = None;
        assert!(envelope.validate().unwrap_err().contains("logits_dump"));

        let mut envelope = ran_envelope(dir.path());
        if let Some(dump) = envelope.logits_dump.as_mut() {
            dump.rows = 1;
        }
        assert!(envelope.validate().unwrap_err().contains("does not match"));
    }

    #[test]
    fn a_run_with_mismatched_positions_or_missing_errors_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut envelope = ran_envelope(dir.path());
        envelope.positions_run = 1;
        assert!(envelope.validate().unwrap_err().contains("positions_run"));

        let mut envelope = ran_envelope(dir.path());
        envelope.positions_run = 2;
        envelope.tokens.positions = vec![130, 129];
        envelope.tokens.start_position = Some(130);
        assert!(envelope.validate().unwrap_err().contains("increasing"));

        let mut envelope = ran_envelope(dir.path());
        envelope.error = Some("boom".to_string());
        assert!(envelope.validate().unwrap_err().contains("boom"));

        let mut envelope = ran_envelope(dir.path());
        envelope.launch_census = serde_json::json!({"routed_experts_down": false});
        assert!(
            envelope
                .validate()
                .unwrap_err()
                .contains("routed_experts_down")
        );
    }

    #[test]
    fn the_dump_carries_the_teacher_forced_reference_vector_per_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        let positions = [129usize, 130, 131];
        let tokens = [491u32, 367, 201];
        let references = [491u32, 367, 201];
        let logits = vec![
            synthetic_logits(490, 1),
            synthetic_logits(367, 2),
            synthetic_logits(201, 3),
        ];
        let dump = write_dump(
            dir.path(),
            0,
            &positions,
            &tokens,
            &logits,
            Some(&references),
        )
        .expect("dump");
        let text = std::fs::read_to_string(&dump.path).expect("jsonl");
        let rows: Vec<serde_json::Value> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("row"))
            .collect();
        assert_eq!(rows.len(), 3);
        for (index, reference) in references.iter().enumerate() {
            assert_eq!(rows[index]["position"], positions[index]);
            assert_eq!(rows[index]["reference_id"], *reference);
            assert_eq!(
                rows[index]["exact_token_equal"], true,
                "row {index} matches its reference"
            );
        }
        // A mismatched reference is now visible: row 1's hybrid token is 367 and its reference was
        // changed to 368, so exactly that row must read false.
        let mismatched = [491u32, 368, 201];
        let dump = write_dump(
            dir.path(),
            0,
            &positions,
            &tokens,
            &logits,
            Some(&mismatched),
        )
        .expect("mismatched dump");
        let text = std::fs::read_to_string(&dump.path).expect("jsonl");
        let rows: Vec<serde_json::Value> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("row"))
            .collect();
        assert_eq!(rows[0]["exact_token_equal"], true);
        assert_eq!(rows[1]["exact_token_equal"], false);
        assert_eq!(rows[2]["exact_token_equal"], true);
        // The fixture's nine-reference vector is what the token run will hand the writer.
        let fixture = [491u32, 367, 201, 80, 3302, 201, 54, 201, 4341];
        assert_eq!(fixture.len(), 9);
        // A reference vector that does not pair one-to-one with the rows is refused, never truncated.
        let error = write_dump(dir.path(), 0, &positions, &tokens, &logits, Some(&fixture))
            .expect_err("nine references for three rows");
        assert!(error.contains("reference ids for 3 positions"), "{error}");
    }

    #[test]
    fn the_dump_reuses_the_frozen_row_fields_and_sidecar_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let positions = [200usize];
        let tokens = [42u32];
        let logits = vec![synthetic_logits(42, 1)];
        let dump = write_dump(dir.path(), 0, &positions, &tokens, &logits, None).expect("dump");
        let text = std::fs::read_to_string(&dump.path).expect("jsonl");
        let rows: Vec<serde_json::Value> = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("row"))
            .collect();
        let expected: Vec<String> = crate::logits_dump::FIELDS
            .iter()
            .map(|field| (*field).to_owned())
            .collect();
        assert_eq!(
            rows[0]
                .as_object()
                .expect("object")
                .keys()
                .cloned()
                .collect::<Vec<String>>(),
            expected
        );
        assert_eq!(rows[0]["position"], 200);
        assert_eq!(rows[0]["device_argmax"], 42);
        assert_eq!(rows[0]["logits_argmax"], 42);
        assert_eq!(rows[0]["hybrid_id"], 42);
        assert!(rows[0]["reference_id"].is_null());
        assert!(rows[0]["fixture_id"].is_null());
        // No reference id, no equality: a replay row cannot claim the reference's token.
        assert_eq!(rows[0]["exact_token_equal"], false);
        assert_eq!(rows[0]["vocab"], VOCAB);
        assert_eq!(rows[0]["logits_offset"], 0);
        assert_eq!(
            dump.bin,
            format!("{}.bin", dump.path),
            "the sidecar is the dump path plus .bin"
        );
        let bin = std::fs::read(&dump.bin).expect("bin");
        assert_eq!(bin.len(), DUMP_BLOCK_BYTES);
        assert_eq!(
            f32::from_le_bytes(bin[42 * 4..43 * 4].try_into().expect("four bytes")),
            7.5
        );
        assert_eq!(dump.bin_sha256.len(), 64);
        assert_eq!(dump.sha256.len(), 64);
    }

    #[test]
    fn a_dump_for_the_wrong_vocabulary_or_an_unpaired_position_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let error =
            write_dump(dir.path(), 0, &[0], &[0], &[vec![0.0; 10]], None).expect_err("short");
        assert!(error.contains("expected 129280"), "{error}");
        let error = write_dump(dir.path(), 0, &[0, 1], &[0], &[], None).expect_err("unpaired");
        assert!(error.contains("do not pair"), "{error}");
        let error = write_dump(dir.path(), 0, &[], &[], &[], None).expect_err("empty");
        assert!(error.contains("nothing to dump"), "{error}");
    }

    #[test]
    fn tokens_push_keeps_positions_ordered_and_detects_a_bad_vocab_vector() {
        let mut tokens = Tokens::empty();
        tokens.push(0, &synthetic_logits(2, 1), 2).expect("first");
        assert!(
            tokens
                .push(0, &synthetic_logits(2, 1), 2)
                .unwrap_err()
                .contains("does not follow")
        );
        assert!(
            tokens
                .push(1, &[0.0; 10], 0)
                .unwrap_err()
                .contains("expected 129280")
        );
        tokens.push(1, &synthetic_logits(0, 1), 0).expect("second");
        assert_eq!(tokens.positions, vec![0, 1]);
    }

    #[test]
    fn step_stats_are_the_measured_samples_only() {
        let stats = StepMs::from_samples(vec![10.0, 20.0, 5.0]);
        assert!((stats.mean_ms - 35.0 / 3.0).abs() < 1e-9);
        assert!((stats.worst_ms - 20.0).abs() < 1e-9);
        let empty = StepMs::default();
        assert_eq!(empty.mean_ms, 0.0);
        assert_eq!(empty.worst_ms, 0.0);
    }
}

#[cfg(test)]
mod consumer_fixture_tests {
    use super::*;

    fn logits_with(winner: usize) -> Vec<f32> {
        let mut logits = vec![-2.0f32; VOCAB];
        logits[winner] = 9.5;
        logits[(winner + 1) % VOCAB] = 8.0;
        logits
    }

    /// The shape qa-lead's consumer accepted in the offline dry run: four rank envelopes, identical
    /// consensus ids on every rank, rank 0 the primary dump owner and replicas without a dump.
    ///
    /// This test builds that fixture in a temp directory and checks every rule the consumer reads
    /// (rank uniqueness, token consistency, dump pointer/rows/positions) so a schema drift fails
    /// here rather than in the window review. It is not a token claim: the values are synthetic.
    #[test]
    fn four_rank_run_evidence_is_consumer_shaped() {
        let root = tempfile::tempdir().expect("tempdir");
        let positions = [129usize, 130];
        let tokens = [5u32, 1];
        let logits = vec![logits_with(5), logits_with(1)];
        let rank0_dir = root.path().join("rank0");
        let dump =
            write_dump(&rank0_dir, 0, &positions, &tokens, &logits, None).expect("rank0 dump");

        let mut consensus = Tokens::empty();
        for index in 0..2 {
            consensus
                .push(positions[index], &logits[index], tokens[index])
                .expect("push");
        }
        consensus.start_position = Some(positions[0]);
        consensus.prompt_len = Some(130);

        let mut envelopes = Vec::new();
        for rank in 0..4 {
            let dir = root.path().join(format!("rank{rank}"));
            std::fs::create_dir_all(&dir).expect("rank dir");
            let envelope = TokenEnvelope::ran(
                rank,
                4,
                consensus.clone(),
                StepMs::from_samples(vec![41.0, 22.0]),
                vec![47.8, 29.0],
                REDUCE_CALLS_PER_STEP,
                if rank == 0 {
                    dump.clone()
                } else {
                    // A replica dump is allowed but not required; the contract keeps rank 0 as
                    // the primary owner so the consumer has exactly one artefact to read.
                    LogitsDump {
                        path: format!("replica rank {rank} has no dump"),
                        bin: String::new(),
                        sha256: String::new(),
                        bin_sha256: String::new(),
                        dtype: "f32-le".to_string(),
                        vocab: VOCAB,
                        positions: positions.to_vec(),
                        rows: positions.len(),
                    }
                },
                serde_json::json!({"routed_experts_down": true}),
            );
            let mut envelope = envelope;
            if rank != 0 {
                envelope.logits_dump = None;
            }
            envelope.validate().expect("consumer-shaped envelope");
            envelopes.push(envelope);
        }
        assert_eq!(
            envelopes[0].logits_dump.as_ref().expect("rank0 dump").rows,
            positions.len()
        );
        assert!(
            envelopes[1..]
                .iter()
                .all(|envelope| envelope.logits_dump.is_none())
        );
        // Rank uniqueness and id consistency are the two cross-rank rules the consumer enforces.
        let ranks: std::collections::BTreeSet<usize> =
            envelopes.iter().map(|envelope| envelope.rank).collect();
        assert_eq!(ranks, std::collections::BTreeSet::from([0, 1, 2, 3]));
        let sequences: Vec<&Vec<u32>> = envelopes
            .iter()
            .map(|envelope| &envelope.tokens.ids)
            .collect();
        assert!(sequences.windows(2).all(|pair| pair[0] == pair[1]));
        // And the dump on disk is exactly the frozen two-block artefact.
        let dump = envelopes[0].logits_dump.as_ref().expect("dump");
        assert_eq!(
            std::fs::read(&dump.bin).expect("bin").len(),
            positions.len() * DUMP_BLOCK_BYTES
        );
    }

    #[test]
    fn a_replica_run_without_a_dump_is_valid_but_must_not_claim_a_launch_it_did_not_make() {
        let dir = tempfile::tempdir().expect("tempdir");
        let logits = vec![logits_with(5)];
        let dump = write_dump(dir.path(), 0, &[129], &[5], &logits, None).expect("dump");
        let mut tokens = Tokens::empty();
        tokens.push(129, &logits[0], 5).expect("push");
        tokens.start_position = Some(129);
        tokens.prompt_len = Some(130);
        let mut replica = TokenEnvelope::ran(
            1,
            4,
            tokens,
            StepMs::from_samples(vec![20.0]),
            vec![20.0],
            REDUCE_CALLS_PER_STEP,
            dump,
            serde_json::json!({"routed_experts_down": true}),
        );
        replica.logits_dump = None;
        replica
            .validate()
            .expect("replica without a dump is allowed");
        replica.launch_census = serde_json::json!({"routed_experts_down": false});
        assert!(
            replica
                .validate()
                .unwrap_err()
                .contains("reduce_calls_per_step")
        );
    }
}
