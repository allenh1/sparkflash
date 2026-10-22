//! Device-chain token source and OpenAI wire assembly for the S7/S8 scaffold.
//!
//! This module is the serving seam that will carry a claimed four-rank device token envelope into
//! /v1/completions and /v1/chat/completions. It is deliberately split in two: a
//! [DeviceTokenSource] returns either a claimed nine-token stream or a typed fallback reason, and
//! the pure assembly functions turn a claimed stream into the same OpenAI payloads the engine path
//! serves. The router runs the host twin (the existing reference engine path) whenever the source
//! falls back, so an unclaimed envelope never changes the wire behaviour of the host path.
//!
//! The source reads a static claimed envelope directory. That is the S8 probe shape: replay one
//! recorded token window against the request that produced it. A live device source will replace
//! the reader with a tail over the window directory and a prompt-match check once a token claims;
//! docs/build/serving-s7-s8-plan.md records exactly what changes. Nothing here measures tokens
//! per second or time to first token, and every test is synthetic and labelled UNMEASURED.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};

use crate::device_chain::{ClaimMode, ClaimedEnvelope, DECODE_POSITIONS, DeviceChainEnvelope};
use crate::engine::FinishReason;
use crate::telemetry_types::OpenAiUsageWithCache;

/// Which OpenAI endpoint asked for the token stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceEndpoint {
    /// POST /v1/completions.
    Completions,
    /// POST /v1/chat/completions.
    ChatCompletions,
}

/// The request facts a device token source is allowed to consult before claiming.
///
/// The source must never look at the engine: a claim is either the recorded device envelope or a
/// named fallback. The prompt hints exist so the future live source can refuse a request whose
/// prompt does not match the recorded window instead of replaying tokens to the wrong request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceTokenRequest {
    /// The endpoint that received the request.
    pub endpoint: DeviceEndpoint,
    /// Whether the client asked for SSE.
    pub stream: bool,
    /// Whether stream_options.include_usage was set.
    pub include_usage: bool,
    /// The model id the request resolved to (already validated against the host engine).
    pub model: Option<String>,
    /// The completion cap, if the client set one.
    pub max_tokens: Option<usize>,
    /// The pre-tokenized prompt, when the request carried token ids.
    pub prompt_token_ids: Option<Vec<u32>>,
}

/// One claimed token and its decoded piece.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceToken {
    /// Absolute model position of the token.
    pub position: u32,
    /// The claimed token id.
    pub token_id: u32,
    /// The decoded piece this token contributes to the response text.
    pub text: String,
}

/// A claimed token stream ready for the OpenAI assembly.
#[derive(Debug, Clone)]
pub struct DeviceTokenStream {
    /// The claimed tokens, in decode order.
    pub tokens: Vec<DeviceToken>,
    /// Prompt tokens consumed; the envelope's prefill count, never a guess.
    pub prompt_tokens: u64,
    /// The finish reason the scaffold reports for a fixed device window.
    pub finish_reason: FinishReason,
    /// The reference ids the device loop compared itself against (for the S8 acceptance check).
    pub reference_ids: Vec<u32>,
    /// Decode positions the claimed envelope covers.
    pub decode_positions: u32,
    /// Whether the claimed envelope also carried the nine-position logits dump.
    pub logits_dump_present: bool,
    /// How the claim was established (an explicit device assertion or the derived loop evidence).
    pub claim_mode: ClaimMode,
    /// The identity of the source that produced this claim (`source_name` of the source).
    ///
    /// Carried on the stream rather than read from the router's `AppState`, because the response
    /// header names *which* device source answered: a live window follower and a one-shot envelope
    /// are different provenances and an operator reading a capture needs to tell them apart.
    pub source: &'static str,
    /// Human-readable provenance of the claimed record (directory, world, dump state).
    pub provenance: String,
}

impl DeviceTokenStream {
    /// The wire label for the claim form behind this stream.
    #[must_use]
    pub fn claim_label(&self) -> &'static str {
        self.claim_mode.label()
    }
}

/// A source's decision for one request.
#[derive(Debug, Clone)]
pub enum DeviceTokenClaim {
    /// Serve this request from the claimed device token stream.
    Claimed(Box<DeviceTokenStream>),
    /// Serve this request with the host twin; the reason is for diagnostics only.
    Fallback {
        /// Why the device source cannot serve this request.
        reason: String,
    },
}

/// The seam a device token source implements.
pub trait DeviceTokenSource: Send + Sync {
    /// A short name for diagnostics and the response label header.
    fn source_name(&self) -> &'static str {
        "device-chain-envelope"
    }

    /// Claims the envelope for one request, or names why the host twin must serve it.
    fn claim(&self, request: &DeviceTokenRequest) -> DeviceTokenClaim;
}

/// Decodes one claimed token id, given the ids already emitted.
///
/// The CLI wiring will implement this over the checkpoint tokenizer seam; the source never
/// guesses text from an id, so a missing decoder is a fallback rather than a placeholder string.
pub trait TokenPieceDecoder: Send + Sync {
    /// The piece contributed by token_id after ids_so_far.
    ///
    /// # Errors
    /// The error text becomes the fallback reason when the host twin must serve the request.
    fn piece(&self, ids_so_far: &[u32], token_id: u32) -> Result<String, String>;
}

/// Reads a static claimed envelope directory and serves its nine tokens.
pub struct DeviceChainTokenSource {
    dir: PathBuf,
    decoder: Arc<dyn TokenPieceDecoder>,
    claim_mode: ClaimMode,
}

impl DeviceChainTokenSource {
    /// Builds a source over one token-window directory.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>, decoder: Arc<dyn TokenPieceDecoder>) -> Self {
        Self {
            dir: dir.into(),
            decoder,
            // The default is the strict S8 form: a window is served only when the device writer
            // asserts token_claimed on every rank. The S7 rehearsal mode is opt-in.
            claim_mode: ClaimMode::Asserted,
        }
    }

    /// Selects the claim form this source accepts.
    ///
    /// [ClaimMode::Derived] exists for the S7 rehearsal against a window whose writer does not yet
    /// flip token_claimed; it requires the complete four-rank loop evidence and never substitutes
    /// for the S8 acceptance form.
    #[must_use]
    pub fn with_claim_mode(mut self, claim_mode: ClaimMode) -> Self {
        self.claim_mode = claim_mode;
        self
    }

    /// The claim form this source accepts.
    #[must_use]
    pub fn claim_mode(&self) -> ClaimMode {
        self.claim_mode
    }

    /// The envelope directory this source reads.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn reference_ids(&self, claimed: &ClaimedEnvelope) -> Vec<u32> {
        claimed.metadata.reference_ids.clone()
    }

    /// Verifies that the request is the prompt the envelope was recorded with.
    ///
    /// A recorded window is a replay harness: serving it for a different prompt would answer the
    /// wrong question with real-looking device tokens. The scaffold therefore requires both the
    /// request's token prompt and the envelope's recorded prompt ids, and matches them exactly. The
    /// m1y7 records do not carry prompt ids yet, so until the device side writes them (wiring point
    /// W2 in docs/build/serving-s7-s8-plan.md) every request falls back to the host twin, including
    /// every chat request: a chat text prompt has no token ids to compare.
    fn verify_prompt(
        &self,
        claimed: &ClaimedEnvelope,
        request: &DeviceTokenRequest,
    ) -> Result<(), String> {
        let Some(requested) = request.prompt_token_ids.as_ref() else {
            return Err(
                "the envelope is a recorded window: the request must carry the exact prompt token ids"
                    .to_owned(),
            );
        };
        if claimed.metadata.prompt_ids.is_empty() {
            return Err(
                "the envelope does not record prompt_ids, so the request prompt cannot be matched"
                    .to_owned(),
            );
        }
        if &claimed.metadata.prompt_ids != requested {
            return Err(format!(
                "the request prompt ({} ids) does not match the recorded window ({} ids)",
                requested.len(),
                claimed.metadata.prompt_ids.len()
            ));
        }
        Ok(())
    }

    fn decode(&self, claimed: &ClaimedEnvelope) -> Result<Vec<DeviceToken>, String> {
        let ids: Vec<u32> = claimed.tokens.iter().map(|token| token.token_id).collect();
        let mut tokens = Vec::with_capacity(ids.len());
        for (index, token) in claimed.tokens.iter().enumerate() {
            let piece = self
                .decoder
                .piece(&ids[..index], token.token_id)
                .map_err(|error| format!("detokenizer failed at position {index}: {error}"))?;
            tokens.push(DeviceToken {
                position: token.position,
                token_id: token.token_id,
                text: piece,
            });
        }
        Ok(tokens)
    }
}

impl DeviceTokenSource for DeviceChainTokenSource {
    fn claim(&self, request: &DeviceTokenRequest) -> DeviceTokenClaim {
        let envelope = match DeviceChainEnvelope::load(&self.dir) {
            Ok(envelope) => envelope,
            Err(error) => {
                return DeviceTokenClaim::Fallback {
                    reason: format!("device envelope unavailable: {error}"),
                };
            }
        };
        let claimed = match envelope.claim_with(self.claim_mode) {
            Ok(claimed) => claimed,
            Err(refusal) => {
                return DeviceTokenClaim::Fallback {
                    reason: format!("device token not claimed: {refusal}"),
                };
            }
        };
        if let Err(reason) = self.verify_prompt(&claimed, request) {
            return DeviceTokenClaim::Fallback { reason };
        }
        let mut tokens = match self.decode(&claimed) {
            Ok(tokens) => tokens,
            Err(error) => {
                return DeviceTokenClaim::Fallback { reason: error };
            }
        };
        // The scaffold serves a fixed window: a max_tokens smaller than the claimed batch truncates
        // it, and the finish reason is length in both cases because a device window has no EOS
        // marker yet. docs/build/serving-s7-s8-plan.md records that rule as a wiring point.
        if let Some(limit) = request.max_tokens {
            tokens.truncate(limit);
        }
        let decode_positions = claimed.metadata.decode_positions.max(DECODE_POSITIONS);
        let dump_present = claimed.logits_dump.is_some();
        DeviceTokenClaim::Claimed(Box::new(DeviceTokenStream {
            tokens,
            prompt_tokens: u64::from(claimed.metadata.prompt_len),
            finish_reason: FinishReason::Length,
            reference_ids: self.reference_ids(&claimed),
            decode_positions,
            logits_dump_present: dump_present,
            claim_mode: self.claim_mode,
            source: self.source_name(),
            provenance: format!(
                "{} (world {}, prompt_len {}, first_step_position {}, claim {}, logits_dump {})",
                self.dir.display(),
                envelope.world(),
                claimed.metadata.prompt_len,
                claimed.metadata.first_step_position,
                self.claim_mode.label(),
                if dump_present { "present" } else { "absent" }
            ),
        }))
    }
}

#[derive(Debug, Clone)]
struct CachedWindowClaim {
    dir: PathBuf,
    mtime: Option<std::time::SystemTime>,
    run_id: Option<String>,
    envelope: ClaimedEnvelope,
}

/// A live window follower token source for serving device tokens as windows complete.
///
/// Implements W5 from `docs/build/serving-s7-s8-plan.md`: instead of requiring a fixed static
/// directory, this source monitors an active window directory or parent directory, invalidates
/// cache when the run ID or modification timestamp transitions, ensures all rank pairs are
/// present, verifies the prompt, and serves claimed tokens.
pub struct LiveWindowTokenSource {
    dir: PathBuf,
    decoder: Arc<dyn TokenPieceDecoder>,
    claim_mode: ClaimMode,
    cached: std::sync::Mutex<Option<CachedWindowClaim>>,
}

impl LiveWindowTokenSource {
    /// Builds a live follower source over a window or parent directory.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>, decoder: Arc<dyn TokenPieceDecoder>) -> Self {
        Self {
            dir: dir.into(),
            decoder,
            claim_mode: ClaimMode::Asserted,
            cached: std::sync::Mutex::new(None),
        }
    }

    /// Selects the claim form this source accepts.
    #[must_use]
    pub fn with_claim_mode(mut self, claim_mode: ClaimMode) -> Self {
        self.claim_mode = claim_mode;
        self
    }

    /// The claim form this source accepts.
    #[must_use]
    pub fn claim_mode(&self) -> ClaimMode {
        self.claim_mode
    }

    /// The directory this source monitors.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn resolve_active_window(&self) -> Result<PathBuf, String> {
        if !self.dir.is_dir() {
            return Err(format!(
                "monitored path {} is not a directory",
                self.dir.display()
            ));
        }
        if self.dir.join("tp4-token-r0.json").is_file()
            || self.dir.join("rank0").join("tp4-token-r0.json").is_file()
        {
            return Ok(self.dir.clone());
        }
        let entries = std::fs::read_dir(&self.dir)
            .map_err(|error| format!("read dir {}: {error}", self.dir.display()))?;
        let mut candidates = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir()
                && (path.join("tp4-token-r0.json").is_file()
                    || path.join("rank0").join("tp4-token-r0.json").is_file())
            {
                let mtime = entry.metadata().and_then(|m| m.modified()).ok();
                candidates.push((path, mtime));
            }
        }
        if candidates.is_empty() {
            return Err(format!(
                "no completed device token window found in {}",
                self.dir.display()
            ));
        }
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
        Ok(candidates.remove(0).0)
    }

    fn read_run_id(dir: &Path) -> Option<String> {
        let meta_path = dir.join("run.meta");
        if let Ok(content) = std::fs::read_to_string(meta_path) {
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("run=") {
                    return Some(rest.split_whitespace().next().unwrap_or(rest).to_owned());
                }
            }
        }
        None
    }

    fn window_mtime(dir: &Path) -> Option<std::time::SystemTime> {
        let token0 = if dir.join("tp4-token-r0.json").is_file() {
            dir.join("tp4-token-r0.json")
        } else {
            dir.join("rank0").join("tp4-token-r0.json")
        };
        std::fs::metadata(token0).and_then(|m| m.modified()).ok()
    }

    fn decode(&self, claimed: &ClaimedEnvelope) -> Result<Vec<DeviceToken>, String> {
        let ids: Vec<u32> = claimed.tokens.iter().map(|token| token.token_id).collect();
        let mut tokens = Vec::with_capacity(ids.len());
        for (index, token) in claimed.tokens.iter().enumerate() {
            let piece = self
                .decoder
                .piece(&ids[..index], token.token_id)
                .map_err(|error| format!("detokenizer failed at position {index}: {error}"))?;
            tokens.push(DeviceToken {
                position: token.position,
                token_id: token.token_id,
                text: piece,
            });
        }
        Ok(tokens)
    }

    fn verify_prompt(
        &self,
        claimed: &ClaimedEnvelope,
        request: &DeviceTokenRequest,
    ) -> Result<(), String> {
        let Some(requested) = request.prompt_token_ids.as_ref() else {
            return Err(
                "live device window requires explicit prompt token ids to verify alignment"
                    .to_owned(),
            );
        };
        if claimed.metadata.prompt_ids.is_empty() {
            return Err(
                "the envelope does not record prompt_ids, so the request prompt cannot be matched"
                    .to_owned(),
            );
        }
        if &claimed.metadata.prompt_ids != requested {
            return Err(format!(
                "the request prompt ({} ids) does not match the active window ({} ids)",
                requested.len(),
                claimed.metadata.prompt_ids.len()
            ));
        }
        Ok(())
    }
}

impl DeviceTokenSource for LiveWindowTokenSource {
    fn source_name(&self) -> &'static str {
        "device-chain-live-window"
    }

    fn claim(&self, request: &DeviceTokenRequest) -> DeviceTokenClaim {
        let active_dir = match self.resolve_active_window() {
            Ok(dir) => dir,
            Err(reason) => return DeviceTokenClaim::Fallback { reason },
        };
        let mtime = Self::window_mtime(&active_dir);
        let run_id = Self::read_run_id(&active_dir);

        let mut lock = self
            .cached
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let needs_reload = match lock.as_ref() {
            Some(cached) => {
                cached.dir != active_dir || cached.mtime != mtime || cached.run_id != run_id
            }
            None => true,
        };

        if needs_reload {
            let envelope = match DeviceChainEnvelope::load(&active_dir) {
                Ok(envelope) => envelope,
                Err(error) => {
                    return DeviceTokenClaim::Fallback {
                        reason: format!(
                            "live device envelope load failed in {}: {error}",
                            active_dir.display()
                        ),
                    };
                }
            };
            let claimed = match envelope.claim_with(self.claim_mode) {
                Ok(claimed) => claimed,
                Err(refusal) => {
                    return DeviceTokenClaim::Fallback {
                        reason: format!(
                            "live device token not claimed in {}: {refusal}",
                            active_dir.display()
                        ),
                    };
                }
            };
            *lock = Some(CachedWindowClaim {
                dir: active_dir,
                mtime,
                run_id,
                envelope: claimed,
            });
        }

        let cached = lock.as_ref().expect("cached claim is present");
        let claimed = &cached.envelope;

        if let Err(reason) = self.verify_prompt(claimed, request) {
            return DeviceTokenClaim::Fallback { reason };
        }

        let mut tokens = match self.decode(claimed) {
            Ok(tokens) => tokens,
            Err(error) => return DeviceTokenClaim::Fallback { reason: error },
        };

        if let Some(limit) = request.max_tokens {
            tokens.truncate(limit);
        }

        let decode_positions = claimed.metadata.decode_positions.max(DECODE_POSITIONS);
        let dump_present = claimed.logits_dump.is_some();

        DeviceTokenClaim::Claimed(Box::new(DeviceTokenStream {
            tokens,
            prompt_tokens: u64::from(claimed.metadata.prompt_len),
            finish_reason: FinishReason::Length,
            reference_ids: claimed.metadata.reference_ids.clone(),
            decode_positions,
            logits_dump_present: dump_present,
            claim_mode: self.claim_mode,
            source: self.source_name(),
            provenance: format!(
                "live-window: {} (prompt_len {}, first_step_position {}, claim {}, run_id {})",
                cached.dir.display(),
                claimed.metadata.prompt_len,
                claimed.metadata.first_step_position,
                self.claim_mode.label(),
                cached.run_id.as_deref().unwrap_or("none")
            ),
        }))
    }
}

/// The OpenAI response identity the assembly needs (filled by the router).
#[derive(Debug, Clone)]
pub struct DeviceResponseContext {
    /// Response id (already allocated from the state counter).
    pub id: String,
    /// Pinned or wall-clock creation timestamp.
    pub created: u64,
    /// Resolved model id.
    pub model: String,
    /// Whether the streamed response carries the final usage chunk.
    pub include_usage: bool,
}

impl DeviceTokenStream {
    /// The concatenated decoded text of the claimed tokens.
    #[must_use]
    pub fn text(&self) -> String {
        self.tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect()
    }

    /// The OpenAI usage block for this claimed stream.
    #[must_use]
    pub fn usage(&self) -> OpenAiUsageWithCache {
        let prompt_tokens = self.prompt_tokens as usize;
        let completion_tokens = self.tokens.len();
        OpenAiUsageWithCache {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_cache_hit_tokens: 0,
            prompt_cache_miss_tokens: prompt_tokens,
            ..OpenAiUsageWithCache::default()
        }
    }

    /// The usage block as the wire value.
    #[must_use]
    pub fn usage_value(&self) -> Value {
        serde_json::to_value(self.usage()).unwrap_or(Value::Null)
    }
}

/// Builds the buffered /v1/completions body from a claimed stream.
#[must_use]
pub fn device_completions_body(
    stream: &DeviceTokenStream,
    context: &DeviceResponseContext,
) -> Value {
    json!({
        "id": context.id,
        "object": "text_completion",
        "created": context.created,
        "model": context.model,
        "choices": [{
            "index": 0,
            "text": stream.text(),
            "logprobs": Value::Null,
            "finish_reason": stream.finish_reason,
            "stop_reason": Value::Null,
            "prompt_logprobs": Value::Null,
        }],
        "usage": stream.usage_value(),
    })
}

/// Builds the buffered /v1/chat/completions body from a claimed stream.
#[must_use]
pub fn device_chat_body(stream: &DeviceTokenStream, context: &DeviceResponseContext) -> Value {
    json!({
        "id": context.id,
        "object": "chat.completion",
        "created": context.created,
        "model": context.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": stream.text()},
            "logprobs": Value::Null,
            "finish_reason": stream.finish_reason,
        }],
        "usage": stream.usage_value(),
    })
}

/// Builds the /v1/completions SSE data frames (without the final [DONE] terminator).
#[must_use]
pub fn device_completions_frames(
    stream: &DeviceTokenStream,
    context: &DeviceResponseContext,
) -> Vec<Value> {
    let mut frames = Vec::with_capacity(stream.tokens.len() + 2);
    for token in &stream.tokens {
        frames.push(json!({
            "id": context.id,
            "object": "text_completion",
            "created": context.created,
            "model": context.model,
            "choices": [{
                "index": 0,
                "text": token.text,
                "logprobs": Value::Null,
                "finish_reason": Value::Null,
            }],
        }));
    }
    frames.push(device_finish_frame(stream, context, true));
    if context.include_usage {
        frames.push(device_usage_frame(stream, context, true));
    }
    frames
}

/// Builds the /v1/chat/completions SSE data frames (without the final [DONE] terminator).
#[must_use]
pub fn device_chat_frames(
    stream: &DeviceTokenStream,
    context: &DeviceResponseContext,
) -> Vec<Value> {
    let mut frames = Vec::with_capacity(stream.tokens.len() + 3);
    frames.push(json!({
        "id": context.id,
        "object": "chat.completion.chunk",
        "created": context.created,
        "model": context.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": Value::Null}],
    }));
    for token in &stream.tokens {
        frames.push(json!({
            "id": context.id,
            "object": "chat.completion.chunk",
            "created": context.created,
            "model": context.model,
            "choices": [{"index": 0, "delta": {"content": token.text}, "finish_reason": Value::Null}],
        }));
    }
    frames.push(device_finish_frame(stream, context, false));
    if context.include_usage {
        frames.push(device_usage_frame(stream, context, false));
    }
    frames
}

fn device_finish_frame(
    stream: &DeviceTokenStream,
    context: &DeviceResponseContext,
    completions: bool,
) -> Value {
    if completions {
        json!({
            "id": context.id,
            "object": "text_completion",
            "created": context.created,
            "model": context.model,
            "choices": [{
                "index": 0,
                "text": "",
                "logprobs": Value::Null,
                "finish_reason": stream.finish_reason,
            }],
        })
    } else {
        json!({
            "id": context.id,
            "object": "chat.completion.chunk",
            "created": context.created,
            "model": context.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": stream.finish_reason}],
        })
    }
}

fn device_usage_frame(
    stream: &DeviceTokenStream,
    context: &DeviceResponseContext,
    completions: bool,
) -> Value {
    json!({
        "id": context.id,
        "object": if completions { "text_completion" } else { "chat.completion.chunk" },
        "created": context.created,
        "model": context.model,
        "choices": [],
        "usage": stream.usage_value(),
    })
}
#[cfg(test)]
mod tests {
    //! Synthetic source and assembly tests only: UNMEASURED, no device run is implied.

    use super::*;
    use crate::device_chain::fixtures::{DECODE_IDS, PROMPT_LEN, write_envelope};
    use tempfile::TempDir;

    struct BracketedDecoder;

    impl TokenPieceDecoder for BracketedDecoder {
        fn piece(&self, _ids_so_far: &[u32], token_id: u32) -> Result<String, String> {
            Ok(format!("[{token_id}]"))
        }
    }

    struct FailingDecoder;

    impl TokenPieceDecoder for FailingDecoder {
        fn piece(&self, _ids_so_far: &[u32], _token_id: u32) -> Result<String, String> {
            Err("synthetic decoder failure".to_owned())
        }
    }

    fn request(endpoint: DeviceEndpoint, max_tokens: Option<usize>) -> DeviceTokenRequest {
        DeviceTokenRequest {
            endpoint,
            stream: false,
            include_usage: false,
            model: Some("synthetic-model".to_owned()),
            max_tokens,
            prompt_token_ids: Some(crate::device_chain::fixtures::prompt_ids()),
        }
    }

    /// A chat request: no token ids, so the scaffold cannot match the recorded window.
    fn chat_request() -> DeviceTokenRequest {
        DeviceTokenRequest {
            endpoint: DeviceEndpoint::ChatCompletions,
            stream: false,
            include_usage: false,
            model: Some("synthetic-model".to_owned()),
            max_tokens: None,
            prompt_token_ids: None,
        }
    }

    fn source(
        claimed: bool,
        decoder: Arc<dyn TokenPieceDecoder>,
    ) -> (TempDir, DeviceChainTokenSource) {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), claimed);
        let source = DeviceChainTokenSource::new(dir.path(), decoder);
        (dir, source)
    }

    fn claimed_stream(source: &DeviceChainTokenSource) -> DeviceTokenStream {
        match source.claim(&request(DeviceEndpoint::Completions, None)) {
            DeviceTokenClaim::Claimed(stream) => *stream,
            DeviceTokenClaim::Fallback { reason } => panic!("expected a claim, got: {reason}"),
        }
    }

    fn context(include_usage: bool) -> DeviceResponseContext {
        DeviceResponseContext {
            id: "cmpl-synthetic-00000000".to_owned(),
            created: 1_700_000_000,
            model: "synthetic-model".to_owned(),
            include_usage,
        }
    }

    #[test]
    fn a_claimed_envelope_serves_the_nine_tokens_and_the_usage_relation() {
        let (_dir, source) = source(true, Arc::new(BracketedDecoder));
        let stream = claimed_stream(&source);
        assert_eq!(stream.tokens.len(), 9);
        assert_eq!(stream.tokens[0].token_id, DECODE_IDS[0]);
        assert_eq!(stream.tokens[8].token_id, DECODE_IDS[8]);
        assert_eq!(stream.text(), "[11][22][33][44][55][66][77][88][99]");
        assert_eq!(stream.prompt_tokens, u64::from(PROMPT_LEN));
        let usage = stream.usage();
        assert_eq!(usage.prompt_tokens, PROMPT_LEN as usize);
        assert_eq!(usage.completion_tokens, 9);
        assert_eq!(usage.total_tokens, PROMPT_LEN as usize + 9);
        assert_eq!(stream.finish_reason, FinishReason::Length);
        assert!(stream.logits_dump_present);
        assert_eq!(stream.claim_label(), "explicit");
        assert_eq!(stream.decode_positions, DECODE_POSITIONS);
    }

    #[test]
    fn an_unclaimed_envelope_falls_back_with_the_refusal_reason() {
        let (_dir, source) = source(false, Arc::new(BracketedDecoder));
        match source.claim(&request(DeviceEndpoint::Completions, None)) {
            DeviceTokenClaim::Claimed(_) => panic!("an unclaimed envelope must fall back"),
            DeviceTokenClaim::Fallback { reason } => {
                assert!(
                    reason.contains("not claimed"),
                    "the fallback must name the refusal: {reason}"
                );
            }
        }
    }

    #[test]
    fn a_missing_directory_falls_back() {
        let source =
            DeviceChainTokenSource::new("/synthetic/does/not/exist", Arc::new(BracketedDecoder));
        match source.claim(&request(DeviceEndpoint::ChatCompletions, None)) {
            DeviceTokenClaim::Claimed(_) => panic!("a missing envelope must fall back"),
            DeviceTokenClaim::Fallback { reason } => {
                assert!(
                    reason.contains("unavailable"),
                    "the fallback must name the missing envelope: {reason}"
                );
            }
        }
    }

    #[test]
    fn a_decoder_failure_falls_back_instead_of_serving_placeholder_text() {
        let (_dir, source) = source(true, Arc::new(FailingDecoder));
        match source.claim(&request(DeviceEndpoint::Completions, None)) {
            DeviceTokenClaim::Claimed(_) => panic!("a decoder failure must fall back"),
            DeviceTokenClaim::Fallback { reason } => {
                assert!(
                    reason.contains("detokenizer failed at position 0"),
                    "the fallback must name the decoder failure: {reason}"
                );
            }
        }
    }

    #[test]
    fn max_tokens_truncates_the_claimed_window() {
        let (_dir, source) = source(true, Arc::new(BracketedDecoder));
        let request = request(DeviceEndpoint::Completions, Some(3));
        let DeviceTokenClaim::Claimed(stream) = source.claim(&request) else {
            panic!("a claimed envelope truncates instead of falling back");
        };
        assert_eq!(stream.tokens.len(), 3);
        assert_eq!(stream.usage().completion_tokens, 3);
        assert_eq!(stream.finish_reason, FinishReason::Length);
    }

    #[test]
    fn the_buffered_completions_body_is_openai_shaped() {
        let (_dir, source) = source(true, Arc::new(BracketedDecoder));
        let stream = claimed_stream(&source);
        let body = device_completions_body(&stream, &context(false));
        assert_eq!(body["object"], "text_completion");
        assert_eq!(body["choices"][0]["index"], 0);
        assert_eq!(body["choices"][0]["text"], stream.text());
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["prompt_tokens"], PROMPT_LEN);
        assert_eq!(body["usage"]["completion_tokens"], 9);
        assert_eq!(body["usage"]["total_tokens"], PROMPT_LEN + 9);
    }

    #[test]
    fn the_chat_frames_are_role_then_content_then_finish_then_usage() {
        let (_dir, source) = source(true, Arc::new(BracketedDecoder));
        let stream = claimed_stream(&source);
        let frames = device_chat_frames(&stream, &context(true));
        assert_eq!(frames.len(), 9 + 3);
        assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(frames[1]["choices"][0]["delta"]["content"], "[11]");
        let finish = &frames[10];
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        let usage = &frames[11];
        assert_eq!(usage["choices"], serde_json::json!([]));
        assert_eq!(usage["usage"]["completion_tokens"], 9);
        let body = device_chat_body(&stream, &context(false));
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["content"], stream.text());
    }

    #[test]
    fn completion_frames_carry_the_usage_chunk_only_when_asked() {
        let (_dir, source) = source(true, Arc::new(BracketedDecoder));
        let stream = claimed_stream(&source);
        let without = device_completions_frames(&stream, &context(false));
        assert_eq!(without.len(), 9 + 1);
        let with = device_completions_frames(&stream, &context(true));
        assert_eq!(with.len(), 9 + 2);
        assert_eq!(with[9]["choices"][0]["finish_reason"], "length");
        assert_eq!(with[10]["choices"], serde_json::json!([]));
    }

    #[test]
    fn a_prompt_that_is_not_the_recorded_window_falls_back() {
        let (_dir, source) = source(true, Arc::new(BracketedDecoder));
        let mut request = request(DeviceEndpoint::Completions, None);
        request.prompt_token_ids = Some(vec![1, 2, 3]);
        match source.claim(&request) {
            DeviceTokenClaim::Claimed(_) => {
                panic!("a recorded window must not answer a different prompt")
            }
            DeviceTokenClaim::Fallback { reason } => {
                assert!(
                    reason.contains("does not match the recorded window"),
                    "the fallback must name the prompt mismatch: {reason}"
                );
            }
        }
    }

    #[test]
    fn a_chat_request_without_token_ids_cannot_be_matched_and_falls_back() {
        let (_dir, source) = source(true, Arc::new(BracketedDecoder));
        match source.claim(&chat_request()) {
            DeviceTokenClaim::Claimed(_) => {
                panic!("a text chat prompt cannot be matched against the recorded window")
            }
            DeviceTokenClaim::Fallback { reason } => {
                assert!(
                    reason.contains("must carry the exact prompt token ids"),
                    "the fallback must name the missing prompt ids: {reason}"
                );
            }
        }
    }

    #[test]
    fn an_envelope_without_recorded_prompt_ids_falls_back() {
        let dir = tempfile::tempdir().expect("temp dir");
        write_envelope(dir.path(), true);
        for rank in 0..crate::device_chain::FOUR_RANK_WORLD {
            crate::device_chain::fixtures::rewrite_value(
                &dir.path().join(format!("tp4-forward-embed-r{rank}.json")),
                |record| {
                    record
                        .as_object_mut()
                        .expect("record object")
                        .remove("prompt_ids");
                },
            );
        }
        let source = DeviceChainTokenSource::new(dir.path(), Arc::new(BracketedDecoder));
        match source.claim(&request(DeviceEndpoint::Completions, None)) {
            DeviceTokenClaim::Claimed(_) => {
                panic!("an envelope without prompt ids must not replay a window")
            }
            DeviceTokenClaim::Fallback { reason } => {
                assert!(
                    reason.contains("does not record prompt_ids"),
                    "the fallback must name the missing prompt ids: {reason}"
                );
            }
        }
    }

    #[test]
    fn the_source_name_labels_the_claim_for_diagnostics() {
        let (_dir, source) = source(true, Arc::new(BracketedDecoder));
        assert_eq!(source.source_name(), "device-chain-envelope");
        assert!(source.dir().is_dir());
    }

    #[test]
    fn live_window_token_source_serves_from_monitored_directory() {
        let dir = tempfile::tempdir().expect("temp dir");
        let sub = dir.path().join("m1y7");
        std::fs::create_dir_all(&sub).expect("create sub");
        write_envelope(&sub, true);
        std::fs::write(sub.join("run.meta"), "run=m1y7 tag=test\n").expect("write run.meta");

        let source = LiveWindowTokenSource::new(dir.path(), Arc::new(BracketedDecoder));
        assert_eq!(source.source_name(), "device-chain-live-window");

        let req = request(DeviceEndpoint::ChatCompletions, None);
        match source.claim(&req) {
            DeviceTokenClaim::Claimed(stream) => {
                assert_eq!(stream.tokens.len(), 9);
                assert!(stream.provenance.contains("run_id m1y7"));
            }
            DeviceTokenClaim::Fallback { reason } => {
                panic!("expected claim, got fallback: {reason}")
            }
        }
    }

    #[test]
    fn live_window_token_source_invalidates_when_window_transitions() {
        let dir = tempfile::tempdir().expect("temp dir");
        let sub1 = dir.path().join("window_1");
        std::fs::create_dir_all(&sub1).expect("create sub1");
        write_envelope(&sub1, true);
        std::fs::write(sub1.join("run.meta"), "run=w1 tag=test\n").expect("write run.meta");

        let source = LiveWindowTokenSource::new(dir.path(), Arc::new(BracketedDecoder));
        let req = request(DeviceEndpoint::ChatCompletions, None);
        let DeviceTokenClaim::Claimed(s1) = source.claim(&req) else {
            panic!("first claim failed");
        };
        assert!(s1.provenance.contains("run_id w1"));

        std::thread::sleep(std::time::Duration::from_millis(15));
        let sub2 = dir.path().join("window_2");
        std::fs::create_dir_all(&sub2).expect("create sub2");
        write_envelope(&sub2, true);
        std::fs::write(sub2.join("run.meta"), "run=w2 tag=test\n").expect("write run.meta");

        let DeviceTokenClaim::Claimed(s2) = source.claim(&req) else {
            panic!("second claim failed");
        };
        assert!(s2.provenance.contains("run_id w2"));
    }
}
