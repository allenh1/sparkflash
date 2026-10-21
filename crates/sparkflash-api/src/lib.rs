//! `sparkflash-api`: the OpenAI/Anthropic-compatible HTTP + SSE surface of SparkFlash-RS.
//!
//! The crate is deliberately split into a weight-free protocol layer (this file and the
//! telemetry / reasoning / loop-abort modules), an engine seam (crate::engine) that either
//! a real 4x DGX Spark engine or crate::engine::MockEngine can implement, and an axum
//! server (crate::server::router) that needs no GPU, no checkpoint and no network to test.

pub mod constrained;
pub mod device_chain;
pub mod device_token;
pub mod encoding;
pub mod engine;
pub mod error;
pub mod loop_abort;
pub mod metrics;
pub mod multimodal;
pub mod prompt;
pub mod reasoning;
pub mod schema;
pub mod server;
pub mod state;
pub mod telemetry_types;
pub mod tool_parser;

pub use encoding::{SparkV41Encoding, TokenizerSeam, TokenizerSeamError};
pub use engine::{
    Capabilities, Capability, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason,
    InferenceEngine, LogprobAlternative, LogprobSelection, LogprobsRequest, MODEL_ID,
    MissingCapability, MockEngine, NodeStats, PromptLogprobs, TokenLogprob, TokenString,
};
pub use error::ApiError;
pub use loop_abort::{LoopAbort, LoopAbortConfig, LoopAbortDetector};
pub use metrics::Metrics;
pub use multimodal::{ImagePart, MultimodalError, count_image_tokens};
pub use prompt::{PromptSummary, estimate_text_tokens};
pub use reasoning::{ReasoningError, ReasoningSpec};
pub use server::{GPU_TRACE_RING_CAPACITY, TraceDirective, arm_traces, disarm_traces, router};
pub use state::AppState;
pub use telemetry_types::{
    CompletionTokensDetails, DSparkStepStats, DeepSeekSparkRequestStats, OpenAiUsageWithCache,
    PromptTokensDetails, RequestTimingStats, RoceTransportStats,
};
