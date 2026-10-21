//! OpenAI-shaped error responses.
//!
//! Every failure path in the router funnels through ApiError so clients always receive
//! `{"error": {"message", "type", "param", "code"}}` with a correct HTTP status and no
//! panics, including on malformed JSON bodies.

use std::fmt;

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

use crate::engine::MissingCapability;
use crate::multimodal::MultimodalError;
use crate::reasoning::ReasoningError;

/// The `error` member of an OpenAI error response.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    /// Human-readable message.
    pub message: String,
    /// OpenAI error type, e.g. `invalid_request_error`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Parameter that caused the error, when known.
    pub param: Option<String>,
    /// The error's HTTP status as an integer, which is vLLM's `ErrorInfo.code` type
    /// (`engine/protocol.py:54-62`: `code: int = 500`).
    pub code: u16,
}

/// The complete OpenAI error envelope.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    /// The error body.
    pub error: ErrorBody,
}

/// An HTTP error that renders as an OpenAI error envelope.
#[derive(Debug, Clone)]
pub struct ApiError {
    status: StatusCode,
    kind: String,
    message: String,
    param: Option<String>,
    code: Option<String>,
}

impl ApiError {
    /// Builds an error from all of its parts.
    #[must_use]
    pub fn new(
        status: StatusCode,
        kind: impl Into<String>,
        message: impl Into<String>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        Self {
            status,
            kind: kind.into(),
            message: message.into(),
            param,
            code,
        }
    }

    /// A 400 with the default OpenAI error type and code.
    #[must_use]
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
            None,
            Some("invalid_request".to_owned()),
        )
    }

    /// A 400 naming the offending parameter and code.
    #[must_use]
    pub fn invalid_request(
        message: impl Into<String>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message,
            param,
            code,
        )
    }

    /// A 401 for a missing or wrong bearer API key.
    ///
    /// The body is the usual OpenAI error envelope; like every other error, the wire `code` is the
    /// HTTP status integer (vLLM's `ErrorInfo.code` type), and `message` stays generic so a probe
    /// cannot distinguish "missing" from "wrong" key.
    #[must_use]
    pub fn unauthorized() -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "Unauthorized",
            None,
            Some("unauthorized".to_owned()),
        )
    }

    /// A 400 for a body that is not valid JSON.
    #[must_use]
    pub fn malformed_json(detail: impl fmt::Display) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("invalid JSON body: {detail}"),
            None,
            Some("invalid_json".to_owned()),
        )
    }

    /// A 404 for an unknown model id.
    #[must_use]
    pub fn model_not_found(model: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("The model '{model}' does not exist or you do not have access to it."),
            Some("model".to_owned()),
            Some("model_not_found".to_owned()),
        )
    }

    /// A 503 for a runtime that cannot serve the request.
    #[must_use]
    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            message,
            None,
            Some("engine_unavailable".to_owned()),
        )
    }

    /// A 500 for internal failures.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            message,
            None,
            Some("internal_error".to_owned()),
        )
    }

    /// Overrides the internal machine-readable code, which is kept for diagnostics only: the
    /// wire `code` is always the HTTP status, as vLLM's `ErrorInfo` defines it.
    #[must_use]
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// The internal machine-readable code, for diagnostics. It is not serialized.
    #[must_use]
    pub fn inner_code(&self) -> Option<&str> {
        self.code.as_deref()
    }

    /// HTTP status.
    #[must_use]
    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// The serializable envelope.
    #[must_use]
    pub fn envelope(&self) -> ErrorEnvelope {
        ErrorEnvelope {
            error: ErrorBody {
                message: self.message.clone(),
                kind: self.kind.clone(),
                param: self.param.clone(),
                code: self.status.as_u16(),
            },
        }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}: {}",
            self.status.as_u16(),
            self.kind,
            self.message
        )
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.envelope())).into_response()
    }
}

impl From<ReasoningError> for ApiError {
    fn from(error: ReasoningError) -> Self {
        let (param, code) = match &error {
            ReasoningError::EffortOutOfRange { .. }
            | ReasoningError::InvalidEffortType { .. }
            | ReasoningError::UnknownEffort { .. } => {
                ("reasoning_effort", "invalid_reasoning_effort")
            }
            ReasoningError::InvalidThinking { .. } => {
                ("chat_template_kwargs.thinking", "invalid_thinking")
            }
        };
        Self::invalid_request(
            error.to_string(),
            Some(param.to_owned()),
            Some(code.to_owned()),
        )
    }
}

impl From<MissingCapability> for ApiError {
    fn from(error: MissingCapability) -> Self {
        // A capability is the engine's own answer, so the refusal is a client error: the request
        // asked for a channel this deployment does not serve, and no retry can change that. `param`
        // is left unset because the image can arrive through `messages`, `prompt`, `input` or
        // `content`, so naming one field would be wrong on the other three routes; the message names
        // the missing capability and the code is `unsupported_capability`.
        Self::invalid_request(error.to_string(), None, Some(error.code().to_owned()))
    }
}

impl From<MultimodalError> for ApiError {
    fn from(error: MultimodalError) -> Self {
        let code = match &error {
            MultimodalError::MalformedDataUrl(_) => "invalid_image_data_url",
            MultimodalError::Unsupported(_) => "invalid_image_url",
        };
        Self::invalid_request(
            error.to_string(),
            Some("image_url".to_owned()),
            Some(code.to_owned()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_matches_the_openai_shape() {
        let error = ApiError::model_not_found("gpt-4");
        let json = serde_json::to_value(error.envelope()).expect("envelope");
        assert_eq!(error.status(), StatusCode::NOT_FOUND);
        let body = json.get("error").expect("error member");
        for key in ["message", "type", "param", "code"] {
            assert!(body.get(key).is_some(), "missing key {key}");
        }
        assert_eq!(body["type"], "invalid_request_error");
        assert_eq!(
            body["code"], 404,
            "vLLM's ErrorInfo.code is the HTTP status integer"
        );
        assert_eq!(body["param"], "model");
    }

    #[test]
    fn reasoning_errors_map_onto_400_envelopes() {
        let error: ApiError = ReasoningError::EffortOutOfRange { value: 101 }.into();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        let envelope = error.envelope();
        assert_eq!(envelope.error.param.as_deref(), Some("reasoning_effort"));
        assert_eq!(envelope.error.code, 400);

        let thinking: ApiError = ReasoningError::InvalidThinking {
            value: "x".to_owned(),
        }
        .into();
        assert_eq!(
            thinking.envelope().error.param.as_deref(),
            Some("chat_template_kwargs.thinking")
        );
    }

    #[test]
    fn multimodal_errors_map_onto_400_envelopes() {
        let error: ApiError = MultimodalError::MalformedDataUrl("nope".to_owned()).into();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error.envelope().error.code, 400);
        assert_eq!(error.envelope().error.param.as_deref(), Some("image_url"));
    }
}
