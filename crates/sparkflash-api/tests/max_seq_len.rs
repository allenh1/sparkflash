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

//! DECISIONS.md D5: SPARK_MAX_SEQ_LEN is the sequence length actually served.
//!
//! Its own test binary because it mutates a process-global environment variable, which
//! would otherwise be visible to the other integration tests running in parallel. Only the
//! async test touches the environment; the pure parser test reads nothing from it.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use serde_json::{Value, json};
use sparkflash_api::engine::{MODEL_CONTEXT_TOKENS, MockEngine, max_seq_len, parse_max_seq_len};
use sparkflash_api::server::router;
use sparkflash_api::state::AppState;
use tower::ServiceExt as _;

const MODEL: &str = "deepseek-ai/DeepSeek-V4.1-Flash";
const ENV: &str = "SPARK_MAX_SEQ_LEN";

fn app() -> Router {
    router(AppState::new(Arc::new(MockEngine::new(MODEL))).with_fixed_time(1_700_000_000))
}

/// GET a route on a router built now, so the value is read fresh on every request.
async fn get(uri: &str) -> Value {
    let request = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .expect("request");
    let response = app().oneshot(request).await.expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("json")
}

async fn served_context_length() -> Value {
    get("/v1/models").await["data"][0]["context_length"].clone()
}

async fn served_cluster_max_seq_len() -> Value {
    get("/v1/cluster/stats").await["max_seq_len"].clone()
}

fn set_env(value: &str) {
    // SAFETY: only the async test mutates the environment and no other thread in this
    // binary reads it concurrently.
    unsafe { std::env::set_var(ENV, value) };
}

fn clear_env() {
    // SAFETY: as above.
    unsafe { std::env::remove_var(ENV) };
}

#[test]
fn boundary_values_are_accepted_or_rejected_explicitly() {
    assert_eq!(
        MODEL_CONTEXT_TOKENS, 1_048_576,
        "D5 sets the default window"
    );
    // Exactly the model window is legal; one token more is not a window we have.
    assert_eq!(parse_max_seq_len("1048576"), Some(1_048_576));
    assert_eq!(parse_max_seq_len("1048577"), None);
    assert_eq!(parse_max_seq_len(&usize::MAX.to_string()), None);
    // usize::from_str accepts a leading sign, so the digits are checked first.
    assert_eq!(parse_max_seq_len("+131072"), None);
    assert_eq!(parse_max_seq_len("-1"), None);
    // Underscore separators are not decimal digits.
    assert_eq!(parse_max_seq_len("1_048_576"), None);
    assert_eq!(parse_max_seq_len("0"), None);
    assert_eq!(parse_max_seq_len(""), None);
    assert_eq!(parse_max_seq_len("not-a-number"), None);
    // Surrounding whitespace survives shell quoting, so it is tolerated.
    assert_eq!(parse_max_seq_len("131072\n"), Some(131_072));
    assert_eq!(parse_max_seq_len("  131072  "), Some(131_072));
}

#[tokio::test]
async fn spark_max_seq_len_overrides_the_served_window() {
    clear_env();
    assert_eq!(max_seq_len(), MODEL_CONTEXT_TOKENS);
    assert_eq!(served_context_length().await, json!(MODEL_CONTEXT_TOKENS));
    assert_eq!(
        served_cluster_max_seq_len().await,
        json!(MODEL_CONTEXT_TOKENS)
    );

    // A usable override reaches both endpoints.
    set_env("262144");
    assert_eq!(max_seq_len(), 262_144);
    assert_eq!(served_context_length().await, json!(262_144));
    assert_eq!(served_cluster_max_seq_len().await, json!(262_144));

    // Exactly the model window is accepted.
    set_env("1048576");
    assert_eq!(max_seq_len(), 1_048_576);
    assert_eq!(served_context_length().await, json!(1_048_576));
    assert_eq!(served_cluster_max_seq_len().await, json!(1_048_576));

    // Everything unusable falls back to the default instead of being advertised.
    let oversized = usize::MAX.to_string();
    for rejected in [
        "1048577",
        oversized.as_str(),
        "+131072",
        "-1",
        "1_048_576",
        "0",
        "",
        "not-a-number",
    ] {
        set_env(rejected);
        assert_eq!(max_seq_len(), MODEL_CONTEXT_TOKENS, "rejected {rejected:?}");
        assert_eq!(
            served_context_length().await,
            json!(MODEL_CONTEXT_TOKENS),
            "rejected {rejected:?}"
        );
        assert_eq!(
            served_cluster_max_seq_len().await,
            json!(MODEL_CONTEXT_TOKENS),
            "rejected {rejected:?}"
        );
    }

    // A trailing newline or space is tolerated.
    set_env("131072\n");
    assert_eq!(max_seq_len(), 131_072);
    assert_eq!(served_context_length().await, json!(131_072));
    assert_eq!(served_cluster_max_seq_len().await, json!(131_072));

    clear_env();
    assert_eq!(max_seq_len(), MODEL_CONTEXT_TOKENS);
}
