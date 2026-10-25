//! End-to-end HTTP tests for `sparkflash serve` over the synthetic tiny plan.
//!
//! These are the only tests that bind a socket and speak HTTP to the real binary: the router
//! tests in `sparkflash-api/tests/router.rs` drive the axum router in-process against MockEngine,
//! and the tests inside `serve.rs` drive the reference driver directly. This file closes the gap
//! for the surfaces a client sees: the wire response bodies, the SSE framing, `[DONE]`, the usage
//! chunk, and the vLLM error envelope.
//!
//! The server is the production `serve --synthetic` path (no checkpoint on this host), bound to
//! `127.0.0.1:0` so the kernel picks the port, using token-id prompts because the synthetic tiny
//! plan's vocabulary is 128 rows and the released tokenizer's ids mostly do not fit it. It is a
//! host-path test: no GPU, no checkpoint, no device. The tiny plan is a reference-arithmetic
//! exercise, not a model-quality or performance measurement.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use sparkflash_api::device_chain::fixtures;

/// How long the server gets to load the tiny plan and print its bound address.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
/// Per-request socket budget: the synthetic plan is small, so this is generous, not a measurement.
const IO_TIMEOUT: Duration = Duration::from_secs(120);

/// A `sparkflash serve --synthetic` process bound to a kernel-picked loopback port.
///
/// The address line goes to stdout (`println!` in serve.rs) while the engine diagnostics go to
/// stderr, so both pipes are drained by reader threads for the whole test. Draining also matters
/// after startup: a full pipe would block the server mid-request.
struct SyntheticServer {
    child: Child,
    base: String,
    log: Arc<Mutex<Vec<String>>>,
    /// The script file a debug-build scripted server was seeded with; removed on drop.
    script: Option<PathBuf>,
}

impl SyntheticServer {
    fn start() -> Self {
        Self::start_with(&[])
    }

    /// Starts the server with extra `serve` flags, for tests that need a non-default mode.
    fn start_with(extra: &[&str]) -> Self {
        let tokenizer = sparkflash_api::encoding::reference_tokenizer_path();
        let mut args = vec!["--synthetic", "--bind", "127.0.0.1:0"];
        args.extend(extra);
        args.push("--tokenizer");
        let mut command = Command::new(env!("CARGO_BIN_EXE_sparkflash"));
        command.arg("serve").args(args).arg(tokenizer);
        Self::launch(command, None)
    }

    /// Starts the debug-build scripted engine (see `SPARKFLASH_SERVE_SCRIPT` in serve.rs).
    ///
    /// The real binary still owns the socket and the router; only the checkpoint is replaced by
    /// a mock engine whose content deltas are the given pieces, so a streaming client can
    /// exercise the SSE/DSML path end to end on a host with no model.
    fn start_scripted(pieces: &[&str], delay_ms: u64, keep_alive_ms: Option<u64>) -> Self {
        Self::start_scripted_with_extra(&[], pieces, delay_ms, keep_alive_ms)
    }

    /// Starts the debug-build scripted engine with extra `serve` flags (for example a device
    /// token window installed beside the scripted host twin).
    fn start_scripted_with_extra(
        extra: &[&str],
        pieces: &[&str],
        delay_ms: u64,
        keep_alive_ms: Option<u64>,
    ) -> Self {
        Self::start_scripted_full(extra, &[], pieces, delay_ms, keep_alive_ms)
    }

    /// Starts the debug-build scripted engine with extra `serve` flags and environment overrides.
    ///
    /// The environment pair carries the drain knobs (`SPARKFLASH_SERVE_DRAIN_*`) so a test can
    /// SIGTERM a busy server and observe the drain window without waiting the production grace.
    fn start_scripted_with_env(
        extra: &[&str],
        env: &[(&str, &str)],
        pieces: &[&str],
        delay_ms: u64,
        keep_alive_ms: Option<u64>,
    ) -> Self {
        Self::start_scripted_full(extra, env, pieces, delay_ms, keep_alive_ms)
    }

    fn start_scripted_full(
        extra: &[&str],
        env: &[(&str, &str)],
        pieces: &[&str],
        delay_ms: u64,
        keep_alive_ms: Option<u64>,
    ) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "sparkflash-serve-script-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("script directory");
        let script = dir.join("script.json");
        let spec = serde_json::json!({
            "pieces": pieces,
            "finish_reason": "stop",
            "delay_ms": delay_ms,
        });
        std::fs::write(&script, serde_json::to_vec(&spec).expect("script JSON"))
            .expect("script file");
        let tokenizer = sparkflash_api::encoding::reference_tokenizer_path();
        let mut args = vec!["--synthetic", "--bind", "127.0.0.1:0"];
        args.extend(extra.iter().copied());
        args.push("--tokenizer");
        let mut command = Command::new(env!("CARGO_BIN_EXE_sparkflash"));
        command
            .arg("serve")
            .args(args)
            .arg(tokenizer)
            .env("SPARKFLASH_SERVE_SCRIPT", &script)
            .envs(env.iter().copied());
        if let Some(milliseconds) = keep_alive_ms {
            command.env("SPARKFLASH_SERVE_KEEP_ALIVE_MS", milliseconds.to_string());
        }
        Self::launch(command, Some(script))
    }
    /// Spawns the child, drains both pipes, and waits for the bound-address line.
    fn launch(mut command: Command, script: Option<PathBuf>) -> Self {
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the serve binary starts");
        let log = Arc::new(Mutex::new(Vec::new()));
        let (line_tx, line_rx) = mpsc::channel();
        if let Some(pipe) = child.stdout.take() {
            spawn_drain(pipe, Arc::clone(&log), line_tx.clone());
        }
        if let Some(pipe) = child.stderr.take() {
            spawn_drain(pipe, Arc::clone(&log), line_tx.clone());
        }
        drop(line_tx);
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        let mut base = None;
        while Instant::now() < deadline {
            match line_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(line) => {
                    if let Some(rest) = line.split("listening on ").nth(1) {
                        base = Some(rest.trim().to_owned());
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if child.try_wait().ok().flatten().is_some() {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        let Some(base) = base else {
            let log = log.lock().expect("log lock").join("\n");
            let _ = child.kill();
            panic!("the synthetic server did not report its address; log:\n{log}");
        };
        Self {
            child,
            base,
            log,
            script,
        }
    }

    /// One HTTP/1.1 exchange, with chunked transfer decoding for the SSE responses.
    fn request(&self, method: &str, path: &str, body: Option<&str>) -> HttpResponse {
        http_exchange(&self.base, method, path, body)
    }

    /// One HTTP/1.1 exchange with extra request headers (the bearer-auth probes).
    fn request_with_headers(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
        headers: &[(&str, &str)],
    ) -> HttpResponse {
        http_exchange_with_headers(&self.base, method, path, body, headers)
    }

    fn log(&self) -> String {
        self.log.lock().expect("log lock").join("\n")
    }

    /// Sends SIGTERM, the drain signal an orchestrator sends.
    fn send_sigterm(&self) {
        let status = Command::new("kill")
            .arg("-TERM")
            .arg(self.child.id().to_string())
            .status()
            .expect("kill runs");
        assert!(status.success(), "SIGTERM delivery to the serve child");
    }

    /// Waits for the child to exit, returning its status or None once the budget is spent.
    fn wait_for_exit(&mut self, timeout: Duration) -> Option<std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(status) = self.child.try_wait().expect("try_wait") {
                return Some(status);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        None
    }
}

/// One HTTP/1.1 exchange, with chunked transfer decoding for the SSE responses.
fn http_exchange(base: &str, method: &str, path: &str, body: Option<&str>) -> HttpResponse {
    http_exchange_with_headers(base, method, path, body, &[])
}

/// The same exchange with extra request headers, used by the bearer-auth probes.
fn http_exchange_with_headers(
    base: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    headers: &[(&str, &str)],
) -> HttpResponse {
    let address = base
        .strip_prefix("http://")
        .expect("the server prints an http URL");
    let mut stream = TcpStream::connect(address).expect("connect to the test server");
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .expect("read timeout");
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .expect("write timeout");
    let body = body.unwrap_or("");
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (name, value) in headers {
        request.push_str(name);
        request.push_str(": ");
        request.push_str(value);
        request.push_str("\r\n");
    }
    request.push_str("\r\n");
    request.push_str(body);
    stream
        .write_all(request.as_bytes())
        .expect("write the request");
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read the response");
    HttpResponse::parse(&raw)
}

/// Reads one child pipe line by line into the shared log and the startup channel.
///
/// Generic over the pipe type because stdout and stderr are different `ChildStdout` and
/// `ChildStderr` types.
fn spawn_drain(
    pipe: impl Read + Send + 'static,
    log: Arc<Mutex<Vec<String>>>,
    line_tx: mpsc::Sender<String>,
) {
    std::thread::spawn(move || {
        let reader = BufReader::new(pipe);
        for line in reader.lines().map_while(Result::ok) {
            log.lock().expect("log lock").push(line.clone());
            let _ = line_tx.send(line);
        }
    });
}

impl Drop for SyntheticServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(script) = self.script.take()
            && let Some(dir) = script.parent()
        {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpResponse {
    fn parse(raw: &[u8]) -> Self {
        let split = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap_or_else(|| {
                panic!("no header/body split in {:?}", String::from_utf8_lossy(raw))
            });
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let body = &raw[split + 4..];
        let mut lines = head.lines();
        let status_line = lines.next().unwrap_or_default();
        let status = status_line
            .split_whitespace()
            .nth(1)
            .expect("status code")
            .parse()
            .expect("numeric status");
        let mut headers = Vec::new();
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
            }
        }
        let chunked = headers
            .iter()
            .any(|(name, value)| name == "transfer-encoding" && value.contains("chunked"));
        let body = if chunked {
            decode_chunked(body)
        } else {
            body.to_vec()
        };
        Self {
            status,
            headers,
            body,
        }
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|error| {
            panic!(
                "body is not JSON ({error}): {:?}",
                String::from_utf8_lossy(&self.body)
            )
        })
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value.as_str())
    }
}

fn decode_chunked(mut body: &[u8]) -> Vec<u8> {
    let mut decoded = Vec::new();
    loop {
        let Some(end) = body.windows(2).position(|window| window == b"\r\n") else {
            return decoded;
        };
        let size_text = String::from_utf8_lossy(&body[..end]).into_owned();
        let size = usize::from_str_radix(size_text.trim(), 16)
            .unwrap_or_else(|error| panic!("chunk size {size_text:?}: {error}"));
        body = &body[end + 2..];
        if size == 0 {
            return decoded;
        }
        assert!(body.len() >= size, "truncated chunk");
        decoded.extend_from_slice(&body[..size]);
        body = &body[size..];
        if body.starts_with(b"\r\n") {
            body = &body[2..];
        }
    }
}

fn chat_body(extra: Value) -> String {
    let mut body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
    });
    if let Value::Object(extra) = extra {
        for (key, value) in extra {
            body[key] = value;
        }
    }
    body.to_string()
}

#[test]
fn health_returns_200_with_an_empty_body() {
    let server = SyntheticServer::start();
    let response = server.request("GET", "/health", None);
    assert_eq!(response.status, 200, "log: {}", server.log());
    assert!(response.body.is_empty(), "health has no body");
}

#[test]
fn a_token_id_completion_round_trips_over_http() {
    let server = SyntheticServer::start();
    let body = serde_json::json!({
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 4,
        "temperature": 0.0,
    })
    .to_string();
    let response = server.request("POST", "/v1/completions", Some(&body));
    assert_eq!(response.status, 200, "log: {}", server.log());
    assert!(
        response
            .header("content-type")
            .is_some_and(|value| value.starts_with("application/json")),
        "the buffered response is JSON"
    );
    let json = response.json();
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["finish_reason"], "length");
    assert_eq!(json["usage"]["prompt_tokens"], 8);
    assert_eq!(json["usage"]["completion_tokens"], 4);
    assert_eq!(json["usage"]["total_tokens"], 12);
    // vLLM's OpenAI default hides special tokens, and the tiny plan's generated token is the BOS
    // special token, so the default answer is the empty string.
    assert_eq!(json["choices"][0]["text"], "", "{json}");
}

#[test]
fn a_stop_string_ends_generation_over_http() {
    let server = SyntheticServer::start();
    let prompt = serde_json::json!({
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 12,
        "temperature": 0.0,
        // The tiny plan only generates the BOS special token, so the stop-string text needs the
        // explicit false passthrough (vLLM's skip_special_tokens: false).
        "skip_special_tokens": false,
    })
    .to_string();
    let full = server.request("POST", "/v1/completions", Some(&prompt));
    assert_eq!(full.status, 200, "log: {}", server.log());
    let full_json = full.json();
    let text = full_json["choices"][0]["text"]
        .as_str()
        .expect("the tiny plan generated text")
        .to_owned();
    assert!(!text.is_empty());
    let stop: String = text.chars().take(3).collect();

    let mut stopped_body: Value = serde_json::from_str(&prompt).expect("prompt is JSON");
    stopped_body["stop"] = Value::String(stop.clone());
    let stopped = server.request("POST", "/v1/completions", Some(&stopped_body.to_string()));
    assert_eq!(stopped.status, 200, "log: {}", server.log());
    let stopped_json = stopped.json();
    let stopped_text = stopped_json["choices"][0]["text"]
        .as_str()
        .expect("text is a string");
    assert!(
        !stopped_text.contains(&stop),
        "the stop string never reaches the client: {stopped_text:?}"
    );
    assert_eq!(
        stopped_json["choices"][0]["finish_reason"], "stop",
        "a matched stop string reports finish_reason stop: {stopped_json}"
    );
    assert_eq!(
        stopped_json["usage"]["completion_tokens"], 1,
        "the token that completed the stop string is counted: {stopped_json}"
    );
}

#[test]
fn an_empty_stop_string_is_a_400_with_vllms_message_over_http() {
    let server = SyntheticServer::start();
    let body = chat_body(serde_json::json!({"stop": [""]}));
    let response = server.request("POST", "/v1/chat/completions", Some(&body));
    assert_eq!(response.status, 400, "log: {}", server.log());
    let json = response.json();
    assert_eq!(
        json["error"]["message"],
        "stop cannot contain an empty string."
    );
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["param"], "stop");
}

#[test]
fn streaming_completions_carry_logprobs_usage_chunk_and_done() {
    let server = SyntheticServer::start();
    let body = serde_json::json!({
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 3,
        "temperature": 0.0,
        "stream": true,
        "stream_options": {"include_usage": true},
        // The streamed text frames carry the BOS special token, so ask for it explicitly.
        "skip_special_tokens": false,
    })
    .to_string();
    let response = server.request("POST", "/v1/completions", Some(&body));
    assert_eq!(response.status, 200, "log: {}", server.log());
    assert_eq!(response.header("content-type"), Some("text/event-stream"));
    let text = response.text();
    assert!(
        text.ends_with("data: [DONE]\n\n"),
        "the stream ends with the SSE terminator: {text:?}"
    );
    let frames: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("each frame is JSON"))
        .collect();
    let content_frames: Vec<&Value> = frames
        .iter()
        .filter(|frame| {
            frame["choices"][0]["text"].is_string()
                && !frame["choices"][0]["text"]
                    .as_str()
                    .unwrap_or("")
                    .is_empty()
        })
        .collect();
    assert_eq!(
        content_frames.len(),
        3,
        "one text frame per generated token"
    );
    let usage = frames
        .iter()
        .find(|frame| frame["usage"].is_object())
        .expect("an include_usage stream carries a final usage chunk");
    assert_eq!(usage["choices"], serde_json::json!([]));
    assert_eq!(usage["usage"]["prompt_tokens"], 8);
    assert_eq!(usage["usage"]["completion_tokens"], 3);
}

#[test]
fn models_card_publishes_the_vllm_members_and_the_health_route() {
    let server = SyntheticServer::start();
    let response = server.request("GET", "/v1/models", None);
    assert_eq!(response.status, 200, "log: {}", server.log());
    let json = response.json();
    let card = &json["data"][0];
    assert_eq!(card["object"], "model");
    assert_eq!(card["owned_by"], "sparkflash");
    assert_eq!(card["root"], card["id"]);
    assert!(card["parent"].is_null());
    assert_eq!(card["max_model_len"], card["context_length"]);
    assert!(card["permission"].is_array());
    let endpoints = card["capabilities"]["endpoints"]
        .as_array()
        .expect("endpoints is an array");
    assert!(
        endpoints.iter().any(|endpoint| endpoint == "/health"),
        "the advertised endpoint list names /health: {endpoints:?}"
    );
}

/// The OpenAI sampling parameters cross the wire, are validated with vLLM's messages, and the
/// disabled defaults are the pre-options request: one token-for-token identity claim and one
/// candidate-filter cross-check, all on the synthetic host path.
#[test]
fn sampler_options_are_validated_and_defaulted_over_http() {
    let server = SyntheticServer::start();
    let base = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 4,
        "temperature": 1.0,
        "seed": 7,
        // The tiny plan only generates the BOS special token, so the text identity below needs the
        // explicit false passthrough; the default-hidden behaviour has its own e2e case.
        "skip_special_tokens": false,
    });
    let plain_response = server.request("POST", "/v1/completions", Some(&base.to_string()));
    assert_eq!(plain_response.status, 200, "log: {}", server.log());
    let plain = plain_response.json();

    // The explicit disabled configuration is the same request: default path, same tokens.
    let mut explicit = base.clone();
    explicit["top_p"] = serde_json::json!(1.0);
    explicit["top_k"] = serde_json::json!(0);
    explicit["repetition_penalty"] = serde_json::json!(1.0);
    explicit["presence_penalty"] = serde_json::json!(0.0);
    explicit["frequency_penalty"] = serde_json::json!(0.0);
    let defaults_response = server.request("POST", "/v1/completions", Some(&explicit.to_string()));
    assert_eq!(defaults_response.status, 200, "log: {}", server.log());
    let defaults = defaults_response.json();
    assert_eq!(
        defaults["choices"][0]["text"], plain["choices"][0]["text"],
        "the disabled defaults must produce the pre-options tokens"
    );
    assert_eq!(
        defaults["choices"][0]["finish_reason"],
        plain["choices"][0]["finish_reason"]
    );
    assert_eq!(defaults["usage"], plain["usage"]);

    // top_p 0.0 and top_k 1 both reduce the candidate set to the single most likely row at every
    // step, so the two requests must agree token for token.
    let mut top_p = base.clone();
    top_p["top_p"] = serde_json::json!(0.0);
    let top_p_response = server.request("POST", "/v1/completions", Some(&top_p.to_string()));
    assert_eq!(top_p_response.status, 200, "log: {}", server.log());
    let top_p_json = top_p_response.json();
    let mut top_k = base.clone();
    top_k["top_k"] = serde_json::json!(1);
    let top_k_response = server.request("POST", "/v1/completions", Some(&top_k.to_string()));
    assert_eq!(top_k_response.status, 200, "log: {}", server.log());
    let top_k_json = top_k_response.json();
    assert_eq!(
        top_p_json["choices"][0]["text"], top_k_json["choices"][0]["text"],
        "top_p 0.0 and top_k 1 filter to the same single candidate"
    );
    assert_eq!(
        top_p_json["choices"][0]["finish_reason"],
        top_k_json["choices"][0]["finish_reason"]
    );

    // vLLM's refusals reach the wire before generation, so the synthetic host limitation (a real
    // tokenizer prompt needs more than the tiny plan's 128 embedding rows) cannot mask them.
    let cases = [
        (
            serde_json::json!({"top_k": -2}),
            "top_k",
            "top_k must be 0 (disable), or at least 1, got -2.",
        ),
        (
            serde_json::json!({"presence_penalty": 2.5}),
            "presence_penalty",
            "presence_penalty must be in [-2, 2], got 2.5.",
        ),
        (
            serde_json::json!({"repetition_penalty": 0.0}),
            "repetition_penalty",
            "repetition_penalty must be greater than zero, got 0.0.",
        ),
    ];
    for (extra, parameter, message) in cases {
        let response = server.request("POST", "/v1/chat/completions", Some(&chat_body(extra)));
        assert_eq!(response.status, 400, "case {parameter}: {}", server.log());
        let json = response.json();
        assert_eq!(json["error"]["message"], message, "case {parameter}");
        assert_eq!(json["error"]["param"], parameter, "case {parameter}");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }
}

/// vLLM's OpenAI default hides special tokens; an explicit false is the only way to see them.
#[test]
fn skip_special_tokens_defaults_to_true_and_explicit_false_passes_through() {
    let server = SyntheticServer::start();
    let hidden = serde_json::json!({
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 2,
        "temperature": 0.0,
    })
    .to_string();
    let hidden_response = server.request("POST", "/v1/completions", Some(&hidden));
    assert_eq!(hidden_response.status, 200, "log: {}", server.log());
    assert_eq!(
        hidden_response.json()["choices"][0]["text"],
        "",
        "the default hides the generated BOS special token"
    );

    let mut shown: Value = serde_json::from_str(&hidden).expect("request JSON");
    shown["skip_special_tokens"] = Value::Bool(false);
    let shown_response = server.request("POST", "/v1/completions", Some(&shown.to_string()));
    assert_eq!(shown_response.status, 200, "log: {}", server.log());
    let text = shown_response.json()["choices"][0]["text"]
        .as_str()
        .expect("text is a string")
        .to_owned();
    assert!(
        !text.is_empty() && text.contains('｜'),
        "an explicit false passes the special token through: {text:?}"
    );
}

/// The opt-in S4 host scheduler keeps the same HTTP contract under concurrent requests.
///
/// This is host serving behaviour on the reference CPU engine, not a throughput measurement: every
/// concurrent request is answered, and each answer is the one the serial path produces for the same
/// token-id prompt.
#[test]
fn continuous_batching_answers_concurrent_requests_with_the_serial_paths_output() {
    let serial = SyntheticServer::start();
    let body = serde_json::json!({
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 6,
        "temperature": 0.0,
        "skip_special_tokens": false,
    })
    .to_string();
    let expected = serial.request("POST", "/v1/completions", Some(&body));
    assert_eq!(expected.status, 200, "log: {}", serial.log());
    let expected = expected.json();
    let expected_text = expected["choices"][0]["text"]
        .as_str()
        .expect("the serial answer has text")
        .to_owned();
    let expected_completion = expected["usage"]["completion_tokens"]
        .as_u64()
        .expect("the serial answer counts its tokens");

    let batched = SyntheticServer::start_with(&["--continuous-batching", "--max-batch", "4"]);
    let base = batched.base.clone();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let base = base.clone();
            let body = body.clone();
            std::thread::spawn(move || http_exchange(&base, "POST", "/v1/completions", Some(&body)))
        })
        .collect();
    for handle in handles {
        let response = handle.join().expect("the client thread finishes");
        assert_eq!(response.status, 200, "log: {}", batched.log());
        let json = response.json();
        assert_eq!(
            json["choices"][0]["text"], expected_text,
            "a batched answer must be the serial answer: {json}"
        );
        assert_eq!(
            json["usage"]["completion_tokens"], expected_completion,
            "the same token budget is spent: {json}"
        );
        assert_eq!(
            json["choices"][0]["finish_reason"], "length",
            "six tokens exhausts the budget: {json}"
        );
    }
}

/// Deadline for a live-gauge observation in the metrics tests.
const GAUGE_POLL_TIMEOUT: Duration = Duration::from_secs(15);

/// Polls `/metrics` until every wanted sample reads its value, returning the satisfying text.
///
/// The gauges track live requests. The synthetic engine can finish a short request, or accept a
/// queued one, inside a fixed sleep window on a fast host, so the tests poll to a deadline instead
/// of sampling once; the last text is returned on timeout and the caller's assertion prints it.
fn wait_for_metrics(server: &SyntheticServer, wanted: &[(&str, f64)], timeout: Duration) -> String {
    let deadline = Instant::now() + timeout;
    loop {
        let metrics = metrics_text(server);
        if wanted
            .iter()
            .all(|(sample, want)| metric_value(&metrics, sample) == *want)
        {
            return metrics;
        }
        if Instant::now() >= deadline {
            return metrics;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Reads the Prometheus text exposition from the running server.
fn metrics_text(server: &SyntheticServer) -> String {
    let response = server.request("GET", "/metrics", None);
    assert_eq!(response.status, 200, "log: {}", server.log());
    response.text()
}

/// One sample value from the Prometheus text exposition, by full sample name.
fn metric_value(metrics: &str, sample: &str) -> f64 {
    for line in metrics.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some(sample) {
            let raw = parts
                .next()
                .unwrap_or_else(|| panic!("no value on: {line}"));
            return raw
                .parse()
                .unwrap_or_else(|error| panic!("sample {sample} value {raw}: {error}"));
        }
    }
    panic!("sample {sample} is missing from /metrics:\n{metrics}");
}

/// POSTs a completion body and returns the raw HTTP response.
fn completion(base: &str, body: Value) -> HttpResponse {
    http_exchange(base, "POST", "/v1/completions", Some(&body.to_string()))
}

/// The JSON data frames of an SSE response, in order; the [DONE] terminator is checked separately.
fn sse_frames(response: &HttpResponse) -> Vec<Value> {
    response
        .text()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| {
            serde_json::from_str(payload)
                .unwrap_or_else(|error| panic!("SSE frame {payload:?}: {error}"))
        })
        .collect()
}

/// Drops the fields two separate servers legitimately differ in: request ids, wall-clock
/// creation times and measured timings. Everything else is deterministic reference arithmetic.
fn strip_volatile(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(key, _)| !matches!(key.as_str(), "id" | "created" | "timing"))
                .map(|(key, value)| (key.clone(), strip_volatile(value)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(strip_volatile).collect()),
        other => other.clone(),
    }
}

/// The key-and-type skeleton of a payload: an equal shape means a client parses both the same way.
fn payload_shape(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), payload_shape(value)))
                .collect(),
        ),
        Value::Array(items) => serde_json::json!({
            "array_len": items.len(),
            "item": items.first().map(payload_shape).unwrap_or(Value::String("none".to_owned())),
        }),
        Value::Null => serde_json::json!("null"),
        Value::Bool(_) => Value::String("bool".to_owned()),
        Value::Number(_) => Value::String("number".to_owned()),
        Value::String(_) => Value::String("string".to_owned()),
    }
}

/// Asserts the two buffered responses are the same payload: same shape, same stable values.
fn assert_same_buffered(serial: &HttpResponse, batched: &HttpResponse, context: &str) {
    assert_eq!(serial.status, 200, "{context}: serial status");
    assert_eq!(batched.status, 200, "{context}: batched status");
    let serial_json = serial.json();
    let batched_json = batched.json();
    assert_eq!(
        payload_shape(&serial_json),
        payload_shape(&batched_json),
        "{context}: payload shape"
    );
    assert_eq!(
        strip_volatile(&serial_json),
        strip_volatile(&batched_json),
        "{context}: stable payload values"
    );
}

/// Asserts the two SSE responses carry the same frames in the same order, and terminate in [DONE].
fn assert_same_stream(serial: &HttpResponse, batched: &HttpResponse, context: &str) -> Vec<Value> {
    assert_eq!(serial.status, 200, "{context}: serial status");
    assert_eq!(batched.status, 200, "{context}: batched status");
    for response in [serial, batched] {
        assert!(
            response.text().contains("data: [DONE]"),
            "{context}: stream terminates in [DONE]"
        );
    }
    let serial_frames = sse_frames(serial);
    let batched_frames = sse_frames(batched);
    assert_eq!(
        serial_frames.len(),
        batched_frames.len(),
        "{context}: frame count"
    );
    for (index, (serial_frame, batched_frame)) in
        serial_frames.iter().zip(batched_frames.iter()).enumerate()
    {
        assert_eq!(
            payload_shape(serial_frame),
            payload_shape(batched_frame),
            "{context}: frame {index} shape"
        );
        assert_eq!(
            strip_volatile(serial_frame),
            strip_volatile(batched_frame),
            "{context}: frame {index} stable values"
        );
    }
    serial_frames
}

/// The ops families describe the requests the driver really ran, and are live while one does.
#[test]
fn metrics_report_the_ops_families_the_driver_feeds() {
    let serial = SyntheticServer::start();
    let prompt = serde_json::json!([1, 2, 3, 4, 5, 6, 7, 8]);
    for max_tokens in [4, 3] {
        let body = serde_json::json!({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        });
        let response = completion(&serial.base, body);
        assert_eq!(response.status, 200, "log: {}", serial.log());
        assert_eq!(response.json()["usage"]["completion_tokens"], max_tokens);
    }
    // Two requests generated 4 + 3 tokens and each produced exactly one TTFT sample. The worker
    // leaves the running gauge just after it sends the final reply, so the gauge is waited for.
    let metrics = wait_for_metrics(
        &serial,
        &[("sparkflash_num_requests_running", 0.0)],
        GAUGE_POLL_TIMEOUT,
    );
    assert_eq!(
        metric_value(&metrics, "sparkflash_requests_total"),
        2.0,
        "{metrics}"
    );
    assert_eq!(
        metric_value(&metrics, "sparkflash_generation_tokens_total"),
        7.0,
        "{metrics}"
    );
    assert_eq!(
        metric_value(&metrics, "sparkflash_time_to_first_token_seconds_count"),
        2.0,
        "{metrics}"
    );
    assert!(
        metric_value(&metrics, "sparkflash_time_to_first_token_seconds_sum") > 0.0,
        "the TTFT sum is a measured duration: {metrics}"
    );
    assert_eq!(
        metric_value(&metrics, "sparkflash_num_requests_running"),
        0.0,
        "{metrics}"
    );
    assert_eq!(
        metric_value(&metrics, "sparkflash_num_requests_waiting"),
        0.0,
        "{metrics}"
    );
    // vLLM's histogram shape: named buckets including the smallest vLLM bound and the +Inf bucket.
    assert!(
        metrics.contains("sparkflash_time_to_first_token_seconds_bucket{le=\"0.001\"}"),
        "{metrics}"
    );
    assert_eq!(
        metric_value(
            &metrics,
            "sparkflash_time_to_first_token_seconds_bucket{le=\"+Inf\"}"
        ),
        2.0,
        "{metrics}"
    );
    assert!(
        metrics.contains("# HELP sparkflash_time_to_first_token_seconds"),
        "{metrics}"
    );

    // A long request is executing while a second one waits in the engine queue.
    let base = serial.base.clone();
    let long = std::thread::spawn(move || {
        let body = serde_json::json!({
            "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
            "max_tokens": 50,
            "temperature": 0.0,
        });
        completion(&base, body)
    });
    let during = wait_for_metrics(
        &serial,
        &[("sparkflash_num_requests_running", 1.0)],
        GAUGE_POLL_TIMEOUT,
    );
    assert_eq!(
        metric_value(&during, "sparkflash_num_requests_running"),
        1.0,
        "the executed request is in the running gauge: {during}"
    );
    assert_eq!(
        metric_value(&during, "sparkflash_num_requests_waiting"),
        0.0,
        "nothing waits behind it yet: {during}"
    );
    let base = serial.base.clone();
    let queued = std::thread::spawn(move || {
        let body = serde_json::json!({
            "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
            "max_tokens": 8,
            "temperature": 0.0,
        });
        completion(&base, body)
    });
    let queued_metrics = wait_for_metrics(
        &serial,
        &[
            ("sparkflash_num_requests_running", 1.0),
            ("sparkflash_num_requests_waiting", 1.0),
        ],
        GAUGE_POLL_TIMEOUT,
    );
    assert_eq!(
        metric_value(&queued_metrics, "sparkflash_num_requests_running"),
        1.0,
        "the first request still executes: {queued_metrics}"
    );
    assert_eq!(
        metric_value(&queued_metrics, "sparkflash_num_requests_waiting"),
        1.0,
        "the second request is queued, not running: {queued_metrics}"
    );
    assert_eq!(long.join().expect("the long client finishes").status, 200);
    assert_eq!(
        queued.join().expect("the queued client finishes").status,
        200
    );
    let after = wait_for_metrics(
        &serial,
        &[("sparkflash_num_requests_running", 0.0)],
        GAUGE_POLL_TIMEOUT,
    );
    assert_eq!(
        metric_value(&after, "sparkflash_num_requests_running"),
        0.0,
        "every request left the running gauge: {after}"
    );
    assert_eq!(
        metric_value(&after, "sparkflash_num_requests_waiting"),
        0.0,
        "{after}"
    );
    assert_eq!(
        metric_value(&after, "sparkflash_requests_total"),
        4.0,
        "{after}"
    );
    assert_eq!(
        metric_value(&after, "sparkflash_generation_tokens_total"),
        65.0,
        "4 + 3 + 50 + 8 generated token ids: {after}"
    );
    assert_eq!(
        metric_value(&after, "sparkflash_time_to_first_token_seconds_count"),
        4.0,
        "one TTFT sample per token-producing request: {after}"
    );
}

/// The continuous-batching driver feeds the same families, including the waiting gauge.
#[test]
fn batched_metrics_report_two_running_requests_and_one_waiting_slot() {
    let batched = SyntheticServer::start_with(&["--continuous-batching", "--max-batch", "2"]);
    let base = batched.base.clone();
    let handles: Vec<_> = (0..3)
        .map(|_| {
            let base = base.clone();
            std::thread::spawn(move || {
                let body = serde_json::json!({
                    "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
                    "max_tokens": 40,
                    "temperature": 0.0,
                });
                completion(&base, body)
            })
        })
        .collect();
    let during = wait_for_metrics(
        &batched,
        &[
            ("sparkflash_num_requests_running", 2.0),
            ("sparkflash_num_requests_waiting", 1.0),
        ],
        GAUGE_POLL_TIMEOUT,
    );
    assert_eq!(
        metric_value(&during, "sparkflash_num_requests_running"),
        2.0,
        "max-batch 2 executes two requests: {during}"
    );
    assert_eq!(
        metric_value(&during, "sparkflash_num_requests_waiting"),
        1.0,
        "the third accepted request has no slot yet: {during}"
    );
    for handle in handles {
        assert_eq!(
            handle.join().expect("the client finishes").status,
            200,
            "log: {}",
            batched.log()
        );
    }
    let after = wait_for_metrics(
        &batched,
        &[("sparkflash_num_requests_running", 0.0)],
        GAUGE_POLL_TIMEOUT,
    );
    assert_eq!(
        metric_value(&after, "sparkflash_num_requests_running"),
        0.0,
        "{after}"
    );
    assert_eq!(
        metric_value(&after, "sparkflash_num_requests_waiting"),
        0.0,
        "{after}"
    );
    assert_eq!(
        metric_value(&after, "sparkflash_requests_total"),
        3.0,
        "{after}"
    );
    assert_eq!(
        metric_value(&after, "sparkflash_generation_tokens_total"),
        120.0,
        "3 x 40 generated token ids: {after}"
    );
    assert_eq!(
        metric_value(&after, "sparkflash_time_to_first_token_seconds_count"),
        3.0,
        "{after}"
    );
}

/// Buffered and SSE probes of the prompt_logprobs and stop features agree byte-for-byte in shape
/// and stable values between the serial engine loop and the continuous-batching scheduler.
#[test]
fn batched_wire_probes_match_the_serial_payload_for_logprobs_and_stop() {
    let serial = SyntheticServer::start();
    let batched = SyntheticServer::start_with(&["--continuous-batching", "--max-batch", "4"]);
    let serial_base = serial.base.clone();
    let batched_base = batched.base.clone();
    let prompt = serde_json::json!([1, 2, 3, 4, 5, 6, 7, 8]);

    // The only text the synthetic tiny plan generates is the BOS token repeated, so take the
    // stop string from the first three characters of a real response.
    let probe = completion(
        &serial_base,
        serde_json::json!({
            "prompt": prompt,
            "max_tokens": 12,
            "temperature": 0.0,
            "skip_special_tokens": false,
        }),
    );
    assert_eq!(probe.status, 200, "log: {}", serial.log());
    let generated = probe.json()["choices"][0]["text"]
        .as_str()
        .expect("the probe text is a string")
        .to_owned();
    assert!(generated.chars().count() > 3, "probe text: {generated:?}");
    let stop: String = generated.chars().take(3).collect();

    // 1. Buffered prompt_logprobs, both the dropped-zero and the typed top-1 payload.
    for value in [0, 1] {
        let body = serde_json::json!({
            "prompt": prompt,
            "max_tokens": 3,
            "temperature": 0.0,
            "prompt_logprobs": value,
            "skip_special_tokens": false,
        });
        let context = format!("buffered prompt_logprobs={value}");
        assert_same_buffered(
            &completion(&serial_base, body.clone()),
            &completion(&batched_base, body),
            &context,
        );
    }

    // 2. Buffered stop string: the stop cuts the text and the finish reason is stop.
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 12,
        "temperature": 0.0,
        "skip_special_tokens": false,
        "stop": stop,
    });
    let serial_stop = completion(&serial_base, body.clone());
    let batched_stop = completion(&batched_base, body);
    assert_same_buffered(&serial_stop, &batched_stop, "buffered stop");
    assert_eq!(
        serial_stop.json()["choices"][0]["finish_reason"],
        "stop",
        "payload: {}",
        serial_stop.json()
    );
    let stopped_text = serial_stop.json()["choices"][0]["text"]
        .as_str()
        .expect("stop text is a string")
        .to_owned();
    assert!(
        !stopped_text.contains(&stop),
        "the stop string is not part of the emitted text: {stopped_text:?}"
    );

    // 3. SSE stop string: the frames and [DONE] are identical.
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 12,
        "temperature": 0.0,
        "skip_special_tokens": false,
        "stop": stop,
        "stream": true,
    });
    let serial_frames = assert_same_stream(
        &completion(&serial_base, body.clone()),
        &completion(&batched_base, body),
        "SSE stop",
    );
    let last = serial_frames.last().expect("the stop stream has frames");
    assert_eq!(
        last["choices"][0]["finish_reason"], "stop",
        "last frame: {last}"
    );

    // 4. SSE sample logprobs: vLLM's logprobs chunk shape is identical on both paths.
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 3,
        "temperature": 0.0,
        "skip_special_tokens": false,
        "logprobs": 2,
        "stream": true,
    });
    let frames = assert_same_stream(
        &completion(&serial_base, body.clone()),
        &completion(&batched_base, body),
        "SSE logprobs",
    );
    assert!(
        frames
            .iter()
            .any(|frame| frame["choices"][0]["logprobs"].is_object()),
        "the logprobs stream carries logprob objects: {frames:?}"
    );

    // 5. An explicit prompt_logprobs on a stream is refused with the same typed error in both
    // modes. This is vLLM's refusal; the tiny plan would otherwise silently drop it.
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 3,
        "temperature": 0.0,
        "prompt_logprobs": 1,
        "stream": true,
    });
    let serial_refusal = completion(&serial_base, body.clone());
    let batched_refusal = completion(&batched_base, body);
    assert_eq!(serial_refusal.status, 400, "log: {}", serial.log());
    assert_eq!(batched_refusal.status, 400, "log: {}", batched.log());
    assert_eq!(
        strip_volatile(&serial_refusal.json()),
        strip_volatile(&batched_refusal.json()),
        "the refusal payload is identical"
    );
    assert_eq!(serial_refusal.json()["error"]["param"], "prompt_logprobs");

    // 6. prompt_logprobs=0 on a stream is accepted and dropped, which is vLLM's rule; both modes
    // agree and no chunk invents a prompt_logprobs field.
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 3,
        "temperature": 0.0,
        "prompt_logprobs": 0,
        "stream": true,
    });
    let frames = assert_same_stream(
        &completion(&serial_base, body.clone()),
        &completion(&batched_base, body),
        "SSE prompt_logprobs=0",
    );
    for frame in &frames {
        assert!(
            !frame.to_string().contains("prompt_logprobs"),
            "a stream chunk has no prompt_logprobs field: {frame}"
        );
    }
}

/// One-pass prompt logprobs over the wire: a buffered request scores every prompt position in
/// order, with position 0 null, exactly as the token-at-a-time path did; repeating it reproduces
/// the same stable payload, and the batched scheduler still returns the same bytes.
///
/// SSE cannot carry prompt logprobs in vLLM's contract: an explicit positive value is the typed
/// 400 refusal and zero is accepted-and-dropped, both covered by the probe test above. The
/// one-pass path therefore shows up on the buffered surface, where the payload is built.
#[test]
fn buffered_prompt_logprobs_score_every_prompt_position_once() {
    let serial = SyntheticServer::start();
    let batched = SyntheticServer::start_with(&["--continuous-batching", "--max-batch", "4"]);
    let prompt_ids = [1u64, 2, 3, 4, 5, 6, 7, 8];
    let body = serde_json::json!({
        "prompt": prompt_ids,
        "max_tokens": 3,
        "temperature": 0.0,
        "prompt_logprobs": 1,
        "skip_special_tokens": false,
    });
    let first = completion(&serial.base, body.clone());
    let repeated = completion(&serial.base, body.clone());
    let batched_response = completion(&batched.base, body);
    for response in [&first, &repeated, &batched_response] {
        assert_eq!(
            response.status,
            200,
            "serial log:\n{}\nbatched log:\n{}",
            serial.log(),
            batched.log()
        );
    }
    let first_json = first.json();
    assert_eq!(
        strip_volatile(&first_json),
        strip_volatile(&repeated.json()),
        "repeating the prompt-logprobs request returns the same stable payload"
    );
    assert_eq!(
        payload_shape(&first_json),
        payload_shape(&batched_response.json()),
        "the serial and batched payloads have the same shape"
    );
    assert_eq!(
        strip_volatile(&first_json),
        strip_volatile(&batched_response.json()),
        "the serial and batched payloads carry the same stable prompt logprobs"
    );
    let positions = first_json["choices"][0]["prompt_logprobs"]
        .as_array()
        .expect("prompt_logprobs is an array");
    assert_eq!(
        positions.len(),
        prompt_ids.len(),
        "one entry per prompt position, including the null first one"
    );
    assert_eq!(
        positions[0],
        Value::Null,
        "position 0 has no conditioning token"
    );
    for (index, position) in positions.iter().enumerate().skip(1) {
        let alternatives = position
            .as_object()
            .unwrap_or_else(|| panic!("position {index} is an id-keyed object: {position}"));
        let own = alternatives
            .get(&prompt_ids[index].to_string())
            .unwrap_or_else(|| {
                panic!(
                    "position {index} scores prompt token {}: {position}",
                    prompt_ids[index]
                )
            });
        assert!(
            own.get("logprob").and_then(Value::as_f64).is_some(),
            "position {index} has a logprob: {own}"
        );
        assert!(
            own.get("rank")
                .and_then(Value::as_u64)
                .is_some_and(|rank| rank >= 1),
            "the prompt token's own entry is ranked: {own}"
        );
        assert!(
            own.get("decoded_token").and_then(Value::as_str).is_some(),
            "the entry carries the decoded token: {own}"
        );
    }
}

// ---------------------------------------------------------------------------------------
// Scripted-engine wire parity: the real serve binary and router, driven over a loopback
// socket with a debug-build mock engine, so the streaming DSML path is checked as a client
// sees it. Host path only: no checkpoint, no device, no GPU.
// ---------------------------------------------------------------------------------------

/// Splits a scripted answer into content pieces at character boundaries.
fn wire_pieces(text: &str, char_size: usize) -> Vec<String> {
    text.chars()
        .collect::<Vec<_>>()
        .chunks(char_size)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect()
}

/// Starts a debug-build scripted server whose content deltas are these pieces.
fn scripted_server(
    pieces: &[String],
    delay_ms: u64,
    keep_alive_ms: Option<u64>,
) -> SyntheticServer {
    let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    SyntheticServer::start_scripted(&refs, delay_ms, keep_alive_ms)
}

/// A chat request that offers one tool, so the DSML parser is active.
fn wire_tools_body(stream: bool) -> String {
    chat_body(serde_json::json!({
        // The scripted answer is longer than the harness default of 4 tokens, and the real
        // handler truncates to max_tokens; the block has to arrive whole.
        "max_tokens": 256,
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Read the weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }}],
        "stream": stream,
    }))
}

fn wire_raw_frames(text: &str) -> Vec<&str> {
    text.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .collect()
}

fn wire_sse_frames(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("each data frame is JSON"))
        .collect()
}

fn wire_streamed_content(frames: &[Value]) -> String {
    let mut content = String::new();
    for frame in frames {
        if let Some(text) = frame["choices"][0]["delta"]["content"].as_str() {
            content.push_str(text);
        }
    }
    content
}

/// Reassembles the streamed OpenAI tool_calls array exactly as a client would.
fn wire_assemble_calls(frames: &[Value]) -> Vec<Value> {
    let mut calls: Vec<Value> = Vec::new();
    for frame in frames {
        let Some(deltas) = frame["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for delta in deltas {
            let index = delta["index"].as_u64().expect("tool_call index") as usize;
            if let Some(id) = delta["id"].as_str() {
                assert_eq!(index, calls.len(), "a call starts before its fragments");
                // The streamed delta carries index as its per-delta bookkeeping key; the buffered
                // array carries the same value per call, so the assembled object keeps it too and
                // the two arrays are directly comparable.
                calls.push(serde_json::json!({
                    "index": index,
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": delta["function"]["name"].as_str().expect("name"),
                        "arguments": delta["function"]["arguments"]
                            .as_str()
                            .expect("the first delta carries an arguments string"),
                    },
                }));
            } else {
                let fragment = delta["function"]["arguments"]
                    .as_str()
                    .expect("fragments are strings");
                let arguments = calls[index]["function"]["arguments"]
                    .as_str()
                    .expect("a call has an arguments string")
                    .to_owned()
                    + fragment;
                calls[index]["function"]["arguments"] = Value::String(arguments);
            }
        }
    }
    calls
}

/// The DSML answer the buffered wire and router tests use.
fn wire_dsml_answer() -> String {
    [
        "I will check the weather.",
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"get_weather\">\n",
        "<｜DSML｜ parameter name=\"city\" string=\"true\">Beijing</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    ]
    .concat()
}

#[test]
fn scripted_wire_streaming_dsml_tool_calls_match_the_buffered_body() {
    let answer = wire_dsml_answer();
    let pieces = wire_pieces(&answer, 5);

    // The buffered answer, from a fresh server so the response ids start at the same place.
    let buffered_server = scripted_server(&pieces, 0, None);
    let buffered_response = buffered_server.request(
        "POST",
        "/v1/chat/completions",
        Some(&wire_tools_body(false)),
    );
    assert_eq!(
        buffered_response.status,
        200,
        "log: {}",
        buffered_server.log()
    );
    let buffered = buffered_response.json();
    assert_eq!(
        buffered["choices"][0]["finish_reason"], "tool_calls",
        "{buffered}"
    );
    let buffered_calls = buffered["choices"][0]["message"]["tool_calls"].clone();
    assert!(
        buffered_calls
            .as_array()
            .is_some_and(|calls| !calls.is_empty()),
        "the buffered answer lifts the block: {buffered}"
    );
    let buffered_content = buffered["choices"][0]["message"]["content"]
        .as_str()
        .expect("buffered content")
        .to_owned();
    drop(buffered_server);

    // The streamed answer, from another fresh server. A 5 ms keep-alive against a 20 ms engine
    // event gap makes the comment/data interleave observable on the real wire.
    let stream_server = scripted_server(&pieces, 20, Some(5));
    let started = Instant::now();
    let response =
        stream_server.request("POST", "/v1/chat/completions", Some(&wire_tools_body(true)));
    let wall_ms = started.elapsed().as_millis();
    assert_eq!(response.status, 200, "log: {}", stream_server.log());
    assert_eq!(response.header("content-type"), Some("text/event-stream"));
    let text = response.text();
    assert!(
        text.ends_with("data: [DONE]\n\n"),
        "the stream ends with the SSE terminator: {text}"
    );
    let raw = wire_raw_frames(&text);
    let comment_positions: Vec<usize> = raw
        .iter()
        .enumerate()
        .filter(|(_, frame)| **frame == ": ping")
        .map(|(index, _)| index)
        .collect();
    assert!(
        !comment_positions.is_empty(),
        "an idle engine gap carries keep-alive comments: {text}"
    );
    assert_ne!(
        comment_positions.first(),
        Some(&0),
        "no comment precedes the first data frame: {text}"
    );
    assert_eq!(
        raw.last().copied(),
        Some("data: [DONE]"),
        "the terminator is the last frame: {text}"
    );
    for (index, frame) in raw.iter().enumerate() {
        if *frame == ": ping" {
            continue;
        }
        assert!(
            frame.starts_with("data: "),
            "frame {index} is neither a comment nor a data frame: {frame}"
        );
    }
    let frames = wire_sse_frames(&text);
    assert_eq!(
        frames[0]["choices"][0]["delta"]["role"], "assistant",
        "the role frame is first"
    );
    let streamed_calls = wire_assemble_calls(&frames);
    let streamed_content = wire_streamed_content(&frames);
    let content_equal = streamed_content == buffered_content;
    let tool_calls_equal = Value::Array(streamed_calls.clone()) == buffered_calls;
    assert!(
        tool_calls_equal,
        "the streamed deltas must reassemble the buffered tool_calls byte for byte; \
         streamed {streamed_calls:?}, buffered {buffered_calls:?}"
    );
    assert!(
        content_equal,
        "the text before the block streams as content: {streamed_content:?}"
    );
    assert!(
        !streamed_content.contains("DSML"),
        "the DSML block is never sent as content: {streamed_content:?}"
    );
    let final_choice = &frames.last().expect("data frames")["choices"][0];
    assert_eq!(
        final_choice["finish_reason"], "tool_calls",
        "the final choice chunk carries the override"
    );
    let mut content_deltas = 0usize;
    let mut call_start_deltas = 0usize;
    let mut argument_fragment_deltas = 0usize;
    for frame in &frames {
        let delta = &frame["choices"][0]["delta"];
        if delta.get("content").is_some() {
            content_deltas += 1;
        }
        if let Some(deltas) = delta["tool_calls"].as_array() {
            for call_delta in deltas {
                if call_delta.get("id").is_some() {
                    call_start_deltas += 1;
                } else {
                    argument_fragment_deltas += 1;
                }
            }
        }
    }
    assert_eq!(
        call_start_deltas,
        buffered_calls.as_array().expect("buffered calls").len(),
        "one call start per tool call"
    );
    assert!(
        content_deltas >= 1,
        "the text before the block streams as content deltas"
    );
    assert!(
        argument_fragment_deltas >= call_start_deltas,
        "every call sends at least one arguments fragment"
    );
    let probe = serde_json::json!({
        "schema": "sparkflash-tool-stream-wire-probe/1",
        "generated_at_unix": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |delta| delta.as_secs()),
        "server_binary": env!("CARGO_BIN_EXE_sparkflash"),
        "script": {"pieces": pieces.len(), "delay_ms": 20, "keep_alive_ms": 5},
        "buffered": {
            "content": buffered_content,
            "finish_reason": buffered["choices"][0]["finish_reason"].clone(),
            "tool_calls": buffered_calls.clone(),
        },
        "streamed": {
            "content": streamed_content,
            "finish_reason": final_choice["finish_reason"].clone(),
            "tool_calls": Value::Array(streamed_calls.clone()),
        },
        "equality": {"content": content_equal, "tool_calls_byte_for_byte": tool_calls_equal},
        "delta_counts": {
            "content_deltas": content_deltas,
            "call_start_deltas": call_start_deltas,
            "argument_fragment_deltas": argument_fragment_deltas,
            "sse_data_frames": frames.len(),
        },
        "keep_alive": {
            "raw_frames": raw.len(),
            "comment_frames": comment_positions.len(),
            "comment_positions": comment_positions.clone(),
            "first_frame_is_data": raw.first().is_some_and(|frame| frame.starts_with("data: ")),
            "last_frame_is_done": raw.last().copied() == Some("data: [DONE]"),
            "comments_are_whole_frames": true,
        },
        "sse": {
            "role_frame_first": true,
            "ends_with_done": true,
            "dsml_leaked_into_content": streamed_content.contains("DSML"),
        },
        "wall": {"stream_wall_ms": wall_ms},
        "non_claims": [
            "host path, loopback socket only; no cluster, no GPU, no device, no vLLM comparison",
            "debug-build scripted stub engine, not a checkpoint: this is a parser/wire test, not a model-quality measurement",
            "single-run wall millisecond observations are repeatability context, not speed claims",
        ],
    });
    if let Ok(path) = std::env::var("SPARKFLASH_TOOL_STREAM_PROBE") {
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&probe).expect("probe JSON"),
        )
        .expect("probe file writes");
        eprintln!("tool-stream wire probe written to {path}");
    }
}

#[test]
fn scripted_wire_malformed_dsml_is_a_typed_sse_error() {
    let malformed = [
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"x\">\n",
        "<｜DSML｜ junk name=\"a\" string=\"true\">v</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n",
        "</｜DSML｜ calls>",
    ]
    .concat();
    let pieces = wire_pieces(&malformed, 4);
    let server = scripted_server(&pieces, 0, None);
    let response = server.request("POST", "/v1/chat/completions", Some(&wire_tools_body(true)));
    assert_eq!(response.status, 200, "log: {}", server.log());
    let text = response.text();
    assert!(
        text.ends_with("data: [DONE]\n\n"),
        "the terminator is last: {text}"
    );
    assert!(
        text.contains("\"code\":\"tool_call_parse_error\""),
        "the refusal code is typed: {text}"
    );
    let raw = wire_raw_frames(&text);
    assert_eq!(raw.last().copied(), Some("data: [DONE]"));
    let error_index = raw
        .iter()
        .position(|frame| frame.contains("\"error\""))
        .expect("an error frame");
    assert_eq!(
        error_index,
        raw.len() - 2,
        "the error frame is terminal, directly before the terminator: {text}"
    );
    let frames = wire_sse_frames(&text);
    assert!(
        frames
            .iter()
            .all(|frame| frame["choices"][0]["finish_reason"].is_null()),
        "no finish chunk rides a refused stream: {frames:?}"
    );

    // An unterminated block is the same typed refusal, discovered at end of stream instead of on
    // a malformed tag.
    let truncated = [
        "\n\n",
        "<｜DSML｜ calls>\n",
        "<｜DSML｜ invoke name=\"x\">\n",
        "<｜DSML｜ parameter name=\"a\" string=\"true\">unfinished",
    ]
    .concat();
    let pieces = wire_pieces(&truncated, 5);
    let server = scripted_server(&pieces, 0, None);
    let response = server.request("POST", "/v1/chat/completions", Some(&wire_tools_body(true)));
    assert_eq!(response.status, 200, "log: {}", server.log());
    let text = response.text();
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    assert!(
        text.contains("\"code\":\"tool_call_parse_error\""),
        "the unterminated block is a typed error: {text}"
    );
}

// ---------------------------------------------------------------------------------------
// Task-20 wire probes: multiple choices, response_format refusals, bearer auth
// ---------------------------------------------------------------------------------------

/// The buffered chat response for the given body.
fn wire_chat_response(server: &SyntheticServer, body: &str) -> HttpResponse {
    server.request("POST", "/v1/chat/completions", Some(body))
}

#[test]
fn wire_chat_n_multiple_choices_over_the_scripted_wire() {
    // The synthetic tiny plan's 128-row vocabulary cannot tokenize chat traffic, so the chat
    // n probes run over the debug-build scripted engine (the same seam the DSML wire tests use);
    // the token-id completion probes below cover the synthetic serial/batched paths directly.
    let pieces = wire_pieces("Hello world", 4);
    let server = scripted_server(&pieces, 0, None);
    let baseline = wire_chat_response(&server, &chat_body(serde_json::json!({})));
    assert_eq!(baseline.status, 200, "log: {}", server.log());
    let baseline = baseline.json();
    let response = wire_chat_response(&server, &chat_body(serde_json::json!({"n": 2})));
    assert_eq!(response.status, 200, "log: {}", server.log());
    let json = response.json();
    let choices = json["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 2, "n=2 returns two choices: {json}");
    for (index, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], index, "choices are in index order: {json}");
        assert_eq!(choice["message"]["role"], "assistant");
        assert_eq!(
            choice["message"]["content"],
            baseline["choices"][0]["message"]["content"]
        );
    }
    let usage = &json["usage"];
    assert_eq!(
        usage["prompt_tokens"], baseline["usage"]["prompt_tokens"],
        "the prompt is counted once: {json}"
    );
    assert_eq!(
        usage["completion_tokens"].as_u64().expect("completion"),
        baseline["usage"]["completion_tokens"]
            .as_u64()
            .expect("baseline completion")
            * 2,
        "completion tokens are summed over the choices: {json}"
    );
    assert_eq!(
        usage["total_tokens"].as_u64().expect("total"),
        usage["prompt_tokens"].as_u64().expect("prompt")
            + usage["completion_tokens"].as_u64().expect("completion")
    );
    assert!(
        json["x_deepseek_spark_stats"].is_object(),
        "the extension is aggregated: {json}"
    );
}

#[test]
fn wire_chat_n_stream_over_the_scripted_wire() {
    let pieces = wire_pieces("Hello world", 4);
    let server = scripted_server(&pieces, 0, None);
    let body = chat_body(serde_json::json!({
        "n": 2,
        "stream": true,
        "stream_options": {"include_usage": true},
    }));
    let response = wire_chat_response(&server, &body);
    assert_eq!(response.status, 200, "log: {}", server.log());
    assert!(
        response.text().ends_with(
            "data: [DONE]

"
        ),
        "the stream terminates in [DONE]: {}",
        response.text()
    );
    let frames = sse_frames(&response);
    let role_frames = frames
        .iter()
        .filter(|frame| frame["choices"][0]["delta"]["role"] == "assistant")
        .count();
    assert_eq!(
        role_frames, 1,
        "one role chunk opens the response: {frames:?}"
    );
    let finish_indexes: Vec<u64> = frames
        .iter()
        .filter(|frame| {
            frame["choices"]
                .as_array()
                .is_some_and(|choices| !choices.is_empty())
                && frame["choices"][0]["finish_reason"].is_string()
        })
        .map(|frame| frame["choices"][0]["index"].as_u64().expect("index"))
        .collect();
    assert_eq!(finish_indexes, vec![0, 1], "each choice finishes in order");
    let usage_frame = frames
        .iter()
        .find(|frame| frame["choices"].as_array().is_some_and(Vec::is_empty))
        .expect("an include_usage chunk after all choices");
    assert_eq!(usage_frame["object"], "chat.completion.chunk");
    assert!(
        usage_frame["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion")
            > 0,
        "the usage chunk aggregates the choices: {usage_frame}"
    );
    assert_eq!(
        usage_frame["usage"]["total_tokens"]
            .as_u64()
            .expect("total"),
        usage_frame["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt")
            + usage_frame["usage"]["completion_tokens"]
                .as_u64()
                .expect("completion")
    );
}

#[test]
fn wire_completions_n_choices_serial_and_batched() {
    let serial = SyntheticServer::start();
    let batched = SyntheticServer::start_with(&["--continuous-batching", "--max-batch", "2"]);
    let prompt = serde_json::json!([1, 2, 3, 4, 5, 6, 7, 8]);
    let mut payloads = Vec::new();
    for server in [&serial, &batched] {
        let baseline = completion(
            &server.base,
            serde_json::json!({"prompt": prompt, "max_tokens": 4, "temperature": 0.0}),
        );
        assert_eq!(baseline.status, 200, "log: {}", server.log());
        let baseline = baseline.json();
        let response = completion(
            &server.base,
            serde_json::json!({"prompt": prompt, "max_tokens": 4, "temperature": 0.0, "n": 2}),
        );
        assert_eq!(response.status, 200, "log: {}", server.log());
        let json = response.json();
        let choices = json["choices"].as_array().expect("choices");
        let indexes: Vec<u64> = choices
            .iter()
            .map(|choice| choice["index"].as_u64().expect("index"))
            .collect();
        assert_eq!(indexes, vec![0, 1], "one prompt's two choices: {json}");
        assert_eq!(json["object"], "text_completion");
        let usage = &json["usage"];
        assert_eq!(
            usage["prompt_tokens"], baseline["usage"]["prompt_tokens"],
            "the prompt is counted once: {json}"
        );
        assert_eq!(
            usage["completion_tokens"].as_u64().expect("completion"),
            baseline["usage"]["completion_tokens"]
                .as_u64()
                .expect("baseline completion")
                * 2
        );
        payloads.push(strip_volatile(&json));
    }
    assert_eq!(
        payloads[0], payloads[1],
        "the serial and continuous-batching paths answer a completion n=2 identically"
    );
}

#[test]
fn wire_completions_n_stream_serial_and_batched() {
    let serial = SyntheticServer::start();
    let batched = SyntheticServer::start_with(&["--continuous-batching", "--max-batch", "2"]);
    let body = serde_json::json!({
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 4,
        "temperature": 0.0,
        "n": 2,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let serial_response = completion(&serial.base, body.clone());
    let batched_response = completion(&batched.base, body);
    let frames = assert_same_stream(
        &serial_response,
        &batched_response,
        "completions n=2 stream",
    );
    let finish_indexes: Vec<u64> = frames
        .iter()
        .filter(|frame| {
            frame["choices"]
                .as_array()
                .is_some_and(|choices| !choices.is_empty())
                && frame["choices"][0]["finish_reason"].is_string()
        })
        .map(|frame| frame["choices"][0]["index"].as_u64().expect("index"))
        .collect();
    assert_eq!(finish_indexes, vec![0, 1]);
    let usage_frame = frames
        .iter()
        .find(|frame| frame["choices"].as_array().is_some_and(Vec::is_empty))
        .expect("an include_usage chunk after all choices");
    assert!(
        usage_frame["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion")
            > 0
    );
}

#[test]
fn wire_response_format_json_and_n_zero_are_typed_400s() {
    let pieces = wire_pieces("Hello", 4);
    let server = scripted_server(&pieces, 0, None);
    for response_format in [
        serde_json::json!({"type": "json_object"}),
        serde_json::json!({"type": "json_schema", "json_schema": {
            "name": "answer",
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        }}),
    ] {
        let body = chat_body(serde_json::json!({"response_format": response_format.clone()}));
        let response = wire_chat_response(&server, &body);
        assert_eq!(
            response.status,
            400,
            "an unenforceable response_format is refused: {}",
            response.text()
        );
        let error = response.json();
        assert_eq!(error["error"]["param"], "response_format", "{error}");
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert_eq!(error["error"]["code"], 400);
        assert!(
            error["error"]["message"]
                .as_str()
                .expect("message")
                .contains("constrained decoding"),
            "{error}"
        );
    }
    // A text response_format is the OpenAI default and a no-op.
    let text = wire_chat_response(
        &server,
        &chat_body(serde_json::json!({"response_format": {"type": "text"}})),
    );
    assert_eq!(text.status, 200, "log: {}", server.log());
    // n = 0 on both endpoints is a typed 400 before generation.
    let chat = wire_chat_response(&server, &chat_body(serde_json::json!({"n": 0})));
    assert_eq!(chat.status, 400);
    assert_eq!(chat.json()["error"]["param"], "n");
    let completions = completion(
        &server.base,
        serde_json::json!({"prompt": [1, 2, 3], "n": 0}),
    );
    assert_eq!(completions.status, 400);
    assert_eq!(completions.json()["error"]["param"], "n");
}

#[test]
fn wire_bearer_key_protects_routes_and_keeps_health_open() {
    let server = SyntheticServer::start_with(&["--api-key", "s3cret"]);
    // Liveness, readiness and the scrape are public: probes and the local Prometheus cannot be
    // given a secret. Readiness is the drain-aware answer, so a probe sees 503 during a drain
    // without an authorization header.
    for path in ["/health", "/readyz", "/metrics"] {
        let response = server.request("GET", path, None);
        assert_eq!(
            response.status,
            200,
            "{path} stays reachable without the key: {}",
            response.text()
        );
    }
    let models = server.request("GET", "/v1/models", None);
    assert_eq!(models.status, 401);
    assert_eq!(
        models.json(),
        serde_json::json!({"error": {
            "message": "Unauthorized",
            "type": "invalid_request_error",
            "param": null,
            "code": 401,
        }})
    );
    for header in ["Bearer wrong", "Basic s3cret", "s3cret"] {
        let response =
            server.request_with_headers("GET", "/v1/models", None, &[("authorization", header)]);
        assert_eq!(response.status, 401, "header: {header:?}");
    }
    for header in ["Bearer s3cret", "bearer s3cret"] {
        let response =
            server.request_with_headers("GET", "/v1/models", None, &[("authorization", header)]);
        assert_eq!(response.status, 200, "header: {header:?}");
        assert!(
            response.json()["data"].is_array(),
            "the models card is served with the key"
        );
    }
    // A generation endpoint is protected too: the synthetic plan's token-id completion path is
    // the one the un-keyed wire tests already cover, so use it as the positive control.
    let body = serde_json::json!({"prompt": [1, 2, 3, 4], "max_tokens": 4, "temperature": 0.0})
        .to_string();
    let missing = server.request("POST", "/v1/completions", Some(&body));
    assert_eq!(missing.status, 401);
    let wrong = server.request_with_headers(
        "POST",
        "/v1/completions",
        Some(&body),
        &[("authorization", "Bearer nope")],
    );
    assert_eq!(wrong.status, 401);
    let correct = server.request_with_headers(
        "POST",
        "/v1/completions",
        Some(&body),
        &[("authorization", "Bearer s3cret")],
    );
    assert_eq!(correct.status, 200, "log: {}", server.log());
    assert!(correct.json()["choices"].is_array());
}

#[test]
fn a_device_token_window_serves_its_claimed_completion_and_labels_it_unmeasured() {
    // A synthetic four-rank window (paper records; UNMEASURED) served through the real binary:
    // the request carries exactly the recorded prompt ids, so the device source claims it and the
    // response is the tokenizer's decode of the recorded ids with the device provenance headers.
    // The second request is not the recorded window, so the host twin must answer it as before.
    let window = tempfile::tempdir().expect("window dir");
    sparkflash_api::device_chain::fixtures::write_envelope(window.path(), true);
    let window_arg = window.path().to_str().expect("temporary paths are UTF-8");
    let server = SyntheticServer::start_with(&["--device-token-dir", window_arg]);
    let seam = sparkflash_api::encoding::TokenizerSeam::from_path(
        &sparkflash_api::encoding::reference_tokenizer_path(),
    )
    .expect("reference tokenizer");
    let ids = sparkflash_api::device_chain::fixtures::DECODE_IDS;
    let expected = seam.decode(&ids).expect("the claimed ids decode");
    let prompt = sparkflash_api::device_chain::fixtures::prompt_ids();
    let body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": prompt,
        "max_tokens": 9,
    })
    .to_string();
    let response = server.request("POST", "/v1/completions", Some(&body));
    assert_eq!(response.status, 200, "log: {}", server.log());
    assert_eq!(
        response.header("x-sparkflash-token-source"),
        Some("device-chain-envelope (UNMEASURED)"),
        "the device body is labelled UNMEASURED: {}",
        server.log()
    );
    assert_eq!(
        response.header("x-sparkflash-token-claim"),
        Some("explicit"),
        "the claim form is on the wire: {}",
        server.log()
    );
    assert!(
        response.header("x-deepseek-spark-stats").is_none(),
        "the device path carries no host stats block"
    );
    let json = response.json();
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["choices"][0]["finish_reason"], "length");
    assert_eq!(json["choices"][0]["text"], expected.as_str());
    // The usage block keeps the host path's shape (prompt-cache details included); the token
    // arithmetic is the claimed window's: 130 prompt ids, nine emitted tokens, the sum.
    assert_eq!(
        json["usage"]["prompt_tokens"],
        sparkflash_api::device_chain::fixtures::PROMPT_LEN
    );
    assert_eq!(json["usage"]["completion_tokens"], 9);
    assert_eq!(
        json["usage"]["total_tokens"],
        sparkflash_api::device_chain::fixtures::PROMPT_LEN + 9
    );
    assert_eq!(json["usage"]["prompt_cache_hit_tokens"], 0);

    // The startup line is printed on stderr; the drain threads catch up on their own schedule.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !server.log().contains("claim ok") && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        server.log().contains("device token source"),
        "the startup line names the source: {}",
        server.log()
    );

    // A request that is not the recorded window falls back to the engine path, byte-shape intact.
    let host_body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 4,
    })
    .to_string();
    let host = server.request("POST", "/v1/completions", Some(&host_body));
    assert_eq!(host.status, 200, "log: {}", server.log());
    assert!(
        host.header("x-sparkflash-token-source").is_none(),
        "the fallback answer must not carry the device provenance header"
    );
    let host_json = host.json();
    assert_eq!(host_json["usage"]["prompt_tokens"], 8);
    assert_eq!(host_json["choices"][0]["finish_reason"], "length");
}

#[test]
fn a_device_window_recorded_for_the_rendered_chat_prompt_serves_the_chat_endpoint() {
    // W4 over the real binary: chat renders + tokenizes the prompt before asking the device
    // source, so a window recorded for exactly that prompt is claimable from
    // /v1/chat/completions instead of always falling back to the host twin.
    let seam = sparkflash_api::encoding::TokenizerSeam::from_path(
        &sparkflash_api::encoding::reference_tokenizer_path(),
    )
    .expect("reference tokenizer");
    let body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 9,
    });
    // The server renders the released template from the raw request body and tokenizes it; the
    // window must record those exact ids for the exact-match rule to accept the request.
    let ids = seam
        .encode_chat(&body)
        .expect("the chat prompt renders and tokenizes");
    let window = tempfile::tempdir().expect("window dir");
    fixtures::write_envelope_for_prompt(window.path(), &ids);
    let window_arg = window.path().to_str().expect("temporary paths are UTF-8");
    let server = SyntheticServer::start_with(&["--device-token-dir", window_arg]);

    let expected = seam
        .decode(&fixtures::DECODE_IDS)
        .expect("the claimed ids decode");
    let response = server.request(
        "POST",
        "/v1/chat/completions",
        Some(&serde_json::to_string(&body).expect("chat body")),
    );
    assert_eq!(
        response.status,
        200,
        "log: {} body: {}",
        server.log(),
        response.text()
    );
    assert_eq!(
        response.header("x-sparkflash-token-source"),
        Some("device-chain-envelope (UNMEASURED)"),
        "the chat request must reach the device source, not the host twin: {}",
        server.log()
    );
    assert_eq!(
        response.header("x-sparkflash-token-claim"),
        Some("explicit")
    );
    let json = response.json();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["content"], expected.as_str());
    assert_eq!(json["choices"][0]["finish_reason"], "length");
    assert_eq!(json["usage"]["prompt_tokens"], ids.len());
}

#[test]
fn the_serve_help_documents_the_device_token_flags() {
    // Wiring W3 is a CLI surface, so the documented flags are part of the contract the node
    // operator follows. This runs the real binary's help, not the parser.
    let output = Command::new(env!("CARGO_BIN_EXE_sparkflash"))
        .args(["serve", "--help"])
        .output()
        .expect("serve --help runs");
    // The CLI prints serve help on stderr and returns its usage exit code (2) even for --help;
    // the flag inventory is the contract, not the stream or the status.
    let help = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(help.contains("--device-token-dir"), "{help}");
    assert!(help.contains("--device-token-derived-claim"), "{help}");
    assert!(help.contains("--device-token-live"), "{help}");
}

#[test]
fn a_live_window_follower_serves_the_newest_completed_window_and_falls_back_for_the_rest() {
    // W5 over the real binary: the operator points --device-token-dir at a *parent* directory and
    // passes --device-token-live. The source must find the completed window underneath it, serve
    // the recorded prompt, and fall back (host twin, no device header) for anything else.
    let parent = tempfile::tempdir().expect("parent dir");
    let window = parent.path().join("window_a");
    std::fs::create_dir_all(&window).expect("window directory");
    fixtures::write_envelope(&window, true);
    std::fs::write(window.join("run.meta"), "run=live-a tag=test\n").expect("run meta");
    let parent_arg = parent.path().to_str().expect("temporary paths are UTF-8");
    let server =
        SyntheticServer::start_with(&["--device-token-dir", parent_arg, "--device-token-live"]);
    let seam = sparkflash_api::encoding::TokenizerSeam::from_path(
        &sparkflash_api::encoding::reference_tokenizer_path(),
    )
    .expect("reference tokenizer");
    let ids = fixtures::DECODE_IDS;
    let expected = seam.decode(&ids).expect("the claimed ids decode");

    let body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": fixtures::prompt_ids(),
        "max_tokens": 9,
    })
    .to_string();
    let response = server.request("POST", "/v1/completions", Some(&body));
    assert_eq!(response.status, 200, "log: {}", server.log());
    assert_eq!(
        response.header("x-sparkflash-token-source"),
        Some("device-chain-live-window (UNMEASURED)"),
        "the live source labels its own claim: {}",
        server.log()
    );
    assert_eq!(response.json()["choices"][0]["text"], expected.as_str());

    // A prompt the active window did not record falls back, and the fallback is the host twin.
    let host_body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 4,
    })
    .to_string();
    let host = server.request("POST", "/v1/completions", Some(&host_body));
    assert_eq!(host.status, 200, "log: {}", server.log());
    assert!(
        host.header("x-sparkflash-token-source").is_none(),
        "the fallback answer carries no device provenance header"
    );
}

#[test]
fn a_live_window_follower_serves_a_window_recorded_after_startup() {
    // The follower must re-read on transition, not snapshot at boot: start with an empty parent,
    // then write the window the request will ask for, and only then send it.
    let parent = tempfile::tempdir().expect("parent dir");
    let parent_arg = parent.path().to_str().expect("temporary paths are UTF-8");
    let server =
        SyntheticServer::start_with(&["--device-token-dir", parent_arg, "--device-token-live"]);
    // Before any window exists the live source reports the absence and the host twin answers.
    // (A small prompt, because the synthetic host twin's embedding table only holds 128 rows; the
    // fixture prompt ids are 1000..1129 and the host twin is only expected to answer the ids the
    // device window recorded.)
    let early_body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 4,
    })
    .to_string();
    let early = server.request("POST", "/v1/completions", Some(&early_body));
    assert_eq!(
        early.status,
        200,
        "log: {} body: {}",
        server.log(),
        early.text()
    );
    assert!(
        early.header("x-sparkflash-token-source").is_none(),
        "no window yet, so the host twin answers"
    );

    let window = parent.path().join("window_b");
    std::fs::create_dir_all(&window).expect("window directory");
    fixtures::write_envelope(&window, true);
    std::fs::write(window.join("run.meta"), "run=live-b tag=test\n").expect("run meta");

    let window_body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": fixtures::prompt_ids(),
        "max_tokens": 9,
    })
    .to_string();
    let response = server.request("POST", "/v1/completions", Some(&window_body));
    assert_eq!(
        response.header("x-sparkflash-token-source"),
        Some("device-chain-live-window (UNMEASURED)"),
        "the window written after startup is picked up: {}",
        server.log()
    );
}

#[test]
fn a_device_token_window_streams_its_claimed_completion_and_labels_it_unmeasured() {
    // The buffered device test above proves the body; this one proves the SSE path over the same
    // synthetic window (paper records, UNMEASURED): nine content frames, the finish chunk, the
    // usage chunk and the terminator, all from the claimed envelope.
    let window = tempfile::tempdir().expect("window dir");
    fixtures::write_envelope(window.path(), true);
    let window_arg = window.path().to_str().expect("temporary paths are UTF-8");
    let server = SyntheticServer::start_with(&["--device-token-dir", window_arg]);
    let seam = sparkflash_api::encoding::TokenizerSeam::from_path(
        &sparkflash_api::encoding::reference_tokenizer_path(),
    )
    .expect("reference tokenizer");
    let ids = fixtures::DECODE_IDS;
    let expected = seam.decode(&ids).expect("the claimed ids decode");
    let prompt = fixtures::prompt_ids();
    let body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": prompt,
        "max_tokens": 9,
        "stream": true,
        "stream_options": {"include_usage": true},
    })
    .to_string();
    let response = server.request("POST", "/v1/completions", Some(&body));
    assert_eq!(response.status, 200, "log: {}", server.log());
    assert_eq!(
        response.header("x-sparkflash-token-source"),
        Some("device-chain-envelope (UNMEASURED)")
    );
    assert_eq!(
        response.header("x-sparkflash-token-claim"),
        Some("explicit")
    );
    let text = response.text();
    assert!(
        text.ends_with("data: [DONE]\n\n"),
        "the stream ends with the terminator: {text}"
    );
    let frames = wire_sse_frames(&text);
    let mut assembled = String::new();
    let mut content_frames = 0usize;
    let mut finish_reasons = Vec::new();
    let mut usage = None;
    for frame in &frames {
        for choice in frame["choices"].as_array().into_iter().flatten() {
            if choice["finish_reason"].is_null() {
                content_frames += 1;
                assembled.push_str(choice["text"].as_str().unwrap_or(""));
            } else {
                finish_reasons.push(choice["finish_reason"].clone());
            }
        }
        if frame["usage"].is_object() {
            usage = Some(frame["usage"].clone());
        }
    }
    assert_eq!(
        content_frames,
        sparkflash_api::device_chain::DECODE_POSITIONS as usize,
        "{text}"
    );
    assert_eq!(assembled, expected);
    assert_eq!(finish_reasons, vec![serde_json::json!("length")]);
    let usage = usage.expect("the device stream carries a usage chunk");
    assert_eq!(usage["prompt_tokens"], fixtures::PROMPT_LEN);
    assert_eq!(usage["completion_tokens"], 9);
    assert_eq!(
        usage["total_tokens"],
        fixtures::PROMPT_LEN + sparkflash_api::device_chain::DECODE_POSITIONS
    );
    assert!(server.log().contains("claim ok"), "{}", server.log());
}

#[test]
fn chat_requests_fall_back_to_the_host_twin_with_a_device_window_installed() {
    // The device source refuses a chat text prompt by design (there is no prompt capture in the
    // window yet), so a chat request must be answered by the host twin exactly as before: no
    // provenance header and the host stats block still in the body.
    let window = tempfile::tempdir().expect("window dir");
    fixtures::write_envelope(window.path(), true);
    let window_arg = window.path().to_str().expect("temporary paths are UTF-8");
    let pieces = ["Hello", " host", " twin"];
    let server = SyntheticServer::start_scripted_with_extra(
        &["--device-token-dir", window_arg],
        &pieces,
        0,
        None,
    );
    let buffered = server.request(
        "POST",
        "/v1/chat/completions",
        Some(&chat_body(serde_json::json!({ "max_tokens": 64 }))),
    );
    assert_eq!(buffered.status, 200, "log: {}", server.log());
    assert!(
        buffered.header("x-sparkflash-token-source").is_none(),
        "chat is a host-twin request: {}",
        server.log()
    );
    assert!(buffered.header("x-sparkflash-token-claim").is_none());
    let json = buffered.json();
    assert_eq!(json["choices"][0]["message"]["content"], "Hello host twin");
    assert!(
        json["x_deepseek_spark_stats"].is_object(),
        "the host path still answers with its stats block: {json}"
    );

    let streamed = server.request(
        "POST",
        "/v1/chat/completions",
        Some(&chat_body(serde_json::json!({
            "max_tokens": 64,
            "stream": true,
        }))),
    );
    assert_eq!(streamed.status, 200, "log: {}", server.log());
    assert!(streamed.header("x-sparkflash-token-source").is_none());
    let text = streamed.text();
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    let frames = wire_sse_frames(&text);
    let content: String = frames
        .iter()
        .filter_map(|frame| frame["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "Hello host twin");
    assert_eq!(
        frames.last().expect("data frames")["choices"][0]["finish_reason"],
        "stop"
    );
}

#[test]
fn streaming_tool_calls_still_work_through_the_host_twin_with_a_device_window_installed() {
    // The DSML streaming parser sits above the engine seam, so with a device window installed a
    // chat request that offers tools must still take the host twin and keep the exact buffered /
    // streamed tool_calls equivalence. This pins that the device source never intercepts chat.
    let answer = wire_dsml_answer();
    let pieces = wire_pieces(&answer, 5);
    let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let window = tempfile::tempdir().expect("window dir");
    fixtures::write_envelope(window.path(), true);
    let window_arg = window.path().to_str().expect("temporary paths are UTF-8");
    let extra = ["--device-token-dir", window_arg];

    let buffered_server = SyntheticServer::start_scripted_with_extra(&extra, &refs, 0, None);
    let buffered_response = buffered_server.request(
        "POST",
        "/v1/chat/completions",
        Some(&wire_tools_body(false)),
    );
    assert_eq!(
        buffered_response.status,
        200,
        "log: {}",
        buffered_server.log()
    );
    assert!(
        buffered_response
            .header("x-sparkflash-token-source")
            .is_none(),
        "tool-call chat stays on the host twin"
    );
    let buffered = buffered_response.json();
    assert_eq!(buffered["choices"][0]["finish_reason"], "tool_calls");
    let buffered_calls = buffered["choices"][0]["message"]["tool_calls"].clone();
    assert!(
        buffered_calls
            .as_array()
            .is_some_and(|calls| !calls.is_empty()),
        "the buffered answer lifts the block: {buffered}"
    );
    drop(buffered_server);

    let stream_server = SyntheticServer::start_scripted_with_extra(&extra, &refs, 20, Some(5));
    let streamed_response =
        stream_server.request("POST", "/v1/chat/completions", Some(&wire_tools_body(true)));
    assert_eq!(
        streamed_response.status,
        200,
        "log: {}",
        stream_server.log()
    );
    assert!(
        streamed_response
            .header("x-sparkflash-token-source")
            .is_none()
    );
    assert_eq!(
        streamed_response.header("content-type"),
        Some("text/event-stream")
    );
    let text = streamed_response.text();
    assert!(text.ends_with("data: [DONE]\n\n"), "{text}");
    let frames = wire_sse_frames(&text);
    let streamed_calls = wire_assemble_calls(&frames);
    assert_eq!(
        Value::Array(streamed_calls),
        buffered_calls,
        "the streamed deltas reassemble the buffered tool_calls"
    );
    assert_eq!(
        frames.last().expect("data frames")["choices"][0]["finish_reason"],
        "tool_calls"
    );
}

#[test]
fn a_refused_device_window_is_fail_closed_and_names_the_reason() {
    // A refused claim must not serve a device token: the startup line names every reason and the
    // request falls back to the host twin. The same window schema still claims in the derived
    // form when the writer flag is false but the loop evidence is complete.
    let refused = tempfile::tempdir().expect("refused window dir");
    fixtures::write_envelope(refused.path(), false);
    let refused_arg = refused.path().to_str().expect("temporary paths are UTF-8");
    let server = SyntheticServer::start_with(&["--device-token-dir", refused_arg]);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !server.log().contains("claim refused") && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let log = server.log();
    assert!(log.contains("device token source"), "{log}");
    assert!(log.contains("claim refused"), "{log}");
    assert!(
        log.contains("token_claimed") || log.contains("does not assert"),
        "the refusal names the missing assertion: {log}"
    );
    // The claim is refused before matching, so the fallback request can use ids the synthetic
    // tiny plan can actually embed (the window's released-tokenizer ids would be a 503 there).
    let body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": [1, 2, 3, 4, 5, 6, 7, 8],
        "max_tokens": 4,
    })
    .to_string();
    let response = server.request("POST", "/v1/completions", Some(&body));
    assert_eq!(response.status, 200, "log: {}", server.log());
    assert!(
        response.header("x-sparkflash-token-source").is_none(),
        "a refused claim must not produce a device answer: {}",
        server.log()
    );
    assert!(
        response.json()["x_deepseek_spark_stats"].is_object(),
        "the host twin answers the request instead"
    );
    drop(server);

    let derived = tempfile::tempdir().expect("derived window dir");
    fixtures::write_derived_envelope(derived.path());
    let derived_arg = derived.path().to_str().expect("temporary paths are UTF-8");
    let derived_server = SyntheticServer::start_with(&[
        "--device-token-dir",
        derived_arg,
        "--device-token-derived-claim",
    ]);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !derived_server.log().contains("claim ok") && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    let seam = sparkflash_api::encoding::TokenizerSeam::from_path(
        &sparkflash_api::encoding::reference_tokenizer_path(),
    )
    .expect("reference tokenizer");
    let expected = seam.decode(&fixtures::DECODE_IDS).expect("ids decode");
    let body = serde_json::json!({
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "prompt": fixtures::prompt_ids(),
        "max_tokens": 9,
    })
    .to_string();
    let response = derived_server.request("POST", "/v1/completions", Some(&body));
    assert_eq!(response.status, 200, "log: {}", derived_server.log());
    assert_eq!(
        response.header("x-sparkflash-token-claim"),
        Some("derived"),
        "the derived form is labelled on the wire: {}",
        derived_server.log()
    );
    assert_eq!(response.json()["choices"][0]["text"], expected.as_str());
}

#[test]
fn sigterm_drains_in_flight_streams_and_turns_readiness_off() {
    // A supervisor sends SIGTERM, not SIGKILL: the process must stop reporting ready immediately,
    // keep finishing what it already accepted, and then exit 0. The scripted engine's delay makes
    // the in-flight request deterministic: five pieces, one every 100 ms.
    let mut server = SyntheticServer::start_scripted_with_env(
        &[],
        &[
            ("SPARKFLASH_SERVE_DRAIN_DELAY_MS", "1500"),
            ("SPARKFLASH_SERVE_DRAIN_TIMEOUT_MS", "10000"),
        ],
        &["a", "b", "c", "d", "e"],
        100,
        None,
    );
    let address = server
        .base
        .strip_prefix("http://")
        .expect("the server prints an http URL")
        .to_owned();
    let body = chat_body(serde_json::json!({"stream": true}));
    let mut stream = TcpStream::connect(&address).expect("connect to the test server");
    stream
        .set_read_timeout(Some(IO_TIMEOUT))
        .expect("read timeout");
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .expect("write the streaming request");
    // Read the first bytes so the request is provably in flight when the signal arrives.
    let mut head = [0u8; 256];
    let read = stream
        .read(&mut head)
        .expect("read the first response bytes");
    assert!(read > 0, "log: {}", server.log());
    server.send_sigterm();
    // Readiness turns 503 immediately; the listener keeps accepting for the pre-stop delay, so a
    // retry loop observes the transition instead of racing the signal handler.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        assert!(
            Instant::now() < deadline,
            "readyz never turned 503; log:\n{}",
            server.log()
        );
        if let Ok(mut probe) = TcpStream::connect(&address) {
            let _ = probe.set_read_timeout(Some(Duration::from_secs(2)));
            let _ =
                probe.write_all(b"GET /readyz HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n");
            let mut raw = Vec::new();
            let _ = probe.read_to_end(&mut raw);
            if !raw.is_empty() {
                let response = HttpResponse::parse(&raw);
                if response.status == 503 {
                    break;
                }
                assert_eq!(
                    response.status,
                    200,
                    "only 200 before the drain signal turns 503; log:\n{}",
                    server.log()
                );
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    // Liveness stays 200: the process is alive and finishing the accepted request.
    let health = server.request("GET", "/health", None);
    assert_eq!(health.status, 200, "log: {}", server.log());
    // The in-flight stream finishes normally; the drain does not cut an accepted request.
    let mut rest = Vec::new();
    stream
        .read_to_end(&mut rest)
        .expect("read the rest of the stream");
    let wire = format!(
        "{}{}",
        String::from_utf8_lossy(&head[..read]),
        String::from_utf8_lossy(&rest)
    );
    assert!(
        wire.contains("[DONE]"),
        "drained stream is complete: {wire}"
    );
    // The process exits 0 once the drain window is over, with the timeline a supervisor reads.
    let status = server
        .wait_for_exit(Duration::from_secs(10))
        .expect("the server exits after the drain");
    assert!(
        status.success(),
        "a completed drain exits 0: {status:?}\n{}",
        server.log()
    );
    let log = server.log();
    for phase in [
        "phase=drain_start",
        "phase=drain_accepting_stopped",
        "phase=drain_complete",
    ] {
        assert!(log.contains(phase), "missing {phase} in:\n{log}");
    }
}
