//! M1 offline contract for `crates/sparkflash-device` (design 7 M1, tests/engine_offline.rs).
//!
//! No device, checkpoint, cluster or network: these run on a CPU-only build. They pin the M1
//! boundary only - the capability report, the typed vision refusal, the non-greedy refusal and the
//! delegation - and make no device-execution claim.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::stream::{BoxStream, StreamExt};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, InferenceEngine,
    MissingCapability, MockEngine, MockResponse,
};
use sparkflash_api::{ImagePart, MODEL_ID};
use sparkflash_device::{DeviceEngine, DeviceRefusal, FakeSession, NON_GREEDY_CODE};

/// A reference engine that counts how often it was entered, so a refusal can prove it did not
/// delegate and a greedy request can prove it did.
struct CountingReference {
    inner: MockEngine,
    calls: Arc<AtomicUsize>,
}

impl CountingReference {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        (
            Self {
                inner: MockEngine::new(MODEL_ID).with_response(MockResponse {
                    reasoning: Vec::new(),
                    content: vec!["hi".to_owned()],
                    tool_calls: Vec::new(),
                    finish_reason: FinishReason::Stop,
                }),
                calls: Arc::clone(&calls),
            },
            calls,
        )
    }
}

impl InferenceEngine for CountingReference {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.generate(request)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.inner.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
}

fn request(temperature: f32) -> EngineRequest {
    EngineRequest {
        request_id: "m1".to_owned(),
        model: MODEL_ID.to_owned(),
        prompt_tokens: vec![1, 2, 3],
        images: Vec::new(),
        max_tokens: 4,
        temperature,
        top_p: 1.0,
        thinking: false,
        reasoning_effort: None,
        stop: Vec::new(),
        seed: Some(0),
        trace_scopes: 0,
        trace_steps: 0,
    }
}

fn collect(stream: BoxStream<'static, EngineEvent>) -> Vec<EngineEvent> {
    futures::executor::block_on(stream.collect::<Vec<_>>())
}

fn engine() -> (DeviceEngine<FakeSession>, Arc<AtomicUsize>) {
    let (reference, calls) = CountingReference::new();
    let engine = DeviceEngine::open(FakeSession::new(), Arc::new(reference))
        .expect("the fake session opens");
    (engine, calls)
}

#[test]
fn the_capability_report_says_vision_false_and_fake_session() {
    let (engine, _) = engine();
    let report = engine.report();
    assert!(!report.vision, "M1 has no vision path");
    assert!(report.fake_session, "M1 runs on the fake session");
    assert!(!report.device_execution, "M1 launches no device kernel");
    assert_eq!(engine.session_name(), "fake");
    assert!(engine.fake_session());
    // The api seam itself has no vision channel: the conservative set is the trait default.
    assert_eq!(engine.capabilities(), Capabilities::default());
}

#[test]
fn an_image_request_is_refused_through_the_typed_missing_capability_path() {
    let (engine, calls) = engine();
    let mut image_request = request(0.0);
    image_request.images = vec![ImagePart::Url {
        url: "https://example.invalid/x.png".to_owned(),
    }];

    let missing = engine
        .validate_request(&image_request)
        .expect_err("an image part must be refused");
    assert_eq!(missing, MissingCapability::vision());
    assert_eq!(missing.capability(), sparkflash_api::Capability::Vision);
    assert_eq!(missing.code(), "unsupported_capability");

    // A direct caller of `generate` gets the same refusal as one terminal error event, and the
    // reference engine is never entered.
    let events = collect(engine.generate(image_request));
    assert_eq!(events.len(), 1);
    match &events[0] {
        EngineEvent::Error { code, message } => {
            assert_eq!(code, "unsupported_capability");
            assert!(message.contains("vision"), "message: {message}");
        }
        other => panic!("expected one error event, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.session_requests(), 0);
}

#[test]
fn a_non_greedy_request_is_refused_before_the_reference_engine() {
    let (engine, calls) = engine();
    let events = collect(engine.generate(request(0.7)));
    assert_eq!(events.len(), 1);
    match &events[0] {
        EngineEvent::Error { code, message } => {
            assert_eq!(code, NON_GREEDY_CODE);
            assert!(message.contains("0.7"), "message: {message}");
        }
        other => panic!("expected one error event, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.session_requests(), 0);
    assert_eq!(
        engine.check_request(&request(0.7)),
        Err(DeviceRefusal::NonGreedy { temperature: 0.7 })
    );
}

#[test]
fn a_greedy_request_delegates_to_the_reference_engine() {
    let (engine, calls) = engine();
    let events = collect(engine.generate(request(0.0)));
    assert_eq!(calls.load(Ordering::SeqCst), 1, "delegated once");
    assert_eq!(engine.session_requests(), 1);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, EngineEvent::ContentDelta(text) if text == "hi")),
        "events: {events:?}"
    );
}
