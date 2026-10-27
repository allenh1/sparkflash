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

//! Offline contract for `crates/sparkflash-device` (design 7 M1 and 7 M2, tests/engine_offline.rs).
//!
//! No device, checkpoint, cluster or network: these run on a CPU-only build. They pin the boundary
//! and the honesty of the report - the capability report, the typed vision refusal, the non-greedy
//! refusal, the delegation, and the rule that `device_execution` is derived from the session's
//! evidence, never from a flag. The M2 launch itself is a dev-window run, not a test.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::stream::{BoxStream, StreamExt};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, FinishReason, InferenceEngine,
    MissingCapability, MockEngine, MockResponse,
};
use sparkflash_api::{ImagePart, MODEL_ID};
use sparkflash_device::{
    DeviceEngine, DeviceRefusal, DeviceSession, ExecutionEvidence, FakeSession, NON_GREEDY_CODE,
    SessionError,
};

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
        logprobs: sparkflash_api::LogprobsRequest::default(),
        images: Vec::new(),
        vision: Vec::new(),
        max_tokens: 4,
        temperature,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        skip_special_tokens: true,
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

/// A non-fake session that reports one piece of execution evidence, so the report's derivation can
/// be tested without a device. It is a test double for the `DeviceSession` seam, not a claim
/// that anything ran.
struct ProofSession {
    evidence: ExecutionEvidence,
}

impl DeviceSession for ProofSession {
    fn name(&self) -> &'static str {
        "proof"
    }

    fn fake(&self) -> bool {
        false
    }

    fn open(&self) -> Result<(), SessionError> {
        Ok(())
    }

    fn record_request(&self) {}

    fn requests(&self) -> usize {
        0
    }

    fn execution(&self) -> Option<ExecutionEvidence> {
        Some(self.evidence.clone())
    }
}

fn proof_evidence() -> ExecutionEvidence {
    ExecutionEvidence {
        kernel: "moe_router_sm121",
        cubin_sha256: "5992cef91c3e01be46c0691a5d905877b38a2c3002d90d5c31b80a003924769b".to_owned(),
        cubin_bytes: 37_384,
        launches: 1,
        tokens: 8,
        outputs_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            .to_owned(),
    }
}

#[test]
fn a_non_fake_session_derives_device_execution_from_its_evidence() {
    let (reference, _) = CountingReference::new();
    let engine = DeviceEngine::open(
        ProofSession {
            evidence: proof_evidence(),
        },
        Arc::new(reference),
    )
    .expect("the proof session opens");
    let report = engine.report();
    assert!(!report.fake_session, "the session is not fake");
    assert!(report.device_execution, "evidence means something ran");
    assert!(!report.chain_execution, "M2 has no chain");
    let line = report.startup_line();
    let evidence = report.evidence.expect("the evidence is repeated");
    assert_eq!(evidence.kernel, "moe_router_sm121");
    assert_eq!(evidence.launches, 1);
    assert!(line.contains("M2 session/launch proof"), "{line}");
    assert!(!line.contains("M1 fake-session skeleton"), "{line}");
}

#[test]
fn the_fake_session_reports_no_execution_and_documents_the_m1_boundary() {
    let (engine, _) = engine();
    let report = engine.report();
    assert!(report.fake_session);
    assert!(!report.device_execution);
    assert!(!report.chain_execution);
    assert!(report.evidence.is_none());
    let line = report.startup_line();
    assert!(line.contains("M1 fake-session skeleton"), "{line}");
    assert!(line.contains("first real session is M2"), "{line}");
    assert!(line.contains("unsupported_sampling"), "{line}");
    assert!(line.contains("503"), "{line}");
}

/// A session that contradicts itself: it says it is fake and also hands back evidence. The report
/// must resolve that contradiction in the safe direction - fake wins.
struct ContradictorySession;

impl DeviceSession for ContradictorySession {
    fn name(&self) -> &'static str {
        "contradictory"
    }

    fn fake(&self) -> bool {
        true
    }

    fn open(&self) -> Result<(), SessionError> {
        Ok(())
    }

    fn record_request(&self) {}

    fn requests(&self) -> usize {
        0
    }

    fn execution(&self) -> Option<ExecutionEvidence> {
        Some(proof_evidence())
    }
}

/// A real session that has not executed anything: there is no evidence to claim.
struct EvidenceLessRealSession;

impl DeviceSession for EvidenceLessRealSession {
    fn name(&self) -> &'static str {
        "real-no-evidence"
    }

    fn fake(&self) -> bool {
        false
    }

    fn open(&self) -> Result<(), SessionError> {
        Ok(())
    }

    fn record_request(&self) {}

    fn requests(&self) -> usize {
        0
    }
}

#[test]
fn a_fake_session_ignores_evidence_and_never_prints_the_m2_label() {
    let (reference, _) = CountingReference::new();
    let engine = DeviceEngine::open(ContradictorySession, Arc::new(reference))
        .expect("the contradictory session opens");
    let report = engine.report();
    assert!(report.fake_session);
    assert!(
        !report.device_execution,
        "a fake session cannot have executed anything"
    );
    assert!(report.evidence.is_none(), "the evidence is ignored");
    let line = report.startup_line();
    assert!(line.contains("M1 fake-session skeleton"), "{line}");
    assert!(line.contains("fake_session=true"), "{line}");
    assert!(line.contains("device_execution=false"), "{line}");
    assert!(!line.contains("M2 session/launch proof"), "{line}");
}

#[test]
fn a_real_session_without_evidence_claims_nothing() {
    let (reference, _) = CountingReference::new();
    let engine = DeviceEngine::open(EvidenceLessRealSession, Arc::new(reference))
        .expect("the no-evidence session opens");
    let report = engine.report();
    assert!(!report.fake_session);
    assert!(!report.device_execution);
    assert!(!report.chain_execution);
    assert!(report.evidence.is_none());
    let line = report.startup_line();
    assert!(
        line.contains("real device session, no execution evidence"),
        "{line}"
    );
    assert!(line.contains("device_execution=false"), "{line}");
    assert!(!line.contains("M1 fake-session skeleton"), "{line}");
    assert!(!line.contains("M2 session/launch proof"), "{line}");
}
