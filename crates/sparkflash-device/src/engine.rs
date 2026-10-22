//! M2: `DeviceEngine`, the serving engine in front of a device session.
//!
//! This is design `docs/archive/build/device-forward-chain.md` 1.3, 7 M1 and 7 M2:
//!
//! - **Seam only.** It implements [`sparkflash_api::engine::InferenceEngine`] over a
//!   [`DeviceSession`] and delegates `generate` to a reference `Arc<dyn InferenceEngine>`.
//!   The reference driver is bin-only, so the cli supplies it (design 1.3); this crate cannot name
//!   `ReferenceDriver`.
//! - **Fake session is the default.** [`FakeSession`] records requests and produces no device
//!   result, so the default build, the CI host and every test here stay GPU-free. M2's real
//!   `crate::CudaSession` is behind the `cuda` feature and selected only by the cli.
//! - **Evidence, not arithmetic.** The engine holds the session's [`ExecutionEvidence`] (plain
//!   data), never a driver, device pointer or cubin. `device_execution` is derived from that
//!   evidence, so a session cannot claim a launch the engine cannot see. `chain_execution` stays
//!   false in M2: the chain is still absent.
//! - **Capability report.** [`DeviceEngine::report`] states `vision: false`, the session name and
//!   what actually ran, with the kernel id and cubin sha256. The api seam's [`Capabilities`] has no
//!   vision channel (D27/!68) and the delegated reference engine advertises no optional channel, so
//!   [`InferenceEngine::capabilities`] deliberately stays [`Capabilities::default()`]: the report is
//!   the explicit device surface, the trait set is the conservative protocol answer. Images are
//!   refused through the typed [`MissingCapability::vision`] path before any delegation.
//! - **Non-greedy refused.** A request with `temperature != 0.0` is refused with
//!   [`NON_GREEDY_CODE`] before the reference engine is called. The api seam has no typed
//!   non-capability refusal, so the refusal surfaces as `EngineEvent::Error`: HTTP 503 with code
//!   `unsupported_sampling` for a non-streaming request, an SSE error event for a streaming one.
//!   The startup line documents that behaviour beside the flag.
//! - **No token/logit claim.** M2 runs one real kernel as a session proof; the token path, boundary
//!   capture and criterion 1 are M3 work (design 3.2b, 4.3, 7).
//!
//! `DeviceEngine<S>` is `Send + Sync` whenever `S: DeviceSession` is (the trait requires it), and
//! a compile-time assertion below pins that for both the fake session and, under the feature, the
//! real one. The api seam requires the bound.

use std::sync::Arc;

use futures::stream::{self, BoxStream, StreamExt};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, InferenceEngine,
    MissingCapability,
};

use crate::session::{DeviceSession, ExecutionEvidence, FakeSession, SessionError};

/// The error code a non-greedy request is refused with.
///
/// Deliberately not `unsupported_capability`: that code is the typed
/// [`MissingCapability`] vocabulary for a *channel* the engine lacks, while non-greedy sampling is
/// a configuration this engine does not implement. The HTTP layer maps an
/// `EngineEvent::Error` to a server error, so the code is machine-readable on the stream and in
/// the non-streaming 503 body.
pub const NON_GREEDY_CODE: &str = "unsupported_sampling";

/// The sampling rule this engine implements: temperature exactly zero is greedy.
///
/// `top_p` is irrelevant at temperature zero (the argmax is the whole distribution), so it is not
/// part of this predicate. A future sampler widens this function, not the engine's contract.
#[must_use]
pub fn is_greedy(request: &EngineRequest) -> bool {
    request.temperature == 0.0
}

/// A request `DeviceEngine` refuses before it touches the reference engine.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceRefusal {
    /// The request carries an image part and the engine has no vision path.
    Vision(MissingCapability),
    /// The request asks for non-greedy sampling and this engine has no sampler.
    NonGreedy {
        /// The temperature the request asked for.
        temperature: f32,
    },
}

impl DeviceRefusal {
    /// The machine-readable code carried by the refusal event.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Vision(missing) => missing.code(),
            Self::NonGreedy { .. } => NON_GREEDY_CODE,
        }
    }

    /// The human-readable message carried by the refusal event.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::Vision(missing) => missing.to_string(),
            Self::NonGreedy { temperature } => format!(
                "non-greedy sampling is not implemented by the device engine: temperature must be 0.0 (greedy), got {temperature}; the chain is greedy-only (design 4.2)"
            ),
        }
    }

    /// The single terminal event a refusal produces: no engine work happens behind it.
    #[must_use]
    pub fn event(&self) -> EngineEvent {
        EngineEvent::Error {
            message: self.message(),
            code: self.code().to_owned(),
        }
    }
}

/// What the device engine can serve and what it actually is.
///
/// [`Capabilities`] has no vision field on purpose (the image path is absent, not a capability
/// that defaults to false; D18/D27). This report is the device path's explicit surface:
/// `vision: false`, whether the session is fake, what a real session executed (kernel and cubin
/// sha256, or nothing), and whether the forward chain ran. The cli prints
/// [`CapabilityReport::startup_line`] when the device selection is made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityReport {
    /// Always false: there is no vision forward, no aligner and no visual-token id path.
    pub vision: bool,
    /// Whether the delegated reference engine advertises tool calls.
    pub tools: bool,
    /// Whether the delegated reference engine advertises a reasoning channel.
    pub thinking: bool,
    /// Whether the delegated reference engine honours `reasoning_effort`.
    pub reasoning_effort: bool,
    /// True when the engine is running on [`FakeSession`]; false for a real session.
    pub fake_session: bool,
    /// True only when the session handed back [`ExecutionEvidence`]: something really ran.
    pub device_execution: bool,
    /// Always false in M2: the forward chain has not run, whatever the session executed.
    pub chain_execution: bool,
    /// What the session executed, or `None` when it executed nothing.
    pub evidence: Option<ExecutionEvidence>,
}

impl CapabilityReport {
    /// Derives the report from the session that is actually held, never from a caller's claim.
    ///
    /// `fake_session` comes from [`DeviceSession::fake`]. A fake session cannot have executed
    /// anything, so its [`DeviceSession::execution`] is ignored rather than trusted: a
    /// contradictory session that reports both `fake() = true` and some evidence yields
    /// `fake_session = true, device_execution = false, evidence = None`. For a real session,
    /// `device_execution` is exactly `execution().is_some()`; no evidence means nothing ran.
    #[must_use]
    pub fn for_session<S: DeviceSession>(session: &S, reference: Capabilities) -> Self {
        let fake_session = session.fake();
        let evidence = if fake_session {
            None
        } else {
            session.execution()
        };
        Self {
            vision: false,
            tools: reference.tools,
            thinking: reference.thinking,
            reasoning_effort: reference.reasoning_effort,
            fake_session,
            device_execution: evidence.is_some(),
            chain_execution: false,
            evidence,
        }
    }

    /// The startup line the cli prints before binding, and the same facts the module docs state.
    ///
    /// The label branches on [`Self::fake_session`], not on the presence of evidence, so a
    /// contradictory report cannot print the M2 label for a fake session:
    ///
    /// - fake session: the M1 skeleton, `device_execution=false`, no kernel launch, first real
    ///   session is M2;
    /// - real session with evidence: the M2 session/launch proof with the kernel id, cubin sha256
    ///   and size, launches and outputs sha, and no token/logits or chain claim;
    /// - real session without evidence: says exactly that - the driver may be open but nothing has
    ///   run, so there is no execution claim to print yet.
    ///
    /// All three document the non-greedy refusal behaviour, because the api seam turns it into a
    /// 503 body or an SSE error event rather than a typed 400.
    #[must_use]
    pub fn startup_line(&self) -> String {
        let non_greedy = "non-greedy requests refused (code unsupported_sampling: HTTP 503 non-streaming, SSE error event streaming)";
        if self.fake_session {
            return format!(
                "engine device (M1 fake-session skeleton; fake_session={}, device_execution=false, chain=false, vision={}); no device claim, no kernel launch; {non_greedy}; the first real session is M2",
                self.fake_session, self.vision
            );
        }
        match &self.evidence {
            None => format!(
                "engine device (real device session, no execution evidence; fake_session=false, device_execution=false, chain=false, vision={}); the session has not reported a completed kernel, so no token/logits claim and no chain; {non_greedy}",
                self.vision
            ),
            Some(evidence) => format!(
                "engine device (M2 session/launch proof; fake_session=false, device_execution=true, chain=false, kernel={}, cubin_sha256={}, cubin_bytes={}, launches={}, tokens={}, outputs_sha256={}); one real kernel, no token/logits claim, no chain; {non_greedy}",
                evidence.kernel,
                evidence.cubin_sha256,
                evidence.cubin_bytes,
                evidence.launches,
                evidence.tokens,
                evidence.outputs_sha256
            ),
        }
    }
}

/// The device engine: a capability boundary, a session, and a delegation to the reference pass.
///
/// It holds the session and the reference engine that does the actual generation. It does not hold
/// weights, a driver, a cubin or a device buffer: a real session's driver lives on its worker thread
/// and the engine keeps only the [`ExecutionEvidence`].
pub struct DeviceEngine<S: DeviceSession> {
    session: S,
    reference: Arc<dyn InferenceEngine>,
    report: CapabilityReport,
}

// Compile-time proof of the api seam's requirement: InferenceEngine: Send + Sync, so the engine
// that implements it must be too. This is type-checked at compile time even though the closure is
// never called; it fails the build (not a test run) if the bound stops holding.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DeviceEngine<FakeSession>>();
    #[cfg(feature = "cuda")]
    assert_send_sync::<DeviceEngine<crate::CudaSession>>();
};

impl<S: DeviceSession> DeviceEngine<S> {
    /// Opens the session and builds the engine.
    ///
    /// # Errors
    /// [`SessionError`] when the session cannot be opened: [`FakeSession`] always succeeds,
    /// while the M2 CUDA session can fail on the cubin pin, the memory floor or the driver. The cli
    /// prints the message instead of starting a server that cannot serve.
    pub fn open(session: S, reference: Arc<dyn InferenceEngine>) -> Result<Self, SessionError> {
        session.open()?;
        let report = CapabilityReport::for_session(&session, reference.capabilities());
        Ok(Self {
            session,
            reference,
            report,
        })
    }

    /// What this engine can serve, what it is, and what it actually executed.
    #[must_use]
    pub fn report(&self) -> CapabilityReport {
        self.report.clone()
    }

    /// The session's short name (`"fake"` for the M1 session, `"cuda"` for M2's).
    #[must_use]
    pub fn session_name(&self) -> &'static str {
        self.session.name()
    }

    /// Whether the session is M1's fake session.
    #[must_use]
    pub fn fake_session(&self) -> bool {
        self.session.fake()
    }

    /// How many requests the session has observed.
    #[must_use]
    pub fn session_requests(&self) -> usize {
        self.session.requests()
    }

    /// Refuses a request this engine cannot serve, before any delegation.
    ///
    /// Checked in this order: an image part (the typed missing-capability path) first, then
    /// non-greedy sampling. The router calls `validate_request` before `generate`, so the image
    /// case is normally refused at the HTTP boundary; `generate` checks again for a direct caller
    /// that bypasses the router.
    ///
    /// # Errors
    /// [`DeviceRefusal::Vision`] or [`DeviceRefusal::NonGreedy`].
    pub fn check_request(&self, request: &EngineRequest) -> Result<(), DeviceRefusal> {
        if !request.images.is_empty() {
            return Err(DeviceRefusal::Vision(MissingCapability::vision()));
        }
        if !is_greedy(request) {
            return Err(DeviceRefusal::NonGreedy {
                temperature: request.temperature,
            });
        }
        Ok(())
    }
}

impl<S: DeviceSession + 'static> InferenceEngine for DeviceEngine<S> {
    fn model_id(&self) -> &str {
        self.reference.model_id()
    }

    fn generate(&self, request: EngineRequest) -> BoxStream<'static, EngineEvent> {
        if let Err(refusal) = self.check_request(&request) {
            return stream::iter([refusal.event()]).boxed();
        }
        self.session.record_request();
        // The reference engine owns the actual forward pass; the device session contributes only
        // the M2 launch evidence and no token arithmetic.
        self.reference.generate(request)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.reference.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        // Decision, deliberately: the trait set stays the conservative default. The api seam has
        // no vision channel (D27/!68) and the delegated reference engine advertises no optional
        // channel, so claiming one here would be a claim nothing serves. The explicit device
        // surface is report(), including what a real session executed.
        Capabilities::default()
    }

    fn validate_request(&self, request: &EngineRequest) -> Result<(), MissingCapability> {
        if !request.images.is_empty() {
            return Err(MissingCapability::vision());
        }
        Ok(())
    }
}
