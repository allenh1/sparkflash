//! M1: `DeviceEngine`, the fake-session engine behind the serving capability flag.
//!
//! This is design `docs/build/device-forward-chain.md` 7 M1 and nothing later:
//!
//! - **Seam only.** It implements [`sparkflash_api::engine::InferenceEngine`] over a
//!   [`DeviceSession`] and delegates `generate` to a reference `Arc<dyn InferenceEngine>`.
//!   The reference driver is bin-only, so the cli supplies it (design 1.3); this crate cannot name
//!   `ReferenceDriver`.
//! - **Fake session.** [`FakeSession`] records requests and produces no device result; the engine
//!   never launches a kernel in M1.
//! - **Capability report.** [`DeviceEngine::report`] says `vision: false` and
//!   `fake_session: true`; the api seam's [`Capabilities`] has no vision channel (D27/!68), so the
//!   report is how the device path states the same fact. Images are refused through the typed
//!   [`MissingCapability::vision`] path before any delegation.
//! - **Non-greedy refused.** A request with `temperature != 0.0` is refused with
//!   [`NON_GREEDY_CODE`] before the reference engine is called; M1 has no sampler and the first
//!   chain is greedy-only (design 1.3, 4.2).
//! - **No device claim.** The token path, boundary capture and criterion 1 are M2/M3 work (design
//!   3.2b, 4.3, 7); this engine reports `device_execution: false` and the cli selection defaults to
//!   `reference`.
//!
//! `DeviceEngine<S>` is `Send + Sync` whenever `S: DeviceSession` is (the trait requires it), and
//! a compile-time assertion below pins that for the M1 session. The api seam requires the bound.

use std::sync::Arc;

use futures::stream::{self, BoxStream, StreamExt};
use sparkflash_api::engine::{
    Capabilities, ClusterStatsSnapshot, EngineEvent, EngineRequest, InferenceEngine,
    MissingCapability,
};

use crate::session::{DeviceSession, FakeSession, SessionError};

/// The error code a non-greedy request is refused with.
///
/// Deliberately not `unsupported_capability`: that code is the typed
/// [`MissingCapability`] vocabulary for a *channel* the engine lacks, while non-greedy sampling is
/// a configuration this skeleton does not implement. The HTTP layer turns an `EngineEvent::Error`
/// into a server error event, so the code is machine-readable on the stream.
pub const NON_GREEDY_CODE: &str = "unsupported_sampling";

/// The sampling rule M1 implements: temperature exactly zero is greedy.
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
    /// The request asks for non-greedy sampling and M1 has no sampler.
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
                "non-greedy sampling is not implemented by the M1 device engine: temperature must be 0.0 (greedy), got {temperature}; the first real chain is greedy-only (design 4.2)"
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

/// What the M1 device engine can serve and what it actually is.
///
/// [`Capabilities`] has no vision field on purpose (the image path is absent, not a capability
/// that defaults to false; D18/D27). This report is the device path's explicit statement:
/// `vision: false`, and `fake_session: true` so a reader cannot mistake M1 for device execution.
/// The cli prints it when `--engine device` is selected, and the module docs say the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityReport {
    /// Always false in M1: there is no vision forward, no aligner and no visual-token id path.
    pub vision: bool,
    /// Whether the delegated reference engine advertises tool calls.
    pub tools: bool,
    /// Whether the delegated reference engine advertises a reasoning channel.
    pub thinking: bool,
    /// Whether the delegated reference engine honours `reasoning_effort`.
    pub reasoning_effort: bool,
    /// True when the engine is running on a fake session (M1). False for a real session (M2+).
    pub fake_session: bool,
    /// Always false in M1: no kernel is launched and no device result is produced.
    pub device_execution: bool,
}

impl CapabilityReport {
    /// The report for M1's fake session: `vision: false`, `fake_session: true`,
    /// `device_execution: false`.
    #[must_use]
    pub fn m1_fake_session(reference: Capabilities) -> Self {
        Self {
            vision: false,
            tools: reference.tools,
            thinking: reference.thinking,
            reasoning_effort: reference.reasoning_effort,
            fake_session: true,
            device_execution: false,
        }
    }
}

/// The M1 device engine: a capability boundary and a delegation, not a chain.
///
/// It holds the session and the reference engine that does the actual work. It does not hold
/// weights, a driver, a cubin or a device buffer; there is nothing device-shaped in it yet, which
/// is the point of M1 (design 7).
pub struct DeviceEngine<S: DeviceSession> {
    session: S,
    reference: Arc<dyn InferenceEngine>,
    report: CapabilityReport,
}

// Compile-time proof of the api seam's requirement: `InferenceEngine: Send + Sync`, so the engine
// that implements it must be too. This is type-checked at compile time even though the closure is
// never called; it fails the build (not a test run) if the bound stops holding.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DeviceEngine<FakeSession>>();
};

impl<S: DeviceSession> DeviceEngine<S> {
    /// Opens the session and builds the engine.
    ///
    /// # Errors
    /// [`SessionError`] when the session cannot be opened. M1's [`FakeSession`] always succeeds;
    /// the M2 real session may fail here, and the cli reports the message instead of starting a
    /// server that cannot serve.
    pub fn open(session: S, reference: Arc<dyn InferenceEngine>) -> Result<Self, SessionError> {
        session.open()?;
        let report = CapabilityReport::m1_fake_session(reference.capabilities());
        Ok(Self {
            session,
            reference,
            report,
        })
    }

    /// What this engine can serve, and whether it is a fake session.
    #[must_use]
    pub fn report(&self) -> CapabilityReport {
        self.report
    }

    /// The session's short name (`"fake"` in M1).
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

    /// Refuses a request this skeleton cannot serve, before any delegation.
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
        // The reference engine owns the actual forward pass; M1 adds no arithmetic.
        self.reference.generate(request)
    }

    fn cluster_stats(&self) -> ClusterStatsSnapshot {
        self.reference.cluster_stats()
    }

    fn capabilities(&self) -> Capabilities {
        // The api seam's set: the reference driver declares no optional channel, and there is no
        // vision field to declare false (D27/!68). `report()` carries the device path's explicit
        // `vision: false` statement.
        Capabilities::default()
    }

    fn validate_request(&self, request: &EngineRequest) -> Result<(), MissingCapability> {
        if !request.images.is_empty() {
            return Err(MissingCapability::vision());
        }
        Ok(())
    }
}
