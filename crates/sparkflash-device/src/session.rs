//! Device sessions: everything `DeviceEngine` may ask a device for.
//!
//! M1 has exactly one implementation, [`FakeSession`]. It records how many engine requests were
//! handed to the reference engine and reports `fake() == true`; it opens no driver, loads no
//! cubin, allocates no device pointer and talks to no node (design
//! `docs/build/device-forward-chain.md` 7 M1). The trait is the M2 seam: the real session will
//! load the driver and stage weights there, and the engine above it does not change.
//!
//! Nothing in this module is unsafe, links libcuda, or spawns a thread.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A failure to open a device session.
///
/// M1's [`FakeSession`] never returns one; the type exists so the M2 real session can fail at
/// `open` and surface a message instead of panicking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionError {
    detail: String,
}

impl SessionError {
    /// Builds an error from the message the caller should print.
    #[must_use]
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }

    /// The human-readable detail.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for SessionError {}

/// A device session: the seam between `DeviceEngine` and whatever drives the device.
///
/// M1 implements this with [`FakeSession`] only. The trait is `Send + Sync` because
/// [`crate::DeviceEngine`] is (and the api engine trait requires it); a session that is not
/// thread-safe cannot be held by the engine.
pub trait DeviceSession: Send + Sync {
    /// Short stable name, used in the engine's report and error messages.
    fn name(&self) -> &'static str;

    /// Whether this is M1's fake session rather than a real device.
    fn fake(&self) -> bool;

    /// Opens the session.
    ///
    /// # Errors
    /// [`SessionError`] when the device, driver or cubins cannot be opened. M1's fake session
    /// always succeeds; the M2 session may fail here.
    fn open(&self) -> Result<(), SessionError>;

    /// Records one engine request. Called by `DeviceEngine`, not by callers.
    fn record_request(&self);

    /// How many engine requests this session has observed.
    fn requests(&self) -> usize;
}

/// The M1 fake session: records requests, produces no device result.
///
/// It is deliberately free of CUDA, checkpoints and cluster access, so the M1 contract is
/// testable on a CPU-only CI host: a device that is not there cannot make the test flaky.
#[derive(Debug, Default)]
pub struct FakeSession {
    requests: AtomicUsize,
}

impl FakeSession {
    /// Creates a fake session with no recorded requests.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl DeviceSession for FakeSession {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn fake(&self) -> bool {
        true
    }

    fn open(&self) -> Result<(), SessionError> {
        Ok(())
    }

    fn record_request(&self) {
        self.requests.fetch_add(1, Ordering::Relaxed);
    }

    fn requests(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }
}
