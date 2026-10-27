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

//! Device sessions: everything `DeviceEngine` may ask a device for.
//!
//! M1 has [`FakeSession`]: it records requests and produces no device result, so the M1 contract runs
//! on a CPU-only CI host. M2 adds `cuda::CudaSession` behind the `cuda` feature: it opens the
//! driver, loads the pinned router cubin, allocates the router buffers and executes exactly one real
//! kernel launch, then reports the evidence and nothing else (design
//! `docs/archive/build/device-forward-chain.md` 7 M2). Neither session claims a forward chain, a token or a
//! logit; M2 is a session/launch proof.
//!
//! The trait is the seam the engine holds. It is `Send + Sync`, and the engine is `Send + Sync`
//! because a session may only hand back plain evidence, never a device handle.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A failure to open a device session.
///
/// [`FakeSession`] never returns one; the M2 CUDA session returns the driver's or the pin check's
/// message here, and the cli prints it instead of starting a server that cannot serve.
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

/// What a real session actually executed.
///
/// This is evidence, not a claim about arithmetic: M2 records the one router launch it ran, the
/// pinned cubin it ran it from, and the hash of the bytes it read back. The engine repeats these
/// fields in its [`crate::CapabilityReport`] so the report can be honest about what ran, and
/// [`ExecutionEvidence::outputs_sha256`] is a read-back proof, not a comparison against a golden.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionEvidence {
    /// The kernel that ran, by its ABI name (M2: the device MoE router).
    pub kernel: &'static str,
    /// SHA-256 of the cubin image that was loaded, lower-case hex.
    pub cubin_sha256: String,
    /// Size of that cubin in bytes.
    pub cubin_bytes: usize,
    /// How many launches this session ran (M2: one).
    pub launches: usize,
    /// Tokens in the launch.
    pub tokens: u32,
    /// SHA-256 of the ids, weights and scores bytes read back after the launch, lower-case hex.
    ///
    /// The three blobs are hashed as one plain concatenation in the order ids, weights, scores:
    /// no length prefixes, no separators. That is what the read-back proof covers; it is a hash of
    /// the bytes the device produced, not a comparison against a golden.
    pub outputs_sha256: String,
}

/// A device session: the seam between `DeviceEngine` and whatever drives the device.
///
/// The trait is `Send + Sync` because [`crate::DeviceEngine`] is (the api engine trait requires
/// it); a session that is not thread-safe cannot be held by the engine. M2's `cuda::CudaSession`
/// satisfies that by doing all driver work on a one-shot worker thread and keeping only the
/// [`ExecutionEvidence`] on the engine side.
pub trait DeviceSession: Send + Sync {
    /// Short stable name, used in the engine's report and error messages.
    fn name(&self) -> &'static str;

    /// Whether this is M1's fake session rather than a real device.
    fn fake(&self) -> bool;

    /// Opens the session.
    ///
    /// # Errors
    /// [`SessionError`] when the device, driver, cubin or pin check fails. M2's CUDA session
    /// returns the pin check or driver message here.
    fn open(&self) -> Result<(), SessionError>;

    /// Records one engine request. Called by `DeviceEngine`, not by callers.
    fn record_request(&self);

    /// How many engine requests this session has observed.
    fn requests(&self) -> usize;

    /// What this session actually executed, or `None` when it has executed nothing.
    ///
    /// The default is `None`, which is the truth for [`FakeSession`]. `DeviceEngine` derives
    /// `device_execution` from this, so a session cannot claim execution the engine cannot see.
    fn execution(&self) -> Option<ExecutionEvidence> {
        None
    }
}

/// The M1 fake session: records requests, produces no device result.
///
/// It is deliberately free of CUDA, checkpoints and cluster access, so the M1 contract is testable
/// on a CPU-only CI host: a device that is not there cannot make the test flaky. It is still the
/// CI default in M2; the real session needs the `cuda` feature and an explicit cubin.
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

/// The M2 real session: one pinned router launch, then evidence.
///
/// This module is only compiled with the `cuda` feature, so the default build and CI host have no
/// driver path at all. `Cuda` owns raw pointers and is neither `Send` nor `Sync`, so it cannot
/// live in `CudaSession` (which the engine holds across threads): `CudaSession::open` spawns a
/// one-shot worker thread that owns the driver, runs the launch, drops the context and returns the
/// evidence. The engine then holds only [`ExecutionEvidence`], which is plain data.
///
/// M2 is a session/launch proof, not a forward chain: one `moe_router_sm121` launch over zero-filled
/// operands with the ids/weights/scores read back and hashed. It makes no token, logit or
/// correctness claim, and the chain is still absent (design 7 M2).
///
/// The worker is deliberately one-shot: M2 only has to prove that this process can open the
/// driver, load the pinned cubin, execute a kernel and read the result back. M3's chain session
/// will keep a worker alive across steps instead; that change stays behind this same
/// [`DeviceSession`] seam and does not reach the engine.
#[cfg(feature = "cuda")]
pub mod cuda {
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use sha2::{Digest, Sha256};
    use sparkflash_kernels::cuda_driver::{Cuda, mem_available_kb};
    use sparkflash_kernels::router_abi;
    use sparkflash_kernels::router_launch::{self, RouterLaunchParams};

    use super::{DeviceSession, ExecutionEvidence, SessionError};

    /// Tokens in the one launch: exactly one router block, which the kernel's grid geometry fixes.
    const TOKENS: u32 = 8;

    /// The guard before allocating: the session allocates a few MB, and this floor refuses to open
    /// a context on a node that is already out of memory. It is a guard, not a budget.
    const MEMORY_FLOOR_KB: u64 = 1024 * 1024;

    /// The M2 real session: opens the CUDA driver and runs one pinned router launch.
    ///
    /// Construction is cheap and touches no device; the driver, cubin and launch all happen inside
    /// [`DeviceSession::open`], which blocks until the one-shot worker thread has finished.
    pub struct CudaSession {
        cubin: PathBuf,
        evidence: OnceLock<Result<ExecutionEvidence, String>>,
        requests: AtomicUsize,
    }

    impl CudaSession {
        /// Creates a session that will load `cubin` when opened.
        #[must_use]
        pub fn new(cubin: impl Into<PathBuf>) -> Self {
            Self {
                cubin: cubin.into(),
                evidence: OnceLock::new(),
                requests: AtomicUsize::new(0),
            }
        }

        /// The cubin path this session was built for.
        #[must_use]
        pub fn cubin(&self) -> &Path {
            &self.cubin
        }
    }

    impl DeviceSession for CudaSession {
        fn name(&self) -> &'static str {
            "cuda"
        }

        fn fake(&self) -> bool {
            false
        }

        fn open(&self) -> Result<(), SessionError> {
            if self.evidence.get().is_some() {
                return Err(SessionError::new("the CUDA session is already open"));
            }
            let path = self.cubin.clone();
            let outcome = std::thread::spawn(move || prove(&path))
                .join()
                .map_err(|_| SessionError::new("the CUDA session worker thread panicked"))?;
            self.evidence
                .set(outcome)
                .map_err(|_| SessionError::new("the CUDA session is already open"))?;
            match self.evidence.get() {
                Some(Ok(_)) => Ok(()),
                Some(Err(detail)) => Err(SessionError::new(detail.clone())),
                None => unreachable!("the evidence was just set"),
            }
        }

        fn record_request(&self) {
            self.requests.fetch_add(1, Ordering::Relaxed);
        }

        fn requests(&self) -> usize {
            self.requests.load(Ordering::Relaxed)
        }

        fn execution(&self) -> Option<ExecutionEvidence> {
            self.evidence
                .get()
                .and_then(|outcome| outcome.as_ref().ok())
                .cloned()
        }
    }

    /// The one-shot worker body: verify the pin, open the driver, run one router launch, read back.
    ///
    /// Everything here runs on the worker thread that owns `Cuda`. The function returns plain
    /// [`ExecutionEvidence`] so nothing device-shaped escapes the thread.
    fn prove(cubin_path: &Path) -> Result<ExecutionEvidence, String> {
        let cubin = std::fs::read(cubin_path)
            .map_err(|error| format!("{}: {error}", cubin_path.display()))?;
        let digest: [u8; 32] = Sha256::digest(&cubin).into();
        let sha = hex(&digest);
        if cubin.len() != router_abi::CUBIN_BYTES || sha != router_abi::CUBIN_SHA256 {
            return Err(format!(
                "the cubin is not the pinned {}: expected {} B / {}, got {} B / {sha}",
                router_abi::ROUTER_KERNEL,
                router_abi::CUBIN_BYTES,
                router_abi::CUBIN_SHA256,
                cubin.len()
            ));
        }
        let available_kb = mem_available_kb()
            .ok_or_else(|| "cannot read MemAvailable from /proc/meminfo".to_owned())?;
        if available_kb < MEMORY_FLOOR_KB {
            return Err(format!(
                "MemAvailable is {available_kb} kB, below the {MEMORY_FLOOR_KB} kB floor: refusing to open a CUDA session"
            ));
        }

        let cuda = Cuda::load()?;
        let x = vec![0u8; TOKENS as usize * router_abi::HIDDEN as usize * 2];
        let gate = vec![0u8; router_abi::N_ROUTED as usize * router_abi::HIDDEN as usize * 2];
        let bias = vec![0u8; router_abi::N_ROUTED as usize * 4];
        let ids_len = TOKENS as usize * router_abi::TOP_K as usize * 4;
        let weights_len = ids_len;
        let scores_len = TOKENS as usize * router_abi::N_ROUTED as usize * 4;

        let mut total = 0usize;
        let dx = cuda.alloc(x.len(), &mut total)?;
        let dgate = cuda.alloc(gate.len(), &mut total)?;
        let dbias = cuda.alloc(bias.len(), &mut total)?;
        let dids = cuda.alloc(ids_len, &mut total)?;
        let dweights = cuda.alloc(weights_len, &mut total)?;
        let dscores = cuda.alloc(scores_len, &mut total)?;
        let allocations = [dx, dgate, dbias, dids, dweights, dscores];

        cuda.copy_htod(&dx, &x)?;
        cuda.copy_htod(&dgate, &gate)?;
        cuda.copy_htod(&dbias, &bias)?;

        let (module, kernel, topk_module, topk) = router_launch::load(&cuda, &cubin)?;
        let params = RouterLaunchParams {
            x: dx,
            gate: dgate,
            bias: dbias,
            ids: dids,
            weights: dweights,
            scores: dscores,
            tokens: TOKENS,
            trace_ring: 0,
            layer_id: 0,
            step_seq: 0,
        };
        // Two launches (scores, then top-6); the typed helper synchronises each, so a returned Ok
        // means both kernels completed.
        router_launch::launch(&cuda, kernel, topk, &params)?;

        let mut ids = vec![0u8; ids_len];
        let mut weights = vec![0u8; weights_len];
        let mut scores = vec![0u8; scores_len];
        cuda.copy_dtoh(&mut ids, &dids)?;
        cuda.copy_dtoh(&mut weights, &dweights)?;
        cuda.copy_dtoh(&mut scores, &dscores)?;

        let outputs = outputs_digest(&ids, &weights, &scores);

        cuda.free(module, &allocations);
        cuda.free(topk_module, &[]);
        drop(cuda);

        Ok(ExecutionEvidence {
            kernel: router_abi::ROUTER_KERNEL,
            cubin_sha256: sha,
            cubin_bytes: cubin.len(),
            launches: 2,
            tokens: TOKENS,
            outputs_sha256: hex(&outputs),
        })
    }

    /// SHA-256 of the three read-back blobs as one plain concatenation.
    ///
    /// Plain concatenation, not a framed encoding: the ids, weights and scores are each contiguous
    /// byte buffers whose lengths are fixed by the router ABI, so a length prefix would add a
    /// framing convention with no information and a second thing to test. The report is the only
    /// place the digest is described, and it says plain concatenation.
    fn outputs_digest(ids: &[u8], weights: &[u8], scores: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(ids);
        hasher.update(weights);
        hasher.update(scores);
        hasher.finalize().into()
    }

    /// Lower-case hex, the spelling the pins and the report use.
    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_missing_cubin_is_refused_before_the_driver_is_opened() {
            // No GPU is touched: the cubin read is the first step of `prove`, so this test is
            // safe on a CPU-only CI host even with the cuda feature compiled in.
            let session = CudaSession::new("/definitely/not/a/router.cubin");
            let error = session.open().expect_err("there is no cubin there");
            assert!(error.detail().contains("router.cubin"), "{error}");
            assert!(session.execution().is_none());
            assert!(!session.fake());
        }

        #[test]
        fn a_wrong_cubin_is_refused_by_the_pin_before_the_driver_is_opened() {
            // Zeros padded to the pinned size have the pinned length but not the pinned sha256. The
            // pin check
            // in `prove` runs before `Cuda::load`, so this stays a CPU-only test and a changed
            // cubin can never reach the driver.
            let dir = tempfile::tempdir().expect("a temporary directory");
            let path = dir.path().join("wrong.cubin");
            std::fs::write(&path, vec![0u8; router_abi::CUBIN_BYTES])
                .expect("the wrong cubin is writable");
            let session = CudaSession::new(&path);
            let error = session.open().expect_err("the sha does not match the pin");
            assert!(error.detail().contains("not the pinned"), "{error}");
            assert!(
                error
                    .detail()
                    .contains(&router_abi::CUBIN_BYTES.to_string()),
                "{error}"
            );
            assert!(!error.detail().contains("MemAvailable"), "{error}");
            assert!(session.execution().is_none());
        }

        #[test]
        fn the_outputs_digest_is_a_plain_concatenation() {
            // Pins the framing: no length prefixes, no separators, ids then weights then scores.
            // The expected value is sha256("ids" + "weights" + "scores") computed independently.
            let digest = outputs_digest(b"ids", b"weights", b"scores");
            assert_eq!(
                hex(&digest),
                "0a7a874e574d1a92decca6079c7c581ae145ce928417dc54a0f8b07f8cbff819"
            );
            // A different split of the same bytes hashes the same: that is concatenation.
            assert_eq!(outputs_digest(b"id", b"sw", b"eightsscores"), digest);
            assert_eq!(outputs_digest(b"idsw", b"", b"eightsscores"), digest);
        }
    }
}
