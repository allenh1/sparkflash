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

//! NCCL 2.28+ FFI, loaded at run time with `libc::dlopen`.
//!
//! The host has no `libnccl`: the library exists only inside the deployed `vllm-node:latest`
//! image (measured 2026-10-21 audit: NCCL 2.28.3-1, host `ldconfig -p | grep nccl` empty on all
//! four nodes; report `analysis/rdma-prereq-20261021.md`, blocker B5). This module therefore
//! resolves every entry point at run time and has **no build-time dependency**: a default build
//! links nothing new, and a host without the library gets a typed refusal, not a link error and
//! not a silent fallback.
//!
//! ## What is bound
//!
//! [`NcclApi::load`] resolves the subset of the NCCL 2.x C API the dispatcher needs:
//! `ncclGetVersion`, `ncclGetUniqueId`, `ncclCommInitRankConfig`, `ncclAllReduce`,
//! `ncclAllGather`, `ncclGroupStart`, `ncclGroupEnd`, `ncclCommDestroy`, `ncclGetErrorString`,
//! `ncclCommRegister` and `ncclCommDeregister`. The first two are not in the task's six-function
//! list but are prerequisites for a real communicator: the version probe enforces the 2.28 floor,
//! and a unique id is what bootstraps the four ranks. The registration pair (NCCL 2.19+, so inside
//! the floor) is how the run registers its collective buffers once at open (S2-a). They are
//! resolved as required symbols so a truncated library fails at load with a named symbol rather
//! than at the first collective.
//!
//! ## The ABI choices that matter
//!
//! * `ncclUniqueId` is a 128-byte opaque blob (`char internal[128]` in NCCL's header). It is
//!   mirrored as `[u8; 128]` in a `repr(C)` struct with a compile-time size assert; the byte
//!   type differs from the header's `char` but the size and alignment (1) are identical, which
//!   is all the by-value ABI depends on.
//! * `ncclCommInitRankConfig` takes a pointer to `ncclConfig_t`, whose layout is versioned and
//!   differs between NCCL releases (it carries a `size` and `magic` the callee validates).
//!   This binding passes `NULL`, which NCCL documents as "use the default configuration"; the
//!   knobs we need (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, chunk sizes) are environment variables
//!   that the deployed recipe already sets. Mirroring a versioned struct we cannot exercise on
//!   this host would be guessing at an ABI; passing NULL is the supported, checkable path.
//! * `ncclDataType_t` and `ncclRedOp_t` are C enums (int-sized). Their values are part of the
//!   NCCL ABI and are pinned here by tests: `ncclBfloat16 = 9`, `ncclSum = 0`. The named
//!   aliases (`ncclHalf = 6`, `ncclFloat = 7`) are the same values.
//! * `cudaStream_t` is an opaque pointer. This binding passes the caller's stream through
//!   unchanged and treats NULL as the default stream; it never creates a stream.
//!
//! ## Why the function pointers are raw
//!
//! ~~~ignore
//! type Fn = unsafe extern "C" fn(...);
//! ~~~
//! The pointers stay `*mut c_void` in storage and are transmuted to the typed signature at the
//! call site, inside an `unsafe` block. That keeps the loader testable with a fake symbol table:
//! the tests resolve names and assert the missing-symbol path without ever calling through a
//! pointer. No test on this host can call NCCL (there is no GPU and no library), so the tests pin
//! the loader, the ABI constants and the argument validation, and the doc says exactly that.
//!
//! ## Explicit fallback
//!
//! [`NcclApi::load`] returns [`NcclError::LibraryAbsent`] naming both sonames and the vLLM image.
//! The dispatcher must treat that as "NCCL unavailable" and refuse a >2 MiB collective or chunk it,
//! never silently substitute the one-shot path; see [`crate::dispatch::DispatchPolicy::route_with`].
//!
//! ## Run-boundary failure detection
//!
//! `ncclCommGetAsyncError` is the one entry point this binding treats as optional. A library that
//! exports it gets a real failure check ([`NcclApi::check_async_failure`]): the query's own failure
//! is [`NcclError::Call`], `ncclSuccess`/`ncclInProgress` are "no failure", and any other
//! communicator status is [`NcclError::AsyncFailure`]. A library without the symbol loads and runs;
//! [`NcclApi::has_async_failure_check`] then reports the check as unavailable (a documented no-op,
//! not an error) and the port's step sync is the only detection. The distinction matters because
//! the run record must not claim a check that could not be performed.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fmt;
use std::ptr;

/// Library names tried in order. `.so.2` is NCCL's SONAME for the 2.x ABI.
pub const NCCL_LIBRARY_NAMES: [&str; 2] = ["libnccl.so.2", "libnccl.so"];

/// Smallest acceptable `ncclGetVersion` code: NCCL 2.28.0, encoded as `major*10000 + minor*100 + patch`.
pub const NCCL_MIN_VERSION_CODE: c_int = 22_800;

/// Bytes in `ncclUniqueId` (`char internal[128]` in NCCL's public header).
pub const NCCL_UNIQUE_ID_BYTES: usize = 128;

/// The deployed image NCCL was observed in, used in refusal messages (measured 2026-10-21).
pub const NCCL_OBSERVED_IMAGE: &str = "vllm-node:latest";

/// NCCL entry points the dispatcher requires, in resolution order.
///
/// [`ASYNC_ERROR_SYMBOL`] is resolved separately and is optional: a library that predates it loads
/// and runs, and the run boundary then reports the failure check as unavailable
/// ([`NcclApi::has_async_failure_check`]) instead of treating the absence as an error.
pub const REQUIRED_SYMBOLS: [&str; 11] = [
    "ncclGetVersion",
    "ncclGetUniqueId",
    "ncclCommInitRankConfig",
    "ncclAllReduce",
    "ncclAllGather",
    "ncclGroupStart",
    "ncclGroupEnd",
    "ncclCommDestroy",
    "ncclGetErrorString",
    "ncclCommRegister",
    "ncclCommDeregister",
];

/// The NCCL environment variables that decide which transport a run's collectives take.
///
/// NCCL reads them itself at communicator init and this binding never sets them; the run header and
/// the bench JSON record each one's value (or its absence) so a row states the fabric configuration
/// it ran with (S2-a), instead of a later reader inferring it from a launch script. The first five
/// are the issue's list (`NCCL_IB_HCA`, the GDR level, `NCCL_ALGO`/`NCCL_PROTO`); the rest are the
/// knobs the deployed recipe sets (fabric.md) and the registration switch.
pub const TRANSPORT_ENV: [&str; 9] = [
    "NCCL_IB_HCA",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_NET_GDR_READ",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NCCL_IB_GID_INDEX",
    "NCCL_SOCKET_IFNAME",
    "NCCL_DMABUF_ENABLE",
    "NCCL_LOCAL_REGISTER",
];

/// The optional run-boundary failure-detection entry point.
///
/// Resolved when present, never required: see [`NcclApi::check_async_failure`] for what a missing
/// symbol means (the port's step sync remains the only detection) and for the typed failure it
/// reports when the symbol is present.
pub const ASYNC_ERROR_SYMBOL: &str = "ncclCommGetAsyncError";

/// The opaque `ncclComm_t` handle.
pub type NcclCommHandle = *mut c_void;

/// The opaque `cudaStream_t` handle. NULL is CUDA's default stream.
pub type CudaStream = *mut c_void;

/// NCCL's success code.
pub const NCCL_SUCCESS: c_int = 0;

/// `ncclInProgress`: an asynchronous operation is still running.
///
/// `ncclCommGetAsyncError` reports this as the communicator's status while work is in flight; it is
/// not a failure and the run boundary must not treat it as one. The value is the NCCL ABI's and is
/// pinned by a test.
pub const NCCL_IN_PROGRESS: c_int = 7;

/// Encodes an NCCL version the way `ncclGetVersion` reports it.
#[must_use]
pub const fn nccl_version_code(major: c_int, minor: c_int, patch: c_int) -> c_int {
    major * 10_000 + minor * 100 + patch
}

/// Mirrors `ncclUniqueId`: an opaque 128-byte value copied verbatim between processes.
///
/// The bytes have no meaning outside NCCL; rank 0 obtains one from [`NcclApi::unique_id`] and the
/// other ranks receive the same 128 bytes through the caller's bootstrap channel (the runner uses
/// a hex string; see [`NcclUniqueId::to_hex`]).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NcclUniqueId {
    bytes: [u8; NCCL_UNIQUE_ID_BYTES],
}

const _: () = assert!(std::mem::size_of::<NcclUniqueId>() == NCCL_UNIQUE_ID_BYTES);
const _: () = assert!(std::mem::align_of::<NcclUniqueId>() == 1);

impl NcclUniqueId {
    /// Wraps 128 opaque bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; NCCL_UNIQUE_ID_BYTES]) -> Self {
        Self { bytes }
    }

    /// The opaque bytes, byte-for-byte.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; NCCL_UNIQUE_ID_BYTES] {
        &self.bytes
    }

    /// Lowercase hex, 256 characters. This is the bootstrap representation the runner passes
    /// between nodes; it round-trips through [`Self::from_hex`].
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut text = String::with_capacity(NCCL_UNIQUE_ID_BYTES * 2);
        for byte in self.bytes {
            text.push(nibble(byte >> 4));
            text.push(nibble(byte & 0x0f));
        }
        text
    }

    /// Parses the 256-character lowercase or uppercase hex form.
    ///
    /// # Errors
    /// Returns [`NcclError::Config`] for the wrong length or a non-hex character.
    pub fn from_hex(text: &str) -> Result<Self, NcclError> {
        let text = text.trim();
        if text.len() != NCCL_UNIQUE_ID_BYTES * 2 {
            return Err(NcclError::Config(format!(
                "ncclUniqueId hex must be {} characters ({} bytes), got {}",
                NCCL_UNIQUE_ID_BYTES * 2,
                NCCL_UNIQUE_ID_BYTES,
                text.len()
            )));
        }
        let mut bytes = [0u8; NCCL_UNIQUE_ID_BYTES];
        let chars: Vec<char> = text.chars().collect();
        for (index, slot) in bytes.iter_mut().enumerate() {
            let high = hex_value(chars[index * 2]).ok_or_else(|| {
                NcclError::Config(format!(
                    "ncclUniqueId hex has a non-hex character at {}",
                    index * 2
                ))
            })?;
            let low = hex_value(chars[index * 2 + 1]).ok_or_else(|| {
                NcclError::Config(format!(
                    "ncclUniqueId hex has a non-hex character at {}",
                    index * 2 + 1
                ))
            })?;
            *slot = (high << 4) | low;
        }
        Ok(Self { bytes })
    }
}

impl fmt::Debug for NcclUniqueId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Do not print the blob; it is a bootstrap secret in the loose sense and unreadable anyway.
        formatter
            .debug_struct("NcclUniqueId")
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

fn nibble(value: u8) -> char {
    match value {
        0..=9 => char::from(b'0' + value),
        _ => char::from(b'a' + (value - 10)),
    }
}

fn hex_value(character: char) -> Option<u8> {
    match character {
        '0'..='9' => Some(character as u8 - b'0'),
        'a'..='f' => Some(character as u8 - b'a' + 10),
        'A'..='F' => Some(character as u8 - b'A' + 10),
        _ => None,
    }
}

/// Mirrors `ncclDataType_t`. Discriminants are the NCCL ABI and are pinned by tests.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclDataType {
    /// `ncclInt8` / `ncclChar`.
    Int8 = 0,
    /// `ncclUint8`.
    Uint8 = 1,
    /// `ncclInt32` / `ncclInt`.
    Int32 = 2,
    /// `ncclUint32`.
    Uint32 = 3,
    /// `ncclInt64`.
    Int64 = 4,
    /// `ncclUint64`.
    Uint64 = 5,
    /// `ncclFloat16` / `ncclHalf`.
    Float16 = 6,
    /// `ncclFloat32` / `ncclFloat`.
    Float32 = 7,
    /// `ncclFloat64` / `ncclDouble`.
    Float64 = 8,
    /// `ncclBfloat16`: the type every SparkFlash collective uses.
    Bfloat16 = 9,
}

impl NcclDataType {
    /// The dtype SparkFlash collectives carry.
    #[must_use]
    pub const fn for_bf16() -> Self {
        Self::Bfloat16
    }

    /// Bytes per element.
    #[must_use]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::Int8 | Self::Uint8 => 1,
            Self::Int32 | Self::Uint32 | Self::Float32 => 4,
            Self::Int64 | Self::Uint64 | Self::Float64 => 8,
            Self::Float16 | Self::Bfloat16 => 2,
        }
    }
}

/// Mirrors `ncclRedOp_t`. Discriminants are the NCCL ABI and are pinned by tests.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclRedOp {
    /// `ncclSum`: the only op the SparkFlash TP seams use.
    Sum = 0,
    /// `ncclProd`.
    Prod = 1,
    /// `ncclMax`.
    Max = 2,
    /// `ncclMin`.
    Min = 3,
    /// `ncclAvg` (NCCL 2.10+); present in 2.28.
    Avg = 4,
}

/// Why the NCCL binding refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NcclError {
    /// `dlopen` failed for every candidate soname.
    LibraryAbsent {
        /// The sonames that were tried.
        tried: Vec<&'static str>,
        /// `dlerror` text from the last attempt.
        detail: String,
    },
    /// The library loaded but a required entry point is missing.
    MissingSymbol {
        /// The loaded library name.
        library: String,
        /// The missing symbol.
        symbol: &'static str,
    },
    /// The selected all-reduce arm cannot run: not built, not implemented, or blocked on an
    /// infrastructure item.
    ArmUnavailable {
        /// The arm's stable label.
        arm: &'static str,
        /// Why it cannot run, in the words the run record reports.
        detail: String,
    },
    /// The library is older than the 2.28 floor.
    Version {
        /// Version code reported by `ncclGetVersion`.
        found: c_int,
        /// The required floor.
        minimum: c_int,
    },
    /// An NCCL call returned a non-zero `ncclResult_t`.
    Call {
        /// The entry point that failed.
        call: &'static str,
        /// The `ncclResult_t` value.
        result: c_int,
        /// `ncclGetErrorString` text when it could be read, otherwise the numeric code.
        detail: String,
    },
    /// The communicator reported an asynchronous operation failure at the run boundary.
    ///
    /// This is failure detection, not completion: the step's own sync has already observed the
    /// collective, and this says the communicator's internal work failed (a transport drop, a peer's
    /// error, or a CUDA fault surfacing through NCCL). The run must abort on every rank, because the
    /// failed collective changed no rank's result consistently.
    AsyncFailure {
        /// The `ncclResult_t` the communicator reported asynchronously.
        result: c_int,
        /// `ncclGetErrorString` text for that result.
        detail: String,
    },
    /// The caller passed an unusable configuration (bad world/rank, malformed unique id).
    Config(String),
    /// The run's rendezvous failed before NCCL was entered: the unique-id exchange over the run's
    /// own TCP transport. This is an abort-before-NCCL error on purpose - no rank has opened a
    /// communicator when it occurs, so no peer can be left inside a collective.
    Bootstrap(String),
    /// The declared S1 host-bridge fallback failed while it was running.
    ///
    /// The bridge is the one permitted device->host / host reduce / host->device path, so its
    /// failure is reported by name and phase rather than flattened into a device-collective error:
    /// the run record must be able to say which half of the only permitted copy failed. A failed
    /// phase is not counted in [`crate::device_reduce::DeviceReduceStats`].
    HostBridge {
        /// Which half failed: `device-to-host`, `host-reduce` or `host-to-device`.
        phase: &'static str,
        /// What the phase reported.
        detail: String,
    },
}

impl fmt::Display for NcclError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryAbsent { tried, detail } => write!(
                formatter,
                "could not load NCCL ({}); host libnccl is absent on the four nodes (measured 2026-10-21) and NCCL 2.28.3 exists only inside the {NCCL_OBSERVED_IMAGE} image, so a host-process NCCL path needs WS-B infra ticket B5: {detail}",
                tried.join(" or ")
            ),
            Self::ArmUnavailable { arm, detail } => {
                write!(
                    formatter,
                    "the {arm} all-reduce arm is unavailable: {detail}"
                )
            }
            Self::MissingSymbol { library, symbol } => write!(
                formatter,
                "{library} is loaded but does not export {symbol}; this binding needs the NCCL 2.28+ C API surface and refuses rather than calling through a null entry point"
            ),
            Self::Version { found, minimum } => write!(
                formatter,
                "NCCL reports version code {found} ({major}.{minor}.{patch}) but this binding requires at least {minimum} ({min_major}.{min_minor}.0)",
                major = found / 10_000,
                minor = (found % 10_000) / 100,
                patch = found % 100,
                min_major = minimum / 10_000,
                min_minor = (minimum % 10_000) / 100,
            ),
            Self::Call {
                call,
                result,
                detail,
            } => write!(
                formatter,
                "{call} failed with ncclResult_t {result}: {detail}"
            ),
            Self::AsyncFailure { result, detail } => write!(
                formatter,
                "the communicator reported an asynchronous failure at the run boundary: ncclResult_t {result}: {detail}; this is failure detection, not completion - the step sync has already observed the collective, and the run must abort on every rank"
            ),
            Self::Config(text) => write!(formatter, "invalid NCCL configuration: {text}"),
            Self::Bootstrap(text) => write!(
                formatter,
                "NCCL bootstrap failed before any communicator was opened: {text}"
            ),
            Self::HostBridge { phase, detail } => write!(
                formatter,
                "the declared S1 host-bridge fallback failed at {phase}: {detail}; a failed bridge is not counted and the run must abort on every rank"
            ),
        }
    }
}

impl std::error::Error for NcclError {}

/// Test-only: tries to open exactly the given names and returns the typed absence when none is
/// there.
///
/// The production loader always uses NCCL_LIBRARY_NAMES; this hook exists so the absence refusal can
/// be pinned on a host that may or may not have libnccl installed.
#[cfg(test)]
pub(crate) fn load_named_for_test(names: &[&'static str]) -> Result<NcclApi, NcclError> {
    let loader: Box<dyn SymbolLoader> = Box::new(DlLoader::open_named(names)?);
    NcclApi::from_loader(loader)
}

/// A source of dynamically resolved symbols, split out so the loader is testable without a library.
pub trait SymbolLoader {
    /// The library name reported in errors.
    fn library_name(&self) -> &str;

    /// Resolves one symbol, or `None` when the loaded library does not export it.
    fn symbol(&self, name: &str) -> Option<*mut c_void>;
}

/// The real `dlopen` loader.
struct DlLoader {
    handle: *mut c_void,
    name: String,
}

impl DlLoader {
    fn open() -> Result<Self, NcclError> {
        Self::open_named(&NCCL_LIBRARY_NAMES)
    }

    /// Opens the first of the names that dlopen finds, and reports the typed absence otherwise.
    ///
    /// The name list is a parameter so a test can ask for a soname no host has: pinning the refusal
    /// shape must not depend on whether the machine happens to ship libnccl. Production always
    /// passes NCCL_LIBRARY_NAMES.
    fn open_named(names: &[&'static str]) -> Result<Self, NcclError> {
        let mut detail = String::from("dlopen returned a null handle without a dlerror message");
        for name in names {
            let cname = CString::new(*name).expect("soname has no interior nul");
            // SAFETY: cname is a valid NUL-terminated string and the flags are dlopen constants.
            let handle = unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                return Ok(Self {
                    handle,
                    name: (*name).to_owned(),
                });
            }
            detail = dl_error();
        }
        Err(NcclError::LibraryAbsent {
            tried: names.to_vec(),
            detail,
        })
    }
}

impl Drop for DlLoader {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: the handle came from dlopen and is closed exactly once here.
            unsafe { libc::dlclose(self.handle) };
        }
    }
}

impl SymbolLoader for DlLoader {
    fn library_name(&self) -> &str {
        &self.name
    }

    fn symbol(&self, name: &str) -> Option<*mut c_void> {
        let cname = CString::new(name).ok()?;
        // Clear any stale error, then look the symbol up.
        let _ = dl_error();
        // SAFETY: the handle is open and cname is a valid NUL-terminated string.
        let pointer = unsafe { libc::dlsym(self.handle, cname.as_ptr()) };
        if pointer.is_null() {
            None
        } else {
            Some(pointer)
        }
    }
}

fn dl_error() -> String {
    // SAFETY: dlerror returns either NULL or a valid NUL-terminated string owned by libc.
    unsafe {
        let pointer = libc::dlerror();
        if pointer.is_null() {
            "no dlerror message".to_owned()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    }
}

type GetVersionFn = unsafe extern "C" fn(*mut c_int) -> c_int;
type GetUniqueIdFn = unsafe extern "C" fn(*mut NcclUniqueId) -> c_int;
type CommInitRankConfigFn =
    unsafe extern "C" fn(*mut NcclCommHandle, c_int, NcclUniqueId, c_int, *mut c_void) -> c_int;
type AllReduceFn = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    c_int,
    c_int,
    NcclCommHandle,
    CudaStream,
) -> c_int;
type AllGatherFn = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    usize,
    c_int,
    NcclCommHandle,
    CudaStream,
) -> c_int;
type GroupFn = unsafe extern "C" fn() -> c_int;
/// `ncclSend(const void *, size_t, ncclDataType_t, int peer, ncclComm_t, cudaStream_t)`.
type SendFn =
    unsafe extern "C" fn(*const c_void, usize, c_int, c_int, NcclCommHandle, CudaStream) -> c_int;
/// `ncclRecv(void *, size_t, ncclDataType_t, int peer, ncclComm_t, cudaStream_t)`.
type RecvFn =
    unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, NcclCommHandle, CudaStream) -> c_int;
type CommDestroyFn = unsafe extern "C" fn(NcclCommHandle) -> c_int;
type GetErrorStringFn = unsafe extern "C" fn(c_int) -> *const c_char;
/// `ncclCommGetAsyncError(ncclComm_t, ncclResult_t *asyncError) -> ncclResult_t`: the return value
/// is the query's own result; `asyncError` is the communicator's asynchronous status.
type CommGetAsyncErrorFn = unsafe extern "C" fn(NcclCommHandle, *mut c_int) -> c_int;
/// `ncclCommRegister(const ncclComm_t, void *buff, size_t size, void **handle) -> ncclResult_t`.
type CommRegisterFn =
    unsafe extern "C" fn(NcclCommHandle, *mut c_void, usize, *mut *mut c_void) -> c_int;
/// `ncclCommDeregister(const ncclComm_t, void *handle) -> ncclResult_t`.
type CommDeregisterFn = unsafe extern "C" fn(NcclCommHandle, *mut c_void) -> c_int;
/// `ncclCommSplit(ncclComm_t, int color, int key, ncclComm_t *newcomm, ncclConfig_t *) -> ncclResult_t`.
type CommSplitFn =
    unsafe extern "C" fn(NcclCommHandle, c_int, c_int, *mut NcclCommHandle, *mut c_void) -> c_int;

/// `ncclConfig_t` as NCCL 2.17 laid it out: the fields through `netName`. The struct is versioned
/// (`size`, `magic`, `version`) and every later release copies only `size` bytes of it and gives
/// the fields past them their defaults, so this prefix is the supported way to set `maxCTAs`
/// without mirroring the current, longer layout.
#[repr(C)]
struct NcclConfigV217 {
    size: usize,
    magic: u32,
    version: u32,
    blocking: c_int,
    cga_cluster_size: c_int,
    min_ctas: c_int,
    max_ctas: c_int,
    net_name: *const c_char,
}

/// `NCCL_CONFIG_UNDEF_INT`: the field keeps NCCL's default (its environment variable, if set).
const NCCL_CONFIG_UNDEF_INT: c_int = c_int::MIN;

struct RawApi {
    get_version: *mut c_void,
    get_unique_id: *mut c_void,
    comm_init_rank_config: *mut c_void,
    all_reduce: *mut c_void,
    all_gather: *mut c_void,
    group_start: *mut c_void,
    group_end: *mut c_void,
    comm_destroy: *mut c_void,
    get_error_string: *mut c_void,
    comm_register: *mut c_void,
    comm_deregister: *mut c_void,
    /// Optional: None when the loaded library predates `ncclCommGetAsyncError`.
    comm_get_async_error: Option<*mut c_void>,
    /// Optional point-to-point entry points (the one-shot exchange).
    send: Option<*mut c_void>,
    recv: Option<*mut c_void>,
    /// Optional: `ncclCommSplit` (NCCL 2.18+), for a child communicator with its own CTA cap.
    comm_split: Option<*mut c_void>,
    /// Optional: `ncclReduceScatter`, for the bulk prefill's sequence-parallel seams.
    reduce_scatter: Option<*mut c_void>,
}

/// A loaded NCCL library plus its resolved entry points.
///
/// Construct with [`NcclApi::load`]. One instance owns the `dlopen` handle; [`NcclComm`] borrows
/// it, so a communicator cannot outlive the library. The type is intentionally not `Send`: NCCL
/// communicators are used from the thread that owns the collective, and the raw pointers in the
/// function table carry that restriction at compile time.
pub struct NcclApi {
    loader: Box<dyn SymbolLoader>,
    raw: RawApi,
    version: Option<c_int>,
}

impl NcclApi {
    /// Loads `libnccl.so.2` (then `libnccl.so`) and resolves the required API.
    ///
    /// # Errors
    /// Returns [`NcclError::LibraryAbsent`] when no candidate loads (the expected result on all
    /// four nodes today), [`NcclError::MissingSymbol`] for a truncated library,
    /// [`NcclError::Call`] when `ncclGetVersion` itself fails and [`NcclError::Version`] below
    /// the 2.28 floor. It never falls back to host staging or to the one-shot path.
    pub fn load() -> Result<Self, NcclError> {
        let loader: Box<dyn SymbolLoader> = Box::new(DlLoader::open()?);
        let mut api = Self::from_loader(loader)?;
        let version = api.probe_version()?;
        if version < NCCL_MIN_VERSION_CODE {
            return Err(NcclError::Version {
                found: version,
                minimum: NCCL_MIN_VERSION_CODE,
            });
        }
        api.version = Some(version);
        Ok(api)
    }

    /// Resolves the required symbols but does not call into the library.
    ///
    /// Private because a successful resolution without a version probe is not a usable API; the
    /// tests use it to pin the loader's missing-symbol behaviour without a real library.
    fn from_loader(loader: Box<dyn SymbolLoader>) -> Result<Self, NcclError> {
        let library = loader.library_name().to_owned();
        let resolve = |symbol: &'static str| {
            loader
                .symbol(symbol)
                .ok_or_else(|| NcclError::MissingSymbol {
                    library: library.clone(),
                    symbol,
                })
        };
        let raw = RawApi {
            get_version: resolve("ncclGetVersion")?,
            get_unique_id: resolve("ncclGetUniqueId")?,
            comm_init_rank_config: resolve("ncclCommInitRankConfig")?,
            all_reduce: resolve("ncclAllReduce")?,
            all_gather: resolve("ncclAllGather")?,
            group_start: resolve("ncclGroupStart")?,
            group_end: resolve("ncclGroupEnd")?,
            comm_destroy: resolve("ncclCommDestroy")?,
            get_error_string: resolve("ncclGetErrorString")?,
            comm_register: resolve("ncclCommRegister")?,
            comm_deregister: resolve("ncclCommDeregister")?,
            // Optional: an image without it loads and runs; the boundary reports the check as
            // unavailable rather than refusing a library that is otherwise complete.
            comm_get_async_error: loader.symbol(ASYNC_ERROR_SYMBOL),
            send: loader.symbol("ncclSend"),
            recv: loader.symbol("ncclRecv"),
            comm_split: loader.symbol("ncclCommSplit"),
            reduce_scatter: loader.symbol("ncclReduceScatter"),
        };
        Ok(Self {
            loader,
            raw,
            version: None,
        })
    }

    /// Version code reported by `ncclGetVersion`, when [`Self::load`] ran it.
    #[must_use]
    pub fn version(&self) -> Option<c_int> {
        self.version
    }

    /// `major.minor.patch` string for the probed version.
    #[must_use]
    pub fn version_string(&self) -> Option<String> {
        self.version
            .map(|code| format!("{}.{}.{}", code / 10_000, (code % 10_000) / 100, code % 100))
    }

    /// The loaded library name.
    #[must_use]
    pub fn library_name(&self) -> &str {
        self.loader.library_name()
    }

    fn probe_version(&mut self) -> Result<c_int, NcclError> {
        let mut version: c_int = 0;
        // SAFETY: the symbol resolved as ncclGetVersion and the out-pointer is valid for the call.
        let result = unsafe {
            let function: GetVersionFn = std::mem::transmute(self.raw.get_version);
            function(&mut version)
        };
        if result == NCCL_SUCCESS {
            Ok(version)
        } else {
            Err(self.call_error("ncclGetVersion", result))
        }
    }

    /// Builds the error for a failed NCCL call, using `ncclGetErrorString` for the detail.
    fn call_error(&self, call: &'static str, result: c_int) -> NcclError {
        NcclError::Call {
            call,
            result,
            detail: self.error_text(result),
        }
    }

    /// `ncclGetErrorString` text for a result code, or the numeric code when it cannot be read.
    fn error_text(&self, result: c_int) -> String {
        // SAFETY: the symbol resolved as ncclGetErrorString; NCCL owns the returned string.
        unsafe {
            let function: GetErrorStringFn = std::mem::transmute(self.raw.get_error_string);
            let pointer = function(result);
            if pointer.is_null() {
                format!("ncclResult_t {result}")
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        }
    }

    /// Whether the loaded library exports `ncclCommGetAsyncError`.
    ///
    /// This is the failure check's availability, not a failure: a library without the symbol is
    /// complete for everything this binding requires, and the port's step sync is then the only
    /// failure detection. The run record should state this so a run is not accused of skipping a
    /// check it could not perform.
    #[must_use]
    pub fn has_async_failure_check(&self) -> bool {
        self.raw.comm_get_async_error.is_some()
    }

    /// Queries the communicator's asynchronous error status: failure detection, not completion.
    ///
    /// Called at the run boundary, after the step's sync and before the counters are read. Two rules
    /// keep the result honest:
    /// * **The check is optional at the symbol level.** A library without
    ///   [`ASYNC_ERROR_SYMBOL`] returns `Ok(())` here and [`Self::has_async_failure_check`] reports
    ///   the check as unavailable; an absent symbol is not a failure and never becomes one.
    /// * **Only a real asynchronous failure is an error.** `ncclSuccess` and `ncclInProgress` both
    ///   mean "no failure reported" (`ncclInProgress` is normal while work is in flight); every
    ///   other communicator status is [`NcclError::AsyncFailure`], and a failure of the query call
    ///   itself is [`NcclError::Call`].
    ///
    /// # Safety
    /// `handle` must be a live communicator created by this library and not yet destroyed.
    ///
    /// # Errors
    /// Returns [`NcclError::Call`] when the query call fails and [`NcclError::AsyncFailure`] when the
    /// communicator reports an asynchronous failure.
    pub(crate) unsafe fn check_async_failure(
        &self,
        handle: NcclCommHandle,
    ) -> Result<(), NcclError> {
        let Some(symbol) = self.raw.comm_get_async_error else {
            return Ok(());
        };
        let mut async_error: c_int = NCCL_SUCCESS;
        // SAFETY: the caller guarantees a live handle; symbol resolved as ncclCommGetAsyncError and
        // async_error is a valid out-parameter for it.
        let result = unsafe {
            let function: CommGetAsyncErrorFn = std::mem::transmute(symbol);
            function(handle, &mut async_error)
        };
        if result != NCCL_SUCCESS {
            return Err(self.call_error(ASYNC_ERROR_SYMBOL, result));
        }
        self.classify_async_error(async_error)
    }

    /// Maps the communicator's asynchronous status to this binding's result.
    ///
    /// Split out so the mapping is testable without a library or a device: `ncclSuccess` and
    /// `ncclInProgress` are "no failure", everything else is a typed failure.
    fn classify_async_error(&self, async_error: c_int) -> Result<(), NcclError> {
        if async_error == NCCL_SUCCESS || async_error == NCCL_IN_PROGRESS {
            return Ok(());
        }
        Err(NcclError::AsyncFailure {
            result: async_error,
            detail: self.error_text(async_error),
        })
    }

    /// Asks NCCL for a fresh unique id. Rank 0 does this and broadcasts the 128 bytes.
    ///
    /// # Errors
    /// Returns [`NcclError::Call`] when the library refuses.
    pub fn unique_id(&self) -> Result<NcclUniqueId, NcclError> {
        let mut id = NcclUniqueId {
            bytes: [0u8; NCCL_UNIQUE_ID_BYTES],
        };
        // SAFETY: the symbol resolved as ncclGetUniqueId and id is a valid out-parameter.
        let result = unsafe {
            let function: GetUniqueIdFn = std::mem::transmute(self.raw.get_unique_id);
            function(&mut id)
        };
        if result == NCCL_SUCCESS {
            Ok(id)
        } else {
            Err(self.call_error("ncclGetUniqueId", result))
        }
    }

    /// Creates this rank's communicator from the shared unique id.
    ///
    /// The `ncclConfig_t*` argument is NULL: NCCL documents that as the default configuration,
    /// and the tuning knobs the deployed recipe needs are environment variables. Passing NULL
    /// avoids mirroring a struct whose layout is versioned (see the module docs).
    ///
    /// # Errors
    /// Returns [`NcclError::Config`] for a rank outside the world or a zero world, and
    /// [`NcclError::Call`] when NCCL refuses the communicator.
    pub fn comm_init_rank_config(
        &self,
        world_size: usize,
        rank: usize,
        unique_id: NcclUniqueId,
    ) -> Result<NcclComm<'_>, NcclError> {
        let handle = self.comm_init_rank_handle(world_size, rank, unique_id)?;
        Ok(NcclComm { api: self, handle })
    }

    /// Creates the raw communicator handle, without borrowing the library.
    ///
    /// This is the form an owned wrapper needs: it can hold the library and the handle in one
    /// object and destroy the handle in Drop, which a borrowing NcclComm cannot express without a
    /// self-referential struct.
    ///
    /// # Errors
    /// Returns NcclError::Config for a rank outside the world or a zero world, and
    /// NcclError::Call when NCCL refuses the communicator.
    pub(crate) fn comm_init_rank_handle(
        &self,
        world_size: usize,
        rank: usize,
        unique_id: NcclUniqueId,
    ) -> Result<NcclCommHandle, NcclError> {
        if world_size == 0 {
            return Err(NcclError::Config(
                "a communicator needs at least one rank".to_owned(),
            ));
        }
        if rank >= world_size {
            return Err(NcclError::Config(format!(
                "rank {rank} is out of range for world {world_size}"
            )));
        }
        let world = c_int::try_from(world_size)
            .map_err(|_| NcclError::Config(format!("world {world_size} does not fit in an int")))?;
        let index = c_int::try_from(rank)
            .map_err(|_| NcclError::Config(format!("rank {rank} does not fit in an int")))?;
        let mut handle: NcclCommHandle = ptr::null_mut();
        // SAFETY: the symbol resolved as ncclCommInitRankConfig; handle is a valid out-parameter,
        // unique_id is a by-value copy of 128 bytes, and a null config is the documented default.
        let result = unsafe {
            let function: CommInitRankConfigFn =
                std::mem::transmute(self.raw.comm_init_rank_config);
            function(&mut handle, world, unique_id, index, ptr::null_mut())
        };
        if result != NCCL_SUCCESS {
            return Err(self.call_error("ncclCommInitRankConfig", result));
        }
        Ok(handle)
    }

    /// A child of `parent` over the same ranks (one color, keyed by `rank`) whose collectives run
    /// on at most `max_ctas` CTAs. Collective over `parent`: every rank calls it at the same point
    /// of its collective sequence. The child has its own channels and buffers (no resource
    /// sharing), so its collectives can run beside the parent's.
    ///
    /// # Errors
    /// NcclError::Config for a zero cap, NcclError::MissingSymbol when the library has no
    /// `ncclCommSplit`, and NcclError::Call when NCCL refuses the split.
    pub(crate) fn comm_split_handle(
        &self,
        parent: NcclCommHandle,
        rank: usize,
        max_ctas: u32,
    ) -> Result<NcclCommHandle, NcclError> {
        let split = self
            .raw
            .comm_split
            .ok_or_else(|| NcclError::MissingSymbol {
                library: self.loader.library_name().to_owned(),
                symbol: "ncclCommSplit",
            })?;
        let cap = c_int::try_from(max_ctas)
            .ok()
            .filter(|&cap| cap > 0)
            .ok_or_else(|| NcclError::Config(format!("a CTA cap of {max_ctas} is not usable")))?;
        let key = c_int::try_from(rank)
            .map_err(|_| NcclError::Config(format!("rank {rank} does not fit in an int")))?;
        let mut config = NcclConfigV217 {
            size: std::mem::size_of::<NcclConfigV217>(),
            magic: 0xcafe_beef,
            version: 21_700,
            blocking: NCCL_CONFIG_UNDEF_INT,
            cga_cluster_size: NCCL_CONFIG_UNDEF_INT,
            min_ctas: NCCL_CONFIG_UNDEF_INT,
            max_ctas: cap,
            net_name: ptr::null(),
        };
        let mut handle: NcclCommHandle = ptr::null_mut();
        // SAFETY: the symbol resolved as ncclCommSplit; parent is a live communicator, handle a
        // valid out-parameter and config a 2.17-versioned ncclConfig_t that outlives the call.
        let result = unsafe {
            let function: CommSplitFn = std::mem::transmute(split);
            function(
                parent,
                0,
                key,
                &mut handle,
                (&raw mut config).cast::<c_void>(),
            )
        };
        if result != NCCL_SUCCESS {
            return Err(self.call_error("ncclCommSplit", result));
        }
        if handle.is_null() {
            return Err(NcclError::Config(
                "ncclCommSplit returned no communicator for this rank".to_owned(),
            ));
        }
        Ok(handle)
    }

    /// Destroys a raw communicator handle created by Self::comm_init_rank_handle.
    ///
    /// A null handle is a no-op, so a teardown path can call this unconditionally.
    ///
    /// # Errors
    /// Returns NcclError::Call when NCCL reports a failure; the caller has already given up the
    /// handle either way, because NCCL marks a communicator destroyed once destroy is entered.
    pub(crate) fn comm_destroy_handle(&self, handle: NcclCommHandle) -> Result<(), NcclError> {
        if handle.is_null() {
            return Ok(());
        }
        // SAFETY: the handle came from ncclCommInitRankConfig and this is its single destroy.
        let result = unsafe {
            let function: CommDestroyFn = std::mem::transmute(self.raw.comm_destroy);
            function(handle)
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclCommDestroy", result))
        }
    }

    /// Registers one device buffer with a raw communicator handle and returns NCCL's registration.
    ///
    /// Registration is setup, never step work: the caller registers the run's collective buffers
    /// once at open and deregisters them before the communicator is destroyed. A zero-byte or null
    /// buffer is refused before NCCL is entered, because NCCL would accept neither and its error
    /// would not name the buffer.
    ///
    /// # Safety
    /// The handle must be live, and `buffer` must be the start of `bytes` bytes of device memory on
    /// the communicator's device that stays allocated until the registration is deregistered.
    ///
    /// # Errors
    /// Returns NcclError::Config for a null or empty buffer and NcclError::Call when NCCL refuses.
    pub(crate) unsafe fn comm_register_handle(
        &self,
        handle: NcclCommHandle,
        buffer: *mut c_void,
        bytes: usize,
    ) -> Result<*mut c_void, NcclError> {
        if buffer.is_null() || bytes == 0 {
            return Err(NcclError::Config(format!(
                "ncclCommRegister needs a non-null buffer of at least one byte, got {buffer:p} with {bytes} bytes"
            )));
        }
        let mut registration: *mut c_void = ptr::null_mut();
        // SAFETY: the caller upholds the handle and buffer contract; registration is a valid
        // out-parameter for ncclCommRegister.
        let result = unsafe {
            let function: CommRegisterFn = std::mem::transmute(self.raw.comm_register);
            function(handle, buffer, bytes, &mut registration)
        };
        if result != NCCL_SUCCESS {
            return Err(self.call_error("ncclCommRegister", result));
        }
        if registration.is_null() {
            return Err(NcclError::Config(format!(
                "ncclCommRegister returned success but a null registration for {bytes} bytes at {buffer:p}"
            )));
        }
        Ok(registration)
    }

    /// Releases one registration made by [`Self::comm_register_handle`].
    ///
    /// A null registration is a no-op, so a teardown path can call this unconditionally.
    ///
    /// # Safety
    /// The handle must be the live communicator the registration was made on, and the registration
    /// must not have been released already.
    ///
    /// # Errors
    /// Returns NcclError::Call when NCCL reports a failure.
    pub(crate) unsafe fn comm_deregister_handle(
        &self,
        handle: NcclCommHandle,
        registration: *mut c_void,
    ) -> Result<(), NcclError> {
        if registration.is_null() {
            return Ok(());
        }
        // SAFETY: the caller upholds the handle and registration contract documented above.
        let result = unsafe {
            let function: CommDeregisterFn = std::mem::transmute(self.raw.comm_deregister);
            function(handle, registration)
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclCommDeregister", result))
        }
    }

    /// Enqueues one bf16 AllReduce(Sum) on a raw communicator handle.
    ///
    /// NcclComm::all_reduce_bf16 delegates here; owned wrappers call this directly.
    ///
    /// # Safety
    /// The handle must be live; send must point to at least count bf16 values and recv to at least
    /// count writable bf16 values, both in memory NCCL can access on the current device; stream
    /// must be a valid CUDA stream on that device or NULL for the default stream.
    ///
    /// # Errors
    /// Returns NcclError::Call when NCCL refuses the enqueue.
    pub(crate) unsafe fn enqueue_all_reduce_bf16(
        &self,
        handle: NcclCommHandle,
        send: *const c_void,
        recv: *mut c_void,
        count: usize,
        op: NcclRedOp,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        // SAFETY: the caller upholds the handle, pointer and stream contract documented above.
        let result = unsafe {
            let function: AllReduceFn = std::mem::transmute(self.raw.all_reduce);
            function(
                send,
                recv,
                count,
                NcclDataType::Bfloat16 as c_int,
                op as c_int,
                handle,
                stream,
            )
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclAllReduce", result))
        }
    }

    /// Enqueues one bf16 ReduceScatter(Sum) on a raw communicator handle: rank r's `recv` gets the
    /// sum over the ranks of their `send` values `r * recv_count ..` (`ncclReduceScatter`'s
    /// signature is AllReduce's, with the per-rank count).
    ///
    /// # Safety
    /// The handle must be live; send must point to at least recv_count * world bf16 values and recv
    /// to recv_count writable ones, in NCCL-accessible memory; stream must be a valid CUDA stream on
    /// that device or NULL.
    ///
    /// # Errors
    /// NcclError::MissingSymbol when the library lacks ncclReduceScatter, NcclError::Call when NCCL
    /// refuses the enqueue.
    pub(crate) unsafe fn enqueue_reduce_scatter_bf16(
        &self,
        handle: NcclCommHandle,
        send: *const c_void,
        recv: *mut c_void,
        recv_count: usize,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        let entry = self
            .raw
            .reduce_scatter
            .ok_or_else(|| NcclError::MissingSymbol {
                library: self.loader.library_name().to_owned(),
                symbol: "ncclReduceScatter",
            })?;
        // SAFETY: the caller upholds the handle, pointer and stream contract documented above.
        let result = unsafe {
            let function: AllReduceFn = std::mem::transmute(entry);
            function(
                send,
                recv,
                recv_count,
                NcclDataType::Bfloat16 as c_int,
                NcclRedOp::Sum as c_int,
                handle,
                stream,
            )
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclReduceScatter", result))
        }
    }

    /// Enqueues one bf16 AllGather of `send_count` values per rank on a raw communicator handle.
    ///
    /// # Safety
    /// The handle must be live; send must point to at least send_count bf16 values and recv to
    /// send_count * world writable bf16 values in NCCL-accessible memory (send may be recv's own
    /// slot, NCCL's in-place form); stream must be a valid CUDA stream on that device or NULL.
    ///
    /// # Errors
    /// Returns NcclError::Call when NCCL refuses the enqueue.
    pub(crate) unsafe fn enqueue_all_gather_bf16(
        &self,
        handle: NcclCommHandle,
        send: *const c_void,
        recv: *mut c_void,
        send_count: usize,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        // SAFETY: the caller upholds the handle, pointer and stream contract documented above.
        let result = unsafe {
            let function: AllGatherFn = std::mem::transmute(self.raw.all_gather);
            function(
                send,
                recv,
                send_count,
                NcclDataType::Bfloat16 as c_int,
                handle,
                stream,
            )
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclAllGather", result))
        }
    }

    /// Enqueues a bf16 send of `count` values to `peer` on a raw communicator handle.
    ///
    /// # Safety
    /// The handle must be live and `send` a device buffer of at least `count` bf16 values that stays
    /// live until the stream reaches the operation.
    ///
    /// # Errors
    /// NcclError::Call when the library lacks ncclSend or refuses the enqueue.
    pub(crate) unsafe fn enqueue_send_bf16(
        &self,
        handle: NcclCommHandle,
        send: *const c_void,
        count: usize,
        peer: c_int,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        let Some(symbol) = self.raw.send else {
            return Err(self.call_error("ncclSend", -1));
        };
        // SAFETY: the symbol resolved as ncclSend; the caller upholds the buffer contract.
        let result = unsafe {
            let function: SendFn = std::mem::transmute(symbol);
            function(
                send,
                count,
                NcclDataType::Bfloat16 as c_int,
                peer,
                handle,
                stream,
            )
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclSend", result))
        }
    }

    /// Enqueues a bf16 receive of `count` values from `peer` on a raw communicator handle.
    ///
    /// # Safety
    /// The handle must be live and `recv` a writable device buffer of at least `count` bf16 values
    /// that stays live until the stream reaches the operation.
    ///
    /// # Errors
    /// NcclError::Call when the library lacks ncclRecv or refuses the enqueue.
    pub(crate) unsafe fn enqueue_recv_bf16(
        &self,
        handle: NcclCommHandle,
        recv: *mut c_void,
        count: usize,
        peer: c_int,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        let Some(symbol) = self.raw.recv else {
            return Err(self.call_error("ncclRecv", -1));
        };
        // SAFETY: the symbol resolved as ncclRecv; the caller upholds the buffer contract.
        let result = unsafe {
            let function: RecvFn = std::mem::transmute(symbol);
            function(
                recv,
                count,
                NcclDataType::Bfloat16 as c_int,
                peer,
                handle,
                stream,
            )
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclRecv", result))
        }
    }

    /// Opens a group so several collectives are enqueued in one NCCL launch.
    ///
    /// # Errors
    /// Returns [`NcclError::Call`] when NCCL refuses.
    pub fn group_start(&self) -> Result<(), NcclError> {
        // SAFETY: the symbol resolved as ncclGroupStart.
        let result = unsafe {
            let function: GroupFn = std::mem::transmute(self.raw.group_start);
            function()
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclGroupStart", result))
        }
    }

    /// Whether the library exports ncclSend and ncclRecv.
    #[must_use]
    pub fn has_point_to_point(&self) -> bool {
        self.raw.send.is_some() && self.raw.recv.is_some()
    }

    /// Closes a group.
    ///
    /// # Errors
    /// Returns [`NcclError::Call`] when NCCL refuses.
    pub fn group_end(&self) -> Result<(), NcclError> {
        // SAFETY: the symbol resolved as ncclGroupEnd.
        let result = unsafe {
            let function: GroupFn = std::mem::transmute(self.raw.group_end);
            function()
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.call_error("ncclGroupEnd", result))
        }
    }
}

/// One NCCL communicator, borrowing the library that created it.
pub struct NcclComm<'api> {
    api: &'api NcclApi,
    handle: NcclCommHandle,
}

impl NcclComm<'_> {
    /// The raw `ncclComm_t`, for a caller that needs to pass it to another NCCL call.
    #[must_use]
    pub fn as_raw(&self) -> NcclCommHandle {
        self.handle
    }

    /// Enqueues a bf16 send of `count` values to `peer` (inside a group for the one-shot exchange).
    ///
    /// # Safety
    /// `send` is a device buffer of at least `count` bf16 values that stays live until the stream
    /// reaches the operation.
    ///
    /// # Errors
    /// [`NcclError::Call`] when the library lacks ncclSend or refuses the enqueue.
    pub unsafe fn send_bf16(
        &self,
        send: *const c_void,
        count: usize,
        peer: c_int,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        // SAFETY: the caller upholds the buffer contract.
        unsafe {
            self.api
                .enqueue_send_bf16(self.handle, send, count, peer, stream)
        }
    }

    /// Enqueues a bf16 receive of `count` values from `peer`.
    ///
    /// # Safety
    /// `recv` is a device buffer of at least `count` bf16 values that stays live until the stream
    /// reaches the operation.
    ///
    /// # Errors
    /// [`NcclError::Call`] when the library lacks ncclRecv or refuses the enqueue.
    pub unsafe fn recv_bf16(
        &self,
        recv: *mut c_void,
        count: usize,
        peer: c_int,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        // SAFETY: the caller upholds the buffer contract.
        unsafe {
            self.api
                .enqueue_recv_bf16(self.handle, recv, count, peer, stream)
        }
    }

    /// Enqueues one bf16 AllReduce(Sum) on the caller's stream.
    ///
    /// # Safety
    /// `send` must point to at least `count` bf16 values and `recv` to at least `count`
    /// writable bf16 values, both in memory NCCL can access on the current device; `stream` must
    /// be a valid CUDA stream on that device or NULL for the default stream. This wrapper cannot
    /// check any of that from a host without a GPU, so it stays `unsafe`.
    ///
    /// # Errors
    /// Returns [`NcclError::Call`] when NCCL refuses the enqueue.
    pub unsafe fn all_reduce_bf16(
        &self,
        send: *const c_void,
        recv: *mut c_void,
        count: usize,
        op: NcclRedOp,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        // SAFETY: the caller upholds the pointer and stream contract documented above.
        unsafe {
            self.api
                .enqueue_all_reduce_bf16(self.handle, send, recv, count, op, stream)
        }
    }

    /// Enqueues one bf16 AllGather on the caller's stream.
    ///
    /// # Safety
    /// `send` must point to at least `send_count` bf16 values and `recv` to
    /// `send_count * world_size` writable bf16 values in NCCL-accessible memory; `stream` must
    /// be valid on the current device or NULL.
    ///
    /// # Errors
    /// Returns [`NcclError::Call`] when NCCL refuses the enqueue.
    pub unsafe fn all_gather_bf16(
        &self,
        send: *const c_void,
        recv: *mut c_void,
        send_count: usize,
        stream: CudaStream,
    ) -> Result<(), NcclError> {
        // SAFETY: the caller upholds the pointer and stream contract documented above.
        let result = unsafe {
            let function: AllGatherFn = std::mem::transmute(self.api.raw.all_gather);
            function(
                send,
                recv,
                send_count,
                NcclDataType::Bfloat16 as c_int,
                self.handle,
                stream,
            )
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.api.call_error("ncclAllGather", result))
        }
    }
}

impl Drop for NcclComm<'_> {
    fn drop(&mut self) {
        // The handle came from ncclCommInitRankConfig and is destroyed exactly once here; a failed
        // destroy still consumes it, so the error is not actionable in Drop.
        let _ = self.api.comm_destroy_handle(self.handle);
        self.handle = ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A symbol table a test controls, with no library behind it.
    struct FakeLoader {
        name: String,
        symbols: HashMap<&'static str, *mut c_void>,
    }

    impl FakeLoader {
        fn with_all() -> Self {
            let mut symbols = HashMap::new();
            for (index, symbol) in REQUIRED_SYMBOLS.iter().enumerate() {
                // Any non-null address stands in for the entry point; from_loader never calls it.
                symbols.insert(*symbol, ptr::without_provenance_mut(index + 1));
            }
            Self {
                name: "libnccl-fake.so.2".to_owned(),
                symbols,
            }
        }

        fn without(symbol: &'static str) -> Self {
            let mut loader = Self::with_all();
            loader.symbols.remove(symbol);
            loader
        }

        /// `with_all` plus a callable `ncclCommGetAsyncError` and a callable `ncclGetErrorString`.
        ///
        /// The failure path reads the detail through `ncclGetErrorString`, so that entry has to be a
        /// real function here; `with_all`'s dummy addresses are only ever resolved, never called.
        fn with_async(entry: *mut c_void) -> Self {
            let mut loader = Self::with_all();
            loader
                .symbols
                .insert("ncclGetErrorString", fake_error_entry(fake_error_string));
            loader.symbols.insert(ASYNC_ERROR_SYMBOL, entry);
            loader
        }
    }

    unsafe extern "C" fn fake_error_string(code: c_int) -> *const c_char {
        match code {
            2 => c"simulated asynchronous system error".as_ptr(),
            _ => c"simulated nccl error".as_ptr(),
        }
    }

    unsafe extern "C" fn async_ok(_handle: NcclCommHandle, out: *mut c_int) -> c_int {
        unsafe {
            *out = NCCL_SUCCESS;
        }
        NCCL_SUCCESS
    }

    unsafe extern "C" fn async_in_progress(_handle: NcclCommHandle, out: *mut c_int) -> c_int {
        unsafe {
            *out = NCCL_IN_PROGRESS;
        }
        NCCL_SUCCESS
    }

    unsafe extern "C" fn async_system_failure(_handle: NcclCommHandle, out: *mut c_int) -> c_int {
        unsafe {
            *out = 2;
        }
        NCCL_SUCCESS
    }

    unsafe extern "C" fn async_query_fails(_handle: NcclCommHandle, _out: *mut c_int) -> c_int {
        3
    }

    /// The symbol-table representation of a callable fake entry point.
    fn fake_entry(
        function: unsafe extern "C" fn(NcclCommHandle, *mut c_int) -> c_int,
    ) -> *mut c_void {
        function as *const () as *mut c_void
    }

    /// The symbol-table representation of the fake `ncclGetErrorString`.
    fn fake_error_entry(function: unsafe extern "C" fn(c_int) -> *const c_char) -> *mut c_void {
        function as *const () as *mut c_void
    }

    impl SymbolLoader for FakeLoader {
        fn library_name(&self) -> &str {
            &self.name
        }

        fn symbol(&self, name: &str) -> Option<*mut c_void> {
            self.symbols.get(name).copied()
        }
    }

    #[test]
    fn a_missing_library_is_a_typed_refusal_not_a_fallback() {
        // A soname no host ships: the refusal shape is checked independently of what is installed.
        let names = ["libnccl-absent-for-test.so.2"];
        let error = match load_named_for_test(&names) {
            Ok(_) => panic!("an absent library must refuse"),
            Err(error) => error,
        };
        match &error {
            NcclError::LibraryAbsent { tried, .. } => assert_eq!(tried, &names),
            other => panic!("expected LibraryAbsent, got {other:?}"),
        }
        let text = error.to_string();
        assert!(text.contains("libnccl-absent-for-test.so.2"), "{text}");
    }

    fn api_with_all() -> NcclApi {
        NcclApi::from_loader(Box::new(FakeLoader::with_all()))
            .unwrap_or_else(|error| panic!("{error}"))
    }

    fn api_with_async(entry: *mut c_void) -> NcclApi {
        NcclApi::from_loader(Box::new(FakeLoader::with_async(entry)))
            .unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn the_version_floor_is_nccl_2_28() {
        assert_eq!(NCCL_MIN_VERSION_CODE, 22_800);
        assert_eq!(nccl_version_code(2, 28, 0), NCCL_MIN_VERSION_CODE);
        assert_eq!(nccl_version_code(2, 28, 3), 22_803);
        assert_eq!(nccl_version_code(2, 30, 7), 23_007);
        assert!(nccl_version_code(2, 30, 7) > NCCL_MIN_VERSION_CODE);
    }

    #[test]
    fn the_unique_id_is_the_pinned_128_byte_opaque_value() {
        assert_eq!(std::mem::size_of::<NcclUniqueId>(), 128);
        assert_eq!(std::mem::align_of::<NcclUniqueId>(), 1);
        let mut bytes = [0u8; NCCL_UNIQUE_ID_BYTES];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = index as u8;
        }
        let id = NcclUniqueId::from_bytes(bytes);
        assert_eq!(id.as_bytes(), &bytes);
        let hex = id.to_hex();
        assert_eq!(hex.len(), 256);
        assert_eq!(NcclUniqueId::from_hex(&hex), Ok(id));
        assert_eq!(
            NcclUniqueId::from_hex(&hex.to_uppercase()),
            Ok(id),
            "the bootstrap hex may arrive uppercase"
        );
        assert!(NcclUniqueId::from_hex("00").is_err());
        assert!(NcclUniqueId::from_hex(&"zz".repeat(128)).is_err());
        // Every byte value survives the round trip, including 0xff.
        let all = NcclUniqueId::from_bytes([0xff; NCCL_UNIQUE_ID_BYTES]);
        assert_eq!(NcclUniqueId::from_hex(&all.to_hex()), Ok(all));
    }

    #[test]
    fn the_dtype_and_redop_discriminants_are_the_nccl_abi() {
        for (dtype, value) in [
            (NcclDataType::Int8, 0),
            (NcclDataType::Uint8, 1),
            (NcclDataType::Int32, 2),
            (NcclDataType::Uint32, 3),
            (NcclDataType::Int64, 4),
            (NcclDataType::Uint64, 5),
            (NcclDataType::Float16, 6),
            (NcclDataType::Float32, 7),
            (NcclDataType::Float64, 8),
            (NcclDataType::Bfloat16, 9),
        ] {
            assert_eq!(dtype as c_int, value);
        }
        assert_eq!(NcclDataType::for_bf16(), NcclDataType::Bfloat16);
        assert_eq!(NcclDataType::Bfloat16.element_bytes(), 2);
        assert_eq!(NcclDataType::Float32.element_bytes(), 4);
        for (op, value) in [
            (NcclRedOp::Sum, 0),
            (NcclRedOp::Prod, 1),
            (NcclRedOp::Max, 2),
            (NcclRedOp::Min, 3),
            (NcclRedOp::Avg, 4),
        ] {
            assert_eq!(op as c_int, value);
        }
        // The run-boundary check's "still running" code is the NCCL ABI's, not this crate's guess.
        assert_eq!(NCCL_SUCCESS, 0);
        assert_eq!(NCCL_IN_PROGRESS, 7);
    }

    #[test]
    fn a_complete_symbol_table_loads_without_a_real_library() {
        let api = api_with_all();
        assert_eq!(api.library_name(), "libnccl-fake.so.2");
        assert_eq!(
            api.version(),
            None,
            "from_loader must not call into the library; only load() probes the version"
        );
        assert_eq!(api.version_string(), None);
        assert!(
            !api.has_async_failure_check(),
            "a library without ncclCommGetAsyncError must load: the boundary check is optional"
        );
    }

    #[test]
    fn a_missing_async_error_symbol_is_a_documented_no_op() {
        let api = api_with_all();
        assert!(!api.has_async_failure_check());
        // SAFETY: no symbol is present, so the check calls nothing; the null handle is never dereferenced.
        assert_eq!(unsafe { api.check_async_failure(ptr::null_mut()) }, Ok(()));
    }

    #[test]
    fn the_async_failure_check_reports_only_real_asynchronous_failures() {
        let handle = ptr::without_provenance_mut(0x1000);

        let api = api_with_async(fake_entry(async_ok));
        assert!(api.has_async_failure_check());
        // SAFETY: the fake ignores the handle.
        assert_eq!(unsafe { api.check_async_failure(handle) }, Ok(()));

        let api = api_with_async(fake_entry(async_in_progress));
        // SAFETY: the fake ignores the handle.
        assert_eq!(
            unsafe { api.check_async_failure(handle) },
            Ok(()),
            "ncclInProgress means work is in flight, not that it failed"
        );

        let api = api_with_async(fake_entry(async_system_failure));
        // SAFETY: the fake ignores the handle.
        match unsafe { api.check_async_failure(handle) } {
            Err(NcclError::AsyncFailure { result, detail }) => {
                assert_eq!(result, 2);
                assert!(
                    detail.contains("simulated asynchronous system error"),
                    "{detail}"
                );
                let text = NcclError::AsyncFailure { result, detail }.to_string();
                assert!(text.contains("run boundary"), "{text}");
                assert!(text.contains("not completion"), "{text}");
                assert!(text.contains("abort on every rank"), "{text}");
            }
            other => panic!("expected AsyncFailure, got {other:?}"),
        }

        let api = api_with_async(fake_entry(async_query_fails));
        // SAFETY: the fake ignores the handle.
        match unsafe { api.check_async_failure(handle) } {
            Err(NcclError::Call { call, result, .. }) => {
                assert_eq!(call, ASYNC_ERROR_SYMBOL);
                assert_eq!(
                    result, 3,
                    "the query's own failure is a Call, not an AsyncFailure"
                );
            }
            other => panic!("expected Call, got {other:?}"),
        }
    }

    #[test]
    fn every_required_symbol_is_named_and_a_missing_one_is_a_typed_error() {
        assert_eq!(REQUIRED_SYMBOLS.len(), 11);
        for (index, symbol) in REQUIRED_SYMBOLS.iter().enumerate() {
            assert_eq!(
                REQUIRED_SYMBOLS[index], *symbol,
                "the resolution order is part of the error message contract"
            );
            match NcclApi::from_loader(Box::new(FakeLoader::without(symbol))) {
                Err(NcclError::MissingSymbol {
                    library,
                    symbol: missing,
                }) => {
                    assert_eq!(missing, *symbol);
                    assert_eq!(library, "libnccl-fake.so.2");
                }
                Ok(_) => {
                    panic!("a missing {symbol} must be a MissingSymbol error, got a loaded API")
                }
                Err(other) => {
                    panic!("a missing {symbol} must be a MissingSymbol error, got {other}")
                }
            }
        }
    }

    #[test]
    fn a_host_without_libnccl_refuses_with_the_image_and_the_ticket() {
        match NcclApi::load() {
            Ok(api) => {
                // A host that does have NCCL is allowed to pass, but must be 2.28+.
                let version = api
                    .version()
                    .unwrap_or_else(|| panic!("load() must probe the version"));
                assert!(
                    version >= NCCL_MIN_VERSION_CODE,
                    "load() accepted version code {version}"
                );
                assert!(api.version_string().unwrap_or_default().contains('.'));
            }
            Err(NcclError::LibraryAbsent { tried, detail }) => {
                assert_eq!(tried, NCCL_LIBRARY_NAMES.to_vec());
                let text = NcclError::LibraryAbsent { tried, detail }.to_string();
                assert!(text.contains("libnccl.so.2"), "{text}");
                assert!(text.contains(NCCL_OBSERVED_IMAGE), "{text}");
                assert!(text.contains("B5"), "{text}");
            }
            Err(other) => panic!("load() refused for an unexpected reason: {other}"),
        }
    }

    #[test]
    fn a_bad_rank_is_refused_before_nccl_is_called() {
        let api = api_with_all();
        let id = NcclUniqueId::from_bytes([7u8; NCCL_UNIQUE_ID_BYTES]);
        assert!(matches!(
            api.comm_init_rank_config(0, 0, id),
            Err(NcclError::Config(_))
        ));
        assert!(matches!(
            api.comm_init_rank_config(4, 4, id),
            Err(NcclError::Config(_))
        ));
        let text = NcclError::Config("rank 4".to_owned()).to_string();
        assert!(text.contains("rank 4"), "{text}");
    }

    #[test]
    fn the_abstention_message_does_not_claim_a_fallback() {
        // The dispatcher's contract: when NCCL is absent the caller refuses or chunks; the loader
        // must never suggest that the one-shot path silently covers a >2 MiB payload.
        let text = NcclError::LibraryAbsent {
            tried: NCCL_LIBRARY_NAMES.to_vec(),
            detail: "test".to_owned(),
        }
        .to_string();
        assert!(!text.contains("falling back"), "{text}");
        assert!(!text.contains("using oneshot"), "{text}");
    }

    #[test]
    fn a_host_bridge_failure_names_the_phase_and_the_no_count_rule() {
        let text = NcclError::HostBridge {
            phase: "host-reduce",
            detail: "peer rank 2 is no longer reachable".to_owned(),
        }
        .to_string();
        assert!(text.contains("host-bridge"), "{text}");
        assert!(text.contains("host-reduce"), "{text}");
        assert!(text.contains("not counted"), "{text}");
        assert!(text.contains("abort on every rank"), "{text}");
    }

    unsafe extern "C" fn register_ok(
        _comm: NcclCommHandle,
        _buffer: *mut c_void,
        _bytes: usize,
        out: *mut *mut c_void,
    ) -> c_int {
        unsafe {
            *out = ptr::without_provenance_mut(0x5150);
        }
        NCCL_SUCCESS
    }

    unsafe extern "C" fn register_refused(
        _comm: NcclCommHandle,
        _buffer: *mut c_void,
        _bytes: usize,
        _out: *mut *mut c_void,
    ) -> c_int {
        3
    }

    unsafe extern "C" fn register_null(
        _comm: NcclCommHandle,
        _buffer: *mut c_void,
        _bytes: usize,
        _out: *mut *mut c_void,
    ) -> c_int {
        NCCL_SUCCESS
    }

    unsafe extern "C" fn deregister_ok(_comm: NcclCommHandle, _registration: *mut c_void) -> c_int {
        NCCL_SUCCESS
    }

    /// `with_all` plus callable registration entries and a callable `ncclGetErrorString`.
    fn api_with_register(
        register: unsafe extern "C" fn(
            NcclCommHandle,
            *mut c_void,
            usize,
            *mut *mut c_void,
        ) -> c_int,
    ) -> NcclApi {
        let mut loader = FakeLoader::with_all();
        loader
            .symbols
            .insert("ncclGetErrorString", fake_error_entry(fake_error_string));
        loader
            .symbols
            .insert("ncclCommRegister", register as *const () as *mut c_void);
        loader.symbols.insert(
            "ncclCommDeregister",
            deregister_ok as unsafe extern "C" fn(NcclCommHandle, *mut c_void) -> c_int as *const ()
                as *mut c_void,
        );
        NcclApi::from_loader(Box::new(loader)).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn the_gate_script_forwards_exactly_the_recorded_transport_env() {
        // scripts/gpu-reduce-bench.sh forwards the operator's NCCL_FORWARD list to every rank; the
        // bench JSON and the run header record TRANSPORT_ENV. The two lists must be the same names,
        // or the gate would compare a variable it never forwarded.
        let script = include_str!("../../../scripts/gpu-reduce-bench.sh");
        let start = script
            .find("NCCL_FORWARD=(")
            .unwrap_or_else(|| panic!("the gate script declares NCCL_FORWARD"));
        let body = &script[start + "NCCL_FORWARD=(".len()..];
        let end = body
            .find(')')
            .unwrap_or_else(|| panic!("NCCL_FORWARD is closed"));
        let forwarded: Vec<&str> = body[..end].split_whitespace().collect();
        assert_eq!(forwarded, TRANSPORT_ENV);
    }

    #[test]
    fn a_registration_round_trips_and_an_empty_buffer_never_reaches_nccl() {
        let api = api_with_register(register_ok);
        let comm: NcclCommHandle = ptr::without_provenance_mut(0x10);
        let buffer: *mut c_void = ptr::without_provenance_mut(0x1000);
        // SAFETY: the fakes never dereference the handle or the buffer.
        let registration = unsafe { api.comm_register_handle(comm, buffer, 10_240) }
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(!registration.is_null());
        // SAFETY: as above; the registration is the fake's and is released once.
        unsafe { api.comm_deregister_handle(comm, registration) }
            .unwrap_or_else(|error| panic!("{error}"));
        // SAFETY: a null registration is the documented no-op and calls nothing.
        unsafe { api.comm_deregister_handle(comm, ptr::null_mut()) }
            .unwrap_or_else(|error| panic!("{error}"));
        for (pointer, bytes) in [(buffer, 0usize), (ptr::null_mut(), 16)] {
            // SAFETY: the argument check refuses before any entry point is called.
            match unsafe { api.comm_register_handle(comm, pointer, bytes) } {
                Err(NcclError::Config(text)) => {
                    assert!(text.contains("ncclCommRegister"), "{text}")
                }
                other => panic!("an empty or null buffer must be a Config refusal, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_refused_or_null_registration_is_an_error_never_a_handle() {
        let comm: NcclCommHandle = ptr::without_provenance_mut(0x10);
        let buffer: *mut c_void = ptr::without_provenance_mut(0x1000);
        let refused = api_with_register(register_refused);
        // SAFETY: the fake never dereferences the handle or the buffer.
        match unsafe { refused.comm_register_handle(comm, buffer, 10_240) } {
            Err(NcclError::Call { call, result, .. }) => {
                assert_eq!(call, "ncclCommRegister");
                assert_eq!(result, 3);
            }
            other => panic!("a refused registration must be a Call error, got {other:?}"),
        }
        let null = api_with_register(register_null);
        // SAFETY: as above.
        match unsafe { null.comm_register_handle(comm, buffer, 10_240) } {
            Err(NcclError::Config(text)) => assert!(text.contains("null registration"), "{text}"),
            other => panic!("a null registration must not be returned as a handle, got {other:?}"),
        }
    }
}
