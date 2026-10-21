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
//! `ncclAllGather`, `ncclGroupStart`, `ncclGroupEnd`, `ncclCommDestroy` and
//! `ncclGetErrorString`. The first two are not in the task's six-function list but are
//! prerequisites for a real communicator: the version probe enforces the 2.28 floor, and a
//! unique id is what bootstraps the four ranks. They are resolved as required symbols so a
//! truncated library fails at load with a named symbol rather than at the first collective.
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
pub const REQUIRED_SYMBOLS: [&str; 9] = [
    "ncclGetVersion",
    "ncclGetUniqueId",
    "ncclCommInitRankConfig",
    "ncclAllReduce",
    "ncclAllGather",
    "ncclGroupStart",
    "ncclGroupEnd",
    "ncclCommDestroy",
    "ncclGetErrorString",
];

/// The opaque `ncclComm_t` handle.
pub type NcclCommHandle = *mut c_void;

/// The opaque `cudaStream_t` handle. NULL is CUDA's default stream.
pub type CudaStream = *mut c_void;

/// NCCL's success code.
pub const NCCL_SUCCESS: c_int = 0;

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
    /// The caller passed an unusable configuration (bad world/rank, malformed unique id).
    Config(String),
}

impl fmt::Display for NcclError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryAbsent { tried, detail } => write!(
                formatter,
                "could not load NCCL ({}); host libnccl is absent on the four nodes (measured 2026-10-21) and NCCL 2.28.3 exists only inside the {NCCL_OBSERVED_IMAGE} image, so a host-process NCCL path needs WS-B infra ticket B5: {detail}",
                tried.join(" or ")
            ),
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
            Self::Config(text) => write!(formatter, "invalid NCCL configuration: {text}"),
        }
    }
}

impl std::error::Error for NcclError {}

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
        let mut detail = String::from("dlopen returned a null handle without a dlerror message");
        for name in NCCL_LIBRARY_NAMES {
            let cname = CString::new(name).expect("soname has no interior nul");
            // SAFETY: cname is a valid NUL-terminated string and the flags are dlopen constants.
            let handle = unsafe { libc::dlopen(cname.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                return Ok(Self {
                    handle,
                    name: name.to_owned(),
                });
            }
            detail = dl_error();
        }
        Err(NcclError::LibraryAbsent {
            tried: NCCL_LIBRARY_NAMES.to_vec(),
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
type CommDestroyFn = unsafe extern "C" fn(NcclCommHandle) -> c_int;
type GetErrorStringFn = unsafe extern "C" fn(c_int) -> *const c_char;

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
        // SAFETY: the symbol resolved as ncclGetErrorString; NCCL owns the returned string.
        let detail = unsafe {
            let function: GetErrorStringFn = std::mem::transmute(self.raw.get_error_string);
            let pointer = function(result);
            if pointer.is_null() {
                format!("ncclResult_t {result}")
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        NcclError::Call {
            call,
            result,
            detail,
        }
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
        Ok(NcclComm { api: self, handle })
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
        let result = unsafe {
            let function: AllReduceFn = std::mem::transmute(self.api.raw.all_reduce);
            function(
                send,
                recv,
                count,
                NcclDataType::Bfloat16 as c_int,
                op as c_int,
                self.handle,
                stream,
            )
        };
        if result == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(self.api.call_error("ncclAllReduce", result))
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
        if self.handle.is_null() {
            return;
        }
        // SAFETY: the handle came from ncclCommInitRankConfig and is destroyed exactly once here.
        unsafe {
            let function: CommDestroyFn = std::mem::transmute(self.api.raw.comm_destroy);
            let _ = function(self.handle);
        }
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
    }

    impl SymbolLoader for FakeLoader {
        fn library_name(&self) -> &str {
            &self.name
        }

        fn symbol(&self, name: &str) -> Option<*mut c_void> {
            self.symbols.get(name).copied()
        }
    }

    fn api_with_all() -> NcclApi {
        NcclApi::from_loader(Box::new(FakeLoader::with_all()))
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
    }

    #[test]
    fn every_required_symbol_is_named_and_a_missing_one_is_a_typed_error() {
        assert_eq!(REQUIRED_SYMBOLS.len(), 9);
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
}
