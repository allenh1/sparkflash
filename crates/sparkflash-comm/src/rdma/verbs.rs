//! Runtime `dlopen` of libibverbs and the typed, RAII wrappers this backend calls.
//!
//! Nothing here links libibverbs at build time. `Verbs::load` opens `libibverbs.so.1` (falling
//! back to `libibverbs.so`) with `RTLD_NOW | RTLD_LOCAL`, resolves the 16 entry points the library
//! exports by name, and fails with a typed error that names the symbol if the library is a
//! different ABI. The other three entry points this backend needs - `ibv_post_send`,
//! `ibv_post_recv` and `ibv_poll_cq` - are `static inline` in rdma-core's public header and are
//! NOT exported by `libibverbs.so.1` (measured with `nm -D` against rdma-core 39: 16 of the 19
//! resolve). They are supplied by the bundled C shim in `src/rdma/ibverbs_shim.c`, which
//! `build.rs` compiles with the real header and which calls the inline wrappers. That shim calls
//! no libibverbs symbol, so nothing is linked against libibverbs and the run-time dlopen still
//! owns the device context. Mirroring `struct ibv_context_ops` in Rust instead was rejected: that
//! vtable is internal to rdma-core, its layout changes between releases and no published header
//! promises it; the header's inline functions are the stable way to reach the same calls.
//!
//! This is the same run-time choice crates/sparkflash-kernels/src/cuda_driver.rs makes for
//! libcuda: a workspace build on a host with no RDMA install still compiles, the default build
//! gains no dependency, and the refusal happens at run time on the host that would have used the
//! device. A build host without <infiniband/verbs.h> still compiles the feature: build.rs then
//! emits ENOSYS stubs and `load` reports the missing build prerequisite by name.
//!
//! Every library object is owned by an RAII wrapper whose `Drop` calls the matching `ibv_destroy_*`
//! or `ibv_dealloc_*` / `ibv_close_device` entry point, and every call maps a non-zero return onto
//! `CommError::Verbs` with the library's errno text. No function in this module can panic on a
//! missing device, a refused call or a null pointer.
//!
//! Threading: libibverbs objects are safe to use from several threads as long as a single QP is not
//! driven concurrently. The backend serialises its own runtime state behind a mutex, so this module
//! only relies on that documented property rather than adding its own lock.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

use super::abi::{
    IbvCompChannel, IbvContext, IbvCq, IbvDevice, IbvGid, IbvMr, IbvPd, IbvPortAttr, IbvQp,
    IbvQpAttr, IbvQpInitAttr, IbvRecvWr, IbvSendWr, IbvWc,
};
use crate::error::CommError;

type GetDeviceList = unsafe extern "C" fn(*mut c_int) -> *mut *mut IbvDevice;
type FreeDeviceList = unsafe extern "C" fn(*mut *mut IbvDevice);
type GetDeviceName = unsafe extern "C" fn(*const IbvDevice) -> *const c_char;
type OpenDevice = unsafe extern "C" fn(*mut IbvDevice) -> *mut IbvContext;
type CloseDevice = unsafe extern "C" fn(*mut IbvContext) -> c_int;
type QueryPort = unsafe extern "C" fn(*mut IbvContext, u8, *mut IbvPortAttr) -> c_int;
type QueryGid = unsafe extern "C" fn(*mut IbvContext, u8, c_int, *mut IbvGid) -> c_int;
type AllocPd = unsafe extern "C" fn(*mut IbvContext) -> *mut IbvPd;
type DeallocPd = unsafe extern "C" fn(*mut IbvPd) -> c_int;
type CreateCq = unsafe extern "C" fn(
    *mut IbvContext,
    c_int,
    *mut c_void,
    *mut IbvCompChannel,
    c_int,
) -> *mut IbvCq;
type DestroyCq = unsafe extern "C" fn(*mut IbvCq) -> c_int;
type RegMr = unsafe extern "C" fn(*mut IbvPd, *mut c_void, usize, c_int) -> *mut IbvMr;
type DeregMr = unsafe extern "C" fn(*mut IbvMr) -> c_int;
type CreateQp = unsafe extern "C" fn(*mut IbvPd, *mut IbvQpInitAttr) -> *mut IbvQp;
type DestroyQp = unsafe extern "C" fn(*mut IbvQp) -> c_int;
type ModifyQp = unsafe extern "C" fn(*mut IbvQp, *mut IbvQpAttr, c_int) -> c_int;
type PostSend = unsafe extern "C" fn(*mut IbvQp, *mut IbvSendWr, *mut *mut IbvSendWr) -> c_int;
type PostRecv = unsafe extern "C" fn(*mut IbvQp, *mut IbvRecvWr, *mut *mut IbvRecvWr) -> c_int;
type PollCq = unsafe extern "C" fn(*mut IbvCq, c_int, *mut IbvWc) -> c_int;

// The three inline entry points, compiled into the bundled shim (see build.rs).
//
// libibverbs does not export these; they are `static inline` in infiniband/verbs.h and dispatch
// through the QP's provider ops. The shim's prototypes match the C header exactly.
unsafe extern "C" {
    fn sparkflash_ibv_post_send(
        qp: *mut IbvQp,
        wr: *mut IbvSendWr,
        bad: *mut *mut IbvSendWr,
    ) -> c_int;
    fn sparkflash_ibv_post_recv(
        qp: *mut IbvQp,
        wr: *mut IbvRecvWr,
        bad: *mut *mut IbvRecvWr,
    ) -> c_int;
    fn sparkflash_ibv_poll_cq(cq: *mut IbvCq, count: c_int, wc: *mut IbvWc) -> c_int;
}

/// How `build.rs` built the inline shim: `system` (header present, real wrappers) or `stub`.
const IBVERBS_SHIM: Option<&str> = option_env!("SPARKFLASH_IBVERBS_SHIM");

/// The build mode of the inline shim, so a caller can refuse a stub build before touching a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShimMode {
    /// The build had `<infiniband/verbs.h>` and compiled the real inline wrappers.
    System,
    /// The build had no header: the inline entry points return ENOSYS, and `Verbs::load` refuses.
    Stub,
}

/// Which shim this binary embeds.
///
/// Anything other than `SPARKFLASH_IBVERBS_SHIM=system` - the stub, and an unset value that would
/// mean the build script did not run - reports [`ShimMode::Stub`]. The refusing mode is the safe
/// default: a binary that cannot post must not be mistaken for one that can.
#[must_use]
pub fn shim_mode() -> ShimMode {
    match IBVERBS_SHIM {
        Some("system") => ShimMode::System,
        _ => ShimMode::Stub,
    }
}

/// The subset of libibverbs this backend resolves. Every pointer comes from one `dlopen` handle
/// stored in `Verbs`, and `Fns` is `Copy` so each RAII wrapper can carry the entry points it
/// needs without borrowing the loader.
#[derive(Clone, Copy)]
pub(crate) struct Fns {
    get_device_list: GetDeviceList,
    free_device_list: FreeDeviceList,
    get_device_name: GetDeviceName,
    open_device: OpenDevice,
    close_device: CloseDevice,
    query_port: QueryPort,
    query_gid: QueryGid,
    alloc_pd: AllocPd,
    dealloc_pd: DeallocPd,
    create_cq: CreateCq,
    destroy_cq: DestroyCq,
    reg_mr: RegMr,
    dereg_mr: DeregMr,
    create_qp: CreateQp,
    destroy_qp: DestroyQp,
    modify_qp: ModifyQp,
    post_send: PostSend,
    post_recv: PostRecv,
    poll_cq: PollCq,
}

/// One loaded libibverbs, resolved from one `dlopen` handle.
pub(crate) struct Verbs {
    library: *mut c_void,
    fns: Fns,
}

// SAFETY: the loader handle only keeps the library alive; the entry points this module calls are
// documented as thread-safe for distinct objects, and the backend serialises use of each QP behind
// its own mutex. dlclose runs only after every wrapper is gone (wrappers hold function pointers,
// not ownership of the library, and the backend declares the loader before them).
unsafe impl Send for Verbs {}
// SAFETY: as above; the backend does not share one QP between threads.
unsafe impl Sync for Verbs {}

impl Verbs {
    /// Opens libibverbs and resolves every entry point in the ABI subset.
    ///
    /// # Errors
    /// Returns [`CommError::Unsupported`] when neither `libibverbs.so.1` nor `libibverbs.so` can
    /// be loaded, or when a symbol is missing (a library whose ABI predates this subset). The
    /// message names the library and the missing symbol, which is the prerequisite an operator
    /// needs.
    pub(crate) fn load() -> Result<Self, CommError> {
        let mut first_error = String::new();
        let mut last_error = String::new();
        let mut library: *mut c_void = ptr::null_mut();
        for (index, name) in [&b"libibverbs.so.1\0"[..], &b"libibverbs.so\0"[..]]
            .into_iter()
            .enumerate()
        {
            // SAFETY: `name` is a NUL-terminated C string literal and RTLD_NOW makes a missing
            // symbol fail the load instead of the first call.
            library =
                unsafe { libc::dlopen(name.as_ptr().cast(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            let label = String::from_utf8_lossy(&name[..name.len() - 1]).into_owned();
            if !library.is_null() {
                if index > 0 {
                    last_error = format!("{label} loaded after libibverbs.so.1 failed; ");
                }
                break;
            }
            let detail = last_dl_error();
            if index == 0 {
                first_error = format!("{label}: {detail}");
            }
            last_error = format!("{label}: {detail}");
        }
        if library.is_null() {
            return Err(CommError::Unsupported(format!(
                "the ibverbs backend cannot load libibverbs (tried libibverbs.so.1 and libibverbs.so; {first_error}; {last_error})"
            )));
        }
        if IBVERBS_SHIM != Some("system") {
            // The library is present but this build has no <infiniband/verbs.h>, so the three
            // inline wrappers are ENOSYS stubs. Refusing here is honest: a device would open and
            // then every post would fail with a misleading errno.
            // SAFETY: `library` came from dlopen above and is not stored anywhere yet.
            unsafe { libc::dlclose(library) };
            return Err(CommError::Unsupported(
                "libibverbs is present but this build compiled the rdma feature without <infiniband/verbs.h>: the inline ibv_post_send/ibv_post_recv/ibv_poll_cq wrappers are ENOSYS stubs. Install libibverbs-dev (rdma-core headers) and rebuild with --features rdma.".to_owned(),
            ));
        }
        match resolve(library) {
            Ok(fns) => Ok(Self { library, fns }),
            Err(error) => {
                // SAFETY: `library` came from dlopen above and is not stored anywhere else yet.
                unsafe { libc::dlclose(library) };
                Err(error)
            }
        }
    }

    /// Names of every RDMA device libibverbs enumerates, in library order.
    ///
    /// Only the unit tests call this today; the bring-up path opens a device by name and
    /// [`Self::open_device`] already reports the devices that do exist when the name is missing.
    ///
    /// # Errors
    /// Returns [`CommError::FabricUnavailable`] when the library returns no device list.
    #[cfg(test)]
    pub(crate) fn device_names(&self) -> Result<Vec<String>, CommError> {
        // SAFETY: ibv_get_device_list writes the count and returns a null-terminated array that
        // ibv_free_device_list releases; the pointers are read only inside the list's lifetime.
        let mut count: c_int = 0;
        let list = unsafe { (self.fns.get_device_list)(&mut count) };
        if list.is_null() {
            return Err(CommError::FabricUnavailable(
                "libibverbs returned no device list: the RDMA subsystem is not visible to this process (check /dev/infiniband access and the rdma-core kernel modules)".to_owned(),
            ));
        }
        let mut names = Vec::new();
        if count > 0 {
            for index in 0..count as usize {
                // SAFETY: the list is a null-terminated array of `count` device pointers.
                let device = unsafe { *list.add(index) };
                if device.is_null() {
                    continue;
                }
                // SAFETY: the device belongs to the live list.
                let name = unsafe { (self.fns.get_device_name)(device) };
                if name.is_null() {
                    names.push(format!("(device {index}: null name)"));
                } else {
                    // SAFETY: ibv_get_device_name returns a NUL-terminated string owned by the list.
                    names.push(
                        unsafe { CStr::from_ptr(name) }
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
        }
        // SAFETY: `list` came from ibv_get_device_list and is freed exactly once here.
        unsafe { (self.fns.free_device_list)(list) };
        Ok(names)
    }

    /// Opens the device whose `ibv_get_device_name` is exactly `name`.
    ///
    /// # Errors
    /// Returns [`CommError::FabricUnavailable`] naming the devices that do exist when `name` is not
    /// among them, and [`CommError::Verbs`] when `ibv_open_device` refuses.
    pub(crate) fn open_device(&self, name: &str) -> Result<Context, CommError> {
        // SAFETY: as in device_names; the list stays alive until the device is opened.
        let mut count: c_int = 0;
        let list = unsafe { (self.fns.get_device_list)(&mut count) };
        if list.is_null() {
            return Err(CommError::FabricUnavailable(
                "libibverbs returned no device list while opening an RDMA device".to_owned(),
            ));
        }
        let mut found: *mut IbvDevice = ptr::null_mut();
        let mut seen = Vec::new();
        if count > 0 {
            for index in 0..count as usize {
                // SAFETY: null-terminated array of `count` device pointers.
                let device = unsafe { *list.add(index) };
                if device.is_null() {
                    continue;
                }
                // SAFETY: the device belongs to the live list.
                let pointer = unsafe { (self.fns.get_device_name)(device) };
                if pointer.is_null() {
                    continue;
                }
                // SAFETY: NUL-terminated string owned by the list.
                let device_name = unsafe { CStr::from_ptr(pointer) }
                    .to_string_lossy()
                    .into_owned();
                if device_name == name {
                    found = device;
                    break;
                }
                seen.push(device_name);
            }
        }
        if found.is_null() {
            // SAFETY: `list` came from ibv_get_device_list and is freed exactly once here.
            unsafe { (self.fns.free_device_list)(list) };
            seen.sort();
            return Err(CommError::FabricUnavailable(format!(
                "libibverbs has no device named {name}; devices visible to this process: {}",
                if seen.is_empty() {
                    "(none)".to_owned()
                } else {
                    seen.join(", ")
                }
            )));
        }
        // SAFETY: `found` is a device pointer from the live list.
        let raw = unsafe { (self.fns.open_device)(found) };
        // SAFETY: `list` came from ibv_get_device_list and is freed exactly once here.
        unsafe { (self.fns.free_device_list)(list) };
        if raw.is_null() {
            return Err(CommError::verbs(
                "ibv_open_device",
                format!(
                    "returned a null context for {name}; the device is present but not openable by this process"
                ),
            ));
        }
        Ok(Context { raw, fns: self.fns })
    }
}

impl Drop for Verbs {
    fn drop(&mut self) {
        if !self.library.is_null() {
            // SAFETY: this Verbs owns the handle returned by dlopen and drops exactly once; every
            // wrapper holding the library's function pointers is dropped before this by field order
            // in the owning backend struct.
            unsafe { libc::dlclose(self.library) };
        }
    }
}

/// Resolves every symbol the backend uses, naming the first one a library does not export.
fn resolve(library: *mut c_void) -> Result<Fns, CommError> {
    Ok(Fns {
        get_device_list: symbol(library, b"ibv_get_device_list\0")?,
        free_device_list: symbol(library, b"ibv_free_device_list\0")?,
        get_device_name: symbol(library, b"ibv_get_device_name\0")?,
        open_device: symbol(library, b"ibv_open_device\0")?,
        close_device: symbol(library, b"ibv_close_device\0")?,
        query_port: symbol(library, b"ibv_query_port\0")?,
        query_gid: symbol(library, b"ibv_query_gid\0")?,
        alloc_pd: symbol(library, b"ibv_alloc_pd\0")?,
        dealloc_pd: symbol(library, b"ibv_dealloc_pd\0")?,
        create_cq: symbol(library, b"ibv_create_cq\0")?,
        destroy_cq: symbol(library, b"ibv_destroy_cq\0")?,
        reg_mr: symbol(library, b"ibv_reg_mr\0")?,
        dereg_mr: symbol(library, b"ibv_dereg_mr\0")?,
        create_qp: symbol(library, b"ibv_create_qp\0")?,
        destroy_qp: symbol(library, b"ibv_destroy_qp\0")?,
        modify_qp: symbol(library, b"ibv_modify_qp\0")?,
        // Not dlsym'd: libibverbs does not export these three; build.rs compiles the inline
        // wrappers into the bundled shim (see ibverbs_shim.c).
        post_send: sparkflash_ibv_post_send,
        post_recv: sparkflash_ibv_post_recv,
        poll_cq: sparkflash_ibv_poll_cq,
    })
}

/// Resolves one symbol of a known C signature.
///
/// # Errors
/// Returns [`CommError::Unsupported`] naming the symbol when libibverbs does not export it.
fn symbol<T: Copy>(library: *mut c_void, name: &'static [u8]) -> Result<T, CommError> {
    // SAFETY: `library` is a live dlopen handle and `name` is NUL-terminated.
    let pointer = unsafe { libc::dlsym(library, name.as_ptr().cast()) };
    if pointer.is_null() {
        let symbol = String::from_utf8_lossy(&name[..name.len() - 1]);
        return Err(CommError::Unsupported(format!(
            "libibverbs is loaded but does not export {symbol}: this backend was written against rdma-core's public ABI subset and will not guess at a different one"
        )));
    }
    // SAFETY: dlsym returns the address of a C function; T is the documented signature of the
    // symbol named by the caller, and every call site names the matching entry point.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&pointer) })
}

/// The thread-local dlerror text, or a fallback when the loader has no detail.
fn last_dl_error() -> String {
    // SAFETY: dlerror returns a NUL-terminated string owned by the loader, or null.
    let pointer = unsafe { libc::dlerror() };
    if pointer.is_null() {
        return "no dlerror detail".to_owned();
    }
    // SAFETY: non-null and NUL-terminated, as above.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

/// Formats a libibverbs return code (an errno value) for an error message.
fn errno_text(code: c_int) -> String {
    if code > 0 {
        format!("errno {code} ({})", std::io::Error::from_raw_os_error(code))
    } else {
        format!("return code {code}")
    }
}

/// Builds the [`CommError::Verbs`] for a non-zero entry-point return.
fn verbs_error(call: &'static str, code: c_int) -> CommError {
    CommError::verbs(call, errno_text(code))
}

/// The subset of `ibv_port_attr` the backend validates and records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PortInfo {
    /// `enum ibv_port_state`; 4 is ACTIVE.
    pub(crate) state: c_int,
    /// `enum ibv_mtu` negotiated by the link.
    pub(crate) active_mtu: c_int,
    /// Number of GID-table entries the port exposes.
    pub(crate) gid_tbl_len: c_int,
    /// `enum ibv_link_layer`; 2 is Ethernet (the RoCE case).
    pub(crate) link_layer: u8,
}

/// One open device context.
pub(crate) struct Context {
    raw: *mut IbvContext,
    fns: Fns,
}

impl Context {
    /// Queries the port's state, negotiated MTU and GID table length.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_query_port` refuses, or [`CommError::Protocol`] when
    /// the library leaves the returned struct in a state this backend cannot name.
    pub(crate) fn query_port(&self, port: u8) -> Result<PortInfo, CommError> {
        // SAFETY: the struct is all integers and the library writes every field it has; a zeroed
        // buffer is a valid value for it.
        let mut attr: IbvPortAttr = unsafe { std::mem::zeroed() };
        // SAFETY: `raw` is a live context and `port` is the 1-based port number the caller
        // validated against the device's port count.
        let code = unsafe { (self.fns.query_port)(self.raw, port, &mut attr) };
        if code != 0 {
            return Err(verbs_error("ibv_query_port", code));
        }
        Ok(PortInfo {
            state: attr.state,
            active_mtu: attr.active_mtu,
            gid_tbl_len: attr.gid_tbl_len,
            link_layer: attr.link_layer,
        })
    }

    /// Reads GID table entry `index`.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_query_gid` refuses.
    pub(crate) fn query_gid(&self, port: u8, index: c_int) -> Result<IbvGid, CommError> {
        let mut gid = IbvGid::default();
        // SAFETY: `raw` is a live context, the port is validated by the caller, and `gid` is a
        // writable union buffer.
        let code = unsafe { (self.fns.query_gid)(self.raw, port, index, &mut gid) };
        if code != 0 {
            return Err(verbs_error("ibv_query_gid", code));
        }
        Ok(gid)
    }

    /// Allocates a protection domain.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_alloc_pd` returns null.
    pub(crate) fn alloc_pd(&self) -> Result<Pd, CommError> {
        // SAFETY: `raw` is a live context.
        let raw = unsafe { (self.fns.alloc_pd)(self.raw) };
        if raw.is_null() {
            return Err(CommError::verbs(
                "ibv_alloc_pd",
                "returned a null protection domain; the device is open but its resources are exhausted or the process lacks device access",
            ));
        }
        Ok(Pd { raw, fns: self.fns })
    }

    /// Creates a completion queue with room for `cqe` entries.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_create_cq` returns null.
    pub(crate) fn create_cq(&self, cqe: c_int) -> Result<Cq, CommError> {
        // SAFETY: `raw` is a live context; null completion channel selects polling-only mode.
        let raw =
            unsafe { (self.fns.create_cq)(self.raw, cqe, ptr::null_mut(), ptr::null_mut(), 0) };
        if raw.is_null() {
            return Err(CommError::verbs(
                "ibv_create_cq",
                format!("returned a null completion queue for {cqe} entries"),
            ));
        }
        Ok(Cq { raw, fns: self.fns })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: this Context owns the device context and drops exactly once. Failure to close is
        // not actionable at drop time and the process is exiting the device anyway.
        unsafe { (self.fns.close_device)(self.raw) };
    }
}

/// One protection domain.
pub(crate) struct Pd {
    raw: *mut IbvPd,
    fns: Fns,
}

impl Pd {
    /// Registers a memory region.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_reg_mr` returns null: usually `RLIMIT_MEMLOCK`,
    /// which is why the bring-up error text tells the operator to raise it.
    pub(crate) fn reg_mr(
        &self,
        addr: *mut c_void,
        length: usize,
        access: c_int,
    ) -> Result<Mr, CommError> {
        // SAFETY: the caller owns `addr`/`length` for the lifetime of the returned Mr (the
        // backend keeps the registered allocation alive) and `raw` is a live protection domain.
        let raw = unsafe { (self.fns.reg_mr)(self.raw, addr, length, access) };
        if raw.is_null() {
            return Err(CommError::verbs(
                "ibv_reg_mr",
                format!(
                    "refused a {length}-byte registration: check RLIMIT_MEMLOCK (the DGX Spark runbook uses ulimit -l unlimited) and that the address is unlocked host or device memory"
                ),
            ));
        }
        Ok(Mr { raw, fns: self.fns })
    }

    /// Creates a reliable-connected QP that posts to `send_cq` and consumes `recv_cq`.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_create_qp` returns null.
    pub(crate) fn create_qp(
        &self,
        send_cq: &Cq,
        recv_cq: &Cq,
        cap: super::abi::IbvQpCap,
    ) -> Result<Qp, CommError> {
        let mut attr = IbvQpInitAttr {
            qp_context: ptr::null_mut(),
            send_cq: send_cq.raw,
            recv_cq: recv_cq.raw,
            srq: ptr::null_mut(),
            cap,
            qp_type: super::abi::qp_type::RC,
            sq_sig_all: 0,
            reserved: [0; 4],
        };
        // SAFETY: `raw` is a live protection domain, the CQs outlive the QP in the owning
        // backend, and `attr` is a fully initialised struct.
        let raw = unsafe { (self.fns.create_qp)(self.raw, &mut attr) };
        if raw.is_null() {
            return Err(CommError::verbs(
                "ibv_create_qp",
                "returned a null QP; the requested queue capacities may exceed the device's",
            ));
        }
        Ok(Qp { raw, fns: self.fns })
    }
}

impl Drop for Pd {
    fn drop(&mut self) {
        // SAFETY: owns the protection domain and drops once, after every MR/QP that uses it is
        // dropped (field order in the owning backend).
        unsafe { (self.fns.dealloc_pd)(self.raw) };
    }
}

/// One completion queue.
pub(crate) struct Cq {
    raw: *mut IbvCq,
    fns: Fns,
}

impl Cq {
    /// Polls up to `wcs.len()` completions, returning how many were written.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_poll_cq` reports a negative errno, and
    /// [`CommError::Unsupported`] when the requested count does not fit the C entry point's int.
    pub(crate) fn poll(&self, wcs: &mut [IbvWc]) -> Result<usize, CommError> {
        let count = c_int::try_from(wcs.len()).map_err(|_| {
            CommError::Unsupported(format!(
                "a poll of {} completions exceeds the int-sized C entry point",
                wcs.len()
            ))
        })?;
        // SAFETY: `raw` is a live CQ and `wcs` is a writable slice of that many entries.
        let code = unsafe { (self.fns.poll_cq)(self.raw, count, wcs.as_mut_ptr()) };
        if code < 0 {
            let errno = code.checked_neg().unwrap_or(code);
            return Err(CommError::verbs(
                "ibv_poll_cq",
                format!("returned {code} ({})", errno_text(errno)),
            ));
        }
        Ok(code as usize)
    }
}

impl Drop for Cq {
    fn drop(&mut self) {
        // SAFETY: owns the CQ and drops once, after every QP attached to it.
        unsafe { (self.fns.destroy_cq)(self.raw) };
    }
}

/// One registered memory region.
pub(crate) struct Mr {
    raw: *mut IbvMr,
    fns: Fns,
}

impl Mr {
    /// Base address the registration covers.
    pub(crate) fn addr(&self) -> u64 {
        // SAFETY: `raw` is a live MR and the public prefix is stable; the backend reads it
        // immediately after registration and never mutates it.
        unsafe { (*self.raw).addr as u64 }
    }

    /// Local key for SGEs.
    pub(crate) fn lkey(&self) -> u32 {
        // SAFETY: as `addr`.
        unsafe { (*self.raw).lkey }
    }

    /// Remote key authorising peer RDMA operations.
    pub(crate) fn rkey(&self) -> u32 {
        // SAFETY: as `addr`.
        unsafe { (*self.raw).rkey }
    }
}

impl Drop for Mr {
    fn drop(&mut self) {
        // SAFETY: owns the MR and drops once, after every QP that could reference it.
        unsafe { (self.fns.dereg_mr)(self.raw) };
    }
}

/// One reliable-connected queue pair.
pub(crate) struct Qp {
    raw: *mut IbvQp,
    fns: Fns,
}

impl Qp {
    /// The QP number the peer's RTR transition needs.
    pub(crate) fn qp_num(&self) -> u32 {
        // SAFETY: `raw` is a live QP and `qp_num` is inside the public prefix asserted in abi.rs.
        unsafe { (*self.raw).qp_num }
    }

    /// Applies a QP attribute transition.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_modify_qp` refuses the transition.
    pub(crate) fn modify(&self, attr: &mut IbvQpAttr, mask: c_int) -> Result<(), CommError> {
        // SAFETY: `raw` is a live QP and `attr` is a fully initialised struct of the size the
        // library's own struct needs (asserted in abi.rs).
        let code = unsafe { (self.fns.modify_qp)(self.raw, attr, mask) };
        if code != 0 {
            return Err(verbs_error("ibv_modify_qp", code));
        }
        Ok(())
    }

    /// Posts the work-request chain starting at `wr`.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_post_send` reports the first failed WR.
    pub(crate) fn post_send(&self, wr: &mut IbvSendWr) -> Result<(), CommError> {
        let mut bad: *mut IbvSendWr = ptr::null_mut();
        // SAFETY: `raw` is a live QP in RTS state; `wr` is a fully initialised head of a chain.
        let code = unsafe { (self.fns.post_send)(self.raw, wr, &mut bad) };
        if code != 0 {
            return Err(CommError::verbs(
                "ibv_post_send",
                format!(
                    "{} (first failed WR {})",
                    errno_text(code),
                    if bad.is_null() {
                        "not reported".to_owned()
                    } else {
                        format!("wr_id {:#x}", unsafe { (*bad).wr_id })
                    }
                ),
            ));
        }
        Ok(())
    }

    /// Posts the receive work-request chain starting at `wr`.
    ///
    /// # Errors
    /// Returns [`CommError::Verbs`] when `ibv_post_recv` reports the first failed WR.
    pub(crate) fn post_recv(&self, wr: &mut IbvRecvWr) -> Result<(), CommError> {
        let mut bad: *mut IbvRecvWr = ptr::null_mut();
        // SAFETY: `raw` is a live QP whose RQ has room; `wr` is a fully initialised chain.
        let code = unsafe { (self.fns.post_recv)(self.raw, wr, &mut bad) };
        if code != 0 {
            return Err(CommError::verbs(
                "ibv_post_recv",
                format!(
                    "{} (first failed WR {})",
                    errno_text(code),
                    if bad.is_null() {
                        "not reported".to_owned()
                    } else {
                        format!("wr_id {:#x}", unsafe { (*bad).wr_id })
                    }
                ),
            ));
        }
        Ok(())
    }
}

impl Drop for Qp {
    fn drop(&mut self) {
        // SAFETY: owns the QP and drops once, before the CQs and PD it references.
        unsafe { (self.fns.destroy_qp)(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether this host can dlopen libibverbs, probed without opening a device.
    fn libibverbs_is_present() -> bool {
        for name in [&b"libibverbs.so.1\0"[..], &b"libibverbs.so\0"[..]] {
            // SAFETY: NUL-terminated name; the probe closes the handle it opened.
            let handle =
                unsafe { libc::dlopen(name.as_ptr().cast(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
            if !handle.is_null() {
                // SAFETY: the handle came from dlopen and is closed exactly once here.
                unsafe { libc::dlclose(handle) };
                return true;
            }
        }
        false
    }

    #[test]
    fn the_load_path_matches_what_this_host_provides() {
        let library_present = libibverbs_is_present();
        let shim_is_system = IBVERBS_SHIM == Some("system");
        match Verbs::load() {
            Ok(verbs) => {
                assert!(
                    library_present,
                    "load succeeded without dlopen finding libibverbs"
                );
                assert!(
                    shim_is_system,
                    "load succeeded but build.rs recorded the stub shim: {IBVERBS_SHIM:?}"
                );
                let names = verbs
                    .device_names()
                    .unwrap_or_else(|error| panic!("{error}"));
                assert!(names.iter().all(|name| !name.is_empty()));
            }
            Err(CommError::Unsupported(text)) => {
                if !library_present {
                    assert!(text.contains("libibverbs"), "{text}");
                } else if !shim_is_system {
                    assert!(
                        text.contains("verbs.h") || text.contains("ENOSYS"),
                        "the refusal must name the missing header when the shim is a stub: {text}"
                    );
                } else {
                    panic!("load failed on a host with libibverbs and a system shim: {text}");
                }
            }
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn the_inline_shim_mode_is_recorded() {
        match IBVERBS_SHIM {
            Some("system") | Some("stub") => {}
            other => {
                panic!("build.rs set SPARKFLASH_IBVERBS_SHIM={other:?}; expected system or stub")
            }
        }
    }

    #[test]
    fn errno_text_names_the_os_error() {
        assert!(errno_text(13).contains("errno 13"));
        assert!(errno_text(13).contains("Permission denied"));
        assert_eq!(errno_text(-5), "return code -5");
        assert!(matches!(
            verbs_error("ibv_post_send", 12),
            CommError::Verbs {
                call: "ibv_post_send",
                ..
            }
        ));
    }
}
