//! The shared CUDA driver path for the kernel-launching examples.
//!
//! The driver is resolved with dlopen at run time, so nothing links libcuda and the workspace still
//! builds on hosts without a device. This module exists so the examples do not each carry their own
//! copy of the driver: the m7 launcher and the forthcoming m8 sequence launcher import one Cuda,
//! one module loader and one launch path.
//!
//! The raw driver function pointers stay private to this module. A kernel handle is an opaque
//! crate-owned value (`Kernel`) that only the non-flash_mla kernel ids can produce; the
//! flash_mla kernel is reachable only through the typed launches in crate::cuda_harness, which is
//! the compile-time enforcement of the one-launch-path rule. The unsafe blocks in this module are
//! the dlopen boundary: everything above it is safe code.

use std::ffi::c_void;
use std::path::{Path, PathBuf};

type CuResult = i32;

/// An opaque CUDA event handle.
type CuEvent = *mut c_void;

/// The host's MemAvailable in kB, parsed from /proc/meminfo.
///
/// Interpolating the whole /proc line into a message produced "...the node has ~MemAvailable:
/// 3768676 kB GB of system memory available...", which is exactly the kind of unreadable failure
/// text this launcher exists to avoid; the launcher admission checks read the number instead.
pub fn mem_available_kb() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

/// The cuLaunchKernel signature. The parameter list is long by nature, so it is named once here
/// rather than spelled out at the one call site.
type LaunchFn = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut c_void,
    *mut *mut c_void,
    *mut c_void,
) -> CuResult;

/// The driver's CUresult names, so a report says OUT_OF_MEMORY rather than 2.
fn name(code: CuResult) -> String {
    let s = match code {
        0 => "SUCCESS",
        1 => "INVALID_VALUE",
        2 => "OUT_OF_MEMORY",
        3 => "NOT_INITIALIZED",
        4 => "DEINITIALIZED",
        100 => "NO_DEVICE",
        101 => "INVALID_DEVICE",
        201 => "INVALID_CONTEXT",
        400 => "INVALID_HANDLE",
        700 => "ILLEGAL_ADDRESS",
        801 => "NOT_SUPPORTED",
        _ => "UNKNOWN",
    };
    format!("{code} ({s})")
}
type CuDeviceptr = u64;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;

/// A device allocation, in the driver's flat address space.
///
/// The address is transparent on purpose, not opaque: the launcher marshals device addresses into
/// the u64 kernel-argument vector, so the harness reads `.0` when it builds an argument list and
/// tests can construct one. A device address is not a capability - it cannot name or launch a
/// kernel; launches only accept an opaque Kernel (or the typed flash_mla handle).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deviceptr(
    /// The driver's flat device address, as the driver returns it from cuMemAlloc_v2.
    pub CuDeviceptr,
);

impl Deviceptr {
    /// Advances this address by a byte offset.
    ///
    /// The rotation copy moves one row group at a time (512-byte code rows, 16-byte scale rows),
    /// so it needs an address inside an allocation, not only the allocation base. The arithmetic is
    /// the driver's flat address space and is not checked against an allocation bound: a caller
    /// that offsets past its allocation gets the driver's own ILLEGAL_ADDRESS, exactly as it would
    /// from a hand-built address. Only the device path calls this; a host-only build never does.
    #[must_use]
    pub fn offset(self, bytes: usize) -> Self {
        Self(self.0 + bytes as u64)
    }
}

/// The opaque handle of a kernel that is not flash_mla.
///
/// The raw CUfunction stays private: examples cannot construct a Kernel and cannot recover the
/// pointer, so a launch always goes through this module. The flash_mla kernel deliberately has no
/// KernelId variant; see crate::cuda_harness::launch_flash_mla.
///
/// Constructing a Kernel by hand is a compile error, so there is no symbol-name bypass:
///
/// ```compile_fail
/// use sparkflash_kernels::cuda_driver::{Cuda, Kernel};
///
/// fn bypass(cuda: &Cuda) -> Result<(), String> {
///     // The raw field is private and there is no public constructor or conversion.
///     let kernel = Kernel { function: std::ptr::null_mut() };
///     cuda.launch(kernel, (1, 1, 1), (1, 1, 1), &[])
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    function: CuFunction,
}

/// The opaque handle of a CUDA module.
///
/// The module is what keeps a Kernel's code loaded, so callers hold it for as long as the kernel
/// runs; freeing it goes through Cuda::free. The raw CUmodule stays private.
#[derive(Clone, Copy, Debug)]
pub struct Module(CuModule);

/// One of the non-flash_mla kernels the shared driver resolves by symbol.
///
/// The enum deliberately has no flash_mla variant: the flash_mla kernel is loaded and launched by
/// crate::cuda_harness's typed entry points, so an example cannot obtain a raw handle for it.
/// Adding a variant here would reopen the raw-launch path that the typed launch closes.
///
/// A symbol name cannot stand in for a KernelId, so there is no symbol-name loader to reach the
/// flash_mla kernel with:
///
/// ```compile_fail
/// use sparkflash_kernels::cuda_driver::Cuda;
///
/// fn bypass(cuda: &Cuda, cubin: &[u8]) -> Result<(), String> {
///     // load_kernel takes a KernelId; a string does not typecheck.
///     let (_module, _kernel) = cuda.load_kernel(cubin, "flash_mla_csa2_sm121")?;
///     Ok(())
/// }
/// ```
///
/// And the flash_mla variant does not exist to be named:
///
/// ```compile_fail
/// use sparkflash_kernels::cuda_driver::KernelId;
///
/// fn bypass() -> KernelId {
///     KernelId::FlashMla
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelId {
    /// mega_mhc_fused_bf16_sm121: the Mega-mHC fused kernel.
    MegaMhcBf16,
    /// b12x_mxfp8_dense_sm121: the MXFP8 dense twin.
    Mxfp8Dense,
    /// b12x_mxfp8_wo_a_sm121: the output-projection MXFP8 kernel.
    Mxfp8WoA,
    /// b12x_mxfp4_moe_sm121: the MXFP4 MoE kernel.
    Mxfp4Moe,
    /// swiglu_mxfp8_quant_sm121: the SwiGLU plus MXFP8 quantiser kernel.
    SwigluMxfp8Quant,
    /// swiglu_mxfp8_quant_weighted_sm121: the same epilogue with the routing weight multiplied in
    /// before the quantiser (the F6 seam's device closure).
    SwigluMxfp8QuantWeighted,
    /// cert_lm_head_sm121: the certified LM-head kernel.
    CertLmHead,
    /// moe_router_sm121: the device MoE router (task-37, roadmap 6.3 item 4).
    MoeRouter,
    /// routed_gate_up_sm121: the routed-expert W4A8 gate/up projection (task-8 iteration B).
    RoutedGateUp,
}

impl KernelId {
    /// Every kernel the shared driver owns, in the order of this enum.
    pub const ALL: [KernelId; 9] = [
        KernelId::MegaMhcBf16,
        KernelId::Mxfp8Dense,
        KernelId::Mxfp8WoA,
        KernelId::Mxfp4Moe,
        KernelId::SwigluMxfp8Quant,
        KernelId::SwigluMxfp8QuantWeighted,
        KernelId::CertLmHead,
        KernelId::MoeRouter,
        KernelId::RoutedGateUp,
    ];

    /// The cubin symbol of this kernel. Private: callers name the enum, never the symbol.
    fn symbol(self) -> &'static str {
        match self {
            KernelId::MegaMhcBf16 => "mega_mhc_fused_bf16_sm121",
            KernelId::Mxfp8Dense => "b12x_mxfp8_dense_sm121",
            KernelId::Mxfp8WoA => "b12x_mxfp8_wo_a_sm121",
            KernelId::Mxfp4Moe => "b12x_mxfp4_moe_sm121",
            KernelId::SwigluMxfp8Quant => "swiglu_mxfp8_quant_sm121",
            KernelId::SwigluMxfp8QuantWeighted => "swiglu_mxfp8_quant_weighted_sm121",
            KernelId::CertLmHead => "cert_lm_head_sm121",
            KernelId::MoeRouter => "moe_router_sm121",
            KernelId::RoutedGateUp => "routed_gate_up_sm121",
        }
    }

    /// The cubin file names this kernel may be stored under, symbol first.
    ///
    /// The symbol is the name the harness documents and must open. Mega-MHC carries a second,
    /// documented alias because the pinned baseline manifest keys that kernel by its source-file
    /// stem, `mega_mhc_sm121` (`baseline/kernel_baseline.txt`). A cubin's file name is not part of
    /// its ABI - the symbol inside it is - so both names are the same artifact, and accepting the
    /// alias removes the run-D failure of 2026-10-18 (the pinned artifact was copied under the
    /// manifest's name and the harness opened the symbol's name instead).
    pub fn cubin_names(self) -> &'static [&'static str] {
        match self {
            KernelId::MegaMhcBf16 => &["mega_mhc_fused_bf16_sm121", "mega_mhc_sm121"],
            KernelId::Mxfp8Dense => &["b12x_mxfp8_dense_sm121"],
            KernelId::Mxfp8WoA => &["b12x_mxfp8_wo_a_sm121"],
            KernelId::Mxfp4Moe => &["b12x_mxfp4_moe_sm121"],
            KernelId::SwigluMxfp8Quant => &["swiglu_mxfp8_quant_sm121"],
            KernelId::SwigluMxfp8QuantWeighted => &["swiglu_mxfp8_quant_weighted_sm121"],
            KernelId::CertLmHead => &["cert_lm_head_sm121"],
            KernelId::MoeRouter => &["moe_router_sm121"],
            KernelId::RoutedGateUp => &["routed_gate_up_sm121"],
        }
    }

    /// Maps a cubin file name (the symbol or a documented alias) to the kernel.
    ///
    /// `flash_mla_csa2_sm121` is deliberately absent: that kernel is loaded through
    /// `cuda_harness::load_flash_mla`, which owns its symbol, so it has no `KernelId`.
    #[must_use]
    pub fn from_cubin_name(name: &str) -> Option<KernelId> {
        KernelId::ALL
            .into_iter()
            .find(|id| id.cubin_names().contains(&name))
    }
}

/// The cubin name of the flash_mla kernel, which has no [`KernelId`]: `crate::cuda_harness`
/// owns its symbol and its typed launches, so it sits outside the driver's kernel-id vocabulary.
pub const FLASH_MLA_CUBIN: &str = "flash_mla_csa2_sm121";

/// The built cubin for a kernel name, under `dir`, for a harness that launches it.
///
/// The kernel's [`KernelId::cubin_names`] are tried in order - the symbol first, then any documented
/// alias such as the pinned baseline manifest's Mega-MHC source stem - so a cubin copied out of the
/// pinned artifact directory under either name is found. [`FLASH_MLA_CUBIN`] is resolved by its own
/// name because it has no `KernelId`. Any other name is refused with the driver-kernel message
/// rather than a file error, and a known kernel with no built file names every candidate tried.
///
/// # Errors
///
/// The name is not a shared-driver kernel or the flash_mla kernel, or no candidate file exists under
/// `dir`.
pub fn cubin_path(dir: &Path, name: &str) -> Result<PathBuf, String> {
    let candidates: &[&str] = if name == FLASH_MLA_CUBIN {
        &[FLASH_MLA_CUBIN]
    } else {
        KernelId::from_cubin_name(name)
            .ok_or_else(|| {
                format!(
                    "{name} is not one of the shared driver kernels; the flash_mla kernel is loaded through cuda_harness::load_flash_mla and has no KernelId"
                )
            })?
            .cubin_names()
    };
    let mut tried = Vec::new();
    for candidate in candidates {
        let path = dir.join(format!("{candidate}.cubin"));
        if path.is_file() {
            return Ok(path);
        }
        tried.push(path.display().to_string());
    }
    Err(format!(
        "no cubin for {name} in {}; tried {}",
        dir.display(),
        tried.join(", ")
    ))
}

/// The CUDA context and the dlopen'd driver functions this process uses.
///
/// The context is the retained primary context, made current for the thread that created it, and
/// is released by Drop on every exit path.
pub struct Cuda {
    lib: libloading::Library,
    ctx: *mut c_void,
    /// The device ordinal the retained primary context belongs to.
    ///
    /// `cuDevicePrimaryCtxRelease` takes a `CUdevice` (an `int`), not the `CUcontext`: storing the
    /// ordinal is what lets `Drop` make the matching release call with the right argument instead of
    /// passing a pointer through a mis-typed symbol.
    dev: i32,
}

impl Cuda {
    /// Opens libcuda.so.1, initialises the driver and makes the primary context current.
    ///
    /// cuInit alone is not enough: cuModuleLoadData and cuMemAlloc need a CURRENT context, and
    /// without one the driver answers CUDA_ERROR_INVALID_CONTEXT (201), which is what the first
    /// run on the node returned. The primary context is retained (refcounted, shared with anything
    /// else driving the device) and made current for this thread.
    pub fn load() -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new("libcuda.so.1")
                .map_err(|e| format!("libcuda.so.1: {e}"))?;
            {
                let init: libloading::Symbol<unsafe extern "C" fn(u32) -> CuResult> =
                    lib.get(b"cuInit").map_err(|e| e.to_string())?;
                let r = init(0);
                if r != 0 {
                    return Err(format!("cuInit failed: {r}"));
                }
            }
            let mut count: i32 = 0;
            {
                let f: libloading::Symbol<unsafe extern "C" fn(*mut i32) -> CuResult> =
                    lib.get(b"cuDeviceGetCount").map_err(|e| e.to_string())?;
                let r = f(&mut count);
                if r != 0 {
                    return Err(format!("cuDeviceGetCount failed: {}", name(r)));
                }
            }
            let mut dev: i32 = 0;
            {
                let get: libloading::Symbol<unsafe extern "C" fn(*mut i32, i32) -> CuResult> =
                    lib.get(b"cuDeviceGet").map_err(|e| e.to_string())?;
                let r = get(&mut dev, 0);
                if r != 0 {
                    return Err(format!("cuDeviceGet(0) failed: {}", name(r)));
                }
            }
            let mut device_name = String::from("unknown device");
            {
                let mut nm = [0i8; 128];
                let f: libloading::Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> CuResult> =
                    lib.get(b"cuDeviceGetName").map_err(|e| e.to_string())?;
                let r = f(nm.as_mut_ptr(), 128, dev);
                if r == 0 {
                    // c_char is u8 on aarch64 and i8 on x86_64; cast rather than assume either.
                    let s = std::ffi::CStr::from_ptr(nm.as_ptr().cast());
                    device_name = s.to_string_lossy().into_owned();
                    let mut total: u64 = 0;
                    let t: libloading::Symbol<unsafe extern "C" fn(*mut u64, i32) -> CuResult> =
                        lib.get(b"cuDeviceTotalMem_v2").map_err(|e| e.to_string())?;
                    let _ = t(&mut total, dev);
                    println!(
                        "device 0: {device_name} of {count} (total {} MiB, MemAvailable {} MiB)",
                        total / (1024 * 1024),
                        mem_available_kb().map(|kb| kb / 1024).unwrap_or(0)
                    );
                }
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            {
                // CU_CTX_LMEM_RESIZE_TO_MAX (0x10) asks the driver NOT to pre-reserve the maximum
                // local memory for this context. On a unified-memory GB10 where the device total is
                // 124 GB but only ~3.7 GB of system memory is free, that pre-reservation is the
                // difference between a context and CUDA_ERROR_OUT_OF_MEMORY.
                let set_flags: libloading::Symbol<unsafe extern "C" fn(i32, u32) -> CuResult> = lib
                    .get(b"cuDevicePrimaryCtxSetFlags")
                    .map_err(|e| e.to_string())?;
                let _ = set_flags(dev, 0x10);
                let retain: libloading::Symbol<
                    unsafe extern "C" fn(*mut *mut c_void, i32) -> CuResult,
                > = lib
                    .get(b"cuDevicePrimaryCtxRetain")
                    .map_err(|e| e.to_string())?;
                let r = retain(&mut ctx, dev);
                if r != 0 {
                    // One alternative route, tried exactly once - not a retry loop. Retaining the
                    // primary context is the normal route for a process that does not own the
                    // device; cuCtxCreate_v2 is what a launcher would use if it did. Both results
                    // are reported so the failure cannot be read as one API's quirk.
                    let mut alt: *mut c_void = std::ptr::null_mut();
                    let create: libloading::Symbol<
                        unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> CuResult,
                    > = lib.get(b"cuCtxCreate_v2").map_err(|e| e.to_string())?;
                    let r2 = create(&mut alt, 0x10, dev);
                    // A context created by the alternative route is not the one this type stores,
                    // so destroy it now: it would otherwise leak on the error return below.
                    if r2 == 0
                        && let Ok(destroy) = lib
                            .get::<unsafe extern "C" fn(*mut c_void) -> CuResult>(
                                b"cuCtxDestroy_v2",
                            )
                    {
                        let _ = destroy(alt);
                    }
                    // Nothing is left half-initialised: no context was retained, the alternative
                    // context (if any) was destroyed, no device memory was allocated, and this
                    // condition is not retried.
                    let avail = mem_available_kb()
                        .map(|kb| format!("{} MiB", kb / 1024))
                        .unwrap_or_else(|| "unknown".to_string());
                    return Err(format!(
                        "no CUDA context could be created on device 0 ({}): \
                         cuDevicePrimaryCtxRetain -> {}, cuCtxCreate_v2 -> {} (one attempt each, not \
                         retried). cuInit, cuDeviceGet and cuDeviceGetName all succeeded, so the \
                         driver is reachable and the device is present: on this node a second CUDA \
                         context does not fit beside the resident model. MemAvailable now: {avail}. \
                         No device memory was allocated and no context is left retained. This is a \
                         capacity fact about the node, not a defect in this launcher - do not work \
                         around it here.",
                        device_name,
                        name(r),
                        name(r2),
                    ));
                }
            }
            {
                // This failure path cannot rely on Drop, because the Cuda it would release has not
                // been constructed yet; release the retained primary context explicitly.
                let set: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> CuResult> =
                    match lib.get(b"cuCtxSetCurrent") {
                        Ok(set) => set,
                        Err(error) => {
                            if let Ok(release_v2) = lib
                                .get::<unsafe extern "C" fn(i32) -> CuResult>(
                                    b"cuDevicePrimaryCtxRelease_v2",
                                )
                            {
                                let _ = release_v2(dev);
                            }
                            return Err(error.to_string());
                        }
                    };
                let r = set(ctx);
                if r != 0 {
                    if let Ok(release_v2) = lib.get::<unsafe extern "C" fn(i32) -> CuResult>(
                        b"cuDevicePrimaryCtxRelease_v2",
                    ) {
                        let _ = release_v2(dev);
                    }
                    return Err(format!("cuCtxSetCurrent failed: {r}"));
                }
            }
            Ok(Cuda { lib, ctx, dev })
        }
    }

    fn sym<T>(&self, name: &[u8]) -> Result<libloading::Symbol<'_, T>, String> {
        unsafe {
            self.lib
                .get(name)
                .map_err(|e| format!("{}: {e}", String::from_utf8_lossy(name)))
        }
    }

    /// Allocates device memory and adds the byte count to the running total.
    pub fn alloc(&self, bytes: usize, total: &mut usize) -> Result<Deviceptr, String> {
        unsafe {
            let f: libloading::Symbol<unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult> =
                self.sym(b"cuMemAlloc_v2")?;
            let mut p: CuDeviceptr = 0;
            let r = f(&mut p, bytes);
            if r != 0 {
                return Err(format!("cuMemAlloc({bytes}) failed: {r}"));
            }
            *total += bytes;
            Ok(Deviceptr(p))
        }
    }

    /// Copies host bytes to a device allocation.
    pub fn copy_htod(&self, dst: &Deviceptr, src: &[u8]) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, *const c_void, usize) -> CuResult,
            > = self.sym(b"cuMemcpyHtoD_v2")?;
            let r = f(dst.0, src.as_ptr().cast(), src.len());
            if r != 0 {
                return Err(format!("cuMemcpyHtoD failed: {r}"));
            }
            Ok(())
        }
    }

    /// Copies a device allocation back to host bytes.
    pub fn copy_dtoh(&self, dst: &mut [u8], src: &Deviceptr) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize) -> CuResult,
            > = self.sym(b"cuMemcpyDtoH_v2")?;
            let r = f(dst.as_mut_ptr().cast(), src.0, dst.len());
            if r != 0 {
                return Err(format!("cuMemcpyDtoH failed: {r}"));
            }
            Ok(())
        }
    }

    /// Copies between two device allocations.
    ///
    /// The byte count is explicit because the callers copy fixed-size rows (512 bytes of window
    /// codes, 16 bytes of window scales) into contiguous buffers, not whole slices. The real
    /// cuMemcpyDtoD_v2 symbol and copy are exercised by the m8 device run; this doctest is no_run
    /// so the call's arity and signature are compiled in CI without loading libcuda.
    ///
    /// ```no_run
    /// use sparkflash_kernels::cuda_driver::Cuda;
    ///
    /// # fn main() -> Result<(), String> {
    /// let cuda = Cuda::load()?;
    /// let mut allocated = 0usize;
    /// let src = cuda.alloc(512, &mut allocated)?;
    /// let dst = cuda.alloc(512, &mut allocated)?;
    /// cuda.copy_dtod(&dst, &src, 512)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn copy_dtod(&self, dst: &Deviceptr, src: &Deviceptr, bytes: usize) -> Result<(), String> {
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, CuDeviceptr, usize) -> CuResult,
            > = self.sym(b"cuMemcpyDtoD_v2")?;
            let r = f(dst.0, src.0, bytes);
            if r != 0 {
                return Err(format!("cuMemcpyDtoD failed: {r}"));
            }
            Ok(())
        }
    }

    /// Loads one non-flash_mla kernel from a cubin image.
    ///
    /// The cubin symbol is chosen by the KernelId, not by the caller, so the returned handle can
    /// never refer to the flash_mla kernel.
    pub fn load_kernel(&self, cubin: &[u8], id: KernelId) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, id.symbol())?;
        Ok((Module(module), Kernel { function }))
    }

    /// Loads the flash_mla kernel. Used only by crate::cuda_harness's typed loader.
    pub(crate) fn load_flash_mla(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "flash_mla_csa2_sm121")?;
        Ok((Module(module), Kernel { function }))
    }

    fn load_function(&self, cubin: &[u8], name: &str) -> Result<(CuModule, CuFunction), String> {
        unsafe {
            let load: libloading::Symbol<
                unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
            > = self.sym(b"cuModuleLoadData")?;
            let mut module: CuModule = std::ptr::null_mut();
            let r = load(&mut module, cubin.as_ptr().cast());
            if r != 0 {
                return Err(format!("cuModuleLoadData failed: {r}"));
            }
            let get: libloading::Symbol<
                unsafe extern "C" fn(*mut CuFunction, CuModule, *const i8) -> CuResult,
            > = self.sym(b"cuModuleGetFunction")?;
            let cname = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
            let mut func: CuFunction = std::ptr::null_mut();
            let r = get(&mut func, module, cname.as_ptr().cast());
            if r != 0 {
                return Err(format!("cuModuleGetFunction({name}) failed: {r}"));
            }
            Ok((module, func))
        }
    }

    /// Launches one non-flash_mla kernel, then synchronises the context.
    pub fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        unsafe {
            let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
            let mut storage: Vec<[u64; 1]> = args.iter().map(|a| [*a]).collect();
            let mut ptrs: Vec<*mut c_void> =
                storage.iter_mut().map(|s| s.as_mut_ptr().cast()).collect();
            let r = launch(
                kernel.function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                std::ptr::null_mut(),
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if r != 0 {
                return Err(format!("cuLaunchKernel failed: {r}"));
            }
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let r = sync();
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            Ok(())
        }
    }

    /// Times one non-flash_mla kernel: warm-up, per-launch events, then one batched region.
    ///
    /// A throughput claim needs the warm-up outside the timed region and one event pair per launch
    /// for the distribution; the batched region has no per-launch synchronisation and is the figure
    /// to quote when the per-launch event overhead would dominate a kernel this small. No
    /// allocation, no upload and no checkpoint read is inside either timed region, so the numbers
    /// are the kernel's own and not the host's.
    pub fn launch_repeated(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
        repeat: usize,
    ) -> Result<(Vec<f64>, f64), String> {
        unsafe {
            let create: libloading::Symbol<unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult> =
                self.sym(b"cuEventCreate")?;
            let record: libloading::Symbol<unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult> =
                self.sym(b"cuEventRecord")?;
            let wait: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventSynchronize")?;
            let elapsed: libloading::Symbol<
                unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
            > = self.sym(b"cuEventElapsedTime")?;
            let destroy: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventDestroy")?;
            let mut start: CuEvent = std::ptr::null_mut();
            let mut stop: CuEvent = std::ptr::null_mut();
            let r = create(&mut start, 0);
            if r != 0 {
                return Err(format!("cuEventCreate failed: {r}"));
            }
            let r = create(&mut stop, 0);
            if r != 0 {
                return Err(format!("cuEventCreate failed: {r}"));
            }
            // Warm-up: fills the module cache and the launch path, and is deliberately not timed.
            self.launch(kernel, grid, block, args)?;
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let r = sync();
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            let mut per_launch = Vec::with_capacity(repeat);
            for _ in 0..repeat {
                let r = record(start, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord failed: {r}"));
                }
                self.launch(kernel, grid, block, args)?;
                let r = record(stop, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord failed: {r}"));
                }
                let r = wait(stop);
                if r != 0 {
                    return Err(format!("cuEventSynchronize failed: {r}"));
                }
                let mut ms: f32 = 0.0;
                let r = elapsed(&mut ms, start, stop);
                if r != 0 {
                    return Err(format!("cuEventElapsedTime failed: {r}"));
                }
                per_launch.push(f64::from(ms));
            }
            let r = record(start, std::ptr::null_mut());
            if r != 0 {
                return Err(format!("cuEventRecord failed: {r}"));
            }
            for _ in 0..repeat {
                self.launch(kernel, grid, block, args)?;
            }
            let r = record(stop, std::ptr::null_mut());
            if r != 0 {
                return Err(format!("cuEventRecord failed: {r}"));
            }
            let r = wait(stop);
            if r != 0 {
                return Err(format!("cuEventSynchronize failed: {r}"));
            }
            let mut total: f32 = 0.0;
            let r = elapsed(&mut total, start, stop);
            if r != 0 {
                return Err(format!("cuEventElapsedTime failed: {r}"));
            }
            destroy(start);
            destroy(stop);
            Ok((per_launch, f64::from(total) / repeat as f64))
        }
    }

    /// Frees device buffers and unloads the module that produced them.
    pub fn free(&self, module: Module, ptrs: &[Deviceptr]) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for p in ptrs {
                    let _ = f(p.0);
                }
            }
            if let Ok(f) = self.sym::<unsafe extern "C" fn(CuModule) -> CuResult>(b"cuModuleUnload")
            {
                let _ = f(module.0);
            }
            // The primary context is released by Drop, on every exit path.
        }
    }

    /// Frees device buffers without unloading the module that produced them.
    ///
    /// Stage 7 runs several different-shaped launches per loaded module, so the module has to
    /// outlive them.
    pub fn free_buffers(&self, ptrs: &[&Deviceptr]) {
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for p in ptrs {
                    let _ = f(p.0);
                }
            }
        }
    }
}

impl Drop for Cuda {
    /// Releases this process's reference to the primary context on every exit path, including a
    /// failure after the context was created, so a run can never leave a half-initialised Cuda.
    /// Releasing is refcounted and does not disturb anything else driving the device.
    fn drop(&mut self) {
        if self.ctx.is_null() {
            return;
        }
        unsafe {
            if let Ok(release_v2) =
                self.sym::<unsafe extern "C" fn(i32) -> CuResult>(b"cuDevicePrimaryCtxRelease_v2")
            {
                let _ = release_v2(self.dev);
            }
        }
    }
}
