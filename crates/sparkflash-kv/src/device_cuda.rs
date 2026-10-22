//! The CUDA driver backend for the device KV store, behind the crate's cuda feature.
//!
//! Nothing here computes: it allocates device memory, copies bytes in and out, frees it, and
//! reports the driver's free-memory figure so a run can be measured against the store's own
//! accounting. No kernel is loaded or launched. The driver is resolved through dlopen, so no
//! process links libcuda; this is the same invariant docs/archive/build/device-path-design.md records
//! for the project's other device probes.
//!
//! The primary context is retained with CU_CTX_LMEM_RESIZE_TO_MAX (0x10), the flag the existing
//! probe needed on this unified-memory node: without it the driver pre-reserves the maximum
//! local memory for the context, which is the difference between a context and
//! CUDA_ERROR_OUT_OF_MEMORY.

use std::ffi::c_void;

use crate::device_store::{STORE_BUFFER_ALIGNMENT, StoreError, StoreMemory};

type CuResult = i32;
type CuDeviceptr = u64;

/// One device allocation owned by the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    /// Device address returned by cuMemAlloc_v2.
    pub address: CuDeviceptr,
    /// Bytes requested.
    pub bytes: usize,
}

/// The CUDA driver-backed [StoreMemory] implementation.
pub struct CudaMemory {
    lib: libloading::Library,
    device: i32,
    name: String,
    allocated: u64,
    allocations: usize,
}

impl CudaMemory {
    /// Loads the driver, initialises it, and makes the primary context current.
    ///
    /// # Errors
    /// Returns [StoreError::Backend] with the driver's own result name when any step fails.
    pub fn load() -> Result<Self, StoreError> {
        // SAFETY: the symbols below have the driver API's fixed C signatures; the library stays
        // alive for the lifetime of this value because every resolved symbol borrows from it.
        unsafe {
            let lib = libloading::Library::new("libcuda.so.1")
                .map_err(|error| StoreError::Backend(format!("libcuda.so.1: {error}")))?;
            let init: libloading::Symbol<unsafe extern "C" fn(u32) -> CuResult> = lib
                .get(b"cuInit")
                .map_err(|error| StoreError::Backend(format!("cuInit: {error}")))?;
            let result = init(0);
            if result != 0 {
                return Err(StoreError::Backend(driver_error("cuInit", result)));
            }
            let mut count: i32 = 0;
            let get_count: libloading::Symbol<unsafe extern "C" fn(*mut i32) -> CuResult> = lib
                .get(b"cuDeviceGetCount")
                .map_err(|error| StoreError::Backend(format!("cuDeviceGetCount: {error}")))?;
            let result = get_count(&mut count);
            if result != 0 || count <= 0 {
                return Err(StoreError::Backend(format!(
                    "no CUDA device: {} devices, {}",
                    count,
                    driver_error("cuDeviceGetCount", result)
                )));
            }
            let mut device: i32 = 0;
            let get: libloading::Symbol<unsafe extern "C" fn(*mut i32, i32) -> CuResult> = lib
                .get(b"cuDeviceGet")
                .map_err(|error| StoreError::Backend(format!("cuDeviceGet: {error}")))?;
            let result = get(&mut device, 0);
            if result != 0 {
                return Err(StoreError::Backend(driver_error("cuDeviceGet", result)));
            }
            let mut name_bytes = [0i8; 128];
            let get_name: libloading::Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> CuResult> =
                lib.get(b"cuDeviceGetName")
                    .map_err(|error| StoreError::Backend(format!("cuDeviceGetName: {error}")))?;
            let result = get_name(name_bytes.as_mut_ptr(), 128, device);
            let name = if result == 0 {
                std::ffi::CStr::from_ptr(name_bytes.as_ptr().cast())
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::from("unknown device")
            };
            let set_flags: libloading::Symbol<unsafe extern "C" fn(i32, u32) -> CuResult> =
                lib.get(b"cuDevicePrimaryCtxSetFlags").map_err(|error| {
                    StoreError::Backend(format!("cuDevicePrimaryCtxSetFlags: {error}"))
                })?;
            let _ = set_flags(device, 0x10);
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let retain: libloading::Symbol<
                unsafe extern "C" fn(*mut *mut c_void, i32) -> CuResult,
            > = lib.get(b"cuDevicePrimaryCtxRetain").map_err(|error| {
                StoreError::Backend(format!("cuDevicePrimaryCtxRetain: {error}"))
            })?;
            let result = retain(&mut ctx, device);
            if result != 0 {
                return Err(StoreError::Backend(driver_error(
                    "cuDevicePrimaryCtxRetain",
                    result,
                )));
            }
            let set: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> CuResult> = lib
                .get(b"cuCtxSetCurrent")
                .map_err(|error| StoreError::Backend(format!("cuCtxSetCurrent: {error}")))?;
            let result = set(ctx);
            if result != 0 {
                return Err(StoreError::Backend(driver_error("cuCtxSetCurrent", result)));
            }
            Ok(Self {
                lib,
                device,
                name,
                allocated: 0,
                allocations: 0,
            })
        }
    }

    /// The device's reported name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.name
    }

    /// The alignment every allocation satisfies.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        STORE_BUFFER_ALIGNMENT
    }

    /// The driver's free device memory, in bytes (cuMemGetInfo_v2).
    ///
    /// # Errors
    /// [StoreError::Backend] when the driver call fails.
    pub fn free_bytes(&self) -> Result<u64, StoreError> {
        let (free, _total) = self.memory_info()?;
        Ok(free)
    }

    /// The driver's total device memory, in bytes (cuMemGetInfo_v2).
    ///
    /// # Errors
    /// [StoreError::Backend] when the driver call fails.
    pub fn total_bytes(&self) -> Result<u64, StoreError> {
        let (_free, total) = self.memory_info()?;
        Ok(total)
    }

    /// Number of live allocations.
    #[must_use]
    pub fn allocations(&self) -> usize {
        self.allocations
    }

    fn memory_info(&self) -> Result<(u64, u64), StoreError> {
        // SAFETY: cuMemGetInfo_v2 writes two u64s; the symbol borrows the library, which is alive.
        unsafe {
            let symbol: libloading::Symbol<unsafe extern "C" fn(*mut u64, *mut u64) -> CuResult> =
                self.lib
                    .get(b"cuMemGetInfo_v2")
                    .map_err(|error| StoreError::Backend(format!("cuMemGetInfo_v2: {error}")))?;
            let mut free = 0u64;
            let mut total = 0u64;
            let result = symbol(&mut free, &mut total);
            if result != 0 {
                return Err(StoreError::Backend(driver_error("cuMemGetInfo_v2", result)));
            }
            Ok((free, total))
        }
    }
}

impl Drop for CudaMemory {
    fn drop(&mut self) {
        // SAFETY: releasing the primary context this value retained is the matching call. The
        // return code is ignored on drop: there is no recovery path and the process is exiting
        // the store's lifetime.
        unsafe {
            if let Ok(release) = self
                .lib
                .get::<unsafe extern "C" fn(i32) -> CuResult>(b"cuDevicePrimaryCtxRelease_v2")
            {
                let _ = release(self.device);
            }
        }
    }
}

impl StoreMemory for CudaMemory {
    type Buffer = DeviceBuffer;

    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer, StoreError> {
        // SAFETY: cuMemAlloc_v2 takes a pointer to receive the address and a size; both are valid.
        unsafe {
            let symbol: libloading::Symbol<
                unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult,
            > = self
                .lib
                .get(b"cuMemAlloc_v2")
                .map_err(|error| StoreError::Backend(format!("cuMemAlloc_v2: {error}")))?;
            let mut address: CuDeviceptr = 0;
            let result = symbol(&mut address, bytes);
            if result != 0 {
                return Err(StoreError::Backend(driver_error("cuMemAlloc_v2", result)));
            }
            self.allocated += bytes as u64;
            self.allocations += 1;
            Ok(DeviceBuffer { address, bytes })
        }
    }

    fn release(&mut self, buffer: Self::Buffer) {
        // SAFETY: the buffer was returned by cuMemAlloc_v2 and is released exactly once by the
        // store's drop. A failure here cannot be reported from release; it is counted only.
        unsafe {
            if let Ok(symbol) = self
                .lib
                .get::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                let _ = symbol(buffer.address);
            }
        }
        self.allocated = self.allocated.saturating_sub(buffer.bytes as u64);
        self.allocations = self.allocations.saturating_sub(1);
    }

    fn write(
        &mut self,
        buffer: &Self::Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), StoreError> {
        let start = usize::try_from(offset).map_err(|_| StoreError::TooLarge)?;
        let end = start.checked_add(bytes.len()).ok_or(StoreError::TooLarge)?;
        if end > buffer.bytes {
            return Err(StoreError::OutOfBounds {
                what: "backend write",
                value: end as u64,
                bound: buffer.bytes as u64,
            });
        }
        // SAFETY: cuMemcpyHtoD_v2 copies from a host slice into device memory; the range was
        // checked against the allocation above.
        unsafe {
            let symbol: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, *const c_void, usize) -> CuResult,
            > = self
                .lib
                .get(b"cuMemcpyHtoD_v2")
                .map_err(|error| StoreError::Backend(format!("cuMemcpyHtoD_v2: {error}")))?;
            let result = symbol(buffer.address + offset, bytes.as_ptr().cast(), bytes.len());
            if result != 0 {
                return Err(StoreError::Backend(driver_error("cuMemcpyHtoD_v2", result)));
            }
        }
        Ok(())
    }

    fn read(&self, buffer: &Self::Buffer, offset: u64, out: &mut [u8]) -> Result<(), StoreError> {
        let start = usize::try_from(offset).map_err(|_| StoreError::TooLarge)?;
        let end = start.checked_add(out.len()).ok_or(StoreError::TooLarge)?;
        if end > buffer.bytes {
            return Err(StoreError::OutOfBounds {
                what: "backend read",
                value: end as u64,
                bound: buffer.bytes as u64,
            });
        }
        // SAFETY: cuMemcpyDtoH_v2 copies from device memory into the host slice; the range was
        // checked against the allocation above.
        unsafe {
            let symbol: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize) -> CuResult,
            > = self
                .lib
                .get(b"cuMemcpyDtoH_v2")
                .map_err(|error| StoreError::Backend(format!("cuMemcpyDtoH_v2: {error}")))?;
            let result = symbol(out.as_mut_ptr().cast(), buffer.address + offset, out.len());
            if result != 0 {
                return Err(StoreError::Backend(driver_error("cuMemcpyDtoH_v2", result)));
            }
        }
        Ok(())
    }

    fn address(&self, buffer: &Self::Buffer) -> u64 {
        buffer.address
    }

    fn allocated_bytes(&self) -> u64 {
        self.allocated
    }
}

fn driver_error(call: &str, code: CuResult) -> String {
    let label = match code {
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
    format!("{call} -> {code} ({label})")
}
