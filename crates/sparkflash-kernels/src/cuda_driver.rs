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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

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

/// An opaque CUDA stream handle.
type CuStream = *mut c_void;

/// An opaque CUDA graph handle.
type CuGraph = *mut c_void;

/// An opaque CUDA graph-executable handle.
type CuGraphExec = *mut c_void;

/// `CU_STREAM_NON_BLOCKING`: capture runs on a stream this handle creates, never on the legacy
/// default stream (which cannot be captured).
const CU_STREAM_NON_BLOCKING: u32 = 0x1;

/// `CU_STREAM_CAPTURE_MODE_GLOBAL`: record every launch issued on the capture stream.
const CU_STREAM_CAPTURE_MODE_GLOBAL: u32 = 0;

/// The launch path's graph state: no segment is active, launches go to the default stream.
const GRAPH_DIRECT: u8 = 0;
/// A capture is active: launches are recorded into a graph on the capture stream.
const GRAPH_CAPTURE: u8 = 1;
/// A record is active: launches are collected in host memory, not issued to the driver.
const GRAPH_RECORD: u8 = 2;

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
    /// The canonical cubin pin name, so a timing bracket can key its sample by kernel.
    name: &'static str,
}

/// Summary of a parallel direct-I/O file read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectReadReport {
    /// Total logical bytes read into the destination buffer.
    pub bytes_read: usize,
    /// Number of chunk reads issued across the worker pool.
    pub chunks: usize,
    /// Worker threads used.
    pub concurrency: usize,
    /// Chunk size in bytes.
    pub chunk_bytes: usize,
    /// Whether `O_DIRECT` remained active for the read.
    pub direct_io: bool,
}

/// Default chunk size for parallel direct I/O (16 MiB, a multiple of 4096).
pub const DEFAULT_DIRECT_CHUNK_BYTES: usize = 16 * 1024 * 1024;
/// Default number of concurrent reads in flight for parallel direct I/O.
pub const DEFAULT_DIRECT_CONCURRENCY: usize = 8;
/// Required alignment for `O_DIRECT` offsets, lengths, and buffer addresses.
pub const DIRECT_IO_ALIGN: usize = 4096;

/// Opens a file with `O_DIRECT` if supported by the filesystem, returning `(file, is_direct)`.
fn open_for_direct_read(path: &Path, want_direct: bool) -> Result<(std::fs::File, bool), String> {
    use std::os::unix::fs::OpenOptionsExt;
    if want_direct
        && let Ok(file) = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
    {
        return Ok((file, true));
    }
    let file =
        std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    Ok((file, false))
}

/// Reads `dst.len()` bytes starting at `file_offset` from `path` directly into `dst` using a
/// bounded thread pool issuing `pread` (`read_exact_at`) over `chunk_bytes` (8-16 MiB) chunks.
///
/// When `use_odirect` is true and the file offset, chunk length, and destination pointer are
/// 4096-byte aligned, the read goes straight from the NVMe `O_DIRECT` descriptor into `dst` with
/// zero intermediate buffers. If `O_DIRECT` is rejected by the filesystem (e.g. tmpfs in unit
/// tests) or on a non-aligned tail chunk, that chunk falls back to the buffered descriptor so the
/// read is always exact.
///
/// # Errors
/// Returns an error if the file cannot be opened or any chunk read fails or hits EOF.
pub fn read_file_parallel_into_slice(
    path: &Path,
    file_offset: u64,
    dst: &mut [u8],
    chunk_bytes: usize,
    concurrency: usize,
    use_odirect: bool,
) -> Result<DirectReadReport, String> {
    use std::os::unix::fs::FileExt;
    let total_len = dst.len();
    let chunk_bytes = chunk_bytes.max(DIRECT_IO_ALIGN);
    let chunk_bytes = (chunk_bytes / DIRECT_IO_ALIGN) * DIRECT_IO_ALIGN;
    let concurrency = concurrency.clamp(1, 32);
    if total_len == 0 {
        return Ok(DirectReadReport {
            bytes_read: 0,
            chunks: 0,
            concurrency,
            chunk_bytes,
            direct_io: false,
        });
    }
    let (direct_file, opened_direct) = open_for_direct_read(path, use_odirect)?;
    let buffered_file =
        std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let direct_used = AtomicBool::new(opened_direct);
    let mut chunks_vec: Vec<(usize, &mut [u8])> = Vec::new();
    let mut rel_offset = 0usize;
    for slice in dst.chunks_mut(chunk_bytes) {
        let len = slice.len();
        chunks_vec.push((rel_offset, slice));
        rel_offset += len;
    }
    let total_chunks = chunks_vec.len();
    let workers = concurrency.min(total_chunks);
    let queue = Mutex::new(chunks_vec.into_iter());
    let first_error: Mutex<Option<String>> = Mutex::new(None);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    if first_error
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .is_some()
                    {
                        break;
                    }
                    let next = queue.lock().unwrap_or_else(|p| p.into_inner()).next();
                    let Some((rel, slice)) = next else {
                        break;
                    };
                    let at = file_offset + rel as u64;
                    let ptr_addr = slice.as_ptr() as usize;
                    let aligned = (at as usize).is_multiple_of(DIRECT_IO_ALIGN)
                        && slice.len().is_multiple_of(DIRECT_IO_ALIGN)
                        && ptr_addr.is_multiple_of(DIRECT_IO_ALIGN);
                    let result = if opened_direct && aligned {
                        match direct_file.read_exact_at(slice, at) {
                            Ok(()) => Ok(()),
                            Err(_) => {
                                direct_used.store(false, Ordering::Relaxed);
                                buffered_file.read_exact_at(slice, at)
                            }
                        }
                    } else {
                        if slice.len() == chunk_bytes {
                            direct_used.store(false, Ordering::Relaxed);
                        }
                        buffered_file.read_exact_at(slice, at)
                    };
                    if let Err(error) = result {
                        let mut guard = first_error.lock().unwrap_or_else(|p| p.into_inner());
                        if guard.is_none() {
                            *guard = Some(format!(
                                "parallel pread {} at offset {at} ({} bytes): {error}",
                                path.display(),
                                slice.len()
                            ));
                        }
                        break;
                    }
                }
            });
        }
    });

    if let Some(err) = first_error.into_inner().unwrap_or_else(|p| p.into_inner()) {
        return Err(err);
    }
    Ok(DirectReadReport {
        bytes_read: total_len,
        chunks: total_chunks,
        concurrency: workers,
        chunk_bytes,
        direct_io: direct_used.load(Ordering::Relaxed),
    })
}

/// A page-locked host allocation with its UVA device mapping.
///
/// The host pointer is the allocation's owner; the device pointer is only a mapping into it and
/// must not be freed separately. [Cuda::free_host_mapped] releases the host allocation.
#[derive(Clone, Copy, Debug)]
pub struct HostMapped {
    host: *mut c_void,
    device: Deviceptr,
    len: usize,
}

// Safety: `HostMapped` points to a page-locked, portable CUDA host slab (`CU_MEMHOSTALLOC_PORTABLE`)
// whose address range `host..host+len` is valid across threads until `Cuda::free_host_mapped`.
unsafe impl Send for HostMapped {}
unsafe impl Sync for HostMapped {}

impl HostMapped {
    /// The device-visible pointer for kernel arguments.
    #[must_use]
    pub fn device(&self) -> Deviceptr {
        self.device
    }

    /// The host-visible pointer the caller fills before a launch.
    #[must_use]
    pub fn host_ptr(&self) -> *mut c_void {
        self.host
    }

    /// Length of the mapped host slab in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slab is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Writes `src` directly into the host-mapped slab at `offset`, with no separate H2D copy.
    ///
    /// # Errors
    /// Returns an error if `offset + src.len()` exceeds the slab length.
    pub fn write_at(&self, offset: usize, src: &[u8]) -> Result<(), String> {
        let end = offset
            .checked_add(src.len())
            .ok_or_else(|| format!("host-mapped write {offset}+{} overflows", src.len()))?;
        if end > self.len {
            return Err(format!(
                "host-mapped write {offset}..{end} exceeds allocation of {} bytes",
                self.len
            ));
        }
        if src.is_empty() {
            return Ok(());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                (self.host as *mut u8).add(offset),
                src.len(),
            );
        }
        Ok(())
    }

    /// Reads bytes from the host-mapped slab at `offset` into `dst`.
    ///
    /// # Errors
    /// Returns an error if `offset + dst.len()` exceeds the slab length.
    pub fn read_at(&self, offset: usize, dst: &mut [u8]) -> Result<(), String> {
        let end = offset
            .checked_add(dst.len())
            .ok_or_else(|| format!("host-mapped read {offset}+{} overflows", dst.len()))?;
        if end > self.len {
            return Err(format!(
                "host-mapped read {offset}..{end} exceeds allocation of {} bytes",
                self.len
            ));
        }
        if dst.is_empty() {
            return Ok(());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.host as *const u8).add(offset),
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        Ok(())
    }

    /// Streams `total_len` bytes from `path` (starting at `file_offset`) directly into this
    /// host-mapped slab at `arena_offset` using parallel `O_DIRECT` / `pread` chunks, with no
    /// intermediate `Vec<u8>` and no H2D copy.
    ///
    /// # Errors
    /// Returns an error if the range exceeds the slab or any file read fails.
    #[allow(clippy::too_many_arguments)]
    pub fn read_file_parallel(
        &self,
        path: &Path,
        file_offset: u64,
        arena_offset: usize,
        total_len: usize,
        chunk_bytes: usize,
        concurrency: usize,
        use_odirect: bool,
    ) -> Result<DirectReadReport, String> {
        let end = arena_offset
            .checked_add(total_len)
            .ok_or_else(|| format!("host-mapped file read {arena_offset}+{total_len} overflows"))?;
        if end > self.len {
            return Err(format!(
                "host-mapped file read {arena_offset}..{end} exceeds allocation of {} bytes",
                self.len
            ));
        }
        let dst = unsafe {
            std::slice::from_raw_parts_mut((self.host as *mut u8).add(arena_offset), total_len)
        };
        read_file_parallel_into_slice(
            path,
            file_offset,
            dst,
            chunk_bytes,
            concurrency,
            use_odirect,
        )
    }
}

/// A read-only `mmap` of a converted rank file registered with the CUDA driver for zero-copy
/// unified-memory access on GB10 (`cuMemHostRegister` + `cuMemHostGetDevicePointer_v2`).
#[derive(Debug)]
pub struct RegisteredMmap {
    mmap: memmap2::Mmap,
    host: *mut c_void,
    device: Deviceptr,
    len: usize,
    registered: bool,
}

impl RegisteredMmap {
    /// Device-visible base pointer for kernel arguments.
    #[must_use]
    pub fn device(&self) -> Deviceptr {
        self.device
    }

    /// Mapped byte slice on the host.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap[..self.len]
    }

    /// Length of the mapped region in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapped region is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
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
    /// b12x_mxfp4_moe_stored_sm121: the same rank-partial GEMM reading the released plan's stored
    /// 576-K rows (288 code bytes + 18 separate scale bytes per output row). Selected by plan
    /// layout; the incumbent above stays for the A8 pre-padded path.
    Mxfp4MoeStored,
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
    /// rmsnorm_sm121: the standalone final RMSNorm of the serving tail (BF16 in, BF16 out).
    RmsNorm,
    /// lm_head_reduce_sm121: the device block reduction and argmax over the four certified-head
    /// shard outputs.
    LmHeadReduce,
    /// flash_mla_csa2_seq_sm121: the deferred sequence's position-causal CSA2 attention entry, the
    /// start_pos > 0 path the c4a-2 call-site guard currently denies.
    FlashMlaCsa2Seq,
    /// flash_mla_csa2_ring_sm121: D49's ring-capacity CSA2 entry. Capacity is an explicit argument
    /// separate from the 128-token arithmetic window; the frozen sequence entry is not mutated.
    FlashMlaCsa2Ring,
    /// b12x_bf16a_mxfp8w_dense_sm121: the BF16-activation variant of the dense MXFP8 projection
    /// (D37's precision-first path); weights stay B12x FP8 codes and ue8m0 scales.
    Bf16aMxfp8wDense,
    /// b12x_bf16a_mxfp8w_wo_a_sm121: the BF16-activation variant of the grouped wo_a projection.
    Bf16aMxfp8wWoA,
    /// engram_embed_sm121: the Engram table row reader and activation quantiser (task-9, audit F4).
    EngramEmbed,
    /// engram_wkv_sm121: the Engram wkv projection, fp8 W8A8 (task-9, audit F4).
    EngramWkv,
    /// engram_gate_write_sm121: the Engram gate and residual write (task-9, audit F4).
    EngramGateWrite,
    /// routed_gate_up_swiglu_splitk_sm121: the split-K grouped routed gate/up projection with the
    /// weighted SwiGLU and the activation quantiser fused into one epilogue (D51 amendment A). A
    /// NEW symbol: the pinned nineteen-word flash_mla entry and every pinned kernel are untouched.
    /// Default-off at the launch site, which is the Lead's wiring; the kernel's ABI, tolerance row
    /// and pin checklist live in crate::routed_gate_up_swiglu_splitk_abi.
    RoutedGateUpSwigluSplitK,
}

impl KernelId {
    /// Every kernel the shared driver owns, in the order of this enum.
    pub const ALL: [KernelId; 20] = [
        KernelId::MegaMhcBf16,
        KernelId::Mxfp8Dense,
        KernelId::Mxfp8WoA,
        KernelId::Mxfp4Moe,
        KernelId::Mxfp4MoeStored,
        KernelId::SwigluMxfp8Quant,
        KernelId::SwigluMxfp8QuantWeighted,
        KernelId::CertLmHead,
        KernelId::MoeRouter,
        KernelId::RoutedGateUp,
        KernelId::RmsNorm,
        KernelId::LmHeadReduce,
        KernelId::FlashMlaCsa2Seq,
        KernelId::FlashMlaCsa2Ring,
        KernelId::Bf16aMxfp8wDense,
        KernelId::Bf16aMxfp8wWoA,
        KernelId::EngramEmbed,
        KernelId::EngramWkv,
        KernelId::EngramGateWrite,
        KernelId::RoutedGateUpSwigluSplitK,
    ];

    /// The cubin symbol of this kernel. Private: callers name the enum, never the symbol.
    fn symbol(self) -> &'static str {
        match self {
            KernelId::MegaMhcBf16 => "mega_mhc_fused_bf16_sm121",
            KernelId::Mxfp8Dense => "b12x_mxfp8_dense_sm121",
            KernelId::Mxfp8WoA => "b12x_mxfp8_wo_a_sm121",
            KernelId::Mxfp4Moe => "b12x_mxfp4_moe_sm121",
            KernelId::Mxfp4MoeStored => "b12x_mxfp4_moe_stored_sm121",
            KernelId::SwigluMxfp8Quant => "swiglu_mxfp8_quant_sm121",
            KernelId::SwigluMxfp8QuantWeighted => "swiglu_mxfp8_quant_weighted_sm121",
            KernelId::CertLmHead => "cert_lm_head_sm121",
            KernelId::MoeRouter => "moe_router_sm121",
            KernelId::RoutedGateUp => "routed_gate_up_sm121",
            KernelId::RmsNorm => "rmsnorm_sm121",
            KernelId::LmHeadReduce => "lm_head_reduce_sm121",
            KernelId::FlashMlaCsa2Seq => "flash_mla_csa2_seq_sm121",
            KernelId::FlashMlaCsa2Ring => "flash_mla_csa2_ring_sm121",
            KernelId::Bf16aMxfp8wDense => "b12x_bf16a_mxfp8w_dense_sm121",
            KernelId::Bf16aMxfp8wWoA => "b12x_bf16a_mxfp8w_wo_a_sm121",
            KernelId::EngramEmbed => "engram_embed_sm121",
            KernelId::EngramWkv => "engram_wkv_sm121",
            KernelId::EngramGateWrite => "engram_gate_write_sm121",
            KernelId::RoutedGateUpSwigluSplitK => "routed_gate_up_swiglu_splitk_sm121",
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
            KernelId::Mxfp4MoeStored => &["b12x_mxfp4_moe_stored_sm121"],
            KernelId::SwigluMxfp8Quant => &["swiglu_mxfp8_quant_sm121"],
            KernelId::SwigluMxfp8QuantWeighted => &["swiglu_mxfp8_quant_weighted_sm121"],
            KernelId::CertLmHead => &["cert_lm_head_sm121"],
            KernelId::MoeRouter => &["moe_router_sm121"],
            KernelId::RoutedGateUp => &["routed_gate_up_sm121"],
            KernelId::RmsNorm => &["rmsnorm_sm121"],
            KernelId::LmHeadReduce => &["lm_head_reduce_sm121"],
            KernelId::FlashMlaCsa2Seq => &["flash_mla_csa2_seq_sm121"],
            KernelId::FlashMlaCsa2Ring => &["flash_mla_csa2_ring_sm121"],
            KernelId::Bf16aMxfp8wDense => &["b12x_bf16a_mxfp8w_dense_sm121"],
            KernelId::Bf16aMxfp8wWoA => &["b12x_bf16a_mxfp8w_wo_a_sm121"],
            KernelId::EngramEmbed => &["engram_embed_sm121"],
            KernelId::EngramWkv => &["engram_wkv_sm121"],
            KernelId::EngramGateWrite => &["engram_gate_write_sm121"],
            KernelId::RoutedGateUpSwigluSplitK => &["routed_gate_up_swiglu_splitk_sm121"],
        }
    }

    /// Maps a cubin file name (the symbol or a documented alias) to the kernel.
    ///
    /// `flash_mla_csa2_sm121` is deliberately absent: that kernel is loaded through
    /// `cuda_harness::load_flash_mla`, which owns its symbol, so it has no `KernelId`. The
    /// deferred sequence entry IS here: it is a separate cubin with its own symbol, and the call
    /// site resolves it like any other shared-driver kernel.
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

/// The cubin name of the kv-path kernels, which have no [`KernelId`]: one cubin carries three
/// symbols (`rope_rotate_bf16`, `window_quant_fp8_bf16`, `kv_append_bf16`) and one baseline entry,
/// and a [`KernelId`] names one symbol of one cubin. `crate::kv_path_launch` owns their typed
/// launches and resolves each symbol by name from this cubin.
pub const KV_PATH_CUBIN: &str = "kv_path_sm121";

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
    } else if name == KV_PATH_CUBIN {
        &[KV_PATH_CUBIN]
    } else {
        KernelId::from_cubin_name(name)
            .ok_or_else(|| {
                format!(
                    "{name} is not one of the shared driver kernels; flash_mla and the kv-path cubin are loaded through their own typed helpers and have no KernelId"
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

/// The chain scope a launch is attributed to: phase, step, layer and position.
///
/// The harness sets it at its position/step loop; every launch until the next set is attributed to
/// that scope. All fields default to 0 (and the phase to "stage") so an unwired arm still produces
/// a well-formed ledger row rather than a wrong one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StageScope {
    /// The phase name: prefill, decode or a stage-arm name.
    pub phase: &'static str,
    /// The step index inside the phase.
    pub step: usize,
    /// The backbone layer, when the caller knows it.
    pub layer: usize,
    /// The absolute token position.
    pub position: usize,
}

/// One kernel's aggregated device time in one phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelTimingRow {
    /// The phase the launches were attributed to.
    pub phase: &'static str,
    /// The canonical cubin pin name.
    pub kernel: &'static str,
    /// How many launches were bracketed.
    pub calls: u64,
    /// The summed event time, nanoseconds.
    pub total_ns: u64,
    /// The mean event time, nanoseconds.
    pub mean_ns: u64,
    /// The shortest launch, nanoseconds.
    pub min_ns: u64,
    /// The longest launch, nanoseconds.
    pub max_ns: u64,
    /// The ordinal of the first launch of this kernel in this phase.
    pub first_launch_order: usize,
}

/// One kernel's launches inside one step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepLaunchRow {
    /// The canonical cubin pin name.
    pub kernel: &'static str,
    /// How many launches were bracketed.
    pub calls: u64,
    /// The summed event time, nanoseconds.
    pub total_ns: u64,
}

/// One step's launches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepTimingRow {
    /// The step index the scope reported.
    pub step: usize,
    /// True when the caller marked this step as warmup (cold-vs-steady separation).
    pub warm: bool,
    /// How many launches were bracketed.
    pub launches: u64,
    /// The summed event time, nanoseconds.
    pub kernel_ns: u64,
    /// The per-kernel rows, in kernel-name order.
    pub kernels: Vec<StepLaunchRow>,
}

/// The per-kernel ledger one run collected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageTimingReport {
    /// The per-phase, per-kernel rows, in first-launch order.
    pub kernels: Vec<KernelTimingRow>,
    /// The per-step rows, in step order.
    pub steps: Vec<StepTimingRow>,
    /// How many distinct kernel names beyond the 128-name cap were dropped.
    pub dropped_names: usize,
}

/// The distinct-kernel cap, so a run cannot grow the ledger without bound.
const STAGE_TIMING_KERNEL_CAP: usize = 128;

#[derive(Default)]
struct KernelRow {
    calls: u64,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    first_order: usize,
}

#[derive(Default)]
struct StepRow {
    launches: u64,
    kernel_ns: u64,
    kernels: BTreeMap<&'static str, (u64, u64)>,
    warm: bool,
}

struct StageRecorder {
    scope: StageScope,
    order: usize,
    dropped: usize,
    kernels: BTreeMap<(&'static str, &'static str), KernelRow>,
    steps: BTreeMap<usize, StepRow>,
}

impl Default for StageRecorder {
    fn default() -> Self {
        Self {
            scope: StageScope {
                phase: "stage",
                step: 0,
                layer: 0,
                position: 0,
            },
            order: 0,
            dropped: 0,
            kernels: BTreeMap::new(),
            steps: BTreeMap::new(),
        }
    }
}

/// The process-wide recorder. The ledger is per process, not per [Cuda], because one chain opens
/// several driver handles (attention, MoE and the mHC launcher) and all of their launches belong in
/// one table.
static STAGE_RECORDER: Mutex<Option<StageRecorder>> = Mutex::new(None);

/// Set by the device arm before its dummy steps; read when a step row is first touched.
static WARM_STEP: AtomicBool = AtomicBool::new(false);

/// Whether SPARKFLASH_STAGE_TIMING=1 armed the bracket. Off by default: no events, no allocations
/// and a byte-identical run when it is unset.
pub fn stage_timing_enabled() -> bool {
    std::env::var("SPARKFLASH_STAGE_TIMING").ok().as_deref() == Some("1")
}

/// Sets the scope every following launch is attributed to. A no-op when the recorder is off.
pub fn stage_timing_scope(phase: &'static str, step: usize, layer: usize, position: usize) {
    if !stage_timing_enabled() {
        return;
    }
    let Ok(mut guard) = STAGE_RECORDER.lock() else {
        return;
    };
    let Some(recorder) = guard.as_mut() else {
        return;
    };
    recorder.scope = StageScope {
        phase,
        step,
        layer,
        position,
    };
}

/// Marks subsequent steps as warmup steps until cleared. The device arm sets this for its dummy
/// pre-serving steps and clears it before the first real token, so the ledger can split first-step
/// (cold) time from steady-state time instead of averaging them together.
pub fn stage_timing_set_warm(warm: bool) {
    WARM_STEP.store(warm, Ordering::Relaxed);
}

/// Updates only the layer of the current scope (the chain's per-layer loop calls this).
pub fn stage_timing_set_layer(layer: usize) {
    if !stage_timing_enabled() {
        return;
    }
    let Ok(mut guard) = STAGE_RECORDER.lock() else {
        return;
    };
    if let Some(recorder) = guard.as_mut() {
        recorder.scope.layer = layer;
    }
}

/// Records one bracketed launch. Called by [Cuda::launch] only.
fn record_launch(kernel: &'static str, elapsed_ns: u64) {
    let Ok(mut guard) = STAGE_RECORDER.lock() else {
        return;
    };
    let Some(recorder) = guard.as_mut() else {
        return;
    };
    let phase = recorder.scope.phase;
    let step = recorder.scope.step;
    let order = recorder.order;
    recorder.order += 1;
    let known = recorder.kernels.contains_key(&(phase, kernel));
    if !known && recorder.kernels.len() >= STAGE_TIMING_KERNEL_CAP {
        recorder.dropped += 1;
        return;
    }
    let row = recorder
        .kernels
        .entry((phase, kernel))
        .or_insert(KernelRow {
            first_order: order,
            min_ns: elapsed_ns,
            max_ns: elapsed_ns,
            ..KernelRow::default()
        });
    row.calls += 1;
    row.total_ns = row.total_ns.saturating_add(elapsed_ns);
    row.min_ns = row.min_ns.min(elapsed_ns);
    row.max_ns = row.max_ns.max(elapsed_ns);
    let step_row = recorder.steps.entry(step).or_default();
    step_row.warm = WARM_STEP.load(Ordering::Relaxed);
    step_row.launches += 1;
    step_row.kernel_ns = step_row.kernel_ns.saturating_add(elapsed_ns);
    let entry = step_row.kernels.entry(kernel).or_insert((0, 0));
    entry.0 += 1;
    entry.1 = entry.1.saturating_add(elapsed_ns);
}

/// The ledger one run collected, or None when the recorder was never armed.
pub fn stage_timing_report() -> Option<StageTimingReport> {
    let guard = STAGE_RECORDER.lock().ok()?;
    let recorder = guard.as_ref()?;
    let mut kernels: Vec<KernelTimingRow> = recorder
        .kernels
        .iter()
        .map(|((phase, kernel), row)| KernelTimingRow {
            phase,
            kernel,
            calls: row.calls,
            total_ns: row.total_ns,
            mean_ns: row.total_ns.checked_div(row.calls).unwrap_or(0),
            min_ns: row.min_ns,
            max_ns: row.max_ns,
            first_launch_order: row.first_order,
        })
        .collect();
    kernels.sort_by_key(|row| row.first_launch_order);
    let steps: Vec<StepTimingRow> = recorder
        .steps
        .iter()
        .map(|(step, row)| StepTimingRow {
            step: *step,
            warm: row.warm,
            launches: row.launches,
            kernel_ns: row.kernel_ns,
            kernels: row
                .kernels
                .iter()
                .map(|(kernel, (calls, total_ns))| StepLaunchRow {
                    kernel,
                    calls: *calls,
                    total_ns: *total_ns,
                })
                .collect(),
        })
        .collect();
    Some(StageTimingReport {
        kernels,
        steps,
        dropped_names: recorder.dropped,
    })
}

/// Counts one launch bracket site, so a test can prove the bracket is armed.
#[must_use]
pub fn stage_timing_launches() -> u64 {
    STAGE_RECORDER
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|recorder| recorder.order as u64))
        .unwrap_or(0)
}

/// One reading of the driver-entry meter the forward perf probe arms.
///
/// The first probe kit reported host wall per block only, so a block could not be called
/// device-bound or host-bound from its own row. This meter is the other half: the driver entry
/// points ([Cuda::launch], [Cuda::synchronize], the three copies, the allocations, the frees and
/// the module loads) add to these counters while the probe has them armed, and the probe turns the
/// difference between two readings into the per-block kernel / staging / host-sync / launch
/// overhead split the P0 gate asks for. Every field is a plain monotone counter: a consumer
/// subtracts two readings with [ProbeMeterReading::since], so an absolute value is never quoted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProbeMeterReading {
    /// cuLaunchKernel calls.
    pub launches: u64,
    /// The summed cuEventElapsedTime of those launches, in nanoseconds: the kernels' own device
    /// time, without the host's launch or drain cost.
    pub kernel_ns: u64,
    /// cuCtxSynchronize calls: one per [Cuda::launch] on this driver, plus whatever the numeric
    /// path asks for directly.
    pub syncs: u64,
    /// The summed host wall spent inside those cuCtxSynchronize calls, in nanoseconds. This is the
    /// host-sync bucket: kernel_ns is device time, this is the host waiting for it.
    pub sync_ns: u64,
    /// cuMemcpyHtoD_v2 calls (host to device staging).
    pub h2d_count: u64,
    /// Bytes staged host to device.
    pub h2d_bytes: u64,
    /// The summed host wall of those copies, in nanoseconds.
    pub h2d_ns: u64,
    /// cuMemcpyDtoH_v2 calls (device to host reads).
    pub d2h_count: u64,
    /// Bytes read back to the host.
    pub d2h_bytes: u64,
    /// The summed host wall of those copies, in nanoseconds.
    pub d2h_ns: u64,
    /// cuMemcpyDtoD_v2 calls.
    pub dtod_count: u64,
    /// Bytes copied device to device.
    pub dtod_bytes: u64,
    /// The summed host wall of those copies, in nanoseconds.
    pub dtod_ns: u64,
    /// Device or managed or page-locked allocations made.
    pub alloc_count: u64,
    /// Allocations released.
    pub free_count: u64,
    /// Cubin modules loaded (one per [Cuda::load_kernel] and its siblings).
    pub module_load_count: u64,
}

impl ProbeMeterReading {
    /// The work that happened between an earlier reading and this one.
    ///
    /// Saturating because the counters are process-wide: a reading taken on one thread can be
    /// overtaken by a launch on another, and a negative delta would be a worse answer than zero.
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            launches: self.launches.saturating_sub(earlier.launches),
            kernel_ns: self.kernel_ns.saturating_sub(earlier.kernel_ns),
            syncs: self.syncs.saturating_sub(earlier.syncs),
            sync_ns: self.sync_ns.saturating_sub(earlier.sync_ns),
            h2d_count: self.h2d_count.saturating_sub(earlier.h2d_count),
            h2d_bytes: self.h2d_bytes.saturating_sub(earlier.h2d_bytes),
            h2d_ns: self.h2d_ns.saturating_sub(earlier.h2d_ns),
            d2h_count: self.d2h_count.saturating_sub(earlier.d2h_count),
            d2h_bytes: self.d2h_bytes.saturating_sub(earlier.d2h_bytes),
            d2h_ns: self.d2h_ns.saturating_sub(earlier.d2h_ns),
            dtod_count: self.dtod_count.saturating_sub(earlier.dtod_count),
            dtod_bytes: self.dtod_bytes.saturating_sub(earlier.dtod_bytes),
            dtod_ns: self.dtod_ns.saturating_sub(earlier.dtod_ns),
            alloc_count: self.alloc_count.saturating_sub(earlier.alloc_count),
            free_count: self.free_count.saturating_sub(earlier.free_count),
            module_load_count: self
                .module_load_count
                .saturating_sub(earlier.module_load_count),
        }
    }

    /// Two readings summed, which is how a scope's blocks are folded into its row.
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self {
            launches: self.launches.saturating_add(other.launches),
            kernel_ns: self.kernel_ns.saturating_add(other.kernel_ns),
            syncs: self.syncs.saturating_add(other.syncs),
            sync_ns: self.sync_ns.saturating_add(other.sync_ns),
            h2d_count: self.h2d_count.saturating_add(other.h2d_count),
            h2d_bytes: self.h2d_bytes.saturating_add(other.h2d_bytes),
            h2d_ns: self.h2d_ns.saturating_add(other.h2d_ns),
            d2h_count: self.d2h_count.saturating_add(other.d2h_count),
            d2h_bytes: self.d2h_bytes.saturating_add(other.d2h_bytes),
            d2h_ns: self.d2h_ns.saturating_add(other.d2h_ns),
            dtod_count: self.dtod_count.saturating_add(other.dtod_count),
            dtod_bytes: self.dtod_bytes.saturating_add(other.dtod_bytes),
            dtod_ns: self.dtod_ns.saturating_add(other.dtod_ns),
            alloc_count: self.alloc_count.saturating_add(other.alloc_count),
            free_count: self.free_count.saturating_add(other.free_count),
            module_load_count: self
                .module_load_count
                .saturating_add(other.module_load_count),
        }
    }

    /// Whether any counter moved. A probe row that finds a block's delta clear writes the block's
    /// wall alone, and a test that has not touched the meter expects this to hold.
    #[must_use]
    pub fn is_clear(self) -> bool {
        self == Self::default()
    }
}

/// The meter's cells. One struct of relaxed atomics keeps the hot-path cost of an armed meter to a
/// handful of uncontended atomic adds.
struct ProbeMeter {
    launches: std::sync::atomic::AtomicU64,
    kernel_ns: std::sync::atomic::AtomicU64,
    syncs: std::sync::atomic::AtomicU64,
    sync_ns: std::sync::atomic::AtomicU64,
    h2d_count: std::sync::atomic::AtomicU64,
    h2d_bytes: std::sync::atomic::AtomicU64,
    h2d_ns: std::sync::atomic::AtomicU64,
    d2h_count: std::sync::atomic::AtomicU64,
    d2h_bytes: std::sync::atomic::AtomicU64,
    d2h_ns: std::sync::atomic::AtomicU64,
    dtod_count: std::sync::atomic::AtomicU64,
    dtod_bytes: std::sync::atomic::AtomicU64,
    dtod_ns: std::sync::atomic::AtomicU64,
    alloc_count: std::sync::atomic::AtomicU64,
    free_count: std::sync::atomic::AtomicU64,
    module_load_count: std::sync::atomic::AtomicU64,
}

impl ProbeMeter {
    const fn new() -> Self {
        Self {
            launches: std::sync::atomic::AtomicU64::new(0),
            kernel_ns: std::sync::atomic::AtomicU64::new(0),
            syncs: std::sync::atomic::AtomicU64::new(0),
            sync_ns: std::sync::atomic::AtomicU64::new(0),
            h2d_count: std::sync::atomic::AtomicU64::new(0),
            h2d_bytes: std::sync::atomic::AtomicU64::new(0),
            h2d_ns: std::sync::atomic::AtomicU64::new(0),
            d2h_count: std::sync::atomic::AtomicU64::new(0),
            d2h_bytes: std::sync::atomic::AtomicU64::new(0),
            d2h_ns: std::sync::atomic::AtomicU64::new(0),
            dtod_count: std::sync::atomic::AtomicU64::new(0),
            dtod_bytes: std::sync::atomic::AtomicU64::new(0),
            dtod_ns: std::sync::atomic::AtomicU64::new(0),
            alloc_count: std::sync::atomic::AtomicU64::new(0),
            free_count: std::sync::atomic::AtomicU64::new(0),
            module_load_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn clear(&self) {
        for cell in [
            &self.launches,
            &self.kernel_ns,
            &self.syncs,
            &self.sync_ns,
            &self.h2d_count,
            &self.h2d_bytes,
            &self.h2d_ns,
            &self.d2h_count,
            &self.d2h_bytes,
            &self.d2h_ns,
            &self.dtod_count,
            &self.dtod_bytes,
            &self.dtod_ns,
            &self.alloc_count,
            &self.free_count,
            &self.module_load_count,
        ] {
            cell.store(0, Ordering::Relaxed);
        }
    }

    fn read(&self) -> ProbeMeterReading {
        ProbeMeterReading {
            launches: self.launches.load(Ordering::Relaxed),
            kernel_ns: self.kernel_ns.load(Ordering::Relaxed),
            syncs: self.syncs.load(Ordering::Relaxed),
            sync_ns: self.sync_ns.load(Ordering::Relaxed),
            h2d_count: self.h2d_count.load(Ordering::Relaxed),
            h2d_bytes: self.h2d_bytes.load(Ordering::Relaxed),
            h2d_ns: self.h2d_ns.load(Ordering::Relaxed),
            d2h_count: self.d2h_count.load(Ordering::Relaxed),
            d2h_bytes: self.d2h_bytes.load(Ordering::Relaxed),
            d2h_ns: self.d2h_ns.load(Ordering::Relaxed),
            dtod_count: self.dtod_count.load(Ordering::Relaxed),
            dtod_bytes: self.dtod_bytes.load(Ordering::Relaxed),
            dtod_ns: self.dtod_ns.load(Ordering::Relaxed),
            alloc_count: self.alloc_count.load(Ordering::Relaxed),
            free_count: self.free_count.load(Ordering::Relaxed),
            module_load_count: self.module_load_count.load(Ordering::Relaxed),
        }
    }
}

/// The process-wide meter. Process-wide, like the stage ledger, because one run opens several
/// driver handles (attention, MoE and the mHC launcher) and all of their launches belong to the
/// same block table.
static PROBE_METER: ProbeMeter = ProbeMeter::new();

/// Whether the forward perf probe armed the meter. Off by default: every instrumented entry point
/// reads this one relaxed flag and does nothing else.
static PROBE_METER_ARMED: AtomicBool = AtomicBool::new(false);

/// Armed by the no-sync probe mode (`--forward-perf-probe-nosync`): every launch records one event
/// pair and leaves the launch queued, and the elapsed times are resolved once at run end by
/// [Cuda::probe_meter_harvest]. The counters keep recording because [meter_recording] reads both
/// flags, so a host-only-wall row keeps its launch/sync/copy census and its readback waits; only
/// the device time moves off the block wall and into the harvest.
static PROBE_METER_DEFERRED: AtomicBool = AtomicBool::new(false);

/// The unresolved deferred event pairs, one `(start, stop)` handle per launch, in launch order.
/// Handles travel as `usize` so the static stays `Send`; the driver reinterprets them as events.
static DEFERRED_EVENTS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Arms or clears the driver meter; either way its cells are cleared, so a disarmed meter never
/// holds a stale reading and a run starts from zero.
///
/// Called by the forward perf probe at run scope and by this module's own tests; it is the only
/// switch. Clearing is what lets a second token stage in one process report its own numbers instead
/// of the first stage's plus its own.
pub fn probe_meter_arm(armed: bool) {
    PROBE_METER.clear();
    PROBE_METER_ARMED.store(armed, Ordering::Relaxed);
}

/// Whether the driver meter is armed (one relaxed load, the whole off-path cost).
#[must_use]
pub fn probe_meter_armed() -> bool {
    PROBE_METER_ARMED.load(Ordering::Relaxed)
}

/// The meter's absolute reading; a consumer subtracts two of them ([ProbeMeterReading::since]).
#[must_use]
pub fn probe_meter_read() -> ProbeMeterReading {
    PROBE_METER.read()
}

/// Arms the no-sync (host-only wall) probe meter: launches queue and record one event pair each,
/// and the pairs are resolved by [Cuda::probe_meter_harvest] after the run's final synchronize.
///
/// This is a diagnostic switch, never an arithmetic one: the launch order, the kernel arguments and
/// the copied bytes are the drained path's, and only the host's wait moves off the block wall. The
/// counters the probe differences are still written, so a row keeps its census next to a wall that
/// no longer contains the device.
pub fn probe_meter_arm_deferred(armed: bool) {
    PROBE_METER_DEFERRED.store(armed, Ordering::Relaxed);
    if armed && let Ok(mut events) = DEFERRED_EVENTS.lock() {
        events.clear();
    }
}

/// Whether the no-sync meter is armed (the host-only-wall probe mode).
#[must_use]
pub fn probe_meter_deferred() -> bool {
    PROBE_METER_DEFERRED.load(Ordering::Relaxed)
}

/// Whether a driver entry point the probe counts should add to the meter: the drained meter or the
/// no-sync one. Both feed the same counters, so a row's census and readback waits are comparable
/// across the two probe modes.
fn meter_recording() -> bool {
    probe_meter_armed() || probe_meter_deferred()
}

/// Counts one queued launch on the no-sync path; its kernel time arrives from the harvest.
fn note_launch_queued() {
    if !probe_meter_deferred() {
        return;
    }
    PROBE_METER.launches.fetch_add(1, Ordering::Relaxed);
}

fn note_launch(kernel_ns: u64) {
    if !probe_meter_armed() {
        return;
    }
    PROBE_METER.launches.fetch_add(1, Ordering::Relaxed);
    PROBE_METER
        .kernel_ns
        .fetch_add(kernel_ns, Ordering::Relaxed);
}

fn note_sync(sync_ns: u64) {
    if !meter_recording() {
        return;
    }
    PROBE_METER.syncs.fetch_add(1, Ordering::Relaxed);
    PROBE_METER.sync_ns.fetch_add(sync_ns, Ordering::Relaxed);
}

fn note_h2d(bytes: usize, ns: u64) {
    if !meter_recording() {
        return;
    }
    PROBE_METER.h2d_count.fetch_add(1, Ordering::Relaxed);
    PROBE_METER
        .h2d_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    PROBE_METER.h2d_ns.fetch_add(ns, Ordering::Relaxed);
}

fn note_d2h(bytes: usize, ns: u64) {
    if !meter_recording() {
        return;
    }
    PROBE_METER.d2h_count.fetch_add(1, Ordering::Relaxed);
    PROBE_METER
        .d2h_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    PROBE_METER.d2h_ns.fetch_add(ns, Ordering::Relaxed);
}

fn note_dtod(bytes: usize, ns: u64) {
    if !meter_recording() {
        return;
    }
    PROBE_METER.dtod_count.fetch_add(1, Ordering::Relaxed);
    PROBE_METER
        .dtod_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    PROBE_METER.dtod_ns.fetch_add(ns, Ordering::Relaxed);
}

fn note_alloc() {
    if !meter_recording() {
        return;
    }
    PROBE_METER.alloc_count.fetch_add(1, Ordering::Relaxed);
}

fn note_free(count: u64) {
    if !meter_recording() || count == 0 {
        return;
    }
    PROBE_METER.free_count.fetch_add(count, Ordering::Relaxed);
}

fn note_module_load() {
    if !meter_recording() {
        return;
    }
    PROBE_METER
        .module_load_count
        .fetch_add(1, Ordering::Relaxed);
}

/// Whether [Cuda::launch] may skip its per-launch context drain.
///
/// The four-rank forward loop turns this on for its own run scope through
/// [enable_forward_async_launches]; every other caller (the kernel harnesses, the examples) leaves
/// it off, so [Cuda::launch] keeps its historical synchronous semantics there. A launch bracket is
/// also always synchronous: the stage-timing ledger and the probe meter both read a per-launch
/// cuEventElapsedTime after the drain that follows the launch, so those diagnostic consumers keep
/// the drained path even inside a forward run.
static ASYNC_LAUNCHES: AtomicBool = AtomicBool::new(false);

/// The opt-out spelling a forward run accepts: a non-empty value of `SPARKFLASH_SYNC_LAUNCHES`
/// other than `0` puts the per-launch drain back.
///
/// Kept as a pure function so the mode selection is testable on a host with no device.
#[must_use]
fn sync_launches_requested(env: Option<&str>) -> bool {
    match env {
        None => false,
        Some(value) => !value.is_empty() && value != "0",
    }
}

/// Turns the async launch path on or off and returns the value in force.
///
/// The four-rank forward loop calls this once at run scope; a process that never calls it keeps
/// the synchronous driver path.
pub fn set_async_launches(enabled: bool) -> bool {
    ASYNC_LAUNCHES.store(enabled, Ordering::Relaxed);
    enabled
}

/// Whether [Cuda::launch] is allowed to leave its kernels queued.
#[must_use]
pub fn async_launches() -> bool {
    ASYNC_LAUNCHES.load(Ordering::Relaxed)
}

/// Selects the forward run's launch mode from the environment and returns the value in force.
///
/// The default is the async path; `SPARKFLASH_SYNC_LAUNCHES=1` opts the run back into the
/// per-launch drain (the pre-M2 host path, kept as the one-switch rollback).
pub fn enable_forward_async_launches() -> bool {
    let requested =
        sync_launches_requested(std::env::var("SPARKFLASH_SYNC_LAUNCHES").ok().as_deref());
    set_async_launches(!requested)
}

/// Whether a driver call that only *queues* device work may skip its per-call drain.
///
/// The launch path and the device-to-device stage copy share this one decision: both are invisible
/// to the host, and both are made safe by the next observation point (a stream-ordered
/// [Cuda::copy_dtoh] or an explicit [Cuda::synchronize]). Either bracket consumer - the stage-timing
/// ledger or the probe meter - reads a post-drain elapsed time, so its presence puts both calls
/// back on the drained path.
#[must_use]
fn queued_dispatch_wanted() -> bool {
    !stage_timing_enabled() && !probe_meter_armed() && async_launches()
}

/// One run's cubin images and their loaded modules.
///
/// M2/C1: the four-rank layer loop used to read and hash eight cubin images and run nine
/// cuModuleLoadData/cuModuleGetFunction pairs on every layer. A cubin is now interned once by its
/// own bytes and its modules are resolved from the same id plus the symbol name, which turns the
/// per-layer module path into two map lookups. Two images with equal bytes share one id; two images
/// whose bytes differ never share an id, even when a freed buffer's address has been reused by an
/// image of the same length. Modules owned here are unloaded by [Cuda]'s Drop, never by a per-layer
/// free.
#[derive(Default)]
struct ModuleCache {
    /// The interned images, kept so a later slice with different bytes can never be mistaken for
    /// an earlier one and so a pointer lookup can be reused.
    images: Vec<CachedImage>,
    /// `(data_ptr, len)` of an interned image -> its index in `images`. A hit is the common
    /// hot-path case, but it is only trusted after the cached bytes are compared with the slice
    /// (one memcmp, no allocation).
    ptr_ids: HashMap<(usize, usize), usize>,
    next_id: u64,
    /// `(image id, symbol name)` -> loaded module + function.
    modules: HashMap<(u64, String), (CuModule, CuFunction)>,
    /// The module handles this cache owns, so [Cuda::free] can leave them loaded.
    module_ptrs: HashSet<usize>,
}

struct CachedImage {
    id: u64,
    bytes: Vec<u8>,
}

impl ModuleCache {
    /// Interns one cubin image and returns its id.
    ///
    /// Guarantees: equal bytes always map to one id, and different bytes never share an id. The
    /// `(pointer, length)` fast path does one map lookup plus one byte comparison and allocates
    /// nothing; a key whose cached image no longer has the same bytes (a freed buffer's address
    /// reused by another image of the same length) is not trusted and falls through to the
    /// byte-comparison path, which interns the different bytes as a new id.
    fn intern(&mut self, cubin: &[u8]) -> u64 {
        self.intern_with_key(cubin.as_ptr() as usize, cubin.len(), cubin)
    }

    /// Interns one image under an explicit pointer key.
    ///
    /// [ModuleCache::intern] passes the slice's own address. The keyed form exists so a test can
    /// pin the dangerous address-reuse direction deterministically - the same key with different
    /// bytes must be a different id - without asking the allocator to hand a freed address back.
    fn intern_with_key(&mut self, key_ptr: usize, key_len: usize, cubin: &[u8]) -> u64 {
        let key = (key_ptr, key_len);
        if let Some(&index) = self.ptr_ids.get(&key) {
            let image = &self.images[index];
            if image.bytes.as_slice() == cubin {
                return image.id;
            }
            // The address was reused by different bytes. Do not trust the stale mapping: fall
            // through to the byte comparison below, which either finds the equal image or
            // interns these bytes as a new id, and remaps this key either way.
        }
        if let Some(index) = self
            .images
            .iter()
            .position(|image| image.bytes.as_slice() == cubin)
        {
            let id = self.images[index].id;
            self.ptr_ids.insert(key, index);
            return id;
        }
        self.next_id += 1;
        let id = self.next_id;
        self.images.push(CachedImage {
            id,
            bytes: cubin.to_vec(),
        });
        self.ptr_ids.insert(key, self.images.len() - 1);
        id
    }

    fn function(&self, image: u64, name: &str) -> Option<(CuModule, CuFunction)> {
        self.modules.get(&(image, name.to_string())).copied()
    }

    fn insert_function(&mut self, image: u64, name: &str, module: CuModule, function: CuFunction) {
        self.module_ptrs.insert(module as usize);
        self.modules
            .insert((image, name.to_string()), (module, function));
    }

    fn owns_module(&self, module: CuModule) -> bool {
        self.module_ptrs.contains(&(module as usize))
    }
}

/// One run's device buffers, keyed by their exact byte size.
///
/// M2/C2: an ordinary layer used to issue 34 cuMemAlloc/cuMemFree pairs. [Cuda::alloc_pooled]
/// checks a released buffer of the same size out again instead, and [Cuda::release_pooled] stamps
/// it with the launch counter current at release. A buffer is only reusable once the context's
/// completion counter has reached that stamp, i.e. once a stream-ordered device-to-host copy or an
/// explicit [Cuda::synchronize] taken after the release has proven every earlier launch finished.
/// That is the exactness rule that lets the async launch path hand the same bytes back to the next
/// layer without draining the context.
///
/// Ownership is tracked per pointer: a pooled allocation is checked out from the moment it is
/// recorded until [BufferPool::release] puts it back, and checked in from that release until the
/// next checkout. A pointer that is already checked in is refused rather than added to the free
/// list a second time, so one allocation can never be handed to two live kernels.
#[derive(Default)]
struct BufferPool {
    free: HashMap<usize, Vec<(Deviceptr, u64)>>,
    /// Every pooled allocation, so a release knows its size and Drop knows what it owns.
    sizes: HashMap<u64, usize>,
    /// The subset of `sizes` currently checked in (present in `free`). A release is only
    /// accepted for a checked-out pointer, which is what makes a double release refuse.
    checked_in: HashSet<u64>,
}

impl BufferPool {
    fn checkout(&mut self, bytes: usize, completed: u64) -> Option<Deviceptr> {
        let list = self.free.get_mut(&bytes)?;
        let index = list
            .iter()
            .rposition(|(_ptr, needed)| *needed <= completed)?;
        let ptr = list.swap_remove(index).0;
        self.checked_in.remove(&ptr.0);
        Some(ptr)
    }

    fn record_allocation(&mut self, ptr: Deviceptr, bytes: usize) {
        self.sizes.insert(ptr.0, bytes);
    }

    /// Returns a checked-out pooled pointer to the free list at launch stamp `needed`.
    ///
    /// The fence is unchanged: [BufferPool::checkout] hands the entry out again only once
    /// `completed >= needed`. A pointer this pool does not own, and a pointer already checked in,
    /// are refused with an error instead of inserting a duplicate free-list entry.
    fn release(&mut self, ptr: Deviceptr, needed: u64) -> Result<(), String> {
        let Some(bytes) = self.sizes.get(&ptr.0).copied() else {
            return Err(format!("buffer {:#x} is not a pooled allocation", ptr.0));
        };
        if !self.checked_in.insert(ptr.0) {
            return Err(format!(
                "pooled buffer {:#x} was released twice without a checkout in between",
                ptr.0
            ));
        }
        self.free.entry(bytes).or_default().push((ptr, needed));
        Ok(())
    }

    /// Removes a pooled pointer from the pool's bookkeeping without freeing its device memory.
    ///
    /// [Cuda::free] and [Cuda::free_buffers] use this for a pointer the pool owns: the driver
    /// allocation must not be passed to cuMemFree, because the pool would free it again from Drop,
    /// and the pool must not hand it out again. Removing the entry from the free list and from
    /// `sizes` leaves the allocation to be reclaimed when the primary context is released. Returns
    /// false when the pool does not own the pointer, and the caller frees it as an ordinary
    /// allocation.
    fn forget(&mut self, ptr: Deviceptr) -> bool {
        let Some(bytes) = self.sizes.remove(&ptr.0) else {
            return false;
        };
        if let Some(list) = self.free.get_mut(&bytes) {
            list.retain(|(candidate, _needed)| candidate.0 != ptr.0);
            let empty = list.is_empty();
            if empty {
                self.free.remove(&bytes);
            }
        }
        self.checked_in.remove(&ptr.0);
        true
    }
}

/// A handle to one CUDA graph executable owned by the [Cuda] that instantiated it.
///
/// The id is only meaningful to the handle that captured it: launching with another handle's id
/// is an error, so a stale handle cannot name a graph from a different context. The executable is
/// destroyed by the owning handle's `Drop` (or by [Cuda::graph_destroy]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphId(u64);

/// One kernel launch recorded while a CUDA graph segment was being captured or recorded.
///
/// A record is the launch's identity: the `cuFunction` handle, the grid and block, and every
/// argument word. Two records compare equal only when all of those are identical, which is what
/// makes replay safe. A launch whose argument vector changed - a different pooled buffer, a
/// different weight slice, another layer - is a different record, and the wrapper executes it
/// instead of replaying a graph captured for the old arguments. No argument is ever patched.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaLaunchRecord {
    function: usize,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    args: Vec<u64>,
}

impl CudaLaunchRecord {
    /// The launch identity as a flat word vector: the function handle, the grid, the block, then
    /// every argument word in order.
    ///
    /// Two fingerprints are equal exactly when the launch lists are equal; the graph wrapper in
    /// `sparkflash-cli` compares these instead of the private record fields.
    pub fn fingerprint_words(&self) -> Vec<u64> {
        let mut words = Vec::with_capacity(9 + self.args.len());
        words.push(self.function as u64);
        words.extend([
            u64::from(self.grid.0),
            u64::from(self.grid.1),
            u64::from(self.grid.2),
            u64::from(self.block.0),
            u64::from(self.block.1),
            u64::from(self.block.2),
        ]);
        words.extend(self.args.iter().copied());
        words
    }

    /// A short description for a fallback report, without exposing the raw argument words.
    pub fn describe(&self) -> String {
        format!(
            "grid {}x{}x{}, block {}x{}x{}, {} args",
            self.grid.0,
            self.grid.1,
            self.grid.2,
            self.block.0,
            self.block.1,
            self.block.2,
            self.args.len()
        )
    }
}

/// The most graph executables one [Cuda] keeps alive.
///
/// A capture is a warm-up event, not a steady-state one. The wrapper keeps at most one executable
/// per distinct launch list (a per-layer weight set is a distinct list), and this cap bounds a run
/// that keeps producing new lists without replaying any of them: past the cap a segment executes
/// its recorded launches instead of leaking another executable.
const MAX_GRAPH_EXECUTABLES: usize = 128;

/// The driver state a graph segment needs; the launch hot path never locks it.
#[derive(Default)]
struct GraphControl {
    /// The private stream capture is running on; null unless [GRAPH_CAPTURE] is active.
    capture_stream: CuStream,
    /// Launches issued since the segment began, in order. Filled by both capture and record so a
    /// captured graph always has the launch list it replays.
    records: Vec<CudaLaunchRecord>,
    /// Instantiated graph executables, keyed by [GraphId].
    execs: BTreeMap<u64, CuGraphExec>,
    /// The next [GraphId] to hand out; never reused, so a stale id can only fail, not alias.
    next_exec: u64,
}

/// The non-launch operations a graph segment refuses.
///
/// Stream capture records kernels only. An allocation, a copy, a synchronize or a module load
/// issued inside a segment is either illegal during capture (the driver answers with a capture
/// error) or would silently run outside the recorded graph; both destroy the result's identity, so
/// each entry point refuses with a named error instead. The wrapper in `sparkflash-cli` reports
/// the refusal and falls back to the direct launch path for that segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SegmentOp {
    Copy,
    Synchronize,
    Allocation,
    Free,
    Release,
    ModuleLoad,
    RepeatedLaunch,
}

impl SegmentOp {
    fn refusal(self) -> &'static str {
        match self {
            SegmentOp::Copy => {
                "a host or device copy cannot run inside a CUDA graph segment (capture records kernels only)"
            }
            SegmentOp::Synchronize => {
                "cuCtxSynchronize cannot run inside a CUDA graph segment (capture cannot cross a host wait)"
            }
            SegmentOp::Allocation => {
                "an allocation cannot run inside a CUDA graph segment (buffers must be checked out before capture)"
            }
            SegmentOp::Free => {
                "a buffer free cannot run inside a CUDA graph segment (a captured graph reads the buffers it was captured with)"
            }
            SegmentOp::Release => {
                "release_pooled cannot run inside a CUDA graph segment (the captured buffers must stay checked out across replays)"
            }
            SegmentOp::ModuleLoad => {
                "a module load cannot run inside a CUDA graph segment (load and warm every kernel before capturing)"
            }
            SegmentOp::RepeatedLaunch => {
                "launch_repeated cannot run inside a CUDA graph segment (it creates events and synchronizes)"
            }
        }
    }
}

/// The one place that decides whether an operation may run while a graph segment is active.
fn graph_segment_refusal(mode: u8, op: SegmentOp) -> Option<&'static str> {
    if mode == GRAPH_DIRECT {
        None
    } else {
        Some(op.refusal())
    }
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
    /// The run-scope cubin/module cache (M2/C1).
    modules: Mutex<ModuleCache>,
    /// The run-scope device buffer pool (M2/C2).
    pool: Mutex<BufferPool>,
    /// Launches queued since this handle opened, and the high-water mark a completion point has
    /// proven finished. The buffer pool's reuse fence reads both.
    issued: AtomicU64,
    completed: AtomicU64,
    /// Capture/record bookkeeping. Only segment control and segment launches lock it; the hot
    /// launch path reads [Cuda::graph_mode] and leaves the mutex alone in direct mode.
    graph: Mutex<GraphControl>,
    /// 0 direct, 1 capture, 2 record. One relaxed load at the top of every launch.
    graph_mode: AtomicU8,
}

impl Cuda {
    /// Blocks until every queued operation on this context has completed (the M2 probe sync point).
    ///
    /// The forward perf probe calls this once at each timed block boundary so the host `Instant`
    /// delta is device-inclusive; nothing on the numeric path calls it.
    ///
    /// # Errors
    /// The driver's cuCtxSynchronize failure code.
    pub fn synchronize(&self) -> Result<(), String> {
        if let Some(refusal) = self.graph_refusal(SegmentOp::Synchronize) {
            return Err(refusal.to_string());
        }
        let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
            self.sym(b"cuCtxSynchronize")?;
        let metered = meter_recording();
        let started = metered.then(std::time::Instant::now);
        // SAFETY: the symbol was resolved from this context's own library and takes no arguments;
        // it only blocks the calling thread until the context's queued work completes.
        let r = unsafe { sync() };
        if let Some(started) = started {
            note_sync(started.elapsed().as_nanos() as u64);
        }
        if r != 0 {
            return Err(format!("cuCtxSynchronize failed: {r}"));
        }
        // Every launch queued before this point has completed, so the pool may reuse anything
        // released up to the current launch counter.
        self.completed
            .store(self.issued.load(Ordering::Relaxed), Ordering::Relaxed);
        Ok(())
    }

    /// The refusal a non-launch operation gets while a graph segment is active, or `None`.
    fn graph_refusal(&self, op: SegmentOp) -> Option<&'static str> {
        graph_segment_refusal(self.graph_mode.load(Ordering::Relaxed), op)
    }

    /// Destroys the private stream a capture ran on, best effort.
    fn destroy_capture_stream(&self, stream: CuStream) {
        if stream.is_null() {
            return;
        }
        if let Ok(destroy) =
            self.sym::<unsafe extern "C" fn(CuStream) -> CuResult>(b"cuStreamDestroy_v2")
        {
            // SAFETY: the stream was created by this handle and is not in use after capture ended.
            let _ = unsafe { destroy(stream) };
        }
    }

    /// Destroys a captured graph handle, best effort.
    fn destroy_graph_handle(&self, graph: CuGraph) {
        if graph.is_null() {
            return;
        }
        if let Ok(destroy) =
            self.sym::<unsafe extern "C" fn(CuGraph) -> CuResult>(b"cuGraphDestroy")
        {
            // SAFETY: the graph was returned by cuStreamEndCapture and is not in use.
            let _ = unsafe { destroy(graph) };
        }
    }

    /// Destroys one instantiated graph executable, best effort.
    fn destroy_graph_exec(&self, exec: CuGraphExec) {
        if exec.is_null() {
            return;
        }
        if let Ok(destroy) =
            self.sym::<unsafe extern "C" fn(CuGraphExec) -> CuResult>(b"cuGraphExecDestroy")
        {
            // SAFETY: the executable is owned by this handle and no graph is launching now.
            let _ = unsafe { destroy(exec) };
        }
    }

    /// Starts a CUDA graph capture on a private stream.
    ///
    /// While the segment is active, [Cuda::launch] and [Cuda::launch_async] enqueue their kernels
    /// on the capture stream (where the driver records them) and add a [CudaLaunchRecord] to the
    /// segment. Nothing else may run: allocations, copies, synchronizes, module loads, frees and
    /// releases are refused by their own entry points, so a capture cannot silently omit device
    /// work that the direct path would have done. The stage-timing ledger and the probe meter are
    /// refused up front because both bracket launches with events and a host wait.
    ///
    /// A segment is process-wide on this handle: start, end and abort are not meant to race other
    /// threads' launches.
    ///
    /// # Errors
    /// A segment is already active, a launch bracket is armed, or the driver refuses to create the
    /// capture stream or begin capture.
    pub fn graph_capture_begin(&self) -> Result<(), String> {
        if self.graph_mode.load(Ordering::Relaxed) != GRAPH_DIRECT {
            return Err("a CUDA graph segment is already active".to_string());
        }
        if stage_timing_enabled() {
            return Err(
                "cannot capture a CUDA graph while stage timing is armed (its event bracket is not capturable)"
                    .to_string(),
            );
        }
        if meter_recording() {
            return Err(
                "cannot capture a CUDA graph while a probe meter is armed (its event bracket is not capturable)"
                    .to_string(),
            );
        }
        let create: libloading::Symbol<unsafe extern "C" fn(*mut CuStream, u32) -> CuResult> =
            self.sym(b"cuStreamCreate")?;
        let mut stream: CuStream = std::ptr::null_mut();
        // SAFETY: the symbol belongs to this context's library and the out-pointer is valid for
        // the call; the flags are the documented non-blocking flag.
        let r = unsafe { create(&mut stream, CU_STREAM_NON_BLOCKING) };
        if r != 0 {
            return Err(format!(
                "cuStreamCreate for graph capture failed: {}",
                name(r)
            ));
        }
        let begin: libloading::Symbol<unsafe extern "C" fn(CuStream, u32) -> CuResult> = match self
            .sym(b"cuStreamBeginCapture_v2")
        {
            Ok(symbol) => symbol,
            Err(_) => match self
                .sym::<unsafe extern "C" fn(CuStream, u32) -> CuResult>(b"cuStreamBeginCapture")
            {
                Ok(symbol) => symbol,
                Err(error) => {
                    self.destroy_capture_stream(stream);
                    return Err(format!(
                        "neither cuStreamBeginCapture_v2 nor cuStreamBeginCapture is available: {error}"
                    ));
                }
            },
        };
        // SAFETY: capture mode global on a stream this handle owns; no other caller uses it.
        let r = unsafe { begin(stream, CU_STREAM_CAPTURE_MODE_GLOBAL) };
        if r != 0 {
            self.destroy_capture_stream(stream);
            return Err(format!("cuStreamBeginCapture failed: {}", name(r)));
        }
        let mut control = self
            .graph
            .lock()
            .map_err(|error| format!("graph state lock: {error}"))?;
        control.capture_stream = stream;
        control.records.clear();
        self.graph_mode.store(GRAPH_CAPTURE, Ordering::Relaxed);
        Ok(())
    }

    /// Ends stream capture, instantiates the graph, and returns its handle plus the launches the
    /// segment issued in order.
    ///
    /// The capture stream is destroyed here; the graph executable is owned by this handle and is
    /// destroyed by `Drop` or by [Cuda::graph_destroy]. Failure leaves no segment active: the
    /// mode is already back to direct even when the driver refuses the end or the instantiate.
    ///
    /// # Errors
    /// No capture is active, or the driver rejects `cuStreamEndCapture` /
    /// `cuGraphInstantiateWithFlags`.
    pub fn graph_capture_end(&self) -> Result<(GraphId, Vec<CudaLaunchRecord>), String> {
        if self.graph_mode.load(Ordering::Relaxed) != GRAPH_CAPTURE {
            return Err("no CUDA graph capture is active".to_string());
        }
        let mut control = self
            .graph
            .lock()
            .map_err(|error| format!("graph state lock: {error}"))?;
        let stream = std::mem::take(&mut control.capture_stream);
        if stream.is_null() {
            self.graph_mode.store(GRAPH_DIRECT, Ordering::Relaxed);
            control.records.clear();
            return Err("graph capture is active without a capture stream".to_string());
        }
        let end: libloading::Symbol<unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult> =
            match self.sym(b"cuStreamEndCapture") {
                Ok(symbol) => symbol,
                Err(error) => {
                    self.destroy_capture_stream(stream);
                    self.graph_mode.store(GRAPH_DIRECT, Ordering::Relaxed);
                    control.records.clear();
                    return Err(error);
                }
            };
        let mut graph: CuGraph = std::ptr::null_mut();
        // SAFETY: the stream is the capture stream this handle created and is still valid.
        let r = unsafe { end(stream, &mut graph) };
        self.destroy_capture_stream(stream);
        self.graph_mode.store(GRAPH_DIRECT, Ordering::Relaxed);
        let records = std::mem::take(&mut control.records);
        if r != 0 {
            self.destroy_graph_handle(graph);
            return Err(format!("cuStreamEndCapture failed: {}", name(r)));
        }
        if graph.is_null() {
            return Err("cuStreamEndCapture returned a null graph".to_string());
        }
        let instantiate: libloading::Symbol<
            unsafe extern "C" fn(*mut CuGraphExec, CuGraph, u64) -> CuResult,
        > = match self.sym(b"cuGraphInstantiateWithFlags") {
            Ok(symbol) => symbol,
            Err(error) => {
                self.destroy_graph_handle(graph);
                return Err(error);
            }
        };
        let mut exec: CuGraphExec = std::ptr::null_mut();
        // SAFETY: the graph handle is valid and owns the nodes; flags 0 instantiate immediately.
        let r = unsafe { instantiate(&mut exec, graph, 0) };
        self.destroy_graph_handle(graph);
        if r != 0 {
            return Err(format!("cuGraphInstantiateWithFlags failed: {}", name(r)));
        }
        if exec.is_null() {
            return Err("cuGraphInstantiateWithFlags returned a null executable".to_string());
        }
        if control.execs.len() >= MAX_GRAPH_EXECUTABLES {
            self.destroy_graph_exec(exec);
            return Err(format!(
                "graph executable budget ({MAX_GRAPH_EXECUTABLES}) is full; not instantiating another"
            ));
        }
        control.next_exec += 1;
        let id = GraphId(control.next_exec);
        control.execs.insert(id.0, exec);
        Ok((id, records))
    }

    /// Abandons an active capture, best effort: ends stream capture, destroys the graph and the
    /// capture stream, and clears the recorded launches.
    ///
    /// Called by [Cuda]`s `Drop` and by the wrapper when a guarded operation ends a capture
    /// early. A no-op when no capture is active.
    pub fn graph_capture_abort(&self) {
        if self.graph_mode.load(Ordering::Relaxed) != GRAPH_CAPTURE {
            return;
        }
        let Ok(mut control) = self.graph.lock() else {
            self.graph_mode.store(GRAPH_DIRECT, Ordering::Relaxed);
            return;
        };
        let stream = std::mem::take(&mut control.capture_stream);
        self.graph_mode.store(GRAPH_DIRECT, Ordering::Relaxed);
        control.records.clear();
        if stream.is_null() {
            return;
        }
        if let Ok(end) = self
            .sym::<unsafe extern "C" fn(CuStream, *mut CuGraph) -> CuResult>(b"cuStreamEndCapture")
        {
            let mut graph: CuGraph = std::ptr::null_mut();
            // SAFETY: the stream is the capture stream this handle created; ending the capture is
            // the only valid way to leave capture state.
            let r = unsafe { end(stream, &mut graph) };
            if r == 0 {
                self.destroy_graph_handle(graph);
            }
        }
        self.destroy_capture_stream(stream);
    }

    /// Starts recording launches into host memory without capturing them.
    ///
    /// A record lets the wrapper fingerprint what a segment would launch before it decides
    /// whether a captured graph may be replayed. Nothing is issued to the driver, so a record
    /// segment can run even when no graph exists yet.
    ///
    /// # Errors
    /// A graph segment is already active.
    pub fn graph_record_begin(&self) -> Result<(), String> {
        if self.graph_mode.load(Ordering::Relaxed) != GRAPH_DIRECT {
            return Err("a CUDA graph segment is already active".to_string());
        }
        let mut control = self
            .graph
            .lock()
            .map_err(|error| format!("graph state lock: {error}"))?;
        control.records.clear();
        self.graph_mode.store(GRAPH_RECORD, Ordering::Relaxed);
        Ok(())
    }

    /// Ends a record segment and returns the launches it issued, in order.
    ///
    /// # Errors
    /// No record is active.
    pub fn graph_record_end(&self) -> Result<Vec<CudaLaunchRecord>, String> {
        if self.graph_mode.load(Ordering::Relaxed) != GRAPH_RECORD {
            return Err("no CUDA graph record is active".to_string());
        }
        let mut control = self
            .graph
            .lock()
            .map_err(|error| format!("graph state lock: {error}"))?;
        self.graph_mode.store(GRAPH_DIRECT, Ordering::Relaxed);
        Ok(std::mem::take(&mut control.records))
    }

    /// Abandons an active record segment and clears its launches. A no-op when none is active.
    pub fn graph_record_abort(&self) {
        if self.graph_mode.load(Ordering::Relaxed) != GRAPH_RECORD {
            return;
        }
        if let Ok(mut control) = self.graph.lock() {
            control.records.clear();
        }
        self.graph_mode.store(GRAPH_DIRECT, Ordering::Relaxed);
    }

    /// Queues one captured graph executable on the default stream.
    ///
    /// The replay is one issue to the driver, ordered on the default stream exactly where the
    /// direct launches it replaces would have been; the copies and readbacks around the segment
    /// keep their order. The launch counter advances by one, which is what the pooled-buffer fence
    /// needs: the replay cannot have completed before the queue reaches this point.
    ///
    /// # Errors
    /// A segment is active, the id is not one this handle instantiated, or the driver rejects the
    /// launch.
    pub fn graph_launch(&self, graph: GraphId) -> Result<(), String> {
        if self.graph_mode.load(Ordering::Relaxed) != GRAPH_DIRECT {
            return Err("a CUDA graph segment is already active".to_string());
        }
        let control = self
            .graph
            .lock()
            .map_err(|error| format!("graph state lock: {error}"))?;
        let Some(exec) = control.execs.get(&graph.0).copied() else {
            return Err(format!(
                "unknown CUDA graph handle {} (it was not instantiated by this Cuda)",
                graph.0
            ));
        };
        let launch: libloading::Symbol<unsafe extern "C" fn(CuGraphExec, CuStream) -> CuResult> =
            self.sym(b"cuGraphLaunch")?;
        // SAFETY: the executable is owned by this handle and stays alive until Drop; the null
        // stream is the default stream the surrounding copies use.
        let r = unsafe { launch(exec, std::ptr::null_mut()) };
        if r != 0 {
            return Err(format!("cuGraphLaunch failed: {}", name(r)));
        }
        self.issued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Queues recorded launches on the default stream, in order.
    ///
    /// This is the wrapper's fallback: a segment whose launch list changed since the capture
    /// cannot replay the old graph, so the launches it just recorded are executed exactly as
    /// recorded - same function handles, same grids and blocks, same argument words, same order.
    /// Nothing is patched and nothing is approximated.
    ///
    /// The wrapper only enables graphs on the async launch path, where queuing without a drain is
    /// the caller's existing contract; the next stream-ordered readback or synchronize is the
    /// observation point.
    ///
    /// # Errors
    /// A segment is active, or the driver rejects one of the launches.
    pub fn graph_replay_records(&self, records: &[CudaLaunchRecord]) -> Result<(), String> {
        if self.graph_mode.load(Ordering::Relaxed) != GRAPH_DIRECT {
            return Err("a CUDA graph segment is already active".to_string());
        }
        let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
        for (index, record) in records.iter().enumerate() {
            let mut storage: Vec<[u64; 1]> = record.args.iter().map(|a| [*a]).collect();
            let mut ptrs: Vec<*mut c_void> =
                storage.iter_mut().map(|s| s.as_mut_ptr().cast()).collect();
            // SAFETY: the record came from this handle's launch path, every argument is a u64 or a
            // pointer to one that outlives the call, and the function handle is still loaded (the
            // module cache owns it until Drop).
            let r = unsafe {
                launch(
                    record.function as CuFunction,
                    record.grid.0,
                    record.grid.1,
                    record.grid.2,
                    record.block.0,
                    record.block.1,
                    record.block.2,
                    0,
                    std::ptr::null_mut(),
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if r != 0 {
                return Err(format!(
                    "recorded launch {index} ({}) failed: {}",
                    record.describe(),
                    name(r)
                ));
            }
        }
        self.issued
            .fetch_add(records.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Destroys one graph executable this handle instantiated, best effort.
    ///
    /// The handle is not reused; a later [Cuda::graph_launch] with it fails as unknown.
    pub fn graph_destroy(&self, graph: GraphId) {
        let Ok(mut control) = self.graph.lock() else {
            return;
        };
        if let Some(exec) = control.execs.remove(&graph.0) {
            self.destroy_graph_exec(exec);
        }
    }

    /// Queues (capture mode) or records (record mode) one launch for a graph segment.
    ///
    /// Capture mode enqueues the call on the segment's own stream, where the driver records it
    /// instead of executing it, and keeps the record so a later call's launch list can be compared
    /// against the captured one. Record mode does not call the driver at all: the wrapper decides
    /// afterwards whether to replay a captured graph or to queue the recorded launches itself.
    fn graph_segment_launch(
        &self,
        mode: u8,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        let mut control = self
            .graph
            .lock()
            .map_err(|error| format!("graph state lock: {error}"))?;
        let record = CudaLaunchRecord {
            function: kernel.function as usize,
            grid,
            block,
            args: args.to_vec(),
        };
        match mode {
            GRAPH_RECORD => {
                control.records.push(record);
                Ok(())
            }
            GRAPH_CAPTURE => {
                let stream = control.capture_stream;
                if stream.is_null() {
                    return Err("graph capture is active without a capture stream".to_string());
                }
                let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
                let mut storage: Vec<[u64; 1]> = args.iter().map(|a| [*a]).collect();
                let mut ptrs: Vec<*mut c_void> =
                    storage.iter_mut().map(|s| s.as_mut_ptr().cast()).collect();
                // SAFETY: the symbol comes from this context's library, every argument is a u64
                // or a pointer to one that outlives the call, and the stream is the private
                // capture stream; the driver records this call into the graph being built.
                let r = unsafe {
                    launch(
                        kernel.function,
                        grid.0,
                        grid.1,
                        grid.2,
                        block.0,
                        block.1,
                        block.2,
                        0,
                        stream,
                        ptrs.as_mut_ptr(),
                        std::ptr::null_mut(),
                    )
                };
                if r != 0 {
                    return Err(format!(
                        "cuLaunchKernel failed while capturing a graph: {}",
                        name(r)
                    ));
                }
                control.records.push(record);
                Ok(())
            }
            _ => {
                debug_assert!(false, "graph_segment_launch reached in direct mode");
                Err("graph launch path reached in direct mode".to_string())
            }
        }
    }

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
            Ok(Cuda {
                lib,
                ctx,
                dev,
                modules: Mutex::new(ModuleCache::default()),
                pool: Mutex::new(BufferPool::default()),
                issued: AtomicU64::new(0),
                completed: AtomicU64::new(0),
                graph: Mutex::new(GraphControl::default()),
                graph_mode: AtomicU8::new(GRAPH_DIRECT),
            })
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
        if let Some(refusal) = self.graph_refusal(SegmentOp::Allocation) {
            return Err(refusal.to_string());
        }
        unsafe {
            let f: libloading::Symbol<unsafe extern "C" fn(*mut CuDeviceptr, usize) -> CuResult> =
                self.sym(b"cuMemAlloc_v2")?;
            let mut p: CuDeviceptr = 0;
            let r = f(&mut p, bytes);
            if r != 0 {
                return Err(format!("cuMemAlloc({bytes}) failed: {r}"));
            }
            *total += bytes;
            note_alloc();
            Ok(Deviceptr(p))
        }
    }

    /// Checks a pooled device buffer of exactly `bytes` out of the run scope, allocating one only
    /// when the pool has no finished buffer of that size.
    ///
    /// Pair every successful call with [Cuda::release_pooled]; the pooled allocation stays owned by
    /// the handle. A buffer that is checked in when Drop runs is freed by Drop; a buffer still
    /// checked out then is reclaimed when the primary context is released. `total` is only advanced
    /// when a fresh cuMemAlloc happens.
    ///
    /// # Errors
    /// The driver's cuMemAlloc failure code.
    pub fn alloc_pooled(&self, bytes: usize, total: &mut usize) -> Result<Deviceptr, String> {
        if let Some(refusal) = self.graph_refusal(SegmentOp::Allocation) {
            return Err(refusal.to_string());
        }
        let completed = self.completed.load(Ordering::Relaxed);
        if let Ok(mut pool) = self.pool.lock()
            && let Some(ptr) = pool.checkout(bytes, completed)
        {
            return Ok(ptr);
        }
        let ptr = self.alloc(bytes, total)?;
        if let Ok(mut pool) = self.pool.lock() {
            pool.record_allocation(ptr, bytes);
        }
        Ok(ptr)
    }

    /// Returns buffers from [Cuda::alloc_pooled] to the run scope.
    ///
    /// The buffer is stamped with the launch counter current here; [Cuda::alloc_pooled] will not
    /// hand it out again until a stream-ordered device-to-host copy or a synchronize taken after
    /// this point has proven every launch issued before the release finished. That fence is what
    /// makes reuse safe while the async path still has work queued.
    ///
    /// A release the pool refuses - a pointer it does not own, or a second release of a pointer
    /// that is already checked in - is not inserted into the free list, so one allocation can
    /// never be handed to two live kernels. The signature stays a unit because the existing
    /// callers pair it with [Cuda::alloc_pooled] and cannot ignore a `Result`; a debug build
    /// asserts on the refusal (the crate's tests pin it through `BufferPool::release`) and a
    /// release build keeps the pool consistent and drops the bad entry.
    pub fn release_pooled(&self, ptrs: &[&Deviceptr]) {
        if self.graph_refusal(SegmentOp::Release).is_some() {
            debug_assert!(false, "release_pooled inside a CUDA graph segment");
            return;
        }
        let Ok(mut pool) = self.pool.lock() else {
            return;
        };
        let needed = self.issued.load(Ordering::Relaxed);
        for ptr in ptrs {
            let released = pool.release(**ptr, needed);
            debug_assert!(
                released.is_ok(),
                "pool release refused: {:?}",
                released.err()
            );
        }
    }

    /// Forgets every pointer the run-scope pool owns and reports, in order, whether it was pooled.
    ///
    /// [Cuda::free] and [Cuda::free_buffers] use this so a pooled allocation is never passed to
    /// cuMemFree: the pool owns it, and freeing it here would leave Drop to free it a second time.
    fn forget_pooled<'a>(&self, ptrs: impl ExactSizeIterator<Item = &'a Deviceptr>) -> Vec<bool> {
        let count = ptrs.len();
        match self.pool.lock() {
            Ok(mut pool) => ptrs.map(|ptr| pool.forget(*ptr)).collect(),
            Err(_) => vec![false; count],
        }
    }

    /// Allocates managed (UVA/unified) memory with CU_MEM_ATTACH_GLOBAL.
    ///
    /// The returned pointer is valid on both the host and the device on a UMA system, so a caller
    /// can fill an arena once and launch kernels against it without a separate copy. Device code
    /// may migrate pages; the allocation is freed with [Cuda::free_managed] (or cuMemFree).
    pub fn alloc_managed(&self, bytes: usize, total: &mut usize) -> Result<Deviceptr, String> {
        if let Some(refusal) = self.graph_refusal(SegmentOp::Allocation) {
            return Err(refusal.to_string());
        }
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(*mut CuDeviceptr, usize, u32) -> CuResult,
            > = self.sym(b"cuMemAllocManaged")?;
            let mut p: CuDeviceptr = 0;
            const CU_MEM_ATTACH_GLOBAL: u32 = 1;
            let r = f(&mut p, bytes, CU_MEM_ATTACH_GLOBAL);
            if r != 0 {
                return Err(format!("cuMemAllocManaged({bytes}) failed: {r}"));
            }
            *total += bytes;
            note_alloc();
            Ok(Deviceptr(p))
        }
    }

    /// Allocates a page-locked host slab and resolves its device mapping.
    ///
    /// The slab is portable and device-mapped, so the device can read it through UVA while the
    /// host keeps the original pointer for filling and inspection. The returned [HostMapped] owns
    /// both pointers; free it with [Cuda::free_host_mapped], not cuMemFree.
    pub fn alloc_host_mapped(&self, bytes: usize, total: &mut usize) -> Result<HostMapped, String> {
        if let Some(refusal) = self.graph_refusal(SegmentOp::Allocation) {
            return Err(refusal.to_string());
        }
        unsafe {
            let alloc: libloading::Symbol<
                unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CuResult,
            > = self.sym(b"cuMemHostAlloc")?;
            const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
            const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 2;
            let mut host: *mut c_void = std::ptr::null_mut();
            let r = alloc(
                &mut host,
                bytes,
                CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP,
            );
            if r != 0 {
                return Err(format!("cuMemHostAlloc({bytes}) failed: {r}"));
            }
            let get: libloading::Symbol<
                unsafe extern "C" fn(*mut CuDeviceptr, *mut c_void, u32) -> CuResult,
            > = match self.sym(b"cuMemHostGetDevicePointer_v2") {
                Ok(symbol) => symbol,
                Err(error) => {
                    if let Ok(free) =
                        self.sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuMemFreeHost")
                    {
                        let _ = free(host);
                    }
                    return Err(error);
                }
            };
            let mut device: CuDeviceptr = 0;
            let r = get(&mut device, host, 0);
            if r != 0 {
                if let Ok(free) =
                    self.sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuMemFreeHost")
                {
                    let _ = free(host);
                }
                return Err(format!("cuMemHostGetDevicePointer failed: {r}"));
            }
            *total += bytes;
            note_alloc();
            Ok(HostMapped {
                host,
                device: Deviceptr(device),
                len: bytes,
            })
        }
    }

    /// Maps `len` bytes of `path` starting at 4096-aligned `offset` into the process address space
    /// and registers the mapping with the CUDA driver (`cuMemHostRegister_v2` with
    /// `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`), resolving the GPU
    /// `Deviceptr`. On a warm restart, pages come straight from the Linux page cache in seconds
    /// with zero read syscalls and zero H2D copies.
    pub fn mmap_registered(
        &self,
        path: &Path,
        offset: u64,
        len: usize,
        total: &mut usize,
    ) -> Result<RegisteredMmap, String> {
        if let Some(refusal) = self.graph_refusal(SegmentOp::Allocation) {
            return Err(refusal.to_string());
        }
        let file = std::fs::File::open(path)
            .map_err(|error| format!("open {} for mmap: {error}", path.display()))?;
        let mmap = unsafe {
            memmap2::MmapOptions::new()
                .offset(offset)
                .len(len)
                .map(&file)
                .map_err(|error| {
                    format!(
                        "mmap {} (offset {offset}, len {len}): {error}",
                        path.display()
                    )
                })?
        };
        let _ = mmap.advise(memmap2::Advice::Sequential);
        let host = mmap.as_ptr() as *mut c_void;
        let register_len = len.next_multiple_of(DIRECT_IO_ALIGN);
        unsafe {
            const CU_MEMHOSTREGISTER_PORTABLE: u32 = 1;
            const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 2;
            const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 8;
            let reg: Result<
                libloading::Symbol<unsafe extern "C" fn(*mut c_void, usize, u32) -> CuResult>,
                String,
            > = self
                .sym(b"cuMemHostRegister_v2")
                .or_else(|_| self.sym(b"cuMemHostRegister"));
            let registered = true;
            if let Ok(reg_fn) = reg {
                let mut r = reg_fn(
                    host,
                    register_len,
                    CU_MEMHOSTREGISTER_PORTABLE
                        | CU_MEMHOSTREGISTER_DEVICEMAP
                        | CU_MEMHOSTREGISTER_READ_ONLY,
                );
                if r != 0 {
                    // Fallback without READ_ONLY flag for drivers that reject it on file-backed mappings.
                    r = reg_fn(
                        host,
                        register_len,
                        CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP,
                    );
                }
                if r != 0 {
                    return Err(format!(
                        "cuMemHostRegister({register_len}) on {} failed: {}",
                        path.display(),
                        name(r)
                    ));
                }
            } else {
                return Err("cuMemHostRegister_v2 symbol not found in CUDA driver".to_string());
            }
            let get: libloading::Symbol<
                unsafe extern "C" fn(*mut CuDeviceptr, *mut c_void, u32) -> CuResult,
            > = self.sym(b"cuMemHostGetDevicePointer_v2")?;
            let mut device: CuDeviceptr = 0;
            let r = get(&mut device, host, 0);
            if r != 0 {
                if registered
                    && let Ok(unreg) = self.sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(
                        b"cuMemHostUnregister",
                    )
                {
                    let _ = unreg(host);
                }
                return Err(format!(
                    "cuMemHostGetDevicePointer on mmap failed: {}",
                    name(r)
                ));
            }
            *total += len;
            note_alloc();
            Ok(RegisteredMmap {
                mmap,
                host,
                device: Deviceptr(device),
                len,
                registered,
            })
        }
    }

    /// Unregisters and drops a `RegisteredMmap` created by [Cuda::mmap_registered].
    pub fn free_mmap_registered(&self, mapped: &RegisteredMmap) {
        if self.graph_refusal(SegmentOp::Free).is_some() {
            debug_assert!(false, "a buffer free inside a CUDA graph segment");
            return;
        }
        if async_launches() {
            let _ = self.synchronize();
        }
        if mapped.registered {
            unsafe {
                if let Ok(unreg) = self
                    .sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuMemHostUnregister")
                {
                    let _ = unreg(mapped.host);
                    note_free(1);
                }
            }
        }
    }

    /// Frees an allocation returned by [Cuda::alloc_managed].
    ///
    /// Managed memory is released through cuMemFree, the same call as an ordinary device
    /// allocation. The byte count is not tracked here; the caller's arena keeps its own total.
    pub fn free_managed(&self, ptr: &Deviceptr) {
        if self.graph_refusal(SegmentOp::Free).is_some() {
            debug_assert!(false, "a buffer free inside a CUDA graph segment");
            return;
        }
        // A queued kernel may still read this allocation; the async path waits before returning
        // the memory to the driver.
        if async_launches() {
            let _ = self.synchronize();
        }
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                let _ = f(ptr.0);
                note_free(1);
            }
        }
    }

    /// Frees an allocation returned by [Cuda::alloc_host_mapped].
    pub fn free_host_mapped(&self, mapped: &HostMapped) {
        if self.graph_refusal(SegmentOp::Free).is_some() {
            debug_assert!(false, "a buffer free inside a CUDA graph segment");
            return;
        }
        if async_launches() {
            let _ = self.synchronize();
        }
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(*mut c_void) -> CuResult>(b"cuMemFreeHost")
            {
                let _ = f(mapped.host);
                note_free(1);
            }
        }
    }

    /// Copies host bytes to a device allocation.
    pub fn copy_htod(&self, dst: &Deviceptr, src: &[u8]) -> Result<(), String> {
        if let Some(refusal) = self.graph_refusal(SegmentOp::Copy) {
            return Err(refusal.to_string());
        }
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, *const c_void, usize) -> CuResult,
            > = self.sym(b"cuMemcpyHtoD_v2")?;
            let metered = meter_recording();
            let started = metered.then(std::time::Instant::now);
            let r = f(dst.0, src.as_ptr().cast(), src.len());
            if let Some(started) = started {
                note_h2d(src.len(), started.elapsed().as_nanos() as u64);
            }
            if r != 0 {
                return Err(format!("cuMemcpyHtoD failed: {r}"));
            }
            Ok(())
        }
    }

    /// Copies a device allocation back to host bytes.
    pub fn copy_dtoh(&self, dst: &mut [u8], src: &Deviceptr) -> Result<(), String> {
        if let Some(refusal) = self.graph_refusal(SegmentOp::Copy) {
            return Err(refusal.to_string());
        }
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(*mut c_void, CuDeviceptr, usize) -> CuResult,
            > = self.sym(b"cuMemcpyDtoH_v2")?;
            let metered = meter_recording();
            let started = metered.then(std::time::Instant::now);
            let len = dst.len();
            let r = f(dst.as_mut_ptr().cast(), src.0, len);
            if let Some(started) = started {
                note_d2h(len, started.elapsed().as_nanos() as u64);
            }
            if r != 0 {
                return Err(format!("cuMemcpyDtoH failed: {r}"));
            }
            // cuMemcpyDtoH_v2 is synchronous and stream-ordered: it returns only after every
            // launch queued before it on the default stream has finished. That makes it the
            // observation point the async path relies on, and it advances the pool fence exactly
            // like an explicit synchronize does.
            if async_launches() {
                self.completed
                    .store(self.issued.load(Ordering::Relaxed), Ordering::Relaxed);
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
        if let Some(refusal) = self.graph_refusal(SegmentOp::Copy) {
            return Err(refusal.to_string());
        }
        // A stage copy is invisible to the host: its consumers are the kernels the caller launches
        // next, so the queued path is the same switch as [Cuda::launch]'s. Both bracket consumers
        // (stage timing, the probe meter) need the drained call's post-drain elapsed time and keep
        // the synchronous path.
        if queued_dispatch_wanted() {
            return self.copy_dtod_async(dst, src, bytes);
        }
        unsafe {
            let f: libloading::Symbol<
                unsafe extern "C" fn(CuDeviceptr, CuDeviceptr, usize) -> CuResult,
            > = self.sym(b"cuMemcpyDtoD_v2")?;
            let metered = meter_recording();
            let started = metered.then(std::time::Instant::now);
            let r = f(dst.0, src.0, bytes);
            if let Some(started) = started {
                note_dtod(bytes, started.elapsed().as_nanos() as u64);
            }
            if r != 0 {
                return Err(format!("cuMemcpyDtoD failed: {r}"));
            }
            Ok(())
        }
    }

    /// Queues one device-to-device copy on the default stream and returns without draining.
    ///
    /// The copy sits in the default stream's order exactly where the synchronous call sat: it runs
    /// after every launch issued before it and before every launch issued after it, so a staging
    /// buffer it fills is read by the next launch as the drained copy's bytes were. The host must
    /// not read the destination until an observation point - [Cuda::copy_dtoh] or
    /// [Cuda::synchronize] - and the launch counter is advanced so the pooled-buffer fence keeps
    /// the buffer out of circulation until one of those proves the copy finished.
    ///
    /// Not metered: the probe's `dtod_ms` bucket times the drained call, and an armed meter keeps
    /// [Cuda::copy_dtod] on that drained path, so a queued copy is never inside a meter window.
    ///
    /// # Errors
    /// The driver's cuMemcpyDtoDAsync failure code.
    pub fn copy_dtod_async(
        &self,
        dst: &Deviceptr,
        src: &Deviceptr,
        bytes: usize,
    ) -> Result<(), String> {
        if let Some(refusal) = self.graph_refusal(SegmentOp::Copy) {
            return Err(refusal.to_string());
        }
        let copy: libloading::Symbol<
            unsafe extern "C" fn(CuDeviceptr, CuDeviceptr, usize, *mut c_void) -> CuResult,
        > = self.sym(b"cuMemcpyDtoDAsync_v2")?;
        // SAFETY: the symbol comes from this context's library, both pointers are device addresses
        // this process owns, and the null stream is the default stream the launches use.
        let r = unsafe { copy(dst.0, src.0, bytes, std::ptr::null_mut()) };
        if r != 0 {
            return Err(format!("cuMemcpyDtoDAsync failed: {r}"));
        }
        self.issued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Loads one non-flash_mla kernel from a cubin image.
    ///
    /// The cubin symbol is chosen by the KernelId, not by the caller, so the returned handle can
    /// never refer to the flash_mla kernel.
    pub fn load_kernel(&self, cubin: &[u8], id: KernelId) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, id.symbol())?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: id.cubin_names()[0],
            },
        ))
    }

    /// Loads the router's stage-two top-6 merge kernel from the router cubin.
    ///
    /// Stage two is a second symbol in the same cubin as [KernelId::MoeRouter]. It is not a new
    /// [KernelId] because it has no cubin of its own and no independent baseline entry: the
    /// router's cubin pin covers both symbols. The symbol is deliberately not resolvable through
    /// [KernelId::from_cubin_name], so a harness cannot accidentally launch the merge as a router.
    pub fn load_router_topk(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "moe_router_topk_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "moe_router_topk_sm121",
            },
        ))
    }
    /// Loads the K8 batched certified-head symbol by name from the cert cubin.
    pub fn load_cert_lm_head_batched(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "cert_lm_head_batched_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "cert_lm_head_batched_sm121",
            },
        ))
    }

    /// Loads the five layer-seam symbols from the one cubin the whole-layer port pins
    /// (kernels/layer_seams_sm121.cu).
    ///
    /// The five kernels share a cubin and its baseline pin, so they are resolved together: a staged
    /// image that carries one symbol and not the others is refused here rather than surfacing as a
    /// nil function at launch. The cache interns the image bytes once; each symbol still resolves to
    /// its own module handle (the driver hands out one module per load), so the caller frees all
    /// five.
    ///
    /// # Errors
    /// The driver's error when the cubin does not load or is missing one of the five symbols.
    #[allow(clippy::type_complexity)]
    pub fn load_layer_seams(
        &self,
        cubin: &[u8],
    ) -> Result<
        (
            Module,
            Kernel,
            Module,
            Kernel,
            Module,
            Kernel,
            Module,
            Kernel,
            Module,
            Kernel,
            Module,
            Kernel,
            Module,
            Kernel,
            Module,
            Kernel,
        ),
        String,
    > {
        use crate::layer_seams_launch as seams;
        let load = |symbol: &'static str| -> Result<(Module, Kernel), String> {
            let (module, function) = self.load_function(cubin, symbol)?;
            Ok((
                Module(module),
                Kernel {
                    function,
                    name: symbol,
                },
            ))
        };
        let entry = load(seams::ENTRY_KERNEL)?;
        let ffn = load(seams::FFN_KERNEL)?;
        let prologue = load(seams::PROLOGUE_KERNEL)?;
        let attn_residual = load(seams::ATTN_RESIDUAL_KERNEL)?;
        let ffn_epilogue = load(seams::FFN_EPILOGUE_KERNEL)?;
        let shared_swiglu = load(seams::SHARED_SWIGLU_KERNEL)?;
        let prologue_bf16 = load(seams::PROLOGUE_BF16_KERNEL)?;
        let rope_rows = load(seams::ROPE_ROWS_KERNEL)?;
        Ok((
            entry.0,
            entry.1,
            ffn.0,
            ffn.1,
            prologue.0,
            prologue.1,
            attn_residual.0,
            attn_residual.1,
            ffn_epilogue.0,
            ffn_epilogue.1,
            shared_swiglu.0,
            shared_swiglu.1,
            prologue_bf16.0,
            prologue_bf16.1,
            rope_rows.0,
            rope_rows.1,
        ))
    }

    /// Loads the K1 warp-per-column dense symbol from the dense cubin.
    ///
    /// The scalar `b12x_mxfp8_dense_sm121` symbol remains the pinned reference and the incumbent
    /// harness target; this second symbol is the one the chain switches to after its own
    /// tolerance-qualified run. It shares the cubin pin with the scalar symbol.
    pub fn load_dense_warp(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_dense_warp_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_dense_warp_sm121",
            },
        ))
    }

    /// Loads the kv-path rope rotation by name from the kv_path cubin.
    ///
    /// kernels/kv_path_sm121.cu carries three symbols and one baseline entry, so the symbols have
    /// no KernelId: the by-name route is the same one the M=1 dense and decode GEMV symbols use for
    /// a kernel with its own cubin.
    pub fn load_rope_rotate(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "rope_rotate_bf16")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "rope_rotate_bf16",
            },
        ))
    }

    /// Loads the kv-path fused kv_norm + rope kernel by name from the kv_path cubin.
    pub fn load_kv_norm_rope(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "kv_norm_rope_bf16")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "kv_norm_rope_bf16",
            },
        ))
    }

    /// ref_select_sm121: the reference candidate select, loaded by name from its own cubin (the
    /// pin is owned by the launch module's constants, and no KernelId routes to it).
    ///
    /// # Errors
    /// Returns the driver's error when the cubin does not contain the symbol.
    pub fn load_ref_select(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "ref_select_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "ref_select_sm121",
            },
        ))
    }

    /// Loads the kv-path cache append by name from the kv_path cubin.
    pub fn load_kv_append(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "kv_append_bf16")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "kv_append_bf16",
            },
        ))
    }

    /// Loads the kv-path window quantiser by name from the kv_path cubin.
    pub fn load_window_quant(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "window_quant_fp8_bf16")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "window_quant_fp8_bf16",
            },
        ))
    }

    /// engram_write_bf16: the Engram residual write, loaded by name from its own cubin (the pin is
    /// owned by the launch module's constants, and no KernelId routes to it).
    ///
    /// # Errors
    /// Returns the driver's error when the cubin does not contain the symbol.
    pub fn load_engram_write(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "engram_write_bf16")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "engram_write_bf16",
            },
        ))
    }

    /// engram_narrow_bf16: the f32 -> bf16 stream conversion, loaded by name from its own cubin (the
    /// pin is owned by the launch module's constants, and no KernelId routes to it).
    ///
    /// # Errors
    /// Returns the driver's error when the cubin does not contain the symbol.
    pub fn load_engram_narrow(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "engram_narrow_bf16")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "engram_narrow_bf16",
            },
        ))
    }

    /// engram_widen_f32: the bf16 -> f32 stream conversion, loaded by name from the same cubin as
    /// the narrow.
    ///
    /// # Errors
    /// Returns the driver's error when the cubin does not contain the symbol.
    pub fn load_engram_widen(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "engram_widen_f32")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "engram_widen_f32",
            },
        ))
    }

    /// Loads the WS-D columns-only M=1 dense symbol from its own cubin by name.
    ///
    /// This symbol lives in its own cubin (`kernels/b12x_mxfp8_dense_m1_sm121.cu`), not in the
    /// pinned dense cubin: the addition must not move the pinned kernel's bytes or digest. The
    /// pinned `b12x_mxfp8_dense_sm121` symbol stays the reference oracle and the frozen ABI id.
    pub fn load_dense_m1(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_dense_m1_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_dense_m1_sm121",
            },
        ))
    }

    /// Loads the WS-D decode GEMV symbol from its own cubin by name.
    ///
    /// This symbol lives in its own cubin (`kernels/b12x_mxfp8_gemv_m1_sm121.cu`), not in the
    /// pinned dense cubin: the addition must not move the pinned kernel's bytes or digest, and the
    /// pinned `b12x_mxfp8_dense_sm121` symbol stays the reference oracle and the frozen ABI id.
    pub fn load_dense_gemv(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        self.load_dense_gemv_mode(cubin, "b12x_mxfp8_gemv_m1_sm121")
    }

    /// Loads the SwiGLU + MX-quantise epilogue (`swiglu_mxfp8_quant_sm121`) by name.
    ///
    /// The shared expert's post-SwiGLU quantise is the wide (cols = 2304) shape of this entry
    /// point: gate and up are bf16, the activation is rounded to bf16 exactly as the released model
    /// casts it, and the result is the E4M3 + ue8m0 operand a down projection consumes.
    pub fn load_swiglu_quant(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        self.load_dense_gemv_mode(cubin, "swiglu_mxfp8_quant_sm121")
    }

    /// Loads one instantiation of the decode GEMV by name.
    ///
    /// The kernel file emits one templated body as four symbols - the dense projections
    /// (`b12x_mxfp8_gemv_m1_sm121`), the shared expert's coded arm (`b12x_mxfp8_gemv_m1_nf_sm121`),
    /// the compressor seam (`b12x_mxfp8_gemv_m1_nf_f32a_sm121`) and the shared expert's
    /// f32-reference arm (`b12x_mxfp8_gemv_m1_f32a_f32o_sm121`) - all in the same cubin, so the
    /// mode is a symbol lookup rather than a second module.
    pub fn load_dense_gemv_mode(
        &self,
        cubin: &[u8],
        symbol: &'static str,
    ) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, symbol)?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: symbol,
            },
        ))
    }

    /// Loads the K1-b A-reuse tiled dense symbol by name.
    pub fn load_dense_tile(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_dense_tile_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_dense_tile_sm121",
            },
        ))
    }

    /// Loads the K1-c(a) parallel-K dense symbol by name.
    pub fn load_dense_par_k(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_dense_par_k_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_dense_par_k_sm121",
            },
        ))
    }

    /// Loads the K8 batched mHC main and init symbols by name.
    pub fn load_mega_mhc_batched(
        &self,
        cubin: &[u8],
    ) -> Result<(Module, Kernel, Module, Kernel), String> {
        let (mm, mf) = self.load_function(cubin, "mega_mhc_fused_bf16_batched_sm121")?;
        let (im, init) = self.load_function(cubin, "mega_mhc_init_sm121")?;
        Ok((
            Module(mm),
            Kernel {
                function: mf,
                name: "mega_mhc_fused_bf16_batched_sm121",
            },
            Module(im),
            Kernel {
                function: init,
                name: "mega_mhc_init_sm121",
            },
        ))
    }

    /// Loads the K8 batched LM-head reduction symbol by name.
    pub fn load_lm_head_reduce_batched(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "lm_head_reduce_batched_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "lm_head_reduce_batched_sm121",
            },
        ))
    }

    /// Loads both K1-c(b) split-K symbols by name.
    pub fn load_dense_split_k(
        &self,
        cubin: &[u8],
    ) -> Result<(Module, Kernel, Module, Kernel), String> {
        let (pm, pf) = self.load_function(cubin, "b12x_mxfp8_dense_split_k_sm121")?;
        let (cm, cf) = self.load_function(cubin, "b12x_mxfp8_dense_split_k_combine_sm121")?;
        Ok((
            Module(pm),
            Kernel {
                function: pf,
                name: "b12x_mxfp8_dense_split_k_sm121",
            },
            Module(cm),
            Kernel {
                function: cf,
                name: "b12x_mxfp8_dense_split_k_combine_sm121",
            },
        ))
    }

    /// Loads the D49 ring-capacity CSA2 symbol by name.
    pub fn load_flash_mla_csa2_ring(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "flash_mla_csa2_ring_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "flash_mla_csa2_ring_sm121",
            },
        ))
    }

    /// Loads the K8 batched CSA2 sequence symbol by name.
    pub fn load_flash_mla_csa2_seq_batched(
        &self,
        cubin: &[u8],
    ) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "flash_mla_csa2_seq_batched_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "flash_mla_csa2_seq_batched_sm121",
            },
        ))
    }

    /// Loads the K1 warp-per-column wo_a symbol from the wo_a cubin.
    pub fn load_wo_a_warp(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_wo_a_warp_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_wo_a_warp_sm121",
            },
        ))
    }

    /// Loads the K1-b A-reuse tiled wo_a symbol by name.
    pub fn load_wo_a_tile(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_wo_a_tile_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_wo_a_tile_sm121",
            },
        ))
    }

    /// Loads the K8 batched wo_a symbol by name.
    pub fn load_wo_a_batched(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "b12x_mxfp8_wo_a_batched_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: "b12x_mxfp8_wo_a_batched_sm121",
            },
        ))
    }

    /// Loads the flash_mla kernel. Used only by crate::cuda_harness's typed loader.
    pub(crate) fn load_flash_mla(&self, cubin: &[u8]) -> Result<(Module, Kernel), String> {
        let (module, function) = self.load_function(cubin, "flash_mla_csa2_sm121")?;
        Ok((
            Module(module),
            Kernel {
                function,
                name: FLASH_MLA_CUBIN,
            },
        ))
    }

    fn load_function(&self, cubin: &[u8], name: &str) -> Result<(CuModule, CuFunction), String> {
        // M2/C1: a cubin image the run has already loaded resolves to its cached module and
        // function. The key is the image's own bytes (interned byte-for-byte, so a changed image
        // is a new id) plus the symbol name; a caller that hands over different bytes can never
        // hit another image's entry.
        let image = self
            .modules
            .lock()
            .map_err(|error| format!("module cache lock: {error}"))?
            .intern(cubin);
        {
            let cache = self
                .modules
                .lock()
                .map_err(|error| format!("module cache lock: {error}"))?;
            if let Some(hit) = cache.function(image, name) {
                return Ok(hit);
            }
        }
        if let Some(refusal) = self.graph_refusal(SegmentOp::ModuleLoad) {
            return Err(refusal.to_string());
        }
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
            note_module_load();
            self.modules
                .lock()
                .map_err(|error| format!("module cache lock: {error}"))?
                .insert_function(image, name, module, func);
            Ok((module, func))
        }
    }

    /// Launches one non-flash_mla kernel without draining the context.
    ///
    /// This is the M2/C3 async entry point: it queues the launch and returns. The caller may rely
    /// on the result only after a stream-ordered device-to-host copy (cuMemcpyDtoH is synchronous
    /// and waits for every earlier launch on the default stream), an explicit
    /// [Cuda::synchronize], or [Cuda::launch_repeated]'s own event wait. It must not release the
    /// module or the memory the kernel reads before one of those boundaries; the buffer pool and
    /// the module cache exist so the four-rank loop never has to. Arithmetic is unchanged: only
    /// the host's wait moves.
    ///
    /// # Errors
    /// The driver's cuLaunchKernel failure code.
    pub fn launch_async(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        let mode = self.graph_mode.load(Ordering::Relaxed);
        if mode != GRAPH_DIRECT {
            return self.graph_segment_launch(mode, kernel, grid, block, args);
        }
        let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
        let mut storage: Vec<[u64; 1]> = args.iter().map(|a| [*a]).collect();
        let mut ptrs: Vec<*mut c_void> =
            storage.iter_mut().map(|s| s.as_mut_ptr().cast()).collect();
        // SAFETY: the symbol comes from this context's library, every argument is a u64 or a
        // pointer to one that outlives the call, and the kernel handle is crate-owned.
        let r = unsafe {
            launch(
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
            )
        };
        if r != 0 {
            return Err(format!("cuLaunchKernel failed: {r}"));
        }
        self.issued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Launches one non-flash_mla kernel, draining the context unless the run selected async.
    ///
    /// With SPARKFLASH_STAGE_TIMING=1, or with the probe meter armed, the launch is bracketed by
    /// one cuEventCreate pair recorded
    /// immediately around cuLaunchKernel and resolved after the existing cuCtxSynchronize; the
    /// bracketed launches are the same launches in the same order and nothing else changes. Without
    /// the variable no event is created, no allocation happens and the launch path is unchanged.
    ///
    /// The forward perf probe arms the same bracket through [probe_meter_arm] instead: the event
    /// elapsed time feeds the probe's per-block kernel_ns and the cuCtxSynchronize call feeds its
    /// host-sync bucket. Both switches share one event pair, so a run with both set pays for the
    /// bracket once. With neither set the bracket is skipped entirely and the path is unchanged.
    pub fn launch(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        // A graph segment intercepts before the bracket choice: the launch belongs to a capture
        // stream, and a bracketed launch could not be captured at all.
        let mode = self.graph_mode.load(Ordering::Relaxed);
        if mode != GRAPH_DIRECT {
            return self.graph_segment_launch(mode, kernel, grid, block, args);
        }
        // The no-sync probe queues the launch and records one event pair for it; the elapsed times
        // are resolved once at run end, so a block's wall never contains the device wait.
        if probe_meter_deferred() {
            return self.launch_deferred(kernel, grid, block, args);
        }
        // The bracket consumers need the drain that follows each launch, so they always take the
        // synchronous path. Everything else takes the async path when the run selected it.
        if queued_dispatch_wanted() {
            return self.launch_async(kernel, grid, block, args);
        }
        unsafe {
            let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
            let mut storage: Vec<[u64; 1]> = args.iter().map(|a| [*a]).collect();
            let mut ptrs: Vec<*mut c_void> =
                storage.iter_mut().map(|s| s.as_mut_ptr().cast()).collect();
            let timing = stage_timing_enabled();
            let metered = probe_meter_armed();
            let mut start_event: CuEvent = std::ptr::null_mut();
            let mut stop_event: CuEvent = std::ptr::null_mut();
            // One bracket serves both consumers: the stage ledger and the probe meter.
            let bracket = timing || metered;
            if bracket {
                let create: libloading::Symbol<
                    unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult,
                > = self.sym(b"cuEventCreate")?;
                let record: libloading::Symbol<
                    unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult,
                > = self.sym(b"cuEventRecord")?;
                let mut r = create(&mut start_event, 0);
                if r == 0 {
                    r = create(&mut stop_event, 0);
                }
                if r != 0 {
                    return Err(format!(
                        "cuEventCreate for the stage timing bracket failed: {r}"
                    ));
                }
                let r = record(start_event, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord(start) failed: {r}"));
                }
            }
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
            self.issued.fetch_add(1, Ordering::Relaxed);
            if bracket {
                let record: libloading::Symbol<
                    unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult,
                > = self.sym(b"cuEventRecord")?;
                let r = record(stop_event, std::ptr::null_mut());
                if r != 0 {
                    return Err(format!("cuEventRecord(stop) failed: {r}"));
                }
            }
            let sync: libloading::Symbol<unsafe extern "C" fn() -> CuResult> =
                self.sym(b"cuCtxSynchronize")?;
            let sync_started = metered.then(std::time::Instant::now);
            let r = sync();
            if let Some(started) = sync_started {
                note_sync(started.elapsed().as_nanos() as u64);
            }
            if r != 0 {
                return Err(format!("cuCtxSynchronize failed: {r}"));
            }
            self.completed
                .store(self.issued.load(Ordering::Relaxed), Ordering::Relaxed);
            if bracket {
                let elapsed: libloading::Symbol<
                    unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
                > = self.sym(b"cuEventElapsedTime")?;
                let destroy: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                    self.sym(b"cuEventDestroy")?;
                let mut ms: f32 = 0.0;
                let r = elapsed(&mut ms, start_event, stop_event);
                destroy(start_event);
                destroy(stop_event);
                if r != 0 {
                    return Err(format!("cuEventElapsedTime failed: {r}"));
                }
                let kernel_ns = (f64::from(ms) * 1_000_000.0) as u64;
                if timing {
                    record_launch(kernel.name, kernel_ns);
                }
                if metered {
                    note_launch(kernel_ns);
                }
            }
            Ok(())
        }
    }

    /// Launches one kernel on the no-sync probe path: queued, bracketed by one event pair, and
    /// resolved at run end by [Cuda::probe_meter_harvest].
    ///
    /// The events bracket exactly the cuLaunchKernel, so their elapsed time is the kernel's device
    /// time and not the host gap around it; nothing drains, so the caller must not read anything
    /// the kernel writes until an observation point. The launch counter and the meter's launch
    /// census advance exactly as on the drained path.
    fn launch_deferred(
        &self,
        kernel: Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[u64],
    ) -> Result<(), String> {
        unsafe {
            let launch: libloading::Symbol<LaunchFn> = self.sym(b"cuLaunchKernel")?;
            let create: libloading::Symbol<unsafe extern "C" fn(*mut CuEvent, u32) -> CuResult> =
                self.sym(b"cuEventCreate")?;
            let record: libloading::Symbol<unsafe extern "C" fn(CuEvent, *mut c_void) -> CuResult> =
                self.sym(b"cuEventRecord")?;
            let destroy: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
                self.sym(b"cuEventDestroy")?;
            let mut start_event: CuEvent = std::ptr::null_mut();
            let mut stop_event: CuEvent = std::ptr::null_mut();
            let mut r = create(&mut start_event, 0);
            if r == 0 {
                r = create(&mut stop_event, 0);
            }
            if r != 0 {
                if !start_event.is_null() {
                    destroy(start_event);
                }
                if !stop_event.is_null() {
                    destroy(stop_event);
                }
                return Err(format!(
                    "cuEventCreate for the no-sync probe bracket failed: {r}"
                ));
            }
            let r = record(start_event, std::ptr::null_mut());
            if r != 0 {
                destroy(start_event);
                destroy(stop_event);
                return Err(format!("cuEventRecord(start) failed: {r}"));
            }
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
                destroy(start_event);
                destroy(stop_event);
                return Err(format!("cuLaunchKernel failed: {r}"));
            }
            let r = record(stop_event, std::ptr::null_mut());
            if r != 0 {
                destroy(start_event);
                destroy(stop_event);
                return Err(format!("cuEventRecord(stop) failed: {r}"));
            }
            self.issued.fetch_add(1, Ordering::Relaxed);
            note_launch_queued();
            if let Ok(mut events) = DEFERRED_EVENTS.lock() {
                events.push((start_event as usize, stop_event as usize));
            }
            Ok(())
        }
    }

    /// Resolves the no-sync probe's per-launch brackets: one cuEventElapsedTime per queued launch,
    /// in launch order, then destroys the events.
    ///
    /// The caller must have observed the context first (the run's final [Cuda::synchronize]); an
    /// elapsed time read before the work has run is not a measurement. The returned vector holds one
    /// kernel-ns figure per launch the deferred meter counted since the run armed it, so a block
    /// row's kernel_ns is the sum of its own launch-count slice of this vector.
    ///
    /// # Errors
    /// A cuEventElapsedTime failure, with the launch index named. Every created event is destroyed
    /// either way.
    pub fn probe_meter_harvest(&self) -> Result<Vec<u64>, String> {
        let events: Vec<(usize, usize)> = match DEFERRED_EVENTS.lock() {
            Ok(mut guard) => guard.drain(..).collect(),
            Err(error) => error.into_inner().drain(..).collect(),
        };
        if events.is_empty() {
            return Ok(Vec::new());
        }
        let elapsed: libloading::Symbol<
            unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
        > = self.sym(b"cuEventElapsedTime")?;
        let destroy: libloading::Symbol<unsafe extern "C" fn(CuEvent) -> CuResult> =
            self.sym(b"cuEventDestroy")?;
        let mut durations = Vec::with_capacity(events.len());
        let mut failure: Option<String> = None;
        for (index, (start_event, stop_event)) in events.into_iter().enumerate() {
            let start = start_event as CuEvent;
            let stop = stop_event as CuEvent;
            let mut ms: f32 = 0.0;
            // SAFETY: both handles are events this meter created for one launch, the out-pointer is
            // valid for the call, and the caller's synchronize has completed both of them.
            let r = unsafe { elapsed(&mut ms, start, stop) };
            unsafe {
                destroy(start);
                destroy(stop);
            }
            if r == 0 {
                durations.push((f64::from(ms) * 1_000_000.0) as u64);
            } else if failure.is_none() {
                failure = Some(format!(
                    "cuEventElapsedTime for deferred launch {index} failed: {r}"
                ));
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(durations),
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
        if let Some(refusal) = self.graph_refusal(SegmentOp::RepeatedLaunch) {
            return Err(refusal.to_string());
        }
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
    ///
    /// M2/C1: a module the run-scope cache owns is left loaded (its Drop unloads it), so the
    /// per-layer free becomes a no-op on the hot path. Everything still unloaded or freed here is
    /// guarded by a context drain on the async path, because the driver must not recycle memory a
    /// queued kernel may still read.
    ///
    /// A pointer the run-scope pool owns is not passed to cuMemFree: it is forgotten from the
    /// pool's bookkeeping and reclaimed when the primary context is released, so Drop can never
    /// free it a second time. Return a pooled buffer with [Cuda::release_pooled] instead.
    pub fn free(&self, module: Module, ptrs: &[Deviceptr]) {
        if self.graph_refusal(SegmentOp::Free).is_some() {
            debug_assert!(false, "a buffer free inside a CUDA graph segment");
            return;
        }
        let cached = self
            .modules
            .lock()
            .map(|cache| cache.owns_module(module.0))
            .unwrap_or(false);
        let forgotten = self.forget_pooled(ptrs.iter());
        if async_launches() && (!cached || !ptrs.is_empty()) {
            let _ = self.synchronize();
        }
        let freed = forgotten.iter().filter(|is_pooled| !**is_pooled).count();
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for (p, is_pooled) in ptrs.iter().zip(&forgotten) {
                    if !*is_pooled {
                        let _ = f(p.0);
                    }
                }
                note_free(freed as u64);
            }
            if !cached
                && let Ok(f) =
                    self.sym::<unsafe extern "C" fn(CuModule) -> CuResult>(b"cuModuleUnload")
            {
                let _ = f(module.0);
            }
            // The primary context is released by Drop, on every exit path.
        }
    }

    /// Frees device buffers without unloading the module that produced them.
    ///
    /// Stage 7 runs several different-shaped launches per loaded module, so the module has to
    /// outlive them. As in [Cuda::free], a pointer the run-scope pool owns is forgotten from the
    /// pool's bookkeeping instead of being passed to cuMemFree.
    pub fn free_buffers(&self, ptrs: &[&Deviceptr]) {
        if self.graph_refusal(SegmentOp::Free).is_some() {
            debug_assert!(false, "a buffer free inside a CUDA graph segment");
            return;
        }
        let forgotten = self.forget_pooled(ptrs.iter().copied());
        if async_launches() {
            let _ = self.synchronize();
        }
        let freed = forgotten.iter().filter(|is_pooled| !**is_pooled).count();
        unsafe {
            if let Ok(f) =
                self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
            {
                for (p, is_pooled) in ptrs.iter().zip(&forgotten) {
                    if !*is_pooled {
                        let _ = f(p.0);
                    }
                }
                note_free(freed as u64);
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
        // A capture still active when Drop runs has to be ended before the context is drained:
        // cuStreamEndCapture is the only valid exit from capture state, and the executable it
        // would produce is not wanted.
        self.graph_capture_abort();
        self.graph_record_abort();
        // No queued launch may outlive the handles it reads: drain first, then unload the cached
        // modules and free the pooled buffers, then release this process's context reference.
        let _ = self.synchronize();
        unsafe {
            if let Ok(mut cache) = self.modules.lock() {
                if let Ok(unload) =
                    self.sym::<unsafe extern "C" fn(CuModule) -> CuResult>(b"cuModuleUnload")
                {
                    for (_key, (module, _function)) in cache.modules.drain() {
                        let _ = unload(module);
                    }
                }
                cache.module_ptrs.clear();
            }
            if let Ok(mut pool) = self.pool.lock() {
                if let Ok(free) =
                    self.sym::<unsafe extern "C" fn(CuDeviceptr) -> CuResult>(b"cuMemFree_v2")
                {
                    // Only the free lists are authoritative: a pointer still checked out belongs
                    // to a caller, and a pointer forgotten by free()/free_buffers() is no longer
                    // in the bookkeeping. Neither may be freed here a second time; the primary
                    // context release below reclaims them.
                    for list in pool.free.values() {
                        for (ptr, _needed) in list {
                            let _ = free(ptr.0);
                        }
                    }
                }
                pool.free.clear();
                pool.sizes.clear();
                pool.checked_in.clear();
            }
            if let Ok(mut control) = self.graph.lock() {
                for (_id, exec) in std::mem::take(&mut control.execs) {
                    self.destroy_graph_exec(exec);
                }
                control.next_exec = 0;
            }
            if let Ok(release_v2) =
                self.sym::<unsafe extern "C" fn(i32) -> CuResult>(b"cuDevicePrimaryCtxRelease_v2")
            {
                let _ = release_v2(self.dev);
            }
        }
    }
}

#[cfg(test)]
mod stage_timing_tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset() {
        *STAGE_RECORDER.lock().expect("lock") = Some(StageRecorder::default());
    }

    fn scope(phase: &'static str, step: usize, layer: usize, position: usize) {
        let mut guard = STAGE_RECORDER.lock().expect("lock");
        let recorder = guard.as_mut().expect("armed");
        recorder.scope = StageScope {
            phase,
            step,
            layer,
            position,
        };
    }

    #[test]
    fn the_ledger_aggregates_per_kernel_phase_and_step() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        reset();
        scope("decode", 2, 5, 7);
        record_launch("b12x_mxfp8_dense_sm121", 1_000);
        record_launch("b12x_mxfp8_dense_sm121", 3_000);
        record_launch("moe_router_sm121", 500);
        let report = stage_timing_report().expect("report");
        assert_eq!(report.kernels.len(), 2);
        let dense = &report.kernels[0];
        assert_eq!(dense.phase, "decode");
        assert_eq!(dense.kernel, "b12x_mxfp8_dense_sm121");
        assert_eq!(dense.calls, 2);
        assert_eq!(dense.total_ns, 4_000);
        assert_eq!(dense.mean_ns, 2_000);
        assert_eq!(dense.min_ns, 1_000);
        assert_eq!(dense.max_ns, 3_000);
        assert_eq!(dense.first_launch_order, 0);
        assert_eq!(report.steps.len(), 1);
        assert_eq!(report.steps[0].step, 2);
        assert_eq!(report.steps[0].launches, 3);
        assert_eq!(report.steps[0].kernel_ns, 4_500);
        *STAGE_RECORDER.lock().expect("lock") = None;
    }

    #[test]
    fn the_ledger_caps_the_distinct_kernel_names() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        reset();
        scope("decode", 0, 0, 0);
        let names: Vec<&'static str> = (0..STAGE_TIMING_KERNEL_CAP + 8)
            .map(|index| Box::leak(format!("kernel_{index}").into_boxed_str()) as &'static str)
            .collect();
        for name in &names {
            record_launch(name, 10);
        }
        let report = stage_timing_report().expect("report");
        assert_eq!(report.kernels.len(), STAGE_TIMING_KERNEL_CAP);
        assert_eq!(report.dropped_names, 8);
        *STAGE_RECORDER.lock().expect("lock") = None;
    }

    #[test]
    fn the_ledger_is_absent_until_armed_and_the_bracket_is_off_by_default() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        *STAGE_RECORDER.lock().expect("lock") = None;
        assert!(stage_timing_report().is_none());
        assert_eq!(stage_timing_launches(), 0);
    }

    #[test]
    fn warmup_steps_are_marked_so_cold_and_steady_state_can_be_reported_separately() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        reset();
        stage_timing_set_warm(true);
        scope("decode", 0, 0, 0);
        record_launch("b12x_mxfp8_dense_sm121", 9_000);
        stage_timing_set_warm(false);
        scope("decode", 1, 0, 0);
        record_launch("b12x_mxfp8_dense_sm121", 1_000);
        let report = stage_timing_report().expect("report");
        assert_eq!(report.steps.len(), 2);
        assert!(report.steps[0].warm, "step 0 is the warmup step");
        assert!(!report.steps[1].warm, "step 1 is steady state");
        *STAGE_RECORDER.lock().expect("lock") = None;
    }
}

#[cfg(test)]
mod probe_meter_tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn the_meter_is_off_by_default_arms_explicitly_and_reads_back_what_was_added() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        probe_meter_arm(false);
        assert!(
            !probe_meter_armed(),
            "the default path leaves the meter off"
        );
        assert!(probe_meter_read().is_clear());
        note_launch(1_000);
        note_sync(200);
        note_h2d(64, 10);
        assert!(
            probe_meter_read().is_clear(),
            "a disarmed meter records nothing"
        );

        probe_meter_arm(true);
        assert!(probe_meter_armed());
        assert!(
            probe_meter_read().is_clear(),
            "arming clears the cells so one run reports only its own work"
        );
        assert!(probe_meter_read().plus(probe_meter_read()).is_clear());
        let start = probe_meter_read();
        note_launch(1_000);
        note_launch(3_000);
        note_sync(200);
        note_h2d(64, 10);
        note_d2h(128, 20);
        note_dtod(256, 30);
        note_alloc();
        note_alloc();
        note_free(2);
        note_module_load();
        let delta = probe_meter_read().since(start);
        assert_eq!(delta.launches, 2);
        assert_eq!(delta.kernel_ns, 4_000);
        assert_eq!(delta.syncs, 1);
        assert_eq!(delta.sync_ns, 200);
        assert_eq!(delta.h2d_count, 1);
        assert_eq!(delta.h2d_bytes, 64);
        assert_eq!(delta.h2d_ns, 10);
        assert_eq!(delta.d2h_count, 1);
        assert_eq!(delta.d2h_bytes, 128);
        assert_eq!(delta.d2h_ns, 20);
        assert_eq!(delta.dtod_count, 1);
        assert_eq!(delta.dtod_bytes, 256);
        assert_eq!(delta.dtod_ns, 30);
        assert_eq!(delta.alloc_count, 2);
        assert_eq!(delta.free_count, 2);
        assert_eq!(delta.module_load_count, 1);
        assert!(!delta.is_clear());
        probe_meter_arm(false);
    }

    #[test]
    fn a_reading_is_never_negative_and_an_unchanged_meter_is_clear() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        probe_meter_arm(true);
        let before = probe_meter_read();
        note_launch(5);
        let after = probe_meter_read();
        assert!(after.since(after).is_clear());
        let backwards = before.since(after);
        assert_eq!(backwards, ProbeMeterReading::default());
        probe_meter_arm(false);
    }
}

/// M2/C3 + C1 + C2 mode and cache invariants, tested without a device.
///
/// The driver path itself cannot run on this host, so the parts that decide *what* the driver will
/// do are pure data structures here: the launch-mode selection, the cubin interning key and the
/// buffer-pool reuse fence. That is the contract the device window exercises.
#[cfg(test)]
mod async_launch_cache_tests {
    use super::*;

    /// Serialises the tests that touch the process-wide async switch.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn the_forward_opt_out_is_one_environment_variable() {
        assert!(!sync_launches_requested(None));
        assert!(!sync_launches_requested(Some("")));
        assert!(!sync_launches_requested(Some("0")));
        assert!(sync_launches_requested(Some("1")));
        assert!(sync_launches_requested(Some("yes")));
    }

    #[test]
    fn the_async_switch_is_off_until_a_run_selects_it() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        set_async_launches(false);
        assert!(!async_launches());
        assert!(set_async_launches(true));
        assert!(async_launches());
        set_async_launches(false);
        assert!(!async_launches());
    }

    // The launch path and the stage-copy path must never disagree: a queued copy is only safe
    // because the queued launches it feeds are queued too, and a bracket consumer must put both
    // back on the drained path or its elapsed time would be measured after a drain that no longer
    // follows the call. One predicate decides both, and this is that predicate's truth table.
    #[test]
    fn a_stage_copy_is_queued_exactly_when_a_launch_is() {
        let _serial = TEST_LOCK.lock().expect("test lock");
        set_async_launches(false);
        assert!(!queued_dispatch_wanted(), "drained run queues nothing");
        assert!(set_async_launches(true));
        assert!(
            queued_dispatch_wanted(),
            "queued run queues launches and copies"
        );
        probe_meter_arm(true);
        assert!(
            !queued_dispatch_wanted(),
            "the probe meter's per-launch bracket keeps both calls drained"
        );
        probe_meter_arm(false);
        assert!(queued_dispatch_wanted());
        set_async_launches(false);
        assert!(!queued_dispatch_wanted());
    }

    #[test]
    fn the_module_cache_keys_by_bytes_not_by_address() {
        // Two live allocations with the same bytes must share one id; a third with different
        // bytes must not. This is the "different cubin bytes -> different key" misuse case.
        let first = vec![7u8, 8, 9];
        let same_bytes = vec![7u8, 8, 9];
        let different = vec![7u8, 8, 10];
        assert_ne!(first.as_ptr(), same_bytes.as_ptr());
        let mut cache = ModuleCache::default();
        let first_id = cache.intern(&first);
        assert_eq!(cache.intern(&same_bytes), first_id);
        let different_id = cache.intern(&different);
        assert_ne!(first_id, different_id);
        assert_ne!(
            (first_id, "symbol".to_string()),
            (different_id, "symbol".to_string())
        );
        // The pointer fast path re-checks the cached bytes and returns the same id for the same
        // image; it is a comparison, not an assumption that the address is enough.
        assert_eq!(cache.intern(&first), first_id);
    }

    #[test]
    fn a_pointer_key_reused_by_different_bytes_is_a_different_image() {
        // The dangerous direction: image A's cubin buffer is freed and image B of the same length
        // is allocated at the same address. A pointer-only fast path would answer A's id and
        // load_function would silently return A's module. The byte comparison must intern B as a
        // new id, so the two images never resolve each other's functions.
        let first = [7u8, 8, 9, 10];
        let second = [7u8, 8, 9, 11];
        let key_ptr = 0xdead_0000usize;
        let key_len = first.len();
        let mut cache = ModuleCache::default();

        let first_id = cache.intern_with_key(key_ptr, key_len, &first);
        let first_module = 0x1111usize as CuModule;
        let first_function = 0x2222usize as CuFunction;
        cache.insert_function(first_id, "kernel", first_module, first_function);

        // Same pointer key and length, different bytes: the reused-allocation case.
        let second_id = cache.intern_with_key(key_ptr, key_len, &second);
        assert_ne!(
            second_id, first_id,
            "a reused address with different bytes must not reuse the old image id"
        );
        assert!(
            cache.function(second_id, "kernel").is_none(),
            "the second image must not resolve the first image's loaded module"
        );

        let second_module = 0x3333usize as CuModule;
        let second_function = 0x4444usize as CuFunction;
        cache.insert_function(second_id, "kernel", second_module, second_function);
        assert_eq!(
            cache.function(first_id, "kernel"),
            Some((first_module, first_function))
        );
        assert_eq!(
            cache.function(second_id, "kernel"),
            Some((second_module, second_function))
        );
        assert_ne!(
            cache.function(first_id, "kernel"),
            cache.function(second_id, "kernel"),
            "each image must keep its own module and function"
        );

        // Interning the first image through the same key restores its id by byte match, not by
        // an assumed pointer match.
        assert_eq!(cache.intern_with_key(key_ptr, key_len, &first), first_id);
        assert_eq!(cache.intern(&first), first_id);
    }

    #[test]
    fn a_cached_module_is_owned_so_free_leaves_it_loaded() {
        let mut cache = ModuleCache::default();
        let image = cache.intern(&[1u8, 2, 3]);
        let module = 0x1234usize as CuModule;
        cache.insert_function(image, "kernel", module, std::ptr::null_mut());
        assert!(cache.owns_module(module));
        assert!(!cache.owns_module(0x9999usize as CuModule));
        assert_eq!(
            cache.function(image, "kernel").map(|(m, _)| m),
            Some(module)
        );
        assert!(cache.function(image, "other_kernel").is_none());
    }

    #[test]
    fn the_pool_reuses_only_after_a_completion_point_and_never_overlaps() {
        let ptr = Deviceptr(0x1000);
        let mut pool = BufferPool::default();
        pool.record_allocation(ptr, 64);
        // Nothing released yet.
        assert_eq!(pool.checkout(64, 100), None);
        pool.release(ptr, 5)
            .expect("first release after a checkout");
        // Released before launch 5; completion at 4 does not prove launch 5 finished.
        assert_eq!(pool.checkout(64, 4), None);
        // A completion exactly at the release stamp is enough.
        assert_eq!(pool.checkout(64, 5), Some(ptr));
        // Checked out means the same buffer cannot be handed out twice.
        assert_eq!(pool.checkout(64, 5), None);
        // Size classes are exact, so a 32-byte request never gets a 64-byte buffer.
        assert_eq!(pool.checkout(32, u64::MAX), None);
        // After a new release a later completion may reuse it again.
        pool.release(ptr, 6)
            .expect("release after the new checkout");
        assert_eq!(pool.checkout(64, 6), Some(ptr));
    }

    #[test]
    fn a_double_release_is_refused_and_never_duplicates_a_free_list() {
        let ptr = Deviceptr(0x2000);
        let mut pool = BufferPool::default();
        pool.record_allocation(ptr, 128);
        pool.release(ptr, 7).expect("the first release is accepted");
        assert!(
            pool.release(ptr, 7).is_err(),
            "a second release of a checked-in pointer must be refused"
        );
        // One entry, not two: the first checkout gets the buffer and the next finds nothing, so
        // the same allocation is never handed to two live kernels.
        assert_eq!(pool.checkout(128, 7), Some(ptr));
        assert_eq!(pool.checkout(128, u64::MAX), None);
        // A pointer the pool does not own is refused too, instead of joining a free list.
        assert!(pool.release(Deviceptr(0xbad), 7).is_err());
    }

    #[test]
    fn a_pooled_pointer_freed_through_free_is_forgotten_not_freed_again() {
        // Cuda::free and Cuda::free_buffers call BufferPool::forget for a pointer the pool owns.
        // The device calls cannot run without a GPU, so the contract pinned here is the pool
        // bookkeeping: a forgotten pointer is removed from every map, so Drop's authoritative
        // free lists are empty and cannot pass it to cuMemFree a second time.
        let checked_in = Deviceptr(0x3000);
        let checked_out = Deviceptr(0x3100);
        let mut pool = BufferPool::default();
        pool.record_allocation(checked_in, 64);
        pool.record_allocation(checked_out, 96);
        pool.release(checked_in, 4).expect("release");
        assert!(pool.forget(checked_in));
        assert!(
            !pool.sizes.contains_key(&checked_in.0),
            "a forgotten pointer is no longer owned by the pool"
        );
        assert!(
            pool.free.values().all(Vec::is_empty),
            "a forgotten pointer is off every free list, so Drop frees nothing for it"
        );
        // A live checkout passed to free() is forgotten as well: it must not remain in the
        // pool's ownership, or Drop would free memory the caller already handed back.
        assert!(pool.forget(checked_out));
        assert!(pool.sizes.is_empty());
        assert!(pool.free.is_empty());
        assert!(pool.checked_in.is_empty());
        // Forgetting twice is harmless and reports that the pool does not own it.
        assert!(!pool.forget(checked_in));
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn a_graph_segment_refuses_every_non_launch_operation() {
        // Capture records kernels only. Every other operation this driver can perform has to be
        // refused with a named error while a segment is active, because running it outside the
        // recorded graph (or letting the driver fail the capture) changes the arithmetic path.
        for op in [
            SegmentOp::Copy,
            SegmentOp::Synchronize,
            SegmentOp::Allocation,
            SegmentOp::Free,
            SegmentOp::Release,
            SegmentOp::ModuleLoad,
            SegmentOp::RepeatedLaunch,
        ] {
            assert!(
                graph_segment_refusal(GRAPH_DIRECT, op).is_none(),
                "{op:?} must run outside a segment"
            );
            let capture = graph_segment_refusal(GRAPH_CAPTURE, op)
                .unwrap_or_else(|| panic!("{op:?} must be refused during capture"));
            let record = graph_segment_refusal(GRAPH_RECORD, op)
                .unwrap_or_else(|| panic!("{op:?} must be refused during a record"));
            assert_eq!(capture, record, "one refusal text for both segment modes");
            assert!(!capture.is_empty());
        }
    }

    #[test]
    fn a_launch_record_identity_covers_grid_block_and_every_argument_word() {
        // The graph wrapper replays only when the whole launch list compares equal. A changed
        // argument word is a changed record: a moved pooled buffer or a different weight slice
        // must never be replayed against a graph captured for the old address.
        let base = CudaLaunchRecord {
            function: 0x1000,
            grid: (1, 1, 1),
            block: (2, 1, 1),
            args: vec![7, 8, 9],
        };
        let same = base.clone();
        assert_eq!(base, same);
        assert_eq!(base.fingerprint_words(), same.fingerprint_words());
        assert_eq!(base.describe(), "grid 1x1x1, block 2x1x1, 3 args");

        let mut pointer_moved = base.clone();
        pointer_moved.args[1] = 0xdead_beef;
        assert_ne!(
            base, pointer_moved,
            "a changed device pointer must be a different launch"
        );
        assert_ne!(base.fingerprint_words(), pointer_moved.fingerprint_words());

        let mut grid_changed = base.clone();
        grid_changed.grid.0 = 2;
        assert_ne!(base, grid_changed);
        assert_ne!(base.fingerprint_words(), grid_changed.fingerprint_words());

        let mut function_changed = base.clone();
        function_changed.function = 0x2000;
        assert_ne!(base, function_changed);
        assert_ne!(
            base.fingerprint_words(),
            function_changed.fingerprint_words()
        );
    }
}
