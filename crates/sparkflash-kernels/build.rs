//! Build script for sparkflash-kernels.
//!
//! Two jobs:
//!
//! 1. Decide whether a CUDA toolchain is available. Nothing in the workspace may require a GPU
//!    to build or test, so every CUDA path is behind the node's `cuda` feature and the CPU
//!    reference kernels are always compiled.
//! 2. When a toolchain IS present, compile the shared trace ABI header. That header is included
//!    by every instrumented kernel and its Rust counterpart lives in sparkflash-trace, so an
//!    edit that breaks either side has to fail the build instead of lurking until the first real
//!    kernel compile. The check compiles a device-only probe that static-asserts every struct
//!    size, alignment and field offset and instantiates all four trace macros; it needs no GPU,
//!    only the CUDA frontend.
//!
//! `-arch` defaults to the DGX Spark target (sm_121a) and can be overridden with
//! SPARKFLASH_CUDA_ARCH. An architecture the installed toolkit does not know is downgraded to a
//! build-script warning, because CI may carry an older CUDA; any other failure is fatal.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The probe that pins the Rust/CUDA trace ABI.
///
/// The assertions live inside a kernel so they are evaluated during device-only compilation,
/// and the macro instantiations force the ENTER/READ_TIMER/EXIT/EXIT_ROCE bodies to be
/// type-checked. HEADER_PATH is substituted with the canonical header path.
const PROBE: &str = r#"#include <stddef.h>
#include <stdint.h>

#define SPARKFLASH_ENABLE_GPU_TRACE 1
#include "HEADER_PATH"

__global__ void sparkflash_trace_abi_probe(GpuTraceRing *ring, uint64_t *sink) {
    static_assert(sizeof(GpuTraceEvent) == 32, "GpuTraceEvent must be 32 bytes");
    static_assert(alignof(GpuTraceEvent) == 32, "GpuTraceEvent must be 32-byte aligned");
    static_assert(offsetof(GpuTraceEvent, start_ns) == 0, "GpuTraceEvent.start_ns offset");
    static_assert(offsetof(GpuTraceEvent, end_ns) == 8, "GpuTraceEvent.end_ns offset");
    static_assert(offsetof(GpuTraceEvent, sub_wait_ns) == 16, "GpuTraceEvent.sub_wait_ns offset");
    static_assert(
        offsetof(GpuTraceEvent, scope_and_rank) == 20, "GpuTraceEvent.scope_and_rank offset");
    static_assert(offsetof(GpuTraceEvent, layer_id) == 24, "GpuTraceEvent.layer_id offset");
    static_assert(offsetof(GpuTraceEvent, payload_meta) == 28, "GpuTraceEvent.payload_meta offset");
    static_assert(sizeof(GpuTraceRing) == 32, "GpuTraceRing must be 32 bytes");
    static_assert(sizeof(GpuTraceRing) % 32 == 0, "events must stay aligned after the header");
    static_assert(alignof(GpuTraceRing) == 32, "GpuTraceRing must be 32-byte aligned");
    static_assert(offsetof(GpuTraceRing, active_scope_mask) == 0, "GpuTraceRing.active_scope_mask");
    static_assert(offsetof(GpuTraceRing, write_idx) == 8, "GpuTraceRing.write_idx offset");
    static_assert(offsetof(GpuTraceRing, capacity_mask) == 12, "GpuTraceRing.capacity_mask offset");
    static_assert(offsetof(GpuTraceRing, reserved) == 16, "GpuTraceRing.reserved offset");

    uint64_t t_enter = 0ULL;
    uint64_t t_staged = 0ULL;
    uint64_t t_ready = 0ULL;
    SPARK_GPU_TRACE_ENTER(ring, TRACE_SCOPE_ENGRAM_PREFETCH, t_enter);
    SPARK_GPU_READ_TIMER(ring, TRACE_SCOPE_ENGRAM_IO_URING, t_staged);
    SPARK_GPU_TRACE_EXIT(ring, TRACE_SCOPE_ENGRAM_PREFETCH, 1U, 2U, t_enter);
    SPARK_GPU_TRACE_EXIT_ROCE(
        ring, TRACE_SCOPE_ROCENANTE_RDMA, 3U, 4U, t_enter, t_staged, t_ready);
    GpuTraceEvent *ev = (GpuTraceEvent *)((char *)ring + sizeof(GpuTraceRing));
    if (sink != nullptr) {
        *sink = ev->start_ns + t_enter + t_staged + t_ready;
    }
}
"#;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../kernels");
    println!("cargo:rustc-check-cfg=cfg(sparkflash_cuda_toolkit)");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_NVCC");

    let Some(nvcc) = which_nvcc() else {
        println!("cargo:warning=sparkflash-kernels: nvcc not found; CPU reference kernels only");
        return;
    };
    println!("cargo:rustc-cfg=sparkflash_cuda_toolkit");

    let arch = std::env::var("SPARKFLASH_CUDA_ARCH").unwrap_or_else(|_| "sm_121a".to_owned());
    let header = Path::new("../../kernels/trace_globaltimer_sm121.cuh");
    if !header.is_file() {
        println!(
            "cargo:warning=sparkflash-kernels: {} is missing; skipping the trace ABI check",
            header.display()
        );
        return;
    }

    match check_trace_header(&nvcc, &arch, header) {
        Ok(()) => {}
        Err(failure) if failure.unusable_toolchain => {
            println!(
                "cargo:warning=sparkflash-kernels: {} cannot compile the trace ABI probe for \
                 {arch} ({}); set SPARKFLASH_CUDA_ARCH to a supported architecture",
                nvcc.display(),
                failure.stderr.trim(),
            );
        }
        // A header that does not compile must never reach a GPU host: the ABI is shared with
        // sparkflash-trace, so a drift here would corrupt every trace record.
        Err(failure) => panic!(
            "sparkflash-kernels: the shared CUDA trace ABI header failed to compile:\n{}",
            failure.stderr
        ),
    }
}

struct ProbeFailure {
    stderr: String,
    /// True when nvcc itself refused the request (unknown option or unsupported architecture)
    /// rather than reporting an error inside the header.
    unusable_toolchain: bool,
}

/// Compiles a device-only probe of the shared trace header with nvcc.
fn check_trace_header(nvcc: &Path, arch: &str, header: &Path) -> Result<(), ProbeFailure> {
    let fail = |stderr: String| ProbeFailure {
        unusable_toolchain: stderr.contains("Unsupported gpu architecture")
            || stderr.contains("unsupported architecture")
            || stderr.contains("Unknown option")
            || stderr.contains("Unrecognized"),
        stderr,
    };

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|e| fail(e.to_string()))?);
    let canonical = header
        .canonicalize()
        .map_err(|e| fail(format!("cannot canonicalize {}: {e}", header.display())))?;
    let probe = out_dir.join("trace_abi_probe.cu");
    let source = PROBE.replace("HEADER_PATH", &canonical.display().to_string());
    std::fs::write(&probe, source).map_err(|e| fail(e.to_string()))?;

    let mut command = Command::new(nvcc);
    command.args([
        "-std=c++17",
        &format!("-arch={arch}"),
        // Device code only, to PTX: no host codegen, no link, but the frontend still evaluates
        // every static_assert and type-checks the macros. (--device-c would work too; --ptx
        // additionally avoids needing a host compiler.)
        "--ptx",
    ]);
    command
        .arg(&probe)
        .arg("-o")
        .arg(out_dir.join("trace_abi_probe.ptx"));
    if let Some(parent) = canonical.parent() {
        command.arg("-I").arg(parent);
    }
    let output = command
        .output()
        .map_err(|e| fail(format!("failed to run {}: {e}", nvcc.display())))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(fail(String::from_utf8_lossy(&output.stderr).into_owned()))
    }
}

fn which_nvcc() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("SPARKFLASH_NVCC") {
        let path = PathBuf::from(explicit);
        return path.is_file().then_some(path);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join("nvcc"))
        .find(|candidate| candidate.is_file())
}
