//! Build script for sparkflash-kernels.
//!
//! Two jobs:
//!
//! 1. Decide whether a CUDA toolchain is available. Nothing in the workspace may require a GPU
//!    to build or test: this script probes for a frontend and emits the sparkflash_cuda_toolkit
//!    cfg, which is what the ABI checks are gated on, and the CPU reference kernels are always
//!    compiled. The crate's optional "cuda" feature is a separate thing - it only selects the
//!    cudarc dependency, and no code in this crate reads it (see Cargo.toml).
//! 2. When a CUDA frontend IS present, compile the shared headers, the RoCEnante kernel and the
//!    seven device kernels of the M4/M7 set. The trace ABI header, the RoCEnante slot ABI and the
//!    pinned kernel set are contracts between Rust and device code, so a drift has to fail the
//!    build instead of lurking until the first real kernel launch:
//!
//!    * the trace probe compiles a device-only translation unit that static-asserts every
//!      GpuTraceEvent/GpuTraceRing size, alignment and field offset and instantiates all four
//!      trace macros;
//!    * the RoCEnante probe compiles kernels/rocenante_oneshot_sm121.cu together with a
//!      redeclaration of the kernel's __global__ signature and a second, literal copy of the
//!      slot layout assertions;
//!    * the kernel itself is compiled standalone, so it must be a self-contained translation
//!      unit;
//!    * every kernel of the M4/M7 set is compiled on its own, so a failure names the file, and
//!      then all seven are compiled together with the trace macros enabled, which is the only
//!      check that can catch a collision BETWEEN kernel files.
//!
//!    None of that needs a GPU, only a CUDA frontend. When nvcc is absent but clang is present,
//!    the kernel, the RoCEnante probe and the M4/M7 kernel set are instead handed to clang's CUDA
//!    frontend with a stubbed cuda_bf16.h
//!    (crates/sparkflash-kernels/cuda_stub/cuda_stub.h) and compiled device-only with
//!    -fsyntax-only. Compiling the probe rather than the kernel alone matters:
//!    it turns SPARKFLASH_ENABLE_GPU_TRACE on, so the trace macros in the kernel body are
//!    type-checked instead of expanding to no-ops, it re-declares the __global__ signature
//!    against the definition, and it re-states the slot layout as literals. That is still a
//!    syntax and type check, not a substitute for nvcc: the stub's intrinsics carry no
//!    semantics and clang does not validate inline PTX.
//!
//! "-arch" defaults to the DGX Spark target (sm_121a) and can be overridden with
//! SPARKFLASH_CUDA_ARCH. An architecture or option the installed frontend does not know is
//! downgraded to a build-script warning, because CI may carry an older toolkit; any other
//! failure is fatal.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The probe that pins the Rust/CUDA trace ABI.
///
/// The assertions live inside a kernel so they are evaluated during device-only compilation,
/// and the macro instantiations force the ENTER/READ_TIMER/EXIT/EXIT_ROCE bodies to be
/// type-checked. HEADER_PATH is substituted with the canonical header path.
const TRACE_PROBE: &str = r#"#include <stddef.h>
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

/// The probe that pins the RoCEnante one-shot slot ABI.
///
/// It redeclares the kernel's signature and then includes the kernel source, so the
/// declaration, the definition and the slot layout are all checked in one translation unit.
/// The layout numbers are deliberately written out as literals instead of reusing the header's
/// macros: if somebody weakens the header's static_asserts, this copy still fails.
const ROCE_PROBE: &str = r#"#include <stddef.h>
#include <stdint.h>

#define SPARKFLASH_ENABLE_GPU_TRACE 1
#include "rocenante_oneshot_sm121.cuh"

extern "C" __global__ void rocenante_oneshot_allreduce_bf16_tp4(
    const __nv_bfloat16* __restrict__ local_in,
    __nv_bfloat16* __restrict__ local_out,
    RoceOneShotSlot* __restrict__ slot,
    volatile uint64_t* __restrict__ host_doorbell_seq,
    GpuTraceRing* __restrict__ trace_ring,
    const uint32_t layer_id,
    const uint32_t my_rank,
    const uint32_t num_elements,
    const uint64_t step_seq);

#include "rocenante_oneshot_sm121.cu"

__global__ void sparkflash_roce_abi_probe(RoceOneShotSlot *slot, uint64_t *sink) {
    static_assert(sizeof(__nv_bfloat16) == 2, "bf16 must be 2 bytes");
    static_assert(alignof(__nv_bfloat16) == 2, "bf16 must be 2-byte aligned");
    static_assert(sizeof(RoceOneShotSlot) == 8388640, "slot must be 8 MiB + 32 bytes");
    static_assert(alignof(RoceOneShotSlot) == 8, "slot must keep its natural 8-byte alignment");
    static_assert(offsetof(RoceOneShotSlot, arrival_seq) == 0, "arrival_seq must start the slot");
    static_assert(offsetof(RoceOneShotSlot, payload) == 32, "payload must start at offset 32");
    static_assert((1024 * 1024) * 2 == 2097152, "one payload row is 2 MiB");
    static_assert(4 * (1024 * 1024) * 2 + 32 == 8388640, "8 MiB + 32 bytes");
    static_assert(SPARK_ROCE_SLOT_RANKS == 4, "the one-shot group is TP=4");
    static_assert(SPARK_ROCE_SLOT_ROW_ELEMS == 1024 * 1024, "one row is 1 Mi elements");
    static_assert(SPARK_ROCENANTE_PEER_WAIT_NS > 0, "the peer wait must be bounded but non-zero");
    static_assert(SPARK_ROCENANTE_SPIN_ITERS > 0, "the spin backoff must be non-zero");
    static_assert(
        SPARK_ROCE_PAYLOAD_DIAG_FLAG > SPARK_ROCE_SLOT_ROW_BYTES,
        "the diagnostic flag must live above every possible byte count");
    static_assert(SPARK_ROCE_DIAG_MISSING_SHIFT + 4 <= 31, "the missing mask must fit the flag");
    static_assert(
        sizeof(slot->payload[0]) == SPARK_ROCE_SLOT_ROW_BYTES, "one slot row is 2 MiB");

    if (sink != nullptr) {
        // Force the diagnostic NaN constructor through the real cuda_bf16.h conversion path.
        // Do NOT read __nv_bfloat16::__x here: CUDA 12.9 declares that member protected, so a
        // probe that touches it only compiles against a stub that makes it public. nvcc 12.9.86
        // rejects it, which is how this line was found.
        __nv_bfloat16 nan = spark_roce_diag_nan_bf16();
        float nan_value = __bfloat162float(nan);
        *sink = (uint64_t)SPARK_ROCE_SLOT_BYTES + slot->arrival_seq[0]
                + (uint64_t)(nan_value != nan_value ? 1u : 0u);
    }
}
"#;

/// The seven device kernel sources the crate probes and pins, with the stem used for their PTX
/// output.
///
/// Six of them are the Milestone 4 kernels of specification 6.2, the set m4_abi's frozen KERNELS
/// array pins. The seventh is the M7 SwiGLU + MX-quantise epilogue
/// (kernels/swiglu_mxfp8_quant_sm121.cu): it is deliberately OUTSIDE that frozen array, because
/// adding an entry there would be an ABI revision, but it is a device source this crate ships,
/// the clang/nvcc probes compile it like the others, and baseline/kernel_baseline.txt records its
/// cubin as the seventh entry. Every one of the seven is probed on every build: a kernel that is
/// only ever compiled by hand on a GPU host is a kernel nobody has type-checked, and the CI
/// cuda-abi job runs exactly this probe.
const PROBED_KERNELS: [(&str, &str); 7] = [
    ("mega_mhc_sm121.cu", "mega_mhc_sm121"),
    ("b12x_mxfp8_dense_sm121.cu", "b12x_mxfp8_dense_sm121"),
    ("b12x_mxfp4_moe_sm121.cu", "b12x_mxfp4_moe_sm121"),
    ("flash_mla_csa2_sm121.cu", "flash_mla_csa2_sm121"),
    ("b12x_mxfp8_wo_a_sm121.cu", "b12x_mxfp8_wo_a_sm121"),
    ("cert_lm_head_sm121.cu", "cert_lm_head_sm121"),
    ("swiglu_mxfp8_quant_sm121.cu", "swiglu_mxfp8_quant_sm121"),
];

/// The shared header every one of those kernels includes for its model constants, the MX codecs
/// and the B12x layout. Probing the kernel files is not enough if this one is missing: it is the
/// one include they all share.
const M4_COMMON_HEADER: &str = "m4_common_sm121.cuh";

/// The combined probe: all seven kernel sources in one translation unit with the trace macros
/// ENABLED. This is the only check in the build that can catch a collision BETWEEN kernels (two
/// files defining the same macro to different values, or two identical non-inline symbols), and it
/// is also the only one that compiles the enabled branch of every trace scope.
fn combined_probe_source() -> String {
    let mut source = String::from("#define SPARKFLASH_ENABLE_GPU_TRACE 1\n");
    source.push_str("#include \"");
    source.push_str(M4_COMMON_HEADER);
    source.push_str("\"\n");
    for (file, _) in PROBED_KERNELS {
        source.push_str("#include \"");
        source.push_str(file);
        source.push_str("\"\n");
    }
    source
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../kernels");
    println!("cargo:rerun-if-changed=cuda_stub");
    println!("cargo:rustc-check-cfg=cfg(sparkflash_cuda_toolkit)");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_NVCC");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_CLANG");

    let arch = std::env::var("SPARKFLASH_CUDA_ARCH").unwrap_or_else(|_| "sm_121a".to_owned());

    let Some(nvcc) = which_nvcc() else {
        // No nvcc: the clang CUDA frontend can still parse the kernel against the stub
        // headers. That check is weaker than nvcc (no PTX validation, no ptxas), so it is
        // reported as a different kind of evidence, not as a substitute.
        match which_clang() {
            Some(clang) => report_clang(&clang, &arch),
            None => println!(
                "cargo:warning=sparkflash-kernels: neither nvcc nor clang++ was found; the \
                 CUDA ABI and kernel sources are UNVERIFIED by this build (CPU reference \
                 kernels only). The unit tests in roce_abi.rs still check the Rust/CUDA \
                 conformance textually."
            ),
        }
        return;
    };
    println!("cargo:rustc-cfg=sparkflash_cuda_toolkit");

    let kernels = Path::new("../../kernels");
    if !kernels.is_dir() {
        println!(
            "cargo:warning=sparkflash-kernels: {} is missing; skipping every CUDA ABI check",
            kernels.display()
        );
        return;
    }

    let trace_header = kernels.join("trace_globaltimer_sm121.cuh");
    if !trace_header.is_file() {
        println!(
            "cargo:warning=sparkflash-kernels: {} is missing; skipping the trace ABI check",
            trace_header.display()
        );
    } else {
        report(
            &nvcc,
            &arch,
            "the shared CUDA trace ABI header",
            check_trace_header(&nvcc, &arch, &trace_header),
        );
    }

    let roce_header = kernels.join("rocenante_oneshot_sm121.cuh");
    let roce_kernel = kernels.join("rocenante_oneshot_sm121.cu");
    if !roce_header.is_file() || !roce_kernel.is_file() {
        println!(
            "cargo:warning=sparkflash-kernels: the RoCEnante one-shot sources are missing; \
             skipping the slot ABI check"
        );
        return;
    }

    report(
        &nvcc,
        &arch,
        "the RoCEnante slot ABI probe (which includes the kernel)",
        check_roce_probe(&nvcc, &arch, kernels),
    );
    report(
        &nvcc,
        &arch,
        "the RoCEnante one-shot kernel",
        compile_file(
            &nvcc,
            &arch,
            &roce_kernel,
            &[Path::new("../../kernels")],
            "rocenante_oneshot_sm121",
        ),
    );

    // The M4/M7 kernel set. Every kernel is probed individually so that a failure names the file,
    // and then all seven are probed together with the trace macros enabled.
    if !kernels.join(M4_COMMON_HEADER).is_file() {
        println!(
            "cargo:warning=sparkflash-kernels: {} is missing; skipping every M4/M7 kernel probe",
            M4_COMMON_HEADER
        );
        return;
    }
    for (file, stem) in PROBED_KERNELS {
        let path = kernels.join(file);
        if !path.is_file() {
            println!(
                "cargo:warning=sparkflash-kernels: {file} is missing; the M4/M7 kernel set is \
                 incomplete (the six Milestone 4 kernels plus the M7 SwiGLU quantiser \
                 epilogue); the report would cover fewer kernels than the baseline names"
            );
            continue;
        }
        report(
            &nvcc,
            &arch,
            &format!("the probed kernel {file}"),
            compile_file(&nvcc, &arch, &path, &[kernels], stem),
        );
    }
    report(
        &nvcc,
        &arch,
        "the combined M4/M7 probe (all seven kernels, trace macros enabled)",
        check_m4_probe(&nvcc, &arch, kernels),
    );
}

/// Writes and compiles the combined M4/M7 probe with nvcc.
fn check_m4_probe(nvcc: &Path, arch: &str, kernels: &Path) -> Result<(), ProbeFailure> {
    let probe = out_dir()?.join("m4_kernels_probe.cu");
    std::fs::write(&probe, combined_probe_source())
        .map_err(|e| ProbeFailure::new(e.to_string()))?;
    compile_file(nvcc, arch, &probe, &[kernels], "m4_kernels_probe")
}

/// Prints a warning for an unusable toolchain and aborts the build for a real compile error.
fn report(nvcc: &Path, arch: &str, what: &str, result: Result<(), ProbeFailure>) {
    match result {
        Ok(()) => {}
        Err(failure) if failure.unusable_toolchain => println!(
            "cargo:warning=sparkflash-kernels: {} cannot compile {what} for {arch} ({}); set \
             SPARKFLASH_CUDA_ARCH to a supported architecture",
            nvcc.display(),
            failure.stderr.trim(),
        ),
        // A header or kernel that does not compile must never reach a GPU host: the ABI is
        // shared with sparkflash-trace and sparkflash-comm, so a drift here would corrupt
        // every trace record and every collective.
        Err(failure) => panic!(
            "sparkflash-kernels: {what} failed to compile:\n{}",
            failure.stderr
        ),
    }
}

struct ProbeFailure {
    stderr: String,
    /// True when the frontend itself refused the request (unknown option or unsupported
    /// architecture) rather than reporting an error inside the source.
    unusable_toolchain: bool,
}

impl ProbeFailure {
    fn new(stderr: String) -> Self {
        let unusable_toolchain = stderr.contains("Unsupported gpu architecture")
            || stderr.contains("unsupported architecture")
            || stderr.contains("unsupported CUDA gpu architecture")
            || stderr.contains("Unknown option")
            || stderr.contains("Unrecognized")
            || stderr.contains("unknown argument")
            || stderr.contains("not supported")
            || stderr.contains("cannot find CUDA installation");
        Self {
            stderr,
            unusable_toolchain,
        }
    }
}

/// Compiles a device-only probe of the shared trace header with nvcc.
fn check_trace_header(nvcc: &Path, arch: &str, header: &Path) -> Result<(), ProbeFailure> {
    let out_dir = out_dir()?;
    let canonical = header
        .canonicalize()
        .map_err(|e| ProbeFailure::new(format!("cannot canonicalize {}: {e}", header.display())))?;
    let probe = out_dir.join("trace_abi_probe.cu");
    let source = TRACE_PROBE.replace("HEADER_PATH", &canonical.display().to_string());
    std::fs::write(&probe, source).map_err(|e| ProbeFailure::new(e.to_string()))?;
    let include = canonical.parent().map(Path::to_path_buf);
    let includes: Vec<&Path> = include.iter().map(PathBuf::as_path).collect();
    compile_file(nvcc, arch, &probe, &includes, "trace_abi_probe")
}

/// Compiles the RoCEnante slot ABI probe (and, through it, the kernel) with nvcc.
fn check_roce_probe(nvcc: &Path, arch: &str, kernels: &Path) -> Result<(), ProbeFailure> {
    let out_dir = out_dir()?;
    let probe = out_dir.join("roce_abi_probe.cu");
    std::fs::write(&probe, ROCE_PROBE).map_err(|e| ProbeFailure::new(e.to_string()))?;
    compile_file(nvcc, arch, &probe, &[kernels], "roce_abi_probe")
}

/// Compiles one translation unit to PTX with nvcc. No host codegen and no link, but the
/// frontend still evaluates every static_assert and type-checks every macro.
fn compile_file(
    nvcc: &Path,
    arch: &str,
    source: &Path,
    includes: &[&Path],
    stem: &str,
) -> Result<(), ProbeFailure> {
    let out_dir = out_dir()?;
    let mut command = Command::new(nvcc);
    command.args(["-std=c++17", &format!("-arch={arch}"), "--ptx"]);
    for include in includes {
        command.arg("-I").arg(include);
    }
    command
        .arg(source)
        .arg("-o")
        .arg(out_dir.join(format!("{stem}.ptx")));
    let output = command
        .output()
        .map_err(|e| ProbeFailure::new(format!("failed to run {}: {e}", nvcc.display())))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(ProbeFailure::new(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ))
    }
}

fn out_dir() -> Result<PathBuf, ProbeFailure> {
    std::env::var("OUT_DIR")
        .map(PathBuf::from)
        .map_err(|e| ProbeFailure::new(e.to_string()))
}

/// Parses the kernel with clang's CUDA frontend, which is all a host without nvcc can do.
fn report_clang(clang: &Path, arch: &str) {
    match check_kernel_syntax(clang, arch) {
        Ok(()) => report_clang_probe(clang, arch),
        // Retry once on a baseline architecture: the syntax and the slot layout do not depend
        // on the target, and an older clang may not know sm_121a yet.
        Err(failure) if failure.unusable_toolchain && arch != "sm_90" => {
            match check_kernel_syntax(clang, "sm_90") {
                Ok(()) => {
                    println!(
                        "cargo:warning=sparkflash-kernels: clang cannot target {arch}; the kernel \
                         parsed cleanly for sm_90 instead. Set SPARKFLASH_CUDA_ARCH (or \
                         SPARKFLASH_CLANG) to check the real target."
                    );
                    report_clang_probe(clang, "sm_90");
                }
                Err(second) => clang_outcome(second, arch),
            }
        }
        Err(failure) => clang_outcome(failure, arch),
    }
}

/// Runs the RoCEnante ABI probe through clang after the kernel parsed, and reports what that
/// buys. The probe is the same translation unit nvcc would compile, so on this host it is the
/// only place where the trace macros, the signature redeclaration and the literal slot
/// assertions are evaluated together.
fn report_clang_probe(clang: &Path, arch: &str) {
    match check_roce_probe_syntax(clang, arch) {
        Ok(()) => match check_probed_kernels_syntax(clang, arch) {
            Ok((checked, missing)) => println!(
                "cargo:warning=sparkflash-kernels: nvcc not found; clang's CUDA frontend parsed \
                 kernels/rocenante_oneshot_sm121.cu, the RoCEnante ABI probe (trace macros \
                 enabled, signature redeclared, slot static_asserts), {checked} of the seven \
                 probed kernels (the six Milestone 4 ABI kernels plus the M7 SwiGLU quantiser \
                 epilogue; {missing} missing) and the combined M4/M7 probe with the trace macros \
                 enabled, against stub headers (-fsyntax-only, {arch}). That \
                 checks syntax, types and every static_assert, NOT PTX and NOT numerics. A CUDA \
                 host must run the nvcc probe before any of these kernels is trusted."
            ),
            Err(failure) => panic!(
                "sparkflash-kernels: the clang CUDA syntax probe failed; a Milestone 4 kernel \
                 does not parse or the combined probe does not compile:\n{}",
                failure.stderr
            ),
        },
        Err(failure) if failure.unusable_toolchain => println!(
            "cargo:warning=sparkflash-kernels: clang parsed the kernel but cannot run the \
             RoCEnante ABI probe for {arch} ({}); the probe's literal slot assertions were not \
             evaluated here",
            failure.stderr.trim()
        ),
        Err(failure) => panic!(
            "sparkflash-kernels: the clang CUDA ABI probe failed; the RoCEnante slot ABI, the \
             kernel signature and the kernel body do not agree:\n{}",
            failure.stderr
        ),
    }
}

/// Parses the seven probed kernels and the combined probe with clang. Returns how many kernels were
/// found and how many were missing, so the warning can say which evidence was gathered.
fn check_probed_kernels_syntax(clang: &Path, arch: &str) -> Result<(usize, usize), ProbeFailure> {
    let kernels = Path::new("../../kernels");
    let mut checked = 0usize;
    let mut missing = 0usize;
    for (file, _) in PROBED_KERNELS {
        let path = kernels.join(file);
        if path.is_file() {
            clang_syntax_check(clang, arch, &path)?;
            checked += 1;
        } else {
            missing += 1;
        }
    }
    if checked > 0 {
        let probe = out_dir()?.join("m4_kernels_probe.cu");
        std::fs::write(&probe, combined_probe_source())
            .map_err(|e| ProbeFailure::new(e.to_string()))?;
        clang_syntax_check(clang, arch, &probe)?;
    }
    Ok((checked, missing))
}

/// Turns a clang probe result into either a portability warning or a hard build failure.
fn clang_outcome(failure: ProbeFailure, arch: &str) {
    if failure.unusable_toolchain {
        println!(
            "cargo:warning=sparkflash-kernels: clang++ cannot run the CUDA syntax probe for \
             {arch} ({}); the kernel sources are UNVERIFIED by this build",
            failure.stderr.trim()
        );
    } else {
        panic!(
            "sparkflash-kernels: the clang CUDA syntax probe failed; \
             kernels/rocenante_oneshot_sm121.cu does not parse:\n{}",
            failure.stderr
        );
    }
}

/// Parses the kernel alone: the weakest clang check, and the one that must pass before the
/// probe is worth running.
fn check_kernel_syntax(clang: &Path, arch: &str) -> Result<(), ProbeFailure> {
    let kernel = Path::new("../../kernels/rocenante_oneshot_sm121.cu");
    if !kernel.is_file() {
        return Err(ProbeFailure::new(format!(
            "{} is missing",
            kernel.display()
        )));
    }
    clang_syntax_check(clang, arch, kernel)
}

/// Parses the RoCEnante ABI probe: the same source the nvcc path compiles, with the trace
/// macros enabled and the kernel's signature re-declared before its definition.
fn check_roce_probe_syntax(clang: &Path, arch: &str) -> Result<(), ProbeFailure> {
    let kernels = Path::new("../../kernels");
    if !kernels.is_dir() {
        return Err(ProbeFailure::new(format!(
            "{} is missing",
            kernels.display()
        )));
    }
    let probe = out_dir()?.join("roce_abi_probe.cu");
    std::fs::write(&probe, ROCE_PROBE).map_err(|e| ProbeFailure::new(e.to_string()))?;
    clang_syntax_check(clang, arch, &probe)
}

/// Runs clang's CUDA device-only syntax check on one translation unit against the stub headers.
fn clang_syntax_check(clang: &Path, arch: &str, source: &Path) -> Result<(), ProbeFailure> {
    let stub = Path::new("cuda_stub");
    if !stub.is_dir() {
        return Err(ProbeFailure::new(format!(
            "{} is missing; the clang probe needs the stub CUDA headers",
            stub.display()
        )));
    }
    let mut command = Command::new(clang);
    command.args([
        "-x",
        "cuda",
        "--cuda-device-only",
        &format!("--cuda-gpu-arch={arch}"),
        "-nocudainc",
        "-nocudalib",
        "-std=c++17",
        "-fsyntax-only",
        "-DSPARKFLASH_CUDA_SYNTAX_STUB=1",
        "-I",
        "../../kernels",
        "-I",
        "cuda_stub",
    ]);
    command.arg(source);
    let output = command
        .output()
        .map_err(|e| ProbeFailure::new(format!("failed to run {}: {e}", clang.display())))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(ProbeFailure::new(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ))
    }
}

fn which_nvcc() -> Option<PathBuf> {
    which("SPARKFLASH_NVCC", "nvcc")
}

fn which_clang() -> Option<PathBuf> {
    which("SPARKFLASH_CLANG", "clang++")
}

fn which(env: &str, program: &str) -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os(env) {
        let path = PathBuf::from(explicit);
        return path.is_file().then_some(path);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
}
