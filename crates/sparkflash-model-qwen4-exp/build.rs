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

//! Build the Qwen4-Exp device kernel images when a CUDA frontend is available.
//!
//! The kernel sources live in this crate's kernels/ directory, one cubin each: the token kernels
//! (qwen4_exp_sm121.cu, correctness-first, the decode path and the f32 reference), the
//! tensor-core prefill kernels (qwen4_prefill_sm121.cu) and the DFlash 2 drafter's own kernels
//! (dflash2_sm121.cu). A GPU host compiles them and embeds the
//! cubins; a CPU-only host still syntax-checks them with clang's CUDA frontend against the shared
//! stub headers, so a broken kernel fails the build here rather than at the first launch.
//!
//! sparkflash_qwen4_cubin is set only when nvcc produced every cubin. The device backend checks
//! that cfg at runtime and refuses the device path with a direct explanation when it is absent; it
//! never falls back to a host forward.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Each kernel source, the cubin it compiles to, and the env var its path is exported under.
const IMAGES: [(&str, &str, &str); 3] = [
    (
        "kernels/qwen4_exp_sm121.cu",
        "qwen4_exp_sm121.cubin",
        "SPARKFLASH_QWEN4_CUBIN",
    ),
    (
        "kernels/qwen4_prefill_sm121.cu",
        "qwen4_prefill_sm121.cubin",
        "SPARKFLASH_QWEN4_PREFILL_CUBIN",
    ),
    (
        "kernels/dflash2_sm121.cu",
        "dflash2_sm121.cubin",
        "SPARKFLASH_QWEN4_DFLASH2_CUBIN",
    ),
];

/// The workspace's shared kernel headers the sources include.
const SHARED_HEADERS: [&str; 6] = [
    "../../kernels/quant_decode_sm121.cuh",
    "../../kernels/opt_pdl_sm121.cuh",
    "../../kernels/prefetch_sm121.cuh",
    "../../kernels/opt_mma_common_sm121.cuh",
    "../../kernels/opt_mma_bf16_sm121.cuh",
    "../sparkflash-kernels/cuda_stub",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for (source, _, _) in IMAGES {
        println!("cargo:rerun-if-changed={source}");
    }
    for header in SHARED_HEADERS {
        println!("cargo:rerun-if-changed={header}");
    }
    println!("cargo:rustc-check-cfg=cfg(sparkflash_qwen4_cubin)");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_NVCC");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_CLANG");
    println!("cargo:rerun-if-env-changed=SPARKFLASH_NVCC_HOST");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    for (source, cubin, env) in IMAGES {
        println!("cargo:rustc-env={env}={}", out_dir.join(cubin).display());
        if !Path::new(source).is_file() {
            panic!(
                "sparkflash-model-qwen4-exp: {source} is missing; the device path cannot be built"
            );
        }
    }
    let arch = std::env::var("SPARKFLASH_CUDA_ARCH").unwrap_or_else(|_| "sm_121a".to_owned());

    if let Some(nvcc) = which("SPARKFLASH_NVCC", "nvcc") {
        for (source, cubin, _) in IMAGES {
            if !compile(&nvcc, &arch, Path::new(source), &out_dir.join(cubin)) {
                return;
            }
        }
        println!("cargo:rustc-cfg=sparkflash_qwen4_cubin");
        return;
    }

    if let Some(clang) = which("SPARKFLASH_CLANG", "clang++") {
        for (source, _, _) in IMAGES {
            if !syntax_check(&clang, &arch, Path::new(source)) {
                return;
            }
        }
        return;
    }

    println!(
        "cargo:warning=sparkflash-model-qwen4-exp: neither nvcc nor clang++ was found; the \
         Qwen4-Exp device kernel sources are UNVERIFIED and the device path is not built"
    );
}

/// Compile one source to a cubin. False (with a warning) when nvcc cannot target `arch`; a
/// compile error panics.
fn compile(nvcc: &Path, arch: &str, source: &Path, cubin: &Path) -> bool {
    let mut command = Command::new(nvcc);
    command.args([
        "-std=c++17",
        "-allow-unsupported-compiler",
        &format!("-arch={arch}"),
        "-cubin",
        "-I",
        "kernels",
        // The workspace's shared kernel headers: the coded-weight decoders and the tensor-core
        // GEMM body live in one place so a second pack cannot grow a second copy with its own
        // corner-case bugs.
        "-I",
        "../../kernels",
    ]);
    // nvcc 12.9 refuses the host gcc this distribution ships (gcc 16). Prefer a host compiler
    // it supports; SPARKFLASH_NVCC_HOST overrides the search. This mirrors what the shared
    // kernel crate does rather than silently skipping the compile.
    if let Some(host) = std::env::var_os("SPARKFLASH_NVCC_HOST")
        .map(PathBuf::from)
        .or_else(|| {
            ["g++-14", "g++-13", "g++-12"]
                .iter()
                .map(|name| PathBuf::from("/usr/bin").join(name))
                .find(|candidate| candidate.is_file())
        })
    {
        command.arg("-ccbin").arg(host);
    }
    command.arg(source).arg("-o").arg(cubin);
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", nvcc.display()));
    if output.status.success() {
        return true;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("Unsupported gpu architecture")
        || stderr.contains("unsupported architecture")
        || stderr.contains("unsupported CUDA gpu architecture")
    {
        println!(
            "cargo:warning=sparkflash-model-qwen4-exp: {} cannot target {arch}; set \
             SPARKFLASH_CUDA_ARCH to a supported architecture. The Qwen4-Exp device path is \
             not built.",
            nvcc.display()
        );
        return false;
    }
    panic!(
        "sparkflash-model-qwen4-exp: {} failed to compile with {}:\n{}",
        source.display(),
        nvcc.display(),
        stderr
    );
}

/// Parse one source with clang's CUDA frontend against the syntax stubs. False when the probe
/// could not run to completion, which leaves the source unverified.
fn syntax_check(clang: &Path, arch: &str, source: &Path) -> bool {
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
        "kernels",
        "-I",
        "../../kernels",
        "-I",
        "../sparkflash-kernels/cuda_stub",
    ]);
    command.arg(source);
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", clang.display()));
    if output.status.success() {
        println!(
            "cargo:warning=sparkflash-model-qwen4-exp: no nvcc, so the Qwen4-Exp device path \
             is not built; clang parsed {} for {arch} against the syntax stubs",
            source.display()
        );
        return true;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!(
        "cargo:warning=sparkflash-model-qwen4-exp: clang could not complete the Qwen4-Exp \
         CUDA syntax probe for {} ({}); the device kernel source is UNVERIFIED and the device \
         path is not built. Set SPARKFLASH_NVCC to compile it.",
        source.display(),
        stderr.trim()
    );
    false
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
