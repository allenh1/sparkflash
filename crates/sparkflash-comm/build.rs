//! Build-time half of the RDMA backend: compiles the three inline libibverbs entry points.
//!
//! libibverbs' public header declares ibv_post_send, ibv_post_recv and ibv_poll_cq as `static
//! inline`; they are not exported by libibverbs.so.1 (measured with `nm -D` against rdma-core 39:
//! 16 of the 19 entry points this backend uses are exported, those three are not). The library
//! still has to be opened at run time - that is what keeps the default build free of an RDMA
//! dependency - so this script compiles a four-function C shim that includes the real header and
//! lets the inline wrappers do what rdma-core intends. The shim calls no libibverbs symbol: the
//! inline wrappers dispatch through the QP's provider ops, so nothing is linked against
//! libibverbs and the run-time dlopen still owns the device context.
//!
//! Why not mirror `struct ibv_context_ops` in Rust and call the provider directly: that vtable is
//! internal to rdma-core, its layout changes between releases, and no published header promises
//! it. The header's inline functions are the only stable way to reach the same calls, and they
//! need the header at build time, which is exactly what this script uses.
//!
//! A host without <infiniband/verbs.h> still builds the rdma feature: the shim falls back to
//! functions returning ENOSYS and `Verbs::load` reports a typed refusal naming the missing
//! build prerequisite. That keeps `ci/check_features.py` honest on the hardware-free CI host
//! while a node built with libibverbs-dev gets the real path.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/rdma/ibverbs_shim.c");
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_RDMA").is_none() {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let target = env::var("TARGET").expect("cargo sets TARGET");
    let compiler = tool(&[
        format!("CC_{}", target.replace('-', "_")),
        "CC".to_owned(),
        "cc".to_owned(),
    ]);
    let archiver = tool(&[
        format!("AR_{}", target.replace('-', "_")),
        "AR".to_owned(),
        "ar".to_owned(),
    ]);
    let source = Path::new("src/rdma/ibverbs_shim.c");
    match compile_shim(&compiler, &archiver, source, &out) {
        Ok(()) => {
            println!("cargo:rustc-link-search=native={}", out.display());
            println!("cargo:rustc-link-lib=static=sparkflash_ibverbs_shim");
            println!("cargo:rustc-env=SPARKFLASH_IBVERBS_SHIM=system");
        }
        Err(detail) => {
            let stub = out.join("ibverbs_shim_stub.c");
            fs::write(&stub, STUB).expect("write the ENOSYS shim stub");
            compile_shim(&compiler, &archiver, &stub, &out).unwrap_or_else(|error| {
                panic!("cannot compile the libibverbs shim ({detail}) nor its ENOSYS stub ({error}); set CC/AR for target {target}");
            });
            println!("cargo:rustc-link-search=native={}", out.display());
            println!("cargo:rustc-link-lib=static=sparkflash_ibverbs_shim");
            println!("cargo:rustc-env=SPARKFLASH_IBVERBS_SHIM=stub");
            println!(
                "cargo:warning=sparkflash-comm: rdma feature built without <infiniband/verbs.h> ({detail}); ibv_post_send/ibv_post_recv/ibv_poll_cq are stubs returning ENOSYS. Install libibverbs-dev and rebuild to get the real inline wrappers."
            );
        }
    }
}

/// Writes the C shim, assembles it with the archiver, and fails with the compiler's own text.
fn compile_shim(compiler: &str, archiver: &str, source: &Path, out: &Path) -> Result<(), String> {
    let object = out.join("sparkflash_ibverbs_shim.o");
    let archive = out.join("libsparkflash_ibverbs_shim.a");
    let compile = Command::new(compiler)
        .arg("-O2")
        .arg("-fPIC")
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(&object)
        .output()
        .map_err(|error| format!("cannot run {compiler}: {error}"))?;
    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        return Err(format!(
            "{} -c {} failed: {}",
            compiler,
            source.display(),
            stderr.trim()
        ));
    }
    let _ = fs::remove_file(&archive);
    let assemble = Command::new(archiver)
        .arg("crs")
        .arg(&archive)
        .arg(&object)
        .output()
        .map_err(|error| format!("cannot run {archiver}: {error}"))?;
    if !assemble.status.success() {
        let stderr = String::from_utf8_lossy(&assemble.stderr);
        return Err(format!("{archiver} failed: {}", stderr.trim()));
    }
    Ok(())
}

/// The first tool in `candidates` that is set in the environment, or the last fallback.
fn tool(candidates: &[String]) -> String {
    for name in candidates {
        if let Ok(value) = env::var(name)
            && !value.trim().is_empty()
        {
            return value;
        }
    }
    candidates
        .last()
        .cloned()
        .unwrap_or_else(|| "cc".to_owned())
}

/// The ENOSYS fallback used when the build host has no libibverbs headers.
const STUB: &str = r#"#include <errno.h>
int sparkflash_ibv_post_send(void *qp, void *wr, void **bad) {
    (void)qp; (void)wr; if (bad) { *bad = 0; } return ENOSYS;
}
int sparkflash_ibv_post_recv(void *qp, void *wr, void **bad) {
    (void)qp; (void)wr; if (bad) { *bad = 0; } return ENOSYS;
}
int sparkflash_ibv_poll_cq(void *cq, int count, void *wc) {
    (void)cq; (void)count; (void)wc; return -ENOSYS;
}
"#;
