//! The cubin catalog and its fail-closed verification.
//!
//! The device chain must not launch a cubin it has not verified against the repository baseline
//! (docs/build/device-forward-chain.md 4.2 and 7 M3): the catalog is parsed from
//! crates/sparkflash-kernels/baseline/kernel_baseline.txt through sparkflash_kernels::baseline,
//! not from hand-copied
//! SHA-256 constants, and [verify_pins] checks size **and** digest before any launch path exists.
//! Every failure is an error: a missing file, a short read, a size mismatch or a digest mismatch.
//! There is no "best effort" mode and no fallback to an unpinned cubin.
//!
//! The M3-a skeleton does not open a driver; this module is the catalog and the check the first
//! device pass will call. Its tests use temporary files, so the fail-closed behavior is exercised
//! on a CPU-only host.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use sparkflash_kernels::baseline::{KernelPin, pin, pins};

/// The six kernels the chain stage map names (design 2.2 / 4.2).
pub const CHAIN_KERNELS: [&str; 6] = [
    "mega_mhc_sm121",
    "b12x_mxfp8_dense_sm121",
    "b12x_mxfp4_moe_sm121",
    "flash_mla_csa2_sm121",
    "b12x_mxfp8_wo_a_sm121",
    "cert_lm_head_sm121",
];

/// The router kernel measured in the M2 window: merged, one launch, not wired into the chain.
pub const ROUTER_KERNEL: &str = "moe_router_sm121";

/// The routed gate/up kernel merged on main b54f11e: measured in isolation, not wired into the chain.
pub const GATE_UP_KERNEL: &str = "routed_gate_up_sm121";

/// Every kernel the device path may launch today or after the first wiring: the six chain kernels,
/// the router, the routed gate/up and the c4c tail RMSNorm. The SwiGLU/quantiser is a device stage in
/// the c4c shared expert, but the driver's chain catalog does not require it here; the CLI backend
/// verifies it directly against the baseline.
pub const REQUIRED_KERNELS: [&str; 9] = [
    "mega_mhc_sm121",
    "b12x_mxfp8_dense_sm121",
    "b12x_mxfp4_moe_sm121",
    "flash_mla_csa2_sm121",
    "b12x_mxfp8_wo_a_sm121",
    "cert_lm_head_sm121",
    "moe_router_sm121",
    "routed_gate_up_sm121",
    "rmsnorm_sm121",
];

/// The catalog of required kernel pins, read from the embedded baseline.
///
/// # Errors
/// A baseline parse error, or a required kernel the baseline does not pin (fail closed: the catalog
/// is never silently short).
pub fn required_pins() -> Result<Vec<KernelPin>, String> {
    let mut required = Vec::with_capacity(REQUIRED_KERNELS.len());
    for name in REQUIRED_KERNELS {
        let found = pin(name)?.ok_or_else(|| {
            format!(
                "kernel {name} is required by the device chain but the baseline does not pin it"
            )
        })?;
        required.push(found);
    }
    Ok(required)
}

/// The path of one pinned cubin inside a directory.
#[must_use]
pub fn cubin_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.cubin"))
}

/// Verifies one cubin's bytes against one pin.
///
/// # Errors
/// A size mismatch (with both sizes) or a SHA-256 mismatch (with both digests).
pub fn verify_bytes(bytes: &[u8], expected: &KernelPin) -> Result<(), String> {
    if bytes.len() != expected.cubin_bytes {
        return Err(format!(
            "cubin {}: {} bytes, baseline pins {}",
            expected.name,
            bytes.len(),
            expected.cubin_bytes
        ));
    }
    let digest = hex(&Sha256::digest(bytes));
    if digest != expected.cubin_sha256 {
        return Err(format!(
            "cubin {}: sha256 {digest}, baseline pins {}",
            expected.name, expected.cubin_sha256
        ));
    }
    Ok(())
}

/// Verifies every pinned cubin in a directory, failing closed on the first missing or mismatched one.
///
/// # Errors
/// A missing/unreadable cubin, a size mismatch or a digest mismatch, named by kernel.
pub fn verify_pins(dir: &Path, expected: &[KernelPin]) -> Result<(), String> {
    for expected in expected {
        let path = cubin_path(dir, &expected.name);
        let bytes = std::fs::read(&path).map_err(|error| {
            format!(
                "cubin {}: cannot read {}: {error}",
                expected.name,
                path.display()
            )
        })?;
        verify_bytes(&bytes, expected)?;
    }
    Ok(())
}

/// Verifies the required catalog in a cubin directory before any launch path is built.
///
/// # Errors
/// [required_pins]' catalog error or [verify_pins]' verification error.
pub fn verify_required_dir(dir: &Path) -> Result<Vec<KernelPin>, String> {
    let required = required_pins()?;
    verify_pins(dir, &required)?;
    Ok(required)
}

/// Lower-case hex, the baseline's digest spelling.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

/// The baseline pins for the named kernels, for tests and for grouping report lines.
///
/// # Errors
/// [pins]' parse error.
pub fn baseline_pins() -> Result<Vec<KernelPin>, String> {
    pins()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("temp dir")
    }

    fn write_pinned(dir: &Path, name: &str, contents: &[u8]) -> KernelPin {
        std::fs::write(cubin_path(dir, name), contents).expect("write cubin");
        KernelPin {
            name: name.to_string(),
            cubin_bytes: contents.len(),
            cubin_sha256: hex(&Sha256::digest(contents)),
            ptx_membar_lines: 0,
        }
    }

    #[test]
    fn the_required_catalog_covers_the_chain_router_and_gate_up() {
        let required = required_pins().expect("catalog");
        assert_eq!(required.len(), REQUIRED_KERNELS.len());
        for name in CHAIN_KERNELS {
            assert!(
                required.iter().any(|pin| pin.name == name),
                "{name} required"
            );
        }
        assert!(required.iter().any(|pin| pin.name == ROUTER_KERNEL));
        assert!(required.iter().any(|pin| pin.name == GATE_UP_KERNEL));
        // Six chain kernels plus router, gate/up and the c4c tail norm; SwiGLU stays a host seam in
        // the routed path and runs on device only in the shared expert.
        assert_eq!(required.len(), 9);
        // Every m4_abi kernel is one of the chain kernels (m4_abi stays frozen, D26).
        for (source_name, _) in sparkflash_kernels::m4_abi::KERNELS {
            let name = source_name.strip_suffix(".cu").unwrap_or(source_name);
            assert!(
                required.iter().any(|pin| pin.name == name),
                "m4_abi kernel {source_name} is not in the required catalog"
            );
        }
    }

    #[test]
    fn verification_passes_on_matching_bytes_and_fails_closed_on_every_mismatch() {
        let dir = temp_dir();
        let pins: Vec<KernelPin> = REQUIRED_KERNELS
            .iter()
            .map(|name| {
                write_pinned(
                    dir.path(),
                    name,
                    format!("cubin bytes for {name}").as_bytes(),
                )
            })
            .collect();
        verify_pins(dir.path(), &pins).expect("matching cubins verify");
        verify_required_dir(dir.path())
            .expect_err("the fake cubins are not the real baseline bytes");

        let bad_digest = KernelPin {
            cubin_sha256: "0".repeat(64),
            ..pins[0].clone()
        };
        assert!(verify_pins(dir.path(), &[bad_digest]).is_err());
        let bad_size = KernelPin {
            cubin_bytes: pins[0].cubin_bytes + 1,
            ..pins[0].clone()
        };
        assert!(
            verify_bytes(b"short", &bad_size)
                .expect_err("size")
                .contains("bytes")
        );
        std::fs::remove_file(cubin_path(dir.path(), CHAIN_KERNELS[0])).expect("remove");
        assert!(
            verify_pins(dir.path(), &[pins[0].clone()])
                .expect_err("missing file")
                .contains("cannot read")
        );
    }

    #[test]
    fn extra_files_in_the_directory_do_not_satisfy_a_missing_pin() {
        let dir = temp_dir();
        std::fs::write(dir.path().join("unrelated.cubin"), b"x").expect("write");
        let expected = KernelPin {
            name: "missing_kernel".to_string(),
            cubin_bytes: 1,
            cubin_sha256: "0".repeat(64),
            ptx_membar_lines: 0,
        };
        assert!(verify_pins(dir.path(), &[expected]).is_err());
    }
}
