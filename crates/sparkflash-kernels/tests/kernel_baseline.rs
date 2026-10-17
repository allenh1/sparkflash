//! The committed kernel baseline: cubin bytes, cubin sha256 and PTX membar count per kernel.
//!
//! `scripts/check-kernel-baseline.sh` recompiles the baseline kernels with nvcc and compares
//! them with `baseline/kernel_baseline.txt`. That is the check which would catch a kernel silently
//! changing - the acquire fence disappearing, a codec constant moving - and it is the only check in
//! this crate that reads what the compiler really produced rather than what the source says.
//!
//! It runs here as well as in the cuda-abi CI job so that a CUDA host running plain `cargo test`
//! covers it. On a host without nvcc there is nothing to compile with, so the test skips and says so
//! on stdout rather than passing silently or failing for the wrong reason.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The nvcc to run: $SPARKFLASH_NVCC when it names a file, otherwise the first `nvcc` on $PATH.
fn nvcc() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("SPARKFLASH_NVCC") {
        let path = PathBuf::from(explicit);
        return path.is_file().then_some(path);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join("nvcc"))
        .find(|candidate| candidate.is_file())
}

/// The six shipped cubins, recompiled and compared byte for byte, on a host that has nvcc.
///
/// On a host WITHOUT nvcc the check returns early, and the line it prints is on the test's captured
/// stdout, so a green `cargo test` run does not show it - libtest only shows it under `--nocapture`.
/// The channel that states the same thing on every run is the build script's `cargo:warning`, which
/// names the arch and the sources it could not compile, and CI's `cuda-abi` job has nvcc, so this
/// check is not skipped where it matters. Nothing here claims a skip is loud.
#[test]
fn the_kernels_still_compile_to_the_recorded_baseline() {
    let Some(nvcc) = nvcc() else {
        println!(
            "skipping the kernel baseline check: no nvcc on PATH and SPARKFLASH_NVCC is unset or \
             not a file, so there is nothing to compile the six kernels with. Run it on a CUDA host \
             or inside nvidia/cuda:13.0.3-devel-ubuntu22.04: /bin/sh \
             crates/sparkflash-kernels/scripts/check-kernel-baseline.sh"
        );
        return;
    };
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = crate_dir.join("scripts/check-kernel-baseline.sh");
    assert!(script.is_file(), "missing {}", script.display());
    let output = Command::new("/bin/sh")
        .arg(&script)
        .env("SPARKFLASH_NVCC", &nvcc)
        .current_dir(crate_dir)
        .output()
        .unwrap_or_else(|error| panic!("cannot run {}: {error}", script.display()));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "the kernel baseline check failed with {}:\n{stdout}\n{stderr}",
        output.status
    );
    print!("{stdout}");
}

/// The kernel list in `check-kernel-baseline.sh`, the manifest and the kernels on disk must name the
/// same six kernels. The script asserts the first two when it runs; this pins both without nvcc, and
/// adds the third - a kernel file added under `kernels/` has to be added to the list AND to the
/// manifest, which is the drift that would otherwise let the success line mean less than it says.
#[test]
fn the_baseline_list_matches_the_manifest_and_the_kernels_on_disk() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = std::fs::read_to_string(crate_dir.join("scripts/check-kernel-baseline.sh"))
        .expect("the check script must be readable");
    let list_line = script
        .lines()
        .find(|line| line.starts_with("kernels="))
        .expect("the script must name its kernel list");
    let listed: Vec<&str> = list_line
        .trim_start_matches("kernels=")
        .trim_matches('"')
        .split_whitespace()
        .collect();
    assert_eq!(
        listed.len(),
        7,
        "the list must name every baseline kernel - the six Milestone 4 kernels and the M7 SwiGLU          quantiser epilogue: {listed:?}"
    );

    let manifest = std::fs::read_to_string(crate_dir.join("baseline/kernel_baseline.txt"))
        .expect("the manifest must be readable");
    let recorded: Vec<&str> = manifest
        .lines()
        .filter_map(|line| line.strip_prefix("kernel "))
        .filter_map(|rest| rest.split_whitespace().next())
        .collect();
    let mut sorted_listed = listed.clone();
    sorted_listed.sort_unstable();
    let mut sorted_recorded = recorded.clone();
    sorted_recorded.sort_unstable();
    assert_eq!(
        sorted_listed, sorted_recorded,
        "the script's list and the manifest must name the same kernels"
    );

    // Every listed kernel is a file under kernels/, and every kernel file there except the RoCEnante
    // one is listed: that kernel belongs to another milestone, is compiled by build.rs's own probe and
    // is covered by the syntax gate, not by this baseline.
    let kernels_dir = crate_dir.join("../../kernels");
    let mut on_disk: Vec<String> = std::fs::read_dir(&kernels_dir)
        .expect("kernels/ must be readable")
        .map(|entry| {
            entry
                .expect("a readable directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.ends_with(".cu"))
        .collect();
    on_disk.sort();
    for name in &listed {
        assert!(
            on_disk.iter().any(|file| file == &format!("{name}.cu")),
            "{name}.cu must exist under kernels/: {on_disk:?}"
        );
    }
    let unlisted: Vec<&str> = on_disk
        .iter()
        .map(String::as_str)
        .filter(|file| !listed.contains(&file.trim_end_matches(".cu")))
        .collect();
    assert_eq!(
        unlisted,
        vec!["rocenante_oneshot_sm121.cu"],
        "only the RoCEnante kernel may sit outside the baseline list"
    );
}
