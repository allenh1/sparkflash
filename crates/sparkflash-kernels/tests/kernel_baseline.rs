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

/// The twelve shipped cubins, recompiled and compared byte for byte, on a host that has nvcc.
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
             not a file, so there is nothing to compile the twelve kernels with. Run it on a CUDA host \
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

/// The kernel list in `check-kernel-baseline.sh`, the manifest, the kernels on disk and the build
/// script's `PROBED_KERNELS` table must name the same twelve kernels. The script asserts the first
/// two when it runs; this pins all four without nvcc, so a kernel file added under `kernels/` - or a
/// probe list edited without the baseline - fails here instead of letting the success line mean less
/// than it says.
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
        12,
        "the list must name every baseline kernel - the twelve entries: the six Milestone 4 kernels, \
         the M7 SwiGLU quantiser epilogue and its weighted routed variant, the task-37 device \
         router, the task-8 routed gate/up GEMM, the serving tail's RMSNorm and the certified \
         head's device reduce: {listed:?}"
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

    // The probe table in build.rs is the fourth place a kernel has to be named. A kernel that is
    // pinned and listed but never probed on every build is a kernel nobody type-checked, which is
    // exactly the drift this task repaired: the manifest and the script named seven while the probe
    // table named six. The table is parsed by [probed_kernel_files], which reads the body as a whole
    // and pins the declared length, so a rustfmt-wrapped tuple cannot hide an entry and a one-line
    // table is not a false red; the parser's own behaviour is tested below.
    let build_script = std::fs::read_to_string(crate_dir.join("build.rs"))
        .expect("the build script must be readable");
    let (declared, probed) = probed_kernel_files(&build_script)
        .unwrap_or_else(|error| panic!("build.rs PROBED_KERNELS: {error}"));
    let mut sorted_probed: Vec<&str> = probed
        .iter()
        .map(|file| file.trim_end_matches(".cu"))
        .collect();
    sorted_probed.sort_unstable();
    assert_eq!(
        sorted_probed, sorted_listed,
        "build.rs PROBED_KERNELS ({probed:?}) and the baseline list ({listed:?}) must name the same kernels"
    );
    assert_eq!(
        declared,
        probed.len(),
        "the declared length and the parsed entries must agree"
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

/// The kernel files named by build.rs's `PROBED_KERNELS` table, and the length its type declares.
///
/// The table is `const PROBED_KERNELS: [(&str, &str); N] = [("file.cu", "symbol"), ...];`. The body
/// is parsed by collecting its string literals in order and keeping every file name, which is
/// robust to rustfmt wrapping a tuple and to a one-line table - the two shapes a line-prefix parser
/// gets wrong in opposite directions. The declared `N` is checked against the parsed entries, so a
/// count that disagrees with the body is an error here rather than a silent pass.
fn probed_kernel_files(build_script: &str) -> Result<(usize, Vec<String>), String> {
    let table_start = build_script
        .find("const PROBED_KERNELS")
        .ok_or("build.rs must declare the probed kernel table")?;
    let table = &build_script[table_start..];
    let open = table
        .find(';')
        .ok_or("the table type must declare its length")?
        + 1;
    let close = table[open..]
        .find(']')
        .ok_or("the declared length must be closed")?
        + open;
    let declared: usize = table[open..close]
        .trim()
        .parse()
        .map_err(|_| "the declared length must be a number".to_owned())?;
    let body_start = table.find("= [").ok_or("the table must be initialised")? + 3;
    let body_end = table[body_start..]
        .find("];")
        .ok_or("the probed kernel table must be closed")?
        + body_start;
    let mut literals: Vec<&str> = Vec::new();
    let mut rest = &table[body_start..body_end];
    while let Some(open) = rest.find('"') {
        let after_open = &rest[open + 1..];
        let close = after_open
            .find('"')
            .ok_or("a quoted string is always closed")?;
        literals.push(&after_open[..close]);
        rest = &after_open[close + 1..];
    }
    if !literals.len().is_multiple_of(2) {
        return Err(format!(
            "each PROBED_KERNELS entry is a (file, symbol) string pair: {literals:?}"
        ));
    }
    let probed: Vec<String> = literals
        .iter()
        .step_by(2)
        .map(|file| (*file).to_owned())
        .collect();
    if probed.len() != declared {
        return Err(format!(
            "the PROBED_KERNELS body names {} kernels but its type declares {declared}: {probed:?}",
            probed.len()
        ));
    }
    Ok((declared, probed))
}

/// The probe-table parser itself: wrapped tuples, one-line tables and a declared length that
/// disagrees with the body. This tests the parser without touching build.rs, because the drift the
/// checker exists for is exactly a build.rs edit that still compiles.
#[test]
fn the_probe_table_parser_handles_wrapping_one_liners_and_count_drift() {
    let one_line = r#"const PROBED_KERNELS: [(&str, &str); 2] = [("a_sm121.cu", "a_sm121"), ("b_sm121.cu", "b_sm121")];"#;
    assert_eq!(
        probed_kernel_files(one_line).expect("a one-line table must parse"),
        (2, vec!["a_sm121.cu".to_owned(), "b_sm121.cu".to_owned()])
    );

    // rustfmt wraps a long tuple; every entry must still be seen.
    let wrapped = r#"const PROBED_KERNELS: [(&str, &str); 2] = [
    (
        "a_very_long_kernel_name_sm121.cu",
        "a_very_long_kernel_name_sm121",
    ),
    ("b_sm121.cu", "b_sm121"),
];"#;
    assert_eq!(
        probed_kernel_files(wrapped).expect("a wrapped table must parse"),
        (
            2,
            vec![
                "a_very_long_kernel_name_sm121.cu".to_owned(),
                "b_sm121.cu".to_owned()
            ]
        )
    );

    // The declared length disagreeing with the body is the drift that used to pass when the parser
    // was line-prefix based: reject it here, even though build.rs would compile.
    let count_drift = r#"const PROBED_KERNELS: [(&str, &str); 3] = [("a_sm121.cu", "a_sm121"), ("b_sm121.cu", "b_sm121")];"#;
    let error = probed_kernel_files(count_drift).expect_err("a count drift must be rejected");
    assert!(
        error.contains("names 2 kernels but its type declares 3"),
        "the error must name both counts: {error}"
    );

    // And the case the test is really for: a wrapped 8th entry with the count raised, against a
    // 7-kernel manifest. The parser sees 8, so the list comparison above is what fails.
    let drifted = r#"const PROBED_KERNELS: [(&str, &str); 8] = [
    ("a_sm121.cu", "a_sm121"),
    ("b_sm121.cu", "b_sm121"),
    ("c_sm121.cu", "c_sm121"),
    ("d_sm121.cu", "d_sm121"),
    ("e_sm121.cu", "e_sm121"),
    ("f_sm121.cu", "f_sm121"),
    ("g_sm121.cu", "g_sm121"),
    (
        "a_long_new_kernel_sm121.cu",
        "a_long_new_kernel_sm121",
    ),
];"#;
    let (declared, probed) = probed_kernel_files(drifted).expect("the drifted table parses");
    assert_eq!((declared, probed.len()), (8, 8));
    let mut sorted: Vec<&str> = probed
        .iter()
        .map(|file| file.trim_end_matches(".cu"))
        .collect();
    sorted.sort_unstable();
    let manifest = [
        "a_sm121", "b_sm121", "c_sm121", "d_sm121", "e_sm121", "f_sm121", "g_sm121",
    ];
    assert_ne!(
        sorted, manifest,
        "the 8-entry table against a 7-kernel manifest must not compare equal"
    );
}

/// The loader resolves every name the pinned baseline uses, including the Mega-MHC alias.
///
/// The manifest keys the fused Mega-MHC kernel by its source-file stem, \`mega_mhc_sm121\`, while
/// the harness and the device open the cubin by its symbol, \`mega_mhc_fused_bf16_sm121\`. Run D
/// failed on 2026-10-18 because the pinned artifact had been copied under the manifest's name and
/// the harness opened the symbol's name (docs/build/phase3-indexer.md sections 2 and 3). A cubin's
/// file name is not part of its ABI, so both names resolve to the same kernel; this test fails if
/// either the manifest key or the alias list drifts away from the other.
#[test]
fn the_loader_resolves_every_name_the_pinned_baseline_uses() {
    use sparkflash_kernels::cuda_driver::KernelId;

    // Every kernel the shared driver owns is reachable from its symbol, and its primary cubin
    // name is that symbol: the file the harness documents and opens.
    for id in KernelId::ALL {
        let names = id.cubin_names();
        assert!(!names.is_empty(), "{id:?} must have a cubin name");
        assert_eq!(
            KernelId::from_cubin_name(names[0]),
            Some(id),
            "{id:?}'s primary name {} must resolve back to it",
            names[0]
        );
        assert!(
            !names.contains(&"flash_mla_csa2_sm121"),
            "{id:?} must not claim the flash_mla kernel, which has no KernelId"
        );
    }
    assert_eq!(
        KernelId::from_cubin_name("flash_mla_csa2_sm121"),
        None,
        "flash_mla is loaded through load_flash_mla and must not resolve to a KernelId"
    );
    // The alias is one kernel, not a second one.
    assert_eq!(
        KernelId::from_cubin_name("mega_mhc_sm121"),
        Some(KernelId::MegaMhcBf16)
    );
    assert_eq!(
        KernelId::from_cubin_name("mega_mhc_fused_bf16_sm121"),
        Some(KernelId::MegaMhcBf16)
    );

    // And the manifest's own key must be a name the loader answers to.
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = std::fs::read_to_string(crate_dir.join("baseline/kernel_baseline.txt"))
        .expect("the manifest must be readable");
    let mut checked = 0;
    for line in manifest.lines() {
        let Some(rest) = line.strip_prefix("kernel ") else {
            continue;
        };
        let name = rest
            .split_whitespace()
            .next()
            .expect("a manifest kernel line names the kernel");
        if name == "flash_mla_csa2_sm121" {
            continue;
        }
        assert!(
            KernelId::from_cubin_name(name).is_some(),
            "the baseline manifest names {name}, which the loader does not resolve"
        );
        checked += 1;
    }
    assert_eq!(
        checked, 11,
        "the manifest pins eleven driver kernels (twelve entries; flash_mla has no KernelId)"
    );
}

/// The name resolver stages 2-9 use: alias, symbol preference, flash_mla and the two error paths.
///
/// This exercises `cuda_driver::cubin_path` itself, not the helper's pieces: the HIGH this test
/// covers was stage 2 passing `flash_mla_csa2_sm121` to a resolver that only knew `KernelId`s and
/// therefore errored before reading a cubin. A temp directory holds the files so the test needs no
/// CUDA and no built cubin.
#[test]
fn the_cubin_resolver_handles_aliases_flash_mla_and_unknown_names() {
    use sparkflash_kernels::cuda_driver::{KernelId, cubin_path};

    let dir = std::env::temp_dir().join(format!(
        "sparkflash-cubin-path-{}-{}",
        std::process::id(),
        "resolver"
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a temp directory");

    // The pinned manifest's Mega-MHC stem is enough: the alias resolves to that file.
    std::fs::write(dir.join("mega_mhc_sm121.cubin"), b"alias").expect("write the alias cubin");
    assert_eq!(
        cubin_path(&dir, "mega_mhc_fused_bf16_sm121").expect("the alias resolves"),
        dir.join("mega_mhc_sm121.cubin")
    );

    // When both files exist the symbol's own name wins: it is the primary name, not a fallback.
    std::fs::write(dir.join("mega_mhc_fused_bf16_sm121.cubin"), b"symbol")
        .expect("write the symbol");
    assert_eq!(
        cubin_path(&dir, "mega_mhc_fused_bf16_sm121").expect("the symbol resolves"),
        dir.join("mega_mhc_fused_bf16_sm121.cubin")
    );

    // flash_mla has no KernelId (load_flash_mla owns the symbol) and is what stage 2 opens.
    std::fs::write(dir.join("flash_mla_csa2_sm121.cubin"), b"mla")
        .expect("write the flash_mla cubin");
    assert_eq!(
        cubin_path(&dir, "flash_mla_csa2_sm121").expect("flash_mla resolves"),
        dir.join("flash_mla_csa2_sm121.cubin")
    );
    assert_eq!(
        KernelId::from_cubin_name("flash_mla_csa2_sm121"),
        None,
        "the crate's kernel-id vocabulary must keep flash_mla outside it"
    );

    // A name that is neither is refused as an unknown kernel, before any file is looked at.
    let unknown = cubin_path(&dir, "not_a_kernel_sm121").expect_err("an unknown kernel is refused");
    assert!(
        unknown.contains("not one of the shared driver kernels"),
        "the unknown-name error must name the driver-kernel contract: {unknown}"
    );

    // A known kernel with no built file names every candidate it tried.
    let missing =
        cubin_path(&dir, "b12x_mxfp8_dense_sm121").expect_err("a missing cubin is an error");
    assert!(
        missing.contains("b12x_mxfp8_dense_sm121.cubin"),
        "the missing-file error must list the candidate it tried: {missing}"
    );

    std::fs::remove_dir_all(&dir).expect("clean up the temp directory");
}
