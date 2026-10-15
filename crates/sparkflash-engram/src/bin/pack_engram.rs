//! §sparkflash pack-engram§: materialise the per-rank Engram tables from a safetensors
//! checkpoint.
//!
//! The heavy lifting lives in §[sparkflash_engram::packer]§; this binary only owns the
//! process exit status and the summary line, so the same code path is reachable through
//! §sparkflash pack-engram§.

use clap::Parser;
use sparkflash_engram::packer::{CliArgs, PackReport, run_cli};

/// Command line surface of the standalone packer.
#[derive(Debug, Parser)]
#[command(
    name = "pack-engram",
    version,
    about = "Pack the Engram embedding tables into 512-byte aligned per-rank files"
)]
struct PackEngram {
    /// Packing options, shared verbatim with §sparkflash pack-engram§.
    #[command(flatten)]
    args: CliArgs,
}

fn main() -> std::process::ExitCode {
    let PackEngram { args } = PackEngram::parse();
    match run_cli(args) {
        Ok(report) => {
            summarise(&report);
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("pack-engram: {error:#}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Prints what a real run produced; a dry run has already printed its plan.
fn summarise(report: &PackReport) {
    if report.dry_run {
        return;
    }
    println!(
        "wrote {} file(s), {} bytes, into {}",
        report.files_written.len(),
        report.bytes_written,
        report.output_dir.display()
    );
    println!("manifest: {}", report.manifest_path.display());
    if report.verified_files > 0 {
        println!(
            "verified {} file(s) against {} sampled rows",
            report.verified_files, report.verified_sample_rows
        );
    }
}
