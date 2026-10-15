//! `sparkflash` command-line entry point.
//!
//! The binary is a thin dispatcher: each subcommand is one arm here and the logic lives in
//! the crate that owns it, so everything the binary can do is also reachable as a library call
//! and testable without a process. `doctor`, `trace` and `pack-engram` are implemented today;
//! `serve` arrives with the crate that backs it.
//!
//! The arms parse differently on purpose. `doctor` parses its own arguments inside
//! `sparkflash_cli::doctor` because it takes a repeated `--nodes` list and echoes probe output
//! verbatim, so its arm forwards the remaining arguments untouched. `pack-engram` is a clap
//! command whose options are shared verbatim with the standalone `pack-engram` binary, so its
//! arguments are handed to clap with the subcommand name restored as the program name, and
//! `trace` forwards to the scheduler's own CLI for the same reason `doctor` does.

use std::process::ExitCode;

use clap::Parser;
use sparkflash_cli::doctor;
use sparkflash_engram::packer::{CliArgs, run_cli};

/// `pack-engram` options, shared verbatim with the standalone binary.
#[derive(Debug, Parser)]
#[command(
    name = "sparkflash pack-engram",
    version,
    about = "Pack the Engram embedding tables into 512-byte aligned per-rank files"
)]
struct PackEngramCli {
    /// Packing options.
    #[command(flatten)]
    args: CliArgs,
}

fn main() -> ExitCode {
    let mut args = std::env::args();
    let _program = args.next();
    match args.next().as_deref() {
        Some("doctor") => doctor::run(args),
        Some("trace") => sparkflash_scheduler::trace_cli::run(args),
        Some("pack-engram") => pack_engram(args),
        _ => {
            println!("sparkflash: usage: sparkflash <doctor|trace|pack-engram>");
            ExitCode::SUCCESS
        }
    }
}

/// Runs `pack-engram` with the arguments that followed the subcommand.
fn pack_engram(args: impl Iterator<Item = String>) -> ExitCode {
    let cli = PackEngramCli::parse_from(
        std::iter::once("sparkflash pack-engram".to_string()).chain(args),
    );
    match run_cli(cli.args) {
        Ok(report) => {
            if !report.dry_run {
                println!(
                    "pack-engram: wrote {} file(s), {} bytes, into {}",
                    report.files_written.len(),
                    report.bytes_written,
                    report.output_dir.display()
                );
                println!("pack-engram: manifest: {}", report.manifest_path.display());
                if report.verified_files > 0 {
                    println!(
                        "pack-engram: verified {} file(s) against {} sampled rows",
                        report.verified_files, report.verified_sample_rows
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("sparkflash pack-engram: {error:#}");
            ExitCode::FAILURE
        }
    }
}
