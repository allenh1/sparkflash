//! `sparkflash` command-line entry point.
//!
//! The binary is a thin dispatcher: each subcommand is one arm here and the logic lives in
//! `sparkflash_cli`, so it can be exercised as a library with injected probes and without
//! spawning a process. `doctor` and `trace` are implemented today; `pack-engram` and
//! `serve` arrive with the crates that back them.

use std::process::ExitCode;

use sparkflash_cli::doctor;

fn main() -> ExitCode {
    let mut args = std::env::args();
    let _program = args.next();
    match args.next().as_deref() {
        Some("doctor") => doctor::run(args),
        Some("trace") => sparkflash_scheduler::trace_cli::run(args),
        _ => {
            println!("sparkflash: Milestone 1 scaffolding");
            ExitCode::SUCCESS
        }
    }
}
