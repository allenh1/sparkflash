//! Writes a synthetic four-rank device token window for the S7/S8 serving rehearsal.
//!
//! The records are paper evidence (UNMEASURED): no device produced their ids, text, positions or
//! dump. The point is to stand up the exact S7/S8 command chain before a real window exists, and to
//! let a node-bound rehearsal use the same flags and script as the real probe.
//!
//!     cargo run -p sparkflash-api --features device-token-fixtures \
//!         --example write_device_window -- /tmp/sparkflash-window asserted
//!
//! The second argument selects the claim form the window is written for: asserted (every rank
//! token_claimed true) or derived (complete loop evidence with the writer's flag still false).

use std::path::PathBuf;
use std::process::ExitCode;

use sparkflash_api::device_chain::fixtures;

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let Some(dir) = arguments.next().map(PathBuf::from) else {
        eprintln!("usage: write_device_window DIR [asserted|derived]");
        return ExitCode::from(2);
    };
    let mode = arguments.next().unwrap_or_else(|| "asserted".to_owned());
    if arguments.next().is_some() {
        eprintln!("usage: write_device_window DIR [asserted|derived]");
        return ExitCode::from(2);
    }
    std::fs::create_dir_all(&dir).expect("window directory");
    match mode.as_str() {
        "asserted" => fixtures::write_envelope(&dir, true),
        "derived" => fixtures::write_derived_envelope(&dir),
        other => {
            eprintln!("mode must be asserted or derived, not {other}");
            return ExitCode::from(2);
        }
    }
    fixtures::relocate_to_rank_directories(&dir);
    println!(
        "wrote a synthetic {mode} four-rank token window (UNMEASURED, paper records) to {}",
        dir.display()
    );
    ExitCode::SUCCESS
}
