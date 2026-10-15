//! `sparkflash` command line interface.
//!
//! The binary is a thin dispatcher: every subcommand lives in its own module here so that it
//! can be exercised as a library, with injected probes, without spawning a process.

pub mod doctor;
