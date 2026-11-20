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

//! Parse a numerics registry exactly as a serving binary does.
//!
//! `cargo run -q -p sparkflash-model --example validate_registry -- numerics/registry.toml`
//! exits 0 when the file parses and validates, 1 with the parser's error otherwise. The
//! declaration check runs it, so the Python gate cannot accept a registry the Rust parser
//! refuses (the version 2 declaration's `[[retired]]` reference was exactly that gap).

use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: validate_registry <registry.toml>");
        return ExitCode::from(2);
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("{path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    match sparkflash_model::numerics::Registry::from_toml(&text) {
        Ok(registry) => {
            println!(
                "ok: {} configuration(s), {} anchor(s), {} retired",
                registry.config.len(),
                registry.anchor.len(),
                registry.retired.len()
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{path}: {error}");
            ExitCode::FAILURE
        }
    }
}
