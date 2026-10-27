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

//! rocenante-probe: the instrument behind the doctor's one-shot All-Reduce latency gate.
//!
//! The doctor in sparkflash-cli runs rocenante-probe --bytes SIZE --peer IP --json and reads
//! oneshot_allreduce_us from stdout. This binary writes exactly one JSON object to stdout when
//! --json is given and keeps every diagnostic on stderr, so the doctor's single
//! serde_json::from_str over stdout either parses the measurement or fails loudly.
//!
//! Exit codes: 0 measured, 2 the command line is not usable, 3 the measurement path is not
//! available on this host or in this build, 4 the exchange ran and failed. On every non-zero exit
//! stdout stays empty, because a probe that prints a plausible number it could not measure is
//! worse than a probe that fails; there is no fallback, no estimate and no partial run.

use clap::Parser;
use sparkflash_comm::probe::{Args, ProbeError, ProbeRequest, run};

fn main() {
    let args = Args::parse();
    let json = args.json;
    let request = match ProbeRequest::from_args(args) {
        Ok(request) => request,
        Err(error) => fail(&error),
    };
    match run(&request) {
        Ok(report) => {
            if json {
                println!("{}", report.to_json_string());
            } else {
                println!("{}", report.to_human_string());
            }
        }
        Err(error) => fail(&error),
    }
}

fn fail(error: &ProbeError) -> ! {
    eprintln!("rocenante-probe: {error}");
    std::process::exit(error.exit_code());
}
