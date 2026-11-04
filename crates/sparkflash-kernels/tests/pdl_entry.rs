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

//! Programmatic dependent launch is safe only for a kernel that waits on its predecessor.
//!
//! The driver launches every kernel named with one of [PROGRAMMATIC_PREFIXES] with programmatic
//! stream serialization, so it may start while the previous kernel still runs. Stream order holds
//! only if the kernel reaches griddepcontrol.wait (OPT_PDL_WAIT, or OPT_PDL_ENTRY) before it reads
//! anything an earlier kernel writes or writes anything at all. A kernel that forgot it would
//! race, and nothing would say so; this test reads every such kernel in kernels/ and refuses one
//! whose body never waits.

use sparkflash_kernels::cuda_driver::PROGRAMMATIC_PREFIXES;

/// Kernels that need no wait: L2 prefetches of constant weight spans, which write nothing.
const WRITES_NOTHING: [&str; 2] = ["opt_rows_prefetch_l2", "q4e_prefetch_l2"];

/// The index just past the bracket that closes the one opened before `from`.
fn closing(text: &str, from: usize, open: u8, close: u8) -> usize {
    let mut depth = 1usize;
    for (offset, byte) in text.as_bytes()[from..].iter().enumerate() {
        if *byte == open {
            depth += 1;
        } else if *byte == close {
            depth -= 1;
            if depth == 0 {
                return from + offset + 1;
            }
        }
    }
    panic!("unbalanced {} in a kernel source", open as char);
}

/// Every `extern "C" __global__` definition in `text`, as (name, body).
fn kernels(text: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(offset) = text[at..].find("extern \"C\" __global__") {
        // A kernel returns void; its name runs from there to the parameter list, past any
        // __launch_bounds__(...) before it.
        let start = at + offset;
        let named = start + text[start..].find(" void ").expect("a kernel returns void") + 6;
        let open = named
            + text[named..]
                .find('(')
                .expect("a kernel has a parameter list");
        let name = text[named..open].trim().to_owned();
        let params = closing(text, open + 1, b'(', b')');
        at = params;
        if text[params..].trim_start().starts_with(';') {
            continue;
        }
        let brace = params
            + text[params..]
                .find('{')
                .expect("a kernel definition has a body");
        let end = closing(text, brace + 1, b'{', b'}');
        found.push((name, text[brace..end].to_owned()));
        at = end;
    }
    found
}

#[test]
fn every_programmatic_kernel_waits_for_the_previous_grid() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut sources: Vec<_> = std::fs::read_dir(&root)
        .expect("kernels/ must be readable from the crate")
        .map(|entry| entry.expect("a kernels/ entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "cu"))
        .collect();
    sources.sort();
    let mut checked = 0;
    let mut missing = Vec::new();
    for path in &sources {
        let text = std::fs::read_to_string(path).expect("a kernel source must be readable");
        for (name, body) in kernels(&text) {
            if !PROGRAMMATIC_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix))
            {
                continue;
            }
            checked += 1;
            let waits = body.contains("OPT_PDL_ENTRY()") || body.contains("OPT_PDL_WAIT()");
            if !waits && !WRITES_NOTHING.contains(&name.as_str()) {
                missing.push(format!("{} in {}", name, path.display()));
            }
        }
    }
    assert!(
        checked >= 90,
        "found only {checked} programmatic kernels; the scan has gone stale"
    );
    assert!(
        missing.is_empty(),
        "programmatic kernels that never wait for the previous grid: {missing:#?}"
    );
}
