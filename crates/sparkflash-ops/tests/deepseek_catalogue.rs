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

//! The DeepSeek catalogue tripwire.
//!
//! `sparkflash-deepseek-forward` picks its kernels by name: every launch resolves through
//! `sparkflash_kernels::cuda_driver::Cuda::load_named`. The names reach those sites as string
//! literals through local loader closures (`load`, `load_misc`, the draft's `load(file, name)`,
//! and the two `opt_select` loaders), so the literal set is the set the forward loads. This test
//! reads the forward crate's sources, keeps the literals that name an [Image::Deepseek] export,
//! and requires every one of them to be a symbol in [CATALOGUE]. The one allowed-exceptions table
//! names any symbol that is deliberately absent and why; it is empty today.
//!
//! This is the mechanical closure of the phase-6 gap the audit measured: the image exported 219
//! symbols, 127 of them named by the forward, and the catalogue covered only a fraction. A
//! kernel the arm launches but the catalogue does not know about now fails a test instead of
//! waiting for the resolver to disagree with the launch path.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sparkflash_kernels::images::Image;
use sparkflash_ops::CATALOGUE;

/// A symbol the forward names but the catalogue deliberately does not carry, with the reason.
///
/// A line is a promise with a citation, not a shortcut: the test fails if the reason is empty or
/// if the symbol is no longer named by the forward, so the table cannot go stale.
const ALLOWED_UNCATALOGUED: &[(&str, &str)] = &[];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/sparkflash-ops is two levels below the workspace root")
        .to_path_buf()
}

fn rust_sources(directory: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// The string literals in a Rust source, without a parser.
///
/// The scan skips an escaped character inside a literal and steps over a character literal, so a
/// `"""` or a `'\n'` in the source does not begin or end a string early. Comments are not
/// special: a quoted kernel name in a comment is also required to be catalogued, which is a
/// harmless superset of the names the loader closures carry.
fn string_literals(source: &str) -> impl Iterator<Item = &str> {
    let bytes = source.as_bytes();
    let mut literals = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let start = index + 1;
                index = start;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => index += 2,
                        b'"' => break,
                        _ => index += 1,
                    }
                }
                let end = index.min(bytes.len());
                literals.push(&source[start..end]);
                index = end + 1;
            }
            // A character literal is one character or one escape between single quotes; a
            // lifetime such as 'static is not, and its quote is simply stepped over.
            b'\''
                if index + 3 < bytes.len()
                    && bytes[index + 1] == b'\\'
                    && bytes[index + 3] == b'\'' =>
            {
                index += 4;
            }
            b'\'' if index + 2 < bytes.len() && bytes[index + 2] == b'\'' => index += 3,
            _ => index += 1,
        }
    }
    literals.into_iter()
}

#[test]
fn every_forward_named_deepseek_kernel_is_catalogued() {
    let forward = workspace_root()
        .join("crates")
        .join("sparkflash-deepseek-forward")
        .join("src");
    assert!(
        forward.is_dir(),
        "the DeepSeek forward source tree is missing at {}",
        forward.display()
    );

    let exports = Image::Deepseek.exported_symbols();
    let catalogue: BTreeSet<&str> = CATALOGUE
        .iter()
        .flat_map(|entry| entry.symbols.iter().map(|symbol| symbol.name))
        .collect();

    let mut sources = Vec::new();
    rust_sources(&forward, &mut sources);
    assert!(
        !sources.is_empty(),
        "no Rust sources under {}",
        forward.display()
    );

    let mut load_sites = 0usize;
    let mut named: BTreeSet<&str> = BTreeSet::new();
    for path in &sources {
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        if source.contains("load_named") {
            load_sites += 1;
        }
        for literal in string_literals(&source) {
            if let Some(symbol) = exports.iter().copied().find(|export| *export == literal) {
                named.insert(symbol);
            }
        }
    }
    assert!(
        load_sites > 0,
        "the forward no longer resolves any kernel by name; the tripwire is not reading the \
         mechanism it guards"
    );

    let exceptions: BTreeSet<&str> = ALLOWED_UNCATALOGUED.iter().map(|(name, _)| *name).collect();
    for (name, reason) in ALLOWED_UNCATALOGUED {
        assert!(
            !reason.trim().is_empty(),
            "{name} is excepted without a reason"
        );
        assert!(
            named.contains(name),
            "{name} is excepted but the forward no longer names it; delete the line"
        );
    }

    let missing: Vec<&str> = named
        .iter()
        .copied()
        .filter(|name| !catalogue.contains(name) && !exceptions.contains(name))
        .collect();
    assert!(
        missing.is_empty(),
        "{} DeepSeek symbols the forward loads by name are not in CATALOGUE: {missing:?}",
        missing.len()
    );
}
