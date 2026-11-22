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

//! The open manifest: one line for every model value, bound tensor and loaded kernel.
//!
//! With `SPARKFLASH_OPEN_MANIFEST=<file>` set, the DeepSeek forward appends one JSON object per
//! record to that file. The file is per process: a four-rank run gives each rank its own path, so
//! two ranks never write one file. Lines are deduplicated by their JSON text; a later comparison
//! sorts them, because loads can happen on more than one thread.
//!
//! The three record kinds are:
//!
//! * `{"kind":"model","name":...,"value":...}` - a value the forward reads;
//! * `{"kind":"tensor","name":...,"shape":...,"format":...,"rank_slice":...,...}` - a tensor the
//!   rank binds, with the checkpoint's stored shape and the rank's slice;
//! * `{"kind":"kernel","cubin_sha256":...,"symbol":...}` - a kernel entry the driver loads. The
//!   cubin is named by its sha256, which is the same measurement the kernel baseline pins.
//!
//! Unset, every call is one `OnceLock` read and no file is opened. Set, recording still changes
//! no launch: it opens a file, hashes a cubin once per symbol and writes a line.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

/// The environment variable naming the manifest file.
pub const ENV: &str = "SPARKFLASH_OPEN_MANIFEST";

struct Writer {
    file: File,
    seen: HashSet<String>,
}

fn manifest() -> Option<&'static Mutex<Writer>> {
    static MANIFEST: OnceLock<Option<Mutex<Writer>>> = OnceLock::new();
    MANIFEST
        .get_or_init(|| {
            let path = std::env::var_os(ENV).filter(|path| !path.is_empty())?;
            let path = crate::trace_path::expand(&path);
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|error| {
                    panic!(
                        "{ENV}={} cannot be opened for the open manifest: {error}",
                        path.display()
                    )
                });
            Some(Mutex::new(Writer {
                file,
                seen: HashSet::new(),
            }))
        })
        .as_ref()
}

/// Sets the process rank a `{rank}` placeholder in [ENV] expands to.
pub fn set_rank(rank: usize) {
    crate::trace_path::set_rank(rank);
}

/// Whether this process records an open manifest.
#[must_use]
pub fn enabled() -> bool {
    manifest().is_some()
}

/// Appends one record, once per line, when the manifest is enabled.
///
/// # Panics
/// A manifest file that stops taking writes: a manifest with a hole compares wrong.
pub fn record(value: &Value) {
    let Some(manifest) = manifest() else {
        return;
    };
    let mut line = serde_json::to_string(value)
        .expect("an open-manifest record is a JSON value and always serialises");
    line.push('\n');
    let mut writer = manifest
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !writer.seen.insert(line.clone()) {
        return;
    }
    writer
        .file
        .write_all(line.as_bytes())
        .unwrap_or_else(|error| panic!("{ENV}: an open-manifest line did not write: {error}"));
}

/// Records a kernel the driver resolved, when the manifest is enabled.
///
/// Called for cache hits as well as loads: the manifest states the entries the forward asked the
/// driver for, not only the ones the driver had to compile.
pub(crate) fn note_kernel(cubin: &[u8], symbol: &str) {
    if !enabled() {
        return;
    }
    record(&json!({
        "kind": "kernel",
        "cubin_sha256": sha256_hex(cubin),
        "cubin_bytes": cubin.len(),
        "symbol": symbol,
    }));
}

/// Lower-case hex sha256 of one cubin image.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    digest
        .iter()
        .fold(String::with_capacity(64), |mut text, byte| {
            let _ = write!(text, "{byte:02x}");
            text
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cubin_hash_is_lower_case_hex() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_record_is_one_json_line() {
        let value = json!({"kind": "kernel", "symbol": "k", "cubin_sha256": "00"});
        let line = serde_json::to_string(&value).expect("record");
        assert_eq!(
            line,
            r#"{"kind":"kernel","symbol":"k","cubin_sha256":"00"}"#
        );
    }
}
