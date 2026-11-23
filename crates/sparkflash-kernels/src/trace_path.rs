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

//! The per-process rank used to expand a trace path's `{rank}` placeholder.
//!
//! The four-rank launcher forwards one environment value to every rank. A path that names one
//! file would then be written by four processes, so a diagnostic path can spell `{rank}` and each
//! rank resolves its own file. A path without the placeholder is unchanged.

use std::ffi::OsStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static RANK: AtomicUsize = AtomicUsize::new(0);

/// Sets this process's rank; called by the chain before the first trace or load.
pub(crate) fn set_rank(rank: usize) {
    RANK.store(rank, Ordering::Relaxed);
}

/// Expands `{rank}` in a trace path with this process's rank.
pub(crate) fn expand(path: &OsStr) -> PathBuf {
    let text = path.to_string_lossy();
    if !text.contains("{rank}") {
        return PathBuf::from(path);
    }
    PathBuf::from(text.replace("{rank}", &RANK.load(Ordering::Relaxed).to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_without_the_placeholder_is_unchanged() {
        assert_eq!(
            expand(OsStr::new("/tmp/trace")),
            PathBuf::from("/tmp/trace")
        );
    }

    #[test]
    fn the_placeholder_names_this_rank() {
        set_rank(2);
        assert_eq!(
            expand(OsStr::new("/tmp/trace-r{rank}.jsonl")),
            PathBuf::from("/tmp/trace-r2.jsonl")
        );
    }
}
