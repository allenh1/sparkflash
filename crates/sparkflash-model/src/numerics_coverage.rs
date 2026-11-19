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

//! The coverage dump (docs/NUMERICS.md section 5).
//!
//! The `coverage` prerequisite is a dump proving that the declared inputs reach every routed
//! expert of every decoder layer. `scripts/numerics.py coverage` judges one against the model's
//! committed expert layout under `numerics/layouts/`. This module is the writer: a recorder is
//! created with the model's per-layer expert counts, an input is opened, and each routing decision
//! adds one to its layer and expert.
//!
//! It is opt-in. Nothing is recorded unless `SPARKFLASH_NUMERICS_COVERAGE` names an output path;
//! `SPARKFLASH_NUMERICS_MODEL` overrides the dump's model id for a tiny or synthetic fixture.
//! With the switch unset the caller holds `None` and records nothing.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The dump schema this module writes.
pub const SCHEMA: &str = "sparkflash-numerics-coverage/1";

/// The environment switch that names the output path. Unset means off.
pub const ENV: &str = "SPARKFLASH_NUMERICS_COVERAGE";

/// The environment override for the dump's model id.
pub const MODEL_ENV: &str = "SPARKFLASH_NUMERICS_MODEL";

/// One input the recording forward ran.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    /// The input's name, unique within the dump.
    pub name: String,
    /// The number of tokens the input covered.
    pub tokens: usize,
}

/// One decoder layer's hit counts, one per routed expert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer {
    /// The zero-based decoder layer.
    pub layer: usize,
    /// The hit count of each routed expert, in expert order. Empty for a dense layer.
    pub hits: Vec<u64>,
}

/// The coverage document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dump {
    /// [SCHEMA].
    pub schema: String,
    /// The served model id, or [MODEL_ENV]'s override.
    pub model: String,
    /// The inputs that were run.
    pub inputs: Vec<Input>,
    /// Every decoder layer, dense ones included with empty hit lists.
    pub layers: Vec<Layer>,
}

/// A coverage recorder: a [Dump] plus the path it writes when the forward finishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recorder {
    dump: Dump,
    path: PathBuf,
}

/// The FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// The FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A content-addressed input name for a forward's token ids: `input-<16 hex digits>`.
///
/// The digest is FNV-1a over the ids' little-endian bytes. It is deliberately small and stable:
/// `scripts/numerics.py coverage-sweep` computes the same name from `/tokenize`'s ids, so a
/// dump can prove an input was already run without storing the prompt text or the ids.
#[must_use]
pub fn input_name(tokens: &[u32]) -> String {
    let mut digest = FNV_OFFSET;
    for token in tokens {
        for byte in token.to_le_bytes() {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(FNV_PRIME);
        }
    }
    format!("input-{digest:016x}")
}

impl Recorder {
    /// A recorder for `experts` routed experts per decoder layer.
    #[must_use]
    pub fn new(model: impl Into<String>, experts: Vec<usize>, path: impl Into<PathBuf>) -> Self {
        let layers = experts
            .into_iter()
            .enumerate()
            .map(|(layer, experts)| Layer {
                layer,
                hits: vec![0; experts],
            })
            .collect();
        Self {
            dump: Dump {
                schema: SCHEMA.to_owned(),
                model: model.into(),
                inputs: Vec::new(),
                layers,
            },
            path: path.into(),
        }
    }

    /// A recorder from an environment lookup, or `None` when [ENV] is unset.
    ///
    /// The lookup is a parameter rather than a direct `std::env` read so a test can drive the
    /// switch without mutating the process environment.
    #[must_use]
    pub fn from_lookup(
        lookup: impl Fn(&str) -> Option<String>,
        model: &str,
        experts: Vec<usize>,
    ) -> Option<Self> {
        let path = lookup(ENV).filter(|path| !path.is_empty())?;
        let model = lookup(MODEL_ENV)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| model.to_owned());
        let mut recorder = Self::new(model, experts, path);
        // A serving process handles several forwards. Merge the document already at the path so
        // the coverage suite accumulates across them; a dump for another model or layout is not
        // merged, and is replaced by this run's document.
        if let Ok(bytes) = std::fs::read(&recorder.path)
            && let Ok(stored) = serde_json::from_slice::<Dump>(&bytes)
            && stored.schema == SCHEMA
            && stored.model == recorder.dump.model
            && stored.layers.len() == recorder.dump.layers.len()
            && stored
                .layers
                .iter()
                .zip(&recorder.dump.layers)
                .all(|(stored, mine)| {
                    stored.layer == mine.layer && stored.hits.len() == mine.hits.len()
                })
        {
            recorder.dump.inputs = stored.inputs;
            recorder.dump.layers = stored.layers;
        }
        Some(recorder)
    }

    /// A recorder from the process environment, or `None` when the switch is unset.
    #[must_use]
    pub fn from_env(model: &str, experts: Vec<usize>) -> Option<Self> {
        Self::from_lookup(|name| std::env::var(name).ok(), model, experts)
    }

    /// Open an input; every [Self::hit] after this call belongs to it.
    pub fn input(&mut self, name: impl Into<String>, tokens: usize) {
        self.dump.inputs.push(Input {
            name: name.into(),
            tokens,
        });
    }

    /// Open an input named after the document's input count (`input-0`, `input-1`, ...).
    ///
    /// This is the position-addressed form, kept for callers that have only a token count. A
    /// serving forward uses [Self::input_auto_ids] instead: its name is the content, so a sweep
    /// can tell whether it has already run the same input.
    pub fn input_auto(&mut self, tokens: usize) {
        let name = format!("input-{}", self.dump.inputs.len());
        self.input(name, tokens);
    }

    /// Open a content-addressed input for a forward's token ids.
    ///
    /// The name is [input_name], a stable digest of the ids, so the dump itself says which input
    /// it holds: a sweep can skip a prompt without re-running it, and a repeated forward does not
    /// append a second copy of the same input. The routing hits still accumulate, as they do for
    /// every forward.
    pub fn input_auto_ids(&mut self, tokens: &[u32]) {
        let name = input_name(tokens);
        if self.dump.inputs.iter().any(|input| input.name == name) {
            return;
        }
        self.input(name, tokens.len());
    }

    /// Record one routing decision. An out-of-range layer or expert is ignored, never a panic in
    /// a serving forward; the dump's own shape is checked by `scripts/numerics.py coverage`.
    pub fn hit(&mut self, layer: usize, expert: usize) {
        if let Some(layer) = self.dump.layers.get_mut(layer)
            && let Some(hit) = layer.hits.get_mut(expert)
        {
            *hit += 1;
        }
    }

    /// The document as it stands.
    #[must_use]
    pub fn dump(&self) -> &Dump {
        &self.dump
    }

    /// The output path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write the document to [Self::path], creating its parent directory.
    ///
    /// # Errors
    /// An I/O failure, or a serialisation failure (which cannot happen for this shape).
    pub fn write(&self) -> io::Result<()> {
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(&self.dump)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        std::fs::write(&self.path, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_switch_emits_nothing() {
        assert!(Recorder::from_lookup(|_| None, "toy/model", vec![2]).is_none());
        assert!(
            Recorder::from_lookup(|name| (name == ENV).then(String::new), "toy/model", vec![2])
                .is_none(),
            "an empty path is off"
        );
    }

    #[test]
    fn the_switch_names_the_path_and_can_override_the_model() {
        let recorder = Recorder::from_lookup(
            |name| match name {
                ENV => Some("/tmp/coverage.json".to_owned()),
                MODEL_ENV => Some("toy/model".to_owned()),
                _ => None,
            },
            "released/model",
            vec![0, 2],
        )
        .expect("the switch is set");
        assert_eq!(recorder.path(), Path::new("/tmp/coverage.json"));
        assert_eq!(recorder.dump().model, "toy/model");
        assert_eq!(recorder.dump().layers.len(), 2);
        assert_eq!(recorder.dump().layers[0].hits, Vec::<u64>::new());
        assert_eq!(recorder.dump().layers[1].hits, vec![0, 0]);
    }

    #[test]
    fn hits_are_counted_per_layer_and_expert_and_out_of_range_is_ignored() {
        let mut recorder = Recorder::new("toy/model", vec![0, 2], "/tmp/coverage.json");
        recorder.input("probe", 4);
        recorder.hit(1, 0);
        recorder.hit(1, 0);
        recorder.hit(1, 1);
        recorder.hit(9, 0);
        recorder.hit(1, 9);
        assert_eq!(recorder.dump().layers[0].hits, Vec::<u64>::new());
        assert_eq!(recorder.dump().layers[1].hits, vec![2, 1]);
        assert_eq!(recorder.dump().inputs.len(), 1);
        assert_eq!(recorder.dump().inputs[0].name, "probe");
        assert_eq!(recorder.dump().inputs[0].tokens, 4);
    }

    #[test]
    fn a_second_recorder_at_the_same_path_merges_the_suite() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let mut first = Recorder::new("toy/model", vec![2], &path);
        first.input_auto(3);
        first.hit(0, 0);
        first.write().expect("the first dump writes");
        let mut second = Recorder::from_lookup(
            |name| (name == ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/model",
            vec![2],
        )
        .expect("the switch is set");
        assert_eq!(second.dump().inputs.len(), 1, "the first input is merged");
        second.input_auto(4);
        second.hit(0, 1);
        second.write().expect("the second dump writes");
        let text = std::fs::read_to_string(&path).expect("the dump is readable");
        let document: serde_json::Value = serde_json::from_str(&text).expect("the dump is JSON");
        assert_eq!(document["inputs"].as_array().map(Vec::len), Some(2));
        assert_eq!(document["inputs"][0]["name"], "input-0");
        assert_eq!(document["inputs"][1]["name"], "input-1");
        assert_eq!(document["layers"][0]["hits"], serde_json::json!([1, 1]));
    }

    #[test]
    fn a_dump_for_another_model_is_not_merged() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let mut other = Recorder::new("toy/other", vec![2], &path);
        other.input_auto(3);
        other.hit(0, 0);
        other.write().expect("the other dump writes");
        let mine = Recorder::from_lookup(
            |name| (name == ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/model",
            vec![2],
        )
        .expect("the switch is set");
        assert_eq!(mine.dump().model, "toy/model");
        assert!(
            mine.dump().inputs.is_empty(),
            "another model's inputs are not merged"
        );
        assert_eq!(mine.dump().layers[0].hits, vec![0, 0]);
    }

    #[test]
    fn a_content_addressed_name_is_the_stable_digest_of_the_ids() {
        // The Python sweep computes these same names from `/tokenize`'s ids; pin the FNV-1a
        // contract on both sides of the language boundary.
        assert_eq!(input_name(&[1, 2, 3]), "input-fd1f0f4381eb0395");
        assert_eq!(input_name(&[4]), "input-cd3ac65e44f721b1");
        assert_eq!(input_name(&[]), "input-cbf29ce484222325");
    }

    #[test]
    fn a_repeated_input_adds_no_second_entry_but_still_counts_its_hits() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let mut recorder = Recorder::new("toy/model", vec![2], &path);
        recorder.input_auto_ids(&[1, 2]);
        recorder.hit(0, 0);
        recorder.input_auto_ids(&[1, 2]);
        recorder.hit(0, 1);
        recorder.input_auto_ids(&[3]);
        recorder.hit(0, 0);
        assert_eq!(recorder.dump().inputs.len(), 2);
        assert_eq!(recorder.dump().inputs[0].name, "input-c9c28939c99668c6");
        assert_eq!(recorder.dump().inputs[0].tokens, 2);
        assert_eq!(recorder.dump().inputs[1].name, input_name(&[3]));
        assert_eq!(recorder.dump().layers[0].hits, vec![2, 1]);
    }

    #[test]
    fn the_written_document_has_the_schema_and_every_layer() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("nested/coverage.json");
        let mut recorder = Recorder::new("toy/model", vec![0, 2], &path);
        recorder.input("a", 3);
        recorder.hit(1, 0);
        recorder.hit(1, 1);
        recorder.write().expect("the dump writes");
        let text = std::fs::read_to_string(&path).expect("the dump is readable");
        let document: serde_json::Value = serde_json::from_str(&text).expect("the dump is JSON");
        assert_eq!(document["schema"], SCHEMA);
        assert_eq!(document["model"], "toy/model");
        assert_eq!(document["inputs"][0]["name"], "a");
        assert_eq!(document["layers"][0]["hits"], serde_json::json!([]));
        assert_eq!(document["layers"][1]["hits"], serde_json::json!([1, 1]));
    }
}
