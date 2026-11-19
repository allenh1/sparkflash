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

//! The monitor dump (docs/NUMERICS.md section 5).
//!
//! The `monitors` prerequisite compares a candidate forward with the reference forward per
//! decoder layer and per routed expert. This module is the writer side: a recorder samples a fixed
//! set of hidden indices from each layer's output and each selected expert's output, and
//! `scripts/numerics.py monitors` computes the relative L2 error between a reference dump and a
//! candidate dump.
//!
//! It is opt-in. Nothing is recorded unless `SPARKFLASH_NUMERICS_MONITORS` names an output path.
//! `SPARKFLASH_NUMERICS_MODEL` overrides the model id for a tiny fixture, and
//! `SPARKFLASH_NUMERICS_SAMPLE` sets how many evenly spaced hidden indices the document
//! declares (default 8). With the switch unset the caller holds `None` and records nothing.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The dump schema this module writes.
pub const SCHEMA: &str = "sparkflash-numerics-monitors/1";

/// The environment switch that names the output path. Unset means off.
pub const ENV: &str = "SPARKFLASH_NUMERICS_MONITORS";

/// The environment override for the dump's model id.
pub const MODEL_ENV: &str = "SPARKFLASH_NUMERICS_MODEL";

/// The environment switch that sets the sample count.
pub const SAMPLE_ENV: &str = "SPARKFLASH_NUMERICS_SAMPLE";

/// The default number of evenly spaced hidden indices.
pub const DEFAULT_SAMPLE: usize = 8;

/// The probe both dumps must share.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Probe {
    /// The input's name.
    pub name: String,
    /// The input's token ids.
    pub prompt_ids: Vec<u32>,
    /// The absolute position of the first row.
    pub position: usize,
}

/// The fixed hidden-index sample both dumps must share.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sample {
    /// The hidden width the indices refer to.
    pub width: usize,
    /// The sampled positions, strictly increasing and inside the width.
    pub indices: Vec<usize>,
}

/// One routed expert's sampled output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expert {
    /// The expert id in the layer's layout.
    pub expert: usize,
    /// One value per [Sample::indices] entry.
    pub values: Vec<f32>,
}

/// One decoder layer's sampled output and routed experts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// The zero-based decoder layer.
    pub layer: usize,
    /// One value per [Sample::indices] entry.
    pub values: Vec<f32>,
    /// The experts the probe routed to.
    pub experts: Vec<Expert>,
}

/// The monitor document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dump {
    /// [SCHEMA].
    pub schema: String,
    /// The served model id, or [MODEL_ENV]'s override.
    pub model: String,
    /// The input this dump measured.
    pub probe: Probe,
    /// The hidden-index sample.
    pub sample: Sample,
    /// Every decoder layer the probe reached.
    pub layers: Vec<Layer>,
}

/// A monitor recorder: a [Dump] plus the path it writes when the forward finishes.
#[derive(Debug, Clone, PartialEq)]
pub struct Recorder {
    dump: Dump,
    path: PathBuf,
    conflict: Option<String>,
}

impl Recorder {
    /// A recorder for a probe, hidden width and explicit sample indices.
    ///
    /// # Errors
    /// A zero width, an empty or non-increasing sample, an index outside the width, or an invalid
    /// probe.
    pub fn new(
        model: impl Into<String>,
        probe: Probe,
        width: usize,
        indices: Vec<usize>,
        path: impl Into<PathBuf>,
    ) -> io::Result<Self> {
        if width == 0 {
            return Err(invalid("the hidden width is zero"));
        }
        if probe.name.is_empty() {
            return Err(invalid("the probe has no name"));
        }
        if indices.is_empty() {
            return Err(invalid("the sample holds no indices"));
        }
        if indices
            .iter()
            .zip(indices.iter().skip(1))
            .any(|(left, right)| left >= right)
        {
            return Err(invalid("the sample indices are not strictly increasing"));
        }
        if indices.last().is_some_and(|last| *last >= width) {
            return Err(invalid("a sample index is outside the hidden width"));
        }
        Ok(Self {
            dump: Dump {
                schema: SCHEMA.to_owned(),
                model: model.into(),
                probe,
                sample: Sample { width, indices },
                layers: Vec::new(),
            },
            path: path.into(),
            conflict: None,
        })
    }

    /// A recorder from an environment lookup, or `None` when [ENV] is unset.
    ///
    /// The lookup is a parameter rather than a direct `std::env` read so a test can drive the
    /// switch without mutating the process environment. An existing document at the path is merged
    /// only when its model, probe and sample match; a different one is refused.
    ///
    /// # Errors
    /// The switch names a bad sample count, the existing document does not parse, or it names a
    /// different model, probe or sample.
    pub fn from_lookup(
        lookup: impl Fn(&str) -> Option<String>,
        model: &str,
        probe: Probe,
        width: usize,
    ) -> io::Result<Option<Self>> {
        let path = lookup(ENV).filter(|path| !path.is_empty());
        let Some(path) = path else {
            return Ok(None);
        };
        let model = lookup(MODEL_ENV)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| model.to_owned());
        let count = match lookup(SAMPLE_ENV).filter(|value| !value.is_empty()) {
            Some(value) => value.parse::<usize>().map_err(|_| {
                invalid(format!("{SAMPLE_ENV}={value:?} is not a positive integer"))
            })?,
            None => DEFAULT_SAMPLE,
        };
        if count == 0 || width == 0 {
            return Err(invalid(format!(
                "sample count {count} or width {width} is zero"
            )));
        }
        let indices = sample_indices(width, count);
        let mut recorder = Self::new(model, probe, width, indices, path)?;
        match std::fs::read(&recorder.path) {
            Ok(bytes) => {
                let stored: Dump = serde_json::from_slice(&bytes)
                    .map_err(|error| invalid(format!("{}: {error}", recorder.path.display())))?;
                if stored.schema != SCHEMA {
                    return Err(invalid(format!(
                        "{}: schema {:?} is not {SCHEMA}",
                        recorder.path.display(),
                        stored.schema
                    )));
                }
                if stored.model != recorder.dump.model {
                    return Err(invalid(format!(
                        "{}: the dump names {:?}, but this run names {:?}",
                        recorder.path.display(),
                        stored.model,
                        recorder.dump.model
                    )));
                }
                if stored.probe != recorder.dump.probe {
                    return Err(already(format!(
                        "{}: the dump was measured on a different probe; a monitor dump is one probe",
                        recorder.path.display()
                    )));
                }
                if stored.sample != recorder.dump.sample {
                    return Err(already(format!(
                        "{}: the dump sampled different indices or a different width",
                        recorder.path.display()
                    )));
                }
                recorder.dump.layers = stored.layers;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
        Ok(Some(recorder))
    }

    /// A recorder from the process environment, or `None` when the switch is unset.
    ///
    /// # Errors
    /// As [Self::from_lookup].
    pub fn from_env(model: &str, probe: Probe, width: usize) -> io::Result<Option<Self>> {
        Self::from_lookup(|name| std::env::var(name).ok(), model, probe, width)
    }

    /// Record a layer's full output, sampled at the declared indices.
    pub fn record_layer(&mut self, layer: usize, values: &[f32]) {
        let sampled = match self.sample(values, &format!("layer {layer}")) {
            Ok(values) => values,
            Err(message) => return self.fail(message),
        };
        if let Some(entry) = self
            .dump
            .layers
            .iter_mut()
            .find(|entry| entry.layer == layer)
        {
            if entry.values.is_empty() {
                entry.values = sampled;
            } else if entry.values != sampled {
                self.fail(format!(
                    "layer {layer}: the same probe produced different values"
                ));
            }
        } else {
            self.dump.layers.push(Layer {
                layer,
                values: sampled,
                experts: Vec::new(),
            });
        }
    }

    /// Record one routed expert's full output, sampled at the declared indices.
    pub fn record_expert(&mut self, layer: usize, expert: usize, values: &[f32]) {
        let sampled = match self.sample(values, &format!("layer {layer} expert {expert}")) {
            Ok(values) => values,
            Err(message) => return self.fail(message),
        };
        let entry = match self
            .dump
            .layers
            .iter_mut()
            .find(|entry| entry.layer == layer)
        {
            Some(entry) => entry,
            None => {
                self.dump.layers.push(Layer {
                    layer,
                    values: Vec::new(),
                    experts: Vec::new(),
                });
                self.dump
                    .layers
                    .last_mut()
                    .expect("the layer was just pushed")
            }
        };
        if let Some(existing) = entry
            .experts
            .iter_mut()
            .find(|entry| entry.expert == expert)
        {
            if existing.values != sampled {
                self.fail(format!(
                    "layer {layer} expert {expert}: the same probe produced different values"
                ));
            }
        } else {
            entry.experts.push(Expert {
                expert,
                values: sampled,
            });
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
    /// A conflict recorded during the forward, a layer or expert with the wrong sample length or no
    /// expert, or an I/O failure.
    pub fn write(&mut self) -> io::Result<()> {
        if let Some(conflict) = self.conflict.take() {
            return Err(invalid(conflict));
        }
        if self.dump.layers.is_empty() {
            return Err(invalid("the monitor dump holds no layers"));
        }
        for layer in &self.dump.layers {
            if layer.values.len() != self.dump.sample.indices.len() {
                return Err(invalid(format!(
                    "layer {}: {} values for {} sampled indices",
                    layer.layer,
                    layer.values.len(),
                    self.dump.sample.indices.len()
                )));
            }
            if layer.experts.is_empty() {
                return Err(invalid(format!("layer {}: no routed experts", layer.layer)));
            }
            for expert in &layer.experts {
                if expert.values.len() != self.dump.sample.indices.len() {
                    return Err(invalid(format!(
                        "layer {} expert {}: {} values for {} sampled indices",
                        layer.layer,
                        expert.expert,
                        expert.values.len(),
                        self.dump.sample.indices.len()
                    )));
                }
            }
        }
        self.dump.layers.sort_by_key(|left| left.layer);
        for layer in &mut self.dump.layers {
            layer.experts.sort_by_key(|left| left.expert);
        }
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(&self.dump)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        std::fs::write(&self.path, bytes)
    }

    fn sample(&self, values: &[f32], origin: &str) -> Result<Vec<f32>, String> {
        if values.len() != self.dump.sample.width {
            return Err(format!(
                "{origin}: {} values for a declared width of {}",
                values.len(),
                self.dump.sample.width
            ));
        }
        if let Some((index, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "{origin}: value {value} at index {index} is not finite"
            ));
        }
        Ok(self
            .dump
            .sample
            .indices
            .iter()
            .map(|index| values[*index])
            .collect())
    }

    fn fail(&mut self, message: String) {
        if self.conflict.is_none() {
            self.conflict = Some(message);
        }
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn already(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::AlreadyExists, message.into())
}

fn sample_indices(width: usize, count: usize) -> Vec<usize> {
    let count = count.clamp(1, width.max(1));
    (0..count).map(|index| index * width / count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe() -> Probe {
        Probe {
            name: "probe".to_owned(),
            prompt_ids: vec![1, 2],
            position: 0,
        }
    }

    fn recorder(path: &Path) -> Recorder {
        let mut recorder =
            Recorder::new("toy/model", probe(), 2, vec![0, 1], path).expect("the sample is valid");
        recorder.record_layer(0, &[1.0, 1.0]);
        recorder.record_expert(0, 0, &[1.0, 1.0]);
        recorder.record_expert(0, 1, &[1.0, 1.0]);
        recorder
    }

    #[test]
    fn an_unset_switch_emits_nothing() {
        assert!(
            Recorder::from_lookup(|_| None, "toy/model", probe(), 2)
                .expect("unset is not an error")
                .is_none()
        );
        assert!(
            Recorder::from_lookup(
                |name| (name == ENV).then(String::new),
                "toy/model",
                probe(),
                2
            )
            .expect("an empty path is off")
            .is_none()
        );
    }

    #[test]
    fn the_written_document_has_the_shape_the_judge_reads() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("nested/monitors.json");
        let mut recorder = recorder(&path);
        recorder.write().expect("the dump writes");
        let text = std::fs::read_to_string(&path).expect("the dump is readable");
        let document: serde_json::Value = serde_json::from_str(&text).expect("the dump is JSON");
        assert_eq!(document["schema"], SCHEMA);
        assert_eq!(document["model"], "toy/model");
        assert_eq!(document["probe"]["prompt_ids"], serde_json::json!([1, 2]));
        assert_eq!(document["sample"]["width"], 2);
        assert_eq!(document["sample"]["indices"], serde_json::json!([0, 1]));
        assert_eq!(
            document["layers"][0]["values"],
            serde_json::json!([1.0, 1.0])
        );
        assert_eq!(document["layers"][0]["experts"][1]["expert"], 1);
    }

    #[test]
    fn the_same_probe_merges_and_a_different_one_is_refused() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        recorder(&path).write().expect("the first dump writes");
        let merged = Recorder::from_lookup(
            |name| (name == ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/model",
            probe(),
            2,
        )
        .expect("the switch is set")
        .expect("the document parses");
        assert_eq!(merged.dump().layers, recorder(&path).dump().layers);
        let different = Probe {
            name: "other".to_owned(),
            ..probe()
        };
        let error = Recorder::from_lookup(
            |name| (name == ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/model",
            different,
            2,
        )
        .expect_err("a different probe is refused");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert!(error.to_string().contains("different probe"), "{error}");
        let error = Recorder::from_lookup(
            |name| (name == ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/model",
            probe(),
            3,
        )
        .expect_err("a different sample is refused");
        assert!(error.to_string().contains("different indices"), "{error}");
    }

    #[test]
    fn a_conflicting_value_is_refused_at_write() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        recorder(&path).write().expect("the first dump writes");
        let mut merged = Recorder::from_lookup(
            |name| (name == ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/model",
            probe(),
            2,
        )
        .expect("the switch is set")
        .expect("the document parses");
        merged.record_layer(0, &[1.0, 2.0]);
        let error = merged.write().expect_err("a changed value is refused");
        assert!(error.to_string().contains("layer 0"), "{error}");
    }

    fn python_monitors(reference: &Path, candidate: &Path, bar: &str) -> std::process::Output {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/numerics.py");
        std::process::Command::new("python3")
            .arg(script)
            .arg("monitors")
            .arg("--reference")
            .arg(reference)
            .arg("--candidate")
            .arg(candidate)
            .arg("--layer-bar")
            .arg(bar)
            .arg("--expert-bar")
            .arg(bar)
            .output()
            .expect("python3 runs scripts/numerics.py")
    }

    #[test]
    fn the_dump_round_trips_through_the_python_monitor_command() {
        if std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("python3 is unavailable; skipping the cross-language monitor check");
            return;
        }
        let directory = tempfile::tempdir().expect("a temporary directory");
        let reference = directory.path().join("reference.json");
        let candidate = directory.path().join("candidate.json");
        recorder(&reference).write().expect("the reference writes");
        std::fs::copy(&reference, &candidate).expect("the candidate copies");
        let clean = python_monitors(&reference, &candidate, "0.5");
        assert!(
            clean.status.success(),
            "the same dump must judge clean: {}",
            String::from_utf8_lossy(&clean.stderr)
        );

        let read = || {
            serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&candidate).expect("the candidate reads"),
            )
            .expect("the candidate is JSON")
        };

        // A layer-only change fails the layer bar, not an expert bar.
        let mut document = read();
        document["layers"][0]["values"][0] = serde_json::json!(2.0);
        std::fs::write(
            &candidate,
            serde_json::to_string_pretty(&document).expect("the layer change serialises"),
        )
        .expect("the layer change writes");
        let layer_only = python_monitors(&reference, &candidate, "0.5");
        assert!(!layer_only.status.success(), "a changed layer must fail");
        let stdout = String::from_utf8_lossy(&layer_only.stdout);
        assert!(stdout.contains("layer 0:"), "{stdout}");
        assert!(!stdout.contains("expert 0:"), "{stdout}");

        // An expert-only change fails the expert bar while the layer stays within its bar.
        std::fs::copy(&reference, &candidate).expect("the candidate resets");
        let mut document = read();
        document["layers"][0]["experts"][0]["values"][0] = serde_json::json!(2.0);
        std::fs::write(
            &candidate,
            serde_json::to_string_pretty(&document).expect("the expert change serialises"),
        )
        .expect("the expert change writes");
        let expert_only = python_monitors(&reference, &candidate, "0.5");
        assert!(!expert_only.status.success(), "a changed expert must fail");
        let stdout = String::from_utf8_lossy(&expert_only.stdout);
        assert!(stdout.contains("expert 0:"), "{stdout}");
        assert!(!stdout.contains("layer 0:"), "{stdout}");
    }
}
