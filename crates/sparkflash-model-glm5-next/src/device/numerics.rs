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

//! The monitor dump's GLM plumbing (docs/NUMERICS.md section 5).
//!
//! The writer is [sparkflash_model::numerics_monitors], the same one Qwen3.8's device arm uses.
//! This module is the GLM arm's side of it: the switch is read once when the rank loads, the first
//! forward through the rank is the probe, and [record_probe] turns a layer's routed ids and weights
//! and the sampled rows the device read back into the calls the recorder takes.
//!
//! Off is the default, and the cost of off is one `Option` check a forward and one a layer: no
//! environment read, no formatting and no allocation. The rules themselves — the document's shape,
//! the one-probe rule, the merge and the off switch — are tested here with a synthetic table, so
//! they need no GPU.

use std::io;
use std::path::PathBuf;
use std::sync::Once;

use sparkflash_model::numerics_monitors::{
    ENV as MONITOR_ENV, MODEL_ENV, Probe, Recorder, SAMPLE_ENV,
};

/// The numerics monitor switches a rank reads once, when it loads; unset is the default.
pub(super) struct Switches {
    /// The dump's model id: [MODEL_ENV]'s override or the arm's descriptor.
    model: String,
    /// `SPARKFLASH_NUMERICS_MONITORS`'s path; `None` is off.
    path: Option<PathBuf>,
    /// `SPARKFLASH_NUMERICS_SAMPLE`'s raw value, parsed by the recorder at [Self::begin].
    sample: Option<String>,
    /// The hidden width the sampled indices refer to.
    width: usize,
}

impl Switches {
    /// Read the switches from the process environment once, for a `width`-wide hidden state.
    pub(super) fn from_env(model: &str, width: usize) -> Self {
        Self::from_lookup(|name| std::env::var(name).ok(), model, width)
    }

    /// [Self::from_env] over a lookup, so a test can drive the switch without mutating the process
    /// environment.
    fn from_lookup(lookup: impl Fn(&str) -> Option<String>, model: &str, width: usize) -> Self {
        let model = lookup(MODEL_ENV)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| model.to_owned());
        let path = lookup(MONITOR_ENV)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        let sample = lookup(SAMPLE_ENV).filter(|value| !value.is_empty());
        Self {
            model,
            path,
            sample,
            width,
        }
    }

    /// Whether `SPARKFLASH_NUMERICS_MONITORS` is unset.
    pub(super) fn is_off(&self) -> bool {
        self.path.is_none()
    }

    /// Open this forward's recorder, or `None` when the switch is unset or the recorder refuses
    /// the document at the path.
    ///
    /// The first forward through the rank is the probe; a later forward is a different probe, which
    /// the recorder reports as `AlreadyExists`. That is logged once and the forward serves on
    /// without a monitor dump.
    pub(super) fn begin(&self, tokens: &[u32], position: usize) -> Option<Forward> {
        self.path.as_ref()?;
        let probe = Probe {
            name: format!("position-{position}"),
            prompt_ids: tokens.to_vec(),
            position,
        };
        let recorder = Recorder::from_lookup(
            |name| match name {
                MODEL_ENV => Some(self.model.clone()),
                MONITOR_ENV => self
                    .path
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
                SAMPLE_ENV => self.sample.clone(),
                _ => None,
            },
            &self.model,
            probe,
            self.width,
        );
        match recorder {
            Ok(recorder) => recorder.map(|recorder| Forward { recorder }),
            Err(error) => {
                // A diagnostic must never fail a request: say once why the dump is not happening
                // and keep serving.
                warn(&WARNED, &error);
                None
            }
        }
    }
}

/// One layer's read-back: the router's routed ids and weights, the layer's combined row and
/// each slot's row as its down projection wrote it.
pub(super) struct ProbeRow {
    pub(super) routed: Vec<(usize, f32)>,
    pub(super) values: Vec<f32>,
    pub(super) experts: Vec<Vec<f32>>,
}

/// This forward's recorder, opened by [Switches::begin] and written when the step ends.
pub(super) struct Forward {
    recorder: Recorder,
}

impl Forward {
    /// Write the document this forward filled. A write that fails is logged once and serving goes
    /// on; the dump is a diagnostic, not part of the answer.
    pub(super) fn finish(mut self) {
        if let Err(error) = self.recorder.write() {
            warn(&WARNED, &error);
        }
    }
}

/// Record one layer of the probe: the layer's combined output row and each routed expert's body.
///
/// `routed` is the router's `(expert id, combine weight)` in slot order, `values` the layer's
/// row and `experts[slot]` the slot's row as the expert's down projection wrote it,
/// `weight * body`. The format compares activations, so an expert is recorded as its body, the
/// row divided by its weight. A slot with a zero or non-finite weight, or with no sampled row, is
/// not recorded. With no recorder this is one `Option` check; the divisions and the recorder's
/// allocation only run when the switch names a path.
pub(super) fn record_probe(
    monitors: &mut Option<Forward>,
    layer: usize,
    routed: &[(usize, f32)],
    values: &[f32],
    experts: &[Vec<f32>],
) {
    let Some(forward) = monitors.as_mut() else {
        return;
    };
    for (slot, (expert, weight)) in routed.iter().enumerate() {
        if *weight == 0.0 || !weight.is_finite() {
            continue;
        }
        let Some(body) = experts.get(slot) else {
            continue;
        };
        let body: Vec<f32> = body.iter().map(|value| value / weight).collect();
        forward.recorder.record_expert(layer, *expert, &body);
    }
    forward.recorder.record_layer(layer, values);
}

/// Decode little-endian f32 values, as a device read-back gives them.
pub(super) fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Log a monitor read-back's first failure.
pub(super) fn warn_read_back(error: &str) {
    static READ_BACK_WARNED: Once = Once::new();
    READ_BACK_WARNED.call_once(|| {
        eprintln!("sparkflash numerics: monitor read-back: {error}; not recording this forward");
    });
}

/// Log the monitor dump's first failure.
fn warn(once: &Once, error: &io::Error) {
    once.call_once(|| {
        eprintln!("sparkflash numerics: monitor dump: {error}; not recording it");
    });
}

static WARNED: Once = Once::new();

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::path::Path;

    fn read(path: &Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).expect("the dump is readable"))
            .expect("the dump is JSON")
    }

    fn values(width: usize) -> Vec<f32> {
        (0..width).map(|index| index as f32 + 0.5).collect()
    }

    fn switches(path: &Path, sample: Option<&str>) -> Switches {
        Switches::from_lookup(
            |name| match name {
                MONITOR_ENV => Some(path.to_string_lossy().into_owned()),
                SAMPLE_ENV => sample.map(str::to_owned),
                _ => None,
            },
            "toy/glm",
            16,
        )
    }

    fn probe(
        switches: &Switches,
        tokens: &[u32],
        position: usize,
        layer: usize,
        routed: &[(usize, f32)],
    ) -> Option<Forward> {
        let mut forward = Some(
            switches
                .begin(tokens, position)
                .expect("the switch is set and the document is fresh"),
        );
        record_probe(
            &mut forward,
            layer,
            routed,
            &values(16),
            &routed.iter().map(|_| values(16)).collect::<Vec<_>>(),
        );
        forward
    }

    #[test]
    fn an_unset_switch_opens_no_recorder() {
        let switches = Switches::from_lookup(|_| None, "toy/glm", 16);
        assert!(switches.is_off());
        assert!(switches.begin(&[1, 2], 0).is_none());
        record_probe(&mut None, 0, &[(1, 0.5)], &values(16), &[values(16)]);
    }

    #[test]
    fn the_document_has_the_shape_the_judge_reads() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("nested/monitors.json");
        let switches = switches(&path, Some("4"));
        assert!(!switches.is_off());
        let mut forward = probe(&switches, &[1, 2, 3], 0, 3, &[(5, 0.25), (9, 0.5)]);
        forward.take().expect("the probe").finish();

        let document = read(&path);
        assert_eq!(
            document["schema"],
            sparkflash_model::numerics_monitors::SCHEMA
        );
        assert_eq!(document["model"], "toy/glm");
        assert_eq!(document["probe"]["name"], "position-0");
        assert_eq!(document["probe"]["prompt_ids"], json!([1, 2, 3]));
        assert_eq!(document["probe"]["position"], 0);
        assert_eq!(document["sample"]["width"], 16);
        assert_eq!(
            document["sample"]["indices"],
            json!([0, 4, 8, 12]),
            "SPARKFLASH_NUMERICS_SAMPLE=4 sets the sample count"
        );
        assert_eq!(
            document["layers"][0]["values"],
            json!([0.5, 4.5, 8.5, 12.5])
        );
        assert_eq!(document["layers"][0]["layer"], 3);
        assert_eq!(
            document["layers"][0]["experts"].as_array().map(Vec::len),
            Some(2)
        );
        assert_eq!(document["layers"][0]["experts"][0]["expert"], 5);
        assert_eq!(
            document["layers"][0]["experts"][0]["values"],
            json!([2.0, 18.0, 34.0, 50.0]),
            "a slot's row is weight * body, so the recorded value is the body"
        );
    }

    #[test]
    fn the_model_id_prefers_the_environment_over_the_descriptor() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = Switches::from_lookup(
            |name| match name {
                MODEL_ENV => Some("fixture/glm".to_owned()),
                MONITOR_ENV => Some(path.to_string_lossy().into_owned()),
                _ => None,
            },
            "toy/glm",
            16,
        );
        let mut forward = probe(&switches, &[1], 0, 0, &[(1, 1.0)]);
        forward.take().expect("the probe").finish();
        assert_eq!(read(&path)["model"], "fixture/glm");
    }

    #[test]
    fn a_later_forward_is_a_different_probe_and_records_nothing() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = switches(&path, None);
        let mut first = probe(&switches, &[1], 0, 0, &[(0, 1.0)]);
        first.take().expect("the probe").finish();
        // A second forward is a different probe; the recorder reports AlreadyExists, which begin
        // logs once and drops, so this forward has no monitor dump.
        assert!(switches.begin(&[2], 1).is_none());
        let document = read(&path);
        assert_eq!(document["probe"]["prompt_ids"], json!([1]));
        assert_eq!(document["probe"]["position"], 0);
    }

    #[test]
    fn the_same_probe_merges_the_document_on_disk() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = switches(&path, None);
        let mut first = probe(&switches, &[1], 0, 0, &[(0, 1.0)]);
        first.take().expect("the probe").finish();
        let mut second = probe(&switches, &[1], 0, 1, &[(2, 1.0)]);
        second.take().expect("the same probe merges").finish();
        let document = read(&path);
        let layers = document["layers"].as_array().expect("the layers");
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0]["layer"], 0);
        assert_eq!(layers[1]["layer"], 1);
    }

    #[test]
    fn a_zero_weight_slot_is_not_recorded() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = switches(&path, None);
        let mut forward = Some(
            switches
                .begin(&[1], 0)
                .expect("the switch is set and the document is fresh"),
        );
        let routed = [(3, 0.0), (4, 0.5)];
        record_probe(
            &mut forward,
            0,
            &routed,
            &values(16),
            &[values(16), values(16)],
        );
        forward.take().expect("the probe").finish();
        let document = read(&path);
        let experts = document["layers"][0]["experts"]
            .as_array()
            .expect("the experts");
        assert_eq!(experts.len(), 1);
        assert_eq!(experts[0]["expert"], 4);
    }

    #[test]
    fn a_write_failure_is_logged_and_does_not_fail_serving() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = switches(&path, None);
        let mut forward = probe(&switches, &[1], 0, 0, &[(0, 1.0)]);
        // The path a directory is now: the recorder cannot write a file there.
        std::fs::create_dir(&path).expect("a directory where the dump would go");
        forward.take().expect("the probe").finish();
    }
}
