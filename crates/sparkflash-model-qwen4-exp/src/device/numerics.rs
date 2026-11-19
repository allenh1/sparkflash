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

//! The numerics dumps' device plumbing (docs/NUMERICS.md section 5).
//!
//! The writers are [sparkflash_model::numerics_coverage] and
//! [sparkflash_model::numerics_monitors], and the Qwen host reference wires them in
//! [crate::forward::moe_forward]. This module is the same wiring for the served pack: the
//! switches are read once when a session is made, each forward opens its recorders, and
//! [record_routing] and [record_probe] turn a layer's chosen experts and sampled values into the
//! calls the recorders take. The device read-backs that feed those two functions live beside the
//! kernels (DeviceSequence::record_routing and DeviceSequence::record_probe), so the rules
//! themselves are tested here with a synthetic table.
//!
//! Off is the default, and the cost of off is one Option check a forward and one a layer: no
//! environment read, no formatting and no allocation.

use std::io;
use std::path::PathBuf;
use std::sync::Once;

use sparkflash_model::numerics_coverage::{
    ENV as COVERAGE_ENV, MODEL_ENV, Recorder as CoverageRecorder,
};
use sparkflash_model::numerics_monitors::{
    ENV as MONITOR_ENV, Probe as MonitorProbe, Recorder as MonitorRecorder, SAMPLE_ENV,
};

/// The numerics switches a session reads once, when it is made; both off is the default.
pub(crate) struct Switches {
    coverage: Option<CoverageSwitch>,
    monitors: Option<MonitorSwitch>,
}

/// What SPARKFLASH_NUMERICS_COVERAGE asked for: a path, the dump's model id and the model's
/// per-layer routed-expert counts.
struct CoverageSwitch {
    path: PathBuf,
    model: String,
    experts: Vec<usize>,
}

/// What SPARKFLASH_NUMERICS_MONITORS asked for: a path, the dump's model id, the hidden width
/// and the raw SPARKFLASH_NUMERICS_SAMPLE value the recorder parses at [Switches::begin].
struct MonitorSwitch {
    path: PathBuf,
    model: String,
    width: usize,
    sample: Option<String>,
}

impl Switches {
    /// Read the switches from the process environment once, for a model with `experts` routed
    /// experts per decoder layer and a `width`-wide hidden state.
    pub(crate) fn from_env(model: &str, experts: Vec<usize>, width: usize) -> Self {
        Self::from_lookup(|name| std::env::var(name).ok(), model, experts, width)
    }

    /// [Self::from_env] over a lookup, so a test can drive the switches without mutating the
    /// process environment.
    fn from_lookup(
        lookup: impl Fn(&str) -> Option<String>,
        model: &str,
        experts: Vec<usize>,
        width: usize,
    ) -> Self {
        let model = lookup(MODEL_ENV)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| model.to_owned());
        let coverage_path = lookup(COVERAGE_ENV).filter(|path| !path.is_empty());
        let monitor_path = lookup(MONITOR_ENV).filter(|path| !path.is_empty());
        let coverage = coverage_path.map(|path| CoverageSwitch {
            path: path.into(),
            model: model.clone(),
            experts,
        });
        let monitors = monitor_path.map(|path| MonitorSwitch {
            path: path.into(),
            model,
            width,
            sample: lookup(SAMPLE_ENV).filter(|value| !value.is_empty()),
        });
        Self { coverage, monitors }
    }

    /// Whether both switches are unset.
    pub(crate) fn is_off(&self) -> bool {
        self.coverage.is_none() && self.monitors.is_none()
    }

    /// Open the recorders for one forward over `tokens` at `position`, or None when both
    /// switches are off (or neither recorder could be opened).
    ///
    /// The coverage recorder opens one content-addressed input, `input-<digest>`, named from
    /// this forward's token ids, so a serving process accumulates a suite across forwards exactly
    /// as the host reference does and a repeated input adds no second entry. The monitor recorder
    /// takes this forward as the probe; a later forward is a different probe, which the recorder
    /// reports as AlreadyExists. That is logged once and this forward serves on without a monitor
    /// dump.
    pub(crate) fn begin(&self, tokens: &[u32], position: usize) -> Option<Forward> {
        if self.is_off() {
            return None;
        }
        let coverage = self.coverage.as_ref().and_then(|switch| {
            let mut recorder = CoverageRecorder::from_lookup(
                |name| match name {
                    MODEL_ENV => Some(switch.model.clone()),
                    COVERAGE_ENV => Some(switch.path.to_string_lossy().into_owned()),
                    _ => None,
                },
                &switch.model,
                switch.experts.clone(),
            );
            if let Some(recorder) = recorder.as_mut() {
                recorder.input_auto_ids(tokens);
            }
            recorder
        });
        let monitors = self.monitors.as_ref().and_then(|switch| {
            let probe = MonitorProbe {
                name: format!("position-{position}"),
                prompt_ids: tokens.to_vec(),
                position,
            };
            let recorder = MonitorRecorder::from_lookup(
                |name| match name {
                    MODEL_ENV => Some(switch.model.clone()),
                    MONITOR_ENV => Some(switch.path.to_string_lossy().into_owned()),
                    SAMPLE_ENV => switch.sample.clone(),
                    _ => None,
                },
                &switch.model,
                probe,
                switch.width,
            );
            match recorder {
                Ok(recorder) => recorder,
                Err(error) => {
                    // A diagnostic must never fail a request: say once why the dump is not
                    // happening and keep serving.
                    warn(&MONITOR_WARNED, "monitor", &error);
                    None
                }
            }
        });
        if coverage.is_none() && monitors.is_none() {
            return None;
        }
        Some(Forward { coverage, monitors })
    }
}

/// The recorders of one forward, opened by [Switches::begin] and written by [Forward::finish].
pub(crate) struct Forward {
    coverage: Option<CoverageRecorder>,
    monitors: Option<MonitorRecorder>,
}

impl Forward {
    /// Whether this forward counts routing decisions.
    pub(crate) fn coverage_enabled(&self) -> bool {
        self.coverage.is_some()
    }

    /// Whether this forward samples the monitor probe.
    pub(crate) fn monitors_enabled(&self) -> bool {
        self.monitors.is_some()
    }

    /// Count one layer's chosen experts.
    pub(crate) fn routing(&mut self, layer: usize, selected: &[usize]) {
        record_routing(&mut self.coverage, layer, selected);
    }

    /// Sample one probe row: the layer's output and each routed expert's.
    pub(crate) fn probe(&mut self, layer: usize, values: &[f32], experts: &[(usize, Vec<f32>)]) {
        record_probe(&mut self.monitors, layer, values, experts);
    }

    /// Write the dumps this forward filled. A write that fails is logged once and serving goes
    /// on; the dumps are diagnostics, not part of the answer.
    pub(crate) fn finish(self) {
        if let Some(recorder) = self.coverage
            && let Err(error) = recorder.write()
        {
            warn(&COVERAGE_WARNED, "coverage", &error);
        }
        if let Some(mut recorder) = self.monitors
            && let Err(error) = recorder.write()
        {
            warn(&MONITOR_WARNED, "monitor", &error);
        }
    }
}

/// Count one layer's routing decisions in the coverage dump: one [CoverageRecorder::hit] a
/// selected expert. With no recorder this is one Option check.
pub(crate) fn record_routing(
    coverage: &mut Option<CoverageRecorder>,
    layer: usize,
    selected: &[usize],
) {
    let Some(recorder) = coverage.as_mut() else {
        return;
    };
    for expert in selected {
        recorder.hit(layer, *expert);
    }
}

/// Sample one probe row in the monitor dump: the layer's output and every routed expert's,
/// sampled at the document's fixed hidden indices. With no recorder this is one Option check.
pub(crate) fn record_probe(
    monitors: &mut Option<MonitorRecorder>,
    layer: usize,
    values: &[f32],
    experts: &[(usize, Vec<f32>)],
) {
    let Some(recorder) = monitors.as_mut() else {
        return;
    };
    for (expert, expert_values) in experts {
        recorder.record_expert(layer, *expert, expert_values);
    }
    recorder.record_layer(layer, values);
}

/// Decode little-endian f32 values, as a device read-back gives them.
pub(crate) fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// Decode little-endian i32 expert ids, dropping the -1 a slot no expert owns.
pub(crate) fn experts(bytes: &[u8]) -> Vec<usize> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .filter_map(|chunk| {
            let expert = i32::from_le_bytes(*chunk);
            (expert >= 0).then_some(expert as usize)
        })
        .collect()
}

/// Log one dump's first failure.
fn warn(once: &Once, dump: &'static str, error: &io::Error) {
    once.call_once(|| {
        eprintln!("sparkflash numerics: {dump} dump: {error}; not recording it");
    });
}

static COVERAGE_WARNED: Once = Once::new();
static MONITOR_WARNED: Once = Once::new();

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn read(path: &std::path::Path) -> Value {
        serde_json::from_str(&std::fs::read_to_string(path).expect("the dump is readable"))
            .expect("the dump is JSON")
    }

    fn values(width: usize) -> Vec<f32> {
        (0..width).map(|index| index as f32 + 0.5).collect()
    }

    #[test]
    fn an_unset_switch_opens_no_recorder() {
        let switches = Switches::from_lookup(|_| None, "toy/qwen", vec![0, 2], 8);
        assert!(switches.is_off());
        assert!(switches.begin(&[1, 2], 0).is_none());
        record_routing(&mut None, 0, &[0]);
    }

    #[test]
    fn coverage_names_one_input_a_forward_and_accumulates_the_hits() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let switches = Switches::from_lookup(
            |name| (name == COVERAGE_ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/qwen",
            vec![0, 2],
            4,
        );
        assert!(!switches.is_off());
        let mut first = switches.begin(&[1, 2, 3], 0).expect("a recorder");
        first.routing(1, &[0, 1]);
        first.routing(1, &[1]);
        first.finish();
        let mut second = switches.begin(&[4], 3).expect("a second recorder");
        second.routing(1, &[0]);
        second.finish();
        let document = read(&path);
        assert_eq!(
            document["schema"],
            sparkflash_model::numerics_coverage::SCHEMA
        );
        assert_eq!(document["model"], "toy/qwen");
        assert_eq!(document["inputs"].as_array().map(Vec::len), Some(2));
        assert_eq!(
            document["inputs"][0]["name"],
            sparkflash_model::numerics_coverage::input_name(&[1, 2, 3])
        );
        assert_eq!(document["inputs"][0]["name"], "input-fd1f0f4381eb0395");
        assert_eq!(document["inputs"][0]["tokens"], 3);
        assert_eq!(
            document["inputs"][1]["name"],
            sparkflash_model::numerics_coverage::input_name(&[4])
        );
        assert_eq!(document["inputs"][1]["name"], "input-cd3ac65e44f721b1");
        assert_eq!(document["inputs"][1]["tokens"], 1);
        let layers = document["layers"].as_array().expect("the layers");
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0]["hits"], json!([]));
        assert_eq!(layers[1]["hits"], json!([2, 2]));
    }

    #[test]
    fn monitors_take_the_first_forward_as_the_probe() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = Switches::from_lookup(
            |name| match name {
                MONITOR_ENV => Some(path.to_string_lossy().into_owned()),
                SAMPLE_ENV => Some("4".to_owned()),
                _ => None,
            },
            "toy/qwen",
            vec![2, 2],
            8,
        );
        let mut probe = switches.begin(&[1, 2], 0).expect("the probe");
        probe.probe(0, &values(8), &[(0, values(8)), (1, values(8))]);
        probe.finish();
        let document = read(&path);
        assert_eq!(
            document["schema"],
            sparkflash_model::numerics_monitors::SCHEMA
        );
        assert_eq!(document["model"], "toy/qwen");
        assert_eq!(document["probe"]["name"], "position-0");
        assert_eq!(document["probe"]["prompt_ids"], json!([1, 2]));
        assert_eq!(document["probe"]["position"], 0);
        assert_eq!(document["sample"]["width"], 8);
        assert_eq!(
            document["sample"]["indices"].as_array().map(Vec::len),
            Some(4),
            "SPARKFLASH_NUMERICS_SAMPLE=4 sets the sample count"
        );
        assert_eq!(document["layers"][0]["values"], json!([0.5, 2.5, 4.5, 6.5]));
        assert_eq!(document["layers"][0]["experts"][0]["expert"], 0);
        assert_eq!(
            document["layers"][0]["experts"].as_array().map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn a_later_forward_is_a_different_probe_and_records_nothing() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = Switches::from_lookup(
            |name| (name == MONITOR_ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/qwen",
            vec![1],
            4,
        );
        let mut probe = switches.begin(&[1], 0).expect("the probe");
        probe.probe(0, &values(4), &[(0, values(4))]);
        probe.finish();
        // A second forward is a different probe; the recorder reports AlreadyExists, which
        // begin logs once and drops, so this forward has no monitor dump.
        assert!(switches.begin(&[2], 1).is_none());
        let document = read(&path);
        assert_eq!(document["probe"]["prompt_ids"], json!([1]));
        assert_eq!(document["probe"]["position"], 0);
    }

    #[test]
    fn an_off_switch_writes_no_file() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let switches = Switches::from_lookup(|_| None, "toy/qwen", vec![2], 4);
        assert!(switches.begin(&[1], 0).is_none());
        assert!(!path.exists());
    }
}
