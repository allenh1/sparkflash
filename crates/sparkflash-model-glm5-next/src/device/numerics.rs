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

//! The numerics dumps' GLM plumbing (docs/NUMERICS.md section 5).
//!
//! The writers are [sparkflash_model::numerics_coverage] and
//! [sparkflash_model::numerics_monitors], the same ones Qwen3.8's device arm uses. This module is
//! the GLM arm's side of them: the switches are read once when the rank loads, each forward opens
//! its coverage input (and, when it is a decode step and this is rank 0, the monitor probe), and
//! [record_routing] and [record_probe] turn the router's chosen expert ids and the sampled rows
//! the device read back into the calls the recorders take.
//!
//! Off is the default, and the cost of off is one `Option` check a forward and one a layer: no
//! environment read, no formatting and no allocation. The rules themselves — the documents'
//! shapes, the content-addressed input naming and accumulation, the one-probe rule and the off
//! switch — are tested here with a synthetic table, so they need no GPU.

use std::io;
use std::path::PathBuf;
use std::sync::Once;

use sparkflash_model::numerics_coverage::{
    ENV as COVERAGE_ENV, MODEL_ENV, Recorder as CoverageRecorder,
};
use sparkflash_model::numerics_monitors::{
    ENV as MONITOR_ENV, Probe, Recorder, SAMPLE_ENV, probe_eligible,
};

/// The numerics switches a rank reads once, when it loads; both unset is the default.
pub(super) struct Switches {
    coverage: Option<CoverageSwitch>,
    monitors: Option<MonitorSwitch>,
}

/// What `SPARKFLASH_NUMERICS_COVERAGE` asked for: a path, the dump's model id and the model's
/// per-layer routed-expert counts.
struct CoverageSwitch {
    path: PathBuf,
    model: String,
    experts: Vec<usize>,
}

/// What `SPARKFLASH_NUMERICS_MONITORS` asked for: a path, the dump's model id, the hidden width
/// and the raw `SPARKFLASH_NUMERICS_SAMPLE` value the recorder parses at [Switches::begin].
struct MonitorSwitch {
    path: PathBuf,
    model: String,
    width: usize,
    sample: Option<String>,
}

impl Switches {
    /// Read the switches from the process environment once, for a model with `experts` routed
    /// experts per decoder layer and a `width`-wide hidden state.
    pub(super) fn from_env(model: &str, experts: Vec<usize>, width: usize) -> Self {
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
    pub(super) fn is_off(&self) -> bool {
        !self.coverage_enabled() && !self.monitors_enabled()
    }

    /// Whether `SPARKFLASH_NUMERICS_COVERAGE` names a path.
    pub(super) fn coverage_enabled(&self) -> bool {
        self.coverage.is_some()
    }

    /// Whether `SPARKFLASH_NUMERICS_MONITORS` names a path.
    pub(super) fn monitors_enabled(&self) -> bool {
        self.monitors.is_some()
    }

    /// Open this forward's recorders, or `None` when both switches are unset (or neither recorder
    /// could be opened).
    ///
    /// The coverage recorder opens one content-addressed input, `input-<digest>`, named from this
    /// forward's token ids, so a serving process accumulates a suite across forwards exactly as the
    /// host reference does and a repeated input adds no second entry.
    ///
    /// The monitor recorder takes this forward as the probe only when it is the first *decode
    /// step*: a forward of at most [probe_eligible]'s few-row band, and only on rank 0, which is
    /// the rank the dump's tensor-parallel share belongs to. A prompt's prefill is wider, runs the
    /// bulk GEMM on every build, and must not consume the probe, so it opens no monitor recorder;
    /// the first decode step after it becomes the probe. A later decode step is a different probe,
    /// which the recorder reports as `AlreadyExists`. That is logged once and this forward serves
    /// on without a monitor dump.
    pub(super) fn begin(
        &self,
        tokens: &[u32],
        position: usize,
        rows: usize,
        record_monitors: bool,
    ) -> Option<Forward> {
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
        let monitors = self
            .monitors
            .as_ref()
            .filter(|_| record_monitors && probe_eligible(rows))
            .and_then(|switch| {
                let probe = Probe {
                    name: format!("decode-position-{position}"),
                    prompt_ids: tokens.to_vec(),
                    position,
                    rows,
                };
                let recorder = Recorder::from_lookup(
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

/// One layer's read-back for the monitor dump: the router's routed ids and weights, the layer's
/// combined row and each slot's row as its down projection wrote it.
pub(super) struct ProbeRow {
    pub(super) routed: Vec<(usize, f32)>,
    pub(super) values: Vec<f32>,
    pub(super) experts: Vec<Vec<f32>>,
}

/// The recorders of one forward, opened by [Switches::begin] and written when the step ends.
pub(super) struct Forward {
    coverage: Option<CoverageRecorder>,
    monitors: Option<Recorder>,
}

impl Forward {
    /// Whether this forward counts routing decisions.
    pub(super) fn coverage_enabled(&self) -> bool {
        self.coverage.is_some()
    }

    /// Count one layer's chosen experts.
    pub(super) fn routing(&mut self, layer: usize, selected: &[usize]) {
        if let Some(recorder) = self.coverage.as_mut() {
            for expert in selected {
                recorder.hit(layer, *expert);
            }
        }
    }

    /// Write the dumps this forward filled. A write that fails is logged once and serving goes
    /// on; the dumps are diagnostics, not part of the answer.
    pub(super) fn finish(self) {
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

/// Count one layer's routing decisions in the coverage dump: one [CoverageRecorder::hit] a selected
/// expert. `layer` is the decoder layer; `None` is the MTP layer's draft pass, which has no
/// decoder layer and records nothing. With no recorder this is one `Option` check.
pub(super) fn record_routing(
    coverage: &mut Option<Forward>,
    layer: Option<usize>,
    selected: &[usize],
) {
    let (Some(forward), Some(layer)) = (coverage.as_mut(), layer) else {
        return;
    };
    forward.routing(layer, selected);
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
    let Some(recorder) = forward.monitors.as_mut() else {
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
        recorder.record_expert(layer, *expert, &body);
    }
    recorder.record_layer(layer, values);
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

/// Decode little-endian i32 expert ids, dropping the -1 a slot no expert owns.
pub(super) fn experts(bytes: &[u8]) -> Vec<usize> {
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

/// Log one dump's first write failure.
fn warn(once: &Once, dump: &'static str, error: &io::Error) {
    once.call_once(|| {
        eprintln!("sparkflash numerics: {dump} dump: {error}; not recording it");
    });
}

/// Log one dump's first read-back failure.
fn warn_read_back(once: &Once, dump: &'static str, error: &str) {
    once.call_once(|| {
        eprintln!("sparkflash numerics: {dump} read-back: {error}; not recording this forward");
    });
}

/// Log the monitor read-back's first failure.
pub(super) fn warn_monitor_read_back(error: &str) {
    warn_read_back(&MONITOR_READ_BACK_WARNED, "monitor", error);
}

/// Log the coverage read-back's first failure.
pub(super) fn warn_coverage_read_back(error: &str) {
    warn_read_back(&COVERAGE_READ_BACK_WARNED, "coverage", error);
}

static COVERAGE_WARNED: Once = Once::new();
static MONITOR_WARNED: Once = Once::new();
static COVERAGE_READ_BACK_WARNED: Once = Once::new();
static MONITOR_READ_BACK_WARNED: Once = Once::new();

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use sparkflash_model::numerics_coverage::input_name;
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
            vec![16],
            16,
        )
    }

    fn coverage_switches(path: &Path) -> Switches {
        Switches::from_lookup(
            |name| (name == COVERAGE_ENV).then(|| path.to_string_lossy().into_owned()),
            "toy/glm",
            vec![0, 2, 0, 3],
            16,
        )
    }

    fn probe(
        switches: &Switches,
        tokens: &[u32],
        position: usize,
        rows: usize,
        layer: usize,
        routed: &[(usize, f32)],
    ) -> Option<Forward> {
        let mut forward =
            Some(switches.begin(tokens, position, rows, true).expect(
                "the switch is set, the forward is a decode step and the document is fresh",
            ));
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
        let switches = Switches::from_lookup(|_| None, "toy/glm", vec![16], 16);
        assert!(switches.is_off());
        assert!(!switches.coverage_enabled());
        assert!(switches.begin(&[1, 2], 0, 1, true).is_none());
        record_probe(&mut None, 0, &[(1, 0.5)], &values(16), &[values(16)]);
    }

    #[test]
    fn the_token_band_is_the_arm_s_decode_band() {
        assert_eq!(
            sparkflash_model::numerics_monitors::TOKEN_BAND,
            crate::device::MAX_ROWS,
            "the shared probe rule's band is the arm's widest decode step"
        );
    }

    #[test]
    fn a_prefill_forward_does_not_consume_the_probe() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = switches(&path, None);
        // The prompt's prefill is wider than the few-row band: this step opens no monitor
        // recorder and writes nothing.
        assert!(switches.begin(&[1, 2, 3], 0, 63, true).is_none());
        assert!(!path.exists(), "a prefill forward writes no dump");
        // The first decode step after it is the probe.
        let mut forward = probe(&switches, &[9], 63, 1, 0, &[(0, 1.0)]);
        forward.take().expect("the first decode step").finish();
        let document = read(&path);
        assert_eq!(document["probe"]["name"], "decode-position-63");
        assert_eq!(document["probe"]["prompt_ids"], json!([9]));
        assert_eq!(document["probe"]["position"], 63);
        assert_eq!(document["probe"]["rows"], 1);
    }

    #[test]
    fn the_dumped_values_come_from_the_decode_step() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = switches(&path, Some("4"));
        // The prompt's prefill row is a table of nines: recording it would be visible.
        let prefill = vec![9.0f32; 16];
        assert!(
            switches.begin(&[1, 2, 3], 0, 63, true).is_none(),
            "a prefill is not the probe"
        );
        // A step the rule did not make the probe opened no monitor recorder, so it records
        // nothing.
        let mut monitors = None;
        record_probe(
            &mut monitors,
            0,
            &[(0, 1.0)],
            &prefill,
            std::slice::from_ref(&prefill),
        );
        // The first decode step's row is the sampled table, and only it is recorded.
        let mut forward = probe(&switches, &[4], 3, 1, 0, &[(0, 1.0)]);
        forward.take().expect("the first decode step").finish();
        let document = read(&path);
        assert_eq!(document["probe"]["rows"], 1);
        assert_eq!(
            document["layers"][0]["values"],
            json!([0.5, 4.5, 8.5, 12.5])
        );
    }

    #[test]
    fn the_document_has_the_shape_the_judge_reads() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("nested/monitors.json");
        let switches = switches(&path, Some("4"));
        assert!(!switches.is_off());
        let mut forward = probe(&switches, &[7], 3, 1, 3, &[(5, 0.25), (9, 0.5)]);
        forward.take().expect("the probe").finish();

        let document = read(&path);
        assert_eq!(
            document["schema"],
            sparkflash_model::numerics_monitors::SCHEMA
        );
        assert_eq!(document["model"], "toy/glm");
        assert_eq!(document["probe"]["name"], "decode-position-3");
        assert_eq!(document["probe"]["prompt_ids"], json!([7]));
        assert_eq!(document["probe"]["position"], 3);
        assert_eq!(document["probe"]["rows"], 1);
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
            vec![16],
            16,
        );
        let mut forward = probe(&switches, &[1], 0, 1, 0, &[(1, 1.0)]);
        forward.take().expect("the probe").finish();
        assert_eq!(read(&path)["model"], "fixture/glm");
    }

    #[test]
    fn a_later_decode_step_is_a_different_probe_and_records_nothing() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = switches(&path, None);
        let mut first = probe(&switches, &[1], 0, 1, 0, &[(0, 1.0)]);
        first.take().expect("the probe").finish();
        // A later decode step is a different probe; the recorder reports AlreadyExists, which
        // begin logs once and drops, so this step has no monitor dump.
        assert!(switches.begin(&[2], 1, 1, true).is_none());
        let document = read(&path);
        assert_eq!(document["probe"]["prompt_ids"], json!([1]));
        assert_eq!(document["probe"]["position"], 0);
        assert_eq!(document["probe"]["rows"], 1);
    }

    #[test]
    fn the_same_probe_merges_the_document_on_disk() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("monitors.json");
        let switches = switches(&path, None);
        let mut first = probe(&switches, &[1], 0, 1, 0, &[(0, 1.0)]);
        first.take().expect("the probe").finish();
        let mut second = probe(&switches, &[1], 0, 1, 1, &[(2, 1.0)]);
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
                .begin(&[1], 0, 1, true)
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
        let mut forward = probe(&switches, &[1], 0, 1, 0, &[(0, 1.0)]);
        // The path a directory is now: the recorder cannot write a file there.
        std::fs::create_dir(&path).expect("a directory where the dump would go");
        forward.take().expect("the probe").finish();
    }

    #[test]
    fn an_unset_coverage_switch_opens_no_recorder() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let switches = Switches::from_lookup(|_| None, "toy/glm", vec![2], 16);
        assert!(switches.is_off());
        assert!(!switches.coverage_enabled());
        assert!(switches.begin(&[1], 0, 1, true).is_none());
        record_routing(&mut None, Some(0), &[0]);
        assert!(!path.exists());
    }

    #[test]
    fn coverage_names_one_input_a_forward_and_accumulates_the_hits() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let switches = coverage_switches(&path);
        assert!(switches.coverage_enabled());
        assert!(!switches.is_off());
        // Coverage opens on every forward, the prompt's 3-row prefill included.
        let mut first = Some(switches.begin(&[1, 2, 3], 0, 3, true).expect("a recorder"));
        record_routing(&mut first, Some(1), &[0, 1]);
        record_routing(&mut first, Some(1), &[1]);
        first.take().expect("the first forward").finish();
        let mut second = Some(switches.begin(&[4], 3, 1, true).expect("a second recorder"));
        record_routing(&mut second, Some(3), &[2]);
        second.take().expect("the second forward").finish();

        let document = read(&path);
        assert_eq!(
            document["schema"],
            sparkflash_model::numerics_coverage::SCHEMA
        );
        assert_eq!(document["model"], "toy/glm");
        assert_eq!(document["inputs"].as_array().map(Vec::len), Some(2));
        assert_eq!(document["inputs"][0]["name"], input_name(&[1, 2, 3]));
        assert_eq!(document["inputs"][0]["tokens"], 3);
        assert_eq!(document["inputs"][1]["name"], input_name(&[4]));
        assert_eq!(document["inputs"][1]["tokens"], 1);
        let layers = document["layers"].as_array().expect("the layers");
        assert_eq!(layers.len(), 4);
        assert_eq!(
            layers[0]["hits"],
            json!([]),
            "a dense layer carries no hits"
        );
        assert_eq!(layers[1]["hits"], json!([1, 2]));
        assert_eq!(layers[2]["hits"], json!([]));
        assert_eq!(layers[3]["hits"], json!([0, 0, 1]));
    }

    #[test]
    fn a_repeated_input_adds_no_second_entry_but_still_counts_its_hits() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let switches = coverage_switches(&path);
        let mut first = Some(switches.begin(&[1, 2], 0, 2, true).expect("a recorder"));
        record_routing(&mut first, Some(1), &[0]);
        first.take().expect("the first forward").finish();
        let mut second = Some(switches.begin(&[1, 2], 2, 2, true).expect("a recorder"));
        record_routing(&mut second, Some(1), &[1]);
        second.take().expect("the second forward").finish();
        let document = read(&path);
        assert_eq!(document["inputs"].as_array().map(Vec::len), Some(1));
        assert_eq!(document["inputs"][0]["name"], input_name(&[1, 2]));
        assert_eq!(document["inputs"][0]["tokens"], 2);
        assert_eq!(document["layers"][1]["hits"], json!([1, 1]));
    }

    #[test]
    fn the_mtp_draft_pass_records_nothing() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let switches = coverage_switches(&path);
        let mut forward = Some(switches.begin(&[7], 0, 1, true).expect("a recorder"));
        // The MTP layer is a draft pass: combine_experts passes None and the routing is not
        // counted.
        record_routing(&mut forward, None, &[0, 1, 2]);
        forward.take().expect("the forward").finish();
        let document = read(&path);
        assert_eq!(
            document["inputs"].as_array().map(Vec::len),
            Some(1),
            "the forward still names its input"
        );
        for (index, layer) in document["layers"]
            .as_array()
            .expect("the layers")
            .iter()
            .enumerate()
        {
            for hit in layer["hits"].as_array().expect("the layer's hits") {
                assert_eq!(hit, &json!(0), "layer {index} recorded the draft pass");
            }
        }
    }

    #[test]
    fn the_coverage_model_id_prefers_the_environment_over_the_descriptor() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let switches = Switches::from_lookup(
            |name| match name {
                MODEL_ENV => Some("fixture/glm".to_owned()),
                COVERAGE_ENV => Some(path.to_string_lossy().into_owned()),
                _ => None,
            },
            "toy/glm",
            vec![2],
            16,
        );
        let mut forward = Some(switches.begin(&[1], 0, 1, true).expect("a recorder"));
        record_routing(&mut forward, Some(0), &[1]);
        forward.take().expect("the forward").finish();
        assert_eq!(read(&path)["model"], "fixture/glm");
    }

    #[test]
    fn a_rank_that_writes_no_monitor_still_counts_coverage() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let coverage = directory.path().join("coverage.json");
        let monitors = directory.path().join("monitors.json");
        let switches = Switches::from_lookup(
            |name| match name {
                COVERAGE_ENV => Some(coverage.to_string_lossy().into_owned()),
                MONITOR_ENV => Some(monitors.to_string_lossy().into_owned()),
                _ => None,
            },
            "toy/glm",
            vec![1],
            16,
        );
        // Rank 1 opens the coverage recorder, which counts this rank's routing, and no monitor
        // recorder: the monitor dump is rank 0's share of the sum.
        let mut forward = Some(
            switches
                .begin(&[1], 0, 1, false)
                .expect("coverage opens on every rank"),
        );
        let opened = forward.as_ref().expect("the forward");
        assert!(opened.coverage_enabled());
        assert!(opened.monitors.is_none());
        record_routing(&mut forward, Some(0), &[0]);
        forward.take().expect("the forward").finish();
        assert_eq!(read(&coverage)["layers"][0]["hits"], json!([1]));
        assert!(!monitors.exists(), "rank 1 writes no monitor dump");
    }

    #[test]
    fn a_coverage_write_failure_is_logged_and_does_not_fail_serving() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("coverage.json");
        let switches = coverage_switches(&path);
        let mut forward = Some(switches.begin(&[1], 0, 1, true).expect("a recorder"));
        record_routing(&mut forward, Some(1), &[0]);
        // The path a directory is now: the recorder cannot write a file there.
        std::fs::create_dir(&path).expect("a directory where the dump would go");
        forward.take().expect("the forward").finish();
    }

    #[test]
    fn the_router_ids_decode_drops_a_slot_no_expert_owns() {
        let ids = [3i32, -1, 7];
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        assert_eq!(experts(&bytes), vec![3, 7]);
    }
}
