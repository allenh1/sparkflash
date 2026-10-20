//! Opt-in device-chain timing: the staging-vs-kernel lines D42 requires from one window.
//!
//! The chain already prints per-position totals, but D42 asks for a line that separates the
//! staging cost by source (window rows, absolute main-KV rows, indexer key pool) from the host
//! copy and kernel time. This module is the accounting half of that: Recorder accepts one
//! LayerSample per attention body call and one StepSample per sequence step, and writes a single
//! JSON document when the operator asked for it.
//!
//! The recorder is disabled unless --timing-json was passed. A disabled recorder accepts samples
//! as no-ops and write touches no file, so the default run's output and files are unchanged. The
//! device code builds a sample from the lengths of the buffers it actually uploads through the
//! window_bytes, absolute_bytes and indexer_pool_bytes helpers here, so the byte formulas are
//! exercised by host tests even though the uploads themselves need a device.
//!
//! See docs/build/device-chain-timing.md for what is measured versus inferred.

use std::cell::RefCell;
use std::io;
use std::path::Path;
use std::rc::Rc;

use serde::Serialize;

/// The schema tag of the JSON document written by Recorder::write.
pub const SCHEMA: &str = "sparkflash-chain-timing/1";

/// Bytes of one staged window row's codes (the csa window row is 512 B of codes).
pub const WINDOW_ROW_CODE_BYTES: u64 = 512;
/// Bytes of one staged window row's block scales.
pub const WINDOW_ROW_SCALE_BYTES: u64 = 16;
/// Bytes of one staged absolute main-KV row's FP4 codes.
pub const ABSOLUTE_ROW_CODE_BYTES: u64 = 256;
/// Bytes of one staged absolute main-KV row's block scales.
pub const ABSOLUTE_ROW_SCALE_BYTES: u64 = 32;
/// Bytes of one indexer key-pool entry (128 BF16 values, the HSI key row).
pub const INDEXER_KEY_ROW_BYTES: u64 = 128 * 2;

/// Bytes of rows staged window rows (codes plus scales).
#[must_use]
pub const fn window_bytes(rows: u64) -> u64 {
    rows * (WINDOW_ROW_CODE_BYTES + WINDOW_ROW_SCALE_BYTES)
}

/// Bytes of rows staged absolute main-KV rows (FP4 codes plus scales).
#[must_use]
pub const fn absolute_bytes(rows: u64) -> u64 {
    rows * (ABSOLUTE_ROW_CODE_BYTES + ABSOLUTE_ROW_SCALE_BYTES)
}

/// Bytes of rows staged indexer key-pool entries.
#[must_use]
pub const fn indexer_pool_bytes(rows: u64) -> u64 {
    rows * INDEXER_KEY_ROW_BYTES
}

/// One attention body call's staged operands and measured time.
///
/// Counts and bytes are measured from the buffers the body uploads; the byte relations are the
/// window_bytes, absolute_bytes and indexer_pool_bytes helpers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LayerSample {
    /// The sequence position the body ran at.
    pub position: u64,
    /// The layer index the body ran for.
    pub layer: u64,
    /// Window rows staged for the body.
    pub window_rows: u64,
    /// Window bytes uploaded: codes plus block scales of every staged row.
    pub window_bytes: u64,
    /// Absolute main-KV candidate rows staged (quantised from the host KV).
    pub absolute_rows: u64,
    /// Absolute main-KV pool bytes uploaded; includes the ratio-driven zero rows the kernel
    /// indexes over, so it can exceed absolute_rows * 288.
    pub absolute_bytes: u64,
    /// Indexer key-pool bytes staged by this body. Zero for the current sequence path, which
    /// zeroes the HSI pool once at open instead of rebuilding it per step.
    pub indexer_pool_bytes: u64,
    /// Host wall time spent building, packing and quantising the operands before the first upload.
    pub host_staging_ns: u64,
    /// Host-to-device copy calls made by the body.
    pub h2d_calls: u64,
    /// Bytes moved by those host-to-device copies.
    pub h2d_bytes: u64,
    /// Kernel launches issued by the body (the format-2 attention launch; projections are counted
    /// by the projection ledger, not here).
    pub launches: u64,
    /// Device wall time from the launch call to the end of the existing readback, which already
    /// synchronises: kernel time plus the D2H transfer, not a GPU timestamp.
    pub sync_gated_device_ns: u64,
}

/// One sequence step's total wall time, recorded by the chain around the engine step call.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StepSample {
    /// The sequence position the step produced.
    pub position: u64,
    /// The step's total wall time.
    pub total_ns: u64,
}

/// Sums over the recorded samples. Every field is a measured sum of the samples, never a model
/// estimate.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Totals {
    /// Attention body samples recorded.
    pub layer_samples: u64,
    /// Window rows summed over the samples.
    pub window_rows: u64,
    /// Window bytes summed over the samples.
    pub window_bytes: u64,
    /// Absolute main-KV candidate rows summed over the samples.
    pub absolute_rows: u64,
    /// Absolute main-KV bytes summed over the samples.
    pub absolute_bytes: u64,
    /// Indexer key-pool bytes summed over the samples.
    pub indexer_pool_bytes: u64,
    /// Host staging time summed over the samples.
    pub host_staging_ns: u64,
    /// H2D copy calls summed over the samples.
    pub h2d_calls: u64,
    /// H2D bytes summed over the samples.
    pub h2d_bytes: u64,
    /// Kernel launches summed over the samples.
    pub launches: u64,
    /// Sync-gated device time summed over the samples.
    pub sync_gated_device_ns: u64,
    /// Sequence steps recorded.
    pub step_samples: u64,
    /// Total step time summed over the steps.
    pub step_total_ns: u64,
}

/// The JSON document written to --timing-json.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct Document {
    /// The document schema tag, always SCHEMA.
    pub schema: String,
    /// The per-layer attention body samples, in recording order.
    pub samples: Vec<LayerSample>,
    /// The per-step chain timings, in recording order.
    pub steps: Vec<StepSample>,
    /// The measured sums over both sample lists.
    pub totals: Totals,
}

/// The opt-in timing recorder shared by the chain loop and the attention backend.
///
/// Clone the Rc when both need to write to it; the chain is single-threaded. A disabled recorder
/// records nothing, so its document has empty sample lists and its write writes no file.
#[derive(Debug, Default)]
pub struct Recorder {
    enabled: bool,
    samples: Vec<LayerSample>,
    steps: Vec<StepSample>,
}

impl Recorder {
    /// A recorder that records nothing, used when --timing-json was not passed.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            samples: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// A recorder that accepts samples, created when --timing-json was passed.
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            samples: Vec::new(),
            steps: Vec::new(),
        }
    }

    /// Whether this recorder accepts samples.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Records one attention body sample; a no-op on a disabled recorder.
    pub fn record_layer(&mut self, sample: LayerSample) {
        if self.enabled {
            self.samples.push(sample);
        }
    }

    /// Records one sequence step sample; a no-op on a disabled recorder.
    pub fn record_step(&mut self, sample: StepSample) {
        if self.enabled {
            self.steps.push(sample);
        }
    }

    /// The measured sums over the recorded samples.
    #[must_use]
    pub fn totals(&self) -> Totals {
        let mut totals = Totals {
            layer_samples: self.samples.len() as u64,
            step_samples: self.steps.len() as u64,
            ..Totals::default()
        };
        for sample in &self.samples {
            totals.window_rows = totals.window_rows.saturating_add(sample.window_rows);
            totals.window_bytes = totals.window_bytes.saturating_add(sample.window_bytes);
            totals.absolute_rows = totals.absolute_rows.saturating_add(sample.absolute_rows);
            totals.absolute_bytes = totals.absolute_bytes.saturating_add(sample.absolute_bytes);
            totals.indexer_pool_bytes = totals
                .indexer_pool_bytes
                .saturating_add(sample.indexer_pool_bytes);
            totals.host_staging_ns = totals
                .host_staging_ns
                .saturating_add(sample.host_staging_ns);
            totals.h2d_calls = totals.h2d_calls.saturating_add(sample.h2d_calls);
            totals.h2d_bytes = totals.h2d_bytes.saturating_add(sample.h2d_bytes);
            totals.launches = totals.launches.saturating_add(sample.launches);
            totals.sync_gated_device_ns = totals
                .sync_gated_device_ns
                .saturating_add(sample.sync_gated_device_ns);
        }
        for step in &self.steps {
            totals.step_total_ns = totals.step_total_ns.saturating_add(step.total_ns);
        }
        totals
    }

    /// The JSON document for the samples recorded so far.
    #[must_use]
    pub fn document(&self) -> Document {
        Document {
            schema: SCHEMA.to_owned(),
            samples: self.samples.clone(),
            steps: self.steps.clone(),
            totals: self.totals(),
        }
    }

    /// Writes the JSON document to path.
    ///
    /// A disabled recorder writes nothing and returns Ok(()): the default run must not create a
    /// file the operator did not ask for.
    ///
    /// # Errors
    /// The path cannot be written, or the document cannot be serialised.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let json = serde_json::to_string_pretty(&self.document())
            .map_err(|error| io::Error::other(format!("timing document: {error}")))?;
        std::fs::write(path, format!("{json}\n"))
    }
}

/// A shared handle to a recorder, the type the chain and the attention backend both hold.
pub type SharedRecorder = Rc<RefCell<Recorder>>;

/// Wraps an enabled recorder in the shared handle the chain passes to the backend.
#[must_use]
pub fn share(recorder: Recorder) -> SharedRecorder {
    Rc::new(RefCell::new(recorder))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(
        position: u64,
        layer: u64,
        window_rows: u64,
        absolute_rows: u64,
        absolute_pool: u64,
    ) -> LayerSample {
        LayerSample {
            position,
            layer,
            window_rows,
            window_bytes: window_bytes(window_rows),
            absolute_rows,
            absolute_bytes: absolute_bytes(absolute_pool),
            indexer_pool_bytes: 0,
            host_staging_ns: 1_000 + layer,
            h2d_calls: 7,
            h2d_bytes: window_bytes(window_rows) + absolute_bytes(absolute_pool),
            launches: 1,
            sync_gated_device_ns: 10_000 + layer,
        }
    }

    #[test]
    fn row_byte_formulas_are_the_measured_row_sizes() {
        assert_eq!(WINDOW_ROW_CODE_BYTES + WINDOW_ROW_SCALE_BYTES, 528);
        assert_eq!(ABSOLUTE_ROW_CODE_BYTES + ABSOLUTE_ROW_SCALE_BYTES, 288);
        assert_eq!(INDEXER_KEY_ROW_BYTES, 256);
        assert_eq!(window_bytes(128), 128 * 528);
        assert_eq!(absolute_bytes(64), 64 * 288);
        assert_eq!(indexer_pool_bytes(2048 * 8), 2048 * 8 * 256);
    }

    #[test]
    fn a_disabled_recorder_records_nothing_and_writes_no_file() {
        let mut recorder = Recorder::disabled();
        recorder.record_layer(sample(0, 0, 1, 0, 0));
        recorder.record_step(StepSample {
            position: 0,
            total_ns: 5,
        });
        assert!(!recorder.is_enabled());
        assert_eq!(recorder.totals(), Totals::default());
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("timing.json");
        recorder.write(&path).expect("a disabled write is a no-op");
        assert!(!path.exists(), "a disabled recorder must not create a file");
    }

    #[test]
    fn a_synthetic_run_totals_the_hand_computed_sum() {
        let mut recorder = Recorder::enabled();
        // A two-layer ring step at position 128: 128 window rows, 64 absolute candidates.
        recorder.record_layer(sample(128, 0, 128, 64, 64));
        // Position 129: 128 window rows, 65 candidates but 66 pool rows because the ratio-2
        // store rounds the odd row up.
        recorder.record_layer(sample(129, 1, 128, 65, 66));
        recorder.record_step(StepSample {
            position: 128,
            total_ns: 1_000_000,
        });
        recorder.record_step(StepSample {
            position: 129,
            total_ns: 1_200_000,
        });

        let totals = recorder.totals();
        assert_eq!(totals.layer_samples, 2);
        assert_eq!(totals.window_rows, 256);
        assert_eq!(totals.window_bytes, 2 * 128 * 528);
        assert_eq!(totals.absolute_rows, 129);
        assert_eq!(totals.absolute_bytes, 64 * 288 + 66 * 288);
        assert_eq!(totals.host_staging_ns, 1_000 + 1_001);
        assert_eq!(totals.h2d_calls, 14);
        assert_eq!(totals.launches, 2);
        assert_eq!(totals.step_samples, 2);
        assert_eq!(totals.step_total_ns, 2_200_000);
        assert_eq!(
            totals.h2d_bytes,
            window_bytes(256) + absolute_bytes(64 + 66)
        );
    }

    #[test]
    fn the_json_document_carries_the_schema_samples_and_totals() {
        let mut recorder = Recorder::enabled();
        recorder.record_layer(sample(0, 3, 1, 0, 0));
        recorder.record_step(StepSample {
            position: 0,
            total_ns: 42,
        });
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().join("timing.json");
        recorder.write(&path).expect("the timing file");
        let text = std::fs::read_to_string(&path).expect("the written timing file");
        let value: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["samples"].as_array().expect("samples").len(), 1);
        assert_eq!(value["steps"].as_array().expect("steps").len(), 1);
        assert_eq!(value["samples"][0]["window_rows"], 1);
        assert_eq!(value["totals"]["layer_samples"], 1);
        assert_eq!(value["totals"]["step_total_ns"], 42);
        assert_eq!(value["totals"]["window_bytes"].as_u64(), Some(528));
        let again = recorder.document();
        assert_eq!(again, recorder.document());
    }
}
