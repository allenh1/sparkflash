//! The Phase 1 one-token run, as a text report with a fixed contract.
//!
//! One token is the cheapest forward that exists and the first place the whole backbone runs:
//! forty layers, all forty mHC passes, the host router, the combine, the head and the sampler. The
//! released-depth harness runs those at tiny shapes with the compression, indexer and Engram lists
//! cleared, so a compressor/indexer/Engram path and the three DSpark stages are not exercised by
//! it; a tiny ratio-2 configuration does exercise the indexer audit class. The released-geometry
//! table waits on the Engram-paged source. The report is the deliverable - it records
//! the per-layer boundary table's row count and classes, the five host seams of the roadmap
//! (section 4.3), the attention sink line (section 4.3, task-36), the tensor audit (section 4.5
//! criterion 5) and run determinism (criterion 6). A run that misses any of them is an error, not
//! a warning: the point of the report is that it can fail.
//!
//! What this report is **not**: a model-fidelity claim. The weights come from whatever
//! [WeightSource] the caller passes - [crate::synthetic::SyntheticWeights] in tests, the real
//! checkpoint through the CLI's loader on the node - and the comparison here is host-vs-host, so
//! its bar is bit-exactness of two runs of the same input, not agreement with the released model.
//! The device half (docs/archive/build/roadmap-to-serving.md section 4.5 criteria 1 and 3) diffs this
//! table against the device path and bar by bar; this module produces the reference table it diffs.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use crate::config::ModelConfig;
use crate::forward::boundary::TENSOR_ORDER;
use crate::forward::{InferenceEngine, SequenceState};
use crate::linalg::{self, EngineError};
use crate::tensor::{TensorPlan, TensorSpec};
use crate::weights::{WeightError, WeightSource};

/// The five host seams of the roadmap's section 4.3, plus the sink line of task-36.
///
/// The text is part of the run's output contract: a test greps the report for each line and fails
/// if one disappears, so a seam that is moved or dropped cannot silently stop being labelled.
pub const SEAM_LABELS: [&str; 6] = [
    "seam: router (gate GEMM, sqrt-softplus, noaux_tc bias, top-6, norm_topk_prob) - host; task-37 moves it",
    "seam: gate/up GEMM (w1/w3) - host; Phase 3 moves it",
    "seam: SwiGLU and the activation quantiser - host; task-43 compares it, Phase 3 moves it",
    "seam: HSI candidate merge (top-512 over the pool) - host; Phase 2/3 move it",
    "seam: head and sampling - host; Phase 1 stage 4 / Phase 3 move it",
    "sink: layers.N.attn.attn_sink F32 [64] consumed on both sides (kernel argument and mla_ref::attention_with_sink)",
];

/// What the weight source holds, which decides what the logits line may claim.
///
/// A synthetic seed is *bytes*, not numbers: an E4M3 code can decode to NaN or inf, so a run over
/// it can only record the logits' shape and determinism, never a prediction. The real checkpoint is
/// *numbers*, so a non-finite logit there is a numerical failure and the report refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightClass {
    /// The source holds the checkpoint's real values; a non-finite logit is an error.
    Numbers,
    /// The source holds seeded bytes; the logits' values are not a model prediction.
    Bytes,
}

/// A source wrapper that records every tensor name the engine reads, for the audit line.
struct CountingSource {
    inner: Box<dyn WeightSource>,
    reads: Arc<Mutex<BTreeSet<String>>>,
}

impl WeightSource for CountingSource {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        self.reads
            .lock()
            .expect("the audit set is not poisoned")
            .insert(name.to_owned());
        self.inner.read(name)
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, WeightError>> {
        // A paged Engram read is a read of the named tensor, so the audit records it exactly like a
        // whole read; otherwise a paged run would look like it never touched the table.
        let result = self.inner.read_ranges(name, ranges)?;
        self.reads
            .lock()
            .expect("the audit set is not poisoned")
            .insert(name.to_owned());
        Some(result)
    }
}

/// Runs one token through the engine and returns the Phase 1 report.
///
/// The engine is built here rather than passed in so the audit can see the reads: the source is
/// wrapped in a counting reader and every name the forward asks for is recorded.
///
/// # Errors
///
/// [EngineError] from the forward, or [EngineError::Shape] when the run breaks the report's
/// contract: a first step that does not record exactly `n_layers` rows, rows out of order, a row
/// missing one of the documented tensors, or two runs of the same token that are not byte-identical.
pub fn one_token_report(
    config: &ModelConfig,
    source: Box<dyn WeightSource>,
    limit: usize,
    token: u32,
    class: WeightClass,
) -> Result<String, EngineError> {
    let plan = TensorPlan::language_model(config);
    let reads = Arc::new(Mutex::new(BTreeSet::new()));
    let counting = CountingSource {
        inner: source,
        reads: Arc::clone(&reads),
    };
    let engine = InferenceEngine::new(config.clone(), Box::new(counting), limit)?;

    let mut first_state = SequenceState::new(config);
    let first = engine.prefill_observed(&mut first_state, &[token])?;
    let mut second_state = SequenceState::new(config);
    let second = engine.prefill_observed(&mut second_state, &[token])?;

    let diff = first.boundary.diff(&second.boundary)?;
    let logits_identical = first.output.logits.len() == second.output.logits.len()
        && first
            .output
            .logits
            .iter()
            .zip(&second.output.logits)
            .all(|(a, b)| a.to_bits() == b.to_bits());

    let mut report = String::new();
    report.push_str(&format!(
        "one-token report: n_layers={} n_mtp_layers={} dim={} vocab={} token={token}\n",
        config.n_layers, config.n_mtp_layers, config.dim, config.vocab_size
    ));
    report.push_str(&format!(
        "rows: {} (a first step records n_layers={} backbone rows)\n",
        first.boundary.len(),
        config.n_layers
    ));
    if first.boundary.len() != config.n_layers {
        return Err(EngineError::Shape {
            what: "one-token report".to_owned(),
            detail: format!(
                "a first step recorded {} rows, not the configured {}",
                first.boundary.len(),
                config.n_layers
            ),
        });
    }
    for (index, row) in first.boundary.rows().iter().enumerate() {
        if row.layer() != index {
            return Err(EngineError::Shape {
                what: "one-token report".to_owned(),
                detail: format!(
                    "row {index} is layer {}; rows must be in layer order",
                    row.layer()
                ),
            });
        }
        if row.tensors().len() != TENSOR_ORDER.len() {
            return Err(EngineError::Shape {
                what: "one-token report".to_owned(),
                detail: format!(
                    "layer {} recorded {} tensors, not the documented {}",
                    row.layer(),
                    row.tensors().len(),
                    TENSOR_ORDER.len()
                ),
            });
        }
        for (tensor, expected) in row.tensors().iter().zip(TENSOR_ORDER) {
            if tensor.name() != expected {
                return Err(EngineError::Shape {
                    what: "one-token report".to_owned(),
                    detail: format!(
                        "layer {} recorded {} where the documented order has {expected}",
                        row.layer(),
                        tensor.name()
                    ),
                });
            }
        }
        report.push_str(&format!(
            "row {}: tensors={}/{} class=host-vs-host bit-exact\n",
            row.layer(),
            row.tensors().len(),
            TENSOR_ORDER.len()
        ));
    }
    if !diff.is_identical() || !logits_identical {
        return Err(EngineError::Shape {
            what: "one-token report".to_owned(),
            detail: format!(
                "two runs of the same token differ: differing layers {:?}, logits identical {logits_identical}",
                diff.differing_layers()
            ),
        });
    }
    report.push_str(&format!(
        "determinism: two runs byte-identical (boundary tables and logits; {} rows, {} logits)\n",
        first.boundary.len(),
        first.output.logits.len()
    ));

    if first.output.logits.len() != config.vocab_size {
        return Err(EngineError::Shape {
            what: "one-token report".to_owned(),
            detail: format!(
                "the step returned {} logits, not the configured vocabulary {}",
                first.output.logits.len(),
                config.vocab_size
            ),
        });
    }
    let finite = first
        .output
        .logits
        .iter()
        .filter(|value| value.is_finite())
        .count();
    if class == WeightClass::Numbers && finite != first.output.logits.len() {
        return Err(EngineError::Shape {
            what: "one-token report".to_owned(),
            detail: format!(
                "{} of {} logits are not finite on a real weight source",
                first.output.logits.len() - finite,
                first.output.logits.len()
            ),
        });
    }
    let mut top: Vec<usize> = (0..first.output.logits.len()).collect();
    top.sort_by(|&a, &b| {
        first.output.logits[b]
            .total_cmp(&first.output.logits[a])
            .then(a.cmp(&b))
    });
    let argmax = top.first().copied().unwrap_or(0);
    let shown = top.len().min(5);
    let top5: Vec<String> = top[..shown]
        .iter()
        .map(|&index| format!("{index}:{:.9}", first.output.logits[index]))
        .collect();
    report.push_str(&format!(
        "logits: n={} argmax={argmax} finite={finite} top5=[{}]\n",
        first.output.logits.len(),
        top5.join(", ")
    ));
    match class {
        WeightClass::Bytes => report.push_str(
            "logits class: the weight source is bytes, not numbers, so the argmax/top5 are a shape and determinism record, not a model prediction\n",
        ),
        WeightClass::Numbers => report.push_str(
            "logits class: real checkpoint values; every logit is finite and the argmax/top5 are a prediction of this reference pass\n",
        ),
    }

    for seam in SEAM_LABELS {
        report.push_str(seam);
        report.push('\n');
    }

    let plan_names: BTreeSet<String> = plan.specs().iter().map(|s| s.name.clone()).collect();
    let read = reads.lock().expect("the audit set is not poisoned");
    let unread: Vec<&String> = plan_names
        .iter()
        .filter(|name| !read.contains(name.as_str()))
        .collect();
    // At a first step the unread names fall into four classes the format document and criterion 5
    // name, and classifying them disjointly is what makes the line a partition rather than a hope:
    // the experts the router did not select; the DSpark stages (every mtp.* tensor, including the
    // stage experts, so it is tested first); the text-path tensor gate.bias_vl, which D18 excludes;
    // and an indexer no position needs at compress_len == 0. Anything outside those classes is a
    // tensor the plan expects and this pass never read, and the report refuses rather than hides it.
    let stages = unread
        .iter()
        .filter(|name| name.starts_with("mtp."))
        .count();
    let experts = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.contains(".ffn.experts."))
        .count();
    let text_path = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.ends_with("ffn.gate.bias_vl"))
        .count();
    let indexer = unread
        .iter()
        .filter(|name| !name.starts_with("mtp.") && name.contains(".attn.indexer."))
        .count();
    let other: Vec<&&String> = unread
        .iter()
        .filter(|name| {
            !name.starts_with("mtp.")
                && !name.contains(".ffn.experts.")
                && !name.ends_with("ffn.gate.bias_vl")
                && !name.contains(".attn.indexer.")
        })
        .collect();
    if !other.is_empty() {
        return Err(EngineError::Shape {
            what: "one-token report".to_owned(),
            detail: format!(
                "{} plan tensors were not read and are in none of the four expected classes: {:?}",
                other.len(),
                other.iter().take(8).collect::<Vec<_>>()
            ),
        });
    }
    // The four filters are constructed to be disjoint, and a report whose own classes do not add
    // up to its unread count is a bookkeeping bug, not a report: check the partition it prints.
    if experts + stages + text_path + indexer != unread.len() {
        return Err(EngineError::Shape {
            what: "one-token report".to_owned(),
            detail: format!(
                "the audit classes {experts}+{stages}+{text_path}+{indexer} do not partition the {} unread tensors",
                unread.len()
            ),
        });
    }
    report.push_str(&format!(
        "audit: plan={} read={} unread={} expected-unread=[unselected-expert {experts}, DSpark-stage {stages}, text-path {text_path}, indexer {indexer}] first_unread={:?}\n",
        plan_names.len(),
        read.len(),
        unread.len(),
        unread.iter().take(8).collect::<Vec<_>>()
    ));

    Ok(report)
}

/// The reference element limit for a small-shape configuration run, for callers that do not want
/// to choose one. Real checkpoint tensors exceed it and are refused; see [linalg::REFERENCE_TENSOR_LIMIT].
pub const SMALL_SHAPE_LIMIT: usize = linalg::REFERENCE_TENSOR_LIMIT;
