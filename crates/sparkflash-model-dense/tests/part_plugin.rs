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

//! A part from outside the dense crate, run by the dense walk.
//!
//! This file is a crate of its own, so it sees only what a model crate sees: the part trait
//! ([sparkflash_runtime::PartExec]), the dense host's two contexts and its registry. It writes a
//! feed-forward part, registers it in place of the catalogue's dense MLP and runs a prompt and
//! two token steps through the walk; and it shows a step whose part has no executor refused by
//! the part's name.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use sparkflash_model::backend::{BackendError, ModelSession};
use sparkflash_model::{Architecture, Capabilities, ModelDescriptor, Part, TpPlan};
use sparkflash_model_dense::{
    Dense, DenseDeviceBackend, DensePart, DenseParts, OpenCx, PassCx, ScratchSlot,
};
use sparkflash_runtime::{PartExec, RankSplit, Region};
use sparkflash_weights::{TensorSpec, WeightError, WeightSource};

/// The test's own feed-forward: the catalogue's SwiGLU MLP under another name. It holds the
/// catalogue part its constructor built and adds what a family's part would add around it: a
/// reservation in the walk's scratch, and a count of the rows each body ran.
struct Renamed {
    inner: DensePart,
    scratch: ScratchSlot,
    prompt_rows: Arc<AtomicUsize>,
    token_rows: Arc<AtomicUsize>,
}

impl PartExec<Dense> for Renamed {
    fn state(&self) -> &[Region] {
        self.inner.state()
    }

    fn split(&self, plan: &TpPlan) -> RankSplit {
        self.inner.split(plan)
    }

    fn rows(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        pass.scratch(self.scratch)?;
        self.prompt_rows.fetch_add(pass.rows(), Ordering::Relaxed);
        self.inner.rows(pass)
    }

    fn token(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        pass.scratch(self.scratch)?;
        self.token_rows.fetch_add(pass.rows(), Ordering::Relaxed);
        self.inner.token(pass)
    }
}

/// The catalogue's registry with [Renamed] as the dense MLP's executor: the one registration
/// line a family writes.
fn renamed_parts(prompt_rows: &Arc<AtomicUsize>, token_rows: &Arc<AtomicUsize>) -> DenseParts {
    let (prompt_rows, token_rows) = (Arc::clone(prompt_rows), Arc::clone(token_rows));
    let mut parts = Dense::catalogue();
    parts
        .replace(Part::DenseMlp, move |open: &mut OpenCx<'_>| {
            let scratch = open.scratch(open.hidden());
            let part: DensePart = Box::new(Renamed {
                inner: open.dense_mlp()?,
                scratch,
                prompt_rows: Arc::clone(&prompt_rows),
                token_rows: Arc::clone(&token_rows),
            });
            Ok(part)
        })
        .expect("the catalogue has a dense MLP to take over");
    parts
}

/// The dense conformance checkpoint: `SPARKFLASH_CONFORMANCE_CHECKPOINT`, or the directory
/// `scripts/onboard-gate.sh dense` writes.
fn checkpoint() -> PathBuf {
    let path = match std::env::var_os("SPARKFLASH_CONFORMANCE_CHECKPOINT") {
        Some(path) if !path.is_empty() => PathBuf::from(path),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/conformance/dense-synthetic"),
    };
    assert!(
        path.join("config.json").is_file() && path.join("model.safetensors").is_file(),
        "the dense conformance checkpoint at {} is not written; run scripts/onboard-gate.sh dense",
        path.display()
    );
    path
}

/// The logits of a four-token prompt and of two token steps after it, as bits, and the tokens
/// chosen.
fn run(
    architecture: &Architecture,
    source: &dyn WeightSource,
    parts: &DenseParts,
) -> (Vec<Vec<u32>>, Vec<Option<u32>>) {
    let context = architecture.max_position_embeddings;
    let descriptor =
        ModelDescriptor::from_architecture("part-plugin", architecture, context, Capabilities::ALL);
    let backend = DenseDeviceBackend::open_with(descriptor, architecture, source, context, parts)
        .expect("the arm opens");
    let mut session = backend.session().expect("a session");
    let mut logits = Vec::new();
    let mut tokens = Vec::new();
    let mut output = session.prefill(&[1, 2, 3, 4]).expect("the prompt");
    for _ in 0..2 {
        let next = output.next_token.expect("a token");
        logits.push(output.logits.iter().map(|value| value.to_bits()).collect());
        tokens.push(output.next_token);
        output = session.decode(next).expect("the token step");
    }
    logits.push(output.logits.iter().map(|value| value.to_bits()).collect());
    tokens.push(output.next_token);
    (logits, tokens)
}

/// The walk runs a part it was handed from outside: its constructor at open, its prompt body
/// over the prompt's rows and its token body over each token step, in every layer whose step
/// names its kind. The part is the catalogue's under another name, so the logits are the
/// catalogue's bit for bit.
#[test]
#[ignore = "boots the arm's fixture; run scripts/onboard-gate.sh dense"]
fn a_part_registered_from_outside_runs_in_the_walk() {
    let path = checkpoint();
    let architecture = Architecture::from_checkpoint_dir(&path).expect("the architecture");
    let source = sparkflash_weights::safetensors::open_checkpoint(&path).expect("the weights");
    let layers = architecture
        .steps()
        .iter()
        .filter(|step| step.roles().iter().any(|(_, part)| *part == Part::DenseMlp))
        .count();
    assert!(layers > 0, "the fixture has a dense MLP to take over");

    let catalogue = run(&architecture, &source, &Dense::catalogue());

    let prompt_rows = Arc::new(AtomicUsize::new(0));
    let token_rows = Arc::new(AtomicUsize::new(0));
    let plugged = run(
        &architecture,
        &source,
        &renamed_parts(&prompt_rows, &token_rows),
    );

    assert_eq!(
        prompt_rows.load(Ordering::Relaxed),
        4 * layers,
        "the prompt body ran the prompt's four rows in every dense-MLP layer"
    );
    assert_eq!(
        token_rows.load(Ordering::Relaxed),
        2 * layers,
        "the token body ran each of the two token steps in every dense-MLP layer"
    );
    assert_eq!(plugged.1, catalogue.1, "the same tokens");
    assert!(
        plugged.0 == catalogue.0,
        "the catalogue's part under another name leaves the catalogue's logits, bit for bit"
    );
}

/// A source with no tensors: the refusal below comes before the arm reads one.
struct NoTensors;

impl WeightSource for NoTensors {
    fn names(&self) -> Vec<String> {
        Vec::new()
    }

    fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
        Err(WeightError::Missing {
            name: name.to_owned(),
        })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
        Err(WeightError::Missing {
            name: name.to_owned(),
        })
    }
}

/// A step whose part has no registered executor is refused by the part's name, before a kernel
/// image or a tensor is read: the walk never skips a part. No device is opened.
#[test]
fn a_part_with_no_executor_is_refused_by_name() {
    let architecture = Architecture::from_json(
        r#"{
            "model_type": "acme_dense",
            "architectures": ["AcmeDenseForCausalLM"],
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "hidden_act": "silu",
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16"
        }"#,
    )
    .expect("the architecture");
    let mut parts = Dense::catalogue();
    parts
        .remove(Part::DenseMlp)
        .expect("the catalogue had the part");
    let descriptor =
        ModelDescriptor::from_architecture("part-plugin", &architecture, 128, Capabilities::ALL);
    let Err(refusal) =
        DenseDeviceBackend::open_with(descriptor, &architecture, &NoTensors, 128, &parts)
    else {
        panic!("the arm opened a walk with a part it has no executor for");
    };
    assert!(
        refusal.contains(&format!(
            "no executor is registered for part {}",
            Part::DenseMlp.name()
        )),
        "the refusal names the part: {refusal}"
    );
    assert!(
        refusal.contains("layer 0"),
        "the refusal names the first layer that needs it: {refusal}"
    );
}
