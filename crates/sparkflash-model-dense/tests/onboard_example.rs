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

//! The worked example of onboarding a model family: everything a family with one part of its
//! own writes, in one file that is its own crate.
//!
//! The family is "Acme Sandwich", a made-up checkpoint: a Llama-shaped decoder whose feed-forward
//! is normed again after it (the sandwich norm some released families use) under a tensor of its
//! own, `model.layers.N.post_feedforward_layernorm.weight`. Everything else is catalogue parts.
//! The file supplies, in order:
//!
//! 1. the checkpoint's `config.json` ([CONFIG]), which `Architecture` maps with no code;
//! 2. the one part of its own, [SandwichMlp], a [PartExec] over the shared dense host;
//! 3. its registration, one `replace` on the catalogue registry ([acme_parts]);
//! 4. a tiny fixture checkpoint written from the config ([write_fixture]);
//! 5. the fixture and the stated profile the generated checks run under ([AcmeFixture],
//!    [stated]), and the `conformance!` line.
//!
//! No walk, sequence, lane, scheduler, head or sampler is written: the runtime's
//! [sparkflash_runtime::PlanWalk] runs the family, and the generated checks drive it on the tick
//! engine the arm serves on.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_model::backend::{BackendError, ModelSession};
use sparkflash_model::{Architecture, Capabilities, ModelDescriptor, Part, TpPlan};
use sparkflash_model_dense::{
    Dense, DenseDeviceBackend, DensePart, DenseParts, DenseWalk, OpenCx, PassCx, ScratchSlot,
};
use sparkflash_runtime::testing::ArmProfile;
use sparkflash_runtime::{
    ArmFixture, Fixture, FormatRule, Limits, PartExec, RankSplit, Region, StateLayout,
};
use sparkflash_weights::{DType, TensorSpec};

/// The family's checkpoint config (step 1), as its repository ships it.
const CONFIG: &str = r#"{
    "model_type": "acme_sandwich",
    "architectures": ["AcmeSandwichForCausalLM"],
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 512,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16"
}"#;

/// The family's own tensor: each layer's norm after its feed-forward.
fn sandwich_norm(layer: usize) -> String {
    format!("model.layers.{layer}.post_feedforward_layernorm.weight")
}

/// The family's part (step 2): the catalogue's SwiGLU MLP, its output normed again under the
/// layer's own gains before the residual takes it.
struct SandwichMlp {
    mlp: DensePart,
    gains: Deviceptr,
    normed: ScratchSlot,
}

impl SandwichMlp {
    /// Opens the part for one layer. A rank of a group would hold a partial of the MLP's output,
    /// which a norm cannot take before the group's sum, so a group is refused here.
    fn open(open: &mut OpenCx<'_>) -> Result<DensePart, String> {
        if open.plan().world() > 1 {
            return Err(format!(
                "the sandwich norm reads whole output rows, and a rank of {} holds a partial",
                open.plan().world()
            ));
        }
        let gains = open.vector(&sandwich_norm(open.layer()), open.hidden())?;
        let normed = open.scratch(open.hidden());
        Ok(Box::new(Self {
            mlp: open.dense_mlp()?,
            gains,
            normed,
        }))
    }

    /// The MLP's body, then the norm of its rows back into the part's output.
    fn after(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        let normed = pass.scratch(self.normed)?;
        pass.norm(pass.output(), self.gains, normed)?;
        pass.copy_rows(normed, pass.output())
    }
}

impl PartExec<Dense> for SandwichMlp {
    fn state(&self) -> &[Region] {
        &[]
    }

    /// Every rank holds the whole part: [Self::open] refuses a group.
    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Replicated
    }

    fn rows(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        self.mlp.rows(pass)?;
        self.after(pass)
    }

    fn token(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        self.mlp.token(pass)?;
        self.after(pass)
    }
}

/// The registration (step 3): the catalogue, with the family's part as its feed-forward.
fn acme_parts() -> DenseParts {
    let mut parts = Dense::catalogue();
    parts
        .replace(Part::DenseMlp, SandwichMlp::open)
        .expect("the catalogue has a dense MLP to take over");
    parts
}

/// The tiny fixture checkpoint (step 4): the config and seeded BF16 weights under the family's
/// names, written once a test process into the target's scratch directory.
fn write_fixture() -> Result<PathBuf, String> {
    static WRITTEN: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    WRITTEN
        .get_or_init(|| {
            let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join("acme-sandwich");
            let architecture = Architecture::from_json(CONFIG).map_err(|e| e.to_string())?;
            let (hidden, vocab) = (architecture.hidden_size, architecture.vocab_size);
            let (heads, kv, dim, width) = (4usize, 2usize, 64usize, 512usize);
            let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
            let mut tensor = |name: String, shape: Vec<usize>, scale: f32| {
                let values: usize = shape.iter().product();
                let mut bytes = Vec::with_capacity(values * 2);
                for _ in 0..values {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let unit = (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                    let bits = (unit * scale).to_bits();
                    bytes.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
                }
                let spec = TensorSpec {
                    name,
                    dtype: DType::BF16,
                    shape,
                };
                (spec, bytes)
            };
            let ones = |name: String| {
                let bytes = std::iter::repeat_n(0x3f80u16.to_le_bytes(), hidden)
                    .flatten()
                    .collect();
                let spec = TensorSpec {
                    name,
                    dtype: DType::BF16,
                    shape: vec![hidden],
                };
                (spec, bytes)
            };
            let mut tensors = vec![
                tensor("model.embed_tokens.weight".into(), vec![vocab, hidden], 1.0),
                tensor("lm_head.weight".into(), vec![vocab, hidden], 0.5),
                ones("model.norm.weight".into()),
            ];
            for layer in 0..architecture.layers.len() {
                let p = format!("model.layers.{layer}");
                let projection = 1.0 / (hidden as f32).sqrt();
                tensors.extend([
                    ones(format!("{p}.input_layernorm.weight")),
                    ones(format!("{p}.post_attention_layernorm.weight")),
                    ones(sandwich_norm(layer)),
                    tensor(
                        format!("{p}.self_attn.q_proj.weight"),
                        vec![heads * dim, hidden],
                        projection,
                    ),
                    tensor(
                        format!("{p}.self_attn.k_proj.weight"),
                        vec![kv * dim, hidden],
                        projection,
                    ),
                    tensor(
                        format!("{p}.self_attn.v_proj.weight"),
                        vec![kv * dim, hidden],
                        projection,
                    ),
                    tensor(
                        format!("{p}.self_attn.o_proj.weight"),
                        vec![hidden, heads * dim],
                        projection,
                    ),
                    tensor(
                        format!("{p}.mlp.gate_proj.weight"),
                        vec![width, hidden],
                        projection,
                    ),
                    tensor(
                        format!("{p}.mlp.up_proj.weight"),
                        vec![width, hidden],
                        projection,
                    ),
                    tensor(
                        format!("{p}.mlp.down_proj.weight"),
                        vec![hidden, width],
                        projection,
                    ),
                ]);
            }
            sparkflash_weights::safetensors::write_checkpoint(&directory, &tensors, 1)
                .map_err(|error| error.to_string())?;
            std::fs::write(directory.join("config.json"), CONFIG).map_err(|e| e.to_string())?;
            Ok(directory)
        })
        .clone()
}

/// The family's arm over the fixture, with `parts` as its executors.
fn open(parts: &DenseParts) -> Result<(DenseDeviceBackend, Architecture), BackendError> {
    let fixture = |error: String| BackendError::new("fixture", error);
    let path = write_fixture().map_err(fixture)?;
    let architecture =
        Architecture::from_checkpoint_dir(&path).map_err(|e| fixture(e.to_string()))?;
    let source = sparkflash_weights::safetensors::open_checkpoint(&path)
        .map_err(|error| fixture(error.to_string()))?;
    let context = architecture.max_position_embeddings;
    let descriptor = ModelDescriptor::from_architecture(
        "acme-sandwich",
        &architecture,
        context,
        Capabilities::ALL,
    );
    let backend = DenseDeviceBackend::open_with(descriptor, &architecture, &source, context, parts)
        .map_err(fixture)?;
    Ok((backend, architecture))
}

/// The fixture the generated checks boot (step 5): the family's arm at one rank.
struct AcmeFixture;

impl ArmFixture for AcmeFixture {
    type Walk = DenseWalk;

    fn fixture(world: usize) -> Result<Fixture<DenseWalk>, BackendError> {
        if world > 1 {
            return Err(BackendError::new(
                "group",
                "the sandwich norm reads whole output rows, so the family serves at TP 1",
            ));
        }
        let (backend, architecture) = open(&acme_parts())?;
        let bytes = backend.weight_bytes() as u64;
        Ok(Fixture::new(
            backend.walk_arm(64)?,
            "the Acme Sandwich tiny checkpoint",
            bytes,
            bytes,
            FormatRule::classes(&architecture.quantization),
            None,
        ))
    }

    /// The family serves on the tick engine, as the dense host does.
    fn tick_engine() -> bool {
        true
    }
}

/// What the family states for the generated profile tests: the dense host's ledger row, its
/// pass limits, a positional cache, and the chunks a long prompt runs in.
fn stated() -> ArmProfile {
    let limits = Limits::new(256, 32_768);
    ArmProfile::new(
        "dense",
        StateLayout::new(vec![Region::Positional {
            name: "kv",
            bytes_per_row: 4,
        }]),
        limits,
    )
    .with_code("dense_device")
    .prompt(limits, 600, &[256, 256, 88])
}

sparkflash_runtime::conformance!(AcmeFixture, profile: stated());

/// The family's part runs: the same checkpoint under the plain catalogue (which reads no
/// sandwich norm) leaves other logits, so the generated checks above compared the family's
/// arithmetic and not the catalogue's.
#[test]
#[ignore = "boots the family's fixture on the device"]
fn the_family_part_changes_the_logits() {
    let logits = |parts: &DenseParts| {
        let (backend, _) = open(parts).expect("the arm opens");
        let mut session = backend.session().expect("a session");
        let output = session.prefill(&[7, 9, 11, 13]).expect("the prompt");
        output
            .logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_ne!(
        logits(&acme_parts()),
        logits(&Dense::catalogue()),
        "the sandwich norm moved the logits"
    );
}
