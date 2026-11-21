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

//! A family whose token mixer is a kind the catalogue has no part for runs on the shared walk.
//!
//! The config declares a latent (MLA-shaped) attention, `Part::MlaAttention`, which the dense
//! host's catalogue does not run. The family registers a part for that kind, and the host then
//! takes the checkpoint: its structure gate, layer shapes and tensor names leave a mixer the
//! family owns to the family's part. The part here is a stand-in (each row normed under the
//! layer's own gains, no state), since the point is the host, not an attention: the generated
//! checks run it on the tick engine like any other family.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_model::backend::BackendError;
use sparkflash_model::{Architecture, Capabilities, ModelDescriptor, Part, TpPlan};
use sparkflash_model_dense::{
    Dense, DenseDeviceBackend, DensePart, DenseParts, DenseWalk, OpenCx, PassCx, ScratchSlot,
};
use sparkflash_runtime::testing::ArmProfile;
use sparkflash_runtime::{
    ArmFixture, Fixture, FormatRule, Limits, PartExec, RankSplit, Region, StateLayout,
};
use sparkflash_weights::{DType, TensorSpec};

/// A two-layer decoder whose attention is latent: `kv_lora_rank` makes each layer an MLA.
const CONFIG: &str = r#"{
    "model_type": "acme_latent",
    "architectures": ["AcmeLatentForCausalLM"],
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 32,
    "v_head_dim": 64,
    "vocab_size": 512,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "silu",
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16"
}"#;

/// The stand-in mixer's gains, under the layer's attention prefix.
fn mixer_gains(layer: usize) -> String {
    format!("model.layers.{layer}.self_attn.mix.weight")
}

/// The family's mixer: each row normed under the layer's gains.
struct LatentMixer {
    gains: Deviceptr,
    normed: ScratchSlot,
}

impl LatentMixer {
    fn open(open: &mut OpenCx<'_>) -> Result<DensePart, String> {
        let gains = open.vector(&mixer_gains(open.layer()), open.hidden())?;
        let normed = open.scratch(open.hidden());
        Ok(Box::new(Self { gains, normed }))
    }
}

impl PartExec<Dense> for LatentMixer {
    fn state(&self) -> &[Region] {
        &[]
    }

    fn split(&self, _plan: &TpPlan) -> RankSplit {
        RankSplit::Replicated
    }

    fn rows(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        let normed = pass.scratch(self.normed)?;
        pass.norm(pass.input(), self.gains, normed)?;
        pass.copy_rows(normed, pass.output())
    }

    fn token(&self, pass: &mut PassCx<'_>) -> Result<(), BackendError> {
        self.rows(pass)
    }
}

/// The registration: the catalogue plus the family's part for the kind it lacks.
fn latent_parts() -> DenseParts {
    let mut parts = Dense::catalogue();
    parts
        .register(Part::MlaAttention, LatentMixer::open)
        .expect("the catalogue has no MLA part");
    parts
}

/// The tiny checkpoint, written once a test process.
fn write_fixture() -> Result<PathBuf, String> {
    static WRITTEN: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    WRITTEN
        .get_or_init(|| {
            let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join("acme-latent");
            let architecture = Architecture::from_json(CONFIG).map_err(|e| e.to_string())?;
            let (hidden, vocab, width) = (architecture.hidden_size, architecture.vocab_size, 512);
            let mut seed = 0x2545_f491_4f6c_dd1d_u64;
            let mut tensor = |name: String, shape: Vec<usize>, scale: f32| {
                let values: usize = shape.iter().product();
                let mut bytes = Vec::with_capacity(values * 2);
                for _ in 0..values {
                    seed = sparkflash_weights::rng::splitmix64(seed);
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
            let projection = 1.0 / (hidden as f32).sqrt();
            let mut tensors = vec![
                tensor("model.embed_tokens.weight".into(), vec![vocab, hidden], 1.0),
                tensor("lm_head.weight".into(), vec![vocab, hidden], 0.5),
                tensor("model.norm.weight".into(), vec![hidden], 1.0),
            ];
            for layer in 0..architecture.layers.len() {
                let p = format!("model.layers.{layer}");
                tensors.extend([
                    tensor(format!("{p}.input_layernorm.weight"), vec![hidden], 1.0),
                    tensor(
                        format!("{p}.post_attention_layernorm.weight"),
                        vec![hidden],
                        1.0,
                    ),
                    tensor(mixer_gains(layer), vec![hidden], 1.0),
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
        "acme-latent",
        &architecture,
        context,
        Capabilities::ALL,
    );
    let backend = DenseDeviceBackend::open_with(descriptor, &architecture, &source, context, parts)
        .map_err(fixture)?;
    Ok((backend, architecture))
}

struct LatentFixture;

impl ArmFixture for LatentFixture {
    type Walk = DenseWalk;

    fn fixture(world: usize) -> Result<Fixture<DenseWalk>, BackendError> {
        if world > 1 {
            return Err(BackendError::new(
                "group",
                "the stand-in mixer is stated at TP 1",
            ));
        }
        let (backend, architecture) = open(&latent_parts())?;
        let bytes = backend.weight_bytes() as u64;
        Ok(Fixture::new(
            backend.walk_arm(64)?,
            "the Acme Latent tiny checkpoint",
            bytes,
            bytes,
            FormatRule::classes(&architecture.quantization),
            None,
        ))
    }

    fn tick_engine() -> bool {
        true
    }
}

fn stated() -> ArmProfile {
    let limits = Limits::new(256, 32_768);
    ArmProfile::new("dense", StateLayout::new(Vec::new()), limits)
        .with_code("dense_device")
        .prompt(limits, 600, &[256, 256, 88])
}

sparkflash_runtime::conformance!(LatentFixture, profile: stated());

/// Without the family's part the host refuses the latent mixer by name, before a tensor is read.
#[test]
fn the_catalogue_alone_refuses_the_latent_mixer() {
    let architecture = Architecture::from_json(CONFIG).expect("the config parses");
    let refusal = sparkflash_ops::parts::dense_device_refusal(&architecture)
        .expect("the catalogue does not run an MLA");
    assert!(refusal.contains("mla_attention"), "{refusal}");
    assert_eq!(
        sparkflash_ops::parts::dense_device_refusal_hosting(&architecture, &[Part::MlaAttention]),
        None,
        "a family that owns the mixer takes the layer"
    );
}
