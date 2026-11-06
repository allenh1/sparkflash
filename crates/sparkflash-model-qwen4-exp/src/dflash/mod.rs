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

//! The DFlash 2 drafter for incoai/Qwen3.8-Flash-Next-DFlash2: the shared configuration
//! ([sparkflash_model::dflash2]) and this target's [DraftTarget].
//!
//! The device arm runs it from [crate::device]. The drafter captures target layers
//! 5/15/25/35/45 as the mean of the four hyper-connection streams.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

pub use sparkflash_model::dflash2::*;
use sparkflash_weights::hub;
use sparkflash_weights::safetensors::write_checkpoint;
use sparkflash_weights::{DType, TensorSpec};

use crate::config::Qwen4ExpConfig;

/// The drafter's canonical Hugging Face repository id.
pub const DFLASH2_MODEL_ID: &str = "incoai/Qwen3.8-Flash-Next-DFlash2";

/// The drafter directory the device arm should load.
///
/// `SPARKFLASH_QWEN4_DFLASH2` still wins when set: a path that exists is used as-is, and a repo id
/// or a missing path is looked up in the Hugging Face cache. When it is unset, the canonical
/// `incoai/Qwen3.8-Flash-Next-DFlash2` snapshot in the cache is used. That keeps the normal case
/// to `hf download incoai/Qwen3.8-Flash-Next-DFlash2` with no path in the environment; `None`
/// means no drafter is configured, so the MTP head serves instead.
#[must_use]
pub fn drafter_directory() -> Option<PathBuf> {
    let profile = sparkflash_model::active_profile();
    let configured = profile.speculation.drafter.as_deref();
    resolve_drafter_dir(configured.map(OsStr::new))
}

fn resolve_drafter_dir(value: Option<&OsStr>) -> Option<PathBuf> {
    let cache = hub::default_cache_dir();
    resolve_drafter_dir_in(cache.as_deref(), value)
}

fn resolve_drafter_dir_in(cache: Option<&Path>, value: Option<&OsStr>) -> Option<PathBuf> {
    if let Some(value) = value.filter(|value| !value.is_empty()) {
        let path = PathBuf::from(value);
        if path.exists() {
            return Some(path);
        }
        if let Some(snapshot) =
            cache.and_then(|cache| hub::cached_model_in(cache, &path.to_string_lossy()))
        {
            return Some(snapshot);
        }
        return Some(path);
    }
    cache.and_then(|cache| hub::cached_model_in(cache, DFLASH2_MODEL_ID))
}

impl From<&Qwen4ExpConfig> for DraftTarget {
    fn from(target: &Qwen4ExpConfig) -> Self {
        Self {
            hidden_size: target.hidden_size,
            vocab_size: target.vocab_size,
            num_hidden_layers: target.num_hidden_layers,
        }
    }
}

/// Every tensor a drafter of `config` stores, at the released names, shapes and dtype (BF16):
/// the ones [crate::device::GpuDflash2::load] reads.
#[must_use]
pub fn drafter_tensor_specs(config: &Dflash2Config) -> Vec<TensorSpec> {
    let hidden = config.hidden_size;
    let spec = |name: String, shape: Vec<usize>| TensorSpec {
        name,
        dtype: DType::BF16,
        shape,
    };
    let mut specs = vec![
        spec(
            "candidate_selector.hidden_projection.weight".to_owned(),
            vec![config.selector_rank, hidden],
        ),
        spec(
            "candidate_selector.predecessor_codebook".to_owned(),
            vec![config.vocab_size, config.selector_rank],
        ),
        spec(
            "candidate_selector.successor_codebook".to_owned(),
            vec![config.vocab_size, config.selector_rank],
        ),
        spec("fc.weight".to_owned(), vec![hidden, config.feature_width()]),
        spec("hidden_norm.weight".to_owned(), vec![hidden]),
        spec("norm.weight".to_owned(), vec![hidden]),
    ];
    for layer in 0..config.num_hidden_layers {
        let at = |name: &str| format!("layers.{layer}.{name}");
        for conv in ["attention_conv", "mlp_conv"] {
            specs.push(spec(
                at(&format!("{conv}.base_kernel")),
                vec![2, config.conv_kernel_size, hidden],
            ));
            specs.push(spec(
                at(&format!("{conv}.kernel_projection.weight")),
                vec![config.conv_taps_width(), hidden],
            ));
        }
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            specs.push(spec(at(&format!("{norm}.weight")), vec![hidden]));
        }
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![config.q_width(), hidden]),
            ("self_attn.k_proj.weight", vec![config.kv_width(), hidden]),
            ("self_attn.v_proj.weight", vec![config.kv_width(), hidden]),
            ("self_attn.o_proj.weight", vec![hidden, config.q_width()]),
            ("self_attn.q_norm.weight", vec![config.head_dim]),
            ("self_attn.k_norm.weight", vec![config.head_dim]),
            (
                "mlp.gate_proj.weight",
                vec![config.intermediate_size, hidden],
            ),
            ("mlp.up_proj.weight", vec![config.intermediate_size, hidden]),
            (
                "mlp.down_proj.weight",
                vec![hidden, config.intermediate_size],
            ),
        ] {
            specs.push(spec(at(name), shape));
        }
    }
    specs
}

/// Write a synthetic drafter into `directory`: `config_json` as its `config.json` and every
/// tensor of [drafter_tensor_specs] in one `model.safetensors`, valued as the target fixtures
/// are ([crate::checkpoint::write_synthetic_checkpoint]).
///
/// # Errors
/// A document that is not a drafter this arm runs, or a file that cannot be written.
pub fn write_synthetic_drafter(
    directory: &Path,
    config_json: &str,
    seed: u64,
) -> Result<Dflash2Config, String> {
    let config = Dflash2Config::from_json(config_json).map_err(|error| error.to_string())?;
    let tensors = drafter_tensor_specs(&config)
        .into_iter()
        .map(|spec| {
            let mut bytes = crate::checkpoint::synthetic_bytes(&spec, seed)?;
            // The target's norms add one to their weight and the drafter's multiply by it, so the
            // drafter's gains move to one: gains near zero shrink every sublayer's input until the
            // mask token's embedding alone picks the drafts, the same token at every position.
            if spec.name.contains("norm") && spec.name.ends_with(".weight") {
                for value in bytes.as_chunks_mut::<2>().0 {
                    let gain = half::bf16::from_le_bytes(*value).to_f32();
                    *value = half::bf16::from_f32(1.0 + gain).to_le_bytes();
                }
            }
            Ok((spec, bytes))
        })
        .collect::<Result<Vec<_>, String>>()?;
    write_checkpoint(directory, &tensors, 1).map_err(|error| error.to_string())?;
    std::fs::write(directory.join("config.json"), config_json)
        .map_err(|error| format!("write the drafter config: {error}"))?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The carried target is the geometry the released drafter was trained for, which
    /// [sparkflash_model::dflash2]'s tests check the released drafter against.
    #[test]
    fn the_carried_target_is_the_released_drafters() {
        let target = Qwen4ExpConfig::qwen38_flash_next_nvfp4().expect("the carried config parses");
        assert_eq!(
            DraftTarget::from(&target),
            DraftTarget {
                hidden_size: 2560,
                vocab_size: 248_320,
                num_hidden_layers: 48,
            }
        );
    }

    #[test]
    fn a_drafter_resolves_from_an_override_or_the_canonical_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let repo = dir
            .path()
            .join("models--incoai--Qwen3.8-Flash-Next-DFlash2");
        let snapshot = repo.join("snapshots").join("rev");
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        std::fs::create_dir_all(repo.join("refs")).expect("refs");
        std::fs::write(repo.join("refs").join("main"), "rev").expect("ref");
        assert_eq!(
            resolve_drafter_dir_in(Some(dir.path()), None),
            Some(snapshot.clone())
        );

        let custom = dir.path().join("custom-drafter");
        std::fs::create_dir_all(&custom).expect("custom");
        assert_eq!(
            resolve_drafter_dir_in(Some(dir.path()), Some(custom.as_os_str())),
            Some(custom)
        );
        assert_eq!(
            resolve_drafter_dir_in(
                Some(dir.path()),
                Some(std::ffi::OsStr::new(DFLASH2_MODEL_ID))
            ),
            Some(snapshot)
        );

        let missing = dir.path().join("not-there");
        assert_eq!(
            resolve_drafter_dir_in(Some(dir.path()), Some(missing.as_os_str())),
            Some(missing)
        );
        assert_eq!(resolve_drafter_dir_in(None, None), None);
    }
}
