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

//! Checkpoint plumbing shared by the host oracle and the device arm.
//!
//! The device arm uploads a checkpoint that this module can also read on the host, which is what
//! makes a parity test possible at all: both sides open the same safetensors index, address the
//! same PLE shard rows through [CheckpointRows], and decode the same NVFP4 bytes. Nothing here
//! re-derives a name from a second table; every name that matters is either spelled once in
//! [crate::device] or once here, never both.
//!
//! Three pieces live here:
//!
//! * [CheckpointRows] pages one PLE layer's shards, and is used by the host table and the device
//!   table alike so a row addresses the same bytes in both.
//! * [load_weights] builds the host reference weights ([crate::forward::Qwen4ExpWeights]) from a
//!   [sparkflash_weights::WeightSource]. It is the oracle the device arithmetic is compared
//!   against, so it dequantises rather than high-precision re-implementing.
//! * [write_synthetic_checkpoint] writes a complete, shard-shaped checkpoint for a configuration.
//!   The tiny fixture is two layers and a few megabytes, so the whole device loop - upload,
//!   conformance, PLE paging, twenty kernels - runs on a workstation instead of a 128 GiB node.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sparkflash_weights::DType;
use sparkflash_weights::WeightSource;
use sparkflash_weights::nvfp4::dequantize_nvfp4;
use sparkflash_weights::rng::TensorRng;
use sparkflash_weights::safetensors::{SafeTensorsSource, write_checkpoint};

use crate::config::Qwen4ExpConfig;
use crate::forward::{
    BranchWeights, Expert, HyperBlock, LayerWeights, Mat, MoeWeights, PleWeights, Qwen4ExpWeights,
};
use crate::ple::{PleTable, RowReader, shard_tensor_name};
use crate::tensor::Qwen4ExpTensorPlan;

pub use sparkflash_weights::safetensors::INDEX_NAME;

/// The per-tensor scale a synthetic NVFP4 expert carries.
const SYNTHETIC_SCALE_2: f32 = 0.02;

/// The per-tensor scale of a synthetic PLE table.
const SYNTHETIC_PLE_SCALE: f32 = 0.05;

/// The magnitude of a synthetic matrix entry.
const SYNTHETIC_WEIGHT: f32 = 0.05;

/// Pre-resolved file handle and byte range for one PLE shard tensor.
#[derive(Debug, Clone)]
struct ResolvedShard {
    file: Arc<std::fs::File>,
    data_offset: u64,
    byte_len: u64,
}

/// A paged reader over one PLE layer's safetensors shards.
///
/// Both the host table and the device table wrap this, so a row requested by the hasher lands on
/// the same bytes whichever side runs the model. Shards are resolved to (Arc<File>, offset, len)
/// at open time and advised with POSIX_FADV_RANDOM so 160-byte n-gram row reads do not trigger
/// 128 KiB kernel readahead or contend on header/index mutexes.
#[derive(Debug)]
pub struct CheckpointRows {
    source: Arc<SafeTensorsSource>,
    layer: usize,
    shard_rows: u64,
    shards: Vec<Option<ResolvedShard>>,
}

impl CheckpointRows {
    /// Open one PLE layer's shards from a checkpoint directory.
    ///
    /// # Errors
    /// When the index cannot be read, or the configuration's table does not divide into the
    /// shards the checkpoint declares.
    pub fn open(directory: &Path, config: &Qwen4ExpConfig, layer: usize) -> Result<Self, String> {
        let source = Arc::new(
            SafeTensorsSource::open(directory, &directory.join(INDEX_NAME))
                .map_err(|error| error.to_string())?,
        );
        let shard_rows = config.ple_table_rows() / config.split_ngram_parts as u64;
        if shard_rows == 0 {
            return Err("the PLE table has no rows per shard".to_owned());
        }
        let mut files: std::collections::BTreeMap<String, Arc<std::fs::File>> =
            std::collections::BTreeMap::new();
        let mut shards = Vec::with_capacity(config.split_ngram_parts);
        for shard in 0..config.split_ngram_parts {
            let name = shard_tensor_name(layer, shard);
            let resolved = (|| -> Option<ResolvedShard> {
                let shard_file = source.index().shard_of(&name).ok()?;
                let header = source.header(&name).ok()?;
                let tensor = header.tensors.get(&name)?;
                let file = if let Some(existing) = files.get(shard_file) {
                    Arc::clone(existing)
                } else {
                    let opened = std::fs::File::open(source.shard_path(shard_file)).ok()?;
                    #[cfg(target_os = "linux")]
                    unsafe {
                        use std::os::unix::io::AsRawFd;
                        unsafe extern "C" {
                            fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
                        }
                        const POSIX_FADV_RANDOM: i32 = 1;
                        let _ = posix_fadvise(opened.as_raw_fd(), 0, 0, POSIX_FADV_RANDOM);
                    }
                    let arc = Arc::new(opened);
                    files.insert(shard_file.to_owned(), Arc::clone(&arc));
                    arc
                };
                let [begin, end] = tensor.data_offsets;
                Some(ResolvedShard {
                    file,
                    data_offset: header.data_start.checked_add(begin)?,
                    byte_len: end.checked_sub(begin)?,
                })
            })();
            shards.push(resolved);
        }
        Ok(Self {
            source,
            layer,
            shard_rows,
            shards,
        })
    }

    /// The checkpoint-relative name of one shard.
    #[must_use]
    pub fn tensor_name(&self, shard: usize) -> String {
        shard_tensor_name(self.layer, shard)
    }
}

impl RowReader for CheckpointRows {
    fn read_rows(&self, shard: usize, ranges: &[(u64, usize)]) -> Result<Vec<u8>, String> {
        if let Some(Some(resolved)) = self.shards.get(shard) {
            use std::os::unix::fs::FileExt;
            let total_bytes: usize = ranges.iter().map(|(_, len)| *len).sum();
            let mut out = vec![0u8; total_bytes];
            let mut cursor = 0usize;
            for &(offset, len) in ranges {
                let end = offset
                    .checked_add(len as u64)
                    .ok_or_else(|| format!("PLE shard {shard} range overflow"))?;
                if end > resolved.byte_len {
                    return Err(format!(
                        "PLE shard {shard} range {offset}..{end} exceeds shard length {}",
                        resolved.byte_len
                    ));
                }
                if len > 0 {
                    resolved
                        .file
                        .read_exact_at(
                            &mut out[cursor..cursor + len],
                            resolved.data_offset + offset,
                        )
                        .map_err(|error| format!("PLE shard {shard} pread: {error}"))?;
                }
                cursor += len;
            }
            return Ok(out);
        }
        let name = self.tensor_name(shard);
        self.source
            .read_ranges_paged(&name, ranges)
            .map_err(|error| error.to_string())
    }

    fn read_shards(
        &self,
        requests: &[(usize, Vec<(u64, usize)>)],
    ) -> Result<Vec<(usize, Vec<u8>)>, String> {
        if requests.len() <= 1 {
            return requests
                .iter()
                .map(|(shard, ranges)| Ok((*shard, self.read_rows(*shard, ranges)?)))
                .collect();
        }
        let workers = requests.len().min(16);
        let next = std::sync::atomic::AtomicUsize::new(0);
        type ShardReadSlot = std::sync::Mutex<Option<Result<(usize, Vec<u8>), String>>>;
        let slots: Vec<ShardReadSlot> = (0..requests.len())
            .map(|_| std::sync::Mutex::new(None))
            .collect();
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let idx = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if idx >= requests.len() {
                            break;
                        }
                        let (shard, ranges) = &requests[idx];
                        let res = self.read_rows(*shard, ranges).map(|bytes| (*shard, bytes));
                        if let Ok(mut slot) = slots[idx].lock() {
                            *slot = Some(res);
                        }
                    }
                });
            }
        });
        slots
            .into_iter()
            .map(|slot| {
                slot.into_inner()
                    .map_err(|error| error.to_string())?
                    .unwrap_or_else(|| Err("PLE shard worker did not complete".to_owned()))
            })
            .collect()
    }

    fn shard_rows(&self) -> u64 {
        self.shard_rows
    }
}

/// Open the paged PLE table of one layer from a checkpoint directory.
///
/// The table's per-tensor FP8 scale is read from the checkpoint rather than assumed, because the
/// released checkpoint carries it as a BF16 scalar next to the shards.
///
/// # Errors
/// When the index or the scale cannot be read, or the shards do not match the configuration.
pub fn open_ple_table(
    directory: &Path,
    config: &Qwen4ExpConfig,
    layer: usize,
) -> Result<PleTable<CheckpointRows>, String> {
    let source = SafeTensorsSource::open(directory, &directory.join(INDEX_NAME))
        .map_err(|error| error.to_string())?;
    let name = format!(
        "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.weight_scale"
    );
    let bytes = source.read(&name).map_err(|error| error.to_string())?;
    let scale = bf16_scalar(&bytes, &name)?;
    let rows = CheckpointRows::open(directory, config, layer)?;
    PleTable::new(config, rows, scale)
}

fn bf16_scalar(bytes: &[u8], name: &str) -> Result<f32, String> {
    let array = <[u8; 2]>::try_from(bytes)
        .map_err(|_| format!("{name} is not one BF16 ({} bytes)", bytes.len()))?;
    Ok(half::bf16::from_le_bytes(array).to_f32())
}

/// A reader over one [WeightSource], with the shapes the model expects.
struct Reader<'a> {
    source: &'a dyn WeightSource,
}

impl Reader<'_> {
    fn bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        self.source.read(name).map_err(|error| error.to_string())
    }

    fn mat(&self, name: &str, rows: usize, columns: usize) -> Result<Mat, String> {
        let bytes = self.bytes(name)?;
        if bytes.len() != rows * columns * 2 {
            return Err(format!(
                "{name} is {} bytes, expected {} for a BF16 [{rows}, {columns}]",
                bytes.len(),
                rows * columns * 2
            ));
        }
        let data = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| half::bf16::from_le_bytes([pair[0], pair[1]]).to_f32())
            .collect();
        Ok(Mat::new(rows, columns, data))
    }

    fn vector(&self, name: &str, elements: usize) -> Result<Vec<f32>, String> {
        let bytes = self.bytes(name)?;
        if bytes.len() != elements * 2 {
            return Err(format!(
                "{name} is {} bytes, expected {} for {elements} BF16 values",
                bytes.len(),
                elements * 2
            ));
        }
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| half::bf16::from_le_bytes([pair[0], pair[1]]).to_f32())
            .collect())
    }

    fn f32_scalar(&self, name: &str) -> Result<f32, String> {
        let bytes = self.bytes(name)?;
        let array = <[u8; 4]>::try_from(bytes.as_slice())
            .map_err(|_| format!("{name} is not one F32 ({} bytes)", bytes.len()))?;
        Ok(f32::from_le_bytes(array))
    }

    fn nvfp4(&self, prefix: &str, rows: usize, columns: usize) -> Result<Mat, String> {
        let weight = self.bytes(&format!("{prefix}.weight"))?;
        let scales = self.bytes(&format!("{prefix}.weight_scale"))?;
        let scale_2 = self.f32_scalar(&format!("{prefix}.weight_scale_2"))?;
        let data = dequantize_nvfp4(&weight, &scales, scale_2, rows, columns)
            .map_err(|error| error.to_string())?;
        Ok(Mat::new(rows, columns, data))
    }
}

/// Build the host reference weights from a checkpoint source.
///
/// This mirrors the device upload tensor for tensor, which is the point: the two sides differ in
/// where the arithmetic runs, not in which tensors exist or how a quantised one decodes.
///
/// # Errors
/// When a tensor is missing, truncated, or not the shape the configuration implies.
pub fn load_weights(
    config: &Qwen4ExpConfig,
    source: &dyn WeightSource,
) -> Result<Qwen4ExpWeights, String> {
    let reader = Reader { source };
    let hidden = config.hidden_size;
    let width = config.hyper_width();
    let hyper = |prefix: &str, inject: bool| -> Result<HyperBlock, String> {
        Ok(HyperBlock {
            norm: reader.vector(&format!("{prefix}.hc_norm.weight"), width)?,
            down: reader.mat(
                &format!("{prefix}.input_mix_weight_down.weight"),
                config.hc_lowrank,
                width,
            )?,
            up: reader.mat(
                &format!("{prefix}.input_mix_weight_up.weight"),
                width,
                config.hc_lowrank,
            )?,
            inject: if inject {
                Some(reader.mat(
                    &format!("{prefix}.block_inject_weight.weight"),
                    config.hc_count,
                    width,
                )?)
            } else {
                None
            },
        })
    };
    let embed = reader.mat(
        "model.language_model.embed_tokens.weight",
        config.vocab_size,
        hidden,
    )?;
    let lm_head = reader.mat("lm_head.weight", config.vocab_size, hidden)?;
    let mixer = hyper("model.language_model.hyper_connection_mixer", false)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for (index, kind) in config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        let branch = match kind {
            crate::config::LayerKind::Linear => {
                let prefix = format!("{prefix}.linear_attn");
                BranchWeights::Linear {
                    in_qkv: reader.mat(
                        &format!("{prefix}.in_proj_qkv.weight"),
                        config.linear_conv_width(),
                        hidden,
                    )?,
                    in_z: reader.mat(
                        &format!("{prefix}.in_proj_z.weight"),
                        config.linear_value_width(),
                        hidden,
                    )?,
                    in_b: reader.mat(
                        &format!("{prefix}.in_proj_b.weight"),
                        config.linear_num_value_heads,
                        hidden,
                    )?,
                    in_a: reader.mat(
                        &format!("{prefix}.in_proj_a.weight"),
                        config.linear_num_value_heads,
                        hidden,
                    )?,
                    out_proj: reader.mat(
                        &format!("{prefix}.out_proj.weight"),
                        hidden,
                        config.linear_value_width(),
                    )?,
                    conv: reader.vector(
                        &format!("{prefix}.conv1d.weight"),
                        config.linear_conv_width() * config.linear_conv_kernel_dim,
                    )?,
                    a_log: reader
                        .vector(&format!("{prefix}.A_log"), config.linear_num_value_heads)?,
                    dt_bias: reader
                        .vector(&format!("{prefix}.dt_bias"), config.linear_num_value_heads)?,
                    norm: reader.vector(
                        &format!("{prefix}.norm.weight"),
                        config.linear_value_head_dim,
                    )?,
                }
            }
            crate::config::LayerKind::Sparse => {
                let prefix = format!("{prefix}.self_attn");
                BranchWeights::Sparse {
                    q_proj: reader.mat(
                        &format!("{prefix}.q_proj.weight"),
                        config.q_proj_width(),
                        hidden,
                    )?,
                    k_proj: reader.mat(
                        &format!("{prefix}.k_proj.weight"),
                        config.kv_proj_width(),
                        hidden,
                    )?,
                    v_proj: reader.mat(
                        &format!("{prefix}.v_proj.weight"),
                        config.kv_proj_width(),
                        hidden,
                    )?,
                    o_proj: reader.mat(
                        &format!("{prefix}.o_proj.weight"),
                        hidden,
                        config.attn_out_width(),
                    )?,
                    q_norm: reader.vector(&format!("{prefix}.q_norm.weight"), config.head_dim)?,
                    k_norm: reader.vector(&format!("{prefix}.k_norm.weight"), config.head_dim)?,
                    index_qk: reader.mat(
                        &format!("{prefix}.indexer.index_qk_proj.weight"),
                        (config.indexer_n_heads + config.indexer_kv_heads)
                            * config.indexer_head_dim,
                        hidden,
                    )?,
                    index_q_norm: reader.vector(
                        &format!("{prefix}.indexer.q_layernorm.weight"),
                        config.indexer_head_dim,
                    )?,
                    index_k_norm: reader.vector(
                        &format!("{prefix}.indexer.k_layernorm.weight"),
                        config.indexer_head_dim,
                    )?,
                }
            }
        };
        let mlp = format!("{prefix}.mlp");
        let shared = |name: &str, rows: usize, columns: usize| -> Result<Mat, String> {
            reader.mat(&format!("{mlp}.shared_expert.{name}"), rows, columns)
        };
        let moe = MoeWeights {
            router: reader.mat(&format!("{mlp}.gate.weight"), config.num_experts, hidden)?,
            shared: Expert {
                gate: shared(
                    "gate_proj.weight",
                    config.shared_expert_intermediate_size,
                    hidden,
                )?,
                up: shared(
                    "up_proj.weight",
                    config.shared_expert_intermediate_size,
                    hidden,
                )?,
                down: shared(
                    "down_proj.weight",
                    hidden,
                    config.shared_expert_intermediate_size,
                )?,
            },
            shared_gate: reader.mat(&format!("{mlp}.shared_expert_gate.weight"), 1, hidden)?,
            experts: (0..config.num_experts)
                .map(|expert| {
                    let prefix = format!("{mlp}.experts.{expert}");
                    Ok(Expert {
                        gate: reader.nvfp4(
                            &format!("{prefix}.gate_proj"),
                            config.moe_intermediate_size,
                            hidden,
                        )?,
                        up: reader.nvfp4(
                            &format!("{prefix}.up_proj"),
                            config.moe_intermediate_size,
                            hidden,
                        )?,
                        down: reader.nvfp4(
                            &format!("{prefix}.down_proj"),
                            hidden,
                            config.moe_intermediate_size,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        };
        let ple = if config.ple_layer_ids.contains(&(index + 1)) {
            let prefix = format!("{prefix}.ple");
            Some(PleWeights {
                key_proj: reader.mat(
                    &format!("{prefix}.key_proj.weight"),
                    width,
                    config.ple_embed_dim,
                )?,
                value_proj: reader.mat(
                    &format!("{prefix}.value_proj.weight"),
                    hidden,
                    config.ple_embed_dim,
                )?,
                norm_key: reader.vector(&format!("{prefix}.norm_key.weight"), width)?,
                norm_query: reader.vector(&format!("{prefix}.norm_query.weight"), width)?,
                norm_conv: reader.vector(&format!("{prefix}.norm_conv.weight"), width)?,
                conv: reader.vector(
                    &format!("{prefix}.conv1d.weight"),
                    width * config.ple_conv_kernel_size,
                )?,
            })
        } else {
            None
        };
        layers.push(LayerWeights {
            branch,
            attn_hyper: hyper(&format!("{prefix}.attn_hyper_connection"), true)?,
            mlp_hyper: hyper(&format!("{prefix}.mlp_hyper_connection"), true)?,
            moe,
            ple,
        });
    }
    Ok(Qwen4ExpWeights {
        config: config.clone(),
        embed,
        lm_head,
        mixer,
        layers,
    })
}

/// The value a synthetic tensor entry carries.
///
/// Norms follow the released zero-centred convention (the reference multiplies by 1 + weight), so
/// zero is the identity and one is the Gated DeltaNet output-gate convention; everything else is a
/// small pseudo-random number. The point of the fixture is arithmetic coverage, not distribution
/// fidelity, so the only hard requirement is that no value is degenerate.
fn value_at(name: &str, rng: &mut TensorRng) -> f32 {
    if name.contains("ngram_embedding.weight_scale") {
        return SYNTHETIC_PLE_SCALE;
    }
    if name.ends_with("weight_scale_2") || name.ends_with("input_scale") {
        return SYNTHETIC_SCALE_2;
    }
    if name.ends_with("A_log") {
        // -log(A) = 1: a decay inside the range the delta rule expects.
        return -1.0;
    }
    if name.ends_with("dt_bias") {
        return 0.0;
    }
    if name.contains("norm") && name.ends_with(".weight") {
        // Gated DeltaNet's output-gate norm multiplies the weight; every other norm is zero-centred
        // and adds one. Both get a real coefficient rather than the identity: a fixture whose norms
        // are all zero cannot tell a loader that drops the weight from one that applies it, and the
        // first version of this file did exactly that.
        return if name.contains("linear_attn.norm.") {
            1.0 + rng.signed() * 0.1
        } else {
            rng.signed() * 0.1
        };
    }
    rng.signed() * SYNTHETIC_WEIGHT
}

/// The nearest E4M3 code to a value, chosen by exhaustive scan.
///
/// The scan is the honest way to write this: every code decodes through the same function the
/// loader uses, so the fixture cannot encode a value the decoders disagree about.
#[must_use]
pub fn encode_e4m3(value: f32) -> u8 {
    let mut best = 0u8;
    let mut best_error = f32::INFINITY;
    for byte in 0u8..=255 {
        let candidate = sparkflash_weights::quant::f8e4m3_to_f32(byte);
        if !candidate.is_finite() {
            continue;
        }
        let error = (candidate - value).abs();
        if error < best_error {
            best_error = error;
            best = byte;
        }
    }
    best
}

/// Encode one tensor of the plan as the bytes a safetensors header would describe.
///
/// # Errors
/// When the dtype is one this pack's plan never emits.
pub(crate) fn synthetic_bytes(
    spec: &sparkflash_weights::TensorSpec,
    seed: u64,
) -> Result<Vec<u8>, String> {
    let elements = usize::try_from(spec.elements())
        .map_err(|_| format!("{} is too large for this machine", spec.name))?;
    let mut rng = TensorRng::new(seed, &spec.name);
    let mut out = Vec::with_capacity(spec.dtype.storage_bytes(elements));
    match spec.dtype {
        DType::BF16 => {
            for _ in 0..elements {
                let value = value_at(&spec.name, &mut rng);
                out.extend_from_slice(&half::bf16::from_f32(value).to_le_bytes());
            }
        }
        DType::F32 => {
            for _ in 0..elements {
                out.extend_from_slice(&SYNTHETIC_SCALE_2.to_le_bytes());
            }
        }
        DType::F8E4M3 => {
            let table = spec.name.contains("ngram_embedding.shard_");
            for _ in 0..elements {
                let value = if table {
                    // A table row is a real embedding: keep it on the codebook so the FP8 scale,
                    // not the code, controls the magnitude.
                    let choices = [0.25f32, 0.5, 1.0, 1.5, 2.0];
                    choices[(rng.unit() * choices.len() as f32) as usize % choices.len()]
                } else {
                    // A block scale. The released expert scales span the codebook, including
                    // E4M3's exponent-15 codes (0x78..0x7E) - finite values up to 448, and the
                    // corner a decoder that mistakes them for infinity gets wrong. A fixture that
                    // encodes only 1.0 cannot see that, so the spread reaches it deliberately.
                    const SPREAD: [f32; 6] = [1.0, 0.5, 256.0, 448.0, 0.001_953_125, 12.0];
                    SPREAD[(rng.unit() * SPREAD.len() as f32) as usize % SPREAD.len()]
                };
                out.push(encode_e4m3(value));
            }
        }
        DType::F4E2M1 => {
            for _ in 0..elements.div_ceil(2) {
                out.push(rng.nibble());
            }
        }
        DType::I64 => {
            for _ in 0..elements {
                out.extend_from_slice(&0i64.to_le_bytes());
            }
        }
        DType::U8 => {
            for _ in 0..elements {
                out.push(rng.nibble());
            }
        }
        other => {
            return Err(format!(
                "the synthetic fixture cannot encode a {} tensor ({})",
                other.name(),
                spec.name
            ));
        }
    }
    Ok(out)
}

/// Write a complete, shard-shaped synthetic checkpoint for a configuration.
///
/// The directory receives the configuration, one safetensors shard and its index. Every served
/// tensor of the plan is present with the released dtype, including the 128 PLE shards, so a
/// loader that conformance-checks the checkpoint (the device arm does) accepts it unmodified.
///
/// # Errors
/// When the directory cannot be created, a file cannot be written, or a spec has a dtype the
/// fixture cannot encode.
pub fn write_synthetic_checkpoint(
    directory: &Path,
    config: &Qwen4ExpConfig,
    seed: u64,
) -> Result<(), String> {
    write_synthetic_checkpoint_sharded(directory, config, seed, 1)
}

/// Write the fixture split across this many safetensors files.
///
/// The released checkpoint is eleven shards and its index maps each tensor to one of them, so a
/// single-file fixture would leave the multi-shard path - the index's weight map, the per-shard
/// headers, a range read that spans files - untested. Tensors are dealt round-robin, which also
/// spreads the PLE table shards across files.
///
/// The writer materialises every tensor in memory, so it is a fixture writer, not a converter.
///
/// # Errors
/// As [write_synthetic_checkpoint], plus a zero shard count.
pub fn write_synthetic_checkpoint_sharded(
    directory: &Path,
    config: &Qwen4ExpConfig,
    seed: u64,
    shards: usize,
) -> Result<(), String> {
    let plan = Qwen4ExpTensorPlan::served_and_mtp(config);
    let tensors = plan
        .specs()
        .iter()
        .map(|spec| Ok((spec.clone(), synthetic_bytes(spec, seed)?)))
        .collect::<Result<Vec<_>, String>>()?;
    write_checkpoint(directory, &tensors, shards).map_err(|error| error.to_string())?;
    std::fs::write(directory.join("config.json"), config.to_json().into_bytes())
        .map_err(|error| format!("write the fixture config: {error}"))?;
    Ok(())
}

/// A synthetic checkpoint that was written to disk, with the configuration it describes.
#[derive(Debug, Clone)]
pub struct SyntheticCheckpoint {
    /// The checkpoint directory.
    pub directory: PathBuf,
    /// The configuration it was written for.
    pub config: Qwen4ExpConfig,
}

impl SyntheticCheckpoint {
    /// Write the tiny fixture into a caller-provided directory.
    ///
    /// # Errors
    /// As [write_synthetic_checkpoint].
    pub fn write(directory: impl Into<PathBuf>) -> Result<Self, String> {
        let directory = directory.into();
        let config = Qwen4ExpConfig::tiny();
        write_synthetic_checkpoint(&directory, &config, 7)?;
        Ok(Self { directory, config })
    }
}
