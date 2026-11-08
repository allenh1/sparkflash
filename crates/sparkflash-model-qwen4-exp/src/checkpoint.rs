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

impl CheckpointRows {
    /// One shard's ranges, read into `out` (their lengths summed) in request order.
    fn read_into(
        &self,
        shard: usize,
        ranges: &[(u64, usize)],
        out: &mut [u8],
    ) -> Result<(), String> {
        let Some(Some(resolved)) = self.shards.get(shard) else {
            let bytes = self
                .source
                .read_ranges_paged(&self.tensor_name(shard), ranges)
                .map_err(|error| error.to_string())?;
            if bytes.len() != out.len() {
                return Err(format!(
                    "PLE shard {shard} returned {} bytes for {} requested",
                    bytes.len(),
                    out.len()
                ));
            }
            out.copy_from_slice(&bytes);
            return Ok(());
        };
        check_ranges(shard, resolved, ranges)?;
        read_resolved(shard, resolved, ranges, out)
    }
}

/// Refuse a range past the end of its shard before any read is made.
fn check_ranges(
    shard: usize,
    resolved: &ResolvedShard,
    ranges: &[(u64, usize)],
) -> Result<(), String> {
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
    }
    Ok(())
}

/// One resolved shard's checked ranges, read into `out` (their lengths summed) in order.
fn read_resolved(
    shard: usize,
    resolved: &ResolvedShard,
    ranges: &[(u64, usize)],
    out: &mut [u8],
) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    let mut cursor = 0usize;
    for &(offset, len) in ranges {
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
    Ok(())
}

/// Ranges one pread worker takes at a time, and the workers the process keeps for them. A token's
/// sixteen rows are sixteen shards and so sixteen pieces, as before; a prefill chunk's 65,536
/// rows are about a thousand, and the deeper queue is what lets the NVMe answer them in parallel
/// rather than 512 serial reads per shard.
const PLE_READ_PIECE: usize = 64;
const PLE_READ_WORKERS: usize = 64;

/// A pread worker's stack: a worker only loops over `read_exact_at`.
const PLE_READ_STACK: usize = 256 << 10;

/// One piece of a [CheckpointRows::read_shards] call on a [ReadPool] worker: a resolved shard's
/// checked ranges, whose bytes land in `slot` of the call's [PieceBatch].
struct PieceRead {
    shard: usize,
    resolved: ResolvedShard,
    ranges: Vec<(u64, usize)>,
    slot: usize,
    batch: Arc<PieceBatch>,
}

/// A piece's bytes or error, once it is read.
type PieceResult = Option<Result<Vec<u8>, String>>;

/// The pieces of one call still out, and each one's bytes or error by slot.
struct PieceBatch {
    state: std::sync::Mutex<(usize, Vec<PieceResult>)>,
    done: std::sync::Condvar,
}

impl PieceRead {
    fn run(self) {
        let bytes = self.ranges.iter().map(|(_, len)| *len).sum();
        let mut out = vec![0u8; bytes];
        let result =
            read_resolved(self.shard, &self.resolved, &self.ranges, &mut out).map(|()| out);
        // A poisoned batch is a caller that already panicked: nothing waits for the bytes.
        if let Ok(mut state) = self.batch.state.lock() {
            state.1[self.slot] = Some(result);
            state.0 -= 1;
            if state.0 == 0 {
                self.batch.done.notify_all();
            }
        }
    }
}

/// The process's PLE pread workers, started on the first read of more than one piece and kept.
///
/// A decode step reads a few dozen rows, a piece each. Spawning a scoped thread a piece cost more
/// than the reads: in the serving process, under its memory pressure, `pthread_create` measured
/// 0.1 to 2.3 ms a thread, about 3 ms of host time a Qwen3.8 DFlash 2 step against 0.08 ms of
/// `pread`, and the device idled through all of it after the first layer. The workers instead
/// wait on one queue, and the caller takes pieces from it too while it waits.
struct ReadPool {
    queue: std::sync::Mutex<std::collections::VecDeque<PieceRead>>,
    ready: std::sync::Condvar,
}

impl ReadPool {
    fn shared() -> Result<&'static Self, String> {
        static POOL: std::sync::OnceLock<Result<ReadPool, String>> = std::sync::OnceLock::new();
        POOL.get_or_init(|| {
            for index in 0..PLE_READ_WORKERS {
                std::thread::Builder::new()
                    .name(format!("qwen4-ple-read-{index}"))
                    .stack_size(PLE_READ_STACK)
                    .spawn(|| {
                        let Ok(pool) = Self::shared() else { return };
                        while let Some(piece) = pool.wait() {
                            piece.run();
                        }
                    })
                    .map_err(|error| {
                        format!("the PLE read worker {index} did not start: {error}")
                    })?;
            }
            Ok(Self {
                queue: std::sync::Mutex::new(std::collections::VecDeque::new()),
                ready: std::sync::Condvar::new(),
            })
        })
        .as_ref()
        .map_err(Clone::clone)
    }

    /// The next piece, waiting for one; None once the queue is poisoned.
    fn wait(&self) -> Option<PieceRead> {
        let mut queue = self.queue.lock().ok()?;
        loop {
            if let Some(piece) = queue.pop_front() {
                return Some(piece);
            }
            queue = self.ready.wait(queue).ok()?;
        }
    }

    /// The next piece, if one is queued.
    fn take(&self) -> Option<PieceRead> {
        self.queue.lock().ok()?.pop_front()
    }

    fn submit(&self, pieces: Vec<PieceRead>) -> Result<(), String> {
        let count = pieces.len();
        self.queue
            .lock()
            .map_err(|_| "the PLE read queue is poisoned".to_owned())?
            .extend(pieces);
        if count >= PLE_READ_WORKERS {
            self.ready.notify_all();
        } else {
            for _ in 0..count {
                self.ready.notify_one();
            }
        }
        Ok(())
    }
}

impl RowReader for CheckpointRows {
    fn read_rows(&self, shard: usize, ranges: &[(u64, usize)]) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; ranges.iter().map(|(_, len)| *len).sum()];
        self.read_into(shard, ranges, &mut out)?;
        Ok(out)
    }

    fn read_shards(
        &self,
        requests: &[(usize, Vec<(u64, usize)>)],
    ) -> Result<Vec<(usize, Vec<u8>)>, String> {
        // Each request's ranges in pieces, in order: (request, ranges).
        let mut pieces: Vec<(usize, &[(u64, usize)])> = Vec::new();
        for (request, (_, ranges)) in requests.iter().enumerate() {
            pieces.extend(ranges.chunks(PLE_READ_PIECE).map(|group| (request, group)));
        }
        let mut outputs: Vec<Vec<u8>> = requests
            .iter()
            .map(|(_, ranges)| Vec::with_capacity(ranges.iter().map(|(_, len)| *len).sum()))
            .collect();
        let read_here = |request: usize, group: &[(u64, usize)]| {
            let mut out = vec![0u8; group.iter().map(|(_, len)| *len).sum()];
            self.read_into(requests[request].0, group, &mut out)
                .map(|()| out)
        };
        if pieces.len() <= 1 {
            for (request, group) in pieces {
                outputs[request].extend_from_slice(&read_here(request, group)?);
            }
        } else {
            // The resolved shards' pieces go to the workers; a shard the index did not resolve is
            // read here, through the source, which the workers do not share.
            let mut queued = Vec::with_capacity(pieces.len());
            let mut here = Vec::new();
            let batch = Arc::new(PieceBatch {
                state: std::sync::Mutex::new((0, vec![None; pieces.len()])),
                done: std::sync::Condvar::new(),
            });
            for (slot, &(request, group)) in pieces.iter().enumerate() {
                let shard = requests[request].0;
                let Some(Some(resolved)) = self.shards.get(shard) else {
                    here.push(slot);
                    continue;
                };
                check_ranges(shard, resolved, group)?;
                queued.push(PieceRead {
                    shard,
                    resolved: resolved.clone(),
                    ranges: group.to_vec(),
                    slot,
                    batch: Arc::clone(&batch),
                });
            }
            let poisoned = || "the PLE read batch is poisoned".to_owned();
            batch.state.lock().map_err(|_| poisoned())?.0 = queued.len();
            let pool = ReadPool::shared()?;
            pool.submit(queued)?;
            let mut read = Vec::with_capacity(here.len());
            for &slot in &here {
                let (request, group) = pieces[slot];
                read.push((slot, read_here(request, group)));
            }
            while let Some(piece) = pool.take() {
                piece.run();
            }
            let mut state = batch.state.lock().map_err(|_| poisoned())?;
            while state.0 > 0 {
                state = batch.done.wait(state).map_err(|_| poisoned())?;
            }
            let mut results = std::mem::take(&mut state.1);
            drop(state);
            for (slot, result) in read {
                results[slot] = Some(result);
            }
            for (&(request, _), result) in pieces.iter().zip(results) {
                outputs[request].extend_from_slice(&result.ok_or_else(poisoned)??);
            }
        }
        Ok(requests
            .iter()
            .map(|(shard, _)| *shard)
            .zip(outputs)
            .collect())
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

    /// A dense projection in the configuration's dense format: BF16 `{prefix}.weight`, or the
    /// Qwen3.5/3.6 FP8 release's 128x128-tile E4M3 codes and BF16 inverse scales.
    fn linear(
        &self,
        prefix: &str,
        rows: usize,
        columns: usize,
        quant: crate::config::DenseQuant,
    ) -> Result<Mat, String> {
        match quant {
            crate::config::DenseQuant::Bf16 => self.mat(&format!("{prefix}.weight"), rows, columns),
            crate::config::DenseQuant::Fp8Block => self.fp8(prefix, rows, columns),
        }
    }

    /// One block-scaled FP8 matrix: E4M3 codes and the weight_scale_inv grid, one scale a
    /// 128x128 tile. The official FP8 checkpoints store their experts (and, for Qwen3.5/3.6, the
    /// dense projections) this way.
    fn fp8(&self, prefix: &str, rows: usize, columns: usize) -> Result<Mat, String> {
        let mut weight = sparkflash_weights::linalg::decode_by_name(
            self.source,
            &format!("{prefix}.weight"),
            usize::MAX,
        )
        .map_err(|error| error.to_string())?;
        let scales = sparkflash_weights::linalg::decode_by_name(
            self.source,
            &format!("{prefix}.weight_scale_inv"),
            usize::MAX,
        )
        .map_err(|error| error.to_string())?;
        sparkflash_weights::linalg::apply_scales(
            &mut weight,
            rows,
            columns,
            &scales,
            sparkflash_weights::linalg::ScaleGrid::Tile(128),
        )
        .map_err(|error| error.to_string())?;
        Ok(Mat::new(rows, columns, weight))
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
    let dense = config.dense_quant;
    // A plain residual's "hyper-connection" is the branch's pre-norm alone: no mixer, and an
    // injection of one (see forward::hyper_mix).
    let plain = |name: &str| -> Result<HyperBlock, String> {
        Ok(HyperBlock {
            norm: reader.vector(name, hidden)?,
            down: Mat::zeros(0, hidden),
            up: Mat::zeros(hidden, 0),
            inject: None,
        })
    };
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
    let mixer = if config.plain_residual() {
        plain("model.language_model.norm.weight")?
    } else {
        hyper("model.language_model.hyper_connection_mixer", false)?
    };
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for (index, kind) in config.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        let branch = match kind {
            crate::config::LayerKind::Linear => {
                let prefix = format!("{prefix}.linear_attn");
                BranchWeights::Linear {
                    in_qkv: reader.linear(
                        &format!("{prefix}.in_proj_qkv"),
                        config.linear_conv_width(),
                        hidden,
                        dense,
                    )?,
                    in_z: reader.linear(
                        &format!("{prefix}.in_proj_z"),
                        config.linear_value_width(),
                        hidden,
                        dense,
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
                    out_proj: reader.linear(
                        &format!("{prefix}.out_proj"),
                        hidden,
                        config.linear_value_width(),
                        dense,
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
                    q_proj: reader.linear(
                        &format!("{prefix}.q_proj"),
                        config.q_proj_width(),
                        hidden,
                        dense,
                    )?,
                    k_proj: reader.linear(
                        &format!("{prefix}.k_proj"),
                        config.kv_proj_width(),
                        hidden,
                        dense,
                    )?,
                    v_proj: reader.linear(
                        &format!("{prefix}.v_proj"),
                        config.kv_proj_width(),
                        hidden,
                        dense,
                    )?,
                    o_proj: reader.linear(
                        &format!("{prefix}.o_proj"),
                        hidden,
                        config.attn_out_width(),
                        dense,
                    )?,
                    q_norm: reader.vector(&format!("{prefix}.q_norm.weight"), config.head_dim)?,
                    k_norm: reader.vector(&format!("{prefix}.k_norm.weight"), config.head_dim)?,
                    // Full attention carries no indexer; its three fields stay empty.
                    index_qk: if config.full_attention() {
                        Mat::zeros(0, hidden)
                    } else {
                        reader.mat(
                            &format!("{prefix}.indexer.index_qk_proj.weight"),
                            (config.indexer_n_heads + config.indexer_kv_heads)
                                * config.indexer_head_dim,
                            hidden,
                        )?
                    },
                    index_q_norm: if config.full_attention() {
                        Vec::new()
                    } else {
                        reader.vector(
                            &format!("{prefix}.indexer.q_layernorm.weight"),
                            config.indexer_head_dim,
                        )?
                    },
                    index_k_norm: if config.full_attention() {
                        Vec::new()
                    } else {
                        reader.vector(
                            &format!("{prefix}.indexer.k_layernorm.weight"),
                            config.indexer_head_dim,
                        )?
                    },
                }
            }
        };
        let mlp = format!("{prefix}.mlp");
        let shared = |name: &str, rows: usize, columns: usize| -> Result<Mat, String> {
            reader.linear(&format!("{mlp}.shared_expert.{name}"), rows, columns, dense)
        };
        let moe = MoeWeights {
            router: reader.mat(&format!("{mlp}.gate.weight"), config.num_experts, hidden)?,
            shared: Expert {
                gate: shared("gate_proj", config.shared_expert_intermediate_size, hidden)?,
                up: shared("up_proj", config.shared_expert_intermediate_size, hidden)?,
                down: shared("down_proj", hidden, config.shared_expert_intermediate_size)?,
            },
            shared_gate: reader.mat(&format!("{mlp}.shared_expert_gate.weight"), 1, hidden)?,
            experts: if config.expert_quant == crate::config::ExpertQuant::Bf16 {
                fused_experts(&reader, config, &mlp)?
            } else {
                (0..config.num_experts)
                    .map(|expert| {
                        let prefix = format!("{mlp}.experts.{expert}");
                        let decode =
                            |name: &str, rows: usize, columns: usize| match config.expert_quant {
                                crate::config::ExpertQuant::Nvfp4 => {
                                    reader.nvfp4(&format!("{prefix}.{name}"), rows, columns)
                                }
                                crate::config::ExpertQuant::Fp8Block => {
                                    reader.fp8(&format!("{prefix}.{name}"), rows, columns)
                                }
                                crate::config::ExpertQuant::Bf16 => {
                                    unreachable!("BF16 experts are read fused above")
                                }
                            };
                        Ok(Expert {
                            gate: decode("gate_proj", config.moe_intermediate_size, hidden)?,
                            up: decode("up_proj", config.moe_intermediate_size, hidden)?,
                            down: decode("down_proj", hidden, config.moe_intermediate_size)?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?
            },
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
            attn_hyper: if config.plain_residual() {
                plain(&format!("{prefix}.input_layernorm.weight"))?
            } else {
                hyper(&format!("{prefix}.attn_hyper_connection"), true)?
            },
            mlp_hyper: if config.plain_residual() {
                plain(&format!("{prefix}.post_attention_layernorm.weight"))?
            } else {
                hyper(&format!("{prefix}.mlp_hyper_connection"), true)?
            },
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

/// Qwen3.5/3.6's BF16 experts, split out of the fused `[experts, 2 * intermediate, hidden]`
/// gate/up tensor (gate rows first) and the `[experts, hidden, intermediate]` down tensor.
fn fused_experts(
    reader: &Reader<'_>,
    config: &Qwen4ExpConfig,
    mlp: &str,
) -> Result<Vec<Expert>, String> {
    let hidden = config.hidden_size;
    let intermediate = config.moe_intermediate_size;
    let experts = config.num_experts;
    let gate_up = reader.vector(
        &format!("{mlp}.experts.gate_up_proj"),
        experts * 2 * intermediate * hidden,
    )?;
    let down = reader.vector(
        &format!("{mlp}.experts.down_proj"),
        experts * hidden * intermediate,
    )?;
    let block = intermediate * hidden;
    Ok((0..experts)
        .map(|expert| {
            let gate_up = &gate_up[expert * 2 * block..(expert + 1) * 2 * block];
            Expert {
                gate: Mat::new(intermediate, hidden, gate_up[..block].to_vec()),
                up: Mat::new(intermediate, hidden, gate_up[block..].to_vec()),
                down: Mat::new(
                    hidden,
                    intermediate,
                    down[expert * block..(expert + 1) * block].to_vec(),
                ),
            }
        })
        .collect())
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
    let plan = Qwen4ExpTensorPlan::served_and_mtp(config).map_err(|error| error.to_string())?;
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
