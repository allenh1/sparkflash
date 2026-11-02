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

//! The Qwen3.8-Flash-Next (Qwen4-Exp) pack, for the NVFP4 checkpoint at TP=1 on the device.
//!
//! nvidia/Qwen3.8-Flash-Next-NVFP4 is a 125B-parameter hybrid-attention MoE with a 51B n-gram
//! embedding, post-trained from Qwen/Qwen3.8-Flash-Next and quantised with NVIDIA ModelOpt 0.46.
//! This crate owns the model definition:
//!
//! * [config] parses the released configuration and enforces the architecture invariants the
//!   reference implementation checks (QSA, PLE, hyper-connections, Gated DeltaNet).
//! * [tensor] plans all 299,545 tensor names of the checkpoint and pins the byte identity of the
//!   complete plan against the released index total, so the plan cannot silently drift.
//! * [ngram] is the PLE hash: the Engram splitmix64 multiplier draw, prime bucket layout and
//!   rolling multiplicative-XOR, with Qwen's EOS segment rule.
//! * [ple] pages the 51.2 GiB n-gram table on the host with a bounded row cache and a prefetch
//!   entry point, the same serving shape sparkflash-engram uses for V4.1.
//!
//! The NVFP4 experts and the FP8 PLE shards decode through `sparkflash_weights::nvfp4`, which
//! every ModelOpt checkpoint in the tree shares.
//! * [forward] is the host reference forward at any geometry, including the tiny fixtures.
//! * [checkpoint] is the shared checkpoint plumbing: the host loader the device arm is compared
//!   against, the paged PLE reader both sides use, and a synthetic-checkpoint writer so the whole
//!   device loop can run on a tiny fixture instead of the 132.6 GB release.
//! * [convert] writes and reads the converted TP=1 rank file, the exact bytes the weight walk
//!   uploads, so a boot is one direct-I/O read instead of 77.8 GiB of per-tensor uploads.
//! * [device] is the TP=1 device arm: one rank, one cubin, no host fallback.
//! * [dflash] is the DFlash 2 block drafter's configuration, which the device arm serves as its
//!   speculative drafter when a drafter checkpoint is configured.
//!
//! The MTP stage and the vision tower are planned but not served; the released checkpoint carries
//! them and the plan's byte total includes them so the identity is exact.
//!
//! # Attribution
//!
//! * **Model architecture & reference:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
//!   (Alibaba Cloud Qwen team) and the Qwen4ExpForConditionalGeneration reference implementation
//!   in [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).
//! * **Quantization format:** [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
//!   (NVIDIA TensorRT ModelOpt 0.46).
//! * **SM120/SM121 kernel & PLE paging prior art:** [b12x](https://github.com/local-inference-lab/b12x)
//!   (flashinfer/experimental/b12x, Apache-2.0; Luke Alonso @lukealonso, @original-el8, and
//!   contributors) and [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) (Apache-2.0).
//!   See the root NOTICE file for full provenance details.
//! * **DFlash 2 speculative drafting:** the DFlash method of Chen, Liang and Liu
//!   ([z-lab/dflash](https://github.com/z-lab/dflash), MIT), Inco AI's
//!   [Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
//!   drafter (Apache-2.0), and [SGLang](https://github.com/sgl-project/sglang)'s DFlash support
//!   (Apache-2.0), read for the arithmetic and the verify protocol; no source is copied.

pub mod cache;
pub mod checkpoint;
pub mod codec;
pub mod config;
pub mod convert;
pub mod device;
pub mod dflash;
pub mod forward;
pub mod ngram;
pub mod ple;
pub mod sampling;
pub mod tensor;
pub mod vision;

pub use cache::{cache_dir, cached_model_dir, cached_model_in};
pub use checkpoint::{
    CheckpointRows, SyntheticCheckpoint, load_weights, open_ple_table, write_synthetic_checkpoint,
    write_synthetic_checkpoint_sharded,
};
pub use codec::{ByteCodec, QwenCodec};
pub use config::{ConfigError, LayerKind, Qwen4ExpConfig};
pub use forward::{ForwardError, Qwen4ExpWeights, ReferenceState, ReferenceWeights};
pub use ngram::{NgramHead, PleHasher, PleLayout, layer_multipliers};
pub use ple::{PleCacheCounters, PleTable, RowReader, shard_tensor_name};
pub use tensor::{Qwen4ExpTensorPlan, Role};
pub use vision::{Qwen4Vision, QwenVisionCodec, VisionTowerConfig};

/// The served model id.
pub const MODEL_ID: &str = "nvidia/Qwen3.8-Flash-Next-NVFP4";

/// The control-plane weight selector.
pub const WEIGHTS_ID: &str = "qwen3.8-flash-next-nvfp4";

/// The model family, for artifact compatibility checks.
pub const FAMILY: &str = "qwen4-exp";

/// The largest context the released checkpoint serves.
pub const CONTEXT_TOKENS: usize = 262_144;

/// The protocol upper bound on one request's generated tokens for this build.
pub const MAX_OUTPUT_TOKENS: usize = 32_768;

/// The context window the TP=1 device arm allocates by default.
///
/// This is the released checkpoint's full 262,144-position window. The QSA indexer selects
/// 2,048 tokens per query out of complete 4-token blocks, so the KV cache grows with the
/// sequence rather than staying in a sliding window. The QSA K/V cache at 12 sparse layers is
/// 12 x 262,144 x 2 x 256 x 4 bytes = 6.4 GiB at f32, plus 1.6 GiB of indexer keys and 113 MiB
/// of Gated DeltaNet state, beside 77.8 GiB of device weights. That fits a 128 GiB node; a
/// caller that wants a smaller window can still pass `--context`.
pub const DEFAULT_DEVICE_CONTEXT: usize = CONTEXT_TOKENS;

/// QSA K/V bytes one context token adds at the 12 sparse layers.
pub const QSA_KV_BYTES_PER_TOKEN: u64 = 12 * 2 * 256 * 4;
/// Indexer-key bytes one context token adds at the 12 sparse layers.
pub const INDEXER_KV_BYTES_PER_TOKEN: u64 = 12 * 128 * 4;
/// The Gated DeltaNet recurrent state, which is independent of context length.
pub const GDN_RECURRENT_BYTES: u64 = 36 * 48 * 128 * 128 * 4;
/// The KV memory a packed serve may spend before it must refuse more workers.
///
/// This is not a node limit: the 77.8 GiB of weights and the host PLE table sit beside it, and the
/// table keeps paging in as n-grams are touched. It is the slice the session caches are allowed to
/// claim, and it is sized for the machine this arm is developed and served on - a 116 GiB unified
/// memory node, where the weights leave about 35 GiB and the table grows into most of it. One
/// released 262,144-token window is 8.17 GiB, so this admits exactly one full window, and a second
/// session has to come from a shorter `--context`. A node with more room raises it with
/// `--kv-budget-gib`; the earlier 20 GiB slice admitted two full windows and is what this laptop
/// could not hold.
pub const CONCURRENCY_KV_BUDGET_BYTES: u64 = 10 << 30;

/// The KV bytes a budget of `gib` gibibytes allows.
#[must_use]
pub const fn kv_budget_bytes(gib: usize) -> u64 {
    (gib as u64) << 30
}

/// KV bytes one session allocates at `context` tokens.
#[must_use]
pub const fn sequence_kv_bytes(context: usize) -> u64 {
    (context as u64) * (QSA_KV_BYTES_PER_TOKEN + INDEXER_KV_BYTES_PER_TOKEN) + GDN_RECURRENT_BYTES
}

/// KV bytes `concurrency` sessions allocate at `context` tokens.
#[must_use]
pub const fn concurrency_kv_bytes(context: usize, concurrency: usize) -> u64 {
    sequence_kv_bytes(context) * concurrency as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_window_is_the_released_context() {
        assert_eq!(CONTEXT_TOKENS, 262_144);
        assert_eq!(DEFAULT_DEVICE_CONTEXT, CONTEXT_TOKENS);
        assert_eq!(MAX_OUTPUT_TOKENS, 32_768);
    }

    #[test]
    fn the_kv_budget_admits_one_default_session_on_this_laptop_and_refuses_two() {
        let one = sequence_kv_bytes(DEFAULT_DEVICE_CONTEXT);
        assert_eq!(one, 8_166_309_888);
        assert!(one <= CONCURRENCY_KV_BUDGET_BYTES);
        assert!(concurrency_kv_bytes(DEFAULT_DEVICE_CONTEXT, 2) > CONCURRENCY_KV_BUDGET_BYTES);
        // Two sessions fit the same slice at half the window, which is the split a laptop serves.
        assert!(concurrency_kv_bytes(DEFAULT_DEVICE_CONTEXT / 2, 2) <= CONCURRENCY_KV_BUDGET_BYTES);
    }
}
