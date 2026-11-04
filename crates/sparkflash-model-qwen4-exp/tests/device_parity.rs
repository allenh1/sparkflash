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

//! Device parity: the twenty-kernel TP=1 loop against the host reference, on a real checkpoint.
//!
//! The fixture is the tiny configuration written as a complete safetensors checkpoint, so this
//! runs the same code paths the released 132.6 GB checkpoint would: the conformance check, the
//! weight upload (BF16, FP8 and NVFP4 experts with both per-tensor scales), host-paged PLE rows,
//! prefill, decode, and the final logits read-back.
//!
//! The test skips - loudly - on a build without a cubin or a machine without a CUDA driver. It
//! does not downgrade a wrong answer to a skip: once the device opens, a mismatch fails the test.

use sparkflash_model::ModelBackend;
use sparkflash_model_qwen4_exp::device::{
    DENSE_ROWS, PrefillPlan, Qwen4ExpDeviceBackend, cubin_available, embedded_prefill_cubin,
};
use sparkflash_model_qwen4_exp::{
    LayerKind, PleHasher, Qwen4ExpConfig, load_weights, open_ple_table, write_synthetic_checkpoint,
};
use sparkflash_weights::safetensors::SafeTensorsSource;

const INDEX: &str = "model.safetensors.index.json";

/// The chunk these tests prefill in: the width every wider chunk computes its rows as.
const CHUNK: usize = DENSE_ROWS;

/// The relative tolerance one logit may differ by.
///
/// The two sides do not run identical arithmetic: the device accumulates each GEMV in a block
/// reduction and its PLE table is read in shard batches, while the host sums one term at a time in
/// f32 order. Every weight is the same bytes, so the difference is floating-point association, not
/// a different model - which is why the bound is tight relative to the row's own magnitude.
const RELATIVE_TOLERANCE: f32 = 1e-5;

/// The tensor-core prefill carries each projection's and expert's f32 input as a BF16 head and
/// remainder (16 significant bits, where the token kernels keep 24), so its bound is that split's
/// rounding carried through the stack: about 1e-5 measured, bounded here at ten times that. A
/// single-BF16 operand lands near 5e-3 and a wrong index or a dropped row O(1), both far past it.
const TENSOR_CORE_TOLERANCE: f32 = 1e-4;

fn compare(stage: &str, host: &[f32], device: &[f32], tolerance: f32) {
    assert_eq!(
        host.len(),
        device.len(),
        "{stage}: {} host logits vs {} device logits",
        host.len(),
        device.len()
    );
    assert!(
        device.iter().all(|value| value.is_finite()),
        "{stage}: the device produced a non-finite logit"
    );
    let mut largest = 0.0f32;
    let mut where_largest = 0usize;
    for (index, (host, device)) in host.iter().zip(device).enumerate() {
        let difference = (host - device).abs();
        if difference > largest {
            largest = difference;
            where_largest = index;
        }
    }
    let scale = host
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let relative = largest / scale;
    eprintln!(
        "{stage}: max |host - device| = {largest:.6} at {where_largest} (row scale {scale:.4}), relative {relative:.3e}"
    );
    assert!(
        relative < tolerance,
        "{stage}: relative difference {relative:.3e} exceeds {tolerance:.3e}; host {host:?} device {device:?}"
    );
    let host_argmax = host
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    let device_argmax = device
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index);
    assert_eq!(host_argmax, device_argmax, "{stage}: the argmax moved");
}

/// Run the whole TP=1 loop on one configuration and compare it with the host oracle.
///
/// Returns false when this build or machine cannot run the device, so the caller can skip.
fn parity(config: Qwen4ExpConfig, label: &str) -> bool {
    parity_with(config, label, 11)
}

/// The same loop over a chosen prompt length, so a test can reach a prefill chunk past the GEMV's
/// eight-token register tile.
///
/// Every chunk runs on the token kernels, which agree with the oracle to f32 association; the
/// tensor-core image has its own test below.
fn parity_with(config: Qwen4ExpConfig, label: &str, prompt_len: usize) -> bool {
    parity_on(
        config,
        label,
        prompt_len,
        PrefillPlan::new(CHUNK, false),
        RELATIVE_TOLERANCE,
        false,
    )
}

/// The loop under a chosen prefill plan, with the bound the prefill's logits must meet. The
/// decode steps after it always run on the token kernels, so they also check the caches and
/// recurrent state the prefill left behind.
fn parity_on(
    mut config: Qwen4ExpConfig,
    label: &str,
    prompt_len: usize,
    plan: PrefillPlan,
    tolerance: f32,
    converted: bool,
) -> bool {
    if !cubin_available() {
        eprintln!(
            "skipping {label}: this build has no Qwen4-Exp cubin (rebuild with nvcc, or set \
             SPARKFLASH_CUDA_ARCH to this GPU's architecture)"
        );
        return false;
    }
    config.num_hidden_layers = config.layer_types.len();
    let temp = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(temp.path(), &config, 7).expect("the fixture writes");
    // The context has to hold the prompt plus the decode steps, and a test that prefills more
    // than one chunk needs a window past CHUNK.
    let session_limit = 32.max(prompt_len + 8).min(config.max_position_embeddings);
    let converted_dir = tempfile::tempdir().expect("a converted directory");
    let converted_path = if converted {
        let source = SafeTensorsSource::open(temp.path(), &temp.path().join(INDEX))
            .expect("the fixture index opens");
        let file = sparkflash_model_qwen4_exp::convert::convert_rank(
            &source,
            &config,
            sparkflash_weights::tp::TpLayout {
                rank: 0,
                world_size: 1,
            },
            temp.path(),
            converted_dir.path(),
            "fixture",
        )
        .expect("the rank converts");
        Some(file.path)
    } else {
        None
    };
    let opened = match &converted_path {
        Some(path) => Qwen4ExpDeviceBackend::open_with_prefill_converted(
            temp.path(),
            session_limit,
            plan,
            path,
        ),
        None => Qwen4ExpDeviceBackend::open_with_prefill(temp.path(), session_limit, plan),
    };
    let backend = match opened {
        Ok(backend) => backend,
        Err(error)
            if error.code() == "qwen4_device"
                && (error.message().contains("CUDA:")
                    || error.message().contains("cuModuleLoadData failed: 209")) =>
        {
            eprintln!("skipping {label}: {error}");
            return false;
        }
        Err(error) => panic!("{label}: the device arm refused the fixture checkpoint: {error}"),
    };
    let source = SafeTensorsSource::open(temp.path(), &temp.path().join(INDEX))
        .expect("the fixture index opens");
    let host = load_weights(&config, &source).expect("the host loader decodes the checkpoint");
    let layer = config.ple_layer_indices()[0];
    let mut table = open_ple_table(temp.path(), &config, layer).expect("the PLE table opens");
    let mut hasher = PleHasher::new(&config, 0).expect("the hasher builds");
    let mut state = host.new_state();
    let mut session = backend.open_session().expect("a device session opens");

    // A prompt long enough that the indexer selects among several blocks, then decode steps.
    let prompt: Vec<u32> = (0..prompt_len)
        .map(|index| 3 + (index % 9) as u32)
        .collect();
    let rows = hasher.push(&prompt, None).expect("the hash rows");
    let embeddings: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| table.embedding(row))
        .collect::<Result<_, String>>()
        .expect("the table serves every row");
    let host_logits = host
        .forward(&mut state, &prompt, None, Some(&embeddings))
        .expect("the host prefill runs");
    let prefill = session.prefill(&prompt).expect("the device prefill runs");
    compare(
        &format!("{label} prefill"),
        host_logits.last().expect("one row per token"),
        &prefill.logits,
        tolerance,
    );

    for token in [25u32, 27, 29] {
        let rows = hasher.push(&[token], None).expect("the hash rows");
        let embedding = table.embedding(&rows[0]).expect("the table serves the row");
        let host_logits = host
            .forward(&mut state, &[token], None, Some(&[embedding]))
            .expect("the host decode runs");
        let step = session.decode(token).expect("the device decode runs");
        compare(
            &format!("{label} decode {token}"),
            host_logits.last().expect("one row per token"),
            &step.logits,
            tolerance,
        );
    }

    // A prompt with an end-of-generation id in the middle: Qwen's PLE segment rule restarts the
    // n-gram window there, so the two hashers have to agree on which tokens contribute and which
    // are replaced by EOS. The logits only match if the rows they read match.
    let mut hasher = PleHasher::new(&config, 0).expect("the hasher builds");
    let mut state = host.new_state();
    let mut session = backend
        .open_session()
        .expect("a second device session opens");
    let eos = config.eos_token_ids[0];
    let segmented = [3u32, 5, 7, eos, 9, 11, 13];
    let rows = hasher.push(&segmented, None).expect("the hash rows");
    let embeddings: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| table.embedding(row))
        .collect::<Result<_, String>>()
        .expect("the table serves every row");
    let host_logits = host
        .forward(&mut state, &segmented, None, Some(&embeddings))
        .expect("the host segmented prefill runs");
    let prefill = session
        .prefill(&segmented)
        .expect("the device segmented prefill runs");
    compare(
        &format!("{label} segmented prefill"),
        host_logits.last().expect("one row per token"),
        &prefill.logits,
        tolerance,
    );
    true
}

#[test]
fn the_tp1_device_loop_matches_the_host_oracle() {
    let _ = parity(Qwen4ExpConfig::tiny(), "tiny");
}

/// The converted rank file is the same bits: the device loop over a file the converter wrote must
/// produce the checkpoint path's logits, including the NVFP4 scales and the rebuilt expert tables.
#[test]
fn the_converted_rank_file_matches_the_checkpoint_path() {
    let _ = parity_on(
        Qwen4ExpConfig::tiny(),
        "tiny converted",
        11,
        PrefillPlan::new(CHUNK, false),
        RELATIVE_TOLERANCE,
        true,
    );
}

/// A second, deliberately asymmetric geometry.
///
/// The tiny fixture is powers of two everywhere: two residual streams, four value heads, an
/// eight-wide head, a two-token indexer block. A kernel that assumed a power of two, or that only
/// worked when a width divided the 256-thread block, would pass on it and fail on the release. This
/// configuration is odd where it can be (three streams, a five-row bottleneck, six query heads,
/// three-way top-k, an n-gram window of two) while staying inside the architecture's validation.
#[test]
fn an_asymmetric_geometry_matches_the_host_oracle() {
    let mut config = Qwen4ExpConfig::tiny();
    config.hidden_size = 48;
    config.hc_count = 3;
    config.hc_lowrank = 5;
    config.num_attention_heads = 6;
    config.num_key_value_heads = 2;
    config.head_dim = 16;
    config.vocab_size = 96;
    config.mrope_section = [2, 1, 1];
    config.layer_types = vec![LayerKind::Linear, LayerKind::Linear, LayerKind::Sparse];
    config.linear_conv_kernel_dim = 3;
    config.linear_key_head_dim = 8;
    config.linear_value_head_dim = 8;
    config.linear_num_key_heads = 2;
    config.linear_num_value_heads = 6;
    config.moe_intermediate_size = 12;
    config.shared_expert_intermediate_size = 12;
    config.num_experts = 6;
    config.num_experts_per_tok = 3;
    config.ple_layer_ids = vec![2];
    config.ple_embed_dim = 12;
    config.ple_conv_kernel_size = 3;
    config.ngram_size = 2;
    config.heads_per_ngram = 3;
    config.ngram_vocab_size_base = 500;
    config.make_ngram_vocab_size_divisible_by = 8;
    config.split_ngram_parts = 4;
    config.indexer_n_heads = 3;
    config.indexer_kv_heads = 1;
    config.indexer_head_dim = 8;
    config.indexer_budget = 4;
    config.indexer_compress_ratio = 2;
    config.eos_token_ids = vec![5];
    config.max_position_embeddings = 64;
    let _ = parity(config, "asymmetric");
}
/// A third geometry: the indexer's rotary width is *narrower* than its head.
///
/// Both configurations above rope the whole index head, so the kernel's `half = rotary / 2` branch
/// and the host's `rotary.min(index_dim)` never disagree. The released model ropes 64 of a 128-wide
/// index head, which is exactly the narrow case, so it needs a fixture that exercises it. The block
/// width is four here rather than two, which changes how many tokens one selected block covers.
#[test]
fn a_narrow_indexer_rotary_matches_the_host_oracle() {
    let mut config = Qwen4ExpConfig::tiny();
    config.hidden_size = 40;
    config.hc_count = 2;
    config.hc_lowrank = 3;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.head_dim = 16;
    config.partial_rotary_factor = 0.25;
    config.mrope_section = [1, 1, 0];
    config.vocab_size = 48;
    config.layer_types = vec![LayerKind::Linear, LayerKind::Linear, LayerKind::Sparse];
    config.linear_conv_kernel_dim = 2;
    config.linear_key_head_dim = 4;
    config.linear_value_head_dim = 4;
    config.linear_num_key_heads = 2;
    config.linear_num_value_heads = 4;
    config.moe_intermediate_size = 10;
    config.shared_expert_intermediate_size = 10;
    config.num_experts = 5;
    config.num_experts_per_tok = 2;
    config.ple_layer_ids = vec![2];
    config.ple_embed_dim = 12;
    config.ple_conv_kernel_size = 2;
    config.ngram_size = 3;
    config.heads_per_ngram = 2;
    config.ngram_vocab_size_base = 300;
    config.make_ngram_vocab_size_divisible_by = 8;
    config.split_ngram_parts = 8;
    config.indexer_n_heads = 2;
    config.indexer_kv_heads = 1;
    config.indexer_head_dim = 8;
    config.indexer_budget = 8;
    config.indexer_compress_ratio = 4;
    config.eos_token_ids = vec![4];
    config.max_position_embeddings = 64;
    config.num_hidden_layers = config.layer_types.len();
    let _ = parity(config, "narrow rotary");
}

/// A chunk larger than the GEMV's eight-token register tile.
///
/// Every fixture above prefills fewer tokens than `GEMV_MAX_BATCH`, so the shared-memory GEMM,
/// the grouped hyper-connection and shared-expert passes and the routed-expert L2 sort with more
/// slots in flight than a block has threads were all unexercised. The released model prefils in
/// chunks of 128, which reaches every one of them.
#[test]
fn a_chunk_past_the_gemv_register_tile_matches_the_host_oracle() {
    let mut config = Qwen4ExpConfig::tiny();
    config.hidden_size = 64;
    config.hc_count = 4;
    config.hc_lowrank = 16;
    config.moe_intermediate_size = 32;
    config.shared_expert_intermediate_size = 32;
    config.num_experts = 12;
    config.num_experts_per_tok = 9;
    let _ = parity_with(config, "wide chunk", 29);
}

/// A prompt long enough for more than one prefill chunk.
///
/// A prompt 22 tokens past CHUNK runs a full chunk and a partial one, which is the
/// only way to exercise the second chunk's absolute positions: the indexer's block count, the KV
/// append offset and the per-token selection rows all move with the chunk boundary. The full chunk
/// is also past ATTENTION_CHUNK, so the sparse block's sub-chunk offsets run too.
#[test]
fn a_prompt_spanning_two_chunks_matches_the_host_oracle() {
    let mut config = Qwen4ExpConfig::tiny();
    config.hidden_size = 64;
    config.hc_count = 4;
    config.hc_lowrank = 16;
    config.moe_intermediate_size = 32;
    config.shared_expert_intermediate_size = 32;
    config.num_experts = 12;
    config.num_experts_per_tok = 9;
    config.max_position_embeddings = 1024;
    let _ = parity_with(config, "two chunks", CHUNK + 22);
}

/// The tensor-core prefill image against the oracle, on a geometry every one of its paths takes:
/// 128-wide GDN heads with three value heads a key head (the release's grouping), a hidden width
/// the routed-expert MMA tiles, 32-row multiples for the dense projections, and a prompt of one
/// full chunk and a 100-row one, whose last MMA tile is partial. The decode steps after it run on
/// the token kernels over the caches and recurrent state the chunks left.
#[test]
fn the_tensor_core_prefill_matches_the_host_oracle() {
    let mut config = Qwen4ExpConfig::tiny();
    config.hidden_size = 128;
    config.hc_count = 4;
    config.hc_lowrank = 32;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.linear_num_key_heads = 1;
    config.linear_num_value_heads = 3;
    config.moe_intermediate_size = 64;
    config.shared_expert_intermediate_size = 64;
    config.num_experts = 16;
    config.num_experts_per_tok = 4;
    config.max_position_embeddings = 1024;
    tensor_core_parity(config, "tensor-core prefill", CHUNK + 100);
}

/// The same at the release's proportions: several GDN key heads (each with its own slice of the
/// conv rows and the recurrent state), ten routed experts a token over many experts, and a prompt
/// of a single partial chunk.
#[test]
fn the_tensor_core_prefill_matches_the_host_oracle_at_release_proportions() {
    let mut config = Qwen4ExpConfig::tiny();
    config.hidden_size = 256;
    config.hc_count = 4;
    config.hc_lowrank = 32;
    config.layer_types = vec![LayerKind::Linear, LayerKind::Linear, LayerKind::Sparse];
    config.ple_layer_ids = vec![2];
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.linear_num_key_heads = 4;
    config.linear_num_value_heads = 12;
    config.moe_intermediate_size = 128;
    config.shared_expert_intermediate_size = 128;
    config.num_experts = 64;
    config.num_experts_per_tok = 10;
    config.max_position_embeddings = 1024;
    tensor_core_parity(config.clone(), "release proportions 73", 73);
    tensor_core_parity(config, "release proportions 612", CHUNK + 100);
}

fn tensor_core_parity(config: Qwen4ExpConfig, label: &str, prompt_len: usize) {
    if embedded_prefill_cubin().is_empty() {
        eprintln!("skipping {label}: this build has no prefill image");
        return;
    }
    let _ = parity_on(
        config,
        label,
        prompt_len,
        PrefillPlan::new(CHUNK, true),
        TENSOR_CORE_TOLERANCE,
        false,
    );
}

/// A fourth geometry exercising all vectorised fast paths of the released model:
/// hc_count = 4, hc_lowrank divisible by 8 (16), hidden_size divisible by 32 (64),
/// and moe_intermediate_size divisible by 32 (32).
#[test]
fn a_vectorised_four_stream_geometry_matches_the_host_oracle() {
    let mut config = Qwen4ExpConfig::tiny();
    config.hidden_size = 64;
    config.hc_count = 4;
    config.hc_lowrank = 16;
    config.moe_intermediate_size = 32;
    config.shared_expert_intermediate_size = 32;
    let _ = parity(config, "vectorised_hc4");
}

/// A chunk wider than DENSE_ROWS computes every row to the bits DENSE_ROWS-wide chunks do: its
/// dense projections run a DENSE_ROWS piece at a time with the split such a chunk took, and a tail
/// too short for the tensor cores runs on its own, as it did then. The hidden width is wide enough
/// that one 2,048-row launch of the 32-row router would split K four ways where a 512-row one
/// splits it eight, so a piece computed any other way shows here. One prompt ends in a 30-row tail
/// (its own forward), the other in a 100-row one (a short piece of the wide forward).
#[test]
fn a_wide_chunk_computes_every_row_as_dense_rows_chunks_do() {
    if !cubin_available() || embedded_prefill_cubin().is_empty() {
        eprintln!("skipping the wide chunk: this build has no Qwen4-Exp cubin or prefill image");
        return;
    }
    let mut config = Qwen4ExpConfig::tiny();
    config.hidden_size = 2048;
    config.hc_count = 4;
    config.hc_lowrank = 32;
    config.layer_types = vec![LayerKind::Linear, LayerKind::Linear, LayerKind::Sparse];
    config.ple_layer_ids = vec![2];
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.linear_num_key_heads = 2;
    config.linear_num_value_heads = 6;
    config.moe_intermediate_size = 64;
    config.shared_expert_intermediate_size = 64;
    config.num_experts = 32;
    config.num_experts_per_tok = 4;
    config.max_position_embeddings = 2048;
    config.num_hidden_layers = config.layer_types.len();
    let temp = tempfile::tempdir().expect("a temporary directory");
    write_synthetic_checkpoint(temp.path(), &config, 11).expect("the fixture writes");
    for prompt_len in [3 * DENSE_ROWS + 30, 2 * DENSE_ROWS + 100] {
        let Some(narrow) = wide_chunk_logits(temp.path(), prompt_len, CHUNK) else {
            return;
        };
        let wide = wide_chunk_logits(temp.path(), prompt_len, 4 * DENSE_ROWS)
            .expect("the device opened once already");
        for (step, (narrow, wide)) in narrow.iter().zip(&wide).enumerate() {
            let differ = narrow
                .iter()
                .zip(wide)
                .filter(|(narrow, wide)| narrow.to_bits() != wide.to_bits())
                .count();
            assert_eq!(
                differ,
                0,
                "{prompt_len}-token prompt, step {step}: {differ} logits differ between {CHUNK}- \
                 and {}-token chunks",
                4 * DENSE_ROWS
            );
        }
    }
}

/// The prefill's logits and three decode steps' after it under `chunk`-token tensor-core chunks,
/// or None when this machine cannot run the device.
fn wide_chunk_logits(
    checkpoint: &std::path::Path,
    prompt_len: usize,
    chunk: usize,
) -> Option<Vec<Vec<f32>>> {
    let backend = match Qwen4ExpDeviceBackend::open_with_prefill(
        checkpoint,
        prompt_len + 8,
        PrefillPlan::new(chunk, true),
    ) {
        Ok(backend) => backend,
        Err(error) if error.code() == "qwen4_device" && error.message().contains("CUDA:") => {
            eprintln!("skipping the wide chunk: {error}");
            return None;
        }
        Err(error) => panic!("the device arm refused the fixture checkpoint: {error}"),
    };
    let mut session = backend.open_session().expect("a device session opens");
    let prompt: Vec<u32> = (0..prompt_len)
        .map(|index| 3 + (index * 7 % 13) as u32)
        .collect();
    let mut logits = vec![session.prefill(&prompt).expect("the prefill runs").logits];
    for token in [25u32, 27, 29] {
        logits.push(session.decode(token).expect("the decode runs").logits);
    }
    Some(logits)
}
