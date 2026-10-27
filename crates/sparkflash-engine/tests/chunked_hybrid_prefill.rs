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

//! A chunked hybrid step is bit-identical to the same tokens stepped one at a time.
//!
//! The chunk API is the TTFT path: one layer-major pass over a prompt chunk instead of one
//! decode-shaped forward per token. This test pins the property that makes it safe to use: the
//! chunk's last-position logits, head input, chosen token and advanced state are bit-for-bit what
//! the one-token loop produces, with the host FFN and with a world-4 device backend whose chunk
//! method reduces the same per-token payloads. Both weight sources are compared: the F32 fixture
//! (the exact old fallback path) and the plan's own coded F4-E2M1 experts / F8-E4M3 attention
//! weights, whose fused kernels are what the released checkpoint runs.

mod common;

use common::{CodedSource, FiniteSource};

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::attention;
use sparkflash_engine::forward::engram;
use sparkflash_engine::forward::layer_zero::{LayerZeroEntry, LayerZeroEntrySource};
use sparkflash_engine::forward::mhc::{self, Mixes};
use sparkflash_engine::forward::moe::{self, MoeDeviceBackend, MoeScratch};
use sparkflash_engine::forward::{
    DeferredHandoff, HybridEntry, HybridPost, InferenceEngine, SequenceState,
};
use sparkflash_engine::linalg::{self, EngineError};
use sparkflash_engine::parallel_tp::canonical_reduce_into;
use sparkflash_engine::tensor::TensorPlan;
use sparkflash_engine::weights::WeightSource;

const LIMIT: usize = usize::MAX;

/// The test's host MhcEntry handoff.
struct HostEntry<'a> {
    config: &'a ModelConfig,
    source: &'a dyn WeightSource,
}

impl HostEntry<'_> {
    fn entry(
        &self,
        layer: usize,
        hidden: &[f32],
        pre_mix: &[f32],
    ) -> Result<HybridEntry, EngineError> {
        let config = self.config;
        let source = self.source;
        let prefix = format!("layers.{layer}");
        let dim = config.dim;
        let fn_w = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_fn"), LIMIT)?;
        let scale = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_scale"), LIMIT)?;
        let base = linalg::decode_by_name(source, &format!("{prefix}.hc_attn_base"), LIMIT)?;
        let mut mixes = Mixes::for_config(config);
        mhc::hc_mixes(config, hidden, &fn_w, &scale, &base, &mut mixes)?;
        let mut attn_in = vec![0.0f32; dim];
        mhc::hc_pre(config, pre_mix, hidden, &mut attn_in)?;
        let norm = linalg::decode_by_name(source, &format!("{prefix}.attn_norm.weight"), LIMIT)?;
        let mut attention_hidden = attn_in;
        linalg::rms_norm(&mut attention_hidden, &norm, dim, config.norm_eps as f32)?;
        Ok(HybridEntry {
            entry: LayerZeroEntry {
                pre_attention_hidden: attention_hidden,
                pre: mixes.pre,
                post: mixes.post,
                comb: mixes.comb,
                source: LayerZeroEntrySource::Host,
            },
            x: hidden.to_vec(),
        })
    }

    /// The deferred form: materialise the previous layer's epilogue, apply Engram at its layer,
    /// then the entry.
    fn deferred_entry(
        &self,
        layer: usize,
        hidden: &[f32],
        pre_mix: &[f32],
        deferred: Option<&DeferredHandoff>,
        hashes: &[u32],
    ) -> Result<HybridEntry, EngineError> {
        let config = self.config;
        let width = config.hc_width();
        let mut x = if let Some(deferred) = deferred {
            let mut x = vec![0.0f32; width];
            mhc::hc_post(
                config,
                &deferred.y,
                &deferred.x,
                &deferred.mixes.post,
                &deferred.mixes.comb,
                &mut x,
            )?;
            x
        } else {
            hidden.to_vec()
        };
        if config.engram_rows(layer).is_some() {
            engram::engram_forward(config, layer, &mut x, hashes, self.source, LIMIT)?;
        }
        self.entry(layer, &x, pre_mix)
    }
}

/// The host post hook: one x row and one normalized FFN row per position, mixes in for_tokens
/// layout.
fn host_post(
    config: &ModelConfig,
    source: &dyn WeightSource,
    layer: usize,
    residual: &[f32],
    attention_out: &[f32],
    attn_mixes: &Mixes,
) -> Result<HybridPost, EngineError> {
    let prefix = format!("layers.{layer}");
    let dim = config.dim;
    let width = config.hc_width();
    let positions = residual.len() / width;
    let mut x = vec![0.0f32; residual.len()];
    mhc::hc_post(
        config,
        attention_out,
        residual,
        &attn_mixes.post,
        &attn_mixes.comb,
        &mut x,
    )?;
    let fn_w = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_fn"), LIMIT)?;
    let scale = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_scale"), LIMIT)?;
    let base = linalg::decode_by_name(source, &format!("{prefix}.hc_ffn_base"), LIMIT)?;
    let mut ffn_mixes = Mixes::for_tokens(config, positions);
    mhc::hc_mixes(config, &x, &fn_w, &scale, &base, &mut ffn_mixes)?;
    let mut ffn_in = vec![0.0f32; positions * dim];
    mhc::hc_pre(config, &attn_mixes.pre, &x, &mut ffn_in)?;
    let ffn_norm = linalg::decode_by_name(source, &format!("{prefix}.ffn_norm.weight"), LIMIT)?;
    linalg::rms_norm(&mut ffn_in, &ffn_norm, dim, config.norm_eps as f32)?;
    Ok(HybridPost {
        x,
        ffn_in,
        ffn_mixes,
        source: LayerZeroEntrySource::Host,
    })
}

/// A fake world-4 device backend: it computes the host routed block and then applies the canonical
/// reduce over four copies of that payload, so every rank's wire contribution is the same value and
/// the chunk method is exactly the per-token method.
struct FakeWorld4Backend {
    scratch: MoeScratch,
    dimension: usize,
}

impl FakeWorld4Backend {
    fn new(config: &ModelConfig) -> Self {
        Self {
            scratch: MoeScratch::for_config(config),
            dimension: config.dim,
        }
    }

    fn reduce_in_place(&self, values: &mut [f32]) -> Result<(), EngineError> {
        let payload = values.to_vec();
        let payloads: Vec<&[f32]> = core::iter::repeat_n(&payload[..], 4).collect();
        canonical_reduce_into(values, &payloads)
    }
}

impl MoeDeviceBackend for FakeWorld4Backend {
    fn ffn(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let _ = (layer, x.len());
        let mut routed = vec![0.0f32; out.len()];
        moe::routed_forward(
            config,
            prefix,
            x,
            source,
            limit,
            &mut self.scratch,
            &mut routed,
        )?;
        self.reduce_in_place(&mut routed)?;
        out.copy_from_slice(&routed);
        Ok(true)
    }

    fn ffn_chunk(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        tokens: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let dim = self.dimension;
        assert_eq!(tokens * dim, x.len());
        for token in 0..tokens {
            self.ffn(
                config,
                layer,
                prefix,
                &x[token * dim..(token + 1) * dim],
                source,
                limit,
                &mut out[token * dim..(token + 1) * dim],
            )?;
        }
        Ok(true)
    }

    fn shared_expert(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let _ = layer;
        moe::add_shared_expert(config, prefix, x, source, limit, &mut self.scratch, out)?;
        Ok(true)
    }

    fn shared_expert_chunk(
        &mut self,
        config: &ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        source: &dyn WeightSource,
        limit: usize,
        _tokens: usize,
        out: &mut [f32],
    ) -> Result<bool, EngineError> {
        let _ = layer;
        moe::add_shared_expert(config, prefix, x, source, limit, &mut self.scratch, out)?;
        Ok(true)
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The chunk and the token loop over the same tokens, with or without the fake world-4 backend,
/// on the F32 fixture or on the plan's coded F4/F8 dtypes.
fn compare_chunk_with_token_loop(config: ModelConfig, tokens: &[u32], coded: bool, backend: bool) {
    let plan = TensorPlan::language_model(&config);
    let host_source: Box<dyn WeightSource> = if coded {
        Box::new(CodedSource::from_plan(&plan, 0x5eed))
    } else {
        Box::new(FiniteSource::from_plan(&plan, 0x5eed))
    };
    let engine = InferenceEngine::new(
        config.clone(),
        if coded {
            Box::new(CodedSource::from_plan(&plan, 0x5eed))
        } else {
            Box::new(FiniteSource::from_plan(&plan, 0x5eed))
        },
        LIMIT,
    )
    .expect("engine");
    let source: &dyn WeightSource = host_source.as_ref();
    let table = linalg::decode_by_name(source, "embed.weight", LIMIT).expect("embed table");
    let embeddings: Vec<f32> = tokens
        .iter()
        .flat_map(|&token| {
            table[token as usize * config.dim..(token as usize + 1) * config.dim].to_vec()
        })
        .collect();

    let chunk_hashes = {
        let mut hashes = Vec::new();
        engram::ngram_hashes(&config, tokens, 0, &mut hashes).expect("chunk hashes");
        hashes
    };
    // One hash block per position: every Engram layer's columns for that token.
    assert!(chunk_hashes.len().is_multiple_of(tokens.len()));
    let hashes_per_token = chunk_hashes.len() / tokens.len();

    let mut chunk_state = SequenceState::new(&config);
    let mut chunk_backend = backend.then(|| FakeWorld4Backend::new(&config));
    let mut chunk_provider = |layer: usize,
                              position: usize,
                              hidden: &[f32],
                              pre_mix: &[f32],
                              deferred: Option<&DeferredHandoff>| {
        HostEntry {
            config: &config,
            source,
        }
        .deferred_entry(
            layer,
            hidden,
            pre_mix,
            deferred,
            &chunk_hashes[position * hashes_per_token..(position + 1) * hashes_per_token],
        )
    };
    let mut chunk_post = |layer: usize, residual: &[f32], attention_out: &[f32], mixes: &Mixes| {
        host_post(&config, source, layer, residual, attention_out, mixes)
    };
    let mut progress = |_layer: usize, _host_ms: u64| {};
    let chunk = engine
        .run_hybrid_sequence_chunk(
            &mut chunk_state,
            tokens,
            &embeddings,
            0,
            &mut chunk_provider,
            &mut chunk_post,
            None,
            chunk_backend
                .as_mut()
                .map(|backend| backend as &mut dyn MoeDeviceBackend),
            &[],
            &mut progress,
        )
        .expect("chunked hybrid step");

    let mut token_state = SequenceState::new(&config);
    let mut token_backend = backend.then(|| FakeWorld4Backend::new(&config));
    let mut last = None;
    for (position, &token) in tokens.iter().enumerate() {
        let mut token_hashes: Vec<u32> = Vec::new();
        engram::ngram_hashes(
            &config,
            &tokens[..position + 1],
            position,
            &mut token_hashes,
        )
        .expect("token hashes");
        assert_eq!(token_hashes.len(), hashes_per_token);
        let mut provider =
            |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
                HostEntry {
                    config: &config,
                    source,
                }
                .deferred_entry(layer, hidden, pre_mix, deferred, &token_hashes)
            };
        let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], mixes: &Mixes| {
            host_post(&config, source, layer, residual, attention_out, mixes)
        };
        let embedding = &embeddings[position * config.dim..(position + 1) * config.dim];
        last = Some(
            engine
                .run_hybrid_sequence_step(
                    &mut token_state,
                    token,
                    embedding,
                    position,
                    &mut provider,
                    &mut post,
                    None,
                    token_backend
                        .as_mut()
                        .map(|backend| backend as &mut dyn MoeDeviceBackend),
                    &mut progress,
                )
                .expect("one-token hybrid step"),
        );
    }
    let last = last.expect("at least one token step");

    assert_eq!(chunk_state.tokens(), token_state.tokens());
    assert_eq!(chunk.token, last.token, "chunk token differs");
    assert_eq!(
        bits(&chunk.head_input),
        bits(&last.head_input),
        "head input differs"
    );
    assert_eq!(bits(&chunk.logits), bits(&last.logits), "logits differ");
}

#[test]
fn a_chunked_hybrid_step_is_bit_identical_to_the_one_token_loop_on_the_host_ffn() {
    compare_chunk_with_token_loop(ModelConfig::tiny(), &[3u32, 5, 7], false, false);
}

#[test]
fn a_chunked_hybrid_step_is_bit_identical_to_the_one_token_loop_with_a_world4_backend() {
    compare_chunk_with_token_loop(ModelConfig::tiny(), &[3u32, 5, 7], false, true);
}

/// The same comparison on the plan's own coded F4-E2M1 experts and F8-E4M3 attention weights:
/// the released kernels' per-row arithmetic must also be independent of the chunk.
#[test]
fn a_chunked_hybrid_step_is_bit_identical_to_the_one_token_loop_on_coded_weights() {
    compare_chunk_with_token_loop(ModelConfig::tiny(), &[3u32, 5, 7], true, false);
}

#[test]
fn a_chunked_hybrid_step_is_bit_identical_to_the_one_token_loop_on_coded_weights_with_a_world4_backend()
 {
    compare_chunk_with_token_loop(ModelConfig::tiny(), &[3u32, 5, 7], true, true);
}
/// A wider coded configuration than [ModelConfig::tiny]: the F4 expert kernel crosses its
/// parallel-multiply threshold and the chunk holds more than one compression block, which the tiny
/// fixture (a single serial 32-wide dot) cannot exercise. This is the shape regime the released
/// checkpoint runs, at a size a unit test can generate.
fn wide_coded_config() -> ModelConfig {
    let mut config = ModelConfig::tiny();
    config.dim = 512;
    config.moe_inter_dim = 256;
    config.n_layers = 8;
    // The released decoder is 64 heads of 512 (wide = 32768) against dim 5120, and hc_mult 4: the
    // attention write is wider than dim and the mHC stream holds four copies.
    config.n_heads = 8;
    config.head_dim = 128;
    config.q_lora_rank = 128;
    config.o_groups = 8;
    config.o_lora_rank = 64;
    config.hc_mult = 4;
    config.window_size = 8;
    config.vocab_size = 128;
    config.engram_head_dim = 8;
    config.engram_n_heads = 2;
    // One ratio per backbone layer plus the MTP stage. The released model is a mix of ratio 0 (no
    // compressor), 2 and 1; twenty of its forty backbone layers compress every single token, and
    // the index sources after the last key owner reuse its ratio-1 keys, so this mirrors that
    // structure at test size.
    config.compress_ratios = vec![0, 0, 2, 2, 2, 1, 1, 1, 0];
    config.kv_source_layers = vec![2, 5];
    config.index_source_layers = vec![2, 5, 6, 7];
    config.candidate_source_layer = 5;
    config
}

#[test]
fn a_chunked_hybrid_step_is_bit_identical_on_a_wider_coded_config() {
    let tokens = [3u32, 7, 11, 5, 13, 2, 9, 4, 6, 1, 12, 15, 8, 10, 14, 0, 5];
    compare_chunk_with_token_loop(wide_coded_config(), &tokens, true, false);
}

/// A small-memory stand-in for the released backbone's conditioning pattern: ratio-0 layers, a
/// ratio-2 block, then ratio-1 layers whose index sources share the keys of the last key owner, with
/// a candidate-block channel. This is the structure the checkpoint runs and the only one that
/// diverges between a chunk and the same prompt stepped token by token.
fn mixed_coded_config() -> ModelConfig {
    let mut config = ModelConfig::tiny();
    config.dim = 256;
    config.moe_inter_dim = 128;
    config.n_layers = 8;
    config.n_heads = 4;
    config.head_dim = 64;
    config.q_lora_rank = 64;
    config.o_groups = 4;
    config.o_lora_rank = 64;
    config.hc_mult = 4;
    config.window_size = 8;
    config.vocab_size = 128;
    config.engram_head_dim = 8;
    config.engram_n_heads = 2;
    config.compress_ratios = vec![0, 0, 2, 2, 2, 1, 1, 1, 0];
    config.kv_source_layers = vec![2, 5];
    config.index_source_layers = vec![2, 5, 6, 7];
    config.candidate_source_layer = 5;
    config
}

fn mixed_tokens() -> Vec<u32> {
    (0..37u32).map(|index| (index * 7 + 3) % 127).collect()
}

#[test]
fn a_chunked_hybrid_step_is_bit_identical_on_a_real_like_conditioning() {
    compare_chunk_with_token_loop(mixed_coded_config(), &mixed_tokens(), true, false);
}

#[test]
fn a_two_token_chunk_is_bit_identical_on_a_real_like_conditioning() {
    compare_chunk_with_token_loop(mixed_coded_config(), &[3, 5], true, false);
}

#[test]
fn a_chunked_hybrid_step_is_bit_identical_on_a_real_like_conditioning_with_a_world4_backend() {
    compare_chunk_with_token_loop(mixed_coded_config(), &mixed_tokens(), true, true);
}

/// The shared channel must hold one KV source's **whole** cache, not just the rows the last call
/// added: a consumer layer that runs before this step's source would otherwise read the previous
/// step's last publisher. The hole is observable on the token loop alone, with no chunk involved.
#[test]
fn the_shared_channel_holds_the_whole_cache_of_its_owner_on_the_token_loop() {
    let config = mixed_coded_config();
    let tokens: Vec<u32> = (0..5u32).map(|index| (index * 7 + 3) % 127).collect();
    let plan = TensorPlan::language_model(&config);
    let host_source: Box<dyn WeightSource> = Box::new(CodedSource::from_plan(&plan, 0x5eed));
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(CodedSource::from_plan(&plan, 0x5eed)),
        LIMIT,
    )
    .expect("engine");
    let source: &dyn WeightSource = host_source.as_ref();
    let table = linalg::decode_by_name(source, "embed.weight", LIMIT).expect("embed table");
    attention::reset_shared_attention();
    let mut state = SequenceState::new(&config);
    for (position, &token) in tokens.iter().enumerate() {
        let mut hashes: Vec<u32> = Vec::new();
        engram::ngram_hashes(&config, &tokens[..position + 1], position, &mut hashes)
            .expect("token hashes");
        let mut provider =
            |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
                HostEntry {
                    config: &config,
                    source,
                }
                .deferred_entry(layer, hidden, pre_mix, deferred, &hashes)
            };
        let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], mixes: &Mixes| {
            host_post(&config, source, layer, residual, attention_out, mixes)
        };
        let embedding = &table[token as usize * config.dim..(token as usize + 1) * config.dim];
        engine
            .run_hybrid_sequence_step(
                &mut state,
                token,
                embedding,
                position,
                &mut provider,
                &mut post,
                None,
                None,
                &mut |_layer: usize, _host_ms: u64| {},
            )
            .expect("one-token step");
    }
    // Layer 5 is the last KV source in this config and therefore the channel's final owner.
    let owner = 5usize;
    let (compressed_owner, compressed_rows, compressed) = attention::published_compressed_rows();
    assert_eq!(compressed_owner, Some(owner), "the channel owner");
    let cache = &state.attention(owner).expect("owner state").compressed;
    assert_eq!(
        compressed_rows,
        cache.len() / config.head_dim,
        "the channel must publish the owner's whole cache"
    );
    assert_eq!(
        bits(&compressed),
        bits(cache),
        "the channel's compressed rows must be the owner's cache"
    );
    let (keys_owner, keys_rows, keys) = attention::published_index_keys();
    assert_eq!(keys_owner, Some(owner), "the key channel owner");
    let key_cache = &state.attention(owner).expect("owner state").indexer;
    assert_eq!(keys_rows, key_cache.len() / config.index_head_dim);
    assert_eq!(
        bits(&keys),
        bits(key_cache),
        "the channel's index keys must be the owner's cache"
    );
}
