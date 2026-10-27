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

//! The host-path scaffolding shared by the batch-18 engine tests.
//!
//! These are the same provider and post hooks the TP=4 host driver builds (see
//! crates/sparkflash-cli/src/tp4_host.rs): the layer input goes through mHC and the attention norm,
//! and the post hook runs mHC, the FFN norm and the deferred handoff. The tests feed these to
//! InferenceEngine::run_hybrid_sequence_step so a sharded engine and a whole-checkpoint engine see
//! exactly the same host arithmetic and only the shard layout differs.
//!
//! The rank-order bus is a test-only BitGather: each rank contributes one segment per round and
//! every rank leaves with the concatenation in ascending rank order, which is the transport
//! contract the real TP=4 gather implements.

use std::sync::{Arc, Condvar, Mutex};

use super::CodedSource;

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::engram;
use sparkflash_engine::forward::layer_zero::{LayerZeroEntry, LayerZeroEntrySource};
use sparkflash_engine::forward::mhc::{self, Mixes};
use sparkflash_engine::forward::{
    DeferredHandoff, HybridEntry, HybridPost, InferenceEngine, SequenceState,
};
use sparkflash_engine::linalg::{self, EngineError};
use sparkflash_engine::parallel_tp::BitGather;
use sparkflash_engine::tensor::TensorPlan;
use sparkflash_engine::weights::WeightSource;

/// No element cap: the coded fixtures are small enough to materialise whole.
pub const LIMIT: usize = usize::MAX;
/// The fixture seed every test shares, so a reference and a shard decode the same values.
pub const SEED: u64 = 0x5eed;

/// The host MhcEntry handoff: the same provider shape the TP=4 host driver builds.
pub struct HostEntry<'a> {
    /// The model geometry.
    pub config: &'a ModelConfig,
    /// The rank source the provider reads its non-sharded names from.
    pub source: &'a dyn WeightSource,
}

impl HostEntry<'_> {
    /// Builds one layer entry from the hidden state and the pre-mix row.
    pub fn entry(
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

    /// Builds one layer entry from the deferred handoff (and the Engram rows for its layers).
    pub fn deferred_entry(
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

/// The host post hook: one x row and one normalized FFN row per position.
pub fn host_post(
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

/// A rank-order gather bus. Each rank contributes one segment per round; every rank leaves with
/// the concatenation in ascending rank order.
pub struct Bus {
    state: Mutex<BusState>,
    ready: Condvar,
}

#[derive(Default)]
struct BusState {
    round: u64,
    slots: Vec<Option<Vec<f32>>>,
    arrived: usize,
    result: Option<Vec<f32>>,
    copied: usize,
}

impl Bus {
    /// One bus for a world of ranks.
    pub fn new(world: usize) -> Self {
        Self {
            state: Mutex::new(BusState {
                slots: vec![None; world],
                ..BusState::default()
            }),
            ready: Condvar::new(),
        }
    }

    /// Contributes one rank segment and returns the rank-order concatenation.
    pub fn gather(&self, rank: usize, local: &[f32]) -> Vec<f32> {
        let mut state = self.state.lock().expect("bus lock");
        let round = state.round;
        state.slots[rank] = Some(local.to_vec());
        state.arrived += 1;
        if state.arrived == state.slots.len() {
            let mut result = Vec::new();
            for slot in &state.slots {
                result.extend_from_slice(slot.as_ref().expect("every rank arrived"));
            }
            state.result = Some(result);
            self.ready.notify_all();
        }
        while state.round == round && state.result.is_none() {
            state = self.ready.wait(state).expect("bus wait");
        }
        let result = state.result.as_ref().expect("gather result").clone();
        state.copied += 1;
        if state.copied == state.slots.len() {
            state.slots.fill(None);
            state.result = None;
            state.arrived = 0;
            state.copied = 0;
            state.round = round + 1;
            self.ready.notify_all();
        }
        result
    }
}

/// One rank BitGather over a [Bus].
pub struct BusGather {
    /// This rank.
    pub rank: usize,
    /// The world size.
    pub world: usize,
    /// The shared bus.
    pub bus: Arc<Bus>,
}

impl BitGather for BusGather {
    fn world(&self) -> usize {
        self.world
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn label(&self) -> &'static str {
        "test bus"
    }

    fn gather_bits(&mut self, local: &[f32], out: &mut [f32]) -> Result<(), EngineError> {
        let gathered = self.bus.gather(self.rank, local);
        if out.len() != gathered.len() {
            return Err(EngineError::Shape {
                what: "test bus gather".to_owned(),
                detail: format!("{} gathered into {}", gathered.len(), out.len()),
            });
        }
        out.copy_from_slice(&gathered);
        Ok(())
    }
}

/// One step's observable: the chosen token and the full logits.
#[derive(Debug)]
pub struct Step {
    /// The greedy token.
    pub token: Option<u32>,
    /// The full logits.
    pub logits: Vec<f32>,
}

/// The raw bit patterns of a slice, for an exactness assertion.
pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Steps tokens through one engine on the hybrid path and records every step.
pub fn run_steps(
    config: &ModelConfig,
    engine: &InferenceEngine,
    provider_source: &dyn WeightSource,
    tokens: &[u32],
) -> Result<Vec<Step>, EngineError> {
    let table = linalg::decode_by_name(provider_source, "embed.weight", LIMIT)?;
    let mut state = SequenceState::new(config);
    let mut steps = Vec::with_capacity(tokens.len());
    for (position, &token) in tokens.iter().enumerate() {
        let mut hashes: Vec<u32> = Vec::new();
        engram::ngram_hashes(config, &tokens[..position + 1], position, &mut hashes)?;
        let mut provider =
            |layer: usize, hidden: &[f32], pre_mix: &[f32], deferred: Option<&DeferredHandoff>| {
                HostEntry {
                    config,
                    source: provider_source,
                }
                .deferred_entry(layer, hidden, pre_mix, deferred, &hashes)
            };
        let mut post = |layer: usize, residual: &[f32], attention_out: &[f32], mixes: &Mixes| {
            host_post(
                config,
                provider_source,
                layer,
                residual,
                attention_out,
                mixes,
            )
        };
        let embedding = &table[token as usize * config.dim..(token as usize + 1) * config.dim];
        let output = engine.run_hybrid_sequence_step(
            &mut state,
            token,
            embedding,
            position,
            &mut provider,
            &mut post,
            None,
            None,
            &mut |_layer: usize, _host_ms: u64| {},
        )?;
        steps.push(Step {
            token: output.token,
            logits: output.logits,
        });
    }
    Ok(steps)
}

/// The whole-checkpoint reference steps over the coded fixture.
pub fn reference_steps(config: &ModelConfig, tokens: &[u32]) -> Vec<Step> {
    let plan = TensorPlan::language_model(config);
    let engine = InferenceEngine::new(
        config.clone(),
        Box::new(CodedSource::from_plan(&plan, SEED)),
        LIMIT,
    )
    .expect("reference engine");
    let provider = CodedSource::from_plan(&plan, SEED);
    run_steps(config, &engine, &provider, tokens).expect("reference steps")
}
