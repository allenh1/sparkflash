// Copyright (c) 2026 Hunter L. Allen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The Inkling-2-Small f32 reference forward.
//!
//! This is the parity oracle for the device arm, not a throughput path. It reads each tensor
//! through [sparkflash_weights::WeightSource] when a step needs it, so a released 393 GB
//! checkpoint is correct but slow; the tiny synthetic fixture is the fast test.

use std::sync::Arc;

use sparkflash_model::architecture::WeightFormat;
use sparkflash_weights::WeightSource;
use sparkflash_weights::linalg::decode_by_name;
use sparkflash_weights::quant::{decode_f8e4m3, f4e2m1_to_f32};

use crate::config::Inkling2Config;

const ELEMENT_LIMIT: usize = 1 << 30;

/// Whether the host reference prints per-stage timings.
///
/// Set `SPARKFLASH_INKLING_TRACE=1` to see where a released-checkpoint step spends its time;
/// the reference reads a tensor only when a step needs it, so the stage split is the only way to
/// tell a slow decode from a slow read.
#[must_use]
pub fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("SPARKFLASH_INKLING_TRACE")
            .is_some_and(|value| value != "0" && !value.is_empty())
    })
}

macro_rules! trace_timing {
    ($label:expr, $start:expr) => {
        if trace_on() {
            eprintln!(
                "[inkling] {} {:.1} ms",
                $label,
                $start.elapsed().as_secs_f64() * 1e3
            );
        }
    };
}

/// The read-only weights of one checkpoint.
pub struct Inkling2Weights {
    config: Inkling2Config,
    source: Arc<dyn WeightSource>,
}

impl std::fmt::Debug for Inkling2Weights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Inkling2Weights")
            .field("family", &self.config.architecture.family)
            .field("layers", &self.config.num_hidden_layers)
            .finish()
    }
}

impl Inkling2Weights {
    /// Open the weights over a source. Nothing is materialised here beyond the embedding and
    /// head shapes, so a large checkpoint opens cheaply.
    ///
    /// # Errors
    /// A missing or mis-shaped embedding/head, or a tensor the source cannot describe.
    pub fn load(config: Inkling2Config, source: Arc<dyn WeightSource>) -> Result<Self, String> {
        let weights = Self { config, source };
        weights.check_shape(
            "model.llm.embed.weight",
            &[weights.config.vocab_size, weights.config.hidden_size],
        )?;
        weights.check_shape(
            "model.llm.unembed.weight",
            &[weights.config.vocab_size, weights.config.hidden_size],
        )?;
        weights.check_shape("model.llm.norm.weight", &[weights.config.hidden_size])?;
        Ok(weights)
    }

    /// The typed configuration.
    #[must_use]
    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }

    fn check_shape(&self, name: &str, shape: &[usize]) -> Result<(), String> {
        let spec = self
            .source
            .spec(name)
            .map_err(|error| format!("{name}: {error}"))?;
        if spec.shape != shape {
            return Err(format!("{name} is {:?}, expected {shape:?}", spec.shape));
        }
        Ok(())
    }

    fn vector(&self, name: &str) -> Result<Vec<f32>, String> {
        decode_by_name(self.source.as_ref(), name, ELEMENT_LIMIT)
            .map_err(|error| format!("{name}: {error}"))
    }

    /// One row of a two-dimensional tensor, read through a byte range so a 1.85 GB embedding
    /// table is never materialised on the host path.
    fn row(&self, name: &str, row: usize, columns: usize) -> Result<Vec<f32>, String> {
        let spec = self
            .source
            .spec(name)
            .map_err(|error| format!("{name}: {error}"))?;
        let bytes = spec.dtype.bytes_per_element();
        let start = (row * columns * bytes) as u64;
        let ranges = [(start, columns * bytes)];
        let raw = match self.source.read_ranges(name, &ranges) {
            Some(Ok(bytes)) => bytes,
            Some(Err(error)) => return Err(format!("{name}: {error}")),
            None => self.raw(name)?[start as usize..start as usize + columns * bytes].to_vec(),
        };
        sparkflash_weights::quant::decode(spec.dtype, &raw, columns)
            .map_err(|error| format!("{name} row {row}: {error}"))
    }

    /// A matrix-vector product over a large row-major table, streamed in chunks.
    fn matvec_rows(
        &self,
        name: &str,
        input: &[f32],
        rows: usize,
        columns: usize,
        chunk: usize,
    ) -> Result<Vec<f32>, String> {
        let spec = self
            .source
            .spec(name)
            .map_err(|error| format!("{name}: {error}"))?;
        let bytes = spec.dtype.bytes_per_element();
        let mut out = Vec::with_capacity(rows);
        let mut row = 0;
        while row < rows {
            let count = chunk.min(rows - row);
            let start = (row * columns * bytes) as u64;
            let ranges = [(start, count * columns * bytes)];
            let raw = match self.source.read_ranges(name, &ranges) {
                Some(Ok(bytes)) => bytes,
                Some(Err(error)) => return Err(format!("{name}: {error}")),
                None => {
                    let all = self.raw(name)?;
                    all[start as usize..start as usize + count * columns * bytes].to_vec()
                }
            };
            let values = sparkflash_weights::quant::decode(spec.dtype, &raw, count * columns)
                .map_err(|error| format!("{name}: {error}"))?;
            out.extend(matvec(&values, input, count, columns));
            row += count;
        }
        Ok(out)
    }

    fn matrix(&self, name: &str, rows: usize, columns: usize) -> Result<Vec<f32>, String> {
        let values = self.vector(name)?;
        if values.len() != rows * columns {
            return Err(format!(
                "{name} decoded {} values, expected {}x{}",
                values.len(),
                rows,
                columns
            ));
        }
        Ok(values)
    }

    fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        self.source
            .read(name)
            .map_err(|error| format!("{name}: {error}"))
    }

    /// One byte range of a tensor, for an expert tensor whose whole allocation is over a
    /// source's single-read limit.
    fn range(&self, name: &str, offset: usize, len: usize) -> Result<Vec<u8>, String> {
        let ranges = [(offset as u64, len)];
        match self.source.read_ranges(name, &ranges) {
            Some(Ok(bytes)) => Ok(bytes),
            Some(Err(error)) => Err(format!("{name}: {error}")),
            None => {
                let all = self.raw(name)?;
                all.get(offset..offset + len)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| {
                        format!(
                            "{name}: range {offset}..{} is outside the tensor",
                            offset + len
                        )
                    })
            }
        }
    }

    #[allow(dead_code)]
    fn mxfp8_matrix(
        &self,
        prefix: &str,
        rows: usize,
        columns: usize,
        block: usize,
    ) -> Result<Vec<f32>, String> {
        let codes = self.raw(&format!("{prefix}.weight"))?;
        let scales = self.raw(&format!("{prefix}.weight_scale"))?;
        let scale_columns = columns.div_ceil(block);
        if codes.len() != rows * columns || scales.len() != rows * scale_columns {
            return Err(format!(
                "{prefix}: MXFP8 shapes do not match {rows}x{columns} / block {block}"
            ));
        }
        let mut out = vec![0f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let code = decode_f8e4m3(&[codes[row * columns + column]])[0];
                let exponent = scales[row * scale_columns + column / block];
                out[row * columns + column] = code * e8m0_scale(exponent);
            }
        }
        Ok(out)
    }

    /// One expert's matrix-vector product, streamed from the checkpoint.
    ///
    /// The weights are never materialised: the codes and scales for the one expert are read and
    /// multiplied straight into the product, which is what keeps a routed expert a few megabytes
    /// of I/O instead of a four-gigabyte buffer.
    fn expert_matvec(
        &self,
        layer: usize,
        expert: usize,
        which: &str,
        input: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, String> {
        let prefix = format!("model.llm.layers.{layer}.mlp.experts.{which}");
        match self
            .config
            .layer_format(layer)
            .unwrap_or(WeightFormat::Bf16)
        {
            WeightFormat::NvFp4 { group_size, .. } => {
                let amax = self.vector(&format!("{prefix}_weight_amax"))?;
                let outer = self.vector(&format!("{prefix}_weight_outer_scale_denominator"))?;
                let code_stride = rows * columns.div_ceil(2);
                let scale_stride = rows * columns.div_ceil(group_size);
                let codes = self.range(
                    &format!("{prefix}_weight"),
                    expert * code_stride,
                    code_stride,
                )?;
                let scales = self.range(
                    &format!("{prefix}_weight_scale"),
                    expert * scale_stride,
                    scale_stride,
                )?;
                let scale_2 =
                    nvfp4_outer_scale(amax[expert], outer.first().copied().unwrap_or(1.0));
                crate::reference::nvfp4_matvec(&codes, &scales, scale_2, input, rows, columns)
                    .map_err(|error| format!("{prefix}: {error}"))
            }
            WeightFormat::MxFp8Row { block } => {
                let code_stride = rows * columns;
                let scale_stride = rows * columns.div_ceil(block);
                let codes = self.range(
                    &format!("{prefix}_weight"),
                    expert * code_stride,
                    code_stride,
                )?;
                let scales = self.range(
                    &format!("{prefix}_weight_scale"),
                    expert * scale_stride,
                    scale_stride,
                )?;
                crate::reference::mxfp8_matvec(&codes, &scales, input, rows, columns, block)
                    .map_err(|error| format!("{prefix}: {error}"))
            }
            WeightFormat::Bf16 => {
                // A synthetic fixture stores the experts as one dense BF16 tensor
                // [experts, rows, columns].
                let values = self.vector(&format!("{prefix}_weight"))?;
                let stride = rows * columns;
                let total = self.config.n_routed_experts * stride;
                if values.len() != total {
                    return Err(format!(
                        "{prefix}_weight decoded {} values, expected {total}",
                        values.len()
                    ));
                }
                Ok(matvec(
                    &values[expert * stride..(expert + 1) * stride],
                    input,
                    rows,
                    columns,
                ))
            }
            other => Err(format!(
                "layer {layer} expert format {other:?} is not supported by the host"
            )),
        }
    }

    /// Prefill a prompt into a fresh state and return the final token's logits.
    ///
    /// # Errors
    /// As [Self::forward_token].
    pub fn forward_prompt(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("the prompt is empty".to_owned());
        }
        let mut state = Inkling2State::new(&self.config);
        let mut logits = Vec::new();
        for token in tokens {
            logits = self.forward_token(*token, &mut state)?;
        }
        Ok(logits)
    }

    /// One step: consume a token at the state's next position and return the scored logits.
    ///
    /// # Errors
    /// A missing/mis-shaped tensor, or a position outside the model's window.
    pub fn forward_token(&self, token: u32, state: &mut Inkling2State) -> Result<Vec<f32>, String> {
        let config = &self.config;
        if state.position >= config.context_tokens() {
            return Err(format!(
                "position {} is outside the {} token window",
                state.position,
                config.context_tokens()
            ));
        }
        let vocab = config.vocab_size;
        if token as usize >= vocab {
            return Err(format!("token {token} is outside the vocabulary {vocab}"));
        }
        let step = std::time::Instant::now();
        let mut x = self.row("model.llm.embed.weight", token as usize, config.hidden_size)?;
        if trace_on() {
            let rms = (x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32).sqrt();
            eprintln!(
                "[inkling] host embed_row rms={rms:.5} first={:.5} second={:.5}",
                x[0], x[1]
            );
        }
        let embed_norm = self.vector("model.llm.embed_norm.weight")?;
        x = rms_norm(&x, Some(&embed_norm), config.architecture.norm_epsilon);
        trace_timing!(format!("token={token} embed"), step);

        let position = state.position;
        // The block-summary stack is this token's residual chain: one entry a block boundary,
        // read by the layers after it and discarded when the token is done. Keeping it across
        // tokens would grow it a block group a token and feed the next token's sublayers an
        // older token's stream, which is not a residual the released model has.
        state.block_states.clear();
        // A looped release runs its core more than once. The schedule below is the producer's
        // (modeling_inkling2_loop.py:1390-1470): the prelude once, the core `passes` times and
        // the coda once. Every pass after the first restarts from the prelude's hidden state and
        // block stack, and receives a message from the pass before it.
        if let Some(plan) = config.loop_plan {
            let schedule = config.applications();
            let mut prelude_hidden: Option<Vec<f32>> = None;
            let mut prelude_blocks: Vec<Vec<f32>> = Vec::new();
            let mut previous: Option<(Vec<Vec<f32>>, Vec<f32>)> = None;
            let mut running_pass = 0usize;
            for (layer, pass) in schedule {
                if pass > running_pass {
                    // The pass just finished: take the summary the message reads, then rewind to
                    // the prelude. The producer drops the prelude's blocks and the first block
                    // the pass appended (`:1419`), and keeps the rest with the pass's output.
                    let skip = prelude_blocks.len() + 1;
                    let blocks = state.block_states.get(skip..).unwrap_or_default().to_vec();
                    previous = Some((blocks, x.clone()));
                    x = prelude_hidden
                        .clone()
                        .expect("the prelude ran before the core");
                    state.block_states = prelude_blocks.clone();
                    running_pass = pass;
                }
                let messages = match (&previous, pass) {
                    (Some((blocks, hidden)), p) if p > 0 => {
                        Some(self.cross_pass(layer - plan.core_start, blocks, hidden)?)
                    }
                    _ => None,
                };
                let start = std::time::Instant::now();
                let cache = config.cache_slot(layer, pass);
                x = self.layer(cache, layer, position, x, state, messages.as_ref())?;
                if trace_on() {
                    let rms =
                        (x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32).sqrt();
                    eprintln!(
                        "[inkling] host   layer {layer} pass {pass} x0={:.5} x1={:.5} rms={rms:.5}",
                        x[0], x[1]
                    );
                }
                trace_timing!(format!("token={token} layer={layer} pass={pass}"), start);
                if pass == 0 && layer + 1 == plan.core_start {
                    prelude_hidden = Some(x.clone());
                    prelude_blocks = state.block_states.clone();
                }
            }
        } else {
            for layer in 0..config.num_hidden_layers {
                let start = std::time::Instant::now();
                x = self.layer(layer, layer, position, x, state, None)?;
                if trace_on() {
                    let rms =
                        (x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32).sqrt();
                    eprintln!(
                        "[inkling] host   layer {layer} x0={:.5} x1={:.5} rms={rms:.5}",
                        x[0], x[1]
                    );
                }
                trace_timing!(format!("token={token} layer={layer}"), start);
            }
        }
        let head = std::time::Instant::now();
        let final_input = block_residual(
            &x,
            &state.block_states,
            1,
            Some(&self.vector("model.llm.attn_res.output_query")?),
            config.attn_res_normalize_values,
            config.architecture.norm_epsilon,
        )?;
        if trace_on() {
            let rms = (final_input[0]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / final_input[0].len() as f32)
                .sqrt();
            eprintln!(
                "[inkling] host final_input rms={rms:.5} first={:.5}",
                final_input[0][0]
            );
        }
        let final_norm = self.vector("model.llm.norm.weight")?;
        let x = rms_norm(
            &final_input[0],
            Some(&final_norm),
            config.architecture.norm_epsilon,
        );
        let mut logits = self.matvec_rows(
            "model.llm.unembed.weight",
            &x,
            vocab,
            config.hidden_size,
            4096,
        )?;
        if trace_on() {
            let rms = (x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32).sqrt();
            eprintln!("[inkling] host final_norm rms={rms:.5} first={:.5}", x[0]);
            eprintln!(
                "[inkling] host raw_logits rms={:.5} first={:.5}",
                (logits
                    .iter()
                    .take(4)
                    .map(|value| value * value)
                    .sum::<f32>()
                    / 4.0)
                    .sqrt(),
                logits[0]
            );
        }
        if let Some(cap) = config.final_logit_softcap {
            let capped = crate::reference::logit_softcap(
                &logits,
                1,
                vocab,
                config.unpadded_vocab_size,
                cap as f32,
            );
            logits = capped;
        } else {
            logits.truncate(config.unpadded_vocab_size);
        }
        trace_timing!(format!("token={token} head"), head);
        state.position += 1;
        Ok(logits)
    }

    /// The four messages a core layer consumes on a pass after the first.
    ///
    /// `modeling_inkling2_loop.py:147-176`: the previous pass's block summaries and its final
    /// hidden state are stacked, unweighted-RMS normed, and read by four learned queries (one
    /// a stream) with a `hidden**-0.5` scale. There is no mask: the read is over a handful of
    /// summaries, not over positions.
    fn cross_pass(
        &self,
        loop_index: usize,
        blocks: &[Vec<f32>],
        hidden: &[f32],
    ) -> Result<LoopMessages, String> {
        let config = &self.config;
        let epsilon = config.architecture.norm_epsilon;
        let mut states: Vec<Vec<f32>> = blocks.to_vec();
        states.push(hidden.to_vec());
        for state in &mut states {
            *state = unweighted_rms(state, epsilon);
        }
        let scale = 1.0 / (config.hidden_size as f32).sqrt();
        let attend = |query: Vec<f32>| -> Result<Vec<f32>, String> {
            let mut scores: Vec<f32> = states
                .iter()
                .map(|state| {
                    query
                        .iter()
                        .zip(state)
                        .map(|(left, right)| left * right)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let largest = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut total = 0.0f32;
            for score in &mut scores {
                *score = (*score - largest).exp();
                total += *score;
            }
            let mut out = vec![0.0f32; config.hidden_size];
            for (weight, state) in scores.iter().zip(&states) {
                let weight = weight / total;
                for (out, value) in out.iter_mut().zip(state) {
                    *out += weight * value;
                }
            }
            Ok(out)
        };
        // The producer's bank is `(core layers, 4, hidden)` in query, key, value, MLP order. The
        // checkpoint splits it into two 24-row tensors; which two streams each holds is a
        // conversion fact the layout records, and this is the split it states: the query and
        // MLP streams in `middle_cycle_cross_pass_queries`, the key and value streams in
        // `middle_cycle_cross_pass_kv_queries`.
        let row = |bank: &str, sub: usize| self.row(bank, loop_index * 2 + sub, config.hidden_size);
        let q = attend(row("model.llm.middle_cycle_cross_pass_queries", 0)?)?;
        let mlp = attend(row("model.llm.middle_cycle_cross_pass_queries", 1)?)?;
        let k = attend(row("model.llm.middle_cycle_cross_pass_kv_queries", 0)?)?;
        let v = attend(row("model.llm.middle_cycle_cross_pass_kv_queries", 1)?)?;
        Ok(LoopMessages { q, k, v, mlp })
    }

    /// One stream's message gate: `(1 - tanh(forget)) * source + tanh(gain) * message`.
    ///
    /// `modeling_inkling2_loop.py:81-99`: the gates are projected from an unweighted-RMSNorm of
    /// the concatenated source and message through a rank-bottlenecked down projection. The
    /// producer builds the gate parameters in f32 and casts at the boundary; the checkpoint
    /// stores them BF16, which is the type the layout states and the loader decodes.
    fn message_gate(
        &self,
        layer: usize,
        stream: &str,
        source: &[f32],
        message: &[f32],
    ) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let epsilon = config.architecture.norm_epsilon;
        let hidden = config.hidden_size;
        let rank = config.loop_plan.map_or(0, |plan| plan.gate_rank);
        let prefix = format!("model.llm.layers.{layer}.controllers.{stream}");
        let down = self.matrix(&format!("{prefix}.down_weight"), rank, 2 * hidden)?;
        let forget = self.matrix(&format!("{prefix}.forget_weight"), hidden, rank)?;
        let gain = self.matrix(&format!("{prefix}.gain_weight"), hidden, rank)?;
        let mut joined = unweighted_rms(source, epsilon);
        joined.extend(unweighted_rms(message, epsilon));
        let bottleneck = matvec(&down, &joined, rank, 2 * hidden);
        let forget = matvec(&forget, &bottleneck, hidden, rank);
        let gain = matvec(&gain, &bottleneck, hidden, rank);
        Ok(source
            .iter()
            .zip(&forget)
            .zip(&gain)
            .zip(message)
            .map(|(((source, forget), gain), message)| {
                (1.0 - forget.tanh()) * source + gain.tanh() * message
            })
            .collect())
    }

    fn layer(
        &self,
        cache: usize,
        layer: usize,
        position: usize,
        hidden: Vec<f32>,
        state: &mut Inkling2State,
        messages: Option<&LoopMessages>,
    ) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let prefix = format!("model.llm.layers.{layer}");
        let sliding = config.local_layer_ids.contains(&layer);
        let block_residual_state = hidden.clone();
        let attention_input = if layer == 0 {
            vec![hidden.clone()]
        } else {
            // The released bank is addressed by the layer before the one that consumes it:
            // bank 4 is the three-query input of full layer 5.
            let queries_name = format!("model.llm.attn_res.attn_queries.{}", layer - 1);
            let queries = self.vector(&queries_name)?;
            let count = if config.full_layer_ids.contains(&layer) {
                3
            } else {
                1
            };
            block_residual(
                &hidden,
                &state.block_states,
                count,
                Some(&queries),
                config.attn_res_normalize_values,
                config.architecture.norm_epsilon,
            )?
        };

        // A later pass folds the previous pass's message into each sublayer input before the
        // input norm (modeling_inkling2_loop.py:1001-1012, 1152-1160). The gate is per stream:
        // the query path, the key path, the value path and the MLP.
        let gated = |stream: &str, source: &[f32], message: Option<&Vec<f32>>| match message {
            Some(message) => self.message_gate(layer, stream, source, message),
            None => Ok(source.to_vec()),
        };
        let (q_input, k_input, v_input) = if sliding {
            let norm = self.vector(&format!("{prefix}.attn_norm.weight"))?;
            // The block-attention residual is read on a sliding layer too: `attention_input` is the
            // softmax mix over the summary stack, and it is the layer's attention input, not the
            // raw stream. At layer 0 the two coincide because there is no stack yet.
            let source = gated("attn", &attention_input[0], messages.map(|held| &held.q))?;
            let x = rms_norm(&source, Some(&norm), config.architecture.norm_epsilon);
            (x.clone(), x.clone(), x)
        } else {
            let qn = self.vector(&format!("{prefix}.q_input_norm.weight"))?;
            let kn = self.vector(&format!("{prefix}.k_input_norm.weight"))?;
            let vn = self.vector(&format!("{prefix}.v_input_norm.weight"))?;
            let q_source = gated("attn", &attention_input[0], messages.map(|held| &held.q))?;
            let k_source = gated(
                "k",
                attention_input.get(1).unwrap_or(&attention_input[0]),
                messages.map(|held| &held.k),
            )?;
            let v_source = gated(
                "v",
                attention_input.get(2).unwrap_or(&attention_input[0]),
                messages.map(|held| &held.v),
            )?;
            (
                rms_norm(&q_source, Some(&qn), config.architecture.norm_epsilon),
                rms_norm(&k_source, Some(&kn), config.architecture.norm_epsilon),
                rms_norm(&v_source, Some(&vn), config.architecture.norm_epsilon),
            )
        };

        let stage = std::time::Instant::now();
        let attn = self.attention(
            cache, layer, position, sliding, &q_input, &k_input, &v_input, state,
        )?;
        if trace_on() {
            let rms =
                (attn.iter().map(|value| value * value).sum::<f32>() / attn.len() as f32).sqrt();
            eprintln!(
                "[inkling] host attn_proj layer {layer} rms={rms:.6} first={:.6} second={:.6}",
                attn[0], attn[1]
            );
        }
        trace_timing!(format!("layer={layer} attention"), stage);
        let attn_norm = self.vector(&format!("{prefix}.attn_norm.weight")).ok();
        let _ = attn_norm;
        let attn = self.conv(
            &format!("{prefix}.attn_sconv.weight"),
            &attn,
            state,
            cache,
            2,
        )?;
        let residual = if layer.is_multiple_of(config.n_layers_per_attn_res_block) {
            state.block_states.push(block_residual_state);
            attn.clone()
        } else {
            add(&block_residual_state, &attn)
        };

        let mlp_queries = self.vector(&format!("model.llm.attn_res.mlp_queries.{layer}"))?;
        let mlp_input = block_residual(
            &residual,
            &state.block_states,
            1,
            Some(&mlp_queries),
            config.attn_res_normalize_values,
            config.architecture.norm_epsilon,
        )?;
        let norm = self.vector(&format!("{prefix}.mlp_norm.weight"))?;
        let mlp_source = gated("mlp", &mlp_input[0], messages.map(|held| &held.mlp))?;
        let mlp_in = rms_norm(&mlp_source, Some(&norm), config.architecture.norm_epsilon);
        let stage = std::time::Instant::now();
        let mlp = self.moe(layer, &mlp_in)?;
        trace_timing!(format!("layer={layer} moe"), stage);
        let mlp = self.conv(&format!("{prefix}.mlp_sconv.weight"), &mlp, state, cache, 3)?;
        Ok(add(&residual, &mlp))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        cache: usize,
        layer: usize,
        position: usize,
        sliding: bool,
        q_input: &[f32],
        k_input: &[f32],
        v_input: &[f32],
        state: &mut Inkling2State,
    ) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let prefix = format!("model.llm.layers.{layer}.attn");
        let h = config.hidden_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let kv_width = kv_heads * head_dim;
        let q = self.matvec_name(
            &format!("{prefix}.wq_du.weight"),
            q_input,
            heads * head_dim,
            h,
        )?;
        let mut k = self.matvec_name(&format!("{prefix}.wk_dv.weight"), k_input, kv_width, h)?;
        let mut v = self.matvec_name(&format!("{prefix}.wv_dv.weight"), v_input, kv_width, h)?;
        let rel = self.matvec_name(
            &format!("{prefix}.wr_du.weight"),
            q_input,
            heads * config.d_rel,
            h,
        )?;
        let gate = self.matvec_name(
            &format!("{prefix}.gate_attn.weight"),
            q_input,
            heads * head_dim,
            h,
        )?;
        let sink = self.matvec_name(
            &format!("{prefix}.attn_sink_proj.weight"),
            q_input,
            heads,
            h,
        )?;

        if sliding {
            k = self.conv(&format!("{prefix}.k_sconv.weight"), &k, state, cache, 0)?;
            v = self.conv(&format!("{prefix}.v_sconv.weight"), &v, state, cache, 1)?;
        } else {
            k = self.conv_dynamic(&format!("{prefix}.k_sconv"), &k, k_input, state, cache, 0)?;
            v = self.conv_dynamic(&format!("{prefix}.v_sconv"), &v, v_input, state, cache, 1)?;
        }

        let q_scale = self.vector(&format!("{prefix}.q_l2_clip.scale"))?;
        let rel_scale = self.vector(&format!("{prefix}.rel_l2_clip.scale"))?;
        let tau = if sliding {
            1.0
        } else if config.log_scaling_n_floor > 0 {
            let ratio = ((position + 1) as f32 / config.log_scaling_n_floor as f32).max(1.0);
            1.0 + config.log_scaling_alpha as f32 * ratio.ln()
        } else {
            1.0
        };
        if trace_on() {
            for head in 0..heads {
                eprintln!(
                    "[inkling] host head {head} q={:.6} {:.6} g={:.6} s={:.6} r={:.6}",
                    q[head * head_dim],
                    q[head * head_dim + 1],
                    gate[head * head_dim],
                    sink[head],
                    rel[head * config.d_rel]
                );
            }
            for head in 0..kv_heads {
                eprintln!(
                    "[inkling] host kv {head} k={:.6} {:.6} v={:.6} {:.6}",
                    k[head * head_dim],
                    k[head * head_dim + 1],
                    v[head * head_dim],
                    v[head * head_dim + 1]
                );
            }
        }
        let layer_state = &mut state.layers[cache];
        layer_state.k.push(k);
        layer_state.v.push(v);
        let start = if sliding {
            layer_state
                .k
                .len()
                .saturating_sub(config.sliding_window_size)
        } else {
            0
        };
        let rel_bank = self.vector(&format!("{prefix}.rel_logits_proj.proj"))?;
        let rel_extent = if sliding {
            config.sliding_window_size
        } else {
            config.rel_extent
        };
        // The window is applied by handing the oracle the cached rows it may see, so the decode
        // and prefill kernels' `window` argument has exactly this meaning.
        let output = crate::reference::attention(
            &q,
            &gate,
            &layer_state.k[start..],
            &layer_state.v[start..],
            &rel,
            &rel_bank,
            &sink,
            &q_scale,
            None,
            &rel_scale,
            &crate::reference::AttentionParams {
                heads,
                kv_heads,
                head_dim,
                d_rel: config.d_rel,
                rel_extent,
                qk_clip: config.qk_l2_clip_threshold as f32,
                rel_clip: config.rel_l2_clip_threshold as f32,
                tau,
                value_norm: true,
                eps: config.architecture.norm_epsilon,
            },
        )?;
        self.matvec_name(
            &format!("{prefix}.wo_ud.weight"),
            &output,
            h,
            heads * head_dim,
        )
    }

    fn matvec_name(
        &self,
        name: &str,
        input: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, String> {
        let weight = self.matrix(name, rows, columns)?;
        Ok(matvec_parallel(&weight, input, rows, columns))
    }

    /// One decode step of a static short convolution.
    ///
    /// The state is `[channels, kernel - 1]` oldest first, the layout the decode kernel keeps,
    /// and the arithmetic is [crate::reference::short_conv_decode].
    fn conv(
        &self,
        weight_name: &str,
        input: &[f32],
        state: &mut Inkling2State,
        cache: usize,
        slot: usize,
    ) -> Result<Vec<f32>, String> {
        let width = input.len();
        let kernel = self.config.sconv_kernel_size;
        let weights = self.matrix(weight_name, width, kernel)?;
        let history = &mut state.layers[cache].conv[slot];
        Ok(crate::reference::short_conv_decode(
            input, &weights, history, width, kernel,
        ))
    }

    /// One decode step of the dynamic short convolution.
    fn conv_dynamic(
        &self,
        prefix: &str,
        input: &[f32],
        dynamic: &[f32],
        state: &mut Inkling2State,
        cache: usize,
        slot: usize,
    ) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let kernel = config.sconv_kernel_size;
        let static_w = self.matrix(&format!("{prefix}.static_w"), kernel, kv_width)?;
        let projected = self.matvec_name(
            &format!("{prefix}.weight_proj"),
            dynamic,
            config.num_key_value_heads * kernel,
            config.hidden_size,
        )?;
        let history = &mut state.layers[cache].conv[slot];
        Ok(crate::reference::short_conv_dynamic_decode(
            input,
            &static_w,
            &projected,
            history,
            kv_width,
            kernel,
            config.head_dim,
        ))
    }

    fn moe(&self, layer: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let prefix = format!("model.llm.layers.{layer}.mlp");
        let gate = self.matrix(
            &format!("{prefix}.gate.weight"),
            config.n_routed_experts,
            config.hidden_size,
        )?;
        let bias = self.vector(&format!("{prefix}.gate.bias"))?;
        let logits = matvec_parallel(&gate, input, config.n_routed_experts, config.hidden_size);
        // The router is the oracle's: a sigmoid a expert, the learned bias on the selection, and
        // the selected scores' shares under route_scale.
        let (selected, mixes) = crate::reference::router_topk(
            &logits,
            &bias,
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.route_scale as f32,
        );
        let intermediate = 2 * config.moe_intermediate_size;
        // The routed band is a handful of independent experts, each one a dequantise plus two
        // products. They run on every core; the mix below stays in router order so the f32 sum is
        // the sequential one, not a thread-order-dependent one.
        let mut pieces: Vec<(usize, Vec<f32>)> = Vec::with_capacity(selected.len());
        std::thread::scope(|scope| -> Result<(), String> {
            let mut handles = Vec::with_capacity(selected.len());
            for (slot, expert) in selected.iter().copied().enumerate() {
                let this = &*self;
                handles.push(scope.spawn(move || -> Result<(usize, Vec<f32>), String> {
                    let gate_up = this.expert_matvec(
                        layer,
                        expert,
                        "w13",
                        input,
                        intermediate,
                        config.hidden_size,
                    )?;
                    let act = crate::reference::softcapped_swiglu(
                        &gate_up,
                        config.moe_intermediate_size,
                        config.mlp_gate_softcap as f32,
                        config.mlp_up_softcap as f32,
                    );
                    Ok((
                        slot,
                        this.expert_matvec(
                            layer,
                            expert,
                            "w2",
                            &act,
                            config.hidden_size,
                            config.moe_intermediate_size,
                        )?,
                    ))
                }));
            }
            for handle in handles {
                pieces.push(handle.join().unwrap()?);
            }
            Ok(())
        })?;
        pieces.sort_by_key(|(slot, _)| *slot);
        let mut out = vec![0f32; config.hidden_size];
        for ((_, value), weight) in pieces.iter().zip(&mixes) {
            for (out, value) in out.iter_mut().zip(value) {
                *out += weight * value;
            }
        }
        Ok(out)
    }
}

/// The four sublayer inputs one loop pass hands the next.
#[derive(Debug, Clone)]
struct LoopMessages {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    mlp: Vec<f32>,
}

/// An unweighted RMSNorm in f32, as the loop's message path applies it.
fn unweighted_rms(input: &[f32], epsilon: f64) -> Vec<f32> {
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean + epsilon as f32).sqrt().recip();
    input.iter().map(|value| value * scale).collect()
}

/// One session's growing state.
#[derive(Debug, Clone)]
pub struct Inkling2State {
    position: usize,
    block_states: Vec<Vec<f32>>,
    layers: Vec<LayerState>,
}

#[derive(Debug, Clone, Default)]
struct LayerState {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// The four short convolutions' histories, `[channels, kernel - 1]` oldest first, in slot
    /// order: k, v, attention output, mlp output.
    conv: [Vec<f32>; 4],
}

impl Inkling2State {
    /// A fresh state.
    #[must_use]
    pub fn new(config: &Inkling2Config) -> Self {
        let history = config.conv_history();
        let kv_width = config.num_key_value_heads * config.head_dim;
        let hidden = config.hidden_size;
        Self {
            position: 0,
            block_states: Vec::new(),
            layers: (0..config.cache_slots())
                .map(|_| LayerState {
                    k: Vec::new(),
                    v: Vec::new(),
                    conv: [
                        vec![0f32; kv_width * history],
                        vec![0f32; kv_width * history],
                        vec![0f32; hidden * history],
                        vec![0f32; hidden * history],
                    ],
                })
                .collect(),
        }
    }

    /// The number of tokens consumed.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }
}

/// ModelOpt's NVFP4 outer scale: `amax / outer_scale_denominator`.
///
/// The released groups store a per-expert amax and one per-tensor denominator, and that
/// denominator is already six times the E4M3 maximum (the checkpoint carries 2,688 = 6 * 448), so
/// the `/ 6` the NVIDIA reference folds into it is inside the stored number. Dividing by it twice
/// scales every routed expert's weight by a sixth: the decoded matrix then has an rms of 0.0025
/// where the layer's BF16 projections have 0.0147, which is how this was found.
fn nvfp4_outer_scale(amax: f32, denominator: f32) -> f32 {
    let denominator = if denominator == 0.0 { 1.0 } else { denominator };
    amax / denominator
}

/// A ue8m0 scale code: the power of two 2^(code - 127).
///
/// E8M0 has no zero; code 0 is the smallest exponent. The released MXFP8 layers store one of
/// these codes per 32-element block, as a U8 tensor.
fn e8m0_scale(code: u8) -> f32 {
    2f32.powi(i32::from(code) - 127)
}

fn matvec(weight: &[f32], input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows];
    for row in 0..rows {
        let mut value = 0.0f32;
        for column in 0..columns {
            value += weight[row * columns + column] * input[column];
        }
        out[row] = value;
    }
    out
}

/// [matvec] spread over the machine's cores.
///
/// Every row is an independent dot product, so splitting rows changes no accumulation order and
/// the result is bit-for-bit what [matvec] returns; only the wall clock moves.
fn matvec_parallel(weight: &[f32], input: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(rows);
    if threads <= 1 || rows < 256 {
        return matvec(weight, input, rows, columns);
    }
    let mut out = vec![0f32; rows];
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest = out.as_mut_slice();
        let mut start = 0usize;
        while start < rows {
            let count = chunk.min(rows - start);
            let (head, tail) = rest.split_at_mut(count);
            let rows_here = &weight[start * columns..(start + count) * columns];
            scope.spawn(move || {
                for (row, slot) in head.iter_mut().enumerate() {
                    let line = &rows_here[row * columns..(row + 1) * columns];
                    let mut value = 0.0f32;
                    for (weight, input) in line.iter().zip(input) {
                        value += weight * input;
                    }
                    *slot = value;
                }
            });
            rest = tail;
            start += count;
        }
    });
    out
}

fn add(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

fn rms_norm(input: &[f32], weight: Option<&[f32]>, epsilon: f64) -> Vec<f32> {
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean + epsilon as f32).sqrt();
    input
        .iter()
        .enumerate()
        .map(|(index, value)| value * scale * weight.map_or(1.0, |weight| weight[index]))
        .collect()
}

fn rms_norm_in_place(input: &mut [f32], epsilon: f64) {
    let mean = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean + epsilon as f32).sqrt();
    for value in input {
        *value *= scale;
    }
}

fn block_residual(
    partial: &[f32],
    block_states: &[Vec<f32>],
    queries: usize,
    query: Option<&[f32]>,
    normalize: bool,
    epsilon: f64,
) -> Result<Vec<Vec<f32>>, String> {
    let hidden = partial.len();
    let mut stack: Vec<Vec<f32>> = block_states.to_vec();
    stack.push(partial.to_vec());
    if normalize {
        for state in &mut stack {
            rms_norm_in_place(state, epsilon);
        }
    }
    let query = query.ok_or_else(|| "block residual has no query".to_owned())?;
    let mut out = Vec::with_capacity(queries);
    for q in 0..queries {
        let offset = if queries == 1 { 0 } else { q * hidden };
        let mut scores = Vec::with_capacity(stack.len());
        for state in &stack {
            let mut dot = 0.0f32;
            for index in 0..hidden {
                dot += query[offset + index] * state[index];
            }
            scores.push(dot / (hidden as f32).sqrt());
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter().map(|score| (score - max).exp()).sum();
        let mut value = vec![0f32; hidden];
        for (state, score) in stack.iter().zip(scores) {
            let weight = (score - max).exp() / sum;
            for (out, item) in value.iter_mut().zip(state) {
                *out += weight * item;
            }
        }
        out.push(value);
    }
    Ok(out)
}

/// A value re-exported for the host's FP4 test fixture.
#[doc(hidden)]
#[must_use]
pub fn decode_fp4(code: u8) -> f32 {
    f4e2m1_to_f32(code & 0x0F)
}

#[cfg(test)]
mod tests {
    use super::{matvec, matvec_parallel};

    /// The parallel product exists only to move the wall clock, so it must agree with the
    /// sequential one bit for bit: same rows, same accumulation order, same f32 result.
    #[test]
    fn the_parallel_product_is_the_sequential_product() {
        let rows = 1024;
        let columns = 96;
        let weight: Vec<f32> = (0..rows * columns)
            .map(|index| ((index * 37 % 401) as f32 - 200.0) / 128.0)
            .collect();
        let input: Vec<f32> = (0..columns)
            .map(|index| ((index * 53 % 97) as f32 - 48.0) / 64.0)
            .collect();
        let sequential = matvec(&weight, &input, rows, columns);
        let parallel = matvec_parallel(&weight, &input, rows, columns);
        assert_eq!(sequential, parallel);
    }

    /// A narrow table takes the sequential path; it still has to be the same product.
    #[test]
    fn a_narrow_table_takes_the_sequential_path() {
        let weight: Vec<f32> = (0..4 * 8).map(|index| index as f32).collect();
        let input: Vec<f32> = (0..8).map(|index| index as f32 * 0.5).collect();
        assert_eq!(
            matvec(&weight, &input, 4, 8),
            matvec_parallel(&weight, &input, 4, 8)
        );
    }
}
