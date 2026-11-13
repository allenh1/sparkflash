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
use sparkflash_weights::nvfp4::dequantize_nvfp4;
use sparkflash_weights::quant::{decode_f8e4m3, f4e2m1_to_f32, f8e4m3_to_f32};

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
        weights.check_shape("model.llm.embed.weight", &[weights.config.vocab_size, weights.config.hidden_size])?;
        weights.check_shape("model.llm.unembed.weight", &[weights.config.vocab_size, weights.config.hidden_size])?;
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
                values.len(), rows, columns
            ));
        }
        Ok(values)
    }

    fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        self.source.read(name).map_err(|error| format!("{name}: {error}"))
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
                    .ok_or_else(|| format!("{name}: range {offset}..{} is outside the tensor", offset + len))
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

    fn expert_matrix(
        &self,
        layer: usize,
        expert: usize,
        which: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<f32>, String> {
        let prefix = format!("model.llm.layers.{layer}.mlp.experts.{which}");
        match self.config.layer_format(layer).unwrap_or(WeightFormat::Bf16) {
            WeightFormat::NvFp4 { group_size } => {
                let amax = self.vector(&format!("{prefix}_weight_amax"))?;
                let outer = self.vector(&format!("{prefix}_weight_outer_scale_denominator"))?;
                let packed = columns.div_ceil(2);
                let scale_columns = columns.div_ceil(group_size);
                let code_stride = rows * packed;
                let scale_stride = rows * scale_columns;
                let start = expert * code_stride;
                let scale_start = expert * scale_stride;
                let codes = self.range(&format!("{prefix}_weight"), start, code_stride)?;
                let scales = self.range(&format!("{prefix}_weight_scale"), scale_start, scale_stride)?;
                let scale_2 = nvfp4_outer_scale(
                    amax[expert],
                    outer.first().copied().unwrap_or(1.0),
                );
                dequantize_nvfp4(&codes, &scales, scale_2, rows, columns)
                    .map_err(|error| format!("{prefix}: {error}"))
            }
            WeightFormat::MxFp8 { block } => {
                let scale_columns = columns.div_ceil(block);
                let code_stride = rows * columns;
                let scale_stride = rows * scale_columns;
                let code_start = expert * code_stride;
                let scale_start = expert * scale_stride;
                let codes = self.range(&format!("{prefix}_weight"), code_start, code_stride)?;
                let scales = self.range(&format!("{prefix}_weight_scale"), scale_start, scale_stride)?;
                let mut out = vec![0f32; rows * columns];
                for row in 0..rows {
                    let codes = &codes[row * columns..(row + 1) * columns];
                    let scales = &scales[row * scale_columns..(row + 1) * scale_columns];
                    let out_row = &mut out[row * columns..(row + 1) * columns];
                    for (column, value) in out_row.iter_mut().enumerate() {
                        // One E4M3 byte times its block's ue8m0 power of two. Decoding inline
                        // matters here: an E4M3 decode per element through a one-byte slice was
                        // sixteen times slower than the NVFP4 rows on the same checkpoint.
                        *value = f8e4m3_to_f32(codes[column]) * e8m0_scale(scales[column / block]);
                    }
                }
                Ok(out)
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
                Ok(values[expert * stride..(expert + 1) * stride].to_vec())
            }
            other => Err(format!("layer {layer} expert format {other:?} is not supported by the host")),
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
    pub fn forward_token(
        &self,
        token: u32,
        state: &mut Inkling2State,
    ) -> Result<Vec<f32>, String> {
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
        let embed_norm = self.vector("model.llm.embed_norm.weight")?;
        x = rms_norm(&x, Some(&embed_norm), config.architecture.norm_epsilon);
        trace_timing!(format!("token={token} embed"), step);

        let position = state.position;
        for layer in 0..config.num_hidden_layers {
            let start = std::time::Instant::now();
            x = self.layer(layer, position, x, state)?;
            trace_timing!(format!("token={token} layer={layer}"), start);
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
        let final_norm = self.vector("model.llm.norm.weight")?;
        let x = rms_norm(&final_input[0], Some(&final_norm), config.architecture.norm_epsilon);
        let mut logits = self.matvec_rows(
            "model.llm.unembed.weight",
            &x,
            vocab,
            config.hidden_size,
            4096,
        )?;
        if let Some(cap) = config.final_logit_softcap {
            for value in &mut logits {
                *value = (*value / cap as f32).tanh() * cap as f32;
            }
        }
        logits.truncate(config.unpadded_vocab_size);
        trace_timing!(format!("token={token} head"), head);
        state.position += 1;
        Ok(logits)
    }

    fn layer(
        &self,
        layer: usize,
        position: usize,
        hidden: Vec<f32>,
        state: &mut Inkling2State,
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
            let count = if config.full_layer_ids.contains(&layer) { 3 } else { 1 };
            block_residual(
                &hidden,
                &state.block_states,
                count,
                Some(&queries),
                config.attn_res_normalize_values,
                config.architecture.norm_epsilon,
            )?
        };

        let (q_input, k_input, v_input) = if sliding {
            let norm = self.vector(&format!("{prefix}.attn_norm.weight"))?;
            let x = rms_norm(&hidden, Some(&norm), config.architecture.norm_epsilon);
            (x.clone(), x.clone(), x)
        } else {
            let qn = self.vector(&format!("{prefix}.q_input_norm.weight"))?;
            let kn = self.vector(&format!("{prefix}.k_input_norm.weight"))?;
            let vn = self.vector(&format!("{prefix}.v_input_norm.weight"))?;
            (
                rms_norm(&attention_input[0], Some(&qn), config.architecture.norm_epsilon),
                rms_norm(&attention_input.get(1).unwrap_or(&attention_input[0]), Some(&kn), config.architecture.norm_epsilon),
                rms_norm(&attention_input.get(2).unwrap_or(&attention_input[0]), Some(&vn), config.architecture.norm_epsilon),
            )
        };

        let stage = std::time::Instant::now();
        let attn = self.attention(layer, position, sliding, &q_input, &k_input, &v_input, state)?;
        trace_timing!(format!("layer={layer} attention"), stage);
        let attn_norm = self.vector(&format!("{prefix}.attn_norm.weight")).ok();
        let _ = attn_norm;
        let attn = self.conv(&format!("{prefix}.attn_sconv.weight"), &attn, None, state, layer, 2, sliding)?;
        let residual = if layer % config.n_layers_per_attn_res_block == 0 {
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
        let mlp_in = rms_norm(&mlp_input[0], Some(&norm), config.architecture.norm_epsilon);
        let stage = std::time::Instant::now();
        let mlp = self.moe(layer, &mlp_in)?;
        trace_timing!(format!("layer={layer} moe"), stage);
        let mlp = self.conv(&format!("{prefix}.mlp_sconv.weight"), &mlp, None, state, layer, 3, sliding)?;
        Ok(add(&residual, &mlp))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
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
        let q = self.matvec_name(&format!("{prefix}.wq_du.weight"), q_input, heads * head_dim, h)?;
        let mut k = self.matvec_name(&format!("{prefix}.wk_dv.weight"), k_input, kv_width, h)?;
        let mut v = self.matvec_name(&format!("{prefix}.wv_dv.weight"), v_input, kv_width, h)?;
        let rel = self.matvec_name(&format!("{prefix}.wr_du.weight"), q_input, heads * config.d_rel, h)?;
        let gate = self.matvec_name(&format!("{prefix}.gate_attn.weight"), q_input, heads * head_dim, h)?;
        let sink = self.matvec_name(&format!("{prefix}.attn_sink_proj.weight"), q_input, heads, h)?;

        if sliding {
            k = self.conv(&format!("{prefix}.k_sconv.weight"), &k, Some(k_input), state, layer, 0, sliding)?;
            v = self.conv(&format!("{prefix}.v_sconv.weight"), &v, Some(v_input), state, layer, 1, sliding)?;
        } else {
            k = self.conv_dynamic(&format!("{prefix}.k_sconv"), &k, k_input, state, layer, 0)?;
            v = self.conv_dynamic(&format!("{prefix}.v_sconv"), &v, v_input, state, layer, 1)?;
        }

        let q_scale = self.vector(&format!("{prefix}.q_l2_clip.scale"))?;
        let rel_scale = self.vector(&format!("{prefix}.rel_l2_clip.scale"))?;
        let mut q = q;
        for head in 0..heads {
            let row = &mut q[head * head_dim..(head + 1) * head_dim];
            for (value, scale) in row.iter_mut().zip(&q_scale[head * head_dim..(head + 1) * head_dim]) {
                *value *= *scale;
            }
            clip_l2(row, config.qk_l2_clip_threshold as f32);
        }
        for head in 0..kv_heads {
            clip_l2(&mut k[head * head_dim..(head + 1) * head_dim], config.qk_l2_clip_threshold as f32);
            rms_norm_in_place(&mut v[head * head_dim..(head + 1) * head_dim], config.architecture.norm_epsilon);
        }
        let mut rel = rel;
        for head in 0..heads {
            let row = &mut rel[head * config.d_rel..(head + 1) * config.d_rel];
            for (value, scale) in row.iter_mut().zip(&rel_scale[head * config.d_rel..(head + 1) * config.d_rel]) {
                *value *= *scale;
            }
            clip_l2(row, config.rel_l2_clip_threshold as f32);
        }
        let tau = if sliding {
            1.0
        } else if config.log_scaling_n_floor > 0 {
            let ratio = ((position + 1) as f32 / config.log_scaling_n_floor as f32).max(1.0);
            1.0 + config.log_scaling_alpha as f32 * ratio.ln()
        } else {
            1.0
        };
        for value in &mut q {
            *value *= tau;
        }
        for value in &mut rel {
            *value *= tau;
        }

        let cache = &mut state.layers[layer];
        cache.k.push(k);
        cache.v.push(v);
        let start = if sliding {
            cache.k.len().saturating_sub(config.sliding_window_size)
        } else {
            0
        };
        let rel_bank = self.vector(&format!("{prefix}.rel_logits_proj.proj"))?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0f32; heads * head_dim];
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            let mut scores = Vec::with_capacity(cache.k.len() - start);
            for key in start..cache.k.len() {
                let mut dot = 0.0f32;
                for dim in 0..head_dim {
                    dot += q[head * head_dim + dim] * cache.k[key][kv_head * head_dim + dim];
                }
                let distance = position - key;
                let rel_extent = if sliding {
                    config.sliding_window_size
                } else {
                    config.rel_extent
                };
                let bias = if distance < rel_extent {
                    let mut value = 0.0f32;
                    for rel_dim in 0..config.d_rel {
                        value += rel[head * config.d_rel + rel_dim]
                            * rel_bank[rel_dim * config.rel_extent + distance];
                    }
                    value
                } else {
                    0.0
                };
                scores.push(dot * scale + bias);
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                .max(sink[head]);
            let mut denominator = (sink[head] - max).exp();
            for score in &mut scores {
                *score = (*score - max).exp();
                denominator += *score;
            }
            for (index, score) in scores.iter().enumerate() {
                let weight = *score / denominator;
                let key = start + index;
                for dim in 0..head_dim {
                    output[head * head_dim + dim] += weight * cache.v[key][kv_head * head_dim + dim];
                }
            }
        }
        for (value, gate) in output.iter_mut().zip(&gate) {
            *value *= 1.0 / (1.0 + (-gate).exp());
        }
        self.matvec_name(&format!("{prefix}.wo_ud.weight"), &output, h, heads * head_dim)
    }

    fn matvec_name(&self, name: &str, input: &[f32], rows: usize, columns: usize) -> Result<Vec<f32>, String> {
        let weight = self.matrix(name, rows, columns)?;
        Ok(matvec_parallel(&weight, input, rows, columns))
    }

    fn conv(
        &self,
        weight_name: &str,
        input: &[f32],
        _dynamic: Option<&[f32]>,
        state: &mut Inkling2State,
        layer: usize,
        slot: usize,
        sliding: bool,
    ) -> Result<Vec<f32>, String> {
        let width = input.len();
        let kernel = self.config.sconv_kernel_size;
        let weights = self.matrix(weight_name, width, kernel)?;
        let history = &mut state.layers[layer].conv[slot];
        let mut out = vec![0f32; width];
        for channel in 0..width {
            for tap in 0..kernel {
                let value = if tap < history.len() {
                    history[history.len() - 1 - tap][channel]
                } else {
                    0.0
                };
                // The checkpoint's conv weight index runs oldest-first (PyTorch conv1d with
                // padding K-1); tap here is the backward distance, current token first.
                out[channel] += weights[channel * kernel + (kernel - 1 - tap)] * value;
            }
            out[channel] += input[channel];
        }
        for value in &mut out {
            *value /= ((kernel + 1) as f32).sqrt();
        }
        history.push(input.to_vec());
        if history.len() > kernel - 1 {
            history.remove(0);
        }
        let _ = sliding;
        Ok(out)
    }

    fn conv_dynamic(
        &self,
        prefix: &str,
        input: &[f32],
        dynamic: &[f32],
        state: &mut Inkling2State,
        layer: usize,
        slot: usize,
    ) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let kernel = config.sconv_kernel_size;
        let static_w = self.matrix(&format!("{prefix}.static_w"), kernel, kv_width)?;
        let projected = self.matvec_name(&format!("{prefix}.weight_proj"), dynamic, config.num_key_value_heads * kernel, config.hidden_size)?;
        let history = &mut state.layers[layer].conv[slot];
        let mut out = vec![0f32; kv_width];
        for head in 0..config.num_key_value_heads {
            for channel in 0..config.head_dim {
                let index = head * config.head_dim + channel;
                for tap in 0..kernel {
                    let value = if tap < history.len() {
                        history[history.len() - 1 - tap][index]
                    } else {
                        0.0
                    };
                    let dynamic_tap = projected[head * kernel + tap];
                    out[index] += (static_w[tap * kv_width + index] + dynamic_tap) * value;
                }
                out[index] += input[index];
            }
        }
        for value in &mut out {
            *value /= ((2 * kernel + 1) as f32).sqrt();
        }
        history.push(input.to_vec());
        if history.len() > kernel - 1 {
            history.remove(0);
        }
        Ok(out)
    }

    fn moe(&self, layer: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        let config = &self.config;
        let prefix = format!("model.llm.layers.{layer}.mlp");
        let gate = self.matrix(&format!("{prefix}.gate.weight"), config.n_routed_experts, config.hidden_size)?;
        let bias = self.vector(&format!("{prefix}.gate.bias"))?;
        let logits = matvec_parallel(&gate, input, config.n_routed_experts, config.hidden_size);
        let scores: Vec<f32> = logits.iter().map(|value| 1.0 / (1.0 + (-value).exp())).collect();
        let mut selected: Vec<usize> = (0..config.n_routed_experts).collect();
        selected.sort_by(|a, b| (scores[*b] + bias[*b]).total_cmp(&(scores[*a] + bias[*a])));
        selected.truncate(config.num_experts_per_tok);
        let sum: f32 = selected.iter().map(|index| scores[*index]).sum();
        let mixes: Vec<f32> = selected
            .iter()
            .map(|expert| scores[*expert] / (sum + 1e-20) * config.route_scale as f32)
            .collect();
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
                    let gate_up = this.expert_matrix(
                        layer,
                        expert,
                        "w13",
                        intermediate,
                        config.hidden_size,
                    )?;
                    let down = this.expert_matrix(
                        layer,
                        expert,
                        "w2",
                        config.hidden_size,
                        config.moe_intermediate_size,
                    )?;
                    let gate_up = matvec(&gate_up, input, intermediate, config.hidden_size);
                    let mut act = vec![0f32; config.moe_intermediate_size];
                    for index in 0..config.moe_intermediate_size {
                        let g = gate_up[index];
                        let u = gate_up[index + config.moe_intermediate_size];
                        let sg = (g / config.mlp_gate_softcap as f32).tanh()
                            * config.mlp_gate_softcap as f32;
                        let su = (u / config.mlp_up_softcap as f32).tanh()
                            * config.mlp_up_softcap as f32;
                        act[index] = sg * (1.0 / (1.0 + (-g).exp())) * su;
                    }
                    Ok((slot, matvec(&down, &act, config.hidden_size, config.moe_intermediate_size)))
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
    conv: [Vec<Vec<f32>>; 4],
}

impl Inkling2State {
    /// A fresh state.
    #[must_use]
    pub fn new(config: &Inkling2Config) -> Self {
        Self {
            position: 0,
            block_states: Vec::new(),
            layers: (0..config.num_hidden_layers).map(|_| LayerState::default()).collect(),
        }
    }

    /// The number of tokens consumed.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }
}

/// ModelOpt's NVFP4 outer scale.
///
/// The released Inkling groups record weight_scale_format amax_denominator and store a
/// per-expert amax plus a per-tensor denominator. NVIDIA's NVFP4 reference divides the amax by
/// six times that denominator; this is the one place the host asserts that convention, so a
/// converted checkpoint with a different producer can be corrected here rather than in three
/// call sites.
fn nvfp4_outer_scale(amax: f32, denominator: f32) -> f32 {
    let denominator = if denominator == 0.0 { 1.0 } else { denominator };
    amax / (6.0 * denominator)
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

fn clip_l2(input: &mut [f32], threshold: f32) {
    let norm = input.iter().map(|value| value * value).sum::<f32>().sqrt();
    let scale = (threshold / (norm + 1e-6)).min(1.0);
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
    let query = if queries == 1 { query.to_vec() } else { query.to_vec() };
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
