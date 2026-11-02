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

//! The layer blocks the model's own stack and the MTP layer share.
//!
//! The MTP module's one layer is a sparse-attention + MoE layer like the stack's, with its own
//! caches (the slot after the last layer) and FP8 rather than NVFP4 routed experts. It used to
//! carry its own copy of every launch below; both now run these, so a chunking or kernel change
//! lands in one place.
//!
//! # Attribution
//!
//! * The layer structure (Qwen Sparse Attention's indexer, block selection and gated GQA; the
//!   512-expert top-10 MoE with a sigmoid-gated shared expert) originates from
//!   [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Alibaba Cloud Qwen
//!   team) and the vLLM reference implementation
//!   ([vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0). See the root
//!   NOTICE file.

use sparkflash_kernels::cuda_driver::Deviceptr;
use sparkflash_model::BackendError;

use super::prefill::ATTENTION_CHUNK;
use super::{
    ATTN_GROUP_HEADS, ATTN_GROUP_SELECTED, DeviceSequence, GEMM_MIN_BATCH, GEMV_MAX_BATCH,
    GpuBranch, GpuExperts, GpuMoe, MAX_SELECTED_TOKENS,
};

/// The routed slots q4e_moe_permute sorts in shared memory (Q4E_MAX_PERMUTE).
const MAX_PERMUTE_SLOTS: usize = 4096;
/// The indexer blocks one q4e_qsa_scores CTA scores for a row (Q4E_QSA_TILE).
const QSA_TILE: usize = 64;

impl DeviceSequence {
    /// One sparse-attention block over `batch` rows of `mixed`, whose positions start at
    /// `position`: the projections, the chunk's keys and values appended to the caches in `slot`,
    /// the indexer's block selection and the gated attention over it. Leaves the block's output in
    /// `self.branch`.
    ///
    /// The block is three calls so a batch of sequences can share the two row-independent halves
    /// and run only the state-dependent one per row: [Self::sparse_project], [Self::sparse_attend]
    /// and [Self::sparse_output]. This path runs them back to back, which is the same launch
    /// sequence the block had before the split.
    pub(super) fn sparse_attention(
        &self,
        attention: &GpuBranch,
        slot: usize,
        mixed: Deviceptr,
        position: usize,
        batch: usize,
    ) -> Result<(), BackendError> {
        self.sparse_project(attention, mixed, batch)?;
        self.sparse_attend(attention, slot, position, batch)?;
        self.sparse_output(attention, slot, batch)
    }

    /// The row-independent half of a sparse block: the query, key, value and index projections
    /// over `batch` rows of `mixed`, the head deinterleave and the two head norms. It reads only
    /// `mixed` and the layer's own weights, so a batched step runs every row through one launch.
    pub(super) fn sparse_project(
        &self,
        attention: &GpuBranch,
        mixed: Deviceptr,
        batch: usize,
    ) -> Result<(), BackendError> {
        let GpuBranch::Sparse {
            q_proj,
            k_proj,
            v_proj,
            index_qk,
            q_norm,
            k_norm,
            ..
        } = attention
        else {
            return Err(BackendError::new(
                "qwen4_device",
                "a sparse-attention block was given a linear-attention layer",
            ));
        };
        let config = &self.backend.config;
        let kernels = &*self.backend.kernels;
        let kv_width = config.kv_proj_width();
        let attn_width = config.attn_out_width();
        self.gemv4(
            q_proj,
            k_proj,
            v_proj,
            index_qk,
            mixed,
            self.qg,
            self.kv_row,
            self.value_row,
            self.index_row,
            batch,
        )?;
        let deint_blocks = attn_width.div_ceil(256).max(1) as u32;
        // The whole chunk goes through each projection once, instead of the chunk
        // going through every projection once per token: 13 launches per chunk
        // rather than 13 per token on every sparse layer.
        self.launch(
            kernels.deinterleave,
            (deint_blocks, batch as u32, 1),
            (256, 1, 1),
            &[
                self.qg.0,
                self.query.0,
                self.gate.0,
                config.num_attention_heads as u64,
                config.head_dim as u64,
            ],
        )?;
        self.norm(
            self.query,
            *q_norm,
            self.query,
            attn_width,
            config.head_dim,
            true,
            true,
            batch,
        )?;
        self.norm(
            self.kv_row,
            *k_norm,
            self.kv_row,
            kv_width,
            config.head_dim,
            true,
            true,
            batch,
        )?;
        self.prof_mark("sparse_proj")?;
        Ok(())
    }

    /// The state-dependent half of a sparse block: the rotary of `query` and `kv_row` at
    /// `position`, the append to the caches in `slot`, the indexer's pooled blocks, selection and
    /// the gated attention over it. Every launch here reads or writes one sequence's caches at its
    /// own position, so a batched step runs it once per row.
    pub(super) fn sparse_attend(
        &self,
        attention: &GpuBranch,
        slot: usize,
        position: usize,
        batch: usize,
    ) -> Result<(), BackendError> {
        let GpuBranch::Sparse {
            index_q_norm,
            index_k_norm,
            ..
        } = attention
        else {
            return Err(BackendError::new(
                "qwen4_device",
                "a sparse-attention block was given a linear-attention layer",
            ));
        };
        let config = &self.backend.config;
        let kernels = &*self.backend.kernels;
        let keys = self.keys[slot].expect("a sparse layer has a key cache");
        let values = self.values[slot].expect("a sparse layer has a value cache");
        let index_keys = self.index_keys[slot].expect("a sparse layer has an index-key cache");
        let kv_width = config.kv_proj_width();
        let attn_width = config.attn_out_width();
        let idx_row_width =
            (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        let idx_q_width = config.indexer_n_heads * config.indexer_head_dim;
        let ratio = config.indexer_compress_ratio.max(1);
        let score_stride = self.score_stride;
        self.launch(
            kernels.rope,
            (config.num_attention_heads as u32, batch as u32, 1),
            (128, 1, 1),
            &[
                self.query.0,
                config.head_dim as u64,
                config.rotary_dim() as u64,
                position as u64,
                (config.rope_theta as f32).to_bits() as u64,
                attn_width as u64,
                batch as u64,
            ],
        )?;
        self.launch(
            kernels.rope,
            (config.num_key_value_heads as u32, batch as u32, 1),
            (128, 1, 1),
            &[
                self.kv_row.0,
                config.head_dim as u64,
                config.rotary_dim() as u64,
                position as u64,
                (config.rope_theta as f32).to_bits() as u64,
                kv_width as u64,
                batch as u64,
            ],
        )?;
        // The chunk's keys and values land in the cache as one contiguous run, so the append
        // narrows batch * kv_width f32 projections to BF16 in one pass rather than one copy per
        // token. The cache holds the model dtype, not the kernels' working f32.
        let kv_elems = batch * kv_width;
        let kv_blocks = kv_elems.div_ceil(256).max(1) as u32;
        self.launch(
            kernels.cache_bf16,
            (kv_blocks, 1, 1),
            (256, 1, 1),
            &[
                keys.offset(position * kv_width * 2).0,
                self.kv_row.0,
                kv_elems as u64,
            ],
        )?;
        self.launch(
            kernels.cache_bf16,
            (kv_blocks, 1, 1),
            (256, 1, 1),
            &[
                values.offset(position * kv_width * 2).0,
                self.value_row.0,
                kv_elems as u64,
            ],
        )?;
        // The indexer's query is the leading slice of the wider per-token index
        // row, so it is normed with an explicit source stride, then rotated in place.
        self.launch(
            kernels.rmsnorm_rows,
            (
                (idx_q_width / config.indexer_head_dim).max(1) as u32,
                batch as u32,
                1,
            ),
            (256, 1, 1),
            &[
                self.index_row.0,
                index_q_norm.0,
                self.index_query.0,
                idx_q_width as u64,
                config.indexer_head_dim as u64,
                (config.rms_norm_eps as f32).to_bits() as u64,
                1,
                1,
                idx_row_width as u64,
                idx_q_width as u64,
            ],
        )?;
        self.launch(
            kernels.rope,
            (config.indexer_n_heads as u32, batch as u32, 1),
            (128, 1, 1),
            &[
                self.index_query.0,
                config.indexer_head_dim as u64,
                config.rotary_dim().min(config.indexer_head_dim) as u64,
                position as u64,
                (config.rope_theta as f32).to_bits() as u64,
                idx_q_width as u64,
                batch as u64,
            ],
        )?;
        self.launch(
            kernels.cache_rows_bf16,
            (
                config.indexer_head_dim.div_ceil(256).max(1) as u32,
                batch as u32,
                1,
            ),
            (256, 1, 1),
            &[
                index_keys.offset(position * config.indexer_head_dim * 2).0,
                self.index_row.offset(idx_q_width * 4).0,
                config.indexer_head_dim as u64,
                config.indexer_head_dim as u64,
                idx_row_width as u64,
            ],
        )?;
        self.probe_finite(slot, "sparse_query", self.query, batch * attn_width)?;
        self.probe_finite(slot, "sparse_keys", self.kv_row, batch * kv_width)?;
        self.prof_mark("sparse_kv")?;
        // The score, selection and attention launches each index their own token, so nothing
        // serialises on t; they take the chunk ATTENTION_CHUNK rows at a time because a row's
        // block scores are a context's worth of floats (65,537 at the native window), and a
        // 512-token chunk of them would be 128 MiB of scratch for a stage that is not
        // bandwidth-bound on it.
        // Every block the chunk's last row can see is pooled once, for all the rows.
        let pooled_blocks = (position + batch) / ratio;
        if pooled_blocks > 0 {
            self.launch(
                kernels.qsa_pool,
                (pooled_blocks as u32, 1, 1),
                (128, 1, 1),
                &[
                    index_keys.0,
                    index_k_norm.0,
                    self.qsa_pooled_keys.0,
                    config.indexer_head_dim as u64,
                    config.indexer_compress_ratio as u64,
                    config.rotary_dim().min(config.indexer_head_dim) as u64,
                    (config.rope_theta as f32).to_bits() as u64,
                    (config.rms_norm_eps as f32).to_bits() as u64,
                ],
            )?;
        }
        let mut base = 0usize;
        while base < batch {
            let rows = ATTENTION_CHUNK.min(batch - base);
            let at = position + base;
            self.launch(
                kernels.qsa_scores,
                (
                    ((at + rows) / ratio).div_ceil(QSA_TILE).max(1) as u32,
                    rows as u32,
                    1,
                ),
                (256, 1, 1),
                &[
                    self.index_query.offset(base * idx_q_width * 4).0,
                    self.qsa_pooled_keys.0,
                    self.qsa_block_scores.0,
                    at as u64,
                    config.indexer_n_heads as u64,
                    config.indexer_head_dim as u64,
                    config.indexer_compress_ratio as u64,
                    score_stride as u64,
                ],
            )?;
            self.prof_mark("sparse_score")?;
            self.launch(
                kernels.qsa_pick,
                (rows as u32, 1, 1),
                (256, 1, 1),
                &[
                    self.qsa_block_scores.0,
                    self.selected.0,
                    self.selected_count.0,
                    at as u64,
                    config.indexer_compress_ratio as u64,
                    config.indexer_block_budget() as u64,
                    score_stride as u64,
                    MAX_SELECTED_TOKENS as u64,
                ],
            )?;
            self.probe_selected(slot, base, rows)?;
            self.prof_mark("sparse_pick")?;
            // The grouped kernel reads a token's selection once for four heads and computes each
            // as q4e_attn would; it takes any configuration whose groups and selection it holds.
            let (attn, heads) = kernels
                .attn_group
                .filter(|_| {
                    config.query_groups().is_multiple_of(ATTN_GROUP_HEADS)
                        && config.max_selected_tokens() <= ATTN_GROUP_SELECTED
                })
                .map_or((kernels.attn, 1), |kernel| (kernel, ATTN_GROUP_HEADS));
            self.launch(
                attn,
                ((config.num_attention_heads / heads) as u32, rows as u32, 1),
                (256, 1, 1),
                &[
                    self.query.offset(base * attn_width * 4).0,
                    self.gate.offset(base * attn_width * 4).0,
                    keys.0,
                    values.0,
                    self.selected.0,
                    self.selected_count.0,
                    self.attention_out.offset(base * attn_width * 4).0,
                    config.num_attention_heads as u64,
                    config.num_key_value_heads as u64,
                    config.head_dim as u64,
                    config.query_groups() as u64,
                    MAX_SELECTED_TOKENS as u64,
                ],
            )?;
            // Marked inside the loop, or the next piece's pick would be charged for this attention.
            self.prof_mark("sparse_attn")?;
            base += rows;
        }
        Ok(())
    }

    /// The row-independent tail of a sparse block: the output projection of `attention_out` into
    /// `self.branch`, over `batch` rows.
    pub(super) fn sparse_output(
        &self,
        attention: &GpuBranch,
        slot: usize,
        batch: usize,
    ) -> Result<(), BackendError> {
        let GpuBranch::Sparse { o_proj, .. } = attention else {
            return Err(BackendError::new(
                "qwen4_device",
                "a sparse-attention block was given a linear-attention layer",
            ));
        };
        let config = &self.backend.config;
        let attn_width = config.attn_out_width();
        self.probe_finite(
            slot,
            "attention_out",
            self.attention_out,
            batch * attn_width,
        )?;
        self.gemv(o_proj, self.attention_out, self.branch, batch)?;
        self.probe_finite(
            slot,
            "attention_branch",
            self.branch,
            batch * config.hidden_size,
        )?;
        Ok(())
    }

    /// The row-independent half of a linear (Gated DeltaNet) block: the four input projections
    /// over `batch` rows of `mixed` into the qkv, z, b and a buffers. A batched step runs every
    /// row through one launch; the convolution and the recurrence that follow are per sequence.
    pub(super) fn linear_project(
        &self,
        branch: &GpuBranch,
        mixed: Deviceptr,
        b_out: Deviceptr,
        a_out: Deviceptr,
        batch: usize,
    ) -> Result<(), BackendError> {
        let GpuBranch::Linear {
            in_qkv,
            in_z,
            in_b,
            in_a,
            ..
        } = branch
        else {
            return Err(BackendError::new(
                "qwen4_device",
                "a linear block was given a sparse-attention layer",
            ));
        };
        self.gemv4(
            in_qkv,
            in_z,
            in_b,
            in_a,
            mixed,
            self.linear_qkv,
            self.linear_z,
            b_out,
            a_out,
            batch,
        )?;
        self.prof_mark("gdn_proj")
    }

    /// The state-dependent half of a linear block: the causal convolution and the delta-rule
    /// recurrence over `batch` rows, reading and writing `self.gdn_conv[index]` and
    /// `self.gdn_recurrent[index]` and leaving the value rows in `self.linear_out`. A batched step
    /// runs it once per row with `batch` 1.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn linear_state(
        &self,
        index: usize,
        branch: &GpuBranch,
        conv_out: Deviceptr,
        b_out: Deviceptr,
        a_out: Deviceptr,
        batch: usize,
        chunk: bool,
        snapshot_rows: usize,
        conv_snapshot: u64,
        region_stride: u64,
    ) -> Result<(), BackendError> {
        let GpuBranch::Linear {
            conv,
            a_log,
            dt_bias,
            norm,
            ..
        } = branch
        else {
            return Err(BackendError::new(
                "qwen4_device",
                "a linear block was given a sparse-attention layer",
            ));
        };
        let config = &self.backend.config;
        let kernels = &*self.backend.kernels;
        let conv_state = self.gdn_conv[index].expect("a linear layer has a conv state");
        let recurrent_state =
            self.gdn_recurrent[index].expect("a linear layer has a recurrent state");
        // A verify leaves the state as it found it and its last row's in the spare.
        let recurrent_out = if snapshot_rows == 0 {
            recurrent_state.0
        } else {
            self.gdn_spare[index]
                .expect("a verified linear layer has a spare state")
                .0
        };
        let chunked = chunk
            && self.gdn_chunk(
                *conv,
                conv_state,
                recurrent_state,
                *a_log,
                *dt_bias,
                *norm,
                batch,
            )?;
        if !chunked {
            self.launch(
                kernels.gdn_conv,
                (config.linear_conv_width().div_ceil(256).max(1) as u32, 1, 1),
                (256, 1, 1),
                &[
                    self.linear_qkv.0,
                    conv_state.0,
                    conv.0,
                    conv_out.0,
                    config.linear_conv_width() as u64,
                    config.linear_conv_kernel_dim as u64,
                    batch as u64,
                    conv_snapshot,
                    snapshot_rows as u64,
                    region_stride,
                ],
            )?;
            self.probe_finite(
                index,
                "gdn_conv_out",
                conv_out,
                batch * config.linear_conv_width(),
            )?;
            self.prof_mark("gdn_conv")?;
            self.launch(
                kernels.gdn_recur,
                (config.linear_num_value_heads as u32, 1, 1),
                (128, 1, 1),
                &[
                    conv_out.0,
                    self.linear_z.0,
                    b_out.0,
                    a_out.0,
                    a_log.0,
                    dt_bias.0,
                    norm.0,
                    recurrent_state.0,
                    recurrent_out,
                    self.linear_out.0,
                    config.linear_key_width() as u64,
                    config.linear_num_value_heads as u64,
                    config.linear_key_head_dim as u64,
                    config.linear_value_head_dim as u64,
                    config.value_groups() as u64,
                    (config.rms_norm_eps as f32).to_bits() as u64,
                    u64::from(config.output_gate_is_sigmoid()),
                    batch as u64,
                    u64::from(kernels.gdn_recur.shared_bytes()),
                    // No channel decay: Gated DeltaNet decays a head by one factor.
                    0,
                ],
            )?;
            self.probe_finite(
                index,
                "gdn_recur_out",
                self.linear_out,
                batch * config.linear_value_width(),
            )?;
            self.prof_mark("gdn_recur")?;
        }
        Ok(())
    }

    /// The row-independent tail of a linear block: the output projection of the value rows into
    /// `self.branch` over `batch` rows.
    pub(super) fn linear_output(
        &self,
        index: usize,
        branch: &GpuBranch,
        batch: usize,
    ) -> Result<(), BackendError> {
        let GpuBranch::Linear { out_proj, .. } = branch else {
            return Err(BackendError::new(
                "qwen4_device",
                "a linear block was given a sparse-attention layer",
            ));
        };
        self.gemv(out_proj, self.linear_out, self.branch, batch)?;
        self.prof_mark("gdn_branch")?;
        self.probe_finite(
            index,
            "gdn_branch",
            self.branch,
            batch * self.backend.config.hidden_size,
        )
    }
    /// The MoE half of a layer over `batch` rows of `mixed`: route, run the routed and shared
    /// experts, and add their combination into the widened residual `hyper` through
    /// `injection`. The combined row is left in `self.branch`; `layer` names the layer in probes.
    ///
    /// `chunk` lets an NVFP4 layer's routed experts take the tensor-core prefill image.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mlp(
        &self,
        moe: &GpuMoe,
        mixed: Deviceptr,
        hyper: Deviceptr,
        injection: Deviceptr,
        layer: usize,
        chunk: bool,
        batch: usize,
    ) -> Result<(), BackendError> {
        let config = &self.backend.config;
        let kernels = &*self.backend.kernels;
        self.gemv(&moe.router, mixed, self.probabilities, batch)?;
        self.probe_finite(
            layer,
            "router_logits",
            self.probabilities,
            batch * config.num_experts,
        )?;
        self.launch(
            kernels.router,
            (batch as u32, 1, 1),
            (256, 1, 1),
            &[
                self.probabilities.0,
                self.probabilities.0,
                self.expert_indices.0,
                config.num_experts as u64,
                config.num_experts_per_tok as u64,
            ],
        )?;
        self.prof_mark("router")?;
        let tensor_cores = match moe.experts {
            GpuExperts::Nvfp4 { table, scales } if chunk => {
                self.routed_experts_mma(table, scales, mixed, batch)?
            }
            _ => false,
        };
        if !tensor_cores {
            self.routed_experts(&moe.experts, mixed, batch)?;
        }
        self.probe_finite(
            layer,
            "expert_partials",
            self.expert_accumulator,
            batch * config.num_experts_per_tok * config.hidden_size,
        )?;
        self.prof_mark("experts")?;
        self.shared_expert(moe, mixed, batch)?;
        self.prof_mark("shared")?;
        self.launch(
            kernels.moe_combine,
            (
                config.hidden_size.div_ceil(256).max(1) as u32,
                batch as u32,
                1,
            ),
            (256, 1, 1),
            &[
                self.branch.0,
                self.expert_accumulator.0,
                self.shared_out.0,
                self.expert_gate.0,
                hyper.0,
                injection.0,
                config.num_experts_per_tok as u64,
                config.hidden_size as u64,
                config.hc_count as u64,
            ],
        )?;
        self.probe_finite(
            layer,
            "moe_combine_out",
            self.branch,
            batch * config.hidden_size,
        )
    }

    /// The routed experts on the token kernels: the slots sorted into runs of one expert, then
    /// gate_up and down per run, into `self.expert_accumulator`'s [slot][hidden] partials.
    ///
    /// q4e_moe_permute sorts at most MAX_PERMUTE_SLOTS slots in shared memory and leaves a wider
    /// batch unsorted, so a chunk past that runs in slices of whole tokens, each offset into the
    /// chunk's indices, weights and partials.
    fn routed_experts(
        &self,
        experts: &GpuExperts,
        mixed: Deviceptr,
        batch: usize,
    ) -> Result<(), BackendError> {
        let config = &self.backend.config;
        let kernels = &*self.backend.kernels;
        let topk = config.num_experts_per_tok;
        let hidden = config.hidden_size;
        let intermediate = config.moe_intermediate_size;
        // The expert dot is latency-bound on its weight rows, so `expert_bench` reads best with
        // rows to spare in flight: two rows per warp and eight warps a block - one sixteen-row
        // panel - beat one row per warp at every sharing level, and beat wider blocks too. The
        // row counts mirror `Q4E_ROWS_PER_WARP` and, for the NVFP4 down pass's shorter rows,
        // `Q4E_DOWN_ROWS_PER_WARP` in the kernel.
        const WARPS: usize = 8;
        const ROWS: usize = 2;
        const DOWN_ROWS: usize = 4;
        let per_slice = (MAX_PERMUTE_SLOTS / topk.max(1)).max(1);
        let mut base = 0usize;
        while base < batch {
            let count = per_slice.min(batch - base);
            let indices = self.expert_indices.offset(base * topk * 4);
            let weights = self.probabilities.offset(base * config.num_experts * 4);
            let input = mixed.offset(base * hidden * 4);
            let partials = self.expert_accumulator.offset(base * topk * hidden * 4);
            self.launch(
                kernels.moe_permute,
                (1, 1, 1),
                (256, 1, 1),
                &[
                    indices.0,
                    self.expert_perm.0,
                    self.expert_run.0,
                    (count * topk) as u64,
                ],
            )?;
            let gate_up_grid = (
                intermediate.div_ceil(WARPS * ROWS) as u32,
                topk as u32,
                count as u32,
            );
            let down_grid = |rows: usize| {
                (
                    hidden.div_ceil(WARPS * rows) as u32,
                    topk as u32,
                    count as u32,
                )
            };
            let block = ((WARPS * 32) as u32, 1, 1);
            match *experts {
                GpuExperts::Nvfp4 { table, scales } => {
                    self.launch(
                        kernels.expert_gate_up,
                        gate_up_grid,
                        block,
                        &[
                            table.0,
                            scales.0,
                            indices.0,
                            self.expert_perm.0,
                            self.expert_run.0,
                            input.0,
                            self.expert_activated.0,
                            topk as u64,
                            hidden as u64,
                            intermediate as u64,
                        ],
                    )?;
                    self.launch(
                        kernels.expert_down,
                        down_grid(DOWN_ROWS),
                        block,
                        &[
                            table.0,
                            scales.0,
                            indices.0,
                            self.expert_perm.0,
                            self.expert_run.0,
                            weights.0,
                            self.expert_activated.0,
                            partials.0,
                            topk as u64,
                            hidden as u64,
                            intermediate as u64,
                            config.num_experts as u64,
                        ],
                    )?;
                }
                GpuExperts::Fp8 { table } => {
                    self.launch(
                        kernels.expert_gate_up_fp8,
                        gate_up_grid,
                        block,
                        &[
                            table.0,
                            indices.0,
                            self.expert_perm.0,
                            self.expert_run.0,
                            input.0,
                            self.expert_activated.0,
                            topk as u64,
                            hidden as u64,
                            intermediate as u64,
                        ],
                    )?;
                    self.launch(
                        kernels.expert_down_fp8,
                        down_grid(ROWS),
                        block,
                        &[
                            table.0,
                            indices.0,
                            self.expert_perm.0,
                            self.expert_run.0,
                            weights.0,
                            self.expert_activated.0,
                            partials.0,
                            topk as u64,
                            hidden as u64,
                            intermediate as u64,
                            config.num_experts as u64,
                        ],
                    )?;
                }
            }
            base += count;
        }
        Ok(())
    }

    /// The shared expert: SwiGLU of the gate and up projections, its sigmoid gate's logit in
    /// `self.expert_gate`, and the down projection in `self.shared_out`.
    fn shared_expert(
        &self,
        moe: &GpuMoe,
        mixed: Deviceptr,
        batch: usize,
    ) -> Result<(), BackendError> {
        let config = &self.backend.config;
        let kernels = &*self.backend.kernels;
        let intermediate = config.shared_expert_intermediate_size;
        if batch >= GEMM_MIN_BATCH {
            // The same argument as the hyper passes: one weight pass per chunk instead of one per
            // group of eight tokens.
            self.gemm(&moe.shared, mixed, self.shared_hidden, batch, 0, 1.0)?;
            self.gemm(&moe.shared_up, mixed, self.shared_up, batch, 0, 1.0)?;
            self.launch(
                kernels.swiglu,
                ((batch * intermediate).div_ceil(256).max(1) as u32, 1, 1),
                (256, 1, 1),
                &[
                    self.shared_hidden.0,
                    self.shared_up.0,
                    self.shared_up.0,
                    (batch * intermediate) as u64,
                ],
            )?;
            self.gemm(&moe.shared_gate, mixed, self.expert_gate, batch, 0, 1.0)?;
        } else {
            // Same register-accumulator bound as the hyper passes: one launch per group of
            // GEMV_MAX_BATCH tokens, offset into the chunk's staging rows.
            const WARPS: usize = 8;
            let mut base = 0usize;
            while base < batch {
                let count = GEMV_MAX_BATCH.min(batch - base);
                self.launch(
                    kernels.shared_gate_up,
                    ((intermediate + 1).div_ceil(WARPS) as u32, 1, 1),
                    ((WARPS * 32) as u32, 1, 1),
                    &[
                        moe.shared.pointer.0,
                        moe.shared_up.pointer.0,
                        moe.shared_gate.pointer.0,
                        mixed.offset(base * config.hidden_size * 4).0,
                        self.shared_up.offset(base * intermediate * 4).0,
                        self.expert_gate.offset(base * 4).0,
                        intermediate as u64,
                        config.hidden_size as u64,
                        count as u64,
                    ],
                )?;
                base += count;
            }
        }
        self.gemv(&moe.shared_down, self.shared_up, self.shared_out, batch)
    }
}
