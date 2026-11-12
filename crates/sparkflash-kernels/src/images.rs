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

//! The shared kernel images: every cubin the arms launch, behind one accessor each.
//!
//! Before the one-stack move, the Qwen3.8 and GLM-5.3 sources were compiled by
//! sparkflash-model-qwen4-exp's build script and every other arm reached them through that model
//! crate. The build script here compiles all ten images from kernels/ and exports the paths, so
//! a model crate carries no CUDA and no kernel build script and no model crate depends on another
//! for a kernel.

/// The CUDA architecture this build's kernels target (`SPARKFLASH_CUDA_ARCH`, default `sm_121a`).
///
/// The numerics registry keys every configuration by it: the same source compiled for another
/// architecture is a different arithmetic until a reference says otherwise.
pub const CUDA_ARCH: &str = env!("SPARKFLASH_BUILT_CUDA_ARCH");

/// One image the shared stack builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Image {
    /// The Qwen3.8 token, norm, dense and hyper kernels.
    Qwen4Exp,
    /// The Qwen3.8 attention, sparse-indexer and KV-cache token kernels.
    Qwen4Attn,
    /// The Qwen3.8 decode/token routed-expert (MoE) kernels.
    Qwen4Moe,
    /// The Qwen3.8 tensor-core bulk prefill.
    Qwen4Prefill,
    /// The Qwen3.8 Gated DeltaNet / Kimi Delta Attention scan.
    Qwen4Scan,
    /// The DFlash 2 block-drafter kernels.
    Dflash2,
    /// The GLM-5.3 block-scaled FP8 bulk prefill.
    GlmPrefill,
    /// The GLM-5.3 sparse MLA attention and its per-head projections.
    GlmAttention,
    /// The GLM-5.3 bulk routed-expert (MoE) kernels.
    GlmMoe,
    /// The GLM-5.3 decode/token kernels: hyper-connection mixers, router, indexer and absorbed MLA.
    GlmDecode,
    /// The routed-expert plan every arm shares.
    MoePlan,
    /// The Dory1.0-4B nGPT token kernels.
    Dory,
    /// The DeepSeek kernels loaded from the shared PTX set.
    Deepseek,
    /// The RoCEnante one-shot collective.
    Rocenante,
}

#[cfg(sparkflash_model_cubin)]
const QWEN4_EXP: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_EXP: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const QWEN4_ATTN: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_ATTN_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_ATTN: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const QWEN4_MOE: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_MOE_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_MOE: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const QWEN4_PREFILL: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_PREFILL_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_PREFILL: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const QWEN4_SCAN: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_SCAN_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const QWEN4_SCAN: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const DFLASH2: &[u8] = include_bytes!(env!("SPARKFLASH_QWEN4_DFLASH2_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const DFLASH2: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_PREFILL: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_PREFILL_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_PREFILL: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_ATTENTION: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_ATTENTION_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_ATTENTION: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_MOE: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_MOE_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_MOE: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const GLM_DECODE: &[u8] = include_bytes!(env!("SPARKFLASH_GLM_DECODE_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const GLM_DECODE: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const MOE_PLAN: &[u8] = include_bytes!(env!("SPARKFLASH_MOE_PLAN_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const MOE_PLAN: &[u8] = &[];

#[cfg(sparkflash_model_cubin)]
const DORY: &[u8] = include_bytes!(env!("SPARKFLASH_DORY_CUBIN"));
#[cfg(not(sparkflash_model_cubin))]
const DORY: &[u8] = &[];

/// The Qwen3.8 token image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_exp_cubin() -> &'static [u8] {
    QWEN4_EXP
}

/// The Qwen3.8 attention image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_attn_cubin() -> &'static [u8] {
    QWEN4_ATTN
}

/// The Qwen3.8 MoE image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_moe_cubin() -> &'static [u8] {
    QWEN4_MOE
}

/// The Qwen3.8 tensor-core prefill image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_prefill_cubin() -> &'static [u8] {
    QWEN4_PREFILL
}

/// The Qwen3.8 scan image, empty on a CPU-only build.
#[must_use]
pub fn qwen4_scan_cubin() -> &'static [u8] {
    QWEN4_SCAN
}

/// The DFlash 2 drafter image, empty on a CPU-only build.
#[must_use]
pub fn dflash2_cubin() -> &'static [u8] {
    DFLASH2
}

/// The GLM-5.3 bulk-prefill image, empty on a CPU-only build.
#[must_use]
pub fn glm_prefill_cubin() -> &'static [u8] {
    GLM_PREFILL
}

/// The GLM-5.3 sparse MLA attention image, empty on a CPU-only build.
#[must_use]
pub fn glm_attention_cubin() -> &'static [u8] {
    GLM_ATTENTION
}

/// The GLM-5.3 routed-expert image, empty on a CPU-only build.
#[must_use]
pub fn glm_moe_cubin() -> &'static [u8] {
    GLM_MOE
}

/// The GLM-5.3 decode/token image, empty on a CPU-only build.
#[must_use]
pub fn glm_decode_cubin() -> &'static [u8] {
    GLM_DECODE
}

/// The routed-expert plan image every arm shares, empty on a CPU-only build.
#[must_use]
pub fn moe_plan_cubin() -> &'static [u8] {
    MOE_PLAN
}

/// The Dory1.0-4B token image, empty on a CPU-only build.
#[must_use]
pub fn dory_cubin() -> &'static [u8] {
    DORY
}

impl Image {
    /// Every kernel symbol this image exports, by the name the driver loads.
    ///
    /// This is the registry the catalogue is checked against: a catalogue entry may
    /// name only a symbol that appears here for its image, so a renamed or misfiled
    /// kernel fails a CPU test instead of a launch.
    #[must_use]
    pub const fn exported_symbols(self) -> &'static [&'static str] {
        match self {
            Self::Qwen4Exp => &[
                "q4e_add",
                "q4e_argmax_slices",
                "q4e_copy_flat",
                "q4e_decode_f8e4m3",
                "q4e_deinterleave",
                "q4e_embed",
                "q4e_fold_streams",
                "q4e_gdn_conv",
                "q4e_gdn_recur",
                "q4e_gdn_recur_columns",
                "q4e_gemm_bf16",
                "q4e_gemm_reduce",
                "q4e_gemv4_bf16",
                "q4e_gemv_bf16",
                "q4e_gemv_tiled_bf16",
                "q4e_gemv_tiled_e4m3",
                "q4e_gemv_tiled_e4m3_tile",
                "q4e_hyper_down",
                "q4e_hyper_down_tiled",
                "q4e_hyper_inject",
                "q4e_hyper_mix",
                "q4e_hyper_up",
                "q4e_hyper_up_tiled",
                "q4e_inject",
                "q4e_kda_decay",
                "q4e_ple_conv",
                "q4e_ple_gate",
                "q4e_quantize_e4m3_rows",
                "q4e_rmsnorm",
                "q4e_rmsnorm_rows",
                "q4e_rope",
                "q4e_shared_gate_up",
                "q4e_sum_slots",
                "q4e_swiglu",
                "q4e_widen_streams",
                "q4e_zero",
            ],
            Self::Qwen4Attn => &[
                "q4e_attn",
                "q4e_attn_full",
                "q4e_attn_full_combine",
                "q4e_attn_group",
                "q4e_attn_group6",
                "q4e_cache_bf16",
                "q4e_cache_rows_bf16",
                "q4e_qsa_pick",
                "q4e_qsa_pool",
                "q4e_qsa_scores",
            ],
            Self::Qwen4Moe => &[
                "q4e_expert_down",
                "q4e_expert_down_fp8",
                "q4e_expert_gate_up",
                "q4e_expert_gate_up_fp8",
                "q4e_moe_combine",
                "q4e_router",
            ],
            Self::Qwen4Prefill => &[
                "q4p_dense_fp8",
                "q4p_dense_mma",
                "q4p_dense_rows",
                "q4p_epilogue",
                "q4p_gemm_rows",
                "q4p_hyper_up_rows",
                "q4p_moe_down",
                "q4p_moe_down_fp8",
                "q4p_moe_gate_up",
                "q4p_moe_gate_up_fp8",
                "q4p_moe_rows_down",
                "q4p_moe_rows_gate_up",
                "q4p_moe_swiglu_fp8",
                "q4p_split_bf16",
            ],
            Self::Qwen4Scan => &[
                "q4p_gdn_conv",
                "q4p_gdn_conv_state",
                "q4p_gdn_gate",
                "q4p_gdn_prep",
                "q4p_gdn_scan",
                "q4p_gdn_scan_128_256",
                "q4p_gdn_scan_64_64",
                "q4p_kda_prep",
                "q4p_kda_scan",
                "q4p_kda_scan_128_256",
                "q4p_kda_scan_64_64",
            ],
            Self::Dflash2 => &[
                "q4d_attn",
                "q4d_block_embed",
                "q4d_capture",
                "q4d_conv",
                "q4d_kv_ring",
                "q4d_select",
                "q4d_topk_final",
                "q4d_topk_partial",
            ],
            Self::GlmPrefill => &[
                "glm_prefill_bf16_mma",
                "glm_prefill_bf16_mma_bf16a",
                "glm_prefill_bf16_wide",
                "glm_prefill_bf16_wide_bf16a",
                "glm_prefill_fp8b128_mma",
                "glm_prefill_fp8b128_mma_bf16a",
                "glm_prefill_q8_mma",
                "glm_quantise_rows",
            ],
            Self::GlmAttention => &[
                "glm_prefill_heads_bf16_mma",
                "glm_prefill_heads_fp8b128_mma",
                "glm_prefill_mla_attend",
                "glm_prefill_mla_attend_bf16a",
            ],
            Self::GlmMoe => &[
                "glm_moe_fp8b128_mma",
                "glm_moe_fp8b128_mma_bf16a",
                "glm_moe_q8_mma",
                "glm_moe_swiglu",
                "glm_moe_swiglu_q8",
            ],
            Self::GlmDecode => &[
                "q4e_expert_gemv_e4m3_block",
                "q4e_gemv_tiled_bf16_parts",
                "q4e_gemv_tiled_e4m3_block",
                "q4e_glm_index_scores",
                "q4e_glm_mla_absorb",
                "q4e_glm_mla_attend",
                "q4e_glm_mla_cache",
                "q4e_glm_mla_cache_fp8",
                "q4e_glm_mla_mix",
                "q4e_glm_mla_project",
                "q4e_glm_mla_scores",
                "q4e_glm_mla_softmax",
                "q4e_glm_pick",
                "q4e_glm_pool",
                "q4e_layer_norm",
                "q4e_mhc_in",
                "q4e_mhc_post",
                "q4e_noaux_router",
                "q4e_prefetch_l2",
                "q4e_swiglu_limited",
            ],
            Self::MoePlan => &["moe_plan"],
            Self::Dory => &[
                "dory_attn_combine",
                "dory_attn_mma128",
                "dory_attn_mma16",
                "dory_attn_mma256",
                "dory_attn_mma64",
                "dory_attn_split",
                "dory_justnorm",
                "dory_logit_scale",
                "dory_loop_combine",
                "dory_mlp_epilogue",
                "dory_mma_wide",
                "dory_mma_wide4",
                "dory_rope_norm",
                "dory_rope_norm_lanes",
                "dory_slerp",
            ],
            Self::Rocenante => &["rocenante_oneshot_allreduce_bf16_tp4"],
            Self::Deepseek => &[
                "b12x_bf16a_mxfp8w_dense_sm121",
                "b12x_bf16a_mxfp8w_wo_a_sm121",
                "b12x_hostw_gemv_m1_bf16w_f32o_sm121",
                "b12x_hostw_gemv_m1_bf16w_sm121",
                "b12x_hostw_gemv_m1_f32w_f32o_sm121",
                "b12x_mxfp4_moe_sm121",
                "b12x_mxfp4_moe_stored_sm121",
                "b12x_mxfp8_dense_m1_sm121",
                "b12x_mxfp8_dense_par_k_sm121",
                "b12x_mxfp8_dense_sm121",
                "b12x_mxfp8_dense_split_k_combine_sm121",
                "b12x_mxfp8_dense_split_k_sm121",
                "b12x_mxfp8_dense_tile_sm121",
                "b12x_mxfp8_dense_warp_sm121",
                "b12x_mxfp8_gemv_m1_f32a_f32o_sm121",
                "b12x_mxfp8_gemv_m1_nf_f32a_sm121",
                "b12x_mxfp8_gemv_m1_nf_sm121",
                "b12x_mxfp8_gemv_m1_sm121",
                "b12x_mxfp8_wo_a_batched_perrow_sm121",
                "b12x_mxfp8_wo_a_batched_sm121",
                "b12x_mxfp8_wo_a_sm121",
                "b12x_mxfp8_wo_a_tile_sm121",
                "b12x_mxfp8_wo_a_warp_sm121",
                "cert_lm_head_batched_sm121",
                "cert_lm_head_sm121",
                "compressed_append_sm121",
                "compressed_selected_fill_sm121",
                "engram_embed_sm121",
                "engram_gate_write_sm121",
                "engram_narrow_bf16",
                "engram_widen_f32",
                "engram_wkv_sm121",
                "engram_write_bf16",
                "flash_mla_csa2_ring_sm121",
                "flash_mla_csa2_seq_batched_sm121",
                "flash_mla_csa2_seq_sm121",
                "flash_mla_csa2_sm121",
                "fp4_round_trip_bf16_to_f32",
                "hc_pre_collapse_bf16",
                "k_norm_f32_to_bf16",
                "kv_append_bf16",
                "kv_norm_rope_bf16",
                "lm_head_reduce_batched_sm121",
                "lm_head_reduce_sm121",
                "mega_mhc_fused_bf16_batched_sm121",
                "mega_mhc_fused_bf16_sm121",
                "mega_mhc_init_sm121",
                "mhc_attn_residual",
                "mhc_entry_bf16",
                "mhc_ffn_epilogue",
                "mhc_ffn_seams",
                "moe_expert_stage_sm121",
                "moe_routed_combine_sm121",
                "moe_router_sm121",
                "moe_router_topk_sm121",
                "mxfp8_quantize_activations_sm121",
                "opt_bulk_attn",
                "opt_bulk_engram_embed",
                "opt_bulk_engram_post",
                "opt_bulk_fp4_pack",
                "opt_bulk_mhc_partials",
                "opt_bulk_mhc_post_partials",
                "opt_bulk_mhc_tail",
                "opt_bulk_seam_derive",
                "opt_draft_attn",
                "opt_draft_embed_blocks",
                "opt_draft_fold_prob",
                "opt_draft_fold_seqs",
                "opt_draft_head_fp8",
                "opt_draft_head_fp8t",
                "opt_draft_logits",
                "opt_draft_logits_fp8",
                "opt_draft_markov",
                "opt_draft_markov_seqs",
                "opt_draft_markov_seqs_mma",
                "opt_draft_norm_quant",
                "opt_draft_pick_prob",
                "opt_draft_pick_seqs",
                "opt_draft_taps",
                "opt_engram_gate_write",
                "opt_engram_wkv",
                "opt_flash_mla_csa2_seq",
                "opt_gemv_bf16a_bf16o_t1",
                "opt_gemv_bf16a_bf16o_t2",
                "opt_gemv_bf16a_bf16o_t4",
                "opt_gemv_bf16a_bf16o_t8",
                "opt_gemv_f32a_bf16o_t1",
                "opt_gemv_f32a_bf16o_t2",
                "opt_gemv_f32a_bf16o_t4",
                "opt_gemv_f32a_bf16o_t8",
                "opt_gemv_f32a_f32o_t1",
                "opt_gemv_f32a_f32o_t2",
                "opt_gemv_f32a_f32o_t4",
                "opt_gemv_f32a_f32o_t8",
                "opt_gemv_fp8a_bf16o_t1",
                "opt_gemv_fp8a_bf16o_t2",
                "opt_gemv_fp8a_bf16o_t4",
                "opt_gemv_fp8a_bf16o_t8",
                "opt_gemv_fp8a_f32o_t1",
                "opt_gemv_fp8a_f32o_t8",
                "opt_gemv_mma_bf16a_bf16o_m1",
                "opt_gemv_mma_bf16a_bf16o_m2",
                "opt_gemv_mma_fp8a_bf16o_m1",
                "opt_gemv_rows_bf16a_bf16o_r8c1",
                "opt_gemv_rows_bf16a_bf16o_r8c2",
                "opt_gemv_rows_f32a_f32o_r4c1",
                "opt_gemv_rows_f32a_f32o_r4c2",
                "opt_gemv_rows_fp8a_bf16o_r4c1",
                "opt_gemv_rows_fp8a_bf16o_r8c1",
                "opt_gemv_rows_fp8a_bf16o_r8c2",
                "opt_k_norm_rows",
                "opt_kv_norm_rope",
                "opt_lm_head",
                "opt_mhc_post",
                "opt_mhc_seam",
                "opt_mma_bf16a_bf16o",
                "opt_mma_bf16a_f32o",
                "opt_mma_bf16a_fp8row_f32o",
                "opt_mma_f32a_bf16o",
                "opt_mma_f32a_f32o",
                "opt_mma_fp8_bf16o",
                "opt_mma_fp8_f32o",
                "opt_mma_fp8_rows_bf16o",
                "opt_mma_fp8_rowscale_f32o",
                "opt_mma_head_fp8t_f32o",
                "opt_mma_hostw_bf16a_bf16w",
                "opt_mma_hostw_bf16a_bf16w_f32o",
                "opt_mma_hostw_bf16a_f32w_f32o",
                "opt_mma_hostw_bf16w",
                "opt_mma_hostw_bf16w_f32o",
                "opt_mma_hostw_f32w_f32o",
                "opt_mma_index_scores",
                "opt_mma_moe_combine",
                "opt_mma_moe_down",
                "opt_mma_moe_gate_up",
                "opt_mma_router_dots",
                "opt_mma_router_topk",
                "opt_moe_combine_k3_sm121",
                "opt_moe_combine_sm121",
                "opt_moe_down_combine_k3_sm121",
                "opt_moe_down_combine_sm121",
                "opt_moe_down_dedup_sm121",
                "opt_moe_gate_up_dedup_k3_sm121",
                "opt_moe_gate_up_dedup_sm121",
                "opt_moe_gate_up_k3_sm121",
                "opt_moe_gate_up_sm121",
                "opt_moe_plan_sm121",
                "opt_pool_rows",
                "opt_ref_select",
                "opt_ref_select_fast",
                "opt_ref_select_pre",
                "opt_rmsnorm_bf16",
                "opt_rope_heads_bf16",
                "opt_router_e128k3_sm121",
                "opt_router_sm121",
                "opt_rows_argmax_fold",
                "opt_rows_argmax_pick",
                "opt_rows_attn",
                "opt_rows_attn_scores",
                "opt_rows_attn_softmax",
                "opt_rows_attn_values",
                "opt_rows_embed_own",
                "opt_rows_embed_stream",
                "opt_rows_embed_table",
                "opt_rows_engram_post",
                "opt_rows_engram_wkv",
                "opt_rows_engram_wkv_s",
                "opt_rows_group_append",
                "opt_rows_group_keys",
                "opt_rows_group_pool",
                "opt_rows_hc_pre_collapse",
                "opt_rows_hostw_bf16w",
                "opt_rows_hostw_bf16w_f32o",
                "opt_rows_hostw_bf16w_f32o_ring",
                "opt_rows_hostw_f32w_f32o",
                "opt_rows_hostw_f32w_f32o_ring",
                "opt_rows_inv_rope",
                "opt_rows_kv",
                "opt_rows_lm_head",
                "opt_rows_mhc_post",
                "opt_rows_mhc_seam",
                "opt_rows_mhc_tail",
                "opt_rows_prefetch_l2",
                "opt_rows_prologue_bf16",
                "opt_rows_rank_permute",
                "opt_rows_shared_swiglu",
                "opt_rows_store_u32",
                "opt_rows_topc",
                "opt_select_rows",
                "opt_select_rows_bf16",
                "opt_select_rows_tab",
                "opt_select_scores",
                "opt_select_scores_rows",
                "opt_select_scores_seg",
                "opt_select_scores_segs",
                "opt_sum_ranks_bf16",
                "opt_vision_attn",
                "opt_vision_gemm",
                "opt_vision_patchify",
                "opt_vision_patchify_16",
                "opt_vision_rmsnorm",
                "opt_vision_rope",
                "opt_vision_route",
                "opt_vision_span_rows",
                "opt_vision_swiglu",
                "opt_vision_unfold",
                "pool_group_f32",
                "ref_select_sm121",
                "rmsnorm_sm121",
                "rope_rotate_bf16",
                "rope_rotate_rows_bf16",
                "routed_gate_up_sm121",
                "scale_index_weights_bf16_to_f32",
                "shared_swiglu_f32",
                "swiglu_mxfp8_quant_sm121",
                "swiglu_mxfp8_quant_weighted_sm121",
                "w8a8_prologue_bf16_e4m3",
                "w8a8_prologue_e4m3",
                "window_quant_fp8_bf16",
            ],
        }
    }
    /// The shared token/norm/dense domain image (`Qwen4Exp`).
    pub const TOKEN: Self = Self::Qwen4Exp;
    /// The shared attention, sparse-indexer and KV-cache domain image (`Qwen4Attn`).
    pub const ATTENTION: Self = Self::Qwen4Attn;
    /// The shared routed-expert decode domain image (`Qwen4Moe`).
    pub const ROUTED_EXPERTS: Self = Self::Qwen4Moe;
    /// The shared tensor-core bulk-prefill domain image (`Qwen4Prefill`).
    pub const PREFILL: Self = Self::Qwen4Prefill;
    /// The shared linear-attention scan domain image (`Qwen4Scan`).
    pub const SCAN: Self = Self::Qwen4Scan;
    /// The block-scaled bulk-prefill domain image (`GlmPrefill`).
    pub const BLOCK_PREFILL: Self = Self::GlmPrefill;
    /// The sparse latent-attention domain image (`GlmAttention`).
    pub const SPARSE_MLA: Self = Self::GlmAttention;
    /// The block-scaled routed-expert domain image (`GlmMoe`).
    pub const BLOCK_ROUTED_EXPERTS: Self = Self::GlmMoe;
    /// The token/decode domain image (`GlmDecode`).
    pub const DECODE: Self = Self::GlmDecode;
    /// The block-drafter domain image (`Dflash2`).
    pub const DRAFTER: Self = Self::Dflash2;

    /// The image's bytes, empty when this build or this process carries none.
    #[must_use]
    pub fn bytes(self) -> &'static [u8] {
        match self {
            Self::Qwen4Exp => qwen4_exp_cubin(),
            Self::Qwen4Attn => qwen4_attn_cubin(),
            Self::Qwen4Moe => qwen4_moe_cubin(),
            Self::Qwen4Prefill => qwen4_prefill_cubin(),
            Self::Qwen4Scan => qwen4_scan_cubin(),
            Self::Dflash2 => dflash2_cubin(),
            Self::GlmPrefill => glm_prefill_cubin(),
            Self::GlmAttention => glm_attention_cubin(),
            Self::GlmMoe => glm_moe_cubin(),
            Self::GlmDecode => glm_decode_cubin(),
            Self::MoePlan => moe_plan_cubin(),
            Self::Dory => dory_cubin(),
            // DeepSeek loads its kernels through the shared driver from the PTX set, and the
            // RoCEnante path compiles its own image at run time.
            Self::Deepseek | Self::Rocenante => &[],
        }
    }

    /// Whether this build embeds the image.
    #[must_use]
    pub fn available(self) -> bool {
        !self.bytes().is_empty()
    }

    /// The image's stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Qwen4Exp => "qwen4_exp_sm121",
            Self::Qwen4Attn => "qwen4_attn_sm121",
            Self::Qwen4Moe => "qwen4_moe_sm121",
            Self::Qwen4Prefill => "qwen4_prefill_sm121",
            Self::Qwen4Scan => "qwen4_scan_sm121",
            Self::Dflash2 => "dflash2_sm121",
            Self::GlmPrefill => "glm_prefill_sm121",
            Self::GlmAttention => "glm_attention_sm121",
            Self::GlmMoe => "glm_moe_sm121",
            Self::GlmDecode => "glm_decode_sm121",
            Self::MoePlan => "moe_plan_sm121",
            Self::Dory => "dory_sm121",
            Self::Deepseek => "deepseek_ptx",
            Self::Rocenante => "rocenante_oneshot_sm121",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_image_has_a_stable_name_and_a_consistent_availability_bit() {
        for image in [
            Image::Qwen4Exp,
            Image::Qwen4Attn,
            Image::Qwen4Moe,
            Image::Qwen4Prefill,
            Image::Qwen4Scan,
            Image::Dflash2,
            Image::GlmPrefill,
            Image::GlmAttention,
            Image::GlmMoe,
            Image::GlmDecode,
            Image::Deepseek,
            Image::Dory,
            Image::Rocenante,
        ] {
            assert!(!image.name().is_empty());
            assert_eq!(image.available(), !image.bytes().is_empty());
        }
        assert_eq!(Image::TOKEN, Image::Qwen4Exp);
        assert_eq!(Image::ATTENTION, Image::Qwen4Attn);
        assert_eq!(Image::ROUTED_EXPERTS, Image::Qwen4Moe);
        assert_eq!(Image::PREFILL, Image::Qwen4Prefill);
        assert_eq!(Image::SCAN, Image::Qwen4Scan);
        assert_eq!(Image::BLOCK_PREFILL, Image::GlmPrefill);
        assert_eq!(Image::SPARSE_MLA, Image::GlmAttention);
        assert_eq!(Image::BLOCK_ROUTED_EXPERTS, Image::GlmMoe);
        assert_eq!(Image::DECODE, Image::GlmDecode);
        assert_eq!(Image::DRAFTER, Image::Dflash2);
    }
}
