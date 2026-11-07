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

//! The typed plans keep their identities, domains and launch contracts.
//!
//! Every plan is built from a PartRequirement, so the geometry that selects the kernel is the
//! execution plan's geometry and not a second, partial shape. Each arm's expected KernelId is
//! pinned by value; set membership is not enough to catch a plan that swapped images.

use sparkflash_model::{Part, Target};
use sparkflash_ops::attention::{AttentionKind, AttentionPlan, SplitKvDecodePlan};
use sparkflash_ops::collective::{CollectiveKind, CollectivePlan};
use sparkflash_ops::engine_plan::{EnginePlan, PartRequirement};
use sparkflash_ops::epilogue::{
    ActivationKind, ActivationPlan, GatherPlan, NormKind, NormPlan, QuantizePlan, RopePlan,
};
use sparkflash_ops::error::OpError;
use sparkflash_ops::gemm::{DenseGemmPlan, ExpertCombinePlan, GroupedMoeGemmPlan, MoeRoutingPlan};
use sparkflash_ops::images::Image;
use sparkflash_ops::parity::{LayerPlan, ParityReport, Tolerance};
use sparkflash_ops::plan::{
    DType, GroupedFp8Body, KernelId, KernelPhase, Layout, Matrix, OpKind, OpPlan, PartGeometry,
    Precision, StreamId,
};
use sparkflash_ops::resolver::PlanContext;
use sparkflash_ops::scan::{ScanKind, ScanPlan};

const CTX: PlanContext = PlanContext::SM121_CATALOGUE;

const DEEPSEEK: &str =
    include_str!("../../sparkflash-model-deepseek-v41/config/deepseek_v41_flash.released.json");
const GLM: &str = include_str!("../../sparkflash-model-glm5-next/config/glm53_flash.hf.json");
const QWEN: &str =
    include_str!("../../sparkflash-model-qwen4-exp/config/qwen38_flash_next_nvfp4.hf.json");

fn arm(text: &str, world: usize) -> EnginePlan {
    EnginePlan::from_json(text, Target::Sm121, world, 0).expect("a released plan")
}

fn requirement(plan: &EnginePlan, part: Part) -> PartRequirement {
    *plan
        .requirements()
        .find(|requirement| requirement.part == part)
        .unwrap_or_else(|| panic!("the plan has no {part} requirement"))
}

fn synthetic(
    part: Part,
    hidden: usize,
    intermediate: usize,
    rows: usize,
    precision: Precision,
) -> PartRequirement {
    PartRequirement {
        part,
        geometry: PartGeometry {
            hidden,
            intermediate,
            rows,
            positions: 262_144,
            world: 1,
            ..PartGeometry::new()
        },
        precision,
    }
}

fn expected(target: Target, image: Image, symbol: &'static str) -> KernelId {
    KernelId::new(target, image, symbol)
}

fn dense(
    requirement: &PartRequirement,
    rows: usize,
    columns: usize,
    split_k: usize,
) -> DenseGemmPlan {
    let layout = match requirement.precision {
        Precision::MxFp8 => Layout::BlockScaled { block: 32 },
        Precision::NvFp4 => Layout::BlockScaled { block: 16 },
        Precision::Fp8E4M3 => Layout::BlockScaled { block: 128 },
        Precision::F32 | Precision::Bf16 => Layout::RowMajor,
    };
    DenseGemmPlan::new(
        CTX,
        requirement,
        Matrix::new(rows, columns),
        requirement.geometry.hidden,
        layout,
        StreamId(0),
        split_k,
    )
    .expect("a valid dense GEMM")
}

#[test]
fn a_dense_gemm_names_one_kernel_and_allocates_nothing_at_launch() {
    let requirement = synthetic(Part::DenseGemm, 5120, 4096, 64, Precision::MxFp8);
    let plan = DenseGemmPlan::new(
        CTX,
        &requirement,
        Matrix::new(64, 4096),
        5120,
        Layout::BlockScaled { block: 32 },
        StreamId(3),
        1,
    )
    .expect("valid");
    assert_eq!(plan.kind(), OpKind::DenseGemm);
    assert_eq!(plan.workspace_floats(), 0);
    assert_eq!(plan.launch().stream, StreamId(3));
    assert_eq!(
        plan.kernel(),
        expected(Target::Sm121, Image::Deepseek, "b12x_mxfp8_dense_sm121")
    );
    assert_eq!(plan.kernel().target, Target::Sm121);
}

#[test]
fn a_split_k_gemm_reserves_its_partials_and_names_the_split_kernel() {
    let requirement = synthetic(Part::DenseGemm, 5120, 128, 64, Precision::MxFp8);
    let plan = DenseGemmPlan::new(
        CTX,
        &requirement,
        Matrix::new(64, 128),
        5120,
        Layout::BlockScaled { block: 32 },
        StreamId(1),
        5,
    )
    .expect("valid");
    assert_eq!(plan.kind(), OpKind::SplitKGemm);
    assert_eq!(plan.workspace_floats(), 5 * 64 * 128);
    assert_eq!(
        plan.kernel(),
        expected(
            Target::Sm121,
            Image::Deepseek,
            "b12x_mxfp8_dense_split_k_sm121"
        )
    );
}

#[test]
fn an_uneven_split_is_refused_before_a_launch() {
    let requirement = synthetic(Part::DenseGemm, 2559, 128, 64, Precision::MxFp8);
    let error = DenseGemmPlan::new(
        CTX,
        &requirement,
        Matrix::new(64, 128),
        2559,
        Layout::BlockScaled { block: 32 },
        StreamId(0),
        4,
    )
    .expect_err("2559 does not split into four");
    assert!(error.to_string().contains("does not split"));
}

#[test]
fn the_generic_mxfp8_body_resolves_at_2560_and_5120() {
    for hidden in [2560, 5120] {
        let requirement = synthetic(Part::DenseGemm, hidden, 2560, 8, Precision::MxFp8);
        let plan = dense(&requirement, 8, 2560, 1);
        assert_eq!(
            plan.kernel(),
            expected(Target::Sm121, Image::Deepseek, "b12x_mxfp8_dense_sm121"),
            "hidden {hidden}"
        );
        assert_eq!(plan.geometry.hidden, hidden);
    }
}

#[test]
fn a_dense_nvfp4_gemm_is_refused_because_no_dense_fp4_body_exists() {
    let requirement = synthetic(Part::DenseGemm, 2560, 1024, 8, Precision::NvFp4);
    let error = DenseGemmPlan::new(
        CTX,
        &requirement,
        Matrix::new(8, 1024),
        2560,
        Layout::BlockScaled { block: 16 },
        StreamId(0),
        1,
    )
    .expect_err("no dense NVFP4 body");
    assert!(error.to_string().contains("nvfp4"), "{error}");
}

#[test]
fn a_grouped_moe_gemm_validates_its_expert_table() {
    let mut requirement = synthetic(Part::GroupedMoeGemm, 2560, 1024, 8, Precision::NvFp4);
    requirement.geometry.experts = 512;
    requirement.geometry.top_k = 10;
    // The host takes its grouped body only from 64 tokens, so the plan's threshold is in
    // tokens while its rows are the routed slots those tokens select.
    requirement.geometry.grouped_min_tokens = 64;
    let plan = GroupedMoeGemmPlan::new(CTX, &requirement, 512, StreamId(0)).expect("valid");
    assert_eq!(plan.kind(), OpKind::GroupedMoeGemm);
    assert_eq!(plan.tokens, 512);
    assert_eq!(plan.rows, 512 * 10);
    assert!(plan.workspace_floats() > 0);
    assert_eq!(
        plan.kernel(),
        expected(Target::Sm121, Image::Qwen4Prefill, "q4p_moe_gate_up")
    );
    let decode =
        GroupedMoeGemmPlan::new(CTX, &requirement, 6, StreamId(0)).expect("the token body");
    assert_eq!(decode.tokens, 6);
    assert_eq!(decode.rows, 60);
    assert_eq!(
        decode.kernel(),
        expected(Target::Sm121, Image::Qwen4Moe, "q4e_expert_gate_up")
    );

    let mut inverted = requirement;
    inverted.geometry.experts = 2;
    inverted.geometry.top_k = 3;
    let error = GroupedMoeGemmPlan::new(CTX, &inverted, 1, StreamId(0))
        .expect_err("top-k past the expert count must be refused");
    assert!(matches!(error, OpError::Shape { .. }), "{error}");
}

#[test]
fn the_qwen_fp8_grouped_gemm_uses_its_token_threshold() {
    // The Qwen FP8 host takes the grouped 128-tile only from GEMM_MIN_BATCH (64 tokens); below
    // it the token FP8 experts run. The threshold is in tokens, and the plan derives the
    // routed rows as tokens * top_k, so six tokens are 60 slots and sixty-four are 640.
    let mut requirement = synthetic(Part::GroupedMoeGemm, 2560, 128, 0, Precision::Fp8E4M3);
    requirement.geometry.experts = 512;
    requirement.geometry.top_k = 10;
    requirement.geometry.grouped_min_tokens = 64;
    requirement.geometry.grouped_fp8_body = Some(GroupedFp8Body::Tiled);
    let short =
        GroupedMoeGemmPlan::new(CTX, &requirement, 6, StreamId(0)).expect("a 6-token FP8 batch");
    assert_eq!(short.rows, 60, "six tokens select ten experts each");
    assert_eq!(
        short.kernel(),
        expected(Target::Sm121, Image::Qwen4Moe, "q4e_expert_gate_up_fp8"),
        "six tokens is below Qwen's grouped threshold"
    );
    let chunk =
        GroupedMoeGemmPlan::new(CTX, &requirement, 64, StreamId(0)).expect("a 64-token chunk");
    assert_eq!(chunk.rows, 640, "sixty-four tokens select ten experts each");
    assert_eq!(
        chunk.kernel(),
        expected(Target::Sm121, Image::Qwen4Prefill, "q4p_moe_gate_up_fp8"),
        "64 tokens takes Qwen's grouped tile"
    );

    // An FP8 expert is whole 128-column scale tiles; a grouped plan cannot name a body for a
    // width that would split a tile.
    let mut narrow = requirement;
    narrow.geometry.intermediate = 144;
    let error = GroupedMoeGemmPlan::new(CTX, &narrow, 64, StreamId(0))
        .expect_err("144 columns is not whole 128-wide scale tiles");
    assert!(
        error.to_string().contains("128-column scale tiles"),
        "{error}"
    );
}

#[test]
fn the_glm_fp8_grouped_gemm_uses_its_eight_token_limit_and_q8_body() {
    // GLM's host keeps the token GEMV while a step has at most MAX_ROWS (8) token rows and
    // groups from 9 tokens. Its released dynamic-activation profile launches q8, so the plan's
    // symbol priority must name q8 before the BF16-activation block MMA.
    let mut requirement = synthetic(Part::GroupedMoeGemm, 4096, 2048, 0, Precision::Fp8E4M3);
    requirement.geometry.experts = 288;
    requirement.geometry.top_k = 8;
    requirement.geometry.grouped_min_tokens = 9;
    requirement.geometry.grouped_fp8_body = Some(GroupedFp8Body::QuantizedActivation);
    let token =
        GroupedMoeGemmPlan::new(CTX, &requirement, 8, StreamId(0)).expect("an 8-token step");
    assert_eq!(token.rows, 64, "eight tokens select eight experts each");
    assert_eq!(
        token.kernel(),
        expected(
            Target::Sm121,
            Image::GlmDecode,
            "q4e_expert_gemv_e4m3_block"
        ),
        "eight tokens is GLM's token path"
    );
    for tokens in [32, 64] {
        let chunk = GroupedMoeGemmPlan::new(CTX, &requirement, tokens, StreamId(0))
            .expect("a GLM FP8 chunk");
        assert_eq!(chunk.rows, tokens * 8);
        assert_eq!(
            chunk.kernel(),
            expected(Target::Sm121, Image::GlmMoe, "glm_moe_q8_mma"),
            "{tokens} tokens takes GLM's q8 body, not the BF16-activation block MMA"
        );
    }
}

#[test]
fn a_grouped_nvfp4_plan_without_a_token_threshold_falls_back_on_tokens() {
    // A zero grouped threshold must choose the pass from the batch tokens, not from the
    // derived routed rows: one token with top-k ten has ten rows but is still a decode step,
    // and its body is the token expert, not the grouped prefill tile.
    let mut requirement = synthetic(Part::GroupedMoeGemm, 2560, 1024, 1, Precision::NvFp4);
    requirement.geometry.experts = 512;
    requirement.geometry.top_k = 10;
    requirement.geometry.grouped_min_tokens = 0;
    let one =
        GroupedMoeGemmPlan::new(CTX, &requirement, 1, StreamId(0)).expect("a one-token NVFP4 step");
    assert_eq!(one.rows, 10, "one token selects ten experts");
    assert_eq!(
        one.kernel(),
        expected(Target::Sm121, Image::Qwen4Moe, "q4e_expert_gate_up"),
        "one token is a decode step even with ten routed rows"
    );
    let two = GroupedMoeGemmPlan::new(CTX, &requirement, 2, StreamId(0))
        .expect("a two-token NVFP4 batch");
    assert_eq!(two.rows, 20);
    assert_eq!(
        two.kernel(),
        expected(Target::Sm121, Image::Qwen4Prefill, "q4p_moe_gate_up"),
        "two tokens is a prefill batch"
    );
}

#[test]
fn the_glm_fp8_grouped_gemm_names_the_profiles_activation_body() {
    // GLM has three prefill activation forms and three different bodies. Naming the profile
    // form keeps the q8 body from being selected for a bf16 or split profile.
    let mut requirement = synthetic(Part::GroupedMoeGemm, 4096, 2048, 1, Precision::Fp8E4M3);
    requirement.geometry.experts = 288;
    requirement.geometry.top_k = 8;
    for (body, symbol) in [
        (GroupedFp8Body::QuantizedActivation, "glm_moe_q8_mma"),
        (GroupedFp8Body::Bf16Activation, "glm_moe_fp8b128_mma_bf16a"),
        (GroupedFp8Body::SplitActivation, "glm_moe_fp8b128_mma"),
    ] {
        requirement.geometry.grouped_fp8_body = Some(body);
        let chunk = GroupedMoeGemmPlan::new(CTX, &requirement, 32, StreamId(0))
            .unwrap_or_else(|error| panic!("{body:?} GLM chunk: {error}"));
        assert_eq!(
            chunk.kernel(),
            expected(Target::Sm121, Image::GlmMoe, symbol),
            "{body:?}"
        );
    }
    // A token step still falls through every prefill-only body to the decode GEMV.
    requirement.geometry.grouped_fp8_body = Some(GroupedFp8Body::Bf16Activation);
    let token =
        GroupedMoeGemmPlan::new(CTX, &requirement, 1, StreamId(0)).expect("a one-token GLM step");
    assert_eq!(
        token.kernel(),
        expected(
            Target::Sm121,
            Image::GlmDecode,
            "q4e_expert_gemv_e4m3_block"
        )
    );
}

#[test]
fn an_fp8_grouped_moe_gemm_must_name_its_body() {
    // There is no default body: a GLM or NVFP4 caller that leaves the field unset must not
    // inherit the Qwen grouped tile.
    let mut requirement = synthetic(Part::GroupedMoeGemm, 4096, 2048, 1, Precision::Fp8E4M3);
    requirement.geometry.experts = 288;
    requirement.geometry.top_k = 8;
    requirement.geometry.grouped_fp8_body = None;
    let error = GroupedMoeGemmPlan::new(CTX, &requirement, 32, StreamId(0))
        .expect_err("an FP8 grouped plan without a body");
    assert!(
        error.to_string().contains("names no grouped body"),
        "{error}"
    );
}

#[test]
fn a_bf16_grouped_moe_gemm_has_no_body_to_name() {
    // The released experts are E2M1 or E4M3; no catalogued routed-expert body reads BF16
    // weights, so a BF16 document must be refused rather than launched on a packed body.
    let mut requirement = synthetic(Part::GroupedMoeGemm, 2560, 128, 8, Precision::Bf16);
    requirement.geometry.experts = 512;
    requirement.geometry.top_k = 10;
    let error = GroupedMoeGemmPlan::new(CTX, &requirement, 8, StreamId(0))
        .expect_err("no BF16 routed-expert body");
    assert!(error.to_string().contains("bf16"), "{error}");
}

#[test]
fn grouped_attention_selects_the_kernel_for_its_head_group() {
    let requirement = requirement(&arm(QWEN, 1), Part::GqaAttention);
    let context = requirement.geometry.positions;
    let head_dim = requirement.geometry.head_dim;
    let six = AttentionPlan::new(
        CTX,
        &requirement,
        AttentionKind::Gqa {
            query_heads: 24,
            kv_heads: 4,
        },
        8,
        context,
        head_dim,
        StreamId(0),
    )
    .expect("valid");
    assert_eq!(six.group_size(), 6);
    assert_eq!(
        six.kernel(),
        expected(Target::Sm121, Image::Qwen4Attn, "q4e_attn_group6")
    );
    assert_eq!(six.launch().shared_bytes, 63_600);

    for (query_heads, kv_heads) in [(24, 6), (32, 8)] {
        let other = AttentionPlan::new(
            CTX,
            &requirement,
            AttentionKind::Gqa {
                query_heads,
                kv_heads,
            },
            8,
            context,
            head_dim,
            StreamId(0),
        )
        .expect("valid");
        assert_eq!(other.group_size(), 4);
        assert_eq!(
            other.kernel(),
            expected(Target::Sm121, Image::Qwen4Attn, "q4e_attn_group")
        );
        assert_eq!(other.launch().shared_bytes, 45_136);
    }

    let twelve = AttentionPlan::new(
        CTX,
        &requirement,
        AttentionKind::Gqa {
            query_heads: 24,
            kv_heads: 2,
        },
        8,
        context,
        head_dim,
        StreamId(0),
    )
    .expect("valid");
    assert_eq!(twelve.group_size(), 12);
    assert_eq!(
        twelve.kernel(),
        expected(Target::Sm121, Image::Qwen4Attn, "q4e_attn")
    );
    assert_eq!(twelve.launch().shared_bytes, 0);
}

#[test]
fn a_grouped_moe_gemm_refuses_mxfp8_until_a_grouped_kernel_exists() {
    let requirement = synthetic(Part::GroupedMoeGemm, 2560, 1024, 512, Precision::MxFp8);
    let mut requirement = requirement;
    requirement.geometry.experts = 512;
    requirement.geometry.top_k = 10;
    let error = GroupedMoeGemmPlan::new(CTX, &requirement, 512, StreamId(0))
        .expect_err("no grouped MXFP8 expert kernel");
    assert!(error.to_string().contains("mxfp8"));
}

#[test]
fn a_deepseek_moe_gemm_at_tp1_is_refused_by_its_intermediate_width() {
    let mut requirement = synthetic(Part::GroupedMoeGemm, 5120, 2304, 8, Precision::NvFp4);
    requirement.geometry.experts = 384;
    requirement.geometry.top_k = 6;
    let error = GroupedMoeGemmPlan::new(CTX, &requirement, 8, StreamId(0))
        .expect_err("the b12x kernel is compiled at 576");
    assert!(error.to_string().contains("does not equal 576"), "{error}");
}

#[test]
fn attention_resolves_the_arm_the_requirement_describes() {
    let deepseek = requirement(&arm(DEEPSEEK, 4), Part::MlaAttention);
    let mla = AttentionPlan::new(
        CTX,
        &deepseek,
        AttentionKind::Mla,
        8,
        131_072,
        deepseek.geometry.head_dim,
        StreamId(0),
    )
    .expect("DeepSeek MLA");
    assert_eq!(
        mla.kernel(),
        expected(Target::Sm121, Image::Deepseek, "flash_mla_csa2_sm121")
    );

    let glm = requirement(&arm(GLM, 4), Part::MlaAttention);
    let mla = AttentionPlan::new(
        CTX,
        &glm,
        AttentionKind::Mla,
        8,
        131_072,
        glm.geometry.head_dim,
        StreamId(0),
    )
    .expect("GLM MLA");
    assert_eq!(
        mla.kernel(),
        expected(Target::Sm121, Image::GlmAttention, "glm_prefill_mla_attend")
    );

    let glm_sparse = requirement(&arm(GLM, 4), Part::SparseIndexer);
    let sparse = AttentionPlan::new(
        CTX,
        &glm_sparse,
        AttentionKind::Sparse {
            selected: 2048,
            block: 4,
        },
        8,
        1 << 20,
        glm_sparse.geometry.head_dim,
        StreamId(0),
    )
    .expect("GLM sparse attention");
    assert_eq!(
        sparse.kernel(),
        expected(Target::Sm121, Image::GlmDecode, "q4e_glm_pick")
    );

    let qwen_sparse = requirement(&arm(QWEN, 1), Part::SparseIndexer);
    let sparse = AttentionPlan::new(
        CTX,
        &qwen_sparse,
        AttentionKind::Sparse {
            selected: 2048,
            block: 4,
        },
        8,
        1 << 20,
        qwen_sparse.geometry.head_dim,
        StreamId(0),
    )
    .expect("Qwen sparse attention");
    assert_eq!(
        sparse.kernel(),
        expected(Target::Sm121, Image::Qwen4Attn, "q4e_qsa_pick")
    );

    let csa2 = requirement(&arm(DEEPSEEK, 4), Part::Csa2Compression);
    let csa2 = AttentionPlan::new(
        CTX,
        &csa2,
        AttentionKind::Csa2 {
            ratio: csa2.geometry.ratio,
        },
        8,
        131_072,
        128,
        StreamId(0),
    )
    .expect("CSA2");
    assert_eq!(
        csa2.kernel(),
        expected(Target::Sm121, Image::Deepseek, "flash_mla_csa2_ring_sm121")
    );

    let decode = SplitKvDecodePlan::new(CTX, mla_or_deepseek(), 4).expect("split-KV decode");
    assert_eq!(
        decode.kernel(),
        expected(Target::Sm121, Image::Deepseek, "opt_flash_mla_csa2_seq")
    );
    assert_eq!(decode.launch().grid.2, 4);
}

fn mla_or_deepseek() -> AttentionPlan {
    let requirement = requirement(&arm(DEEPSEEK, 4), Part::MlaAttention);
    AttentionPlan::new(
        CTX,
        &requirement,
        AttentionKind::Mla,
        8,
        131_072,
        requirement.geometry.head_dim,
        StreamId(0),
    )
    .expect("DeepSeek MLA")
}

#[test]
fn norm_rope_activation_and_quantize_resolve_their_own_parts() {
    let deepseek = arm(DEEPSEEK, 4);
    let rms = requirement(&deepseek, Part::RmsNorm);
    let norm = NormPlan::new(
        CTX,
        &rms,
        NormKind::RmsNorm,
        8,
        rms.geometry.hidden,
        DType::Bf16,
        StreamId(0),
    )
    .expect("RMSNorm");
    assert_eq!(
        norm.kernel(),
        expected(Target::Sm121, Image::Deepseek, "rmsnorm_sm121")
    );

    let hyper = requirement(&deepseek, Part::HyperConnection);
    let norm = NormPlan::new(
        CTX,
        &hyper,
        NormKind::HyperConnection,
        1,
        hyper.geometry.hidden,
        DType::Bf16,
        StreamId(0),
    )
    .expect("DeepSeek hyper-connection");
    assert_eq!(
        norm.kernel(),
        expected(Target::Sm121, Image::Deepseek, "mega_mhc_fused_bf16_sm121")
    );

    let qwen = arm(QWEN, 1);
    let hyper = requirement(&qwen, Part::HyperConnection);
    let norm = NormPlan::new(
        CTX,
        &hyper,
        NormKind::HyperConnection,
        1,
        hyper.geometry.hidden,
        DType::Bf16,
        StreamId(0),
    )
    .expect("Qwen hyper-connection");
    assert_eq!(
        norm.kernel(),
        expected(Target::Sm121, Image::Qwen4Exp, "q4e_hyper_mix")
    );

    let glm = arm(GLM, 4);
    let manifold = requirement(&glm, Part::ManifoldHyperConnection);
    let norm = NormPlan::new(
        CTX,
        &manifold,
        NormKind::ManifoldHyperConnection,
        1,
        manifold.geometry.hidden,
        DType::Bf16,
        StreamId(0),
    )
    .expect("GLM manifold hyper-connection");
    assert_eq!(
        norm.kernel(),
        expected(Target::Sm121, Image::GlmDecode, "q4e_mhc_in")
    );

    let layer = synthetic(Part::LayerNorm, 4096, 0, 1, Precision::Bf16);
    let norm = NormPlan::new(
        CTX,
        &layer,
        NormKind::LayerNorm,
        1,
        4096,
        DType::Bf16,
        StreamId(0),
    )
    .expect("GLM LayerNorm");
    assert_eq!(
        norm.kernel(),
        expected(Target::Sm121, Image::GlmDecode, "q4e_layer_norm")
    );

    let qwen_rope = synthetic(Part::Rope, 2560, 0, 8, Precision::Bf16);
    let rope = RopePlan::new(CTX, &qwen_rope, 8, 24, 64, 2560, 0, StreamId(0)).expect("Qwen RoPE");
    assert_eq!(
        rope.kernel(),
        expected(Target::Sm121, Image::Qwen4Exp, "q4e_rope")
    );

    let deepseek_rope = synthetic(Part::Rope, 5120, 0, 1, Precision::Bf16);
    let rope =
        RopePlan::new(CTX, &deepseek_rope, 1, 64, 64, 5120, 0, StreamId(0)).expect("DeepSeek RoPE");
    assert_eq!(
        rope.kernel(),
        expected(Target::Sm121, Image::Deepseek, "opt_rope_heads_bf16")
    );

    let deepseek_mlp = synthetic(Part::Activation, 5120, 2304, 8, Precision::MxFp8);
    let activation = ActivationPlan::new(
        CTX,
        &deepseek_mlp,
        ActivationKind::SwiGlu,
        8,
        2304,
        StreamId(0),
    )
    .expect("DeepSeek SwiGLU");
    assert_eq!(
        activation.kernel(),
        expected(Target::Sm121, Image::Deepseek, "swiglu_mxfp8_quant_sm121")
    );

    let qwen_mlp = synthetic(Part::Activation, 2560, 640, 8, Precision::Bf16);
    let activation =
        ActivationPlan::new(CTX, &qwen_mlp, ActivationKind::SwiGlu, 8, 640, StreamId(0))
            .expect("Qwen SwiGLU");
    assert_eq!(
        activation.kernel(),
        expected(Target::Sm121, Image::Qwen4Exp, "q4e_swiglu")
    );

    // The MoE pipeline's fused gated activation is the FP8 body, not a BF16 epilogue: a Qwen
    // FP8 chunk's gate/up output is E4M3 at the 128-wide tile.
    let qwen_fp8_mlp = synthetic(Part::Activation, 2560, 128, 64, Precision::Fp8E4M3);
    let activation = ActivationPlan::new(
        CTX,
        &qwen_fp8_mlp,
        ActivationKind::SwiGlu,
        64,
        128,
        StreamId(0),
    )
    .expect("Qwen FP8 MoE SwiGLU");
    assert_eq!(
        activation.kernel(),
        expected(Target::Sm121, Image::Qwen4Prefill, "q4p_moe_swiglu_fp8")
    );

    let glm_mlp = synthetic(Part::Activation, 4096, 3072, 1, Precision::Bf16);
    let activation =
        ActivationPlan::new(CTX, &glm_mlp, ActivationKind::SwiGlu, 1, 3072, StreamId(0))
            .expect("GLM SwiGLU");
    assert_eq!(
        activation.kernel(),
        expected(Target::Sm121, Image::GlmDecode, "q4e_swiglu_limited")
    );

    let quantize = synthetic(Part::Quantize, 0, 0, 1, Precision::Bf16);
    let plan = QuantizePlan::new(
        CTX,
        &quantize,
        Matrix::new(1, 2560),
        DType::F32,
        DType::NvFp4,
        Layout::BlockScaled { block: 16 },
        StreamId(0),
    )
    .expect("NVFP4 quantizer");
    assert_eq!(
        plan.kernel(),
        expected(Target::Sm121, Image::Deepseek, "opt_bulk_fp4_pack")
    );

    let plan = QuantizePlan::new(
        CTX,
        &quantize,
        Matrix::new(1, 2560),
        DType::Bf16,
        DType::Fp8E4M3,
        Layout::BlockScaled { block: 128 },
        StreamId(0),
    )
    .expect("bf16 to FP8 quantizer");
    assert_eq!(
        plan.kernel(),
        expected(Target::Sm121, Image::Qwen4Exp, "q4e_quantize_e4m3_rows")
    );

    let plan = QuantizePlan::new(
        CTX,
        &quantize,
        Matrix::new(2, 2560),
        DType::F32,
        DType::Fp8E4M3,
        Layout::BlockScaled { block: 128 },
        StreamId(0),
    )
    .expect("f32 to FP8 quantizer");
    assert_eq!(
        plan.kernel(),
        expected(Target::Sm121, Image::GlmPrefill, "glm_quantise_rows")
    );
}

#[test]
fn gather_and_collective_plans_use_their_catalogue_entries() {
    let engram = requirement(&arm(DEEPSEEK, 4), Part::Engram);
    // The Engram table row width is engram_head_dim (256), which is the geometry's head width.
    let gather = GatherPlan::new(CTX, &engram, true, 8, 256, 16, StreamId(0)).expect("Engram");
    assert_eq!(
        gather.kernel(),
        expected(Target::Sm121, Image::Deepseek, "engram_embed_sm121")
    );

    let embedding = requirement(&arm(QWEN, 1), Part::Embedding);
    let gather =
        GatherPlan::new(CTX, &embedding, false, 8, 2560, 1, StreamId(0)).expect("embedding");
    assert_eq!(
        gather.kernel(),
        expected(Target::Sm121, Image::Qwen4Exp, "q4e_embed")
    );

    let collective = synthetic(Part::Collective, 0, 0, 0, Precision::Bf16);
    let mut collective = collective;
    collective.geometry.world = 4;
    let one_shot = CollectivePlan::new(
        CTX,
        &collective,
        CollectiveKind::OneShot,
        1024,
        2,
        4,
        StreamId(0),
    )
    .expect("one-shot");
    assert_eq!(
        one_shot.kernel(),
        expected(
            Target::Sm121,
            Image::Rocenante,
            "rocenante_oneshot_allreduce_bf16_tp4"
        )
    );

    let two_shot = CollectivePlan::new(
        CTX,
        &collective,
        CollectiveKind::TwoShot,
        1024,
        4,
        4,
        StreamId(0),
    )
    .expect("two-shot");
    assert!(two_shot.two_shot_splits());
    assert_eq!(two_shot.workspace_floats(), 256);
    assert_eq!(
        two_shot.kernel(),
        expected(Target::Sm121, Image::Qwen4Exp, "q4e_sum_slots")
    );
}

#[test]
fn the_combine_is_reachable_for_every_arm_shape() {
    let deepseek = synthetic(Part::MoeCombine, 5120, 0, 8, Precision::Bf16);
    let mut deepseek = deepseek;
    deepseek.geometry.top_k = 3;
    let combine = ExpertCombinePlan::new(CTX, &deepseek, 8, DType::Bf16, StreamId(0))
        .expect("DeepSeek MTP combine");
    assert_eq!(
        combine.kernel(),
        expected(Target::Sm121, Image::Deepseek, "moe_routed_combine_sm121")
    );

    let glm = synthetic(Part::MoeCombine, 4096, 0, 8, Precision::Bf16);
    let mut glm = glm;
    glm.geometry.top_k = 8;
    let combine =
        ExpertCombinePlan::new(CTX, &glm, 8, DType::F32, StreamId(0)).expect("GLM combine");
    assert_eq!(
        combine.kernel(),
        expected(Target::Sm121, Image::Qwen4Moe, "q4e_moe_combine")
    );

    let qwen = synthetic(Part::MoeCombine, 2560, 0, 8, Precision::Bf16);
    let mut qwen = qwen;
    qwen.geometry.top_k = 10;
    let combine =
        ExpertCombinePlan::new(CTX, &qwen, 8, DType::F32, StreamId(0)).expect("Qwen combine");
    assert_eq!(
        combine.kernel(),
        expected(Target::Sm121, Image::Qwen4Moe, "q4e_moe_combine")
    );
}

#[test]
fn an_unsupported_combine_shape_is_refused() {
    let requirement = synthetic(Part::MoeCombine, 1234, 0, 8, Precision::Bf16);
    let mut requirement = requirement;
    requirement.geometry.top_k = 8;
    let error = ExpertCombinePlan::new(CTX, &requirement, 8, DType::Bf16, StreamId(0))
        .expect_err("no combine body at 1234");
    assert!(error.to_string().contains("hidden"), "{error}");
}

#[test]
fn the_router_resolves_each_arm() {
    let qwen = synthetic(Part::MoeRouter, 2560, 0, 1, Precision::Bf16);
    let mut qwen = qwen;
    qwen.geometry.experts = 512;
    qwen.geometry.top_k = 10;
    let router = MoeRoutingPlan::new(CTX, &qwen, 1, StreamId(0)).expect("Qwen router");
    assert_eq!(
        router.kernel(),
        expected(Target::Sm121, Image::Qwen4Moe, "q4e_router")
    );

    let glm = synthetic(Part::MoeRouter, 4096, 0, 1, Precision::Fp8E4M3);
    let mut glm = glm;
    glm.geometry.experts = 288;
    glm.geometry.top_k = 8;
    let router = MoeRoutingPlan::new(CTX, &glm, 1, StreamId(0)).expect("GLM router");
    assert_eq!(
        router.kernel(),
        expected(Target::Sm121, Image::GlmDecode, "q4e_noaux_router")
    );

    let deepseek = synthetic(Part::MoeRouter, 5120, 0, 1, Precision::Bf16);
    let mut deepseek = deepseek;
    deepseek.geometry.experts = 384;
    deepseek.geometry.top_k = 6;
    let router = MoeRoutingPlan::new(CTX, &deepseek, 1, StreamId(0)).expect("DeepSeek router");
    assert_eq!(
        router.kernel(),
        expected(Target::Sm121, Image::Deepseek, "opt_router_sm121")
    );
}

#[test]
fn attention_validates_before_it_divides() {
    let gqa = requirement(&arm(QWEN, 1), Part::GqaAttention);
    let context = gqa.geometry.positions;
    let head_dim = gqa.geometry.head_dim;
    let error = AttentionPlan::new(
        CTX,
        &gqa,
        AttentionKind::Gqa {
            query_heads: 8,
            kv_heads: 0,
        },
        1,
        context,
        head_dim,
        StreamId(0),
    )
    .expect_err("zero kv heads");
    assert!(matches!(error, OpError::Shape { .. }), "{error}");

    let sparse = requirement(&arm(QWEN, 1), Part::SparseIndexer);
    let error = AttentionPlan::new(
        CTX,
        &sparse,
        AttentionKind::Sparse {
            selected: 64,
            block: 0,
        },
        1,
        1 << 20,
        sparse.geometry.head_dim,
        StreamId(0),
    )
    .expect_err("zero block");
    assert!(matches!(error, OpError::Shape { .. }), "{error}");
}

#[test]
fn an_nvfp4_attention_is_unsupported_not_unresolved() {
    let mut requirement = requirement(&arm(DEEPSEEK, 4), Part::MlaAttention);
    requirement.precision = Precision::NvFp4;
    let error = AttentionPlan::new(
        CTX,
        &requirement,
        AttentionKind::Mla,
        1,
        131_072,
        requirement.geometry.head_dim,
        StreamId(0),
    )
    .expect_err("attention has no NVFP4 body");
    assert!(matches!(error, OpError::Unsupported { .. }), "{error}");
}

#[test]
fn a_scan_resolves_only_the_implemented_head_template() {
    let qwen = requirement(&arm(QWEN, 1), Part::GdnScan);
    let plan = ScanPlan::new(
        CTX,
        &qwen,
        ScanKind::Gdn,
        8,
        qwen.geometry.heads,
        128,
        128,
        8,
        StreamId(0),
    )
    .expect("valid");
    assert_eq!(plan.launch().grid, (qwen.geometry.heads as u32, 8, 1));
    assert_eq!(
        plan.kernel(),
        expected(Target::Sm121, Image::Qwen4Scan, "q4p_gdn_scan")
    );

    // The launch's own head widths select the instantiated template: 64/64 and 128/256 are
    // separate symbols, and the released 128/128 body keeps `q4p_gdn_scan`. A plan that always
    // named q4p_gdn_scan would launch the 128/128 body on the other two pairs.
    for (key, value, symbol) in [
        (64, 64, "q4p_gdn_scan_64_64"),
        (128, 128, "q4p_gdn_scan"),
        (128, 256, "q4p_gdn_scan_128_256"),
    ] {
        let plan = ScanPlan::new(
            CTX,
            &qwen,
            ScanKind::Gdn,
            8,
            qwen.geometry.heads,
            key,
            value,
            8,
            StreamId(0),
        )
        .unwrap_or_else(|error| panic!("{key}x{value}: {error}"));
        assert_eq!(
            plan.kernel(),
            expected(Target::Sm121, Image::Qwen4Scan, symbol),
            "{key}x{value}"
        );
    }
    // 96/96 is not an instantiated template.
    let error = ScanPlan::new(
        CTX,
        &qwen,
        ScanKind::Gdn,
        8,
        qwen.geometry.heads,
        96,
        96,
        8,
        StreamId(0),
    )
    .expect_err("96/96 is not an instantiated scan template");
    let reason = error.to_string();
    assert!(
        reason.contains("head pair (96, 96) is not one of [(64, 64), (128, 128), (128, 256)]"),
        "{reason}"
    );

    let glm = requirement(&arm(GLM, 1), Part::KdaScan);
    let kda = ScanPlan::new(
        CTX,
        &glm,
        ScanKind::Kda,
        8,
        glm.geometry.heads,
        glm.geometry.head_dim,
        glm.geometry.v_head,
        8,
        StreamId(0),
    )
    .expect("GLM KDA");
    assert_eq!(
        kda.kernel(),
        expected(Target::Sm121, Image::Qwen4Scan, "q4p_kda_scan")
    );
    // The image exports 64/64 and 128/256 KDA entries, but the GLM host launches only the
    // released 128/128 body, so the catalogue claims only the template the host selects.
    let error = ScanPlan::new(
        CTX,
        &glm,
        ScanKind::Kda,
        8,
        glm.geometry.heads,
        64,
        64,
        8,
        StreamId(0),
    )
    .expect_err("the GLM KDA host launches only 128/128");
    let reason = error.to_string();
    assert!(
        reason.contains("head pair (64, 64) is not one of [(128, 128)]"),
        "{reason}"
    );

    // The requirement stays a valid 128/128 template; only the schedule's head width is the
    // unsupported value. The plan must resolve the geometry it will launch, not the original.
    assert_eq!((qwen.geometry.head_dim, qwen.geometry.v_head), (128, 128));
}

/// The schedule's head width and group are the kernel's obligations. A valid requirement must
/// not let an unsupported constructor argument resolve a body whose constraint rejects it.
#[test]
fn attention_resolves_the_head_shape_the_schedule_asks_for() {
    let requirement = requirement(&arm(QWEN, 1), Part::GqaAttention);
    assert_eq!(requirement.geometry.head_dim, 256);
    let context = requirement.geometry.positions;
    let error = AttentionPlan::new(
        CTX,
        &requirement,
        AttentionKind::Gqa {
            query_heads: 24,
            kv_heads: 2,
        },
        8,
        context,
        12,
        StreamId(0),
    )
    .expect_err("the 12-wide head is not a 16-wide multiple");
    assert!(error.to_string().contains("head width 12"), "{error}");

    let error = AttentionPlan::new(
        CTX,
        &requirement,
        AttentionKind::Gqa {
            query_heads: 10,
            kv_heads: 2,
        },
        8,
        context,
        256,
        StreamId(0),
    )
    .expect_err("a five-head group is not one of the published groups");
    assert!(error.to_string().contains("group size 5"), "{error}");
}

/// The gather's table width is a kernel obligation (the hidden row for an embedding, the head
/// width for Engram), not a value the requirement's original shape may override.
#[test]
fn a_gather_resolves_the_table_width_it_was_built_for() {
    let engram = requirement(&arm(DEEPSEEK, 4), Part::Engram);
    assert_eq!(engram.geometry.hidden, 5120);
    let error = GatherPlan::new(CTX, &engram, true, 8, 4096, 16, StreamId(0))
        .expect_err("the Engram body is compiled for a 256-wide head");
    assert!(error.to_string().contains("head width 4096"), "{error}");
    let error = GatherPlan::new(CTX, &engram, true, 8, 0, 16, StreamId(0))
        .expect_err("an empty table width is refused before it resolves");
    assert!(error.to_string().contains("a dimension is zero"), "{error}");
}

/// The GLM device launches its LayerNorm, activation and indexer bodies over a prefill chunk's
/// rows, not only a token's. Each symbol must therefore serve the prefill pass.
#[test]
fn glm_prefill_row_bodies_resolve_in_the_prefill_pass() {
    let glm = arm(GLM, 4);
    let layer = synthetic(Part::LayerNorm, 4096, 0, 32, Precision::Bf16);
    let norm = NormPlan::new(
        CTX,
        &layer,
        NormKind::LayerNorm,
        32,
        4096,
        DType::Bf16,
        StreamId(0),
    )
    .expect("prefill-row LayerNorm");
    assert_eq!(
        norm.kernel(),
        expected(Target::Sm121, Image::GlmDecode, "q4e_layer_norm")
    );

    let activation = synthetic(Part::Activation, 4096, 3072, 32, Precision::Bf16);
    let swiglu = CTX
        .resolve(&activation, KernelPhase::Prefill, &["q4e_swiglu_limited"])
        .expect("prefill-row GLM SwiGLU");
    assert_eq!(swiglu.image, Image::GlmDecode);
    assert_eq!(swiglu.symbol_name(), "q4e_swiglu_limited");
    ActivationPlan::new(
        CTX,
        &activation,
        ActivationKind::SwiGlu,
        32,
        3072,
        StreamId(0),
    )
    .expect("the GLM activation plan resolves a prefill body");

    let indexer = requirement(&glm, Part::SparseIndexer);
    for symbol in ["q4e_glm_index_scores", "q4e_glm_pool", "q4e_glm_pick"] {
        let resolved = CTX
            .resolve(&indexer, KernelPhase::Prefill, &[symbol])
            .unwrap_or_else(|error| panic!("{symbol}: {error}"));
        assert_eq!(resolved.image, Image::GlmDecode, "{symbol}");
    }
    let plan = AttentionPlan::new(
        CTX,
        &indexer,
        AttentionKind::Sparse {
            selected: 2048,
            block: 4,
        },
        32,
        1 << 20,
        indexer.geometry.head_dim,
        StreamId(0),
    )
    .expect("the GLM sparse indexer resolves a prefill body");
    assert_eq!(plan.kernel().image, Image::GlmDecode);
}

/// The 16-patch kernel exists: `opt_vision_sm121.cu` exports `opt_vision_patchify_16` and
/// `VisionShape::of_with_patch` derives its grid. The catalogue still claims only the 14-patch
/// tower because the Qwen tower's orchestration and weight loading are its own arm's (D84),
/// and a shared catalogue entry would claim a tower this tree does not run. The release that
/// asks for 16 is refused, naming the patch, even when the 16 symbol is requested by name.
#[test]
fn an_unimplemented_vision_patch_is_refused() {
    let qwen = requirement(&arm(QWEN, 1), Part::Vision);
    assert_eq!(qwen.geometry.patch_size, 16);
    let error = CTX
        .resolve(&qwen, KernelPhase::Prefill, &["opt_vision_patchify_16"])
        .expect_err("no orchestrated 16-patch tower is claimed");
    assert!(
        error
            .to_string()
            .contains("vision patch 16 is not one of [14]"),
        "{error}"
    );
}

#[test]
fn a_layer_norm_at_an_unimplemented_width_is_refused_not_fallen_back_to_rms() {
    let requirement = synthetic(Part::LayerNorm, 2560, 0, 1, Precision::Bf16);
    let error = NormPlan::new(
        CTX,
        &requirement,
        NormKind::LayerNorm,
        1,
        2560,
        DType::Bf16,
        StreamId(0),
    )
    .expect_err("only the 4096-wide LayerNorm body exists");
    assert!(error.to_string().contains("does not equal 4096"), "{error}");
}

#[test]
fn a_collective_refuses_a_world_with_nothing_to_collect() {
    let collective = synthetic(Part::Collective, 0, 0, 0, Precision::Bf16);
    let error = CollectivePlan::new(
        CTX,
        &collective,
        CollectiveKind::TwoShot,
        1024,
        4,
        1,
        StreamId(0),
    )
    .expect_err("a one-rank collective is a bug");
    assert!(matches!(error, OpError::Shape { .. }), "{error}");
}

#[test]
fn a_reduce_scatter_has_no_catalogued_kernel_rather_than_a_split_k_gemm() {
    let collective = synthetic(Part::Collective, 0, 0, 0, Precision::Bf16);
    let mut collective = collective;
    collective.geometry.world = 4;
    let error = CollectivePlan::new(
        CTX,
        &collective,
        CollectiveKind::ReduceScatter,
        1024,
        4,
        4,
        StreamId(0),
    )
    .expect_err("no reduce-scatter kernel");
    assert!(
        error
            .to_string()
            .contains("no catalogued collective kernel")
    );
}

#[test]
fn a_plan_never_resolves_a_symbol_from_another_precision_class() {
    // q4e_gemm_bf16 is exported by the token image, but it is not an NVFP4 body: the NVFP4
    // quantiser in the same image must not make it eligible.
    let requirement = synthetic(Part::DenseMlp, 2560, 640, 1, Precision::NvFp4);
    let error = CTX
        .resolve(&requirement, KernelPhase::Decode, &["q4e_gemm_bf16"])
        .expect_err("q4e_gemm_bf16 is not NVFP4");
    assert!(error.to_string().contains("q4e_gemm_bf16"), "{error}");
}

#[test]
fn a_layer_plan_validates_in_order_and_sums_its_workspace() {
    let rms = synthetic(Part::RmsNorm, 2560, 0, 8, Precision::Bf16);
    let mut layer = LayerPlan::new("layer 0");
    layer.push(
        NormPlan::new(
            CTX,
            &rms,
            NormKind::RmsNorm,
            8,
            2560,
            DType::Bf16,
            StreamId(0),
        )
        .expect("valid"),
    );
    let dense_requirement = synthetic(Part::DenseGemm, 5120, 2560, 8, Precision::MxFp8);
    layer.push(dense(&dense_requirement, 8, 2560, 1));
    assert_eq!(layer.ops.len(), 2);
    layer.validate().expect("both plans are valid");
}

#[test]
fn the_parity_harness_names_the_first_disagreeing_element() {
    let mut report = ParityReport::new("layer 0");
    report
        .record(
            "attention",
            OpKind::MlaAttention,
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0],
            Tolerance::EXACT,
        )
        .expect("equal");
    let error = report
        .record(
            "mlp",
            OpKind::DenseGemm,
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.5],
            Tolerance::F32_ROUNDING,
        )
        .expect_err("3.5 is past the tolerance");
    assert!(error.to_string().contains("at 2"), "{error}");
    assert!(!report.passed());
    assert_eq!(report.failures(), vec!["layer 0 at 2"]);
}

#[test]
fn a_non_finite_device_value_never_passes_parity() {
    let error = sparkflash_ops::parity::compare(
        "logits",
        OpKind::Norm,
        &[0.0],
        &[f32::NAN],
        Tolerance::new(f32::INFINITY, f32::INFINITY),
    )
    .expect_err("NaN must fail");
    assert!(error.to_string().contains("at 0"));
}
