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

use sparkflash_ops::attention::{AttentionKind, AttentionPlan, SplitKvDecodePlan};
use sparkflash_ops::collective::{CollectiveKind, CollectivePlan};
use sparkflash_ops::epilogue::{GatherPlan, NormKind, NormPlan, QuantizePlan};
use sparkflash_ops::gemm::{DenseGemmPlan, ExpertCombinePlan, GroupedMoeGemmPlan, MoeRoutingPlan};
use sparkflash_ops::images::Image;
use sparkflash_ops::parity::{LayerPlan, ParityReport, Tolerance};
use sparkflash_ops::plan::{DType, KernelId, Layout, Matrix, OpKind, OpPlan, Precision, StreamId};
use sparkflash_ops::scan::{ScanKind, ScanPlan};

/// Every kernel the typed plans can produce, paired with the image that exports its symbol.
///
/// [Image] exposes no symbol list, so this literal table is the pin: a symbol rename that leaves
/// a plan pointing at another image fails here instead of launching from the wrong cubin.
const PLAN_KERNELS: &[(Image, &str)] = &[
    (Image::Deepseek, "b12x_mxfp8_dense_sm121"),
    (Image::Deepseek, "b12x_mxfp8_dense_split_k_sm121"),
    (Image::Deepseek, "b12x_mxfp4_moe_sm121"),
    (Image::GlmPrefill, "glm_prefill_fp8b128_mma"),
    (Image::GlmPrefill, "glm_prefill_bf16_mma"),
    (Image::Qwen4Exp, "q4e_gemm_bf16"),
    (Image::GlmMoe, "glm_moe_fp8b128_mma"),
    (Image::GlmMoe, "glm_moe_fp8b128_mma_bf16a"),
    (Image::Qwen4Moe, "q4e_expert_gate_up"),
    (Image::Deepseek, "moe_router_sm121"),
    (Image::Deepseek, "moe_routed_combine_sm121"),
    (Image::Deepseek, "flash_mla_csa2_sm121"),
    (Image::Qwen4Attn, "q4e_attn"),
    (Image::Qwen4Attn, "q4e_attn_group"),
    (Image::Qwen4Attn, "q4e_attn_group6"),
    (Image::Deepseek, "flash_mla_csa2_seq_sm121"),
    (Image::Deepseek, "flash_mla_csa2_ring_sm121"),
    (Image::Deepseek, "opt_attn_sm121"),
];

fn dense_kernel(precision: Precision, split_k: usize) -> KernelId {
    let layout = match precision {
        Precision::MxFp8 => Layout::BlockScaled { block: 32 },
        Precision::NvFp4 => Layout::BlockScaled { block: 16 },
        Precision::Fp8E4M3 => Layout::BlockScaled { block: 128 },
        Precision::F32 | Precision::Bf16 => Layout::RowMajor,
    };
    DenseGemmPlan::new(
        Matrix::new(64, 2560),
        2560,
        precision,
        layout,
        StreamId(0),
        split_k,
    )
    .expect("a valid dense GEMM")
    .kernel()
}

fn grouped_kernel(precision: Precision) -> KernelId {
    GroupedMoeGemmPlan::new(512, 2560, 1024, 512, 10, precision, StreamId(0))
        .expect("a valid grouped MoE GEMM")
        .kernel()
}

fn attention_kernel(kind: AttentionKind) -> KernelId {
    AttentionPlan::new(kind, 8, 262_144, 128, Precision::Bf16, StreamId(0))
        .expect("a valid attention plan")
        .kernel()
}

#[test]
fn a_dense_gemm_names_one_kernel_and_allocates_nothing_at_launch() {
    let plan = DenseGemmPlan::new(
        Matrix::new(64, 4096),
        2560,
        Precision::MxFp8,
        Layout::BlockScaled { block: 32 },
        StreamId(3),
        1,
    )
    .expect("valid");
    assert_eq!(plan.kind(), OpKind::DenseGemm);
    assert_eq!(plan.workspace_floats(), 0);
    assert_eq!(plan.launch().stream, StreamId(3));
    assert_eq!(plan.launch().kernel.symbol, "b12x_mxfp8_dense_sm121");
}

#[test]
fn a_split_k_gemm_reserves_its_partials_and_names_the_split_kernel() {
    let plan = DenseGemmPlan::new(
        Matrix::new(64, 128),
        2560,
        Precision::MxFp8,
        Layout::BlockScaled { block: 32 },
        StreamId(1),
        5,
    )
    .expect("valid");
    assert_eq!(plan.kind(), OpKind::SplitKGemm);
    assert_eq!(plan.workspace_floats(), 5 * 64 * 128);
    assert_eq!(
        plan.launch().kernel.symbol,
        "b12x_mxfp8_dense_split_k_sm121"
    );
}

#[test]
fn an_uneven_split_is_refused_before_a_launch() {
    let error = DenseGemmPlan::new(
        Matrix::new(64, 128),
        2559,
        Precision::MxFp8,
        Layout::BlockScaled { block: 32 },
        StreamId(0),
        4,
    )
    .expect_err("2559 does not split into four");
    assert!(error.to_string().contains("does not split"));
}

#[test]
fn a_grouped_moe_gemm_validates_its_expert_table() {
    let plan = GroupedMoeGemmPlan::new(512, 2560, 1024, 512, 10, Precision::NvFp4, StreamId(0))
        .expect("valid");
    assert_eq!(plan.kind(), OpKind::GroupedMoeGemm);
    assert!(plan.workspace_floats() > 0);
    assert!(
        GroupedMoeGemmPlan::new(1, 1, 1, 2, 3, Precision::NvFp4, StreamId(0)).is_err(),
        "top-k past the expert count must be refused"
    );
}

#[test]
fn grouped_attention_selects_the_kernel_for_its_head_group() {
    let six = AttentionPlan::new(
        AttentionKind::Gqa {
            query_heads: 24,
            kv_heads: 4,
        },
        8,
        262_144,
        128,
        Precision::Bf16,
        StreamId(0),
    )
    .expect("valid");
    assert_eq!(six.group_size(), 6);
    assert_eq!(
        six.kernel(),
        KernelId::new(Image::Qwen4Attn, "q4e_attn_group6")
    );
    assert_eq!(six.launch().shared_bytes, 63_600);

    for (query_heads, kv_heads) in [(24, 6), (32, 8)] {
        let other = AttentionPlan::new(
            AttentionKind::Gqa {
                query_heads,
                kv_heads,
            },
            8,
            262_144,
            128,
            Precision::Bf16,
            StreamId(0),
        )
        .expect("valid");
        assert_eq!(other.group_size(), 4);
        assert_eq!(
            other.kernel(),
            KernelId::new(Image::Qwen4Attn, "q4e_attn_group")
        );
        assert_eq!(other.launch().shared_bytes, 45_136);
    }

    let twelve = AttentionPlan::new(
        AttentionKind::Gqa {
            query_heads: 24,
            kv_heads: 2,
        },
        8,
        262_144,
        128,
        Precision::Bf16,
        StreamId(0),
    )
    .expect("valid");
    assert_eq!(twelve.group_size(), 12);
    assert_eq!(twelve.kernel(), KernelId::new(Image::Qwen4Attn, "q4e_attn"));
    assert_eq!(twelve.launch().shared_bytes, 0);
}

#[test]
fn a_grouped_moe_gemm_refuses_mxfp8_until_a_grouped_kernel_exists() {
    let error = GroupedMoeGemmPlan::new(512, 2560, 1024, 512, 10, Precision::MxFp8, StreamId(0))
        .expect_err("no grouped MXFP8 expert kernel");
    assert!(error.to_string().contains("mxfp8"));
}

#[test]
fn every_plan_kernel_names_the_image_that_exports_its_symbol() {
    let decode = AttentionPlan::new(
        AttentionKind::Mla,
        8,
        262_144,
        128,
        Precision::Bf16,
        StreamId(0),
    )
    .expect("valid");
    let produced = [
        ("dense mxfp8", dense_kernel(Precision::MxFp8, 1)),
        ("dense mxfp8 split-k", dense_kernel(Precision::MxFp8, 5)),
        ("dense nvfp4", dense_kernel(Precision::NvFp4, 1)),
        ("dense e4m3", dense_kernel(Precision::Fp8E4M3, 1)),
        ("dense bf16", dense_kernel(Precision::Bf16, 1)),
        ("dense f32", dense_kernel(Precision::F32, 1)),
        ("grouped nvfp4", grouped_kernel(Precision::NvFp4)),
        ("grouped e4m3", grouped_kernel(Precision::Fp8E4M3)),
        ("grouped bf16", grouped_kernel(Precision::Bf16)),
        ("grouped f32", grouped_kernel(Precision::F32)),
        (
            "router",
            MoeRoutingPlan::new(8, 2560, 512, 10, StreamId(0))
                .expect("valid")
                .kernel(),
        ),
        (
            "combine f32",
            ExpertCombinePlan::new(8, 2560, 10, DType::F32, StreamId(0))
                .expect("valid")
                .kernel(),
        ),
        (
            "combine bf16",
            ExpertCombinePlan::new(8, 2560, 10, DType::Bf16, StreamId(0))
                .expect("valid")
                .kernel(),
        ),
        ("mla", attention_kernel(AttentionKind::Mla)),
        (
            "gqa group 4",
            attention_kernel(AttentionKind::Gqa {
                query_heads: 24,
                kv_heads: 6,
            }),
        ),
        (
            "gqa group 6",
            attention_kernel(AttentionKind::Gqa {
                query_heads: 24,
                kv_heads: 4,
            }),
        ),
        (
            "gqa group 12",
            attention_kernel(AttentionKind::Gqa {
                query_heads: 24,
                kv_heads: 2,
            }),
        ),
        (
            "sparse",
            attention_kernel(AttentionKind::Sparse {
                selected: 64,
                block: 4,
            }),
        ),
        ("csa2", attention_kernel(AttentionKind::Csa2 { ratio: 4 })),
        (
            "split-kv decode",
            SplitKvDecodePlan::new(decode, 4).expect("valid").kernel(),
        ),
    ];

    for (name, kernel) in &produced {
        assert!(
            PLAN_KERNELS.contains(&(kernel.image, kernel.symbol)),
            "{name} produces {}::{} which is not in the pinned table",
            kernel.image.name(),
            kernel.symbol
        );
    }
    for expected in PLAN_KERNELS {
        assert!(
            produced
                .iter()
                .any(|(_, kernel)| (kernel.image, kernel.symbol) == *expected),
            "no plan produces {}::{}",
            expected.0.name(),
            expected.1
        );
    }
}

#[test]
fn a_router_and_attention_plan_share_the_typed_vocabulary() {
    let router = MoeRoutingPlan::new(8, 2560, 512, 10, StreamId(0)).expect("valid");
    assert_eq!(router.kind(), OpKind::MoeRouting);
    let attention = AttentionPlan::new(
        AttentionKind::Gqa {
            query_heads: 24,
            kv_heads: 4,
        },
        8,
        262_144,
        128,
        Precision::Bf16,
        StreamId(0),
    )
    .expect("valid");
    assert_eq!(attention.group_size(), 6);
    assert_eq!(attention.kind(), OpKind::GqaAttention);
    let decode = SplitKvDecodePlan::new(attention, 4).expect("valid");
    assert_eq!(decode.launch().grid.2, 4);
}

#[test]
fn a_scan_uses_the_shared_geometry_rule() {
    let plan = ScanPlan::new(
        ScanKind::Gdn,
        8,
        48,
        128,
        128,
        8,
        Precision::F32,
        StreamId(0),
    )
    .expect("valid");
    assert_eq!(plan.launch().grid, (48, 8, 1));
    assert_eq!(plan.launch().block, (64, 1, 1));
}

#[test]
fn a_collective_refuses_a_world_with_nothing_to_collect() {
    assert!(
        CollectivePlan::new(CollectiveKind::TwoShot, 1024, 4, 1, StreamId(0)).is_err(),
        "a one-rank collective is a bug"
    );
    let plan =
        CollectivePlan::new(CollectiveKind::TwoShot, 1024, 4, 4, StreamId(0)).expect("valid");
    assert!(plan.two_shot_splits());
    assert_eq!(plan.workspace_floats(), 256);
}

#[test]
fn epilogues_and_gathers_carry_their_own_validation() {
    let norm = NormPlan::new(NormKind::RmsNorm, 8, 2560, DType::Bf16, StreamId(0)).expect("valid");
    assert_eq!(norm.kind(), OpKind::Norm);
    assert!(
        QuantizePlan::new(
            Matrix::new(8, 2560),
            DType::F32,
            DType::NvFp4,
            Layout::BlockScaled { block: 16 },
            StreamId(0),
        )
        .is_ok()
    );
    let gather = GatherPlan::new(true, 8, 160, 16, StreamId(0)).expect("valid");
    assert_eq!(gather.kind(), OpKind::EngramGather);
}

#[test]
fn a_layer_plan_validates_in_order_and_sums_its_workspace() {
    let mut layer = LayerPlan::new("layer 0");
    layer.push(NormPlan::new(NormKind::RmsNorm, 8, 2560, DType::Bf16, StreamId(0)).expect("valid"));
    layer.push(
        DenseGemmPlan::new(
            Matrix::new(8, 2560),
            2560,
            Precision::MxFp8,
            Layout::BlockScaled { block: 32 },
            StreamId(0),
            1,
        )
        .expect("valid"),
    );
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
