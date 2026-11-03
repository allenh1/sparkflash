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
use sparkflash_ops::gemm::{DenseGemmPlan, GroupedMoeGemmPlan, MoeRoutingPlan};
use sparkflash_ops::parity::{LayerPlan, ParityReport, Tolerance};
use sparkflash_ops::plan::{DType, Layout, Matrix, OpKind, OpPlan, Precision, StreamId};
use sparkflash_ops::scan::{ScanKind, ScanPlan};

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
