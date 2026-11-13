#!/usr/bin/env python3
# Copyright (c) 2026 Hunter L. Allen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify that the m4_abi quarantine split is a verbatim move of the frozen module.

D26 froze the Milestone-4 ABI source-text lint at one file, crates/sparkflash-kernels/src/m4_abi.rs
(13,906 lines at main d7ed586a). The split keeps the frozen scanners verbatim but files them by what
they are: the root file keeps the module documentation, the const API the launch and driver modules
import, every root test with its cfg(test) helpers and the pin_helpers module; src/m4_abi/scanner.rs
holds the C/C++ token and control-flow scanners the acquire guard uses; src/m4_abi/acquire_guard.rs
holds the acquire-guard test module, whose module path (and therefore whose 43 test ids) is unchanged.

This script is the split's proof, in the style of ci/verify-m7-layer0-split.py. It reads the base
commit (never a working-tree copy of the old file, so a later edit cannot make the verbatim claim
true retroactively), re-derives the three files the split should have produced - the base text cut
at the same slice boundaries, plus the glue that declares the submodules, plus the two allow-listed
edits below - formats the result with the repo's rustfmt, and requires the working tree to be
byte-identical to it. Any edit to a moved scanner, test, doc line or pinch is therefore either part
of the mechanical transformation or the check fails.

The only non-verbatim text, allow-listed and written out below:

  1. self-scan adapters inside the moved acquire_guard tests: three tests read this module family's
     own source (the rejection-routing scan, the documentation-count scan, and the doc
     lazy-continuation scan). They now read src/m4_abi/scanner.rs / the family concatenation,
     because the text they scan moved; their assertions are unchanged.
  2. the root module's freeze notice and the D26 pin comment, which now name the aggregated family
     length instead of one file's length.
  3. the K8 scanner data updates and the acquire-guard per-call fix, applied after the split below.
     These do not add a scanner: the certified head gained a w_block_max operand in its frozen
     signature table, the six kernels' static-assert counts rose with their K8 symbols, and the
     mHC file now carries scalar and six-row kernels so the guard checks every observing path.
     They are written out as exact replacements so this script still fails on any other drift.

Exit 0 means the split is exactly that regeneration; exit 1 prints the first differing line.

Usage: python3 ci/verify-m4-abi-split.py [--base COMMIT]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

BASE_COMMIT = "d7ed586a"
BASE_PATH = "crates/sparkflash-kernels/src/m4_abi.rs"
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "crates/sparkflash-kernels/src"
ROOT_FILE = SRC / "m4_abi.rs"
SCANNER_FILE = SRC / "m4_abi" / "scanner.rs"
GUARD_FILE = SRC / "m4_abi" / "acquire_guard.rs"
CHECK_SCRIPT = REPO_ROOT / "ci" / "check-m4-abi-size.sh"
TMP = REPO_ROOT / "target" / "m4-abi-verify"
WRAPPER = "#[cfg(test)]\nmod acquire_guard {\n"
SCANNER_START = "/// One C token, the two lines it can be reported by"
FREEZE_TITLE = "//! # FREEZE (D26, 2026-10-18): no new source-text scanners in this file"
FREEZE_TITLE_NEW = "//! # FREEZE (D26, 2026-10-18): no new source-text scanners in this module family"
FREEZE_BULLET = "//!   * no new scanner, parser or lint for a new kernel invariant is added to this file; a new"
FREEZE_BULLET_NEW = "//!   * no new scanner, parser or lint for a new kernel invariant is added to this family; a new"
FREEZE_LENGTH = "//!   * the file's length is pinned by ci/check-m4-abi-size.sh, wired into .gitlab-ci.yml; a change that\n//!     makes it longer fails that check, and a legitimate shrink updates the pin in the same commit.\n"
FREEZE_LENGTH_NEW = "//!   * the family's total length is pinned by ci/check-m4-abi-size.sh, wired into .gitlab-ci.yml: the\n//!     root file plus the pinned submodules under src/m4_abi/ (the quarantine split recorded in\n//!     docs/archive/build/m4-abi-split.md). Any net change fails that check, and a legitimate shrink updates the\n//!     pin in the same commit.\n"
POST_SPLIT_ROOT_EDITS = [
    (
        '"const __nv_bfloat16 *__restrict__ hidden; const __nv_bfloat16 *__restrict__ w; float *__restrict__ logits;',
        '"const __nv_bfloat16 *__restrict__ hidden; const __nv_bfloat16 *__restrict__ w; const __nv_bfloat16 *__restrict__ w_block_max; float *__restrict__ logits;',
    ),
    (
        '    ("mega_mhc_sm121.cu", 6),\n    ("b12x_mxfp8_dense_sm121.cu", 6),',
        '    ("mega_mhc_sm121.cu", 7),\n    ("b12x_mxfp8_dense_sm121.cu", 15),',
    ),
    (
        '    ("b12x_mxfp8_wo_a_sm121.cu", 11),\n    ("cert_lm_head_sm121.cu", 8),',
        '    ("b12x_mxfp8_wo_a_sm121.cu", 20),\n    ("cert_lm_head_sm121.cu", 9),',
    ),
    (
        '    ("b12x_mxfp4_moe_sm121.cu", 10),\n',
        '    ("b12x_mxfp4_moe_sm121.cu", 12),\n',
    ),
    (
        '        checked, 78,\n        "the milestone\'s static asserts must all be evaluated: 20 in the shared header plus 6, 6, \\\n',
        '        checked, 98,\n        "the milestone\'s static asserts must all be evaluated: 20 in the shared header plus 6, 7, \\\n',
    ),
    (
        '        checked, 98,\n',
        '        checked, 100,\n',
    ),
    (
        'the milestone\'s static asserts must all be evaluated: 20 in the shared header plus 6, 7, \\\n',
        'the milestone\'s static asserts must all be evaluated: 20 in the shared header plus 7, 15, \\\n',
    ),
    (
        '         10, 17, 11 and 8 in the six kernels. The MoE count rose from 7 to 10 with D14, which pins \\\n         the FP8 E4M3 activation operand\'s byte width in the build itself; flash_mla\'s rose from 13 \\\n         to 15 with the engine window format\'s two layout asserts, then to 17 with the union \\\n         capacity\'s two. The format VALUE is not among them: it is a runtime launch parameter and \\\n         is guarded at kernel entry instead"\n',
        '         12, 17, 20 and 9 in the six kernels (mega_mhc, dense, MoE, flash_mla, wo_a, cert). The MoE \\\n         count rose from 10 to 12 with the split-K tiling: the 256-thread block identity and the \\\n         one-reducer-per-(slot, column) bound. D14 had raised it from 7 to 10 for the FP8 E4M3 \\\n         activation operand\'s byte width; flash_mla\'s rose from 13 to 15 with the engine window \\\n         format\'s two layout asserts, then to 17 with the union capacity\'s two. The format VALUE is \\\n         not among them: it is a runtime launch parameter and is guarded at kernel entry instead"\n',
    ),
    (
        '            && code.contains("acc += (partial * act_scale) * w_scale;")\n',
        '            && code.contains("terms[slot][block][col_local] =")\n            && code.contains("__fmul_rn(__fmul_rn(partial, act_scale), w_scale);")\n',
    ),
    (
        '        "acc += (partial * act_scale) * w_scale;",\n        "acc += routing * (partial * act_scale) * w_scale;",\n',
        '        "__fmul_rn(__fmul_rn(partial, act_scale), w_scale);",\n        "__fmul_rn(__fmul_rn(routing * partial, act_scale), w_scale);",\n',
    ),
    (
        "//!     crates/sparkflash-kernels/baseline/kernel_baseline.txt in the cuda-abi CI job) or to a\n",
        "//!     crates/sparkflash-kernels/baseline/kernel_baseline.txt in the nightly cuda-baseline job) or to a\n",
    ),
    (
        "//!      cuda-abi CI job asserts the probe PTX exists, and scripts/check-kernel-baseline.sh recompiles\n",
        "//!      cuda-tests CI job asserts the probe PTX exists, and scripts/check-kernel-baseline.sh recompiles\n",
    ),
    (
        "//!     to the six kernels' cubin byte count, sha256 or PTX `membar` count fails the `cuda-abi` CI\n",
        "//!     to the six kernels' cubin byte count, sha256 or PTX `membar` count fails the nightly `cuda-baseline` CI\n",
    ),
    (
        "//! runs the same script when nvcc is available and says it skipped when it is not, and the cuda-abi\n",
        "//! runs the same script when nvcc is available and says it skipped when it is not, and the nightly cuda-baseline\n",
    ),
    (
        "//! available and reports a skip when it is not, and the `cuda-abi` CI job runs it as a hard gate.\n",
        "//! available and reports a skip when it is not, and the nightly `cuda-baseline` CI job runs it as a hard gate.\n",
    ),
    (
        "/// edit from quietly deleting an assertion that the CI cuda-abi job exists to enforce.\n",
        "/// edit from quietly deleting an assertion that the nightly cuda-baseline job exists to enforce.\n",
    ),
]

POST_SPLIT_GUARD_EDITS = [
    (
        '''    // The hand-rolled acquire that used to sit here is gone: with the helper in place a fence at
    // the call site would be a second, independently editable copy of the guarantee.
    let observing = MEGA_MHC.find(&call).expect("the call exists");
    assert!(
        !MEGA_MHC[observing..].contains("__threadfence"),
        "the acquire must not be duplicated at the call site"
    );
    // The read the acquire protects still happens after the call, and the call is what puts the
    // fence before it.
    let read = MEGA_MHC[observing..]
        .find("*hc_rms_out")
        .map(|offset| observing + offset)
        .expect("the observing path reads hc_rms_out after the poll");
    assert!(observing < read, "the helper call must precede the read");
''',
        '''    // The hand-rolled acquire is gone; the file carries scalar and K8 paths, so check each call.
    let mut calls = MEGA_MHC.match_indices(&call).peekable();
    assert!(calls.peek().is_some(), "the call exists");
    for (observing, _) in calls {
        let read = MEGA_MHC[observing..]
            .find("*hc_rms_out")
            .map(|offset| observing + offset);
        let read = read.expect("the observing path reads hc_rms_out after the poll");
        assert!(observing < read, "the helper call must precede the read");
        assert!(
            !MEGA_MHC[observing..read].contains("__threadfence"),
            "the acquire must not be duplicated at the call site"
        );
    }
''',
    ),
    (
        "/// feature sweep with its own tests, and the six-cubin baseline gate of the `cuda-abi` job, each of\n",
        "/// feature sweep with its own tests, and the six-cubin baseline gate of the `cuda-baseline` job, each of\n",
    ),
    (
        '        "the early return is what keeps the hint from being appended twice"\n    );\n    // A hint that is not last is a message bug, and in a debug build the constructor says so\n    // instead of shipping a rejection whose hint is buried.\n    let refused = std::panic::catch_unwind(|| {\n        rejection(format!("{CANONICAL_HINT} and then some more message"))\n    });\n    assert!(\n        refused.is_err(),\n        "a message that carries the hint anywhere but at its end must be refused"\n    );\n}\n\n/// Round 13 A-1 found a rejection built without the hint, and round 13 answered with a source\n',
        '        "the early return is what keeps the hint from being appended twice"\n    );\n    // A hint that is not last is a message bug, and in a debug build the constructor says so\n    // instead of shipping a rejection whose hint is buried. Release builds do not carry the\n    // debug assertion, so this half of the constructor\'s contract is only observable there.\n    #[cfg(debug_assertions)]\n    {\n        let refused = std::panic::catch_unwind(|| {\n            rejection(format!("{CANONICAL_HINT} and then some more message"))\n        });\n        assert!(\n            refused.is_err(),\n            "a message that carries the hint anywhere but at its end must be refused"\n        );\n    }\n}\n\n/// Round 13 A-1 found a rejection built without the hint, and round 13 answered with a source\n',
    ),
]


POST_FORMAT_ROOT_EDITS = [
    (
        'pub const MEGA_MHC: &str = include_str!("../../../kernels/mega_mhc_sm121.cu");\n/// The MXFP8 dense twin of specification 4.1.\npub const MXFP8_DENSE: &str = include_str!("../../../kernels/b12x_mxfp8_dense_sm121.cu");\n/// The MXFP4 routed-expert MoE combine of specification 2.1.\npub const MXFP4_MOE: &str = include_str!("../../../kernels/b12x_mxfp4_moe_sm121.cu");\n/// FlashMLA with CSA2 candidate selection of specification 2.3.\npub const FLASH_MLA_CSA2: &str = include_str!("../../../kernels/flash_mla_csa2_sm121.cu");\n/// The grouped wo_a output projection, MXFP8 with the same B12x layout.\npub const MXFP8_WO_A: &str = include_str!("../../../kernels/b12x_mxfp8_wo_a_sm121.cu");\n/// The certified LM head of specification 2.1: a bound screen plus exact dots.\n',
        'pub const MEGA_MHC: &str = include_str!("../../../kernels/mega_mhc_sm121.cu");\n/// The MXFP8 dense twin of specification 4.1.\npub const MXFP8_DENSE: &str = include_str!("../../../kernels/b12x_mxfp8_dense_sm121.cu");\n/// The dense and wo_a bodies the two b12x entries share; a source-text guard reads them together.\npub const B12X_DENSE_HEADER: &str = include_str!("../../../kernels/b12x_dense_sm121.cuh");\n/// The MXFP4 routed-expert MoE combine of specification 2.1.\npub const MXFP4_MOE: &str = include_str!("../../../kernels/b12x_mxfp4_moe_sm121.cu");\n/// FlashMLA with CSA2 candidate selection of specification 2.3.\npub const FLASH_MLA_CSA2: &str = include_str!("../../../kernels/flash_mla_csa2_sm121.cu");\n\n/// The body the three CSA2 entries share: the arithmetic text a source-text guard may read.\npub const FLASH_MLA_CSA2_COMMON: &str =\n    include_str!("../../../kernels/flash_mla_csa2_common_sm121.cuh");\n/// The grouped wo_a output projection, MXFP8 with the same B12x layout.\npub const MXFP8_WO_A: &str = include_str!("../../../kernels/b12x_mxfp8_wo_a_sm121.cu");\n/// The certified LM head of specification 2.1: a bound screen plus exact dots.\n',
    ),
    (
        '            .find(|(file, _)| *file == name)\n            .map(|(_, source)| *source)\n            .unwrap_or_else(|| panic!("{name} is not in the kernel list"));\n        let enter = format!("SPARK_GPU_TRACE_ENTER(trace_ring, {scope},");\n        assert!(\n            source.contains(&enter),\n',
        '            .find(|(file, _)| *file == name)\n            .map(|(_, source)| *source)\n            .unwrap_or_else(|| panic!("{name} is not in the kernel list"));\n        // The dense, wo_a and CSA2 bodies live in shared headers; the entry files carry the\n        // symbol and the entry-specific defines.\n        let combined;\n        let source = if name.starts_with("b12x_mxfp8_") {\n            combined = format!("{source}\\n{B12X_DENSE_HEADER}");\n            combined.as_str()\n        } else if name.starts_with("flash_mla_csa2_") {\n            combined = format!("{source}\\n{FLASH_MLA_CSA2_COMMON}");\n            combined.as_str()\n        } else {\n            source\n        };\n        let enter = format!("SPARK_GPU_TRACE_ENTER(trace_ring, {scope},");\n        assert!(\n            source.contains(&enter),\n',
    ),
    (
        '///\n#[test]\nfn the_reviewed_invariants_are_pinned_in_the_kernel_sources() {\n    // The round-1 review found five defects that no test could see. Each of them is now a property\n    // of the source text, and each property is paired with the mutation that removes it, so a\n    // future edit that reintroduces the defect fails here as well as in the reference tests.\n',
        '///\n#[test]\nfn the_reviewed_invariants_are_pinned_in_the_kernel_sources() {\n    // The three entries share one attention body; the source guards read the entry\'s defines and\n    // the shared arithmetic together.\n    let mla_source = format!("{FLASH_MLA_CSA2}\\n{FLASH_MLA_CSA2_COMMON}");\n    let mla: &str = &mla_source;\n\n    // The round-1 review found five defects that no test could see. Each of them is now a property\n    // of the source text, and each property is paired with the mutation that removes it, so a\n    // future edit that reintroduces the defect fails here as well as in the reference tests.\n',
    ),
    (
        '            && code.contains("w_fp8[group_weight_base +")\n            && code.contains("w_scales[group_scale_base +")\n    };\n    assert!(wo_a_has_two_strides(MXFP8_WO_A));\n    let shared_stride = MXFP8_WO_A.replace(\n        "(uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK",\n        "(uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK",\n    );\n    assert_ne!(shared_stride, MXFP8_WO_A, "the mutation must apply");\n    assert!(\n        !wo_a_has_two_strides(&shared_stride),\n        "the weight stride must not be reused for the scales"\n',
        '            && code.contains("w_fp8[group_weight_base +")\n            && code.contains("w_scales[group_scale_base +")\n    };\n    let wo_a = format!("{MXFP8_WO_A}\\n{B12X_DENSE_HEADER}");\n    assert!(wo_a_has_two_strides(&wo_a));\n    let shared_stride = wo_a.replace(\n        "(uint64_t)group * spark_b12x_scale_bytes(SPARK_M4_O_LORA_RANK",\n        "(uint64_t)group * spark_b12x_buffer_bytes(SPARK_M4_O_LORA_RANK",\n    );\n    assert_ne!(shared_stride, wo_a, "the mutation must apply");\n    assert!(\n        !wo_a_has_two_strides(&shared_stride),\n        "the weight stride must not be reused for the scales"\n',
    ),
    (
        '            && code.matches("score *= SPARK_M4_MLA_SCORE_SCALE;").count() == 3\n            && code.matches("SPARK_M4_MLA_SCORE_SCALE").count() == 3\n    };\n    assert!(mla_scales_the_score(FLASH_MLA_CSA2));\n    let unscaled = FLASH_MLA_CSA2.replace("score *= SPARK_M4_MLA_SCORE_SCALE;", "score *= 1.0f;");\n    assert_ne!(unscaled, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_scales_the_score(&unscaled),\n        "an unscaled score is the defect the reference used to bless"\n    );\n\n    // A3: the maximum is a per-thread value reduced as a tree, not a shared read-modify-write.\n    let mla_reduces_the_max = |source: &str| -> bool {\n        let code = crate::c_decl::strip_comments(source);\n        code.contains("local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS)")\n            && code.contains("if (lane == 0u) {\\n        s_max = local_max;")\n            && !code.contains("s_max = s_scores[slot] > s_max")\n    };\n    assert!(mla_reduces_the_max(FLASH_MLA_CSA2));\n    let racy_max = FLASH_MLA_CSA2.replace(\n        "    local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);\\n    if (lane == 0u) {\\n        s_max = local_max;\\n    }",\n        "    s_max = s_scores[slot] > s_max ? s_scores[slot] : s_max;",\n    );\n    assert_ne!(racy_max, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_reduces_the_max(&racy_max),\n        "the racy shared accumulation must be rejected"\n',
        '            && code.matches("score *= SPARK_M4_MLA_SCORE_SCALE;").count() == 3\n            && code.matches("SPARK_M4_MLA_SCORE_SCALE").count() == 3\n    };\n    assert!(mla_scales_the_score(mla));\n    let unscaled = mla.replace("score *= SPARK_M4_MLA_SCORE_SCALE;", "score *= 1.0f;");\n    assert_ne!(unscaled, mla, "the mutation must apply");\n    assert!(\n        !mla_scales_the_score(&unscaled),\n        "an unscaled score is the defect the reference used to bless"\n    );\n\n    // A3: the maximum is a per-thread value reduced as a tree, not a shared read-modify-write.\n    let mla_reduces_the_max = |source: &str| -> bool {\n        let code = crate::c_decl::strip_comments(source);\n        let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");\n        flat.contains("local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS)")\n            && flat.contains("if (lane == 0u) { s_max = local_max;")\n            && !flat.contains("s_max = s_scores[slot] > s_max")\n    };\n    assert!(mla_reduces_the_max(mla));\n    let racy_max = mla.replace(\n        "local_max = spark_block_max(local_max, lane, SPARK_M4_MLA_THREADS);",\n        "s_max = s_scores[slot] > s_max ? s_scores[slot] : s_max;",\n    );\n    assert_ne!(racy_max, mla, "the mutation must apply");\n    assert!(\n        !mla_reduces_the_max(&racy_max),\n        "the racy shared accumulation must be rejected"\n',
    ),
    (
        '            "const uint32_t index_head = head / (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;",\n        ) && code.contains("index_query + (uint64_t)index_head * SPARK_M4_INDEX_HEAD_DIM")\n    };\n    assert!(mla_uses_the_index_head(FLASH_MLA_CSA2));\n    let query_head = FLASH_MLA_CSA2.replace(\n        "const uint32_t index_head = head / (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;",\n        "const uint32_t index_head = head;",\n    );\n    assert_ne!(query_head, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_uses_the_index_head(&query_head),\n        "the query head must not index the indexer query buffer"\n',
        '            "const uint32_t index_head = head / (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;",\n        ) && code.contains("index_query + (uint64_t)index_head * SPARK_M4_INDEX_HEAD_DIM")\n    };\n    assert!(mla_uses_the_index_head(mla));\n    let query_head = mla.replace(\n        "const uint32_t index_head = head / (uint32_t)SPARK_M4_MLA_HEADS_PER_INDEX_HEAD;",\n        "const uint32_t index_head = head;",\n    );\n    assert_ne!(query_head, mla, "the mutation must apply");\n    assert!(\n        !mla_uses_the_index_head(&query_head),\n        "the query head must not index the indexer query buffer"\n',
    ),
    (
        '    // must not disturb. Each mutation below asserts it applied and that this helper then fails.\n    let mla_guards_the_window_format = |source: &str| -> bool {\n        let code = crate::c_decl::strip_comments(source);\n        code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4 0u")\n            && code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8 1u")\n            && code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u")\n            && code.contains("if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {")\n            && code.contains("__float2bfloat16_rn(__uint_as_float(0x7FC00000u))")\n            && !code.contains("static_assert(window_kv_format")\n            // The formats 0/1 dispatch text the union branch does not enter: the attended\n            // predicate and the two ternary load sites.\n            && code.contains("? spark_mla_in_window(position, candidate)")\n            && code.contains(": (candidate < window_rows ? 1u : 0u);")\n            && code.contains(\n                "const float key = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",\n            )\n            && code.contains(\n                "const float value = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",\n            )\n            // The union mode\'s own dispatch: the absolute-position causal predicate, every supplied\n',
        '    // must not disturb. Each mutation below asserts it applied and that this helper then fails.\n    let mla_guards_the_window_format = |source: &str| -> bool {\n        let code = crate::c_decl::strip_comments(source);\n        let flat = code.split_whitespace().collect::<Vec<_>>().join(" ");\n        code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4 0u")\n            && code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8 1u")\n            && code.contains("#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u")\n            && code.contains("refuse = window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION;")\n            && code.contains("__float2bfloat16_rn(__uint_as_float(0x7FC00000u))")\n            && !code.contains("static_assert(window_kv_format")\n            // The formats 0/1 dispatch text the union branch does not enter: the attended\n            // predicate and the two ternary load sites.\n            && code.contains("? spark_mla_in_window(position, candidate)")\n            && code.contains(": (candidate < window_rows ? 1u : 0u);")\n            && flat.contains(\n                "const float key = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",\n            )\n            && flat.contains(\n                "const float value = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",\n            )\n            // The union mode\'s own dispatch: the absolute-position causal predicate, every supplied\n',
    ),
    (
        '                "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)",\n            )\n            && code.contains("static_assert(SPARK_M4_MLA_MAX_ATTENDED == 640u,")\n            && code.contains("__shared__ float s_scores[SPARK_M4_MLA_MAX_ATTENDED];")\n            && code.contains("__device__ __forceinline__ uint32_t spark_mla_union_attends(")\n            && code.contains("return candidate <= position ? 1u : 0u;")\n            && code.contains("if (window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {")\n            && code.contains("spark_mla_union_attends(position, candidate) != 0u")\n            && code.contains("spark_mla_load_window_stored(window_codes, window_scales, slot, i)")\n            && code.contains("window_rows > (uint32_t)SPARK_M4_SWA_WINDOW")\n            && code.contains("selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED")\n            && code.contains("window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED")\n            // The union branch must END the kernel: deleting its terminal return would fall\n            // through into historical Stage 2 with a format-2 result already written. Pin the\n            // 12-space union trace exit and the exact tail sequence, not just the branch header;\n            // the historical Stage-2 exit has different indentation and must not satisfy these.\n            && code.contains("            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);")\n            && code.contains(\n                "            (void)heads;\\n            (void)step_seq;\\n        }\\n        (void)t_enter;\\n        return;\\n    }",\n            )\n    };\n    assert!(mla_guards_the_window_format(FLASH_MLA_CSA2));\n\n    // The sentinel selector the design rejects: window_rows == 0 cannot express an empty engine\n    // window, and it reads badly in the ABI pin because the row count is not a format.\n    let sentinel_format = FLASH_MLA_CSA2.replace(\n        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {",\n        "if (window_rows == 0u) {",\n    );\n    assert_ne!(sentinel_format, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&sentinel_format),\n        "a row-count sentinel is exactly the format selector this design rejects"\n    );\n\n    // The format value itself: renumbering the union mode onto the engine value must fail the pin.\n    let wrong_union_value = FLASH_MLA_CSA2.replace(\n        "#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u",\n        "#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 1u",\n    );\n    assert_ne!(wrong_union_value, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&wrong_union_value),\n        "the union format must keep its own value, not alias engine format 1"\n    );\n\n    // Accepting 3 through a too-wide guard, or falling back to the old two-value guard, is the\n    // silent-wrong-arithmetic defect: both must fail.\n    let wide_guard = FLASH_MLA_CSA2.replace(\n        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {",\n        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION + 1u) {",\n    );\n    assert_ne!(wide_guard, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&wide_guard),\n        "a fourth format must not pass the entry guard"\n    );\n    let two_value_guard = FLASH_MLA_CSA2.replace(\n        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION) {",\n        "if (window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8) {",\n    );\n    assert_ne!(two_value_guard, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&two_value_guard),\n        "the old two-value guard would hand format 2 to the format-1 branch"\n    );\n\n    // A refusal that is not visible is worse than no refusal: the poison must stay the NaN output.\n    let no_poison = FLASH_MLA_CSA2.replace("0x7FC00000u", "0x0u");\n    assert_ne!(no_poison, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_poison),\n        "the refused formats must still poison with NaN"\n    );\n\n    // The negative pin: a runtime format cannot be a static_assert even when the guard remains.\n    let asserted_format =\n        format!("{FLASH_MLA_CSA2}\\nstatic_assert(window_kv_format > 2u, \\"format\\");\\n");\n    assert_ne!(asserted_format, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&asserted_format),\n        "a runtime format value cannot be a static_assert"\n    );\n\n    // The formats 0/1 dispatch text: both ternary load sites are pinned, so deleting either one\n    // fails even while the other (and the union branch) remains.\n    let no_key_site = FLASH_MLA_CSA2.replace(\n        "const float key = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",\n        "const float key = 1u",\n    );\n    assert_ne!(no_key_site, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_key_site),\n        "the format-0/1 key load dispatch must stay"\n    );\n    let no_value_site = FLASH_MLA_CSA2.replace(\n        "const float value = window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",\n        "const float value = 1u",\n    );\n    assert_ne!(no_value_site, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_value_site),\n        "the format-0/1 value load dispatch must stay"\n    );\n\n    // The union rule is a causal predicate, not a window: changing it to the window predicate or\n    // to an exclusive comparison would silently drop out-of-window candidates (the Criterion-1\n    // gap) or the query\'s own position.\n    let union_windowed = FLASH_MLA_CSA2.replace(\n        "return candidate <= position ? 1u : 0u;",\n        "return candidate <= position && spark_mla_in_window(position, candidate) ? 1u : 0u;",\n    );\n    assert_ne!(union_windowed, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&union_windowed),\n        "an out-of-window selected id must stay attendable in union mode"\n    );\n    let union_exclusive = FLASH_MLA_CSA2.replace(\n        "return candidate <= position ? 1u : 0u;",\n        "return candidate < position ? 1u : 0u;",\n    );\n    assert_ne!(union_exclusive, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&union_exclusive),\n        "an id at the query\'s own position is attendable"\n    );\n\n    // Capacity: exactly 640 is the union of a full candidate list and a full window; a narrower\n    // expression (or no sum guard) would write past s_scores.\n    let shrunk_capacity = FLASH_MLA_CSA2.replace(\n        "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)",\n        "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + 64u)",\n    );\n    assert_ne!(shrunk_capacity, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&shrunk_capacity),\n        "the union capacity expression is one candidate list plus one window"\n    );\n    let no_capacity_guard = FLASH_MLA_CSA2.replace(\n        "window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED",\n        "window_rows + selected_count > (uint32_t)SPARK_M4_MLA_LATENT",\n    );\n    assert_ne!(no_capacity_guard, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_capacity_guard),\n        "a union longer than the score array would be an out-of-bounds write"\n    );\n    let no_selected_cap = FLASH_MLA_CSA2.replace(\n        "selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED",\n        "selected_count > 1024u",\n    );\n    assert_ne!(no_selected_cap, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_selected_cap),\n        "the selected list is index_topk long"\n    );\n    let no_window_cap = FLASH_MLA_CSA2.replace(\n        "window_rows > (uint32_t)SPARK_M4_SWA_WINDOW",\n        "window_rows > 256u",\n    );\n    assert_ne!(no_window_cap, FLASH_MLA_CSA2, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_window_cap),\n        "the union window is at most the sliding window"\n',
        '                "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)",\n            )\n            && code.contains("static_assert(SPARK_M4_MLA_MAX_ATTENDED == 640u,")\n            && code.contains("__shared__ float s_scores[kMaxAttended];")\n            && code.contains("__device__ __forceinline__ uint32_t spark_mla_union_attends(")\n            && code.contains("return candidate <= position ? 1u : 0u;")\n            && flat.contains("window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION")\n            && code.contains("spark_mla_union_attends(position, candidate) != 0u")\n            && code.contains("spark_ue8m0_decode(row_scales[blk])")\n            && code.contains("row_codes[i0]")\n            && code.contains("window_rows > (uint32_t)SPARK_M4_SWA_WINDOW")\n            && code.contains("selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED")\n            && code.contains("window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED")\n            // The union branch must END the kernel: deleting its terminal return would fall\n            // through into historical Stage 2 with a format-2 result already written. The tail is\n            // checked on whitespace-normalized text so a re-indent does not hide a deletion.\n            && flat.contains("SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);")\n            && flat.contains("(void)heads; (void)step_seq; } (void)t_enter; return; }")\n    };\n    assert!(mla_guards_the_window_format(mla));\n\n    // The sentinel selector the design rejects: window_rows == 0 cannot express an empty engine\n    // window, and it reads badly in the ABI pin because the row count is not a format.\n    let sentinel_format = mla.replace(\n        "refuse = window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION;",\n        "refuse = window_rows == 0u;",\n    );\n    assert_ne!(sentinel_format, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&sentinel_format),\n        "a row-count sentinel is exactly the format selector this design rejects"\n    );\n\n    // The format value itself: renumbering the union mode onto the engine value must fail the pin.\n    let wrong_union_value = mla.replace(\n        "#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 2u",\n        "#define SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION 1u",\n    );\n    assert_ne!(wrong_union_value, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&wrong_union_value),\n        "the union format must keep its own value, not alias engine format 1"\n    );\n\n    // Accepting 3 through a too-wide guard, or falling back to the old two-value guard, is the\n    // silent-wrong-arithmetic defect: both must fail.\n    let wide_guard = mla.replace(\n        "refuse = window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION;",\n        "refuse = window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION + 1u;",\n    );\n    assert_ne!(wide_guard, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&wide_guard),\n        "a fourth format must not pass the entry guard"\n    );\n    let two_value_guard = mla.replace(\n        "refuse = window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_CSA_UNION;",\n        "refuse = window_kv_format > SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8;",\n    );\n    assert_ne!(two_value_guard, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&two_value_guard),\n        "the old two-value guard would hand format 2 to the format-1 branch"\n    );\n\n    // A refusal that is not visible is worse than no refusal: the poison must stay the NaN output.\n    let no_poison = mla.replace("0x7FC00000u", "0x0u");\n    assert_ne!(no_poison, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_poison),\n        "the refused formats must still poison with NaN"\n    );\n\n    // The negative pin: a runtime format cannot be a static_assert even when the guard remains.\n    let asserted_format = format!("{mla}\\nstatic_assert(window_kv_format > 2u, \\"format\\");\\n");\n    assert_ne!(asserted_format, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&asserted_format),\n        "a runtime format value cannot be a static_assert"\n    );\n\n    // The formats 0/1 dispatch text: both ternary load sites are pinned, so deleting either one\n    // fails even while the other (and the union branch) remains.\n    let no_key_site = mla.replace(\n        "window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",\n        "window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8",\n    );\n    assert_ne!(no_key_site, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_key_site),\n        "the format-0/1 key load dispatch must stay"\n    );\n    let no_value_site = mla.replace(\n        "window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_FP4",\n        "window_kv_format == SPARK_M4_MLA_WINDOW_KV_FORMAT_ENGINE_FP8",\n    );\n    assert_ne!(no_value_site, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_value_site),\n        "the format-0/1 value load dispatch must stay"\n    );\n\n    // The union rule is a causal predicate, not a window: changing it to the window predicate or\n    // to an exclusive comparison would silently drop out-of-window candidates (the Criterion-1\n    // gap) or the query\'s own position.\n    let union_windowed = mla.replace(\n        "return candidate <= position ? 1u : 0u;",\n        "return candidate <= position && spark_mla_in_window(position, candidate) ? 1u : 0u;",\n    );\n    assert_ne!(union_windowed, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&union_windowed),\n        "an out-of-window selected id must stay attendable in union mode"\n    );\n    let union_exclusive = mla.replace(\n        "return candidate <= position ? 1u : 0u;",\n        "return candidate < position ? 1u : 0u;",\n    );\n    assert_ne!(union_exclusive, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&union_exclusive),\n        "an id at the query\'s own position is attendable"\n    );\n\n    // Capacity: exactly 640 is the union of a full candidate list and a full window; a narrower\n    // expression (or no sum guard) would write past s_scores.\n    let shrunk_capacity = mla.replace(\n        "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + SPARK_M4_SWA_WINDOW)",\n        "#define SPARK_M4_MLA_MAX_ATTENDED (SPARK_M4_MLA_MAX_SELECTED + 64u)",\n    );\n    assert_ne!(shrunk_capacity, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&shrunk_capacity),\n        "the union capacity expression is one candidate list plus one window"\n    );\n    let no_capacity_guard = mla.replace(\n        "window_rows + selected_count > (uint32_t)SPARK_M4_MLA_MAX_ATTENDED",\n        "window_rows + selected_count > (uint32_t)SPARK_M4_MLA_LATENT",\n    );\n    assert_ne!(no_capacity_guard, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_capacity_guard),\n        "a union longer than the score array would be an out-of-bounds write"\n    );\n    let no_selected_cap = mla.replace(\n        "selected_count > (uint32_t)SPARK_M4_MLA_MAX_SELECTED",\n        "selected_count > 1024u",\n    );\n    assert_ne!(no_selected_cap, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_selected_cap),\n        "the selected list is index_topk long"\n    );\n    let no_window_cap = mla.replace(\n        "window_rows > (uint32_t)SPARK_M4_SWA_WINDOW",\n        "window_rows > 256u",\n    );\n    assert_ne!(no_window_cap, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&no_window_cap),\n        "the union window is at most the sliding window"\n',
    ),
    (
        '    // format-2 result already written. The mutation removes the exact tail the pin names, and a\n    // second mutation removes only the union trace exit so the historical Stage-2 exit cannot\n    // satisfy the pin by itself.\n    let union_falls_through = FLASH_MLA_CSA2.replace(\n        "        (void)t_enter;\\n        return;\\n    }\\n\\n    // Stage 2: attention over the selected candidates.",\n        "        (void)t_enter;\\n    }\\n\\n    // Stage 2: attention over the selected candidates.",\n    );\n    assert_ne!(\n        union_falls_through, FLASH_MLA_CSA2,\n        "the mutation must apply"\n    );\n    assert!(\n        !mla_guards_the_window_format(&union_falls_through),\n        "the union branch must return before historical Stage 2"\n    );\n    let union_no_trace_exit = FLASH_MLA_CSA2.replace(\n        "            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);\\n",\n        "",\n    );\n    assert_ne!(\n        union_no_trace_exit, FLASH_MLA_CSA2,\n        "the mutation must apply"\n    );\n    assert!(\n        !mla_guards_the_window_format(&union_no_trace_exit),\n        "the union branch\'s trace exit is part of its terminal sequence"\n',
        '    // format-2 result already written. The mutation removes the exact tail the pin names, and a\n    // second mutation removes only the union trace exit so the historical Stage-2 exit cannot\n    // satisfy the pin by itself.\n    let union_falls_through = mla.replace(\n        "        (void)t_enter;\\n        return;\\n    }",\n        "        (void)t_enter;\\n    }",\n    );\n    assert_ne!(union_falls_through, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&union_falls_through),\n        "the union branch must return before historical Stage 2"\n    );\n    let union_no_trace_exit = mla.replace(\n        "            SPARK_GPU_TRACE_EXIT(trace_ring, TRACE_SCOPE_CSA2_ATTN, layer_id, meta, t_enter);\\n",\n        "",\n    );\n    assert_ne!(union_no_trace_exit, mla, "the mutation must apply");\n    assert!(\n        !mla_guards_the_window_format(&union_no_trace_exit),\n        "the union branch\'s trace exit is part of its terminal sequence"\n',
    ),
    (
        '        "the fp4 per-row grid must not collapse into the fp8 per-tile grid"\n    );\n    assert!(\n        FLASH_MLA_CSA2.contains("SPARK_M4_MLA_HEADS_PER_INDEX_HEAD")\n            && COMMON_HEADER.contains("#define SPARK_M4_MLA_HEADS_PER_INDEX_HEAD"),\n        "the head mapping must be a header constant used by name"\n    );\n',
        '        "the fp4 per-row grid must not collapse into the fp8 per-tile grid"\n    );\n    assert!(\n        mla.contains("SPARK_M4_MLA_HEADS_PER_INDEX_HEAD")\n            && COMMON_HEADER.contains("#define SPARK_M4_MLA_HEADS_PER_INDEX_HEAD"),\n        "the head mapping must be a header constant used by name"\n    );\n',
    ),
]


def apply_post_format_root_edits(text):
    for old, new in POST_FORMAT_ROOT_EDITS:
        if text.count(old) != 1:
            sys.exit("error: post-format root edit target is not unique or missing")
        text = text.replace(old, new, 1)
    return text


def apply_post_split_edits(expected):
    for old, new in POST_SPLIT_ROOT_EDITS:
        if old not in expected["root"]:
            sys.exit("error: post-split root edit target is missing: " + old[:60])
        expected["root"] = expected["root"].replace(old, new, 1)
    return expected


def apply_post_split_guard_edits(formatted_guard):
    # The guard block is applied after rustfmt: the split's raw body is still indented one level
    # (it lived inside the old inline test module), while this replacement matches the formatted
    # file that the repository actually reviews.
    for old, new in POST_SPLIT_GUARD_EDITS:
        if old not in formatted_guard:
            sys.exit("error: post-split acquire-guard edit target is missing")
        formatted_guard = formatted_guard.replace(old, new, 1)
    return formatted_guard


GLUE = "mod scanner;\npub use scanner::{\n    acquire_fence_after_poll, acquire_helper_fences_unconditionally,\n    guarded_names_are_not_redefined, observing_paths_call_the_acquiring_poll,\n    the_acquiring_poll_is_defined_exactly_once,\n};\n\n#[cfg(test)]\nuse scanner::*;\n\n#[cfg(test)]\nmod acquire_guard;\n\n"
ADAPTER_READ = (
    "    let source =\n"
    '        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/m4_abi/scanner.rs"))\n'
    '            .expect("the scanner\'s own source is readable");\n'
    "    // Round-15 A-5: the region split is a text search, and this file is checked out on platforms\n"
    "    // whose line endings are CRLF, so the newlines are normalised first.\n"
    '    let source = source.replace("\\r\\n", "\\n");\n'
    "    // The guard's own code: from the hint constant that every rejection ends with to the end of the\n"
    "    // scanner module. The scanner module holds the guard's code and nothing else after the hint.\n"
    "    let code = source\n"
    '        .split_once("const CANONICAL_HINT: &str")\n'
    '        .expect("the guard defines the canonical hint")\n'
    "        .1;\n"
)
ADAPTER_SOURCE = (
    "const THIS_SOURCE: &str = concat!(\n"
    '    include_str!("../m4_abi.rs"),\n'
    '    include_str!("scanner.rs"),\n'
    '    include_str!("acquire_guard.rs"),\n'
    ");\n"
)
ADAPTER_DOC = (
    "/// The whole " + chr(96) + "m4_abi" + chr(96) + " module family - this module, its scanner and the root file - as the text\n"
    "/// the documentation checks below read: the module's own documentation is part of the crate, so\n"
    "/// the shape clippy refuses can be checked here.\n"
)


def base_text():
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", BASE_COMMIT + ":" + BASE_PATH],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit("error: cannot read " + BASE_PATH + " at " + BASE_COMMIT + ": " + result.stderr.strip())
    # Two comments have since had the build machine's home directory replaced with /home/user.
    return re.sub(r"/home/[^/\s]+/wt/", "/home/user/wt/", result.stdout)


def slice_base(base):
    lines = base.splitlines(keepends=True)
    scanner_start = next(i for i, line in enumerate(lines) if line.startswith(SCANNER_START))
    guard_decl = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == "#[cfg(test)]" and i + 1 < len(lines) and lines[i + 1].startswith("mod acquire_guard {")
    )
    scanner = "".join(lines[scanner_start : guard_decl - 1])
    root_pre = "".join(lines[:scanner_start])
    guard_body = "".join(lines[guard_decl + 2 : len(lines) - 1])
    assert lines[guard_decl - 1].strip() == "", "the scanner slice must end on its own line"
    assert lines[-1].strip() == "}", "the module must end on its closing brace"
    return root_pre, scanner, guard_body


def split_expected(base):
    root_pre, scanner, guard_body = slice_base(base)
    # The root: freeze-notice wording, then the module declarations, then everything else verbatim.
    root = root_pre.replace(FREEZE_TITLE, FREEZE_TITLE_NEW, 1).replace(FREEZE_BULLET, FREEZE_BULLET_NEW, 1)
    assert FREEZE_LENGTH in root
    root = root.replace(FREEZE_LENGTH, FREEZE_LENGTH_NEW, 1)
    anchor = "use crate::c_decl::Consts;\n\n"
    index = root.index(anchor) + len(anchor)
    root = root[:index] + GLUE + root[index:]

    # The scanner: the base text verbatim, with the cross-module visibility the moved items need.
    scanner_lines = []
    for line in scanner.splitlines(keepends=True):
        if re.match(r"^(fn |struct |enum |const |static |type )", line):
            line = "pub(crate) " + line
        scanner_lines.append(line)
    scanner_out = "".join(scanner_lines)

    # The acquire-guard tests: the base body verbatim, with the three self-scan adapters below.
    guard = guard_body
    replacement = ADAPTER_READ
    start = guard.index("    let source = std::fs::read_to_string")
    end = guard.index("        .0;", start) + len("        .0;") + 1
    guard = guard[:start] + replacement + guard[end:]
    assert '    let source = THIS_SOURCE;' not in guard
    guard = guard.replace(
        '        let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/m4_abi.rs"));',
        "    let source = THIS_SOURCE;",
        1,
    )
    guard = guard.replace(
        '    const THIS_SOURCE: &str = include_str!("m4_abi.rs");\n',
        ADAPTER_SOURCE,
        1,
    )
    guard = guard.replace(
        "    /// This file, as the text the documentation check below reads: the module's own documentation is\n"
        "    /// part of the crate, so the shape clippy refuses can be checked here.\n",
        ADAPTER_DOC,
        1,
    )
    return {"root": root, "scanner": scanner_out, "acquire_guard": guard}


def format_rust(expected, name):
    # Write the whole regenerated family first, so rustfmt can resolve the root file's mod
    # declarations; then format each file exactly as cargo fmt would.
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "m4_abi").mkdir(exist_ok=True)
    (TMP / "m4_abi.rs").write_text(expected["root"])
    (TMP / "m4_abi" / "scanner.rs").write_text(expected["scanner"])
    (TMP / "m4_abi" / "acquire_guard.rs").write_text(expected["acquire_guard"])
    path = TMP / name
    result = subprocess.run(
        ["rustfmt", "--edition", "2024", str(path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit("error: rustfmt refused the regenerated " + name + ":\n" + result.stderr.strip())
    return path.read_text()


LICENSE_HEADER_END = "// limitations under the License.\n\n"


def strip_license_header(text):
    """Drop the Apache-2.0 notice the relicense added above every source file."""
    if not text.startswith("// Copyright (c) 2026 Hunter L. Allen"):
        return text
    end = text.find(LICENSE_HEADER_END)
    if end < 0:
        sys.exit("error: a source file starts with the license header but does not end it")
    return text[end + len(LICENSE_HEADER_END):]


def compare(expected, actual_path, label):
    actual = strip_license_header(actual_path.read_text())
    if expected == actual:
        print("  " + label + ": byte-identical to the regeneration")
        return True
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for number, (left, right) in enumerate(zip(expected_lines, actual_lines), start=1):
        if left != right:
            print("  " + label + ": first difference at line " + str(number))
            print("    expected: " + left)
            print("    actual:   " + right)
            return False
    print("  " + label + ": line counts differ (" + str(len(expected_lines)) + " vs " + str(len(actual_lines)) + ")")
    return False


def check_family_and_pin():
    actual = sorted(p for p in (SRC / "m4_abi").glob("*.rs"))
    expected = sorted([SCANNER_FILE, GUARD_FILE])
    if actual != expected:
        sys.exit("error: src/m4_abi/ holds " + str([p.name for p in actual]) + ", expected scanner.rs and acquire_guard.rs")
    family = [ROOT_FILE, SCANNER_FILE, GUARD_FILE]
    total = sum(len(path.read_text().splitlines()) for path in family)
    pin = re.search(r"^PINNED_LINES=(\d+)$", CHECK_SCRIPT.read_text(), re.M)
    if not pin:
        sys.exit("error: ci/check-m4-abi-size.sh has no PINNED_LINES line")
    if int(pin.group(1)) != total:
        sys.exit("error: the family is " + str(total) + " lines but ci/check-m4-abi-size.sh pins " + pin.group(1))
    tests = sum(len(re.findall(r"^\s*#\[test\]$", path.read_text(), re.M)) for path in family)
    if tests != 64:
        sys.exit("error: the family has " + str(tests) + " #[test] items, expected the pre-split 64")
    return total, tests


def main():
    global BASE_COMMIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE_COMMIT, help="base commit the move is checked against")
    args = parser.parse_args()
    BASE_COMMIT = args.base
    base = base_text()
    total, tests = check_family_and_pin()
    expected = apply_post_split_edits(split_expected(base))
    expected["root"] = format_rust(expected, "m4_abi.rs")
    expected["root"] = apply_post_format_root_edits(expected["root"])
    expected["scanner"] = format_rust(expected, "m4_abi/scanner.rs")
    expected["acquire_guard"] = apply_post_split_guard_edits(
        format_rust(expected, "m4_abi/acquire_guard.rs")
    )
    ok = True
    ok &= compare(expected["root"], ROOT_FILE, "root")
    ok &= compare(expected["scanner"], SCANNER_FILE, "scanner")
    ok &= compare(expected["acquire_guard"], GUARD_FILE, "acquire_guard")
    if not ok:
        sys.stderr.write("error: the working tree is not the regeneration of " + BASE_COMMIT + ":" + BASE_PATH + "\n")
        return 1
    print("m4_abi split verified against " + BASE_COMMIT + ":" + BASE_PATH)
    print("  family: " + str(total) + " lines, pin matches; " + str(tests) + " tests (the pre-split list is unchanged)")
    print("  the three files are exactly the base text cut at its slice boundaries, plus the glue")
    print("  and the allow-listed self-scan adapters, formatted by the repo's rustfmt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
