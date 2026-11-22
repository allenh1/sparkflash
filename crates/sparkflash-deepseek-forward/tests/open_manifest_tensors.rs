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

//! The open manifest's tensor records, built from a header-only resident plan.

use std::collections::BTreeSet;

use sparkflash_deepseek_forward::open_manifest::record_tensors;
use sparkflash_device::resident::{HeaderEntry, ResidentPlan};

#[test]
fn a_bound_tensor_records_its_stored_shape_format_and_rank_slice() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("manifest.jsonl");
    // SAFETY: this test binary has one test, and the manifest is initialized on this thread after
    // the variable is set. No other test reads or writes the process environment.
    unsafe {
        std::env::set_var(sparkflash_kernels::open_manifest::ENV, &path);
    }
    let name = "layers.0.attn.wq_b.weight";
    let plan = ResidentPlan::plan(1, 4, [HeaderEntry::new(name, "F8_E4M3", vec![32768, 1280])])
        .expect("a four-rank plan");
    let bound: BTreeSet<String> = [name.to_string()].into_iter().collect();
    record_tensors(&plan, &bound, 1, 4);
    let text = std::fs::read_to_string(&path).expect("the manifest file");
    assert!(text.contains("\"kind\":\"tensor\""), "{text}");
    assert!(
        text.contains("\"name\":\"layers.0.attn.wq_b.weight\""),
        "{text}"
    );
    assert!(text.contains("\"shape\":[32768,1280]"), "{text}");
    assert!(text.contains("\"format\":\"F8_E4M3\""), "{text}");
    assert!(text.contains("\"rank_slice\":[8192,1280]"), "{text}");
    assert!(text.contains("\"rank\":1"), "{text}");
    assert!(text.contains("\"world\":4"), "{text}");
}
