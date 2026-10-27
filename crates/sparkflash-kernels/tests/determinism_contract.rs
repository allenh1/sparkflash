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

//! The determinism contract (DECISIONS.md D21), checked instead of described.
//!
//! D21 records three statements - D-a kernel determinism, D-b reduction determinism, D-c no assumed
//! cross-boundary equality - and the M7 harness cites them from its printed output. A decision that
//! only lives in a document is a description; this test fails when either half disappears, so the
//! contract cannot rot into prose.

#[test]
fn the_determinism_contract_is_written_down_and_the_harness_cites_it() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let decisions = std::fs::read_to_string(root.join("docs/DECISIONS.md"))
        .expect("docs/DECISIONS.md must be readable from the crate");
    for statement in [
        "D-a: kernel determinism.",
        "D-b: reduction determinism.",
        "D-c: no assumed cross-boundary equality.",
    ] {
        assert!(
            decisions.contains(statement),
            "the determinism contract is missing this statement: {statement}"
        );
    }
    let mut harness = String::new();
    let mut modules: Vec<std::path::PathBuf> =
        std::fs::read_dir(root.join("crates/sparkflash-kernels/examples/m7_layer0"))
            .expect("the split M7 harness directory must be readable from the crate")
            .map(|entry| entry.expect("a harness module must be readable").path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
            .collect();
    modules.sort();
    for module in modules {
        harness.push_str(
            &std::fs::read_to_string(&module).expect("an M7 harness module must be readable"),
        );
    }
    for citation in [
        "D-a (kernel determinism)",
        "D-b (reduction determinism)",
        "D-c (no assumed cross-boundary equality)",
    ] {
        assert!(
            harness.contains(citation),
            "the harness does not cite the contract from its printed output: {citation}"
        );
    }
}
