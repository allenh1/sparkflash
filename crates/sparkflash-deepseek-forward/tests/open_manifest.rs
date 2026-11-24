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

//! The open manifest's model-value records, written with no GPU.

use sparkflash_deepseek_forward::open_manifest::record_model_config;
use sparkflash_deepseek_forward::plan::tiny_values;

#[test]
fn every_configuration_field_is_one_ranked_record() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("manifest.jsonl");
    // SAFETY: this test binary has one test, and the manifest is initialized on this thread after
    // the variable is set. No other test reads or writes the process environment.
    unsafe {
        std::env::set_var(sparkflash_kernels::open_manifest::ENV, &path);
    }
    record_model_config(&tiny_values(), 3, 4);
    let text = std::fs::read_to_string(&path).expect("the manifest file");
    let lines: Vec<&str> = text.lines().collect();
    // One line per ModelConfig field; the stored format's block and scale values are two of them.
    assert_eq!(lines.len(), 68, "one line per ModelConfig field");
    assert!(lines.iter().all(|line| line.contains("\"kind\":\"model\"")));
    assert!(lines.iter().all(|line| line.contains("\"rank\":3")));
    assert!(text.contains("\"name\":\"dim\",\"value\":64"));
    assert!(text.contains("\"name\":\"vision_n_layers\",\"value\":null"));
}
