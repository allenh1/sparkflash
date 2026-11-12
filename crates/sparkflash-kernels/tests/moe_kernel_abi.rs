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

//! The host launch argument lists against the kernels' own parameter lists.
//!
//! The driver appends a spare zero to a launch's parameters, so a call site that passes fewer
//! arguments than the kernel declares still compiles and runs: the missing parameter is silently
//! zero. That is how the k3 dedup entry gained the hoisted plan's input and the drafter kept
//! passing twelve arguments for weeks - it took the in-block branch every time, and every host gate
//! stayed green. These tests count the parameters out of the CUDA source so the two lists cannot
//! drift again without a failure.

use sparkflash_kernels::moe_routed_device::{OPT_MOE_GATE_UP_DEDUP_ARGS, OPT_MOE_PLAN_ARGS};

fn source() -> String {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/opt_moe_sm121.cu");
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

/// The top-level parameter count of the parenthesised list whose `(` is at `open`.
fn parameter_count(text: &str, open: usize) -> usize {
    let bytes = text.as_bytes();
    assert_eq!(bytes[open], b'(', "expected a parameter list");
    let mut depth = 0usize;
    let mut commas = 0usize;
    for &byte in bytes.iter().skip(open) {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return commas + 1;
                }
            }
            b',' if depth == 1 => commas += 1,
            _ => {}
        }
    }
    panic!("unbalanced parameter list");
}

#[test]
fn the_dedup_entry_takes_the_arguments_the_host_passes() {
    let text = source();
    let define = text
        .find("#define OPT_MOE_GATE_UP_DEDUP_ENTRY(")
        .expect("the dedup entry macro");
    // The macro's own parameter list comes first; the function declarator is the body's `name(`.
    let body = define + "#define OPT_MOE_GATE_UP_DEDUP_ENTRY(".len();
    let declarator = body
        + text[body..]
            .find("name(")
            .expect("the entry's declarator in the macro body");
    assert_eq!(
        parameter_count(&text, declarator + "name".len()),
        OPT_MOE_GATE_UP_DEDUP_ARGS,
        "kernels/opt_moe_sm121.cu's OPT_MOE_GATE_UP_DEDUP_ENTRY grew or lost a parameter"
    );
}

#[test]
fn the_plan_kernel_takes_the_arguments_the_host_passes() {
    let text = source();
    let open = text
        .find("opt_moe_plan_sm121(")
        .expect("opt_moe_plan_sm121's definition");
    assert_eq!(
        parameter_count(&text, open + "opt_moe_plan_sm121".len()),
        OPT_MOE_PLAN_ARGS,
        "kernels/opt_moe_sm121.cu's opt_moe_plan_sm121 grew or lost a parameter"
    );
}
