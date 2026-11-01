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

//! The first device gate: load the cubin and run one RMSNorm, comparing the read-back.
//!
//! Run on a GPU host built with nvcc: cargo run -p sparkflash-model-qwen4-exp --example device_self_test

fn main() -> std::process::ExitCode {
    match sparkflash_model_qwen4_exp::device::device_self_test() {
        Ok(()) => {
            println!("qwen4-exp device self-test: the cubin loaded and the RMSNorm matched");
            std::process::ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("qwen4-exp device self-test: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
