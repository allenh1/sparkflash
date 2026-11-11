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

//! The device arm against the host oracle on the tiny fixture.

use sparkflash_model::backend::ModelBackend;
use sparkflash_model_dory::DoryDeviceBackend;
use sparkflash_model_dory::config::DoryConfig;
use sparkflash_model_dory::fixture;
use sparkflash_model_dory::forward::DoryWeights;
use sparkflash_weights::safetensors::open_checkpoint;

#[test]
#[ignore = "needs a Dory/Qwen cubin and a CUDA device"]
fn the_device_arm_matches_the_host_or_skips_loudly() {
    if !DoryDeviceBackend::available() {
        eprintln!(
            "skipping the Dory device parity: this build has no Dory/Qwen cubin (rebuild with \
             nvcc, or set SPARKFLASH_CUDA_ARCH to this GPU's architecture)"
        );
        return;
    }
    let config = DoryConfig::tiny_ring();
    let directory = tempfile::tempdir().expect("a temporary checkpoint");
    fixture::write(directory.path(), &DoryConfig::tiny_ring_json()).expect("the fixture writes");
    let source = open_checkpoint(directory.path()).expect("the fixture opens");
    let host_weights = DoryWeights::load(&config, &source).expect("the host weights load");
    let prompt: Vec<u32> = (0..4200).map(|index| (index % 63 + 1) as u32).collect();
    let host = host_weights
        .forward_prompt(&prompt)
        .expect("the host prompt runs");
    let backend = match DoryDeviceBackend::open(directory.path(), 5000) {
        Ok(backend) => backend,
        Err(error) if error.contains("CUDA:") || error.contains("cuModuleLoadData failed: 209") => {
            eprintln!("skipping the Dory device parity: {error}");
            return;
        }
        Err(error) => panic!("the device arm refused the fixture: {error}"),
    };
    let mut session = backend.open_session().expect("a device session");
    let output = session.prefill(&prompt).expect("the device prompt runs");
    let mut sequential = backend.open_session().expect("a sequential device session");
    let mut sequential_logits = Vec::new();
    for token in &prompt {
        sequential_logits = sequential
            .decode(*token)
            .expect("the sequential device prompt runs")
            .logits;
    }
    let sequential_largest = output
        .logits
        .iter()
        .zip(&sequential_logits)
        .map(|(batch, step)| (batch - step).abs())
        .fold(0.0f32, f32::max);
    let sequential_scale = sequential_logits
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let sequential_relative = sequential_largest / sequential_scale;
    eprintln!(
        "Dory batch vs sequential: max |batch - step| = {sequential_largest:.6}, relative \
         {sequential_relative:.3e}"
    );
    assert!(
        sequential_relative < 1e-3,
        "batch-vs-sequential relative difference {sequential_relative:.3e} exceeds 1e-3"
    );
    assert_eq!(host.len(), output.logits.len());
    assert!(
        output.logits.iter().all(|value| value.is_finite()),
        "the device produced a non-finite logit"
    );
    let scale = host
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1e-3);
    let largest = host
        .iter()
        .zip(&output.logits)
        .map(|(host, device)| (host - device).abs())
        .fold(0.0f32, f32::max);
    let relative = largest / scale;
    eprintln!("Dory device parity: max |host - device| = {largest:.6}, relative {relative:.3e}");
    // The device rounds every weight to BF16 and accumulates each projection in f32; the host
    // oracle keeps the released weights in f32. A 4,200-token prompt drifts a little further
    // than the four-token fixture did, and the batch-vs-sequential check above pins the device
    // arithmetic itself to the token-at-a-time path.
    assert!(
        relative < 1e-3,
        "relative difference {relative:.3e} exceeds 1e-3"
    );
}
