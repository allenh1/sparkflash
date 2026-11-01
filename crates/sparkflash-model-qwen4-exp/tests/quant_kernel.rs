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

//! The device's E4M3 decode against the host's, over every code.
//!
//! The NVFP4 block scales in the released checkpoint are E4M3, and the format's corner cases are
//! where a decoder goes quietly wrong: it has no infinity, so exponent 15 with a mantissa below 7
//! is a finite value up to 448, and only 0x7F/0xFF are NaN. The device used to return infinity for
//! the finite ones, which the tiny fixture never exercised because it encoded every block scale as
//! 1.0 - and the released experts do use them. This gate compares all 256 codes.
//!
//! It skips - loudly - without an embedded cubin or a CUDA driver.

use sparkflash_kernels::cuda_driver::Cuda;
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_cubin};
use sparkflash_weights::quant::f8e4m3_to_f32;

#[test]
fn every_e4m3_code_decodes_the_way_the_host_decodes_it() {
    if !cubin_available() {
        eprintln!("skipped: this build embeds no Qwen4-Exp cubin");
        return;
    }
    let cuda = match Cuda::load() {
        Ok(cuda) => cuda,
        Err(error) => {
            eprintln!("skipped: no CUDA driver ({error})");
            return;
        }
    };
    let codes: Vec<u8> = (0u8..=255).collect();
    let mut total = 0_usize;
    let device_codes = cuda.alloc(codes.len(), &mut total).expect("codes");
    cuda.copy_htod(&device_codes, &codes).expect("upload codes");
    let device_out = cuda.alloc(codes.len() * 4, &mut total).expect("decoded");
    let (_, kernel) = cuda
        .load_named(embedded_cubin(), "q4e_decode_f8e4m3")
        .expect("the cubin carries q4e_decode_f8e4m3");
    cuda.launch(
        kernel,
        (1, 1, 1),
        (256, 1, 1),
        &[device_codes.0, device_out.0, codes.len() as u64],
    )
    .expect("decode launch");
    let mut bytes = vec![0_u8; codes.len() * 4];
    cuda.copy_dtoh(&mut bytes, &device_out)
        .expect("decode read-back");
    let decoded: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut diverged = Vec::new();
    for (code, device) in codes.iter().zip(&decoded) {
        let host = f8e4m3_to_f32(*code);
        let agrees = if host.is_nan() {
            device.is_nan()
        } else {
            device.is_finite() && (device - host).abs() <= host.abs().max(1.0) * 1e-6
        };
        if !agrees {
            diverged.push((*code, host, *device));
        }
    }
    assert!(
        diverged.is_empty(),
        "the device and host disagree about {} E4M3 codes, first few: {:?}",
        diverged.len(),
        &diverged[..diverged.len().min(8)]
    );

    // The corner the released checkpoint actually uses: exponent 15 with a mantissa below 7 is
    // finite on both sides, and the largest of them is 448.
    let largest = decoded[0x7E];
    assert!(
        largest.is_finite() && (largest - 448.0).abs() < 1e-3,
        "0x7E should decode to 448, got {largest}"
    );
    assert!(decoded[0x7F].is_nan(), "0x7F is E4M3's NaN");
    assert!(decoded[0xFF].is_nan(), "0xFF is E4M3's NaN");
    assert_eq!(decoded[0x78], 256.0, "0x78 is 2^8");
}
