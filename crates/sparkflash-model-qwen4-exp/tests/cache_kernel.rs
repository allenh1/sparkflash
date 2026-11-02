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

//! The KV cache's narrowing stores: f32 projection rows into BF16 cache rows.
//!
//! Both kernels are the only writers of the Qwen4-Exp cache, so their rounding mode and row
//! strides are pinned here rather than inferred from a full model run. They skip - loudly -
//! without an embedded cubin or a CUDA driver.

use half::bf16;
use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};
use sparkflash_model_qwen4_exp::device::{cubin_available, embedded_cubin};

fn open() -> Option<Cuda> {
    if !cubin_available() {
        eprintln!("skipped: this build embeds no Qwen4-Exp cubin");
        return None;
    }
    match Cuda::load() {
        Ok(cuda) => Some(cuda),
        Err(error) => {
            eprintln!("skipped: no CUDA driver ({error})");
            None
        }
    }
}

fn upload_f32(cuda: &Cuda, values: &[f32]) -> Deviceptr {
    let mut total = 0;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let pointer = cuda.alloc(bytes.len().max(4), &mut total).expect("alloc");
    cuda.copy_htod(&pointer, &bytes).expect("upload");
    pointer
}

fn download_u16(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<u16> {
    cuda.synchronize().expect("synchronize");
    let mut bytes = vec![0u8; elements * 2];
    cuda.copy_dtoh(&mut bytes, pointer).expect("download");
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

#[test]
fn the_flat_cache_store_narrows_every_value_with_rne() {
    let Some(cuda) = open() else {
        return;
    };
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        1.0 + 1.0 / 256.0,
        1.0 + 3.0 / 256.0,
        -2.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let source = upload_f32(&cuda, &values);
    let mut total = 0;
    let destination = cuda
        .alloc(values.len() * 2, &mut total)
        .expect("cache alloc");
    let (_, kernel) = cuda
        .load_named(embedded_cubin(), "q4e_cache_bf16")
        .expect("the cubin carries q4e_cache_bf16");
    cuda.launch(
        kernel,
        (1, 1, 1),
        (256, 1, 1),
        &[destination.0, source.0, values.len() as u64],
    )
    .expect("cache store");
    let got = download_u16(&cuda, &destination, values.len());
    let want: Vec<u16> = values
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect();
    assert_eq!(got, want);
}

#[test]
fn the_strided_cache_store_honours_the_destination_stride() {
    let Some(cuda) = open() else {
        return;
    };
    const WIDTH: usize = 5;
    const DEST_STRIDE: usize = 8;
    const SOURCE_STRIDE: usize = 7;
    const ROWS: usize = 3;
    let mut source = Vec::with_capacity(ROWS * SOURCE_STRIDE);
    for row in 0..ROWS {
        for column in 0..SOURCE_STRIDE {
            source.push((row * 10 + column) as f32 / 8.0);
        }
    }
    let source_on = upload_f32(&cuda, &source);
    let mut total = 0;
    let destination = cuda
        .alloc(ROWS * DEST_STRIDE * 2, &mut total)
        .expect("cache alloc");
    // A sentinel in every byte, so a store that ignored the destination stride would leave a
    // mark where the next row should have started.
    cuda.copy_htod(&destination, &[0xFFu8; ROWS * DEST_STRIDE * 2])
        .expect("destination sentinel");
    let (_, kernel) = cuda
        .load_named(embedded_cubin(), "q4e_cache_rows_bf16")
        .expect("the cubin carries q4e_cache_rows_bf16");
    cuda.launch(
        kernel,
        (1, ROWS as u32, 1),
        (256, 1, 1),
        &[
            destination.0,
            source_on.0,
            WIDTH as u64,
            DEST_STRIDE as u64,
            SOURCE_STRIDE as u64,
        ],
    )
    .expect("strided cache store");
    let got = download_u16(&cuda, &destination, ROWS * DEST_STRIDE);
    let mut want = vec![0xFFFFu16; ROWS * DEST_STRIDE];
    for row in 0..ROWS {
        for column in 0..WIDTH {
            want[row * DEST_STRIDE + column] =
                bf16::from_f32(source[row * SOURCE_STRIDE + column]).to_bits();
        }
    }
    assert_eq!(got, want);
}
