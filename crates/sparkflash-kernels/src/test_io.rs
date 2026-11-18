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

//! Kernel-test support: the context a test runs on or skips without, a slice of any plain type
//! into a new allocation, and a run of plain values read back once the context has synchronized.
//! The copies panic on a driver error, which is what a test wants; serving code takes the
//! driver's results.

use crate::cuda_driver::{Cuda, Deviceptr};

/// A context for a kernel test, or `None` when it skips: `missing` names an image this build does
/// not embed, or no driver loads. The reason goes to stderr.
#[must_use]
pub fn cuda_or_skip(missing: Option<&str>) -> Option<Cuda> {
    if let Some(image) = missing {
        eprintln!("skipped: this build embeds no {image}");
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

/// `values` copied into a new allocation of their bytes (one word at least, so an empty slice
/// still has an address).
///
/// # Panics
/// On an allocation or copy failure.
#[must_use]
pub fn upload<T: bytemuck::NoUninit>(cuda: &Cuda, values: &[T]) -> Deviceptr {
    let bytes: &[u8] = bytemuck::cast_slice(values);
    let pointer = cuda.alloc(bytes.len().max(4), &mut 0).expect("alloc");
    cuda.copy_htod(&pointer, bytes).expect("upload");
    pointer
}

/// `elements` values at `pointer`, read once the context has synchronized.
///
/// # Panics
/// On a synchronize or copy failure.
#[must_use]
pub fn download<T: bytemuck::Pod>(cuda: &Cuda, pointer: &Deviceptr, elements: usize) -> Vec<T> {
    cuda.synchronize().expect("synchronize");
    let mut values = vec![T::zeroed(); elements];
    cuda.copy_dtoh(bytemuck::cast_slice_mut(&mut values), pointer)
        .expect("download");
    values
}
