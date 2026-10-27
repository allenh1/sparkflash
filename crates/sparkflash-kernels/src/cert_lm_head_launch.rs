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

//! Typed launch capability of the certified LM head (`kernels/cert_lm_head_sm121.cu`).
//!
//! The argument array is the `cert_lm_head_sm121` declaration's parameter list in order - seven
//! eight device pointers and the per-rank vocabulary - so a launch cannot swap the certification side
//! buffers for the logits or drop the rejection count. The declaration is pinned here against the
//! frozen ABI's embedded copy ([m4_abi::CERT_LM_HEAD]); [PARAMS] is the pinned name and type of
//! every parameter and [declaration_matches] is the check, with a mutation test proving it is not
//! vacuous.
//!
//! Geometry. One block screens one 1024-entry chunk of this rank's vocabulary (128 threads times 8
//! entries each), so the grid is `(ceil(vocab / 1024), 1, 1)` blocks of 128 threads and [grid]
//! refuses a zero vocabulary rather than launching an empty grid. The kernel's thread and chunk
//! constants are asserted against [CHUNK] in the tests.
//!
//! No cubin pin. The device chain's catalog owns the cubin's bytes and digest.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};
use crate::m4_abi;

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "cert_lm_head_sm121";
/// The kernel source the declaration pin is read from (the frozen ABI's embedded copy).
pub const SOURCE: &str = m4_abi::CERT_LM_HEAD;
/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = 12;

/// The declaration's parameters in order: `(name, base type, pointer, pointee const)`.
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("hidden", "__nv_bfloat16", true, true),
    ("w", "__nv_bfloat16", true, true),
    ("w_block_max", "__nv_bfloat16", true, true),
    ("logits", "float", true, false),
    ("certified", "uint8_t", true, false),
    ("certified_count", "unsigned int", true, false),
    ("block_max", "float", true, false),
    ("block_argmax", "uint32_t", true, false),
    ("vocab", "uint32_t", false, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// Parses the kernel declaration out of the pinned source.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KERNEL)
}

/// Whether a parsed declaration has [PARAMS]' shape and order.
#[must_use]
pub fn declaration_matches(decl: &KernelDecl) -> bool {
    decl.name == KERNEL
        && decl.extern_c
        && decl.global
        && decl.params.len() == ARG_WORDS
        && decl.params.iter().zip(PARAMS).all(|(param, pin)| {
            param.name == pin.0
                && param.base == pin.1
                && param.pointer == pin.2
                && param.const_pointee == pin.3
                && (!param.pointer || param.restrict)
        })
}

/// The K8 batched symbol: six rows in one launch, same arguments.
pub const BATCH_KERNEL: &str = "cert_lm_head_batched_sm121";
/// The batched head's fixed row count.
pub const BATCH_ROWS: u32 = 6;
/// The batched symbol's argument words: the scalar 12 plus the transposed [40][vocab] maxima.
pub const BATCH_ARG_WORDS: usize = 13;

/// One K8 batched launch's arguments. The transposed block maxima are batched-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CertLmHeadBatchParams([u64; BATCH_ARG_WORDS]);

impl CertLmHeadBatchParams {
    /// Builds the batched argument array in the batched declaration's order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        hidden: Deviceptr,
        w: Deviceptr,
        w_block_max_t: Deviceptr,
        logits: Deviceptr,
        certified: Deviceptr,
        certified_count: Deviceptr,
        block_max: Deviceptr,
        block_argmax: Deviceptr,
        vocab: u32,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            hidden.0,
            w.0,
            w_block_max_t.0,
            logits.0,
            certified.0,
            certified_count.0,
            block_max.0,
            block_argmax.0,
            u64::from(vocab),
            trace_ring,
            u64::from(layer_id),
            step_seq,
            0,
        ])
    }

    /// The parameter words a launch passes.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }

    /// The screened vocabulary.
    #[must_use]
    pub fn vocab(&self) -> u32 {
        self.0[8] as u32
    }
}

/// The kernel id in the shared driver's enum.
pub const KERNEL_ID: KernelId = KernelId::CertLmHead;
/// 128 threads, one 1024-entry vocabulary chunk per block.
pub const BLOCK: (u32, u32, u32) = (128, 1, 1);
/// The vocabulary entries one block screens.
pub const CHUNK: u32 = 1024;
/// The index of `vocab` in [CertLmHeadParams::as_slice].
pub const VOCAB_INDEX: usize = 8;

/// The grid for a rank's `vocab` entries: `(ceil(vocab / 1024), 1, 1)`.
///
/// # Errors
/// A zero vocabulary is refused rather than launched as a zero-sized grid.
pub fn grid(vocab: u32) -> Result<(u32, u32, u32), String> {
    if vocab == 0 {
        return Err(
            "the head screens at least one vocabulary entry, so the grid is not zero-sized"
                .to_owned(),
        );
    }
    Ok((vocab.div_ceil(CHUNK), 1, 1))
}

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CertLmHeadParams([u64; ARG_WORDS]);

impl CertLmHeadParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        hidden: Deviceptr,
        w: Deviceptr,
        w_block_max: Deviceptr,
        logits: Deviceptr,
        certified: Deviceptr,
        certified_count: Deviceptr,
        block_max: Deviceptr,
        block_argmax: Deviceptr,
        vocab: u32,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            hidden.0,
            w.0,
            w_block_max.0,
            logits.0,
            certified.0,
            certified_count.0,
            block_max.0,
            block_argmax.0,
            u64::from(vocab),
            trace_ring,
            u64::from(layer_id),
            step_seq,
        ])
    }

    /// The per-rank vocabulary size.
    #[must_use]
    pub fn vocab(&self) -> u32 {
        self.0[VOCAB_INDEX] as u32
    }

    /// The parameter words a launch passes, the shape `cuLaunchKernel` takes.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [KERNEL].
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Loads the K8 batched symbol by name from the same cubin.
pub fn load_batched(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_cert_lm_head_batched(cubin)
}

static OPT_FUNCTION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const OPT_SYMBOL: &str = "opt_lm_head";

/// Loads the exact warp-per-row head (opt_misc_sm121.cubin) and routes [`launch`] through it: same
/// grid and outputs, every logit exact, nothing certified.
///
/// # Errors
/// The driver's error when the cubin lacks the symbol.
pub fn install_opt(cuda: &Cuda, cubin: &[u8]) -> Result<Module, String> {
    let (module, kernel) = cuda.load_named(cubin, OPT_SYMBOL)?;
    OPT_FUNCTION.store(kernel.to_raw().0, std::sync::atomic::Ordering::Release);
    Ok(module)
}

/// Launches the screening pass once.
///
/// # Errors
/// Refuses a zero vocabulary, and returns the driver's error otherwise.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &CertLmHeadParams) -> Result<(), String> {
    let grid = grid(params.vocab())?;
    let word = OPT_FUNCTION.load(std::sync::atomic::Ordering::Acquire);
    if word != 0 {
        return cuda.launch(
            Kernel::from_raw(word, OPT_SYMBOL),
            grid,
            (512, 1, 1),
            params.as_slice(),
        );
    }
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

/// Launches the K8 batched head once over the fixed six-row batch.
///
/// The batched ABI is the scalar 12 words plus `w_block_max_t` [40][vocab] at index 2;
/// `hidden`, `logits`, `certified`, `certified_count`, `block_max` and `block_argmax` are
/// batch-major over [BATCH_ROWS] rows.
///
/// # Errors
/// Refuses a zero vocabulary, and returns the driver's error otherwise.
pub fn launch_batched(
    cuda: &Cuda,
    kernel: Kernel,
    params: &CertLmHeadBatchParams,
) -> Result<(), String> {
    let grid = grid(params.vocab())?;
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;

    fn params(vocab: u32) -> CertLmHeadParams {
        CertLmHeadParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x2500),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            Deviceptr(0x5000),
            Deviceptr(0x6000),
            Deviceptr(0x7000),
            vocab,
            0x8000,
            7,
            11,
        )
    }

    fn constants() -> Consts {
        let mut table = Consts::parse(SOURCE);
        table.extend(m4_abi::header_source());
        table
    }

    #[test]
    fn the_kernel_declaration_is_the_pinned_one() {
        let decl = declaration();
        assert!(
            declaration_matches(&decl),
            "the declaration drifted: {decl:?}"
        );
        assert_eq!(decl.params.len(), ARG_WORDS);
    }

    #[test]
    fn the_declaration_pin_is_not_vacuous() {
        let mutated = SOURCE.replace(
            "unsigned int *__restrict__ certified_count",
            "unsigned int * certified_count",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "the qualifier pin is vacuous: {decl:?}"
        );
    }

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let p = params(32_320);
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x2500, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 32_320, 0x8000, 7,
                11
            ]
        );
        assert_eq!(p.vocab(), 32_320);
        assert_eq!(p.as_slice().len(), ARG_WORDS);
    }

    #[test]
    fn the_grid_is_the_kernel_chunking() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_LM_HEAD_THREADS"), i128::from(BLOCK.0));
        assert_eq!(table.value("SPARK_M4_LM_CHUNK"), i128::from(CHUNK));
        assert_eq!(grid(32_320), Ok((32, 1, 1)));
        assert_eq!(grid(1), Ok((1, 1, 1)));
        assert_eq!(grid(1025), Ok((2, 1, 1)));
        assert!(grid(0).is_err());
    }
}
