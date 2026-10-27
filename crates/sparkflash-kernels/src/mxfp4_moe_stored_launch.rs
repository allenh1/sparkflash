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

//! Typed launch capability of the stored-layout MXFP4 down/combine MoE kernel
//! (kernels/b12x_mxfp4_moe_stored_sm121.cu).
//!
//! The argument array is the frozen combine declaration's parameter list, in order - six device
//! pointers, the token count, the trace ring, the layer id and the step sequence - and it is the
//! SAME array type as the incumbent launch ([crate::mxfp4_moe_launch::Mxfp4MoeParams]), because the
//! stored kernel is a drop-in for the same call site and the word order must not be transcribed
//! twice. The declaration is pinned against [crate::mxfp4_moe_stored_abi::STORED_PARAMS], which is
//! the incumbent's pinned parameter list by identity.
//!
//! Geometry. One block computes one 32-column hidden tile of one token's routed row, with 32 output
//! columns x 8 parallel K groups = 256 threads; the grid is (tokens, 160, 1) blocks of 256 threads
//! and [grid] refuses a zero token count rather than launching an empty grid. The hidden axis is
//! HIDDEN / TILE_D = 160 tiles because the stored kernel's tile is 32 columns, not the incumbent's
//! 256: the two kernels have different grids and this module is the statement of the stored one.
//!
//! No cubin pin here. The device chain's catalog owns the cubin's bytes and digest; the pin itself
//! (bytes, sha256, pinned-toolkit release) lives in [crate::mxfp4_moe_stored_abi] and is measured by
//! the baseline scripts.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Kernel, KernelId, Module};
use crate::mxfp4_moe_launch::Mxfp4MoeParams;
use crate::mxfp4_moe_stored_abi;

/// The kernel symbol in the cubin.
pub const KERNEL: &str = mxfp4_moe_stored_abi::KERNEL;
/// The kernel source the declaration pin is read from.
pub const SOURCE: &str = mxfp4_moe_stored_abi::SOURCE;
/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = mxfp4_moe_stored_abi::ARG_WORDS;

/// The stored kernel's launch parameters. It is the incumbent's type on purpose: the chain swaps
/// cubins, not argument arrays, and a second struct would be a second place for the order to drift.
pub type Mxfp4MoeStoredParams = Mxfp4MoeParams;

/// The declaration's parameters in order: (name, base type, pointer, pointee const).
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = mxfp4_moe_stored_abi::STORED_PARAMS;

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

/// The kernel id in the shared driver's enum.
pub const KERNEL_ID: KernelId = KernelId::Mxfp4MoeStored;
/// 256 threads: 32 output columns x 8 parallel K groups.
pub const BLOCK: (u32, u32, u32) = (mxfp4_moe_stored_abi::THREADS, 1, 1);
/// The hidden values one block writes.
pub const TILE_D: u32 = mxfp4_moe_stored_abi::TILE_D;
/// The parallel K groups one block's threads form.
pub const K_GROUPS: u32 = mxfp4_moe_stored_abi::K_GROUPS;
/// The number of hidden tiles: HIDDEN / TILE_D.
pub const GRID_Y: u32 = mxfp4_moe_stored_abi::GRID_Y;
/// The index of tokens in [Mxfp4MoeParams::as_slice].
pub const TOKENS_INDEX: usize = 6;

/// The grid for tokens routed rows: (tokens, GRID_Y, 1).
///
/// # Errors
/// A zero token count is refused rather than launched as a zero-sized grid.
pub fn grid(tokens: u32) -> Result<(u32, u32, u32), String> {
    if tokens == 0 {
        return Err(
            "the stored combine runs on at least one token, so the grid is not zero-sized"
                .to_owned(),
        );
    }
    Ok((tokens, GRID_Y, 1))
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [KERNEL].
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the stored combine once.
///
/// # Errors
/// Refuses a zero token count, and returns the driver's error otherwise.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &Mxfp4MoeStoredParams) -> Result<(), String> {
    let grid = grid(params.tokens())?;
    cuda.launch(kernel, grid, BLOCK, params.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_driver::Deviceptr;

    fn constants() -> c_decl::Consts {
        mxfp4_moe_stored_abi::constants()
    }

    fn params(tokens: u32) -> Mxfp4MoeStoredParams {
        Mxfp4MoeStoredParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            Deviceptr(0x5000),
            Deviceptr(0x6000),
            tokens,
            0x7000,
            7,
            11,
        )
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
            "    const uint8_t *__restrict__ w_scales,",
            "    const uint8_t * w_scales,",
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
        let p = params(130);
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 130, 0x7000, 7, 11
            ]
        );
        assert_eq!(p.tokens(), 130);
        assert_eq!(p.as_slice().len(), ARG_WORDS);
        // The stored launch reuses the incumbent's array type; the words must be identical for the
        // same arguments, which is the property the chain's selector relies on.
        let incumbent = crate::mxfp4_moe_launch::Mxfp4MoeParams::new(
            Deviceptr(0x1000),
            Deviceptr(0x2000),
            Deviceptr(0x3000),
            Deviceptr(0x4000),
            Deviceptr(0x5000),
            Deviceptr(0x6000),
            130,
            0x7000,
            7,
            11,
        );
        assert_eq!(p.as_slice(), incumbent.as_slice());
    }

    #[test]
    fn the_grid_is_the_stored_kernel_tiling() {
        let table = constants();
        assert_eq!(table.value("SPARK_M4_STORED_TILE_D"), i128::from(TILE_D));
        assert_eq!(table.value("SPARK_M4_STORED_THREADS"), i128::from(BLOCK.0));
        assert_eq!(
            table.value("SPARK_M4_STORED_K_GROUPS"),
            i128::from(K_GROUPS)
        );
        assert_eq!(
            table.value("SPARK_M4_HIDDEN") / i128::from(TILE_D),
            i128::from(GRID_Y)
        );
        assert_eq!(grid(8), Ok((8, GRID_Y, 1)));
        assert!(grid(0).is_err());
    }

    #[test]
    fn the_kernel_id_resolves_the_pinned_symbol() {
        assert_eq!(KERNEL_ID.cubin_names(), &[KERNEL]);
        assert_eq!(KERNEL_ID.cubin_names()[0], "b12x_mxfp4_moe_stored_sm121");
        assert_eq!(
            crate::cuda_driver::KernelId::from_cubin_name(KERNEL),
            Some(KERNEL_ID)
        );
    }
}
