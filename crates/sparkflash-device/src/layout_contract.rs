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

//! The single source of truth for every pool stride and offset the device seam assumes.
//!
//! The paged staging path (crate::paged_kv), the paged-read design reference
//! (crate::paged_read) and the pinned CUDA attention entry that consumes their operands all address
//! the same format-2 pool: interleaved entries whose main codes start at +0, main scales at +256,
//! indexer K data at +288 and its scales at +352, in 64-token pages. The layout is defined once in
//! sparkflash_kv::layout; the ASSUMED_* values below are what the device side bakes into its
//! addressing. The const assertions make any disagreement between the two a compile error in this
//! crate rather than a silent read of the wrong bytes - the failure mode the task-49 design note
//! warned about.
//!
//! The runtime tests in crates/sparkflash-device/tests/layout_contract.rs extend the guard to the
//! already-wired paths: they check the byte offsets PagedKvView computes, the window gather stride
//! and the HSI indexer pool against these same constants, and they parse the CUDA common header
//! (sparkflash_kernels::m4_abi::COMMON_HEADER) so the kernel's own defines are held to the same
//! source of truth without changing the frozen ABI.
//!
//! The ASSUMED_* literals are deliberately duplicated here. A literal in this module is a reviewed
//! statement of what the device code assumes; the const assertions are what make an unreviewed
//! change to sparkflash_kv::layout fail loudly instead of corrupting device reads.

use sparkflash_kernels::cuda_harness::WindowRingLayout;
use sparkflash_kernels::m4_abi::{COMMON_HEADER, FLASH_MLA_CSA2};
use sparkflash_kernels::mla_ref;
use sparkflash_kv::config::KV_PAGE_TOKENS;
use sparkflash_kv::layout::{
    GLOBAL_ENTRY_BYTES, INDEXER_K_DATA_BYTES, INDEXER_K_DATA_OFFSET, INDEXER_K_DIM,
    INDEXER_K_ENTRY_BYTES, INDEXER_K_SCALE_BYTES, INDEXER_K_SCALE_GROUP, INDEXER_K_SCALE_OFFSET,
    MAIN_KV_DATA_BYTES, MAIN_KV_DATA_OFFSET, MAIN_KV_DIM, MAIN_KV_ENTRY_BYTES, MAIN_KV_SCALE_BYTES,
    MAIN_KV_SCALE_GROUP, MAIN_KV_SCALE_OFFSET, SWA_FP8_SCALE_GROUP, SWA_LAYERS, SWA_SLOT_BYTES,
    SWA_SLOT_DATA_BYTES, SWA_SLOT_DIM, SWA_SLOT_SCALE_BYTES, SWA_SLOTS_PER_TAIL,
};

// The interleaved global entry: main-KV codes, main-KV scales, indexer-K data, indexer-K scales.
/// Bytes of one main-KV entry: 256 codes + 32 scales.
pub const ASSUMED_MAIN_KV_ENTRY_BYTES: usize = 288;
/// Bytes of one interleaved global entry: 288 main + 68 indexer.
pub const ASSUMED_GLOBAL_ENTRY_BYTES: usize = 356;
/// Start of the main-KV codes inside a global entry.
pub const ASSUMED_MAIN_KV_DATA_OFFSET: usize = 0;
/// Bytes of packed FP4 main-KV codes.
pub const ASSUMED_MAIN_KV_DATA_BYTES: usize = 256;
/// Start of the main-KV E4M3 scales inside a global entry.
pub const ASSUMED_MAIN_KV_SCALE_OFFSET: usize = 256;
/// Bytes of main-KV E4M3 scales.
pub const ASSUMED_MAIN_KV_SCALE_BYTES: usize = 32;
/// Start of the indexer-K FP4 data inside a global entry.
pub const ASSUMED_INDEXER_K_DATA_OFFSET: usize = 288;
/// Bytes of packed FP4 indexer-K data.
pub const ASSUMED_INDEXER_K_DATA_BYTES: usize = 64;
/// Start of the indexer-K E8M0 scales inside a global entry.
pub const ASSUMED_INDEXER_K_SCALE_OFFSET: usize = 352;
/// Bytes of indexer-K E8M0 scales.
pub const ASSUMED_INDEXER_K_SCALE_BYTES: usize = 4;
/// Bytes of one indexer-K entry.
pub const ASSUMED_INDEXER_K_ENTRY_BYTES: usize = 68;

// Page and latent geometry.
/// Tokens one pool page covers.
pub const ASSUMED_KV_PAGE_TOKENS: usize = 64;
/// The MLA latent width (main-KV codes are packed two per byte).
pub const ASSUMED_MLA_LATENT: usize = 512;
/// Query heads the attention entry serves.
pub const ASSUMED_MLA_HEADS: usize = 64;
/// Channels per main-KV E4M3 scale.
pub const ASSUMED_MAIN_KV_SCALE_GROUP: usize = 16;
/// The indexer key width.
pub const ASSUMED_INDEXER_K_DIM: usize = 128;
/// Channels per indexer-K E8M0 scale.
pub const ASSUMED_INDEXER_K_SCALE_GROUP: usize = 32;
/// Selected-candidate capacity of the union attention (index_topk).
pub const ASSUMED_INDEX_TOPK: usize = 512;

// The sliding-window ring: 128 slots of 512 B codes + 16 B scales.
/// Slots in one layer's window ring.
pub const ASSUMED_SWA_SLOTS: usize = 128;
/// Layers that own a window ring.
pub const ASSUMED_SWA_LAYERS: usize = 40;
/// Width of one window slot's code row in channels.
pub const ASSUMED_SWA_SLOT_DIM: usize = 512;
/// Bytes of one window slot's code row.
pub const ASSUMED_SWA_SLOT_DATA_BYTES: usize = 512;
/// Bytes of one window slot's scale row.
pub const ASSUMED_SWA_SLOT_SCALE_BYTES: usize = 16;
/// Bytes of one window slot across both arrays.
pub const ASSUMED_SWA_SLOT_BYTES: usize = 528;
/// The E8M0 scale group of the window codes.
pub const ASSUMED_SWA_SCALE_GROUP: usize = 32;

// The high-speed index (HSI) candidate pool the indexer operand is built from.
/// HSI blocks in the indexer candidate pool.
pub const ASSUMED_HSI_BLOCKS: usize = 2048;
/// Tokens per HSI block.
pub const ASSUMED_HSI_BLOCK_TOKENS: usize = 8;
/// Bytes of the HSI index-key operand: blocks x tokens x channels x bf16.
pub const ASSUMED_HSI_INDEX_KEY_BYTES: usize = 4_194_304;

// The compile-time guard. Each assertion compares a device-side assumption with the layout
// definition; if the layout changes, sparkflash-device stops compiling and the assumption must be
// re-reviewed instead of silently reading the wrong byte.
const _: () = assert!(ASSUMED_MAIN_KV_DATA_OFFSET == MAIN_KV_DATA_OFFSET);
const _: () = assert!(ASSUMED_MAIN_KV_DATA_BYTES == MAIN_KV_DATA_BYTES);
const _: () = assert!(ASSUMED_MAIN_KV_SCALE_OFFSET == MAIN_KV_SCALE_OFFSET);
const _: () = assert!(ASSUMED_MAIN_KV_SCALE_BYTES == MAIN_KV_SCALE_BYTES);
const _: () = assert!(ASSUMED_MAIN_KV_ENTRY_BYTES == MAIN_KV_ENTRY_BYTES);
const _: () = assert!(ASSUMED_INDEXER_K_DATA_OFFSET == INDEXER_K_DATA_OFFSET);
const _: () = assert!(ASSUMED_INDEXER_K_DATA_BYTES == INDEXER_K_DATA_BYTES);
const _: () = assert!(ASSUMED_INDEXER_K_SCALE_OFFSET == INDEXER_K_SCALE_OFFSET);
const _: () = assert!(ASSUMED_INDEXER_K_SCALE_BYTES == INDEXER_K_SCALE_BYTES);
const _: () = assert!(ASSUMED_INDEXER_K_ENTRY_BYTES == INDEXER_K_ENTRY_BYTES);
const _: () = assert!(ASSUMED_GLOBAL_ENTRY_BYTES == GLOBAL_ENTRY_BYTES);
const _: () = assert!(ASSUMED_KV_PAGE_TOKENS == KV_PAGE_TOKENS);
const _: () = assert!(ASSUMED_MLA_LATENT == MAIN_KV_DIM);
const _: () = assert!(ASSUMED_MAIN_KV_SCALE_GROUP == MAIN_KV_SCALE_GROUP);
const _: () = assert!(ASSUMED_INDEXER_K_DIM == INDEXER_K_DIM);
const _: () = assert!(ASSUMED_INDEXER_K_SCALE_GROUP == INDEXER_K_SCALE_GROUP);
const _: () = assert!(ASSUMED_SWA_SLOTS == SWA_SLOTS_PER_TAIL);
const _: () = assert!(ASSUMED_SWA_LAYERS == SWA_LAYERS);
const _: () = assert!(ASSUMED_SWA_SLOT_DATA_BYTES == SWA_SLOT_DATA_BYTES);
const _: () = assert!(ASSUMED_SWA_SLOT_SCALE_BYTES == SWA_SLOT_SCALE_BYTES);
const _: () = assert!(ASSUMED_SWA_SLOT_BYTES == SWA_SLOT_BYTES);
const _: () = assert!(ASSUMED_SWA_SCALE_GROUP == SWA_FP8_SCALE_GROUP);
const _: () = assert!(ASSUMED_MLA_LATENT == mla_ref::MLA_LATENT);
const _: () = assert!(ASSUMED_MLA_HEADS == mla_ref::MLA_HEADS);
const _: () = assert!(ASSUMED_INDEXER_K_DIM == mla_ref::INDEX_HEAD_DIM);
const _: () = assert!(ASSUMED_HSI_BLOCKS == mla_ref::HSI_BLOCKS);
const _: () = assert!(ASSUMED_HSI_BLOCK_TOKENS == mla_ref::HSI_BLOCK_TOKENS);
const _: () = assert!(WindowRingLayout::ENGINE.slots == ASSUMED_SWA_SLOTS);
const _: () = assert!(WindowRingLayout::ENGINE.code_row_bytes == ASSUMED_SWA_SLOT_DATA_BYTES);
const _: () = assert!(WindowRingLayout::ENGINE.scale_row_bytes == ASSUMED_SWA_SLOT_SCALE_BYTES);

// The assumed values must also agree with each other; a reviewed change to one literal without the
// others is caught here even before it is compared with the layout.
const _: () = assert!(
    ASSUMED_MAIN_KV_DATA_OFFSET + ASSUMED_MAIN_KV_DATA_BYTES == ASSUMED_MAIN_KV_SCALE_OFFSET
);
const _: () = assert!(
    ASSUMED_MAIN_KV_SCALE_OFFSET + ASSUMED_MAIN_KV_SCALE_BYTES == ASSUMED_MAIN_KV_ENTRY_BYTES
);
const _: () = assert!(
    ASSUMED_MAIN_KV_ENTRY_BYTES + ASSUMED_INDEXER_K_ENTRY_BYTES == ASSUMED_GLOBAL_ENTRY_BYTES
);
const _: () = assert!(ASSUMED_INDEXER_K_DATA_OFFSET == ASSUMED_MAIN_KV_ENTRY_BYTES);
const _: () = assert!(
    ASSUMED_INDEXER_K_DATA_OFFSET + ASSUMED_INDEXER_K_DATA_BYTES == ASSUMED_INDEXER_K_SCALE_OFFSET
);
const _: () = assert!(
    ASSUMED_INDEXER_K_SCALE_OFFSET + ASSUMED_INDEXER_K_SCALE_BYTES == ASSUMED_GLOBAL_ENTRY_BYTES
);
const _: () =
    assert!(ASSUMED_SWA_SLOT_DATA_BYTES + ASSUMED_SWA_SLOT_SCALE_BYTES == ASSUMED_SWA_SLOT_BYTES);
const _: () = assert!(ASSUMED_SWA_SLOT_DIM == ASSUMED_SWA_SLOT_DATA_BYTES);
const _: () = assert!(ASSUMED_SWA_SLOT_DIM.is_multiple_of(ASSUMED_SWA_SCALE_GROUP));
const _: () = assert!(
    ASSUMED_HSI_BLOCKS * ASSUMED_HSI_BLOCK_TOKENS * ASSUMED_INDEXER_K_DIM * 2
        == ASSUMED_HSI_INDEX_KEY_BYTES
);

// SWA_SLOT_DIM is imported for the assertion above only.
const _: () = assert!(ASSUMED_SWA_SLOT_DIM == SWA_SLOT_DIM);

/// Parses an integer define out of a CUDA source, for the kernel half of the same guard.
///
/// Only a plain decimal with an optional u suffix is accepted; a define the guard cannot parse
/// returns None, which the guard test treats as a failure. That keeps an expression-valued define
/// from silently passing the check.
#[must_use]
pub fn kernel_define_in(source: &str, name: &str) -> Option<u64> {
    let prefix = format!("#define {name} ");
    source.lines().find_map(|line| {
        let rest = line.trim().strip_prefix(prefix.as_str())?;
        let raw = rest.split_whitespace().next()?;
        let digits = raw.strip_suffix('u').unwrap_or(raw);
        digits.parse().ok()
    })
}

/// Parses a define from the shared CUDA common header (sparkflash_kernels::m4_abi::COMMON_HEADER).
#[must_use]
pub fn kernel_common_define(name: &str) -> Option<u64> {
    kernel_define_in(COMMON_HEADER, name)
}

/// Parses a define from the flash MLA kernel's own source.
#[must_use]
pub fn kernel_flash_mla_define(name: &str) -> Option<u64> {
    kernel_define_in(FLASH_MLA_CSA2, name)
}
