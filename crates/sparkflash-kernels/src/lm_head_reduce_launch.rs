//! Typed launch capability of the certified head's device reduction
//! (kernels/lm_head_reduce_sm121.cu).
//!
//! The argument array is the kernel declaration's parameter list in order - four per-rank logit
//! pointers, four per-rank certified-flag pointers, the eight report pointers and the three
//! ABI-uniform trace words - so a launch cannot swap the shard buffers for the report, or put the
//! top-5 where the block array goes. The declaration is pinned here against the kernel source
//! (SOURCE): PARAMS is the pinned name and type of every parameter and declaration_matches is the
//! check, with a mutation test proving it is not vacuous.
//!
//! Geometry. One launch reduces all four shards: a single block of BLOCK.0 threads, eight warps x
//! 16 of the 128 m16 blocks. There is no grid input to get wrong and no grid-wide rendezvous; the
//! harness calls the launcher once and reads the small report back.
//!
//! No cubin pin. The device chain's catalog owns the cubin's bytes and digest, as for the other
//! launch modules here; the harness verifies the pinned artifact through baseline::pin.

use crate::c_decl::{self, KernelDecl};
use crate::cuda_driver::{Cuda, Deviceptr, Kernel, KernelId, Module};

/// The kernel symbol in the cubin.
pub const KERNEL: &str = "lm_head_reduce_sm121";
/// The kernel source the declaration pin is read from.
pub const SOURCE: &str = include_str!("../../../kernels/lm_head_reduce_sm121.cu");
/// The number of argument words a launch fills.
pub const ARG_WORDS: usize = 19;

/// The four tensor-parallel shards the certified head writes.
pub const RANKS: usize = 4;
/// The per-rank vocabulary of the released head.
pub const VOCAB_PER_RANK: usize = 32_320;
/// The released vocabulary.
pub const VOCAB: usize = 129_280;
/// One m16 block: the certified screen's 1,024-entry chunk.
pub const CHUNK: usize = 1_024;
/// The m16 blocks per rank (the last one has 576 entries).
pub const CHUNKS_PER_RANK: usize = 32;
/// The m16 blocks over all four ranks, rank-major.
pub const BLOCKS: usize = RANKS * CHUNKS_PER_RANK;
/// The top-5 entries the kernel reports.
pub const TOP: usize = 5;
/// 256 threads: eight warps, one warp per block at a time.
pub const BLOCK: (u32, u32, u32) = (256, 1, 1);

/// The argmax entry is certified, so its stored value is a bound, not its exact value: the token is
/// diagnostic and the caller must not claim it (refuse rather than assert). A sound certificate
/// makes this impossible, because a certified entry satisfies bound < running maximum.
pub const STATUS_WINNER_CERTIFIED: u32 = 0x1;
/// The top-5 set is exact and may be claimed: set only when no entry was certified. This is the
/// device-side half of m16's claims_top5 (the host still requires a reference before it claims the
/// set).
pub const STATUS_TOP5_CLAIMED: u32 = 0x2;
/// At least one entry was certified (informational; the top-5 is not claimed).
pub const STATUS_ANY_CERTIFIED: u32 = 0x4;

/// Every status bit the kernel defines, for a caller that wants to reject an unknown bit.
pub const STATUS_BITS: [u32; 3] = [
    STATUS_WINNER_CERTIFIED,
    STATUS_TOP5_CLAIMED,
    STATUS_ANY_CERTIFIED,
];

/// The declaration's parameters in order: (name, base type, pointer, pointee const).
pub const PARAMS: [(&str, &str, bool, bool); ARG_WORDS] = [
    ("logits_r0", "float", true, true),
    ("logits_r1", "float", true, true),
    ("logits_r2", "float", true, true),
    ("logits_r3", "float", true, true),
    ("certified_r0", "uint8_t", true, true),
    ("certified_r1", "uint8_t", true, true),
    ("certified_r2", "uint8_t", true, true),
    ("certified_r3", "uint8_t", true, true),
    ("token", "unsigned int", true, false),
    ("value", "float", true, false),
    ("block_max", "float", true, false),
    ("block_argmax", "uint32_t", true, false),
    ("top5_value", "float", true, false),
    ("top5_index", "uint32_t", true, false),
    ("status", "unsigned int", true, false),
    ("certified_count", "unsigned int", true, false),
    ("trace_ring", "GpuTraceRing", true, false),
    ("layer_id", "uint32_t", false, false),
    ("step_seq", "uint64_t", false, false),
];

/// The parameter index of the first logit pointer.
pub const LOGITS_INDEX: usize = 0;
/// The parameter index of the first certified-flag pointer.
pub const CERTIFIED_INDEX: usize = 4;
/// The parameter index of the token word.
pub const TOKEN_INDEX: usize = 8;
/// The parameter index of the argmax value.
pub const VALUE_INDEX: usize = 9;
/// The parameter index of the 128 block maxima.
pub const BLOCK_MAX_INDEX: usize = 10;
/// The parameter index of the 128 block argmax entries.
pub const BLOCK_ARGMAX_INDEX: usize = 11;
/// The parameter index of the five top-5 values.
pub const TOP5_VALUE_INDEX: usize = 12;
/// The parameter index of the five top-5 entries.
pub const TOP5_INDEX_INDEX: usize = 13;
/// The parameter index of the status word.
pub const STATUS_INDEX: usize = 14;
/// The parameter index of the total certified count.
pub const CERTIFIED_COUNT_INDEX: usize = 15;
/// The parameter index of the trace ring.
pub const TRACE_INDEX: usize = 16;
/// The parameter index of the layer id.
pub const LAYER_INDEX: usize = 17;
/// The parameter index of the step sequence.
pub const STEP_INDEX: usize = 18;

/// Parses the kernel declaration out of the pinned source.
#[must_use]
pub fn declaration() -> KernelDecl {
    c_decl::kernel_declaration(SOURCE, KERNEL)
}

/// Whether a parsed declaration has PARAMS' shape and order.
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
pub const KERNEL_ID: KernelId = KernelId::LmHeadReduce;

/// The one launch's grid: exactly one block, because the reduction finalises in that block.
pub const GRID: (u32, u32, u32) = (1, 1, 1);
/// K8 batched reduction symbol: the same block reduces six rows in one launch.
pub const BATCH_KERNEL: &str = "lm_head_reduce_batched_sm121";
/// The verifier batch is six rows.
pub const BATCH_ROWS: u32 = 6;

/// One launch's arguments, in the kernel declaration's order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LmHeadReduceParams([u64; ARG_WORDS]);

impl LmHeadReduceParams {
    /// Builds the argument array with the kernel's own parameter names, in order.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        logits: [Deviceptr; RANKS],
        certified: [Deviceptr; RANKS],
        token: Deviceptr,
        value: Deviceptr,
        block_max: Deviceptr,
        block_argmax: Deviceptr,
        top5_value: Deviceptr,
        top5_index: Deviceptr,
        status: Deviceptr,
        certified_count: Deviceptr,
        trace_ring: u64,
        layer_id: u32,
        step_seq: u64,
    ) -> Self {
        Self([
            logits[0].0,
            logits[1].0,
            logits[2].0,
            logits[3].0,
            certified[0].0,
            certified[1].0,
            certified[2].0,
            certified[3].0,
            token.0,
            value.0,
            block_max.0,
            block_argmax.0,
            top5_value.0,
            top5_index.0,
            status.0,
            certified_count.0,
            trace_ring,
            u64::from(layer_id),
            step_seq,
        ])
    }

    /// The parameter words a launch passes, the shape cuLaunchKernel takes.
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        &self.0
    }

    /// One argument word by PARAMS index.
    #[must_use]
    pub fn word(&self, index: usize) -> u64 {
        self.0[index]
    }
}

/// Loads the K8 batched reduction symbol by name from the same cubin.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain [BATCH_KERNEL].
pub fn load_batched(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_lm_head_reduce_batched(cubin)
}

/// Loads the kernel from a cubin, resolving the pinned symbol.
///
/// # Errors
/// Returns the driver's error when the cubin does not contain KERNEL.
pub fn load(cuda: &Cuda, cubin: &[u8]) -> Result<(Module, Kernel), String> {
    cuda.load_kernel(cubin, KERNEL_ID)
}

/// Launches the reduction once.
///
/// # Errors
/// Returns the driver's error.
pub fn launch(cuda: &Cuda, kernel: Kernel, params: &LmHeadReduceParams) -> Result<(), String> {
    cuda.launch(kernel, GRID, BLOCK, params.as_slice())
}

/// Launches the K8 batched reduction once over six rows.
///
/// The 19-word ABI is unchanged; every pointer is batch-major, with `token`, `value`, `status`
/// and `certified_count` [6] and the block/top-5 arrays [6][...].
///
/// # Errors
/// Returns the driver's error; the kernel itself is one block and has no grid launch risk.
pub fn launch_batched(
    cuda: &Cuda,
    kernel: Kernel,
    params: &LmHeadReduceParams,
) -> Result<(), String> {
    cuda.launch(kernel, GRID, BLOCK, params.as_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c_decl::Consts;
    use crate::m4_abi;

    fn params() -> LmHeadReduceParams {
        LmHeadReduceParams::new(
            [
                Deviceptr(0x1000),
                Deviceptr(0x2000),
                Deviceptr(0x3000),
                Deviceptr(0x4000),
            ],
            [
                Deviceptr(0x5000),
                Deviceptr(0x6000),
                Deviceptr(0x7000),
                Deviceptr(0x8000),
            ],
            Deviceptr(0x9000),
            Deviceptr(0xa000),
            Deviceptr(0xb000),
            Deviceptr(0xc000),
            Deviceptr(0xd000),
            Deviceptr(0xe000),
            Deviceptr(0xf000),
            Deviceptr(0x10000),
            0x11000,
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
            "uint32_t *__restrict__ block_argmax",
            "uint32_t * block_argmax",
        );
        assert_ne!(mutated, SOURCE, "the mutation must change the text");
        let decl = c_decl::kernel_declaration(&mutated, KERNEL);
        assert!(
            !declaration_matches(&decl),
            "the qualifier pin is vacuous: {decl:?}"
        );
    }

    #[test]
    fn the_batched_reduction_is_six_rows_in_the_same_one_block() {
        assert_eq!(BATCH_ROWS, 6);
        assert_eq!(GRID, (1, 1, 1));
        assert_eq!(BLOCK.0, 256);
        assert_ne!(BATCH_KERNEL, KERNEL);
    }

    #[test]
    fn the_argument_array_is_the_declaration_order() {
        let p = params();
        assert_eq!(
            p.as_slice(),
            [
                0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x8000, 0x9000, 0xa000,
                0xb000, 0xc000, 0xd000, 0xe000, 0xf000, 0x10000, 0x11000, 7, 11
            ]
        );
        for (index, (name, _, _, _)) in PARAMS.iter().enumerate() {
            assert_eq!(p.word(index), p.as_slice()[index], "param {name}");
        }
        assert_eq!(p.as_slice().len(), ARG_WORDS);
    }

    #[test]
    fn the_geometry_is_the_kernels() {
        let table = constants();
        assert_eq!(table.value("SPARK_LM_REDUCE_RANKS"), RANKS as i128);
        assert_eq!(
            table.value("SPARK_LM_REDUCE_VOCAB_PER_RANK"),
            VOCAB_PER_RANK as i128
        );
        assert_eq!(table.value("SPARK_LM_REDUCE_VOCAB"), VOCAB as i128);
        assert_eq!(table.value("SPARK_LM_REDUCE_CHUNK"), CHUNK as i128);
        assert_eq!(
            table.value("SPARK_LM_REDUCE_CHUNKS_PER_RANK"),
            CHUNKS_PER_RANK as i128
        );
        assert_eq!(table.value("SPARK_LM_REDUCE_BLOCKS"), BLOCKS as i128);
        assert_eq!(table.value("SPARK_LM_REDUCE_THREADS"), i128::from(BLOCK.0));
        assert_eq!(table.value("SPARK_LM_REDUCE_TOP"), TOP as i128);
        assert_eq!(VOCAB, RANKS * VOCAB_PER_RANK);
        assert_eq!(BLOCKS, 128);
        assert_eq!(GRID, (1, 1, 1));
        assert_eq!(
            table.value("SPARK_M4_VOCAB_PER_RANK"),
            VOCAB_PER_RANK as i128
        );
    }

    #[test]
    fn the_status_bits_are_the_pinned_contract() {
        let table = constants();
        assert_eq!(
            table.value("SPARK_LM_REDUCE_STATUS_WINNER_CERTIFIED"),
            i128::from(STATUS_WINNER_CERTIFIED)
        );
        assert_eq!(
            table.value("SPARK_LM_REDUCE_STATUS_TOP5_CLAIMED"),
            i128::from(STATUS_TOP5_CLAIMED)
        );
        assert_eq!(
            table.value("SPARK_LM_REDUCE_STATUS_ANY_CERTIFIED"),
            i128::from(STATUS_ANY_CERTIFIED)
        );
        assert_eq!(STATUS_BITS, [0x1, 0x2, 0x4]);
    }

    #[test]
    fn the_partial_block_is_the_m16_layout() {
        // The m16 screen's block 31 covers entries [31 * 1024, 32,320), i.e. 576 entries.
        let first = CHUNKS_PER_RANK - 1;
        let covered = first * CHUNK;
        assert_eq!(VOCAB_PER_RANK - covered, 576);
        assert_eq!(covered + 576, VOCAB_PER_RANK);
    }
}
