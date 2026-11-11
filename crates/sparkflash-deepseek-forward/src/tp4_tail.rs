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

//! The serving tail of the four-rank forward loop: final RMSNorm, the certified head shard, the
//! exact logit all-gather and the argmax.
//!
//! What this module owns, in the order the loop will call it:
//!
//! 1. TailStage::open loads the two tail kernels from the staged cubin directory, allocates the
//!    per-rank buffers, and derives the certified head's w_block_max side tensor ONCE from the
//!    bound arena's head shard (kernel-lead's direction, 2026-10-20: init-time derivation from the
//!    arena, never a per-token weight H2D, and TailBind is not widened for it).
//! 2. TailStage::screen normalises the hidden state of one position with rmsnorm_sm121 against the
//!    arena's final-norm weight and screens the rank's own 32,320 vocabulary rows with
//!    cert_lm_head_sm121, returning the rank's f32 logits.
//! 3. gather_logits moves those logits to every rank EXACTLY: each f32 is split into its two u16
//!    halves, the words travel through the existing tcp all-gather (pure word movement, no
//!    arithmetic, so no rounding), and every rank reassembles the full 129,280-entry vector and
//!    takes the argmax with the lowest-id tie break.
//!
//! The tail is deliberately separate from the layer body: it is the last stage of every position
//! and the easiest one to validate against the host twin (lm_head_ref, rmsnorm_ref), so it is
//! written and host-tested first while the arena sequence is still being wired.
//!
//! No token is ever claimed here: the caller decides whether the screened logits came from a real
//! hidden state, and only the loop's envelope may turn a Logits value into a token id.

use std::path::{Path, PathBuf};

use sparkflash_comm::transport::Transport;

/// The hidden size of the released configuration.
pub const HIDDEN: usize = 5120;
/// The full vocabulary of the released configuration.
pub const VOCAB: usize = 129_280;
/// The per-rank head shard width (VOCAB / 4).
pub const VOCAB_PER_RANK: usize = 32_320;
/// The certified head's screening blocks per output row (HIDDEN / 128).
pub const BLOCKS_PER_ENTRY: usize = 40;
/// The certified head's chunk width (one block per chunk).
pub const CHUNK: u32 = sparkflash_kernels::cert_lm_head_launch::CHUNK;
/// The per-rank screening chunk count (the certified-head side tensors' length).
pub const CHUNKS_PER_RANK: usize = VOCAB_PER_RANK.div_ceil(CHUNK as usize);
/// The f32 logits one rank produces.
pub const LOGITS_BYTES_PER_RANK: usize = VOCAB_PER_RANK * 4;
/// The BF16 bytes of one rank's derived head block-maxima side tensor.
pub const W_BLOCK_MAX_BYTES: usize = VOCAB_PER_RANK * BLOCKS_PER_ENTRY * 2;

/// The bytes of one rank's side-buffer gather payload: the certified mask, the certified count,
/// the per-chunk block maxima and the per-chunk block argmax.
pub const SIDE_BYTES: usize = VOCAB_PER_RANK + 4 + CHUNKS_PER_RANK * 4 * 2;

/// The kernel symbols the tail loads, in load order.
pub const TAIL_CUBINS: [&str; 2] = ["rmsnorm_sm121", "cert_lm_head_sm121"];

/// Splits f32 values into the two u16 words each that the all-gather moves, in place order
/// (high half first, low half second) per value.
#[must_use]
pub fn split_f32_words(values: &[f32]) -> Vec<u16> {
    let mut words = Vec::with_capacity(values.len() * 2);
    for value in values {
        let bits = value.to_bits();
        words.push((bits >> 16) as u16);
        words.push(bits as u16);
    }
    words
}

/// Reassembles f32 values from the word stream split_f32_words produces, bit for bit.
///
/// # Errors
/// An odd word count is not a word stream this module wrote.
pub fn join_f32_words(words: &[u16]) -> Result<Vec<f32>, String> {
    if !words.len().is_multiple_of(2) {
        return Err(format!(
            "exact logit gather: {} u16 words is not a stream of f32 halves",
            words.len()
        ));
    }
    let mut values = Vec::with_capacity(words.len() / 2);
    for index in (0..words.len()).step_by(2) {
        let bits = (u32::from(words[index]) << 16) | u32::from(words[index + 1]);
        values.push(f32::from_bits(bits));
    }
    Ok(values)
}

/// One position's screened, all-gathered logits.
#[derive(Debug, Clone, PartialEq)]
pub struct Logits {
    /// The full vocabulary in rank order (rank 0's 32,320 rows, then rank 1's, ...).
    pub full: Vec<f32>,
    /// The argmax id over the full vector, lowest id on ties.
    pub argmax: u32,
    /// The top-1 value.
    pub top1: f32,
    /// The top-2 value (equal to top1 when the vocabulary has a single entry).
    pub top2: f32,
    /// top1 - top2.
    pub margin: f32,
}

/// Picks the argmax with the lowest-id tie break and the top-2 margin from a full vector.
///
/// # Errors
/// An empty vector has no argmax.
pub fn argmax_of(full: Vec<f32>) -> Result<Logits, String> {
    let mut best: Option<(usize, f32)> = None;
    let mut runner_up = f32::NEG_INFINITY;
    for (index, &value) in full.iter().enumerate() {
        match best {
            None => best = Some((index, value)),
            Some((_, current)) if value > current => {
                runner_up = current;
                best = Some((index, value));
            }
            Some(_) => {
                if value > runner_up {
                    runner_up = value;
                }
            }
        }
    }
    let Some((index, top1)) = best else {
        return Err("argmax of an empty logit vector".to_string());
    };
    Ok(Logits {
        full,
        argmax: index as u32,
        top1,
        top2: runner_up,
        margin: top1 - runner_up,
    })
}

/// Assembles rank-ordered per-rank chunks into the full vector and picks the argmax.
///
/// # Errors
/// A chunk that is not VOCAB_PER_RANK long, or a chunk set that is not the full vocabulary.
pub fn assemble_chunks(chunks: &[Vec<f32>]) -> Result<Logits, String> {
    let mut full = Vec::with_capacity(VOCAB);
    for (rank, chunk) in chunks.iter().enumerate() {
        if chunk.len() != VOCAB_PER_RANK {
            return Err(format!(
                "exact logit gather: rank {rank} produced {} logits, the plan's shard is {VOCAB_PER_RANK}",
                chunk.len()
            ));
        }
        full.extend_from_slice(chunk);
    }
    if full.len() != VOCAB {
        return Err(format!(
            "exact logit gather: {} ranks cover {} logits, the vocabulary is {VOCAB}",
            chunks.len(),
            full.len()
        ));
    }
    argmax_of(full)
}

/// All-gathers one rank's f32 logits exactly and returns the full vocabulary and the argmax.
///
/// Every rank must call this with the same seq; the words travel through the existing bf16
/// all-gather, which moves them verbatim.
///
/// # Errors
/// A local shard of the wrong width, a collective failure, or an empty reassembled vector.
pub fn gather_logits(transport: &dyn Transport, local: &[f32], seq: u64) -> Result<Logits, String> {
    if local.len() != VOCAB_PER_RANK {
        return Err(format!(
            "exact logit gather: this rank holds {} logits, the plan's shard is {VOCAB_PER_RANK}",
            local.len()
        ));
    }
    let words = split_f32_words(local);
    let world = transport.world_size();
    let mut gathered = vec![0u16; world * words.len()];
    transport
        .all_gather_bf16(&words, &mut gathered, seq)
        .map_err(|error| format!("exact logit all-gather: {error}"))?;
    let chunks: Vec<Vec<f32>> = gathered
        .chunks(words.len())
        .map(join_f32_words)
        .collect::<Result<_, _>>()?;
    assemble_chunks(&chunks)
}

/// The cubin path the tail stage expects for a symbol in a staged kernel directory.
#[must_use]
pub fn cubin_path(dir: &Path, symbol: &str) -> PathBuf {
    dir.join(format!("{symbol}.cubin"))
}

#[cfg(feature = "cuda")]
mod device {
    use super::{
        CHUNKS_PER_RANK, HIDDEN, LOGITS_BYTES_PER_RANK, TAIL_CUBINS, VOCAB_PER_RANK,
        W_BLOCK_MAX_BYTES, cubin_path, join_f32_words, split_f32_words,
    };
    use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, Module};
    use sparkflash_kernels::{cert_lm_head_launch, head_screen_ref, lm_head_ref, rmsnorm_launch};
    use std::cell::Cell;
    use std::path::{Path, PathBuf};

    /// The per-rank device buffers one tail stage owns.
    #[derive(Debug, Clone, Copy)]
    pub struct TailBuffers {
        /// One position's hidden state, BF16.
        pub hidden: Deviceptr,
        /// The normalised hidden state, BF16.
        pub normed: Deviceptr,
        /// The certified head's derived block-maxima side tensor, BF16 [32320][40].
        pub w_block_max: Deviceptr,
        /// This rank's f32 logits [32320].
        pub logits: Deviceptr,
        /// The certified mask [32320].
        pub certified: Deviceptr,
        /// The certified count.
        pub certified_count: Deviceptr,
        /// The per-chunk block maxima [32] f32.
        pub block_max: Deviceptr,
        /// The per-chunk block argmax [32] u32.
        pub block_argmax: Deviceptr,
    }

    /// What the device tail left in its side buffers for the evidence record.
    #[derive(Debug, Clone, PartialEq)]
    pub struct TailOutputs {
        /// The normalised hidden state the certified head consumed, BF16 (the normed buffer).
        pub normed_bf16: Vec<u8>,
        /// The certified mask: VOCAB_PER_RANK bytes, 1 = the entry was pruned by the block bound.
        pub certified: Vec<u8>,
        /// The number of certified entries, as the kernel wrote it.
        pub certified_count: u32,
        /// The per-chunk block maxima the kernel screened with (CHUNKS_PER_RANK f32).
        pub block_max: Vec<f32>,
        /// The per-chunk block argmax entries (CHUNKS_PER_RANK u32, the lowest on a block tie).
        pub block_argmax: Vec<u32>,
    }

    /// Whether the rows head is screened when the profile does not say
    /// (`diagnostics.head_screen`).
    const HEAD_SCREEN_DEFAULT: bool = true;
    /// The tiles of int8 codes a rank's head slice fills (kernels/opt_rows_sm121.cu, OHS_TILE).
    const SCREEN_TILES: usize = VOCAB_PER_RANK.div_ceil(head_screen_ref::TILE);
    /// The most hidden rows a step screens: the rows head's R_MAX.
    const SCREEN_ROWS: usize = 64;
    /// The rows the self test screens: two row blocks, the second partly filled.
    const SCREEN_TEST_ROWS: usize = 41;

    /// opt_rows_lm_head's noise arguments, in its order: the temperature's inverse as f32 bits (zero
    /// for greedy rows), the seed, the first row's position, the rank's first id and the per-row
    /// parameter buffer (zero when the scalars cover every row).
    pub(crate) type HeadNoise = [u64; 5];

    /// What the checked steps saw (`diagnostics.head_screen_check`).
    #[derive(Debug, Clone, Copy, Default)]
    struct ScreenTally {
        steps: u64,
        rows: u64,
        scored: u64,
        most: u32,
    }

    /// The rows head's screen (kernels/opt_rows_sm121.cu, opt_head_screen_*): an int8 copy of
    /// the rank's head slice whose integer dot products bound every logit from both sides, so
    /// opt_rows_lm_head's arithmetic runs only over the entries a bound cannot rule out. The
    /// arg-max and its value are that kernel's bits. One a rank, built once when the tail opens.
    pub struct HeadScreen {
        modules: Vec<Module>,
        rows: Kernel,
        mma: Kernel,
        exact: Kernel,
        /// The codes, `[SCREEN_TILES][160][128][32]` int8.
        codes: Deviceptr,
        /// `[SCREEN_TILES * 128][4]` f32: each entry's scale and its two error coefficients.
        coef: Deviceptr,
        /// A step's hidden rows as two int8 digits each, `[2 * SCREEN_ROWS][HIDDEN]`.
        act: Deviceptr,
        /// `[SCREEN_ROWS][4]` f32: each row's scale and its two norms.
        meta: Deviceptr,
        /// `[SCREEN_ROWS][VOCAB_PER_RANK]` f32: every logit's upper bound.
        upper: Deviceptr,
        /// `[SCREEN_ROWS][SCREEN_TILES]` f32: each tile's best lower bound.
        tile_lo: Deviceptr,
        /// `[SCREEN_ROWS]` u32: the entries scored for each row (checked steps only).
        counts: Deviceptr,
        /// Whether every screened step also runs the full head and is compared with it.
        pub check: bool,
        tally: Cell<ScreenTally>,
    }

    impl HeadScreen {
        /// Loads the screen's kernels, builds the table from the bound head slice and waits for
        /// it (a step's screen reads the codes before it waits on its predecessor, so they have
        /// to be there before the first step). `shard` is the slice's bytes, which the caller
        /// has already read; with `check` the device's table and a screen of made-up rows are
        /// compared with the host reference before anything is served.
        fn open(
            cuda: &Cuda,
            kernel_dir: &Path,
            head: Deviceptr,
            shard: &[u8],
            check: bool,
            allocated: &mut usize,
        ) -> Result<Self, String> {
            let (cubin, _sha) =
                crate::tp4_forward::cached_cubin(&kernel_dir.join("opt_rows_sm121.cubin"))
                    .map_err(|error| format!("head screen: {error}"))?;
            let mut modules = Vec::new();
            let mut load = |name: &'static str| -> Result<Kernel, String> {
                let (module, kernel) = cuda.load_named(&cubin, name)?;
                modules.push(module);
                Ok(kernel)
            };
            let build = load("opt_head_screen_build")?;
            let rows = load("opt_head_screen_rows")?;
            let mma = load("opt_head_screen_mma")?;
            let exact = load("opt_head_screen_exact")?;
            let mut alloc = |bytes: usize, what: &str| {
                cuda.alloc(bytes, allocated)
                    .map_err(|error| format!("head screen {what}: {error}"))
            };
            let entries = SCREEN_TILES * head_screen_ref::TILE;
            let screen = Self {
                codes: alloc(head_screen_ref::table_bytes(VOCAB_PER_RANK), "codes")?,
                coef: alloc(entries * 16, "coefficients")?,
                act: alloc(2 * SCREEN_ROWS * HIDDEN, "digits")?,
                meta: alloc(SCREEN_ROWS * 16, "row scales")?,
                upper: alloc(SCREEN_ROWS * VOCAB_PER_RANK * 4, "upper bounds")?,
                tile_lo: alloc(SCREEN_ROWS * SCREEN_TILES * 4, "tile bounds")?,
                counts: alloc(SCREEN_ROWS * 4, "counts")?,
                modules,
                rows,
                mma,
                exact,
                check,
                tally: Cell::new(ScreenTally::default()),
            };
            cuda.launch(
                build,
                (entries as u32, 1, 1),
                (256, 1, 1),
                &[head.0, screen.codes.0, screen.coef.0, VOCAB_PER_RANK as u64],
            )?;
            // The read waits for the build. An entry the bound does not cover (a value that is
            // not finite, or squares that sum past the kernel's cut) is always scored, which is
            // still exact; a head holding one is not a head this path was measured on, so say so
            // rather than serve it.
            let coef = screen.read_f32(cuda, screen.coef, entries * 4)?;
            if let Some(entry) = (0..VOCAB_PER_RANK).find(|&entry| {
                coef[entry * 4..entry * 4 + 3]
                    .iter()
                    .any(|value| !value.is_finite())
            }) {
                return Err(format!(
                    "head screen: entry {entry} of this rank's head slice is not finite or too \
                     large for the screen's bound; serve it with SPARKFLASH_HEAD_SCREEN=0"
                ));
            }
            // The self test runs whenever the screen is in use, not only when the diagnostic
            // switch is set. The screen decides which logits the certified head is allowed not to
            // compute, so an under-estimated bound silently skips the true arg-max and changes a
            // token with no other symptom. The test is one grid over this rank's slice and a host
            // comparison of forty-one rows, paid once at open; a diagnostic switch that defaults to
            // off cannot be the only thing standing between the default serving path and an
            // unsound bound.
            screen.self_test(cuda, shard, &coef).map_err(|error| {
                format!(
                    "{error} (SPARKFLASH_HEAD_SCREEN_CHECK=1 runs this test on its own; SPARKFLASH_HEAD_SCREEN=0 serves the full head)"
                )
            })?;
            // What the screen costs, where an operator watching memory will find it: the int8
            // code table is the large part (~158 MiB a rank at 32 320 x 5 120), and until this line
            // existed nothing in the logs or the docs said so.
            eprintln!(
                "sparkflash serve: head screen on over {} x {} int8 codes: {} bytes of codes, {} of bounds for {} rows; self test passed{}",
                VOCAB_PER_RANK,
                HIDDEN,
                head_screen_ref::table_bytes(VOCAB_PER_RANK),
                SCREEN_ROWS * (VOCAB_PER_RANK + SCREEN_TILES) * 4,
                SCREEN_ROWS,
                if check {
                    " (diagnostic check also armed)"
                } else {
                    ""
                }
            );
            Ok(screen)
        }

        fn read_f32(&self, cuda: &Cuda, from: Deviceptr, count: usize) -> Result<Vec<f32>, String> {
            let mut bytes = vec![0u8; count * 4];
            cuda.copy_dtoh(&mut bytes, &from)
                .map_err(|error| format!("head screen read: {error}"))?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect())
        }

        /// Queues the screen of `(hidden, n)`, `n` rows of bf16 `[n][HIDDEN]`, and the exact pass over
        /// what it keeps, which writes the rows' chunk maxima where opt_rows_lm_head writes them.
        /// `noise` is that kernel's `(inv_t bits, seed, first position, id base, row parameters)`,
        /// all zero for a greedy step. With `count` the entries scored for each row are left in
        /// the count buffer.
        pub(crate) fn launch(
            &self,
            cuda: &Cuda,
            (hidden, n): (u64, usize),
            head: Deviceptr,
            (block_max, block_arg): (u64, u64),
            noise: HeadNoise,
            count: bool,
        ) -> Result<(), String> {
            if n == 0 || n > SCREEN_ROWS {
                return Err(format!(
                    "head screen: {n} rows are outside 1..={SCREEN_ROWS}"
                ));
            }
            cuda.launch(
                self.rows,
                (n as u32, 1, 1),
                (256, 1, 1),
                &[hidden, self.act.0, self.meta.0],
            )?;
            cuda.launch(
                self.mma,
                (n.div_ceil(32) as u32, SCREEN_TILES as u32, 1),
                (256, 1, 1),
                &[
                    self.act.0,
                    self.meta.0,
                    self.codes.0,
                    self.coef.0,
                    self.upper.0,
                    self.tile_lo.0,
                    VOCAB_PER_RANK as u64,
                    n as u64,
                    noise[0],
                    noise[1],
                    noise[2],
                    noise[3],
                    noise[4],
                ],
            )?;
            if count {
                cuda.copy_htod(&self.counts, &[0u8; SCREEN_ROWS * 4])
                    .map_err(|error| format!("head screen count reset: {error}"))?;
            }
            cuda.launch(
                self.exact,
                (n.div_ceil(4) as u32, CHUNKS_PER_RANK as u32, 1),
                (512, 1, 1),
                &[
                    hidden,
                    head.0,
                    self.upper.0,
                    self.tile_lo.0,
                    block_max,
                    block_arg,
                    VOCAB_PER_RANK as u64,
                    n as u64,
                    SCREEN_TILES as u64,
                    if count { self.counts.0 } else { 0 },
                    noise[0],
                    noise[1],
                    noise[2],
                    noise[3],
                    noise[4],
                ],
            )
        }

        /// Adds a checked step's rows to the tally (the entries the exact pass scored for each)
        /// and reports it every so often: the check's evidence is its count of compared rows.
        pub(crate) fn tally_checked(&self, cuda: &Cuda, n: usize) -> Result<(), String> {
            let mut bytes = vec![0u8; n * 4];
            cuda.copy_dtoh(&mut bytes, &self.counts)
                .map_err(|error| format!("head screen count read: {error}"))?;
            let mut tally = self.tally.get();
            tally.steps += 1;
            for word in bytes.as_chunks::<4>().0 {
                let scored = u32::from_le_bytes(*word);
                tally.rows += 1;
                tally.scored += u64::from(scored);
                tally.most = tally.most.max(scored);
            }
            self.tally.set(tally);
            if tally.steps.is_multiple_of(1024) {
                self.report();
            }
            Ok(())
        }

        fn report(&self) {
            let tally = self.tally.get();
            if tally.rows != 0 {
                eprintln!(
                    "head screen check: {} steps and {} rows agree with the full head; {:.1} of \
                     {VOCAB_PER_RANK} entries scored a row, {} at most",
                    tally.steps,
                    tally.rows,
                    tally.scored as f64 / tally.rows as f64,
                    tally.most
                );
            }
        }

        /// Compares what the device built and what its screen computes with the host reference
        /// (sparkflash_kernels::head_screen_ref): every entry's codes and scale to the bit and
        /// its coefficients on the safe side, then, for rows that fill two row blocks (plain rows
        /// at several scales, a sparse one and three the bound does not cover), every digit to
        /// the bit and every upper bound and tile bound against the exact integer dot product.
        fn self_test(&self, cuda: &Cuda, shard: &[u8], coef: &[f32]) -> Result<(), String> {
            let refuse = |what: String| Err(format!("head screen self test: {what}"));
            let words = |bytes: &[u8]| -> Vec<u16> {
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|word| u16::from_le_bytes(*word))
                    .collect()
            };
            // Within `slack` above the exact value, and never below it.
            let safe = |device: f32, exact: f64, slack: f64| {
                f64::from(device) >= exact && f64::from(device) <= exact * slack + 2e-17
            };
            let mut device_codes = vec![0u8; head_screen_ref::table_bytes(VOCAB_PER_RANK)];
            cuda.copy_dtoh(&mut device_codes, &self.codes)
                .map_err(|error| format!("head screen codes read: {error}"))?;
            let mut codes = vec![0i8; VOCAB_PER_RANK * HIDDEN];
            for entry in 0..VOCAB_PER_RANK {
                let host = head_screen_ref::entry_codes(&words(
                    &shard[entry * HIDDEN * 2..][..HIDDEN * 2],
                ));
                let c = &coef[entry * 4..entry * 4 + 4];
                if c[0].to_bits() != host.scale.to_bits()
                    || !safe(c[1], host.a_floor(), 1.02)
                    || !safe(c[2], host.coded_norm, 1.02)
                {
                    return refuse(format!(
                        "entry {entry}: the device holds scale {} and coefficients {} {}, the \
                         reference {} {} {}",
                        c[0],
                        c[1],
                        c[2],
                        host.scale,
                        host.a_floor(),
                        host.coded_norm
                    ));
                }
                for block in 0..head_screen_ref::K_BLOCKS {
                    let at = head_screen_ref::code_offset(entry, block * 32);
                    let device = &device_codes[at..at + 32];
                    let expect = &host.codes[block * 32..block * 32 + 32];
                    if device.iter().zip(expect).any(|(&d, &e)| d as i8 != e) {
                        return refuse(format!(
                            "entry {entry}: the codes of K block {block} differ"
                        ));
                    }
                }
                codes[entry * HIDDEN..(entry + 1) * HIDDEN].copy_from_slice(&host.codes);
            }
            drop(device_codes);

            // The made-up rows: a reproducible stream at scales 1/16 .. 16, then the special ones.
            let n = SCREEN_TEST_ROWS;
            let mut state = 0x9e37_79b9_7f4a_7c15u64;
            let mut unit = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            };
            let bf16 = sparkflash_kernels::m4_quant::bf16_bits;
            let mut hidden = vec![0u16; n * HIDDEN];
            for (row, values) in hidden.chunks_mut(HIDDEN).enumerate() {
                let scale = 2.0f32.powi(row as i32 % 9 - 4);
                for (k, value) in values.iter_mut().enumerate() {
                    let v = (unit() + unit() + unit() + unit()) * 0.5 * scale;
                    *value = match row {
                        // All zero; one value not a number; too large for the bound; sparse.
                        37 => 0,
                        38 if k == 1234 => 0x7fc0,
                        39 => bf16(3.0e16),
                        40 if k % 61 != 0 => 0,
                        _ => bf16(v),
                    };
                }
            }
            let bytes: Vec<u8> = hidden.iter().flat_map(|word| word.to_le_bytes()).collect();
            let mut scratch = 0usize;
            let staged = cuda
                .alloc(bytes.len(), &mut scratch)
                .map_err(|error| format!("head screen self test rows: {error}"))?;
            let outcome = (|| {
                cuda.copy_htod(&staged, &bytes)?;
                cuda.launch(
                    self.rows,
                    (n as u32, 1, 1),
                    (256, 1, 1),
                    &[staged.0, self.act.0, self.meta.0],
                )?;
                cuda.launch(
                    self.mma,
                    (n.div_ceil(32) as u32, SCREEN_TILES as u32, 1),
                    (256, 1, 1),
                    &[
                        self.act.0,
                        self.meta.0,
                        self.codes.0,
                        self.coef.0,
                        self.upper.0,
                        self.tile_lo.0,
                        VOCAB_PER_RANK as u64,
                        n as u64,
                        // Greedy rows: the bounds are the logit's own.
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                )?;
                let mut act = vec![0u8; 2 * SCREEN_ROWS * HIDDEN];
                cuda.copy_dtoh(&mut act, &self.act)?;
                Ok::<_, String>((
                    act,
                    self.read_f32(cuda, self.meta, n * 4)?,
                    self.read_f32(cuda, self.upper, n * VOCAB_PER_RANK)?,
                    self.read_f32(cuda, self.tile_lo, n * SCREEN_TILES)?,
                ))
            })();
            cuda.free_buffers(&[&staged]);
            let (act, meta, upper, tile_lo) = outcome?;

            let check_row = |row: usize| -> Result<(), String> {
                let host = head_screen_ref::row_digits(&hidden[row * HIDDEN..(row + 1) * HIDDEN]);
                let m = &meta[row * 4..row * 4 + 4];
                let upper = &upper[row * VOCAB_PER_RANK..(row + 1) * VOCAB_PER_RANK];
                let tile_lo = &tile_lo[row * SCREEN_TILES..(row + 1) * SCREEN_TILES];
                if host.usable != matches!(row, 0..=36 | 40) {
                    return Err(format!(
                        "row {row}: the reference's own cut is not the expected one"
                    ));
                }
                if m[0].to_bits() != host.scale.to_bits() {
                    return Err(format!("row {row}: scale {} for {}", m[0], host.scale));
                }
                for k in 0..HIDDEN {
                    let (p0, p1) = host.pair(k);
                    let lo = act[head_screen_ref::digit_row(row, 0) * HIDDEN + k] as i8;
                    let hi = act[head_screen_ref::digit_row(row, 1) * HIDDEN + k] as i8;
                    if (lo, hi) != (p0, p1) {
                        return Err(format!("row {row}: the digits at {k} differ"));
                    }
                }
                if !host.usable {
                    // Never ruled out: every upper bound is +infinity and no tile has a bound.
                    return if m[1] == f32::INFINITY
                        && upper.iter().all(|&value| value == f32::INFINITY)
                        && tile_lo.iter().all(|&value| value == f32::NEG_INFINITY)
                    {
                        Ok(())
                    } else {
                        Err(format!(
                            "row {row}: a row the bound does not cover was screened"
                        ))
                    };
                }
                if !safe(m[1], host.norm, 1.002) || !safe(m[2], host.dropped_norm, 1.002) {
                    return Err(format!(
                        "row {row}: the device took norms {} {}, the reference {} {}",
                        m[1], m[2], host.norm, host.dropped_norm
                    ));
                }
                let mut best = vec![(f32::NEG_INFINITY, 0.0f32); SCREEN_TILES];
                for entry in 0..VOCAB_PER_RANK {
                    let c = &coef[entry * 4..entry * 4 + 4];
                    let dot = head_screen_ref::integer_dot(
                        &codes[entry * HIDDEN..(entry + 1) * HIDDEN],
                        &host.digits,
                    );
                    let p = head_screen_ref::proxy(c[0], m[0], dot);
                    let half = head_screen_ref::half_width(c[1], c[2], m[1], m[2], p);
                    // The device may fuse the half width's operations: a few units in the last
                    // place of it, against an integer dot product that is exact.
                    let slack = 1e-5 * (p.abs() + half);
                    let off = (upper[entry] - (p + half)).abs();
                    if off.is_nan() || off > slack {
                        return Err(format!(
                            "row {row}, entry {entry}: upper bound {} for {} + {}",
                            upper[entry], p, half
                        ));
                    }
                    let tile = &mut best[entry / head_screen_ref::TILE];
                    if p - half > tile.0 {
                        *tile = (p - half, slack);
                    }
                }
                for (tile, &(lower, slack)) in best.iter().enumerate() {
                    let off = (tile_lo[tile] - lower).abs();
                    if off.is_nan() || off > slack {
                        return Err(format!(
                            "row {row}, tile {tile}: lower bound {} for {lower}",
                            tile_lo[tile]
                        ));
                    }
                }
                Ok(())
            };
            let next = std::sync::atomic::AtomicUsize::new(0);
            let workers =
                std::thread::available_parallelism().map_or(4, |count| count.get().min(16));
            let failure = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..workers)
                    .map(|_| {
                        scope.spawn(|| {
                            loop {
                                let row = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if row >= n {
                                    return Ok(());
                                }
                                check_row(row)?;
                            }
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .filter_map(|handle| match handle.join() {
                        Ok(result) => result.err(),
                        Err(_) => Some("a checker thread panicked".to_owned()),
                    })
                    .next()
            });
            if let Some(what) = failure {
                return refuse(what);
            }
            eprintln!(
                "head screen self test: {VOCAB_PER_RANK} entries' codes and {n} rows' bounds match \
                 the reference"
            );
            Ok(())
        }

        fn close(self, cuda: &Cuda) {
            if self.check {
                self.report();
            }
            let ptrs = [
                &self.codes,
                &self.coef,
                &self.act,
                &self.meta,
                &self.upper,
                &self.tile_lo,
                &self.counts,
            ];
            cuda.free_buffers(&ptrs);
            for module in self.modules {
                cuda.free(module, &[]);
            }
        }
    }

    /// The loaded, ready-to-screen serving tail of one rank.
    pub struct TailStage {
        /// The directory the two cubins came from (evidence).
        pub kernel_dir: PathBuf,
        /// The configured RMSNorm epsilon.
        pub eps: f32,
        /// The derived head shard digest (evidence: proves which bytes the maxima came from).
        pub head_shard_sha256: String,
        /// The bytes the tail allocated (arena-independent).
        pub allocated: usize,
        /// The bound arena's final-norm weight pointer.
        pub norm_weight: Deviceptr,
        /// The bound arena's head shard pointer.
        pub head_weight: Deviceptr,
        /// The per-rank buffers.
        pub buffers: TailBuffers,
        /// The rows head's screen (`diagnostics.head_screen`), when the profile has it on.
        pub head_screen: Option<HeadScreen>,
        rmsnorm_module: Module,
        rmsnorm: Kernel,
        head_module: Module,
        head: Kernel,
    }

    // The loop's next increment calls these; kept compiled and clippy-clean in the meantime so the
    // device half cannot bit-rot while the arena sequence is wired.
    #[allow(dead_code)]
    impl TailStage {
        /// Loads the tail kernels, allocates the buffers and derives w_block_max from the bound
        /// arena's head shard, once. Crate-visible: the bind type is crate-private, and only the
        /// four-rank loop (which owns the arena epoch) may open the tail.
        ///
        /// # Errors
        /// A missing cubin, a read failure, an allocation/copy failure, or a bound tail pointer
        /// that cannot be read.
        pub fn open(
            cuda: &Cuda,
            kernel_dir: &Path,
            bind: &crate::tp4_resident::DeviceBind,
            eps: f32,
        ) -> Result<Self, String> {
            let rmsnorm_bytes = std::fs::read(cubin_path(kernel_dir, TAIL_CUBINS[0]))
                .map_err(|error| format!("tail rmsnorm cubin: {error}"))?;
            let head_bytes = std::fs::read(cubin_path(kernel_dir, TAIL_CUBINS[1]))
                .map_err(|error| format!("tail certified-head cubin: {error}"))?;
            let mut allocated = 0usize;
            let hidden = cuda
                .alloc(HIDDEN * 2, &mut allocated)
                .map_err(|error| format!("tail hidden buffer: {error}"))?;
            let normed = cuda
                .alloc(HIDDEN * 2, &mut allocated)
                .map_err(|error| format!("tail normed buffer: {error}"))?;
            let w_block_max = cuda
                .alloc(W_BLOCK_MAX_BYTES, &mut allocated)
                .map_err(|error| format!("tail w_block_max buffer: {error}"))?;
            let logits = cuda
                .alloc(LOGITS_BYTES_PER_RANK, &mut allocated)
                .map_err(|error| format!("tail logits buffer: {error}"))?;
            let certified = cuda
                .alloc(VOCAB_PER_RANK, &mut allocated)
                .map_err(|error| format!("tail certified buffer: {error}"))?;
            let certified_count = cuda
                .alloc(4, &mut allocated)
                .map_err(|error| format!("tail certified-count buffer: {error}"))?;
            let block_max = cuda
                .alloc(CHUNKS_PER_RANK * 4, &mut allocated)
                .map_err(|error| format!("tail block-max buffer: {error}"))?;
            let block_argmax = cuda
                .alloc(CHUNKS_PER_RANK * 4, &mut allocated)
                .map_err(|error| format!("tail block-argmax buffer: {error}"))?;

            // Derive w_block_max once, from the arena's head shard bytes, at loop open: this is
            // init-time work, not a per-token weight H2D, and it keeps TailBind at three pointers.
            let shard_bytes = VOCAB_PER_RANK * HIDDEN * 2;
            let mut shard = vec![0u8; shard_bytes];
            cuda.copy_dtoh(&mut shard, &bind.tail.head)
                .map_err(|error| format!("head shard read for w_block_max: {error}"))?;
            let maxima = lm_head_ref::block_maxima_bf16(&shard);
            debug_assert_eq!(maxima.len(), W_BLOCK_MAX_BYTES);
            cuda.copy_htod(&w_block_max, &maxima)
                .map_err(|error| format!("w_block_max upload: {error}"))?;

            // cuMemAlloc does not zero: the certified-head kernel atomically accumulates into the
            // count buffer without initialising it, so an unzeroed allocation reports garbage
            // certified entries (m1y9: per-rank ~2.0e9 counts). Initialise it here, once per open.
            cuda.copy_htod(&certified_count, &[0u8; 4])
                .map_err(|error| format!("tail certified-count reset: {error}"))?;

            let (rmsnorm_module, rmsnorm) = rmsnorm_launch::load(cuda, &rmsnorm_bytes)?;
            let (head_module, head) = cert_lm_head_launch::load(cuda, &head_bytes)?;
            let diagnostics = &sparkflash_model::active_profile().diagnostics;
            let check = diagnostics.head_screen_check.unwrap_or(false);
            let head_screen = if diagnostics.head_screen.unwrap_or(HEAD_SCREEN_DEFAULT) {
                Some(HeadScreen::open(
                    cuda,
                    kernel_dir,
                    bind.tail.head,
                    &shard,
                    check,
                    &mut allocated,
                )?)
            } else if check {
                return Err(
                    "head screen check: SPARKFLASH_HEAD_SCREEN_CHECK is set but the screen is \
                     off (SPARKFLASH_HEAD_SCREEN=0), so there is nothing to check"
                        .to_owned(),
                );
            } else {
                None
            };
            Ok(Self {
                kernel_dir: kernel_dir.to_path_buf(),
                eps,
                head_shard_sha256: crate::tp4_forward::sha256_hex(&shard),
                allocated,
                norm_weight: bind.tail.norm,
                head_weight: bind.tail.head,
                buffers: TailBuffers {
                    hidden,
                    normed,
                    w_block_max,
                    logits,
                    certified,
                    certified_count,
                    block_max,
                    block_argmax,
                },
                head_screen,
                rmsnorm_module,
                rmsnorm,
                head_module,
                head,
            })
        }

        /// Normalises one position's hidden state and screens this rank's head shard.
        ///
        /// Returns this rank's VOCAB_PER_RANK f32 logits; the caller gathers them with
        /// super::gather_logits so no rank ever argmaxes a partial vector.
        ///
        /// # Errors
        /// A hidden state that is not one BF16 row, a zero vocabulary, or a driver failure.
        pub fn screen(
            &self,
            cuda: &Cuda,
            hidden_bf16: &[u8],
            layer_id: u32,
            step_seq: u64,
        ) -> Result<Vec<f32>, String> {
            if hidden_bf16.len() != HIDDEN * 2 {
                return Err(format!(
                    "tail screen: the hidden state is {} bytes, one BF16 row is {}",
                    hidden_bf16.len(),
                    HIDDEN * 2
                ));
            }
            cuda.copy_htod(&self.buffers.hidden, hidden_bf16)
                .map_err(|error| format!("tail hidden htod: {error}"))?;
            // cuMemAlloc does not zero, and the certified head only atomicAdds into this buffer, so
            // the count it reads back would otherwise be whatever the allocator handed us (m1y9: a
            // recycled block, ~2.0e9, differing per rank). Reset it before every screen, not only in
            // open, so a stage reused across head positions is safe by construction.
            cuda.copy_htod(&self.buffers.certified_count, &0u32.to_le_bytes())
                .map_err(|error| format!("tail certified-count reset: {error}"))?;
            let norm = rmsnorm_launch::RmsNormLaunchParams {
                x: self.buffers.hidden,
                weight: self.norm_weight,
                out: self.buffers.normed,
                rows: 1,
                cols: HIDDEN as u32,
                eps: self.eps,
                trace_ring: 0,
                layer_id,
                step_seq,
            };
            rmsnorm_launch::launch(cuda, self.rmsnorm, &norm)?;
            let head = cert_lm_head_launch::CertLmHeadParams::new(
                self.buffers.normed,
                self.head_weight,
                self.buffers.w_block_max,
                self.buffers.logits,
                self.buffers.certified,
                self.buffers.certified_count,
                self.buffers.block_max,
                self.buffers.block_argmax,
                VOCAB_PER_RANK as u32,
                0,
                layer_id,
                step_seq,
            );
            cert_lm_head_launch::launch(cuda, self.head, &head)?;
            let mut raw = vec![0u8; LOGITS_BYTES_PER_RANK];
            cuda.copy_dtoh(&mut raw, &self.buffers.logits)
                .map_err(|error| format!("tail logits dtoh: {error}"))?;
            let mut values = Vec::with_capacity(VOCAB_PER_RANK);
            for index in (0..raw.len()).step_by(4) {
                values.push(f32::from_le_bytes([
                    raw[index],
                    raw[index + 1],
                    raw[index + 2],
                    raw[index + 3],
                ]));
            }
            Ok(values)
        }

        /// The generation head: the same norm and head launches as [TailStage::screen_device], then
        /// only the per-chunk maxima come back (two small copies instead of the shard's logits),
        /// and one four-word gather gives every rank the global winner. The reduction keeps the
        /// reference tie rule: chunks and ranks are visited in vocabulary order and a later entry
        /// wins only when strictly greater. Returns (token, logit).
        pub fn screen_device_argmax(
            &self,
            cuda: &Cuda,
            transport: &dyn sparkflash_comm::transport::Transport,
            layer_id: u32,
            step_seq: u64,
            seq: u64,
        ) -> Result<(u32, f32), String> {
            let norm = rmsnorm_launch::RmsNormLaunchParams {
                x: self.buffers.hidden,
                weight: self.norm_weight,
                out: self.buffers.normed,
                rows: 1,
                cols: HIDDEN as u32,
                eps: self.eps,
                trace_ring: 0,
                layer_id,
                step_seq,
            };
            rmsnorm_launch::launch(cuda, self.rmsnorm, &norm)?;
            let head = cert_lm_head_launch::CertLmHeadParams::new(
                self.buffers.normed,
                self.head_weight,
                self.buffers.w_block_max,
                self.buffers.logits,
                self.buffers.certified,
                self.buffers.certified_count,
                self.buffers.block_max,
                self.buffers.block_argmax,
                VOCAB_PER_RANK as u32,
                0,
                layer_id,
                step_seq,
            );
            cert_lm_head_launch::launch(cuda, self.head, &head)?;
            let chunks = CHUNKS_PER_RANK;
            let mut max_bytes = vec![0u8; chunks * 4];
            let mut arg_bytes = vec![0u8; chunks * 4];
            cuda.copy_dtoh(&mut max_bytes, &self.buffers.block_max)
                .map_err(|error| format!("tail block max dtoh: {error}"))?;
            cuda.copy_dtoh(&mut arg_bytes, &self.buffers.block_argmax)
                .map_err(|error| format!("tail block argmax dtoh: {error}"))?;
            let mut best = (f32::NEG_INFINITY, 0u32, false);
            for chunk in 0..chunks {
                let at = chunk * 4;
                let value = f32::from_le_bytes([
                    max_bytes[at],
                    max_bytes[at + 1],
                    max_bytes[at + 2],
                    max_bytes[at + 3],
                ]);
                let index = u32::from_le_bytes([
                    arg_bytes[at],
                    arg_bytes[at + 1],
                    arg_bytes[at + 2],
                    arg_bytes[at + 3],
                ]);
                if !best.2 || value > best.0 {
                    best = (value, index, true);
                }
            }
            let mut words = split_f32_words(&[best.0]);
            words.push((best.1 >> 16) as u16);
            words.push(best.1 as u16);
            let world = transport.world_size();
            let mut gathered = vec![0u16; world * 4];
            transport
                .all_gather_bf16(&words, &mut gathered, seq)
                .map_err(|error| format!("argmax all-gather: {error}"))?;
            let mut winner = (f32::NEG_INFINITY, 0u32, false);
            for rank in 0..world {
                let w = &gathered[rank * 4..rank * 4 + 4];
                let value = join_f32_words(&w[..2])?[0];
                let index =
                    ((u32::from(w[2]) << 16) | u32::from(w[3])) + (rank * VOCAB_PER_RANK) as u32;
                if !winner.2 || value > winner.0 {
                    winner = (value, index, true);
                }
            }
            Ok((winner.1, winner.0))
        }

        /// Screens the rank's vocabulary directly from `self.buffers.hidden` (already populated on
        /// device alongside a zeroed `self.buffers.certified_count`), performing only the single
        /// logits D2H readback.
        pub fn screen_device(
            &self,
            cuda: &Cuda,
            layer_id: u32,
            step_seq: u64,
        ) -> Result<Vec<f32>, String> {
            let norm = rmsnorm_launch::RmsNormLaunchParams {
                x: self.buffers.hidden,
                weight: self.norm_weight,
                out: self.buffers.normed,
                rows: 1,
                cols: HIDDEN as u32,
                eps: self.eps,
                trace_ring: 0,
                layer_id,
                step_seq,
            };
            rmsnorm_launch::launch(cuda, self.rmsnorm, &norm)?;
            let head = cert_lm_head_launch::CertLmHeadParams::new(
                self.buffers.normed,
                self.head_weight,
                self.buffers.w_block_max,
                self.buffers.logits,
                self.buffers.certified,
                self.buffers.certified_count,
                self.buffers.block_max,
                self.buffers.block_argmax,
                VOCAB_PER_RANK as u32,
                0,
                layer_id,
                step_seq,
            );
            cert_lm_head_launch::launch(cuda, self.head, &head)?;
            let mut raw = vec![0u8; LOGITS_BYTES_PER_RANK];
            cuda.copy_dtoh(&mut raw, &self.buffers.logits)
                .map_err(|error| format!("tail logits dtoh: {error}"))?;
            let mut values = Vec::with_capacity(VOCAB_PER_RANK);
            for index in (0..raw.len()).step_by(4) {
                values.push(f32::from_le_bytes([
                    raw[index],
                    raw[index + 1],
                    raw[index + 2],
                    raw[index + 3],
                ]));
            }
            Ok(values)
        }

        /// Reads the screen pass's side buffers back for the evidence record: the normed BF16 row,
        /// the certified mask and count, and the per-chunk block maxima and argmax. The logits are
        /// read by [TailStage::screen] itself; every digest in the record comes from these buffers.
        ///
        /// # Errors
        /// A driver copy failure, or a certified count that is not inside the vocab.
        pub fn read_outputs(&self, cuda: &Cuda) -> Result<TailOutputs, String> {
            let mut normed = vec![0u8; HIDDEN * 2];
            cuda.copy_dtoh(&mut normed, &self.buffers.normed)
                .map_err(|error| format!("tail normed dtoh: {error}"))?;
            let mut certified = vec![0u8; VOCAB_PER_RANK];
            cuda.copy_dtoh(&mut certified, &self.buffers.certified)
                .map_err(|error| format!("tail certified dtoh: {error}"))?;
            let mut count_raw = [0u8; 4];
            cuda.copy_dtoh(&mut count_raw, &self.buffers.certified_count)
                .map_err(|error| format!("tail certified-count dtoh: {error}"))?;
            let certified_count = u32::from_le_bytes(count_raw);
            if certified_count as usize > VOCAB_PER_RANK {
                return Err(format!(
                    "tail certified count {certified_count} is larger than the rank vocabulary {VOCAB_PER_RANK}"
                ));
            }
            let mut max_raw = vec![0u8; CHUNKS_PER_RANK * 4];
            cuda.copy_dtoh(&mut max_raw, &self.buffers.block_max)
                .map_err(|error| format!("tail block-max dtoh: {error}"))?;
            let block_max = max_raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            let mut argmax_raw = vec![0u8; CHUNKS_PER_RANK * 4];
            cuda.copy_dtoh(&mut argmax_raw, &self.buffers.block_argmax)
                .map_err(|error| format!("tail block-argmax dtoh: {error}"))?;
            let block_argmax = argmax_raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            Ok(TailOutputs {
                normed_bf16: normed,
                certified,
                certified_count,
                block_max,
                block_argmax,
            })
        }

        /// Frees the stage's buffers and unloads the two modules.
        pub fn close(self, cuda: &Cuda) {
            let buffers = self.buffers;
            let ptrs = [
                buffers.hidden,
                buffers.normed,
                buffers.w_block_max,
                buffers.logits,
                buffers.certified,
                buffers.certified_count,
                buffers.block_max,
                buffers.block_argmax,
            ];
            cuda.free_buffers(&ptrs.iter().collect::<Vec<_>>());
            if let Some(screen) = self.head_screen {
                screen.close(cuda);
            }
            cuda.free(self.rmsnorm_module, &[]);
            cuda.free(self.head_module, &[]);
        }
    }
}

#[cfg(feature = "cuda")]
pub use device::{TailBuffers, TailOutputs, TailStage};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_word_split_round_trips_every_f32_bit_for_bit() {
        let values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::from_bits(1),
            f32::from_bits(0x007f_ffff),
            f32::MAX,
            f32::MIN,
            f32::from_bits(0x8000_0001),
        ];
        let words = split_f32_words(&values);
        assert_eq!(words.len(), values.len() * 2);
        let back = join_f32_words(&words).expect("join");
        for (index, (value, expected)) in back.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "value {index} must round-trip bit for bit"
            );
        }
        assert!(join_f32_words(&[1u16]).is_err(), "an odd stream is refused");
    }

    #[test]
    fn the_argmax_breaks_ties_by_the_lowest_id_and_reports_the_margin() {
        let mut values = vec![0.0f32; VOCAB];
        values[7] = 3.5;
        values[900] = 3.5;
        values[129_000] = 3.0;
        let logits = argmax_of(values).expect("argmax");
        assert_eq!(logits.argmax, 7, "ties keep the lowest id");
        assert_eq!(logits.top1, 3.5);
        assert_eq!(logits.top2, 3.5);
        assert_eq!(logits.margin, 0.0);
        assert!(argmax_of(Vec::new()).is_err());
    }

    #[test]
    fn assembling_the_four_rank_chunks_is_the_full_vocabulary_in_rank_order() {
        let chunks: Vec<Vec<f32>> = (0..4u32)
            .map(|rank| {
                let mut chunk = vec![0.0f32; VOCAB_PER_RANK];
                chunk[0] = 10.0 * rank as f32;
                chunk
            })
            .collect();
        let logits = assemble_chunks(&chunks).expect("assemble");
        assert_eq!(logits.full.len(), VOCAB);
        assert_eq!(logits.argmax, 3 * VOCAB_PER_RANK as u32);
        assert_eq!(logits.top1, 30.0);
        let short = vec![vec![0.0f32; VOCAB_PER_RANK], vec![0.0f32; 1]];
        assert!(assemble_chunks(&short).is_err());
        assert!(
            assemble_chunks(&chunks[..3]).is_err(),
            "three ranks are not the vocabulary"
        );
    }

    #[test]
    fn the_exact_gather_moves_f32_bit_patterns_through_the_u16_collective() {
        use sparkflash_comm::transport::{LoopbackCluster, LoopbackConfig};
        use std::time::Duration;

        let cluster = LoopbackCluster::new(LoopbackConfig {
            world_size: 4,
            timeout: Duration::from_secs(10),
            ..LoopbackConfig::default()
        })
        .expect("cluster");
        let seq = 41;
        let expected: Vec<Vec<f32>> = (0..4usize)
            .map(|rank| {
                (0..VOCAB_PER_RANK)
                    .map(|index| {
                        let bits = (rank as u32) << 24 | (index as u32) << 4 | (index as u32 & 0xf);
                        f32::from_bits(bits)
                    })
                    .collect()
            })
            .collect();
        let mut results: Vec<Option<Logits>> = (0..4).map(|_| None).collect();
        std::thread::scope(|scope| {
            for (rank, slot) in results.iter_mut().enumerate() {
                let transport = cluster.handle(rank).expect("handle");
                let local = expected[rank].clone();
                scope.spawn(move || {
                    *slot = Some(gather_logits(&transport, &local, seq).expect("gather"));
                });
            }
        });
        let assembled: Vec<Logits> = results
            .into_iter()
            .map(|slot| slot.expect("slot"))
            .collect();
        for rank in 0..4 {
            assert_eq!(assembled[rank].full.len(), VOCAB);
            for (full_rank, chunk) in expected.iter().enumerate() {
                let base = full_rank * VOCAB_PER_RANK;
                for (index, want) in chunk.iter().enumerate() {
                    assert_eq!(
                        assembled[rank].full[base + index].to_bits(),
                        want.to_bits(),
                        "rank {rank} reassembly at {base}+{index}"
                    );
                }
            }
            assert_eq!(
                assembled[rank], assembled[0],
                "every rank argmaxes the same full vector"
            );
        }
    }

    #[test]
    fn the_tail_cubin_names_are_the_pinned_symbols() {
        assert_eq!(TAIL_CUBINS, ["rmsnorm_sm121", "cert_lm_head_sm121"]);
        let path = cubin_path(Path::new("/cubins"), TAIL_CUBINS[1]);
        assert_eq!(path, PathBuf::from("/cubins/cert_lm_head_sm121.cubin"));
        assert_eq!(CHUNKS_PER_RANK, 32);
        assert_eq!(W_BLOCK_MAX_BYTES, 2_585_600);
        assert_eq!(LOGITS_BYTES_PER_RANK, 129_280);
        assert_eq!(VOCAB_PER_RANK * 4, VOCAB);
        assert_eq!(BLOCKS_PER_ENTRY, 40);
        assert_eq!(CHUNK, 1024);
    }
}
