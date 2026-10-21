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
        W_BLOCK_MAX_BYTES, cubin_path,
    };
    use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr, Kernel, Module};
    use sparkflash_kernels::{cert_lm_head_launch, lm_head_ref, rmsnorm_launch};
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
        pub(crate) fn open(
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
