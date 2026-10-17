//! The Engram n-gram path: the hash state and the layer write.
//!
//! Spec: docs/engine/01-forward-order.md sections 3.1 (steps 2 and 7a), 3.7 and 5. The two halves
//! are the hash state (NgramHashState.forward, engram.py:159-184) and the write (Engram.forward,
//! model.py:350-365). The whole-model entry point computes the hashes ONCE for the whole chunk before
//! any layer runs (model.py:1252) and then writes them into the INPUT of layers 1 and 14, before
//! those blocks run (model.py:1262-1263).
//!
//! # What this module reproduces exactly
//!
//! - The bucket layout: max_ngram_size - 1 n-gram orders, each split over engram_n_heads heads, each
//!   head owning the next unused prime above engram_vocab_size (engram.py:9-14, 109-118), with each
//!   layer's offsets being the running sum of its own primes (engram.py:149). Every hash id lands in
//!   its own head's disjoint range, which the tests check.
//! - The hash formula: the look-back window's ids are multiplied by a per-(layer, look-back)
//!   multiplier and XOR-accumulated one look-back at a time; each running value is reduced modulo that
//!   order's prime and shifted by its offset (engram.py:171-184).
//! - The pad substitution: a look-back that reaches before the sequence start, or that passes through
//!   a dead token, uses the pad id for that shift and every later one (engram.py:171-174).
//! - The write: h + gate (x) value, one gate per (token, hc copy) from a normalised dot product, with
//!   a signed square root before the sigmoid (model.py:356-365).
//!
//! # What it does NOT reproduce, stated rather than hidden
//!
//! - The multiplier stream is the reference's ALGORITHM, ported line by line from NumPy's own
//!   sources into [numpy_rng]: the SeedSequence mix and state (bit_generator.pyx:56-64, 66-78,
//!   158-169, 351-383, 413-460), PCG64's seeding, LCG step and XSL RR 128/64 output (pcg64.c:20-34,
//!   pcg64.h), and the Lemire bounded draw that Generator.integers documents (distributions.c:
//!   1224-1258, 1425-1455). So [multipliers] is what
//!   np.random.default_rng(10007 * layer_id).integers(0, bound, size=(max_ngram,)) * 2 + 1 computes
//!   (engram.py:75-82). What is still unverified is EQUALITY with the real library: there is no NumPy
//!   on this host, so no golden vector from it was ever compared against. The port is pinned by its
//!   own closed forms and by a second written-out copy of the chain in the tests, not by a reference
//!   output. Read [numpy_rng] before treating an id as reference-verified.
//! - The compressed token map. build_compressed_token_map (engram.py:17-61) needs the tokenizer's
//!   normalizer chain (spec section 8 item 7). The ids handed to [ngram_hashes] are taken to be already
//!   compressed, and engram_pad_id is used as the pad where the reference pads with
//!   token_map[engram_pad_id] (engram.py:147).
//! - The bf16 round trip. The reference casts the looked-up rows to bfloat16 (model.py:320) and the
//!   write back to the input dtype (model.py:365). This crate's reference pass is f32 throughout, as
//!   forward/mod.rs states, so the values are the f32 ones.
//! - The table read. [engram_forward] pages the rows a chunk looks up through
//!   [WeightSource::read_ranges] when the source supports it (the CLI's loader does), so the
//!   released 98 GB tables are read a few rows at a time rather than refused; a source that only
//!   serves whole tensors keeps the whole-table read under the element limit, which is what turns a
//!   released-size table into an explicit refusal there.

use std::collections::{BTreeMap, BTreeSet};

use crate::config::{ConfigError, ModelConfig};
use crate::dtype::DType;
use crate::linalg::{self, EngineError, ScaleGrid};
use crate::quant;
use crate::weights::WeightSource;

/// The id that marks a position taking no part in an n-gram.
///
/// NgramHashState.DEAD is -1 (engram.py:138) and the ids here are unsigned, so it is u32::MAX. The
/// reference substitutes the pad id for a dead token and relies on the token mask to zero the gate
/// (engram.py:166, model.py:363-364); this module writes the sentinel across that position's whole
/// hash row instead, and [engram_forward] then skips the lookup AND the write for it. The observable
/// result is the same - the position is left exactly as it was - and it is testable without a mask
/// argument these signatures do not carry.
pub const HASH_SENTINEL: u32 = u32::MAX;

/// How many hash columns one token has: one per (n-gram order above the unigram, head) pair.
///
/// The released configuration has 4 - 1 orders and 8 heads, so 24 columns (spec section 5);
/// ModelConfig::tiny has one order and two heads, so a hardcoded 24 would be wrong for it. The
/// whole-model entry point needs this to size the buffer it computes once per chunk (model.py:1252).
#[must_use]
pub fn hash_columns(cfg: &ModelConfig) -> usize {
    cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads
}

/// The bucket layout of the hash tables, plus the multiplier stream.
#[derive(Debug)]
struct Layout {
    /// primes[layer][order * n_heads + head], in the order engram.py:109-118 draws them.
    primes: Vec<Vec<u64>>,
    /// offsets[layer][column]: the running sum of that layer's primes (engram.py:149).
    offsets: Vec<Vec<u64>>,
    /// multipliers[layer][look_back], forced odd and bounded (engram.py:71-82).
    multipliers: Vec<Vec<i64>>,
}

impl Layout {
    /// Build the layout for a configuration.
    ///
    /// # Errors
    ///
    /// [EngineError::Config] when a layer's table is too small for its own bucket layout. The
    /// reference indexes the table with the id the hash produced (model.py:309, 317) and never wraps
    /// it back into the table, so the rows must cover the layout: config.json's released
    /// engram_num_embeddings are exactly this layout's per-layer totals (384,006,168 and 384,016,682),
    /// which is the proof that the ids are the raw bucket ids. A configuration whose rows are below
    /// the total is inconsistent, and this refuses it instead of silently returning ids that address
    /// the wrong rows.
    fn for_config(cfg: &ModelConfig) -> Result<Layout, EngineError> {
        let columns = hash_columns(cfg);
        // engram.py:109-118: one shared seen set across layers, so no prime is reused anywhere, and
        // the draw starts one above engram_vocab_size.
        let mut seen = BTreeSet::new();
        let mut candidate = cfg.engram_vocab_size.saturating_sub(1) as u64;
        let mut primes = Vec::with_capacity(cfg.engram_layer_ids.len());
        let mut offsets = Vec::with_capacity(cfg.engram_layer_ids.len());
        for (index, &layer_id) in cfg.engram_layer_ids.iter().enumerate() {
            let mut layer = Vec::with_capacity(columns);
            let mut offset = Vec::with_capacity(columns);
            let mut running = 0u64;
            for _ in 0..columns {
                candidate = find_next_prime(candidate, &seen);
                seen.insert(candidate);
                offset.push(running);
                running = running.saturating_add(candidate);
                layer.push(candidate);
            }
            let rows = cfg.engram_num_embeddings.get(index).copied().unwrap_or(0) as u64;
            if running > rows {
                return Err(EngineError::Config(ConfigError::Invalid {
                    field: "engram_num_embeddings",
                    detail: format!(
                        "layer {layer_id} has {rows} rows but its {columns} bucket primes above \
                         engram_vocab_size {} sum to {running}, and the reference indexes the table \
                         with the hash id directly (model.py:309, 317)",
                        cfg.engram_vocab_size
                    ),
                }));
            }
            primes.push(layer);
            offsets.push(offset);
        }
        Ok(Layout {
            primes,
            offsets,
            multipliers: multipliers(cfg),
        })
    }
}

/// The smallest prime above start that has not been handed out yet (engram.py:9-14).
fn find_next_prime(start: u64, seen: &BTreeSet<u64>) -> u64 {
    let mut candidate = start.saturating_add(1);
    while !is_prime(candidate) || seen.contains(&candidate) {
        candidate = candidate.saturating_add(1);
    }
    candidate
}

/// Whether value is prime, by trial division. Only used for the drawn bucket moduli.
fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3u64;
    while divisor.saturating_mul(divisor) <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The reference's multiplier stream: NumPy's default_rng, ported from NumPy's own sources.
///
/// engram.py:75-82 draws one multiplier per (layer, look-back) with
/// np.random.default_rng(10007 * layer_id).integers(0, bound, size=(max_ngram,)) * 2 + 1, so the
/// stream is a SeedSequence of one integer entropy, a PCG64 bit generator, and a Lemire bounded
/// draw. All three are integer-deterministic, so all three are reproduced here:
///
/// - numpy/random/bit_generator.pyx: the SeedSequence constants INIT_A, MULT_A, INIT_B, MULT_B,
///   MIX_MULT_L, MIX_MULT_R and XSHIFT (56-64), _int_to_uint32_array (66-78), hashmix and mix
///   (158-169), mix_entropy (351-383) and generate_state (413-460). The pool size is 4.
/// - numpy/random/src/pcg64/pcg64.c: pcg64_set_seed (20-34), whose two state words are the HIGH half
///   of the seed. numpy/random/src/pcg64/pcg64.h: the LCG step, pcg_setseq_128_srandom_r, the
///   multiplier PCG_DEFAULT_MULTIPLIER_HIGH/LOW, and the XSL RR 128/64 output function
///   pcg_output_xsl_rr_128_64 = rotr64(high ^ low, high >> 58). The increment is odd and the
///   multiplier is 1 mod 4, so the LCG has its full 2**128 period.
/// - numpy/random/src/distributions/distributions.c: bounded_lemire_uint64 (1224-1258) and the
///   dispatcher random_bounded_uint64 (1425-1455). Generator.integers documents Lemire's algorithm
///   as its method, so the Lemire branch is ported, not the masked-rejection branch. The C function
///   takes an INCLUSIVE range: for integers(0, high) that is high - 1, because
///   _bounded_integers.pyx.in:141 subtracts is_open = not closed.
///
/// # What is still unverified
///
/// NumPy is not installed on this host, so NO golden vector from the real library exists here and
/// equality with a real draw is unverified. What is pinned is the algorithm: the tests re-derive the
/// LCG step, the two-step seeding and the output function with independent u128 arithmetic and the
/// literals from those headers, re-derive the SeedSequence pool and state words with a second copy of
/// the mixing, and check the whole chain against a second copy of the draw. Treat an id as
/// reference-ALGORITHM, not as reference-verified.
mod numpy_rng {
    /// SeedSequence's entropy pool (bit_generator.pyx:56).
    const POOL_SIZE: usize = 4;
    /// bit_generator.pyx:57-64.
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_MULT_L: u32 = 0xca01_f9dd;
    const MIX_MULT_R: u32 = 0x4973_f715;
    const XSHIFT: u32 = 16;
    const MASK32: u32 = 0xffff_ffff;
    /// PCG_DEFAULT_MULTIPLIER_HIGH and _LOW of pcg64.h, as one 128-bit constant.
    const MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

    /// The 32-bit words of a non-negative integer, lowest first (bit_generator.pyx:66-78).
    ///
    /// Zero is one zero word, not an empty array, exactly as _int_to_uint32_array builds it.
    pub(super) fn limbs(value: u64) -> Vec<u32> {
        let mut remaining = value;
        if remaining == 0 {
            return vec![0];
        }
        let mut words = Vec::new();
        while remaining > 0 {
            words.push((remaining & u64::from(MASK32)) as u32);
            remaining >>= 32;
        }
        words
    }

    /// SeedSequence.hashmix, whose accumulator is an input-output argument (bit_generator.pyx:158-164).
    fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
        let mut mixed = value ^ *hash_const;
        *hash_const = hash_const.wrapping_mul(MULT_A);
        mixed = mixed.wrapping_mul(*hash_const);
        mixed ^ (mixed >> XSHIFT)
    }

    /// SeedSequence.mix (bit_generator.pyx:166-169).
    fn mix(x: u32, y: u32) -> u32 {
        let result = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        result ^ (result >> XSHIFT)
    }

    /// SeedSequence.mix_entropy over a fresh pool (bit_generator.pyx:324-325, 351-383).
    ///
    /// The reference walks both loops with hash_const carried between every call, so the call order
    /// is part of the stream; these index loops keep that order, and the second one does not skip
    /// i_src == i_dst because the reference's does not.
    pub(super) fn pool(entropy: &[u32]) -> [u32; POOL_SIZE] {
        let mut mixer = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        for (index, slot) in mixer.iter_mut().enumerate() {
            let word = entropy.get(index).copied().unwrap_or(0);
            *slot = hashmix(word, &mut hash_const);
        }
        let mut source_index = 0;
        while source_index < POOL_SIZE {
            let source = mixer[source_index];
            let mut destination = 0;
            while destination < POOL_SIZE {
                if source_index != destination {
                    mixer[destination] = mix(mixer[destination], hashmix(source, &mut hash_const));
                }
                destination += 1;
            }
            source_index += 1;
        }
        let mut extra = POOL_SIZE;
        while extra < entropy.len() {
            let source = entropy[extra];
            let mut destination = 0;
            while destination < POOL_SIZE {
                mixer[destination] = mix(mixer[destination], hashmix(source, &mut hash_const));
                destination += 1;
            }
            extra += 1;
        }
        mixer
    }

    /// SeedSequence.generate_state(4, np.uint64) (bit_generator.pyx:413-460).
    ///
    /// Eight uint32 words are drawn from the pool in a cycle and then viewed as little-endian uint64.
    pub(super) fn state_words(pool: &[u32; POOL_SIZE]) -> [u64; 4] {
        let mut words = [0u32; 8];
        let mut hash_const = INIT_B;
        for (index, slot) in words.iter_mut().enumerate() {
            let mut value = pool[index % POOL_SIZE] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_const);
            *slot = value ^ (value >> XSHIFT);
        }
        let mut state = [0u64; 4];
        for (index, pair) in words.as_chunks::<2>().0.iter().enumerate() {
            state[index] = u64::from(pair[0]) | (u64::from(pair[1]) << 32);
        }
        state
    }

    /// NumPy's PCG64 with the 2018 seeding: XSL RR 128/64 over a 128-bit LCG (pcg64.h).
    pub(super) struct Pcg64 {
        state: u128,
        inc: u128,
    }

    impl Pcg64 {
        /// pcg64_set_seed (pcg64.c:20-34), then pcg_setseq_128_srandom_r (pcg64.h).
        ///
        /// The seed words are big-endian pairs, so the state is (words[0] << 64) | words[1]. The
        /// increment is (initseq << 1) | 1, which is what makes it odd, and the seeding is:
        /// state = 0, step, state += initstate, step.
        pub(super) fn seeded(words: &[u64; 4]) -> Pcg64 {
            let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
            let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
            let mut rng = Pcg64 {
                state: 0,
                inc: (initseq << 1) | 1,
            };
            rng.step();
            rng.state = rng.state.wrapping_add(initstate);
            rng.step();
            rng
        }

        /// pcg_setseq_128_step_r: state <- state * MULTIPLIER + inc, modulo 2**128.
        fn step(&mut self) {
            self.state = self.state.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
        }

        /// pcg_setseq_128_xsl_rr_64_random_r: one step, then the XSL RR output function.
        pub(super) fn next_u64(&mut self) -> u64 {
            self.step();
            let high = (self.state >> 64) as u64;
            let low = self.state as u64;
            (high ^ low).rotate_right((high >> 58) as u32)
        }

        /// bounded_lemire_uint64 (distributions.c:1224-1258) for an exclusive bound.
        ///
        /// The C code is handed the inclusive range rng = rng_excl - 1 and keeps the high word of the
        /// scaled product, rejecting while the low word falls below Lemire's threshold.
        pub(super) fn below(&mut self, rng_excl: u64) -> u64 {
            let mut scaled = u128::from(self.next_u64()) * u128::from(rng_excl);
            if (scaled as u64) < rng_excl {
                let threshold = (u64::MAX - (rng_excl - 1)) % rng_excl;
                while (scaled as u64) < threshold {
                    scaled = u128::from(self.next_u64()) * u128::from(rng_excl);
                }
            }
            (scaled >> 64) as u64
        }
    }

    /// One layer's draw: default_rng(10007 * layer_id).integers(0, bound) * 2 + 1 (engram.py:75-82).
    ///
    /// The entropy is a Python int small enough to be one uint32 word, so the pool is the mix of that
    /// one word and the bit generator is seeded from generate_state(4, uint64).
    pub(super) fn multipliers(layer: usize, max_ngram: usize, bound: u64) -> Vec<i64> {
        let entropy = limbs(10_007u64.wrapping_mul(layer as u64));
        let state = state_words(&pool(&entropy));
        let mut rng = Pcg64::seeded(&state);
        (0..max_ngram)
            .map(|_| {
                let value = rng.below(bound) as i64;
                value.wrapping_mul(2).wrapping_add(1)
            })
            .collect()
    }
}

/// One multiplier per (layer, look-back), forced odd and bounded so id * multiplier cannot overflow
/// (engram.py:64-83).
///
/// [numpy_rng] holds the ported stream and states what is reproduced and what is not.
fn multipliers(cfg: &ModelConfig) -> Vec<Vec<i64>> {
    let vocab = cfg.engram_compressed_vocab_size.max(1) as i64;
    let bound = (i64::MAX / vocab / 2).max(1) as u64;
    cfg.engram_layer_ids
        .iter()
        .map(|&layer| numpy_rng::multipliers(layer, cfg.engram_max_ngram_size, bound))
        .collect()
}

/// The n-gram hash ids of every position from start_pos to the end of tokens.
///
/// One row per token, each engram_layer_ids.len() * [hash_columns] ids wide, flattened in that order -
/// the [b, s, n_engram_layers, n_hash_cols] of model.py:1252 - so the whole chunk is hashed once and
/// both Engram layers read their own block of it.
///
/// tokens is the sequence THROUGH THE END OF THIS CHUNK, positions 0..tokens.len(): the look-back of
/// the reference reads its cache, which carries the prefill/decode history (engram.py:167), and
/// passing only the new chunk would silently hash the sequence start as if it were the boundary. The
/// rows produced are for start_pos..tokens.len().
///
/// out is replaced, never appended to. A token equal to [HASH_SENTINEL] is a dead token: its row is
/// all sentinels, and [engram_forward] leaves that position alone.
///
/// # Errors
///
/// [EngineError::Shape] when start_pos is past the tokens handed in.
pub fn ngram_hashes(
    cfg: &ModelConfig,
    tokens: &[u32],
    start_pos: usize,
    out: &mut Vec<u32>,
) -> Result<(), EngineError> {
    if start_pos > tokens.len() {
        return Err(EngineError::Shape {
            what: "engram hashes".to_owned(),
            detail: format!(
                "start position {start_pos} is past the {} tokens handed in",
                tokens.len()
            ),
        });
    }
    let columns = hash_columns(cfg);
    let layers = cfg.engram_layer_ids.len();
    let orders = cfg.engram_max_ngram_size;
    out.clear();
    if layers == 0 || columns == 0 || orders == 0 {
        return Ok(());
    }
    let layout = Layout::for_config(cfg)?;
    let seqlen = tokens.len() - start_pos;
    out.reserve(seqlen * layers * columns);
    // One scratch row of look-back tokens, allocated once for the whole chunk.
    let mut window = vec![0u32; orders];
    for position in start_pos..tokens.len() {
        if tokens[position] == HASH_SENTINEL {
            out.resize(out.len() + layers * columns, HASH_SENTINEL);
            continue;
        }
        // engram.py:171-175: blocked is sticky, so a look-back that reaches before the sequence start,
        // or through a dead token, pads this shift and every later one.
        let mut blocked = false;
        for (shift, slot) in window.iter_mut().enumerate() {
            let before_start = position < shift;
            let source = if before_start {
                cfg.engram_pad_id
            } else {
                tokens[position - shift]
            };
            blocked = blocked || before_start || source == HASH_SENTINEL;
            *slot = if blocked { cfg.engram_pad_id } else { source };
        }
        // engram.py:179-184: multiply, XOR the look-backs together, reduce each running value modulo
        // that order's prime, then shift it into the layer's own bucket range.
        for (layer, layer_multipliers) in layout.multipliers.iter().enumerate() {
            let mut rolling = i64::from(window[0]).wrapping_mul(layer_multipliers[0]);
            for order in 1..orders {
                rolling ^= i64::from(window[order]).wrapping_mul(layer_multipliers[order]);
                for head in 0..cfg.engram_n_heads {
                    let column = (order - 1) * cfg.engram_n_heads + head;
                    let prime = layout.primes[layer][column] as i64;
                    let bucket = rolling.rem_euclid(prime) as u64;
                    let id = bucket.saturating_add(layout.offsets[layer][column]);
                    out.push(u32::try_from(id).unwrap_or(HASH_SENTINEL));
                }
            }
        }
    }
    Ok(())
}

/// Reads and dequantises one Engram table into rows x width values.
///
/// The released table is 98 GB, so this is the call the element limit exists for: a limit below
/// rows * width refuses before a single byte is read (spec section 5; the tables are 98,305,579,008
/// and 98,308,270,592 bytes). The scale grid is one E8M0 scale per 32 columns of each row (push_engram
/// in [crate::tensor], the PerRow32 rule of [crate::dtype]).
///
/// # Errors
///
/// [EngineError::TooLarge] when the table is above limit, [EngineError::Missing] when the source does
/// not hold it, and [EngineError::Shape] when the scale grid does not fit the table.
fn read_table(
    prefix: &str,
    src: &dyn WeightSource,
    rows: usize,
    width: usize,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let mut table = linalg::decode_by_name(src, &format!("{prefix}.engram.embed.weight"), limit)?;
    let scales = linalg::decode_by_name(src, &format!("{prefix}.engram.embed.scale"), limit)?;
    linalg::apply_scales(&mut table, rows, width, &scales, ScaleGrid::PerRow32)?;
    Ok(table)
}

/// Reads only the rows a chunk looks up, when the source can page the table.
///
/// The table is [rows, width] F8E4M3 with one E8M0 scale per 32 columns of each row, so row `id`
/// occupies `width` weight bytes and `width.div_ceil(32)` scale bytes of the two stored tensors.
/// Those are the plan's pinned storage rules for the table (loader.rs, tensor.rs), not something a
/// caller may vary: a source that reported a different dtype would be decoded wrongly, which is why
/// the paged path is only taken through a plan-verified source.
/// Returns `None` when the source serves whole tensors only (the caller falls back to
/// [read_table]); otherwise the decoded rows in sorted-id order plus the id-to-slot map the lookup
/// needs, or the error that stopped the paging read.
type PagedTable = Option<Result<(Vec<f32>, BTreeMap<usize, usize>), EngineError>>;

fn read_paged_table(
    prefix: &str,
    src: &dyn WeightSource,
    ids: &BTreeSet<usize>,
    width: usize,
) -> PagedTable {
    let scale_width = width.div_ceil(32);
    let weight_name = format!("{prefix}.engram.embed.weight");
    let scale_name = format!("{prefix}.engram.embed.scale");
    let weight_ranges: Vec<(u64, usize)> = ids
        .iter()
        .map(|&id| (id as u64 * width as u64, width))
        .collect();
    let scale_ranges: Vec<(u64, usize)> = ids
        .iter()
        .map(|&id| (id as u64 * scale_width as u64, scale_width))
        .collect();
    let weight_bytes = match src.read_ranges(&weight_name, &weight_ranges)? {
        Ok(bytes) => bytes,
        Err(error) => return Some(Err(error.into())),
    };
    let scale_bytes = match src.read_ranges(&scale_name, &scale_ranges)? {
        Ok(bytes) => bytes,
        Err(error) => return Some(Err(error.into())),
    };
    let mut table = vec![0.0f32; ids.len() * width];
    let mut slot_of = BTreeMap::new();
    for (slot, &id) in ids.iter().enumerate() {
        slot_of.insert(id, slot);
        let weight = match quant::decode(
            DType::F8E4M3,
            &weight_bytes[slot * width..(slot + 1) * width],
            width,
        ) {
            Ok(values) => values,
            Err(error) => return Some(Err(error.into())),
        };
        let scales = match quant::decode(
            DType::F8E8M0,
            &scale_bytes[slot * scale_width..(slot + 1) * scale_width],
            scale_width,
        ) {
            Ok(values) => values,
            Err(error) => return Some(Err(error.into())),
        };
        for column in 0..width {
            table[slot * width + column] = weight[column] * scales[column / 32];
        }
    }
    Some(Ok((table, slot_of)))
}

/// Reads a dense fp8 weight and applies its 32x32 E8M0 grid.
///
/// # Errors
///
/// As [read_table], for the dense grid.
fn read_dense(
    name: &str,
    src: &dyn WeightSource,
    rows: usize,
    cols: usize,
    limit: usize,
) -> Result<Vec<f32>, EngineError> {
    let scale_name = name.replace("weight", "scale");
    let mut weight = linalg::decode_by_name(src, name, limit)?;
    let scales = linalg::decode_by_name(src, &scale_name, limit)?;
    linalg::apply_scales(&mut weight, rows, cols, &scales, ScaleGrid::Block32)?;
    Ok(weight)
}

/// Writes the n-gram lookup of one layer into the residual stream.
///
/// This is Engram.forward (model.py:350-365) for one chunk, on the INPUT of an Engram layer, before
/// that block runs (model.py:1262-1263):
///
/// 1. the hash row of each position fetches hash_columns rows of the fp8 table, dequantised per
///    32-column block and concatenated (model.py:312-320, 353);
/// 2. wkv turns them into one key per hc copy plus a single shared value (model.py:353-354);
/// 3. the gate is a normalised dot product of the stream against that key, with a signed square root
///    before the sigmoid (model.py:356-362);
/// 4. the write adds gate (x) value to every hc copy, in place (model.py:365, [engram_write]).
///
/// Returns false when the layer has no Engram table at all (model.py:1262), in which case nothing is
/// read and x_hc is untouched. A position whose hash row holds [HASH_SENTINEL] is skipped, which is
/// how the reference's zeroed gate (model.py:363-364) is reproduced without a mask argument.
///
/// hashes is either the whole buffer [ngram_hashes] produced - one row of
/// engram_layer_ids.len() * [hash_columns] ids per token - or just this layer's block of it.
///
/// # Errors
///
/// [EngineError::TooLarge] for a table above limit, [EngineError::Missing] for a weight the source does
/// not hold, and [EngineError::Shape] when a length disagrees with the configuration or a hash id falls
/// outside the table.
///
/// An id outside the table is a refusal rather than an index. The two numbers the id depends on come
/// from different config fields - the bucket primes start above engram_vocab_size, and the rows are
/// engram_num_embeddings - and the reference indexes the table with the id directly, so the layout
/// must FIT the table. [Layout::for_config] refuses a configuration where it does not, naming both
/// numbers, and this check is the second line of defence for a caller that assembled the ids itself.
pub fn engram_forward(
    cfg: &ModelConfig,
    layer: usize,
    x_hc: &mut [f32],
    hashes: &[u32],
    src: &dyn WeightSource,
    limit: usize,
) -> Result<bool, EngineError> {
    let Some(rows) = cfg.engram_rows(layer) else {
        return Ok(false);
    };
    let Some(layer_index) = cfg.engram_layer_ids.iter().position(|&id| id == layer) else {
        return Ok(false);
    };
    let dim = cfg.dim;
    let hc = cfg.hc_mult;
    let width = cfg.engram_head_dim;
    let columns = hash_columns(cfg);
    let hc_width = cfg.hc_width();
    if dim == 0 || hc == 0 || width == 0 || columns == 0 || hc_width == 0 {
        return Err(EngineError::Shape {
            what: "engram write".to_owned(),
            detail: format!(
                "the configuration gives dim {dim}, hc_mult {hc}, head_dim {width}, {columns} hash columns"
            ),
        });
    }
    if !x_hc.len().is_multiple_of(hc_width) {
        return Err(EngineError::Shape {
            what: "engram write".to_owned(),
            detail: format!("{} values are not whole rows of {hc_width}", x_hc.len()),
        });
    }
    let seqlen = x_hc.len() / hc_width;
    let layers = cfg.engram_layer_ids.len();
    let stride = if hashes.len() == seqlen * layers * columns {
        layers * columns
    } else if hashes.len() == seqlen * columns {
        columns
    } else {
        return Err(EngineError::Shape {
            what: "engram write".to_owned(),
            detail: format!(
                "{seqlen} tokens need {} hash ids (or {} for all {layers} layers), got {}",
                seqlen * columns,
                seqlen * layers * columns,
                hashes.len()
            ),
        });
    };
    let block = layer_index * columns;
    let prefix = format!("layers.{layer}");

    // Which rows this chunk actually looks up. A sentinel position is skipped entirely (the mask),
    // and the table is read through the paged path when the source can serve ranges: for the
    // released tables a whole read is 98 GB and a one-token chunk needs a few dozen rows. A chunk
    // whose every position is a sentinel therefore reads no table at all.
    let mut wanted: BTreeSet<usize> = BTreeSet::new();
    for token in 0..seqlen {
        let row = &hashes[token * stride + block..token * stride + block + columns];
        if row.contains(&HASH_SENTINEL) {
            continue;
        }
        for &id in row {
            let index = id as usize;
            if index >= rows {
                return Err(EngineError::Shape {
                    what: prefix.clone(),
                    detail: format!(
                        "hash id {index} is outside the {rows}-row Engram table: the bucket layout and the planned rows disagree"
                    ),
                });
            }
            wanted.insert(index);
        }
    }
    let (table, slot_of) = match read_paged_table(&prefix, src, &wanted, width) {
        Some(result) => {
            let (table, slot_of) = result?;
            (table, Some(slot_of))
        }
        None if wanted.is_empty() => (Vec::new(), Some(BTreeMap::new())),
        None => (read_table(&prefix, src, rows, width, limit)?, None),
    };
    let out_width = dim * (hc + 1);
    let fetched_width = columns * width;
    let wkv = read_dense(
        &format!("{prefix}.engram.wkv.weight"),
        src,
        out_width,
        fetched_width,
        limit,
    )?;
    // model.py:356: the two gate weights are only ever used as their product.
    let q_weight = linalg::decode_by_name(src, &format!("{prefix}.engram.q_weight"), limit)?;
    let k_weight = linalg::decode_by_name(src, &format!("{prefix}.engram.k_weight"), limit)?;
    if q_weight.len() != hc * dim || k_weight.len() != hc * dim {
        return Err(EngineError::Shape {
            what: format!("{prefix}.engram.q_weight"),
            detail: format!(
                "the hc-aware gate needs two {} value weights, got {} and {}",
                hc * dim,
                q_weight.len(),
                k_weight.len()
            ),
        });
    }
    let mut product = vec![0.0f32; hc * dim];
    for (out, (q, k)) in product.iter_mut().zip(q_weight.iter().zip(k_weight.iter())) {
        *out = q * k;
    }

    let eps = cfg.norm_eps as f32;
    let scale = 1.0 / (dim as f32).sqrt();
    let clamp = 1e-6f32;
    // Scratch, allocated once for the chunk rather than once per token.
    let mut fetched = vec![0.0f32; fetched_width];
    let mut kv = vec![0.0f32; out_width];
    let mut gate = vec![0.0f32; hc];
    for token in 0..seqlen {
        let row = &hashes[token * stride + block..token * stride + block + columns];
        if row.contains(&HASH_SENTINEL) {
            continue;
        }
        for (column, &id) in row.iter().enumerate() {
            let index = id as usize;
            if index >= rows {
                return Err(EngineError::Shape {
                    what: prefix.clone(),
                    detail: format!(
                        "hash id {index} is outside the {rows}-row Engram table: the bucket layout and the planned rows disagree"
                    ),
                });
            }
            let at = match &slot_of {
                Some(slots) => {
                    slots
                        .get(&index)
                        .copied()
                        .ok_or_else(|| EngineError::Shape {
                            what: prefix.clone(),
                            detail: format!("the paged Engram read has no row for hash id {index}"),
                        })?
                        * width
                }
                None => index * width,
            };
            fetched[column * width..(column + 1) * width].copy_from_slice(&table[at..at + width]);
        }
        linalg::matmul(&mut kv, &fetched, &wkv, out_width, fetched_width, 1)?;
        let (key, value) = kv.split_at(hc * dim);
        for copy in 0..hc {
            let stream_at = token * hc_width + copy * dim;
            let stream = &x_hc[stream_at..stream_at + dim];
            let key_copy = &key[copy * dim..(copy + 1) * dim];
            // model.py:359-360: both means are over the last dimension, per (token, hc copy), not
            // jointly over the copies; the dot product is then scaled by rstd and dim^-0.5.
            let mut stream_square = 0.0f32;
            let mut key_square = 0.0f32;
            let mut dot = 0.0f32;
            for column in 0..dim {
                let h = stream[column];
                let k = key_copy[column];
                stream_square += h * h;
                key_square += k * k;
                dot += h * product[copy * dim + column] * k;
            }
            let rstd = (1.0 / (stream_square / dim as f32 + eps).sqrt())
                * (1.0 / (key_square / dim as f32 + eps).sqrt());
            let dot = dot * rstd * scale;
            // model.py:362: the signed square root keeps the sign of the dot product.
            gate[copy] = linalg::sigmoid(dot.abs().max(clamp).sqrt().copysign(dot));
        }
        engram_write(
            cfg,
            &gate,
            value,
            &mut x_hc[token * hc_width..(token + 1) * hc_width],
        )?;
    }
    Ok(true)
}

/// Adds the Engram write to a buffer that already holds the stream: out += rows (x) cols.
///
/// rows is the per-hc-copy gate (one value per hc copy, the gate.unsqueeze(-1) of model.py:365) and
/// cols is the single shared value vector (value.unsqueeze(-2)), so the element at hc copy i and
/// column d gains rows[i] * cols[d]. Both are indexed by their own axis: a transposed pair - a value
/// per copy and a gate per column - would produce the same shape and different numbers, which is why
/// the axes are named in the signature.
///
/// # Errors
///
/// [EngineError::Shape] when rows is not one value per hc copy, cols is not dim long, or out_hc is not
/// one whole hc row.
pub fn engram_write(
    cfg: &ModelConfig,
    rows: &[f32],
    cols: &[f32],
    out_hc: &mut [f32],
) -> Result<(), EngineError> {
    let hc = cfg.hc_mult;
    let dim = cfg.dim;
    let hc_width = cfg.hc_width();
    if rows.len() != hc {
        return Err(EngineError::Shape {
            what: "engram write".to_owned(),
            detail: format!(
                "the gate is one value per hc copy: {hc} expected, {} given",
                rows.len()
            ),
        });
    }
    if cols.len() != dim {
        return Err(EngineError::Shape {
            what: "engram write".to_owned(),
            detail: format!("the shared value is {dim} long, {} given", cols.len()),
        });
    }
    if out_hc.len() != hc_width {
        return Err(EngineError::Shape {
            what: "engram write".to_owned(),
            detail: format!(
                "{} values are not one {hc_width}-value hc row",
                out_hc.len()
            ),
        });
    }
    for (copy, &weight) in rows.iter().enumerate() {
        for (column, &value) in cols.iter().enumerate() {
            out_hc[copy * dim + column] += weight * value;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::dtype::{DType, SCALE_BLOCK};
    use crate::linalg::REFERENCE_TENSOR_LIMIT;
    use crate::quant;
    use crate::synthetic::SyntheticWeights;
    use crate::tensor::{TensorPlan, TensorSpec};
    use crate::weights::WeightError;

    /// The element limit the reference pass uses by default (linalg::REFERENCE_TENSOR_LIMIT).
    const BIG: usize = REFERENCE_TENSOR_LIMIT;

    /// A tiny Engram whose bucket ranges fit inside its table.
    ///
    /// engram.py:112 starts the primes one above engram_vocab_size, so 20 gives the primes 23 and 29
    /// with offsets 0 and 23 and a largest id of 51, against the 64 rows a tiny configuration plans.
    /// ModelConfig::tiny as shipped has engram_vocab_size 100, whose primes 101 and 103 push the ids
    /// past those 64 rows - which is why a_hash_id_outside_the_table_is_refused_by_name can hand an id
    /// in directly instead of depending on the substitute multiplier stream.
    fn test_config() -> ModelConfig {
        ModelConfig {
            engram_vocab_size: 20,
            ..ModelConfig::tiny()
        }
    }

    /// A weight source whose tensors are written by hand, so an expectation can be computed rather
    /// than copied out of the code under test.
    struct Hand {
        held: BTreeMap<String, (DType, Vec<usize>, Vec<u8>)>,
    }

    impl Hand {
        fn new() -> Hand {
            Hand {
                held: BTreeMap::new(),
            }
        }

        fn put_f32(&mut self, name: &str, shape: &[usize], values: &[f32]) -> &mut Hand {
            let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            self.held
                .insert(name.to_owned(), (DType::F32, shape.to_vec(), bytes));
            self
        }

        fn put_e4m3(&mut self, name: &str, shape: &[usize], values: &[f32]) -> &mut Hand {
            let bytes = values.iter().map(|&v| quant::e4m3_encode(v)).collect();
            self.held
                .insert(name.to_owned(), (DType::F8E4M3, shape.to_vec(), bytes));
            self
        }

        fn put_e8m0(&mut self, name: &str, shape: &[usize], scales: &[f32]) -> &mut Hand {
            let bytes = scales.iter().map(|&s| quant::e8m0_encode_ceil(s)).collect();
            self.held
                .insert(name.to_owned(), (DType::F8E8M0, shape.to_vec(), bytes));
            self
        }
    }

    impl WeightSource for Hand {
        fn names(&self) -> Vec<String> {
            self.held.keys().cloned().collect()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.held
                .get(name)
                .map(|(dtype, shape, _)| TensorSpec {
                    name: name.to_owned(),
                    dtype: *dtype,
                    shape: shape.clone(),
                })
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            self.held
                .get(name)
                .map(|(_, _, data)| data.clone())
                .ok_or_else(|| WeightError::Missing {
                    name: name.to_owned(),
                })
        }
    }

    /// Every Engram tensor of a configuration, with the table rows all 1.0 and the wkv matrix chosen
    /// by the caller: key = fetched (rows below hc * dim) and value = fetched (rows above it).
    fn engram_source(cfg: &ModelConfig, weight_at: impl Fn(usize, usize) -> f32) -> Hand {
        let width = cfg.engram_head_dim;
        let fetched = hash_columns(cfg) * width;
        let out_width = cfg.dim * (cfg.hc_mult + 1);
        let mut hand = Hand::new();
        for (position, &layer) in cfg.engram_layer_ids.iter().enumerate() {
            let prefix = format!("layers.{layer}");
            let rows = cfg.engram_num_embeddings[position];
            let scale_width = width.div_ceil(SCALE_BLOCK);
            hand.put_e4m3(
                &format!("{prefix}.engram.embed.weight"),
                &[rows, width],
                &vec![1.0; rows * width],
            );
            hand.put_e8m0(
                &format!("{prefix}.engram.embed.scale"),
                &[rows, scale_width],
                &vec![1.0; rows * scale_width],
            );
            let mut wkv = Vec::with_capacity(out_width * fetched);
            for row in 0..out_width {
                for column in 0..fetched {
                    wkv.push(weight_at(row, column));
                }
            }
            hand.put_f32(
                &format!("{prefix}.engram.wkv.weight"),
                &[out_width, fetched],
                &wkv,
            );
            hand.put_e8m0(
                &format!("{prefix}.engram.wkv.scale"),
                &[
                    out_width.div_ceil(SCALE_BLOCK),
                    fetched.div_ceil(SCALE_BLOCK),
                ],
                &vec![1.0; out_width.div_ceil(SCALE_BLOCK) * fetched.div_ceil(SCALE_BLOCK)],
            );
            hand.put_f32(
                &format!("{prefix}.engram.q_weight"),
                &[cfg.hc_mult, cfg.dim],
                &vec![1.0; cfg.hc_mult * cfg.dim],
            );
            hand.put_f32(
                &format!("{prefix}.engram.k_weight"),
                &[cfg.hc_mult, cfg.dim],
                &vec![1.0; cfg.hc_mult * cfg.dim],
            );
        }
        hand
    }

    #[test]
    fn the_bucket_layout_is_the_reference_prime_draw() {
        let cfg = test_config();
        let layout = Layout::for_config(&cfg).expect("a consistent test configuration");
        // find_next_prime(19) is 23 and the next unused prime above it is 29 (engram.py:9-14, 112-116).
        assert_eq!(layout.primes, vec![vec![23, 29]]);
        assert_eq!(layout.offsets, vec![vec![0, 23]]);

        // Two layers keep drawing from one shared set, so no prime is handed out twice, and each
        // layer's offsets are its own running sum (engram.py:109-118, 149).
        let two = ModelConfig {
            engram_layer_ids: vec![0, 1],
            engram_num_embeddings: vec![64, 128],
            ..test_config()
        };
        let layout = Layout::for_config(&two).expect("a consistent test configuration");
        assert_eq!(layout.primes, vec![vec![23, 29], vec![31, 37]]);
        assert_eq!(layout.offsets, vec![vec![0, 23], vec![0, 31]]);

        // Every multiplier is odd and below 2 * (i64::MAX / compressed_vocab / 2), so a product with
        // any id cannot overflow (engram.py:71-72, 82), and the two layers get different streams
        // (engram.py:74-75).
        let bound = (i64::MAX / two.engram_compressed_vocab_size as i64 / 2) as i64;
        for layer in &layout.multipliers {
            assert_eq!(layer.len(), two.engram_max_ngram_size);
            for &multiplier in layer {
                assert!(multiplier > 0 && multiplier % 2 == 1 && multiplier < 2 * bound);
            }
        }
        assert_ne!(layout.multipliers[0], layout.multipliers[1]);

        // The released configuration has 3 orders x 8 heads = 24 columns; the tiny one has 1 x 2.
        assert_eq!(hash_columns(ModelConfig::v41_flash_cached()), 24);
        assert_eq!(hash_columns(&cfg), 2);
    }

    /// The documented hash rule, recomputed here rather than read out of the module (engram.py:171-184).
    ///
    /// The multiplier stream is shared with the module - it is the substitute the module docs describe -
    /// so this pins the formula (pad substitution, XOR accumulation, prime reduction, offsets) and not
    /// the reference's NumPy stream. A dead token contributes a row of sentinels, like the module.
    fn expected_hashes(cfg: &ModelConfig, tokens: &[u32]) -> Vec<u32> {
        let layout = Layout::for_config(cfg).expect("a consistent test configuration");
        let columns = hash_columns(cfg);
        let orders = cfg.engram_max_ngram_size;
        let layers = cfg.engram_layer_ids.len();
        let mut out = Vec::new();
        for position in 0..tokens.len() {
            if tokens[position] == HASH_SENTINEL {
                out.extend(std::iter::repeat_n(HASH_SENTINEL, layers * columns));
                continue;
            }
            let mut window = vec![0i64; orders];
            let mut blocked = false;
            for (shift, slot) in window.iter_mut().enumerate() {
                let before_start = position < shift;
                let source = if before_start {
                    cfg.engram_pad_id
                } else {
                    tokens[position - shift]
                };
                blocked = blocked || before_start || source == HASH_SENTINEL;
                *slot = i64::from(if blocked { cfg.engram_pad_id } else { source });
            }
            for layer in 0..layers {
                let mut rolling = window[0].wrapping_mul(layout.multipliers[layer][0]);
                for (order, &look_back) in window.iter().enumerate().skip(1) {
                    rolling ^= look_back.wrapping_mul(layout.multipliers[layer][order]);
                    for head in 0..cfg.engram_n_heads {
                        let column = (order - 1) * cfg.engram_n_heads + head;
                        let prime = layout.primes[layer][column] as i64;
                        let id = (rolling.rem_euclid(prime) as u64 + layout.offsets[layer][column])
                            as u32;
                        out.push(id);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn the_hash_follows_the_reference_formula() {
        let cfg = test_config();
        let columns = hash_columns(&cfg);
        let tokens = [5u32, 9, 3, 11];
        let mut out = Vec::new();
        ngram_hashes(&cfg, &tokens, 0, &mut out).expect("hashing a four-token sequence");
        assert_eq!(out.len(), tokens.len() * columns);

        assert_eq!(out, expected_hashes(&cfg, &tokens));

        // Position 0's second look-back is before the sequence start, so it takes the pad id:
        // expected_hashes is where that substitution is pinned, and this is the value itself.
        assert_eq!(cfg.engram_pad_id, 2);

        // A dead token's row is all sentinels, and the rows before it are untouched.
        let mut with_dead = tokens.to_vec();
        with_dead[2] = HASH_SENTINEL;
        let mut dead_out = Vec::new();
        ngram_hashes(&cfg, &with_dead, 0, &mut dead_out).expect("hashing a dead token");
        assert!(
            dead_out[2 * columns..3 * columns]
                .iter()
                .all(|&id| id == HASH_SENTINEL)
        );
        assert_eq!(&dead_out[..2 * columns], &out[..2 * columns]);
        // Position 3's look-back passes through that dead token, so its shifted id takes the pad:
        // dropping the pad substitution for a blocked look-back is exactly what this catches.
        assert_eq!(
            &dead_out[3 * columns..],
            &expected_hashes(&cfg, &with_dead)[3 * columns..]
        );

        // Causality: a later token cannot change an earlier position's ids.
        let mut later = tokens.to_vec();
        later[3] = 999;
        let mut later_out = Vec::new();
        ngram_hashes(&cfg, &later, 0, &mut later_out).expect("hashing a changed tail");
        assert_eq!(&later_out[..3 * columns], &out[..3 * columns]);

        // start_pos selects rows; the history before it still comes from the tokens handed in.
        let mut tail = Vec::new();
        ngram_hashes(&cfg, &tokens, 2, &mut tail).expect("hashing the tail only");
        assert_eq!(tail, out[2 * columns..].to_vec());
    }

    #[test]
    fn one_hash_pass_serves_every_engram_layer() {
        let cfg = ModelConfig {
            engram_layer_ids: vec![0, 1],
            engram_num_embeddings: vec![64, 128],
            ..test_config()
        };
        let columns = hash_columns(&cfg);
        let mut out = Vec::new();
        ngram_hashes(&cfg, &[1, 2, 3], 0, &mut out).expect("two Engram layers");
        assert_eq!(out.len(), 3 * 2 * columns);
        // Each layer reads its own block of the one buffer (model.py:1252), and every id falls inside
        // that layer's own bucket ranges - which is what makes the blocks addressable separately.
        let layout = Layout::for_config(&cfg).expect("a consistent test configuration");
        for token in 0..3 {
            for layer in 0..2 {
                for column in 0..columns {
                    let id = u64::from(out[token * 2 * columns + layer * columns + column]);
                    let offset = layout.offsets[layer][column];
                    assert!(
                        id >= offset && id < offset + layout.primes[layer][column],
                        "token {token} layer {layer} column {column} id {id} is outside [{offset}, {})",
                        offset + layout.primes[layer][column]
                    );
                }
            }
        }
    }

    #[test]
    fn a_layer_without_an_engram_reads_nothing() {
        let cfg = test_config();
        // A source that holds nothing at all: any read would fail, so this also shows no read happens.
        let empty = Hand::new();
        let mut x = vec![0.5f32; cfg.hc_width()];
        let before = x.clone();
        let wrote =
            engram_forward(&cfg, 0, &mut x, &[], &empty, BIG).expect("layer 0 has no Engram");
        assert!(!wrote);
        assert_eq!(x, before);
    }

    #[test]
    fn the_write_is_the_gate_times_the_value() {
        let cfg = test_config();
        let dim = cfg.dim;
        let gate = [2.0f32, -1.0];
        let value: Vec<f32> = (0..dim).map(|d| d as f32).collect();
        let mut out = vec![1.0f32; cfg.hc_width()];
        engram_write(&cfg, &gate, &value, &mut out).expect("a hand-built write");
        assert_eq!(out[0], 1.0 + 2.0 * 0.0);
        assert_eq!(out[1], 1.0 + 2.0 * 1.0);
        // Copy 1 uses gate[1] and shares the same value vector; the axes are not interchangeable.
        assert_eq!(out[dim], 1.0 - 1.0 * 0.0);
        assert_eq!(out[dim + 3], 1.0 - 1.0 * 3.0);

        // A zero gate writes nothing at all.
        let mut untouched = vec![0.25f32; cfg.hc_width()];
        let before = untouched.clone();
        engram_write(&cfg, &[0.0, 0.0], &value, &mut untouched).expect("a zero gate");
        assert_eq!(untouched, before);

        // Every length is checked, and the message says which one is wrong.
        let error =
            engram_write(&cfg, &[1.0], &value, &mut out).expect_err("one gate for two copies");
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        assert!(error.to_string().contains("hc copy"), "{error}");
        let error =
            engram_write(&cfg, &gate, &value[..dim - 1], &mut out).expect_err("a short value");
        assert!(error.to_string().contains("shared value"), "{error}");
        let mut short = vec![0.0f32; cfg.hc_width() - 1];
        let error = engram_write(&cfg, &gate, &value, &mut short).expect_err("a short row");
        assert!(error.to_string().contains("hc row"), "{error}");
    }

    #[test]
    fn the_gate_is_the_reference_normalised_dot_product() {
        let cfg = test_config();
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let fetched = hash_columns(&cfg) * cfg.engram_head_dim;
        let average = 1.0 / fetched as f32;

        // Case 1: wkv averages the fetched row, so every key and every value is 1.0. The stream is all
        // ones, so dot = dim, rstd = 1 and dim^-0.5 = 1/8: the gate is sigmoid(sqrt(8)).
        let ones = engram_source(&cfg, |_row, _column| average);
        let mut x = vec![1.0f32; cfg.hc_width()];
        assert!(
            engram_forward(&cfg, 1, &mut x, &[0, 0], &ones, BIG).expect("a full row of ones"),
            "layer 1 has an Engram"
        );
        let gate = 1.0 / (1.0 + (-(dim as f32).sqrt().sqrt()).exp());
        for (index, value) in x.iter().enumerate() {
            assert!(
                (value - (1.0 + gate)).abs() < 1e-6,
                "element {index} is {value}"
            );
        }

        // Case 2: only the first key element is non-zero, so mean(key^2) is 1/dim and rstd is 8. dot is
        // then 1 * 8 * 1/8 = 1: the gate is sigmoid(1). Copy 1's key is all zeros, which is where the
        // 1e-6 clamp on the dot product (model.py:341, 362) decides the result.
        let unit = engram_source(&cfg, |row, _column| {
            if row == 0 || row >= hc * dim {
                average
            } else {
                0.0
            }
        });
        let mut x = vec![1.0f32; cfg.hc_width()];
        engram_forward(&cfg, 1, &mut x, &[0, 0], &unit, BIG).expect("a one-hot key");
        let positive = 1.0 / (1.0 + (-1.0f32).exp());
        let clamped = 1.0 / (1.0 + (-1e-6f32.sqrt()).exp());
        for (index, value) in x.iter().enumerate() {
            let expected = if index < dim {
                1.0 + positive
            } else {
                1.0 + clamped
            };
            assert!(
                (value - expected).abs() < 1e-6,
                "element {index} is {value}"
            );
        }

        // Case 3: the same key negated makes the dot product negative. The signed square root keeps the
        // sign, so the gate is sigmoid(-1); dropping the sign would give sigmoid(1) instead.
        let negative = engram_source(&cfg, |row, _column| {
            if row == 0 {
                -average
            } else if row >= hc * dim {
                average
            } else {
                0.0
            }
        });
        let mut x = vec![1.0f32; cfg.hc_width()];
        engram_forward(&cfg, 1, &mut x, &[0, 0], &negative, BIG).expect("a negative dot product");
        let signed = 1.0 / (1.0 + 1.0f32.exp());
        for (index, value) in x.iter().enumerate().take(dim) {
            assert!(
                (*value - (1.0 + signed)).abs() < 1e-6,
                "element {index} is {value}"
            );
        }
    }

    #[test]
    fn a_sentinel_row_writes_nothing_and_the_pass_is_deterministic() {
        let cfg = test_config();
        let fetched = hash_columns(&cfg) * cfg.engram_head_dim;
        let source = engram_source(&cfg, |_row, _column| 1.0 / fetched as f32);

        // A dead position is left exactly as it was (model.py:363-364's zeroed gate).
        let mut skipped = vec![1.0f32; 2 * cfg.hc_width()];
        let before = skipped.clone();
        engram_forward(
            &cfg,
            1,
            &mut skipped,
            &[HASH_SENTINEL, HASH_SENTINEL, HASH_SENTINEL, HASH_SENTINEL],
            &source,
            BIG,
        )
        .expect("two dead positions");
        assert_eq!(skipped, before);

        // The same input twice is the same output: nothing here reads a clock or a random stream.
        let hashes = [1u32, 1];
        let mut first = vec![0.5f32; cfg.hc_width()];
        let mut second = first.clone();
        engram_forward(&cfg, 1, &mut first, &hashes, &source, BIG).expect("a live position");
        engram_forward(&cfg, 1, &mut second, &hashes, &source, BIG)
            .expect("the same position again");
        assert_eq!(first, second);
        assert_ne!(first, vec![0.5f32; cfg.hc_width()]);
    }

    #[test]
    fn a_hash_id_outside_the_table_is_refused_by_name() {
        // This test builds its own table with a row count it names here, so it asserts the refusal
        // rather than whatever ModelConfig::tiny() happens to hold: a test that breaks when an
        // unrelated configuration field moves is testing the wrong thing.
        let rows = 4_096usize;
        let cfg = ModelConfig {
            engram_num_embeddings: vec![rows],
            ..test_config()
        };
        let source = engram_source(&cfg, |_row, _column| 0.0);
        let mut x = vec![0.0f32; cfg.hc_width()];
        let outside = (rows + 1) as u32;
        let error = engram_forward(&cfg, 1, &mut x, &[outside, 0], &source, BIG)
            .expect_err("an id past the last row");
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        let message = error.to_string();
        assert!(message.contains(&outside.to_string()), "{message}");
        assert!(message.contains(&format!("{rows}-row")), "{message}");

        // A hash buffer of the wrong length is refused too, and says what it wanted.
        let error = engram_forward(&cfg, 1, &mut x, &[0, 0, 0], &source, BIG)
            .expect_err("three ids for one token");
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        assert!(error.to_string().contains("hash ids"), "{error}");
    }

    #[test]
    fn the_released_engram_table_is_refused_by_the_element_limit() {
        let cfg = ModelConfig::v41_flash_cached();
        let plan = TensorPlan::language_model(cfg);
        let source = SyntheticWeights::from_plan(&plan, 11);
        let mut x = vec![0.0f32; cfg.hc_width()];
        let hashes = vec![0u32; cfg.engram_layer_ids.len() * hash_columns(cfg)];
        let error = engram_forward(cfg, 1, &mut x, &hashes, &source, BIG)
            .expect_err("a 98 GB table is not materialised");
        match error {
            EngineError::TooLarge {
                name,
                elements,
                limit,
            } => {
                assert_eq!(name, "layers.1.engram.embed.weight");
                assert_eq!(elements, 98_305_579_008);
                assert_eq!(limit, BIG);
            }
            other => panic!("expected a size refusal, got {other}"),
        }
    }

    #[test]
    fn a_start_position_past_the_tokens_is_refused() {
        let cfg = test_config();
        let mut out = Vec::new();
        let error = ngram_hashes(&cfg, &[1, 2], 3, &mut out).expect_err("start past the end");
        assert!(matches!(error, EngineError::Shape { .. }), "{error}");
        assert!(error.to_string().contains("past"), "{error}");
    }

    /// The released configuration's own row counts ARE this layout's totals: the checkpoint's
    /// engram_num_embeddings were produced by exactly this prime draw, so this pins the algorithm
    /// against the checkpoint rather than against a golden this repository wrote for itself.
    #[test]
    fn the_released_layout_reproduces_the_checkpoint_row_counts() {
        let cfg = ModelConfig {
            engram_vocab_size: 16_000_000,
            engram_layer_ids: vec![1, 14],
            engram_num_embeddings: vec![384_006_168, 384_016_682],
            engram_max_ngram_size: 4,
            engram_n_heads: 8,
            ..test_config()
        };
        let layout = Layout::for_config(&cfg).expect("the released configuration is consistent");
        // (max_ngram_size - 1) orders x n_heads columns per layer (engram.py:109-118).
        assert_eq!(layout.primes[0].len(), 24);
        assert_eq!(layout.primes[0][0], 16_000_057);
        for (index, primes) in layout.primes.iter().enumerate() {
            let total: u64 = primes.iter().sum();
            assert_eq!(
                total, cfg.engram_num_embeddings[index] as u64,
                "layer {} must sum to its row count",
                cfg.engram_layer_ids[index]
            );
        }
        // One shared seen set across layers, so layer 14's primes start after layer 1's.
        assert!(layout.primes[1][0] > layout.primes[0][23]);
        assert_eq!(layout.offsets[0][23] + layout.primes[0][23], 384_006_168);
    }

    /// A table whose row count is below the layout's total is refused, not wrapped: the reference
    /// indexes the table with the id the hash produced (model.py:309, 317), so such an id has nowhere
    /// to go, and quietly reducing it would address a different row than the checkpoint's.
    ///
    /// This is the shape ModelConfig::tiny() is in - engram_vocab_size 100 with a 64-row table, whose
    /// primes 101 and 103 sum to 204 - which is why an integration pass over tiny sees hash id 130
    /// outside a 64-row table. The fix belongs to the configuration, not to a reduction here.
    #[test]
    fn a_table_smaller_than_the_bucket_layout_is_refused_by_name() {
        let small = ModelConfig {
            engram_vocab_size: 100,
            engram_num_embeddings: vec![64],
            ..test_config()
        };
        let error = Layout::for_config(&small).expect_err("64 rows cannot hold 204 buckets");
        assert!(matches!(error, EngineError::Config(_)), "{error}");
        let text = error.to_string();
        assert!(text.contains("64 rows"), "{text}");
        assert!(text.contains("204"), "{text}");
        assert!(text.contains("engram_num_embeddings"), "{text}");

        // A configuration whose rows cover the layout yields ids that are all inside the table, which
        // is the property the reference relies on.
        let cfg = test_config();
        let mut ids = Vec::new();
        ngram_hashes(&cfg, &[1, 2, 3], 1, &mut ids).expect("a consistent configuration hashes");
        let rows = cfg.engram_num_embeddings[0] as u32;
        assert!(
            ids.iter().all(|&id| id < rows),
            "every id must be inside the {rows}-row table: {ids:?}"
        );
    }

    /// The 32-bit words of a non-negative integer, lowest first (bit_generator.pyx:66-78).
    ///
    /// Written out again here with its own literals so that the checks below do not lean on the code
    /// they are checking.
    fn reference_limbs(value: u64) -> Vec<u32> {
        let mut remaining = value;
        let mut words = Vec::new();
        if remaining == 0 {
            return vec![0];
        }
        while remaining > 0 {
            words.push((remaining & 0xffff_ffff) as u32);
            remaining >>= 32;
        }
        words
    }

    /// A second copy of SeedSequence.mix_entropy for one integer entropy (bit_generator.pyx:56-64,
    /// 158-169, 351-383).
    fn reference_pool(limbs: &[u32]) -> [u32; 4] {
        let mut mixer = [0u32; 4];
        let mut hash_const: u32 = 0x43b0_d7e5;
        for (index, slot) in mixer.iter_mut().enumerate() {
            let word = limbs.get(index).copied().unwrap_or(0);
            let mut mixed = word ^ hash_const;
            hash_const = hash_const.wrapping_mul(0x931e_8875);
            mixed = mixed.wrapping_mul(hash_const);
            *slot = mixed ^ (mixed >> 16);
        }
        let mut source_index = 0;
        while source_index < 4 {
            let source = mixer[source_index];
            let mut destination = 0;
            while destination < 4 {
                if source_index != destination {
                    let mut mixed = source ^ hash_const;
                    hash_const = hash_const.wrapping_mul(0x931e_8875);
                    mixed = mixed.wrapping_mul(hash_const);
                    mixed ^= mixed >> 16;
                    let result = 0xca01_f9ddu32
                        .wrapping_mul(mixer[destination])
                        .wrapping_sub(0x4973_f715u32.wrapping_mul(mixed));
                    mixer[destination] = result ^ (result >> 16);
                }
                destination += 1;
            }
            source_index += 1;
        }
        mixer
    }

    /// A second copy of SeedSequence.generate_state(4, np.uint64) (bit_generator.pyx:413-460).
    fn reference_state_words(entropy: u64) -> [u64; 4] {
        let pool = reference_pool(&reference_limbs(entropy));
        let mut words = [0u32; 8];
        let mut hash_const: u32 = 0x8b51_f9dd;
        for (index, slot) in words.iter_mut().enumerate() {
            let mut value = pool[index % 4] ^ hash_const;
            hash_const = hash_const.wrapping_mul(0x58f3_8ded);
            value = value.wrapping_mul(hash_const);
            *slot = value ^ (value >> 16);
        }
        let mut state = [0u64; 4];
        for (index, pair) in words.as_chunks::<2>().0.iter().enumerate() {
            state[index] = u64::from(pair[0]) | (u64::from(pair[1]) << 32);
        }
        state
    }

    /// A second copy of the whole reference draw for one layer: PCG64 seeded from the SeedSequence
    /// (pcg64.c:20-34, pcg64.h) and then Lemire's bounded draw (distributions.c:1224-1258, 1425-1455).
    fn reference_multipliers(layer_id: u64, count: usize, bound: u64) -> Vec<i64> {
        const MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;
        let words = reference_state_words(10_007u64.wrapping_mul(layer_id));
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        let initseq = (u128::from(words[2]) << 64) | u128::from(words[3]);
        let inc = (initseq << 1) | 1;
        let mut state = inc;
        state = state.wrapping_add(initstate);
        state = state.wrapping_mul(MULTIPLIER).wrapping_add(inc);
        let draw = |state: &mut u128| -> u64 {
            *state = state.wrapping_mul(MULTIPLIER).wrapping_add(inc);
            let high = (*state >> 64) as u64;
            (high ^ (*state as u64)).rotate_right((high >> 58) as u32)
        };
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let mut scaled = u128::from(draw(&mut state)) * u128::from(bound);
            if (scaled as u64) < bound {
                let threshold = (u64::MAX - (bound - 1)) % bound;
                while (scaled as u64) < threshold {
                    scaled = u128::from(draw(&mut state)) * u128::from(bound);
                }
            }
            out.push(((scaled >> 64) as i64).wrapping_mul(2).wrapping_add(1));
        }
        out
    }

    #[test]
    fn the_limbs_are_the_low_words_first() {
        // bit_generator.pyx:66-78: 32-bit words, lowest first, and zero is one zero word.
        assert_eq!(numpy_rng::limbs(0), vec![0]);
        assert_eq!(numpy_rng::limbs(12_345), vec![12_345]);
        assert_eq!(numpy_rng::limbs(0x1_2345_6789), vec![0x2345_6789, 1]);
        assert_eq!(numpy_rng::limbs(u64::MAX), vec![0xffff_ffff, 0xffff_ffff]);
    }

    #[test]
    fn the_lcg_step_is_the_reference_closed_form() {
        // PCG64's state transition is a full-period LCG: x <- x * MULT + inc (mod 2**128), with MULT
        // being PCG_DEFAULT_MULTIPLIER_HIGH << 64 | LOW (pcg64.h) and inc = (initseq << 1) | 1
        // (pcg_setseq_128_srandom_r). The literals are written out here, so a changed constant in
        // numpy_rng breaks the comparison rather than moving both sides together.
        const MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;
        let words = [1u64, 2, 3, 4];
        let inc = ((u128::from(words[2]) << 64) | u128::from(words[3])) << 1 | 1;
        assert_eq!(inc & 1, 1, "the increment must be odd for the full period");
        let initstate = (u128::from(words[0]) << 64) | u128::from(words[1]);
        // srandom: state = 0, step, state += initstate, step.
        let mut expected = inc;
        expected = expected.wrapping_add(initstate);
        expected = expected.wrapping_mul(MULTIPLIER).wrapping_add(inc);

        let mut rng = numpy_rng::Pcg64::seeded(&words);
        for draw in 0..4 {
            expected = expected.wrapping_mul(MULTIPLIER).wrapping_add(inc);
            let high = (expected >> 64) as u64;
            let output = (high ^ (expected as u64)).rotate_right((high >> 58) as u32);
            assert_eq!(rng.next_u64(), output, "draw {draw}");
        }

        // The two-step jump identity x2 = M**2 * x + (M + 1) * inc, which no other pair satisfies.
        let start = 0x0123_4567_89ab_cdef_0011_2233_4455_6677u128;
        let stepped_twice = start
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(inc)
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(inc);
        let jumped = start
            .wrapping_mul(MULTIPLIER.wrapping_mul(MULTIPLIER))
            .wrapping_add(inc.wrapping_mul(MULTIPLIER.wrapping_add(1)));
        assert_eq!(stepped_twice, jumped);
    }

    #[test]
    fn the_pool_is_the_reference_seed_sequence_mix() {
        // layer_ids [1] means the reference seeds with 10007 * 1 (engram.py:75), one uint32 word.
        let entropy = numpy_rng::limbs(10_007);
        assert_eq!(entropy, vec![10_007]);
        assert_eq!(numpy_rng::pool(&entropy), reference_pool(&entropy));
        assert_eq!(
            numpy_rng::state_words(&numpy_rng::pool(&entropy)),
            reference_state_words(10_007)
        );
        // The layer id reaches the stream: a different entropy is a different pool.
        assert_ne!(numpy_rng::pool(&entropy), numpy_rng::pool(&[10_009]));
        assert_ne!(
            numpy_rng::state_words(&numpy_rng::pool(&entropy)),
            reference_state_words(10_009)
        );
    }

    #[test]
    fn the_stream_matches_a_second_copy_of_the_reference_chain() {
        let cfg = test_config();
        let bound = (i64::MAX / cfg.engram_compressed_vocab_size as i64 / 2) as u64;
        let layout = Layout::for_config(&cfg).expect("a consistent test configuration");
        assert_eq!(layout.multipliers.len(), 1, "one Engram layer");
        assert_eq!(layout.multipliers[0].len(), cfg.engram_max_ngram_size);
        assert_eq!(
            layout.multipliers[0],
            reference_multipliers(1, cfg.engram_max_ngram_size, bound)
        );
        // Determinism: the same configuration draws the same stream again.
        assert_eq!(
            Layout::for_config(&cfg)
                .expect("a consistent test configuration")
                .multipliers,
            layout.multipliers
        );
        // And the stream is per layer, not shared (engram.py:74-75).
        let two = ModelConfig {
            engram_layer_ids: vec![0, 1],
            engram_num_embeddings: vec![64, 128],
            ..test_config()
        };
        let two_layout = Layout::for_config(&two).expect("a consistent test configuration");
        assert_eq!(
            two_layout.multipliers[0],
            reference_multipliers(0, two.engram_max_ngram_size, bound)
        );
        assert_eq!(two_layout.multipliers[1], layout.multipliers[0]);
        assert_ne!(two_layout.multipliers[0], two_layout.multipliers[1]);
    }

    /// The paged read must produce exactly what the whole-table read produces. The wrapper refuses
    /// every whole-tensor read of the two Engram tables (like the CLI's loader does), so a fallback
    /// would fail the test instead of silently taking the other path.
    struct PagedEngram<'a> {
        inner: &'a Hand,
    }

    impl WeightSource for PagedEngram<'_> {
        fn names(&self) -> Vec<String> {
            self.inner.names()
        }

        fn spec(&self, name: &str) -> Result<TensorSpec, WeightError> {
            self.inner.spec(name)
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, WeightError> {
            if name.ends_with(".engram.embed.weight") || name.ends_with(".engram.embed.scale") {
                return Err(WeightError::Refused {
                    name: name.to_owned(),
                    detail: "the paged test source serves these tables by range only".to_owned(),
                });
            }
            self.inner.read(name)
        }

        fn read_ranges(
            &self,
            name: &str,
            ranges: &[(u64, usize)],
        ) -> Option<Result<Vec<u8>, WeightError>> {
            let table = self.inner.read(name).expect("the test holds the table");
            let mut out = Vec::new();
            for &(offset, len) in ranges {
                let start = offset as usize;
                out.extend_from_slice(&table[start..start + len]);
            }
            Some(Ok(out))
        }
    }

    #[test]
    fn the_paged_table_read_is_the_whole_table_read() {
        let cfg = test_config();
        let layer = cfg.engram_layer_ids[0];
        let width = cfg.engram_head_dim;
        let columns = hash_columns(&cfg);
        let rows = cfg.engram_num_embeddings[0];
        let hand = engram_source(&cfg, |row, column| {
            ((row * 31 + column * 7) % 29) as f32 / 7.0 - 2.0
        });
        // Two tokens, no sentinel: ids chosen inside the table.
        let hashes: Vec<u32> = (0..2 * columns)
            .map(|at| ((at * 97 + 13) % rows) as u32)
            .collect();
        let seqlen = 2;
        let base: Vec<f32> = (0..seqlen * cfg.hc_width())
            .map(|at| 1.0 + at as f32)
            .collect();
        let mut whole = base.clone();
        let mut paged = base.clone();
        let ran_whole = engram_forward(&cfg, layer, &mut whole, &hashes, &hand, 1 << 22)
            .expect("the whole-table read runs");
        let ran_paged = engram_forward(
            &cfg,
            layer,
            &mut paged,
            &hashes,
            &PagedEngram { inner: &hand },
            1 << 22,
        )
        .expect("the paged read runs");
        assert!(ran_whole);
        assert!(ran_paged);
        assert_eq!(
            whole, paged,
            "{seqlen} tokens of width {width} must be identical whether the {rows}-row table was read whole or row by row"
        );
    }
}
