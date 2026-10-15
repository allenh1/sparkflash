//! Counter-based Philox-4x32-10 and the sync-free DSpark draft seeder.
//!
//! Spec 5.6.3 removes the per-step rank-0 draft-token broadcast by having all four ranks seed
//! the DSpark Markov head from one deterministic counter-based philox RNG. That is only sound
//! if the seed depends on nothing rank-local, so `draft_seed` takes no rank argument at all:
//! a draft is a pure function of `(step, request id, position)` plus the configured noise
//! token id, and two independent rank instances therefore agree without communicating.
//!
//! The generator is Random123's philox4x32-10: ten rounds of
//!
//! `c = (mulhi(M0, c0) ^ c1 ^ k0, mullo(M0, c0), mulhi(M1, c2) ^ c3 ^ k1, mullo(M1, c2))`
//!
//! with `M0 = 0xD2511F53`, `M1 = 0xCD9E8D57` and a key bumped by `W32 = 0x9E3779B9` and `M0`
//! every round. No published known-answer vector was reachable from this worktree, so the
//! tests here pin the properties the scheduler actually depends on - determinism,
//! avalanche, uniformity and lane uniqueness - rather than Random123's reference bit stream,
//! and the module says so instead of implying a KAT it does not have.

use crate::config::SchedulerConfig;
use crate::error::SchedulerError;

/// Philox multiplier for the even lanes.
pub const PHILOX_M0: u32 = 0xD251_1F53;
/// Philox multiplier for the odd lanes.
pub const PHILOX_M1: u32 = 0xCD9E_8D57;
/// Philox key schedule constant.
pub const PHILOX_W32: u32 = 0x9E37_79B9;
/// Number of rounds.
pub const PHILOX_ROUNDS: usize = 10;

fn mul_hi_lo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    ((product >> 32) as u32, product as u32)
}

fn philox_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mul_hi_lo(PHILOX_M0, counter[0]);
    let (hi1, lo1) = mul_hi_lo(PHILOX_M1, counter[2]);
    [
        hi1 ^ counter[1] ^ key[0],
        lo1,
        hi0 ^ counter[3] ^ key[1],
        lo0,
    ]
}

fn bump_key(key: [u32; 2]) -> [u32; 2] {
    [
        key[0].wrapping_add(PHILOX_W32),
        key[1].wrapping_add(PHILOX_M0),
    ]
}

/// A stateless counter-based Philox-4x32-10 stream.
///
/// ### Sync-freedom
/// The struct holds nothing but a key, a block counter and four buffered words; it has no
/// interior mutability, no locks and no reference to a transport. Two instances built from
/// the same operands produce the same words in the same order on any rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Philox4x32 {
    key: u64,
    block_counter: u64,
    lane: usize,
    buffered: [u32; 4],
}

impl Philox4x32 {
    /// A stream for one `(key, counter)` pair.
    #[must_use]
    pub const fn new(key: u64, counter: u64) -> Self {
        Self {
            key,
            block_counter: counter,
            lane: 4,
            buffered: [0; 4],
        }
    }

    /// The pure core: one 128-bit block for one counter value.
    #[must_use]
    pub fn block(key: u64, counter: u64) -> [u32; 4] {
        let mut key_words = [key as u32, (key >> 32) as u32];
        let mut words = [counter as u32, (counter >> 32) as u32, 0, 0];
        for round in 0..PHILOX_ROUNDS {
            words = philox_round(words, key_words);
            if round + 1 != PHILOX_ROUNDS {
                key_words = bump_key(key_words);
            }
        }
        words
    }

    /// The next block, advancing the counter by one.
    pub fn next_block(&mut self) -> [u32; 4] {
        let block = Self::block(self.key, self.block_counter);
        self.block_counter = self.block_counter.wrapping_add(1);
        self.lane = 4;
        block
    }

    /// The next word, drawing a fresh block every four words.
    pub fn next_u32(&mut self) -> u32 {
        if self.lane == 4 {
            self.buffered = self.next_block();
            self.lane = 0;
        }
        let word = self.buffered[self.lane];
        self.lane += 1;
        word
    }

    /// Fills a slice with consecutive words.
    pub fn fill_u32(&mut self, out: &mut [u32]) {
        for slot in out.iter_mut() {
            *slot = self.next_u32();
        }
    }

    /// The block counter this stream will use next.
    #[must_use]
    pub const fn counter(&self) -> u64 {
        self.block_counter
    }

    /// The stream key.
    #[must_use]
    pub const fn key(&self) -> u64 {
        self.key
    }
}

/// Splits a 64-bit value into mixed bits (splitmix64's finalizer).
fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The rank-free seed for one draft position.
///
/// The parameters deliberately contain no rank: the noise token id domain-separates the
/// drafter from every other philox user, the step and request id make the stream unique per
/// decode step and request, and the position picks the block. Two ranks that agree on those
/// four numbers agree on the seed bit for bit.
#[must_use]
pub fn draft_seed(noise_token_id: u32, step: u64, request_id: u64, position: usize) -> u64 {
    let domain = (u64::from(noise_token_id) << 32) | 0xD5A7_E5C0;
    mix64(domain ^ mix64(step.rotate_left(23) ^ mix64(request_id ^ (position as u64))))
}

/// The sync-free DSpark drafter: drafts that are identical on every rank by construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncFreeDrafter {
    noise_token_id: u32,
    block_size: usize,
    vocab: u32,
}

impl SyncFreeDrafter {
    /// A drafter for one configuration.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` when the vocabulary is empty or the block
    /// size is outside `1..=8`.
    pub fn new(noise_token_id: u32, block_size: usize, vocab: u32) -> Result<Self, SchedulerError> {
        if block_size == 0 || block_size > 8 {
            return Err(SchedulerError::InvalidArgument(format!(
                "draft block size {block_size} must be in 1..=8"
            )));
        }
        if vocab == 0 {
            return Err(SchedulerError::InvalidArgument(
                "the draft vocabulary must not be empty".to_owned(),
            ));
        }
        Ok(Self {
            noise_token_id,
            block_size,
            vocab,
        })
    }

    /// A drafter for the draft block size and noise token id of `config`.
    ///
    /// # Errors
    /// Returns `SchedulerError::InvalidArgument` when the vocabulary is empty.
    pub fn from_config(config: &SchedulerConfig, vocab: u32) -> Result<Self, SchedulerError> {
        Self::new(
            config.dspark_noise_token_id,
            config.dspark_block_size,
            vocab,
        )
    }

    /// The block size.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// The noise token id that domain-separates the draft stream.
    #[must_use]
    pub const fn noise_token_id(&self) -> u32 {
        self.noise_token_id
    }

    /// The seed for one draft position.
    #[must_use]
    pub fn seed(&self, step: u64, request_id: u64, position: usize) -> u64 {
        draft_seed(self.noise_token_id, step, request_id, position)
    }

    /// The token id proposed at one position.
    #[must_use]
    pub fn position_token(&self, step: u64, request_id: u64, position: usize) -> u32 {
        let block = Philox4x32::block(self.seed(step, request_id, position), position as u64);
        block[position % 4] % self.vocab
    }

    /// The draft: `block_size` token ids, identical on every rank.
    #[must_use]
    pub fn draft(&self, step: u64, request_id: u64) -> Vec<u32> {
        (0..self.block_size)
            .map(|position| self.position_token(step, request_id, position))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const KEY: u64 = 0x0123_4567_89AB_CDEF;

    #[test]
    fn the_pure_block_and_the_stateful_stream_agree() {
        let mut stream = Philox4x32::new(KEY, 9);
        for counter in 9..25 {
            assert_eq!(stream.next_block(), Philox4x32::block(KEY, counter));
        }
        assert_eq!(stream.counter(), 25);
        assert_eq!(stream.key(), KEY);
    }

    #[test]
    fn words_are_drawn_four_at_a_time_from_successive_blocks() {
        let mut stream = Philox4x32::new(KEY, 0);
        let mut words = [0u32; 12];
        stream.fill_u32(&mut words);
        for (index, word) in words.iter().enumerate() {
            assert_eq!(*word, Philox4x32::block(KEY, (index / 4) as u64)[index % 4]);
        }
    }

    #[test]
    fn the_same_operands_give_the_same_stream_on_every_rank() {
        let left = Philox4x32::new(KEY, 0);
        let right = Philox4x32::new(KEY, 0);
        assert_eq!(left, right);
        let mut left = left;
        let mut right = right;
        let mut a = [0u32; 64];
        let mut b = [0u32; 64];
        left.fill_u32(&mut a);
        right.fill_u32(&mut b);
        assert_eq!(a, b);
        assert_ne!(Philox4x32::block(KEY, 1), Philox4x32::block(KEY, 2));
        assert_ne!(Philox4x32::block(KEY, 1), Philox4x32::block(KEY + 1, 1));
    }

    #[test]
    fn flipping_one_counter_bit_changes_about_half_the_output_bits() {
        let base = Philox4x32::block(KEY, 42);
        let mut changed = 0u32;
        for bit in 0..64 {
            let other = Philox4x32::block(KEY, 42 ^ (1u64 << bit));
            for lane in 0..4 {
                changed += (base[lane] ^ other[lane]).count_ones();
            }
        }
        let total = 64 * 128;
        let fraction = f64::from(changed) / f64::from(total);
        assert!(
            (0.35..=0.65).contains(&fraction),
            "avalanche fraction {fraction} is not near one half"
        );
    }

    #[test]
    fn the_lanes_do_not_repeat_within_a_block_and_words_are_spread() {
        let mut stream = Philox4x32::new(KEY, 0);
        // Heap-backed: 400 KB of local array overflows the test thread's stack in debug.
        let mut words = vec![0u32; 4096];
        stream.fill_u32(&mut words);
        let unique: HashSet<u32> = words.iter().copied().collect();
        assert!(
            unique.len() >= 4090,
            "4096 draws collapsed to {} distinct words",
            unique.len()
        );
        let mut buckets = [0usize; 16];
        let mut sample = vec![0u32; 100_000];
        Philox4x32::new(KEY, 7).fill_u32(&mut sample);
        for word in sample {
            buckets[(word >> 28) as usize] += 1;
        }
        let expected = 100_000.0 / 16.0;
        for (index, count) in buckets.iter().enumerate() {
            let deviation = (*count as f64 - expected).abs() / expected;
            assert!(
                deviation < 0.15,
                "bucket {index} holds {count} words, expected about {expected}"
            );
        }
    }

    #[test]
    fn the_seed_carries_no_rank_and_separates_step_request_and_position() {
        let base = draft_seed(128_799, 5, 11, 0);
        assert_eq!(base, draft_seed(128_799, 5, 11, 0));
        assert_ne!(base, draft_seed(128_799, 6, 11, 0));
        assert_ne!(base, draft_seed(128_799, 5, 12, 0));
        assert_ne!(base, draft_seed(128_799, 5, 11, 1));
        assert_ne!(base, draft_seed(128_800, 5, 11, 0));
    }

    #[test]
    fn two_rank_instances_draft_identically_without_talking() {
        // Both drafters are built from the same configuration. There is no rank parameter to
        // differ in, which is the type-level half of the sync-free proof.
        let config = SchedulerConfig::default();
        let rank_zero = SyncFreeDrafter::from_config(&config, 129_280).expect("valid drafter");
        let rank_three = SyncFreeDrafter::from_config(&config, 129_280).expect("valid drafter");
        for step in 0..64 {
            let left = rank_zero.draft(step, 7);
            let right = rank_three.draft(step, 7);
            assert_eq!(left, right, "rank instances disagreed at step {step}");
            assert_eq!(left.len(), config.dspark_block_size);
            assert!(left.iter().all(|token| *token < 129_280));
        }
        assert_ne!(rank_zero.draft(3, 7), rank_zero.draft(4, 7));
        assert_ne!(rank_zero.draft(3, 7), rank_zero.draft(3, 8));
        assert_eq!(rank_zero.noise_token_id(), config.dspark_noise_token_id);
        assert_eq!(rank_zero.block_size(), 5);
    }

    #[test]
    fn a_bad_drafter_argument_is_refused() {
        assert!(SyncFreeDrafter::new(1, 0, 10).is_err());
        assert!(SyncFreeDrafter::new(1, 9, 10).is_err());
        assert!(SyncFreeDrafter::new(1, 5, 0).is_err());
        assert!(SyncFreeDrafter::new(1, 5, 10).is_ok());
    }
}
