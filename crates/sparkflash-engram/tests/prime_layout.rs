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

//! Re-derives the Engram bucket layout from scratch and compares it with the constants.
//!
//! Nothing here trusts `src/primes.rs`: the 48 primes are redrawn with an independent
//! deterministic Miller-Rabin implementation in the documented order, the offsets are
//! recomputed as running sums, and the multipliers are checked against the documented numpy
//! derivation. A change to any constant fails this test.

use std::collections::HashSet;

use sparkflash_engram::layout::{COMPRESSED_VOCAB_SIZE, ENGRAM_NUM_EMBEDDINGS, ENGRAM_NUM_HEADS};
use sparkflash_engram::primes::{
    ENGRAM_VOCAB_SIZE, MULTIPLIERS, OFFSETS, PRIMES, bucket_len, bucket_offset, bucket_range,
    column_index, flat_primes, layer_rows, multiplier_bound, multiplier_ceiling,
};

/// Bases sufficient for a deterministic Miller-Rabin test below 3.3e24.
const MILLER_RABIN_BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn pow_mod(mut base: u64, mut exponent: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = mul_mod(result, base, m);
        }
        base = mul_mod(base, base, m);
        exponent >>= 1;
    }
    result
}

/// Deterministic Miller-Rabin primality test for `u64`.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in MILLER_RABIN_BASES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut s = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }
    'witness: for a in MILLER_RABIN_BASES {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Smallest prime above `start` that has not been handed out yet.
fn find_next_prime(start: u64, seen: &mut HashSet<u64>) -> u64 {
    let mut candidate = start + 1;
    while !is_prime(candidate) || seen.contains(&candidate) {
        candidate += 1;
    }
    seen.insert(candidate);
    candidate
}

/// Redraws the whole 2 x 3 x 8 table in the reference's order.
fn derive_primes() -> [[[u32; ENGRAM_NUM_HEADS]; 3]; 2] {
    let mut seen = HashSet::new();
    let mut derived = [[[0u32; ENGRAM_NUM_HEADS]; 3]; 2];
    for layer in derived.iter_mut() {
        for order in layer.iter_mut() {
            let mut current = ENGRAM_VOCAB_SIZE as u64 - 1;
            for slot in order.iter_mut() {
                current = find_next_prime(current, &mut seen);
                *slot = current as u32;
            }
        }
    }
    derived
}

#[test]
fn miller_rabin_agrees_with_known_primes() {
    // Fixed points, including Carmichael numbers and the primes the table uses.
    for &p in &[
        2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 16_000_057, 16_000_889,
    ] {
        assert!(is_prime(p), "{p} should be prime");
    }
    for &c in &[
        0u64, 1, 4, 9, 15, 21, 25, 561, 1105, 1729, 2465, 2821, 6601, 16_000_059,
    ] {
        assert!(!is_prime(c), "{c} should be composite");
    }
}

#[test]
fn primes_are_rederived_exactly() {
    assert_eq!(derive_primes(), PRIMES);
}

#[test]
fn every_prime_is_prime_and_globally_unused() {
    let mut seen = HashSet::new();
    for (layer, layer_primes) in PRIMES.iter().enumerate() {
        for (order, order_primes) in layer_primes.iter().enumerate() {
            for (head, &prime) in order_primes.iter().enumerate() {
                let p = u64::from(prime);
                assert!(
                    is_prime(p),
                    "PRIMES[{layer}][{order}][{head}] = {p} is not prime"
                );
                assert!(seen.insert(p), "prime {p} is reused");
            }
        }
    }
    assert_eq!(seen.len(), 48);
}

#[test]
fn flat_primes_are_strictly_ascending() {
    let flat: Vec<u32> = (0..2).flat_map(flat_primes).collect();
    assert_eq!(flat.len(), 48);
    for pair in flat.windows(2) {
        assert!(pair[0] < pair[1], "{} is not below {}", pair[0], pair[1]);
    }
    assert_eq!(flat_primes(0), flat[..24].to_vec().as_slice());
    assert_eq!(flat_primes(1), flat[24..].to_vec().as_slice());
}

#[test]
fn layer_prime_sums_reproduce_the_config_row_counts() {
    assert_eq!(ENGRAM_NUM_EMBEDDINGS, [384_006_168, 384_016_682]);
    for (layer, &rows) in ENGRAM_NUM_EMBEDDINGS.iter().enumerate() {
        let sum: u64 = flat_primes(layer).iter().map(|&p| u64::from(p)).sum();
        assert_eq!(sum, rows, "layer {layer} primes sum to {sum}");
    }
}

#[test]
fn offsets_are_the_running_sum_of_the_primes() {
    for layer in 0..2 {
        let flat = flat_primes(layer);
        let mut running = 0u64;
        for column in 0..24 {
            assert_eq!(
                OFFSETS[layer][column], running,
                "layer {layer} column {column} offset"
            );
            running += flat[column] as u64;
        }
        // The last offset plus the last prime is exactly the layer's row count, and layer 14's
        // offsets (absent from the golden file) end at 384,016,682.
        assert_eq!(running, ENGRAM_NUM_EMBEDDINGS[layer]);
        assert_eq!(running, layer_rows(layer));
    }
    assert_eq!(OFFSETS[1][23], 368_015_793);
    assert_eq!(OFFSETS[0][23], 368_005_705);
}

#[test]
fn bucket_ranges_are_disjoint_and_inside_the_table() {
    for layer in 0..2 {
        let rows = ENGRAM_NUM_EMBEDDINGS[layer];
        let mut previous_end = 0u64;
        for order in 0..3 {
            for head in 0..ENGRAM_NUM_HEADS {
                let column = column_index(order, head);
                let range = bucket_range(layer, order, head);
                let prime = bucket_len(layer, order, head) as u64;
                assert_eq!(range.start, bucket_offset(layer, order, head));
                assert_eq!(range.start, OFFSETS[layer][column]);
                assert_eq!(range.end - range.start, prime);
                assert_eq!(
                    range.start, previous_end,
                    "layer {layer} column {column} gap"
                );
                assert!(
                    range.end <= rows,
                    "layer {layer} column {column} ends at {} past {rows}",
                    range.end
                );
                // prime + offset stays inside the row count
                assert!(prime + range.start <= rows);
                previous_end = range.end;
            }
        }
        assert_eq!(previous_end, rows, "layer {layer} buckets leave a hole");
    }
}

#[test]
fn offsets_are_strictly_increasing() {
    for (layer, offsets) in OFFSETS.iter().enumerate() {
        for (column, pair) in offsets.windows(2).enumerate() {
            assert!(pair[0] < pair[1], "layer {layer} offset {}", column + 1);
        }
    }
}

#[test]
fn multiplier_bound_matches_the_documented_arithmetic() {
    let expected = ((i64::MAX as u64) / COMPRESSED_VOCAB_SIZE as u64) / 2;
    assert_eq!(expected, 46_539_438_283_891);
    assert_eq!(multiplier_bound(), expected);
    assert_eq!(multiplier_bound(), 46_539_438_283_891);
    // The bound exists so a token id times a multiplier cannot overflow i64.
    let max_token = COMPRESSED_VOCAB_SIZE as u64 - 1;
    let max_multiplier = multiplier_ceiling();
    assert!(max_token * max_multiplier <= i64::MAX as u64);
}

#[test]
fn multipliers_are_odd_bounded_and_layer_distinct() {
    let bound = multiplier_bound();
    let ceiling = multiplier_ceiling();
    assert_eq!(ceiling, 93_078_876_567_781);
    for (layer, multipliers) in MULTIPLIERS.iter().enumerate() {
        for (shift, &m) in multipliers.iter().enumerate() {
            assert!(
                m > 0,
                "layer {layer} shift {shift} multiplier {m} is not positive"
            );
            assert_eq!(
                m % 2,
                1,
                "layer {layer} shift {shift} multiplier {m} is even"
            );
            // The reference stores values * 2 + 1, so the multiplier is odd and below
            // 2 * bound, not below bound itself.
            let drawn = (m as u64 - 1) / 2;
            assert!(
                drawn < bound,
                "layer {layer} shift {shift}: drawn value {drawn} is not below {bound}"
            );
            assert!(
                (m as u64) <= ceiling,
                "layer {layer} shift {shift} multiplier {m} is past the ceiling {ceiling}"
            );
        }
    }
    assert_ne!(MULTIPLIERS[0], MULTIPLIERS[1]);
    for (a, b) in MULTIPLIERS[0].iter().zip(MULTIPLIERS[1].iter()) {
        assert_ne!(a, b, "the two layers must not share a multiplier");
    }
}

#[test]
fn golden_multiplier_constants_are_pinned() {
    // Pinned from /home/user/dsv41-research/engram_golden.json, which was produced by
    // np.random.default_rng(10007 * layer_id).integers(0, bound, size=4). * 2 + 1.
    assert_eq!(
        MULTIPLIERS[0],
        [
            76_632_096_046_245,
            4_839_876_093_313,
            35_959_672_319_349,
            73_987_337_458_391
        ]
    );
    assert_eq!(
        MULTIPLIERS[1],
        [
            67_716_810_739_261,
            51_510_806_800_915,
            30_921_347_202_721,
            82_619_226_485_591
        ]
    );
}

#[test]
fn accessors_agree_with_the_tables() {
    for (layer, layer_primes) in PRIMES.iter().enumerate() {
        for (order, order_primes) in layer_primes.iter().enumerate() {
            for (head, &prime) in order_primes.iter().enumerate() {
                let column = column_index(order, head);
                assert_eq!(column, order * ENGRAM_NUM_HEADS + head);
                assert_eq!(bucket_len(layer, order, head), prime);
                assert_eq!(bucket_offset(layer, order, head), OFFSETS[layer][column]);
            }
        }
    }
}
