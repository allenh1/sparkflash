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

//! Producer re-entrancy: a callback that reaches back into the cache must be refused, not
//! deadlocked.
//!
//! Both tests run the replay on a worker thread and fail on a timeout instead of hanging, so
//! a regression shows up as a failed assertion rather than a stuck test binary. The cache's
//! guard panics on the worker thread and the default hook prints one line naming
//! SwaProducer::fill_layer: that line is the guard firing, not a test failure, and the
//! assertions below check its text.

use std::sync::mpsc::{RecvTimeoutError, channel};
use std::time::Duration;

use sparkflash_kv::config::SWA_WINDOW_SIZE;
use sparkflash_kv::layout::MemoryBudget;
use sparkflash_kv::swa::{ReplayKind, SwaProducer};
use sparkflash_kv::{KvCache, KvCacheConfig, KvError};

const WORKER_TIMEOUT: Duration = Duration::from_secs(30);

fn small_cache() -> KvCache {
    let config = KvCacheConfig {
        budget: MemoryBudget::new(1 << 40, 1 << 20).expect("budget"),
        uma_pages: 2,
        dram_pages: 2,
        nvme_pages: 4,
        swa_tail_slots: 2,
        nvme_path: None,
    };
    KvCache::new(config).expect("cache")
}

fn prompt(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| {
            (index as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(12_345)
                % 100_003
        })
        .collect()
}

/// Runs an encoder replay whose producer calls `reenter`, and returns the panic message.
///
/// Panics with an explicit message when the replay deadlocks instead of being refused, so the
/// failure names the cause rather than the test harness timing out.
fn run_replay_that_reenters<F>(reenter: F) -> String
where
    F: Fn(&KvCache) + Send + 'static,
{
    struct Producer<'a, F> {
        cache: &'a KvCache,
        reenter: F,
    }

    impl<F: Fn(&KvCache)> SwaProducer for Producer<'_, F> {
        fn fill_layer(
            &mut self,
            _layer: usize,
            _window: &[u32],
            _out: &mut [u8],
        ) -> Result<(), KvError> {
            (self.reenter)(self.cache);
            Ok(())
        }
    }

    let cache = small_cache();
    let tokens = prompt(SWA_WINDOW_SIZE + 32);
    let (tx, rx) = channel();
    let worker = std::thread::spawn(move || {
        let mut producer = Producer {
            cache: &cache,
            reenter,
        };
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.bounded_replay(ReplayKind::Encoder, &tokens, tokens.len(), &mut producer)
        }));
        let message = match caught {
            Ok(outcome) => format!("the replay returned {outcome:?} instead of being refused"),
            Err(payload) => payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_string())
                })
                .unwrap_or_else(|| "panicked without a message".to_string()),
        };
        let _ = tx.send(message);
    });
    let message = match rx.recv_timeout(WORKER_TIMEOUT) {
        Ok(message) => message,
        Err(RecvTimeoutError::Timeout) => {
            panic!("re-entering the cache from a producer deadlocked: the guard did not fire")
        }
        Err(RecvTimeoutError::Disconnected) => {
            panic!("the worker thread died without reporting")
        }
    };
    worker.join().expect("the test worker itself must not fail");
    message
}

/// A producer that reads a snapshot takes the global-KV lock and then the SWA lock while the
/// replay already holds SWA, which deadlocks without the guard.
#[test]
fn a_producer_that_reads_a_snapshot_is_refused_not_deadlocked() {
    let message = run_replay_that_reenters(|cache| {
        let _ = cache.snapshot();
    });
    assert!(
        message.contains("SwaProducer::fill_layer"),
        "the guard must name the contract it enforced, got: {message}"
    );
}

/// A producer that probes the SWA ring takes the SWA lock a second time on the same thread,
/// which deadlocks without the guard.
#[test]
fn a_producer_that_probes_the_swa_ring_is_refused_not_deadlocked() {
    let message = run_replay_that_reenters(|cache| {
        let _ = cache.probe_swa_tail(&[7u32; SWA_WINDOW_SIZE]);
    });
    assert!(
        message.contains("SwaProducer::fill_layer"),
        "the guard must name the contract it enforced, got: {message}"
    );
}

/// The guard fires for one call and then clears: a producer that re-enters once and returns
/// does not leave the cache unusable for the next caller.
#[test]
fn the_guard_does_not_leak_into_the_next_replay() {
    let cache = small_cache();
    let tokens = prompt(SWA_WINDOW_SIZE);
    struct Counting {
        allowed: bool,
    }
    impl SwaProducer for Counting {
        fn fill_layer(
            &mut self,
            _layer: usize,
            window: &[u32],
            out: &mut [u8],
        ) -> Result<(), KvError> {
            assert!(self.allowed);
            for (slot, token) in window.iter().enumerate() {
                let base = slot * sparkflash_kv::layout::SWA_SLOT_BYTES;
                out[base] = *token as u8;
            }
            Ok(())
        }
    }
    let mut producer = Counting { allowed: true };
    let first = cache
        .bounded_replay(ReplayKind::Encoder, &tokens, tokens.len(), &mut producer)
        .expect("replay");
    assert_eq!(first.global_kv_pages_written, 0);
    assert_eq!(cache.store_telemetry().snapshot().replay_scopes, 1);
    // The scope is closed again, so the global-KV path is usable.
    cache
        .cache_prefix(&tokens, |_block, out| {
            out.fill(0);
            Ok(())
        })
        .expect("cache_prefix after a replay");
    assert!(cache.page_count() > 0);
}
