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

//! Hardware-only checks.
//!
//! Every test in this file is ignored by default because it needs a real device or a real
//! allocation. Run one explicitly, for example:
//!
//! SPARKFLASH_KV_NVME_PATH=/mnt/nvme/kv.bin cargo test -p sparkflash-kv --test hardware -- --ignored --nocapture
//!
//! Running them is a deliberate act: they write to the named path and will allocate the
//! pool sizes they are given.

use std::sync::Arc;

use sparkflash_kv::layout::{GLOBAL_PAGE_BYTES, MemoryBudget, PageId};
use sparkflash_kv::tier::{FileBackedTier, Tier, TierBackend};
use sparkflash_kv::{KvCache, KvCacheConfig, KvStoreTelemetry};

fn pattern(page: u64) -> Vec<u8> {
    (0..GLOBAL_PAGE_BYTES)
        .map(|index| ((page as usize * 37 + index * 13) % 251) as u8)
        .collect()
}

#[test]
#[ignore = "needs SPARKFLASH_KV_NVME_PATH and explicit permission to write to a real device"]
fn a_real_nvme_file_tier_round_trips_with_the_devices_own_alignment() {
    let path = std::env::var("SPARKFLASH_KV_NVME_PATH")
        .expect("set SPARKFLASH_KV_NVME_PATH to a file on the target device");
    let path = std::path::PathBuf::from(path);
    let capacity = 8u64;
    let tier = FileBackedTier::open(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, capacity, true)
        .expect("open the real device");
    println!(
        "device alignment {} direct {} slot_bytes {}",
        tier.alignment(),
        tier.is_direct(),
        tier.slot_bytes()
    );
    assert!(
        tier.is_direct(),
        "the target device should support O_DIRECT"
    );
    for page in 0..4u64 {
        tier.write_page(PageId(page), &pattern(page))
            .expect("write");
    }
    tier.flush().expect("flush");
    for page in 0..4u64 {
        let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
        tier.read_page(PageId(page), &mut out).expect("read");
        assert_eq!(out, pattern(page), "page {page}");
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(tier.roster_path());
}

#[test]
#[ignore = "allocates the configured pool sizes for real"]
fn the_default_pool_can_be_instantiated_on_the_target_node() {
    let path = std::env::var("SPARKFLASH_KV_NVME_PATH")
        .expect("set SPARKFLASH_KV_NVME_PATH to a file on the target device");
    let budget = MemoryBudget::default();
    assert!(budget.usable_bytes() > 100 * (1 << 30));
    let config = KvCacheConfig {
        budget,
        uma_pages: 256,
        dram_pages: 256,
        nvme_pages: 256,
        swa_tail_slots: 4,
        nvme_path: Some(std::path::PathBuf::from(path)),
    };
    let cache = KvCache::new(config).expect("cache");
    assert_eq!(cache.snapshot().cache.total_kv_pages, 256);
    assert_eq!(cache.swa_capacity(), 4);
    let telemetry = Arc::new(KvStoreTelemetry::default());
    assert_eq!(telemetry.snapshot().page_inserts, 0);
}
