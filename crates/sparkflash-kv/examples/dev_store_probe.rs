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

//! Allocation/read-back/measurement probe for the Phase 2 device KV store (milestone 1).
//!
//! This example does no GPU compute and launches no kernel: it allocates the full live-geometry
//! store through the CUDA driver, measures the driver's free-memory delta, writes one 64-token
//! host page for all four source layers plus one window slot per layer and the two published
//! buffers, reads everything back, audits it, corrupts one stored byte and proves the audit
//! fails, then frees the store and reports the free-memory delta again.
//!
//! Usage: dev_store_probe [tokens]   (default 1048576 = MODEL_CONTEXT_TOKENS, D5)
//!
//! Build (aarch64, dev host, from the worktree root):
//!   scripts/build-aarch64.sh --print-env   # once, to install the pinned cross toolchain
//!   CARGO_TARGET_DIR=$PWD/target \
//!     CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=$HOME/opt/bin/sparkflash-aarch64-cc \
//!     CC_aarch64_unknown_linux_gnu=$HOME/opt/bin/sparkflash-aarch64-cc \
//!     CXX_aarch64_unknown_linux_gnu=$HOME/opt/bin/sparkflash-aarch64-cxx \
//!     AR_aarch64_unknown_linux_gnu=$HOME/opt/bin/sparkflash-aarch64-ar \
//!     $HOME/.cargo/bin/cargo +1.98.1 build --release --target aarch64-unknown-linux-gnu \
//!     -p sparkflash-kv --features cuda --example dev_store_probe

use sparkflash_kv::device_cuda::CudaMemory;
use sparkflash_kv::device_store::{
    DeviceKvStore, HSI_POOL_BYTES, StoreEntry, StoreError, StoreGeometry, StoreMemory,
};
use sparkflash_kv::fp4::{quantize_indexer_k, quantize_main_kv, quantize_swa_slot};
use sparkflash_kv::layout::{
    GLOBAL_PAGE_BYTES, INDEXER_K_DIM, INDEXER_K_ENTRY_BYTES, MAIN_KV_DIM, MAIN_KV_ENTRY_BYTES,
    SWA_SLOT_BYTES, SWA_SLOT_DIM, page_entry_offset,
};

const DEFAULT_TOKENS: usize = 1_048_576;

fn main() {
    if let Err(error) = run() {
        println!("RESULT: FAIL");
        eprintln!("probe failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let tokens = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_TOKENS);

    let memory = CudaMemory::load().map_err(boxed)?;
    println!("device: {}", memory.device_name());
    let free_before = memory.free_bytes().map_err(boxed)?;
    let device_total = memory.total_bytes().map_err(boxed)?;
    println!("DEVICE_TOTAL_BYTES {device_total}");
    println!("MEMFREE_BEFORE {free_before}");

    let geometry = StoreGeometry::for_max_seq_len(tokens).map_err(boxed)?;
    println!("GEOMETRY_TOKENS {}", geometry.max_seq_len());
    println!("GEOMETRY_PAGES {}", geometry.pages());
    println!("GEOMETRY_ENTRIES {}", geometry.total_entries());
    for source in geometry.sources() {
        println!(
            "SOURCE layer={} ratio={} page_entries={} entries={} entry_base={}",
            source.layer, source.ratio, source.page_entries, source.entries, source.entry_base
        );
    }
    for spec in geometry.buffer_plan() {
        println!("BUFFER {} {}", spec.name, spec.bytes);
    }
    println!("STORE_GLOBAL_KV_BYTES {}", geometry.global_kv_bytes());
    println!("STORE_WINDOW_BYTES {}", geometry.window_bytes());
    println!("STORE_HANDLE_BYTES {}", geometry.handle_bytes());
    println!("STORE_TOTAL_BYTES {}", geometry.total_bytes());
    println!(
        "STORE_KV_BYTES_PER_TOKEN {:.3}",
        geometry.kv_bytes_per_token()
    );

    let mut store = DeviceKvStore::new(memory, geometry.clone()).map_err(boxed)?;
    let free_after_alloc = store.memory().free_bytes().map_err(boxed)?;
    let accounted = store.allocated_bytes();
    println!("STORE_ALLOCATED_BYTES {accounted}");
    println!("STORE_BACKEND_ALLOCATIONS {}", store.memory().allocations());
    println!("MEMFREE_AFTER_ALLOC {free_after_alloc}");
    println!(
        "MEMFREE_ALLOC_DELTA {}",
        free_before.saturating_sub(free_after_alloc)
    );
    if accounted != geometry.total_bytes() {
        return Err(format!(
            "store accounting {accounted} != geometry total {}",
            geometry.total_bytes()
        )
        .into());
    }

    // One host page for all four source layers, page 0.
    let page_payload = build_page_payload(&geometry, 0);
    let mut expected = geometry.page_manifest(0, &page_payload).map_err(boxed)?;
    store.write_page(0, &page_payload).map_err(boxed)?;
    println!("PAGE_WRITTEN entries={}", expected.len());

    // One window slot (slot 0) per backbone layer.
    let mut window_entries = 0usize;
    for layer in 0..40 {
        let slot = build_window_slot(layer, 0);
        store.write_window_slot(layer, 0, &slot).map_err(boxed)?;
        expected.push(
            sparkflash_kv::device_store::AuditRow::from_stored(
                StoreEntry::Window { layer, slot: 0 },
                &slot,
            )
            .map_err(boxed)?,
        );
        window_entries += 1;
    }
    println!("WINDOW_WRITTEN entries={window_entries}");

    // The two published handles.
    let hsi = build_hsi_pool();
    store.publish_hsi_pool(&hsi).map_err(boxed)?;
    let selected: Vec<u32> = (0..512u32).collect();
    store.publish_selected(&selected).map_err(boxed)?;
    println!("HSI_POOL_BYTES_PUBLISHED {}", hsi.len());
    println!("SELECTED_PUBLISHED {}", selected.len());

    // Audit must pass on the clean store.
    store.audit(&expected).map_err(boxed)?;
    println!("AUDIT_CLEAN PASS entries={}", expected.len());

    // Read a page entry straight back and compare with the payload (layout/stride proof).
    let read_back = store
        .read_stored(StoreEntry::MainKv {
            layer: 2,
            position: 0,
        })
        .map_err(boxed)?;
    let page_entry =
        &page_payload[page_entry_offset(0, 0)..page_entry_offset(0, 0) + MAIN_KV_ENTRY_BYTES];
    println!("READBACK_IDENTICAL {}", read_back == page_entry);

    // Corruption: flip one bit of one stored entry; the audit must fail and name it.
    store
        .inject_corruption(
            StoreEntry::MainKv {
                layer: 14,
                position: 3,
            },
            0,
        )
        .map_err(boxed)?;
    match store.audit(&expected) {
        Ok(()) => return Err("audit passed after corruption; it must fail".into()),
        Err(StoreError::AuditMismatch { entry, .. }) => {
            println!("AUDIT_CORRUPTION_FAILED true");
            println!("AUDIT_CORRUPTION_ENTRY {entry:?}");
        }
        Err(other) => return Err(format!("unexpected corruption error: {other}").into()),
    }

    // Restore the corrupted entry from the payload and prove the audit passes again.
    let offset = page_entry_offset(2, 3);
    store
        .write_main_entry(14, 3, &page_payload[offset..offset + MAIN_KV_ENTRY_BYTES])
        .map_err(boxed)?;
    store.audit(&expected).map_err(boxed)?;
    println!("AUDIT_RESTORED PASS");

    // The kernel-ABI handles, printed so the sequence milestone can pin them.
    let source = store.source_handles(14).map_err(boxed)?;
    println!(
        "HANDLE layer14 kv_fp4=0x{:x} kv_scales=0x{:x} index_k_fp4=0x{:x} index_k_scales=0x{:x}",
        source.kv_fp4.address,
        source.kv_scales.address,
        source.index_k_fp4.address,
        source.index_k_scales.address
    );
    let resolved = store.source_handles(21).map_err(boxed)?;
    println!(
        "HANDLE layer21 resolves_to layer{} bytes={}",
        resolved.layer, resolved.kv_fp4.bytes
    );
    let window = store.window_handles(39).map_err(boxed)?;
    println!(
        "HANDLE window layer39 codes=0x{:x} scales=0x{:x}",
        window.codes.address, window.scales.address
    );
    let selected = store.selected_handle().map_err(boxed)?;
    println!(
        "HANDLE selected address=0x{:x} count={}",
        selected.address, selected.count
    );
    let hsi = store.hsi_pool_handle().map_err(boxed)?;
    println!(
        "HANDLE hsi_pool address=0x{:x} bytes={}",
        hsi.address, hsi.bytes
    );

    // Free everything and measure the driver's free memory again.
    let memory = store.into_memory();
    println!("STORE_ALLOCATED_AFTER_FREE {}", memory.allocated_bytes());
    let free_after_free = memory.free_bytes().map_err(boxed)?;
    println!("MEMFREE_AFTER_FREE {free_after_free}");
    println!(
        "MEMFREE_RELEASE_DELTA {}",
        free_after_free.saturating_sub(free_after_alloc)
    );
    drop(memory);

    println!("RESULT: PASS");
    Ok(())
}

fn boxed(error: impl std::fmt::Display) -> Box<dyn std::error::Error> {
    error.to_string().into()
}

/// Deterministic pseudo-random sample in about [-10, 10], so payloads are reproducible.
fn sample(seed: u64) -> f32 {
    let mut x = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (((x >> 40) % 2001) as f32 - 1000.0) / 100.0
}

fn entry_seed(layer: usize, position: u64, index: usize, tag: u64) -> u64 {
    (layer as u64)
        .wrapping_mul(0x9e37_79b9)
        .wrapping_add(position.wrapping_mul(0x85eb_ca6b))
        .wrapping_add((index as u64).wrapping_mul(0xc2b2_ae35))
        .wrapping_add(tag.wrapping_mul(0x27d4_eb2f))
}

fn build_page_payload(geometry: &StoreGeometry, page: u64) -> Vec<u8> {
    let mut payload = vec![0u8; GLOBAL_PAGE_BYTES];
    for (slot, source) in geometry.sources().iter().enumerate() {
        for entry in 0..source.page_entries {
            let position = page * source.page_entries + entry;
            let offset = page_entry_offset(slot, entry as usize);
            let mut main = [0.0f32; MAIN_KV_DIM];
            for (index, value) in main.iter_mut().enumerate() {
                *value = sample(entry_seed(source.layer, position, index, 1));
            }
            quantize_main_kv(&main)
                .expect("main kv sample quantises")
                .to_bytes(&mut payload[offset..offset + MAIN_KV_ENTRY_BYTES])
                .expect("main kv entry serialises");
            let mut index_values = [0.0f32; INDEXER_K_DIM];
            for (index, value) in index_values.iter_mut().enumerate() {
                *value = sample(entry_seed(source.layer, position, index, 2));
            }
            quantize_indexer_k(&index_values)
                .expect("indexer k sample quantises")
                .to_bytes(
                    &mut payload[offset + MAIN_KV_ENTRY_BYTES
                        ..offset + MAIN_KV_ENTRY_BYTES + INDEXER_K_ENTRY_BYTES],
                )
                .expect("indexer k entry serialises");
        }
    }
    payload
}

fn build_window_slot(layer: usize, slot: usize) -> Vec<u8> {
    let mut values = [0.0f32; SWA_SLOT_DIM];
    for (index, value) in values.iter_mut().enumerate() {
        *value = sample(entry_seed(layer, slot as u64, index, 3));
    }
    let mut out = vec![0u8; SWA_SLOT_BYTES];
    let stored = quantize_swa_slot(&values).expect("window sample quantises");
    out[..SWA_SLOT_DIM].copy_from_slice(&stored.data);
    out[SWA_SLOT_DIM..].copy_from_slice(&stored.scales);
    out
}

fn build_hsi_pool() -> Vec<u8> {
    let mut pool = vec![0u8; HSI_POOL_BYTES];
    for (index, byte) in pool.iter_mut().enumerate() {
        *byte = sample(index as u64) as u8;
    }
    pool
}
