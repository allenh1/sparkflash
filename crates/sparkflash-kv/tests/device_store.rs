//! CPU tests for the device KV store: layout, write protocol, bounds, handles and audit.
//!
//! Every test here runs without a GPU: the store's [StoreMemory] backend is a byte vector, so
//! plain `cargo test --workspace` exercises the page unpacking, the ratio-derived refusals and
//! the corruption-detecting audit exactly as the node run does against device memory.

use sparkflash_kv::device_store::{
    AuditRow, DeviceKvStore, HSI_POOL_BYTES, SourceShape, StoreEntry, StoreError, StoreGeometry,
    StoreMemory, WINDOW_RINGS, WINDOW_SLOTS,
};
use sparkflash_kv::fp4::{quantize_indexer_k, quantize_main_kv, quantize_swa_slot};
use sparkflash_kv::layout::{
    GLOBAL_PAGE_BYTES, INDEXER_K_DIM, INDEXER_K_ENTRY_BYTES, MAIN_KV_DIM, MAIN_KV_ENTRY_BYTES,
    SWA_SLOT_BYTES, SWA_SLOT_DIM, page_entry_offset,
};

/// A [StoreMemory] over owned byte vectors: the CPU stand-in for the CUDA driver.
#[derive(Default)]
struct HostMemory {
    buffers: Vec<Vec<u8>>,
    addresses: Vec<u64>,
    next_address: u64,
}

impl HostMemory {
    fn live_bytes(&self) -> u64 {
        self.buffers.iter().map(|buffer| buffer.len() as u64).sum()
    }
}

impl StoreMemory for HostMemory {
    type Buffer = usize;

    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer, StoreError> {
        let id = self.buffers.len();
        let address = self.next_address;
        self.next_address = self.next_address.wrapping_add(bytes as u64 + 256);
        self.buffers.push(vec![0u8; bytes]);
        self.addresses.push(address);
        Ok(id)
    }

    fn release(&mut self, buffer: Self::Buffer) {
        if let Some(slot) = self.buffers.get_mut(buffer) {
            *slot = Vec::new();
        }
    }

    fn write(
        &mut self,
        buffer: &Self::Buffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), StoreError> {
        let target = self.buffers.get_mut(*buffer).expect("live buffer");
        let start = usize::try_from(offset).map_err(|_| StoreError::TooLarge)?;
        let end = start.checked_add(bytes.len()).ok_or(StoreError::TooLarge)?;
        if end > target.len() {
            return Err(StoreError::OutOfBounds {
                what: "host backend write",
                value: end as u64,
                bound: target.len() as u64,
            });
        }
        target[start..end].copy_from_slice(bytes);
        Ok(())
    }

    fn read(&self, buffer: &Self::Buffer, offset: u64, out: &mut [u8]) -> Result<(), StoreError> {
        let source = self.buffers.get(*buffer).expect("live buffer");
        let start = usize::try_from(offset).map_err(|_| StoreError::TooLarge)?;
        let end = start.checked_add(out.len()).ok_or(StoreError::TooLarge)?;
        if end > source.len() {
            return Err(StoreError::OutOfBounds {
                what: "host backend read",
                value: end as u64,
                bound: source.len() as u64,
            });
        }
        out.copy_from_slice(&source[start..end]);
        Ok(())
    }

    fn address(&self, buffer: &Self::Buffer) -> u64 {
        self.addresses[*buffer]
    }

    fn allocated_bytes(&self) -> u64 {
        self.live_bytes()
    }
}

fn store_for(tokens: usize) -> DeviceKvStore<HostMemory> {
    let geometry = StoreGeometry::for_max_seq_len(tokens).expect("geometry");
    DeviceKvStore::new(HostMemory::default(), geometry).expect("store")
}

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
                .expect("main kv quantises")
                .to_bytes(&mut payload[offset..offset + MAIN_KV_ENTRY_BYTES])
                .expect("main kv serialises");
            let mut index_values = [0.0f32; INDEXER_K_DIM];
            for (index, value) in index_values.iter_mut().enumerate() {
                *value = sample(entry_seed(source.layer, position, index, 2));
            }
            quantize_indexer_k(&index_values)
                .expect("indexer k quantises")
                .to_bytes(
                    &mut payload[offset + MAIN_KV_ENTRY_BYTES
                        ..offset + MAIN_KV_ENTRY_BYTES + INDEXER_K_ENTRY_BYTES],
                )
                .expect("indexer k serialises");
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
    let stored = quantize_swa_slot(&values).expect("window quantises");
    out[..SWA_SLOT_DIM].copy_from_slice(&stored.data);
    out[SWA_SLOT_DIM..].copy_from_slice(&stored.scales);
    out
}

#[test]
fn page_zero_round_trips_and_audits() {
    let geometry = StoreGeometry::for_max_seq_len(128).expect("geometry");
    let mut store = DeviceKvStore::new(HostMemory::default(), geometry.clone()).expect("store");
    let payload = build_page_payload(&geometry, 0);
    let expected = geometry.page_manifest(0, &payload).expect("manifest");
    assert_eq!(expected.len(), 160 * 2);
    store.write_page(0, &payload).expect("write");
    for row in &expected {
        let stored = store.read_stored(row.entry).expect("read");
        let entry = match row.entry {
            StoreEntry::MainKv { layer, position } => {
                &payload[page_entry_offset(layer_slot(layer), position as usize)..]
                    [..MAIN_KV_ENTRY_BYTES]
            }
            StoreEntry::IndexerK { layer, position } => {
                &payload[(page_entry_offset(layer_slot(layer), position as usize)
                    + MAIN_KV_ENTRY_BYTES)..][..INDEXER_K_ENTRY_BYTES]
            }
            StoreEntry::Window { .. } => continue,
        };
        assert_eq!(stored.as_slice(), entry);
    }
    store.audit(&expected).expect("clean audit");
}

fn layer_slot(layer: usize) -> usize {
    [2usize, 8, 14, 20]
        .iter()
        .position(|candidate| *candidate == layer)
        .expect("full layer")
}

#[test]
fn page_bound_is_exact() {
    let geometry = StoreGeometry::for_max_seq_len(128).expect("geometry");
    let mut store = DeviceKvStore::new(HostMemory::default(), geometry.clone()).expect("store");
    let payload = build_page_payload(&geometry, 0);
    assert_eq!(geometry.pages(), 2);
    store.write_page(1, &payload).expect("last legal page");
    let error = store.write_page(2, &payload).expect_err("bound+1");
    assert!(matches!(
        error,
        StoreError::OutOfBounds {
            what: "page",
            value: 2,
            bound: 2,
        }
    ));
}

#[test]
fn source_bounds_are_exact() {
    let geometry = StoreGeometry::for_max_seq_len(128).expect("geometry");
    let mut store = DeviceKvStore::new(HostMemory::default(), geometry.clone()).expect("store");
    let main = [0u8; MAIN_KV_ENTRY_BYTES];
    let index = [0u8; INDEXER_K_ENTRY_BYTES];
    // Two pages: ratio-2 layers hold 64 entries, layer 20 holds 128.
    store
        .write_main_entry(2, 63, &main)
        .expect("last legal layer-2 entry");
    let error = store
        .write_main_entry(2, 64, &main)
        .expect_err("layer-2 bound+1");
    assert!(matches!(
        error,
        StoreError::LayerOutOfBounds {
            layer: 2,
            what: "main KV position",
            value: 64,
            bound: 64,
        }
    ));
    store
        .write_main_entry(20, 127, &main)
        .expect("last legal layer-20 entry");
    let error = store
        .write_main_entry(20, 128, &main)
        .expect_err("layer-20 bound+1");
    assert!(matches!(
        error,
        StoreError::LayerOutOfBounds {
            layer: 20,
            what: "main KV position",
            value: 128,
            bound: 128,
        }
    ));
    store
        .write_indexer_entry(14, 63, &index)
        .expect("last legal indexer entry");
    let error = store
        .write_indexer_entry(14, 64, &index)
        .expect_err("indexer bound+1");
    assert!(matches!(
        error,
        StoreError::LayerOutOfBounds {
            layer: 14,
            what: "indexer-K position",
            value: 64,
            bound: 64,
        }
    ));
}

#[test]
fn window_slot_bound_is_exact() {
    let mut store = store_for(128);
    let slot = build_window_slot(0, 0);
    store
        .write_window_slot(0, 127, &slot)
        .expect("last legal slot");
    let error = store
        .write_window_slot(0, 128, &slot)
        .expect_err("slot bound+1");
    assert!(matches!(
        error,
        StoreError::LayerOutOfBounds {
            layer: 0,
            what: "window slot",
            value: 128,
            bound: 128,
        }
    ));
    let error = store
        .write_window_slot(WINDOW_RINGS, 0, &slot)
        .expect_err("layer bound+1");
    assert!(matches!(
        error,
        StoreError::OutOfBounds {
            what: "window layer",
            value: 40,
            bound: 40,
        }
    ));
    assert_eq!(WINDOW_SLOTS, 128);
}

#[test]
fn wrong_lengths_and_wrong_layer_mode_are_refused() {
    let geometry = StoreGeometry::for_max_seq_len(128).expect("geometry");
    let mut store = DeviceKvStore::new(HostMemory::default(), geometry.clone()).expect("store");
    assert!(matches!(
        store.write_page(0, &[0u8; 10]),
        Err(StoreError::InvalidLength {
            context: "host page payload",
            expected: GLOBAL_PAGE_BYTES,
            actual: 10,
        })
    ));
    assert!(matches!(
        store.write_window_slot(0, 0, &[0u8; SWA_SLOT_BYTES - 1]),
        Err(StoreError::InvalidLength {
            context: "window slot",
            ..
        })
    ));
    assert!(matches!(
        store.write_main_entry(3, 0, &[0u8; MAIN_KV_ENTRY_BYTES]),
        Err(StoreError::WrongLayerMode {
            layer: 3,
            what: "main KV",
        })
    ));
    assert!(matches!(
        store.write_indexer_entry(0, 0, &[0u8; INDEXER_K_ENTRY_BYTES]),
        Err(StoreError::WrongLayerMode {
            layer: 0,
            what: "indexer K",
        })
    ));
}

#[test]
fn handles_resolve_and_publication_is_explicit() {
    let geometry = StoreGeometry::for_max_seq_len(128).expect("geometry");
    let mut store = DeviceKvStore::new(HostMemory::default(), geometry.clone()).expect("store");
    store.set_current_source(2).expect("full layer");
    assert_eq!(store.current_source(), Some(2));
    assert!(matches!(
        store.set_current_source(3),
        Err(StoreError::WrongLayerMode {
            layer: 3,
            what: "compressed KV",
        })
    ));
    assert_eq!(
        store.source_handles(0).err().map(|e| format!("{e}")),
        Some("layer 0 does not own compressed KV".to_string())
    );
    // Reuse layers resolve to the nearest preceding Full layer.
    assert_eq!(store.source_handles(3).expect("reuse resolves").layer, 2);
    assert_eq!(store.source_handles(7).expect("reuse resolves").layer, 2);
    assert_eq!(store.source_handles(19).expect("reuse resolves").layer, 14);
    assert_eq!(store.source_handles(21).expect("reuse resolves").layer, 20);
    assert_eq!(store.source_handles(39).expect("reuse resolves").layer, 20);
    // The four source layers have distinct array bases and the entry counts of their ratios.
    let layers: Vec<SourceShape> = store.geometry().sources().to_vec();
    let layer2 = store.source_handles(2).expect("layer 2");
    let layer20 = store.source_handles(20).expect("layer 20");
    assert_ne!(layer2.kv_fp4.address, layer20.kv_fp4.address);
    assert_eq!(layer2.kv_fp4.bytes, layers[0].entries * 256);
    assert_eq!(layer20.kv_fp4.bytes, layers[3].entries * 256);
    assert_eq!(layer2.kv_fp4.stride_bytes, 256);
    assert_eq!(layer2.kv_scales.stride_bytes, 32);
    assert_eq!(layer2.index_k_fp4.stride_bytes, 64);
    assert_eq!(layer2.index_k_scales.stride_bytes, 4);
    assert_ne!(
        store.window_handles(0).expect("window 0").codes.address,
        store.window_handles(1).expect("window 1").codes.address
    );
    assert_eq!(
        store
            .window_handles(0)
            .expect("window 0")
            .codes
            .stride_bytes,
        512
    );
    assert_eq!(
        store
            .window_handles(0)
            .expect("window 0")
            .scales
            .stride_bytes,
        16
    );
    assert!(matches!(
        store.window_handles(40),
        Err(StoreError::OutOfBounds {
            what: "window layer",
            ..
        })
    ));
    // Publication is explicit for both handles.
    assert!(matches!(
        store.selected_handle(),
        Err(StoreError::NotPublished { .. })
    ));
    assert!(matches!(
        store.hsi_pool_handle(),
        Err(StoreError::NotPublished { .. })
    ));
    store.publish_selected(&[7, 8, 9]).expect("publish");
    let selected = store.selected_handle().expect("published");
    assert_eq!(selected.count, 3);
    let error = store
        .publish_selected(&vec![1u32; 513])
        .expect_err("513 ids");
    assert!(matches!(
        error,
        StoreError::InvalidLength {
            context: "selected top-k list",
            ..
        }
    ));
    let error = store
        .publish_hsi_pool(&vec![0u8; HSI_POOL_BYTES - 1])
        .expect_err("short hsi pool");
    assert!(matches!(
        error,
        StoreError::InvalidLength {
            context: "HSI pool",
            ..
        }
    ));
    store
        .publish_hsi_pool(&vec![0u8; HSI_POOL_BYTES])
        .expect("publish hsi");
    assert_eq!(
        store.hsi_pool_handle().expect("published").bytes,
        HSI_POOL_BYTES as u64
    );
}

#[test]
fn corruption_is_detected_and_named() {
    let geometry = StoreGeometry::for_max_seq_len(128).expect("geometry");
    let mut store = DeviceKvStore::new(HostMemory::default(), geometry.clone()).expect("store");
    let payload = build_page_payload(&geometry, 0);
    let mut expected = geometry.page_manifest(0, &payload).expect("manifest");
    let slot = build_window_slot(5, 127);
    store
        .write_window_slot(5, 127, &slot)
        .expect("write window");
    expected.extend(
        [StoreEntry::Window {
            layer: 5,
            slot: 127,
        }]
        .map(|entry| AuditRow::from_stored(entry, &slot).expect("window audit row")),
    );
    store.write_page(0, &payload).expect("write page");
    store.audit(&expected).expect("clean audit");
    store
        .inject_corruption(
            StoreEntry::MainKv {
                layer: 8,
                position: 3,
            },
            0,
        )
        .expect("corrupt main kv");
    let error = store.audit(&expected).expect_err("corrupted audit");
    assert!(
        matches!(
            error,
            StoreError::AuditMismatch {
                entry: StoreEntry::MainKv {
                    layer: 8,
                    position: 3
                },
                ..
            }
        ),
        "unexpected error: {error}"
    );
    // Restore the main entry from the payload before corrupting the window, so the audit's
    // first mismatch is the entry this step is about.
    store
        .write_main_entry(
            8,
            3,
            &payload[page_entry_offset(1, 3)..][..MAIN_KV_ENTRY_BYTES],
        )
        .expect("restore main");
    store
        .inject_corruption(
            StoreEntry::Window {
                layer: 5,
                slot: 127,
            },
            500,
        )
        .expect("corrupt window");
    let error = store.audit(&expected).expect_err("corrupted window audit");
    assert!(matches!(
        error,
        StoreError::AuditMismatch {
            entry: StoreEntry::Window {
                layer: 5,
                slot: 127
            },
            ..
        }
    ));
    store
        .write_window_slot(5, 127, &slot)
        .expect("restore window");
    store.audit(&expected).expect("restored audit");
}

#[test]
fn accounting_and_release_are_exact() {
    let geometry = StoreGeometry::for_max_seq_len(128).expect("geometry");
    let store = DeviceKvStore::new(HostMemory::default(), geometry.clone()).expect("store");
    assert_eq!(store.allocated_bytes(), geometry.total_bytes());
    let memory = store.into_memory();
    assert_eq!(memory.allocated_bytes(), 0);
    assert_eq!(memory.live_bytes(), 0);
}
