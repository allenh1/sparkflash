//! Tier backend and promotion/demotion tests: byte-exactness, alignment and persistence.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sparkflash_kv::layout::{GLOBAL_PAGE_BYTES, MemoryBudget, PageId};
use sparkflash_kv::telemetry::KvStoreTelemetry;
use sparkflash_kv::tier::{
    FileBackedTier, InMemoryTier, Tier, TierBackend, TieredPageStore, probe_direct_alignment,
};
use sparkflash_kv::{KvCache, KvCacheConfig, KvError};

/// Directory the file-backed tests use.
///
/// Defaults to the OS temporary directory, and setting SPARKFLASH_KV_TEST_DIR picks a
/// different filesystem for the O_DIRECT path. Neither choice selects the buffered fallback:
/// whether a filesystem accepts O_DIRECT is not implied by its type. Measured on this host,
/// tmpfs answers the probe with 512 and opens with O_DIRECT successfully, ext4 with 4096 and
/// ZFS with 512. The buffered fallback is exercised deterministically by
/// FileBackedTier::open_buffered, not by the choice of test directory.
fn test_dir() -> PathBuf {
    std::env::var("SPARKFLASH_KV_TEST_DIR").map_or_else(|_| std::env::temp_dir(), PathBuf::from)
}

fn pattern(page: u64) -> Vec<u8> {
    (0..GLOBAL_PAGE_BYTES)
        .map(|index| ((page as usize * 31 + index * 7) % 251) as u8)
        .collect()
}

fn in_memory_store(uma: u64, dram: u64, nvme: u64) -> (TieredPageStore, Arc<KvStoreTelemetry>) {
    let telemetry = Arc::new(KvStoreTelemetry::default());
    let store = TieredPageStore::new(
        Box::new(InMemoryTier::new(Tier::Uma, GLOBAL_PAGE_BYTES, uma)),
        Box::new(InMemoryTier::new(Tier::Dram, GLOBAL_PAGE_BYTES, dram)),
        Box::new(InMemoryTier::new(Tier::Nvme, GLOBAL_PAGE_BYTES, nvme)),
        Arc::clone(&telemetry),
    )
    .expect("matching page sizes");
    (store, telemetry)
}

/// A fresh file tier every time: these tests start from an empty pool, and recovery is a
/// raw-tier facility that a store refuses to adopt anyway (see TieredPageStore::new).
fn open_file_tier(directory: &Path, name: &str, capacity: u64) -> FileBackedTier {
    let path = directory.join(name);
    FileBackedTier::open(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, capacity, false).expect("open tier")
}

#[test]
fn in_memory_pages_round_trip_byte_exactly() {
    let (mut store, _) = in_memory_store(8, 8, 8);
    let protected = HashSet::new();
    for page in 0..4u64 {
        store
            .insert(PageId(page), &pattern(page), &protected)
            .expect("insert");
    }
    for page in 0..4u64 {
        let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
        let tier = store.read(PageId(page), &mut out).expect("read");
        assert_eq!(tier, Tier::Uma);
        assert_eq!(out, pattern(page));
    }
    assert_eq!(store.tracked_pages(), 4);
    assert_eq!(store.resident(Tier::Uma), 4);
    store.check_invariants().expect("store invariants");
}

#[test]
fn capacity_pressure_demotes_uma_to_dram_then_nvme_without_changing_a_byte() {
    let (mut store, telemetry) = in_memory_store(1, 1, 8);
    let protected = HashSet::new();
    // Page 0 fills UMA, page 1 pushes page 0 to DRAM, page 2 pushes page 1 to DRAM and
    // cascades page 0 to NVMe.
    store
        .insert(PageId(0), &pattern(0), &protected)
        .expect("p0");
    store
        .insert(PageId(1), &pattern(1), &protected)
        .expect("p1");
    store
        .insert(PageId(2), &pattern(2), &protected)
        .expect("p2");
    assert_eq!(store.location(PageId(0)).map(|l| l.tier), Some(Tier::Nvme));
    assert_eq!(store.location(PageId(1)).map(|l| l.tier), Some(Tier::Dram));
    assert_eq!(store.location(PageId(2)).map(|l| l.tier), Some(Tier::Uma));
    for page in 0..3u64 {
        let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
        let tier = store.read(PageId(page), &mut out).expect("read");
        let expected_tier = match page {
            0 => Tier::Nvme,
            1 => Tier::Dram,
            _ => Tier::Uma,
        };
        assert_eq!(tier, expected_tier, "page {page}");
        assert_eq!(out, pattern(page), "page {page} payload");
    }
    store.check_invariants().expect("store invariants");
    let snapshot = telemetry.snapshot();
    assert_eq!(snapshot.page_inserts, 3);
    assert_eq!(snapshot.demotions, 3);
    assert_eq!(snapshot.bytes_written, 6 * GLOBAL_PAGE_BYTES as u64);
}

#[test]
fn promotion_from_nvme_stages_through_dram_byte_exactly() {
    let (mut store, telemetry) = in_memory_store(1, 1, 8);
    let protected = HashSet::new();
    for page in 0..3u64 {
        store
            .insert(PageId(page), &pattern(page), &protected)
            .expect("insert");
    }
    assert_eq!(store.location(PageId(0)).map(|l| l.tier), Some(Tier::Nvme));
    let from = store.promote(PageId(0), &protected).expect("promote");
    assert_eq!(from, Tier::Nvme, "a Tier-2 restore must report Nvme");
    assert_eq!(store.location(PageId(0)).map(|l| l.tier), Some(Tier::Uma));
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    store.read(PageId(0), &mut out).expect("read");
    assert_eq!(out, pattern(0));
    // Page 1 was displaced out of DRAM to make the staging slot, and page 2 left UMA.
    assert!(telemetry.snapshot().nvme_reads >= 1);
    // One restore, one promotion: the DRAM hop is staging inside the same restore, not a
    // second page arriving in UMA.
    assert_eq!(telemetry.snapshot().uma_promotions, 1);
    // The regression that motivated check_invariants: the NVMe source copy must be gone, so
    // only the page the cascade displaced is left in the terminal tier.
    assert_eq!(store.resident(Tier::Nvme), 1);
    assert_eq!(store.tracked_pages(), 3);
    store.check_invariants().expect("store invariants");
}

#[test]
fn a_full_terminal_tier_refuses_instead_of_dropping_a_page() {
    let (mut store, telemetry) = in_memory_store(1, 0, 0);
    let protected = HashSet::new();
    store
        .insert(PageId(0), &pattern(0), &protected)
        .expect("first page fits");
    let error = store
        .insert(PageId(1), &pattern(1), &protected)
        .expect_err("second page cannot be staged");
    assert!(matches!(
        error,
        sparkflash_kv::KvError::CapacityExceeded { .. }
    ));
    assert_eq!(store.tracked_pages(), 1);
    assert!(telemetry.snapshot().capacity_refusals >= 1);
    store.check_invariants().expect("store invariants");
}

#[test]
fn a_protected_page_is_never_chosen_as_the_victim() {
    let (mut store, _) = in_memory_store(1, 1, 1);
    let mut protected = HashSet::new();
    store
        .insert(PageId(0), &pattern(0), &protected)
        .expect("p0");
    protected.insert(PageId(0));
    let error = store.insert(PageId(1), &pattern(1), &protected);
    assert!(error.is_err(), "both tiers hold the only protected page");
    assert_eq!(store.location(PageId(0)).map(|l| l.tier), Some(Tier::Uma));
    store.check_invariants().expect("store invariants");
}

#[test]
fn the_buffered_fallback_round_trips_and_persists_without_direct_io() {
    // The buffered path is selected explicitly rather than by hoping the test directory is on
    // a filesystem that refuses O_DIRECT: tmpfs accepts it on this host, so a test that relied
    // on the temporary directory would silently cover the direct path twice.
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let path = directory.join(format!("kv-buffered-{}.bin", std::process::id()));
    let roster = PathBuf::from(format!("{}.roster", path.to_string_lossy()));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
    let tier = FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, false)
        .expect("open");
    assert!(!tier.is_direct(), "open_buffered must not use O_DIRECT");
    assert_eq!(tier.alignment(), 1);
    assert_eq!(
        tier.slot_bytes(),
        GLOBAL_PAGE_BYTES as u64,
        "a buffered slot is one page, with no alignment padding"
    );
    tier.write_page(PageId(1), &pattern(1)).expect("write");
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    tier.read_page(PageId(1), &mut out).expect("read");
    assert_eq!(out, pattern(1), "buffered round trip changed the payload");
    tier.persist().expect("persist");
    drop(tier);

    let reopened = FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, true)
        .expect("reopen buffered");
    assert!(!reopened.is_direct());
    assert!(reopened.contains(PageId(1)));
    assert_eq!(reopened.resident_pages(), 1);
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    reopened
        .read_page(PageId(1), &mut out)
        .expect("read after reopen");
    assert_eq!(
        out,
        pattern(1),
        "the buffered path lost the payload across a reopen"
    );
    assert!(reopened.read_page(PageId(0), &mut out).is_err());
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}

#[test]
fn a_global_kv_publish_inside_a_replay_scope_is_refused_and_counted() {
    let (mut store, telemetry) = in_memory_store(4, 4, 4);
    let protected = HashSet::new();
    assert_eq!(telemetry.snapshot().replay_global_kv_writes, 0);
    assert_eq!(telemetry.snapshot().replay_scopes, 0);

    let scope = store.begin_replay();
    assert!(store.in_replay());
    assert_eq!(store.replay_depth(), 1);
    let error = store
        .insert(PageId(0), &pattern(0), &protected)
        .expect_err("a replay must not publish global KV");
    assert!(
        matches!(error, KvError::ReplayWroteGlobalKv(PageId(0))),
        "{error}"
    );
    assert_eq!(telemetry.snapshot().replay_global_kv_writes, 1);
    assert_eq!(telemetry.snapshot().replay_scopes, 1);
    // Refused means refused: nothing was written, tracked or counted as an insert.
    assert!(!store.contains(PageId(0)));
    assert_eq!(store.resident(Tier::Uma), 0);
    assert_eq!(telemetry.snapshot().page_inserts, 0);
    store.check_invariants().expect("store invariants");
    drop(scope);
    assert!(!store.in_replay());
    assert_eq!(store.replay_depth(), 0);

    // Outside the scope the same publish succeeds and does not touch the replay counter.
    store
        .insert(PageId(0), &pattern(0), &protected)
        .expect("insert");
    let snapshot = telemetry.snapshot();
    assert_eq!(
        snapshot.replay_global_kv_writes, 1,
        "a healthy publish is not a replay write"
    );
    assert_eq!(snapshot.page_inserts, 1);
    store.check_invariants().expect("store invariants");
}

#[test]
fn a_replay_scope_only_reserves_the_thread_that_opened_it() {
    // The scope is per thread so that a concurrent publisher is not mistaken for a replay
    // writing global KV. Without the thread filter this test counts two writes instead of one.
    let (store, telemetry) = in_memory_store(4, 4, 4);
    let store = std::sync::Arc::new(parking_lot::Mutex::new(store));
    let protected = HashSet::new();
    let scope = store.lock().begin_replay();
    assert!(store.lock().in_replay());
    let other = std::sync::Arc::clone(&store);
    let worker = std::thread::spawn(move || {
        let mut guard = other.lock();
        assert!(!guard.in_replay(), "the scope leaked to another thread");
        guard
            .insert(PageId(9), &pattern(9), &HashSet::new())
            .expect("another thread publishes normally");
    });
    worker.join().expect("worker");
    assert_eq!(
        telemetry.snapshot().replay_global_kv_writes,
        0,
        "another thread's publish is not a replay write"
    );
    let error = store
        .lock()
        .insert(PageId(0), &pattern(0), &protected)
        .expect_err("the owning thread is still in scope");
    assert!(
        matches!(error, KvError::ReplayWroteGlobalKv(PageId(0))),
        "{error}"
    );
    assert_eq!(telemetry.snapshot().replay_global_kv_writes, 1);
    drop(scope);
    assert!(!store.lock().in_replay());
    store.lock().check_invariants().expect("store invariants");
}

#[test]
fn mismatched_backend_page_sizes_are_rejected() {
    let telemetry = Arc::new(KvStoreTelemetry::default());
    let result = TieredPageStore::new(
        Box::new(InMemoryTier::new(Tier::Uma, GLOBAL_PAGE_BYTES, 4)),
        Box::new(InMemoryTier::new(Tier::Dram, GLOBAL_PAGE_BYTES / 2, 4)),
        Box::new(InMemoryTier::new(Tier::Nvme, GLOBAL_PAGE_BYTES, 4)),
        telemetry,
    );
    assert!(result.is_err());
}

#[test]
fn the_alignment_probe_asks_the_device_and_never_assumes_512() {
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let probe = probe_direct_alignment(&directory).expect("probe");
    println!(
        "probe_direct_alignment({}) = {probe:?}",
        directory.display()
    );
    match probe {
        Some(alignment) => {
            assert!(alignment >= 512, "reported alignment {alignment}");
            assert!(
                alignment.is_power_of_two(),
                "alignment {alignment} is not a power of two"
            );
        }
        None => {
            // This filesystem refuses O_DIRECT, so the tier below must report buffered mode;
            // the Some arm is the normal case on every filesystem measured here, tmpfs
            // included, which is why open_buffered exists to test the fallback at all.
            assert!(!open_file_tier(&directory, "probe-alignment.bin", 2).is_direct());
        }
    }
    let tier = open_file_tier(&directory, "probe-alignment.bin", 2);
    println!(
        "FileBackedTier direct={} alignment={} slot_bytes={}",
        tier.is_direct(),
        tier.alignment(),
        tier.slot_bytes()
    );
    assert!(tier.alignment() >= 1);
    let _ = std::fs::remove_file(directory.join("probe-alignment.bin"));
    let _ = std::fs::remove_file(directory.join("probe-alignment.bin.roster"));
    if tier.is_direct() {
        assert_eq!(tier.slot_bytes() % tier.alignment(), 0);
        assert!(tier.slot_bytes() >= GLOBAL_PAGE_BYTES as u64);
    } else {
        assert_eq!(tier.slot_bytes(), GLOBAL_PAGE_BYTES as u64);
    }
}

#[test]
fn file_backed_pages_survive_a_flush_and_a_reopen() {
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let name = format!("kv-persist-{}.bin", std::process::id());
    let page_bytes = GLOBAL_PAGE_BYTES;
    let path = directory.join(&name);
    let capacity = 4u64;
    let roster = PathBuf::from(format!("{}.roster", path.to_string_lossy()));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
    // One three-page prefix through one store: the third insert cascades the oldest page all
    // the way to the file-backed tier, which is the path that has to survive a reopen. The
    // store opens its file tier for a clean start, because the index is not persisted.
    {
        let protected = HashSet::new();
        let mut store = TieredPageStore::new(
            Box::new(InMemoryTier::new(Tier::Uma, page_bytes, 1)),
            Box::new(InMemoryTier::new(Tier::Dram, page_bytes, 1)),
            Box::new(
                FileBackedTier::open(Tier::Nvme, &path, page_bytes, capacity, false).expect("nvme"),
            ),
            Arc::new(KvStoreTelemetry::default()),
        )
        .expect("store");
        for page in 0..3u64 {
            store
                .insert(PageId(page), &pattern(page), &protected)
                .expect("insert");
        }
        assert_eq!(
            store.location(PageId(0)).map(|entry| entry.tier),
            Some(Tier::Nvme),
            "the oldest page must be the one that reached the file tier"
        );
        store.check_invariants().expect("store invariants");
        store.flush().expect("store flush");
    }
    assert!(roster.exists(), "flush must persist a roster");
    assert!(roster.to_string_lossy().ends_with(".bin.roster"));
    let reopened =
        FileBackedTier::open(Tier::Nvme, &path, page_bytes, capacity, true).expect("reopen tier");
    assert!(reopened.contains(PageId(0)));
    let mut out = vec![0u8; page_bytes];
    reopened
        .read_page(PageId(0), &mut out)
        .expect("read after reopen");
    assert_eq!(out, pattern(0), "page 0 changed across a reopen");
    let mut untouched = vec![0u8; page_bytes];
    assert!(reopened.read_page(PageId(2), &mut untouched).is_err());
    // Only the one slot that reached the file tier may be in the roster.
    assert_eq!(reopened.resident_pages(), 1);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}

#[test]
fn a_corrupt_roster_is_loud_not_silent() {
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let path = directory.join(format!("kv-corrupt-{}.bin", std::process::id()));
    let roster = directory.join(format!("kv-corrupt-{}.bin.roster", std::process::id()));
    std::fs::write(&path, [0u8; 8]).expect("seed data file");
    std::fs::write(&roster, [1u8, 2, 3]).expect("short roster");
    let error = FileBackedTier::open(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 1, true)
        .expect_err("short roster must be rejected");
    assert!(matches!(error, sparkflash_kv::KvError::Corrupt(_)));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}

#[test]
fn the_telemetry_delta_between_two_snapshots_is_exact() {
    let (mut store, telemetry) = in_memory_store(4, 4, 4);
    let protected = HashSet::new();
    let before = telemetry.snapshot();
    store
        .insert(PageId(0), &pattern(0), &protected)
        .expect("insert");
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    store.read(PageId(0), &mut out).expect("read");
    let delta = telemetry.delta_since(&before);
    assert_eq!(delta.page_inserts, 1);
    assert_eq!(delta.uma_reads, 1);
    assert_eq!(delta.bytes_written, GLOBAL_PAGE_BYTES as u64);
    assert_eq!(delta.bytes_read, GLOBAL_PAGE_BYTES as u64);
    assert!(!delta.is_zero());
    store.check_invariants().expect("store invariants");
    assert!(telemetry.delta_since(&telemetry.snapshot()).is_zero());
}

#[test]
fn draining_and_dropping_frees_the_pool() {
    let (mut store, telemetry) = in_memory_store(2, 2, 2);
    let protected = HashSet::new();
    for page in 0..3u64 {
        store
            .insert(PageId(page), &pattern(page), &protected)
            .expect("insert");
    }
    for page in 0..3u64 {
        store.drop_page(PageId(page)).expect("drop");
    }
    assert_eq!(store.tracked_pages(), 0);
    assert_eq!(store.resident(Tier::Uma), 0);
    assert_eq!(store.resident(Tier::Dram), 0);
    assert_eq!(store.resident(Tier::Nvme), 0);
    assert_eq!(telemetry.snapshot().page_drops, 3);
}

/// A store whose NVMe level is the file tier at a path, opened for a clean start exactly the
/// way KvCache::new opens it.
fn store_with_file_tier(path: &Path, capacity: u64) -> TieredPageStore {
    TieredPageStore::new(
        Box::new(InMemoryTier::new(Tier::Uma, GLOBAL_PAGE_BYTES, 1)),
        Box::new(InMemoryTier::new(Tier::Dram, GLOBAL_PAGE_BYTES, 1)),
        Box::new(
            FileBackedTier::open(Tier::Nvme, path, GLOBAL_PAGE_BYTES, capacity, false)
                .expect("file tier"),
        ),
        Arc::new(KvStoreTelemetry::default()),
    )
    .expect("store")
}

#[test]
fn replay_scopes_nest_and_release_one_level_at_a_time() {
    // The scope is a per-thread count, not a set: without the count, begin/begin/end would
    // leave the thread outside the scope while a replay is still running, and the publish path
    // would stop refusing in the middle of one.
    let (mut store, telemetry) = in_memory_store(4, 4, 4);
    let outer = store.begin_replay();
    assert_eq!(store.replay_depth(), 1);
    let inner = store.begin_replay();
    assert_eq!(store.replay_depth(), 2);
    assert_eq!(telemetry.snapshot().replay_scopes, 2);
    drop(inner);
    assert_eq!(store.replay_depth(), 1);
    assert!(
        store.in_replay(),
        "dropping one guard must not release the outer scope"
    );
    let error = store
        .insert(PageId(0), &pattern(0), &HashSet::new())
        .expect_err("the outer scope is still open");
    assert!(
        matches!(error, KvError::ReplayWroteGlobalKv(PageId(0))),
        "{error}"
    );
    drop(outer);
    assert_eq!(store.replay_depth(), 0);
    assert!(!store.in_replay());
    store
        .insert(PageId(0), &pattern(0), &HashSet::new())
        .expect("the last guard releases the scope");
    assert_eq!(telemetry.snapshot().replay_global_kv_writes, 1);
}

#[test]
fn a_panicking_scope_is_released_while_the_panic_unwinds() {
    // The scope is released by dropping the guard, and a panic drops it: without that, one
    // panicking replay would leave the thread marked as replaying for the life of the store
    // and every later publish on it would fail.
    let (mut store, telemetry) = in_memory_store(4, 4, 4);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _scope = store.begin_replay();
        panic!("the producer blew up");
    }));
    assert!(panicked.is_err(), "the panic must propagate");
    assert_eq!(store.replay_depth(), 0);
    assert!(!store.in_replay(), "the scope outlived the panic");
    store
        .insert(PageId(0), &pattern(0), &HashSet::new())
        .expect("a publish after a panicking replay must not be refused");
    assert_eq!(telemetry.snapshot().replay_scopes, 1);
    assert_eq!(telemetry.snapshot().replay_global_kv_writes, 0);
    store.check_invariants().expect("store invariants");
}

#[test]
fn a_backend_write_inside_a_replay_is_refused_like_the_store_publish() {
    // backend() hands out the tier itself, so before the guard was installed on the backends
    // that was an unguarded global-KV write path sitting behind a public accessor.
    let (store, telemetry) = in_memory_store(4, 4, 4);
    let scope = store.begin_replay();
    let error = store
        .backend(Tier::Uma)
        .write_page(PageId(0), &pattern(0))
        .expect_err("a replay must not write a global KV page through a backend");
    assert!(
        matches!(error, KvError::ReplayWroteGlobalKv(PageId(0))),
        "{error}"
    );
    assert_eq!(telemetry.snapshot().replay_global_kv_writes, 1);
    assert!(!store.contains(PageId(0)));
    drop(scope);
    store
        .backend(Tier::Uma)
        .write_page(PageId(0), &pattern(0))
        .expect("outside a replay the backend writes");
    assert!(store.backend(Tier::Uma).contains(PageId(0)));
    // The backend is still a raw escape hatch: the store's own map never saw that write, which
    // is why the published path stays insert(). The invariant check is what makes the
    // difference loud rather than silent.
    assert!(store.check_invariants().is_err());
    store
        .backend(Tier::Uma)
        .remove_page(PageId(0))
        .expect("cleanup");
    store.check_invariants().expect("store invariants");
}

#[test]
fn a_store_reopen_starts_clean_and_cannot_alias_a_recovered_page() {
    // Two runs against the same directory: the second one must not build a store whose
    // location map is empty while the file tier still holds pages, and a fresh page id must
    // not read back bytes a previous run wrote into the same slot.
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let path = directory.join(format!("kv-reopen-{}.bin", std::process::id()));
    let roster = PathBuf::from(format!("{}.roster", path.to_string_lossy()));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
    let capacity = 4u64;

    {
        let mut first = store_with_file_tier(&path, capacity);
        for page in 0..3u64 {
            first
                .insert(PageId(page), &pattern(page), &HashSet::new())
                .expect("insert");
        }
        assert_eq!(
            first.resident(Tier::Nvme),
            1,
            "run 1 must reach the file tier"
        );
        first.check_invariants().expect("run 1 invariants");
        first.flush().expect("flush");
    }
    assert!(roster.exists(), "run 1 must leave a roster behind");

    let mut second = store_with_file_tier(&path, capacity);
    assert_eq!(second.tracked_pages(), 0);
    assert_eq!(
        second.resident(Tier::Nvme),
        0,
        "a clean start must not adopt the recovered roster"
    );
    second
        .check_invariants()
        .expect("a reopened store must be coherent, not half recovered");
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    assert!(
        second.read(PageId(0), &mut out).is_err(),
        "a page from a previous run must not be readable"
    );
    second
        .insert(PageId(0), &pattern(9), &HashSet::new())
        .expect("a fresh page id is free");
    assert_eq!(second.read(PageId(0), &mut out).expect("read"), Tier::Uma);
    assert_eq!(
        out,
        pattern(9),
        "the fresh page must not alias the recovered page's bytes"
    );
    second.check_invariants().expect("run 2 invariants");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}

#[test]
fn a_store_refuses_backends_that_already_hold_pages() {
    // Recovery is a raw-tier facility. The store's location map is not persisted, so a
    // recovered roster could never be reached by a lookup and would make the store's own
    // invariant fail on the first check; saying so at construction is the loud option.
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let path = directory.join(format!("kv-refuse-{}.bin", std::process::id()));
    let roster = PathBuf::from(format!("{}.roster", path.to_string_lossy()));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
    {
        let tier = FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, false)
            .expect("open");
        tier.write_page(PageId(0), &pattern(0)).expect("write");
        tier.persist().expect("persist");
    }
    let recovered = FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, true)
        .expect("reopen");
    assert_eq!(recovered.resident_pages(), 1);
    let error = TieredPageStore::new(
        Box::new(InMemoryTier::new(Tier::Uma, GLOBAL_PAGE_BYTES, 2)),
        Box::new(InMemoryTier::new(Tier::Dram, GLOBAL_PAGE_BYTES, 2)),
        Box::new(recovered),
        Arc::new(KvStoreTelemetry::default()),
    )
    .expect_err("a recovered roster must not become an incoherent store");
    assert!(matches!(error, KvError::Corrupt(_)), "{error}");
    assert!(
        format!("{error}").contains("already holds 1 pages"),
        "{error}"
    );
    // The same path opened without recovery starts clean and is accepted.
    let clean = FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, false)
        .expect("clean open");
    assert_eq!(clean.resident_pages(), 0);
    TieredPageStore::new(
        Box::new(InMemoryTier::new(Tier::Uma, GLOBAL_PAGE_BYTES, 2)),
        Box::new(InMemoryTier::new(Tier::Dram, GLOBAL_PAGE_BYTES, 2)),
        Box::new(clean),
        Arc::new(KvStoreTelemetry::default()),
    )
    .expect("a clean tier is accepted");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}

#[test]
fn a_file_tier_is_bounded_by_capacity_not_by_page_id() {
    // Page ids are allocated globally and grow monotonically, so the file has to be sized by
    // its capacity: writing page 10 into a four-page tier must not produce a ten-slot file,
    // and a fifth distinct page must be refused rather than silently extending the file.
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let path = directory.join(format!("kv-bounded-{}.bin", std::process::id()));
    let roster = PathBuf::from(format!("{}.roster", path.to_string_lossy()));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
    let capacity = 4u64;
    let tier = FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, capacity, false)
        .expect("open");
    let bounded = capacity * tier.slot_bytes();
    tier.write_page(PageId(10), &pattern(10))
        .expect("a sparse page id takes a slot, not an offset");
    assert!(tier.contains(PageId(10)));
    assert_eq!(tier.resident_pages(), 1);
    assert_eq!(
        std::fs::metadata(&path).expect("metadata").len(),
        bounded,
        "page id 10 must not size the file"
    );
    for page in [11u64, 12, 13] {
        tier.write_page(PageId(page), &pattern(page)).expect("fill");
    }
    let error = tier
        .write_page(PageId(14), &pattern(14))
        .expect_err("the capacity must bind before the file grows");
    assert!(matches!(error, KvError::CapacityExceeded { .. }), "{error}");
    assert_eq!(tier.resident_pages(), 4);
    assert_eq!(std::fs::metadata(&path).expect("metadata").len(), bounded);
    // A removal frees its slot, so ids can keep growing without the file growing with them.
    tier.remove_page(PageId(11)).expect("remove");
    tier.write_page(PageId(9_999), &pattern(9_999))
        .expect("a freed slot is reusable");
    assert_eq!(tier.resident_pages(), 4);
    assert_eq!(std::fs::metadata(&path).expect("metadata").len(), bounded);
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    tier.read_page(PageId(9_999), &mut out).expect("read");
    assert_eq!(out, pattern(9_999));
    assert!(tier.read_page(PageId(11), &mut out).is_err());
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}

/// A cache whose NVMe level lives at a path, built the way an embedder builds one.
fn cache_at(path: &Path) -> KvCache {
    KvCache::new(KvCacheConfig {
        budget: MemoryBudget::new(1 << 40, 1 << 20).expect("budget"),
        uma_pages: 2,
        dram_pages: 2,
        nvme_pages: 4,
        swa_tail_slots: 4,
        nvme_path: Some(path.to_path_buf()),
    })
    .expect("cache")
}

#[test]
fn a_cache_reopen_against_the_same_nvme_path_starts_clean() {
    // The cache-level version of the reopen probe. KvCache::new used to open its file tier with
    // recovery on, so a second run adopted a roster whose pages the (unpersisted) radix index
    // could never reach: the store tracked none of them, check_invariants failed, and a fresh
    // page id overwrote a recovered page. A cache now starts clean, and the file is only ever
    // reused as fresh scratch space.
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let path = directory.join(format!("kv-cache-reopen-{}.bin", std::process::id()));
    let roster = PathBuf::from(format!("{}.roster", path.to_string_lossy()));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
    let tokens: Vec<u32> = (0..320u32).map(|index| index * 7 + 1).collect();
    let fill = |offset: u64| {
        move |block: usize, out: &mut [u8]| -> Result<(), KvError> {
            out.copy_from_slice(&pattern(block as u64 + offset));
            Ok(())
        }
    };

    let first = cache_at(&path);
    let outcome = first.cache_prefix(&tokens, fill(0)).expect("run 1");
    assert_eq!(outcome.pages_added, 5);
    assert_eq!(first.page_count(), 5);
    // The file holds pages, not just a set_len: open() sizes the file up front, so a length
    // check would be trivially true. A page in the NVMe tier is the real precondition, or the
    // clean-start probe proves nothing.
    assert!(
        first.resident_pages(Tier::Nvme) > 0,
        "run 1 must reach the file tier"
    );
    first.check_invariants().expect("run 1 invariants");
    first.flush().expect("flush");
    assert!(roster.exists(), "run 1 must leave a roster behind");
    let first_fingerprint = first.resident_page_fingerprint().expect("fingerprint");
    drop(first);

    let second = cache_at(&path);
    assert_eq!(
        second.page_count(),
        0,
        "a clean start must not adopt the recovered roster"
    );
    second
        .check_invariants()
        .expect("a reopened cache must be coherent, not half recovered");
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    assert!(
        second.read_page(PageId(0), &mut out).is_err(),
        "a page from a previous run must not be readable"
    );
    let outcome = second.cache_prefix(&tokens, fill(1_000)).expect("run 2");
    assert_eq!(outcome.pages_added, 5);
    second.read_page(PageId(0), &mut out).expect("read");
    assert_eq!(out, pattern(1_000), "run 2 must read back its own payload");
    assert_ne!(
        out,
        pattern(0),
        "the previous run's bytes must not be visible under a fresh page id"
    );
    assert_ne!(
        second.resident_page_fingerprint().expect("fingerprint"),
        first_fingerprint,
        "the two runs must not share pages"
    );
    second.check_invariants().expect("run 2 invariants");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}

#[test]
fn a_promote_inside_a_replay_scope_is_refused_without_losing_the_page() {
    // The reviewer's probe: promote used to lift the payload, release the DRAM copy and only
    // then meet the guard, so one refused promote destroyed the page id, its bytes and its
    // location entry while check_invariants still passed - and it counted two refusals for one
    // refused call. The refusal now happens before the first mutation.
    let (mut store, telemetry) = in_memory_store(2, 2, 2);
    let protected = HashSet::new();
    store
        .insert(PageId(0), &pattern(0), &protected)
        .expect("insert");
    store
        .insert(PageId(1), &pattern(1), &protected)
        .expect("insert");
    store
        .insert(PageId(2), &pattern(2), &protected)
        .expect("demote page 0 to DRAM");
    assert_eq!(
        store.location(PageId(0)).expect("tracked").tier,
        Tier::Dram,
        "the test needs a DRAM-resident page"
    );
    // Free the UMA slot the promotion moves into, so no cascade is involved.
    store.drop_page(PageId(1)).expect("free a UMA slot");
    assert_eq!(store.resident(Tier::Uma), 1);
    assert_eq!(store.resident(Tier::Dram), 1);

    let refuse_before = telemetry.snapshot().replay_global_kv_writes;
    let scope = store.begin_replay();
    let error = store
        .promote(PageId(0), &protected)
        .expect_err("a replay must not move a page");
    assert!(
        matches!(error, KvError::ReplayWroteGlobalKv(PageId(0))),
        "{error}"
    );
    assert_eq!(
        telemetry.snapshot().replay_global_kv_writes - refuse_before,
        1,
        "one refused promote is one refusal, not one per backend write it would have reached"
    );
    drop(scope);

    // Nothing moved, nothing was lost, and the invariants that missed the old bug still hold.
    assert!(store.contains(PageId(0)), "the page id must survive");
    assert_eq!(store.location(PageId(0)).expect("tracked").tier, Tier::Dram);
    assert_eq!(store.resident(Tier::Dram), 1);
    assert_eq!(store.resident(Tier::Uma), 1);
    assert_eq!(store.tracked_pages(), 2);
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    assert_eq!(
        store.read(PageId(0), &mut out).expect("read"),
        Tier::Dram,
        "the page must still live where it did"
    );
    assert_eq!(out, pattern(0), "the bytes must survive a refused promote");
    store.check_invariants().expect("store invariants");
    assert_eq!(telemetry.snapshot().uma_promotions, 0);

    // And the same promotion works once the scope is gone.
    assert_eq!(
        store.promote(PageId(0), &protected).expect("promote"),
        Tier::Dram
    );
    assert_eq!(store.location(PageId(0)).expect("tracked").tier, Tier::Uma);
    assert_eq!(store.read(PageId(0), &mut out).expect("read"), Tier::Uma);
    assert_eq!(out, pattern(0));
    assert_eq!(telemetry.snapshot().uma_promotions, 1);
    store.check_invariants().expect("store invariants");
}

#[test]
fn a_file_backed_write_inside_a_replay_is_refused_and_claims_no_slot() {
    // The backend guard has to hold on the file tier too, and the refusal must happen before
    // the slot claim: a refused write that consumed capacity would shrink the pool silently.
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let path = directory.join(format!("kv-fileguard-{}.bin", std::process::id()));
    let roster = PathBuf::from(format!("{}.roster", path.to_string_lossy()));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
    let telemetry = Arc::new(KvStoreTelemetry::default());
    let store = TieredPageStore::new(
        Box::new(InMemoryTier::new(Tier::Uma, GLOBAL_PAGE_BYTES, 2)),
        Box::new(InMemoryTier::new(Tier::Dram, GLOBAL_PAGE_BYTES, 2)),
        Box::new(
            FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, false)
                .expect("file tier"),
        ),
        Arc::clone(&telemetry),
    )
    .expect("store");
    let scope = store.begin_replay();
    let error = store
        .backend(Tier::Nvme)
        .write_page(PageId(0), &pattern(0))
        .expect_err("a replay must not write the file tier");
    assert!(
        matches!(error, KvError::ReplayWroteGlobalKv(PageId(0))),
        "{error}"
    );
    assert_eq!(
        store.backend(Tier::Nvme).resident_pages(),
        0,
        "a refused write must not claim a slot"
    );
    assert!(!store.backend(Tier::Nvme).contains(PageId(0)));
    assert!(
        !roster.exists(),
        "a refused write must not persist anything"
    );
    assert_eq!(telemetry.snapshot().replay_global_kv_writes, 1);
    drop(scope);
    // Both slots are still free, which is what proves the claim never happened.
    for page in [7u64, 3] {
        store
            .backend(Tier::Nvme)
            .write_page(PageId(page), &pattern(page))
            .expect("write");
    }
    assert_eq!(store.backend(Tier::Nvme).resident_pages(), 2);
    let mut out = vec![0u8; GLOBAL_PAGE_BYTES];
    store
        .backend(Tier::Nvme)
        .read_page(PageId(7), &mut out)
        .expect("read");
    assert_eq!(out, pattern(7));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}

#[test]
fn a_roster_with_a_hostile_count_is_rejected_instead_of_panicking() {
    // The count comes off the disk, and the header arithmetic used to overflow: a roster
    // claiming u64::MAX entries panicked with "attempt to multiply with overflow" instead of
    // returning KvError::Corrupt, which this crate documents it never does on bad input.
    let directory = test_dir();
    std::fs::create_dir_all(&directory).expect("test directory");
    let path = directory.join(format!("kv-hostile-{}.bin", std::process::id()));
    let roster = PathBuf::from(format!("{}.roster", path.to_string_lossy()));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
    {
        // A real roster first, so the magic is the crate's own rather than a copy of it here.
        let tier = FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, false)
            .expect("open");
        tier.write_page(PageId(0), &pattern(0)).expect("write");
        tier.persist().expect("persist");
    }
    let valid = std::fs::read(&roster).expect("roster");
    // A count whose product overflows, and one whose product does not but whose sum with the
    // header does: both must come back as Corrupt rather than panicking.
    for count in [u64::MAX, u64::MAX / 16] {
        let mut bytes = valid.clone();
        bytes[8..16].copy_from_slice(&count.to_le_bytes());
        bytes.truncate(16);
        std::fs::write(&roster, &bytes).expect("hostile roster");
        let error = FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, true)
            .expect_err("a roster with a hostile count must be rejected");
        assert!(
            matches!(error, KvError::Corrupt(_)),
            "count {count}: {error}"
        );
    }
    // The honest roster still opens, so the rejection was of the count and nothing else.
    std::fs::write(&roster, &valid).expect("restore the roster");
    let recovered =
        FileBackedTier::open_buffered(Tier::Nvme, &path, GLOBAL_PAGE_BYTES, 2, true).expect("open");
    assert_eq!(recovered.resident_pages(), 1);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&roster);
}
