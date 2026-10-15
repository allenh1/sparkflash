//! Integration test: the packer's plan, the O_DIRECT reader, the UMA/DRAM cache and the
//! prefetch engine composed over real files.
//!
//! Every file here is a synthetic rank file written into a temporary directory; the packer
//! planning tests drive `packer::plan` from a synthetic safetensors index, so no model
//! weights, cluster or NVMe device is involved.

#[path = "fixtures/store/synthetic.rs"]
mod synthetic;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sparkflash_engram::cache::{CacheConfig, CacheTier, EngramRowCache, EvictionPolicy};
use sparkflash_engram::layout::ENGRAM_ROW_BYTES;
use sparkflash_engram::packer::{CliArgs, PackOptions, plan, run, run_cli};
use sparkflash_engram::prefetch::{BackendRequest, PrefetchConfig, PrefetchEngine};
use sparkflash_engram::store::{DirectTable, RowTable, TableSpec};
use sparkflash_engram::telemetry::EngramLayerTelemetry;

/// A temporary directory with one 41-row rank file.
struct Fixture {
    _dir: tempfile::TempDir,
    path: PathBuf,
    spec: TableSpec,
    expected: Vec<[u8; ENGRAM_ROW_BYTES]>,
}

const ROWS: u64 = 41;

fn fixture(window_bytes: u64) -> Fixture {
    let dir = tempfile::tempdir().expect("temporary directory");
    let path = dir.path().join("engram-l1-r0of4.bin");
    synthetic::write_dense_table(&path, 1, 0, ROWS);
    Fixture {
        path: path.clone(),
        spec: synthetic::spec(1, 0, 0, 4, &path, 0, ROWS, ROWS, window_bytes),
        expected: (0..ROWS).map(|row| synthetic::row_for(1, row)).collect(),
        _dir: dir,
    }
}

fn reader(fixture: &Fixture) -> DirectTable {
    DirectTable::open(vec![fixture.spec.clone()]).expect("open the reader")
}

fn cache() -> EngramRowCache {
    EngramRowCache::new(CacheConfig {
        uma_bytes: 4096 * 8,
        dram_bytes: 4096 * 16,
        ways: 4,
        policy: EvictionPolicy::Clock,
        layer_count: 2,
    })
    .expect("build the cache")
}

fn prefetch_config() -> PrefetchConfig {
    PrefetchConfig {
        queue_depth: 64,
        lead_rows: 8,
        max_batch_rows: 32,
        workers: 2,
        ready_capacity: 256,
        window_bytes: 512,
    }
}

/// Polls the engine until `row` arrives.
fn wait_for(engine: &PrefetchEngine, row: u64) -> [u8; ENGRAM_ROW_BYTES] {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(data) = engine.take(0, row) {
            return data;
        }
        assert!(
            Instant::now() < deadline,
            "row {row} never arrived: {:?}",
            engine.snapshot()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn the_cache_serves_rows_that_the_reader_fetches() {
    let fixture = fixture(512);
    let table = reader(&fixture);
    let cache = cache();
    let mut out = [0u8; ENGRAM_ROW_BYTES];

    // The first lookup of a row is a store miss filled by the reader.
    assert_eq!(
        cache.get_or_fill(&table, 0, 7, &mut out).expect("fill"),
        CacheTier::Nvme
    );
    assert_eq!(out, fixture.expected[7]);
    // The second lookup is served by the UMA tier.
    out = [0u8; ENGRAM_ROW_BYTES];
    assert_eq!(
        cache.get_or_fill(&table, 0, 7, &mut out).expect("hit"),
        CacheTier::Uma
    );
    assert_eq!(out, fixture.expected[7]);

    let snapshot = cache.snapshot();
    assert_eq!(snapshot.counters.lookups, 2);
    assert_eq!(snapshot.counters.misses, 1);
    assert_eq!(snapshot.counters.uma_hits, 1);
    assert_eq!(snapshot.counters.bytes_read_nvme, ENGRAM_ROW_BYTES as u64);
    assert!((snapshot.hit_rate - 0.5).abs() < 1e-9);

    // A batched lookup fills every missing row from one merged store read.
    let rows: Vec<u32> = (0..16).collect();
    let mut buffer = vec![[0u8; ENGRAM_ROW_BYTES]; rows.len()];
    let tiers = cache
        .get_or_fill_rows(&table, 0, &rows, &mut buffer)
        .expect("batched fill");
    assert_eq!(tiers.len(), rows.len());
    for (row, data) in buffer.iter().enumerate() {
        assert_eq!(data, &fixture.expected[row], "row {row}");
    }
    assert_eq!(tiers[7], CacheTier::Uma);

    // The cache's per-layer counters can be exported as a telemetry delta.
    let telemetry = EngramLayerTelemetry::default();
    assert!(cache.record_telemetry(0, &telemetry));
    let exported = telemetry.snapshot(1);
    assert_eq!(
        exported.lookups_total,
        snapshot.counters.lookups + rows.len() as u64
    );
    assert_eq!(exported.tier2_uma_hits, 2, "row 7 hit twice");
    assert_eq!(
        exported.tier3_nvme_misses,
        snapshot.counters.misses + 15,
        "fifteen rows missed the cache"
    );
    // Counters are consumed: a second export adds nothing.
    assert!(cache.record_telemetry(0, &telemetry));
    assert_eq!(telemetry.snapshot(1).lookups_total, exported.lookups_total);
    // An out-of-range layer index is refused, not silently dropped.
    assert!(!cache.record_telemetry(9, &telemetry));
}

#[test]
fn the_prefetcher_warms_the_cache_from_the_reader() {
    let fixture = fixture(512);
    let source: Arc<dyn RowTable> = Arc::new(reader(&fixture));
    let engine = PrefetchEngine::with_backend(
        Arc::clone(&source),
        prefetch_config(),
        BackendRequest::ThreadPool,
    )
    .expect("start the prefetcher");
    let cache = cache();

    let rows: Vec<u32> = (0..ROWS as u32).collect();
    assert_eq!(
        engine.prefetch_ahead(0, &rows, 0),
        8,
        "the lead depth is eight"
    );
    assert_eq!(engine.prefetch_ahead(0, &rows, 8), 8);

    // prefetch_ahead starts one row past the cursor, so rows 1..17 were requested.
    let mut filled = 0usize;
    for row in 1..17u64 {
        let data = wait_for(&engine, row);
        assert_eq!(data, fixture.expected[row as usize]);
        cache.insert(1, row, &data);
        filled += 1;
    }
    assert_eq!(filled, 16);

    // Every prefetched row is a cache hit, so the store is not touched again.
    let mut out = [0u8; ENGRAM_ROW_BYTES];
    for row in 1..17u64 {
        let tier = cache.get(1, row, &mut out);
        assert_ne!(tier, CacheTier::Nvme, "row {row} should be cached");
        assert_eq!(out, fixture.expected[row as usize]);
    }
    let counters = cache.snapshot().counters;
    assert_eq!(counters.lookups, 16);
    assert_eq!(counters.misses, 0);
    assert_eq!(counters.bytes_read_nvme, 0);

    // A row that was never prefetched falls back to the store.
    assert_eq!(
        engine.fill_or_read(0, 40).expect("read"),
        fixture.expected[40]
    );
    let snapshot = engine.snapshot();
    assert!(snapshot.completed >= 16);
    assert!(snapshot.pool_reads >= 1);
    assert_eq!(snapshot.backpressure, 0);
}

#[test]
fn every_layer_gets_its_own_row_from_the_reader() {
    // Two tables in one reader: local row indices are per file, so row 0 of table 1 must
    // not be served from table 0.
    let dir = tempfile::tempdir().expect("temporary directory");
    let first = dir.path().join("engram-l1-r0of4.bin");
    let second = dir.path().join("engram-l14-r0of4.bin");
    synthetic::write_dense_table(&first, 1, 0, 8);
    synthetic::write_dense_table(&second, 14, 0, 8);
    let table = DirectTable::open(vec![
        synthetic::spec(1, 0, 0, 4, &first, 0, 8, 8, 512),
        synthetic::spec(14, 1, 0, 4, &second, 0, 8, 8, 512),
    ])
    .expect("open both layers");
    assert_eq!(table.table_count(), 2);
    assert_eq!(
        table.read_row(0, 3).expect("read"),
        synthetic::row_for(1, 3)
    );
    assert_eq!(
        table.read_row(1, 3).expect("read"),
        synthetic::row_for(14, 3)
    );
    // read_global resolves the global row to the owning table and local row.
    assert_eq!(
        table.read_global(1, 3).expect("global read"),
        Some(synthetic::row_for(1, 3))
    );
    assert_eq!(
        table.read_global(14, 3).expect("global read"),
        Some(synthetic::row_for(14, 3))
    );
    assert_eq!(table.read_global(7, 3).expect("no such layer"), None);
}

#[test]
fn the_packer_plans_the_real_geometry_from_an_index() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let index = dir.path().join("model.safetensors.index.json");
    std::fs::write(
        &index,
        synthetic::index_json("model-00047-of-00048.safetensors", &[1]),
    )
    .expect("write the index");

    let mut options = PackOptions::new(&index, dir.path().join("out"));
    options.layers = vec![1];
    options.world_size = 4;
    options.ranks = vec![3];
    options.dry_run = true;
    options.tokenizer_path = Some(dir.path().join("missing-tokenizer.json"));

    let planned = plan(&options).expect("plan");
    assert_eq!(planned.layers.len(), 1);
    let layer = &planned.layers[0];
    assert_eq!(layer.layer, 1);
    assert_eq!(layer.layer_index, 0);
    assert_eq!(layer.shards.len(), 1);
    let shard = &layer.shards[0];
    assert_eq!(shard.rank, 3);
    assert_eq!(shard.file, "engram-l1-r3of4.bin");
    assert_eq!(shard.row_start, 288_004_626);
    assert_eq!(shard.rows, 96_001_542);
    assert_eq!(shard.padded_rows, 0);
    assert_eq!(shard.file_bytes, 25_344_407_552);
    assert_eq!(planned.total_rows, 384_006_168);
    assert_eq!(planned.total_file_bytes, 25_344_407_552);

    // A dry run reports the plan and writes nothing at all.
    let report = run(&options).expect("dry run");
    assert!(report.dry_run);
    assert!(report.files_written.is_empty());
    assert_eq!(report.bytes_written, 0);
    assert!(report.manifest_path.ends_with("manifest.json"));
    assert!(
        !options.output_dir.exists(),
        "a dry run creates no directory"
    );
}

#[test]
fn the_packer_refuses_a_broken_index() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let index = dir.path().join("model.safetensors.index.json");

    // The scale tensor is missing entirely.
    std::fs::write(
        &index,
        synthetic::index_json_from(&[("layers.1.engram.embed.weight", "shard-47.safetensors")]),
    )
    .expect("write the index");
    let mut options = PackOptions::new(&index, dir.path().join("out"));
    options.layers = vec![1];
    let error = plan(&options).expect_err("the scale tensor is missing");
    assert!(error.to_string().contains("scale"), "{error}");

    // Both tensors exist but live in different shards.
    std::fs::write(
        &index,
        synthetic::index_json_from(&[
            ("layers.1.engram.embed.weight", "shard-47.safetensors"),
            ("layers.1.engram.embed.scale", "shard-48.safetensors"),
        ]),
    )
    .expect("write the index");
    let error = plan(&options).expect_err("the tensors are split");
    assert!(error.to_string().contains("one shard"), "{error}");

    // A layer without an Engram table is rejected before any I/O.
    options.layers = vec![7];
    let error = plan(&options).expect_err("layer 7 has no table");
    assert!(error.to_string().contains("layer 7"), "{error}");
}

#[test]
fn the_cli_surface_plans_a_dry_run() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let index = dir.path().join("model.safetensors.index.json");
    std::fs::write(
        &index,
        synthetic::index_json("model-00047-of-00048.safetensors", &[1]),
    )
    .expect("write the index");
    let args = CliArgs {
        index: Some(index),
        checkpoint_dir: Some(dir.path().to_path_buf()),
        output_dir: dir.path().join("out"),
        layers: vec![1],
        world_size: 4,
        rank: vec![3],
        dry_run: true,
        verify: false,
        tokenizer: None,
        min_free_gib: 0,
        chunk_rows: 4096,
        force: false,
        no_fsync: false,
    };
    let report = run_cli(args.clone()).expect("dry run through the CLI");
    assert!(report.dry_run);
    assert!(report.files_written.is_empty());

    // Without an index and without a checkpoint directory the CLI explains what to pass.
    let mut without = args.clone();
    without.index = None;
    without.checkpoint_dir = None;
    let error = run_cli(without).expect_err("no index");
    assert!(error.to_string().contains("--index"), "{error}");
}

#[test]
fn the_whole_pipeline_produces_the_file_bytes() {
    // Reader -> prefetcher -> cache for the full synthetic table, row by row.
    let fixture = fixture(4096);
    let source: Arc<dyn RowTable> = Arc::new(reader(&fixture));
    let engine =
        PrefetchEngine::with_backend(source, prefetch_config(), BackendRequest::ThreadPool)
            .expect("start");
    let cache = cache();
    let rows: Vec<u32> = (0..ROWS as u32).collect();
    engine.prefetch_ahead(0, &rows, 0);
    let fallback = Arc::clone(engine.source());
    let mut out = [0u8; ENGRAM_ROW_BYTES];
    for row in 0..ROWS {
        let data = match engine.take(0, row) {
            Some(data) => data,
            None => {
                // Not prefetched yet: fall back to the store through the cache.
                cache
                    .get_or_fill(fallback.as_ref(), 0, row, &mut out)
                    .expect("fill");
                out
            }
        };
        assert_eq!(data, fixture.expected[row as usize], "row {row}");
    }
    cache.record_telemetry(0, &EngramLayerTelemetry::default());
    assert!(fixture.path.exists());
}
