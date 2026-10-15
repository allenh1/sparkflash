//! Integration tests for the O_DIRECT windowed reader against real packed files.
//!
//! These tests write synthetic rank files with the same dense layout, alignment padding and
//! tail rows the packer produces, so they exercise the real `pread` path, the real mapping
//! fallback and the real EOF clamping without any model weights.

#[path = "fixtures/store/synthetic.rs"]
mod synthetic;

use std::path::Path;

use sparkflash_engram::layout::ENGRAM_ROW_BYTES;
use sparkflash_engram::packer::{PackManifest, align_up};
use sparkflash_engram::store::{
    BackendPreference, DirectTable, ReadBackend, RowTable, TableSpec, TailFill, specs_from_manifest,
};

/// A temporary directory holding one dense rank file plus the rows it must produce.
struct Fixture {
    _dir: tempfile::TempDir,
    spec: TableSpec,
    expected: Vec<[u8; ENGRAM_ROW_BYTES]>,
    file_bytes: u64,
}

fn fixture(rows: u64, window_bytes: u64) -> Fixture {
    let dir = tempfile::tempdir().expect("temporary directory");
    let path = dir.path().join("engram-l1-r0of4.bin");
    let file_bytes = synthetic::write_dense_table(&path, 1, 0, rows);
    Fixture {
        _dir: dir,
        spec: synthetic::spec(1, 0, 0, 4, &path, 0, rows, rows, window_bytes),
        expected: (0..rows).map(|row| synthetic::row_for(1, row)).collect(),
        file_bytes,
    }
}

fn open(fixture: &Fixture) -> DirectTable {
    DirectTable::open(vec![fixture.spec.clone()]).expect("open the fixture table")
}

#[test]
fn every_backend_reads_every_row_of_a_real_file() {
    for window in [512u64, 4096] {
        let fixture = fixture(37, window);
        for preference in [BackendPreference::Auto, BackendPreference::Mapped] {
            let table = DirectTable::open_with(vec![fixture.spec.clone()], preference, &mut |_| {})
                .expect("open the fixture table");
            assert_eq!(table.table_count(), 1);
            assert!(!table.is_empty());
            let backend = table.backend_of(0).expect("backend");
            assert_eq!(table.file_bytes(0), Some(fixture.file_bytes));
            if preference == BackendPreference::Mapped {
                assert_eq!(backend, ReadBackend::Mapped);
            }
            for (row, expected) in fixture.expected.iter().enumerate() {
                let got = table.read_row(0, row as u64).expect("read row");
                assert_eq!(
                    &got, expected,
                    "window {window} backend {backend:?} row {row}"
                );
            }
            assert!(table.read_ops() >= 1);
        }
    }
}

#[test]
fn the_final_window_is_clamped_to_the_file_end() {
    // 17 rows * 264 bytes = 4488 bytes, padded to 4608: the last row's 4096-byte window
    // runs past the end of the file and must be clamped, not read.
    let fixture = fixture(17, 4096);
    assert_eq!(fixture.file_bytes, 4608);
    let table = open(&fixture);
    for (row, expected) in fixture.expected.iter().enumerate() {
        let got = table.read_row(0, row as u64).expect("read row");
        assert_eq!(&got, expected, "row {row}");
    }
    // Batch the whole file in one request: same clamping path.
    let all = table
        .read_rows(0, &(0..17u32).collect::<Vec<_>>())
        .expect("read every row");
    for (row, got) in all.iter().enumerate() {
        assert_eq!(got, &fixture.expected[row], "batched row {row}");
    }
}

#[test]
fn batched_reads_merge_nearby_rows_and_keep_request_order() {
    let fixture = fixture(64, 512);
    let table = open(&fixture).with_merge_gap_bytes(0);
    let before = table.read_ops();
    let out = table.read_rows(0, &[3, 0, 33, 1]).expect("read rows");
    assert_eq!(out.len(), 4);
    // The output follows the request order, not the sorted order.
    assert_eq!(out[0], fixture.expected[3]);
    assert_eq!(out[1], fixture.expected[0]);
    assert_eq!(out[2], fixture.expected[33]);
    assert_eq!(out[3], fixture.expected[1]);
    // Rows 0, 1 and 3 are contiguous, so they merge into one window; row 33 is far away.
    assert_eq!(table.read_ops() - before, 2, "expected two real reads");

    // Consecutive rows never split.
    let before = table.read_ops();
    let consecutive = table.read_rows(0, &[10, 11, 12, 13]).expect("read rows");
    assert_eq!(
        table.read_ops() - before,
        1,
        "four consecutive rows are one read"
    );
    for (slot, row) in [10usize, 11, 12, 13].into_iter().enumerate() {
        assert_eq!(consecutive[slot], fixture.expected[row]);
    }

    // A larger merge gap merges a pair that the zero gap split.
    let generous = open(&fixture).with_merge_gap_bytes(1024);
    let before = generous.read_ops();
    let pair = generous.read_rows(0, &[0, 3]).expect("read rows");
    assert_eq!(pair[0], fixture.expected[0]);
    assert_eq!(pair[1], fixture.expected[3]);
    assert_eq!(
        generous.read_ops() - before,
        1,
        "a 528-byte gap merges at 1 KiB"
    );
}

#[test]
fn rows_outside_the_table_are_rejected() {
    let fixture = fixture(8, 512);
    let table = open(&fixture);
    let error = table.read_row(0, 8).expect_err("row 8 does not exist");
    assert!(error.to_string().contains('8'), "{error}");
    assert!(table.read_row(0, u64::MAX).is_err());
    let mut out = vec![[0u8; ENGRAM_ROW_BYTES]; 1];
    assert!(table.read_rows_into(0, &[9], &mut out).is_err());
    assert!(table.read_row(1, 0).is_err(), "there is only one table");
    assert!(table.spec(1).is_none());
}

#[test]
fn tail_rows_use_the_configured_fill() {
    let dir = tempfile::tempdir().expect("temporary directory");
    let path = dir.path().join("engram-l14-r3of4.bin");
    let rows = 12u64;
    let source_rows = 10u64;
    synthetic::write_dense_table(&path, 14, 0, rows);
    let spec = synthetic::spec(14, 1, 3, 4, &path, 0, rows, source_rows, 512);
    assert!(spec.is_tail(source_rows));
    assert!(!spec.is_tail(source_rows - 1));
    assert_eq!(spec.source_rows(), source_rows);
    assert_eq!(spec.local_row(0), Some(0));
    assert_eq!(spec.global_row(5), 5);

    let reference = DirectTable::open(vec![spec.clone()]).expect("open");
    assert_eq!(
        reference.read_row(0, 11).expect("tail row"),
        TailFill::Reference.row()
    );

    let zeros = DirectTable::open(vec![spec.with_tail_fill(TailFill::Zeros)]).expect("open");
    assert_eq!(
        zeros.read_row(0, 11).expect("tail row"),
        [0u8; ENGRAM_ROW_BYTES]
    );
    assert_eq!(
        zeros.read_row(0, 9).expect("real row"),
        synthetic::row_for(14, 9)
    );
}

#[test]
fn a_short_or_unaligned_file_is_refused() {
    let dir = tempfile::tempdir().expect("temporary directory");

    let short = dir.path().join("short.bin");
    std::fs::write(&short, [0u8; 512]).expect("write");
    let error = DirectTable::open(vec![synthetic::spec(1, 0, 0, 4, &short, 0, 8, 8, 512)])
        .expect_err("a 512-byte file cannot hold eight 264-byte rows");
    assert!(error.to_string().contains("needs"), "{error}");

    let unaligned = dir.path().join("unaligned.bin");
    let written = synthetic::write_unaligned_table(&unaligned, 1, 8);
    assert_eq!(written, 8 * ENGRAM_ROW_BYTES as u64);
    assert_ne!(written % 512, 0);
    let error = DirectTable::open(vec![synthetic::spec(1, 0, 0, 4, &unaligned, 0, 8, 8, 512)])
        .expect_err("the packer always pads to a 512-byte multiple");
    assert!(error.to_string().contains("512"), "{error}");
}

#[test]
fn manifest_specs_match_the_compiled_geometry() {
    let dir = tempfile::tempdir().expect("temporary directory");

    // Layer 1, rank 3 of 4: 96,001,542 rows starting at global row 288,004,626.
    let rows = 96_001_542u64;
    let table_rows = 384_006_168u64;
    let data_bytes = rows * ENGRAM_ROW_BYTES as u64;
    let file_bytes = align_up(data_bytes);
    let json = synthetic::manifest_json(
        1,
        0,
        3,
        4,
        "engram-l1-r3of4.bin",
        288_004_626,
        rows,
        rows,
        table_rows,
        data_bytes,
        file_bytes,
        "model-00047-of-00048.safetensors",
    );
    let manifest: PackManifest = serde_json::from_str(&json).expect("manifest parses");
    assert_eq!(manifest.format, "sparkflash-engram");
    assert_eq!(manifest.layout.row_bytes, ENGRAM_ROW_BYTES as u64);
    assert!(manifest.layout.dense);
    let specs = specs_from_manifest(&manifest, dir.path(), 3, 4).expect("specs");
    assert_eq!(specs.len(), 1);
    let spec = &specs[0];
    assert_eq!(spec.layer_id, 1);
    assert_eq!(spec.layer_index, 0);
    assert_eq!(spec.rank, 3);
    assert_eq!(spec.global_start(), 288_004_626);
    assert_eq!(spec.global_row(0), 288_004_626);
    assert_eq!(spec.local_row(288_004_626), Some(0));
    assert_eq!(
        spec.local_row(288_004_625),
        None,
        "another rank owns that row"
    );
    assert_eq!(spec.physical_rows, rows);
    assert_eq!(spec.source_rows(), rows);
    assert!(!spec.is_tail(rows - 1));
    assert_eq!(spec.logical_rows, table_rows);
    assert_eq!(spec.file_bytes(), file_bytes);
    assert!(spec.window_bytes.is_multiple_of(512));
    assert!(spec.path.ends_with("engram-l1-r3of4.bin"));

    // The padded last rank of layer 14 keeps its two synthetic tail rows.
    let rows = 96_004_171u64;
    let source_rows = 96_004_169u64;
    let table_rows = 384_016_682u64;
    let data_bytes = rows * ENGRAM_ROW_BYTES as u64;
    let file_bytes = align_up(data_bytes);
    let json = synthetic::manifest_json(
        14,
        1,
        3,
        4,
        "engram-l14-r3of4.bin",
        288_012_513,
        rows,
        source_rows,
        table_rows,
        data_bytes,
        file_bytes,
        "model-00048-of-00048.safetensors",
    );
    let manifest: PackManifest = serde_json::from_str(&json).expect("manifest parses");
    let specs = specs_from_manifest(&manifest, dir.path(), 3, 4).expect("specs");
    let spec = &specs[0];
    assert_eq!(spec.layer_id, 14);
    assert_eq!(spec.rank, 3);
    assert_eq!(spec.global_start(), 288_012_513);
    assert_eq!(spec.physical_rows, rows);
    assert_eq!(spec.source_rows(), source_rows);
    assert!(spec.is_tail(source_rows));
    assert!(spec.is_tail(rows - 1));
    assert_eq!(spec.file_bytes(), file_bytes);

    // A manifest whose row count disagrees with the geometry is refused.
    let json = synthetic::manifest_json(
        14,
        1,
        3,
        4,
        "engram-l14-r3of4.bin",
        288_012_513,
        rows + 1,
        source_rows,
        table_rows,
        data_bytes,
        file_bytes,
        "model-00048-of-00048.safetensors",
    );
    let manifest: PackManifest = serde_json::from_str(&json).expect("manifest parses");
    assert!(specs_from_manifest(&manifest, dir.path(), 3, 4).is_err());

    // A manifest packed for another world size is refused.
    let json = synthetic::manifest_json(
        1,
        0,
        3,
        4,
        "engram-l1-r3of4.bin",
        288_004_626,
        96_001_542,
        96_001_542,
        384_006_168,
        data_bytes,
        file_bytes,
        "model-00047-of-00048.safetensors",
    );
    let manifest: PackManifest = serde_json::from_str(&json).expect("manifest parses");
    assert!(specs_from_manifest(&manifest, dir.path(), 3, 8).is_err());
}

#[test]
fn direct_io_engages_on_a_supporting_filesystem() {
    // Portable by design: the direct path is only asserted where a filesystem really
    // supports O_DIRECT. Set SPARK_ENGRAM_DIRECT_DIR to such a directory to run it.
    let Some(dir) = std::env::var_os("SPARK_ENGRAM_DIRECT_DIR") else {
        eprintln!("skipping: SPARK_ENGRAM_DIRECT_DIR is not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("create the direct-IO directory");
    let path = dir.join("engram-l1-r0of4.bin");
    synthetic::write_dense_table(&path, 1, 0, 41);
    let spec = synthetic::spec(1, 0, 0, 4, &path, 0, 41, 41, 512);
    match DirectTable::open_with(vec![spec.clone()], BackendPreference::Require, &mut |_| {}) {
        Ok(table) => {
            assert_eq!(table.backend_of(0), Some(ReadBackend::Direct));
            assert!(table.uses_o_direct());
            assert_eq!(table.fallback_reason(), None);
            for row in 0..41u64 {
                assert_eq!(
                    table.read_row(0, row).expect("read"),
                    synthetic::row_for(1, row)
                );
            }
            println!("O_DIRECT engaged on {}", dir.display());
        }
        Err(error) => {
            // Unsupported here; the mapping fallback must still serve identical bytes.
            println!("O_DIRECT unavailable on {}: {error:#}", dir.display());
            let table =
                DirectTable::open_with(vec![spec], BackendPreference::Auto, &mut |line: &str| {
                    println!("fallback: {line}")
                })
                .expect("the mapping fallback must work");
            assert_eq!(table.backend_of(0), Some(ReadBackend::Mapped));
            assert!(table.fallback_reason().is_some());
            assert_eq!(
                table.read_row(0, 40).expect("read"),
                synthetic::row_for(1, 40)
            );
        }
    }
}

#[test]
fn a_forced_alignment_keeps_the_window_legal_on_both_device_shapes() {
    // The deployment target reports 512-byte logical blocks; 4Kn drives, dm-crypt volumes
    // and some NVMe namespaces report 4096. Force each shape and check the mechanics: a
    // 512-byte device keeps a 512-byte window, a 4096-byte device raises it, and the rows
    // are identical either way.
    let rows = 64u64;
    for align in [512u64, 4096] {
        let fixture = fixture(rows, 512);
        let table = DirectTable::open_with_alignment(
            vec![fixture.spec.clone()],
            BackendPreference::Auto,
            align,
            &mut |line: &str| println!("align {align}: {line}"),
        )
        .expect("open the fixture table");
        assert_eq!(table.alignment_of(0), Some(align));
        assert_eq!(
            table.window_of(0),
            Some(align),
            "a {align}-byte device must keep a {align}-byte window"
        );
        for (row, expected) in fixture.expected.iter().enumerate() {
            let got = table.read_row(0, row as u64).expect("read row");
            assert_eq!(&got, expected, "align {align} row {row}");
        }
    }

    // Whatever this filesystem reports, it must be something the reader can serve, and the
    // window must never be lower than the alignment.
    let fixture = fixture(16, 512);
    let detected = DirectTable::open(vec![fixture.spec.clone()]).expect("open");
    let align = detected.alignment_of(0).expect("alignment");
    assert!(
        (512..=4096).contains(&align) && align.is_multiple_of(512),
        "detected {align}"
    );
    assert!(detected.window_of(0).expect("window") >= align);
    // An alignment the reader cannot honour is refused before any file is touched.
    assert!(
        DirectTable::open_with_alignment(
            vec![fixture.spec.clone()],
            BackendPreference::Auto,
            256,
            &mut |_| {},
        )
        .is_err()
    );
    assert!(
        DirectTable::open_with_alignment(
            vec![fixture.spec.clone()],
            BackendPreference::Auto,
            8192,
            &mut |_| {},
        )
        .is_err()
    );
}

#[test]
fn a_device_with_a_larger_logical_block_size_falls_back_honestly() {
    // A 512-byte O_DIRECT read on a 4096-logical-block device fails with EINVAL. Forcing
    // 512 must then either work (the device really is a 512-byte one, as on the cluster) or
    // be reported: the table may not claim ReadBackend::Direct while its reads go through
    // the mapping, and it must still return the right rows.
    let root = std::env::var_os("SPARK_ENGRAM_DIRECT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/var/tmp"));
    let dir = root.join("sparkflash-engram-alignment");
    if std::fs::create_dir_all(&dir).is_err() {
        eprintln!("skipping: {} is not writable", dir.display());
        return;
    }
    let path = dir.join("engram-l1-r0of4.bin");
    synthetic::write_dense_table(&path, 1, 0, 41);
    let spec = synthetic::spec(1, 0, 0, 4, &path, 0, 41, 41, 512);
    let mut warnings = Vec::new();
    let table = DirectTable::open_with_alignment(
        vec![spec.clone()],
        BackendPreference::Auto,
        512,
        &mut |line: &str| warnings.push(line.to_owned()),
    )
    .expect("the mapping fallback must serve a 512-byte alignment");
    assert_eq!(table.window_of(0), Some(512));
    match table.backend_of(0).expect("backend") {
        ReadBackend::Direct => {
            // A 512-byte-logical-block device accepted the aligned read.
            assert!(warnings.is_empty(), "{warnings:?}");
            assert_eq!(table.fallback_reason(), None);
            println!("{} accepts 512-byte O_DIRECT reads", root.display());
        }
        ReadBackend::Mapped | ReadBackend::Buffered => {
            // A 4096-byte-logical-block device refused the probe: the reader says so rather
            // than reporting Direct and failing later.
            let reason = table.fallback_reason().expect("a fallback must say why");
            println!(
                "{} refuses 512-byte O_DIRECT reads: {reason}",
                root.display()
            );
        }
    }
    for row in 0..41u64 {
        assert_eq!(
            table.read_row(0, row).expect("read"),
            synthetic::row_for(1, row)
        );
    }
    // Asking for the alignment the device really needs always works.
    let forced =
        DirectTable::open_with_alignment(vec![spec], BackendPreference::Auto, 4096, &mut |_| {})
            .expect("open at the device alignment");
    assert_eq!(forced.alignment_of(0), Some(4096));
    assert_eq!(forced.window_of(0), Some(4096));
    for row in 0..41u64 {
        assert_eq!(
            forced.read_row(0, row).expect("read"),
            synthetic::row_for(1, row)
        );
    }
    let _ = std::fs::remove_file(&path);
}
#[test]
fn a_missing_file_is_reported_with_its_path() {
    let missing = Path::new("/nonexistent/engram-l1-r0of4.bin");
    let error = DirectTable::open(vec![synthetic::spec(1, 0, 0, 4, missing, 0, 8, 8, 512)])
        .expect_err("the file does not exist");
    assert!(error.to_string().contains("engram-l1-r0of4.bin"), "{error}");
}
