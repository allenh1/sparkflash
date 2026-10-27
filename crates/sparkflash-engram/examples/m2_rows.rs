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

//! task-29 (a): packed-row retrieval on a synthetic Engram table.
//!
//! Read-only with respect to any checkpoint: it writes its own synthetic table under the path
//! given on the command line, then measures DirectTable + EngramRowCache on it.
//!
//! Usage: m2_rows make    <path>          # write ROWS 264-byte rows
//!        m2_rows cold    <path> <n>      # random distinct ids, cache at the real budget
//!        m2_rows warm    <path> <n>      # repeat the same ids
//!        m2_rows batch   <path> <n>      # get_or_fill_rows, random vs contiguous
//!        m2_rows pipelined <path> <n> <depth>

use sparkflash_engram::cache::EngramRowCache;
use sparkflash_engram::layout::{ENGRAM_LAYERS, ENGRAM_ROW_BYTES};
use sparkflash_engram::prefetch::{PrefetchConfig, PrefetchEngine};
use sparkflash_engram::store::{BackendPreference, DirectTable, RowTable, TableSpec};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Default synthetic table: 2^24 rows x 264 B = 4,429,185,024 B, a multiple of 512.
const DEFAULT_ROWS: u64 = 1 << 24;
/// The serving node's measured budget: about 1 GiB, split as the defaults are shaped.
const UMA_BYTES: usize = 256 * 1024 * 1024;
const DRAM_BYTES: usize = 768 * 1024 * 1024;

fn dev_io() -> (u64, u64) {
    let text = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    let mut read_bytes = 0u64;
    let mut rchar = 0u64;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("read_bytes: ") {
            read_bytes = v.trim().parse().unwrap_or(0);
        }
        if let Some(v) = line.strip_prefix("rchar: ") {
            rchar = v.trim().parse().unwrap_or(0);
        }
    }
    (read_bytes, rchar)
}

fn table_rows(path: &Path) -> u64 {
    std::fs::metadata(path).expect("stat").len() / ENGRAM_ROW_BYTES as u64
}

fn spec(path: PathBuf, rows: u64) -> TableSpec {
    TableSpec {
        layer_id: ENGRAM_LAYERS[0],
        layer_index: 0,
        rank: 0,
        world_size: 1,
        path,
        row_start: 0,
        physical_rows: rows,
        logical_rows: rows,
        window_bytes: 4096,
        tail_fill: Default::default(),
    }
}

fn open(path: &Path) -> DirectTable {
    let rows = table_rows(path);
    let table = DirectTable::open_with(
        vec![spec(path.to_path_buf(), rows)],
        BackendPreference::Auto,
        &mut |m| eprintln!("store: {m}"),
    )
    .expect("open synthetic table");
    println!(
        "BACKEND rows={} o_direct={} alignment={:?} window={:?} file_bytes={:?} fallback={:?}",
        rows,
        table.uses_o_direct(),
        table.alignment_of(0),
        table.window_of(0),
        table.file_bytes(0),
        table.fallback_reason()
    );
    table
}

fn cache_with(uma: usize, dram: usize) -> EngramRowCache {
    let c = EngramRowCache::with_capacities(uma, dram, ENGRAM_LAYERS.len()).expect("cache");
    let usable = c.capacity_bytes();
    println!(
        "CACHE budget uma={uma} dram={dram} usable={usable:?} rows={}",
        (usable.0 + usable.1) as u64 / ENGRAM_ROW_BYTES as u64
    );
    c
}

/// Deterministic uniform ids in 0..ROWS.
fn ids(n: usize, seed: u64, rows: u64) -> Vec<u64> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // SplitMix-style mixing so successive ids are not correlated with the table stride.
        let mixed = state.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        out.push((mixed >> 11) % rows);
    }
    out
}

fn make(path: &Path, rows: u64) {
    let file = std::fs::File::create(path).expect("create");
    let mut w = std::io::BufWriter::with_capacity(8 << 20, file);
    let mut row = [0u8; ENGRAM_ROW_BYTES];
    for r in 0..rows {
        for (i, b) in row.iter_mut().enumerate() {
            *b = (r as u8).wrapping_add(i as u8);
        }
        for b in row.iter_mut().skip(256) {
            *b = 0x7F;
        }
        w.write_all(&row).expect("write");
    }
    w.flush().expect("flush");
    drop(w);
    // Pad the sub-512-byte tail, exactly as the packer does.
    let bytes = std::fs::metadata(path).expect("stat").len();
    let pad = (512 - (bytes % 512)) % 512;
    if pad > 0 {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .expect("append");
        f.write_all(&vec![0u8; pad as usize]).expect("pad");
    }
    println!(
        "MADE rows={rows} bytes={} padded_to={}",
        rows * ENGRAM_ROW_BYTES as u64,
        std::fs::metadata(path).expect("stat").len()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("cold");
    let path = PathBuf::from(
        args.get(2)
            .cloned()
            .unwrap_or_else(|| "/tmp/engram_synth.bin".into()),
    );
    if mode == "make" {
        let rows: u64 = args
            .get(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_ROWS);
        make(&path, rows);
        return;
    }
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let rows = table_rows(&path);
    let table = open(&path);
    let mut out = [0u8; ENGRAM_ROW_BYTES];

    match mode {
        "cold" | "warm" | "hitrate" => {
            let cache = if mode == "hitrate" {
                let uma: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6 << 20);
                let dram: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(5 << 20);
                cache_with(uma, dram)
            } else {
                cache_with(UMA_BYTES, DRAM_BYTES)
            };
            let stream = ids(n, 0x9E37_79B9_7F4A_7C15, rows);
            let (rb0, rc0) = dev_io();
            let start = Instant::now();
            for &row in &stream {
                cache.get_or_fill(&table, 0, row, &mut out).expect("fill");
            }
            let elapsed = start.elapsed();
            let (rb1, rc1) = dev_io();
            let dev = rb1.saturating_sub(rb0);
            let logical = (n as u64) * ENGRAM_ROW_BYTES as u64;
            let snap = cache.snapshot();
            println!(
                "{mode} n={n} rows={rows} ns_per_row={:.1} rows_per_s={:.0} dev_bytes={dev} dev_bytes_per_row={:.1} rchar={} amplification={:.2} predicted=15.5 snapshot={snap:?}",
                elapsed.as_nanos() as f64 / n as f64,
                n as f64 / elapsed.as_secs_f64(),
                dev as f64 / n as f64,
                rc1.saturating_sub(rc0),
                dev as f64 / logical as f64,
            );
        }
        "batch" => {
            let cache = cache_with(UMA_BYTES, DRAM_BYTES);
            let random = ids(n, 0x1234_5678_9ABC_DEF1, rows);
            let base = ids(1, 0x5555_5555_5555_5555, rows.saturating_sub(n as u64 + 8))[0];
            let contiguous: Vec<u64> = (base..base + n as u64).collect();
            for (label, stream) in [("batch_random", &random), ("batch_contiguous", &contiguous)] {
                let rows: Vec<u32> = stream.iter().map(|r| *r as u32).collect();
                let mut out = vec![[0u8; ENGRAM_ROW_BYTES]; n];
                let (rb0, rc0) = dev_io();
                let start = Instant::now();
                let tiers = cache
                    .get_or_fill_rows(&table, 0, &rows, &mut out)
                    .expect("fill rows");
                let elapsed = start.elapsed();
                let (rb1, rc1) = dev_io();
                let dev = rb1.saturating_sub(rb0);
                let logical = (n as u64) * ENGRAM_ROW_BYTES as u64;
                let misses = tiers
                    .iter()
                    .filter(|t| **t == sparkflash_engram::cache::CacheTier::Nvme)
                    .count();
                let asked = rc1.saturating_sub(rc0);
                println!(
                    "{label} n={n} ns_per_row={:.1} dev_bytes={dev} dev_amplification={:.2} rchar={asked} asked_amplification={:.2} misses={misses}",
                    elapsed.as_nanos() as f64 / n as f64,
                    dev as f64 / logical as f64,
                    asked as f64 / logical as f64
                );
            }
        }
        "pipelined" => {
            let depth: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
            let stream = ids(n, 0xDEAD_BEEF_CAFE_F00D, rows);
            let source: Arc<dyn RowTable> = Arc::new(open(&path));
            let config = PrefetchConfig {
                queue_depth: depth,
                max_batch_rows: depth.max(1),
                ..PrefetchConfig::default()
            };
            let engine = PrefetchEngine::new(source, config).expect("engine");
            println!(
                "PREFETCH backend={:?} fallback={:?}",
                engine.backend(),
                engine.fallback_reason()
            );
            let (rb0, _) = dev_io();
            let start = Instant::now();
            let mut issued = 0usize;
            while issued < depth.min(n) {
                engine.request(0, stream[issued]);
                issued += 1;
            }
            let mut done = 0usize;
            let mut spins = 0u64;
            while done < n {
                let row = stream[done];
                loop {
                    if engine.take(0, row).is_some() {
                        break;
                    }
                    spins += 1;
                    std::hint::spin_loop();
                    if spins.is_multiple_of(1024) {
                        std::thread::yield_now();
                    }
                }
                done += 1;
                if issued < n {
                    engine.request(0, stream[issued]);
                    issued += 1;
                }
            }
            let elapsed = start.elapsed();
            let (rb1, _) = dev_io();
            println!(
                "pipelined depth={depth} n={n} ns_per_row={:.1} rows_per_s={:.0} dev_bytes_per_row={:.1} spins={spins}",
                elapsed.as_nanos() as f64 / n as f64,
                n as f64 / elapsed.as_secs_f64(),
                rb1.saturating_sub(rb0) as f64 / n as f64
            );
        }
        other => eprintln!("unknown mode {other}"),
    }
}
