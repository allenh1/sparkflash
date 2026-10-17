//! What the Engram head actually costs on the CPU, using the reference rather than a reconstruction.
//!
//! Two questions, both from the task-33 brief:
//!   1. does the per-row consumer line move when the decode is engram_head_ref::decode_row instead of
//!      the inline E4M3 lookup table the first measurement used?
//!   2. what is the real gate, now that the formula is the tree's (forward/engram.rs:624-672) and not
//!      a 3-order sigmoid mix invented for the first pass?
//!
//! usage: head_ref_cost [iterations] [dim] [hc] [columns] [tokens]

use std::io::Write;
use std::time::Instant;

use sparkflash_engram::cache::{CacheTier, EngramRowCache};
use sparkflash_engram::hash::{MAX_NGRAM_SIZE, hash_window};
use sparkflash_engram::layout::{
    ENGRAM_HEAD_DIM, ENGRAM_LAYERS, ENGRAM_ROW_BYTES, ENGRAM_SCALE_BYTES,
};
use sparkflash_engram::store::{BackendPreference, DirectTable, TableSpec};
use sparkflash_kernels::engram_head_ref::{self, HeadShape};
use sparkflash_trace::{TraceScope, arm, disarm, drain_cpu_events, spark_trace_span};

const TABLE_ROWS: u64 = 1 << 20;
const WORKING_SET: u64 = 4096;

fn make_table(path: &std::path::Path, rows: u64) {
    let file = std::fs::File::create(path).expect("create");
    let mut w = std::io::BufWriter::with_capacity(4 << 20, file);
    let mut row = [0u8; ENGRAM_ROW_BYTES];
    let mut state = 0x2468_ACE0_1357_9BDFu64;
    for _ in 0..rows {
        for b in row.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = state as u8;
        }
        for (s, b) in row.iter_mut().skip(ENGRAM_HEAD_DIM).enumerate() {
            *b = if s.is_multiple_of(7) { 0x77 } else { 0x78 };
        }
        w.write_all(&row).expect("write");
    }
    w.flush().expect("flush");
    let bytes = std::fs::metadata(path).expect("stat").len();
    let pad = (512 - (bytes % 512)) % 512;
    if pad > 0 {
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .expect("append");
        f.write_all(&vec![0u8; pad as usize]).expect("pad");
    }
}

fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[((sorted.len() as f64 - 1.0) * q).round() as usize]
}

fn report(label: &str, mut values: Vec<u64>) {
    if values.is_empty() {
        println!("{label:34}: no samples");
        return;
    }
    values.sort_unstable();
    let n = values.len();
    let sum: u128 = values.iter().map(|v| u128::from(*v)).sum();
    println!(
        "{label:34} n={n:8} mean={:9.0} ns p50={:8} p90={:8} p99={:8} max={:10}",
        sum as f64 / n as f64,
        percentile(&values, 0.50),
        percentile(&values, 0.90),
        percentile(&values, 0.99),
        values[n - 1]
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5120);
    let hc: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);
    let columns: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(24);
    let tokens: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);

    let dir = std::env::temp_dir().join(format!("head_ref_cost_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let table_path = dir.join("engram-l1-r0of1.bin");
    make_table(&table_path, TABLE_ROWS);
    let spec = TableSpec {
        layer_id: ENGRAM_LAYERS[0],
        layer_index: 0,
        rank: 0,
        world_size: 1,
        path: table_path.clone(),
        row_start: 0,
        physical_rows: TABLE_ROWS,
        logical_rows: TABLE_ROWS,
        window_bytes: 4096,
        tail_fill: Default::default(),
    };
    let table = DirectTable::open_with(vec![spec], BackendPreference::Auto, &mut |m| {
        eprintln!("store: {m}")
    })
    .expect("open");
    let cache =
        EngramRowCache::with_capacities(256 << 20, 768 << 20, ENGRAM_LAYERS.len()).expect("cache");
    println!(
        "table {} rows, cache {:?}, hit path: working set {WORKING_SET} rows",
        TABLE_ROWS,
        cache.capacity_bytes()
    );

    // ---- question 1: the per-row line with the reference decoder -------------------------------
    let mut buf = [0u8; ENGRAM_ROW_BYTES];
    let mut weight = [0u8; ENGRAM_HEAD_DIM];
    let mut scale = [0u8; ENGRAM_SCALE_BYTES];
    let mut acc = [0.0f32; ENGRAM_HEAD_DIM];
    let mut fetch_ns = Vec::with_capacity(iterations);
    let mut decode_ns = Vec::with_capacity(iterations);
    let mut tiers = 0usize;
    arm(
        TraceScope::ENGRAM_PREFETCH | TraceScope::ENGRAM_GPU_GATE,
        u32::MAX,
    );
    let started = Instant::now();
    for index in 0..iterations {
        let row = (index as u64 * 2_654_435_761) % WORKING_SET;
        let t0 = Instant::now();
        let tier = {
            spark_trace_span!(TraceScope::ENGRAM_PREFETCH, "ref_row_fetch", row as u32);
            cache.get_or_fill(&table, 0, row, &mut buf).expect("fill")
        };
        let t1 = Instant::now();
        if tier != CacheTier::Nvme {
            tiers += 1;
        }
        fetch_ns.push((t1 - t0).as_nanos() as u64);
        let t2 = Instant::now();
        {
            spark_trace_span!(TraceScope::ENGRAM_GPU_GATE, "ref_head_decode", row as u32);
            weight.copy_from_slice(&buf[..ENGRAM_HEAD_DIM]);
            scale.copy_from_slice(&buf[ENGRAM_HEAD_DIM..]);
            let decoded = engram_head_ref::decode_row(&weight, &scale);
            for (slot, value) in acc.iter_mut().zip(decoded.iter()) {
                *slot += *value;
            }
        }
        let t3 = Instant::now();
        decode_ns.push((t3 - t2).as_nanos() as u64);
        if index % 1500 == 0 {
            // Keep the per-thread ring fresh: it holds 8192 spans, and the Instant vectors above are
            // the full-population distribution anyway.
            let _ = drain_cpu_events();
        }
    }
    let elapsed = started.elapsed();
    disarm();
    let events = drain_cpu_events();
    std::hint::black_box(&acc);
    println!("\n== question 1: the row consumer with engram_head_ref::decode_row ==");
    report("reference decode only", decode_ns.clone());
    report("cache fetch only", fetch_ns.clone());
    let fetch_mean = fetch_ns.iter().sum::<u64>() as f64 / fetch_ns.len() as f64;
    let decode_mean = decode_ns.iter().sum::<u64>() as f64 / decode_ns.len() as f64;
    println!(
        "per row: fetch {:.0} ns + reference decode {:.0} ns = {:.0} ns ({:.2} us), cache hits {tiers}/{iterations}, wall {} ms, ring {} spans",
        fetch_mean,
        decode_mean,
        fetch_mean + decode_mean,
        (fetch_mean + decode_mean) / 1000.0,
        elapsed.as_millis(),
        events.len()
    );
    println!(
        "first-pass harness was 223 ns fetch + 824 ns inline-LUT decode = 1.05 us/row; the inline decode also included a per-element scale powi the reference hoists, so the two are not the same work"
    );

    // ---- question 2: the real gate --------------------------------------------------------------
    let fetched_width = columns * ENGRAM_HEAD_DIM;
    let out_width = dim * (hc + 1);
    println!(
        "\n== question 2: the real head, dim {dim} hc {hc} columns {columns}: wkv {out_width} x {fetched_width} = {} f32 ({:.0} MiB) ==",
        out_width * fetched_width,
        (out_width * fetched_width * 4) as f64 / (1024.0 * 1024.0)
    );
    let mut wkv = vec![0.0f32; out_width * fetched_width];
    // A cheap non-zero pattern: the exact values do not change the operation count.
    for (index, value) in wkv.iter_mut().enumerate() {
        *value = ((index % 17) as f32 - 8.0) / 64.0;
    }
    let product = vec![1.0f32; hc * dim];
    let shape = HeadShape {
        dim,
        hc,
        eps: 1e-6,
        head_dim: ENGRAM_HEAD_DIM,
    };
    let mut x_hc = vec![0.5f32; hc * dim];
    let mut windows: Vec<[i64; MAX_NGRAM_SIZE]> = Vec::new();
    let mut state = 0x1357_9BDF_2468_ACE0u64;
    for _ in 0..tokens {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let base = (state % 40_000) as i64;
        windows.push([base, base + 1, base + 2, base + 3]);
    }
    let mut gate_values: Vec<Vec<f32>> = Vec::new();
    let mut head_ns = Vec::new();
    for window in &windows {
        let ids_full = hash_window(0, window);
        let ids: Vec<u32> = ids_full.iter().take(columns).copied().collect();
        let mut rows = |id: u32| {
            let row = u64::from(id) % WORKING_SET;
            let mut buf = [0u8; ENGRAM_ROW_BYTES];
            cache.get_or_fill(&table, 0, row, &mut buf).expect("fill");
            let mut weight = [0u8; ENGRAM_HEAD_DIM];
            let mut scale = [0u8; ENGRAM_SCALE_BYTES];
            weight.copy_from_slice(&buf[..ENGRAM_HEAD_DIM]);
            scale.copy_from_slice(&buf[ENGRAM_HEAD_DIM..]);
            engram_head_ref::decode_row(&weight, &scale).to_vec()
        };
        let t0 = Instant::now();
        let gates = engram_head_ref::head_ref(&mut x_hc, &ids, &mut rows, &wkv, &product, shape)
            .expect("no sentinel in this stream");
        let t1 = Instant::now();
        head_ns.push((t1 - t0).as_nanos() as u64);
        gate_values.push(gates);
    }
    std::hint::black_box(&x_hc);
    let head_ms: Vec<f64> = head_ns.iter().map(|v| *v as f64 / 1e6).collect();
    println!("per token, {tokens} tokens: {head_ms:?} ms");
    let macs_per_token_per_layer = (out_width as u128) * (fetched_width as u128);
    println!(
        "one token, one layer: {macs_per_token_per_layer} multiply-accumulates for wkv, plus {hc} gates of {dim} and a {hc} x {dim} write"
    );
    println!(
        "two layers per token: {} MAC = {:.3} GFLOP; the same token's I/O at depth 32 is 0.87 ms",
        macs_per_token_per_layer * 2,
        (macs_per_token_per_layer * 2 * 2) as f64 / 1e9
    );
    println!(
        "gates (first three tokens): {:?}",
        gate_values.iter().take(3).collect::<Vec<_>>()
    );
    let _ = std::fs::remove_dir_all(&dir);
}
