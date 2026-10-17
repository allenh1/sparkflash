//! Per-row consumer cost: the row fetch plus the Engram head computation that consumes it.
//!
//! Producer/consumer budget work for the M2 read path. Everything here is local and synthetic:
//! a temporary rank file, the real cache geometry, the real 24-column hash, and the M1 trace ring
//! for the per-phase distributions. No checkpoint and no node I/O.
//!
//! usage: row_consumer [table_rows] [tokens] [working_set_rows]

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use sparkflash_engram::cache::{CacheTier, EngramRowCache};
use sparkflash_engram::hash::{MAX_NGRAM_SIZE, hash_window};
use sparkflash_engram::layout::{
    ENGRAM_HEAD_DIM, ENGRAM_LAYERS, ENGRAM_ROW_BYTES, ENGRAM_SCALE_BYTES, SCALE_BLOCK,
};
use sparkflash_engram::store::{BackendPreference, DirectTable, TableSpec};
use sparkflash_trace::{
    TraceScope, arm, cpu_ring_stats, disarm, drain_cpu_events, spark_trace_span,
};

const LOOKUPS_PER_TOK_PER_LAYER: usize = 24;
/// One layer's three orders x eight heads x 256 dims.
const WIDE: usize = 3 * 8 * ENGRAM_HEAD_DIM;

/// E4M3 (OCP, bias 7) to f32, built once as the table a real consumer would use.
fn e4m3_table() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for (byte, slot) in t.iter_mut().enumerate() {
        let b = byte as u32;
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let exp = (b >> 3) & 0x0F;
        let mant = b & 0x07;
        let value = match exp {
            0 => (mant as f32) * (1.0 / 8.0) * 2f32.powi(-6),
            0x0F if mant == 0x07 => f32::NAN,
            _ => (1.0 + (mant as f32) / 8.0) * 2f32.powi(exp as i32 - 7),
        };
        *slot = sign * value;
    }
    t
}

fn make_table(path: &PathBuf, rows: u64) {
    let file = std::fs::File::create(path).expect("create");
    let mut w = std::io::BufWriter::with_capacity(4 << 20, file);
    let mut row = [0u8; ENGRAM_ROW_BYTES];
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    for r in 0..rows {
        for b in row.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = state as u8;
        }
        // Keep the block scales in a sane exponent range, like the real rows (about 2^-7).
        for (s, b) in row.iter_mut().skip(ENGRAM_HEAD_DIM).enumerate() {
            *b = if (r as usize + s).is_multiple_of(7) {
                0x77
            } else {
                0x78
            };
        }
        w.write_all(&row).expect("write");
    }
    w.flush().expect("flush");
    drop(w);
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
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx]
}

fn report(label: &str, mut values: Vec<u64>) {
    if values.is_empty() {
        println!("{label}: no samples");
        return;
    }
    values.sort_unstable();
    let n = values.len();
    let sum: u128 = values.iter().map(|v| u128::from(*v)).sum();
    println!(
        "{label:24} n={n:7} mean={:8.0} ns  p50={:6} p90={:6} p99={:7} p99.9={:7} max={:8} min={:5}",
        sum as f64 / n as f64,
        percentile(&values, 0.50),
        percentile(&values, 0.90),
        percentile(&values, 0.99),
        percentile(&values, 0.999),
        values[n - 1],
        values[0]
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let table_rows: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
    let tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let working_set: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);

    let dir = std::env::temp_dir().join(format!("row_consumer_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("engram-l1-r0of1.bin");
    make_table(&path, table_rows);
    println!(
        "table rows={table_rows} bytes={}",
        std::fs::metadata(&path).unwrap().len()
    );

    let spec = TableSpec {
        layer_id: ENGRAM_LAYERS[0],
        layer_index: 0,
        rank: 0,
        world_size: 1,
        path: path.clone(),
        row_start: 0,
        physical_rows: table_rows,
        logical_rows: table_rows,
        window_bytes: 4096,
        tail_fill: Default::default(),
    };
    let table = DirectTable::open_with(vec![spec], BackendPreference::Auto, &mut |m| {
        eprintln!("store: {m}")
    })
    .expect("open");
    println!(
        "backend o_direct={} alignment={:?} window={:?} fallback={:?}",
        table.uses_o_direct(),
        table.alignment_of(0),
        table.window_of(0),
        table.fallback_reason()
    );
    // The real budget: 1 GiB total, the same tiers a node with ~1.2 GB free can host.
    let cache =
        EngramRowCache::with_capacities(256 << 20, 768 << 20, ENGRAM_LAYERS.len()).expect("cache");
    println!(
        "cache usable={:?} rows={}",
        cache.capacity_bytes(),
        (cache.capacity_bytes().0 + cache.capacity_bytes().1) as u64 / ENGRAM_ROW_BYTES as u64
    );

    // A token stream: the real 24-column hash, one window of 4 token ids per token per layer.
    let mut windows: Vec<[i64; MAX_NGRAM_SIZE]> = Vec::with_capacity(tokens);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for t in 0..tokens as i64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let base = (state % 50_000) as i64 + t;
        windows.push([
            base % 99_092,
            (base + 1) % 99_092,
            (base + 2) % 99_092,
            (base + 3) % 99_092,
        ]);
    }

    let lut = e4m3_table();
    let mut acc = vec![0.0f32; WIDE];
    let mut out = vec![0.0f32; 8 * ENGRAM_HEAD_DIM];
    let mut buf = [0u8; ENGRAM_ROW_BYTES];
    let mut hits = 0usize;
    let mut tiers: HashMap<&'static str, usize> = HashMap::new();
    // Instant-based cross-check, in case the ring drops events.
    let mut wall_fetch = Vec::new();
    let mut wall_dequant = Vec::new();
    let mut wall_gate = Vec::new();
    let mut rows_done = 0usize;
    let mut tokens_done = 0usize;
    // The per-thread ring holds 8,192 spans, so drain it well inside that or the tail is all we see.
    let mut by_label: HashMap<&'static str, Vec<u64>> = HashMap::new();
    let mut since_drain = 0usize;

    arm(
        TraceScope::ENGRAM_PREFETCH | TraceScope::ENGRAM_GPU_GATE,
        u32::MAX,
    );
    let started = Instant::now();
    'outer: for (tok, window) in windows.iter().enumerate() {
        let _ = tok;
        let ids = hash_window(0, window);
        for order in 0..3usize {
            for head in 0..8usize {
                let col = order * 8 + head;
                let span_rows = if working_set == 0 {
                    table_rows
                } else {
                    working_set.min(table_rows)
                };
                let row = u64::from(ids[col]) % span_rows;
                let t0 = Instant::now();
                let tier = {
                    spark_trace_span!(TraceScope::ENGRAM_PREFETCH, "engram_row_fetch", row as u32);
                    cache.get_or_fill(&table, 0, row, &mut buf).expect("fill")
                };
                let t1 = Instant::now();
                wall_fetch.push((t1 - t0).as_nanos() as u64);
                if tier != CacheTier::Nvme {
                    hits += 1;
                }
                *tiers
                    .entry(match tier {
                        CacheTier::Uma => "uma",
                        CacheTier::Dram => "dram",
                        CacheTier::Nvme => "nvme",
                    })
                    .or_default() += 1;

                let t2 = Instant::now();
                {
                    spark_trace_span!(
                        TraceScope::ENGRAM_GPU_GATE,
                        "engram_head_dequant",
                        col as u32
                    );
                    let base = order * 8 * ENGRAM_HEAD_DIM + head * ENGRAM_HEAD_DIM;
                    for i in 0..ENGRAM_HEAD_DIM {
                        let scale = 2f32.powi(buf[ENGRAM_HEAD_DIM + i / SCALE_BLOCK] as i32 - 127);
                        acc[base + i] += lut[buf[i] as usize] * scale;
                    }
                    debug_assert_eq!(ENGRAM_SCALE_BYTES, ENGRAM_HEAD_DIM / SCALE_BLOCK);
                }
                let t3 = Instant::now();
                wall_dequant.push((t3 - t2).as_nanos() as u64);
                rows_done += 1;
                since_drain += 1;
                if since_drain >= 1500 {
                    for event in drain_cpu_events() {
                        by_label
                            .entry(event.label)
                            .or_default()
                            .push(event.duration_ns());
                    }
                    since_drain = 0;
                }
                if rows_done.is_multiple_of(LOOKUPS_PER_TOK_PER_LAYER) {
                    // One token, one layer: mix the three 2048-wide order vectors through the gate.
                    let t4 = Instant::now();
                    {
                        spark_trace_span!(
                            TraceScope::ENGRAM_GPU_GATE,
                            "engram_head_gate",
                            tokens_done as u32
                        );
                        for i in 0..out.len() {
                            let a = acc[i];
                            let b = acc[8 * ENGRAM_HEAD_DIM + i];
                            let c = acc[2 * 8 * ENGRAM_HEAD_DIM + i];
                            let g = 1.0 / (1.0 + (-(a + b + c)).exp());
                            out[i] = g * (a + b) + (1.0 - g) * c;
                        }
                        for slot in acc.iter_mut() {
                            *slot = 0.0;
                        }
                    }
                    let t5 = Instant::now();
                    wall_gate.push((t5 - t4).as_nanos() as u64);
                    // Keep the whole chain alive: out is written by the gate, acc by the dequant.
                    std::hint::black_box(&out);
                    tokens_done += 1;
                }
                if tokens_done >= tokens {
                    break 'outer;
                }
            }
        }
    }
    let elapsed = started.elapsed();
    disarm();

    for event in drain_cpu_events() {
        by_label
            .entry(event.label)
            .or_default()
            .push(event.duration_ns());
    }
    let (threads, recorded, dropped) = cpu_ring_stats();
    println!(
        "\n== M1 trace ring: {} spans reported, {} recorded, {} dropped, {} thread(s) ==",
        by_label.values().map(Vec::len).sum::<usize>(),
        recorded,
        dropped,
        threads
    );
    let mut labels: Vec<&'static str> = by_label.keys().copied().collect();
    labels.sort();
    for label in labels {
        let values = by_label.remove(label).unwrap();
        report(label, values);
    }
    println!("\n== wall clock cross-check ==");
    report("instant engram_row_fetch", wall_fetch.clone());
    report("instant engram_head_dequant", wall_dequant.clone());
    report("instant engram_head_gate", wall_gate.clone());
    println!(
        "\nrows={rows_done} tokens={tokens_done} hit_rate={:.4} tiers={tiers:?} elapsed_ms={}",
        hits as f64 / rows_done.max(1) as f64,
        elapsed.as_millis()
    );
    let per_row = (wall_fetch.iter().sum::<u64>() + wall_dequant.iter().sum::<u64>()) as f64
        / rows_done.max(1) as f64;
    let gate_per_row = wall_gate.iter().sum::<u64>() as f64 / rows_done.max(1) as f64;
    println!(
        "consumer per row: fetch+dequant = {per_row:.0} ns, plus amortised gate {gate_per_row:.0} ns = {:.0} ns ({} us)",
        per_row + gate_per_row,
        (per_row + gate_per_row) / 1000.0
    );
    println!(
        "verdict: consumer {} the 18 us needed to keep 32 reads in flight",
        if per_row + gate_per_row >= 18_000.0 {
            "MEETS"
        } else {
            "is FAR BELOW"
        }
    );
    let _ = std::fs::remove_dir_all(&dir);
}
