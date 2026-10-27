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

//! Trace export: Chrome/Perfetto trace.json, gzip output, and per-scope percentile tables.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::event::{CpuTraceEvent, GpuTraceEvent};
use crate::scope::TraceScope;

/// Chrome trace "category" for events that could not be attributed to a scope.
pub const UNKNOWN_CATEGORY: &str = "UNKNOWN";

/// Options controlling Chrome trace emission.
#[derive(Debug, Clone, Copy)]
pub struct TraceExportOptions {
    /// Process id stamped into every event.
    pub pid: u32,
    /// Thread id offset used for GPU events so they do not collide with CPU threads.
    pub gpu_tid_base: u32,
    /// Emit GPU events as well as CPU events.
    pub include_gpu: bool,
}

impl Default for TraceExportOptions {
    fn default() -> Self {
        Self {
            pid: 1,
            gpu_tid_base: 1000,
            include_gpu: true,
        }
    }
}

/// Per-scope aggregate statistics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScopeSummary {
    /// Scope bit.
    pub scope: u64,
    /// Canonical scope name.
    pub name: String,
    /// Number of spans.
    pub count: usize,
    /// Total time in nanoseconds.
    pub total_ns: u64,
    /// Mean span duration in nanoseconds.
    pub mean_ns: u64,
    /// 50th percentile span duration in nanoseconds.
    pub p50_ns: u64,
    /// 99th percentile span duration in nanoseconds.
    pub p99_ns: u64,
    /// Longest span in nanoseconds.
    pub max_ns: u64,
}

/// Nearest-rank percentile over an ascending slice. `q` is in [0, 1].
#[must_use]
pub fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let q = q.clamp(0.0, 1.0);
    let rank = (q * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

/// Aggregates CPU spans per scope, sorted by descending total time.
#[must_use]
pub fn summarize(events: &[CpuTraceEvent]) -> Vec<ScopeSummary> {
    let mut summaries = Vec::new();
    for (name, bit) in TraceScope::TABLE {
        let mut durations: Vec<u64> = events
            .iter()
            .filter(|e| e.scope & bit != 0)
            .map(CpuTraceEvent::duration_ns)
            .collect();
        if durations.is_empty() {
            continue;
        }
        durations.sort_unstable();
        let total_ns: u64 = durations.iter().sum();
        summaries.push(ScopeSummary {
            scope: bit,
            name: name.to_owned(),
            count: durations.len(),
            total_ns,
            mean_ns: total_ns / durations.len() as u64,
            p50_ns: percentile(&durations, 0.50),
            p99_ns: percentile(&durations, 0.99),
            max_ns: *durations.last().unwrap_or(&0),
        });
    }
    let unknown: Vec<u64> = events
        .iter()
        .filter(|e| e.scope & TraceScope::ALL == 0)
        .map(CpuTraceEvent::duration_ns)
        .collect();
    if !unknown.is_empty() {
        let total_ns: u64 = unknown.iter().sum();
        let mut sorted = unknown.clone();
        sorted.sort_unstable();
        summaries.push(ScopeSummary {
            scope: 0,
            name: UNKNOWN_CATEGORY.to_owned(),
            count: sorted.len(),
            total_ns,
            mean_ns: total_ns / sorted.len() as u64,
            p50_ns: percentile(&sorted, 0.50),
            p99_ns: percentile(&sorted, 0.99),
            max_ns: *sorted.last().unwrap_or(&0),
        });
    }
    summaries.sort_by(|a, b| b.total_ns.cmp(&a.total_ns).then(a.name.cmp(&b.name)));
    summaries
}

fn ns_to_us(ns: u64) -> f64 {
    ns as f64 / 1000.0
}

fn category_for(scope: u64) -> String {
    let names = TraceScope::describe(scope);
    if names.is_empty() {
        UNKNOWN_CATEGORY.to_owned()
    } else {
        names
    }
}

/// Serializes CPU and GPU spans into the Chrome tracing JSON format.
#[must_use]
pub fn to_chrome_trace_json(
    cpu: &[CpuTraceEvent],
    gpu: &[GpuTraceEvent],
    options: TraceExportOptions,
) -> String {
    let mut events: Vec<serde_json::Value> = Vec::with_capacity(cpu.len() + gpu.len());
    // The CPU side reads CLOCK_MONOTONIC-family time (on aarch64, CNTVCT_EL0) and the GPU side
    // reads %globaltimer. Both are boot-relative on GB10, but they are different counters, so
    // rebase the whole trace onto its earliest event and record the raw epoch in the metadata:
    // a viewer then shows offsets from the start of the capture instead of two absolute
    // nanosecond counters that happen to be comparable.
    let epoch_ns = cpu
        .iter()
        .map(|e| e.start_ns)
        .chain(gpu.iter().map(|e| e.start_ns))
        .min()
        .unwrap_or(0);
    let rebase = |ns: u64| ns_to_us(ns.saturating_sub(epoch_ns));
    for event in cpu {
        events.push(serde_json::json!({
            "name": event.label,
            "cat": category_for(event.scope),
            "ph": "X",
            "ts": rebase(event.start_ns),
            "dur": ns_to_us(event.duration_ns()),
            "pid": options.pid,
            "tid": event.thread_id,
            "args": { "meta": event.meta, "scope_mask": event.scope },
        }));
    }
    if options.include_gpu {
        for event in gpu {
            events.push(serde_json::json!({
                "name": format!("gpu:{}", category_for(event.scope_mask())),
                "cat": category_for(event.scope_mask()),
                "ph": "X",
                "ts": rebase(event.start_ns),
                "dur": ns_to_us(event.duration_ns()),
                "pid": options.pid,
                "tid": options.gpu_tid_base + event.rank(),
                "args": {
                    "sub_wait_ns": event.sub_wait_ns,
                    "layer_id": event.layer_id,
                    "payload_meta": event.payload_meta,
                    "rank": event.rank(),
                },
            }));
        }
    }
    let document = serde_json::json!({
        "traceEvents": events,
        "displayTimeUnit": "ms", // timestamps below are microseconds, the Chrome convention
        "metadata": {
            "generator": "sparkflash-trace",
            "epoch_offset_ns": epoch_ns,
            "clock_note": "CPU and GPU timestamps are rebased onto the earliest event of this capture",
            "scope_table": TraceScope::TABLE.iter().map(|(n, b)| serde_json::json!({ "name": n, "bit": b })).collect::<Vec<_>>(),
        },
    });
    serde_json::to_string(&document).unwrap_or_else(|_| "{\"traceEvents\":[]}".to_owned())
}

/// Writes a Chrome trace JSON document.
///
/// # Errors
/// Propagates any I/O failure from the filesystem.
pub fn write_chrome_trace_json(
    path: impl AsRef<Path>,
    cpu: &[CpuTraceEvent],
    gpu: &[GpuTraceEvent],
    options: TraceExportOptions,
) -> std::io::Result<()> {
    std::fs::write(path, to_chrome_trace_json(cpu, gpu, options))
}

/// Writes a gzip-compressed Chrome trace (trace.json.gz, directly loadable by Perfetto).
///
/// # Errors
/// Propagates any I/O failure from the filesystem.
pub fn write_chrome_trace_gz(
    path: impl AsRef<Path>,
    cpu: &[CpuTraceEvent],
    gpu: &[GpuTraceEvent],
    options: TraceExportOptions,
) -> std::io::Result<()> {
    let json = to_chrome_trace_json(cpu, gpu, options);
    let file = std::fs::File::create(path)?;
    let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::default());
    encoder.write_all(json.as_bytes())?;
    encoder.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::GpuTraceEvent;

    fn span(label: &'static str, scope: u64, start: u64, end: u64) -> CpuTraceEvent {
        CpuTraceEvent {
            label,
            scope,
            meta: 3,
            thread_id: 1,
            start_ns: start,
            end_ns: end,
        }
    }

    #[test]
    fn ch_json_contains_expected_fields() {
        let cpu = vec![span(
            "engram_lookup",
            TraceScope::ENGRAM_PREFETCH,
            1_000,
            3_500,
        )];
        let gpu = vec![GpuTraceEvent {
            start_ns: 4_000,
            end_ns: 9_000,
            sub_wait_ns: 2_000,
            scope_and_rank: (14u32 << 8) | 2,
            layer_id: 1,
            payload_meta: 2048,
        }];
        let json = to_chrome_trace_json(&cpu, &gpu, TraceExportOptions::default());
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let events = parsed["traceEvents"].as_array().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["name"], "engram_lookup");
        assert_eq!(events[0]["cat"], "ENGRAM_PREFETCH");
        assert_eq!(events[0]["ph"], "X");
        assert!((events[0]["dur"].as_f64().unwrap() - 2.5).abs() < 1e-9);
        assert_eq!(events[1]["cat"], "ROCENANTE_RDMA");
        assert_eq!(events[1]["tid"], 1000 + 2);
        assert_eq!(events[1]["args"]["sub_wait_ns"], 2000);
        assert!(parsed["metadata"]["scope_table"].as_array().unwrap().len() == 16);
    }

    #[test]
    fn gpu_events_can_be_excluded() {
        let gpu = vec![GpuTraceEvent::default()];
        let options = TraceExportOptions {
            include_gpu: false,
            ..Default::default()
        };
        let json = to_chrome_trace_json(&[], &gpu, options);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["traceEvents"].as_array().unwrap().is_empty());
    }

    #[test]
    fn gzip_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.json.gz");
        let cpu = vec![span("kv_radix_plan", TraceScope::KV_RADIX, 0, 1_000)];
        write_chrome_trace_gz(&path, &cpu, &[], TraceExportOptions::default()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let mut decoder = flate2::read::GzDecoder::new(&bytes[..]);
        let mut text = String::new();
        std::io::Read::read_to_string(&mut decoder, &mut text).unwrap();
        assert!(text.contains("kv_radix_plan"));
        assert!(text.contains("KV_RADIX"));
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let sorted = [10u64, 20, 30, 40, 50];
        assert_eq!(percentile(&sorted, 0.0), 10);
        assert_eq!(percentile(&sorted, 0.5), 30);
        assert_eq!(percentile(&sorted, 1.0), 50);
        assert_eq!(percentile(&[], 0.5), 0);
    }

    #[test]
    fn summary_aggregates_per_scope() {
        let cpu = vec![
            span("a", TraceScope::MEGA_MHC, 0, 1_000),
            span("b", TraceScope::MEGA_MHC, 0, 3_000),
            span("c", TraceScope::MOE_MXFP4, 0, 500),
        ];
        let summary = summarize(&cpu);
        assert_eq!(summary[0].name, "MEGA_MHC");
        assert_eq!(summary[0].count, 2);
        assert_eq!(summary[0].total_ns, 4_000);
        assert_eq!(summary[0].max_ns, 3_000);
        assert_eq!(summary[1].name, "MOE_MXFP4");
    }
}
