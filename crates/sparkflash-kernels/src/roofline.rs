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

//! Per-node bandwidth roofline reporting for the decode-shape kernel harnesses.
//!
//! One node is one GB10. The roadmap budgets device work against 273 GB/s per node of achieved
//! weight bandwidth, so a harness that times a kernel can say how far it is from that floor
//! without claiming a throughput result: the number is a floor ratio, not a measurement of
//! sustained bandwidth and not a token rate. The helpers here exist so every harness prints the
//! ratio the same way, in its own log, next to the latency it already prints.

/// Per-node achieved-weight-bandwidth budget the roadmap uses: 273 GB/s for one GB10.
pub const NODE_GBPS: f64 = 273.0;

/// Achieved GB/s for `bytes` of traffic in one `ms` launch, or NaN when the time is not positive.
#[must_use]
pub fn achieved_gbs(bytes: u64, p50_ms: f64) -> f64 {
    if p50_ms > 0.0 {
        bytes as f64 / (p50_ms * 1.0e6)
    } else {
        f64::NAN
    }
}

/// Achieved GB/s as a percentage of [NODE_GBPS].
#[must_use]
pub fn percent_of_node(gbs: f64) -> f64 {
    100.0 * gbs / NODE_GBPS
}

/// The one-line roofline report a harness prints beside its latency.
///
/// `traffic_bytes` is the launch's own declared traffic (the operands the arguments address); it is
/// the harness's number, not a guess made here.
#[must_use]
pub fn line(kernel: &str, traffic_bytes: u64, p50_ms: f64) -> String {
    let gbs = achieved_gbs(traffic_bytes, p50_ms);
    let percent = percent_of_node(gbs);
    format!(
        "roofline: kernel={kernel} traffic_bytes={traffic_bytes} p50_ms={p50_ms:.4} \
         achieved_gbs={gbs:.2} node_gbs={NODE_GBPS:.1} pct_of_node={percent:.1}% \
         (per-node weight-bandwidth floor ratio, not a throughput claim)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_node_floor_is_one_hundred_percent() {
        // 273 GB in one second is exactly the node floor.
        assert!((achieved_gbs(273_000_000_000, 1000.0) - NODE_GBPS).abs() < 1e-9);
        assert!((percent_of_node(NODE_GBPS) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn the_cert_head_eleven_percent_row_is_reproduced() {
        // The brief's recorded cert_lm_head row: 11.1% of the node floor, which is the four rank
        // shard launches' 1,334,169,600 B of weight traffic inside 44 ms.
        let weight_bytes = 4 * (32_320u64 * 5120 * 2 + 32_320 * 40 * 2);
        assert_eq!(weight_bytes, 1_334_169_600);
        let percent = percent_of_node(achieved_gbs(weight_bytes, 44.0));
        assert!((percent - 11.1).abs() < 0.1, "percent {percent}");
    }

    #[test]
    fn a_zero_or_negative_time_is_not_a_ratio() {
        assert!(achieved_gbs(1, 0.0).is_nan());
        assert!(achieved_gbs(1, -1.0).is_nan());
    }

    #[test]
    fn the_line_carries_the_bytes_latency_and_percentage() {
        let text = line("k", 2_730_000_000, 1000.0);
        assert!(text.contains("traffic_bytes=2730000000"), "{text}");
        assert!(text.contains("p50_ms=1000.0000"), "{text}");
        assert!(text.contains("achieved_gbs=2.73"), "{text}");
        assert!(text.contains("pct_of_node=1.0%"), "{text}");
    }
}
