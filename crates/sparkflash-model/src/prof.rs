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

//! Per-stage device timings behind one declared switch.
//!
//! Every device arm that wanted to know where a step's milliseconds go grew its own accumulator,
//! its own print format and its own meaning for "rows": Qwen4-Exp's `SPARKFLASH_QWEN4_PROF` is
//! the only one, and GLM-5.3-Flash had none, which is why its 86 ms step could only be reasoned
//! about from a model rather than measured. This module is the accumulator and the report; an arm
//! places marks and owns the switch.
//!
//! The rules for reading a report are the ones the Qwen arm's page states, because they are
//! properties of the technique, not of a model:
//!
//! * **It serialises.** A mark drains the stream, so work that normally overlaps - a prefetch, a
//!   collective on another rank - is charged at its full cost. A stage's number is an upper bound
//!   on what removing it saves.
//! * **A mark costs a round trip** (about 4.5 us), so a step's report is longer than its own
//!   wall clock and the totals do not compare across builds.
//! * **Time is charged to the next mark**, so the first mark of a step collects whatever the host
//!   did between steps unless the caller marks the start.

use std::collections::HashMap;
use std::time::Instant;

/// One forward's stage timings.
#[derive(Debug)]
pub struct StageProfiler {
    enabled: bool,
    last: Instant,
    /// Per stage: how many marks, and the milliseconds they charged.
    stages: HashMap<&'static str, (usize, f64)>,
}

impl StageProfiler {
    /// A profiler that is on or off, without reading the environment.
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            last: Instant::now(),
            stages: HashMap::new(),
        }
    }

    /// Whether the arm should place marks at all: a disabled profiler costs one branch a mark.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Charges the time since the last mark, or since [Self::start], to `name`.
    ///
    /// The caller drains the stream first: that drain is what makes the charge real, and it is
    /// also why a profiled step is slower than an unprofiled one.
    pub fn mark(&mut self, name: &'static str) {
        if !self.enabled {
            return;
        }
        let now = Instant::now();
        let elapsed = now.duration_since(self.last).as_secs_f64() * 1000.0;
        let entry = self.stages.entry(name).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += elapsed;
        self.last = now;
    }

    /// Starts a forward's accounting.
    pub fn start(&mut self) {
        self.last = Instant::now();
        self.stages.clear();
    }

    /// The report for one forward, sorted by cost, or `None` when the profiler is off or the
    /// forward placed no marks.
    ///
    /// `rows` is the forward's row count, so a caller can group its reports by the shape it ran:
    /// a one-row step, an eight-row verify and a 4,096-row prefill chunk are different programs.
    #[must_use]
    pub fn report(&mut self, rows: usize, total_ms: f64) -> Option<String> {
        if !self.enabled || self.stages.is_empty() {
            return None;
        }
        let mut stages: Vec<(&'static str, usize, f64)> = self
            .stages
            .drain()
            .map(|(name, (count, ms))| (name, count, ms))
            .collect();
        stages.sort_by(|left, right| right.2.total_cmp(&left.2));
        let mut text = format!("prof rows {rows}: {total_ms:.2} ms total\n");
        for (name, count, ms) in stages {
            let share = if total_ms > 0.0 {
                100.0 * ms / total_ms
            } else {
                0.0
            };
            text.push_str(&format!(
                "prof   {name:<18} {ms:8.2} ms {count:4} x {share:5.1}%\n"
            ));
        }
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disabled_profiler_records_nothing_and_reports_nothing() {
        let mut profiler = StageProfiler::new(false);
        assert!(!profiler.is_enabled());
        profiler.start();
        profiler.mark("step");
        assert!(profiler.report(1, 1.0).is_none());
    }

    #[test]
    fn marks_accumulate_by_name_and_report_in_cost_order() {
        let mut profiler = StageProfiler::new(true);
        profiler.start();
        std::thread::sleep(std::time::Duration::from_millis(4));
        profiler.mark("experts");
        profiler.mark("mix");
        std::thread::sleep(std::time::Duration::from_millis(4));
        profiler.mark("experts");
        let report = profiler.report(8, 20.0).expect("a report");
        let experts = report.find("experts").expect("the stage");
        assert!(report[experts..].contains("2 x"), "{report}");
        assert!(
            report.starts_with("prof rows 8: 20.00 ms total"),
            "{report}"
        );
        // The experts stage accumulated two charges and outranks the mixer's one.
        assert!(experts < report.find("mix").expect("the mixer"), "{report}");
    }

    #[test]
    fn a_report_empties_the_accumulator_for_the_next_forward() {
        let mut profiler = StageProfiler::new(true);
        profiler.start();
        profiler.mark("head");
        assert!(profiler.report(1, 5.0).is_some());
        assert!(
            profiler.report(1, 5.0).is_none(),
            "a forward that placed no marks reports nothing"
        );
    }
}
