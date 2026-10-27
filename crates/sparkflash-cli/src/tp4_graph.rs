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

//! WS-C: CUDA graph capture for the lean decode path.
//!
//! The lean decode loop is still host-driven: each layer-position allocates pooled buffers,
//! uploads inputs, issues a short run of kernel launches, then reads a few results back. A CUDA
//! graph can only cover the launch-only stretches between those host operations, and only while
//! the device addresses those launches read stay the same. This module is the wrapper that decides,
//! per segment, whether a captured graph may be replayed:
//!
//! * [`segment`] runs one launch-only stretch through the driver. It records the launches first
//!   (cheap host bookkeeping, no driver call), and replays a previously captured graph only when
//!   the recorded launch list is identical - same function handles, grids, blocks and every
//!   argument word. A changed argument (a different pooled buffer, another layer's weight slice)
//!   is never patched into a graph: the recorded launches are executed directly instead.
//! * A stretch that is not launch-only (it allocates, copies, synchronizes, loads a module, frees
//!   or releases a pooled buffer) is refused by the driver with a named error. The segment is then
//!   disabled for the rest of the run and its launches go through the direct path; the refusal is
//!   reported once. That report is the useful part of the experiment: it names the exact host
//!   operation that keeps the decode loop from being capturable.
//! * Capture itself is a warm-up event: the first occurrence of a launch list is captured and
//!   replayed once, and the same list replays on later layers and positions. Capture failures
//!   (an allocation or a readback inside the stretch, a bracket that cannot be captured) are
//!   reported and never retried, so the run cannot oscillate between graph and direct paths.
//!
//! The switch is `SPARKFLASH_CUDA_GRAPH`:
//! * `off` (default, or `0`): never touch the driver's graph state.
//! * `on` (or `1`): capture and replay when the launch list matches.
//! * `report`: attempt one capture per distinct launch list and report it, then run the direct
//!   path for the rest of the run. The row's arithmetic is the direct path's; the report says
//!   what capture would have covered and what blocked it.
//!
//! This module only wraps launches; it does not change a kernel, an argument, a grid or the order
//! of anything. The arithmetic path of a replayed graph is the same launch list that was captured.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use sparkflash_kernels::cuda_driver::{Cuda, CudaLaunchRecord, GraphId};

/// The most distinct launch lists one seam captures.
///
/// A launch list is distinct when any kernel, grid, block or argument word differs. A decode layer
/// whose weight slices are copied into the same staged buffers reuses one list for every layer and
/// position; a layer that launches directly from its own weights needs one list per layer. The cap
/// bounds the executables a run can instantiate (the driver also caps its own); past it the seam
/// executes recorded launches and reports that it stopped capturing.
const MAX_CAPTURES_PER_SEAM: usize = 32;

/// How often the cumulative seam report is printed, in segment calls.
const SUMMARY_EVERY: u64 = 256;

/// The launch list identity: every recorded launch's function, grid, block and argument words.
///
/// Comparing these is the whole safety condition for a replay. The wrapper never inspects or
/// patches a device address; a changed word is simply a different fingerprint, which executes the
/// recorded launches instead of the graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LaunchFingerprint(Vec<u64>);

impl LaunchFingerprint {
    /// Flattens a recorded launch list into one word vector.
    fn of(records: &[CudaLaunchRecord]) -> Self {
        let mut words = Vec::new();
        for record in records {
            words.extend(record.fingerprint_words());
        }
        Self(words)
    }

    /// A fingerprint from raw words, for the host tests of the capture plan.
    #[cfg(test)]
    fn from_words(words: &[u64]) -> Self {
        Self(words.to_vec())
    }
}

/// The `SPARKFLASH_CUDA_GRAPH` switch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphMode {
    /// Never touch the driver's graph state; every segment runs the direct path.
    Off,
    /// Capture once per launch list to report coverage and blockers, then run the direct path.
    Report,
    /// Capture and replay when the launch list matches; fall back to the direct path otherwise.
    On,
}

impl GraphMode {
    /// Parses one value of `SPARKFLASH_CUDA_GRAPH`. An unknown value is an error, not a silent
    /// fallback: a typo must not look like a graph row.
    fn from_env_value(raw: Option<&str>) -> Result<Self, String> {
        match raw.map(str::trim) {
            None | Some("") | Some("0") | Some("off") => Ok(Self::Off),
            Some("1") | Some("on") | Some("capture") => Ok(Self::On),
            Some("report") => Ok(Self::Report),
            Some(other) => Err(format!(
                "SPARKFLASH_CUDA_GRAPH={other:?} is not one of off|report|on"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Report => "report",
            Self::On => "on",
        }
    }
}

/// The switch, read once per process.
fn mode() -> Result<GraphMode, String> {
    static MODE: OnceLock<Result<GraphMode, String>> = OnceLock::new();
    MODE.get_or_init(|| {
        GraphMode::from_env_value(std::env::var("SPARKFLASH_CUDA_GRAPH").ok().as_deref())
    })
    .clone()
}

/// The state a run has to satisfy before a graph is worth capturing.
///
/// The async launch path is a precondition because capture is only enabled where the surrounding
/// code already tolerates queued launches: the wrapper's fallback queues the recorded launches and
/// lets the next readback be the observation point, which is exactly the async contract. The two
/// bracket consumers are refused because both record events around a launch, and an event pair
/// cannot be part of a captured graph.
fn check_preconditions() -> Result<(), String> {
    if !sparkflash_kernels::cuda_driver::async_launches() {
        return Err(
            "graph replay needs the async launch path (SPARKFLASH_SYNC_LAUNCHES is set)"
                .to_string(),
        );
    }
    if sparkflash_kernels::cuda_driver::stage_timing_enabled() {
        return Err(
            "stage timing arms an event bracket around launches and cannot be captured".to_string(),
        );
    }
    if sparkflash_kernels::cuda_driver::probe_meter_armed() {
        return Err(
            "the probe meter arms an event bracket and a drain around launches and cannot be captured"
                .to_string(),
        );
    }
    Ok(())
}

/// The precondition result, computed once per process.
fn preconditions() -> Result<(), String> {
    static PRECONDITIONS: OnceLock<Result<(), String>> = OnceLock::new();
    PRECONDITIONS.get_or_init(check_preconditions).clone()
}

/// One captured graph executable.
struct CapturedGraph {
    handle: GraphId,
}

/// What to do with the launches that were just recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlanAction {
    /// Run the segment under real capture, instantiate the graph, and replay it once.
    Capture,
    /// Replay the captured graph for this launch list.
    Replay,
    /// Queue the recorded launches directly: no captured graph matches, or graphs are disabled.
    ExecuteRecorded,
}

/// The pure decision part of a seam, separated from the driver calls so the plan is testable on a
/// host with no device.
struct SeamPlan {
    mode: GraphMode,
    /// Launch lists that have a captured graph executable.
    captured: HashSet<LaunchFingerprint>,
    /// Set once, with the reason, when the seam must stop capturing. Every later call executes the
    /// direct path; a failed or refused capture is never retried, so one bad stretch cannot make
    /// the run oscillate between graph and direct paths.
    disabled: Option<String>,
}

impl SeamPlan {
    fn new(mode: GraphMode) -> Self {
        Self {
            mode,
            captured: HashSet::new(),
            disabled: None,
        }
    }

    /// Decides what to do with one recorded launch list.
    fn decide(&mut self, fingerprint: &LaunchFingerprint) -> PlanAction {
        if self.mode == GraphMode::Off {
            return PlanAction::ExecuteRecorded;
        }
        if self.disabled.is_some() {
            return PlanAction::ExecuteRecorded;
        }
        if self.captured.contains(fingerprint) {
            return match self.mode {
                GraphMode::On => PlanAction::Replay,
                _ => PlanAction::ExecuteRecorded,
            };
        }
        if self.captured.len() >= MAX_CAPTURES_PER_SEAM {
            self.disabled = Some(format!(
                "captured {MAX_CAPTURES_PER_SEAM} distinct launch lists and the budget is full; the remaining lists run the direct path"
            ));
            return PlanAction::ExecuteRecorded;
        }
        PlanAction::Capture
    }

    /// Records that the launch list now has a captured graph executable.
    fn note_captured(&mut self, fingerprint: LaunchFingerprint) {
        self.captured.insert(fingerprint);
    }

    /// Disables capture for this seam, keeping the first reason.
    fn note_failure(&mut self, reason: String) {
        if self.disabled.is_none() {
            self.disabled = Some(reason);
        }
    }
}

/// One captured seam: its plan, its executables, and its counters.
struct GraphSeam {
    name: &'static str,
    plan: SeamPlan,
    captured: HashMap<LaunchFingerprint, CapturedGraph>,
    calls: u64,
    replays: u64,
    fallbacks: u64,
    captured_launches: usize,
    disable_reported: bool,
}

impl GraphSeam {
    fn new(name: &'static str, mode: GraphMode) -> Self {
        Self {
            name,
            plan: SeamPlan::new(mode),
            captured: HashMap::new(),
            calls: 0,
            replays: 0,
            fallbacks: 0,
            captured_launches: 0,
            disable_reported: false,
        }
    }

    /// Reports a seam refusal once, and remembers that the run already heard it.
    fn report_disable(&mut self, reason: &str) {
        if self.disable_reported {
            return;
        }
        self.disable_reported = true;
        println!(
            "tp4-graph: seam {:?} disabled for the rest of the run: {reason}",
            self.name
        );
    }

    /// Prints a cumulative report line every [SUMMARY_EVERY] calls.
    fn summary_if_due(&self) {
        if !self.calls.is_multiple_of(SUMMARY_EVERY) {
            return;
        }
        println!("tp4-graph: {}", self.summary());
    }

    fn summary(&self) -> String {
        let mut text = format!(
            "seam {:?} mode={} calls={} captured_lists={} captured_launches={} replays={} fallbacks={}",
            self.name,
            self.plan.mode.as_str(),
            self.calls,
            self.captured.len(),
            self.captured_launches,
            self.replays,
            self.fallbacks
        );
        if let Some(reason) = &self.plan.disabled {
            text.push_str(&format!(" disabled={reason:?}"));
        }
        text
    }

    /// Runs one segment: record, decide, and either replay, capture or execute.
    fn run<F>(&mut self, cuda: &Cuda, segment: &mut F) -> Result<(), String>
    where
        F: FnMut(&Cuda) -> Result<(), String>,
    {
        self.calls += 1;
        let result = self.run_recorded(cuda, segment);
        self.summary_if_due();
        result
    }

    /// The record/decide/replay body of [GraphSeam::run].
    fn run_recorded<F>(&mut self, cuda: &Cuda, segment: &mut F) -> Result<(), String>
    where
        F: FnMut(&Cuda) -> Result<(), String>,
    {
        // A disabled seam never records again: the direct path is the run's path from here.
        if self.plan.disabled.is_some() {
            return segment(cuda);
        }
        if cuda.graph_record_begin().is_err() {
            // Another segment is active (nested seam, or a mode the wrapper did not expect). The
            // direct path is always correct; a graph is an optimization, never the semantics.
            return segment(cuda);
        }
        let recorded = match segment(cuda) {
            Ok(()) => match cuda.graph_record_end() {
                Ok(records) => records,
                Err(error) => {
                    cuda.graph_record_abort();
                    return Err(format!("tp4-graph: record ended badly: {error}"));
                }
            },
            Err(error) => {
                // The stretch is not launch-only: a copy, allocation, synchronize, module load or
                // free was refused with a named reason. That is the experiment's finding, not a
                // run failure: disable capture and execute the stretch on the direct path.
                cuda.graph_record_abort();
                self.plan
                    .note_failure(format!("segment is not launch-only: {error}"));
                let reason = self
                    .plan
                    .disabled
                    .clone()
                    .unwrap_or_else(|| "segment is not launch-only".to_string());
                self.report_disable(&reason);
                return segment(cuda);
            }
        };
        let fingerprint = LaunchFingerprint::of(&recorded);
        match self.plan.decide(&fingerprint) {
            PlanAction::Replay => {
                let Some(captured) = self.captured.get(&fingerprint) else {
                    self.plan
                        .note_failure("captured launch list has no executable".to_string());
                    return cuda.graph_replay_records(&recorded);
                };
                let handle = captured.handle;
                if let Err(error) = cuda.graph_launch(handle) {
                    // A graph that cannot launch is an optimization failure, never an arithmetic
                    // one: execute the launches the record just captured and stop capturing.
                    self.plan
                        .note_failure(format!("captured graph could not launch: {error}"));
                    let reason = self
                        .plan
                        .disabled
                        .clone()
                        .unwrap_or_else(|| "captured graph could not launch".to_string());
                    self.report_disable(&reason);
                    self.fallbacks += 1;
                    return cuda.graph_replay_records(&recorded);
                }
                self.replays += 1;
                Ok(())
            }
            PlanAction::ExecuteRecorded => {
                self.fallbacks += 1;
                if let Some(reason) = self.plan.disabled.clone() {
                    self.report_disable(&reason);
                }
                cuda.graph_replay_records(&recorded)
            }
            PlanAction::Capture => self.capture_and_replay(cuda, segment, &fingerprint, &recorded),
        }
    }

    /// Captures the segment once and replays it, or disables the seam with the driver's reason.
    fn capture_and_replay<F>(
        &mut self,
        cuda: &Cuda,
        segment: &mut F,
        fingerprint: &LaunchFingerprint,
        recorded: &[CudaLaunchRecord],
    ) -> Result<(), String>
    where
        F: FnMut(&Cuda) -> Result<(), String>,
    {
        if let Err(error) = cuda.graph_capture_begin() {
            self.plan
                .note_failure(format!("capture could not begin: {error}"));
            let reason = self
                .plan
                .disabled
                .clone()
                .unwrap_or_else(|| "capture could not begin".to_string());
            self.report_disable(&reason);
            self.fallbacks += 1;
            return cuda.graph_replay_records(recorded);
        }
        let captured = match segment(cuda) {
            Ok(()) => cuda.graph_capture_end(),
            Err(error) => {
                cuda.graph_capture_abort();
                Err(format!("capture aborted: {error}"))
            }
        };
        match captured {
            Ok((handle, launches)) => {
                let captured_fingerprint = LaunchFingerprint::of(&launches);
                if captured_fingerprint != *fingerprint {
                    // The stretch launched a different list under capture than under record. That
                    // means the closure is not deterministic; replaying either list would be a
                    // guess, so execute what the record saw and stop capturing this seam.
                    cuda.graph_destroy(handle);
                    self.plan.note_failure(
                        "segment launched a different list under capture".to_string(),
                    );
                    let reason = self.plan.disabled.clone().unwrap_or_else(|| {
                        "segment launched a different list under capture".to_string()
                    });
                    self.report_disable(&reason);
                    return cuda.graph_replay_records(recorded);
                }
                if let Err(error) = cuda.graph_launch(handle) {
                    cuda.graph_destroy(handle);
                    self.plan
                        .note_failure(format!("captured graph could not launch: {error}"));
                    let reason = self
                        .plan
                        .disabled
                        .clone()
                        .unwrap_or_else(|| "captured graph could not launch".to_string());
                    self.report_disable(&reason);
                    self.fallbacks += 1;
                    return cuda.graph_replay_records(recorded);
                }
                self.captured_launches = launches.len();
                self.captured
                    .insert(fingerprint.clone(), CapturedGraph { handle });
                self.plan.note_captured(fingerprint.clone());
                self.replays += 1;
                println!(
                    "tp4-graph: seam {:?} captured graph {:?} ({} launches) and replayed it; the launch list is a warm-up artifact from here on",
                    self.name,
                    handle,
                    launches.len()
                );
                Ok(())
            }
            Err(reason) => {
                self.plan.note_failure(reason.clone());
                self.report_disable(&reason);
                cuda.graph_replay_records(recorded)
            }
        }
    }
}

/// The per-name seams, so the decode loop does not have to thread a handle through every layer
/// helper. A segment name is a static string chosen by the call site; one name is one seam.
fn registry() -> &'static Mutex<BTreeMap<&'static str, GraphSeam>> {
    static SEAMS: OnceLock<Mutex<BTreeMap<&'static str, GraphSeam>>> = OnceLock::new();
    SEAMS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Runs one launch-only stretch of the decode path, replaying a captured CUDA graph when possible.
///
/// `segment` may run its closure more than once per call: once under record (to learn the launch
/// list without issuing it) and, on the first occurrence of a list, once under capture. Both runs
/// only queue launches - a record queues none and a capture queues into a graph - so the closure
/// must be launch-only and must not change host state that later code depends on. The driver
/// refuses every non-launch operation inside a segment with a named error; the wrapper reports the
/// refusal, disables the seam and runs the closure once more on the direct path.
///
/// The direct path is always the fallback and always produces the run's normal arithmetic. A
/// replay is allowed only when the recorded launch list is identical to the captured one, so no
/// kernel, argument word, grid or block is ever approximated or patched.
///
/// \# Errors
/// The switch has an unknown value, or the direct path itself fails.
pub(crate) fn segment<F>(cuda: &Cuda, name: &'static str, mut segment: F) -> Result<(), String>
where
    F: FnMut(&Cuda) -> Result<(), String>,
{
    let mode = mode()?;
    if mode == GraphMode::Off {
        return segment(cuda);
    }
    if let Err(reason) = preconditions() {
        static REPORTED: OnceLock<()> = OnceLock::new();
        if REPORTED.set(()).is_ok() {
            println!("tp4-graph: disabled for this run: {reason}");
        }
        return segment(cuda);
    }
    let mut seams = registry()
        .lock()
        .map_err(|error| format!("tp4-graph: seam registry lock: {error}"))?;
    let seam = seams
        .entry(name)
        .or_insert_with(|| GraphSeam::new(name, mode));
    seam.run(cuda, &mut segment)
}

/// A one-line report of every seam this process opened, for a window's end-of-run line.
#[allow(dead_code)] // printed by the run loop when the seam wiring lands; kept next to the seams
pub(crate) fn report() -> String {
    let Ok(seams) = registry().lock() else {
        return "tp4-graph: seam registry lock poisoned".to_string();
    };
    if seams.is_empty() {
        return "tp4-graph: no segments were opened (SPARKFLASH_CUDA_GRAPH=off?)".to_string();
    }
    seams
        .values()
        .map(GraphSeam::summary)
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_parses_every_documented_value_and_refuses_typos() {
        assert_eq!(GraphMode::from_env_value(None), Ok(GraphMode::Off));
        assert_eq!(GraphMode::from_env_value(Some("")), Ok(GraphMode::Off));
        assert_eq!(GraphMode::from_env_value(Some("0")), Ok(GraphMode::Off));
        assert_eq!(GraphMode::from_env_value(Some("off")), Ok(GraphMode::Off));
        assert_eq!(GraphMode::from_env_value(Some("1")), Ok(GraphMode::On));
        assert_eq!(GraphMode::from_env_value(Some("on")), Ok(GraphMode::On));
        assert_eq!(
            GraphMode::from_env_value(Some("capture")),
            Ok(GraphMode::On)
        );
        assert_eq!(
            GraphMode::from_env_value(Some("report")),
            Ok(GraphMode::Report)
        );
        let error = GraphMode::from_env_value(Some("on ")).expect("trailing space trims");
        assert_eq!(error, GraphMode::On);
        let error = GraphMode::from_env_value(Some("ony")).expect_err("typo is refused");
        assert!(error.contains("SPARKFLASH_CUDA_GRAPH"), "{error}");
        assert!(error.contains("off|report|on"), "{error}");
    }

    #[test]
    fn the_first_launch_list_is_captured_then_replayed_exactly() {
        let mut plan = SeamPlan::new(GraphMode::On);
        let list = LaunchFingerprint::from_words(&[1, 2, 3]);
        assert_eq!(plan.decide(&list), PlanAction::Capture);
        plan.note_captured(list.clone());
        assert_eq!(plan.decide(&list), PlanAction::Replay);
    }

    #[test]
    fn a_changed_argument_word_is_a_different_list_and_never_replays() {
        let mut plan = SeamPlan::new(GraphMode::On);
        let first = LaunchFingerprint::from_words(&[1, 0x1000, 3]);
        plan.note_captured(first.clone());
        // The same number of launches, but one device-pointer word moved - exactly what a
        // non-static pooled buffer looks like. It must be captured (or executed) afresh, not
        // replayed against the first graph.
        let moved = LaunchFingerprint::from_words(&[1, 0x2000, 3]);
        assert_ne!(first, moved);
        assert_eq!(plan.decide(&moved), PlanAction::Capture);
        plan.note_captured(moved.clone());
        assert_eq!(plan.decide(&moved), PlanAction::Replay);
        assert_eq!(plan.decide(&first), PlanAction::Replay);
        assert_eq!(plan.captured.len(), 2);
    }

    #[test]
    fn a_refused_capture_disables_the_seam_for_the_whole_run() {
        let mut plan = SeamPlan::new(GraphMode::On);
        let list = LaunchFingerprint::from_words(&[1]);
        assert_eq!(plan.decide(&list), PlanAction::Capture);
        plan.note_failure("segment is not launch-only: copy_dtoh refused".to_string());
        assert_eq!(plan.decide(&list), PlanAction::ExecuteRecorded);
        let other = LaunchFingerprint::from_words(&[2]);
        assert_eq!(plan.decide(&other), PlanAction::ExecuteRecorded);
        assert!(plan.disabled.is_some());
    }

    #[test]
    fn report_mode_captures_once_and_then_runs_the_direct_path() {
        let mut plan = SeamPlan::new(GraphMode::Report);
        let list = LaunchFingerprint::from_words(&[9]);
        assert_eq!(plan.decide(&list), PlanAction::Capture);
        plan.note_captured(list.clone());
        assert_eq!(plan.decide(&list), PlanAction::ExecuteRecorded);
        let other = LaunchFingerprint::from_words(&[10]);
        assert_eq!(plan.decide(&other), PlanAction::Capture);
    }

    #[test]
    fn the_capture_budget_stops_instead_of_instantiating_without_bound() {
        let mut plan = SeamPlan::new(GraphMode::On);
        for index in 0..MAX_CAPTURES_PER_SEAM {
            plan.note_captured(LaunchFingerprint::from_words(&[index as u64]));
        }
        assert_eq!(plan.captured.len(), MAX_CAPTURES_PER_SEAM);
        let one_more = LaunchFingerprint::from_words(&[u64::MAX]);
        assert_eq!(plan.decide(&one_more), PlanAction::ExecuteRecorded);
        assert!(
            plan.disabled
                .as_deref()
                .is_some_and(|reason| reason.contains("budget")),
            "the budget refusal is reported: {:?}",
            plan.disabled
        );
    }

    #[test]
    fn off_mode_never_records_and_never_captures() {
        let mut plan = SeamPlan::new(GraphMode::Off);
        let list = LaunchFingerprint::from_words(&[1, 2]);
        assert_eq!(plan.decide(&list), PlanAction::ExecuteRecorded);
        plan.note_captured(list.clone());
        assert_eq!(plan.decide(&list), PlanAction::ExecuteRecorded);
    }
}
