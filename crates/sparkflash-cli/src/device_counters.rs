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

//! The steady-state invariants of the device-only decoder step, counted instead of observed.
//!
//! The S1 milestone's acceptance shape is a step that crosses PCIe once: the position's final
//! logits row, read back for the numerical gate (D54). Everything else the lean host arm does today
//! (the per-layer observation points, the host reduce machinery, the per-stage boundary copies) must
//! not come back. "No copies" cannot be checked by reading the code once: a debug flag, a
//! fallback arm or a fresh allocation inside a stage silently restores the traffic, and a timed row
//! then reports a number that does not belong to the port. So the loop counts the three crossings it
//! can make and refuses the step when the count is not the declared one:
//!
//! | counter | allowed | why |
//! | --- | --- | --- |
//! | `d2h_copies` | exactly 1 (the final logits) | the one legitimate readback (steady-state invariant 5, `docs/build/ws-whole-layer-device-port-20261022.md`) |
//! | `allocations` | 0 | rule 2: every span is sized at open; a step that allocates has a stage allocating |
//! | `sync_waits` | at most 1 | the step boundary sync; a collective is queued, not waited |
//!
//! The declared host-bridge reduce is the single exception, and it is an exception only because it
//! is declared: a run that keeps the host reduce for the two per-layer collectives pays one readback
//! per collective - `DECODER_LAYERS * COLLECTIVES_PER_LAYER` = 80 copies - and must say so
//! ([`HostBridgeReduce::Declared`]). A run that says nothing is held to the logits-only count, so an
//! undeclared 80-copy step fails and `81` is accepted only with the declaration. The summary line
//! prints which declaration it used, so the row cannot carry a number whose allowance is invisible.
//!
//! The module is deliberately arithmetic-free, device-free and clock-free: its boundaries are CPU
//! tests. The loop wiring is the Lead's edit to `tp4_forward` (the exact call sites are listed in
//! `docs/build/s1-qa-harness.md`); nothing here reads a device, the environment or a timer.

/// Device-to-host copies one steady-state step is allowed with the logits-only arm.
///
/// The final logits row for the position, read back once for the numerical gate (D54). It is the
/// only crossing steady-state invariant 5 names.
pub const D2H_LOGITS_COPIES: u64 = 1;

/// Decoder layers in the released geometry: the S1 milestone runs all of them device-only.
pub const DECODER_LAYERS: u64 = 40;

/// Cross-rank collectives per layer: the attention delta and the routed MoE delta.
///
/// Kept equal to `layer_device::COLLECTIVES_PER_LAYER` by the test
/// `device_counters_host_bridge_allowance_is_the_declared_80`; the two numbers are pinned against
/// each other so a third collective cannot make the bridge allowance stale.
pub const COLLECTIVES_PER_LAYER: u64 = 2;

/// Device-to-host copies a declared host-bridge reduce costs: one readback per collective.
///
/// This is the `+80` the S1 row records when the declared fallback reduce is in use, not a hidden
/// allowance: [`HostBridgeReduce::Declared`] is what admits it.
pub const HOST_BRIDGE_D2H_COPIES: u64 = DECODER_LAYERS * COLLECTIVES_PER_LAYER;

/// Synchronous waits one step may take: the step boundary sync.
///
/// The two collectives are enqueued on the stream and never waited; the only sync is the step
/// boundary that closes the timer. A second sync is the host waiting on device work inside the step,
/// which is the cost the port exists to remove.
pub const MAX_SYNC_WAITS: u64 = 1;

/// Whether the run declared the host-bridge reduce for the two per-layer collectives.
///
/// The declaration is a run-level fact, decided before the loop (the same way the arm is): it is an
/// argument to [`check`] and [`worst`] rather than a field read from the environment, so no test and
/// no row can inherit an allowance by accident.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HostBridgeReduce {
    /// The device reduce is the run's only collective path: one logits readback per step.
    #[default]
    Disabled,
    /// The host-bridge reduce is declared: its 80 per-collective readbacks are documented and
    /// measured, and the row prints the declaration next to the count.
    Declared,
}

impl HostBridgeReduce {
    /// The number of device-to-host copies the step is allowed under this declaration.
    #[must_use]
    pub const fn d2h_copies_allowed(self) -> u64 {
        match self {
            Self::Disabled => D2H_LOGITS_COPIES,
            Self::Declared => D2H_LOGITS_COPIES + HOST_BRIDGE_D2H_COPIES,
        }
    }

    /// Whether the run declared the host-bridge reduce.
    #[must_use]
    pub const fn is_declared(self) -> bool {
        matches!(self, Self::Declared)
    }

    /// The declaration as the row's summary line prints it.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Disabled => "logits only",
            Self::Declared => "final logits + declared host-bridge reduce",
        }
    }
}

/// Which steady-state invariant a step broke, with the count that broke it.
///
/// [`Display`](std::fmt::Display) is the failure text the loop returns and the row prints: it names
/// the invariant, the actual count and the allowed count in one sentence, so a refused step is a
/// diagnosis rather than a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterViolation {
    /// The step allocated device memory. No stage allocates in steady state (rule 2).
    Allocations {
        /// How many allocations the step recorded.
        actual: u64,
    },
    /// The step crossed PCIe more, or fewer, times than the declared arm allows.
    D2hCopies {
        /// The copies the declared arm allows (1, or 81 with the host bridge).
        expected: u64,
        /// How many copies the step recorded.
        actual: u64,
        /// The declaration that produced `expected`.
        bridge: HostBridgeReduce,
    },
    /// The step synchronised the stream more than once.
    SyncWaits {
        /// How many synchronous waits the step recorded.
        actual: u64,
    },
}

impl CounterViolation {
    /// The invariant this violation broke, as a short label for a record or a test assertion.
    #[must_use]
    pub const fn invariant(self) -> &'static str {
        match self {
            Self::Allocations { .. } => "allocations",
            Self::D2hCopies { .. } => "d2h copies",
            Self::SyncWaits { .. } => "sync waits",
        }
    }
}

impl std::fmt::Display for CounterViolation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Allocations { actual } => write!(
                formatter,
                "device step invariant broken: allocations = {actual}, expected 0 \
                 (no stage allocates in steady state; the slab is sized at open)"
            ),
            Self::D2hCopies {
                expected,
                actual,
                bridge,
            } => write!(
                formatter,
                "device step invariant broken: d2h copies = {actual}, expected {expected} ({}) \
                 (the final logits row is the only legitimate readback)",
                bridge.label()
            ),
            Self::SyncWaits { actual } => write!(
                formatter,
                "device step invariant broken: sync waits = {actual}, expected at most \
                 {MAX_SYNC_WAITS} (the step boundary sync; a collective is queued, not waited)"
            ),
        }
    }
}

impl std::error::Error for CounterViolation {}

/// The worst-broken steady-state invariant of a step, or `None` when the step conforms.
///
/// "Worst" is the reporting order, documented so a step that breaks more than one thing always
/// produces the same sentence: an allocation first (rule 2, the strongest steady-state rule), then a
/// d2h-copy mismatch (the traffic the port exists to remove), then a second synchronous wait. Any
/// one of them fails the step; the order only decides which is named when several are broken.
#[must_use]
pub fn worst(
    d2h_copies: u64,
    allocations: u64,
    sync_waits: u64,
    bridge: HostBridgeReduce,
) -> Option<CounterViolation> {
    let expected = bridge.d2h_copies_allowed();
    if allocations != 0 {
        return Some(CounterViolation::Allocations {
            actual: allocations,
        });
    }
    if d2h_copies != expected {
        return Some(CounterViolation::D2hCopies {
            expected,
            actual: d2h_copies,
            bridge,
        });
    }
    if sync_waits > MAX_SYNC_WAITS {
        return Some(CounterViolation::SyncWaits { actual: sync_waits });
    }
    None
}

/// Checks a step's three counts against the declared invariants.
///
/// The d2h allowance is read from the declaration: `d2h == 1` without it, `d2h == 81` with the
/// declared host-bridge reduce. Allocations must be exactly zero and synchronous waits at most one.
///
/// # Errors
/// Returns the worst-broken invariant ([`worst`]'s order) with the count that broke it.
pub fn check(
    d2h_copies: u64,
    allocations: u64,
    sync_waits: u64,
    bridge: HostBridgeReduce,
) -> Result<(), CounterViolation> {
    match worst(d2h_copies, allocations, sync_waits, bridge) {
        Some(violation) => Err(violation),
        None => Ok(()),
    }
}

/// The per-step recorder the device arm's steady-state loop carries.
///
/// The loop resets it at the start of every step, increments the three counters where the step
/// crosses a boundary, and checks it before the row prints. [`reset`](Self::reset) advances the
/// step number, so a summary line says which step failed as well as why.
///
/// The struct has no defaulting of the host-bridge declaration on purpose: [`check`] and
/// [`summary`](Self::summary) take [`HostBridgeReduce`] explicitly, so the declaration is a call-site
/// fact the loop states once and the counters cannot acquire.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceStepCounters {
    step: u64,
    d2h_copies: u64,
    sync_waits: u64,
    allocations: u64,
}

impl DeviceStepCounters {
    /// A fresh recorder at step 0 with no counts.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            step: 0,
            d2h_copies: 0,
            sync_waits: 0,
            allocations: 0,
        }
    }

    /// Starts the next step: the step number advances and every per-step count returns to zero.
    pub fn reset(&mut self) {
        self.step = self.step.wrapping_add(1);
        self.d2h_copies = 0;
        self.sync_waits = 0;
        self.allocations = 0;
    }

    /// The step this recorder is counting (0 before the first [`reset`](Self::reset)).
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Device-to-host copies recorded in this step.
    #[must_use]
    pub const fn d2h_copies(&self) -> u64 {
        self.d2h_copies
    }

    /// Device allocations recorded in this step.
    #[must_use]
    pub const fn allocations(&self) -> u64 {
        self.allocations
    }

    /// Synchronous waits recorded in this step.
    #[must_use]
    pub const fn sync_waits(&self) -> u64 {
        self.sync_waits
    }

    /// Records the final logits readback. The loop calls this at the one `copy_dtoh` that is allowed.
    pub fn record_d2h_copy(&mut self) {
        self.d2h_copies = self.d2h_copies.wrapping_add(1);
    }

    /// Records a device allocation. In a conforming step this is never called; it exists so a
    /// step that does allocate fails `check` instead of passing silently.
    pub fn record_allocation(&mut self) {
        self.allocations = self.allocations.wrapping_add(1);
    }

    /// Records a synchronous wait (a stream synchronise inside the step).
    pub fn record_sync_wait(&mut self) {
        self.sync_waits = self.sync_waits.wrapping_add(1);
    }

    /// The worst-broken invariant of this recorder's counts, or `None` when the step conforms.
    #[must_use]
    pub fn worst(&self, bridge: HostBridgeReduce) -> Option<CounterViolation> {
        worst(self.d2h_copies, self.allocations, self.sync_waits, bridge)
    }

    /// Checks this recorder's counts against the declared invariants.
    ///
    /// # Errors
    /// Returns the worst-broken invariant with the count that broke it.
    pub fn check(&self, bridge: HostBridgeReduce) -> Result<(), CounterViolation> {
        check(self.d2h_copies, self.allocations, self.sync_waits, bridge)
    }

    /// The one line the S1 row prints: the three counts, the declaration and PASS/FAIL.
    ///
    /// Example: `device step 7: d2h=1 (allowed 1, logits only) sync=1 (allowed <= 1) alloc=0
    /// (allowed 0) PASS`. A failing step appends the [`CounterViolation`] text.
    #[must_use]
    pub fn summary(&self, bridge: HostBridgeReduce) -> String {
        let allowed = bridge.d2h_copies_allowed();
        let verdict = match self.check(bridge) {
            Ok(()) => "PASS".to_owned(),
            Err(violation) => format!("FAIL {violation}"),
        };
        format!(
            "device step {}: d2h={} (allowed {}, {}) sync={} (allowed <= {}) alloc={} \
             (allowed 0) {verdict}",
            self.step,
            self.d2h_copies,
            allowed,
            bridge.label(),
            self.sync_waits,
            MAX_SYNC_WAITS,
            self.allocations,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        COLLECTIVES_PER_LAYER, CounterViolation, D2H_LOGITS_COPIES, DECODER_LAYERS,
        DeviceStepCounters, HOST_BRIDGE_D2H_COPIES, HostBridgeReduce, MAX_SYNC_WAITS, check, worst,
    };

    /// A recorder holding the three counts, as the loop would have recorded them.
    fn recorded(d2h_copies: u64, allocations: u64, sync_waits: u64) -> DeviceStepCounters {
        let mut counters = DeviceStepCounters::new();
        for _ in 0..d2h_copies {
            counters.record_d2h_copy();
        }
        for _ in 0..allocations {
            counters.record_allocation();
        }
        for _ in 0..sync_waits {
            counters.record_sync_wait();
        }
        counters
    }

    #[test]
    fn device_counters_constants_pin_the_declared_allowance() {
        assert_eq!(D2H_LOGITS_COPIES, 1);
        assert_eq!(DECODER_LAYERS, 40);
        assert_eq!(COLLECTIVES_PER_LAYER, 2);
        assert_eq!(HOST_BRIDGE_D2H_COPIES, 80);
        assert_eq!(MAX_SYNC_WAITS, 1);
        assert_eq!(HostBridgeReduce::Disabled.d2h_copies_allowed(), 1);
        assert_eq!(HostBridgeReduce::Declared.d2h_copies_allowed(), 81);
        assert!(!HostBridgeReduce::Disabled.is_declared());
        assert!(HostBridgeReduce::Declared.is_declared());
        assert_eq!(HostBridgeReduce::default(), HostBridgeReduce::Disabled);
    }

    #[test]
    fn device_counters_host_bridge_allowance_is_the_declared_80() {
        // The +80 is `layers * collectives`, so a third collective in layer_device cannot leave this
        // module's allowance stale without failing here.
        assert_eq!(
            HOST_BRIDGE_D2H_COPIES,
            DECODER_LAYERS * COLLECTIVES_PER_LAYER
        );
        assert_eq!(
            COLLECTIVES_PER_LAYER,
            crate::layer_device::COLLECTIVES_PER_LAYER as u64
        );
    }

    #[test]
    fn device_counters_d2h_boundaries_without_a_declared_host_bridge() {
        let bridge = HostBridgeReduce::Disabled;
        // 0: the logits copy is missing.
        assert_eq!(
            check(0, 0, 1, bridge),
            Err(CounterViolation::D2hCopies {
                expected: 1,
                actual: 0,
                bridge,
            })
        );
        // 1: the only conforming count without the bridge.
        assert_eq!(check(1, 0, 1, bridge), Ok(()));
        // 80: the host bridge's count without its declaration.
        assert_eq!(
            check(80, 0, 1, bridge),
            Err(CounterViolation::D2hCopies {
                expected: 1,
                actual: 80,
                bridge,
            })
        );
        // 81: one logits copy plus 80 undeclared bridge copies.
        assert_eq!(
            check(81, 0, 1, bridge),
            Err(CounterViolation::D2hCopies {
                expected: 1,
                actual: 81,
                bridge,
            })
        );
    }

    #[test]
    fn device_counters_d2h_boundaries_with_the_declared_host_bridge() {
        let bridge = HostBridgeReduce::Declared;
        // 0 and 1: a declared bridge expects its 80 copies, so the logits-only counts are short.
        for actual in [0u64, 1] {
            assert_eq!(
                check(actual, 0, 1, bridge),
                Err(CounterViolation::D2hCopies {
                    expected: 81,
                    actual,
                    bridge,
                })
            );
        }
        // 80: the bridge copies without the logits copy.
        assert_eq!(
            check(80, 0, 1, bridge),
            Err(CounterViolation::D2hCopies {
                expected: 81,
                actual: 80,
                bridge,
            })
        );
        // 81: the declaration's exact count.
        assert_eq!(check(81, 0, 1, bridge), Ok(()));
        // 82: one copy too many is still a broken invariant.
        assert!(matches!(
            check(82, 0, 1, bridge),
            Err(CounterViolation::D2hCopies { expected: 81, .. })
        ));
    }

    #[test]
    fn device_counters_allocation_boundaries() {
        // 0 allocations: conforming.
        assert_eq!(check(1, 0, 1, HostBridgeReduce::Disabled), Ok(()));
        // 1 allocation: rule 2 broken, named with its count.
        assert_eq!(
            check(1, 1, 1, HostBridgeReduce::Disabled),
            Err(CounterViolation::Allocations { actual: 1 })
        );
        assert!(matches!(
            check(1, 2, 1, HostBridgeReduce::Disabled),
            Err(CounterViolation::Allocations { actual: 2 })
        ));
    }

    #[test]
    fn device_counters_sync_boundaries() {
        // 1 sync: the step boundary, allowed.
        assert_eq!(check(1, 0, 1, HostBridgeReduce::Disabled), Ok(()));
        // 0 syncs: the contract is "at most one", so zero is conforming too.
        assert_eq!(check(1, 0, 0, HostBridgeReduce::Disabled), Ok(()));
        // 2 syncs: one more than the boundary.
        assert_eq!(
            check(1, 0, 2, HostBridgeReduce::Disabled),
            Err(CounterViolation::SyncWaits { actual: 2 })
        );
        assert!(matches!(
            check(1, 0, 3, HostBridgeReduce::Disabled),
            Err(CounterViolation::SyncWaits { actual: 3 })
        ));
    }

    #[test]
    fn device_counters_worst_names_the_strongest_broken_invariant() {
        let bridge = HostBridgeReduce::Disabled;
        // Allocation outranks a d2h mismatch and an extra sync.
        assert_eq!(
            worst(0, 1, 2, bridge),
            Some(CounterViolation::Allocations { actual: 1 })
        );
        // A d2h mismatch outranks an extra sync.
        assert_eq!(
            worst(0, 0, 2, bridge),
            Some(CounterViolation::D2hCopies {
                expected: 1,
                actual: 0,
                bridge,
            })
        );
        // The sync is reported when it is the only violation.
        assert_eq!(
            worst(1, 0, 2, bridge),
            Some(CounterViolation::SyncWaits { actual: 2 })
        );
        assert_eq!(worst(1, 0, 1, bridge), None);
    }

    #[test]
    fn device_counters_failure_text_names_the_broken_invariant() {
        let allocation = CounterViolation::Allocations { actual: 1 }.to_string();
        assert!(allocation.contains("allocations = 1"), "{allocation}");
        assert!(allocation.contains("expected 0"), "{allocation}");

        let d2h = CounterViolation::D2hCopies {
            expected: 1,
            actual: 0,
            bridge: HostBridgeReduce::Disabled,
        }
        .to_string();
        assert!(d2h.contains("d2h copies = 0"), "{d2h}");
        assert!(d2h.contains("expected 1"), "{d2h}");
        assert!(d2h.contains("logits only"), "{d2h}");

        let sync = CounterViolation::SyncWaits { actual: 2 }.to_string();
        assert!(sync.contains("sync waits = 2"), "{sync}");
        assert!(sync.contains("at most 1"), "{sync}");

        // The short label is stable for records and assertions.
        assert_eq!(
            CounterViolation::Allocations { actual: 1 }.invariant(),
            "allocations"
        );
        assert_eq!(
            CounterViolation::D2hCopies {
                expected: 1,
                actual: 0,
                bridge: HostBridgeReduce::Disabled,
            }
            .invariant(),
            "d2h copies"
        );
        assert_eq!(
            CounterViolation::SyncWaits { actual: 2 }.invariant(),
            "sync waits"
        );
    }

    #[test]
    fn device_counters_default_recorder_fails_until_the_logits_copy_is_recorded() {
        // A fresh recorder has produced a step with no logits readback: that is a broken step, not
        // a passing zero, which is what stops a loop that forgot the call from reporting PASS.
        let counters = DeviceStepCounters::new();
        assert_eq!(counters.step(), 0);
        assert_eq!(counters.d2h_copies(), 0);
        assert_eq!(counters.allocations(), 0);
        assert_eq!(counters.sync_waits(), 0);
        assert!(counters.check(HostBridgeReduce::Disabled).is_err());
    }

    #[test]
    fn device_counters_record_and_reset_are_per_step() {
        let mut counters = DeviceStepCounters::new();
        counters.reset();
        assert_eq!(counters.step(), 1);
        counters.record_d2h_copy();
        counters.record_sync_wait();
        assert_eq!(counters.check(HostBridgeReduce::Disabled), Ok(()));

        // The next step starts empty: nothing from step 1 is carried over.
        counters.reset();
        assert_eq!(counters.step(), 2);
        assert_eq!(counters.d2h_copies(), 0);
        assert_eq!(counters.sync_waits(), 0);
        assert_eq!(counters.allocations(), 0);
        assert!(counters.check(HostBridgeReduce::Disabled).is_err());

        counters.record_d2h_copy();
        counters.record_sync_wait();
        assert_eq!(counters.check(HostBridgeReduce::Disabled), Ok(()));
    }

    #[test]
    fn device_counters_summary_prints_the_declaration_and_verdict() {
        let mut counters = recorded(1, 0, 1);
        counters.reset();
        counters.record_d2h_copy();
        counters.record_sync_wait();
        let summary = counters.summary(HostBridgeReduce::Disabled);
        assert!(summary.starts_with("device step 1:"), "{summary}");
        assert!(
            summary.contains("d2h=1 (allowed 1, logits only)"),
            "{summary}"
        );
        assert!(summary.contains("sync=1 (allowed <= 1)"), "{summary}");
        assert!(summary.contains("alloc=0 (allowed 0)"), "{summary}");
        assert!(summary.ends_with("PASS"), "{summary}");

        // The declared host bridge prints its 81-copy allowance and PASSes at 81.
        let mut bridged = recorded(81, 0, 1);
        bridged.reset();
        for _ in 0..81 {
            bridged.record_d2h_copy();
        }
        bridged.record_sync_wait();
        let summary = bridged.summary(HostBridgeReduce::Declared);
        assert!(
            summary.contains("d2h=81 (allowed 81, final logits + declared host-bridge reduce)"),
            "{summary}"
        );
        assert!(summary.ends_with("PASS"), "{summary}");

        // A failing step appends the violation text, not just FAIL.
        let broken = recorded(1, 1, 1);
        let summary = broken.summary(HostBridgeReduce::Disabled);
        assert!(summary.contains("FAIL"), "{summary}");
        assert!(summary.contains("allocations = 1"), "{summary}");
    }
}
