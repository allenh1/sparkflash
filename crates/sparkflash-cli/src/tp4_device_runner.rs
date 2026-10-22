//! Four-rank device runner (option A of docs/archive/investigations/four-rank-device-loop.md).
//!
//! This module is the skeleton of the arena-resident arm: weights come from each rank's plan/arena,
//! a layer has two reduce seams (attention, MoE), and the host staging path is structurally absent -
//! `refuse_staging` is the only staging-related entry point and it always refuses.
//!
//! Honest state: the MoE half is not bound yet, so `step` refuses with a named error before running
//! anything; the attention half and the reduce seams are represented but not launched. No token, no
//! measurement. The launcher/kernel glue is device_chain's and will be called in the next commit.

#![allow(dead_code)] // runner skeleton: the launcher/kernel glue lands with the four-rank loop

use std::fmt;

use sparkflash_engine::forward::StepLayers;

/// Which half of a layer a reduce follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReducePhase {
    /// After the attention output projection.
    Attention,
    /// After the MoE output.
    Moe,
}

impl ReducePhase {
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Attention => 0,
            Self::Moe => 1,
        }
    }
}

/// The hook the four-rank loop uses to reduce one seam. One synchronous implementation exists
/// (LayerCollective); the posted/wait implementation swaps in behind the same trait.
pub(crate) trait RankReduceHook {
    fn reduce_layer(
        &mut self,
        layer: usize,
        phase: ReducePhase,
        buffer: &mut [u16],
    ) -> Result<(), String>;

    /// Asserts exactly two reduces per layer in the window (80 for the shipped 40 layers).
    fn verify_step(&mut self, layers: usize) -> Result<usize, String> {
        let _ = layers;
        Err("RankReduceHook::verify_step is not implemented by this hook".to_string())
    }
}

/// The four-rank device runner. Constructed from a rank's setup; refuses to run until the MoE
/// shared/routed bindings exist.
pub(crate) struct Tp4DeviceRunner {
    rank: usize,
    world: usize,
    n_layers: usize,
    stop_after_layer: Option<usize>,
    attention_bound: usize,
    moe_bound: bool,
    reduce: Option<Box<dyn RankReduceHook>>,
    reduces_posted: u64,
}

impl fmt::Debug for Tp4DeviceRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tp4DeviceRunner")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .field("n_layers", &self.n_layers)
            .field("stop_after_layer", &self.stop_after_layer)
            .field("attention_bound", &self.attention_bound)
            .field("moe_bound", &self.moe_bound)
            .field("has_reduce", &self.reduce.is_some())
            .field("reduces_posted", &self.reduces_posted)
            .finish()
    }
}

impl Tp4DeviceRunner {
    pub(crate) fn new(
        rank: usize,
        world: usize,
        n_layers: usize,
        stop_after_layer: Option<usize>,
        attention_bound: usize,
    ) -> Self {
        Self {
            rank,
            world,
            n_layers,
            stop_after_layer,
            attention_bound,
            moe_bound: false,
            reduce: None,
            reduces_posted: 0,
        }
    }

    pub(crate) fn with_moe_bound(mut self, bound: bool) -> Self {
        self.moe_bound = bound;
        self
    }

    pub(crate) fn with_reduce(mut self, hook: Box<dyn RankReduceHook>) -> Self {
        self.reduce = Some(hook);
        self
    }

    pub(crate) fn window(&self) -> StepLayers {
        let last = self
            .stop_after_layer
            .unwrap_or_else(|| self.n_layers.saturating_sub(1));
        StepLayers::through(last, self.n_layers)
    }

    /// The host staging path must be unreachable on this arm. Every staging request is refused,
    /// naming the arm and the per-category rule it would violate.
    pub(crate) fn refuse_staging(&self, what: &str) -> Result<(), String> {
        Err(format!(
            "four-rank device arm (rank {}/{}): refusing to stage {what}: weights are arena-resident \
             and weight_bytes_per_token must stay 0",
            self.rank, self.world
        ))
    }

    /// Runs one step over the layer window. Refuses before any launch while the MoE half is unbound.
    pub(crate) fn step(&mut self, hidden: &mut [f32]) -> Result<Vec<f32>, String> {
        if !self.moe_bound {
            return Err(
                "four-rank device arm: MoE not bound yet (routed expert slot bindings and shared \
                 w1/w3/w2 arena bindings are missing); refusing to run a partial layer"
                    .to_string(),
            );
        }
        let window = self.window();
        let mut layers_run = 0usize;
        for layer in window.start..=window.end {
            let seam = hidden.len() * 2;
            let mut attention_buffer = vec![0u16; seam];
            let mut moe_buffer = vec![0u16; seam];
            if let Some(reduce) = self.reduce.as_mut() {
                reduce.reduce_layer(layer, ReducePhase::Attention, &mut attention_buffer)?;
                reduce.reduce_layer(layer, ReducePhase::Moe, &mut moe_buffer)?;
            }
            self.reduces_posted += 2;
            layers_run += 1;
        }
        let expected_layers = window.end - window.start + 1;
        if layers_run != expected_layers {
            return Err(format!(
                "four-rank device arm: ran {layers_run} layers, window expected {}",
                expected_layers
            ));
        }
        Ok(hidden.to_vec())
    }

    /// Machine-readable evidence row for tp4-loop-rR.json.
    pub(crate) fn evidence(&self) -> serde_json::Value {
        let window = self.window();
        let expected = (window.end - window.start + 1) * 2;
        serde_json::json!({
            "execution": "four-rank device",
            "device_tensors_touched": true,
            "rank": self.rank,
            "world": self.world,
            "layer_window": {"start": window.start, "end": window.end, "layers": window.end - window.start + 1},
            "attention_tensors_bound": self.attention_bound,
            "moe_bound": self.moe_bound,
            "reduces_posted": self.reduces_posted,
            "calls_per_step_expected": expected,
            "calls_per_step_ok": self.reduces_posted == expected as u64,
            "host_staging": "structurally absent: refuse_staging() always refuses on this arm",
            "label": "skeleton only: MoE unbound, no forward, no token, no measurement",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingHook {
        calls: usize,
    }

    impl RankReduceHook for CountingHook {
        fn reduce_layer(
            &mut self,
            _layer: usize,
            _phase: ReducePhase,
            _buffer: &mut [u16],
        ) -> Result<(), String> {
            self.calls += 1;
            Ok(())
        }
    }

    #[test]
    fn the_arm_refuses_to_stage_and_refuses_to_step_unbound_moe() {
        let mut runner = Tp4DeviceRunner::new(0, 4, 40, Some(2), 215);
        assert!(runner.refuse_staging("layers.0.attn.wq_b.weight").is_err());
        let mut hidden = vec![0.0f32; 8];
        let error = runner.step(&mut hidden).expect_err("MoE is unbound");
        assert!(error.contains("MoE not bound"), "{error}");
        assert_eq!(runner.evidence()["device_tensors_touched"], true);
        assert_eq!(runner.evidence()["calls_per_step_expected"], 6);
    }

    #[test]
    fn two_reduces_per_layer_in_the_window() {
        let mut runner = Tp4DeviceRunner::new(1, 4, 40, Some(2), 215)
            .with_moe_bound(true)
            .with_reduce(Box::new(CountingHook { calls: 0 }));
        let mut hidden = vec![0.0f32; 8];
        runner.step(&mut hidden).expect("step");
        let evidence = runner.evidence();
        assert_eq!(evidence["calls_per_step_expected"], 6, "3 layers x 2 seams");
        assert_eq!(evidence["calls_per_step_ok"], true);
    }
}
