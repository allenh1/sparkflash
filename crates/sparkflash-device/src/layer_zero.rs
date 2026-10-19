//! M3-c c1/c2: one layer-0 host pass through the chain's stage order.
//!
//! The arithmetic itself lives in [sparkflash_engine::forward::layer_zero]; this module turns its
//! stage trace into the chain's stable seam ledger and records the execution classification for
//! every active stage. The pass is one layer and one position, not a token: the tail and the 39
//! remaining layers do not run here.
//!
//! # Execution classification
//!
//! * `embedding`, `dispatch`, `attn_norm`, `attn_rope`, `window_kv`, `moe_routing_weight`,
//!   `moe_swiglu_quant`, `shared_expert` and `ffn_epilogue` are [LayerZeroExecution::Host] on
//!   the no-GPU path: their arithmetic ran on the host and their seam records carry the reference
//!   tensors' bytes.
//! * In the window hybrid, `attn_norm` is [LayerZeroExecution::DeviceFused], not Host: the
//!   normalised attention input arrives inside the device MhcEntry capture, so there is no separate
//!   host seam and the plan drops it rather than recording a zero-byte host seam.
//! * `attention_projections`, `attention`, `attention_output_projection`, `mhc_post`,
//!   `moe_router`, `moe_gate_up`, `moe_down_combine` and `shared_expert_gate` are
//!   [LayerZeroExecution::HostStandIn] in this MR. The M3-a stage map names device kernels for
//!   them, but no typed builder/readback is wired for the rest of layer 0 yet, so this pass runs
//!   the existing reference arithmetic and reports them as host arithmetic rather than pretending
//!   a device stage ran.
//! * `mhc_entry` is [LayerZeroExecution::HostStandIn] on the default no-GPU path and becomes
//!   [LayerZeroExecution::Device] only in [run_layer_zero_device_entry], after the staged launch
//!   returns and its readback is consumed; that function holds the private one-shot
//!   `DeviceEntryReceipt` and no host-only caller can manufacture one.
//!   [sparkflash_engine::forward::layer_zero::LayerZeroEntry] is the handoff type.
//!
//! The dry-run/test path therefore has no driver, no cubin and no allocation. A window run can
//! replace the `mhc_entry` stand-in with the staged device launch and the readback, but it must
//! keep this classification honest in its report.

use sparkflash_engine::config::ModelConfig;
use sparkflash_engine::forward::boundary::LayerBoundary;
use sparkflash_engine::forward::layer_zero::{LAYER_ZERO_HOST_STAGES, LayerZeroTrace, StageEvent};
use sparkflash_engine::weights::WeightSource;
use sparkflash_kernels::sequence_chain::{SeamLedger, SeamRecord};
use sparkflash_kv::config::{Csa2Mode, SWA_WINDOW_SIZE};

use crate::chain::{ChainLayer, ChainSchedule, ChainSeamPlan, ChainStage, LAYER_STAGE_ORDER};

/// How a layer-0 stage ran in this pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerZeroExecution {
    /// The host arithmetic ran and produced the ledger's host bytes.
    Host,
    /// The reference host arithmetic ran in place of a device stage whose builder is not wired.
    HostStandIn,
    /// The device stage really ran and its readback fed the host continuation.
    Device,
    /// The arithmetic ran inside a device stage's capture and has no separate host seam.
    DeviceFused,
}

impl LayerZeroExecution {
    /// The report word for this execution class.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::HostStandIn => "host stand-in (device builder not wired)",
            Self::Device => "device",
            Self::DeviceFused => "device-fused (no separate host seam)",
        }
    }
}

/// One active stage's execution and seam record.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerZeroStageRun {
    /// The chain stage.
    pub stage: ChainStage,
    /// How it ran in this pass.
    pub execution: LayerZeroExecution,
    /// Its host seam record; None for a stage with no host-seam plan entry.
    pub seam: Option<SeamRecord>,
    /// The reference arithmetic's stage event.
    pub event: StageEvent,
}

/// The c1 result: the layer-0 trace, the seam ledger, and the explicit stage classification.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerZeroRun {
    /// The one-layer, one-position schedule this pass uses.
    pub schedule: ChainSchedule,
    /// The token embedded at position 0.
    pub token: u32,
    /// The host trace.
    pub trace: LayerZeroTrace,
    /// The seam plan for this one-layer schedule.
    pub seam_plan: ChainSeamPlan,
    /// The seams the pass recorded.
    pub ledger: SeamLedger,
    /// The active stages, in layer order, with their classification.
    pub stages: Vec<LayerZeroStageRun>,
}

/// Runs one layer-0 host pass and builds its seam ledger.
///
/// This is the no-GPU c1 path. It calls the reference host arithmetic, then records the stage
/// execution classes above and validates its seam ledger against the one-layer plan.
///
/// # Errors
/// A reference arithmetic error, or a trace that does not match the layer-0 stage plan.
pub fn run_layer_zero_host(
    config: &ModelConfig,
    source: &dyn WeightSource,
    token: u32,
    limit: usize,
) -> Result<LayerZeroRun, String> {
    let trace =
        sparkflash_engine::forward::layer_zero::run_layer_zero(config, source, token, limit, None)
            .map_err(|error| format!("layer-0 host pass: {error}"))?;
    LayerZeroRun::from_trace(config, trace)
}

/// Runs one layer-0 pass whose MhcEntry comes from a real device launch.
///
/// The stage buffers must already be staged by [crate::chain_device::CudaLauncher::stage_layer_zero]
/// and the mHC cubin must be loaded. The launch writes the embedding row into X, runs the pinned
/// mHC kernel, reads back the f32 normalised capture and its B/A/C coefficients, and hands them to
/// [sparkflash_engine::forward::layer_zero::run_layer_zero_with_embedding]. `attn_norm` is
/// [LayerZeroExecution::DeviceFused] because it is inside the capture; the other device-status
/// layer-0 stages stay on the host reference arithmetic, exactly as
/// [LayerZeroExecution::HostStandIn] declares.
///
/// This is the first hybrid substitution, not a device chain: only one device stage runs and the
/// report must keep saying so. It is window-only because it needs the driver and the pinned cubins;
/// the default test build never calls it.
///
/// # Errors
/// Staging, launch or readback failure, or a reference arithmetic failure.
#[cfg(feature = "cuda")]
pub fn run_layer_zero_device_entry(
    launcher: &mut crate::chain_device::CudaLauncher,
    row: &crate::staging::EmbedRow,
    config: &ModelConfig,
    source: &dyn WeightSource,
    token: u32,
    limit: usize,
) -> Result<
    (
        LayerZeroRun,
        sparkflash_engine::forward::layer_zero::LayerZeroEntry,
        LayerZeroDeviceEvidence,
    ),
    String,
> {
    use crate::chain::ChainStage;
    use crate::chain_device::{DeviceLauncher, DeviceStageCall};

    if config.hc_mult != 4 {
        return Err(format!(
            "the pinned mHC kernel is the released four-stream kernel, but the config declares hc_mult {}",
            config.hc_mult
        ));
    }
    let expected_x = row.x_bf16();
    let embedding = row.values_f32();
    let width = config.dim;
    let mut x_prev = vec![0.0f32; width * config.hc_mult];
    for copy in 0..config.hc_mult {
        x_prev[copy * config.dim..(copy + 1) * config.dim].copy_from_slice(&embedding);
    }
    let (b_prev, c_prev, _a_prev, y_prev) = crate::staging::LayerZeroStaging::layer_zero_seed();
    let expected_update = sparkflash_engine::forward::layer_zero::kernel_mhc_update(
        &x_prev, &b_prev, &c_prev, &y_prev,
    );
    let expected_update_bf16 = sparkflash_engine::forward::layer_zero::bf16_bytes(&expected_update);
    launcher.write_layer_zero_x(row)?;
    // Read X back BEFORE the launch: the kernel updates X in place to X_l, so after the launch the
    // buffer no longer holds the seed. This check proves the device row is the admitted row the
    // host trace widened, not a stale or zero-filled buffer.
    let copied_x = launcher.read_layer_zero_x()?;
    let x_matches = copied_x == expected_x;
    if !x_matches {
        return Err(format!(
            "the device X readback ({} bytes) differs from the admitted four-copy embedding row ({} bytes) before launch",
            copied_x.len(),
            expected_x.len()
        ));
    }
    DeviceLauncher::launch(
        launcher,
        DeviceStageCall {
            stage: ChainStage::MhcEntry,
            layer: 0,
            position: 0,
            kernel: "mega_mhc_sm121",
        },
    )
    .map_err(|error| format!("layer-0 device MhcEntry launch: {error}"))?;
    // The kernel writes X_l back in place. This compares the BF16 write with the host twin's
    // step-1 update built with the kernel's FMA order (C*Y as one mul, then one mul_add per B
    // term): at the released layer-0 seed that update is the identity, so the written stream is
    // still the admitted row. m11 b7 measured X_l bit-identical; this is not a new bar.
    let written_x = launcher.read_layer_zero_x()?;
    let x_after_launch_matches = written_x == expected_update_bf16;
    let x_after_launch_first_difference = if written_x.len() == expected_update_bf16.len() {
        written_x
            .as_chunks::<2>()
            .0
            .iter()
            .zip(expected_update_bf16.as_chunks::<2>().0)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
            .map(|(index, (actual, expected))| {
                (
                    index,
                    u16::from_le_bytes(*actual),
                    u16::from_le_bytes(*expected),
                )
            })
    } else {
        None
    };
    let entry = launcher.read_layer_zero_entry()?;
    let inv_rms = launcher.read_layer_zero_inv_rms()?;
    let trace = sparkflash_engine::forward::layer_zero::run_layer_zero_with_embedding(
        config,
        source,
        token,
        &embedding,
        limit,
        Some(entry.clone()),
    )
    .map_err(|error| format!("layer-0 pass after the device entry: {error}"))?;
    let mut run = LayerZeroRun::from_trace_inner(config, trace, true)?;
    let mut receipt = DeviceEntryReceipt::launched();
    run.record_device_stage(
        &mut receipt,
        "device MhcEntry launch + f32 xhat_normed/B/A/C readback; the rest of layer 0 is host reference arithmetic",
    )?;
    Ok((
        run,
        entry,
        LayerZeroDeviceEvidence {
            x_matches,
            x_after_launch_matches,
            x_after_launch_first_difference,
            x_bytes: copied_x.len(),
            inv_rms,
        },
    ))
}

/// Evidence the device arm gathers before and after the one launch.
///
/// The X comparison is taken after the copy and before the launch because the kernel updates X in
/// place; `inv_rms` is the kernel's own `1/rms(X_hat)` scalar. The window prints both so a failed
/// row copy or a zero-fill cannot be hidden behind the coefficient bars.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerZeroDeviceEvidence {
    /// The X readback before launch equals the admitted four-copy BF16 row.
    pub x_matches: bool,
    /// The X_l readback after launch equals the BF16 write of the FMA host update.
    pub x_after_launch_matches: bool,
    /// First differing BF16 element of the X_l readback: index, actual bits, expected bits.
    pub x_after_launch_first_difference: Option<(usize, u16, u16)>,
    /// The number of bytes read back from X.
    pub x_bytes: usize,
    /// The kernel's `1/rms(X_hat)` output.
    pub inv_rms: f32,
}

/// Proof that a real MhcEntry launch returned and its readback was consumed.
///
/// The type is deliberately hard to construct: the fields are private and `launched` is private to
/// this module. [run_layer_zero_device_entry] is the only production caller that creates a
/// launched receipt, and `LayerZeroRun::record_device_stage` is the only consumer.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug)]
pub(crate) struct DeviceEntryReceipt {
    launched: bool,
    consumed: bool,
}

#[cfg(any(feature = "cuda", test))]
impl DeviceEntryReceipt {
    /// The receipt [run_layer_zero_device_entry] creates after the kernel and readback succeed.
    #[cfg(any(feature = "cuda", test))]
    fn launched() -> Self {
        Self {
            launched: true,
            consumed: false,
        }
    }

    /// A deliberately receipt-less stand-in for the host-only refusal test.
    #[cfg(test)]
    fn host_only() -> Self {
        Self {
            launched: false,
            consumed: false,
        }
    }
}

impl LayerZeroRun {
    /// Builds the ledger/classification from an already-computed trace.
    ///
    /// # Errors
    /// A missing stage event or a ledger/plan mismatch.
    pub fn from_trace(config: &ModelConfig, trace: LayerZeroTrace) -> Result<Self, String> {
        Self::from_trace_inner(config, trace, false)
    }

    /// Builds the ledger/classification, knowing whether the entry came from a device launch.
    ///
    /// `entry_was_device` makes `attn_norm` a [LayerZeroExecution::DeviceFused] stage instead of a
    /// host seam: the device mHC capture already holds the normalised attention input, so the host
    /// must not claim it ran the norm. The stage is not emitted into the seam ledger.
    fn from_trace_inner(
        config: &ModelConfig,
        trace: LayerZeroTrace,
        entry_was_device: bool,
    ) -> Result<Self, String> {
        let schedule = one_layer_schedule(config, trace.token)?;
        let found = trace
            .stages
            .iter()
            .map(|event| event.stage)
            .collect::<Vec<_>>();
        if found.as_slice() != LAYER_ZERO_HOST_STAGES.as_slice() {
            return Err(format!(
                "layer-0 trace stages are {found:?}; expected {LAYER_ZERO_HOST_STAGES:?}"
            ));
        }
        let mut seam_plan = ChainSeamPlan::for_schedule(&schedule);
        // This pass stops after layer 0; the tail's final norm and greedy argmax are not part of it.
        seam_plan
            .entries
            .retain(|entry| !matches!(entry.name, "final_norm" | "lm_head_argmax"));
        if entry_was_device {
            // The device mHC capture already contains the normalised attention input; there is no
            // host attn_norm seam to admit or account for in the hybrid pass.
            seam_plan.entries.retain(|entry| entry.name != "attn_norm");
        }
        let mut ledger = SeamLedger::default();
        let mut stages = Vec::with_capacity(LAYER_STAGE_ORDER.len() + 1);
        stages.push(LayerZeroStageRun {
            stage: ChainStage::Embedding,
            execution: LayerZeroExecution::Host,
            seam: Some(SeamRecord {
                name: "embedding_gather",
                kind: sparkflash_kernels::sequence_chain::SeamKind::Other,
                host_bytes_in: 0,
                host_bytes_out: (trace.embedding.len() as u64).saturating_mul(4),
                transfer_bytes: 0,
                phase3_followup: true,
                note: "reference embedding row gathered on the host for the layer-0 pass",
            }),
            event: StageEvent {
                stage: "embedding",
                host_bytes_in: 0,
                host_bytes_out: (trace.embedding.len() as u64).saturating_mul(4),
                note: "reference embedding row gathered on the host",
            },
        });
        let embedding_seam = stages[0].seam.as_ref().expect("embedding seam").clone();
        ledger.record(embedding_seam);
        for stage in LAYER_STAGE_ORDER {
            if !stage.runs_at(&schedule.layers[0]) {
                continue;
            }
            let event = trace
                .stages
                .iter()
                .find(|event| event.stage == stage.name())
                .cloned()
                .ok_or_else(|| format!("layer-0 trace has no event for stage {}", stage.name()))?;
            if entry_was_device && *stage == ChainStage::AttnNorm {
                stages.push(LayerZeroStageRun {
                    stage: *stage,
                    execution: LayerZeroExecution::DeviceFused,
                    seam: None,
                    event: StageEvent {
                        note: "attn_norm is part of the device MhcEntry capture; no separate host seam exists",
                        ..event
                    },
                });
                continue;
            }
            match stage.status() {
                crate::chain::StageStatus::HostSeam(seam) => {
                    let entry = seam_plan.entry(seam).ok_or_else(|| {
                        format!("seam {seam} is not planned for the one-layer schedule")
                    })?;
                    let record = SeamRecord {
                        name: entry.name,
                        kind: entry.kind,
                        host_bytes_in: event.host_bytes_in,
                        host_bytes_out: event.host_bytes_out,
                        transfer_bytes: 0,
                        phase3_followup: !entry.removal.is_empty(),
                        note: event.note,
                    };
                    ledger.record(record.clone());
                    stages.push(LayerZeroStageRun {
                        stage: *stage,
                        execution: LayerZeroExecution::Host,
                        seam: Some(record),
                        event,
                    });
                }
                crate::chain::StageStatus::DeviceKernel(kernel)
                | crate::chain::StageStatus::DeviceMeasuredNotWired { kernel, .. } => {
                    stages.push(LayerZeroStageRun {
                        stage: *stage,
                        execution: LayerZeroExecution::HostStandIn,
                        seam: None,
                        event: StageEvent {
                            note: match kernel {
                                "mega_mhc_sm121" => {
                                    "host reference mHC arithmetic; the device MhcEntry handoff is the window-only substitution"
                                }
                                "flash_mla_csa2_sm121" => {
                                    "host reference attention arithmetic; the device flash-MLA stage is not wired into this pass"
                                }
                                "b12x_mxfp8_dense_sm121" => {
                                    "host reference dense-projection arithmetic; the device mxfp8-dense stage is not wired into this pass"
                                }
                                "b12x_mxfp8_wo_a_sm121" => {
                                    "host reference output-projection arithmetic; the device wo_a stage is not wired into this pass"
                                }
                                "b12x_mxfp4_moe_sm121" => {
                                    "host reference MoE down/combine arithmetic; the device MXFP4-MoE stage is not wired into this pass"
                                }
                                _ => "host reference arithmetic; the named device stage is not wired into this pass",
                            },
                            ..event
                        },
                    });
                }
            }
        }
        seam_plan
            .validate(&ledger)
            .map_err(|error| format!("layer-0 seam ledger: {error}"))?;
        Ok(Self {
            schedule,
            token: trace.token,
            trace,
            seam_plan,
            ledger,
            stages,
        })
    }

    /// The report lines: execution flags, the stage classification, the seam ledger and the
    /// comparison class. They never claim a token or a chain execution; a device execution is
    /// claimed only when `record_device_stage` marked the stage after its launch/readback.
    #[must_use]
    pub fn report_lines(&self) -> Vec<String> {
        let device = self.stages.iter().any(|run| {
            matches!(
                run.execution,
                LayerZeroExecution::Device | LayerZeroExecution::DeviceFused
            )
        });
        let mut lines = vec![
            "sparkflash device-chain: M3-c LAYER-0 PASS - one layer, one position, no token"
                .to_string(),
            format!(
                "token {} at position 0; flags: chain_execution=false; device_execution={device}; token_produced=false",
                self.token
            ),
            "stage execution:".to_string(),
        ];
        for run in &self.stages {
            let seam = run
                .seam
                .as_ref()
                .map(|record| {
                    format!(
                        " seam {} host in {} B, out {} B",
                        record.name, record.host_bytes_in, record.host_bytes_out
                    )
                })
                .unwrap_or_default();
            lines.push(format!(
                "  {}: {}{} [{}]",
                run.stage.name(),
                run.execution.label(),
                seam,
                run.event.note
            ));
        }
        lines.extend(self.ledger.report_lines());
        if device {
            lines.push(
                "execution classification: the stages marked device ran on the device; every host stand-in is still reference host arithmetic and is NOT a device stage"
                    .to_string(),
            );
        } else {
            lines.push(
                "execution classification: host-executed stages are the reference arithmetic; the device-status stages marked host stand-in did NOT execute on a device in this pass"
                    .to_string(),
            );
        }
        lines
    }

    /// Marks the MhcEntry stage device-executed after a successful launch and readback.
    ///
    /// The launch receipt is the authority: only [DeviceEntryReceipt::launched], private to this
    /// module and called only by [run_layer_zero_device_entry] after the kernel returns, can mark a
    /// stage. A host-only run has no such receipt, so it cannot relabel a host stand-in as device.
    /// The receipt is one-shot; marking twice is refused.
    ///
    /// # Errors
    /// A missing/host-seam MhcEntry stage, no launch receipt, or an already-consumed receipt.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn record_device_stage(
        &mut self,
        receipt: &mut DeviceEntryReceipt,
        note: &'static str,
    ) -> Result<(), String> {
        if !receipt.launched {
            return Err(
                "refusing to mark a stage device-executed: no successful MhcEntry launch receipt"
                    .to_owned(),
            );
        }
        if receipt.consumed {
            return Err(
                "refusing to mark a stage device-executed: the receipt was already consumed"
                    .to_owned(),
            );
        }
        let run = self
            .stages
            .iter_mut()
            .find(|run| run.stage == ChainStage::MhcEntry)
            .ok_or_else(|| "MhcEntry is not an active layer-0 stage".to_owned())?;
        if run.seam.is_some() {
            return Err(
                "MhcEntry is a host-seam stage and cannot be marked device-executed".to_owned(),
            );
        }
        run.execution = LayerZeroExecution::Device;
        run.event.note = note;
        receipt.consumed = true;
        Ok(())
    }

    /// Compares the trace against the reference engine's layer-0 boundary row.
    ///
    /// This delegates to the engine's comparison, whose only honest class for the same reference
    /// arithmetic over the same bytes is Exact.
    ///
    /// # Errors
    /// A missing boundary tensor or a length mismatch.
    pub fn compare_to_reference(
        &self,
        row: &LayerBoundary,
    ) -> Result<sparkflash_engine::forward::layer_zero::LayerZeroComparison, String> {
        sparkflash_engine::forward::layer_zero::compare_to_reference(&self.trace, row)
            .map_err(|error| format!("layer-0 comparison: {error}"))
    }
}

fn one_layer_schedule(config: &ModelConfig, _token: u32) -> Result<ChainSchedule, String> {
    let ratio = config.compress_ratios.first().copied().unwrap_or(0);
    if ratio != 0 {
        return Err(format!(
            "layer 0 has compress ratio {ratio}; this pass only supports the SWA-only ratio-0 schedule"
        ));
    }
    Ok(ChainSchedule {
        prompt_tokens: 1,
        decode_steps: 0,
        window: SWA_WINDOW_SIZE,
        layers: vec![ChainLayer {
            layer: 0,
            first_position: 0,
            tokens: 1,
            mode: Csa2Mode::SwaOnly,
            ratio: 0,
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkflash_engine::forward::mhc::Mixes;

    fn config() -> ModelConfig {
        let mut config = ModelConfig::tiny();
        config.compress_ratios = vec![0; config.n_layers + config.n_mtp_layers];
        config
    }

    fn trace() -> LayerZeroTrace {
        let config = config();
        let dim = config.dim;
        let hc = config.hc_mult;
        let mut stages = Vec::new();
        for stage in LAYER_ZERO_HOST_STAGES {
            stages.push(StageEvent {
                stage,
                host_bytes_in: 4,
                host_bytes_out: 4,
                note: "test stage",
            });
        }
        LayerZeroTrace {
            token: 1,
            position: 0,
            embedding: vec![0.0; dim],
            attn_in: vec![0.0; dim],
            pre_attention_hidden: vec![0.0; dim],
            attn_mixes: Mixes::for_config(&config),
            attention_out: vec![0.0; dim],
            post_attention_hidden: vec![0.0; hc * dim],
            ffn_in: vec![0.0; dim],
            pre_ffn_hidden: vec![0.0; dim],
            ffn_mixes: Mixes::for_config(&config),
            ffn_out: vec![0.0; dim],
            residual_out: vec![0.0; hc * dim],
            stages,
        }
    }

    #[test]
    fn the_layer_zero_ledger_covers_every_active_host_seam_once() {
        let config = config();
        let run = LayerZeroRun::from_trace(&config, trace()).expect("layer-zero ledger");
        assert_eq!(run.stages.len(), 1 + LAYER_ZERO_HOST_STAGES.len());
        assert_eq!(run.stages[0].stage, ChainStage::Embedding);
        assert_eq!(
            run.stages[1..]
                .iter()
                .map(|stage| stage.stage.name())
                .collect::<Vec<_>>(),
            LAYER_ZERO_HOST_STAGES.to_vec(),
            "the device classification must follow the engine's 17-stage order exactly"
        );
        let host = run
            .stages
            .iter()
            .filter(|stage| stage.execution == LayerZeroExecution::Host)
            .count();
        let stand_in = run
            .stages
            .iter()
            .filter(|stage| stage.execution == LayerZeroExecution::HostStandIn)
            .count();
        assert_eq!(
            host + stand_in,
            1 + LAYER_STAGE_ORDER
                .iter()
                .filter(|stage| stage.runs_at(&run.schedule.layers[0]))
                .count()
        );
        assert_eq!(run.ledger.records().len(), host);
        for stage in &run.stages {
            if stage.execution == LayerZeroExecution::Host {
                assert!(stage.seam.is_some(), "{} has no seam", stage.stage.name());
            }
        }
        assert_eq!(
            run.stages
                .iter()
                .filter(|stage| stage.stage == ChainStage::MhcEntry)
                .map(|stage| stage.execution)
                .collect::<Vec<_>>(),
            vec![LayerZeroExecution::HostStandIn]
        );
    }

    #[test]
    fn a_host_only_run_cannot_mark_any_stage_device() {
        let config = config();
        let mut run = LayerZeroRun::from_trace(&config, trace()).expect("layer-zero ledger");
        let mut receipt = DeviceEntryReceipt::host_only();
        assert!(
            run.record_device_stage(&mut receipt, "not a launch")
                .is_err(),
            "a host-only run has no launch receipt and must be refused"
        );
        assert!(
            run.stages
                .iter()
                .all(|stage| stage.execution != LayerZeroExecution::Device),
            "a refused mark must not relabel any stage"
        );
        let text = run.report_lines().join("\n");
        assert!(text.contains("chain_execution=false"));
        assert!(text.contains("device_execution=false"));
        assert!(text.contains("token_produced=false"));
        assert!(text.contains("host stand-in (device builder not wired)"));
        assert!(text.contains("did NOT execute on a device"));
    }

    #[test]
    fn a_device_entry_run_marks_mhc_entry_and_folds_attn_norm() {
        let config = config();
        let mut run = LayerZeroRun::from_trace_inner(&config, trace(), true)
            .expect("the device-entry classification");
        assert!(
            run.seam_plan.entry("attn_norm").is_none(),
            "the fused norm cannot remain a planned host seam"
        );
        assert!(
            run.ledger
                .records()
                .iter()
                .all(|record| record.name != "attn_norm"),
            "the fused norm cannot be emitted as host bytes"
        );
        let norm = run
            .stages
            .iter()
            .find(|stage| stage.stage == ChainStage::AttnNorm)
            .expect("attn_norm stage");
        assert_eq!(norm.execution, LayerZeroExecution::DeviceFused);
        assert!(norm.seam.is_none());

        let mut receipt = DeviceEntryReceipt::launched();
        run.record_device_stage(&mut receipt, "test device entry")
            .expect("a launched receipt marks MhcEntry");
        assert_eq!(
            run.stages
                .iter()
                .find(|stage| stage.stage == ChainStage::MoeRouter)
                .map(|stage| stage.execution),
            Some(LayerZeroExecution::HostStandIn),
            "the receipt marks only MhcEntry; MoeRouter stays a host stand-in"
        );
        let text = run.report_lines().join("\n");
        assert!(text.contains("device_execution=true"));
        assert!(text.contains("mhc_entry: device"));
        assert!(text.contains("attn_norm: device-fused"));
        assert!(
            !text.contains("attn_norm: host"),
            "the hybrid report must not claim the host ran attn_norm"
        );
        assert!(
            run.record_device_stage(&mut receipt, "again").is_err(),
            "the launch receipt is one-shot"
        );
    }

    #[test]
    fn an_extra_trace_stage_is_refused_not_ignored() {
        let mut trace = trace();
        trace.stages.push(StageEvent {
            stage: "not_a_layer_zero_stage",
            host_bytes_in: 0,
            host_bytes_out: 0,
            note: "probe",
        });
        let error = LayerZeroRun::from_trace(&config(), trace)
            .expect_err("an extra event must not be silently ignored");
        assert!(error.contains("not_a_layer_zero_stage"), "{error}");
    }

    #[test]
    fn a_nonzero_layer_zero_ratio_is_refused() {
        let error = LayerZeroRun::from_trace(&ModelConfig::tiny(), trace())
            .expect_err("tiny's layer 0 has compress ratio 2");
        assert!(error.contains("compress ratio 2"), "{error}");
    }
}
