//! The four-rank forward/token stage: `--forward` behind the `--forward-preflight` interlock.
//!
//! This module is the gate in front of the device forward loop, and the loop itself once the gate
//! passes. The split matters for the next window: the preflight is host-only and unit-tested (it
//! derives the routed down-projection row geometry from the bind, runs the combine layout selector,
//! and either returns a launchable plan or a `sparkflash-tp4-token/1` refusal envelope), while the
//! loop is the only part that touches a GPU.
//!
//! The interlock is deliberate. `--forward` alone never launches a kernel: the window's collector
//! can pass it to get the refusal/evidence path first, and `--forward-preflight` is the explicit
//! "this argv is allowed to execute the four-rank loop" word. A stage that cannot launch refuses
//! with a named artefact; it never fabricates a token.
//!
//! State of the wiring (honest): the preflight, envelope and dump paths are complete and tested on
//! host; the device loop is compiled but not yet executed anywhere (no GPU window), and the routed
//! combine is blocked behind kernel-lead's task-6 stored-layout ABI, so the preflight's expected
//! verdict on the next window is a named refusal, not a token.

#[cfg(feature = "cuda")]
use std::cell::Cell;
use std::collections::BTreeMap;
#[cfg(feature = "cuda")]
use std::fs;
use std::path::{Path, PathBuf};

use sparkflash_kernels::cuda_driver::Deviceptr;

use sparkflash_comm::tcp::TcpTransport;

use crate::tp4_combine::{CombineGeometry, CombineKernelSet, CombineSelection, select_combine};
#[cfg(feature = "cuda")]
use crate::tp4_prefill::{StagedAttention, TwinStats};
use crate::tp4_token::{REDUCE_CALLS_PER_STEP, StepMs, TokenEnvelope, Tokens};

/// The safety interlock flag: without it, `--forward` produces a refusal envelope naming it.
pub const PREFLIGHT_FLAG: &str = "--forward-preflight";

/// The hidden size of the released configuration (the embedding row width).
pub const HIDDEN: usize = 5120;
/// The per-rank vocabulary/head shard width of the released plan.
pub const VOCAB_PER_RANK: u32 = 32_320;
/// The exchange sequence the embedding all-gather uses. It is far above the walk's sequences so a
/// reused transport can never confuse the two exchanges, and it never aliases a layer reduce seq.
pub const EMBED_EXCHANGE_SEQ: u64 = u64::MAX - 7;

/// The forward loop end-of-loop barrier: a dedicated namespace beat every rank joins before the
/// transport closes and the walk starts its own reduce sequence.
pub const FORWARD_BARRIER_SEQ: u64 = u64::MAX - 0x3_0000;
/// The exchange sequence the attention all-reduce uses. Offset a whole block below the embedding
/// gathers so a reused transport cannot confuse the two exchanges, and per-position by the caller.
pub const ATTENTION_REDUCE_SEQ: u64 = u64::MAX - 0x1_0000;

/// The MoE all-reduce exchange sequence, below the attention namespace.
pub const MOE_REDUCE_SEQ: u64 = u64::MAX - 0x2_0000;

/// The base sequence for the per-layer shared-expert shard gathers: six gathers per layer,
/// one per released tensor, at base - layer*8 - index.
pub const SHARED_EXPERT_GATHER_SEQ: u64 = u64::MAX - 0x4_0000;

/// The base sequence for the exact f32 head-logit gather at base - 2*position.
pub const HEAD_GATHER_SEQ: u64 = u64::MAX - 0x5_0000;
/// The base sequence for the head side-buffer gather at base - position.
pub const HEAD_SIDE_GATHER_SEQ: u64 = u64::MAX - 0x6_0000;

/// The four-rank token-consensus gather: each rank's decoded id is broadcast before a token claim
/// is written, so an envelope never reports a token the ranks disagree on.
pub const TOKEN_CONSENSUS_SEQ: u64 = u64::MAX - 0x7_0000;

/// The base sequence for the M2 native-combine slot-accumulator all-gather: every rank's f32 slot
/// accumulator travels as raw u16 words at base - layer, so the released combine epilogue can run
/// on the full f32 rank sum.
pub const NATIVE_COMBINE_GATHER_SEQ: u64 = u64::MAX - 0x8_0000;

#[cfg(feature = "cuda")]
/// The sequence of the reduce-path agreement gather: one readiness word per rank on the reference
/// TCP transport, in the same far-above-the-walk block as the embedding exchange. It is the only
/// extras collective a RoCE reduce run adds, and a TCP run never issues it.
pub const REDUCE_AGREEMENT_SEQ: u64 = u64::MAX - 0x9_0000;

#[cfg(feature = "cuda")]
/// The sequence of the reduce-path barrier a RoCE run issues before any QP is opened: one word on
/// the reference transport, so all four ranks begin the endpoint exchange together.
pub const REDUCE_BARRIER_SEQ: u64 = u64::MAX - 0xA_0000;

#[cfg(feature = "cuda")]
/// The RoCE reduce handle in a build that has the ibverbs backend compiled in.
#[cfg(feature = "rdma")]
type RoceReduceHandle = sparkflash_comm::rdma::RdmaTransport;

#[cfg(feature = "cuda")]
/// A build without the `rdma` feature has no RoCE handle. The uninhabited type keeps the reduce
/// slot's signature identical in both builds, and the switch reports the missing feature instead
/// of pretending a bring-up was attempted.
#[cfg(not(feature = "rdma"))]
enum RoceReduceHandle {}

#[cfg(feature = "cuda")]
/// The transport the two per-layer reduces take (`SPARK_COMM_REDUCE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceSwitch {
    /// The reference exchange the forward has always used, and the default.
    Tcp,
    /// The canonical RoCE one-shot: the same schedule driver and the same reduction rule.
    Roce,
}

#[cfg(feature = "cuda")]
impl ReduceSwitch {
    /// Parses a `SPARK_COMM_REDUCE` value. Unset/empty and `tcp` are the reference path; `roce`
    /// (also `rdma`) asks for the one-shot; anything else is reported by [`Self::from_env`] as a
    /// typo and treated as `tcp`, so the row's line always names the path that actually ran.
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "roce" | "rdma" => Self::Roce,
            _ => Self::Tcp,
        }
    }

    /// Reads the switch from the environment and returns the value the run reports.
    fn from_env() -> (Self, String) {
        let raw = std::env::var("SPARK_COMM_REDUCE").unwrap_or_default();
        let switch = Self::parse(&raw);
        let requested = raw.trim();
        if switch == Self::Tcp && !requested.is_empty() && !requested.eq_ignore_ascii_case("tcp") {
            println!(
                "forward reduce: SPARK_COMM_REDUCE={requested:?} is not roce|tcp; keeping the tcp exchange"
            );
        }
        (switch, requested.to_owned())
    }
}

#[cfg(feature = "cuda")]
/// Opens the RoCE reduce transport when the switch asks for it and this build has the backend.
///
/// A bring-up failure is a fallback, never a run failure: the caller keeps the TCP exchange and the
/// run reports the reason. Returns the transport and the reason string the run reports.
fn open_roce_reduce(rank: usize, world: usize) -> (Option<RoceReduceHandle>, String) {
    #[cfg(feature = "rdma")]
    {
        use sparkflash_comm::rdma::{RdmaConfig, RdmaTransport};
        let config = match RdmaConfig::from_env(world, rank) {
            Ok(config) => config,
            Err(error) => {
                return (
                    None,
                    format!("roce requested; configuration refused ({error}); using tcp"),
                );
            }
        };
        match RdmaTransport::open_with(config) {
            Ok(transport) => (Some(transport), "roce".to_owned()),
            Err(error) => (
                None,
                format!("roce requested; bring-up refused ({error}); using tcp"),
            ),
        }
    }
    #[cfg(not(feature = "rdma"))]
    {
        let _ = (rank, world);
        (
            None,
            "roce requested but this build has no rdma feature; using tcp".to_owned(),
        )
    }
}

#[cfg(feature = "cuda")]
/// The forward's reduce transport slot for the two per-layer collectives (attention output and the
/// routed MoE combine).
///
/// The slot changes **only which transport moves the frames**. Both paths run
/// `sparkflash_comm::exchange::all_reduce` and
/// `sparkflash_comm::reduce::canonical_all_reduce_into`: f32 accumulation from 0.0, ascending rank
/// order, one round-to-nearest-even bf16 store. No tolerance, no reassociation, no new kernel: the
/// canonical order is the file's existing rule, and the two runs' evidence digests are directly
/// comparable.
pub struct ForwardReduce<'a> {
    tcp: &'a TcpTransport,
    roce: Option<&'a RoceReduceHandle>,
    /// Requested switch, resolved path and (on a fallback) why. Printed before position 0 and again
    /// when the run drops the slot, so a row can prove which path its calls took even if it fails.
    report: String,
    roce_calls: Cell<u64>,
    tcp_calls: Cell<u64>,
}

#[cfg(feature = "cuda")]
impl<'a> ForwardReduce<'a> {
    /// Resolves the slot for the whole run.
    ///
    /// The requested RoCE transport is opened once, then every rank agrees on one path through a
    /// one-word all-gather on the reference transport before the loop's first reduce. The agreement
    /// is what makes the switch safe: a rank that opened RoCE while a peer stayed on TCP would
    /// block in a different exchange at the first reduce, so the path is all-or-nothing per run.
    fn resolve(
        tcp: &'a TcpTransport,
        maybe_roce: &'a Option<RoceReduceHandle>,
        switch: ReduceSwitch,
        requested: &str,
        open_detail: &str,
        world: usize,
    ) -> Result<Self, String> {
        use sparkflash_comm::transport::Transport;

        let mut report = format!("requested={requested:?} ");
        let roce = match switch {
            ReduceSwitch::Tcp => {
                report.push_str("path=tcp (reference exchange)");
                None
            }
            ReduceSwitch::Roce => {
                let ready = [u16::from(maybe_roce.is_some())];
                let mut seen = vec![0u16; world];
                tcp.all_gather_bf16(&ready, &mut seen, REDUCE_AGREEMENT_SEQ)
                    .map_err(|error| {
                        format!("forward reduce: reduce-path agreement gather: {error}")
                    })?;
                if seen.iter().all(|&word| word == 1) {
                    report.push_str("path=roce (every rank agreed)");
                    maybe_roce.as_ref()
                } else {
                    report.push_str(&format!(
                        "path=tcp (not every rank opened roce: {seen:?}; {open_detail})"
                    ));
                    None
                }
            }
        };
        Ok(Self {
            tcp,
            roce,
            report,
            roce_calls: Cell::new(0),
            tcp_calls: Cell::new(0),
        })
    }

    /// The path this run resolved to, for the evidence line.
    #[must_use]
    pub fn path(&self) -> &'static str {
        if self.roce.is_some() { "roce" } else { "tcp" }
    }

    /// All-reduces in place on the run's reduce path.
    ///
    /// The chosen transport's failure is returned, never silently retried on the other one: the
    /// peers of a failed collective are already mid-exchange, so switching transport mid-run would
    /// desynchronise the ranks. The fallback point is the bring-up/agreement above, before any
    /// reduce; a first-call failure is a genuine transport fault and the run must say so.
    fn all_reduce_bf16(
        &self,
        buf: &mut [u16],
        seq: u64,
        layout: &sparkflash_comm::layout::CollectiveLayout,
    ) -> Result<(), sparkflash_comm::CommError> {
        use sparkflash_comm::transport::Transport;
        #[cfg(feature = "rdma")]
        if let Some(roce) = self.roce {
            roce.all_reduce_bf16(buf, seq, layout)?;
            self.roce_calls.set(self.roce_calls.get() + 1);
            return Ok(());
        }
        #[cfg(not(feature = "rdma"))]
        debug_assert!(
            self.roce.is_none(),
            "a build without the rdma feature cannot hold a RoCE transport"
        );
        self.tcp.all_reduce_bf16(buf, seq, layout)?;
        self.tcp_calls.set(self.tcp_calls.get() + 1);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for ForwardReduce<'_> {
    fn drop(&mut self) {
        println!(
            "forward reduce: {} path={} roce_calls={} tcp_calls={}",
            self.report,
            self.path(),
            self.roce_calls.get(),
            self.tcp_calls.get()
        );
    }
}

/// The default-off M2 forward diagnostics switches (--forward-lean, --forward-perf-probe).
///
/// Lean removes diagnostics only: the host twin is not requested, the per-layer progress record is
/// written on the non-head cadence (every 8 layers plus the last), and the per-position head seam
/// f32 files are skipped except the last-position alias; every sha and the head record shape stay.
/// The perf probe installs host block timers with one cuCtxSynchronize per block and writes a
/// SEPARATE tp4-forward-perf-r{rank}.json: it never adds a field to the main evidence files.
/// No numeric input, launch or logits byte changes: the logits bin sha must stay byte-identical
/// with either switch, alone or together.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ForwardPerf {
    /// --forward-perf-probe: host block timers, one synchronise per block, separate perf file.
    pub probe: bool,
    /// --forward-perf-probe-nosync: the same per-block instrumentation with no boundary drain, so a
    /// block's wall is the host's and its kernel_ns arrives from one deferred event harvest.
    pub probe_nosync: bool,
    /// --forward-lean: twin off, deferred progress writes, deferred head seam files.
    pub lean: bool,
}

impl ForwardPerf {
    /// The default: diagnostics as before, the existing code path.
    pub const OFF: Self = Self {
        probe: false,
        probe_nosync: false,
        lean: false,
    };
    /// Whether either diagnostic switch is set (validation and the run log).
    #[must_use]
    pub fn any(self) -> bool {
        self.probe || self.probe_nosync || self.lean
    }
    /// The mode line of this run's separate perf evidence; only written by a probe run.
    #[must_use]
    pub fn probe_mode(self, rank: usize) -> ProbeMode {
        ProbeMode::for_run(self, rank)
    }
    /// The twin request decision: the existing head/twin cadence, unless lean removed the twin.
    #[must_use]
    pub fn twin_requested(self, is_head: bool, position: usize) -> bool {
        !self.lean && (is_head || position < 4 || position.is_multiple_of(8))
    }
    /// The per-layer progress cadence: head positions rewrite after every layer by default; lean
    /// uses the non-head cadence so the progress file stays O(n_layers).
    #[must_use]
    pub fn progress_due(self, is_head: bool, layer: usize, n_layers: usize) -> bool {
        (is_head && !self.lean) || layer % 8 == 7 || layer + 1 == n_layers
    }
    /// Whether this head position materialises any seam f32 file: default every head position;
    /// lean only the last position (the alias the m27 harness reads).
    #[must_use]
    pub fn write_head_seam_file(self, last_position: bool) -> bool {
        !self.lean || last_position
    }
    /// One-line description for the run log and the evidence.
    #[must_use]
    pub fn describe(self) -> String {
        if self.probe_nosync {
            format!("probe=nosync lean={}", self.lean)
        } else {
            format!("probe={} lean={}", self.probe, self.lean)
        }
    }
}

/// The mode line of the separate perf evidence file (default-off --forward-perf-probe).
///
/// Only a probe run writes one: the document records how the measurement was taken so a probe row
/// can be compared with its no-flag twin without reading the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProbeMode {
    /// The lean switch the row ran with (diagnostics only).
    pub lean: bool,
    /// The rank this file describes (every rank writes its own file).
    pub probe_rank: usize,
    /// Whether every block boundary drained the context: the probe's definition of a block time.
    pub synchronize_per_block: bool,
    /// Whether a block's wall is the host's alone: no boundary drain, and kernel_ns comes from the
    /// deferred event harvest instead of the per-launch drained bracket.
    pub host_only_wall: bool,
}

impl ProbeMode {
    /// The mode a run writes: one context synchronise at every block boundary.
    #[must_use]
    pub fn for_run(perf: ForwardPerf, rank: usize) -> Self {
        Self {
            lean: perf.lean,
            probe_rank: rank,
            synchronize_per_block: !perf.probe_nosync,
            host_only_wall: perf.probe_nosync,
        }
    }

    /// The `mode` object of the perf evidence.
    #[must_use]
    pub fn as_json(self) -> serde_json::Value {
        serde_json::json!({
            "lean": self.lean,
            "probe_rank": self.probe_rank,
            "synchronize_per_block": self.synchronize_per_block,
            "wall_clock": if self.host_only_wall {
                "host-only"
            } else {
                "device-inclusive"
            },
        })
    }
}

/// The clock and counter reading one probe block starts from; armed only when the probe is on.
///
/// Kept as a plain `Instant` plus one counter snapshot so the off arm is a single `None` (no clock
/// read, no state) and the on arm can be unit-tested: the loop only ever asks for the elapsed
/// milliseconds and, at the block boundary, for the counters the block moved against this reading.
/// The snapshot is what gives a block its kernel_ns and its launch/staging/sync counters while
/// letting nested blocks stay honest: a `layer_total` differences its own reading, so it cannot
/// double-count the blocks inside it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockTimer {
    started: Option<std::time::Instant>,
    snapshot: Option<probe::Snapshot>,
}

impl BlockTimer {
    /// Starts the clock when `enabled`; otherwise the timer stays unarmed and `elapsed_ms` is None.
    #[must_use]
    pub fn start(enabled: bool) -> Self {
        Self {
            started: enabled.then(std::time::Instant::now),
            snapshot: probe::snapshot(enabled),
        }
    }

    /// A timer built from an already-running clock (a function-level `Instant` reused as a total).
    #[must_use]
    pub fn since(started: std::time::Instant, enabled: bool) -> Self {
        Self {
            started: enabled.then_some(started),
            snapshot: probe::snapshot(enabled),
        }
    }

    /// Whether the clock was started.
    #[must_use]
    pub fn is_armed(self) -> bool {
        self.started.is_some()
    }

    /// The counter reading the block started from; the probe differences it at the boundary.
    ///
    /// Only the device path differences a reading, so a build without the `cuda` feature carries
    /// the accessor unused rather than dropping it from the type.
    #[must_use]
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn snapshot(self) -> Option<probe::Snapshot> {
        self.snapshot
    }

    /// The elapsed milliseconds, or None when the block is not being measured.
    #[must_use]
    pub fn elapsed_ms(self) -> Option<f64> {
        self.started
            .map(|start| start.elapsed().as_secs_f64() * 1000.0)
    }
}

/// The native-arithmetic rounding switches for the four-rank forward loop (the D41 native arm).
///
/// [NativeRounding::NONE] is the f32 arm and selects exactly the existing arithmetic: every switch
/// only turns on an additive native path, and the default branches are the pre-existing code. The
/// active set is printed in the run log and recorded in the forward evidence, because a window row
/// is only attributable if the running arithmetic is named next to its logits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeRounding {
    /// M2: gather every rank's f32 slot accumulator, round each expert's full output once, add the
    /// shared expert's BF16 output and round the MoE output once (model.py:893-904).
    pub combine: bool,
    /// M4: keep the residual stream at BF16 at every mHC boundary (hc_pre, norm and hc_post
    /// outputs), as the released block does (model.py:288-293, 954-966).
    pub mhc: bool,
    /// M3 loop path: run the dense shared expert on the released FP8-E4M3 32x32 store with the
    /// released BF16 boundaries, instead of the f32 reference seam (model.py:887, 903, 841-851).
    pub shared: bool,
}

impl NativeRounding {
    /// The default f32 arm: no native path is selected.
    pub const NONE: Self = Self {
        combine: false,
        mhc: false,
        shared: false,
    };

    /// Parses the --native-rounding value: none, combine, mhc, shared, both (or all), and the
    /// comma list of combine/mhc/shared. An empty value means none.
    ///
    /// # Errors
    /// An unknown term, or none combined with another term.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let mut selected = Self::NONE;
        let mut saw_none = false;
        for part in raw
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            match part {
                "none" => saw_none = true,
                "combine" => selected.combine = true,
                "mhc" => selected.mhc = true,
                "shared" => selected.shared = true,
                "both" | "all" => {
                    selected.combine = true;
                    selected.mhc = true;
                }
                other => {
                    return Err(format!(
                        "--native-rounding {raw}: unknown term {other:?}; expected none, combine, mhc, shared, both, or a comma list of combine/mhc/shared"
                    ));
                }
            }
        }
        if saw_none && selected != Self::NONE {
            return Err(format!(
                "--native-rounding {raw}: none cannot be combined with another term"
            ));
        }
        Ok(selected)
    }

    /// The value recorded in the run log and the forward evidence.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match (self.combine, self.mhc, self.shared) {
            (false, false, false) => "none",
            (true, false, false) => "combine",
            (false, true, false) => "mhc",
            (false, false, true) => "shared",
            (true, true, false) => "both",
            (true, false, true) => "combine,shared",
            (false, true, true) => "mhc,shared",
            (true, true, true) => "combine,mhc,shared",
        }
    }

    /// Whether any native path is selected.
    #[must_use]
    pub fn any(self) -> bool {
        self.combine || self.mhc || self.shared
    }
}

/// Where the last prompt token embedding row lives: (owner rank, row inside its shard).
#[must_use]
pub fn embedding_location(token: u32) -> (usize, usize) {
    let token = token as usize;
    (
        token / VOCAB_PER_RANK as usize,
        token % VOCAB_PER_RANK as usize,
    )
}

/// One layer-0 attention operand as the *plan* describes it (dtype from the checkpoint header).
///
/// This is the authoritative layout source for the census: `b12x_layout` tiles an operand only when
/// the plan says F8_E4M3 with a [cols, inner] shape in a tiled group, so the census derives the
/// expected layout from this record and only then checks that the bind actually bound it that way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionOperandPlan {
    /// The planned tensor name.
    pub name: String,
    /// The dtype from the checkpoint header/plan.
    pub dtype: String,
    /// The layout the kernels expect: "b12x" for F8_E4M3, otherwise "stored".
    pub layout: &'static str,
    /// Whether the plan's layout is launchable by the projection kernel families (B12x only).
    pub launchable: bool,
    /// The planned B12x tile geometry (cols, inner), None for a stored or absent operand.
    pub b12x_geometry: Option<(usize, usize)>,
    /// The planned tiled weight bytes (None unless B12x).
    pub weight_bytes: Option<u64>,
    /// The planned scale-grid bytes (None unless B12x).
    pub scale_bytes: Option<u64>,
}

/// Builds [AttentionOperandPlan] rows for the five layer-0 attention operands from plan dtypes.
///
/// A name missing from the iterator is kept with dtype "absent": the census reports it as an
/// unplanned operand instead of silently dropping it.
#[must_use]
pub fn attention_plan_from_dtypes<'a, I>(shards: I) -> Vec<AttentionOperandPlan>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut dtypes: BTreeMap<&str, &str> = BTreeMap::new();
    for (name, dtype) in shards {
        dtypes.insert(name, dtype);
    }
    LAYER0_ATTENTION_OPERANDS
        .iter()
        .map(|(name, _kernel, _needs_b12x)| {
            let dtype = dtypes.get(name).copied().unwrap_or("absent");
            let layout = if dtype == "F8_E4M3" { "b12x" } else { "stored" };
            AttentionOperandPlan {
                name: (*name).to_string(),
                dtype: dtype.to_string(),
                layout,
                launchable: layout == "b12x",
                b12x_geometry: None,
                weight_bytes: None,
                scale_bytes: None,
            }
        })
        .collect()
}

/// Builds [AttentionOperandPlan] rows from the rank plan's own operand-layout predicate.
///
/// This is the same predicate the arena upload and the attention bind use
/// (`ResidentShard::operand_layout`, derived from dtype/group/shape alone), so the census cannot
/// disagree with the plan silently: a B12x plan record carries its tile geometry and byte counts, a
/// stored record carries only its dtype and bytes, and a name the plan does not carry is "absent".
#[must_use]
pub fn attention_plan_from_plan(
    plan: &sparkflash_device::resident::ResidentPlan,
) -> Vec<AttentionOperandPlan> {
    use sparkflash_device::resident::OperandLayout;
    LAYER0_ATTENTION_OPERANDS
        .iter()
        .map(|(name, _kernel, _needs_b12x)| {
            let dtype = plan
                .shards()
                .iter()
                .find(|shard| shard.name == *name)
                .map_or_else(|| "absent".to_string(), |shard| shard.dtype.clone());
            match plan.operand_layout(name) {
                Some(OperandLayout::B12x {
                    cols,
                    inner,
                    weight_bytes,
                    scale_bytes,
                }) => AttentionOperandPlan {
                    name: (*name).to_string(),
                    dtype,
                    layout: "b12x",
                    launchable: true,
                    b12x_geometry: Some((cols, inner)),
                    weight_bytes: Some(weight_bytes),
                    scale_bytes: Some(scale_bytes),
                },
                Some(OperandLayout::Stored { .. }) | None => AttentionOperandPlan {
                    name: (*name).to_string(),
                    dtype,
                    layout: "stored",
                    launchable: false,
                    b12x_geometry: None,
                    weight_bytes: None,
                    scale_bytes: None,
                },
            }
        })
        .collect()
}

/// Whether a plan tensor name is one of the five attention projection weights the loop launches.
#[must_use]
pub fn is_attention_projection_weight(name: &str) -> bool {
    if !(name.starts_with("layers.") || name.starts_with("mtp.")) || !name.contains(".attn.") {
        return false;
    }
    [
        "wq_a.weight",
        "wkv.weight",
        "wq_b.weight",
        "wo_a.weight",
        "wo_b.weight",
    ]
    .iter()
    .any(|suffix| name.ends_with(&format!(".{suffix}")))
}

/// The full-plan attention operand census: every attention tensor, by group, with the plan's own
/// layout predicate deciding B12x versus stored.
///
/// This is the plan-only half of kernel-lead's wiring contract (the bind half is
/// check_operand_bind, run per layer-0 operand in the loop): it counts the 215 projection weights
/// and the three attention groups across all 43 attention layers (40 backbone + 3 MTP), and reports
/// tiled tensor counts with weight/scale byte totals so the loop's evidence and the standalone
/// attn-operands gate can be compared line for line. It launches nothing and reads no bind.
#[must_use]
pub fn attention_operand_census(
    plan: &sparkflash_device::resident::ResidentPlan,
) -> serde_json::Value {
    use sparkflash_device::resident::{OperandLayout, ResidentGroup};

    let mut projection_weights = 0usize;
    let mut projection_b12x = 0usize;
    let mut projection_stored = 0usize;
    // (tensors, tiled weights, tiled weight bytes, scale bytes, resident norm bytes)
    let mut groups: std::collections::BTreeMap<&'static str, (usize, usize, u64, u64, u64)> =
        std::collections::BTreeMap::new();
    for shard in plan.shards() {
        let group = match shard.group {
            ResidentGroup::AttentionReplicated => "attention-replicated",
            ResidentGroup::AttentionColumn => "attention-column",
            ResidentGroup::AttentionRow => "attention-row",
            ResidentGroup::AttentionSink => "attention-sink",
            _ => continue,
        };
        let entry = groups.entry(group).or_insert((0, 0, 0, 0, 0));
        entry.0 += 1;
        let projection = is_attention_projection_weight(&shard.name);
        if projection {
            projection_weights += 1;
        }
        match shard.operand_layout() {
            OperandLayout::B12x {
                weight_bytes,
                scale_bytes,
                ..
            } => {
                entry.1 += 1;
                entry.2 += weight_bytes;
                entry.3 += scale_bytes;
                if projection {
                    projection_b12x += 1;
                }
            }
            OperandLayout::Stored { .. } => {
                // The group's byte total includes the resident tensors that are not tiled: the
                // B12x scale grids are already counted from their weight's plan geometry, so
                // only the non-scale stored tensors (q_norm/kv_norm weights) add bytes here.
                if !shard.name.ends_with(".scale") {
                    entry.4 += shard.shard_bytes;
                }
                if projection {
                    projection_stored += 1;
                }
            }
        }
    }
    let rows: Vec<serde_json::Value> = groups
        .iter()
        .map(
            |(group, (tensors, tiled, weight_bytes, scale_bytes, norm_bytes))| {
                serde_json::json!({
                    "group": group,
                    "tensors": tensors,
                    "tiled": tiled,
                    "tiled_weight_bytes": weight_bytes,
                    "scale_bytes": scale_bytes,
                    "resident_norm_bytes": norm_bytes,
                    "total_bytes": weight_bytes + scale_bytes + norm_bytes,
                })
            },
        )
        .collect();
    serde_json::json!({
        "schema": "sparkflash-tp4-attention-operands/1",
        "plan_hash": format!("{:#018x}", plan.plan_hash()),
        "rank": plan.rank(),
        "world": plan.world(),
        "projection_weights": {
            "tensors": projection_weights,
            "b12x": projection_b12x,
            "stored": projection_stored,
            "expected": "all of them are plan-B12x F8_E4M3 with F8_E8M0 scales on the released headers",
        },
        "groups": rows,
        "bind_checked": "per layer-0 operand in attention_stage_census (check_operand_bind); this census is plan-only",
        "expected_gate_lines": {
            "replicated": "attention-replicated 258 tensors (86 tiled, 395066112 B)",
            "column": "attention-column 172 (86 tiled, 812390400 B)",
            "row": "attention-row 86 (43 tiled, 451328000 B)",
            "note": "the totals are tiled weight + B12x scale grid + the resident non-tiled norm weights, the same composition the standalone attn-operands gate prints",
        },
        "repacked_bytes": 0,
        "per_token_weight_h2d_bytes": 0,
        "note": "no new cubin and no pin change: the twenty-entry baseline is unchanged; where a repack would run: nowhere",
    })
}

/// The checkpoint's native dense projection kernel: E4M3 activation codes with one ue8m0 scale per
/// 32 contiguous values against B12x E4M3 weights, f32 accumulate, bf16 output (D39 rule 1;
/// docs/build/native-kernel-arithmetic-audit.md section 2.1). The native kernel for wq_a, wkv, wq_b
/// and wo_b; the D37 bf16a dense family computes a different model (audit Appendix A) and is not
/// native for any of them.
pub const NATIVE_DENSE_KERNEL: &str = "b12x_mxfp8_dense_sm121";

/// The BF16-activation wo_a kernel: the checkpoint's native rule for exactly that projection (D39
/// task-29 note: the released forward dequantises wo_a to bf16 and runs a bf16 grouped einsum with
/// no activation quantiser, so this variant removes only the quantisation native does not perform).
pub const NATIVE_WO_A_KERNEL: &str = "b12x_bf16a_mxfp8w_wo_a_sm121";

/// One layer-0 attention operand the loop must launch, with the kernel the native arithmetic
/// selects for it.
///
/// The third field is whether the kernel family only reads B12x (FP8 codes + scale grid) weights.
/// The bind marks a B12x operand by a scale pointer distinct from the weight pointer; a stored
/// (BF16) operand repeats the weight pointer, which is exactly the case the four-rank arm cannot
/// launch today because it has no per-layer repack staging.
pub const LAYER0_ATTENTION_OPERANDS: [(&str, &str, bool); 5] = [
    ("layers.0.attn.wq_a.weight", NATIVE_DENSE_KERNEL, true),
    ("layers.0.attn.wkv.weight", NATIVE_DENSE_KERNEL, true),
    ("layers.0.attn.wq_b.weight", NATIVE_DENSE_KERNEL, true),
    ("layers.0.attn.wo_a.weight", NATIVE_WO_A_KERNEL, true),
    ("layers.0.attn.wo_b.weight", NATIVE_DENSE_KERNEL, true),
];

/// Whether a bound operand is B12x (distinct scale pointer) or stored (repeated weight pointer).
#[must_use]
pub fn bound_operand_kind(pair: (Deviceptr, Deviceptr)) -> &'static str {
    if pair.0.0 == pair.1.0 {
        "stored"
    } else {
        "b12x"
    }
}

/// The attention stage census: for each layer-0 attention projection, what the bind holds, which
/// kernel would read it, and whether that launch is possible on this arm today.
///
/// This is derived from the bind only (no driver, no device), so it is host-testable and can be
/// written into the forward evidence before a single kernel runs. It is the named answer to "which
/// stage refuses": the first window that runs it will print exactly which operands are launchable.
#[must_use]
pub fn attention_stage_census(
    pointers: &BTreeMap<String, (Deviceptr, Deviceptr)>,
    planned: &[AttentionOperandPlan],
    plan: Option<&sparkflash_device::resident::ResidentPlan>,
) -> serde_json::Value {
    let mut rows = Vec::with_capacity(LAYER0_ATTENTION_OPERANDS.len());
    let mut launchable = 0usize;
    let mut blocked = Vec::new();
    let mut inconsistent = Vec::new();
    let mut disagreements = Vec::new();
    for (name, kernel, needs_b12x) in LAYER0_ATTENTION_OPERANDS {
        let pair = pointers.get(name).copied();
        let bound_kind = pair.map(bound_operand_kind);
        let plan_row = planned.iter().find(|row| row.name == name);
        // The plan is authoritative for the expected layout; the bind is what is actually resident.
        let layout_source = if plan_row.is_some() {
            "plan-dtype"
        } else {
            "bind-pointer"
        };
        let expected =
            plan_row.map_or(if needs_b12x { "b12x" } else { "stored" }, |row| row.layout);
        let consistent = bound_kind.map(|kind| kind == expected);
        if consistent == Some(false) {
            inconsistent.push(name.to_string());
        }
        // The bind is checked against the plan's own layout predicate, not against pointer
        // arithmetic: check_operand_bind names the tensor and the observed pair when they disagree.
        let plan_check = match (plan, pair) {
            (Some(plan), Some((weight, scale))) => plan
                .shards()
                .iter()
                .find(|shard| shard.name == name)
                .map(|shard| {
                    match sparkflash_device::resident::check_operand_bind(shard, weight.0, scale.0)
                    {
                        Ok(layout) => serde_json::json!({
                            "ok": true,
                            "planned": layout.as_str(),
                        }),
                        Err(error) => {
                            disagreements.push(serde_json::json!({
                                "name": name,
                                "error": error,
                            }));
                            serde_json::json!({"ok": false, "error": error})
                        }
                    }
                }),
            _ => None,
        };
        let is_launchable = match (pair, bound_kind) {
            (Some(_), Some("b12x")) => true,
            (Some(_), Some("stored")) if !needs_b12x => true,
            _ => false,
        };
        if is_launchable {
            launchable += 1;
        } else {
            blocked.push(name.to_string());
        }
        rows.push(serde_json::json!({
            "name": name,
            "kernel": kernel,
            "bound": pair.is_some(),
            "layout": bound_kind,
            "layout_source": layout_source,
            "plan_dtype": plan_row.map(|row| row.dtype.clone()),
            "expected_layout": expected,
            "consistent": consistent,
            "plan_check": plan_check,
            "needs_b12x_weights": needs_b12x,
            "launchable": is_launchable,
        }));
    }
    let blocked_reason = if blocked.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(
            "a blocked operand is one the kernel family cannot read: the projection kernels are B12x-only, so an operand is launchable only when the plan says F8_E4M3 and the bind holds a B12x tile with its scale pointer; any other combination is named here instead of being launched".to_string(),
        )
    };
    let inconsistent_reason = if inconsistent.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(
            "the bind is not what the plan's dtype says: a tiled F8_E4M3 operand must be bound with a distinct scale pointer, a non-F8 operand with a repeated weight pointer; this is a bind/plan defect, not a kernel choice".to_string(),
        )
    };
    serde_json::json!({
        "schema": "sparkflash-tp4-attention-census/1",
        "stage": "layer0-attention-projections",
        "rows": rows,
        "launchable": launchable,
        "blocked": blocked,
        "inconsistent": inconsistent,
        "inconsistent_reason": inconsistent_reason,
        "disagreements": disagreements,
        "plan_vs_bind_disagreements": disagreements.len(),
        "blocked_reason": blocked_reason,
        "note": "census only: no kernel is launched by this function and no token can be claimed from it",
    })
}

/// The token id the first dumped row must start from.
///
/// A teacher-forced token list is [prompt token ids] followed by the reference tokens for the later
/// dumped rows, so with `positions` rows the list holds `prompt_len + positions - 1` ids and the
/// first row's input is `prompt_tokens[prompt_len - 1] = prompt_tokens[len - positions]`. The
/// fixture case is 138 ids and 9 rows: prompt_len 130, first row position 129, input ids[129]; the
/// last id is position 137's *reference*, never fed, so using `last()` would start the loop one
/// position late.
#[must_use]
pub fn first_step_token(prompt_tokens: &[u32], positions: usize) -> Option<u32> {
    if positions == 0 || prompt_tokens.len() < positions {
        return None;
    }
    prompt_tokens.get(prompt_tokens.len() - positions).copied()
}

/// The prompt length a teacher-forced token list implies for this many dumped rows.
#[must_use]
pub fn prompt_len_for(prompt_tokens: &[u32], positions: usize) -> Option<usize> {
    if positions == 0 || prompt_tokens.len() < positions {
        return None;
    }
    Some(prompt_tokens.len() - positions + 1)
}

/// The bytes one prefill embedding exchange moves across all ranks.
///
/// Each of the `prompt_len` positions gathers one world-sized bf16 hidden row; the rows assemble in
/// prompt order. This is the size the loop's evidence reports before a single layer runs.
#[must_use]
pub fn prefill_exchange_bytes(world: usize, prompt_len: usize) -> usize {
    world
        .saturating_mul(HIDDEN)
        .saturating_mul(2)
        .saturating_mul(prompt_len)
}

/// The zeroed trace ring the layer-0 projection launches pass: the kernel dereferences the ring
/// unconditionally, so this keeps that read legal with active_scope_mask = 0 (trace off).
#[cfg(feature = "cuda")]
pub const TRACE_RING_BYTES: usize = 64;

/// The mHC stream count of the released model (four residual copies).
pub const HC_MULT: usize = 4;

/// Element limit for the Engram seam's tensor reads. One wkv projection is 157.3 M elements and one
/// whole Engram table is 98.3 G elements, so this is the single-node host chain's own limit
/// (tp4_host.rs ELEMENT_LIMIT): the projection fits, a whole-table fallback on a source that cannot
/// page rows is refused by name instead of materialising 98 GB.
#[cfg(feature = "cuda")]
pub const ENGRAM_ELEMENT_LIMIT: usize = 1 << 31;
/// The released mHC predictor's row count: hc^2 + 2*hc.
pub const PREDICTOR_ROWS: usize = HC_MULT * HC_MULT + 2 * HC_MULT;

/// Little-endian f32 bytes decoded to a vector.
///
/// # Errors
/// A byte length that is not a whole number of f32 values.
pub fn f32_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!("{} bytes are not whole f32 values", bytes.len()));
    }
    Ok((0..bytes.len() / 4)
        .map(|index| {
            let mut word = [0u8; 4];
            word.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
            f32::from_le_bytes(word)
        })
        .collect())
}

/// Little-endian bf16 bytes decoded to f32 exactly (bf16 is the top half of an f32).
///
/// # Errors
/// A byte length that is not a whole number of bf16 values.
pub fn bf16_from_le_bytes(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!("{} bytes are not whole bf16 values", bytes.len()));
    }
    Ok((0..bytes.len() / 2)
        .map(|index| {
            let bits = u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect())
}

/// [bf16_from_le_bytes] into a caller-owned vector, so a loop can reuse the row across calls.
///
/// # Errors
///
/// The byte count is not a whole number of bf16 values.
pub fn bf16_from_le_bytes_into(target: &mut Vec<f32>, bytes: &[u8]) -> Result<(), String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!("{} bytes are not whole bf16 values", bytes.len()));
    }
    target.clear();
    target.reserve(bytes.len() / 2);
    for pair in bytes.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*pair);
        target.push(f32::from_bits(u32::from(bits) << 16));
    }
    Ok(())
}

/// The layer-0 mHC entry seam's host outputs: the three coefficient sets and the attention block
/// input, computed exactly as the released host reference does from the same tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct MhcEntrySeam {
    /// sha256 of the hc residual stream the seam consumed (hc copies of the hidden row).
    pub hidden_sha256: String,
    /// sha256 of the released 24-row predictor.
    pub predictor_sha256: String,
    /// sha256 of the 24-row bias.
    pub base_sha256: String,
    /// The released three group scales.
    pub scale: Vec<f32>,
    /// sha256 of the attention norm weight.
    pub norm_sha256: String,
    /// The input mixes: sigmoid(mix*scale+base)+hc_eps.
    pub pre: Vec<f32>,
    /// The output mixes: 2*sigmoid(mix*scale+base).
    pub post: Vec<f32>,
    /// The residual mixes after the split sinkhorn, [hc, hc] source-row order.
    pub comb: Vec<f32>,
    /// The collapsed attention block input (the layer-0 seed A = [1,0,0,0] applied to the stream).
    pub attn_in: Vec<f32>,
    /// sha256 of [attn_in] as f32-le.
    pub attn_in_sha256: String,
    /// The attention block input after attn_norm (what the projections will consume).
    pub normed: Vec<f32>,
    /// sha256 of [normed] as f32-le.
    pub normed_sha256: String,
    /// The hc residual stream the seam consumed (hc copies of the hidden row), kept for hc_post.
    pub hidden: Vec<f32>,
}

/// Computes one layer's mHC entry on the host from the released tensors.
///
/// This is the exact released reference arithmetic: mhc::hc_mixes over the hc residual copies,
/// then hc_pre with the ATTENTION PRE MIX the reference propagates across the layer boundary.
/// The model starts with the identity seed [1,0,0,0] (no sublayer has produced a pre mix yet), so
/// layer 0 passes that seed; every later layer passes the previous layer's FFN pre mix (the
/// mhc::Mixes pre set from the hc_ffn predictor, verified against the phase1-boundary v1 capture:
/// hc_in_pre(L) == hc_ffn_pre(L-1) for all 39 transitions). It launches no kernel and cannot
/// produce a token; the device launch (mega_mhc_sm121) remains a separate stage.
///
/// # Errors
/// Any tensor whose length disagrees with the model geometry, or a non-finite coefficient.
pub fn mhc_entry_host_seam(
    config: &sparkflash_engine::config::ModelConfig,
    hidden: &[f32],
    predictor: &[f32],
    base: &[f32],
    scale: &[f32],
    attn_norm: &[f32],
    attn_pre_mix: &[f32],
) -> Result<MhcEntrySeam, String> {
    mhc_entry_host_seam_with(
        config,
        hidden,
        predictor,
        base,
        scale,
        attn_norm,
        attn_pre_mix,
        false,
    )
}

/// [mhc_entry_host_seam] with native_mhc, the M4 arm's BF16 boundaries: the hc_pre collapse is
/// rounded to BF16 before the norm and the norm output is rounded to BF16 before the attention
/// sublayer, which is what the released block hands on (model.py:288-293, 954-957).
///
/// # Errors
/// [mhc_entry_host_seam]'s refusals.
#[allow(clippy::too_many_arguments)]
pub fn mhc_entry_host_seam_with(
    config: &sparkflash_engine::config::ModelConfig,
    hidden: &[f32],
    predictor: &[f32],
    base: &[f32],
    scale: &[f32],
    attn_norm: &[f32],
    attn_pre_mix: &[f32],
    native_mhc: bool,
) -> Result<MhcEntrySeam, String> {
    mhc_entry_host_seam_with_digests(
        config,
        hidden,
        predictor,
        base,
        scale,
        attn_norm,
        attn_pre_mix,
        native_mhc,
        None,
        None,
    )
}

/// [mhc_entry_host_seam_with] with the three position-independent seam digests optionally reused
/// from the run-scope seam cache: `None` computes them exactly where the public path always has,
/// `Some` reuses the cached predictor/base/norm sha256 values; the arithmetic is identical either
/// way and the hidden/attn_in/normed digests stay data-dependent per call.
///
/// # Errors
/// [mhc_entry_host_seam]'s refusals.
#[allow(clippy::too_many_arguments)]
fn mhc_entry_host_seam_with_digests(
    config: &sparkflash_engine::config::ModelConfig,
    hidden: &[f32],
    predictor: &[f32],
    base: &[f32],
    scale: &[f32],
    attn_norm: &[f32],
    attn_pre_mix: &[f32],
    native_mhc: bool,
    digests: Option<&MhcSeamDigests>,
    transposed_predictor: Option<&[f32]>,
) -> Result<MhcEntrySeam, String> {
    use sparkflash_engine::forward::mhc;
    use sparkflash_engine::linalg;

    let hc = config.hc_mult;
    let dim = config.dim;
    if hc != HC_MULT {
        return Err(format!(
            "mHC entry: the released seam is the {HC_MULT}-stream reference, config declares {hc}"
        ));
    }
    if hidden.len() != hc * dim {
        return Err(format!(
            "mHC entry: hidden is {} values, expected {hc} copies of {dim}",
            hidden.len()
        ));
    }
    let predictor_values = PREDICTOR_ROWS
        .checked_mul(hc * dim)
        .ok_or_else(|| "mHC entry: predictor length overflows".to_string())?;
    if predictor.len() != predictor_values {
        return Err(format!(
            "mHC entry: predictor is {} values, expected {PREDICTOR_ROWS} rows x {} = {predictor_values}",
            predictor.len(),
            hc * dim
        ));
    }
    if let Some(weight) = transposed_predictor.filter(|weight| weight.len() != predictor_values) {
        return Err(format!(
            "mHC entry: the transposed predictor is {} values, expected {predictor_values}",
            weight.len()
        ));
    }
    if base.len() != PREDICTOR_ROWS {
        return Err(format!(
            "mHC entry: base is {} values, expected {PREDICTOR_ROWS}",
            base.len()
        ));
    }
    if scale.len() != 3 {
        return Err(format!(
            "mHC entry: scale is {} values, expected 3",
            scale.len()
        ));
    }
    if attn_norm.len() != dim {
        return Err(format!(
            "mHC entry: attn_norm is {} values, expected {dim}",
            attn_norm.len()
        ));
    }

    let mut mixes = mhc::Mixes::for_config(config);
    match transposed_predictor {
        Some(weight) => mhc::hc_mixes_transposed(config, hidden, weight, scale, base, &mut mixes),
        None => mhc::hc_mixes(config, hidden, predictor, scale, base, &mut mixes),
    }
    .map_err(|error| format!("mHC entry: hc_mixes: {error}"))?;
    if attn_pre_mix.len() != hc {
        return Err(format!(
            "mHC entry: attn_pre_mix is {} values, expected {hc} (the previous sublayer's pre mix; layer 0 seeds the identity one-hot)",
            attn_pre_mix.len()
        ));
    }
    if attn_pre_mix.iter().any(|value| !value.is_finite()) {
        return Err("mHC entry: attn_pre_mix has a non-finite value".to_string());
    }
    let mut attn_in = vec![0.0f32; dim];
    let mut normed;
    if native_mhc {
        mhc::hc_pre_bf16(config, attn_pre_mix, hidden, &mut attn_in)
            .map_err(|error| format!("mHC entry: hc_pre: {error}"))?;
        normed = attn_in.clone();
        mhc::rms_norm_bf16(&mut normed, attn_norm, dim, config.norm_eps as f32)
            .map_err(|error| format!("mHC entry: attn_norm: {error}"))?;
    } else {
        mhc::hc_pre(config, attn_pre_mix, hidden, &mut attn_in)
            .map_err(|error| format!("mHC entry: hc_pre: {error}"))?;
        normed = attn_in.clone();
        linalg::rms_norm(&mut normed, attn_norm, dim, config.norm_eps as f32)
            .map_err(|error| format!("mHC entry: attn_norm: {error}"))?;
    }

    for (name, values) in [
        ("pre", &mixes.pre),
        ("post", &mixes.post),
        ("comb", &mixes.comb),
        ("attn_in", &attn_in),
        ("normed", &normed),
    ] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "mHC entry: the {name} output has a non-finite value; refusing to stage it"
            ));
        }
    }
    let hidden_sha256 = evidence_f32_sha256(hidden);
    let (predictor_sha256, base_sha256, norm_sha256) = match digests {
        Some(digests) => (
            digests.predictor.clone(),
            digests.base.clone(),
            digests.norm.clone(),
        ),
        None => (
            evidence_f32_sha256(predictor),
            evidence_f32_sha256(base),
            evidence_f32_sha256(attn_norm),
        ),
    };
    Ok(MhcEntrySeam {
        hidden_sha256,
        predictor_sha256,
        base_sha256,
        scale: scale.to_vec(),
        norm_sha256,
        pre: mixes.pre,
        post: mixes.post,
        comb: mixes.comb,
        attn_in_sha256: evidence_f32_sha256(&attn_in),
        attn_in,
        normed_sha256: evidence_f32_sha256(&normed),
        normed,
        hidden: hidden.to_vec(),
    })
}

/// The three position-independent sha256 digests of one layer's mHC-entry seam tensors.
///
/// The predictor/base/norm digests are pure functions of immutable per-layer arena tensors, so the
/// run-scope seam cache computes them once and every per-layer seam reuses them; the hidden,
/// attn_in and normed digests stay data-dependent and are computed per call as before.
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct MhcSeamDigests {
    predictor: String,
    base: String,
    norm: String,
}

/// The seam-cache reader: a tensor name and expected byte length to raw bytes. The CUDA build
/// passes read_arena_bytes; the host tests pass a recording closure.
#[cfg(any(feature = "cuda", test))]
type SeamReader<'a> = dyn FnMut(&str, usize) -> Result<Vec<u8>, String> + 'a;

/// One layer's raw seam tensor bytes, as the seam-cache prewarm reads them.
#[cfg(any(feature = "cuda", test))]
struct LayerSeamRaw {
    attn_fn: Vec<u8>,
    attn_base: Vec<u8>,
    attn_scale: Vec<u8>,
    attn_norm: Vec<u8>,
    ffn_fn: Vec<u8>,
    ffn_base: Vec<u8>,
    ffn_scale: Vec<u8>,
    ffn_norm: Vec<u8>,
}

/// One layer's decoded f32 seam tensors plus the three digests the seam reuses every position.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone)]
struct LayerSeamTensors {
    attn_fn: Vec<f32>,
    /// `attn_fn` transposed to the `[hc_width][PREDICTOR_ROWS]` layout the interleaved hc_*_fn
    /// projection reads. The same f32 bits, staged once per layer so no per-call pass pays for it.
    attn_fn_transposed: Vec<f32>,
    attn_base: Vec<f32>,
    attn_scale: Vec<f32>,
    attn_norm: Vec<f32>,
    attn_predictor_sha256: String,
    attn_base_sha256: String,
    attn_norm_sha256: String,
    ffn_fn: Vec<f32>,
    /// `ffn_fn` transposed, for the same reason as `attn_fn_transposed`.
    ffn_fn_transposed: Vec<f32>,
    ffn_base: Vec<f32>,
    ffn_scale: Vec<f32>,
    ffn_norm: Vec<f32>,
}

#[cfg(any(feature = "cuda", test))]
impl LayerSeamTensors {
    /// The cached position-independent digests the mHC entry seam reuses.
    fn digests(&self) -> MhcSeamDigests {
        MhcSeamDigests {
            predictor: self.attn_predictor_sha256.clone(),
            base: self.attn_base_sha256.clone(),
            norm: self.attn_norm_sha256.clone(),
        }
    }

    /// The decoded f32 memory this layer's cached tensors occupy (norm source dtype independent).
    fn decoded_bytes(&self) -> usize {
        4 * (self.attn_fn.len()
            + self.attn_fn_transposed.len()
            + self.attn_base.len()
            + self.attn_scale.len()
            + self.attn_norm.len()
            + self.ffn_fn.len()
            + self.ffn_fn_transposed.len()
            + self.ffn_base.len()
            + self.ffn_scale.len()
            + self.ffn_norm.len())
    }
}

/// The run-scope seam cache: one decoded tensor set per layer, built once before the position loop.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug)]
struct LayerSeamCache {
    layers: Vec<LayerSeamTensors>,
    reads: usize,
    bytes: usize,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    wall_ms: u64,
}

/// Reads one layer's eight seam tensors through `reader`, validating both norm encodings before any
/// read: the same names, byte lengths and refusals the per-call seam produced at position 0.
///
/// # Errors
/// The per-call seam's dtype/geometry refusals, now raised before the loop instead of at layer 0.
#[cfg(any(feature = "cuda", test))]
fn read_layer_seam_raw(
    reader: &mut SeamReader<'_>,
    layer: usize,
    request: &TokenStageRequest<'_>,
) -> Result<(LayerSeamRaw, usize), String> {
    let config = request
        .config
        .ok_or_else(|| "mHC entry: the loop was built without the model config".to_string())?;
    let dim = config.dim;
    let hc = config.hc_mult;
    let hc_attn_fn_bytes = PREDICTOR_ROWS
        .checked_mul(hc)
        .and_then(|rows| rows.checked_mul(dim))
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| "mHC entry: predictor byte length overflows".to_string())?;
    let attn_norm_name = format!("layers.{layer}.attn_norm.weight");
    let attn_norm_bytes = match planned_dtype(request.resident_plan, &attn_norm_name) {
        Some("F32") => dim * 4,
        Some("BF16") => dim * 2,
        Some(other) => {
            return Err(format!(
                "mHC entry: {attn_norm_name} is planned as {other}; the attn_norm seam decodes F32 or BF16 only"
            ));
        }
        None => {
            return Err(format!(
                "mHC entry: the rank plan does not carry {attn_norm_name}, so its encoding is unknown; refusing to guess"
            ));
        }
    };
    let attn_fn = reader(&format!("layers.{layer}.hc_attn_fn"), hc_attn_fn_bytes)?;
    let attn_base = reader(&format!("layers.{layer}.hc_attn_base"), PREDICTOR_ROWS * 4)?;
    let attn_scale = reader(&format!("layers.{layer}.hc_attn_scale"), 3 * 4)?;
    let attn_norm = reader(&attn_norm_name, attn_norm_bytes)?;
    let ffn_norm_name = format!("layers.{layer}.ffn_norm.weight");
    let ffn_norm_bytes = match planned_dtype(request.resident_plan, &ffn_norm_name) {
        Some("F32") => dim * 4,
        Some("BF16") => dim * 2,
        Some(other) => {
            return Err(format!(
                "FFN seams: {ffn_norm_name} is planned as {other}; the ffn_norm seam decodes F32 or BF16 only"
            ));
        }
        None => {
            return Err(format!(
                "FFN seams: the rank plan does not carry {ffn_norm_name}, so its encoding is unknown; refusing to guess"
            ));
        }
    };
    let hc_ffn_fn_bytes = PREDICTOR_ROWS
        .checked_mul(hc)
        .and_then(|rows| rows.checked_mul(dim))
        .and_then(|values| values.checked_mul(4))
        .ok_or_else(|| "FFN seams: hc_ffn_fn byte length overflows".to_string())?;
    let ffn_fn = reader(&format!("layers.{layer}.hc_ffn_fn"), hc_ffn_fn_bytes)?;
    let ffn_base = reader(&format!("layers.{layer}.hc_ffn_base"), PREDICTOR_ROWS * 4)?;
    let ffn_scale = reader(&format!("layers.{layer}.hc_ffn_scale"), 3 * 4)?;
    let ffn_norm = reader(&ffn_norm_name, ffn_norm_bytes)?;
    Ok((
        LayerSeamRaw {
            attn_fn,
            attn_base,
            attn_scale,
            attn_norm,
            ffn_fn,
            ffn_base,
            ffn_scale,
            ffn_norm,
        },
        8,
    ))
}

/// Proves - on this machine, on this layer's real weights, before any position runs - that the
/// interleaved projection is the ordered projection bit for bit.
///
/// The two loops evaluate the same multiply-then-add chain per coefficient row, so they agree by
/// construction; what a new machine can change is the code the compiler emits for the transposed
/// pass (vector lanes instead of scalar adds). The seam cache therefore probes each staged predictor
/// once, off the step timer: it runs both projections on a fixed stream and compares the f32 bits of
/// every dot product and of the sum of squares, and refuses to stage a projection pair the ordered
/// pass would not reproduce.
#[cfg(any(feature = "cuda", test))]
fn verify_interleaved_projection(
    config: &sparkflash_engine::config::ModelConfig,
    ordered: &[f32],
    transposed: &[f32],
) -> Result<(), String> {
    let width = config.hc_width();
    let probe: Vec<f32> = (0..width)
        .map(|index| ((index % 251) as f32 - 125.0) * 0.007_812_5)
        .collect();
    let ordered_projection =
        sparkflash_engine::forward::mhc::hc_projection(config, &probe, ordered)
            .map_err(|error| format!("ordered projection: {error}"))?;
    let interleaved_projection =
        sparkflash_engine::forward::mhc::hc_projection_transposed(config, &probe, transposed)
            .map_err(|error| format!("interleaved projection: {error}"))?;
    let same = |left: &[f32], right: &[f32]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    };
    if !same(&ordered_projection.raw, &interleaved_projection.raw)
        || !same(
            &ordered_projection.sum_squares,
            &interleaved_projection.sum_squares,
        )
    {
        return Err(
            "the interleaved projection is not the ordered projection on this machine, so the \
             transposed predictor cannot be staged: the seam would not be bit-identical"
                .to_string(),
        );
    }
    Ok(())
}
/// Decodes one layer's raw seam bytes with the same codecs and f32 digests the per-call seam used.
///
/// # Errors
/// A norm byte length that is neither the F32 nor the BF16 row, or a codec length refusal.
#[cfg(any(feature = "cuda", test))]
fn decode_layer_seam_tensors(
    raw: LayerSeamRaw,
    config: &sparkflash_engine::config::ModelConfig,
) -> Result<LayerSeamTensors, String> {
    let dim = config.dim;
    let attn_fn = f32_from_le_bytes(&raw.attn_fn)?;
    let attn_base = f32_from_le_bytes(&raw.attn_base)?;
    let attn_scale = f32_from_le_bytes(&raw.attn_scale)?;
    let attn_norm = if raw.attn_norm.len() == dim * 4 {
        f32_from_le_bytes(&raw.attn_norm)?
    } else if raw.attn_norm.len() == dim * 2 {
        bf16_from_le_bytes(&raw.attn_norm)?
    } else {
        return Err(format!(
            "layer seam prewarm: attn_norm is {} bytes, expected {} (F32) or {} (BF16)",
            raw.attn_norm.len(),
            dim * 4,
            dim * 2
        ));
    };
    let ffn_fn = f32_from_le_bytes(&raw.ffn_fn)?;
    let ffn_base = f32_from_le_bytes(&raw.ffn_base)?;
    let ffn_scale = f32_from_le_bytes(&raw.ffn_scale)?;
    let ffn_norm = if raw.ffn_norm.len() == dim * 4 {
        f32_from_le_bytes(&raw.ffn_norm)?
    } else if raw.ffn_norm.len() == dim * 2 {
        bf16_from_le_bytes(&raw.ffn_norm)?
    } else {
        return Err(format!(
            "layer seam prewarm: ffn_norm is {} bytes, expected {} (F32) or {} (BF16)",
            raw.ffn_norm.len(),
            dim * 4,
            dim * 2
        ));
    };
    // The interleaved projection reads the weight transposed (mhc::hc_projection_transposed), and
    // this is the one place per layer where that copy is made: off the step timer, before the first
    // position. It moves the same f32 bits, so the seam's arithmetic is untouched.
    let mut attn_fn_transposed = vec![0.0f32; attn_fn.len()];
    sparkflash_engine::forward::mhc::hc_transpose_predictor(
        config,
        &attn_fn,
        &mut attn_fn_transposed,
    )
    .map_err(|error| format!("layer seam prewarm: attn predictor transpose: {error}"))?;
    verify_interleaved_projection(config, &attn_fn, &attn_fn_transposed)
        .map_err(|error| format!("layer seam prewarm: attn predictor: {error}"))?;
    let mut ffn_fn_transposed = vec![0.0f32; ffn_fn.len()];
    sparkflash_engine::forward::mhc::hc_transpose_predictor(
        config,
        &ffn_fn,
        &mut ffn_fn_transposed,
    )
    .map_err(|error| format!("layer seam prewarm: ffn predictor transpose: {error}"))?;
    verify_interleaved_projection(config, &ffn_fn, &ffn_fn_transposed)
        .map_err(|error| format!("layer seam prewarm: ffn predictor: {error}"))?;
    Ok(LayerSeamTensors {
        attn_predictor_sha256: evidence_f32_sha256(&attn_fn),
        attn_base_sha256: evidence_f32_sha256(&attn_base),
        attn_norm_sha256: evidence_f32_sha256(&attn_norm),
        attn_fn,
        attn_fn_transposed,
        attn_base,
        attn_scale,
        attn_norm,
        ffn_fn,
        ffn_fn_transposed,
        ffn_base,
        ffn_scale,
        ffn_norm,
    })
}

/// Builds the run-scope seam cache once from `reader`: eight tensors per layer, decoded and hashed
/// exactly as the per-call seam did. Ungated for the host tests; the CUDA build passes
/// read_arena_bytes as the reader.
///
/// # Errors
/// The first dtype/geometry refusal (same messages as the per-call seam), a codec failure, or an
/// overflowing byte total.
#[cfg(any(feature = "cuda", test))]
fn build_layer_seam_cache_with_reader(
    reader: &mut SeamReader<'_>,
    request: &TokenStageRequest<'_>,
    n_layers: usize,
) -> Result<LayerSeamCache, String> {
    let started = std::time::Instant::now();
    let config = request
        .config
        .ok_or_else(|| "mHC entry: the loop was built without the model config".to_string())?;
    let mut layers = Vec::with_capacity(n_layers);
    let mut reads = 0usize;
    let mut bytes = 0usize;
    for layer in 0..n_layers {
        let (raw, layer_reads) = read_layer_seam_raw(reader, layer, request)?;
        let tensors = decode_layer_seam_tensors(raw, config)?;
        bytes = bytes
            .checked_add(tensors.decoded_bytes())
            .ok_or_else(|| "layer seam prewarm: decoded byte total overflows".to_string())?;
        reads += layer_reads;
        layers.push(tensors);
    }
    Ok(LayerSeamCache {
        layers,
        reads,
        bytes,
        wall_ms: started.elapsed().as_millis() as u64,
    })
}

/// The CUDA seam-cache build: the host reader/decoder above with the live arena's
/// read_arena_bytes as the reader.
///
/// # Errors
/// build_layer_seam_cache_with_reader's refusals, or an arena read failure.
#[cfg(feature = "cuda")]
fn build_layer_seam_cache(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    arena: &sparkflash_device::arena::ResidentArena,
    request: &TokenStageRequest<'_>,
    n_layers: usize,
) -> Result<LayerSeamCache, String> {
    let mut reader = |name: &str, expected_bytes: usize| -> Result<Vec<u8>, String> {
        read_arena_bytes(cuda, arena, name, expected_bytes)
    };
    build_layer_seam_cache_with_reader(&mut reader, request, n_layers)
}

/// The little-endian bytes of an f32 row.
#[must_use]
pub fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The little-endian bytes of an f32 row rounded to bf16, round-to-nearest-even.
#[must_use]
pub fn bf16_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            let lsb = (bits >> 16) & 1;
            let rounded = bits.wrapping_add(0x7fff + lsb) & 0xffff_0000;
            ((rounded >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// The lowercase hex sha256 of a byte slice.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    // The perf probe counts every hash while it is installed; the default path pays one atomic
    // load and records nothing.
    #[cfg(feature = "cuda")]
    let timer = probe::counter_timer();
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut text, byte| {
            use std::fmt::Write as _;
            let _ = write!(text, "{byte:02x}");
            text
        });
    #[cfg(feature = "cuda")]
    probe::count_sha256(bytes.len(), timer);
    digest
}

/// One process's cubin image cache (M2/C1).
///
/// The four-rank layer loop reads and hashes the same eight cubin images on every layer. The cache
/// keys on the path and keeps the bytes as a shared `Arc`, so the read, the sha256 and the
/// module-cache pointer identity are established once per run. [reset_cubin_cache] clears it at run
/// scope, so two runs in one process cannot share a stale image.
/// A cubin image and the sha256 its pin records, shared for the life of one run.
#[cfg(feature = "cuda")]
type CubinImage = (std::sync::Arc<[u8]>, String);

#[cfg(feature = "cuda")]
static CUBIN_CACHE: std::sync::Mutex<Option<std::collections::HashMap<PathBuf, CubinImage>>> =
    std::sync::Mutex::new(None);

/// Reads one cubin image once per run and returns its bytes plus the sha256 the pins record.
///
/// The shared `Arc` matters twice: it removes the repeated read and hash, and it keeps the slice
/// address stable so the driver's module cache can hit its pointer fast path on every later layer.
///
/// # Errors
/// The file read failure, with the path in the message.
#[cfg(feature = "cuda")]
pub(crate) fn cached_cubin(path: &Path) -> Result<(std::sync::Arc<[u8]>, String), String> {
    let mut guard = CUBIN_CACHE
        .lock()
        .map_err(|error| format!("cubin cache lock: {error}"))?;
    let cache = guard.get_or_insert_with(std::collections::HashMap::new);
    if let Some((bytes, digest)) = cache.get(path) {
        return Ok((std::sync::Arc::clone(bytes), digest.clone()));
    }
    let bytes: std::sync::Arc<[u8]> = std::fs::read(path)
        .map_err(|error| format!("read {}: {error}", path.display()))?
        .into();
    let digest = sha256_hex(&bytes);
    cache.insert(
        path.to_path_buf(),
        (std::sync::Arc::clone(&bytes), digest.clone()),
    );
    Ok((bytes, digest))
}

/// Starts a fresh cubin cache at run scope.
#[cfg(feature = "cuda")]
pub(crate) fn reset_cubin_cache() {
    if let Ok(mut guard) = CUBIN_CACHE.lock() {
        *guard = Some(std::collections::HashMap::new());
    }
}

/// One run's loaded boundary-projection kernels, keyed by (cubin digest, symbol).
///
/// The two boundary projection blocks ([run_layer_attention_projections] and
/// [run_layer_attention_output]) load the same cubins on every layer: the pinned dense GEMV every
/// projected row runs through, and the bf16a wo_a kernel. The driver's module cache already keys a
/// module by the image's bytes and the symbol, so a reload is a pointer-keyed map lookup plus a
/// byte compare, two mutex acquisitions and a symbol allocation, with the pin verification on top;
/// the block's per-launch cuda.free is the bookkeeping counterpart. This cache is the run-scope
/// front of that table: the key is the pinned cubin digest, so a re-staged image is a different
/// entry and can never reuse another image's handle, and a symbol is only ever loaded under the
/// name its launch type resolves. Entries load on the first layer that needs them (or in
/// [KernelCache::prewarm], before the step timer) and are released once at teardown.
///
/// Identity: a hit returns the exact (module, function) the miss path loaded from the same digest.
/// The launch arguments, the cubin bytes and the symbol names are the ones the per-layer load used,
/// so a block computes the same values in the same order.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct KernelCache {
    entries: Vec<KernelCacheEntry>,
}

/// One cache entry: the digest it was loaded from, the symbol it resolved, and the handles that
/// keep the module and its function alive for the run.
#[cfg(feature = "cuda")]
struct KernelCacheEntry {
    digest: String,
    symbol: &'static str,
    module: sparkflash_kernels::cuda_driver::Module,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
}

#[cfg(feature = "cuda")]
impl KernelCache {
    /// Returns the cached kernel for (digest, symbol), loading it with the loader on the first miss.
    ///
    /// The loader returns the digest it loaded from, so the cache can refuse a mismatch: the digest
    /// in the key is the identity, and an entry produced from different bytes would mix images.
    /// The loader runs at most once per (digest, symbol) per run.
    fn kernel_for(
        &mut self,
        digest: &str,
        symbol: &'static str,
        load: impl FnOnce() -> Result<
            (
                String,
                sparkflash_kernels::cuda_driver::Module,
                sparkflash_kernels::cuda_driver::Kernel,
            ),
            String,
        >,
    ) -> Result<sparkflash_kernels::cuda_driver::Kernel, String> {
        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.symbol == symbol && entry.digest == digest)
        {
            return Ok(entry.kernel);
        }
        let (loaded_digest, module, kernel) = load()?;
        if loaded_digest != digest {
            return Err(format!(
                "kernel cache: {symbol} was loaded from digest {loaded_digest}, the key is {digest}"
            ));
        }
        self.entries.push(KernelCacheEntry {
            digest: loaded_digest,
            symbol,
            module,
            kernel,
        });
        Ok(kernel)
    }

    /// The bf16a wo_a projection kernel and its cubin digest, loaded once per run.
    ///
    /// The digest comes from the shared pinned image cache (a map hit, not a file read), so the
    /// evidence rows record the same sha256 as the per-layer path did. The wo_a pin is not
    /// re-verified here, exactly as before: that block already loaded this cubin by digest only.
    fn wo_a(
        &mut self,
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        kernel_dir: &Path,
        context: &str,
    ) -> Result<(sparkflash_kernels::cuda_driver::Kernel, String), String> {
        use sparkflash_kernels::bf16a_mxfp8w_wo_a_launch as wo_a_launch;

        let (cubin, digest) = cached_cubin(&kernel_dir.join(format!("{NATIVE_WO_A_KERNEL}.cubin")))
            .map_err(|error| format!("{context}: read wo_a cubin: {error}"))?;
        let kernel = self.kernel_for(&digest, wo_a_launch::KERNEL, || {
            let (module, kernel) = wo_a_launch::load(cuda, &cubin)?;
            Ok((digest.clone(), module, kernel))
        })?;
        Ok((kernel, digest))
    }

    /// The pinned dense GEMV kernel and its cubin digest, loaded once per run.
    ///
    /// The pin is verified on the load, once: the loader reads the pinned addition (which compares
    /// the staged sha256 with the baseline) and kernel_for refuses a digest that disagrees with the
    /// key.
    fn gemv(
        &mut self,
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        kernel_dir: &Path,
        context: &str,
    ) -> Result<(sparkflash_kernels::cuda_driver::Kernel, String), String> {
        use sparkflash_kernels::mxfp8_gemv_launch;

        let cubin_name = format!("{}.cubin", mxfp8_gemv_launch::KERNEL);
        // The per-block read resolves the digest from the pinned image cache; the miss path below
        // re-reads the same bytes through the pinned addition so the baseline check runs once.
        let (_cubin, digest) = cached_cubin(&kernel_dir.join(&cubin_name))
            .map_err(|error| format!("{context}: read {cubin_name}: {error}"))?;
        let kernel = self.kernel_for(&digest, mxfp8_gemv_launch::KERNEL, || {
            let (pinned_cubin, pinned_digest) =
                load_pinned_addition(kernel_dir, mxfp8_gemv_launch::KERNEL, context)?;
            let (module, kernel) = mxfp8_gemv_launch::load(cuda, &pinned_cubin)?;
            Ok((pinned_digest, module, kernel))
        })?;
        Ok((kernel, digest))
    }

    /// Loads both projection kernels before the step timer, so no layer pays the first load.
    fn prewarm(
        &mut self,
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        kernel_dir: &Path,
    ) -> Result<(), String> {
        self.wo_a(cuda, kernel_dir, "kernel cache prewarm")?;
        self.gemv(cuda, kernel_dir, "kernel cache prewarm")?;
        Ok(())
    }

    /// Releases every module once, at teardown.
    fn free_all(&mut self, cuda: &sparkflash_kernels::cuda_driver::Cuda) {
        for entry in self.entries.drain(..) {
            cuda.free(entry.module, &[]);
        }
    }
}

// The evidence switch is thread-local, not process-wide, for the same reason the probe's state is:
// one forward loop runs on one thread, and a unit test must be able to switch the chain off without
// racing every other test in the binary. Unit-test threads build with the chain on so the pre-M2
// evidence assertions stay meaningful; the shipped default is off, and [run_forward_loop] always
// selects the run's mode explicitly from `SPARKFLASH_FORWARD_EVIDENCE` before the first launch.
thread_local! {
    static EVIDENCE_RECORD: std::cell::Cell<bool> = const { std::cell::Cell::new(cfg!(test)) };
    static EVIDENCE_HASH_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The evidence opt-in spelling: `SPARKFLASH_FORWARD_EVIDENCE=1` (or `true` / `yes`).
#[must_use]
pub fn evidence_enabled_from_env(env: Option<&str>) -> bool {
    matches!(env, Some("1" | "true" | "yes"))
}

/// Turns the evidence chain on or off and returns the value in force.
pub fn set_evidence_record(enabled: bool) -> bool {
    EVIDENCE_RECORD.with(|record| record.set(enabled));
    enabled
}

/// Whether the per-layer evidence chain (layer JSON and its data-dependent digests) is recorded.
#[must_use]
pub fn evidence_record_enabled() -> bool {
    EVIDENCE_RECORD.with(std::cell::Cell::get)
}

/// Selects the evidence mode from the environment at run scope.
pub fn enable_forward_evidence_from_env() -> bool {
    set_evidence_record(evidence_enabled_from_env(
        std::env::var("SPARKFLASH_FORWARD_EVIDENCE").ok().as_deref(),
    ))
}

/// Whether a per-layer device readback exists only to feed the layer-chain evidence record.
///
/// C10 records the layer-chain entry (and every digest that feeds it) only in evidence mode, so a
/// default or `--forward-lean` run has no consumer for these bytes. The reads are skipped rather
/// than issued and dropped: every `cuMemcpyDtoH` is a stream-ordered drain, so an unread copy is a
/// sync point that puts the host seam behind the layer's device launches for nothing.
#[must_use]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn layer_evidence_readbacks() -> bool {
    evidence_record_enabled()
}

/// The number of evidence digests actually taken on this thread (test hook).
#[cfg(test)]
#[must_use]
pub(crate) fn evidence_hash_calls() -> u64 {
    EVIDENCE_HASH_CALLS.with(std::cell::Cell::get)
}

/// A sha256 that exists only for the evidence chain: empty when the run is not recording evidence.
///
/// Every per-layer/per-position data-dependent digest goes through this, so a default or
/// --forward-lean run hashes nothing on the layer path; the probe's sha256 call counter is the
/// proof. With the evidence flag on it is exactly [sha256_hex], so the identity kit's chain JSON is
/// byte-identical to the pre-M2 evidence-bearing path.
#[must_use]
pub(crate) fn evidence_sha256(bytes: &[u8]) -> String {
    if evidence_record_enabled() {
        EVIDENCE_HASH_CALLS.with(|calls| calls.set(calls.get() + 1));
        sha256_hex(bytes)
    } else {
        String::new()
    }
}

/// [evidence_sha256] of an f32 row's little-endian bytes, with the byte materialisation behind
/// the same switch.
///
/// The per-layer digests are recorded only in evidence mode, but the calling seam used to serialise
/// its row before [evidence_sha256] could return the empty string: one allocation and one copy per
/// f32 buffer per seam per layer, i.e. host work with no consumer in a default or --forward-lean
/// run. This helper checks the flag first, so the lean path allocates nothing and the evidence path
/// hashes the identical bytes.
#[must_use]
pub(crate) fn evidence_f32_sha256(values: &[f32]) -> String {
    if evidence_record_enabled() {
        evidence_sha256(&f32_le_bytes(values))
    } else {
        String::new()
    }
}

/// Quantises a W8A8 activation the way the released forward does (model.py:27-30, 182-205): the
/// value is a BF16 module output, so the f32 host value is narrowed to BF16 first and then
/// quantised to E4M3 codes with one ue8m0 scale per 32-element block (D23's amax floor).
///
/// The narrowing is idempotent on a value that is already BF16-exact (a device GEMM output read
/// back as f32), so it is safe at every W8A8 activation seam. This mirrors the single-node native
/// chain's quantise_native_activation (crates/sparkflash-cli/src/attention_device.rs) exactly, so
/// the four-rank loop and that chain feed the same codes and scales to the same kernel.
///
/// # Errors
/// Propagates the quantiser's refusal when the value count is not a whole number of 32-blocks.
pub fn quantise_native_activation(
    values: &[f32],
    what: &str,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    // One pass: the BF16 narrowing, the per-block ue8m0 scale and the E4M3 codes are computed
    // together, and the codes come from the bit-arithmetic nearest-E4M3 encoder instead of the
    // reference's bisection over the 127 finite codes. The bytes are the two-step path's (narrow,
    // then quantize_activation_fp8); the engine's tests pin the encoder against the bisection and
    // the fused quantiser against that composition, and this function's own test pins the wiring.
    let quantised = sparkflash_engine::quant::quantize_activation_fp8_narrowed(values)
        .map_err(|error| format!("{what}: activation quantiser: {error}"))?;
    Ok((quantised.codes().to_vec(), quantised.scales().to_vec()))
}

/// Where the rank-0 native dense launch-parity block captures (--forward-native-dense-parity).
///
/// The block is a D41 attribution diagnostic: it re-runs the single-node native chain's launch
/// construction on byte-identical activation codes/scales and B12x weight bytes and compares the
/// bf16 output with the four-rank loop's production output, so a distribution gap can be attributed
/// to the MoE/mHC rounding arms instead of the dense launch. Absent from the argv, the code never
/// runs and the run is byte-identical to one without this mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDenseParitySpec {
    /// The layer to capture.
    pub layer: usize,
    /// The position to capture; None resolves to the first decode row (prompt_len - 1).
    pub position: Option<usize>,
}

impl NativeDenseParitySpec {
    /// Parses the LAYER[:POSITION] value.
    ///
    /// # Errors
    /// A term that is not a usize, or an empty value.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("--forward-native-dense-parity needs LAYER or LAYER:POSITION".to_string());
        }
        let parse = |text: &str, what: &str| -> Result<usize, String> {
            text.trim().parse::<usize>().map_err(|_| {
                format!(
                    "--forward-native-dense-parity {raw}: {what} {:?} is not a usize",
                    text.trim()
                )
            })
        };
        match trimmed.split_once(':') {
            Some((layer, position)) => Ok(Self {
                layer: parse(layer, "layer")?,
                position: Some(parse(position, "position")?),
            }),
            None => Ok(Self {
                layer: parse(trimmed, "layer")?,
                position: None,
            }),
        }
    }
}

/// The rank-0 native dense launch-parity report runner (see [NativeDenseParitySpec]).
///
/// One-shot and diagnostic-only: it captures the four native dense operands of one layer/position
/// and writes tp4-native-dense-parity-r{rank}.json. It never changes the loop's arithmetic or its
/// existing evidence, and it contains its own failures in the report (a parity error cannot fail
/// the token run).
#[cfg(feature = "cuda")]
#[derive(Debug, Default)]
pub struct NativeDenseParity {
    armed: Option<(usize, usize)>,
    entries: Vec<serde_json::Value>,
    finished: bool,
}

/// Counts the differing bytes between two payloads and whether their lengths differ.
#[cfg(feature = "cuda")]
fn native_parity_byte_mismatches(left: &[u8], right: &[u8]) -> (usize, bool) {
    let common = left.len().min(right.len());
    let mismatches = (0..common)
        .filter(|index| left[*index] != right[*index])
        .count();
    (mismatches, left.len() != right.len())
}

/// The f32 value of a bf16 word, for the parity comparison's max_abs.
#[cfg(feature = "cuda")]
fn native_parity_bf16_word_f32(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

/// Compares two bf16 word rows: (mismatches, slots, max_abs, length_mismatch).
#[cfg(feature = "cuda")]
fn native_parity_compare_bf16(expected: &[u8], actual: &[u8]) -> (usize, usize, f32, bool) {
    let slots = expected.len().min(actual.len()) / 2;
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    for index in 0..slots {
        let expected = u16::from_le_bytes([expected[index * 2], expected[index * 2 + 1]]);
        let actual = u16::from_le_bytes([actual[index * 2], actual[index * 2 + 1]]);
        if expected != actual {
            mismatches += 1;
            let diff =
                (native_parity_bf16_word_f32(expected) - native_parity_bf16_word_f32(actual)).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
    }
    (mismatches, slots, max_abs, expected.len() != actual.len())
}

/// Derives the dense launch geometry implied by the operand bytes: one output row, inner from the
/// activation code count, cols from the B12x weight scale-grid size, and the weight payload must
/// then be exactly the B12x buffer for that shape.
///
/// The derivation uses only the operands the launch consumes, not the loop's own shape variables, so
/// a swapped or wrong cols/inner in the production launch cannot reproduce the production output
/// through the parity re-launch.
#[cfg(feature = "cuda")]
fn derive_native_dense_geometry(
    activation_code_bytes: usize,
    activation_scale_bytes: usize,
    weight_payload_bytes: usize,
    weight_scale_bytes: usize,
) -> Result<(usize, usize), String> {
    let inner = activation_code_bytes;
    if inner == 0 || !inner.is_multiple_of(32) {
        return Err(format!(
            "activation codes {inner} are not a whole number of 32-value blocks"
        ));
    }
    let inner_blocks = inner / 32;
    if activation_scale_bytes != inner_blocks {
        return Err(format!(
            "activation scales {activation_scale_bytes} are not one per 32 of {inner} codes"
        ));
    }
    if weight_scale_bytes == 0 || !weight_scale_bytes.is_multiple_of(inner_blocks) {
        return Err(format!(
            "weight scales {weight_scale_bytes} are not whole {inner}-value K blocks"
        ));
    }
    let cols = (weight_scale_bytes / inner_blocks) * 32;
    let expected = sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes(cols, inner);
    if weight_payload_bytes != expected {
        return Err(format!(
            "weight payload {weight_payload_bytes} B is not the B12x buffer for {cols}x{inner} ({expected} B)"
        ));
    }
    Ok((cols, inner))
}

#[cfg(feature = "cuda")]
impl NativeDenseParity {
    /// The runner for a run: armed at spec.layer/spec.position (the first decode row when the
    /// position is omitted) on rank 0, or off.
    #[must_use]
    pub fn new(spec: Option<NativeDenseParitySpec>, first_position: usize) -> Self {
        match spec {
            Some(spec) => Self {
                armed: Some((spec.layer, spec.position.unwrap_or(first_position))),
                entries: Vec::new(),
                finished: false,
            },
            None => Self::default(),
        }
    }

    /// Whether this stage should capture.
    fn armed_at(&self, rank: usize, layer: usize, position: usize) -> bool {
        !self.finished && self.armed == Some((layer, position)) && rank == 0
    }

    /// Captures one native dense operand at the armed layer/position.
    #[allow(clippy::too_many_arguments)]
    pub fn capture(
        &mut self,
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        request: &TokenStageRequest<'_>,
        layer: usize,
        position: usize,
        stage: &str,
        operand: &str,
        weight_name: &str,
        claimed: (usize, usize),
        activation: &[f32],
        codes: &[u8],
        scales: &[u8],
        weight: Deviceptr,
        weight_scales: Deviceptr,
        out_bf16: &[u8],
    ) {
        if !self.armed_at(request.rank, layer, position) {
            return;
        }
        let started = std::time::Instant::now();
        let checks = native_dense_parity_checks(
            cuda,
            request,
            layer,
            operand,
            weight_name,
            claimed,
            activation,
            codes,
            scales,
            weight,
            weight_scales,
            out_bf16,
        );
        let mut entry = serde_json::json!({
            "stage": stage,
            "operand": operand,
            "weight": weight_name,
            "kernel": NATIVE_DENSE_KERNEL,
            "activation": {
                "f32_sha256": sha256_hex(&f32_le_bytes(activation)),
                "f32_values": activation.len(),
                "codes_sha256": sha256_hex(codes),
                "scales_sha256": sha256_hex(scales),
                "code_bytes": codes.len(),
                "scale_bytes": scales.len(),
            },
            "production_output": {
                "bf16_sha256": sha256_hex(out_bf16),
                "bytes": out_bf16.len(),
            },
        });
        match checks {
            Ok(serde_json::Value::Object(map)) => {
                for (key, value) in map {
                    entry[key] = value;
                }
            }
            Ok(other) => {
                entry["checks"] = other;
            }
            Err(error) => {
                entry["error"] = serde_json::json!(error);
            }
        }
        entry["elapsed_ms"] = serde_json::json!(started.elapsed().as_secs_f64() * 1000.0);
        self.entries.push(entry);
    }

    /// Writes the report once the armed stage has captured, best-effort: a diagnostic write failure
    /// never fails the token run.
    pub fn finish(&mut self, request: &TokenStageRequest<'_>) {
        let Some((layer, position)) = self.armed else {
            return;
        };
        if self.finished || self.entries.is_empty() {
            return;
        }
        let path = request
            .evidence_dir
            .join(format!("tp4-native-dense-parity-r{}.json", request.rank));
        let cubin_sha256 = request
            .kernel_cubin_dir
            .and_then(|dir| fs::read(dir.join(format!("{NATIVE_DENSE_KERNEL}.cubin"))).ok())
            .map(|bytes| sha256_hex(&bytes));
        let status = if self
            .entries
            .iter()
            .any(|entry| entry.get("error").is_some())
        {
            "error"
        } else if self.entries.len() < 4 {
            "partial"
        } else {
            "ok"
        };
        let report = serde_json::json!({
            "schema": "sparkflash-tp4-native-dense-parity/1",
            "stage": "rank0-native-dense-launch-parity",
            "rank": request.rank,
            "world": request.world,
            "layer": layer,
            "position": position,
            "kernel": NATIVE_DENSE_KERNEL,
            "cubin_sha256": cubin_sha256,
            "status": status,
            "note": "One-shot, rank-0 diagnostic (default off). Each operand re-runs the single-node chain's dense launch construction (Mxfp8DenseParams::new with a zero trace ring, step_seq 0, then mxfp8_dense_launch::launch) on the loop's exact activation codes/scales and the arena's exact B12x weight bytes, compares the bf16 words with the loop's production output, and runs sparkflash_kernels::mxfp8_ref::dense_gemm on the same bytes. wq_a and wkv are replicated at TP4 so their comparison is the literal loop-vs-chain launch; wq_b is a row shard and wo_b an inner shard, so their shard launches are compared on the shard bytes, and no world-1 full-inner bit-equality is claimed for wo_b: the f32 cross-rank reduce order is a designed TP4 non-associativity.",
            "operands": serde_json::Value::Array(self.entries.clone()),
        });
        let outcome = fs::create_dir_all(request.evidence_dir)
            .map_err(|error| format!("create evidence dir: {error}"))
            .and_then(|()| {
                let text = serde_json::to_string_pretty(&report)
                    .map_err(|error| format!("serialize: {error}"))?;
                fs::write(&path, text + "\n")
                    .map_err(|error| format!("write {}: {error}", path.display()))
            });
        match outcome {
            Ok(()) => self.finished = true,
            Err(error) => eprintln!("native dense parity: {error}"),
        }
    }
}

/// Runs the three parity checks for one native dense operand: the chain's host quantiser on the
/// captured f32 values, the chain's device launch construction on the captured codes/scales and the
/// arena weight bytes, and the reference twin on the same bytes.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn native_dense_parity_checks(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    request: &TokenStageRequest<'_>,
    layer: usize,
    operand: &str,
    weight_name: &str,
    claimed: (usize, usize),
    activation: &[f32],
    codes: &[u8],
    scales: &[u8],
    weight: Deviceptr,
    weight_scales: Deviceptr,
    out_bf16: &[u8],
) -> Result<serde_json::Value, String> {
    use sparkflash_kernels::mxfp8_dense_launch;

    let plan = request
        .resident_plan
        .ok_or_else(|| "the loop has no rank plan".to_string())?;
    let (plan_cols, plan_inner) = planned_b12x_geometry(plan, weight_name)?;
    let payload_bytes = sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes(plan_cols, plan_inner);
    let scale_bytes = sparkflash_kernels::mxfp8_ref::b12x_scale_bytes(plan_cols, plan_inner);
    let mut weight_raw = vec![0u8; payload_bytes];
    cuda.copy_dtoh(&mut weight_raw, &weight)
        .map_err(|error| format!("weight payload dtoh: {error}"))?;
    let mut weight_scales_raw = vec![0u8; scale_bytes];
    cuda.copy_dtoh(&mut weight_scales_raw, &weight_scales)
        .map_err(|error| format!("weight scale dtoh: {error}"))?;
    let derived =
        derive_native_dense_geometry(codes.len(), scales.len(), payload_bytes, scale_bytes);

    let chain_quantiser = crate::attention_device::quantise_native_activation(activation, operand)
        .map(|(chain_codes, chain_scales)| {
            let (code_mismatches, code_length_mismatch) =
                native_parity_byte_mismatches(codes, &chain_codes);
            let (scale_mismatches, scale_length_mismatch) =
                native_parity_byte_mismatches(scales, &chain_scales);
            serde_json::json!({
                "fn": "attention_device::quantise_native_activation (bf16_round then quantize_activation_fp8, the single-node chain's activation preparation)",
                "codes_sha256": sha256_hex(&chain_codes),
                "scales_sha256": sha256_hex(&chain_scales),
                "code_mismatches": code_mismatches,
                "code_length_mismatch": code_length_mismatch,
                "scale_mismatches": scale_mismatches,
                "scale_length_mismatch": scale_length_mismatch,
                "bit_equal": code_mismatches == 0
                    && !code_length_mismatch
                    && scale_mismatches == 0
                    && !scale_length_mismatch,
            })
        })
        .map_err(|error| format!("chain quantiser: {error}"))?;

    let kernel_dir = request
        .kernel_cubin_dir
        .ok_or_else(|| "the loop has no kernel cubin directory".to_string())?;
    let cubin = fs::read(kernel_dir.join(format!("{NATIVE_DENSE_KERNEL}.cubin")))
        .map_err(|error| format!("read dense cubin: {error}"))?;
    let cubin_sha256 = sha256_hex(&cubin);
    let (module, kernel) = mxfp8_dense_launch::load(cuda, &cubin)?;
    let mut allocated = 0usize;
    let a_codes = cuda
        .alloc(codes.len(), &mut allocated)
        .map_err(|error| format!("parity activation code buffer: {error}"))?;
    let a_scales = cuda
        .alloc(scales.len(), &mut allocated)
        .map_err(|error| format!("parity activation scale buffer: {error}"))?;
    let out = cuda
        .alloc(plan_cols * 2, &mut allocated)
        .map_err(|error| format!("parity output buffer: {error}"))?;
    let launched = (|| -> Result<Vec<u8>, String> {
        cuda.copy_htod(&a_codes, codes)
            .map_err(|error| format!("parity code htod: {error}"))?;
        cuda.copy_htod(&a_scales, scales)
            .map_err(|error| format!("parity scale htod: {error}"))?;
        // The single-node chain's dense() construction: no trace ring (0), step_seq 0, and the
        // launch's layer id. The loop passes a zeroed ring instead; its active_scope_mask is 0, so
        // the kernel's trace path is off in both.
        let params = mxfp8_dense_launch::Mxfp8DenseParams::new(
            a_codes,
            a_scales,
            weight,
            weight_scales,
            out,
            1,
            plan_cols as u32,
            plan_inner as u32,
            0,
            layer as u32,
            0,
        );
        mxfp8_dense_launch::launch(cuda, kernel, &params)?;
        let mut chain_out = vec![0u8; plan_cols * 2];
        cuda.copy_dtoh(&mut chain_out, &out)
            .map_err(|error| format!("parity output dtoh: {error}"))?;
        Ok(chain_out)
    })();
    cuda.free_buffers(&[&a_codes, &a_scales, &out]);
    cuda.free(module, &[]);
    let chain_out = launched?;
    let (chain_mismatches, chain_slots, chain_max_abs, chain_length_mismatch) =
        native_parity_compare_bf16(&chain_out, out_bf16);
    let chain_launch = serde_json::json!({
        "construction": "Mxfp8DenseParams::new(a_fp8, a_scales, w_fp8, w_scales, out, rows=1, cols, inner, trace_ring=0, layer_id, step_seq=0) + mxfp8_dense_launch::launch (the single-node chain's dense() construction, crates/sparkflash-cli/src/attention_device.rs)",
        "trace_ring": "0 (the chain's value); the loop passes a zeroed 64-B ring whose active_scope_mask is 0, so the trace path is off in both",
        "cubin_sha256": cubin_sha256,
        "out_sha256": sha256_hex(&chain_out),
        "mismatches": chain_mismatches,
        "slots": chain_slots,
        "max_abs": chain_max_abs,
        "length_mismatch": chain_length_mismatch,
        "bit_equal": chain_mismatches == 0 && !chain_length_mismatch,
    });

    // The twin's asserts are a hard panic on a short buffer; a malformed capture must be reported
    // instead of killing the window's run, so the length precondition is checked here.
    let reference_twin = if codes.len() < plan_inner || scales.len() < plan_inner / 32 {
        serde_json::json!({
            "error": format!(
                "the captured activation is shorter than the plan geometry ({} codes, {} scales; the twin needs at least {plan_inner} and {})",
                codes.len(),
                scales.len(),
                plan_inner / 32
            ),
        })
    } else {
        let twin = sparkflash_kernels::mxfp8_ref::dense_gemm(
            codes,
            scales,
            &weight_raw,
            &weight_scales_raw,
            1,
            plan_cols,
            plan_inner,
        );
        let twin_raw: Vec<u8> = twin.iter().flat_map(|word| word.to_le_bytes()).collect();
        let (twin_mismatches, twin_slots, twin_max_abs, twin_length_mismatch) =
            native_parity_compare_bf16(&twin_raw, out_bf16);
        serde_json::json!({
            "fn": "sparkflash_kernels::mxfp8_ref::dense_gemm(codes, scales, the arena's B12x weight bytes, the arena's scale bytes, rows=1, cols, inner)",
            "out_sha256": sha256_hex(&twin_raw),
            "mismatches": twin_mismatches,
            "slots": twin_slots,
            "max_abs": twin_max_abs,
            "length_mismatch": twin_length_mismatch,
            "bit_equal": twin_mismatches == 0 && !twin_length_mismatch,
        })
    };

    Ok(serde_json::json!({
        "geometry": {
            "plan": {
                "rows": 1,
                "cols": plan_cols,
                "inner": plan_inner,
                "weight_payload_bytes": payload_bytes,
                "weight_scale_bytes": scale_bytes,
            },
            "claimed": {"rows": 1, "cols": claimed.0, "inner": claimed.1},
            "claimed_matches_plan": claimed == (plan_cols, plan_inner),
            "bytes_derived": match &derived {
                Ok((cols, inner)) => serde_json::json!({"rows": 1, "cols": cols, "inner": inner}),
                Err(_) => serde_json::Value::Null,
            },
            "bytes_derivation_error": match &derived {
                Ok(_) => serde_json::Value::Null,
                Err(error) => serde_json::json!(error),
            },
            "bytes_match_plan": derived.as_ref().is_ok_and(|pair| *pair == (plan_cols, plan_inner)),
            "weight_payload_sha256": sha256_hex(&weight_raw),
            "weight_scales_sha256": sha256_hex(&weight_scales_raw),
        },
        "chain_quantiser": chain_quantiser,
        "chain_launch": chain_launch,
        "reference_twin": reference_twin,
        "world_1_claim": if operand == "wo_b" {
            serde_json::json!("none: wo_b's inner axis is TP4-sharded, so this compares the rank's shard launch against the chain construction on the same shard bytes; a world-1 full-inner launch cannot be bit-equal by construction (f32 cross-rank reduce order differs)")
        } else {
            serde_json::Value::Null
        },
    }))
}

/// The per-rank row count of the released plan's routed down projection (`w2` output rows).
pub const DEFAULT_COMBINE_ROWS: usize = 5120;

/// One rank's forward/token stage request.
#[derive(Debug, Clone)]
pub struct TokenStageRequest<'a> {
    /// This rank.
    pub rank: usize,
    /// The world size (4).
    pub world: usize,
    /// Layers in the loop (40 on the released plan).
    pub n_layers: usize,
    /// Decode positions requested (`--decode-steps`); the loop advances at most this many.
    pub positions: usize,
    /// Where the token envelope and dump land.
    pub evidence_dir: &'a Path,
    /// The staged kernel cubins (`--kernel-cubin-dir`); `None` means nothing was staged and the
    /// combine preflight refuses by name.
    pub kernel_cubin_dir: Option<&'a Path>,
    /// Whether the argv carried [`PREFLIGHT_FLAG`].
    pub preflight: bool,
    /// Output rows per rank of the routed down projection.
    pub combine_rows: usize,
    /// The bind's own arena slice bytes for `w2`: `(weight, scale)`, when an arena was bound.
    pub combine_bytes: Option<(usize, usize)>,
    /// The prompt/decode token ids the run would start from.
    pub prompt_tokens: &'a [u32],
    /// The teacher-forced reference token per dumped position (QA's like-for-like vector). Empty
    /// means no reference vector was supplied; otherwise it must have one id per decode position.
    pub reference_ids: &'a [u32],
    /// The rank plan, when the preflight is built by the chain: the census derives the expected
    /// attention operand layouts from its dtypes instead of guessing from bind pointers.
    pub resident_plan: Option<&'a sparkflash_device::resident::ResidentPlan>,
    /// The released model geometry, when the preflight is built by the chain: the layer-0 mHC host
    /// seam reads its dimensions, hc stream count and norm epsilon from here.
    pub config: Option<&'a sparkflash_engine::config::ModelConfig>,
    /// The checkpoint directory the reference attention seams read: the compressor, indexer and
    /// candidate select run the reference's own host code on the device chain's per-layer input, and
    /// those tensors are read from the shards rather than through the arena bind.
    pub checkpoint_dir: Option<&'a Path>,
    /// The safetensors index file; absent means checkpoint_dir/model.safetensors.index.json.
    pub checkpoint_index: Option<&'a Path>,
    /// The rank plan's checkpoint source, when the run opened one. The released model writes Engram
    /// into the residual stream at layers 1 and 14, and those tables are 98 GB each: the seam pages
    /// them through this range-capable source instead of the arena. A config with Engram layers and
    /// no source is a refusal, not a silent skip.
    pub weights: Option<&'a sparkflash_engine::safetensors::SafeTensorsSource>,
    /// Run only this many forward positions from the start (the smoke shortcut); absent
    /// means the full prompt plus the decode rows.
    pub prefill_limit: Option<usize>,
    /// The native-rounding switches for this run: [NativeRounding::NONE] is the f32 arm and selects
    /// exactly the existing arithmetic, and [NativeRounding::any] is recorded in the evidence.
    pub native_rounding: NativeRounding,
    /// The rank-0 native dense launch-parity capture (--forward-native-dense-parity); None means the
    /// diagnostic is off and the run is untouched.
    pub native_dense_parity: Option<NativeDenseParitySpec>,
    /// The default-off M2 lean measurement switch.
    pub forward_perf: ForwardPerf,
}

/// What the token stage produced: a run, or a refusal envelope with its named reason.
#[derive(Debug, Clone)]
pub enum TokenStageOutcome {
    /// The loop executed and produced the envelope's values.
    Ran(TokenEnvelope),
    /// Nothing executed; the envelope names the missing precondition.
    Refused(TokenEnvelope),
}

impl TokenStageOutcome {
    /// The envelope either way.
    #[must_use]
    pub fn envelope(&self) -> &TokenEnvelope {
        match self {
            Self::Ran(envelope) | Self::Refused(envelope) => envelope,
        }
    }

    /// Whether the stage refuses.
    #[must_use]
    pub fn is_refused(&self) -> bool {
        matches!(self, Self::Refused(_))
    }

    /// The refusal reason, or `None` for a run.
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        match self {
            Self::Ran(_) => None,
            Self::Refused(envelope) => envelope.error.as_deref(),
        }
    }

    /// Writes the envelope into `dir` and returns the path.
    ///
    /// # Errors
    /// The envelope's shape rules, or a write failure.
    pub fn write(&self, dir: &Path) -> Result<PathBuf, String> {
        let envelope = self.envelope();
        envelope.validate()?;
        envelope.write(dir)
    }
}

/// Builds the refusal outcome for a request that cannot launch, with the stage's own reason.
fn refused(request: &TokenStageRequest<'_>, error: String) -> TokenStageOutcome {
    let mut envelope = TokenEnvelope::refused(request.rank, request.world, error);
    envelope.launch_census = TokenEnvelope::refusal_census(
        "combine preflight refused before any routed expert launched",
    );
    TokenStageOutcome::Refused(envelope)
}

/// Wraps a named refusal reason into the envelope shape without running any preflight.
#[must_use]
pub fn refused_outcome(request: &TokenStageRequest<'_>, message: String) -> TokenStageOutcome {
    refused(request, message)
}

/// The preflight's output: the validated geometry, the selected combine kernel, and the evidence
/// row the envelope records.
#[derive(Debug, Clone)]
pub struct ForwardPlan {
    /// The routed down-projection row geometry derived from the bind.
    pub geometry: CombineGeometry,
    /// The combine kernel choice (always a launchable stored kernel when this exists).
    pub selection: CombineSelection,
    /// The launch census row for the envelope.
    pub launch_census: serde_json::Value,
    /// The plan's own dtype for each layer-0 attention operand, so the loop census can corroborate
    /// the bind instead of inferring layout from pointers alone.
    pub attention_plan: Vec<AttentionOperandPlan>,
    /// The full-plan attention operand census (all 43 attention layers, plan-only).
    pub operand_census: serde_json::Value,
}

/// The host-only half of the token stage: interlock, geometry and combine preflight.
///
/// Returns the plan when every host prerequisite holds, or the refusal outcome to write. This is
/// what the chain calls *before* the arena boot: a missing stored kernel must fail in seconds, not
/// after a 20-minute 72 GiB upload.
#[allow(clippy::result_large_err)]
// The Err variant is the full refusal envelope: it is what the stage writes to disk, not a
// diagnostic that callers discard, so boxing it would only add an allocation on the fail path.
pub fn preflight(request: &TokenStageRequest<'_>) -> Result<ForwardPlan, TokenStageOutcome> {
    if !request.preflight {
        return Err(refused(
            request,
            format!(
                "forward preflight flag absent: pass {PREFLIGHT_FLAG} to allow the four-rank loop to execute device kernels; --forward alone only proves the evidence path"
            ),
        ));
    }
    if request
        .config
        .is_some_and(|config| !config.engram_layer_ids.is_empty())
        && request.weights.is_none()
    {
        return Err(refused(
            request,
            "Engram seam: the released model writes Engram into the residual stream at layers 1 and 14, and the loop was built without a checkpoint source; those tables are 98 GB each and are paged through the source, so --checkpoint (or --tensor-index) is required".to_string(),
        ));
    }
    if !request.reference_ids.is_empty() && request.reference_ids.len() != request.positions.max(1)
    {
        return Err(refused(
            request,
            format!(
                "reference vector: {} reference token ids for {} decode positions; QA's like-for-like rule needs exactly one reference id per dumped row, and a missing row is a prefix mismatch",
                request.reference_ids.len(),
                request.positions.max(1)
            ),
        ));
    }
    let geometry = match request.combine_bytes {
        Some((weight, scale)) => {
            match CombineGeometry::from_shard_bytes(request.combine_rows, weight, scale) {
                Ok(geometry) => geometry,
                Err(error) => return Err(refused(request, error)),
            }
        }
        None => CombineGeometry::stored(request.combine_rows),
    };
    let kernels = request
        .kernel_cubin_dir
        .map_or_else(CombineKernelSet::absent, CombineKernelSet::scan);
    let selection = select_combine(geometry, &kernels);
    if let Some(refusal) = selection.refusal() {
        return Err(refused(request, refusal.to_string()));
    }
    let launch_census = selection.evidence(geometry);
    let attention_plan = request
        .resident_plan
        .map_or_else(Vec::new, attention_plan_from_plan);
    Ok(ForwardPlan {
        geometry,
        selection,
        launch_census,
        operand_census: request.resident_plan.map_or_else(
            || serde_json::json!({"plan": "absent"}),
            attention_operand_census,
        ),
        attention_plan,
    })
}

/// The stage's standalone entry point: preflight, then a refusal that names the missing arena.
///
/// The device loop cannot run from here because the arena pointers are only valid inside the bind
/// epoch; the chain calls [preflight] and then [run_with_arena] from inside
/// [crate::tp4_resident::bind_device_sets_in_arena_with]. This function exists for the argv
/// preflight path and for host tests.
#[must_use]
pub fn token_stage(request: &TokenStageRequest<'_>) -> TokenStageOutcome {
    match preflight(request) {
        Err(outcome) => outcome,
        Ok(plan) => refused(
            request,
            format!(
                "forward preflight passed but no bound arena was supplied: the loop must run inside the arena epoch (combine kernel {})",
                plan.selection.evidence(plan.geometry)["kernel"]
            ),
        ),
    }
}

/// Runs the forward loop with the live arena and bind: the only place the arena pointers are used.
///
/// The caller must hold the arena epoch open; [crate::tp4_resident::bind_device_sets_in_arena_with]
/// provides exactly that, and its closure is the only caller in the chain.
#[must_use]
pub(crate) fn run_with_arena(
    request: &TokenStageRequest<'_>,
    plan: &ForwardPlan,
    arena: &sparkflash_device::arena::ResidentArena,
    bind: &crate::tp4_resident::DeviceBind,
    transport: Option<&TcpTransport>,
) -> TokenStageOutcome {
    match run_forward_loop(request, plan, Some((arena, bind)), transport) {
        Ok(run) => {
            let positions_run = run.tokens.ids.len();
            if positions_run == 0 {
                return refused(
                    request,
                    "forward loop returned no positions: refusing to write a run envelope without a token"
                        .to_string(),
                );
            }
            TokenStageOutcome::Ran(TokenEnvelope::ran(
                request.rank,
                request.world,
                run.tokens,
                run.step_ms,
                run.wall_ms,
                REDUCE_CALLS_PER_STEP,
                run.dump,
                run.launch_census,
            ))
        }
        Err(error) => refused(request, error),
    }
}
/// One executed forward run's values, before they are wrapped into the envelope.
#[derive(Debug)]
pub struct ForwardRun {
    /// Consensus token evidence, one entry per position.
    pub tokens: Tokens,
    /// Measured step times.
    pub step_ms: StepMs,
    /// Per-step wall times.
    pub wall_ms: Vec<f64>,
    /// The dump pointer for the logits that produced the ids.
    pub dump: crate::tp4_token::LogitsDump,
    /// The combine launch census.
    pub launch_census: serde_json::Value,
}

/// The layer-0 attention projection latents read back from the two dense B12x launches.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct Layer0Projections {
    /// sha256 of the native dense cubin bytes the launch resolved.
    pub cubin_sha256: String,
    /// sha256 of the E4M3 activation codes the native dense GEMM consumed.
    pub activation_codes_sha256: String,
    /// sha256 of the ue8m0 activation scale codes, one per 32 values.
    pub activation_scales_sha256: String,
    /// The activation code byte count (one E4M3 byte per served value).
    pub activation_code_bytes: usize,
    /// The activation scale byte count (one E8M0 byte per 32 values).
    pub activation_scale_bytes: usize,
    /// The q_a latent (1280 BF16 values) digest.
    pub wq_a_latent_sha256: String,
    /// The kv latent (512 BF16 values) digest.
    pub wkv_latent_sha256: String,
    /// The q_a latent's L2 norm, a shape sanity check for the evidence.
    pub wq_a_l2: f32,
    /// The kv latent's L2 norm.
    pub wkv_l2: f32,
    /// The wq_a latent bytes (1280 BF16 values), the q_norm seam's input.
    pub wq_a_latent_bytes: Vec<u8>,
    /// The wkv latent bytes (512 BF16 values), the kv_norm seam's input.
    pub wkv_latent_bytes: Vec<u8>,
    /// The launches this stage issued (two dense projections, no attention body yet).
    pub launches: usize,
}

#[cfg(feature = "cuda")]
impl Layer0Projections {
    /// The evidence block for the refused run.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stage": "layer0-attention-projections",
            "kernel": sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
            "arithmetic_family": NATIVE_DENSE_KERNEL,
            "cubin_sha256": self.cubin_sha256,
            "launches": self.launches,
            "rows": 1,
            "wq_a": {
                "cols": 1280,
                "inner": HIDDEN,
                "out_bytes": 1280 * 2,
                "latent_bf16_sha256": self.wq_a_latent_sha256,
                "latent_l2": self.wq_a_l2,
            },
            "wkv": {
                "cols": 512,
                "inner": HIDDEN,
                "out_bytes": 512 * 2,
                "latent_bf16_sha256": self.wkv_latent_sha256,
                "latent_l2": self.wkv_l2,
            },
            "activation": "the mHC seam's normed attention input, narrowed to BF16 then quantised to E4M3 with ue8m0 scales (the released W8A8 rule, D39 rule 1)",
            "activation_quant": {
                "rule": "BF16 module output narrowed to BF16, then E4M3 codes with one ue8m0 scale per 32 contiguous values: 2^ceil(log2(amax/448)) after D23's 1e-4 amax floor, f32 accumulate",
                "block": 32,
                "rows": 1,
                "inner": HIDDEN,
                "codes_sha256": self.activation_codes_sha256,
                "scales_sha256": self.activation_scales_sha256,
                "code_bytes": self.activation_code_bytes,
                "scale_bytes": self.activation_scale_bytes,
            },
            "weights": "arena B12x weights and scale grids bound by the plan-dtype census (no repack, no H2D)",
            "exact": "the native dense family: E4M3 activations with ue8m0 block-32 scales against FP8 E4M3 B12x weights, f32 accumulate, bf16 output (D39 rule 1; native-kernel-arithmetic-audit section 2.1)",
            "trace_ring": {
                "bytes": TRACE_RING_BYTES,
                "zeroed": true,
                "note": "a zeroed ring keeps the kernel's trace dereference legal; active_scope_mask = 0 disables every scope",
            },
        })
    }
}

/// The layer-0 query path: q_norm over the wq_a latent, wq_b, and the RoPE tail rotation.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct Layer0QueryPath {
    /// sha256 of the BF16 q_norm output (the wq_b activation).
    pub q_norm_sha256: String,
    /// sha256 of the raw wq_b output bytes, before rotation.
    pub q_pre_rope_sha256: String,
    /// sha256 of the rotated q rows (f32-le).
    pub q_post_rope_sha256: String,
    /// The rotated q rows' L2 norm.
    pub q_l2: f32,
    /// The per-rank head count (wq_b's shard columns over the q head width).
    pub heads_per_rank: usize,
    /// The q head width: the model's n_heads split the whole wq_b output, not config.head_dim
    /// (512), which is the MLA latent width the wkv and kv_norm rows use.
    pub q_head_width: usize,
    /// The rotated tail width (the last values of every head row).
    pub rope_head_dim: usize,
    /// The q_norm width (q_lora_rank), the wq_b reduction width.
    pub q_norm_width: usize,
    /// The position every head row rotated at.
    pub position: usize,
    /// The cubin digest the launch resolved.
    pub cubin_sha256: String,
    /// sha256 of the E4M3 activation codes the native wq_b GEMM consumed.
    pub activation_codes_sha256: String,
    /// sha256 of the ue8m0 activation scale codes, one per 32 values.
    pub activation_scales_sha256: String,
    /// The activation code byte count (one E4M3 byte per q_norm value).
    pub activation_code_bytes: usize,
    /// The activation scale byte count (one E8M0 byte per 32 values).
    pub activation_scale_bytes: usize,
    /// The rotated q rows as BF16 bytes (the CSA union body's q operand).
    pub q_post_rope_bf16: Vec<u8>,
    /// The q_norm output as the reference seam consumes it: the BF16 activation decoded to f32.
    /// The compressed-candidate path reads this exact row (the reference's q_latent argument).
    pub q_latent_bf16: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl Layer0QueryPath {
    /// The evidence block for the refused run.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stage": "layer0-query-path-q-norm-wq-b-rope",
            "kernel": sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
            "arithmetic_family": NATIVE_DENSE_KERNEL,
            "cubin_sha256": self.cubin_sha256,
            "launches": 1,
            "q_norm": {
                "weight": "layers.0.attn.q_norm.weight, BF16 resident, read from the arena",
                "cols": self.q_norm_width,
                "latent_bf16_sha256": self.q_norm_sha256,
                "bf16_round_trip": true,
            },
            "wq_b": {
                "cols_per_rank": self.heads_per_rank * self.q_head_width,
                "inner": 1280,
                "pre_rope_sha256": self.q_pre_rope_sha256,
                "activation_quant": {
                    "rule": "the BF16 q_norm output quantised to E4M3 codes with one ue8m0 scale per 32 contiguous values (the released W8A8 rule, D39 rule 1)",
                    "block": 32,
                    "codes_sha256": self.activation_codes_sha256,
                    "scales_sha256": self.activation_scales_sha256,
                    "code_bytes": self.activation_code_bytes,
                    "scale_bytes": self.activation_scale_bytes,
                },
            },
            "q_geometry": {
                "heads_per_rank": self.heads_per_rank,
                "q_head_width": self.q_head_width,
                "mla_latent_width": 512,
                "note": "config.head_dim is 512, the MLA latent width the wkv/kv_norm rows use; the q head width is the whole stored wq_b output over n_heads ((8192 per-rank Column rows x world 4)/64 = 512), not a small per-rank slice",
            },
            "rope": {
                "position": self.position,
                "rope_head_dim": self.rope_head_dim,
                "post_rope_sha256": self.q_post_rope_sha256,
                "post_rope_l2": self.q_l2,
            },
            "exact": "q_norm is the released RMSNorm (engine linalg::rms_norm) on the wq_a latent, narrowed to BF16; wq_b is the native dense GEMM (E4M3 activations, ue8m0 block-32 scales, E4M3 B12x weights, f32 accumulate, bf16 output); RoPE is engine linalg::rope_rotate with rope_tables_for_layer, the same host reference the twin uses",
        })
    }
}

/// The kv_norm row width of one MLA latent row, which is also the window quantiser's row width.
pub const MLA_LATENT: usize = 512;
/// The latent channels one window E8M0 scale covers (the released window quantiser's block).
pub const WINDOW_QUANT_BLOCK: usize = 32;
/// The E4M3FN code bytes of one engine-format window row.
pub const WINDOW_CODE_BYTES: usize = MLA_LATENT;
/// The E8M0 scale bytes of one engine-format window row.
pub const WINDOW_SCALE_BYTES: usize = MLA_LATENT / WINDOW_QUANT_BLOCK;

/// The engine-format window bytes of one rotated kv row: E4M3FN codes, one per latent channel, and
/// one E8M0 power-of-two scale per WINDOW_QUANT_BLOCK channels.
///
/// This is the window fp8 round-trip the attention body consumes: the deployed
/// sparkflash_kernels::cuda_harness::engine_window_row performs the engine's storage-dtype
/// (BF16) round, quantises with the released engine quantiser, and refuses unless that result is
/// byte-identical to m4_quant::act_quant_fp8_row on the same bf16 row. The kernel decodes the
/// codes back as bf16(code * scale) (WINDOW_QUANT_BLOCK, power-of-two scales), so the round-trip
/// is the stored byte stream, not a re-derivation.
///
/// # Errors
/// Refuses a row that is not one whole MLA latent row or that is not finite, and propagates the
/// helper's refusal when the engine quantiser and the format oracle disagree.
pub fn window_row_bytes(latent: &[f32]) -> Result<(Vec<u8>, Vec<u8>), String> {
    if latent.len() != MLA_LATENT {
        return Err(format!(
            "window row: the rotated kv row is {} values, one window row is {MLA_LATENT}",
            latent.len()
        ));
    }
    if !latent.iter().all(|value| value.is_finite()) {
        return Err("window row: the rotated kv row is not finite".to_string());
    }
    let row = sparkflash_kernels::cuda_harness::engine_window_row(latent)?;
    if row.codes.len() != WINDOW_CODE_BYTES || row.scales.len() != WINDOW_SCALE_BYTES {
        return Err(format!(
            "window row: the quantiser produced {} code bytes and {} scale bytes, one engine window row is {WINDOW_CODE_BYTES} and {WINDOW_SCALE_BYTES}",
            row.codes.len(),
            row.scales.len()
        ));
    }
    Ok((row.codes, row.scales))
}

/// The layer-0 kv path: kv_norm over the wkv latent and the RoPE tail rotation.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct Layer0KvPath {
    /// sha256 of the BF16 kv_norm output.
    pub kv_norm_sha256: String,
    /// sha256 of the rotated kv row (BF16 bytes, the window quantiser's input).
    pub kv_post_rope_sha256: String,
    /// The rotated row's L2 norm.
    pub kv_l2: f32,
    /// The MLA latent width (the row width kv_norm normalises).
    pub mla_width: usize,
    /// The rotated tail width.
    pub rope_head_dim: usize,
    /// The position the row rotated at.
    pub position: usize,
    /// sha256 of the engine-format E4M3FN window codes (one byte per latent channel).
    pub window_codes_sha256: String,
    /// sha256 of the engine-format E8M0 window scales (one byte per 32 channels).
    pub window_scales_sha256: String,
    /// The window code bytes (512).
    pub window_codes_len: usize,
    /// The window scale bytes (16).
    pub window_scales_len: usize,
    /// The engine-format E4M3FN window codes (the body's window_codes operand).
    pub window_codes: Vec<u8>,
    /// The engine-format E8M0 window scales (the body's window_scales operand).
    pub window_scales: Vec<u8>,
}

#[cfg(feature = "cuda")]
impl Layer0KvPath {
    /// The evidence block for the refused run.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stage": "layer0-kv-path-kv-norm-rope",
            "kv_norm": {
                "weight": "layers.0.attn.kv_norm.weight, BF16 resident, read from the arena",
                "cols": self.mla_width,
                "latent_bf16_sha256": self.kv_norm_sha256,
                "bf16_round_trip": true,
            },
            "rope": {
                "position": self.position,
                "width": self.mla_width,
                "rope_head_dim": self.rope_head_dim,
                "post_rope_sha256": self.kv_post_rope_sha256,
                "post_rope_l2": self.kv_l2,
            },
            "fp8_round_trip": {
                "codes_sha256": self.window_codes_sha256,
                "scales_sha256": self.window_scales_sha256,
                "codes_len": self.window_codes_len,
                "scales_len": self.window_scales_len,
                "block": WINDOW_QUANT_BLOCK,
                "scale_rule": "E8M0 power-of-two per 32-channel block",
                "exact": "sparkflash_kernels::cuda_harness::engine_window_row: the engine storage-dtype BF16 round, the released engine fp8 quantiser, and a byte-identity check against m4_quant::act_quant_fp8_row (the format oracle) on the same bf16 row; the kernel decodes bf16(code * scale)",
            },
            "exact": "kv_norm is the released RMSNorm (engine linalg::rms_norm) over the 512-wide MLA latent row, narrowed to BF16, then engine linalg::rope_rotate with rope_tables_for_layer on the last rope_head_dim values",
        })
    }
}

/// Runs the layer-0 kv path: kv_norm over the wkv latent from the arena weight and the RoPE tail
/// rotation. This is a host seam only - no kernel is launched - and it stops short of the window
/// fp8 round-trip, which belongs to the window/selected construction stage.
#[cfg(feature = "cuda")]
fn run_layer_kv_path(
    layer: usize,
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    request: &TokenStageRequest<'_>,
    arena: &sparkflash_device::arena::ResidentArena,
    projections: &Layer0Projections,
    position: usize,
) -> Result<Layer0KvPath, String> {
    use sparkflash_engine::linalg;

    let config = request.config.ok_or_else(|| {
        "layer-0 kv path: the loop was built without the model config".to_string()
    })?;
    let mla_width = config.head_dim;
    let rope_head_dim = config.rope_head_dim;
    let mut kv = bf16_from_le_bytes(&projections.wkv_latent_bytes)?;
    if kv.len() != mla_width {
        return Err(format!(
            "layer-0 kv path: the wkv latent is {} values, one MLA latent row is {mla_width}",
            kv.len()
        ));
    }

    let norm_name = format!("layers.{layer}.attn.kv_norm.weight");
    let norm_bytes = read_arena_bytes(cuda, arena, &norm_name, mla_width * 2)?;
    let norm_weight = bf16_from_le_bytes(&norm_bytes)?;
    linalg::rms_norm(&mut kv, &norm_weight, mla_width, config.norm_eps as f32)
        .map_err(|error| format!("layer-0 kv path: kv_norm: {error}"))?;
    let normed_bytes = bf16_le_bytes(&kv);

    if rope_head_dim == 0 || rope_head_dim > mla_width || !rope_head_dim.is_multiple_of(2) {
        return Err(format!(
            "layer-0 kv path: LATENT width {mla_width}, rope_head_dim {rope_head_dim}: the rotated tail does not fit the row"
        ));
    }
    let tables =
        sparkflash_engine::forward::attention::rope_tables_for_layer(config, layer, &[position]);
    let start = mla_width - rope_head_dim;
    linalg::rope_rotate(&mut kv[start..], &tables.cos, &tables.sin)
        .map_err(|error| format!("layer-0 kv path: rope: {error}"))?;
    let rotated_bytes = bf16_le_bytes(&kv);
    let l2 = if layer_evidence_readbacks() {
        kv.iter().map(|value| value * value).sum::<f32>().sqrt()
    } else {
        0.0
    };
    let (window_codes, window_scales) = window_row_bytes(&kv)?;
    Ok(Layer0KvPath {
        kv_norm_sha256: evidence_sha256(&normed_bytes),
        kv_post_rope_sha256: evidence_sha256(&rotated_bytes),
        kv_l2: l2,
        mla_width,
        rope_head_dim,
        position,
        window_codes_sha256: evidence_sha256(&window_codes),
        window_scales_sha256: evidence_sha256(&window_scales),
        window_codes_len: window_codes.len(),
        window_scales_len: window_scales.len(),
        window_codes,
        window_scales,
    })
}

/// The CSA-union attention body's fixed device buffers and the per-layer sink shards.
///
/// The body is the flash_mla_csa2_seq_sm121 launch on the per-rank heads. Its buffers are allocated
/// once per run (the largest shape each can need) and every per-position launch restages only the
/// bytes that change: the query, the window row ring, the expanded compressed store, the selected
/// list and the layer's sink shard. The index-key pool, the index query, the block score scratch and
/// the trace ring are the zeroed wiring operands: the kernel's selected list comes from the
/// reference candidate select, so no consumer reads them, but the ABI requires them addressable.
#[cfg(feature = "cuda")]
pub struct AttentionBody {
    /// The loaded module the kernel belongs to; held so the code stays loaded for the whole run.
    #[allow(dead_code)]
    module: sparkflash_kernels::cuda_driver::Module,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    /// The per-rank query heads.
    heads: usize,
    /// The q head width (the stored wq_b output over the model heads).
    head_width: usize,
    q: Deviceptr,
    window_codes: Deviceptr,
    window_scales: Deviceptr,
    kv_codes: Deviceptr,
    kv_scales: Deviceptr,
    selected: Deviceptr,
    sink: Deviceptr,
    index_key: Deviceptr,
    index_query: Deviceptr,
    block_scores: Deviceptr,
    block_best: Deviceptr,
    out: Deviceptr,
    trace_ring: Deviceptr,
    /// Every layer's replicated sink shard bytes, read from the arena once.
    sinks: Vec<Vec<u8>>,
    /// The cubin digest the launch resolves.
    cubin_sha256: String,
    /// The HSI blocks the zeroed index key pool covers.
    index_blocks_staged: usize,
    /// M2/C5: whether the expanded compressed store currently on the device is all zero.
    ///
    /// The store is a kernel input that no launch writes, so a layer whose staged store is all
    /// zero (store_rows 0) can skip its 147,456-byte upload while this flag says the device bytes
    /// are already zero. `open` zero-fills the store once, which makes the first skip provable.
    store_zero: std::cell::Cell<bool>,
}

/// One executed attention body launch: the staged operands, the kernel's out, the host twin's
/// comparison and the inverse-RoPE activation the wo_a projection consumes.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct AttentionStep {
    /// The per-rank query heads the launch covered.
    pub heads_per_rank: usize,
    /// The q head width (the whole stored wq_b output over the model heads).
    pub head_width: usize,
    /// The position the body ran at.
    pub position: usize,
    /// The window rows supplied (the attended window width).
    pub window_rows: usize,
    /// The selected candidate count staged.
    pub selected_count: usize,
    /// The compressed rows the selected list wrote into the expanded store.
    pub store_rows: usize,
    /// The layer's compression ratio the store rows were expanded with.
    pub ratio: usize,
    /// The launch grid.
    pub grid: (u32, u32, u32),
    /// The cubin digest the launch resolved.
    pub cubin_sha256: String,
    /// sha256 of the staged BF16 q bytes (the query path's rotated rows).
    pub q_sha256: String,
    /// sha256 of the layer's sink shard bytes.
    pub sink_sha256: String,
    /// sha256 of the staged window codes.
    pub window_codes_sha256: String,
    /// sha256 of the staged window scales.
    pub window_scales_sha256: String,
    /// sha256 of the expanded compressed store codes.
    pub kv_codes_sha256: String,
    /// sha256 of the expanded compressed store scales.
    pub kv_scales_sha256: String,
    /// sha256 of the staged selected list (u32 le).
    pub selected_sha256: String,
    /// sha256 of the BF16 out buffer the body wrote (before the inverse RoPE).
    pub out_sha256: String,
    /// The out buffer's L2 norm.
    pub out_l2: f32,
    /// sha256 of the activation after the inverse RoPE (the wo_a BF16 activation).
    pub inverse_rope_sha256: String,
    /// The activation's L2 norm.
    pub inverse_rope_l2: f32,
    /// The BF16 activation bytes (post inverse RoPE, head-major).
    pub activation_bf16: Vec<u8>,
    /// The host twin over the exact staged bytes against the kernel out; absent when the run
    /// sampled this position out (a full prefill twins the head positions and every eighth
    /// position, the smoke windows twin every position).
    pub twin: Option<TwinStats>,
    /// The HSI blocks the zeroed index key pool covers (the indexer is not a consumer here).
    pub index_blocks_staged: usize,
}

#[cfg(feature = "cuda")]
impl AttentionStep {
    /// The evidence block for the forward record.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stage": "csa-union-attention-body",
            "kernel": "flash_mla_csa2_seq_sm121",
            "cubin_sha256": self.cubin_sha256,
            "position": self.position,
            "launches": 1,
            "grid": [self.grid.0, self.grid.1, self.grid.2],
            "block_threads": 256,
            "window_rows": self.window_rows,
            "selected_count": self.selected_count,
            "store_rows": self.store_rows,
            "ratio": self.ratio,
            "q": {"sha256": self.q_sha256, "bytes": self.heads_per_rank * self.head_width * 2},
            "sink": {"sha256": self.sink_sha256, "values": self.heads_per_rank},
            "window": {"codes_sha256": self.window_codes_sha256, "scales_sha256": self.window_scales_sha256, "rows": self.window_rows},
            "compressed_store": {"codes_sha256": self.kv_codes_sha256, "scales_sha256": self.kv_scales_sha256, "expanded_rows": crate::tp4_prefill::EXPANDED_ROWS},
            "selected": {"sha256": self.selected_sha256, "count": self.selected_count},
            "indexer": {"blocks_staged": self.index_blocks_staged, "query_key": "zeroed: the selected list is the reference candidate select's ids, so no consumer reads the block scores"},
            "out": {"sha256": self.out_sha256, "l2": self.out_l2},
            "inverse_rope": {"sha256": self.inverse_rope_sha256, "l2": self.inverse_rope_l2, "note": "the reference applies the inverse RoPE to the sparse-attention out before wo_a (model.py:772); the kernel writes the pre-inverse-RoPE bf16 row, so the loop rotates each head's tail with the layer's own tables and narrows the result to the wo_a activation"},
            "twin": match &self.twin {
                Some(twin) => serde_json::json!({
                    "reference": "mla_ref::attention_with_sink_union over the exact staged bytes",
                    "heads": twin.heads,
                    "window_rows": twin.window_rows,
                    "selected": twin.selected,
                    "bf16_mismatches": twin.mismatches,
                    "slots": twin.slots,
                    "exact_fraction": twin.exact_fraction,
                    "max_abs_diff": twin.max_abs,
                }),
                None => serde_json::json!({"skipped": "sampled out: a full prefill twins the head positions and every eighth position"}),
            },
            "exact": "format-2 CSA union attention on the per-rank heads: window ids are the visited positions oldest first in the staging order, the selected list is the reference candidate select's ascending shared ids expanded to j*ratio store rows",
        })
    }
}

#[cfg(feature = "cuda")]
impl AttentionBody {
    /// Allocates the body's buffers, loads the pinned cubin and reads every layer's sink shard.
    ///
    /// # Errors
    /// A cubin read/load failure, an allocation failure, or a sink shard read failure.
    pub fn open(
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        request: &TokenStageRequest<'_>,
        arena: &sparkflash_device::arena::ResidentArena,
        heads: usize,
        head_width: usize,
    ) -> Result<Self, String> {
        use sparkflash_kernels::flash_mla_csa2_seq_launch as seq;
        use sparkflash_kernels::mla_ref;

        let config = request.config.ok_or_else(|| {
            "attention body: the loop was built without the model config".to_string()
        })?;
        if head_width != mla_ref::MLA_LATENT {
            return Err(format!(
                "attention body: the q head width is {head_width}, the body's latent is {}",
                mla_ref::MLA_LATENT
            ));
        }
        let kernel_dir = request
            .kernel_cubin_dir
            .ok_or_else(|| "attention body: the loop has no kernel cubin directory".to_string())?;
        let cubin = std::fs::read(kernel_dir.join("flash_mla_csa2_seq_sm121.cubin"))
            .map_err(|error| format!("attention body: read cubin: {error}"))?;
        let cubin_sha256 = sha256_hex(&cubin);
        let (module, kernel) = seq::load(cuda, &cubin)?;
        let mut allocated = 0usize;
        let mut alloc = |bytes: usize, what: &str| -> Result<Deviceptr, String> {
            cuda.alloc(bytes, &mut allocated)
                .map_err(|error| format!("attention body: {what} buffer: {error}"))
        };
        let q_width = heads * mla_ref::MLA_LATENT * 2;
        let index_blocks_staged = mla_ref::HSI_BLOCKS;
        let mut body = Self {
            module,
            kernel,
            heads,
            head_width,
            q: alloc(q_width, "q")?,
            window_codes: alloc(config.window_size * mla_ref::MLA_LATENT, "window codes")?,
            window_scales: alloc(
                config.window_size * mla_ref::MLA_LATENT / 32,
                "window scales",
            )?,
            kv_codes: alloc(
                crate::tp4_prefill::EXPANDED_ROWS * crate::tp4_prefill::STORE_CODE_BYTES,
                "expanded store codes",
            )?,
            kv_scales: alloc(
                crate::tp4_prefill::EXPANDED_ROWS * crate::tp4_prefill::STORE_SCALE_BYTES,
                "expanded store scales",
            )?,
            selected: alloc(crate::tp4_prefill::MAX_SELECTED * 4, "selected")?,
            sink: alloc(heads * 4, "sink")?,
            index_key: alloc(
                mla_ref::HSI_BLOCKS * mla_ref::HSI_BLOCK_TOKENS * mla_ref::INDEX_HEAD_DIM * 2,
                "index key",
            )?,
            index_query: alloc(
                mla_ref::INDEX_HEADS * mla_ref::INDEX_HEAD_DIM * 2,
                "index query",
            )?,
            block_scores: alloc(mla_ref::HSI_BLOCKS * 4, "block scores")?,
            block_best: alloc(mla_ref::HSI_BLOCKS * 4, "block best")?,
            out: alloc(q_width, "out")?,
            trace_ring: alloc(TRACE_RING_BYTES, "trace ring")?,
            sinks: Vec::with_capacity(config.n_layers),
            cubin_sha256,
            index_blocks_staged,
            store_zero: std::cell::Cell::new(true),
        };
        for layer in 0..config.n_layers {
            let sink = read_arena_bytes(
                cuda,
                arena,
                &format!("layers.{layer}.attn.attn_sink"),
                heads * 4,
            )?;
            body.sinks.push(sink);
        }
        // The wiring operands the kernel dereferences but no consumer reads with a caller-selected
        // candidate list.
        cuda.copy_htod(
            &body.index_key,
            &vec![
                0u8;
                mla_ref::HSI_BLOCKS * mla_ref::HSI_BLOCK_TOKENS * mla_ref::INDEX_HEAD_DIM * 2
            ],
        )
        .map_err(|error| format!("attention body: index key zero fill: {error}"))?;
        cuda.copy_htod(
            &body.index_query,
            &vec![0u8; mla_ref::INDEX_HEADS * mla_ref::INDEX_HEAD_DIM * 2],
        )
        .map_err(|error| format!("attention body: index query zero fill: {error}"))?;
        cuda.copy_htod(&body.block_scores, &vec![0u8; mla_ref::HSI_BLOCKS * 4])
            .map_err(|error| format!("attention body: block scores zero fill: {error}"))?;
        cuda.copy_htod(&body.block_best, &vec![0u8; mla_ref::HSI_BLOCKS * 4])
            .map_err(|error| format!("attention body: block best zero fill: {error}"))?;
        cuda.copy_htod(&body.trace_ring, &[0u8; TRACE_RING_BYTES])
            .map_err(|error| format!("attention body: trace ring zero fill: {error}"))?;
        // M2/C5: the expanded store is a kernel input, so after this one zero fill every layer
        // whose staged store is all zero can skip its upload and still leave the same bytes.
        cuda.copy_htod(
            &body.kv_codes,
            &vec![0u8; crate::tp4_prefill::EXPANDED_ROWS * crate::tp4_prefill::STORE_CODE_BYTES],
        )
        .map_err(|error| format!("attention body: expanded store codes zero fill: {error}"))?;
        cuda.copy_htod(
            &body.kv_scales,
            &vec![0u8; crate::tp4_prefill::EXPANDED_ROWS * crate::tp4_prefill::STORE_SCALE_BYTES],
        )
        .map_err(|error| format!("attention body: expanded store scales zero fill: {error}"))?;
        Ok(body)
    }

    /// Runs one position's body launch and returns its out, twin and inverse-RoPE activation.
    ///
    /// # Errors
    /// A staging/geometry disagreement, an upload/launch/readback failure, a non-finite out, or a
    /// twin that refuses its inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        request: &TokenStageRequest<'_>,
        layer: usize,
        query: &Layer0QueryPath,
        staged: &StagedAttention,
        position: usize,
        twin_requested: bool,
    ) -> Result<AttentionStep, String> {
        use sparkflash_engine::linalg;
        use sparkflash_kernels::flash_mla_csa2_seq_launch as seq;
        use sparkflash_kernels::mla_ref;

        let config = request.config.ok_or_else(|| {
            "attention body: the loop was built without the model config".to_string()
        })?;
        if query.q_post_rope_bf16.len() != self.heads * self.head_width * 2 {
            return Err(format!(
                "attention body: q is {} bytes, {} heads of {} need {}",
                query.q_post_rope_bf16.len(),
                self.heads,
                self.head_width,
                self.heads * self.head_width * 2
            ));
        }
        if staged.window_rows > config.window_size {
            return Err(format!(
                "attention body: {} window rows, the ring is {}",
                staged.window_rows, config.window_size
            ));
        }
        let sink = self
            .sinks
            .get(layer)
            .ok_or_else(|| format!("attention body: no sink shard was read for layer {layer}"))?;
        let sink_sha256 = evidence_sha256(sink);
        let selected_bytes: Vec<u8> = staged
            .selected
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        if staged.window_codes.len() != staged.window_rows * mla_ref::MLA_LATENT
            || staged.window_scales.len() != staged.window_rows * mla_ref::MLA_LATENT / 32
        {
            return Err(format!(
                "attention body: {} window rows staged as {} code and {} scale bytes",
                staged.window_rows,
                staged.window_codes.len(),
                staged.window_scales.len()
            ));
        }
        if staged.kv_codes.len()
            != crate::tp4_prefill::EXPANDED_ROWS * crate::tp4_prefill::STORE_CODE_BYTES
            || staged.kv_scales.len()
                != crate::tp4_prefill::EXPANDED_ROWS * crate::tp4_prefill::STORE_SCALE_BYTES
        {
            return Err(format!(
                "attention body: the expanded store is {} code and {} scale bytes",
                staged.kv_codes.len(),
                staged.kv_scales.len()
            ));
        }
        (|| -> Result<AttentionStep, String> {
            cuda.copy_htod(&self.q, &query.q_post_rope_bf16)
                .map_err(|error| format!("attention body: q htod: {error}"))?;
            cuda.copy_htod(&self.window_codes, &staged.window_codes)
                .map_err(|error| format!("attention body: window codes htod: {error}"))?;
            cuda.copy_htod(&self.window_scales, &staged.window_scales)
                .map_err(|error| format!("attention body: window scales htod: {error}"))?;
            // M2/C5: the staged store is all zero exactly when store_rows is 0 (stage_attention
            // initialises the view to zeros and only writes the selected rows). When the device
            // store is already zero from open() or a previous zero upload, the 147,456-byte upload
            // is a provable no-op and is skipped.
            let store_is_zero = staged.store_rows == 0;
            if !(store_is_zero && self.store_zero.get()) {
                cuda.copy_htod(&self.kv_codes, &staged.kv_codes)
                    .map_err(|error| {
                        format!("attention body: expanded store codes htod: {error}")
                    })?;
                cuda.copy_htod(&self.kv_scales, &staged.kv_scales)
                    .map_err(|error| {
                        format!("attention body: expanded store scales htod: {error}")
                    })?;
                self.store_zero.set(store_is_zero);
            }
            if !selected_bytes.is_empty() {
                cuda.copy_htod(&self.selected, &selected_bytes)
                    .map_err(|error| format!("attention body: selected htod: {error}"))?;
            }
            cuda.copy_htod(&self.sink, sink)
                .map_err(|error| format!("attention body: sink htod: {error}"))?;
            let selected_count = u32::try_from(staged.selected.len()).map_err(|_| {
                format!(
                    "attention body: {} selected ids do not fit a u32",
                    staged.selected.len()
                )
            })?;
            let layer_id = u32::try_from(layer)
                .map_err(|_| format!("attention body: layer {layer} does not fit a u32"))?;
            let params = seq::FlashMlaSeqParams::new(
                self.q,
                self.kv_codes,
                self.kv_scales,
                self.window_codes,
                self.window_scales,
                seq::FORMAT_CSA_UNION,
                u64::try_from(staged.window_rows).map_err(|_| {
                    format!(
                        "attention body: {} window rows do not fit a u64",
                        staged.window_rows
                    )
                })?,
                self.selected,
                self.index_key,
                self.index_query,
                self.sink,
                self.block_scores,
                self.block_best,
                self.out,
                u64::try_from(position).map_err(|_| {
                    format!("attention body: position {position} does not fit a u64")
                })?,
                selected_count,
                self.trace_ring.0,
                layer_id,
                0,
            );
            cuda.launch(
                self.kernel,
                (self.heads as u32, 1, 1),
                (seq::THREADS, 1, 1),
                params.as_slice(),
            )
            .map_err(|error| format!("attention body: launch: {error}"))?;
            let mut out_bf16 = vec![0u8; self.heads * self.head_width * 2];
            cuda.copy_dtoh(&mut out_bf16, &self.out)
                .map_err(|error| format!("attention body: out dtoh: {error}"))?;
            let mut values = bf16_from_le_bytes(&out_bf16)?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(
                    "attention body: the body wrote a non-finite value into out; refusing to present it"
                        .to_string(),
                );
            }
            let out_sha256 = evidence_sha256(&out_bf16);
            // C10: the two L2 norms below are read only by the layer-chain record and the run
            // evidence document, so a default or --forward-lean run (evidence off) skips them.
            let out_l2 = if layer_evidence_readbacks() {
                values.iter().map(|value| value * value).sum::<f32>().sqrt()
            } else {
                0.0
            };
            // The twin sees the bytes the kernel saw: the BF16 q as read, the layer's sink, the
            // staged window/store bytes and the ascending selected list.
            let twin_timer = BlockTimer::start(twin_requested && request.forward_perf.probe);
            let q_read = bf16_from_le_bytes(&query.q_post_rope_bf16)?;
            let sink_values = f32_from_le_bytes(sink)?;
            let twin = if twin_requested {
                Some(crate::tp4_prefill::twin_out(
                    &q_read,
                    &sink_values,
                    staged,
                    position,
                    &values,
                )?)
            } else {
                None
            };
            if twin_requested {
                probe::layer(cuda, layer, "attention_twin", twin_timer);
            }
            // The reference rotates the sparse out's rope tail back before wo_a (model.py:772).
            let tables = sparkflash_engine::forward::attention::rope_tables_for_layer(
                config,
                layer,
                &[position],
            );
            if config.rope_head_dim > self.head_width {
                return Err(format!(
                    "attention body: rope_head_dim {} does not fit a {} head row",
                    config.rope_head_dim, self.head_width
                ));
            }
            for head in 0..self.heads {
                let start = head * self.head_width + self.head_width - config.rope_head_dim;
                let tail = &mut values[start..start + config.rope_head_dim];
                linalg::rope_rotate_inverse(tail, &tables.cos, &tables.sin)
                    .map_err(|error| format!("attention body: inverse rope: {error}"))?;
            }
            let inverse_rope_l2 = if layer_evidence_readbacks() {
                values.iter().map(|value| value * value).sum::<f32>().sqrt()
            } else {
                0.0
            };
            let activation_bf16 = bf16_le_bytes(&values);
            Ok(AttentionStep {
                heads_per_rank: self.heads,
                head_width: self.head_width,
                position,
                window_rows: staged.window_rows,
                selected_count: staged.selected.len(),
                store_rows: staged.store_rows,
                ratio: staged.ratio,
                grid: (self.heads as u32, 1, 1),
                cubin_sha256: self.cubin_sha256.clone(),
                q_sha256: evidence_sha256(&query.q_post_rope_bf16),
                sink_sha256,
                window_codes_sha256: evidence_sha256(&staged.window_codes),
                window_scales_sha256: evidence_sha256(&staged.window_scales),
                kv_codes_sha256: evidence_sha256(&staged.kv_codes),
                kv_scales_sha256: evidence_sha256(&staged.kv_scales),
                selected_sha256: evidence_sha256(&selected_bytes),
                out_sha256,
                out_l2,
                inverse_rope_sha256: evidence_sha256(&activation_bf16),
                inverse_rope_l2,
                activation_bf16,
                twin,
                index_blocks_staged: self.index_blocks_staged,
            })
        })()
    }
}
/// Runs the layer-0 query path: q_norm over the wq_a latent from the arena weight, the wq_b dense
/// B12x launch on the per-rank head rows, and the RoPE tail rotation at the first-step position.
///
/// This is the released reference's `wq_b(q_norm(wq_a(x)))` with its RoPE tail rotated: q_norm is
/// the engine's RMSNorm over the whole q_lora_rank-wide latent, narrowed to BF16 exactly as the
/// reference's bf16 tensor does (the F5 seam), then wq_b's per-rank shard columns are launched
/// through the native dense family (E4M3 activation codes + ue8m0 scales, f32 accumulate), and
/// every head row's last rope_head_dim values are rotated with the engine's table builder for
/// layer 0 at the token's position.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_layer_query_path(
    layer: usize,
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    kernels: &mut KernelCache,
    request: &TokenStageRequest<'_>,
    bind: &crate::tp4_resident::DeviceBind,
    arena: &sparkflash_device::arena::ResidentArena,
    projections: &Layer0Projections,
    position: usize,
    parity: &mut NativeDenseParity,
) -> Result<Layer0QueryPath, String> {
    use sparkflash_engine::linalg;
    use sparkflash_kernels::mxfp8_dense_launch;
    use sparkflash_kernels::mxfp8_gemv_launch;

    let config = request.config.ok_or_else(|| {
        "layer-0 query path: the loop was built without the model config".to_string()
    })?;
    let rank_width = config.q_lora_rank;
    let n_heads = config.n_heads;
    let world = request.world;
    let rope_head_dim = config.rope_head_dim;
    let latent = bf16_from_le_bytes(&projections.wq_a_latent_bytes)?;
    if latent.len() != rank_width {
        return Err(format!(
            "layer-0 query path: the wq_a latent is {} values, q_lora_rank is {rank_width}",
            latent.len()
        ));
    }

    // q_norm: [q_lora_rank] BF16 resident, read from the live arena.
    let norm_name = format!("layers.{layer}.attn.q_norm.weight");
    let norm_bytes = read_arena_bytes(cuda, arena, &norm_name, rank_width * 2)?;
    let norm_weight = bf16_from_le_bytes(&norm_bytes)?;
    let mut normed = latent.clone();
    linalg::rms_norm(
        &mut normed,
        &norm_weight,
        rank_width,
        config.norm_eps as f32,
    )
    .map_err(|error| format!("layer-0 query path: q_norm: {error}"))?;
    // The reference's q_norm output is a bf16 tensor: the F5 seam narrows it before wq_b.
    let normed_bytes = bf16_le_bytes(&normed);
    // D39 rule 1: the W8A8 activation is that BF16 q_norm output, quantised to E4M3 codes with one
    // ue8m0 scale per 32 values before the native dense GEMM. The values are decoded from the same
    // BF16 bytes the buffered launch consumes, so the codes match the F5 seam exactly.
    let activation_values = bf16_from_le_bytes(&normed_bytes)?;
    let (activation_codes, activation_scales) =
        quantise_native_activation(&activation_values, "layer-0 query path")?;

    let (wq_b_cols, wq_b_inner) = bind
        .plan_index
        .b12x_geometry(&format!("layers.{layer}.attn.wq_b.weight"))?;
    let (wq_a_cols, wq_a_inner) = bind
        .plan_index
        .b12x_geometry(&format!("layers.{layer}.attn.wq_a.weight"))?;
    if (wq_a_cols, wq_a_inner) != (rank_width, HIDDEN) {
        return Err(format!(
            "layer-0 query path: the plan's wq_a geometry is {wq_a_cols}x{wq_a_inner}, the model config says {rank_width}x{HIDDEN}"
        ));
    }
    if wq_b_inner != rank_width {
        return Err(format!(
            "layer-0 query path: wq_b reduces over {wq_b_inner}, q_norm is {rank_width} wide"
        ));
    }
    if n_heads == 0 || world == 0 || !n_heads.is_multiple_of(world) {
        return Err(format!(
            "layer-0 query path: {n_heads} heads do not split over a TP world of {world}"
        ));
    }
    let global_wq_b_cols = wq_b_cols
        .checked_mul(world)
        .ok_or_else(|| "layer-0 query path: the wq_b column count overflows".to_string())?;
    if !global_wq_b_cols.is_multiple_of(n_heads) {
        return Err(format!(
            "layer-0 query path: wq_b's {global_wq_b_cols} global columns are not whole {n_heads}-head rows"
        ));
    }
    let q_head_width = global_wq_b_cols / n_heads;
    if q_head_width == 0 || !wq_b_cols.is_multiple_of(q_head_width) {
        return Err(format!(
            "layer-0 query path: wq_b's per-rank {wq_b_cols} columns are not whole {q_head_width}-wide head rows"
        ));
    }
    let heads = wq_b_cols / q_head_width;
    if rope_head_dim == 0 || rope_head_dim > q_head_width || !rope_head_dim.is_multiple_of(2) {
        return Err(format!(
            "layer-0 query path: q head width {q_head_width}, rope_head_dim {rope_head_dim}: the rotated tail does not fit a head row"
        ));
    }
    let (wq_b_weight, wq_b_scales) = bind
        .attention
        .pointers
        .get(&format!("layers.{layer}.attn.wq_b.weight"))
        .copied()
        .ok_or_else(|| "layer-0 query path: the bind has no wq_b pair".to_string())?;
    if wq_b_weight.0 == 0 || wq_b_scales.0 == 0 {
        return Err(
            "layer-0 query path: wq_b bound a null B12x pointer; refusing to launch".to_string(),
        );
    }

    let kernel_dir = request
        .kernel_cubin_dir
        .ok_or_else(|| "layer-0 query path: the loop has no kernel cubin directory".to_string())?;
    // WS-D: the decode step runs wq_b through the dedicated decode GEMV - the same per-output
    // arithmetic as the pinned dense family, tiled one warp per 32-output B12x micro-tile with one
    // output column per lane. It is its own symbol and its own cubin, loaded and verified by name.
    // M3/C1, as in the projections block: the loop's kernel cache loads the decode GEMV once per
    // run and verifies the pin on that load, so this block only looks the digest up. Loading per
    // call would re-read the 110 KB cubin and reload the module on every layer's every position,
    // which is host work the query path does not owe the model.
    let (kernel, cubin_sha256) = kernels.gemv(cuda, kernel_dir, "layer-0 query path")?;
    let mut allocated = 0usize;
    let a_codes = cuda
        .alloc_pooled(activation_codes.len(), &mut allocated)
        .map_err(|error| format!("layer-0 query path: activation code buffer: {error}"))?;
    let a_scales = cuda
        .alloc_pooled(activation_scales.len(), &mut allocated)
        .map_err(|error| format!("layer-0 query path: activation scale buffer: {error}"))?;
    let out = cuda
        .alloc_pooled(wq_b_cols * 2, &mut allocated)
        .map_err(|error| format!("layer-0 query path: wq_b output buffer: {error}"))?;
    let trace_ring = cuda
        .alloc_pooled(TRACE_RING_BYTES, &mut allocated)
        .map_err(|error| format!("layer-0 query path: trace ring buffer: {error}"))?;
    cuda.copy_htod(&trace_ring, &[0u8; TRACE_RING_BYTES])
        .map_err(|error| format!("layer-0 query path: trace ring zero fill: {error}"))?;

    let staged = (|| -> Result<Layer0QueryPath, String> {
        cuda.copy_htod(&a_codes, &activation_codes)
            .map_err(|error| format!("layer-0 query path: activation code htod: {error}"))?;
        cuda.copy_htod(&a_scales, &activation_scales)
            .map_err(|error| format!("layer-0 query path: activation scale htod: {error}"))?;
        let params = mxfp8_dense_launch::Mxfp8DenseParams::new(
            a_codes,
            a_scales,
            wq_b_weight,
            wq_b_scales,
            out,
            1,
            wq_b_cols as u32,
            wq_b_inner as u32,
            trace_ring.0,
            layer as u32,
            0,
        );
        mxfp8_gemv_launch::launch(cuda, kernel, &params)?;
        let mut raw = vec![0u8; wq_b_cols * 2];
        cuda.copy_dtoh(&mut raw, &out)
            .map_err(|error| format!("layer-0 query path: wq_b dtoh: {error}"))?;
        // wq_b is a row shard at TP4: the parity re-launch uses the rank's own shard bytes and the
        // plan's per-rank geometry, so this checks the launch construction, not the sharding maths.
        parity.capture(
            cuda,
            request,
            layer,
            position,
            "layer0-query-path",
            "wq_b",
            &format!("layers.{layer}.attn.wq_b.weight"),
            (wq_b_cols, wq_b_inner),
            &activation_values,
            &activation_codes,
            &activation_scales,
            wq_b_weight,
            wq_b_scales,
            &raw,
        );
        let mut q = bf16_from_le_bytes(&raw)?;
        // The layer's own tables: above ratio 0 the reference rotates with the compressor rope
        // base, so a hardcoded layer 0 would feed every compressed layer the window layer's
        // frequencies (the kv path already used the layer index; this is the same rule).
        let tables = sparkflash_engine::forward::attention::rope_tables_for_layer(
            config,
            layer,
            &[position],
        );
        for head in 0..heads {
            let start = head * q_head_width + q_head_width - rope_head_dim;
            let tail = &mut q[start..start + rope_head_dim];
            linalg::rope_rotate(tail, &tables.cos, &tables.sin)
                .map_err(|error| format!("layer-0 query path: rope: {error}"))?;
        }
        let l2 = if layer_evidence_readbacks() {
            q.iter().map(|value| value * value).sum::<f32>().sqrt()
        } else {
            0.0
        };
        let q_latent_bf16 = bf16_from_le_bytes(&normed_bytes)?;
        Ok(Layer0QueryPath {
            activation_codes_sha256: evidence_sha256(&activation_codes),
            activation_scales_sha256: evidence_sha256(&activation_scales),
            activation_code_bytes: activation_codes.len(),
            activation_scale_bytes: activation_scales.len(),
            q_norm_sha256: evidence_sha256(&normed_bytes),
            q_pre_rope_sha256: evidence_sha256(&raw),
            q_post_rope_sha256: evidence_f32_sha256(&q),
            q_l2: l2,
            heads_per_rank: heads,
            q_head_width,
            rope_head_dim,
            q_norm_width: rank_width,
            position,
            cubin_sha256,
            q_post_rope_bf16: bf16_le_bytes(&q),
            q_latent_bf16,
        })
    })();
    cuda.release_pooled(&[&a_codes, &a_scales, &out, &trace_ring]);
    // The module belongs to the run's kernel cache now (M3/C1), so this block frees only its pooled
    // buffers; the cache unloads the module when the run scope drops it.
    staged
}

/// The layer-0 attention output projection pair and its all-reduce: the grouped wo_a over the
/// per-rank output groups, the row-sharded wo_b, and the canonical bf16 all-reduce that sums the
/// wo_b partials into the attention block's delta.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct Layer0AttentionOutput {
    /// The group count launched (the per-rank output groups).
    pub groups: usize,
    /// The group reduction width (heads per group x head width).
    pub group_inner: usize,
    /// The world size the per-rank geometry was validated against.
    pub world: usize,
    /// The config's global output-group count (the plan's per-rank count is groups).
    pub o_groups_global: usize,
    /// This rank's attention heads.
    pub heads_per_rank: usize,
    /// The per-head MLA latent width.
    pub head_width: usize,
    /// The output LoRA rank.
    pub o_lora_rank: usize,
    /// The wo_a cubin digest.
    pub wo_a_cubin_sha256: String,
    /// sha256 of the BF16 wo_a output (the wo_b activation).
    pub wo_a_out_sha256: String,
    /// The wo_a output's L2 norm.
    pub wo_a_out_l2: f32,
    /// The wo_b cubin digest.
    pub wo_b_cubin_sha256: String,
    /// sha256 of the E4M3 activation codes the native wo_b GEMM consumed.
    pub wo_b_activation_codes_sha256: String,
    /// sha256 of the ue8m0 activation scale codes, one per 32 values.
    pub wo_b_activation_scales_sha256: String,
    /// The wo_b activation code byte count (one E4M3 byte per inner value).
    pub wo_b_activation_code_bytes: usize,
    /// The wo_b activation scale byte count (one E8M0 byte per 32 values).
    pub wo_b_activation_scale_bytes: usize,
    /// sha256 of the per-rank BF16 wo_b partial (before the attention all-reduce).
    pub wo_b_partial_sha256: String,
    /// The partial's L2 norm.
    pub wo_b_partial_l2: f32,
    /// sha256 of the all-reduced BF16 attention delta.
    pub reduced_sha256: String,
    /// The reduced delta's L2 norm.
    pub reduced_l2: f32,
    /// The all-reduce calls this stage issued (one per position per layer).
    pub reduce_calls: usize,
    /// The exchange sequence the reduce used.
    pub reduce_sequence: u64,
    /// The all-reduced BF16 delta bytes (the hc_post seam's attention output).
    pub reduced_bf16: Vec<u8>,
}

#[cfg(feature = "cuda")]
impl Layer0AttentionOutput {
    /// The evidence block for the refused run.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stage": "layer0-attention-output-wo-a-wo-b-reduce",
            "wo_a": {
                "kernel": NATIVE_WO_A_KERNEL,
                "cubin_sha256": self.wo_a_cubin_sha256,
                "launches": 1,
                "grid": [self.groups, 4, 1],
                "block_threads": 256,
                "world": self.world,
                "groups": self.groups,
                "o_groups_global": self.o_groups_global,
                "geometries": {
                    "group_inner": self.group_inner,
                    "heads_per_rank": self.heads_per_rank,
                    "head_width": self.head_width,
                    "o_lora_rank": self.o_lora_rank,
                    "rule": "per-rank plan: groups = o_groups/world, group_inner = (heads_per_rank/groups)*head_width, wo_a = (groups*o_lora_rank, group_inner), wo_b = (HIDDEN, groups*o_lora_rank)",
                },
                "out_sha256": self.wo_a_out_sha256,
                "out_l2": self.wo_a_out_l2,
                "weights": "the arena's per-rank B12x wo_a weight and scale grid (two group buffers), no repack",
            },
            "wo_b": {
                "kernel": sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
                "arithmetic_family": NATIVE_DENSE_KERNEL,
                "cubin_sha256": self.wo_b_cubin_sha256,
                "launches": 1,
                "rows": 1,
                "cols": HIDDEN,
                "inner": self.groups * self.o_lora_rank,
                "activation_quant": {
                    "rule": "the BF16 wo_a output quantised to E4M3 codes with one ue8m0 scale per 32 contiguous values (the released W8A8 rule, D39 rule 1)",
                    "block": 32,
                    "codes_sha256": self.wo_b_activation_codes_sha256,
                    "scales_sha256": self.wo_b_activation_scales_sha256,
                    "code_bytes": self.wo_b_activation_code_bytes,
                    "scale_bytes": self.wo_b_activation_scale_bytes,
                },
                "partial_sha256": self.wo_b_partial_sha256,
                "partial_l2": self.wo_b_partial_l2,
                "note": "the Row plan gives this rank an inner-dimension slice; the partial is not the layer output until the attention all-reduce sums the ranks",
            },
            "attention_all_reduce": {
                "transport": "tcp canonical bf16 all-reduce, ascending rank order",
                "elements": HIDDEN,
                "sequence": self.reduce_sequence,
                "calls": self.reduce_calls,
                "reduced_sha256": self.reduced_sha256,
                "reduced_l2": self.reduced_l2,
                "exact": "the same canonical reduce the walk uses; the sum is over bf16 addends in ascending-rank order",
            },
            "exact": "wo_a is the released rule's BF16-activation grouped projection (BF16 activation, B12x FP8 weights, ue8m0 scales, no activation quantiser; D39 task-29); wo_b is the native dense family with the E4M3/ue8m0 activation quantiser and the row shard's inner slice",
            "not_reference": "the attention delta still comes from the one-window-row smoke body, not the reference 128-row attention, so no token can be claimed from it",
        })
    }
}

/// The attention output block's seven pooled device buffers.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
struct AttentionOutputBuffers {
    /// The activation row byte count the buffers were sized for.
    activation_bytes: usize,
    attn: Deviceptr,
    lo_out: Deviceptr,
    partial: Deviceptr,
    wo_b_act_codes: Deviceptr,
    wo_b_act_scales: Deviceptr,
    wo_a_ring: Deviceptr,
    wo_b_ring: Deviceptr,
}

/// The attention output block's reusable buffers.
///
/// The block runs once per layer per position, and it used to check seven pooled device buffers out
/// and back in and allocate four host vectors on every call. None of that is arithmetic: every
/// buffer is fully written before it is read, so holding them for the whole loop changes no byte a
/// kernel or the reduce consumes. The trace rings are still zero-filled per call, because the
/// launches hand them to the kernels as the trace ring.
#[cfg(feature = "cuda")]
#[derive(Debug, Default)]
pub struct AttentionOutputScratch {
    buffers: Option<AttentionOutputBuffers>,
    lo_raw: Vec<u8>,
    partial_raw: Vec<u8>,
    words: Vec<u16>,
    lo_values: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl AttentionOutputScratch {
    /// An empty slot: the first call checks the device buffers out, every later call reuses them.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The seven device buffers for this shape, checked out of the run-scope pool once.
    fn buffers(
        &mut self,
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        activation_bytes: usize,
        groups: usize,
        o_lora_rank: usize,
        wo_b_inner: usize,
    ) -> Result<AttentionOutputBuffers, String> {
        if let Some(buffers) = self.buffers {
            if buffers.activation_bytes == activation_bytes {
                return Ok(buffers);
            }
            // The shape is constant across layers; a change means a different plan, so the old
            // buffers go back to the pool rather than being handed out for the wrong sizes.
            self.release(cuda);
        }
        let mut allocated = 0usize;
        let mut alloc = |bytes: usize, what: &str| -> Result<Deviceptr, String> {
            cuda.alloc_pooled(bytes, &mut allocated)
                .map_err(|error| format!("layer-0 attention output: {what}: {error}"))
        };
        let buffers = AttentionOutputBuffers {
            activation_bytes,
            attn: alloc(activation_bytes, "attention buffer")?,
            lo_out: alloc(groups * o_lora_rank * 2, "wo_a output buffer")?,
            partial: alloc(HIDDEN * 2, "wo_b output buffer")?,
            wo_b_act_codes: alloc(wo_b_inner, "wo_b act code buffer")?,
            wo_b_act_scales: alloc(wo_b_inner.div_ceil(32), "wo_b act scale buffer")?,
            wo_a_ring: alloc(TRACE_RING_BYTES, "wo_a trace ring")?,
            wo_b_ring: alloc(TRACE_RING_BYTES, "wo_b trace ring")?,
        };
        self.buffers = Some(buffers);
        Ok(buffers)
    }

    /// Returns the buffers to the run-scope pool and drops the host rows.
    pub fn release(&mut self, cuda: &sparkflash_kernels::cuda_driver::Cuda) {
        if let Some(buffers) = self.buffers.take() {
            cuda.release_pooled(&[
                &buffers.attn,
                &buffers.lo_out,
                &buffers.partial,
                &buffers.wo_b_act_codes,
                &buffers.wo_b_act_scales,
                &buffers.wo_a_ring,
                &buffers.wo_b_ring,
            ]);
        }
        self.lo_raw = Vec::new();
        self.partial_raw = Vec::new();
        self.words = Vec::new();
        self.lo_values = Vec::new();
    }
}

/// Runs the layer-0 attention output projection pair and the attention all-reduce.
///
/// The grouped wo_a reads the arena's per-rank B12x weight and scale grid (the plan keeps two group
/// buffers per rank), the row-sharded wo_b reduces over that rank's inner slice, and the canonical
/// bf16 all-reduce sums the four partials into the attention block's delta. This is still the smoke
/// body's output, so the delta is evidence of the projection wiring and the collective, not the
/// reference layer output.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_layer_attention_output(
    layer: usize,
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    kernels: &mut KernelCache,
    request: &TokenStageRequest<'_>,
    bind: &crate::tp4_resident::DeviceBind,
    body: &AttentionStep,
    reduce: &ForwardReduce<'_>,
    position: usize,
    parity: &mut NativeDenseParity,
    scratch: &mut AttentionOutputScratch,
) -> Result<Layer0AttentionOutput, String> {
    use sparkflash_comm::layout::{CollectiveLayout, FabricSchedule};
    use sparkflash_kernels::bf16a_mxfp8w_wo_a_launch as wo_a_launch;
    use sparkflash_kernels::mxfp8_dense_launch;
    use sparkflash_kernels::mxfp8_gemv_launch;

    let config = request.config.ok_or_else(|| {
        "layer-0 attention output: the loop was built without the model config".to_string()
    })?;
    let world = request.world;
    if world == 0 || !config.o_groups.is_multiple_of(world) {
        return Err(format!(
            "layer-0 attention output: {} output groups do not split into {world} ranks",
            config.o_groups
        ));
    }
    // The arena's wo_a/wo_b shards are per-rank: wo_a splits its stored 8192 rows into world row
    // blocks (2 groups each at world 4), and wo_b splits its input axis the same way. The launch
    // geometry must be the per-rank group count, not the config's global one.
    let groups = config.o_groups / world;
    let o_lora_rank = config.o_lora_rank;
    if groups == 0 || o_lora_rank == 0 {
        return Err(format!(
            "layer-0 attention output: the config has {groups} output groups of rank {o_lora_rank}"
        ));
    }
    let heads = body.heads_per_rank;
    if heads == 0 || !heads.is_multiple_of(groups) {
        return Err(format!(
            "layer-0 attention output: {heads} per-rank heads do not split into {groups} output groups"
        ));
    }
    let heads_per_group = heads / groups;
    let group_inner = heads_per_group * body.head_width;
    if body.activation_bf16.len() != groups * group_inner * 2 {
        return Err(format!(
            "layer-0 attention output: the body wrote {} bytes, {groups} groups of {group_inner} need {}",
            body.activation_bf16.len(),
            groups * group_inner * 2
        ));
    }
    let wo_a_geometry = bind
        .plan_index
        .b12x_geometry(&format!("layers.{layer}.attn.wo_a.weight"))?;
    if wo_a_geometry != (groups * o_lora_rank, group_inner) {
        return Err(format!(
            "layer-0 attention output: the plan's wo_a is {wo_a_geometry:?}, the per-rank launch expects {:?} (world {world}, o_groups {} -> {groups} groups/rank, heads_per_rank {heads}, head_width {}, group_inner {group_inner}, o_lora_rank {o_lora_rank})",
            (groups * o_lora_rank, group_inner),
            config.o_groups,
            body.head_width,
            heads = body.heads_per_rank
        ));
    }
    let wo_b_geometry = bind
        .plan_index
        .b12x_geometry(&format!("layers.{layer}.attn.wo_b.weight"))?;
    let wo_b_inner = groups * o_lora_rank;
    if wo_b_geometry != (HIDDEN, wo_b_inner) {
        return Err(format!(
            "layer-0 attention output: the plan's wo_b is {wo_b_geometry:?}, the per-rank launch expects {:?} (world {world}, {groups} groups/rank x o_lora_rank {o_lora_rank})",
            (HIDDEN, wo_b_inner)
        ));
    }
    let pair_for = |name: &str| -> Result<(Deviceptr, Deviceptr), String> {
        let pair = bind
            .attention
            .pointers
            .get(name)
            .ok_or_else(|| format!("layer-0 attention output: the bind has no {name} pair"))?;
        if pair.0.0 == 0 || pair.1.0 == 0 {
            return Err(format!(
                "layer-0 attention output: {name} bound a null pointer; refusing to launch"
            ));
        }
        Ok(*pair)
    };
    let (wo_a_weight, wo_a_scales) = pair_for(&format!("layers.{layer}.attn.wo_a.weight"))?;
    let (wo_b_weight, wo_b_scales) = pair_for(&format!("layers.{layer}.attn.wo_b.weight"))?;
    let kernel_dir = request.kernel_cubin_dir.ok_or_else(|| {
        "layer-0 attention output: the loop has no kernel cubin directory".to_string()
    })?;
    // M3/C1: both projection modules are loaded once per run by the loop's kernel cache and reused
    // by every layer; this block's only per-layer work is the digest-keyed lookup.
    let (wo_a_kernel, wo_a_cubin_sha256) =
        kernels.wo_a(cuda, kernel_dir, "layer-0 attention output")?;
    // WS-D: wo_b runs through the dedicated decode GEMV (one output column per lane, one warp per
    // 32-output B12x micro-tile), its own symbol and cubin, verified against the baseline by name.
    let (wo_b_kernel, wo_b_cubin_sha256) =
        kernels.gemv(cuda, kernel_dir, "layer-0 attention output")?;

    let buffers = scratch.buffers(
        cuda,
        body.activation_bf16.len(),
        groups,
        o_lora_rank,
        wo_b_inner,
    )?;
    let attn = buffers.attn;
    let lo_out = buffers.lo_out;
    let partial = buffers.partial;
    // The native wo_b GEMM's W8A8 activation: one row of E4M3 codes plus one ue8m0 scale per 32
    // values. The row is the wo_a BF16 output, so the buffers are the wo_b inner width.
    let wo_b_act_codes = buffers.wo_b_act_codes;
    let wo_b_act_scales = buffers.wo_b_act_scales;
    let wo_a_ring = buffers.wo_a_ring;
    let wo_b_ring = buffers.wo_b_ring;
    // The rings stay zero-filled per call: the launches hand them to the kernels as the trace ring,
    // so the loop keeps their bytes deterministic.
    cuda.copy_htod(&wo_a_ring, &[0u8; TRACE_RING_BYTES])
        .map_err(|error| format!("layer-0 attention output: wo_a trace ring zero fill: {error}"))?;
    cuda.copy_htod(&wo_b_ring, &[0u8; TRACE_RING_BYTES])
        .map_err(|error| format!("layer-0 attention output: wo_b trace ring zero fill: {error}"))?;

    let mut lo_raw = std::mem::take(&mut scratch.lo_raw);
    let mut partial_raw = std::mem::take(&mut scratch.partial_raw);
    let mut words = std::mem::take(&mut scratch.words);
    let mut lo_values = std::mem::take(&mut scratch.lo_values);
    let staged = (|| -> Result<Layer0AttentionOutput, String> {
        cuda.copy_htod(&attn, &body.activation_bf16)
            .map_err(|error| format!("layer-0 attention output: attention htod: {error}"))?;
        let wo_a_params = wo_a_launch::Bf16aMxfp8wWoAParams::new(
            attn,
            wo_a_weight,
            wo_a_scales,
            lo_out,
            groups as u32,
            wo_a_ring.0,
            0,
            0,
        );
        wo_a_launch::launch(cuda, wo_a_kernel, &wo_a_params)?;
        lo_raw.clear();
        lo_raw.resize(groups * o_lora_rank * 2, 0);
        cuda.copy_dtoh(&mut lo_raw, &lo_out)
            .map_err(|error| format!("layer-0 attention output: wo_a dtoh: {error}"))?;
        bf16_from_le_bytes_into(&mut lo_values, &lo_raw)?;
        // C10: the finite scan and the L2 below are read only by the layer-chain record and the run
        // evidence document, so a default or --forward-lean run (evidence off) skips them. The
        // quantiser still consumes lo_values, which is the numeric path.
        let wo_a_out_l2 = if layer_evidence_readbacks() {
            if lo_values.iter().any(|value| !value.is_finite()) {
                return Err(
                    "layer-0 attention output: wo_a wrote a non-finite value; refusing to reduce it"
                        .to_string(),
                );
            }
            lo_values
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
        } else {
            0.0
        };
        // D39 rule 1: the wo_b activation is the wo_a BF16 output, narrowed to BF16 (idempotent on
        // a value already read back as bf16) then quantised to E4M3 codes with ue8m0 scales,
        // exactly as the single-node native chain does at this seam.
        let (wo_b_codes, wo_b_scale_bytes) =
            quantise_native_activation(&lo_values, "layer-0 attention output")?;
        cuda.copy_htod(&wo_b_act_codes, &wo_b_codes)
            .map_err(|error| format!("layer-0 attention output: wo_b act code htod: {error}"))?;
        cuda.copy_htod(&wo_b_act_scales, &wo_b_scale_bytes)
            .map_err(|error| format!("layer-0 attention output: wo_b act scale htod: {error}"))?;
        let wo_b_params = mxfp8_dense_launch::Mxfp8DenseParams::new(
            wo_b_act_codes,
            wo_b_act_scales,
            wo_b_weight,
            wo_b_scales,
            partial,
            1,
            HIDDEN as u32,
            wo_b_inner as u32,
            wo_b_ring.0,
            layer as u32,
            0,
        );
        mxfp8_gemv_launch::launch(cuda, wo_b_kernel, &wo_b_params)?;
        partial_raw.clear();
        partial_raw.resize(HIDDEN * 2, 0);
        cuda.copy_dtoh(&mut partial_raw, &partial)
            .map_err(|error| format!("layer-0 attention output: wo_b dtoh: {error}"))?;
        // wo_b's inner axis is TP4-sharded: the parity re-launch compares this rank's partial-GEMM
        // launch on its own shard bytes and claims no world-1 full-inner bit equality (the
        // cross-rank f32 reduce order is a designed non-associativity).
        parity.capture(
            cuda,
            request,
            layer,
            position,
            "layer0-attention-output",
            "wo_b",
            &format!("layers.{layer}.attn.wo_b.weight"),
            (HIDDEN, wo_b_inner),
            &lo_values,
            &wo_b_codes,
            &wo_b_scale_bytes,
            wo_b_weight,
            wo_b_scales,
            &partial_raw,
        );
        let wo_b_partial_l2 = if layer_evidence_readbacks() {
            let partial_values = bf16_from_le_bytes(&partial_raw)?;
            if partial_values.iter().any(|value| !value.is_finite()) {
                return Err(
                    "layer-0 attention output: wo_b wrote a non-finite value; refusing to reduce it"
                        .to_string(),
                );
            }
            partial_values
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
        } else {
            0.0
        };
        // The two Vec conversions below (the all-reduce's u16 words and the reduced row's bytes)
        // and the collective layout are arithmetic - the all-reduce consumes the words and the
        // residual seam consumes the reduced row - so they stay unconditional.
        words.clear();
        words.extend(
            partial_raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_le_bytes(*pair)),
        );
        let layout = CollectiveLayout::new(request.world, HIDDEN, FabricSchedule::OneShotAllToAll)
            .map_err(|error| format!("layer-0 attention output: collective layout: {error}"))?;
        let reduce_sequence = ATTENTION_REDUCE_SEQ.wrapping_sub(body.position as u64);
        reduce
            .all_reduce_bf16(&mut words, reduce_sequence, &layout)
            .map_err(|error| format!("layer-0 attention output: all-reduce: {error}"))?;
        let reduced_raw: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let reduced_l2 = if layer_evidence_readbacks() {
            let reduced_values = bf16_from_le_bytes(&reduced_raw)?;
            reduced_values
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
        } else {
            0.0
        };
        Ok(Layer0AttentionOutput {
            groups,
            group_inner,
            world,
            o_groups_global: config.o_groups,
            heads_per_rank: heads,
            head_width: body.head_width,
            o_lora_rank,
            wo_a_cubin_sha256,
            wo_a_out_sha256: evidence_sha256(&lo_raw),
            wo_a_out_l2,
            wo_b_cubin_sha256,
            wo_b_activation_codes_sha256: evidence_sha256(&wo_b_codes),
            wo_b_activation_scales_sha256: evidence_sha256(&wo_b_scale_bytes),
            wo_b_activation_code_bytes: wo_b_codes.len(),
            wo_b_activation_scale_bytes: wo_b_scale_bytes.len(),
            wo_b_partial_sha256: evidence_sha256(&partial_raw),
            wo_b_partial_l2,
            reduced_sha256: evidence_sha256(&reduced_raw),
            reduced_l2,
            reduce_calls: 1,
            reduce_sequence,
            reduced_bf16: reduced_raw,
        })
    })();
    scratch.lo_raw = lo_raw;
    scratch.partial_raw = partial_raw;
    scratch.words = words;
    scratch.lo_values = lo_values;
    staged
}

/// The layer-0 attention residual seam: the reduced attention delta mixed back onto the four hc
/// residual streams with the released hc_post arithmetic.
#[derive(Debug, Clone)]
pub struct AttentionResidualSeam {
    /// sha256 of the BF16 attention delta the projection pair and the all-reduce produced.
    pub attn_out_sha256: String,
    /// sha256 of the f32 residual stream the seam consumed.
    pub residual_sha256: String,
    /// sha256 of the post-mix coefficients.
    pub post_sha256: String,
    /// sha256 of the comb matrix.
    pub comb_sha256: String,
    /// sha256 of the updated hc residual stream (f32-le).
    pub out_sha256: String,
    /// The updated stream's L2 norm.
    pub out_l2: f32,
    /// The L2 norm of each updated stream copy, in copy order.
    pub stream_l2: Vec<f32>,
    /// The updated hc residual stream (the FFN/MoE stage's input).
    pub out: Vec<f32>,
}

impl AttentionResidualSeam {
    /// The evidence block for the refused run.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stage": "layer0-attention-residual-hc-post",
            "attention_output": {
                "encoding": "the wo_b dense output narrowed to BF16, then the canonical bf16 all-reduce sum",
                "sha256": self.attn_out_sha256,
                "width": HIDDEN,
            },
            "residual": {
                "streams": HC_MULT,
                "sha256": self.residual_sha256,
                "note": "the hc residual stream the mHC entry seam consumed",
            },
            "mixes": {
                "post_sha256": self.post_sha256,
                "comb_sha256": self.comb_sha256,
            },
            "out": {
                "sha256": self.out_sha256,
                "l2": self.out_l2,
                "stream_l2": self.stream_l2,
            },
            "exact": "engine mhc::hc_post over the reduced attention delta, the hc residual stream, the post mixes and the sinkhorn comb matrix; the delta enters as the device's BF16 output decoded to f32",
            "not_reference": "the delta still comes from the one-window-row smoke body, so this is not the reference layer-0 attention residual and no token can be claimed from it",
        })
    }
}

/// Computes the layer-0 attention residual on the host: engine mhc::hc_post over the all-reduced
/// attention delta, the hc residual stream the mHC entry consumed, and that entry's post mixes and
/// comb matrix.
///
/// # Errors
/// Refuses a delta that is not one dim-wide BF16 row, a residual stream that is not hc copies of
/// dim, any non-finite output, and propagates hc_post's own shape refusals.
pub fn attention_residual_host_seam(
    config: &sparkflash_engine::config::ModelConfig,
    entry: &MhcEntrySeam,
    attention_delta_bf16: &[u8],
) -> Result<AttentionResidualSeam, String> {
    attention_residual_host_seam_with(config, entry, attention_delta_bf16, false)
}

/// [attention_residual_host_seam] with native_mhc, the M4 arm's BF16 boundaries: hc_post keeps the
/// release expression order and rounds the updated stream to BF16, the dtype the block carries
/// between sublayers (model.py:962-966, 976-994).
///
/// # Errors
/// [attention_residual_host_seam]'s refusals.
pub fn attention_residual_host_seam_with(
    config: &sparkflash_engine::config::ModelConfig,
    entry: &MhcEntrySeam,
    attention_delta_bf16: &[u8],
    native_mhc: bool,
) -> Result<AttentionResidualSeam, String> {
    use sparkflash_engine::forward::mhc;

    let dim = config.dim;
    let hc = config.hc_mult;
    if hc != HC_MULT {
        return Err(format!(
            "attention residual: the released seam is the {HC_MULT}-stream reference, config declares {hc}"
        ));
    }
    let delta = bf16_from_le_bytes(attention_delta_bf16)?;
    if delta.len() != dim {
        return Err(format!(
            "attention residual: the delta is {} values, one row is {dim}",
            delta.len()
        ));
    }
    // C10: the scans and the two L2 norms below are read only by the layer-chain record and the run
    // evidence document, so a default or --forward-lean run (evidence off) skips them.
    if layer_evidence_readbacks() && delta.iter().any(|value| !value.is_finite()) {
        return Err("attention residual: the delta has a non-finite value".to_string());
    }
    let residual = &entry.hidden;
    if residual.len() != hc * dim {
        return Err(format!(
            "attention residual: the residual stream is {} values, expected {hc} copies of {dim}",
            residual.len()
        ));
    }
    let mut out = vec![0.0f32; hc * dim];
    if native_mhc {
        mhc::hc_post_bf16(config, &delta, residual, &entry.post, &entry.comb, &mut out)
            .map_err(|error| format!("attention residual: hc_post: {error}"))?;
    } else {
        mhc::hc_post(config, &delta, residual, &entry.post, &entry.comb, &mut out)
            .map_err(|error| format!("attention residual: hc_post: {error}"))?;
    }
    let (out_l2, stream_l2) = if layer_evidence_readbacks() {
        if out.iter().any(|value| !value.is_finite()) {
            return Err(
                "attention residual: the updated stream has a non-finite value".to_string(),
            );
        }
        let out_l2 = out.iter().map(|value| value * value).sum::<f32>().sqrt();
        let stream_l2: Vec<f32> = out
            .chunks_exact(dim)
            .map(|stream| stream.iter().map(|value| value * value).sum::<f32>().sqrt())
            .collect();
        (out_l2, stream_l2)
    } else {
        (0.0, Vec::new())
    };
    Ok(AttentionResidualSeam {
        attn_out_sha256: evidence_sha256(attention_delta_bf16),
        residual_sha256: evidence_f32_sha256(residual),
        post_sha256: evidence_f32_sha256(&entry.post),
        comb_sha256: evidence_f32_sha256(&entry.comb),
        out_sha256: evidence_f32_sha256(&out),
        out_l2,
        stream_l2,
        out,
    })
}

/// The layer-0 FFN host seams: the FFN mHC mixes, the FFN block input, the pre-FFN norm and the
/// activation quantiser, i.e. everything the device router/gate-up/swiglu/combine launches consume.
#[derive(Debug, Clone)]
pub struct Layer0FfnSeams {
    /// sha256 of the FFN pre mixes.
    pub ffn_pre_sha256: String,
    /// The FFN pre mixes themselves: the released chain hands this set to the NEXT layer as the
    /// attention hc_pre mix (hc_in_pre(L+1) == hc_ffn_pre(L) in the phase1-boundary v1 capture).
    pub ffn_pre: Vec<f32>,
    /// sha256 of the FFN post mixes.
    pub ffn_post_sha256: String,
    /// sha256 of the FFN comb matrix.
    pub ffn_comb_sha256: String,
    /// The FFN post mixes (kept for the ffn hc_post).
    pub ffn_post: Vec<f32>,
    /// The FFN comb matrix (kept for the ffn hc_post).
    pub ffn_comb: Vec<f32>,
    /// The FFN block input (hc_pre of the updated stream with the ATTENTION predictor's pre mix).
    pub ffn_in: Vec<f32>,
    /// sha256 of [ffn_in] as f32-le.
    pub ffn_in_sha256: String,
    /// The pre-FFN norm output, in f32.
    pub pre_ffn_hidden: Vec<f32>,
    /// sha256 of [pre_ffn_hidden] as f32-le.
    pub pre_ffn_hidden_sha256: String,
    /// The E4M3 activation codes, one per element.
    pub act_fp8: Vec<u8>,
    /// sha256 of the activation codes.
    pub act_fp8_sha256: String,
    /// The E8M0 activation scale codes, one per 32-element block.
    pub act_scales: Vec<u8>,
    /// sha256 of the activation scale codes.
    pub act_scales_sha256: String,
}

impl Layer0FfnSeams {
    /// The evidence block for the refused run.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stage": "layer0-ffn-host-seams",
            "ffn_mixes": {
                "pre_sha256": self.ffn_pre_sha256,
                "post_sha256": self.ffn_post_sha256,
                "comb_sha256": self.ffn_comb_sha256,
                "source": "layers.0.hc_ffn_fn/hc_ffn_scale/hc_ffn_base from the live arena",
            },
            "ffn_in": {
                "sha256": self.ffn_in_sha256,
                "bytes": self.ffn_in.len() * 4,
                "arm": "hc_pre with the ATTENTION predictor's pre mix (the FFN block input mix is the attention block's)",
                "l2": self.ffn_in.iter().map(|value| value * value).sum::<f32>().sqrt(),
            },
            "pre_ffn_hidden": {
                "sha256": self.pre_ffn_hidden_sha256,
                "bytes": self.pre_ffn_hidden.len() * 4,
                "l2": self.pre_ffn_hidden.iter().map(|value| value * value).sum::<f32>().sqrt(),
                "rule": "rms_norm(ffn_in, layers.0.ffn_norm.weight, 5120, norm_eps), f32 output",
            },
            "activation": {
                "codes_sha256": self.act_fp8_sha256,
                "scales_sha256": self.act_scales_sha256,
                "codes_bytes": self.act_fp8.len(),
                "scales_bytes": self.act_scales.len(),
                "rule": "D8 quantize_activation_fp8 over the f32 pre-FFN hidden row: E4M3 codes, E8M0 scales per 32-element block",
            },
            "router_input": {
                "rule": "the router consumes the BF16 RNE round of pre_ffn_hidden, derived at the launch; the activation quantiser still takes the f32 row",
                "bf16_sha256": sha256_hex(&crate::tp4_forward::bf16_le_bytes(&self.pre_ffn_hidden)),
            },
            "note": "the five device MoE launches (router, gate/up x2, swiglu quant, stored combine) are not wired yet; these seams are their exact inputs",
            "not_reference": "no router ids, no expert output, no MoE all-reduce and no ffn hc_post ran, so no layer output and no token can be claimed",
        })
    }
}

/// The layer-0 FFN host seams on the host, from the released tensors.
///
/// # Errors
/// Refuses wrong shapes or encodings, a wrong hc_mult, non-finite intermediate values, and
/// propagates hc_mixes/hc_pre/rms_norm/quantise errors.
pub fn layer0_ffn_host_seams(
    config: &sparkflash_engine::config::ModelConfig,
    post_attention_hidden: &[f32],
    attn_pre_mix: &[f32],
    hc_ffn_fn: &[f32],
    hc_ffn_base: &[f32],
    hc_ffn_scale: &[f32],
    ffn_norm: &[f32],
) -> Result<Layer0FfnSeams, String> {
    layer0_ffn_host_seams_with(
        config,
        post_attention_hidden,
        attn_pre_mix,
        hc_ffn_fn,
        hc_ffn_base,
        hc_ffn_scale,
        ffn_norm,
        false,
    )
}

/// [layer0_ffn_host_seams] with native_mhc: the hc_pre collapse is rounded to BF16 before the norm
/// and the norm output is rounded to BF16 before the activation quantiser, which is the boundary
/// the released block hands to the routed MoE (model.py:288-293, 954-957, 976-994).
///
/// # Errors
/// [layer0_ffn_host_seams]'s refusals.
#[allow(clippy::too_many_arguments)]
pub fn layer0_ffn_host_seams_with(
    config: &sparkflash_engine::config::ModelConfig,
    post_attention_hidden: &[f32],
    attn_pre_mix: &[f32],
    hc_ffn_fn: &[f32],
    hc_ffn_base: &[f32],
    hc_ffn_scale: &[f32],
    ffn_norm: &[f32],
    native_mhc: bool,
) -> Result<Layer0FfnSeams, String> {
    layer0_ffn_host_seams_with_transposed(
        config,
        post_attention_hidden,
        attn_pre_mix,
        hc_ffn_fn,
        hc_ffn_base,
        hc_ffn_scale,
        ffn_norm,
        native_mhc,
        None,
    )
}

/// [layer0_ffn_host_seams_with] with the optional transposed `hc_ffn_fn` the run-scope seam cache stages:
/// `Some` evaluates the projection with [sparkflash_engine::forward::mhc::hc_mixes_transposed]'s
/// interleaved loop order on the transposed copy, `None` uses [sparkflash_engine::forward::mhc::hc_mixes]'s
/// own order. Both produce the same f32 values, bit for bit.
///
/// # Errors
/// [layer0_ffn_host_seams]'s refusals.
#[allow(clippy::too_many_arguments)]
fn layer0_ffn_host_seams_with_transposed(
    config: &sparkflash_engine::config::ModelConfig,
    post_attention_hidden: &[f32],
    attn_pre_mix: &[f32],
    hc_ffn_fn: &[f32],
    hc_ffn_base: &[f32],
    hc_ffn_scale: &[f32],
    ffn_norm: &[f32],
    native_mhc: bool,
    transposed_predictor: Option<&[f32]>,
) -> Result<Layer0FfnSeams, String> {
    use sparkflash_engine::forward::mhc;
    use sparkflash_engine::linalg;
    use sparkflash_engine::quant;
    let hc = config.hc_mult;
    let dim = config.dim;
    if hc != HC_MULT {
        return Err(format!(
            "FFN seams: the released seam is the {HC_MULT}-stream reference, config declares {hc}"
        ));
    }
    if post_attention_hidden.len() != hc * dim {
        return Err(format!(
            "FFN seams: the post-attention stream is {} values, expected {hc} copies of {dim}",
            post_attention_hidden.len()
        ));
    }
    if attn_pre_mix.len() != hc {
        return Err(format!(
            "FFN seams: the attention pre mix is {} values, expected {hc}",
            attn_pre_mix.len()
        ));
    }
    if hc_ffn_fn.len() != PREDICTOR_ROWS * hc * dim {
        return Err(format!(
            "FFN seams: hc_ffn_fn is {} values, expected {}",
            hc_ffn_fn.len(),
            PREDICTOR_ROWS * hc * dim
        ));
    }
    if let Some(weight) =
        transposed_predictor.filter(|weight| weight.len() != PREDICTOR_ROWS * hc * dim)
    {
        return Err(format!(
            "FFN seams: the transposed hc_ffn_fn is {} values, expected {}",
            weight.len(),
            PREDICTOR_ROWS * hc * dim
        ));
    }
    if hc_ffn_base.len() != PREDICTOR_ROWS || hc_ffn_scale.len() != 3 {
        return Err(format!(
            "FFN seams: hc_ffn_base is {} and hc_ffn_scale is {}; expected 24 and 3",
            hc_ffn_base.len(),
            hc_ffn_scale.len()
        ));
    }
    if ffn_norm.len() != dim {
        return Err(format!(
            "FFN seams: ffn_norm is {} values, expected {dim}",
            ffn_norm.len()
        ));
    }
    let mut ffn_mixes = mhc::Mixes::for_config(config);
    match transposed_predictor {
        Some(weight) => mhc::hc_mixes_transposed(
            config,
            post_attention_hidden,
            weight,
            hc_ffn_scale,
            hc_ffn_base,
            &mut ffn_mixes,
        ),
        None => mhc::hc_mixes(
            config,
            post_attention_hidden,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
            &mut ffn_mixes,
        ),
    }
    .map_err(|error| format!("FFN seams: hc_mixes: {error}"))?;
    let mut ffn_in = vec![0.0f32; dim];
    let mut pre_ffn_hidden;
    if native_mhc {
        mhc::hc_pre_bf16(config, attn_pre_mix, post_attention_hidden, &mut ffn_in)
            .map_err(|error| format!("FFN seams: hc_pre: {error}"))?;
        pre_ffn_hidden = ffn_in.clone();
        mhc::rms_norm_bf16(&mut pre_ffn_hidden, ffn_norm, dim, config.norm_eps as f32)
            .map_err(|error| format!("FFN seams: ffn_norm: {error}"))?;
    } else {
        mhc::hc_pre(config, attn_pre_mix, post_attention_hidden, &mut ffn_in)
            .map_err(|error| format!("FFN seams: hc_pre: {error}"))?;
        pre_ffn_hidden = ffn_in.clone();
        linalg::rms_norm(&mut pre_ffn_hidden, ffn_norm, dim, config.norm_eps as f32)
            .map_err(|error| format!("FFN seams: ffn_norm: {error}"))?;
    }
    let quantised = quant::quantize_activation_fp8(&pre_ffn_hidden)
        .map_err(|error| format!("FFN seams: activation quantiser: {error}"))?;
    for (name, values) in [("ffn_in", &ffn_in), ("pre_ffn_hidden", &pre_ffn_hidden)] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "FFN seams: the {name} output has a non-finite value; refusing to stage it"
            ));
        }
    }
    Ok(Layer0FfnSeams {
        ffn_pre_sha256: evidence_f32_sha256(&ffn_mixes.pre),
        ffn_pre: ffn_mixes.pre,
        ffn_post_sha256: evidence_f32_sha256(&ffn_mixes.post),
        ffn_comb_sha256: evidence_f32_sha256(&ffn_mixes.comb),
        ffn_post: ffn_mixes.post,
        ffn_comb: ffn_mixes.comb,
        ffn_in_sha256: evidence_f32_sha256(&ffn_in),
        ffn_in,
        pre_ffn_hidden_sha256: evidence_f32_sha256(&pre_ffn_hidden),
        pre_ffn_hidden,
        act_fp8_sha256: evidence_sha256(quantised.codes()),
        act_fp8: quantised.codes().to_vec(),
        act_scales_sha256: evidence_sha256(quantised.scales()),
        act_scales: quantised.scales().to_vec(),
    })
}

/// Runs the layer-0 FFN host seams from the run-scope seam cache: the four FFN tensors were read
/// and decoded once per layer, and the released arithmetic runs on the host exactly as before.
#[cfg(any(feature = "cuda", test))]
fn run_layer_ffn_seams(
    cached: &LayerSeamTensors,
    request: &TokenStageRequest<'_>,
    residual: &AttentionResidualSeam,
    attn_pre_mix: &[f32],
) -> Result<Layer0FfnSeams, String> {
    let config = request
        .config
        .ok_or_else(|| "FFN seams: the loop was built without the model config".to_string())?;
    layer0_ffn_host_seams_with_transposed(
        config,
        &residual.out,
        attn_pre_mix,
        &cached.ffn_fn,
        &cached.ffn_base,
        &cached.ffn_scale,
        &cached.ffn_norm,
        request.native_rounding.mhc,
        Some(&cached.ffn_fn_transposed),
    )
}

/// The layer-0 MoE routed chain: the five device launches, the rank combine, the canonical reduce
/// and the per-stage digests the task-11 contract names.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct Layer0MoeRouted {
    /// This rank.
    pub rank: usize,
    /// The five cubin pins (name, bytes, sha256), for the evidence record.
    pub pins: Vec<(String, usize, String)>,
    /// Router ids in row order (slot indices into the sorted unique selection).
    pub router_ids: Vec<u32>,
    /// The router's normalised weights including routed_scaling_factor (evidence record only; empty
    /// when the evidence chain is off).
    pub router_weights: Vec<f32>,
    /// The router's full score vectors (evidence record only; empty when the evidence chain is off).
    pub router_scores: Vec<f32>,
    /// The sorted unique global expert ids.
    pub slot_ids: Vec<u32>,
    /// The router stage-two ids (global), for the record.
    pub global_ids: Vec<u32>,
    /// The gate projection output (BF16 bytes; evidence record only, empty when evidence is off).
    pub gate_raw: Vec<u8>,
    /// The up projection output (BF16 bytes; evidence record only, empty when evidence is off).
    pub up_raw: Vec<u8>,
    /// The weighted SWiGLU MXFP8 codes (evidence record only, empty when evidence is off).
    pub swiglu_codes: Vec<u8>,
    /// The weighted SWiGLU ue8m0 scales (evidence record only, empty when evidence is off).
    pub swiglu_scales: Vec<u8>,
    /// The stored-combine rank accumulator (f32 values).
    pub slot_acc: Vec<f32>,
    /// The rank's finalised routed contribution (f32, model rule).
    pub rank_routed: Vec<f32>,
    /// The canonical-reduce payload (BF16 bytes).
    pub payload: Vec<u8>,
    /// The all-reduced routed delta (BF16 bytes).
    pub delta: Vec<u8>,
    /// The reduce sequence.
    pub reduce_sequence: u64,
    /// Whether this layer ran the M2 native combine (full f32 rank-sum gather) instead of the
    /// per-rank finalize plus BF16 all-reduce.
    pub native_combine: bool,
    /// The full f32 rank sum when the native combine ran: the epilogue adds the shared BF16 output
    /// and rounds once. None on the default f32 arm.
    pub native_full_acc: Option<Vec<f32>>,
}

#[cfg(feature = "cuda")]
impl Layer0MoeRouted {
    /// sha256 of the router id row.
    #[must_use]
    pub fn router_ids_sha256(&self) -> String {
        evidence_sha256(
            &self
                .router_ids
                .iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
    }

    /// sha256 of the canonical-reduce payload.
    #[must_use]
    pub fn payload_sha256(&self) -> String {
        evidence_sha256(&self.payload)
    }

    /// sha256 of the all-reduced routed delta.
    #[must_use]
    pub fn delta_sha256(&self) -> String {
        evidence_sha256(&self.delta)
    }

    /// The evidence block for the refused run.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let sha = |bytes: &[u8]| sha256_hex(bytes);
        serde_json::json!({
            "schema": "sparkflash-moe-stage-evidence/1",
            "stage": "layer0-moe-routed",
            "rank": self.rank,
            "layer": 0,
            "tokens": 1,
            "ranks": 4,
            "experts": 384,
            "top_k": 6,
            "hidden": HIDDEN,
            "inter_per_rank": 576,
            "inter_padded": 640,
            "hc_mult": HC_MULT,
            "selected_rows": 6,
            "launches_per_rank": 5,
            "launches_stage": 20,
            "all_reduce_calls_per_rank": if self.native_combine { 0 } else { 1 },
            "combine_arith": if self.native_combine { "native_f32_rank_gather" } else { "rank_bf16_reduce" },
            "combine_payload_bytes_per_rank": if self.native_combine { 6 * HIDDEN * 4 } else { self.payload.len() },
            "weight_h2d_bytes_per_token": 0,
            "cubin_pins": self.pins.iter().map(|(name, bytes, digest)| serde_json::json!({"name": name, "bytes": bytes, "sha256": digest})).collect::<Vec<_>>(),
            "router": {
                "ids": self.router_ids,
                "global_ids": self.global_ids,
                "slot_ids": self.slot_ids,
                "slot_ids_sha256": sha(&self.slot_ids.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<u8>>()),
                "ids_sha256": sha(&self.router_ids.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<u8>>()),
                "weights_sha256": sha(&f32_le_bytes(&self.router_weights)),
                "scores_sha256": sha(&f32_le_bytes(&self.router_scores)),
                "stage_launches": 2,
                "note": "ids are the slot indices the gate/up and combine launches consume; slot_ids is the sorted unique global selection every rank shares",
            },
            "gate_up": {
                "gate_sha256": sha(&self.gate_raw),
                "up_sha256": sha(&self.up_raw),
                "gate_bytes": self.gate_raw.len(),
                "up_bytes": self.up_raw.len(),
                "launches": 2,
                "grid": [6, 20, 1],
                "note": "the rank's contiguous 576-row w1/w3 slices, expert-major by slot; the kernel writes the 576..639 zero padding itself",
            },
            "swiglu": {
                "codes_sha256": sha(&self.swiglu_codes),
                "scales_sha256": sha(&self.swiglu_scales),
                "codes_bytes": self.swiglu_codes.len(),
                "scales_bytes": self.swiglu_scales.len(),
                "rows": 6,
                "cols": 640,
                "rule": "weighted swiglu+MXFP8 quantiser, router weights buffer read in place, limit 10.0",
            },
            "combine": {
                "arg_words": 10,
                "grid": [1, 160, 1],
                "block": [256, 1, 1],
                "slot_acc_f32_values": self.slot_acc.len(),
                "slot_acc_sha256": sha(&f32_le_bytes(&self.slot_acc)),
                "rank_routed_f32_sha256": sha(&f32_le_bytes(&self.rank_routed)),
                "note": "the caller-zeroed accumulator, two separate stored-w2 base pointers (288 code / 18 scale per row at this rank's prefix)",
            },
            "reduce": {
                "rank_routed_f32_sha256": sha(&f32_le_bytes(&self.rank_routed)),
                "payload_sha256": sha(&self.payload),
                "routed_delta_sha256": sha(&self.delta),
                "payload_bytes": self.payload.len(),
                "calls": 1,
                "sequence": self.reduce_sequence,
                "rule": "bf16 RNE payload per rank, canonical f32 ascending-rank sum from 0.0, one bf16 RNE round",
            },
            "epilogue": {
                "shared_expert_added": false,
                "ffn_out_sha256": serde_json::Value::Null,
                "hc_post_output_sha256": serde_json::Value::Null,
                "note": "the shared expert and the ffn hc_post are not wired yet, so this routed delta is not the layer-0 FFN output",
            },
            "adjudication": {
                "router_common_by_construction": true,
                "why": "the attention delta is all-reduced before the residual, and the hc residual stream is the same gathered embedding row on every rank, so every rank's H3/pre-FFN-hidden bytes are identical; each rank still launches its own router on those bytes",
                "compare_to_m26": "loop rank r gate_up/swiglu/combine compare to m26 shard r; router ids/weights/scores compare to m26's single router; reduce compares to m26's canonical reduce over its four shard payloads, because those four payloads are the same per-rank weight-shard payloads the loop reduces",
            },
            "verdict": "measured",
            "failed_checks": [],
            "not_reference": "single token row at position 129, not the 130-token prefill; the shared expert and ffn hc_post are unwired; no token can be claimed",
        })
    }
}

/// The per-rank slice base pointers of one routed-expert matrix, accepting either the plan's sharded
/// slice shape or the released full shape (in which case the rank's slice offset is added).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn routed_slice_base(
    bind: &crate::resident_weights::MoeBind,
    plan_index: &crate::tp4_resident::PlanIndex,
    layer: usize,
    expert: usize,
    matrix: crate::resident_weights::Matrix,
    leaf: &str,
    rank: usize,
    slice_shape: [usize; 2],
) -> Result<(Deviceptr, Deviceptr), String> {
    let pair = bind.routed.get(&(layer, expert, matrix)).ok_or_else(|| {
        format!("MoE: the bind has no routed expert {expert} {leaf} (layer {layer})")
    })?;
    let weight_name = format!("layers.{layer}.ffn.experts.{expert}.{leaf}.weight");
    let scale_name = format!("layers.{layer}.ffn.experts.{expert}.{leaf}.scale");
    let slice_values = slice_shape[0] * slice_shape[1];
    let scale_slice = slice_shape[0] * (slice_shape[1] / 16);
    let weight_shape = plan_index
        .shard_shape(&weight_name)
        .ok_or_else(|| format!("MoE: the rank plan does not carry {weight_name}"))?;
    let scale_shape = plan_index
        .shard_shape(&scale_name)
        .ok_or_else(|| format!("MoE: the rank plan does not carry {scale_name}"))?;
    let weight_values = weight_shape.iter().product::<usize>();
    let weight = if weight_values == slice_values {
        pair.weight
    } else if weight_shape.len() == 2
        && weight_shape[1] == slice_shape[1]
        && weight_values == slice_shape[0] * slice_shape[1] * 4
    {
        let offset = rank
            .checked_mul(slice_values)
            .ok_or_else(|| "MoE: weight slice offset overflows".to_string())?;
        pair.weight.offset(offset)
    } else {
        return Err(format!(
            "MoE: {weight_name} shard shape {weight_shape:?} is neither this rank's {slice_shape:?} nor the released full shape; refusing to guess"
        ));
    };
    let scale_values = scale_shape.iter().product::<usize>();
    let scales = if scale_values == scale_slice {
        pair.scales
    } else if scale_values == scale_slice * 4 {
        let offset = rank
            .checked_mul(scale_slice)
            .ok_or_else(|| "MoE: scale slice offset overflows".to_string())?;
        pair.scales.offset(offset)
    } else {
        return Err(format!(
            "MoE: {scale_name} shard shape {scale_shape:?} is neither this rank's {scale_slice} values nor the released full shape; refusing to guess"
        ));
    };
    Ok((weight, scales))
}

/// Runs the layer-0 MoE routed chain on the four-rank arena: the router, the two gate/up launches,
/// the weighted SWiGLU quantiser, the stored combine, the rank combine and the canonical all-reduce.
///
/// # Errors
/// Refuses missing bind entries, plan shapes that are neither this rank's slice nor a full row
/// split, malformed router output, a non-finite accumulator and any transport failure.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_lines)]
fn run_layer_moe_routed(
    layer: usize,
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    request: &TokenStageRequest<'_>,
    bind: &crate::tp4_resident::DeviceBind,
    seams: &Layer0FfnSeams,
    transport: &TcpTransport,
    reduce: &ForwardReduce<'_>,
) -> Result<Layer0MoeRouted, String> {
    use sparkflash_comm::layout::{CollectiveLayout, FabricSchedule};
    use sparkflash_comm::transport::Transport;
    use sparkflash_kernels::cuda_driver::KernelId;
    use sparkflash_kernels::{
        mxfp4_moe_stored_launch, mxfp4_ref, routed_gate_up_launch, router_launch,
    };

    let config = request
        .config
        .ok_or_else(|| "MoE: the loop was built without the model config".to_string())?;
    let world = request.world;
    let rank = request.rank;
    let tokens = 1usize;
    let top_k = 6usize;
    let experts = config.n_routed_experts;
    let inter = config.moe_inter_dim;
    let dim = config.dim;
    if world != 4 || experts != 384 || top_k != 6 || dim != HIDDEN {
        return Err(format!(
            "MoE: this chain is the released four-rank shape (world 4, 384 experts, top-6, dim {HIDDEN}); got world {world}, {experts} experts, top-{top_k}, dim {dim}"
        ));
    }
    if !inter.is_multiple_of(world) || inter / world != 576 {
        return Err(format!(
            "MoE: the released split is 576 rows per rank, this config gives {inter} over {world}"
        ));
    }
    let kernel_dir = request
        .kernel_cubin_dir
        .ok_or_else(|| "MoE: the loop has no kernel cubin directory".to_string())?;
    type CubinPin = (std::sync::Arc<[u8]>, (String, usize, String));
    let read_pin = |name: &str, want: usize| -> Result<CubinPin, String> {
        let (bytes, digest) = cached_cubin(&kernel_dir.join(name))
            .map_err(|error| format!("MoE: read {name}: {error}"))?;
        if bytes.len() != want {
            return Err(format!(
                "MoE: {name} is {} bytes, the pin is {want}",
                bytes.len()
            ));
        }
        let pin = (name.to_string(), bytes.len(), digest);
        Ok((bytes, pin))
    };
    let (router_cubin, router_pin) = read_pin("moe_router_sm121.cubin", 27376)?;
    let (gate_up_cubin, gate_up_pin) = read_pin("routed_gate_up_sm121.cubin", 35656)?;
    let (weighted_cubin, weighted_pin) =
        read_pin("swiglu_mxfp8_quant_weighted_sm121.cubin", 14224)?;
    let (combine_cubin, combine_pin) = read_pin("b12x_mxfp4_moe_stored_sm121.cubin", 30760)?;
    let pins = vec![router_pin, gate_up_pin, weighted_pin, combine_pin];

    let routed_rows = tokens * top_k;
    let router_gate = bind
        .moe
        .router
        .get(&(layer, crate::resident_weights::RouterPart::Weight))
        .copied()
        .ok_or_else(|| "MoE: the bind has no ffn.gate.weight".to_string())?;
    let router_bias = bind
        .moe
        .router
        .get(&(layer, crate::resident_weights::RouterPart::Bias))
        .copied()
        .ok_or_else(|| "MoE: the bind has no ffn.gate.bias".to_string())?;

    let (router_module, router_kernel, topk_module, topk_kernel) =
        router_launch::load(cuda, &router_cubin)?;
    let (gate_up_module, gate_up_kernel) = routed_gate_up_launch::load(cuda, &gate_up_cubin)?;
    let (weighted_module, weighted_kernel) =
        cuda.load_kernel(&weighted_cubin, KernelId::SwigluMxfp8QuantWeighted)?;
    let (combine_module, combine_kernel) = mxfp4_moe_stored_launch::load(cuda, &combine_cubin)?;

    let mut allocated = 0usize;
    let mut alloc = |bytes: usize, what: &str| -> Result<Deviceptr, String> {
        cuda.alloc_pooled(bytes, &mut allocated)
            .map_err(|error| format!("MoE: {what}: {error}"))
    };
    let x_bf16 = alloc(tokens * dim * 2, "router x bf16")?;
    let ids = alloc(routed_rows * 4, "router ids")?;
    let weights = alloc(routed_rows * 4, "router weights")?;
    let scores = alloc(tokens * experts * 4, "router scores")?;
    let act_fp8 = alloc(tokens * dim, "activation codes")?;
    let act_scales = alloc(tokens * dim / 32, "activation scales")?;
    let gate_stage = alloc(top_k * 576 * (dim / 2), "w1 stage")?;
    let gate_scale_stage = alloc(top_k * 576 * (dim / 32), "s1 stage")?;
    let up_stage = alloc(top_k * 576 * (dim / 2), "w3 stage")?;
    let up_scale_stage = alloc(top_k * 576 * (dim / 32), "s3 stage")?;
    let gate_out = alloc(routed_rows * 640 * 2, "gate out")?;
    let up_out = alloc(routed_rows * 640 * 2, "up out")?;
    let codes = alloc(routed_rows * 640, "swiglu codes")?;
    let swiglu_scales = alloc(routed_rows * 20, "swiglu scales")?;
    let w2_stage = alloc(top_k * dim * (inter / 2 / world), "w2 stage")?;
    let w2_scale_stage = alloc(top_k * dim * (inter / 32 / world), "s2 stage")?;
    let slot_ids = alloc(routed_rows * 4, "slot ids")?;
    let acc = alloc(routed_rows * dim * 4, "stored combine accumulator")?;

    let staged = (|| -> Result<Layer0MoeRouted, String> {
        let timer = BlockTimer::start(request.forward_perf.probe);
        let x_bytes = bf16_le_bytes(&seams.pre_ffn_hidden);
        cuda.copy_htod(&x_bf16, &x_bytes)
            .map_err(|error| format!("MoE: router x htod: {error}"))?;
        cuda.copy_htod(&act_fp8, &seams.act_fp8)
            .map_err(|error| format!("MoE: activation codes htod: {error}"))?;
        cuda.copy_htod(&act_scales, &seams.act_scales)
            .map_err(|error| format!("MoE: activation scales htod: {error}"))?;
        cuda.copy_htod(&acc, &vec![0u8; routed_rows * dim * 4])
            .map_err(|error| format!("MoE: accumulator zero fill: {error}"))?;
        let router_params = router_launch::RouterLaunchParams {
            x: x_bf16,
            gate: router_gate,
            bias: router_bias,
            ids,
            weights,
            scores,
            tokens: tokens as u32,
            trace_ring: 0,
            layer_id: 0,
            step_seq: rank as u64,
        };
        router_launch::launch(cuda, router_kernel, topk_kernel, &router_params)?;
        let mut ids_raw = vec![0u8; routed_rows * 4];
        cuda.copy_dtoh(&mut ids_raw, &ids)
            .map_err(|error| format!("MoE: router ids dtoh: {error}"))?;
        let global_ids: Vec<u32> = ids_raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect();
        // The score row and the weight row are read back only for the evidence record; the launches
        // consume the device buffers in place, and the numeric host path needs the ids alone.
        let (router_weights, router_scores) = if layer_evidence_readbacks() {
            let mut weights_raw = vec![0u8; routed_rows * 4];
            cuda.copy_dtoh(&mut weights_raw, &weights)
                .map_err(|error| format!("MoE: router weights dtoh: {error}"))?;
            let mut scores_raw = vec![0u8; tokens * experts * 4];
            cuda.copy_dtoh(&mut scores_raw, &scores)
                .map_err(|error| format!("MoE: router scores dtoh: {error}"))?;
            let router_weights = f32_from_le_bytes(&weights_raw)?;
            let router_scores = f32_from_le_bytes(&scores_raw)?;
            if router_scores.iter().any(|value| !value.is_finite()) {
                return Err("MoE: the router wrote a non-finite score".to_string());
            }
            (router_weights, router_scores)
        } else {
            (Vec::new(), Vec::new())
        };
        let (router_ids, slots) = moe_slots(&global_ids, experts)?;
        let slot_raw: Vec<u8> = router_ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        cuda.copy_htod(&slot_ids, &slot_raw)
            .map_err(|error| format!("MoE: slot ids htod: {error}"))?;
        probe::layer(cuda, layer, "moe_router", timer);
        let timer = BlockTimer::start(request.forward_perf.probe);
        let w_slice = 576 * (dim / 2);
        let s_slice = 576 * (dim / 32);
        let d_slice = dim * (inter / 2 / world);
        let d_scale_slice = dim * (inter / 32 / world);
        for (slot, id) in slots.iter().enumerate() {
            let (w1, s1) = routed_slice_base(
                &bind.moe,
                &bind.plan_index,
                layer,
                *id as usize,
                crate::resident_weights::Matrix::Gate,
                "w1",
                rank,
                [576, dim / 2],
            )?;
            let (w3, s3) = routed_slice_base(
                &bind.moe,
                &bind.plan_index,
                layer,
                *id as usize,
                crate::resident_weights::Matrix::Up,
                "w3",
                rank,
                [576, dim / 2],
            )?;
            let (w2, s2) = routed_slice_base(
                &bind.moe,
                &bind.plan_index,
                layer,
                *id as usize,
                crate::resident_weights::Matrix::Down,
                "w2",
                rank,
                [dim, inter / 2 / world],
            )?;
            cuda.copy_dtod(&gate_stage.offset(slot * w_slice), &w1, w_slice)
                .map_err(|error| format!("MoE: w1 stage copy: {error}"))?;
            cuda.copy_dtod(&gate_scale_stage.offset(slot * s_slice), &s1, s_slice)
                .map_err(|error| format!("MoE: s1 stage copy: {error}"))?;
            cuda.copy_dtod(&up_stage.offset(slot * w_slice), &w3, w_slice)
                .map_err(|error| format!("MoE: w3 stage copy: {error}"))?;
            cuda.copy_dtod(&up_scale_stage.offset(slot * s_slice), &s3, s_slice)
                .map_err(|error| format!("MoE: s3 stage copy: {error}"))?;
            cuda.copy_dtod(&w2_stage.offset(slot * d_slice), &w2, d_slice)
                .map_err(|error| format!("MoE: w2 stage copy: {error}"))?;
            cuda.copy_dtod(
                &w2_scale_stage.offset(slot * d_scale_slice),
                &s2,
                d_scale_slice,
            )
            .map_err(|error| format!("MoE: s2 stage copy: {error}"))?;
        }
        // WS-C: this stretch is launch-only in the lean path and is the decode loop's one CUDA
        // graph candidate; the probe gate keeps the direct path where probe timers synchronize.
        let mut run_moe_launches =
            |cuda: &sparkflash_kernels::cuda_driver::Cuda| -> Result<(), String> {
                let mut params = routed_gate_up_launch::GateUpLaunchParams {
                    act_fp8,
                    act_scales,
                    w_fp4: gate_stage,
                    w_scales: gate_scale_stage,
                    expert_ids: slot_ids,
                    out: gate_out,
                    tokens: tokens as u32,
                    trace_ring: 0,
                    layer_id: 0,
                    step_seq: rank as u64,
                };
                routed_gate_up_launch::launch(cuda, gate_up_kernel, &params)?;
                params.w_fp4 = up_stage;
                params.w_scales = up_scale_stage;
                params.out = up_out;
                params.step_seq = rank as u64 + 1;
                routed_gate_up_launch::launch(cuda, gate_up_kernel, &params)?;
                probe::layer(cuda, layer, "moe_gate_up", timer);
                let timer = BlockTimer::start(request.forward_perf.probe);
                let weighted_words: [u64; 11] = [
                    gate_out.0,
                    up_out.0,
                    weights.0,
                    codes.0,
                    swiglu_scales.0,
                    routed_rows as u64,
                    640,
                    u64::from(10.0f32.to_bits()),
                    0,
                    0,
                    rank as u64,
                ];
                let weighted_grid = ((routed_rows * 640 / 32).div_ceil(8) as u32, 1, 1);
                cuda.launch(weighted_kernel, weighted_grid, (256, 1, 1), &weighted_words)?;
                probe::layer(cuda, layer, "moe_weighted", timer);
                Ok(())
            };
        if request.forward_perf.probe {
            run_moe_launches(cuda)?;
        } else {
            crate::tp4_graph::segment(cuda, "moe-compute", &mut run_moe_launches)?;
        }
        let timer = BlockTimer::start(request.forward_perf.probe);
        let combine_params = mxfp4_moe_stored_launch::Mxfp4MoeStoredParams::new(
            codes,
            swiglu_scales,
            w2_stage,
            w2_scale_stage,
            slot_ids,
            acc,
            tokens as u32,
            0,
            0,
            rank as u64,
        );
        mxfp4_moe_stored_launch::launch(cuda, combine_kernel, &combine_params)?;
        let mut slot_acc_raw = vec![0u8; routed_rows * dim * 4];
        cuda.copy_dtoh(&mut slot_acc_raw, &acc)
            .map_err(|error| format!("MoE: slot acc dtoh: {error}"))?;
        let slot_acc = f32_from_le_bytes(&slot_acc_raw)?;
        if layer_evidence_readbacks() && slot_acc.iter().any(|value| !value.is_finite()) {
            return Err("MoE: the stored combine wrote a non-finite accumulator".to_string());
        }
        probe::layer(cuda, layer, "moe_stored_combine", timer);
        let timer = BlockTimer::start(request.forward_perf.probe);
        let (slot_acc, rank_routed, payload, delta, reduce_sequence, native_full_acc) =
            if request.native_rounding.combine {
                // M2 native arm: every rank sends its f32 slot accumulator as raw u16 words, so each
                // rank can run the released combine epilogue over the f32 rank sum (model.py:891-904
                // keeps y in f32 through the rank reduction; the shared BF16 add and the one MoE
                // BF16 round happen in the epilogue).
                let slot_bytes = f32_le_bytes(&slot_acc);
                let words: Vec<u16> = slot_bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| u16::from_le_bytes(*pair))
                    .collect();
                let mut gathered = vec![0u16; request.world * words.len()];
                let sequence = NATIVE_COMBINE_GATHER_SEQ.wrapping_sub(layer as u64);
                transport
                    .all_gather_bf16(&words, &mut gathered, sequence)
                    .map_err(|error| format!("MoE: native combine all-gather: {error}"))?;
                let gathered_bytes: Vec<u8> = gathered
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                let gathered_values = f32_from_le_bytes(&gathered_bytes)?;
                let stride = words.len() / 2;
                let mut rank_slices: Vec<&[f32]> = Vec::with_capacity(request.world);
                for rank_index in 0..request.world {
                    let start = rank_index * stride;
                    rank_slices.push(&gathered_values[start..start + stride]);
                }
                let full = mxfp4_ref::moe_combine_rank_sum(&rank_slices, tokens, dim);
                let routed = mxfp4_ref::moe_combine_routed_f32(&full, &router_ids, tokens, dim);
                let delta = bf16_le_bytes(&routed);
                (full.clone(), routed, Vec::new(), delta, 0, Some(full))
            } else {
                let rank_routed =
                    mxfp4_ref::moe_combine_routed_f32(&slot_acc, &router_ids, tokens, dim);
                let payload = bf16_le_bytes(&rank_routed);
                let mut words: Vec<u16> = payload
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| u16::from_le_bytes(*pair))
                    .collect();
                let layout =
                    CollectiveLayout::new(request.world, dim, FabricSchedule::OneShotAllToAll)
                        .map_err(|error| format!("MoE: collective layout: {error}"))?;
                let reduce_sequence = MOE_REDUCE_SEQ;
                reduce
                    .all_reduce_bf16(&mut words, reduce_sequence, &layout)
                    .map_err(|error| format!("MoE: all-reduce: {error}"))?;
                let delta: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                (slot_acc, rank_routed, payload, delta, reduce_sequence, None)
            };
        probe::layer(cuda, layer, "moe_reduce", timer);
        // The four stage readbacks below are the evidence record's stage digests and nothing else:
        // the combine consumes the device buffers, so a lean run has no numeric consumer for them.
        let (gate_raw, up_raw, codes_raw, scales_raw) = if layer_evidence_readbacks() {
            let mut gate_raw = vec![0u8; routed_rows * 640 * 2];
            let mut up_raw = vec![0u8; routed_rows * 640 * 2];
            let mut codes_raw = vec![0u8; routed_rows * 640];
            let mut scales_raw = vec![0u8; routed_rows * 20];
            cuda.copy_dtoh(&mut gate_raw, &gate_out)
                .map_err(|error| format!("MoE: gate dtoh: {error}"))?;
            cuda.copy_dtoh(&mut up_raw, &up_out)
                .map_err(|error| format!("MoE: up dtoh: {error}"))?;
            cuda.copy_dtoh(&mut codes_raw, &codes)
                .map_err(|error| format!("MoE: codes dtoh: {error}"))?;
            cuda.copy_dtoh(&mut scales_raw, &swiglu_scales)
                .map_err(|error| format!("MoE: scales dtoh: {error}"))?;
            (gate_raw, up_raw, codes_raw, scales_raw)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        Ok(Layer0MoeRouted {
            rank,
            pins,
            router_ids,
            router_weights,
            router_scores,
            slot_ids: slots,
            global_ids,
            gate_raw,
            up_raw,
            swiglu_codes: codes_raw,
            swiglu_scales: scales_raw,
            slot_acc,
            rank_routed,
            payload,
            delta,
            reduce_sequence,
            native_combine: request.native_rounding.combine,
            native_full_acc,
        })
    })();
    cuda.release_pooled(&[
        &x_bf16,
        &ids,
        &weights,
        &scores,
        &act_fp8,
        &act_scales,
        &gate_stage,
        &gate_scale_stage,
        &up_stage,
        &up_scale_stage,
        &gate_out,
        &up_out,
        &codes,
        &swiglu_scales,
        &w2_stage,
        &w2_scale_stage,
        &slot_ids,
        &acc,
    ]);
    cuda.free(router_module, &[]);
    cuda.free(topk_module, &[]);
    cuda.free(gate_up_module, &[]);
    cuda.free(weighted_module, &[]);
    cuda.free(combine_module, &[]);
    staged
}
/// Maps the router's global top-k ids to the slot convention every rank shares: the sorted unique
/// global selection, with each row's id replaced by its slot index.
///
/// # Errors
/// Refuses an id at or above the expert count and a selection whose slots do not round-trip.
pub fn moe_slots(global_ids: &[u32], experts: usize) -> Result<(Vec<u32>, Vec<u32>), String> {
    if experts == 0 || experts > u32::MAX as usize {
        return Err(format!(
            "moe slots: expert count {experts} is not addressable"
        ));
    }
    let mut slots: Vec<u32> = global_ids.to_vec();
    slots.sort_unstable();
    slots.dedup();
    if let Some(bad) = slots.iter().find(|id| **id >= experts as u32) {
        return Err(format!(
            "moe slots: the router selected global id {bad} but the table has {experts}"
        ));
    }
    let mut ids = Vec::with_capacity(global_ids.len());
    for id in global_ids {
        let slot = slots
            .iter()
            .position(|candidate| candidate == id)
            .ok_or_else(|| format!("moe slots: global id {id} is not in its own selection"))?;
        ids.push(slot as u32);
    }
    Ok((ids, slots))
}

/// The layer-0 FFN epilogue: the shared expert added on top of the routed delta, then the ffn hc_post.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct Layer0FfnEpilogue {
    /// The six per-rank shared-expert shard reads from the live arena.
    pub shared_reads: usize,
    /// The six transport all-gathers that replicated the shards for the seam.
    pub shared_gathers: usize,
    /// The bytes of the full released shared-expert tensors handed to the engine seam.
    pub shared_gathered_bytes: usize,
    /// The arithmetic that produced the shared row: the released FP8 32x32 Expert or the f32 seam.
    pub shared_arith: &'static str,
    /// The kernels the shared row came from, or empty when the host seam produced it.
    pub shared_kernel: &'static str,
    /// The accumulation mode those kernels ran (fused, or rounded multiply then add, and whether
    /// the output was stored bf16 or f32).
    pub shared_mode: &'static str,
    /// The symbol of every launch the shared-expert block ran, in launch order. The f32-reference
    /// arm lists the three GEMV launches; the coded native arm also lists the SwiGLU epilogue.
    pub launch_symbols: Vec<&'static str>,
    /// sha256 of the shared-expert row itself.
    pub shared_sha256: String,
    /// The shared row's L2 norm.
    pub shared_l2: f32,
    /// sha256 of the f32-reference shared row when the native arm also recorded it (layer 0 only).
    pub shared_f32_sha256: Option<String>,
    /// Relative L2 of (native shared row - f32 shared row) when the layer-0 evidence ran.
    pub shared_native_vs_f32_rel_l2: Option<f32>,
    /// sha256 of the routed delta plus the shared expert (the ffn hc_post input).
    pub ffn_out_sha256: String,
    /// The ffn out L2 norm.
    pub ffn_out_l2: f32,
    /// The ffn out row itself (routed delta plus shared expert), kept for the boundary trace.
    pub ffn_out: Vec<f32>,
    /// sha256 of the updated hc residual stream after the ffn hc_post.
    pub hc_post_sha256: String,
    /// The updated stream L2 norm.
    pub hc_post_l2: f32,
    /// The updated stream, kept for the next layer.
    pub hc_post: Vec<f32>,
}

#[cfg(feature = "cuda")]
impl Layer0FfnEpilogue {
    /// The evidence block for the refused run.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "stage": "layer0-ffn-epilogue-shared-and-hc-post",
            "shared_expert": {
                "added": true,
                "arena_reads": self.shared_reads,
                "gather_calls": self.shared_gathers,
                "gathered_bytes": self.shared_gathered_bytes,
                "kernel_launched": !self.launch_symbols.is_empty(),
                "kernel": self.shared_kernel,
                "arithmetic_mode": self.shared_mode,
                "arith": self.shared_arith,
                "launch_symbols": self.launch_symbols,
                "launch_count": self.launch_symbols.len(),
                "row_sha256": self.shared_sha256,
                "row_l2": self.shared_l2,
                "f32_row_sha256": self.shared_f32_sha256,
                "native_vs_f32_rel_l2": self.shared_native_vs_f32_rel_l2,
                "delta_scope": self.shared_f32_sha256.as_ref().map(|_| "layer0"),
                "rule": "each rank's TP-column/row shards of the released shared expert are all-gathered per layer, untiled from the arena B12x micro-tile order back to the checkpoint stored order (scales are never tiled), and joined into the replicated [2304,5120]/[5120,2304] F8-E4M3+E8-E8M0 views, then the engine shared-expert seam runs on the full view (H5 replicated, byte-identical to the twin); the native arm runs the released FP8 32x32 Expert arithmetic instead of the f32 reference",
            },
            "ffn_out": {
                "sha256": self.ffn_out_sha256,
                "bytes": self.ffn_out_len(),
                "l2": self.ffn_out_l2,
                "note": "the routed delta after the shared expert add; this rank's ffn input to hc_post",
            },
            "hc_post": {
                "sha256": self.hc_post_sha256,
                "bytes": self.hc_post.len() * 4,
                "l2": self.hc_post_l2,
                "rule": "engine mhc::hc_post with the FFN post mixes and comb matrix",
            },
            "not_reference": "layer 0 only, one row at the loop requested position, not the 130-token prefill; no token can be claimed",
        })
    }

    fn ffn_out_len(&self) -> usize {
        HIDDEN * 4
    }
}

/// Reassembles one stored tensor from its TP-column shards: the plan gives each rank one contiguous
/// block of the stored axis-0 rows, so the full row-major bytes are the shards in rank order.
#[must_use]
pub fn join_column_shards(shards: &[Vec<u8>]) -> Vec<u8> {
    let mut full = Vec::with_capacity(shards.iter().map(Vec::len).sum());
    for shard in shards {
        full.extend_from_slice(shard);
    }
    full
}

/// Reassembles one stored tensor from its TP-row shards: the plan gives each rank contiguous columns
/// of every stored row, so row j of the full row-major tensor is the rank-ordered concatenation of
/// each shard's row j.
///
/// # Errors
/// Refuses a zero shard count, zero rows or zero shard width, and a shard whose byte count is not
/// rows * shard_cols.
pub fn join_row_shards(
    shards: &[Vec<u8>],
    rows: usize,
    shard_cols: usize,
) -> Result<Vec<u8>, String> {
    if shards.is_empty() || rows == 0 || shard_cols == 0 {
        return Err("row-shard join: a zero geometry has no tensor".to_string());
    }
    let per_shard = rows
        .checked_mul(shard_cols)
        .ok_or_else(|| "row-shard join: a shard size overflows".to_string())?;
    for (rank, shard) in shards.iter().enumerate() {
        if shard.len() != per_shard {
            return Err(format!(
                "row-shard join: rank {rank} holds {} bytes, expected {rows} rows of {shard_cols}",
                shard.len()
            ));
        }
    }
    let mut full = Vec::with_capacity(per_shard * shards.len());
    for row in 0..rows {
        for shard in shards {
            full.extend_from_slice(&shard[row * shard_cols..(row + 1) * shard_cols]);
        }
    }
    Ok(full)
}

/// A WeightSource over host bytes read from the rank arena: what the released host seams that take a
/// source (the shared expert) need, without a checkpoint reader.
#[cfg(feature = "cuda")]
struct ArenaHostSource {
    tensors: std::collections::BTreeMap<String, (sparkflash_engine::tensor::TensorSpec, Vec<u8>)>,
}

#[cfg(feature = "cuda")]
impl sparkflash_engine::weights::WeightSource for ArenaHostSource {
    fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    fn spec(
        &self,
        name: &str,
    ) -> Result<sparkflash_engine::tensor::TensorSpec, sparkflash_engine::weights::WeightError>
    {
        self.tensors
            .get(name)
            .map(|(spec, _)| spec.clone())
            .ok_or_else(|| sparkflash_engine::weights::WeightError::Missing {
                name: name.to_string(),
            })
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, sparkflash_engine::weights::WeightError> {
        self.tensors
            .get(name)
            .map(|(_, bytes)| bytes.clone())
            .ok_or_else(|| sparkflash_engine::weights::WeightError::Missing {
                name: name.to_string(),
            })
    }

    fn read_view(
        &self,
        name: &str,
    ) -> Result<Option<&[u8]>, sparkflash_engine::weights::WeightError> {
        match self.tensors.get(name) {
            Some((_, bytes)) => Ok(Some(bytes)),
            None => Err(sparkflash_engine::weights::WeightError::Missing {
                name: name.to_string(),
            }),
        }
    }
}

/// A checkpoint source with the Engram wkv matrices held resident, so the per-position read of a
/// constant weight becomes a borrow.
///
/// [sparkflash_engine::forward::engram::engram_forward] reads `layers.{l}.engram.wkv.weight` and its
/// scale grid on every call, and [sparkflash_engine::safetensors::SafeTensorsSource] has no
/// `read_view`: through it the loop allocates and copies 157 MB per layer-position (~43 GB over a
/// released-length run) for two matrices that never change. The loop loads them once and this overlay
/// serves those bytes, which the engine's read path prefers; every other name, and the paged Engram
/// tables, delegate to the wrapped source unchanged.
#[cfg(any(feature = "cuda", test))]
struct EngramResidentSource<'a> {
    inner: &'a dyn sparkflash_engine::weights::WeightSource,
    resident: std::collections::BTreeMap<String, Vec<u8>>,
}

#[cfg(any(feature = "cuda", test))]
impl<'a> EngramResidentSource<'a> {
    /// Reads each name once. Nothing mutates the bytes afterwards, which is what lets
    /// [sparkflash_engine::weights::WeightSource::read_view] hand out a borrow.
    fn preload(
        inner: &'a dyn sparkflash_engine::weights::WeightSource,
        names: &[String],
    ) -> Result<Self, String> {
        let mut resident = std::collections::BTreeMap::new();
        for name in names {
            let bytes = inner
                .read(name)
                .map_err(|error| format!("forward loop: Engram preload {name}: {error}"))?;
            resident.insert(name.clone(), bytes);
        }
        Ok(Self { inner, resident })
    }
}

#[cfg(feature = "cuda")]
impl EngramResidentSource<'_> {
    /// The bytes [EngramResidentSource::preload] holds for a name, for a caller that needs them
    /// once: the device wkv arm uploads the 157 MB matrix at run scope instead of reading it on
    /// every layer-position. The borrow keeps the preload's single read.
    fn resident_bytes(&self, name: &str) -> Option<&[u8]> {
        self.resident.get(name).map(Vec::as_slice)
    }
}

#[cfg(any(feature = "cuda", test))]
impl sparkflash_engine::weights::WeightSource for EngramResidentSource<'_> {
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    fn spec(
        &self,
        name: &str,
    ) -> Result<sparkflash_engine::tensor::TensorSpec, sparkflash_engine::weights::WeightError>
    {
        self.inner.spec(name)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, sparkflash_engine::weights::WeightError> {
        match self.resident.get(name) {
            Some(bytes) => Ok(bytes.clone()),
            None => self.inner.read(name),
        }
    }

    fn read_view(
        &self,
        name: &str,
    ) -> Result<Option<&[u8]>, sparkflash_engine::weights::WeightError> {
        match self.resident.get(name) {
            Some(bytes) => Ok(Some(bytes.as_slice())),
            None => self.inner.read_view(name),
        }
    }

    fn read_ranges(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, sparkflash_engine::weights::WeightError>> {
        self.inner.read_ranges(name, ranges)
    }

    fn read_ranges_paged(
        &self,
        name: &str,
        ranges: &[(u64, usize)],
    ) -> Option<Result<Vec<u8>, sparkflash_engine::weights::WeightError>> {
        self.inner.read_ranges_paged(name, ranges)
    }
}

/// The tensors [EngramResidentSource] holds: each Engram layer's wkv codes and scale grid, the two
/// names [sparkflash_engine::forward::engram::engram_forward] reads on every call.
#[cfg(any(feature = "cuda", test))]
fn engram_weight_names(config: &sparkflash_engine::config::ModelConfig) -> Vec<String> {
    config
        .engram_layer_ids
        .iter()
        .flat_map(|layer| {
            [
                format!("layers.{layer}.engram.wkv.weight"),
                format!("layers.{layer}.engram.wkv.scale"),
            ]
        })
        .collect()
}

/// Reads the Engram table rows a run will look up before the loop's first position.
///
/// The rows are a pure function of the token ids: [ngram_hashes] over the same history the loop
/// hands each call produces the same ids, so the union over the run's positions is exactly the set
/// of rows the loop will page in. Each row is [engram_head_dim] weight bytes and
/// [engram_head_dim] / 32 scale bytes, so a 138-position run warms a few thousand of them - about
/// a megabyte of weight per layer, not a preload of the 98 GB tables.
///
/// Why it is needed: !158 replaced the checkpoint loader with an O_DIRECT parallel pread
/// (arena.rs: "CPU file reads go straight into the final GPU-visible arena"), which bypasses the
/// page cache. The old buffered loader left the shards in the cache as a side effect, so the paged
/// Engram reads were hits; on main the first run's lookups are cold disk reads, which is the
/// difference between the 3.5-3.8 ms rows and the 12.4 ms one. Reading the same ranges buffered
/// here leaves them cached, so the loop reads the identical bytes from the identical file pages.
///
/// The q/k gate weights are constant small tensors the seam reads on every call; they are warmed
/// too so those reads are hits from the first call on. A source that cannot serve ranges is left
/// alone: for it, warming would mean reading a whole 98 GB table. Nothing is decoded, checked
/// against a limit or written anywhere; the bytes are discarded and no value changes.
///
/// # Errors
///
/// The source's error for a range it cannot read, named with the tensor.
#[cfg(any(feature = "cuda", test))]
fn warm_engram_rows(
    config: &sparkflash_engine::config::ModelConfig,
    tokens: &[u32],
    total_positions: usize,
    source: &dyn sparkflash_engine::weights::WeightSource,
) -> Result<(), String> {
    use sparkflash_engine::forward::engram::{HASH_SENTINEL, hash_columns, ngram_hashes};

    let columns = hash_columns(config);
    let width = config.engram_head_dim;
    if config.engram_layer_ids.is_empty() || columns == 0 || width == 0 {
        return Ok(());
    }
    let scale_width = width.div_ceil(32);
    let mut rows: Vec<std::collections::BTreeSet<usize>> =
        vec![std::collections::BTreeSet::new(); config.engram_layer_ids.len()];
    let mut hashes = Vec::new();
    for position in 0..total_positions {
        let history = tokens.get(..=position).ok_or_else(|| {
            format!(
                "forward loop: the Engram warm-up needs the tokens up to position {position} but --token-ids has {}",
                tokens.len()
            )
        })?;
        ngram_hashes(config, history, position, &mut hashes).map_err(|error| {
            format!("forward loop: Engram hashes at position {position}: {error}")
        })?;
        for (index, &layer) in config.engram_layer_ids.iter().enumerate() {
            let Some(table_rows) = config.engram_rows(layer) else {
                continue;
            };
            let row = &hashes[index * columns..(index + 1) * columns];
            if row.contains(&HASH_SENTINEL) {
                continue;
            }
            // An id outside the table is the seam's named refusal, raised by the loop with the row
            // count; the warm-up skips it rather than reading outside the tensor.
            if row.iter().any(|&id| id as usize >= table_rows) {
                continue;
            }
            rows[index].extend(row.iter().map(|&id| id as usize));
        }
    }
    for (index, &layer) in config.engram_layer_ids.iter().enumerate() {
        let prefix = format!("layers.{layer}");
        let ids = &rows[index];
        if !ids.is_empty() {
            let weight_name = format!("{prefix}.engram.embed.weight");
            let weight_ranges: Vec<(u64, usize)> = ids
                .iter()
                .map(|&id| (id as u64 * width as u64, width))
                .collect();
            if let Some(result) = source.read_ranges_paged(&weight_name, &weight_ranges) {
                result.map_err(|error| {
                    format!("forward loop: Engram warm-up {weight_name}: {error}")
                })?;
            }
            let scale_name = format!("{prefix}.engram.embed.scale");
            let scale_ranges: Vec<(u64, usize)> = ids
                .iter()
                .map(|&id| (id as u64 * scale_width as u64, scale_width))
                .collect();
            if let Some(result) = source.read_ranges_paged(&scale_name, &scale_ranges) {
                result.map_err(|error| {
                    format!("forward loop: Engram warm-up {scale_name}: {error}")
                })?;
            }
        }
        for suffix in ["q_weight", "k_weight"] {
            let name = format!("{prefix}.engram.{suffix}");
            if source.spec(&name).is_ok() {
                source
                    .read(&name)
                    .map_err(|error| format!("forward loop: Engram warm-up {name}: {error}"))?;
            }
        }
    }
    Ok(())
}

/// Whether the run-start warm pass runs (the default) or is skipped for an A/B row.
///
/// The pass reads only bytes the loop would read anyway, so the switch changes wall time and
/// nothing else: `SPARKFLASH_SEAM_WARM=0` (also `off`, `false`, `no`) skips it, and one binary can
/// run the warm/no-warm rows back to back in the same machine state.
#[cfg(feature = "cuda")]
fn seam_warm_enabled() -> bool {
    // Default ON, with a kill switch. The O_DIRECT parallel loader (!158) never fills the page
    // cache the host seams used to hit, so on a pages-cold node the position loop re-reads the
    // compressor/indexer tensors and the paged Engram rows from disk: measured on rank 0 at
    // ~1.29 GB per 1.5 s of block reads with the host at 0.27 cores. The pass reads those tensors
    // and rows into this process's own memory before position 0, so the loop cannot fault them.
    //
    // Measured, same staged tag (wsd17, sha 96cea90d), same frozen argv, checkpoint page cache
    // dropped on all four nodes before each row, only this switch differing:
    //   pass on : 229.33 ms/step (224-232) and 228.44 (226-231), identity gate PASS
    //   pass off: 255.11 ms/step (250-258)
    // i.e. -25.8 ms/step and below the published 249.25 even though the tree is +15 ms/step slower
    // than the published binary on its clean steps. The earlier default-on row that read 489 ms
    // (wsd16-lean1) was a page-cache-contaminated row, not the pass: fifteen consecutive rows in
    // the clean window never entered that mode.
    //
    // SPARKFLASH_SEAM_WARM=0/off/false/no disables the pass for an investigation row.
    std::env::var("SPARKFLASH_SEAM_WARM").map_or(true, |value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    })
}

/// The device arm of the Engram wkv seam: the already-pinned `engram_wkv_sm121` kernel over the
/// resident 157 MB W8A8 matrix, with the host fetch, activation quantise, gate and write unchanged.
///
/// Why the arithmetic is the same: the module docs of
/// [sparkflash_engine::forward::engram] state the host seam and the three Engram kernels implement
/// the native dtype rule op for op, and the host matmul the seam replaces
/// ([sparkflash_engine::linalg::matmul_f8_w8a8]) is pinned bit-for-bit against
/// `wkv_native`, which is the kernel's five-step shuffle-tree reduction written out
/// (kernels/engram_wkv_sm121.cu). This arm therefore moves the [out_width][inner] block-dot
/// arithmetic only: it hands the kernel exactly the activation codes/scales the host path would
/// have multiplied, decodes the kernel's bf16 output back to f32 (bf16-exact) and lets the engine
/// run its existing f32 gate and write over that vector. The device's only declared deviation in
/// this head is the gate's sigmoid, and the gate stays on the host.
///
/// The weights are uploaded once at run scope: each Engram layer's matrix is a constant, and the
/// released loop would otherwise read it on every layer-position (the block's 18.8 ms per call is
/// that matmul, not I/O: the resident overlay already keeps the bytes).
#[cfg(feature = "cuda")]
struct EngramWkvDevice<'a> {
    cuda: &'a sparkflash_kernels::cuda_driver::Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    /// One (codes, scales) device pair per Engram layer that owns a table.
    weights: std::collections::BTreeMap<usize, (Deviceptr, Deviceptr)>,
    /// [inner] E4M3 activation codes of the current call, one small upload.
    act_codes: Deviceptr,
    /// [inner / 32] E8M0 activation block scales of the current call.
    act_scales: Deviceptr,
    /// [out_width] bf16 kernel output, copied back for the host gate.
    out: Deviceptr,
    /// Bytes of the bf16 output buffer (two per value).
    out_bytes: usize,
    /// The reduction width (hash columns x Engram head width; 6,144 released).
    inner: usize,
    /// The kernel's output width (dim x (hc + 1); 25,600 released).
    out_width: usize,
    /// Every device allocation this arm owns, freed when the run scope drops it.
    allocations: Vec<Deviceptr>,
}

#[cfg(feature = "cuda")]
impl<'a> EngramWkvDevice<'a> {
    /// Verifies the pinned cubin, uploads each Engram layer's wkv matrix and allocates the
    /// per-call staging buffers.
    ///
    /// # Errors
    /// A geometry that is not whole 32-value blocks, a missing resident overlay tensor, a staged
    /// cubin that disagrees with the baseline pin, an allocation or an upload failure.
    fn build(
        cuda: &'a sparkflash_kernels::cuda_driver::Cuda,
        config: &sparkflash_engine::config::ModelConfig,
        kernel_dir: &std::path::Path,
        resident: &EngramResidentSource<'_>,
    ) -> Result<Self, String> {
        let inner =
            sparkflash_engine::forward::engram::hash_columns(config) * config.engram_head_dim;
        let out_width = config.dim * (config.hc_mult + 1);
        if inner == 0 || out_width == 0 {
            return Err(format!(
                "Engram device wkv: the configuration gives out_width {out_width} inner {inner}"
            ));
        }
        if !inner.is_multiple_of(32) || !out_width.is_multiple_of(32) {
            return Err(format!(
                "Engram device wkv: {out_width}x{inner} is not whole 32-value blocks, so it is not the released checkpoint layout"
            ));
        }
        let (cubin, _sha256) = load_pinned_addition(
            kernel_dir,
            sparkflash_kernels::engram_wkv_launch::KERNEL,
            "Engram device wkv",
        )?;
        let (_module, kernel) = sparkflash_kernels::engram_wkv_launch::load(cuda, &cubin)?;
        let mut allocated = 0usize;
        let mut allocations = Vec::new();
        let mut weights = std::collections::BTreeMap::new();
        let want_codes = out_width * inner;
        let want_scales = out_width.div_ceil(32) * inner.div_ceil(32);
        for &layer in &config.engram_layer_ids {
            let codes_name = format!("layers.{layer}.engram.wkv.weight");
            let scales_name = format!("layers.{layer}.engram.wkv.scale");
            let codes = resident.resident_bytes(&codes_name).ok_or_else(|| {
                format!("Engram device wkv: the resident overlay holds no {codes_name}")
            })?;
            let scales = resident.resident_bytes(&scales_name).ok_or_else(|| {
                format!("Engram device wkv: the resident overlay holds no {scales_name}")
            })?;
            if codes.len() != want_codes || scales.len() != want_scales {
                return Err(format!(
                    "Engram device wkv: {codes_name} is {} bytes and {scales_name} {} bytes; the {out_width}x{inner} checkpoint layout is {want_codes} and {want_scales}",
                    codes.len(),
                    scales.len()
                ));
            }
            let codes_dev = cuda.alloc(codes.len(), &mut allocated).map_err(|error| {
                format!("Engram device wkv: {codes_name} device buffer: {error}")
            })?;
            allocations.push(codes_dev);
            cuda.copy_htod(&codes_dev, codes)
                .map_err(|error| format!("Engram device wkv: {codes_name} upload: {error}"))?;
            let scales_dev = cuda.alloc(scales.len(), &mut allocated).map_err(|error| {
                format!("Engram device wkv: {scales_name} device buffer: {error}")
            })?;
            allocations.push(scales_dev);
            cuda.copy_htod(&scales_dev, scales)
                .map_err(|error| format!("Engram device wkv: {scales_name} upload: {error}"))?;
            weights.insert(layer, (codes_dev, scales_dev));
        }
        let act_codes = cuda
            .alloc(inner, &mut allocated)
            .map_err(|error| format!("Engram device wkv: activation code buffer: {error}"))?;
        allocations.push(act_codes);
        let act_scales = cuda
            .alloc(inner.div_ceil(32), &mut allocated)
            .map_err(|error| format!("Engram device wkv: activation scale buffer: {error}"))?;
        allocations.push(act_scales);
        let out_bytes = out_width * 2;
        let out = cuda
            .alloc(out_bytes, &mut allocated)
            .map_err(|error| format!("Engram device wkv: output buffer: {error}"))?;
        allocations.push(out);
        Ok(Self {
            cuda,
            kernel,
            weights,
            act_codes,
            act_scales,
            out,
            out_bytes,
            inner,
            out_width,
            allocations,
        })
    }

    /// Runs the wkv projection for one layer-position through the pinned kernel.
    ///
    /// `out` receives the kernel's bf16 output decoded to f32, i.e. exactly the values the host
    /// `matmul_f8_w8a8` arm hands the gate.
    ///
    /// # Errors
    /// A geometry disagreement, a layer without a resident matrix, or a copy/launch failure.
    fn project(
        &self,
        layer: usize,
        act_codes: &[u8],
        act_scales: &[u8],
        out_width: usize,
        inner: usize,
        out: &mut [f32],
    ) -> Result<(), String> {
        if out_width != self.out_width || inner != self.inner {
            return Err(format!(
                "Engram device wkv: the seam asked for {out_width}x{inner}, the resident kernel geometry is {}x{}",
                self.out_width, self.inner
            ));
        }
        if act_codes.len() != inner
            || act_scales.len() != inner.div_ceil(32)
            || out.len() != out_width
        {
            return Err(format!(
                "Engram device wkv: {} codes, {} scales and {} outputs do not describe one {out_width}x{inner} row",
                act_codes.len(),
                act_scales.len(),
                out.len()
            ));
        }
        let &(weight, weight_scales) = self.weights.get(&layer).ok_or_else(|| {
            format!("Engram device wkv: no resident wkv matrix for layer {layer}")
        })?;
        self.cuda
            .copy_htod(&self.act_codes, act_codes)
            .map_err(|error| format!("Engram device wkv: activation code htod: {error}"))?;
        self.cuda
            .copy_htod(&self.act_scales, act_scales)
            .map_err(|error| format!("Engram device wkv: activation scale htod: {error}"))?;
        let params = sparkflash_kernels::engram_wkv_launch::EngramWkvLaunchParams {
            act_codes: self.act_codes,
            act_scales: self.act_scales,
            weight,
            weight_scales,
            out: self.out,
            out_width: out_width as u32,
            inner: inner as u32,
            trace_ring: 0,
            layer_id: layer as u32,
            step_seq: 0,
        };
        sparkflash_kernels::engram_wkv_launch::launch(self.cuda, self.kernel, &params)?;
        let mut raw = vec![0u8; self.out_bytes];
        self.cuda
            .copy_dtoh(&mut raw, &self.out)
            .map_err(|error| format!("Engram device wkv: output dtoh: {error}"))?;
        let values = bf16_from_le_bytes(&raw)?;
        if values.len() != out.len() {
            return Err(format!(
                "Engram device wkv: the kernel returned {} values, the seam expects {}",
                values.len(),
                out.len()
            ));
        }
        out.copy_from_slice(&values);
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for EngramWkvDevice<'_> {
    fn drop(&mut self) {
        let allocations: Vec<&Deviceptr> = self.allocations.iter().collect();
        self.cuda.free_buffers(&allocations);
    }
}

/// The ref_seam slice's device arm: the indexer's own query projection on the pinned decode GEMV.
///
/// The reference seam runs `layers.N.attn.indexer.wq_b` on the host, in `index_source_wq_b`: the
/// bf16-narrowed q_norm latent is quantised E4M3 over 32-value blocks with ue8m0 scales, multiplied
/// against the fp8 weight accumulating in f32 in ascending order, and the output is bf16-rounded.
/// The pinned `b12x_mxfp8_gemv_m1_nf_f32a_sm121` body is that rule: it stages an f32 activation row,
/// decodes the E4M3 weight with its ue8m0 grid in-kernel, does a rounded multiply and a rounded add
/// per element (no FMA, no split-K, one accumulator per output column) and stores bf16.
///
/// The weight bytes are the checkpoint shard's row-major E4M3 codes permuted into the B12x layout the
/// kernel reads - a pure permutation, done once per layer on first use - and uploaded once, so the
/// kernel reads exactly the bytes the host rule reads. The activation is the engine's own
/// [sparkflash_engine::forward::attention::indexer_activation_native] row staged as f32: the
/// f32-activation mode does not quantise it again.
///
/// The arm is a fallback-safe switch: a layer whose source does not serve the released
/// F8-E4M3 + E8-E8M0 pair (a fixture, or a converted checkpoint that stores it some other way) is
/// declined and the engine's host native rule runs for that layer exactly as it does without the
/// arm. Nothing else in the seam moves: the compressor, the index keys, the per-query weights, the
/// candidate select and the fp4 cache stay host code.
#[cfg(feature = "cuda")]
struct SeamDevice<'a> {
    cuda: &'a sparkflash_kernels::cuda_driver::Cuda,
    kernel: sparkflash_kernels::cuda_driver::Kernel,
    /// One (B12x weight, ue8m0 grid) pair per index-source layer, uploaded on first use.
    weights: std::collections::BTreeMap<usize, (Deviceptr, Deviceptr)>,
    /// Layers whose source did not serve the released F8 pair; the host rule runs for them.
    declined: std::collections::BTreeSet<usize>,
    /// The f32 activation row of the current call.
    act: Deviceptr,
    /// The bf16 output row of the current call.
    out: Deviceptr,
    /// The kernel's output width (`index_n_heads * index_head_dim`).
    cols: usize,
    /// The reduction width (`q_lora_rank`).
    inner: usize,
    /// Every device allocation this arm owns, freed when the run scope drops it.
    allocations: Vec<Deviceptr>,
    /// The host arm, for the projections this arm does not serve.
    host: sparkflash_engine::forward::attention::HostAttentionProjectionBackend,
    /// The host-weight GEMV's three instantiations, from their own pinned cubin.
    hostw: HostwKernels,
    /// Per kv-source layer: the compressor's staged (weight, mode) pair for wkv and, above ratio 1,
    /// for wgate. A layer whose source serves neither released dtype is absent and stays on the host
    /// rule, which is the same decline the wq_b half uses.
    compressor: std::collections::BTreeMap<usize, CompressorWeights>,
    /// Per kv-source layer: the indexer's wk weight in B12x bf16 order.
    index_wk: std::collections::BTreeMap<usize, Deviceptr>,
    /// Per index-source layer: the per-query weights projection in B12x bf16 order.
    weights_proj: std::collections::BTreeMap<usize, Deviceptr>,
    /// The widest activation row any served statement stages: dim f32 values.
    wide_act: Deviceptr,
    /// The widest output row any served statement writes: head_dim f32 values (its f32 form).
    wide_out: Deviceptr,
}

/// One staged compressor projection: the B12x weight and the mode whose load path it needs.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct StagedHostw {
    weight: Deviceptr,
    mode: sparkflash_kernels::hostw_gemv_launch::Mode,
}

/// A kv-source layer's compressor projections: wkv always, wgate only above ratio 1.
#[cfg(feature = "cuda")]
struct CompressorWeights {
    wkv: StagedHostw,
    wgate: Option<StagedHostw>,
}

/// The host-weight GEMV's three symbols, loaded from one cubin.
#[cfg(feature = "cuda")]
struct HostwKernels {
    bf16_weight: sparkflash_kernels::cuda_driver::Kernel,
    bf16_weight_f32_stores: sparkflash_kernels::cuda_driver::Kernel,
    f32_weight_f32_stores: sparkflash_kernels::cuda_driver::Kernel,
}

#[cfg(feature = "cuda")]
impl<'a> SeamDevice<'a> {
    /// Verifies the pinned cubin, loads the f32-activation mode, allocates the per-call row buffers
    /// and stages every index-source layer's matrix from the checkpoint source.
    ///
    /// The staging happens here, before the position loop starts, so each layer's 5.24 MB read, B12x
    /// permutation and upload are one-time warmup and never a step cost. A layer whose source does
    /// not serve the released F8 pair is declined and stays on the host rule.
    ///
    /// # Errors
    /// A geometry the released checkpoint cannot have, a cubin that disagrees with the baseline
    /// pin, a tensor the source serves in the released dtype but with the wrong byte count, or an
    /// allocation/copy failure.
    fn build(
        cuda: &'a sparkflash_kernels::cuda_driver::Cuda,
        config: &sparkflash_engine::config::ModelConfig,
        kernel_dir: &std::path::Path,
        src: &dyn sparkflash_engine::weights::WeightSource,
    ) -> Result<Self, String> {
        let cols = config
            .index_n_heads
            .checked_mul(config.index_head_dim)
            .ok_or_else(|| "ref seam indexer wq_b: the output width overflows".to_string())?;
        let inner = config.q_lora_rank;
        if cols == 0 || inner == 0 || !cols.is_multiple_of(32) || !inner.is_multiple_of(32) {
            return Err(format!(
                "ref seam indexer wq_b: {cols}x{inner} is not whole 32-value blocks, so it is not the released checkpoint layout"
            ));
        }
        if inner > sparkflash_kernels::mxfp8_gemv_launch::MAX_INNER as usize {
            return Err(format!(
                "ref seam indexer wq_b: the reduction width {inner} is above the kernel's {}",
                sparkflash_kernels::mxfp8_gemv_launch::MAX_INNER
            ));
        }
        let (cubin, _sha256) = load_pinned_addition(
            kernel_dir,
            sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
            "ref seam indexer wq_b",
        )?;
        let (_module, kernel) = sparkflash_kernels::mxfp8_gemv_launch::load_mode(
            cuda,
            &cubin,
            sparkflash_kernels::mxfp8_gemv_launch::Mode::NonFusedF32Activation,
        )?;
        let mut allocated = 0usize;
        let mut allocations = Vec::new();
        let act = cuda
            .alloc(inner * 4, &mut allocated)
            .map_err(|error| format!("ref seam indexer wq_b: activation row buffer: {error}"))?;
        allocations.push(act);
        let out = cuda
            .alloc(cols * 2, &mut allocated)
            .map_err(|error| format!("ref seam indexer wq_b: output row buffer: {error}"))?;
        allocations.push(out);
        let width = config.head_dim.max(config.index_head_dim).max(cols);
        let wide_act = cuda
            .alloc(config.dim * 4, &mut allocated)
            .map_err(|error| {
                format!("ref seam host-weight GEMV: activation row buffer: {error}")
            })?;
        allocations.push(wide_act);
        let wide_out = cuda
            .alloc(width * 4, &mut allocated)
            .map_err(|error| format!("ref seam host-weight GEMV: output row buffer: {error}"))?;
        allocations.push(wide_out);
        let (hostw_cubin, _hostw_sha256) = load_pinned_addition(
            kernel_dir,
            sparkflash_kernels::hostw_gemv_launch::KERNEL,
            "ref seam host-weight GEMV",
        )?;
        let loader = |mode| {
            sparkflash_kernels::hostw_gemv_launch::load_mode(cuda, &hostw_cubin, mode)
                .map(|(_module, kernel)| kernel)
        };
        let hostw = HostwKernels {
            bf16_weight: loader(sparkflash_kernels::hostw_gemv_launch::Mode::Bf16Weight)?,
            bf16_weight_f32_stores: loader(
                sparkflash_kernels::hostw_gemv_launch::Mode::Bf16WeightF32Stores,
            )?,
            f32_weight_f32_stores: loader(
                sparkflash_kernels::hostw_gemv_launch::Mode::F32WeightF32Stores,
            )?,
        };
        let mut device = Self {
            cuda,
            kernel,
            weights: std::collections::BTreeMap::new(),
            declined: std::collections::BTreeSet::new(),
            act,
            out,
            cols,
            inner,
            allocations,
            host: sparkflash_engine::forward::attention::HostAttentionProjectionBackend,
            hostw,
            compressor: std::collections::BTreeMap::new(),
            index_wk: std::collections::BTreeMap::new(),
            weights_proj: std::collections::BTreeMap::new(),
            wide_act,
            wide_out,
        };
        for &layer in &config.index_source_layers {
            device
                .weight_for(layer, src)
                .map_err(|error| format!("ref seam indexer wq_b: layer {layer}: {error}"))?;
        }
        // The host-weight statements: the compressor's two projections on the KV-source layers and
        // the indexer's two on the layers that run it. Each is a byte permutation and one upload,
        // done here so no statement pays a read on the step.
        for &layer in &config.kv_source_layers {
            let ratio = config.compress_ratio(layer).unwrap_or(0);
            if ratio == 0 {
                continue;
            }
            let Some(wkv) = device.stage_hostw(
                &format!("layers.{layer}.attn.compressor.wkv.weight"),
                config.dim,
                config.head_dim,
                OutStores::F32Row,
                src,
            )?
            else {
                continue;
            };
            let wgate = if ratio > 1 {
                device.stage_hostw(
                    &format!("layers.{layer}.attn.compressor.wgate.weight"),
                    config.dim,
                    config.head_dim,
                    OutStores::F32Row,
                    src,
                )?
            } else {
                None
            };
            device
                .compressor
                .insert(layer, CompressorWeights { wkv, wgate });
            if let Some(wk) = device.stage_hostw(
                &format!("layers.{layer}.attn.indexer.wk.weight"),
                config.head_dim,
                config.index_head_dim,
                OutStores::F32Row,
                src,
            )? {
                device.index_wk.insert(layer, wk.weight);
            }
        }
        for &layer in &config.index_source_layers {
            if let Some(proj) = device.stage_hostw(
                &format!("layers.{layer}.attn.indexer.weights_proj.weight"),
                config.dim,
                config.index_n_heads,
                OutStores::Bf16Row,
                src,
            )? {
                device.weights_proj.insert(layer, proj.weight);
            }
        }
        Ok(device)
    }

    /// One layer's weight on the device, read from the source the host rule reads on first use and
    /// cached for the run. A source that does not serve the released F8-E4M3 + E8-E8M0 pair for this
    /// layer declines the layer instead of erroring: the arm then answers `None` and the engine's
    /// host rule runs, which is the only other arm that exists.
    ///
    /// # Errors
    /// A tensor that the source serves as the released pair but whose byte length disagrees with the
    /// declared geometry, plus any allocation/copy failure.
    fn weight_for(
        &mut self,
        layer: usize,
        src: &dyn sparkflash_engine::weights::WeightSource,
    ) -> Result<Option<(Deviceptr, Deviceptr)>, sparkflash_engine::linalg::EngineError> {
        if let Some(pair) = self.weights.get(&layer) {
            return Ok(Some(*pair));
        }
        if self.declined.contains(&layer) {
            return Ok(None);
        }
        let shape = |detail: String| sparkflash_engine::linalg::EngineError::Shape {
            what: "ref seam indexer wq_b".to_owned(),
            detail,
        };
        let name = format!("layers.{layer}.attn.indexer.wq_b.weight");
        let scale_name = format!("layers.{layer}.attn.indexer.wq_b.scale");
        let spec = src
            .spec(&name)
            .map_err(|error| shape(format!("{name}: {error}")))?;
        let released = spec.dtype == sparkflash_engine::dtype::DType::F8E4M3
            && spec.shape.as_slice() == [self.cols, self.inner];
        if !released {
            self.declined.insert(layer);
            return Ok(None);
        }
        let scale_spec = src
            .spec(&scale_name)
            .map_err(|error| shape(format!("{scale_name}: {error}")))?;
        let scale_released = scale_spec.dtype == sparkflash_engine::dtype::DType::F8E8M0
            && scale_spec.elements() == (self.cols / 32 * (self.inner / 32)) as u128;
        if !scale_released {
            self.declined.insert(layer);
            return Ok(None);
        }
        let raw = src
            .read(&name)
            .map_err(|error| shape(format!("{name}: {error}")))?;
        if raw.len() != self.cols * self.inner {
            return Err(shape(format!(
                "{name}: {} bytes, the {cols}x{inner} E4M3 payload is {expected}",
                raw.len(),
                cols = self.cols,
                inner = self.inner,
                expected = self.cols * self.inner
            )));
        }
        let scales = src
            .read(&scale_name)
            .map_err(|error| shape(format!("{scale_name}: {error}")))?;
        let expected_scales = (self.cols / 32) * (self.inner / 32);
        if scales.len() != expected_scales {
            return Err(shape(format!(
                "{scale_name}: {} bytes, the grid is {expected_scales}",
                scales.len()
            )));
        }
        // The checkpoint stores the weight row-major; the kernel addresses the B12x layout. The
        // permutation changes no byte, and it is done once per layer, off the step.
        let mut tiled = vec![0u8; raw.len()];
        for col in 0..self.cols {
            for k in 0..self.inner {
                tiled[sparkflash_kernels::mxfp8_ref::b12x_weight_index(col, k, self.inner)] =
                    raw[col * self.inner + k];
            }
        }
        let failure = |what: &str| {
            let what = what.to_owned();
            move |error: String| shape(format!("ref seam indexer wq_b: {what}: {error}"))
        };
        let mut allocated = 0usize;
        let weight = self
            .cuda
            .alloc(tiled.len(), &mut allocated)
            .map_err(failure("weight buffer"))?;
        self.allocations.push(weight);
        self.cuda
            .copy_htod(&weight, &tiled)
            .map_err(failure("weight upload"))?;
        let grid = self
            .cuda
            .alloc(scales.len(), &mut allocated)
            .map_err(failure("scale buffer"))?;
        self.allocations.push(grid);
        self.cuda
            .copy_htod(&grid, &scales)
            .map_err(failure("scale upload"))?;
        self.weights.insert(layer, (weight, grid));
        Ok(Some((weight, grid)))
    }

    /// Stages one host-weight projection: the checkpoint's own values in the kernel's B12x order.
    ///
    /// The weight is the checkpoint's f32 or bf16 tensor copied element by element into the tiling
    /// the kernel addresses. That is a permutation of bytes, not a conversion: no value is decoded,
    /// quantised, scaled or re-rounded, so the kernel multiplies exactly the numbers the host rule's
    /// matmul multiplies. A source that serves the tensor in another dtype, or in another shape,
    /// declines it (answers None) and the engine's host rule runs for that layer, which is the
    /// pre-arm arithmetic and not a third behaviour.
    ///
    /// # Errors
    /// A tensor the source serves in the released dtype but with a byte count that is not the
    /// declared geometry, plus any allocation or upload failure.
    fn stage_hostw(
        &mut self,
        name: &str,
        inner: usize,
        cols: usize,
        stores: OutStores,
        src: &dyn sparkflash_engine::weights::WeightSource,
    ) -> Result<Option<StagedHostw>, String> {
        use sparkflash_engine::dtype::DType;
        use sparkflash_kernels::hostw_gemv_launch::Mode;
        let Ok(spec) = src.spec(name) else {
            return Ok(None);
        };
        if spec.shape.as_slice() != [cols, inner] {
            return Ok(None);
        }
        let mode = match (spec.dtype, stores) {
            (DType::F32, OutStores::F32Row) => Mode::F32WeightF32Stores,
            (DType::BF16, OutStores::F32Row) => Mode::Bf16WeightF32Stores,
            (DType::BF16, OutStores::Bf16Row) => Mode::Bf16Weight,
            _ => return Ok(None),
        };
        let raw = src
            .read(name)
            .map_err(|error| format!("ref seam host-weight GEMV: {name}: {error}"))?;
        let element = mode.weight_element_bytes();
        if raw.len() != cols * inner * element {
            return Err(format!(
                "ref seam host-weight GEMV: {name}: {} bytes, the {cols}x{inner} weight is {}",
                raw.len(),
                cols * inner * element
            ));
        }
        let mut upload = vec![0u8; raw.len()];
        for col in 0..cols {
            for k in 0..inner {
                let destination =
                    sparkflash_kernels::hostw_gemv_ref::b12x_hostw_weight_index(col, k, inner)
                        * element;
                let source = (col * inner + k) * element;
                upload[destination..destination + element]
                    .copy_from_slice(&raw[source..source + element]);
            }
        }
        let mut allocated = 0usize;
        let weight = self
            .cuda
            .alloc(upload.len(), &mut allocated)
            .map_err(|error| {
                format!("ref seam host-weight GEMV: {name}: weight buffer: {error}")
            })?;
        self.allocations.push(weight);
        self.cuda.copy_htod(&weight, &upload).map_err(|error| {
            format!("ref seam host-weight GEMV: {name}: weight upload: {error}")
        })?;
        Ok(Some(StagedHostw { weight, mode }))
    }

    /// One row of one staged host-weight projection through its kernel mode.
    ///
    /// The activation row is the engine's f32 row unchanged (these statements have no activation
    /// quantiser, which is the whole reason they need their own kernel), the output is the
    /// accumulator row as f32 or its single bf16 rounding, and every other step - narrowing, norms,
    /// RoPE, the index factor, pooling - still runs on the host after this returns.
    ///
    /// # Errors
    /// Any copy, launch or driver failure, and a returned row whose length disagrees with the shape
    /// the caller asked for.
    fn run_hostw(
        &mut self,
        staged: StagedHostw,
        x: &[f32],
        cols: usize,
        inner: usize,
        layer: usize,
    ) -> Result<Vec<f32>, sparkflash_engine::linalg::EngineError> {
        use sparkflash_kernels::hostw_gemv_launch::Mode;
        let shape = |detail: String| sparkflash_engine::linalg::EngineError::Shape {
            what: "ref seam host-weight GEMV".to_owned(),
            detail,
        };
        let mut act_bytes = Vec::with_capacity(inner * 4);
        for value in x {
            act_bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.cuda
            .copy_htod(&self.wide_act, &act_bytes)
            .map_err(|error| shape(format!("activation row htod: {error}")))?;
        let kernel = match staged.mode {
            Mode::Bf16Weight => self.hostw.bf16_weight,
            Mode::Bf16WeightF32Stores => self.hostw.bf16_weight_f32_stores,
            Mode::F32WeightF32Stores => self.hostw.f32_weight_f32_stores,
        };
        let params = sparkflash_kernels::hostw_gemv_launch::HostwGemvParams::new(
            self.wide_act,
            staged.weight,
            self.wide_out,
            1,
            cols as u32,
            inner as u32,
            0,
            layer as u32,
            0,
        );
        sparkflash_kernels::hostw_gemv_launch::launch_mode(self.cuda, kernel, staged.mode, &params)
            .map_err(|error| shape(format!("launch: {error}")))?;
        let mut raw = vec![0u8; cols * staged.mode.out_element_bytes()];
        self.cuda
            .copy_dtoh(&mut raw, &self.wide_out)
            .map_err(|error| shape(format!("output row dtoh: {error}")))?;
        let values = decode_hostw_row(&raw, cols, staged.mode).map_err(shape)?;
        Ok(values)
    }
}

/// Whether one host-weight statement's kernel stores its accumulator as f32 or as one bf16 round.
///
/// It is the caller's rule, not the weight's dtype: the compressor's rows are pooled in f32 and the
/// indexer's key rows feed rms_norm in f32, while the per-query weights are a bf16 Linear and the
/// host narrows them anyway.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum OutStores {
    /// The f32 accumulator itself.
    F32Row,
    /// The accumulator rounded once to bf16.
    Bf16Row,
}

/// Decodes one host-weight kernel output row into f32 values.
#[cfg(feature = "cuda")]
fn decode_hostw_row(
    raw: &[u8],
    cols: usize,
    mode: sparkflash_kernels::hostw_gemv_launch::Mode,
) -> Result<Vec<f32>, String> {
    let element = mode.out_element_bytes();
    if raw.len() != cols * element {
        return Err(format!(
            "the host-weight GEMV returned {} bytes, the {cols}-column row is {}",
            raw.len(),
            cols * element
        ));
    }
    let mut values = Vec::with_capacity(cols);
    for chunk in raw.chunks_exact(element) {
        if element == 4 {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        } else {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            values.push(f32::from_bits(u32::from(bits) << 16));
        }
    }
    Ok(values)
}

#[cfg(feature = "cuda")]
impl Drop for SeamDevice<'_> {
    fn drop(&mut self) {
        let allocations: Vec<&Deviceptr> = self.allocations.iter().collect();
        self.cuda.free_buffers(&allocations);
    }
}

#[cfg(feature = "cuda")]
impl sparkflash_engine::forward::attention::AttentionProjectionBackend for SeamDevice<'_> {
    fn wq_a(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn sparkflash_engine::weights::WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, sparkflash_engine::linalg::EngineError> {
        sparkflash_engine::forward::attention::AttentionProjectionBackend::wq_a(
            &mut self.host,
            config,
            layer,
            prefix,
            x,
            src,
            limit,
        )
    }

    fn wq_b(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        layer: usize,
        prefix: &str,
        latent: &[f32],
        src: &dyn sparkflash_engine::weights::WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, sparkflash_engine::linalg::EngineError> {
        sparkflash_engine::forward::attention::AttentionProjectionBackend::wq_b(
            &mut self.host,
            config,
            layer,
            prefix,
            latent,
            src,
            limit,
        )
    }

    fn wkv(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        layer: usize,
        prefix: &str,
        x: &[f32],
        src: &dyn sparkflash_engine::weights::WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, sparkflash_engine::linalg::EngineError> {
        sparkflash_engine::forward::attention::AttentionProjectionBackend::wkv(
            &mut self.host,
            config,
            layer,
            prefix,
            x,
            src,
            limit,
        )
    }

    fn wo_a(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        prefix: &str,
        wide: &[f32],
        src: &dyn sparkflash_engine::weights::WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, sparkflash_engine::linalg::EngineError> {
        sparkflash_engine::forward::attention::AttentionProjectionBackend::wo_a(
            &mut self.host,
            config,
            prefix,
            wide,
            src,
            limit,
        )
    }

    fn wo_b(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        prefix: &str,
        grouped: &[f32],
        src: &dyn sparkflash_engine::weights::WeightSource,
        limit: usize,
    ) -> Result<Vec<f32>, sparkflash_engine::linalg::EngineError> {
        sparkflash_engine::forward::attention::AttentionProjectionBackend::wo_b(
            &mut self.host,
            config,
            prefix,
            grouped,
            src,
            limit,
        )
    }

    /// The indexer's own `wq_b` on the pinned decode GEMV, in the one mode whose arithmetic is the
    /// host rule: f32 activation row (the engine's native activation, unchanged), E4M3 weight with
    /// its ue8m0 grid decoded in-kernel, a rounded multiply and a rounded add per element (no FMA),
    /// one accumulator per output column (no split-K), stored as bf16 (the host's `bf16_round`).
    ///
    /// `None` means "let the engine's host rule answer": a row count this arm does not serve, or a
    /// layer whose source does not serve the released F8 pair. The host rule is the only other arm,
    /// so a decline is the pre-arm arithmetic, not a third behaviour.
    fn indexer_wq_b(
        &mut self,
        _config: &sparkflash_engine::config::ModelConfig,
        layer: usize,
        _prefix: &str,
        latent: &[f32],
        src: &dyn sparkflash_engine::weights::WeightSource,
        _limit: usize,
    ) -> Result<Option<Vec<f32>>, sparkflash_engine::linalg::EngineError> {
        if latent.len() != self.inner {
            return Ok(None);
        }
        let Some((weight, scales)) = self.weight_for(layer, src)? else {
            return Ok(None);
        };
        let shape = |detail: String| sparkflash_engine::linalg::EngineError::Shape {
            what: "ref seam indexer wq_b".to_owned(),
            detail,
        };
        let activation =
            sparkflash_engine::forward::attention::indexer_activation_native(latent, self.inner);
        let mut act_bytes = Vec::with_capacity(self.inner * 4);
        for value in &activation {
            act_bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.cuda
            .copy_htod(&self.act, &act_bytes)
            .map_err(|error| shape(format!("activation row htod: {error}")))?;
        let params = sparkflash_kernels::mxfp8_dense_launch::Mxfp8DenseParams::new(
            self.act,
            self.act,
            weight,
            scales,
            self.out,
            1,
            self.cols as u32,
            self.inner as u32,
            0,
            layer as u32,
            0,
        );
        sparkflash_kernels::mxfp8_gemv_launch::launch(self.cuda, self.kernel, &params)
            .map_err(|error| shape(format!("launch: {error}")))?;
        let mut raw = vec![0u8; self.cols * 2];
        self.cuda
            .copy_dtoh(&mut raw, &self.out)
            .map_err(|error| shape(format!("output row dtoh: {error}")))?;
        let values = bf16_from_le_bytes(&raw).map_err(shape)?;
        if values.len() != self.cols {
            return Err(shape(format!(
                "the kernel returned {} values, the row is {} wide",
                values.len(),
                self.cols
            )));
        }
        Ok(Some(values))
    }

    /// The compressor's wkv projection on the host-weight decode GEMV.
    ///
    /// The layer's f32 (ratio > 1) or bf16 (ratio 1) weight is staged once, off the step, and the
    /// row the engine passes is the layer input unchanged: no activation quantiser sits in front of
    /// this statement, which is why it cannot use the coded GEMV the wq_b half runs on. The kernel
    /// writes the f32 accumulator row, and the engine's narrowing, norm and pooling run on it.
    ///
    /// None means the engine's host rule answers: a multi-row call (the seam step is one query at a
    /// time, so this is chunked prefill), or a layer this arm declined at build.
    fn compressor_wkv(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        layer: usize,
        _prefix: &str,
        x: &[f32],
        _src: &dyn sparkflash_engine::weights::WeightSource,
        _limit: usize,
    ) -> Result<Option<Vec<f32>>, sparkflash_engine::linalg::EngineError> {
        let Some(staged) = self.compressor.get(&layer).map(|entry| entry.wkv) else {
            return Ok(None);
        };
        if x.len() != config.dim {
            return Ok(None);
        }
        self.run_hostw(staged, x, config.head_dim, config.dim, layer)
            .map(Some)
    }

    /// The compressor's wgate projection on the host-weight decode GEMV; see [Self::compressor_wkv].
    fn compressor_wgate(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        layer: usize,
        _prefix: &str,
        x: &[f32],
        _src: &dyn sparkflash_engine::weights::WeightSource,
        _limit: usize,
    ) -> Result<Option<Vec<f32>>, sparkflash_engine::linalg::EngineError> {
        let Some(staged) = self.compressor.get(&layer).and_then(|entry| entry.wgate) else {
            return Ok(None);
        };
        if x.len() != config.dim {
            return Ok(None);
        }
        self.run_hostw(staged, x, config.head_dim, config.dim, layer)
            .map(Some)
    }

    /// The indexer's wk projection (bf16 weight) on the host-weight decode GEMV: the f32 row the
    /// engine's k_norm, RoPE and fp4 round trip then consume.
    ///
    /// None for a multi-row latent (chunked prefill), a layer this arm declined at build, or a
    /// layer that does not own the index keys.
    fn indexer_wk(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        layer: usize,
        _prefix: &str,
        latent: &[f32],
        _src: &dyn sparkflash_engine::weights::WeightSource,
        _limit: usize,
    ) -> Result<Option<Vec<f32>>, sparkflash_engine::linalg::EngineError> {
        let Some(weight) = self.index_wk.get(&layer).copied() else {
            return Ok(None);
        };
        if latent.len() != config.head_dim {
            return Ok(None);
        }
        let staged = StagedHostw {
            weight,
            mode: sparkflash_kernels::hostw_gemv_launch::Mode::Bf16WeightF32Stores,
        };
        self.run_hostw(
            staged,
            latent,
            config.index_head_dim,
            config.head_dim,
            layer,
        )
        .map(Some)
    }

    /// The indexer's per-query weights projection (bf16 weight, bf16 output) on the host-weight
    /// decode GEMV: the rows the host then multiplies by the index factor and narrows.
    ///
    /// None for a multi-row call or a layer this arm declined at build.
    fn indexer_weights_proj(
        &mut self,
        config: &sparkflash_engine::config::ModelConfig,
        layer: usize,
        _prefix: &str,
        x: &[f32],
        seqlen: usize,
        _src: &dyn sparkflash_engine::weights::WeightSource,
        _limit: usize,
    ) -> Result<Option<Vec<f32>>, sparkflash_engine::linalg::EngineError> {
        if seqlen != 1 || x.len() != config.dim {
            return Ok(None);
        }
        let Some(weight) = self.weights_proj.get(&layer).copied() else {
            return Ok(None);
        };
        let staged = StagedHostw {
            weight,
            mode: sparkflash_kernels::hostw_gemv_launch::Mode::Bf16Weight,
        };
        self.run_hostw(staged, x, config.index_n_heads, config.dim, layer)
            .map(Some)
    }
}

/// The gathered, replicated released shared-expert view for one layer.
#[cfg(feature = "cuda")]
struct SharedExpertGather {
    /// The full released views, as the engine's replicated H5 seam reads them.
    source: ArenaHostSource,
    /// The six per-rank arena reads (w1/w3/w2 weights and their scale grids).
    local_reads: usize,
    /// The six transport all-gathers that replicated the shards.
    gather_calls: usize,
    /// The bytes of the full released tensors reassembled for the engine seam.
    full_bytes: usize,
}

/// Replicates the released shared expert for one layer before the seam runs.
///
/// The resident plan stores the shared expert TP-column/row sharded (w1/w3 as column shards, w2 as a
/// row shard; the plan is fixed), while the reference H5 seam is a *replicated* dense expert. So each
/// rank reads its six shards from the arena, the loop all-gathers them over the forward transport,
/// and the full [2304,5120]/[5120,2304] F8-E4M3 + E8-E8M0 views are reassembled host-side. The
/// gathered bytes are transient: they are dropped once the layer's seam has run.
///
/// # Errors
/// Refuses missing plan entries, dtypes other than the released F8-E4M3/E8-E8M0 pair, geometries that
/// are not the plan's whole 32-value-block column/row split, a byte-size overflow, or an arena slice
/// whose byte count is not its shard's.
#[cfg(feature = "cuda")]
fn gather_shared_expert_source(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    arena: &sparkflash_device::arena::ResidentArena,
    request: &TokenStageRequest<'_>,
    layer: usize,
    transport: &TcpTransport,
) -> Result<SharedExpertGather, String> {
    use sparkflash_comm::transport::Transport;
    use sparkflash_engine::dtype::DType;
    use sparkflash_engine::tensor::TensorSpec;

    let config = request.config.ok_or_else(|| {
        "shared expert gather: the loop was built without the model config".to_string()
    })?;
    let dim = config.dim;
    let world = request.world;
    let inter = config
        .moe_inter_dim
        .checked_mul(config.n_shared_experts.max(1))
        .ok_or_else(|| "shared expert gather: the intermediate width overflows".to_string())?;
    if world == 0 || !inter.is_multiple_of(world) {
        return Err(format!(
            "shared expert gather: intermediate width {inter} does not split over {world} ranks"
        ));
    }
    let local_inter = inter / world;
    if !dim.is_multiple_of(32) || !inter.is_multiple_of(32) || !local_inter.is_multiple_of(32) {
        return Err(format!(
            "shared expert gather: dim {dim}, intermediate {inter} and shard {local_inter} must be whole 32-value blocks"
        ));
    }
    let dim_blocks = dim / 32;
    let inter_blocks = inter / 32;
    let shard_blocks = local_inter / 32;
    let prefix = format!("layers.{layer}.ffn.shared_experts");
    let mut local_reads = 0usize;
    let mut gather_calls = 0usize;
    let mut tensors = std::collections::BTreeMap::new();

    let mut read_shard = |leaf: &str, suffix: &str, want: usize| -> Result<Vec<u8>, String> {
        let name = format!("{prefix}.{leaf}.{suffix}");
        let want_dtype = if suffix == "scale" {
            "F8_E8M0"
        } else {
            "F8_E4M3"
        };
        match planned_dtype(request.resident_plan, &name) {
            Some(dtype) if dtype == want_dtype => {}
            Some(other) => {
                return Err(format!(
                    "shared expert gather: {name} is planned as {other}; the released shared expert is {want_dtype}"
                ));
            }
            None => {
                return Err(format!(
                    "shared expert gather: the rank plan does not carry {name}, so its encoding is unknown"
                ));
            }
        }
        local_reads += 1;
        read_arena_bytes(cuda, arena, &name, want)
    };

    let mut gather_one = |local: &[u8], index: usize, what: &str| -> Result<Vec<Vec<u8>>, String> {
        if !local.len().is_multiple_of(2) {
            return Err(format!(
                "shared expert gather: {what} shard is {} bytes, not a whole number of u16 words",
                local.len()
            ));
        }
        let words: Vec<u16> = local
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        let mut gathered = vec![0u16; world * words.len()];
        let sequence = SHARED_EXPERT_GATHER_SEQ.wrapping_sub(
            (layer as u64)
                .saturating_mul(8)
                .saturating_add(index as u64),
        );
        transport
            .all_gather_bf16(&words, &mut gathered, sequence)
            .map_err(|error| {
                format!(
                    "shared expert gather: layer {layer} {what} all-gather ({sequence:#x}): {error}"
                )
            })?;
        gather_calls += 1;
        let mut shards = Vec::with_capacity(world);
        for rank in 0..world {
            let start = rank * words.len();
            shards.push(
                gathered[start..start + words.len()]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect(),
            );
        }
        Ok(shards)
    };

    // The arena stores these weight shards B12x micro-tiled for the dense kernels (the shared
    // expert is a B12x group; the census reports them tiled), but the engine's host H5 seam reads
    // the checkpoint's stored row-major order. Untile each rank's shard with its own shard
    // geometry first: w1/w3 are [local_inter, dim] row blocks, w2 is [dim, local_inter] column
    // blocks. The scale grids are NOT tiled (the checkpoint grid already is the B12x grid), so they
    // are joined as read.
    let untile_shard =
        |tiled: &[u8], cols: usize, inner: usize, what: &str| -> Result<Vec<u8>, String> {
            sparkflash_device::arena::untile_b12x(tiled, cols, inner).map_err(|error| {
                format!("shared expert gather: {what} is not B12x-untileable: {error}")
            })
        };
    let w1_weight = read_shard("w1", "weight", local_inter * dim)?;
    let w1_scale = read_shard("w1", "scale", shard_blocks * dim_blocks)?;
    let w3_weight = read_shard("w3", "weight", local_inter * dim)?;
    let w3_scale = read_shard("w3", "scale", shard_blocks * dim_blocks)?;
    let w2_weight = read_shard("w2", "weight", dim * local_inter)?;
    let w2_scale = read_shard("w2", "scale", dim_blocks * shard_blocks)?;

    // The plan itself is the proof the shards are B12x and of the geometry to untile them with:
    // b12x_layout is what the arena upload used, so a plan that reports B12x here is the same
    // predicate that tiled the bytes. Refuse a plan whose geometry is not the released shard's.
    let b12x_geometry = |leaf: &str| -> Result<(usize, usize), String> {
        let name = format!("{prefix}.{leaf}");
        let plan = request.resident_plan.ok_or_else(|| {
            "shared expert gather: the loop was built without the rank plan".to_string()
        })?;
        match plan.operand_layout(&name) {
            Some(sparkflash_device::resident::OperandLayout::B12x { cols, inner, .. }) => {
                Ok((cols, inner))
            }
            Some(other) => Err(format!(
                "shared expert gather: {name} is planned as {}; the shared-expert weight shards are B12x",
                other.as_str()
            )),
            None => Err(format!(
                "shared expert gather: the rank plan does not carry {name}"
            )),
        }
    };
    for (leaf, want) in [
        ("w1.weight", (local_inter, dim)),
        ("w3.weight", (local_inter, dim)),
        ("w2.weight", (dim, local_inter)),
    ] {
        let got = b12x_geometry(leaf)?;
        if got != want {
            return Err(format!(
                "shared expert gather: {leaf} is planned B12x {got:?}, the released shard is {want:?}"
            ));
        }
    }
    let w1_weight = untile_shard(&w1_weight, local_inter, dim, "w1.weight")?;
    let w3_weight = untile_shard(&w3_weight, local_inter, dim, "w3.weight")?;
    let w2_weight = untile_shard(&w2_weight, dim, local_inter, "w2.weight")?;

    let full_w1 = join_column_shards(&gather_one(&w1_weight, 0, "w1.weight")?);
    let full_w1_scale = join_column_shards(&gather_one(&w1_scale, 1, "w1.scale")?);
    let full_w3 = join_column_shards(&gather_one(&w3_weight, 2, "w3.weight")?);
    let full_w3_scale = join_column_shards(&gather_one(&w3_scale, 3, "w3.scale")?);
    let full_w2 = join_row_shards(&gather_one(&w2_weight, 4, "w2.weight")?, dim, local_inter)?;
    let full_w2_scale = join_row_shards(
        &gather_one(&w2_scale, 5, "w2.scale")?,
        dim_blocks,
        shard_blocks,
    )?;

    let mut full_bytes = 0usize;
    let mut insert = |leaf: &str, suffix: &str, dtype: DType, shape: Vec<usize>, bytes: Vec<u8>| {
        let name = format!("{prefix}.{leaf}.{suffix}");
        full_bytes += bytes.len();
        tensors.insert(name.clone(), (TensorSpec { name, dtype, shape }, bytes));
    };
    insert("w1", "weight", DType::F8E4M3, vec![inter, dim], full_w1);
    insert(
        "w1",
        "scale",
        DType::F8E8M0,
        vec![inter_blocks, dim_blocks],
        full_w1_scale,
    );
    insert("w3", "weight", DType::F8E4M3, vec![inter, dim], full_w3);
    insert(
        "w3",
        "scale",
        DType::F8E8M0,
        vec![inter_blocks, dim_blocks],
        full_w3_scale,
    );
    insert("w2", "weight", DType::F8E4M3, vec![dim, inter], full_w2);
    insert(
        "w2",
        "scale",
        DType::F8E8M0,
        vec![dim_blocks, inter_blocks],
        full_w2_scale,
    );

    Ok(SharedExpertGather {
        source: ArenaHostSource { tensors },
        local_reads,
        gather_calls,
        full_bytes,
    })
}

/// The shared expert's resident device operands (WS-D).
///
/// The released shared expert is a replicated dense expert and the resident plan stores it
/// TP-column/row sharded, so [gather_shared_expert_source] all-gathers the shards once per layer at
/// loop start, and the host seam then decoded and multiplied those bytes on every position. This
/// struct keeps the same bytes on the device instead: w1/w3/w2 B12x-tiled once per layer here
/// (warmup, before the step timer) and their three E8M0 grids uploaded as stored, since the arena
/// already holds them in B12x order.
///
/// The arithmetic is the reference's by construction. The activation is bf16-rounded and quantised
/// with quant::quantize_activation_fp8, the same function the host seam calls; the gate, up and
/// down products run the decode GEMV's fused mode, which is bit-identical to the engine's
/// rounded-multiply-then-rounded-add twin because an E4M3 x E4M3 product is exact in f32 (proved
/// over all 256 codes and every served shape in tests/mxfp8_gemv_oracle.rs); and the SwiGLU +
/// quantise step is the wide (cols = 2304) shape of swiglu_mxfp8_quant_sm121, which rounds the
/// activation to bf16 before quantising exactly as the released model does.
///
/// Lifetime: one instance is built before the position loop and borrowed by every position and
/// layer. Its buffers belong to the process's CUDA context, so they are released when the run
/// ends; nothing is allocated per position.
#[cfg(feature = "cuda")]
struct SharedExpertDevice {
    /// One entry per layer, in plan order.
    layers: Vec<SharedExpertDeviceLayer>,
    /// The per-token scratch, allocated once.
    workspace: SharedExpertWorkspace,
    /// The fused decode GEMV (the coded native arm).
    gemv: sparkflash_kernels::cuda_driver::Kernel,
    /// The decode GEMV's f32-activation, f32-store instantiation (the f32-reference arm).
    gemv_f32: sparkflash_kernels::cuda_driver::Kernel,
    /// The SwiGLU + MX-quantise epilogue (the coded native arm only; the f32 arm's SwiGLU is host).
    swiglu: sparkflash_kernels::cuda_driver::Kernel,
    _gemv_module: sparkflash_kernels::cuda_driver::Module,
    _gemv_f32_module: sparkflash_kernels::cuda_driver::Module,
    _swiglu_module: sparkflash_kernels::cuda_driver::Module,
    /// Hidden size.
    dim: usize,
    /// Intermediate width (2304 on the released plan).
    inter: usize,
}

/// One layer's resident shared-expert weights and scales, B12x-tiled.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct SharedExpertDeviceLayer {
    w1: sparkflash_kernels::cuda_driver::Deviceptr,
    s1: sparkflash_kernels::cuda_driver::Deviceptr,
    w3: sparkflash_kernels::cuda_driver::Deviceptr,
    s3: sparkflash_kernels::cuda_driver::Deviceptr,
    w2: sparkflash_kernels::cuda_driver::Deviceptr,
    s2: sparkflash_kernels::cuda_driver::Deviceptr,
}

/// The shared expert's per-token scratch: one row of each operand and result.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct SharedExpertWorkspace {
    a_codes: sparkflash_kernels::cuda_driver::Deviceptr,
    a_scales: sparkflash_kernels::cuda_driver::Deviceptr,
    gate: sparkflash_kernels::cuda_driver::Deviceptr,
    up: sparkflash_kernels::cuda_driver::Deviceptr,
    codes: sparkflash_kernels::cuda_driver::Deviceptr,
    scales: sparkflash_kernels::cuda_driver::Deviceptr,
    out: sparkflash_kernels::cuda_driver::Deviceptr,
    /// The f32-reference arm's activation row: the pre-FFN hidden row, 5120 f32 = 20 KB.
    a_f32: sparkflash_kernels::cuda_driver::Deviceptr,
    /// The f32-reference arm's gate row, 2304 f32.
    gate_f32: sparkflash_kernels::cuda_driver::Deviceptr,
    /// The f32-reference arm's up row, 2304 f32.
    up_f32: sparkflash_kernels::cuda_driver::Deviceptr,
    /// The f32-reference arm's SwiGLU row, 2304 f32.
    swiglu_f32: sparkflash_kernels::cuda_driver::Deviceptr,
    /// The f32-reference arm's down output, 5120 f32.
    out_f32: sparkflash_kernels::cuda_driver::Deviceptr,
}

/// Stages the shared expert on the device for every layer: B12x repack, upload, budget admission.
///
/// Warmup only. The repack is permutation-only ([sparkflash_device::arena::tile_b12x]) and the
/// scale grids are uploaded as stored, so this function never touches the values. It admits its
/// own memory through [sparkflash_device::budget::BudgetAdmission] before the first allocation and
/// refuses by name when the projection does not fit, rather than discovering it mid-repack.
///
/// # Errors
/// A missing plan tensor or dtype, a repack geometry mismatch, a budget refusal, or a driver
/// allocation/copy failure.
#[cfg(feature = "cuda")]
fn build_shared_expert_device(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    request: &TokenStageRequest<'_>,
    shared: &[SharedExpertGather],
) -> Result<SharedExpertDevice, String> {
    use sparkflash_engine::weights::WeightSource;
    use sparkflash_kernels::mxfp8_gemv_launch;

    let kernel_dir = request.kernel_cubin_dir.ok_or_else(|| {
        "shared expert (device): the forward loop was built without --kernel-cubin-dir, so the GEMV and swiglu cubins are not staged".to_string()
    })?;
    let config = request.config.ok_or_else(|| {
        "shared expert (device): the loop was built without the model config".to_string()
    })?;
    let dim = config.dim;
    let inter = config
        .moe_inter_dim
        .checked_mul(config.n_shared_experts.max(1))
        .ok_or_else(|| "shared expert (device): the intermediate width overflows".to_string())?;
    if shared.len() != request.n_layers {
        return Err(format!(
            "shared expert (device): {} gathered layers for {} layers in the loop",
            shared.len(),
            request.n_layers
        ));
    }
    let (gemv_cubin, _gemv_sha) =
        load_pinned_addition(kernel_dir, mxfp8_gemv_launch::KERNEL, "shared expert")?;
    let (gemv_module, gemv) = mxfp8_gemv_launch::load(cuda, &gemv_cubin)?;
    // The same cubin carries the f32-reference arm's symbol; both modes are loaded here because
    // which arm runs is decided per token by --native-rounding, not at build time.
    let (gemv_f32_module, gemv_f32) = mxfp8_gemv_launch::load_mode(
        cuda,
        &gemv_cubin,
        mxfp8_gemv_launch::Mode::NonFusedF32ActivationF32Stores,
    )?;
    let (swiglu_cubin, _swiglu_sha) =
        load_pinned_addition(kernel_dir, "swiglu_mxfp8_quant_sm121", "shared expert")?;
    let (swiglu_module, swiglu) = cuda.load_swiglu_quant(&swiglu_cubin)?;
    // One B12x weight buffer and one scale grid per tensor, plus the per-token scratch.
    let weight_bytes = (2 * inter * dim + dim * inter) as u64;
    let scale_bytes = (2 * (inter / 32) * (dim / 32) + (dim / 32) * (inter / 32)) as u64;
    // The per-token scratch of both arms: the coded arm's row plus the f32 arm's five f32 buffers
    // (a_f32 20 KB, gate/up/swiglu 9.2 KB each, out_f32 20 KB).
    let workspace_bytes =
        dim + dim / 32 + 4 * inter + inter + inter / 32 + 2 * dim + 8 * dim + 12 * inter;
    let projected_device =
        request.n_layers as u64 * (weight_bytes + scale_bytes) + workspace_bytes as u64;
    // The host projection is the transient repack of one layer plus the scale grids.
    let projected_host = weight_bytes + scale_bytes;
    let available = sparkflash_device::budget::read_mem_available_bytes()
        .map_err(|error| format!("shared expert (device): MemAvailable: {error}"))?;
    sparkflash_device::budget::BudgetAdmission::admit(sparkflash_device::budget::BudgetRequest {
        mem_available_bytes: available,
        projected_host_bytes: projected_host,
        projected_device_bytes: projected_device,
    })
    .map_err(|error| format!("shared expert (device): budget: {error}"))?;
    let mut allocated = 0usize;
    let mut layers = Vec::with_capacity(shared.len());
    for (layer, gather) in shared.iter().enumerate() {
        let source: &dyn WeightSource = &gather.source;
        let base = format!("layers.{layer}.ffn.shared_experts");
        let (s1, s3, s2) = crate::moe_device::read_shared_scale_grids(source, &base, inter, dim)
            .map_err(|error| format!("shared expert (device): layer {layer} scales: {error}"))?;
        let mut stage = |leaf: &str,
                         cols: usize,
                         inner: usize|
         -> Result<sparkflash_kernels::cuda_driver::Deviceptr, String> {
            let name = format!("{base}.{leaf}.weight");
            let codes = source.read(&name).map_err(|error| {
                format!("shared expert (device): layer {layer} {name}: {error}")
            })?;
            let tiled =
                sparkflash_device::arena::tile_b12x(&codes, cols, inner).map_err(|error| {
                    format!("shared expert (device): layer {layer} {name}: {error}")
                })?;
            let pointer = cuda.alloc(tiled.len(), &mut allocated).map_err(|error| {
                format!("shared expert (device): layer {layer} {name} allocation: {error}")
            })?;
            cuda.copy_htod(&pointer, &tiled).map_err(|error| {
                format!("shared expert (device): layer {layer} {name} upload: {error}")
            })?;
            Ok(pointer)
        };
        let w1 = stage("w1", inter, dim)?;
        let w3 = stage("w3", inter, dim)?;
        let w2 = stage("w2", dim, inter)?;
        let mut upload = |bytes: &[u8],
                          what: &str|
         -> Result<sparkflash_kernels::cuda_driver::Deviceptr, String> {
            let pointer = cuda.alloc(bytes.len(), &mut allocated).map_err(|error| {
                format!("shared expert (device): layer {layer} {what} allocation: {error}")
            })?;
            cuda.copy_htod(&pointer, bytes).map_err(|error| {
                format!("shared expert (device): layer {layer} {what} upload: {error}")
            })?;
            Ok(pointer)
        };
        let s1 = upload(&s1, "w1.scale")?;
        let s3 = upload(&s3, "w3.scale")?;
        let s2 = upload(&s2, "w2.scale")?;
        layers.push(SharedExpertDeviceLayer {
            w1,
            s1,
            w3,
            s3,
            w2,
            s2,
        });
    }
    let mut fresh =
        |bytes: usize, what: &str| -> Result<sparkflash_kernels::cuda_driver::Deviceptr, String> {
            cuda.alloc(bytes, &mut allocated).map_err(|error| {
                format!("shared expert (device): workspace {what} allocation: {error}")
            })
        };
    let workspace = SharedExpertWorkspace {
        a_codes: fresh(dim, "a_codes")?,
        a_scales: fresh(dim / 32, "a_scales")?,
        gate: fresh(2 * inter, "gate")?,
        up: fresh(2 * inter, "up")?,
        codes: fresh(inter, "codes")?,
        scales: fresh(inter / 32, "scales")?,
        out: fresh(2 * dim, "out")?,
        a_f32: fresh(4 * dim, "a_f32")?,
        gate_f32: fresh(4 * inter, "gate_f32")?,
        up_f32: fresh(4 * inter, "up_f32")?,
        swiglu_f32: fresh(4 * inter, "swiglu_f32")?,
        out_f32: fresh(4 * dim, "out_f32")?,
    };
    Ok(SharedExpertDevice {
        layers,
        workspace,
        gemv,
        gemv_f32,
        swiglu,
        _gemv_module: gemv_module,
        _gemv_f32_module: gemv_f32_module,
        _swiglu_module: swiglu_module,
        dim,
        inter,
    })
}

/// Runs the shared expert for one position through the resident device operands.
///
/// Four launches: gate, up, the wide SwiGLU + quantise, and down. Returns the BF16 row the host
/// reference would have produced, as f32 (the addend the f32 MoE sum takes, model.py:893-904).
///
/// # Errors
/// A shape disagreement, a quantise refusal, an upload/launch/readback failure, or a non-finite
/// BF16 row.
#[cfg(feature = "cuda")]
fn run_shared_expert_device(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    request: &TokenStageRequest<'_>,
    layer: usize,
    hidden: &[f32],
    device: &SharedExpertDevice,
    out: &mut [f32],
) -> Result<(), String> {
    use sparkflash_kernels::mxfp8_dense_launch::Mxfp8DenseParams;
    use sparkflash_kernels::mxfp8_gemv_launch;

    let config = request.config.ok_or_else(|| {
        "shared expert (device): the loop was built without the model config".to_string()
    })?;
    let dim = device.dim;
    let inter = device.inter;
    if hidden.len() != dim || out.len() != dim {
        return Err(format!(
            "shared expert (device): hidden {} and out {} do not match dim {dim}",
            hidden.len(),
            out.len()
        ));
    }
    let operands = device
        .layers
        .get(layer)
        .ok_or_else(|| format!("shared expert (device): layer {layer} has no resident operands"))?;
    // The released Linear rounds its input to BF16 and then quantises it; the host seam calls the
    // same two functions, so the operand is the reference operand, not an approximation of it.
    let rounded: Vec<f32> = hidden
        .iter()
        .map(|value| sparkflash_engine::quant::bf16_round(*value))
        .collect();
    let quantised = sparkflash_engine::quant::quantize_activation_fp8(&rounded)
        .map_err(|error| format!("shared expert (device): activation quantise: {error}"))?;
    let workspace = &device.workspace;
    cuda.copy_htod(&workspace.a_codes, quantised.codes())
        .map_err(|error| format!("shared expert (device): activation upload: {error}"))?;
    cuda.copy_htod(&workspace.a_scales, quantised.scales())
        .map_err(|error| format!("shared expert (device): activation scales upload: {error}"))?;
    let gate = Mxfp8DenseParams::new(
        workspace.a_codes,
        workspace.a_scales,
        operands.w1,
        operands.s1,
        workspace.gate,
        1,
        inter as u32,
        dim as u32,
        0,
        layer as u32,
        0,
    );
    mxfp8_gemv_launch::launch(cuda, device.gemv, &gate)?;
    let up = Mxfp8DenseParams::new(
        workspace.a_codes,
        workspace.a_scales,
        operands.w3,
        operands.s3,
        workspace.up,
        1,
        inter as u32,
        dim as u32,
        0,
        layer as u32,
        0,
    );
    mxfp8_gemv_launch::launch(cuda, device.gemv, &up)?;
    // The wide SwiGLU + quantise: cols = 2304, one row, the released clamp limit as an f32 word.
    let blocks = (inter / WINDOW_QUANT_BLOCK) as u32;
    let swiglu_words = [
        workspace.gate.0,
        workspace.up.0,
        workspace.codes.0,
        workspace.scales.0,
        1,
        inter as u64,
        u64::from((config.swiglu_limit as f32).to_bits()),
        0,
        layer as u64,
        0,
    ];
    cuda.launch(
        device.swiglu,
        (blocks.div_ceil(8), 1, 1),
        (256, 1, 1),
        &swiglu_words,
    )?;
    let down = Mxfp8DenseParams::new(
        workspace.codes,
        workspace.scales,
        operands.w2,
        operands.s2,
        workspace.out,
        1,
        dim as u32,
        inter as u32,
        0,
        layer as u32,
        0,
    );
    mxfp8_gemv_launch::launch(cuda, device.gemv, &down)?;
    let mut bytes = vec![0u8; dim * 2];
    cuda.copy_dtoh(&mut bytes, &workspace.out)
        .map_err(|error| format!("shared expert (device): readback: {error}"))?;
    let row = bf16_from_le_bytes(&bytes)?;
    if row.iter().any(|value| !value.is_finite()) {
        return Err("shared expert (device): the BF16 row is not finite".to_string());
    }
    out.copy_from_slice(&row);
    Ok(())
}

/// The engine's `sigmoid`, copied rather than called because `moe::swiglu` is private: the two
/// branches are part of the arithmetic (the negative branch keeps `exp(x)` below one), not a
/// stylistic choice, and the f32 reference's row must be reproduced bit for bit.
#[cfg(feature = "cuda")]
fn shared_sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}

/// `silu(x) = x * sigmoid(x)`, the engine's `linalg::silu` expression.
#[cfg(feature = "cuda")]
fn shared_silu(x: f32) -> f32 {
    x * shared_sigmoid(x)
}

/// The f32-reference SwiGLU over the intermediate row: exactly `engine::forward::moe`'s `swiglu`,
/// i.e. the reference's clamps (each wrapped in `limit > 0`) around `silu`, and NO bf16 rounding.
///
/// # Errors
/// Mismatched gate, up and output lengths.
#[cfg(feature = "cuda")]
fn shared_swiglu_f32(limit: f32, gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<(), String> {
    if gate.len() != up.len() || out.len() != gate.len() {
        return Err(format!(
            "shared expert (f32 device): swiglu over {} gate, {} up and {} output values",
            gate.len(),
            up.len(),
            out.len()
        ));
    }
    if limit > 0.0 {
        for ((slot, gate), up) in out.iter_mut().zip(gate).zip(up) {
            let clamped_gate = gate.min(limit);
            let clamped_up = up.clamp(-limit, limit);
            *slot = shared_silu(clamped_gate) * clamped_up;
        }
    } else {
        for ((slot, gate), up) in out.iter_mut().zip(gate).zip(up) {
            *slot = shared_silu(*gate) * *up;
        }
    }
    Ok(())
}

/// Runs the shared expert's f32-reference arm for one position on resident device operands.
///
/// This is the arithmetic of the host `shared_expert_rows` seam - no activation quantisation and no
/// bf16 boundary rounding - with the three long reductions on the device's decode GEMV and only the
/// 2304-wide SwiGLU on the host:
///
/// * H2D the f32 pre-FFN row (20 KB, no bf16 round, no quantise),
/// * gate through the f32-activation/f32-store GEMV,
/// * up through the same GEMV,
/// * D2H gate and up as f32 (2 x 9.2 KB), host SwiGLU (clamps only),
/// * H2D the SwiGLU row as f32 (9.2 KB), down through the same GEMV,
/// * D2H 20 KB f32 and copy into the row with no bf16 round.
///
/// The f32-store kernel writes the accumulator itself, so the row is bit-identical to the host f32
/// reference while the weight traffic runs on the device.
///
/// # Errors
/// A shape disagreement, a quantise-free upload/launch/readback failure, or a non-finite row.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_shared_expert_f32_device(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    request: &TokenStageRequest<'_>,
    layer: usize,
    hidden: &[f32],
    device: &SharedExpertDevice,
    out: &mut [f32],
) -> Result<(), String> {
    use sparkflash_kernels::mxfp8_dense_launch::Mxfp8DenseParams;
    use sparkflash_kernels::mxfp8_gemv_launch::{self, Mode};

    let config = request.config.ok_or_else(|| {
        "shared expert (f32 device): the loop was built without the model config".to_string()
    })?;
    let dim = device.dim;
    let inter = device.inter;
    if hidden.len() != dim || out.len() != dim {
        return Err(format!(
            "shared expert (f32 device): hidden {} and out {} do not match dim {dim}",
            hidden.len(),
            out.len()
        ));
    }
    let operands = device.layers.get(layer).ok_or_else(|| {
        format!("shared expert (f32 device): layer {layer} has no resident operands")
    })?;
    // Attribute the three launches to this layer in the stage-timing ledger, as the other device
    // stages do, so the probe can read ffn_shared's kernel time and launch count.
    sparkflash_kernels::cuda_driver::stage_timing_set_layer(layer);
    let workspace = &device.workspace;
    // H2D the f32 row unchanged. The host reference this arm replaces rounds neither to bf16 nor to
    // E4M3: the row it multiplies is the pre-FFN hidden row itself.
    cuda.copy_htod(&workspace.a_f32, &f32_le_bytes(hidden))
        .map_err(|error| format!("shared expert (f32 device): activation upload: {error}"))?;
    let gemv = |activation: sparkflash_kernels::cuda_driver::Deviceptr,
                weight: sparkflash_kernels::cuda_driver::Deviceptr,
                weight_scales: sparkflash_kernels::cuda_driver::Deviceptr,
                output: sparkflash_kernels::cuda_driver::Deviceptr,
                cols: usize,
                inner: usize|
     -> Result<(), String> {
        let params = Mxfp8DenseParams::new(
            activation,
            workspace.a_scales,
            weight,
            weight_scales,
            output,
            1,
            cols as u32,
            inner as u32,
            0,
            layer as u32,
            0,
        );
        mxfp8_gemv_launch::launch_mode(
            cuda,
            device.gemv_f32,
            Mode::NonFusedF32ActivationF32Stores,
            &params,
        )
    };
    gemv(
        workspace.a_f32,
        operands.w1,
        operands.s1,
        workspace.gate_f32,
        inter,
        dim,
    )
    .map_err(|error| format!("shared expert (f32 device): gate launch: {error}"))?;
    gemv(
        workspace.a_f32,
        operands.w3,
        operands.s3,
        workspace.up_f32,
        inter,
        dim,
    )
    .map_err(|error| format!("shared expert (f32 device): up launch: {error}"))?;
    let mut gate_bytes = vec![0u8; inter * 4];
    cuda.copy_dtoh(&mut gate_bytes, &workspace.gate_f32)
        .map_err(|error| format!("shared expert (f32 device): gate readback: {error}"))?;
    let mut up_bytes = vec![0u8; inter * 4];
    cuda.copy_dtoh(&mut up_bytes, &workspace.up_f32)
        .map_err(|error| format!("shared expert (f32 device): up readback: {error}"))?;
    let gate = f32_from_le_bytes(&gate_bytes)
        .map_err(|error| format!("shared expert (f32 device): gate decode: {error}"))?;
    let up = f32_from_le_bytes(&up_bytes)
        .map_err(|error| format!("shared expert (f32 device): up decode: {error}"))?;
    let mut swiglu = vec![0.0f32; inter];
    shared_swiglu_f32(config.swiglu_limit as f32, &gate, &up, &mut swiglu)?;
    cuda.copy_htod(&workspace.swiglu_f32, &f32_le_bytes(&swiglu))
        .map_err(|error| format!("shared expert (f32 device): swiglu upload: {error}"))?;
    gemv(
        workspace.swiglu_f32,
        operands.w2,
        operands.s2,
        workspace.out_f32,
        dim,
        inter,
    )
    .map_err(|error| format!("shared expert (f32 device): down launch: {error}"))?;
    let mut out_bytes = vec![0u8; dim * 4];
    cuda.copy_dtoh(&mut out_bytes, &workspace.out_f32)
        .map_err(|error| format!("shared expert (f32 device): readback: {error}"))?;
    let row = f32_from_le_bytes(&out_bytes)
        .map_err(|error| format!("shared expert (f32 device): output decode: {error}"))?;
    if row.iter().any(|value| !value.is_finite()) {
        return Err("shared expert (f32 device): the f32 row is not finite".to_string());
    }
    // No bf16 round: the f32 store is the reference row.
    out.copy_from_slice(&row);
    Ok(())
}
/// Runs the layer-0 FFN epilogue from the live arena: read the released shared-expert bytes, add the
/// dense shared expert on top of the routed delta with the engine seam, then run the ffn hc_post.
///
/// # Errors
/// Refuses missing plan entries or arena slices, an unknown shared-expert dtype, a non-finite seam
/// output, and propagates the engine seams' own refusals.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_layer_ffn_epilogue(
    layer: usize,
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    request: &TokenStageRequest<'_>,
    seams: &Layer0FfnSeams,
    moe: &Layer0MoeRouted,
    residual: &AttentionResidualSeam,
    shared: &SharedExpertGather,
    shared_device: Option<&SharedExpertDevice>,
) -> Result<Layer0FfnEpilogue, String> {
    use sparkflash_engine::forward::mhc;
    use sparkflash_engine::forward::moe;
    use sparkflash_kernels::mxfp4_ref;

    let config = request
        .config
        .ok_or_else(|| "FFN epilogue: the loop was built without the model config".to_string())?;
    let dim = config.dim;
    let hc = config.hc_mult;
    // The plan shards the released shared expert TP-column/row; the reference seam is replicated, so
    // the shards were all-gathered once per layer at loop start (H5 stays byte-identical) and every
    // position of the layer borrows the same full views.
    let source = &shared.source;
    let mut scratch = moe::MoeScratch::for_config(config);
    let limit = 1usize << 40;
    // WS-D: name the kernels, the per-launch symbols and the accumulation mode the shared row came
    // from, so the row's identity evidence says which arithmetic produced it and whether the arm
    // stores bf16 or f32.
    //
    // The plan always carries one shared expert (ModelConfig: n_shared_experts == 1), so a missing
    // device staging is a REFUSAL, never a fallback: the host f32 seam is what this device arm
    // replaces, and silently running it again is the regression the row exists to remove.
    let Some(device) = shared_device else {
        return Err(format!(
            "FFN epilogue: layer {layer} has a shared expert in the plan but no resident device shared expert; refusing the host arithmetic fallback"
        ));
    };
    let (shared_arith, shared_kernel, shared_mode, launch_symbols): (
        &'static str,
        &'static str,
        &'static str,
        Vec<&'static str>,
    ) = if request.native_rounding.shared {
        (
            "native_fp8_block32_bf16_resident",
            "b12x_mxfp8_gemv_m1_sm121+swiglu_mxfp8_quant_sm121",
            sparkflash_kernels::mxfp8_gemv_launch::Mode::Fused.arithmetic_mode(),
            vec![
                sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
                sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
                "swiglu_mxfp8_quant_sm121",
                sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
            ],
        )
    } else {
        // The f32-reference arm: the same arithmetic the host seam ran, on the device, with the
        // f32 accumulator stored as f32. `arith` stays host_f32_reference because the row IS the
        // host reference's row bit for bit; the arithmetic_mode records how the device produced it.
        (
            "host_f32_reference",
            sparkflash_kernels::mxfp8_gemv_launch::KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT,
            sparkflash_kernels::mxfp8_gemv_launch::Mode::NonFusedF32ActivationF32Stores
                .arithmetic_mode(),
            vec![
                sparkflash_kernels::mxfp8_gemv_launch::KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT,
                sparkflash_kernels::mxfp8_gemv_launch::KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT,
                sparkflash_kernels::mxfp8_gemv_launch::KERNEL_NONFUSED_F32_ACTIVATION_F32_OUT,
            ],
        )
    };
    // The shared expert runs on the same pre-FFN row the device launched. The coded native arm
    // produces a BF16 row (the addend of the f32 MoE sum, model.py:893-904); the default arm keeps
    // the f32 reference row. It is computed once into its own row so the evidence can hash it in
    // either arm without a second shared-expert pass.
    let timer = BlockTimer::start(request.forward_perf.probe);
    let mut shared_only = vec![0.0f32; dim];
    if request.native_rounding.shared {
        run_shared_expert_device(
            cuda,
            request,
            layer,
            &seams.pre_ffn_hidden,
            device,
            &mut shared_only,
        )?;
    } else {
        run_shared_expert_f32_device(
            cuda,
            request,
            layer,
            &seams.pre_ffn_hidden,
            device,
            &mut shared_only,
        )?;
    }
    let shared_sha256 = evidence_f32_sha256(&shared_only);
    // C10: the L2 norms and the finite scans below are read only by the layer-chain record and the
    // run evidence document, so a default or --forward-lean run (evidence off) skips them.
    let shared_l2 = if layer_evidence_readbacks() {
        shared_only
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
    } else {
        0.0
    };
    // Layer-0 evidence only: the f32 reference row once per position. One extra shared-expert row
    // per position is the cheapest way to show the native term is non-vacuous on captured bytes
    // without a second pass on every layer.
    let (shared_f32_sha256, shared_native_vs_f32_rel_l2) =
        if request.native_rounding.shared && layer == 0 {
            let mut f32_only = vec![0.0f32; dim];
            moe::shared_expert_rows(
                config,
                &format!("layers.{layer}"),
                &seams.pre_ffn_hidden,
                source,
                limit,
                &mut scratch,
                &mut f32_only,
            )
            .map_err(|error| format!("FFN epilogue: f32 shared reference: {error}"))?;
            let difference = f32_only
                .iter()
                .zip(&shared_only)
                .map(|(f32_row, native_row)| {
                    let delta = native_row - f32_row;
                    delta * delta
                })
                .sum::<f32>()
                .sqrt();
            let norm = f32_only
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let relative = if norm == 0.0 { 0.0 } else { difference / norm };
            (Some(evidence_f32_sha256(&f32_only)), Some(relative))
        } else {
            (None, None)
        };
    probe::layer(cuda, layer, "ffn_shared", timer);
    let timer = BlockTimer::start(request.forward_perf.probe);
    let ffn_out = if let Some(full) = moe.native_full_acc.as_deref() {
        // M2 native arm: the shared expert's BF16 output is one addend of the f32 MoE sum, and
        // the MoE output is rounded to BF16 once (model.py:893-904). The residual is added later
        // by hc_post in f32 (model.py:962-966).
        let shared_bytes = bf16_le_bytes(&shared_only);
        let shared_bits: Vec<u16> = shared_bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        let native = mxfp4_ref::moe_combine_finalize(full, &shared_bits, &moe.router_ids, 1, dim);
        native
            .iter()
            .map(|bits| f32::from_bits(u32::from(*bits) << 16))
            .collect()
    } else {
        // The canonical reduce sums BF16 addends, so the routed delta is a BF16 row: widen it to
        // f32 before the shared expert adds into it.
        let mut widened = bf16_from_le_bytes(&moe.delta)?;
        if widened.len() != dim {
            return Err(format!(
                "FFN epilogue: the routed delta is {} values, one row is {dim}",
                widened.len()
            ));
        }
        for (accumulated, value) in widened.iter_mut().zip(&shared_only) {
            *accumulated += *value;
        }
        widened
    };
    let ffn_out_l2 = if layer_evidence_readbacks() {
        if ffn_out.iter().any(|value| !value.is_finite()) {
            return Err("FFN epilogue: the shared expert wrote a non-finite value".to_string());
        }
        ffn_out
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
    } else {
        0.0
    };
    probe::layer(cuda, layer, "ffn_combine", timer);
    let timer = BlockTimer::start(request.forward_perf.probe);
    let mut hc_post = vec![0.0f32; hc * dim];
    if request.native_rounding.mhc {
        mhc::hc_post_bf16(
            config,
            &ffn_out,
            &residual.out,
            &seams.ffn_post,
            &seams.ffn_comb,
            &mut hc_post,
        )
        .map_err(|error| format!("FFN epilogue: ffn hc_post: {error}"))?;
    } else {
        mhc::hc_post(
            config,
            &ffn_out,
            &residual.out,
            &seams.ffn_post,
            &seams.ffn_comb,
            &mut hc_post,
        )
        .map_err(|error| format!("FFN epilogue: ffn hc_post: {error}"))?;
    }
    let hc_post_l2 = if layer_evidence_readbacks() {
        if hc_post.iter().any(|value| !value.is_finite()) {
            return Err("FFN epilogue: the ffn hc_post wrote a non-finite value".to_string());
        }
        hc_post
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
    } else {
        0.0
    };
    probe::layer(cuda, layer, "ffn_hc_post", timer);
    Ok(Layer0FfnEpilogue {
        shared_reads: shared.local_reads,
        shared_gathers: shared.gather_calls,
        shared_gathered_bytes: shared.full_bytes,
        shared_arith,
        shared_kernel,
        shared_mode,
        launch_symbols,
        shared_sha256,
        shared_l2,
        shared_f32_sha256,
        shared_native_vs_f32_rel_l2,
        ffn_out_sha256: evidence_f32_sha256(&ffn_out),
        ffn_out_l2,
        ffn_out,
        hc_post_sha256: evidence_f32_sha256(&hc_post),
        hc_post_l2,
        hc_post,
    })
}

/// One planned B12x operand's (cols, inner) geometry, or a named refusal.
#[cfg(feature = "cuda")]
fn planned_b12x_geometry(
    plan: &sparkflash_device::resident::ResidentPlan,
    name: &str,
) -> Result<(usize, usize), String> {
    match plan.operand_layout(name) {
        Some(sparkflash_device::resident::OperandLayout::B12x { cols, inner, .. }) => {
            Ok((cols, inner))
        }
        Some(other) => Err(format!(
            "layer-0 query path: {name} is planned as {}; a B12x operand is required",
            other.as_str()
        )),
        None => Err(format!(
            "layer-0 query path: the rank plan does not carry {name}"
        )),
    }
}

/// Runs the layer-0 wq_a and wkv dense B12x projections from the B12x arena weights.
///
/// Loads a WS-D addition's cubin from the window's staged directory and fails closed unless the
/// staged bytes are the artifact the baseline pins.
///
/// The pinned kernels have a chain stage that verifies their cubin; an addition with its own symbol
/// has no chain stage of its own, so its call sites verify it rather than assuming it. `context`
/// prefixes the error so a refusal names the stage that would not launch.
#[cfg(feature = "cuda")]
fn load_pinned_addition(
    kernel_dir: &std::path::Path,
    kernel: &str,
    context: &str,
) -> Result<(std::sync::Arc<[u8]>, String), String> {
    let cubin_name = format!("{kernel}.cubin");
    let (cubin, cubin_sha256) = cached_cubin(&kernel_dir.join(&cubin_name))
        .map_err(|error| format!("{context}: read {cubin_name}: {error}"))?;
    let pinned = sparkflash_kernels::baseline::pin(kernel)
        .map_err(|error| format!("{context}: baseline: {error}"))?
        .ok_or_else(|| format!("{context}: the baseline does not pin {kernel}"))?;
    if cubin.len() != pinned.cubin_bytes || cubin_sha256 != pinned.cubin_sha256 {
        return Err(format!(
            "{context}: the staged {cubin_name} is {} bytes sha256 {cubin_sha256}, the baseline pins {} bytes sha256 {}",
            cubin.len(),
            pinned.cubin_bytes,
            pinned.cubin_sha256
        ));
    }
    Ok((cubin, cubin_sha256))
}

/// This is the first real attention launch of the four-rank loop: one BF16 activation row (the mHC
/// seam's normed attention input) against the plan-bound B12x weight/scale pairs, through the D37
/// bf16a dense kernel family. It reads the latents back so the evidence can pin what the next stage
/// (q_norm/kv_norm and the attention body) will consume. No weight is uploaded: the B12x bytes are
/// the arena's own slices from the bind, and the plan-dtype census has already refused a pair that
/// disagrees with the plan.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_layer_attention_projections(
    layer: usize,
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    kernels: &mut KernelCache,
    request: &TokenStageRequest<'_>,
    bind: &crate::tp4_resident::DeviceBind,
    normed: &[f32],
    position: usize,
    parity: &mut NativeDenseParity,
) -> Result<Layer0Projections, String> {
    use sparkflash_kernels::mxfp8_dense_launch;
    use sparkflash_kernels::mxfp8_gemv_launch;

    let wq_a_cols: usize = 1280;
    let wkv_cols: usize = 512;
    if normed.len() != HIDDEN {
        return Err(format!(
            "layer-0 projections: the normed attention input is {} values, one row is {HIDDEN}",
            normed.len()
        ));
    }
    // D39 rule 1: the W8A8 activation is a BF16 module output, narrowed to BF16 and then quantised
    // to E4M3 codes with one ue8m0 scale per 32 values before the native dense GEMM.
    let (activation_codes, activation_scales) =
        quantise_native_activation(normed, "layer-0 projections")?;
    // WS-D: the decode step is one output row, and the pinned 64x64 dense kernel gives each of its
    // 256 threads a 4x4 sub-tile, so at M=1 only lanes 0-15 have a row inside the tile. The decode
    // GEMV is the same per-output arithmetic with one warp per 32-output B12x micro-tile and one
    // output column per lane; it is a separate symbol and a separate cubin, so the pinned dense
    // kernel's bytes and digest are untouched, and it is loaded and verified by name here.
    let kernel_dir = request.kernel_cubin_dir.ok_or_else(|| {
        "layer-0 projections: the loop has no kernel cubin directory, so the dense B12x kernel cannot be loaded"
            .to_string()
    })?;
    // M3/C1: the GEMV module is loaded once per run by the loop's kernel cache and reused by every
    // layer; the pin is verified on that load, and the block only looks its digest up.
    let (kernel, cubin_sha256) = kernels.gemv(cuda, kernel_dir, "layer-0 projections")?;

    let pair_for = |name: &str| -> Result<
        (
            sparkflash_kernels::cuda_driver::Deviceptr,
            sparkflash_kernels::cuda_driver::Deviceptr,
        ),
        String,
    > {
        let pair = bind
            .attention
            .pointers
            .get(name)
            .ok_or_else(|| format!("layer-0 projections: the bind has no {name} pair"))?;
        if pair.0.0 == 0 || pair.1.0 == 0 {
            return Err(format!(
                "layer-0 projections: {name} bound a null pointer (weight {:#x}, scale {:#x}); refusing to launch",
                pair.0.0, pair.1.0
            ));
        }
        Ok(*pair)
    };
    let (wq_a_weight, wq_a_scales) = pair_for(&format!("layers.{layer}.attn.wq_a.weight"))?;
    let (wkv_weight, wkv_scales) = pair_for(&format!("layers.{layer}.attn.wkv.weight"))?;
    for (name, cols, inner) in [
        (
            &format!("layers.{layer}.attn.wq_a.weight"),
            wq_a_cols,
            HIDDEN,
        ),
        (&format!("layers.{layer}.attn.wkv.weight"), wkv_cols, HIDDEN),
    ] {
        let (planned_cols, planned_inner) = bind.plan_index.b12x_geometry(name)?;
        if (planned_cols, planned_inner) != (cols, inner) {
            return Err(format!(
                "layer-0 projections: {name} is planned {planned_cols}x{planned_inner}, the launch expects {cols}x{inner}"
            ));
        }
    }

    let mut allocated = 0usize;
    let a_codes = cuda
        .alloc_pooled(activation_codes.len(), &mut allocated)
        .map_err(|error| format!("layer-0 projections: activation code buffer: {error}"))?;
    let a_scales = cuda
        .alloc_pooled(activation_scales.len(), &mut allocated)
        .map_err(|error| format!("layer-0 projections: activation scale buffer: {error}"))?;
    let out_q = cuda
        .alloc_pooled(wq_a_cols * 2, &mut allocated)
        .map_err(|error| format!("layer-0 projections: wq_a latent buffer: {error}"))?;
    let out_kv = cuda
        .alloc_pooled(wkv_cols * 2, &mut allocated)
        .map_err(|error| format!("layer-0 projections: wkv latent buffer: {error}"))?;
    // SPARK_GPU_TRACE_ENTER dereferences the ring before it reads active_scope_mask, so a null
    // trace ring would fault inside the kernel. A zeroed ring keeps that dereference legal and its
    // mask zero, which is the trace-off state.
    let trace_ring = cuda
        .alloc_pooled(TRACE_RING_BYTES, &mut allocated)
        .map_err(|error| format!("layer-0 projections: trace ring buffer: {error}"))?;
    cuda.copy_htod(&trace_ring, &[0u8; TRACE_RING_BYTES])
        .map_err(|error| format!("layer-0 projections: trace ring zero fill: {error}"))?;

    let staged = (|| -> Result<Layer0Projections, String> {
        cuda.copy_htod(&a_codes, &activation_codes)
            .map_err(|error| format!("layer-0 projections: activation code htod: {error}"))?;
        cuda.copy_htod(&a_scales, &activation_scales)
            .map_err(|error| format!("layer-0 projections: activation scale htod: {error}"))?;
        let q = mxfp8_dense_launch::Mxfp8DenseParams::new(
            a_codes,
            a_scales,
            wq_a_weight,
            wq_a_scales,
            out_q,
            1,
            wq_a_cols as u32,
            HIDDEN as u32,
            trace_ring.0,
            layer as u32,
            0,
        );
        mxfp8_gemv_launch::launch(cuda, kernel, &q)?;
        let kv = mxfp8_dense_launch::Mxfp8DenseParams::new(
            a_codes,
            a_scales,
            wkv_weight,
            wkv_scales,
            out_kv,
            1,
            wkv_cols as u32,
            HIDDEN as u32,
            trace_ring.0,
            layer as u32,
            0,
        );
        mxfp8_gemv_launch::launch(cuda, kernel, &kv)?;

        let mut raw_q = vec![0u8; wq_a_cols * 2];
        cuda.copy_dtoh(&mut raw_q, &out_q)
            .map_err(|error| format!("layer-0 projections: wq_a latent dtoh: {error}"))?;
        let mut raw_kv = vec![0u8; wkv_cols * 2];
        cuda.copy_dtoh(&mut raw_kv, &out_kv)
            .map_err(|error| format!("layer-0 projections: wkv latent dtoh: {error}"))?;
        // C10: the latent rows themselves are arithmetic (the query/kv blocks widen them again);
        // only their L2 norms are evidence, so the two widening passes are taken in evidence mode
        // only.
        let (wq_a_l2, wkv_l2) = if layer_evidence_readbacks() {
            let values_q = bf16_from_le_bytes(&raw_q)?;
            let values_kv = bf16_from_le_bytes(&raw_kv)?;
            let l2 = |values: &[f32]| -> f32 {
                values.iter().map(|value| value * value).sum::<f32>().sqrt()
            };
            (l2(&values_q), l2(&values_kv))
        } else {
            (0.0, 0.0)
        };
        Ok(Layer0Projections {
            cubin_sha256,
            activation_codes_sha256: evidence_sha256(&activation_codes),
            activation_scales_sha256: evidence_sha256(&activation_scales),
            activation_code_bytes: activation_codes.len(),
            activation_scale_bytes: activation_scales.len(),
            wq_a_latent_sha256: evidence_sha256(&raw_q),
            wkv_latent_sha256: evidence_sha256(&raw_kv),
            wq_a_l2,
            wkv_l2,
            wq_a_latent_bytes: raw_q,
            wkv_latent_bytes: raw_kv,
            launches: 2,
        })
    })();

    cuda.release_pooled(&[&a_codes, &a_scales, &out_q, &out_kv, &trace_ring]);
    let projections = staged?;
    // wq_a and wkv are replicated at TP4: the parity re-launch on these bytes and the arena bind is
    // the literal loop-vs-chain comparison the diagnostic exists for.
    parity.capture(
        cuda,
        request,
        layer,
        position,
        "layer0-attention-projections",
        "wq_a",
        &format!("layers.{layer}.attn.wq_a.weight"),
        (wq_a_cols, HIDDEN),
        normed,
        &activation_codes,
        &activation_scales,
        wq_a_weight,
        wq_a_scales,
        &projections.wq_a_latent_bytes,
    );
    parity.capture(
        cuda,
        request,
        layer,
        position,
        "layer0-attention-projections",
        "wkv",
        &format!("layers.{layer}.attn.wkv.weight"),
        (wkv_cols, HIDDEN),
        normed,
        &activation_codes,
        &activation_scales,
        wkv_weight,
        wkv_scales,
        &projections.wkv_latent_bytes,
    );
    Ok(projections)
}

/// One head position: the collapsed stream, the certified head tail's token and its evidence.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct HeadPosition {
    /// The position the head ran at.
    pub position: usize,
    /// The winning token id (lowest global id on ties).
    pub token: u32,
    /// The winning pre-softmax logit.
    pub value: f64,
    /// Whether the winning entry survived every rank's screen (a claimable token).
    pub claimable: bool,
    /// The full-vocabulary logits after the exact f32 gather.
    pub logits: Vec<f32>,
    /// sha256 of the full-vocabulary f32-le vector.
    pub logits_sha256: String,
    /// The head record the evidence nests.
    pub head_record: serde_json::Value,
    /// The wall time of this head position's whole layer pass plus head.
    pub wall_ms: u64,
}

/// The head record's `input.file`: the per-position seam file the default path writes, or the
/// last-position alias under --forward-lean, where the per-position file is deliberately not
/// materialised. Evidence only; no arithmetic reads this path.
#[cfg(any(feature = "cuda", test))]
fn head_seam_input_file(lean: bool, seam_path: &Path, alias_path: &Path) -> String {
    if lean {
        alias_path.display().to_string()
    } else {
        seam_path.display().to_string()
    }
}

/// Runs the certified head tail on one position's collapsed stream (model.py:1268-1269 then the
/// four rank screens and the exact f32 gather).
///
/// # Errors
/// A hc_pre/norm non-finite value, a missing rank-plan dtype, a kernel/cubin failure, a gather
/// failure, or a side record that disagrees with its own certified count.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn run_head_position(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    request: &TokenStageRequest<'_>,
    tail_stage: &crate::tp4_tail::TailStage,
    transport: &TcpTransport,
    config: &sparkflash_engine::config::ModelConfig,
    final_norm: &[f32],
    stream: &[f32],
    tail_pre_mix: &[f32],
    position: usize,
) -> Result<HeadPosition, String> {
    use sparkflash_comm::transport::Transport;

    let started = std::time::Instant::now();
    let probe_on = request.forward_perf.probe;
    // Started here, not at the record site: a timer built at the end would snapshot the counters
    // after the head had already moved them and report an empty head_total delta.
    let head_total_timer = BlockTimer::start(probe_on);
    let config_hc = config.hc_mult;
    if tail_pre_mix.len() != config_hc {
        return Err(format!(
            "head position {position}: the collapse pre mix is {} values, expected {config_hc}",
            tail_pre_mix.len()
        ));
    }
    let mut collapsed = vec![0.0f32; HIDDEN];
    let timer = BlockTimer::start(probe_on);
    sparkflash_engine::forward::mhc::hc_pre(config, tail_pre_mix, stream, &mut collapsed)
        .map_err(|error| format!("head position {position}: hc_pre: {error}"))?;
    probe::head(cuda, "hc_pre", timer);
    let timer = BlockTimer::start(probe_on);
    let mut final_normed = collapsed.clone();
    sparkflash_engine::linalg::rms_norm(
        &mut final_normed,
        final_norm,
        HIDDEN,
        config.norm_eps as f32,
    )
    .map_err(|error| format!("head position {position}: final norm: {error}"))?;
    for (name, values) in [("collapsed", &collapsed), ("final normed", &final_normed)] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "head position {position}: the {name} row has a non-finite value; refusing to present it"
            ));
        }
    }
    probe::head(cuda, "final_norm", timer);
    let timer = BlockTimer::start(probe_on);
    let seam_bytes = f32_le_bytes(&collapsed);
    let seam_sha256 = evidence_sha256(&seam_bytes);
    let seam_bf16 = bf16_le_bytes(&collapsed);
    let seam_bf16_sha256 = evidence_sha256(&seam_bf16);
    let total_positions = prompt_len_for(request.prompt_tokens, request.positions.max(1))
        .unwrap_or(0)
        .saturating_add(request.positions.max(1))
        .saturating_sub(1);
    let last_position = position + 1 == total_positions;
    let head_dir = if request.rank == 0 {
        request.evidence_dir.join("head-input")
    } else {
        request
            .evidence_dir
            .join(format!("head-input-r{}", request.rank))
    };
    fs::create_dir_all(&head_dir)
        .map_err(|error| format!("head seam dir {}: {error}", head_dir.display()))?;
    let seam_path = head_dir.join(format!("pre_head_hidden-pos{position}.f32"));
    let alias_path = head_dir.join("pre_head_hidden.f32");
    // Lean skips the per-position seam materialisation (shas stay in the evidence) and writes only
    // the last-position alias; default rows write every head position's file and the alias.
    if request.forward_perf.write_head_seam_file(last_position) {
        if !request.forward_perf.lean {
            fs::write(&seam_path, &seam_bytes)
                .map_err(|error| format!("write {}: {error}", seam_path.display()))?;
        }
        if last_position {
            // The collapse above is the reference's final hc_pre with the last layer's FFN pre mix.
            // The m27 harness reads this fixed name for the last head position.
            fs::write(&alias_path, &seam_bytes)
                .map_err(|error| format!("write {}: {error}", alias_path.display()))?;
        }
    }
    // The head record names the f32 file this position actually wrote: the per-position seam file
    // on the default path, the last-position alias under --forward-lean (lean writes no per-position
    // file, so naming it would be a path to a file that does not exist).
    let input_file = head_seam_input_file(request.forward_perf.lean, &seam_path, &alias_path);
    probe::head(cuda, "seam_sha_write", timer);
    let head_eps = config.norm_eps as f32;
    let head_layer = config.n_layers.saturating_sub(1) as u32;
    let timer = BlockTimer::start(probe_on);
    let raw_logits = tail_stage.screen(cuda, &seam_bf16, head_layer, position as u64)?;
    probe::head(cuda, "screen", timer);
    let timer = BlockTimer::start(probe_on);
    let tail_outputs = tail_stage.read_outputs(cuda)?;
    probe::head(cuda, "read_outputs", timer);
    let timer = BlockTimer::start(probe_on);
    let gathered = crate::tp4_tail::gather_logits(
        transport,
        &raw_logits,
        HEAD_GATHER_SEQ.wrapping_sub(2 * position as u64),
    )?;
    probe::head(cuda, "gather_logits", timer);
    let timer = BlockTimer::start(probe_on);
    let mut side_bytes = Vec::with_capacity(crate::tp4_tail::SIDE_BYTES);
    side_bytes.extend_from_slice(&tail_outputs.certified);
    side_bytes.extend_from_slice(&tail_outputs.certified_count.to_le_bytes());
    for value in &tail_outputs.block_max {
        side_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &tail_outputs.block_argmax {
        side_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let side_words: Vec<u16> = side_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    let mut gathered_side = vec![0u16; request.world * side_words.len()];
    transport
        .all_gather_bf16(
            &side_words,
            &mut gathered_side,
            HEAD_SIDE_GATHER_SEQ.wrapping_sub(position as u64),
        )
        .map_err(|error| format!("head position {position}: side all-gather: {error}"))?;
    let mut rank_sides = Vec::with_capacity(request.world);
    for rank in 0..request.world {
        let start = rank * side_words.len();
        let bytes: Vec<u8> = gathered_side[start..start + side_words.len()]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        rank_sides.push(parse_head_side(&bytes)?);
    }
    if rank_sides.len() != request.world {
        return Err(format!(
            "head position {position}: {} rank side records gathered, the world is {}",
            rank_sides.len(),
            request.world
        ));
    }
    probe::head(cuda, "side_gather", timer);
    let timer = BlockTimer::start(probe_on);
    let winner_rank = gathered.argmax as usize / crate::tp4_tail::VOCAB_PER_RANK;
    let winner_index = gathered.argmax as usize % crate::tp4_tail::VOCAB_PER_RANK;
    let winner_certified = rank_sides
        .get(winner_rank)
        .and_then(|side| side.certified.get(winner_index))
        .copied()
        .ok_or_else(|| {
            format!(
                "head position {position}: argmax {} is rank {winner_rank} entry {winner_index}, outside the gathered side records",
                gathered.argmax
            )
        })?
        != 0;
    let claimable = !winner_certified;
    probe::head(cuda, "winner_check", timer);
    let timer = BlockTimer::start(probe_on);
    let normed_sha256 = evidence_sha256(&tail_outputs.normed_bf16);
    let gathered_sha256 = evidence_f32_sha256(&gathered.full);
    let mut head_ranks = Vec::with_capacity(request.world);
    for (rank, side) in rank_sides.iter().enumerate() {
        let base = rank * crate::tp4_tail::VOCAB_PER_RANK;
        let rank_logits = &gathered.full[base..base + crate::tp4_tail::VOCAB_PER_RANK];
        head_ranks.push(serde_json::json!({
            "rank": rank,
            "logits_sha256": evidence_f32_sha256(rank_logits),
            "logits_f32_values": rank_logits.len(),
            "certified_count": side.certified_count,
            "certified_sha256": evidence_sha256(&side.certified),
            "block_max_sha256": evidence_f32_sha256(&side.block_max),
            "block_argmax_sha256": evidence_sha256(&u32_le_bytes(&side.block_argmax)),
        }));
    }
    probe::head(cuda, "digests", timer);
    let timer = BlockTimer::start(probe_on);
    let head_record = serde_json::json!({
        "schema": "sparkflash-head-tail-evidence/1",
        "stage": "head-tail",
        "position": position,
        "rank": request.rank,
        "input": {
            "sha256": seam_sha256,
            "bytes": seam_bytes.len(),
            "bf16_codes_sha256": seam_bf16_sha256,
            "bf16_codes_bytes": seam_bf16.len(),
            "file": input_file,
        },
        "norm": {
            "kernel": "rmsnorm_sm121",
            "eps": head_eps,
            "normed_bf16_sha256": normed_sha256,
            "normed_bytes": tail_outputs.normed_bf16.len(),
        },
        "head": {
            "kernel": "cert_lm_head_sm121",
            "grid": [crate::tp4_tail::CHUNKS_PER_RANK, 1, 1],
            "block": [128, 1, 1],
            "ranks": head_ranks,
        },
        "gather": {
            "route": "exact-f32-word-gather",
            "gathered_logits_f32_values": gathered.full.len(),
            "gathered_logits_sha256": gathered_sha256,
        },
        "argmax": {
            "token": gathered.argmax,
            "value": gathered.top1,
            "certified": winner_certified,
            "claimable": claimable,
            "twin_token": serde_json::Value::Null,
            "twin_value": serde_json::Value::Null,
        },
        "reduce": {
            "reduce_mode": "dropped-single-stream",
            "chain_head_launches_per_position": 5,
            "single_stream_head_launches_per_position": 4,
            "reduce_launches_dropped": 1,
            "batched_reduce": {"fixed_rows": 6, "callable_with_one_row": false},
        },
        "launches": {
            "norm": request.world,
            "head": request.world,
            "reduce": 0,
        },
        "note": "seam = the position's post-layer-39 hc stream collapsed with hc_pre and the reference's cross-layer pre mix (the last layer's FFN pre mix; the identity one-hot only before layer 0), 5120 f32 pre-norm; the loop claims the token only when the four ranks agree and the winning entry survived every rank's screen",
    });
    probe::head(cuda, "head_record", timer);
    probe::head(cuda, "head_total", head_total_timer);
    Ok(HeadPosition {
        position,
        token: gathered.argmax,
        value: f64::from(gathered.top1),
        claimable,
        logits: gathered.full.clone(),
        logits_sha256: gathered_sha256,
        head_record,
        wall_ms: started.elapsed().as_millis() as u64,
    })
}

/// Runs the four-rank forward loop. This is the only GPU entry point of the module.
///
/// `bound` is the live arena and its bind; without the cuda feature, or before the loop body is
/// wired, the function fails with a named error instead of pretending to have run.
#[cfg(not(feature = "cuda"))]
fn run_forward_loop(
    _request: &TokenStageRequest<'_>,
    _plan: &ForwardPlan,
    _bound: Option<(
        &sparkflash_device::arena::ResidentArena,
        &crate::tp4_resident::DeviceBind,
    )>,
    _transport: Option<&TcpTransport>,
) -> Result<ForwardRun, String> {
    Err(
        "built without the cuda feature: the four-rank forward loop is not in this binary"
            .to_string(),
    )
}

/// The per-layer progress record: what a mid-run failure (or the window collector while the loop
/// is still running) can read without waiting for the end-of-loop evidence. It is rewritten after
/// every layer with the per-layer wall time and the accumulated stage digests.
#[must_use]
pub fn forward_progress(
    status: &str,
    request: &TokenStageRequest<'_>,
    n_layers: usize,
    entries: &[serde_json::Value],
    total: std::time::Duration,
) -> serde_json::Value {
    serde_json::json!({
        "schema": "sparkflash-tp4-forward-progress/1",
        "rank": request.rank,
        "world": request.world,
        "n_layers": n_layers,
        "status": status,
        "layers_completed": entries.len(),
        "last_completed_layer": entries
            .last()
            .and_then(|entry| entry.get("layer"))
            .and_then(serde_json::Value::as_u64),
        "total_wall_ms": u64::try_from(total.as_millis()).unwrap_or(u64::MAX),
        "entries": entries,
    })
}

/// Writes (and replaces) the per-layer progress record, returning the serialised byte count so a
/// probe run can report the write's bytes and host time.
///
/// # Errors
/// The filesystem write failure, with the path named.
pub fn write_forward_progress(
    path: &std::path::Path,
    progress: &serde_json::Value,
) -> Result<usize, String> {
    let json = serde_json::to_vec_pretty(progress)
        .map_err(|error| format!("forward progress json: {error}"))?;
    std::fs::write(path, &json).map_err(|error| format!("write {}: {error}", path.display()))?;
    Ok(json.len())
}

/// The default-off host timing probe behind `--forward-perf-probe` (M2 perf).
///
/// The probe is a diagnostic recorder, never an arithmetic path: every block is timed with a host
/// [BlockTimer] and one `Cuda::synchronize` at its boundary, so a recorded block is device-inclusive
/// and the synchronise cost is probe overhead (measured by a no-probe twin row, never subtracted
/// here). The samples live in a thread-local because the four-rank loop is single-threaded; when
/// the switch is off the hot gate is one relaxed atomic load and nothing is recorded. The writer
/// emits a SEPARATE `tp4-forward-perf-r{rank}.json`; the main evidence documents are untouched.
/// The default-off host timing probe behind `--forward-perf-probe` (M2 perf).
///
/// The probe is a diagnostic recorder, never an arithmetic path: every block is timed with a host
/// [BlockTimer] and one `Cuda::synchronize` at its boundary, so a recorded block is
/// device-inclusive. The samples live in a thread-local because the four-rank loop is
/// single-threaded; when the switch is off the hot gate is one relaxed atomic load and nothing is
/// recorded. The writer emits a SEPARATE `tp4-forward-perf-r{rank}.json`; the main evidence
/// documents are untouched.
///
/// The second kit adds the half of the table the P0 gate was missing. A host wall alone cannot say
/// whether a block was device-bound or host-bound, so every block also reports:
///
/// * `kernel_ns`: the summed `cuEventElapsedTime` of the cuLaunchKernel calls the
///   block issued, read from the driver's probe meter
///   (`sparkflash_kernels::cuda_driver::probe_meter_*`), which the probe arms at run scope and
///   disarms at exit;
/// * `overhead_ms`: `wall_ms - kernel_ms`, the host code, the per-launch drain,
///   the staging copies and the launch cost;
/// * `launches`, `syncs`, `sync_ms`, `h2d_*`, `d2h_*`, `dtod_*`,
///   `alloc_count`, `free_count` and `module_load_count`: the driver entry points the
///   block called, counted only while the probe is armed, so the staging and host-sync buckets can
///   be priced separately from launch overhead;
/// * `arena_reads` and `sha256`: the host counters the block moved, per block and
///   per layer.
///
/// The probe's own boundary drain is excluded from a block's counters and priced once in
/// `run.probe_overhead`, so a block's numbers describe the code path and not the instrument.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) mod probe {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    use serde_json::{Map, Value, json};
    #[cfg(feature = "cuda")]
    use sparkflash_kernels::cuda_driver::probe_meter_deferred;
    use sparkflash_kernels::cuda_driver::{
        ProbeMeterReading, probe_meter_arm, probe_meter_arm_deferred, probe_meter_read,
    };

    use super::{BlockTimer, ForwardPerf, ProbeMode};

    /// The scope a sample belongs to; the writer maps each to its evidence slot.
    #[derive(Clone, Copy)]
    enum Scope {
        Run,
        Position,
        Layer(usize),
        Head,
    }

    impl Scope {
        /// The scope name the run-level block summary keys on.
        fn name(self) -> &'static str {
            match self {
                Scope::Run => "run",
                Scope::Position => "position",
                Scope::Layer(_) => "layer",
                Scope::Head => "head",
            }
        }
    }

    /// One host worker's counter set: how many calls, how many bytes, how long.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct HostCounters {
        count: u64,
        bytes: u64,
        ns: u64,
    }

    impl HostCounters {
        fn since(self, earlier: Self) -> Self {
            Self {
                count: self.count.saturating_sub(earlier.count),
                bytes: self.bytes.saturating_sub(earlier.bytes),
                ns: self.ns.saturating_sub(earlier.ns),
            }
        }

        fn plus(self, other: Self) -> Self {
            Self {
                count: self.count.saturating_add(other.count),
                bytes: self.bytes.saturating_add(other.bytes),
                ns: self.ns.saturating_add(other.ns),
            }
        }
    }

    /// One reading of every counter a block can move: the driver's launch/staging meter plus the two
    /// host work counters the probe keeps itself.
    ///
    /// A [BlockTimer] carries the reading it took when the block started, and the block's row is the
    /// difference between that and the reading taken at the boundary. One struct of plain integers
    /// is what makes a nested block honest: `layer_total` differences its own snapshots, so it
    /// never double-counts the blocks inside it.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub(crate) struct Snapshot {
        meter: ProbeMeterReading,
        arena: HostCounters,
        sha: HostCounters,
        /// How many blocks the probe had already closed at this reading. The difference over a
        /// block is the number of nested blocks inside it, whose boundary drains are the probe's
        /// own cost and must not land in that block's sync counters.
        probe_blocks: u64,
        /// The summed probe drain time at this reading, for the same subtraction.
        probe_sync_ns: u64,
    }

    impl Snapshot {
        fn since(self, earlier: Self) -> Self {
            Self {
                meter: self.meter.since(earlier.meter),
                arena: self.arena.since(earlier.arena),
                sha: self.sha.since(earlier.sha),
                probe_blocks: self.probe_blocks.saturating_sub(earlier.probe_blocks),
                probe_sync_ns: self.probe_sync_ns.saturating_sub(earlier.probe_sync_ns),
            }
        }

        fn plus(self, other: Self) -> Self {
            Self {
                meter: self.meter.plus(other.meter),
                arena: self.arena.plus(other.arena),
                sha: self.sha.plus(other.sha),
                probe_blocks: self.probe_blocks.saturating_add(other.probe_blocks),
                probe_sync_ns: self.probe_sync_ns.saturating_add(other.probe_sync_ns),
            }
        }
    }

    struct Sample {
        scope: Scope,
        position: usize,
        name: &'static str,
        wall_ms: f64,
        delta: Snapshot,
        /// The meter's cumulative launch count at the block's start and end: on the host-only path
        /// this is the slice of the deferred harvest the block owns.
        launches: (u64, u64),
    }

    #[derive(Default)]
    struct State {
        samples: Vec<Sample>,
        /// One kernel-ns figure per launch, in launch order, filled by [finish] on the host-only
        /// path and spent by the writer; None on the drained path.
        deferred_ns: Option<Vec<u64>>,
        /// The host wall of the run's final synchronize, priced as probe overhead in the same file.
        final_sync_ms: f64,
        position: usize,
        heads: BTreeMap<usize, bool>,
        progress: Vec<(usize, u64, f64)>,
        arena: HostCounters,
        sha: HostCounters,
        probe_blocks: u64,
        probe_sync_ns: u64,
    }

    thread_local! {
        static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
    }

    /// The cheap gate the hot helpers read: false is the default path.
    static ACTIVE: AtomicBool = AtomicBool::new(false);

    /// Arms or clears the probe for this thread, and arms the driver meter with it; called once at
    /// run scope and once at exit. Clearing the meter here is what makes a second token stage in one
    /// process report its own work instead of the first stage's plus its own.
    pub(crate) fn begin_run(perf: ForwardPerf) {
        let enabled = perf.probe;
        ACTIVE.store(enabled, Ordering::Relaxed);
        let deferred = enabled && perf.probe_nosync;
        // One meter at a time: the drained one keeps its per-launch bracket, the deferred one queues
        // and harvests. Both feed the same counters, so the census is comparable across modes.
        probe_meter_arm_deferred(deferred);
        probe_meter_arm(enabled && !deferred);
        STATE.with(|state| {
            *state.borrow_mut() = if enabled {
                Some(State::default())
            } else {
                None
            };
        });
    }

    /// Whether a probe run is installed on this thread.
    pub(crate) fn installed() -> bool {
        ACTIVE.load(Ordering::Relaxed)
    }

    /// Clears the probe (tests, and a second run in one process).
    pub(crate) fn reset() {
        begin_run(ForwardPerf::OFF);
    }

    /// Whether this run's blocks measure the host-side wall only (the no-sync probe mode).
    #[cfg(feature = "cuda")]
    pub(crate) fn deferred() -> bool {
        installed() && probe_meter_deferred()
    }

    /// Closes a host-only-wall run: one synchronize (the probe's only one), then the deferred event
    /// harvest, whose per-launch kernel times the writer attributes to the blocks that issued them.
    ///
    /// A no-op on the drained path, where every block already closed itself.
    #[cfg(feature = "cuda")]
    pub(crate) fn finish(cuda: &sparkflash_kernels::cuda_driver::Cuda) {
        if !deferred() {
            return;
        }
        let started = Instant::now();
        if let Err(error) = cuda.synchronize() {
            eprintln!("forward perf probe: final cuCtxSynchronize: {error}");
        }
        let final_sync_ms = started.elapsed().as_secs_f64() * 1000.0;
        let harvest = cuda.probe_meter_harvest();
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                state.final_sync_ms = final_sync_ms;
                match harvest {
                    Ok(per_launch) => state.deferred_ns = Some(per_launch),
                    Err(error) => eprintln!("forward perf probe: deferred harvest: {error}"),
                }
            }
        });
    }

    /// The counter reading now, when the probe is installed: the driver meter plus the two host
    /// counters this module keeps.
    fn current() -> Option<Snapshot> {
        let host = STATE.with(|state| {
            let borrow = state.borrow();
            let state = borrow.as_ref()?;
            Some((
                state.arena,
                state.sha,
                state.probe_blocks,
                state.probe_sync_ns,
            ))
        })?;
        Some(Snapshot {
            meter: probe_meter_read(),
            arena: host.0,
            sha: host.1,
            probe_blocks: host.2,
            probe_sync_ns: host.3,
        })
    }

    /// The reading a block starts from; None on the default path.
    pub(crate) fn snapshot(enabled: bool) -> Option<Snapshot> {
        if !enabled {
            return None;
        }
        current()
    }

    /// Starts a counter clock when the probe is installed; None in the default path.
    pub(crate) fn counter_timer() -> Option<Instant> {
        installed().then(Instant::now)
    }

    fn elapsed_ns(timer: Option<Instant>) -> Option<u64> {
        timer.map(|start| start.elapsed().as_nanos() as u64)
    }

    /// Adds one `read_arena_bytes` call to the run and layer counters.
    pub(crate) fn count_arena_read(bytes: usize, timer: Option<Instant>) {
        let Some(ns) = elapsed_ns(timer) else {
            return;
        };
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                state.arena.count += 1;
                state.arena.bytes += bytes as u64;
                state.arena.ns += ns;
            }
        });
    }

    /// Adds one `sha256_hex` call to the run and layer counters.
    pub(crate) fn count_sha256(bytes: usize, timer: Option<Instant>) {
        let Some(ns) = elapsed_ns(timer) else {
            return;
        };
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                state.sha.count += 1;
                state.sha.bytes += bytes as u64;
                state.sha.ns += ns;
            }
        });
    }

    fn push(scope: Scope, name: &'static str, wall_ms: f64, delta: Snapshot) {
        push_with_launches(scope, name, wall_ms, delta, (0, 0));
    }

    fn push_with_launches(
        scope: Scope,
        name: &'static str,
        wall_ms: f64,
        delta: Snapshot,
        launches: (u64, u64),
    ) {
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                state.samples.push(Sample {
                    scope,
                    position: state.position,
                    name,
                    wall_ms,
                    delta,
                    launches,
                });
            }
        });
    }

    /// Marks the current position (and whether it materialises head blocks).
    pub(crate) fn begin_position(position: usize, head: bool) {
        if !installed() {
            return;
        }
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                state.position = position;
                state.heads.insert(position, head);
            }
        });
    }

    /// Records one progress write (serialised bytes, host ms) against the current position.
    pub(crate) fn progress(bytes: usize, timer: BlockTimer) {
        let Some(ms) = timer.elapsed_ms() else {
            return;
        };
        if !installed() {
            return;
        }
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                let position = state.position;
                state.progress.push((position, bytes as u64, ms));
            }
        });
    }

    /// Closes one block: reads its host wall, drains the context, and differences the two counter
    /// readings into the block's row.
    ///
    /// The wall is read *before* the drain so the probe's own synchronise is not charged to the
    /// block. That does not change what the block measured: this driver already drains the context
    /// at every launch, so a block's wall is device-inclusive either way, and the drain is a safety
    /// net that the writer prices once in `run.probe_overhead`.
    #[cfg(feature = "cuda")]
    fn record(
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        scope: Scope,
        name: &'static str,
        timer: BlockTimer,
    ) {
        if !installed() {
            return;
        }
        let Some(start) = timer.snapshot() else {
            return;
        };
        let Some(wall_ms) = timer.elapsed_ms() else {
            return;
        };
        // Host-only wall: close the block with no drain. The launch census it moved against its
        // start is what the writer turns into kernel_ns once the run-end harvest has the figures.
        if deferred() {
            let Some(end) = current() else {
                return;
            };
            let delta = end.since(start);
            push_with_launches(
                scope,
                name,
                wall_ms,
                delta,
                (start.meter.launches, end.meter.launches),
            );
            return;
        }
        let drained = Instant::now();
        if let Err(error) = cuda.synchronize() {
            eprintln!("forward perf probe: cuCtxSynchronize: {error}");
        }
        let probe_sync_ns = drained.elapsed().as_nanos() as u64;
        let Some(end) = current() else {
            return;
        };
        let mut delta = end.since(start);
        // The boundary drains are the probe's own cost, not the block's: remove this block's own
        // synchronise and every drain its nested blocks made between the two readings, and price
        // them once as run-scope probe overhead instead. Without this a `layer_total` would report
        // the ~20 drains of the blocks inside it as its own host-sync time.
        let nested_blocks = delta.probe_blocks;
        let nested_ns = delta.probe_sync_ns;
        delta.meter.syncs = delta
            .meter
            .syncs
            .saturating_sub(nested_blocks.saturating_add(1));
        delta.meter.sync_ns = delta
            .meter
            .sync_ns
            .saturating_sub(nested_ns.saturating_add(probe_sync_ns));
        STATE.with(|state| {
            if let Some(state) = state.borrow_mut().as_mut() {
                state.probe_blocks += 1;
                state.probe_sync_ns += probe_sync_ns;
            }
        });
        push(scope, name, wall_ms, delta);
    }

    /// Records a run-scope block (one synchronise at its boundary).
    #[cfg(feature = "cuda")]
    pub(crate) fn run(
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        name: &'static str,
        timer: BlockTimer,
    ) {
        record(cuda, Scope::Run, name, timer);
    }

    /// Records a position-scope block.
    #[cfg(feature = "cuda")]
    pub(crate) fn position(
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        name: &'static str,
        timer: BlockTimer,
    ) {
        record(cuda, Scope::Position, name, timer);
    }

    /// Records a layer-scope block for `layer`.
    #[cfg(feature = "cuda")]
    pub(crate) fn layer(
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        layer: usize,
        name: &'static str,
        timer: BlockTimer,
    ) {
        record(cuda, Scope::Layer(layer), name, timer);
    }

    /// Records a head-scope block (grouped under the current position).
    #[cfg(feature = "cuda")]
    pub(crate) fn head(
        cuda: &sparkflash_kernels::cuda_driver::Cuda,
        name: &'static str,
        timer: BlockTimer,
    ) {
        record(cuda, Scope::Head, name, timer);
    }

    /// Rounds to the microsecond so a full-layer probe file stays as small as it can.
    fn round_ms(value: f64) -> f64 {
        (value * 1000.0).round() / 1000.0
    }

    fn ms_of(ns: u64) -> f64 {
        round_ms(ns as f64 / 1_000_000.0)
    }

    /// The `{count, bytes, ms}` triple the host counters and the copy counters share.
    fn trio_json(counters: HostCounters) -> Value {
        json!({
            "count": counters.count,
            "bytes": counters.bytes,
            "ms": ms_of(counters.ns),
        })
    }

    /// The per-layer counters the P0 gate asks for: what this layer's host seams read back and
    /// hashed, with the time they spent doing it.
    fn counters_json(delta: &Snapshot) -> Value {
        json!({
            "arena_reads": trio_json(delta.arena),
            "sha256": trio_json(delta.sha),
        })
    }

    /// The block-row field order: one block is a JSON array in this order, trimmed at its last
    /// non-zero field, with `wall_ms` always first.
    ///
    /// The names cost more than the values at this scale. A probe row writes 106,829 block rows per
    /// rank, so naming six keys on every one of them turns a 2.5 MB wall table into a 12 MB one;
    /// the names therefore live here once and the writer copies this array into the document's
    /// `legend`. The fields are the four buckets the P0 gate asks for: `kernel_ns` (device),
    /// `h2d/d2h/dtod` (staging), `syncs/sync_ms` (host-sync) and `wall_ms - kernel_ms`
    /// (`overhead_ms`: host code plus launch cost).
    const BLOCK_FIELDS: [&str; 18] = [
        "wall_ms",
        "kernel_ns",
        "overhead_ms",
        "launches",
        "syncs",
        "sync_ms",
        "h2d_count",
        "h2d_bytes",
        "h2d_ms",
        "d2h_count",
        "d2h_bytes",
        "d2h_ms",
        "dtod_count",
        "dtod_bytes",
        "dtod_ms",
        "alloc_count",
        "free_count",
        "module_load_count",
    ];

    /// The host-only-wall row appends two buckets the drained row does not have: the block's own
    /// host wait (its cuCtxSynchronize calls and device-to-host reads, i.e. the readback blocks) and
    /// the host code outside that wait. Both are milliseconds so the two modes compare directly.
    const HOST_ONLY_BLOCK_FIELDS: [&str; 2] = ["wait_ms", "host_ms"];

    /// The field names this mode's rows carry, in order; the document's legend spells them out.
    fn block_field_names(host_only: bool) -> Vec<&'static str> {
        let mut names = BLOCK_FIELDS.to_vec();
        if host_only {
            names.extend_from_slice(&HOST_ONLY_BLOCK_FIELDS);
        }
        names
    }

    /// The kernel time a host-only row's launch slice owns: the harvest is one figure per launch in
    /// launch order, so the block's figures are [start, end) exactly like its counter difference.
    fn deferred_kernel_ns(per_launch: &[u64], launches: (u64, u64)) -> u64 {
        let start = usize::try_from(launches.0).unwrap_or(usize::MAX);
        let end = usize::try_from(launches.1).unwrap_or(usize::MAX);
        per_launch
            .get(start..end)
            .map_or(0, |slice| slice.iter().sum::<u64>())
    }

    fn is_zero(value: &Value) -> bool {
        matches!(value, Value::Number(number) if number.as_f64() == Some(0.0))
    }

    /// Every field of a block row, in [BLOCK_FIELDS] order, before trimming.
    ///
    /// The kernel split is written whenever the block launched anything, even if the events round
    /// to zero, so a reader can tell `no device work` from `device work too short to measure`.
    fn block_fields(wall_ms: f64, delta: &Snapshot, host_only: bool) -> Vec<Value> {
        let kernel_ns = delta.meter.kernel_ns;
        let launched = delta.meter.launches > 0;
        let mut fields = vec![
            json!(round_ms(wall_ms)),
            json!(kernel_ns),
            json!(if launched {
                round_ms(wall_ms - kernel_ns as f64 / 1_000_000.0)
            } else {
                0.0
            }),
            json!(delta.meter.launches),
            json!(delta.meter.syncs),
            json!(ms_of(delta.meter.sync_ns)),
            json!(delta.meter.h2d_count),
            json!(delta.meter.h2d_bytes),
            json!(ms_of(delta.meter.h2d_ns)),
            json!(delta.meter.d2h_count),
            json!(delta.meter.d2h_bytes),
            json!(ms_of(delta.meter.d2h_ns)),
            json!(delta.meter.dtod_count),
            json!(delta.meter.dtod_bytes),
            json!(ms_of(delta.meter.dtod_ns)),
            json!(delta.meter.alloc_count),
            json!(delta.meter.free_count),
            json!(delta.meter.module_load_count),
        ];
        if host_only {
            let wait_ns = delta.meter.sync_ns.saturating_add(delta.meter.d2h_ns);
            fields.push(json!(ms_of(wait_ns)));
            fields.push(json!(round_ms(
                (wall_ms - wait_ns as f64 / 1_000_000.0).max(0.0)
            )));
        }
        fields
    }

    fn trim_trailing_zeros(fields: &mut Vec<Value>) {
        while fields.len() > 1 && fields.last().is_some_and(is_zero) {
            fields.pop();
        }
    }

    /// One block row, trimmed: a block that ran host code only is `[0.202]`, which is what keeps a
    /// full per-layer per-position probe file near the size of the first kit's wall-only table.
    fn block_json(wall_ms: f64, delta: &Snapshot, host_only: bool) -> Value {
        let mut fields = block_fields(wall_ms, delta, host_only);
        trim_trailing_zeros(&mut fields);
        Value::Array(fields)
    }

    /// The same row with its names spelled out, for the run-level table a reader (or the P0 gate
    /// table) opens first: 40 rows can afford 40 keys, 106,829 cannot.
    fn summary_json(
        scope: &str,
        name: &str,
        count: u64,
        wall_ms: f64,
        delta: &Snapshot,
        host_only: bool,
    ) -> Value {
        let mut fields = block_fields(wall_ms, delta, host_only);
        trim_trailing_zeros(&mut fields);
        let names = block_field_names(host_only);
        let mut object = Map::new();
        object.insert("scope".to_string(), json!(scope));
        object.insert("name".to_string(), json!(name));
        object.insert("count".to_string(), json!(count));
        for (index, value) in fields.into_iter().enumerate() {
            object.insert(names[index].to_string(), value);
        }
        // The named summary keeps the four buckets adjacent whatever the trimming did.
        for field in ["kernel_ns", "overhead_ms", "launches", "syncs", "sync_ms"] {
            if !object.contains_key(field) {
                object.insert(field.to_string(), json!(0));
            }
        }
        Value::Object(object)
    }

    /// One block name's row inside its scope: the summed host wall and the summed counters.
    #[derive(Clone, Copy, Default)]
    struct Row {
        wall_ms: f64,
        delta: Snapshot,
    }

    impl Row {
        fn add(&mut self, wall_ms: f64, delta: &Snapshot) {
            self.wall_ms += wall_ms;
            self.delta = self.delta.plus(*delta);
        }
    }

    #[derive(Default)]
    struct LayerSamples {
        blocks: BTreeMap<&'static str, Row>,
        total_ms: f64,
        total: Snapshot,
    }

    #[derive(Default)]
    struct PositionSamples {
        blocks: BTreeMap<&'static str, Row>,
        head_blocks: BTreeMap<&'static str, Row>,
        layers: BTreeMap<usize, LayerSamples>,
        progress_count: u64,
        progress_bytes: u64,
        progress_ms: f64,
    }

    /// Writes the separate perf file when this run asked for the probe. `false` means it did not:
    /// no file is created, so the default path's evidence directory is byte-for-byte unchanged.
    ///
    /// # Errors
    /// A serialisation failure or the filesystem write, with the path named.
    pub(crate) fn write(
        path: &Path,
        rank: usize,
        world: usize,
        perf: ForwardPerf,
    ) -> Result<bool, String> {
        if !perf.probe || !installed() {
            return Ok(false);
        }
        let bytes = STATE.with(|state| -> Result<Vec<u8>, String> {
            let borrow = state.borrow();
            let state = borrow.as_ref().ok_or_else(|| {
                "forward perf probe: the writer ran without an installed probe".to_string()
            })?;
            let mut run: BTreeMap<&'static str, Row> = BTreeMap::new();
            let mut positions: BTreeMap<usize, PositionSamples> = BTreeMap::new();
            let mut summary: BTreeMap<(&'static str, &'static str), (u64, Row)> = BTreeMap::new();
            let host_only = perf.probe_nosync;
            let deferred_ns = state.deferred_ns.as_deref();
            for sample in &state.samples {
                // On the host-only path the block's kernel time is not in its counter difference
                // (nothing resolved the events at the boundary); it is the harvest slice its launch
                // census names. The drained path keeps the meter's own difference unchanged.
                let mut adjusted = sample.delta;
                if let Some(per_launch) = deferred_ns {
                    adjusted.meter.kernel_ns = deferred_kernel_ns(per_launch, sample.launches);
                }
                let delta = &adjusted;
                let slot = summary
                    .entry((sample.scope.name(), sample.name))
                    .or_default();
                slot.0 += 1;
                slot.1.add(sample.wall_ms, delta);
                if matches!(sample.scope, Scope::Run) {
                    run.entry(sample.name)
                        .or_default()
                        .add(sample.wall_ms, delta);
                    continue;
                }
                let entry = positions.entry(sample.position).or_default();
                match sample.scope {
                    Scope::Run => unreachable!("the run scope returned above"),
                    Scope::Position => {
                        entry
                            .blocks
                            .entry(sample.name)
                            .or_default()
                            .add(sample.wall_ms, delta);
                    }
                    Scope::Head => {
                        entry
                            .head_blocks
                            .entry(sample.name)
                            .or_default()
                            .add(sample.wall_ms, delta);
                    }
                    Scope::Layer(layer) => {
                        let layer_entry = entry.layers.entry(layer).or_default();
                        if sample.name == "layer_total" {
                            layer_entry.total_ms += sample.wall_ms;
                            layer_entry.total = layer_entry.total.plus(*delta);
                        } else {
                            layer_entry
                                .blocks
                                .entry(sample.name)
                                .or_default()
                                .add(sample.wall_ms, delta);
                        }
                    }
                }
            }
            for (position, bytes, ms) in &state.progress {
                let entry = positions.entry(*position).or_default();
                entry.progress_count += 1;
                entry.progress_bytes += bytes;
                entry.progress_ms += ms;
            }
            let mut run_object = Map::new();
            for (name, row) in &run {
                run_object.insert(
                    (*name).to_string(),
                    block_json(row.wall_ms, &row.delta, host_only),
                );
            }
            run_object.insert(
                "counters".to_string(),
                json!({
                    "arena_reads": trio_json(state.arena),
                    "sha256": trio_json(state.sha),
                }),
            );
            let probe_overhead = if host_only {
                json!({
                    "blocks": 0,
                    "syncs": 1,
                    "sync_ms": round_ms(state.final_sync_ms),
                    "note": "host-only wall: no cuCtxSynchronize at a block boundary; the single destroy-and-harvest synchronize at run end is priced here instead, so a block's wall is the host's and its kernel_ns is the device's",
                })
            } else {
                json!({
                    "blocks": state.probe_blocks,
                    "syncs": state.probe_blocks,
                    "sync_ms": ms_of(state.probe_sync_ns),
                    "note": "one cuCtxSynchronize per recorded block, outside the block's own wall; subtract this from a run total before comparing a probe row with its no-flag twin",
                })
            };
            run_object.insert("probe_overhead".to_string(), probe_overhead);
            let summary_rows: Vec<Value> = summary
                .into_iter()
                .map(|((scope, name), (count, row))| {
                    summary_json(scope, name, count, row.wall_ms, &row.delta, host_only)
                })
                .collect();
            run_object.insert("block_summary".to_string(), Value::Array(summary_rows));
            let position_json: Vec<Value> = positions
                .into_iter()
                .map(|(position, mut entry)| {
                    let total = entry.blocks.remove("position_total").unwrap_or_default();
                    let mut layer_sum_ms = 0.0;
                    let layers: Vec<Value> = entry
                        .layers
                        .into_iter()
                        .map(|(layer, layer_samples)| {
                            layer_sum_ms += layer_samples.total_ms;
                            let blocks: Map<String, Value> = layer_samples
                                .blocks
                                .iter()
                                .map(|(name, row)| {
                                    (
                                        (*name).to_string(),
                                        block_json(row.wall_ms, &row.delta, host_only),
                                    )
                                })
                                .collect();
                            let mut object = Map::new();
                            object.insert("layer".to_string(), json!(layer));
                            object.insert(
                                "total_ms".to_string(),
                                json!(round_ms(layer_samples.total_ms)),
                            );
                            if layer_samples.total.meter.kernel_ns > 0 {
                                object.insert(
                                    "kernel_ns".to_string(),
                                    json!(layer_samples.total.meter.kernel_ns),
                                );
                                object.insert(
                                    "overhead_ms".to_string(),
                                    json!(round_ms(
                                        layer_samples.total_ms
                                            - layer_samples.total.meter.kernel_ns as f64
                                                / 1_000_000.0
                                    )),
                                );
                            }
                            if layer_samples.total.meter.syncs > 0 {
                                object.insert(
                                    "syncs".to_string(),
                                    json!(layer_samples.total.meter.syncs),
                                );
                                object.insert(
                                    "sync_ms".to_string(),
                                    json!(ms_of(layer_samples.total.meter.sync_ns)),
                                );
                            }
                            object.insert(
                                "counters".to_string(),
                                counters_json(&layer_samples.total),
                            );
                            object.insert("blocks".to_string(), Value::Object(blocks));
                            Value::Object(object)
                        })
                        .collect();
                    let mut object = Map::new();
                    object.insert("position".to_string(), json!(position));
                    object.insert(
                        "head".to_string(),
                        json!(state.heads.get(&position).copied().unwrap_or(false)),
                    );
                    object.insert("total_ms".to_string(), json!(round_ms(total.wall_ms)));
                    object.insert("layer_sum_ms".to_string(), json!(round_ms(layer_sum_ms)));
                    if total.delta.meter.kernel_ns > 0 {
                        object.insert(
                            "kernel_ns".to_string(),
                            json!(total.delta.meter.kernel_ns),
                        );
                        object.insert(
                            "overhead_ms".to_string(),
                            json!(round_ms(
                                total.wall_ms - total.delta.meter.kernel_ns as f64 / 1_000_000.0
                            )),
                        );
                    }
                    object.insert("counters".to_string(), counters_json(&total.delta));
                    object.insert(
                        "blocks".to_string(),
                        Value::Object(
                            entry
                                .blocks
                                .iter()
                                .map(|(name, row)| {
                                    (
                                        (*name).to_string(),
                                        block_json(row.wall_ms, &row.delta, host_only),
                                    )
                                })
                                .collect(),
                        ),
                    );
                    if entry.progress_count > 0 {
                        object.insert(
                            "progress_writes".to_string(),
                            json!({
                                "count": entry.progress_count,
                                "bytes": entry.progress_bytes,
                                "ms": round_ms(entry.progress_ms),
                            }),
                        );
                    }
                    if !entry.head_blocks.is_empty() {
                        object.insert(
                            "head_blocks".to_string(),
                            Value::Object(
                                entry
                                    .head_blocks
                                    .iter()
                                    .map(|(name, row)| {
                                        (
                                            (*name).to_string(),
                                            block_json(row.wall_ms, &row.delta, host_only),
                                        )
                                    })
                                    .collect(),
                            ),
                        );
                    }
                    if !layers.is_empty() {
                        object.insert("layers".to_string(), Value::Array(layers));
                    }
                    Value::Object(object)
                })
                .collect();
            let mode = ProbeMode::for_run(perf, rank);
            let field_names = block_field_names(host_only);
            let mut legend = Map::new();
            legend.insert("block_fields".to_string(), json!(field_names));
            legend.insert(
                "units".to_string(),
                json!("wall_ms, overhead_ms, sync_ms, h2d_ms, d2h_ms, dtod_ms, wait_ms and host_ms are milliseconds; kernel_ns is nanoseconds; the rest are counts and bytes"),
            );
            if host_only {
                legend.insert(
                    "host_only_fields".to_string(),
                    json!("wait_ms is the block's own host wait (its cuCtxSynchronize calls plus its device-to-host reads, which is the readback blocks' wait); host_ms is wall_ms - wait_ms, the host code outside that wait"),
                );
            }
            legend.insert(
                "mode".to_string(),
                json!("mode.wall_clock names what wall_ms contains: device-inclusive rows read the wall before a boundary drain and pay one synchronize per block; host-only rows never drain at a boundary and take kernel_ns from one deferred event harvest at run end. A reader must not sum or average rows across the two"),
            );
            let wall_note = if host_only {
                "host Instant around the block, read with no context drain: the host's seam/launch path plus the block's own readback waits, without the device wait"
            } else {
                "host Instant around the block, read before the probe's boundary cuCtxSynchronize; the block is device-inclusive because this driver drains the context at every launch"
            };
            let kernel_note = if host_only {
                "summed cuEventElapsedTime of the block's own launches, one event pair per queued launch, resolved once after the run's final cuCtxSynchronize (deferred meter, --forward-perf-probe-nosync): kernel time without any host wait"
            } else {
                "summed cuEventElapsedTime of the cuLaunchKernel calls inside the block (driver probe meter, armed by --forward-perf-probe only)"
            };
            let overhead_note = if host_only {
                "wall_ms - kernel_ms: host code plus the block's own readback waits (wait_ms names that wait; host_ms is the host code outside it), with no per-launch drain"
            } else {
                "wall_ms - kernel_ms: host code, the per-launch cuCtxSynchronize, the staging copies and the launch cost"
            };
            let counters_note = if host_only {
                "driver entry points the block called: launches, syncs, sync_ms, h2d/d2h/dtod count+bytes+ms, alloc_count, free_count, module_load_count; a queued device-to-device copy is not inside a meter window, and the run-end synchronize is priced in run.probe_overhead"
            } else {
                "driver entry points the block called: launches, syncs, sync_ms, h2d/d2h/dtod count+bytes+ms, alloc_count, free_count, module_load_count; the probe's own boundary drain is excluded and priced in run.probe_overhead"
            };
            let document = json!({
                "schema": "sparkflash-tp4-forward-perf/2",
                "rank": rank,
                "world": world,
                "unit": "ms",
                "mode": mode.as_json(),
                "measurement": {
                    "wall": wall_note,
                    "kernel": kernel_note,
                    "overhead": overhead_note,
                    "counters": counters_note,
                    "block_rows": "every block is an array in legend.block_fields order, trimmed at its last non-zero field, so a host-only block is [wall_ms] and nothing repeats a field name 106,829 times; run.block_summary spells the same fields out as named keys",
                    "layers": "a layer carries its own arena_reads/sha256 counters and the kernel split of its layer_total; block_summary is the same table summed over every position",
                    "run_blocks": "preloop_arrival is the run's first collective: it completes when the slowest rank has finished its pre-loop work (arena upload, cubin loads), so it holds peer-arrival skew and not embedding cost. embedding_gathers measures the gather loop from the second prompt position on, which is why it is now comparable across ranks, rows and cache states",
                },
                "legend": legend,
                "run": run_object,
                "positions": position_json,
            });
            serde_json::to_vec(&document).map_err(|error| format!("forward perf json: {error}"))
        })?;
        std::fs::write(path, bytes)
            .map_err(|error| format!("write {}: {error}", path.display()))?;
        Ok(true)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The probe's installed flag is process-wide while its state is thread-local, so two test
        /// threads writing at once would race on the flag and see each other's installs. These
        /// tests serialise on one lock; production only ever installs from one thread.
        static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

        fn serial() -> std::sync::MutexGuard<'static, ()> {
            TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner())
        }

        fn unique_path(name: &str) -> std::path::PathBuf {
            std::env::temp_dir().join(format!(
                "sparkflash-perf-{}-{:?}-{name}",
                std::process::id(),
                std::thread::current().id()
            ))
        }

        fn meter(launches: u64, kernel_ns: u64) -> ProbeMeterReading {
            ProbeMeterReading {
                launches,
                kernel_ns,
                syncs: launches,
                sync_ns: kernel_ns / 4,
                ..ProbeMeterReading::default()
            }
        }

        #[test]
        fn the_writer_is_a_no_op_unless_the_flag_and_the_state_are_both_set() {
            let _serial = serial();
            let path = unique_path("gated.json");
            let _ = std::fs::remove_file(&path);
            reset();
            assert!(!write(&path, 0, 4, ForwardPerf::OFF).expect("flag off"));
            assert!(
                !path.exists(),
                "the default path must not create a perf file"
            );
            let probe_row = ForwardPerf {
                probe: true,
                lean: false,
                probe_nosync: false,
            };
            assert!(!write(&path, 0, 4, probe_row).expect("no installed state"));
            assert!(
                !path.exists(),
                "a probe flag without installed state writes nothing"
            );
            begin_run(ForwardPerf {
                probe: true,
                probe_nosync: false,
                lean: false,
            });
            assert!(write(&path, 0, 4, probe_row).expect("an installed probe writes"));
            assert!(path.exists());
            reset();
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        fn a_host_only_block_is_one_number_and_a_device_block_carries_its_split() {
            let _serial = serial();
            assert_eq!(
                block_json(0.202_345, &Snapshot::default(), false),
                json!([0.202]),
                "a block that ran no device work and moved no counter is its wall alone"
            );
            // A seam that only stages (no kernel) still carries its copies: the trailing-zero trim
            // must not swallow a field that did move.
            let staged = Snapshot {
                meter: ProbeMeterReading {
                    d2h_count: 4,
                    d2h_bytes: 57_344,
                    d2h_ns: 400_000,
                    ..ProbeMeterReading::default()
                },
                ..Snapshot::default()
            };
            let row = block_json(1.25, &staged, false);
            let row = row.as_array().expect("array");
            assert_eq!(row.len(), 12);
            assert_eq!(row[0], json!(1.25));
            assert_eq!(row[2], json!(0.0), "no kernel ran, so no kernel split");
            assert_eq!(row[9], json!(4));
            assert_eq!(row[10], json!(57_344));
            assert_eq!(row[11], json!(0.4));
            let device = Snapshot {
                meter: ProbeMeterReading {
                    launches: 6,
                    kernel_ns: 5_123_456,
                    syncs: 6,
                    sync_ns: 412_000,
                    h2d_count: 1,
                    h2d_bytes: 4096,
                    h2d_ns: 30_000,
                    ..ProbeMeterReading::default()
                },
                ..Snapshot::default()
            };
            let block = block_json(6.936, &device, false);
            let row = block.as_array().expect("array");
            assert_eq!(row.len(), 9, "the trailing zero counters are trimmed");
            assert_eq!(row[0], json!(6.936), "{}", BLOCK_FIELDS[0]);
            assert_eq!(row[1], json!(5_123_456), "{}", BLOCK_FIELDS[1]);
            assert_eq!(row[2], json!(1.813), "{}", BLOCK_FIELDS[2]);
            assert_eq!(row[3], json!(6), "{}", BLOCK_FIELDS[3]);
            assert_eq!(row[4], json!(6), "{}", BLOCK_FIELDS[4]);
            assert_eq!(row[5], json!(0.412), "{}", BLOCK_FIELDS[5]);
            assert_eq!(row[6], json!(1), "{}", BLOCK_FIELDS[6]);
            assert_eq!(row[7], json!(4096), "{}", BLOCK_FIELDS[7]);
            assert_eq!(row[8], json!(0.03), "{}", BLOCK_FIELDS[8]);
            assert_eq!(BLOCK_FIELDS.len(), 18);
        }

        #[test]
        fn the_writer_emits_the_perf_schema_block_split_layer_counters_and_summary() {
            let _serial = serial();
            reset();
            begin_run(ForwardPerf {
                probe: true,
                probe_nosync: false,
                lean: false,
            });
            begin_position(0, false);
            push(Scope::Position, "engram_hashes", 0.5, Snapshot::default());
            push(
                Scope::Position,
                "position_total",
                2.5,
                Snapshot {
                    meter: meter(2, 1_500_000),
                    ..Snapshot::default()
                },
            );
            begin_position(1, true);
            push(Scope::Position, "position_total", 3.25, Snapshot::default());
            push(
                Scope::Layer(0),
                "mhc_entry",
                1.25,
                Snapshot {
                    arena: HostCounters {
                        count: 4,
                        bytes: 57_344,
                        ns: 400_000,
                    },
                    ..Snapshot::default()
                },
            );
            push(
                Scope::Layer(0),
                "layer_total",
                2.0,
                Snapshot {
                    meter: meter(3, 1_000_000),
                    arena: HostCounters {
                        count: 4,
                        bytes: 57_344,
                        ns: 400_000,
                    },
                    sha: HostCounters {
                        count: 2,
                        bytes: 128,
                        ns: 100_000,
                    },
                    ..Snapshot::default()
                },
            );
            push(Scope::Head, "hc_pre", 0.75, Snapshot::default());
            push(
                Scope::Run,
                "loop_total",
                12.5,
                Snapshot {
                    meter: meter(5, 4_000_000),
                    ..Snapshot::default()
                },
            );
            progress(17, BlockTimer::start(true));
            count_arena_read(64, Some(Instant::now()));
            count_sha256(32, Some(Instant::now()));
            let path = unique_path("shape.json");
            let _ = std::fs::remove_file(&path);
            assert!(
                write(
                    &path,
                    0,
                    4,
                    ForwardPerf {
                        probe: true,
                        lean: true,
                        probe_nosync: false,
                    },
                )
                .expect("write")
            );
            let document: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
            assert_eq!(document["schema"], "sparkflash-tp4-forward-perf/2");
            assert_eq!(document["rank"], 0);
            assert_eq!(document["world"], 4);
            assert_eq!(document["mode"]["lean"], true);
            assert_eq!(document["mode"]["probe_rank"], 0);
            assert_eq!(document["mode"]["synchronize_per_block"], true);
            assert_eq!(document["mode"]["wall_clock"], "device-inclusive");
            assert!(document["measurement"]["kernel"].is_string());
            let positions = document["positions"].as_array().expect("positions");
            assert_eq!(positions.len(), 2);
            // Position 0: the total carries the split, the block keeps its one number.
            assert_eq!(positions[0]["position"], 0);
            assert_eq!(positions[0]["head"], false);
            assert_eq!(positions[0]["total_ms"], 2.5);
            assert_eq!(positions[0]["kernel_ns"], 1_500_000);
            assert_eq!(positions[0]["overhead_ms"], 1.0);
            assert_eq!(positions[0]["blocks"]["engram_hashes"], json!([0.5]));
            assert!(
                positions[0]["blocks"]["position_total"].is_null(),
                "the position total is the position row, not a block"
            );
            assert!(positions[0].get("layers").is_none());
            // Position 1: layers carry their own counters and the kernel split of layer_total.
            assert_eq!(positions[1]["position"], 1);
            assert_eq!(positions[1]["head"], true);
            assert_eq!(positions[1]["total_ms"], 3.25);
            assert_eq!(positions[1]["layer_sum_ms"], 2.0);
            assert_eq!(positions[1]["layers"][0]["layer"], 0);
            assert_eq!(positions[1]["layers"][0]["total_ms"], 2.0);
            assert_eq!(positions[1]["layers"][0]["kernel_ns"], 1_000_000);
            assert_eq!(positions[1]["layers"][0]["overhead_ms"], 1.0);
            assert_eq!(
                positions[1]["layers"][0]["counters"]["arena_reads"]["count"],
                4
            );
            assert_eq!(
                positions[1]["layers"][0]["counters"]["arena_reads"]["bytes"],
                57_344
            );
            assert_eq!(positions[1]["layers"][0]["counters"]["sha256"]["count"], 2);
            assert_eq!(
                positions[1]["layers"][0]["blocks"]["mhc_entry"],
                json!([1.25]),
                "a layer block row is the trimmed array legend.block_fields describes"
            );
            assert_eq!(positions[1]["head_blocks"]["hc_pre"], json!([0.75]));
            assert_eq!(positions[1]["progress_writes"]["count"], 1);
            assert_eq!(positions[1]["progress_writes"]["bytes"], 17);
            assert!(
                positions[1]["progress_writes"]["ms"]
                    .as_f64()
                    .is_some_and(|ms| ms >= 0.0)
            );
            // The run scope keeps the host counters and gains the split, the summary and the
            // probe's own priced overhead.
            assert_eq!(
                document["run"]["loop_total"],
                json!([12.5, 4_000_000, 8.5, 5, 5, 1.0])
            );
            assert_eq!(
                document["legend"]["block_fields"],
                json!(BLOCK_FIELDS),
                "the legend is what makes a block row readable without a second file"
            );
            assert_eq!(document["run"]["counters"]["arena_reads"]["count"], 1);
            assert_eq!(document["run"]["counters"]["arena_reads"]["bytes"], 64);
            assert_eq!(document["run"]["counters"]["sha256"]["count"], 1);
            assert_eq!(document["run"]["counters"]["sha256"]["bytes"], 32);
            assert_eq!(document["run"]["probe_overhead"]["blocks"], 0);
            let summary = document["run"]["block_summary"]
                .as_array()
                .expect("block summary");
            let mhc = summary
                .iter()
                .find(|row| row["scope"] == "layer" && row["name"] == "mhc_entry")
                .expect("mhc_entry in the summary");
            assert_eq!(mhc["count"], 1);
            assert_eq!(mhc["wall_ms"], 1.25);
            assert_eq!(mhc["kernel_ns"], 0, "the named summary fills every bucket");
            let run_total = summary
                .iter()
                .find(|row| row["scope"] == "run" && row["name"] == "loop_total")
                .expect("loop_total in the summary");
            assert_eq!(run_total["kernel_ns"], 4_000_000);
            reset();
            let _ = std::fs::remove_file(&path);

            // The no-sync mode writes the same schema under a different wall clock: the block's
            // kernel_ns is its slice of the deferred harvest, and its row gains the wait/host split
            // (wait_ms: what the readback blocks spend inside their own drains; host_ms: the host
            // code outside that wait), which is the pair a pipelining decision compares.
            reset();
            begin_run(ForwardPerf {
                probe: true,
                probe_nosync: true,
                lean: false,
            });
            begin_position(0, false);
            push_with_launches(
                Scope::Layer(0),
                "attention_output",
                2.0,
                Snapshot {
                    meter: ProbeMeterReading {
                        launches: 2,
                        syncs: 1,
                        sync_ns: 100_000,
                        d2h_count: 2,
                        d2h_bytes: 128,
                        d2h_ns: 200_000,
                        ..ProbeMeterReading::default()
                    },
                    ..Snapshot::default()
                },
                (0, 2),
            );
            push_with_launches(
                Scope::Layer(0),
                "layer_total",
                3.0,
                Snapshot {
                    meter: ProbeMeterReading {
                        launches: 2,
                        ..ProbeMeterReading::default()
                    },
                    ..Snapshot::default()
                },
                (0, 2),
            );
            STATE.with(|state| {
                let mut borrow = state.borrow_mut();
                let state = borrow.as_mut().expect("the no-sync probe is installed");
                state.deferred_ns = Some(vec![120_000, 380_000]);
                state.final_sync_ms = 0.75;
            });
            let host_path = unique_path("host-only.json");
            let _ = std::fs::remove_file(&host_path);
            assert!(
                write(
                    &host_path,
                    0,
                    4,
                    ForwardPerf {
                        probe: true,
                        probe_nosync: true,
                        lean: false,
                    },
                )
                .expect("write")
            );
            let host: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&host_path).expect("read")).expect("json");
            assert_eq!(host["mode"]["wall_clock"], "host-only");
            assert_eq!(host["mode"]["synchronize_per_block"], false);
            assert_eq!(
                host["legend"]["block_fields"],
                json!(block_field_names(true)),
                "the legend names the two extra fields this mode appends"
            );
            assert!(host["legend"]["host_only_fields"].is_string());
            assert_eq!(host["run"]["probe_overhead"]["blocks"], 0);
            assert_eq!(host["run"]["probe_overhead"]["sync_ms"], 0.75);
            let layer = &host["positions"][0]["layers"][0];
            // The two launches this block owns are the harvest's first two figures: 500 us of
            // device time against a 2.000 ms host wall, with a 0.300 ms readback wait named apart.
            let row = layer["blocks"]["attention_output"]
                .as_array()
                .expect("a host-only row");
            assert_eq!(row.len(), 20);
            assert_eq!(row[1], json!(500_000));
            assert_eq!(row[18], json!(0.3), "wait_ms");
            assert_eq!(row[19], json!(1.7), "host_ms = wall_ms - wait_ms");
            assert_eq!(layer["kernel_ns"], json!(500_000));
            assert_eq!(layer["overhead_ms"], json!(2.5));
            reset();
            let _ = std::fs::remove_file(&host_path);
        }
    }
}

/// One rank's side-buffer payload as the loop reassembles it from the head stage's exact gather.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadSide {
    /// The certified mask: one byte per rank vocabulary entry, 1 = pruned by the block bound.
    pub certified: Vec<u8>,
    /// The number of certified entries.
    pub certified_count: u32,
    /// The per-chunk block maxima.
    pub block_max: Vec<f32>,
    /// The per-chunk block argmax entries.
    pub block_argmax: Vec<u32>,
}

/// The little-endian bytes of a u32 slice (the block-argmax digest input).
#[must_use]
pub fn u32_le_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Parses one rank's side-buffer gather payload: the certified mask, its count, and the per-chunk
/// block maxima and argmax. The count is checked against the mask, so a rank that published a count
/// that disagrees with its flags is refused instead of being adjudicated.
///
/// # Errors
/// A payload that is not [crate::tp4_tail::SIDE_BYTES] long, a count outside the vocabulary, or a
/// count that does not equal the number of set flags.
pub fn parse_head_side(bytes: &[u8]) -> Result<HeadSide, String> {
    let vocab = crate::tp4_tail::VOCAB_PER_RANK;
    let chunks = crate::tp4_tail::CHUNKS_PER_RANK;
    let expected = crate::tp4_tail::SIDE_BYTES;
    if bytes.len() != expected {
        return Err(format!(
            "head side gather: {} bytes, expected {expected} (certified {vocab}, count 4, maxima {}, argmax {})",
            bytes.len(),
            chunks * 4,
            chunks * 4
        ));
    }
    let certified = bytes[..vocab].to_vec();
    let count_raw = &bytes[vocab..vocab + 4];
    let certified_count =
        u32::from_le_bytes([count_raw[0], count_raw[1], count_raw[2], count_raw[3]]);
    if certified_count as usize > vocab {
        return Err(format!(
            "head side gather: certified count {certified_count} is larger than the rank vocabulary {vocab}"
        ));
    }
    let flags = certified.iter().filter(|flag| **flag != 0).count();
    if flags != certified_count as usize {
        return Err(format!(
            "head side gather: certified count {certified_count} but {flags} flags are set"
        ));
    }
    let max_base = vocab + 4;
    let block_max = bytes[max_base..max_base + chunks * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    let argmax_base = max_base + chunks * 4;
    let block_argmax = bytes[argmax_base..argmax_base + chunks * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    Ok(HeadSide {
        certified,
        certified_count,
        block_max,
        block_argmax,
    })
}

/// One tensor's bytes read back from the live arena: the host seams' only device read.
#[cfg(feature = "cuda")]
fn read_arena_bytes(
    cuda: &sparkflash_kernels::cuda_driver::Cuda,
    arena: &sparkflash_device::arena::ResidentArena,
    name: &str,
    expected_bytes: usize,
) -> Result<Vec<u8>, String> {
    let slice = arena
        .slice(name)
        .ok_or_else(|| format!("arena read: the arena has no {name} slice"))?;
    if slice.bytes != expected_bytes {
        return Err(format!(
            "arena read: {name} is {} bytes in the arena, expected {expected_bytes}",
            slice.bytes
        ));
    }
    let mut bytes = vec![0u8; expected_bytes];
    let timer = probe::counter_timer();
    cuda.copy_dtoh(&mut bytes, &slice.ptr)
        .map_err(|error| format!("arena read: read {name}: {error}"))?;
    probe::count_arena_read(expected_bytes, timer);
    Ok(bytes)
}

/// The planned dtype of one layer-0 tensor, when the loop was given the rank plan.
#[cfg(any(feature = "cuda", test))]
fn planned_dtype<'a>(
    plan: Option<&'a sparkflash_device::resident::ResidentPlan>,
    name: &str,
) -> Option<&'a str> {
    plan.and_then(|plan| {
        plan.shards()
            .iter()
            .find(|shard| shard.name == name)
            .map(|shard| shard.dtype.as_str())
    })
}

/// The layer-0 mHC host seam as the loop runs it: the four seam tensors and the three
/// position-independent digests come from the run-scope seam cache, so the call performs no arena
/// read (the returned count is 0) and the released arithmetic runs on the same decoded f32 values.
#[cfg(any(feature = "cuda", test))]
fn run_layer_mhc_entry(
    cached: &LayerSeamTensors,
    request: &TokenStageRequest<'_>,
    stream: &[f32],
    attn_pre_mix: &[f32],
) -> Result<(MhcEntrySeam, usize), String> {
    let config = request
        .config
        .ok_or_else(|| "mHC entry: the loop was built without the model config".to_string())?;
    let hc = config.hc_mult;
    let dim = config.dim;
    if stream.len() != hc * dim {
        return Err(format!(
            "mHC entry: the residual stream is {} values, expected {hc} copies of {dim}",
            stream.len()
        ));
    }
    let seam = mhc_entry_host_seam_with_digests(
        config,
        stream,
        &cached.attn_fn,
        &cached.attn_base,
        &cached.attn_scale,
        &cached.attn_norm,
        attn_pre_mix,
        request.native_rounding.mhc,
        Some(&cached.digests()),
        Some(&cached.attn_fn_transposed),
    )?;
    Ok((seam, 0))
}

/// Writes one layer's phase1-boundary-compatible snapshots for the debug hook.
///
/// The tensor names and widths are the boundary v1 capture's: residual_in (hc copies), hc_in_pre,
/// hc_attn_pre/post/comb, pre_attention_hidden, attention_out, post_attention_hidden,
/// hc_ffn_pre/post/comb, pre_ffn_hidden, ffn_out and residual_out. f32 little endian, one file per
/// tensor; the device's attention_out is its BF16 all-reduced delta decoded to f32, so the compare
/// script must allow the BF16 round. Every value is the value the layer actually consumed or
/// produced, written after the layer's last stage passed its finite check.
///
/// # Errors
/// A directory or file write failure, or a BF16 decode failure.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn write_boundary_trace_row(
    trace_root: &std::path::Path,
    position: usize,
    layer: usize,
    residual_in: &[f32],
    hc_in_pre: &[f32],
    entry: &MhcEntrySeam,
    attention_delta_bf16: &[u8],
    attention_residual: &AttentionResidualSeam,
    ffn_seams: &Layer0FfnSeams,
    ffn_epilogue: &Layer0FfnEpilogue,
) -> Result<(), String> {
    let dir = trace_root.join(format!("pos{position}"));
    fs::create_dir_all(&dir)
        .map_err(|error| format!("boundary trace dir {}: {error}", dir.display()))?;
    let attention_out = bf16_from_le_bytes(attention_delta_bf16)?;
    for (name, values) in [
        ("residual_in", residual_in),
        ("hc_in_pre", hc_in_pre),
        ("hc_attn_pre", &entry.pre),
        ("hc_attn_post", &entry.post),
        ("hc_attn_comb", &entry.comb),
        ("pre_attention_hidden", &entry.normed),
        ("attention_out", &attention_out),
        ("post_attention_hidden", &attention_residual.out),
        ("hc_ffn_pre", &ffn_seams.ffn_pre),
        ("hc_ffn_post", &ffn_seams.ffn_post),
        ("hc_ffn_comb", &ffn_seams.ffn_comb),
        ("pre_ffn_hidden", &ffn_seams.pre_ffn_hidden),
        ("ffn_out", &ffn_epilogue.ffn_out),
        ("residual_out", &ffn_epilogue.hc_post),
    ] {
        let path = dir.join(format!("row{layer}.{name}.f32"));
        fs::write(&path, f32_le_bytes(values))
            .map_err(|error| format!("boundary trace {}: {error}", path.display()))?;
    }
    Ok(())
}

/// Runs the four-rank forward loop. This is the only GPU entry point of the module.
#[cfg(feature = "cuda")]
fn run_forward_loop(
    request: &TokenStageRequest<'_>,
    plan: &ForwardPlan,
    bound: Option<(
        &sparkflash_device::arena::ResidentArena,
        &crate::tp4_resident::DeviceBind,
    )>,
    transport: Option<&TcpTransport>,
) -> Result<ForwardRun, String> {
    use sparkflash_comm::transport::Transport;
    use sparkflash_engine::forward::boundary::{
        ATTENTION_OUT, BoundaryTable, FFN_OUT, HC_ATTN_COMB, HC_ATTN_POST, HC_ATTN_PRE,
        HC_FFN_COMB, HC_FFN_POST, HC_FFN_PRE, HC_IN_PRE, LayerBoundary, POST_ATTENTION_HIDDEN,
        PRE_ATTENTION_HIDDEN, PRE_ATTENTION_SCALES, PRE_ATTENTION_XHAT, PRE_FFN_HIDDEN,
        PRE_FFN_SCALES, PRE_FFN_XHAT, RESIDUAL_IN, RESIDUAL_OUT,
    };
    use sparkflash_kernels::cuda_driver::{Cuda, Deviceptr};

    let Some((arena, bind)) = bound else {
        return Err(
            "forward loop called without a bound arena: preflight alone cannot run a layer"
                .to_string(),
        );
    };
    let Some(transport) = transport else {
        return Err(
            "the four-rank forward loop needs the tcp transport: the embedding row of the last prompt token is gathered from its owner rank over the same collective the walk uses, and --forward without --comm-peers cannot even build the first hidden state"
                .to_string(),
        );
    };
    // The two per-layer reduces take the reference TCP exchange unless `SPARK_COMM_REDUCE=roce`
    // asks for the canonical RoCE one-shot and every rank can open it. The switch is resolved once,
    // before the loop's first collective: the barrier and the agreement gather below are what keep
    // all four ranks on one transport (see ForwardReduce::resolve). A default run never issues
    // either of them, so its collectives, bytes and arithmetic are exactly the ones they were
    // before this slot existed.
    let (reduce_switch, reduce_requested) = ReduceSwitch::from_env();
    if reduce_switch == ReduceSwitch::Roce {
        // The QP endpoint exchange has a bounded deadline and the arena upload ends at a different
        // time on every rank, so one one-word reduce on the already-connected reference transport
        // makes all four ranks start bring-up together. It moves no reduce payload and no value.
        let barrier_layout = sparkflash_comm::layout::CollectiveLayout::new(
            request.world,
            1,
            sparkflash_comm::layout::FabricSchedule::OneShotAllToAll,
        )
        .map_err(|error| format!("forward reduce: reduce-path barrier layout: {error}"))?;
        transport
            .all_reduce_bf16(&mut [0u16; 1], REDUCE_BARRIER_SEQ, &barrier_layout)
            .map_err(|error| format!("forward reduce: reduce-path barrier: {error}"))?;
    }
    let (maybe_roce, roce_detail) = if reduce_switch == ReduceSwitch::Roce {
        open_roce_reduce(request.rank, request.world)
    } else {
        (None, String::new())
    };
    let reduce = ForwardReduce::resolve(
        transport,
        &maybe_roce,
        reduce_switch,
        &reduce_requested,
        &roce_detail,
        request.world,
    )?;
    println!("forward reduce: {}", reduce.report);

    let positions = request.positions.max(1);
    let Some(token) = first_step_token(request.prompt_tokens, positions) else {
        return Err(format!(
            "--token-ids has {} ids for {positions} decode positions: a teacher-forced list must carry at least one id per row, and the first row's input is ids[len - positions]; refusing rather than starting one position late",
            request.prompt_tokens.len()
        ));
    };
    let prompt_len = prompt_len_for(request.prompt_tokens, positions).unwrap_or(0);

    let cuda = Cuda::load().map_err(|error| format!("forward loop: {error}"))?;
    // M2 run-scope switches. The launch path is async by default because every numeric observation
    // in this loop is a synchronous, stream-ordered cuMemcpyDtoH; SPARKFLASH_SYNC_LAUNCHES=1 puts
    // the per-launch drain back. The per-layer evidence chain is off unless
    // SPARKFLASH_FORWARD_EVIDENCE=1 asks for the identity kit's chain. Both are logged so a window
    // row carries the mode it was collected with.
    let async_launches = sparkflash_kernels::cuda_driver::enable_forward_async_launches();
    let evidence = enable_forward_evidence_from_env();
    reset_cubin_cache();
    println!(
        "forward loop: launch path {} (kernels run ahead of the host), evidence chain {}",
        if async_launches {
            "async"
        } else {
            "synchronous"
        },
        if evidence { "recording" } else { "off" }
    );
    if request.forward_perf.probe_nosync && !async_launches {
        // The no-sync meter queues its launches to keep the device wait off the host wall, so a row
        // that opted back into the pre-M2 drain would still measure a host-only wall for launches
        // but a device-inclusive one for staging copies. Say so rather than let the row pass.
        eprintln!(
            "forward perf probe: --forward-perf-probe-nosync queues its launches, so SPARKFLASH_SYNC_LAUNCHES=1 does not restore the drained launch path for this row"
        );
    }
    // The M2 perf probe is armed for the whole run: when the flag is off this is one relaxed atomic
    // store and the only state the writer could see stays empty.
    probe::begin_run(request.forward_perf);
    let probe_on = request.forward_perf.probe;
    let row_bytes = HIDDEN * 2;
    // The forward needs every position's embedding row, not only the last prompt one: position
    // 129's hidden state is built from the 130-token prefill, and the decode rows after it are
    // teacher-forced from the same fixture ids. Each row is read by its owner rank from the arena
    // and gathered with the same pure word movement the walk uses; the rows concatenate in
    // position order.
    let full_positions = prompt_len.saturating_add(positions).saturating_sub(1);
    if request.prompt_tokens.len() < full_positions {
        return Err(format!(
            "--token-ids has {} ids for a {} prompt plus {} decode positions: a teacher-forced run needs {} ids (the prompt, then one input per dumped row after the first)",
            request.prompt_tokens.len(),
            prompt_len,
            positions,
            full_positions
        ));
    }
    let mut prefill_hidden: Vec<u8> = Vec::with_capacity(full_positions * row_bytes);
    let mut first_step_hidden_sha256 = String::new();
    let mut first_step_owner = 0usize;
    let mut first_step_row = 0usize;
    // Two run-scope blocks, not one: the loop's FIRST collective is the run's first barrier, so it
    // completes only when the slowest rank has arrived. The first kit charged that wait to
    // `embedding_gathers`, and the four ranks of one row then disagreed by three orders of magnitude
    // (56.9 s / 9.5 s / 0.04 s / 36.4 s in row m1y35-probe-default) because the pre-loop arena
    // upload ends at a different time on every rank: upload_ms + embedding_gathers was constant
    // across the four ranks to within 45 ms, which is a barrier, not a timer fault. The wait is now
    // booked where it belongs - peer-arrival skew in `preloop_arrival` - and the embedding block
    // measures the loop itself from the second position on, so it is comparable across ranks, rows
    // and cache states. Nothing arithmetic changes: the gather loop, its order and its bytes are
    // the same, and the only added work is one probe synchronise at the split point.
    let mut timer = BlockTimer::start(probe_on);
    for (position, &prompt_token) in request.prompt_tokens[..full_positions].iter().enumerate() {
        let (owner, row) = embedding_location(prompt_token);
        // Only the owner reads its arena shard row; the other ranks contribute zeros.
        let local_bytes = if request.rank == owner {
            let mut bytes = vec![0u8; row_bytes];
            let row_ptr = Deviceptr(
                bind.tail
                    .embed
                    .0
                    .checked_add((row * row_bytes) as u64)
                    .ok_or_else(|| {
                        "embedding row offset overflows the arena address".to_string()
                    })?,
            );
            cuda.copy_dtoh(&mut bytes, &row_ptr).map_err(|error| {
                format!("embedding row copy (position {position}, token {prompt_token}): {error}")
            })?;
            bytes
        } else {
            vec![0u8; row_bytes]
        };
        let local_words: Vec<u16> = (0..local_bytes.len() / 2)
            .map(|index| u16::from_le_bytes([local_bytes[index * 2], local_bytes[index * 2 + 1]]))
            .collect();
        let mut gathered = vec![0u16; request.world * HIDDEN];
        let exchange_seq = EMBED_EXCHANGE_SEQ.wrapping_sub(position as u64);
        transport
            .all_gather_bf16(&local_words, &mut gathered, exchange_seq)
            .map_err(|error| {
                format!("embedding all-gather (position {position}, token {prompt_token}): {error}")
            })?;
        let hidden_row: Vec<u8> = gathered[owner * HIDDEN..(owner + 1) * HIDDEN]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        if position + 1 == prompt_len {
            first_step_hidden_sha256 = sha256_hex(&hidden_row);
            first_step_owner = owner;
            first_step_row = row;
        }
        prefill_hidden.extend_from_slice(&hidden_row);
        if position == 0 {
            // All four ranks have now passed the same collective, so the skew this rank paid is
            // over; from here on the loop's remaining work is the embedding seam's own cost.
            probe::run(&cuda, "preloop_arrival", timer);
            timer = BlockTimer::start(probe_on);
        }
    }
    probe::run(&cuda, "embedding_gathers", timer);
    let prompt_hidden_sha256 = sha256_hex(&prefill_hidden[..prompt_len * row_bytes]);
    let forward_hidden_sha256 = sha256_hex(&prefill_hidden);
    let hidden_sha256 = prompt_hidden_sha256.clone();
    let owner = first_step_owner;
    let row = first_step_row;
    let first_position = prompt_len.saturating_sub(1);
    let config = request
        .config
        .ok_or_else(|| "forward loop: the loop was built without the model config".to_string())?;
    let n_layers = config.n_layers;
    let config_hc = config.hc_mult;
    // The released model writes Engram into the residual stream BEFORE the block at layers 1 and 14
    // (the engine's own host forward calls engram_forward on the stream before block(), so the
    // block's mHC entry sees the written stream). The four-rank loop mirrors that exactly: a
    // replicated host seam - every rank reads the same table rows from its local checkpoint, so it
    // needs no collective - paged through the source because each table is 98 GB.
    let engram_source = if config.engram_layer_ids.is_empty() {
        None
    } else {
        Some(request.weights.ok_or_else(|| {
            format!(
                "forward loop: the model writes Engram at layers {:?} but the loop was built without a checkpoint source; the tables are 98 GB each and must be paged, not bound into the arena",
                config.engram_layer_ids
            )
        })?)
    };
    if engram_source.is_some() {
        // Engram's dead-token rule is sticky: a sentinel pads this look-back and every later one, so
        // a placeholder id would silently rewrite the seam's entire future. The teacher-forced ids
        // are real vocabulary ids; refuse by name rather than run the dead-token path by accident.
        if let Some(index) = request.prompt_tokens[..full_positions]
            .iter()
            .position(|token| *token == sparkflash_engine::forward::engram::HASH_SENTINEL)
        {
            return Err(format!(
                "forward loop: the id at --token-ids index {index} is the Engram dead-token sentinel; the Engram seam refuses a placeholder history"
            ));
        }
    }
    // The two Engram wkv matrices are constant, but the seam reads them once per layer-position and
    // [sparkflash_engine::safetensors::SafeTensorsSource] cannot borrow: without this the loop copies
    // 157 MB per layer-position. Hold them resident once and let the engine's read path borrow them.
    // Value-transparent: the same bytes reach the same matmul in the same order.
    let engram_resident = match engram_source {
        Some(source) => Some(EngramResidentSource::preload(
            source,
            &engram_weight_names(config),
        )?),
        None => None,
    };
    let engram_read_source: Option<&dyn sparkflash_engine::weights::WeightSource> =
        match &engram_resident {
            Some(resident) => Some(resident as &dyn sparkflash_engine::weights::WeightSource),
            None => {
                engram_source.map(|source| source as &dyn sparkflash_engine::weights::WeightSource)
            }
        };
    // The Engram wkv projection is the one host seam with an already-pinned device kernel that
    // computes the same W8A8 block dots (see EngramWkvDevice): upload the constant matrices once
    // and launch the kernel per layer-position instead of running the 157 M-MAC host matmul. The
    // host fetch, activation quantise, gate and write stay exactly where they are. A run without a
    // staged cubin directory (preflight, host-only paths) keeps the host matmul.
    let engram_wkv_device = match (&engram_resident, request.kernel_cubin_dir) {
        (Some(resident), Some(kernel_dir)) => {
            Some(EngramWkvDevice::build(&cuda, config, kernel_dir, resident)?)
        }
        _ => None,
    };
    let total_positions = request
        .prefill_limit
        .map_or(full_positions, |limit| limit.min(full_positions));
    let head_count = positions.min(total_positions);
    if head_count == 0 || total_positions == 0 {
        return Err(
            "forward loop: the prefill limit leaves no position to run or no head row to dump"
                .to_string(),
        );
    }
    let head_positions: Vec<usize> = (total_positions - head_count..total_positions).collect();
    if head_positions.first().copied() != Some(total_positions - head_count) {
        return Err(
            "forward loop: the head positions do not start where the limit says".to_string(),
        );
    }
    // The reference seams need the checkpoint: the compressor, the indexer and the candidate select
    // are host code the loop calls on the device chain's own per-layer input, and their tensors are
    // read from the shards rather than through the arena bind.
    let checkpoint_dir = request.checkpoint_dir.ok_or_else(|| {
        "forward loop: the run needs --checkpoint DIR: the reference compressor/indexer/candidate select read their weights from the shards, and no device kernel implements the selected-candidate construction"
            .to_string()
    })?;
    let checkpoint_index = request.checkpoint_index.map_or_else(
        || checkpoint_dir.join("model.safetensors.index.json"),
        Path::to_path_buf,
    );
    let mut reference_attention =
        crate::tp4_prefill::ReferenceAttention::open(config, checkpoint_dir, &checkpoint_index)?;
    // The ref_seam slice: the indexer's own wq_b on the pinned decode GEMV, behind the same
    // staged-cubin switch the other device arms use. SPARKFLASH_SEAM_REF_HOST=1 keeps the whole seam
    // on the host (the A/B arm). A decline at call time - a layer whose source does not serve the
    // released F8-E4M3 + E8-E8M0 pair - falls back to the engine's host native rule for that layer,
    // never to a third arithmetic.
    let mut seam_wq_b: Option<SeamDevice<'_>> = if std::env::var("SPARKFLASH_SEAM_REF_HOST").is_ok()
    {
        None
    } else {
        match request.kernel_cubin_dir {
            Some(kernel_dir) => Some(SeamDevice::build(
                &cuda,
                config,
                kernel_dir,
                reference_attention.source(),
            )?),
            None => None,
        }
    };
    // !158's loader reads the checkpoint with O_DIRECT and so never fills the page cache; the
    // buffered loader it replaced did, which is why the seam reads used to be hits. Warm the two
    // families the loop re-reads - the reference attention seams and the paged Engram rows this
    // run will look up - before position 0, with buffered reads and the bytes discarded. The loop
    // then reads the same bytes from the same pages, and the cold burst is setup time instead of a
    // first-position stall the other ranks wait on. `SPARKFLASH_SEAM_WARM=0` skips the pass, so one
    // binary can run the A/B in the same environment.
    if seam_warm_enabled() {
        reference_attention.warm_seams()?;
        if let Some(source) = engram_read_source {
            warm_engram_rows(config, request.prompt_tokens, total_positions, source)?;
        }
    }
    let body = AttentionBody::open(
        &cuda,
        request,
        arena,
        config.n_heads / request.world.max(1),
        config.head_dim,
    )?;
    let mut kv_cache: Vec<Vec<(Vec<u8>, Vec<u8>)>> = (0..n_layers)
        .map(|_| Vec::with_capacity(total_positions))
        .collect();
    // The layers, ascending: every stage is the same sequence the layer-0 negative proved on
    // device, re-parameterised by the layer index and run position-major. The surviving hc stream is
    // the next layer's input, exactly the reference's hidden carried between blocks (the block's own
    // mHC entry re-computes the attention/FFN mixes from it).
    // The one-shot rank-0 native dense launch-parity runner (--forward-native-dense-parity): off
    // unless the argv armed it. Its path never feeds the loop's arithmetic.
    let mut native_dense_parity =
        NativeDenseParity::new(request.native_dense_parity, first_position);
    let loop_started = std::time::Instant::now();
    let progress_path = request
        .evidence_dir
        .join(format!("tp4-forward-progress-r{}.json", request.rank));
    if let Some(parent) = progress_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("forward progress dir {}: {error}", parent.display()))?;
    }
    type LayerZero = (
        MhcEntrySeam,
        Layer0Projections,
        Layer0QueryPath,
        Layer0KvPath,
        AttentionStep,
        Layer0AttentionOutput,
        AttentionResidualSeam,
        Layer0FfnSeams,
        Layer0MoeRouted,
        Layer0FfnEpilogue,
        usize,
    );
    // The plan shards the released shared expert TP-column/row while the reference H5 seam is
    // replicated. A prefill reads the same replicated views for every position, so the six
    // all-gathers are paid once per layer here and every position of the layer borrows them: the
    // bytes are identical to the per-position gathers the earlier single-position loop issued.
    let shared_started = std::time::Instant::now();
    // Same rule as head_total: the counters must be read before the prewarm moves them.
    let shared_prewarm_timer = BlockTimer::start(probe_on);
    let mut shared_gathers: Vec<SharedExpertGather> = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        shared_gathers.push(gather_shared_expert_source(
            &cuda, arena, request, layer, transport,
        )?);
    }
    // WS-D: the same full-width bytes stay on the device from here on. The B12x repack of w1/w3/w2
    // and their scale grids is warmup work inside shared_prewarm, so it is outside the step timer;
    // from the first position on, the shared expert's three matrix products run on resident
    // operands instead of seven host reads, a host decode and three host matrix products per
    // position.
    //
    // The staging is UNCONDITIONAL on the cuda path whenever the plan carries a shared expert: the
    // arm (coded native vs f32 reference) is selected per token by --native-rounding. A missing
    // --kernel-cubin-dir or a cubin that does not match the baseline is a hard error here, never a
    // fallback to the host f32 seam.
    let shared_device = if config.n_shared_experts >= 1 {
        Some(build_shared_expert_device(&cuda, request, &shared_gathers)?)
    } else {
        None
    };
    probe::run(&cuda, "shared_prewarm", shared_prewarm_timer);
    let shared_prewarm_ms = shared_started.elapsed().as_millis() as u64;
    let shared_prewarm_bytes: usize = shared_gathers.iter().map(|gather| gather.full_bytes).sum();
    let shared_prewarm_reads: usize = shared_gathers.iter().map(|gather| gather.local_reads).sum();
    let shared_prewarm_gathers: usize = shared_gathers
        .iter()
        .map(|gather| gather.gather_calls)
        .sum();

    // M3/C1: the boundary projection pair re-reads and re-loads the same two cubins on every layer.
    // One cache, warmed here before the step timer and freed after the loop, keeps those modules
    // loaded for the whole run; a layer's block only looks its handle up by the pinned digest.
    let mut kernel_cache = KernelCache::default();
    if let Some(kernel_dir) = request.kernel_cubin_dir {
        kernel_cache.prewarm(&cuda, kernel_dir)?;
    }
    // The CSA-union staging operands are rebuilt once per layer per position. The slot keeps the
    // four buffers and the window/selected vectors alive across that whole loop, and memoises the
    // quantised compressed rows (verified bit-for-bit against the f32 row before reuse), so the
    // 147,456-byte store is not re-allocated and the selected rows are not re-quantised on every
    // call. The staged bytes are exactly stage_attention's; only the per-call work is gone.
    let mut attention_staging = crate::tp4_prefill::AttentionStaging::new();
    // Same idea for the attention output block: its seven pooled buffers and four host rows are
    // storage only, so they are checked out once for the loop instead of once per layer per
    // position.
    let mut attention_output_scratch = AttentionOutputScratch::new();
    // M2 Phase 2C: the mHC-entry and FFN seam tensors are immutable per-layer arena slices, so they
    // are read, decoded and hashed ONCE here and borrowed by every position. The seam arithmetic
    // and the recorded predictor/base/norm digests are the same bytes the per-call read path
    // produced; only the per-layer mhc_entry.arena_reads drops from 4 to 0.
    let layer_seams = build_layer_seam_cache(&cuda, arena, request, n_layers)?;
    // M2 Phase 2A: the tail stage and the final-norm weight are position-independent, so they are
    // opened/read ONCE at run scope and borrowed by every head position. Bytes are identical (same
    // cubins; w_block_max and head_shard_sha256 derived once from the same arena head shard; same
    // norm encoding), so the default row's head record shas and logits bin sha stay unchanged.
    let timer = BlockTimer::start(probe_on);
    let final_norm: Option<Vec<f32>> = if head_count == 0 {
        None
    } else {
        let final_norm_name = "norm.weight";
        let final_norm_bytes = match planned_dtype(request.resident_plan, final_norm_name) {
            Some("F32") => HIDDEN * 4,
            Some("BF16") => HIDDEN * 2,
            Some(other) => {
                return Err(format!(
                    "forward loop: {final_norm_name} is planned as {other}; the final norm decodes F32 or BF16 only"
                ));
            }
            None => {
                return Err(format!(
                    "forward loop: the rank plan does not carry {final_norm_name}, so its encoding is unknown"
                ));
            }
        };
        let raw = read_arena_bytes(&cuda, arena, final_norm_name, final_norm_bytes)?;
        Some(if final_norm_bytes == HIDDEN * 4 {
            f32_from_le_bytes(&raw)?
        } else {
            bf16_from_le_bytes(&raw)?
        })
    };
    let mut tail_stage = if head_count == 0 {
        None
    } else {
        let kernel_dir = request.kernel_cubin_dir.ok_or_else(|| {
            "head stage: the forward loop was built without --kernel-cubin-dir, so the tail cubins are not staged".to_string()
        })?;
        Some(crate::tp4_tail::TailStage::open(
            &cuda,
            kernel_dir,
            bind,
            config.norm_eps as f32,
        )?)
    };
    // The run-scope tail open: the final-norm weight DTOH is inside this block (the arena_reads
    // counter carries its bytes/time); the head path no longer repeats it.
    probe::run(&cuda, "tail_open", timer);
    // Debug hook for the phase1-boundary v1 cross-check: when SPARKFLASH_BOUNDARY_TRACE names a
    // directory (relative names live under the evidence dir), rank 0 writes the same per-layer
    // tensor set the engine capture holds for every position listed in SPARKFLASH_TRACE_POSITIONS
    // (comma separated). The rows are row{L}.{tensor}.f32 under pos{P}/ so a host script can diff
    // them against boundary.blob row by row. Off unless the variable is set.
    let trace_root = std::env::var_os("SPARKFLASH_BOUNDARY_TRACE").map(std::path::PathBuf::from);
    let trace_root = trace_root.map(|root| {
        if root.is_absolute() {
            root
        } else {
            request.evidence_dir.join(root)
        }
    });
    let trace_positions: Vec<usize> = std::env::var("SPARKFLASH_TRACE_POSITIONS")
        .ok()
        .map(|text| {
            text.split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_default();
    // Diagnostic boundary replay (off by default): SPARKFLASH_TP4_BOUNDARY_DIR names a directory
    // (relative names live under the evidence dir) where rank 0 writes the engine's phase-1
    // boundary v1 pair for the one position in SPARKFLASH_TP4_BOUNDARY_POSITION (default 0). The
    // 18 tensors are the values this chain consumed or produced, mapped 1:1 onto
    // boundary::TENSOR_ORDER, so the committed boundary_diff example localises the first layer and
    // tensor whose device value leaves f32/bf16 noise of the host reference. attention_out is the
    // chain's BF16 all-reduced delta widened to f32: the diff classes already allow the BF16 round
    // for that tensor.
    let boundary_dir =
        std::env::var_os("SPARKFLASH_TP4_BOUNDARY_DIR").map(std::path::PathBuf::from);
    let boundary_dir = boundary_dir.map(|dir| {
        if dir.is_absolute() {
            dir
        } else {
            request.evidence_dir.join(dir)
        }
    });
    // The replay owns its directory: create it before any of its writers run. The split dumps below
    // write inside the layer loop, before the pair writer's own create_dir_all, so without this the
    // first layer would fail with "No such file or directory" on a fresh evidence dir.
    if let Some(dir) = boundary_dir.as_deref() {
        fs::create_dir_all(dir)
            .map_err(|error| format!("boundary replay dir {}: {error}", dir.display()))?;
    }
    let boundary_position: usize = std::env::var("SPARKFLASH_TP4_BOUNDARY_POSITION")
        .ok()
        .and_then(|text| text.trim().parse::<usize>().ok())
        .unwrap_or(0);
    let mut boundary_table: Option<BoundaryTable> = match &boundary_dir {
        Some(_) => {
            let token = request
                .prompt_tokens
                .get(boundary_position)
                .copied()
                .ok_or_else(|| {
                    format!(
                        "boundary replay: position {boundary_position} is outside the {} prompt tokens",
                        request.prompt_tokens.len()
                    )
                })?;
            Some(BoundaryTable::new(
                boundary_position,
                &[token],
                request
                    .prompt_tokens
                    .get(..boundary_position)
                    .unwrap_or(&[]),
            ))
        }
        None => None,
    };
    let mut layer_zero: Option<LayerZero> = None;
    let mut layer_chain: Vec<serde_json::Value> = Vec::new();
    // Counted independently of the evidence chain, so progress rows and the position record still
    // report the layers completed when the chain is switched off.
    let mut layers_completed: usize = 0;
    let mut position_records: Vec<serde_json::Value> = Vec::with_capacity(total_positions);
    let mut heads: Vec<HeadPosition> = Vec::with_capacity(head_count);
    let mut step_samples: Vec<f64> = Vec::with_capacity(head_count);
    let mut final_stream_sha256 = String::new();
    let loop_total_timer = BlockTimer::start(probe_on);
    for position in 0..total_positions {
        let position_started = std::time::Instant::now();
        // The head flag is position metadata only; it is computed here so the probe can tag this
        // position before the first position block is recorded.
        let is_head = position + head_count >= total_positions;
        probe::begin_position(position, is_head);
        let position_total_timer = BlockTimer::since(position_started, probe_on);
        // The Engram n-gram hash row for THIS position: the tokens up to and including this row's
        // input, with start_pos = position, which is ngram_hashes' (history, start_pos) contract.
        // One row of layers x columns ids; a look-back past the sequence start pads with
        // engram_pad_id, as the reference does.
        let timer = BlockTimer::start(probe_on);
        let engram_hashes = match engram_read_source {
            Some(_) => {
                let history = request.prompt_tokens.get(..=position).ok_or_else(|| {
                    format!(
                        "forward loop: Engram needs the tokens up to position {position} but --token-ids has {}",
                        request.prompt_tokens.len()
                    )
                })?;
                let mut hashes = Vec::new();
                sparkflash_engine::forward::engram::ngram_hashes(
                    config,
                    history,
                    position,
                    &mut hashes,
                )
                .map_err(|error| {
                    format!("forward loop: Engram hashes at position {position}: {error}")
                })?;
                hashes
            }
            None => Vec::new(),
        };
        probe::position(&cuda, "engram_hashes", timer);
        let timer = BlockTimer::start(probe_on);
        let hidden =
            bf16_from_le_bytes(&prefill_hidden[position * row_bytes..(position + 1) * row_bytes])?;
        let mut stream: Vec<f32> = Vec::with_capacity(config_hc * HIDDEN);
        for _ in 0..config_hc {
            stream.extend_from_slice(&hidden);
        }
        let mut position_chain: Vec<serde_json::Value> = Vec::with_capacity(n_layers);
        let mut position_zero: Option<LayerZero> = None;
        // The released chain's cross-layer state: layer 0's attention sees the identity seed, and
        // every later layer's attention hc_pre uses the PREVIOUS layer's FFN pre mix. The
        // phase1-boundary v1 capture confirms it byte for byte (hc_in_pre(L) == hc_ffn_pre(L-1)).
        let mut attn_pre_mix = vec![0.0f32; config_hc];
        attn_pre_mix[0] = 1.0;
        probe::position(&cuda, "stream_init", timer);
        for layer in 0..n_layers {
            let layer_started = std::time::Instant::now();
            let layer_total_timer = BlockTimer::since(layer_started, probe_on);
            // Engram write (layers 1 and 14 only): the stream this block starts from gains
            // bf16_round(stream + gate[copy] * value[d]) before the block's own mHC entry, exactly
            // where the engine's host forward applies it (model.py:365). The write is identical on
            // all four ranks, so it costs no collective; the layer record keeps its sha/l2 so the
            // window can attribute the row-1/row-14 stream step to this seam alone.
            let mut engram_record: Option<serde_json::Value> = None;
            let timer = BlockTimer::start(probe_on);
            if let Some(source) = engram_read_source
                && config.engram_rows(layer).is_some()
            {
                // The write record is evidence only: the array below is the difference the seam
                // made to the stream, and nothing outside the evidence chain reads it. A lean run
                // skips the clone, the subtraction and the l2 instead of materialising them.
                let before = evidence_record_enabled().then(|| stream.clone());
                match &engram_wkv_device {
                    Some(device) => {
                        let mut projection =
                            |act_codes: &[u8],
                             act_scales: &[u8],
                             _wkv_codes: &[u8],
                             _wkv_scales: &[u8],
                             out_width: usize,
                             inner: usize,
                             out: &mut [f32]| {
                                device
                                    .project(layer, act_codes, act_scales, out_width, inner, out)
                                    .map_err(|error| {
                                        sparkflash_engine::linalg::EngineError::Shape {
                                            what: "Engram device wkv".to_owned(),
                                            detail: error,
                                        }
                                    })
                            };
                        sparkflash_engine::forward::engram::engram_forward_with_projection(
                            config,
                            layer,
                            &mut stream,
                            &engram_hashes,
                            source,
                            ENGRAM_ELEMENT_LIMIT,
                            &mut projection,
                        )
                    }
                    None => sparkflash_engine::forward::engram::engram_forward(
                        config,
                        layer,
                        &mut stream,
                        &engram_hashes,
                        source,
                        ENGRAM_ELEMENT_LIMIT,
                    ),
                }
                .map_err(|error| {
                    format!("Engram seam: layer {layer} position {position}: {error}")
                })?;
                if let Some(before) = before {
                    let mut write = vec![0.0f32; stream.len()];
                    for ((slot, after), prior) in
                        write.iter_mut().zip(stream.iter()).zip(before.iter())
                    {
                        *slot = after - prior;
                    }
                    let write_l2 = write.iter().map(|value| value * value).sum::<f32>().sqrt();
                    engram_record = Some(serde_json::json!({
                        "layer": layer,
                        "hashes": engram_hashes.len(),
                        "write_sha256": evidence_f32_sha256(&write),
                        "write_l2": write_l2,
                    }));
                }
                probe::layer(&cuda, layer, "engram_write", timer);
            }
            let traced =
                request.rank == 0 && trace_root.is_some() && trace_positions.contains(&position);
            let boundary_here = boundary_table.is_some() && position == boundary_position;
            let layer_stream_in = if traced || boundary_here {
                Some(stream.clone())
            } else {
                None
            };
            let shared = shared_gathers.get(layer).ok_or_else(|| {
                format!("forward loop: no shared-expert gather was built for layer {layer}")
            })?;
            let cached_seams = layer_seams.layers.get(layer).ok_or_else(|| {
                format!("forward loop: no seam cache was built for layer {layer}")
            })?;
            let attempt: Result<(Option<serde_json::Value>, Option<LayerZero>), String> = (|| {
                let timer = BlockTimer::start(probe_on);
                let (entry, mhc_reads) =
                    run_layer_mhc_entry(cached_seams, request, &stream, &attn_pre_mix)?;
                probe::layer(&cuda, layer, "mhc_entry", timer);
                let timer = BlockTimer::start(probe_on);
                let projections = run_layer_attention_projections(
                    layer,
                    &cuda,
                    &mut kernel_cache,
                    request,
                    bind,
                    &entry.normed,
                    position,
                    &mut native_dense_parity,
                )?;
                probe::layer(&cuda, layer, "attention_projections", timer);
                let timer = BlockTimer::start(probe_on);
                let query_path = run_layer_query_path(
                    layer,
                    &cuda,
                    &mut kernel_cache,
                    request,
                    bind,
                    arena,
                    &projections,
                    position,
                    &mut native_dense_parity,
                )?;
                probe::layer(&cuda, layer, "query_path", timer);
                let timer = BlockTimer::start(probe_on);
                let kv_path =
                    run_layer_kv_path(layer, &cuda, request, arena, &projections, position)?;
                probe::layer(&cuda, layer, "kv_path", timer);
                let cache = kv_cache.get_mut(layer).ok_or_else(|| {
                    format!("forward loop: no kv cache row list was built for layer {layer}")
                })?;
                if cache.len() != position {
                    return Err(format!(
                        "forward loop: layer {layer}'s kv cache has {} rows before position {position}: the position loop left the cache out of step",
                        cache.len()
                    ));
                }
                cache.push((kv_path.window_codes.clone(), kv_path.window_scales.clone()));
                // The reference seams run on the device chain's own per-layer input: the compressor
                // and the candidate select see this block's normed row and this rank's q latent,
                // never a second reference forward. The seam gives the window width, the ascending
                // shared compressed ids and the compressed rows they land on.
                let timer = BlockTimer::start(probe_on);
                let seam = match seam_wq_b.as_mut() {
                    Some(device) => reference_attention.step_with_backend(
                        layer,
                        position,
                        &entry.normed,
                        &query_path.q_latent_bf16,
                        device,
                    )?,
                    None => reference_attention.step(
                        layer,
                        position,
                        &entry.normed,
                        &query_path.q_latent_bf16,
                    )?,
                };
                probe::layer(&cuda, layer, "ref_seam", timer);
                if seam.window_width == 0 || seam.window_width > cache.len() {
                    return Err(format!(
                        "forward loop: layer {layer} position {position}: the reference seam attended {} rows, the chain has {} rows cached",
                        seam.window_width,
                        cache.len()
                    ));
                }
                let window_start = cache.len() - seam.window_width;
                let timer = BlockTimer::start(probe_on);
                let window_rows: Vec<(Vec<u8>, Vec<u8>)> = cache[window_start..].to_vec();
                let staged = attention_staging.stage(layer, &window_rows, &seam)?;
                probe::layer(&cuda, layer, "stage_attention", timer);
                let twin_requested = request.forward_perf.twin_requested(is_head, position);
                let timer = BlockTimer::start(probe_on);
                let attention_step = body.run(
                    &cuda,
                    request,
                    layer,
                    &query_path,
                    &staged,
                    position,
                    twin_requested,
                )?;
                attention_staging.recycle(staged);
                probe::layer(&cuda, layer, "attention_body", timer);
                let timer = BlockTimer::start(probe_on);
                let attention_output = run_layer_attention_output(
                    layer,
                    &cuda,
                    &mut kernel_cache,
                    request,
                    bind,
                    &attention_step,
                    &reduce,
                    position,
                    &mut native_dense_parity,
                    &mut attention_output_scratch,
                )?;
                probe::layer(&cuda, layer, "attention_output", timer);
                native_dense_parity.finish(request);
                let timer = BlockTimer::start(probe_on);
                let attention_residual = attention_residual_host_seam_with(
                    request.config.ok_or_else(|| {
                        "layer attention residual: the loop was built without the model config"
                            .to_string()
                    })?,
                    &entry,
                    &attention_output.reduced_bf16,
                    request.native_rounding.mhc,
                )?;
                probe::layer(&cuda, layer, "attention_residual", timer);
                let timer = BlockTimer::start(probe_on);
                let ffn_seams =
                    run_layer_ffn_seams(cached_seams, request, &attention_residual, &entry.pre)?;
                probe::layer(&cuda, layer, "ffn_seams", timer);
                let moe_routed = run_layer_moe_routed(
                    layer, &cuda, request, bind, &ffn_seams, transport, &reduce,
                )?;
                let ffn_epilogue = run_layer_ffn_epilogue(
                    layer,
                    &cuda,
                    request,
                    &ffn_seams,
                    &moe_routed,
                    &attention_residual,
                    shared,
                    shared_device.as_ref(),
                )?;
                stream = ffn_epilogue.hc_post.clone();
                if layer_evidence_readbacks() && stream.iter().any(|value| !value.is_finite()) {
                    return Err(format!(
                        "forward loop: layer {layer}'s ffn hc_post stream has a non-finite value; refusing to feed the next layer"
                    ));
                }
                if let (Some(root), Some(layer_stream_in)) = (&trace_root, &layer_stream_in) {
                    write_boundary_trace_row(
                        root,
                        position,
                        layer,
                        layer_stream_in,
                        &attn_pre_mix,
                        &entry,
                        &attention_output.reduced_bf16,
                        &attention_residual,
                        &ffn_seams,
                        &ffn_epilogue,
                    )?;
                }
                if boundary_here {
                    let table = boundary_table
                        .as_mut()
                        .ok_or_else(|| "boundary replay: the table vanished".to_string())?;
                    let residual_in = layer_stream_in
                        .clone()
                        .ok_or_else(|| "boundary replay: no layer input row".to_string())?;
                    // The chain's attention delta is BF16; the reference records f32, so widen it
                    // instead of pretending the bytes are the f32 result.
                    let attention_out = bf16_from_le_bytes(&attention_output.reduced_bf16)?;
                    let mut row = LayerBoundary::new(layer);
                    row.push_stream(config, RESIDUAL_IN, residual_in)
                        .map_err(|error| format!("boundary replay residual_in: {error}"))?;
                    row.push_mixes(config, HC_IN_PRE, attn_pre_mix.clone())
                        .map_err(|error| format!("boundary replay hc_in_pre: {error}"))?;
                    row.push_mixes(config, HC_ATTN_PRE, entry.pre.clone())
                        .map_err(|error| format!("boundary replay hc_attn_pre: {error}"))?;
                    row.push_mixes(config, HC_ATTN_POST, entry.post.clone())
                        .map_err(|error| format!("boundary replay hc_attn_post: {error}"))?;
                    row.push_comb(config, HC_ATTN_COMB, entry.comb.clone())
                        .map_err(|error| format!("boundary replay hc_attn_comb: {error}"))?;
                    row.push_block_input(
                        config,
                        PRE_ATTENTION_HIDDEN,
                        PRE_ATTENTION_XHAT,
                        PRE_ATTENTION_SCALES,
                        &entry.normed,
                    )
                    .map_err(|error| format!("boundary replay pre_attention_hidden: {error}"))?;
                    row.push_dense(config, ATTENTION_OUT, attention_out)
                        .map_err(|error| format!("boundary replay attention_out: {error}"))?;
                    row.push_stream(
                        config,
                        POST_ATTENTION_HIDDEN,
                        attention_residual.out.clone(),
                    )
                    .map_err(|error| format!("boundary replay post_attention_hidden: {error}"))?;
                    row.push_mixes(config, HC_FFN_PRE, ffn_seams.ffn_pre.clone())
                        .map_err(|error| format!("boundary replay hc_ffn_pre: {error}"))?;
                    row.push_mixes(config, HC_FFN_POST, ffn_seams.ffn_post.clone())
                        .map_err(|error| format!("boundary replay hc_ffn_post: {error}"))?;
                    row.push_comb(config, HC_FFN_COMB, ffn_seams.ffn_comb.clone())
                        .map_err(|error| format!("boundary replay hc_ffn_comb: {error}"))?;
                    row.push_block_input(
                        config,
                        PRE_FFN_HIDDEN,
                        PRE_FFN_XHAT,
                        PRE_FFN_SCALES,
                        &ffn_seams.pre_ffn_hidden,
                    )
                    .map_err(|error| format!("boundary replay pre_ffn_hidden: {error}"))?;
                    row.push_dense(config, FFN_OUT, ffn_epilogue.ffn_out.clone())
                        .map_err(|error| format!("boundary replay ffn_out: {error}"))?;
                    row.push_stream(config, RESIDUAL_OUT, ffn_epilogue.hc_post.clone())
                        .map_err(|error| format!("boundary replay residual_out: {error}"))?;
                    table
                        .push_row(row)
                        .map_err(|error| format!("boundary replay row {layer}: {error}"))?;
                    #[allow(clippy::collapsible_if)]
                    if layer == 0 {
                        if let Some(dir) = boundary_dir.as_ref() {
                            // The boundary format has no stage split, but the one number it cannot
                            // disambiguate - ffn_out low because the routed kernel or because the
                            // host shared-expert seam - needs the routed delta and the shared-only
                            // remainder. Both are the values the epilogue actually used.
                            let routed = bf16_from_le_bytes(&moe_routed.delta)?;
                            if routed.len() != ffn_epilogue.ffn_out.len() {
                                return Err(format!(
                                    "boundary replay: routed delta is {} values, ffn_out {}",
                                    routed.len(),
                                    ffn_epilogue.ffn_out.len()
                                ));
                            }
                            let shared: Vec<f32> = ffn_epilogue
                                .ffn_out
                                .iter()
                                .zip(routed.iter())
                                .map(|(total, delta)| total - delta)
                                .collect();
                            let routed_path = dir.join("layer0-routed-delta.f32");
                            let shared_path = dir.join("layer0-shared-delta.f32");
                            fs::write(&routed_path, f32_le_bytes(&routed)).map_err(|error| {
                                format!(
                                    "boundary replay routed split {}: {error}",
                                    routed_path.display()
                                )
                            })?;
                            fs::write(&shared_path, f32_le_bytes(&shared)).map_err(|error| {
                                format!(
                                    "boundary replay shared split {}: {error}",
                                    shared_path.display()
                                )
                            })?;
                        }
                    }
                }
                // The reference's cross-layer pre mix: the next layer's attention hc_pre consumes
                // THIS layer's FFN pre mix (hc_in_pre(L+1) == hc_ffn_pre(L)).
                attn_pre_mix = ffn_seams.ffn_pre.clone();
                let timer = BlockTimer::start(probe_on);
                // M2/C10: the layer-chain entry and every digest that only feeds it are recorded
                // only in evidence mode. A default/--forward-lean run pays the block timer and
                // nothing else; the probe still gets an entry_json row so both window rows can be
                // read with the same block set.
                let entry_json = if evidence_record_enabled() {
                    Some(serde_json::json!({
                            "layer": layer,
                            "native_rounding": request.native_rounding.as_str(),
                    "forward_perf": { "lean": request.forward_perf.lean },
                            "mhc_entry": {
                                "attn_in_sha256": entry.attn_in_sha256,
                                "normed_sha256": entry.normed_sha256,
                                "predictor_sha256": entry.predictor_sha256,
                                "arena_reads": mhc_reads,
                            },
                            "attention_body": {
                                "position": attention_step.position,
                                "out_sha256": attention_step.out_sha256,
                                "inverse_rope_sha256": attention_step.inverse_rope_sha256,
                                "window_rows": attention_step.window_rows,
                                "selected_count": attention_step.selected_count,
                                "store_rows": attention_step.store_rows,
                                "ratio": attention_step.ratio,
                                "q_sha256": attention_step.q_sha256,
                                "sink_sha256": attention_step.sink_sha256,
                                "staged_selected_sha256": attention_step.selected_sha256,
                                "twin_bf16_mismatches": attention_step.twin.as_ref().map(|twin| twin.mismatches),
                                "twin_slots": attention_step.twin.as_ref().map(|twin| twin.slots),
                                "twin_max_abs_diff": attention_step.twin.as_ref().map(|twin| twin.max_abs),
                                "twin_skipped": attention_step.twin.is_none(),
                            },
                            "attention_output": {
                                "wo_a_out_sha256": attention_output.wo_a_out_sha256,
                                "wo_b_partial_sha256": attention_output.wo_b_partial_sha256,
                                "reduced_sha256": attention_output.reduced_sha256,
                                "reduced_l2": attention_output.reduced_l2,
                            },
                            "attention_residual": {
                                "out_sha256": attention_residual.out_sha256,
                                "out_l2": attention_residual.out_l2,
                            },
                            "ffn_seams": {
                                "ffn_in_sha256": ffn_seams.ffn_in_sha256,
                                "pre_ffn_hidden_sha256": ffn_seams.pre_ffn_hidden_sha256,
                                "act_fp8_sha256": ffn_seams.act_fp8_sha256,
                                "act_scales_sha256": ffn_seams.act_scales_sha256,
                            },
                            "moe_routed": {
                                "router_ids_sha256": moe_routed.router_ids_sha256(),
                                "payload_sha256": moe_routed.payload_sha256(),
                                "routed_delta_sha256": moe_routed.delta_sha256(),
                            },
                            "ffn_epilogue": {
                                "ffn_out_sha256": ffn_epilogue.ffn_out_sha256,
                                "ffn_out_l2": ffn_epilogue.ffn_out_l2,
                                "shared_arith": ffn_epilogue.shared_arith,
                                "shared_kernel": ffn_epilogue.shared_kernel,
                                "shared_arithmetic_mode": ffn_epilogue.shared_mode,
                                "shared_launch_symbols": ffn_epilogue.launch_symbols,
                                "shared_launch_count": ffn_epilogue.launch_symbols.len(),
                                "shared_sha256": ffn_epilogue.shared_sha256,
                                "shared_l2": ffn_epilogue.shared_l2,
                                "shared_f32_sha256": ffn_epilogue.shared_f32_sha256,
                                "shared_native_vs_f32_rel_l2": ffn_epilogue.shared_native_vs_f32_rel_l2,
                                "hc_post_sha256": ffn_epilogue.hc_post_sha256,
                                "hc_post_l2": ffn_epilogue.hc_post_l2,
                            },
                            "stream_sha256": evidence_f32_sha256(&stream),
                        }))
                } else {
                    None
                };
                probe::layer(&cuda, layer, "entry_json", timer);
                let zero = if layer == 0 {
                    Some((
                        entry,
                        projections,
                        query_path,
                        kv_path,
                        attention_step,
                        attention_output,
                        attention_residual,
                        ffn_seams,
                        moe_routed,
                        ffn_epilogue,
                        mhc_reads,
                    ))
                } else {
                    None
                };
                Ok((entry_json, zero))
            })(
            );
            match attempt {
                Ok((entry_json, zero)) => {
                    // Clear the record every layer, even when the evidence chain is off: a stale
                    // layer-N engram must never be attached to a later layer's entry.
                    let engram_record = engram_record.take();
                    if let Some(mut entry_json) = entry_json {
                        if let Some(object) = entry_json.as_object_mut() {
                            object.insert(
                                "wall_ms".to_string(),
                                serde_json::json!(layer_started.elapsed().as_millis() as u64),
                            );
                            if let Some(engram) = engram_record {
                                object.insert("engram".to_string(), engram);
                            }
                        }
                        position_chain.push(entry_json);
                    }
                    layers_completed = layer + 1;
                    if let Some(zero) = zero {
                        position_zero = Some(zero);
                    }
                    probe::layer(&cuda, layer, "layer_total", layer_total_timer);
                    if request.forward_perf.progress_due(is_head, layer, n_layers) {
                        let mut progress = forward_progress(
                            "running",
                            request,
                            n_layers,
                            &position_chain,
                            loop_started.elapsed(),
                        );
                        if let Some(object) = progress.as_object_mut() {
                            object.insert("position".to_string(), serde_json::json!(position));
                            object.insert(
                                "total_positions".to_string(),
                                serde_json::json!(total_positions),
                            );
                            object.insert(
                                "layers_completed".to_string(),
                                serde_json::json!(layers_completed),
                            );
                        }
                        let timer = BlockTimer::start(probe_on);
                        let bytes = write_forward_progress(&progress_path, &progress)?;
                        probe::progress(bytes, timer);
                    }
                }
                Err(error) => {
                    let mut progress = forward_progress(
                        "failed",
                        request,
                        n_layers,
                        &position_chain,
                        loop_started.elapsed(),
                    );
                    if let Some(object) = progress.as_object_mut() {
                        object.insert("failed_layer".to_string(), serde_json::json!(layer));
                        object.insert("failed_position".to_string(), serde_json::json!(position));
                        object.insert("error".to_string(), serde_json::json!(error));
                        object.insert(
                            "layers_completed".to_string(),
                            serde_json::json!(layers_completed),
                        );
                    }
                    let timer = BlockTimer::start(probe_on);
                    let bytes = write_forward_progress(&progress_path, &progress)?;
                    probe::progress(bytes, timer);
                    return Err(format!(
                        "four-rank forward loop: position {position} layer {layer} refused: {error}"
                    ));
                }
            }
        }
        #[allow(clippy::collapsible_if)]
        if position == boundary_position {
            if let (Some(table), Some(dir)) = (boundary_table.as_ref(), boundary_dir.as_ref()) {
                fs::create_dir_all(dir)
                    .map_err(|error| format!("boundary replay dir {}: {error}", dir.display()))?;
                let manifest_path = dir.join("boundary-device.manifest");
                let blob_path = dir.join("boundary-device.blob");
                let mut manifest = fs::File::create(&manifest_path).map_err(|error| {
                    format!(
                        "boundary replay manifest {}: {error}",
                        manifest_path.display()
                    )
                })?;
                let mut blob = fs::File::create(&blob_path).map_err(|error| {
                    format!("boundary replay blob {}: {error}", blob_path.display())
                })?;
                table
                    .write_v1(n_layers, &mut manifest, &mut blob)
                    .map_err(|error| {
                        format!("boundary replay write {}: {error}", manifest_path.display())
                    })?;
            }
        }
        let head = if is_head {
            Some(run_head_position(
                &cuda,
                request,
                tail_stage.as_ref().ok_or_else(|| {
                    "forward loop: head position but the tail stage is closed".to_string()
                })?,
                transport,
                config,
                final_norm.as_deref().ok_or_else(|| {
                    "forward loop: head position but the final norm is not cached".to_string()
                })?,
                &stream,
                &attn_pre_mix,
                position,
            )?)
        } else {
            None
        };
        let timer = BlockTimer::start(probe_on);
        let wall_ms = position_started.elapsed().as_millis() as u64;
        let stream_sha256 = evidence_f32_sha256(&stream);
        if position + 1 == total_positions {
            final_stream_sha256 = stream_sha256.clone();
        }
        let mut record = serde_json::json!({
            "position": position,
            "wall_ms": wall_ms,
            "head_position": is_head,
            "stream_sha256": stream_sha256,
            "layers_completed": layers_completed,
        });
        if let Some(head) = &head {
            let mut head_json = serde_json::json!({
                "token": head.token,
                "value": head.value,
                "claimable": head.claimable,
                "logits_sha256": head.logits_sha256,
                "position": head.position,
            });
            if let Some(reference) = request.reference_ids.get(heads.len()) {
                head_json["reference_id"] = serde_json::json!(reference);
                head_json["matches_reference"] = serde_json::json!(*reference == head.token);
            }
            record["head"] = head_json;
        }
        position_records.push(record);
        probe::position(&cuda, "position_record", timer);
        if let Some(head) = head {
            if !head.claimable {
                return Err(format!(
                    "four-rank forward loop: position {}'s head winner is certified by a rank screen; refusing to claim a pruned token",
                    head.position
                ));
            }
            step_samples.push(wall_ms as f64);
            heads.push(head);
        }
        let mut progress = forward_progress(
            "position_complete",
            request,
            n_layers,
            &position_chain,
            loop_started.elapsed(),
        );
        if let Some(object) = progress.as_object_mut() {
            object.insert("position".to_string(), serde_json::json!(position));
            object.insert(
                "total_positions".to_string(),
                serde_json::json!(total_positions),
            );
            object.insert(
                "head_positions".to_string(),
                serde_json::json!(head_positions),
            );
            object.insert("layers_completed".to_string(), serde_json::json!(n_layers));
        }
        let timer = BlockTimer::start(probe_on);
        let bytes = write_forward_progress(&progress_path, &progress)?;
        probe::progress(bytes, timer);
        if position + 1 == total_positions {
            layer_chain = position_chain;
            layer_zero = position_zero;
        }
        probe::position(&cuda, "position_total", position_total_timer);
    }
    probe::run(&cuda, "loop_total", loop_total_timer);
    // M3/C1: the run-scope kernel cache: the projection modules stayed loaded for every layer and
    // are released once here, after the loop's last timed sample.
    kernel_cache.free_all(&cuda);
    attention_output_scratch.release(&cuda);
    let timer = BlockTimer::start(probe_on);
    if let Some(stage) = tail_stage.take() {
        stage.close(&cuda);
    }
    probe::run(&cuda, "tail_close", timer);
    let mut loop_complete = forward_progress(
        "loop_complete",
        request,
        n_layers,
        &layer_chain,
        loop_started.elapsed(),
    );
    if let Some(object) = loop_complete.as_object_mut() {
        object.insert("layers_completed".to_string(), serde_json::json!(n_layers));
    }
    write_forward_progress(&progress_path, &loop_complete)?;
    let (
        mhc_entry,
        projections,
        query_path,
        kv_path,
        attention_body,
        attention_output,
        attention_residual,
        ffn_seams,
        moe_routed,
        ffn_epilogue,
        mhc_arena_reads,
    ) = layer_zero.ok_or_else(|| {
        "forward loop: the layer loop ran no layer, so there is no layer-0 evidence".to_string()
    })?;
    let census = attention_stage_census(
        &bind.attention.pointers,
        &plan.attention_plan,
        request.resident_plan,
    );
    let census_launchable = census["launchable"].as_u64().unwrap_or(0);
    let census_blocked: Vec<String> = census["blocked"]
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    // The replay vectors serve-lead's DeviceChainTokenSource needs to reconstruct a claimed
    // envelope's input without the fixture file: the prompt the run fed, and the full
    // teacher-forced input (prompt plus decode positions) whose rows produced the head records.
    // They are evidence only; no arithmetic reads them.
    let prompt_ids: Vec<u32> = request
        .prompt_tokens
        .iter()
        .take(prompt_len)
        .copied()
        .collect();
    let forward_ids: Vec<u32> = request
        .prompt_tokens
        .iter()
        .take(total_positions)
        .copied()
        .collect();
    let evidence = serde_json::json!({
        "schema": "sparkflash-tp4-forward-embed/1",
        "stage": "embedding-gathered",
        "rank": request.rank,
        "world": request.world,
        "prompt_token": token,
        "prompt_len": prompt_len,
        "first_step_position": first_position,
        "decode_positions": positions,
        "reference_ids": request.reference_ids,
        "prefill_tokens": prompt_len,
        "prefill_embedding_sha256": hidden_sha256,
        "first_step_hidden_bf16_sha256": first_step_hidden_sha256,
        "prefill_exchange_calls": prompt_len,
        "prefill_exchange_bytes": prefill_exchange_bytes(request.world, prompt_len),
        "owner_rank": owner,
        "shard_row": row,
        "row_bytes": row_bytes,
        "exchange_seq": EMBED_EXCHANGE_SEQ,
        "exact": "bf16 all-gather of raw u16 words; the assembled row is bit-identical to the owner arena bytes",
        "rank_read_its_shard": request.rank == owner,
        "hidden_bf16_sha256": first_step_hidden_sha256,
        "next_missing": "selected-candidate construction, the 130-position prefill sequence, and the certified head tail (four rank screen launches + exact f32 logits gather + argmax over 129280)",
        "mhc_entry": {
            "stage": "layer0-mhc-entry-host-seam",
            "hidden_sha256": mhc_entry.hidden_sha256,
            "predictor_sha256": mhc_entry.predictor_sha256,
            "base_sha256": mhc_entry.base_sha256,
            "scale": mhc_entry.scale,
            "attn_norm_sha256": mhc_entry.norm_sha256,
            "pre_sha256": sha256_hex(&f32_le_bytes(&mhc_entry.pre)),
            "post_sha256": sha256_hex(&f32_le_bytes(&mhc_entry.post)),
            "comb_sha256": sha256_hex(&f32_le_bytes(&mhc_entry.comb)),
            "attn_in_sha256": mhc_entry.attn_in_sha256,
            "normed_attn_input_sha256": mhc_entry.normed_sha256,
            "normed_attn_input_bf16_sha256": sha256_hex(&bf16_le_bytes(&mhc_entry.normed)),
            "arena_reads": mhc_arena_reads,
            "seed": "layer-0 seed A = [1,0,0,0] through hc_pre, then attn_norm RMSNorm; layers 1-39 use the previous layer's FFN pre mix as the reference does (hc_in_pre(L) == hc_ffn_pre(L-1))",
            "exact": "the released host reference arithmetic (hc_mixes + hc_pre + RMSNorm); no kernel launched",
            "kernel_launched": false,
        },
        "attention_projections": projections.to_json(),
        "query_path": query_path.to_json(),
        "kv_path": kv_path.to_json(),
        "attention_body": attention_body.to_json(),
        "attention_output": attention_output.to_json(),
        "attention_residual": attention_residual.to_json(),
        "ffn_seams": ffn_seams.to_json(),
        "moe_routed": moe_routed.to_json(),
        "ffn_epilogue": ffn_epilogue.to_json(),
        "attention_operand_census": plan.operand_census,
        "attention_stage_census": census,
        "token_claimed": false,
        "note": "the layers 0 through 39 ran the same stage sequence on the single staged row: each layer is the device projections, the CSA union attention body (still a one-row wiring smoke, window_rows 1, selected_count 0), the wo_a/wo_b projection pair, the canonical attention all-reduce, the host mHC residual and FFN seams, the five device MoE launches, the MoE all-reduce, the host shared expert and the FFN hc_post; the surviving stream fed the next layer and layer 39 was collapsed with the reference's cross-layer hc_pre pre mix and final-normed. This is a structural 40-layer chain, not a token: selected-candidate construction, the 130-position prefill sequence and the certified head tail are not wired, so screening logits from this row would still be wrong numbers",
        "combine_preflight": plan.selection.evidence(plan.geometry),
        "attention_launchable": census_launchable,
        "attention_blocked": census_blocked,
    });
    // These fields are inserted after the macro so the evidence literal stays under serde_json's
    // json! recursion limit; the per-layer chain is a normal array, not macro syntax.
    let mut evidence = evidence;
    if let Some(object) = evidence.as_object_mut() {
        object.insert(
            "layer_chain".to_string(),
            serde_json::Value::Array(layer_chain),
        );
        object.insert("layers_executed".to_string(), serde_json::json!(n_layers));
        // The chain can legitimately be empty (the shipped default records no evidence), so the
        // artifact says so in-band instead of leaving a reader to infer it from an empty array.
        object.insert(
            "evidence_mode".to_string(),
            serde_json::json!(if evidence_record_enabled() {
                "recording"
            } else {
                "off"
            }),
        );
        object.insert(
            "evidence_digest_calls".to_string(),
            serde_json::json!(EVIDENCE_HASH_CALLS.with(std::cell::Cell::get)),
        );
        object.insert(
            "shared_expert_prewarm".to_string(),
            serde_json::json!({
                "layers": n_layers,
                "gather_calls": shared_prewarm_gathers,
                "arena_reads": shared_prewarm_reads,
                "gathered_bytes": shared_prewarm_bytes,
                "wall_ms": shared_prewarm_ms,
                "note": "the replicated H5 shared-expert views are gathered once per layer and reused by every position of a prefill instead of once per position; the layer's seam reads the same bytes the per-position gather produced",
            }),
        );
        object.insert(
            "layer_seam_prewarm".to_string(),
            serde_json::json!({
                "layers": n_layers,
                "arena_reads": layer_seams.reads,
                "hits": n_layers * total_positions,
                "decoded_bytes": layer_seams.bytes,
                "wall_ms": layer_seams.wall_ms,
                "note": "the position-independent mHC-entry and FFN seam tensors are read, decoded and hashed once per layer before the position loop and reused by every position; the seam arithmetic and the predictor/base/norm sha256 values are the same bytes as the per-call read path, and the per-layer mhc_entry.arena_reads is 0 after the prewarm",
            }),
        );
        object.insert(
            "evidence_written".to_string(),
            serde_json::json!("loop-complete"),
        );
    }
    let timer = BlockTimer::start(probe_on);
    let evidence_path = request
        .evidence_dir
        .join(format!("tp4-forward-embed-r{}.json", request.rank));
    fs::create_dir_all(request.evidence_dir).map_err(|error| {
        format!(
            "forward evidence dir {}: {error}",
            request.evidence_dir.display()
        )
    })?;
    fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&evidence)
            .map_err(|error| format!("forward evidence json: {error}"))?,
    )
    .map_err(|error| format!("write {}: {error}", evidence_path.display()))?;
    probe::run(&cuda, "evidence_write", timer);

    // The expanded m26 bundle: the layer-0 MoE stage's host-seam outputs, so the twin can adjudicate
    // the next window's device run against this refused negative's exact input.
    // Rank 0 writes the canonical fixed-name bundle the task-11 m26 harness consumes; every other
    // rank writes the same three files under its own directory so the four ranks cannot race on one
    // path (each rank's H3 bytes differ because its attention shard differs).
    let bundle_name = if request.rank == 0 {
        "moe-stage-input".to_string()
    } else {
        format!("moe-stage-input-r{}", request.rank)
    };
    let bundle_dir = request.evidence_dir.join(bundle_name);
    fs::create_dir_all(&bundle_dir).map_err(|error| {
        format!(
            "moe-stage-input bundle dir {}: {error}",
            bundle_dir.display()
        )
    })?;
    let pre_ffn_bytes = f32_le_bytes(&ffn_seams.pre_ffn_hidden);
    fs::write(bundle_dir.join("pre_ffn_hidden.f32"), &pre_ffn_bytes)
        .map_err(|error| format!("write pre_ffn_hidden.f32: {error}"))?;
    fs::write(bundle_dir.join("act_fp8.bin"), &ffn_seams.act_fp8)
        .map_err(|error| format!("write act_fp8.bin: {error}"))?;
    fs::write(bundle_dir.join("act_scales.bin"), &ffn_seams.act_scales)
        .map_err(|error| format!("write act_scales.bin: {error}"))?;
    let moe_input = serde_json::json!({
        "schema": "sparkflash-moe-stage-input/1",
        "layer": 0,
        "tokens": 1,
        "ranks": request.world,
        "hidden": HIDDEN,
        "hc_mult": HC_MULT,
        "note": "the refused negative's only row (position 129) after the FFN host seams; this rank's H3/H4 bytes are deterministic, rank 0's are under moe-stage-input/ (the m26 --input convention) and ranks 1-3 under moe-stage-input-r1/r2/r3/, so four ranks never race on one path",
        "files": {
            "pre_ffn_hidden.f32": {"path": "pre_ffn_hidden.f32", "dtype": "f32-le", "values": ffn_seams.pre_ffn_hidden.len(), "bytes": pre_ffn_bytes.len(), "sha256": ffn_seams.pre_ffn_hidden_sha256},
            "act_fp8.bin": {"path": "act_fp8.bin", "dtype": "e4m3", "values": ffn_seams.act_fp8.len(), "bytes": ffn_seams.act_fp8.len(), "sha256": ffn_seams.act_fp8_sha256},
            "act_scales.bin": {"path": "act_scales.bin", "dtype": "e8m0", "values": ffn_seams.act_scales.len(), "bytes": ffn_seams.act_scales.len(), "sha256": ffn_seams.act_scales_sha256},
        },
        "seams": {
            "post_attention_hidden_sha256": attention_residual.out_sha256,
            "attn_pre_mix_sha256": sha256_hex(&f32_le_bytes(&mhc_entry.pre)),
            "ffn_in_sha256": ffn_seams.ffn_in_sha256,
            "pre_ffn_hidden_sha256": ffn_seams.pre_ffn_hidden_sha256,
            "act_fp8_sha256": ffn_seams.act_fp8_sha256,
            "act_scales_sha256": ffn_seams.act_scales_sha256,
        },
        "device_stage": {
            "wired": true,
            "launches_per_rank": 5,
            "kernels": ["moe_router_sm121", "routed_gate_up_sm121", "swiglu_mxfp8_quant_weighted_sm121", "b12x_mxfp4_moe_stored_sm121"],
            "reason": "the five MoE launches ran on this rank's seams; this bundle is their exact input so the m26 twin and device records diff field by field against the loop record",
        },
    });
    fs::write(
        bundle_dir.join("moe-stage-input.json"),
        serde_json::to_vec_pretty(&moe_input)
            .map_err(|error| format!("moe-stage-input json: {error}"))?,
    )
    .map_err(|error| format!("write moe-stage-input.json: {error}"))?;

    // The position loop ran every forward position through all layers with the reference seams and
    // the real CSA union body. The four ranks now agree on the decoded ids before any envelope can
    // claim one; a disagreement is a named refusal, never a silently averaged token.
    if heads.len() != head_count {
        return Err(format!(
            "forward loop: {} head positions ran, {} were requested",
            heads.len(),
            head_count
        ));
    }
    let timer = BlockTimer::start(probe_on);
    let mut id_words: Vec<u16> = Vec::with_capacity(heads.len() * 2);
    for head in &heads {
        let bytes = head.token.to_le_bytes();
        id_words.push(u16::from_le_bytes([bytes[0], bytes[1]]));
        id_words.push(u16::from_le_bytes([bytes[2], bytes[3]]));
    }
    let mut gathered_ids = vec![0u16; request.world * id_words.len()];
    transport
        .all_gather_bf16(&id_words, &mut gathered_ids, TOKEN_CONSENSUS_SEQ)
        .map_err(|error| format!("forward loop: token consensus all-gather: {error}"))?;
    let mut consensus: Option<Vec<u32>> = None;
    for rank in 0..request.world {
        let start_index = rank * id_words.len();
        let mut ids = Vec::with_capacity(heads.len());
        for index in 0..heads.len() {
            let low = gathered_ids[start_index + index * 2];
            let high = gathered_ids[start_index + index * 2 + 1];
            ids.push(u32::from(low) | (u32::from(high) << 16));
        }
        match &consensus {
            None => consensus = Some(ids),
            Some(first) => {
                if first != &ids {
                    return Err(format!(
                        "forward loop: the four ranks disagree on the decoded ids: rank 0 wrote {first:?} and rank {rank} wrote {ids:?}; refusing to claim a token the ranks did not agree on"
                    ));
                }
            }
        }
    }
    let consensus =
        consensus.ok_or_else(|| "forward loop: no rank published a decoded id".to_string())?;
    let token_matches: Vec<Option<bool>> = consensus
        .iter()
        .enumerate()
        .map(|(index, token)| {
            request
                .reference_ids
                .get(index)
                .map(|reference| reference == token)
        })
        .collect();
    probe::run(&cuda, "consensus", timer);
    let timer = BlockTimer::start(probe_on);
    let mut tokens = Tokens {
        start_position: head_positions.first().copied(),
        prompt_len: Some(prompt_len),
        ..Tokens::default()
    };
    let mut dump_logits: Vec<Vec<f32>> = Vec::with_capacity(heads.len());
    for (index, head) in heads.iter().enumerate() {
        let top1 = head
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut second = f32::NEG_INFINITY;
        for value in &head.logits {
            if *value > second && *value < top1 {
                second = *value;
            }
        }
        tokens.ids.push(consensus[index]);
        tokens.positions.push(head.position);
        tokens.argmax_logits.push(head.value);
        tokens.top1_margins.push(f64::from(top1 - second));
        dump_logits.push(head.logits.clone());
    }
    let reference_slice =
        if request.reference_ids.len() == tokens.ids.len() && !request.reference_ids.is_empty() {
            Some(request.reference_ids)
        } else {
            None
        };
    let dump = crate::tp4_token::write_dump(
        request.evidence_dir,
        request.rank,
        &tokens.positions,
        &tokens.ids,
        &dump_logits,
        reference_slice,
    )?;
    probe::run(&cuda, "dump", timer);
    tokens.logits_sha256 = Some(crate::tp4_token::logits_sha256(
        &heads[heads.len() - 1].logits,
    )?);
    let last_head = heads
        .last()
        .ok_or_else(|| "forward loop: the head positions produced no token".to_string())?;
    let final_normed_sha256 = last_head
        .head_record
        .pointer("/norm/normed_bf16_sha256")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    let final_collapsed_sha256 = last_head
        .head_record
        .pointer("/input/sha256")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    // The evidence record the loop built is rewritten with the head fields, the per-position
    // summary and the final digests. It is written before the barrier, so even a rank that fails to
    // reach the barrier leaves its full evidence on disk.
    let timer = BlockTimer::start(probe_on);
    {
        let evidence_dir = request.evidence_dir;
        fs::create_dir_all(evidence_dir)
            .map_err(|error| format!("forward evidence dir {}: {error}", evidence_dir.display()))?;
        let path = evidence_dir.join(format!("tp4-forward-embed-r{}.json", request.rank));
        let mut evidence = evidence;
        if let Some(object) = evidence.as_object_mut() {
            object.insert(
                "forward_positions".to_string(),
                serde_json::json!(total_positions),
            );
            object.insert(
                "head_positions".to_string(),
                serde_json::json!(head_positions),
            );
            object.insert(
                "position_records".to_string(),
                serde_json::json!(position_records),
            );
            object.insert(
                "forward_embedding_sha256".to_string(),
                serde_json::json!(forward_hidden_sha256),
            );
            object.insert(
                "final_stream_sha256".to_string(),
                serde_json::json!(final_stream_sha256),
            );
            object.insert(
                "final_collapsed_sha256".to_string(),
                serde_json::json!(final_collapsed_sha256),
            );
            object.insert(
                "final_normed_sha256".to_string(),
                serde_json::json!(final_normed_sha256),
            );
            object.insert(
                "tokens".to_string(),
                serde_json::json!({
                    "ids": tokens.ids,
                    "positions": tokens.positions,
                    "consensus": true,
                    "claimable": heads.iter().map(|head| head.claimable).collect::<Vec<bool>>(),
                    "matches_reference": token_matches,
                    "argmax_logits": tokens.argmax_logits,
                    "top1_margins": tokens.top1_margins,
                }),
            );
            // S8 asserted: one claim that ties the envelope to the exact contract it was measured
            // against. A token is claimed only when this run dumped a head row per requested
            // position (a prefill-limit run is a replay, never a claim), every dumped row names the
            // teacher-forced reference token QA supplied, and every head winner survived the rank
            // screens. The other evidence fields are untouched.
            object.insert(
                "token_claimed".to_string(),
                serde_json::json!(
                    request.prefill_limit.is_none()
                        && tokens.positions.len() == positions
                        && token_matches.len() == positions
                        && token_matches.iter().all(|m| *m == Some(true))
                        && heads.iter().all(|h| h.claimable)
                ),
            );
            object.insert(
                "attention".to_string(),
                serde_json::json!({
                    "body": attention_body.to_json(),
                    "decoded_tokens_agree_on_four_ranks": true,
                    "sequence": "position-major: every position runs all 40 layers with the reference compressor/indexer/candidate seams on the chain's own per-layer input, stages the real 128-row window and the expanded compressed store, launches the CSA union body per head block, applies the inverse RoPE before wo_a, and only then runs the residual, FFN and next layer",
                }),
            );
            object.insert("head".to_string(), serde_json::json!(last_head.head_record));
            object.insert("prompt_ids".to_string(), serde_json::json!(prompt_ids));
            object.insert("forward_ids".to_string(), serde_json::json!(forward_ids));
            object.insert(
                "prefill_limit".to_string(),
                serde_json::json!(request.prefill_limit),
            );
            object.insert(
                "replay_note".to_string(),
                serde_json::json!("prompt_ids is the exact prompt the run fed (first prompt_len ids); forward_ids is the full teacher-forced input including the decode rows, so a consumer can replay a claimed envelope without the fixture file"),
            );
            object.insert(
                "next_missing".to_string(),
                serde_json::json!("nothing on the token path: the four-rank loop ran the reference seams and the real CSA union attention body for every position, and the head tail produced the consensus tokens"),
            );
            object.insert(
                "evidence_written".to_string(),
                serde_json::json!(
                    "loop-complete, position-complete, head-complete, token-consensus-complete"
                ),
            );
        }
        fs::write(
            &path,
            serde_json::to_vec_pretty(&evidence)
                .map_err(|error| format!("forward evidence json: {error}"))?,
        )
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    probe::run(&cuda, "evidence_write", timer);
    let step_ms = StepMs::from_samples(step_samples.clone());
    let decode_samples: Vec<f64> = if step_samples.len() > 1 {
        step_samples[1..].to_vec()
    } else {
        step_samples.clone()
    };
    let _ = &decode_samples;
    // Every rank joins this beat before any rank can close the forward transport and reach the
    // walk: without it a rank that left the loop first could start walk reduces while another rank
    // was still posting prefill-gather frames.
    let timer = BlockTimer::start(probe_on);
    {
        use sparkflash_comm::layout::{CollectiveLayout, FabricSchedule};
        let barrier_layout =
            CollectiveLayout::new(request.world, 8, FabricSchedule::OneShotAllToAll)
                .map_err(|error| format!("forward loop end barrier layout: {error}"))?;
        let mut barrier = vec![(request.rank as u16).wrapping_add(1); 8];
        transport
            .all_reduce_bf16(&mut barrier, FORWARD_BARRIER_SEQ, &barrier_layout)
            .map_err(|error| format!("forward loop end barrier: {error}"))?;
    }
    probe::run(&cuda, "barrier", timer);
    // The separate M2 perf file: written after the barrier so its own write time is not charged to
    // any measured block. With the flag off nothing is installed and nothing is created.
    if probe_on {
        probe::finish(&cuda);
        let perf_path = request
            .evidence_dir
            .join(format!("tp4-forward-perf-r{}.json", request.rank));
        probe::write(
            &perf_path,
            request.rank,
            request.world,
            request.forward_perf,
        )?;
    }
    // The run's probe state is spent once its file is on disk; a second token stage in this process
    // starts from the default path rather than appending to the first run's samples.
    probe::reset();
    // The combine plan's census describes the follow-on combine stage (and reports
    // routed_experts_down=false because the decide-preflight never launches an expert). A completed
    // forward run DID launch the routed down experts for every layer of every position, and the
    // envelope validates that flag for forward=run, so state it here instead of inheriting the
    // preflight's answer.
    let mut launch_census = plan.launch_census.clone();
    let rounding_record = serde_json::json!({
        "active": request.native_rounding.as_str(),
        "combine": request.native_rounding.combine,
        "mhc": request.native_rounding.mhc,
        "shared": request.native_rounding.shared,
        "note": "the arithmetic switches this run executed; none is the f32 arm",
    });
    if let serde_json::Value::Object(ref mut map) = launch_census {
        map.insert("routed_experts_down".to_string(), serde_json::json!(true));
        map.insert("native_rounding".to_string(), rounding_record);
    } else {
        launch_census =
            serde_json::json!({"routed_experts_down": true, "native_rounding": rounding_record});
    }
    Ok(ForwardRun {
        tokens,
        step_ms,
        wall_ms: step_samples,
        dump,
        launch_census,
    })
}

/// The argv fragment the next window's collector must pass for the forward stage, documented next
/// to the code that consumes it so the kit and the binary cannot drift.
#[must_use]
pub fn expected_argv_fragment() -> &'static str {
    "--forward --forward-preflight --kernel-cubin-dir <staged-cubins> --token-ids <prompt ids> --forward-reference-ids <reference ids>"
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tp4_combine::{CombineKernelSet, PRE_ADDED_COMBINE_KERNEL, STORED_COMBINE_KERNEL};

    #[test]
    fn forward_perf_defaults_to_off() {
        let perf = ForwardPerf::OFF;
        assert!(!perf.any());
        assert_eq!(perf, ForwardPerf::default());
        assert!(!perf.lean);
        assert!(!perf.probe);
        assert_eq!(perf.describe(), "probe=false lean=false");
    }

    #[test]
    fn the_probe_switch_is_independent_of_lean() {
        let probe = ForwardPerf {
            probe: true,
            lean: false,
            probe_nosync: false,
        };
        let lean = ForwardPerf {
            probe: false,
            lean: true,
            probe_nosync: false,
        };
        let both = ForwardPerf {
            probe: true,
            lean: true,
            probe_nosync: false,
        };
        assert!(probe.any());
        assert!(lean.any());
        assert!(both.any());
        assert_eq!(probe.describe(), "probe=true lean=false");
        assert_eq!(both.describe(), "probe=true lean=true");
        // The probe does not change the diagnostic cadence decisions; lean alone does.
        assert_eq!(
            probe.twin_requested(true, 129),
            ForwardPerf::OFF.twin_requested(true, 129)
        );
        assert_eq!(
            probe.progress_due(true, 3, 40),
            ForwardPerf::OFF.progress_due(true, 3, 40)
        );
        assert!(probe.twin_requested(true, 129));
        assert!(!lean.twin_requested(true, 129));
        assert!(probe.write_head_seam_file(false));
    }

    #[test]
    fn the_probe_mode_struct_names_the_row_it_measured() {
        let mode = ProbeMode::for_run(
            ForwardPerf {
                probe: true,
                lean: true,
                probe_nosync: false,
            },
            2,
        );
        assert_eq!(
            mode,
            ProbeMode {
                lean: true,
                probe_rank: 2,
                synchronize_per_block: true,
                host_only_wall: false,
            }
        );
        assert_eq!(
            mode.as_json(),
            serde_json::json!({
                "lean": true,
                "probe_rank": 2,
                "synchronize_per_block": true,
                "wall_clock": "device-inclusive",
            })
        );
        // The default run still has a well-defined mode line (it just never writes one).
        let off = ForwardPerf::OFF.probe_mode(0);
        assert!(!off.lean);
        assert_eq!(off.probe_rank, 0);
        assert!(off.synchronize_per_block);
        assert!(!off.host_only_wall);
        // The no-sync mode is a different wall clock, and it says so in the one added field.
        let host_only = ProbeMode::for_run(
            ForwardPerf {
                probe: true,
                probe_nosync: true,
                lean: false,
            },
            1,
        );
        assert!(!host_only.synchronize_per_block);
        assert!(host_only.host_only_wall);
        assert_eq!(
            host_only.as_json()["wall_clock"],
            serde_json::json!("host-only")
        );
        assert_eq!(host_only.as_json()["probe_rank"], serde_json::json!(1));
    }

    #[test]
    fn the_block_timer_never_starts_on_the_default_path() {
        let off = BlockTimer::start(false);
        assert!(!off.is_armed());
        assert_eq!(off.elapsed_ms(), None);
        assert_eq!(off, BlockTimer::default());
        let on = BlockTimer::start(true);
        assert!(on.is_armed());
        let measured = on.elapsed_ms().expect("an armed timer measures");
        assert!((0.0..60_000.0).contains(&measured), "{measured} ms");
        // A total reuses the function-level clock but stays gated by the same switch.
        let started = std::time::Instant::now();
        assert!(!BlockTimer::since(started, false).is_armed());
        assert!(BlockTimer::since(started, true).elapsed_ms().is_some());
    }

    #[test]
    fn probe_off_takes_the_non_probe_branches() {
        probe::reset();
        // The default run installs no probe state, so the counter clock is None (no Instant read)
        // and every block timer is unarmed: the loop takes the existing path.
        probe::begin_run(ForwardPerf::OFF);
        assert!(!probe::installed());
        assert_eq!(probe::counter_timer(), None);
        let path = std::env::temp_dir().join(format!(
            "sparkflash-perf-off-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);
        assert!(
            !probe::write(&path, 0, 4, ForwardPerf::OFF).expect("the writer is a no-op"),
            "a default run must not write the perf file"
        );
        assert!(!path.exists());
        probe::reset();
    }

    #[test]
    fn lean_only_changes_diagnostics() {
        let default = ForwardPerf::OFF;
        let lean = ForwardPerf {
            probe: false,
            lean: true,
            probe_nosync: false,
        };
        assert!(lean.any());
        // twin: the head/twin cadence is gone everywhere.
        assert!(!lean.twin_requested(true, 129));
        assert!(!lean.twin_requested(false, 0));
        assert!(default.twin_requested(true, 129));
        assert!(default.twin_requested(false, 0));
        // twin at non-head positions follows the same cadence, and lean turns it off there too.
        assert!(!default.twin_requested(false, 4));
        assert!(default.twin_requested(false, 8));
        assert!(!lean.twin_requested(false, 8));
        // progress: default rewrites after every layer at a head position; lean uses the non-head
        // cadence (every 8 layers plus the last), so 40 layers -> 5 writes.
        assert!(default.progress_due(true, 3, 40));
        assert!(!lean.progress_due(true, 3, 40));
        assert!(lean.progress_due(true, 7, 40));
        assert!(lean.progress_due(true, 39, 40));
        let default_writes = (0..40)
            .filter(|l| default.progress_due(true, *l, 40))
            .count();
        let lean_writes = (0..40).filter(|l| lean.progress_due(true, *l, 40)).count();
        assert_eq!(default_writes, 40);
        assert_eq!(lean_writes, 5);
        // head seam files: default every head position; lean only the last (the alias).
        assert!(default.write_head_seam_file(false));
        assert!(default.write_head_seam_file(true));
        assert!(!lean.write_head_seam_file(false));
        assert!(lean.write_head_seam_file(true));
    }

    #[test]
    fn the_tail_stage_is_opened_once_per_run() {
        // Phase 2A: exactly one code call to TailStage::open (the run-scope one); the head path
        // borrows the stage. The needle is concatenated so this test's own source does not count.
        let needle = concat!("TailStage", "::open(");
        let source = include_str!("tp4_forward.rs");
        let opens = source
            .lines()
            .filter(|line| line.contains(needle) && !line.trim_start().starts_with("//"))
            .count();
        assert_eq!(opens, 1);
        let head = source
            .split("fn run_head_position(")
            .nth(1)
            .expect("run_head_position is in this file");
        let head_section = head.split("\nfn ").next().unwrap_or(head);
        assert!(!head_section.lines().any(|line| line.contains(needle)));
    }

    #[test]
    fn column_shards_join_in_rank_order() {
        let shards = vec![vec![1u8, 2], vec![3u8, 4], vec![5u8, 6], vec![7u8, 8]];
        assert_eq!(join_column_shards(&shards), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn row_shards_join_each_row_across_ranks() {
        // Two ranks, each holding both rows' two columns: row 0 is rank 0's row 0 then rank 1's.
        let shards = vec![vec![1u8, 2, 3, 4], vec![5u8, 6, 7, 8]];
        assert_eq!(
            join_row_shards(&shards, 2, 2).unwrap(),
            vec![1, 2, 5, 6, 3, 4, 7, 8]
        );
        assert!(join_row_shards(&shards, 3, 2).is_err());
        assert!(join_row_shards(&[], 1, 1).is_err());
    }

    #[test]
    fn the_shared_expert_gather_namespace_is_disjoint_from_the_reduce_namespaces() {
        let namespaces = [
            EMBED_EXCHANGE_SEQ,
            ATTENTION_REDUCE_SEQ,
            MOE_REDUCE_SEQ,
            SHARED_EXPERT_GATHER_SEQ,
            HEAD_GATHER_SEQ,
            HEAD_SIDE_GATHER_SEQ,
            TOKEN_CONSENSUS_SEQ,
            NATIVE_COMBINE_GATHER_SEQ,
        ];
        for (index, left) in namespaces.iter().enumerate() {
            for right in &namespaces[index + 1..] {
                assert!(
                    left.abs_diff(*right) > 0x8000,
                    "{left:#x} and {right:#x} share the per-layer stride"
                );
            }
        }
    }

    fn request<'a>(dir: &'a Path, kernels: Option<&'a Path>) -> TokenStageRequest<'a> {
        TokenStageRequest {
            rank: 0,
            world: 4,
            n_layers: 40,
            positions: 1,
            evidence_dir: dir,
            kernel_cubin_dir: kernels,
            preflight: true,
            combine_rows: DEFAULT_COMBINE_ROWS,
            combine_bytes: Some((DEFAULT_COMBINE_ROWS * 288, DEFAULT_COMBINE_ROWS * 18)),
            prompt_tokens: &[129],
            reference_ids: &[],
            resident_plan: None,
            config: None,
            checkpoint_dir: None,
            checkpoint_index: None,
            weights: None,
            prefill_limit: None,
            native_rounding: NativeRounding::NONE,
            native_dense_parity: None,
            forward_perf: ForwardPerf::OFF,
        }
    }

    #[test]
    fn the_preflight_refuses_an_engram_config_without_a_checkpoint_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = sparkflash_engine::config::ModelConfig::v41_flash().expect("released config");
        let mut req = request(dir.path(), None);
        req.config = Some(&config);
        let outcome = token_stage(&req);
        assert!(
            outcome.is_refused(),
            "an Engram config without a source must refuse"
        );
        let error = outcome.error().expect("refusal reason");
        assert!(error.contains("Engram seam"), "{error}");
        assert!(error.contains("--checkpoint"), "{error}");
    }

    #[test]
    fn the_interlock_refuses_without_the_preflight_flag_and_writes_a_valid_envelope() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut req = request(dir.path(), None);
        req.preflight = false;
        let outcome = token_stage(&req);
        assert!(outcome.is_refused());
        let error = outcome.error().expect("refusal");
        assert!(error.contains(PREFLIGHT_FLAG), "{error}");
        assert!(outcome.envelope().logits_dump.is_none());
        let path = outcome.write(dir.path()).expect("write refusal");
        assert_eq!(path, dir.path().join("tp4-token-r0.json"));
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read")).expect("json");
        assert_eq!(value["schema"], "sparkflash-tp4-token/1");
        assert_eq!(value["forward"], "refused");
        assert_eq!(value["positions_run"], 0);
        assert!(value["tokens"]["ids"].as_array().expect("ids").is_empty());
        assert!(
            std::fs::read_dir(dir.path()).expect("dir").count() == 1,
            "only the envelope"
        );
    }

    #[test]
    fn a_stored_geometry_without_a_stored_cubin_refuses_by_name_before_any_launch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let req = request(dir.path(), None);
        let outcome = token_stage(&req);
        let error = outcome.error().expect("refusal");
        assert!(
            error.contains("stored-layout combine kernel unavailable"),
            "{error}"
        );
        assert!(error.contains(STORED_COMBINE_KERNEL), "{error}");
        outcome.write(dir.path()).expect("write refusal");
        assert_eq!(
            outcome.envelope().launch_census["routed_experts_down"],
            false
        );
    }

    #[test]
    fn the_reference_vector_is_paired_one_to_one_with_the_decode_positions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let kernels = tempfile::tempdir().expect("kernels");
        std::fs::write(
            kernels
                .path()
                .join(format!("{STORED_COMBINE_KERNEL}.cubin")),
            b"staged",
        )
        .expect("write");
        let fixture = [491u32, 367, 201, 80, 3302, 201, 54, 201, 4341];

        let mut req = request(dir.path(), Some(kernels.path()));
        req.reference_ids = &fixture;
        req.positions = 9;
        assert!(
            preflight(&req).is_ok(),
            "nine references pair with nine positions"
        );

        req.positions = 8;
        let outcome = preflight(&req).expect_err("nine references for eight positions");
        let error = outcome.error().expect("refusal");
        assert!(
            error.contains("9 reference token ids for 8 decode positions"),
            "{error}"
        );

        req.positions = 9;
        req.reference_ids = &fixture[..3];
        let outcome = preflight(&req).expect_err("three references for nine positions");
        let error = outcome.error().expect("refusal");
        assert!(
            error.contains("3 reference token ids for 9 decode positions"),
            "{error}"
        );
    }

    #[test]
    fn the_attention_census_names_b12x_and_stored_operands() {
        let b12x = (Deviceptr(0x1000), Deviceptr(0x2000));
        let stored = (Deviceptr(0x3000), Deviceptr(0x3000));
        let mut pointers = BTreeMap::new();
        pointers.insert("layers.0.attn.wq_a.weight".to_string(), stored);
        pointers.insert("layers.0.attn.wkv.weight".to_string(), stored);
        for name in [
            "layers.0.attn.wq_b.weight",
            "layers.0.attn.wo_a.weight",
            "layers.0.attn.wo_b.weight",
        ] {
            pointers.insert(name.to_string(), b12x);
        }
        let census = attention_stage_census(&pointers, &[], None);
        assert_eq!(census["launchable"], 4 - 1);
        assert_eq!(
            census["blocked"],
            serde_json::json!(["layers.0.attn.wq_a.weight", "layers.0.attn.wkv.weight"])
        );
        let rows = census["rows"].as_array().expect("rows");
        assert_eq!(rows.len(), 5);
        for row in rows {
            let name = row["name"].as_str().expect("name");
            assert_eq!(row["bound"], true, "{name}");
            let expected = if name.ends_with("wq_a.weight") || name.ends_with("wkv.weight") {
                "stored"
            } else {
                "b12x"
            };
            assert_eq!(row["layout"], expected, "{name}");
        }
        assert_eq!(census["blocked_reason"], census["blocked_reason"]);

        // A bind that has not happened yet says so: every operand unbound and unlaunchable.
        let empty = attention_stage_census(&BTreeMap::new(), &[], None);
        assert_eq!(empty["launchable"], 0);
        assert_eq!(empty["blocked"].as_array().expect("blocked").len(), 5);
        assert!(
            empty["rows"]
                .as_array()
                .expect("rows")
                .iter()
                .all(|row| row["bound"] == false)
        );
    }

    #[test]
    fn the_prefill_embedding_exchange_is_one_world_sized_row_per_prompt_token() {
        assert_eq!(prefill_exchange_bytes(4, 1), 4 * 5120 * 2);
        assert_eq!(prefill_exchange_bytes(4, 130), 130 * 4 * 5120 * 2);
        assert_eq!(prefill_exchange_bytes(1, 0), 0);
    }

    #[test]
    fn the_first_dumped_row_starts_from_the_last_prompt_token_not_the_last_fed_id() {
        let fixture_ids: Vec<u32> = (0..138).collect();
        assert_eq!(first_step_token(&fixture_ids, 9), Some(129));
        assert_eq!(prompt_len_for(&fixture_ids, 9), Some(130));
        // The last fed id is position 137's reference token, never the first row's input.
        assert_ne!(
            first_step_token(&fixture_ids, 9),
            fixture_ids.last().copied()
        );
        // One row: the first and only input is the last id.
        assert_eq!(first_step_token(&[5, 7, 9], 1), Some(9));
        // A list shorter than the rows has no first-row input at all.
        assert_eq!(first_step_token(&[5, 7], 3), None);
        assert_eq!(prompt_len_for(&[5, 7], 3), None);
        assert_eq!(first_step_token(&[], 1), None);
        assert_eq!(first_step_token(&[5], 0), None);
    }

    #[test]
    fn the_full_plan_attention_operand_census_counts_the_groups_and_bytes() {
        use sparkflash_device::resident::{HeaderEntry, ResidentPlan};
        // Two layers of the released attention shapes: 6 replicated tensors per layer
        // (wq_a/wkv weights and scales, q_norm, kv_norm), 4 column and 2 row.
        let mut entries: Vec<HeaderEntry> = Vec::new();
        for layer in 0..2 {
            for (name, dtype, shape) in [
                ("wq_a.weight", "F8_E4M3", vec![1280, 5120]),
                ("wq_a.scale", "F8_E8M0", vec![40, 160]),
                ("wkv.weight", "F8_E4M3", vec![512, 5120]),
                ("wkv.scale", "F8_E8M0", vec![16, 160]),
                ("q_norm.weight", "BF16", vec![1280]),
                ("kv_norm.weight", "BF16", vec![512]),
                ("wq_b.weight", "F8_E4M3", vec![8192, 1280]),
                ("wq_b.scale", "F8_E8M0", vec![256, 40]),
                ("wo_a.weight", "F8_E4M3", vec![2048, 4096]),
                ("wo_a.scale", "F8_E8M0", vec![64, 128]),
                ("wo_b.weight", "F8_E4M3", vec![5120, 2048]),
                ("wo_b.scale", "F8_E8M0", vec![160, 64]),
            ] {
                entries.push(HeaderEntry::new(
                    format!("layers.{layer}.attn.{name}"),
                    dtype,
                    shape,
                ));
            }
        }
        let plan = ResidentPlan::plan(0, 4, entries).expect("plan");
        let census = attention_operand_census(&plan);
        assert_eq!(
            census["plan_hash"],
            format!("{:#018x}", plan.plan_hash()),
            "the census names the plan it measured"
        );
        assert_eq!(census["projection_weights"]["tensors"], 10);
        assert_eq!(census["projection_weights"]["b12x"], 10);
        assert_eq!(census["projection_weights"]["stored"], 0);
        let group = |name: &str| -> serde_json::Value {
            census["groups"]
                .as_array()
                .expect("groups")
                .iter()
                .find(|row| row["group"] == name)
                .unwrap_or_else(|| panic!("group {name} missing"))
                .clone()
        };
        let replicated = group("attention-replicated");
        assert_eq!(replicated["tensors"], 12);
        assert_eq!(replicated["tiled"], 4);
        let per_layer = 6_553_600u64 + 2_621_440 + 6_400 + 2_560 + 3_584;
        assert_eq!(
            replicated["tiled_weight_bytes"],
            2 * (6_553_600 + 2_621_440)
        );
        assert_eq!(replicated["scale_bytes"], 2 * (6_400 + 2_560));
        assert_eq!(replicated["resident_norm_bytes"], 2 * 3_584);
        assert_eq!(replicated["total_bytes"], 2 * per_layer);
        let column = group("attention-column");
        assert_eq!(column["tensors"], 8);
        assert_eq!(column["tiled"], 4);
        let row = group("attention-row");
        assert_eq!(row["tensors"], 4);
        assert_eq!(row["tiled"], 2);
        assert_eq!(census["repacked_bytes"], 0);
        assert_eq!(census["per_token_weight_h2d_bytes"], 0);
    }

    #[test]
    fn the_moe_slot_map_is_the_sorted_unique_selection() {
        let (ids, slots) = moe_slots(&[7, 3, 7, 0, 5, 3], 384).expect("slots");
        assert_eq!(slots, vec![0, 3, 5, 7]);
        assert_eq!(ids, vec![3, 1, 3, 0, 2, 1]);
        let error = moe_slots(&[0, 384], 384).expect_err("an id off the table is refused");
        assert!(error.contains("384"), "{error}");
    }

    #[test]
    fn the_ffn_host_seams_build_the_moe_stage_inputs() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let mut hidden = Vec::with_capacity(hc * dim);
        for copy in 0..hc {
            for index in 0..dim {
                hidden.push(((index * 7 % 101) as f32 * 0.002) - copy as f32 * 0.25);
            }
        }
        let attn_pre_mix = vec![0.25f32, 0.25, 0.25, 0.25];
        let func: Vec<f32> = (0..PREDICTOR_ROWS * hc * dim)
            .map(|index| (index % 89) as f32 * 0.0002)
            .collect();
        let base = vec![0.0f32; PREDICTOR_ROWS];
        let scale = vec![0.0f32; 3];
        let ffn_norm = vec![1.0f32; dim];
        let seams = layer0_ffn_host_seams(
            config,
            &hidden,
            &attn_pre_mix,
            &func,
            &base,
            &scale,
            &ffn_norm,
        )
        .expect("ffn seams");
        assert_eq!(seams.ffn_in.len(), dim);
        assert_eq!(seams.pre_ffn_hidden.len(), dim);
        assert_eq!(seams.ffn_in_sha256.len(), 64);
        assert_eq!(seams.ffn_pre_sha256.len(), 64);
        assert_eq!(seams.act_fp8.len(), dim);
        assert_eq!(seams.act_scales.len(), dim / 32);
        assert_eq!(seams.act_fp8_sha256, sha256_hex(&seams.act_fp8));
        assert_eq!(seams.ffn_post.len(), HC_MULT);
        assert_eq!(seams.ffn_comb.len(), HC_MULT * HC_MULT);
        assert_eq!(
            seams.ffn_post_sha256,
            sha256_hex(&f32_le_bytes(&seams.ffn_post))
        );
        assert_eq!(
            seams.ffn_comb_sha256,
            sha256_hex(&f32_le_bytes(&seams.ffn_comb))
        );
        assert!(seams.ffn_in.iter().all(|value| value.is_finite()));
        assert!(seams.pre_ffn_hidden.iter().all(|value| value.is_finite()));
        assert!(seams.to_json()["note"].is_string());
        let error = layer0_ffn_host_seams(
            config,
            &hidden[..dim],
            &attn_pre_mix,
            &func,
            &base,
            &scale,
            &ffn_norm,
        )
        .expect_err("a short stream is refused");
        assert!(error.contains("post-attention stream"), "{error}");
    }

    #[test]
    fn the_attention_residual_seam_mixes_the_delta_onto_the_streams() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let mut hidden = Vec::with_capacity(hc * dim);
        for copy in 0..hc {
            for index in 0..dim {
                hidden.push((index as f32 * 0.001) + copy as f32 * 0.5);
            }
        }
        let predictor: Vec<f32> = (0..PREDICTOR_ROWS * hc * dim)
            .map(|index| (index % 97) as f32 * 0.0001)
            .collect();
        let base = vec![0.0f32; PREDICTOR_ROWS];
        let scale = vec![0.0f32; 3];
        let attn_norm = vec![1.0f32; dim];
        let entry = mhc_entry_host_seam(
            config,
            &hidden,
            &predictor,
            &base,
            &scale,
            &attn_norm,
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect("mhc entry");
        let delta = bf16_le_bytes(&vec![0.01f32; dim]);
        let seam = attention_residual_host_seam(config, &entry, &delta).expect("hc post");
        assert_eq!(seam.attn_out_sha256.len(), 64);
        assert_eq!(seam.out_sha256.len(), 64);
        assert_eq!(seam.residual_sha256, sha256_hex(&f32_le_bytes(&hidden)));
        assert_eq!(seam.stream_l2.len(), hc);
        assert!(seam.stream_l2.iter().all(|value| value.is_finite()));
        assert!(seam.out_l2.is_finite() && seam.out_l2 > 0.0);
        assert_eq!(seam.post_sha256, sha256_hex(&f32_le_bytes(&entry.post)));
        assert_eq!(seam.comb_sha256, sha256_hex(&f32_le_bytes(&entry.comb)));
        let error = attention_residual_host_seam(config, &entry, &delta[..dim])
            .expect_err("a short delta is refused");
        assert!(error.contains("one row is 5120"), "{error}");
    }

    #[test]
    fn native_rounding_names_every_supported_arm() {
        assert_eq!(
            NativeRounding::parse("").expect("empty is none"),
            NativeRounding::NONE
        );
        assert_eq!(
            NativeRounding::parse("none").expect("none"),
            NativeRounding::NONE
        );
        assert_eq!(
            NativeRounding::parse("combine").expect("combine").as_str(),
            "combine"
        );
        assert_eq!(NativeRounding::parse("mhc").expect("mhc").as_str(), "mhc");
        assert_eq!(
            NativeRounding::parse("shared").expect("shared").as_str(),
            "shared"
        );
        assert_eq!(
            NativeRounding::parse("combine,shared")
                .expect("combine and shared")
                .as_str(),
            "combine,shared"
        );
        let both = NativeRounding::parse("combine,mhc").expect("comma list");
        assert_eq!(both, NativeRounding::parse("both").expect("both"));
        assert_eq!(both.as_str(), "both");
        let all = NativeRounding::parse("both,shared").expect("all three terms");
        assert_eq!(all.as_str(), "combine,mhc,shared");
        assert_eq!(
            NativeRounding::parse("combine,mhc,shared")
                .expect("explicit all")
                .as_str(),
            "combine,mhc,shared"
        );
        assert_eq!(
            all,
            NativeRounding::parse("all,shared").expect("all and shared")
        );
        assert!(both.any() && all.any() && !NativeRounding::NONE.any());
        assert!(NativeRounding::parse("shared").expect("shared").any());
        assert_eq!(NativeRounding::default(), NativeRounding::NONE);
        assert!(NativeRounding::parse("none,combine").is_err());
        assert!(NativeRounding::parse("none,shared").is_err());
        assert!(NativeRounding::parse("fp32").is_err());
    }

    /// The default path through every rewritten seam must be the pre-wiring f32 arithmetic: the
    /// wrappers with the native switch off are bit-equal to the original functions. This is the
    /// host half of the default-path byte-identity argument; the logits bin itself is a device run
    /// that needs the checkpoint and four ranks, so it is re-run on the window as the default row.
    #[test]
    fn the_default_seams_are_the_unchanged_f32_arm() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let mut hidden = Vec::with_capacity(hc * dim);
        for copy in 0..hc {
            for index in 0..dim {
                hidden.push((index as f32 * 0.001) + copy as f32 * 0.5);
            }
        }
        let predictor: Vec<f32> = (0..PREDICTOR_ROWS * hc * dim)
            .map(|index| (index % 97) as f32 * 0.0001)
            .collect();
        let base = vec![0.0f32; PREDICTOR_ROWS];
        let scale = vec![0.0f32; 3];
        let attn_norm = vec![1.0f32; dim];
        let mix = [1.0f32, 0.0, 0.0, 0.0];
        let entry =
            mhc_entry_host_seam(config, &hidden, &predictor, &base, &scale, &attn_norm, &mix)
                .expect("default mhc entry");
        let entry_none = mhc_entry_host_seam_with(
            config, &hidden, &predictor, &base, &scale, &attn_norm, &mix, false,
        )
        .expect("none mhc entry");
        assert_eq!(entry, entry_none);
        let delta = bf16_le_bytes(&vec![0.01f32; dim]);
        let residual = attention_residual_host_seam(config, &entry, &delta)
            .expect("default attention residual");
        let residual_none = attention_residual_host_seam_with(config, &entry, &delta, false)
            .expect("none attention residual");
        assert_eq!(residual.out_sha256, residual_none.out_sha256);
        assert_eq!(residual.out, residual_none.out);
        assert_eq!(residual.attn_out_sha256, residual_none.attn_out_sha256);
        assert_eq!(residual.residual_sha256, residual_none.residual_sha256);
        let func = vec![0.0f32; PREDICTOR_ROWS * hc * dim];
        let seams = layer0_ffn_host_seams(
            config,
            &residual.out,
            &entry.pre,
            &func,
            &base,
            &scale,
            &attn_norm,
        )
        .expect("default ffn seams");
        let seams_none = layer0_ffn_host_seams_with(
            config,
            &residual.out,
            &entry.pre,
            &func,
            &base,
            &scale,
            &attn_norm,
            false,
        )
        .expect("none ffn seams");
        assert_eq!(seams.ffn_in_sha256, seams_none.ffn_in_sha256);
        assert_eq!(
            seams.pre_ffn_hidden_sha256,
            seams_none.pre_ffn_hidden_sha256
        );
        assert_eq!(seams.act_fp8_sha256, seams_none.act_fp8_sha256);
        assert_eq!(seams.act_scales_sha256, seams_none.act_scales_sha256);
    }

    /// The native mHC arm must actually move the stream at the released BF16 boundaries: hc_pre and
    /// the norm output come back bf16-representable and differ from the f32 arm on the same input.
    #[test]
    fn the_native_mhc_seams_round_the_stream_at_the_release_boundaries() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let mut hidden = Vec::with_capacity(hc * dim);
        for copy in 0..hc {
            for index in 0..dim {
                hidden.push((index as f32 * 0.001) + copy as f32 * 0.5);
            }
        }
        let predictor: Vec<f32> = (0..PREDICTOR_ROWS * hc * dim)
            .map(|index| (index % 97) as f32 * 0.0001)
            .collect();
        let base = vec![0.0f32; PREDICTOR_ROWS];
        let scale = vec![0.0f32; 3];
        let attn_norm = vec![1.0f32; dim];
        let mix = [1.0f32, 0.0, 0.0, 0.0];
        let f32_entry =
            mhc_entry_host_seam(config, &hidden, &predictor, &base, &scale, &attn_norm, &mix)
                .expect("f32 mhc entry");
        let native_entry = mhc_entry_host_seam_with(
            config, &hidden, &predictor, &base, &scale, &attn_norm, &mix, true,
        )
        .expect("native mhc entry");
        assert_ne!(f32_entry.attn_in_sha256, native_entry.attn_in_sha256);
        assert_ne!(f32_entry.normed_sha256, native_entry.normed_sha256);
        let widened_attn_in =
            bf16_from_le_bytes(&bf16_le_bytes(&native_entry.attn_in)).expect("bf16 decode");
        assert_eq!(widened_attn_in, native_entry.attn_in);
        let widened_normed =
            bf16_from_le_bytes(&bf16_le_bytes(&native_entry.normed)).expect("bf16 decode");
        assert_eq!(widened_normed, native_entry.normed);
        let delta = bf16_le_bytes(&vec![0.01f32; dim]);
        let f32_residual = attention_residual_host_seam(config, &native_entry, &delta)
            .expect("f32 attention residual");
        let native_residual =
            attention_residual_host_seam_with(config, &native_entry, &delta, true)
                .expect("native attention residual");
        assert_ne!(f32_residual.out_sha256, native_residual.out_sha256);
        let widened =
            bf16_from_le_bytes(&bf16_le_bytes(&native_residual.out)).expect("bf16 decode");
        assert_eq!(widened, native_residual.out);
    }

    #[test]
    fn the_window_row_round_trip_is_one_engine_row() {
        let mut row = vec![0.0f32; MLA_LATENT];
        for (index, slot) in row.iter_mut().enumerate() {
            *slot = (index as f32 * 0.03125).sin() * 11.0;
        }
        let (codes, scales) = window_row_bytes(&row).expect("window row");
        assert_eq!(codes.len(), WINDOW_CODE_BYTES);
        assert_eq!(scales.len(), WINDOW_SCALE_BYTES);
        let (codes_again, scales_again) = window_row_bytes(&row).expect("stable");
        assert_eq!(codes, codes_again);
        assert_eq!(scales, scales_again);
        assert!(
            codes.iter().any(|code| *code != 0),
            "a nonzero row must produce at least one nonzero code"
        );
        assert!(
            codes.iter().all(|code| *code != 0x7f),
            "the engine quantiser never emits the E4M3 NaN code"
        );
        for index in 0..MLA_LATENT {
            let value = sparkflash_kernels::mla_ref::window_kv_value(&codes, &scales, 0, index);
            assert!(value.is_finite(), "window row value {index} is not finite");
        }
    }

    #[test]
    fn the_window_row_round_trip_refuses_a_non_row() {
        let error = window_row_bytes(&[0.0f32; 32]).expect_err("32 values are not a row");
        assert!(error.contains("512"), "{error}");
        let mut row = vec![0.0f32; MLA_LATENT];
        row[7] = f32::NAN;
        let error = window_row_bytes(&row).expect_err("a NaN row is refused");
        assert!(error.contains("finite"), "{error}");
    }

    #[test]
    fn the_projection_weight_names_are_the_five_attention_leaves() {
        assert!(is_attention_projection_weight("layers.0.attn.wq_a.weight"));
        assert!(is_attention_projection_weight("layers.39.attn.wo_b.weight"));
        assert!(is_attention_projection_weight("mtp.2.attn.wq_b.weight"));
        assert!(!is_attention_projection_weight("layers.0.attn.wq_a.scale"));
        assert!(!is_attention_projection_weight(
            "layers.0.attn.q_norm.weight"
        ));
        assert!(!is_attention_projection_weight("layers.0.mlp.w1.weight"));
    }

    #[test]
    fn the_layer0_mhc_entry_seam_reproduces_the_released_reference() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        assert_eq!(hc, HC_MULT);
        // A constant hidden row keeps the reference arithmetic checkable by hand: the layer-0 seed
        // A = [1,0,0,0] must collapse the stream to exactly its first copy.
        let row = vec![2.0f32; dim];
        let mut stream = Vec::with_capacity(hc * dim);
        for _ in 0..hc {
            stream.extend_from_slice(&row);
        }
        let predictor = vec![0.0f32; PREDICTOR_ROWS * hc * dim];
        let base = vec![0.0f32; PREDICTOR_ROWS];
        let scale = [1.0f32, 1.0, 1.0];
        let norm = vec![1.0f32; dim];
        let seam = mhc_entry_host_seam(
            config,
            &stream,
            &predictor,
            &base,
            &scale,
            &norm,
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect("seam");

        assert_eq!(
            seam.attn_in, row,
            "the seed A=[1,0,0,0] selects stream copy 0"
        );
        let expected_pre = 0.5 + config.hc_eps as f32;
        assert!((seam.pre[0] - expected_pre).abs() < 1e-6, "{:?}", seam.pre);
        assert_eq!(seam.post, vec![1.0; HC_MULT], "2*sigmoid(0) = 1");
        for value in &seam.comb {
            assert!((value - 0.25).abs() < 1e-4, "uniform comb: {value}");
        }
        let expected_normed = 2.0 / (4.0 + config.norm_eps as f32).sqrt();
        for value in &seam.normed {
            assert!((value - expected_normed).abs() < 1e-5, "{value}");
        }
        assert_eq!(seam.scale, scale.to_vec());
        assert!(!seam.normed_sha256.is_empty());
        assert_ne!(seam.attn_in_sha256, seam.normed_sha256);

        // Deterministic and sensitive to the input: the digests are evidence, not decoration.
        let again = mhc_entry_host_seam(
            config,
            &stream,
            &predictor,
            &base,
            &scale,
            &norm,
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect("seam");
        assert_eq!(again, seam);
        let mut changed = stream.clone();
        // Copy 0 is the one the layer-0 seed collapses, so its change must move attn_in too.
        changed[0] += 1.0;
        let different = mhc_entry_host_seam(
            config,
            &changed,
            &predictor,
            &base,
            &scale,
            &norm,
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect("seam");
        assert_ne!(different.hidden_sha256, seam.hidden_sha256);
        assert_ne!(different.attn_in_sha256, seam.attn_in_sha256);
    }

    #[test]
    fn the_mhc_entry_seam_honours_the_cross_layer_pre_mix() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = HC_MULT;
        // Distinct copies so the selected copy is identifiable.
        let mut stream = Vec::with_capacity(hc * dim);
        for copy in 0..hc {
            for index in 0..dim {
                stream.push((copy as f32 + 1.0) * (index as f32 * 0.001));
            }
        }
        let predictor = vec![0.0f32; PREDICTOR_ROWS * hc * dim];
        let base = vec![0.0f32; PREDICTOR_ROWS];
        let scale = [1.0f32, 1.0, 1.0];
        let norm = vec![1.0f32; dim];
        let identity = [1.0f32, 0.0, 0.0, 0.0];
        let layer_zero =
            mhc_entry_host_seam(config, &stream, &predictor, &base, &scale, &norm, &identity)
                .expect("layer-0 seed");
        assert_eq!(
            layer_zero.attn_in,
            stream[..dim].to_vec(),
            "the identity seed collapses stream copy 0"
        );
        // A later layer passes the previous layer's FFN pre mix: copy 2 here.
        let cross_layer = [0.0f32, 0.0, 1.0, 0.0];
        let later = mhc_entry_host_seam(
            config,
            &stream,
            &predictor,
            &base,
            &scale,
            &norm,
            &cross_layer,
        )
        .expect("cross-layer mix");
        assert_eq!(
            later.attn_in,
            stream[2 * dim..3 * dim].to_vec(),
            "the cross-layer pre mix selects the copy it weights"
        );
        assert_ne!(later.attn_in_sha256, layer_zero.attn_in_sha256);
        let error = mhc_entry_host_seam(
            config,
            &stream,
            &predictor,
            &base,
            &scale,
            &norm,
            &[0.0f32; 3],
        )
        .expect_err("a short pre mix is refused");
        assert!(error.contains("attn_pre_mix"), "{error}");
    }

    #[test]
    fn the_layer0_mhc_entry_seam_refuses_geometry_that_does_not_match_the_released_model() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let stream = vec![0.0f32; HC_MULT * dim];
        let predictor = vec![0.0f32; PREDICTOR_ROWS * HC_MULT * dim];
        let base = vec![0.0f32; PREDICTOR_ROWS];
        let scale = [1.0f32, 1.0, 1.0];
        let norm = vec![1.0f32; dim];

        let short_stream = &stream[..HC_MULT * dim - 1];
        let error = mhc_entry_host_seam(
            config,
            short_stream,
            &predictor,
            &base,
            &scale,
            &norm,
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect_err("short stream");
        assert!(error.contains("hidden is"), "{error}");

        let error = mhc_entry_host_seam(
            config,
            &stream,
            &predictor[..predictor.len() - 1],
            &base,
            &scale,
            &norm,
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect_err("short predictor");
        assert!(error.contains("predictor is"), "{error}");

        let error = mhc_entry_host_seam(
            config,
            &stream,
            &predictor,
            &base[..PREDICTOR_ROWS - 1],
            &scale,
            &norm,
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect_err("short base");
        assert!(error.contains("base is"), "{error}");

        let error = mhc_entry_host_seam(
            config,
            &stream,
            &predictor,
            &base,
            &[1.0, 1.0],
            &norm,
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect_err("short scale");
        assert!(error.contains("scale is"), "{error}");

        let error = mhc_entry_host_seam(
            config,
            &stream,
            &predictor,
            &base,
            &scale,
            &norm[..dim - 1],
            &[1.0, 0.0, 0.0, 0.0],
        )
        .expect_err("short norm");
        assert!(error.contains("attn_norm is"), "{error}");
    }

    #[test]
    fn the_byte_codecs_are_exact_for_their_encodings() {
        assert_eq!(
            f32_from_le_bytes(&f32_le_bytes(&[1.0, -2.5, 0.0])).expect("f32"),
            vec![1.0, -2.5, 0.0]
        );
        assert!(
            f32_from_le_bytes(&[1, 2, 3])
                .expect_err("not whole f32")
                .contains("f32")
        );

        let bf16 = bf16_le_bytes(&[1.0, -2.0, 0.5, 0.0]);
        assert_eq!(bf16.len(), 8);
        assert_eq!(
            bf16_from_le_bytes(&bf16).expect("bf16"),
            vec![1.0, -2.0, 0.5, 0.0]
        );
        assert!(
            bf16_from_le_bytes(&[1, 2, 3])
                .expect_err("not whole bf16")
                .contains("bf16")
        );
    }

    #[test]
    fn the_plan_operand_layout_is_checked_against_the_bind_by_the_device_predicate() {
        use sparkflash_device::resident::{HeaderEntry, ResidentPlan};
        let entries = [
            ("layers.0.attn.wq_a.weight", vec![1280, 5120]),
            ("layers.0.attn.wkv.weight", vec![512, 5120]),
            ("layers.0.attn.wq_b.weight", vec![8192, 1280]),
            ("layers.0.attn.wo_a.weight", vec![2048, 4096]),
            ("layers.0.attn.wo_b.weight", vec![5120, 2048]),
        ]
        .into_iter()
        .map(|(name, shape)| HeaderEntry::new(name, "F8_E4M3", shape));
        let plan = ResidentPlan::plan(0, 4, entries).expect("the five layer-0 weights plan");
        let planned = attention_plan_from_plan(&plan);
        assert_eq!(planned.len(), 5);
        assert!(
            planned
                .iter()
                .all(|row| row.launchable && row.layout == "b12x")
        );
        assert_eq!(
            planned[0].b12x_geometry,
            Some((1280, 5120)),
            "wq_a tile geometry comes from the plan predicate"
        );
        assert!(planned[0].weight_bytes.is_some() && planned[0].scale_bytes.is_some());
        assert_eq!(planned[0].scale_bytes, Some(40 * 160));

        let b12x = (Deviceptr(0x1000), Deviceptr(0x2000));
        let mut pointers = BTreeMap::new();
        for (name, _kernel, _) in LAYER0_ATTENTION_OPERANDS {
            pointers.insert(name.to_string(), b12x);
        }
        let census = attention_stage_census(&pointers, &planned, Some(&plan));
        assert_eq!(
            census["plan_vs_bind_disagreements"], 0,
            "{:?}",
            census["disagreements"]
        );
        for row in census["rows"].as_array().expect("rows") {
            assert_eq!(row["plan_check"]["ok"], true, "{row}");
            assert_eq!(row["plan_check"]["planned"], "b12x");
        }

        // A stored pair on a plan-B12x tensor is the named refusal, not a guessed layout.
        pointers.insert(
            "layers.0.attn.wq_a.weight".to_string(),
            (Deviceptr(0x3000), Deviceptr(0x3000)),
        );
        let mismatch = attention_stage_census(&pointers, &planned, Some(&plan));
        assert_eq!(mismatch["plan_vs_bind_disagreements"], 1);
        let error = mismatch["disagreements"][0]["error"]
            .as_str()
            .expect("error string");
        assert!(error.contains("wq_a"), "{error}");
        assert!(error.contains("refusing to launch"), "{error}");
    }

    #[test]
    fn the_plan_dtype_is_the_authoritative_layout_and_the_bind_is_corroborated() {
        // The released plan says all five layer-0 attention projections are F8_E4M3 (kernel-lead's
        // header read: wq_a/wkv included, with E8M0 scale grids): all five are B12x-launchable.
        let fixture: Vec<(&str, &str)> = LAYER0_ATTENTION_OPERANDS
            .iter()
            .map(|(name, _kernel, _)| (*name, "F8_E4M3"))
            .collect();
        let plan = attention_plan_from_dtypes(fixture.iter().copied());
        assert_eq!(plan.len(), 5);
        assert!(
            plan.iter()
                .all(|row| row.layout == "b12x" && row.launchable)
        );

        let b12x = (Deviceptr(0x1000), Deviceptr(0x2000));
        let mut pointers = BTreeMap::new();
        for (name, _kernel, _) in LAYER0_ATTENTION_OPERANDS {
            pointers.insert(name.to_string(), b12x);
        }
        let census = attention_stage_census(&pointers, &plan, None);
        assert_eq!(census["launchable"], 5);
        assert_eq!(
            census["inconsistent"]
                .as_array()
                .expect("inconsistent")
                .len(),
            0
        );
        for row in census["rows"].as_array().expect("rows") {
            assert_eq!(row["layout_source"], "plan-dtype");
            assert_eq!(row["plan_dtype"], "F8_E4M3");
            assert_eq!(row["expected_layout"], "b12x");
            assert_eq!(row["consistent"], true);
        }

        // A bind that contradicts the plan is named, not silently launched.
        pointers.insert(
            "layers.0.attn.wq_a.weight".to_string(),
            (Deviceptr(0x3000), Deviceptr(0x3000)),
        );
        let mismatch = attention_stage_census(&pointers, &plan, None);
        assert_eq!(
            mismatch["inconsistent"],
            serde_json::json!(["layers.0.attn.wq_a.weight"])
        );
        assert_eq!(mismatch["launchable"], 4);
        assert!(mismatch["inconsistent_reason"].is_string());

        // Without a plan the census falls back to the bind pointer and says so.
        let unplanned = attention_stage_census(&pointers, &[], None);
        assert!(
            unplanned["rows"]
                .as_array()
                .expect("rows")
                .iter()
                .all(|row| row["layout_source"] == "bind-pointer")
        );
    }

    #[test]
    fn the_embedding_location_is_the_owner_rank_and_the_row_inside_its_shard() {
        assert_eq!(embedding_location(0), (0, 0));
        assert_eq!(embedding_location(1), (0, 1));
        assert_eq!(embedding_location(32_319), (0, 32_319));
        assert_eq!(embedding_location(32_320), (1, 0));
        assert_eq!(embedding_location(129_279), (3, 32_319));
        assert_eq!(HIDDEN, 5120);
        assert_eq!(VOCAB_PER_RANK * 4, 129_280);
    }

    #[test]
    fn the_hidden_digest_is_a_plain_sha256_of_the_bf16_bytes() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_staged_stored_cubin_passes_the_preflight_and_the_standalone_stage_names_the_missing_arena()
    {
        let dir = tempfile::tempdir().expect("tempdir");
        let kernels = tempfile::tempdir().expect("kernels");
        std::fs::write(
            kernels
                .path()
                .join(format!("{STORED_COMBINE_KERNEL}.cubin")),
            b"staged",
        )
        .expect("write");
        let req = request(dir.path(), Some(kernels.path()));
        // kernel-lead's ABI module is in this build: the host preflight is now a pass, and the only
        // thing standing between it and a launch is the arena epoch the chain supplies.
        let plan = preflight(&req).expect("stored ABI ready and cubin staged");
        assert!(plan.selection.is_launchable());
        assert_eq!(plan.launch_census["launch_compatible"], true);
        assert_eq!(plan.launch_census["abi_ready"], true);
        let outcome = token_stage(&req);
        assert!(outcome.is_refused(), "a standalone stage has no arena");
        let error = outcome.error().expect("refusal");
        assert!(error.contains("arena epoch"), "{error}");
    }

    #[test]
    fn a_present_pre_added_cubin_does_not_rescue_a_stored_geometry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let kernels = tempfile::tempdir().expect("kernels");
        std::fs::write(
            kernels
                .path()
                .join(format!("{PRE_ADDED_COMBINE_KERNEL}.cubin")),
            b"pre-added",
        )
        .expect("write");
        let req = request(dir.path(), Some(kernels.path()));
        let outcome = token_stage(&req);
        assert!(outcome.is_refused());
        assert!(
            outcome
                .error()
                .expect("refusal")
                .contains("stored-layout combine kernel unavailable")
        );
    }

    #[test]
    fn a_bind_geometry_that_belongs_to_neither_layout_refuses_with_the_arithmetic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut req = request(dir.path(), None);
        req.combine_bytes = Some((DEFAULT_COMBINE_ROWS * 289, DEFAULT_COMBINE_ROWS * 18));
        let outcome = token_stage(&req);
        let error = outcome.error().expect("refusal");
        assert!(error.contains("bind and the selector disagree"), "{error}");
        assert!(error.contains("288+18"), "{error}");
    }

    #[test]
    fn the_expected_argv_fragment_names_the_preflight_flag_and_the_kernel_dir() {
        let fragment = expected_argv_fragment();
        for needle in [
            "--forward",
            PREFLIGHT_FLAG,
            "--kernel-cubin-dir",
            "--token-ids",
        ] {
            assert!(fragment.contains(needle), "{fragment} lacks {needle}");
        }
    }

    #[test]
    fn the_kernel_set_used_by_the_preflight_is_a_plain_directory_scan() {
        let dir = tempfile::tempdir().expect("kernels");
        let set = CombineKernelSet::scan(dir.path());
        assert!(set.was_scanned());
        assert!(set.stored_cubin().is_none());
    }

    #[test]
    fn the_head_side_payload_round_trips_and_rejects_tampering() {
        let vocab = crate::tp4_tail::VOCAB_PER_RANK;
        let chunks = crate::tp4_tail::CHUNKS_PER_RANK;
        let mut certified = vec![0u8; vocab];
        certified[0] = 1;
        certified[vocab - 1] = 1;
        certified[vocab / 2] = 1;
        let block_max: Vec<f32> = (0..chunks).map(|index| index as f32 * 0.25).collect();
        let block_argmax: Vec<u32> = (0..chunks).map(|index| index as u32 * 1024 + 7).collect();
        let mut payload = Vec::with_capacity(crate::tp4_tail::SIDE_BYTES);
        payload.extend_from_slice(&certified);
        payload.extend_from_slice(&3u32.to_le_bytes());
        for value in &block_max {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        for value in &block_argmax {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(payload.len(), crate::tp4_tail::SIDE_BYTES);
        let side = parse_head_side(&payload).expect("payload parses");
        assert_eq!(side.certified, certified);
        assert_eq!(side.certified_count, 3);
        assert_eq!(side.block_max, block_max);
        assert_eq!(side.block_argmax, block_argmax);

        let mut wrong_length = payload.clone();
        wrong_length.pop();
        assert!(
            parse_head_side(&wrong_length).is_err(),
            "short payload accepted"
        );

        let mut wrong_count = payload;
        let count_at = vocab;
        wrong_count[count_at..count_at + 4].copy_from_slice(&9u32.to_le_bytes());
        assert!(
            parse_head_side(&wrong_count).is_err(),
            "count that disagrees with the flags accepted"
        );
    }

    #[test]
    fn the_block_argmax_digest_input_is_little_endian() {
        let bytes = u32_le_bytes(&[0x0102_0304, 0xffff_ffff]);
        assert_eq!(bytes, vec![0x04, 0x03, 0x02, 0x01, 0xff, 0xff, 0xff, 0xff]);
    }

    #[test]
    fn the_head_side_payload_is_the_sum_of_its_parts() {
        let expected =
            crate::tp4_tail::VOCAB_PER_RANK + 4 + crate::tp4_tail::CHUNKS_PER_RANK * 4 * 2;
        assert_eq!(crate::tp4_tail::SIDE_BYTES, expected);
        assert_eq!(
            crate::tp4_tail::SIDE_BYTES % 2,
            0,
            "the side payload is gathered as u16 words"
        );
    }

    /// The captured m1y16 activation fixture: rank 0's pre-head hidden at position 129, the
    /// 5120-wide row the widest swapped projection reduces over.
    fn m1y16_pre_head_hidden_pos129() -> Vec<f32> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/m1y16-pre-head-hidden-pos129.f32");
        let bytes = std::fs::read(&path).expect("the captured m1y16 activation fixture");
        assert_eq!(bytes.len(), HIDDEN * 4, "one row of {HIDDEN} f32 values");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    }

    /// One synthetic B12x weight for the fixture test: a deterministic E4M3 code grid plus its
    /// ue8m0 32x32-block scales over the released wq_a geometry ([1280] x [5120]), returned in
    /// both the engine's plain [rows][inner] code layout and the B12x tiled layout, with the one
    /// scale grid both paths index identically (engine ScaleGrid::Block32 at cols == inner is the
    /// same [row/32 * inner/32 + col/32] grid B12x uses).
    fn synthetic_b12x_weight(cols: usize, inner: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use sparkflash_kernels::mxfp8_ref::{
            b12x_scale_bytes, b12x_scale_index, b12x_tile_weights,
        };
        assert_eq!(inner % 32, 0, "one full MX block row");
        let mut plain = vec![0u8; cols * inner];
        let mut scales = vec![0u8; b12x_scale_bytes(cols, inner)];
        for row in 0..cols {
            for col in 0..inner {
                // Finite E4M3 patterns only: skip the NaN encodings 0x7f and 0xff.
                let mut code = (row * 37 + col * 11 + 13) % 254;
                if code >= 127 {
                    code += 1;
                }
                plain[row * inner + col] = code as u8;
                scales[b12x_scale_index(row, col, inner)] = 120 + ((row + col) % 8) as u8;
            }
        }
        let tiled = b12x_tile_weights(&plain, cols, inner);
        (plain, tiled, scales)
    }

    #[test]
    fn the_native_dense_launch_arithmetic_equals_the_engine_coded_matmul_on_the_captured_row() {
        // The host half of the native-dense wiring check: the kernel's arithmetic twin
        // (sparkflash_kernels::mxfp8_ref::dense_gemm, documented as exactly what
        // b12x_mxfp8_dense_sm121.cu computes) against the engine's native coded matmul
        // (linalg::matmul_f8_scaled), on the captured m1y16 activation row and a synthetic B12x
        // weight of the released wq_a shape.
        //
        // The comparison is bit-equal after the kernel's final bf16 round: both paths decode the
        // same E4M3 codes and ue8m0 scales, multiply activation-value by weight-value in f32 and
        // accumulate in the same ascending-K order (kernel: 32-value MX blocks ascending, values
        // ascending inside a block, which is the same order as the engine's ascending-col
        // accumulator). The launch itself cannot execute on a host without a GPU, so this test
        // proves the arithmetic both sides promise; the device-side equality is an m1w29 check.
        let hidden = m1y16_pre_head_hidden_pos129();
        let (activation_codes, activation_scales) =
            quantise_native_activation(&hidden, "host twin").expect("quantise the captured row");
        assert_eq!(activation_codes.len(), HIDDEN);
        assert_eq!(activation_scales.len(), HIDDEN / 32);
        let activation_values: Vec<f32> = activation_codes
            .iter()
            .enumerate()
            .map(|(index, code)| {
                sparkflash_kernels::m4_quant::fp8_e4m3_decode(*code)
                    * sparkflash_kernels::m4_quant::ue8m0_decode(activation_scales[index / 32])
            })
            .collect();

        let (cols, inner) = (1280usize, HIDDEN);
        let (weight_codes, weight_tiled, weight_scales) = synthetic_b12x_weight(cols, inner);
        let kernel_out = sparkflash_kernels::mxfp8_ref::dense_gemm(
            &activation_codes,
            &activation_scales,
            &weight_tiled,
            &weight_scales,
            1,
            cols,
            inner,
        );
        let mut engine_out = vec![0f32; cols];
        sparkflash_engine::linalg::matmul_f8_scaled(
            &mut engine_out,
            &activation_values,
            &weight_codes,
            &weight_scales,
            sparkflash_engine::linalg::ScaleGrid::Block32,
            cols,
            inner,
            1,
        )
        .expect("the engine coded matmul accepts the fixture geometry");
        let mismatched: Vec<(usize, u16, u16)> = engine_out
            .iter()
            .zip(&kernel_out)
            .enumerate()
            .filter_map(|(index, (value, kernel))| {
                let engine = sparkflash_kernels::m4_quant::bf16_bits(*value);
                (engine != *kernel).then_some((index, engine, *kernel))
            })
            .collect();
        assert!(
            mismatched.is_empty(),
            "the native arithmetic twin and the engine coded matmul disagree at {} of {cols} outputs, first (index, engine, kernel): {:?}",
            mismatched.len(),
            mismatched.first()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_evidence_blocks_name_the_native_dense_kernels_and_their_activation_quant_census() {
        // The evidence names are the QA/Lead contract for m1w29: the kernel set must be readable
        // from tp4-forward-embed/tp4-token without trusting a binary. The structs are cuda-gated,
        // so this test only compiles with --features cuda; it never opens a driver.
        let projections = Layer0Projections {
            cubin_sha256: "cubin".to_string(),
            activation_codes_sha256: "codes".to_string(),
            activation_scales_sha256: "scales".to_string(),
            activation_code_bytes: HIDDEN,
            activation_scale_bytes: HIDDEN / 32,
            wq_a_latent_sha256: "q".to_string(),
            wkv_latent_sha256: "kv".to_string(),
            wq_a_l2: 1.0,
            wkv_l2: 2.0,
            wq_a_latent_bytes: vec![0u8; 1280 * 2],
            wkv_latent_bytes: vec![0u8; 512 * 2],
            launches: 2,
        };
        let json = projections.to_json();
        assert_eq!(
            json["kernel"],
            sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
            "the projections stage records the symbol it actually launched"
        );
        assert_eq!(json["arithmetic_family"], NATIVE_DENSE_KERNEL);
        assert_eq!(json["activation_quant"]["block"], 32);
        assert_eq!(json["activation_quant"]["code_bytes"], HIDDEN);
        assert_eq!(json["activation_quant"]["scale_bytes"], HIDDEN / 32);
        assert_eq!(json["activation_quant"]["codes_sha256"], "codes");
        assert!(
            json["exact"]
                .as_str()
                .unwrap_or("")
                .contains("native dense family"),
            "projection exact string: {json}"
        );

        let query = Layer0QueryPath {
            q_norm_sha256: "norm".to_string(),
            q_pre_rope_sha256: "pre".to_string(),
            q_post_rope_sha256: "post".to_string(),
            q_l2: 3.0,
            heads_per_rank: 16,
            q_head_width: 512,
            rope_head_dim: 64,
            q_norm_width: 1280,
            position: 129,
            cubin_sha256: "cubin".to_string(),
            activation_codes_sha256: "codes".to_string(),
            activation_scales_sha256: "scales".to_string(),
            activation_code_bytes: 1280,
            activation_scale_bytes: 40,
            q_post_rope_bf16: vec![0u8; 16 * 512 * 2],
            q_latent_bf16: vec![0.0f32; 1280],
        };
        let json = query.to_json();
        assert_eq!(
            json["kernel"],
            sparkflash_kernels::mxfp8_gemv_launch::KERNEL,
            "the query path records the symbol it actually launched"
        );
        assert_eq!(json["arithmetic_family"], NATIVE_DENSE_KERNEL);
        assert_eq!(json["wq_b"]["activation_quant"]["code_bytes"], 1280);
        assert_eq!(json["wq_b"]["activation_quant"]["scale_bytes"], 40);

        let output = Layer0AttentionOutput {
            groups: 2,
            group_inner: 8192,
            world: 4,
            o_groups_global: 8,
            heads_per_rank: 16,
            head_width: 512,
            o_lora_rank: 1024,
            wo_a_cubin_sha256: "wo_a".to_string(),
            wo_a_out_sha256: "wo_a_out".to_string(),
            wo_a_out_l2: 4.0,
            wo_b_cubin_sha256: "wo_b".to_string(),
            wo_b_activation_codes_sha256: "codes".to_string(),
            wo_b_activation_scales_sha256: "scales".to_string(),
            wo_b_activation_code_bytes: 2048,
            wo_b_activation_scale_bytes: 64,
            wo_b_partial_sha256: "partial".to_string(),
            wo_b_partial_l2: 5.0,
            reduced_sha256: "reduced".to_string(),
            reduced_l2: 6.0,
            reduce_calls: 1,
            reduce_sequence: 1,
            reduced_bf16: vec![0u8; HIDDEN * 2],
        };
        let json = output.to_json();
        assert_eq!(json["wo_a"]["kernel"], NATIVE_WO_A_KERNEL);
        assert_eq!(
            json["wo_b"]["kernel"],
            sparkflash_kernels::mxfp8_gemv_launch::KERNEL
        );
        assert_eq!(json["wo_b"]["arithmetic_family"], NATIVE_DENSE_KERNEL);
        assert_eq!(json["wo_b"]["activation_quant"]["code_bytes"], 2048);
        assert_eq!(json["wo_b"]["activation_quant"]["scale_bytes"], 64);
        assert!(
            json["exact"]
                .as_str()
                .unwrap_or("")
                .contains("native dense family"),
            "attention output exact string: {json}"
        );
    }

    #[test]
    fn the_native_dense_parity_spec_parses_layer_and_optional_position() {
        assert_eq!(
            NativeDenseParitySpec::parse("0"),
            Ok(NativeDenseParitySpec {
                layer: 0,
                position: None,
            })
        );
        assert_eq!(
            NativeDenseParitySpec::parse("12:129"),
            Ok(NativeDenseParitySpec {
                layer: 12,
                position: Some(129),
            })
        );
        assert!(NativeDenseParitySpec::parse("").is_err());
        assert!(NativeDenseParitySpec::parse("layer3").is_err());
        assert!(NativeDenseParitySpec::parse("3:").is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_native_dense_parity_geometry_derivation_recovers_the_four_operand_shapes() {
        // The loop's per-rank native dense geometry: wq_a and wkv replicated, wq_b a row shard,
        // wo_b an inner shard. The derivation uses only the operand byte counts, so each shape must
        // round-trip from (activation codes, activation scales, weight payload, weight scales).
        for (cols, inner) in [
            (1280usize, 5120usize),
            (512, 5120),
            (8192, 1280),
            (5120, 2048),
        ] {
            let payload = sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes(cols, inner);
            let scales = sparkflash_kernels::mxfp8_ref::b12x_scale_bytes(cols, inner);
            let derived = derive_native_dense_geometry(inner, inner / 32, payload, scales)
                .expect("the operand bytes must imply their launch geometry");
            assert_eq!(derived, (cols, inner), "cols {cols} inner {inner}");
        }
        let payload = sparkflash_kernels::mxfp8_ref::b12x_buffer_bytes(1280, 5120);
        let scales = sparkflash_kernels::mxfp8_ref::b12x_scale_bytes(1280, 5120);
        // One code per inner value, one scale per 32: a wrong scale count cannot pass.
        assert!(derive_native_dense_geometry(5120, 159, payload, scales).is_err());
        // The weight payload must be exactly the B12x buffer the scale grid derives.
        assert!(derive_native_dense_geometry(5120, 160, payload + 32, scales).is_err());
        assert!(derive_native_dense_geometry(0, 0, payload, scales).is_err());
        assert!(derive_native_dense_geometry(5121, 160, payload, scales).is_err());
        assert!(derive_native_dense_geometry(5120, 160, payload, 0).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn the_native_dense_parity_bf16_comparison_counts_word_mismatches() {
        // 1.0, 1.0 as bf16-le words.
        let one_one = [0x80u8, 0x3f, 0x80, 0x3f];
        assert_eq!(
            native_parity_compare_bf16(&one_one, &one_one),
            (0, 2, 0.0, false)
        );
        // 1.0, 1.5: one word differs by 0.5.
        let one_one_half = [0x80u8, 0x3f, 0xc0, 0x3f];
        let (mismatches, slots, max_abs, length) =
            native_parity_compare_bf16(&one_one, &one_one_half);
        assert_eq!((mismatches, slots, length), (1, 2, false));
        assert!((max_abs - 0.5).abs() < 1e-6, "max_abs {max_abs}");
        let (mismatches, slots, _, length) = native_parity_compare_bf16(&one_one, &one_one[..2]);
        assert_eq!((mismatches, slots, length), (0, 1, true));
        assert_eq!(
            native_parity_byte_mismatches(&[1, 2, 3], &[1, 9, 3]),
            (1, false)
        );
        assert_eq!(
            native_parity_byte_mismatches(&[1, 2], &[1, 2, 3]),
            (0, true)
        );
    }

    // Phase 2C host tests T1-T5 (prepared in device-lead/m2-2c-prep-20261021, applied here) plus
    // the lean head-record path test. They exercise the same private helpers the loop uses.
    #[allow(clippy::too_many_arguments)]
    fn cache_layer_seam_raw(
        attn_fn: &[f32],
        attn_base: &[f32],
        attn_scale: &[f32],
        attn_norm_bytes: Vec<u8>,
        ffn_fn: &[f32],
        ffn_base: &[f32],
        ffn_scale: &[f32],
        ffn_norm_bytes: Vec<u8>,
    ) -> LayerSeamRaw {
        LayerSeamRaw {
            attn_fn: f32_le_bytes(attn_fn),
            attn_base: f32_le_bytes(attn_base),
            attn_scale: f32_le_bytes(attn_scale),
            attn_norm: attn_norm_bytes,
            ffn_fn: f32_le_bytes(ffn_fn),
            ffn_base: f32_le_bytes(ffn_base),
            ffn_scale: f32_le_bytes(ffn_scale),
            ffn_norm: ffn_norm_bytes,
        }
    }

    fn cache_small_layer_seam_config() -> sparkflash_engine::config::ModelConfig {
        // The cache build only needs dim and hc_mult; the released config keeps the hc geometry.
        let mut config =
            sparkflash_engine::config::ModelConfig::v41_flash().expect("released config");
        config.dim = 64;
        config
    }

    fn cache_seam_plan(
        config: &sparkflash_engine::config::ModelConfig,
        dtype: &str,
        n_layers: usize,
    ) -> sparkflash_device::resident::ResidentPlan {
        use sparkflash_device::resident::{HeaderEntry, ResidentPlan};
        let mut entries: Vec<HeaderEntry> = Vec::new();
        for layer in 0..n_layers {
            entries.push(HeaderEntry::new(
                format!("layers.{layer}.attn_norm.weight"),
                dtype,
                vec![config.dim],
            ));
            entries.push(HeaderEntry::new(
                format!("layers.{layer}.ffn_norm.weight"),
                dtype,
                vec![config.dim],
            ));
        }
        ResidentPlan::plan(0, 4, entries).expect("seam plan")
    }

    fn cache_production_seam_vectors() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let predictor: Vec<f32> = (0..PREDICTOR_ROWS * hc * dim)
            .map(|index| (index % 89) as f32 * 0.0002)
            .collect();
        let base: Vec<f32> = (0..PREDICTOR_ROWS)
            .map(|index| (index as f32) * 0.125)
            .collect();
        let scale = vec![1.0f32, 0.5, 0.25];
        let norm: Vec<f32> = (0..dim)
            .map(|index| 1.0 + (index % 7) as f32 * 0.01)
            .collect();
        (predictor, base, scale, norm)
    }

    // T1: the cache decode equals the per-call decode for both released norm encodings.
    #[test]
    fn the_layer_seam_cache_decode_reproduces_the_per_call_decode() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let (attn_fn, attn_base, attn_scale, attn_norm) = cache_production_seam_vectors();
        let ffn_fn: Vec<f32> = (0..PREDICTOR_ROWS * hc * dim)
            .map(|index| (index % 97) as f32 * 0.0001)
            .collect();
        let ffn_base: Vec<f32> = (0..PREDICTOR_ROWS)
            .map(|index| 0.5 - (index as f32) * 0.01)
            .collect();
        let ffn_scale = vec![2.0f32, 1.0, 0.5];
        let ffn_norm: Vec<f32> = (0..dim)
            .map(|index| 1.0 - (index % 5) as f32 * 0.001)
            .collect();
        let per_layer_bytes = (attn_fn.len()
            + attn_fn.len()
            + attn_base.len()
            + attn_scale.len()
            + attn_norm.len()
            + ffn_fn.len()
            + ffn_fn.len()
            + ffn_base.len()
            + ffn_scale.len()
            + ffn_norm.len())
            * 4;

        // F32 norms: the plan can carry either encoding for these two tensors.
        let raw = cache_layer_seam_raw(
            &attn_fn,
            &attn_base,
            &attn_scale,
            f32_le_bytes(&attn_norm),
            &ffn_fn,
            &ffn_base,
            &ffn_scale,
            f32_le_bytes(&ffn_norm),
        );
        let tensors = decode_layer_seam_tensors(raw, config).expect("decode F32 norms");
        assert_eq!(tensors.attn_fn, attn_fn);
        assert_eq!(tensors.attn_base, attn_base);
        assert_eq!(tensors.attn_scale, attn_scale);
        assert_eq!(tensors.attn_norm, attn_norm);
        assert_eq!(tensors.ffn_fn, ffn_fn);
        assert_eq!(tensors.ffn_base, ffn_base);
        assert_eq!(tensors.ffn_scale, ffn_scale);
        assert_eq!(tensors.ffn_norm, ffn_norm);
        assert_eq!(
            tensors.attn_predictor_sha256,
            sha256_hex(&f32_le_bytes(&attn_fn))
        );
        assert_eq!(
            tensors.attn_base_sha256,
            sha256_hex(&f32_le_bytes(&attn_base))
        );
        assert_eq!(
            tensors.attn_norm_sha256,
            sha256_hex(&f32_le_bytes(&attn_norm))
        );
        assert_eq!(tensors.decoded_bytes(), per_layer_bytes);

        // BF16 norms: decode to the same f32 values the per-call bf16_from_le_bytes produces, and the
        // digest is over the decoded f32 row, exactly as mhc_entry_host_seam_with hashes it today.
        let raw_bf16 = cache_layer_seam_raw(
            &attn_fn,
            &attn_base,
            &attn_scale,
            bf16_le_bytes(&attn_norm),
            &ffn_fn,
            &ffn_base,
            &ffn_scale,
            bf16_le_bytes(&ffn_norm),
        );
        let tensors_bf16 = decode_layer_seam_tensors(raw_bf16, config).expect("decode BF16 norms");
        let expected_attn_norm =
            bf16_from_le_bytes(&bf16_le_bytes(&attn_norm)).expect("bf16 decode");
        let expected_ffn_norm = bf16_from_le_bytes(&bf16_le_bytes(&ffn_norm)).expect("bf16 decode");
        assert_eq!(tensors_bf16.attn_norm, expected_attn_norm);
        assert_eq!(tensors_bf16.ffn_norm, expected_ffn_norm);
        assert_eq!(
            tensors_bf16.attn_norm_sha256,
            sha256_hex(&f32_le_bytes(&expected_attn_norm))
        );
        assert_eq!(tensors_bf16.decoded_bytes(), per_layer_bytes);
    }

    // T2: reusing the cached digests changes nothing in the MhcEntrySeam, on both rounding arms.
    #[test]
    fn the_cached_digest_seam_is_the_computed_digest_seam() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let (predictor, base, scale, attn_norm) = cache_production_seam_vectors();
        let mut hidden = Vec::with_capacity(hc * dim);
        for copy in 0..hc {
            for index in 0..dim {
                hidden.push(((index * 7 % 101) as f32 * 0.002) - copy as f32 * 0.25);
            }
        }
        let mix = [1.0f32, 0.0, 0.0, 0.0];
        let raw = cache_layer_seam_raw(
            &predictor,
            &base,
            &scale,
            f32_le_bytes(&attn_norm),
            &predictor,
            &base,
            &scale,
            f32_le_bytes(&attn_norm),
        );
        let tensors = decode_layer_seam_tensors(raw, config).expect("decode");
        let digests = tensors.digests();
        assert_eq!(digests.predictor, sha256_hex(&f32_le_bytes(&predictor)));
        assert_eq!(digests.base, sha256_hex(&f32_le_bytes(&base)));
        assert_eq!(digests.norm, sha256_hex(&f32_le_bytes(&attn_norm)));
        for native in [false, true] {
            let computed = mhc_entry_host_seam_with(
                config,
                &hidden,
                &tensors.attn_fn,
                &tensors.attn_base,
                &tensors.attn_scale,
                &tensors.attn_norm,
                &mix,
                native,
            )
            .expect("computed-digest seam");
            let cached = mhc_entry_host_seam_with_digests(
                config,
                &hidden,
                &tensors.attn_fn,
                &tensors.attn_base,
                &tensors.attn_scale,
                &tensors.attn_norm,
                &mix,
                native,
                Some(&digests),
                None,
            )
            .expect("cached-digest seam");
            let transposed = mhc_entry_host_seam_with_digests(
                config,
                &hidden,
                &tensors.attn_fn,
                &tensors.attn_base,
                &tensors.attn_scale,
                &tensors.attn_norm,
                &mix,
                native,
                Some(&digests),
                Some(&tensors.attn_fn_transposed),
            )
            .expect("transposed-projection seam");
            assert_eq!(
                computed, transposed,
                "the interleaved projection is not the ordered one, native_mhc {native}"
            );
            assert_eq!(computed, cached, "native_mhc {native}");
            assert_eq!(cached.predictor_sha256, digests.predictor);
            assert_eq!(cached.base_sha256, digests.base);
            assert_eq!(cached.norm_sha256, digests.norm);
        }
    }

    // T3: the refactored cache-hit seam entry points equal the direct public seam calls.
    #[test]
    fn the_cached_layer_seams_are_the_direct_seams() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let dim = config.dim;
        let hc = config.hc_mult;
        let (predictor, base, scale, norm) = cache_production_seam_vectors();
        let raw = cache_layer_seam_raw(
            &predictor,
            &base,
            &scale,
            f32_le_bytes(&norm),
            &predictor,
            &base,
            &scale,
            f32_le_bytes(&norm),
        );
        let tensors = decode_layer_seam_tensors(raw, config).expect("decode");
        let mut stream = Vec::with_capacity(hc * dim);
        for copy in 0..hc {
            for index in 0..dim {
                stream.push((copy as f32 + 1.0) * (index as f32 * 0.001));
            }
        }
        let mix = [0.25f32, 0.25, 0.25, 0.25];
        let dir = tempfile::tempdir().expect("tempdir");
        let mut req = request(dir.path(), None);
        req.config = Some(config);

        let (entry, reads) =
            run_layer_mhc_entry(&tensors, &req, &stream, &mix).expect("cached mhc entry");
        assert_eq!(reads, 0, "a cache hit performs no arena reads");
        let direct = mhc_entry_host_seam(
            config,
            &stream,
            &tensors.attn_fn,
            &tensors.attn_base,
            &tensors.attn_scale,
            &tensors.attn_norm,
            &mix,
        )
        .expect("direct mhc entry");
        assert_eq!(entry, direct);

        let delta = bf16_le_bytes(&vec![0.01f32; dim]);
        let residual = attention_residual_host_seam(config, &entry, &delta).expect("residual");
        let cached_ffn =
            run_layer_ffn_seams(&tensors, &req, &residual, &entry.pre).expect("cached ffn seams");
        let direct_ffn = layer0_ffn_host_seams(
            config,
            &residual.out,
            &entry.pre,
            &tensors.ffn_fn,
            &tensors.ffn_base,
            &tensors.ffn_scale,
            &tensors.ffn_norm,
        )
        .expect("direct ffn seams");
        assert_eq!(cached_ffn.ffn_pre_sha256, direct_ffn.ffn_pre_sha256);
        assert_eq!(cached_ffn.ffn_post_sha256, direct_ffn.ffn_post_sha256);
        assert_eq!(cached_ffn.ffn_comb_sha256, direct_ffn.ffn_comb_sha256);
        assert_eq!(cached_ffn.ffn_in_sha256, direct_ffn.ffn_in_sha256);
        assert_eq!(
            cached_ffn.pre_ffn_hidden_sha256,
            direct_ffn.pre_ffn_hidden_sha256
        );
        assert_eq!(cached_ffn.act_fp8_sha256, direct_ffn.act_fp8_sha256);
        assert_eq!(cached_ffn.act_scales_sha256, direct_ffn.act_scales_sha256);
        assert_eq!(cached_ffn.ffn_pre, direct_ffn.ffn_pre);
        assert_eq!(cached_ffn.ffn_post, direct_ffn.ffn_post);
        assert_eq!(cached_ffn.ffn_comb, direct_ffn.ffn_comb);
        assert_eq!(cached_ffn.ffn_in, direct_ffn.ffn_in);
        assert_eq!(cached_ffn.pre_ffn_hidden, direct_ffn.pre_ffn_hidden);
        assert_eq!(cached_ffn.act_fp8, direct_ffn.act_fp8);
        assert_eq!(cached_ffn.act_scales, direct_ffn.act_scales);
    }

    // T4a: the build reads exactly the eight tensors once per layer, with the expected names and sizes.
    #[test]
    fn the_seam_cache_build_reads_each_tensor_once_per_layer() {
        let config = cache_small_layer_seam_config();
        let n_layers = 2;
        let plan = cache_seam_plan(&config, "F32", n_layers);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut req = request(dir.path(), None);
        req.config = Some(&config);
        req.resident_plan = Some(&plan);

        let predictor_bytes = PREDICTOR_ROWS * config.hc_mult * config.dim * 4;
        let norm_bytes = config.dim * 4;
        let mut expected: Vec<(String, usize)> = Vec::new();
        for layer in 0..n_layers {
            expected.push((format!("layers.{layer}.hc_attn_fn"), predictor_bytes));
            expected.push((format!("layers.{layer}.hc_attn_base"), PREDICTOR_ROWS * 4));
            expected.push((format!("layers.{layer}.hc_attn_scale"), 3 * 4));
            expected.push((format!("layers.{layer}.attn_norm.weight"), norm_bytes));
            expected.push((format!("layers.{layer}.hc_ffn_fn"), predictor_bytes));
            expected.push((format!("layers.{layer}.hc_ffn_base"), PREDICTOR_ROWS * 4));
            expected.push((format!("layers.{layer}.hc_ffn_scale"), 3 * 4));
            expected.push((format!("layers.{layer}.ffn_norm.weight"), norm_bytes));
        }
        let mut calls: Vec<(String, usize)> = Vec::new();
        let mut reader = |name: &str, want: usize| -> Result<Vec<u8>, String> {
            calls.push((name.to_string(), want));
            Ok(vec![0u8; want])
        };
        let cache = build_layer_seam_cache_with_reader(&mut reader, &req, n_layers).expect("cache");
        assert_eq!(cache.layers.len(), n_layers);
        assert_eq!(cache.reads, 8 * n_layers);
        assert_eq!(calls, expected, "one read of each seam tensor per layer");
        // Each sublayer stages its predictor twice: the decoded [24][20480] rows that the digests and
        // the ordered fallback read, and the [20480][24] transpose the interleaved projection reads.
        let per_layer = 2 * (2 * (predictor_bytes / 4) + PREDICTOR_ROWS + 3 + config.dim) * 4;
        assert_eq!(cache.bytes, per_layer * n_layers);

        // BF16 norm dtype: the planned encoding still decides the requested norm byte length.
        let plan_bf16 = cache_seam_plan(&config, "BF16", n_layers);
        let mut req_bf16 = request(dir.path(), None);
        req_bf16.config = Some(&config);
        req_bf16.resident_plan = Some(&plan_bf16);
        let mut bf16_calls: Vec<(String, usize)> = Vec::new();
        let mut reader_bf16 = |name: &str, want: usize| -> Result<Vec<u8>, String> {
            bf16_calls.push((name.to_string(), want));
            Ok(vec![0u8; want])
        };
        let cache_bf16 = build_layer_seam_cache_with_reader(&mut reader_bf16, &req_bf16, n_layers)
            .expect("bf16 cache");
        assert_eq!(cache_bf16.reads, 8 * n_layers);
        assert!(
            bf16_calls
                .iter()
                .filter(|(name, _)| name.ends_with(".attn_norm.weight"))
                .all(|(_, want)| *want == config.dim * 2)
        );
        // Same decoded f32 memory whether the source norm encoding is F32 or BF16.
        assert_eq!(cache_bf16.bytes, cache.bytes);
    }

    // T4b: fail fast before the loop, with the same message the per-call seam produces today.
    #[test]
    fn the_seam_cache_build_refuses_before_the_loop_with_the_per_call_message() {
        let config = cache_small_layer_seam_config();
        let dir = tempfile::tempdir().expect("tempdir");
        let mut req = request(dir.path(), None);
        req.config = Some(&config);
        // No resident plan: the norm encoding is unknown and the per-call seam refuses by name.
        let mut reads = 0usize;
        let mut reader = |_name: &str, want: usize| -> Result<Vec<u8>, String> {
            reads += 1;
            Ok(vec![0u8; want])
        };
        let error = build_layer_seam_cache_with_reader(&mut reader, &req, 1)
            .expect_err("a missing rank plan must refuse");
        assert!(
            error.contains("the rank plan does not carry layers.0.attn_norm.weight"),
            "{error}"
        );
        assert!(error.contains("refusing to guess"), "{error}");
        assert_eq!(reads, 0, "the dtype is checked before any tensor read");
    }

    // T5: the production per-layer cache is 7,905,496 B - two predictors, each staged decoded and
    // transposed - so the 40-layer cache is 316,219,840 B.
    #[test]
    fn the_seam_cache_memory_is_the_production_per_layer_bytes() {
        let config = sparkflash_engine::config::ModelConfig::v41_flash_cached();
        let n_layers = 1;
        let plan = cache_seam_plan(config, "F32", n_layers);
        let dir = tempfile::tempdir().expect("tempdir");
        let mut req = request(dir.path(), None);
        req.config = Some(config);
        req.resident_plan = Some(&plan);
        let mut reader =
            |_name: &str, want: usize| -> Result<Vec<u8>, String> { Ok(vec![0u8; want]) };
        let cache = build_layer_seam_cache_with_reader(&mut reader, &req, n_layers).expect("cache");
        assert_eq!(cache.reads, 8);
        let per_layer = 2
            * (2 * PREDICTOR_ROWS * config.hc_mult * config.dim + PREDICTOR_ROWS + 3 + config.dim)
            * 4;
        assert_eq!(per_layer, 7_905_496);
        assert_eq!(cache.bytes, per_layer);
    }

    // The lean evidence nit: under --forward-lean only the last-position alias is written, so the
    // head record must name that alias instead of the per-position file lean deliberately skips.
    #[test]
    fn the_lean_head_record_names_the_written_alias() {
        let seam = Path::new("/ev/head-input/pre_head_hidden-pos137.f32");
        let alias = Path::new("/ev/head-input/pre_head_hidden.f32");
        assert_eq!(
            head_seam_input_file(false, seam, alias),
            seam.display().to_string()
        );
        assert_eq!(
            head_seam_input_file(true, seam, alias),
            alias.display().to_string()
        );
    }
}

#[cfg(test)]
mod evidence_switch_tests {
    use super::*;

    // M2/C10: the default and --forward-lean rows must hash nothing on the layer path. The switch
    // is thread-local (like the probe's state), so this test can turn it off without disturbing any
    // other test in the binary, and the counter proves the off path takes no sha256 at all.
    #[test]
    fn the_evidence_chain_is_off_unless_the_environment_asks_for_it_and_the_off_path_hashes_nothing()
     {
        // The shipped run default: no variable means the run loop selects off.
        assert!(!evidence_enabled_from_env(None));
        assert!(!evidence_enabled_from_env(Some("0")));
        assert!(evidence_enabled_from_env(Some("1")));
        assert!(evidence_enabled_from_env(Some("true")));
        assert!(evidence_enabled_from_env(Some("yes")));

        set_evidence_record(false);
        assert!(!evidence_record_enabled());
        let before = evidence_hash_calls();
        assert_eq!(evidence_sha256(b"evidence-only bytes"), "");
        assert_eq!(evidence_hash_calls(), before);

        set_evidence_record(true);
        assert!(evidence_record_enabled());
        assert_eq!(
            evidence_sha256(b"evidence-only bytes"),
            sha256_hex(b"evidence-only bytes")
        );
        assert_eq!(evidence_hash_calls(), before + 1);
        set_evidence_record(false);
    }

    // The f32-row digest helper must make the byte materialisation conditional, not only the hash:
    // a default/--forward-lean run serialises nothing, and the hash counter stays put; an evidence
    // run hashes exactly the bytes the inline call used to build.
    #[test]
    fn the_f32_row_digest_serialises_only_in_evidence_mode() {
        set_evidence_record(false);
        let values = vec![1.0f32, -2.5, 0.0, 123.25];
        let before = evidence_hash_calls();
        assert_eq!(evidence_f32_sha256(&values), "");
        assert_eq!(evidence_hash_calls(), before);
        set_evidence_record(true);
        assert_eq!(
            evidence_f32_sha256(&values),
            sha256_hex(&f32_le_bytes(&values))
        );
        assert_eq!(evidence_hash_calls(), before + 1);
        set_evidence_record(false);
    }

    // C10 records the layer-chain entry only in evidence mode, so the six evidence-only device
    // readbacks in the MoE layer (router weights and scores, the gate and up outputs, the weighted
    // SWiGLU codes and scales) exist only when that record is on. The copy sites must ask exactly
    // this question; asking it here proves the switch they see is the record switch, and that a
    // lean run never issues the drain.
    #[test]
    fn the_moe_evidence_readbacks_exist_only_in_evidence_mode() {
        set_evidence_record(false);
        assert!(!layer_evidence_readbacks());
        set_evidence_record(true);
        assert!(layer_evidence_readbacks());
        set_evidence_record(false);
        assert!(!layer_evidence_readbacks());
    }

    // The two run-scope switches must not tangle: the launch path is async by default, the
    // environment opt-out is the only way back to the drained path, and switching the evidence
    // chain on or off must leave the launch path alone.
    #[test]
    fn the_launch_path_and_the_evidence_chain_are_independent_switches() {
        use sparkflash_kernels::cuda_driver as driver;
        driver::set_async_launches(false);
        assert!(!driver::async_launches());
        // No SPARKFLASH_SYNC_LAUNCHES in the test environment, so the run-scope call restores async.
        assert!(driver::enable_forward_async_launches());
        assert!(driver::async_launches());

        set_evidence_record(false);
        driver::set_async_launches(false);
        assert!(!driver::async_launches());
        set_evidence_record(true);
        assert!(!driver::async_launches());

        driver::set_async_launches(true);
        set_evidence_record(false);
        assert!(driver::async_launches());
    }

    /// One recorded [CountingSource::read_ranges_paged] call: the tensor name and its ranges.
    type PagedCall = (String, Vec<(u64, usize)>);

    /// A [sparkflash_engine::weights::WeightSource] over in-memory bytes with read counters, so the
    /// Engram overlay can be checked without a checkpoint: a resident read must not touch it at all.
    struct CountingSource {
        held: BTreeMap<String, (sparkflash_engine::tensor::TensorSpec, Vec<u8>)>,
        reads: std::sync::atomic::AtomicUsize,
        paged: std::sync::atomic::AtomicUsize,
        /// Every paged call in order, with the name and ranges it asked for: the warm-up test reads
        /// the ranges back out of here.
        paged_calls: std::sync::Mutex<Vec<PagedCall>>,
    }

    impl CountingSource {
        fn new() -> Self {
            Self {
                held: BTreeMap::new(),
                reads: std::sync::atomic::AtomicUsize::new(0),
                paged: std::sync::atomic::AtomicUsize::new(0),
                paged_calls: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn put(
            &mut self,
            name: &str,
            dtype: sparkflash_engine::dtype::DType,
            shape: &[usize],
            bytes: &[u8],
        ) {
            self.held.insert(
                name.to_owned(),
                (
                    sparkflash_engine::tensor::TensorSpec {
                        name: name.to_owned(),
                        dtype,
                        shape: shape.to_vec(),
                    },
                    bytes.to_vec(),
                ),
            );
        }
    }

    impl sparkflash_engine::weights::WeightSource for CountingSource {
        fn names(&self) -> Vec<String> {
            self.held.keys().cloned().collect()
        }

        fn spec(
            &self,
            name: &str,
        ) -> Result<sparkflash_engine::tensor::TensorSpec, sparkflash_engine::weights::WeightError>
        {
            self.held
                .get(name)
                .map(|(spec, _)| spec.clone())
                .ok_or_else(|| sparkflash_engine::weights::WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read(&self, name: &str) -> Result<Vec<u8>, sparkflash_engine::weights::WeightError> {
            self.reads
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.held
                .get(name)
                .map(|(_, bytes)| bytes.clone())
                .ok_or_else(|| sparkflash_engine::weights::WeightError::Missing {
                    name: name.to_owned(),
                })
        }

        fn read_ranges_paged(
            &self,
            name: &str,
            ranges: &[(u64, usize)],
        ) -> Option<Result<Vec<u8>, sparkflash_engine::weights::WeightError>> {
            self.paged
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.paged_calls
                .lock()
                .expect("paged calls")
                .push((name.to_owned(), ranges.to_vec()));
            let wanted: usize = ranges.iter().map(|(_, length)| length).sum();
            Some(
                self.held
                    .get(name)
                    .map(|(_, bytes)| bytes[..wanted].to_vec())
                    .ok_or_else(|| sparkflash_engine::weights::WeightError::Missing {
                        name: name.to_owned(),
                    }),
            )
        }
    }

    /// The overlay is the whole fix for the per-position copy: the resident matrices are read once
    /// and then borrowed, and every other name - including the paged Engram tables - still reaches
    /// the wrapped source.
    #[test]
    fn the_engram_overlay_holds_the_wkv_matrices_and_delegates_everything_else() {
        use sparkflash_engine::dtype::DType;
        use sparkflash_engine::weights::WeightSource;
        let mut inner = CountingSource::new();
        inner.put(
            "layers.1.engram.wkv.weight",
            DType::F8E4M3,
            &[2, 2],
            &[1, 2, 3, 4],
        );
        inner.put("layers.1.engram.wkv.scale", DType::F8E8M0, &[1, 1], &[7]);
        inner.put(
            "layers.1.engram.embed.weight",
            DType::F8E4M3,
            &[1, 2],
            &[5, 6],
        );

        let names = vec![
            "layers.1.engram.wkv.weight".to_owned(),
            "layers.1.engram.wkv.scale".to_owned(),
        ];
        let overlay = EngramResidentSource::preload(&inner, &names).expect("the matrices preload");
        assert_eq!(
            inner.reads.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "each resident matrix is read exactly once"
        );

        for (name, expected) in [
            ("layers.1.engram.wkv.weight", &[1u8, 2, 3, 4][..]),
            ("layers.1.engram.wkv.scale", &[7u8][..]),
        ] {
            assert_eq!(
                overlay.read_view(name).expect("a resident view result"),
                Some(expected),
                "{name} is not served as a borrow"
            );
            assert_eq!(
                overlay.read(name).expect("a resident copy").as_slice(),
                expected
            );
        }
        assert_eq!(
            inner.reads.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "a resident name reached the wrapped source"
        );

        assert_eq!(
            overlay
                .read_view("layers.1.engram.wkv.weight")
                .expect("a resident view result"),
            Some(&[1u8, 2, 3, 4][..])
        );
        assert_eq!(
            overlay
                .read_view("nothing")
                .expect("an absent name is a delegated miss"),
            None
        );
        assert_eq!(
            overlay
                .read("layers.1.engram.embed.weight")
                .expect("a delegated read"),
            vec![5u8, 6]
        );
        assert_eq!(
            inner.reads.load(std::sync::atomic::Ordering::Relaxed),
            3,
            "the delegated read is the wrapped one"
        );
        assert!(overlay.spec("layers.1.engram.embed.weight").is_ok());
        assert!(overlay.spec("nothing").is_err());
        assert!(
            overlay
                .names()
                .contains(&"layers.1.engram.embed.weight".to_owned())
        );
        let paged = overlay
            .read_ranges_paged("layers.1.engram.embed.weight", &[(0, 2)])
            .expect("the overlay serves the paged Engram path")
            .expect("the paged bytes");
        assert_eq!(paged, vec![5u8, 6]);
        assert_eq!(
            inner.paged.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "the paged read was not delegated"
        );
    }

    /// The resident list is exactly each Engram layer's wkv codes and grid - the two names the seam
    /// reads on every call.
    #[test]
    fn the_engram_resident_names_are_each_layers_wkv_codes_and_grid() {
        let mut config = sparkflash_engine::config::ModelConfig::tiny();
        config.engram_layer_ids = vec![1, 14];
        assert_eq!(
            engram_weight_names(&config),
            vec![
                "layers.1.engram.wkv.weight".to_owned(),
                "layers.1.engram.wkv.scale".to_owned(),
                "layers.14.engram.wkv.weight".to_owned(),
                "layers.14.engram.wkv.scale".to_owned(),
            ]
        );
    }

    /// The warm-up is exactly the rows the loop will look up: the union over the run's positions of
    /// the same hash rows, in ascending id order, at the table row offsets and widths the paged read
    /// uses, plus the constant gate weights - and nothing else.
    #[test]
    fn the_engram_warm_up_reads_exactly_the_rows_the_run_will_look_up() {
        use sparkflash_engine::dtype::DType;
        use sparkflash_engine::forward::engram::{hash_columns, ngram_hashes};
        let mut config = sparkflash_engine::config::ModelConfig::tiny();
        config.engram_layer_ids = vec![1];
        let tokens = [5u32, 7, 9, 11, 13];
        let total_positions = 4usize;
        let width = config.engram_head_dim;
        let scale_width = width.div_ceil(32);
        let rows = config.engram_rows(1).expect("the layer owns a table");
        let mut source = CountingSource::new();
        source.put(
            "layers.1.engram.embed.weight",
            DType::F8E4M3,
            &[rows, width],
            &vec![0u8; rows * width],
        );
        source.put(
            "layers.1.engram.embed.scale",
            DType::F8E8M0,
            &[rows, scale_width],
            &vec![0u8; rows * scale_width],
        );
        source.put("layers.1.engram.q_weight", DType::BF16, &[1], &[0, 0]);
        source.put("layers.1.engram.k_weight", DType::BF16, &[1], &[0, 0]);

        // The rows the loop will read, derived here from the same public hash function.
        let columns = hash_columns(&config);
        let mut hashes = Vec::new();
        let mut wanted = std::collections::BTreeSet::new();
        for position in 0..total_positions {
            ngram_hashes(&config, &tokens[..=position], position, &mut hashes)
                .expect("the run's hash row");
            wanted.extend(hashes[..columns].iter().map(|&id| id as usize));
        }
        assert!(!wanted.is_empty() && wanted.len() <= columns * total_positions);

        warm_engram_rows(&config, &tokens, total_positions, &source).expect("the warm-up");
        let calls = source.paged_calls.lock().expect("paged calls");
        assert_eq!(calls.len(), 2, "one weight read and one scale read");
        assert_eq!(calls[0].0, "layers.1.engram.embed.weight");
        assert_eq!(calls[1].0, "layers.1.engram.embed.scale");
        let want_weight: Vec<(u64, usize)> = wanted
            .iter()
            .map(|&id| (id as u64 * width as u64, width))
            .collect();
        let want_scale: Vec<(u64, usize)> = wanted
            .iter()
            .map(|&id| (id as u64 * scale_width as u64, scale_width))
            .collect();
        assert_eq!(calls[0].1, want_weight);
        assert_eq!(calls[1].1, want_scale);
        drop(calls);
        assert_eq!(
            source.reads.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "the two constant gate weights are warmed"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn reduce_switch_parses_only_the_two_paths_and_a_typo_stays_on_the_reference() {
        assert_eq!(ReduceSwitch::parse("roce"), ReduceSwitch::Roce);
        assert_eq!(ReduceSwitch::parse(" RDMA "), ReduceSwitch::Roce);
        assert_eq!(ReduceSwitch::parse("tcp"), ReduceSwitch::Tcp);
        assert_eq!(ReduceSwitch::parse(""), ReduceSwitch::Tcp);
        // A typo keeps the reference path and from_env reports it; it never silently picks the
        // one-shot, and the row's line always names the path that actually ran.
        assert_eq!(ReduceSwitch::parse("roc"), ReduceSwitch::Tcp);
    }
    /// The loop's quantiser must feed the seams the two-step reference's bytes: narrow the row to
    /// BF16, then run the engine's quantiser, on the three shipped activation widths.
    #[test]
    fn the_loop_activation_quantiser_matches_the_two_step_reference() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 2048.0 - 0.25
        };
        for count in [1280usize, 2048, 5120] {
            let values: Vec<f32> = (0..count).map(|_| next()).collect();
            let narrowed: Vec<f32> = values
                .iter()
                .map(|value| sparkflash_engine::quant::bf16_round(*value))
                .collect();
            let expected = sparkflash_engine::quant::quantize_activation_fp8(&narrowed)
                .expect("the reference quantises");
            let (codes, scales) =
                quantise_native_activation(&values, "test").expect("the loop quantises");
            assert_eq!(codes, expected.codes(), "codes differ at {count} values");
            assert_eq!(scales, expected.scales(), "scales differ at {count} values");
        }
    }
}
