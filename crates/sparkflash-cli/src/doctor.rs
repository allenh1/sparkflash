//! `sparkflash doctor`: the DGX Spark preflight gate (specification section 6.1).
//!
//! # The specification says seven and enumerates five
//!
//! Section 6.1 is titled "Seven-Point DGX Spark Preflight Verification" and points at a
//! seven-point RoCE checklist, but its own numbered list stops at five. This module implements
//! exactly those five and deliberately does not invent the missing two. The discrepancy is
//! printed by `--help` and appears in every report, so a green run is never mistaken
//! for a complete seven-point verification.
//!
//! # The five checks
//!
//! 1. Driver and kernel: driver >= `580.178.04`, driver `580.142` rejected outright,
//!    kernel >= `6.17.0-1018` or `7.0.0-1019-nvidia-64k`, nvidia-uvm 64K packing active.
//! 2. Dual-rail ConnectX-7 link speed and MTU: both halves at `200000 Mb/s` with MTU `9000`,
//!    confirmed by a do-not-fragment probe (`ping -M do -s 8972`).
//! 3. Dual-rail RDMA: aggregate cross-node bandwidth >= `180 Gbps` and 10 KiB one-shot
//!    All-Reduce latency < `22 us`.
//! 4. SM clock cap and headless memory: clocks lockable to `[0, 2200] MHz` and
//!    `MemAvailable >= 114 GiB`.
//! 5. Local NVMe Engram shards: `engram-l1-r{rank}of4.bin` and `engram-l14-r{rank}of4.bin`
//!    present, correctly sized, 512-byte aligned and matching the packer manifest's SHA256.
//!
//! # Everything is a probe
//!
//! No check touches the machine directly: each reads the world through one of five traits
//! ([`DriverProbe`], [`LinkProbe`], [`RdmaProbe`], [`GpuProbe`],
//! [`ShardProbe`]). The verdict, the thresholds, the ordering and the remediation text are
//! therefore unit-testable on a laptop with no cluster, no GPU and no NVMe.
//!
//! # Read-only, and off by default
//!
//! The default invocation runs no hardware probe at all: the five checks report
//! "not applicable" and the command exits zero. Hardware probes run only when the operator
//! opts in with `--local` (this node) or `--nodes` (this node plus the cluster), and no
//! command that changes state - `nvidia-smi -lgc` is the only one - is executed without
//! `--apply`.

use std::cmp::Ordering;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::Parser;
use serde_json::{Value, json};

/// The one specification section this command implements.
pub const SPEC_SECTION: &str = "6.1";

/// Number of checks the specification's heading claims ("Seven-Point").
pub const SPECIFIED_CHECK_COUNT: usize = 7;

/// Number of checks the specification actually enumerates, and the number implemented here.
pub const IMPLEMENTED_CHECK_COUNT: usize = 5;

/// One-line statement of the specification's internal contradiction, printed in the report.
pub const SPEC_DISCREPANCY: &str = "specification 6.1 is titled 'Seven-Point' but enumerates five checks; doctor implements the five that are written down and does not invent the missing two";

// ---------------------------------------------------------------------------------------------
// Thresholds. These are the gate: changing one changes what the cluster must look like.
// ---------------------------------------------------------------------------------------------

/// Minimum acceptable NVIDIA driver version.
pub const MIN_DRIVER_VERSION: &str = "580.178.04";

/// Driver versions that are rejected even though they compare above or near the minimum.
///
/// `580.142` is called out by the specification because it throttles ConnectX-7 to
/// 13 Gbps. Note that it also compares *below* [`MIN_DRIVER_VERSION` component-wise
/// (142 < 178), so this list is a second, explicit line of defence rather than the only one.
pub const REJECTED_DRIVER_VERSIONS: [&str; 1] = ["580.142"];

/// Minimum kernel base version for the 6.x branch.
pub const MIN_KERNEL_BASE: [u64; 3] = [6, 17, 0];

/// Minimum kernel ABI number on the 6.x branch.
pub const MIN_KERNEL_ABI: u64 = 1018;

/// Minimum kernel base version for the 7.x branch.
pub const ALT_MIN_KERNEL_BASE: [u64; 3] = [7, 0, 0];

/// Minimum kernel ABI number on the 7.x branch.
pub const ALT_MIN_KERNEL_ABI: u64 = 1019;

/// Required negotiated link speed per ConnectX-7 half, in Mb/s.
pub const LINK_SPEED_MBPS: u64 = 200_000;

/// Required MTU on both ConnectX-7 halves.
pub const LINK_MTU: u32 = 9000;

/// ICMP payload used by the do-not-fragment probe: 8972 + 28 bytes of IP/ICMP header = MTU 9000.
pub const DF_PROBE_PAYLOAD_BYTES: usize = 8972;

/// Minimum aggregate dual-rail cross-node bandwidth, in Gbps.
pub const MIN_AGGREGATE_RDMA_GBPS: f64 = 180.0;

/// Payload of the one-shot All-Reduce latency gate.
pub const ONESHOT_AR_BYTES: usize = 10 * 1024;

/// Maximum acceptable 10 KiB one-shot All-Reduce latency, in microseconds (strictly less than).
pub const MAX_ONESHOT_AR_LATENCY_US: f64 = 22.0;

/// Lowest SM clock in the permitted lock window, in MHz.
pub const MIN_SM_CLOCK_MHZ: u64 = 0;

/// Highest SM clock in the permitted lock window, in MHz.
pub const MAX_SM_CLOCK_MHZ: u64 = 2200;

/// Minimum `MemAvailable` per node before weights are loaded, in GiB.
pub const MIN_MEM_AVAILABLE_GIB: u64 = 114;

/// Alignment every Engram shard file is padded to, for O_DIRECT reads.
pub const O_DIRECT_ALIGNMENT: u64 = 512;

/// Default directory holding the local Engram shards and the packer manifest.
pub const DEFAULT_ENGRAM_DIR: &str = "/var/lib/sparkflash/engram";

/// Name of the packer manifest written next to the shards (M2 `sparkflash-engram`).
pub const MANIFEST_NAME: &str = "manifest.json";

/// The two ConnectX-7 halves, as (netdev, rdma device) pairs.
pub const RAILS: [Rail; 2] = [
    Rail {
        netdev: "enp1s0f0np0",
        rdma: "rocep1s0f0",
    },
    Rail {
        netdev: "enP2p1s0f0np0",
        rdma: "roceP2p1s0f0",
    },
];

/// One ConnectX-7 half: the Ethernet netdev and the RDMA device on top of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rail {
    /// Ethernet interface name, as it appears to `ip link` and `ethtool`.
    pub netdev: &'static str,
    /// RDMA device name, as it appears to `ibv_devinfo`.
    pub rdma: &'static str,
}

// ---------------------------------------------------------------------------------------------
// Probe error type
// ---------------------------------------------------------------------------------------------

/// Why a probe could not produce a reading.
///
/// The distinction matters: [`ProbeError::NotApplicable` means the operator did not ask
/// for the check (no `--nodes`, no `--local`), so the check is reported as "not
/// applicable" and does not fail the run. [`ProbeError::Unavailable` means the operator
/// did ask, but the reading could not be taken - a missing tool, an unreadable file, a probe that
/// ran and disagreed - and a preflight gate must treat that as a failure rather than a pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeError {
    /// The check does not apply here; report it as not applicable.
    NotApplicable(String),
    /// The check applies but the probe could not be taken; report it as a failure.
    Unavailable(String),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbeError::NotApplicable(message) => write!(f, "not applicable: {message}"),
            ProbeError::Unavailable(message) => write!(f, "unavailable: {message}"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// Result of a single probe reading.
pub type Probe<T> = Result<T, ProbeError>;

// ---------------------------------------------------------------------------------------------
// Probe traits: one per domain named in the task
// ---------------------------------------------------------------------------------------------

/// Where a 64K-packing switch would live, if the driver exposed one.
pub const UVM_PARAMETERS_DIR: &str = "/sys/module/nvidia_uvm/parameters";

/// One RDMA device as the kernel reports it.
///
/// Used to find the rails that are actually up, and to tell a disconnected cable apart from a
/// specification that names the wrong function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RailStatus {
    /// RDMA device name, e.g. `rocep1s0f1`.
    pub rdma: String,
    /// Ethernet netdev on top of it, when the kernel exposes one.
    pub netdev: Option<String>,
    /// Port state, e.g. `ACTIVE` or `DOWN`.
    pub state: String,
    /// Port rate in Mb/s, when the kernel reports one.
    pub speed_mbps: Option<u64>,
    /// PCI address without the function, e.g. `0000:01:00`, so sibling functions of one card
    /// can be grouped when working out which function the specification meant.
    pub pci: Option<String>,
}

impl RailStatus {
    /// Whether the kernel reports the port as up.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state.to_ascii_uppercase().contains("ACTIVE")
    }

    /// Whether the port is up at the speed the specification requires.
    #[must_use]
    pub fn is_fast_active(&self) -> bool {
        self.is_active()
            && self
                .speed_mbps
                .is_some_and(|speed| speed >= LINK_SPEED_MBPS)
    }

    /// Whether this device is one of the two the specification names.
    #[must_use]
    pub fn is_spec_rail(&self, rail: Rail) -> bool {
        self.rdma == rail.rdma || self.netdev.as_deref() == Some(rail.netdev)
    }

    /// One-line description for the report.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "{}/{} {} {}",
            self.rdma,
            self.netdev.as_deref().unwrap_or("(no netdev)"),
            self.state,
            self.speed_mbps.map_or_else(
                || "(rate unknown)".to_owned(),
                |speed| format!("{speed} Mb/s")
            )
        )
    }
}

/// Finds an ACTIVE, fast sibling function of the same PCI device.
///
/// This is the difference between "the cable is unplugged" and "the specification names the
/// wrong function": on real GB10 hardware the functions the specification names are siblings of
/// the ones that carry the 200 Gb/s fabric.
#[must_use]
pub fn fast_sibling<'a>(inventory: &'a [RailStatus], of: &RailStatus) -> Option<&'a RailStatus> {
    let pci = of.pci.as_deref()?;
    inventory.iter().find(|other| {
        other.rdma != of.rdma && other.pci.as_deref() == Some(pci) && other.is_fast_active()
    })
}

/// Driver, kernel and nvidia-uvm readings (`/proc` and `nvidia-smi`).
pub trait DriverProbe {
    /// Loaded NVIDIA driver version, e.g. `580.178.04`.
    fn driver_version(&self) -> Probe<String>;
    /// Kernel release, e.g. `6.17.0-1018-nvidia-64k`.
    fn kernel_release(&self) -> Probe<String>;
    /// Whether nvidia-uvm is packing 64K pages, with what the probe actually saw.
    fn uvm_page_packing(&self) -> Probe<UvmPacking>;
}

/// Ethernet-level readings for the ConnectX-7 rails (`ethtool`, sysfs, `ping`).
pub trait LinkProbe {
    /// Every RDMA device on this node with its netdev, port state and rate (`/sys/class/infiniband`).
    fn rail_inventory(&self) -> Probe<Vec<RailStatus>>;
    /// Negotiated speed of `netdev`, in Mb/s.
    fn link_speed_mbps(&self, netdev: &str) -> Probe<u64>;
    /// Configured MTU of `netdev`.
    fn mtu(&self, netdev: &str) -> Probe<u32>;
    /// Do-not-fragment probe to `peer` with `payload_bytes`; returns a human detail on success.
    fn df_probe(&self, netdev: &str, peer: &str, payload_bytes: usize) -> Probe<String>;
}

/// One peer's measured cross-node write bandwidth, in Gbps.
///
/// Check 3 gates on the worst peer rather than on an aggregate: a mean over four ranks hides the
/// one link that is degraded, which is exactly the failure the gate exists to catch.
#[derive(Debug, Clone, PartialEq)]
pub struct PeerBandwidth {
    /// The peer the reading belongs to, as named in --nodes.
    pub peer: String,
    /// Dual-rail write bandwidth measured toward that peer, in Gbps.
    pub gbps: f64,
}

/// RDMA bandwidth and latency readings (ibverbs / the RoCEnante probe).
pub trait RdmaProbe {
    /// Dual-rail cross-node write bandwidth toward every peer in `peers`, each in Gbps, averaged
    /// over `seconds`.
    fn dual_rail_write_gbps(&self, peers: &[String], seconds: u64) -> Probe<Vec<PeerBandwidth>>;
    /// 10 KiB one-shot All-Reduce latency in microseconds, measured from `rank` in a world made of
    /// `rank` plus `peers`.
    fn oneshot_allreduce_us(&self, rank: usize, peers: &[String], bytes: usize) -> Probe<f64>;
}

/// GPU clock and host memory readings (`nvidia-smi`, `/proc/meminfo`).
pub trait GpuProbe {
    /// The lockable SM clock window, in MHz.
    fn sm_clock_range_mhz(&self) -> Probe<ClockRange>;
    /// Requests a clock lock. Only ever called when `--apply` and `--lock-sm-clocks` are both
    /// passed: locking clocks mutates a running node and must never be a default.
    fn lock_sm_clocks(&self, min_mhz: u64, max_mhz: u64) -> Probe<String>;
    /// `MemAvailable`, in GiB.
    fn mem_available_gib(&self) -> Probe<u64>;
}

/// Filesystem and manifest readings for the local Engram shards.
pub trait ShardProbe {
    /// Length of `path` in bytes.
    fn file_len(&self, path: &Path) -> Probe<u64>;
    /// Lower-case hex SHA256 of `path`.
    fn sha256(&self, path: &Path) -> Probe<String>;
    /// Mount backing `path`.
    fn mount(&self, path: &Path) -> Probe<MountInfo>;
    /// The packer manifest in `dir`.
    fn manifest(&self, dir: &Path) -> Probe<EngramManifest>;
}

/// State of nvidia-uvm 64K page packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UvmPagePacking {
    /// 64K packing is active.
    Enabled,
    /// The module is loaded but 64K packing is off.
    Disabled,
    /// The probe found no readable knob; the operator must confirm by hand.
    Unknown,
}

impl fmt::Display for UvmPagePacking {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UvmPagePacking::Enabled => write!(f, "64K packing active"),
            UvmPagePacking::Disabled => write!(f, "64K packing inactive"),
            UvmPagePacking::Unknown => write!(f, "no 64K-packing knob found"),
        }
    }
}

/// What the nvidia-uvm packing probe saw, so the report can say whether the node exposes a
/// knob at all instead of implying the operator missed something.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UvmPacking {
    /// The verdict.
    pub state: UvmPagePacking,
    /// How many module parameters the node exposes, when the directory was readable.
    pub parameters_seen: Option<usize>,
    /// Names of anything that looked like a packing switch.
    pub candidates: Vec<String>,
}

/// A lockable SM clock window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockRange {
    /// Lowest lockable SM clock, in MHz.
    pub min_mhz: u64,
    /// Highest lockable SM clock, in MHz.
    pub max_mhz: u64,
}

/// Where a path lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountInfo {
    /// Block device or pseudo-device backing the path, e.g. `/dev/nvme0n1p2`.
    pub source: String,
    /// Filesystem type, e.g. `ext4` or `nfs4`.
    pub fstype: String,
}

/// One file's entry in the packer manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestEntry {
    /// File name as recorded in the manifest.
    pub file: String,
    /// SHA256 of the complete file, when the manifest records one.
    pub sha256: Option<String>,
    /// Complete file length in bytes, when the manifest records one.
    pub file_bytes: Option<u64>,
}

/// The packer manifest, reduced to what the preflight needs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EngramManifest {
    /// Entries the manifest describes, in no particular order.
    pub entries: Vec<ManifestEntry>,
}

impl EngramManifest {
    /// Finds the manifest entry for `file_name`, comparing basenames only.
    #[must_use]
    pub fn entry(&self, file_name: &str) -> Option<&ManifestEntry> {
        self.entries.iter().find(|entry| {
            Path::new(&entry.file)
                .file_name()
                .is_some_and(|name| name == file_name)
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Probe bundle
// ---------------------------------------------------------------------------------------------

/// The five probe domains, as trait objects.
pub struct Probes {
    /// Driver, kernel and uvm readings.
    pub driver: Box<dyn DriverProbe>,
    /// Link speed, MTU and do-not-fragment probes.
    pub link: Box<dyn LinkProbe>,
    /// RDMA bandwidth and latency.
    pub rdma: Box<dyn RdmaProbe>,
    /// GPU clocks and host memory.
    pub gpu: Box<dyn GpuProbe>,
    /// Engram shards and manifest.
    pub shards: Box<dyn ShardProbe>,
}

impl Probes {
    /// Every probe reports "not applicable": the dev-box default.
    #[must_use]
    pub fn not_applicable(reason: &str) -> Self {
        Self {
            driver: Box::new(NotApplicableDriver(reason.to_owned())),
            link: Box::new(NotApplicableLink(reason.to_owned())),
            rdma: Box::new(NotApplicableRdma(reason.to_owned())),
            gpu: Box::new(NotApplicableGpu(reason.to_owned())),
            shards: Box::new(NotApplicableShards(reason.to_owned())),
        }
    }

    /// The real probes, reading this machine through `nvidia-smi`, `ethtool`, `ping`,
    /// `ibv_write_bw`, `sha256sum` and the filesystem.
    #[must_use]
    pub fn system() -> Self {
        Self {
            driver: Box::new(SystemProbes),
            link: Box::new(SystemProbes),
            rdma: Box::new(SystemProbes),
            gpu: Box::new(SystemProbes),
            shards: Box::new(SystemProbes),
        }
    }
}

/// Stand-in probe used by [`Probes::not_applicable`] for every domain.
#[derive(Debug, Clone)]
struct NotApplicableDriver(String);

impl DriverProbe for NotApplicableDriver {
    fn driver_version(&self) -> Probe<String> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn kernel_release(&self) -> Probe<String> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn uvm_page_packing(&self) -> Probe<UvmPacking> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
}

#[derive(Debug, Clone)]
struct NotApplicableLink(String);

impl LinkProbe for NotApplicableLink {
    fn rail_inventory(&self) -> Probe<Vec<RailStatus>> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn link_speed_mbps(&self, _netdev: &str) -> Probe<u64> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn mtu(&self, _netdev: &str) -> Probe<u32> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn df_probe(&self, _netdev: &str, _peer: &str, _payload_bytes: usize) -> Probe<String> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
}

#[derive(Debug, Clone)]
struct NotApplicableRdma(String);

impl RdmaProbe for NotApplicableRdma {
    fn dual_rail_write_gbps(&self, _peers: &[String], _seconds: u64) -> Probe<Vec<PeerBandwidth>> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn oneshot_allreduce_us(&self, _rank: usize, _peers: &[String], _bytes: usize) -> Probe<f64> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
}

#[derive(Debug, Clone)]
struct NotApplicableGpu(String);

impl GpuProbe for NotApplicableGpu {
    fn sm_clock_range_mhz(&self) -> Probe<ClockRange> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn lock_sm_clocks(&self, _min_mhz: u64, _max_mhz: u64) -> Probe<String> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn mem_available_gib(&self) -> Probe<u64> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
}

#[derive(Debug, Clone)]
struct NotApplicableShards(String);

impl ShardProbe for NotApplicableShards {
    fn file_len(&self, _path: &Path) -> Probe<u64> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn sha256(&self, _path: &Path) -> Probe<String> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn mount(&self, _path: &Path) -> Probe<MountInfo> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
    fn manifest(&self, _dir: &Path) -> Probe<EngramManifest> {
        Err(ProbeError::NotApplicable(self.0.clone()))
    }
}

// ---------------------------------------------------------------------------------------------
// Real probes
// ---------------------------------------------------------------------------------------------

/// The real, read-only probes for this machine.
///
/// Every method shells out to the same tool an operator would use by hand, so a failure can be
/// reproduced exactly. Nothing here changes system state; [`GpuProbe::lock_sm_clocks`] is
/// the single exception and is only reachable through `--apply`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemProbes;

impl SystemProbes {
    fn output(program: &str, args: &[&str]) -> Probe<String> {
        let out = Command::new(program).args(args).output().map_err(|error| {
            ProbeError::Unavailable(format!("{program} could not be run: {error}"))
        })?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(ProbeError::Unavailable(format!(
                "{program} exited with {}: {}",
                out.status,
                stderr.trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
    }
}

impl DriverProbe for SystemProbes {
    fn driver_version(&self) -> Probe<String> {
        let raw = Self::output(
            "nvidia-smi",
            &["--query-gpu=driver_version", "--format=csv,noheader"],
        )?;
        raw.lines()
            .next()
            .map(|line| line.trim().to_owned())
            .filter(|line| !line.is_empty())
            .ok_or_else(|| ProbeError::Unavailable("nvidia-smi reported no driver version".into()))
    }

    fn kernel_release(&self) -> Probe<String> {
        let raw = std::fs::read_to_string("/proc/sys/kernel/osrelease").map_err(|error| {
            ProbeError::Unavailable(format!("/proc/sys/kernel/osrelease: {error}"))
        })?;
        let release = raw.trim().to_owned();
        if release.is_empty() {
            return Err(ProbeError::Unavailable(
                "/proc/sys/kernel/osrelease was empty".into(),
            ));
        }
        Ok(release)
    }

    fn uvm_page_packing(&self) -> Probe<UvmPacking> {
        // The specification requires "nvidia-uvm 64K page packing" but names no knob, and a real
        // 580-series node exposes none under a conventional name (56 parameters, none matching
        // *pack* or *64k*). Enumerate every parameter and look for anything that reads like a
        // packing switch rather than guessing three literal file names, and report how many
        // parameters were seen so the operator can tell a missing knob from a wrong guess.
        let mut parameters_seen = None;
        let mut candidates = Vec::new();
        let mut state = None;
        if let Ok(entries) = std::fs::read_dir(UVM_PARAMETERS_DIR) {
            let mut seen = 0_usize;
            for entry in entries.flatten() {
                seen += 1;
                let name = entry.file_name().to_string_lossy().into_owned();
                let lower = name.to_ascii_lowercase();
                if !(lower.contains("pack") || lower.contains("64k")) {
                    continue;
                }
                let value = read_trim(&entry.path())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                state = state.or_else(|| packing_from_value(&value));
                candidates.push(name);
            }
            parameters_seen = Some(seen);
        }
        if state.is_none() {
            // /proc/driver/nvidia/params is the other conventional home for a driver switch.
            if let Some(raw) = read_trim(Path::new("/proc/driver/nvidia/params")) {
                for line in raw.lines() {
                    let lower = line.to_ascii_lowercase();
                    if lower.contains("uvm") && (lower.contains("pack") || lower.contains("64k")) {
                        state = state.or_else(|| packing_from_value(&lower));
                        candidates.push(line.trim().to_owned());
                    }
                }
            }
        }
        Ok(UvmPacking {
            state: state.unwrap_or(UvmPagePacking::Unknown),
            parameters_seen,
            candidates,
        })
    }
}

/// Reads a file and trims it, or nothing when it cannot be read.
fn read_trim(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|raw| raw.trim().to_owned())
}

/// Maps a parameter value onto a packing verdict.
fn packing_from_value(value: &str) -> Option<UvmPagePacking> {
    let value = value.trim().to_ascii_lowercase();
    for token in value.split([':', '=', ' ']) {
        match token {
            "y" | "yes" | "1" | "true" | "64k" => return Some(UvmPagePacking::Enabled),
            "n" | "no" | "0" | "false" | "4k" => return Some(UvmPagePacking::Disabled),
            _ => {}
        }
    }
    None
}

/// `200 Gb/sec (4X HDR)` and `40 Gb/sec` become 200000 and 40000 Mb/s.
fn parse_ib_rate(raw: &str) -> Option<u64> {
    let gbps: f64 = raw.split_whitespace().next()?.parse().ok()?;
    Some((gbps * 1000.0).round() as u64)
}

/// Drops the PCI function from an address: `0000:01:00.0` becomes `0000:01:00`.
fn strip_pci_function(address: &str) -> String {
    match address.rfind('.') {
        Some(index) => address[..index].to_owned(),
        None => address.to_owned(),
    }
}

impl LinkProbe for SystemProbes {
    fn rail_inventory(&self) -> Probe<Vec<RailStatus>> {
        let root = Path::new("/sys/class/infiniband");
        let entries = std::fs::read_dir(root)
            .map_err(|error| ProbeError::Unavailable(format!("{}: {error}", root.display())))?;
        let mut inventory = Vec::new();
        for entry in entries.flatten() {
            let rdma = entry.file_name().to_string_lossy().into_owned();
            let port = entry.path().join("ports/1");
            let raw_state = read_trim(&port.join("state")).unwrap_or_else(|| "UNKNOWN".to_owned());
            let state = raw_state
                .rsplit(':')
                .next()
                .unwrap_or(&raw_state)
                .trim()
                .to_owned();
            let speed_mbps = read_trim(&port.join("rate")).and_then(|raw| parse_ib_rate(&raw));
            let netdev = std::fs::read_dir(entry.path().join("device/net"))
                .ok()
                .and_then(|mut dir| dir.find_map(|item| item.ok()))
                .map(|item| item.file_name().to_string_lossy().into_owned());
            let pci = std::fs::canonicalize(entry.path().join("device"))
                .ok()
                .and_then(|path| {
                    path.file_name()
                        .map(|name| strip_pci_function(&name.to_string_lossy()))
                });
            inventory.push(RailStatus {
                rdma,
                netdev,
                state,
                speed_mbps,
                pci,
            });
        }
        if inventory.is_empty() {
            return Err(ProbeError::Unavailable(format!(
                "{} lists no RDMA devices; the ConnectX-7 rails are invisible to ibverbs",
                root.display()
            )));
        }
        inventory.sort_by(|left, right| left.rdma.cmp(&right.rdma));
        Ok(inventory)
    }

    fn link_speed_mbps(&self, netdev: &str) -> Probe<u64> {
        let raw = Self::output("ethtool", &[netdev])?;
        for line in raw.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("Speed:") {
                let value = rest.trim();
                let digits: String = value.chars().take_while(char::is_ascii_digit).collect();
                return digits.parse::<u64>().map_err(|_| {
                    ProbeError::Unavailable(format!(
                        "{netdev} reports its speed as '{value}', which is not a negotiated rate: the link is down or has not trained"
                    ))
                });
            }
        }
        Err(ProbeError::Unavailable(format!(
            "ethtool did not report a Speed line for {netdev}"
        )))
    }

    fn mtu(&self, netdev: &str) -> Probe<u32> {
        let path = format!("/sys/class/net/{netdev}/mtu");
        let raw = std::fs::read_to_string(&path)
            .map_err(|error| ProbeError::Unavailable(format!("{path}: {error}")))?;
        raw.trim()
            .parse::<u32>()
            .map_err(|error| ProbeError::Unavailable(format!("{path}: {error}")))
    }

    fn df_probe(&self, netdev: &str, peer: &str, payload_bytes: usize) -> Probe<String> {
        let size = payload_bytes.to_string();
        let out = Command::new("ping")
            .args(["-M", "do", "-s", &size, "-c", "1", "-W", "2", peer])
            .output()
            .map_err(|error| ProbeError::Unavailable(format!("ping could not be run: {error}")))?;
        if out.status.success() {
            Ok(format!(
                "{netdev}: {payload_bytes}-byte do-not-fragment probe to {peer} succeeded"
            ))
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            Err(ProbeError::Unavailable(format!(
                "{netdev}: {payload_bytes}-byte do-not-fragment probe to {peer} failed ({}): {}{}",
                out.status,
                stdout.trim().lines().last().unwrap_or("").trim(),
                stderr.trim()
            )))
        }
    }
}

/// Reads the one-shot All-Reduce latency from the rocenante-probe JSON document, requiring the
/// probe's own transport labels before the number is trusted.
///
/// The probe labels every measurement with the transport it used and whether that transport says
/// anything about the fabric, precisely so a consumer can tell a fabric number from an in-process
/// pipeline number. The doctor is that consumer, so the labels are a requirement rather than
/// decoration: a preflight that scored a loopback run would report the 200 Gb/s gate from a
/// measurement that never touched the fabric, either passing it falsely on a fast host or blaming
/// the RoCEnante staging fence for a pipeline latency. Missing or unreadable labels fail closed.
fn oneshot_allreduce_us_from_probe_json(raw: &str) -> Probe<f64> {
    let value: Value = serde_json::from_str(raw).map_err(|error| {
        ProbeError::Unavailable(format!("rocenante-probe returned unreadable JSON: {error}"))
    })?;
    let transport = value
        .get("transport")
        .and_then(Value::as_str)
        .unwrap_or("(no transport label)");
    let fabric = match value.get("fabric") {
        Some(Value::Bool(true)) => "true".to_owned(),
        Some(Value::Bool(false)) => "false".to_owned(),
        _ => "(no fabric label)".to_owned(),
    };
    if fabric != "true" || transport != "rdma" {
        return Err(ProbeError::Unavailable(format!(
            "rocenante-probe reported transport {transport} with fabric {fabric}, which is not the RDMA fabric path, so this number is refused: the one-shot All-Reduce gate is the fabric measurement and an in-process pipeline run must never be scored as it. Run the probe on a node whose RDMA rails are ACTIVE"
        )));
    }
    value
        .get("oneshot_allreduce_us")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            ProbeError::Unavailable("rocenante-probe JSON has no oneshot_allreduce_us field".into())
        })
}

impl RdmaProbe for SystemProbes {
    fn dual_rail_write_gbps(&self, peers: &[String], seconds: u64) -> Probe<Vec<PeerBandwidth>> {
        if peers.is_empty() {
            return Err(ProbeError::NotApplicable(
                "no peer was given for the RDMA probe".into(),
            ));
        }
        let seconds = seconds.to_string();
        let mut readings = Vec::with_capacity(peers.len());
        for peer in peers {
            let mut total = 0.0_f64;
            for rail in RAILS {
                let raw = Self::output(
                    "ibv_write_bw",
                    &["-d", rail.rdma, "--report_gbits", "-D", &seconds, peer],
                )?;
                let value = raw
                    .lines()
                    .find_map(|line| {
                        let mut fields = line.split_whitespace();
                        let first = fields.next()?;
                        let value = first.parse::<f64>().ok()?;
                        Some(value)
                    })
                    .ok_or_else(|| {
                        ProbeError::Unavailable(format!(
                            "ibv_write_bw on {} toward {peer} produced no bandwidth line",
                            rail.rdma
                        ))
                    })?;
                total += value;
            }
            readings.push(PeerBandwidth {
                peer: peer.clone(),
                gbps: total,
            });
        }
        Ok(readings)
    }

    fn oneshot_allreduce_us(&self, rank: usize, peers: &[String], bytes: usize) -> Probe<f64> {
        if peers.is_empty() {
            return Err(ProbeError::NotApplicable(
                "no peer was given for the All-Reduce probe".into(),
            ));
        }
        // The probe needs one control-plane address per rank. A rank other than 0 only dials
        // table[0], which is a --nodes peer, so its own slot can be a placeholder. Rank 0 BINDS
        // table[0], so it must know its own real address: that cannot be guessed from --nodes,
        // and binding a placeholder would make every peer dial an unreachable address.
        let mut table = match std::env::var("SPARK_COMM_PEERS") {
            Ok(full) if !full.trim().is_empty() => {
                let table: Vec<String> =
                    full.split(',').map(|peer| peer.trim().to_owned()).collect();
                if table.len() != peers.len() + 1 {
                    return Err(ProbeError::NotApplicable(format!(
                        "SPARK_COMM_PEERS names {} ranks but this doctor call has {} peer(s); it must name every rank including this node (world {})",
                        table.len(),
                        peers.len(),
                        peers.len() + 1
                    )));
                }
                if table.get(rank).is_none_or(|peer| peer.is_empty()) {
                    return Err(ProbeError::NotApplicable(format!(
                        "SPARK_COMM_PEERS has no address for rank {rank}"
                    )));
                }
                table
            }
            _ if rank == 0 => {
                return Err(ProbeError::NotApplicable(
                    "a rank-0 doctor cannot guess its own rail address: set SPARK_COMM_PEERS to the full rank table (rank 0 first) so the control plane can bind a reachable address".into(),
                ));
            }
            _ => {
                let mut table = peers.to_vec();
                table.insert(rank.min(table.len()), "127.0.0.1".to_owned());
                table
            }
        };
        table.truncate(peers.len() + 1);
        let size = bytes.to_string();
        let world = table.len().to_string();
        let rank_text = rank.to_string();
        let mut arguments: Vec<&str> = vec![
            "--bytes", &size, "--world", &world, "--rank", &rank_text, "--json",
        ];
        for peer in &table {
            arguments.push("--peer");
            arguments.push(peer.as_str());
        }
        // The probe either measures the fabric path or exits non-zero; it never prints a number it
        // could not measure. Its own labels then decide whether that number belongs to this gate: a
        // loopback run is a real measurement of the wrong thing, so the reader refuses it.
        let raw = Self::output("rocenante-probe", &arguments)?;
        oneshot_allreduce_us_from_probe_json(&raw)
    }
}

impl GpuProbe for SystemProbes {
    fn sm_clock_range_mhz(&self) -> Probe<ClockRange> {
        let raw = Self::output(
            "nvidia-smi",
            &[
                "--query-gpu=clocks.max.sm,clocks.min.sm",
                "--format=csv,noheader,nounits",
            ],
        )?;
        let line = raw
            .lines()
            .next()
            .ok_or_else(|| ProbeError::Unavailable("nvidia-smi reported no clock row".into()))?;
        let mut fields = line.split(',').map(|field| field.trim());
        let parse = |value: Option<&str>, what: &str| -> Probe<u64> {
            value
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or_else(|| {
                    ProbeError::Unavailable(format!("nvidia-smi reported no usable {what} clock"))
                })
        };
        Ok(ClockRange {
            min_mhz: parse(fields.next(), "minimum SM")?,
            max_mhz: parse(fields.next(), "maximum SM")?,
        })
    }

    fn lock_sm_clocks(&self, min_mhz: u64, max_mhz: u64) -> Probe<String> {
        let window = format!("{min_mhz},{max_mhz}");
        let out = Command::new("nvidia-smi")
            .args(["-lgc", &window])
            .output()
            .map_err(|error| {
                ProbeError::Unavailable(format!("nvidia-smi -lgc could not be run: {error}"))
            })?;
        if out.status.success() {
            Ok(format!("nvidia-smi -lgc {window} applied"))
        } else {
            Err(ProbeError::Unavailable(format!(
                "nvidia-smi -lgc {window} exited with {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )))
        }
    }

    fn mem_available_gib(&self) -> Probe<u64> {
        let raw = std::fs::read_to_string("/proc/meminfo")
            .map_err(|error| ProbeError::Unavailable(format!("/proc/meminfo: {error}")))?;
        for line in raw.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kib = rest
                    .split_whitespace()
                    .next()
                    .and_then(|value| value.parse::<u64>().ok())
                    .ok_or_else(|| {
                        ProbeError::Unavailable("MemAvailable was not a number".into())
                    })?;
                return Ok(kib / (1024 * 1024));
            }
        }
        Err(ProbeError::Unavailable(
            "/proc/meminfo had no MemAvailable line".into(),
        ))
    }
}

impl ShardProbe for SystemProbes {
    fn file_len(&self, path: &Path) -> Probe<u64> {
        std::fs::metadata(path)
            .map(|metadata| metadata.len())
            .map_err(|error| ProbeError::Unavailable(format!("{}: {error}", path.display())))
    }

    fn sha256(&self, path: &Path) -> Probe<String> {
        // Coreutils rather than a second Rust SHA256 implementation: same digest, no new
        // dependency. Milestone 2's sparkflash_engram::packer::sha256_file can replace this
        // once packer.rs is on main.
        let raw = Self::output("sha256sum", &["--binary", &path.display().to_string()])?;
        raw.split_whitespace()
            .next()
            .map(|hash| hash.to_ascii_lowercase())
            .ok_or_else(|| {
                ProbeError::Unavailable(format!("sha256sum gave no digest for {}", path.display()))
            })
    }

    fn mount(&self, path: &Path) -> Probe<MountInfo> {
        let out = Command::new("df")
            .args(["--output=source,fstype", &path.display().to_string()])
            .output()
            .map_err(|error| ProbeError::Unavailable(format!("df could not be run: {error}")))?;
        if !out.status.success() {
            return Err(ProbeError::Unavailable(format!(
                "df {} exited with {}: {}",
                path.display(),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let row = text.lines().nth(1).ok_or_else(|| {
            ProbeError::Unavailable(format!("df {} printed no row", path.display()))
        })?;
        let mut fields = row.split_whitespace();
        let source = fields.next().unwrap_or_default().to_owned();
        let fstype = fields.next().unwrap_or_default().to_owned();
        if source.is_empty() {
            return Err(ProbeError::Unavailable(format!(
                "df {} reported no source device",
                path.display()
            )));
        }
        Ok(MountInfo { source, fstype })
    }

    fn manifest(&self, dir: &Path) -> Probe<EngramManifest> {
        let path = dir.join(MANIFEST_NAME);
        // A missing manifest means the packer has not run here, which is not the same thing
        // as a corrupt one: the caller reports it as not applicable rather than as a pass.
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ProbeError::NotApplicable(format!(
                    "{} does not exist; shard integrity cannot be verified until pack-engram has run",
                    path.display()
                )));
            }
            Err(error) => {
                return Err(ProbeError::Unavailable(format!(
                    "{}: {error}",
                    path.display()
                )));
            }
        };
        let value: Value = serde_json::from_str(&raw)
            .map_err(|error| ProbeError::Unavailable(format!("{}: {error}", path.display())))?;
        Ok(parse_manifest(&value))
    }
}

/// Reads the packer manifest tolerantly.
///
/// Milestone 2 writes `{"shards": [{"path": ..., "sha256": ..., "file_bytes": ...}]}`; a
/// filename-keyed object is accepted too, and unknown fields are ignored, so the preflight does
/// not have to be re-released every time the packer grows a field.
#[must_use]
pub fn parse_manifest(value: &Value) -> EngramManifest {
    let mut entries = Vec::new();
    let mut push = |file: String, entry: &Value| {
        let sha256 = entry
            .get("sha256")
            .and_then(Value::as_str)
            .map(str::to_ascii_lowercase);
        let file_bytes = entry
            .get("file_bytes")
            .or_else(|| entry.get("bytes"))
            .or_else(|| entry.get("len"))
            .and_then(Value::as_u64);
        entries.push(ManifestEntry {
            file,
            sha256,
            file_bytes,
        });
    };
    if let Some(shards) = value.get("shards").and_then(Value::as_array) {
        for shard in shards {
            let name = ["path", "file", "name", "file_name"]
                .iter()
                .find_map(|key| shard.get(*key).and_then(Value::as_str));
            if let Some(name) = name {
                push(name.to_owned(), shard);
            }
        }
    } else if let Some(files) = value.get("files").and_then(Value::as_object) {
        for (name, entry) in files {
            push(name.clone(), entry);
        }
    } else if let Some(object) = value.as_object() {
        for (name, entry) in object {
            if entry.is_object() {
                push(name.clone(), entry);
            }
        }
    }
    EngramManifest { entries }
}

// ---------------------------------------------------------------------------------------------
// Version arithmetic
// ---------------------------------------------------------------------------------------------

/// Splits a dotted numeric version into its components.
#[must_use]
pub fn version_parts(version: &str) -> Option<Vec<u64>> {
    let trimmed = version.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed
        .split('.')
        .map(|part| part.parse::<u64>().ok())
        .collect()
}

/// Compares two dotted numeric versions component-wise, treating missing components as zero.
#[must_use]
pub fn compare_versions(left: &str, right: &str) -> Option<Ordering> {
    let left = version_parts(left)?;
    let right = version_parts(right)?;
    for index in 0..left.len().max(right.len()) {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        match a.cmp(&b) {
            Ordering::Equal => {}
            other => return Some(other),
        }
    }
    Some(Ordering::Equal)
}

/// Why a driver version is unacceptable, if it is.
pub fn driver_version_problem(version: &str) -> Option<String> {
    if REJECTED_DRIVER_VERSIONS.contains(&version.trim()) {
        return Some(format!(
            "driver {} is explicitly rejected: it throttles ConnectX-7 to 13 Gbps",
            version.trim()
        ));
    }
    match compare_versions(version, MIN_DRIVER_VERSION) {
        Some(Ordering::Less) => Some(format!(
            "driver {} is older than the required {MIN_DRIVER_VERSION}",
            version.trim()
        )),
        Some(_) => None,
        None => Some(format!(
            "driver version {} is not parseable",
            version.trim()
        )),
    }
}

/// A parsed kernel release such as `6.17.0-1018-nvidia-64k`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRelease {
    /// Numeric base version, e.g. `[6, 17, 0]`.
    pub base: Vec<u64>,
    /// ABI number after the first dash, e.g. `1018`.
    pub abi: Option<u64>,
    /// The release string exactly as read.
    pub raw: String,
}

/// Parses a kernel release string.
#[must_use]
pub fn parse_kernel_release(raw: &str) -> Option<KernelRelease> {
    let trimmed = raw.trim();
    let mut fields = trimmed.split('-');
    let base = version_parts(fields.next()?)?;
    if base.len() < 2 {
        return None;
    }
    let abi = fields.next().and_then(|field| field.parse::<u64>().ok());
    Some(KernelRelease {
        base,
        abi,
        raw: trimmed.to_owned(),
    })
}

/// Whether a parsed base version is at least `minimum`.
#[must_use]
pub fn base_at_least(base: &[u64], minimum: [u64; 3]) -> bool {
    for (index, floor) in minimum.iter().enumerate() {
        let value = base.get(index).copied().unwrap_or(0);
        match value.cmp(floor) {
            Ordering::Greater => return true,
            Ordering::Less => return false,
            Ordering::Equal => {}
        }
    }
    true
}

/// Why a kernel release is unacceptable, if it is.
pub fn kernel_release_problem(release: &KernelRelease) -> Option<String> {
    let Some(abi) = release.abi else {
        return Some(format!(
            "kernel {} has no ABI number; expected >= {MIN_KERNEL_ABI}",
            release.raw
        ));
    };
    // The ABI floor belongs to the branch the base version lands on: a 7.x kernel must clear
    // the 7.x floor, not the 6.x one, so 7.0.0-1018 is refused even though its base is newer
    // than 6.17.0. A 6.x kernel newer than 6.17.0 still only has to clear 1018.
    let acceptable = if base_at_least(&release.base, ALT_MIN_KERNEL_BASE) {
        abi >= ALT_MIN_KERNEL_ABI
    } else {
        base_at_least(&release.base, MIN_KERNEL_BASE) && abi >= MIN_KERNEL_ABI
    };
    if acceptable {
        None
    } else {
        Some(format!(
            "kernel {} is below 6.17.0-{MIN_KERNEL_ABI} or 7.0.0-{ALT_MIN_KERNEL_ABI}",
            release.raw
        ))
    }
}

/// Rounds `value` up to the next multiple of `alignment`.
#[must_use]
pub fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

// ---------------------------------------------------------------------------------------------
// Check outcomes and report
// ---------------------------------------------------------------------------------------------

/// The five checks, in specification order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckId {
    /// Check 1: driver, kernel and nvidia-uvm.
    DriverKernel,
    /// Check 2: dual-rail link speed and MTU.
    LinkMtu,
    /// Check 3: dual-rail RDMA bandwidth and one-shot latency.
    RdmaBandwidth,
    /// Check 4: SM clock cap and available memory.
    ClocksMemory,
    /// Check 5: local NVMe Engram shards.
    EngramShards,
}

impl CheckId {
    /// Stable machine-readable identifier.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            CheckId::DriverKernel => "driver-kernel",
            CheckId::LinkMtu => "link-mtu",
            CheckId::RdmaBandwidth => "rdma-bandwidth",
            CheckId::ClocksMemory => "clocks-memory",
            CheckId::EngramShards => "engram-shards",
        }
    }

    /// Human title, as the specification heads the check.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            CheckId::DriverKernel => "Driver and kernel",
            CheckId::LinkMtu => "Dual-rail ConnectX-7 link speed and MTU",
            CheckId::RdmaBandwidth => "Dual-rail RDMA bandwidth and one-shot latency",
            CheckId::ClocksMemory => "SM clock cap and available memory",
            CheckId::EngramShards => "Local NVMe Engram shards",
        }
    }
}

/// The five checks in the order the specification lists them.
pub const CHECK_ORDER: [CheckId; IMPLEMENTED_CHECK_COUNT] = [
    CheckId::DriverKernel,
    CheckId::LinkMtu,
    CheckId::RdmaBandwidth,
    CheckId::ClocksMemory,
    CheckId::EngramShards,
];

/// Result of one check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// The check ran and the hardware satisfies it.
    Pass,
    /// The check ran and the hardware does not satisfy it, or the probe could not be taken.
    Fail,
    /// The operator did not ask for this check; it did not run.
    NotApplicable,
}

impl CheckStatus {
    /// Stable machine-readable identifier.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            CheckStatus::Pass => "pass",
            CheckStatus::Fail => "fail",
            CheckStatus::NotApplicable => "not-applicable",
        }
    }

    /// Single-character marker used by the human report.
    #[must_use]
    pub fn marker(self) -> char {
        match self {
            CheckStatus::Pass => '+',
            CheckStatus::Fail => 'x',
            CheckStatus::NotApplicable => '-',
        }
    }
}

/// One check's verdict, evidence and remedy.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckOutcome {
    /// Which check this is.
    pub id: CheckId,
    /// Outcome of the check.
    pub status: CheckStatus,
    /// One-line verdict.
    pub summary: String,
    /// Evidence lines: what was measured, in order.
    pub details: Vec<String>,
    /// What the operator must do when the check failed.
    pub remediation: Option<String>,
}

/// Overall verdict of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Every check ran and passed.
    Ready,
    /// At least one check failed.
    NotReady,
    /// Nothing failed, but at least one check did not run.
    Incomplete,
}

impl Verdict {
    /// Stable machine-readable identifier.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::Ready => "ready",
            Verdict::NotReady => "not_ready",
            Verdict::Incomplete => "incomplete",
        }
    }

    /// Exit status for the process: zero unless a check failed.
    #[must_use]
    pub fn exit_code(self) -> u8 {
        match self {
            Verdict::Ready | Verdict::Incomplete => 0,
            Verdict::NotReady => 1,
        }
    }
}

/// The whole run.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Outcomes, in specification order.
    pub outcomes: Vec<CheckOutcome>,
    /// Overall verdict.
    pub verdict: Verdict,
}

impl Report {
    /// Derives the overall verdict from the outcomes.
    #[must_use]
    pub fn from_outcomes(outcomes: Vec<CheckOutcome>) -> Self {
        let verdict = if outcomes.iter().any(|o| o.status == CheckStatus::Fail) {
            Verdict::NotReady
        } else if outcomes
            .iter()
            .any(|o| o.status == CheckStatus::NotApplicable)
        {
            Verdict::Incomplete
        } else {
            Verdict::Ready
        };
        Self { outcomes, verdict }
    }

    /// Process exit status.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        self.verdict.exit_code()
    }

    /// Counts of (passed, failed, not applicable).
    #[must_use]
    pub fn counts(&self) -> (usize, usize, usize) {
        let count = |status: CheckStatus| {
            self.outcomes
                .iter()
                .filter(|outcome| outcome.status == status)
                .count()
        };
        (
            count(CheckStatus::Pass),
            count(CheckStatus::Fail),
            count(CheckStatus::NotApplicable),
        )
    }

    /// Machine-readable report.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let (passed, failed, skipped) = self.counts();
        json!({
            "tool": "sparkflash doctor",
            "spec_section": SPEC_SECTION,
            "specified_checks": SPECIFIED_CHECK_COUNT,
            "implemented_checks": IMPLEMENTED_CHECK_COUNT,
            "spec_discrepancy": SPEC_DISCREPANCY,
            "verdict": self.verdict.as_str(),
            "exit_code": self.exit_code(),
            "counts": { "passed": passed, "failed": failed, "not_applicable": skipped },
            "checks": self.outcomes.iter().map(|outcome| json!({
                "id": outcome.id.as_str(),
                "title": outcome.id.title(),
                "status": outcome.status.as_str(),
                "summary": outcome.summary,
                "details": outcome.details,
                "remediation": outcome.remediation,
            })).collect::<Vec<_>>(),
        })
    }

    /// Human-readable report.
    #[must_use]
    pub fn to_human(&self) -> String {
        let mut out = String::new();
        out.push_str("sparkflash doctor - DGX Spark preflight (specification 6.1)\n");
        out.push_str(&format!("note: {SPEC_DISCREPANCY}\n\n"));
        for outcome in &self.outcomes {
            out.push_str(&format!(
                "[{}] {} ({})\n",
                outcome.status.marker(),
                outcome.id.title(),
                outcome.id.as_str()
            ));
            out.push_str(&format!("      {}\n", outcome.summary));
            for detail in &outcome.details {
                out.push_str(&format!("      - {detail}\n"));
            }
            if let Some(remediation) = &outcome.remediation {
                out.push_str(&format!("      fix: {remediation}\n"));
            }
            out.push('\n');
        }
        let (passed, failed, skipped) = self.counts();
        out.push_str(&format!(
            "{IMPLEMENTED_CHECK_COUNT} of {SPECIFIED_CHECK_COUNT} specified checks implemented: {passed} passed, {failed} failed, {skipped} not applicable\n"
        ));
        let verdict = match self.verdict {
            Verdict::Ready => "READY: all five checks passed".to_owned(),
            Verdict::NotReady => {
                "NOT READY: fix the failed checks before loading weights".to_owned()
            }
            Verdict::Incomplete => {
                "INCOMPLETE: no check failed, but the hardware checks did not run; pass --local or --nodes to run them"
                    .to_owned()
            }
        };
        out.push_str(&format!("verdict: {verdict}\n"));
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------

/// Everything the command was asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// Cluster nodes to probe. Empty means the hardware checks do not run.
    pub nodes: Vec<String>,
    /// Machine-readable output.
    pub json: bool,
    /// Master switch for state-changing commands. On its own it changes nothing.
    pub apply: bool,
    /// Opt in to locking SM clocks. Requires [`Options::apply`] as well.
    pub lock_sm_clocks: bool,
    /// Also run the checks that need only this node.
    pub local: bool,
    /// Directory holding the Engram shards and manifest.
    pub engram_dir: PathBuf,
    /// Rank of this node.
    pub rank: usize,
    /// Number of ranks in the cluster.
    pub world_size: usize,
    /// Seconds each RDMA bandwidth probe runs.
    pub rdma_seconds: u64,
    /// Skip the whole-file SHA256 of each shard.
    pub skip_hash: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            json: false,
            apply: false,
            lock_sm_clocks: false,
            local: false,
            engram_dir: PathBuf::from(DEFAULT_ENGRAM_DIR),
            rank: 0,
            world_size: 4,
            rdma_seconds: 1,
            skip_hash: false,
        }
    }
}

impl Options {
    /// Whether any hardware probe should run at all.
    #[must_use]
    pub fn live(&self) -> bool {
        self.local || !self.nodes.is_empty()
    }

    /// Whether the cluster-wide checks (link probes to peers, RDMA) should run.
    #[must_use]
    pub fn cluster(&self) -> bool {
        !self.nodes.is_empty()
    }

    /// Whether the clock lock may run: the master switch and the named action, both explicit.
    #[must_use]
    pub fn may_lock_clocks(&self) -> bool {
        self.apply && self.lock_sm_clocks
    }
}

/// Command line, parsed by clap and then handed to [`Options`].
#[derive(Debug, Parser)]
#[command(
    name = "sparkflash doctor",
    about = "DGX Spark preflight gate (specification 6.1)",
    long_about = "Validates this node (and optionally the cluster) against the five checks the \
specification actually enumerates. The specification's heading claims seven points; doctor \
implements the five that are written down and says so in every report. Nothing is probed by \
default: pass --local for this node, or --nodes for the cluster. The default run is read-only; \
no command that changes state is executed unless --apply and the action-specific flag (for \
example --lock-sm-clocks) are both given."
)]
pub struct DoctorArgs {
    /// Cluster nodes to probe, comma separated; enables the cluster-wide checks.
    #[arg(long, value_delimiter = ',')]
    pub nodes: Vec<String>,
    /// Emit the machine-readable report.
    #[arg(long)]
    pub json: bool,
    /// Permit state-changing commands. On its own it changes nothing; a specific mutation flag
    /// such as --lock-sm-clocks also has to be given.
    #[arg(long)]
    pub apply: bool,
    /// Run `nvidia-smi -lgc 0,2200`, which locks this node's SM clock range to [0, 2200] MHz.
    /// This changes the clock policy of a node that may be serving traffic, so it is refused
    /// unless --apply is given too.
    #[arg(long)]
    pub lock_sm_clocks: bool,
    /// Probe this node only (driver, kernel, clocks, memory, shards).
    #[arg(long)]
    pub local: bool,
    /// Directory holding the Engram shards and manifest.
    #[arg(long, default_value = DEFAULT_ENGRAM_DIR)]
    pub engram_dir: PathBuf,
    /// Rank of this node in the cluster.
    #[arg(long, default_value_t = 0)]
    pub rank: usize,
    /// Number of ranks in the cluster.
    #[arg(long, default_value_t = 4)]
    pub world_size: usize,
    /// Seconds each RDMA bandwidth probe runs.
    #[arg(long, default_value_t = 1)]
    pub rdma_seconds: u64,
    /// Skip the whole-file SHA256 of each Engram shard.
    #[arg(long)]
    pub skip_hash: bool,
}

impl DoctorArgs {
    /// Splits the parsed arguments into runtime options, validating what clap cannot.
    pub fn into_options(self) -> Result<Options, String> {
        if self.world_size == 0 {
            return Err("--world-size must be at least 1".to_owned());
        }
        if self.rank >= self.world_size {
            return Err(format!(
                "--rank {} is outside --world-size {}",
                self.rank, self.world_size
            ));
        }
        Ok(Options {
            nodes: self.nodes,
            json: self.json,
            apply: self.apply,
            lock_sm_clocks: self.lock_sm_clocks,
            local: self.local,
            engram_dir: self.engram_dir,
            rank: self.rank,
            world_size: self.world_size,
            rdma_seconds: self.rdma_seconds,
            skip_hash: self.skip_hash,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------------------------

/// Accumulates evidence for one check.
#[derive(Debug, Default)]
struct Evidence {
    failed: bool,
    skipped: bool,
    details: Vec<String>,
    remediation: Vec<String>,
}

impl Evidence {
    fn note(&mut self, detail: impl Into<String>) {
        self.details.push(detail.into());
    }

    fn pass(&mut self, detail: impl Into<String>) {
        self.details.push(detail.into());
    }

    fn fail(&mut self, detail: impl Into<String>, remediation: impl Into<String>) {
        self.details.push(detail.into());
        self.remediation.push(remediation.into());
        self.failed = true;
    }

    fn skip(&mut self, detail: impl Into<String>) {
        self.details.push(detail.into());
        self.skipped = true;
    }

    /// Routes a probe result into pass/fail/skip evidence.
    fn resolve<T>(&mut self, reading: Probe<T>, on_value: impl FnOnce(&mut Evidence, T)) {
        match reading {
            Ok(value) => on_value(self, value),
            Err(ProbeError::NotApplicable(message)) => self.skip(message),
            Err(ProbeError::Unavailable(message)) => {
                self.fail(
                    message,
                    "install or fix the tool named in the probe error above (or run doctor on the node that owns the hardware), then re-run",
                )
            }
        }
    }

    fn status(&self) -> CheckStatus {
        if self.failed {
            CheckStatus::Fail
        } else if self.skipped {
            CheckStatus::NotApplicable
        } else {
            CheckStatus::Pass
        }
    }

    /// Turns the evidence into an outcome. `goal` states the requirement, never a result, so
    /// that the same sentence reads correctly as satisfied, not satisfied or not run.
    fn finish(self, id: CheckId, goal: impl Into<String>) -> CheckOutcome {
        let status = self.status();
        let goal = goal.into();
        let summary = match status {
            CheckStatus::Pass => format!("{goal}: satisfied"),
            CheckStatus::Fail => format!("NOT SATISFIED: {goal}"),
            CheckStatus::NotApplicable => format!("DID NOT RUN: {goal}"),
        };
        // A skipped or unavailable probe repeats its reason once per probe call; report each
        // distinct line once so the operator sees the cause, not the repetition.
        let mut details: Vec<String> = Vec::with_capacity(self.details.len());
        for detail in self.details {
            if !details.contains(&detail) {
                details.push(detail);
            }
        }
        CheckOutcome {
            id,
            status,
            summary,
            details,
            remediation: if self.remediation.is_empty() {
                None
            } else {
                // One remedy per distinct cause, in the order the probes reported them.
                let mut remedies: Vec<&str> = Vec::with_capacity(self.remediation.len());
                for remedy in &self.remediation {
                    if !remedies.contains(&remedy.as_str()) {
                        remedies.push(remedy);
                    }
                }
                Some(remedies.join("; "))
            },
        }
    }
}

/// Check 1: driver, kernel and nvidia-uvm packing.
#[must_use]
pub fn check_driver_kernel(probes: &Probes) -> CheckOutcome {
    let mut evidence = Evidence::default();
    evidence.resolve(probes.driver.driver_version(), |evidence, version| {
        match driver_version_problem(&version) {
            None => evidence.pass(format!(
                "driver {version} satisfies >= {MIN_DRIVER_VERSION}"
            )),
            Some(problem) => evidence.fail(
                problem,
                format!(
                    "install an NVIDIA driver >= {MIN_DRIVER_VERSION}; {} is rejected",
                    REJECTED_DRIVER_VERSIONS.join(", ")
                ),
            ),
        }
    });
    evidence.resolve(probes.driver.kernel_release(), |evidence, release| {
        match parse_kernel_release(&release) {
            None => evidence.fail(
                format!("kernel release {release} is not parseable"),
                format!(
                    "run kernel >= {}.{}.{}-{MIN_KERNEL_ABI} or >= {}.{}.{}-{ALT_MIN_KERNEL_ABI}",
                    MIN_KERNEL_BASE[0],
                    MIN_KERNEL_BASE[1],
                    MIN_KERNEL_BASE[2],
                    ALT_MIN_KERNEL_BASE[0],
                    ALT_MIN_KERNEL_BASE[1],
                    ALT_MIN_KERNEL_BASE[2]
                ),
            ),
            Some(parsed) => match kernel_release_problem(&parsed) {
                None => evidence.pass(format!("kernel {} satisfies the gate", parsed.raw)),
                Some(problem) => evidence.fail(
                    problem,
                    format!(
                        "run kernel >= {}.{}.{}-{MIN_KERNEL_ABI} or >= {}.{}.{}-{ALT_MIN_KERNEL_ABI}",
                        MIN_KERNEL_BASE[0],
                        MIN_KERNEL_BASE[1],
                        MIN_KERNEL_BASE[2],
                        ALT_MIN_KERNEL_BASE[0],
                        ALT_MIN_KERNEL_BASE[1],
                        ALT_MIN_KERNEL_BASE[2]
                    ),
                ),
            },
        }
    });
    evidence.resolve(probes.driver.uvm_page_packing(), |evidence, found| {
        let detail = match (found.parameters_seen, found.candidates.is_empty()) {
            (Some(seen), true) => format!(
                "nvidia-uvm: {} ({seen} module parameters listed under {UVM_PARAMETERS_DIR}, none named *pack* or *64k*)",
                found.state
            ),
            (Some(seen), false) => format!(
                "nvidia-uvm: {} ({seen} module parameters listed; candidates: {})",
                found.state,
                found.candidates.join(", ")
            ),
            (None, _) => format!("nvidia-uvm: {}", found.state),
        };
        match found.state {
            UvmPagePacking::Enabled => evidence.pass(detail),
            UvmPagePacking::Disabled => evidence.fail(
                detail,
                "switch nvidia-uvm back to 64K page packing before loading weights",
            ),
            UvmPagePacking::Unknown => evidence.fail(
                detail,
                format!(
                    "no 64K-packing knob was found under any conventional name on a real 580-series node, so doctor cannot verify this requirement and specification {SPEC_SECTION}'s nvidia-uvm clause looks like a specification gap rather than something you missed; confirm by hand with `ls {UVM_PARAMETERS_DIR}` (a 580-series node lists about 56 parameters and none match *pack* or *64k*), `cat /proc/driver/nvidia/params` and `nvidia-smi --query-gpu=driver_version --format=csv`, then either point doctor at the real knob or record the gap"
                ),
            ),
        }
    });
    evidence.finish(
        CheckId::DriverKernel,
        format!(
            "driver >= {MIN_DRIVER_VERSION} ({} rejected), kernel >= {}.{}.{}-{MIN_KERNEL_ABI} or {}.{}.{}-{ALT_MIN_KERNEL_ABI}-nvidia-64k, nvidia-uvm 64K packing",
            REJECTED_DRIVER_VERSIONS.join(", "),
            MIN_KERNEL_BASE[0],
            MIN_KERNEL_BASE[1],
            MIN_KERNEL_BASE[2],
            ALT_MIN_KERNEL_BASE[0],
            ALT_MIN_KERNEL_BASE[1],
            ALT_MIN_KERNEL_BASE[2]
        ),
    )
}

/// Check 2: dual-rail ConnectX-7 link speed, MTU and do-not-fragment reachability.
///
/// The specification names two specific functions (`enp1s0f0np0`/`rocep1s0f0` and
/// `enP2p1s0f0np0`/`roceP2p1s0f0`). On real GB10 hardware those are the wrong functions: the
/// functions carrying the 200 Gb/s rails are siblings on the same cards. This check does both
/// jobs - it reports what the specification asks for, and it gates on the rails that are
/// actually ACTIVE at [`LINK_SPEED_MBPS`], naming the sibling when the specification's function
/// is down. That is the difference between a disconnected cable and a wrong name in the
/// specification.
#[must_use]
pub fn check_link_mtu(probes: &Probes, options: &Options) -> CheckOutcome {
    let mut evidence = Evidence::default();
    let mut inventory: Option<Vec<RailStatus>> = None;
    match probes.link.rail_inventory() {
        Ok(found) => {
            evidence.note(format!(
                "RDMA devices on this node: {}",
                found
                    .iter()
                    .map(RailStatus::describe)
                    .collect::<Vec<_>>()
                    .join("; ")
            ));
            inventory = Some(found);
        }
        Err(ProbeError::NotApplicable(message)) => evidence.skip(message),
        Err(ProbeError::Unavailable(message)) => evidence.fail(
            message,
            "install the ibverbs user-space tools and run doctor on the node itself",
        ),
    }

    if let Some(inventory) = &inventory {
        let active: Vec<&RailStatus> = inventory
            .iter()
            .filter(|status| status.is_fast_active())
            .collect();
        if active.len() >= RAILS.len() {
            evidence.pass(format!(
                "{} rails are ACTIVE at >= {LINK_SPEED_MBPS} Mb/s: {}",
                active.len(),
                active
                    .iter()
                    .map(|status| status.rdma.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        } else {
            evidence.fail(
                format!(
                    "only {} of the {} required rails are ACTIVE at >= {LINK_SPEED_MBPS} Mb/s",
                    active.len(),
                    RAILS.len()
                ),
                format!(
                    "bring {} rails up at {LINK_SPEED_MBPS} Mb/s before loading weights: the dual-rail requirement cannot be met by fewer",
                    RAILS.len()
                ),
            );
        }
    }

    for rail in RAILS {
        if let Some(inventory) = &inventory {
            match inventory.iter().find(|status| status.is_spec_rail(rail)) {
                Some(status) => {
                    if status.is_active() {
                        evidence.pass(format!(
                            "{} ({}) is ACTIVE at {}",
                            rail.netdev,
                            rail.rdma,
                            status.speed_mbps.map_or_else(
                                || "an unknown rate".to_owned(),
                                |speed| format!("{speed} Mb/s")
                            )
                        ));
                    } else {
                        evidence.fail(
                            format!(
                                "{} ({}) is {}, not ACTIVE: the specification asks for this rail",
                                rail.netdev, rail.rdma, status.state
                            ),
                            format!(
                                "bring {} up, or treat the specification's rail names as wrong if a sibling function on the same card is carrying the traffic",
                                rail.rdma
                            ),
                        );
                    }
                    if let Some(sibling) = fast_sibling(inventory, status) {
                        evidence.note(format!(
                            "{} is {} but its sibling function {} on the same PCI device ({}) is ACTIVE at {} Mb/s with netdev {} - a wrong name in the specification, not a disconnected cable",
                            status.rdma,
                            status.state,
                            sibling.rdma,
                            status.pci.as_deref().unwrap_or("unknown PCI address"),
                            sibling.speed_mbps.unwrap_or_default(),
                            sibling.netdev.as_deref().unwrap_or("(no netdev)")
                        ));
                    }
                }
                None => evidence.fail(
                    format!("{} ({}) is not present on this node", rail.netdev, rail.rdma),
                    "list /sys/class/infiniband on the node and correct the specification or the hardware: the specification names a device this node does not expose",
                ),
            }
        }
        evidence.resolve(probes.link.link_speed_mbps(rail.netdev), |evidence, speed| {
            if speed == LINK_SPEED_MBPS {
                evidence.pass(format!("{}: {} Mb/s", rail.netdev, speed));
            } else {
                evidence.fail(
                    format!("{}: {speed} Mb/s, expected {LINK_SPEED_MBPS}", rail.netdev),
                    format!(
                        "re-seat the {} link ({}) until it negotiates {LINK_SPEED_MBPS} Mb/s, or correct the specification if it names the wrong function",
                        rail.netdev, rail.rdma
                    ),
                );
            }
        });
        evidence.resolve(probes.link.mtu(rail.netdev), |evidence, mtu| {
            if mtu == LINK_MTU {
                evidence.pass(format!("{}: MTU {mtu}", rail.netdev));
            } else {
                evidence.fail(
                    format!("{}: MTU {mtu}, expected {LINK_MTU}", rail.netdev),
                    format!("set MTU {LINK_MTU} on {}", rail.netdev),
                );
            }
        });
    }

    // The rails that actually carry the fabric must meet the MTU requirement too; on real
    // hardware these are the siblings of the functions the specification names.
    let mut gated_netdevs: Vec<String> = RAILS.iter().map(|rail| rail.netdev.to_owned()).collect();
    if let Some(inventory) = &inventory {
        for status in inventory.iter().filter(|status| status.is_active()) {
            let Some(netdev) = status.netdev.as_deref() else {
                // Say so rather than quietly skipping: an unchecked rail must be visible.
                evidence.note(format!(
                    "{} is ACTIVE but the kernel exposes no netdev for it, so its MTU was not checked here",
                    status.rdma
                ));
                continue;
            };
            if gated_netdevs.iter().any(|known| known == netdev) {
                continue;
            }
            gated_netdevs.push(netdev.to_owned());
            match probes.link.mtu(netdev) {
                Ok(mtu) if mtu == LINK_MTU => {
                    evidence.pass(format!("{netdev} (active {}): MTU {mtu}", status.rdma));
                }
                Ok(mtu) => evidence.fail(
                    format!(
                        "{netdev} (active {}): MTU {mtu}, expected {LINK_MTU}",
                        status.rdma
                    ),
                    format!("set MTU {LINK_MTU} on {netdev}"),
                ),
                Err(ProbeError::NotApplicable(message)) => evidence.skip(message),
                Err(ProbeError::Unavailable(message)) => evidence.fail(
                    message,
                    format!("read /sys/class/net/{netdev}/mtu on the node itself"),
                ),
            }
        }
    }

    if options.cluster() {
        for netdev in &gated_netdevs {
            for peer in &options.nodes {
                if peer_is_self(peer, options) {
                    continue;
                }
                evidence.resolve(
                    probes.link.df_probe(netdev, peer, DF_PROBE_PAYLOAD_BYTES),
                    |evidence, detail| evidence.pass(detail),
                );
            }
        }
        for peer in &options.nodes {
            if !peer_is_self(peer, options) {
                continue;
            }
            evidence.note(format!(
                "{peer} is this node (rank {}); no probe is sent to it",
                options.rank
            ));
        }
    } else {
        for netdev in &gated_netdevs {
            evidence.skip(format!(
                "{netdev}: do-not-fragment probe skipped, no peer was given (--nodes)"
            ));
        }
    }

    evidence.finish(
        CheckId::LinkMtu,
        format!(
            "two ACTIVE ConnectX-7 rails at {LINK_SPEED_MBPS} Mb/s with MTU {LINK_MTU} and a {DF_PROBE_PAYLOAD_BYTES}-byte do-not-fragment probe"
        ),
    )
}

/// Whether a node in the list is this node.
fn peer_is_self(peer: &str, options: &Options) -> bool {
    peer == "self" || peer == "localhost" || peer == options.rank.to_string()
}

/// Check 3: dual-rail RDMA bandwidth and 10 KiB one-shot All-Reduce latency.
#[must_use]
pub fn check_rdma(probes: &Probes, options: &Options) -> CheckOutcome {
    let mut evidence = Evidence::default();
    if !options.cluster() {
        evidence.skip("no peer was given (--nodes); the RDMA gate did not run");
        return evidence.finish(
            CheckId::RdmaBandwidth,
            format!(
                "dual-rail RDMA aggregate >= {MIN_AGGREGATE_RDMA_GBPS} Gbps and 10 KiB one-shot All-Reduce < {MAX_ONESHOT_AR_LATENCY_US} us"
            ),
        );
    }
    let peers: Vec<String> = options
        .nodes
        .iter()
        .filter(|peer| !peer_is_self(peer, options))
        .cloned()
        .collect();
    // Bandwidth: one reading per peer, scored at the worst peer. A mean across ranks would pass a
    // cluster with one degraded cable, which is the failure this gate exists to catch.
    evidence.resolve(
        probes.rdma.dual_rail_write_gbps(&peers, options.rdma_seconds),
        |evidence, readings| {
            if readings.is_empty() {
                evidence.fail(
                    "the RDMA bandwidth probe reported no peer readings",
                    "point --nodes at the other three nodes so every peer is measured, then re-run",
                );
                return;
            }
            let measured = readings
                .iter()
                .map(|reading| format!("{} {:.1} Gbps", reading.peer, reading.gbps))
                .collect::<Vec<_>>()
                .join(", ");
            let worst = readings
                .iter()
                .min_by(|left, right| left.gbps.total_cmp(&right.gbps))
                .unwrap_or(&readings[0]);
            if worst.gbps >= MIN_AGGREGATE_RDMA_GBPS {
                evidence.pass(format!(
                    "aggregate dual-rail bandwidth {:.1} Gbps worst-case over {} peers probed for {} s: {measured}",
                    worst.gbps,
                    readings.len(),
                    options.rdma_seconds
                ));
            } else {
                evidence.fail(
                    format!(
                        "aggregate dual-rail bandwidth {:.1} Gbps to {} is below {MIN_AGGREGATE_RDMA_GBPS} Gbps; measured {measured}",
                        worst.gbps, worst.peer
                    ),
                    format!(
                        "raise aggregate bandwidth to >= {MIN_AGGREGATE_RDMA_GBPS} Gbps to every peer: check both ConnectX-7 halves are up and that the RoCEnante probe uses both rails"
                    ),
                );
            }
        },
    );
    // Latency: a doctor process can measure exactly one rank's view - its own, named by
    // --rank. One process cannot observe another rank's latency, and pretending otherwise (by
    // looping over rank views with a fabricated address table) produced a check that could never
    // rendezvous on a healthy cluster. The other ranks must run their probe processes at the same
    // time (Lead orchestration), which is what the failure advice below says.
    let rank = options.rank;
    match probes
        .rdma
        .oneshot_allreduce_us(rank, &peers, ONESHOT_AR_BYTES)
    {
        Ok(micros) => {
            let peers_text = if peers.is_empty() {
                "no peer".to_owned()
            } else {
                peers.join(", ")
            };
            if micros < MAX_ONESHOT_AR_LATENCY_US {
                evidence.pass(format!(
                    "10 KiB one-shot All-Reduce {micros:.2} us on rank {rank} (< {MAX_ONESHOT_AR_LATENCY_US} us); peers {peers_text}"
                ));
            } else {
                evidence.fail(
                    format!(
                        "10 KiB one-shot All-Reduce {micros:.2} us on rank {rank}, expected < {MAX_ONESHOT_AR_LATENCY_US} us; peers {peers_text}"
                    ),
                    format!(
                        "get the 10 KiB one-shot All-Reduce below {MAX_ONESHOT_AR_LATENCY_US} us: check the RoCEnante staging fence, peer flag wait and canonical-order reduce"
                    ),
                );
            }
        }
        Err(ProbeError::NotApplicable(message)) | Err(ProbeError::Unavailable(message)) => {
            // Any unmeasurable probe fails the check: a rank that could not be measured is not
            // evidence that it met the bar.
            evidence.fail(
                format!("rank {rank}'s 10 KiB one-shot All-Reduce could not be measured: {message}"),
                "start the probe on every node at the same time (the control plane rendezvous is the barrier), pass --rank for this node, then re-run",
            );
        }
    }
    evidence.finish(
        CheckId::RdmaBandwidth,
        format!(
            "dual-rail RDMA aggregate >= {MIN_AGGREGATE_RDMA_GBPS} Gbps and 10 KiB one-shot All-Reduce < {MAX_ONESHOT_AR_LATENCY_US} us"
        ),
    )
}

/// Check 4: lockable SM clocks and available memory.
#[must_use]
pub fn check_clocks_memory(probes: &Probes, options: &Options) -> CheckOutcome {
    let mut evidence = Evidence::default();
    evidence.resolve(probes.gpu.sm_clock_range_mhz(), |evidence, range| {
        // MIN_SM_CLOCK_MHZ is zero, so no u64 minimum can violate the floor; the gate is the
        // cap, plus a sanity check that the probe did not read the window backwards.
        if range.max_mhz <= MAX_SM_CLOCK_MHZ && range.min_mhz <= range.max_mhz {
            evidence.pass(format!(
                "SM clocks lockable within [{}, {}] MHz (device reports {}..{} MHz)",
                MIN_SM_CLOCK_MHZ, MAX_SM_CLOCK_MHZ, range.min_mhz, range.max_mhz
            ));
        } else {
            evidence.fail(
                format!(
                    "SM clock range {}..{} MHz is outside the permitted lock window [{}, {}] MHz",
                    range.min_mhz, range.max_mhz, MIN_SM_CLOCK_MHZ, MAX_SM_CLOCK_MHZ
                ),
                format!(
                    "lock SM clocks to {MIN_SM_CLOCK_MHZ},{MAX_SM_CLOCK_MHZ} to avoid EC overcurrent trips"
                ),
            );
        }
    });
    // Locking clocks mutates a node's clock policy, and this preflight is expected to run
    // against a cluster that is serving a live model: never lock without both the master
    // --apply switch and the action-specific --lock-sm-clocks opt-in.
    if options.may_lock_clocks() {
        evidence.resolve(
            probes
                .gpu
                .lock_sm_clocks(MIN_SM_CLOCK_MHZ, MAX_SM_CLOCK_MHZ),
            |evidence, detail| evidence.pass(format!("--lock-sm-clocks: {detail}")),
        );
    } else {
        evidence.note(format!(
            "read-only: `nvidia-smi -lgc {MIN_SM_CLOCK_MHZ},{MAX_SM_CLOCK_MHZ}` not run, because locking SM clocks changes this node's clock policy while it may be serving traffic; pass --apply --lock-sm-clocks to do it deliberately"
        ));
    }
    evidence.resolve(probes.gpu.mem_available_gib(), |evidence, gib| {
        if gib >= MIN_MEM_AVAILABLE_GIB {
            evidence.pass(format!(
                "MemAvailable {gib} GiB >= {MIN_MEM_AVAILABLE_GIB} GiB"
            ));
        } else {
            evidence.fail(
                format!("MemAvailable {gib} GiB, expected >= {MIN_MEM_AVAILABLE_GIB} GiB"),
                format!(
                    "free unified memory until MemAvailable >= {MIN_MEM_AVAILABLE_GIB} GiB (stop other engines, drop page cache) before loading weights"
                ),
            );
        }
    });
    evidence.finish(
        CheckId::ClocksMemory,
        format!(
            "SM clocks lockable within [{MIN_SM_CLOCK_MHZ}, {MAX_SM_CLOCK_MHZ}] MHz and MemAvailable >= {MIN_MEM_AVAILABLE_GIB} GiB"
        ),
    )
}

/// Expected complete size of one rank's shard for `layer_index`, padding included.
///
/// Every rank's file has the same length: the packer shards with `div_ceil`, zero-pads the
/// last rank's short slice, and pads the whole file to [`O_DIRECT_ALIGNMENT`].
#[must_use]
pub fn expected_shard_bytes(layer_index: usize, world_size: usize) -> u64 {
    let rows = sparkflash_engram::layout::rows_per_rank(layer_index, world_size);
    let data = rows * sparkflash_engram::layout::ENGRAM_ROW_BYTES as u64;
    align_up(data, O_DIRECT_ALIGNMENT)
}

/// File name of one rank's shard for a layer number.
#[must_use]
pub fn shard_file_name(layer: usize, rank: usize, world_size: usize) -> String {
    format!("engram-l{layer}-r{rank}of{world_size}.bin")
}

/// Check 5: local NVMe Engram shards.
#[must_use]
pub fn check_engram_shards(probes: &Probes, options: &Options) -> CheckOutcome {
    let mut evidence = Evidence::default();
    if !options.live() {
        evidence.skip("no hardware probe was requested (--local or --nodes)");
        return evidence.finish(
            CheckId::EngramShards,
            "both rank shards present, 512-byte aligned on local NVMe and matching the manifest SHA256",
        );
    }
    let mut manifest: Option<EngramManifest> = None;
    let mut manifest_missing = false;
    match probes.shards.manifest(&options.engram_dir) {
        Ok(found) => {
            if found.entries.is_empty() {
                evidence.fail(
                    format!(
                        "{} lists no shards",
                        options.engram_dir.join(MANIFEST_NAME).display()
                    ),
                    "re-run sparkflash pack-engram to write the manifest",
                );
            } else {
                evidence.pass(format!(
                    "{} lists {} shard(s)",
                    options.engram_dir.join(MANIFEST_NAME).display(),
                    found.entries.len()
                ));
            }
            manifest = Some(found);
        }
        Err(ProbeError::NotApplicable(message)) => {
            manifest_missing = true;
            evidence.skip(format!("shard verification not applicable: {message}"));
        }
        Err(ProbeError::Unavailable(message)) => evidence.fail(
            message,
            "re-run sparkflash pack-engram so the manifest is readable",
        ),
    }
    for (layer_index, layer) in sparkflash_engram::layout::ENGRAM_LAYERS.iter().enumerate() {
        let name = shard_file_name(*layer, options.rank, options.world_size);
        let path = options.engram_dir.join(&name);
        let expected = expected_shard_bytes(layer_index, options.world_size);
        match probes.shards.file_len(&path) {
            Ok(actual) => {
                if actual == expected {
                    evidence.pass(format!("{name}: {actual} bytes as planned"));
                } else {
                    evidence.fail(
                        format!("{name}: {actual} bytes, plan says {expected}"),
                        format!("re-run sparkflash pack-engram for {name}"),
                    );
                }
                if actual % O_DIRECT_ALIGNMENT == 0 {
                    evidence.pass(format!(
                        "{name}: {actual} bytes is a multiple of the {O_DIRECT_ALIGNMENT}-byte O_DIRECT alignment"
                    ));
                } else {
                    evidence.fail(
                        format!(
                            "{name}: {actual} bytes is not {O_DIRECT_ALIGNMENT}-byte aligned"
                        ),
                        format!("re-run sparkflash pack-engram: it pads every file to {O_DIRECT_ALIGNMENT} bytes"),
                    );
                }
                match manifest.as_ref().and_then(|m| m.entry(&name)) {
                    Some(entry) => {
                        if let Some(recorded) = entry.file_bytes.filter(|recorded| *recorded != actual) {
                            evidence.fail(
                                format!("{name}: manifest says {recorded} bytes, file is {actual}"),
                                "re-run sparkflash pack-engram; the manifest is stale",
                            );
                        }
                        match &entry.sha256 {
                            Some(expected_hash) if options.skip_hash => {
                                evidence.note(format!(
                                    "{name}: SHA256 check skipped (--skip-hash), manifest records {expected_hash}"
                                ));
                            }
                            Some(expected_hash) => {
                                evidence.resolve(probes.shards.sha256(&path), |evidence, actual_hash| {
                                    if actual_hash.eq_ignore_ascii_case(expected_hash) {
                                        evidence.pass(format!("{name}: SHA256 matches the manifest"));
                                    } else {
                                        evidence.fail(
                                            format!("{name}: SHA256 {actual_hash} does not match manifest {expected_hash}"),
                                            "re-run sparkflash pack-engram; the shard is corrupt or truncated",
                                        );
                                    }
                                });
                            }
                            None => evidence.fail(
                                format!("{name}: the manifest records no SHA256"),
                                "re-run sparkflash pack-engram so the manifest carries per-file SHA256",
                            ),
                        }
                    }
                    None if manifest_missing => evidence.skip(format!(
                        "{name}: presence, size and alignment checked, but no manifest means the SHA256 was not verified"
                    )),
                    None => evidence.fail(
                        format!("{name}: not listed in {MANIFEST_NAME}"),
                        "re-run sparkflash pack-engram so every rank file is in the manifest",
                    ),
                }
                match probes.shards.mount(&path) {
                    Ok(mount) => {
                        let on_nvme = mount.source.starts_with("/dev/nvme");
                        let network = matches!(
                            mount.fstype.as_str(),
                            "nfs" | "nfs4" | "cifs" | "smb3" | "fuse.sshfs"
                        );
                        if on_nvme && !network {
                            evidence.pass(format!("{name}: local NVMe ({})", mount.source));
                        } else {
                            evidence.fail(
                                format!(
                                    "{name}: lives on {} ({}) which is not local NVMe",
                                    mount.source, mount.fstype
                                ),
                                "put the Engram shards on local PCIe Gen5 NVMe",
                            );
                        }
                    }
                    Err(ProbeError::NotApplicable(message)) => evidence.skip(message),
                    Err(ProbeError::Unavailable(message)) => {
                        evidence.fail(message, "check the mount of the Engram directory")
                    }
                }
            }
            Err(ProbeError::NotApplicable(message)) => evidence.skip(message),
            Err(ProbeError::Unavailable(message)) => evidence.fail(
                message,
                format!(
                    "run sparkflash pack-engram to create {name} under {}",
                    options.engram_dir.display()
                ),
            ),
        }
    }
    evidence.finish(
        CheckId::EngramShards,
        "both rank shards present, 512-byte aligned on local NVMe and matching the manifest SHA256",
    )
}

/// Runs all five checks in specification order.
#[must_use]
pub fn execute(options: &Options, probes: &Probes) -> Report {
    let outcomes = vec![
        check_driver_kernel(probes),
        check_link_mtu(probes, options),
        check_rdma(probes, options),
        check_clocks_memory(probes, options),
        check_engram_shards(probes, options),
    ];
    Report::from_outcomes(outcomes)
}

/// Runs the command with no hardware probes: the dev-box default.
#[must_use]
pub fn execute_default(options: &Options) -> Report {
    let reason = if options.live() {
        "probe bundle was not constructed"
    } else {
        "no hardware probe requested: pass --local for this node or --nodes for the cluster"
    };
    execute(options, &Probes::not_applicable(reason))
}

/// Entry point main.rs calls: parses arguments and prints the report.
pub fn run(args: impl IntoIterator<Item = String>) -> ExitCode {
    let cli = match DoctorArgs::try_parse_from(
        std::iter::once("sparkflash doctor".to_owned()).chain(args),
    ) {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            return ExitCode::from(2);
        }
    };
    let options = match cli.into_options() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("sparkflash doctor: {message}");
            return ExitCode::from(2);
        }
    };
    let probes = if options.live() {
        Probes::system()
    } else {
        Probes::not_applicable(
            "no hardware probe requested: pass --local for this node or --nodes for the cluster",
        )
    };
    let report = execute(&options, &probes);
    if options.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report.to_json()).unwrap_or_else(|_| "{}".to_owned())
        );
    } else {
        print!("{}", report.to_human());
    }
    ExitCode::from(report.exit_code())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    /// Plan sizes the M2 packer produces for rank 0 of 4: rows * 264 rounded up to 512.
    const SHARD_L1: u64 = 25_344_407_552;
    /// Same, for the layer-14 table.
    const SHARD_L14: u64 = 25_345_101_312;
    /// The digest every fake shard and manifest entry agrees on.
    const DIGEST: &str = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0";

    /// Records every state-changing call, so tests can prove read-only mode.
    #[derive(Debug, Default)]
    struct CallLog(Mutex<Vec<String>>);

    impl CallLog {
        fn record(&self, entry: impl Into<String>) {
            self.0.lock().expect("call log").push(entry.into());
        }
        fn entries(&self) -> Vec<String> {
            self.0.lock().expect("call log").clone()
        }
    }

    /// Per-file shard readings, keyed by file name.
    #[derive(Default)]
    struct ShardTable {
        lens: BTreeMap<String, Probe<u64>>,
        digests: BTreeMap<String, Probe<String>>,
        default_len: Option<Probe<u64>>,
        default_digest: Option<Probe<String>>,
    }

    impl ShardTable {
        fn len_for(&self, name: &str) -> Probe<u64> {
            self.lens
                .get(name)
                .cloned()
                .or_else(|| self.default_len.clone())
                .unwrap_or_else(|| Err(ProbeError::Unavailable(format!("{name} is missing"))))
        }

        fn digest_for(&self, name: &str) -> Probe<String> {
            self.digests
                .get(name)
                .cloned()
                .or_else(|| self.default_digest.clone())
                .unwrap_or_else(|| Err(ProbeError::Unavailable(format!("{name} is missing"))))
        }
    }

    /// A fully passing cluster whose readings can be overridden one at a time.
    struct Fake {
        driver_version: Probe<String>,
        kernel_release: Probe<String>,
        uvm: Probe<UvmPacking>,
        inventory: Probe<Vec<RailStatus>>,
        speed: Probe<u64>,
        mtu: Probe<u32>,
        df: Probe<String>,
        gbps: Probe<f64>,
        allreduce: Probe<f64>,
        clocks: Probe<ClockRange>,
        lock_result: Probe<String>,
        mem_gib: Probe<u64>,
        shards: ShardTable,
        mount: Probe<MountInfo>,
        manifest: Probe<EngramManifest>,
        calls: Arc<CallLog>,
    }

    fn manifest_with(digest: &str) -> EngramManifest {
        EngramManifest {
            entries: vec![
                ManifestEntry {
                    file: "engram-l1-r0of4.bin".to_owned(),
                    sha256: Some(digest.to_owned()),
                    file_bytes: Some(SHARD_L1),
                },
                ManifestEntry {
                    file: "engram-l14-r0of4.bin".to_owned(),
                    sha256: Some(digest.to_owned()),
                    file_bytes: Some(SHARD_L14),
                },
            ],
        }
    }

    /// Two spec-named rails, ACTIVE at the required speed, one per card.
    fn happy_inventory() -> Vec<RailStatus> {
        RAILS
            .iter()
            .enumerate()
            .map(|(index, rail)| RailStatus {
                rdma: rail.rdma.to_owned(),
                netdev: Some(rail.netdev.to_owned()),
                state: "ACTIVE".to_owned(),
                speed_mbps: Some(LINK_SPEED_MBPS),
                pci: Some(format!("0000:0{}:00", index + 1)),
            })
            .collect()
    }

    impl Fake {
        fn happy() -> Self {
            let mut shards = ShardTable::default();
            shards
                .lens
                .insert("engram-l1-r0of4.bin".to_owned(), Ok(SHARD_L1));
            shards
                .lens
                .insert("engram-l14-r0of4.bin".to_owned(), Ok(SHARD_L14));
            shards
                .digests
                .insert("engram-l1-r0of4.bin".to_owned(), Ok(DIGEST.to_owned()));
            shards
                .digests
                .insert("engram-l14-r0of4.bin".to_owned(), Ok(DIGEST.to_owned()));
            Self {
                driver_version: Ok(MIN_DRIVER_VERSION.to_owned()),
                kernel_release: Ok("6.17.0-1018-nvidia-64k".to_owned()),
                uvm: Ok(UvmPacking {
                    state: UvmPagePacking::Enabled,
                    parameters_seen: Some(56),
                    candidates: Vec::new(),
                }),
                inventory: Ok(happy_inventory()),
                speed: Ok(LINK_SPEED_MBPS),
                mtu: Ok(LINK_MTU),
                df: Ok("probe ok".to_owned()),
                gbps: Ok(200.0),
                allreduce: Ok(20.0),
                clocks: Ok(ClockRange {
                    min_mhz: 0,
                    max_mhz: MAX_SM_CLOCK_MHZ,
                }),
                lock_result: Ok("locked".to_owned()),
                mem_gib: Ok(120),
                shards,
                mount: Ok(MountInfo {
                    source: "/dev/nvme0n1p2".to_owned(),
                    fstype: "ext4".to_owned(),
                }),
                manifest: Ok(manifest_with(DIGEST)),
                calls: Arc::new(CallLog::default()),
            }
        }
    }

    struct FakeDriver(Arc<Fake>);
    struct FakeLink(Arc<Fake>);
    struct FakeRdma(Arc<Fake>);
    struct FakeGpu(Arc<Fake>);
    struct FakeShards(Arc<Fake>);

    impl DriverProbe for FakeDriver {
        fn driver_version(&self) -> Probe<String> {
            self.0.driver_version.clone()
        }
        fn kernel_release(&self) -> Probe<String> {
            self.0.kernel_release.clone()
        }
        fn uvm_page_packing(&self) -> Probe<UvmPacking> {
            self.0.uvm.clone()
        }
    }

    impl LinkProbe for FakeLink {
        fn rail_inventory(&self) -> Probe<Vec<RailStatus>> {
            self.0.inventory.clone()
        }
        fn link_speed_mbps(&self, _netdev: &str) -> Probe<u64> {
            self.0.speed.clone()
        }
        fn mtu(&self, _netdev: &str) -> Probe<u32> {
            self.0.mtu.clone()
        }
        fn df_probe(&self, _netdev: &str, _peer: &str, _payload_bytes: usize) -> Probe<String> {
            self.0.df.clone()
        }
    }

    impl RdmaProbe for FakeRdma {
        fn dual_rail_write_gbps(
            &self,
            peers: &[String],
            seconds: u64,
        ) -> Probe<Vec<PeerBandwidth>> {
            self.0
                .calls
                .record(format!("rdma-bandwidth peers={peers:?} seconds={seconds}"));
            self.0.gbps.clone().map(|gbps| {
                peers
                    .iter()
                    .map(|peer| PeerBandwidth {
                        peer: peer.clone(),
                        gbps,
                    })
                    .collect()
            })
        }
        fn oneshot_allreduce_us(&self, rank: usize, peers: &[String], bytes: usize) -> Probe<f64> {
            self.0.calls.record(format!(
                "rdma-allreduce rank={rank} peers={peers:?} bytes={bytes}"
            ));
            self.0.allreduce.clone()
        }
    }

    impl GpuProbe for FakeGpu {
        fn sm_clock_range_mhz(&self) -> Probe<ClockRange> {
            self.0.clocks.clone()
        }
        fn lock_sm_clocks(&self, min_mhz: u64, max_mhz: u64) -> Probe<String> {
            self.0
                .calls
                .record(format!("nvidia-smi -lgc {min_mhz},{max_mhz}"));
            self.0.lock_result.clone()
        }
        fn mem_available_gib(&self) -> Probe<u64> {
            self.0.mem_gib.clone()
        }
    }

    impl ShardProbe for FakeShards {
        fn file_len(&self, path: &Path) -> Probe<u64> {
            self.0.shards.len_for(&name_of(path))
        }
        fn sha256(&self, path: &Path) -> Probe<String> {
            self.0.shards.digest_for(&name_of(path))
        }
        fn mount(&self, _path: &Path) -> Probe<MountInfo> {
            self.0.mount.clone()
        }
        fn manifest(&self, _dir: &Path) -> Probe<EngramManifest> {
            self.0.manifest.clone()
        }
    }

    fn name_of(path: &Path) -> String {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    fn fake_probes(fake: &Arc<Fake>) -> Probes {
        Probes {
            driver: Box::new(FakeDriver(Arc::clone(fake))),
            link: Box::new(FakeLink(Arc::clone(fake))),
            rdma: Box::new(FakeRdma(Arc::clone(fake))),
            gpu: Box::new(FakeGpu(Arc::clone(fake))),
            shards: Box::new(FakeShards(Arc::clone(fake))),
        }
    }

    fn cluster() -> Options {
        Options {
            nodes: vec!["192.0.2.84".to_owned()],
            local: true,
            ..Options::default()
        }
    }

    fn status_of(report: &Report, id: CheckId) -> CheckStatus {
        report
            .outcomes
            .iter()
            .find(|outcome| outcome.id == id)
            .expect("check present")
            .status
    }

    fn outcome_of(report: &Report, id: CheckId) -> &CheckOutcome {
        report
            .outcomes
            .iter()
            .find(|outcome| outcome.id == id)
            .expect("check present")
    }

    // ---- version arithmetic -------------------------------------------------------------

    #[test]
    fn version_comparison_is_component_wise() {
        assert_eq!(
            compare_versions("580.178.04", "580.178.04"),
            Some(Ordering::Equal)
        );
        assert_eq!(
            compare_versions("580.179", "580.178.04"),
            Some(Ordering::Greater)
        );
        assert_eq!(
            compare_versions("580.178.03", "580.178.04"),
            Some(Ordering::Less)
        );
        // A missing component counts as zero: 580.178 == 580.178.0 < 580.178.04.
        assert_eq!(
            compare_versions("580.178", "580.178.04"),
            Some(Ordering::Less)
        );
        assert_eq!(compare_versions("nonsense", "580.178.04"), None);
    }

    #[test]
    fn driver_minimum_boundary_is_exact() {
        assert!(driver_version_problem(MIN_DRIVER_VERSION).is_none());
        assert!(driver_version_problem("580.178.05").is_none());
        assert!(driver_version_problem("580.179").is_none());
        assert!(driver_version_problem("580.178.03").is_some());
        assert!(driver_version_problem("580.177.99").is_some());
    }

    #[test]
    fn rejected_driver_is_rejected_and_is_also_below_the_minimum() {
        // The task text called 580.142 "numerically higher"; it is not. Both guards are kept:
        // the explicit denylist AND the minimum, so a future renumbering cannot silently admit
        // a driver the specification calls out as harmful.
        assert!(REJECTED_DRIVER_VERSIONS.contains(&"580.142"));
        let problem = driver_version_problem("580.142").expect("580.142 must be rejected");
        assert!(problem.contains("13 Gbps"), "{problem}");
        assert_eq!(
            compare_versions("580.142", MIN_DRIVER_VERSION),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn kernel_boundaries_accept_both_branches() {
        for accepted in [
            "6.17.0-1018-nvidia-64k",
            "6.17.0-1018-generic",
            "6.17.1-1018",
            "6.18.0-1018-nvidia-64k",
            "7.0.0-1019-nvidia-64k",
        ] {
            let parsed = parse_kernel_release(accepted).expect("parseable");
            assert!(
                kernel_release_problem(&parsed).is_none(),
                "{accepted} should pass"
            );
        }
        for refused in [
            "6.17.0-1017-nvidia-64k",
            "6.16.0-1018-nvidia-64k",
            "6.16.99-1018-nvidia-64k",
            "7.0.0-1018-nvidia-64k",
        ] {
            let parsed = parse_kernel_release(refused).expect("parseable");
            assert!(
                kernel_release_problem(&parsed).is_some(),
                "{refused} should fail"
            );
        }
    }

    #[test]
    fn kernel_release_parsing_rejects_junk() {
        assert!(parse_kernel_release("").is_none());
        assert!(parse_kernel_release("linux").is_none());
        assert!(parse_kernel_release("6").is_none());
        let parsed = parse_kernel_release("6.17.0-1018-nvidia-64k").expect("parseable");
        assert_eq!(parsed.base, vec![6, 17, 0]);
        assert_eq!(parsed.abi, Some(1018));
    }

    #[test]
    fn alignment_rounds_up_to_the_documented_stride() {
        assert_eq!(align_up(0, O_DIRECT_ALIGNMENT), 0);
        assert_eq!(align_up(1, O_DIRECT_ALIGNMENT), 512);
        assert_eq!(align_up(512, O_DIRECT_ALIGNMENT), 512);
        assert_eq!(align_up(513, O_DIRECT_ALIGNMENT), 1024);
    }

    #[test]
    fn expected_shard_sizes_match_the_m2_plan() {
        assert_eq!(expected_shard_bytes(0, 4), SHARD_L1);
        assert_eq!(expected_shard_bytes(1, 4), SHARD_L14);
        let total_gib = (SHARD_L1 + SHARD_L14) as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            (47.0..48.5).contains(&total_gib),
            "per-node shard total {total_gib} GiB"
        );
        assert_eq!(shard_file_name(1, 0, 4), "engram-l1-r0of4.bin");
        assert_eq!(shard_file_name(14, 3, 4), "engram-l14-r3of4.bin");
    }

    // ---- check 1 ------------------------------------------------------------------------

    #[test]
    fn driver_kernel_check_passes_on_a_healthy_node() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_driver_kernel(&fake_probes(&fake));
        assert_eq!(outcome.status, CheckStatus::Pass);
        assert!(outcome.remediation.is_none());
    }

    #[test]
    fn driver_kernel_check_fails_on_old_driver_with_remediation() {
        let mut fake = Fake::happy();
        fake.driver_version = Ok("580.177.00".to_owned());
        let outcome = check_driver_kernel(&fake_probes(&Arc::new(fake)));
        assert_eq!(outcome.status, CheckStatus::Fail);
        let remediation = outcome.remediation.expect("remediation");
        assert!(remediation.contains(MIN_DRIVER_VERSION), "{remediation}");
    }

    #[test]
    fn driver_kernel_check_fails_on_bad_kernel_and_unknown_uvm() {
        let mut fake = Fake::happy();
        fake.kernel_release = Ok("6.16.0-1018-nvidia-64k".to_owned());
        fake.uvm = Ok(UvmPacking {
            state: UvmPagePacking::Unknown,
            parameters_seen: Some(56),
            candidates: Vec::new(),
        });
        let outcome = check_driver_kernel(&fake_probes(&Arc::new(fake)));
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert_eq!(outcome.details.len(), 3);
        assert!(outcome.details.iter().any(|d| d.contains("6.16.0")));
        assert!(
            outcome
                .details
                .iter()
                .any(|d| d.contains("56 module parameters listed"))
        );
    }

    #[test]
    fn driver_kernel_check_reports_not_applicable_by_default() {
        let outcome = check_driver_kernel(&Probes::not_applicable("dev box"));
        assert_eq!(outcome.status, CheckStatus::NotApplicable);
        assert!(outcome.summary.starts_with("DID NOT RUN"));
    }

    // ---- check 2 ------------------------------------------------------------------------

    #[test]
    fn link_check_passes_at_the_exact_thresholds() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_link_mtu(&fake_probes(&fake), &cluster());
        assert_eq!(outcome.status, CheckStatus::Pass);
        let details = outcome.details.join("\n");
        assert!(details.contains("enp1s0f0np0: 200000 Mb/s"), "{details}");
        assert!(details.contains("enP2p1s0f0np0: 200000 Mb/s"), "{details}");
        assert!(details.contains("enp1s0f0np0: MTU 9000"), "{details}");
        assert!(details.contains("probe ok"), "{details}");
    }

    #[test]
    fn link_check_fails_one_mbps_short() {
        let mut fake = Fake::happy();
        fake.speed = Ok(LINK_SPEED_MBPS - 1);
        let outcome = check_link_mtu(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(outcome.remediation.expect("remediation").contains("200000"));
    }

    #[test]
    fn link_check_fails_on_mtu_8999() {
        let mut fake = Fake::happy();
        fake.mtu = Ok(LINK_MTU - 1);
        let outcome = check_link_mtu(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(outcome.remediation.expect("remediation").contains("9000"));
    }

    #[test]
    fn link_check_fails_when_the_do_not_fragment_probe_fails() {
        let mut fake = Fake::happy();
        fake.df = Err(ProbeError::Unavailable(
            "ping: local error: message too long".to_owned(),
        ));
        let outcome = check_link_mtu(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(outcome.details.iter().any(|d| d.contains("too long")));
    }

    #[test]
    fn link_check_is_not_applicable_without_peers() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_link_mtu(
            &fake_probes(&fake),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::NotApplicable);
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("no peer was given"))
        );
    }

    // ---- check 3 ------------------------------------------------------------------------

    #[test]
    fn rdma_check_passes_at_the_exact_thresholds() {
        let mut fake = Fake::happy();
        fake.gbps = Ok(MIN_AGGREGATE_RDMA_GBPS);
        fake.allreduce = Ok(MAX_ONESHOT_AR_LATENCY_US - 0.01);
        let outcome = check_rdma(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Pass);
    }

    #[test]
    fn rdma_check_fails_just_under_the_bandwidth_gate() {
        let mut fake = Fake::happy();
        fake.gbps = Ok(179.9);
        let outcome = check_rdma(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(outcome.remediation.expect("remediation").contains("both"));
    }

    #[test]
    fn allreduce_latency_gate_is_strictly_less_than_22us() {
        let mut fake = Fake::happy();
        fake.allreduce = Ok(MAX_ONESHOT_AR_LATENCY_US);
        let outcome = check_rdma(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(outcome.remediation.expect("remediation").contains("22"));
    }

    /// The bytes rocenante-probe prints for an explicitly requested in-process loopback run: a real
    /// measurement of the pipeline, labelled as not the fabric. Round 6 of review put a shim on
    /// PATH that ran the real probe with --transport loopback and the doctor scored it as the gate.
    const LOOPBACK_ALLREDUCE_JSON: &str = "{\"bytes\":10240,\"fabric\":false,\"max_us\":695.605,\"min_us\":587.454,\"oneshot_allreduce_us\":678.801,\"p90_us\":683.572,\"peers\":[],\"samples\":20,\"statistic\":\"median\",\"statistic_note\":\"median (p50) of the post-warmup samples\",\"transport\":\"loopback\",\"warmup_discarded\":3}";

    #[test]
    fn a_loopback_labelled_number_is_refused_by_the_reader() {
        match oneshot_allreduce_us_from_probe_json(LOOPBACK_ALLREDUCE_JSON) {
            Ok(micros) => panic!("a loopback number was accepted as the fabric gate: {micros}"),
            Err(ProbeError::Unavailable(text)) => {
                assert!(text.contains("loopback"), "{text}");
                assert!(text.contains("fabric false"), "{text}");
            }
            Err(other) => panic!("expected Unavailable, got {other}"),
        }
    }

    #[test]
    fn the_reader_accepts_only_a_fabric_labelled_number() {
        let fabric = "{\"bytes\":10240,\"fabric\":true,\"oneshot_allreduce_us\":21.25,\"samples\":50,\"transport\":\"rdma\"}";
        assert_eq!(
            oneshot_allreduce_us_from_probe_json(fabric).ok(),
            Some(21.25)
        );
        for unlabelled in [
            "{\"oneshot_allreduce_us\":21.25}",
            "{\"fabric\":true,\"oneshot_allreduce_us\":21.25}",
            "{\"transport\":\"rdma\",\"oneshot_allreduce_us\":21.25}",
            "{\"fabric\":false,\"transport\":\"rdma\",\"oneshot_allreduce_us\":21.25}",
        ] {
            assert!(
                matches!(
                    oneshot_allreduce_us_from_probe_json(unlabelled),
                    Err(ProbeError::Unavailable(_))
                ),
                "a payload without the fabric labels must fail closed: {unlabelled}"
            );
        }
    }

    #[test]
    fn check_rdma_does_not_score_a_loopback_measurement() {
        let mut fake = Fake::happy();
        fake.allreduce = oneshot_allreduce_us_from_probe_json(LOOPBACK_ALLREDUCE_JSON);
        let outcome = check_rdma(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(
            outcome
                .details
                .iter()
                .any(|line| line.contains("loopback") && line.contains("fabric")),
            "check 3 must refuse the loopback payload by name: {:?}",
            outcome.details
        );
    }

    #[test]
    fn rdma_check_is_not_applicable_without_peers() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_rdma(
            &fake_probes(&fake),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::NotApplicable);
    }

    /// Three peers plus this node: the shape check 3 must measure in full.
    fn three_peer_cluster() -> Options {
        Options {
            nodes: vec![
                "192.0.2.84".to_owned(),
                "192.0.2.68".to_owned(),
                "192.0.2.59".to_owned(),
            ],
            ..cluster()
        }
    }

    #[test]
    fn check_rdma_hands_every_peer_to_the_bandwidth_probe() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_rdma(&fake_probes(&fake), &three_peer_cluster());
        assert_eq!(outcome.status, CheckStatus::Pass);
        let calls = fake.calls.entries();
        let bandwidth = calls
            .iter()
            .find(|call| call.starts_with("rdma-bandwidth"))
            .unwrap_or_else(|| panic!("no bandwidth probe was recorded: {calls:?}"));
        for peer in ["192.0.2.84", "192.0.2.68", "192.0.2.59"] {
            assert!(
                bandwidth.contains(peer),
                "the bandwidth probe did not hand {peer} to the measurement: {bandwidth}"
            );
        }
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("worst-case")
                    && detail.contains("192.0.2.84")
                    && detail.contains("192.0.2.59")),
            "the evidence must name every probed peer: {:?}",
            outcome.details
        );
    }

    #[test]
    fn check_rdma_probes_only_this_node_rank_and_scores_it() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_rdma(&fake_probes(&fake), &three_peer_cluster());
        assert_eq!(outcome.status, CheckStatus::Pass);
        let calls = fake.calls.entries();
        let latency_calls: Vec<&String> = calls
            .iter()
            .filter(|call| call.starts_with("rdma-allreduce"))
            .collect();
        assert_eq!(
            latency_calls.len(),
            1,
            "one doctor process must issue exactly one latency probe (its own rank): {calls:?}"
        );
        assert!(
            latency_calls[0].contains("rank=0 "),
            "the probe must use the local rank: {}",
            latency_calls[0]
        );
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("10 KiB one-shot All-Reduce")
                    && detail.contains("rank 0")),
            "the evidence must name the probed rank: {:?}",
            outcome.details
        );
    }

    #[test]
    fn check_rdma_names_the_requested_rank() {
        let fake = Arc::new(Fake::happy());
        let options = Options {
            rank: 2,
            ..three_peer_cluster()
        };
        let outcome = check_rdma(&fake_probes(&fake), &options);
        assert_eq!(outcome.status, CheckStatus::Pass);
        let calls = fake.calls.entries();
        assert!(
            calls
                .iter()
                .any(|call| call.contains("rdma-allreduce rank=2 ")),
            "the requested rank was not probed: {calls:?}"
        );
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("rank 2")),
            "the evidence must name rank 2: {:?}",
            outcome.details
        );
    }

    #[test]
    fn check_rdma_fails_when_this_rank_cannot_be_measured() {
        let mut fake = Fake::happy();
        fake.allreduce = Err(ProbeError::Unavailable(
            "no peer probe is running".to_owned(),
        ));
        let fake = Arc::new(fake);
        let outcome = check_rdma(&fake_probes(&fake), &three_peer_cluster());
        assert_eq!(
            outcome.status,
            CheckStatus::Fail,
            "an unmeasurable rank must fail, not pass on other evidence: {:?}",
            outcome.details
        );
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("rank 0")
                    && detail.contains("no peer probe is running")),
            "the failure must name the rank and the reason: {:?}",
            outcome.details
        );
    }

    // ---- check 4 ------------------------------------------------------------------------

    #[test]
    fn clock_window_accepts_the_documented_cap_and_refuses_one_mhz_more() {
        let fake = Arc::new(Fake::happy());
        assert_eq!(
            check_clocks_memory(
                &fake_probes(&fake),
                &Options {
                    local: true,
                    ..Options::default()
                }
            )
            .status,
            CheckStatus::Pass
        );
        let mut over = Fake::happy();
        over.clocks = Ok(ClockRange {
            min_mhz: 0,
            max_mhz: MAX_SM_CLOCK_MHZ + 1,
        });
        let outcome = check_clocks_memory(
            &fake_probes(&Arc::new(over)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(outcome.remediation.expect("remediation").contains("2200"));
    }

    #[test]
    fn memory_gate_is_114_gib() {
        let mut fake = Fake::happy();
        fake.mem_gib = Ok(MIN_MEM_AVAILABLE_GIB);
        assert_eq!(
            check_clocks_memory(
                &fake_probes(&Arc::new(fake)),
                &Options {
                    local: true,
                    ..Options::default()
                }
            )
            .status,
            CheckStatus::Pass
        );
        let mut short = Fake::happy();
        short.mem_gib = Ok(MIN_MEM_AVAILABLE_GIB - 1);
        let outcome = check_clocks_memory(
            &fake_probes(&Arc::new(short)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(outcome.remediation.expect("remediation").contains("114"));
    }

    #[test]
    fn clock_lock_requires_both_the_master_switch_and_the_named_action() {
        // Locking clocks mutates a node that may be serving a live model: --apply alone must
        // not do it, and neither must the action flag alone.
        for (apply, lock, expected_calls, expect_read_only_note) in [
            (false, false, 0_usize, true),
            (true, false, 0, true),
            (false, true, 0, true),
            (true, true, 1, false),
        ] {
            let fake = Arc::new(Fake::happy());
            let options = Options {
                local: true,
                apply,
                lock_sm_clocks: lock,
                ..Options::default()
            };
            let outcome = check_clocks_memory(&fake_probes(&fake), &options);
            assert_eq!(
                fake.calls.entries().len(),
                expected_calls,
                "apply={apply} lock={lock} must not lock clocks"
            );
            let note = outcome
                .details
                .iter()
                .any(|detail| detail.contains("read-only"));
            assert_eq!(note, expect_read_only_note, "apply={apply} lock={lock}");
            assert_eq!(outcome.status, CheckStatus::Pass);
        }
    }

    #[test]
    fn read_only_run_says_so_in_the_evidence() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_clocks_memory(
            &fake_probes(&fake),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("read-only") && detail.contains("--apply")),
            "{:?}",
            outcome.details
        );
    }

    // ---- check 5 ------------------------------------------------------------------------

    #[test]
    fn shard_check_passes_with_a_matching_manifest() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_engram_shards(
            &fake_probes(&fake),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Pass, "{:?}", outcome.details);
    }

    #[test]
    fn shard_check_fails_when_a_shard_is_missing() {
        let mut fake = Fake::happy();
        fake.shards.lens.clear();
        fake.shards.default_len = Some(Err(ProbeError::Unavailable(
            "engram-l1-r0of4.bin is missing".to_owned(),
        )));
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(
            outcome
                .remediation
                .expect("remediation")
                .contains("pack-engram")
        );
    }

    #[test]
    fn shard_check_fails_on_a_size_disagreement() {
        let mut fake = Fake::happy();
        fake.shards.lens.insert(
            "engram-l1-r0of4.bin".to_owned(),
            Ok(SHARD_L1 - O_DIRECT_ALIGNMENT),
        );
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("plan says"))
        );
    }

    #[test]
    fn shard_check_fails_on_an_unaligned_file() {
        let mut fake = Fake::happy();
        // Same size, but not a multiple of the O_DIRECT alignment: 511 extra bytes.
        fake.shards.lens.insert(
            "engram-l1-r0of4.bin".to_owned(),
            Ok(SHARD_L1 - O_DIRECT_ALIGNMENT + 1),
        );
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("not 512-byte aligned")),
            "{:?}",
            outcome.details
        );
    }

    #[test]
    fn shard_check_fails_on_a_digest_mismatch() {
        let mut fake = Fake::happy();
        fake.shards
            .digests
            .insert("engram-l14-r0of4.bin".to_owned(), Ok("deadbeef".to_owned()));
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("does not match manifest"))
        );
    }

    #[test]
    fn shard_check_skips_integrity_when_the_manifest_is_missing() {
        let mut fake = Fake::happy();
        fake.manifest = Err(ProbeError::NotApplicable(
            "manifest.json does not exist".to_owned(),
        ));
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        // Presence, size and alignment were verified; integrity was not, so this is explicitly
        // not a pass and explicitly not a failure.
        assert_eq!(outcome.status, CheckStatus::NotApplicable);
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("no manifest means the SHA256 was not verified"))
        );
        assert!(outcome.remediation.is_none());
    }

    #[test]
    fn shard_check_fails_when_the_manifest_lists_nothing() {
        let mut fake = Fake::happy();
        fake.manifest = Ok(EngramManifest::default());
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
    }

    #[test]
    fn shard_check_fails_when_a_file_is_absent_from_the_manifest() {
        let mut fake = Fake::happy();
        fake.manifest = Ok(EngramManifest {
            entries: vec![ManifestEntry {
                file: "engram-l1-r0of4.bin".to_owned(),
                sha256: Some(DIGEST.to_owned()),
                file_bytes: Some(SHARD_L1),
            }],
        });
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
    }

    #[test]
    fn shard_check_fails_on_a_network_mount() {
        let mut fake = Fake::happy();
        fake.mount = Ok(MountInfo {
            source: "nfs-server:/engram".to_owned(),
            fstype: "nfs4".to_owned(),
        });
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("not local NVMe"))
        );
    }

    #[test]
    fn shard_check_fails_when_the_digest_tool_is_unavailable() {
        let mut fake = Fake::happy();
        fake.shards.default_digest = Some(Err(ProbeError::Unavailable(
            "sha256sum could not be run".to_owned(),
        )));
        fake.shards.digests.clear();
        let outcome = check_engram_shards(
            &fake_probes(&Arc::new(fake)),
            &Options {
                local: true,
                ..Options::default()
            },
        );
        assert_eq!(outcome.status, CheckStatus::Fail);
    }

    #[test]
    fn skip_hash_leaves_the_digest_probe_alone() {
        let mut fake = Fake::happy();
        fake.shards.digests.clear();
        fake.shards.default_digest = Some(Err(ProbeError::Unavailable(
            "sha256sum must not be called".to_owned(),
        )));
        let options = Options {
            local: true,
            skip_hash: true,
            ..Options::default()
        };
        let outcome = check_engram_shards(&fake_probes(&Arc::new(fake)), &options);
        assert_eq!(outcome.status, CheckStatus::Pass);
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("skipped (--skip-hash)"))
        );
    }

    #[test]
    fn shard_check_is_not_applicable_on_a_dev_box() {
        let outcome = check_engram_shards(&Probes::not_applicable("dev box"), &Options::default());
        assert_eq!(outcome.status, CheckStatus::NotApplicable);
    }

    // ---- report and verdict ---------------------------------------------------------------

    #[test]
    fn default_invocation_reports_every_check_not_applicable_and_exits_zero() {
        let report = execute_default(&Options::default());
        assert_eq!(report.outcomes.len(), IMPLEMENTED_CHECK_COUNT);
        assert!(
            report
                .outcomes
                .iter()
                .all(|outcome| outcome.status == CheckStatus::NotApplicable),
            "{:?}",
            report.outcomes
        );
        assert_eq!(report.verdict, Verdict::Incomplete);
        assert_eq!(report.exit_code(), 0);
        assert_eq!(report.counts(), (0, 0, 5));
    }

    #[test]
    fn a_healthy_cluster_is_ready_and_exits_zero() {
        let fake = Arc::new(Fake::happy());
        let report = execute(&cluster(), &fake_probes(&fake));
        assert_eq!(report.verdict, Verdict::Ready);
        assert_eq!(report.exit_code(), 0);
        assert_eq!(report.counts(), (5, 0, 0));
    }

    #[test]
    fn a_single_failure_makes_the_run_not_ready_and_exits_one() {
        let mut fake = Fake::happy();
        fake.mem_gib = Ok(113);
        let report = execute(&cluster(), &fake_probes(&Arc::new(fake)));
        assert_eq!(report.verdict, Verdict::NotReady);
        assert_eq!(report.exit_code(), 1);
        assert_eq!(status_of(&report, CheckId::ClocksMemory), CheckStatus::Fail);
        assert_eq!(status_of(&report, CheckId::DriverKernel), CheckStatus::Pass);
    }

    #[test]
    fn checks_run_in_specification_order() {
        let fake = Arc::new(Fake::happy());
        let report = execute(&cluster(), &fake_probes(&fake));
        let ids: Vec<CheckId> = report.outcomes.iter().map(|outcome| outcome.id).collect();
        assert_eq!(ids, CHECK_ORDER.to_vec());
    }

    #[test]
    fn unimplemented_specification_checks_are_not_invented() {
        assert_eq!(CHECK_ORDER.len(), IMPLEMENTED_CHECK_COUNT);
        assert_eq!(IMPLEMENTED_CHECK_COUNT, 5);
        assert_eq!(SPECIFIED_CHECK_COUNT, 7);
        assert!(
            SPEC_DISCREPANCY.contains("enumerates five"),
            "{SPEC_DISCREPANCY}"
        );
    }

    #[test]
    fn human_report_states_the_discrepancy_and_the_verdict() {
        let report = execute_default(&Options::default());
        let text = report.to_human();
        assert!(text.contains("five"), "{text}");
        assert!(text.contains("specification 6.1"), "{text}");
        assert!(text.contains("INCOMPLETE"), "{text}");
        assert!(
            text.contains("5 of 7 specified checks implemented"),
            "{text}"
        );
    }

    #[test]
    fn json_report_carries_the_discrepancy_counts_and_remediation() {
        let mut fake = Fake::happy();
        fake.allreduce = Ok(25.0);
        let report = execute(&cluster(), &fake_probes(&Arc::new(fake)));
        let value = report.to_json();
        assert_eq!(value["specified_checks"], 7);
        assert_eq!(value["implemented_checks"], 5);
        assert_eq!(value["verdict"], "not_ready");
        assert_eq!(value["exit_code"], 1);
        assert_eq!(value["counts"]["failed"], 1);
        let checks = value["checks"].as_array().expect("checks array");
        assert_eq!(checks.len(), 5);
        let rdma = checks
            .iter()
            .find(|check| check["id"] == "rdma-bandwidth")
            .expect("rdma check");
        assert!(
            rdma["remediation"]
                .as_str()
                .expect("remediation")
                .contains("22")
        );
    }

    #[test]
    fn check_ids_and_statuses_are_stable_strings() {
        assert_eq!(CheckId::DriverKernel.as_str(), "driver-kernel");
        assert_eq!(CheckId::LinkMtu.as_str(), "link-mtu");
        assert_eq!(CheckId::RdmaBandwidth.as_str(), "rdma-bandwidth");
        assert_eq!(CheckId::ClocksMemory.as_str(), "clocks-memory");
        assert_eq!(CheckId::EngramShards.as_str(), "engram-shards");
        assert_eq!(CheckStatus::NotApplicable.as_str(), "not-applicable");
        assert_eq!(Verdict::NotReady.as_str(), "not_ready");
        assert_eq!(CheckStatus::Pass.marker(), '+');
        assert_eq!(CheckStatus::Fail.marker(), 'x');
        assert_eq!(CheckStatus::NotApplicable.marker(), '-');
    }

    #[test]
    fn rails_match_the_specification() {
        assert_eq!(RAILS[0].netdev, "enp1s0f0np0");
        assert_eq!(RAILS[0].rdma, "rocep1s0f0");
        assert_eq!(RAILS[1].netdev, "enP2p1s0f0np0");
        assert_eq!(RAILS[1].rdma, "roceP2p1s0f0");
        // 8972 bytes of ICMP payload plus 28 bytes of IP and ICMP header is exactly MTU 9000.
        assert_eq!(DF_PROBE_PAYLOAD_BYTES + 28, LINK_MTU as usize);
    }

    // ---- manifest parsing and options ------------------------------------------------------

    #[test]
    fn manifest_reader_accepts_the_m2_shape_and_is_tolerant_of_extras() {
        let value = json!({
            "file_alignment": 512,
            "shards": [
                {
                    "layer": 1,
                    "rank": 0,
                    "path": "/var/lib/sparkflash/engram/engram-l1-r0of4.bin",
                    "rows": 96_001_542,
                    "data_bytes": 25_344_407_088_u64,
                    "padding": 464,
                    "file_bytes": SHARD_L1,
                    "sha256": DIGEST,
                    "future_field": true
                }
            ]
        });
        let manifest = parse_manifest(&value);
        assert_eq!(manifest.entries.len(), 1);
        let entry = manifest.entry("engram-l1-r0of4.bin").expect("entry");
        assert_eq!(entry.file_bytes, Some(SHARD_L1));
        assert_eq!(entry.sha256.as_deref(), Some(DIGEST));
        assert!(manifest.entry("engram-l9-r0of4.bin").is_none());
    }

    #[test]
    fn manifest_reader_accepts_a_filename_keyed_object() {
        let value = json!({ "engram-l14-r0of4.bin": { "sha256": DIGEST, "bytes": SHARD_L14 } });
        let manifest = parse_manifest(&value);
        let entry = manifest.entry("engram-l14-r0of4.bin").expect("entry");
        assert_eq!(entry.file_bytes, Some(SHARD_L14));
    }

    #[test]
    fn manifest_reader_matches_on_the_basename_only() {
        let value = json!({ "shards": [ { "path": "/somewhere/else/engram-l1-r0of4.bin", "sha256": DIGEST } ] });
        let manifest = parse_manifest(&value);
        assert!(manifest.entry("engram-l1-r0of4.bin").is_some());
    }

    #[test]
    fn world_size_and_rank_are_validated() {
        let mut args = DoctorArgs::try_parse_from(["sparkflash doctor"]).expect("parse");
        args.world_size = 0;
        assert!(args.into_options().is_err());

        let mut args = DoctorArgs::try_parse_from(["sparkflash doctor"]).expect("parse");
        args.rank = 4;
        assert!(args.into_options().is_err());

        let args = DoctorArgs::try_parse_from([
            "sparkflash doctor",
            "--nodes",
            "192.0.2.80,192.0.2.84",
            "--rank",
            "1",
            "--apply",
        ])
        .expect("parse");
        let options = args.into_options().expect("valid");
        assert_eq!(options.nodes.len(), 2);
        assert!(options.apply && options.cluster() && options.live());
        assert_eq!(options.rank, 1);
    }

    #[test]
    fn default_options_probe_nothing() {
        let options = Options::default();
        assert!(!options.live());
        assert!(!options.cluster());
        assert_eq!(options.engram_dir, PathBuf::from(DEFAULT_ENGRAM_DIR));
        assert_eq!(options.world_size, 4);
        assert!(!options.apply);
    }

    #[test]
    fn peer_is_self_recognises_the_local_rank_and_localhost() {
        let options = Options {
            nodes: vec![
                "self".to_owned(),
                "localhost".to_owned(),
                "192.0.2.84".to_owned(),
            ],
            rank: 2,
            ..Options::default()
        };
        assert!(peer_is_self("self", &options));
        assert!(peer_is_self("localhost", &options));
        assert!(peer_is_self("2", &options));
        assert!(!peer_is_self("192.0.2.84", &options));
    }

    #[test]
    fn outcome_summaries_are_marked_by_status() {
        let fake = Arc::new(Fake::happy());
        let healthy = execute(&cluster(), &fake_probes(&fake));
        let passing = outcome_of(&healthy, CheckId::DriverKernel);
        assert!(
            passing.summary.ends_with(": satisfied"),
            "{}",
            passing.summary
        );
        let mut broken = Fake::happy();
        broken.mem_gib = Ok(1);
        let report = execute(&cluster(), &fake_probes(&Arc::new(broken)));
        let failed = outcome_of(&report, CheckId::ClocksMemory);
        assert!(
            failed.summary.starts_with("NOT SATISFIED:"),
            "{}",
            failed.summary
        );
        assert!(failed.remediation.is_some());
    }

    // ---- the specification names the wrong function --------------------------------------

    fn down_spec_rail_with_active_sibling() -> Vec<RailStatus> {
        vec![
            RailStatus {
                rdma: "rocep1s0f0".to_owned(),
                netdev: Some("enp1s0f0np0".to_owned()),
                state: "DOWN".to_owned(),
                speed_mbps: Some(40_000),
                pci: Some("0000:01:00".to_owned()),
            },
            RailStatus {
                rdma: "rocep1s0f1".to_owned(),
                netdev: Some("enp1s0f1np1".to_owned()),
                state: "ACTIVE".to_owned(),
                speed_mbps: Some(LINK_SPEED_MBPS),
                pci: Some("0000:01:00".to_owned()),
            },
            RailStatus {
                rdma: "roceP2p1s0f0".to_owned(),
                netdev: Some("enP2p1s0f0np0".to_owned()),
                state: "ACTIVE".to_owned(),
                speed_mbps: Some(LINK_SPEED_MBPS),
                pci: Some("0000:02:00".to_owned()),
            },
        ]
    }

    #[test]
    fn spec_rail_down_with_an_active_sibling_is_reported_as_a_wrong_specification_name() {
        let mut fake = Fake::happy();
        fake.inventory = Ok(down_spec_rail_with_active_sibling());
        // A down link is also unreadable to ethtool, which is the second, independent signal.
        fake.speed = Err(ProbeError::Unavailable(
            "enp1s0f0np0: Speed: Unknown!".to_owned(),
        ));
        let outcome = check_link_mtu(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        let details = outcome.details.join("\n");
        assert!(details.contains("rocep1s0f0 is DOWN"), "{details}");
        assert!(details.contains("sibling function rocep1s0f1"), "{details}");
        assert!(
            details.contains("wrong name in the specification"),
            "{details}"
        );
        assert!(details.contains("not a disconnected cable"), "{details}");
        // Two rails really are active, so the dual-rail gate passes: only the specification's
        // chosen names are wrong.
        assert!(
            details.contains("2 rails are ACTIVE at >= 200000 Mb/s"),
            "{details}"
        );
        // The specification's names are still reported, as what the specification asks for.
        assert!(
            details.contains("enP2p1s0f0np0 (roceP2p1s0f0) is ACTIVE"),
            "{details}"
        );
    }

    #[test]
    fn active_rails_with_mtu_1500_fail_with_the_exact_remediation() {
        let mut fake = Fake::happy();
        fake.mtu = Ok(1500);
        let outcome = check_link_mtu(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        let remediation = outcome.remediation.expect("remediation");
        assert!(
            remediation.contains("set MTU 9000 on enp1s0f0np0"),
            "{remediation}"
        );
        assert!(
            remediation.contains("set MTU 9000 on enP2p1s0f0np0"),
            "{remediation}"
        );
        let details = outcome.details.join("\n");
        assert!(details.contains("MTU 1500, expected 9000"), "{details}");
    }

    #[test]
    fn an_active_sibling_rail_is_mtu_checked_even_when_the_specification_does_not_name_it() {
        let mut fake = Fake::happy();
        fake.inventory = Ok(down_spec_rail_with_active_sibling());
        let outcome = check_link_mtu(&fake_probes(&Arc::new(fake)), &cluster());
        // The sibling carries the traffic, so its MTU has to be 9000 too.
        let details = outcome.details.join("\n");
        assert!(
            details.contains("enp1s0f1np1 (active rocep1s0f1): MTU 9000"),
            "{details}"
        );
    }

    #[test]
    fn a_single_active_rail_fails_the_dual_rail_gate() {
        let mut fake = Fake::happy();
        fake.inventory = Ok(vec![RailStatus {
            rdma: "rocep1s0f1".to_owned(),
            netdev: Some("enp1s0f1np1".to_owned()),
            state: "ACTIVE".to_owned(),
            speed_mbps: Some(LINK_SPEED_MBPS),
            pci: Some("0000:01:00".to_owned()),
        }]);
        let outcome = check_link_mtu(&fake_probes(&Arc::new(fake)), &cluster());
        assert_eq!(outcome.status, CheckStatus::Fail);
        let details = outcome.details.join("\n");
        assert!(
            details.contains("only 1 of the 2 required rails are ACTIVE"),
            "{details}"
        );
        assert!(details.contains("is not present on this node"), "{details}");
    }

    #[test]
    fn the_inventory_lists_every_rdma_device_in_the_report() {
        let fake = Arc::new(Fake::happy());
        let outcome = check_link_mtu(&fake_probes(&fake), &cluster());
        let details = outcome.details.join("\n");
        assert!(details.contains("RDMA devices on this node:"), "{details}");
        assert!(details.contains("rocep1s0f0"), "{details}");
        assert!(details.contains("roceP2p1s0f0"), "{details}");
    }

    // ---- nvidia-uvm: a requirement with no knob ------------------------------------------

    #[test]
    fn uvm_unknown_reports_the_specification_gap_and_the_hand_checks() {
        let mut fake = Fake::happy();
        fake.uvm = Ok(UvmPacking {
            state: UvmPagePacking::Unknown,
            parameters_seen: Some(56),
            candidates: Vec::new(),
        });
        let outcome = check_driver_kernel(&fake_probes(&Arc::new(fake)));
        assert_eq!(outcome.status, CheckStatus::Fail);
        let details = outcome.details.join("\n");
        assert!(details.contains("56 module parameters listed"), "{details}");
        assert!(details.contains("no 64K-packing knob found"), "{details}");
        let remediation = outcome.remediation.expect("remediation");
        assert!(remediation.contains("specification gap"), "{remediation}");
        assert!(
            remediation.contains("nvidia-smi --query-gpu=driver_version"),
            "{remediation}"
        );
        assert!(
            remediation.contains("/proc/driver/nvidia/params"),
            "{remediation}"
        );
        assert!(remediation.contains("56"), "{remediation}");
    }

    #[test]
    fn uvm_evidence_reports_how_many_parameters_were_listed() {
        let mut fake = Fake::happy();
        fake.uvm = Ok(UvmPacking {
            state: UvmPagePacking::Enabled,
            parameters_seen: Some(56),
            candidates: vec!["uvm_64k_packing".to_owned()],
        });
        let outcome = check_driver_kernel(&fake_probes(&Arc::new(fake)));
        assert_eq!(outcome.status, CheckStatus::Pass);
        assert!(
            outcome
                .details
                .iter()
                .any(|detail| detail.contains("candidates: uvm_64k_packing")),
            "{:?}",
            outcome.details
        );
    }

    // ---- sysfs parsing helpers ------------------------------------------------------------

    #[test]
    fn parse_ib_rate_reads_the_sysfs_rate() {
        assert_eq!(parse_ib_rate("200 Gb/sec (4X HDR)"), Some(200_000));
        assert_eq!(parse_ib_rate("40 Gb/sec (4X QDR)"), Some(40_000));
        assert_eq!(parse_ib_rate("2.5 Gb/sec"), Some(2_500));
        assert_eq!(parse_ib_rate("N/A"), None);
        assert_eq!(parse_ib_rate(""), None);
    }

    #[test]
    fn strip_pci_function_groups_sibling_functions() {
        assert_eq!(strip_pci_function("0000:01:00.0"), "0000:01:00");
        assert_eq!(strip_pci_function("0000:01:00.1"), "0000:01:00");
        assert_eq!(strip_pci_function("0000:01:00"), "0000:01:00");
    }

    #[test]
    fn fast_sibling_requires_the_same_card_and_a_fast_active_port() {
        let spec = RailStatus {
            rdma: "rocep1s0f0".to_owned(),
            netdev: Some("enp1s0f0np0".to_owned()),
            state: "DOWN".to_owned(),
            speed_mbps: Some(40_000),
            pci: Some("0000:01:00".to_owned()),
        };
        let sibling = RailStatus {
            rdma: "rocep1s0f1".to_owned(),
            netdev: Some("enp1s0f1np1".to_owned()),
            state: "ACTIVE".to_owned(),
            speed_mbps: Some(LINK_SPEED_MBPS),
            pci: Some("0000:01:00".to_owned()),
        };
        let other_card = RailStatus {
            rdma: "rocep2s0f1".to_owned(),
            netdev: Some("enp2s0f1np1".to_owned()),
            state: "ACTIVE".to_owned(),
            speed_mbps: Some(LINK_SPEED_MBPS),
            pci: Some("0000:02:00".to_owned()),
        };
        let slow_sibling = RailStatus {
            rdma: "rocep1s0f2".to_owned(),
            netdev: None,
            state: "ACTIVE".to_owned(),
            speed_mbps: Some(40_000),
            pci: Some("0000:01:00".to_owned()),
        };
        let inventory = vec![spec.clone(), sibling.clone(), other_card, slow_sibling];
        assert_eq!(
            fast_sibling(&inventory, &spec).map(|found| found.rdma.as_str()),
            Some("rocep1s0f1")
        );
        // Without a PCI address there is nothing to group by, so no sibling can be claimed.
        let unknown = RailStatus {
            pci: None,
            ..sibling
        };
        assert!(fast_sibling(&inventory, &unknown).is_none());
    }
}
