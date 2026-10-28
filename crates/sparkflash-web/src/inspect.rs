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

//! What each node has: its fast network, the software, the model and its converted weights.
//!
//! One read-only shell script ([`INSPECT_SCRIPT`]) runs on every node, over SSH or locally for
//! the lead, and prints one fact per line. [`parse_facts`] reads those lines and [`analyze`] turns
//! the four nodes' facts into the report the guided setup page draws: the NCCL settings that match
//! the cabling, what is missing on which node, and a command for every problem it can fix.
//! Nothing here changes a node. A fix is text the operator chooses to run.
//!
//! The fabric rules are the ones `docs/serving.md` describes. Each cabled ConnectX-7 port shows
//! up as two RoCE devices (one per PCIe half), and NCCL uses both when `NCCL_IB_HCA` lists both.
//! The collectives address each rank by the IPv4 address in its RoCE v2 GID, and two rails on one
//! subnet need `arp_ignore=1` / `arp_announce=2` so each address answers on its own port.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::path::Path;

use serde::Serialize;

use crate::config::{SparkConfig, WORLD_SIZE, shell_quote};

/// The read-only probe. Positional arguments: the binary, the cubin directory, the NCCL library
/// directory (may be empty), the checkpoint directory (may be empty), this rank's converted file,
/// the conversion manifest, and the hub cache directory (may be empty).
pub const INSPECT_SCRIPT: &str = r#"export LC_ALL=C
binary=$1 cubins=$2 nccl=$3 checkpoint=$4 rankfile=$5 manifest=$6 cache=$7
echo "host $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "gpu $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
fi
for dev in /sys/class/infiniband/*; do
  [ -d "$dev/ports/1" ] || continue
  name=${dev##*/}
  state=$(cat "$dev/ports/1/state" 2>/dev/null)
  rate=$(cat "$dev/ports/1/rate" 2>/dev/null)
  echo "port $name ${state#*: } ${rate%% *}"
  for i in 0 1 2 3 4 5 6 7; do
    kind=$(cat "$dev/ports/1/gid_attrs/types/$i" 2>/dev/null) || continue
    [ -n "$kind" ] || continue
    gid=$(cat "$dev/ports/1/gids/$i" 2>/dev/null)
    ndev=$(cat "$dev/ports/1/gid_attrs/ndevs/$i" 2>/dev/null)
    echo "gid $name $i ${kind// /_} $gid ${ndev:--}"
  done
done
echo "arp all $(cat /proc/sys/net/ipv4/conf/all/arp_ignore 2>/dev/null || echo 0) $(cat /proc/sys/net/ipv4/conf/all/arp_announce 2>/dev/null || echo 0)"
for net in /sys/class/net/*; do
  name=${net##*/}
  [ -e "$net/device" ] || continue
  echo "link $name $(cat "$net/mtu") $(cat "$net/operstate") $(cat "/proc/sys/net/ipv4/conf/$name/arp_ignore" 2>/dev/null || echo 0) $(cat "/proc/sys/net/ipv4/conf/$name/arp_announce" 2>/dev/null || echo 0)"
done
ip -4 -o addr show 2>/dev/null | awk '{d = "static"; for (i = 1; i <= NF; i++) if ($i == "dynamic") d = "dynamic"; print "addr", $2, $4, d}'
ip -4 route show 2>/dev/null | sed 's/^/route /'
filefact() {
  if [ -f "$2" ]; then
    sum=-
    [ "$3" = hash ] && sum=$(sha256sum < "$2" | cut -c1-64)
    echo "file $1 $(stat -Lc '%s %Y' "$2") $sum"
  else
    echo "file $1 - - -"
  fi
}
treefact() {
  if [ -d "$2" ]; then
    sum=-
    [ "$3" = hash ] && sum=$(cd "$2" && find -L . -type f -print0 | sort -z | xargs -0r sha256sum | sha256sum | cut -c1-64)
    echo "tree $1 $(find -L "$2" -type f -printf '%s\n' | awk '{n += 1; b += $1} END {printf "%.0f %.0f", n, b}') $sum"
  else
    echo "tree $1 - - -"
  fi
}
freefact() {
  p=$2
  while [ -n "$p" ] && [ ! -e "$p" ]; do p=$(dirname "$p"); done
  [ -n "$p" ] && echo "free $1 $(df -PB1 "$p" | awk 'NR == 2 {print $4}')"
}
filefact binary "$binary" hash
treefact cubins "$cubins" hash
[ -n "$nccl" ] && filefact nccl "$nccl/libnccl.so.2" hash
[ -n "$checkpoint" ] && treefact checkpoint "$checkpoint"
filefact converted "$rankfile"
filefact manifest "$manifest"
[ -n "$cache" ] && freefact cache "$cache"
freefact converted "$(dirname "$rankfile")"
exit 0
"#;

/// The netplan file a network fix writes.
pub const NETPLAN_FILE: &str = "/etc/netplan/60-sparkflash-fabric.yaml";

/// The sysctl file a network fix writes.
pub const SYSCTL_FILE: &str = "/etc/sysctl.d/60-sparkflash-fabric.conf";

/// The MTU the fast ports should run.
pub const FABRIC_MTU: u32 = 9000;

/// The subnet suggested when no node has a fast-network address yet.
const DEFAULT_SUBNET: (Ipv4Addr, u8) = (Ipv4Addr::new(10, 200, 0, 0), 24);

/// One RDMA port.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RdmaPort {
    /// The RDMA device, e.g. `rocep1s0f1`.
    pub device: String,
    /// Whether the port state is ACTIVE (cabled and up).
    pub active: bool,
    /// The link rate in Gb/s, when the kernel reports one.
    pub rate_gbps: Option<u32>,
}

/// One GID table entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GidEntry {
    /// The RDMA device.
    pub device: String,
    /// The GID index.
    pub index: u16,
    /// Whether it is a RoCE v2 entry.
    pub roce_v2: bool,
    /// The IPv4 address an IPv4-mapped GID carries.
    pub ipv4: Option<Ipv4Addr>,
    /// The netdev the entry belongs to.
    pub netdev: Option<String>,
}

/// One physical network interface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NetLink {
    /// The netdev name.
    pub name: String,
    /// Its MTU.
    pub mtu: u32,
    /// Whether the operational state is `up`.
    pub up: bool,
    /// Its own `arp_ignore`.
    pub arp_ignore: u8,
    /// Its own `arp_announce`.
    pub arp_announce: u8,
}

/// One IPv4 address.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Address {
    /// The netdev carrying it.
    pub netdev: String,
    /// The address.
    pub ip: Ipv4Addr,
    /// The prefix length.
    pub prefix: u8,
    /// Whether DHCP assigned it.
    pub dynamic: bool,
}

/// One IPv4 route.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Route {
    /// The destination, e.g. `198.51.100.0/24` or `default`.
    pub destination: String,
    /// The netdev it leaves by.
    pub netdev: String,
    /// The metric; the kernel prefers the lowest.
    pub metric: u32,
    /// Whether DHCP installed it.
    pub dhcp: bool,
}

/// A file's size, modification time and (for small files) content hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FileFact {
    /// Bytes.
    pub size: u64,
    /// Unix seconds.
    pub mtime: u64,
    /// SHA-256, when the probe hashed it.
    pub sha256: Option<String>,
}

/// A directory tree's file count, total size and (for small trees) a hash of its contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TreeFact {
    /// Regular files, following symlinks.
    pub files: u64,
    /// Their total size.
    pub bytes: u64,
    /// A hash over every file's SHA-256, when the probe computed one.
    pub sha256: Option<String>,
}

/// Everything one probe reported.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct NodeFacts {
    /// The node's own hostname.
    pub hostname: String,
    /// The GPU name, when `nvidia-smi` answered.
    pub gpu: Option<String>,
    /// RDMA ports.
    pub ports: Vec<RdmaPort>,
    /// GID entries.
    pub gids: Vec<GidEntry>,
    /// Physical netdevs.
    pub links: Vec<NetLink>,
    /// IPv4 addresses.
    pub addresses: Vec<Address>,
    /// IPv4 routes.
    pub routes: Vec<Route>,
    /// `net.ipv4.conf.all.arp_ignore` and `arp_announce`.
    pub arp_all: (u8, u8),
    /// Files by label (`binary`, `nccl`, `converted`, `manifest`); `None` when absent.
    pub files: BTreeMap<String, Option<FileFact>>,
    /// Trees by label (`cubins`, `checkpoint`); `None` when absent.
    pub trees: BTreeMap<String, Option<TreeFact>>,
    /// Free bytes by label (`cache`, `converted`).
    pub free: BTreeMap<String, u64>,
}

/// Reads the probe's output. Unknown or malformed lines are skipped, so an older or newer probe
/// still yields what it can.
#[must_use]
pub fn parse_facts(text: &str) -> NodeFacts {
    let mut facts = NodeFacts::default();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let Some((&kind, rest)) = fields.split_first() else {
            continue;
        };
        match (kind, rest) {
            ("host", [name, ..]) => facts.hostname = (*name).to_owned(),
            ("gpu", words) if !words.is_empty() => facts.gpu = Some(words.join(" ")),
            ("port", [device, state, rest @ ..]) => facts.ports.push(RdmaPort {
                device: (*device).to_owned(),
                active: *state == "ACTIVE",
                rate_gbps: rest.first().and_then(|rate| rate.parse().ok()),
            }),
            ("gid", [device, index, kind, gid, netdev]) => {
                let Ok(index) = index.parse() else { continue };
                facts.gids.push(GidEntry {
                    device: (*device).to_owned(),
                    index,
                    roce_v2: *kind == "RoCE_v2",
                    ipv4: gid
                        .parse::<Ipv6Addr>()
                        .ok()
                        .and_then(|gid| gid.to_ipv4_mapped()),
                    netdev: (*netdev != "-").then(|| (*netdev).to_owned()),
                });
            }
            ("arp", ["all", ignore, announce]) => {
                facts.arp_all = (ignore.parse().unwrap_or(0), announce.parse().unwrap_or(0));
            }
            ("link", [name, mtu, state, ignore, announce]) => facts.links.push(NetLink {
                name: (*name).to_owned(),
                mtu: mtu.parse().unwrap_or(0),
                up: *state == "up",
                arp_ignore: ignore.parse().unwrap_or(0),
                arp_announce: announce.parse().unwrap_or(0),
            }),
            ("addr", [netdev, cidr, origin]) => {
                let Some((ip, prefix)) = cidr.split_once('/') else {
                    continue;
                };
                let (Ok(ip), Ok(prefix)) = (ip.parse(), prefix.parse()) else {
                    continue;
                };
                facts.addresses.push(Address {
                    netdev: (*netdev).to_owned(),
                    ip,
                    prefix,
                    dynamic: *origin == "dynamic",
                });
            }
            ("route", [destination, rest @ ..]) => {
                let after = |key: &str| {
                    rest.iter()
                        .position(|word| *word == key)
                        .and_then(|at| rest.get(at + 1))
                };
                let Some(netdev) = after("dev") else { continue };
                facts.routes.push(Route {
                    destination: (*destination).to_owned(),
                    netdev: (*netdev).to_owned(),
                    metric: after("metric")
                        .and_then(|metric| metric.parse().ok())
                        .unwrap_or(0),
                    dhcp: after("proto").is_some_and(|proto| *proto == "dhcp"),
                });
            }
            ("file", [label, size, mtime, sum]) => {
                let fact = size.parse().ok().map(|size| FileFact {
                    size,
                    mtime: mtime.parse().unwrap_or(0),
                    sha256: hash_field(sum),
                });
                facts.files.insert((*label).to_owned(), fact);
            }
            ("tree", [label, files, bytes, sum]) => {
                let fact = files.parse().ok().map(|files| TreeFact {
                    files,
                    bytes: bytes.parse().unwrap_or(0),
                    sha256: hash_field(sum),
                });
                facts.trees.insert((*label).to_owned(), fact);
            }
            ("free", [label, bytes]) => {
                if let Ok(bytes) = bytes.parse() {
                    facts.free.insert((*label).to_owned(), bytes);
                }
            }
            _ => {}
        }
    }
    facts
}

fn hash_field(value: &str) -> Option<String> {
    (value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| value.to_owned())
}

/// One usable fast-network lane: an active RoCE device whose RoCE v2 GID carries an IPv4 address.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Rail {
    /// The RDMA device, an `NCCL_IB_HCA` entry.
    pub device: String,
    /// Its netdev, an `NCCL_SOCKET_IFNAME` candidate.
    pub netdev: String,
    /// The RoCE v2 IPv4 GID index (`NCCL_IB_GID_INDEX`).
    pub gid_index: u16,
    /// The address the GID carries.
    pub ip: Ipv4Addr,
    /// Its prefix length, when the address is on the netdev.
    pub prefix: Option<u8>,
    /// The netdev's MTU.
    pub mtu: Option<u32>,
    /// The link rate in Gb/s.
    pub rate_gbps: Option<u32>,
}

impl NodeFacts {
    /// The usable lanes, in a stable order (device names compared case-insensitively).
    #[must_use]
    pub fn rails(&self) -> Vec<Rail> {
        let mut rails: Vec<Rail> = self
            .ports
            .iter()
            .filter(|port| port.active)
            .filter_map(|port| {
                let gid = self
                    .gids
                    .iter()
                    .filter(|gid| gid.device == port.device && gid.roce_v2 && gid.ipv4.is_some())
                    .min_by_key(|gid| gid.index)?;
                let netdev = gid.netdev.clone()?;
                let ip = gid.ipv4?;
                Some(Rail {
                    prefix: self
                        .addresses
                        .iter()
                        .find(|address| address.netdev == netdev && address.ip == ip)
                        .map(|address| address.prefix),
                    mtu: self.link(&netdev).map(|link| link.mtu),
                    rate_gbps: port.rate_gbps,
                    device: port.device.clone(),
                    netdev,
                    gid_index: gid.index,
                    ip,
                })
            })
            .collect();
        rails.sort_by_key(|rail| rail.device.to_ascii_lowercase());
        rails
    }

    /// Netdevs of active RDMA ports that have no IPv4 address: cabled but not configured.
    #[must_use]
    pub fn unaddressed(&self) -> Vec<String> {
        let rails = self.rails();
        let mut names: Vec<String> = self
            .ports
            .iter()
            .filter(|port| port.active && !rails.iter().any(|rail| rail.device == port.device))
            .filter_map(|port| {
                self.gids
                    .iter()
                    .filter(|gid| gid.device == port.device)
                    .find_map(|gid| gid.netdev.clone())
            })
            .collect();
        names.sort_by_key(|name| name.to_ascii_lowercase());
        names.dedup();
        names
    }

    fn link(&self, name: &str) -> Option<&NetLink> {
        self.links.iter().find(|link| link.name == name)
    }

    /// The effective `arp_ignore` / `arp_announce` of a netdev: the larger of `all` and its own.
    fn arp(&self, netdev: &str) -> (u8, u8) {
        let own = self
            .link(netdev)
            .map_or((0, 0), |link| (link.arp_ignore, link.arp_announce));
        (own.0.max(self.arp_all.0), own.1.max(self.arp_all.1))
    }

    /// The rail the kernel prefers for the fast subnet: the lowest route metric, then the first.
    fn primary_rail<'a>(&self, rails: &'a [Rail]) -> Option<&'a Rail> {
        rails.iter().min_by_key(|rail| {
            self.routes
                .iter()
                .filter(|route| route.netdev == rail.netdev && route.destination != "default")
                .map(|route| route.metric)
                .min()
                .unwrap_or(u32::MAX)
        })
    }
}

/// The NCCL settings and peer addresses that match the cabling.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FabricPlan {
    /// Every rail's RDMA device, comma separated.
    pub nccl_ib_hca: String,
    /// The bootstrap netdev (comma separated when the nodes name it differently).
    pub nccl_socket_ifname: String,
    /// The shared GID index, `None` when the nodes disagree and NCCL should choose.
    pub nccl_ib_gid_index: Option<u16>,
    /// Each rank's collective address, in rank order.
    pub peers: Vec<String>,
}

impl FabricPlan {
    /// Whether `config` already carries this plan.
    #[must_use]
    pub fn matches(&self, config: &SparkConfig) -> bool {
        config.network.nccl_ib_hca == self.nccl_ib_hca
            && config.network.nccl_socket_ifname == self.nccl_socket_ifname
            && config.network.nccl_ib_gid_index == self.nccl_ib_gid_index
            && config
                .cluster
                .nodes
                .iter()
                .map(|node| node.peer.as_str())
                .eq(self.peers.iter().map(String::as_str))
    }

    /// Writes the plan into `config`.
    pub fn apply(&self, config: &mut SparkConfig) {
        config.network.nccl_ib_hca.clone_from(&self.nccl_ib_hca);
        config
            .network
            .nccl_socket_ifname
            .clone_from(&self.nccl_socket_ifname);
        config.network.nccl_ib_gid_index = self.nccl_ib_gid_index;
        for (node, peer) in config.cluster.nodes.iter_mut().zip(&self.peers) {
            node.peer.clone_from(peer);
        }
    }
}

/// How serious a finding is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// The cluster cannot start until it is fixed.
    Error,
    /// The cluster runs, but slower or less reliably.
    Warning,
}

/// One problem, in plain language.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    /// The rank it is about, `None` for the whole cluster.
    pub rank: Option<usize>,
    /// Whether it blocks a start.
    pub severity: Severity,
    /// What is wrong and what to do.
    pub message: String,
}

/// Whether something is where it should be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Check {
    /// Present and matching the lead.
    Ready,
    /// Not there.
    Missing,
    /// There, but different from the lead's copy.
    Outdated,
    /// Started but not finished.
    Partial,
    /// Could not be checked (the node did not answer, or there is nothing to compare with).
    Unknown,
}

/// One node's line in the setup report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NodeCheck {
    /// The rank.
    pub rank: usize,
    /// The configured SSH name.
    pub ssh: String,
    /// Whether the probe ran.
    pub reachable: bool,
    /// Why it did not.
    pub error: Option<String>,
    /// The node's own hostname.
    pub hostname: Option<String>,
    /// Its GPU.
    pub gpu: Option<String>,
    /// Its usable fast-network lanes.
    pub rails: Vec<Rail>,
    /// The server binary, kernels and NCCL library.
    pub software: Check,
    /// The checkpoint.
    pub model: Check,
    /// Checkpoint files present.
    pub model_files: u64,
    /// Checkpoint bytes present.
    pub model_bytes: u64,
    /// This rank's converted weights.
    pub converted: Check,
    /// Free bytes where the model is kept.
    pub free_bytes: Option<u64>,
    /// A command that configures this node's fast ports, when they need it.
    pub network_fix: Option<String>,
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SetupReport {
    /// When the probes ran, Unix milliseconds.
    pub checked_ms: u64,
    /// One entry per rank.
    pub nodes: Vec<NodeCheck>,
    /// The NCCL settings that match the cabling, when every node has a usable lane.
    pub plan: Option<FabricPlan>,
    /// Problems, errors first.
    pub findings: Vec<Finding>,
    /// The checkpoint files and bytes on the lead, which the other nodes are compared with.
    pub model_files: u64,
    /// See `model_files`.
    pub model_bytes: u64,
}

impl SetupReport {
    /// Whether any finding blocks a start.
    #[must_use]
    pub fn network_ok(&self) -> bool {
        self.plan.is_some()
            && !self
                .findings
                .iter()
                .any(|finding| finding.severity == Severity::Error)
    }
}

/// One node's probe result: its facts, or why the probe failed.
pub type ProbeResult = Result<NodeFacts, String>;

/// Builds the report from the four probes, in rank order.
#[must_use]
pub fn analyze(
    config: &SparkConfig,
    probes: &[ProbeResult],
    model_complete_on_lead: bool,
    checked_ms: u64,
) -> SetupReport {
    let mut findings = Vec::new();
    let name = |rank: usize| -> String {
        config
            .cluster
            .nodes
            .get(rank)
            .map_or_else(|| format!("rank {rank}"), |node| node.ssh.clone())
    };
    let lead = probes.first().and_then(|probe| probe.as_ref().ok());

    // Fabric: every node needs a lane, and the lanes must agree on what NCCL is told.
    let mut rails_by_rank: Vec<Vec<Rail>> = Vec::new();
    for (rank, probe) in probes.iter().enumerate() {
        let rails = match probe {
            Ok(facts) => facts.rails(),
            Err(error) => {
                findings.push(Finding {
                    rank: Some(rank),
                    severity: Severity::Error,
                    message: format!("Can't reach {}: {error}", name(rank)),
                });
                Vec::new()
            }
        };
        if let Ok(facts) = probe {
            if rails.is_empty() {
                let message = if facts.unaddressed().is_empty() {
                    format!(
                        "{} has no fast-network link. Check that its QSFP cable is plugged in at \
                         both ends.",
                        name(rank)
                    )
                } else {
                    format!(
                        "{}'s fast-network port ({}) is cabled but has no IP address yet. Run the \
                         network command shown for it.",
                        name(rank),
                        facts.unaddressed().join(", ")
                    )
                };
                findings.push(Finding {
                    rank: Some(rank),
                    severity: Severity::Error,
                    message,
                });
            }
            for rail in &rails {
                if rail.mtu.is_some_and(|mtu| mtu < FABRIC_MTU) {
                    findings.push(Finding {
                        rank: Some(rank),
                        severity: Severity::Warning,
                        message: format!(
                            "{}'s port {} uses MTU {}; {FABRIC_MTU} is faster. Run the network \
                             command shown for it.",
                            name(rank),
                            rail.netdev,
                            rail.mtu.unwrap_or_default()
                        ),
                    });
                }
                if rail.rate_gbps.is_some_and(|rate| rate < 200) {
                    findings.push(Finding {
                        rank: Some(rank),
                        severity: Severity::Warning,
                        message: format!(
                            "{}'s port {} links at {} Gb/s, not 200. Check the cable and switch \
                             port.",
                            name(rank),
                            rail.device,
                            rail.rate_gbps.unwrap_or_default()
                        ),
                    });
                }
            }
            let netdevs: Vec<&str> = rails.iter().map(|rail| rail.netdev.as_str()).collect();
            if shares_subnet(&rails) && !arp_ok(facts, &netdevs) {
                findings.push(Finding {
                    rank: Some(rank),
                    severity: Severity::Warning,
                    message: format!(
                        "{} has two fast lanes on one subnet without arp_ignore=1 and \
                         arp_announce=2, so replies can leave by the wrong lane. Run the network \
                         command shown for it.",
                        name(rank)
                    ),
                });
            }
        }
        rails_by_rank.push(rails);
    }
    let lane_counts: BTreeSet<usize> = rails_by_rank
        .iter()
        .filter(|rails| !rails.is_empty())
        .map(Vec::len)
        .collect();
    if lane_counts.len() > 1 {
        for (rank, rails) in rails_by_rank.iter().enumerate() {
            if rails.len() == 1 && lane_counts.iter().any(|count| *count > 1) {
                findings.push(Finding {
                    rank: Some(rank),
                    severity: Severity::Warning,
                    message: format!(
                        "Only one fast lane is up on {}; the others have {}. It works, at lower \
                         speed.",
                        name(rank),
                        lane_counts.iter().max().copied().unwrap_or(2)
                    ),
                });
            }
        }
    }

    let plan = plan_fabric(probes, &rails_by_rank, &mut findings, &name);

    // Software, model and converted weights, each node against the lead.
    let lead_file = |label: &str| lead.and_then(|facts| facts.files.get(label).cloned().flatten());
    let lead_tree = |label: &str| lead.and_then(|facts| facts.trees.get(label).cloned().flatten());
    let lead_model = lead_tree("checkpoint");
    let wants_nccl = config.nccl_lib_path().is_some();
    let mut nodes = Vec::with_capacity(probes.len());
    for (rank, probe) in probes.iter().enumerate() {
        let ssh = name(rank);
        let Ok(facts) = probe else {
            nodes.push(NodeCheck {
                rank,
                ssh,
                reachable: false,
                error: probe.as_ref().err().cloned(),
                hostname: None,
                gpu: None,
                rails: Vec::new(),
                software: Check::Unknown,
                model: Check::Unknown,
                model_files: 0,
                model_bytes: 0,
                converted: Check::Unknown,
                free_bytes: None,
                network_fix: None,
            });
            continue;
        };
        let file = |label: &str| facts.files.get(label).cloned().flatten();
        let tree = |label: &str| facts.trees.get(label).cloned().flatten();
        let software = if rank == 0 {
            let present = file("binary").is_some()
                && tree("cubins").is_some_and(|tree| tree.files > 0)
                && (!wants_nccl || file("nccl").is_some());
            if present {
                Check::Ready
            } else {
                Check::Missing
            }
        } else {
            let same_file = |label: &str| matches!((file(label), lead_file(label)), (Some(mine), Some(theirs)) if mine.sha256 == theirs.sha256);
            let same_tree = |label: &str| matches!((tree(label), lead_tree(label)), (Some(mine), Some(theirs)) if mine.sha256 == theirs.sha256 && mine.files > 0);
            if file("binary").is_none() {
                Check::Missing
            } else if same_file("binary")
                && same_tree("cubins")
                && (!wants_nccl || same_file("nccl"))
            {
                Check::Ready
            } else {
                Check::Outdated
            }
        };
        let mine = tree("checkpoint");
        let (model_files, model_bytes) = mine
            .as_ref()
            .map_or((0, 0), |tree| (tree.files, tree.bytes));
        let model = match (&mine, &lead_model) {
            (None, _) => Check::Missing,
            (Some(_), _) if rank == 0 => {
                if model_complete_on_lead {
                    Check::Ready
                } else {
                    Check::Partial
                }
            }
            (Some(_), None) => Check::Unknown,
            (Some(mine), Some(lead)) if mine.files == lead.files && mine.bytes == lead.bytes => {
                if model_complete_on_lead {
                    Check::Ready
                } else {
                    Check::Partial
                }
            }
            (Some(_), Some(_)) => Check::Partial,
        };
        let converted = match (file("converted"), file("manifest")) {
            (None, _) => Check::Missing,
            (Some(rank_file), Some(manifest)) if manifest.mtime >= rank_file.mtime => Check::Ready,
            (Some(_), _) => Check::Partial,
        };
        nodes.push(NodeCheck {
            rank,
            ssh,
            reachable: true,
            error: None,
            hostname: Some(facts.hostname.clone()).filter(|name| !name.is_empty()),
            gpu: facts.gpu.clone(),
            rails: rails_by_rank.get(rank).cloned().unwrap_or_default(),
            software,
            model,
            model_files,
            model_bytes,
            converted,
            free_bytes: facts.free.get("cache").copied(),
            network_fix: None,
        });
    }

    // A network command for every node whose ports need addresses, MTU or ARP settings.
    let fixes = network_fixes(probes, &name);
    for (node, fix) in nodes.iter_mut().zip(fixes) {
        node.network_fix = fix;
    }

    findings.sort_by_key(|finding| {
        (
            finding.severity == Severity::Warning,
            finding.rank.unwrap_or(usize::MAX),
        )
    });
    let (model_files, model_bytes) = lead_model.map_or((0, 0), |tree| (tree.files, tree.bytes));
    SetupReport {
        checked_ms,
        nodes,
        plan,
        findings,
        model_files,
        model_bytes,
    }
}

fn plan_fabric(
    probes: &[ProbeResult],
    rails_by_rank: &[Vec<Rail>],
    findings: &mut Vec<Finding>,
    name: &dyn Fn(usize) -> String,
) -> Option<FabricPlan> {
    if rails_by_rank.len() != WORLD_SIZE || rails_by_rank.iter().any(Vec::is_empty) {
        return None;
    }
    let mut primaries = Vec::new();
    for (probe, rails) in probes.iter().zip(rails_by_rank) {
        let facts = probe.as_ref().ok()?;
        primaries.push(facts.primary_rail(rails)?.clone());
    }
    let lead = &primaries[0];
    let mut blocked = false;
    for (rank, primary) in primaries.iter().enumerate().skip(1) {
        let same_subnet = match lead.prefix {
            Some(prefix) => network(primary.ip, prefix) == network(lead.ip, prefix),
            None => true,
        };
        if !same_subnet {
            blocked = true;
            findings.push(Finding {
                rank: Some(rank),
                severity: Severity::Error,
                message: format!(
                    "{}'s fast port has address {}, which is not on {}'s fast network ({}/{}). \
                     Give every Spark an address on one subnet.",
                    name(rank),
                    primary.ip,
                    name(0),
                    lead.ip,
                    lead.prefix.unwrap_or(32)
                ),
            });
        }
    }
    let peers: Vec<String> = primaries.iter().map(|rail| rail.ip.to_string()).collect();
    let unique: BTreeSet<&String> = peers.iter().collect();
    if unique.len() != peers.len() {
        blocked = true;
        findings.push(Finding {
            rank: None,
            severity: Severity::Error,
            message: format!(
                "Two Sparks have the same fast-network address ({}). Give each its own.",
                peers.join(", ")
            ),
        });
    }
    if blocked {
        return None;
    }

    let mut devices: Vec<String> = rails_by_rank
        .iter()
        .flatten()
        .map(|rail| rail.device.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    devices.sort_by_key(|device| device.to_ascii_lowercase());
    let mut socket_names: Vec<String> = Vec::new();
    for primary in &primaries {
        if !socket_names.contains(&primary.netdev) {
            socket_names.push(primary.netdev.clone());
        }
    }
    let gid_indices: BTreeSet<u16> = rails_by_rank
        .iter()
        .flatten()
        .map(|rail| rail.gid_index)
        .collect();
    let nccl_ib_gid_index = if gid_indices.len() == 1 {
        gid_indices.first().copied()
    } else {
        findings.push(Finding {
            rank: None,
            severity: Severity::Warning,
            message: "The fast ports keep their address at different GID slots, so NCCL picks the \
                      slot itself."
                .to_owned(),
        });
        None
    };
    Some(FabricPlan {
        nccl_ib_hca: devices.join(","),
        nccl_socket_ifname: socket_names.join(","),
        nccl_ib_gid_index,
        peers,
    })
}

fn network(ip: Ipv4Addr, prefix: u8) -> u32 {
    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix.min(32)))
    };
    u32::from(ip) & mask
}

fn shares_subnet(rails: &[Rail]) -> bool {
    let subnets: Vec<u32> = rails
        .iter()
        .filter_map(|rail| rail.prefix.map(|prefix| network(rail.ip, prefix)))
        .collect();
    let unique: BTreeSet<&u32> = subnets.iter().collect();
    unique.len() < subnets.len()
}

fn arp_ok(facts: &NodeFacts, netdevs: &[&str]) -> bool {
    netdevs.iter().all(|netdev| facts.arp(netdev) == (1, 2))
}

/// The netplan and sysctl commands each node needs, or `None` when its ports are already right.
///
/// A port that DHCP addresses keeps DHCP. A port without an address gets one on the subnet the
/// other Sparks use (a free host number), or on 10.200.0.0/24 when none has an address yet. Both
/// lanes of a node land on one subnet, so the ARP settings come with any address change.
fn network_fixes(probes: &[ProbeResult], name: &dyn Fn(usize) -> String) -> Vec<Option<String>> {
    let mut used: BTreeSet<Ipv4Addr> = probes
        .iter()
        .flatten()
        .flat_map(|facts| facts.addresses.iter().map(|address| address.ip))
        .collect();
    let subnet = probes
        .iter()
        .flatten()
        .find_map(|facts| {
            let rails = facts.rails();
            let primary = facts.primary_rail(&rails)?;
            Some((
                Ipv4Addr::from(network(primary.ip, primary.prefix?)),
                primary.prefix?,
            ))
        })
        .unwrap_or(DEFAULT_SUBNET);

    let mut fixes = Vec::new();
    for (rank, probe) in probes.iter().enumerate() {
        let Ok(facts) = probe else {
            fixes.push(None);
            continue;
        };
        let rails = facts.rails();
        let unaddressed = facts.unaddressed();
        let mut ports: Vec<(String, String)> = Vec::new();
        let mut needs_netplan = !unaddressed.is_empty();
        for rail in &rails {
            if rail.mtu.is_some_and(|mtu| mtu < FABRIC_MTU) {
                needs_netplan = true;
            }
        }
        if needs_netplan {
            for rail in &rails {
                // NetworkManager's DHCP leases can show as permanent addresses; the routes it
                // installs still say `proto dhcp`.
                let dynamic = facts
                    .addresses
                    .iter()
                    .any(|address| address.netdev == rail.netdev && address.dynamic)
                    || facts
                        .routes
                        .iter()
                        .any(|route| route.netdev == rail.netdev && route.dhcp);
                let line = if dynamic {
                    "dhcp4: true".to_owned()
                } else {
                    format!(
                        "addresses: [{}/{}]",
                        rail.ip,
                        rail.prefix.unwrap_or(subnet.1)
                    )
                };
                ports.push((rail.netdev.clone(), line));
            }
            for (lane, netdev) in unaddressed.iter().enumerate() {
                let ip = free_address(subnet, rank, rails.len() + lane, &used);
                used.insert(ip);
                ports.push((netdev.clone(), format!("addresses: [{ip}/{}]", subnet.1)));
            }
            ports.sort_by_key(|(netdev, _)| netdev.to_ascii_lowercase());
        }
        // New addresses join the shared subnet, so two lanes afterwards means one subnet.
        let lanes: Vec<&str> = rails
            .iter()
            .map(|rail| rail.netdev.as_str())
            .chain(unaddressed.iter().map(String::as_str))
            .collect();
        let needs_arp = lanes.len() > 1
            && (shares_subnet(&rails) || !unaddressed.is_empty())
            && !arp_ok(facts, &lanes);
        if ports.is_empty() && !needs_arp {
            fixes.push(None);
            continue;
        }
        let mut script = format!(
            "# Run on {} in a terminal (it asks for your password).\n",
            name(rank)
        );
        if !ports.is_empty() {
            script.push_str(&format!("sudo tee {NETPLAN_FILE} > /dev/null <<'EOF'\n"));
            script.push_str("network:\n  version: 2\n  ethernets:\n");
            for (netdev, line) in &ports {
                script.push_str(&format!(
                    "    {netdev}:\n      mtu: {FABRIC_MTU}\n      {line}\n"
                ));
            }
            script.push_str("EOF\n");
            script.push_str(&format!("sudo chmod 600 {NETPLAN_FILE}\n"));
        }
        if needs_arp {
            script.push_str(&format!("sudo tee {SYSCTL_FILE} > /dev/null <<'EOF'\n"));
            script.push_str(
                "net.ipv4.conf.all.arp_ignore = 1\nnet.ipv4.conf.all.arp_announce = 2\nEOF\n",
            );
            script.push_str(&format!("sudo sysctl -p {SYSCTL_FILE}\n"));
        }
        if !ports.is_empty() {
            script.push_str("sudo netplan apply\n");
        }
        fixes.push(Some(script));
    }
    fixes
}

/// A free address on `subnet` for lane `lane` of `rank`: host `11 + 10 * lane + rank`, moving up
/// past anything already in use.
fn free_address(
    subnet: (Ipv4Addr, u8),
    rank: usize,
    lane: usize,
    used: &BTreeSet<Ipv4Addr>,
) -> Ipv4Addr {
    let base = u32::from(subnet.0);
    let hosts = if subnet.1 >= 31 {
        1
    } else {
        (1u32 << (32 - u32::from(subnet.1))) - 2
    };
    let start = u32::try_from(11 + 10 * lane + rank).unwrap_or(11);
    let mut host = start;
    loop {
        let candidate = Ipv4Addr::from(base + 1 + (host - 1) % hosts.max(1));
        if !used.contains(&candidate) || host > start + hosts {
            return candidate;
        }
        host += 1;
    }
}

/// The arguments [`INSPECT_SCRIPT`] takes for `rank`. `checkpoint` is the lead's checkpoint
/// directory, `None` while the model is not downloaded; every node keeps it at the same path.
#[must_use]
pub fn script_args(config: &SparkConfig, rank: usize, checkpoint: Option<&Path>) -> Vec<String> {
    let converted = config.converted_path();
    vec![
        config.binary_path().display().to_string(),
        config.cubins_path().display().to_string(),
        config
            .nccl_lib_path()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        checkpoint
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
        // `sparkflash convert` names its output rank-{rank}-of-{world}.bin and writes the
        // manifest after the rank file is complete.
        converted
            .join(format!("rank-{rank}-of-{WORLD_SIZE}.bin"))
            .display()
            .to_string(),
        converted
            .join("sparkflash-converted-manifest.json")
            .display()
            .to_string(),
        config
            .hub_cache_dir()
            .map(|path| path.display().to_string())
            .unwrap_or_default(),
    ]
}

/// The remote command line that runs the probe with `args` under `bash`.
#[must_use]
pub fn remote_command(args: &[String]) -> String {
    let mut line = format!("bash -c {} sparkflash-inspect", shell_quote(INSPECT_SCRIPT));
    for arg in args {
        line.push(' ');
        line.push_str(&shell_quote(arg));
    }
    line
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A node with both halves of port 1 cabled, same subnet, MTU 9000 and the ARP settings.
    pub(crate) fn node_text(host: &str, a: u8, b: u8) -> String {
        format!(
            "host {host}
gpu NVIDIA GB10
port rocep1s0f0 DOWN 40
gid rocep1s0f0 0 IB/RoCE_v1 fe80:0000:0000:0000:4ebb:47ff:fe00:0001 enp1s0f0np0
port rocep1s0f1 ACTIVE 200
gid rocep1s0f1 0 IB/RoCE_v1 fe80:0000:0000:0000:4ebb:47ff:fe00:0002 enp1s0f1np1
gid rocep1s0f1 1 RoCE_v2 fe80:0000:0000:0000:4ebb:47ff:fe00:0002 enp1s0f1np1
gid rocep1s0f1 2 IB/RoCE_v1 0000:0000:0000:0000:0000:ffff:0a00:01{a:02x} enp1s0f1np1
gid rocep1s0f1 3 RoCE_v2 0000:0000:0000:0000:0000:ffff:0a00:01{a:02x} enp1s0f1np1
port roceP2p1s0f1 ACTIVE 200
gid roceP2p1s0f1 3 RoCE_v2 0000:0000:0000:0000:0000:ffff:0a00:01{b:02x} enP2p1s0f1np1
arp all 1 2
link enp1s0f1np1 9000 up 0 0
link enP2p1s0f1np1 9000 up 0 0
link enp1s0f0np0 1500 down 0 0
addr enp1s0f1np1 198.51.100.{a}/24 static
addr enP2p1s0f1np1 198.51.100.{b}/24 static
route default via 10.0.0.1 dev enP7s7 proto dhcp metric 100
route 198.51.100.0/24 dev enP2p1s0f1np1 proto kernel scope link src 198.51.100.{b} metric 100
route 198.51.100.0/24 dev enp1s0f1np1 proto kernel scope link src 198.51.100.{a} metric 101
file binary 104857600 1700000000 {bin}
tree cubins 12 3000000 {cub}
file nccl 346000000 1700000000 {nccl}
tree checkpoint 55 510000000000 -
file converted 19500000000 1700000100 -
file manifest 4096 1700000200 -
free cache 2000000000000
free converted 2000000000000
",
            bin = "a".repeat(64),
            cub = "b".repeat(64),
            nccl = "c".repeat(64),
        )
    }

    fn cluster() -> Vec<ProbeResult> {
        vec![
            Ok(parse_facts(&node_text("spark-0", 11, 21))),
            Ok(parse_facts(&node_text("spark-1", 12, 22))),
            Ok(parse_facts(&node_text("spark-2", 13, 23))),
            Ok(parse_facts(&node_text("spark-3", 14, 24))),
        ]
    }

    #[test]
    fn the_probe_output_parses_into_facts() {
        let facts = parse_facts(&node_text("spark-0", 11, 21));
        assert_eq!(facts.hostname, "spark-0");
        assert_eq!(facts.gpu.as_deref(), Some("NVIDIA GB10"));
        assert_eq!(facts.ports.len(), 3);
        assert_eq!(facts.arp_all, (1, 2));
        let rails = facts.rails();
        assert_eq!(
            rails
                .iter()
                .map(|rail| rail.device.as_str())
                .collect::<Vec<_>>(),
            ["rocep1s0f1", "roceP2p1s0f1"]
        );
        assert_eq!(rails[0].ip, Ipv4Addr::new(10, 0, 1, 11));
        assert_eq!(rails[0].gid_index, 3);
        assert_eq!(rails[0].prefix, Some(24));
        assert_eq!(rails[1].mtu, Some(9000));
        assert_eq!(
            facts.files["binary"].as_ref().map(|file| file.size),
            Some(104_857_600)
        );
        assert_eq!(
            facts.trees["checkpoint"].as_ref().map(|tree| tree.bytes),
            Some(510_000_000_000)
        );
        assert_eq!(
            facts.trees["checkpoint"]
                .as_ref()
                .and_then(|tree| tree.sha256.clone()),
            None
        );
        assert_eq!(facts.free["cache"], 2_000_000_000_000);
        assert!(facts.unaddressed().is_empty());
    }

    #[test]
    fn a_healthy_cluster_yields_the_dual_rail_plan_on_the_preferred_lane() {
        let config = SparkConfig::default();
        let report = analyze(&config, &cluster(), true, 1);
        assert!(report.findings.is_empty(), "{:?}", report.findings);
        let plan = report.plan.clone().expect("plan");
        assert_eq!(plan.nccl_ib_hca, "rocep1s0f1,roceP2p1s0f1");
        // The route with the lower metric wins: the kernel sources the fast subnet from it.
        assert_eq!(plan.nccl_socket_ifname, "enP2p1s0f1np1");
        assert_eq!(plan.nccl_ib_gid_index, Some(3));
        assert_eq!(
            plan.peers,
            ["198.51.100.21", "198.51.100.22", "198.51.100.23", "198.51.100.24"]
        );
        assert!(report.network_ok());
        for node in &report.nodes {
            assert_eq!(node.software, Check::Ready, "rank {}", node.rank);
            assert_eq!(node.model, Check::Ready, "rank {}", node.rank);
            assert_eq!(node.converted, Check::Ready, "rank {}", node.rank);
            assert!(node.network_fix.is_none(), "rank {}", node.rank);
        }

        let mut config = SparkConfig::default();
        assert!(!plan.matches(&config));
        plan.apply(&mut config);
        assert!(plan.matches(&config));
        config.validate().expect("the applied plan validates");
    }

    #[test]
    fn a_node_that_differs_from_the_lead_needs_its_software_and_model_copied() {
        let mut probes = cluster();
        let text = node_text("spark-2", 13, 23)
            .replace(&"a".repeat(64), &"d".repeat(64))
            .replace(
                "tree checkpoint 55 510000000000",
                "tree checkpoint 20 190000000000",
            )
            .replace("file manifest 4096 1700000200 -", "file manifest - - -");
        probes[2] = Ok(parse_facts(&text));
        let text = node_text("spark-3", 14, 24)
            .replace(
                &format!("file binary 104857600 1700000000 {}", "a".repeat(64)),
                "file binary - - -",
            )
            .replace("tree checkpoint 55 510000000000 -", "tree checkpoint - - -")
            .replace(
                "file converted 19500000000 1700000100 -",
                "file converted - - -",
            );
        probes[3] = Ok(parse_facts(&text));
        let report = analyze(&SparkConfig::default(), &probes, true, 1);
        assert_eq!(report.nodes[2].software, Check::Outdated);
        assert_eq!(report.nodes[2].model, Check::Partial);
        assert_eq!(report.nodes[2].converted, Check::Partial);
        assert_eq!(report.nodes[3].software, Check::Missing);
        assert_eq!(report.nodes[3].model, Check::Missing);
        assert_eq!(report.nodes[3].converted, Check::Missing);
        assert_eq!(report.model_bytes, 510_000_000_000);

        let report = analyze(&SparkConfig::default(), &cluster(), false, 1);
        assert_eq!(
            report.nodes[0].model,
            Check::Partial,
            "an unfinished download"
        );
    }

    #[test]
    fn an_unreachable_node_blocks_the_plan_and_says_why() {
        let mut probes = cluster();
        probes[1] = Err("Permission denied (publickey)".to_owned());
        let report = analyze(&SparkConfig::default(), &probes, true, 1);
        assert!(report.plan.is_none());
        assert!(!report.network_ok());
        assert!(!report.nodes[1].reachable);
        assert_eq!(report.findings[0].rank, Some(1));
        assert!(report.findings[0].message.contains("Permission denied"));
    }

    #[test]
    fn a_fresh_node_gets_addresses_mtu_and_arp_in_one_command() {
        let mut probes = cluster();
        let fresh = "host spark-3
port rocep1s0f1 ACTIVE 200
gid rocep1s0f1 0 IB/RoCE_v1 fe80:0000:0000:0000:4ebb:47ff:fe00:0002 enp1s0f1np1
gid rocep1s0f1 1 RoCE_v2 fe80:0000:0000:0000:4ebb:47ff:fe00:0002 enp1s0f1np1
port roceP2p1s0f1 ACTIVE 200
gid roceP2p1s0f1 0 IB/RoCE_v1 fe80:0000:0000:0000:4ebb:47ff:fe00:0006 enP2p1s0f1np1
arp all 0 0
link enp1s0f1np1 1500 up 0 0
link enP2p1s0f1np1 1500 up 0 0
";
        probes[3] = Ok(parse_facts(fresh));
        let report = analyze(&SparkConfig::default(), &probes, true, 1);
        assert!(report.plan.is_none());
        let finding = report
            .findings
            .iter()
            .find(|finding| finding.rank == Some(3))
            .expect("rank 3 finding");
        assert_eq!(finding.severity, Severity::Error);
        assert!(
            finding.message.contains("no IP address"),
            "{}",
            finding.message
        );
        let fix = report.nodes[3]
            .network_fix
            .clone()
            .expect("a fix for rank 3");
        // Free host numbers on the subnet the others use, one per lane.
        assert!(
            fix.contains("enp1s0f1np1:\n      mtu: 9000\n      addresses: [198.51.100.14/24]"),
            "{fix}"
        );
        assert!(
            fix.contains("enP2p1s0f1np1:\n      mtu: 9000\n      addresses: [198.51.100.24/24]"),
            "{fix}"
        );
        assert!(fix.contains("net.ipv4.conf.all.arp_ignore = 1"), "{fix}");
        assert!(fix.contains("sudo netplan apply"), "{fix}");
        assert!(report.nodes[0].network_fix.is_none());
    }

    #[test]
    fn a_cluster_with_no_addresses_is_put_on_the_default_subnet() {
        let fresh = |host: &str| {
            parse_facts(&format!(
                "host {host}
port rocep1s0f1 ACTIVE 200
gid rocep1s0f1 0 IB/RoCE_v1 fe80:0000:0000:0000:4ebb:47ff:fe00:0002 enp1s0f1np1
arp all 0 0
link enp1s0f1np1 1500 up 0 0
"
            ))
        };
        let probes: Vec<ProbeResult> = (0..4)
            .map(|rank| Ok(fresh(&format!("spark-{rank}"))))
            .collect();
        let report = analyze(&SparkConfig::default(), &probes, false, 1);
        let fixes: Vec<String> = report
            .nodes
            .iter()
            .map(|node| node.network_fix.clone().expect("every node needs one"))
            .collect();
        for (rank, fix) in fixes.iter().enumerate() {
            assert!(
                fix.contains(&format!("addresses: [10.200.0.{}/24]", 11 + rank)),
                "{fix}"
            );
            // One lane: no ARP settings needed.
            assert!(!fix.contains("arp_ignore"), "{fix}");
        }
    }

    #[test]
    fn mtu_and_arp_problems_are_warnings_with_a_fix_that_keeps_dhcp() {
        let mut probes = cluster();
        let text = node_text("spark-1", 12, 22)
            .replace("arp all 1 2", "arp all 0 0")
            .replace("link enp1s0f1np1 9000 up", "link enp1s0f1np1 1500 up")
            .replace(
                "addr enp1s0f1np1 198.51.100.12/24 static",
                "addr enp1s0f1np1 198.51.100.12/24 dynamic",
            );
        probes[1] = Ok(parse_facts(&text));
        let report = analyze(&SparkConfig::default(), &probes, true, 1);
        assert!(report.network_ok(), "{:?}", report.findings);
        let messages: Vec<&str> = report
            .findings
            .iter()
            .map(|finding| finding.message.as_str())
            .collect();
        assert!(
            messages.iter().any(|message| message.contains("MTU 1500")),
            "{messages:?}"
        );
        assert!(
            messages
                .iter()
                .any(|message| message.contains("arp_ignore=1")),
            "{messages:?}"
        );
        let fix = report.nodes[1].network_fix.clone().expect("fix");
        assert!(
            fix.contains("enp1s0f1np1:\n      mtu: 9000\n      dhcp4: true"),
            "{fix}"
        );
        assert!(
            fix.contains("enP2p1s0f1np1:\n      mtu: 9000\n      addresses: [198.51.100.22/24]"),
            "{fix}"
        );
        assert!(fix.contains("arp_announce = 2"), "{fix}");
    }

    #[test]
    fn nodes_on_different_subnets_or_sharing_an_address_are_refused() {
        let mut probes = cluster();
        let text = node_text("spark-2", 13, 23)
            .replace("198.51.100.", "10.9.1.")
            .replace("ffff:0a00:01", "ffff:0a09:01");
        probes[2] = Ok(parse_facts(&text));
        let report = analyze(&SparkConfig::default(), &probes, true, 1);
        assert!(report.plan.is_none());
        assert!(
            report
                .findings
                .iter()
                .any(|finding| finding.message.contains("not on"))
        );

        let mut probes = cluster();
        probes[3] = Ok(parse_facts(&node_text("spark-3", 13, 23)));
        let report = analyze(&SparkConfig::default(), &probes, true, 1);
        assert!(report.plan.is_none());
        assert!(
            report
                .findings
                .iter()
                .any(|finding| finding.message.contains("same fast-network address"))
        );
    }

    #[test]
    fn the_remote_command_quotes_the_script_and_every_argument() {
        let mut config = SparkConfig::default();
        config.paths.repo_root = "/srv/spark flash".to_owned();
        let args = script_args(&config, 2, Some(Path::new("/models/snap")));
        assert_eq!(args[3], "/models/snap");
        assert_eq!(args[0], "/srv/spark flash/target/release/sparkflash");
        assert!(
            args[4].ends_with("/sparkflash/converted/rank-2-of-4.bin"),
            "{}",
            args[4]
        );
        let line = remote_command(&args);
        assert!(line.starts_with("bash -c '"), "{line}");
        assert!(
            line.contains(" sparkflash-inspect '/srv/spark flash/target/release/sparkflash' "),
            "{line}"
        );
    }
}
