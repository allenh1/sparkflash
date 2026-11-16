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
echo "forward $(cat /proc/sys/net/ipv4/ip_forward 2>/dev/null || echo 0)"
echo "rpfilter $(cat /proc/sys/net/ipv4/conf/all/rp_filter 2>/dev/null || echo 0)"
for net in /sys/class/net/*; do
  name=${net##*/}
  [ -e "$net/device" ] || continue
  echo "link $name $(cat "$net/mtu") $(cat "$net/operstate") $(cat "/proc/sys/net/ipv4/conf/$name/arp_ignore" 2>/dev/null || echo 0) $(cat "/proc/sys/net/ipv4/conf/$name/arp_announce" 2>/dev/null || echo 0) $(cat "/proc/sys/net/ipv4/conf/$name/rp_filter" 2>/dev/null || echo 0)"
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

/// The metric the fabric's host routes are written with. It has to be higher than the connected
/// route of the address itself: the check picks the rail the kernel sources the fast subnet from
/// by the lowest metric, and a host route must not change which rail that is. A `/32` with this
/// metric is the port setup's own route, so a re-cable that no longer needs it removes it.
const FABRIC_ROUTE_METRIC: u32 = 200;

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
    /// Its own `rp_filter`; the kernel uses the larger of this and `all`.
    pub rp_filter: u8,
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
    /// The next hop, when the route has one.
    pub via: Option<String>,
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
    /// Whether `net.ipv4.ip_forward` is on, which a ring's relayed diagonals need.
    pub forward: bool,
    /// `net.ipv4.conf.all.rp_filter`: 1 (strict) drops the traffic a relay forwards, 2 (loose)
    /// passes it, 0 disables the check.
    pub rp_filter: u8,
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
            ("forward", [on]) => facts.forward = *on == "1",
            ("rpfilter", [value]) => facts.rp_filter = value.parse().unwrap_or(0),
            ("link", [name, mtu, state, ignore, announce]) => facts.links.push(NetLink {
                name: (*name).to_owned(),
                mtu: mtu.parse().unwrap_or(0),
                up: *state == "up",
                arp_ignore: ignore.parse().unwrap_or(0),
                arp_announce: announce.parse().unwrap_or(0),
                rp_filter: 0,
            }),
            ("link", [name, mtu, state, ignore, announce, rp]) => facts.links.push(NetLink {
                name: (*name).to_owned(),
                mtu: mtu.parse().unwrap_or(0),
                up: *state == "up",
                arp_ignore: ignore.parse().unwrap_or(0),
                arp_announce: announce.parse().unwrap_or(0),
                rp_filter: rp.parse().unwrap_or(0),
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
                    via: after("via").map(|via| (*via).to_owned()),
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

    /// Whether any of these ports is under strict reverse-path filtering. The kernel takes the
    /// larger of the port's own value and `all`, so a per-interface 1 with `all=0` is strict too.
    fn rp_strict(&self, netdevs: &[&str]) -> bool {
        netdevs.iter().any(|netdev| {
            self.link(netdev)
                .map_or(0, |link| link.rp_filter)
                .max(self.rp_filter)
                == 1
        })
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

/// One measured link: a port of `rank` reaches `peer` on the same layer-2 segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FabricLink {
    /// The Spark whose port answered.
    pub rank: usize,
    /// The port's netdev.
    pub netdev: String,
    /// The Spark on the other end.
    pub peer: usize,
    /// The peer address that answered on this port. The port is two PCIe halves with an address
    /// each, so one cable can answer for two of the peer's addresses; the setup needs the address
    /// because it is also the next hop of a route through the Spark the cable reaches.
    pub address: String,
}

/// The cabling the fast ports describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FabricShape {
    /// Every port reaches every other Spark: the cables run through a switch.
    Switched,
    /// Every port reaches exactly one Spark, and the pairs form the 0-1-2-3-0 ring.
    Ring,
    /// Each Spark's two ports reach the same Spark: two-node pairs, not a ring.
    Pairs,
    /// A cable is missing, a port is down, or the wiring is not one of the named shapes.
    Unknown,
}

/// What the link check measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Fabric {
    /// The links the probes proved, in rank order.
    pub links: Vec<FabricLink>,
    /// For each rank in order, the ranks its own ports pinged; `None` when its probe did not run.
    pub reachable: Vec<Option<Vec<usize>>>,
    /// The shape the links describe.
    pub shape: FabricShape,
    /// Whether the ports answered an address probe at all (`false` when `arping` is missing).
    pub links_measured: bool,
}

/// The shape a set of measured links describes.
///
/// A switch floods an ARP request to every port, so every port of every Spark answers for every
/// address: each port reaches all `world - 1` peers. A DAC between two Sparks is its own segment,
/// so a port reaches only the Spark plugged into it. A cabled ConnectX-7 port is two PCIe halves
/// with a netdev and an address each (`docs/serving.md`), so a Spark cabled to two neighbours
/// answers on four netdevs; the classifier groups by the peers a rank reaches, not by netdev, and
/// reads the ring from those peers (the two neighbours) and the pairs wiring from a single one.
#[must_use]
pub fn shape_of(links: &[FabricLink], world: usize) -> FabricShape {
    if world < 2 || links.is_empty() {
        return FabricShape::Unknown;
    }
    let mut per_port: BTreeMap<(usize, &str), BTreeSet<usize>> = BTreeMap::new();
    for link in links {
        per_port
            .entry((link.rank, link.netdev.as_str()))
            .or_default()
            .insert(link.peer);
    }
    let switched = (0..world).all(|rank| {
        let mut mine = per_port
            .iter()
            .filter(|((owner, _), _)| *owner == rank)
            .map(|(_, peers)| peers);
        let mut any = false;
        for peers in mine.by_ref() {
            any = true;
            if peers.len() != world - 1 {
                return false;
            }
        }
        any
    });
    if switched {
        return FabricShape::Switched;
    }
    if !per_port.values().all(|peers| peers.len() == 1) {
        return FabricShape::Unknown;
    }
    let mut neighbours: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for ((rank, _), peers) in &per_port {
        if let Some(peer) = peers.iter().next() {
            neighbours.entry(*rank).or_default().insert(*peer);
        }
    }
    // Only the Sparks a rank can reach count, not the netdevs it reaches them from: a cabled
    // ConnectX-7 port is two PCIe halves, so a two-cable Spark presents four netdevs and every
    // cable to one neighbour answers on two of them.
    let pairs = (0..world).all(|rank| {
        neighbours
            .get(&rank)
            .is_some_and(|targets| targets.len() == 1)
    });
    if pairs {
        return FabricShape::Pairs;
    }
    let ring = (0..world).all(|rank| {
        neighbours.get(&rank).is_some_and(|targets| {
            let expected: BTreeSet<usize> = [(rank + world - 1) % world, (rank + 1) % world]
                .into_iter()
                .collect();
            *targets == expected
        })
    });
    if ring {
        return FabricShape::Ring;
    }
    FabricShape::Unknown
}

/// Every fast-network address of every rank, for the link probe: one entry per rail with the
/// primary rail first, deduplicated. `None` when a rank has no fast address to probe with.
///
/// The probe needs them all. A cross-connect's two addresses usually sit on different ports, so
/// probing only the collective address (`primary_rail`) would leave the far end of one cable
/// unanswered and make a correct ring look like two-node pairs.
#[must_use]
pub fn probe_addresses(config: &SparkConfig, probes: &[ProbeResult]) -> Option<Vec<Vec<String>>> {
    if config.world_size() < 2 || probes.len() != config.world_size() {
        return None;
    }
    let mut addresses = Vec::with_capacity(probes.len());
    for probe in probes {
        let facts = probe.as_ref().ok()?;
        let rails = facts.rails();
        let primary = facts.primary_rail(&rails)?;
        let mut mine = vec![primary.ip.to_string()];
        for rail in &rails {
            let address = rail.ip.to_string();
            if !mine.contains(&address) {
                mine.push(address);
            }
        }
        addresses.push(mine);
    }
    Some(addresses)
}

/// The address the plan would give each rank, or `None` when the facts do not support a plan.
#[must_use]
pub fn planned_peers(config: &SparkConfig, probes: &[ProbeResult]) -> Option<Vec<String>> {
    if config.world_size() < 2 || probes.len() != config.world_size() {
        return None;
    }
    let mut peers = Vec::with_capacity(probes.len());
    for probe in probes {
        let facts = probe.as_ref().ok()?;
        let rails = facts.rails();
        let primary = facts.primary_rail(&rails)?;
        peers.push(primary.ip.to_string());
    }
    Some(peers)
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
    /// A command that configures the fast ports, when they need it.
    pub network_fix: Option<String>,
    /// The same configuration without the password prompts or the leading comments, for running
    /// over SSH under sudo -n bash -c.
    pub network_fix_remote: Option<String>,
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
    /// What the link check measured of the fast network, `None` when it could not run.
    pub fabric: Option<Fabric>,
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
    analyze_fabric(config, probes, None, model_complete_on_lead, checked_ms)
}

/// Builds the report from the probes and the link check.
#[must_use]
pub fn analyze_fabric(
    config: &SparkConfig,
    probes: &[ProbeResult],
    fabric: Option<Fabric>,
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

    // Fabric: every node needs a lane, and the lanes must agree on what NCCL is told. A TP=1
    // pack has no collectives, so its single node needs no fast-network port at all.
    let distributed = config.world_size() > 1;
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
        if distributed && let Ok(facts) = probe {
            if rails.is_empty() {
                let message = if facts.unaddressed().is_empty() {
                    format!(
                        "{} has no fast-network link. Check that its QSFP cable is plugged in \
                             at both ends.",
                        name(rank)
                    )
                } else {
                    format!(
                        "{}'s fast-network port ({}) is cabled but has no IP address yet. Run \
                             the network command shown for it.",
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
            // The ring's routes send traffic out the half of a port that does not own the source
            // address, so strict reverse-path filtering drops it: name that as the cause instead
            // of letting the check blame the cables.
            if fabric
                .as_ref()
                .is_some_and(|fabric| fabric.shape == FabricShape::Ring)
                && facts.rp_strict(&netdevs)
            {
                findings.push(Finding {
                    rank: Some(rank),
                    severity: Severity::Error,
                    message: format!(
                        "{} has strict reverse-path filtering on its fast ports, which drops the \
                         traffic the ring's routes send out the other half of a port (a direct \
                         neighbour and a diagonal alike). Run the network command shown for it \
                         (it sets rp_filter=2), then check again.",
                        name(rank)
                    ),
                });
            }
            if shares_subnet(&rails) && !arp_ok(facts, &netdevs) {
                findings.push(Finding {
                    rank: Some(rank),
                    severity: Severity::Warning,
                    message: format!(
                        "{} has two fast lanes on one subnet without arp_ignore=1 and \
                             arp_announce=2, so replies can leave by the wrong lane. Run the \
                             network command shown for it.",
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
    if distributed && lane_counts.len() > 1 {
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

    let plan = if distributed {
        plan_fabric(probes, &rails_by_rank, &mut findings, &name)
    } else {
        None
    };

    // The link check measures what the cables actually are: a switch, a
    // direct ring, or something to fix. The schedule is a hint from the
    // config; the shapes below compare it with the measurements.
    let schedule = crate::config::fabric_schedule_name(&config.cluster.fabric_schedule);
    if let Some(fabric) = &fabric {
        // The peers each rank has a measured link to: the pairs a cable actually joins. A
        // non-neighbour on a ring is not a broken cable, so the message below names the hairpin
        // that has to forward it instead.
        let mut linked: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); fabric.reachable.len()];
        for link in &fabric.links {
            if let Some(set) = linked.get_mut(link.rank) {
                set.insert(link.peer);
            }
        }
        for (rank, reached) in fabric.reachable.iter().enumerate() {
            // The node answered the main probe but not the link probe: say so instead of
            // silently showing a wiring verdict computed from the other nodes alone.
            if reached.is_none() && probes.get(rank).is_some_and(|probe| probe.is_ok()) {
                findings.push(Finding {
                    rank: Some(rank),
                    severity: Severity::Warning,
                    message: format!(
                        "the link check did not complete on {}; the wiring could not be measured",
                        name(rank)
                    ),
                });
                continue;
            }
            let Some(reached) = reached else {
                continue;
            };
            for peer in 0..fabric.reachable.len() {
                if peer == rank || reached.contains(&peer) {
                    continue;
                }
                let neighbour = linked.get(rank).is_some_and(|set| set.contains(&peer));
                // The reference transport opens a connection to every rank's address before the
                // launch can start, so a peer that does not answer is an error whatever the cables
                // measure; only the message depends on the shape. A ring's non-neighbour is
                // reached through the Spark between the two cables, which has to forward that
                // address: the port setup writes that route and turns on forwarding, and without it
                // a bare DAC ring carries nothing across the gap.
                let message = match fabric.shape {
                    _ if !fabric.links_measured => format!(
                        "{} cannot reach {} over the fast network. The wiring could not be \
                         measured, so this is the reachability result alone; the launch opens a \
                         connection to every rank and needs {} to answer.",
                        name(rank),
                        name(peer),
                        name(peer)
                    ),
                    FabricShape::Ring => format!(
                        "{} cannot reach {} over the fast network. The ring needs a route to that \
                         address on the cable that reaches it: a neighbour's collective address \
                         can sit on its other port, and a non-neighbour goes through the Spark \
                         between the two cables. Set the ports up: the command writes those \
                         routes and turns on forwarding, then check again.",
                        name(rank),
                        name(peer)
                    ),
                    FabricShape::Pairs if !neighbour => format!(
                        "{} cannot reach {} over the fast network; the four Sparks are wired as \
                         two pairs, so a rank in the other pair has no fast path. The launch \
                         opens a connection to every rank.",
                        name(rank),
                        name(peer)
                    ),
                    _ => format!(
                        "{} cannot reach {} over the fast network{}. The launch opens a connection \
                         to every rank and needs this one.",
                        name(rank),
                        name(peer),
                        if fabric.shape == FabricShape::Unknown && !neighbour {
                            ", and no cable was measured to it"
                        } else {
                            ""
                        }
                    ),
                };
                findings.push(Finding {
                    rank: Some(rank),
                    severity: Severity::Error,
                    message,
                });
            }
        }
        // A probe that could not measure the cables has to say so in the report itself, not only
        // in the browser's picker: /api/state is what other consumers read.
        if !fabric.links_measured {
            findings.push(Finding {
                rank: None,
                severity: Severity::Warning,
                message: "The wiring could not be measured (a link probe did not run, or arping \
                          is missing or cannot open a raw socket on a Spark), so only \
                          reachability was checked."
                    .to_owned(),
            });
        }
        // arping present and every attempt answered nothing: either the ports are all
        // disconnected, or the tool cannot open a raw socket (no CAP_NET_RAW). Say which is
        // unknown instead of reporting the wiring as unrecognised.
        if fabric.links.is_empty() && fabric.links_measured && fabric.reachable.len() > 1 {
            findings.push(Finding {
                rank: None,
                severity: Severity::Warning,
                message:
                    "No fast port answered an ARP probe on any Spark, so the wiring could not \
                          be measured; check that every port is cabled and addressed, and that \
                          arping can open a raw socket on each Spark."
                        .to_owned(),
            });
        }
        match fabric.shape {
            FabricShape::Switched => {
                if schedule == "switchless" {
                    findings.push(Finding {
                        rank: None,
                        severity: Severity::Warning,
                        message: "The fabric measures as a switch but the configured schedule is \
                                  switchless; the schedule is a traffic hint, so this works either \
                                  way, but the configurations should match the wiring."
                            .to_owned(),
                    });
                }
            }
            FabricShape::Ring => {
                if schedule == "switched" {
                    findings.push(Finding {
                        rank: None,
                        severity: Severity::Warning,
                        message: "The fast ports are wired as a direct ring but the configured \
                                  schedule is switched; choose switchless so each Spark talks to its \
                                  neighbours over the ring."
                            .to_owned(),
                    });
                }
            }
            FabricShape::Pairs => {
                findings.push(Finding {
                    rank: None,
                    severity: Severity::Warning,
                    message: "Each Spark's two fast ports reach the same Spark: the four machines \
                              are wired as two-node pairs, not a ring. A pair can talk, but ranks \
                              across the pairs cannot use the fast fabric."
                        .to_owned(),
                });
            }
            FabricShape::Unknown => {
                if fabric.links_measured {
                    findings.push(Finding {
                        rank: None,
                        severity: Severity::Warning,
                        message:
                            "The measured links match neither a switch nor a ring; check that \
                                  the cables land in the matching ports on each Spark."
                                .to_owned(),
                    });
                }
            }
        }
    }

    // Software, model and converted weights, each node against the lead.
    let lead_file = |label: &str| lead.and_then(|facts| facts.files.get(label).cloned().flatten());
    let lead_tree = |label: &str| lead.and_then(|facts| facts.trees.get(label).cloned().flatten());
    let lead_model = lead_tree("checkpoint");
    let wants_nccl = config.nccl_lib_path().is_some();
    let wants_cubins = config.requires_cubins();
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
                network_fix_remote: None,
            });
            continue;
        };
        let file = |label: &str| facts.files.get(label).cloned().flatten();
        let tree = |label: &str| facts.trees.get(label).cloned().flatten();
        let software = if rank == 0 {
            let present = file("binary").is_some()
                && (!wants_cubins || tree("cubins").is_some_and(|tree| tree.files > 0))
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
                && (!wants_cubins || same_tree("cubins"))
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
        let converted = if !config.requires_conversion() {
            // A TP=1 pack loads the released checkpoint directly; there is no converted layout.
            Check::Ready
        } else {
            match (file("converted"), file("manifest")) {
                (None, _) => Check::Missing,
                // The GLM converter writes the rank file and no manifest; that file is the
                // completion marker. V4.1 writes its manifest after the rank file.
                (Some(_), _) if !config.requires_conversion_manifest() => Check::Ready,
                (Some(rank_file), Some(manifest)) if manifest.mtime >= rank_file.mtime => {
                    Check::Ready
                }
                (Some(_), _) => Check::Partial,
            }
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
            network_fix_remote: None,
        });
    }

    // A network command for every node whose ports need addresses, MTU, ARP settings or the
    // routes a ring needs.
    let fixes = network_fixes(probes, fabric.as_ref(), &name);
    for (node, fix) in nodes.iter_mut().zip(fixes) {
        if let Some((text, remote)) = fix {
            node.network_fix = Some(text);
            node.network_fix_remote = Some(remote);
        }
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
        fabric,
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
    netdevs.iter().all(|netdev| {
        let (ignore, announce) = facts.arp(netdev);
        // 1 and 2 both restrict a reply to the port that owns the address (a hardened host may
        // pin 2); anything else needs the fix, which writes 1/2. Announcing at least 1 avoids
        // advertising another port's address.
        (1..=2).contains(&ignore) && announce >= 1
    })
}

/// One route the port setup pins on a Spark: a peer's collective address on the port that reaches
/// it. A cabled neighbour answers on that port; a ring's diagonal has no cable, so it goes via the
/// Spark between the two cables and the kernel relays it there.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FabricRoute {
    /// The peer's collective address.
    to: Ipv4Addr,
    /// The next hop, `None` when the peer is cabled directly.
    via: Option<Ipv4Addr>,
    /// The port the route leaves by.
    netdev: String,
}

/// Whether `have` is the host route `route`. iproute2 prints a /32 destination without the
/// prefix (both `10.0.0.7` and `10.0.0.7/32` are the same route), and the setup compares what the
/// node reports against what it wrote.
fn same_host_route(have: &Route, route: &FabricRoute) -> bool {
    let (destination, prefix) = have
        .destination
        .split_once('/')
        .map_or((have.destination.as_str(), 32), |(destination, prefix)| {
            (destination, prefix.parse().unwrap_or(0))
        });
    let via = route.via.map(|via| via.to_string());
    destination.parse::<Ipv4Addr>() == Ok(route.to)
        && prefix == 32
        && have.netdev == route.netdev
        && have.via == via
}

/// Whether a route is one the setup wrote: a host route with the setup's metric. iproute2 prints a
/// /32 destination with or without the prefix.
fn is_fabric_host_route(route: &Route) -> bool {
    route.metric == FABRIC_ROUTE_METRIC
        && route
            .destination
            .strip_suffix("/32")
            .unwrap_or(&route.destination)
            .parse::<Ipv4Addr>()
            .is_ok()
}

/// Whether the node already carries every route.
fn routes_installed(facts: &NodeFacts, routes: &[FabricRoute]) -> bool {
    routes
        .iter()
        .all(|route| facts.routes.iter().any(|have| same_host_route(have, route)))
}

/// One cable, as the probe saw it: the port it leaves by and the peer address that answered on
/// it. That address is the next hop, and it need not be the peer's collective address: a Spark's
/// two ports are on two segments and each address lives on one of them, so the address that
/// answers on this cable is the peer's address on this cable. A packet whose destination is any of
/// the peer's addresses is delivered to the peer wherever it arrives, so this next hop reaches the
/// collective address too.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Cable {
    netdev: String,
    next_hop: Ipv4Addr,
}

/// The routes a ring needs, per rank. Every peer's collective address is pinned to a next hop that
/// is on the cable that reaches it: the peer itself when the two Sparks are cabled, and otherwise a
/// Spark cabled to both ends (the lowest-numbered one, so both directions of a pair use it), which
/// forwards the packet over its own cable.
fn fabric_routes(probes: &[ProbeResult], fabric: Option<&Fabric>) -> Vec<Vec<FabricRoute>> {
    let world = probes.len();
    let mut routes: Vec<Vec<FabricRoute>> = vec![Vec::new(); world];
    let Some(fabric) = fabric.filter(|fabric| fabric.shape == FabricShape::Ring) else {
        return routes;
    };
    // The map is only a cable map once every port answers ARP for its own address. With
    // arp_ignore=0 a peer answers for every one of its addresses on every cable, so "it answered
    // here" does not say which cable the address is on, and routing from that map writes next
    // hops that the same pass's arp_ignore=1 then makes unresolvable. The ARP settings go in
    // first; the next check measures the cables and these routes are written then.
    let measurable = probes.iter().all(|probe| match probe {
        Ok(facts) => {
            let rails = facts.rails();
            let netdevs: Vec<&str> = rails.iter().map(|rail| rail.netdev.as_str()).collect();
            netdevs.is_empty() || arp_ok(facts, &netdevs)
        }
        Err(_) => false,
    });
    if !measurable {
        return routes;
    }
    // Every fast address the nodes report, per rank. The transport does not bind a source
    // address, so a connection can leave with any of a Spark's addresses as its source; the peer
    // then has to route that address back, and therefore every address of every peer needs a
    // route, not only the collective.
    let addresses: Vec<Vec<Ipv4Addr>> = probes
        .iter()
        .map(|probe| {
            let mut list: Vec<Ipv4Addr> = probe
                .as_ref()
                .map(|facts| facts.rails().iter().map(|rail| rail.ip).collect())
                .unwrap_or_default();
            list.sort_unstable();
            list.dedup();
            list
        })
        .collect();
    // Netplan refuses a next hop outside the port's subnet, so only an answer from the fabric
    // subnet can be a next hop. The plan already puts every collective address on one subnet.
    let subnet = probes.iter().flatten().find_map(|facts| {
        let rails = facts.rails();
        let primary = facts.primary_rail(&rails)?;
        Some((
            Ipv4Addr::from(network(primary.ip, primary.prefix?)),
            primary.prefix?,
        ))
    });
    // Every cable, with the peer address that answered on it. The smallest port and address win,
    // so the two ends of one cable agree on the next hop.
    let mut cables: Vec<BTreeMap<usize, Cable>> = vec![BTreeMap::new(); world];
    for link in &fabric.links {
        if link.rank >= world || link.peer >= world || link.rank == link.peer {
            continue;
        }
        let Ok(address) = link.address.parse::<Ipv4Addr>() else {
            continue;
        };
        if let Some((fabric, prefix)) = subnet
            && Ipv4Addr::from(network(address, prefix)) != fabric
        {
            continue;
        }
        let cable = Cable {
            netdev: link.netdev.clone(),
            next_hop: address,
        };
        match cables[link.rank].get(&link.peer) {
            Some(have)
                if (have.netdev.as_str(), have.next_hop)
                    <= (cable.netdev.as_str(), cable.next_hop) => {}
            _ => {
                cables[link.rank].insert(link.peer, cable);
            }
        }
    }
    // The next hop that reaches a peer at all: its own address on the shared cable, or the Spark
    // cabled to both ends for a diagonal. Every address of that peer goes through it, because a
    // Spark accepts a packet addressed to any of its addresses on any of its ports.
    let reach = |rank: usize, peer: usize| -> Option<Cable> {
        if let Some(cable) = cables[rank].get(&peer) {
            return Some(cable.clone());
        }
        let relay = (0..world).find(|candidate| {
            *candidate != rank
                && *candidate != peer
                && cables[rank].contains_key(candidate)
                && cables[*candidate].contains_key(&peer)
        })?;
        cables[rank].get(&relay).cloned()
    };
    for (rank, mine) in routes.iter_mut().enumerate() {
        for (peer, targets) in addresses.iter().enumerate() {
            if peer == rank {
                continue;
            }
            let Some(hop) = reach(rank, peer) else {
                continue;
            };
            for target in targets {
                mine.push(FabricRoute {
                    to: *target,
                    // A next hop equal to the destination is the destination itself, which a
                    // gateway may not be: that cable carries the address, so a link route is right.
                    via: (hop.next_hop != *target).then_some(hop.next_hop),
                    netdev: hop.netdev.clone(),
                });
            }
        }
    }
    routes
}

/// The netplan and sysctl commands each node needs, or `None` when its ports are already right.
///
/// A port that DHCP addresses keeps DHCP. A port without an address gets one on the subnet the
/// other Sparks use (a free host number), or on 10.200.0.0/24 when none has an address yet. Both
/// lanes of a node land on one subnet, so the ARP settings come with any address change. On a ring
/// the same command pins every peer's fast addresses to a next hop on a cable that reaches that
/// peer (a diagonal through the Spark in between) and turns on forwarding, so a bare DAC ring
/// carries the rank-to-rank transport's connections to every rank in both directions.
fn network_fixes(
    probes: &[ProbeResult],
    fabric: Option<&Fabric>,
    name: &dyn Fn(usize) -> String,
) -> Vec<Option<(String, String)>> {
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
    let routes_by_rank = fabric_routes(probes, fabric);

    let mut fixes = Vec::new();
    for (rank, probe) in probes.iter().enumerate() {
        let Ok(facts) = probe else {
            fixes.push(None);
            continue;
        };
        let rails = facts.rails();
        let unaddressed = facts.unaddressed();
        let required = &routes_by_rank[rank];
        let routes_missing = !required.is_empty() && !routes_installed(facts, required);
        // A route the setup wrote that the measured fabric no longer needs: the netplan file is
        // rewritten without it, so a re-cable to a switch does not leave the ring's detours (or a
        // black hole once a next hop moves) behind.
        let obsolete_routes = facts
            .routes
            .iter()
            .filter(|route| is_fabric_host_route(route))
            .any(|have| !required.iter().any(|route| same_host_route(have, route)));
        // The ring's relay is on whenever any route goes through another Spark. The sysctl file
        // keeps the switch for a reboot even when only the ARP settings need rewriting.
        // The relay is on when a route goes through another Spark: either one this check derived,
        // or one the setup already wrote (a route with a next hop and the setup's metric). The
        // second case keeps the forwarding switch in the file when only the ARP settings need a
        // rewrite, even though an unhardened host's map derives no routes.
        let routing_forwarding = required.iter().any(|route| route.via.is_some())
            || facts
                .routes
                .iter()
                .any(|route| is_fabric_host_route(route) && route.via.is_some());
        let needs_forwarding = routing_forwarding && !facts.forward;
        // Loose reverse-path filtering is part of the relay: strict drops the forwarded packets.
        let port_names: Vec<&str> = rails.iter().map(|rail| rail.netdev.as_str()).collect();
        let strict_rp = routing_forwarding && facts.rp_strict(&port_names);
        let mut ports: Vec<(String, String)> = Vec::new();
        let mut needs_netplan = !unaddressed.is_empty() || routes_missing || obsolete_routes;
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
        let needs_sysctl = needs_arp || needs_forwarding || strict_rp;
        if !needs_netplan && !needs_sysctl {
            fixes.push(None);
            continue;
        }
        let mut body = String::from("set -e\n");
        if needs_netplan {
            let mut yaml = String::from("network:\n  version: 2\n  ethernets:\n");
            for (netdev, line) in &ports {
                yaml.push_str(&format!(
                    "    {netdev}:\n      mtu: {FABRIC_MTU}\n      {line}\n"
                ));
                let on_port: Vec<&FabricRoute> = required
                    .iter()
                    .filter(|route| route.netdev == *netdev)
                    .collect();
                if !on_port.is_empty() {
                    yaml.push_str("      routes:\n");
                    for route in on_port {
                        yaml.push_str(&format!("        - to: {}/32\n", route.to));
                        if let Some(via) = route.via {
                            yaml.push_str(&format!("          via: {via}\n"));
                        } else {
                            yaml.push_str("          scope: link\n");
                        }
                        yaml.push_str(&format!("          metric: {FABRIC_ROUTE_METRIC}\n"));
                    }
                }
            }
            body.push_str(&format!(
                "sudo tee {NETPLAN_FILE} > /dev/null <<'EOF'\n{yaml}EOF\n"
            ));
            body.push_str(&format!("sudo chmod 600 {NETPLAN_FILE}\n"));
        }
        if needs_sysctl {
            body.push_str(&format!("sudo tee {SYSCTL_FILE} > /dev/null <<'EOF'\n"));
            body.push_str("net.ipv4.conf.all.arp_ignore = 1\nnet.ipv4.conf.all.arp_announce = 2\n");
            if routing_forwarding {
                body.push_str("net.ipv4.ip_forward = 1\nnet.ipv4.conf.all.rp_filter = 2\n");
            }
            body.push_str("EOF\n");
            body.push_str(&format!("sudo sysctl -p {SYSCTL_FILE}\n"));
        }
        if needs_netplan {
            body.push_str("sudo netplan apply\n");
        }
        // The paste-into-a-terminal variant prompts for a password via sudo; the copy that
        // runs over SSH under `sudo -n bash -c` must not carry the sudo prefix, so it is dropped.
        let remote = body
            .lines()
            .map(|line| line.strip_prefix("sudo ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let text = format!(
            "# Run on {} in a terminal (it asks for your password).\n{body}",
            name(rank)
        );
        fixes.push(Some((text, remote)));
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
        // `sparkflash convert` names its output rank-{rank}-of-{world}.bin; V4.1 writes the
        // manifest after the rank file is complete, while GLM writes only the rank file.
        converted
            .join(format!("rank-{rank}-of-{}.bin", config.world_size()))
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

/// The read-only link probe: which port of this Spark answers each peer.
///
/// Arguments are repeatable: `ndev=<netdev>` for each fast port, `peer=<rank>:<address>`
/// for every fast address of every other Spark (one per rail: a cabled port is two PCIe halves,
/// so a two-cable ring presents four netdevs and four addresses per Spark), and
/// `ping=<rank>:<address>` for the collective address of every other Spark. Prints one
/// `link <rank> <netdev> ok|no` line per port per address, one `ping <rank> <netdev> ok|no`
/// line per port, and `arping missing` when the tool is absent (only the ping lines then carry
/// information and the shape stays unknown). Every attempt runs in parallel: a disconnected
/// four-port fabric would otherwise spend a second per unanswered ARP twice over.
pub const FABRIC_PROBE_SCRIPT: &str = r#"export LC_ALL=C
devs=()
peers=()
pings=()
for arg in "$@"; do
  case "$arg" in
    ndev=*) devs+=("${arg#ndev=}") ;;
    peer=*) peers+=("${arg#peer=}") ;;
    ping=*) pings+=("${arg#ping=}") ;;
  esac
done
have_arping=1
if ! command -v arping >/dev/null 2>&1; then
  echo "arping missing"
  have_arping=0
fi
out=$(mktemp) || exit 0
errs=$(mktemp) || exit 0
trap 'rm -f "$out" "$errs"' EXIT
if [ "$have_arping" = 1 ]; then
  for spec in "${peers[@]}"; do
    peer=${spec%%:*}
    addr=${spec#*:}
    for dev in "${devs[@]}"; do
      (
        if arping -q -c1 -w1 -I "$dev" "$addr" >/dev/null 2>>"$errs"; then
          echo "link $peer $dev $addr ok"
        else
          echo "link $peer $dev $addr no"
        fi
      ) >> "$out" &
    done
  done
fi
for spec in "${pings[@]}"; do
  peer=${spec%%:*}
  addr=${spec#*:}
  for dev in "${devs[@]}"; do
    (
      if ping -c1 -W1 -I "$dev" "$addr" >/dev/null 2>&1; then
        echo "ping $peer $dev ok"
      else
        echo "ping $peer $dev no"
      fi
    ) >> "$out" &
  done
done
wait
sort "$out"
# An arping that fails on every attempt *and says why* is a broken tool, not a missing cable
# (no CAP_NET_RAW, a refused socket, an unknown interface, a different implementation). Say so
# instead, and the check reports the wiring as unmeasured rather than blaming the cables.
if [ "$have_arping" = 1 ] && [ -s "$errs" ] && ! grep -q '^link .* ok$' "$out"; then
  echo "arping unusable"
fi
exit 0
"#;

/// The arguments for [`FABRIC_PROBE_SCRIPT`] for one rank: this rank's ports as `ndev=`
/// entries, every peer address as a `peer=` entry (a cabled port is two rails, so a peer may
/// appear twice per cable), and each peer's collective address as a `ping=` entry.
#[must_use]
pub fn fabric_probe_args(
    netdevs: &[String],
    peers: &[(usize, String)],
    pings: &[(usize, String)],
) -> Vec<String> {
    let mut args = Vec::with_capacity(netdevs.len() + peers.len() + pings.len());
    for netdev in netdevs {
        args.push(format!("ndev={netdev}"));
    }
    for (rank, address) in peers {
        args.push(format!("peer={rank}:{address}"));
    }
    for (rank, address) in pings {
        args.push(format!("ping={rank}:{address}"));
    }
    args
}

/// The remote command line that runs the link probe with the given args under bash.
///
/// It runs under `sudo -n`: `arping` needs a raw socket, and distributions install it without the
/// capability. The port setup already needs passwordless sudo, so this is the same requirement.
#[must_use]
pub fn fabric_probe_remote_command(args: &[String]) -> String {
    let mut line = format!(
        "sudo -n bash -c {} sparkflash-fabric",
        shell_quote(FABRIC_PROBE_SCRIPT)
    );
    for arg in args {
        line.push(' ');
        line.push_str(&shell_quote(arg));
    }
    line
}

/// One node's answer to [`FABRIC_PROBE_SCRIPT`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FabricProbe {
    /// The links that answered, in the order the probe printed them.
    pub links: Vec<FabricLink>,
    /// The ranks this node pinged, sorted.
    pub reached: Vec<usize>,
    /// Whether `arping` was available: without it, no link could be measured.
    pub measured: bool,
}

/// Reads one node's link probe output: the links that answered, the peers it reached, and whether
/// arping was available. The probe prints one `link` line per port per address, so a peer answers
/// once for each of the peer's addresses and each is recorded (the setup routes the addresses);
/// a repeated line is dropped. Unknown or malformed lines are skipped.
#[must_use]
pub fn parse_fabric(rank: usize, text: &str) -> FabricProbe {
    let mut links: Vec<FabricLink> = Vec::new();
    let mut reached = Vec::new();
    let mut links_measured = true;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next()) {
            (Some("arping"), Some("missing" | "unusable")) => links_measured = false,
            (Some("link"), Some(peer)) => {
                let (Some(netdev), Some(address), Some(state), Ok(peer)) = (
                    fields.next(),
                    fields.next(),
                    fields.next(),
                    peer.parse::<usize>(),
                ) else {
                    continue;
                };
                if state == "ok"
                    && !links.iter().any(|link| {
                        link.netdev == netdev && link.peer == peer && link.address == address
                    })
                {
                    links.push(FabricLink {
                        rank,
                        netdev: netdev.to_owned(),
                        peer,
                        address: address.to_owned(),
                    });
                }
            }
            (Some("ping"), Some(peer)) => {
                // One line per port; a peer counts as reached when any port reached it.
                let _port = fields.next();
                if let (Some("ok"), Ok(peer)) = (fields.next(), peer.parse::<usize>()) {
                    reached.push(peer);
                }
            }
            _ => {}
        }
    }
    reached.sort_unstable();
    reached.dedup();
    FabricProbe {
        links,
        reached,
        measured: links_measured,
    }
}

/// Assembles the fabric report from every rank's probe output, in rank order. A rank whose
/// probe did not run contributes neither links nor reachability, and marks the shape unmeasured.
#[must_use]
pub fn fabric_from_probes(results: Vec<Option<FabricProbe>>, world: usize) -> Fabric {
    let mut links = Vec::new();
    let mut reachable = Vec::with_capacity(results.len());
    let mut links_measured = true;
    for result in results {
        match result {
            Some(probe) => {
                links.extend(probe.links);
                reachable.push(Some(probe.reached));
                links_measured &= probe.measured;
            }
            None => {
                reachable.push(None);
                links_measured = false;
            }
        }
    }
    links.sort_by(|a, b| (a.rank, &a.netdev, a.peer).cmp(&(b.rank, &b.netdev, b.peer)));
    let shape = shape_of(&links, world);
    Fabric {
        links,
        reachable,
        shape,
        links_measured,
    }
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
gid rocep1s0f1 2 IB/RoCE_v1 0000:0000:0000:0000:0000:ffff:c633:64{a:02x} enp1s0f1np1
gid rocep1s0f1 3 RoCE_v2 0000:0000:0000:0000:0000:ffff:c633:64{a:02x} enp1s0f1np1
port roceP2p1s0f1 ACTIVE 200
gid roceP2p1s0f1 3 RoCE_v2 0000:0000:0000:0000:0000:ffff:c633:64{b:02x} enP2p1s0f1np1
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
        assert_eq!(rails[0].ip, Ipv4Addr::new(198, 51, 100, 11));
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
            [
                "198.51.100.21",
                "198.51.100.22",
                "198.51.100.23",
                "198.51.100.24"
            ]
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
    fn a_glm_rank_file_is_complete_without_a_manifest() {
        let probes: Vec<ProbeResult> = cluster()
            .into_iter()
            .map(|probe| {
                probe.map(|mut facts| {
                    facts.files.remove("manifest");
                    facts
                })
            })
            .collect();
        let mut config = SparkConfig::default();
        config.model.weights = crate::config::GLM_WEIGHTS_ID.to_owned();
        config.model.id = crate::config::GLM_MODEL_ID.to_owned();
        config.model.checkpoint = "/data/GLM-5.3-Flash".to_owned();
        config.engine.kind = crate::config::GLM_ENGINE_ID.to_owned();
        let report = analyze(&config, &probes, true, 1);
        assert!(report.findings.is_empty(), "{:?}", report.findings);
        for node in &report.nodes {
            assert_eq!(node.converted, Check::Ready, "rank {}", node.rank);
        }

        // V4.1 uses the manifest as its completion marker, so the same file set is partial.
        let report = analyze(&SparkConfig::default(), &probes, true, 1);
        assert_eq!(report.nodes[0].converted, Check::Partial);
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
            .replace("ffff:c633:64", "ffff:0a09:01");
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
        let rank_file = config
            .converted_path()
            .join("rank-2-of-4.bin")
            .display()
            .to_string();
        assert_eq!(args[4], rank_file);
        let line = remote_command(&args);
        assert!(line.starts_with("bash -c '"), "{line}");
        assert!(
            line.contains(" sparkflash-inspect '/srv/spark flash/target/release/sparkflash' "),
            "{line}"
        );
    }

    fn fabric_link(rank: usize, netdev: &str, peer: usize) -> FabricLink {
        FabricLink {
            rank,
            netdev: netdev.to_owned(),
            peer,
            address: format!("198.51.100.{}", 100 + peer),
        }
    }

    #[test]
    fn the_fabric_shape_is_read_from_the_measured_links() {
        let mut ring = Vec::new();
        for rank in 0..4usize {
            ring.push(fabric_link(rank, "port0", (rank + 1) % 4));
            ring.push(fabric_link(rank, "port1", (rank + 3) % 4));
        }
        assert_eq!(shape_of(&ring, 4), FabricShape::Ring);

        let mut switched = Vec::new();
        for rank in 0..4usize {
            for port in ["port0", "port1"] {
                for peer in 0..4usize {
                    if peer != rank {
                        switched.push(fabric_link(rank, port, peer));
                    }
                }
            }
        }
        assert_eq!(shape_of(&switched, 4), FabricShape::Switched);

        let pairs = vec![
            fabric_link(0, "port0", 1),
            fabric_link(0, "port1", 1),
            fabric_link(1, "port0", 0),
            fabric_link(1, "port1", 0),
            fabric_link(2, "port0", 3),
            fabric_link(2, "port1", 3),
            fabric_link(3, "port0", 2),
            fabric_link(3, "port1", 2),
        ];
        assert_eq!(shape_of(&pairs, 4), FabricShape::Pairs);
        assert_eq!(shape_of(&[], 4), FabricShape::Unknown);

        // A rank that reaches exactly one peer is the two-node-pair wiring, however many
        // netdevs answered for that peer.
        let one_peer_each = vec![
            fabric_link(0, "port0", 1),
            fabric_link(1, "port0", 0),
            fabric_link(2, "port0", 3),
            fabric_link(3, "port0", 2),
        ];
        assert_eq!(shape_of(&one_peer_each, 4), FabricShape::Pairs);

        // The real hardware: one cabled ConnectX-7 port is two PCIe halves, so a Spark cabled to
        // two neighbours answers on four netdevs, two per neighbour.
        let mut halves = Vec::new();
        for rank in 0..4usize {
            for (dev, peer) in [
                ("enp1s0f0np0", (rank + 1) % 4),
                ("enp1s0f1np1", (rank + 1) % 4),
                ("enP2p1s0f0np0", (rank + 3) % 4),
                ("enP2p1s0f1np1", (rank + 3) % 4),
            ] {
                halves.push(fabric_link(rank, dev, peer));
            }
        }
        assert_eq!(shape_of(&halves, 4), FabricShape::Ring);

        // One silent cable on one Spark: that rank reaches a single peer while the others reach
        // two, so the wiring is neither a ring nor two pairs.
        let mut missing_cable = halves.clone();
        missing_cable.retain(|link| !(link.rank == 2 && link.peer == 1));
        assert_eq!(shape_of(&missing_cable, 4), FabricShape::Unknown);

        // A netdev that reaches two peers is not a DAC segment on any wiring.
        let mut two_peers = ring_links();
        two_peers.push(fabric_link(0, "port0", 2));
        assert_eq!(shape_of(&two_peers, 4), FabricShape::Unknown);
    }

    #[test]
    fn the_probe_addresses_cover_both_rails_and_put_the_primary_first() {
        // The link probe must ARP every address a Spark answers on: with arp_ignore=1 an address
        // answers only on the port that owns it, so probing one address per rank would leave one
        // end of a cable unanswered and make a correct ring look like two-point pairs.
        let probes = cluster();
        let addresses = probe_addresses(&SparkConfig::default(), &probes).expect("addresses");
        assert_eq!(addresses.len(), 4);
        for (rank, mine) in addresses.iter().enumerate() {
            assert!(mine.len() >= 2, "rank {rank} probes {mine:?}");
            assert_eq!(
                mine[0],
                planned_peers(&SparkConfig::default(), &probes).expect("peers")[rank]
            );
            let mut unique = mine.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), mine.len(), "rank {rank} repeats an address");
        }
    }

    #[cfg(unix)]
    fn write_stub(dir: &std::path::Path, name: &str, body: &str) {
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(dir).expect("stub dir");
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).expect("stub");
        writeln!(file, "#!/bin/bash").expect("stub header");
        write!(file, "{body}").expect("stub body");
        drop(file);
        let mut perms = std::fs::metadata(&path).expect("stub meta").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("stub chmod");
    }

    /// Runs the real probe script under bash with the given stubs first on PATH.
    #[cfg(unix)]
    fn run_probe(dir: &std::path::Path, args: &[&str]) -> String {
        let bin = dir.join("bin");
        std::fs::create_dir_all(&bin).expect("bin");
        let path = format!("{}:/usr/local/bin:/usr/bin:/bin", bin.display());
        let mut command = std::process::Command::new("bash");
        command
            .arg("-c")
            .arg(FABRIC_PROBE_SCRIPT)
            .arg("sparkflash-fabric")
            .args(args)
            .env("PATH", path);
        let output = command.output().expect("bash runs the probe");
        assert!(
            output.status.success(),
            "probe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    #[cfg(unix)]
    #[test]
    fn the_probe_script_measures_links_and_pings_without_a_self_check() {
        // The script itself, not just the parser: a stub arping answers only one peer address and
        // a stub ping answers both. The output must carry one link line per port per address and
        // one ping line per port, and must never claim the tool is unusable merely because a host
        // does not answer its own ARP request (a normal Linux host does not).
        let dir = tempfile::tempdir().expect("tempdir");
        write_stub(
            &dir.path().join("bin"),
            "arping",
            "for a in \"$@\"; do last=$a; done\n[ \"$last\" = 10.0.0.3 ]\n",
        );
        write_stub(
            &dir.path().join("bin"),
            "ping",
            "for a in \"$@\"; do last=$a; done\ncase \"$last\" in 10.0.0.3|10.0.0.5) exit 0;; esac\nexit 1\n",
        );
        let text = run_probe(
            dir.path(),
            &[
                "ndev=enp1s0f0np0",
                "ndev=enP2p1s0f0np0",
                "peer=1:10.0.0.3",
                "peer=1:10.1.0.3",
                "peer=2:10.0.0.5",
                "peer=2:10.1.0.5",
                "ping=1:10.0.0.3",
                "ping=2:10.0.0.5",
            ],
        );
        assert!(
            text.contains("link 1 enp1s0f0np0 10.0.0.3 ok"),
            "the answering address is part of the link: {text}"
        );
        assert!(text.contains("link 1 enP2p1s0f0np0 10.0.0.3 ok"), "{text}");
        assert!(
            text.contains("link 2 enp1s0f0np0 10.0.0.5 no"),
            "an address that did not answer is a no, not a missing line: {text}"
        );
        assert!(text.contains("ping 2 enp1s0f0np0 ok"), "{text}");
        assert!(!text.contains("arping missing"), "{text}");
        assert!(
            !text.contains("arping unusable"),
            "a host that does not answer its own ARP is normal: {text}"
        );
        // One link line per (address x port): 4 addresses x 2 ports.
        assert_eq!(
            text.lines()
                .filter(|line| line.starts_with("link "))
                .count(),
            8,
            "{text}"
        );
        // And one ping line per (collective address x port).
        assert_eq!(
            text.lines()
                .filter(|line| line.starts_with("ping "))
                .count(),
            4,
            "{text}"
        );

        // A fabric where nothing answers: every link is a no, and the parser reports it as an
        // unmeasured-looking empty set rather than inventing a shape.
        let dir = tempfile::tempdir().expect("tempdir");
        write_stub(&dir.path().join("bin"), "arping", "exit 1\n");
        write_stub(&dir.path().join("bin"), "ping", "exit 1\n");
        let text = run_probe(
            dir.path(),
            &["ndev=enp1s0f0np0", "peer=1:10.0.0.3", "ping=1:10.0.0.3"],
        );
        let probe = parse_fabric(0, &text);
        assert!(probe.links.is_empty(), "{text}");
        assert!(probe.reached.is_empty(), "{text}");
        assert!(probe.measured, "arping ran, it just got no answer: {text}");
    }

    #[cfg(unix)]
    #[test]
    fn an_arping_that_cannot_open_a_raw_socket_is_reported_as_unmeasured() {
        // Without CAP_NET_RAW arping is present but every attempt fails with a permission error.
        // That is not "nothing is plugged in": the script has to say the tool is unusable so the
        // check reports the wiring as unmeasured instead of blaming the cables.
        let dir = tempfile::tempdir().expect("tempdir");
        write_stub(
            &dir.path().join("bin"),
            "arping",
            "echo 'arping: socket: Operation not permitted' >&2\nexit 2\n",
        );
        write_stub(&dir.path().join("bin"), "ping", "exit 1\n");
        let text = run_probe(
            dir.path(),
            &["ndev=dev0", "peer=1:10.0.0.3", "ping=1:10.0.0.3"],
        );
        assert!(text.contains("arping unusable"), "{text}");
        let probe = parse_fabric(0, &text);
        assert!(
            !probe.measured,
            "the wiring is unmeasured, not absent: {text}"
        );
        assert!(probe.links.is_empty(), "{text}");
        assert!(probe.reached.is_empty(), "{text}");

        // A different failure (an unknown interface, say) is the same story: the tool rather than
        // the cables.
        let dir = tempfile::tempdir().expect("tempdir");
        write_stub(
            &dir.path().join("bin"),
            "arping",
            "echo 'arping: unknown interface: dev0' >&2\nexit 2\n",
        );
        write_stub(&dir.path().join("bin"), "ping", "exit 1\n");
        let text = run_probe(dir.path(), &["ndev=dev0", "peer=1:10.0.0.3"]);
        assert!(text.contains("arping unusable"), "{text}");
    }

    #[test]
    fn the_probe_arguments_name_ports_peers_and_collective_addresses() {
        let args = fabric_probe_args(
            &["enp1s0f0np0".to_owned(), "enp1s0f1np1".to_owned()],
            &[(1, "10.0.0.3".to_owned()), (1, "10.1.0.3".to_owned())],
            &[(1, "10.0.0.3".to_owned())],
        );
        assert_eq!(
            args,
            vec![
                "ndev=enp1s0f0np0",
                "ndev=enp1s0f1np1",
                "peer=1:10.0.0.3",
                "peer=1:10.1.0.3",
                "ping=1:10.0.0.3",
            ]
        );
    }

    #[test]
    fn parse_fabric_reads_links_pings_and_a_missing_arping() {
        let probe = parse_fabric(
            2,
            "arping missing\nlink 1 enp1s0f1np1 10.0.0.3 ok\nlink 3 enP2p1s0f1np1 10.0.0.5 no\n             ping 1 enp1s0f1np1 ok\nping 3 enP2p1s0f1np1 no\nnoise\n",
        );
        assert_eq!(
            probe.links,
            vec![FabricLink {
                rank: 2,
                netdev: "enp1s0f1np1".to_owned(),
                peer: 1,
                address: "10.0.0.3".to_owned(),
            }]
        );
        assert_eq!(probe.reached, vec![1]);
        assert!(!probe.measured);

        // One line per port: a peer counts as reached when any port reached it, once.
        let probe = parse_fabric(
            0,
            "link 1 a 10.0.0.3 ok\nlink 1 b 10.1.0.3 no\nping 1 a ok\nping 1 b ok\nping 2 a no\nping 2 b no\n",
        );
        assert_eq!(
            probe.links,
            vec![FabricLink {
                rank: 0,
                netdev: "a".to_owned(),
                peer: 1,
                address: "10.0.0.3".to_owned(),
            }]
        );
        assert!(probe.measured);
        assert_eq!(probe.reached, vec![1]);

        // arping that ran but answered nothing (no CAP_NET_RAW) is as unmeasured as a missing one.
        let probe = parse_fabric(0, "arping unusable\nlink 1 a no\nping 1 a ok\n");
        assert!(!probe.measured);
        assert_eq!(probe.reached, vec![1]);
    }

    #[test]
    fn a_failed_probe_marks_the_fabric_unmeasured() {
        let one = |rank: usize| {
            Some(FabricProbe {
                links: vec![fabric_link(rank, "a", (rank + 1) % 4)],
                reached: vec![(rank + 1) % 4],
                measured: true,
            })
        };
        let fabric = fabric_from_probes(vec![one(0), one(1), one(2), None], 4);
        assert!(!fabric.links_measured);
        assert_eq!(fabric.reachable[3], None);
        assert_eq!(fabric.reachable[0], Some(vec![1]));
        assert_eq!(fabric.shape, FabricShape::Unknown);
    }

    fn ring_links() -> Vec<FabricLink> {
        let mut links = Vec::new();
        for rank in 0..4usize {
            links.push(fabric_link(rank, "port0", (rank + 1) % 4));
            links.push(fabric_link(rank, "port1", (rank + 3) % 4));
        }
        links
    }

    fn warnings(report: &SetupReport) -> Vec<&str> {
        report
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .map(|f| f.message.as_str())
            .collect()
    }

    fn errors(report: &SetupReport) -> Vec<&str> {
        report
            .findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .map(|f| f.message.as_str())
            .collect()
    }

    #[test]
    fn a_ring_that_cannot_ping_its_diagonals_blocks_the_launch() {
        // The ring reaches its two neighbours over the DACs and the opposite rank through the
        // hairpin on the Spark between the two cables. The launch opens a connection to every
        // rank before it starts, so a silent diagonal is an error that names the hairpin, not a
        // warning the guide can walk past.
        let probes = cluster();
        let ring = Fabric {
            links: ring_links(),
            reachable: vec![
                Some(vec![1, 3]),
                Some(vec![0, 2]),
                Some(vec![1, 3]),
                Some(vec![0, 2]),
            ],
            shape: FabricShape::Ring,
            links_measured: true,
        };
        let report = analyze_fabric(
            &SparkConfig::default(),
            &probes,
            Some(ring.clone()),
            true,
            1,
        );
        // The default schedule is switched: the check advises switchless, and the silent
        // diagonals stop the guide.
        assert!(
            warnings(&report)
                .iter()
                .any(|m| m.contains("choose switchless")),
            "{:?}",
            report.findings
        );
        let stopped = errors(&report);
        assert!(
            stopped.iter().any(
                |m| m.contains("spark0 cannot reach spark2") && m.contains("through the Spark")
            ),
            "{stopped:?}"
        );
        assert!(!report.network_ok(), "{:?}", report.findings);

        // A ring whose diagonals do answer is a working fabric: saving switchless leaves no
        // finding at all.
        let forwarding = Fabric {
            reachable: vec![
                Some(vec![1, 2, 3]),
                Some(vec![0, 2, 3]),
                Some(vec![0, 1, 3]),
                Some(vec![0, 1, 2]),
            ],
            ..ring
        };
        let mut config = SparkConfig::default();
        config.cluster.fabric_schedule = "switchless".to_owned();
        let report = analyze_fabric(&config, &probes, Some(forwarding), true, 1);
        assert!(errors(&report).is_empty(), "{:?}", errors(&report));
        assert!(
            !warnings(&report)
                .iter()
                .any(|m| m.contains("choose switchless") || m.contains("cannot reach")),
            "{:?}",
            report.findings
        );
    }

    #[test]
    fn a_linked_neighbour_that_does_not_answer_is_an_error() {
        // Rank 0 is cabled to rank 3 (port1 -> 3) but only pings 1: the cable works, the address
        // does not, so the guide must stop for it. The 0-2 diagonal is not cabled at all, so it
        // stops the guide too, with the message naming the Spark in between.
        let probes = cluster();
        let ring = Fabric {
            links: ring_links(),
            reachable: vec![
                Some(vec![1]),
                Some(vec![0, 2]),
                Some(vec![1, 3]),
                Some(vec![0, 2]),
            ],
            shape: FabricShape::Ring,
            links_measured: true,
        };
        let report = analyze_fabric(&SparkConfig::default(), &probes, Some(ring), true, 1);
        let errors = errors(&report);
        assert!(
            errors
                .iter()
                .any(|m| m.contains("spark0 cannot reach spark3")),
            "{errors:?}"
        );
        assert!(
            errors.iter().any(
                |m| m.contains("spark0 cannot reach spark2") && m.contains("through the Spark")
            ),
            "{errors:?}"
        );
    }

    #[test]
    fn an_unchecked_node_that_does_not_answer_does_not_claim_a_missing_cable() {
        // arping missing (or unable to open its socket) leaves the links unmeasured. The pings
        // still say which Sparks answer, and the message must not assert a cable is absent when
        // nothing measured the cables.
        let probes = cluster();
        let unmeasured = Fabric {
            links: Vec::new(),
            reachable: vec![Some(vec![1]), Some(vec![0]), Some(vec![3]), Some(vec![2])],
            shape: FabricShape::Unknown,
            links_measured: false,
        };
        let report = analyze_fabric(&SparkConfig::default(), &probes, Some(unmeasured), true, 1);
        let errors = errors(&report);
        let message = errors
            .iter()
            .find(|m| m.contains("spark0 cannot reach spark3"))
            .expect("the ping failure is reported");
        assert!(message.contains("could not be measured"), "{message}");
        assert!(!message.contains("no cable was measured"), "{message}");
        // The report itself says the wiring was not measured, not only the browser picker.
        assert!(
            warnings(&report)
                .iter()
                .any(|m| m.contains("wiring could not be measured")),
            "{:?}",
            report.findings
        );
    }

    #[test]
    fn a_peer_address_that_answers_twice_is_one_link() {
        let probe = parse_fabric(
            2,
            "arping missing
link 1 enp1s0f1np1 198.51.100.11 ok
link 1 enp1s0f1np1 198.51.100.11 ok
link 1 enp1s0f1np1 198.51.100.21 ok
link 1 enP2p1s0f1np1 198.51.100.11 ok
link 3 enp1s0f1np1 198.51.100.13 ok
",
        );
        assert!(!probe.measured, "the arping line is honoured");
        assert_eq!(
            probe.links,
            vec![
                FabricLink {
                    rank: 2,
                    netdev: "enp1s0f1np1".to_owned(),
                    peer: 1,
                    address: "198.51.100.11".to_owned(),
                },
                FabricLink {
                    rank: 2,
                    netdev: "enp1s0f1np1".to_owned(),
                    peer: 1,
                    address: "198.51.100.21".to_owned(),
                },
                FabricLink {
                    rank: 2,
                    netdev: "enP2p1s0f1np1".to_owned(),
                    peer: 1,
                    address: "198.51.100.11".to_owned(),
                },
                FabricLink {
                    rank: 2,
                    netdev: "enp1s0f1np1".to_owned(),
                    peer: 3,
                    address: "198.51.100.13".to_owned(),
                },
            ]
        );
    }

    /// A four-Spark ring measured from `cluster()` the way the hardware presents it: each Spark's
    /// two ports are on two segments, an address lives on one of them, and the primary (the second
    /// rail, lowest route metric) is on the port that faces the next rank. So the cable to the next
    /// rank carries the next rank's *other* address, and the cable to the previous rank carries its
    /// collective address.
    fn ring_fabric() -> Fabric {
        let mut links = Vec::new();
        for rank in 0..4usize {
            let next = (rank + 1) % 4;
            let previous = (rank + 3) % 4;
            links.push(FabricLink {
                rank,
                netdev: "enP2p1s0f1np1".to_owned(),
                peer: next,
                address: format!("198.51.100.{}", 11 + next),
            });
            links.push(FabricLink {
                rank,
                netdev: "enp1s0f1np1".to_owned(),
                peer: previous,
                address: format!("198.51.100.{}", 21 + previous),
            });
        }
        Fabric {
            links,
            reachable: vec![Some(vec![1]), Some(vec![2]), Some(vec![3]), Some(vec![0])],
            shape: FabricShape::Ring,
            links_measured: true,
        }
    }

    #[test]
    fn a_host_route_matches_with_and_without_the_prefix_length() {
        // iproute2 prints a /32 destination as the bare address, so the convergence check has to
        // accept both forms (and still reject a route with the wrong next hop or port).
        let target = FabricRoute {
            to: Ipv4Addr::new(198, 51, 100, 22),
            via: Some(Ipv4Addr::new(198, 51, 100, 12)),
            netdev: "enp1s0f1np1".to_owned(),
        };
        let route = |destination: &str, netdev: &str, via: Option<&str>| Route {
            destination: destination.to_owned(),
            netdev: netdev.to_owned(),
            via: via.map(str::to_owned),
            metric: 0,
            dhcp: false,
        };
        assert!(same_host_route(
            &route("198.51.100.22", "enp1s0f1np1", Some("198.51.100.12")),
            &target
        ));
        assert!(same_host_route(
            &route("198.51.100.22/32", "enp1s0f1np1", Some("198.51.100.12")),
            &target
        ));
        assert!(!same_host_route(
            &route("198.51.100.22", "enP2p1s0f1np1", Some("198.51.100.12")),
            &target
        ));
        assert!(!same_host_route(
            &route("198.51.100.22", "enp1s0f1np1", Some("198.51.100.13")),
            &target
        ));
        assert!(!same_host_route(
            &route("198.51.100.23", "enp1s0f1np1", Some("198.51.100.12")),
            &target
        ));
        // A link route is not the same as a route through a next hop.
        assert!(!same_host_route(
            &route("198.51.100.22", "enp1s0f1np1", None),
            &target
        ));
    }

    #[test]
    fn a_ring_setup_pins_every_peer_route_and_enables_forwarding() {
        // A bare ring has no cable between opposite Sparks, and a neighbour's collective address
        // may sit on its other cable. The port setup has to pin every peer's collective address to
        // a next hop on the cable that reaches it (the peer's address on that cable, or the Spark
        // in between) and turn on forwarding, or the transport cannot connect to every rank.
        let probes = cluster();
        let report = analyze_fabric(
            &SparkConfig::default(),
            &probes,
            Some(ring_fabric()),
            true,
            1,
        );
        for node in &report.nodes {
            let fix = node.network_fix.as_deref().expect("a port fix");
            assert!(fix.contains("routes:"), "{fix}");
            assert!(fix.contains("scope: link"), "{fix}");
            assert!(fix.contains("via: 198.51.100."), "{fix}");
            assert!(fix.contains("net.ipv4.ip_forward = 1"), "{fix}");
            assert!(fix.contains("net.ipv4.conf.all.rp_filter = 2"), "{fix}");
            let remote = node.network_fix_remote.as_deref().expect("remote");
            assert_eq!(remote.lines().next(), Some("set -e"), "{remote}");
            assert!(
                !remote.lines().any(|line| line.starts_with("sudo ")),
                "{remote}"
            );
            // Every peer gets a route: the other three collective addresses must all be there.
            for peer in 0..4usize {
                if peer == node.rank {
                    continue;
                }
                let destination = format!("to: 198.51.100.{}/32", 21 + peer);
                assert!(
                    fix.contains(&destination),
                    "rank {} lacks {destination}: {fix}",
                    node.rank
                );
            }
        }
        // Rank 0: the cable to rank 1 carries rank 1's other address (12), so the route to rank 1's
        // collective address (22) goes via it; rank 3's collective address (24) is on that cable
        // itself; rank 2 is not cabled to rank 0 and is reached through rank 1.
        let node = report.nodes[0].network_fix.as_deref().expect("fix");
        assert!(
            node.contains("to: 198.51.100.22/32") && node.contains("via: 198.51.100.12"),
            "{node}"
        );
        assert!(
            node.contains("to: 198.51.100.24/32") && node.contains("scope: link"),
            "{node}"
        );
        assert!(
            node.contains("to: 198.51.100.23/32") && node.contains("via: 198.51.100.12"),
            "{node}"
        );
    }

    #[test]
    fn strict_reverse_path_filtering_is_named_as_the_cause_on_a_ring() {
        // The ring's routes send traffic out the half of a port that does not own the source
        // address; strict rp_filter drops it, so the finding says that instead of blaming the
        // cable.
        let mut probes = cluster();
        for facts in probes.iter_mut().flatten() {
            facts.rp_filter = 1;
        }
        let report = analyze_fabric(
            &SparkConfig::default(),
            &probes,
            Some(ring_fabric()),
            true,
            1,
        );
        let stopped = errors(&report);
        assert!(
            stopped
                .iter()
                .any(|m| m.contains("strict reverse-path filtering") && m.contains("rp_filter=2")),
            "{stopped:?}"
        );
        let fix = report.nodes[0]
            .network_fix
            .as_deref()
            .expect("a port fix is offered");
        assert!(fix.contains("net.ipv4.conf.all.rp_filter = 2"), "{fix}");

        // The kernel takes the larger of the port's own value and `all`, so a per-port 1 with
        // `all=0` is strict too.
        let mut probes = cluster();
        for facts in probes.iter_mut().flatten() {
            facts.rp_filter = 0;
            for link in &mut facts.links {
                link.rp_filter = 1;
            }
        }
        let report = analyze_fabric(
            &SparkConfig::default(),
            &probes,
            Some(ring_fabric()),
            true,
            1,
        );
        assert!(
            errors(&report)
                .iter()
                .any(|m| m.contains("strict reverse-path filtering")),
            "{:?}",
            report.findings
        );
    }

    #[test]
    fn routes_wait_for_the_arp_settings_that_make_the_cable_map_unambiguous() {
        // With arp_ignore=0 every port of a peer answers for every address, so an answer does not
        // say which cable the address is on. Routing from that map writes next hops the same
        // pass's arp_ignore=1 then makes unresolvable; the ARP settings go in first.
        let mut probes = cluster();
        for facts in probes.iter_mut().flatten() {
            facts.arp_all = (0, 0);
        }
        let fabric = ring_fabric();
        assert!(
            fabric_routes(&probes, Some(&fabric))
                .iter()
                .all(Vec::is_empty),
            "an ARP-fluxy map must not be routed"
        );
        let report = analyze_fabric(
            &SparkConfig::default(),
            &probes,
            Some(fabric.clone()),
            true,
            1,
        );
        let fix = report.nodes[0].network_fix.as_deref().expect("an ARP fix");
        assert!(fix.contains("arp_ignore"), "{fix}");
        assert!(!fix.contains("routes:"), "{fix}");

        // The next check, with the settings in place, measures the cables and routes them.
        for facts in probes.iter_mut().flatten() {
            facts.arp_all = (1, 2);
        }
        let routes = fabric_routes(&probes, Some(&fabric));
        assert!(routes.iter().all(|list| !list.is_empty()), "{routes:?}");
    }

    #[test]
    fn stale_ring_routes_are_removed_when_the_fabric_is_no_longer_a_ring() {
        // A ring writes /32 routes; re-cabled to a switch they are obsolete, and the check has to
        // rewrite the netplan without them instead of certifying the fabric.
        let mut probes = cluster();
        let mut links = Vec::new();
        for rank in 0..4usize {
            for peer in 0..4usize {
                if peer != rank {
                    links.push(FabricLink {
                        rank,
                        netdev: "enP2p1s0f1np1".to_owned(),
                        peer,
                        address: format!("198.51.100.{}", 21 + peer),
                    });
                }
            }
        }
        let fabric = Fabric {
            links,
            reachable: vec![
                Some(vec![1, 2, 3]),
                Some(vec![0, 2, 3]),
                Some(vec![0, 1, 3]),
                Some(vec![0, 1, 2]),
            ],
            shape: FabricShape::Switched,
            links_measured: true,
        };
        for (rank, probe) in probes.iter_mut().enumerate() {
            let facts = probe.as_mut().expect("facts");
            let peer = (rank + 1) % 4;
            facts.routes.push(Route {
                destination: format!("198.51.100.{}", 21 + peer),
                netdev: "enP2p1s0f1np1".to_owned(),
                via: Some(format!("198.51.100.{}", 21 + ((rank + 3) % 4))),
                metric: FABRIC_ROUTE_METRIC,
                dhcp: false,
            });
        }
        let report = analyze_fabric(
            &SparkConfig::default(),
            &probes,
            Some(fabric.clone()),
            true,
            1,
        );
        for node in &report.nodes {
            let fix = node.network_fix.as_deref().expect("a cleanup fix");
            assert!(fix.contains("netplan apply"), "{fix}");
            assert!(
                !fix.contains("routes:"),
                "the stale routes are rewritten away: {fix}"
            );
        }
        // Once they are gone the check converges.
        for facts in probes.iter_mut().flatten() {
            facts.routes.retain(|route| !is_fabric_host_route(route));
        }
        let report = analyze_fabric(&SparkConfig::default(), &probes, Some(fabric), true, 1);
        assert!(
            report.nodes.iter().all(|node| node.network_fix.is_none()),
            "{:?}",
            report
                .nodes
                .iter()
                .map(|node| (&node.network_fix, node.rank))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_sysctl_rewrite_keeps_the_rings_forwarding() {
        // An ARP-only pass rewrites the sysctl file. On a ring that file also has to keep
        // ip_forward, or a reboot drops the relay the diagonals go through.
        let mut probes = cluster();
        let fabric = ring_fabric();
        let routes = fabric_routes(&probes, Some(&fabric));
        for (rank, probe) in probes.iter_mut().enumerate() {
            let facts = probe.as_mut().expect("facts");
            facts.forward = true;
            facts.arp_all = (0, 0);
            for link in &mut facts.links {
                link.arp_ignore = 0;
                link.arp_announce = 0;
            }
            for route in &routes[rank] {
                facts.routes.push(Route {
                    destination: route.to.to_string(),
                    netdev: route.netdev.clone(),
                    via: route.via.map(|via| via.to_string()),
                    metric: FABRIC_ROUTE_METRIC,
                    dhcp: false,
                });
            }
        }
        let report = analyze_fabric(&SparkConfig::default(), &probes, Some(fabric), true, 1);
        let fix = report.nodes[0]
            .network_fix
            .as_deref()
            .expect("an ARP fix is offered");
        assert!(fix.contains("arp_ignore"), "{fix}");
        assert!(fix.contains("net.ipv4.ip_forward = 1"), "{fix}");
    }

    #[test]
    fn a_ring_whose_routes_and_forwarding_are_installed_needs_no_further_fix() {
        // The setup has to converge: once the routes and forwarding are in place the check must
        // stop offering the port command, or the guide would ask for it forever.
        let mut probes = cluster();
        let fabric = ring_fabric();
        let routes = fabric_routes(&probes, Some(&fabric));
        for (rank, probe) in probes.iter_mut().enumerate() {
            let facts = probe.as_mut().expect("facts");
            facts.forward = true;
            for route in &routes[rank] {
                // iproute2 prints a host route without the /32; that is what the probe parses.
                facts.routes.push(Route {
                    destination: route.to.to_string(),
                    netdev: route.netdev.clone(),
                    via: route.via.map(|via| via.to_string()),
                    metric: FABRIC_ROUTE_METRIC,
                    dhcp: false,
                });
            }
        }
        let report = analyze_fabric(&SparkConfig::default(), &probes, Some(fabric), true, 1);
        for node in &report.nodes {
            assert!(
                node.network_fix.is_none(),
                "rank {} still gets a fix: {:?}",
                node.rank,
                node.network_fix
            );
        }
    }

    #[test]
    fn a_switched_fabric_must_reach_every_pair() {
        let probes = cluster();
        let mut links = Vec::new();
        for rank in 0..4usize {
            for port in ["port0", "port1"] {
                for peer in 0..4usize {
                    if peer != rank {
                        links.push(fabric_link(rank, port, peer));
                    }
                }
            }
        }
        let switched = Fabric {
            links,
            reachable: vec![
                Some(vec![1, 2, 3]),
                Some(vec![0]),
                Some(vec![0, 1, 3]),
                Some(vec![0, 1, 2]),
            ],
            shape: FabricShape::Switched,
            links_measured: true,
        };
        let report = analyze_fabric(&SparkConfig::default(), &probes, Some(switched), true, 1);
        let errors = errors(&report);
        assert!(
            errors
                .iter()
                .any(|m| m.contains("spark1 cannot reach spark2")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|m| m.contains("spark1 cannot reach spark3")),
            "{errors:?}"
        );
    }

    #[test]
    fn the_remote_fix_drops_the_sudo_prefix_and_keeps_the_body() {
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
        let node = &report.nodes[3];
        let remote = node.network_fix_remote.as_deref().expect("remote fix");
        assert!(
            !remote.lines().any(|line| line.starts_with("sudo ")),
            "{remote}"
        );
        // A step that fails must stop the body: `netplan apply` at the end must not turn a failed
        // `tee` into a successful job.
        assert_eq!(remote.lines().next(), Some("set -e"), "{remote}");
        assert!(
            remote.contains("tee /etc/netplan/60-sparkflash-fabric.yaml"),
            "{remote}"
        );
        assert!(remote.contains("netplan apply"), "{remote}");
        let text = node.network_fix.as_deref().expect("fix");
        assert!(text.contains("sudo netplan apply"), "{text}");
    }
}
