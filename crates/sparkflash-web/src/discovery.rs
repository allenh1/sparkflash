//! Node discovery and passwordless-SSH assessment.
//!
//! Discovery is DNS-SD over mDNS: this process advertises `_sparkflash._tcp.local.`, and it also
//! browses the service types a stock DGX OS / Avahi install commonly publishes. The result is a
//! candidate list, never an assumption: each SSH target is probed with `BatchMode=yes`, and a
//! failure produces the exact `ssh-copy-id` command the operator can run.
//!
//! The network is behind [`NodeDiscovery`] and [`SshProbe`] so the web routes can be tested on CI
//! without multicast or an SSH server.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mdns_sd::{RecvTimeoutError, ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::Serialize;

use crate::commands::{
    CommandSpec, local_hostname, local_ipv4, ssh_fix_command, ssh_probe_command,
};
use crate::config::{SparkConfig, WORLD_SIZE};

/// The service type this control plane advertises and browses first.
pub const SPARKFLASH_SERVICE: &str = "_sparkflash._tcp.local.";

/// Service types worth browsing on a Spark cluster.
///
/// `_sparkflash` is ours. The others cover hosts that advertise SSH or their machine info through
/// Avahi before SparkFlash has ever run on them, which is exactly a clean-cluster bring-up.
pub const BROWSE_SERVICE_TYPES: &[&str] = &[
    SPARKFLASH_SERVICE,
    "_ssh._tcp.local.",
    "_sftp-ssh._tcp.local.",
    "_workstation._tcp.local.",
    "_device-info._tcp.local.",
];

/// One mDNS answer, normalized to the fields the UI needs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveredNode {
    /// The host part of the SRV record, without a trailing `.local.`.
    pub hostname: String,
    /// IPv4 addresses from the A records, sorted.
    pub addresses: Vec<String>,
    /// The advertised port, when the service has one.
    pub port: Option<u16>,
    /// The DNS-SD type that produced this answer, e.g. `_ssh._tcp.local.`.
    pub services: Vec<String>,
    /// The TXT record as key/value pairs.
    pub properties: BTreeMap<String, String>,
}

/// One candidate node after merging mDNS with the configured rank plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NodeReport {
    /// The configured rank, or `None` for a host mDNS found that the TOML does not name yet.
    pub rank: Option<usize>,
    /// The SSH name or address.
    pub ssh: String,
    /// The configured RoCE peer address.
    pub peer: Option<String>,
    /// Whether the TOML already names this node.
    pub configured: bool,
    /// Whether this rank's commands run on the lead node.
    pub local: bool,
    /// The mDNS hostname, when one was found.
    pub hostname: Option<String>,
    /// Known IPv4 addresses.
    pub addresses: Vec<String>,
    /// DNS-SD types that named this host.
    pub services: Vec<String>,
    /// The advertised service port.
    pub port: Option<u16>,
    /// The TXT record, as advertised.
    pub properties: BTreeMap<String, String>,
    /// `Some(true)` when passwordless SSH succeeded, `Some(false)` when it was refused, `None`
    /// when no probe has run yet.
    pub ssh_ok: Option<bool>,
    /// The probe's short explanation, empty on success.
    pub ssh_message: String,
}

/// The complete discovery response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DiscoveryReport {
    /// Configured nodes first (rank order), then newly discovered hosts.
    pub nodes: Vec<NodeReport>,
    /// A copy-paste command for every failed SSH target, empty when all probes passed.
    pub fix_command: String,
    /// A non-fatal mDNS failure, e.g. multicast being unavailable in a container.
    pub mdns_error: Option<String>,
    /// How many hosts mDNS returned.
    pub discovered: usize,
}

/// A node-discovery source.
pub trait NodeDiscovery: Send + Sync {
    /// Collects mDNS answers for up to `timeout`.
    ///
    /// # Errors
    /// Returns a message when the mDNS daemon cannot browse at all.
    fn discover(&self, timeout: Duration) -> Result<Vec<DiscoveredNode>, String>;

    /// Advertises this control plane's web port.
    ///
    /// # Errors
    /// Returns a message when registration is refused.
    fn advertise(&self, port: u16) -> Result<String, String> {
        let _ = port;
        Ok(String::new())
    }
}

/// The result of one SSH probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeOutcome {
    /// Whether `ssh … true` exited zero.
    pub ok: bool,
    /// A short explanation, empty on success.
    pub message: String,
}

/// A passwordless-SSH probe.
pub trait SshProbe: Send + Sync {
    /// Runs one probe command and reports its outcome.
    fn probe(&self, command: &CommandSpec) -> ProbeOutcome;
}

/// The production probe: spawn `ssh` with a batch-mode command.
#[derive(Debug, Default, Clone, Copy)]
pub struct CommandSshProbe;

impl SshProbe for CommandSshProbe {
    fn probe(&self, command: &CommandSpec) -> ProbeOutcome {
        match std::process::Command::new(&command.program)
            .args(&command.args)
            .output()
        {
            Ok(output) if output.status.success() => ProbeOutcome {
                ok: true,
                message: String::new(),
            },
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let detail = stderr.trim();
                ProbeOutcome {
                    ok: false,
                    message: if detail.is_empty() {
                        format!("ssh exited with {}", output.status)
                    } else {
                        truncate(detail, 300)
                    },
                }
            }
            Err(error) => ProbeOutcome {
                ok: false,
                message: format!("cannot run {}: {error}", command.program),
            },
        }
    }
}

/// The production mDNS browser and advertiser.
pub struct MdnsDiscovery {
    daemon: ServiceDaemon,
    hostname: String,
    advertised: Mutex<Option<String>>,
}

impl MdnsDiscovery {
    /// Starts a daemon.
    ///
    /// # Errors
    /// Returns a message when the multicast socket cannot be created.
    pub fn new() -> Result<Self, String> {
        let daemon = ServiceDaemon::new().map_err(|error| format!("mDNS daemon: {error}"))?;
        Ok(Self {
            daemon,
            hostname: local_hostname(),
            advertised: Mutex::new(None),
        })
    }

    /// The local hostname used to filter our own advert.
    #[must_use]
    pub fn hostname(&self) -> &str {
        &self.hostname
    }
}

impl NodeDiscovery for MdnsDiscovery {
    fn discover(&self, timeout: Duration) -> Result<Vec<DiscoveredNode>, String> {
        let mut receivers = Vec::new();
        let mut browse_error = None;
        for service in BROWSE_SERVICE_TYPES {
            match self.daemon.browse(service) {
                Ok(receiver) => receivers.push((*service, receiver)),
                Err(error) => {
                    browse_error.get_or_insert_with(|| format!("browse {service}: {error}"));
                }
            }
        }
        if receivers.is_empty() {
            return Err(browse_error.unwrap_or_else(|| "no mDNS browse could start".to_owned()));
        }

        let mut found: BTreeMap<String, DiscoveredNode> = BTreeMap::new();
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            for (service, receiver) in &receivers {
                match receiver.recv_timeout(Duration::from_millis(25)) {
                    Ok(ServiceEvent::ServiceResolved(info)) => {
                        let hostname = normalize_hostname(info.get_hostname());
                        if hostname.is_empty() || hostname == normalize_hostname(&self.hostname) {
                            continue;
                        }
                        let entry =
                            found
                                .entry(hostname.clone())
                                .or_insert_with(|| DiscoveredNode {
                                    hostname,
                                    addresses: Vec::new(),
                                    port: None,
                                    services: Vec::new(),
                                    properties: BTreeMap::new(),
                                });
                        let mut addresses: Vec<String> = info
                            .get_addresses_v4()
                            .into_iter()
                            .map(|address| address.to_string())
                            .collect();
                        addresses.sort();
                        for address in addresses {
                            if !entry.addresses.contains(&address) {
                                entry.addresses.push(address);
                            }
                        }
                        if entry.port.is_none() {
                            entry.port = Some(info.get_port());
                        }
                        if !entry.services.iter().any(|known| known == service) {
                            entry.services.push((*service).to_owned());
                        }
                        for property in info.get_properties().iter() {
                            entry
                                .properties
                                .entry(property.key().to_owned())
                                .or_insert_with(|| property.val_str().to_owned());
                        }
                    }
                    Ok(_) => {}
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => {}
                }
            }
        }
        // Each scan starts its own browses; left running, every scan would add another set of
        // periodic queries to the daemon.
        for (service, _) in &receivers {
            let _ = self.daemon.stop_browse(service);
        }
        Ok(found.into_values().collect())
    }

    fn advertise(&self, port: u16) -> Result<String, String> {
        if let Some(fullname) = self.advertised.lock().expect("advert lock").clone() {
            return Ok(fullname);
        }
        let instance = format!("sparkflash-{}", sanitize_instance(&self.hostname));
        let host = format!("{}.local.", self.hostname);
        let address = local_ipv4();
        let properties = [
            ("app", "sparkflash"),
            ("role", "lead"),
            ("version", env!("CARGO_PKG_VERSION")),
        ];
        let info = ServiceInfo::new(
            SPARKFLASH_SERVICE,
            &instance,
            &host,
            address.as_str(),
            port,
            &properties[..],
        )
        .map_err(|error| format!("mDNS service info: {error}"))?;
        let fullname = info.get_fullname().to_owned();
        self.daemon
            .register(info)
            .map_err(|error| format!("mDNS register: {error}"))?;
        *self.advertised.lock().expect("advert lock") = Some(fullname.clone());
        Ok(fullname)
    }
}

impl Drop for MdnsDiscovery {
    fn drop(&mut self) {
        let _ = self.daemon.shutdown();
    }
}

/// Merges configured ranks and mDNS answers, then probes every SSH target concurrently.
///
/// The function is blocking on purpose: the web handler calls it from
/// [`tokio::task::spawn_blocking`], and keeping it synchronous makes it testable with a fake probe.
#[must_use]
pub fn report_nodes(
    config: &SparkConfig,
    discovered: &[DiscoveredNode],
    probe: &dyn SshProbe,
) -> DiscoveryReport {
    let mut candidates: Vec<Candidate> = Vec::new();
    for (rank, node) in config.cluster.nodes.iter().enumerate() {
        candidates.push(Candidate {
            rank: Some(rank),
            ssh: node.ssh.clone(),
            peer: Some(node.peer.clone()),
            configured: true,
            local: node.local,
            discovered: None,
        });
    }
    for node in discovered {
        let normalized = normalize_hostname(&node.hostname);
        let existing = candidates.iter_mut().find(|candidate| {
            normalize_hostname(&candidate.ssh) == normalized
                || candidate
                    .peer
                    .as_ref()
                    .is_some_and(|peer| node.addresses.iter().any(|address| address == peer))
                || candidate
                    .discovered
                    .as_ref()
                    .is_some_and(|found: &DiscoveredNode| {
                        found.addresses.iter().any(|address| {
                            node.addresses.iter().any(|candidate| candidate == address)
                        })
                    })
        });
        if let Some(candidate) = existing {
            candidate.discovered = Some(node.clone());
        } else {
            // A discovered host's SSH target must be resolvable without relying on the same
            // mDNS query that found it: prefer the A record, then the `.local` name.
            let ssh = node
                .addresses
                .first()
                .cloned()
                .unwrap_or_else(|| format!("{normalized}.local"));
            candidates.push(Candidate {
                rank: None,
                ssh,
                peer: None,
                configured: false,
                local: false,
                discovered: Some(node.clone()),
            });
        }
    }
    candidates.truncate(24);

    let mut reports = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for candidate in &candidates {
            // The probe logs in as the launcher will: as cluster.ssh_user, not the caller.
            let probe_command = ssh_probe_command(config, &config.ssh_target(&candidate.ssh));
            let candidate = candidate.clone();
            handles.push(scope.spawn(move || {
                // A local rank's commands run on this node through `bash -c`, so it needs no SSH
                // login, and a lead that cannot ssh to itself must not be reported as broken.
                if candidate.configured && candidate.local {
                    return candidate.report(ProbeOutcome {
                        ok: true,
                        message: String::new(),
                    });
                }
                let outcome = probe.probe(&probe_command);
                candidate.report(outcome)
            }));
        }
        handles
            .into_iter()
            .map(|handle| {
                handle.join().unwrap_or_else(|_| NodeReport {
                    rank: None,
                    ssh: String::new(),
                    peer: None,
                    configured: false,
                    local: false,
                    hostname: None,
                    addresses: Vec::new(),
                    services: Vec::new(),
                    port: None,
                    properties: BTreeMap::new(),
                    ssh_ok: Some(false),
                    ssh_message: "the SSH probe thread panicked".to_owned(),
                })
            })
            .collect::<Vec<_>>()
    });
    reports.sort_by_key(|report| {
        (
            report.rank.is_none(),
            report.rank.unwrap_or(usize::MAX),
            report.ssh.clone(),
        )
    });

    let mut failures: Vec<String> = Vec::new();
    for report in &reports {
        if report.ssh_ok == Some(false) && !report.ssh.is_empty() {
            failures.push(report.ssh.clone());
        }
    }
    let fix_command = ssh_fix_command(config, &failures);

    DiscoveryReport {
        nodes: reports,
        fix_command,
        mdns_error: None,
        discovered: discovered.len(),
    }
}

#[derive(Clone)]
struct Candidate {
    rank: Option<usize>,
    ssh: String,
    peer: Option<String>,
    configured: bool,
    local: bool,
    discovered: Option<DiscoveredNode>,
}

impl Candidate {
    fn report(&self, outcome: ProbeOutcome) -> NodeReport {
        let discovered = self.discovered.as_ref();
        NodeReport {
            rank: self.rank,
            ssh: self.ssh.clone(),
            peer: self.peer.clone(),
            configured: self.configured,
            local: self.local,
            hostname: discovered.map(|node| normalize_hostname(&node.hostname)),
            addresses: discovered
                .map(|node| node.addresses.clone())
                .unwrap_or_default(),
            services: discovered
                .map(|node| node.services.clone())
                .unwrap_or_default(),
            port: discovered.and_then(|node| node.port),
            properties: discovered
                .map(|node| node.properties.clone())
                .unwrap_or_default(),
            ssh_ok: Some(outcome.ok),
            ssh_message: outcome.message,
        }
    }
}

/// Strips the mDNS domain from a hostname and lowercases it.
#[must_use]
pub fn normalize_hostname(hostname: &str) -> String {
    let trimmed = hostname.trim().trim_end_matches('.');
    let trimmed = trimmed
        .strip_suffix(".local")
        .or_else(|| trimmed.strip_suffix(".LOCAL"))
        .unwrap_or(trimmed);
    trimmed.to_ascii_lowercase()
}

fn sanitize_instance(hostname: &str) -> String {
    let clean: String = hostname
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' {
                character
            } else {
                '-'
            }
        })
        .collect();
    let clean = clean.trim_matches('-');
    if clean.is_empty() {
        "lead".to_owned()
    } else {
        clean.chars().take(40).collect()
    }
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    let mut shortened: String = text.chars().take(limit).collect();
    shortened.push('…');
    shortened
}

/// The configured ranks as simple labels, used by the state endpoint before any discovery ran.
#[must_use]
pub fn configured_labels(config: &SparkConfig) -> Vec<(&str, &str)> {
    config
        .cluster
        .nodes
        .iter()
        .take(WORLD_SIZE)
        .map(|node| (node.ssh.as_str(), node.peer.as_str()))
        .collect()
}

/// Collects the distinct SSH targets from a discovery report, for display.
#[must_use]
pub fn ssh_targets(report: &DiscoveryReport) -> BTreeSet<String> {
    report
        .nodes
        .iter()
        .filter(|node| !node.ssh.is_empty())
        .map(|node| node.ssh.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AllPass;
    impl SshProbe for AllPass {
        fn probe(&self, _command: &CommandSpec) -> ProbeOutcome {
            ProbeOutcome {
                ok: true,
                message: String::new(),
            }
        }
    }

    struct FailNamed;
    impl SshProbe for FailNamed {
        fn probe(&self, command: &CommandSpec) -> ProbeOutcome {
            let failed = command
                .args
                .iter()
                .any(|argument| argument.contains("spark2"));
            if failed {
                ProbeOutcome {
                    ok: false,
                    message: "Permission denied (publickey)".to_owned(),
                }
            } else {
                ProbeOutcome {
                    ok: true,
                    message: String::new(),
                }
            }
        }
    }

    fn discovered(hostname: &str, address: &str, service: &str) -> DiscoveredNode {
        DiscoveredNode {
            hostname: hostname.to_owned(),
            addresses: vec![address.to_owned()],
            port: Some(22),
            services: vec![service.to_owned()],
            properties: BTreeMap::new(),
        }
    }

    struct FailAll;
    impl SshProbe for FailAll {
        fn probe(&self, _command: &CommandSpec) -> ProbeOutcome {
            ProbeOutcome {
                ok: false,
                message: "Permission denied (publickey)".to_owned(),
            }
        }
    }

    #[test]
    fn a_local_rank_is_not_probed() {
        let config = SparkConfig::default();
        assert!(config.cluster.nodes[0].local);
        let report = report_nodes(&config, &[], &FailAll);
        assert_eq!(report.nodes[0].ssh_ok, Some(true));
        for node in &report.nodes[1..WORLD_SIZE] {
            assert_eq!(node.ssh_ok, Some(false));
        }
        assert!(!report.fix_command.contains("spark0"));
        assert!(report.fix_command.contains("spark1"));
    }

    #[test]
    fn configured_ranks_come_first_and_are_probed() {
        let config = SparkConfig::default();
        let report = report_nodes(&config, &[], &AllPass);
        assert_eq!(report.nodes.len(), WORLD_SIZE);
        for (rank, node) in report.nodes.iter().enumerate() {
            assert_eq!(node.rank, Some(rank));
            assert!(node.configured);
            assert_eq!(node.ssh_ok, Some(true));
        }
        assert_eq!(report.fix_command, "");
    }

    #[test]
    fn mdns_answers_merge_into_the_matching_rank() {
        let config = SparkConfig::default();
        let report = report_nodes(
            &config,
            &[
                discovered("spark0.local.", "10.0.0.1", "_sparkflash._tcp.local."),
                discovered("spark9.local.", "10.0.0.9", "_ssh._tcp.local."),
            ],
            &AllPass,
        );
        assert_eq!(report.discovered, 2);
        let rank0 = report
            .nodes
            .iter()
            .find(|node| node.rank == Some(0))
            .expect("rank 0");
        assert_eq!(rank0.hostname.as_deref(), Some("spark0"));
        assert_eq!(rank0.services, vec!["_sparkflash._tcp.local.".to_owned()]);
        let extra = report
            .nodes
            .iter()
            .find(|node| !node.configured)
            .expect("extra");
        assert_eq!(extra.ssh, "10.0.0.9");
        assert_eq!(extra.addresses, vec!["10.0.0.9".to_owned()]);
    }

    #[test]
    fn an_mdns_address_also_merges_into_a_configured_peer() {
        let config = SparkConfig::default();
        let mut answer = discovered("renamed-spark.local.", "10.0.0.2", "_ssh._tcp.local.");
        answer.addresses = vec!["10.0.0.2".to_owned()];
        let report = report_nodes(&config, &[answer], &AllPass);
        assert_eq!(report.nodes.len(), WORLD_SIZE);
        let rank1 = report
            .nodes
            .iter()
            .find(|node| node.rank == Some(1))
            .expect("rank 1");
        assert_eq!(rank1.hostname.as_deref(), Some("renamed-spark"));
    }

    #[test]
    fn a_failed_probe_produces_the_exact_fix_command() {
        let config = SparkConfig::default();
        let report = report_nodes(&config, &[], &FailNamed);
        let failed = report
            .nodes
            .iter()
            .find(|node| node.ssh == "spark2")
            .expect("spark2");
        assert_eq!(failed.ssh_ok, Some(false));
        assert_eq!(failed.ssh_message, "Permission denied (publickey)");
        assert!(report.fix_command.contains("spark2"));
        assert!(!report.fix_command.contains("spark1"));
        assert!(report.fix_command.contains("ssh-copy-id"));
    }

    struct Recording(Mutex<Vec<String>>);
    impl SshProbe for Recording {
        fn probe(&self, command: &CommandSpec) -> ProbeOutcome {
            // The destination is the argument before the remote command.
            let destination = command.args[command.args.len() - 2].clone();
            self.0.lock().expect("probe log").push(destination);
            ProbeOutcome {
                ok: true,
                message: String::new(),
            }
        }
    }

    #[test]
    fn the_probe_logs_in_as_the_configured_ssh_user() {
        let mut config = SparkConfig::default();
        config.cluster.ssh_user = "ops".to_owned();
        let probe = Recording(Mutex::new(Vec::new()));
        let report = report_nodes(
            &config,
            &[discovered("spark9.local.", "10.0.0.9", "_ssh._tcp.local.")],
            &probe,
        );
        let mut seen = probe.0.into_inner().expect("probe log");
        seen.sort();
        // Rank 0 is `local = true` in the defaults, so it is not probed.
        assert_eq!(
            seen,
            ["ops@10.0.0.9", "ops@spark1", "ops@spark2", "ops@spark3"]
        );
        // The report still names the host as configured; only the login carries the user.
        assert!(report.nodes.iter().any(|node| node.ssh == "spark1"));
    }

    #[test]
    fn hostnames_are_normalized() {
        assert_eq!(normalize_hostname("Spark0.LOCAL."), "spark0");
        assert_eq!(normalize_hostname("spark1"), "spark1");
        assert_eq!(normalize_hostname("spark2.local"), "spark2");
    }

    #[test]
    fn ssh_targets_are_distinct() {
        let config = SparkConfig::default();
        let report = report_nodes(&config, &[], &AllPass);
        let targets = ssh_targets(&report);
        assert_eq!(targets.len(), WORLD_SIZE);
    }
}
