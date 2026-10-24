//! The forward run's header: the arm, the collective path and the NCCL transport environment,
//! decided once at startup and recorded verbatim in the token envelope and the forward record (S2-a).
//!
//! Three rules, all enforced by [`RunHeader::decide`] before any device work:
//!
//! 1. `SPARK_COMM_REDUCE` has NO default. Since the device arm became the default (S2), an unset
//!    value would silently mean "the NCCL communicator" on one binary and "the TCP exchange" on the
//!    previous one, so a run that does not name its collective path refuses to start.
//! 2. The path must match the arm: the device arm's per-layer collectives are NCCL enqueues
//!    (`nccl`), and the host arm's are the reference exchange (`tcp`) or the RoCE one-shot (`roce`).
//!    A mismatch is a refusal, never a quiet reinterpretation, so the header always names the path
//!    the run's collectives actually took.
//! 3. Every switch is read here, exactly once, and the raw values are recorded next to the decision
//!    with the NCCL transport variables ([`sparkflash_comm::nccl::TRANSPORT_ENV`]). The NCCL build
//!    itself is known only after the communicator opens, so the forward record adds it there.

use crate::layer_device::{LAYER_DEVICE_ENV, LayerArm, decide_layer_arm};

/// The collective-path switch. Required: [`ReducePath::parse`] refuses an unset value.
pub const COMM_REDUCE_ENV: &str = "SPARK_COMM_REDUCE";

/// The per-layer evidence chain switch; recording forces the host arm.
pub const FORWARD_EVIDENCE_ENV: &str = "SPARKFLASH_FORWARD_EVIDENCE";

/// The header schema tag.
pub const SCHEMA: &str = "sparkflash-run-header/1";

/// The three switches the arm and path decision reads, in the order the header records them.
pub const DECISION_ENV: [&str; 3] = [LAYER_DEVICE_ENV, FORWARD_EVIDENCE_ENV, COMM_REDUCE_ENV];

/// The path the run's two per-layer collectives take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducePath {
    /// The run's NCCL communicator (the device arm).
    Nccl,
    /// The reference TCP exchange (the host arm).
    Tcp,
    /// The canonical RoCE one-shot (the host arm; falls back to tcp at bring-up and says so).
    Roce,
}

impl ReducePath {
    /// Parses a `SPARK_COMM_REDUCE` value: `nccl`, `tcp`, or `roce` (also `rdma`), ASCII
    /// case-insensitive.
    ///
    /// # Errors
    /// Unset, empty, or any other spelling (surrounding whitespace included): the run refuses rather
    /// than guessing which collective path was meant.
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        let Some(raw) = value.filter(|raw| !raw.is_empty()) else {
            return Err(format!(
                "{COMM_REDUCE_ENV} is unset: a forward run names its collective path explicitly - nccl for the device arm (the default), tcp or roce for the host arm ({LAYER_DEVICE_ENV}=0)"
            ));
        };
        match raw.to_ascii_lowercase().as_str() {
            "nccl" => Ok(Self::Nccl),
            "tcp" => Ok(Self::Tcp),
            "roce" | "rdma" => Ok(Self::Roce),
            _ => Err(format!(
                "{COMM_REDUCE_ENV}={raw:?} names no collective path: nccl (device arm), tcp or roce (host arm)"
            )),
        }
    }

    /// The header's spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nccl => "nccl",
            Self::Tcp => "tcp",
            Self::Roce => "roce",
        }
    }

    /// Checks the path against the arm the run takes.
    ///
    /// # Errors
    /// The device arm with `tcp`/`roce`, or the host arm with `nccl`; the message names the arm's
    /// reason, so an evidence run that forced the host arm says why `nccl` no longer fits.
    pub fn require_for_arm(self, arm: LayerArm) -> Result<(), String> {
        match (arm, self) {
            (LayerArm::Device, Self::Nccl) | (LayerArm::Host(_), Self::Tcp | Self::Roce) => Ok(()),
            (LayerArm::Device, path) => Err(format!(
                "{COMM_REDUCE_ENV}={} is a host-arm exchange, but this run takes the device arm, whose per-layer collectives are NCCL enqueues: set {COMM_REDUCE_ENV}=nccl, or {LAYER_DEVICE_ENV}=0 for the host arm",
                path.as_str()
            )),
            (LayerArm::Host(reason), Self::Nccl) => Err(format!(
                "{COMM_REDUCE_ENV}=nccl needs the device arm, but this run takes the {reason}: set {COMM_REDUCE_ENV}=tcp or roce"
            )),
        }
    }
}

/// The run's startup decision and the environment it was made from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunHeader {
    /// The arm, from [`decide_layer_arm`].
    pub arm: LayerArm,
    /// Whether the per-layer evidence chain is recorded.
    pub evidence: bool,
    /// The collective path, checked against the arm.
    pub reduce: ReducePath,
    /// The raw [`DECISION_ENV`] values, `None` when unset.
    pub switches: Vec<(&'static str, Option<String>)>,
    /// The raw [`sparkflash_comm::nccl::TRANSPORT_ENV`] values, `None` when unset.
    pub nccl_env: Vec<(&'static str, Option<String>)>,
}

impl RunHeader {
    /// Decides the arm and the collective path from one environment lookup.
    ///
    /// # Errors
    /// An arm switch that names neither arm, an unset or unknown `SPARK_COMM_REDUCE`, or a path that
    /// does not match the arm. [`Self::observed_json`] records the same lookup for the refusal.
    pub fn decide(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let switches = read(&DECISION_ENV, &lookup);
        let value = |name: &str| {
            switches
                .iter()
                .find(|(key, _)| *key == name)
                .and_then(|(_, value)| value.as_deref())
        };
        let (arm, evidence) =
            decide_layer_arm(value(LAYER_DEVICE_ENV), value(FORWARD_EVIDENCE_ENV))?;
        let reduce = ReducePath::parse(value(COMM_REDUCE_ENV))?;
        reduce.require_for_arm(arm)?;
        Ok(Self {
            arm,
            evidence,
            reduce,
            switches,
            nccl_env: read(&sparkflash_comm::nccl::TRANSPORT_ENV, &lookup),
        })
    }

    /// [`Self::decide`] on the process environment.
    ///
    /// # Errors
    /// As [`Self::decide`].
    pub fn from_env() -> Result<Self, String> {
        Self::decide(|name| std::env::var(name).ok())
    }

    /// The raw switches and transport variables of one lookup, for a run that refused before a
    /// header existed: the refusal record still shows what the operator set.
    #[must_use]
    pub fn observed_json(lookup: impl Fn(&str) -> Option<String>) -> serde_json::Value {
        serde_json::json!({
            "schema": SCHEMA,
            "decided": false,
            "switches": object(&read(&DECISION_ENV, &lookup)),
            "nccl_env": object(&read(&sparkflash_comm::nccl::TRANSPORT_ENV, &lookup)),
        })
    }

    /// The header record. `nccl_build` is null here; the forward record fills it once the
    /// communicator has opened.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "schema": SCHEMA,
            "decided": true,
            "arm": if self.arm.is_device() { "device" } else { "host" },
            "arm_why": self.arm.why(),
            "evidence": self.evidence,
            "reduce_path": self.reduce.as_str(),
            "switches": object(&self.switches),
            "nccl_env": object(&self.nccl_env),
            "nccl_build": serde_json::Value::Null,
        })
    }

    /// The run-log lines: the decision, then every transport variable (`-` when unset).
    #[must_use]
    pub fn lines(&self, rank: usize) -> Vec<String> {
        let render = |values: &[(&'static str, Option<String>)]| {
            values
                .iter()
                .map(|(name, value)| format!("{name}={}", value.as_deref().unwrap_or("-")))
                .collect::<Vec<_>>()
                .join(" ")
        };
        vec![
            format!(
                "tp4-chain r{rank} run header: arm={} reduce={} evidence={} ({})",
                if self.arm.is_device() {
                    "device"
                } else {
                    "host"
                },
                self.reduce.as_str(),
                self.evidence,
                self.arm.why()
            ),
            format!("tp4-chain r{rank} run header: {}", render(&self.switches)),
            format!("tp4-chain r{rank} run header: {}", render(&self.nccl_env)),
        ]
    }
}

fn read(
    names: &[&'static str],
    lookup: &impl Fn(&str) -> Option<String>,
) -> Vec<(&'static str, Option<String>)> {
    names.iter().map(|name| (*name, lookup(name))).collect()
}

fn object(values: &[(&'static str, Option<String>)]) -> serde_json::Value {
    values
        .iter()
        .map(|(name, value)| ((*name).to_string(), serde_json::json!(value)))
        .collect::<serde_json::Map<_, _>>()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    #[test]
    fn an_unset_or_unknown_reduce_path_refuses_and_names_the_switch() {
        for value in [None, Some("")] {
            let error = ReducePath::parse(value).expect_err("unset refuses");
            assert!(error.contains("SPARK_COMM_REDUCE is unset"), "{error}");
        }
        for value in ["roc", " tcp", "nccl ", "ib", "0"] {
            let error = ReducePath::parse(Some(value)).expect_err("a typo refuses");
            assert!(
                error.contains("names no collective path"),
                "{value:?}: {error}"
            );
        }
        assert_eq!(ReducePath::parse(Some("NCCL")), Ok(ReducePath::Nccl));
        assert_eq!(ReducePath::parse(Some("tcp")), Ok(ReducePath::Tcp));
        assert_eq!(ReducePath::parse(Some("roce")), Ok(ReducePath::Roce));
        assert_eq!(ReducePath::parse(Some("RDMA")), Ok(ReducePath::Roce));
    }

    #[test]
    fn the_path_must_match_the_arm() {
        let host = LayerArm::Host("host arm (test)");
        assert!(ReducePath::Nccl.require_for_arm(LayerArm::Device).is_ok());
        assert!(ReducePath::Tcp.require_for_arm(host).is_ok());
        assert!(ReducePath::Roce.require_for_arm(host).is_ok());
        let error = ReducePath::Tcp
            .require_for_arm(LayerArm::Device)
            .expect_err("tcp on the device arm");
        assert!(error.contains("SPARK_COMM_REDUCE=nccl"), "{error}");
        let error = ReducePath::Nccl
            .require_for_arm(host)
            .expect_err("nccl on the host arm");
        assert!(
            error.contains("host arm (test)"),
            "the arm's reason is quoted: {error}"
        );
    }

    #[test]
    fn the_default_run_is_the_device_arm_on_nccl_and_an_unset_path_refuses_it() {
        let header = RunHeader::decide(env(&[("SPARK_COMM_REDUCE", "nccl")])).expect("decides");
        assert_eq!(header.arm, LayerArm::Device);
        assert_eq!(header.reduce, ReducePath::Nccl);
        assert!(!header.evidence);
        let error = RunHeader::decide(env(&[])).expect_err("unset refuses");
        assert!(error.contains("SPARK_COMM_REDUCE is unset"), "{error}");
        let host = RunHeader::decide(env(&[
            ("SPARKFLASH_LAYER_DEVICE", "0"),
            ("SPARK_COMM_REDUCE", "tcp"),
        ]))
        .expect("the host arm stays selectable");
        assert!(!host.arm.is_device());
        assert_eq!(host.reduce, ReducePath::Tcp);
    }

    #[test]
    fn evidence_recording_forces_the_host_arm_so_nccl_refuses_with_the_reason() {
        let error = RunHeader::decide(env(&[
            ("SPARKFLASH_FORWARD_EVIDENCE", "1"),
            ("SPARK_COMM_REDUCE", "nccl"),
        ]))
        .expect_err("evidence forces host");
        assert!(error.contains("evidence recording forces it"), "{error}");
        let header = RunHeader::decide(env(&[
            ("SPARKFLASH_FORWARD_EVIDENCE", "1"),
            ("SPARK_COMM_REDUCE", "tcp"),
        ]))
        .expect("evidence on tcp");
        assert!(header.evidence && !header.arm.is_device());
    }

    #[test]
    fn a_bad_arm_spelling_refuses_before_the_path_is_read() {
        let error = RunHeader::decide(env(&[
            ("SPARKFLASH_LAYER_DEVICE", "on"),
            ("SPARK_COMM_REDUCE", "nccl"),
        ]))
        .expect_err("bad arm spelling");
        assert!(error.contains("SPARKFLASH_LAYER_DEVICE"), "{error}");
    }

    #[test]
    fn the_header_records_every_switch_and_transport_variable_set_or_not() {
        let header = RunHeader::decide(env(&[
            ("SPARK_COMM_REDUCE", "nccl"),
            ("NCCL_IB_HCA", "rocep1s0f1,roceP2p1s0f1"),
            ("NCCL_NET_GDR_LEVEL", "SYS"),
        ]))
        .expect("decides");
        let json = header.to_json();
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["decided"], true);
        assert_eq!(json["arm"], "device");
        assert_eq!(json["reduce_path"], "nccl");
        assert_eq!(json["switches"]["SPARK_COMM_REDUCE"], "nccl");
        assert!(json["switches"]["SPARKFLASH_LAYER_DEVICE"].is_null());
        assert_eq!(json["nccl_env"]["NCCL_IB_HCA"], "rocep1s0f1,roceP2p1s0f1");
        assert_eq!(json["nccl_env"]["NCCL_NET_GDR_LEVEL"], "SYS");
        for name in sparkflash_comm::nccl::TRANSPORT_ENV {
            assert!(
                json["nccl_env"].get(name).is_some(),
                "{name} is recorded even when unset"
            );
        }
        assert!(
            json["nccl_build"].is_null(),
            "the build is the forward record's to fill"
        );
        let lines = header.lines(2);
        assert!(lines[0].starts_with("tp4-chain r2 run header: arm=device reduce=nccl"));
        assert!(lines[2].contains("NCCL_NET_GDR_LEVEL=SYS"), "{lines:?}");
        assert!(lines[2].contains("NCCL_ALGO=-"), "{lines:?}");
    }

    #[test]
    fn a_refusal_still_records_what_the_operator_set() {
        let json = RunHeader::observed_json(env(&[("NCCL_PROTO", "LL")]));
        assert_eq!(json["decided"], false);
        assert!(json["switches"]["SPARK_COMM_REDUCE"].is_null());
        assert_eq!(json["nccl_env"]["NCCL_PROTO"], "LL");
    }
}
