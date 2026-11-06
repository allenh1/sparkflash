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

//! The forward run's header: the arm, the collective path and the NCCL transport environment,
//! decided once at startup and recorded verbatim in the token envelope and the forward record (S2-a).
//!
//! [`RunHeader::decide`] settles all three before any device work:
//!
//! 1. An unset `SPARK_COMM_REDUCE` takes the arm's own path ([`ReducePath::default_for`]): `nccl`
//!    on the device arm, the default arm, and `tcp` on the host arm. Every model's serve takes the
//!    same default. The header records the switch as unset beside the path it decided.
//! 2. A path that is set must match the arm: the device arm's per-layer collectives are NCCL
//!    enqueues (`nccl`), and the host arm's are the reference exchange (`tcp`) or the RoCE one-shot
//!    (`roce`). A mismatch is a refusal, never a quiet reinterpretation, so the header always names
//!    the path the run's collectives actually took.
//! 3. Every switch is read here, exactly once, and the raw values are recorded next to the decision
//!    with the NCCL transport variables ([`sparkflash_comm::nccl::TRANSPORT_ENV`]). The NCCL build
//!    itself is known only after the communicator opens, so the forward record adds it there.

use crate::layer_device::{LAYER_DEVICE_ENV, LayerArm, decide_layer_arm};

/// The collective-path switch; unset takes the arm's own path ([`ReducePath::default_for`]).
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
    /// The path an unset switch takes on `arm`: the device arm's collectives are NCCL enqueues,
    /// and the host arm's reference exchange is TCP.
    #[must_use]
    pub fn default_for(arm: LayerArm) -> Self {
        if arm.is_device() {
            Self::Nccl
        } else {
            Self::Tcp
        }
    }

    /// Parses a `SPARK_COMM_REDUCE` value: `nccl`, `tcp`, or `roce` (also `rdma`), ASCII
    /// case-insensitive. Unset or empty takes `default`.
    ///
    /// # Errors
    /// Any other spelling (surrounding whitespace included): the run refuses rather than guessing
    /// which collective path was meant.
    pub fn parse(value: Option<&str>, default: Self) -> Result<Self, String> {
        let Some(raw) = value.filter(|raw| !raw.is_empty()) else {
            return Ok(default);
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
    /// An arm switch that names neither arm, an unknown `SPARK_COMM_REDUCE`, or a path that does
    /// not match the arm. [`Self::observed_json`] records the same lookup for the refusal.
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
        let reduce = ReducePath::parse(value(COMM_REDUCE_ENV), ReducePath::default_for(arm))?;
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
        let profile = sparkflash_model::active_profile();
        Self::decide(|name| {
            if name == LAYER_DEVICE_ENV {
                return profile.diagnostics.layer_device.clone();
            }
            if name == FORWARD_EVIDENCE_ENV {
                return profile
                    .diagnostics
                    .forward_evidence
                    .map(|value| if value { "1" } else { "0" }.to_owned());
            }
            std::env::var(name).ok()
        })
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
    fn an_unset_reduce_path_takes_the_default_and_an_unknown_one_refuses() {
        for value in [None, Some("")] {
            for default in [ReducePath::Nccl, ReducePath::Tcp] {
                assert_eq!(ReducePath::parse(value, default), Ok(default));
            }
        }
        for value in ["roc", " tcp", "nccl ", "ib", "0"] {
            let error =
                ReducePath::parse(Some(value), ReducePath::Nccl).expect_err("a typo refuses");
            assert!(
                error.contains("names no collective path"),
                "{value:?}: {error}"
            );
        }
        let tcp = ReducePath::Tcp;
        assert_eq!(ReducePath::parse(Some("NCCL"), tcp), Ok(ReducePath::Nccl));
        assert_eq!(ReducePath::parse(Some("tcp"), ReducePath::Nccl), Ok(tcp));
        assert_eq!(ReducePath::parse(Some("roce"), tcp), Ok(ReducePath::Roce));
        assert_eq!(ReducePath::parse(Some("RDMA"), tcp), Ok(ReducePath::Roce));
        assert_eq!(ReducePath::default_for(LayerArm::Device), ReducePath::Nccl);
        assert_eq!(ReducePath::default_for(LayerArm::Host("test")), tcp);
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
    fn the_default_run_is_the_device_arm_on_nccl_named_or_not() {
        for pairs in [&[][..], &[("SPARK_COMM_REDUCE", "nccl")][..]] {
            let header = RunHeader::decide(env(pairs)).expect("decides");
            assert_eq!(header.arm, LayerArm::Device);
            assert_eq!(header.reduce, ReducePath::Nccl);
            assert!(!header.evidence);
        }
        let json = RunHeader::decide(env(&[])).expect("decides").to_json();
        assert_eq!(json["reduce_path"], "nccl");
        assert!(
            json["switches"]["SPARK_COMM_REDUCE"].is_null(),
            "the header says the path was the default"
        );
        for (value, path) in [(None, ReducePath::Tcp), (Some("roce"), ReducePath::Roce)] {
            let mut pairs = vec![("SPARKFLASH_LAYER_DEVICE", "0")];
            pairs.extend(value.map(|value| ("SPARK_COMM_REDUCE", value)));
            let host = RunHeader::decide(env(&pairs)).expect("the host arm stays selectable");
            assert!(!host.arm.is_device());
            assert_eq!(host.reduce, path);
        }
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
        let header = RunHeader::decide(env(&[("SPARKFLASH_FORWARD_EVIDENCE", "1")]))
            .expect("evidence alone takes the host arm's default");
        assert_eq!(header.reduce, ReducePath::Tcp);
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
