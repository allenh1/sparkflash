//! Lead-node web control plane for SparkFlash.
//!
//! The crate is deliberately outside the inference path. It reads a TOML file, discovers the
//! other Sparks with mDNS, probes passwordless SSH, drives the existing launcher and converter,
//! and samples the local rank-0 telemetry endpoints. The binary entry point is
//! `sparkflash --web`, dispatched from `sparkflash-cli`.
//!
//! The modules are split along the seams that can be tested without a cluster:
//!
//! - [`config`] owns the typed TOML and every derived command-line environment value.
//! - [`commands`] builds process argument vectors and quotes remote shell fragments.
//! - [`hf`] resolves a Hugging Face model id to its checkpoint in the hub cache and lists the
//!   cached models.
//! - [`discovery`] is DNS-SD plus a `BatchMode=yes` SSH probe behind traits.
//! - [`jobs`] runs long actions asynchronously with a bounded log tail.
//! - [`stats`] samples Prometheus and cluster JSON into a downsampleable history and writes it
//!   only while the cluster is idle.
//! - [`api`] is the axum router and the embedded single-page dark UI.

pub mod api;
pub mod commands;
pub mod config;
pub mod discovery;
pub mod hf;
pub mod jobs;
mod run;
pub mod stats;

pub use api::{WebState, WebStateInit, router};
pub use commands::{
    CommandSpec, ServeAction, convert_command, serve_action_command, ssh_fix_command,
    ssh_probe_command,
};
pub use config::{ConfigError, SparkConfig};
pub use discovery::{
    CommandSshProbe, DiscoveredNode, DiscoveryReport, MdnsDiscovery, NodeDiscovery, NodeReport,
    ProbeOutcome, SshProbe, report_nodes,
};
pub use jobs::{JobDetail, JobManager, JobStatus, JobSummary};
pub use run::run;
pub use stats::{
    ClusterTelemetry, MetricsSnapshot, RawSnapshot, Sample, SeriesResponse, StatsStore,
    TelemetrySource, Totals,
};
