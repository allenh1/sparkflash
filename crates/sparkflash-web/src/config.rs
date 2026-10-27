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

//! The lead node's TOML configuration.
//!
//! The file is deliberately the whole setup: model weights, engine, the four SSH targets and
//! RoCE peers, paths, NCCL variables, and the sampling policy. Everything the launcher and the
//! converter need is derived from this one typed value, so the web form and a hand-written TOML
//! cannot drift apart.
//!
//! Relative paths are resolved against [`PathsConfig::repo_root`], which is itself resolved
//! against the process working directory. This keeps a checkout-relative sample useful while
//! still accepting the absolute paths a production install uses.

use std::collections::BTreeMap;
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The model ids this build can serve.
///
/// "Model" means the weights, not the runtime. Adding a second checkpoint is a deliberate
/// change here, plus the conversion and engine code that knows how to read it.
pub const SUPPORTED_WEIGHTS: &[&str] = &["deepseek-v4.1-flash"];

/// The engines this build can run the weights with.
///
/// Only the tensor-parallel-4 device engine exists today. The web selector keeps the concept
/// distinct from the weights so a future engine can be added without inventing a new surface.
pub const SUPPORTED_ENGINES: &[&str] = &["deepseek-v4.1-flash"];

/// The canonical Hugging Face id of the checkpoint this engine reads.
pub const DEFAULT_MODEL_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

/// The maximum context window of V4.1-Flash.
pub const MODEL_CONTEXT_TOKENS: usize = 1_048_576;

/// The engine's current serving width; four DGX Sparks at tensor parallelism four.
pub const WORLD_SIZE: usize = 4;

/// The largest concurrency the current engine is designed for.
pub const MAX_CONCURRENCY: usize = 32;

/// The default config path, relative to the lead node's working directory.
pub const DEFAULT_CONFIG_PATH: &str = "sparkflash.toml";

/// A configuration that failed to parse or validate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    /// The human-readable reason, suitable for the web form and stderr.
    pub message: String,
}

impl ConfigError {
    /// Builds an error with `message`.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ConfigError {}

/// The typed contents of `sparkflash.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SparkConfig {
    /// The weight checkpoint and the paths converted from it.
    pub model: ModelConfig,
    /// How those weights are run.
    pub engine: EngineConfig,
    /// Context and concurrency selections.
    pub runtime: RuntimeConfig,
    /// The four-node SSH and fabric plan.
    pub cluster: ClusterConfig,
    /// Paths on every node.
    pub paths: PathsConfig,
    /// NCCL and extra environment variables passed to every rank.
    pub network: NetworkConfig,
    /// The web listener and telemetry source.
    pub web: WebConfig,
    /// Telemetry sampling and persistence policy.
    pub stats: StatsConfig,
}

/// The weights and their converted layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// The weights id shown in the model selector.
    pub weights: String,
    /// The checkpoint's Hugging Face model id, e.g. `deepseek-ai/DeepSeek-V4.1-Flash`.
    pub id: String,
    /// The checkpoint directory, the same path on every node. Empty looks `id` up in the
    /// Hugging Face hub cache (see [`crate::hf`]).
    pub checkpoint: String,
    /// The Hugging Face hub cache to look `id` up in. Empty means the standard location:
    /// `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
    pub hub_cache: String,
    /// The `tokenizer.json` path. Empty means `<checkpoint>/tokenizer.json`.
    pub tokenizer: String,
    /// The directory of per-rank converted files, the same path on every node.
    pub converted_dir: String,
    /// The cubin directory built by `scripts/build-cubins.sh`.
    pub cubins: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            weights: SUPPORTED_WEIGHTS[0].to_owned(),
            id: DEFAULT_MODEL_ID.to_owned(),
            checkpoint: String::new(),
            hub_cache: String::new(),
            tokenizer: String::new(),
            converted_dir: "/data/sparkflash-conv".to_owned(),
            cubins: "/opt/sparkflash/cubins".to_owned(),
        }
    }
}

/// How the weights are executed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    /// The engine id shown in the engine selector.
    pub kind: String,
    /// Run the TP=4 device forward. The host twin is only useful for a CPU-only smoke test.
    pub device: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            kind: SUPPORTED_ENGINES[0].to_owned(),
            device: true,
        }
    }
}

/// The two runtime knobs the model exposes to the operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    /// The sequence window advertised and served (`SPARK_MAX_SEQ_LEN`).
    pub max_context_tokens: usize,
    /// The number of sequences the scheduler may run at once (`SPARKFLASH_LANES`).
    pub max_concurrency: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: MODEL_CONTEXT_TOKENS,
            max_concurrency: MAX_CONCURRENCY,
        }
    }
}

/// One node's place in the four-rank group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodeConfig {
    /// The SSH name or address, in launch order.
    pub ssh: String,
    /// The RoCE address the collectives use for this rank.
    pub peer: String,
    /// Run this rank's commands on the lead node instead of over SSH.
    #[serde(default)]
    pub local: bool,
}

impl NodeConfig {
    /// Builds a node entry.
    #[must_use]
    pub fn new(ssh: impl Into<String>, peer: impl Into<String>, local: bool) -> Self {
        Self {
            ssh: ssh.into(),
            peer: peer.into(),
            local,
        }
    }
}

/// The four-node plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClusterConfig {
    /// Rank 0's HTTP bind address on the cluster (`SPARKFLASH_HTTP_BIND`).
    pub http_bind: String,
    /// The first collective port; rank R uses `base_port + R`.
    pub base_port: u16,
    /// The account used for passwordless SSH, empty for the current user.
    pub ssh_user: String,
    /// Extra SSH options, e.g. `-o StrictHostKeyChecking=accept-new`.
    pub ssh_options: Vec<String>,
    /// Exactly [`WORLD_SIZE`] nodes, in rank order.
    pub nodes: Vec<NodeConfig>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            http_bind: "0.0.0.0:8000".to_owned(),
            base_port: 29601,
            ssh_user: std::env::var("USER").unwrap_or_else(|_| "spark".to_owned()),
            ssh_options: vec![
                "-o".to_owned(),
                "BatchMode=yes".to_owned(),
                "-o".to_owned(),
                "ConnectTimeout=8".to_owned(),
            ],
            nodes: vec![
                NodeConfig::new("spark0", "10.0.0.1", true),
                NodeConfig::new("spark1", "10.0.0.2", false),
                NodeConfig::new("spark2", "10.0.0.3", false),
                NodeConfig::new("spark3", "10.0.0.4", false),
            ],
        }
    }
}

/// Paths that must be the same on every node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PathsConfig {
    /// The checkout root on every node.
    pub repo_root: String,
    /// The server binary, relative to `repo_root` or absolute.
    pub binary: String,
    /// The four-node launcher, relative to `repo_root` or absolute.
    pub serve_script: String,
    /// The directory holding `libnccl.so.2`, empty when NCCL is installed system-wide.
    pub nccl_lib: String,
    /// The remote log directory (`SPARKFLASH_LOG_ROOT`).
    pub log_root: String,
}

impl Default for PathsConfig {
    fn default() -> Self {
        Self {
            repo_root: ".".to_owned(),
            binary: "target/release/sparkflash".to_owned(),
            serve_script: "scripts/serve-four-node.sh".to_owned(),
            nccl_lib: String::new(),
            log_root: "target/serve-tp4".to_owned(),
        }
    }
}

/// NCCL variables and an escape hatch for extra rank environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NetworkConfig {
    /// The RoCE device name (`NCCL_IB_HCA`), empty to leave NCCL's default.
    pub nccl_ib_hca: String,
    /// The netdev carrying that device (`NCCL_SOCKET_IFNAME`), empty to leave the default.
    pub nccl_socket_ifname: String,
    /// The RoCE GID index (`NCCL_IB_GID_INDEX`); `None` leaves NCCL's default.
    pub nccl_ib_gid_index: Option<u16>,
    /// Extra environment variables exported to every rank.
    pub env: BTreeMap<String, String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            nccl_ib_hca: String::new(),
            nccl_socket_ifname: String::new(),
            nccl_ib_gid_index: Some(3),
            env: BTreeMap::new(),
        }
    }
}

/// The web listener and the telemetry source it samples.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WebConfig {
    /// The web listener on the lead node. All interfaces by default, because Sparks run headless;
    /// the console has no login of its own, so keep it on a trusted network.
    pub bind: String,
    /// Rank 0's serving base URL, as seen from the lead node.
    pub serve_url: String,
    /// How long an mDNS browse collects answers.
    pub discovery_timeout_secs: u64,
    /// How long one SSH probe may take before it is reported as unreachable.
    pub ssh_timeout_secs: u64,
    /// Advertise `_sparkflash._tcp.local.` from this process.
    pub advertise: bool,
    /// Host names the UI may be reached by besides an IP address, `localhost` and this machine's
    /// own name, e.g. the name a reverse proxy forwards. Any other `Host` is refused, so a web
    /// page cannot rebind its own name to this listener.
    pub allowed_hosts: Vec<String>,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            bind: "0.0.0.0:8790".to_owned(),
            serve_url: "http://127.0.0.1:8000".to_owned(),
            discovery_timeout_secs: 3,
            ssh_timeout_secs: 5,
            advertise: true,
            allowed_hosts: Vec::new(),
        }
    }
}

/// Sampling and idle-only persistence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StatsConfig {
    /// Seconds between cheap `/metrics` + `/v1/cluster/stats` scrapes.
    pub sample_interval_secs: u64,
    /// How much history the in-memory graph buffer keeps.
    pub retention_secs: u64,
    /// Where the idle snapshot is written.
    pub persist_path: String,
    /// Minimum seconds between two idle writes.
    pub flush_interval_secs: u64,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            sample_interval_secs: 2,
            retention_secs: 86_400,
            persist_path: "target/web/stats.json".to_owned(),
            flush_interval_secs: 60,
        }
    }
}

impl SparkConfig {
    /// Reads and validates a TOML file.
    ///
    /// # Errors
    /// Returns [`ConfigError`] when the file cannot be read, is not TOML, or fails validation.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            ConfigError::new(format!("cannot read {}: {error}", path.display()))
        })?;
        Self::from_toml(&text)
            .map_err(|error| ConfigError::new(format!("{}: {}", path.display(), error.message)))
    }

    /// Parses and validates TOML text.
    ///
    /// # Errors
    /// Returns [`ConfigError`] when the text is malformed or fails validation.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text)
            .map_err(|error| ConfigError::new(format!("invalid TOML: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Serializes the configuration as TOML.
    ///
    /// # Errors
    /// Returns [`ConfigError`] if the value cannot be represented or is invalid.
    pub fn to_toml(&self) -> Result<String, ConfigError> {
        self.validate()?;
        toml::to_string_pretty(self)
            .map_err(|error| ConfigError::new(format!("cannot encode TOML: {error}")))
    }

    /// Validates every invariant the launcher and converter rely on.
    ///
    /// # Errors
    /// Returns a message naming the first field that is wrong.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !SUPPORTED_WEIGHTS.contains(&self.model.weights.as_str()) {
            return Err(ConfigError::new(format!(
                "model.weights {:?} is not supported by this build (supported: {})",
                self.model.weights,
                SUPPORTED_WEIGHTS.join(", ")
            )));
        }
        if !SUPPORTED_ENGINES.contains(&self.engine.kind.as_str()) {
            return Err(ConfigError::new(format!(
                "engine.kind {:?} is not supported by this build (supported: {})",
                self.engine.kind,
                SUPPORTED_ENGINES.join(", ")
            )));
        }
        if self.model.id.trim().is_empty() {
            return Err(ConfigError::new("model.id must not be empty"));
        }
        if self.model.checkpoint.trim().is_empty() && !crate::hf::is_model_id(&self.model.id) {
            return Err(ConfigError::new(format!(
                "model.id {:?} is not a Hugging Face model id, so the checkpoint cannot be looked \
                 up in the cache; set model.checkpoint to its directory",
                self.model.id
            )));
        }
        if self.model.converted_dir.trim().is_empty() {
            return Err(ConfigError::new("model.converted_dir must not be empty"));
        }
        if self.engine.device && self.model.cubins.trim().is_empty() {
            return Err(ConfigError::new(
                "model.cubins must not be empty when engine.device is true",
            ));
        }
        if !(1..=MODEL_CONTEXT_TOKENS).contains(&self.runtime.max_context_tokens) {
            return Err(ConfigError::new(format!(
                "runtime.max_context_tokens must be in 1..={MODEL_CONTEXT_TOKENS}, got {}",
                self.runtime.max_context_tokens
            )));
        }
        if !(1..=MAX_CONCURRENCY).contains(&self.runtime.max_concurrency) {
            return Err(ConfigError::new(format!(
                "runtime.max_concurrency must be in 1..={MAX_CONCURRENCY}, got {}",
                self.runtime.max_concurrency
            )));
        }
        if self.cluster.nodes.len() != WORLD_SIZE {
            return Err(ConfigError::new(format!(
                "cluster.nodes must list exactly {WORLD_SIZE} nodes in rank order, got {}",
                self.cluster.nodes.len()
            )));
        }
        // ssh reads a destination or user that starts with '-' as an option (-oProxyCommand=...),
        // and the launcher splits SPARK_NODES on whitespace.
        let user = &self.cluster.ssh_user;
        if user.starts_with('-') || user.contains('@') || user.chars().any(char::is_whitespace) {
            return Err(ConfigError::new(format!(
                "cluster.ssh_user {user:?} must be an account name: no whitespace, '@' or leading '-'"
            )));
        }
        let mut ssh_names = std::collections::BTreeSet::new();
        let mut peers = std::collections::BTreeSet::new();
        for (rank, node) in self.cluster.nodes.iter().enumerate() {
            if node.ssh.trim().is_empty() {
                return Err(ConfigError::new(format!(
                    "cluster.nodes[{rank}].ssh must not be empty"
                )));
            }
            if node.ssh.starts_with('-') || node.ssh.chars().any(char::is_whitespace) {
                return Err(ConfigError::new(format!(
                    "cluster.nodes[{rank}].ssh {:?} must be a host name or address: no whitespace or leading '-'",
                    node.ssh
                )));
            }
            if !ssh_names.insert(node.ssh.clone()) {
                return Err(ConfigError::new(format!(
                    "cluster.nodes[{rank}].ssh {:?} is repeated",
                    node.ssh
                )));
            }
            node.peer.parse::<std::net::IpAddr>().map_err(|_| {
                ConfigError::new(format!(
                    "cluster.nodes[{rank}].peer {:?} is not an IP address",
                    node.peer
                ))
            })?;
            if !peers.insert(node.peer.clone()) {
                return Err(ConfigError::new(format!(
                    "cluster.nodes[{rank}].peer {:?} is repeated",
                    node.peer
                )));
            }
        }
        parse_socket_addr("cluster.http_bind", &self.cluster.http_bind)?;
        parse_socket_addr("web.bind", &self.web.bind)?;
        validate_serve_url(&self.web.serve_url)?;
        if !(1..=60).contains(&self.web.discovery_timeout_secs) {
            return Err(ConfigError::new(
                "web.discovery_timeout_secs must be in 1..=60",
            ));
        }
        if !(1..=60).contains(&self.web.ssh_timeout_secs) {
            return Err(ConfigError::new("web.ssh_timeout_secs must be in 1..=60"));
        }
        for option in &self.cluster.ssh_options {
            if option.trim().is_empty() || option.chars().any(char::is_whitespace) {
                return Err(ConfigError::new(format!(
                    "cluster.ssh_options entry {option:?} must be a single argument with no whitespace"
                )));
            }
        }
        for key in self.network.env.keys() {
            let mut chars = key.chars();
            let valid = chars
                .next()
                .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
                && chars.all(|rest| rest == '_' || rest.is_ascii_alphanumeric());
            if !valid {
                return Err(ConfigError::new(format!(
                    "network.env key {key:?} is not an environment variable name"
                )));
            }
        }
        for host in &self.web.allowed_hosts {
            if host.trim().is_empty()
                || host.contains(['/', ':'])
                || host.contains(char::is_whitespace)
            {
                return Err(ConfigError::new(format!(
                    "web.allowed_hosts entry {host:?} must be a bare host name, without a scheme or port"
                )));
            }
        }
        if !(1..=300).contains(&self.stats.sample_interval_secs) {
            return Err(ConfigError::new(
                "stats.sample_interval_secs must be in 1..=300",
            ));
        }
        if self.stats.retention_secs < 60 {
            return Err(ConfigError::new("stats.retention_secs must be at least 60"));
        }
        if self.stats.flush_interval_secs < 1 {
            return Err(ConfigError::new(
                "stats.flush_interval_secs must be at least 1",
            ));
        }
        if self.stats.flush_interval_secs > self.stats.retention_secs {
            return Err(ConfigError::new(
                "stats.flush_interval_secs must not exceed stats.retention_secs",
            ));
        }
        if self.stats.persist_path.trim().is_empty() {
            return Err(ConfigError::new("stats.persist_path must not be empty"));
        }
        Ok(())
    }

    /// Saves the configuration atomically: write a sibling temp file, then rename.
    ///
    /// # Errors
    /// Returns [`ConfigError`] when validation, directory creation, writing or renaming fails.
    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        let text = self.to_toml()?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|error| {
                ConfigError::new(format!("cannot create {}: {error}", parent.display()))
            })?;
        }
        let temp = path.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&temp, text).map_err(|error| {
            ConfigError::new(format!("cannot write {}: {error}", temp.display()))
        })?;
        std::fs::rename(&temp, path).map_err(|error| {
            let _ = std::fs::remove_file(&temp);
            ConfigError::new(format!("cannot replace {}: {error}", path.display()))
        })
    }

    /// The Hugging Face hub cache `model.id` is looked up in.
    #[must_use]
    pub fn hub_cache_dir(&self) -> Option<PathBuf> {
        if self.model.hub_cache.trim().is_empty() {
            crate::hf::default_cache_dir()
        } else {
            Some(PathBuf::from(&self.model.hub_cache))
        }
    }

    /// The checkpoint directory: `model.checkpoint` when set, else `model.id`'s snapshot in the
    /// Hugging Face hub cache.
    ///
    /// # Errors
    ///
    /// When the checkpoint is left to the cache and the model is not in it.
    pub fn resolve_checkpoint(&self) -> Result<PathBuf, ConfigError> {
        if !self.model.checkpoint.trim().is_empty() {
            return Ok(PathBuf::from(&self.model.checkpoint));
        }
        let cache = self.hub_cache_dir().ok_or_else(|| {
            ConfigError::new(
                "no Hugging Face cache directory: set HF_HUB_CACHE, model.hub_cache or \
                 model.checkpoint",
            )
        })?;
        crate::hf::resolve(&cache, &self.model.id)
            .map(|snapshot| snapshot.path)
            .map_err(ConfigError::new)
    }

    /// A copy with `model.checkpoint` resolved, which the launcher and converter commands are
    /// built from.
    ///
    /// # Errors
    ///
    /// See [`Self::resolve_checkpoint`].
    pub fn resolved(&self) -> Result<Self, ConfigError> {
        let mut config = self.clone();
        config.model.checkpoint = self.resolve_checkpoint()?.display().to_string();
        Ok(config)
    }

    /// The tokenizer path, deriving `<checkpoint>/tokenizer.json` when the field is empty.
    #[must_use]
    pub fn effective_tokenizer(&self) -> PathBuf {
        if self.model.tokenizer.trim().is_empty() {
            Path::new(&self.model.checkpoint).join("tokenizer.json")
        } else {
            PathBuf::from(&self.model.tokenizer)
        }
    }

    /// The checkout root on the lead node.
    #[must_use]
    pub fn repo_root(&self) -> PathBuf {
        PathBuf::from(&self.paths.repo_root)
    }

    /// The binary path, resolved against `repo_root` when it is relative.
    #[must_use]
    pub fn binary_path(&self) -> PathBuf {
        resolve_against(&self.repo_root(), &self.paths.binary)
    }

    /// The launcher script path, resolved against `repo_root` when it is relative.
    #[must_use]
    pub fn serve_script_path(&self) -> PathBuf {
        resolve_against(&self.repo_root(), &self.paths.serve_script)
    }

    /// The telemetry file path, resolved against `repo_root` when it is relative.
    #[must_use]
    pub fn stats_path(&self) -> PathBuf {
        resolve_against(&self.repo_root(), &self.stats.persist_path)
    }

    /// The environment the four-node launcher receives.
    ///
    /// The keys match `scripts/serve-four-node.sh` and `sparkflash serve` exactly. Empty optional
    /// values are omitted, so the launcher keeps its own default instead of receiving an empty
    /// path.
    #[must_use]
    pub fn launch_env(&self) -> BTreeMap<String, String> {
        let mut env = BTreeMap::new();
        env.insert(
            "SPARK_NODES".to_owned(),
            self.cluster
                .nodes
                .iter()
                .map(|node| node.ssh.as_str())
                .collect::<Vec<_>>()
                .join(" "),
        );
        env.insert(
            "SPARK_PEERS".to_owned(),
            self.cluster
                .nodes
                .iter()
                .map(|node| node.peer.as_str())
                .collect::<Vec<_>>()
                .join(" "),
        );
        env.insert(
            "SPARKFLASH_CHECKPOINT".to_owned(),
            self.model.checkpoint.clone(),
        );
        env.insert(
            "SPARKFLASH_TOKENIZER".to_owned(),
            self.effective_tokenizer().display().to_string(),
        );
        env.insert(
            "SPARKFLASH_CONVERTED_DIR".to_owned(),
            self.model.converted_dir.clone(),
        );
        env.insert(
            "SPARKFLASH_HTTP_BIND".to_owned(),
            self.cluster.http_bind.clone(),
        );
        env.insert(
            "SPARKFLASH_BIN".to_owned(),
            self.binary_path().display().to_string(),
        );
        env.insert(
            "SPARKFLASH_LOG_ROOT".to_owned(),
            resolve_against(&self.repo_root(), &self.paths.log_root)
                .display()
                .to_string(),
        );
        env.insert(
            "SPARKFLASH_TP4_BASE_PORT".to_owned(),
            self.cluster.base_port.to_string(),
        );
        env.insert(
            "SPARK_MAX_SEQ_LEN".to_owned(),
            self.runtime.max_context_tokens.to_string(),
        );
        env.insert(
            "SPARKFLASH_LANES".to_owned(),
            self.runtime.max_concurrency.to_string(),
        );
        if !self.cluster.ssh_user.trim().is_empty() {
            env.insert("SSH_USER".to_owned(), self.cluster.ssh_user.clone());
        }
        if !self.cluster.ssh_options.is_empty() {
            env.insert("SSH_OPTS".to_owned(), self.cluster.ssh_options.join(" "));
        }
        let local_ranks = self
            .cluster
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.local)
            .map(|(rank, _)| rank.to_string())
            .collect::<Vec<_>>();
        if !local_ranks.is_empty() {
            env.insert("SPARK_LOCAL_RANKS".to_owned(), local_ranks.join(" "));
        }
        if self.engine.device && !self.model.cubins.trim().is_empty() {
            env.insert("SPARKFLASH_CUBINS".to_owned(), self.model.cubins.clone());
        }
        if !self.paths.nccl_lib.trim().is_empty() {
            env.insert(
                "SPARKFLASH_NCCL_LIB".to_owned(),
                self.paths.nccl_lib.clone(),
            );
        }
        if !self.network.nccl_ib_hca.trim().is_empty() {
            env.insert("NCCL_IB_HCA".to_owned(), self.network.nccl_ib_hca.clone());
        }
        if !self.network.nccl_socket_ifname.trim().is_empty() {
            env.insert(
                "NCCL_SOCKET_IFNAME".to_owned(),
                self.network.nccl_socket_ifname.clone(),
            );
        }
        if let Some(index) = self.network.nccl_ib_gid_index {
            env.insert("NCCL_IB_GID_INDEX".to_owned(), index.to_string());
        }
        for (key, value) in &self.network.env {
            env.insert(key.clone(), value.clone());
        }
        env
    }

    /// The SSH destination for `rank`, `user@host` or just `host`.
    #[must_use]
    pub fn ssh_destination(&self, rank: usize) -> String {
        self.ssh_target(&self.cluster.nodes[rank].ssh)
    }

    /// The SSH destination for `host` as the launcher reaches it: `cluster.ssh_user@host`, unless
    /// the user is empty or `host` already names one.
    #[must_use]
    pub fn ssh_target(&self, host: &str) -> String {
        if self.cluster.ssh_user.trim().is_empty() || host.contains('@') {
            host.to_owned()
        } else {
            format!("{}@{host}", self.cluster.ssh_user)
        }
    }
}

/// Resolves `path` against `root` unless it is already absolute.
#[must_use]
pub fn resolve_against(root: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        root.join(candidate)
    }
}

fn parse_socket_addr(field: &str, value: &str) -> Result<SocketAddr, ConfigError> {
    value.parse::<SocketAddr>().map_err(|_| {
        ConfigError::new(format!(
            "{field} {value:?} is not an IP:port socket address"
        ))
    })
}

fn validate_serve_url(value: &str) -> Result<(), ConfigError> {
    let rest = value
        .strip_prefix("http://")
        .or_else(|| value.strip_prefix("https://"));
    let Some(rest) = rest else {
        return Err(ConfigError::new(format!(
            "web.serve_url {value:?} must start with http:// or https://"
        )));
    };
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.is_empty() {
        return Err(ConfigError::new("web.serve_url must name a host and port"));
    }
    Ok(())
}

/// Shell-quotes one argument for `sh`.
///
/// The result is safe to paste into a remote command line: single quotes protect everything
/// except a literal quote, which is closed, escaped and reopened.
#[must_use]
pub fn shell_quote(value: &str) -> String {
    if value.is_empty() {
        return "''".to_owned();
    }
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"-._/:@%+=,".contains(&byte))
    {
        return value.to_owned();
    }
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('\'');
    for character in value.chars() {
        if character == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(character);
        }
    }
    quoted.push('\'');
    quoted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_ssh_would_read_as_options_are_refused() {
        let mut config = SparkConfig::default();
        config.cluster.nodes[2].ssh = "-oProxyCommand=touch /tmp/x".to_owned();
        assert!(config.validate().is_err());
        config.cluster.nodes[2].ssh = "spark two".to_owned();
        assert!(config.validate().is_err());
        config.cluster.nodes[2].ssh = "spark2".to_owned();
        config.cluster.ssh_user = "-oProxyCommand=x".to_owned();
        assert!(config.validate().is_err());
        config.cluster.ssh_user = "ops".to_owned();
        config
            .network
            .env
            .insert("BAD KEY".to_owned(), "1".to_owned());
        assert!(config.validate().is_err());
        config.network.env.clear();
        config.web.allowed_hosts = vec!["http://console.example".to_owned()];
        assert!(config.validate().is_err());
        config.web.allowed_hosts = vec!["console.example".to_owned()];
        config.validate().expect("valid");
    }

    #[test]
    fn the_defaults_are_a_valid_four_spark_plan() {
        let config = SparkConfig::default();
        config.validate().expect("defaults validate");
        assert_eq!(config.cluster.nodes.len(), WORLD_SIZE);
        let env = config.launch_env();
        assert_eq!(env["SPARK_NODES"], "spark0 spark1 spark2 spark3");
        assert_eq!(env["SPARK_PEERS"], "10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4");
        assert_eq!(env["SPARK_MAX_SEQ_LEN"], MODEL_CONTEXT_TOKENS.to_string());
        assert_eq!(env["SPARKFLASH_LANES"], MAX_CONCURRENCY.to_string());
        assert_eq!(env["NCCL_IB_GID_INDEX"], "3");
        assert_eq!(env["SPARK_LOCAL_RANKS"], "0");
    }

    #[test]
    fn local_ranks_are_omitted_when_every_rank_is_remote() {
        let mut config = SparkConfig::default();
        for node in &mut config.cluster.nodes {
            node.local = false;
        }
        assert!(!config.launch_env().contains_key("SPARK_LOCAL_RANKS"));
    }

    #[test]
    fn toml_round_trip_preserves_every_selection() {
        let mut config = SparkConfig::default();
        config.runtime.max_context_tokens = 262_144;
        config.runtime.max_concurrency = 8;
        config.network.nccl_ib_hca = "rocep1s0f1".to_owned();
        config
            .network
            .env
            .insert("NCCL_NET_GDR_LEVEL".to_owned(), "5".to_owned());
        let text = config.to_toml().expect("encode");
        let parsed = SparkConfig::from_toml(&text).expect("parse");
        assert_eq!(parsed, config);
    }

    #[test]
    fn validation_names_the_bad_field() {
        let mut config = SparkConfig::default();
        config.engine.kind = "vllm".to_owned();
        let error = config.validate().expect_err("unsupported engine");
        assert!(error.message.contains("engine.kind"), "{error}");

        let mut config = SparkConfig::default();
        config.runtime.max_context_tokens = MODEL_CONTEXT_TOKENS + 1;
        let error = config.validate().expect_err("context too wide");
        assert!(error.message.contains("max_context_tokens"), "{error}");

        let mut config = SparkConfig::default();
        config.runtime.max_concurrency = 0;
        let error = config.validate().expect_err("zero concurrency");
        assert!(error.message.contains("max_concurrency"), "{error}");

        let mut config = SparkConfig::default();
        config.cluster.nodes.pop();
        let error = config.validate().expect_err("three nodes");
        assert!(error.message.contains("exactly 4"), "{error}");

        let mut config = SparkConfig::default();
        config.cluster.nodes[1].peer = "not-an-ip".to_owned();
        let error = config.validate().expect_err("bad peer");
        assert!(error.message.contains("not an IP address"), "{error}");

        let mut config = SparkConfig::default();
        config.cluster.nodes[1].peer = config.cluster.nodes[0].peer.clone();
        let error = config.validate().expect_err("duplicate peer");
        assert!(error.message.contains("repeated"), "{error}");

        let mut config = SparkConfig::default();
        config.web.serve_url = "localhost:8000".to_owned();
        let error = config.validate().expect_err("bad URL");
        assert!(error.message.contains("http://"), "{error}");
    }

    #[test]
    fn an_empty_checkpoint_resolves_the_model_id_in_the_hub_cache() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot =
            crate::hf::tests::cache_model(dir.path(), DEFAULT_MODEL_ID, "1bae0e1", "deepseek_v41");
        let mut config = SparkConfig::default();
        config.model.hub_cache = dir.path().display().to_string();
        config.validate().expect("an id is enough");
        assert_eq!(config.resolve_checkpoint().expect("cached"), snapshot);
        let resolved = config.resolved().expect("cached");
        assert_eq!(resolved.model.checkpoint, snapshot.display().to_string());
        assert_eq!(
            resolved.launch_env()["SPARKFLASH_TOKENIZER"],
            snapshot.join("tokenizer.json").display().to_string()
        );

        config.model.id = "org/absent".to_owned();
        let error = config.resolve_checkpoint().expect_err("not cached");
        assert!(
            error
                .message
                .contains("org/absent is not in the Hugging Face cache")
        );

        config.model.checkpoint = "/data/explicit".to_owned();
        assert_eq!(
            config.resolve_checkpoint().expect("explicit"),
            PathBuf::from("/data/explicit")
        );
    }

    #[test]
    fn a_non_hub_id_needs_an_explicit_checkpoint() {
        let mut config = SparkConfig::default();
        config.model.id = "my model".to_owned();
        let error = config.validate().expect_err("no lookup possible");
        assert!(error.message.contains("model.checkpoint"), "{error}");
        config.model.checkpoint = "/data/DeepSeek-V4.1-Flash".to_owned();
        config
            .validate()
            .expect("an explicit checkpoint needs no hub id");
    }

    #[test]
    fn tokenizer_defaults_beside_the_checkpoint() {
        let mut config = SparkConfig::default();
        config.model.checkpoint = "/data/DeepSeek-V4.1-Flash".to_owned();
        assert_eq!(
            config.effective_tokenizer(),
            PathBuf::from("/data/DeepSeek-V4.1-Flash/tokenizer.json")
        );
        config.model.tokenizer = "/srv/tokenizer.json".to_owned();
        assert_eq!(
            config.effective_tokenizer(),
            PathBuf::from("/srv/tokenizer.json")
        );
    }

    #[test]
    fn relative_paths_resolve_against_the_checkout() {
        let mut config = SparkConfig::default();
        config.paths.repo_root = "/opt/sparkflash".to_owned();
        assert_eq!(
            config.binary_path(),
            PathBuf::from("/opt/sparkflash/target/release/sparkflash")
        );
        assert_eq!(
            config.serve_script_path(),
            PathBuf::from("/opt/sparkflash/scripts/serve-four-node.sh")
        );
        assert_eq!(
            config.stats_path(),
            PathBuf::from("/opt/sparkflash/target/web/stats.json")
        );
        config.paths.binary = "/usr/local/bin/sparkflash".to_owned();
        assert_eq!(
            config.binary_path(),
            PathBuf::from("/usr/local/bin/sparkflash")
        );
    }

    #[test]
    fn save_is_atomic_and_load_reads_it_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested/sparkflash.toml");
        let config = SparkConfig::default();
        config.save(&path).expect("save");
        let loaded = SparkConfig::load(&path).expect("load");
        assert_eq!(loaded, config);
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.contains("[model]"));
        assert!(text.contains("max_context_tokens"));
    }

    #[test]
    fn shell_quoting_survives_spaces_and_quotes() {
        assert_eq!(shell_quote("plain"), "plain");
        assert_eq!(
            shell_quote("/data/DeepSeek-V4.1-Flash"),
            "/data/DeepSeek-V4.1-Flash"
        );
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
        assert_eq!(shell_quote(""), "''");
    }

    #[test]
    fn the_checked_in_example_stays_valid() {
        let text = include_str!("../../../sparkflash.toml.example");
        let config = SparkConfig::from_toml(text).expect("example parses and validates");
        assert_eq!(config.runtime.max_context_tokens, MODEL_CONTEXT_TOKENS);
        assert_eq!(config.cluster.nodes.len(), WORLD_SIZE);
        assert!(text.contains("max_concurrency"));
        assert!(text.contains("_sparkflash._tcp.local."));
    }

    #[test]
    fn optional_network_values_are_omitted_when_blank() {
        let mut config = SparkConfig::default();
        config.paths.nccl_lib.clear();
        config.network.nccl_ib_hca.clear();
        config.network.nccl_socket_ifname.clear();
        config.network.nccl_ib_gid_index = None;
        let env = config.launch_env();
        assert!(!env.contains_key("SPARKFLASH_NCCL_LIB"));
        assert!(!env.contains_key("NCCL_IB_HCA"));
        assert!(!env.contains_key("NCCL_SOCKET_IFNAME"));
        assert!(!env.contains_key("NCCL_IB_GID_INDEX"));
    }
}
