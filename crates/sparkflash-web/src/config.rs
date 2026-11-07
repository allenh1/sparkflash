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
//! against the process working directory, and a leading `~/` is the home directory of the account
//! the control plane runs as. Every command is built with the resulting absolute paths, which must
//! name the same place on all four nodes: use the same account name on every Spark.

use std::collections::BTreeMap;
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

/// The model ids this build can serve.
///
/// "Model" means the weights, not the runtime. Adding another checkpoint is a deliberate
/// change here, plus the conversion and engine code that knows how to read it.
pub const SUPPORTED_WEIGHTS: &[&str] = sparkflash_model::WEIGHT_IDS;

/// The engines this build can run the weights with.
///
/// V4.1-Flash and GLM-5.3-Flash are tensor-parallel-4 packs. Qwen3.8-Flash-Next is the TP=1
/// pack engine: it converts its checkpoint into one rank file and boots from that file.
pub const SUPPORTED_ENGINES: &[&str] = &[
    DEFAULT_ENGINE_ID,
    QWEN_ENGINE_ID,
    GLM_ENGINE_ID,
    DYNAMIC_ENGINE_ID,
];

/// The control-plane engine selector for a checkpoint the dynamic host reference serves.
///
/// The weights selector is the checkpoint's own Hugging Face id: a novel family has no registry
/// entry, so the id is the only stable selector it has.
pub const DYNAMIC_ENGINE_ID: &str = "dynamic";

/// The control-plane weight selector for V4.1-Flash.
pub const DEFAULT_WEIGHTS_ID: &str = sparkflash_model::DEFAULT_WEIGHTS_ID;

/// The control-plane engine selector for the TP=4 V4.1 engine.
pub const DEFAULT_ENGINE_ID: &str = "deepseek-v4.1-flash";

/// The canonical Hugging Face id of the checkpoint the default engine reads.
pub const DEFAULT_MODEL_ID: &str = sparkflash_model::DEFAULT_MODEL_ID;

/// The control-plane weight selector for Qwen3.8-Flash-Next (NVFP4).
pub const QWEN_WEIGHTS_ID: &str = sparkflash_model::descriptor::QWEN38_FLASH_NEXT_WEIGHTS_ID;

/// The control-plane weight selector for the official Qwen3.8-Flash-Next FP8 release: the same
/// pack engine and world, with its own weights id.
pub const QWEN_FP8_WEIGHTS_ID: &str =
    sparkflash_model::descriptor::QWEN38_FLASH_NEXT_FP8_WEIGHTS_ID;

/// The control-plane engine selector for the TP=1 Qwen pack.
pub const QWEN_ENGINE_ID: &str = "qwen4-exp";

/// The canonical Hugging Face id of the Qwen3.8-Flash-Next checkpoint.
pub const QWEN_MODEL_ID: &str = sparkflash_model::descriptor::QWEN38_FLASH_NEXT_ID;

/// The control-plane weight selector for the GLM-5.3-Flash FP8 checkpoint.
pub const GLM_WEIGHTS_ID: &str = sparkflash_model::descriptor::GLM53_FLASH_WEIGHTS_ID;

/// The control-plane engine selector for the TP=4 GLM-5.3-Flash pack.
pub const GLM_ENGINE_ID: &str = "glm5-next";

/// The canonical Hugging Face id of the GLM-5.3-Flash checkpoint.
pub const GLM_MODEL_ID: &str = sparkflash_model::descriptor::GLM53_FLASH_ID;

/// The maximum context window of V4.1-Flash.
pub const MODEL_CONTEXT_TOKENS: usize = sparkflash_model::V41_CONTEXT_TOKENS;

/// The maximum context window of Qwen3.8-Flash-Next.
pub const QWEN_CONTEXT_TOKENS: usize = sparkflash_model::descriptor::QWEN38_CONTEXT_TOKENS;

/// The maximum context window of GLM-5.3-Flash.
pub const GLM_CONTEXT_TOKENS: usize = sparkflash_model::descriptor::GLM53_CONTEXT_TOKENS;

/// The tensor-parallel width of the V4.1 and GLM clusters.
pub const WORLD_SIZE: usize = 4;

/// The tensor-parallel width of a TP=1 pack such as Qwen3.8-Flash-Next.
pub const SOLO_WORLD_SIZE: usize = 1;

/// The largest concurrency the current engine is designed for.
pub const MAX_CONCURRENCY: usize = 32;

/// The default config path, relative to the lead node's working directory.
pub const DEFAULT_CONFIG_PATH: &str = "sparkflash.toml";

/// One checkpoint family this build knows how to run, and the selector pair that runs it.
///
/// The Hugging Face cache is the source of model *instances*; this table is the source of model
/// *families*. A cached checkpoint is selectable only when its `config.json` names one of these
/// `model_type`s, and choosing it fixes `model.weights` and `engine.kind` instead of leaving the
/// operator to pair them by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelProfile {
    /// The `model_type` the checkpoint's `config.json` carries.
    pub model_type: &'static str,
    /// The family name the model browser shows beside DeepSeek, Qwen and GLM.
    pub label: &'static str,
    /// The default Hugging Face search term for this family.
    pub search: &'static str,
    /// How many ranks the family runs on.
    pub world_size: usize,
    /// The organization most of the family's checkpoints live under.
    pub org: &'static str,
    /// The control-plane weight selector.
    pub weights: &'static str,
    /// The engine selector that runs those weights.
    pub engine: &'static str,
    /// The canonical Hugging Face id, used for the option shown before the model is downloaded.
    pub id: &'static str,
    /// The model's context ceiling.
    pub context_tokens: usize,
    /// Whether launch needs a per-rank converted file.
    pub requires_conversion: bool,
    /// Whether launch loads the external cubin directory built by scripts/build-cubins.sh.
    pub requires_cubins: bool,
}

/// The deployment fields a model profile needs that the model registry does not own.
struct Deployment {
    /// The engine selector that runs those weights.
    engine: &'static str,
    /// The default Hugging Face search term for this family.
    search: &'static str,
    /// How many ranks the family runs on.
    world_size: usize,
    /// The organization most of the family's checkpoints live under.
    org: &'static str,
    /// Whether launch needs a per-rank converted file.
    requires_conversion: bool,
    /// Whether launch loads the external cubin directory built by scripts/build-cubins.sh.
    requires_cubins: bool,
}

/// The deployment fields for the weights selector, or a panic naming the unregistered family.
fn deployment_for(weights: &str) -> Deployment {
    match weights {
        DEFAULT_WEIGHTS_ID => Deployment {
            engine: DEFAULT_ENGINE_ID,
            search: "DeepSeek-V4.1",
            world_size: WORLD_SIZE,
            org: "deepseek-ai",
            requires_conversion: true,
            requires_cubins: true,
        },
        QWEN_WEIGHTS_ID | QWEN_FP8_WEIGHTS_ID => Deployment {
            engine: QWEN_ENGINE_ID,
            search: "Qwen3.8-Flash",
            world_size: SOLO_WORLD_SIZE,
            org: "Qwen",
            requires_conversion: true,
            requires_cubins: false,
        },
        GLM_WEIGHTS_ID => Deployment {
            engine: GLM_ENGINE_ID,
            search: "GLM-5.3-Flash",
            world_size: WORLD_SIZE,
            org: "zai-org",
            requires_conversion: true,
            requires_cubins: false,
        },
        other => panic!("no web deployment profile for the registry weights {other}"),
    }
}

/// Every model family the web control plane can configure.
///
/// The identity fields (model_type, label, id, weights, context) are the model registry's; only
/// the deployment fields above are the web layer's. A new registry model must add a deployment
/// entry, and a test holds the two sets to the same entries.
pub static MODEL_PROFILES: LazyLock<Vec<ModelProfile>> = LazyLock::new(|| {
    sparkflash_model::MODELS
        .iter()
        .map(|model| {
            let deployment = deployment_for(&model.weights_id);
            ModelProfile {
                model_type: &model.model_type,
                label: &model.label,
                search: deployment.search,
                world_size: deployment.world_size,
                org: deployment.org,
                weights: &model.weights_id,
                engine: deployment.engine,
                id: &model.id,
                context_tokens: model.context_tokens,
                requires_conversion: deployment.requires_conversion,
                requires_cubins: deployment.requires_cubins,
            }
        })
        .collect()
});

/// The profile for a checkpoint's `config.json` `model_type`, including the text-only Qwen
/// alias the pack also releases.
#[must_use]
pub fn profile_for_model_type(model_type: &str) -> Option<&'static ModelProfile> {
    let wanted = match model_type {
        "qwen4_exp" | "qwen4_exp_text" => "qwen4_exp",
        other => other,
    };
    MODEL_PROFILES
        .iter()
        .find(|profile| profile.model_type == wanted)
}

/// The profile for a control-plane weight selector.
#[must_use]
pub fn profile_for_weights(weights: &str) -> Option<&'static ModelProfile> {
    MODEL_PROFILES
        .iter()
        .find(|profile| profile.weights.eq_ignore_ascii_case(weights.trim()))
}

/// The profile for a canonical released Hugging Face id.
#[must_use]
pub fn profile_for_id(id: &str) -> Option<&'static ModelProfile> {
    MODEL_PROFILES
        .iter()
        .find(|profile| profile.id.eq_ignore_ascii_case(id.trim()))
}

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
    /// The root for per-rank converted files, the same path on every node. The console adds a
    /// checkpoint-specific child (repository and resolved-path hash) under it.
    pub converted_dir: String,
    /// The cubin directory built by `scripts/build-cubins.sh`.
    pub cubins: String,
    /// The checkpoint's own window, stored when a cached model is selected.
    ///
    /// A dynamic checkpoint has no registry profile to carry its ceiling, so
    /// [`SparkConfig::max_context_tokens`] falls back to this value instead of the V4.1 default.
    /// It is the last resort, not the first: an explicit `model.checkpoint`'s document wins,
    /// then the cached `model.id`'s document, and only then this stored value. `None` means no
    /// selection has stored one yet; the ceiling is derived from the checkpoint at
    /// `checkpoint` or in the cache when that document is available.
    #[serde(default)]
    pub context_tokens: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            weights: SUPPORTED_WEIGHTS[0].to_owned(),
            id: DEFAULT_MODEL_ID.to_owned(),
            checkpoint: String::new(),
            hub_cache: String::new(),
            tokenizer: String::new(),
            converted_dir: "~/sparkflash/converted".to_owned(),
            cubins: "target/cubins".to_owned(),
            context_tokens: None,
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
    /// Either the four V4.1 ranks or one TP=1 node, in rank order.
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
    /// The directory holding `libnccl.so.2` (`scripts/fetch-nccl.sh` fills `target/deps/nccl`),
    /// empty when NCCL is installed system-wide.
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
            nccl_lib: "target/deps/nccl".to_owned(),
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
        let dynamic = self.is_dynamic();
        if !dynamic && !SUPPORTED_WEIGHTS.contains(&self.model.weights.as_str()) {
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
        if !dynamic {
            let expected_engine = profile_for_weights(&self.model.weights)
                .map_or(DEFAULT_ENGINE_ID, |profile| profile.engine);
            if self.engine.kind != expected_engine {
                return Err(ConfigError::new(format!(
                    "model.weights {:?} runs with engine.kind {expected_engine:?}, not {:?}",
                    self.model.weights, self.engine.kind
                )));
            }
        } else if self.model.weights.trim().is_empty() {
            return Err(ConfigError::new(
                "a dynamic engine needs model.weights (the checkpoint id it serves)",
            ));
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
        self.validate_model_family()?;
        if self.model.converted_dir.trim().is_empty() {
            return Err(ConfigError::new("model.converted_dir must not be empty"));
        }
        if self.requires_cubins() && self.model.cubins.trim().is_empty() {
            return Err(ConfigError::new(
                "model.cubins must not be empty when engine.device is true",
            ));
        }
        let max_context = self.max_context_tokens();
        if !(1..=max_context).contains(&self.runtime.max_context_tokens) {
            return Err(ConfigError::new(format!(
                "runtime.max_context_tokens must be in 1..={max_context} for {:?}, got {}",
                self.model.weights, self.runtime.max_context_tokens
            )));
        }
        if !(1..=MAX_CONCURRENCY).contains(&self.runtime.max_concurrency) {
            return Err(ConfigError::new(format!(
                "runtime.max_concurrency must be in 1..={MAX_CONCURRENCY}, got {}",
                self.runtime.max_concurrency
            )));
        }
        let solo = self.is_qwen() || self.is_dynamic();
        let expected_nodes = if solo { SOLO_WORLD_SIZE } else { WORLD_SIZE };
        if self.cluster.nodes.len() != expected_nodes {
            let topology = if solo {
                "one node at TP=1".to_owned()
            } else {
                format!("{WORLD_SIZE} nodes at TP={WORLD_SIZE}")
            };
            return Err(ConfigError::new(format!(
                "cluster.nodes must list exactly {expected_nodes} node{} for {:?} ({topology}), got {}",
                if expected_nodes == 1 { "" } else { "s" },
                self.model.weights,
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
            if node.peer.trim().is_empty() {
                if !solo {
                    return Err(ConfigError::new(format!(
                        "cluster.nodes[{rank}].peer must name the fast-network address"
                    )));
                }
            } else {
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
        }
        if solo && !self.cluster.nodes[0].local {
            return Err(ConfigError::new(
                "cluster.nodes[0].local must be true: the TP=1 node runs on this Spark",
            ));
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

    /// Refuses a model this build cannot run before anything is downloaded or converted.
    ///
    /// The cache is authoritative for a cached checkpoint: its `config.json` `model_type` must
    /// be one of [`MODEL_PROFILES`], and the saved weight and engine selectors must be the pair
    /// that runs it. An explicit checkpoint is checked the same way when it carries a
    /// `config.json`. A model that is neither cached nor an explicit directory must be one of
    /// the canonical releases, so the download step cannot be pointed at an arbitrary repository.
    fn validate_model_family(&self) -> Result<(), ConfigError> {
        if self.is_dynamic() {
            return self.validate_dynamic_family();
        }
        let model_type = if !self.model.checkpoint.trim().is_empty() {
            let path = self.node_path(&self.model.checkpoint).join("config.json");
            match std::fs::read_to_string(&path) {
                Ok(text) => serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .and_then(|config| config["model_type"].as_str().map(str::to_owned)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                Err(error) => {
                    return Err(ConfigError::new(format!(
                        "cannot read the checkpoint configuration {}: {error}",
                        path.display()
                    )));
                }
            }
        } else if let Some(cached) = self
            .hub_cache_dir()
            .as_deref()
            .and_then(|cache| crate::hf::lookup(cache, &self.model.id))
        {
            if cached.model_type.is_none() {
                // A download writes the large shards before config.json. A canonical id still
                // identifies the family, so selecting it while it downloads is valid.
                if let Some(profile) = profile_for_id(&self.model.id) {
                    if self.model.weights != profile.weights || self.engine.kind != profile.engine {
                        return Err(ConfigError::new(format!(
                            "{} runs with model.weights {:?} and engine.kind {:?}, not {:?}/{:?}",
                            self.model.id,
                            profile.weights,
                            profile.engine,
                            self.model.weights,
                            self.engine.kind
                        )));
                    }
                    return Ok(());
                }
                if !cached.complete {
                    // The repository is still downloading and is not a canonical id. There is no
                    // model_type to check yet; the download step remains the next action.
                    return Ok(());
                }
                return Err(ConfigError::new(format!(
                    "the cached checkpoint {} has no config.json model_type, so this build cannot \
                     tell whether it can run it",
                    self.model.id
                )));
            }
            cached.model_type
        } else {
            let profile = profile_for_id(&self.model.id).ok_or_else(|| {
                ConfigError::new(format!(
                    "model.id {:?} is not in the Hugging Face cache and is not a release this build \
                     knows; download a supported model or select one already in the cache",
                    self.model.id
                ))
            })?;
            if self.model.weights != profile.weights || self.engine.kind != profile.engine {
                return Err(ConfigError::new(format!(
                    "{} runs with model.weights {:?} and engine.kind {:?}, not {:?}/{:?}",
                    self.model.id,
                    profile.weights,
                    profile.engine,
                    self.model.weights,
                    self.engine.kind
                )));
            }
            return Ok(());
        };

        let Some(model_type) = model_type else {
            // An explicit checkpoint without a config.json is a fixture-style directory. Its
            // weights selector is the only evidence available, so keep the existing validation.
            return Ok(());
        };
        let profile = profile_for_model_type(&model_type).ok_or_else(|| {
            ConfigError::new(format!(
                "the checkpoint's model_type {model_type:?} is not one this build can run \
                 (supported: {})",
                MODEL_PROFILES
                    .iter()
                    .map(|profile| profile.model_type)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        if self.model.weights != profile.weights || self.engine.kind != profile.engine {
            return Err(ConfigError::new(format!(
                "the checkpoint's model_type {model_type:?} runs with model.weights {:?} and \
                 engine.kind {:?}, not {:?}/{:?}",
                profile.weights, profile.engine, self.model.weights, self.engine.kind
            )));
        }
        Ok(())
    }

    /// Refuses a dynamic checkpoint the host reference cannot execute.
    ///
    /// An explicit `model.checkpoint` comes first: its `config.json` is parsed and checked
    /// against the same `HOST_REFERENCE_PARTS` set the CLI uses, and its shards are inspected
    /// for storage-only quantization. Otherwise the cached entry for `model.id` answers. There
    /// is no profile-family check: a dynamic checkpoint is deliberately one the registry does
    /// not name.
    fn validate_dynamic_family(&self) -> Result<(), ConfigError> {
        if !self.model.checkpoint.trim().is_empty() {
            let directory = self.node_path(&self.model.checkpoint);
            let path = directory.join("config.json");
            let text = std::fs::read_to_string(&path).map_err(|error| {
                ConfigError::new(format!(
                    "cannot read the dynamic checkpoint configuration {}: {error}",
                    path.display()
                ))
            })?;
            let architecture =
                sparkflash_model::Architecture::from_json(&text).map_err(|error| {
                    ConfigError::new(format!(
                        "{} is not a usable architecture: {error}",
                        path.display()
                    ))
                })?;
            if let Some(reason) = sparkflash_ops::parts::host_reference_refusal(&architecture) {
                return Err(ConfigError::new(format!(
                    "the checkpoint's model_type {:?}: {reason}",
                    architecture.family
                )));
            }
            // The declared-format gate above needs only the document. The storage half needs
            // the tensor headers, exactly as the CLI backend reads them, so an explicit
            // checkpoint is judged on its shards too rather than only on its declaration.
            let source =
                sparkflash_weights::safetensors::open_checkpoint(&directory).map_err(|error| {
                    ConfigError::new(format!(
                        "cannot read the dynamic checkpoint shards in {}: {error}",
                        directory.display()
                    ))
                })?;
            if let Some(reason) = sparkflash_ops::parts::host_reference_storage_quantization_refusal(
                &architecture,
                &source,
            ) {
                return Err(ConfigError::new(format!(
                    "the checkpoint's model_type {:?}: {reason}",
                    architecture.family
                )));
            }
            return Ok(());
        }
        let cached = self
            .hub_cache_dir()
            .as_deref()
            .and_then(|cache| crate::hf::lookup(cache, &self.model.id))
            .ok_or_else(|| {
                ConfigError::new(format!(
                    "model.id {:?} is not in the Hugging Face cache; download it or set model.checkpoint",
                    self.model.id
                ))
            })?;
        if !cached.supported {
            return Err(ConfigError::new(format!(
                "{} is not runnable by the dynamic host path: {}",
                cached.id,
                if cached.missing_parts.is_empty() {
                    "no supported parts were found".to_owned()
                } else {
                    cached.missing_parts.join(", ")
                }
            )));
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

    /// Point `paths.binary` at the executable that is running this control plane when the
    /// configured path does not exist.
    ///
    /// This is what makes `cargo run --bin sparkflash -- --web` self-contained: the debug (or
    /// release) binary cargo just built is the one the Install step copies, instead of a
    /// `target/release/sparkflash` that was never built. An explicitly built path that exists is
    /// left alone.
    pub fn adopt_running_binary(&mut self, executable: &Path) -> bool {
        if self.binary_path().is_file() || !executable.is_file() {
            return false;
        }
        self.paths.binary = executable.display().to_string();
        if let Some(root) = executable
            .ancestors()
            .skip(1)
            .find(|directory| directory.join("Cargo.toml").is_file())
        {
            self.paths.repo_root = root.display().to_string();
        }
        true
    }

    /// If the selected model is not a complete cache snapshot, switch to the best complete cached
    /// model in the same weight family.
    ///
    /// A fresh checkout keeps the canonical id in its defaults; on a Spark whose cache already
    /// holds `...-1017`, that would make the Setup page offer a 510 GB download that is already
    /// on disk. Returns the selected entry when it changed the configuration.
    pub fn prefer_cached_model(&mut self) -> Option<crate::hf::CachedModel> {
        let cache = self.hub_cache_dir()?;
        if crate::hf::lookup(&cache, &self.model.id).is_some_and(|model| model.complete) {
            return None;
        }
        let profile = profile_for_weights(&self.model.weights)?;
        let models = crate::hf::list(&cache).ok()?;
        let preferred =
            crate::hf::preferred_cached(&models, &self.model.weights, profile.id)?.clone();
        if preferred.id == self.model.id {
            return None;
        }
        self.model.id = preferred.id.clone();
        self.model.context_tokens = preferred.context_tokens;
        if let Some(weights) = preferred.weights.as_deref() {
            self.model.weights = weights.to_owned();
        }
        if let Some(engine) = preferred.engine.as_deref() {
            self.engine.kind = engine.to_owned();
        }
        if let Some(context) = preferred.context_tokens {
            self.runtime.max_context_tokens = self.runtime.max_context_tokens.min(context);
        }
        self.model.checkpoint.clear();
        self.model.tokenizer.clear();
        Some(preferred)
    }

    /// A copy that selects `model` from the Hugging Face cache and resolves its checkpoint.
    ///
    /// The architecture, engine and context ceiling come from the cached checkpoint's profile,
    /// never from the caller. This is the one place the model shelf and the swap action build a
    /// target configuration, so a prepared model cannot be paired with the wrong engine.
    ///
    /// # Errors
    /// When the cache entry is not runnable by this build, or the target cannot share this
    /// cluster (a TP=1 pack on a four-Spark plan, for example), or its checkpoint does not
    /// resolve.
    pub fn for_cached_model(&self, model: &crate::hf::CachedModel) -> Result<Self, ConfigError> {
        if !model.supported {
            return Err(ConfigError::new(format!(
                "{} is not a model type this build can run",
                model.id
            )));
        }
        if !model.complete {
            return Err(ConfigError::new(format!(
                "{} is still downloading, so it cannot be selected yet",
                model.id
            )));
        }
        let weights = model.weights.as_deref().ok_or_else(|| {
            ConfigError::new(format!(
                "{} has no control-plane weights selector",
                model.id
            ))
        })?;
        let engine = model.engine.as_deref().ok_or_else(|| {
            ConfigError::new(format!("{} has no control-plane engine selector", model.id))
        })?;
        let mut config = self.clone();
        config.model.id = model.id.clone();
        config.model.context_tokens = model.context_tokens;
        config.model.checkpoint.clear();
        config.model.tokenizer.clear();
        config.model.weights = weights.to_owned();
        config.engine.kind = engine.to_owned();
        if config.is_dynamic() && !config.is_solo() {
            // The host reference is TP1; selecting it from a four-Spark plan moves the launch to
            // this node rather than refusing a model the cache can already serve.
            let mut node = config.cluster.nodes[0].clone();
            node.peer.clear();
            node.local = true;
            config.cluster.nodes = vec![node];
        }
        if let Some(context) = model.context_tokens {
            config.runtime.max_context_tokens = config.runtime.max_context_tokens.min(context);
        }
        config.validate()?;
        config.resolved()
    }

    /// Whether this plan runs the dynamic host reference under a checkpoint's own id.
    #[must_use]
    pub fn is_dynamic(&self) -> bool {
        self.engine.kind == DYNAMIC_ENGINE_ID
    }

    /// Whether the selected weights are the TP=1 Qwen pack.
    #[must_use]
    pub fn is_qwen(&self) -> bool {
        matches!(
            self.model.weights.as_str(),
            QWEN_WEIGHTS_ID | QWEN_FP8_WEIGHTS_ID
        )
    }

    /// Whether the selected weights are the TP=4 GLM-5.3-Flash pack.
    #[must_use]
    pub fn is_glm(&self) -> bool {
        self.model.weights == GLM_WEIGHTS_ID
    }

    /// The model selector a packed four-rank launcher passes as its --model argument.
    ///
    /// The V4.1 engine is the launcher's default and takes no --model; a packed model names its
    /// stable weight selector so a fine-tune of the same family still dispatches to its arm.
    #[must_use]
    pub fn packed_model(&self) -> Option<&'static str> {
        let profile = profile_for_weights(&self.model.weights)?;
        (profile.weights != DEFAULT_WEIGHTS_ID).then_some(profile.weights)
    }

    /// Whether this configuration runs on one node instead of the four-node cluster.
    #[must_use]
    pub fn is_solo(&self) -> bool {
        self.cluster.nodes.len() == SOLO_WORLD_SIZE
    }

    /// The number of ranks this configuration names.
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.cluster.nodes.len()
    }

    /// The context ceiling of the selected weights.
    ///
    /// A registry profile carries its family's ceiling. A dynamic checkpoint has no profile;
    /// its ceiling is the document's own `max_position_embeddings`, read in that precedence:
    /// an explicit `model.checkpoint` first, then the cached `model.id`, then the value a
    /// selection stored in [`ModelConfig::context_tokens`]. Falling back to the V4.1 default
    /// would let the web accept a window the CLI refuses and show a limit the checkpoint does
    /// not have.
    #[must_use]
    pub fn max_context_tokens(&self) -> usize {
        if self.is_dynamic() {
            // An explicit checkpoint is authoritative: it is the directory launch and validation
            // read, so a stored selection's ceiling or a cached id must not outvote it.
            if let Some(context) = self.explicit_checkpoint_context() {
                return context;
            }
            if let Some(context) = self
                .hub_cache_dir()
                .as_deref()
                .and_then(|cache| crate::hf::lookup(cache, &self.model.id))
                .and_then(|model| model.context_tokens)
                .filter(|context| *context > 0)
            {
                return context;
            }
            if let Some(context) = self.model.context_tokens.filter(|context| *context > 0) {
                return context;
            }
            return MODEL_CONTEXT_TOKENS;
        }
        profile_for_weights(&self.model.weights)
            .map_or(MODEL_CONTEXT_TOKENS, |profile| profile.context_tokens)
    }

    /// The window an explicit `model.checkpoint`'s own `config.json` declares.
    fn explicit_checkpoint_context(&self) -> Option<usize> {
        if self.model.checkpoint.trim().is_empty() {
            return None;
        }
        let path = self.node_path(&self.model.checkpoint).join("config.json");
        let text = std::fs::read_to_string(path).ok()?;
        let architecture = sparkflash_model::Architecture::from_json(&text).ok()?;
        (architecture.max_position_embeddings > 0).then_some(architecture.max_position_embeddings)
    }

    /// Clears a stored selection ceiling that no longer describes the selected checkpoint.
    ///
    /// The selection is `model.checkpoint` first, then a cached `model.id`. If that source's
    /// document names a different window, the stored `context_tokens` came from an earlier
    /// selection and must not be used as the ceiling. The next [Self::max_context_tokens] reads
    /// the authoritative document instead.
    pub fn discard_stale_context_tokens(&mut self) {
        if !self.is_dynamic() {
            return;
        }
        let authoritative = if self.model.checkpoint.trim().is_empty() {
            self.hub_cache_dir()
                .as_deref()
                .and_then(|cache| crate::hf::lookup(cache, &self.model.id))
                .and_then(|model| model.context_tokens)
        } else {
            self.explicit_checkpoint_context()
        };
        if let Some(context) = authoritative.filter(|context| *context > 0)
            && self.model.context_tokens != Some(context)
        {
            self.model.context_tokens = None;
        }
    }

    /// Whether launch must run the one-time per-rank conversion first.
    ///
    /// Every pack boots from the rank file its converter writes: V4.1 and GLM at four ranks,
    /// Qwen3.8-Flash-Next at one. [Self::requires_conversion_manifest] says whether the file
    /// alone is the completion marker.
    #[must_use]
    pub fn requires_conversion(&self) -> bool {
        if self.is_dynamic() {
            // The dynamic host reference reads the checkpoint's safetensors directly. A rank
            // file can be written with `sparkflash convert`, but launch does not need one.
            return false;
        }
        profile_for_weights(&self.model.weights).is_none_or(|profile| profile.requires_conversion)
    }

    /// Whether the engine needs the external cubin directory.
    #[must_use]
    pub fn requires_cubins(&self) -> bool {
        if self.is_dynamic() {
            return false;
        }
        self.engine.device
            && profile_for_weights(&self.model.weights)
                .is_none_or(|profile| profile.requires_cubins)
    }

    /// Whether the node check should require a conversion manifest beside this rank's file.
    ///
    /// The GLM and Qwen converters write one rank file and no manifest; that file is the
    /// completion marker. V4.1 writes a manifest after each rank file.
    #[must_use]
    pub fn requires_conversion_manifest(&self) -> bool {
        !(self.is_glm() || self.is_qwen() || self.is_dynamic())
    }

    /// The single-node launcher script, next to the four-node one.
    #[must_use]
    pub fn local_serve_script_path(&self) -> PathBuf {
        self.repo_root().join("scripts/serve-single-node.sh")
    }

    /// The Hugging Face hub cache `model.id` is looked up in.
    #[must_use]
    pub fn hub_cache_dir(&self) -> Option<PathBuf> {
        if self.model.hub_cache.trim().is_empty() {
            crate::hf::default_cache_dir()
        } else {
            Some(self.node_path(&self.model.hub_cache))
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
            return Ok(self.node_path(&self.model.checkpoint));
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
            self.node_path(&self.model.checkpoint)
                .join("tokenizer.json")
        } else {
            self.node_path(&self.model.tokenizer)
        }
    }

    /// The checkout root, absolute: `.` is the directory the control plane was started in.
    #[must_use]
    pub fn repo_root(&self) -> PathBuf {
        let root = expand_home(&self.paths.repo_root);
        std::path::absolute(&root).unwrap_or(root)
    }

    /// A configured path as every node sees it: `~/` expanded, a relative path under the checkout.
    #[must_use]
    pub fn node_path(&self, value: &str) -> PathBuf {
        let path = expand_home(value);
        if path.is_absolute() {
            path
        } else {
            self.repo_root().join(path)
        }
    }

    /// The binary path, resolved against `repo_root` when it is relative.
    #[must_use]
    pub fn binary_path(&self) -> PathBuf {
        self.node_path(&self.paths.binary)
    }

    /// The launcher script path, resolved against `repo_root` when it is relative.
    #[must_use]
    pub fn serve_script_path(&self) -> PathBuf {
        self.node_path(&self.paths.serve_script)
    }

    /// The telemetry file path, resolved against `repo_root` when it is relative.
    #[must_use]
    pub fn stats_path(&self) -> PathBuf {
        self.node_path(&self.stats.persist_path)
    }

    /// The converted weights directory for the selected checkpoint.
    ///
    /// `model.converted_dir` is a root, not the directory itself: each checkpoint gets its own
    /// child named after the repository and a hash of the resolved checkpoint path. Swapping
    /// models therefore cannot silently reuse another checkpoint's rank files; the new path is
    /// empty, and the node check reports the conversion as missing until it runs again.
    #[must_use]
    pub fn converted_path(&self) -> PathBuf {
        self.node_path(&self.model.converted_dir)
            .join(self.conversion_scope())
    }

    /// The per-checkpoint child under `model.converted_dir`.
    #[must_use]
    pub fn conversion_scope(&self) -> String {
        let mut scope = if crate::hf::is_model_id(&self.model.id) {
            crate::hf::repo_folder(&self.model.id)
        } else {
            sanitize_scope(&self.model.id)
        };
        if scope.is_empty() {
            scope.push_str("model");
        }
        // Always identify the resolved checkpoint path, not the spelling in the file. A command
        // built from `resolved()` has `model.checkpoint` filled in, while the node check uses
        // the unresolved config; both must name the same directory. Hashing the resolved path
        // also gives a re-download or a same-basename explicit checkpoint its own directory.
        if let Ok(checkpoint) = self.resolve_checkpoint() {
            let marker = short_hash(&checkpoint.to_string_lossy());
            let suffix = format!("--{marker}");
            if !marker.is_empty()
                && !scope
                    .to_ascii_lowercase()
                    .ends_with(&suffix.to_ascii_lowercase())
            {
                scope.push_str(&suffix);
            }
        }
        scope
    }

    /// The kernel cubin directory.
    #[must_use]
    pub fn cubins_path(&self) -> PathBuf {
        self.node_path(&self.model.cubins)
    }

    /// The directory holding `libnccl.so.2`, when one is configured.
    #[must_use]
    pub fn nccl_lib_path(&self) -> Option<PathBuf> {
        let value = self.paths.nccl_lib.trim();
        (!value.is_empty()).then(|| self.node_path(value))
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
            self.node_path(&self.model.checkpoint).display().to_string(),
        );
        env.insert(
            "SPARKFLASH_TOKENIZER".to_owned(),
            self.effective_tokenizer().display().to_string(),
        );
        env.insert(
            "SPARKFLASH_CONVERTED_DIR".to_owned(),
            self.converted_path().display().to_string(),
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
            self.node_path(&self.paths.log_root).display().to_string(),
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
        // A packed four-rank model is selected by --model and sizes its window with --context;
        // the V4.1 engine is the launcher's default and reads SPARK_MAX_SEQ_LEN instead.
        if let Some(model) = self.packed_model() {
            env.insert("SPARKFLASH_MODEL".to_owned(), model.to_owned());
            env.insert(
                "SPARKFLASH_CONTEXT".to_owned(),
                self.runtime.max_context_tokens.to_string(),
            );
        }
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
        if self.requires_cubins() && !self.model.cubins.trim().is_empty() {
            env.insert(
                "SPARKFLASH_CUBINS".to_owned(),
                self.cubins_path().display().to_string(),
            );
        }
        if let Some(nccl_lib) = self.nccl_lib_path() {
            env.insert(
                "SPARKFLASH_NCCL_LIB".to_owned(),
                nccl_lib.display().to_string(),
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

    /// The environment the single-node launcher receives.
    ///
    /// A TP=1 pack runs in this process's place: no SSH, no collective, and no converted weights.
    /// The script only needs the binary, the checkpoint, the tokenizer, the HTTP bind and the
    /// context window. Extra rank environment is passed through unchanged.
    #[must_use]
    pub fn local_launch_env(&self) -> BTreeMap<String, String> {
        let mut env = BTreeMap::new();
        let checkpoint = self
            .resolve_checkpoint()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        env.insert("SPARKFLASH_MODEL".to_owned(), self.model.weights.clone());
        env.insert("SPARKFLASH_CHECKPOINT".to_owned(), checkpoint);
        env.insert(
            "SPARKFLASH_TOKENIZER".to_owned(),
            self.effective_tokenizer().display().to_string(),
        );
        env.insert(
            "SPARKFLASH_CONTEXT".to_owned(),
            self.runtime.max_context_tokens.to_string(),
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
            self.node_path(&self.paths.log_root).display().to_string(),
        );
        // A converted Qwen rank file is one slab read instead of 77.8 GiB of per-tensor uploads.
        // The checkpoint stays the configuration, tokenizer and host-paged PLE source.
        if self.is_qwen() {
            env.insert(
                "SPARKFLASH_QWEN_CONVERTED_DIR".to_owned(),
                self.converted_path().display().to_string(),
            );
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

/// Make a model id or directory name safe as one path component.
/// A stable short digest for a path used as one converted-weights directory component.
fn short_hash(value: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn sanitize_scope(value: &str) -> String {
    let mut result = String::with_capacity(value.len());
    let mut last_was_dash = false;
    for character in value.trim().chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-') {
            result.push(character);
            last_was_dash = false;
        } else if !last_was_dash {
            result.push('-');
            last_was_dash = true;
        }
    }
    result.trim_matches(['-', '.']).to_owned()
}

/// `path` with a leading `~/` (or a bare `~`) replaced by `$HOME`.
#[must_use]
pub fn expand_home(path: &str) -> PathBuf {
    let home = || std::env::var_os("HOME").map(PathBuf::from);
    if path == "~" {
        return home().unwrap_or_else(|| PathBuf::from(path));
    }
    match (path.strip_prefix("~/"), home()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
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
    fn model_profiles_take_their_identity_from_the_model_registry() {
        assert_eq!(MODEL_PROFILES.len(), sparkflash_model::MODELS.len());
        assert_eq!(SUPPORTED_WEIGHTS, sparkflash_model::WEIGHT_IDS);
        for model in sparkflash_model::MODELS.iter() {
            let profile = profile_for_weights(&model.weights_id).unwrap_or_else(|| {
                panic!(
                    "the registry weights {} has no web profile",
                    model.weights_id
                )
            });
            assert_eq!(profile.id, model.id);
            assert_eq!(profile.weights, model.weights_id);
            assert_eq!(profile.model_type, model.model_type);
            assert_eq!(profile.label, model.label);
            assert_eq!(profile.context_tokens, model.context_tokens);
            assert_eq!(profile_for_id(&model.id), Some(profile));
            // Two releases share the qwen4_exp model_type (NVFP4 and FP8); a checkpoint's own
            // document names the family, and either release's profile is the family's.
            let family = profile_for_model_type(&model.model_type)
                .unwrap_or_else(|| panic!("model_type {} has no profile", model.model_type));
            assert_eq!(family.model_type, model.model_type);
        }
    }

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
    fn a_solo_qwen_plan_validates_and_uses_the_local_launcher() {
        let mut config = SparkConfig::default();
        config.model.weights = QWEN_WEIGHTS_ID.to_owned();
        config.model.id = QWEN_MODEL_ID.to_owned();
        config.model.checkpoint = "/data/Qwen3.8-Flash-Next-NVFP4".to_owned();
        config.model.cubins.clear();
        config.engine.kind = QWEN_ENGINE_ID.to_owned();
        config.runtime.max_context_tokens = QWEN_CONTEXT_TOKENS;
        config.cluster.nodes = vec![NodeConfig::new("this-spark.local", "", true)];
        config.validate().expect("solo Qwen validates");
        assert!(config.is_qwen());
        assert!(config.is_solo());
        assert_eq!(config.world_size(), 1);
        assert!(config.requires_conversion());
        assert!(!config.requires_conversion_manifest());
        assert!(!config.requires_cubins());
        assert_eq!(config.max_context_tokens(), QWEN_CONTEXT_TOKENS);
        let env = config.local_launch_env();
        assert_eq!(env["SPARKFLASH_MODEL"], QWEN_WEIGHTS_ID);
        assert_eq!(
            env["SPARKFLASH_QWEN_CONVERTED_DIR"],
            config.converted_path().display().to_string()
        );
        assert_eq!(
            env["SPARKFLASH_CHECKPOINT"],
            "/data/Qwen3.8-Flash-Next-NVFP4"
        );
        assert_eq!(env["SPARKFLASH_CONTEXT"], QWEN_CONTEXT_TOKENS.to_string());
        assert!(!env.contains_key("SPARK_NODES"));

        // The tensor-parallel sizes are not interchangeable.
        config.cluster.nodes = SparkConfig::default().cluster.nodes;
        let error = config.validate().expect_err("Qwen on four nodes");
        assert!(error.message.contains("exactly 1"), "{error}");

        let mut four = SparkConfig::default();
        four.cluster.nodes = vec![NodeConfig::new("this-spark.local", "", true)];
        let error = four.validate().expect_err("V4.1 on one node");
        assert!(error.message.contains("exactly 4"), "{error}");
    }

    #[test]
    fn a_glm_plan_validates_and_uses_the_packed_ranks() {
        assert_eq!(
            profile_for_model_type("glm5_next").map(|profile| profile.weights),
            Some(GLM_WEIGHTS_ID)
        );
        let mut config = SparkConfig::default();
        config.model.weights = GLM_WEIGHTS_ID.to_owned();
        config.model.id = GLM_MODEL_ID.to_owned();
        config.model.checkpoint = "/data/GLM-5.3-Flash".to_owned();
        config.model.cubins.clear();
        config.engine.kind = GLM_ENGINE_ID.to_owned();
        config.runtime.max_context_tokens = GLM_CONTEXT_TOKENS;
        config
            .validate()
            .expect("the GLM four-Spark plan validates");
        assert!(config.is_glm());
        assert!(!config.is_qwen());
        assert!(!config.is_solo());
        assert_eq!(config.world_size(), WORLD_SIZE);
        assert!(config.requires_conversion());
        assert!(!config.requires_cubins());
        assert!(!config.requires_conversion_manifest());
        assert_eq!(config.max_context_tokens(), GLM_CONTEXT_TOKENS);
        let env = config.launch_env();
        assert_eq!(env["SPARKFLASH_MODEL"], GLM_WEIGHTS_ID);
        assert_eq!(env["SPARKFLASH_CONTEXT"], GLM_CONTEXT_TOKENS.to_string());
        assert_eq!(env["SPARK_MAX_SEQ_LEN"], GLM_CONTEXT_TOKENS.to_string());
        assert!(!env.contains_key("SPARKFLASH_CUBINS"));
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
    fn node_paths_are_absolute_with_home_expanded() {
        let home = std::env::var("HOME").expect("HOME is set for tests");
        let mut config = SparkConfig::default();
        config.paths.repo_root = "/srv/sparkflash".to_owned();
        let converted =
            Path::new(&home).join("sparkflash/converted/models--deepseek-ai--DeepSeek-V4.1-Flash");
        assert_eq!(config.converted_path(), converted);
        assert_eq!(
            config.cubins_path(),
            PathBuf::from("/srv/sparkflash/target/cubins")
        );
        assert_eq!(
            config.nccl_lib_path(),
            Some(PathBuf::from("/srv/sparkflash/target/deps/nccl"))
        );
        let env = config.launch_env();
        assert_eq!(
            env["SPARKFLASH_CONVERTED_DIR"],
            converted.display().to_string()
        );
        assert_eq!(env["SPARKFLASH_CUBINS"], "/srv/sparkflash/target/cubins");
        assert_eq!(
            env["SPARKFLASH_NCCL_LIB"],
            "/srv/sparkflash/target/deps/nccl"
        );

        config.paths.repo_root = ".".to_owned();
        let cwd = std::env::current_dir().expect("cwd");
        assert_eq!(config.repo_root(), cwd);
        assert_eq!(config.binary_path(), cwd.join("target/release/sparkflash"));
        assert_eq!(expand_home("~"), PathBuf::from(&home));
        assert_eq!(expand_home("~other/x"), PathBuf::from("~other/x"));
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
    fn a_cached_checkpoint_must_match_the_selected_model_family() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path();
        crate::hf::tests::cache_model(cache, DEFAULT_MODEL_ID, "abc123", "deepseek_v41");
        let mut config = SparkConfig::default();
        config.model.hub_cache = cache.display().to_string();
        config.validate().expect("the supported pair validates");

        config.model.weights = QWEN_WEIGHTS_ID.to_owned();
        config.engine.kind = QWEN_ENGINE_ID.to_owned();
        let error = config
            .validate()
            .expect_err("a V4.1 checkpoint cannot run Qwen");
        assert!(
            error.message.contains("model.weights") && error.message.contains("engine.kind"),
            "{error}"
        );

        let mut unknown = SparkConfig::default();
        unknown.model.hub_cache = cache.display().to_string();
        unknown.model.id = "org/llama".to_owned();
        crate::hf::tests::cache_model(cache, "org/llama", "def456", "llama");
        let error = unknown.validate().expect_err("an unsupported checkpoint");
        assert!(error.message.contains("model_type"), "{error}");

        let mut absent = SparkConfig::default();
        absent.model.hub_cache = cache.display().to_string();
        absent.model.id = "org/not-downloaded".to_owned();
        let error = absent.validate().expect_err("an unknown uncached model");
        assert!(
            error.message.contains("not in the Hugging Face cache"),
            "{error}"
        );
    }

    #[test]
    fn a_canonical_model_is_valid_while_its_download_is_incomplete() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot =
            crate::hf::tests::cache_model(dir.path(), DEFAULT_MODEL_ID, "abc123", "deepseek_v41");
        std::fs::remove_file(snapshot.join("config.json")).expect("drop config");
        let mut config = SparkConfig::default();
        config.model.hub_cache = dir.path().display().to_string();
        config
            .validate()
            .expect("the selected canonical model is still identifiable");
    }

    #[test]
    fn a_fresh_default_prefers_a_complete_cached_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::hf::tests::cache_model(
            dir.path(),
            "deepseek-ai/DeepSeek-V4.1-Flash-1017",
            "aaa",
            "deepseek_v41",
        );
        crate::hf::tests::cache_model(
            dir.path(),
            "pqhaz/DeepSeek-V4.1-Flash-1017-heretic",
            "bbb",
            "deepseek_v41",
        );
        let mut config = SparkConfig::default();
        config.model.hub_cache = dir.path().display().to_string();
        let selected = config.prefer_cached_model().expect("a cached model");
        assert_eq!(selected.id, "deepseek-ai/DeepSeek-V4.1-Flash-1017");
        assert_eq!(config.model.id, selected.id);
        assert_eq!(config.model.weights, DEFAULT_WEIGHTS_ID);
        assert_eq!(config.engine.kind, DEFAULT_ENGINE_ID);
        assert_eq!(config.model.checkpoint, "");
        config
            .validate()
            .expect("the selected cache model validates");
    }

    #[test]
    fn converted_weights_are_scoped_to_the_selected_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path();
        crate::hf::tests::cache_model(cache, DEFAULT_MODEL_ID, "aaaaaaaa", "deepseek_v41");
        crate::hf::tests::cache_model(cache, "org/finetune", "bbbbbbbb", "deepseek_v41");
        let mut config = SparkConfig::default();
        config.model.hub_cache = cache.display().to_string();
        let first = config.converted_path();
        assert!(
            first
                .display()
                .to_string()
                .contains("models--deepseek-ai--DeepSeek-V4.1-Flash--")
        );
        // The conversion command is built from `resolved()`; the node check is not. They must
        // agree on the directory or every finished conversion is reported missing.
        assert_eq!(first, config.resolved().expect("cached").converted_path());
        config.model.id = "org/finetune".to_owned();
        let second = config.converted_path();
        assert!(
            second
                .display()
                .to_string()
                .contains("models--org--finetune--")
        );
        assert_ne!(first, second);

        // Two explicit checkpoints with the same basename must not share converted weights.
        let mut explicit = SparkConfig::default();
        explicit.model.checkpoint = "/data/one/model".to_owned();
        let one = explicit.converted_path();
        explicit.model.checkpoint = "/data/two/model".to_owned();
        let two = explicit.converted_path();
        assert_ne!(one, two);
    }

    #[test]
    fn a_missing_configured_binary_adopts_the_running_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("Cargo.toml"), "[workspace]").expect("manifest");
        let executable = dir.path().join("target/debug/sparkflash");
        std::fs::create_dir_all(executable.parent().expect("parent")).expect("target");
        std::fs::write(&executable, "binary").expect("binary");
        let mut config = SparkConfig::default();
        config.paths.binary = dir
            .path()
            .join("target/release/sparkflash")
            .display()
            .to_string();
        config.paths.repo_root = "/not/the/checkout".to_owned();
        assert!(config.adopt_running_binary(&executable));
        assert_eq!(config.binary_path(), executable);
        assert_eq!(config.repo_root(), dir.path());
        assert!(!config.adopt_running_binary(&executable));
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
    fn a_host_runnable_cached_checkpoint_is_selectable_with_the_dynamic_engine() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snapshot = crate::hf::tests::cache_config(
            dir.path(),
            "acme/Custom-Flash-Test",
            "abc123",
            &crate::hf::tests::custom_config(),
        );
        // Selecting a cached model resolves its checkpoint, and validation reopens the shards.
        sparkflash_weights::safetensors::write_checkpoint(
            &snapshot,
            &[(
                sparkflash_weights::TensorSpec {
                    name: "model.embed_tokens.weight".to_owned(),
                    dtype: sparkflash_weights::DType::BF16,
                    shape: vec![1],
                },
                vec![0; 2],
            )],
            1,
        )
        .expect("the cached checkpoint shard");
        let mut config = SparkConfig::default();
        config.model.hub_cache = dir.path().display().to_string();
        let model = crate::hf::lookup(dir.path(), "acme/Custom-Flash-Test").expect("cache entry");
        let selected = config
            .for_cached_model(&model)
            .expect("the acme fixture is selectable");
        assert!(selected.is_dynamic());
        assert!(selected.is_solo());
        assert_eq!(selected.model.id, "acme/Custom-Flash-Test");
        assert_eq!(selected.model.weights, "acme/Custom-Flash-Test");
        assert_eq!(selected.engine.kind, DYNAMIC_ENGINE_ID);
        assert!(!selected.requires_conversion());
        assert!(!selected.requires_cubins());
        assert!(!selected.requires_conversion_manifest());
        assert_eq!(selected.runtime.max_context_tokens, 128);
        // The selector has no registry profile, so the ceiling comes from the checkpoint's own
        // 128-token window rather than the V4.1 default. A 129-token window must be refused.
        assert_eq!(selected.max_context_tokens(), 128);
        let mut too_wide = selected.clone();
        too_wide.runtime.max_context_tokens = 129;
        let error = too_wide
            .validate()
            .expect_err("the checkpoint window caps the runtime");
        assert!(error.message.contains("max_context_tokens"), "{error}");
        assert_eq!(
            selected.local_launch_env()["SPARKFLASH_MODEL"],
            "acme/Custom-Flash-Test"
        );
    }

    #[test]
    fn an_explicit_checkpoint_window_beats_a_cached_id_and_stored_ceiling() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = dir.path().join("hub");
        std::fs::create_dir_all(&cache).expect("cache");
        let mut cached = crate::hf::tests::custom_config();
        cached["max_position_embeddings"] = serde_json::json!(262_144);
        crate::hf::tests::cache_config(&cache, "acme/Wide-Cached", "abc123", &cached);
        let explicit = dir.path().join("explicit");
        std::fs::create_dir_all(&explicit).expect("explicit checkpoint");
        std::fs::write(
            explicit.join("config.json"),
            serde_json::to_vec(&crate::hf::tests::custom_config()).expect("config JSON"),
        )
        .expect("config");
        sparkflash_weights::safetensors::write_checkpoint(
            &explicit,
            &[(
                sparkflash_weights::TensorSpec {
                    name: "model.embed_tokens.weight".to_owned(),
                    dtype: sparkflash_weights::DType::BF16,
                    shape: vec![1],
                },
                vec![0; 2],
            )],
            1,
        )
        .expect("the explicit checkpoint shard");

        let mut config = SparkConfig::default();
        config.model.hub_cache = cache.display().to_string();
        config.model.id = "acme/Wide-Cached".to_owned();
        config.model.weights = "acme/Wide-Cached".to_owned();
        config.model.checkpoint = explicit.display().to_string();
        config.model.context_tokens = Some(262_144);
        config.engine.kind = DYNAMIC_ENGINE_ID.to_owned();
        config.runtime.max_context_tokens = 200_000;
        config.cluster.nodes = vec![NodeConfig::new("this-spark.local", "", true)];

        // The explicit checkpoint declares 128 positions. The cached id and the stored field
        // still say 262144; the directory launch and validation read must win.
        assert_eq!(config.max_context_tokens(), 128);
        let error = config
            .validate()
            .expect_err("200000 exceeds the explicit checkpoint's 128-token window");
        assert!(error.message.contains("max_context_tokens"), "{error}");

        config.discard_stale_context_tokens();
        assert_eq!(
            config.model.context_tokens, None,
            "a ceiling from the cached id is stale under an explicit checkpoint"
        );
        config.runtime.max_context_tokens = 128;
        config
            .validate()
            .expect("the runtime window fits the explicit checkpoint");
    }

    #[test]
    fn an_explicit_storage_only_quantized_checkpoint_is_refused() {
        // The explicit branch used to run only the document-level gate, so a checkpoint whose
        // config.json omits quantization_config while its shards store U8 codes validated and
        // saved in the console even though the CLI refused it. The explicit branch must open the
        // shards and fold in the same storage predicate the cached branch uses.
        let dir = tempfile::tempdir().expect("tempdir");
        let checkpoint = dir.path().join("checkpoint");
        std::fs::create_dir_all(&checkpoint).expect("explicit checkpoint");
        std::fs::write(
            checkpoint.join("config.json"),
            serde_json::to_vec(&crate::hf::tests::custom_config()).expect("config JSON"),
        )
        .expect("config");
        sparkflash_weights::safetensors::write_checkpoint(
            &checkpoint,
            &[(
                sparkflash_weights::TensorSpec {
                    name: "model.embed_tokens.weight".to_owned(),
                    dtype: sparkflash_weights::DType::U8,
                    shape: vec![16],
                },
                vec![0x21; 16],
            )],
            1,
        )
        .expect("the storage-only quantized shard writes");

        let mut config = SparkConfig::default();
        config.model.hub_cache = dir.path().join("hub").display().to_string();
        config.model.id = "acme/Explicit-Storage-Quantized".to_owned();
        config.model.weights = config.model.id.clone();
        config.model.checkpoint = checkpoint.display().to_string();
        config.engine.kind = DYNAMIC_ENGINE_ID.to_owned();
        config.cluster.nodes = vec![NodeConfig::new("this-spark.local", "", true)];

        let error = config
            .validate()
            .expect_err("a storage-only quantized explicit checkpoint is refused");
        assert!(
            error.message.contains("U8") && error.message.contains("quantized"),
            "the refusal names the stored dtype: {error}"
        );
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
