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

//! Process command construction.
//!
//! The web server never concatenates a shell string from user input without quoting it. Commands
//! are built as a program plus an argument vector, and only the *remote* command handed to SSH is
//! a shell string, where every interpolated value goes through [`shell_quote`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::config::{SparkConfig, shell_quote};
use crate::hf;
use crate::inspect;

/// One process to run, with a display line for the job log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    /// The executable to spawn.
    pub program: String,
    /// Its arguments, already separated.
    pub args: Vec<String>,
    /// Extra environment variables.
    pub env: BTreeMap<String, String>,
    /// Working directory on the lead node.
    pub cwd: Option<PathBuf>,
    /// A short human label ("Convert rank 2", "Launch cluster").
    pub label: String,
    /// A stable job kind used for concurrency guards ("convert", "launch", ...).
    pub kind: String,
}

impl CommandSpec {
    /// A single line for the UI and the logs.
    #[must_use]
    pub fn display(&self) -> String {
        let mut line = String::new();
        for (key, value) in &self.env {
            line.push_str(key);
            line.push('=');
            line.push_str(&shell_quote(value));
            line.push(' ');
        }
        line.push_str(&shell_quote(&self.program));
        for arg in &self.args {
            line.push(' ');
            line.push_str(&shell_quote(arg));
        }
        line
    }
}

/// The one-time weight conversion for one rank.
///
/// Rank 0 can run locally when its node is marked `local`; every other rank runs the same command
/// over SSH. The flags are exactly the documented step in `docs/serving.md`. `config` is the
/// [`SparkConfig::resolved`] copy, so the checkpoint is a directory.
#[must_use]
pub fn convert_command(config: &SparkConfig, rank: usize) -> CommandSpec {
    let node = &config.cluster.nodes[rank];
    let checkpoint = config.node_path(&config.model.checkpoint);
    let output_dir = config.converted_path();
    let rank_text = rank.to_string();
    let mut convert_args = vec![
        "convert".to_owned(),
        "--checkpoint".to_owned(),
        checkpoint.display().to_string(),
        "--output-dir".to_owned(),
        output_dir.display().to_string(),
        "--tp-world-size".to_owned(),
        config.world_size().to_string(),
        "--tp-rank".to_owned(),
        rank_text,
    ];
    // 640-wide routed-expert padding is DeepSeek's layout; the GLM converter refuses the flag.
    if !config.is_glm() {
        convert_args.push("--no-pad-moe-640".to_owned());
    }
    convert_args.push("--no-verify".to_owned());
    let label = format!("Convert rank {rank} ({})", node.ssh);
    if node.local {
        let mut env = BTreeMap::new();
        env.insert(
            "SPARK_MAX_SEQ_LEN".to_owned(),
            config.runtime.max_context_tokens.to_string(),
        );
        return CommandSpec {
            program: config.binary_path().display().to_string(),
            args: convert_args,
            env,
            cwd: Some(config.repo_root()),
            label,
            kind: format!("convert-r{rank}"),
        };
    }

    let binary = config.binary_path().display().to_string();
    let repo = config.repo_root().display().to_string();
    let mut remote_parts = vec![
        "cd".to_owned(),
        shell_quote(&repo),
        "&&".to_owned(),
        shell_quote(&binary),
    ];
    remote_parts.extend(convert_args.iter().map(|arg| shell_quote(arg)));
    let mut args = config.cluster.ssh_options.clone();
    args.push(config.ssh_destination(rank));
    args.push(remote_parts.join(" "));
    CommandSpec {
        program: "ssh".to_owned(),
        args,
        env: BTreeMap::new(),
        cwd: None,
        label,
        kind: format!("convert-r{rank}"),
    }
}

/// The background conversion for one rank of a model that is not the one being served.
///
/// It is the documented `sparkflash convert` command with a distinct kind, so the setup guide
/// does not mistake a background preparation for the active model's conversion and a running
/// server does not block it.
#[must_use]
pub fn prepare_command(config: &SparkConfig, rank: usize) -> CommandSpec {
    let mut spec = convert_command(config, rank);
    spec.kind = format!("prepare-r{rank}");
    spec.label = format!(
        "Prepare {} on rank {rank} ({})",
        config.model.id, config.cluster.nodes[rank].ssh
    );
    spec
}

/// A read-only check that `rank` has everything a launch of `config` needs: the checkpoint, its
/// tokenizer, and (for a converted model) this rank's converted file and manifest.
///
/// It prints `ready`, or `missing` followed by the missing pieces, so the model shelf can say
/// which Spark still needs preparation.
#[must_use]
pub fn prepared_probe_command(config: &SparkConfig, rank: usize) -> CommandSpec {
    let checkpoint = config.node_path(&config.model.checkpoint);
    let tokenizer = config.effective_tokenizer();
    let mut checks = vec![
        format!(
            "[ -d {} ] || missing=\"$missing checkpoint\"",
            shell_quote(&checkpoint.display().to_string())
        ),
        format!(
            "[ -f {} ] || missing=\"$missing tokenizer\"",
            shell_quote(&tokenizer.display().to_string())
        ),
    ];
    if config.requires_conversion() {
        let converted = config.converted_path();
        let rank_file = converted.join(format!("rank-{rank}-of-{}.bin", config.world_size()));
        checks.push(format!(
            "[ -f {} ] || missing=\"$missing rank-file\"",
            shell_quote(&rank_file.display().to_string())
        ));
        if config.requires_conversion_manifest() {
            let manifest = converted.join("sparkflash-converted-manifest.json");
            checks.push(format!(
                "[ -f {} ] || missing=\"$missing manifest\"",
                shell_quote(&manifest.display().to_string())
            ));
            // "not older" matches `inspect` (manifest.mtime >= rank_file.mtime); a strict
            // `-nt` would report a copied or equally-stamped pair as unprepared forever.
            checks.push(format!(
                "[ ! {} -ot {} ] || missing=\"$missing manifest-newer-than-rank\"",
                shell_quote(&manifest.display().to_string()),
                shell_quote(&rank_file.display().to_string())
            ));
        }
    }
    let script = format!(
        "missing=''; {}; if [ -z \"$missing\" ]; then echo ready; else echo \"missing$missing\"; fi",
        checks.join("; ")
    );
    let label = format!(
        "Check {} on rank {rank} ({})",
        config.model.id, config.cluster.nodes[rank].ssh
    );
    if config.cluster.nodes[rank].local {
        return CommandSpec {
            program: "bash".to_owned(),
            args: vec!["-c".to_owned(), script],
            env: BTreeMap::new(),
            cwd: None,
            label,
            kind: "prepared".to_owned(),
        };
    }
    let mut args = batch_ssh_args(config);
    args.push(config.ssh_destination(rank));
    // Run the script under bash, like `inspect` does: `[ -nt ]`/`[ -ot ]` are not POSIX and a
    // login shell's startup output on stdout would stop the probe from printing a bare `ready`.
    args.push(format!(
        "bash -c {} sparkflash-prepared",
        shell_quote(&script)
    ));
    CommandSpec {
        program: "ssh".to_owned(),
        args,
        env: BTreeMap::new(),
        cwd: None,
        label,
        kind: "prepared".to_owned(),
    }
}

/// A launcher subcommand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeAction {
    /// Start the four ranks.
    Launch,
    /// SIGTERM every rank (rank 0 drains first).
    Stop,
    /// Ask whether each rank process is running.
    Status,
}

impl ServeAction {
    /// The spelling `scripts/serve-four-node.sh` accepts.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Launch => "launch",
            Self::Stop => "stop",
            Self::Status => "status",
        }
    }

    /// A short label for the UI.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Launch => "Launch cluster",
            Self::Stop => "Stop cluster",
            Self::Status => "Check rank status",
        }
    }

    /// The label a single-node launch uses instead of the cluster wording.
    #[must_use]
    pub const fn solo_label(self) -> &'static str {
        match self {
            Self::Launch => "Start server",
            Self::Stop => "Stop server",
            Self::Status => "Check server status",
        }
    }
}

/// The launcher for this topology, driven with the TOML-derived environment.
///
/// A four-node V4.1 plan uses `scripts/serve-four-node.sh`; a single TP=1 pack uses the local
/// launcher, which starts one detached `sparkflash serve --model …` process.
#[must_use]
pub fn serve_action_command(config: &SparkConfig, action: ServeAction) -> CommandSpec {
    if config.is_solo() {
        return CommandSpec {
            program: "bash".to_owned(),
            args: vec![
                config.local_serve_script_path().display().to_string(),
                action.as_str().to_owned(),
            ],
            env: config.local_launch_env(),
            cwd: Some(config.repo_root()),
            label: action.solo_label().to_owned(),
            kind: action.as_str().to_owned(),
        };
    }
    CommandSpec {
        program: "bash".to_owned(),
        args: vec![
            config.serve_script_path().display().to_string(),
            action.as_str().to_owned(),
        ],
        env: config.launch_env(),
        cwd: Some(config.repo_root()),
        label: action.label().to_owned(),
        kind: action.as_str().to_owned(),
    }
}

/// The probe that proves passwordless SSH works for one destination.
///
/// `BatchMode=yes` is deliberately always present: without it SSH may block on a password prompt,
/// which is exactly the condition the UI is reporting.
#[must_use]
pub fn ssh_probe_command(config: &SparkConfig, destination: &str) -> CommandSpec {
    let mut args = batch_ssh_args(config);
    args.push(destination.to_owned());
    args.push("true".to_owned());
    CommandSpec {
        program: "ssh".to_owned(),
        args,
        env: BTreeMap::new(),
        cwd: None,
        label: format!("SSH probe {destination}"),
        kind: "ssh-probe".to_owned(),
    }
}

/// `ssh` options for a command that must never wait for a password or a host-key prompt.
fn batch_ssh_args(config: &SparkConfig) -> Vec<String> {
    let mut args = vec!["-n".to_owned(), "-o".to_owned(), "BatchMode=yes".to_owned()];
    args.extend(config.cluster.ssh_options.iter().cloned());
    args.push("-o".to_owned());
    args.push(format!("ConnectTimeout={}", config.web.ssh_timeout_secs));
    args
}

/// The read-only node check of [`inspect`] for `rank`: `bash` on the lead for a local rank, `ssh`
/// otherwise. `checkpoint` is the lead's checkpoint directory, when the model is there.
#[must_use]
pub fn inspect_command(
    config: &SparkConfig,
    rank: usize,
    checkpoint: Option<&Path>,
) -> CommandSpec {
    let node = &config.cluster.nodes[rank];
    let script_args = inspect::script_args(config, rank, checkpoint);
    let label = format!("Check {}", node.ssh);
    if node.local {
        let mut args = vec![
            "-c".to_owned(),
            inspect::INSPECT_SCRIPT.to_owned(),
            "sparkflash-inspect".to_owned(),
        ];
        args.extend(script_args);
        return CommandSpec {
            program: "bash".to_owned(),
            args,
            env: BTreeMap::new(),
            cwd: None,
            label,
            kind: "inspect".to_owned(),
        };
    }
    let mut args = batch_ssh_args(config);
    args.push(config.ssh_destination(rank));
    args.push(inspect::remote_command(&script_args));
    CommandSpec {
        program: "ssh".to_owned(),
        args,
        env: BTreeMap::new(),
        cwd: None,
        label,
        kind: "inspect".to_owned(),
    }
}

/// Downloads `model.id` into the Hugging Face cache on the lead with `scripts/hf-download.py`,
/// resuming whatever an earlier download left.
///
/// # Errors
///
/// When `model.checkpoint` names a directory (nothing is downloaded then), or there is no cache.
pub fn download_command(config: &SparkConfig) -> Result<CommandSpec, String> {
    if !config.model.checkpoint.trim().is_empty() {
        return Err(
            "model.checkpoint names a directory, so the model is read from there; clear it to \
             download into the Hugging Face cache"
                .to_owned(),
        );
    }
    if !hf::is_model_id(&config.model.id) {
        return Err(format!(
            "{:?} is not a Hugging Face model id",
            config.model.id
        ));
    }
    let cache = config
        .hub_cache_dir()
        .ok_or("no Hugging Face cache directory: set HF_HUB_CACHE or model.hub_cache")?;
    let root = config.repo_root();
    Ok(CommandSpec {
        program: "python3".to_owned(),
        args: vec![
            "-I".to_owned(),
            root.join("scripts/hf-download.py").display().to_string(),
            "--repo".to_owned(),
            config.model.id.clone(),
            "--cache".to_owned(),
            cache.display().to_string(),
        ],
        env: BTreeMap::new(),
        cwd: Some(root),
        label: format!("Download {}", config.model.id),
        kind: "download".to_owned(),
    })
}

/// What a sync copies from the lead to the other nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncWhat {
    /// The server binary, the kernels and the NCCL library.
    Software,
    /// The checkpoint (the model's whole Hugging Face cache folder, or `model.checkpoint`).
    Model,
}

impl SyncWhat {
    /// The job kind.
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Software => "sync-software",
            Self::Model => "sync-model",
        }
    }
}

/// One node a sync copies to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncTarget {
    /// The address to connect to, the fast-network one when it is known to work.
    pub address: String,
    /// The configured SSH name, whose known host key the connection checks.
    pub ssh: String,
}

/// Copies software or the model from the lead to `targets` at the same absolute paths, with
/// `scripts/sync-nodes.py` (parallel `rsync` streams over SSH).
///
/// # Errors
///
/// When there is nothing to copy to, or the model's location is not known.
pub fn sync_command(
    config: &SparkConfig,
    what: SyncWhat,
    targets: &[SyncTarget],
) -> Result<CommandSpec, String> {
    if targets.is_empty() {
        return Err("every node is the lead; there is nothing to copy".to_owned());
    }
    let root = config.repo_root();
    let mut args = vec![
        "-I".to_owned(),
        root.join("scripts/sync-nodes.py").display().to_string(),
    ];
    // The connection may use the fast-network address; HostKeyAlias checks it against the key
    // already trusted for the node's name, so nothing new lands in known_hosts.
    for option in config.cluster.ssh_options.iter().chain(&[
        "-o".to_owned(),
        "BatchMode=yes".to_owned(),
        "-o".to_owned(),
        format!("ConnectTimeout={}", config.web.ssh_timeout_secs),
    ]) {
        args.push(format!("--ssh-arg={option}"));
    }
    for target in targets {
        let user = config.ssh_target(&target.address);
        args.push(format!("--host={user},{}", target.ssh));
    }
    let paths: Vec<PathBuf> = match what {
        SyncWhat::Software => {
            args.push(format!("--ensure-dir={}", root.display()));
            let mut paths = vec![config.binary_path()];
            // GLM carries its kernels in the binary and builds no external cubin directory.
            if config.requires_cubins() {
                paths.push(config.cubins_path());
            }
            paths.extend(config.nccl_lib_path());
            paths
        }
        SyncWhat::Model => {
            args.push("--exclude=*.incomplete".to_owned());
            if config.model.checkpoint.trim().is_empty() {
                let cache = config.hub_cache_dir().ok_or(
                    "no Hugging Face cache directory: set HF_HUB_CACHE or model.hub_cache",
                )?;
                vec![cache.join(hf::repo_folder(&config.model.id))]
            } else {
                vec![config.node_path(&config.model.checkpoint)]
            }
        }
    };
    for path in paths {
        args.push(format!("--path={}", path.display()));
    }
    Ok(CommandSpec {
        program: "python3".to_owned(),
        args,
        env: BTreeMap::new(),
        cwd: Some(root),
        label: match what {
            SyncWhat::Software => "Deploy the server".to_owned(),
            SyncWhat::Model => "Copy the model to the other Sparks".to_owned(),
        },
        kind: what.kind().to_owned(),
    })
}

/// A copy-paste command that installs the lead node's public key on every listed host.
///
/// The key is created only when missing. `ssh-copy-id` itself still asks for the target's password
/// once, which is the one manual step passwordless SSH cannot remove.
#[must_use]
pub fn ssh_fix_command(config: &SparkConfig, failed_hosts: &[String]) -> String {
    if failed_hosts.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        "if [ ! -f ~/.ssh/id_ed25519 ]; then ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519; fi"
            .to_owned(),
    ];
    for host in failed_hosts {
        lines.push(format!(
            "ssh-copy-id -i ~/.ssh/id_ed25519.pub {}",
            shell_quote(&config.ssh_target(host))
        ));
    }
    lines.join("\n")
}

/// The local IP used for the mDNS advertisement, best effort.
///
/// A UDP connect performs no traffic; it lets the kernel select the interface it would use for a
/// default route. On a host without a default route the loopback address keeps registration
/// working, and discovery still finds the service by name.
#[must_use]
pub fn local_ipv4() -> String {
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|socket| {
            socket.connect("192.0.2.1:9")?;
            socket.local_addr()
        })
        .map(|address| address.ip().to_string())
        .unwrap_or_else(|_| "127.0.0.1".to_owned())
}

/// The local hostname, without a trailing dot, falling back to a stable label.
#[must_use]
pub fn local_hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|name| name.trim().to_owned())
        .filter(|name| !name.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "sparkflash-lead".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_convert_runs_the_documented_rank_command() {
        let mut config = SparkConfig::default();
        config.model.checkpoint = "/data/DeepSeek-V4.1-Flash".to_owned();
        config.model.converted_dir = "/data/sparkflash-conv".to_owned();
        let command = convert_command(&config, 0);
        let cwd = std::env::current_dir().expect("cwd");
        assert_eq!(
            command.program,
            cwd.join("target/release/sparkflash").display().to_string()
        );
        assert_eq!(command.cwd, Some(cwd));
        let converted = config.converted_path();
        assert!(
            converted
                .display()
                .to_string()
                .starts_with("/data/sparkflash-conv/models--deepseek-ai--DeepSeek-V4.1-Flash"),
            "{}",
            converted.display()
        );
        for expected in [
            "convert",
            "--checkpoint",
            "/data/DeepSeek-V4.1-Flash",
            "--output-dir",
            converted.to_str().expect("utf8"),
            "--tp-world-size",
            "4",
            "--tp-rank",
            "0",
            "--no-pad-moe-640",
            "--no-verify",
        ] {
            assert!(
                command.args.iter().any(|arg| arg == expected),
                "missing {expected:?}: {:?}",
                command.args
            );
        }
    }

    #[test]
    fn a_glm_convert_omits_the_deepseek_padding_flag_and_launch_names_the_pack() {
        let mut config = SparkConfig::default();
        config.model.weights = crate::config::GLM_WEIGHTS_ID.to_owned();
        config.model.id = crate::config::GLM_MODEL_ID.to_owned();
        config.model.checkpoint = "/data/GLM-5.3-Flash".to_owned();
        config.engine.kind = crate::config::GLM_ENGINE_ID.to_owned();
        let convert = convert_command(&config, 0);
        assert_eq!(convert.program, config.binary_path().display().to_string());
        assert!(!convert.args.iter().any(|arg| arg == "--no-pad-moe-640"));
        assert!(convert.args.iter().any(|arg| arg == "--tp-world-size"));
        assert!(convert.args.iter().any(|arg| arg == "4"));

        let launch = serve_action_command(&config, ServeAction::Launch);
        assert_eq!(
            launch.env["SPARKFLASH_MODEL"],
            crate::config::GLM_WEIGHTS_ID
        );
        assert_eq!(
            launch.env["SPARKFLASH_CONTEXT"],
            config.runtime.max_context_tokens.to_string()
        );
        assert!(!launch.env.contains_key("SPARKFLASH_CUBINS"));

        let targets = [SyncTarget {
            address: "198.51.100.12".to_owned(),
            ssh: "spark1".to_owned(),
        }];
        let sync = sync_command(&config, SyncWhat::Software, &targets).expect("software sync");
        assert!(
            !sync.args.iter().any(|arg| arg.ends_with("/target/cubins")),
            "{:?}",
            sync.args
        );
    }

    #[test]
    fn a_background_prepare_is_a_separate_job_kind_and_the_probe_checks_every_rank() {
        let mut config = SparkConfig::default();
        config.model.checkpoint = "/data/DeepSeek-V4.1-Flash".to_owned();
        let prepare = prepare_command(&config, 1);
        assert_eq!(prepare.kind, "prepare-r1");
        assert!(prepare.label.contains(crate::config::DEFAULT_MODEL_ID));
        assert_eq!(prepare.program, "ssh");
        assert!(
            prepare
                .args
                .last()
                .is_some_and(|arg| arg.contains("convert"))
        );

        let local = prepared_probe_command(&config, 0);
        assert_eq!(local.kind, "prepared");
        assert_eq!(local.program, "bash");
        let script = &local.args[1];
        assert!(script.contains("rank-0-of-4.bin"), "{script}");
        assert!(
            script.contains("sparkflash-converted-manifest.json"),
            "{script}"
        );
        assert!(script.contains("tokenizer.json"), "{script}");

        let remote = prepared_probe_command(&config, 2);
        assert_eq!(remote.program, "ssh");
        assert!(
            remote
                .args
                .last()
                .is_some_and(|arg| arg.contains("rank-2-of-4.bin"))
        );
    }

    #[test]
    fn remote_convert_goes_over_ssh_and_quotes_paths() {
        let mut config = SparkConfig::default();
        config.model.checkpoint = "/data/DeepSeek-V4.1-Flash".to_owned();
        config.cluster.nodes[1].ssh = "spark one".to_owned();
        config.paths.repo_root = "/opt/spark flash".to_owned();
        config.model.converted_dir = "/data/sparkflash-conv".to_owned();
        let command = convert_command(&config, 1);
        assert_eq!(command.program, "ssh");
        assert_eq!(command.args.last().expect("remote command"), &{
            let mut expected = String::from(
                "cd '/opt/spark flash' && '/opt/spark flash/target/release/sparkflash'",
            );
            let converted = config.converted_path();
            for arg in [
                "convert",
                "--checkpoint",
                "/data/DeepSeek-V4.1-Flash",
                "--output-dir",
                converted.to_str().expect("utf8"),
                "--tp-world-size",
                "4",
                "--tp-rank",
                "1",
                "--no-pad-moe-640",
                "--no-verify",
            ] {
                expected.push(' ');
                expected.push_str(&shell_quote(arg));
            }
            expected
        });
        assert!(
            command.args.iter().any(|arg| arg.ends_with("spark one")),
            "{:?}",
            command.args
        );
    }

    #[test]
    fn launch_passes_the_tuning_and_fabric_environment() {
        let mut config = SparkConfig::default();
        config.runtime.max_context_tokens = 131_072;
        config.runtime.max_concurrency = 4;
        config.network.nccl_ib_hca = "rocep1s0f1".to_owned();
        let command = serve_action_command(&config, ServeAction::Launch);
        assert_eq!(command.program, "bash");
        assert_eq!(command.args[1], "launch");
        assert_eq!(command.env["SPARK_MAX_SEQ_LEN"], "131072");
        assert_eq!(command.env["SPARKFLASH_LANES"], "4");
        assert_eq!(command.env["NCCL_IB_HCA"], "rocep1s0f1");
        assert_eq!(command.env["SSH_USER"], config.cluster.ssh_user);
    }

    #[test]
    fn a_solo_qwen_launch_uses_the_local_script_and_no_ssh() {
        let mut config = SparkConfig::default();
        config.model.weights = crate::config::QWEN_WEIGHTS_ID.to_owned();
        config.model.id = crate::config::QWEN_MODEL_ID.to_owned();
        config.model.checkpoint = "/data/Qwen3.8-Flash-Next-NVFP4".to_owned();
        config.model.cubins.clear();
        config.engine.kind = crate::config::QWEN_ENGINE_ID.to_owned();
        config.runtime.max_context_tokens = crate::config::QWEN_CONTEXT_TOKENS;
        config.cluster.nodes = vec![crate::config::NodeConfig::new("this-spark.local", "", true)];
        let command = serve_action_command(&config, ServeAction::Launch);
        assert_eq!(command.program, "bash");
        assert!(command.args[0].ends_with("scripts/serve-single-node.sh"));
        assert_eq!(command.args[1], "launch");
        assert_eq!(
            command.env["SPARKFLASH_MODEL"],
            crate::config::QWEN_WEIGHTS_ID
        );
        assert_eq!(
            command.env["SPARKFLASH_CONTEXT"],
            crate::config::QWEN_CONTEXT_TOKENS.to_string()
        );
        assert!(!command.args.iter().any(|arg| arg == "ssh"));
        assert_eq!(command.label, "Start server");
    }

    #[test]
    fn ssh_probe_is_non_interactive_even_without_config_options() {
        let mut config = SparkConfig::default();
        config.cluster.ssh_options.clear();
        let command = ssh_probe_command(&config, "spark1");
        assert!(command.args.iter().any(|arg| arg == "BatchMode=yes"));
        assert!(command.args.iter().any(|arg| arg == "ConnectTimeout=5"));
        assert_eq!(command.args.last().expect("command"), "true");
    }

    #[test]
    fn fix_command_creates_the_key_once_and_names_every_failed_host() {
        let config = SparkConfig::default();
        let fix = ssh_fix_command(&config, &["spark1".to_owned(), "other@spark2".to_owned()]);
        assert!(fix.starts_with("if [ ! -f ~/.ssh/id_ed25519 ]"));
        assert!(fix.contains("ssh-copy-id -i ~/.ssh/id_ed25519.pub"));
        assert!(fix.contains(&format!("{}@spark1", config.cluster.ssh_user)));
        assert!(fix.contains("other@spark2"));
        assert_eq!(ssh_fix_command(&config, &[]), "");
    }

    #[test]
    fn display_quotes_arguments_with_spaces() {
        let spec = CommandSpec {
            program: "bash".to_owned(),
            args: vec!["-c".to_owned(), "echo hi".to_owned()],
            env: BTreeMap::from([("A".to_owned(), "b c".to_owned())]),
            cwd: None,
            label: "test".to_owned(),
            kind: "test".to_owned(),
        };
        assert_eq!(spec.display(), "A='b c' bash -c 'echo hi'");
    }

    #[test]
    fn the_node_check_runs_locally_for_the_lead_and_over_batch_ssh_elsewhere() {
        let config = SparkConfig::default();
        let local = inspect_command(&config, 0, Some(Path::new("/models/snap")));
        assert_eq!(local.program, "bash");
        assert_eq!(local.args[0], "-c");
        assert_eq!(local.args[2], "sparkflash-inspect");
        assert_eq!(local.args[6], "/models/snap");
        let remote = inspect_command(&config, 2, None);
        assert_eq!(remote.program, "ssh");
        assert!(remote.args.iter().any(|arg| arg == "BatchMode=yes"));
        let destination = config.ssh_destination(2);
        let at = remote
            .args
            .iter()
            .position(|arg| *arg == destination)
            .expect("destination");
        assert!(remote.args[at + 1].starts_with("bash -c '"));
        assert_eq!(remote.args.len(), at + 2);
    }

    #[test]
    fn the_download_fills_the_hub_cache_and_only_when_no_directory_is_set() {
        let mut config = SparkConfig::default();
        config.model.hub_cache = "/cache/hub".to_owned();
        let command = download_command(&config).expect("command");
        assert_eq!(command.program, "python3");
        assert_eq!(command.kind, "download");
        assert!(command.args[1].ends_with("scripts/hf-download.py"));
        assert_eq!(
            command.args[2..],
            [
                "--repo",
                "deepseek-ai/DeepSeek-V4.1-Flash",
                "--cache",
                "/cache/hub"
            ]
        );
        config.model.checkpoint = "/data/model".to_owned();
        assert!(download_command(&config).is_err());
    }

    #[test]
    fn a_sync_copies_the_same_absolute_paths_to_every_other_node() {
        let mut config = SparkConfig::default();
        config.paths.repo_root = "/srv/sparkflash".to_owned();
        config.model.hub_cache = "/cache/hub".to_owned();
        config.cluster.ssh_user = "spark".to_owned();
        let targets = [SyncTarget {
            address: "198.51.100.12".to_owned(),
            ssh: "spark1".to_owned(),
        }];
        let software = sync_command(&config, SyncWhat::Software, &targets).expect("software");
        assert_eq!(software.kind, "sync-software");
        for expected in [
            "/srv/sparkflash/scripts/sync-nodes.py",
            "--host=spark@198.51.100.12,spark1",
            "--ensure-dir=/srv/sparkflash",
            "--path=/srv/sparkflash/target/release/sparkflash",
            "--path=/srv/sparkflash/target/cubins",
            "--path=/srv/sparkflash/target/deps/nccl",
            "--ssh-arg=BatchMode=yes",
        ] {
            assert!(
                software.args.iter().any(|arg| arg == expected),
                "{expected}: {:?}",
                software.args
            );
        }
        let model = sync_command(&config, SyncWhat::Model, &targets).expect("model");
        assert_eq!(model.kind, "sync-model");
        assert!(
            model
                .args
                .iter()
                .any(|arg| arg == "--path=/cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash")
        );
        assert!(model.args.iter().any(|arg| arg == "--exclude=*.incomplete"));
        assert!(sync_command(&config, SyncWhat::Model, &[]).is_err());
    }
}
