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
use std::path::PathBuf;

use crate::config::{SparkConfig, shell_quote};

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
    let checkpoint = &config.model.checkpoint;
    let output_dir = &config.model.converted_dir;
    let rank_text = rank.to_string();
    let convert_args = vec![
        "convert".to_owned(),
        "--checkpoint".to_owned(),
        checkpoint.clone(),
        "--output-dir".to_owned(),
        output_dir.clone(),
        "--tp-world-size".to_owned(),
        "4".to_owned(),
        "--tp-rank".to_owned(),
        rank_text,
        "--no-pad-moe-640".to_owned(),
        "--no-verify".to_owned(),
    ];
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
    let repo = config.paths.repo_root.clone();
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
}

/// The existing four-node launcher, driven with the TOML-derived environment.
#[must_use]
pub fn serve_action_command(config: &SparkConfig, action: ServeAction) -> CommandSpec {
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
    let mut args = vec!["-n".to_owned(), "-o".to_owned(), "BatchMode=yes".to_owned()];
    args.extend(config.cluster.ssh_options.iter().cloned());
    args.push("-o".to_owned());
    args.push(format!("ConnectTimeout={}", config.web.ssh_timeout_secs));
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
        let command = convert_command(&config, 0);
        assert_eq!(
            command.program,
            PathBuf::from("./target/release/sparkflash")
                .display()
                .to_string()
        );
        assert_eq!(command.cwd, Some(PathBuf::from(".")));
        for expected in [
            "convert",
            "--checkpoint",
            "/data/DeepSeek-V4.1-Flash",
            "--output-dir",
            "/data/sparkflash-conv",
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
    fn remote_convert_goes_over_ssh_and_quotes_paths() {
        let mut config = SparkConfig::default();
        config.model.checkpoint = "/data/DeepSeek-V4.1-Flash".to_owned();
        config.cluster.nodes[1].ssh = "spark one".to_owned();
        config.paths.repo_root = "/opt/spark flash".to_owned();
        let command = convert_command(&config, 1);
        assert_eq!(command.program, "ssh");
        assert_eq!(command.args.last().expect("remote command"), &{
            let mut expected = String::from(
                "cd '/opt/spark flash' && '/opt/spark flash/target/release/sparkflash'",
            );
            for arg in [
                "convert",
                "--checkpoint",
                "/data/DeepSeek-V4.1-Flash",
                "--output-dir",
                "/data/sparkflash-conv",
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
}
