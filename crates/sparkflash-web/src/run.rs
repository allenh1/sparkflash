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

//! `sparkflash --web` startup: load the TOML, start mDNS, sample telemetry, bind the router.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;

use crate::api::{WebState, WebStateInit, router};
use crate::commands::local_hostname;
use crate::config::{DEFAULT_CONFIG_PATH, DEFAULT_MODEL_ID, SparkConfig};
use crate::discovery::{CommandSshProbe, DiscoveredNode, MdnsDiscovery, NodeDiscovery};
use crate::jobs::JobManager;
use crate::stats::{StatsStore, TelemetrySource, now_ms};

/// Options accepted after `sparkflash --web` (or `sparkflash web`).
#[derive(Debug, Parser)]
#[command(
    name = "sparkflash web",
    version,
    about = "Lead-node web control plane for a SparkFlash four-node cluster"
)]
struct WebArgs {
    /// The TOML file to load and save. Defaults to ./sparkflash.toml.
    #[arg(long, env = "SPARKFLASH_CONFIG", default_value = DEFAULT_CONFIG_PATH)]
    config: PathBuf,
    /// Override web.bind from the TOML.
    #[arg(long, env = "SPARKFLASH_WEB_BIND")]
    bind: Option<String>,
    /// Do not advertise or browse mDNS; still probe the configured nodes.
    #[arg(long)]
    no_mdns: bool,
}

/// An mDNS implementation used when the real daemon cannot start.
struct NoDiscovery(String);

impl NodeDiscovery for NoDiscovery {
    fn discover(&self, _timeout: Duration) -> Result<Vec<DiscoveredNode>, String> {
        Err(self.0.clone())
    }

    fn advertise(&self, _port: u16) -> Result<String, String> {
        Err(self.0.clone())
    }
}

/// Entry point used by the `sparkflash` dispatcher.
pub fn run(args: impl IntoIterator<Item = String>) -> ExitCode {
    let cli = WebArgs::parse_from(std::iter::once("sparkflash web".to_owned()).chain(args));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("sparkflash_web=info")),
        )
        .try_init();

    let exists = cli.config.exists();
    let (mut config, load_error) = match SparkConfig::load(&cli.config) {
        Ok(config) => (config, None),
        Err(error) => {
            let note = if exists {
                format!(
                    "{}; the built-in defaults are loaded until you save a fix",
                    error.message
                )
            } else {
                format!(
                    "{}; the built-in defaults are loaded, save them to create the file",
                    error.message
                )
            };
            eprintln!("sparkflash web: {note}");
            (SparkConfig::default(), Some(note))
        }
    };
    // A fresh checkout should use a complete checkpoint that is already on this Spark instead of
    // offering to download the canonical id again (for example an existing `...-1017` snapshot).
    // A saved file that still names only the built-in canonical default is migrated the same way;
    // an operator who deliberately chose another uncached id is left alone.
    let default_model = config.model.id == DEFAULT_MODEL_ID;
    if (!exists || default_model)
        && let Some(model) = config.prefer_cached_model()
    {
        println!(
            "sparkflash web: using the cached model {} ({})",
            model.id, model.revision
        );
        if exists && let Err(error) = config.save(&cli.config) {
            eprintln!(
                "sparkflash web: using {} in memory, but cannot save it to {}: {}",
                model.id,
                cli.config.display(),
                error.message
            );
        }
    }
    // The binary cargo just built is the one this process is running. Adopt it when the
    // configured path is not there, so the first Install copies a real server instead of asking
    // for scripts/setup.sh.
    if let Ok(executable) = std::env::current_exe()
        && config.adopt_running_binary(&executable)
    {
        println!(
            "sparkflash web: using the running binary {}",
            executable.display()
        );
    }
    let bind_text = cli.bind.clone().unwrap_or_else(|| config.web.bind.clone());
    let bind: SocketAddr = match bind_text.parse() {
        Ok(bind) => bind,
        Err(error) => {
            eprintln!("sparkflash web: cannot parse bind address {bind_text:?}: {error}");
            return ExitCode::from(2);
        }
    };

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("sparkflash web: cannot start the async runtime: {error}");
            return ExitCode::FAILURE;
        }
    };
    runtime.block_on(serve(cli, config, load_error, exists, bind))
}

async fn serve(
    cli: WebArgs,
    config: SparkConfig,
    load_error: Option<String>,
    exists: bool,
    bind: SocketAddr,
) -> ExitCode {
    let mdns: Arc<dyn NodeDiscovery> = if cli.no_mdns {
        Arc::new(NoDiscovery("mDNS is disabled with --no-mdns".to_owned()))
    } else {
        match MdnsDiscovery::new() {
            Ok(discovery) => Arc::new(discovery),
            Err(error) => {
                eprintln!(
                    "sparkflash web: mDNS unavailable ({error}); configured nodes still work"
                );
                Arc::new(NoDiscovery(error))
            }
        }
    };
    let advertised = if config.web.advertise && !cli.no_mdns {
        match mdns.advertise(bind.port()) {
            Ok(fullname) => {
                println!(
                    "sparkflash web: advertising {fullname} on port {}",
                    bind.port()
                );
                Some(fullname)
            }
            Err(error) => {
                eprintln!("sparkflash web: could not advertise mDNS: {error}");
                None
            }
        }
    } else {
        None
    };
    let stats = Arc::new(StatsStore::new(&config.stats, config.stats_path()));
    let telemetry = Arc::new(TelemetrySource::new(&config.web.serve_url));
    let jobs = Arc::new(JobManager::new());
    let state = WebState::new(WebStateInit {
        config: config.clone(),
        config_path: cli.config.clone(),
        load_error,
        exists,
        discovery: Arc::clone(&mdns),
        ssh: Arc::new(CommandSshProbe),
        jobs: Arc::clone(&jobs),
        stats: Arc::clone(&stats),
        telemetry: Arc::clone(&telemetry),
        advertised: advertised.clone(),
    });

    let sampler_state = state.clone();
    tokio::spawn(async move {
        loop {
            let telemetry = sampler_state.telemetry();
            let raw = tokio::task::spawn_blocking(move || telemetry.fetch())
                .await
                .unwrap_or_default();
            let _ = sampler_state.stats().ingest(&raw, now_ms());
            // Re-read the interval each pass so a saved config takes effect on the next sample.
            let interval =
                Duration::from_secs(sampler_state.config().stats.sample_interval_secs.max(1));
            tokio::time::sleep(interval).await;
        }
    });

    // Automatic provisioning: build the local artifacts a fresh checkout lacks, verify every
    // Spark's copy of this same binary, and copy it where it differs. The UI shows these as
    // ordinary jobs, so nothing here is hidden from the operator.
    let provision_state = state.clone();
    tokio::spawn(async move {
        loop {
            // A missing config, an unreachable peer or a failed artifact stays visible on the
            // Setup and Activity pages; the automatic pass simply tries again on its next tick.
            let _ = provision_state.provision_once().await;
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    });

    let app = router(state.clone());
    let listener = match tokio::net::TcpListener::bind(bind).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("sparkflash web: cannot bind {bind}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let bound = listener.local_addr().unwrap_or(bind);
    println!("sparkflash web: control plane on http://{bound}");
    println!(
        "sparkflash web: config {} ({})",
        cli.config.display(),
        if exists { "loaded" } else { "not created yet" }
    );
    println!(
        "sparkflash web: host {}, serving telemetry from {}",
        local_hostname(),
        config.web.serve_url
    );
    let shutdown_state = state.clone();
    let server = axum::serve(listener, app).with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    });
    if let Err(error) = server.await {
        eprintln!("sparkflash web: server stopped: {error}");
        return ExitCode::FAILURE;
    }
    match shutdown_state.stats().persist_if_idle() {
        Ok(true) => println!("sparkflash web: idle telemetry snapshot written"),
        Ok(false) => println!("sparkflash web: cluster was not idle, telemetry snapshot skipped"),
        Err(error) => eprintln!("sparkflash web: cannot persist telemetry: {error}"),
    }
    ExitCode::SUCCESS
}
