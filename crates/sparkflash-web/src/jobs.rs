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

//! Bounded asynchronous process jobs.
//!
//! Convert, launch, stop and status are processes that outlive one HTTP request. Each is a job
//! with an id, a status, an exit code and a bounded in-memory log tail. Nothing here writes the
//! log to disk: the launcher's own remote log files remain the durable record.
//!
//! A long job (a download, a copy to the other nodes) reports how far it is by printing
//! `@progress {"done":N,"total":N,"detail":"..."}` on stdout. Those lines update the job's
//! [`JobProgress`] instead of filling the bounded log.

use std::collections::VecDeque;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::Command;

use crate::commands::CommandSpec;

/// The most log lines kept per job.
pub const MAX_LOG_LINES: usize = 400;

/// The most jobs kept in the list; the oldest finished job is evicted first.
pub const MAX_JOBS: usize = 20;

/// The maximum characters kept from one output line.
const MAX_LINE_CHARS: usize = 2_000;

/// The stdout prefix of a progress report.
pub const PROGRESS_PREFIX: &str = "@progress ";

/// How far a long job is, from its latest `@progress` line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobProgress {
    /// Units finished (bytes for a download or a copy).
    pub done: u64,
    /// Units in all; zero while the job is still sizing the work.
    pub total: u64,
    /// One short human line, e.g. "12 of 55 files, 230 MB/s".
    #[serde(default)]
    pub detail: String,
}

/// A job that could not be accepted (too many running jobs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobError {
    /// The human-readable refusal.
    pub message: String,
}

impl std::fmt::Display for JobError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for JobError {}

/// The state of one process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    /// The process is still running.
    Running,
    /// The process exited zero.
    Succeeded,
    /// The process exited non-zero, was signalled, or could not start.
    Failed,
}

impl JobStatus {
    /// Whether this is a terminal state.
    #[must_use]
    pub const fn is_done(self) -> bool {
        !matches!(self, Self::Running)
    }
}

/// The list entry for one job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobSummary {
    /// Monotone job id.
    pub id: u64,
    /// The stable action kind ("convert-r2", "launch", ...).
    pub kind: String,
    /// The short human label.
    pub label: String,
    /// The process line, with environment variables.
    pub command: String,
    /// Current state.
    pub status: JobStatus,
    /// Unix milliseconds when the process was accepted.
    pub started_ms: u64,
    /// Unix milliseconds when it became terminal.
    pub finished_ms: Option<u64>,
    /// The exit code, when one exists.
    pub exit_code: Option<i32>,
    /// The last output line, useful in the compact list.
    pub last_line: Option<String>,
    /// The latest progress report, for a job that prints them.
    pub progress: Option<JobProgress>,
}

/// The full record for the log viewer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JobDetail {
    /// Everything in [`JobSummary`].
    #[serde(flatten)]
    pub summary: JobSummary,
    /// The bounded log tail, oldest first.
    pub log: Vec<String>,
}

struct JobEntry {
    id: u64,
    kind: String,
    label: String,
    command: String,
    status: JobStatus,
    started_ms: u64,
    finished_ms: Option<u64>,
    exit_code: Option<i32>,
    log: VecDeque<String>,
    progress: Option<JobProgress>,
}

impl JobEntry {
    fn summary(&self) -> JobSummary {
        JobSummary {
            id: self.id,
            kind: self.kind.clone(),
            label: self.label.clone(),
            command: self.command.clone(),
            status: self.status,
            started_ms: self.started_ms,
            finished_ms: self.finished_ms,
            exit_code: self.exit_code,
            last_line: self.log.back().cloned(),
            progress: self.progress.clone(),
        }
    }
}

struct Inner {
    next_id: u64,
    jobs: VecDeque<JobEntry>,
}

/// A running in-process job. The owner logs progress and finishes it; dropping it unfinished is
/// recorded as a failure so a panicked task cannot leave the UI showing "running" forever.
pub struct JobTask {
    manager: Arc<JobManager>,
    id: u64,
    finished: bool,
}

impl JobTask {
    /// The job id shown in the activity list.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Appends one line to the bounded job log.
    pub fn log(&self, line: impl Into<String>) {
        self.manager.append(self.id, line.into());
    }

    /// Records a progress report, for a task that can size its work.
    pub fn progress(&self, progress: JobProgress) {
        self.manager.report(self.id, progress);
    }

    /// Marks the job succeeded or failed. Consumes the task so the drop guard is disarmed.
    pub fn finish(mut self, success: bool) {
        self.finished = true;
        self.manager
            .finish(self.id, Some(if success { 0 } else { 1 }));
    }
}

impl Drop for JobTask {
    fn drop(&mut self) {
        if !self.finished {
            self.manager.append(
                self.id,
                "the in-process task ended without a result".to_owned(),
            );
            self.manager.finish(self.id, None);
        }
    }
}

/// The process manager shared by the API and the background jobs.
#[derive(Clone)]
pub struct JobManager {
    inner: Arc<Mutex<Inner>>,
}

impl Default for JobManager {
    fn default() -> Self {
        Self::new()
    }
}

impl JobManager {
    /// Builds an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                next_id: 1,
                jobs: VecDeque::new(),
            })),
        }
    }

    /// Starts `spec` and returns its job id.
    ///
    /// A program that cannot be spawned is still a job: it is reported as failed with the OS error
    /// in the log, which is more useful to the operator than an ephemeral HTTP 500.
    ///
    /// # Errors
    /// Returns [`JobError`] when [`MAX_JOBS`] running jobs are already in flight.
    pub async fn start(self: &Arc<Self>, spec: CommandSpec) -> Result<u64, JobError> {
        self.start_exclusive(spec, &[]).await
    }

    /// Starts `spec` unless a job of a kind in `exclusive_with`, or of its own kind, is running.
    /// The check and the new job's entry are made under one lock, so two requests that race
    /// cannot both start.
    ///
    /// # Errors
    /// Returns [`JobError`] naming the running job, or when [`MAX_JOBS`] jobs are in flight.
    pub async fn start_exclusive(
        self: &Arc<Self>,
        spec: CommandSpec,
        exclusive_with: &[&str],
    ) -> Result<u64, JobError> {
        let id = self.allocate(&spec, exclusive_with)?;
        let mut command = Command::new(&spec.program);
        command
            .args(&spec.args)
            .envs(&spec.env)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(cwd) = &spec.cwd {
            command.current_dir(cwd);
        }
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                // A missing working directory also fails with ENOENT, which reads as a missing
                // program; name the directory so a wrong paths.repo_root is obvious.
                let reason = match &spec.cwd {
                    Some(cwd) if !cwd.is_dir() => {
                        format!("working directory {} does not exist", cwd.display())
                    }
                    _ => error.to_string(),
                };
                self.append(id, format!("could not start {}: {reason}", spec.program));
                self.finish(id, None);
                return Ok(id);
            }
        };
        let Some(stdout) = child.stdout.take() else {
            self.append(id, "stdout was not captured".to_owned());
            self.finish(id, None);
            return Ok(id);
        };
        let Some(stderr) = child.stderr.take() else {
            self.append(id, "stderr was not captured".to_owned());
            self.finish(id, None);
            return Ok(id);
        };
        let manager = Arc::clone(self);
        let out_manager = Arc::clone(self);
        let err_manager = Arc::clone(self);
        tokio::spawn(async move {
            let out = tokio::spawn(pump(stdout, out_manager, id, "out"));
            let err = tokio::spawn(pump(stderr, err_manager, id, "err"));
            let status = child.wait().await;
            let _ = out.await;
            let _ = err.await;
            let exit_code = status
                .as_ref()
                .ok()
                .and_then(std::process::ExitStatus::code);
            match &status {
                Ok(status) if status.success() => {
                    manager.append(id, "[exit 0]".to_owned());
                }
                Ok(status) => {
                    let line = match status.code() {
                        Some(code) => format!("[exit {code}]"),
                        None => format!("[exit {status}]"),
                    };
                    manager.append(id, line);
                }
                Err(error) => {
                    manager.append(id, format!("[wait failed: {error}]"));
                }
            }
            manager.finish(id, exit_code);
        });
        Ok(id)
    }

    /// Starts a job whose work runs in this process, such as a model swap that stops, saves and
    /// launches in sequence. The caller finishes it through the returned [`JobTask`], and a
    /// dropped task is reported as failed instead of staying "running" forever.
    ///
    /// # Errors
    /// Returns [`JobError`] when a conflicting job is running or [`MAX_JOBS`] jobs are in flight.
    pub fn start_task(
        self: &Arc<Self>,
        kind: &str,
        label: &str,
        command: impl Into<String>,
        exclusive_with: &[&str],
    ) -> Result<JobTask, JobError> {
        let id = self.allocate_entry_scoped(kind, label, command.into(), exclusive_with, None)?;
        Ok(JobTask {
            manager: Arc::clone(self),
            id,
            finished: false,
        })
    }

    /// Starts an in-process job that also refuses to overlap a background prepare job whose
    /// command contains `scope`.
    ///
    /// A swap needs this: preparing a *different* model may keep running through the swap, but a
    /// prepare of the model being swapped to rewrites the very rank files the launch will read. A
    /// kind list cannot express "a prepare for this checkpoint", so the scope is matched against
    /// the prepare job's command line while the job table is locked.
    ///
    /// # Errors
    /// Returns [`JobError`] when a conflicting job is running or [`MAX_JOBS`] jobs are in flight.
    pub fn start_task_scoped(
        self: &Arc<Self>,
        kind: &str,
        label: &str,
        command: impl Into<String>,
        exclusive_with: &[&str],
        scope: &str,
    ) -> Result<JobTask, JobError> {
        let id =
            self.allocate_entry_scoped(kind, label, command.into(), exclusive_with, Some(scope))?;
        Ok(JobTask {
            manager: Arc::clone(self),
            id,
            finished: false,
        })
    }

    /// Lists jobs, newest first.
    #[must_use]
    pub fn list(&self) -> Vec<JobSummary> {
        let inner = self.inner.lock().expect("job lock");
        inner.jobs.iter().rev().map(JobEntry::summary).collect()
    }

    /// Fetches one job with its log.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<JobDetail> {
        let inner = self.inner.lock().expect("job lock");
        inner
            .jobs
            .iter()
            .find(|job| job.id == id)
            .map(|job| JobDetail {
                summary: job.summary(),
                log: job.log.iter().cloned().collect(),
            })
    }

    /// The number of still-running jobs.
    #[must_use]
    pub fn running(&self) -> usize {
        let inner = self.inner.lock().expect("job lock");
        inner
            .jobs
            .iter()
            .filter(|job| job.status == JobStatus::Running)
            .count()
    }

    /// Whether an action in `kinds` is running, used to refuse overlapping conversions or launches.
    #[must_use]
    pub fn running_in(&self, kinds: &[&str]) -> Option<u64> {
        let inner = self.inner.lock().expect("job lock");
        inner
            .jobs
            .iter()
            .find(|job| job.status == JobStatus::Running && kinds.contains(&job.kind.as_str()))
            .map(|job| job.id)
    }

    /// Whether a conversion for `rank` is already in flight, either the foreground
    /// `convert-r` or a background `prepare-r` for another model.
    #[must_use]
    pub fn convert_running(&self, rank: usize) -> bool {
        let convert = format!("convert-r{rank}");
        let prepare = format!("prepare-r{rank}");
        self.running_in(&[convert.as_str(), prepare.as_str()])
            .is_some()
    }

    fn allocate(&self, spec: &CommandSpec, exclusive_with: &[&str]) -> Result<u64, JobError> {
        self.allocate_entry(&spec.kind, &spec.label, spec.display(), exclusive_with)
    }

    fn allocate_entry(
        &self,
        kind: &str,
        label: &str,
        command: String,
        exclusive_with: &[&str],
    ) -> Result<u64, JobError> {
        self.allocate_entry_scoped(kind, label, command, exclusive_with, None)
    }

    /// The one place a job is admitted. The kind list and the optional prepare scope are checked
    /// under the same lock that inserts the entry, so two racing requests cannot both pass.
    fn allocate_entry_scoped(
        &self,
        kind: &str,
        label: &str,
        command: String,
        exclusive_with: &[&str],
        scope: Option<&str>,
    ) -> Result<u64, JobError> {
        let mut inner = self.inner.lock().expect("job lock");
        let conflict = inner.jobs.iter().find(|job| {
            job.status == JobStatus::Running
                && ((!exclusive_with.is_empty()
                    && (job.kind == kind || exclusive_with.contains(&job.kind.as_str())))
                    || scope.is_some_and(|scope| {
                        job.kind.starts_with("prepare-") && job.command.contains(scope)
                    }))
        });
        if let Some(running) = conflict {
            return Err(JobError {
                message: format!("job {} ({}) is still running", running.id, running.label),
            });
        }
        while inner.jobs.len() >= MAX_JOBS {
            let Some(index) = inner.jobs.iter().position(|job| job.status.is_done()) else {
                return Err(JobError {
                    message: format!("{MAX_JOBS} jobs are already running"),
                });
            };
            inner.jobs.remove(index);
        }
        let id = inner.next_id;
        inner.next_id += 1;
        inner.jobs.push_back(JobEntry {
            id,
            kind: kind.to_owned(),
            label: label.to_owned(),
            command,
            status: JobStatus::Running,
            started_ms: now_ms(),
            finished_ms: None,
            exit_code: None,
            log: VecDeque::new(),
            progress: None,
        });
        Ok(id)
    }

    fn append(&self, id: u64, line: String) {
        let mut inner = self.inner.lock().expect("job lock");
        let Some(job) = inner.jobs.iter_mut().find(|job| job.id == id) else {
            return;
        };
        let line = truncate_line(&line);
        job.log.push_back(line);
        while job.log.len() > MAX_LOG_LINES {
            job.log.pop_front();
        }
    }

    fn report(&self, id: u64, progress: JobProgress) {
        let mut inner = self.inner.lock().expect("job lock");
        if let Some(job) = inner.jobs.iter_mut().find(|job| job.id == id) {
            job.progress = Some(progress);
        }
    }

    fn finish(&self, id: u64, exit_code: Option<i32>) {
        let mut inner = self.inner.lock().expect("job lock");
        if let Some(job) = inner.jobs.iter_mut().find(|job| job.id == id) {
            job.status = if exit_code == Some(0) {
                JobStatus::Succeeded
            } else {
                JobStatus::Failed
            };
            job.exit_code = exit_code;
            job.finished_ms = Some(now_ms());
        }
    }
}

async fn pump<R: AsyncRead + Unpin>(
    reader: R,
    manager: Arc<JobManager>,
    id: u64,
    stream: &'static str,
) {
    let mut lines = BufReader::new(reader).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                let progress = line
                    .strip_prefix(PROGRESS_PREFIX)
                    .filter(|_| stream == "out")
                    .and_then(|json| serde_json::from_str::<JobProgress>(json).ok());
                match progress {
                    Some(progress) => manager.report(id, progress),
                    None => manager.append(id, format!("{stream} {line}")),
                }
            }
            Ok(None) => break,
            Err(error) => {
                manager.append(id, format!("{stream} read error: {error}"));
                break;
            }
        }
    }
}

fn truncate_line(line: &str) -> String {
    if line.chars().count() <= MAX_LINE_CHARS {
        return line.to_owned();
    }
    let mut shortened: String = line.chars().take(MAX_LINE_CHARS).collect();
    shortened.push('…');
    shortened
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(program: &str, args: &[&str]) -> CommandSpec {
        CommandSpec {
            program: program.to_owned(),
            args: args.iter().map(|arg| (*arg).to_owned()).collect(),
            env: std::collections::BTreeMap::new(),
            cwd: None,
            label: "test job".to_owned(),
            kind: "test".to_owned(),
        }
    }

    async fn wait_done(manager: &JobManager, id: u64) -> JobDetail {
        for _ in 0..100 {
            let detail = manager.get(id).expect("job exists");
            if detail.summary.status.is_done() {
                return detail;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("job {id} did not finish");
    }

    #[tokio::test]
    async fn captures_stdout_stderr_and_exit_status() {
        let manager = Arc::new(JobManager::new());
        let id = manager
            .start(spec(
                "sh",
                &["-c", "printf 'hello\\n'; printf 'oops\\n' >&2; exit 3"],
            ))
            .await
            .expect("accepted");
        let detail = wait_done(&manager, id).await;
        assert_eq!(detail.summary.status, JobStatus::Failed);
        assert_eq!(detail.summary.exit_code, Some(3));
        let log = detail.log.join("\n");
        assert!(log.contains("out hello"), "{log}");
        assert!(log.contains("err oops"), "{log}");
        assert!(log.contains("[exit 3]"), "{log}");
    }

    #[tokio::test]
    async fn progress_lines_update_the_job_instead_of_the_log() {
        let manager = Arc::new(JobManager::new());
        let id = manager
            .start(spec(
                "sh",
                &[
                    "-c",
                    "printf 'sizing\\n'; \
                     printf '@progress {\"done\":5,\"total\":10,\"detail\":\"half\"}\\n'; \
                     printf '@progress not json\\n'; \
                     printf '@progress {\"done\":10,\"total\":10}\\n'",
                ],
            ))
            .await
            .expect("accepted");
        let detail = wait_done(&manager, id).await;
        assert_eq!(detail.summary.status, JobStatus::Succeeded);
        assert_eq!(
            detail.summary.progress,
            Some(JobProgress {
                done: 10,
                total: 10,
                detail: String::new(),
            })
        );
        let log = detail.log.join("\n");
        assert!(log.contains("out sizing"), "{log}");
        assert!(log.contains("out @progress not json"), "{log}");
        assert!(!log.contains("\"done\""), "{log}");
    }

    #[tokio::test]
    async fn a_missing_program_is_a_failed_job_not_a_lost_request() {
        let manager = Arc::new(JobManager::new());
        let id = manager
            .start(spec("/definitely/not/a/program", &[]))
            .await
            .expect("accepted");
        let detail = wait_done(&manager, id).await;
        assert_eq!(detail.summary.status, JobStatus::Failed);
        assert!(
            detail
                .log
                .iter()
                .any(|line| line.contains("could not start")),
            "{:?}",
            detail.log
        );
    }

    #[tokio::test]
    async fn a_missing_working_directory_is_named() {
        let manager = Arc::new(JobManager::new());
        let mut command = spec("true", &[]);
        command.cwd = Some(std::path::PathBuf::from("/definitely/not/a/checkout"));
        let id = manager.start(command).await.expect("accepted");
        let detail = wait_done(&manager, id).await;
        assert_eq!(detail.summary.status, JobStatus::Failed);
        assert!(
            detail
                .log
                .iter()
                .any(|line| line
                    .contains("working directory /definitely/not/a/checkout does not exist")),
            "{:?}",
            detail.log
        );
    }

    #[tokio::test]
    async fn overlapping_actions_are_visible_by_kind() {
        let manager = Arc::new(JobManager::new());
        let id = manager
            .start(spec("sh", &["-c", "sleep 0.2"]))
            .await
            .expect("accepted");
        assert_eq!(manager.running_in(&["test"]), Some(id));
        assert_eq!(manager.running_in(&["other"]), None);
        wait_done(&manager, id).await;
        assert_eq!(manager.running_in(&["test"]), None);
    }

    #[tokio::test]
    async fn an_in_process_task_is_visible_and_dropping_it_reports_failure() {
        let manager = Arc::new(JobManager::new());
        let task = manager
            .start_task("swap", "Swap model", "in-process swap", &["launch"])
            .expect("accepted");
        assert_eq!(manager.running_in(&["swap"]), Some(task.id()));
        task.log("stopping the old model");
        task.finish(true);
        let detail = manager.get(1).expect("job");
        assert_eq!(detail.summary.status, JobStatus::Succeeded);
        assert_eq!(detail.summary.exit_code, Some(0));
        assert!(detail.log.iter().any(|line| line.contains("stopping")));

        let abandoned = manager
            .start_task("swap", "Swap again", "in-process swap", &[])
            .expect("accepted");
        let id = abandoned.id();
        drop(abandoned);
        let detail = manager.get(id).expect("job");
        assert_eq!(detail.summary.status, JobStatus::Failed);
        assert!(
            detail
                .log
                .iter()
                .any(|line| line.contains("ended without a result")),
            "{:?}",
            detail.log
        );
    }

    #[tokio::test]
    async fn an_exclusive_start_refuses_while_a_conflicting_job_runs() {
        let manager = Arc::new(JobManager::new());
        let mut launch = spec("sh", &["-c", "sleep 0.2"]);
        launch.kind = "launch".to_owned();
        let id = manager
            .start_exclusive(launch.clone(), &["launch", "stop"])
            .await
            .expect("accepted");
        let mut stop = spec("true", &[]);
        stop.kind = "stop".to_owned();
        let refused = manager
            .start_exclusive(stop, &["launch", "stop"])
            .await
            .expect_err("a stop while launching");
        assert!(refused.message.contains(&format!("job {id}")), "{refused}");
        assert!(manager.start_exclusive(launch, &["stop"]).await.is_err());
        wait_done(&manager, id).await;
        assert!(
            manager
                .start_exclusive(spec("true", &[]), &["launch"])
                .await
                .is_ok()
        );
    }
}
