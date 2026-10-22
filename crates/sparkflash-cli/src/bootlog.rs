//! vLLM-shaped boot progress for `sparkflash serve`.
//!
//! The serving logs are read by operators who are watching a multi-minute startup, so the format
//! matters as much as the content. This module emits the three things that startup actually needs:
//!
//! - **One line per resolved decision**, in the reference server's shape:
//!   `(APIServer pid=273) INFO 10-22 14:53:36 [serve.rs:120] Using max model len 1048576`.
//!   The `[file:line]` target is the caller's, so a reader can jump straight to the branch that
//!   decided it. Nothing here is a measurement: a line states a resolved configuration value, not a
//!   performance claim.
//! - **A single-line progress bar for shard loading** (`Loading safetensors checkpoint shards:
//!   62% Completed | 30/48 [00:22<00:14, 1.24it/s]`), rewritten in place on a terminal and printed
//!   on its own line every 10% when not a terminal (a log file must not fill with `\r`).
//! - **A quiet switch**, so the test suite and a scripted window can ask for the pre-existing bytes:
//!   `SPARKFLASH_BOOT_LOG=0` (or `off`/`quiet`) silences every line except progress, which follows
//!   [`progress_enabled`] alone.
//!
//! Output goes to stderr, which is what the launcher redirects into `rank{rank}.log`; stdout stays
//! the machine-readable stream the existing `epoch_ms=...` timing lines already use.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Process-wide gate for [`line`]: on unless `SPARKFLASH_BOOT_LOG` says otherwise.
static ENABLED: AtomicBool = AtomicBool::new(true);
/// Process-wide gate for [`progress`]; the same switch, consulted separately so a caller can turn
/// the bar off without turning off the resolved-decision lines (or the reverse).
static PROGRESS_ENABLED: AtomicBool = AtomicBool::new(true);
/// Whether the first [`init`] already resolved the environment.
static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Reads `SPARKFLASH_BOOT_LOG` once and remembers the answer.
///
/// `0`, `off`, `false`, `no` and `quiet` disable both lines and the progress bar; `progress` keeps
/// the bar and silences the decision lines; anything else (including an unset variable) enables
/// both. Idempotent, so every entry point can call it.
pub fn init() {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return;
    }
    let Ok(value) = std::env::var("SPARKFLASH_BOOT_LOG") else {
        return;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "off" | "false" | "no" | "quiet" => {
            ENABLED.store(false, Ordering::SeqCst);
            PROGRESS_ENABLED.store(false, Ordering::SeqCst);
        }
        "progress" => ENABLED.store(false, Ordering::SeqCst),
        // An unrecognised value is not a typo worth failing a boot over: keep the default.
        _ => {}
    }
}

/// Whether the decision lines are enabled right now.
#[must_use]
pub fn enabled() -> bool {
    init();
    ENABLED.load(Ordering::Relaxed)
}

/// Whether the progress bar is enabled right now.
#[must_use]
pub fn progress_enabled() -> bool {
    init();
    PROGRESS_ENABLED.load(Ordering::Relaxed)
}

/// The role label a rank prints under, mirroring the reference server's process names.
///
/// Rank 0 owns HTTP, the tokenizer and the control channel, so it is the `APIServer`; every other
/// rank runs the resident forward and is named `Worker_TP{rank}`.
#[must_use]
pub fn role_label(rank: usize) -> String {
    if rank == 0 {
        "APIServer".to_owned()
    } else {
        format!("Worker_TP{rank}")
    }
}

/// A process id for the `(Role pid=N)` prefix, or 0 when the platform will not say.
fn pid() -> u32 {
    std::process::id()
}

/// `MM-DD HH:MM:SS` in UTC, the reference server's timestamp shape.
fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Civil-from-days over the Unix epoch; UTC only, which is what the reference prints.
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (month, day) = month_day(days);
    format!("{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}")
}

/// Month and day of month for a count of days since 1970-01-01 (proleptic Gregorian).
fn month_day(days: u64) -> (u64, u64) {
    // 1970-01-01 was a Thursday; the algorithm below only needs the day count.
    let mut year = 1970u64;
    let mut remaining = days;
    loop {
        let leap = is_leap(year);
        let in_year = if leap { 366 } else { 365 };
        if remaining < in_year {
            break;
        }
        remaining -= in_year;
        year += 1;
    }
    let leap = is_leap(year);
    const MONTHS: [u64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u64;
    for days_in_month in MONTHS {
        let length = if month == 2 && leap {
            days_in_month + 1
        } else {
            days_in_month
        };
        if remaining < length {
            break;
        }
        remaining -= length;
        month += 1;
    }
    (month, remaining + 1)
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// Emits one resolved-decision line: `(Role pid=N) INFO MM-DD HH:MM:SS [target] message`.
///
/// `target` is the `file.rs:line` a reader should open; `message` is the resolved value, never a
/// measured one. A failed write is ignored: a closed stderr must not take the server down.
pub fn info(role: &str, target: &str, message: impl std::fmt::Display) {
    if !enabled() {
        return;
    }
    let _ = writeln!(
        std::io::stderr(),
        "({role} pid={}) INFO {} [{target}] {message}",
        pid(),
        timestamp()
    );
}

/// Emits one warning line in the same shape, with the reference server's `WARNING` level word.
pub fn warn(role: &str, target: &str, message: impl std::fmt::Display) {
    if !enabled() {
        return;
    }
    let _ = writeln!(
        std::io::stderr(),
        "({role} pid={}) WARNING {} [{target}] {message}",
        pid(),
        timestamp()
    );
}

/// A one-line progress bar for the shard fill.
///
/// The bar is rewritten in place (`\r`) only when stderr is a terminal; otherwise it is printed on
/// its own line every `step_percent` so a redirected log stays readable. Rendering is rate-limited
/// to [`RENDER_INTERVAL`], because a bar that costs a write per shard is slow on a 48-shard
/// checkpoint and a per-tensor loop can tick hundreds of times.
pub struct Progress {
    label: String,
    total: usize,
    done: usize,
    started: Instant,
    last_render: Option<Instant>,
    last_percent: usize,
    interactive: bool,
    finished: bool,
}

/// The minimum gap between two renders of the same bar.
const RENDER_INTERVAL: Duration = Duration::from_millis(100);

impl Progress {
    /// Starts a bar named `label` over `total` units.
    #[must_use]
    pub fn new(label: impl Into<String>, total: usize) -> Self {
        let interactive = std::io::IsTerminal::is_terminal(&std::io::stderr());
        Self {
            label: label.into(),
            total,
            done: 0,
            started: Instant::now(),
            last_render: None,
            last_percent: 0,
            interactive,
            finished: false,
        }
    }

    /// Records `count` completed units and renders when the rate limit and the 10% step allow.
    pub fn advance(&mut self, count: usize) {
        self.done = (self.done + count).min(self.total);
        if !progress_enabled() || self.total == 0 {
            return;
        }
        let percent = self.percent();
        let stepped = percent / 10 > self.last_percent / 10 || self.done == self.total;
        let cooled = self
            .last_render
            .is_none_or(|last| last.elapsed() >= RENDER_INTERVAL);
        if stepped && cooled {
            self.render();
        }
    }

    /// Renders the final 100% line and ends the bar with a newline.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if !progress_enabled() {
            return;
        }
        self.done = self.total;
        self.render();
        let _ = writeln!(std::io::stderr());
    }

    fn percent(&self) -> usize {
        // An empty bar is already finished; anything else is the integer share of the total.
        self.done
            .saturating_mul(100)
            .checked_div(self.total)
            .map_or(100, |percent| percent.min(100))
    }

    fn render(&mut self) {
        self.last_render = Some(Instant::now());
        self.last_percent = self.percent();
        let elapsed = self.started.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            self.done as f64 / elapsed
        } else {
            0.0
        };
        let remaining = if rate > 0.0 && self.done < self.total {
            Duration::from_secs_f64((self.total - self.done) as f64 / rate)
        } else {
            Duration::ZERO
        };
        let line = format!(
            "{}: {:3}% Completed | {}/{} [{}<{}, {:5.2}it/s]",
            self.label,
            self.percent(),
            self.done,
            self.total,
            clock(elapsed),
            clock(remaining.as_secs_f64()),
            rate
        );
        if self.interactive {
            let _ = write!(std::io::stderr(), "\r{line}");
        } else {
            let _ = writeln!(std::io::stderr(), "{line}");
        }
    }
}

/// `MM:SS` or `H:MM:SS`, the reference progress bar's clock shape.
fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let (hour, minute, second) = (total / 3600, (total % 3600) / 60, total % 60);
    if hour > 0 {
        format!("{hour}:{minute:02}:{second:02}")
    } else {
        format!("{minute:02}:{second:02}")
    }
}

/// A printable byte size (`475.25 GiB`), which the checkpoint-size line needs.
#[must_use]
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_leap_year_changes_february_only() {
        // Day 0 is 1970-01-01. 2024-02-29 is day 19782; 2023-02-28 is day 19416.
        assert_eq!(month_day(19_782), (2, 29));
        assert_eq!(month_day(19_416), (2, 28));
        assert_eq!(month_day(0), (1, 1));
        // 1970-12-31 is day 364; 1971-01-01 is day 365.
        assert_eq!(month_day(364), (12, 31));
        assert_eq!(month_day(365), (1, 1));
    }

    #[test]
    fn the_clock_switches_to_hours_only_after_an_hour() {
        assert_eq!(clock(0.0), "00:00");
        assert_eq!(clock(59.6), "01:00");
        assert_eq!(clock(3_599.0), "59:59");
        assert_eq!(clock(3_600.0), "1:00:00");
        assert_eq!(clock(7_384.0), "2:03:04");
    }

    #[test]
    fn human_bytes_uses_binary_units() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1_024), "1.00 KiB");
        // 475.25 GiB, the checkpoint size the reference server prints.
        assert_eq!(human_bytes(510_295_801_856), "475.25 GiB");
        assert_eq!(human_bytes(1_099_511_627_776), "1.00 TiB");
    }

    #[test]
    fn the_progress_bar_reaches_one_hundred_percent() {
        let mut bar = Progress::new("Loading shards", 4);
        bar.advance(3);
        assert_eq!(bar.percent(), 75);
        bar.finish();
        assert_eq!(bar.percent(), 100);
        // A second finish is a no-op rather than a second newline.
        bar.finish();
    }
}
