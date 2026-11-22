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

//! One line per kernel launch, so two runs of the same code compare equal and a reordered,
//! re-gridded or re-argued launch shows.
//!
//! With `SPARKFLASH_LAUNCH_TRACE=<file>` set, every launch the driver wrapper issues
//! ([crate::cuda_driver::Cuda]'s direct path, and a graph segment's recorded or captured launch
//! at the point the code issues it) appends one line to `<file>`:
//!
//! ```text
//! <thread> <symbol> g=<x>,<y>,<z> b=<x>,<y>,<z> sh=<bytes> s=<stream>[ pdl] a=<arg> <arg> ...
//! ```
//!
//! The thread is the launching thread's name, so a local group's ranks (one thread a rank) stay
//! apart: a stable sort by the first field gives each thread's launches in issue order. Every
//! label canonicalizes its own pointers. An argument inside an allocation the wrapper made (device,
//! managed, host-mapped, registered or a reserved virtual range) prints as `@<n>+<offset>`, `n` the
//! allocation's first-seen order in that thread's launches; any other word at or past 2^32 prints
//! as `?<n>`, by first-seen value; a smaller word prints as itself in hex. A stream prints by its
//! first-seen order, the default stream `0`; a stream the wrapper creates is a new stream even at
//! a destroyed one's handle. A replayed graph's launches are not traced again: the trace is what
//! the code issued.
//!
//! Unset, a launch costs one cached check and nothing else changes.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

/// The environment variable naming the trace file.
pub const TRACE_ENV: &str = "SPARKFLASH_LAUNCH_TRACE";

/// The first word that is not printed as itself: a device address on this platform is above it,
/// and a 32-bit scalar (a count, an f32's bits) below.
const POINTER_FLOOR: u64 = 1 << 32;

struct Trace {
    file: Mutex<File>,
    state: Mutex<State>,
}

/// The allocations the wrapper made and each thread's canonical names.
#[derive(Default)]
struct State {
    /// base -> (end, identity). An allocation over an older one's range replaces it.
    ranges: BTreeMap<u64, (u64, u64)>,
    next_identity: u64,
    /// handle -> creation: a handle the driver reuses names the newest stream made at it.
    streams: HashMap<u64, u64>,
    next_stream: u64,
    labels: HashMap<String, Names>,
}

/// One thread's first-seen orders.
#[derive(Default)]
struct Names {
    allocations: HashMap<u64, usize>,
    unknown: HashMap<u64, usize>,
    /// (handle, creation) -> name; a stream the wrapper did not create is creation 0.
    streams: HashMap<(u64, u64), usize>,
}

fn first_seen<K: Eq + std::hash::Hash>(map: &mut HashMap<K, usize>, key: K, base: usize) -> usize {
    let next = map.len() + base;
    *map.entry(key).or_insert(next)
}

impl State {
    fn note_range(&mut self, base: u64, bytes: u64) {
        let end = base.saturating_add(bytes.max(1));
        let overlapping: Vec<u64> = self
            .ranges
            .range(..end)
            .rev()
            .take_while(|(_, (other_end, _))| *other_end > base)
            .map(|(start, _)| *start)
            .collect();
        for start in overlapping {
            self.ranges.remove(&start);
        }
        // A range that started below `base` and reaches past it was found above, since
        // `take_while` walks down from `end` through every start whose range still covers it.
        self.next_identity += 1;
        self.ranges.insert(base, (end, self.next_identity));
    }

    fn note_stream(&mut self, handle: u64) {
        self.next_stream += 1;
        self.streams.insert(handle, self.next_stream);
    }

    fn containing(&self, word: u64) -> Option<(u64, u64)> {
        let (start, (end, identity)) = self.ranges.range(..=word).next_back()?;
        (word < *end).then_some((*identity, word - start))
    }

    #[allow(clippy::too_many_arguments)]
    fn line(
        &mut self,
        label: &str,
        symbol: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        stream: u64,
        programmatic: bool,
        args: &[u64],
    ) -> String {
        let resolved: Vec<Option<(u64, u64)>> =
            args.iter().map(|&word| self.containing(word)).collect();
        let creation = self.streams.get(&stream).copied().unwrap_or(0);
        let names = self.labels.entry(label.to_owned()).or_default();
        let stream = if stream == 0 {
            0
        } else {
            first_seen(&mut names.streams, (stream, creation), 1)
        };
        let mut line = format!(
            "{label} {symbol} g={},{},{} b={},{},{} sh={shared_bytes} s={stream}{} a=",
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            if programmatic { " pdl" } else { "" }
        );
        for (index, (&word, inside)) in args.iter().zip(resolved).enumerate() {
            if index > 0 {
                line.push(' ');
            }
            match inside {
                Some((identity, offset)) => {
                    let name = first_seen(&mut names.allocations, identity, 0);
                    let _ = write!(line, "@{name}+{offset:x}");
                }
                None if word >= POINTER_FLOOR => {
                    let name = first_seen(&mut names.unknown, word, 0);
                    let _ = write!(line, "?{name}");
                }
                None => {
                    let _ = write!(line, "{word:x}");
                }
            }
        }
        line.push('\n');
        line
    }
}

fn trace() -> Option<&'static Trace> {
    static TRACE: OnceLock<Option<Trace>> = OnceLock::new();
    TRACE
        .get_or_init(|| {
            let path = std::env::var_os(TRACE_ENV).filter(|path| !path.is_empty())?;
            let path = crate::trace_path::expand(&path);
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|error| {
                    panic!(
                        "{TRACE_ENV}={} cannot be opened for the launch trace: {error}",
                        path.display()
                    )
                });
            Some(Trace {
                file: Mutex::new(file),
                state: Mutex::new(State::default()),
            })
        })
        .as_ref()
}

/// Sets the process rank a `{rank}` placeholder in [TRACE_ENV] expands to.
pub fn set_rank(rank: usize) {
    crate::trace_path::set_rank(rank);
}

/// Whether this process traces its launches.
#[must_use]
pub fn enabled() -> bool {
    trace().is_some()
}

/// Notes `bytes` at device address `base` as one allocation, when tracing.
pub(crate) fn note_range(base: u64, bytes: usize) {
    if let Some(trace) = trace()
        && let Ok(mut state) = trace.state.lock()
    {
        state.note_range(base, bytes as u64);
    }
}

/// Notes a stream the wrapper created at `handle`, when tracing.
pub(crate) fn note_stream(handle: u64) {
    if let Some(trace) = trace()
        && let Ok(mut state) = trace.state.lock()
    {
        state.note_stream(handle);
    }
}

/// Appends one launch's line, when tracing.
///
/// # Panics
/// A trace file that stops taking writes: a trace with a hole compares wrong.
#[allow(clippy::too_many_arguments)]
pub(crate) fn note_launch(
    symbol: &str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_bytes: u32,
    stream: u64,
    programmatic: bool,
    args: &[u64],
) {
    let Some(trace) = trace() else {
        return;
    };
    let thread = std::thread::current();
    let label = thread
        .name()
        .map_or_else(|| format!("{:?}", thread.id()), str::to_owned)
        .replace(' ', "_");
    let line = match trace.state.lock() {
        Ok(mut state) => state.line(
            &label,
            symbol,
            grid,
            block,
            shared_bytes,
            stream,
            programmatic,
            args,
        ),
        Err(_) => panic!("the launch trace's state lock is poisoned"),
    };
    match trace.file.lock() {
        Ok(mut file) => file
            .write_all(line.as_bytes())
            .unwrap_or_else(|error| panic!("{TRACE_ENV}: a launch line did not write: {error}")),
        Err(_) => panic!("the launch trace's file lock is poisoned"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch(state: &mut State, label: &str, stream: u64, args: &[u64]) -> String {
        state.line(label, "k", (1, 2, 3), (256, 1, 1), 0, stream, false, args)
    }

    #[test]
    fn pointers_print_by_first_seen_allocation_and_offset() {
        let mut one = State::default();
        let mut two = State::default();
        // The same allocations made in another order, at other addresses.
        one.note_range(0x7000_0000_0000, 4096);
        one.note_range(0x7100_0000_0000, 4096);
        two.note_range(0x9100_0000_0000, 4096);
        two.note_range(0x9000_0000_0000, 4096);
        let a = launch(
            &mut one,
            "t",
            0,
            &[0x7100_0000_0010, 0x7000_0000_0000, 7, 0x3f80_0000],
        );
        let b = launch(
            &mut two,
            "t",
            0,
            &[0x9000_0000_0010, 0x9100_0000_0000, 7, 0x3f80_0000],
        );
        assert_eq!(a, b);
        assert_eq!(
            a,
            "t k g=1,2,3 b=256,1,1 sh=0 s=0 a=@0+10 @1+0 7 3f800000\n"
        );
        // Swapping the two arguments is a different line.
        let c = launch(
            &mut two,
            "t",
            0,
            &[0x9100_0000_0000, 0x9000_0000_0010, 7, 0],
        );
        assert!(c.contains("a=@1+0 @0+10"), "{c}");
    }

    #[test]
    fn each_thread_names_its_own_and_an_overwritten_range_is_a_new_allocation() {
        let mut state = State::default();
        state.note_range(0x7000_0000_0000, 1 << 20);
        let a = launch(&mut state, "rank-1", 5, &[0x7000_0000_0100]);
        let b = launch(&mut state, "rank-2", 9, &[0x7000_0000_0100]);
        assert!(a.ends_with("s=1 a=@0+100\n"), "{a}");
        assert!(b.ends_with("s=1 a=@0+100\n"), "{b}");
        // Freed and reallocated inside the old range: a new identity, the next name.
        state.note_range(0x7000_0000_0000, 256);
        let c = launch(
            &mut state,
            "rank-1",
            5,
            &[0x7000_0000_0010, 0x7000_0000_0200],
        );
        assert!(c.ends_with("a=@1+10 ?0\n"), "{c}");
    }

    #[test]
    fn a_stream_created_at_a_destroyed_ones_handle_is_a_new_stream() {
        let mut state = State::default();
        state.note_stream(0x5000);
        let a = launch(&mut state, "t", 0x5000, &[]);
        state.note_stream(0x6000);
        let b = launch(&mut state, "t", 0x6000, &[]);
        state.note_stream(0x5000);
        let c = launch(&mut state, "t", 0x5000, &[]);
        assert!(a.contains(" s=1 "), "{a}");
        assert!(b.contains(" s=2 "), "{b}");
        assert!(c.contains(" s=3 "), "{c}");
    }
}
