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

//! An arm's bits on its fixture, written down once and compared after: the check a walk's move
//! owes ([`check_record`]).
//!
//! Moving an arm onto another walk must not move one output bit. The generated checks compare an
//! arm with itself (whole against chunked, alone against four); none compares the arm with what
//! it was. This module runs one fixed script over the arm's fixture at TP 1, 2 and 4 and prints
//! it as lines: each run's tokens, and a hash of every logit bit a sampled run reads. With
//! `SPARKFLASH_BITS_RECORD=<dir>` the generated check writes `<dir>/<arm>-tp<world>.txt` the first
//! time and compares every later run line for line; without it the check says it skipped.
//!
//! The script covers what a walk decides: a short prompt and a long one, each whole, in chunks
//! and a token at a time; a sampled run whose logits come back; four at once; and, for a drafting
//! arm, the drafted run. A world the host cannot build is one line naming why, so a record made
//! on one GPU and checked on a cluster says which worlds it holds.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use sparkflash_model::backend::{BackendError, Sampling};

use crate::conformance::{ArmFixture, Fixture};
use crate::walk::{HeadRow, Pass, Residual, Segment, Walk};

/// The environment variable naming the record directory.
pub const RECORD_DIR: &str = "SPARKFLASH_BITS_RECORD";

/// How many tokens each run of the script takes.
const TOKENS: usize = 6;

/// A short prompt, under a window of eight rows.
const SHORT: [u32; 5] = [7, 9, 11, 13, 15];

/// A long prompt: more rows than a short-prompt path takes (40, or what the fixture's context
/// leaves after the run's tokens), in two chunks.
fn long<W: Walk>(walk: &W) -> Vec<u32> {
    let rows = 40
        .min(walk.limits().context.saturating_sub(TOKENS + 1))
        .max(SHORT.len());
    crate::conformance::long_prompt(rows)
}

/// The sampled run's sampling: seeded, so its draws repeat.
fn sampled() -> Sampling {
    Sampling {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        seed: 1234,
        ..Sampling::default()
    }
}

/// FNV-1a over the bits of `values`.
fn hash(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        value
            .to_bits()
            .to_le_bytes()
            .iter()
            .fold(hash, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
            })
    })
}

/// How a prompt is fed.
#[derive(Clone, Copy, Debug)]
enum Feed {
    Whole,
    Halves,
    Tokens,
}

/// One prompt fed by `feed` on a fresh sequence of `walk`, then `TOKENS` steps headed directly:
/// the tokens, and for a sampled run the hash of every logits row the head handed back.
fn direct<W: Walk>(
    walk: &W,
    prompt: &[u32],
    feed: Feed,
    sampling: Option<Sampling>,
) -> Result<String, BackendError> {
    let mut seq = walk.open()?;
    if let Some(sampling) = sampling {
        walk.set_sampling(&mut seq, sampling);
    }
    let mut hidden = match feed {
        Feed::Whole => walk.walk(
            &mut [Segment::new(&mut seq, prompt)],
            Pass::bulk(prompt.len()).with_prompt_end(0),
        )?,
        Feed::Halves => {
            let (first, second) = prompt.split_at(prompt.len() / 2);
            walk.walk(
                &mut [Segment::new(&mut seq, first)],
                Pass::bulk(first.len()).with_prompt_end(second.len()),
            )?;
            walk.walk(
                &mut [Segment::new(&mut seq, second)],
                Pass::bulk(second.len()).with_prompt_end(0),
            )?
        }
        Feed::Tokens => {
            let mut last = None;
            for token in prompt {
                last = Some(walk.walk(
                    &mut [Segment::new(&mut seq, std::slice::from_ref(token))],
                    Pass::token_rows(),
                )?);
            }
            last.ok_or_else(|| BackendError::new("record", "an empty prompt"))?
        }
    };
    let mut line = String::new();
    let row = hidden.rows().saturating_sub(1);
    let mut step = walk
        .head(&mut hidden, &mut [HeadRow::prompt(&mut seq, row, prompt)])?
        .pop()
        .ok_or_else(|| BackendError::new("record", "the head answered no step"))?;
    for taken in 0..TOKENS {
        // A greedy row's logits are the walk's to return or not (a device pick leaves them on
        // the device); a sampled row's come back, and their bits are the record's.
        if sampling.is_some() {
            let _ = write!(line, " {:?}/{:016x}", step.next_token, hash(&step.logits));
        } else {
            let _ = write!(line, " {:?}", step.next_token);
        }
        let Some(fed) = step.next_token else {
            break;
        };
        if taken + 1 == TOKENS {
            break;
        }
        let mut hidden = walk.walk(
            &mut [Segment::new(&mut seq, std::slice::from_ref(&fed))],
            Pass::token_rows(),
        )?;
        step = walk
            .head(&mut hidden, &mut [HeadRow::token(&mut seq, 0, fed)])?
            .pop()
            .ok_or_else(|| BackendError::new("record", "the head answered no step"))?;
    }
    Ok(line)
}

/// The script's lines over `fixture` at `world`.
fn script<A: ArmFixture>(
    world: usize,
    fixture: Fixture<A::Walk>,
) -> Result<Vec<String>, BackendError> {
    let mut lines = vec![format!("source {}", fixture.source)];
    if world > 1 || fixture.group.is_some() {
        let Fixture {
            walk, group, keep, ..
        } = fixture;
        let prompt = long(&walk);
        let tokens = crate::conformance::group_tokens(walk, group, &SHORT, TOKENS)?;
        lines.push(format!("group short {tokens:?}"));
        drop(keep);
        let again = A::fixture(world)?;
        let Fixture {
            walk, group, keep, ..
        } = again;
        let tokens = crate::conformance::group_tokens(walk, group, &prompt, TOKENS)?;
        lines.push(format!("group long {tokens:?}"));
        drop(keep);
        return Ok(lines);
    }
    let walk = &fixture.walk;
    let prompt = long(walk);
    lines.push(format!("long prompt {} rows", prompt.len()));
    for (name, prompt) in [("short", SHORT.to_vec()), ("long", prompt.clone())] {
        for feed in [Feed::Whole, Feed::Halves, Feed::Tokens] {
            let line = direct(walk, &prompt, feed, None)?;
            lines.push(format!("{name} {feed:?} greedy{line}"));
        }
        let line = direct(walk, &prompt, Feed::Whole, Some(sampled()))?;
        lines.push(format!("{name} Whole sampled{line}"));
    }
    let four = crate::conformance::four_tokens::<A>(TOKENS)?;
    lines.push(format!("four {four:?}"));
    let drafted = crate::conformance::drafted_tokens::<A>(&prompt, TOKENS)?;
    lines.push(format!("drafted long {drafted:?}"));
    Ok(lines)
}

/// The record's file for the arm named `arm` at `world` under `directory`. The name is the one
/// the arm's generated checks state ([`crate::conformance!`]'s `ARM`), not its walk's type: a
/// walk that moves keeps its record.
fn file(directory: &Path, arm: &str, world: usize) -> PathBuf {
    let arm: String = arm
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    directory.join(format!("{arm}-tp{world}.txt"))
}

/// The script of the arm named `arm` at every world, written to the record directory the first
/// time and compared with it after.
///
/// # Errors
/// A world whose lines differ from the record (the first differing line named), a script run
/// the arm fails, or a record file that cannot be read or written.
pub fn check_record<A: ArmFixture>(arm: &str) -> Result<String, BackendError> {
    let Some(directory) = std::env::var_os(RECORD_DIR).map(PathBuf::from) else {
        return Ok(format!(
            "SKIP: set {RECORD_DIR}=<dir> to record this arm's bits or compare them with the record"
        ));
    };
    let failed = |what: String| BackendError::new("record", what);
    std::fs::create_dir_all(&directory).map_err(|error| failed(error.to_string()))?;
    let mut report = Vec::new();
    for world in [1usize, 2, 4] {
        let lines = match A::fixture(world) {
            Ok(fixture) => script::<A>(world, fixture)?,
            Err(error) if world > 1 && error.code() == "group" => {
                vec![format!("unavailable: {}", error.message())]
            }
            Err(error) => return Err(error),
        };
        let path = file(&directory, arm, world);
        let text = lines.join("\n") + "\n";
        if path.is_file() {
            let kept = std::fs::read_to_string(&path).map_err(|error| failed(error.to_string()))?;
            if kept != text {
                let differs = kept
                    .lines()
                    .zip(text.lines())
                    .find(|(old, new)| old != new)
                    .map_or_else(
                        || {
                            format!(
                                "{} lines against {}",
                                text.lines().count(),
                                kept.lines().count()
                            )
                        },
                        |(old, new)| format!("recorded `{old}`, now `{new}`"),
                    );
                return Err(failed(format!(
                    "TP {world} moved from {}: {differs}",
                    path.display()
                )));
            }
            report.push(format!("TP {world}: {} lines equal", lines.len()));
        } else {
            std::fs::write(&path, &text).map_err(|error| failed(error.to_string()))?;
            report.push(format!("TP {world}: recorded {} lines", lines.len()));
        }
    }
    Ok(report.join("; "))
}
