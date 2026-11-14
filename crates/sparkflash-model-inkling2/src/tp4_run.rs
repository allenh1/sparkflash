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

//! One rank of a four-host run: the device arm, its converted file and its transport.
//!
//! The device arm is written a rank at a time, so a run is four processes that agree on the world,
//! the address list and the prompt. This module is the whole of that agreement: it opens the
//! rank's converted file, connects the collective in rank order, walks the prompt and the requested
//! decode steps, and reports what the rank saw. Nothing here is model arithmetic - the schedule and
//! the kernels are [crate::device]'s - so a launcher, a test and the CLI all run the same code.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sparkflash_comm::tcp::TcpConfig;
use sparkflash_model::backend::{ModelBackend, Sampling};
use sparkflash_model::{ModelDescriptor, TpPlan};
use sparkflash_weights::tp::TpLayout;

use crate::collective::TcpExchange;
use crate::config::Inkling2Config;
use crate::convert::RankFileSource;
use crate::device::InklingDeviceBackend;

/// What one rank of the run is told.
#[derive(Debug, Clone)]
pub struct Tp4Run {
    /// This process's rank.
    pub rank: usize,
    /// The world, which is the length of [Self::addresses].
    pub world: usize,
    /// Every rank's address, in rank order: element zero is the accepting side.
    pub addresses: Vec<SocketAddr>,
    /// This rank's converted file.
    pub converted: PathBuf,
    /// The admitted context, in tokens.
    pub context: usize,
}

/// What one rank of the run saw.
#[derive(Debug, Clone)]
pub struct Tp4Outcome {
    /// This process's rank.
    pub rank: usize,
    /// The boot line's description of the arm.
    pub banner: String,
    /// The prompt the rank forwarded.
    pub prompt: Vec<u32>,
    /// The tokens every rank chose, in order: the prompt's last step and then one a decode.
    pub generated: Vec<u32>,
    /// The prefill's wall time, in milliseconds.
    pub prefill_ms: u128,
    /// Each decode step's wall time, in milliseconds.
    pub decode_ms: Vec<u128>,
    /// The first logits' width, which is the released unpadded vocabulary.
    pub logits: usize,
    /// The first logits' best entry and its value, the comparison a second rank can make.
    pub best: (usize, f32),
}

impl Tp4Run {
    /// The greedy sampler: a run's four ranks must choose the same token, and a distribution's draw
    /// would need a shared stream.
    #[must_use]
    pub fn greedy() -> Sampling {
        Sampling {
            temperature: 0.0,
            top_k: 1,
            ..Sampling::default()
        }
    }

    /// The run's identity checks, before a 97 GB file is opened.
    ///
    /// # Errors
    /// A world that is not the address list's length, a rank outside the world, or an empty list.
    pub fn check(&self) -> Result<(), String> {
        if self.world == 0 {
            return Err("the world is zero".to_owned());
        }
        if self.addresses.len() != self.world {
            return Err(format!(
                "TP{}/rank{}: {} address(es) were given for a world of {}",
                self.world,
                self.rank,
                self.addresses.len(),
                self.world
            ));
        }
        if self.rank >= self.world {
            return Err(format!("rank {} is outside a world of {}", self.rank, self.world));
        }
        Ok(())
    }

    /// Opens the rank, walks the prompt and the decode steps, and reports the run.
    ///
    /// # Errors
    /// A rank file that does not open, a transport that does not connect, or a device failure. Every
    /// rank of the world has to be in this function at the same time: the collectives are inside the
    /// walk.
    pub fn execute(
        &self,
        descriptor: &ModelDescriptor,
        config: &Inkling2Config,
        prompt: &[u32],
        decode: usize,
    ) -> Result<Tp4Outcome, String> {
        self.check()?;
        if prompt.is_empty() {
            return Err("the prompt is empty".to_owned());
        }
        let layout = TpLayout {
            rank: self.rank,
            world_size: self.world,
        };
        let source = RankFileSource::open(&self.converted, config, layout)
            .map_err(|error| format!("{}: {error}", self.converted.display()))?;
        // Every rank loads ~97 GB before its first collective and the load times differ by
        // minutes, so the frame deadline is the run's patience, not a per-token budget: a rank that
        // reads a slow file must not look like a dead peer to the three that are ready.
        let transport = TcpConfig::with_addresses(self.rank, self.addresses.clone())
            .map_err(|error| format!("TP{}/rank{}: {error}", self.world, self.rank))?
            .with_timeout(Duration::from_secs(900))
            .with_connect_timeout(Duration::from_secs(180));
        let exchange = TcpExchange::connect(&transport)
            .map_err(|error| format!("TP{}/rank{}: {error}", self.world, self.rank))?;
        let tp = TpPlan::new(self.world, self.rank)
            .map_err(|error| format!("TP{}/rank{}: {error}", self.world, self.rank))?;
        let opened = Instant::now();
        let backend = InklingDeviceBackend::open_rank(
            descriptor.clone(),
            config.clone(),
            Arc::new(source),
            self.context,
            tp,
            Box::new(exchange),
        )
        .map_err(|error| format!("TP{}/rank{}: {error}", self.world, self.rank))?;
        let banner = format!(
            "TP{}/rank{} {} on {} ({} weight bytes, opened in {} ms)",
            self.world,
            self.rank,
            descriptor.id,
            self.converted.display(),
            backend.weight_bytes(),
            opened.elapsed().as_millis()
        );
        let mut session = backend
            .open_session()
            .map_err(|error| format!("TP{}/rank{}: {}", self.world, self.rank, error.message()))?;
        session.set_sampling(Self::greedy());

        let started = Instant::now();
        let first = session
            .prefill(prompt)
            .map_err(|error| format!("TP{}/rank{} prefill: {}", self.world, self.rank, error.message()))?;
        let prefill_ms = started.elapsed().as_millis();
        let mut best = (0usize, f32::NEG_INFINITY);
        for (index, value) in first.logits.iter().enumerate() {
            if *value > best.1 {
                best = (index, *value);
            }
        }
        let mut generated = Vec::with_capacity(decode + 1);
        let mut next = first
            .next_token
            .ok_or_else(|| format!("TP{}/rank{}: the prefill chose no token", self.world, self.rank))?;
        generated.push(next);
        let mut decode_ms = Vec::with_capacity(decode);
        for _ in 0..decode {
            let started = Instant::now();
            let step = session
                .decode(next)
                .map_err(|error| format!("TP{}/rank{} decode: {}", self.world, self.rank, error.message()))?;
            decode_ms.push(started.elapsed().as_millis());
            next = step
                .next_token
                .ok_or_else(|| format!("TP{}/rank{}: a decode chose no token", self.world, self.rank))?;
            generated.push(next);
        }
        Ok(Tp4Outcome {
            rank: self.rank,
            banner,
            prompt: prompt.to_vec(),
            generated,
            prefill_ms,
            decode_ms,
            logits: first.logits.len(),
            best,
        })
    }
}

/// The converted file a rank of a world reads, from the directory the release was converted into.
///
/// # Errors
/// A directory that does not hold that rank's file.
pub fn rank_file(directory: &Path, rank: usize, world: usize) -> Result<PathBuf, String> {
    let path = directory.join(format!("rank-{rank}-of-{world}.bin"));
    if path.is_file() {
        return Ok(path);
    }
    Err(format!(
        "{}: no converted file for TP{}/rank{rank} in {}",
        path.display(),
        world,
        directory.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::Tp4Run;

    fn address() -> std::net::SocketAddr {
        "127.0.0.1:29623".parse().expect("an address")
    }

    fn run(world: usize, rank: usize, addresses: usize) -> Tp4Run {
        Tp4Run {
            rank,
            world,
            addresses: vec![address(); addresses],
            converted: std::path::PathBuf::from("/nonexistent"),
            context: 16,
        }
    }

    #[test]
    fn a_world_must_match_its_address_list() {
        assert!(run(4, 0, 4).check().is_ok());
        let short = run(4, 0, 3).check().expect_err("three addresses for a world of four");
        assert!(short.contains("3 address(es)"), "{short}");
        let outside = run(4, 4, 4).check().expect_err("rank beyond the world");
        assert!(outside.contains("outside a world of 4"), "{outside}");
        let empty = run(0, 0, 0).check().expect_err("a world of nothing");
        assert!(empty.contains("zero"), "{empty}");
    }

    #[test]
    fn the_missing_rank_file_names_itself() {
        let directory = std::path::PathBuf::from("/nonexistent-converted");
        let error = super::rank_file(&directory, 2, 4).expect_err("the file is not there");
        assert!(error.contains("rank-2-of-4.bin"), "{error}");
    }
}
