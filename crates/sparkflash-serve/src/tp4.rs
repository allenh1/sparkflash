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

//! The tensor-parallel group's rank seam: what one process of a serve group is told, and what it
//! answers the process that drives it.
//!
//! A rank opener is per-arm construction code and lives in the arm's crate. What travels between
//! the driver and an opener is this module: the checkpoint, the group's geometry and the
//! deadlines, in one struct the arm reads and the driver fills. Nothing here owns arithmetic or a
//! transport.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// What rank 0 passes to the rank factory: the checkpoint and the group geometry, not a tokenizer.
///
/// Token ids are the contract between serve and the rank; the tokenizer stays on rank 0 and never
/// reaches a worker.
#[derive(Debug, Clone)]
pub struct Tp4RankConfig {
    /// This process's rank.
    pub rank: usize,
    /// The number of ranks in the group.
    pub world_size: usize,
    /// Every rank's address, in rank order.
    pub peers: Vec<IpAddr>,
    /// Rank R binds the engine collective on base_port + R.
    pub base_port: u16,
    /// Rank 0's serve control port.
    pub control_port: u16,
    /// The released sharded checkpoint directory or index.
    pub checkpoint: PathBuf,
    /// The element cap for one tensor.
    pub element_limit: usize,
    /// The context one packed-model sequence allocates. The host reference rank sizes its own
    /// sequence state from the checkpoint, so only a packed arm reads this.
    pub context: usize,
    /// The startup and control deadline.
    pub timeout: Duration,
}

impl Tp4RankConfig {
    /// The control channel address.
    #[must_use]
    pub fn control_address(&self) -> SocketAddr {
        SocketAddr::new(self.peers[0], self.control_port)
    }
}

/// The directory a checkpoint path names: the path itself, or the directory holding the file.
///
/// A rank opens the same checkpoint whether the operator named its directory or its index file, so
/// the rule is the seam's rather than each opener's.
#[must_use]
pub fn checkpoint_directory(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.to_path_buf()
    } else {
        checkpoint.parent().unwrap_or(Path::new(".")).to_path_buf()
    }
}
