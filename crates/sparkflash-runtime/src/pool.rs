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

//! Reservations of lane memory, shared by every engine that opens sequences.
//!
//! A [`LaneMemory`] is a reservation, not an allocation: the arm owns the bytes and the pool
//! owns only which lane index is in use. That keeps the policy in one place and lets an arm with
//! its own allocator (dense's lane mapper, DeepSeek's paged cache) say which reservation it is
//! spending without the runtime learning its address space.
//!
//! The pool is an ordinary free list and grows on demand: an engine asks for one lane per slot
//! and gives it back on [crate::Runtime::release]. It never hands the same index out twice while
//! the first holder still has it.

/// One reservation of lane memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LaneMemory {
    index: usize,
    bytes: usize,
}

impl LaneMemory {
    /// A reservation of `bytes` at `index`.
    #[must_use]
    pub const fn new(index: usize, bytes: usize) -> Self {
        Self { index, bytes }
    }

    /// The lane index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }

    /// The bytes reserved.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

/// The lane reservations of one engine.
#[derive(Debug, Clone)]
pub struct LanePool {
    lane_bytes: usize,
    next: usize,
    free: Vec<usize>,
    in_use: Vec<bool>,
    outstanding: usize,
}

impl LanePool {
    /// A pool whose lanes each reserve `lane_bytes`.
    #[must_use]
    pub const fn new(lane_bytes: usize) -> Self {
        Self {
            lane_bytes,
            next: 0,
            free: Vec::new(),
            in_use: Vec::new(),
            outstanding: 0,
        }
    }

    /// Takes a lane, reusing a released one before growing the pool.
    pub fn acquire(&mut self) -> LaneMemory {
        let index = match self.free.pop() {
            Some(index) => {
                self.in_use[index] = true;
                index
            }
            None => {
                let index = self.next;
                self.next += 1;
                self.in_use.push(true);
                index
            }
        };
        self.outstanding += 1;
        LaneMemory::new(index, self.lane_bytes)
    }

    /// Gives `lane` back. Releasing a lane twice, or one this pool did not hand out, changes
    /// nothing, so a double [crate::Runtime::release] cannot corrupt the list.
    pub fn release(&mut self, lane: LaneMemory) {
        if let Some(in_use) = self.in_use.get_mut(lane.index)
            && *in_use
        {
            *in_use = false;
            self.free.push(lane.index);
            self.outstanding = self.outstanding.saturating_sub(1);
        }
    }

    /// The bytes one lane reserves.
    #[must_use]
    pub const fn lane_bytes(&self) -> usize {
        self.lane_bytes
    }

    /// How many lanes are held.
    #[must_use]
    pub const fn outstanding(&self) -> usize {
        self.outstanding
    }

    /// How many lanes have ever been handed out: the pool's high-water mark.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_released_lane_comes_back_before_the_pool_grows() {
        let mut pool = LanePool::new(1_024);
        let first = pool.acquire();
        let second = pool.acquire();
        assert_eq!((first.index(), second.index()), (0, 1));
        assert_eq!(pool.capacity(), 2);
        pool.release(first);
        assert_eq!(pool.outstanding(), 1);
        let again = pool.acquire();
        assert_eq!(again.index(), 0, "the released lane is reused");
        assert_eq!(pool.capacity(), 2, "the pool did not grow");
        assert_eq!(again.bytes(), 1_024);
    }

    #[test]
    fn a_double_release_changes_nothing() {
        let mut pool = LanePool::new(8);
        let lane = pool.acquire();
        pool.release(lane);
        pool.release(lane);
        assert_eq!(pool.outstanding(), 0);
        assert_eq!(pool.acquire().index(), 0);
        assert_eq!(pool.capacity(), 1);
    }
}
