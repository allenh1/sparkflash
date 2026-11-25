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

//! A sequence's rewind point: a copy, taken at the end of a prompt, of the state its cache rows
//! do not hold.
//!
//! A position-addressed cache (keys and values, latents, indexer rows) goes back to an earlier
//! position by its length alone: a row past it is rewritten before anything reads it. A recurrent
//! layer's state (a convolution's history, a linear-attention or delta-rule memory, a drafter's
//! last hidden row) is the state after the last row and nothing else, so a sequence that wants to
//! go back to the end of its last prompt copies those regions out when the prompt ends
//! ([Checkpoint::save]) and back in at a rewind ([Checkpoint::restore]).
//!
//! The arm owns the arena's memory (it allocates [Checkpoint::arena_bytes] from wherever its
//! sequences live) and names the live regions at every copy, since a verify that swaps a recurrent
//! pair moves which buffer is live. A region keeps its place in the arena from one copy to the
//! next: the layout is fixed when the checkpoint is made.

use crate::cuda_driver::{Cuda, Deviceptr, RunStream};

/// Each region starts on this many bytes in the arena, as the device allocator aligns a buffer.
pub const ALIGN: usize = 256;

/// One sequence's rewind point: the arena its regions are copied into and the position they hold.
#[derive(Debug)]
pub struct Checkpoint {
    arena: Deviceptr,
    sizes: Vec<usize>,
    /// The prompt's length while the copies hold its end; `None` before a prompt ends, and once
    /// the sequence forwards past the point again.
    point: Option<usize>,
}

impl Checkpoint {
    /// The bytes an arena for regions of `sizes` bytes takes.
    ///
    /// ```
    /// use sparkflash_kernels::checkpoint::Checkpoint;
    /// assert_eq!(Checkpoint::arena_bytes(&[100, 256, 0, 300]), 256 + 256 + 512);
    /// assert_eq!(Checkpoint::arena_bytes(&[]), 256);
    /// ```
    #[must_use]
    pub fn arena_bytes(sizes: &[usize]) -> usize {
        sizes
            .iter()
            .map(|bytes| bytes.next_multiple_of(ALIGN))
            .sum::<usize>()
            .max(ALIGN)
    }

    /// A checkpoint over regions of `sizes` bytes, in that order, in `arena`, which holds at least
    /// [Checkpoint::arena_bytes] of them. It holds no point until the first [Checkpoint::save].
    #[must_use]
    pub fn new(arena: Deviceptr, sizes: Vec<usize>) -> Self {
        Self {
            arena,
            sizes,
            point: None,
        }
    }

    /// The arena, for the arm that frees it.
    #[must_use]
    pub fn arena(&self) -> Deviceptr {
        self.arena
    }

    /// The position the copies hold, when they hold one.
    #[must_use]
    pub fn point(&self) -> Option<usize> {
        self.point
    }

    /// The copies no longer hold a point: the sequence's state was reset, or a prompt began to
    /// forward past the point and will save its own end.
    pub fn forget(&mut self) {
        self.point = None;
    }

    /// Copies the `live` regions into the arena on `stream` (the default stream when `None`), as
    /// the state at `point`.
    ///
    /// # Errors
    /// A region list unlike the layout, or a driver failure (the copies then hold no point).
    pub fn save(
        &mut self,
        cuda: &Cuda,
        stream: Option<&RunStream>,
        live: &[Deviceptr],
        point: usize,
    ) -> Result<(), String> {
        self.save_with(live, point, |dst, src, bytes| {
            copy(cuda, stream, dst, src, bytes)
        })
    }

    /// Copies the arena back into the `live` regions on `stream`, putting the state back to
    /// `point`. The copies keep the point, so a sequence may go back to it again.
    ///
    /// # Errors
    /// A point the copies do not hold, a region list unlike the layout, or a driver failure.
    pub fn restore(
        &self,
        cuda: &Cuda,
        stream: Option<&RunStream>,
        live: &[Deviceptr],
        point: usize,
    ) -> Result<(), String> {
        self.restore_with(live, point, |dst, src, bytes| {
            copy(cuda, stream, dst, src, bytes)
        })
    }

    /// [Checkpoint::save] through `copy(dst, src, bytes)`.
    ///
    /// # Errors
    /// A region list unlike the layout, or a copy's failure.
    pub fn save_with(
        &mut self,
        live: &[Deviceptr],
        point: usize,
        mut copy: impl FnMut(&Deviceptr, &Deviceptr, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        self.point = None;
        for (kept, region, bytes) in self.regions(live)? {
            copy(&kept, region, bytes)?;
        }
        self.point = Some(point);
        Ok(())
    }

    /// [Checkpoint::restore] through `copy(dst, src, bytes)`.
    ///
    /// # Errors
    /// A point the copies do not hold, a region list unlike the layout, or a copy's failure.
    pub fn restore_with(
        &self,
        live: &[Deviceptr],
        point: usize,
        mut copy: impl FnMut(&Deviceptr, &Deviceptr, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        if self.point != Some(point) {
            return Err(match self.point {
                Some(held) => format!("the checkpoint holds position {held}, not {point}"),
                None => format!("the checkpoint holds no position to go back to {point} from"),
            });
        }
        for (kept, region, bytes) in self.regions(live)? {
            copy(region, &kept, bytes)?;
        }
        Ok(())
    }

    /// Each nonempty region's place in the arena, the live region and its bytes.
    fn regions<'a>(
        &self,
        live: &'a [Deviceptr],
    ) -> Result<Vec<(Deviceptr, &'a Deviceptr, usize)>, String> {
        if live.len() != self.sizes.len() {
            return Err(format!(
                "the checkpoint holds {} region(s) and was handed {}",
                self.sizes.len(),
                live.len()
            ));
        }
        let mut at = 0usize;
        let mut regions = Vec::with_capacity(live.len());
        for (region, &bytes) in live.iter().zip(&self.sizes) {
            if bytes > 0 {
                regions.push((self.arena.offset(at), region, bytes));
            }
            at += bytes.next_multiple_of(ALIGN);
        }
        Ok(regions)
    }
}

/// One copy on `stream`, or on the default stream.
/// A verify pass's snapshots of a stack's recurrent regions, a copy a pass row: the kernels that
/// advance a region over a pass's rows also leave it as it stood after each row (the Gated
/// DeltaNet conv and scan, under sparkflash-ops' `SegmentRows::snapshots`), so a verify that keeps fewer rows
/// than it ran commits the kept row's copy instead of running the rows again.
///
/// The slab holds, for each region in the order a lane lists them ([Checkpoint]'s sizes), `rows`
/// copies one region's bytes apart; a region's run starts 256-byte aligned. It is one pass's: the
/// copies are read before the next verify pass writes them.
#[derive(Debug, Clone)]
pub struct VerifySnapshots {
    slab: Deviceptr,
    sizes: Vec<usize>,
    rows: usize,
}

impl VerifySnapshots {
    /// The slab's bytes for regions of `sizes` bytes over `rows` pass rows (never zero, so the
    /// slab is a real allocation).
    #[must_use]
    pub fn slab_bytes(sizes: &[usize], rows: usize) -> usize {
        sizes
            .iter()
            .map(|bytes| (bytes * rows).next_multiple_of(ALIGN))
            .sum::<usize>()
            .max(ALIGN)
    }

    /// Snapshots in `slab` ([Self::slab_bytes] of `sizes` and `rows`).
    #[must_use]
    pub fn new(slab: Deviceptr, sizes: Vec<usize>, rows: usize) -> Self {
        Self { slab, sizes, rows }
    }

    /// The pass rows the slab holds.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The regions the slab holds.
    #[must_use]
    pub fn regions(&self) -> usize {
        self.sizes.len()
    }

    /// Region `region`'s copy of pass row `row`: a segment whose first row is `row` hands this as
    /// the region's snapshot slab, its later rows' copies a region's bytes apart.
    ///
    /// # Errors
    /// A region or a row the slab does not hold.
    pub fn slot(&self, region: usize, row: usize) -> Result<Deviceptr, String> {
        if region >= self.sizes.len() || row >= self.rows {
            return Err(format!(
                "the verify snapshots hold {} region(s) over {} row(s), not region {region} row {row}",
                self.sizes.len(),
                self.rows
            ));
        }
        let start: usize = self.sizes[..region]
            .iter()
            .map(|bytes| (bytes * self.rows).next_multiple_of(ALIGN))
            .sum();
        Ok(self.slab.offset(start + row * self.sizes[region]))
    }

    /// Copies every region's copy of pass row `row` into the sequence's `live` regions: its state
    /// after that row, the last a verify kept.
    ///
    /// # Errors
    /// A row the slab does not hold, a region count that is not the slab's, or the copy's failure.
    pub fn commit_with(
        &self,
        row: usize,
        live: &[Deviceptr],
        mut copy: impl FnMut(&Deviceptr, &Deviceptr, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        if live.len() != self.sizes.len() {
            return Err(format!(
                "the verify snapshots hold {} region(s) and were handed {}",
                self.sizes.len(),
                live.len()
            ));
        }
        for (region, (live, &bytes)) in live.iter().zip(&self.sizes).enumerate() {
            if bytes > 0 {
                copy(live, &self.slot(region, row)?, bytes)?;
            }
        }
        Ok(())
    }

    /// [Self::commit_with] on `stream` (the device's current stream when `None`).
    ///
    /// # Errors
    /// As [Self::commit_with].
    pub fn commit(
        &self,
        cuda: &Cuda,
        stream: Option<&RunStream>,
        row: usize,
        live: &[Deviceptr],
    ) -> Result<(), String> {
        self.commit_with(row, live, |dst, src, bytes| {
            copy(cuda, stream, dst, src, bytes)
        })
    }
}

fn copy(
    cuda: &Cuda,
    stream: Option<&RunStream>,
    dst: &Deviceptr,
    src: &Deviceptr,
    bytes: usize,
) -> Result<(), String> {
    match stream {
        Some(stream) => cuda.copy_dtod_on(stream, dst, src, bytes),
        None => cuda.copy_dtod_async(dst, src, bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each copy a test saw: destination, source, bytes.
    type Copies = std::cell::RefCell<Vec<(u64, u64, usize)>>;

    fn log() -> (Copies, impl Fn(&Copies, &Deviceptr, &Deviceptr, usize)) {
        (
            std::cell::RefCell::new(Vec::new()),
            |copies: &Copies, dst: &Deviceptr, src, bytes| {
                copies.borrow_mut().push((dst.0, src.0, bytes));
            },
        )
    }

    /// A save copies each region to its aligned place and a restore copies it back; an empty
    /// region takes no copy and no room.
    #[test]
    fn a_save_and_a_restore_copy_each_region_to_its_place() {
        let mut checkpoint = Checkpoint::new(Deviceptr(0x10000), vec![100, 0, 300]);
        let live = [Deviceptr(0xA000), Deviceptr(0xB000), Deviceptr(0xC000)];
        let (copies, record) = log();
        checkpoint
            .save_with(&live, 42, |dst, src, bytes| {
                record(&copies, dst, src, bytes);
                Ok(())
            })
            .expect("save");
        assert_eq!(
            copies.take(),
            vec![(0x10000, 0xA000, 100), (0x10000 + 256, 0xC000, 300)]
        );
        assert_eq!(checkpoint.point(), Some(42));
        checkpoint
            .restore_with(&live, 42, |dst, src, bytes| {
                record(&copies, dst, src, bytes);
                Ok(())
            })
            .expect("restore");
        assert_eq!(
            copies.take(),
            vec![(0xA000, 0x10000, 100), (0xC000, 0x10000 + 256, 300)]
        );
        // The point survives a rewind, so a sequence can go back to it again.
        assert_eq!(checkpoint.point(), Some(42));
    }

    /// A restore to a point the copies do not hold, or of other regions, copies nothing.
    #[test]
    fn a_commit_copies_the_kept_rows_snapshot_of_each_region() {
        let sizes = vec![100, 0, 300];
        assert_eq!(
            VerifySnapshots::slab_bytes(&sizes, 4),
            512 + 1200_usize.next_multiple_of(256)
        );
        let snapshots = VerifySnapshots::new(Deviceptr(0x10000), sizes, 4);
        assert_eq!(snapshots.slot(0, 2).unwrap().0, 0x10000 + 200);
        assert_eq!(snapshots.slot(2, 1).unwrap().0, 0x10000 + 512 + 300);
        assert!(snapshots.slot(0, 4).is_err() && snapshots.slot(3, 0).is_err());
        let live = [Deviceptr(0xA000), Deviceptr(0xB000), Deviceptr(0xC000)];
        let (copies, record) = log();
        snapshots
            .commit_with(3, &live, |dst, src, bytes| {
                record(&copies, dst, src, bytes);
                Ok(())
            })
            .expect("commit");
        assert_eq!(
            copies.take(),
            vec![
                (0xA000, 0x10000 + 300, 100),
                (0xC000, 0x10000 + 512 + 900, 300)
            ]
        );
        assert!(
            snapshots
                .commit_with(0, &live[..2], |_, _, _| Ok(()))
                .is_err()
        );
    }

    #[test]
    fn a_restore_needs_the_point_and_the_layout() {
        let mut checkpoint = Checkpoint::new(Deviceptr(0x10000), vec![64]);
        let live = [Deviceptr(0xA000)];
        let never = |_: &Deviceptr, _: &Deviceptr, _: usize| -> Result<(), String> {
            panic!("no copy may run")
        };
        assert!(checkpoint.restore_with(&live, 7, never).is_err());
        checkpoint
            .save_with(&live, 7, |_, _, _| Ok(()))
            .expect("save");
        assert!(checkpoint.restore_with(&live, 8, never).is_err());
        assert!(checkpoint.restore_with(&[], 7, never).is_err());
        checkpoint.forget();
        assert!(checkpoint.restore_with(&live, 7, never).is_err());
    }

    /// A save that fails part way holds no point.
    #[test]
    fn a_failed_save_holds_no_point() {
        let mut checkpoint = Checkpoint::new(Deviceptr(0x10000), vec![64, 64]);
        let live = [Deviceptr(0xA000), Deviceptr(0xB000)];
        checkpoint
            .save_with(&live, 3, |_, _, _| Ok(()))
            .expect("save");
        let failed = checkpoint.save_with(&live, 9, |_, src, _| {
            if src.0 == 0xB000 {
                Err("refused".to_owned())
            } else {
                Ok(())
            }
        });
        assert!(failed.is_err());
        assert_eq!(checkpoint.point(), None);
    }
}
