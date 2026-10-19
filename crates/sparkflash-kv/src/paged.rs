//! Serving paged allocation over the format-2 device KV pool (task-21, S3).
//!
//! [crate::device_store] owns one sequence's arrays and deliberately never allocates, evicts or
//! infers a position. A serving front end needs the other half: many sequences sharing one device
//! pool. This module is that half, pure Rust - pages, block tables, a free list, prefix sharing,
//! preemption and SWA tail slots. It never touches KV bytes; `sparkflash_device`'s paged path turns
//! a block table into the operand staging the device attention kernel consumes.
//!
//! # The unit it pages
//!
//! A page is [crate::config::KV_PAGE_TOKENS] (64) logical token positions, the device store's own
//! page unit: a Full layer at compression ratio r stores 64 / r entries per page, and a page id in a
//! block table is exactly the page index the store's arrays take ([crate::layout::PageId]). Logical
//! block b covers tokens [64b, 64b + 64); the sequence's block table lists the physical pages those
//! blocks live in. A partial final block still owns a page; it is simply not complete yet.
//!
//! # Policies (fixed here so the tests can be checked against the intent)
//!
//! * Allocation pops the free list LIFO; a freed page is pushed to the back.
//! * A complete block whose hash is already cached is *shared*: the append releases the page it just
//!   built and points its table at the cached page, incrementing that block's reference count.
//! * When the last table reference to a complete block drops, the page goes to the front of the free
//!   list but keeps its block hash while it waits there: a later sequence with the same block
//!   reuses the data instead of recomputing it. Allocating that page for a different block drops the
//!   hash first (that drop is the eviction this module counts).
//! * Only complete 64-token blocks are shared; a partial block's hash would change as it fills.
//!
//! # What is refused
//!
//! A page over capacity, a 65th tail slot, an unknown or duplicate sequence, a preempted sequence
//! preempted twice, a truncation that grows the sequence or does not end on a block boundary (a
//! partial truncation would need a page-splitting rule serving does not have), a fork beyond the
//! donor's complete blocks or of a partial block, and any state that violates
//! [PagedKvCache::check_invariants].
//!
//! # Device measurements
//!
//! Everything here is host arithmetic; the tests are the evidence. The device attention path in
//! `sparkflash-device` compiles and is host-tested for operand equality against a CPU twin, but its
//! device run is a window measurement and is recorded as unclaimed.

use std::collections::{HashMap, VecDeque};

use crate::config::KV_PAGE_TOKENS;
use crate::layout::PageId;
use crate::radix::block_hash;

/// A sequence handle in the serving front end. Opaque here: any u64 works.
pub type SeqId = u64;

/// Pool geometry: whole 64-token pages and SWA tail slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedGeometry {
    pages: u32,
    tails: u32,
}

impl PagedGeometry {
    /// A pool of `pages` pages and `tails` SWA tail slots.
    ///
    /// # Errors
    /// Zero pages or zero tails: a pool that cannot hold one sequence is a configuration error,
    /// not a cache that misses.
    pub fn new(pages: u32, tails: u32) -> Result<Self, PagingError> {
        if pages == 0 {
            return Err(PagingError::BadGeometry("a pool holds at least one page"));
        }
        if tails == 0 {
            return Err(PagingError::BadGeometry(
                "a pool holds at least one SWA tail slot",
            ));
        }
        Ok(Self { pages, tails })
    }

    /// Pages in the pool.
    #[must_use]
    pub const fn pages(self) -> u32 {
        self.pages
    }

    /// SWA tail slots (one per running sequence's 128-slot window).
    #[must_use]
    pub const fn tails(self) -> u32 {
        self.tails
    }
}

/// Whether a sequence currently holds pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// Holds pages and a tail slot.
    Running,
    /// Preempted: its pages and tail were returned and it must be recomputed.
    Preempted,
}

/// One sequence's logical-block to physical-page table.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockTable {
    pages: Vec<PageId>,
}

impl BlockTable {
    /// Blocks mapped so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Physical page of logical block `block`.
    #[must_use]
    pub fn page_for(&self, block: usize) -> Option<PageId> {
        self.pages.get(block).copied()
    }

    /// The whole table, in logical block order.
    #[must_use]
    pub fn pages(&self) -> &[PageId] {
        &self.pages
    }
}

/// One sequence in the pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedSequence {
    id: SeqId,
    state: SequenceState,
    tokens: Vec<u32>,
    table: BlockTable,
    tail: Option<u32>,
    preemptions: u32,
}

impl PagedSequence {
    /// The handle.
    #[must_use]
    pub const fn id(&self) -> SeqId {
        self.id
    }

    /// Running or preempted.
    #[must_use]
    pub const fn state(&self) -> SequenceState {
        self.state
    }

    /// Tokens the sequence owns, in logical order.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The block table.
    #[must_use]
    pub const fn table(&self) -> &BlockTable {
        &self.table
    }

    /// The SWA tail slot, or None when preempted.
    #[must_use]
    pub const fn tail(&self) -> Option<u32> {
        self.tail
    }

    /// Times this sequence was preempted.
    #[must_use]
    pub const fn preemptions(&self) -> u32 {
        self.preemptions
    }
}

/// A cached complete block: its page and how many block tables reference it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedBlock {
    page: PageId,
    refs: u32,
}

/// What an append did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AppendReport {
    /// Tokens appended.
    pub tokens: usize,
    /// Pages allocated for new blocks.
    pub allocated: usize,
    /// Complete blocks whose page was reused from the prefix cache.
    pub shared: usize,
}

/// What freeing (or preempting, or truncating) returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FreeReport {
    /// Pages the sequence returned to the free list (whether or not their data stayed cached).
    pub freed_pages: usize,
    /// Of `freed_pages`, the complete blocks whose hash survived on the free list for a prefix hit.
    pub retained_blocks: usize,
    /// Whether a tail slot was returned.
    pub freed_tail: bool,
}

/// Pool counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PagedStats {
    /// Pages the geometry holds.
    pub pages_total: u32,
    /// Pages on the free list.
    pub pages_free: u32,
    /// Complete blocks currently cached, referenced or waiting on the free list.
    pub blocks_cached: u32,
    /// Distinct pages referenced by a block table.
    pub pages_in_use: u32,
    /// Live sequences (running plus preempted).
    pub sequences: usize,
    /// Running sequences.
    pub running: usize,
    /// Tail slots the geometry holds.
    pub tails_total: u32,
    /// Tail slots free.
    pub tails_free: u32,
    /// Table references beyond the first for a shared block.
    pub shared_references: u32,
    /// Prefix hits so far.
    pub prefix_hits: u64,
    /// Cached blocks dropped because their page was reused for a different block.
    pub evictions: u64,
    /// Preemptions so far.
    pub preemptions: u64,
}

/// The paged serving cache over one device pool.
#[derive(Debug)]
pub struct PagedKvCache {
    geometry: PagedGeometry,
    free: VecDeque<PageId>,
    cached: HashMap<u64, CachedBlock>,
    sequences: HashMap<SeqId, PagedSequence>,
    free_tails: Vec<u32>,
    prefix_hits: u64,
    evictions: u64,
    preemptions: u64,
}

impl PagedKvCache {
    /// A cache whose pages and tail slots all start free.
    ///
    /// The free list is built so that page 0 is allocated first (the tests assert the exact order,
    /// so a reviewer can predict every id).
    #[must_use]
    pub fn new(geometry: PagedGeometry) -> Self {
        Self {
            geometry,
            // Oldest free page first: allocation is FIFO, so a page whose data is cached is
            // reused only after every never-used page, and a later sequence with that block can
            // still hit the hash (the vLLM free-queue order).
            free: (0..geometry.pages())
                .map(|page| PageId(u64::from(page)))
                .collect(),
            cached: HashMap::new(),
            sequences: HashMap::new(),
            free_tails: (0..geometry.tails()).rev().collect(),
            prefix_hits: 0,
            evictions: 0,
            preemptions: 0,
        }
    }

    /// The geometry.
    #[must_use]
    pub const fn geometry(&self) -> PagedGeometry {
        self.geometry
    }

    /// Registers an empty running sequence and gives it a tail slot.
    ///
    /// # Errors
    /// A duplicate id, or no free tail slot.
    pub fn add_sequence(&mut self, id: SeqId) -> Result<(), PagingError> {
        if self.sequences.contains_key(&id) {
            return Err(PagingError::DuplicateSequence(id));
        }
        let tail = self.free_tails.pop().ok_or(PagingError::TailCapacity)?;
        self.sequences.insert(
            id,
            PagedSequence {
                id,
                state: SequenceState::Running,
                tokens: Vec::new(),
                table: BlockTable::default(),
                tail: Some(tail),
                preemptions: 0,
            },
        );
        Ok(())
    }

    /// One sequence, read-only.
    #[must_use]
    pub fn sequence(&self, id: SeqId) -> Option<&PagedSequence> {
        self.sequences.get(&id)
    }

    /// The block table of one sequence.
    ///
    /// # Errors
    /// An unknown sequence.
    pub fn table(&self, id: SeqId) -> Result<&BlockTable, PagingError> {
        self.sequences
            .get(&id)
            .map(PagedSequence::table)
            .ok_or(PagingError::UnknownSequence(id))
    }

    /// Appends `tokens` to the sequence, allocating a page for every block the append opens and
    /// sharing a complete block whose hash is already cached.
    ///
    /// # Errors
    /// An unknown sequence, or no page to allocate (a new block needs one before it can be hashed).
    pub fn append(&mut self, id: SeqId, tokens: &[u32]) -> Result<AppendReport, PagingError> {
        if !self.sequences.contains_key(&id) {
            return Err(PagingError::UnknownSequence(id));
        }
        let mut report = AppendReport::default();
        for &token in tokens {
            let needs_page = self.sequences[&id]
                .tokens
                .len()
                .is_multiple_of(KV_PAGE_TOKENS);
            if needs_page {
                // A preempted sequence gave its tail slot back; recomputing it needs a new one.
                // Take the tail first and return it if the page allocation fails, so neither the
                // free list nor the tail pool can lose an entry to a refused append.
                let tail = if self.sequences[&id].tail.is_none() {
                    Some(self.free_tails.pop().ok_or(PagingError::TailCapacity)?)
                } else {
                    None
                };
                let page = match self.allocate_page() {
                    Ok(page) => page,
                    Err(error) => {
                        if let Some(tail) = tail {
                            self.free_tails.push(tail);
                        }
                        return Err(error);
                    }
                };
                report.allocated += 1;
                let state = self.sequences.get_mut(&id).expect("checked above");
                state.table.pages.push(page);
                state.state = SequenceState::Running;
                if tail.is_some() {
                    state.tail = tail;
                }
            }
            let completed = {
                let state = self.sequences.get_mut(&id).expect("checked above");
                state.tokens.push(token);
                let len = state.tokens.len();
                (len.is_multiple_of(KV_PAGE_TOKENS)).then(|| {
                    let block = len / KV_PAGE_TOKENS - 1;
                    let key = block_hash(&state.tokens[block * KV_PAGE_TOKENS..len]);
                    (block, key)
                })
            };
            if let Some((block, key)) = completed {
                let own = self.sequences[&id].table.pages[block];
                match self.cached.get(&key).copied() {
                    Some(hit) if hit.page != own => {
                        // The block just built duplicates a cached one: give our page back and
                        // point the table at the cached page. A cached page whose last reference
                        // dropped is waiting on the free list, so take it off first.
                        self.retire_page(own);
                        self.detach_free(hit.page);
                        let state = self.sequences.get_mut(&id).expect("checked above");
                        state.table.pages[block] = hit.page;
                        let entry = self.cached.get_mut(&key).expect("hit");
                        entry.refs += 1;
                        self.prefix_hits += 1;
                        report.shared += 1;
                    }
                    Some(_) => {
                        self.detach_free(own);
                        let entry = self.cached.get_mut(&key).expect("hit");
                        entry.refs += 1;
                    }
                    None => {
                        self.cached.insert(key, CachedBlock { page: own, refs: 1 });
                    }
                }
            }
            report.tokens += 1;
        }
        Ok(report)
    }

    /// Frees a sequence; complete blocks keep their data cached until their pages are reused.
    ///
    /// # Errors
    /// An unknown sequence.
    pub fn free_sequence(&mut self, id: SeqId) -> Result<FreeReport, PagingError> {
        let state = self
            .sequences
            .remove(&id)
            .ok_or(PagingError::UnknownSequence(id))?;
        Ok(self.release(&state))
    }

    /// Preempts a sequence: returns every page and its tail slot, clears the table and leaves the
    /// handle [SequenceState::Preempted] so the front end can recompute it.
    ///
    /// # Errors
    /// An unknown sequence, or one that is already preempted.
    pub fn preempt(&mut self, id: SeqId) -> Result<FreeReport, PagingError> {
        let state = self
            .sequences
            .get_mut(&id)
            .ok_or(PagingError::UnknownSequence(id))?;
        if state.state == SequenceState::Preempted {
            return Err(PagingError::AlreadyPreempted(id));
        }
        let mut released = state.clone();
        let report = self.release(&released);
        released.tokens.clear();
        released.table.pages.clear();
        released.tail = None;
        released.state = SequenceState::Preempted;
        released.preemptions += 1;
        let mut state = released;
        state.id = id;
        self.sequences.insert(id, state);
        self.preemptions += 1;
        Ok(report)
    }

    /// Gives a new sequence the first `blocks` complete blocks of `donor` as shared pages.
    ///
    /// # Errors
    /// Unknown donor, duplicate child, a preempted donor, a donor shorter than `blocks` blocks, or
    /// blocks that are not complete in the donor.
    pub fn fork(&mut self, donor: SeqId, child: SeqId, blocks: usize) -> Result<(), PagingError> {
        if blocks == 0 {
            return Err(PagingError::BadFork("a fork shares at least one block"));
        }
        if self.sequences.contains_key(&child) {
            return Err(PagingError::DuplicateSequence(child));
        }
        let (pages, tokens) = {
            let from = self
                .sequences
                .get(&donor)
                .ok_or(PagingError::UnknownSequence(donor))?;
            if from.state != SequenceState::Running {
                return Err(PagingError::BadFork("the donor is preempted"));
            }
            if from.tokens.len() < blocks * KV_PAGE_TOKENS {
                return Err(PagingError::BadFork(
                    "the donor is shorter than the requested prefix",
                ));
            }
            (
                from.table.pages[..blocks].to_vec(),
                from.tokens[..blocks * KV_PAGE_TOKENS].to_vec(),
            )
        };
        let tail = self.free_tails.pop().ok_or(PagingError::TailCapacity)?;
        for (block, page) in pages.iter().enumerate() {
            let key = block_hash(&tokens[block * KV_PAGE_TOKENS..(block + 1) * KV_PAGE_TOKENS]);
            let entry = self
                .cached
                .get_mut(&key)
                .ok_or(PagingError::BadFork("a shared block is not cached"))?;
            debug_assert_eq!(entry.page, *page);
            entry.refs += 1;
        }
        self.sequences.insert(
            child,
            PagedSequence {
                id: child,
                state: SequenceState::Running,
                tokens,
                table: BlockTable { pages },
                tail: Some(tail),
                preemptions: 0,
            },
        );
        Ok(())
    }

    /// Truncates a sequence to `tokens`, which must be a whole number of blocks.
    ///
    /// # Errors
    /// Unknown sequence, a length beyond the current one, or a length that is not block-aligned.
    pub fn truncate(&mut self, id: SeqId, tokens: usize) -> Result<FreeReport, PagingError> {
        let state = self
            .sequences
            .get(&id)
            .ok_or(PagingError::UnknownSequence(id))?;
        if tokens > state.tokens.len() {
            return Err(PagingError::BadTruncate(
                "a truncation cannot grow the sequence",
            ));
        }
        if !tokens.is_multiple_of(KV_PAGE_TOKENS) {
            return Err(PagingError::BadTruncate(
                "a truncation must end on a block boundary",
            ));
        }
        let keep = tokens / KV_PAGE_TOKENS;
        if keep == state.table.pages.len() {
            return Ok(FreeReport::default());
        }
        let mut report = FreeReport::default();
        let dropped = {
            let state = self.sequences.get_mut(&id).expect("checked above");
            state.tokens.truncate(tokens);
            state.table.pages.split_off(keep)
        };
        for page in dropped {
            report.freed_pages += 1;
            if self.retire_page(page) {
                report.retained_blocks += 1;
            }
        }
        Ok(report)
    }

    /// Distinct pages referenced by a block table.
    #[must_use]
    pub fn referenced_pages(&self) -> Vec<PageId> {
        let mut pages: Vec<PageId> = self
            .sequences
            .values()
            .flat_map(|state| state.table.pages.iter().copied())
            .collect();
        pages.sort_unstable();
        pages.dedup();
        pages
    }

    /// The physical page holding logical token `logical` of a sequence.
    ///
    /// # Errors
    /// Unknown sequence, or a logical position the sequence does not hold.
    pub fn page_of_token(&self, id: SeqId, logical: usize) -> Result<PageId, PagingError> {
        let state = self
            .sequences
            .get(&id)
            .ok_or(PagingError::UnknownSequence(id))?;
        if logical >= state.tokens.len() {
            return Err(PagingError::OutOfRange {
                sequence: id,
                logical,
                held: state.tokens.len(),
            });
        }
        state
            .table
            .page_for(logical / KV_PAGE_TOKENS)
            .ok_or_else(|| {
                PagingError::Invariant(format!(
                    "sequence {id} has no page for block {}",
                    logical / KV_PAGE_TOKENS
                ))
            })
    }

    /// Counters computed from the live state.
    #[must_use]
    pub fn stats(&self) -> PagedStats {
        PagedStats {
            pages_total: self.geometry.pages(),
            pages_free: self.free.len() as u32,
            blocks_cached: self.cached.len() as u32,
            pages_in_use: self.referenced_pages().len() as u32,
            sequences: self.sequences.len(),
            running: self
                .sequences
                .values()
                .filter(|state| state.state == SequenceState::Running)
                .count(),
            tails_total: self.geometry.tails(),
            tails_free: self.free_tails.len() as u32,
            shared_references: self
                .cached
                .values()
                .map(|entry| entry.refs.saturating_sub(1))
                .sum(),
            prefix_hits: self.prefix_hits,
            evictions: self.evictions,
            preemptions: self.preemptions,
        }
    }

    /// Checks every paging invariant and names the first one that does not hold.
    ///
    /// # Errors
    /// [PagingError::Invariant] describing the violated rule.
    pub fn check_invariants(&self) -> Result<(), PagingError> {
        let total = self.geometry.pages();
        let mut free = vec![false; total as usize];
        // 1. The free list is unique and in range.
        for page in &self.free {
            let slot = usize::try_from(page.0)
                .map_err(|_| PagingError::Invariant("page id does not fit usize".to_owned()))?;
            if slot >= total as usize {
                return Err(PagingError::Invariant(format!(
                    "free page {} is beyond the {total}-page pool",
                    page.0
                )));
            }
            if std::mem::replace(&mut free[slot], true) {
                return Err(PagingError::Invariant(format!(
                    "page {} appears twice on the free list",
                    page.0
                )));
            }
        }
        // 2. Every block table is well formed; count the table references per page.
        let mut table_refs: HashMap<PageId, u32> = HashMap::new();
        for (id, state) in &self.sequences {
            let expected = state.tokens.len().div_ceil(KV_PAGE_TOKENS);
            if state.table.pages.len() != expected {
                return Err(PagingError::Invariant(format!(
                    "sequence {id} has {} tokens but {} block pages",
                    state.tokens.len(),
                    state.table.pages.len()
                )));
            }
            let mut in_table: Vec<PageId> = Vec::with_capacity(state.table.pages.len());
            for page in &state.table.pages {
                if in_table.contains(page) {
                    return Err(PagingError::Invariant(format!(
                        "sequence {id} names page {} twice",
                        page.0
                    )));
                }
                in_table.push(*page);
                if free[page.0 as usize] {
                    return Err(PagingError::Invariant(format!(
                        "page {} is both free and referenced by sequence {id}",
                        page.0
                    )));
                }
                *table_refs.entry(*page).or_default() += 1;
            }
            if state.state == SequenceState::Running && state.tail.is_none() {
                return Err(PagingError::Invariant(format!(
                    "running sequence {id} has no tail slot"
                )));
            }
            if state.state == SequenceState::Preempted
                && (!state.table.pages.is_empty() || !state.tokens.is_empty())
            {
                return Err(PagingError::Invariant(format!(
                    "preempted sequence {id} still holds state"
                )));
            }
        }
        // 3. A cached block's references match the block tables, and an unreferenced cached block's
        //    page is on the free list.
        let mut cached_pages: Vec<PageId> = Vec::new();
        for (key, entry) in &self.cached {
            let refs = table_refs.get(&entry.page).copied().unwrap_or(0);
            if refs != entry.refs {
                return Err(PagingError::Invariant(format!(
                    "cached block {key:016x} on page {} has {} references, tables name {refs}",
                    entry.page.0, entry.refs
                )));
            }
            if entry.refs == 0 && !free[entry.page.0 as usize] {
                return Err(PagingError::Invariant(format!(
                    "cached block {key:016x} on page {} has no reference and is not free",
                    entry.page.0
                )));
            }
            if cached_pages.contains(&entry.page) {
                return Err(PagingError::Invariant(format!(
                    "page {} is cached under two block hashes",
                    entry.page.0
                )));
            }
            cached_pages.push(entry.page);
        }
        // 4. A referenced page without a cached entry must be a sequence's partial final block.
        let mut partial: Vec<PageId> = self
            .sequences
            .values()
            .filter(|state| !state.tokens.len().is_multiple_of(KV_PAGE_TOKENS))
            .filter_map(|state| state.table.pages.last().copied())
            .collect();
        partial.sort_unstable();
        for page in 0..total {
            let page = PageId(u64::from(page));
            let refs = table_refs.get(&page).copied().unwrap_or(0);
            let is_cached = cached_pages.contains(&page);
            let is_partial = partial.contains(&page);
            if refs == 0 && !free[page.0 as usize] {
                return Err(PagingError::Invariant(format!(
                    "page {} is referenced by no table and is not on the free list",
                    page.0
                )));
            }
            if !is_cached && refs > 0 && !is_partial {
                return Err(PagingError::Invariant(format!(
                    "page {} is referenced by a table but is neither cached nor a partial block",
                    page.0
                )));
            }
            if is_cached && is_partial {
                return Err(PagingError::Invariant(format!(
                    "page {} is a partial block and also a cached complete block",
                    page.0
                )));
            }
        }
        drop(partial);
        // 5. Tail slots are unique and cover the pool.
        let mut tails = vec![false; self.geometry.tails() as usize];
        let mut running = 0usize;
        for state in self.sequences.values() {
            if let Some(tail) = state.tail {
                running += 1;
                let slot = usize::try_from(tail)
                    .map_err(|_| PagingError::Invariant("tail id does not fit usize".to_owned()))?;
                if slot >= tails.len() {
                    return Err(PagingError::Invariant(format!(
                        "tail {tail} is beyond the {}-slot pool",
                        tails.len()
                    )));
                }
                if std::mem::replace(&mut tails[slot], true) {
                    return Err(PagingError::Invariant(format!("tail {tail} is used twice")));
                }
            }
        }
        for tail in &self.free_tails {
            let slot = usize::try_from(*tail)
                .map_err(|_| PagingError::Invariant("tail id does not fit usize".to_owned()))?;
            if slot >= tails.len() || std::mem::replace(&mut tails[slot], true) {
                return Err(PagingError::Invariant(format!(
                    "free tail {tail} is out of range or listed twice"
                )));
            }
        }
        if running + self.free_tails.len() != self.geometry.tails() as usize {
            return Err(PagingError::Invariant(format!(
                "{} running tails + {} free tails do not cover the pool",
                running,
                self.free_tails.len()
            )));
        }
        Ok(())
    }

    /// Takes the oldest free page, dropping the block it used to cache.
    fn allocate_page(&mut self) -> Result<PageId, PagingError> {
        let page = self.free.pop_front().ok_or(PagingError::Capacity {
            pages: self.geometry.pages(),
        })?;
        if self.forget_block(page) {
            self.evictions += 1;
        }
        Ok(page)
    }

    /// Removes a cached block from the map when its page is going to be reused. Returns whether a
    /// block was dropped.
    fn forget_block(&mut self, page: PageId) -> bool {
        let stale = self
            .cached
            .iter()
            .find(|(_, entry)| entry.page == page)
            .map(|(key, _)| *key);
        if let Some(key) = stale {
            self.cached.remove(&key);
            true
        } else {
            false
        }
    }

    /// Takes a page off the free list (a cached block is becoming referenced again).
    fn detach_free(&mut self, page: PageId) {
        if let Some(index) = self.free.iter().position(|candidate| *candidate == page) {
            self.free.remove(index);
        }
    }

    /// Drops one table reference to `page`. Returns whether a complete block's hash survived (the
    /// page went to the free list with its data still addressable by its hash) rather than the page
    /// being an uncached partial block.
    fn retire_page(&mut self, page: PageId) -> bool {
        let key = self
            .cached
            .iter()
            .find(|(_, entry)| entry.page == page)
            .map(|(key, _)| *key);
        if let Some(key) = key {
            let entry = self.cached.get_mut(&key).expect("found above");
            entry.refs = entry.refs.saturating_sub(1);
            if entry.refs == 0 {
                self.free.push_back(page);
            }
            return true;
        }
        self.free.push_back(page);
        false
    }

    /// Releases a sequence's pages and tail slot.
    fn release(&mut self, state: &PagedSequence) -> FreeReport {
        let mut report = FreeReport::default();
        for page in &state.table.pages {
            report.freed_pages += 1;
            if self.retire_page(*page) {
                report.retained_blocks += 1;
            }
        }
        if let Some(tail) = state.tail {
            self.free_tails.push(tail);
            report.freed_tail = true;
        }
        report
    }
}

/// Paging refusals.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PagingError {
    /// A pool of zero pages or zero tails.
    #[error("bad paged geometry: {0}")]
    BadGeometry(&'static str),
    /// A duplicate sequence handle.
    #[error("sequence {0} already exists")]
    DuplicateSequence(SeqId),
    /// An unknown sequence handle.
    #[error("sequence {0} is not in the pool")]
    UnknownSequence(SeqId),
    /// No free page.
    #[error("the {pages}-page pool is full and no page is free")]
    Capacity {
        /// Pages in the pool.
        pages: u32,
    },
    /// No free SWA tail slot.
    #[error("the SWA tail pool is full")]
    TailCapacity,
    /// A logical position the sequence does not hold.
    #[error("sequence {sequence} holds {held} tokens, logical position {logical} is beyond them")]
    OutOfRange {
        /// Sequence handle.
        sequence: SeqId,
        /// Requested logical position.
        logical: usize,
        /// Tokens the sequence holds.
        held: usize,
    },
    /// A block table that violates a paging rule.
    #[error("paging invariant: {0}")]
    Invariant(String),
    /// An already-preempted sequence.
    #[error("sequence {0} is already preempted")]
    AlreadyPreempted(SeqId),
    /// A truncation that grows the sequence or is not block-aligned.
    #[error("bad truncation: {0}")]
    BadTruncate(&'static str),
    /// A fork that cannot apply.
    #[error("bad fork: {0}")]
    BadFork(&'static str),
    /// A compact candidate that is in the future of the query position.
    #[error("sequence {sequence}: candidate {candidate} is past query position {position}")]
    NonCausalCandidate {
        /// Sequence handle.
        sequence: SeqId,
        /// The candidate's logical position.
        candidate: u32,
        /// The query's logical position.
        position: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> PagedGeometry {
        PagedGeometry::new(4, 2).expect("geometry")
    }

    fn tokens(base: u32, count: usize) -> Vec<u32> {
        (0..count as u32).map(|offset| base + offset).collect()
    }

    #[test]
    fn the_invariant_check_catches_a_page_listed_twice() {
        let mut cache = PagedKvCache::new(geometry());
        cache.add_sequence(1).expect("add");
        cache.append(1, &tokens(0, 1)).expect("append");
        let page = cache.table(1).expect("table").page_for(0).expect("page");
        cache.free.push_back(page);
        assert!(matches!(
            cache.check_invariants(),
            Err(PagingError::Invariant(_))
        ));
    }

    #[test]
    fn the_invariant_check_catches_a_referenced_page_with_no_block() {
        let mut cache = PagedKvCache::new(geometry());
        cache.add_sequence(1).expect("add");
        cache.append(1, &tokens(0, KV_PAGE_TOKENS)).expect("append");
        cache.cached.clear();
        assert!(matches!(
            cache.check_invariants(),
            Err(PagingError::Invariant(_))
        ));
    }
}
