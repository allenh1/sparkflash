//! Randomised insert/lookup/attach/evict sequences checked against a naive model.
//!
//! The model is deliberately naive: a node's identity is its whole block path, children are
//! found by linear scan, and every policy decision is recomputed rather than shared with the
//! tree. It covers the one-shot insert, the two-phase insert_page walk the cache itself uses,
//! prefix matching, attach and release (the reference-count transitions), touch, both eviction
//! entry points including the protected set, and remove_page. Page ids are modelled too, so a
//! tree that aliases two nodes onto one page fails here even when the page count still agrees,
//! and every step runs the tree's own invariant check.
//!
//! It deliberately does not cover clear(), whose contract is one call rather than a policy, and
//! it says nothing about the cache-level locking that wraps these calls; the tests under
//! tests/concurrency.rs own that.

use std::collections::HashSet;

use sparkflash_kv::NodeId;
use sparkflash_kv::config::KV_PAGE_TOKENS;
use sparkflash_kv::layout::PageId;
use sparkflash_kv::radix::RadixTree;

/// Page ids the two-phase arm mints for itself. The tree mints its own ids for a one-shot
/// insert, so the test keeps the two ranges far apart and a collision cannot mask aliasing.
const TWO_PHASE_PAGE_BASE: u64 = 1 << 32;

/// Deterministic SplitMix64 so a failing iteration is reproducible from its seed.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() % bound as u64) as usize
        }
    }
}

fn blocks_of(tokens: &[u32]) -> Vec<Vec<u32>> {
    let (chunks, _partial) = tokens.as_chunks::<KV_PAGE_TOKENS>();
    chunks.iter().map(|chunk| chunk.to_vec()).collect()
}

#[derive(Clone)]
struct ModelNode {
    blocks: Vec<Vec<u32>>,
    /// The page id the tree minted for this node, so page identity is compared, not just
    /// how many pages exist.
    page: u64,
    refs: u32,
    last_used: u64,
    alive: bool,
}

/// A deliberately naive prefix tree: linear scans, full-path identity, O(n^2) child tests.
#[derive(Default)]
struct NaiveTree {
    nodes: Vec<ModelNode>,
    clock: u64,
}

impl NaiveTree {
    fn find(&self, path: &[Vec<u32>]) -> Option<usize> {
        self.nodes
            .iter()
            .position(|node| node.alive && node.blocks == path)
    }

    fn matched_blocks(&self, blocks: &[Vec<u32>]) -> usize {
        let mut best = 0;
        for len in 1..=blocks.len() {
            if self.find(&blocks[..len]).is_some() {
                best = len;
            }
        }
        best
    }

    /// Mirrors RadixTree::insert, taking the page ids the tree actually minted.
    fn insert(&mut self, blocks: &[Vec<u32>], pages: &[PageId]) -> (usize, usize) {
        let matched = self.matched_blocks(blocks);
        let mut added = 0;
        for len in matched + 1..=blocks.len() {
            let path = blocks[..len].to_vec();
            if self.find(&path).is_none() {
                self.nodes.push(ModelNode {
                    blocks: path,
                    page: pages[added].0,
                    refs: 0,
                    last_used: 0,
                    alive: true,
                });
                added += 1;
            }
        }
        assert_eq!(added, pages.len(), "model insert page count");
        (matched, added)
    }

    /// Mirrors RadixTree::child_for.
    fn child_for(&self, path: &[Vec<u32>], block: &[u32]) -> Option<usize> {
        let mut child = path.to_vec();
        child.push(block.to_vec());
        self.find(&child)
    }

    /// Mirrors RadixTree::insert_page: one block, one explicit page, no prefix search.
    fn insert_page(&mut self, path: &[Vec<u32>], block: &[u32], page: u64) {
        let mut child = path.to_vec();
        child.push(block.to_vec());
        assert!(
            self.find(&child).is_none(),
            "the model was asked to insert a duplicate page"
        );
        self.nodes.push(ModelNode {
            blocks: child,
            page,
            refs: 0,
            last_used: 0,
            alive: true,
        });
    }

    fn attach(&mut self, blocks: &[Vec<u32>]) {
        self.clock += 1;
        let now = self.clock;
        for len in 1..=blocks.len() {
            if let Some(index) = self.find(&blocks[..len]) {
                self.nodes[index].refs += 1;
                self.nodes[index].last_used = now;
            }
        }
    }

    fn release(&mut self, blocks: &[Vec<u32>]) {
        for len in 1..=blocks.len() {
            if let Some(index) = self.find(&blocks[..len]) {
                assert!(self.nodes[index].refs > 0, "model refcount underflow");
                self.nodes[index].refs -= 1;
            }
        }
    }

    /// Mirrors RadixTree::touch for the last `keep` nodes of a path.
    fn touch_suffix(&mut self, blocks: &[Vec<u32>], keep: usize) {
        self.clock += 1;
        let now = self.clock;
        for len in blocks.len().saturating_sub(keep) + 1..=blocks.len() {
            if let Some(index) = self.find(&blocks[..len]) {
                self.nodes[index].last_used = now;
            }
        }
    }

    /// The reference count the model holds for one full block path.
    fn refs_of(&self, path: &[Vec<u32>]) -> u32 {
        self.find(path).map_or(0, |index| self.nodes[index].refs)
    }

    fn has_child(&self, index: usize) -> bool {
        let path = &self.nodes[index].blocks;
        self.nodes.iter().any(|node| {
            node.alive && node.blocks.len() > path.len() && node.blocks[..path.len()] == path[..]
        })
    }

    fn evict_lru(&mut self, wanted: usize) -> usize {
        self.evict_lru_avoiding(wanted, &HashSet::new())
    }

    /// Mirrors RadixTree::evict_lru_avoiding, protecting by page id.
    fn evict_lru_avoiding(&mut self, wanted: usize, protected: &HashSet<u64>) -> usize {
        let mut freed = 0;
        while freed < wanted {
            let victim = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(index, node)| {
                    node.alive
                        && node.refs == 0
                        && !protected.contains(&node.page)
                        && !self.has_child(*index)
                })
                .min_by_key(|(index, node)| (node.last_used, *index))
                .map(|(index, _)| index);
            match victim {
                Some(index) => {
                    self.nodes[index].alive = false;
                    freed += 1;
                }
                None => break,
            }
        }
        freed
    }

    /// Mirrors RadixTree::remove_page.
    fn remove_page(&mut self, page: u64) -> bool {
        let Some(index) = self
            .nodes
            .iter()
            .position(|node| node.alive && node.page == page)
        else {
            return false;
        };
        if self.nodes[index].refs != 0 || self.has_child(index) {
            return false;
        }
        self.nodes[index].alive = false;
        true
    }

    fn alive(&self) -> usize {
        self.nodes.iter().filter(|node| node.alive).count()
    }

    fn pinned(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.alive && node.refs > 0)
            .count()
    }

    fn evictable(&self) -> usize {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(index, node)| node.alive && node.refs == 0 && !self.has_child(*index))
            .count()
    }

    fn evictable_pages(&self) -> Vec<u64> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(index, node)| node.alive && node.refs == 0 && !self.has_child(*index))
            .map(|(_, node)| node.page)
            .collect()
    }

    fn live_pages(&self) -> Vec<u64> {
        self.nodes
            .iter()
            .filter(|node| node.alive)
            .map(|node| node.page)
            .collect()
    }
}

fn assert_agree(tree: &RadixTree, model: &NaiveTree, context: &str) {
    tree.check_invariants()
        .unwrap_or_else(|error| panic!("tree invariants after {context}: {error}"));
    assert_eq!(
        tree.page_count(),
        model.alive(),
        "page count after {context}"
    );
    assert_eq!(
        tree.is_empty(),
        model.alive() == 0,
        "emptiness after {context}"
    );
    assert_eq!(
        tree.pinned_pages().len(),
        model.pinned(),
        "pinned pages after {context}"
    );
    assert_eq!(
        tree.evictable_pages().len(),
        model.evictable(),
        "evictable pages after {context}"
    );
    // Page identity, not just how many pages there are: a tree that aliases two nodes onto
    // one page keeps the count right and still fails here.
    let mut tree_pages: Vec<u64> = tree.all_pages().iter().map(|page| page.0).collect();
    tree_pages.sort_unstable();
    let mut model_pages = model.live_pages();
    model_pages.sort_unstable();
    assert_eq!(tree_pages, model_pages, "page ids after {context}");
}

/// Compares the tree's reference count with the model's for every node of every live sequence.
fn assert_refcounts(
    tree: &RadixTree,
    model: &NaiveTree,
    held: &[(Vec<NodeId>, Vec<Vec<u32>>)],
    context: &str,
) {
    for (path, blocks) in held {
        for (depth, node) in path.iter().enumerate() {
            assert_eq!(
                tree.refs_of(*node),
                Some(model.refs_of(&blocks[..=depth])),
                "{context}: reference count at depth {depth}"
            );
        }
    }
}

#[test]
fn randomised_operations_match_a_naive_model() {
    for seed in [1u64, 0xDEAD_BEEF, 0x1234_5678_9ABC_DEF0] {
        let mut rng = Rng(seed);
        let mut bases: Vec<Vec<u32>> = Vec::new();
        for _ in 0..8 {
            let len = rng.below(320);
            bases.push((0..len).map(|_| (rng.next_u64() % 13) as u32).collect());
        }
        let mut tree = RadixTree::new();
        let mut model = NaiveTree::default();
        let mut held: Vec<(Vec<NodeId>, Vec<Vec<u32>>)> = Vec::new();
        let mut two_phase_next = TWO_PHASE_PAGE_BASE;

        for step in 0..600 {
            let context = format!("seed {seed} step {step}");
            let mut tokens = bases[rng.below(bases.len())].clone();
            if !tokens.is_empty() && rng.below(3) == 0 {
                let position = rng.below(tokens.len());
                tokens[position] = (rng.next_u64() % 13) as u32 + 100;
            }
            let blocks = blocks_of(&tokens);
            match rng.below(13) {
                0..=1 => {
                    let outcome = tree.insert(&tokens);
                    let pages: Vec<PageId> =
                        outcome.pages_added.iter().map(|(_, page)| *page).collect();
                    let (matched, added) = model.insert(&blocks, &pages);
                    assert_eq!(
                        outcome.matched_tokens,
                        matched * KV_PAGE_TOKENS,
                        "{context}: matched tokens"
                    );
                    assert_eq!(outcome.pages_added.len(), added, "{context}: pages added");
                    assert_eq!(outcome.path.len(), matched + added);
                }
                2..=3 => {
                    let matched = tree.match_prefix(&tokens);
                    assert_eq!(
                        matched.matched_tokens,
                        model.matched_blocks(&blocks) * KV_PAGE_TOKENS,
                        "{context}: match"
                    );
                    assert_eq!(matched.path.len(), matched.matched_tokens / KV_PAGE_TOKENS);
                    assert_eq!(matched.pages.len(), matched.path.len());
                }
                4 => {
                    // The two-phase walk the cache itself performs: look the child up, and
                    // only mint a page when the block is genuinely new.
                    let matched = tree.match_prefix(&tokens);
                    let mut node = matched.path.last().copied().unwrap_or_else(|| tree.root());
                    let mut model_path: Vec<Vec<u32>> = blocks[..matched.path.len()].to_vec();
                    for block in &blocks[matched.path.len()..] {
                        let tree_child = tree.child_for(node, block);
                        let model_child = model.child_for(&model_path, block);
                        assert_eq!(
                            tree_child.is_some(),
                            model_child.is_some(),
                            "{context}: child lookup disagreed"
                        );
                        match (tree_child, model_child) {
                            (Some(next), Some(index)) => {
                                node = next;
                                model_path = model.nodes[index].blocks.clone();
                            }
                            (None, None) => {
                                let page = PageId(two_phase_next);
                                two_phase_next += 1;
                                node = tree.insert_page(node, block, page);
                                model.insert_page(&model_path, block, page.0);
                                model_path.push(block.clone());
                            }
                            _ => unreachable!("the lookup comparison above already failed"),
                        }
                    }
                }
                5 => {
                    let outcome = tree.insert(&tokens);
                    let pages: Vec<PageId> =
                        outcome.pages_added.iter().map(|(_, page)| *page).collect();
                    model.insert(&blocks, &pages);
                    tree.attach(&outcome.path).expect("attach");
                    model.attach(&blocks);
                    held.push((outcome.path, blocks));
                }
                6 => {
                    if let Some((path, blocks)) = held.pop() {
                        tree.release(&path).expect("release");
                        model.release(&blocks);
                    }
                }
                7..=8 => {
                    let wanted = rng.below(4);
                    let freed = tree.evict_lru(wanted);
                    let modelled = model.evict_lru(wanted);
                    assert_eq!(freed.len(), modelled, "{context}: evicted page count");
                    assert_eq!(
                        freed.len(),
                        freed.iter().collect::<HashSet<_>>().len(),
                        "{context}: eviction returned a duplicate page"
                    );
                }
                9 => {
                    let inserted = tree.insert(&tokens);
                    let pages: Vec<PageId> =
                        inserted.pages_added.iter().map(|(_, page)| *page).collect();
                    model.insert(&blocks, &pages);
                    let keep = rng.below(3).min(inserted.path.len());
                    if keep > 0 {
                        let tail = &inserted.path[inserted.path.len() - keep..];
                        tree.touch(tail);
                        model.touch_suffix(&blocks, keep);
                    }
                }
                10 => {
                    // Protected eviction: pick one evictable page and require it to survive.
                    let candidates = model.evictable_pages();
                    let protected: HashSet<u64> = if candidates.is_empty() {
                        HashSet::new()
                    } else {
                        HashSet::from([candidates[rng.below(candidates.len())]])
                    };
                    let wanted = rng.below(4);
                    let protected_pages: HashSet<PageId> =
                        protected.iter().map(|page| PageId(*page)).collect();
                    let freed = tree.evict_lru_avoiding(wanted, &protected_pages);
                    let modelled = model.evict_lru_avoiding(wanted, &protected);
                    assert_eq!(
                        freed.len(),
                        modelled,
                        "{context}: protected eviction freed a different number of pages"
                    );
                    for page in &freed {
                        assert!(
                            !protected.contains(&page.0),
                            "{context}: a protected page was evicted"
                        );
                    }
                }
                _ => {
                    let pages = tree.all_pages();
                    if !pages.is_empty() {
                        let page = pages[rng.below(pages.len())];
                        let removed = tree.remove_page(page);
                        assert_eq!(
                            removed,
                            model.remove_page(page.0),
                            "{context}: remove_page({})",
                            page.0
                        );
                    }
                }
            }
            assert_agree(&tree, &model, &context);
            assert_refcounts(&tree, &model, &held, &context);
        }

        while let Some((path, blocks)) = held.pop() {
            tree.release(&path).expect("release");
            model.release(&blocks);
        }
        let remaining = model.alive();
        assert_eq!(tree.evict_lru(usize::MAX).len(), remaining);
        model.evict_lru(usize::MAX);
        assert_agree(&tree, &model, "final eviction");
        assert!(tree.is_empty(), "seed {seed}: tree not empty at the end");
        assert_eq!(tree.page_count(), 0);
    }
}

#[test]
fn a_partial_trailing_page_is_never_reported_as_a_hit() {
    let mut tree = RadixTree::new();
    let tokens: Vec<u32> = (0..100).collect();
    let inserted = tree.insert(&tokens);
    assert_eq!(inserted.pages_added.len(), 1);
    let matched = tree.match_prefix(&tokens);
    assert_eq!(matched.matched_tokens, 64);
    assert_eq!(matched.matched_tokens % KV_PAGE_TOKENS, 0);
}

#[test]
fn divergent_sequences_that_share_a_first_page_branch_at_the_block_hash() {
    let mut tree = RadixTree::new();
    let a: Vec<u32> = (0..192).collect();
    let mut b = a.clone();
    b[70] = 9_999;
    let first = tree.insert(&a);
    assert_eq!(first.pages_added.len(), 3);
    let outcome = tree.insert(&b);
    assert_eq!(outcome.matched_tokens, 64, "only the first page is shared");
    // b's third block is byte-identical to a's, but it hangs off a different parent, so the
    // radix key is the whole path rather than the block: two nodes, two pages, two copies.
    assert_eq!(outcome.pages_added.len(), 2);
    assert_eq!(tree.page_count(), 5);
    assert_eq!(tree.match_prefix(&a).matched_tokens, 192);
    assert_eq!(tree.match_prefix(&b).matched_tokens, 192);
    assert_ne!(
        tree.match_prefix(&a).pages[2],
        tree.match_prefix(&b).pages[2],
        "identical blocks under different parents are different pages"
    );
}
