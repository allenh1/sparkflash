//! A radix prefix tree whose nodes are whole 64-token global KV pages.
//!
//! The tree is the index over the paged pool. Every non-root node owns exactly one
//! KV_PAGE_TOKENS-token block and exactly one PageId, and a child edge is keyed by the
//! FNV-1a hash of the child's 64 token ids. Keying by the block hash rather than by the first
//! token is what lets two sequences that share a prompt start but diverge inside the first
//! page coexist as siblings: sharing happens at whole pages, which is exactly the granularity
//! plan_request aligns cache hits to, so the tree can never report a partial page as cached.
//!
//! Reference counting is per page. A sequence attaches its matched path, which pins every
//! node on that path; eviction only ever removes a node with no children and no references,
//! so a page that a live sequence is reading can never be recycled underneath it.

use std::collections::{HashMap, HashSet};

use crate::config::KV_PAGE_TOKENS;
use crate::error::KvError;
use crate::layout::PageId;

/// Identifier of a node in the radix tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// FNV-1a hash of a 64-token block; the child-edge key.
///
/// The hash is only a lookup accelerator: every hit is confirmed by comparing the stored
/// block against the query, so a collision degrades to a miss rather than to wrong data.
#[must_use]
pub fn block_hash(block: &[u32]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for token in block {
        for byte in token.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// The longest cached prefix of a prompt.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PrefixMatch {
    /// Matched tokens, always a multiple of KV_PAGE_TOKENS.
    pub matched_tokens: usize,
    /// Matched nodes, root-most first.
    pub path: Vec<NodeId>,
    /// Pages backing path, in the same order.
    pub pages: Vec<PageId>,
}

impl PrefixMatch {
    /// True when nothing matched.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }
}

/// What an insert added to the tree.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InsertOutcome {
    /// Tokens that were already present before the insert.
    pub matched_tokens: usize,
    /// Nodes created by this insert, with their freshly minted pages.
    pub pages_added: Vec<(NodeId, PageId)>,
    /// The full path from the root through every node the insert touched.
    pub path: Vec<NodeId>,
}

#[derive(Debug, Clone)]
struct Node {
    tokens: Box<[u32]>,
    page: Option<PageId>,
    parent: Option<NodeId>,
    children: HashMap<u64, NodeId>,
    refs: u32,
    last_used: u64,
}

/// A 64-token page radix tree.
#[derive(Debug)]
pub struct RadixTree {
    nodes: HashMap<NodeId, Node>,
    page_index: HashMap<PageId, NodeId>,
    root: NodeId,
    next_node: u32,
    next_page: u64,
    clock: u64,
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    /// Creates an empty tree containing only the root sentinel.
    #[must_use]
    pub fn new() -> Self {
        let root = NodeId(0);
        let mut nodes = HashMap::new();
        nodes.insert(
            root,
            Node {
                tokens: Box::new([]),
                page: None,
                parent: None,
                children: HashMap::new(),
                refs: 0,
                last_used: 0,
            },
        );
        Self {
            nodes,
            page_index: HashMap::new(),
            root,
            next_node: 1,
            next_page: 0,
            clock: 0,
        }
    }

    /// The page granularity this tree indexes.
    #[must_use]
    pub const fn page_tokens(&self) -> usize {
        KV_PAGE_TOKENS
    }

    /// Number of page nodes, excluding the root.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len() - 1
    }

    /// True when no page is cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of pages currently indexed.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.page_index.len()
    }

    /// Id of the root sentinel.
    #[must_use]
    pub const fn root(&self) -> NodeId {
        self.root
    }

    /// Token block stored by a node, or None when the node is unknown.
    #[must_use]
    pub fn node_tokens(&self, node: NodeId) -> Option<&[u32]> {
        self.nodes.get(&node).map(|n| n.tokens.as_ref())
    }

    /// Page owned by a node, or None when the node is unknown or is the root.
    #[must_use]
    pub fn page_of(&self, node: NodeId) -> Option<PageId> {
        self.nodes.get(&node).and_then(|n| n.page)
    }

    /// Reference count of a node, or None when the node is unknown.
    #[must_use]
    pub fn refs_of(&self, node: NodeId) -> Option<u32> {
        self.nodes.get(&node).map(|n| n.refs)
    }

    /// Every page currently indexed, in unspecified order.
    #[must_use]
    pub fn all_pages(&self) -> Vec<PageId> {
        self.page_index.keys().copied().collect()
    }

    /// Pages pinned by at least one attached sequence.
    #[must_use]
    pub fn pinned_pages(&self) -> Vec<PageId> {
        self.page_index
            .iter()
            .filter(|(_, node)| self.nodes.get(node).is_some_and(|n| n.refs > 0))
            .map(|(page, _)| *page)
            .collect()
    }

    /// Unpinned leaf pages, i.e. the eviction candidates.
    #[must_use]
    pub fn evictable_pages(&self) -> Vec<PageId> {
        self.page_index
            .iter()
            .filter(|(_, node)| {
                self.nodes
                    .get(node)
                    .is_some_and(|n| n.refs == 0 && n.children.is_empty())
            })
            .map(|(page, _)| *page)
            .collect()
    }

    /// True when the tree indexes this page.
    #[must_use]
    pub fn contains_page(&self, page: PageId) -> bool {
        self.page_index.contains_key(&page)
    }

    /// Longest cached prefix of a prompt.
    #[must_use]
    pub fn match_prefix(&self, tokens: &[u32]) -> PrefixMatch {
        let mut result = PrefixMatch::default();
        let mut node = self.root;
        let mut offset = 0usize;
        while offset + KV_PAGE_TOKENS <= tokens.len() {
            let block = &tokens[offset..offset + KV_PAGE_TOKENS];
            let Some(current) = self.nodes.get(&node) else {
                break;
            };
            let Some(&child) = current.children.get(&block_hash(block)) else {
                break;
            };
            let Some(child_node) = self.nodes.get(&child) else {
                break;
            };
            if child_node.tokens.as_ref() != block {
                break;
            }
            let Some(page) = child_node.page else {
                break;
            };
            result.path.push(child);
            result.pages.push(page);
            offset += KV_PAGE_TOKENS;
            node = child;
        }
        result.matched_tokens = offset;
        result
    }

    /// Adds every whole 64-token block of a prompt that is not cached yet.
    ///
    /// A trailing partial page is deliberately not stored: plan_request aligns every hit down
    /// to a page boundary, so a partial page could never be reported as a hit anyway.
    pub fn insert(&mut self, tokens: &[u32]) -> InsertOutcome {
        let matched = self.match_prefix(tokens);
        let mut path = matched.path.clone();
        let mut pages_added = Vec::new();
        let mut node = path.last().copied().unwrap_or(self.root);
        let mut offset = matched.matched_tokens;
        while offset + KV_PAGE_TOKENS <= tokens.len() {
            let block = &tokens[offset..offset + KV_PAGE_TOKENS];
            let key = block_hash(block);
            let page = PageId(self.next_page);
            self.next_page += 1;
            let id = NodeId(self.next_node);
            self.next_node += 1;
            self.nodes.insert(
                id,
                Node {
                    tokens: block.to_vec().into_boxed_slice(),
                    page: Some(page),
                    parent: Some(node),
                    children: HashMap::new(),
                    refs: 0,
                    last_used: 0,
                },
            );
            self.page_index.insert(page, id);
            if let Some(parent) = self.nodes.get_mut(&node) {
                parent.children.insert(key, id);
            }
            path.push(id);
            pages_added.push((id, page));
            node = id;
            offset += KV_PAGE_TOKENS;
        }
        InsertOutcome {
            matched_tokens: matched.matched_tokens,
            pages_added,
            path,
        }
    }

    /// Looks up an existing child by exact block contents.
    ///
    /// The hash is only a filter; the stored ids are compared, so a collision reports a miss.
    #[must_use]
    pub fn child_for(&self, parent: NodeId, block: &[u32]) -> Option<NodeId> {
        let node = self.nodes.get(&parent)?;
        let child = *node.children.get(&block_hash(block))?;
        let child_node = self.nodes.get(&child)?;
        if child_node.tokens.as_ref() == block {
            Some(child)
        } else {
            None
        }
    }

    /// Adds one page node under a parent.
    ///
    /// This is the two-phase insert the cache uses: the page has already been written to the
    /// tier store by the time the index learns about it.
    ///
    /// # Panics
    /// Panics when the block is not exactly one page long or the parent is unknown.
    pub fn insert_page(&mut self, parent: NodeId, block: &[u32], page: PageId) -> NodeId {
        assert_eq!(
            block.len(),
            KV_PAGE_TOKENS,
            "page block must be one page long"
        );
        assert!(self.nodes.contains_key(&parent), "parent node must exist");
        let id = NodeId(self.next_node);
        self.next_node += 1;
        self.nodes.insert(
            id,
            Node {
                tokens: block.to_vec().into_boxed_slice(),
                page: Some(page),
                parent: Some(parent),
                children: HashMap::new(),
                refs: 0,
                last_used: 0,
            },
        );
        self.page_index.insert(page, id);
        if let Some(parent_node) = self.nodes.get_mut(&parent) {
            parent_node.children.insert(block_hash(block), id);
        }
        id
    }

    /// Pins every node on a path and refreshes its recency.
    ///
    /// # Errors
    /// Returns KvError::UnknownNode when a node was evicted before the attach.
    pub fn attach(&mut self, path: &[NodeId]) -> Result<(), KvError> {
        self.clock += 1;
        let now = self.clock;
        for node in path {
            let Some(entry) = self.nodes.get_mut(node) else {
                return Err(KvError::UnknownNode(*node));
            };
            entry.refs = entry.refs.saturating_add(1);
            entry.last_used = now;
        }
        Ok(())
    }

    /// Releases every node on a path.
    ///
    /// # Errors
    /// Returns KvError::UnknownNode or KvError::RefCountUnderflow when the path does not
    /// correspond to an attached sequence.
    pub fn release(&mut self, path: &[NodeId]) -> Result<(), KvError> {
        for node in path {
            let Some(entry) = self.nodes.get_mut(node) else {
                return Err(KvError::UnknownNode(*node));
            };
            if entry.refs == 0 {
                return Err(KvError::RefCountUnderflow(*node));
            }
            entry.refs -= 1;
        }
        Ok(())
    }

    /// Refreshes the recency of every node on a path without changing its reference count.
    pub fn touch(&mut self, path: &[NodeId]) {
        self.clock += 1;
        let now = self.clock;
        for node in path {
            if let Some(entry) = self.nodes.get_mut(node) {
                entry.last_used = now;
            }
        }
    }

    /// Checks the tree's structural and reference-count invariants.
    ///
    /// Everything here follows from how the paths are attached and released, so a violation
    /// means a mutation went wrong rather than a caller did. The checks are:
    ///
    /// * the page index and the node map describe the same set of pages, exactly once each;
    /// * every non-root node owns a page, has a parent that exists, and is the child its parent
    ///   indexes under the hash of its own block;
    /// * no node is referenced more often than its parent, because a path is attached and
    ///   released prefix-closed (the root is a sentinel that is never attached, so its own
    ///   count stays zero and edges into it are exempt);
    /// * an unreferenced node has no referenced descendant, which is the property that makes
    ///   eviction of an unreferenced leaf safe.
    ///
    /// # Errors
    /// Returns KvError::Corrupt naming the first invariant that does not hold.
    pub fn check_invariants(&self) -> Result<(), KvError> {
        let corrupt = |what: &str| KvError::Corrupt(format!("radix tree: {what}"));
        let Some(root) = self.nodes.get(&self.root) else {
            return Err(corrupt("the root node is missing"));
        };
        if root.page.is_some() {
            return Err(corrupt("the root node owns a page"));
        }
        if root.refs != 0 {
            return Err(corrupt("the root node is referenced"));
        }
        if self.nodes.len() != self.page_index.len() + 1 {
            return Err(corrupt(
                "the node count and the page index do not describe the same tree",
            ));
        }
        for (node_id, node) in &self.nodes {
            if *node_id == self.root {
                continue;
            }
            let Some(page) = node.page else {
                return Err(corrupt("a non-root node owns no page"));
            };
            if self.page_index.get(&page) != Some(node_id) {
                return Err(corrupt("the page index does not point back at its node"));
            }
            let Some(parent_id) = node.parent else {
                return Err(corrupt("a non-root node has no parent"));
            };
            let Some(parent) = self.nodes.get(&parent_id) else {
                return Err(corrupt("a node references a missing parent"));
            };
            // The root is a sentinel: it is never attached, so its own count stays zero while
            // the pages directly under it are pinned. Every other edge must be monotone.
            if parent_id != self.root && node.refs > parent.refs {
                return Err(corrupt(
                    "a node is referenced more often than its own parent",
                ));
            }
            if parent.children.get(&block_hash(&node.tokens)) != Some(node_id) {
                return Err(corrupt(
                    "a node is not indexed by its parent under its own block hash",
                ));
            }
            for (key, child) in &node.children {
                let Some(child_node) = self.nodes.get(child) else {
                    return Err(corrupt("a node references a missing child"));
                };
                if child_node.parent != Some(*node_id) {
                    return Err(corrupt("a child does not point back at its parent"));
                }
                if *key != block_hash(&child_node.tokens) {
                    return Err(corrupt(
                        "an edge is not keyed by the child's own block hash",
                    ));
                }
                if node.refs == 0 && child_node.refs > 0 {
                    return Err(corrupt("an unreferenced node has a referenced child"));
                }
            }
        }
        Ok(())
    }

    /// Removes one page if it is an unpinned leaf.
    ///
    /// Returns true when the page was removed.
    pub fn remove_page(&mut self, page: PageId) -> bool {
        let Some(&node) = self.page_index.get(&page) else {
            return false;
        };
        let evictable = self
            .nodes
            .get(&node)
            .is_some_and(|n| n.refs == 0 && n.children.is_empty());
        if !evictable {
            return false;
        }
        self.detach_node(node);
        true
    }

    /// Evicts up to the requested number of least-recently-used unpinned leaf pages.
    ///
    /// Returns the pages that are now free. It may return fewer than requested when nothing
    /// else is evictable, which is the signal that the pool is fully pinned.
    pub fn evict_lru(&mut self, wanted: usize) -> Vec<PageId> {
        self.evict_lru_avoiding(wanted, &HashSet::new())
    }

    /// Evicts least-recently-used unpinned leaf pages while keeping a set of pages in place.
    ///
    /// A caller that is still building a prefix must pass the pages it has not finished with:
    /// otherwise eviction can remove the very node the caller is descending through, and the
    /// next insert would attach to a node that no longer exists.
    pub fn evict_lru_avoiding(
        &mut self,
        wanted: usize,
        protected: &HashSet<PageId>,
    ) -> Vec<PageId> {
        let mut freed = Vec::new();
        while freed.len() < wanted {
            let victim = self
                .page_index
                .iter()
                .filter(|(page, node)| {
                    !protected.contains(page)
                        && self
                            .nodes
                            .get(node)
                            .is_some_and(|n| n.refs == 0 && n.children.is_empty())
                })
                .min_by_key(|(_, node)| {
                    self.nodes
                        .get(node)
                        .map_or((u64::MAX, u32::MAX), |n| (n.last_used, node.0))
                })
                .map(|(page, _)| *page);
            match victim {
                Some(page) => {
                    self.remove_page(page);
                    freed.push(page);
                }
                None => break,
            }
        }
        freed
    }

    /// Removes every page, returning the pages that were freed.
    pub fn clear(&mut self) -> Vec<PageId> {
        let pages = self.all_pages();
        self.nodes.clear();
        self.page_index.clear();
        self.root = NodeId(0);
        self.nodes.insert(
            self.root,
            Node {
                tokens: Box::new([]),
                page: None,
                parent: None,
                children: HashMap::new(),
                refs: 0,
                last_used: 0,
            },
        );
        self.next_node = 1;
        pages
    }

    fn detach_node(&mut self, node: NodeId) {
        let Some(entry) = self.nodes.remove(&node) else {
            return;
        };
        if let Some(page) = entry.page {
            self.page_index.remove(&page);
        }
        if let Some(parent) = entry.parent
            && let Some(parent_entry) = self.nodes.get_mut(&parent)
        {
            let key = block_hash(&entry.tokens);
            if parent_entry.children.get(&key) == Some(&node) {
                parent_entry.children.remove(&key);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(len: usize, seed: u32) -> Vec<u32> {
        (0..len).map(|i| seed + i as u32).collect()
    }

    #[test]
    fn shared_prefix_is_shared_and_branching_is_page_granular() {
        let mut tree = RadixTree::new();
        let a = tokens(192, 1000);
        let first = tree.insert(&a);
        assert_eq!(first.matched_tokens, 0);
        assert_eq!(first.pages_added.len(), 3);

        let mut b = a.clone();
        b[130] = 999_999;
        let second = tree.insert(&b);
        assert_eq!(second.matched_tokens, 128);
        assert_eq!(second.pages_added.len(), 1);
        assert_eq!(tree.page_count(), 4);

        let third = tree.match_prefix(&a);
        assert_eq!(third.matched_tokens, 192);
        assert_eq!(third.path.len(), 3);
    }

    #[test]
    fn a_pinned_page_is_never_evicted() {
        let mut tree = RadixTree::new();
        let a = tokens(128, 7);
        let inserted = tree.insert(&a);
        tree.attach(&inserted.path).expect("attach");
        assert!(tree.evict_lru(10).is_empty());
        assert_eq!(tree.page_count(), 2);
        tree.release(&inserted.path).expect("release");
        assert_eq!(tree.evict_lru(10).len(), 2);
        assert!(tree.is_empty());
    }

    /// A healthy tree, plus a handle on the three nodes it holds.
    fn attached_tree() -> (RadixTree, Vec<NodeId>) {
        let mut tree = RadixTree::new();
        let inserted = tree.insert(&tokens(192, 11));
        tree.attach(&inserted.path).expect("attach");
        tree.check_invariants().expect("a healthy tree must pass");
        (tree, inserted.path)
    }

    #[test]
    fn the_invariant_check_rejects_a_reference_count_above_its_parent() {
        // The shape a lost release would leave behind: a child held more often than the parent
        // that every attach touches first.
        let (mut tree, path) = attached_tree();
        let child = path[1];
        tree.nodes.get_mut(&child).expect("child").refs += 1;
        let error = tree
            .check_invariants()
            .expect_err("the skewed count must be caught");
        assert!(
            format!("{error}").contains("referenced more often"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn the_invariant_check_rejects_a_referenced_child_of_an_unreferenced_node() {
        let (mut tree, path) = attached_tree();
        // Release the parent only: attach and release are prefix-closed, so a live child under
        // a dead parent is exactly what a botched partial release produces.
        let parent = path[0];
        let child = path[1];
        tree.nodes.get_mut(&parent).expect("parent").refs = 0;
        tree.nodes.get_mut(&child).expect("child").refs = 1;
        let error = tree
            .check_invariants()
            .expect_err("the split must be caught");
        assert!(matches!(error, KvError::Corrupt(_)), "{error}");
    }

    #[test]
    fn the_invariant_check_rejects_a_page_index_that_does_not_point_back() {
        let (mut tree, path) = attached_tree();
        let node = path[1];
        let page = tree.nodes.get(&node).and_then(|n| n.page).expect("page");
        let other = path[0];
        tree.page_index.insert(page, other);
        let error = tree
            .check_invariants()
            .expect_err("an aliased page must be caught");
        assert!(
            format!("{error}").contains("does not point back"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn the_invariant_check_rejects_a_node_missing_from_the_page_index() {
        let (mut tree, path) = attached_tree();
        let node = path[2];
        let page = tree.nodes.get(&node).and_then(|n| n.page).expect("page");
        tree.page_index.remove(&page);
        let error = tree
            .check_invariants()
            .expect_err("a dropped page index entry must be caught");
        assert!(matches!(error, KvError::Corrupt(_)), "{error}");
    }

    #[test]
    fn the_invariant_check_rejects_an_edge_that_is_not_keyed_by_its_block() {
        let (mut tree, path) = attached_tree();
        let parent = path[0];
        let child = path[1];
        tree.nodes
            .get_mut(&parent)
            .expect("parent")
            .children
            .insert(0xDEAD_BEEF, child);
        let error = tree
            .check_invariants()
            .expect_err("a mis-keyed edge must be caught");
        assert!(
            format!("{error}").contains("block hash"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn the_invariant_check_accepts_a_healthy_branching_tree() {
        // The negative control: the same checks must not fire on a tree that shares and
        // branches, including a released suffix.
        let (mut tree, path) = attached_tree();
        let mut other = tokens(192, 11);
        other[70] = 9_999;
        let inserted = tree.insert(&other);
        assert_eq!(inserted.matched_tokens, KV_PAGE_TOKENS);
        tree.attach(&inserted.path).expect("attach branch");
        tree.release(&path[1..]).expect("release a suffix");
        tree.check_invariants().expect("a branching tree must pass");
    }
}
