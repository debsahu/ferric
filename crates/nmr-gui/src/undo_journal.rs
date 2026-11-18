//! Unified, sequence-stamped undo / redo journal shared by user edits and MCP bridge commands,
//! plus a bounded, id-keyed multi-level snapshot stack for destructive spectrum operations
//! (`UI-C2`, `UI-C3`, `UI-C4`, `v0150:P1-6`, `20.W1:P2-5`).
//!
//! Provenance: original clean-room design. Every mutation (from the pointer, the keyboard,
//! a menu or the local MCP bridge) receives a monotonically increasing sequence number from a
//! single counter, so `Cmd/Ctrl+Z` always reverts whichever change happened most recently,
//! regardless of origin. Snapshot stacks are keyed by a stable spectrum identity (page id,
//! cube id or trace id) rather than by "whatever is active", so a 2D operation can never
//! restore into (or clobber) a 3D cube, and repeated operations (`ws, ws, wu, wu`) unwind level
//! by level back to the exact original.

use std::collections::{BTreeMap, VecDeque};

/// Who issued a journaled mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UndoOrigin {
    User,
    Mcp,
}

/// One journal entry: a sequence stamp, its origin, a human label and an app payload.
#[derive(Debug, Clone)]
pub struct JournalEntry<T> {
    pub seq: u64,
    pub origin: UndoOrigin,
    pub label: String,
    pub payload: T,
}

/// Single unified undo / redo journal. `T` is the app-specific reversible payload.
#[derive(Debug, Clone)]
pub struct UndoJournal<T> {
    next_seq: u64,
    capacity: usize,
    undo: VecDeque<JournalEntry<T>>,
    redo: Vec<JournalEntry<T>>,
}

impl<T> Default for UndoJournal<T> {
    fn default() -> Self {
        Self::with_capacity(default_undo_capacity())
    }
}

/// Default bounded depth: tighter on `wasm32` where snapshots live in browser memory.
pub const fn default_undo_capacity() -> usize {
    if cfg!(target_arch = "wasm32") {
        32
    } else {
        256
    }
}

impl<T> UndoJournal<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            next_seq: 1,
            capacity: capacity.max(1),
            undo: VecDeque::new(),
            redo: Vec::new(),
        }
    }

    /// Allocate the next global sequence stamp without recording an entry (used when another
    /// subsystem keeps its own history but must still be ordered against this journal).
    pub fn stamp(&mut self) -> u64 {
        let s = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        s
    }

    /// Record a new reversible mutation. Clears the redo branch. Returns its sequence stamp.
    pub fn record(&mut self, origin: UndoOrigin, label: impl Into<String>, payload: T) -> u64 {
        let seq = self.stamp();
        self.undo.push_back(JournalEntry {
            seq,
            origin,
            label: label.into(),
            payload,
        });
        while self.undo.len() > self.capacity {
            self.undo.pop_front();
        }
        self.redo.clear();
        seq
    }

    /// Most recent undoable entry (any origin).
    pub fn peek_undo(&self) -> Option<&JournalEntry<T>> {
        self.undo.back()
    }

    /// Most recent redoable entry.
    pub fn peek_redo(&self) -> Option<&JournalEntry<T>> {
        self.redo.last()
    }

    /// Sequence stamp of the latest undoable entry with `origin`.
    pub fn latest_seq(&self, origin: UndoOrigin) -> Option<u64> {
        self.undo
            .iter()
            .rev()
            .find(|e| e.origin == origin)
            .map(|e| e.seq)
    }

    /// Pop the most recent entry for undo. The caller applies the inverse and then calls
    /// [`Self::push_redo`] with the payload needed to re-apply it.
    pub fn pop_undo(&mut self) -> Option<JournalEntry<T>> {
        self.undo.pop_back()
    }

    /// Pop the most recent redo entry. The caller re-applies it and then calls
    /// [`Self::push_undo_after_redo`].
    pub fn pop_redo(&mut self) -> Option<JournalEntry<T>> {
        self.redo.pop()
    }

    pub fn push_redo(&mut self, entry: JournalEntry<T>) {
        self.redo.push(entry);
    }

    /// Re-insert an entry on the undo side after a redo (keeps its original stamp).
    pub fn push_undo_after_redo(&mut self, entry: JournalEntry<T>) {
        self.undo.push_back(entry);
        while self.undo.len() > self.capacity {
            self.undo.pop_front();
        }
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    /// Iterate undo entries oldest to newest.
    pub fn iter_undo(&self) -> impl Iterator<Item = &JournalEntry<T>> {
        self.undo.iter()
    }
}

/// Decide which of two independent histories owns the next `Cmd/Ctrl+Z`: the one whose latest
/// entry carries the larger (more recent) sequence stamp.
pub fn most_recent_origin(user_seq: Option<u64>, mcp_seq: Option<u64>) -> Option<UndoOrigin> {
    match (user_seq, mcp_seq) {
        (None, None) => None,
        (Some(_), None) => Some(UndoOrigin::User),
        (None, Some(_)) => Some(UndoOrigin::Mcp),
        (Some(u), Some(m)) => Some(if m > u {
            UndoOrigin::Mcp
        } else {
            UndoOrigin::User
        }),
    }
}

/// Stable identity of a spectrum targeted by a destructive snapshot operation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpectrumKey {
    /// A 1D trace, by its stable trace id.
    Trace1d(String),
    /// A 2D workbook page, by stable page label / id.
    Page2d(String),
    /// A 3D cube, by stable cube id.
    Cube3d(String),
}

/// Default bounded per-key snapshot depth (`wasm32`: 4, native: 16).
pub const fn default_snapshot_depth() -> usize {
    if cfg!(target_arch = "wasm32") {
        4
    } else {
        16
    }
}

/// Bounded, id-keyed multi-level snapshot stack. Each key owns an independent LIFO of
/// pre-operation snapshots, so `ws, ws, wu, wu` restores the intermediate and then the exact
/// original, and operations on one spectrum never restore into another.
#[derive(Debug, Clone)]
pub struct KeyedSnapshotStack<V> {
    depth: usize,
    stacks: BTreeMap<SpectrumKey, Vec<V>>,
}

impl<V> Default for KeyedSnapshotStack<V> {
    fn default() -> Self {
        Self::with_depth(default_snapshot_depth())
    }
}

impl<V> KeyedSnapshotStack<V> {
    pub fn with_depth(depth: usize) -> Self {
        Self {
            depth: depth.max(1),
            stacks: BTreeMap::new(),
        }
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Push a pre-operation snapshot for `key`, evicting the oldest beyond the depth bound.
    pub fn push(&mut self, key: SpectrumKey, snapshot: V) {
        let st = self.stacks.entry(key).or_default();
        st.push(snapshot);
        if st.len() > self.depth {
            let excess = st.len() - self.depth;
            st.drain(0..excess);
        }
    }

    /// Pop the most recent snapshot for `key`.
    pub fn pop(&mut self, key: &SpectrumKey) -> Option<V> {
        let st = self.stacks.get_mut(key)?;
        let v = st.pop();
        if st.is_empty() {
            self.stacks.remove(key);
        }
        v
    }

    pub fn levels(&self, key: &SpectrumKey) -> usize {
        self.stacks.get(key).map_or(0, Vec::len)
    }

    pub fn has(&self, key: &SpectrumKey) -> bool {
        self.levels(key) > 0
    }

    /// Total snapshots held across all keys (memory bound check).
    pub fn total_levels(&self) -> usize {
        self.stacks.values().map(Vec::len).sum()
    }

    pub fn remove_key(&mut self, key: &SpectrumKey) {
        self.stacks.remove(key);
    }

    pub fn clear(&mut self) {
        self.stacks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_orders_user_and_mcp_by_sequence() {
        let mut j: UndoJournal<&'static str> = UndoJournal::with_capacity(8);
        j.record(UndoOrigin::User, "move", "u1");
        j.record(UndoOrigin::Mcp, "add", "m1");
        assert_eq!(j.peek_undo().map(|e| e.origin), Some(UndoOrigin::Mcp));
        let e = j.pop_undo().expect("entry");
        assert_eq!(e.payload, "m1");
        j.push_redo(e);
        assert_eq!(j.peek_undo().map(|e| e.payload), Some("u1"));
        let r = j.pop_redo().expect("redo");
        j.push_undo_after_redo(r);
        assert_eq!(j.peek_undo().map(|e| e.payload), Some("m1"));
    }

    #[test]
    fn keyed_stack_is_lifo_per_key_and_bounded() {
        let mut s = KeyedSnapshotStack::with_depth(2);
        let k = SpectrumKey::Page2d("hsqc".into());
        s.push(k.clone(), 0);
        s.push(k.clone(), 1);
        s.push(k.clone(), 2);
        assert_eq!(s.levels(&k), 2);
        assert_eq!(s.pop(&k), Some(2));
        assert_eq!(s.pop(&k), Some(1));
        assert_eq!(s.pop(&k), None);
        assert!(!s.has(&SpectrumKey::Cube3d("hncacb".into())));
    }

    #[test]
    fn most_recent_origin_picks_larger_stamp() {
        assert_eq!(most_recent_origin(Some(3), Some(5)), Some(UndoOrigin::Mcp));
        assert_eq!(most_recent_origin(Some(7), Some(5)), Some(UndoOrigin::User));
        assert_eq!(most_recent_origin(None, None), None);
    }
}
