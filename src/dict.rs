//! Term interning: the in-memory faulting dictionary mapping a gram to a permanent
//! `u32` term-id.
//!
//! Postings (`post`/`delta`/`term`-df/`fwd`) are keyed by the `u32` id, not the gram
//! text — narrow integer B-trees, fast point seeks, and a smaller index. The gram
//! encoding lives only in the `dict` table. Ids are **monotonic and permanent**: once
//! assigned, a gram keeps its id until a [`rebuild`](crate::Index::rebuild) reassigns
//! the whole space (which bumps the dictionary generation, §coherence).
//!
//! **Reader/writer fault split (a correctness contract the types enforce):**
//! - A reader calls [`Dictionary::resolve`] — a lock-free-ish read that returns `None`
//!   on a miss (the gram is absent from the corpus, an empty posting). It never
//!   mutates, allocates, or writes.
//! - A writer takes an [`InternStage`] (only reachable on a write path, which holds the
//!   single-writer lease) and calls [`InternStage::intern`], which allocates an id and
//!   persists a `dict` row inside the write transaction. The staged grams enter the
//!   shared in-memory map only via [`InternStage::commit`], called **after** the
//!   transaction commits — so a rolled-back write leaves no orphan id.
//!
//! The dict key is produced by [`gram_key`] — the single seam where gram text becomes
//! the stored key (P2 swaps this for the `u128` term encoding without touching the rest
//! of this module).

use std::collections::HashMap;
use std::sync::{PoisonError, RwLock};

use rusqlite::Connection;

use crate::error::Result;
use crate::schema;
use crate::store::Namespace;

/// An interned term identifier. `0` is reserved as "none"; ids start at `1`.
pub(crate) type TermId = u32;

/// The stored dictionary key for a gram. P1: the gram's UTF-8 bytes. P2: the 16-byte
/// big-endian `u128` term encoding. Confined to [`gram_key`] so the swap is local.
pub(crate) type GramKey = Vec<u8>;

/// Encode a gram into its dictionary key — the only place gram text becomes the key.
#[inline]
pub(crate) fn gram_key(gram: &str) -> GramKey {
    gram.as_bytes().to_vec()
}

/// The faulting dictionary: a `gram-key → term-id` map plus the next id to mint and the
/// generation it was loaded at. Shared (behind an `RwLock`) by the single writer and
/// the read pool, mirroring the store's writer-Mutex + read-pool split.
pub(crate) struct Dictionary {
    inner: RwLock<DictInner>,
}

struct DictInner {
    map: HashMap<GramKey, TermId>,
    /// The next id to assign (high-water mark), as `u64` so it can represent the
    /// exhausted sentinel `u32::MAX + 1` without wrapping.
    next_id: u64,
    /// The generation this in-memory state reflects (== `meta.dict_generation` of the
    /// loaded snapshot). Bumped only by rebuild/reset, never by an append.
    generation: u64,
}

impl Dictionary {
    /// An empty dictionary (before [`load`](Self::load)).
    pub(crate) fn empty() -> Self {
        Dictionary {
            inner: RwLock::new(DictInner {
                map: HashMap::new(),
                next_id: 1,
                generation: 0,
            }),
        }
    }

    /// Hydrate the whole dictionary from the `dict` table (a full scan — the vocabulary
    /// is Heaps-bounded). Replaces the in-memory state wholesale; call on open and after
    /// a rebuild swap (under the write lease, so no reader observes a torn map).
    pub(crate) fn load(&self, conn: &Connection, ns: &Namespace) -> Result<()> {
        let sql = format!("SELECT id, gram FROM {}", ns.dict());
        let mut map: HashMap<GramKey, TermId> = HashMap::new();
        let mut max_id: u32 = 0;
        {
            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let id_i64: i64 = r.get(0)?;
                let id = u32::try_from(id_i64)
                    .ok()
                    .filter(|&i| i != 0)
                    .ok_or_else(|| crate::Error::corrupt("dict id out of u32 range"))?;
                let gram: Vec<u8> = r.get(1)?;
                map.insert(gram, id);
                max_id = max_id.max(id);
            }
        }
        let generation = schema::dict_generation(conn, ns)?;
        let mut guard = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        guard.map = map;
        guard.next_id = max_id as u64 + 1;
        guard.generation = generation;
        Ok(())
    }

    /// Reader fault: resolve a gram to its id, or `None` if absent. Never mutates.
    pub(crate) fn resolve(&self, gram: &str) -> Option<TermId> {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .map
            .get(&gram_key(gram))
            .copied()
    }

    /// Resolve a batch of distinct grams to ids under one read-lock, capturing the
    /// in-memory generation atomically with the resolution. The returned map keys are
    /// the input gram strings; absent grams are omitted. The generation lets the caller
    /// detect a concurrent rebuild (compare against the SQLite snapshot's stored value).
    pub(crate) fn resolve_batch(&self, grams: &[&str]) -> (HashMap<String, TermId>, u64) {
        let guard = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        let mut out = HashMap::with_capacity(grams.len());
        for &g in grams {
            if let Some(&id) = guard.map.get(&gram_key(g)) {
                out.insert(g.to_string(), id);
            }
        }
        (out, guard.generation)
    }

    /// Begin a write-scoped interning session. Snapshots the committed high-water mark;
    /// allocations are buffered locally and only merged into the shared map on
    /// [`InternStage::commit`] (after the write transaction commits).
    ///
    /// Only ever called on a write path (which holds the single-writer lease), so the
    /// snapshotted `next_id` cannot move under the stage.
    pub(crate) fn stage(&self) -> InternStage<'_> {
        let guard = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        InternStage {
            dict: self,
            base_next_id: guard.next_id,
            new: Vec::new(),
            new_index: HashMap::new(),
        }
    }
}

/// The mutation-bearing handle — the §4 writer side of the fault split. Reachable only
/// via [`Dictionary::stage`] (a write path), so a shared `&Dictionary` reader cannot
/// allocate ids.
pub(crate) struct InternStage<'a> {
    dict: &'a Dictionary,
    /// The committed high-water mark at stage time; ids assigned this txn run from here.
    base_next_id: u64,
    /// Grams allocated this transaction, in assignment order (for the post-commit merge).
    new: Vec<(GramKey, TermId)>,
    /// Lookup for grams allocated this transaction.
    new_index: HashMap<GramKey, TermId>,
}

impl InternStage<'_> {
    /// Writer fault: resolve `gram`, allocating + persisting a new id on a miss. Lookup
    /// order: committed map → this-txn allocations → allocate. The `dict` row is written
    /// inside the caller's transaction; the shared in-memory map is *not* touched here.
    pub(crate) fn intern(&mut self, gram: &str, conn: &Connection, ns: &Namespace) -> Result<TermId> {
        if let Some(id) = self.dict.resolve(gram) {
            return Ok(id);
        }
        let key = gram_key(gram);
        if let Some(&id) = self.new_index.get(&key) {
            return Ok(id);
        }
        let candidate = self.base_next_id + self.new.len() as u64;
        if candidate == 0 || candidate > u32::MAX as u64 {
            return Err(crate::Error::corrupt("term id space exhausted"));
        }
        let id = candidate as TermId;
        conn.prepare_cached(&format!("INSERT INTO {}(id, gram) VALUES(?1, ?2)", ns.dict()))?
            .execute(rusqlite::params![id as i64, &key])?;
        self.new_index.insert(key.clone(), id);
        self.new.push((key, id));
        Ok(id)
    }

    /// Merge this transaction's allocations into the shared map and advance the
    /// high-water mark. **Call only after the write transaction has committed** — a
    /// rolled-back transaction must drop the stage without calling this, leaving the
    /// shared map untouched (no orphan ids). A no-op if nothing was interned.
    pub(crate) fn commit(self) {
        if self.new.is_empty() {
            return;
        }
        let mut guard = self.dict.inner.write().unwrap_or_else(PoisonError::into_inner);
        for (key, id) in self.new {
            guard.map.entry(key).or_insert(id);
            guard.next_id = guard.next_id.max(id as u64 + 1);
        }
    }
}
