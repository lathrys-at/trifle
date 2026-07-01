//! Term interning: the in-memory faulting dictionary mapping a gram to a permanent
//! `u32` term-id, plus the per-script-class document-frequency statistics that traffic
//! with it.
//!
//! Postings (`post`/`delta`/`term`-df/`fwd`) are keyed by the `u32` id, not the gram —
//! narrow integer B-trees, fast point seeks, a smaller index. The gram's encoding (its
//! [`Term`](crate::Term) `u128`, stored 16-byte big-endian) lives only in the
//! `dict` table. Ids are **monotonic and permanent**: a gram keeps its id until a
//! [`rebuild`](crate::Index::rebuild) reassigns the whole space (bumping the generation).
//!
//! **Reader/writer fault split (a correctness contract the types enforce):**
//! - A reader calls [`Dictionary::resolve`] — a read that returns `None` on a miss (the
//!   gram is absent, an empty posting). It never mutates, allocates, or writes.
//! - A writer takes an [`InternStage`] (only reachable on a write path, which holds the
//!   single-writer lease) and calls [`InternStage::intern_term`], which allocates an id and
//!   persists a `dict` row inside the write transaction. Staged grams enter the shared
//!   in-memory map only via [`InternStage::commit`], called **after** the transaction
//!   commits — so a rolled-back write leaves no orphan id.
//!
//! The dictionary also owns the per-class [`ClassStats`] (recomputed from the df column
//! on open/rebuild, maintained incrementally via [`Dictionary::apply_df_changes`]) and
//! the id→class table both need.

use crate::hash::FxHashMap;
use std::sync::{PoisonError, RwLock};

use rusqlite::Connection;

use crate::error::Result;
use crate::schema;
use crate::store::Namespace;
use crate::term::Term;
use crate::welford::{ClassSnap, ClassStats};

/// An interned term identifier. `0` is reserved as "none"; ids start at `1`.
pub(crate) type TermId = u32;

/// Decode a stored 16-byte `dict.gram` BLOB back to its `u128` key.
fn decode_key(bytes: &[u8]) -> Result<u128> {
    let arr: [u8; 16] = bytes
        .try_into()
        .map_err(|_| crate::Error::corrupt("dict gram blob is not 16 bytes"))?;
    Ok(u128::from_be_bytes(arr))
}

/// The faulting dictionary plus per-class stats, shared (behind an `RwLock`) by the
/// single writer and the read pool.
pub(crate) struct Dictionary {
    inner: RwLock<DictInner>,
}

struct DictInner {
    map: FxHashMap<u128, TermId>,
    /// id → `(script-class byte, order)`, for maintaining [`ClassStats`] by term-id on writes.
    /// v0.4/M5 keys the selection class on `(script, order)` (derivation §5/§8), and a delta
    /// applied in [`apply_df_changes`](Dictionary::apply_df_changes) has only the id, so both
    /// the class byte and the order are stashed here at intern/load time.
    class_of: FxHashMap<TermId, (u8, u8)>,
    classes: ClassStats,
    /// The next id to assign (high-water mark), as `u64` so it can represent the
    /// exhausted sentinel `u32::MAX + 1` without wrapping.
    next_id: u64,
    /// The generation this in-memory state reflects (== `meta.dict_generation`). Bumped
    /// only by rebuild/reset, never by an append.
    generation: u64,
}

impl Dictionary {
    /// An empty dictionary (before [`load`](Self::load)).
    pub(crate) fn empty() -> Self {
        Dictionary {
            inner: RwLock::new(DictInner {
                map: FxHashMap::default(),
                class_of: FxHashMap::default(),
                classes: ClassStats::new(),
                next_id: 1,
                generation: 0,
            }),
        }
    }

    /// Hydrate the whole dictionary from the `dict` table joined to the live df column —
    /// one pass rebuilds the gram→id map, the id→class table, and the per-class Welford
    /// accumulators (recompute, never persist → the df column is the single source of
    /// truth). Replaces the in-memory state wholesale; call on open and after a rebuild
    /// swap (under the write lease, so no reader observes a torn map).
    pub(crate) fn load(&self, conn: &Connection, ns: &Namespace) -> Result<()> {
        let sql = format!(
            "SELECT d.id, d.gram, COALESCE(t.df, 0) FROM {dict} d \
             LEFT JOIN {term} t ON t.id = d.id",
            dict = ns.dict(),
            term = ns.term(),
        );
        let mut map: FxHashMap<u128, TermId> = FxHashMap::default();
        let mut class_of: FxHashMap<TermId, (u8, u8)> = FxHashMap::default();
        let mut classes = ClassStats::new();
        let mut max_id: u32 = 0;
        {
            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query([])?;
            while let Some(r) = rows.next()? {
                let id = u32::try_from(r.get::<_, i64>(0)?)
                    .ok()
                    .filter(|&i| i != 0)
                    .ok_or_else(|| crate::Error::corrupt("dict id out of u32 range"))?;
                let gram: Vec<u8> = r.get(1)?;
                let key = decode_key(&gram)?;
                let df: i64 = r.get(2)?;
                let class = (key >> 120) as u8;
                let order = Term(key).order();
                map.insert(key, id);
                class_of.insert(id, (class, order));
                classes.add_sample(class, order, df);
                max_id = max_id.max(id);
            }
        }
        let generation = schema::dict_generation(conn, ns)?;
        let mut guard = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        guard.map = map;
        guard.class_of = class_of;
        guard.classes = classes;
        guard.next_id = max_id as u64 + 1;
        guard.generation = generation;
        Ok(())
    }

    /// Reader fault: resolve a gram-encoding key to its id, or `None` if absent. Never
    /// mutates. (Batched query resolution goes through [`resolve_terms`](Self::resolve_terms).)
    fn resolve_key(&self, key: u128) -> Option<TermId> {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .map
            .get(&key)
            .copied()
    }

    /// Resolve a batch of distinct terms to ids and snapshot the given `(script, order)` classes,
    /// all under one read-lock, capturing the generation atomically with the resolution. Returns
    /// `(term-key→id, generation, class snapshot)`; a term absent from the corpus is omitted
    /// from the map (and resolves to df 0 downstream).
    ///
    /// `classes` is caller-supplied (rather than derived from `terms`) so the search path can
    /// request a **superset** — every query's classes *plus their sibling orders* — and then carve
    /// per-query [`ClassSnap::subset`]s out of the one snapshot; each entry is a global corpus
    /// statistic, so a superset never perturbs a per-query value (`batch == serial`).
    ///
    /// Keyed by the term's packed `u128` so the read path resolves straight from a tokenizer
    /// token's [`term()`](crate::IntoTerm::term) — no `Token → String → re-encode` round-trip,
    /// matching what the write path already does.
    pub(crate) fn resolve_terms(
        &self,
        terms: &[Term],
        classes: impl IntoIterator<Item = (u8, u8)>,
    ) -> (FxHashMap<u128, TermId>, u64, ClassSnap) {
        let guard = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        let mut out = FxHashMap::with_capacity_and_hasher(terms.len(), Default::default());
        for t in terms {
            if let Some(&id) = guard.map.get(&t.0) {
                out.insert(t.0, id);
            }
        }
        let snap = guard.classes.snapshot_for(classes);
        (out, guard.generation, snap)
    }

    /// Begin a write-scoped interning session. Snapshots the committed high-water mark;
    /// allocations are buffered locally and merged into the shared map only on
    /// [`InternStage::commit`]. Only ever called on a write path (which holds the
    /// single-writer lease), so the snapshotted `next_id` cannot move under the stage.
    ///
    /// The reader/writer fault split is enforced by reachability rather than the type: `stage`
    /// takes `&self` (the dictionary lives behind a shared `&Index`), but `stage`/`InternStage`
    /// are `pub(crate)`, only [`Writer::begin`](crate::Writer) calls `stage` (under the
    /// single-writer lease), and the read pool hands out `SQLITE_OPEN_READ_ONLY` connections, so
    /// an intern's `INSERT` would fail at runtime regardless. No reader path can intern.
    pub(crate) fn stage(&self) -> InternStage<'_> {
        let guard = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        InternStage {
            dict: self,
            base_next_id: guard.next_id,
            new: Vec::new(),
            new_index: FxHashMap::default(),
        }
    }

    /// Apply the per-term `(id, old_df, new_df)` changes from a committed write to the
    /// class accumulators. Call after the write transaction commits **and** after
    /// [`InternStage::commit`], so the id→class table already covers newly-interned terms.
    pub(crate) fn apply_df_changes(&self, changes: &[(TermId, i64, i64)]) {
        if changes.is_empty() {
            return;
        }
        let mut guard = self.inner.write().unwrap_or_else(PoisonError::into_inner);
        for &(id, old_df, new_df) in changes {
            if let Some((class, order)) = guard.class_of.get(&id).copied() {
                guard.classes.update(class, order, old_df, new_df);
            }
        }
    }
}

/// The mutation-bearing handle — the writer side of the fault split. Reachable only
/// via [`Dictionary::stage`] (a write path), so a shared `&Dictionary` reader cannot
/// allocate ids.
pub(crate) struct InternStage<'a> {
    dict: &'a Dictionary,
    /// The committed high-water mark at stage time; ids assigned this txn run from here.
    base_next_id: u64,
    /// Grams allocated this transaction (key, id), in assignment order.
    new: Vec<(u128, TermId)>,
    new_index: FxHashMap<u128, TermId>,
}

impl InternStage<'_> {
    /// Writer fault from an already-packed [`Term`] — the hot-path entry point. The write
    /// path interns straight from a tokenizer token (`token.term()`), skipping a
    /// `Token → String → re-encode` round-trip: resolve the term, allocating + persisting a
    /// new id on a miss. Lookup order: committed map → this-txn allocations → allocate. The
    /// `dict` row is written inside the caller's transaction; the shared in-memory map is
    /// *not* touched here.
    ///
    /// # Errors
    ///
    /// [`Error::Corrupt`](crate::Error::Corrupt) if the id space is exhausted.
    pub(crate) fn intern_term(
        &mut self,
        term: Term,
        conn: &Connection,
        ns: &Namespace,
    ) -> Result<TermId> {
        self.intern_key(term.0, conn, ns)
    }

    /// A rollback marker — the count of grams staged so far. Pair with
    /// [`rollback_to`](Self::rollback_to) to undo the staging of one failed write call,
    /// matching a SQL `SAVEPOINT`/`ROLLBACK TO` that undoes its `dict` rows.
    pub(crate) fn mark(&self) -> usize {
        self.new.len()
    }

    /// Discard grams staged after `mark` (a failed write call's allocations), so a later
    /// [`commit`](Self::commit) never merges an id whose `dict` row was rolled back. The
    /// freed id range is reused by the next allocation.
    pub(crate) fn rollback_to(&mut self, mark: usize) {
        for (key, _) in self.new.drain(mark..) {
            self.new_index.remove(&key);
        }
    }

    /// The shared body of [`intern_term`](Self::intern_term): resolve `key`, allocating +
    /// persisting a new id on a miss.
    fn intern_key(&mut self, key: u128, conn: &Connection, ns: &Namespace) -> Result<TermId> {
        if let Some(id) = self.dict.resolve_key(key) {
            return Ok(id);
        }
        if let Some(&id) = self.new_index.get(&key) {
            return Ok(id);
        }
        let candidate = self.base_next_id + self.new.len() as u64;
        if candidate == 0 || candidate > u32::MAX as u64 {
            return Err(crate::Error::corrupt("term id space exhausted"));
        }
        let id = candidate as TermId;
        conn.prepare_cached(&format!(
            "INSERT INTO {}(id, gram) VALUES(?1, ?2)",
            ns.dict()
        ))?
        .execute(rusqlite::params![id as i64, key.to_be_bytes().as_slice()])?;
        self.new_index.insert(key, id);
        self.new.push((key, id));
        Ok(id)
    }

    /// Merge this transaction's allocations into the shared map (and id→class table) and
    /// advance the high-water mark. **Call only after the write transaction has
    /// committed** — a rolled-back transaction must drop the stage without calling this,
    /// leaving the shared state untouched (no orphan ids). A no-op if nothing was
    /// interned. The class *stats* are updated separately by
    /// [`Dictionary::apply_df_changes`] (which needs the new df values).
    pub(crate) fn commit(self) {
        if self.new.is_empty() {
            return;
        }
        let mut guard = self
            .dict
            .inner
            .write()
            .unwrap_or_else(PoisonError::into_inner);
        for (key, id) in self.new {
            guard.map.entry(key).or_insert(id);
            guard
                .class_of
                .entry(id)
                .or_insert(((key >> 120) as u8, Term(key).order()));
            guard.next_id = guard.next_id.max(id as u64 + 1);
        }
    }
}
