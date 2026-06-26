//! Term interning: the in-memory faulting dictionary mapping a gram to a permanent
//! `u32` term-id, plus the per-script-class document-frequency statistics that traffic
//! with it.
//!
//! Postings (`post`/`delta`/`term`-df/`fwd`) are keyed by the `u32` id, not the gram —
//! narrow integer B-trees, fast point seeks, a smaller index. The gram's encoding (its
//! [`Term`](crate::term::Term) `u128`, stored 16-byte big-endian) lives only in the
//! `dict` table. Ids are **monotonic and permanent**: a gram keeps its id until a
//! [`rebuild`](crate::Index::rebuild) reassigns the whole space (bumping the generation).
//!
//! **Reader/writer fault split (a correctness contract the types enforce):**
//! - A reader calls [`Dictionary::resolve`] — a read that returns `None` on a miss (the
//!   gram is absent, an empty posting). It never mutates, allocates, or writes.
//! - A writer takes an [`InternStage`] (only reachable on a write path, which holds the
//!   single-writer lease) and calls [`InternStage::intern`], which allocates an id and
//!   persists a `dict` row inside the write transaction. Staged grams enter the shared
//!   in-memory map only via [`InternStage::commit`], called **after** the transaction
//!   commits — so a rolled-back write leaves no orphan id.
//!
//! The dictionary also owns the per-class [`ClassStats`] (recomputed from the df column
//! on open/rebuild, maintained incrementally via [`Dictionary::apply_df_changes`]) and
//! the id→class table both need.

use std::collections::HashMap;
use std::sync::{PoisonError, RwLock};

use rusqlite::Connection;

use crate::error::Result;
use crate::schema;
use crate::store::Namespace;
use crate::term::encode_term;
use crate::welford::{ClassSnap, ClassStats};

/// An interned term identifier. `0` is reserved as "none"; ids start at `1`.
pub(crate) type TermId = u32;

/// The packed `u128` dictionary key for a gram, or `None` if it exceeds the
/// 3-codepoint storage ceiling.
#[inline]
fn gram_key(gram: &str) -> Option<u128> {
    encode_term(gram).map(|t| t.0)
}

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
    map: HashMap<u128, TermId>,
    /// id → script-class byte, for maintaining [`ClassStats`] by term-id on writes.
    class_of: HashMap<TermId, u8>,
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
                map: HashMap::new(),
                class_of: HashMap::new(),
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
        let mut map: HashMap<u128, TermId> = HashMap::new();
        let mut class_of: HashMap<TermId, u8> = HashMap::new();
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
                map.insert(key, id);
                class_of.insert(id, class);
                classes.add_sample(class, df);
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
    /// mutates. (Batched query resolution goes through [`resolve_batch`](Self::resolve_batch).)
    fn resolve_key(&self, key: u128) -> Option<TermId> {
        self.inner
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .map
            .get(&key)
            .copied()
    }

    /// Resolve a batch of distinct grams to ids and snapshot the classes they touch, all
    /// under one read-lock, capturing the generation atomically with the resolution.
    /// Returns `(gram→id, generation, class snapshot)`; absent / over-ceiling grams are
    /// omitted from the map (and resolve to df 0 downstream).
    pub(crate) fn resolve_batch(
        &self,
        grams: &[&str],
    ) -> (HashMap<String, TermId>, u64, ClassSnap) {
        let guard = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        let mut out = HashMap::with_capacity(grams.len());
        let mut classes_seen: Vec<u8> = Vec::new();
        for &g in grams {
            if let Some(t) = encode_term(g) {
                classes_seen.push(t.class());
                if let Some(&id) = guard.map.get(&t.0) {
                    out.insert(g.to_string(), id);
                }
            }
        }
        let snap = guard.classes.snapshot_for(classes_seen);
        (out, guard.generation, snap)
    }

    /// Begin a write-scoped interning session. Snapshots the committed high-water mark;
    /// allocations are buffered locally and merged into the shared map only on
    /// [`InternStage::commit`]. Only ever called on a write path (which holds the
    /// single-writer lease), so the snapshotted `next_id` cannot move under the stage.
    pub(crate) fn stage(&self) -> InternStage<'_> {
        let guard = self.inner.read().unwrap_or_else(PoisonError::into_inner);
        InternStage {
            dict: self,
            base_next_id: guard.next_id,
            new: Vec::new(),
            new_index: HashMap::new(),
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
            if let Some(class) = guard.class_of.get(&id).copied() {
                guard.classes.update(class, old_df, new_df);
            }
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
    /// Grams allocated this transaction (key, id), in assignment order.
    new: Vec<(u128, TermId)>,
    new_index: HashMap<u128, TermId>,
}

impl InternStage<'_> {
    /// Writer fault: resolve `gram`, allocating + persisting a new id on a miss. Lookup
    /// order: committed map → this-txn allocations → allocate. The `dict` row is written
    /// inside the caller's transaction; the shared in-memory map is *not* touched here.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if the gram exceeds the
    /// 3-codepoint storage ceiling (e.g. a quad-gram tokenizer — incompatible with
    /// interning); [`Error::Corrupt`](crate::Error::Corrupt) if the id space is exhausted.
    pub(crate) fn intern(
        &mut self,
        gram: &str,
        conn: &Connection,
        ns: &Namespace,
    ) -> Result<TermId> {
        let key = gram_key(gram).ok_or_else(|| {
            crate::Error::InvalidInput(format!(
                "gram {gram:?} exceeds the 3-codepoint term-encoding ceiling"
            ))
        })?;
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
            guard.class_of.entry(id).or_insert((key >> 120) as u8);
            guard.next_id = guard.next_id.max(id as u64 + 1);
        }
    }
}
