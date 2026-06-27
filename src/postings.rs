//! The owned roaring inverted index: one base+delta posting per token, plus the
//! live document-frequency column.
//!
//! Every token resolves to `(base ∪ added) \ removed`, always fresh — the delta is
//! written in the same transaction as the segment, so a read needs no freshness
//! gate and no decode step (an owned roaring posting *is* the bitmap). A write
//! touches only the small `term.df` and `delta` rows; the big `post.base` is
//! rewritten only by [`fold`] or a rebuild.

use crate::hash::FxHashMap;
use std::rc::Rc;

use roaring::RoaringBitmap;
use rusqlite::Connection;
use rusqlite::types::Value;

use crate::dict::TermId;
use crate::error::{Error, Result};
use crate::store::Namespace;

/// Serialize a roaring bitmap to its on-disk blob form.
pub(crate) fn serialize(bm: &RoaringBitmap) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(bm.serialized_size());
    bm.serialize_into(&mut buf).map_err(Error::Posting)?;
    Ok(buf)
}

/// Deserialize a roaring bitmap from its on-disk blob form.
pub(crate) fn deserialize(bytes: &[u8]) -> Result<RoaringBitmap> {
    RoaringBitmap::deserialize_from(bytes).map_err(Error::Posting)
}

/// An `Rc`'d carray of term-ids for an `IN rarray(?1)` bind — one prepared statement
/// for any id count, with no temp btree.
fn id_array(ids: impl IntoIterator<Item = TermId>) -> Rc<Vec<Value>> {
    Rc::new(
        ids.into_iter()
            .map(|id| Value::Integer(id as i64))
            .collect(),
    )
}

/// Read the effective document frequency of each term-id in one batched query. An id
/// with no row is absent from the map (document frequency 0 — a genuinely-absent
/// token, since the column is maintained live).
pub(crate) fn read_dfs(
    conn: &Connection,
    ns: &Namespace,
    ids: &[TermId],
) -> Result<FxHashMap<TermId, i64>> {
    let mut out = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
    if ids.is_empty() {
        return Ok(out);
    }
    let arr = id_array(ids.iter().copied());
    let sql = format!("SELECT id, df FROM {} WHERE id IN rarray(?1)", ns.term());
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut rows = stmt.query(rusqlite::params![arr])?;
    while let Some(r) = rows.next()? {
        out.insert(r.get::<_, i64>(0)? as TermId, r.get::<_, i64>(1)?);
    }
    Ok(out)
}

/// Each requested term-id's effective posting `(base ∪ added) \ removed`, batched into
/// two reads (one over `post`, one over `delta`). A term whose effective posting is
/// empty (or that has no row at all) is omitted from the map.
pub(crate) fn effective_postings(
    conn: &Connection,
    ns: &Namespace,
    ids: &[TermId],
) -> Result<FxHashMap<TermId, RoaringBitmap>> {
    let mut out: FxHashMap<TermId, RoaringBitmap> =
        FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
    if ids.is_empty() {
        return Ok(out);
    }
    let arr = id_array(ids.iter().copied());

    let post_sql = format!("SELECT id, base FROM {} WHERE id IN rarray(?1)", ns.post());
    let mut post_stmt = conn.prepare_cached(&post_sql)?;
    let mut rows = post_stmt.query(rusqlite::params![arr])?;
    while let Some(r) = rows.next()? {
        let id = r.get::<_, i64>(0)? as TermId;
        let blob: Vec<u8> = r.get(1)?;
        out.insert(id, deserialize(&blob)?);
    }
    drop(rows);
    drop(post_stmt);

    let delta_sql = format!(
        "SELECT id, added, removed FROM {} WHERE id IN rarray(?1)",
        ns.delta()
    );
    let mut delta_stmt = conn.prepare_cached(&delta_sql)?;
    let mut rows = delta_stmt.query(rusqlite::params![arr])?;
    while let Some(r) = rows.next()? {
        let id = r.get::<_, i64>(0)? as TermId;
        let added: Vec<u8> = r.get(1)?;
        let removed: Vec<u8> = r.get(2)?;
        let entry = out.entry(id).or_default();
        *entry |= &deserialize(&added)?;
        *entry -= &deserialize(&removed)?;
    }

    out.retain(|_, bm| !bm.is_empty());
    Ok(out)
}

/// One token's pending change in a write: segment ids gained (`add`) and lost
/// (`remove`) under one interned term-id.
///
/// **Contract (monotonic ids):** `add` are freshly-allocated segment ids, absent from
/// every posting; `remove` are live segment ids currently present in this token's
/// posting; the two sets are disjoint. Under this contract the document-frequency
/// change is exactly `add.len() - remove.len()`, which is why [`apply_writes`] need not
/// load the base to maintain `df`. Violating it silently drifts `df` from the true
/// effective cardinality (debug builds assert disjointness).
pub(crate) struct TermWrite<'a> {
    pub id: TermId,
    pub add: &'a [u32],
    pub remove: &'a [u32],
}

/// Apply one write's per-token changes to the deltas and the live document
/// frequencies, in the caller's transaction. `O(touched tokens)`; never touches a
/// base posting. Each `writes` entry must name a distinct term-id and honor the
/// [`TermWrite`] monotonic-id contract.
///
/// Returns the `(term-id, old_df, new_df)` of every touched term, so the caller can
/// maintain the per-class document-frequency statistics (the live df is the single
/// source of truth those accumulators derive from).
pub(crate) fn apply_writes(
    conn: &Connection,
    ns: &Namespace,
    writes: &[TermWrite],
) -> Result<Vec<(TermId, i64, i64)>> {
    if writes.is_empty() {
        return Ok(Vec::new());
    }
    // Batch-load the existing deltas for every touched term-id.
    let arr = id_array(writes.iter().map(|w| w.id));
    let mut deltas: FxHashMap<TermId, (RoaringBitmap, RoaringBitmap)> =
        FxHashMap::with_capacity_and_hasher(writes.len(), Default::default());
    let load_sql = format!(
        "SELECT id, added, removed FROM {} WHERE id IN rarray(?1)",
        ns.delta()
    );
    {
        let mut stmt = conn.prepare_cached(&load_sql)?;
        let mut rows = stmt.query(rusqlite::params![arr])?;
        while let Some(r) = rows.next()? {
            let id = r.get::<_, i64>(0)? as TermId;
            let added: Vec<u8> = r.get(1)?;
            let removed: Vec<u8> = r.get(2)?;
            deltas.insert(id, (deserialize(&added)?, deserialize(&removed)?));
        }
    }
    // Old df per touched term-id (the value the class stats move *from*).
    let touched: Vec<TermId> = writes.iter().map(|w| w.id).collect();
    let old_dfs = read_dfs(conn, ns, &touched)?;
    let mut changes: Vec<(TermId, i64, i64)> = Vec::with_capacity(writes.len());

    let upsert_sql = format!(
        "INSERT INTO {0}(id, added, removed) VALUES(?1, ?2, ?3)
         ON CONFLICT(id) DO UPDATE SET added = excluded.added, removed = excluded.removed",
        ns.delta()
    );
    let delete_sql = format!("DELETE FROM {} WHERE id = ?1", ns.delta());
    // `?2` is referenced twice (positional reuse) — both the INSERT value and the
    // increment on conflict, so a brand-new term lands at `df = ?2` and an existing
    // one moves by `?2`.
    let df_sql = format!(
        "INSERT INTO {0}(id, df) VALUES(?1, ?2)
         ON CONFLICT(id) DO UPDATE SET df = df + ?2",
        ns.term()
    );

    let mut upsert = conn.prepare_cached(&upsert_sql)?;
    let mut delete = conn.prepare_cached(&delete_sql)?;
    let mut df = conn.prepare_cached(&df_sql)?;

    for w in writes {
        // The `df_delta = |add| - |remove|` shortcut below is exact only under the
        // monotonic-id contract (see `TermWrite`): `add` are fresh ids absent from
        // every posting, `remove` are live ids present in this token's posting, and
        // the two are disjoint. Defend the disjointness cheaply in debug builds; a
        // violation would silently drift `df` from the effective cardinality.
        debug_assert!(
            w.add.iter().all(|a| !w.remove.contains(a)),
            "TermWrite add/remove must be disjoint (monotonic-id contract)"
        );
        let id_param = w.id as i64;
        let (mut added, mut removed) = deltas.remove(&w.id).unwrap_or_default();
        for &id in w.remove {
            removed.insert(id);
            added.remove(id);
        }
        for &id in w.add {
            added.insert(id);
            removed.remove(id);
        }
        let df_delta = w.add.len() as i64 - w.remove.len() as i64;
        let old_df = old_dfs.get(&w.id).copied().unwrap_or(0);
        changes.push((w.id, old_df, old_df + df_delta));
        if df_delta != 0 {
            df.execute(rusqlite::params![id_param, df_delta])?;
        }
        if added.is_empty() && removed.is_empty() {
            delete.execute(rusqlite::params![id_param])?;
        } else {
            upsert.execute(rusqlite::params![
                id_param,
                serialize(&added)?,
                serialize(&removed)?
            ])?;
        }
    }
    Ok(changes)
}

/// What a [`fold`] reclaimed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FoldStats {
    /// Tokens whose pending delta was merged into the base.
    pub tokens_folded: u64,
    /// Stale ids removed from base postings by the fold.
    pub ids_purged: u64,
    /// Tokens dropped entirely (effective posting emptied, or document frequency
    /// fell to zero).
    pub terms_dropped: u64,
    /// Delta-blob bytes the fold cleared (the backlog it absorbed).
    pub bytes_reclaimed: u64,
}

/// Fold every pending delta into its base, drop tokens whose effective posting
/// emptied, and prune zero-frequency term rows, in the caller's transaction.
///
/// Cheap for rare tokens (small bases), genuinely costly for common ones (a
/// high-frequency token's base is a large bitset the fold rewrites) — the price of
/// owning all tokens, paid here off the hot path rather than on every write.
pub(crate) fn fold(conn: &Connection, ns: &Namespace) -> Result<FoldStats> {
    // Only tokens with a delta row can have a pending change; that set is the dirty
    // candidate list and is emptied as we go.
    let dirty: Vec<TermId> = {
        let sql = format!("SELECT id FROM {}", ns.delta());
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |r| r.get::<_, i64>(0))?;
        rows.map(|r| r.map(|v| v as TermId))
            .collect::<rusqlite::Result<Vec<_>>>()?
    };

    let base_sel = format!("SELECT base FROM {} WHERE id = ?1", ns.post());
    let delta_sel = format!("SELECT added, removed FROM {} WHERE id = ?1", ns.delta());
    let base_upsert = format!(
        "INSERT INTO {0}(id, base) VALUES(?1, ?2)
         ON CONFLICT(id) DO UPDATE SET base = excluded.base",
        ns.post()
    );
    let base_del = format!("DELETE FROM {} WHERE id = ?1", ns.post());
    let delta_del = format!("DELETE FROM {} WHERE id = ?1", ns.delta());

    let mut base_sel = conn.prepare_cached(&base_sel)?;
    let mut delta_sel = conn.prepare_cached(&delta_sel)?;
    let mut base_upsert = conn.prepare_cached(&base_upsert)?;
    let mut base_del = conn.prepare_cached(&base_del)?;
    let mut delta_del = conn.prepare_cached(&delta_del)?;

    let mut stats = FoldStats::default();
    for id in &dirty {
        let id_param = *id as i64;
        let base_blob: Option<Vec<u8>> = base_sel
            .query_row([id_param], |r| r.get(0))
            .map(Some)
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        let mut bm = match base_blob {
            Some(b) => deserialize(&b)?,
            None => RoaringBitmap::new(),
        };
        let (added_blob, removed_blob): (Vec<u8>, Vec<u8>) =
            delta_sel.query_row([id_param], |r| Ok((r.get(0)?, r.get(1)?)))?;
        stats.bytes_reclaimed += (added_blob.len() + removed_blob.len()) as u64;
        let added = deserialize(&added_blob)?;
        let removed = deserialize(&removed_blob)?;

        bm |= &added;
        let before = bm.len();
        bm -= &removed;
        stats.ids_purged += before - bm.len();

        if bm.is_empty() {
            // Drop the emptied base posting; the token's `df` is already 0 (the write
            // that emptied it decremented it), so the trailing prune below is the
            // single authoritative count of dropped tokens. The `dict` row is *not*
            // dropped — term-ids are permanent until a rebuild reassigns them.
            base_del.execute([id_param])?;
        } else {
            base_upsert.execute(rusqlite::params![id_param, serialize(&bm)?])?;
        }
        delta_del.execute([id_param])?;
        stats.tokens_folded += 1;
    }

    // Prune term rows whose live frequency fell to zero (no live segment carries the
    // token) — `df` equals the effective posting cardinality, so this is exactly the
    // set of tokens that left the index. The sole source of `terms_dropped`, so an
    // emptied token is never double-counted against the base drop above.
    let pruned = conn.execute(&format!("DELETE FROM {} WHERE df <= 0", ns.term()), [])? as u64;
    stats.terms_dropped += pruned;

    Ok(stats)
}

/// Write dense base postings and their document frequencies into the given tables
/// (used by rebuild to populate the shadow tables), keyed by term-id. Empty postings
/// are skipped.
pub(crate) fn write_base_postings<'a>(
    conn: &Connection,
    post_table: &str,
    term_table: &str,
    postings: impl Iterator<Item = (TermId, &'a RoaringBitmap)>,
) -> Result<()> {
    let post_sql = format!("INSERT INTO {post_table}(id, base) VALUES(?1, ?2)");
    let term_sql = format!("INSERT INTO {term_table}(id, df) VALUES(?1, ?2)");
    let mut post_stmt = conn.prepare_cached(&post_sql)?;
    let mut term_stmt = conn.prepare_cached(&term_sql)?;
    for (id, bm) in postings {
        if bm.is_empty() {
            continue;
        }
        post_stmt.execute(rusqlite::params![id as i64, serialize(bm)?])?;
        term_stmt.execute(rusqlite::params![id as i64, bm.len() as i64])?;
    }
    Ok(())
}

/// The number of pending delta rows — the signal for *when* to [`fold`].
pub(crate) fn delta_backlog(conn: &Connection, ns: &Namespace) -> Result<u64> {
    let sql = format!("SELECT count(*) FROM {}", ns.delta());
    Ok(conn.query_row(&sql, [], |r| r.get::<_, i64>(0))? as u64)
}

/// The number of distinct tokens with a live (non-zero) frequency.
pub(crate) fn term_count(conn: &Connection, ns: &Namespace) -> Result<u64> {
    let sql = format!("SELECT count(*) FROM {} WHERE df > 0", ns.term());
    Ok(conn.query_row(&sql, [], |r| r.get::<_, i64>(0))? as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema;
    use crate::store::Namespace;

    /// An in-memory connection with the carray vtab and trifle's tables — the unit
    /// harness for the storage layer (no file, no backend).
    fn harness() -> (Connection, Namespace) {
        let conn = Connection::open_in_memory().unwrap();
        rusqlite::vtab::array::load_module(&conn).unwrap();
        let ns = Namespace::bare();
        schema::create_tables(&conn, &ns, &crate::model::Schema::flat()).unwrap();
        (conn, ns)
    }

    /// Apply a single term-id's add/remove segment ids.
    fn write(conn: &Connection, ns: &Namespace, id: TermId, add: &[u32], remove: &[u32]) {
        apply_writes(conn, ns, &[TermWrite { id, add, remove }]).unwrap();
    }

    fn df(conn: &Connection, ns: &Namespace, id: TermId) -> Option<i64> {
        read_dfs(conn, ns, &[id]).unwrap().get(&id).copied()
    }

    fn posting(conn: &Connection, ns: &Namespace, id: TermId) -> Vec<u32> {
        effective_postings(conn, ns, &[id])
            .unwrap()
            .get(&id)
            .map(|bm| bm.iter().collect())
            .unwrap_or_default()
    }

    #[test]
    fn bitmap_blob_round_trips() {
        let bm: RoaringBitmap = [1u32, 5, 100, 70_000].into_iter().collect();
        let blob = serialize(&bm).unwrap();
        assert_eq!(deserialize(&blob).unwrap(), bm);
    }

    #[test]
    fn add_then_read_posting_and_df() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1, 2, 3], &[]);
        assert_eq!(posting(&conn, &ns, 1), [1, 2, 3]);
        assert_eq!(df(&conn, &ns, 1), Some(3));
    }

    #[test]
    fn remove_excludes_and_decrements_df() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1, 2, 3], &[]);
        write(&conn, &ns, 1, &[], &[2]);
        assert_eq!(posting(&conn, &ns, 1), [1, 3]);
        assert_eq!(df(&conn, &ns, 1), Some(2)); // 3 added, then 1 removed
    }

    #[test]
    fn df_tracks_distinct_ids_per_term() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1, 2], &[]);
        write(&conn, &ns, 1, &[3], &[1]); // +1 -1 net 0
        assert_eq!(df(&conn, &ns, 1), Some(2));
        assert_eq!(posting(&conn, &ns, 1), [2, 3]);
    }

    #[test]
    fn effective_posting_is_base_union_added_minus_removed() {
        let (conn, ns) = harness();
        // Build a base via fold, then layer a delta on top.
        write(&conn, &ns, 1, &[1, 2, 3], &[]);
        fold(&conn, &ns).unwrap(); // base = {1,2,3}, delta cleared
        write(&conn, &ns, 1, &[4], &[2]); // delta: add 4, remove 2
        assert_eq!(posting(&conn, &ns, 1), [1, 3, 4]);
        assert_eq!(df(&conn, &ns, 1), Some(3));
    }

    #[test]
    fn fold_merges_delta_into_base_and_clears_it() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1, 2], &[]);
        assert_eq!(delta_backlog(&conn, &ns).unwrap(), 1);
        let stats = fold(&conn, &ns).unwrap();
        assert_eq!(stats.tokens_folded, 1);
        assert_eq!(delta_backlog(&conn, &ns).unwrap(), 0);
        // Posting still reads correctly from the base alone.
        assert_eq!(posting(&conn, &ns, 1), [1, 2]);
    }

    #[test]
    fn fold_drops_a_token_whose_posting_emptied() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1], &[]);
        fold(&conn, &ns).unwrap();
        write(&conn, &ns, 1, &[], &[1]); // now empty
        let stats = fold(&conn, &ns).unwrap();
        // Exactly one token dropped — not two. The token empties in the fold loop
        // AND its df row is pruned by the trailing sweep; the stat must count it once.
        assert_eq!(stats.terms_dropped, 1);
        assert!(posting(&conn, &ns, 1).is_empty());
        // df row pruned (df <= 0).
        assert_eq!(df(&conn, &ns, 1), None);
        assert_eq!(term_count(&conn, &ns).unwrap(), 0);
    }

    #[test]
    fn fold_is_idempotent_with_no_pending_delta() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1, 2], &[]);
        fold(&conn, &ns).unwrap();
        // A second fold has no delta rows to process and must be a clean no-op.
        let again = fold(&conn, &ns).unwrap();
        assert_eq!(again, FoldStats::default());
        assert_eq!(posting(&conn, &ns, 1), [1, 2]);
    }

    #[test]
    fn readd_of_a_removed_id_before_fold_purges_nothing() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1, 2, 3], &[]);
        fold(&conn, &ns).unwrap(); // base = {1,2,3}
        write(&conn, &ns, 1, &[], &[2]); // stage removal of 2
        write(&conn, &ns, 1, &[2], &[]); // re-add 2 before the fold rescinds it
        let stats = fold(&conn, &ns).unwrap();
        assert_eq!(
            stats.ids_purged, 0,
            "the rescinded removal leaves the base intact"
        );
        assert_eq!(posting(&conn, &ns, 1), [1, 2, 3]);
        assert_eq!(df(&conn, &ns, 1), Some(3));
    }

    #[test]
    fn a_token_resurrects_after_being_fully_pruned() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1], &[]);
        fold(&conn, &ns).unwrap();
        write(&conn, &ns, 1, &[], &[1]);
        fold(&conn, &ns).unwrap(); // fully pruned: no post/term/delta rows
        assert_eq!(df(&conn, &ns, 1), None);
        // A later write on the same token must rebuild it from scratch (no base).
        write(&conn, &ns, 1, &[2], &[]);
        assert_eq!(df(&conn, &ns, 1), Some(1));
        assert_eq!(posting(&conn, &ns, 1), [2]);
        fold(&conn, &ns).unwrap();
        assert_eq!(posting(&conn, &ns, 1), [2]);
    }

    #[test]
    fn replacing_a_terms_id_nets_zero_df_but_updates_the_posting() {
        // The legal monotonic-id replace: an old id leaves and a fresh id arrives in
        // one write. The cardinality is unchanged (df stays), but the posting swaps.
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1], &[]);
        assert_eq!(df(&conn, &ns, 1), Some(1));
        apply_writes(
            &conn,
            &ns,
            &[TermWrite {
                id: 1,
                add: &[2],
                remove: &[1],
            }],
        )
        .unwrap();
        assert_eq!(df(&conn, &ns, 1), Some(1), "one out, one in: df unchanged");
        assert_eq!(posting(&conn, &ns, 1), [2]);
    }

    #[test]
    fn corrupt_base_blob_surfaces_an_error_not_a_panic() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1, 2, 3], &[]);
        fold(&conn, &ns).unwrap(); // base now stored in `post`
        conn.execute(
            &format!("UPDATE {} SET base = ?1 WHERE id = 1", ns.post()),
            [vec![0xFFu8, 0x00, 0x13, 0x37]],
        )
        .unwrap();
        assert!(matches!(
            effective_postings(&conn, &ns, &[1]),
            Err(crate::Error::Posting(_))
        ));
    }

    #[test]
    fn corrupt_delta_blob_surfaces_an_error_not_a_panic() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1], &[]); // delta row exists
        conn.execute(
            &format!("UPDATE {} SET added = ?1 WHERE id = 1", ns.delta()),
            [vec![0xFFu8, 0xFF, 0xFF]],
        )
        .unwrap();
        assert!(matches!(
            effective_postings(&conn, &ns, &[1]),
            Err(crate::Error::Posting(_))
        ));
    }

    #[test]
    fn fold_purges_stale_ids_from_base() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1, 2, 3], &[]);
        fold(&conn, &ns).unwrap();
        write(&conn, &ns, 1, &[], &[2]);
        let stats = fold(&conn, &ns).unwrap();
        assert_eq!(stats.ids_purged, 1);
        assert_eq!(posting(&conn, &ns, 1), [1, 3]);
    }

    #[test]
    fn read_dfs_omits_absent_terms() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1], &[]);
        let dfs = read_dfs(&conn, &ns, &[1, 2]).unwrap();
        assert_eq!(dfs.get(&1), Some(&1));
        assert_eq!(dfs.get(&2), None);
    }

    #[test]
    fn effective_postings_skips_empty_results() {
        let (conn, ns) = harness();
        write(&conn, &ns, 1, &[1], &[]);
        write(&conn, &ns, 1, &[], &[1]); // effective empty (delta-only)
        let map = effective_postings(&conn, &ns, &[1]).unwrap();
        assert!(!map.contains_key(&1), "empty posting omitted");
    }

    #[test]
    fn apply_writes_batches_many_terms_in_one_pass() {
        let (conn, ns) = harness();
        let writes = [
            TermWrite {
                id: 1,
                add: &[1],
                remove: &[],
            },
            TermWrite {
                id: 2,
                add: &[1, 2],
                remove: &[],
            },
            TermWrite {
                id: 3,
                add: &[2],
                remove: &[],
            },
        ];
        apply_writes(&conn, &ns, &writes).unwrap();
        assert_eq!(df(&conn, &ns, 1), Some(1));
        assert_eq!(df(&conn, &ns, 2), Some(2));
        assert_eq!(posting(&conn, &ns, 3), [2]);
    }

    #[test]
    fn write_base_postings_populates_dense_tables() {
        let (conn, ns) = harness();
        let mut map: FxHashMap<TermId, RoaringBitmap> = FxHashMap::default();
        map.insert(1, [1u32, 2].into_iter().collect());
        map.insert(2, [3u32].into_iter().collect());
        write_base_postings(
            &conn,
            ns.post(),
            ns.term(),
            map.iter().map(|(id, b)| (*id, b)),
        )
        .unwrap();
        assert_eq!(posting(&conn, &ns, 1), [1, 2]);
        assert_eq!(df(&conn, &ns, 2), Some(1));
    }
}
