//! On-disk schema: the namespaced DDL, the drift/version stamps, and monotonic id
//! allocation.
//!
//! All names come from a validated [`Namespace`], so interpolating them into DDL
//! has no injection surface. The store is a rebuildable cache: a version mismatch
//! drops everything rather than migrating.
//!
//! There is no `doc` table: the caller key lives directly on `seg` (one row per
//! `(key, label)` segment; `seg.id` is the posting id), so a key with no segments cannot
//! materialize a ghost row and provenance is a single-table point read.

use rusqlite::{Connection, OptionalExtension};

use crate::error::Result;
use crate::model::Schema;
use crate::store::Namespace;

/// trifle's on-disk format version. Bump on any incompatible schema change; an index stamped
/// with a different value is reset on open (the cache is rebuilt, never migrated).
pub(crate) const SCHEMA_VERSION: u32 = 5;

pub(crate) const KEY_SCHEMA_VERSION: &str = "schema_version";
pub(crate) const KEY_DATA_VERSION: &str = "data_version";
pub(crate) const KEY_FINGERPRINT: &str = "tokenizer_fingerprint";
pub(crate) const KEY_SCHEMA_FINGERPRINT: &str = "schema_fingerprint";
pub(crate) const KEY_NEXT_ID: &str = "next_id";
/// Rolling segment count (`N`: reported by `stats()`, the corpus size a custom score may
/// use), maintained in the write transaction so a search reads it as an O(1) point lookup
/// rather than an O(N) `count(*)`.
pub(crate) const KEY_SEG_COUNT: &str = "seg_count";
/// Rolling sum of per-segment **distinct** gram counts (`L_d`, derivation §0/§6). `avgdl =
/// seg_len_sum / seg_count` is then the mean distinct-gram count `L̄`.
pub(crate) const KEY_SEG_LEN_SUM: &str = "seg_len_sum";
/// The dictionary generation (id-assignment epoch): bumped on every reassignment of
/// term-ids (rebuild + reset). The read path compares the snapshot's value to the
/// in-memory dictionary's loaded generation to detect a concurrent reassignment.
pub(crate) const KEY_DICT_GENERATION: &str = "dict_generation";

/// Create every persistent table and index if absent. Idempotent.
pub(crate) fn create_tables(conn: &Connection, ns: &Namespace, schema: &Schema) -> Result<()> {
    let key_sql_type = schema.key_shape().sql_type();
    // The flattened model: `seg` is the only document table — one row per `(key, label)`
    // segment, `seg.id` is the roaring posting id, `seg.key` is the caller key (the one
    // schema-typed column), `seg.txt` the stored text, `seg.len` its distinct gram count (`L_d`).
    // `fwd` holds every segment's term-id set (a roaring bitmap), so delete needs neither
    // the text nor the tokenizer. Postings are keyed by the interned term-id from `dict`;
    // the three-way write-frequency split — `term`/`delta` on every write, `post` only on
    // fold/rebuild — keeps a write from rewriting the big base posting.
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {meta}(key TEXT PRIMARY KEY, value);
         CREATE TABLE IF NOT EXISTS {seg}(
            id    INTEGER PRIMARY KEY,
            key   {keyty} NOT NULL,
            label TEXT    NOT NULL,
            txt   TEXT    NOT NULL,
            len   INTEGER NOT NULL DEFAULT 0
         );
         CREATE INDEX IF NOT EXISTS {seg}_by_key ON {seg}(key);
         CREATE UNIQUE INDEX IF NOT EXISTS {seg}_by_key_label ON {seg}(key, label);
         CREATE TABLE IF NOT EXISTS {fwd}(id INTEGER PRIMARY KEY, tokens BLOB NOT NULL);
         CREATE TABLE IF NOT EXISTS {dict}(id INTEGER PRIMARY KEY, gram BLOB NOT NULL);
         CREATE UNIQUE INDEX IF NOT EXISTS {dict}_by_gram ON {dict}(gram);
         CREATE TABLE IF NOT EXISTS {term}(id INTEGER PRIMARY KEY, df INTEGER NOT NULL);
         CREATE TABLE IF NOT EXISTS {post}(id INTEGER PRIMARY KEY, base BLOB NOT NULL);
         CREATE TABLE IF NOT EXISTS {delta}(
            id INTEGER PRIMARY KEY, added BLOB NOT NULL, removed BLOB NOT NULL
         );",
        meta = ns.meta(),
        seg = ns.seg(),
        fwd = ns.fwd(),
        dict = ns.dict(),
        term = ns.term(),
        post = ns.post(),
        delta = ns.delta(),
        keyty = key_sql_type,
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Drop every persistent table and index (the version-mismatch / desync response).
pub(crate) fn drop_persistent(conn: &Connection, ns: &Namespace) -> Result<()> {
    let sql = format!(
        "DROP INDEX IF EXISTS {seg}_by_key;
         DROP INDEX IF EXISTS {seg}_by_key_label;
         DROP INDEX IF EXISTS {dict}_by_gram;
         DROP TABLE IF EXISTS {seg};
         DROP TABLE IF EXISTS {fwd};
         DROP TABLE IF EXISTS {dict};
         DROP TABLE IF EXISTS {term};
         DROP TABLE IF EXISTS {post};
         DROP TABLE IF EXISTS {delta};",
        seg = ns.seg(),
        fwd = ns.fwd(),
        dict = ns.dict(),
        term = ns.term(),
        post = ns.post(),
        delta = ns.delta(),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Drop any leftover rebuild-shadow tables (a prior aborted rebuild). Idempotent.
pub(crate) fn drop_shadows(conn: &Connection, ns: &Namespace) -> Result<()> {
    let sql = format!(
        "DROP TABLE IF EXISTS {seg};
         DROP TABLE IF EXISTS {fwd};
         DROP TABLE IF EXISTS {dict};
         DROP TABLE IF EXISTS {term};
         DROP TABLE IF EXISTS {post};
         DROP TABLE IF EXISTS {delta};",
        seg = ns.seg_shadow(),
        fwd = ns.fwd_shadow(),
        dict = ns.dict_shadow(),
        term = ns.term_shadow(),
        post = ns.post_shadow(),
        delta = ns.delta_shadow(),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Create fresh, empty rebuild-shadow tables (dropping any leftovers first). No
/// `seg`/`dict` indexes — all are recreated on the live tables after the swap (the
/// shadow bulk-inserts are dup-free, so the unique indexes are not needed during build).
pub(crate) fn create_shadows(conn: &Connection, ns: &Namespace, schema: &Schema) -> Result<()> {
    drop_shadows(conn, ns)?;
    let sql = format!(
        "CREATE TABLE {seg}(
            id INTEGER PRIMARY KEY, key {keyty} NOT NULL, label TEXT NOT NULL,
            txt TEXT NOT NULL, len INTEGER NOT NULL DEFAULT 0
         );
         CREATE TABLE {fwd}(id INTEGER PRIMARY KEY, tokens BLOB NOT NULL);
         CREATE TABLE {dict}(id INTEGER PRIMARY KEY, gram BLOB NOT NULL);
         CREATE TABLE {term}(id INTEGER PRIMARY KEY, df INTEGER NOT NULL);
         CREATE TABLE {post}(id INTEGER PRIMARY KEY, base BLOB NOT NULL);
         CREATE TABLE {delta}(id INTEGER PRIMARY KEY, added BLOB NOT NULL, removed BLOB NOT NULL);",
        seg = ns.seg_shadow(),
        fwd = ns.fwd_shadow(),
        dict = ns.dict_shadow(),
        term = ns.term_shadow(),
        post = ns.post_shadow(),
        delta = ns.delta_shadow(),
        keyty = schema.key_shape().sql_type(),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Swap the freshly-built shadow tables in for the live ones in one transaction:
/// drop live, rename each shadow to its live name, recreate the `seg`/`dict` indexes.
/// A reader sees complete-old or complete-new — never partial.
pub(crate) fn swap_shadows(conn: &Connection, ns: &Namespace) -> Result<()> {
    let sql = format!(
        "DROP INDEX IF EXISTS {seg}_by_key;
         DROP INDEX IF EXISTS {seg}_by_key_label;
         DROP INDEX IF EXISTS {dict}_by_gram;
         DROP TABLE IF EXISTS {seg};   ALTER TABLE {seg_s}   RENAME TO {seg};
         DROP TABLE IF EXISTS {fwd};   ALTER TABLE {fwd_s}   RENAME TO {fwd};
         DROP TABLE IF EXISTS {dict};  ALTER TABLE {dict_s}  RENAME TO {dict};
         DROP TABLE IF EXISTS {term};  ALTER TABLE {term_s}  RENAME TO {term};
         DROP TABLE IF EXISTS {post};  ALTER TABLE {post_s}  RENAME TO {post};
         DROP TABLE IF EXISTS {delta}; ALTER TABLE {delta_s} RENAME TO {delta};
         CREATE INDEX {seg}_by_key ON {seg}(key);
         CREATE UNIQUE INDEX {seg}_by_key_label ON {seg}(key, label);
         CREATE UNIQUE INDEX {dict}_by_gram ON {dict}(gram);",
        seg = ns.seg(),
        fwd = ns.fwd(),
        dict = ns.dict(),
        term = ns.term(),
        post = ns.post(),
        delta = ns.delta(),
        seg_s = ns.seg_shadow(),
        fwd_s = ns.fwd_shadow(),
        dict_s = ns.dict_shadow(),
        term_s = ns.term_shadow(),
        post_s = ns.post_shadow(),
        delta_s = ns.delta_shadow(),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Reset the store to empty: drop persistent tables and any shadows, then recreate
/// the persistent tables. Used when a version stamp mismatches or the id-allocation
/// invariant is found broken at open.
pub(crate) fn reset(conn: &Connection, ns: &Namespace, schema: &Schema) -> Result<()> {
    drop_shadows(conn, ns)?;
    drop_persistent(conn, ns)?;
    create_tables(conn, ns, schema)?;
    // `drop_persistent` keeps `meta`, so explicitly zero the rolling segment stats — the
    // data tables are now empty.
    set_seg_stats(conn, ns, 0, 0)
}

/// The `(seg_count, seg_len_sum)` rolling stats (absent → `0`). `seg_count` is the corpus
/// size `N`; `seg_len_sum` sums the per-segment **distinct** gram counts, so `avgdl =
/// seg_len_sum / seg_count` is the mean distinct-gram count `L̄` (derivation §0/§6).
pub(crate) fn read_seg_stats(conn: &Connection, ns: &Namespace) -> Result<(i64, i64)> {
    let count = meta_get(conn, ns, KEY_SEG_COUNT)?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let len_sum = meta_get(conn, ns, KEY_SEG_LEN_SUM)?
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    Ok((count, len_sum))
}

/// Set the rolling segment stats to absolute values (rebuild swap, reset).
pub(crate) fn set_seg_stats(
    conn: &Connection,
    ns: &Namespace,
    count: i64,
    len_sum: i64,
) -> Result<()> {
    meta_set(conn, ns, KEY_SEG_COUNT, &count.to_string())?;
    meta_set(conn, ns, KEY_SEG_LEN_SUM, &len_sum.to_string())
}

/// Apply `(count_delta, len_delta)` to the rolling segment stats (an incremental write).
/// Runs inside the caller's transaction, so a `SAVEPOINT` rollback reverts it.
pub(crate) fn bump_seg_stats(
    conn: &Connection,
    ns: &Namespace,
    count_delta: i64,
    len_delta: i64,
) -> Result<()> {
    let (count, len_sum) = read_seg_stats(conn, ns)?;
    set_seg_stats(conn, ns, count + count_delta, len_sum + len_delta)
}

/// Read a meta value as a string.
pub(crate) fn meta_get(conn: &Connection, ns: &Namespace, key: &str) -> Result<Option<String>> {
    let sql = format!("SELECT value FROM {} WHERE key = ?1", ns.meta());
    Ok(conn
        .query_row(&sql, [key], |r| r.get::<_, String>(0))
        .optional()?)
}

/// Write a meta value.
pub(crate) fn meta_set(conn: &Connection, ns: &Namespace, key: &str, value: &str) -> Result<()> {
    let sql = format!(
        "INSERT INTO {0}(key, value) VALUES(?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ns.meta()
    );
    conn.execute(&sql, rusqlite::params![key, value])?;
    Ok(())
}

/// The drift stamps, read together.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Stamps {
    pub schema_version: Option<u32>,
    pub data_version: Option<u64>,
    pub fingerprint: Option<u64>,
    pub schema_fingerprint: Option<u64>,
}

/// Read the current drift stamps (any absent → `None`, treated as drift on open).
pub(crate) fn read_stamps(conn: &Connection, ns: &Namespace) -> Result<Stamps> {
    Ok(Stamps {
        schema_version: meta_get(conn, ns, KEY_SCHEMA_VERSION)?.and_then(|s| s.parse().ok()),
        data_version: meta_get(conn, ns, KEY_DATA_VERSION)?.and_then(|s| s.parse().ok()),
        fingerprint: meta_get(conn, ns, KEY_FINGERPRINT)?.and_then(|s| s.parse().ok()),
        schema_fingerprint: meta_get(conn, ns, KEY_SCHEMA_FINGERPRINT)?
            .and_then(|s| s.parse().ok()),
    })
}

/// Stamp the current schema/data/tokenizer/schema-fingerprint versions. Called after a
/// reset and after a successful rebuild — the stamps describe what the index now
/// reflects.
pub(crate) fn write_stamps(
    conn: &Connection,
    ns: &Namespace,
    data_version: u64,
    fingerprint: u64,
    schema_fingerprint: u64,
) -> Result<()> {
    meta_set(conn, ns, KEY_SCHEMA_VERSION, &SCHEMA_VERSION.to_string())?;
    meta_set(conn, ns, KEY_DATA_VERSION, &data_version.to_string())?;
    meta_set(conn, ns, KEY_FINGERPRINT, &fingerprint.to_string())?;
    meta_set(
        conn,
        ns,
        KEY_SCHEMA_FINGERPRINT,
        &schema_fingerprint.to_string(),
    )?;
    Ok(())
}

/// Parse a REQUIRED-integrity meta counter: **absent** is fine (a fresh store — the caller
/// supplies the default), but a **present-yet-unparseable** value is external corruption and
/// fails closed as [`Error::Corrupt`](crate::Error::Corrupt) (v0.5). Silently defaulting was the
/// pre-v0.5 behavior and contradicted the fail-closed philosophy: a defaulted `next_id` restarts
/// id allocation (id **reuse** — postings would alias new segments), and a defaulted
/// `dict_generation` could falsely match a reader's captured generation.
fn parse_counter<T: std::str::FromStr>(key: &str, raw: Option<String>) -> Result<Option<T>> {
    match raw {
        None => Ok(None),
        Some(s) => match s.parse() {
            Ok(v) => Ok(Some(v)),
            Err(_) => Err(crate::Error::corrupt(format!(
                "meta counter {key:?} holds an unparseable value {s:?}; the store is corrupt \
                 (rebuild it)"
            ))),
        },
    }
}

/// The id high-water mark (`meta.next_id`). Absent → `1` (a fresh store); malformed-present →
/// [`Error::Corrupt`](crate::Error::Corrupt).
pub(crate) fn next_id(conn: &Connection, ns: &Namespace) -> Result<i64> {
    Ok(parse_counter(KEY_NEXT_ID, meta_get(conn, ns, KEY_NEXT_ID)?)?.unwrap_or(1))
}

/// Allocate `count` fresh, never-reused segment ids, returning the first. Ids are
/// monotonic (the high-water mark lives in `meta.next_id`): a freed id is never
/// reassigned, so a stale id lingering in a posting until the next fold is harmless
/// (it can never come to mean a different segment), and no REMOVE-before-ADD
/// ordering discipline is needed.
pub(crate) fn alloc_ids(conn: &Connection, ns: &Namespace, count: u64) -> Result<i64> {
    let next: i64 = next_id(conn, ns)?;
    let after = next
        .checked_add(count as i64)
        // Ids are stored in u32 roaring postings, so the real ceiling is u32::MAX, not
        // i64::MAX: the last id minted here is `after - 1` and must round-trip through u32.
        .filter(|&a| a <= u32::MAX as i64 + 1)
        .ok_or_else(|| crate::Error::corrupt("segment id space exhausted"))?;
    meta_set(conn, ns, KEY_NEXT_ID, &after.to_string())?;
    Ok(next)
}

/// Reset the id high-water mark (used by rebuild, which reassigns dense ids). Rejects a
/// mark past the u32 ceiling for the same reason as [`alloc_ids`].
pub(crate) fn set_next_id(conn: &Connection, ns: &Namespace, next_id: i64) -> Result<()> {
    if next_id > u32::MAX as i64 + 1 {
        return Err(crate::Error::corrupt("segment id space exhausted"));
    }
    meta_set(conn, ns, KEY_NEXT_ID, &next_id.to_string())
}

/// The current dictionary generation (term-id-assignment epoch). Absent → 0 (a fresh store);
/// malformed-present → [`Error::Corrupt`](crate::Error::Corrupt) (a silently-defaulted
/// generation could falsely match a reader's captured one).
pub(crate) fn dict_generation(conn: &Connection, ns: &Namespace) -> Result<u64> {
    Ok(
        parse_counter(KEY_DICT_GENERATION, meta_get(conn, ns, KEY_DICT_GENERATION)?)?
            .unwrap_or(0),
    )
}

/// Bump the dictionary generation. Called whenever term-ids are reassigned — a
/// [`rebuild`](crate::Index::rebuild)'s shadow swap or a drift/desync `reset` — so a
/// reader can detect that the in-memory dictionary it resolved against no longer
/// matches its SQLite snapshot. A plain incremental write does *not* bump it.
pub(crate) fn bump_dict_generation(conn: &Connection, ns: &Namespace) -> Result<()> {
    let next = dict_generation(conn, ns)?.wrapping_add(1);
    meta_set(conn, ns, KEY_DICT_GENERATION, &next.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Schema;

    fn conn() -> Connection {
        let c = Connection::open_in_memory().unwrap();
        rusqlite::vtab::array::load_module(&c).unwrap();
        c
    }

    fn schema() -> Schema {
        Schema::flat()
    }

    fn count(conn: &Connection, table: &str) -> i64 {
        conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn create_tables_is_idempotent() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        create_tables(&c, &ns, &schema()).unwrap(); // second call must not error
        assert_eq!(count(&c, ns.seg()), 0);
    }

    #[test]
    fn meta_round_trips_and_missing_is_none() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        assert_eq!(meta_get(&c, &ns, "k").unwrap(), None);
        meta_set(&c, &ns, "k", "v1").unwrap();
        assert_eq!(meta_get(&c, &ns, "k").unwrap().as_deref(), Some("v1"));
        meta_set(&c, &ns, "k", "v2").unwrap(); // overwrite
        assert_eq!(meta_get(&c, &ns, "k").unwrap().as_deref(), Some("v2"));
    }

    #[test]
    fn stamps_round_trip_including_u64_max() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        write_stamps(&c, &ns, u64::MAX, 0xDEAD_BEEF_CAFE_F00D, 0x1234_5678).unwrap();
        let s = read_stamps(&c, &ns).unwrap();
        assert_eq!(s.schema_version, Some(SCHEMA_VERSION));
        assert_eq!(s.data_version, Some(u64::MAX));
        assert_eq!(s.fingerprint, Some(0xDEAD_BEEF_CAFE_F00D));
        assert_eq!(s.schema_fingerprint, Some(0x1234_5678));
    }

    #[test]
    fn read_stamps_on_fresh_store_is_all_none() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        assert_eq!(read_stamps(&c, &ns).unwrap(), Stamps::default());
    }

    #[test]
    fn alloc_ids_is_monotonic_and_never_reuses() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        assert_eq!(alloc_ids(&c, &ns, 3).unwrap(), 1); // [1,2,3]
        assert_eq!(alloc_ids(&c, &ns, 2).unwrap(), 4); // [4,5]
        assert_eq!(alloc_ids(&c, &ns, 1).unwrap(), 6); // [6]
        set_next_id(&c, &ns, 100).unwrap();
        assert_eq!(alloc_ids(&c, &ns, 1).unwrap(), 100);
    }

    #[test]
    fn alloc_ids_overflow_is_a_clean_error_not_a_wrap() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        // Ids land in u32 roaring postings, so the ceiling is u32::MAX, not i64::MAX.
        set_next_id(&c, &ns, u32::MAX as i64).unwrap();
        // The last legal id, leaving next_id one past u32::MAX.
        assert_eq!(alloc_ids(&c, &ns, 1).unwrap(), u32::MAX as i64);
        // A further allocation would mint an id that truncates in u32; it must error.
        assert!(matches!(
            alloc_ids(&c, &ns, 5),
            Err(crate::Error::Corrupt(_))
        ));
    }

    #[test]
    fn set_next_id_rejects_marks_past_u32() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        assert!(set_next_id(&c, &ns, u32::MAX as i64 + 1).is_ok());
        assert!(set_next_id(&c, &ns, u32::MAX as i64 + 2).is_err());
    }

    #[test]
    fn reset_empties_data_but_keeps_meta() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        c.execute(
            &format!(
                "INSERT INTO {}(id, key, label, txt) VALUES(1, 1, 'l', 't')",
                ns.seg()
            ),
            [],
        )
        .unwrap();
        meta_set(&c, &ns, "keep", "yes").unwrap();
        reset(&c, &ns, &schema()).unwrap();
        assert_eq!(count(&c, ns.seg()), 0);
        assert_eq!(meta_get(&c, &ns, "keep").unwrap().as_deref(), Some("yes"));
    }

    #[test]
    fn shadow_build_and_swap_replaces_live_atomically() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        // Live has one row; the shadow will carry a different one.
        c.execute(
            &format!(
                "INSERT INTO {}(id, key, label, txt) VALUES(1, 1, 'l', 'old')",
                ns.seg()
            ),
            [],
        )
        .unwrap();
        create_shadows(&c, &ns, &schema()).unwrap();
        c.execute(
            &format!(
                "INSERT INTO {}(id, key, label, txt) VALUES(9, 2, 'l', 'new')",
                ns.seg_shadow()
            ),
            [],
        )
        .unwrap();
        swap_shadows(&c, &ns).unwrap();
        // Live now reflects the shadow's content, and the index exists again.
        let (id, txt): (i64, String) = c
            .query_row(&format!("SELECT id, txt FROM {}", ns.seg()), [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!((id, txt.as_str()), (9, "new"));
        // The seg-by-key index was recreated.
        let idx: i64 = c
            .query_row(
                &format!(
                    "SELECT count(*) FROM sqlite_master WHERE type='index' AND name='{}_by_key'",
                    ns.seg()
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn malformed_present_counters_fail_closed_as_corrupt() {
        // v0.5 (§1.4 of the post-v0.4 review): an absent counter is a fresh store (defaulted),
        // but a present-yet-unparseable one is external corruption — silently defaulting
        // `next_id` would REUSE ids and a defaulted `dict_generation` could falsely match.
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        // Absent → defaults, no error.
        assert_eq!(next_id(&c, &ns).unwrap(), 1);
        assert_eq!(dict_generation(&c, &ns).unwrap(), 0);
        // Malformed-present → Corrupt, from both the reader and the allocator.
        meta_set(&c, &ns, KEY_NEXT_ID, "not-a-number").unwrap();
        assert!(matches!(next_id(&c, &ns), Err(crate::Error::Corrupt(_))));
        assert!(matches!(alloc_ids(&c, &ns, 1), Err(crate::Error::Corrupt(_))));
        meta_set(&c, &ns, KEY_DICT_GENERATION, "-3").unwrap(); // u64: negative is malformed
        assert!(matches!(
            dict_generation(&c, &ns),
            Err(crate::Error::Corrupt(_))
        ));
    }

    #[test]
    fn drop_shadows_is_a_safe_noop_when_absent() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns, &schema()).unwrap();
        drop_shadows(&c, &ns).unwrap(); // no shadows yet — must not error
    }
}
