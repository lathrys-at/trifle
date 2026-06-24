//! On-disk schema: the namespaced DDL, the drift/version stamps, and monotonic id
//! allocation.
//!
//! All names come from a validated [`Namespace`], so interpolating them into DDL
//! has no injection surface. The store is a rebuildable cache: a version mismatch
//! drops everything rather than migrating (§8.4).

use rusqlite::{Connection, OptionalExtension};

use crate::error::Result;
use crate::store::Namespace;

/// trifle's on-disk format version. Bump on any incompatible schema change; an
/// index stamped with a different value is reset on open.
pub(crate) const SCHEMA_VERSION: u32 = 1;

pub(crate) const KEY_SCHEMA_VERSION: &str = "schema_version";
pub(crate) const KEY_DATA_VERSION: &str = "data_version";
pub(crate) const KEY_FINGERPRINT: &str = "tokenizer_fingerprint";
pub(crate) const KEY_NEXT_ID: &str = "next_id";

/// Create every persistent table and index if absent. Idempotent.
pub(crate) fn create_tables(conn: &Connection, ns: &Namespace) -> Result<()> {
    // `seg.txt` is NULL in contentless mode; `fwd` is populated only in contentless
    // mode (the per-segment token set used by delete). The three-way write-frequency
    // split — `term`/`delta` on every write, `post` only on fold/rebuild — is
    // deliberate: a write never rewrites the big base posting.
    let sql = format!(
        "CREATE TABLE IF NOT EXISTS {meta}(key TEXT PRIMARY KEY, value);
         CREATE TABLE IF NOT EXISTS {seg}(
            id     INTEGER PRIMARY KEY,
            doc_id INTEGER NOT NULL,
            source TEXT    NOT NULL,
            ref    TEXT    NOT NULL,
            txt    TEXT
         );
         CREATE INDEX IF NOT EXISTS {seg}_by_doc ON {seg}(doc_id, source);
         CREATE TABLE IF NOT EXISTS {fwd}(id INTEGER PRIMARY KEY, tokens BLOB NOT NULL);
         CREATE TABLE IF NOT EXISTS {term}(term TEXT PRIMARY KEY, df INTEGER NOT NULL);
         CREATE TABLE IF NOT EXISTS {post}(term TEXT PRIMARY KEY, base BLOB NOT NULL);
         CREATE TABLE IF NOT EXISTS {delta}(
            term TEXT PRIMARY KEY, added BLOB NOT NULL, removed BLOB NOT NULL
         );",
        meta = ns.meta(),
        seg = ns.seg(),
        fwd = ns.fwd(),
        term = ns.term(),
        post = ns.post(),
        delta = ns.delta(),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Drop every persistent table and index (the version-mismatch / desync response).
pub(crate) fn drop_persistent(conn: &Connection, ns: &Namespace) -> Result<()> {
    let sql = format!(
        "DROP INDEX IF EXISTS {seg}_by_doc;
         DROP TABLE IF EXISTS {seg};
         DROP TABLE IF EXISTS {fwd};
         DROP TABLE IF EXISTS {term};
         DROP TABLE IF EXISTS {post};
         DROP TABLE IF EXISTS {delta};",
        seg = ns.seg(),
        fwd = ns.fwd(),
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
         DROP TABLE IF EXISTS {term};
         DROP TABLE IF EXISTS {post};
         DROP TABLE IF EXISTS {delta};",
        seg = ns.seg_shadow(),
        fwd = ns.fwd_shadow(),
        term = ns.term_shadow(),
        post = ns.post_shadow(),
        delta = ns.delta_shadow(),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Create fresh, empty rebuild-shadow tables (dropping any leftovers first). No
/// `seg`-by-doc index — it is recreated on the live table after the swap.
pub(crate) fn create_shadows(conn: &Connection, ns: &Namespace) -> Result<()> {
    drop_shadows(conn, ns)?;
    let sql = format!(
        "CREATE TABLE {seg}(
            id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL,
            source TEXT NOT NULL, ref TEXT NOT NULL, txt TEXT
         );
         CREATE TABLE {fwd}(id INTEGER PRIMARY KEY, tokens BLOB NOT NULL);
         CREATE TABLE {term}(term TEXT PRIMARY KEY, df INTEGER NOT NULL);
         CREATE TABLE {post}(term TEXT PRIMARY KEY, base BLOB NOT NULL);
         CREATE TABLE {delta}(term TEXT PRIMARY KEY, added BLOB NOT NULL, removed BLOB NOT NULL);",
        seg = ns.seg_shadow(),
        fwd = ns.fwd_shadow(),
        term = ns.term_shadow(),
        post = ns.post_shadow(),
        delta = ns.delta_shadow(),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Swap the freshly-built shadow tables in for the live ones in one transaction:
/// drop live, rename each shadow to its live name, recreate the `seg`-by-doc index.
/// A reader sees complete-old or complete-new — never partial.
pub(crate) fn swap_shadows(conn: &Connection, ns: &Namespace) -> Result<()> {
    let sql = format!(
        "DROP INDEX IF EXISTS {seg}_by_doc;
         DROP TABLE IF EXISTS {seg};   ALTER TABLE {seg_s}   RENAME TO {seg};
         DROP TABLE IF EXISTS {fwd};   ALTER TABLE {fwd_s}   RENAME TO {fwd};
         DROP TABLE IF EXISTS {term};  ALTER TABLE {term_s}  RENAME TO {term};
         DROP TABLE IF EXISTS {post};  ALTER TABLE {post_s}  RENAME TO {post};
         DROP TABLE IF EXISTS {delta}; ALTER TABLE {delta_s} RENAME TO {delta};
         CREATE INDEX {seg}_by_doc ON {seg}(doc_id, source);",
        seg = ns.seg(),
        fwd = ns.fwd(),
        term = ns.term(),
        post = ns.post(),
        delta = ns.delta(),
        seg_s = ns.seg_shadow(),
        fwd_s = ns.fwd_shadow(),
        term_s = ns.term_shadow(),
        post_s = ns.post_shadow(),
        delta_s = ns.delta_shadow(),
    );
    conn.execute_batch(&sql)?;
    Ok(())
}

/// Reset the store to empty: drop persistent tables and any shadows, then recreate
/// the persistent tables. Used when a version stamp mismatches or a `seg`↔posting
/// desync is detected at open.
pub(crate) fn reset(conn: &Connection, ns: &Namespace) -> Result<()> {
    drop_shadows(conn, ns)?;
    drop_persistent(conn, ns)?;
    create_tables(conn, ns)
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

/// The three drift stamps, read together.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Stamps {
    pub schema_version: Option<u32>,
    pub data_version: Option<u64>,
    pub fingerprint: Option<u64>,
}

/// Read the current drift stamps (any absent → `None`, treated as drift on open).
pub(crate) fn read_stamps(conn: &Connection, ns: &Namespace) -> Result<Stamps> {
    Ok(Stamps {
        schema_version: meta_get(conn, ns, KEY_SCHEMA_VERSION)?.and_then(|s| s.parse().ok()),
        data_version: meta_get(conn, ns, KEY_DATA_VERSION)?.and_then(|s| s.parse().ok()),
        fingerprint: meta_get(conn, ns, KEY_FINGERPRINT)?.and_then(|s| s.parse().ok()),
    })
}

/// Stamp the current schema/data/tokenizer versions. Called after a reset and after
/// a successful rebuild — the stamps describe what the index now reflects.
pub(crate) fn write_stamps(
    conn: &Connection,
    ns: &Namespace,
    data_version: u64,
    fingerprint: u64,
) -> Result<()> {
    meta_set(conn, ns, KEY_SCHEMA_VERSION, &SCHEMA_VERSION.to_string())?;
    meta_set(conn, ns, KEY_DATA_VERSION, &data_version.to_string())?;
    meta_set(conn, ns, KEY_FINGERPRINT, &fingerprint.to_string())?;
    Ok(())
}

/// Allocate `count` fresh, never-reused segment ids, returning the first. Ids are
/// monotonic (the high-water mark lives in `meta.next_id`): a freed id is never
/// reassigned, so a stale id lingering in a posting until the next fold is harmless
/// (it can never come to mean a different segment), and no REMOVE-before-ADD
/// ordering discipline is needed.
pub(crate) fn alloc_ids(conn: &Connection, ns: &Namespace, count: u64) -> Result<i64> {
    let next: i64 = meta_get(conn, ns, KEY_NEXT_ID)?
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let after = next
        .checked_add(count as i64)
        .ok_or_else(|| crate::Error::corrupt("segment id space exhausted"))?;
    meta_set(conn, ns, KEY_NEXT_ID, &after.to_string())?;
    Ok(next)
}

/// Reset the id high-water mark (used by rebuild, which reassigns dense ids).
pub(crate) fn set_next_id(conn: &Connection, ns: &Namespace, next_id: i64) -> Result<()> {
    meta_set(conn, ns, KEY_NEXT_ID, &next_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conn() -> Connection {
        let c = Connection::open_in_memory().unwrap();
        rusqlite::vtab::array::load_module(&c).unwrap();
        c
    }

    fn count(conn: &Connection, table: &str) -> i64 {
        conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }

    #[test]
    fn create_tables_is_idempotent() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns).unwrap();
        create_tables(&c, &ns).unwrap(); // second call must not error
        assert_eq!(count(&c, ns.seg()), 0);
    }

    #[test]
    fn meta_round_trips_and_missing_is_none() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns).unwrap();
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
        create_tables(&c, &ns).unwrap();
        write_stamps(&c, &ns, u64::MAX, 0xDEAD_BEEF_CAFE_F00D).unwrap();
        let s = read_stamps(&c, &ns).unwrap();
        assert_eq!(s.schema_version, Some(SCHEMA_VERSION));
        assert_eq!(s.data_version, Some(u64::MAX));
        assert_eq!(s.fingerprint, Some(0xDEAD_BEEF_CAFE_F00D));
    }

    #[test]
    fn read_stamps_on_fresh_store_is_all_none() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns).unwrap();
        assert_eq!(read_stamps(&c, &ns).unwrap(), Stamps::default());
    }

    #[test]
    fn alloc_ids_is_monotonic_and_never_reuses() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns).unwrap();
        assert_eq!(alloc_ids(&c, &ns, 3).unwrap(), 1); // [1,2,3]
        assert_eq!(alloc_ids(&c, &ns, 2).unwrap(), 4); // [4,5]
        assert_eq!(alloc_ids(&c, &ns, 1).unwrap(), 6); // [6]
        set_next_id(&c, &ns, 100).unwrap();
        assert_eq!(alloc_ids(&c, &ns, 1).unwrap(), 100);
    }

    #[test]
    fn reset_empties_data_but_keeps_meta() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns).unwrap();
        c.execute(
            &format!(
                "INSERT INTO {}(id, doc_id, source, ref, txt) VALUES(1, 1, 's', 'r', 't')",
                ns.seg()
            ),
            [],
        )
        .unwrap();
        meta_set(&c, &ns, "keep", "yes").unwrap();
        reset(&c, &ns).unwrap();
        assert_eq!(count(&c, ns.seg()), 0);
        assert_eq!(meta_get(&c, &ns, "keep").unwrap().as_deref(), Some("yes"));
    }

    #[test]
    fn shadow_build_and_swap_replaces_live_atomically() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns).unwrap();
        // Live has one row; the shadow will carry a different one.
        c.execute(
            &format!(
                "INSERT INTO {}(id, doc_id, source, ref, txt) VALUES(1, 1, 's', 'r', 'old')",
                ns.seg()
            ),
            [],
        )
        .unwrap();
        create_shadows(&c, &ns).unwrap();
        c.execute(
            &format!(
                "INSERT INTO {}(id, doc_id, source, ref, txt) VALUES(9, 2, 's', 'r', 'new')",
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
        // The seg-by-doc index was recreated.
        let idx: i64 = c
            .query_row(
                &format!(
                    "SELECT count(*) FROM sqlite_master WHERE type='index' AND name='{}_by_doc'",
                    ns.seg()
                ),
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn drop_shadows_is_a_safe_noop_when_absent() {
        let c = conn();
        let ns = Namespace::bare();
        create_tables(&c, &ns).unwrap();
        drop_shadows(&c, &ns).unwrap(); // no shadows yet — must not error
    }
}
