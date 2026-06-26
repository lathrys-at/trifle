//! Storage backends — where connections come from and how trifle's tables are
//! named.
//!
//! trifle's SQLite surface is small and self-contained, so *where connections come
//! from* (an owned file, or a database the caller owns) is abstracted behind
//! [`Backend`]. With no virtual tables, that surface is plain tables and BLOBs,
//! which makes the backend genuinely portable.
//!
//! - [`Sidecar`] (default, recommended) — trifle opens and owns its own file,
//!   sets WAL / `mmap` / pragmas, and runs a single-writer connection plus a
//!   read-only pool. The caller passes a path.
//! - [`Shared`] (opt-in) — trifle's tables live, [namespaced](Namespace), inside a
//!   database the caller owns; the caller supplies connection access. Only for a
//!   hard co-location requirement.

use std::ops::{Deref, DerefMut};
use std::sync::Once;

use rusqlite::Connection;

use crate::error::{Error, Result};

mod pool;
mod shared;
mod sidecar;

pub use pool::ReadConn;
pub use shared::Shared;
pub use sidecar::Sidecar;

/// Abstracts where trifle's SQLite connections come from and how its tables are
/// named.
///
/// trifle is single-writer: [`write`](Backend::write) hands out the one exclusive
/// writer, [`read`](Backend::read) a pooled read-only connection that (under WAL)
/// runs concurrently with the writer and with other readers.
pub trait Backend: Send + Sync {
    /// The exclusive write connection guard; derefs to the [`Connection`].
    type WriteGuard<'a>: DerefMut<Target = Connection>
    where
        Self: 'a;
    /// A pooled read-only connection guard; derefs to the [`Connection`].
    type ReadGuard<'a>: Deref<Target = Connection>
    where
        Self: 'a;

    /// Acquire the exclusive write connection. Blocks until the writer is free.
    fn write(&self) -> Result<Self::WriteGuard<'_>>;

    /// Acquire a pooled read-only connection.
    fn read(&self) -> Result<Self::ReadGuard<'_>>;

    /// The table-naming namespace for this backend.
    fn namespace(&self) -> &Namespace;

    /// Per-connection setup that registers the `carray`/`rarray` vtab (for batched
    /// `WHERE id IN rarray(?1)` hydration) and, in [`Sidecar`] mode, sets pragmas.
    ///
    /// **The backend is responsible for running this** (or equivalent setup) on every
    /// connection it hands out from [`write`](Backend::write) / [`read`](Backend::read)
    /// before first use — the built-in backends do so in their connection factories.
    /// trifle never calls it for you; it relies on the backend having upheld the
    /// invariant. Exposed on the trait so a wrapping/custom backend can reuse the
    /// setup. Must be idempotent.
    fn init_conn(&self, conn: &Connection) -> Result<()>;
}

/// One-time, process-global SQLite tuning. Best-effort and behavior-transparent.
///
/// Disables memory-statistics bookkeeping: with it on (the bundled default) every
/// `sqlite3_malloc`/`free` takes the global `SQLITE_MUTEX_STATIC_MEM` mutex to
/// update counters trifle never reads, which serializes pooled concurrent readers
/// on a lock. Must run before the first connection opens, so every connection-open
/// path calls it behind a [`Once`].
pub(crate) fn configure_sqlite_perf() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: `sqlite3_config` is the variadic C configuration API;
        // `SQLITE_CONFIG_MEMSTATUS` consumes exactly one C int (the on/off flag),
        // which is what is passed. Calling it before SQLite is initialized is the
        // documented contract; a late call returns `SQLITE_MISUSE` and is ignored.
        unsafe {
            let _ = rusqlite::ffi::sqlite3_config(rusqlite::ffi::SQLITE_CONFIG_MEMSTATUS, 0);
        }
    });
}

/// Register the `carray`/`rarray` virtual table on a connection so a whole id list
/// binds to one prepared `WHERE id IN rarray(?1)` statement. Idempotent.
pub(crate) fn register_carray(conn: &Connection) -> Result<()> {
    rusqlite::vtab::array::load_module(conn)?;
    Ok(())
}

/// The resolved name of every table trifle creates.
///
/// Used to name tables in [`Namespace::explicit`]. The names given here are the
/// *persistent* tables; trifle derives its transient rebuild-shadow table names
/// from them (suffix `_shadow`) and validates the whole set for collisions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableMap {
    /// Key/value metadata: schema/data/tokenizer version stamps, rolling counters.
    pub meta: String,
    /// One row per document: the internal dense id and the caller's key.
    pub doc: String,
    /// One row per indexed segment (id, doc_id, label, snapshot text).
    pub seg: String,
    /// Contentless-mode forward index (per-segment token set); unused in snapshot
    /// mode.
    pub fwd: String,
    /// Per-token effective document frequency (the pruner reads this), keyed by the
    /// interned `u32` term-id.
    pub term: String,
    /// Per-token base roaring posting (written only by fold/rebuild), keyed by term-id.
    pub post: String,
    /// Per-token added/removed roaring delta (written on every incremental write),
    /// keyed by term-id.
    pub delta: String,
    /// The interning dictionary: `term-id <-> gram` (the gram is the only place gram
    /// text/encoding is stored; postings reference the `u32` id).
    pub dict: String,
}

impl TableMap {
    /// Every persistent table name, in a stable order.
    fn names(&self) -> [&str; 8] {
        [
            &self.meta,
            &self.doc,
            &self.seg,
            &self.fwd,
            &self.term,
            &self.post,
            &self.delta,
            &self.dict,
        ]
    }
}

/// A validated, enumerable table-naming scheme.
///
/// Construct with [`prefixed`](Namespace::prefixed) (the common case — every table
/// gets a shared prefix) or [`explicit`](Namespace::explicit) (name each table).
/// All names are validated as safe SQL identifiers, distinct, and not reserved
/// (`sqlite_*`), so they can be interpolated into trifle's DDL without an injection
/// surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Namespace {
    map: TableMap,
    // Derived transient rebuild-shadow names, parallel to the persistent tables.
    doc_shadow: String,
    seg_shadow: String,
    fwd_shadow: String,
    term_shadow: String,
    post_shadow: String,
    delta_shadow: String,
    dict_shadow: String,
}

impl Default for Namespace {
    /// The default namespace prefixes every table with `trifle_`.
    fn default() -> Self {
        // The prefix is a compile-time-known valid identifier, so this never fails.
        Namespace::prefixed("trifle_").expect("`trifle_` is a valid namespace prefix")
    }
}

impl Namespace {
    /// Name every table `‹prefix›‹table›` — e.g. `prefixed("trifle_")` yields
    /// `trifle_seg`, `trifle_term`, … An empty prefix yields bare names
    /// (`seg`, `term`, …), appropriate when trifle owns the whole file ([`Sidecar`]).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Namespace`] if the prefix is not identifier-safe or would
    /// produce a reserved (`sqlite_*`) name.
    pub fn prefixed(prefix: &str) -> Result<Self> {
        if !prefix.is_empty() {
            validate_prefix(prefix)?;
        }
        let t = |name: &str| format!("{prefix}{name}");
        Namespace::explicit(TableMap {
            meta: t("meta"),
            doc: t("doc"),
            seg: t("seg"),
            fwd: t("fwd"),
            term: t("term"),
            post: t("post"),
            delta: t("delta"),
            dict: t("dict"),
        })
    }

    /// Bare, unprefixed table names. Equivalent to `prefixed("")`; used by
    /// [`Sidecar`], which owns its file and has no neighbors to collide with.
    pub fn bare() -> Self {
        Namespace::prefixed("").expect("bare names are valid identifiers")
    }

    /// Name each table explicitly.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Namespace`] if any name is not identifier-safe, is reserved
    /// (`sqlite_*`), or collides with another (including a derived `_shadow` name).
    pub fn explicit(map: TableMap) -> Result<Self> {
        for name in map.names() {
            validate_ident(name)?;
        }
        let ns = Namespace {
            doc_shadow: format!("{}_shadow", map.doc),
            seg_shadow: format!("{}_shadow", map.seg),
            fwd_shadow: format!("{}_shadow", map.fwd),
            term_shadow: format!("{}_shadow", map.term),
            post_shadow: format!("{}_shadow", map.post),
            delta_shadow: format!("{}_shadow", map.delta),
            dict_shadow: format!("{}_shadow", map.dict),
            map,
        };
        // Validate the derived shadow names too — a base name within the length bound
        // can still push its `_shadow` suffix past it, which would only fail at DDL
        // time, far from this call. Validate here so the caller hears it immediately.
        for shadow in [
            &ns.doc_shadow,
            &ns.seg_shadow,
            &ns.fwd_shadow,
            &ns.term_shadow,
            &ns.post_shadow,
            &ns.delta_shadow,
            &ns.dict_shadow,
        ] {
            validate_ident(shadow)?;
        }
        // All created tables — persistent + shadows — must be distinct.
        let all: Vec<&str> = ns.table_names().collect();
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                if a == b {
                    return Err(Error::namespace(format!("duplicate table name {a:?}")));
                }
            }
        }
        Ok(ns)
    }

    /// Every table name trifle will create under this namespace — the persistent
    /// tables and the two-per-table rebuild shadows. Useful for a caller's
    /// collision check or migration tooling.
    pub fn table_names(&self) -> impl Iterator<Item = &str> {
        [
            self.map.meta.as_str(),
            self.map.doc.as_str(),
            self.map.seg.as_str(),
            self.map.fwd.as_str(),
            self.map.term.as_str(),
            self.map.post.as_str(),
            self.map.delta.as_str(),
            self.map.dict.as_str(),
            self.doc_shadow.as_str(),
            self.seg_shadow.as_str(),
            self.fwd_shadow.as_str(),
            self.term_shadow.as_str(),
            self.post_shadow.as_str(),
            self.delta_shadow.as_str(),
            self.dict_shadow.as_str(),
        ]
        .into_iter()
    }

    // Accessors used by the schema/postings SQL builders.
    pub(crate) fn meta(&self) -> &str {
        &self.map.meta
    }
    pub(crate) fn doc(&self) -> &str {
        &self.map.doc
    }
    pub(crate) fn seg(&self) -> &str {
        &self.map.seg
    }
    pub(crate) fn fwd(&self) -> &str {
        &self.map.fwd
    }
    pub(crate) fn term(&self) -> &str {
        &self.map.term
    }
    pub(crate) fn post(&self) -> &str {
        &self.map.post
    }
    pub(crate) fn delta(&self) -> &str {
        &self.map.delta
    }
    pub(crate) fn dict(&self) -> &str {
        &self.map.dict
    }
    pub(crate) fn doc_shadow(&self) -> &str {
        &self.doc_shadow
    }
    pub(crate) fn seg_shadow(&self) -> &str {
        &self.seg_shadow
    }
    pub(crate) fn fwd_shadow(&self) -> &str {
        &self.fwd_shadow
    }
    pub(crate) fn term_shadow(&self) -> &str {
        &self.term_shadow
    }
    pub(crate) fn post_shadow(&self) -> &str {
        &self.post_shadow
    }
    pub(crate) fn delta_shadow(&self) -> &str {
        &self.delta_shadow
    }
    pub(crate) fn dict_shadow(&self) -> &str {
        &self.dict_shadow
    }
}

/// Validate a SQL identifier: ASCII, starts with a letter or `_`, then letters /
/// digits / `_`, length-bounded, and not a reserved `sqlite_` name. Used for table
/// names and for schema-derived column names (key / field labels).
pub(crate) fn validate_ident(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::namespace("empty table name"));
    }
    if name.len() > 64 {
        return Err(Error::namespace(format!("table name too long: {name:?}")));
    }
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty checked above");
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(Error::namespace(format!(
            "table name must start with a letter or underscore: {name:?}"
        )));
    }
    if !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(Error::namespace(format!(
            "table name has a non-identifier character: {name:?}"
        )));
    }
    if name.len() >= 7 && name[..7].eq_ignore_ascii_case("sqlite_") {
        return Err(Error::namespace(format!(
            "table name must not begin with the reserved `sqlite_`: {name:?}"
        )));
    }
    Ok(())
}

/// Validate a namespace prefix: it must, when concatenated with a bare table name,
/// remain a valid identifier — so it is ASCII letters / digits / `_`, starting with
/// a letter or `_`.
fn validate_prefix(prefix: &str) -> Result<()> {
    // A prefix is valid iff prefixing a representative bare name yields a valid,
    // non-reserved identifier — which `validate_ident` checks end to end.
    validate_ident(&format!("{prefix}seg"))
        .map_err(|_| Error::namespace(format!("invalid namespace prefix: {prefix:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixed_default_namespaces_every_table() {
        let ns = Namespace::default();
        assert_eq!(ns.seg(), "trifle_seg");
        assert_eq!(ns.term(), "trifle_term");
        assert_eq!(ns.seg_shadow(), "trifle_seg_shadow");
    }

    #[test]
    fn bare_namespace_has_unprefixed_names() {
        let ns = Namespace::bare();
        assert_eq!(ns.seg(), "seg");
        assert_eq!(ns.post(), "post");
    }

    #[test]
    fn table_names_enumerates_persistent_and_shadows() {
        let ns = Namespace::bare();
        let names: Vec<&str> = ns.table_names().collect();
        assert!(names.contains(&"seg"));
        assert!(names.contains(&"seg_shadow"));
        assert_eq!(names.len(), 15); // 8 persistent + 7 shadows
    }

    #[test]
    fn rejects_reserved_and_malformed_prefixes() {
        assert!(Namespace::prefixed("sqlite_").is_err());
        assert!(Namespace::prefixed("1bad").is_err());
        assert!(Namespace::prefixed("has space").is_err());
        assert!(Namespace::prefixed("ok_").is_ok());
    }

    #[test]
    fn explicit_rejects_collisions() {
        let map = TableMap {
            meta: "a".into(),
            doc: "g".into(),
            seg: "a".into(), // collides with meta
            fwd: "b".into(),
            term: "c".into(),
            post: "d".into(),
            delta: "e".into(),
            dict: "f".into(),
        };
        assert!(Namespace::explicit(map).is_err());
    }

    #[test]
    fn explicit_rejects_shadow_collision() {
        // `seg` derives `seg_shadow`; naming another table `seg_shadow` collides.
        let map = TableMap {
            meta: "meta".into(),
            doc: "doc".into(),
            seg: "seg".into(),
            fwd: "seg_shadow".into(),
            term: "term".into(),
            post: "post".into(),
            delta: "delta".into(),
            dict: "dict".into(),
        };
        assert!(Namespace::explicit(map).is_err());
    }
}
